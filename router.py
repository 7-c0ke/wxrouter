"""Forecast router: one address in front of box A and box B.

Policies (switch at runtime with POST /router/config):
  static        always the first backend (box A); the other only on failure
  round_robin   take turns
  least_loaded  fewest requests in flight (router's own count + box queue_depth)
  adaptive      stay on one box; switch only when the other box has been clearly
                better (by `margin`) for `window_s` seconds, then don't switch
                again for `cooldown_s` seconds. A dead active box switches at once.
                With `preferred` set, traffic starts on that box and goes back to
                it once it is healthy and within `return_margin` of the active
                box for `window_s` seconds (cooldown still applies).

Every policy fails over: if a box refuses the connection, times out or returns
5xx, the request is retried once on the other box. Forecasts are read-only, so
sending one twice is safe.

Run:
  python router.py --backend box-a=http://127.0.0.1:8001 \
                   --backend box-b=http://127.0.0.1:8002 --policy adaptive
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
from collections import deque
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

POLICIES = ("static", "round_robin", "least_loaded", "adaptive")
PROBE_BODY = {"lat": 0.0, "lon": 0.0, "hours_ahead": 12, "step_hours": 6, "request_id": "router-probe"}


class Backend:
    def __init__(self, name: str, url: str):
        self.name, self.url = name, url.rstrip("/")
        self.healthy = True
        self.fail_streak = 0
        self.ok_streak = 0
        self.in_flight = 0  # requests this router has outstanding on the box
        self.queue_depth = 0  # from the box's /stats
        self.max_concurrency = 1
        self.gpu_util = 0.0
        self.probe_ms: Optional[float] = None  # EWMA of the probe's GPU compute time
        self.probe_started: Optional[float] = None  # set while a probe is outstanding
        self.served = 0
        self.failures = 0

    def score(self, now: float, stall_ms: float) -> Optional[float]:
        """Adaptive's view of how fast this box is. A probe stuck longer than
        `stall_ms` (queued behind a backlog, or hanging) counts as that slow."""
        if self.probe_started is not None:
            waited = (now - self.probe_started) * 1000
            if waited > stall_ms:
                return max(self.probe_ms or 0.0, waited)
        return self.probe_ms

    def load(self) -> float:
        return (self.in_flight + self.queue_depth) / max(1, self.max_concurrency)

    def view(self) -> dict:
        return {
            "url": self.url, "healthy": self.healthy, "in_flight": self.in_flight,
            "queue_depth": self.queue_depth, "gpu_util": self.gpu_util,
            "probe_ms": None if self.probe_ms is None else round(self.probe_ms, 1),
            "served": self.served, "failures": self.failures,
        }


class AdaptiveConfig(BaseModel):
    margin: float = Field(0.25, ge=0, le=0.95, description="other box must be this much faster (0.25 = 25%)")
    window_s: float = Field(1.0, ge=0, description="...for this long without a break")
    cooldown_s: float = Field(10.0, ge=0, description="minimum time between switches")
    preferred: Optional[str] = Field(None, description="home box; traffic returns here when it recovers")
    return_margin: float = Field(0.20, ge=0, le=1.0,
                                 description="go home once preferred is no more than this much slower (0.20 = 20%)")


class ConfigUpdate(BaseModel):
    policy: Optional[str] = Field(None, pattern="^(static|round_robin|least_loaded|adaptive)$")
    margin: Optional[float] = None
    window_s: Optional[float] = None
    cooldown_s: Optional[float] = None
    preferred: Optional[str] = None  # "" clears it
    return_margin: Optional[float] = None


class Router:
    def __init__(self, backends: list[Backend], policy: str, adaptive: AdaptiveConfig,
                 timeout_s: float, poll_s: float, probe_s: float, down_after: int, up_after: int,
                 stall_ms: float = 1000.0):
        self.backends = backends
        self.by_name = {b.name: b for b in backends}
        self.policy = policy
        self.adaptive = adaptive
        self.timeout_s, self.poll_s, self.probe_s = timeout_s, poll_s, probe_s
        self.down_after, self.up_after = down_after, up_after
        self.stall_ms = stall_ms
        self.rr = itertools.count()
        self.rng = random.Random(0)
        self.started = time.monotonic()
        self.failovers = 0
        self.recent: deque[dict] = deque(maxlen=5000)  # per-request log for dashboards
        self.reset_adaptive()
        self.client: httpx.AsyncClient | None = None

    def reset_adaptive(self) -> None:
        self.active = self.by_name.get(self.adaptive.preferred or "", self.backends[0])
        self.better_since: Optional[float] = None
        self.last_switch = -1e9
        self.switches: list[dict] = []

    # ------------------------------------------------------------ health
    def mark(self, b: Backend, ok: bool, hard: bool = False) -> None:
        """Update health with hysteresis. `hard` = connection refused: down at once."""
        if ok:
            b.fail_streak, b.ok_streak = 0, b.ok_streak + 1
            if not b.healthy and b.ok_streak >= self.up_after:
                b.healthy = True
        else:
            b.ok_streak, b.fail_streak = 0, b.fail_streak + 1
            if hard or b.fail_streak >= self.down_after:
                if b.healthy:
                    b.probe_ms = None  # stale once down; re-measure after it rejoins
                b.healthy = False

    async def poll_loop(self) -> None:
        while True:
            await asyncio.gather(*(self.poll_one(b) for b in self.backends))
            await asyncio.sleep(self.poll_s)

    async def poll_one(self, b: Backend) -> None:
        try:
            h, s = await asyncio.gather(
                self.client.get(f"{b.url}/health", timeout=0.5),
                self.client.get(f"{b.url}/stats", timeout=0.5),
            )
            self.mark(b, h.status_code == 200)
            if s.status_code == 200:
                st = s.json()
                b.queue_depth = st["queue_depth"]
                b.max_concurrency = st["max_concurrency"]
                b.gpu_util = st["gpu_util"]
        except httpx.HTTPError:
            self.mark(b, False)

    async def probe_loop(self) -> None:
        """Send the same tiny forecast to every box, so adaptive can compare
        boxes even when one of them is getting no real traffic."""
        while True:
            # Fire probes without waiting for them: a box with a deep queue must
            # not hold up decisions about the other one.
            for b in self.backends:
                if b.probe_started is None:
                    asyncio.create_task(self.probe_one(b))
            self.decide_adaptive()
            await asyncio.sleep(self.probe_s)

    async def probe_one(self, b: Backend) -> None:
        if not b.healthy:
            return
        t0 = time.perf_counter()
        b.probe_started = time.monotonic()
        try:
            r = await self.client.post(f"{b.url}/v1/forecast", json=PROBE_BODY, timeout=self.timeout_s)
            if r.status_code != 200:
                # 503 = queue full / overloaded: as bad as a timeout.
                r_ms = self.timeout_s * 1000
                b.probe_ms = r_ms if b.probe_ms is None else 0.7 * b.probe_ms + 0.3 * r_ms
                return
            # Use the box's own GPU compute time, not round-trip time. Round trip
            # includes queueing behind our own traffic, so the busy active box
            # would always look worse than the idle one and adaptive would flap.
            # Compute time rises only when the GPU itself is slower (training).
            ms = r.json().get("timing", {}).get("compute_ms") or (time.perf_counter() - t0) * 1000
        except httpx.TimeoutException:
            ms = self.timeout_s * 1000  # a hanging box scores as badly as possible
        except httpx.HTTPError:
            return
        finally:
            b.probe_started = None
        b.probe_ms = ms if b.probe_ms is None else 0.7 * b.probe_ms + 0.3 * ms

    def decide_adaptive(self) -> None:
        now = time.monotonic()
        cur = self.active
        others = [b for b in self.backends if b is not cur and b.healthy]
        if not others:
            self.better_since = None
            return
        if not cur.healthy:  # failover: no hysteresis
            self.switch(others[0], now, "active box down")
            return
        sc = {b.name: b.score(now, self.stall_ms) for b in self.backends}
        pref = self.by_name.get(self.adaptive.preferred or "")
        if pref is not None and pref is not cur and pref.healthy:
            # Away from home: go back once home is about as good as where we are.
            target = pref
            if sc[cur.name] is None or sc[pref.name] is None:
                return
            wants = sc[pref.name] <= sc[cur.name] * (1 + self.adaptive.return_margin)
            reason = f"back to preferred {pref.name} ({sc[pref.name]:.0f}ms vs {sc[cur.name]:.0f}ms)"
        else:
            target = min(others, key=lambda b: sc[b.name] if sc[b.name] is not None else float("inf"))
            if sc[cur.name] is None or sc[target.name] is None:
                return
            wants = sc[target.name] < sc[cur.name] * (1 - self.adaptive.margin)
            reason = f"{target.name} {sc[target.name]:.0f}ms vs {cur.name} {sc[cur.name]:.0f}ms"
        if wants:
            if self.better_since is None:
                self.better_since = now
            elif (now - self.better_since >= self.adaptive.window_s
                  and now - self.last_switch >= self.adaptive.cooldown_s):
                self.switch(target, now, reason)
        else:
            self.better_since = None

    def switch(self, to: Backend, now: float, reason: str) -> None:
        self.switches.append({"t": round(now - self.started, 2), "from": self.active.name,
                              "to": to.name, "reason": reason})
        self.active, self.last_switch, self.better_since = to, now, None

    # ------------------------------------------------------------ routing
    def order(self) -> list[Backend]:
        """Backends in the order to try them. Healthy ones first."""
        bs = self.backends
        if self.policy == "static":
            ranked = list(bs)
        elif self.policy == "round_robin":
            healthy = [b for b in bs if b.healthy] or bs
            first = healthy[next(self.rr) % len(healthy)]
            ranked = [first] + [b for b in bs if b is not first]
        elif self.policy == "least_loaded":
            ranked = sorted(bs, key=lambda b: (b.load(), self.rng.random()))
        else:  # adaptive
            ranked = [self.active] + [b for b in bs if b is not self.active]
        return sorted(ranked, key=lambda b: not b.healthy)  # stable: keeps policy order

    async def forward(self, body: bytes) -> Response:
        t0 = time.perf_counter()
        resp = await self._forward(body)
        self.recent.append({
            "t": round(time.monotonic() - self.started, 3),
            "box": resp.headers.get("X-Served-By", "none"),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "status": resp.status_code,
            "failover": resp.headers.get("X-Router-Failover") == "true",
        })
        return resp

    def metrics(self, window_s: float) -> dict:
        """Rolling stats over the last `window_s` seconds, overall and per box."""
        now = time.monotonic() - self.started
        rows = [r for r in self.recent if r["t"] >= now - window_s]

        def agg(rs: list[dict]) -> dict:
            ok = sorted(r["ms"] for r in rs if r["status"] < 400)
            q = lambda f: ok[min(len(ok) - 1, int(f * len(ok)))] if ok else None
            return {"requests": len(rs), "rps": round(len(rs) / window_s, 2),
                    "errors": sum(r["status"] >= 500 for r in rs),
                    "failovers": sum(r["failover"] for r in rs),
                    "p50_ms": q(0.50), "p95_ms": q(0.95), "p99_ms": q(0.99)}

        return {"now": round(now, 3), "window_s": window_s, "overall": agg(rows),
                "per_box": {b.name: agg([r for r in rows if r["box"] == b.name]) for b in self.backends}}

    async def _forward(self, body: bytes) -> Response:
        tried: list[str] = []
        last_error = "no backends"
        for b in self.order():
            tried.append(b.name)
            b.in_flight += 1
            try:
                r = await self.client.post(
                    f"{b.url}/v1/forecast", content=body,
                    headers={"content-type": "application/json"}, timeout=self.timeout_s)
            except httpx.ConnectError as e:
                b.failures += 1
                self.mark(b, False, hard=True)
                last_error = f"{b.name}: connect failed ({e.__class__.__name__})"
                continue
            except httpx.HTTPError as e:
                b.failures += 1
                self.mark(b, False)
                last_error = f"{b.name}: {e.__class__.__name__}"
                continue
            finally:
                b.in_flight -= 1

            if r.status_code >= 500:
                b.failures += 1
                self.mark(b, False)
                last_error = f"{b.name}: HTTP {r.status_code}"
                continue
            # 2xx and 4xx (bad input) go straight back to the client.
            b.served += r.status_code < 400
            if len(tried) > 1:
                self.failovers += 1
            return Response(content=r.content, status_code=r.status_code,
                            media_type="application/json", headers=self.headers(b, tried))
        return JSONResponse({"detail": f"all backends failed; last: {last_error}"},
                            status_code=503, headers=self.headers(None, tried))

    def headers(self, b: Optional[Backend], tried: list[str]) -> dict:
        return {
            "X-Served-By": b.name if b else "none",
            "X-Router-Policy": self.policy,
            "X-Router-Attempts": str(len(tried)),
            "X-Router-Failover": "true" if len(tried) > 1 else "false",
        }

    def status(self) -> dict:
        return {
            "policy": self.policy,
            "adaptive": {**self.adaptive.model_dump(), "active": self.active.name,
                         "switches": self.switches},
            "failovers": self.failovers,
            "backends": {b.name: b.view() for b in self.backends},
        }


def create_app(router: Router) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
        router.client = httpx.AsyncClient(limits=limits)
        tasks = [asyncio.create_task(router.poll_loop()), asyncio.create_task(router.probe_loop())]
        yield
        for t in tasks:
            t.cancel()
        await router.client.aclose()

    app = FastAPI(title="forecast router", lifespan=lifespan)

    @app.post("/v1/forecast")
    async def forecast(request: Request):
        return await router.forward(await request.body())

    @app.get("/health")
    async def health():
        up = [b.name for b in router.backends if b.healthy]
        return JSONResponse({"status": "ok" if up else "down", "healthy_backends": up},
                            status_code=200 if up else 503)

    @app.get("/router/metrics")
    async def metrics(window: float = 10.0, since: Optional[float] = None):
        """Rolling stats, plus the raw requests after `since` (router clock, s)."""
        m = router.metrics(max(1.0, min(window, 300.0)))
        if since is not None:
            m["requests"] = [r for r in router.recent if r["t"] > since][-2000:]
        return m

    @app.get("/router/status")
    async def status():
        return router.status()

    @app.post("/router/config")
    async def config(u: ConfigUpdate):
        if u.policy and u.policy != router.policy:
            router.policy = u.policy
            router.reset_adaptive()
        for k in ("margin", "window_s", "cooldown_s", "return_margin"):
            if getattr(u, k) is not None:
                setattr(router.adaptive, k, getattr(u, k))
        if u.preferred is not None:
            if u.preferred and u.preferred not in router.by_name:
                return JSONResponse({"detail": f"unknown backend {u.preferred!r}"}, status_code=422)
            router.adaptive.preferred = u.preferred or None
        return router.status()

    return app


def parse_backend(s: str) -> Backend:
    name, _, url = s.partition("=")
    if not url:
        raise argparse.ArgumentTypeError("use NAME=URL, e.g. box-a=http://127.0.0.1:8001")
    return Backend(name, url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", type=parse_backend, action="append", required=True,
                    help="NAME=URL; repeat. The first one is 'box A' for the static policy.")
    ap.add_argument("--policy", choices=POLICIES, default="adaptive")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=5.0, help="per-attempt timeout (s) before failing over")
    ap.add_argument("--poll", type=float, default=0.25, help="health/stats poll interval (s)")
    ap.add_argument("--probe", type=float, default=0.5, help="adaptive probe interval (s)")
    ap.add_argument("--stall-ms", type=float, default=1000.0,
                    help="a probe outstanding longer than this counts as a slow box")
    ap.add_argument("--down-after", type=int, default=2, help="failed checks before a box is marked down")
    ap.add_argument("--up-after", type=int, default=2, help="good checks before a box rejoins")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--preferred", default=None, help="adaptive home box, e.g. box-b")
    ap.add_argument("--return-margin", type=float, default=0.20)
    a = ap.parse_args()
    if a.preferred and a.preferred not in {b.name for b in a.backend}:
        ap.error(f"--preferred {a.preferred!r} is not one of the --backend names")
    r = Router(a.backend, a.policy,
               AdaptiveConfig(margin=a.margin, window_s=a.window, cooldown_s=a.cooldown,
                              preferred=a.preferred, return_margin=a.return_margin),
               a.timeout, a.poll, a.probe, a.down_after, a.up_after, a.stall_ms)
    uvicorn.run(create_app(r), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
