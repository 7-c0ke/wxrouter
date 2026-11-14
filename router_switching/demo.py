"""The "Run Switching Demo" button: one scripted fault run through the team's router, measured live.

Nothing here routes anything. It asks the existing router to use its adaptive rule with GPU A as the
home box, then uses box A's existing /admin/mode test hook to slow it, restore it, fail it and restore
it again, while forecasts flow through the router. Every number it reports is read back from the
router's own request log (GET /router/metrics?since=) and switch list (GET /router/status).

  1 normal           traffic on GPU A (the home box)
  2 slow GPU A       the router's adaptive rule should move traffic to GPU B
  3 restore GPU A    traffic should return home
  4 fail GPU A       requests fail over to GPU B
  5 restore GPU A    GPU A is marked healthy again and traffic returns

When the run ends, or breaks, box A is put back to normal and the router gets back the policy and
preferred box it had before. The result is labelled real GPU, CPU or simulated from what the boxes
themselves report, and saved under results/demo/.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import httpx

from .dashboard import CPU, REAL, SIM, UNKNOWN, live

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "demo"
BODY = {"lat": 41.98, "lon": -87.9, "hours_ahead": 24, "step_hours": 1, "request_id": "switching-demo"}
PHASES = [("normal", "Normal operation"), ("slow", "GPU A slowed"), ("restore", "GPU A back to normal"),
          ("fail", "GPU A fails"), ("recover", "GPU A recovers")]


def quantile(v: list[float], f: float) -> Optional[float]:
    return round(v[min(len(v) - 1, int(f * len(v)))], 1) if v else None


def summarize(rows: list[dict], a: str, b: str, seconds: float) -> dict:
    """Latency, throughput, errors and who served, from the router's own request rows."""
    ok = sorted(r["ms"] for r in rows if r["status"] < 400)
    n = len(rows)
    errors = sum(r["status"] >= 500 for r in rows)
    na, nb = sum(r["box"] == a for r in rows), sum(r["box"] == b for r in rows)
    return {"seconds": round(seconds, 1), "requests": n, "errors": errors,
            "error_pct": round(100 * errors / n, 2) if n else None,
            "rps": round(n / seconds, 1) if seconds > 0 else None,
            "p50_ms": quantile(ok, .5), "p95_ms": quantile(ok, .95), "p99_ms": quantile(ok, .99),
            "share_a_pct": round(100 * na / n, 1) if n else None, "share_b_pct": round(100 * nb / n, 1) if n else None,
            "failovers": sum(bool(r["failover"]) for r in rows)}


def per_second(rows: list[dict], a: str, b: str, t0: float, t1: float) -> list[dict]:
    out = []
    for s in range(int(max(0.0, t1 - t0))):
        rs = [r for r in rows if t0 + s <= r["t"] < t0 + s + 1]
        ok = sorted(r["ms"] for r in rs if r["status"] < 400)
        out.append({"t": s, "a": sum(r["box"] == a for r in rs), "b": sum(r["box"] == b for r in rs),
                    "errors": sum(r["status"] >= 500 for r in rs), "p95_ms": quantile(ok, .95)})
    return out


class Unsupported(RuntimeError):
    """The box has the test hook but not this mode (the real forecast server has no slowdown hook)."""


class SwitchingDemo:
    """One run at a time. `state` is what the page polls."""

    def __init__(self, ops, hold_s: float = 6.0, wait_s: float = 25.0, rps: float = 30.0,
                 slow_factor: float = 8.0, save: bool = True, margin: float = 0.6):
        self.margin = margin
        self.ops, self.hold_s, self.wait_s, self.rps = ops, hold_s, wait_s, rps
        self.slow_factor, self.save = slow_factor, save
        self.task: Optional[asyncio.Task] = None
        self.cleanup: Optional[asyncio.Future] = None
        self.state: dict = {"status": "idle", "phases": [{"key": k, "title": t, "state": "waiting"} for k, t in PHASES]}

    # ------------------------------------------------------------ plumbing
    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def _json(self, method: str, url: str, **kw) -> dict:
        for attempt in range(4):   # a port-forward to the cluster drops a connection now and then
            try:
                r = await self.ops.client.request(method, url, timeout=5, **kw)
                break
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise RuntimeError(f"{method} {url.split('//', 1)[-1]}: no answer after 4 tries ({e.__class__.__name__})")
                await asyncio.sleep(0.4)
        try:
            d = r.json()
        except ValueError:
            d = None
        if r.status_code != 200 or not isinstance(d, dict):
            raise RuntimeError(f"{method} {url.split('//', 1)[-1]}: HTTP {r.status_code} {r.text[:120]}")
        return d

    async def _mode(self, box: str, mode: str) -> None:
        url = f"{self.ops.boxes[box]}/admin/mode"
        for attempt in range(4):
            try:
                r = await self.ops.client.post(url, json={"mode": mode, "slow_factor": self.slow_factor}, timeout=10)
                break
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise RuntimeError(f"{box} did not answer /admin/mode ({e.__class__.__name__})")
                await asyncio.sleep(0.4)
        if r.status_code == 404:
            raise RuntimeError(f"{box} has no /admin/mode test hook (start the inference server with --admin)")
        if r.status_code == 400 and mode == "slow":
            raise Unsupported(f"{box} has no slow mode ({r.text[:100]})")
        if r.status_code != 200:
            raise RuntimeError(f"{box} /admin/mode: HTTP {r.status_code} {r.text[:120]}")

    async def _traffic(self, stop: asyncio.Event) -> None:
        """Forecasts through the router at a steady rate, so there is something to switch."""
        pending: set[asyncio.Task] = set()

        async def one() -> None:
            self.state["sent"] += 1
            try:
                await self.ops.client.post(f"{self.ops.router}/v1/forecast", json=BODY, timeout=30)
            except httpx.HTTPError:
                self.state["unanswered"] += 1   # no answer from the router at all; its own log cannot show these

        while not stop.is_set():
            t = asyncio.create_task(one())
            pending.add(t)
            t.add_done_callback(pending.discard)
            await asyncio.sleep(1 / self.rps)
        if pending:
            await asyncio.wait(pending, timeout=10)

    async def shutdown(self) -> None:
        """Called before the page's HTTP client closes: stop a run in progress and wait for its cleanup."""
        if self.running:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        if self.cleanup is not None and not self.cleanup.done():
            await asyncio.wait([self.cleanup], timeout=30)

    # ------------------------------------------------------------ the run
    def start(self) -> dict:
        if self.running:
            return self.state
        if len(self.ops.boxes) < 2:
            self.state = {**self.state, "status": "failed", "error": "the demo needs two boxes"}
            return self.state
        a, b = list(self.ops.boxes)[:2]
        self.state = {"status": "running", "run_id": time.strftime("%Y%m%d-%H%M%S"), "a": a, "b": b, "kind": UNKNOWN,
                      "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": None, "elapsed_s": 0.0,
                      "phases": [{"key": k, "title": t, "state": "waiting"} for k, t in PHASES],
                      "events": [], "timeline": [], "marks": [], "timings": {}, "sent": 0, "unanswered": 0,
                      "traffic_rps": self.rps, "slow_factor": self.slow_factor, "error": None}
        self.task = asyncio.create_task(self._run(a, b))
        return self.state

    async def _run(self, a: str, b: str) -> None:
        st, router = self.state, self.ops.router
        rows: list[dict] = []
        seen = {"switches": 0, "healthy": {}, "since": 0.0, "t0": 0.0}
        stop = asyncio.Event()
        traffic: Optional[asyncio.Task] = None
        before: Optional[dict] = None
        outcome = "failed"

        def event(t: float, kind: str, text: str) -> None:
            st["events"].append({"t": round(t - seen["t0"], 1), "kind": kind, "text": text})

        async def sample() -> dict:
            """Read the router: new request rows, new switches, health changes. Returns its status."""
            m, status = await asyncio.gather(
                self._json("GET", f"{router}/router/metrics", params={"window": 2, "since": seen["since"]}),
                self._json("GET", f"{router}/router/status"))
            new = m.get("requests", [])
            if new:
                rows.extend(new)
                seen["since"] = new[-1]["t"]
            now = m["now"]
            sw = status["adaptive"]["switches"]
            for s in sw[seen["switches"]:]:
                event(s["t"], "switch", f"router switched {s['from']} → {s['to']}: {s['reason']}")
            seen["switches"] = len(sw)
            for n, bk in status["backends"].items():
                was = seen["healthy"].get(n)
                if was is not None and was != bk["healthy"]:
                    event(now, "health", f"router marked {n} {'healthy' if bk['healthy'] else 'down'}")
                seen["healthy"][n] = bk["healthy"]
            st["elapsed_s"] = round(now - seen["t0"], 1)
            st["timeline"] = per_second(rows, a, b, seen["t0"], now)
            st["active"], st["now"] = status["adaptive"]["active"], summarize(
                [r for r in rows if r["t"] >= now - 3], a, b, 3.0)
            return {"now": now, "status": status}

        async def until(cond, max_s: float) -> Optional[float]:
            """Sample every half second until cond(sample) holds. -> router time it first held, or None."""
            end = time.monotonic() + max_s
            while True:
                s = await sample()
                if cond(s):
                    return s["now"]
                if time.monotonic() >= end:
                    return None
                await asyncio.sleep(0.5)

        async def hold(seconds: float) -> None:
            await until(lambda s: False, seconds)

        def first_switch(to: str, after: float) -> Optional[dict]:
            return next((e for e in st["events"] if e["kind"] == "switch" and f"→ {to}" in e["text"]
                         and e["t"] >= round(after - seen["t0"], 1)), None)

        phase_start: dict[str, float] = {}

        async def begin(key: str) -> float:
            now = (await sample())["now"]
            for p in st["phases"]:
                if p["state"] == "running":
                    p["state"] = "done"
                    p.update(summarize([r for r in rows if phase_start[p["key"]] <= r["t"] < now], a, b,
                                       now - phase_start[p["key"]]))
                if p["key"] == key and p["state"] != "skipped":
                    p["state"] = "running"
            if key != "done":
                phase_start[key] = now
                st["phase"] = key
                st["marks"].append({"t": round(now - seen["t0"], 1), "action": dict(PHASES)[key], "box": a})
            return now

        try:
            lv = await live(self.ops.client, router, {a: self.ops.boxes[a], b: self.ops.boxes[b]})
            if lv["router"] is None:
                raise RuntimeError("the router is unreachable")
            kinds = [lv["boxes"][n]["kind"] for n in (a, b)]
            st["kind"] = SIM if SIM in kinds else UNKNOWN if UNKNOWN in kinds else CPU if CPU in kinds else REAL
            st["boxes"] = {n: {k: lv["boxes"][n][k] for k in ("kind", "device", "gpu_uuid", "model_version")} for n in (a, b)}
            if lv["boxes"][a].get("max_batch_rows") is None and st["kind"] == REAL:
                # the trained model: one forecast fills the GPU for tens of ms, so one box takes far fewer per second
                self.rps = st["traffic_rps"] = min(self.rps, 6.0)
            status = await self._json("GET", f"{router}/router/status")
            before = {"policy": status["policy"], "preferred": status["adaptive"].get("preferred") or "",
                      "margin": status["adaptive"].get("margin")}
            st["policy_before"] = before
            st["thresholds"] = {k: status["adaptive"].get(k) for k in ("margin", "window_s", "cooldown_s", "return_margin")}
            cooldown = float(status["adaptive"].get("cooldown_s") or 0)
            wait_s = self.wait_s + cooldown
            m = await self._json("GET", f"{router}/router/metrics", params={"window": 2})
            seen["t0"] = seen["since"] = m["now"]
            await self._mode(a, "normal")
            # home box first: the router picks its active box at the moment the policy changes
            # A wider margin for the run: on batching GPU servers the box carrying the traffic probes a little
            # slower than the idle one, and at the default 25% the rule leaves it with no fault at all
            # (measured, see README). The slowdown injected here is far past either margin.
            await self._json("POST", f"{router}/router/config", json={"preferred": a, "margin": self.margin})
            st["thresholds"]["margin_during_demo"] = self.margin
            status = await self._json("POST", f"{router}/router/config", json={"policy": "adaptive"})
            seen["switches"] = len(status["adaptive"]["switches"])
            event(seen["t0"], "action", f"router set to adaptive, home box {a}, switch margin {self.margin:.0%} "
                                        f"(was {before['policy']}, margin {before['margin']:.0%}); both are put back at the end")
            traffic = asyncio.create_task(self._traffic(stop))

            # 1 normal: traffic settles on the home box
            await begin("normal")
            if await until(lambda s: s["status"]["adaptive"]["active"] == a and s["status"]["backends"][a]["healthy"], wait_s) is None:
                raise RuntimeError(f"{a} never became the active box")
            await hold(self.hold_s)

            # 2 slow: the adaptive rule should leave box A
            if await until(lambda s: s["status"]["adaptive"]["active"] == a, wait_s) is None:
                raise RuntimeError(f"{a} was not the active box when the slowdown was due")
            t = await begin("slow")
            try:
                await self._mode(a, "slow")
                slow = True
            except Unsupported as e:
                slow = False
                event(t, "note", f"slowdown steps skipped: {e}. Nothing was slowed and no slowdown figure is reported.")
                for p in st["phases"]:
                    if p["key"] in ("slow", "restore"):
                        p["state"] = "skipped"
            if slow:
                event(t, "action", f"{a} slowed (slow mode ×{self.slow_factor:g})")
                hit = await until(lambda s: s["status"]["adaptive"]["active"] == b, wait_s)
                sw = first_switch(b, t)
                st["timings"]["slow_to_switch_s"] = round(sw["t"] - (t - seen["t0"]), 1) if hit is not None and sw else None
                if hit is None:
                    event(t + wait_s, "note", f"the router did not leave {a} within {wait_s:g} s")
                await hold(self.hold_s)

                # 3 restore: traffic returns home
                t = await begin("restore")
                await self._mode(a, "normal")
                event(t, "action", f"{a} back to normal")
                hit = await until(lambda s: s["status"]["adaptive"]["active"] == a, wait_s)
                sw = first_switch(a, t)
                st["timings"]["restore_to_home_s"] = round(sw["t"] - (t - seen["t0"]), 1) if hit is not None and sw else None
                await hold(self.hold_s)

            # 4 fail: every request to box A is refused; the router fails over
            t = await begin("fail")
            await self._mode(a, "error")
            event(t, "action", f"{a} failed (it answers every forecast with HTTP 503)")
            hit = await until(lambda s: s["status"]["adaptive"]["active"] == b and not s["status"]["backends"][a]["healthy"], wait_s)
            sw = first_switch(b, t)
            st["timings"]["fail_to_switch_s"] = round(sw["t"] - (t - seen["t0"]), 1) if hit is not None and sw else None
            await hold(self.hold_s)
            t_end = (await sample())["now"]
            down = [r for r in rows if t <= r["t"] < t_end]
            st["timings"]["errors_while_failed"] = sum(r["status"] >= 500 for r in down)
            st["timings"]["failovers_while_failed"] = sum(bool(r["failover"]) for r in down)

            # 5 recover
            t = await begin("recover")
            await self._mode(a, "normal")
            event(t, "action", f"{a} restored")
            up = await until(lambda s: s["status"]["backends"][a]["healthy"], wait_s)
            st["timings"]["restore_to_healthy_s"] = round(up - t, 1) if up is not None else None
            await until(lambda s: s["status"]["adaptive"]["active"] == a, wait_s)

            def served_by_a() -> Optional[float]:
                return next((r["t"] for r in rows if r["t"] > t and r["box"] == a and r["status"] < 400), None)
            # the router logs a request when it completes, so box A can be active before its first answer shows
            await until(lambda s: served_by_a() is not None, wait_s)
            served = served_by_a()
            st["timings"]["restore_to_serving_s"] = round(served - t, 1) if served is not None else None
            await hold(self.hold_s)
            end = await begin("done")
            st["overall"] = summarize([r for r in rows if r["t"] >= seen["t0"]], a, b, end - seen["t0"])
            outcome = "done"
        except asyncio.CancelledError:
            st["error"] = "stopped: the page was shut down during the run"
            raise
        except Exception as e:  # shown on the page; the cleanup below still runs
            st["error"] = f"{e.__class__.__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)
            for p in st["phases"]:
                if p["state"] == "running":
                    p["state"] = "failed"
        finally:
            stop.set()
            st["phase"] = "cleanup"   # still "running" for the page until box A and the policy are confirmed back
            self.cleanup = asyncio.ensure_future(self._cleanup(a, before, traffic, event))
            try:
                await asyncio.shield(self.cleanup)
            finally:
                if self.cleanup.done():
                    st["phase"] = None
                    if st.get("cleanup_error"):
                        outcome = "failed"
                        st["error"] = st.get("error") or "the run finished but the cleanup did not"
                    st["status"] = outcome
                else:   # cancelled a second time: shutdown() awaits the cleanup itself
                    self.cleanup.add_done_callback(lambda _: st.update(status="failed", phase=None))
            st["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            if self.save and st["status"] == "done":
                try:
                    OUT.mkdir(parents=True, exist_ok=True)
                    path = OUT / f"{st['run_id']}-{st['kind'].replace(' ', '-')}.json"
                    path.write_text(json.dumps(st, indent=1))
                    st["saved_to"] = str(path.relative_to(HERE.parent))
                except OSError as e:
                    st["saved_to"] = f"not saved: {e}"

    async def _cleanup(self, a: str, before: Optional[dict], traffic: Optional[asyncio.Task], event) -> None:
        """Leave the cluster as it was found, whatever happened."""
        try:
            await self._mode(a, "normal")
        except RuntimeError as e:
            self.state["cleanup_error"] = f"{a} may still be in a fault mode: {e}"
        if before is not None:
            try:
                # home box and margin first, then the policy: the router re-picks its active box when the
                # policy changes, so this order leaves no switch in its list that no fault caused
                await self._json("POST", f"{self.ops.router}/router/config",
                                 json={"preferred": before["preferred"], "margin": before["margin"]})
                got = await self._json("POST", f"{self.ops.router}/router/config", json={"policy": before["policy"]})
                if (got["policy"] != before["policy"] or (got["adaptive"].get("preferred") or "") != before["preferred"]
                        or got["adaptive"].get("margin") != before["margin"]):
                    raise RuntimeError(f"router answered policy {got['policy']}")
                self.state["policy_restored"] = before
            except (RuntimeError, httpx.HTTPError) as e:
                self.state["cleanup_error"] = (self.state.get("cleanup_error", "") + " · " if self.state.get("cleanup_error") else "") + f"router policy not restored to {before['policy']}: {e}"
        if traffic is not None:
            try:
                await asyncio.wait_for(traffic, 15)
            except (asyncio.TimeoutError, httpx.HTTPError):
                traffic.cancel()
