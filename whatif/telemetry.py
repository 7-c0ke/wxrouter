"""The recorder: one sample a second of what the stack shows from outside, in a ring buffer.

Each sample is a plain dict (so a buffer saves as JSON and replays exactly):

    {"t": 1763140000.0,                                    wall clock, seconds
     "router": {"policy", "active", "preferred", "switches", "backends": {box: {healthy, in_flight, queue_depth,
                probe_ms, health_ms, err_rate, served, failures}}} | None,
     "traffic": {"rps", "p50_ms", "p95_ms", "errors", "err_pct", "per_box": {box: {"requests", "p95_ms"}}} | None,
     "boxes": {box: {"health": {"code", "ms"}, "stats": {...} | None}},
     "probe": {box: {"t", "code", "compute_ms", "total_ms"} | {"t", "pending_ms"}},   only when there is news
     "jobs": {box: {"servers": {name: state}, "jobs": {name: state}}}                  only with supervisors}

What is left out on purpose: anything a fault injector sets that a real operator would not see. /stats keeps a
whitelist of load numbers (the mock's and the real server's `mode`, `slowdown`, `busy_other` are dropped), and
/health is kept as its HTTP code and round trip only (the mock answers "degraded" in slow mode).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import httpx

WINDOW_S = 180
STATS_KEEP = ("in_flight", "queue_depth", "max_concurrency", "gpu_util", "ewma_latency_ms",
              "p50_latency_ms", "p95_latency_ms", "requests_total", "errors_total")
BACKEND_KEEP = ("healthy", "in_flight", "queue_depth", "probe_ms", "health_ms", "err_rate", "served", "failures")
PROBE_BODY = {"lat": 41.98, "lon": -87.9, "hours_ahead": 12, "step_hours": 6, "request_id": "whatif-probe"}


def strip_stats(stats: Any) -> dict | None:
    if not isinstance(stats, dict):
        return None
    return {k: stats[k] for k in STATS_KEEP if k in stats}


def strip_router(status: Any) -> dict | None:
    if not isinstance(status, dict):
        return None
    adaptive = status.get("adaptive") or {}
    return {"policy": status.get("policy"), "active": adaptive.get("active"), "preferred": adaptive.get("preferred"),
            "switches": len(adaptive.get("switches") or []), "failovers": status.get("failovers"),
            "backends": {name: {k: b.get(k) for k in BACKEND_KEEP}
                         for name, b in (status.get("backends") or {}).items() if isinstance(b, dict)}}


def strip_metrics(m: Any) -> dict | None:
    if not isinstance(m, dict) or not isinstance(m.get("overall"), dict):
        return None
    o = m["overall"]
    n = o.get("requests") or 0
    return {"rps": o.get("rps"), "p50_ms": o.get("p50_ms"), "p95_ms": o.get("p95_ms"), "errors": o.get("errors"),
            "err_pct": round(100.0 * (o.get("errors") or 0) / n, 1) if n else 0.0,
            "per_box": {b: {"requests": v.get("requests"), "p95_ms": v.get("p95_ms")}
                        for b, v in (m.get("per_box") or {}).items()}}


def strip_supervisor(status: Any) -> dict | None:
    if not isinstance(status, dict):
        return None
    def state(v: Any, job: bool) -> Any:
        v = v or {}
        code = v.get("exit")
        if job and v.get("state") == "failed" and isinstance(code, int) and not isinstance(code, bool) and code in (3, -15, -9, -2):
            # Stopped on request: weather/finetune.py leaves with 3 after a stop; -15, -9 and -2 are the signals a
            # stop sends. A crash (-11, -6, -7) stays "failed".
            return "stopped"
        return v.get("state")

    def states(group: Any, job: bool) -> dict:
        return {name: state(v, job) for name, v in (group or {}).items()} if isinstance(group, dict) else {}
    return {"servers": states(status.get("servers"), False), "jobs": states(status.get("jobs"), True)}


class Recorder:
    """Polls every `period_s` into a buffer of the last `window_s` seconds. Run with `await rec.run()`
    (cancel it to stop), or call `await rec.sample_once()` from your own loop."""

    def __init__(self, router: str, boxes: dict[str, str], stacks: dict[str, str] | None = None,
                 period_s: float = 1.0, window_s: int = WINDOW_S, probe_every_s: float = 2.0,
                 probe_timeout_s: float = 3.0, metrics_window_s: float = 2.0,
                 clock: Callable[[], float] = time.time):
        self.router, self.boxes, self.stacks = router.rstrip("/"), dict(boxes), dict(stacks or {})
        self.period_s, self.probe_every_s, self.probe_timeout_s = period_s, probe_every_s, probe_timeout_s
        self.metrics_window_s = metrics_window_s
        self.buf: deque[dict] = deque(maxlen=max(1, int(window_s / period_s)))
        self.clock = clock
        self.client: httpx.AsyncClient | None = None
        self._probe_news: dict[str, dict] = {}
        self._probe_started: dict[str, float] = {}
        self._last_probe = 0.0
        self._stack_ok = {b: True for b in self.stacks}

    # ---------------------------------------------------------------- fetching
    async def _get(self, url: str, timeout: float = 0.8) -> tuple[int, Any, float]:
        """(HTTP code or 0 when no answer, JSON or None, round trip ms)."""
        t0 = time.perf_counter()
        try:
            r = await self.client.get(url, timeout=timeout)
            ms = (time.perf_counter() - t0) * 1000
            try:
                return r.status_code, r.json(), ms
            except ValueError:
                return r.status_code, None, ms
        except httpx.HTTPError:
            return 0, None, (time.perf_counter() - t0) * 1000

    async def _probe(self, box: str, url: str) -> None:
        start = self.clock()
        self._probe_started[box] = start
        t0 = time.perf_counter()
        try:
            r = await self.client.post(f"{url}/v1/forecast", json=PROBE_BODY, timeout=self.probe_timeout_s)
            total = (time.perf_counter() - t0) * 1000
            compute = None
            if r.status_code == 200:
                try:
                    compute = (r.json().get("timing") or {}).get("compute_ms")
                except ValueError:
                    pass
            self._probe_news[box] = {"t": start, "code": r.status_code, "compute_ms": compute, "total_ms": round(total, 1)}
        except httpx.TimeoutException:
            self._probe_news[box] = {"t": start, "code": 0, "compute_ms": None, "total_ms": self.probe_timeout_s * 1000, "timeout": True}
        except httpx.HTTPError:
            self._probe_news[box] = {"t": start, "code": 0, "compute_ms": None, "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        finally:
            self._probe_started.pop(box, None)

    async def sample_once(self) -> dict:
        now = self.clock()
        if now - self._last_probe >= self.probe_every_s - 1e-6:
            self._last_probe = now
            for box, url in self.boxes.items():
                if box not in self._probe_started:
                    asyncio.create_task(self._probe(box, url))

        calls = [self._get(f"{self.router}/router/status"),
                 self._get(f"{self.router}/router/metrics?window={self.metrics_window_s}")]
        for url in self.boxes.values():
            calls += [self._get(f"{url}/health"), self._get(f"{url}/stats")]
        stack_boxes = [b for b in self.stacks if self._stack_ok.get(b, True) or int(now) % 10 == 0]
        calls += [self._get(f"{self.stacks[b]}/status") for b in stack_boxes]
        res = await asyncio.gather(*calls)

        sample: dict = {"t": round(now, 3), "router": strip_router(res[0][1]) if res[0][0] == 200 else None,
                        "traffic": strip_metrics(res[1][1]) if res[1][0] == 200 else None, "boxes": {}}
        for i, box in enumerate(self.boxes):
            h, s = res[2 + 2 * i], res[3 + 2 * i]
            sample["boxes"][box] = {"health": {"code": h[0], "ms": round(h[2], 1)},
                                    "stats": strip_stats(s[1]) if s[0] == 200 else None}
        probe = {}
        for box in self.boxes:
            if box in self._probe_news:
                probe[box] = self._probe_news.pop(box)
            elif box in self._probe_started:
                probe[box] = {"t": self._probe_started[box], "pending_ms": round((now - self._probe_started[box]) * 1000, 1)}
        if probe:
            sample["probe"] = probe
        if stack_boxes:
            jobs = {}
            for box, r in zip(stack_boxes, res[2 + 2 * len(self.boxes):]):
                self._stack_ok[box] = r[0] == 200
                if r[0] == 200:
                    jobs[box] = strip_supervisor(r[1])
            if jobs:
                sample["jobs"] = jobs
        self.buf.append(sample)
        return sample

    async def run(self, duration_s: float | None = None) -> None:
        own = self.client is None
        if own:
            self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=50))
        end = None if duration_s is None else time.monotonic() + duration_s
        try:
            nxt = time.monotonic()
            while end is None or time.monotonic() < end:
                await self.sample_once()
                nxt += self.period_s
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
        finally:
            if own:
                await self.client.aclose()
                self.client = None

    def snapshot(self) -> list[dict]:
        return list(self.buf)


# -------------------------------------------------------------------- save / load
def save_incident(path: Path, incident: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(incident, separators=(",", ":")) + "\n")
    return path


def load_incident(path: Path) -> dict:
    inc = json.loads(Path(path).read_text())
    if not isinstance(inc.get("buffers"), list) or "alert" not in inc:
        raise ValueError(f"{path}: not an incident file (needs 'buffers' and 'alert')")
    return inc


def load_incidents(folder: Path) -> list[dict]:
    return [load_incident(p) for p in sorted(Path(folder).glob("*.json"))]
