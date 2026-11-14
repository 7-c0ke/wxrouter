"""Mock forecast server: same API as the real GPU server, no GPU needed.

It fakes the GPU with a semaphore (`max_concurrency` slots). Each request holds
a slot for a simulated compute time that grows with hours_ahead and
ensemble_members, with log-normal jitter. Extra requests wait in a queue, which
shows up as `queue_depth` in /stats and `queue_ms` in the response.

Run:  python mock_server.py --name box-a --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
import statistics
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from contract import (
    EnsembleMember,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    Health,
    Stats,
    Timing,
)


class AdminMode(BaseModel):
    mode: str = Field(..., pattern="^(normal|slow|error|hang)$")
    slow_factor: float = Field(3.0, ge=1.0, le=50.0)


class MockState:
    def __init__(self, name: str, max_concurrency: int, max_queue: int,
                 base_ms: float, per_hour_ms: float, jitter: float, seed: int | None):
        self.name = name
        self.model_version = "wx-mock-2026-11-14"
        self.started = time.monotonic()
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.base_ms = base_ms
        self.per_hour_ms = per_hour_ms
        self.jitter = jitter
        self.rng = random.Random(seed)
        self.gpu = asyncio.Semaphore(max_concurrency)
        self.in_flight = 0
        self.queued = 0
        self.requests_total = 0
        self.errors_total = 0
        self.latencies: deque[float] = deque(maxlen=200)
        self.ewma = 0.0
        self.mode = "normal"
        self.slow_factor = 3.0

    def compute_ms(self, req: ForecastRequest) -> float:
        """Simulated GPU time: cost grows with forecast length and ensemble size."""
        lead_cost = self.per_hour_ms * req.hours_ahead
        member_cost = lead_cost * req.ensemble_members
        ms = (self.base_ms + member_cost) * self.rng.lognormvariate(0, self.jitter)
        if self.mode == "slow":
            ms *= self.slow_factor
        return ms

    def record(self, total_ms: float) -> None:
        self.latencies.append(total_ms)
        self.ewma = total_ms if self.ewma == 0 else 0.8 * self.ewma + 0.2 * total_ms


def fake_points(req: ForecastRequest, issued: datetime, member: int) -> list[ForecastPoint]:
    """Plausible-looking weather from lat/lon, so values differ by location."""
    rnd = random.Random(hash((round(req.lat, 2), round(req.lon, 2), member)))
    base_t = 27 - 0.45 * abs(req.lat) + rnd.uniform(-1.5, 1.5)
    base_p = 1013 + rnd.uniform(-8, 8)
    pts = []
    for lead in range(req.step_hours, req.hours_ahead + 1, req.step_hours):
        vt = issued + timedelta(hours=lead)
        local_hour = (vt.hour + req.lon / 15) % 24
        diurnal = 5 * math.sin((local_hour - 9) / 24 * 2 * math.pi)
        spread = 0.005 * lead * member  # members drift apart with lead time
        pts.append(ForecastPoint(
            lead_hours=lead,
            valid_time=vt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            temp_c=round(base_t + diurnal + rnd.gauss(0, 0.5) + spread, 1),
            wind_ms=round(max(0.0, 4 + rnd.gauss(0, 1.5)), 1),
            precip_mm=round(max(0.0, rnd.gauss(-0.5, 1.0)), 1),
            pressure_hpa=round(base_p + 2 * math.sin(lead / 30) + rnd.gauss(0, 0.4), 1),
        ))
    return pts


def mean_points(members: list[EnsembleMember]) -> list[ForecastPoint]:
    out = []
    for i, p0 in enumerate(members[0].points):
        col = [m.points[i] for m in members]
        out.append(ForecastPoint(
            lead_hours=p0.lead_hours, valid_time=p0.valid_time,
            temp_c=round(statistics.fmean(p.temp_c for p in col), 1),
            wind_ms=round(statistics.fmean(p.wind_ms for p in col), 1),
            precip_mm=round(statistics.fmean(p.precip_mm for p in col), 1),
            pressure_hpa=round(statistics.fmean(p.pressure_hpa for p in col), 1),
        ))
    return out


def pct(values, q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def create_app(state: MockState) -> FastAPI:
    app = FastAPI(title=f"mock forecast server ({state.name})")

    @app.post("/v1/forecast", response_model=ForecastResponse)
    async def forecast(req: ForecastRequest):
        t0 = time.perf_counter()
        state.requests_total += 1
        if state.mode == "error":
            state.errors_total += 1
            raise HTTPException(503, "box unavailable (mock error mode)")
        if state.mode == "hang":
            await asyncio.sleep(3600)
        if state.queued >= state.max_queue:
            state.errors_total += 1
            raise HTTPException(503, "overloaded: queue full")

        state.queued += 1
        try:
            await state.gpu.acquire()
        finally:
            state.queued -= 1
        t_slot = time.perf_counter()
        state.in_flight += 1
        try:
            await asyncio.sleep(state.compute_ms(req) / 1000)
        finally:
            state.in_flight -= 1
            state.gpu.release()
        t_done = time.perf_counter()

        issued = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if req.ensemble_members > 1:
            members = [EnsembleMember(member=m, points=fake_points(req, issued, m))
                       for m in range(req.ensemble_members)]
            points = mean_points(members)
        else:
            members, points = None, fake_points(req, issued, 0)

        timing = Timing(
            queue_ms=round((t_slot - t0) * 1000, 2),
            compute_ms=round((t_done - t_slot) * 1000, 2),
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        state.record(timing.total_ms)
        return ForecastResponse(
            request_id=req.request_id, box=state.name, model_version=state.model_version,
            issued_at=issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
            lat=req.lat, lon=req.lon, hours_ahead=req.hours_ahead, step_hours=req.step_hours,
            low_confidence=req.hours_ahead > 72,
            forecast=points, members=members, timing=timing,
        )

    @app.get("/health", response_model=Health)
    async def health():
        h = Health(status="ok", box=state.name, model_version=state.model_version,
                   uptime_s=round(time.monotonic() - state.started, 1))
        if state.mode == "error":
            h.status = "down"
            return JSONResponse(h.model_dump(), status_code=503)
        if state.mode == "slow":
            h.status = "degraded"
        return h

    @app.get("/stats", response_model=Stats)
    async def stats():
        util = state.in_flight / state.max_concurrency
        if state.mode == "slow":  # a training job keeps the GPU busy
            util = min(1.0, 0.7 + 0.3 * util)
        return Stats(
            box=state.name, in_flight=state.in_flight, queue_depth=state.queued,
            max_concurrency=state.max_concurrency, gpu_util=round(util, 2),
            ewma_latency_ms=round(state.ewma, 1),
            p50_latency_ms=round(pct(state.latencies, 0.50), 1),
            p95_latency_ms=round(pct(state.latencies, 0.95), 1),
            requests_total=state.requests_total, errors_total=state.errors_total,
            mode=state.mode,
        )

    @app.post("/admin/mode")
    async def set_mode(m: AdminMode):
        state.mode, state.slow_factor = m.mode, m.slow_factor
        return {"box": state.name, "mode": state.mode, "slow_factor": state.slow_factor}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="box-a")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-concurrency", type=int, default=4, help="simulated GPU slots")
    ap.add_argument("--max-queue", type=int, default=64, help="503 when more are waiting")
    ap.add_argument("--base-ms", type=float, default=30.0, help="fixed cost per request")
    ap.add_argument("--per-hour-ms", type=float, default=2.5, help="cost per forecast hour per member")
    ap.add_argument("--jitter", type=float, default=0.15, help="log-normal sigma")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    state = MockState(a.name, a.max_concurrency, a.max_queue, a.base_ms, a.per_hour_ms, a.jitter, a.seed)
    uvicorn.run(create_app(state), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
