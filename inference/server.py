"""Real forecast server: the API.md contract backed by the inference engine.

Drop-in replacement for mock_server.py, so the router needs no changes:

  python -m inference.server --box box-a --port 8001

Adds one endpoint beyond the contract, `GET /v1/metrics`, with the full JSON
metrics snapshot (latency p50/p95/p99, throughput, load, availability, GPU
memory). `--metrics-file` also writes that snapshot to disk every few seconds.

While the model loads, /health answers 503 "down", so the router keeps traffic
away until warm-up has finished.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from contract import ForecastRequest, ForecastResponse, Health, Stats

from .config import EngineConfig, add_engine_args, engine_config_from_args
from .engine import InferenceEngine, InferenceError, InferenceTimeout, Overloaded
from .forecast import build_response, build_rows, issue_time
from .metrics import log_event, percentile, setup_logging
from .model import MODEL_VERSION


def write_json_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def create_app(engine: InferenceEngine, metrics_file: str | None = None,
               metrics_interval_s: float = 2.0) -> FastAPI:
    cfg = engine.cfg

    async def dump_metrics() -> None:
        while True:
            await asyncio.sleep(metrics_interval_s)
            write_json_atomic(metrics_file, engine.snapshot())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [asyncio.create_task(engine.start())]  # load in the background; /health says "down" meanwhile
        if metrics_file:
            tasks.append(asyncio.create_task(dump_metrics()))
        yield
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.stop()

    app = FastAPI(title=f"wxrouter inference server ({cfg.box})", lifespan=lifespan)

    @app.post("/v1/forecast", response_model=ForecastResponse)
    async def forecast(req: ForecastRequest):
        t0 = time.perf_counter()
        issued = issue_time()
        x = build_rows(req.lat, req.lon, req.ensemble_members, issued)
        try:
            y, t = await engine.infer(x, req.hours_ahead)
        except (Overloaded, InferenceTimeout) as e:
            raise HTTPException(503, str(e))
        except InferenceError as e:
            raise HTTPException(500, str(e))
        timing = {"queue_ms": round(t["queue_ms"], 2), "compute_ms": round(t["compute_ms"], 2), "total_ms": 0.0}
        body = build_response(req.model_dump(), y, issued, cfg.box, timing)
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return JSONResponse(body)  # already contract-shaped; skip a second validation pass

    @app.get("/health", response_model=Health)
    async def health():
        status = engine.status()
        body = {"status": status, "box": cfg.box, "model_version": MODEL_VERSION, "contract_version": "1.0",
                "uptime_s": round(time.monotonic() - engine.started, 1)}
        return JSONResponse(body, status_code=503 if status == "down" else 200)

    @app.get("/stats", response_model=Stats)
    async def stats():
        m = engine.metrics
        recent = list(m.total_ms)[-200:]
        return Stats(
            box=cfg.box, in_flight=engine.in_flight, queue_depth=engine.queue_depth,
            max_concurrency=cfg.concurrency, gpu_util=engine.gpu_util(),
            ewma_latency_ms=round(m.ewma_ms, 1),
            p50_latency_ms=round(percentile(recent, 0.50), 1),
            p95_latency_ms=round(percentile(recent, 0.95), 1),
            requests_total=m.requests_total, errors_total=m.errors_total, mode="normal",
        )

    @app.get("/v1/metrics")
    async def metrics():
        return engine.snapshot()

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="alias for --box, to match mock_server.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--metrics-file", help="write the JSON metrics snapshot here every --metrics-interval s")
    ap.add_argument("--metrics-interval", type=float, default=2.0)
    ap.add_argument("--log-level", default="INFO")
    add_engine_args(ap)
    a = ap.parse_args()
    if a.name:
        a.box = a.name
    setup_logging(a.log_level)
    cfg: EngineConfig = engine_config_from_args(a)
    log_event("server_start", box=cfg.box, host=a.host, port=a.port)
    uvicorn.run(create_app(InferenceEngine(cfg), a.metrics_file, a.metrics_interval),
                host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
