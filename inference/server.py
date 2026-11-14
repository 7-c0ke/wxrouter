"""Real forecast server: the API.md contract backed by the inference engine.

Drop-in replacement for mock_server.py, so the router needs no changes:

  python -m inference.server --box box-a --port 8001

Adds one endpoint beyond the contract, `GET /v1/metrics`, with the full JSON
metrics snapshot (latency p50/p95/p99, throughput, load, availability, GPU
memory, node identity). `--metrics-file` also writes that snapshot to disk.

While the model loads, /health answers 503 "down", so the router keeps traffic
away until warm-up has finished.

Test hooks, only with `--admin`:
  POST /admin/mode    {"mode": "normal"|"slow"|"error"|"hang", "slow_factor": 4}
                      same call as the mock, so loadgen.py events work unchanged.
                      "slow" starts a real load on the same GPU (contention.py).
  POST /admin/engine  {"max_batch_rows": 1}   change batching while running
  POST /admin/stop    {"down_s": 15}          exit the process; with --supervise
                      it is started again after down_s (a real stop and cold start)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from contract import ForecastRequest, ForecastResponse, Health, Stats

from .config import EngineConfig, add_engine_args, engine_config_from_args
from .contention import Burner
from .engine import InferenceEngine, InferenceError, InferenceTimeout, Overloaded
from .forecast import build_response, build_rows, issue_time
from .metrics import log_event, percentile, setup_logging
from .model import MODEL_VERSION


class AdminMode(BaseModel):
    mode: str = Field(..., pattern="^(normal|slow|error|hang)$")
    slow_factor: float = Field(3.0, ge=1.0, le=50.0)


class AdminEngine(BaseModel):
    max_batch_rows: int = Field(..., ge=1, le=4096)


class AdminStop(BaseModel):
    down_s: float = Field(10.0, ge=0.0, le=600.0)


def write_json_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def create_app(engine: InferenceEngine, metrics_file: str | None = None, metrics_interval_s: float = 2.0,
               admin: bool = False, restart_file: str | None = None, service=None) -> FastAPI:
    """`service` is an inference.afno.AfnoService to serve the trained model; None serves the tiny one."""
    cfg = engine.cfg
    version = service.backend.model_version if service else MODEL_VERSION
    state = {"mode": "normal", "burner": None}

    async def dump_metrics() -> None:
        while True:
            await asyncio.sleep(metrics_interval_s)
            write_json_atomic(metrics_file, snapshot())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [asyncio.create_task(engine.start())]  # load in the background; /health says "down" meanwhile
        if metrics_file:
            tasks.append(asyncio.create_task(dump_metrics()))
        yield
        if state["burner"]:
            state["burner"].clear()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.stop()

    app = FastAPI(title=f"wxrouter inference server ({cfg.box})", lifespan=lifespan)

    def snapshot() -> dict:
        snap = engine.snapshot()
        snap["mode"] = state["mode"]
        if state["mode"] == "error":
            snap["availability"]["status"] = "down"
        elif state["mode"] == "slow" and snap["availability"]["status"] == "ok":
            snap["availability"]["status"] = "degraded"
        return snap

    @app.post("/v1/forecast", response_model=ForecastResponse)
    async def forecast(req: ForecastRequest):
        t0 = time.perf_counter()
        if state["mode"] == "error":
            engine.metrics.error("rejected")
            raise HTTPException(503, "box unavailable (admin error mode)")
        if state["mode"] == "hang":
            await asyncio.sleep(3600)
        try:
            if service:
                asked = req.model_dump()
                x, steps, aux, ctx = await asyncio.to_thread(service.prepare, asked)
                y, t = await engine.infer(x, steps, aux)
            else:
                issued = issue_time()
                y, t = await engine.infer(build_rows(req.lat, req.lon, req.ensemble_members, issued), req.hours_ahead)
        except (Overloaded, InferenceTimeout, LookupError) as e:
            raise HTTPException(503, str(e)) from e
        except InferenceError as e:
            raise HTTPException(500, str(e)) from e
        timing = {"queue_ms": round(t["queue_ms"], 2), "compute_ms": round(t["compute_ms"], 2), "total_ms": 0.0}
        if service:
            return JSONResponse(service.respond(asked, y, ctx, cfg.box, timing))
        body = build_response(req.model_dump(), y, issued, cfg.box, timing)
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return JSONResponse(body)  # already contract-shaped; skip a second validation pass

    @app.get("/health", response_model=Health)
    async def health():
        status = snapshot()["availability"]["status"]
        body = {"status": status, "box": cfg.box, "model_version": version, "contract_version": "1.0",
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
            requests_total=m.requests_total, errors_total=m.errors_total,
            mode="training" if state["mode"] == "slow" else "normal",  # the contract's word for a busy GPU
        )

    @app.get("/v1/metrics")
    async def metrics():
        return snapshot()

    if not admin:
        return app

    @app.post("/admin/mode")
    async def set_mode(m: AdminMode):
        if not engine.ready:
            raise HTTPException(503, "engine not ready")
        if state["burner"] is None:
            state["burner"] = Burner(engine.backend)
        burner: Burner = state["burner"]
        await asyncio.to_thread(burner.set if m.mode == "slow" else burner.clear,
                                *([m.slow_factor] if m.mode == "slow" else []))
        state["mode"] = m.mode
        log_event("admin_mode", box=cfg.box, mode=m.mode, slow_factor=m.slow_factor, burners=len(burner._threads))
        return {"box": cfg.box, "mode": m.mode, "slow_factor": m.slow_factor, "burner_threads": len(burner._threads)}

    @app.post("/admin/engine")
    async def set_engine(e: AdminEngine):
        cfg.max_batch_rows = e.max_batch_rows
        log_event("admin_engine", box=cfg.box, max_batch_rows=e.max_batch_rows)
        return {"box": cfg.box, "max_batch_rows": cfg.max_batch_rows}

    @app.post("/admin/stop")
    async def stop(s: AdminStop, request: Request):
        server = getattr(request.app.state, "server", None)
        if server is None:
            raise HTTPException(409, "not running under `python -m inference.server`")
        if restart_file:
            with open(restart_file, "w") as f:  # noqa: ASYNC230 - a few bytes, just before exit
                f.write(str(s.down_s))
        log_event("admin_stop", box=cfg.box, down_s=s.down_s, restart=bool(restart_file))
        server.should_exit = True
        return {"box": cfg.box, "stopping": True, "down_s": s.down_s, "will_restart": bool(restart_file)}

    return app


def supervise(argv: list[str]) -> int:
    """Run the server as a child and start it again after an /admin/stop. Ends when the child ends on its own."""
    fd, restart_file = tempfile.mkstemp(prefix="wx-restart-")
    os.close(fd)
    os.unlink(restart_file)
    child: subprocess.Popen | None = None

    def forward(signum, _frame):
        if child and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    args = [a for a in argv if a != "--supervise"]
    while True:
        child = subprocess.Popen([sys.executable, "-m", "inference.server", *args, "--restart-file", restart_file])
        code = child.wait()
        if not os.path.exists(restart_file):
            return code
        with open(restart_file) as f:
            down_s = float(f.read().strip() or 0)
        os.unlink(restart_file)
        log_event("supervisor_restart", down_s=down_s, exit_code=code)
        time.sleep(down_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="alias for --box, to match mock_server.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--metrics-file", help="write the JSON metrics snapshot here every --metrics-interval s")
    ap.add_argument("--metrics-interval", type=float, default=2.0)
    ap.add_argument("--admin", action="store_true", help="enable the /admin test hooks")
    ap.add_argument("--supervise", action="store_true", help="restart the server after an /admin/stop")
    ap.add_argument("--restart-file", help=argparse.SUPPRESS)
    ap.add_argument("--model", choices=["tiny", "afno"], default="tiny",
                    help="tiny: the seeded stand-in; afno: the weather team's trained model (needs --weights, --data)")
    ap.add_argument("--weights", help="afno: the model file written by weather/train.py")
    ap.add_argument("--data", help="afno: the bundle folder (stats.json, states/)")
    ap.add_argument("--log-level", default="INFO")
    add_engine_args(ap)
    a = ap.parse_args()
    setup_logging(a.log_level)
    if a.supervise:
        sys.exit(supervise(sys.argv[1:]))
    if a.name:
        a.box = a.name
    cfg: EngineConfig = engine_config_from_args(a)
    log_event("server_start", box=cfg.box, host=a.host, port=a.port, admin=a.admin)
    engine, service = InferenceEngine(cfg), None
    if a.model == "afno":
        from .afno import AfnoBackend, AfnoService

        if not (a.weights and a.data):
            ap.error("--model afno needs --weights and --data")
        if cfg.max_batch_rows == EngineConfig.max_batch_rows:
            cfg.max_batch_rows = 20  # one row is a whole global state; the team's server also uses 20
        engine.backend = AfnoBackend(a.weights, a.data, "cuda" if cfg.require_gpu else cfg.device)
        if cfg.require_gpu and engine.backend.device != "cuda":
            raise SystemExit("CUDA GPU required but the AFNO model is on " + engine.backend.device)
        service = AfnoService(engine.backend)
    app = create_app(engine, a.metrics_file, a.metrics_interval, a.admin, a.restart_file, service)
    server = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning"))
    app.state.server = server
    server.run()


if __name__ == "__main__":
    main()
