"""Benchmark the inference engine and write machine-readable JSON.

  python -m inference.bench                 # full run, results/inference/<timestamp>/
  python -m inference.bench --quick         # a few seconds; used by the Fuzzball smoke test
  python -m inference.bench --require-gpu   # exit 2 instead of falling back to CPU

What it measures:
  cold_start   import, model load, first inference, then warm inference
  batch_sweep  one direct GPU call per batch size: latency and rows/s
  serving      the async engine under N concurrent clients, with batching off
               (baseline, max_batch_rows=1) and on (optimized), same requests
  precision    fp32 vs fp16 autocast (CUDA only)

Nothing here is simulated. Every number is a wall-clock measurement on the
device named in `environment`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import EngineConfig, add_engine_args, engine_config_from_args
from .engine import InferenceEngine, InferenceError, InferenceTimeout, Overloaded
from .forecast import build_rows, issue_time
from .gpu import query_smi
from .metrics import latency_summary, log_event, setup_logging
from .model import MODEL_VERSION

BENCH_SCHEMA = "wxrouter.inference.bench/v1"
BEGIN, END = "===WXROUTER_BENCH_JSON_BEGIN===", "===WXROUTER_BENCH_JSON_END==="


def rows_for(n: int, seed: int = 0) -> np.ndarray:
    """n single-member requests at seeded random locations."""
    rng = np.random.default_rng(seed)
    issued = issue_time()
    return np.concatenate([build_rows(float(rng.uniform(-60, 60)), float(rng.uniform(-180, 180)), 1, issued)
                           for _ in range(n)])


def environment(engine: InferenceEngine, import_ms: float | None) -> dict:
    b = engine.backend
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
           "backend": b.name, "device": b.device, "device_name": b.device_name, "precision": b.precision,
           "model_version": MODEL_VERSION, "hidden": engine.cfg.hidden, "layers": engine.cfg.layers,
           "numpy": np.__version__, "torch_import_ms": import_ms, "nvidia_smi": query_smi() or None}
    if b.name == "torch":
        env["torch"] = b.torch.__version__
        env["cuda_available"] = b.torch.cuda.is_available()
        env["torch_threads"] = b.torch.get_num_threads()
    return env


def time_import(backend: str) -> float | None:
    if backend == "numpy":
        return None
    t0 = time.perf_counter()
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    return round((time.perf_counter() - t0) * 1000, 1)


def batch_sweep(engine: InferenceEngine, sizes: list[int], hours: int, iters: int) -> list[dict]:
    out = []
    for n in sizes:
        x = rows_for(n)
        engine.infer_sync(x, hours)  # settle allocator for this shape
        engine.backend.reset_peak_memory()
        ms = []
        for _ in range(iters):
            t = time.perf_counter()
            engine.infer_sync(x, hours)
            ms.append((time.perf_counter() - t) * 1000)
        mean = sum(ms) / len(ms)
        out.append({"batch_rows": n, "hours_ahead": hours, "iterations": iters, "latency_ms": latency_summary(ms),
                    "rows_per_s": round(n / mean * 1000, 1), "ms_per_row": round(mean / n, 4),
                    "gpu_memory": engine.backend.memory()})
        log_event("bench_batch", **{k: v for k, v in out[-1].items() if k != "gpu_memory"})
    return out


async def serve_once(cfg: EngineConfig, backend, clients: int, requests: int, hours: int, label: str) -> dict:
    """`clients` closed-loop clients send `requests` single-member forecasts through the engine."""
    engine = InferenceEngine(cfg, backend=backend)
    engine.ready = True  # backend is already warm
    await engine.start()
    xs = rows_for(requests, seed=1)
    lat_ms, errors = [], 0
    todo = iter(range(requests))

    async def client() -> None:
        nonlocal errors
        for i in todo:
            t = time.perf_counter()
            try:
                await engine.infer(xs[i:i + 1], hours)
                lat_ms.append((time.perf_counter() - t) * 1000)
            except (Overloaded, InferenceTimeout, InferenceError):
                errors += 1

    backend.reset_peak_memory()
    t0 = time.perf_counter()
    await asyncio.gather(*(client() for _ in range(clients)))
    wall = time.perf_counter() - t0
    snap = engine.snapshot()
    await engine.stop()
    row = {"mode": label, "clients": clients, "requests": requests, "hours_ahead": hours,
           "max_batch_rows": cfg.max_batch_rows, "engine_concurrency": cfg.concurrency,
           "latency_ms": latency_summary(lat_ms), "requests_per_s": round(len(lat_ms) / wall, 1),
           "errors": errors, "error_rate": round(errors / requests, 5),
           "mean_batch_rows": snap["batches"]["mean_rows"], "gpu_memory": snap["gpu_memory"]}
    log_event("bench_serving", **{k: v for k, v in row.items() if k != "gpu_memory"})
    return row


async def serving_sweep(cfg: EngineConfig, backend, clients: list[int], requests: int, hours: int) -> list[dict]:
    out = []
    for c in clients:
        out.append(await serve_once(replace(cfg, max_batch_rows=1), backend, c, requests, hours, "baseline"))
        out.append(await serve_once(cfg, backend, c, requests, hours, "batched"))
    return out


FP16_TOLERANCE = 0.05  # model outputs are in [-1, 1]


def precision_compare(cfg: EngineConfig, rows: int, hours: int, iters: int) -> list[dict]:
    """fp32 vs fp16 autocast on the same inputs: speed, and how far the fp16 answers drift."""
    out, ref = [], None
    x = rows_for(rows)
    for amp in (False, True):
        eng = InferenceEngine(replace(cfg, amp=amp))
        eng.load()
        y = eng.infer_sync(x, hours)
        r = batch_sweep(eng, [rows], hours, iters)[0]
        row = {"precision": eng.backend.precision, **r, "finite": bool(np.isfinite(y).all())}
        if ref is None:
            ref = y
        else:
            diff = float(np.abs(y - ref).max())
            row["max_abs_diff_vs_fp32"] = round(diff, 5)
            row["numerically_valid"] = row["finite"] and diff <= FP16_TOLERANCE
        out.append(row)
    return out


def markdown(res: dict) -> str:
    env, cold = res["environment"], res["cold_start"]
    lines = [f"# Inference benchmark, {res['created_at']}", "",
             (f"Device: **{env['device_name']}** ({env['backend']}, {env['precision']}), "
             f"model `{env['model_version']}` hidden={env['hidden']} layers={env['layers']}"), "",
             "## Cold start", "", "| torch import ms | model load ms | first inference ms | warm inference ms |",
             "|---:|---:|---:|---:|",
             f"| {env['torch_import_ms']} | {cold['load_ms']} | {cold['first_inference_ms']} | {cold['warm_inference_ms']} |",
             "", f"## One GPU call per batch ({res['params']['hours_ahead']} h forecast)", "",
             "| batch rows | p50 ms | p95 ms | p99 ms | rows/s | ms/row | peak GPU MB |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["batch_sweep"]:
        l = r["latency_ms"]
        lines.append(f"| {r['batch_rows']} | {l['p50']:.2f} | {l['p95']:.2f} | {l['p99']:.2f} | {r['rows_per_s']} | "
                     f"{r['ms_per_row']} | {r['gpu_memory'].get('peak_allocated_mb', '')} |")
    lines += ["", "## Serving: baseline (no batching) vs batched", "",
              "| clients | mode | p50 ms | p95 ms | p99 ms | req/s | mean batch | errors |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in res["serving"]:
        l = r["latency_ms"]
        lines.append(f"| {r['clients']} | {r['mode']} | {l['p50']:.2f} | {l['p95']:.2f} | {l['p99']:.2f} | "
                     f"{r['requests_per_s']} | {r['mean_batch_rows']} | {r['errors']} |")
    if res.get("precision"):
        lines += ["", "## Precision", "", "| precision | batch rows | p50 ms | rows/s | peak GPU MB | max diff vs fp32 | valid |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        lines += [f"| {r['precision']} | {r['batch_rows']} | {r['latency_ms']['p50']:.2f} | {r['rows_per_s']} | "
                  f"{r['gpu_memory'].get('peak_allocated_mb', '')} | {r.get('max_abs_diff_vs_fp32', '')} | "
                  f"{r.get('numerically_valid', '')} |" for r in res["precision"]]
    return "\n".join(lines) + "\n"


def ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


def run(a: argparse.Namespace) -> dict:
    import_ms = time_import(a.backend)
    cfg = engine_config_from_args(a)
    engine = InferenceEngine(cfg)
    engine.load()
    if cfg.require_gpu and engine.backend.device != "cuda":
        raise RuntimeError(f"CUDA GPU required but the model is on {engine.backend.device}")
    x1 = rows_for(1)
    warm = []
    for _ in range(20):
        t = time.perf_counter()
        engine.infer_sync(x1, 12)
        warm.append((time.perf_counter() - t) * 1000)
    cold = {**engine.cold_start, "warm_inference_ms": round(sorted(warm)[len(warm) // 2], 3),
            "note": "first_inference_ms and warm_inference_ms are one 12 h, 1-row forecast"}
    res = {
        "schema": BENCH_SCHEMA,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment(engine, import_ms),
        "params": {"hours_ahead": a.hours, "batch_sizes": a.batch_sizes, "clients": a.clients,
                   "requests": a.requests, "iterations": a.iterations, "quick": a.quick},
        "cold_start": cold,
        "batch_sweep": batch_sweep(engine, a.batch_sizes, a.hours, a.iterations),
        "serving": asyncio.run(serving_sweep(cfg, engine.backend, a.clients, a.requests, a.hours)),
        "precision": (precision_compare(cfg, max(a.batch_sizes), a.hours, a.iterations)
                      if engine.backend.device == "cuda" else []),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/inference", help="a <timestamp>/ folder is created inside")
    ap.add_argument("--hours", type=int, default=48, help="forecast length used for every request")
    ap.add_argument("--batch-sizes", type=ints, default=[1, 4, 16, 64, 256])
    ap.add_argument("--clients", type=ints, default=[1, 8, 32], help="concurrent client counts")
    ap.add_argument("--requests", type=int, default=400, help="requests per serving run")
    ap.add_argument("--iterations", type=int, default=30, help="GPU calls per batch size")
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--emit-markers", action="store_true",
                    help="also print the JSON on stdout between marker lines, for scraping job logs")
    ap.add_argument("--log-level", default="INFO")
    add_engine_args(ap)
    a = ap.parse_args()
    if a.quick:
        a.batch_sizes, a.clients, a.requests, a.iterations = [1, 16, 64], [1, 16], 64, 5
    setup_logging(a.log_level)
    try:
        res = run(a)
    except RuntimeError as e:
        if "GPU required" not in str(e):
            raise
        log_event("bench_no_gpu", error=str(e))
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(2)
    out = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=2))
    (out / "table.md").write_text(markdown(res))
    print(markdown(res))
    print(f"wrote {out}/bench.json and table.md")
    if a.emit_markers:
        print(BEGIN)
        print(json.dumps(res))
        print(END)


if __name__ == "__main__":
    main()
