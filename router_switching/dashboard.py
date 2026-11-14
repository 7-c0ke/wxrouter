"""Adds the inference and switching panels to the team's existing Control Room (ui/).

The page, its design and its controls stay as they are. `attach(app, ops)` adds:

  GET /api/switching/live      what each box and the router report right now
  GET /api/switching/results   router_switching/results/summary.json (built by collect.py)
  POST/GET /api/switching/demo the "Run Switching Demo" button (demo.py): start it, read its progress
  /switching/static/           switching.js and switching.css, which add two cards to the page

Everything in /live comes from endpoints that already exist: the router's /router/status and
/router/metrics, and each box's /health, /stats and (when it has one) /v1/metrics or /v1/info.
A value a box does not report is null and the page shows a dash. Nothing is estimated.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "results" / "summary.json"
REAL, CPU, SIM, UNKNOWN = "real GPU", "CPU", "simulated", "unknown"


async def _get(client: httpx.AsyncClient, url: str, any_status: bool = False):
    try:
        r = await client.get(url, timeout=1.5)
        return r.json() if (r.status_code == 200 or any_status) else None
    except (httpx.HTTPError, ValueError):
        return None


def box_view(health: dict | None, stats: dict | None, metrics: dict | None, info: dict | None) -> dict:
    """One box, from whatever it answered. `kind` says how far its numbers can be trusted as GPU numbers."""
    v = {"reachable": stats is not None or health is not None, "kind": UNKNOWN, "device": None, "gpu_uuid": None,
         "status": (health or {}).get("status"), "model_version": (health or stats or {}).get("model_version"),
         "mode": (stats or {}).get("mode"), "gpu_util": (stats or {}).get("gpu_util"),
         "in_flight": (stats or {}).get("in_flight"), "queue_depth": (stats or {}).get("queue_depth"),
         "latency_p50_ms": (stats or {}).get("p50_latency_ms"), "latency_p95_ms": (stats or {}).get("p95_latency_ms"),
         "compute_p50_ms": None, "compute_p95_ms": None, "mean_batch_rows": None, "max_batch_rows": None,
         "requests_per_s": None, "rows_per_s": None, "gpu_memory_mb": None, "error_rate": None,
         "gpu_util_source": None, "latency_mean_ms": None, "requests_total": (stats or {}).get("requests_total"),
         "errors_total": (stats or {}).get("errors_total"), "timeouts": None, "gpu_memory_peak_mb": None,
         "gpu_memory_device_mb": None, "trained_model": None}
    if metrics and str(metrics.get("schema", "")).startswith("wxrouter.inference.metrics"):
        dev = metrics.get("device") or {}
        v.update(kind=REAL if dev.get("type") == "cuda" else CPU, device=dev.get("name"),
                 gpu_uuid=(metrics.get("node") or {}).get("gpu_uuid"),
                 compute_p50_ms=metrics["compute_ms"]["p50"], compute_p95_ms=metrics["compute_ms"]["p95"],
                 mean_batch_rows=metrics["batches"]["mean_rows"], max_batch_rows=metrics["config"]["max_batch_rows"],
                 requests_per_s=metrics["throughput"]["requests_per_s"], rows_per_s=metrics["throughput"]["rows_per_s"],
                 gpu_memory_mb=(metrics.get("gpu_memory") or {}).get("allocated_mb"),
                 error_rate=metrics["errors"]["error_rate"], latency_mean_ms=(metrics.get("latency_ms") or {}).get("mean"),
                 timeouts=metrics["errors"].get("timeouts"),
                 gpu_memory_peak_mb=(metrics.get("gpu_memory") or {}).get("peak_allocated_mb"),
                 gpu_memory_device_mb=(metrics.get("gpu_memory") or {}).get("device_used_mb"),
                 trained_model="untrained" not in str(metrics.get("model_version", "")),
                 # without nvidia-smi the server reports busy slots, which is not a GPU reading
                 gpu_util_source="nvidia-smi" if dev.get("type") == "cuda" else "busy slots (no GPU)")
    elif info and "device" in info:  # the weather team's server (serve/forecast_server.py)
        mock = bool(info.get("mock"))
        cuda = str(info.get("device", "")).startswith("cuda")
        v.update(kind=SIM if mock else (REAL if cuda else CPU), device=info.get("model"),
                 gpu_util_source=None if mock else "nvidia-smi")
        if mock:
            v["gpu_util"] = None
    elif "mock" in str(v["model_version"] or "").lower() or (stats or {}).get("mock"):
        v.update(kind=SIM, device="mock server", gpu_util_source="simulated")
    if not v["reachable"]:
        v["kind"] = UNKNOWN
    return v


def router_view(status: dict, metrics: dict | None) -> dict:
    ad = status["adaptive"]
    now = (metrics or {}).get("now")
    switches = ad.get("switches", [])
    last = dict(switches[-1]) if switches else None
    if last and now is not None:
        last["age_s"] = round(now - last["t"], 1)
    per_box = (metrics or {}).get("per_box", {})
    total = sum(b.get("requests", 0) for b in per_box.values())
    return {"policy": status["policy"], "active": ad.get("active"), "preferred": ad.get("preferred"),
            "thresholds": {k: ad.get(k) for k in ("margin", "window_s", "cooldown_s", "return_margin")},
            "failovers_total": status.get("failovers"), "last_switch": last, "switches": switches[-8:],
            "window_s": (metrics or {}).get("window_s"),
            "overall": (metrics or {}).get("overall"),
            "share_pct": {n: (round(100 * b.get("requests", 0) / total, 1) if total else None)
                          for n, b in per_box.items()},
            "backends": {n: {"healthy": b["healthy"], "probe_ms": b.get("probe_ms"), "in_flight": b.get("in_flight"),
                             "err_rate": b.get("err_rate"),
                             "health_score": (b.get("health_score") or {}).get("score")
                             if isinstance(b.get("health_score"), dict) else b.get("health_score")}
                         for n, b in status["backends"].items()}}


async def live(client: httpx.AsyncClient, router: str, boxes: dict[str, str]) -> dict:
    async def one(name: str, url: str):
        h, s, m = await asyncio.gather(_get(client, f"{url}/health", any_status=True),  # a box that is down still says so, with 503
                                       _get(client, f"{url}/stats"),
                                       _get(client, f"{url}/v1/metrics"))
        info = None if m else await _get(client, f"{url}/v1/info")
        return name, box_view(h, s, m, info)

    st, mt, bx = await asyncio.gather(_get(client, f"{router}/router/status"),
                                      _get(client, f"{router}/router/metrics?window=10"),
                                      asyncio.gather(*(one(n, u) for n, u in boxes.items())))
    return {"ts": time.time(), "router": router_view(st, mt) if st else None, "boxes": dict(bx)}


def results() -> dict:
    if not SUMMARY.exists():
        return {"runs": [], "not_measured_on_gpu": [], "cpu_vs_gpu": None,
                "note": "No summary yet. Run: python -m router_switching.collect"}
    return json.loads(SUMMARY.read_text())


BURST_BODY = {"lat": 41.98, "lon": -87.9, "hours_ahead": 48}


async def burst(client: httpx.AsyncClient, router: str, n: int, concurrency: int) -> dict:
    """Send n real forecasts through the router, `concurrency` at a time, and say who served them."""
    sem, rows = asyncio.Semaphore(concurrency), []

    async def one(i: int) -> None:
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.post(f"{router}/v1/forecast", json={**BURST_BODY, "request_id": f"burst-{i}"}, timeout=30)
                ok = r.status_code == 200
                rows.append({"ms": (time.perf_counter() - t) * 1000, "ok": ok, "box": r.headers.get("x-served-by", "?"),
                             "failover": r.headers.get("x-router-failover") == "true",
                             "gpu_ms": r.json()["timing"]["compute_ms"] if ok else None})
            except (httpx.HTTPError, ValueError, KeyError):
                rows.append({"ms": (time.perf_counter() - t) * 1000, "ok": False, "box": "none", "failover": False,
                             "gpu_ms": None, "unreachable": True})

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0
    ok = sorted(r["ms"] for r in rows if r["ok"])
    gpu = sorted(r["gpu_ms"] for r in rows if r["gpu_ms"] is not None)

    def q(v, f):
        return round(v[min(len(v) - 1, int(f * len(v)))], 1) if v else None

    by = {}
    for r in rows:
        if r["ok"]:
            by[r["box"]] = by.get(r["box"], 0) + 1
    return {"requests": n, "concurrency": concurrency, "ok": len(ok), "errors": n - len(ok), "wall_s": round(wall, 2),
            "requests_per_s": round(len(ok) / wall, 1) if wall else None, "served_by": by,
            "failovers": sum(r["failover"] for r in rows),
            # no answer at all (connection refused or dropped on the way to the router) vs an error the router sent
            "never_reached_router": sum(1 for r in rows if r.get("unreachable")),
            "client_ms": {"mean": round(sum(ok) / len(ok), 1) if ok else None, "p50": q(ok, 0.5), "p95": q(ok, 0.95)},
            "gpu_call_ms": {"p50": q(gpu, 0.5), "p95": q(gpu, 0.95)},
            "note": "client latency includes the path from this machine to the router"}


def attach(app: FastAPI, ops) -> None:
    """`ops` is the Control Room's Ops object: it has .router, .boxes and (once started) .client."""
    @app.get("/api/switching/live")
    async def switching_live():
        return await live(ops.client, ops.router, ops.boxes)

    @app.post("/api/switching/burst")
    async def switching_burst(n: int = 200, concurrency: int = 32):
        return await burst(ops.client, ops.router, max(1, min(n, 1000)), max(1, min(concurrency, 100)))

    @app.get("/api/switching/results")
    async def switching_results():
        return results()

    from .demo import SwitchingDemo   # here, not at the top: demo.py imports this module
    demo = SwitchingDemo(ops)
    page_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app_):
        """The page's own lifespan, with one addition: a demo in progress is stopped and box A and the
        router policy are put back before the page closes its HTTP client."""
        async with page_lifespan(app_) as state:
            try:
                yield state
            finally:
                await demo.shutdown()
    app.router.lifespan_context = lifespan

    @app.post("/api/switching/demo")
    async def switching_demo_start():
        """Start the scripted fault run (one at a time). The page polls the GET for progress."""
        return demo.start()

    @app.get("/api/switching/demo")
    async def switching_demo_state():
        return demo.state

    app.mount("/switching/static", StaticFiles(directory=HERE / "static"), name="switching-static")
