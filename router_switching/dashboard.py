"""Adds the inference and switching panels to the team's existing Control Room (ui/ or ui2/).

The page, its design and its controls stay as they are. `attach(app, ops)` adds:

  GET /api/switching/live      what each box and the router report right now
  GET /api/switching/results   router_switching/results/summary.json (built by collect.py)
  /switching/static/           switching.js and switching.css, which add two cards to the page

Everything in /live comes from endpoints that already exist: the router's /router/status and
/router/metrics, and each box's /health, /stats and (when it has one) /v1/metrics or /v1/info.
A value a box does not report is null and the page shows a dash. Nothing is estimated.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "results" / "summary.json"
REAL, CPU, SIM, UNKNOWN = "real GPU", "CPU", "simulated", "unknown"


async def _get(client: httpx.AsyncClient, url: str):
    try:
        r = await client.get(url, timeout=1.5)
        return r.json() if r.status_code == 200 else None
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
         "gpu_util_source": None}
    if metrics and str(metrics.get("schema", "")).startswith("wxrouter.inference.metrics"):
        dev = metrics.get("device") or {}
        v.update(kind=REAL if dev.get("type") == "cuda" else CPU, device=dev.get("name"),
                 gpu_uuid=(metrics.get("node") or {}).get("gpu_uuid"),
                 compute_p50_ms=metrics["compute_ms"]["p50"], compute_p95_ms=metrics["compute_ms"]["p95"],
                 mean_batch_rows=metrics["batches"]["mean_rows"], max_batch_rows=metrics["config"]["max_batch_rows"],
                 requests_per_s=metrics["throughput"]["requests_per_s"], rows_per_s=metrics["throughput"]["rows_per_s"],
                 gpu_memory_mb=(metrics.get("gpu_memory") or {}).get("allocated_mb"),
                 error_rate=metrics["errors"]["error_rate"],
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
        h, s, m = await asyncio.gather(_get(client, f"{url}/health"), _get(client, f"{url}/stats"),
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


def attach(app: FastAPI, ops) -> None:
    """`ops` is the Control Room's Ops object: it has .router, .boxes and (once started) .client."""
    @app.get("/api/switching/live")
    async def switching_live():
        return await live(ops.client, ops.router, ops.boxes)

    @app.get("/api/switching/results")
    async def switching_results():
        return results()

    app.mount("/switching/static", StaticFiles(directory=HERE / "static"), name="switching-static")
