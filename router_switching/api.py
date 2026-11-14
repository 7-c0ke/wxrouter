"""A JSON API over the live GPU and routing data, for any front end to connect to. No page is changed.

    GET  /api/v1                 the list of endpoints and where each number comes from
    GET  /api/v1/status          everything below in one answer
    GET  /api/v1/gpus            one object per box: state, CUDA, model, GPU readings, inference timing, counters
    GET  /api/v1/gpus/{box}      one box
    GET  /api/v1/routing         the rule, where requests went, what the router sees of each box, thresholds, switches
    GET  /api/v1/performance     the router's request rate, latency and errors over ?window= seconds (default 10)
    GET  /api/v1/fuzzball        the workflow's status, stages and nodes, from the fuzzball CLI
    POST /api/v1/inference       one real forecast through the router (or to one box): the model's own answer

Rules of this API:
  * Every value is read at request time from the router, the boxes or the fuzzball CLI. Nothing is stored or made up.
  * A value nobody reported is null. A box that does not answer has "state": "OFFLINE" and null values.
  * "kind" says what a box is: "real GPU", "CPU", "simulated" (a mock server) or "unknown". A mock's GPU figure is null.
  * GET answers carry Access-Control-Allow-Origin: *, so a page served from elsewhere can read them.

The workflow for /fuzzball and the node names is the WX_WORKFLOW environment variable, or ?workflow=ID.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .dashboard import CPU, REAL, SIM, _get, box_view, router_view

LAST_OK: dict[str, float] = {}   # box URL or "router" -> wall-clock time of its last answer
FB_CACHE: dict = {}


def _seen(key: str, answered: bool) -> dict:
    now = time.time()
    if answered:
        LAST_OK[key] = now
    last = LAST_OK.get(key)
    return {"last_ok_ts": last, "age_s": round(now - last, 1) if last else None}


def gpu_object(name: str, health, stats, metrics, info, machine, seen: dict) -> dict:
    """One box in the API's shape, from whatever it answered. Unknown is None, never a default."""
    v = box_view(health, stats, metrics, info)
    kind, online = v["kind"], stats is not None
    gpu = (machine or {}).get("gpu") or {}
    util, source = None, None
    if kind == REAL and metrics:   # the inference server: only an nvidia-smi reading counts
        src = (metrics.get("load") or {}).get("gpu_util_source")
        if src != "busy_slots":
            util, source = v["gpu_util"], "nvidia-smi" if src else "nvidia-smi (this server version does not confirm the source)"
    elif kind == REAL and info:    # the weather server: its /stats says 0.0 when it has no reading; /metrics leaves it null
        if gpu.get("util_pct") is not None:
            util, source = gpu["util_pct"] / 100.0, "nvidia-smi, via the server's /metrics"
    dev = (metrics.get("device") or {}).get("type") if metrics else (info or {}).get("device")
    nz = lambda x: None if x in (0, 0.0, None) else x   # servers fill an empty latency window with 0.0
    counted = bool(metrics and metrics["compute_ms"].get("count") != 0)
    return {
        "name": name, "state": "OFFLINE" if not online else "DOWN" if v["status"] not in (None, "ok") else "ONLINE",
        "kind": kind, "cuda": None if dev is None else str(dev).startswith("cuda"), "device": v["device"], "gpu_uuid": v["gpu_uuid"],
        "mode": v["mode"],
        "model": {"version": v["model_version"], "name": (info or {}).get("model"), "params": (info or {}).get("params"),
                  "trained": None if not v["model_version"] else "untrained" not in v["model_version"] and kind != SIM,
                  "starts_from": (info or {}).get("latest")},
        "gpu": {"utilization": None if util is None else round(util, 3), "utilization_source": source,
                "memory_process_mb": gpu.get("proc_mem_mb"), "memory_tensors_mb": v["gpu_memory_mb"],
                "memory_tensors_peak_mb": v["gpu_memory_peak_mb"], "memory_total_mb": v["gpu_memory_device_mb"],
                "temperature_c": gpu.get("temp_c"), "power_w": gpu.get("power_w")},
        "inference": {"gpu_call_ms": {"p50": v["compute_p50_ms"] if counted else None, "p95": v["compute_p95_ms"] if counted else None,
                                      "p99": metrics["compute_ms"].get("p99") if counted else None},
                      "request_ms": {"p50": nz(v["latency_p50_ms"]), "p95": nz(v["latency_p95_ms"])},
                      "ms_per_model_step": (machine or {}).get("ms_per_unit") or (info or {}).get("ms_per_unit")},
        "throughput": {"requests_per_s": v["requests_per_s"], "rows_per_s": v["rows_per_s"]},
        "batch": {"mean_rows": v["mean_batch_rows"], "max_rows": v["max_batch_rows"]},
        "requests": {"total": v["requests_total"], "errors": v["errors_total"], "includes_router_probes": True},
        "queue": {"depth": v["queue_depth"], "in_flight": v["in_flight"]},
        "telemetry": seen,
    }


async def read_gpus(client: httpx.AsyncClient, boxes: dict[str, str]) -> dict:
    async def one(name: str, url: str):
        h, s, m = await asyncio.gather(_get(client, f"{url}/health", any_status=True), _get(client, f"{url}/stats"),
                                       _get(client, f"{url}/v1/metrics"))
        info, machine = (None, None) if m else await asyncio.gather(_get(client, f"{url}/v1/info"), _get(client, f"{url}/metrics"))
        return name, gpu_object(name, h, s, m, info, machine, _seen(url, s is not None))
    return dict(await asyncio.gather(*(one(n, u) for n, u in boxes.items())))


async def read_routing(client: httpx.AsyncClient, router: str) -> dict:
    st, mt = await asyncio.gather(_get(client, f"{router}/router/status"), _get(client, f"{router}/router/metrics?window=10"))
    seen = _seen("router", st is not None)
    if not st:
        return {"state": "OFFLINE", "telemetry": seen}
    r = router_view(st, mt)
    cfg = st.get("config") or {}
    per_box = (mt or {}).get("per_box", {})
    return {"state": "ONLINE", "telemetry": seen, "policy": r["policy"], "startup_policy": (cfg.get("defaults") or {}).get("policy"),
            "preferred": r["preferred"], "drained": cfg.get("drained"),
            # under any rule but adaptive the router still tracks what adaptive would pick; it routes nothing then
            "adaptive": {"routing": r["policy"] == "adaptive", "active": r["active"], "thresholds": r["thresholds"],
                         "switches": (st.get("adaptive") or {}).get("switches", [])},
            "failovers_total": r["failovers_total"], "window_s": r["window_s"],
            "boxes": {n: {**b, "share_pct": (r["share_pct"] or {}).get(n),
                          **{k: (per_box.get(n) or {}).get(k) for k in ("requests", "rps", "p50_ms", "p95_ms", "p99_ms", "errors")}}
                      for n, b in r["backends"].items()}}


async def read_performance(client: httpx.AsyncClient, router: str, window: float) -> dict:
    m = await _get(client, f"{router}/router/metrics?window={window:g}")
    if not m:
        return {"state": "OFFLINE", "window_s": window}
    return {"state": "ONLINE", "window_s": m.get("window_s"), **(m.get("overall") or {}),
            "source": "the router's own request log, GET /router/metrics"}


async def _cli(*args: str, timeout: float = 12.0) -> Optional[str]:
    fb = os.environ.get("FUZZBALL") or str(Path.home() / "bin" / "fuzzball")
    try:
        p = await asyncio.create_subprocess_exec(fb, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
        return out.decode(errors="replace") if p.returncode == 0 else None
    except (OSError, asyncio.TimeoutError):
        return None


def parse_describe(text: str) -> dict:
    """Name, status and the stage table of `fuzzball workflow describe`."""
    out = {"name": None, "status": None, "created": None, "stages": []}
    for line in text.splitlines():
        if line.startswith("Name:"):
            out["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Status:"):
            out["status"] = line.split(":", 1)[1].strip().replace("STAGE_STATUS_", "")
        elif line.startswith("Created:"):
            out["created"] = line.split(":", 1)[1].strip()
        else:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 4 and cols[0] in ("Service", "Job"):
                out["stages"].append({"kind": cols[0], "status": cols[1].upper(), "name": cols[2], "running_for": cols[3] or None})
    return out


def parse_placement(events: str) -> dict:
    """stage -> {node, gpus} from the scheduler's own "Resources allocated" events."""
    out = {}
    for line in events.splitlines():
        m = re.search(r"\] ([\w.-]+): Resources allocated \(node: ([\d.]+)[^,]*,(.*)", line)
        if m:
            g = re.search(r"requested devices: (\d+)", m.group(3))
            out[m.group(1)] = {"node": m.group(2), "gpus": int(g.group(1)) if g else 0}
    return out


async def read_fuzzball(workflow: Optional[str]) -> dict:
    """What Fuzzball says about the workflow, from its CLI. Kept for 10 s per workflow."""
    if not workflow:
        return {"available": False, "reason": "no workflow ID: set WX_WORKFLOW for the server or pass ?workflow=ID"}
    if not re.fullmatch(r"[0-9a-fA-F-]{8,40}", workflow):
        raise HTTPException(422, "workflow must be a Fuzzball workflow ID")
    hit = FB_CACHE.get(workflow)
    if hit and time.monotonic() - hit[0] < 10:
        return hit[1]
    desc, events, nodes = await asyncio.gather(_cli("workflow", "describe", workflow), _cli("workflow", "events", workflow),
                                               _cli("node", "list"))
    if desc is None:
        v = {"available": False, "id": workflow, "reason": "the fuzzball CLI did not answer (not logged in, unknown ID, or off the cluster network)"}
    else:
        host = {}
        for line in (nodes or "").splitlines():
            c = [x.strip() for x in line.split("|")]
            if len(c) >= 3 and "/" in c[0]:
                host[c[0].split("/")[0]] = c[2]
        place = parse_placement(events or "")
        v = {"available": True, "id": workflow, **parse_describe(desc), "checked_ts": time.time(), "source": "fuzzball CLI"}
        for s in v["stages"]:
            p = place.get(s["name"])
            # a stage that is not running holds no node, whatever an earlier event said
            s.update(node=host.get(p["node"], p["node"]) if p and s["status"] == "STARTED" else None, gpus=p["gpus"] if p else None)
    FB_CACHE[workflow] = (time.monotonic(), v)
    return v


def place_on_nodes(gpus: dict, fb: dict) -> None:
    """Add each box's node and Fuzzball stage status; a box Fuzzball has not started is PENDING (its word), not OFFLINE."""
    stages = {s["name"]: s for s in fb.get("stages", [])} if fb.get("available") else {}
    for name, g in gpus.items():
        s = stages.get(name)
        g["node"], g["workflow_stage"] = (s or {}).get("node"), (s or {}).get("status")
        if s and s["status"] != "STARTED" and g["state"] == "OFFLINE":
            g["state"] = s["status"]


class InferenceIn(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    hours_ahead: int = Field(48, ge=1, le=120)
    step_hours: int = Field(6, ge=1, le=24)
    ensemble_members: int = Field(1, ge=1, le=20)
    box: Optional[str] = Field(None, description="send straight to this box instead of through the router")


INDEX = {
    "endpoints": {
        "GET /api/v1/status": "gpus, routing, performance and fuzzball in one answer",
        "GET /api/v1/gpus": "every box; /api/v1/gpus/{box} for one",
        "GET /api/v1/routing": "rule, traffic share, what the router sees of each box, thresholds, switches",
        "GET /api/v1/performance?window=10": "router request rate, latency percentiles, errors",
        "GET /api/v1/fuzzball?workflow=ID": "workflow status, stages, nodes",
        "POST /api/v1/inference": "one real forecast; body {lat, lon, hours_ahead, step_hours, ensemble_members, box?}",
    },
    "sources": {
        "gpus": "each box's GET /health, /stats, and /v1/metrics (inference server) or /v1/info + /metrics (weather server)",
        "routing, performance": "the router's GET /router/status and /router/metrics",
        "fuzzball, node names": "fuzzball workflow describe / events, fuzzball node list",
    },
    "rules": ["null means nobody reported it", "state OFFLINE means no telemetry from that source right now",
              "kind is real GPU, CPU, simulated or unknown; a simulated box never has a GPU utilization"],
}


def attach_api(app: FastAPI, ops) -> None:
    """`ops` is the Control Room's Ops object (.router, .boxes, .client)."""
    api = APIRouter(prefix="/api/v1")
    wf = lambda q: q or os.environ.get("WX_WORKFLOW")

    @api.get("")
    async def index():
        return {**INDEX, "boxes": list(ops.boxes), "workflow": os.environ.get("WX_WORKFLOW")}

    @api.get("/gpus")
    async def gpus(workflow: Optional[str] = None):
        g, fb = await asyncio.gather(read_gpus(ops.client, ops.boxes), read_fuzzball(wf(workflow)))
        place_on_nodes(g, fb)
        return {"ts": time.time(), "gpus": g}

    @api.get("/gpus/{box}")
    async def gpu(box: str, workflow: Optional[str] = None):
        if box not in ops.boxes:
            raise HTTPException(404, f"unknown box {box}; the boxes are: {', '.join(ops.boxes)}")
        g, fb = await asyncio.gather(read_gpus(ops.client, {box: ops.boxes[box]}), read_fuzzball(wf(workflow)))
        place_on_nodes(g, fb)
        return {"ts": time.time(), **g[box]}

    @api.get("/routing")
    async def routing():
        return {"ts": time.time(), **await read_routing(ops.client, ops.router)}

    @api.get("/performance")
    async def performance(window: float = 10.0):
        return {"ts": time.time(), **await read_performance(ops.client, ops.router, max(1.0, min(window, 300.0)))}

    @api.get("/fuzzball")
    async def fuzzball(workflow: Optional[str] = None):
        return await read_fuzzball(wf(workflow))

    @api.get("/status")
    async def status(workflow: Optional[str] = None):
        g, r, p, fb = await asyncio.gather(read_gpus(ops.client, ops.boxes), read_routing(ops.client, ops.router),
                                           read_performance(ops.client, ops.router, 10.0), read_fuzzball(wf(workflow)))
        place_on_nodes(g, fb)
        return {"ts": time.time(), "gpus": g, "routing": r, "performance": p, "fuzzball": fb}

    @api.post("/inference")
    async def inference(x: InferenceIn):
        """Send one forecast and return the server's own answer unchanged, with who served it and how long it took."""
        if x.box is not None and x.box not in ops.boxes:
            raise HTTPException(404, f"unknown box {x.box}; the boxes are: {', '.join(ops.boxes)}")
        url = ops.boxes[x.box] if x.box else ops.router
        body = x.model_dump(exclude={"box"})
        t0 = time.perf_counter()
        try:
            r = await ops.client.post(f"{url}/v1/forecast", json=body, timeout=60)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"{'box ' + x.box if x.box else 'router'} unreachable: {e.__class__.__name__}")
        ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            answer = r.json()
        except ValueError:
            answer = None
        if r.status_code != 200 or not isinstance(answer, dict):
            raise HTTPException(502, f"HTTP {r.status_code} from {'box ' + x.box if x.box else 'the router'}: {r.text[:200]}")
        return {"sent_to": x.box or "router", "served_by": r.headers.get("x-served-by") or answer.get("box"),
                "failover": r.headers.get("x-router-failover") == "true" if not x.box else None,
                "policy": r.headers.get("x-router-policy"), "round_trip_ms": ms, "request": body, "response": answer}

    app.include_router(api)

    @app.middleware("http")
    async def cors_for_api(request, call_next):
        if not request.url.path.startswith("/api/v1"):
            return await call_next(request)
        if request.method == "OPTIONS":   # a page on another origin asking before POST /inference
            return Response(status_code=204, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST",
                                                      "Access-Control-Allow-Headers": "content-type"})
        resp = await call_next(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
