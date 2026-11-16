"""A JSON API over the live GPU and routing data, for any front end to connect to. No page is changed.

    GET  /api/v1                 the list of endpoints and where each number comes from
    GET  /api/v1/status          everything below in one answer
    GET  /api/v1/gpus            one object per box: state, CUDA, model, GPU readings, inference timing, counters
    GET  /api/v1/gpus/{box}      one box
    GET  /api/v1/routing         the rule, where requests went, what the router sees of each box, thresholds, switches
    GET  /api/v1/performance     the router's request rate, latency and errors over ?window= seconds (default 10)
    GET  /api/v1/fuzzball        the workflow's status, stages and nodes, from the fuzzball CLI
    POST /api/v1/inference       one real forecast through the router (or to one box): the model's own answer

For a chat agent (an LLM with function calling, such as Nemotron behind llama-server or vLLM):

    GET  /api/v1/tools           the tools in OpenAI function-calling format: pass the list as `tools` to the LLM
    POST /api/v1/tools/{name}    run one tool; the body is the arguments object the LLM produced

    get_forecast, get_gpu_status, get_routing_status, get_performance      read only, open
    set_routing_policy, set_box_fault, run_switching_check                 change the cluster: need the key

Following the GPU stack (the link stays the same when the team redeploys the stack):

    GET  /api/v1/admin/target    which GPU stack this server talks to
    POST /api/v1/admin/target    {"stack": "<workflow id>"}: talk to that stack from now on (needs the key)
                                 router_switching/follow_stack.sh does this by itself

The key: set WX_API_KEY for the server and send it as `X-API-Key` (or `Authorization: Bearer`). With no key set,
the changing tools answer only to callers on this machine.

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
from fastapi import APIRouter, FastAPI, HTTPException, Request
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
        "GET /api/v1/tools": "the tools for a chat agent, in OpenAI function-calling format",
        "POST /api/v1/tools/{name}": "run one tool with the LLM's arguments object as the body",
    },
    "sources": {
        "gpus": "each box's GET /health, /stats, and /v1/metrics (inference server) or /v1/info + /metrics (weather server)",
        "routing, performance": "the router's GET /router/status and /router/metrics",
        "fuzzball, node names": "fuzzball workflow describe / events, fuzzball node list",
    },
    "rules": ["null means nobody reported it", "state OFFLINE means no telemetry from that source right now",
              "kind is real GPU, CPU, simulated or unknown; a simulated box never has a GPU utilization"],
}


def fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


BOX = {"type": "string", "description": "box name, e.g. box-a or box-b"}
TOOLS = [
    fn("get_forecast", "Run one real weather forecast on a GPU through the router and return the model's answer, which box "
       "(GPU) served it, whether the router had to fail over, and the timings. Set box to send it straight to one GPU.",
       {"lat": {"type": "number", "minimum": -90, "maximum": 90}, "lon": {"type": "number", "minimum": -180, "maximum": 180},
        "hours_ahead": {"type": "integer", "minimum": 1, "maximum": 120, "default": 48},
        "ensemble_members": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1}, "box": BOX}, ["lat", "lon"]),
    fn("get_gpu_status", "Live state of each GPU box: ONLINE or OFFLINE, node, model, GPU utilization, memory, latency, "
       "requests served. null means the box did not report that value.", {}, []),
    fn("get_routing_status", "The router's current rule, where requests went in the last 10 s, which boxes it sees as "
       "healthy, the adaptive switches it made and the failover count.", {}, []),
    fn("get_performance", "The router's request rate, p50/p95/p99 latency and errors over the last `window` seconds.",
       {"window": {"type": "number", "minimum": 1, "maximum": 300, "default": 10}}, []),
    fn("set_routing_policy", "Change the router's rule. CHANGES THE CLUSTER.",
       {"policy": {"type": "string", "enum": ["static", "round_robin", "least_loaded", "adaptive", "balanced"]},
        "preferred": {"type": "string", "description": "home box for static and adaptive; empty string clears it"}}, ["policy"]),
    fn("set_box_fault", "Make one box fail (it answers every forecast with HTTP 503) or put it back to normal, through the "
       "box's test hook. CHANGES THE CLUSTER: always set it back to normal afterwards.",
       {"box": BOX, "mode": {"type": "string", "enum": ["error", "normal"]}}, ["box", "mode"]),
    fn("run_switching_check", "Check inference switching end to end: send forecasts, make one box fail, send forecasts again "
       "(the other box must serve them), restore the box, wait until the router sees it healthy, send forecasts again. "
       "Returns who served every request in each step, errors and failovers. The box is always restored. CHANGES THE CLUSTER "
       "for a few seconds.",
       {"box": {**BOX, "description": "the box to fail; default: the first box"},
        "requests": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6, "description": "forecasts per step"}}, []),
]
WRITE_TOOLS = {"set_routing_policy", "set_box_fault", "run_switching_check"}


def allowed_to_change(request: Request) -> None:
    """The changing tools need WX_API_KEY when one is set; with none set, only this machine may call them."""
    key = os.environ.get("WX_API_KEY")
    sent = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if key:
        if sent != key:
            raise HTTPException(401, "this tool changes the cluster: send the key as X-API-Key")
    elif (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "this tool changes the cluster and the server has no WX_API_KEY set: only callers on its own machine may use it")


async def one_forecast(ops, body: dict, box: Optional[str] = None) -> dict:
    """One forecast, summarised for a step of the switching check. Never raises."""
    url = ops.boxes[box] if box else ops.router
    t0 = time.perf_counter()
    try:
        r = await ops.client.post(f"{url}/v1/forecast", json=body, timeout=60)
        ok = r.status_code == 200
        return {"ok": ok, "status": r.status_code, "served_by": r.headers.get("x-served-by") if ok else None,
                "failover": r.headers.get("x-router-failover") == "true", "ms": round((time.perf_counter() - t0) * 1000, 1)}
    except httpx.HTTPError as e:
        return {"ok": False, "status": None, "served_by": None, "failover": False, "ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": e.__class__.__name__}


def step_summary(name: str, rows: list[dict]) -> dict:
    by: dict[str, int] = {}
    for r in rows:
        if r["ok"]:
            by[r["served_by"] or "?"] = by.get(r["served_by"] or "?", 0) + 1
    return {"step": name, "requests": len(rows), "ok": sum(r["ok"] for r in rows), "errors": sum(not r["ok"] for r in rows),
            "served_by": by, "failovers": sum(r["failover"] for r in rows), "each": rows}


async def box_mode(ops, box: str, mode: str) -> None:
    try:
        r = await ops.client.post(f"{ops.boxes[box]}/admin/mode", json={"mode": mode}, timeout=10)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{box} did not answer its test hook ({e.__class__.__name__})")
    if r.status_code != 200:
        raise HTTPException(502, f"{box} refused mode {mode!r}: HTTP {r.status_code} {r.text[:150]} "
                                 "(the box must run with its test hooks on: --admin)")


async def switching_check(ops, box: Optional[str], n: int) -> dict:
    """Normal -> one box failing -> restored, with real forecasts in each step. The box is restored whatever happens."""
    names = list(ops.boxes)
    box = box or names[0]
    if box not in ops.boxes:
        raise HTTPException(404, f"unknown box {box}; the boxes are: {', '.join(names)}")
    if len(names) < 2:
        raise HTTPException(409, "a switching check needs two boxes")
    body = {"lat": 41.98, "lon": -87.9, "hours_ahead": 24, "step_hours": 6, "request_id": "switching-check"}
    send = lambda: asyncio.gather(*(one_forecast(ops, body) for _ in range(n)))
    before = await _get(ops.client, f"{ops.router}/router/status")
    if not before:
        raise HTTPException(502, "the router is unreachable: nothing was changed")
    steps, restored, healthy_after = [], False, None
    t_start = time.time()
    try:
        steps.append(step_summary("1 normal", await send()))
        await box_mode(ops, box, "error")
        t_fail = time.perf_counter()
        steps.append(step_summary(f"2 {box} failing (HTTP 503)", await send()))
        marked_down = None
        for _ in range(20):   # how long until the router itself calls the box unhealthy
            st = await _get(ops.client, f"{ops.router}/router/status")
            if st and not st["backends"][box]["healthy"]:
                marked_down = round(time.perf_counter() - t_fail, 1)
                break
            await asyncio.sleep(0.25)
        steps[-1]["router_marked_down_after_s"] = marked_down
    finally:
        try:
            await box_mode(ops, box, "normal")
            restored = True
        except HTTPException:
            restored = False
    if restored:
        t_ok = time.perf_counter()
        for _ in range(60):
            st = await _get(ops.client, f"{ops.router}/router/status")
            if st and st["backends"][box]["healthy"]:
                healthy_after = round(time.perf_counter() - t_ok, 1)
                break
            await asyncio.sleep(0.25)
        steps.append(step_summary(f"3 {box} restored", await send()))
        steps[-1]["router_marked_healthy_after_s"] = healthy_after
    after = await _get(ops.client, f"{ops.router}/router/status") or {}
    failing = steps[1] if len(steps) > 1 else None
    other_served = failing and failing["ok"] == failing["requests"] and box not in failing["served_by"]
    return {"box_failed": box, "policy": before["policy"], "started_ts": t_start, "box_restored": restored,
            "switching_worked": bool(other_served),
            "verdict": ("no step ran" if not failing else
                        f"while {box} was failing, all {failing['requests']} forecasts were served by the other box "
                        f"({failing['failovers']} needed a retry)" if other_served else
                        f"while {box} was failing, {failing['errors']} of {failing['requests']} forecasts failed and "
                        f"{failing['served_by'].get(box, 0)} were still served by {box}"),
            "failovers_added": (after.get("failovers") or 0) - (before.get("failovers") or 0),
            "new_switches": (after.get("adaptive") or {}).get("switches", [])[len((before.get("adaptive") or {}).get("switches", [])):],
            "steps": steps}


STACK_IN_HOST = re.compile(r"^(https?://[\w-]+\.)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(\.)")


def current_stack(ops) -> Optional[str]:
    """The workflow id in the router's cluster name (router.<workflow>.<group>.fuzzball), if it has one."""
    m = STACK_IN_HOST.match(ops.router)
    return m.group(2) if m else None


def retarget(ops, new: str) -> dict:
    """Point every address that names the old GPU stack at the new one: router, boxes, supervisors, and the
    chat's LLM. The Agent is found by its .ops, because ui/server.py keeps it inside create_app."""
    import gc
    old = current_stack(ops)
    if old is None:
        raise HTTPException(409, "this server's addresses do not name a Fuzzball workflow; nothing to switch")
    swap = lambda url: url.replace(f".{old}.", f".{new}.") if url else url
    ops.router = swap(ops.router)
    ops.boxes = {n: swap(u) for n, u in ops.boxes.items()}
    ops.stacks = {n: swap(u) for n, u in getattr(ops, "stacks", {}).items()}
    if hasattr(ops, "_model"):
        ops._model = None   # the model card is cached; read it again from the new boxes
    agents = [o for o in gc.get_objects() if type(o).__name__ == "Agent" and getattr(o, "ops", None) is ops]
    for a in agents:
        a.llm_base_url = swap(a.llm_base_url)
    LAST_OK.clear()
    return {"stack": new, "previous": old, "router": ops.router, "boxes": ops.boxes, "chat_agents_moved": len(agents)}


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

    async def run_tool(name: str, args: dict, request: Request):
        if name in WRITE_TOOLS:
            allowed_to_change(request)
        if name == "get_forecast":
            out = await inference(InferenceIn(**args))
            r = out["response"]   # a short form an LLM can read: the first points, not the whole series
            return {**{k: out[k] for k in ("sent_to", "served_by", "failover", "policy", "round_trip_ms")},
                    "model_version": r.get("model_version"), "gpu_ms": (r.get("timing") or {}).get("compute_ms"),
                    "issued_at": r.get("issued_at"), "points": len(r.get("forecast") or []), "forecast": (r.get("forecast") or [])[:8]}
        if name == "get_gpu_status":
            return await gpus(None)
        if name == "get_routing_status":
            return await routing()
        if name == "get_performance":
            return await performance(float(args.get("window", 10)))
        if name == "set_routing_policy":
            update = {k: args[k] for k in ("policy", "preferred") if k in args}
            if update.get("policy") not in ("static", "round_robin", "least_loaded", "adaptive", "balanced"):
                raise HTTPException(422, "policy must be static, round_robin, least_loaded, adaptive or balanced")
            r = await ops.client.post(f"{ops.router}/router/config", json=update, timeout=5)
            if r.status_code != 200:
                raise HTTPException(r.status_code if r.status_code in (422, 404) else 502, r.text[:200])
            st = r.json()
            return {"policy": st["policy"], "preferred": (st.get("adaptive") or {}).get("preferred")}
        if name == "set_box_fault":
            if args.get("box") not in ops.boxes or args.get("mode") not in ("error", "normal"):
                raise HTTPException(422, f"box must be one of {', '.join(ops.boxes)} and mode error or normal")
            await box_mode(ops, args["box"], args["mode"])
            return {"box": args["box"], "mode": args["mode"]}
        if name == "run_switching_check":
            return await switching_check(ops, args.get("box"), max(1, min(int(args.get("requests", 6)), 20)))
        raise HTTPException(404, f"unknown tool {name}; the tools are: {', '.join(t['function']['name'] for t in TOOLS)}")

    @api.get("/admin/target")
    async def target():
        return {"stack": current_stack(ops), "router": ops.router, "boxes": ops.boxes}

    @api.post("/admin/target")
    async def set_target(request: Request):
        allowed_to_change(request)
        try:
            body = await request.json()
        except ValueError:
            body = None
        new = (body or {}).get("stack") if isinstance(body, dict) else None
        if not isinstance(new, str) or not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", new):
            raise HTTPException(422, 'the body must be {"stack": "<Fuzzball workflow id>"}')
        return retarget(ops, new)

    @api.get("/tools")
    async def tools():
        return {"tools": TOOLS, "how_to_call": "POST /api/v1/tools/{name} with the arguments object as the JSON body",
                "need_key": sorted(WRITE_TOOLS), "key_header": "X-API-Key", "key_required": bool(os.environ.get("WX_API_KEY"))}

    @api.post("/tools/{name}")
    async def call_tool(name: str, request: Request):
        try:
            args = await request.json() if await request.body() else {}
        except ValueError:
            raise HTTPException(422, "the body must be a JSON object with the tool's arguments")
        if isinstance(args, str):   # some clients send the LLM's arguments string as is
            import json as _json
            try:
                args = _json.loads(args)
            except ValueError:
                raise HTTPException(422, "the arguments string is not JSON")
        if not isinstance(args, dict):
            raise HTTPException(422, "the body must be a JSON object with the tool's arguments")
        try:
            return {"tool": name, "result": await run_tool(name, args, request)}
        except (TypeError, ValueError) as e:   # wrong argument names or types from the LLM
            raise HTTPException(422, f"bad arguments for {name}: {e}")

    app.include_router(api)

    @app.middleware("http")
    async def cors_for_api(request, call_next):
        if not request.url.path.startswith("/api/v1"):
            return await call_next(request)
        if request.method == "OPTIONS":   # a page on another origin asking before POST /inference
            return Response(status_code=204, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST",
                                                      "Access-Control-Allow-Headers": "content-type, x-api-key, authorization"})
        resp = await call_next(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
