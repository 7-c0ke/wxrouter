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
import os
import re
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
                 gpu_util_source="nvidia-smi" if dev.get("type") == "cuda" else "busy slots (no GPU)")
        src = (metrics.get("load") or {}).get("gpu_util_source")
        if src == "busy_slots" or dev.get("type") != "cuda":   # busy slots is not a GPU reading: show none
            v.update(gpu_util=None, gpu_util_source="no nvidia-smi reading (no GPU, or nvidia-smi gave nothing)")
        elif src is None:
            v["gpu_util_source"] = "nvidia-smi (this server version does not confirm the source)"
        if metrics["compute_ms"].get("count") == 0:   # an empty window is not 0 ms
            v.update(compute_p50_ms=None, compute_p95_ms=None)
    elif info and "device" in info:  # the weather team's server (serve/forecast_server.py)
        mock = bool(info.get("mock"))
        cuda = str(info.get("device", "")).startswith("cuda")
        v.update(kind=SIM if mock else (REAL if cuda else CPU), device=info.get("model"),
                 gpu_util_source=None if mock else "nvidia-smi")
        v["gpu_util"], v["gpu_util_source"] = None, "not read yet"   # its /stats says 0.0 when it has no reading; /metrics decides
        if mock:
            v["gpu_util"] = None
    elif "mock" in str(v["model_version"] or "").lower() or (stats or {}).get("mock"):
        v.update(kind=SIM, device="mock server", gpu_util_source="simulated")
    if not v["reachable"]:
        v["kind"] = UNKNOWN
    for k in ("latency_p50_ms", "latency_p95_ms"):   # servers fill an empty window with 0.0; no request takes 0 ms
        if v[k] == 0:
            v[k] = None
    return v


def machine_view(v: dict, machine: dict | None) -> None:
    """The weather team's server publishes its nvidia-smi sample at GET /metrics. Unknown stays None."""
    gpu = (machine or {}).get("gpu") or {}
    util = gpu.get("util_pct")
    if v["kind"] == REAL:
        v.update(gpu_util=None if util is None else util / 100.0,
                 gpu_util_source="nvidia-smi, via the server's /metrics" if util is not None else "nvidia-smi gave no reading",
                 gpu_proc_mem_mb=gpu.get("proc_mem_mb"), gpu_temp_c=gpu.get("temp_c"), gpu_power_w=gpu.get("power_w"),
                 ms_per_unit=(machine or {}).get("ms_per_unit"))


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


LAST_OK: dict[str, float] = {}   # "router" or a box URL -> wall-clock time of its last answer


def _seen(key: str, answered: bool) -> dict:
    """When this source last answered. Its values are only shown while it answers; this is the receipt."""
    now = time.time()
    if answered:
        LAST_OK[key] = now
    last = LAST_OK.get(key)
    return {"last_ok_ts": last, "last_ok_age_s": round(now - last, 1) if last else None}


async def live_boxes(client: httpx.AsyncClient, boxes: dict[str, str]) -> dict:
    async def one(name: str, url: str):
        h, s, m = await asyncio.gather(_get(client, f"{url}/health", any_status=True),  # a box that is down still says so, with 503
                                       _get(client, f"{url}/stats"),
                                       _get(client, f"{url}/v1/metrics"))
        info, machine = (None, None) if m else await asyncio.gather(_get(client, f"{url}/v1/info"), _get(client, f"{url}/metrics"))
        v = box_view(h, s, m, info)
        if info:
            machine_view(v, machine)
        v.update(_seen(url, s is not None))
        # one word for the page: OFFLINE when the box gives no telemetry; never a remembered value
        v["state"] = "OFFLINE" if s is None else "DOWN" if v["status"] not in (None, "ok") else "ONLINE"
        if m:
            v["compute_p99_ms"], v["latency_p99_ms"] = m["compute_ms"].get("p99"), (m.get("latency_ms") or {}).get("p99")
            v["gpu_memory_device_total_mb"] = (m.get("gpu_memory") or {}).get("device_total_mb")
        elif info:  # the weather team's server: what its /v1/info says about the model
            v["model_params"], v["model_name"] = info.get("params"), info.get("model")
        return name, v
    return dict(await asyncio.gather(*(one(n, u) for n, u in boxes.items())))


async def live(client: httpx.AsyncClient, router: str, boxes: dict[str, str]) -> dict:
    st, mt, bx = await asyncio.gather(_get(client, f"{router}/router/status"),
                                      _get(client, f"{router}/router/metrics?window=10"),
                                      live_boxes(client, boxes))
    return {"ts": time.time(), "router": router_view(st, mt) if st else None, "boxes": bx,
            "router_seen": _seen("router", st is not None)}


def results(real_only: bool = False) -> dict:
    """The recorded benchmark runs. With real_only, only runs that recorded a CUDA device: CPU and mock
    runs stay in the files under results/ but are not served to the page."""
    if not SUMMARY.exists():
        return {"runs": [], "not_measured_on_gpu": [], "cpu_vs_gpu": None, "real_only": real_only,
                "note": "No summary yet. Run: python -m router_switching.collect"}
    d = json.loads(SUMMARY.read_text())
    if real_only:
        hidden = [r["title"] for r in d["runs"] if r["kind"] != REAL]
        # every recorded real-GPU run except the AFNO check used the untrained stand-in model: say so on each
        runs = [{**r, "stand_in_model": not r["source"].startswith("results/afno/")} for r in d["runs"] if r["kind"] == REAL]
        d = {**d, "runs": runs, "cpu_vs_gpu": None, "hidden_runs": hidden}
    return {**d, "real_only": real_only}


FB_CACHE: dict = {"t": 0.0, "value": None}


async def _cli(*args: str, timeout: float = 12.0) -> str | None:
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
                out["stages"].append({"kind": cols[0], "status": cols[1].upper(), "name": cols[2], "duration": cols[3] or None})
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


async def fuzzball(workflow: str | None) -> dict:
    """What Fuzzball itself says about the workflow behind this page, from its CLI. Kept for 10 s."""
    if not workflow:
        return {"available": False, "reason": "no workflow ID given (start ui/server.py with --workflow ID)"}
    if time.monotonic() - FB_CACHE["t"] < 10 and FB_CACHE["value"] and FB_CACHE["value"].get("id") == workflow:
        return FB_CACHE["value"]
    desc, events, nodes = await asyncio.gather(_cli("workflow", "describe", workflow), _cli("workflow", "events", workflow),
                                               _cli("node", "list"))
    if desc is None:
        v = {"available": False, "id": workflow, "reason": "the fuzzball CLI did not answer (not logged in, or off the cluster network)"}
    else:
        host = {}
        for line in (nodes or "").splitlines():
            c = [x.strip() for x in line.split("|")]
            if len(c) >= 3 and "/" in c[0]:
                host[c[0].split("/")[0]] = c[2]
        place = parse_placement(events or "")
        v = {"available": True, "id": workflow, **parse_describe(desc), "checked_ts": time.time()}
        for s in v["stages"]:
            p = place.get(s["name"])
            # a stage that is not running holds no node, whatever an earlier event said
            s.update(node=host.get(p["node"], p["node"]) if p and s["status"] == "STARTED" else None,
                     gpus=p["gpus"] if p else None)
    FB_CACHE.update(t=time.monotonic(), value=v)
    return v


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


def attach(app: FastAPI, ops, real_only: bool = False, workflow: str | None = None) -> None:
    """`ops` is the Control Room's Ops object: it has .router, .boxes and (once started) .client.
    `real_only`: serve only real-GPU recorded runs. `workflow`: the Fuzzball workflow ID to report on."""
    @app.get("/api/switching/live")
    async def switching_live():
        return await live(ops.client, ops.router, ops.boxes)

    @app.post("/api/switching/burst")
    async def switching_burst(n: int = 200, concurrency: int = 32):
        return await burst(ops.client, ops.router, max(1, min(n, 1000)), max(1, min(concurrency, 100)))

    @app.get("/api/switching/results")
    async def switching_results():
        return results(real_only)

    @app.get("/api/switching/fuzzball")
    async def switching_fuzzball():
        return await fuzzball(workflow)

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
