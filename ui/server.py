"""wxrouter Control Room: live dashboard and agent chat in one page.

The ui2/ revision (simulated vs real model, real fine-tune and kill through the box supervisor,
the model card, null-safe forecasts) is merged in here; ui2/ is gone.

Talks to the router (forecasts, status, metrics, config) and directly to each
box (/stats, /health, /v1/info, and /admin/mode for SIMULATED demo scenes). When
a box's supervisor (serve/node_stack.py) is given with --stack, it can also
start and stop a REAL fine-tune job on that box and really kill its forecast
server. It also has a small built-in traffic generator, so the charts move
during a live demo.

Run (with dev_stack.py or ui/dev_stack_supervised.py, or against the cluster via port-forward):
  python ui/server.py --router http://127.0.0.1:8000 \
      --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002 \
      [--stack box-a=http://127.0.0.1:8109 --stack box-b=http://127.0.0.1:8110] \
      [--llm-base-url http://host:8080/v1]
Then open http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import random
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import MAX_STEPS, Agent, clean_steps, was_stopped
from events import EventHub, FuzzballEvents, TelemetryEvents, find_cli, workflow_from_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the repo root: whatif/, router_switching/
from whatif import config as wcfg  # noqa: E402
from whatif import remediate  # noqa: E402
from whatif.llm import LLMClient  # noqa: E402
from whatif.hypotheses import HYPOTHESES  # noqa: E402
from whatif.scheduler import POLICIES, diagnose  # noqa: E402
from whatif.summarize import engine_label  # noqa: E402
from whatif.telemetry import Recorder  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

# weather/finetune.py prints: "step %4d/%d  loss %.6f  lr %.2e  %s", the last part being the speed
# ("1.52 steps/s", or "3.4 s for this step" on the first step).
STEP_LINE = re.compile(r"^\s*step\s+(\d+)\s*/\s*(\d+)\s+loss\s+(\S+)\s+lr\s+(\S+)\s*(.*?)\s*$")


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def numbers(points: list[dict], key: str) -> list[float]:
    """The values of one field that are numbers. A model with no rain output sends precip_mm: null."""
    return [p[key] for p in points if is_num(p.get(key))]


def is_mock(model_version: Any) -> Optional[bool]:
    """A box is a mock when "mock" is in its model_version. None when the version is not known."""
    if not isinstance(model_version, str) or not model_version:
        return None
    return "mock" in model_version.lower()


def finetune_view(job: Any) -> Optional[dict]:
    """The supervisor's view of its `finetune` job, plus what the last "step N/M ..." line says.
    `stopped` is true when the run was stopped on request: the supervisor's state is "failed" for that too
    (exit below 0, or weather/finetune.py's own exit code 3)."""
    if not isinstance(job, dict):
        return None
    lines = [x for x in (job.get("last_lines") or []) if isinstance(x, str) and x.strip()]
    out = {"state": job.get("state"), "seconds": job.get("seconds"), "exit": job.get("exit"),
           "stopped": was_stopped(job),
           "error": job.get("error"), "last_line": lines[-1].strip()[:200] if lines else None,
           "step": None, "total": None, "loss": None, "speed": None}
    for line in reversed(lines):
        if m := STEP_LINE.match(line):
            try:
                loss = float(m.group(3))
            except ValueError:
                loss = None
            out.update(step=int(m.group(1)), total=int(m.group(2)), loss=loss if is_num(loss) else None,
                       speed=m.group(5) or None)
            break
    return out


class Ops:
    """Everything the dashboard and the agent can do to the cluster."""

    def __init__(self, router: str, boxes: dict[str, str], stacks: Optional[dict[str, str]] = None):
        self.router = router.rstrip("/")
        self.boxes = {k: v.rstrip("/") for k, v in boxes.items()}
        self.stacks = {k: v.rstrip("/") for k, v in (stacks or {}).items()}  # box -> its node_stack supervisor
        self.client: httpx.AsyncClient | None = None
        self._model: tuple[float, dict] | None = None  # (good until, answer) for model_info
        self._hung: dict[str, float] = {}  # url -> not asked again before this time: its last call timed out

    async def forecast(self, lat: float, lon: float, hours: int, members: int = 1,
                       place: Optional[str] = None) -> dict:
        t0 = time.perf_counter()
        try:
            r = await self.client.post(f"{self.router}/v1/forecast", timeout=60, json={
                "lat": lat, "lon": lon, "hours_ahead": hours, "step_hours": 1, "ensemble_members": members})
        except httpx.HTTPError as e:
            return {"error": f"router unreachable: {e.__class__.__name__}",
                    "ms": round((time.perf_counter() - t0) * 1000, 1)}
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "ms": ms}
        d = r.json()
        pts = d["forecast"]
        step_h = int(d.get("native_step_hours") or 6)
        # Any of these may be null on every point (the real model has no rain, humidity or cloud): then None.
        temps, rain, wind = numbers(pts, "temp_c"), numbers(pts, "precip_mm"), numbers(pts, "wind_ms")
        hum = numbers(pts, "humidity_pct")
        summary = {"min_c": min(temps) if temps else None, "max_c": max(temps) if temps else None,
                   "precip_mm": round(sum(rain), 1) if rain else None,
                   "max_wind_ms": max(wind) if wind else None,
                   "humidity_avg_pct": round(sum(hum) / len(hum)) if hum else None}
        members_temp = None
        if d.get("members"):
            members_temp = [[p.get("temp_c") if is_num(p.get("temp_c")) else None for p in m["points"]]
                            for m in d["members"]]
            flat = [t for m in members_temp for t in m if t is not None]
            if flat:
                summary.update(ens_min_c=min(flat), ens_max_c=max(flat))
        version = d.get("model_version")

        def val(p: dict, key: str):
            return p[key] if is_num(p.get(key)) else None

        return {"place": place, "lat": lat, "lon": lon, "hours_ahead": hours, "ensemble_members": members,
                "native_step_hours": step_h, "issued_at": d.get("issued_at"),
                "served_by": r.headers.get("x-served-by"), "failover": r.headers.get("x-router-failover") == "true",
                "policy": r.headers.get("x-router-policy"), "latency_ms": ms,
                "server_ms": (d.get("timing") or {}).get("total_ms"), "low_confidence": bool(d.get("low_confidence")),
                "model_version": version, "simulated": is_mock(version), "summary": summary,
                "points": [{"h": p["lead_hours"], "vt": p["valid_time"], "t": val(p, "temp_c"), "w": val(p, "wind_ms"),
                            "pr": val(p, "pressure_hpa"), "p": val(p, "precip_mm"), "hum": val(p, "humidity_pct"),
                            "cl": val(p, "cloud_pct"),
                            # Per-point flag if the server sends one, else: not on a model step.
                            "i": p["interpolated"] if isinstance(p.get("interpolated"), bool)
                            else p["lead_hours"] % step_h != 0} for p in pts],
                "members_temp": members_temp}

    async def router_status(self) -> dict:
        return (await self.client.get(f"{self.router}/router/status", timeout=2)).json()

    async def metrics(self, window: float, since: Optional[float]) -> dict:
        params = {"window": window, **({"since": since} if since is not None else {})}
        return (await self.client.get(f"{self.router}/router/metrics", params=params, timeout=2)).json()

    async def box_stats(self) -> dict:
        async def one(name: str, url: str):
            try:
                return name, (await self.client.get(f"{url}/stats", timeout=1)).json()
            except (httpx.HTTPError, ValueError):
                return name, None
        return dict(await asyncio.gather(*(one(n, u) for n, u in self.boxes.items())))

    async def _get_json(self, url: str, timeout: float = 1.0) -> tuple[Optional[int], Any]:
        """GET url -> (HTTP status, parsed JSON or None). (None, None) when the call fails or takes longer
        than `timeout` seconds in all. It never raises: the extras of the overview must not break it.
        A URL that timed out is left alone for 5 s, so that one that hangs does not slow every overview."""
        if self._hung.get(url, 0.0) > time.monotonic():
            return None, None
        try:
            r = await asyncio.wait_for(self.client.get(url, timeout=timeout), timeout)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, None
        except (asyncio.TimeoutError, httpx.TimeoutException):
            self._hung[url] = time.monotonic() + 5.0
            return None, None
        except Exception:
            return None, None

    async def box_health(self) -> dict:
        """Each box's GET /health: box, model_version, status, and whether it is a mock. None if no answer.
        The teammate's mock has no model_version in /stats, so /health is where the page learns it."""
        async def one(name: str, url: str):
            _, d = await self._get_json(f"{url}/health")
            if not isinstance(d, dict):
                return name, None
            version = d.get("model_version")
            return name, {"box": d.get("box"), "model_version": version, "status": d.get("status"),
                          "simulated": is_mock(version)}
        return dict(await asyncio.gather(*(one(n, u) for n, u in self.boxes.items())))

    async def stack_status(self) -> dict:
        """For each box with a supervisor: its `finetune` job and its `forecast` server, from GET /status.
        None for a supervisor that does not answer within a second."""
        async def one(name: str, url: str):
            status, d = await self._get_json(f"{url}/status")
            if status != 200 or not isinstance(d, dict):
                return name, None
            servers, jobs = d.get("servers"), d.get("jobs")
            fc = servers.get("forecast") if isinstance(servers, dict) else None
            return name, {
                "name": d.get("name"),
                "finetune": finetune_view(jobs.get("finetune") if isinstance(jobs, dict) else None),
                "forecast": {k: fc.get(k) for k in ("state", "restarts", "up_s", "next_start_in_s", "last_exit")}
                if isinstance(fc, dict) else None}
        return dict(await asyncio.gather(*(one(n, u) for n, u in self.stacks.items())))

    async def model_info(self) -> dict:
        """GET /v1/info of the first box that answers it (our forecast server has it; the teammate's mock
        answers 404). Kept for about 10 s."""
        now = time.monotonic()
        if self._model is not None and now < self._model[0]:
            return self._model[1]

        async def one(name: str, url: str):
            return name, await self._get_json(f"{url}/v1/info")

        got = await asyncio.gather(*(one(n, u) for n, u in self.boxes.items()))
        out, keep = {"available": False, "reason": "no_answer"}, 2.0  # ask again soon: a box may be starting
        for name, (status, body) in got:
            if status == 200 and isinstance(body, dict):
                params = body.get("params")
                out = {"available": True, "box": name, "model": body.get("model"), "params": params,
                       "params_m": round(params / 1e6, 1) if is_num(params) else None,
                       "device": body.get("device"), "mock": body.get("mock"), "latest": body.get("latest"),
                       "ranges": body.get("ranges"), "ms_per_unit": body.get("ms_per_unit")}
                keep = 10.0
                break
        else:
            if any(status == 404 for _, (status, _body) in got):
                out, keep = {"available": False, "reason": "no_v1_info"}, 10.0
        self._model = (now + keep, out)
        return out

    async def cluster_status(self) -> dict:
        st, m, bx, hl, sk = await asyncio.gather(self.router_status(), self.metrics(10, None), self.box_stats(),
                                                 self.box_health(), self.stack_status())

        def box(n: str, b: dict) -> dict:
            out = {"healthy": b["healthy"], "drained": b.get("drained", False), "mode": (bx.get(n) or {}).get("mode", "unknown"),
                   "served_share_10s": m["per_box"].get(n, {}).get("requests", 0),
                   "simulated": (hl.get(n) or {}).get("simulated")}
            if n in self.stacks:  # a real fine-tune job can run here
                # "no answer": the job's state is not known. That is not the same as "no job is running".
                out["supervisor"] = "ok" if sk.get(n) else "no answer"
                ft = (sk.get(n) or {}).get("finetune")
                out["finetune"] = {k: ft.get(k) for k in ("state", "step", "total", "speed", "exit")} if ft else None
                if ft and ft.get("stopped"):
                    out["finetune"]["state"] = "stopped"    # the supervisor's "failed" with a stop's exit code
            return out

        return {"policy": st["policy"], "active": st["adaptive"]["active"],
                "preferred": st["adaptive"].get("preferred"), "rps": m["overall"]["rps"],
                "p95_ms": m["overall"]["p95_ms"], "errors_10s": m["overall"]["errors"],
                "boxes": {n: box(n, b) for n, b in st["backends"].items()},
                "recent_switches": st["adaptive"]["switches"][-3:],
                "settings": (st.get("config") or {}).get("settings")}

    async def router_config(self) -> dict:
        """The router's GET /router/config: settings now, plus the rules, boxes and ranges allowed."""
        return (await self.client.get(f"{self.router}/router/config", timeout=2)).json()

    async def set_config(self, update: dict) -> dict:
        """POST /router/config with any of its fields. Returns a short summary, or {"error": ...}
        with the router's own reason (an unknown box, a value out of range) and nothing changed."""
        body = {k: v for k, v in update.items() if v is not None}
        r = await self.client.post(f"{self.router}/router/config", json=body, timeout=2)
        if r.status_code != 200:
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
            if isinstance(detail, list):  # pydantic: [{"loc": [..., "margin"], "msg": "..."}]
                detail = "; ".join(f"{d.get('loc', ['?'])[-1]}: {d.get('msg')}" for d in detail)
            return {"error": str(detail), "status": r.status_code}
        st = r.json()
        return {"policy": st["policy"], "preferred": st["adaptive"].get("preferred"),
                "active": st["adaptive"]["active"], "config": st.get("config")}

    async def set_policy(self, policy: str, preferred: Optional[str] = None) -> dict:
        return await self.set_config({"policy": policy, "preferred": preferred})

    async def set_box_mode(self, box: str, mode: str, slow_factor: float = 2.0) -> dict:
        """SIMULATED trouble, through the box's /admin/mode test hook. Nothing real happens on the box."""
        if box not in self.boxes:
            raise HTTPException(404, f"unknown box {box}")
        try:
            r = await self.client.post(f"{self.boxes[box]}/admin/mode", timeout=2,
                                       json={"mode": mode, "slow_factor": max(1.0, slow_factor)})
        except httpx.HTTPError as e:
            return {"error": f"{box} unreachable: {e.__class__.__name__}"}
        if r.status_code == 404:
            return {"error": f"{box} has no /admin/mode (real server, not the mock)"}
        try:
            out = r.json()
        except ValueError:
            out = None
        if not isinstance(out, dict):
            return {"error": f"{box} /admin/mode: HTTP {r.status_code}: {r.text[:200]}"}
        if r.status_code >= 400 and "error" not in out:
            out["error"] = f"{box} /admin/mode: HTTP {r.status_code}: {r.text[:200]}"
        if "error" not in out:
            out["kind"] = "simulated"
        return out

    async def stack_post(self, box: str, path: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict:
        """POST to the supervisor of a box -> its JSON answer, or {"error": ...}."""
        if box not in self.boxes:
            raise HTTPException(404, f"unknown box {box}")
        url = self.stacks.get(box)
        if not url:
            return {"error": f"{box} has no supervisor address: start ui/server.py with --stack {box}=URL "
                             "(without one, only the simulated modes are there)"}
        try:
            r = await self.client.post(f"{url}{path}", timeout=timeout,
                                       **({"json": body} if body is not None else {}))
        except httpx.HTTPError as e:
            return {"error": f"the supervisor of {box} did not answer: {e.__class__.__name__}"}
        try:
            out = r.json()
        except ValueError:
            out = None
        if not isinstance(out, dict):
            return {"error": f"the supervisor of {box} answered HTTP {r.status_code}: {r.text[:200]}"}
        if r.status_code != 200 and "error" not in out:
            out["error"] = f"the supervisor of {box} answered HTTP {r.status_code}"
        if "error" not in out:
            out["kind"] = "real job"
        return out

    async def finetune(self, box: str, action: str, steps: Optional[int] = None) -> dict:
        """Start or stop the REAL `finetune` job of a box, through its supervisor."""
        if action == "start":
            try:
                steps = clean_steps(steps)      # for every caller, not only the HTTP endpoint: 1 to MAX_STEPS
            except ValueError as e:
                return {"error": str(e)}
            body = {"args": ["--steps", str(steps)]} if steps is not None else None
            return await self.stack_post(box, "/jobs/finetune/start", body)
        # The supervisor answers when the job is gone: SIGTERM first, SIGKILL 30 s later.
        out = await self.stack_post(box, "/jobs/finetune/stop", timeout=45.0)
        if was_stopped(out.get("status")):
            out["stopped"] = True       # its state reads "failed": the supervisor's word for every exit but 0
        return out

    async def kill_forecast(self, box: str) -> dict:
        """Really kill (SIGKILL) the forecast server of a box. Its supervisor starts it again by itself."""
        return await self.stack_post(box, "/servers/forecast/kill")


class LoadGen:
    """Background traffic so the dashboard has something to show."""

    def __init__(self, ops: Ops):
        self.ops, self.rps, self.task, self.sent = ops, 0.0, None, 0
        self.rng = random.Random(7)

    async def _run(self) -> None:
        pending: set[asyncio.Task] = set()
        while self.rps > 0:
            await asyncio.sleep(self.rng.expovariate(self.rps))
            r = self.rng.random()
            hours = self.rng.randint(6, 24) if r < .6 else self.rng.randint(25, 72) if r < .92 else self.rng.randint(73, 120)
            body = {"lat": round(self.rng.uniform(-60, 70), 2), "lon": round(self.rng.uniform(-180, 180), 2),
                    "hours_ahead": hours, "step_hours": 1, "request_id": "ui-load"}
            t = asyncio.create_task(self._send(body))
            pending.add(t)
            t.add_done_callback(pending.discard)

    async def _send(self, body: dict) -> None:
        self.sent += 1
        try:
            await self.ops.client.post(f"{self.ops.router}/v1/forecast", json=body, timeout=30)
        except httpx.HTTPError:
            pass

    def set(self, rps: float) -> None:
        self.rps = max(0.0, min(rps, 200.0))
        if self.rps > 0 and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self._run())


class ChatIn(BaseModel):
    messages: list[dict] = Field(..., max_length=50)
    units: str = Field("C", pattern="^[CF]$")


class LoadIn(BaseModel):
    rps: float = Field(..., ge=0, le=200)


class ModeIn(BaseModel):
    mode: str = Field(..., pattern="^(normal|slow|error)$")
    slow_factor: float = 3.0


class TrainIn(BaseModel):
    action: str = Field(..., pattern="^(start|stop)$")
    steps: Optional[int] = Field(None, ge=1, le=MAX_STEPS)


class RoutingIn(BaseModel):
    """Same fields as the router's POST /router/config; the router checks the ranges."""
    policy: Optional[str] = Field(None, pattern="^(static|round_robin|least_loaded|adaptive|balanced)$")
    preferred: Optional[str] = None
    margin: Optional[float] = None
    window_s: Optional[float] = None
    cooldown_s: Optional[float] = None
    return_margin: Optional[float] = None
    stall_ms: Optional[float] = None
    drain: Optional[dict[str, bool]] = None
    reset: Optional[bool] = None


class DiagnoseIn(BaseModel):
    policy: str = Field("rule", description="whatif.scheduler policy: rule (no LLM), triage, single, equal or whatif")
    pre_s: float = Field(30.0, ge=10, le=80)
    post_s: float = Field(30.0, ge=10, le=80)


class RemedyIn(BaseModel):
    incident: int


class ForecastIn(BaseModel):
    place: Optional[str] = None
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    hours_ahead: int = Field(48, ge=1, le=120)
    ensemble_members: int = Field(1, ge=1, le=20)


class DiagnosisUnavailable(Exception):
    """No WhatIf diagnosis now: an unknown policy, not enough telemetry, no live router, or the LLM failed."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def create_app(ops: Ops, agent: Agent, fuzzball_workflow: Optional[str] = None,
               fuzzball_cli: Optional[str] = None, telemetry: bool = True) -> FastAPI:
    load = LoadGen(ops)
    # One telemetry source for the Events card and for incident diagnosis: the WhatIf recorder, one sample a
    # second of the router, the boxes, a probe forecast and the supervisors (whatif/telemetry.py).
    hub = EventHub()
    seen = TelemetryEvents(hub)
    recorder = Recorder(ops.router, ops.boxes, ops.stacks)
    fuzz = FuzzballEvents(hub, fuzzball_workflow, fuzzball_cli)
    incidents: dict[int, dict] = {}

    async def record() -> None:
        nxt = time.monotonic()
        while True:
            try:
                seen.observe(await recorder.sample_once())
            except Exception as e:  # one bad sample must not stop the recorder
                print(f"telemetry sample failed: {e!r}", file=sys.stderr)
            nxt += recorder.period_s
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ops.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=500))
        recorder.client = ops.client
        tasks = ([asyncio.create_task(record())] if telemetry else []) + [asyncio.create_task(fuzz.run())]
        yield
        for t in tasks:
            t.cancel()
        load.set(0)
        await ops.client.aclose()

    def whatif_config() -> wcfg.Config:
        """WhatIf settings: the LLM ones from the environment / .env, the stack from this page's own flags."""
        base = wcfg.load()
        return wcfg.Config(base.llm_endpoints, base.model, base.node_cap, base.llm_api_key, base.thinking,
                           ops.router, dict(ops.boxes), dict(ops.stacks))

    app = FastAPI(title="wxrouter control room", lifespan=lifespan)
    app.state.hub, app.state.recorder, app.state.fuzzball = hub, recorder, fuzz   # for tests and tools

    @app.middleware("http")
    async def revalidate_page(request, call_next):
        # The page and its scripts change with every deploy: make browsers check (ETag) instead of
        # running an old app.js from their cache against a new server.
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/overview")
    async def overview(since: Optional[float] = None):
        try:
            # The last three never raise: a box or a supervisor that is slow or gone shows as null.
            st, m, bx, hl, sk, model = await asyncio.gather(
                ops.router_status(), ops.metrics(10, since), ops.box_stats(),
                ops.box_health(), ops.stack_status(), ops.model_info())
        except httpx.HTTPError as e:
            raise HTTPException(502, f"router unreachable: {e.__class__.__name__}")
        mocks = [n for n, h in hl.items() if h and h.get("simulated")]
        # "simulated" on the model card: /v1/info says so, or there is no /v1/info and a box is a known mock.
        model = {**model, "simulated": bool(model.get("mock")) if model.get("available") else bool(mocks)}
        return {"router": st, "metrics": m, "boxes": bx, "health": hl, "stacks": sk, "model": model,
                "simulated_boxes": mocks, "load_rps": load.rps, "agent_mode": agent.mode, "ts": time.time()}

    @app.get("/api/model")
    async def model():
        return await ops.model_info()

    @app.post("/api/load")
    async def set_load(x: LoadIn):
        load.set(x.rps)
        return {"rps": load.rps}

    @app.post("/api/box/{box}/mode")
    async def box_mode(box: str, x: ModeIn):
        return await ops.set_box_mode(box, x.mode, x.slow_factor)

    @app.post("/api/box/{box}/train")
    async def box_train(box: str, x: TrainIn):
        return await ops.finetune(box, x.action, x.steps)

    @app.post("/api/box/{box}/kill")
    async def box_kill(box: str):
        return await ops.kill_forecast(box)

    @app.get("/api/router/config")
    async def get_router_config():
        try:
            return await ops.router_config()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"router unreachable: {e.__class__.__name__}")

    @app.post("/api/router/config")
    async def router_config(x: RoutingIn):
        try:
            out = await ops.set_config(x.model_dump(exclude_none=True))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"router unreachable: {e.__class__.__name__}")
        if "error" in out:
            raise HTTPException(out.get("status", 422), out["error"])
        return out

    @app.post("/api/forecast")
    async def forecast(x: ForecastIn):
        return await ops.forecast(x.lat, x.lon, x.hours_ahead, x.ensemble_members, x.place)

    # ---------------------------------------------------------------- events, telemetry, incidents
    @app.get("/api/events")
    async def events(after: int = 0):
        """The Events card's live stream after `after` (seq): telemetry changes, Fuzzball workflow events and
        incident diagnoses. Routing switches and page actions are logged by the page itself."""
        return {"last_seq": hub.seq, "events": hub.since(after), "fuzzball": fuzz.info(),
                "telemetry": {"on": telemetry, "samples": len(recorder.buf),
                              "seconds": round(recorder.buf[-1]["t"] - recorder.buf[0]["t"], 1) if len(recorder.buf) > 1 else 0}}

    @app.get("/api/telemetry")
    async def telemetry_samples(seconds: float = 60.0):
        """The WhatIf recorder's samples of the last `seconds` (format: whatif/telemetry.py)."""
        cut = time.time() - max(1.0, min(seconds, 180.0))
        return {"samples": [s for s in recorder.snapshot() if s["t"] >= cut]}

    async def run_diagnosis(policy: str = "rule", pre_s: float = 30.0, post_s: float = 30.0,
                            source: str = "page") -> dict:
        """Diagnose the recent past with WhatIf: the alert is `post_s` ago, compared with the `pre_s` before it.
        Used by POST /api/incident/diagnose and by the chat (source "chat"). Raises DiagnosisUnavailable."""
        if policy not in POLICIES:
            raise DiagnosisUnavailable(422, f"unknown policy {policy!r}; choose from {', '.join(POLICIES)}")
        samples = recorder.snapshot()
        now = time.time()
        if not samples or now - samples[0]["t"] < pre_s + post_s - 2:
            have = 0 if not samples else now - samples[0]["t"]
            raise DiagnosisUnavailable(409, f"not enough telemetry yet: {have:.0f} s recorded, {pre_s + post_s:.0f} s needed")
        t0 = now - post_s
        if not any(smp.get("router") for smp in samples if smp["t"] >= t0):
            raise DiagnosisUnavailable(503, "router unreachable: there is no live telemetry to diagnose")
        cfg = whatif_config()
        llm = None if policy == "rule" else LLMClient(cfg.llm_endpoints, cfg.model, cfg.node_cap, cfg.llm_api_key)
        iid = len(incidents) + 1
        try:
            d = await diagnose(samples, t0, policy, llm, session=f"{source}-{iid}")
        except Exception as e:  # an LLM that is not reachable, a bad answer
            hub.add("incident", "bad", f"diagnosis ({policy}) failed: {e.__class__.__name__}: {e}"[:300])
            raise DiagnosisUnavailable(502, f"diagnosis failed: {e.__class__.__name__}: {e}")
        finally:
            if llm:
                await llm.aclose()
        target, _, kind = d.top().partition(":")
        if kind == "down" and target in ops.boxes and target not in ops.stacks:
            # WhatIf's remedy for a down box goes through its supervisor; without one there is nothing to call.
            action = remediate.Action(d.top(), None, None, None, f"{target} has no supervisor here: start its forecast server by hand")
        else:
            action = remediate.plan(d.top(), cfg)
        incidents[iid] = {"diagnosis": d.as_dict(), "action": action, "alert": t0}
        out = {"incident": iid, "alert": t0, "policy": policy, "diagnosis": d.as_dict(), "remedy": action.describe(),
               "can_apply": bool(action.method), "engine": "rule (no LLM)" if policy == "rule" else engine_label(cfg.redacted()),
               "meaning": re.split(r"[.(:]", HYPOTHESES.get(d.top(), ""))[0].strip()}
        hub.add("incident", "info", f"diagnosis #{iid} ({policy}{', asked in the chat' if source == 'chat' else ''}): "
                f"most likely {d.top()}" + (f"; then {', '.join(d.ranking[1:3])}" if len(d.ranking) > 1 else "")
                + f". Remedy: {action.describe()}",
                incident=iid, trail=d.trail[:40], ranking=d.ranking[:5], can_apply=bool(action.method))
        return out

    agent.diagnoser = run_diagnosis   # the chat sends infrastructure questions here (ui/agent.py)

    @app.post("/api/incident/diagnose")
    async def incident_diagnose(x: DiagnoseIn):
        """Diagnose the recent past with WhatIf: the alert is `post_s` ago, compared with the `pre_s` before it."""
        try:
            return await run_diagnosis(x.policy, x.pre_s, x.post_s)
        except DiagnosisUnavailable as e:
            raise HTTPException(e.status, e.message)

    @app.post("/api/incident/remediate")
    async def incident_remediate(x: RemedyIn):
        """Carry out the remedy WhatIf planned for incident `incident` (one action; nothing else changes)."""
        inc = incidents.get(x.incident)
        if not inc:
            raise HTTPException(404, f"no incident #{x.incident}")
        action = inc["action"]
        if not action.method:
            raise HTTPException(409, f"incident #{x.incident} has no automatic remedy: {action.note}")
        if action.applied:
            raise HTTPException(409, f"incident #{x.incident}: the remedy was already applied")
        action = await remediate.execute(action, whatif_config(), apply=True, settle_s=0, client=ops.client)
        hub.add("incident", "up" if action.applied else "bad",
                f"remedy for #{x.incident} {'applied' if action.applied else 'FAILED'}: {action.describe()} "
                f"(HTTP {action.result.get('status')}, p95 before {action.result.get('p95_before_ms') or '–'} ms)",
                incident=x.incident)
        return {"incident": x.incident, "applied": action.applied, "result": action.result, "remedy": action.describe()}

    @app.post("/api/chat")
    async def chat(x: ChatIn):
        return await agent.chat(x.messages, x.units)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    try:  # the switching APIs from router_switching/ (/api/switching/live, results, burst); no cards on the page
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from router_switching.dashboard import attach
        attach(app, ops)
    except ImportError:
        pass
    return app


def name_url(texts: list[str], flag: str, ap: argparse.ArgumentParser) -> dict[str, str]:
    out = {}
    for text in texts:
        name, eq, url = text.partition("=")
        if not eq or not name or not url.startswith(("http://", "https://")):
            ap.error(f"{flag} wants NAME=URL, like {flag} box-a=http://127.0.0.1:8001, got {text!r}")
        out[name] = url
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", default="http://127.0.0.1:8000")
    ap.add_argument("--box", action="append", default=[], help="NAME=URL, repeat (for stats and demo modes)")
    ap.add_argument("--stack", action="append", default=[],
                    help="NAME=URL, repeat: the node_stack supervisor of that box (serve/node_stack.py --admin). "
                         "With it the box card starts and stops the real fine-tune job and really kills the "
                         "forecast server; without it the card has the simulated modes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL"),
                    help="OpenAI-compatible base URL, e.g. http://box-a:8080/v1 (llama-server --jinja)")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "nemotron"))
    ap.add_argument("--llm-thinking", choices=["on", "off"], default=os.environ.get("LLM_THINKING", "on"),
                    help="off: ask the LLM server to skip the model's thinking phase, for faster answers "
                         "(sends chat_template_kwargs; llama-server and vLLM accept it)")
    ap.add_argument("--fuzzball-workflow", default=os.environ.get("FUZZBALL_WORKFLOW"),
                    help="workflow id whose Fuzzball events go to the Events card (default: the one deploy/fb.py up "
                         "started, from .fb_deploy.json); needs a logged-in fuzzball CLI on this machine")
    ap.add_argument("--no-fuzzball-events", action="store_true", help="do not read Fuzzball workflow events")
    a = ap.parse_args()
    boxes = name_url(a.box, "--box", ap) or {"box-a": "http://127.0.0.1:8001", "box-b": "http://127.0.0.1:8002"}
    stacks = name_url(a.stack, "--stack", ap)
    for name in stacks:
        if name not in boxes:
            ap.error(f"--stack {name}=... names no box; the boxes are: {', '.join(boxes)}")
    ops = Ops(a.router, boxes, stacks)
    agent = Agent(ops, a.llm_base_url, a.llm_model, os.environ.get("LLM_API_KEY"),
                  llm_thinking=a.llm_thinking != "off")
    wf = None if a.no_fuzzball_events else (a.fuzzball_workflow or workflow_from_state())
    uvicorn.run(create_app(ops, agent, wf, None if a.no_fuzzball_events else find_cli()),
                host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
