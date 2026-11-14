"""wxrouter Control Room: live dashboard and agent chat in one page.

Talks to the router (forecasts, status, metrics, config) and directly to each
box (stats, and /admin/mode on the mock for demo scenes). It also has a small
built-in traffic generator, so the charts move during a live demo.

Run (with the local stack from dev_stack.py, or against the cluster via port-forward):
  python ui/server.py --router http://127.0.0.1:8000 \
      --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002 [--llm-base-url http://host:8080/v1]
Then open http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import Agent

STATIC = Path(__file__).resolve().parent / "static"


class Ops:
    """Everything the dashboard and the agent can do to the cluster."""

    def __init__(self, router: str, boxes: dict[str, str]):
        self.router = router.rstrip("/")
        self.boxes = {k: v.rstrip("/") for k, v in boxes.items()}
        self.client: httpx.AsyncClient | None = None

    async def forecast(self, lat: float, lon: float, hours: int, members: int = 1,
                       place: Optional[str] = None) -> dict:
        t0 = time.perf_counter()
        r = await self.client.post(f"{self.router}/v1/forecast", timeout=60, json={
            "lat": lat, "lon": lon, "hours_ahead": hours, "step_hours": 1, "ensemble_members": members})
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "ms": ms}
        d = r.json()
        pts = d["forecast"]
        step_h = int(d.get("native_step_hours") or 6)
        temps = [p["temp_c"] for p in pts]
        rain = [p.get("precip_mm") for p in pts if p.get("precip_mm") is not None]
        hum = [p.get("humidity_pct") for p in pts if p.get("humidity_pct") is not None]
        summary = {"min_c": min(temps), "max_c": max(temps),
                   "precip_mm": round(sum(rain), 1) if rain else None,  # None: model gave no rain data
                   "max_wind_ms": max(p["wind_ms"] for p in pts),
                   "humidity_avg_pct": round(sum(hum) / len(hum)) if hum else None}
        members_temp = None
        if d.get("members"):
            members_temp = [[p["temp_c"] for p in m["points"]] for m in d["members"]]
            flat = [t for m in members_temp for t in m]
            summary.update(ens_min_c=min(flat), ens_max_c=max(flat))
        return {"place": place, "lat": lat, "lon": lon, "hours_ahead": hours, "ensemble_members": members,
                "native_step_hours": step_h, "issued_at": d.get("issued_at"),
                "served_by": r.headers.get("x-served-by"), "failover": r.headers.get("x-router-failover") == "true",
                "policy": r.headers.get("x-router-policy"), "latency_ms": ms,
                "server_ms": d["timing"]["total_ms"], "low_confidence": d["low_confidence"],
                "model_version": d["model_version"], "summary": summary,
                "points": [{"h": p["lead_hours"], "vt": p["valid_time"], "t": p["temp_c"], "w": p["wind_ms"],
                            "pr": p.get("pressure_hpa"), "p": p.get("precip_mm"), "hum": p.get("humidity_pct"),
                            "cl": p.get("cloud_pct"),
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

    async def cluster_status(self) -> dict:
        st, m, bx = await asyncio.gather(self.router_status(), self.metrics(10, None), self.box_stats())
        return {"policy": st["policy"], "active": st["adaptive"]["active"],
                "preferred": st["adaptive"].get("preferred"), "rps": m["overall"]["rps"],
                "p95_ms": m["overall"]["p95_ms"], "errors_10s": m["overall"]["errors"],
                "boxes": {n: {"healthy": b["healthy"], "mode": (bx.get(n) or {}).get("mode", "unknown"),
                              "served_share_10s": m["per_box"].get(n, {}).get("requests", 0)}
                          for n, b in st["backends"].items()},
                "recent_switches": st["adaptive"]["switches"][-3:]}

    async def set_policy(self, policy: str, preferred: Optional[str] = None) -> dict:
        body = {"policy": policy, **({"preferred": preferred} if preferred is not None else {})}
        st = (await self.client.post(f"{self.router}/router/config", json=body, timeout=2)).json()
        return {"policy": st["policy"], "preferred": st["adaptive"].get("preferred"), "active": st["adaptive"]["active"]}

    async def set_box_mode(self, box: str, mode: str, slow_factor: float = 2.0) -> dict:
        if box not in self.boxes:
            raise HTTPException(404, f"unknown box {box}")
        r = await self.client.post(f"{self.boxes[box]}/admin/mode", timeout=2,
                                   json={"mode": mode, "slow_factor": max(1.0, slow_factor)})
        if r.status_code == 404:
            return {"error": f"{box} has no /admin/mode (real server, not the mock)"}
        return r.json()


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


class PolicyIn(BaseModel):
    policy: str = Field(..., pattern="^(static|round_robin|least_loaded|adaptive|balanced)$")
    preferred: Optional[str] = None


class ForecastIn(BaseModel):
    place: Optional[str] = None
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    hours_ahead: int = Field(48, ge=1, le=120)
    ensemble_members: int = Field(1, ge=1, le=20)


def create_app(ops: Ops, agent: Agent) -> FastAPI:
    load = LoadGen(ops)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ops.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=500))
        yield
        load.set(0)
        await ops.client.aclose()

    app = FastAPI(title="wxrouter control room", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/overview")
    async def overview(since: Optional[float] = None):
        try:
            st, m, bx = await asyncio.gather(ops.router_status(), ops.metrics(10, since), ops.box_stats())
        except httpx.HTTPError as e:
            raise HTTPException(502, f"router unreachable: {e.__class__.__name__}")
        return {"router": st, "metrics": m, "boxes": bx, "load_rps": load.rps,
                "agent_mode": agent.mode, "ts": time.time()}

    @app.post("/api/load")
    async def set_load(x: LoadIn):
        load.set(x.rps)
        return {"rps": load.rps}

    @app.post("/api/box/{box}/mode")
    async def box_mode(box: str, x: ModeIn):
        return await ops.set_box_mode(box, x.mode, x.slow_factor)

    @app.post("/api/router/config")
    async def router_config(x: PolicyIn):
        return await ops.set_policy(x.policy, x.preferred)

    @app.post("/api/forecast")
    async def forecast(x: ForecastIn):
        return await ops.forecast(x.lat, x.lon, x.hours_ahead, x.ensemble_members, x.place)

    @app.post("/api/chat")
    async def chat(x: ChatIn):
        return await agent.chat(x.messages, x.units)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    try:  # inference and switching panels from router_switching/; the page works without them
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from router_switching.dashboard import attach
        attach(app, ops)
    except ImportError:
        pass
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", default="http://127.0.0.1:8000")
    ap.add_argument("--box", action="append", default=[], help="NAME=URL, repeat (for stats and demo modes)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL"),
                    help="OpenAI-compatible base URL, e.g. http://box-a:8080/v1 (llama-server --jinja)")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "nemotron"))
    a = ap.parse_args()
    boxes = dict(x.split("=", 1) for x in a.box) or {"box-a": "http://127.0.0.1:8001", "box-b": "http://127.0.0.1:8002"}
    ops = Ops(a.router, boxes)
    agent = Agent(ops, a.llm_base_url, a.llm_model, os.environ.get("LLM_API_KEY"))
    uvicorn.run(create_app(ops, agent), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
