#!/usr/bin/env python3
"""The team's Control Room (ui/server.py), read-only: safe to give a link to anyone on the network.

Same page, same command line as ui/server.py. What changes:

  works      the dashboard, live GPU / routing / traffic data, forecasts, the chat (Nemotron) for questions,
             GET /api/v1/*, POST /api/v1/inference, the load test up to MAX_LOAD_RPS
  refused    changing the routing rule, home box or settings, Drain, failing / slowing a box, Fine-tune,
             Kill server (from the page and from the chat), the 1,000-request burst, and the /api/v1 tools
             that change the cluster unless the caller has WX_API_KEY

  python router_switching/serve_readonly.py --host 0.0.0.0 --port 18180 --router http://router:8000 \
      --box box-a=http://box-a:8001 --box box-b=http://box-b:8002 --llm-base-url http://box-a:8090/v1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))
sys.path.insert(0, str(ROOT))

import server as ui  # noqa: E402  the team's ui/server.py, unchanged
from fastapi.responses import JSONResponse  # noqa: E402

MAX_LOAD_RPS = float(os.environ.get("WX_MAX_LOAD_RPS", "10"))
NOTE = "This is a read-only link: it shows the cluster and answers forecasts, but cannot change it."
BLOCKED_POST = ("/api/router/config", "/api/box/", "/api/switching/burst")


def make_readonly(ops) -> None:
    """Every way the page or the chat changes the cluster goes through these Ops methods."""
    async def refuse(*_a, **_k):
        return {"error": NOTE, "status": 403}
    for name in ("set_config", "set_policy", "set_box_mode", "finetune", "kill_forecast", "stack_post"):
        setattr(ops, name, refuse)
    ops.stacks = {}   # no supervisor addresses: the page shows no working Fine-tune / Kill buttons


def cap_load() -> None:
    real_set = ui.LoadGen.set

    def capped(self, rps: float) -> None:
        real_set(self, min(float(rps), MAX_LOAD_RPS))
    ui.LoadGen.set = capped


def build(ops, agent):
    make_readonly(ops)
    cap_load()
    app = ui.create_app(ops, agent)

    @app.middleware("http")
    async def refuse_changes(request, call_next):
        if request.method == "POST" and request.url.path.startswith(BLOCKED_POST):
            return JSONResponse({"detail": NOTE}, status_code=403)
        return await call_next(request)

    @app.get("/api/readonly")
    async def readonly():
        return {"read_only": True, "max_load_rps": MAX_LOAD_RPS, "note": NOTE}
    return app


def main() -> None:
    # Let the team's own main() parse the command line and build Ops and Agent; catch them instead of serving.
    captured = {}
    real_run, real_create = ui.uvicorn.run, ui.create_app
    ui.uvicorn.run = lambda app, **kw: captured.update(kw=kw)
    ui.create_app = lambda ops, agent: captured.update(ops=ops, agent=agent)
    try:
        ui.main()
    finally:
        ui.uvicorn.run, ui.create_app = real_run, real_create
    real_run(build(captured["ops"], captured["agent"]), **captured["kw"])


if __name__ == "__main__":
    main()
