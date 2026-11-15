#!/usr/bin/env python3
"""The team's Control Room (ui/server.py), unchanged, that refuses to show anything but real GPU data.

Same page, same command line as ui/server.py. Every /api/ call first checks what the boxes themselves
report (/health, /stats, /v1/metrics or /v1/info, read by router_switching/api.py). If any box is a mock
server ("simulated") or runs on CPU, the call is refused with HTTP 503 and the reason, so no mock or CPU
number can reach the page or the chat. A box that does not answer is not refused: it shows as OFFLINE.

  GET /api/realonly    whether anything is refused right now, and why

  python router_switching/serve_realonly.py --host 0.0.0.0 --port 18180 --router http://router:8000 \
      --box box-a=http://box-a:8001 --box box-b=http://box-b:8002 --llm-base-url http://box-a:8090/v1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))
sys.path.insert(0, str(ROOT))

import server as ui  # noqa: E402  the team's ui/server.py, unchanged
from fastapi.responses import JSONResponse  # noqa: E402

from router_switching.api import read_gpus  # noqa: E402
from router_switching.dashboard import CPU, SIM  # noqa: E402

CHECK_EVERY_S = 3.0


class RealOnly:
    def __init__(self, ops):
        self.ops, self.checked, self.refused = ops, 0.0, {}

    async def check(self) -> dict:
        """{box: reason} for each box that must not be shown; empty when every answering box is a real GPU."""
        if time.monotonic() - self.checked >= CHECK_EVERY_S:
            gpus = await read_gpus(self.ops.client, self.ops.boxes)
            self.refused = {n: (f"{n} is a mock server ({g['model']['version']})" if g["kind"] == SIM
                                else f"{n} runs on CPU, not on a GPU ({g['device']})")
                            for n, g in gpus.items() if g["kind"] in (SIM, CPU)}
            self.checked = time.monotonic()
        return self.refused

    def state(self) -> dict:
        return {"real_only": True, "refused": self.refused,
                "detail": ("Refused: " + "; ".join(self.refused.values()) + ". This page shows real GPU data only.")
                if self.refused else None}


def build(ops, agent):
    app = ui.create_app(ops, agent)
    guard = RealOnly(ops)

    @app.middleware("http")
    async def real_only(request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path != "/api/realonly" and ops.client is not None:
            if await guard.check():
                return JSONResponse(guard.state(), status_code=503)
        return await call_next(request)

    @app.get("/api/realonly")
    async def realonly():
        await guard.check()
        return guard.state()
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
