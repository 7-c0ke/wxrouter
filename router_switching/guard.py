"""Real GPUs only: the Control Room refuses to show a mock server or a CPU box as live data.

`install(app, ops)` is called by ui/server.py unless it was started with --dev. From then on every
/api/ call first checks what the boxes themselves report (the same reading as the GPU inference card:
/health, /stats, /v1/metrics or /v1/info). If a box is a mock server or runs on CPU, the call is refused
with HTTP 503 and the reason, and the page shows that reason instead of numbers. A box that does not
answer is not refused: it is shown as OFFLINE with no values.

Nothing is substituted. The only /api/ path that still answers while refused is GET /api/switching/guard,
which says why.
"""
from __future__ import annotations

import time

from fastapi.responses import JSONResponse

from .dashboard import CPU, SIM, live_boxes

OPEN_PATHS = ("/api/switching/guard", "/api/switching/fuzzball")
CHECK_EVERY_S = 3.0


class RealOnly:
    def __init__(self, ops):
        self.ops, self.checked, self.refused = ops, 0.0, {}

    async def check(self) -> dict:
        """{box: reason} for every box that must not be shown. Empty when all answering boxes are real GPUs."""
        if time.monotonic() - self.checked >= CHECK_EVERY_S:
            boxes = await live_boxes(self.ops.client, self.ops.boxes)
            self.refused = {
                n: (f"{n} is a mock server ({b['model_version'] or b['device']})" if b["kind"] == SIM
                    else f"{n} runs on CPU, not on a GPU ({b['device']})")
                for n, b in boxes.items() if b["kind"] in (SIM, CPU)}
            self.checked = time.monotonic()
        return self.refused

    def state(self) -> dict:
        return {"real_only": True, "refused": self.refused,
                "detail": ("Refused: " + "; ".join(self.refused.values()) + ". This page shows real GPU "
                           "telemetry only. For local development start ui/server.py with --dev.") if self.refused else None}


def install(app, ops) -> RealOnly:
    guard = RealOnly(ops)

    @app.middleware("http")
    async def real_only(request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in OPEN_PATHS and ops.client is not None:
            if await guard.check():
                return JSONResponse(guard.state(), status_code=503)
        return await call_next(request)

    @app.get("/api/switching/guard")
    async def guard_state():
        await guard.check()
        return guard.state()

    app.state.real_only = guard
    return guard
