"""From a diagnosis to one action. Dry run unless apply=True (the CLI's --apply), for the live demo only.

    box-X:slow       POST router /router/config {"preferred": <other box>}      steer traffic off the slow GPU
    box-X:down       POST supervisor /servers/forecast/start                      bring the server back now
    router:misconfig POST router /router/config {"policy": "balanced"}            a policy that uses both boxes
    load:overload    nothing: recommend a rate limit or more replicas
    box-X:error/hang POST box /admin/mode {"mode": "normal"}                     mock fault hooks only

With apply, it reads the router's p95 over the last 30 s, acts, waits `settle_s` (30), and reads it again.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import httpx

from whatif.config import BOXES, Config


@dataclass
class Action:
    hypothesis: str
    method: Optional[str]
    url: Optional[str]
    body: Optional[dict]
    note: str
    applied: bool = False
    result: dict = field(default_factory=dict)

    def describe(self) -> str:
        if not self.method:
            return f"{self.hypothesis}: no automatic action. {self.note}"
        return f"{self.hypothesis}: {self.method} {self.url} {self.body or ''}  ({self.note})"


def plan(hypothesis: str, cfg: Config) -> Action:
    target, _, kind = hypothesis.partition(":")
    other = next((b for b in BOXES if b != target), None)
    if kind == "slow" and target in BOXES:
        return Action(hypothesis, "POST", f"{cfg.router}/router/config", {"preferred": other},
                      f"make {other} the router's home box while {target} is slow")
    if kind == "down" and target in BOXES:
        return Action(hypothesis, "POST", f"{cfg.stacks[target]}/servers/forecast/start", None,
                      f"ask {target}'s supervisor to start the forecast server now")
    if hypothesis == "router:misconfig":
        return Action(hypothesis, "POST", f"{cfg.router}/router/config", {"policy": "balanced"},
                      "switch to the balanced rule, which keeps both boxes in play")
    if hypothesis == "load:overload":
        return Action(hypothesis, None, None, None,
                      "Rate-limit the heaviest clients or add forecast replicas; both boxes are healthy.")
    if kind in ("error", "hang") and target in BOXES:
        return Action(hypothesis, "POST", f"{cfg.boxes[target]}/admin/mode", {"mode": "normal"},
                      "clear the fault hook (mock and --admin servers only)")
    return Action(hypothesis, None, None, None, "unknown hypothesis")


async def router_p95(client: httpx.AsyncClient, router: str, window_s: float = 30.0) -> Optional[float]:
    try:
        r = await client.get(f"{router}/router/metrics", params={"window": window_s}, timeout=3)
        return (r.json().get("overall") or {}).get("p95_ms")
    except (httpx.HTTPError, ValueError):
        return None


async def execute(action: Action, cfg: Config, apply: bool = False, settle_s: float = 30.0,
                  client: httpx.AsyncClient | None = None) -> Action:
    if not apply or not action.method:
        return action
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        action.result["p95_before_ms"] = await router_p95(client, cfg.router)
        try:
            r = await client.request(action.method, action.url, json=action.body, timeout=5)
            action.result["status"] = r.status_code
        except httpx.HTTPError as e:
            action.result["status"] = 0
            action.result["error"] = e.__class__.__name__
        action.applied = action.result["status"] in range(200, 300)
        if settle_s > 0:
            await asyncio.sleep(settle_s)
        action.result["p95_after_ms"] = await router_p95(client, cfg.router)
        return action
    finally:
        if own:
            await client.aclose()
