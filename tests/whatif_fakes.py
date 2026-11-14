"""Shared fakes for the WhatIf tests: synthetic telemetry and an LLM that answers with canned JSON."""
from __future__ import annotations

import json
import re

import httpx

T0 = 1_000_000.0


def make_samples(fault: str = "box-b:slow", t0: float = T0, n_pre: int = 35, n_post: int = 35) -> list[dict]:
    """One sample a second, t0 - n_pre .. t0 + n_post, with `fault` starting at t0 + 2."""
    box, _, kind = fault.partition(":")
    out = []
    for i in range(-n_pre, n_post):
        t = t0 + i
        on = i >= 2
        s = {"t": t, "boxes": {}, "probe": {},
             "router": {"policy": "adaptive", "active": "box-b", "preferred": "box-b", "switches": 0, "failovers": 0,
                        "backends": {b: {"healthy": True, "in_flight": 0, "queue_depth": 0, "probe_ms": 55.0,
                                         "health_ms": 4.0, "err_rate": 0.0, "served": 0, "failures": 0}
                                     for b in ("box-a", "box-b")}},
             "traffic": {"rps": 8.0, "p50_ms": 150.0, "p95_ms": 260.0, "errors": 0, "err_pct": 0.0,
                         "per_box": {"box-a": {"requests": 0, "p95_ms": None}, "box-b": {"requests": 16, "p95_ms": 260.0}}}}
        for b in ("box-a", "box-b"):
            sick = on and b == box
            code = 200
            if sick and kind == "down":
                code = 0
            elif sick and kind == "error":
                code = 503
            s["boxes"][b] = {"health": {"code": code, "ms": 4.0 + (i % 3)},
                             "stats": None if code == 0 else {"in_flight": 1, "queue_depth": 0, "max_concurrency": 4,
                                                              "gpu_util": 0.75 if (sick and kind == "slow") else 0.2,
                                                              "p95_latency_ms": 250.0, "errors_total": 0}}
            if i % 2 == 0:  # a probe every 2 s
                compute = 57.0 + (i % 5)
                if sick and kind == "slow":
                    compute *= 3
                p = {"t": t, "code": 200, "compute_ms": compute, "total_ms": compute + 5}
                if sick and kind == "error":
                    p = {"t": t, "code": 503, "compute_ms": None, "total_ms": 3.0}
                if sick and kind == "down":
                    p = {"t": t, "code": 0, "compute_ms": None, "total_ms": 1.0}
                if sick and kind == "hang":
                    p = {"t": t, "code": 0, "compute_ms": None, "total_ms": 3000.0, "timeout": True}
                s["probe"][b] = p
        if on and fault == "load:overload":
            s["traffic"]["rps"] = 24.0
            s["traffic"]["p95_ms"] = 1800.0
            s["boxes"]["box-b"]["stats"]["queue_depth"] = 20
        if on and fault == "router:misconfig":
            s["router"]["policy"] = "static"
            s["router"]["active"] = "box-a"
            s["traffic"]["per_box"] = {"box-a": {"requests": 16, "p95_ms": 700.0}, "box-b": {"requests": 0, "p95_ms": None}}
            s["traffic"]["p95_ms"] = 700.0
        out.append(s)
    return out


class FakeLLM:
    """An OpenAI chat endpoint as an httpx transport. It answers by the message it sees:
    planner -> `plan`; agent for hypothesis H -> `agents[H]` (a list of replies, one per turn, the last repeats);
    single agent -> `single` (same); judge -> `judge` (one reply, or a list like the agents'). Replies are dicts (sent as JSON text) or raw strings."""

    def __init__(self, plan=None, agents=None, single=None, judge=None, default_p=0.05, fail_status=None,
                 wrap_think=False):
        self.plan = plan or {"plan": ["box-b:slow", "box-a:slow", "load:overload", "router:misconfig"]}
        self.agents = agents or {}
        self.single = single or []
        self.judge = judge
        self.default_p = default_p
        self.fail_status = fail_status  # {endpoint host: status} to simulate a broken server
        self.wrap_think = wrap_think
        self.turns: dict[str, int] = {}
        self.requests: list[dict] = []

    def _next(self, key: str, replies: list):
        i = self.turns.get(key, 0)
        self.turns[key] = i + 1
        return replies[min(i, len(replies) - 1)]

    def answer(self, body: dict):
        user = body["messages"][-1]["content"]
        if "Choose the" in user:
            return self.plan
        if "FINAL JUDGEMENT" in user:
            return self._next("judge", self.judge if isinstance(self.judge, list) else [self.judge or {"ranking": []}])
        if "You test ALL hypotheses" in user:
            return self._next("single", self.single or [{"ranking": [], "next_tool": None}])
        m = re.search(r"YOUR HYPOTHESIS: ([\w:-]+):", user)
        h = m.group(1) if m else "?"
        return self._next(h, self.agents.get(h) or [{"p": self.default_p, "next_tool": None, "reason": "no sign"}])

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        if self.fail_status and request.url.host in self.fail_status:
            return httpx.Response(self.fail_status[request.url.host], text="broken")
        reply = self.answer(body)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        if self.wrap_think:
            text = "<think>let me look at the numbers {not json}</think>\n```json\n" + text + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
