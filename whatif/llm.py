"""The LLM client WhatIf's agents share: two llama-servers (or any OpenAI-compatible bases), one incident per box.

- Sticky per incident: an incident's first call picks the least-loaded endpoint and every later call of that
  incident goes there (llama-server keeps the shared prompt prefix in its slot cache). `X-Session-Id: <incident>`
  is sent too, for a router in front.
- A PriorityGate per endpoint caps concurrent calls at NODE_CAP (llama-server -np 4); waiting calls go in
  priority order (lower number first), then arrival order.
- Failover: a connection error, timeout or 5xx moves the incident to the other endpoint and retries once there.
- Thinking is off unless asked: `chat_template_kwargs: {"enable_thinking": false}` (as ui/agent.py does).
- Every call is appended to calls.jsonl when a log path is given: endpoint, session, ms, tokens, ok.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from whatif.config import key_for


# -------------------------------------------------------------------- JSON out of model text
RATE_LIMIT_RETRIES = 5     # HTTP 429: retry the same endpoint this many times
RATE_LIMIT_BACKOFF_S = 2.0  # first wait; doubles each retry
THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text: str) -> str:
    text = THINK.sub("", text or "")
    if "</think>" in text.lower():  # opening tag eaten by the template: drop everything up to the close
        text = text[text.lower().rindex("</think>") + len("</think>"):]
    if "<think>" in text.lower():  # never closed: the answer did not come
        text = text[:text.lower().index("<think>")]
    return text.strip()


def _balanced(text: str, start: int) -> Optional[str]:
    open_c = text[start]
    close_c = "}" if open_c == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(text: str) -> Any:
    """The first JSON object (or array) in a model reply. Handles <think> blocks, ``` fences, prose around it,
    trailing commas and single-line // comments. Raises ValueError when there is none."""
    t = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(t)
    for cand in candidates:
        for m in re.finditer(r"[{\[]", cand):
            chunk = _balanced(cand, m.start())
            if not chunk:
                continue
            for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", re.sub(r"(?m)//[^\n\"]*$", "", chunk))):
                try:
                    return json.loads(attempt)
                except ValueError:
                    pass
    raise ValueError("no JSON object in the reply: " + (t[:120] + "..." if len(t) > 120 else t))


# -------------------------------------------------------------------- the gate
class PriorityGate:
    """At most `cap` holders; waiters are served lowest priority number first, then first come."""

    def __init__(self, cap: int):
        self.cap = cap
        self.active = 0
        self.peak = 0
        self._heap: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    @property
    def waiting(self) -> int:
        return sum(1 for _, _, f in self._heap if not f.done())

    @property
    def load(self) -> int:
        return self.active + self.waiting

    async def acquire(self, priority: int = 5) -> None:
        if self.active < self.cap and not self.waiting:
            self.active += 1
            self.peak = max(self.peak, self.active)
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():  # we were handed the slot as we were cancelled
                self.release()
            raise

    def release(self) -> None:
        while self._heap:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)  # hand the slot over: active stays the same
                return
        self.active -= 1

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()


# -------------------------------------------------------------------- the client
@dataclass
class Reply:
    text: str
    endpoint: str
    ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking: bool = False
    ok: bool = True
    error: str = ""


@dataclass
class SessionStats:
    calls: int = 0
    thinking_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failovers: int = 0
    errors: int = 0
    rate_limited: int = 0
    endpoints: set = field(default_factory=set)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, endpoints: list[str] | tuple[str, ...], model: str = "nemotron", cap: int = 4,
                 api_key: str | None = None, log_path: Path | None = None, timeout_s: float = 180.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not endpoints:
            raise ValueError("no LLM endpoints")
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.model, self.api_key = model, api_key
        self.gates = {e: PriorityGate(cap) for e in self.endpoints}
        self.sticky: dict[str, str] = {}
        self.stats: dict[str, SessionStats] = {}
        self.log_path = Path(log_path) if log_path else None
        self.client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self.client.aclose()

    def pick(self, session: str) -> str:
        if session not in self.sticky:
            self.sticky[session] = min(self.endpoints, key=lambda e: (self.gates[e].load, self.endpoints.index(e)))
        return self.sticky[session]

    def _log(self, row: dict) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    async def chat(self, messages: list[dict], session: str, priority: int = 5, thinking: bool = False,
                   max_tokens: int = 600, temperature: float = 0.0, who: str = "") -> Reply:
        st = self.stats.setdefault(session, SessionStats())
        first = self.pick(session)
        order = [first] + [e for e in self.endpoints if e != first]
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens + (2048 if thinking else 0), "stream": False,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        last, i, limited = "", 0, 0
        while i < min(2, len(order)):
            ep = order[i]
            headers = {"X-Session-Id": session}
            key = key_for(self.api_key, self.endpoints, ep)
            if key:
                headers["Authorization"] = f"Bearer {key}"
            gate = self.gates[ep]
            await gate.acquire(priority)
            t0 = time.perf_counter()
            try:
                r = await self.client.post(f"{ep}/chat/completions", json=body, headers=headers)
                ms = (time.perf_counter() - t0) * 1000
                if r.status_code >= 500 or r.status_code == 429:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:  # our request is wrong: another endpoint will not help
                    st.errors += 1
                    self._log({"ts": time.time(), "session": session, "who": who, "endpoint": ep, "ok": False, "ms": round(ms), "error": f"HTTP {r.status_code}"})
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                usage = data.get("usage") or {}
                reply = Reply(text=text, endpoint=ep, ms=ms, prompt_tokens=int(usage.get("prompt_tokens") or 0),
                              completion_tokens=int(usage.get("completion_tokens") or 0), thinking=thinking)
                st.calls += 1
                st.thinking_calls += bool(thinking)
                st.prompt_tokens += reply.prompt_tokens
                st.completion_tokens += reply.completion_tokens
                st.endpoints.add(ep)
                self._log({"ts": time.time(), "session": session, "who": who, "endpoint": ep, "ok": True,
                           "ms": round(ms), "prompt_tokens": reply.prompt_tokens,
                           "completion_tokens": reply.completion_tokens, "thinking": thinking, "priority": priority})
                return reply
            except LLMError as e:
                last = str(e)
                if not last.startswith("HTTP 5") and not last.startswith("HTTP 429"):
                    raise
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
                last = f"{e.__class__.__name__}: {e}"
            finally:
                gate.release()
            st.errors += 1
            self._log({"ts": time.time(), "session": session, "who": who, "endpoint": ep, "ok": False, "error": last[:300]})
            if last.startswith("HTTP 429") and limited < RATE_LIMIT_RETRIES:  # rate limited: back off, same endpoint
                st.rate_limited += 1
                await asyncio.sleep(RATE_LIMIT_BACKOFF_S * 2 ** limited)
                limited += 1
                continue
            if i == 0 and len(order) > 1:  # fail over: the incident moves for good
                st.failovers += 1
                self.sticky[session] = order[1]
            i += 1
        raise LLMError(f"all LLM endpoints failed; last: {last}")
