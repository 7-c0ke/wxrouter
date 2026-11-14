"""One live event stream for the Control Room's Events card.

Sources, each a `src` on the event:
  telemetry  changes seen in the WhatIf recorder's samples (whatif/telemetry.py, one a second): the router
             reachable or not, a box marked down or back up by the router, a box's /health code changing,
             the probe forecast failing or stalling, failovers, the error rate jumping, supervisor jobs and
             servers changing state. The recorder is the only thing polling the stack for these.
  fuzzball   the Fuzzball workflow's own stage events (`fuzzball workflow events <id> -o json`).
  incident   WhatIf diagnoses run from the page, and remedies applied.

Routing switches (with the router's reason), routing changes and actions on the page are logged by the page
itself from data it already has; they are not repeated here.

Every event gets a `seq`; the page asks for the ones after the last it has (GET /api/events?after=N).
Nothing is estimated: each line is a change between two samples, or one Fuzzball event as reported.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".fb_deploy.json"
ERR_PCT_HIGH = 5.0          # error rate above this (last 2 s) is an event; back under 1% is another
PROBE_STALL_MS = 1000.0     # same threshold as whatif/tools.py


class EventHub:
    def __init__(self, keep: int = 1000):
        self.events: deque[dict] = deque(maxlen=keep)
        self.seq = 0

    def add(self, src: str, level: str, text: str, ts: Optional[float] = None, **extra) -> dict:
        """level: bad (something broke), up (it came back / started), end (it finished), info."""
        self.seq += 1
        e = {"seq": self.seq, "ts": round(ts if ts is not None else time.time(), 3), "src": src,
             "level": level, "text": text, **extra}
        self.events.append(e)
        return e

    def since(self, after: int = 0) -> list[dict]:
        return [e for e in self.events if e["seq"] > after]


# -------------------------------------------------------------------- telemetry: changes between samples
class TelemetryEvents:
    """Feed it the recorder's samples in order (`observe`); it adds an event for each change worth a line."""

    def __init__(self, hub: EventHub):
        self.hub = hub
        self.prev: Optional[dict] = None
        self.err_high = False
        self.probe_bad: dict[str, str] = {}   # box -> what is wrong with its probe now ("" or absent = fine)

    def _add(self, level: str, text: str, t: float, box: Optional[str] = None) -> None:
        self.hub.add("telemetry", level, text, t, **({"box": box} if box else {}))

    def observe(self, s: dict) -> None:
        p, t = self.prev, s.get("t") or time.time()
        self.prev = s
        r = s.get("router")
        if p is None:
            if r is None:
                self._add("bad", "router not answering", t)
            return
        pr = p.get("router")
        # router reachable or not
        if pr is not None and r is None:
            self._add("bad", "router not answering", t)
        elif pr is None and r is not None:
            self._add("up", "router answering again", t)
        # router's view of each box, failovers
        if pr and r:
            for box, b in (r.get("backends") or {}).items():
                was = (pr.get("backends") or {}).get(box)
                if was and was.get("healthy") != b.get("healthy"):
                    self._add("up" if b.get("healthy") else "bad",
                              f"router marked {box} {'healthy: it gets traffic again' if b.get('healthy') else 'down: no traffic to it'}",
                              t, box)
            fo = (r.get("failovers") or 0) - (pr.get("failovers") or 0)
            if fo > 0:
                self._add("info", f"{fo} failover{'s' if fo > 1 else ''}: request{'s' if fo > 1 else ''} retried on the other box", t)
        # each box's own /health
        for box, b in (s.get("boxes") or {}).items():
            code = (b.get("health") or {}).get("code")
            was = ((p.get("boxes") or {}).get(box) or {}).get("health", {}).get("code")
            if was is not None and code != was:
                if code == 200:
                    self._add("up", f"{box} /health answers again (HTTP 200)", t, box)
                else:
                    self._add("bad", f"{box} /health {'no answer' if not code else f'HTTP {code}'}", t, box)
        # the probe forecast: failing, timing out or stalling is one line when it starts and one when it ends
        # (the recorder has probe news only every 2 s; a box without news keeps its state)
        for box, pb in (s.get("probe") or {}).items():
            if "pending_ms" in pb:
                bad = f"stalled: no answer after {pb['pending_ms'] / 1000:.1f} s" if pb["pending_ms"] > PROBE_STALL_MS else None
                if bad is None:
                    continue            # still waiting, not long yet: no news
            elif pb.get("timeout"):
                bad = "timed out"
            elif pb.get("code") not in (200, None):
                bad = "failed: " + ("no answer" if not pb.get("code") else f"HTTP {pb['code']}")
            else:
                bad = ""
            was = self.probe_bad.get(box, "")
            kind = lambda x: x.split(":")[0]
            if bad and kind(bad) != kind(was):
                self._add("bad", f"{box} probe forecast {bad}", t, box)
            elif not bad and was:
                self._add("up", f"{box} probe forecast answers again", t, box)
            self.probe_bad[box] = bad
        # user-facing error rate
        tr = s.get("traffic") or {}
        pct = tr.get("err_pct") or 0.0
        if not self.err_high and pct > ERR_PCT_HIGH:
            self.err_high = True
            self._add("bad", f"errors {pct:.0f}% of requests (last 2 s), p95 {tr.get('p95_ms') or '–'} ms", t)
        elif self.err_high and pct < 1.0 and (tr.get("rps") or 0) > 0:
            self.err_high = False
            self._add("up", "errors back under 1% of requests", t)
        # supervisors: servers and jobs (fine-tune) changing state
        for box, j in (s.get("jobs") or {}).items():
            was = (p.get("jobs") or {}).get(box)
            if not was:
                continue
            for group in ("servers", "jobs"):
                for name, state in (j.get(group) or {}).items():
                    old = (was.get(group) or {}).get(name)
                    if old is not None and state != old:
                        bad = state in ("failed", "backoff", "exited", "stopped") and group == "servers" or state == "failed"
                        self._add("bad" if bad else "up" if state in ("running", "started") else "end",
                                  f"{box} {'server' if group == 'servers' else 'job'} {name}: {old} -> {state}", t, box)


# -------------------------------------------------------------------- Fuzzball workflow events
PREFIX = "STAGE_EVENT_TYPE_"
REPEATING = {"RESERVATION_HELD", "WAITING_ON_DEPENDENCY", "PRIORITY_SET", "UNSPECIFIED"}
BAD = ("FAIL", "ERROR", "CANCEL", "EVICT", "OOM", "KILL")
UP = ("STARTED", "READY", "COMPLETED", "CACHE_HIT", "CREATED", "SATISFIED", "ALLOCATED")


def find_cli() -> Optional[str]:
    for cand in (os.environ.get("FUZZBALL"), shutil.which("fuzzball"), str(Path.home() / "bin" / "fuzzball")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def workflow_from_state(state_file: Path = STATE) -> Optional[str]:
    """The workflow `deploy/fb.py up` last started, if any."""
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        return None
    for target in ("real", *state):
        entry = state.get(target)
        if isinstance(entry, dict) and entry.get("id"):
            return entry["id"]
    return None


def fuzzball_level(event_type: str) -> str:
    t = event_type.upper()
    if any(w in t for w in BAD):
        return "bad"
    if t.endswith("EXITED") or t.endswith("FINISHED"):
        return "end"
    if any(w in t for w in UP):
        return "up"
    return "info"


def epoch(ts: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class FuzzballEvents:
    """Every `interval_s`: `fuzzball workflow events <id> -o json`, new events only, into the hub.
    "Reservation held", "waiting for ..." and the like repeat while a stage waits: shown once per stage
    until something else happens to it."""

    def __init__(self, hub: EventHub, workflow: Optional[str], cli: Optional[str], interval_s: float = 5.0,
                 tail: int = 1000):
        self.hub, self.workflow, self.cli, self.interval_s, self.tail = hub, workflow, cli, interval_s, tail
        self.seen: set[tuple] = set()
        self.last_shown: dict[str, tuple] = {}
        self.status = "off" if not (workflow and cli) else "starting"
        self.reason = ("no fuzzball CLI on this machine" if not cli
                       else "no workflow id: pass --fuzzball-workflow or run deploy/fb.py up" if not workflow else "")
        self.polled_at: Optional[float] = None

    def info(self) -> dict:
        return {"workflow": self.workflow, "status": self.status, "reason": self.reason, "polled_at": self.polled_at}

    def ingest(self, events: list[dict]) -> int:
        n = 0
        for e in sorted(events, key=lambda e: e.get("timestamp") or ""):
            etype = str(e.get("event_type") or "").removeprefix(PREFIX)
            key = (e.get("timestamp"), e.get("stage_id"), etype, e.get("message"))
            if key in self.seen:
                continue
            self.seen.add(key)
            stage = f"{e.get('stage_kind', '?')}:{e.get('stage_name', '?')}"
            shown = (etype, e.get("message"))
            if etype in REPEATING and self.last_shown.get(stage) == shown:
                continue
            self.last_shown[stage] = shown
            kind = str(e.get("stage_kind") or "").lower()
            name = "workflow" if kind == "workflow" else (e.get("stage_name") or "")
            self.hub.add("fuzzball", fuzzball_level(etype), e.get("message") or etype.replace("_", " ").lower(),
                         epoch(e.get("timestamp")), stage=name, stage_kind=kind, type=etype.lower())
            n += 1
        return n

    async def fetch(self) -> list[dict]:
        # -n: without it the CLI returns only the OLDEST 100 events, so a busy workflow's newest events would
        # never show. The newest TAIL are asked for every time; the ones already seen are dropped in ingest().
        p = await asyncio.create_subprocess_exec(self.cli, "workflow", "events", self.workflow, "-n", str(self.tail), "-o", "json",
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(p.communicate(), timeout=30)
        except asyncio.TimeoutError:
            p.kill()
            raise RuntimeError("fuzzball did not answer within 30 s")
        if p.returncode != 0:
            lines = (err or out).decode(errors="replace").strip().splitlines()
            raise RuntimeError(lines[-1][:200] if lines else f"fuzzball exited {p.returncode}")
        return json.loads(out or b"{}").get("events") or []

    async def poll_once(self) -> None:
        was = self.status
        try:
            self.ingest(await self.fetch())
            self.status, self.reason = "ok", ""
            if was == "unreachable":
                self.hub.add("fuzzball", "up", "Fuzzball API reachable again")
        except (RuntimeError, ValueError, OSError) as e:
            reason = str(e)
            if "connection refused" in reason or "Unavailable" in reason:
                reason = "Fuzzball API unreachable from this machine"
            if was != "unreachable":
                self.hub.add("fuzzball", "bad", reason)
            self.status, self.reason = "unreachable", reason
        self.polled_at = time.time()

    async def run(self) -> None:
        if self.status == "off":
            return
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval_s)
