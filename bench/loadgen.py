#!/usr/bin/env python3
"""Load generator. Sends a seeded, open-loop stream of requests at the router (or at one backend)
and writes what happened: one JSON line per request, plus a summary.

Run it from the repo root. It needs only the standard library.

    python bench/loadgen.py --target http://127.0.0.1:8100 --scenario contention --duration 60 \
        --rate 4 --seed 1 --kind forecast --policy adaptive \
        --admin-a http://127.0.0.1:8101 --admin-b http://127.0.0.1:8102 --out bench/results/run.jsonl

    python bench/loadgen.py --target http://127.0.0.1:8100 --kind llm --stream --sessions 8 \
        --duration 30 --rate 2 --out bench/results/llm.jsonl

Written at the hackathon (Nov 13-14, 2026).

The method, so the numbers can be trusted:

- The schedule comes first. Arrival times and request bodies are all made up front from --seed
  alone. Two runs with the same seed send the same requests at the same offsets, whatever the
  servers answered. That is what lets us compare routing policies fairly.
- Arrivals are a Poisson process at --rate requests per second.
- Open loop. Each request is sent at its time on its own thread. It never waits for an earlier
  answer. If more than --max-inflight requests are in flight, the new one is not sent and is
  recorded with status "shed".
- Forecast requests (POST /v1/forecast, Chicago O'Hare, init "latest"): 70% small (1 member,
  lead 24 or 48 h), 25% medium (4 to 8 members, lead 72 h), 5% large (20 members, lead 120 h).
  That is about 42 units of work per request on average. At 10 ms per unit, --rate 4 keeps
  about 1.7 nodes busy: one node alone falls behind, two are nearly full. Pick --rate with that
  in mind.
- LLM requests (POST /v1/chat/completions, max_tokens 48, temperature 0): each belongs to one of
  --sessions sessions and carries the header X-Session-Id. A session's history grows by one user
  turn (about 200 characters of seeded filler) and one assistant turn each time, so the prompt
  keeps its prefix. The assistant turns in the history are seeded filler too, not the real
  answers. That keeps the bodies independent of what the servers said. After 12 turns a session
  starts a new conversation, so prompts stay small. With --stream, time to first byte is measured.
  The "model" field is --model (vLLM wants the name of the model it serves).
- --kind mixed: half forecast, half LLM.
- Scenarios are timed events. D is the duration.
    steady      nothing
    contention  at D/3: POST <admin-b>/admin/contention {"factor":1.8}; at 2D/3: {"factor":1.0}
    failure     at D/2: POST <admin-b>/admin/fail {"mode":"down"}
    recovery    at D/3: fail "down"; at 2D/3: fail "ok"
    burst       the arrival rate is 4x between D/3 and D/2
  More events can come from --events FILE.json, a JSON list. Each entry is
  {"at_s": 20, "http": {"url": "...", "method": "POST", "json": {...}}} or
  {"at_s": 20, "cmd": ["bash", "-c", "..."]}. Use this on the real machines, where the event is
  starting or killing a job. An entry may carry "name"; name it "fail" to get time_to_recover_s.
  A command is started at its time and not waited for. Each event's actual time is recorded.
- Before the run: --policy NAME is set on the router (POST /router/policy), then POST /admin/reset
  goes to each admin URL given, then we wait (at most 20 s) until the router calls every backend
  healthy, so every run starts from the same state.

What comes out:

- --out FILE.jsonl: one line per request, in schedule order. Fields: t_start (seconds since the
  run began), latency_ms, ttfb_ms (streaming only), status (the HTTP code, or "timeout",
  "refused", "shed", "error"), backend (X-Backend), retries (X-Router-Retries), affinity
  (X-Router-Affinity), kind, units (members x steps for forecasts), session. Also: i (the
  request's number in the schedule), t_sched (when it was due), prompt_tokens and cached_tokens
  (LLM, when the answer reports them) and error (a short text when something went wrong).
- FILE.jsonl.summary.json, also printed as a short table.
- "ok" means an HTTP 2xx answer, read to the end. Everything else is an error, shed included.
- Latency runs from just before connecting until the last byte of the answer. Latency numbers
  are over ok requests only.
- Percentiles use the nearest-rank method on the sorted values: the p-th percentile of N values
  is the value at rank ceil(p/100 x N), counting from 1. Nothing is interpolated.
- Throughput is the number of ok answers that arrived within the duration, divided by the
  duration.
- time_to_recover_s: seconds from the "fail" event until the errors stop. That is the start of
  the first stretch of at least 5 seconds in which no request sent got an error or a timeout
  (and at least one got an ok answer). null when the run holds no such stretch.
"""

import argparse
import datetime
import http.client
import json
import os
import random
import math
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

SCENARIOS = ("steady", "contention", "failure", "recovery", "burst")
KINDS = ("forecast", "llm", "mixed")
OHARE = {"lat": 41.9602, "lon": -87.9316}
CONTENTION_FACTOR = 1.8     # as measured on a GB10: training beside serving
BURST_FACTOR = 4.0
LLM_MAX_TOKENS = 48
LLM_MAX_TURNS = 12          # then a session starts a new conversation, so prompts stay small
RECOVER_WINDOW_S = 5.0      # time_to_recover_s looks for this many clean seconds
SETTLE_MAX_S = 20.0         # longest wait for the router to call every backend healthy
CONTROL_TIMEOUT_S = 10.0    # for the policy, admin, status and event calls
CMD_GRACE_S = 5.0           # at the end, how long to wait for event commands still running
LAG_WARN_MS = 50.0          # warn when the generator itself was this late sending a request

WORDS = ("the wind cold front rain snow over lake airport runway forecast model ensemble member "
         "tomorrow morning evening pressure falling rising gusts clouds clear fog ice warm air "
         "mass jet stream ridge trough storm track shift north south east west visibility ceiling "
         "delay crew fuel plan update brief check trend range spread mean risk low high near "
         "normal").split()


class SetupError(Exception):
    """The run could not start (bad option, target down, policy or admin call refused)."""


# ---------------------------------------------------------------- small helpers

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_ok_status(status):
    return is_number(status) and 200 <= status < 300


def is_ok(record):
    """True when the request got an HTTP 2xx answer."""
    return is_ok_status(record.get("status"))


def percentile(values, p):
    """Nearest-rank percentile: the value at rank ceil(p/100 x N) of the sorted values.

    p is a whole number of percent. Returns None for an empty list.
    """
    data = sorted(values)
    if not data:
        return None
    rank = -((-int(p) * len(data)) // 100)          # ceil(p x N / 100) in whole numbers
    return data[min(max(rank, 1), len(data)) - 1]


def latency_stats(values):
    """p50, p95, p99 and mean of a list of milliseconds. All None when the list is empty."""
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    return {"p50": round(percentile(values, 50), 2), "p95": round(percentile(values, 95), 2),
            "p99": round(percentile(values, 99), 2), "mean": round(sum(values) / float(len(values)), 2)}


def sleep_until(when):
    """Sleep until the monotonic clock reads `when`, to within a fraction of a millisecond.

    One long sleep can wake late (5 ms late on a Mac). So it sleeps half of what is left, again
    and again, and each nap can only be a little late.
    """
    while True:
        left = when - time.monotonic()
        if left <= 0:
            return
        time.sleep(left / 2.0 if left > 0.0002 else 0)


# ---------------------------------------------------------------- the schedule (from the seed alone)

def arrival_times(rng, duration, rate, burst=False):
    """Poisson arrival times in [0, duration), in seconds.

    With burst, the rate is BURST_FACTOR times higher between duration/3 and duration/2.
    Method: draw gaps of a rate-1 Poisson process and stretch time by the rate in force, which
    is exact for a rate that changes in steps. Only rng.random() is used, so the same seed gives
    the same times on every Python version.
    """
    d = float(duration)
    if burst:
        pieces = [(0.0, d / 3, rate), (d / 3, d / 2, rate * BURST_FACTOR), (d / 2, d, rate)]
    else:
        pieces = [(0.0, d, rate)]
    times = []
    need = -math.log(1.0 - rng.random())             # expected arrivals to wait for the next one
    for start, end, piece_rate in pieces:
        t = start
        while True:
            step = need / piece_rate
            if t + step >= end:
                need -= (end - t) * piece_rate       # carry what is left into the next piece
                break
            t += step
            times.append(t)
            need = -math.log(1.0 - rng.random())
    return times


def filler(rng, n_chars):
    """About n_chars characters of seeded plain words."""
    words, size = [], 0
    while size < n_chars:
        word = WORDS[int(rng.random() * len(WORDS))]
        words.append(word)
        size += len(word) + 1
    return " ".join(words)


def forecast_request(rng, request_id):
    """One /v1/forecast body and its cost in units (members x 6-hour steps)."""
    u = rng.random()
    if u < 0.70:                                     # small
        members, lead = 1, (24 if rng.random() < 0.5 else 48)
    elif u < 0.95:                                   # medium
        members, lead = 4 + int(rng.random() * 5), 72
    else:                                            # large
        members, lead = 20, 120
    body = {"init": "latest", "lead_hours": lead, "members": members, "member_offset": 0,
            "point": dict(OHARE), "with_truth": False, "request_id": request_id}
    return body, members * (lead // 6)


def llm_request(rng, histories, sessions, seed, stream, model):
    """The next /v1/chat/completions body of a randomly picked session, and the session's id.

    The new prompt is the session's last prompt plus an assistant turn and a user turn, so it
    keeps its prefix. Both turns are seeded filler: the real answers never enter the schedule.
    """
    n = int(rng.random() * sessions)
    hist = histories.setdefault(n, {"turns": 0, "messages": []})
    if hist["turns"] >= LLM_MAX_TURNS:               # a new conversation in the same session
        hist["turns"], hist["messages"] = 0, []
    if hist["messages"]:
        hist["messages"].append({"role": "assistant", "content": filler(rng, 160)})
    hist["turns"] += 1
    text = "Turn %d. Answer in three sentences. Notes: %s" % (hist["turns"], filler(rng, 155))
    hist["messages"].append({"role": "user", "content": text})
    body = {"model": model, "messages": list(hist["messages"]), "max_tokens": LLM_MAX_TOKENS,
            "temperature": 0, "stream": bool(stream)}
    if stream:
        body["stream_options"] = {"include_usage": True}     # so the last chunk reports token counts
    return body, "seed%d-s%02d" % (seed, n)


def build_schedule(seed, duration, rate, kind, scenario="steady", sessions=8, stream=False,
                   model="default"):
    """Every request of a run, made from the seed alone. A list of dicts in send order.

    Each item: i, t (seconds after the start), kind, path, body, headers, units, session.
    The arrival times do not depend on the kind, so forecast, llm and mixed runs with one seed
    share them.
    """
    def rng_for(name):
        return random.Random("loadgen/%d/%s" % (seed, name))

    times = arrival_times(rng_for("arrivals"), duration, rate, burst=(scenario == "burst"))
    kind_rng, forecast_rng, llm_rng = rng_for("kind"), rng_for("forecast"), rng_for("llm")
    histories = {}
    schedule = []
    for i, t in enumerate(times):
        coin = kind_rng.random()
        this = kind if kind != "mixed" else ("forecast" if coin < 0.5 else "llm")
        if this == "forecast":
            body, units = forecast_request(forecast_rng, "lg%d-%05d" % (seed, i))
            item = {"path": "/v1/forecast", "headers": {}, "units": units, "session": None}
        else:
            body, session = llm_request(llm_rng, histories, sessions, seed, stream, model)
            item = {"path": "/v1/chat/completions", "headers": {"X-Session-Id": session},
                    "units": None, "session": session}
        item.update({"i": i, "t": t, "kind": this, "body": body})
        schedule.append(item)
    return schedule


# ---------------------------------------------------------------- events

def scenario_events(scenario, duration, admin_b):
    """The timed events of a named scenario. They all go to node B's admin hooks.

    Each has a name (the summary uses it) and a label (the charts print it).
    """
    d = float(duration)

    def post(at_s, name, label, path, body):
        return {"at_s": at_s, "name": name, "label": label,
                "http": {"url": "%s%s" % (admin_b, path), "method": "POST", "json": body}}

    if scenario == "contention":
        return [post(d / 3, "contention_on", "B slows down", "/admin/contention", {"factor": CONTENTION_FACTOR}),
                post(2 * d / 3, "contention_off", "B back to normal", "/admin/contention", {"factor": 1.0})]
    if scenario == "failure":
        return [post(d / 2, "fail", "B dies", "/admin/fail", {"mode": "down"})]
    if scenario == "recovery":
        return [post(d / 3, "fail", "B dies", "/admin/fail", {"mode": "down"}),
                post(2 * d / 3, "recover", "B comes back", "/admin/fail", {"mode": "ok"})]
    if scenario == "burst":         # nothing to call: the schedule itself is denser. These mark the chart.
        return [{"at_s": d / 3, "name": "burst_start", "label": "burst starts"},
                {"at_s": d / 2, "name": "burst_end", "label": "burst ends"}]
    return []


def load_events_file(path, duration):
    """Read extra events from a JSON file and check them. See the top of this file for the shape."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise SetupError("cannot read the events file %s: %s" % (path, e))
    if not isinstance(data, list):
        raise SetupError("the events file %s must hold a JSON list" % path)
    events = []
    for n, entry in enumerate(data):
        where = "event %d in %s" % (n + 1, path)
        if not isinstance(entry, dict) or not is_number(entry.get("at_s")):
            raise SetupError('%s needs a number "at_s"' % where)
        if not 0 <= entry["at_s"] <= duration:
            raise SetupError("%s: at_s %s is outside the run (0 to %g s)" % (where, entry["at_s"], duration))
        http_part, cmd = entry.get("http"), entry.get("cmd")
        if (http_part is None) == (cmd is None):
            raise SetupError('%s needs "http" or "cmd" (one of them)' % where)
        event = {"at_s": float(entry["at_s"])}
        if http_part is not None:
            if not isinstance(http_part, dict) or not isinstance(http_part.get("url"), str):
                raise SetupError('%s: "http" needs a "url"' % where)
            split_url(http_part["url"])
            event["http"] = http_part
        else:
            if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
                raise SetupError('%s: "cmd" must be a list of strings' % where)
            event["cmd"] = cmd
        event["name"] = str(entry.get("name") or ("http" if http_part is not None else "cmd"))
        events.append(event)
    return events


def run_events(events, t0, log):
    """Fire each event at its time and note what happened. Runs on its own thread."""
    for event in sorted(events, key=lambda e: e["at_s"]):
        sleep_until(t0 + event["at_s"])
        entry = {"name": event["name"], "label": event.get("label") or event["name"],
                 "at_s": round(event["at_s"], 3), "t_actual": round(time.monotonic() - t0, 3),
                 "what": "mark", "ok": True, "result": None}
        try:
            if event.get("http"):
                call = event["http"]
                method = str(call.get("method") or "POST").upper()
                entry["what"] = "%s %s %s" % (method, call["url"], json.dumps(call.get("json")))
                status, _, error = call_json(method, call["url"], call.get("json"))
                entry["ok"] = is_ok_status(status)
                entry["result"] = "http %s" % status if is_number(status) else "%s: %s" % (status, error)
            elif event.get("cmd"):
                entry["what"] = " ".join(event["cmd"])
                entry["proc"] = subprocess.Popen(event["cmd"], stdin=subprocess.DEVNULL,
                                                 stdout=subprocess.DEVNULL)
                entry["result"] = "started"
        except Exception as e:      # a broken event must show up in the summary, not kill the thread
            entry["ok"], entry["result"] = False, "%s: %s" % (type(e).__name__, e)
        log.append(entry)


def finish_commands(log):
    """Note how each event command ended. One still running is left alone, and said so."""
    deadline = time.monotonic() + CMD_GRACE_S
    for entry in log:
        proc = entry.pop("proc", None)
        if proc is None:
            continue
        try:
            code = proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            entry["ok"], entry["result"] = (code == 0), "exit %d" % code
        except subprocess.TimeoutExpired:
            entry["result"] = "still running (pid %d), left alone" % proc.pid


# ---------------------------------------------------------------- HTTP

def split_url(url):
    """'http://host:8100/x?y' -> (host, port, '/x?y'). Plain http only."""
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        host, port, parts = None, None, None
    if not host or parts.scheme != "http":
        raise SetupError("not a plain http address: %r (expected something like http://127.0.0.1:8100)" % (url,))
    path = parts.path + ("?" + parts.query if parts.query else "")
    return host, port or 80, path


def exchange(host, port, method, path, body=None, headers=None, timeout=CONTROL_TIMEOUT_S):
    """One HTTP request on a fresh connection, with one time limit for the whole exchange.

    Returns a dict: status (the HTTP code, or "timeout", "refused", "error"), headers (names in
    lower case), body (bytes), ttfb_s (time to the first body byte), elapsed_s, error (text or None).
    """
    out = {"status": "error", "headers": {}, "body": b"", "ttfb_s": None, "elapsed_s": None, "error": None}
    chunks = []
    resp = None
    start = time.monotonic()
    deadline = start + timeout
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.connect()
        sock = conn.sock            # keep our own handle: http.client drops its own once the answer starts
        conn.request(method, path, body=body, headers=headers or {})
        sock.settimeout(max(deadline - time.monotonic(), 0.001))
        resp = conn.getresponse()
        out["headers"] = dict((name.lower(), value) for name, value in resp.getheaders())
        while not resp.isclosed():
            left = deadline - time.monotonic()
            if left <= 0:
                raise socket.timeout("time limit reached")
            try:
                sock.settimeout(left)   # so a slow trickle cannot outlive the limit
            except OSError:
                pass                    # the socket is gone already; the read below will tell
            chunk = resp.read1(65536)
            if not chunk:
                break
            if out["ttfb_s"] is None:
                out["ttfb_s"] = time.monotonic() - start
            chunks.append(chunk)
        if resp.length:             # the server closed before sending all it promised
            raise http.client.IncompleteRead(b"", resp.length)
        out["status"] = resp.status
    except (socket.timeout, TimeoutError):
        out["status"], out["error"] = "timeout", "no full answer within %g s" % timeout
    except ConnectionRefusedError as e:
        out["status"], out["error"] = "refused", str(e)
    except (OSError, http.client.HTTPException) as e:
        out["status"], out["error"] = "error", "%s: %s" % (type(e).__name__, e)
    finally:
        if resp is not None:
            resp.close()
        conn.close()
    out["body"] = b"".join(chunks)
    out["elapsed_s"] = time.monotonic() - start
    return out


def call_json(method, url, body=None):
    """A small control call (policy, admin hook, status). Returns (status, parsed JSON or None, error)."""
    host, port, path = split_url(url)
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Connection": "close"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    got = exchange(host, port, method, path or "/", data, headers, CONTROL_TIMEOUT_S)
    try:
        parsed = json.loads(got["body"].decode("utf-8"))
    except ValueError:
        parsed = None
    return got["status"], parsed, got["error"]


def router_status(target):
    """GET /router/status. Returns (status dict or None, problem or None).

    A plain backend has no such page: that gives (None, None). A target that does not answer at
    all gives (None, "why").
    """
    status, data, error = call_json("GET", target + "/router/status")
    if not is_number(status):
        return None, "%s (%s)" % (status, error)
    if status == 200 and isinstance(data, dict):
        return data, None
    return None, None


def unhealthy_backends(status, pools):
    """Names of the backends that /router/status calls unhealthy, in the given pools."""
    names = []
    all_pools = status.get("pools") if isinstance(status, dict) else None
    for pool in pools:
        entry = all_pools.get(pool) if isinstance(all_pools, dict) else None
        backends = entry.get("backends") if isinstance(entry, dict) else None
        for backend in backends if isinstance(backends, list) else []:
            if isinstance(backend, dict) and backend.get("healthy") is False:
                names.append("%s/%s" % (pool, backend.get("name")))
    return names


def wait_healthy(target, pools, say):
    """Wait until the router calls every backend of these pools healthy, at most SETTLE_MAX_S.

    Returns (the last status or None, seconds waited, names still unhealthy).
    """
    start = time.monotonic()
    told = False
    while True:
        status, _ = router_status(target)
        sick = unhealthy_backends(status, pools)
        waited = time.monotonic() - start
        if not sick or waited >= SETTLE_MAX_S:
            return status, waited, sick
        if not told:
            say("waiting until the router calls %s healthy again (at most %g s)" % (", ".join(sick), SETTLE_MAX_S))
            told = True
        time.sleep(0.25)


def counters_of(status):
    counters = status.get("counters") if isinstance(status, dict) else None
    return counters if isinstance(counters, dict) else None


# ---------------------------------------------------------------- sending

def new_record(item, t_start):
    """A record with the schedule's part filled in and the measured part still empty."""
    return {"t_start": round(t_start, 4), "latency_ms": None, "ttfb_ms": None, "status": None,
            "backend": None, "retries": None, "affinity": None, "kind": item["kind"],
            "units": item["units"], "session": item["session"], "i": item["i"],
            "t_sched": round(item["t"], 4), "prompt_tokens": None, "cached_tokens": None, "error": None}


def to_int(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def error_text(body):
    """A short reason from an error answer: its {"error": ...} if it has one, else its first bytes."""
    text = body.decode("utf-8", "replace").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("error"):
            reason = parsed["error"]
            if isinstance(reason, dict) and reason.get("message"):
                reason = reason["message"]
            text = reason if isinstance(reason, str) else json.dumps(reason)
    except ValueError:
        pass
    return text[:200] or None


def read_usage(body, streamed):
    """(prompt_tokens, cached_tokens) from an OpenAI-style answer. None where it does not say.

    Reads usage.prompt_tokens and usage.prompt_tokens_details.cached_tokens, or llama.cpp's
    timings.cache_n and timings.prompt_n. A streamed answer is searched chunk by chunk.
    """
    text = body.decode("utf-8", "replace")
    if streamed:
        pieces = [line[5:] for line in text.splitlines() if line.startswith("data:")]
    else:
        pieces = [text]
    prompt = cached = None
    for piece in pieces:
        try:
            obj = json.loads(piece)
        except ValueError:
            continue                                 # "[DONE]" and the like
        if not isinstance(obj, dict):
            continue
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else {}
        if is_number(usage.get("prompt_tokens")):
            prompt = usage["prompt_tokens"]
        if is_number(details.get("cached_tokens")):
            cached = details["cached_tokens"]
        elif is_number(timings.get("cache_n")):
            cached = timings["cache_n"]
            if prompt is None and is_number(timings.get("prompt_n")):
                prompt = timings["cache_n"] + timings["prompt_n"]
    return prompt, cached


def send_one(target_parts, item, timeout):
    """Send one scheduled request. Returns the measured fields of its record."""
    host, port, base = target_parts
    headers = {"Content-Type": "application/json", "Connection": "close"}
    headers.update(item["headers"])
    data = json.dumps(item["body"]).encode("utf-8")
    got = exchange(host, port, "POST", base.rstrip("/") + item["path"], data, headers, timeout)
    streamed = bool(item["body"].get("stream"))
    out = {"latency_ms": round(got["elapsed_s"] * 1000.0, 2), "status": got["status"],
           "backend": got["headers"].get("x-backend"),
           "retries": to_int(got["headers"].get("x-router-retries")),
           "affinity": got["headers"].get("x-router-affinity"), "error": got["error"]}
    if streamed and got["ttfb_s"] is not None:
        out["ttfb_ms"] = round(got["ttfb_s"] * 1000.0, 2)
    if is_ok_status(got["status"]):
        if item["kind"] == "llm":
            out["prompt_tokens"], out["cached_tokens"] = read_usage(got["body"], streamed)
    elif is_number(got["status"]):
        out["error"] = error_text(got["body"])
    return out


def send_schedule(target_parts, schedule, events, duration, timeout, max_inflight, say):
    """Send every request at its time (open loop) and fire the events.

    Returns (records in schedule order, the event log, seconds until the last answer).
    """
    lock = threading.Lock()
    inflight = [0]
    records = [None] * len(schedule)
    threads = []
    event_log = []
    t0 = time.monotonic()

    def worker(item):
        record = new_record(item, time.monotonic() - t0)
        try:
            record.update(send_one(target_parts, item, timeout))
        except Exception as e:      # never lose a record to a bug
            record.update({"status": "error", "error": "loadgen: %s: %s" % (type(e).__name__, e)})
        finally:
            with lock:
                inflight[0] -= 1
            records[item["i"]] = record

    def not_sent(item, status, why):
        record = new_record(item, time.monotonic() - t0)
        record.update({"status": status, "error": why})
        records[item["i"]] = record

    event_thread = threading.Thread(target=run_events, args=(events, t0, event_log), daemon=True)
    event_thread.start()
    for item in schedule:
        sleep_until(t0 + item["t"])
        with lock:
            room = inflight[0] < max_inflight
            if room:
                inflight[0] += 1
        if not room:
            not_sent(item, "shed", "not sent: %d requests were in flight" % max_inflight)
            continue
        thread = threading.Thread(target=worker, args=(item,), daemon=True)
        try:
            thread.start()
        except RuntimeError as e:
            with lock:
                inflight[0] -= 1
            not_sent(item, "error", "loadgen could not start a thread: %s" % e)
            continue
        threads.append(thread)
    sleep_until(t0 + duration)
    soon = time.monotonic() + 1.0
    for thread in threads:
        thread.join(max(0.0, soon - time.monotonic()))
    waiting = sum(1 for thread in threads if thread.is_alive())
    if waiting:
        say("the %g s are over; still waiting for %d answers (each has at most %g s from when it was sent)"
            % (duration, waiting, timeout))
    for thread in threads:
        thread.join()
    event_thread.join()
    finish_commands(event_log)
    return records, event_log, time.monotonic() - t0


# ---------------------------------------------------------------- the numbers

def time_to_recover(records, t_fail, duration, window=None):
    """Seconds from the fail event until the errors stop, or None when they do not.

    The errors have stopped at the start of the first clean stretch: at least `window` seconds
    (5 by default) in which no request sent went wrong and at least one got an ok answer.
    A stretch starts at the fail event or at a request that went wrong. It ends at the next
    request that went wrong, or with the run. Requests count by the time they were sent.
    """
    window = RECOVER_WINDOW_S if window is None else window
    start, seen_ok = t_fail, False
    for t, ok in sorted((r["t_start"], is_ok(r)) for r in records if r["t_start"] >= t_fail):
        if ok:
            seen_ok = True
        elif seen_ok and t - start >= window:
            return round(start - t_fail, 3)          # the stretch this error ends was long enough
        else:
            start, seen_ok = t, False
    if seen_ok and duration - start >= window:       # the last stretch runs to the end of the run
        return round(start - t_fail, 3)
    return None


def window_p95(records, duration, width=5.0):
    """The p95 latency of ok requests per time window, by send time. Used for the charts.

    Returns a list of dicts: t0, t1, ok, errors, p95_ms (None when the window has no ok request).
    """
    count = max(1, int(math.ceil(duration / float(width) - 1e-9)))
    latencies = [[] for _ in range(count)]
    errors = [0] * count
    for r in records:
        k = min(count - 1, max(0, int(r["t_start"] // width)))
        if is_ok(r):
            latencies[k].append(r["latency_ms"])
        else:
            errors[k] += 1
    return [{"t0": k * width, "t1": min(float(duration), (k + 1) * width), "ok": len(latencies[k]),
             "errors": errors[k], "p95_ms": percentile(latencies[k], 95)} for k in range(count)]


def summarize(records, duration, events=None, router_before=None, router_after=None):
    """The numbers of one run, from its records. Latency numbers are over ok requests only.

    events is the event log (dicts with name and t_actual). router_before and router_after are
    the router's "counters" before and after the run, or None when the target has none.
    """
    events = list(events or [])
    good = [r for r in records if is_ok(r)]
    errors_by_type = {}
    for r in records:
        if not is_ok(r):
            key = str(r.get("status"))
            errors_by_type[key] = errors_by_type.get(key, 0) + 1
    errors = len(records) - len(good)
    in_time = sum(1 for r in good if r["t_start"] + r["latency_ms"] / 1000.0 <= duration)

    thirds = [[], [], []]
    for r in good:
        thirds[min(2, max(0, int(r["t_start"] / (duration / 3.0))))].append(r["latency_ms"])

    answered = {}
    for r in records:
        if r.get("backend"):
            answered[r["backend"]] = answered.get(r["backend"], 0) + 1
    share = dict((name, round(answered[name] / float(sum(answered.values())), 4)) for name in sorted(answered))

    by_kind = {}
    for kind in sorted(set(str(r.get("kind")) for r in records)):
        rows = [r for r in records if str(r.get("kind")) == kind]
        by_kind[kind] = {"requests": len(rows), "ok": sum(1 for r in rows if is_ok(r)),
                         "latency_ms": latency_stats([r["latency_ms"] for r in rows if is_ok(r)])}

    ttfb = [r["ttfb_ms"] for r in good if is_number(r.get("ttfb_ms"))]
    tokens = [(r["prompt_tokens"], r["cached_tokens"]) for r in good
              if is_number(r.get("prompt_tokens")) and is_number(r.get("cached_tokens"))]
    prompt_total = sum(p for p, _ in tokens)

    fails = [e for e in events if e.get("name") == "fail" and e.get("ok", True)]    # one that did not work is no failure
    fail_at = fails[0].get("t_actual", fails[0].get("at_s")) if fails else None
    lags = [(r["t_start"] - r["t_sched"]) * 1000.0 for r in records if is_number(r.get("t_sched"))]

    router = None
    if isinstance(router_before, dict) and isinstance(router_after, dict):
        delta = dict((key, router_after[key] - router_before[key]) for key in router_after
                     if is_number(router_after[key]) and is_number(router_before.get(key)))
        router = {"before": router_before, "after": router_after, "delta": delta}

    return {
        "requests": len(records),
        "ok": len(good),
        "errors": errors,
        "errors_by_type": errors_by_type,
        "error_rate": round(errors / float(len(records)), 4) if records else 0.0,
        "throughput_rps": round(in_time / float(duration), 3),
        "latency_ms": latency_stats([r["latency_ms"] for r in good]),
        "ttfb_ms": latency_stats(ttfb) if ttfb else None,
        "p95_by_third_ms": [percentile(part, 95) for part in thirds],
        "share": share,
        "retries": sum(r["retries"] for r in records if is_number(r.get("retries"))),
        "fail_event_s": fail_at,
        "time_to_recover_s": time_to_recover(records, fail_at, duration) if fail_at is not None else None,
        "cached_prompt_share": round(sum(c for _, c in tokens) / float(prompt_total), 4) if prompt_total > 0 else None,
        "by_kind": by_kind,
        "router": router,
        "events": events,
        "send_lag_ms_max": round(max(lags), 2) if lags else None,
    }


def num(value, digits=1):
    return "-" if value is None else "%.*f" % (digits, value)


def format_summary(summary):
    """The summary as a short table of text."""
    s = summary.get("settings") or {}
    lat = summary["latency_ms"]
    by_type = ", ".join("%s: %d" % pair for pair in sorted(summary["errors_by_type"].items()))
    lines = []
    if s:
        policy = s.get("policy") or ("%s (as the router had it)" % summary["router_policy"]
                                     if summary.get("router_policy") else "not set")
        lines.append("%s scenario, %s requests, %g per second for %g s, seed %s, policy %s"
                     % (s.get("scenario"), s.get("kind"), s.get("rate"), s.get("duration_s"), s.get("seed"), policy))
        lines.append("target %s" % s.get("target"))

    def row(name, text):
        lines.append("  %-14s %s" % (name, text))

    def stats(st):
        return "p50 %s   p95 %s   p99 %s   mean %s" % (num(st["p50"]), num(st["p95"]), num(st["p99"]), num(st["mean"]))

    row("requests", "%d" % summary["requests"])
    row("ok", "%d" % summary["ok"])
    row("errors", "%d%s" % (summary["errors"], "   (%s)" % by_type if by_type else ""))
    row("error rate", "%.2f %%" % (100.0 * summary["error_rate"]))
    row("throughput", "%.2f ok per second (ok answers that arrived within the duration)" % summary["throughput_rps"])
    row("latency ms", stats(lat))
    if len(summary["by_kind"]) > 1:
        for kind in sorted(summary["by_kind"]):
            row("  " + kind, "%s   (%d ok)" % (stats(summary["by_kind"][kind]["latency_ms"]), summary["by_kind"][kind]["ok"]))
    if summary.get("ttfb_ms"):
        row("first byte ms", stats(summary["ttfb_ms"]))
    row("p95 by third", " | ".join(num(v) for v in summary["p95_by_third_ms"]) + "   (ms, over ok requests)")
    row("share", ", ".join("%s %.1f %%" % (name, 100.0 * part) for name, part in sorted(summary["share"].items()))
        or "-   (no X-Backend header in the answers)")
    row("retries", "%d" % summary["retries"])
    if summary["fail_event_s"] is not None:
        row("recover s", num(summary["time_to_recover_s"], 2) if summary["time_to_recover_s"] is not None
            else "not within the run   (fail event at %.2f s)" % summary["fail_event_s"])
    if summary["cached_prompt_share"] is not None:
        row("cached prompt", "%.1f %% of prompt tokens" % (100.0 * summary["cached_prompt_share"]))
    if summary["router"]:
        row("router", ", ".join("%s %+d" % (key, value) for key, value in summary["router"]["delta"].items())
            + "   (counters, after minus before)")
    else:
        row("router", "-   (no /router/status at the target)")
    for event in summary["events"]:
        row("event", "%s at %.2f s: %s -> %s" % (event.get("name"), event.get("t_actual", event.get("at_s", 0.0)),
                                                 event.get("what", "?"), event.get("result") or "done"))
    for warning in summary.get("warnings") or []:
        lines.append("WARNING: %s" % warning)
    return "\n".join(lines)


# ---------------------------------------------------------------- one whole run

def check_options(opts):
    if opts.scenario not in SCENARIOS:
        raise SetupError("unknown scenario %r (known: %s)" % (opts.scenario, ", ".join(SCENARIOS)))
    if opts.kind not in KINDS:
        raise SetupError("unknown kind %r (known: %s)" % (opts.kind, ", ".join(KINDS)))
    for name in ("duration", "rate", "timeout", "sessions", "max_inflight"):
        if not getattr(opts, name) > 0:
            raise SetupError("--%s must be above 0" % name.replace("_", "-"))


def run(opts, quiet=False):
    """One whole run: prepare, send the schedule, write the two files. Returns (summary, records).

    opts is what build_parser() gives (see options()). Raises SetupError when the run cannot start.
    """
    def say(text):
        if not quiet:
            print(text, flush=True)

    check_options(opts)
    target = opts.target.rstrip("/")
    target_parts = split_url(target)
    admins = [(name, url.rstrip("/")) for name, url in (("A", opts.admin_a), ("B", opts.admin_b)) if url]
    for _, url in admins:
        split_url(url)
    admin_b = opts.admin_b.rstrip("/") if opts.admin_b else None
    if opts.scenario in ("contention", "failure", "recovery") and not admin_b:
        raise SetupError("the %s scenario posts to node B's admin hooks: give --admin-b URL. "
                         "On real machines use --scenario steady with --events FILE.json." % opts.scenario)
    events = scenario_events(opts.scenario, opts.duration, admin_b)
    if opts.events:
        events = events + load_events_file(opts.events, opts.duration)
    schedule = build_schedule(opts.seed, opts.duration, opts.rate, opts.kind, opts.scenario,
                              opts.sessions, opts.stream, opts.model)

    status, problem = router_status(target)
    if problem:
        raise SetupError("the target %s is not answering: %s" % (target, problem))
    if opts.policy:
        code, _, error = call_json("POST", target + "/router/policy", {"policy": opts.policy})
        if not is_ok_status(code):
            raise SetupError("could not set the policy %r at %s/router/policy: %s. Is the target the router?"
                             % (opts.policy, target, "HTTP %s" % code if is_number(code) else error))
    for name, url in admins:
        code, _, error = call_json("POST", url + "/admin/reset", {})
        if not is_ok_status(code):
            raise SetupError("could not reset node %s at %s/admin/reset: %s. Is it a mock started with --admin?"
                             % (name, url, "HTTP %s" % code if is_number(code) else error))
    pools = ["forecast", "llm"] if opts.kind == "mixed" else [opts.kind]
    status, waited, sick = wait_healthy(target, pools, say)

    warnings = []
    if sick:
        warnings.append("the run began with unhealthy backends: %s" % ", ".join(sick))
    if opts.policy and isinstance(status, dict) and status.get("policy") not in (None, opts.policy):
        warnings.append("asked for policy %s but the router reports %s" % (opts.policy, status.get("policy")))

    say("sending %d requests over %g s (%s, %s, seed %d)" % (len(schedule), opts.duration, opts.scenario, opts.kind, opts.seed))
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records, event_log, wall = send_schedule(target_parts, schedule, events, opts.duration,
                                             opts.timeout, opts.max_inflight, say)
    after, _ = router_status(target)

    summary = {
        "settings": {"target": target, "scenario": opts.scenario, "kind": opts.kind, "duration_s": opts.duration,
                     "rate": opts.rate, "seed": opts.seed, "policy": opts.policy, "stream": bool(opts.stream),
                     "sessions": opts.sessions, "max_inflight": opts.max_inflight, "timeout_s": opts.timeout,
                     "admin_a": opts.admin_a, "admin_b": opts.admin_b, "events_file": opts.events,
                     "model": opts.model},
        "started_at": started_at,
        "wall_s": round(wall, 2),
        "settle_s": round(waited, 2),
        "router_policy": status.get("policy") if isinstance(status, dict) else None,
    }
    summary.update(summarize(records, opts.duration, event_log, counters_of(status), counters_of(after)))
    for event in event_log:
        if not event["ok"]:
            warnings.append("the event %s did not work: %s -> %s" % (event["name"], event["what"], event["result"]))
    if summary["send_lag_ms_max"] is not None and summary["send_lag_ms_max"] > LAG_WARN_MS:
        warnings.append("the generator fell behind its schedule by up to %.0f ms: this machine is too busy"
                        % summary["send_lag_ms_max"])
    summary["warnings"] = warnings

    out = opts.out
    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    with open(out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    with open(out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    say(format_summary(summary))
    say("wrote %s\nwrote %s" % (out, out + ".summary.json"))
    return summary, records


def add_run_options(parser):
    """The options that describe a load. compare.py takes the same ones."""
    parser.add_argument("--target", metavar="URL", default=os.environ.get("ROUTER_URL", "http://127.0.0.1:8100"),
                        help="the router, or one backend (default: $ROUTER_URL or http://127.0.0.1:8100)")
    parser.add_argument("--duration", metavar="SECONDS", type=float, default=60.0,
                        help="how long requests keep arriving (default 60)")
    parser.add_argument("--rate", metavar="PER_SECOND", type=float, default=4.0,
                        help="mean requests per second (default 4)")
    parser.add_argument("--seed", type=int, default=1, help="same seed, same requests at the same times (default 1)")
    parser.add_argument("--kind", choices=KINDS, default="forecast", help="what to send (default forecast)")
    parser.add_argument("--admin-a", metavar="URL", help="node A's mock, for POST /admin/reset before the run")
    parser.add_argument("--admin-b", metavar="URL",
                        help="node B's mock: reset before the run, and the scenario's events go here")
    parser.add_argument("--events", metavar="FILE.json", help="extra timed events (HTTP calls or commands)")
    parser.add_argument("--stream", action="store_true", help="LLM: ask for a streamed answer and time the first byte")
    parser.add_argument("--sessions", metavar="N", type=int, default=8,
                        help="LLM: how many sessions take turns (default 8)")
    parser.add_argument("--model", metavar="NAME", default="default",
                        help='LLM: the "model" field of each request (default "default"; vLLM wants its model\'s name)')
    parser.add_argument("--max-inflight", metavar="N", type=int, default=256,
                        help="with this many requests in flight, new ones are shed (default 256)")
    parser.add_argument("--timeout", metavar="SECONDS", type=float, default=120.0,
                        help="time allowed per request, start to last byte (default 120)")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Send a seeded, open-loop load at the router (or one backend) and measure it.",
        epilog="Example: python bench/loadgen.py --target http://127.0.0.1:8100 --scenario contention "
               "--admin-a http://127.0.0.1:8101 --admin-b http://127.0.0.1:8102 --policy adaptive")
    add_run_options(parser)
    parser.add_argument("--scenario", choices=SCENARIOS, default="steady", help="the timed events (default steady)")
    parser.add_argument("--policy", metavar="NAME", help="set this routing policy on the router first")
    parser.add_argument("--out", metavar="FILE.jsonl", default="bench/results/run.jsonl",
                        help="one line per request; the summary goes to FILE.jsonl.summary.json "
                             "(default bench/results/run.jsonl)")
    return parser


def options(**changes):
    """The default options with some changed, for calling run() from Python: options(duration=5)."""
    opts = build_parser().parse_args([])
    for name, value in changes.items():
        if not hasattr(opts, name):
            raise SetupError("no such option: %s" % name)
        setattr(opts, name, value)
    return opts


def main(argv=None):
    opts = build_parser().parse_args(argv)
    try:
        run(opts)
    except SetupError as e:
        sys.stderr.write("loadgen: %s\n" % e)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("loadgen: stopped by Ctrl-C. Nothing was written.\n")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
