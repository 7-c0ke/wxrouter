#!/usr/bin/env python3
"""A stand-in for an LLM server (llama.cpp or vLLM). No model, no GPU.

It speaks enough of the OpenAI chat API for the router, the agent and the load
tests: POST /v1/chat/completions (one JSON body, or a stream of events),
GET /v1/models, GET /health and GET /metrics (Prometheus text).

Run it:
    python serve/mock_llm.py --node A --port 8001 --admin
    python serve/mock_llm.py --node B --port 8002 --flavor vllm --tps 60 --slots 4

Try it:
    curl -s http://127.0.0.1:8001/v1/chat/completions \
         -H 'Content-Type: application/json' \
         -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":16}'

How it behaves:
  * A "token" is 4 characters. A request takes
    (prompt tokens not in the cache) / --prefill-tps + (reply tokens) / --tps
    seconds, times the slowdown set by /admin/contention.
  * The reply is filler text that names the node, max_tokens long (default 64).
  * --slots N requests run at once. The rest wait in line ("deferred").
    Each slot remembers the last prompt it served. The start that a new prompt
    shares with it counts as cached and costs no time. A request takes the
    idle slot that shares the longest start (ties: the slot unused the longest).
  * With "tools" in the request and a user message last, the reply is one tool
    call. With a tool result last, the reply quotes that result. So an agent
    loop can be tested end to end.
  * /metrics uses llama.cpp's names, or vLLM's with --flavor vllm. The prompt
    token counter leaves out cached tokens: it counts what was really read.
    As in llama.cpp, the two tokens-per-second gauges cover the requests that
    finished since the last /metrics read, and are 0 if there were none.
  * --load-s N: for the first N seconds every request gets 503 "Loading model".
  * --admin turns on the test hooks:
        POST /admin/contention {"factor":1.8}    everything takes 1.8x as long
        POST /admin/fail {"mode":"down"}         every request gets 503
        POST /admin/fail {"mode":"hang"}         requests are never answered
        POST /admin/fail {"mode":"error"}        chat gets 500, /health stays 200
        POST /admin/fail {"mode":"ok"}           back to normal
        POST /admin/reset                        clear hooks, counters and caches
    "down" and "hang" also stop requests that are waiting or running.
  * --port 0 picks a free port. The first line on stdout is
    "listening on http://HOST:PORT". SIGTERM or Ctrl-C stops it, exit code 0.

Standard library only.
Written at the hackathon (Nov 13-14, 2026).
"""

import argparse
import collections
import json
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-llm"
CHARS_PER_TOKEN = 4
TOKENS_PER_EVENT = 4            # a stream carries this many tokens per event
DEFAULT_MAX_TOKENS = 64
MAX_TOKENS_LIMIT = 1000000      # keeps one reply to a few megabytes
MAX_BODY_BYTES = 32 * 1024 * 1024
FAIL_MODES = ("ok", "down", "hang", "error")
SCRIPT_LAT, SCRIPT_LON = 41.9602, -87.9316   # what the tool script fills in for lat and lon


# ---------------------------------------------------------------- text and tokens

def count_tokens(text):
    """The mock's tokenizer: four characters make one token."""
    return len(text) // CHARS_PER_TOKEN


def common_prefix_len(a, b):
    """How many characters two strings share at the start."""
    low, high = 0, min(len(a), len(b))
    while low < high:                       # binary search: prefix equality only goes one way
        mid = (low + high + 1) // 2
        if a[:mid] == b[:mid]:
            low = mid
        else:
            high = mid - 1
    return low


def message_text(message):
    """The text of one chat message. Content may be a string, a list of parts, or null."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def filler(node, tokens):
    """The same boring text every time, exactly `tokens` tokens long."""
    sentence = "This is mock LLM node %s. " % node
    chars = tokens * CHARS_PER_TOKEN
    return (sentence * (chars // len(sentence) + 1))[:chars]


# ---------------------------------------------------------------- the tool script

def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def pick_tool(tools, user_text):
    """The tool whose name and description share the most words with the user's text. Ties: the first."""
    asked = words(user_text)
    best, best_score = None, -1
    for tool in tools:
        function = tool["function"]
        score = len(asked & words("%s %s" % (function["name"], function.get("description") or "")))
        if score > best_score:
            best, best_score = function, score
    return best


def fill_argument(name, schema):
    """A believable value for one required parameter, from its JSON schema."""
    if "default" in schema:
        return schema["default"]
    if name == "lat":
        return SCRIPT_LAT
    if name == "lon":
        return SCRIPT_LON
    kind = schema.get("type")
    if isinstance(kind, list):              # e.g. ["number", "null"]
        kind = next((k for k in kind if k != "null"), None)
    if kind in ("number", "integer"):
        return schema.get("minimum", 1)
    choices = schema.get("enum")
    if isinstance(choices, list) and choices:
        return choices[0]
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "latest"                         # a string, or a parameter with no type


def tool_call_for(tools, user_text, call_id):
    """The scripted tool call: which tool, and its required arguments as a JSON string."""
    function = pick_tool(tools, user_text)
    schema = function.get("parameters") or {}
    properties = schema.get("properties")
    required = schema.get("required")
    arguments = {}
    for name in required if isinstance(required, list) else []:
        spec = properties.get(name) if isinstance(properties, dict) and isinstance(name, str) else None
        arguments[str(name)] = fill_argument(name, spec if isinstance(spec, dict) else {})
    return {"id": call_id, "name": function["name"], "arguments": json.dumps(arguments)}


# ---------------------------------------------------------------- requests and replies

def max_tokens_of(body):
    limit = body.get("max_tokens")
    return body.get("max_completion_tokens") if limit is None else limit


def check_request(body):
    """Say what is wrong with a chat request, or return None if it is fine."""
    if not isinstance(body, dict):
        return "the request body must be a JSON object"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return '"messages" must be a list with at least one message'
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            return 'messages[%d] must be an object with a "role"' % i
    limit = max_tokens_of(body)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_TOKENS_LIMIT:
            return '"max_tokens" must be a whole number from 1 to %d' % MAX_TOKENS_LIMIT
    tools = body.get("tools")
    if tools is None:
        return None
    if not isinstance(tools, list):
        return '"tools" must be a list'
    shape = '{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}'
    for i, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if (not isinstance(function, dict) or tool.get("type") != "function"
                or not isinstance(function.get("name"), str) or not function["name"]):
            return "tools[%d] must look like %s" % (i, shape)
        if not isinstance(function.get("parameters") or {}, dict):
            return 'tools[%d]: "parameters" must be a JSON schema object' % i
    return None


def plan_reply(body, node, serial):
    """Decide the whole reply up front: {"content", "tool", "finish"}. One of content and tool is None."""
    last = body["messages"][-1]
    tools = body.get("tools") or []
    limit = max_tokens_of(body)

    if tools and last["role"] == "user":
        call = tool_call_for(tools, message_text(last), "call_mock_%s_%d" % (node, serial))
        return {"content": None, "tool": call, "finish": "tool_calls"}

    if last["role"] == "tool":
        text = "Based on the tool: " + message_text(last)[:300]
        if limit is not None and count_tokens(text) > limit:
            return {"content": text[:limit * CHARS_PER_TOKEN], "tool": None, "finish": "length"}
        return {"content": text, "tool": None, "finish": "stop"}

    if limit is None:
        return {"content": filler(node, DEFAULT_MAX_TOKENS), "tool": None, "finish": "stop"}
    return {"content": filler(node, limit), "tool": None, "finish": "length"}   # it stopped at the caller's limit


def reply_message(reply):
    """The reply as an OpenAI assistant message."""
    tool = reply["tool"]
    if tool is None:
        return {"role": "assistant", "content": reply["content"]}
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": tool["id"], "type": "function",
                            "function": {"name": tool["name"], "arguments": tool["arguments"]}}]}


def reply_events(reply):
    """The reply cut into stream pieces: a list of (delta, characters in it)."""
    step = TOKENS_PER_EVENT * CHARS_PER_TOKEN
    tool = reply["tool"]
    if tool is None:
        text = reply["content"]
        return [({"content": text[i:i + step]}, len(text[i:i + step])) for i in range(0, len(text), step)]
    head = {"index": 0, "id": tool["id"], "type": "function",
            "function": {"name": tool["name"], "arguments": ""}}
    events = [({"tool_calls": [head]}, len(tool["name"]))]
    arguments = tool["arguments"]
    for i in range(0, len(arguments), step):
        piece = arguments[i:i + step]
        events.append(({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}, len(piece)))
    return events


def usage_and_timings(prompt_tokens, cached_tokens, completion_tokens, prompt_s, predicted_s):
    """OpenAI's "usage" and llama.cpp's "timings" for one finished request."""
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens,
             "prompt_tokens_details": {"cached_tokens": cached_tokens}}
    timings = {"cache_n": cached_tokens,
               "prompt_n": prompt_tokens - cached_tokens, "prompt_ms": round(prompt_s * 1000.0, 3),
               "predicted_n": completion_tokens, "predicted_ms": round(predicted_s * 1000.0, 3)}
    return usage, timings


def client_gone(connection):
    """True if the client has closed its end of the connection. Never waits."""
    try:
        return connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except (BlockingIOError, InterruptedError):
        return False                # nothing to read: the client is still there, waiting
    except OSError:
        return True


# ---------------------------------------------------------------- log lines

NOTES = queue.Queue(maxsize=1000)


def note(text):
    """Queue one log line for stderr. A request never waits for the log: if stderr is stuck, lines are dropped."""
    try:
        NOTES.put_nowait(text)
    except queue.Full:
        pass


def write_notes():
    """Runs in its own thread: writes the queued lines to stderr."""
    while True:
        text = NOTES.get()
        try:
            os.write(2, (text + "\n").encode("utf-8", "replace"))
        except OSError:
            pass


# ---------------------------------------------------------------- shared state

class Mock:
    """What the server knows: settings, slots, the waiting line, counters. One lock guards it all."""

    def __init__(self, node, flavor, tps, prefill_tps, slots, load_s, admin):
        self.node = node
        self.flavor = flavor
        self.tps = tps
        self.prefill_tps = prefill_tps
        self.load_s = load_s
        self.admin = admin
        self.lock = threading.Condition()
        self.born = time.monotonic()
        self.created = int(time.time())
        self.stopping = False
        self.slowdown = 1.0
        self.fail = "ok"
        self.slots = [{"busy": False, "prompt": "", "used": 0} for _ in range(slots)]
        self.line = collections.deque()     # requests waiting for a slot, first come first served
        self.uses = 0                       # counts slot uses, to find the slot unused the longest
        self.serial = 0                     # counts chat requests, for ids
        self.count = self._zero_counts()

    @staticmethod
    def _zero_counts():
        return {"prompt_tokens": 0, "predicted_tokens": 0,
                # since the last /metrics read, for the two tokens-per-second gauges
                "recent_prompt_tokens": 0, "recent_prompt_s": 0.0,
                "recent_predicted_tokens": 0, "recent_predicted_s": 0.0}

    def loading(self):
        return time.monotonic() - self.born < self.load_s

    def next_serial(self):
        with self.lock:
            self.serial += 1
            return self.serial

    def set_slowdown(self, factor):
        with self.lock:
            self.slowdown = factor

    def set_fail(self, mode):
        with self.lock:
            self.fail = mode
            self.lock.notify_all()          # wake hung and waiting requests so they see the change

    def reset(self):
        with self.lock:
            self.slowdown = 1.0
            self.fail = "ok"
            self.count = self._zero_counts()
            for slot in self.slots:
                slot["prompt"] = ""
            self.lock.notify_all()

    def stop(self):
        with self.lock:
            self.stopping = True
            self.lock.notify_all()

    def take_slot(self, prompt, keep_waiting):
        """Wait for a turn, then take the best idle slot.

        Returns (slot number, cached characters). Returns (None, 0) if
        keep_waiting() said the request should give up.
        """
        me = object()
        with self.lock:
            self.line.append(me)
            try:
                while True:
                    idle = [i for i, slot in enumerate(self.slots) if not slot["busy"]]
                    if idle and self.line[0] is me:
                        break
                    self.lock.wait(0.25)
                    if not keep_waiting():
                        return None, 0
            finally:
                self.line.remove(me)
                self.lock.notify_all()      # the next in line may go
            # Longest shared start wins. Ties go to the slot unused the longest.
            shared = dict((i, common_prefix_len(self.slots[i]["prompt"], prompt)) for i in idle)
            best = max(idle, key=lambda i: (shared[i], -self.slots[i]["used"]))
            self.uses += 1
            self.slots[best]["busy"] = True
            self.slots[best]["used"] = self.uses
            return best, shared[best]

    def free_slot(self, number, prompt, prompt_tokens, prompt_s, predicted_tokens, predicted_s):
        """Give a slot back. `prompt` is what the slot now has cached (None: leave its cache alone)."""
        with self.lock:
            slot = self.slots[number]
            slot["busy"] = False
            if prompt is not None:
                slot["prompt"] = prompt
            count = self.count
            count["prompt_tokens"] += prompt_tokens
            count["predicted_tokens"] += predicted_tokens
            count["recent_prompt_tokens"] += prompt_tokens
            count["recent_prompt_s"] += prompt_s
            count["recent_predicted_tokens"] += predicted_tokens
            count["recent_predicted_s"] += predicted_s
            self.lock.notify_all()

    def metrics_text(self):
        """Prometheus text, with the names the chosen engine uses."""
        with self.lock:
            running = sum(1 for slot in self.slots if slot["busy"])
            waiting = len(self.line)
            count = self.count
            if self.flavor == "vllm":
                label = '{model_name="%s"}' % MODEL
                rows = [
                    ("vllm:num_requests_running", "gauge", "Number of requests currently running on GPU.", float(running)),
                    ("vllm:num_requests_waiting", "gauge", "Number of requests waiting to be processed.", float(waiting)),
                    ("vllm:prompt_tokens_total", "counter", "Number of prefill tokens processed.", float(count["prompt_tokens"])),
                    ("vllm:generation_tokens_total", "counter", "Number of generation tokens processed.", float(count["predicted_tokens"])),
                ]
            else:
                # Like llama.cpp, the two tokens-per-second gauges average over the requests
                # that finished since the last /metrics read, and are 0 if there were none.
                prompt_rate = predicted_rate = 0
                if count["recent_prompt_s"] > 0:
                    prompt_rate = round(count["recent_prompt_tokens"] / count["recent_prompt_s"], 3)
                if count["recent_predicted_s"] > 0:
                    predicted_rate = round(count["recent_predicted_tokens"] / count["recent_predicted_s"], 3)
                count["recent_prompt_tokens"] = count["recent_predicted_tokens"] = 0
                count["recent_prompt_s"] = count["recent_predicted_s"] = 0.0
                label = ""
                rows = [
                    ("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed.", count["prompt_tokens"]),
                    ("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed.", count["predicted_tokens"]),
                    ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.", prompt_rate),
                    ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.", predicted_rate),
                    ("llamacpp:requests_processing", "gauge", "Number of requests processing.", running),
                    ("llamacpp:requests_deferred", "gauge", "Number of requests deferred.", waiting),
                ]
        lines = []
        for name, kind, help_text, value in rows:
            lines.append("# HELP %s %s" % (name, help_text))
            lines.append("# TYPE %s %s" % (name, kind))
            lines.append("%s%s %s" % (name, label, value))
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- HTTP

class Server(ThreadingHTTPServer):
    daemon_threads = True           # a hung request must not keep the process alive
    request_queue_size = 128        # room for a burst of connections

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], OSError):
            return                  # the client dropped the connection; not worth a traceback
        ThreadingHTTPServer.handle_error(self, request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # send small stream events at once

    def log_message(self, fmt, *args):
        pass                        # quiet; chat and admin calls print their own short line

    def end_headers(self):          # these two headers go on every response
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Mock-Node", self.server.mock.node)
        BaseHTTPRequestHandler.end_headers(self)

    def _send(self, code, body=b"", content_type=None):
        self.send_response(code)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, data):
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def _error(self, code, message, kind):
        self._json(code, {"error": {"message": message, "type": kind, "code": code}})

    # ---- the fail hooks

    def _alive(self):
        """False when a request should stop: the server is stopping, the node is down or hung, or the client left."""
        mock = self.server.mock
        return not mock.stopping and mock.fail not in ("down", "hang") and not client_gone(self.connection)

    def _hang(self):
        """Say nothing until the hang ends, the client leaves or the server stops. Then hang up."""
        mock = self.server.mock
        with mock.lock:
            while mock.fail == "hang" and not mock.stopping and not client_gone(self.connection):
                mock.lock.wait(0.5)
        self.close_connection = True

    def _down(self):
        self._error(503, "mock node %s is down (set by /admin/fail)" % self.server.mock.node, "unavailable_error")

    def _gate(self):
        """True if the request may be served. If not, it has been dealt with here."""
        mock = self.server.mock
        if mock.fail == "hang":
            self._hang()
            return False
        if mock.fail == "down":
            self._down()
            return False
        if mock.loading():
            self._error(503, "Loading model", "unavailable_error")
            return False
        return True

    def _give_up(self):
        """A request was stopped before its answer was complete."""
        mock = self.server.mock
        if mock.fail == "hang":
            self._hang()
        elif mock.fail == "down" and not self._stream_open:
            self._down()                    # nothing was sent yet, so the caller can retry elsewhere
        else:
            self.close_connection = True    # cut the stream, or the client is gone anyway

    # ---- GET and OPTIONS

    def do_OPTIONS(self):
        try:
            if not self.path.startswith("/admin/") and not self._gate():
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Session-Id")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except OSError:
            self.close_connection = True

    def do_GET(self):
        try:
            mock = self.server.mock
            path = self.path.split("?", 1)[0]
            if not self._gate():
                return
            if path == "/health":
                if mock.flavor == "vllm":
                    self._send(200)         # vLLM answers 200 with an empty body
                else:
                    self._json(200, {"status": "ok"})
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": MODEL, "object": "model", "created": mock.created, "owned_by": "mock"}]})
            elif path == "/metrics":
                self._send(200, mock.metrics_text().encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._error(404, "no such path: GET %s" % path, "not_found_error")
        except OSError:
            self.close_connection = True    # the client went away

    # ---- POST

    def do_POST(self):
        self._stream_open = False
        try:
            path = self.path.split("?", 1)[0]
            raw = self._read_body()
            if raw is None:
                return
            if path.startswith("/admin/"):
                self._admin(path, raw)
            elif not self._gate():
                return
            elif path == "/v1/chat/completions":
                self._chat(raw)
            else:
                self._error(404, "no such path: POST %s" % path, "not_found_error")
        except OSError:
            self.close_connection = True    # the client went away

    def _read_body(self):
        """The request body as bytes. On a bad request: answer it and return None."""
        length = (self.headers.get("Content-Length") or "0").strip()
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower() or not length.isdigit():
            self.close_connection = True
            self._error(411, "send the body with a Content-Length header", "invalid_request_error")
            return None
        if int(length) > MAX_BODY_BYTES:
            self.close_connection = True
            self._error(413, "the body is larger than %d bytes" % MAX_BODY_BYTES, "invalid_request_error")
            return None
        return self.rfile.read(int(length))

    def _admin(self, path, raw):
        mock = self.server.mock
        if not mock.admin:
            self._error(404, "the admin hooks are off; start the mock with --admin", "not_found_error")
            return
        try:
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._error(400, "the body must be a JSON object", "invalid_request_error")
            return
        if path == "/admin/contention":
            factor = body.get("factor")
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 < factor < float("inf"):
                self._error(400, '"factor" must be a number above 0, e.g. {"factor":1.8}', "invalid_request_error")
                return
            mock.set_slowdown(float(factor))
        elif path == "/admin/fail":
            if body.get("mode") not in FAIL_MODES:
                self._error(400, '"mode" must be one of: %s' % ", ".join(FAIL_MODES), "invalid_request_error")
                return
            mock.set_fail(body["mode"])
        elif path == "/admin/reset":
            mock.reset()
        else:
            self._error(404, "no such path: POST %s" % path, "not_found_error")
            return
        note("[mock %s] %s %s" % (mock.node, path, json.dumps(body)[:200]))
        self._json(200, {"node": mock.node, "factor": mock.slowdown, "mode": mock.fail})

    def _work(self, seconds):
        """Emulate `seconds` of model work (longer under contention). False if the request must stop."""
        mock = self.server.mock
        left = seconds
        while left > 0:
            if not self._alive():
                return False
            step = min(left, 0.05)
            self._due += step * mock.slowdown       # when this slice should be finished
            wait = self._due - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            left -= step
        return self._alive()

    def _chat(self, raw):
        mock = self.server.mock
        if mock.fail == "error":
            self._error(500, "mock node %s fails on purpose (set by /admin/fail)" % mock.node, "server_error")
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._error(400, "the request body is not valid JSON", "invalid_request_error")
            return
        problem = check_request(body)
        if problem:
            self._error(400, problem, "invalid_request_error")
            return

        serial = mock.next_serial()
        reply = plan_reply(body, mock.node, serial)
        stream = bool(body.get("stream"))
        prompt = "".join(message_text(message) for message in body["messages"])
        ident = {"id": "chatcmpl-mock-%s-%d" % (mock.node, serial), "created": int(time.time())}

        arrived = time.monotonic()
        slot, cached_chars = mock.take_slot(prompt, self._alive)        # waits its turn
        if slot is None:
            self._give_up()
            return
        started = time.monotonic()
        cached_tokens = cached_chars // CHARS_PER_TOKEN
        # What this request really did. It goes into the slot's cache and the counters.
        done = {"prompt": None, "prompt_tokens": 0, "prompt_s": 0.0, "predicted_tokens": 0, "predicted_s": 0.0}
        finished = False
        try:
            finished = self._answer(reply, stream, prompt, cached_tokens, ident, done)
        finally:
            mock.free_slot(slot, **done)
            note("[mock %s] chat%s slot %d: %d prompt tokens (%d cached), %d reply tokens, "
                 "waited %.2f s, ran %.2f s%s" % (
                     mock.node, " stream" if stream else "", slot, count_tokens(prompt), cached_tokens,
                     done["predicted_tokens"], started - arrived, time.monotonic() - started,
                     "" if finished else " (stopped early)"))
        if not finished:
            self._give_up()

    def _answer(self, reply, stream, prompt, cached_tokens, ident, done):
        """Do the work of one request and send the answer. True if all of it was sent."""

        def envelope(kind, choice):
            return {"id": ident["id"], "object": kind, "created": ident["created"], "model": MODEL,
                    "choices": [dict(index=0, **choice)]}

        mock = self.server.mock
        tool = reply["tool"]
        said = reply["content"] if tool is None else tool["name"] + tool["arguments"]
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(said)
        fresh_tokens = prompt_tokens - cached_tokens
        started = time.monotonic()
        self._due = started

        # 1. Read the part of the prompt that is not cached.
        if not self._work(fresh_tokens / mock.prefill_tps):
            return False
        prefilled = time.monotonic()
        self._due = prefilled
        done.update(prompt=prompt, prompt_tokens=fresh_tokens, prompt_s=prefilled - started)

        # 2. Write the reply: in one piece, or as a stream of events.
        write_s = completion_tokens / mock.tps
        if not stream:
            if not self._work(write_s):
                return False
            done.update(predicted_tokens=completion_tokens, predicted_s=time.monotonic() - prefilled)
            usage, timings = usage_and_timings(prompt_tokens, cached_tokens, completion_tokens,
                                               done["prompt_s"], done["predicted_s"])
            answer = envelope("chat.completion", {"message": reply_message(reply), "finish_reason": reply["finish"]})
            answer.update(usage=usage, timings=timings)
            self._json(200, answer)
            return True

        def chunk(delta, finish=None):
            return envelope("chat.completion.chunk", {"delta": delta, "finish_reason": finish})

        self._open_stream()
        self._event(chunk({"role": "assistant", "content": "" if tool is None else None}))
        sent = 0
        for delta, chars in reply_events(reply):
            if not self._work(write_s * chars / len(said)):
                break
            self._event(chunk(delta))
            sent += chars
            done.update(predicted_tokens=sent // CHARS_PER_TOKEN, predicted_s=time.monotonic() - prefilled)
        if sent < len(said):
            return False
        usage, timings = usage_and_timings(prompt_tokens, cached_tokens, completion_tokens,
                                           done["prompt_s"], done["predicted_s"])
        last = chunk({}, reply["finish"])
        last.update(usage=usage, timings=timings)       # llama.cpp also reports on the last chunk
        self._event(last)
        self._event(b"[DONE]")
        self._close_stream()
        return True

    # ---- streaming (server-sent events)

    def _open_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._chunked = self.request_version != "HTTP/1.0"
        if self._chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")     # an old client: the stream ends when we hang up
        self.end_headers()
        self._stream_open = True

    def _event(self, payload):
        """Send one event and flush it."""
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        data = b"data: " + data + b"\n\n"
        if self._chunked:
            data = b"%x\r\n" % len(data) + data + b"\r\n"
        self.wfile.write(data)
        self.wfile.flush()

    def _close_stream(self):
        if self._chunked:
            self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


# ---------------------------------------------------------------- command line

def above_zero(text):
    value = float(text)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="A stand-in for an LLM server (llama.cpp or vLLM). No model, no GPU.",
        epilog="Example: python serve/mock_llm.py --node A --port 8001 --admin")
    parser.add_argument("--node", default="A", help="the node's name; it shows in replies and in X-Mock-Node")
    parser.add_argument("--port", type=int, default=8001, help="port to listen on; 0 picks a free one (default 8001)")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--flavor", choices=("llamacpp", "vllm"), default="llamacpp",
                        help="whose /health and /metrics to imitate (default llamacpp)")
    parser.add_argument("--tps", type=above_zero, default=40.0, help="reply tokens per second (default 40)")
    parser.add_argument("--prefill-tps", type=above_zero, default=800.0,
                        help="prompt tokens read per second (default 800)")
    parser.add_argument("--slots", type=int, default=1, help="how many requests run at once (default 1)")
    parser.add_argument("--load-s", type=float, default=0.0,
                        help="pretend to load a model for this many seconds after start (default 0)")
    parser.add_argument("--admin", action="store_true", help="turn on the /admin test hooks")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be 0 to 65535")
    if args.slots < 1:
        parser.error("--slots must be 1 or more")
    if args.load_s < 0:
        parser.error("--load-s must be 0 or more")
    return args


def main(argv=None):
    args = parse_args(argv)
    mock = Mock(args.node, args.flavor, args.tps, args.prefill_tps, args.slots, args.load_s, args.admin)
    try:
        server = Server((args.host, args.port), Handler)
    except OSError as error:
        sys.stderr.write("mock_llm: cannot listen on %s:%d: %s. Is another copy running?\n"
                         % (args.host, args.port, error))
        return 1
    server.mock = mock

    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    for target in (write_notes, lambda: server.serve_forever(poll_interval=0.1)):
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
    # Everything is ready before this line goes out: scripts wait for it.
    print("listening on http://%s:%d" % (args.host, server.server_address[1]), flush=True)
    while not stop.is_set():
        stop.wait(0.5)

    mock.stop()                 # lets waiting, running and hung requests go
    server.shutdown()
    server.server_close()
    deadline = time.monotonic() + 1.0
    while not NOTES.empty() and time.monotonic() < deadline:
        time.sleep(0.01)        # let the last log lines out, but never wait long for a stuck stderr
    return 0


if __name__ == "__main__":
    sys.exit(main())
