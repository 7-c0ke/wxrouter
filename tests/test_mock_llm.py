"""Tests for the LLM mock (serve/mock_llm.py) and the local cluster tool (tools/dev_cluster.py).

Run them from the repo root:
    .venv/bin/python -m unittest tests/test_mock_llm.py -v
    .venv/bin/python -m unittest tests.test_mock_llm.ChatTests -v

Every server started here listens on 127.0.0.1, on a port the OS picks, and is
stopped when its tests end. The whole file takes about 20 seconds.

Written at the hackathon (Nov 13-14, 2026).
"""

import contextlib
import http.client
import importlib.util
import io
import json
import os
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK = os.path.join(REPO, "serve", "mock_llm.py")
TOOL = os.path.join(REPO, "tools", "dev_cluster.py")
SIGNATURE = "Written at the hackathon (Nov 13-14, 2026)."


class MockServer:
    """One mock_llm.py process on a port the OS picked."""

    def __init__(self, *args):
        self.errors = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([sys.executable, MOCK, "--port", "0"] + list(args),
                                     stdout=subprocess.PIPE, stderr=self.errors)
        ready, _, _ = select.select([self.proc.stdout], [], [], 20)
        self.first_line = self.proc.stdout.readline().decode("utf-8").rstrip("\n") if ready else ""
        self.started = time.monotonic()
        match = re.match(r"^listening on http://(127\.0\.0\.1):(\d+)$", self.first_line)
        if not match:
            said = self.stderr_text()
            self.stop()
            raise AssertionError("the mock did not start. First line: %r. stderr: %s" % (self.first_line, said))
        self.host, self.port = match.group(1), int(match.group(2))

    def stderr_text(self):
        """What the process has written to stderr so far."""
        number = self.errors.fileno()
        return os.pread(number, os.fstat(number).st_size, 0).decode("utf-8", "replace")

    def stop(self, sig=signal.SIGTERM):
        """Stop the process (if it still runs) and return its exit code."""
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc.stdout.close()
        self.errors.close()
        return self.proc.returncode

    def request(self, method, path, body=None, timeout=20):
        """One request. Returns (status, headers with lower-case names, body bytes)."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            data = headers = None
            if body is not None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                headers = {"Content-Type": "application/json"}
            conn.request(method, path, body=data, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict((k.lower(), v) for k, v in response.getheaders()), response.read()
        finally:
            conn.close()

    def get(self, path, timeout=20):
        return self.request("GET", path, timeout=timeout)

    def post(self, path, body, timeout=20):
        status, _, raw = self.request("POST", path, body, timeout=timeout)
        return status, json.loads(raw.decode("utf-8"))

    def chat(self, **body):
        """A chat completion. Returns (status, JSON answer, seconds it took)."""
        began = time.monotonic()
        status, answer = self.post("/v1/chat/completions", body)
        return status, answer, time.monotonic() - began

    def stream(self, **body):
        """A streaming chat completion. Returns (status, headers, lines): lines are (seconds since start, text)."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=20)
        try:
            began = time.monotonic()
            conn.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=True)),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            lines = []
            while True:
                line = response.readline()
                if not line:
                    break
                lines.append((time.monotonic() - began, line.decode("utf-8").rstrip("\r\n")))
            return response.status, dict((k.lower(), v) for k, v in response.getheaders()), lines
        finally:
            conn.close()

    def metrics(self):
        """GET /metrics. Returns (text, {sample name with labels: value})."""
        status, _, raw = self.get("/metrics")
        assert status == 200, status
        text = raw.decode("utf-8")
        samples = {}
        for line in text.splitlines():
            if line and not line.startswith("#"):
                name, value = line.rsplit(" ", 1)
                samples[name] = float(value)
        return text, samples


def user(text):
    return {"role": "user", "content": text}


def stream_chunks(lines):
    """The JSON chunks of a stream, with arrival times, after checking the event framing."""
    assert len(lines) % 2 == 0, lines
    for i in range(0, len(lines), 2):
        assert lines[i][1].startswith("data: "), lines[i]
        assert lines[i + 1][1] == "", lines[i + 1]              # a blank line ends each event
    assert lines[-2][1] == "data: [DONE]", lines[-2]
    return [(when, json.loads(text[len("data: "):])) for when, text in lines[:-2:2]]


WEATHER_TOOLS = [
    {"type": "function", "function": {
        "name": "cluster_status", "description": "Show which nodes are healthy and the routing policy.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "forecast_point", "description": "Forecast temperature and wind at one point on the map.",
        "parameters": {"type": "object", "required": ["lat", "lon", "lead_hours", "members", "init", "units", "note"],
                       "properties": {
                           "lat": {"type": "number", "minimum": -90},
                           "lon": {"type": "number"},
                           "lead_hours": {"type": "integer", "minimum": 6},
                           "members": {"type": "integer"},
                           "init": {"type": "string", "enum": ["2019-01-27T12", "latest"]},
                           "units": {"type": "string", "default": "celsius"},
                           "note": {"type": "string"},
                           "with_truth": {"type": "boolean"}}}}},
]
FORECAST_ARGUMENTS = {"lat": 41.9602, "lon": -87.9316, "lead_hours": 6, "members": 1,
                      "init": "2019-01-27T12", "units": "celsius", "note": "latest"}


class ChatTests(unittest.TestCase):
    """The API itself, on one mock: 400 reply tokens and 8000 prompt tokens per second."""

    @classmethod
    def setUpClass(cls):
        cls.mock = MockServer("--node", "A", "--tps", "400", "--prefill-tps", "8000", "--admin")

    @classmethod
    def tearDownClass(cls):
        cls.mock.stop()

    def setUp(self):
        self.mock.post("/admin/reset", {})

    def test_first_line_and_models(self):
        self.assertRegex(self.mock.first_line, r"^listening on http://127\.0\.0\.1:\d+$")
        status, headers, raw = self.mock.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(headers["access-control-allow-origin"], "*")
        models = json.loads(raw.decode("utf-8"))
        self.assertEqual([model["id"] for model in models["data"]], ["mock-llm"])
        status, _, raw = self.mock.get("/health")
        self.assertEqual((status, json.loads(raw.decode("utf-8"))), (200, {"status": "ok"}))

    def test_chat_shape(self):
        prompt = "word " * 320                                  # 1600 characters = 400 tokens = 0.05 s
        status, headers, raw = self.mock.request(
            "POST", "/v1/chat/completions", {"model": "anything", "messages": [user(prompt)], "max_tokens": 40})
        self.assertEqual(status, 200)
        self.assertEqual(headers["x-mock-node"], "A")
        self.assertEqual(headers["access-control-allow-origin"], "*")
        self.assertTrue(headers["content-type"].startswith("application/json"))
        answer = json.loads(raw.decode("utf-8"))
        self.assertEqual(answer["object"], "chat.completion")
        self.assertEqual(answer["model"], "mock-llm")
        self.assertIsInstance(answer["id"], str)
        self.assertIsInstance(answer["created"], int)
        self.assertEqual(len(answer["choices"]), 1)
        choice = answer["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(len(choice["message"]["content"]), 160)           # 40 tokens of 4 characters
        self.assertIn("node A", choice["message"]["content"])
        self.assertEqual(choice["finish_reason"], "length")                # it ran into max_tokens
        self.assertEqual(answer["usage"], {"prompt_tokens": 400, "completion_tokens": 40, "total_tokens": 440,
                                           "prompt_tokens_details": {"cached_tokens": 0}})
        timings = answer["timings"]
        self.assertEqual((timings["cache_n"], timings["prompt_n"], timings["predicted_n"]), (0, 400, 40))
        self.assertGreaterEqual(timings["prompt_ms"], 49.0)                # 400 tokens at 8000 per second
        self.assertGreaterEqual(timings["predicted_ms"], 99.0)             # 40 tokens at 400 per second

    def test_time_is_prefill_plus_reply_and_text_is_the_same_each_time(self):
        self.mock.post("/admin/reset", {})
        status, first, took = self.mock.chat(messages=[user("x" * 3200)], max_tokens=80)
        self.assertEqual(status, 200)
        self.assertGreaterEqual(took, 0.1 + 0.2 - 0.01)                    # 800 / 8000 + 80 / 400 seconds
        self.assertLess(took, 2.0)
        _, second, _ = self.mock.chat(messages=[user("x" * 3200)], max_tokens=80)
        self.assertEqual(first["choices"][0]["message"], second["choices"][0]["message"])

    def test_default_reply_is_64_tokens(self):
        status, answer, _ = self.mock.chat(messages=[user("hello")])
        self.assertEqual(status, 200)
        self.assertEqual(answer["usage"]["completion_tokens"], 64)
        self.assertEqual(answer["choices"][0]["finish_reason"], "stop")

    def test_content_given_as_parts_counts_too(self):
        parts = [{"type": "text", "text": "a" * 40}, {"type": "text", "text": "b" * 40}]
        _, answer, _ = self.mock.chat(messages=[{"role": "system", "content": "c" * 20},
                                                {"role": "user", "content": parts}], max_tokens=4)
        self.assertEqual(answer["usage"]["prompt_tokens"], 25)

    def test_bad_requests_get_a_clear_400(self):
        cases = [b"{not json", {"model": "x"}, {"messages": []}, {"messages": [{"content": "no role"}]},
                 {"messages": [user("hi")], "max_tokens": 0}, {"messages": [user("hi")], "max_tokens": "many"},
                 {"messages": [user("hi")], "tools": [{"name": "bare", "parameters": {}}]}]
        for body in cases:
            status, headers, raw = self.mock.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 400, body)
            self.assertEqual(headers["access-control-allow-origin"], "*")
            self.assertTrue(json.loads(raw.decode("utf-8"))["error"]["message"], body)
        status, headers, raw = self.mock.get("/nowhere")
        self.assertEqual(status, 404)
        self.assertEqual(headers["access-control-allow-origin"], "*")
        status, _, _ = self.mock.request("POST", "/v1/nowhere", {})
        self.assertEqual(status, 404)
        status, _, _ = self.mock.get("/health")                 # and the server is still fine
        self.assertEqual(status, 200)

    def test_preflight(self):
        status, headers, raw = self.mock.request("OPTIONS", "/v1/chat/completions")
        self.assertEqual((status, raw), (204, b""))
        self.assertEqual(headers["access-control-allow-origin"], "*")
        self.assertIn("POST", headers["access-control-allow-methods"])
        self.assertIn("Content-Type", headers["access-control-allow-headers"])

    def test_stream_arrives_in_pieces_over_time(self):
        body = {"messages": [user("tell me a story")], "max_tokens": 120}     # 120 tokens at 400/s = 0.3 s
        status, headers, lines = self.mock.stream(**body)
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(headers["x-mock-node"], "A")
        self.assertEqual(headers["access-control-allow-origin"], "*")
        chunks = stream_chunks(lines)
        for _, chunk in chunks:
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["model"], "mock-llm")
            self.assertEqual(len(chunk["choices"]), 1)
        self.assertEqual(len(set(chunk["id"] for _, chunk in chunks)), 1)
        self.assertEqual(chunks[0][1]["choices"][0]["delta"], {"role": "assistant", "content": ""})

        pieces = [(when, chunk["choices"][0]["delta"]["content"]) for when, chunk in chunks[1:-1]]
        self.assertEqual(len(pieces), 30)                                  # 4 tokens per event
        for _, piece in pieces:
            self.assertEqual(len(piece), 16)
        for _, chunk in chunks[:-1]:
            self.assertIsNone(chunk["choices"][0]["finish_reason"])
        # The pieces come one after another, about 10 ms apart, not in one lump at the end.
        times = [when for when, _ in pieces]
        self.assertGreaterEqual(times[-1] - times[0], 0.25)
        self.assertGreaterEqual(times[-1], 0.29)
        self.assertGreaterEqual(len(set(round(when, 2) for when in times)), 5)

        last = chunks[-1][1]
        self.assertEqual(last["choices"][0], {"index": 0, "delta": {}, "finish_reason": "length"})
        self.assertEqual(last["usage"]["completion_tokens"], 120)
        self.assertEqual(last["usage"]["prompt_tokens_details"], {"cached_tokens": 0})
        self.assertIn("predicted_ms", last["timings"])

        self.mock.post("/admin/reset", {})
        _, whole, _ = self.mock.chat(**body)                    # the same request without streaming
        self.assertEqual("".join(piece for _, piece in pieces), whole["choices"][0]["message"]["content"])

    def test_one_connection_serves_request_after_request(self):
        conn = http.client.HTTPConnection(self.mock.host, self.mock.port, timeout=20)
        self.addCleanup(conn.close)
        body = {"messages": [user("hi")], "max_tokens": 8}
        for stream in (False, True, False, True):
            conn.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, stream=stream)),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            text = response.read().decode("utf-8")              # a stream must end cleanly for the next request to work
            self.assertEqual(response.status, 200)
            self.assertEqual(text.endswith("data: [DONE]\n\n"), stream)
        conn.request("GET", "/health")
        response = conn.getresponse()
        self.assertEqual((response.status, response.read()), (200, b'{"status": "ok"}'))

    def test_stream_for_an_http_1_0_client_ends_when_the_server_hangs_up(self):
        body = json.dumps({"messages": [user("hi")], "max_tokens": 8, "stream": True}).encode("utf-8")
        raw = socket.create_connection((self.mock.host, self.mock.port), timeout=20)
        self.addCleanup(raw.close)
        raw.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body)
        answer = b""
        while True:
            piece = raw.recv(65536)
            if not piece:
                break
            answer += piece
        head, _, events = answer.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 200"), head)
        self.assertNotIn(b"chunked", head.lower())
        self.assertTrue(events.startswith(b"data: {"), events[:40])            # plain events, no chunk sizes
        self.assertTrue(events.endswith(b"data: [DONE]\n\n"), events[-40:])

    def test_tool_script_calls_a_tool_then_quotes_its_result(self):
        question = user("Please forecast the temperature at this point.")
        status, answer, _ = self.mock.chat(messages=[{"role": "system", "content": "You are the desk."}, question],
                                           tools=WEATHER_TOOLS)
        self.assertEqual(status, 200)
        choice = answer["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(len(choice["message"]["tool_calls"]), 1)
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertTrue(call["id"])
        self.assertEqual(call["function"]["name"], "forecast_point")
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(json.loads(call["function"]["arguments"]), FORECAST_ARGUMENTS)   # with_truth is not required
        self.assertGreater(answer["usage"]["completion_tokens"], 0)

        result = "t2m_c 3.9 3.5 3.2 3.15 " + "z" * 400
        messages = [question, choice["message"], {"role": "tool", "tool_call_id": call["id"], "content": result}]
        status, answer, _ = self.mock.chat(messages=messages, tools=WEATHER_TOOLS)
        self.assertEqual(status, 200)
        choice = answer["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"], {"role": "assistant", "content": "Based on the tool: " + result[:300]})

    def test_tool_script_picks_by_shared_words_and_ties_go_to_the_first(self):
        def chosen(text):
            _, answer, _ = self.mock.chat(messages=[user(text)], tools=WEATHER_TOOLS)
            call = answer["choices"][0]["message"]["tool_calls"][0]["function"]
            return call["name"], json.loads(call["arguments"])

        self.assertEqual(chosen("Which nodes are healthy?"), ("cluster_status", {}))
        self.assertEqual(chosen("zzz qqq"), ("cluster_status", {}))                  # no shared words: the first tool
        self.assertEqual(chosen("WIND on the MAP")[0], "forecast_point")             # case does not matter
        _, answer, _ = self.mock.chat(messages=[user("Which nodes are healthy?")], tools=[], max_tokens=8)
        self.assertEqual(answer["choices"][0]["finish_reason"], "length")            # no tools: plain filler

    def test_tool_script_as_a_stream(self):
        question = user("Please forecast the temperature at this point.")
        status, _, lines = self.mock.stream(messages=[question], tools=WEATHER_TOOLS)
        self.assertEqual(status, 200)
        chunks = [chunk for _, chunk in stream_chunks(lines)]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": None})
        head = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual((head["index"], head["type"], head["function"]["name"]), (0, "function", "forecast_point"))
        self.assertTrue(head["id"])
        arguments = head["function"]["arguments"]
        for chunk in chunks[2:-1]:
            (part,) = chunk["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(part["index"], 0)
            arguments += part["function"]["arguments"]
        self.assertGreater(len(chunks), 5)                                 # the arguments came in pieces
        self.assertEqual(json.loads(arguments), FORECAST_ARGUMENTS)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

        messages = [question, {"role": "assistant", "content": None, "tool_calls": [
            {"id": head["id"], "type": "function", "function": {"name": "forecast_point", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": head["id"], "content": "t2m_c 3.9"}]
        status, _, lines = self.mock.stream(messages=messages, tools=WEATHER_TOOLS)
        chunks = [chunk for _, chunk in stream_chunks(lines)]
        text = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks)
        self.assertEqual(text, "Based on the tool: t2m_c 3.9")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_metrics_with_llama_cpp_names(self):
        self.mock.post("/admin/reset", {})
        self.mock.chat(messages=[user("m" * 1600)], max_tokens=40)
        text, samples = self.mock.metrics()
        self.assertEqual(sorted(samples), [
            "llamacpp:predicted_tokens_seconds", "llamacpp:prompt_tokens_seconds", "llamacpp:prompt_tokens_total",
            "llamacpp:requests_deferred", "llamacpp:requests_processing", "llamacpp:tokens_predicted_total"])
        for name in samples:
            self.assertIn("# HELP %s " % name, text)
            self.assertRegex(text, r"(?m)^# TYPE %s (counter|gauge)$" % re.escape(name))
        self.assertEqual(samples["llamacpp:prompt_tokens_total"], 400)
        self.assertEqual(samples["llamacpp:tokens_predicted_total"], 40)
        self.assertEqual(samples["llamacpp:requests_processing"], 0)
        self.assertEqual(samples["llamacpp:requests_deferred"], 0)
        self.assertTrue(40 <= samples["llamacpp:predicted_tokens_seconds"] <= 404, samples)    # set to 400
        self.assertTrue(800 <= samples["llamacpp:prompt_tokens_seconds"] <= 8080, samples)     # set to 8000
        # Like llama.cpp: the two rates cover the time since the last read, the totals stay.
        _, again = self.mock.metrics()
        self.assertEqual(again["llamacpp:predicted_tokens_seconds"], 0)
        self.assertEqual(again["llamacpp:prompt_tokens_seconds"], 0)
        self.assertEqual(again["llamacpp:tokens_predicted_total"], 40)


class CacheTests(unittest.TestCase):
    """The prompt cache: one remembered prompt per slot."""

    def test_a_repeated_prefix_is_faster_and_counts_as_cached(self):
        server = MockServer("--prefill-tps", "2000", "--tps", "100000")
        self.addCleanup(server.stop)
        system = {"role": "system", "content": "s" * 4800}                 # 1200 tokens = 0.6 s to read
        first = [system, user("question one")]
        status, answer, cold = server.chat(messages=first, max_tokens=4)
        self.assertEqual(status, 200)
        self.assertEqual(answer["usage"]["prompt_tokens"], 1203)
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertGreaterEqual(cold, 0.59)

        longer = first + [{"role": "assistant", "content": "an answer"}, user("question two, a little longer")]
        _, answer, warm = server.chat(messages=longer, max_tokens=4)
        self.assertEqual(answer["usage"]["prompt_tokens"], 1212)
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 1203)
        self.assertEqual((answer["timings"]["cache_n"], answer["timings"]["prompt_n"]), (1203, 9))
        self.assertLess(warm, cold / 2)

        _, answer, other = server.chat(messages=[user("t" * 1600)], max_tokens=4)   # nothing in common
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertGreaterEqual(other, 0.19)

        _, samples = server.metrics()                           # only tokens that were really read are counted
        self.assertEqual(samples["llamacpp:prompt_tokens_total"], 1203 + 9 + 400)

    def test_a_request_takes_the_slot_that_knows_its_prompt(self):
        server = MockServer("--slots", "2", "--prefill-tps", "100000", "--tps", "100000")
        self.addCleanup(server.stop)
        one, two = "a" * 2000, "b" * 3000
        for prompt in (one, two):                               # they land in different slots
            _, answer, _ = server.chat(messages=[user(prompt)], max_tokens=4)
            self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        _, answer, _ = server.chat(messages=[user(one), user("and more")], max_tokens=4)
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 500)
        _, answer, _ = server.chat(messages=[user(two), user("and more")], max_tokens=4)
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 750)


class SlotTests(unittest.TestCase):
    """Slots: N requests run at once, the rest wait their turn."""

    def run_at_once(self, server, count, tokens):
        """Send `count` chat requests together. Returns (threads, list of (status, seconds))."""
        results = []

        def one(number):
            status, _, took = server.chat(messages=[user("%d request" % number)], max_tokens=tokens)
            results.append((status, took))

        threads = [threading.Thread(target=one, args=(number,)) for number in range(count)]
        for thread in threads:
            thread.start()
        return threads, results

    def test_one_slot_makes_the_others_wait(self):
        server = MockServer("--slots", "1", "--tps", "100")
        self.addCleanup(server.stop)
        threads, results = self.run_at_once(server, 3, 60)      # 0.6 s each
        time.sleep(0.3)
        _, during = server.metrics()
        for thread in threads:
            thread.join()
        self.assertEqual(during["llamacpp:requests_processing"], 1)
        self.assertEqual(during["llamacpp:requests_deferred"], 2)
        self.assertEqual([status for status, _ in results], [200, 200, 200])
        times = sorted(took for _, took in results)
        self.assertGreaterEqual(times[0], 0.59)
        self.assertGreaterEqual(times[1], 1.19)
        self.assertGreaterEqual(times[2], 1.79)
        _, after = server.metrics()
        self.assertEqual((after["llamacpp:requests_processing"], after["llamacpp:requests_deferred"]), (0, 0))
        self.assertEqual(after["llamacpp:tokens_predicted_total"], 180)

    def test_two_slots_run_two_at_once_and_vllm_names(self):
        server = MockServer("--slots", "2", "--tps", "100", "--flavor", "vllm")
        self.addCleanup(server.stop)
        status, _, raw = server.get("/health")
        self.assertEqual((status, raw), (200, b""))             # vLLM's health has no body
        threads, results = self.run_at_once(server, 3, 60)      # 0.6 s each: two now, one after
        time.sleep(0.3)
        text, during = server.metrics()
        for thread in threads:
            thread.join()
        label = '{model_name="mock-llm"}'
        self.assertEqual(sorted(during), sorted("vllm:%s%s" % (name, label) for name in (
            "num_requests_running", "num_requests_waiting", "prompt_tokens_total", "generation_tokens_total")))
        self.assertNotIn("llamacpp", text)
        self.assertEqual(during["vllm:num_requests_running" + label], 2)
        self.assertEqual(during["vllm:num_requests_waiting" + label], 1)
        times = sorted(took for _, took in results)
        self.assertLess(times[1], 1.1)                          # the first two did not wait for each other
        self.assertGreaterEqual(times[2], 1.19)                 # the third did
        _, after = server.metrics()
        self.assertEqual(after["vllm:generation_tokens_total" + label], 180)
        self.assertEqual(after["vllm:prompt_tokens_total" + label], 6)       # three prompts of 9 characters
        self.assertEqual(after["vllm:num_requests_running" + label], 0)


class AdminTests(unittest.TestCase):
    """The test hooks: contention, fail, reset."""

    @classmethod
    def setUpClass(cls):
        cls.mock = MockServer("--node", "B", "--tps", "400", "--prefill-tps", "100000", "--admin")

    @classmethod
    def tearDownClass(cls):
        cls.mock.stop()

    def setUp(self):
        self.assertEqual(self.mock.post("/admin/reset", {}), (200, {"node": "B", "factor": 1.0, "mode": "ok"}))

    def fail(self, mode):
        status, answer = self.mock.post("/admin/fail", {"mode": mode})
        self.assertEqual((status, answer["mode"]), (200, mode))

    def test_contention_slows_everything_down(self):
        _, _, normal = self.mock.chat(messages=[user("hi")], max_tokens=80)          # 0.2 s
        self.assertEqual(self.mock.post("/admin/contention", {"factor": 3})[1]["factor"], 3.0)
        _, _, slowed = self.mock.chat(messages=[user("hi")], max_tokens=80)          # 0.6 s
        self.assertEqual(self.mock.post("/admin/contention", {"factor": 1.0})[0], 200)
        _, _, cleared = self.mock.chat(messages=[user("hi")], max_tokens=80)
        self.assertGreaterEqual(normal, 0.19)
        self.assertGreaterEqual(slowed, 0.59)
        self.assertLess(normal, 0.55)
        self.assertLess(cleared, 0.55)
        for bad in ({"factor": 0}, {"factor": -2}, {"factor": "fast"}, {}):
            self.assertEqual(self.mock.post("/admin/contention", bad)[0], 400, bad)

    def test_down_gives_503_everywhere_and_ok_recovers(self):
        self.fail("down")
        for path in ("/health", "/v1/models", "/metrics"):
            status, headers, _ = self.mock.get(path)
            self.assertEqual(status, 503, path)
            self.assertEqual(headers["access-control-allow-origin"], "*")
        status, answer, _ = self.mock.chat(messages=[user("hi")])
        self.assertEqual(status, 503)
        self.assertIn("down", answer["error"]["message"])
        self.fail("ok")
        self.assertEqual(self.mock.get("/health")[0], 200)
        self.assertEqual(self.mock.chat(messages=[user("hi")], max_tokens=4)[0], 200)
        self.assertEqual(self.mock.post("/admin/fail", {"mode": "sideways"})[0], 400)

    def test_error_fails_the_work_but_not_the_health_check(self):
        self.fail("error")
        status, answer, _ = self.mock.chat(messages=[user("hi")])
        self.assertEqual(status, 500)
        self.assertTrue(answer["error"]["message"])
        self.assertEqual(self.mock.get("/health")[0], 200)
        self.assertEqual(self.mock.get("/v1/models")[0], 200)

    def test_hang_accepts_and_never_answers(self):
        self.fail("hang")
        for method, path, body in (("GET", "/health", None), ("POST", "/v1/chat/completions", {"messages": [user("hi")]})):
            began = time.monotonic()
            with self.assertRaises(socket.timeout):
                self.mock.request(method, path, body, timeout=0.6)
            self.assertGreaterEqual(time.monotonic() - began, 0.55)
        self.fail("ok")
        self.assertEqual(self.mock.get("/health")[0], 200)

    def test_down_also_stops_requests_that_wait_or_run(self):
        results = []

        def long_request():
            results.append(self.mock.chat(messages=[user("hi")], max_tokens=4000))   # 10 s if left alone

        threads = [threading.Thread(target=long_request) for _ in range(2)]          # one runs, one waits
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        began = time.monotonic()
        self.fail("down")
        for thread in threads:
            thread.join(5)
        self.assertLess(time.monotonic() - began, 2.0)
        self.assertEqual([status for status, _, _ in results], [503, 503])           # nothing was sent, so: retry elsewhere

    def test_down_cuts_a_stream_short(self):
        seen = []

        def long_stream():
            try:
                seen.append(self.mock.stream(messages=[user("hi")], max_tokens=4000)[2])
            except (http.client.HTTPException, OSError) as error:
                seen.append(error)

        thread = threading.Thread(target=long_stream)
        thread.start()
        time.sleep(0.3)
        self.fail("down")
        thread.join(5)
        self.assertFalse(thread.is_alive())
        (outcome,) = seen
        if not isinstance(outcome, Exception):                  # the stream ended early, without its last event
            self.assertNotIn("data: [DONE]", [text for _, text in outcome])

    def test_reset_clears_hooks_counters_and_caches(self):
        prompt = [user("r" * 800)]
        self.mock.chat(messages=prompt, max_tokens=4)
        _, answer, _ = self.mock.chat(messages=prompt, max_tokens=4)
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 200)
        self.mock.post("/admin/contention", {"factor": 5})
        self.fail("error")
        self.assertEqual(self.mock.post("/admin/reset", {}), (200, {"node": "B", "factor": 1.0, "mode": "ok"}))
        _, samples = self.mock.metrics()
        self.assertEqual(samples["llamacpp:prompt_tokens_total"], 0)
        self.assertEqual(samples["llamacpp:tokens_predicted_total"], 0)
        status, answer, took = self.mock.chat(messages=prompt, max_tokens=4)
        self.assertEqual(status, 200)
        self.assertEqual(answer["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertLess(took, 0.5)
        status, _, _ = self.mock.request("POST", "/admin/reset")             # no body at all is fine too
        self.assertEqual(status, 200)

    def test_hooks_are_off_without_the_flag(self):
        server = MockServer("--node", "C")
        self.addCleanup(server.stop)
        for path, body in (("/admin/fail", {"mode": "down"}), ("/admin/contention", {"factor": 9}), ("/admin/reset", {})):
            status, answer = server.post(path, body)
            self.assertEqual(status, 404, path)
            self.assertIn("--admin", answer["error"]["message"])
        self.assertEqual(server.get("/health")[0], 200)


class StartAndStopTests(unittest.TestCase):
    """Loading, and leaving cleanly."""

    def test_health_says_loading_for_the_first_seconds(self):
        server = MockServer("--load-s", "2.5")
        self.addCleanup(server.stop)
        status, _, raw = server.get("/health")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(raw.decode("utf-8"))["error"]["message"], "Loading model")
        self.assertEqual(server.chat(messages=[user("hi")])[0], 503)
        time.sleep(max(0.0, server.started + 2.6 - time.monotonic()))
        status, _, raw = server.get("/health")
        self.assertEqual((status, json.loads(raw.decode("utf-8"))), (200, {"status": "ok"}))
        self.assertEqual(server.chat(messages=[user("hi")], max_tokens=4)[0], 200)

    def test_sigterm_and_sigint_exit_with_code_0(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            server = MockServer()
            self.addCleanup(server.stop)
            self.assertEqual(server.get("/health")[0], 200)
            self.assertEqual(server.stop(sig), 0, sig)

    def test_it_exits_even_with_a_request_hanging(self):
        server = MockServer("--admin")
        self.addCleanup(server.stop)
        server.post("/admin/fail", {"mode": "hang"})
        waiting = socket.create_connection((server.host, server.port), timeout=5)
        self.addCleanup(waiting.close)
        waiting.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        time.sleep(0.2)
        began = time.monotonic()
        self.assertEqual(server.stop(), 0)
        self.assertLess(time.monotonic() - began, 5.0)

    def test_clients_that_drop_the_connection_leave_no_traceback(self):
        server = MockServer()
        self.addCleanup(server.stop)
        rude = socket.create_connection((server.host, server.port), timeout=5)
        rude.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        rude.close()                                            # a reset before any request
        rude = socket.create_connection((server.host, server.port), timeout=5)
        rude.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        time.sleep(0.2)
        rude.close()                                            # leaves without reading the answer
        rude = socket.create_connection((server.host, server.port), timeout=5)
        rude.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 500\r\n\r\n{")
        rude.close()                                            # leaves in the middle of a request
        time.sleep(0.3)
        self.assertEqual(server.get("/health")[0], 200)
        self.assertEqual(server.stderr_text(), "")

    def test_it_keeps_answering_when_nobody_reads_its_stderr(self):
        proc = subprocess.Popen([sys.executable, MOCK, "--port", "0", "--tps", "1000000", "--prefill-tps", "1000000"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)      # stderr is a pipe nobody reads

        def clean_up():
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()

        self.addCleanup(clean_up)
        port = int(proc.stdout.readline().decode("utf-8").rsplit(":", 1)[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.addCleanup(conn.close)
        body = json.dumps({"messages": [user("hi")], "max_tokens": 1})
        for _ in range(1500):                                   # 1500 log lines: more than a pipe holds
            conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
        proc.send_signal(signal.SIGTERM)                        # and it still leaves cleanly
        self.assertEqual(proc.wait(timeout=10), 0)

    def test_a_taken_port_is_reported(self):
        server = MockServer()
        self.addCleanup(server.stop)
        second = subprocess.run([sys.executable, MOCK, "--port", str(server.port)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(second.returncode, 1)
        self.assertEqual(second.stdout, b"")
        self.assertIn(b"cannot listen", second.stderr)

    def test_every_file_says_where_it_was_written(self):
        for path in (MOCK, TOOL, os.path.abspath(__file__)):
            with open(path) as handle:
                self.assertIn(SIGNATURE, handle.read(), path)


def load_tool():
    """Import tools/dev_cluster.py as a module."""
    spec = importlib.util.spec_from_file_location("dev_cluster", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port():
    """A port the OS picked, free a moment ago."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


# The router's config that everyone builds against, as handed out at the start. One value was changed
# when the pieces were put together: spill_inflight went from 4 to 0 (see router/config.example.json).
TEAM_CONFIG = json.loads("""
{"port": 8100, "policy": "adaptive", "probe_s": 1.0, "fail_n": 2, "recover_s": 3.0, "connect_timeout_s": 2.0,
 "adaptive": {"hysteresis": 0.25, "dwell_s": 3.0, "cooldown_s": 10.0, "spill_inflight": 0},
 "score": {"busy_other_penalty": 1.5, "low_mem_gb": 8, "low_mem_penalty": 2.0, "default_latency_ms": 200},
 "heal_after_s": 10, "heal_backoff_s": [5, 15, 60], "heal_max_attempts": 5,
 "decision_log": "router-decisions.jsonl",
 "pools": {
   "forecast": {"first_byte_timeout_s": 120, "backends": [
       {"name": "A", "url": "http://127.0.0.1:8101", "probe": "forecast",
        "heal": {"http": {"url": "http://127.0.0.1:8101/admin/fail", "method": "POST", "json": {"mode": "ok"}}}},
       {"name": "B", "url": "http://127.0.0.1:8102", "probe": "forecast",
        "heal": {"http": {"url": "http://127.0.0.1:8102/admin/fail", "method": "POST", "json": {"mode": "ok"}}}}]},
   "llm": {"first_byte_timeout_s": 600, "affinity": true, "affinity_break": 3.0, "backends": [
       {"name": "A", "url": "http://127.0.0.1:8001", "probe": "llamacpp"},
       {"name": "B", "url": "http://127.0.0.1:8002", "probe": "llamacpp",
        "heal": {"cmd": ["echo", "restart B's LLM here"]}}]}}}
""")


class DevClusterTests(unittest.TestCase):
    """tools/dev_cluster.py, as far as it goes without the forecast server and the router."""

    def setUp(self):
        self.tool = load_tool()
        self.folder = tempfile.mkdtemp(prefix="test_dev_cluster_")
        self.addCleanup(shutil.rmtree, self.folder, True)
        # Keep these tests away from a real cluster's state file and logs.
        self.tool.STATE_FILE = os.path.join(self.folder, "state.json")
        self.tool.LOG_DIR = os.path.join(self.folder, "logs")

    def run_tool(self, *argv):
        """Run one command in this process. Returns (exit code, what it printed)."""
        said = io.StringIO()
        with contextlib.redirect_stdout(said), warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)    # "up" leaves its children running on purpose
            code = self.tool.main(list(argv))
        return code, said.getvalue()

    def only_llm_mocks(self, **fixed_ports):
        """Make "up" start just the two LLM mocks, on ports from the OS, plus one piece that is not written yet."""
        ports = dict((name, free_port()) for name in ("router", "forecast-A", "forecast-B", "llm-A", "llm-B"))
        ports.update(fixed_ports)
        whole_plan = self.tool.plan

        def small_plan(ports, config_path):
            pieces = whole_plan(ports, config_path)
            ghost = dict(pieces[-1], name="ghost", script=os.path.join(REPO, "serve", "not_written_yet.py"))
            return [piece for piece in pieces if piece["name"].startswith("llm-")] + [ghost]

        self.tool.ports_for = lambda base=None: ports
        self.tool.plan = small_plan
        return ports

    def read_state(self):
        with open(self.tool.STATE_FILE) as handle:
            return json.load(handle)

    # ---- the parts that start nothing

    def test_ports_standard_and_shifted(self):
        self.assertEqual(self.tool.ports_for(), {
            "router": 8100, "forecast-A": 8101, "forecast-B": 8102, "llm-A": 8001, "llm-B": 8002})
        self.assertEqual(self.tool.ports_for(18100), {
            "router": 18100, "forecast-A": 18101, "forecast-B": 18102, "llm-A": 18111, "llm-B": 18112})
        for bad in ("80", "65530", "many"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.tool.main(["up", "--ports-from", bad])
        self.assertFalse(os.path.exists(self.tool.STATE_FILE))

    def test_config_for_the_standard_ports_is_the_team_config(self):
        self.assertEqual(self.tool.router_config(self.tool.ports_for()), TEAM_CONFIG)
        # The copy in the repo must say the same, or shifted clusters would run with other settings.
        with open(os.path.join(REPO, "router", "config.example.json")) as handle:
            self.assertEqual(json.load(handle), TEAM_CONFIG)

    def test_config_for_shifted_ports(self):
        config = self.tool.router_config(self.tool.ports_for(18100), "/somewhere/decisions.jsonl")
        self.assertEqual(config["port"], 18100)
        self.assertEqual(config["decision_log"], "/somewhere/decisions.jsonl")
        forecast, llm = config["pools"]["forecast"]["backends"], config["pools"]["llm"]["backends"]
        self.assertEqual([b["url"] for b in forecast], ["http://127.0.0.1:18101", "http://127.0.0.1:18102"])
        self.assertEqual([b["url"] for b in llm], ["http://127.0.0.1:18111", "http://127.0.0.1:18112"])
        self.assertEqual([b["heal"]["http"]["url"] for b in forecast],
                         ["http://127.0.0.1:18101/admin/fail", "http://127.0.0.1:18102/admin/fail"])
        # Nothing else changed: shift the ports back and it is the team config again.
        text = json.dumps(config)
        for shifted, standard in ((18101, 8101), (18102, 8102), (18111, 8001), (18112, 8002)):
            text = text.replace(":%d" % shifted, ":%d" % standard)
        back = json.loads(text)
        back.update(port=8100, decision_log="router-decisions.jsonl")
        self.assertEqual(back, TEAM_CONFIG)

    def test_the_config_file_it_writes(self):
        ports = self.tool.ports_for(18100)
        path, tmp, note = self.tool.choose_config(18100, ports)
        self.addCleanup(shutil.rmtree, tmp, True)
        self.assertIsNone(note)
        self.assertEqual(os.path.dirname(path), tmp)
        self.assertTrue(os.path.basename(tmp).startswith("dev_cluster_"))
        with open(path) as handle:
            self.assertEqual(json.load(handle),
                             self.tool.router_config(ports, os.path.join(tmp, "router-decisions.jsonl")))
        self.tool.remove_tmp(tmp)
        self.assertFalse(os.path.exists(tmp))
        self.tool.remove_tmp(self.folder)                       # not a folder it made: left alone
        self.assertTrue(os.path.isdir(self.folder))

    def test_the_standard_cluster_uses_the_example_config_if_it_points_at_the_mocks(self):
        standard = self.tool.ports_for()
        self.tool.EXAMPLE_CONFIG = os.path.join(self.folder, "config.example.json")
        path, tmp, note = self.tool.choose_config(None, standard)           # no such file yet
        self.addCleanup(shutil.rmtree, tmp, True)
        self.assertIn("does not exist yet", note)
        with open(path) as handle:
            written = json.load(handle)
        self.assertEqual(written, self.tool.router_config(standard, os.path.join(tmp, "router-decisions.jsonl")))

        with open(self.tool.EXAMPLE_CONFIG, "w") as handle:
            json.dump(TEAM_CONFIG, handle)
        self.assertEqual(self.tool.choose_config(None, standard), (self.tool.EXAMPLE_CONFIG, None, None))

        elsewhere = json.loads(json.dumps(TEAM_CONFIG).replace("127.0.0.1:8101", "node-a:8101"))
        with open(self.tool.EXAMPLE_CONFIG, "w") as handle:
            json.dump(elsewhere, handle)
        path, tmp, note = self.tool.choose_config(None, standard)           # it would send work to a real box
        self.addCleanup(shutil.rmtree, tmp, True)
        self.assertNotEqual(path, self.tool.EXAMPLE_CONFIG)
        self.assertIn("does not point at the local mocks", note)

    def test_the_five_command_lines(self):
        pieces = self.tool.plan(self.tool.ports_for(18100), "/somewhere/config.json")
        self.assertEqual([piece["name"] for piece in pieces], ["forecast-A", "forecast-B", "llm-A", "llm-B", "router"])
        for piece in pieces:
            self.assertEqual(piece["cmd"][:2], [sys.executable, piece["script"]])
            self.assertTrue(piece["script"].startswith(REPO + os.sep))
            self.assertEqual(piece["url"], "http://127.0.0.1:%d" % piece["port"])
        by_name = dict((piece["name"], piece) for piece in pieces)
        self.assertEqual(os.path.relpath(by_name["forecast-B"]["script"], REPO), os.path.join("serve", "forecast_server.py"))
        self.assertEqual(by_name["forecast-B"]["cmd"][2:], ["--node", "B", "--mock", "--admin", "--port", "18102"])
        self.assertEqual(os.path.relpath(by_name["llm-A"]["script"], REPO), os.path.join("serve", "mock_llm.py"))
        self.assertEqual(by_name["llm-A"]["cmd"][2:], ["--node", "A", "--flavor", "llamacpp", "--admin", "--port", "18111"])
        self.assertEqual(os.path.relpath(by_name["router"]["script"], REPO), os.path.join("router", "router.py"))
        self.assertEqual(by_name["router"]["cmd"][2:], ["--config", "/somewhere/config.json", "--port", "18100"])
        self.assertEqual([piece["health"] for piece in pieces], ["/health"] * 4 + ["/router/health"])

    def test_the_six_curl_lines(self):
        lines = self.tool.curl_lines(self.tool.ports_for(18100))
        self.assertEqual(len(lines), 6)
        paths = []
        for what, command in lines:
            self.assertTrue(what)
            match = re.match(r"^curl -s http://127\.0\.0\.1:(\d+)(/\S+)( -H 'Content-Type: application/json' -d '(.*)')?$",
                             command)
            self.assertTrue(match, command)
            paths.append((int(match.group(1)), match.group(2)))
            if match.group(4):
                self.assertNotIn("'", match.group(4))           # safe inside the shell's single quotes
                self.assertIsInstance(json.loads(match.group(4)), dict)
        self.assertEqual(paths, [(18100, "/v1/forecast"), (18100, "/v1/ensemble"), (18100, "/v1/chat/completions"),
                                 (18100, "/router/status"), (18100, "/router/policy"), (18102, "/admin/fail")])

    def test_help(self):
        done = subprocess.run([sys.executable, TOOL, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(done.returncode, 0)
        for word in (b"up", b"down", b"status", b"--foreground", b"--ports-from"):
            self.assertIn(word, done.stdout)

    # ---- down's safety check

    def entry(self, name, pid, port, script=MOCK):
        return {"name": name, "pid": pid, "port": port, "url": "http://127.0.0.1:%d" % port,
                "health": "/health", "script": script, "log": ""}

    def write_state(self, pieces, key="19999", keep=None):
        state = dict(keep or {})
        state[key] = {"started": "a while ago", "tmp": None, "pieces": pieces}
        with open(self.tool.STATE_FILE, "w") as handle:
            json.dump(state, handle)

    def test_down_never_signals_a_process_it_did_not_start(self):
        port = free_port()
        server = MockServer("--port", str(port))                # a real piece of this repo, on a known port
        self.addCleanup(server.stop)
        stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.addCleanup(stranger.wait)
        self.addCleanup(stranger.kill)
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        self.write_state([
            self.entry("reused-pid", stranger.pid, port),                   # the PID now belongs to something else
            self.entry("other-port", server.proc.pid, port + 1),            # our script, but another port
            self.entry("other-piece", server.proc.pid, port, TOOL),         # this repo, but another script
            self.entry("outside", server.proc.pid, port, "/usr/bin/true"),  # a script outside this repo
            self.entry("gone", gone.pid, port),                             # no such process any more
            {"name": "junk", "pid": "12", "port": None},                    # not an entry we wrote
        ])
        code, said = self.run_tool("status")
        self.assertEqual(code, 1)
        self.assertEqual(said.count("DEAD"), 6, said)
        code, said = self.run_tool("down")
        self.assertEqual(code, 0)
        self.assertEqual(said.count("left alone"), 6, said)
        self.assertNotIn("stopped", said)
        self.assertNotIn("killed", said)
        time.sleep(0.3)
        self.assertIsNone(stranger.poll())                      # both are still running
        self.assertIsNone(server.proc.poll())
        self.assertEqual(server.get("/health")[0], 200)
        self.assertFalse(os.path.exists(self.tool.STATE_FILE))

    def test_down_stops_the_process_it_did_start(self):
        port = free_port()
        server = MockServer("--port", str(port))
        self.addCleanup(server.stop)
        self.write_state([self.entry("llm-A", server.proc.pid, port)])
        code, said = self.run_tool("status")
        self.assertEqual(code, 0)
        self.assertIn("alive, healthy", said)
        code, said = self.run_tool("down", "--ports-from", "20000")         # another cluster: this one stays
        self.assertEqual(code, 0)
        self.assertIn("No cluster", said)
        self.assertEqual(server.get("/health")[0], 200)
        code, said = self.run_tool("down", "--ports-from", "19999")
        self.assertEqual(code, 0)
        self.assertIn("stopped", said)
        self.assertEqual(server.proc.wait(timeout=10), 0)       # it got SIGTERM and left cleanly
        self.assertFalse(os.path.exists(self.tool.STATE_FILE))
        code, said = self.run_tool("down")
        self.assertEqual(code, 0)
        self.assertIn("Nothing to stop", said)
        code, said = self.run_tool("status")
        self.assertEqual(code, 1)

    # ---- up, with the pieces this test can count on

    def test_up_status_down(self):
        ports = self.only_llm_mocks()
        key = str(ports["router"])
        self.write_state([self.entry("long-gone", 99999999, 19998)])        # someone else's cluster, on record
        code, said = self.run_tool("up")
        self.addCleanup(self.run_tool, "down")                  # whatever happens below, stop them
        self.assertEqual(code, 1, said)                         # one piece is missing
        self.assertIn("not_written_yet.py does not exist yet", said)
        self.assertIn("2 of 3 pieces are up.", said)
        self.assertEqual(said.count("curl -s "), 6)
        self.assertEqual(sorted(self.read_state()), sorted(["19999", key]))
        cluster = self.read_state()[key]
        self.assertEqual([piece["name"] for piece in cluster["pieces"]], ["llm-A", "llm-B"])
        for piece in cluster["pieces"]:
            self.assertEqual(piece["port"], ports[piece["name"]])
            conn = http.client.HTTPConnection("127.0.0.1", piece["port"], timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            self.assertEqual((response.status, response.getheader("X-Mock-Node")), (200, piece["name"][-1]))
            conn.close()
            self.assertEqual(os.path.dirname(piece["log"]), self.tool.LOG_DIR)
            with open(piece["log"]) as handle:
                self.assertEqual(handle.readline().strip(), "listening on http://127.0.0.1:%d" % piece["port"])
        self.assertTrue(os.path.isfile(os.path.join(cluster["tmp"], "router-config.json")))

        code, said = self.run_tool("status")
        self.assertEqual(code, 1, said)                         # the other cluster on record is dead
        self.assertEqual(said.count("alive, healthy"), 2)
        code, said = self.run_tool("up")
        self.assertEqual(code, 1)
        self.assertIn("already up", said)

        # A third cluster appears on record while ours is up. Stopping ours must not lose it.
        self.write_state([self.entry("newcomer", 99999998, 30001)], key="30000", keep=self.read_state())
        code, said = self.run_tool("down", "--ports-from", key)
        self.assertEqual(code, 0)
        self.assertEqual(said.count("stopped"), 2, said)
        self.assertEqual(sorted(self.read_state()), ["19999", "30000"])
        self.assertFalse(os.path.exists(cluster["tmp"]))
        for piece in cluster["pieces"]:
            self.assertFalse(self.tool.port_taken(piece["port"]))
            self.assertIsNotNone(self.tool.why_not_ours(piece))
        code, said = self.run_tool("down")                      # the rest: nothing of it runs, so the file goes
        self.assertEqual(said.count("left alone"), 2, said)
        self.assertFalse(os.path.exists(self.tool.STATE_FILE))

    def test_up_does_not_start_a_piece_whose_port_is_taken(self):
        squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(squatter.close)
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        self.only_llm_mocks(**{"llm-A": squatter.getsockname()[1]})
        code, said = self.run_tool("up")
        self.addCleanup(self.run_tool, "down")
        self.assertEqual(code, 1)
        self.assertIn("port %d is taken" % squatter.getsockname()[1], said)
        (cluster,) = self.read_state().values()
        self.assertEqual([piece["name"] for piece in cluster["pieces"]], ["llm-B"])
        code, said = self.run_tool("down")
        self.assertEqual(said.count("stopped"), 1, said)

    def test_foreground_stops_everything_on_ctrl_c(self):
        ports = self.only_llm_mocks()
        inside = threading.Event()

        def press_ctrl_c():
            """Wait until both mocks answer, then send this process the signal Ctrl-C sends."""
            deadline = time.monotonic() + 30
            while inside.is_set() and time.monotonic() < deadline:
                if all(self.tool.healthy("http://127.0.0.1:%d" % ports[name], "/health") for name in ("llm-A", "llm-B")):
                    time.sleep(0.7)                             # let "up" print its table and settle
                    break
                time.sleep(0.1)
            if inside.is_set():
                os.kill(os.getpid(), signal.SIGINT)

        old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, old_handler)
        self.addCleanup(self.run_tool, "down")
        watcher = threading.Thread(target=press_ctrl_c)
        inside.set()
        watcher.start()
        try:
            code, said = self.run_tool("up", "--foreground")
        finally:
            inside.clear()
            watcher.join()
        self.assertEqual(code, 0, said)
        self.assertIn("Staying up", said)
        self.assertEqual(said.count("stopped"), 2, said)
        self.assertFalse(os.path.exists(self.tool.STATE_FILE))
        for name in ("llm-A", "llm-B"):
            self.assertFalse(self.tool.port_taken(ports[name]))


if __name__ == "__main__":
    unittest.main()
