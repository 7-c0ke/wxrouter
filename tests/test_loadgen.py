"""Tests for bench/loadgen.py and bench/compare.py, against tiny fake servers.

Run from the repo root (about 30 seconds):

    .venv/bin/python -m unittest tests/test_loadgen.py -v
    .venv/bin/python -m unittest tests.test_loadgen.NumbersTest -v

Written at the hackathon (Nov 13-14, 2026).

Every fake server binds 127.0.0.1 on a port the OS picks and is stopped when its test ends.
Nothing here needs a GPU, the real servers or the network.
"""

import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bench"))
import compare  # noqa: E402
import loadgen  # noqa: E402

SPEC_FIELDS = ("t_start", "latency_ms", "ttfb_ms", "status", "backend", "retries", "affinity", "kind",
               "units", "session")


class Fake(object):
    """A tiny stand-in for the router, a backend and the admin hooks, all on one port.

    work says what a work request (/v1/forecast, /v1/chat/completions) gets:
      "ok"       an answer after delay_s
      "hold"     an answer after hold_s (or when the test ends)
      "trickle"  the headers, then one byte every 0.1 s
      "down"     a 503
    POST /admin/fail {"mode":"down"} makes work requests get 503 for down_s seconds.
    """

    def __init__(self, work="ok", delay_s=0.01, hold_s=1.2, down_s=0.5, status_page=True, accept_policy=True):
        self.work, self.delay_s, self.hold_s, self.down_s = work, delay_s, hold_s, down_s
        self.status_page, self.accept_policy = status_page, accept_policy
        self.calls = []                 # (clock, method, path, headers, parsed body), in arrival order
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.policy = "static"
        self.served = 0
        self.down_until = 0.0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.handle(self, None)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else None
                except ValueError:
                    body = None
                fake.handle(self, body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05})
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def seen(self, path):
        """The calls to one path, in arrival order."""
        with self.lock:
            return [call for call in self.calls if call[2] == path]

    def status(self):
        backends = [{"name": "A", "healthy": True, "draining": False}, {"name": "B", "healthy": True, "draining": False}]
        return {"policy": self.policy, "uptime_s": 1.0,
                "counters": {"routed": self.served, "failovers": 0, "switches": 0, "heals": 0, "errors": 0},
                "pools": {"forecast": {"preferred": "A", "backends": backends},
                          "llm": {"preferred": "A", "backends": backends}},
                "decisions": []}

    def reply(self, h, code, obj, extra=None):
        data = json.dumps(obj).encode("utf-8")
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        for name, value in (extra or {}).items():
            h.send_header(name, value)
        h.end_headers()
        h.wfile.write(data)

    def handle(self, h, body):
        now = time.monotonic()
        with self.lock:
            self.calls.append((now, h.command, h.path, h.headers, body))
        try:
            if h.path == "/router/status" and self.status_page:
                self.reply(h, 200, self.status())
            elif h.path == "/router/policy" and self.status_page:
                if not self.accept_policy:
                    return self.reply(h, 400, {"error": "unknown policy"})
                self.policy = body["policy"]
                self.reply(h, 200, {"policy": self.policy})
            elif h.path.startswith("/admin/"):
                if h.path == "/admin/fail" and body.get("mode") == "down":
                    self.down_until = now + self.down_s
                self.reply(h, 200, {"ok": True})
            elif h.path in ("/v1/forecast", "/v1/chat/completions"):
                self.do_work(h, body)
            else:
                self.reply(h, 404, {"error": "no such path"})
        except (BrokenPipeError, ConnectionResetError):
            pass                        # the client gave up first: that is what some tests want

    def do_work(self, h, body):
        with self.lock:
            self.served += 1
            n = self.served
        backend = "A" if n % 2 else "B"
        extra = {"X-Backend": backend, "X-Router-Policy": self.policy, "X-Router-Retries": "0"}
        if self.work == "trickle":
            h.send_response(200)
            h.send_header("Content-Type", "text/plain")
            h.end_headers()
            for _ in range(50):
                if self.release.is_set():
                    break
                h.wfile.write(b"x")
                time.sleep(0.1)
            return
        if self.work == "hold":
            self.release.wait(self.hold_s)
        elif self.work == "down" or time.monotonic() < self.down_until:
            return self.reply(h, 503, {"error": "node is down"}, extra)
        else:
            time.sleep(self.delay_s)
        if h.path == "/v1/forecast":
            return self.reply(h, 200, {"request_id": body.get("request_id"), "node": backend}, extra)
        extra["X-Router-Affinity"] = "hit"
        if not body.get("stream"):
            return self.reply(h, 200, {"choices": [{"message": {"role": "assistant", "content": "fine"}}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 48,
                                                  "prompt_tokens_details": {"cached_tokens": 60}}}, extra)
        h.send_response(200)            # a streamed answer: first chunk after 0.15 s, the rest 0.25 s later
        h.send_header("Content-Type", "text/event-stream")
        for name, value in extra.items():
            h.send_header(name, value)
        h.end_headers()
        time.sleep(0.15)
        h.wfile.write(b'data: {"choices":[{"delta":{"content":"fine"}}]}\n\n')
        time.sleep(0.25)
        h.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":200,"prompt_tokens_details":{"cached_tokens":50}}}\n\n')
        h.wfile.write(b"data: [DONE]\n\n")


def closed_port_url():
    """An address nothing listens on."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return "http://127.0.0.1:%d" % port


def record(t, latency, status, backend=None, retries=None, kind="forecast", **more):
    """A hand-made record for the arithmetic tests."""
    out = {"t_start": t, "t_sched": t, "latency_ms": latency, "status": status, "backend": backend,
           "retries": retries, "kind": kind}
    out.update(more)
    return out


class ScheduleTest(unittest.TestCase):
    """The schedule is made from the seed alone."""

    def test_same_seed_same_schedule_other_seed_other_schedule(self):
        for kind in loadgen.KINDS:
            first = loadgen.build_schedule(1, 30, 4, kind)
            again = loadgen.build_schedule(1, 30, 4, kind)
            other = loadgen.build_schedule(2, 30, 4, kind)
            self.assertGreater(len(first), 60)
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(again, sort_keys=True))
            self.assertNotEqual([item["t"] for item in first], [item["t"] for item in other])
            self.assertNotEqual([item["body"] for item in first][:20], [item["body"] for item in other][:20])

    def test_arrivals_are_in_order_inside_the_run_and_near_the_rate(self):
        times = [item["t"] for item in loadgen.build_schedule(3, 200, 10, "forecast")]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[0], 0.0)
        self.assertLess(times[-1], 200.0)
        self.assertLess(abs(len(times) - 2000), 5 * 2000 ** 0.5)       # Poisson: within 5 standard deviations
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertAlmostEqual(sum(gaps) / len(gaps), 0.1, delta=0.01)

    def test_every_kind_and_scenario_shares_the_arrival_times(self):
        base = [item["t"] for item in loadgen.build_schedule(5, 30, 4, "forecast", "steady")]
        for kind, scenario in (("llm", "steady"), ("mixed", "contention"), ("forecast", "failure"),
                               ("forecast", "recovery")):
            self.assertEqual(base, [item["t"] for item in loadgen.build_schedule(5, 30, 4, kind, scenario)])
        same_bodies = loadgen.build_schedule(5, 30, 4, "forecast", "contention")
        self.assertEqual([item["body"] for item in loadgen.build_schedule(5, 30, 4, "forecast")],
                         [item["body"] for item in same_bodies])

    def test_forecast_requests_follow_the_contract_and_the_size_mix(self):
        schedule = loadgen.build_schedule(1, 2000, 5, "forecast")
        sizes = {"small": 0, "medium": 0, "large": 0}
        for n, item in enumerate(schedule):
            body = item["body"]
            self.assertEqual(item["i"], n)
            self.assertEqual((item["kind"], item["path"], item["session"]), ("forecast", "/v1/forecast", None))
            self.assertEqual(sorted(body), ["init", "lead_hours", "member_offset", "members", "point",
                                            "request_id", "with_truth"])
            self.assertEqual(body["init"], "latest")
            self.assertEqual(body["point"], {"lat": 41.9602, "lon": -87.9316})
            self.assertEqual(item["units"], body["members"] * body["lead_hours"] // 6)
            if body["members"] == 1:
                self.assertIn(body["lead_hours"], (24, 48))
                sizes["small"] += 1
            elif body["members"] == 20:
                self.assertEqual(body["lead_hours"], 120)
                sizes["large"] += 1
            else:
                self.assertTrue(4 <= body["members"] <= 8)
                self.assertEqual(body["lead_hours"], 72)
                sizes["medium"] += 1
        total = float(len(schedule))
        self.assertEqual(len(set(item["body"]["request_id"] for item in schedule)), len(schedule))
        self.assertAlmostEqual(sizes["small"] / total, 0.70, delta=0.02)
        self.assertAlmostEqual(sizes["medium"] / total, 0.25, delta=0.02)
        self.assertAlmostEqual(sizes["large"] / total, 0.05, delta=0.01)
        self.assertEqual(set(item["body"]["members"] for item in schedule), set([1, 4, 5, 6, 7, 8, 20]))

    def test_llm_sessions_grow_and_keep_their_prefix(self):
        schedule = loadgen.build_schedule(1, 60, 2, "llm", sessions=3)
        last = {}
        grew = 0
        for item in schedule:
            body, session = item["body"], item["session"]
            messages = body["messages"]
            self.assertEqual(item["path"], "/v1/chat/completions")
            self.assertEqual(item["headers"], {"X-Session-Id": session})
            self.assertEqual((body["max_tokens"], body["stream"], item["units"]), (48, False, None))
            self.assertEqual([m["role"] for m in messages], (["user", "assistant"] * len(messages))[:len(messages)])
            self.assertEqual(len(messages) % 2, 1)                     # it ends with a user turn
            self.assertLessEqual(len(messages), 2 * loadgen.LLM_MAX_TURNS - 1)
            self.assertTrue(170 <= len(messages[-1]["content"]) <= 230, len(messages[-1]["content"]))
            if session in last and len(messages) > 1:
                before = last[session]
                self.assertEqual(len(messages), len(before) + 2)       # one assistant turn, one user turn
                self.assertEqual(messages[:len(before)], before)       # the prompt keeps its prefix
                grew += 1
            last[session] = messages
        self.assertEqual(len(last), 3)
        self.assertGreater(grew, 60)
        # After LLM_MAX_TURNS turns a session starts a new conversation: count the fresh starts.
        turns = [sum(1 for item in schedule if item["session"] == session) for session in last]
        fresh = sum(-(-n // loadgen.LLM_MAX_TURNS) for n in turns)
        self.assertGreater(fresh, 3)
        self.assertEqual(sum(1 for item in schedule if len(item["body"]["messages"]) == 1), fresh)
        streamed = loadgen.build_schedule(1, 60, 2, "llm", sessions=3, stream=True, model="tiny")
        self.assertTrue(all(item["body"]["stream"] is True and item["body"]["model"] == "tiny" for item in streamed))

    def test_mixed_is_half_and_half(self):
        schedule = loadgen.build_schedule(1, 500, 4, "mixed")
        forecasts = sum(1 for item in schedule if item["kind"] == "forecast")
        self.assertLess(abs(forecasts - len(schedule) / 2.0), 5 * (len(schedule) / 4.0) ** 0.5)
        self.assertEqual(set(item["kind"] for item in schedule), set(["forecast", "llm"]))

    def test_burst_is_four_times_denser_between_a_third_and_a_half(self):
        times = [item["t"] for item in loadgen.build_schedule(1, 600, 5, "forecast", "burst")]
        before = sum(1 for t in times if t < 200)
        during = sum(1 for t in times if 200 <= t < 300)
        after = sum(1 for t in times if t >= 300)
        self.assertLess(abs(before - 1000), 5 * 1000 ** 0.5)
        self.assertLess(abs(during - 2000), 5 * 2000 ** 0.5)
        self.assertLess(abs(after - 1500), 5 * 1500 ** 0.5)

    def test_scenario_events_are_at_the_right_fractions(self):
        def brief(scenario):
            return [(e["at_s"], e["name"], (e.get("http") or {}).get("url"), (e.get("http") or {}).get("json"))
                    for e in loadgen.scenario_events(scenario, 60, "http://b:1")]

        self.assertEqual(brief("steady"), [])
        self.assertEqual(brief("contention"), [(20.0, "contention_on", "http://b:1/admin/contention", {"factor": 1.8}),
                                               (40.0, "contention_off", "http://b:1/admin/contention", {"factor": 1.0})])
        self.assertEqual(brief("failure"), [(30.0, "fail", "http://b:1/admin/fail", {"mode": "down"})])
        self.assertEqual(brief("recovery"), [(20.0, "fail", "http://b:1/admin/fail", {"mode": "down"}),
                                             (40.0, "recover", "http://b:1/admin/fail", {"mode": "ok"})])
        self.assertEqual(brief("burst"), [(20.0, "burst_start", None, None), (30.0, "burst_end", None, None)])
        self.assertEqual([e["label"] for e in loadgen.scenario_events("recovery", 60, "http://b:1")],
                         ["B dies", "B comes back"])


class NumbersTest(unittest.TestCase):
    """Percentiles and the summary, on data small enough to check by hand."""

    def test_percentile_is_nearest_rank(self):
        hundred = list(range(1, 101))
        self.assertEqual([loadgen.percentile(hundred, p) for p in (50, 95, 99, 100)], [50, 95, 99, 100])
        five = [35, 20, 15, 50, 40]                                    # not sorted on purpose
        self.assertEqual([loadgen.percentile(five, p) for p in (5, 30, 40, 50, 100)], [15, 20, 20, 35, 50])
        self.assertEqual(loadgen.percentile([1, 2, 3, 4], 50), 2)      # rank ceil(2.0) = 2: no interpolation
        self.assertEqual(loadgen.percentile(list(range(1, 21)), 95), 19)
        self.assertEqual(loadgen.percentile([7], 99), 7)
        self.assertIsNone(loadgen.percentile([], 95))

    def test_summary_on_known_data(self):
        records = [
            record(1, 100.0, 200, "A", 0), record(2, 200.0, 200, "A", 0), record(3, 300.0, 200, "B", 1),
            record(11, 400.0, 200, "A", 0), record(12, 400.0, "timeout"), record(13, 50.0, 503, "B", 1),
            record(21, 500.0, 200, "B", 0), record(15, None, "shed"), record(25, 700.0, 200, "A", 0),
            record(29.8, 600.0, 200, "A", 0),                          # its answer comes at 30.4 s: too late
        ]
        events = [{"name": "fail", "at_s": 11.5, "t_actual": 11.5}]
        s = loadgen.summarize(records, 30.0, events, {"routed": 10, "failovers": 1},
                              {"routed": 20, "failovers": 3, "switches": 2})
        self.assertEqual((s["requests"], s["ok"], s["errors"]), (10, 7, 3))
        self.assertEqual(s["errors_by_type"], {"timeout": 1, "503": 1, "shed": 1})
        self.assertEqual(s["error_rate"], 0.3)
        self.assertEqual(s["throughput_rps"], 0.2)                     # 6 ok answers within the 30 s
        self.assertEqual(s["latency_ms"], {"p50": 400.0, "p95": 700.0, "p99": 700.0, "mean": 400.0})
        self.assertEqual(s["p95_by_third_ms"], [300.0, 400.0, 700.0])
        self.assertEqual(s["share"], {"A": 0.625, "B": 0.375})
        self.assertEqual(s["retries"], 2)
        self.assertEqual(s["router"]["delta"], {"failovers": 2, "routed": 10})
        self.assertEqual(s["fail_event_s"], 11.5)
        self.assertEqual(s["time_to_recover_s"], 3.5)                  # clean from the shed at 15 s on
        self.assertIsNone(s["cached_prompt_share"])
        self.assertIsNone(s["ttfb_ms"])
        text = loadgen.format_summary(s)
        for part in ("requests", "p95 700.0", "A 62.5 %", "503: 1, shed: 1, timeout: 1", "recover s", "3.50",
                     "routed +10", "fail at 11.50 s"):
            self.assertIn(part, text)
        # A fail event that did not work is no failure: there is nothing to recover from.
        missed = loadgen.summarize(records, 30.0, [{"name": "fail", "at_s": 11.5, "t_actual": 11.5, "ok": False}])
        self.assertEqual((missed["fail_event_s"], missed["time_to_recover_s"]), (None, None))
        self.assertNotIn("recover s", loadgen.format_summary(missed))

    def test_summary_of_nothing_and_of_a_plain_backend(self):
        s = loadgen.summarize([], 10.0)
        self.assertEqual((s["requests"], s["ok"], s["errors"], s["error_rate"], s["throughput_rps"]), (0, 0, 0, 0.0, 0.0))
        self.assertEqual(s["latency_ms"], {"p50": None, "p95": None, "p99": None, "mean": None})
        self.assertEqual(s["p95_by_third_ms"], [None, None, None])
        self.assertEqual((s["share"], s["router"], s["time_to_recover_s"], s["fail_event_s"]), ({}, None, None, None))
        self.assertIn("no /router/status", loadgen.format_summary(s))

    def test_cached_prompt_share_and_first_byte(self):
        records = [record(1, 900.0, 200, kind="llm", prompt_tokens=100, cached_tokens=60, ttfb_ms=200.0),
                   record(2, 700.0, 200, kind="llm", prompt_tokens=300, cached_tokens=100, ttfb_ms=100.0),
                   record(3, 800.0, 200, kind="llm", prompt_tokens=None, cached_tokens=None),
                   record(4, 100.0, 500, kind="llm", prompt_tokens=50, cached_tokens=50),
                   record(5, 20.0, 200, kind="forecast")]
        s = loadgen.summarize(records, 10.0)
        self.assertEqual(s["cached_prompt_share"], 0.4)                # (60 + 100) / (100 + 300), ok answers only
        self.assertEqual(s["ttfb_ms"], {"p50": 100.0, "p95": 200.0, "p99": 200.0, "mean": 150.0})
        self.assertEqual(s["by_kind"]["llm"]["latency_ms"]["p50"], 800.0)
        self.assertEqual((s["by_kind"]["llm"]["requests"], s["by_kind"]["forecast"]["ok"]), (4, 1))

    def test_usage_is_read_from_three_shapes(self):
        openai = json.dumps({"usage": {"prompt_tokens": 90, "prompt_tokens_details": {"cached_tokens": 30}}})
        llamacpp = json.dumps({"timings": {"cache_n": 40, "prompt_n": 10}})
        stream = 'data: {"choices":[{"delta":{}}]}\n\ndata: {"usage":{"prompt_tokens":70,' \
                 '"prompt_tokens_details":{"cached_tokens":7}}}\n\ndata: [DONE]\n\n'
        self.assertEqual(loadgen.read_usage(openai.encode(), False), (90, 30))
        self.assertEqual(loadgen.read_usage(llamacpp.encode(), False), (50, 40))
        self.assertEqual(loadgen.read_usage(stream.encode(), True), (70, 7))
        self.assertEqual(loadgen.read_usage(b'{"usage": null}', False), (None, None))
        self.assertEqual(loadgen.read_usage(b"not json", False), (None, None))

    def test_time_to_recover(self):
        def sent(*pairs):
            return [record(t, 10.0, 200 if ok else 503) for t, ok in pairs]

        clean = sent((1, False), (11, True), (12, True), (19, True))
        self.assertEqual(loadgen.time_to_recover(clean, 10.0, 20.0), 0.0)          # no error at all after the event
        broken = sent((10.5, False), (11, False), (12.25, False), (13, True), (16, True), (19, True))
        self.assertEqual(loadgen.time_to_recover(broken, 10.0, 20.0), 2.25)        # clean from the last error on
        late = sent((10.5, False), (16, False), (17, True), (19, True))
        self.assertIsNone(loadgen.time_to_recover(late, 10.0, 20.0))               # no 5 clean seconds left in the run
        relapse = sent((10.5, False), (11, True), (14, False), (15, True), (19.5, True), (25, True))
        self.assertEqual(loadgen.time_to_recover(relapse, 10.0, 30.0), 4.0)        # the first calm spell was too short
        self.assertEqual(loadgen.time_to_recover(relapse, 10.0, 30.0, window=3.0), 0.5)   # long enough for 3 s
        again = sent((10.5, False), (11, True), (17, False), (18, True))
        self.assertEqual(loadgen.time_to_recover(again, 10.0, 20.0), 0.5)          # 6.5 clean seconds, then a new error
        self.assertIsNone(loadgen.time_to_recover(sent((10.5, False)), 10.0, 30.0))   # silence is not recovery
        self.assertIsNone(loadgen.time_to_recover([], 10.0, 30.0))

    def test_window_p95(self):
        records = [record(0.5, 10.0, 200), record(4.9, 30.0, 200), record(5.0, 99.0, 200), record(7.0, 5.0, 503),
                   record(11.9, 50.0, 200), record(12.0, 70.0, 200)]
        windows = loadgen.window_p95(records, 12.0, 5.0)
        self.assertEqual([(w["t0"], w["t1"], w["ok"], w["errors"], w["p95_ms"]) for w in windows],
                         [(0.0, 5.0, 2, 0, 30.0), (5.0, 10.0, 1, 1, 99.0), (10.0, 12.0, 2, 0, 70.0)])
        self.assertEqual([w["p95_ms"] for w in loadgen.window_p95([], 10.0, 5.0)], [None, None])


class RunTest(unittest.TestCase):
    """Whole runs against a fake target."""

    def setUp(self):
        self.fakes = []
        self.tmp = tempfile.mkdtemp(prefix="loadgen-test-")
        self.out = os.path.join(self.tmp, "run.jsonl")

    def tearDown(self):
        for fake in self.fakes:
            fake.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake(self, **settings):
        fake = Fake(**settings)
        self.fakes.append(fake)
        return fake

    def load(self, fake, **changes):
        settings = {"target": fake.url, "out": self.out, "duration": 1.0, "rate": 20.0, "timeout": 5.0}
        settings.update(changes)
        return loadgen.run(loadgen.options(**settings), quiet=True)

    def other_warnings(self, summary):
        """The run's warnings, without the one about a busy machine (this laptop may be busy)."""
        return [text for text in summary["warnings"] if "fell behind" not in text]

    def test_record_fields_and_files(self):
        fake = self.fake()
        summary, records = self.load(fake)
        schedule = loadgen.build_schedule(1, 1.0, 20.0, "forecast")
        with open(self.out) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(lines, records)
        self.assertEqual(len(lines), len(schedule))
        self.assertGreater(len(lines), 5)
        for item, rec in zip(schedule, lines):
            for field in SPEC_FIELDS + ("i", "t_sched", "prompt_tokens", "cached_tokens", "error"):
                self.assertIn(field, rec)
            self.assertEqual(list(rec)[:len(SPEC_FIELDS)], list(SPEC_FIELDS))
            self.assertEqual((rec["i"], rec["kind"], rec["units"], rec["session"]), (item["i"], "forecast", item["units"], None))
            self.assertEqual((rec["status"], rec["retries"], rec["affinity"], rec["ttfb_ms"], rec["error"]),
                             (200, 0, None, None, None))
            self.assertIn(rec["backend"], ("A", "B"))
            self.assertTrue(5.0 <= rec["latency_ms"] <= 1000.0, rec)
            self.assertAlmostEqual(rec["t_sched"], item["t"], places=3)
            self.assertTrue(-0.001 <= rec["t_start"] - rec["t_sched"] <= 0.25, rec)   # sent on time
        sent = [call[4] for call in fake.seen("/v1/forecast")]
        self.assertEqual(sorted(body["request_id"] for body in sent), [item["body"]["request_id"] for item in schedule])
        self.assertEqual(sorted(sent, key=lambda body: body["request_id"]), [item["body"] for item in schedule])
        with open(self.out + ".summary.json") as f:
            self.assertEqual(json.load(f), json.loads(json.dumps(summary)))
        self.assertEqual((summary["requests"], summary["ok"], summary["errors"]), (len(lines), len(lines), 0))
        self.assertEqual(summary["router"]["delta"]["routed"], len(lines))
        self.assertAlmostEqual(sum(summary["share"].values()), 1.0, places=3)
        self.assertEqual(sorted(summary["share"]), ["A", "B"])
        self.assertEqual((summary["settings"]["seed"], summary["settings"]["scenario"]), (1, "steady"))
        self.assertEqual(self.other_warnings(summary), [])

    def test_same_seed_sends_the_same_requests_whatever_the_answers(self):
        healthy, broken = self.fake(), self.fake(work="down")
        _, good = self.load(healthy, seed=7)
        _, bad = self.load(broken, seed=7, out=os.path.join(self.tmp, "bad.jsonl"))
        self.assertEqual([r["status"] for r in bad], [503] * len(bad))
        self.assertEqual(bad[0]["error"], "node is down")
        self.assertEqual([(r["i"], r["t_sched"], r["units"]) for r in good], [(r["i"], r["t_sched"], r["units"]) for r in bad])

        def bodies(fake):
            return sorted((call[4] for call in fake.seen("/v1/forecast")), key=lambda body: body["request_id"])

        self.assertEqual(bodies(healthy), bodies(broken))
        self.assertGreater(len(bodies(healthy)), 5)

    def test_llm_requests_carry_the_session_and_report_the_cache(self):
        fake = self.fake()
        summary, records = self.load(fake, kind="llm", sessions=2, model="tiny")
        calls = fake.seen("/v1/chat/completions")
        self.assertEqual(len(calls), len(records))
        self.assertEqual(sorted(set(call[3].get("X-Session-Id") for call in calls)), ["seed1-s00", "seed1-s01"])
        for call in calls:
            self.assertEqual((call[4]["max_tokens"], call[4]["stream"], call[4]["model"]), (48, False, "tiny"))
        for rec in records:
            self.assertEqual((rec["kind"], rec["status"], rec["affinity"], rec["units"], rec["ttfb_ms"]),
                             ("llm", 200, "hit", None, None))
            self.assertIn(rec["session"], ("seed1-s00", "seed1-s01"))
            self.assertEqual((rec["prompt_tokens"], rec["cached_tokens"]), (100, 60))
        self.assertEqual(summary["cached_prompt_share"], 0.6)
        self.assertIn("cached prompt", loadgen.format_summary(summary))

    def test_stream_measures_time_to_first_byte(self):
        fake = self.fake()
        summary, records = self.load(fake, kind="llm", stream=True, rate=6.0)
        self.assertGreater(len(records), 1)
        for rec in records:
            self.assertEqual(rec["status"], 200)
            self.assertTrue(120.0 <= rec["ttfb_ms"] <= 500.0, rec)                 # the fake's first chunk: 150 ms
            self.assertGreaterEqual(rec["latency_ms"], rec["ttfb_ms"] + 200.0)     # the rest: 250 ms later
            self.assertEqual((rec["prompt_tokens"], rec["cached_tokens"]), (200, 50))
        self.assertTrue(all(call[4]["stream"] is True for call in fake.seen("/v1/chat/completions")))
        self.assertEqual(summary["cached_prompt_share"], 0.25)
        self.assertTrue(120.0 <= summary["ttfb_ms"]["p50"] <= 500.0)

    def test_policy_and_resets_come_before_the_load(self):
        fake = self.fake()
        summary, records = self.load(fake, policy="adaptive", admin_a=fake.url, admin_b=fake.url, duration=0.5)
        paths = [call[2] for call in fake.calls if call[2] != "/router/status"]
        self.assertEqual(paths[:3], ["/router/policy", "/admin/reset", "/admin/reset"])
        self.assertEqual(fake.seen("/router/policy")[0][4], {"policy": "adaptive"})
        self.assertEqual(set(paths[3:]), set(["/v1/forecast"]))
        self.assertEqual((summary["settings"]["policy"], fake.policy), ("adaptive", "adaptive"))
        self.assertEqual(summary["router_policy"], "adaptive")
        self.assertIn("policy adaptive", loadgen.format_summary(summary))

    def test_setup_problems_stop_the_run_with_a_reason(self):
        fake = self.fake(accept_policy=False)
        with self.assertRaisesRegex(loadgen.SetupError, "could not set the policy 'nope'.*HTTP 400"):
            self.load(fake, policy="nope")
        with self.assertRaisesRegex(loadgen.SetupError, "could not reset node B"):
            self.load(fake, admin_b=closed_port_url())
        with self.assertRaisesRegex(loadgen.SetupError, "give --admin-b"):
            self.load(fake, scenario="failure")
        with self.assertRaisesRegex(loadgen.SetupError, "is not answering"):
            self.load(fake, target=closed_port_url())
        with self.assertRaisesRegex(loadgen.SetupError, "must be above 0"):
            self.load(fake, rate=0.0)
        self.assertEqual(fake.seen("/v1/forecast"), [])
        self.assertFalse(os.path.exists(self.out))

    def test_a_plain_backend_works_without_router_status(self):
        fake = self.fake(status_page=False)
        summary, records = self.load(fake, duration=0.5)
        self.assertIsNone(summary["router"])
        self.assertIsNone(summary["router_policy"])
        self.assertEqual(summary["ok"], len(records))
        self.assertGreater(len(records), 2)
        self.assertIn("policy not set", loadgen.format_summary(summary))

    def test_contention_events_hit_the_admin_hook_on_time(self):
        fake = self.fake()
        summary, records = self.load(fake, scenario="contention", duration=3.0, rate=8.0,
                                     admin_a=fake.url, admin_b=fake.url)
        resets, hits = fake.seen("/admin/reset"), fake.seen("/admin/contention")
        work = fake.seen("/v1/forecast")
        self.assertEqual(len(resets), 2)
        self.assertLess(resets[-1][0], work[0][0])                     # both resets before the first request
        self.assertEqual([call[4] for call in hits], [{"factor": 1.8}, {"factor": 1.0}])
        self.assertAlmostEqual(hits[1][0] - hits[0][0], 1.0, delta=0.4)
        events = summary["events"]
        self.assertEqual([(e["name"], e["at_s"], e["ok"], e["result"]) for e in events],
                         [("contention_on", 1.0, True, "http 200"), ("contention_off", 2.0, True, "http 200")])
        self.assertAlmostEqual(events[0]["t_actual"], 1.0, delta=0.4)
        self.assertAlmostEqual(events[1]["t_actual"], 2.0, delta=0.4)
        # Seen from the server: the requests that came before the first event are those due before it.
        due_before = sum(1 for r in records if r["t_sched"] < 1.0)
        came_before = sum(1 for call in work if call[0] < hits[0][0])
        self.assertLessEqual(abs(due_before - came_before), 2)
        self.assertIsNone(summary["time_to_recover_s"])
        self.assertEqual(self.other_warnings(summary), [])

    def test_failure_scenario_measures_the_recovery(self):
        self.addCleanup(setattr, loadgen, "RECOVER_WINDOW_S", loadgen.RECOVER_WINDOW_S)
        loadgen.RECOVER_WINDOW_S = 1.0                                 # a 5 s window would need a long test
        fake = self.fake(down_s=0.5)                                   # after the fail call: 503 for half a second
        summary, records = self.load(fake, scenario="failure", duration=4.0, rate=20.0, admin_b=fake.url)
        self.assertEqual(fake.seen("/admin/fail")[0][4], {"mode": "down"})
        self.assertAlmostEqual(summary["fail_event_s"], 2.0, delta=0.4)
        failed = [r for r in records if r["status"] == 503]
        self.assertGreater(len(failed), 2)
        self.assertEqual(summary["errors_by_type"], {"503": len(failed)})
        self.assertTrue(all(summary["fail_event_s"] - 0.05 <= r["t_start"] <= summary["fail_event_s"] + 0.8 for r in failed))
        self.assertTrue(0.2 <= summary["time_to_recover_s"] <= 0.8, summary["time_to_recover_s"])
        self.assertAlmostEqual(summary["time_to_recover_s"], max(r["t_start"] for r in failed) - summary["fail_event_s"],
                               places=2)

    def test_shed_is_recorded(self):
        fake = self.fake(work="hold", hold_s=1.3)                      # answers come after the 1 s of arrivals
        summary, records = self.load(fake, rate=40.0, max_inflight=3)
        shed = [r for r in records if r["status"] == "shed"]
        sent = [r for r in records if r["status"] != "shed"]
        self.assertEqual(len(sent), 3)                                 # the first three fill the limit
        self.assertEqual([r["i"] for r in sent], [0, 1, 2])
        self.assertEqual([r["status"] for r in sent], [200, 200, 200])
        self.assertGreater(len(shed), 10)
        for rec in shed:
            self.assertEqual((rec["latency_ms"], rec["backend"], rec["retries"]), (None, None, None))
            self.assertAlmostEqual(rec["t_start"], rec["t_sched"], delta=0.25)
        self.assertEqual(len(fake.seen("/v1/forecast")), 3)            # the shed ones were never sent
        self.assertEqual(summary["errors_by_type"], {"shed": len(shed)})
        self.assertEqual((summary["ok"], summary["errors"]), (3, len(shed)))
        self.assertEqual(summary["throughput_rps"], 0.0)               # the three answers came after the duration

    def test_timeout_is_recorded_and_bounds_the_wait(self):
        fake = self.fake(work="hold", hold_s=4.0)
        started = time.monotonic()
        summary, records = self.load(fake, duration=0.5, rate=10.0, timeout=0.4)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertGreater(len(records), 1)
        for rec in records:
            self.assertEqual(rec["status"], "timeout")
            self.assertTrue(390.0 <= rec["latency_ms"] <= 1000.0, rec)
        self.assertEqual(summary["errors_by_type"], {"timeout": len(records)})
        self.assertEqual(summary["latency_ms"]["p95"], None)

    def test_a_trickling_answer_cannot_outlive_the_timeout(self):
        fake = self.fake(work="trickle")
        summary, records = self.load(fake, duration=0.5, rate=10.0, timeout=0.6)
        self.assertGreater(len(records), 1)
        for rec in records:
            self.assertEqual(rec["status"], "timeout")
            self.assertTrue(590.0 <= rec["latency_ms"] <= 1200.0, rec)

    def test_refused_is_recorded(self):
        host, port, base = loadgen.split_url(closed_port_url())
        item = loadgen.build_schedule(1, 5, 4, "forecast")[0]
        got = loadgen.send_one((host, port, base), item, 2.0)
        self.assertEqual(got["status"], "refused")
        self.assertIsNone(got["backend"])

    def test_events_file_calls_and_commands(self):
        fake = self.fake()
        path = os.path.join(self.tmp, "events.json")
        with open(path, "w") as f:
            json.dump([{"at_s": 0.3, "name": "fail", "http": {"url": fake.url + "/admin/custom", "json": {"x": 1}}},
                       {"at_s": 0.5, "cmd": [sys.executable, "-c", "pass"]},
                       {"at_s": 0.6, "name": "bad", "cmd": [sys.executable, "-c", "import sys; sys.exit(3)"]}], f)
        summary, records = self.load(fake, events=path)
        events = summary["events"]
        self.assertEqual([(e["name"], e["ok"], e["result"]) for e in events],
                         [("fail", True, "http 200"), ("cmd", True, "exit 0"), ("bad", False, "exit 3")])
        for event, due in zip(events, (0.3, 0.5, 0.6)):
            self.assertAlmostEqual(event["t_actual"], due, delta=0.4)
        self.assertEqual(fake.seen("/admin/custom")[0][4], {"x": 1})
        self.assertAlmostEqual(summary["fail_event_s"], 0.3, delta=0.4)       # an event named "fail" starts the clock
        self.assertEqual(len(self.other_warnings(summary)), 1)
        self.assertIn("the event bad did not work", self.other_warnings(summary)[0])
        self.assertIn("WARNING: the event bad", loadgen.format_summary(summary))

    def test_a_bad_events_file_is_refused(self):
        fake = self.fake()
        path = os.path.join(self.tmp, "events.json")
        for bad, why in (([{"at_s": 5, "cmd": ["true"]}], "outside the run"),
                         ([{"at_s": 0.1}], '"http" or "cmd"'),
                         ([{"at_s": 0.1, "cmd": "true"}], "list of strings"),
                         ([{"cmd": ["true"]}], '"at_s"'),
                         ({"at_s": 0.1}, "JSON list")):
            with open(path, "w") as f:
                json.dump(bad, f)
            with self.assertRaisesRegex(loadgen.SetupError, why):
                self.load(fake, events=path)
        self.assertEqual(fake.seen("/v1/forecast"), [])

    def test_command_line(self):
        fake = self.fake()
        script = os.path.join(REPO, "bench", "loadgen.py")
        done = subprocess.run([sys.executable, script, "--target", fake.url, "--duration", "1", "--rate", "10",
                               "--seed", "3", "--out", self.out], cwd=REPO, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        for part in ("requests", "latency ms", "p95", "throughput", "wrote"):
            self.assertIn(part, done.stdout)
        self.assertTrue(os.path.exists(self.out) and os.path.exists(self.out + ".summary.json"))
        refused = subprocess.run([sys.executable, script, "--target", closed_port_url(), "--duration", "1",
                                  "--out", self.out + ".no"], cwd=REPO, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("is not answering", refused.stderr)
        self.assertFalse(os.path.exists(self.out + ".no"))


def row(policy, p95, errors=0, recover=None, fail_at=None, share=None, warnings=None):
    """A hand-made run for the report tests."""
    if p95 is None:
        windows = [{"t0": 5.0 * k, "t1": 5.0 * k + 5.0, "ok": 0, "errors": 20, "p95_ms": None} for k in range(5)]
    else:
        windows = [{"t0": 5.0 * k, "t1": 5.0 * k + 5.0, "ok": 20, "errors": 0, "p95_ms": value}
                   for k, value in enumerate([p95, p95 * 2, None, p95 * 3, p95])]
    summary = {"ok": 100 - errors, "requests": 100, "errors": errors,
               "latency_ms": {"p50": None if p95 is None else p95 / 2.0, "p95": p95,
                              "p99": None if p95 is None else p95 * 1.5, "mean": None if p95 is None else p95 / 1.6},
               "share": share if share is not None else {"A": 0.6, "B": 0.4}, "retries": 2,
               "router": {"delta": {"routed": 100, "failovers": 1, "switches": 3}},
               "fail_event_s": fail_at, "time_to_recover_s": recover, "warnings": warnings or [],
               "events": [{"name": "contention_on", "label": "B slows down", "at_s": 8.0, "t_actual": 8.01},
                          {"name": "contention_off", "label": "B back to normal", "at_s": 16.0, "t_actual": 16.02}]}
    return {"scenario": "contention", "policy": policy, "jsonl": "contention_%s.jsonl" % policy,
            "summary": summary, "windows": windows}


SETTINGS = {"target": "http://127.0.0.1:8100", "kind": "forecast", "duration_s": 25.0, "rate": 4.0, "seed": 1,
            "policies": ["static", "round_robin", "adaptive"], "scenarios": ["contention"], "admin_a": None,
            "admin_b": "http://127.0.0.1:8102", "timeout_s": 120.0, "max_inflight": 256, "sessions": 8,
            "stream": False, "model": "default", "events_file": None, "pause_s": 3.0, "window_s": 5.0}


class ReportTest(unittest.TestCase):
    """RESULTS.md and the chart, from hand-made runs."""

    def test_reading_is_computed_against_round_robin(self):
        rows = [row("static", 400.0), row("round_robin", 200.0), row("adaptive", 150.0)]
        self.assertEqual(compare.reading(rows),
                         "Reading: adaptive had the best p95, 150.0 ms: 25.0% lower than round_robin (200.0 ms).")
        rows = [row("round_robin", 200.0), row("adaptive", 250.0), row("static", 300.0)]
        self.assertEqual(compare.reading(rows), "Reading: round_robin itself had the best p95, 200.0 ms. "
                                                "The next best, adaptive, was 25.0% higher (250.0 ms).")
        rows = [row("static", 400.0), row("adaptive", 150.0)]
        self.assertIn("no round_robin p95 here", compare.reading(rows))
        rows = [row("round_robin", 200.0, errors=0), row("static", 90.0, errors=40)]
        self.assertTrue(compare.reading(rows).endswith(
            "But static had 40 errors (the fewest was 0, round_robin), and p95 counts ok answers only."))
        self.assertIn("no policy got an ok answer", compare.reading([row("static", None, errors=100)]))

    def test_table_has_the_columns_and_bolds_the_best(self):
        rows = [row("static", 400.0, errors=5, fail_at=10.0), row("round_robin", 200.0, errors=1, fail_at=10.0, recover=2.5),
                row("adaptive", 150.0, errors=1, share={"A": 1.0}), row("broken", None, errors=100, share={})]
        lines = compare.table_lines(rows)
        self.assertEqual(lines[0], "| policy | ok | errors | p50 ms | p95 ms | p99 ms | mean ms | share A / B | retries "
                                   "| failovers | switches | recover s |")
        self.assertEqual(lines[1].count("|"), lines[0].count("|"))
        cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]
        self.assertEqual(cells[0], ["static", "95", "5", "200.0", "400.0", "600.0", "250.0", "60% / 40%", "2", "1", "3",
                                    "not in run"])
        self.assertEqual(cells[1][:5] + cells[1][-1:], ["round_robin", "99", "**1**", "100.0", "200.0", "2.5"])
        self.assertEqual(cells[2][:5] + cells[2][7:8] + cells[2][-1:],
                         ["adaptive", "99", "**1**", "75.0", "**150.0**", "100% / 0%", "-"])
        self.assertEqual(cells[3][:8], ["broken", "0", "100", "-", "-", "-", "-", "0% / 0%"])

    def test_results_markdown_has_settings_table_reading_and_notes(self):
        runs = [row("static", 400.0), row("round_robin", 200.0), row("adaptive", 150.0, warnings=["the event x did not work"])]
        text = compare.results_markdown(SETTINGS, runs, "2026-11-14 09:30")
        for part in ("# Routing policies compared", "## Settings", "seed 1", "4 per second", "25 s per run",
                     "policies, in run order: static, round_robin, adaptive", "nearest-rank",
                     "## contention", "contention_on at 8.0 s; contention_off at 16.0 s",
                     "| adaptive | 100 | **0** | 75.0 | **150.0** |", "25.0% lower than round_robin",
                     "- Note, adaptive: the event x did not work", "(contention.svg)"):
            self.assertIn(part, text)

    def test_axis_choice(self):
        self.assertEqual(compare.y_axis([120.0, 480.0]), (0.0, 500.0, [0.0, 100.0, 200.0, 300.0, 400.0, 500.0], False))
        low, high, ticks, log = compare.y_axis([130.0, 40000.0])
        self.assertEqual((low, high, log), (100.0, 50000.0, True))                 # a log axis for a wide range
        self.assertEqual(ticks, [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0])
        self.assertEqual(compare.y_axis([3.0, 900000.0])[2], [2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0])
        self.assertEqual(compare.y_axis([3.0, 3.0])[:2], (0.0, 3.0))
        self.assertEqual([compare.nice_step(x) for x in (0.375, 7.5, 75.0, 100.0)], [0.5, 10.0, 100.0, 100.0])

    def test_chart_is_well_formed_svg_with_lines_legend_events_and_units(self):
        rows = [row("static", 9000.0), row("round_robin", 200.0), row("adaptive", 150.0), row("mine", 180.0)]
        text = compare.svg_chart("contention", rows, SETTINGS)
        root = ET.fromstring(text)
        ns = "{http://www.w3.org/2000/svg}"
        self.assertEqual((root.tag, root.get("width"), root.get("height")), (ns + "svg", "900", "360"))
        words = [node.text or "" for node in root.iter(ns + "text")]
        for name in ("static", "round_robin", "adaptive", "mine", "B slows down", "B back to normal",
                     "time since the run began (s)", "p95 latency (s, log scale)"):
            self.assertIn(name, words)
        paths = [node for node in root.iter(ns + "path")]
        self.assertEqual(len(paths), 4)                                # one line per policy
        self.assertEqual(len(set(node.get("stroke") for node in paths)), 4)
        self.assertEqual(paths[0].get("d").count("M"), 2)              # the window with no ok answer breaks the line
        self.assertEqual(len([node for node in root.iter(ns + "title")]), 1 + 4 * 4)   # the chart's, and one per point
        small = compare.svg_chart("steady", [row("round_robin", 200.0), row("adaptive", 150.0)], SETTINGS)
        self.assertIn("p95 latency (ms)", small)
        empty = compare.svg_chart("failure", [row("static", None, errors=100)], SETTINGS)
        self.assertIn("no ok answers", empty)
        ET.fromstring(small), ET.fromstring(empty)


class CompareTest(unittest.TestCase):
    """compare.py end to end: two policies, one scenario, 3 seconds each, against a fake target."""

    def test_compare_writes_its_files(self):
        fake = Fake()
        tmp = tempfile.mkdtemp(prefix="compare-test-")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(fake.stop)
        self.addCleanup(setattr, compare, "PAUSE_S", compare.PAUSE_S)
        compare.PAUSE_S = 0.3
        out = os.path.join(tmp, "results")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = compare.main(["--target", fake.url, "--admin-a", fake.url, "--admin-b", fake.url,
                                 "--policies", "round_robin,adaptive", "--scenarios", "contention",
                                 "--duration", "3", "--rate", "8", "--seed", "1", "--kind", "forecast", "--out", out])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(os.listdir(out)),
                         ["RESULTS.md", "contention.svg", "contention_adaptive.jsonl",
                          "contention_adaptive.jsonl.summary.json", "contention_round_robin.jsonl",
                          "contention_round_robin.jsonl.summary.json", "summary.json"])

        def read_lines(name):
            with open(os.path.join(out, name)) as f:
                return [json.loads(line) for line in f]

        first, second = read_lines("contention_round_robin.jsonl"), read_lines("contention_adaptive.jsonl")
        self.assertGreater(len(first), 10)
        self.assertEqual([(r["i"], r["t_sched"], r["units"]) for r in first],
                         [(r["i"], r["t_sched"], r["units"]) for r in second])     # the same load for both
        with open(os.path.join(out, "summary.json")) as f:
            summary = json.load(f)
        self.assertEqual((summary["settings"]["policies"], summary["settings"]["scenarios"], summary["settings"]["seed"]),
                         (["round_robin", "adaptive"], ["contention"], 1))
        self.assertEqual([(run["scenario"], run["policy"]) for run in summary["runs"]],
                         [("contention", "round_robin"), ("contention", "adaptive")])
        for run in summary["runs"]:
            self.assertEqual(run["summary"]["ok"], len(first))
            self.assertEqual(run["summary"]["settings"]["policy"], run["policy"])
            self.assertEqual([e["name"] for e in run["summary"]["events"]], ["contention_on", "contention_off"])
            self.assertEqual(len(run["windows"]), 1)
            self.assertEqual(run["summary"]["router"]["delta"]["routed"], len(first))
        with open(os.path.join(out, "RESULTS.md")) as f:
            text = f.read()
        for part in ("## Settings", "## contention", "| policy | ok | errors | p50 ms | p95 ms | p99 ms | mean ms | share A / B |",
                     "| round_robin | %d | **0** |" % len(first), "| adaptive | %d | **0** |" % len(first),
                     "Reading: ", "round_robin", "![contention: p95 latency over time](contention.svg)"):
            self.assertIn(part, text)
        self.assertIn(text, printed.getvalue())
        root = ET.parse(os.path.join(out, "contention.svg")).getroot()
        ns = "{http://www.w3.org/2000/svg}"
        words = [node.text or "" for node in root.iter(ns + "text")]
        for name in ("round_robin", "adaptive", "B slows down", "B back to normal"):
            self.assertIn(name, words)
        self.assertEqual(len([node for node in root.iter(ns + "path")]), 2)
        # The router was asked about each policy first, told each in turn, then given back the one it had.
        self.assertEqual([call[4]["policy"] for call in fake.seen("/router/policy")],
                         ["round_robin", "adaptive", "round_robin", "adaptive", "static"])
        self.assertEqual(fake.policy, "static")
        self.assertEqual(len(fake.seen("/admin/contention")), 4)
        self.assertEqual(len(fake.seen("/v1/forecast")), 2 * len(first))

    def test_bad_lists_are_refused(self):
        fake = Fake()
        self.addCleanup(fake.stop)
        tmp = tempfile.mkdtemp(prefix="compare-test-")
        self.addCleanup(shutil.rmtree, tmp, True)
        picky = Fake(accept_policy=False)
        self.addCleanup(picky.stop)
        errors, printed = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(printed):
            self.assertEqual(compare.main(["--target", fake.url, "--scenarios", "steady,nope", "--out", tmp]), 2)
            self.assertEqual(compare.main(["--target", fake.url, "--policies", "static,static", "--out", tmp]), 2)
            self.assertEqual(compare.main(["--target", closed_port_url(), "--scenarios", "steady", "--out", tmp]), 2)
            self.assertEqual(compare.main(["--target", picky.url, "--scenarios", "steady,burst",
                                           "--policies", "static,fastest", "--out", tmp]), 2)
            self.assertEqual(compare.main(["--target", fake.url, "--scenarios", "steady,failure", "--out", tmp]), 2)
        self.assertIn("the failure scenario posts to node B's admin hooks: give --admin-b", errors.getvalue())
        self.assertIn("unknown scenario 'nope'", errors.getvalue())
        self.assertIn("lists a name twice", errors.getvalue())
        self.assertIn("is not answering", errors.getvalue())
        self.assertIn("refused the policy 'static': HTTP 400. Nothing was run.", errors.getvalue())
        self.assertEqual([name for name in os.listdir(tmp) if not name.startswith(".")], [])
        self.assertEqual(fake.seen("/v1/forecast") + picky.seen("/v1/forecast"), [])


if __name__ == "__main__":
    unittest.main()
