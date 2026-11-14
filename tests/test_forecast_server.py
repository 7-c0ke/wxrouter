"""Tests for serve/forecast_server.py: mock mode in full, and a few requests through the real model.

    .venv/bin/python -m unittest tests/test_forecast_server.py -v

Every server here is a child process on 127.0.0.1, on a port the OS picks, and is stopped when its test ends.
The real-model tests are skipped unless .local/ holds the test bundle and weights and PyTorch is installed.

Written at the hackathon (Nov 13-14, 2026).
"""
import base64
import datetime
import http.client
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, "serve", "forecast_server.py")
BUNDLE = os.path.join(REPO, ".local", "bundle")
WEIGHTS = os.environ.get("WX_TEST_WEIGHTS", os.path.join(REPO, ".local", "weights", "model.pt"))
OHARE = {"lat": 41.9602, "lon": -87.9316}
SERIES = ["t2m_c", "t2m_c_mean", "t2m_c_min", "t2m_c_max", "wind10_ms_mean", "msl_hpa_mean"]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))       # straight to 127.0.0.1, no proxy

sys.path.insert(0, os.path.join(REPO, "serve"))
import forecast_server as fs  # noqa: E402   (standard library only at its top level)


def why_no_real_mode():
    """Why the real-model tests cannot run on this machine; None if they can."""
    for path in (WEIGHTS, os.path.join(BUNDLE, "bundle.json"), os.path.join(BUNDLE, "stats.json")):
        if not os.path.exists(path):
            return "missing %s" % path
    for module in ("numpy", "torch", "physicsnemo"):
        if importlib.util.find_spec(module) is None:
            return "%s is not installed" % module
    return None


NO_REAL_MODE = why_no_real_mode()


class Running(object):
    """A forecast server running as a child process on 127.0.0.1, on a port the OS picked."""

    def __init__(self, *flags, **options):
        self.errors = tempfile.NamedTemporaryFile(prefix="forecast_server_test_", suffix=".log")    # its stderr
        command = [sys.executable] + options.get("python_flags", []) + [SERVER, "--host", "127.0.0.1", "--port", "0"]
        self.proc = subprocess.Popen(command + list(flags), cwd=REPO, stdout=subprocess.PIPE, stderr=self.errors,
                                     universal_newlines=True)
        watchdog = threading.Timer(20, self.proc.kill)          # never wait forever for the first line
        watchdog.start()
        self.first_line = self.proc.stdout.readline().rstrip("\n")
        watchdog.cancel()
        self.drain = threading.Thread(target=self.proc.stdout.read, daemon=True)     # so the pipe cannot fill up
        self.drain.start()
        found = re.fullmatch(r"listening on (http://127\.0\.0\.1:(\d+))", self.first_line)
        if not found:
            text = self.stderr()
            self.stop()
            raise AssertionError("the server did not start. First line: %r. stderr: %s" % (self.first_line, text))
        self.url, self.port = found.group(1), int(found.group(2))

    def stderr(self):
        with open(self.errors.name) as f:
            return f.read()

    def stop(self, sig=signal.SIGTERM):
        """Stop the child if it still runs. Returns its exit code. Safe to call twice."""
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.drain.join(5)
        self.proc.stdout.close()
        self.errors.close()
        return self.proc.returncode


def call(url, path, body=None, method=None, timeout=10, raw=None):
    """One HTTP request. Returns (status, headers, body); a JSON body comes back parsed."""
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
    request = urllib.request.Request(url + path, data=data, method=method)
    try:
        with OPENER.open(request, timeout=timeout) as response:
            status, headers, text = response.status, response.headers, response.read()
    except urllib.error.HTTPError as e:
        status, headers, text = e.code, e.headers, e.read()
        e.close()
    text = text.decode("utf-8")
    if headers.get("Content-Type", "").startswith("application/json"):
        return status, headers, json.loads(text)
    return status, headers, text


def forecast_body(**changes):
    body = {"init": "2019-01-27T12", "lead_hours": 24, "members": 1, "point": dict(OHARE)}
    body.update(changes)
    return body


class MockApiTest(unittest.TestCase):
    """The mock's API, on one shared server. Each test starts from a reset server."""

    @classmethod
    def setUpClass(cls):
        cls.server = Running("--mock", "--admin", "--node", "A", "--latest", "2026-11-14T00")
        cls.addClassCleanup(cls.server.stop)
        cls.url = cls.server.url

    def setUp(self):
        self.assertEqual(call(self.url, "/admin/reset", method="POST")[0], 200)

    def forecast(self, **changes):
        status, _, body = call(self.url, "/v1/forecast", forecast_body(**changes))
        self.assertEqual(status, 200, body)
        return body

    # ---- the team contract (API.md, v1.1): lat, lon, hours_ahead, step_hours, ensemble_members

    def v1(self, **body):
        status, _, answer = call(self.url, "/v1/forecast", body)
        self.assertEqual(status, 200, answer)
        return answer

    def test_v1_shape_is_the_contract(self):
        body = self.v1(lat=OHARE["lat"], lon=OHARE["lon"], hours_ahead=24, step_hours=6, init="2019-01-27T12",
                       request_id="abc")
        for name in ("request_id", "box", "model_version", "contract_version", "issued_at", "lat", "lon",
                     "hours_ahead", "step_hours", "low_confidence", "forecast", "members", "timing"):
            self.assertIn(name, body)
        self.assertEqual((body["request_id"], body["box"], body["contract_version"]), ("abc", "A", "1.1"))
        self.assertEqual((body["hours_ahead"], body["step_hours"], body["low_confidence"], body["members"]),
                         (24, 6, False, None))
        self.assertEqual(sorted(body["timing"]), ["compute_ms", "queue_ms", "total_ms"])
        self.assertEqual([p["lead_hours"] for p in body["forecast"]], [6, 12, 18, 24])
        self.assertEqual(body["forecast"][0]["valid_time"], "2019-01-27T18:00:00Z")
        for name in ("temp_c", "wind_ms", "precip_mm", "pressure_hpa"):
            self.assertIn(name, body["forecast"][0])
        self.assertIsNone(body["forecast"][0]["precip_mm"])          # the model has no precipitation
        own = self.forecast()
        self.assertEqual([p["temp_c"] for p in body["forecast"]], own["point"]["t2m_c_mean"])

    def test_v1_default_step_is_hourly_and_interpolated(self):
        body = self.v1(lat=41.0, lon=-87.0, hours_ahead=8, init="2019-01-27T12")
        self.assertEqual([p["lead_hours"] for p in body["forecast"]], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(body["interpolated"])
        six = self.v1(lat=41.0, lon=-87.0, hours_ahead=12, step_hours=6, init="2019-01-27T12")["forecast"]
        temps = dict((p["lead_hours"], p["temp_c"]) for p in body["forecast"])
        self.assertEqual(temps[6], six[0]["temp_c"])                 # the model's own step is not changed
        self.assertAlmostEqual(temps[7], six[0]["temp_c"] + (six[1]["temp_c"] - six[0]["temp_c"]) / 6.0, places=1)
        self.assertEqual(body["units"], 2)                           # 8 hours ahead = two 6-hour steps

    def test_v1_members_and_low_confidence(self):
        body = self.v1(lat=41.0, lon=-87.0, hours_ahead=96, step_hours=24, ensemble_members=3, init="2019-01-27T12")
        self.assertTrue(body["low_confidence"])
        self.assertEqual([m["member"] for m in body["members"]], [0, 1, 2])
        self.assertEqual([p["lead_hours"] for p in body["members"][0]["points"]], [24, 48, 72, 96])
        mean = sum(m["points"][-1]["temp_c"] for m in body["members"]) / 3.0
        self.assertAlmostEqual(body["forecast"][-1]["temp_c"], mean, places=1)

    def test_v1_bad_input_is_422(self):
        for bad in ({"lat": 41.0, "lon": -87.0, "hours_ahead": 24, "step_hours": 25},
                    {"lat": 41.0, "lon": -87.0},
                    {"lat": 41.0, "lon": -87.0, "hours_ahead": 0},
                    {"lat": 141.0, "lon": -87.0, "hours_ahead": 24},
                    {"lat": 41.0, "lon": -87.0, "hours_ahead": 24, "ensemble_members": 99},
                    {"lat": 41.0, "lon": -87.0, "hours_ahead": 24, "surprise": 1}):
            status, _, body = call(self.url, "/v1/forecast", bad)
            self.assertEqual(status, 422, (bad, body))
            self.assertIn("error", body)

    def test_stats_and_health_have_the_contract_names(self):
        status, _, body = call(self.url, "/stats")
        self.assertEqual(status, 200, body)
        for name in ("box", "in_flight", "queue_depth", "max_concurrency", "gpu_util", "ewma_latency_ms",
                     "p50_latency_ms", "p95_latency_ms", "requests_total", "errors_total", "mode"):
            self.assertIn(name, body)
        self.assertEqual((body["box"], body["in_flight"], body["queue_depth"], body["mode"]), ("A", 0, 0, "normal"))
        self.assertTrue(0.0 <= body["gpu_util"] <= 1.0)
        self.assertEqual(call(self.url, "/admin/contention", {"factor": 2.0})[0], 200)
        self.assertEqual(call(self.url, "/stats")[2]["mode"], "training")
        health = call(self.url, "/health")[2]
        for name in ("status", "box", "model_version", "contract_version", "uptime_s"):
            self.assertIn(name, health)

    def test_admin_mode_is_the_team_contracts_switch(self):
        self.assertEqual(call(self.url, "/admin/mode", {"mode": "slow", "slow_factor": 4.0})[0], 200)
        self.assertEqual(call(self.url, "/stats")[2]["mode"], "training")
        self.assertEqual(call(self.url, "/admin/mode", {"mode": "error"})[0], 200)
        self.assertEqual(call(self.url, "/health")[0], 503)
        self.assertEqual(call(self.url, "/admin/mode", {"mode": "normal"})[0], 200)
        self.assertEqual(call(self.url, "/health")[0], 200)
        self.assertEqual(call(self.url, "/stats")[2]["mode"], "normal")
        self.assertEqual(call(self.url, "/admin/mode", {"mode": "sideways"})[0], 400)

    def metrics(self):
        status, _, body = call(self.url, "/metrics")
        self.assertEqual(status, 200, body)
        return body

    # --- shapes

    def test_health(self):
        status, _, body = call(self.url, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(dict((k, body[k]) for k in ("status", "node", "model", "ready")),
                         {"status": "ok", "node": "A", "model": "mock", "ready": True})

    def test_info(self):
        status, _, body = call(self.url, "/v1/info")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(body), ["device", "latest", "mock", "model", "ms_per_unit", "node", "params", "ranges"])
        self.assertEqual(body["node"], "A")
        self.assertIs(body["mock"], True)
        self.assertEqual(body["latest"], "2026-11-14T00")
        self.assertEqual(body["ranges"], [{"first": "1979-01-01T00", "last": "2026-11-14T00"}])
        self.assertEqual(body["ms_per_unit"], 10.0)

    def test_forecast_shape(self):
        body = self.forecast(request_id="abc")
        self.assertEqual(sorted(body), ["init_time", "leads_h", "member_offset", "members", "model", "node", "point",
                                        "request_id", "timing_ms", "units", "valid_times"])
        self.assertEqual(body["request_id"], "abc")
        self.assertEqual((body["node"], body["model"], body["init_time"]), ("A", "mock", "2019-01-27T12"))
        self.assertEqual((body["members"], body["member_offset"], body["units"]), (1, 0, 4))
        self.assertEqual(body["leads_h"], [6, 12, 18, 24])
        self.assertEqual(body["valid_times"], ["2019-01-27T18", "2019-01-28T00", "2019-01-28T06", "2019-01-28T12"])
        point = body["point"]
        self.assertEqual(sorted(point), sorted(["lat", "lon"] + SERIES))       # no truth unless it is asked for
        self.assertEqual((point["lat"], point["lon"]), (OHARE["lat"], OHARE["lon"]))
        for name in SERIES:
            self.assertEqual(len(point[name]), 4, name)
            for value in point[name]:
                self.assertIsInstance(value, float, name)
        self.assertEqual(point["t2m_c"], point["t2m_c_mean"])                   # one member: all the same
        self.assertEqual(point["t2m_c"], point["t2m_c_min"])
        self.assertEqual(point["t2m_c"], point["t2m_c_max"])
        for value in point["t2m_c"]:
            self.assertTrue(-40 < value < 25, value)                            # Chicago in January, more or less
        for value in point["wind10_ms_mean"]:
            self.assertTrue(0 <= value < 30, value)
        for value in point["msl_hpa_mean"]:
            self.assertTrue(950 < value < 1070, value)
        self.assertEqual(sorted(body["timing_ms"]), ["gpu", "queue", "total"])
        self.assertGreaterEqual(body["timing_ms"]["total"], body["timing_ms"]["gpu"])

    def test_defaults_and_request_id(self):
        status, _, body = call(self.url, "/v1/forecast", {"lead_hours": 6, "point": {"lat": 0, "lon": 0}})
        self.assertEqual(status, 200, body)
        self.assertEqual((body["init_time"], body["members"], body["member_offset"]), ("2026-11-14T00", 1, 0))
        self.assertTrue(body["request_id"])                                    # the server made one up
        longest = self.forecast(lead_hours=120, members=1)
        self.assertEqual(longest["leads_h"], list(range(6, 126, 6)))
        self.assertEqual(longest["valid_times"][-1], "2019-02-01T12")
        self.assertEqual(longest["units"], 20)

    def test_ensemble_fields(self):
        point = self.forecast(members=6)["point"]
        spreads = []
        for k in range(4):
            self.assertLessEqual(point["t2m_c_min"][k], point["t2m_c_mean"][k])
            self.assertLessEqual(point["t2m_c_mean"][k], point["t2m_c_max"][k])
            self.assertLessEqual(point["t2m_c_min"][k], point["t2m_c"][k])
            self.assertLessEqual(point["t2m_c"][k], point["t2m_c_max"][k])
            spreads.append(point["t2m_c_max"][k] - point["t2m_c_min"][k])
        self.assertGreater(spreads[0], 0.0)                                    # members differ
        self.assertGreater(spreads[-1], spreads[0])                            # and drift apart with the lead

    def test_member_0_is_only_in_its_own_shard(self):
        whole = self.forecast(members=3)["point"]
        self.assertEqual(self.forecast(members=1)["point"]["t2m_c"], whole["t2m_c"])    # member 0 has no offset
        rest = self.forecast(members=2, member_offset=1)
        self.assertEqual(rest["member_offset"], 1)
        self.assertIsNone(rest["point"]["t2m_c"])
        self.assertEqual(len(rest["point"]["t2m_c_mean"]), 4)

    def test_same_request_same_answer(self):
        first = self.forecast(members=3)["point"]
        self.assertEqual(first, self.forecast(members=3)["point"])
        self.assertNotEqual(first, self.forecast(members=3, init="2019-01-28T12")["point"])
        east = self.forecast(point={"lat": 41.9602, "lon": 272.0684})["point"]       # the same place, lon + 360
        west = self.forecast()["point"]
        for name in SERIES:
            for a, b in zip(east[name], west[name]):
                self.assertAlmostEqual(a, b, delta=0.011)

    def test_with_truth(self):
        point = self.forecast(with_truth=True, lead_hours=48)["point"]
        self.assertEqual(len(point["truth_t2m_c"]), 8)
        for forecast, truth in zip(point["t2m_c"], point["truth_t2m_c"]):
            self.assertLess(abs(forecast - truth), 3.0)                         # made up, but near the forecast
        future = self.forecast(init="latest", with_truth=True, lead_hours=12)["point"]
        self.assertEqual(future["truth_t2m_c"], [None, None])                   # nobody knows tomorrow's truth
        edge = self.forecast(init="2026-11-13T18", with_truth=True, lead_hours=12)["point"]
        self.assertIsInstance(edge["truth_t2m_c"][0], float)
        self.assertIsNone(edge["truth_t2m_c"][1])

    def test_time_spellings(self):
        for init in ("2019-01-27T12:00", "2019-01-27T12:00:00Z", "2019-01-27T12Z"):
            self.assertEqual(self.forecast(init=init)["init_time"], "2019-01-27T12")

    def test_init_as_npy_is_not_decoded(self):
        body = self.forecast(init={"npy_b64": "bm90IHJlYWxseSBhIHN0YXRl", "time": "2030-05-01T06"}, lead_hours=6)
        self.assertEqual(body["init_time"], "2030-05-01T06")
        self.assertEqual(body["valid_times"], ["2030-05-01T12"])

    def test_state_push_moves_latest(self):
        server = Running("--mock", "--latest", "2026-11-14T00")
        self.addCleanup(server.stop)
        status, _, body = call(server.url, "/v1/state", {"time": "2026-11-13T00", "npy_b64": "eA=="})
        self.assertEqual((status, body), (200, {"latest": "2026-11-14T00"}))    # older: latest stays
        status, _, body = call(server.url, "/v1/state", {"time": "2026-11-14T06", "npy_b64": "eA=="})
        self.assertEqual((status, body), (200, {"latest": "2026-11-14T06"}))    # newer: it is the latest now
        self.assertEqual(call(server.url, "/v1/info")[2]["latest"], "2026-11-14T06")
        status, _, body = call(server.url, "/v1/forecast", forecast_body(init="latest"))
        self.assertEqual(body["init_time"], "2026-11-14T06")

    # --- cost and the queue

    def test_cost_grows_with_units(self):
        small = self.forecast(lead_hours=6, members=1)
        t0 = time.time()
        large = self.forecast(lead_hours=60, members=4)
        seconds = time.time() - t0
        self.assertEqual((small["units"], large["units"]), (1, 40))
        self.assertGreaterEqual(small["timing_ms"]["gpu"], 9)                  # 1 unit x 10 ms
        self.assertGreaterEqual(large["timing_ms"]["gpu"], 380)                # 40 units x 10 ms
        self.assertLess(small["timing_ms"]["gpu"], large["timing_ms"]["gpu"] / 2)
        self.assertGreaterEqual(seconds, 0.38)

    def test_requests_queue_as_on_one_gpu(self):
        answers = []

        def one():
            answers.append(self.forecast(lead_hours=60, members=4))             # 40 units = 0.4 s each

        threads = [threading.Thread(target=one) for _ in range(3)]
        t0 = time.time()
        for thread in threads:
            thread.start()
        seen = None
        while time.time() < t0 + 3:                                             # catch it with one running, two waiting
            m = self.metrics()
            if (m["inflight"], m["queued"]) == (1, 2):
                seen = m
                break
            time.sleep(0.01)
        for thread in threads:
            thread.join(20)
        seconds = time.time() - t0
        self.assertIsNotNone(seen, "never saw 1 in flight and 2 queued")
        self.assertEqual(seen["queued_units"], 120)                             # waiting or running
        self.assertEqual(seen["gpu"]["util_pct"], 96)
        self.assertEqual(len(answers), 3)
        self.assertGreaterEqual(seconds, 3 * 0.38)                              # one after the other, not together
        waits = sorted(a["timing_ms"]["queue"] for a in answers)
        self.assertLess(waits[0], 300)                                          # the first one did not wait
        self.assertGreaterEqual(waits[1], 300)
        self.assertGreaterEqual(waits[2], 700)
        after = self.metrics()
        self.assertEqual((after["inflight"], after["queued"], after["queued_units"]), (0, 0, 0))
        self.assertEqual((after["served_total"], after["units_total"], after["errors_total"]), (3, 120, 0))
        self.assertGreaterEqual(after["latency_ms"]["p95"], after["latency_ms"]["p50"])
        self.assertGreaterEqual(after["latency_ms"]["p50"], 380)
        self.assertGreater(after["latency_ms"]["ewma"], 0)

    # --- /metrics

    def test_metrics_json(self):
        m = self.metrics()
        self.assertEqual(list(m), ["node", "ts", "inflight", "queued", "queued_units", "served_total", "errors_total",
                                   "units_total", "ms_per_unit", "latency_ms", "gpu", "host", "busy_other", "slowdown"])
        self.assertEqual(sorted(m["latency_ms"]), ["ewma", "p50", "p95"])
        self.assertEqual(sorted(m["gpu"]), ["power_w", "proc_mem_mb", "temp_c", "util_pct"])
        self.assertEqual(sorted(m["host"]), ["load1", "mem_available_gb"])
        self.assertEqual(m["node"], "A")
        self.assertLess(abs(m["ts"] - time.time()), 5)
        self.assertEqual(m["latency_ms"], {"p50": None, "p95": None, "ewma": None})     # nothing served yet
        self.assertEqual((m["ms_per_unit"], m["busy_other"], m["slowdown"]), (10.0, False, 1.0))
        self.forecast(members=2)
        m = self.metrics()
        self.assertEqual((m["served_total"], m["units_total"]), (1, 8))
        self.assertGreaterEqual(m["latency_ms"]["p50"], 76)
        self.assertEqual(m["latency_ms"]["p50"], m["latency_ms"]["p95"])
        self.assertEqual(call(self.url, "/metrics?format=json")[2]["served_total"], 1)

    def test_metrics_prometheus(self):
        self.forecast(members=2)
        status, headers, text = call(self.url, "/metrics?format=prom")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/plain"))
        values = {}
        for line in text.splitlines():
            if not line.startswith("#"):
                name, value = line.split(" ")
                self.assertTrue(name.startswith("wx_"), line)
                values[name] = float(value)
        m = self.metrics()
        for name in ("inflight", "queued", "queued_units", "served_total", "errors_total", "units_total",
                     "ms_per_unit", "slowdown"):
            self.assertEqual(values["wx_" + name], m[name], name)
        self.assertEqual(values["wx_latency_ms_p50"], m["latency_ms"]["p50"])
        self.assertEqual(values["wx_gpu_proc_mem_mb"], m["gpu"]["proc_mem_mb"])
        self.assertEqual(values["wx_host_mem_available_gb"], m["host"]["mem_available_gb"])
        self.assertEqual(values["wx_busy_other"], 0)
        self.assertIn("# TYPE wx_served_total counter", text)
        self.assertIn("# TYPE wx_queued gauge", text)
        self.assertEqual(call(self.url, "/metrics?format=xml")[0], 400)

    # --- bad requests

    def test_bad_forecasts_get_400(self):
        nan_point = b'{"lead_hours": 24, "point": {"lat": NaN, "lon": 0}}'
        bad = [
            ("not JSON", b"{lead_hours: 24"), ("a list", b"[1, 2]"), ("no body", b""), ("bytes", b"\xff\xfe"),
            ("NaN latitude", nan_point),
            ("no lead_hours", {"point": dict(OHARE)}), ("lead_hours null", forecast_body(lead_hours=None)),
            ("lead 0", forecast_body(lead_hours=0)), ("lead 7", forecast_body(lead_hours=7)),
            ("lead 126", forecast_body(lead_hours=126)), ("lead text", forecast_body(lead_hours="24")),
            ("lead 12.5", forecast_body(lead_hours=12.5)), ("lead true", forecast_body(lead_hours=True)),
            ("members 0", forecast_body(members=0)), ("members 41", forecast_body(members=41)),
            ("members 1.5", forecast_body(members=1.5)), ("members text", forecast_body(members="2")),
            ("offset -1", forecast_body(member_offset=-1)), ("offset 0.5", forecast_body(member_offset=0.5)),
            ("no point", {"lead_hours": 24}), ("point text", forecast_body(point="ohare")),
            ("no lon", forecast_body(point={"lat": 41.0})), ("lat 91", forecast_body(point={"lat": 91, "lon": 0})),
            ("lat -90.5", forecast_body(point={"lat": -90.5, "lon": 0})),
            ("lat text", forecast_body(point={"lat": "41", "lon": 0})),
            ("lon -181", forecast_body(point={"lat": 0, "lon": -181})),
            ("lon 360.5", forecast_body(point={"lat": 0, "lon": 360.5})),
            ("point extra", forecast_body(point={"lat": 0, "lon": 0, "name": "x"})),
            ("with_truth text", forecast_body(with_truth="yes")), ("with_truth 1", forecast_body(with_truth=1)),
            ("request_id number", forecast_body(request_id=5)),
            ("unknown field", forecast_body(lead_hour=24)),
            ("init word", forecast_body(init="yesterday")), ("init month 13", forecast_body(init="2019-13-01T00")),
            ("init Feb 30", forecast_body(init="2019-02-30T00")), ("init number", forecast_body(init=5)),
            ("init minutes", forecast_body(init="2019-01-27T12:30")),
            ("init npy without time", forecast_body(init={"npy_b64": "eA=="})),
            ("init npy empty", forecast_body(init={"npy_b64": "", "time": "2019-01-27T12"})),
            ("init npy at 13h", forecast_body(init={"npy_b64": "eA==", "time": "2019-01-27T13"})),
            ("init npy extra", forecast_body(init={"npy_b64": "eA==", "time": "2019-01-27T12", "x": 1})),
        ]
        for label, body in bad:
            with self.subTest(label):
                if isinstance(body, bytes):
                    status, _, answer = call(self.url, "/v1/forecast", raw=body, method="POST")
                else:
                    status, _, answer = call(self.url, "/v1/forecast", body)
                self.assertEqual(status, 400, answer)
                self.assertEqual(list(answer), ["error"])
                self.assertTrue(answer["error"])
        self.assertIn("lead_hours", call(self.url, "/v1/forecast", forecast_body(lead_hours=7))[2]["error"])
        self.assertIn("lead_hour", call(self.url, "/v1/forecast", forecast_body(lead_hour=24))[2]["error"])
        m = self.metrics()
        self.assertEqual((m["errors_total"], m["served_total"]), (0, 0))        # a 400 is the client's error
        self.forecast()                                                         # and the server is still fine

    def test_bad_states_get_400(self):
        bad = [{}, {"time": "2026-11-14T00"}, {"npy_b64": "eA=="}, {"time": "soon", "npy_b64": "eA=="},
               {"time": "2026-11-14T03", "npy_b64": "eA=="}, {"time": "2026-11-14T00", "npy_b64": 5},
               {"time": "2026-11-14T00", "npy_b64": "eA==", "x": 1}, [1],
               {"time": "2200-12-31T18", "npy_b64": "eA=="}]            # a state from the future
        for body in bad:
            with self.subTest(body=body):
                status, _, answer = call(self.url, "/v1/state", body)
                self.assertEqual(status, 400, answer)
                self.assertTrue(answer["error"])

    def test_a_state_to_keep_cannot_be_dated_in_the_future(self):
        # A late date would become "latest" on every box and hide each real analysis after it.
        now = datetime.datetime(2026, 11, 14, 22, 40)
        ok = fs.check_state_request({"time": "2026-11-14T18", "npy_b64": "eA=="}, now=now)
        self.assertEqual(ok, (datetime.datetime(2026, 11, 14, 18), "eA=="))
        for near in ("2026-11-15T00", "2026-11-15T18"):                        # the next cycles, played ahead in a test
            self.assertEqual(fs.check_state_request({"time": near, "npy_b64": "eA=="}, now=now)[1], "eA==")
        for late in ("2026-11-16T00", "2027-01-01T00", "2200-12-31T18"):
            with self.subTest(time=late):
                with self.assertRaises(fs.ApiError) as caught:
                    fs.check_state_request({"time": late, "npy_b64": "eA=="}, now=now)
                self.assertEqual(caught.exception.status, 400)
                self.assertIn("in the future", str(caught.exception))

    def test_bad_content_length(self):
        for length in ("abc", "-5", "999999999999"):
            with self.subTest(length=length):
                sock = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)
                self.addCleanup(sock.close)
                sock.sendall(("POST /v1/forecast HTTP/1.1\r\nHost: x\r\nContent-Length: %s\r\n\r\n" % length).encode())
                answer = b""
                while True:                                                     # the server answers, then hangs up
                    part = sock.recv(65536)
                    if not part:
                        break
                    answer += part
                head, _, body = answer.partition(b"\r\n\r\n")
                self.assertTrue(head.startswith(b"HTTP/1.1 400 "), head)
                self.assertIn(b"Access-Control-Allow-Origin: *", head)
                self.assertTrue(json.loads(body.decode())["error"])
        for junk in (b"what is this\r\n\r\n", b"\x00\x01\x02\r\n\r\n", b"GET /health HTTP/9.9\r\n\r\n"):
            sock = socket.create_connection(("127.0.0.1", self.server.port), timeout=5)    # not HTTP at all
            self.addCleanup(sock.close)
            sock.sendall(junk)
            while sock.recv(65536):                                             # some answer, then it hangs up
                pass
        self.assertEqual(call(self.url, "/health")[0], 200)                     # and the server shrugs it off

    def test_times_and_paths_not_held_get_404(self):
        for init in ("2030-01-01T00", "2026-11-14T06", "1978-12-31T18", "2019-01-27T13"):
            with self.subTest(init=init):
                status, _, answer = call(self.url, "/v1/forecast", forecast_body(init=init))
                self.assertEqual(status, 404, answer)
                self.assertIn("not held", answer["error"])
        self.assertEqual(call(self.url, "/nothing/here")[0], 404)
        self.assertEqual(call(self.url, "/v1/nothing", {})[0], 404)

    def test_wrong_method(self):
        status, headers, answer = call(self.url, "/v1/forecast")
        self.assertEqual((status, headers["Allow"]), (405, "POST"))
        status, headers, answer = call(self.url, "/health", {})
        self.assertEqual((status, headers["Allow"]), (405, "GET"))
        status, headers, answer = call(self.url, "/health", method="DELETE")
        self.assertEqual(status, 501)
        self.assertTrue(answer["error"])

    # --- HTTP manners

    def test_cors_and_content_length_everywhere(self):
        requests = [("/health", None, None), ("/metrics", None, None), ("/metrics?format=prom", None, None),
                    ("/v1/info", None, None), ("/v1/forecast", forecast_body(), None),
                    ("/v1/forecast", forecast_body(lead_hours=7), None),             # 400
                    ("/v1/forecast", forecast_body(init="2030-01-01T00"), None),     # 404
                    ("/nothing", None, None), ("/v1/forecast", None, None),          # 404, 405
                    ("/health", None, "PUT"), ("/admin/fail", {"mode": "sideways"}, None)]
        for path, body, method in requests:
            with self.subTest(path=path, method=method):
                _, headers, _ = call(self.url, path, body, method)
                self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
                self.assertTrue(headers["Content-Length"].isdigit())
        for path in ("/v1/forecast", "/admin/fail", "/metrics"):
            status, headers, text = call(self.url, path, method="OPTIONS")           # the browser's preflight
            self.assertEqual((status, text, headers["Content-Length"]), (200, "", "0"))
            self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
            self.assertIn("POST", headers["Access-Control-Allow-Methods"])
            self.assertTrue(headers["Access-Control-Allow-Headers"])

    def test_keep_alive(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=10)
        self.addCleanup(conn.close)
        steps = [("GET", "/health", None, 200), ("POST", "/v1/forecast", forecast_body(lead_hours=7), 400),
                 ("POST", "/v1/forecast", forecast_body(), 200), ("GET", "/nothing", None, 404),
                 ("GET", "/metrics", None, 200)]
        sockets = set()
        for method, path, body, expected in steps:
            conn.request(method, path, None if body is None else json.dumps(body))
            response = conn.getresponse()
            json.loads(response.read().decode())
            self.assertEqual(response.status, expected, path)
            self.assertEqual(response.version, 11)
            self.assertIsNone(response.getheader("Connection"))                 # the server does not hang up
            sockets.add(id(conn.sock))
        self.assertEqual(len(sockets), 1)                                       # one connection carried them all
        self.assertIsNotNone(conn.sock)

    # --- the test hooks

    def test_admin_contention(self):
        status, _, body = call(self.url, "/admin/contention", {"factor": 2.0})
        self.assertEqual((status, body), (200, {"mode": "ok", "factor": 2.0}))
        m = self.metrics()
        self.assertEqual((m["busy_other"], m["slowdown"], m["ms_per_unit"]), (True, 2.0, 20.0))
        self.assertEqual(call(self.url, "/v1/info")[2]["ms_per_unit"], 20.0)
        slow = self.forecast(lead_hours=60, members=1)                          # 10 units at 20 ms
        self.assertGreaterEqual(slow["timing_ms"]["gpu"], 190)
        self.assertEqual(call(self.url, "/admin/contention", {"factor": 1.0})[2], {"mode": "ok", "factor": 1.0})
        m = self.metrics()
        self.assertEqual((m["busy_other"], m["slowdown"], m["ms_per_unit"]), (False, 1.0, 10.0))
        fast = self.forecast(lead_hours=60, members=1)
        self.assertLess(fast["timing_ms"]["gpu"], slow["timing_ms"]["gpu"])
        for body in ({"factor": 0.5}, {"factor": "big"}, {}, {"factor": True}, {"factor": 2, "x": 1}):
            self.assertEqual(call(self.url, "/admin/contention", body)[0], 400, body)

    def test_admin_fail_down(self):
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "down"})[2], {"mode": "down", "factor": 1.0})
        status, headers, body = call(self.url, "/health")
        self.assertEqual(status, 503)
        self.assertEqual(dict((k, body[k]) for k in ("status", "node", "model", "ready")),
                         {"status": "down", "node": "A", "model": "mock", "ready": False})
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
        state = {"time": "2026-11-14T06", "npy_b64": "eA=="}
        for path, payload in (("/v1/forecast", forecast_body()), ("/v1/state", state), ("/metrics", None),
                              ("/v1/info", None)):
            status, _, body = call(self.url, path, payload)
            self.assertEqual(status, 503, path)
            self.assertTrue(body["error"])
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "ok"})[0], 200)
        self.assertEqual(call(self.url, "/health")[0], 200)
        self.assertEqual(self.metrics()["errors_total"], 0)                     # a 503 is not a 500
        self.forecast()

    def test_admin_fail_error(self):
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "error"})[0], 200)
        status, _, body = call(self.url, "/v1/forecast", forecast_body())
        self.assertEqual(status, 500)
        self.assertTrue(body["error"])
        self.assertEqual(call(self.url, "/v1/state", {"time": "2026-11-14T06", "npy_b64": "eA=="})[0], 500)
        self.assertEqual(call(self.url, "/health")[0], 200)                     # health stays green
        self.assertEqual(call(self.url, "/v1/info")[0], 200)
        m = self.metrics()
        self.assertEqual((m["errors_total"], m["served_total"]), (2, 0))
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "ok"})[0], 200)
        self.forecast()
        for body in ({"mode": "sideways"}, {}, {"mode": ["down"]}, {"mode": "down", "x": 1}):
            self.assertEqual(call(self.url, "/admin/fail", body)[0], 400, body)
        self.assertEqual(call(self.url, "/admin/nothing", {})[0], 404)

    def test_admin_fail_hang(self):
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "hang"})[0], 200)
        for path, payload in (("/v1/forecast", forecast_body()), ("/health", None)):
            with self.assertRaises((socket.timeout, urllib.error.URLError)):    # accepted, never answered
                call(self.url, path, payload, timeout=0.25)
        outcome = []

        def patient():
            try:
                outcome.append(call(self.url, "/v1/forecast", forecast_body(), timeout=10)[0])
            except Exception as e:
                outcome.append(e)

        thread = threading.Thread(target=patient)
        thread.start()
        time.sleep(0.3)
        self.assertEqual(outcome, [])                                           # still hanging
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "ok"})[0], 200)  # the hooks themselves still answer
        thread.join(5)
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], Exception)                            # let go without an answer
        self.forecast()
        self.assertEqual(self.metrics()["served_total"], 1)

    def test_admin_reset(self):
        self.forecast()
        call(self.url, "/admin/contention", {"factor": 1.5})
        call(self.url, "/admin/fail", {"mode": "error"})
        call(self.url, "/v1/forecast", forecast_body())
        self.assertEqual(call(self.url, "/admin/reset", method="POST")[2], {"mode": "ok", "factor": 1.0})
        m = self.metrics()
        self.assertEqual((m["served_total"], m["errors_total"], m["units_total"]), (0, 0, 0))
        self.assertEqual(m["latency_ms"], {"p50": None, "p95": None, "ewma": None})
        self.assertEqual((m["busy_other"], m["slowdown"]), (False, 1.0))
        self.forecast()


class MockProcessTest(unittest.TestCase):
    """Things that need their own server: starting, stopping, two nodes side by side."""

    def start(self, *flags, **options):
        server = Running(*flags, **options)
        self.addCleanup(server.stop)
        return server

    def fast_node(self, name):
        return self.start("--mock", "--node", name, "--ms-per-unit", "0")      # no emulated work: answers at once

    def test_first_line_and_clean_exit(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            server = self.start("--mock")
            self.assertRegex(server.first_line, r"^listening on http://127\.0\.0\.1:\d+$")
            self.assertEqual(call(server.url, "/health")[0], 200)
            self.assertEqual(server.stop(sig), 0)

    def test_two_nodes_give_the_same_numbers(self):
        a, b = self.fast_node("A"), self.fast_node("B")
        body = forecast_body(members=6, lead_hours=72, with_truth=True)
        _, _, from_a = call(a.url, "/v1/forecast", body)
        _, _, from_b = call(b.url, "/v1/forecast", body)
        self.assertEqual((from_a["node"], from_b["node"]), ("A", "B"))
        for key in ("init_time", "leads_h", "valid_times", "point", "units", "members"):
            self.assertEqual(from_a[key], from_b[key], key)

    def test_an_ensemble_can_be_split_across_nodes(self):
        a, b = self.fast_node("A"), self.fast_node("B")
        for total, first in ((4, 2), (5, 3), (40, 13)):
            with self.subTest(total=total, first=first):
                rest = total - first
                whole = call(a.url, "/v1/forecast", forecast_body(members=total, lead_hours=48))[2]["point"]
                one = call(a.url, "/v1/forecast", forecast_body(members=first, lead_hours=48))[2]["point"]
                two = call(b.url, "/v1/forecast",
                           forecast_body(members=rest, member_offset=first, lead_hours=48))[2]["point"]
                self.assertEqual(one["t2m_c"], whole["t2m_c"])                  # member 0 lives in the first shard
                self.assertIsNone(two["t2m_c"])
                for k in range(8):
                    for name in ("t2m_c_mean", "wind10_ms_mean", "msl_hpa_mean"):
                        merged = (first * one[name][k] + rest * two[name][k]) / total
                        self.assertAlmostEqual(merged, whole[name][k], delta=0.011, msg=name)
                    self.assertEqual(min(one["t2m_c_min"][k], two["t2m_c_min"][k]), whole["t2m_c_min"][k])
                    self.assertEqual(max(one["t2m_c_max"][k], two["t2m_c_max"][k]), whole["t2m_c_max"][k])

    def test_hooks_are_off_without_admin(self):
        server = self.start("--mock")
        for path, body in (("/admin/fail", {"mode": "down"}), ("/admin/contention", {"factor": 3}),
                           ("/admin/reset", None)):
            status, _, answer = call(server.url, path, body, method="POST")
            self.assertEqual(status, 403, path)
            self.assertIn("--admin", answer["error"])
        self.assertEqual(call(server.url, "/health")[0], 200)
        self.assertEqual(call(server.url, "/metrics")[2]["slowdown"], 1.0)

    def test_mock_needs_only_the_standard_library(self):
        code = ("import sys; sys.path.insert(0, %r); import forecast_server; "
                "print(sorted(m for m in ('numpy', 'torch', 'forecast_core', 'wx_common') if m in sys.modules))"
                % os.path.join(REPO, "serve"))
        done = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "[]"))
        server = self.start("--mock", python_flags=["-S"])                      # -S: no site-packages at all
        status, _, body = call(server.url, "/v1/forecast", forecast_body(members=3))
        self.assertEqual(status, 200, body)
        self.assertEqual(server.stop(), 0)

    def test_a_hung_request_does_not_block_shutdown(self):
        server = self.start("--mock", "--admin")
        self.assertEqual(call(server.url, "/admin/fail", {"mode": "hang"})[0], 200)
        outcome = []

        def hung():
            try:
                outcome.append(call(server.url, "/v1/forecast", forecast_body(), timeout=10))
            except Exception as e:
                outcome.append(e)

        threads = [threading.Thread(target=hung) for _ in range(3)]
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        t0 = time.time()
        self.assertEqual(server.stop(), 0)
        self.assertLess(time.time() - t0, 5)
        for thread in threads:
            thread.join(5)
        self.assertEqual(len(outcome), 3)

    def test_a_waiting_request_does_not_block_shutdown(self):
        server = self.start("--mock", "--ms-per-unit", "100")
        outcome = []

        def slow():
            try:
                body = forecast_body(members=40, lead_hours=120)
                outcome.append(call(server.url, "/v1/forecast", body, timeout=10)[0])
            except Exception as e:
                outcome.append(e)

        threads = [threading.Thread(target=slow) for _ in range(2)]             # 80 s of work each: it must not run
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        t0 = time.time()
        self.assertEqual(server.stop(), 0)
        self.assertLess(time.time() - t0, 5)
        for thread in threads:
            thread.join(5)
        self.assertEqual(len(outcome), 2)

    def test_latest_follows_the_clock(self):
        server = self.start("--mock")
        latest = call(server.url, "/v1/info")[2]["latest"]
        when = datetime.datetime.strptime(latest, "%Y-%m-%dT%H")
        age = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - when
        self.assertEqual(when.hour % 6, 0)
        self.assertGreaterEqual(age, datetime.timedelta(hours=4))               # old enough to exist
        self.assertLess(age, datetime.timedelta(hours=10, minutes=1))           # and the most recent such

    def test_bad_flags_stop_at_once(self):
        env = dict((k, v) for k, v in os.environ.items() if k not in ("WX_WEIGHTS", "WX_DATA"))
        for flags, word in ((["--mock", "--latest", "tomorrow"], "--latest"),
                            (["--mock", "--latest", "2026-11-14T03"], "--latest"),
                            (["--port", "0"], "--weights"),
                            (["--port", "0", "--weights", os.path.join(REPO, "no", "such.pt")], "not there"),
                            (["--port", "0", "--weights", SERVER, "--data", os.path.join(REPO, "no")], "stats.json")):
            with self.subTest(flags=flags):
                done = subprocess.run([sys.executable, SERVER] + flags, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
                self.assertEqual(done.returncode, 2)
                self.assertEqual(done.stdout, "")
                self.assertIn(word, done.stderr)

    def test_a_taken_port_is_reported(self):
        server = self.start("--mock")
        done = subprocess.run([sys.executable, SERVER, "--mock", "--host", "127.0.0.1", "--port", str(server.port)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual((done.returncode, done.stdout), (2, ""))
        self.assertIn("cannot listen", done.stderr)


class PiecesTest(unittest.TestCase):
    """Small parts of the server, called directly. None of them needs NumPy, PyTorch or a GPU."""

    def test_percentile(self):
        self.assertIsNone(fs.percentile([], 50))
        self.assertEqual(fs.percentile([7.0], 95), 7.0)
        hundred = [float(n) for n in range(1, 101)]
        self.assertEqual((fs.percentile(hundred, 50), fs.percentile(hundred, 95)), (50.0, 95.0))
        self.assertEqual(fs.percentile([1.0, 2.0, 3.0], 50), 2.0)

    def test_prometheus_text_leaves_unknowns_out(self):
        text = fs.prom_text({"node": "A", "queued": 3, "ms_per_unit": None, "gpu": {"util_pct": None, "temp_c": 70},
                             "busy_other": True, "served_total": 5})
        self.assertEqual(text, "# TYPE wx_queued gauge\nwx_queued 3\n# TYPE wx_gpu_temp_c gauge\nwx_gpu_temp_c 70\n"
                               "# TYPE wx_busy_other gauge\nwx_busy_other 1\n"
                               "# TYPE wx_served_total counter\nwx_served_total 5\n")

    def fake_nvidia_smi(self, gpu_line, app_lines, exit_code=0):
        """Put a pretend nvidia-smi first on PATH for this test."""
        folder = tempfile.mkdtemp(prefix="forecast_server_test_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "nvidia-smi")
        with open(path, "w") as f:
            f.write("#!/bin/sh\ncase \"$1\" in\n  --query-gpu=*) echo '%s' ;;\n"
                    "  --query-compute-apps=*) printf '%s' ;;\nesac\nexit %d\n"
                    % (gpu_line, "".join(line + "\\n" for line in app_lines), exit_code))
        os.chmod(path, 0o755)
        old = os.environ.get("PATH", "")
        os.environ["PATH"] = folder + os.pathsep + old
        self.addCleanup(os.environ.__setitem__, "PATH", old)

    def test_machine_numbers_from_nvidia_smi(self):
        me = os.getpid()
        self.fake_nvidia_smi("96, 70, 60.12", ["%d, 7115" % me, "4242, 30000"])
        gpu, host, busy_other = fs.read_machine(True)
        self.assertEqual(gpu, {"util_pct": 96, "temp_c": 70, "power_w": 60.1, "proc_mem_mb": 37115})
        self.assertIs(busy_other, True)                                         # pid 4242 holds 30 GB
        self.assertEqual(sorted(host), ["load1", "mem_available_gb"])

    def test_own_process_and_small_ones_are_not_busy_other(self):
        self.fake_nvidia_smi("3, 45, 11.0", ["%d, 7115" % os.getpid(), "4242, 300"])
        gpu, _, busy_other = fs.read_machine(True)
        self.assertEqual(gpu["proc_mem_mb"], 7415)
        self.assertIs(busy_other, False)

    def test_machine_numbers_that_nvidia_smi_does_not_know(self):
        self.fake_nvidia_smi("[N/A], 45, [N/A]", [])
        gpu, _, busy_other = fs.read_machine(True)
        self.assertEqual(gpu, {"util_pct": None, "temp_c": 45, "power_w": None, "proc_mem_mb": 0})
        self.assertIs(busy_other, False)

    def test_machine_numbers_without_nvidia_smi(self):
        nothing = {"util_pct": None, "temp_c": None, "power_w": None, "proc_mem_mb": None}
        self.fake_nvidia_smi("96, 70, 60", ["4242, 30000"], exit_code=9)        # it is there, but it fails
        self.assertEqual(fs.read_machine(True)[::2], (nothing, False))
        gpu, host, busy_other = fs.read_machine(False)                          # it is not there at all
        self.assertEqual((gpu, busy_other), (nothing, False))
        self.assertTrue(host["load1"] is None or host["load1"] >= 0)

    def test_mem_available(self):
        with tempfile.NamedTemporaryFile("w", prefix="forecast_server_test_") as f:
            f.write("MemTotal:       127535892 kB\nMemFree:         1000000 kB\nMemAvailable:   106115892 kB\n")
            f.flush()
            self.assertEqual(fs.mem_available_gb(f.name), 101.2)
        self.assertIsNone(fs.mem_available_gb(os.path.join(REPO, "no", "such", "file")))

    def test_slowdown_compares_like_with_like(self):
        engine = fs.RealEngine("weights", "data", "cpu", 20, threading.Event())
        self.assertEqual(engine.speed(), (None, 1.0))
        for _ in range(10):                                 # a healthy mix: big batches are cheaper per member
            engine.timed(20, 80, 80 * 6.0)
            engine.timed(1, 4, 4 * 12.0)
        self.assertAlmostEqual(engine.slowdown, 1.0)
        self.assertTrue(6.0 < engine.ms_per_unit < 12.0)
        for _ in range(20):                                 # a training job arrives: everything takes 1.5x
            engine.timed(20, 80, 80 * 9.0)
            engine.timed(1, 4, 4 * 18.0)
        self.assertAlmostEqual(engine.slowdown, 1.5, places=2)
        for _ in range(20):                                 # and leaves
            engine.timed(20, 80, 80 * 6.0)
        self.assertAlmostEqual(engine.slowdown, 1.0, places=2)

    def test_mock_weather_is_smooth_and_repeatable(self):
        when = datetime.datetime(2019, 1, 27, 12)
        here = fs.mock_weather(when, 41.96, -87.93)
        self.assertEqual(here, fs.mock_weather(when, 41.96, -87.93))
        near = fs.mock_weather(when, 41.97, -87.92)
        later = fs.mock_weather(when + datetime.timedelta(hours=6), 41.96, -87.93)
        for a, b in zip(here, near):
            self.assertLess(abs(a - b), 0.1)
        self.assertLess(abs(here[0] - later[0]), 8.0)
        self.assertLess(fs.mock_weather(when, 80.0, 0.0)[0], fs.mock_weather(when, 0.0, 0.0)[0])    # poles are colder
        noon = fs.mock_weather(datetime.datetime(2019, 7, 1, 18), 40.0, -45.0)[0]        # 15h local time
        night = fs.mock_weather(datetime.datetime(2019, 7, 1, 6), 40.0, -45.0)[0]        # 03h local time
        self.assertGreater(noon, night)


@unittest.skipIf(NO_REAL_MODE, NO_REAL_MODE)
class RealServerTest(unittest.TestCase):
    """The real model behind the server, on CPU. The tests run in name order and share one server."""

    @classmethod
    def setUpClass(cls):
        cls.data = tempfile.mkdtemp(prefix="forecast_server_test_")             # so pushed states do not land in .local
        cls.addClassCleanup(shutil.rmtree, cls.data, True)
        for name in os.listdir(BUNDLE):
            if os.path.isfile(os.path.join(BUNDLE, name)):
                os.symlink(os.path.join(BUNDLE, name), os.path.join(cls.data, name))
        cls.server = Running("--node", "T", "--device", "cpu", "--weights", WEIGHTS, "--data", cls.data)
        cls.addClassCleanup(cls.server.stop)
        cls.url = cls.server.url
        cls.first_health = call(cls.url, "/health")
        cls.first_forecast = call(cls.url, "/v1/forecast", forecast_body())
        deadline = time.time() + 45
        while True:
            status, _, body = call(cls.url, "/health")
            if status == 200:
                break
            if body["status"] != "loading" or time.time() > deadline:
                raise AssertionError("the model did not load: %s\n%s" % (body, cls.server.stderr()))
            time.sleep(0.25)

    def test_1_health_says_loading_until_the_model_is_ready(self):
        status, _, body = self.first_health
        self.assertEqual(status, 503)
        self.assertEqual(dict((k, body[k]) for k in ("status", "node", "model", "ready")),
                         {"status": "loading", "node": "T", "model": "afno-19m", "ready": False})
        status, _, body = self.first_forecast
        self.assertEqual(status, 503)
        self.assertIn("loading", body["error"])
        status, _, body = call(self.url, "/health")
        self.assertEqual((status, dict((k, body[k]) for k in ("status", "node", "model", "ready"))),
                         (200, {"status": "ok", "node": "T", "model": "afno-19m", "ready": True}))

    def test_2_info(self):
        status, _, body = call(self.url, "/v1/info")
        self.assertEqual(status, 200)
        self.assertEqual((body["node"], body["model"], body["params"], body["device"], body["mock"]),
                         ("T", "afno-19m", 19128832, "cpu", False))
        self.assertEqual(body["latest"], "2019-02-03T18")                       # no states/ yet: the bundle's last time
        self.assertEqual(body["ranges"], [{"first": "2019-01-20T00", "last": "2019-02-03T18"}])
        self.assertGreater(body["ms_per_unit"], 0)                              # measured while warming up

    def test_3_forecast_through_the_model(self):
        status, _, body = call(self.url, "/v1/forecast", forecast_body(members=2, with_truth=True, request_id="r1"),
                               timeout=60)
        self.assertEqual(status, 200, body)
        self.assertEqual((body["request_id"], body["node"], body["model"]), ("r1", "T", "afno-19m"))
        self.assertEqual((body["init_time"], body["members"], body["member_offset"], body["units"]),
                         ("2019-01-27T12", 2, 0, 8))
        self.assertEqual(body["leads_h"], [6, 12, 18, 24])
        self.assertEqual(body["valid_times"], ["2019-01-27T18", "2019-01-28T00", "2019-01-28T06", "2019-01-28T12"])
        point = body["point"]
        self.assertEqual(sorted(point), sorted(["lat", "lon", "truth_t2m_c"] + SERIES))
        for name in SERIES + ["truth_t2m_c"]:
            self.assertEqual(len(point[name]), 4, name)
            for value in point[name]:
                self.assertIsInstance(value, float, name)
        for k in range(4):
            self.assertLessEqual(point["t2m_c_min"][k], point["t2m_c_mean"][k])
            self.assertLessEqual(point["t2m_c_mean"][k], point["t2m_c_max"][k])
            self.assertIn(point["t2m_c"][k], (point["t2m_c_min"][k], point["t2m_c_max"][k]))    # one of the two
            self.assertLess(point["t2m_c_max"][k] - point["t2m_c_min"][k], 3.0)   # the other one was nudged, not thrown
            self.assertTrue(-45 < point["t2m_c"][k] < 10, point["t2m_c"])         # Chicago, late January 2019
            self.assertLess(abs(point["t2m_c"][k] - point["truth_t2m_c"][k]), 6.0)    # a day ahead: near the truth
            self.assertTrue(0 <= point["wind10_ms_mean"][k] < 30)
            self.assertTrue(970 < point["msl_hpa_mean"][k] < 1060)
        self.assertNotEqual(point["t2m_c_min"], point["t2m_c_max"])              # member 1 is perturbed
        timing = body["timing_ms"]
        self.assertGreater(timing["gpu"], 0)
        self.assertGreaterEqual(timing["total"], timing["gpu"])

        alone = call(self.url, "/v1/forecast", forecast_body(), timeout=60)[2]["point"]     # member 0 by itself
        for a, b in zip(alone["t2m_c"], point["t2m_c"]):
            self.assertAlmostEqual(a, b, delta=0.02)
        self.assertEqual(alone["t2m_c"], alone["t2m_c_mean"])

        m = call(self.url, "/metrics")[2]
        self.assertEqual((m["node"], m["served_total"], m["units_total"], m["errors_total"]), ("T", 2, 12, 0))
        self.assertEqual((m["inflight"], m["queued"], m["queued_units"]), (0, 0, 0))
        self.assertGreater(m["ms_per_unit"], 0)
        self.assertGreaterEqual(m["slowdown"], 1.0)
        self.assertGreater(m["latency_ms"]["p50"], 0)
        self.assertEqual(sorted(m["gpu"]), ["power_w", "proc_mem_mb", "temp_c", "util_pct"])    # null on a laptop
        self.assertEqual(sorted(m["host"]), ["load1", "mem_available_gb"])
        self.assertIn(m["busy_other"], (True, False))
        self.assertIn("wx_units_total 12", call(self.url, "/metrics?format=prom")[2])

    def test_4_bad_requests(self):
        status, _, body = call(self.url, "/v1/forecast", forecast_body(init="2018-06-01T00"))
        self.assertEqual(status, 404, body)
        self.assertIn("not held", body["error"])
        self.assertEqual(call(self.url, "/v1/forecast", forecast_body(lead_hours=7))[0], 400)
        for npy_b64 in ("not base64 !!!", "bm90IGEgbnVtcHkgZmlsZQ=="):
            status, _, body = call(self.url, "/v1/forecast",
                                   forecast_body(init={"npy_b64": npy_b64, "time": "2026-11-14T00"}))
            self.assertEqual(status, 400, body)
            self.assertIn("npy_b64", body["error"])
        self.assertEqual(call(self.url, "/admin/fail", {"mode": "down"})[0], 403)        # started without --admin
        self.assertEqual(call(self.url, "/health")[0], 200)

    def test_5_push_a_state(self):
        import numpy as np

        def packed(array):
            buffer = io.BytesIO()
            np.save(buffer, array)
            return base64.b64encode(buffer.getvalue()).decode("ascii")

        rows = np.load(os.path.join(BUNDLE, "states_2019-01-20T00_60.npy"), mmap_mode="r")
        state = np.array(rows[-1])                                              # the analysis of 2019-02-03T18
        for wrong in (state.astype(np.float32), state[:, :60], np.full_like(state, np.nan)):
            status, _, body = call(self.url, "/v1/state", {"time": "2019-02-04T00", "npy_b64": packed(wrong)})
            self.assertEqual(status, 400, body)
        status, _, body = call(self.url, "/v1/state", {"time": "2019-02-04T00", "npy_b64": packed(state)})
        self.assertEqual((status, body), (200, {"latest": "2019-02-04T00"}))
        saved = np.load(os.path.join(self.data, "states", "2019020400.npy"))    # written, so it survives a restart
        self.assertTrue((saved == state).all())
        info = call(self.url, "/v1/info")[2]
        self.assertEqual(info["latest"], "2019-02-04T00")
        self.assertEqual(info["ranges"], [{"first": "2019-01-20T00", "last": "2019-02-04T00"}])
        status, _, body = call(self.url, "/v1/state", {"time": "2018-01-01T00", "npy_b64": packed(state)})
        self.assertEqual((status, body), (200, {"latest": "2019-02-04T00"}))    # an older one does not become latest

        status, _, body = call(self.url, "/v1/forecast",
                               forecast_body(init="latest", lead_hours=6, with_truth=True), timeout=60)
        self.assertEqual(status, 200, body)
        self.assertEqual((body["init_time"], body["valid_times"]), ("2019-02-04T00", ["2019-02-04T06"]))
        self.assertEqual(body["point"]["truth_t2m_c"], [None])                  # nothing held for the valid time
        status, _, sent = call(self.url, "/v1/forecast", forecast_body(
            init={"npy_b64": packed(state), "time": "2026-11-14T00"}, lead_hours=6), timeout=60)
        self.assertEqual(status, 200, sent)
        self.assertEqual(sent["init_time"], "2026-11-14T00")
        self.assertAlmostEqual(sent["point"]["t2m_c"][0], body["point"]["t2m_c"][0], delta=0.02)    # the same state

    def test_6_stops_cleanly(self):
        self.assertEqual(self.server.stop(), 0)


@unittest.skipIf(NO_REAL_MODE, NO_REAL_MODE)
class RealStartStopTest(unittest.TestCase):
    """A load that fails must show in /health. SIGTERM must end the real server with code 0 at any moment,
    also while the model loads or is busy."""

    def start(self):
        server = Running("--device", "cpu", "--weights", WEIGHTS, "--data", BUNDLE)
        self.addCleanup(server.stop)
        return server

    def test_a_load_that_fails_is_reported_not_fatal(self):
        data = tempfile.mkdtemp(prefix="forecast_server_test_")
        self.addCleanup(shutil.rmtree, data, True)
        with open(os.path.join(data, "stats.json"), "w") as f:              # not the statistics of our 20 channels
            json.dump({"channels": ["some", "other", "model"], "mean": [0, 0, 0], "std": [1, 1, 1]}, f)
        server = Running("--device", "cpu", "--weights", WEIGHTS, "--data", data)
        self.addCleanup(server.stop)
        deadline = time.time() + 45
        while True:
            status, _, body = call(server.url, "/health")
            if body["status"] != "loading":
                break
            self.assertLess(time.time(), deadline, "still loading")
            time.sleep(0.2)
        self.assertEqual(status, 503)
        self.assertEqual((body["status"], body["ready"], body["node"]), ("failed", False, "A"))
        self.assertIn("channels", body["error"])                            # /health says why
        status, _, body = call(server.url, "/v1/forecast", forecast_body())
        self.assertEqual(status, 503)
        self.assertIn("failed", body["error"])
        self.assertIn("FAILED to load the model", server.stderr())
        self.assertEqual(server.stop(), 0)

    def test_stop_while_the_model_loads(self):
        server = self.start()
        self.assertEqual(call(server.url, "/health")[0], 503)                   # still loading
        t0 = time.time()
        self.assertEqual(server.stop(), 0)
        self.assertLess(time.time() - t0, 5)

    def test_stop_in_the_middle_of_a_forecast(self):
        server = self.start()
        deadline = time.time() + 45
        while call(server.url, "/health")[0] != 200:
            self.assertLess(time.time(), deadline, "the model did not load")
            time.sleep(0.25)
        outcome = []

        def long_forecast():                                # 40 units: several seconds on CPU
            try:
                body = forecast_body(members=2, lead_hours=120)
                outcome.append(call(server.url, "/v1/forecast", body, timeout=60)[0])
            except Exception as e:
                outcome.append(e)

        thread = threading.Thread(target=long_forecast)
        thread.start()
        time.sleep(0.6)
        self.assertEqual(call(server.url, "/metrics")[2]["inflight"], 1)        # the model is at work
        t0 = time.time()
        self.assertEqual(server.stop(), 0)
        self.assertLess(time.time() - t0, 4)                                    # it did not wait for the forecast
        thread.join(10)
        self.assertEqual(len(outcome), 1)
        self.assertNotEqual(outcome[0], 200)                                    # a 503, or the connection dropped


if __name__ == "__main__":
    unittest.main()
