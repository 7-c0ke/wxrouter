"""Tests for the agent's side: agent/tools.py, agent/weather_mcp.py and agent/desk_agent.py.

    .venv/bin/python -m unittest tests/test_agent.py -v

A fake router (a tiny HTTP server on 127.0.0.1, on a port the OS picks) gives canned answers for every
tool and plays a scripted model. The MCP server and the desk agent also run as child processes, the way
a client would start them. No GPU, no real router, no network. Takes a few seconds.

Written at the hackathon (Nov 13-14, 2026).
"""
import contextlib
import datetime
import http.server
import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(os.path.dirname(HERE), "agent")
MCP_SERVER = os.path.join(AGENT, "weather_mcp.py")
DESK_AGENT = os.path.join(AGENT, "desk_agent.py")
sys.path.insert(0, AGENT)

import desk_agent  # noqa: E402
import tools       # noqa: E402

TOOL_NAMES = ["forecast_point", "ohare_tomorrow", "ensemble_range", "storm_2019", "cluster_status",
              "set_routing_policy"]
OHARE = {"lat": 41.9602, "lon": -87.9316}


# ---------------------------------------------------------------- the fake router and its canned answers

# "latest" forecasts: the temperature depends only on the valid time's UTC hour. Chicago is 6 hours
# behind, so tomorrow's four values are 1.26 (00 h local), -2.63 (06 h), 6.71 (12 h) and 4.44 (18 h).
HOUR_TEMP = {6: 1.26, 12: -2.63, 18: 6.71, 0: 4.44}

# The 2019 storm, +6 h to +72 h. The first truth value is null: "null where the server has no data".
STORM_INIT = "2019-01-27T12"
STORM_FORECAST = [-10.12, -11.41, -13.04, -15.33, -16.92, -18.18, -20.04, -21.76, -23.52, -25.07, -26.63, -27.94]
STORM_TRUTH = [None, -12.02, -13.91, -16.08, -17.83, -19.88, -21.71, -23.44, -25.93, -27.31, -29.04, -30.21]

STATUS = {
    "policy": "adaptive", "uptime_s": 120.5,
    "counters": {"routed": 100, "failovers": 1, "switches": 2, "heals": 0, "errors": 0},
    "pools": {
        "forecast": {"preferred": "A", "backends": [
            {"name": "A", "url": "http://127.0.0.1:8101", "probe": "forecast", "healthy": True, "draining": False,
             "healing": False, "inflight": 1, "latency_ewma_ms": 210.4, "score": 0.42, "routed": 60,
             "failures": 0, "last_error": None,
             "probe_data": {"queued": 0, "queued_units": 0, "gpu_util_pct": 96, "busy_other": False,
                            "slowdown": 1.0, "mem_available_gb": 101.2, "ms_per_unit": 10.2}},
            # A dead backend, full of nulls. No reader may crash on a null.
            {"name": "B", "url": "http://127.0.0.1:8102", "probe": "forecast", "healthy": False, "draining": False,
             "healing": True, "inflight": 0, "latency_ewma_ms": None, "score": None, "routed": 40,
             "failures": 3, "last_error": "connection refused", "probe_data": None}]},
        "llm": {"preferred": "B", "backends": [
            {"name": "A", "url": "http://127.0.0.1:8001", "probe": "llamacpp", "healthy": True, "draining": True,
             "healing": False, "inflight": 0, "latency_ewma_ms": 850.0, "score": 0.048649, "routed": 12,
             "failures": 0, "last_error": None, "probe_data": {"queued": 0, "running": 1}},
            {"name": "B"}]},            # nearly every field missing
    },
    "decisions": [
        {"ts": 1794550000.1 + i, "pool": "forecast", "backend": backend, "policy": "adaptive", "kind": kind,
         "reason": reason, "retries": 0}
        for i, (backend, kind, reason) in enumerate([
            ("A", "route", "preferred"), ("B", "route", "spill"), ("B", "unhealthy", "3 straight failures"),
            ("A", "failover", "B refused the connection"), ("A", "switch", "B is unhealthy"),
            ("B", "heal", "ran the heal action, attempt 1"), ("A", "route", "preferred")])],
}


def fake_forecast(body, latest):
    """A /v1/forecast answer with made-up, repeatable numbers. Returns (HTTP status, data)."""
    init = body.get("init")
    members = body.get("members", 1)
    leads = list(range(6, body.get("lead_hours", 24) + 1, 6))
    if init == "latest":
        init_time = latest
    elif init == STORM_INIT:
        init_time = init
    else:
        return 404, {"error": "init time not held"}
    start = datetime.datetime.strptime(init_time, "%Y-%m-%dT%H")
    valid = [start + datetime.timedelta(hours=lead) for lead in leads]
    mean = STORM_FORECAST[:len(leads)] if init == STORM_INIT else [HOUR_TEMP[when.hour] for when in valid]
    spread = 0.0 if members == 1 else 1.0
    point = {
        "lat": body["point"]["lat"], "lon": body["point"]["lon"],
        "t2m_c": [value + 0.3 * spread for value in mean],          # member 0 is not the mean
        "t2m_c_mean": mean,
        "t2m_c_min": [value - 0.71 * spread for value in mean],
        "t2m_c_max": [value + 0.83 * spread for value in mean],
        "wind10_ms_mean": [4.26 + 0.1 * i for i in range(len(leads))],
        "msl_hpa_mean": [1013.25 - 0.5 * i for i in range(len(leads))],
    }
    if body.get("with_truth"):
        point["truth_t2m_c"] = STORM_TRUTH[:len(leads)] if init == STORM_INIT else [None] * len(leads)
    data = {"request_id": body.get("request_id"), "node": "A", "model": "afno-19m", "init_time": init_time,
            "members": members, "member_offset": 0, "leads_h": leads,
            "valid_times": [when.strftime("%Y-%m-%dT%H") for when in valid], "point": point,
            "units": members * len(leads), "timing_ms": {"queue": 0.4, "gpu": 212.0, "total": 214.1}}
    if body["point"]["lat"] == 0.0:     # a sparse answer: member 0 absent, nulls, fields missing
        point["t2m_c"] = None
        point["wind10_ms_mean"] = [None] * len(leads)
        del point["msl_hpa_mean"]
        del data["valid_times"]
        del data["timing_ms"]
    return 200, data


def fake_chat(body):
    """A scripted model. Which script it plays depends on the model name in the request."""
    model = body.get("model")
    tool_results = [message for message in body.get("messages", []) if message.get("role") == "tool"]

    def reply(content=None, tool=None, arguments="{}"):
        message = {"role": "assistant", "content": content}
        if tool:
            message["tool_calls"] = [{"id": "call_%d" % (len(tool_results) + 1), "type": "function",
                                      "function": {"name": tool, "arguments": arguments}}]
        return {"id": "chatcmpl-1", "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}]}

    if model == "always-tools":
        return reply(tool="cluster_status")
    if model == "no-tools":
        return reply("It will be 21.5 C tomorrow. Bring a coat.")
    if model == "object-arguments":            # some servers send the arguments as an object, and no id
        if tool_results:
            return reply("Done.")
        answer = reply(tool="set_routing_policy")
        answer["choices"][0]["message"]["tool_calls"] = [
            {"type": "function", "function": {"name": "set_routing_policy", "arguments": {"policy": "static"}}}]
        return answer
    if model == "bad-arguments":
        return reply("The tool did not work.") if tool_results else reply(tool="forecast_point", arguments="{lat: 4")
    # "scripted": first a tool call, then an answer with one true number and one invented number.
    if not tool_results:
        return reply(tool="storm_2019")
    return reply("<think>The tool gave -27.9.</think>At +72 hours the forecast was -27.9 C. "
                 "The real temperature was -45.6 C.")


class FakeRouter(http.server.BaseHTTPRequestHandler):
    """Canned answers for everything the agent's side asks the router. It remembers each request."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.route(None)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            self.route(json.loads(raw.decode("utf-8")))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.reply(400, {"error": "bad request: %s" % exc})

    def route(self, body):
        server = self.server
        server.seen.append({"method": self.command, "path": self.path, "body": body,
                            "session": self.headers.get("X-Session-Id")})
        where = (self.command, self.path)
        if where == ("GET", "/router/status"):
            self.reply(200, STATUS)
        elif where == ("POST", "/router/policy"):
            if body.get("policy") in ("static", "round_robin", "least_loaded", "adaptive"):
                self.reply(200, {"policy": body["policy"]})
            else:
                self.reply(400, {"error": "unknown policy"})
        elif where == ("POST", "/v1/forecast"):
            if body.get("init") == "not-json":
                self.reply(200, None, backend="A")
            else:
                status, data = fake_forecast(body, server.latest or newest_cycle())
                self.reply(status, data, backend="A")
        elif where == ("POST", "/v1/ensemble"):
            status, data = fake_forecast(body, server.latest or newest_cycle())
            first = body["members"] - body["members"] // 2
            data["shards"] = [
                {"backend": "A", "members": first, "member_offset": 0, "ms": 1304.6, "retried": False},
                {"backend": "B", "members": body["members"] - first, "member_offset": first, "ms": 903.2,
                 "retried": True}]
            self.reply(status, data)            # no X-Backend: two nodes answered
        elif where == ("GET", "/v1/models"):
            self.reply(200, {"object": "list", "data": [{"id": "scripted", "object": "model"}]}, backend="B")
        elif where == ("POST", "/v1/chat/completions"):
            self.reply(200, fake_chat(body), backend="B")
        elif where == ("GET", "/empty"):
            self.reply(200, "")
        elif where == ("GET", "/vllm-error"):
            self.reply(404, {"object": "error", "message": "The model `x` does not exist.", "code": 404})
        elif where == ("GET", "/openai-error"):
            self.reply(500, {"error": {"message": "the model fell over", "type": "server_error"}})
        else:
            self.reply(404, {"error": "no such path"})

    def reply(self, status, data, backend=None):
        """Send JSON. data None sends something that is not JSON; data "" sends an empty body."""
        if data is None:
            payload = b"<html>oops</html>"
        elif data == "":
            payload = b""
        else:
            payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if backend:
            self.send_header("X-Backend", backend)
            self.send_header("X-Router-Policy", "adaptive")
        self.end_headers()
        self.wfile.write(payload)


def newest_cycle():
    """The newest 6-hourly time that is not in the future, such as "2026-11-14T06"."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(hour=now.hour - now.hour % 6).strftime("%Y-%m-%dT%H")


def dead_address():
    """An address on this machine that refuses connections: a port the OS just gave us, closed again."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return "http://127.0.0.1:%d" % sock.getsockname()[1]


ROUTER = None           # the fake router, one for the whole file
ROUTER_THREAD = None
OLD_ROUTER_URL = None


def setUpModule():
    global ROUTER, ROUTER_THREAD, OLD_ROUTER_URL
    ROUTER = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)
    ROUTER.daemon_threads = True
    ROUTER.seen = []
    ROUTER.latest = None
    ROUTER_THREAD = threading.Thread(target=ROUTER.serve_forever, daemon=True)
    ROUTER_THREAD.start()
    OLD_ROUTER_URL = os.environ.get("ROUTER_URL")
    os.environ["ROUTER_URL"] = "http://127.0.0.1:%d" % ROUTER.server_address[1]


def tearDownModule():
    ROUTER.shutdown()
    ROUTER.server_close()
    ROUTER_THREAD.join(5)
    if OLD_ROUTER_URL is None:
        os.environ.pop("ROUTER_URL", None)
    else:
        os.environ["ROUTER_URL"] = OLD_ROUTER_URL


class AgentTestCase(unittest.TestCase):
    """Every test starts with a fake router that has seen nothing and holds a fresh "latest"."""

    def setUp(self):
        del ROUTER.seen[:]
        ROUTER.latest = None

    def seen(self, path):
        return [request for request in ROUTER.seen if request["path"] == path]

    def set_clock(self, *when):
        """Make tools.py believe it is this UTC time, until the test ends."""
        real = tools._utc_now
        tools._utc_now = lambda: datetime.datetime(*when)
        self.addCleanup(setattr, tools, "_utc_now", real)


# ---------------------------------------------------------------- agent/tools.py

class ToolListTest(unittest.TestCase):

    def test_names_descriptions_and_schemas(self):
        self.assertEqual([tool["name"] for tool in tools.TOOLS], TOOL_NAMES)
        json.dumps(tools.TOOLS)                                 # plain data
        for tool in tools.TOOLS:
            description = tool["description"]
            self.assertTrue(description.endswith("."), description)
            self.assertNotIn(". ", description, "one sentence: " + description)
            schema = tool["input_schema"]
            self.assertEqual(schema["type"], "object")
            for name in schema["required"]:
                self.assertIn(name, schema["properties"])
            for spec in schema["properties"].values():
                self.assertIn(spec["type"], ("number", "integer", "string"))
                self.assertTrue(spec["description"])
        required = dict((tool["name"], tool["input_schema"]["required"]) for tool in tools.TOOLS)
        self.assertEqual(required, {"forecast_point": ["lat", "lon"], "ohare_tomorrow": [],
                                    "ensemble_range": ["lat", "lon"], "storm_2019": [], "cluster_status": [],
                                    "set_routing_policy": ["policy"]})

    def test_router_url_comes_from_the_environment(self):
        old = os.environ.get("ROUTER_URL")
        try:
            os.environ["ROUTER_URL"] = "http://nodeA:8100/"
            self.assertEqual(tools.router_url(), "http://nodeA:8100")
            del os.environ["ROUTER_URL"]
            self.assertEqual(tools.router_url(), "http://127.0.0.1:8100")
        finally:
            if old is not None:
                os.environ["ROUTER_URL"] = old


class CallToolTest(AgentTestCase):
    """call_tool checks names and types before anything is sent."""

    def assert_refused(self, name, arguments, *words):
        result = tools.call_tool(name, arguments)
        self.assertEqual(list(result), ["error"], result)
        for word in words:
            self.assertIn(word, result["error"])
        self.assertEqual(ROUTER.seen, [], "nothing may reach the router")

    def test_unknown_tool(self):
        self.assert_refused("make_it_rain", {}, "unknown tool", "make_it_rain", "ohare_tomorrow")
        self.assert_refused(None, {}, "unknown tool")

    def test_missing_required_argument(self):
        self.assert_refused("forecast_point", {"lat": 41.9}, "missing required argument", "'lon'")
        self.assert_refused("forecast_point", {"lat": 41.9, "lon": None}, "missing required argument", "'lon'")
        self.assert_refused("set_routing_policy", None, "missing required argument", "'policy'")

    def test_wrong_types(self):
        self.assert_refused("forecast_point", {"lat": "41.9", "lon": -87.9}, "'lat'", "a number", "'41.9'")
        self.assert_refused("forecast_point", {"lat": True, "lon": -87.9}, "'lat'", "a number", "true")
        self.assert_refused("forecast_point", {"lat": 41.9, "lon": float("nan")}, "'lon'", "a number")
        self.assert_refused("forecast_point", dict(OHARE, lead_hours=24.5), "'lead_hours'", "a whole number")
        self.assert_refused("forecast_point", dict(OHARE, lead_hours="24"), "'lead_hours'", "a whole number")
        self.assert_refused("forecast_point", dict(OHARE, init=2019), "'init'", "text")
        self.assert_refused("ensemble_range", dict(OHARE, members=[20]), "'members'", "a list")

    def test_unknown_argument(self):
        self.assert_refused("forecast_point", dict(OHARE, hours=48), "unknown argument", "'hours'", "lead_hours")
        self.assert_refused("ohare_tomorrow", {"day": "Sunday"}, "unknown argument", "no arguments")

    def test_value_not_allowed(self):
        self.assert_refused("set_routing_policy", {"policy": "fastest"}, "one of", "adaptive", "'fastest'")

    def test_arguments_must_be_an_object(self):
        self.assert_refused("cluster_status", [1, 2], "JSON object", "a list")
        self.assert_refused("cluster_status", "now", "JSON object")
        self.assert_refused("cluster_status", {1: "a", "b": 2})         # keys no JSON object could have

    def test_whole_number_as_float_and_null_for_an_optional(self):
        result = tools.call_tool("forecast_point", dict(OHARE, lead_hours=12.0, members=None, init=None))
        self.assertNotIn("error", result)
        body = self.seen("/v1/forecast")[0]["body"]
        self.assertEqual(body["lead_hours"], 12)
        self.assertIsInstance(body["lead_hours"], int)
        self.assertEqual((body["members"], body["init"]), (1, "latest"))


class ForecastToolsTest(AgentTestCase):

    def test_forecast_point(self):
        ROUTER.latest = "2026-11-14T06"
        result = tools.forecast_point(41.9602, -87.9316, lead_hours=24)
        # What was sent is the contract's /v1/forecast body.
        self.assertEqual(self.seen("/v1/forecast")[0]["body"],
                         {"init": "latest", "lead_hours": 24, "members": 1, "point": OHARE, "with_truth": False})
        self.assertEqual(result["by_lead"], [
            {"lead_h": 6, "valid_time_utc": "2026-11-14T12", "temp_c": -2.6, "wind_mps": 4.3, "pressure_hpa": 1013.2},
            {"lead_h": 12, "valid_time_utc": "2026-11-14T18", "temp_c": 6.7, "wind_mps": 4.4, "pressure_hpa": 1012.8},
            {"lead_h": 18, "valid_time_utc": "2026-11-15T00", "temp_c": 4.4, "wind_mps": 4.5, "pressure_hpa": 1012.2},
            {"lead_h": 24, "valid_time_utc": "2026-11-15T06", "temp_c": 1.3, "wind_mps": 4.6, "pressure_hpa": 1011.8},
        ])
        self.assertEqual(result["last_lead"], result["by_lead"][-1])
        del result["by_lead"], result["last_lead"]
        self.assertEqual(result, {"lat": 41.9602, "lon": -87.9316, "init_time_utc": "2026-11-14T06",
                                  "model": "afno-19m", "members": 1, "total_ms": 214, "node": "A"})

    def test_forecast_point_with_members_and_a_past_init(self):
        result = tools.forecast_point(41.9602, -87.9316, lead_hours=12, members=5, init=STORM_INIT)
        body = self.seen("/v1/forecast")[0]["body"]
        self.assertEqual((body["init"], body["members"]), (STORM_INIT, 5))
        self.assertEqual(result["init_time_utc"], STORM_INIT)
        # temp_c is member 0 (the mean plus 0.3 in the fake), next to the lowest and highest member.
        self.assertEqual(result["last_lead"], {"lead_h": 12, "valid_time_utc": "2019-01-28T00", "temp_c": -11.1,
                                               "temp_min_c": -12.1, "temp_max_c": -10.6, "wind_mps": 4.4,
                                               "pressure_hpa": 1012.8})

    def test_forecast_point_puts_up_with_nulls_and_missing_fields(self):
        ROUTER.latest = "2026-11-14T06"
        result = tools.forecast_point(0.0, 10.0, lead_hours=12)
        # No member 0: the mean is used. No valid_times: worked out from the init time.
        self.assertEqual(result["by_lead"], [
            {"lead_h": 6, "valid_time_utc": "2026-11-14T12", "temp_c": -2.6, "wind_mps": None, "pressure_hpa": None},
            {"lead_h": 12, "valid_time_utc": "2026-11-14T18", "temp_c": 6.7, "wind_mps": None, "pressure_hpa": None}])
        self.assertIsNone(result["total_ms"])
        self.assertEqual(result["node"], "A")

    def test_ohare_tomorrow(self):
        self.set_clock(2026, 11, 14, 15, 0)             # Saturday 9 AM in Chicago
        ROUTER.latest = "2026-11-14T06"
        result = tools.ohare_tomorrow()
        self.assertEqual(result, {
            "place": "Chicago O'Hare", "lat": 41.9602, "lon": -87.9316,
            "date_local": "2026-11-15",
            "local_time": "UTC-6, a fixed offset (Chicago standard time, right for November)",
            "init_time_utc": "2026-11-14T06",
            "low": {"temp_c": -2.6, "valid_time_utc": "2026-11-15T12", "valid_time_local": "2026-11-15 06:00"},
            "high": {"temp_c": 6.7, "valid_time_utc": "2026-11-15T18", "valid_time_local": "2026-11-15 12:00"},
            "near_6am": {"temp_c": -2.6, "wind_mps": 4.7, "valid_time_utc": "2026-11-15T12",
                         "valid_time_local": "2026-11-15 06:00"},
            "node": "A"})
        # One request, from "latest", far enough to reach Sunday 18:00 local (Monday 00 UTC).
        self.assertEqual([request["body"] for request in self.seen("/v1/forecast")],
                         [{"init": "latest", "lead_hours": 48, "members": 1, "point": OHARE, "with_truth": False}])

    def test_ohare_tomorrow_asks_again_when_the_newest_analysis_is_old(self):
        self.set_clock(2026, 11, 14, 15, 0)
        ROUTER.latest = "2026-11-13T00"                 # 39 hours old
        result = tools.ohare_tomorrow()
        self.assertEqual([request["body"]["lead_hours"] for request in self.seen("/v1/forecast")], [48, 72])
        self.assertEqual(result["init_time_utc"], "2026-11-13T00")
        self.assertEqual(result["date_local"], "2026-11-15")
        self.assertEqual((result["low"]["temp_c"], result["low"]["valid_time_utc"]), (-2.6, "2026-11-15T12"))
        self.assertEqual((result["high"]["temp_c"], result["high"]["valid_time_utc"]), (6.7, "2026-11-15T18"))

    def test_ohare_tomorrow_says_so_when_the_newest_analysis_is_too_old(self):
        self.set_clock(2026, 11, 14, 15, 0)
        ROUTER.latest = "2026-11-09T00"
        result = tools.ohare_tomorrow()
        self.assertIn("too old", result["error"])
        self.assertIn("2026-11-09T00", result["error"])
        self.assertEqual(result["node"], "A")

    def test_ohare_tomorrow_around_local_midnight(self):
        ROUTER.latest = "2026-11-15T00"
        self.set_clock(2026, 11, 15, 5, 50)             # Saturday 23:50 in Chicago: tomorrow is Sunday
        late = tools.ohare_tomorrow()
        self.assertEqual(late["date_local"], "2026-11-15")
        self.assertEqual(late["near_6am"]["valid_time_local"], "2026-11-15 06:00")
        self.set_clock(2026, 11, 15, 6, 10)             # Sunday 00:10 in Chicago: tomorrow is Monday
        early = tools.ohare_tomorrow()
        self.assertEqual(early["date_local"], "2026-11-16")
        self.assertEqual(early["low"]["valid_time_local"], "2026-11-16 06:00")
        self.assertEqual(early["high"]["valid_time_local"], "2026-11-16 12:00")
        self.assertEqual([request["body"]["lead_hours"] for request in self.seen("/v1/forecast")], [36, 54])

    def test_ensemble_range(self):
        ROUTER.latest = "2026-11-14T06"
        result = tools.ensemble_range(41.9602, -87.9316, lead_hours=12)
        self.assertEqual(self.seen("/v1/ensemble")[0]["body"],
                         {"init": "latest", "lead_hours": 12, "members": 20, "point": OHARE, "with_truth": False})
        self.assertEqual(result, {
            "lat": 41.9602, "lon": -87.9316, "init_time_utc": "2026-11-14T06", "members": 20,
            "last_lead": {"lead_h": 12, "valid_time_utc": "2026-11-14T18", "temp_mean_c": 6.7, "temp_min_c": 6.0,
                          "temp_max_c": 7.5},
            "by_lead": [
                {"lead_h": 6, "valid_time_utc": "2026-11-14T12", "temp_mean_c": -2.6, "temp_min_c": -3.3,
                 "temp_max_c": -1.8},
                {"lead_h": 12, "valid_time_utc": "2026-11-14T18", "temp_mean_c": 6.7, "temp_min_c": 6.0,
                 "temp_max_c": 7.5}],
            "shards": [{"node": "A", "members": 10, "ms": 1305, "retried": False},
                       {"node": "B", "members": 10, "ms": 903, "retried": True}],
            "node": "A+B"})

    def test_storm_2019(self):
        result = tools.storm_2019()
        self.assertEqual(self.seen("/v1/forecast")[0]["body"],
                         {"init": "2019-01-27T12", "lead_hours": 72, "members": 1, "point": OHARE, "with_truth": True})
        self.assertEqual(result["checks"], [
            {"lead_h": 24, "valid_time_utc": "2019-01-28T12", "forecast_temp_c": -15.3, "truth_temp_c": -16.1},
            {"lead_h": 48, "valid_time_utc": "2019-01-29T12", "forecast_temp_c": -21.8, "truth_temp_c": -23.4},
            {"lead_h": 72, "valid_time_utc": "2019-01-30T12", "forecast_temp_c": -27.9, "truth_temp_c": -30.2}])
        self.assertEqual(result["error_at_72h_c"], 2.3)
        self.assertIn("forecast minus truth", result["error_is"])
        self.assertEqual((result["init_time_utc"], result["node"]), ("2019-01-27T12", "A"))


class RouterToolsTest(AgentTestCase):

    def test_cluster_status(self):
        result = tools.cluster_status()
        self.assertEqual(result["policy"], "adaptive")
        self.assertEqual(result["counters"], {"routed": 100, "failovers": 1, "switches": 2, "heals": 0, "errors": 0})
        self.assertEqual(result["node"], "router")
        forecast, llm = result["pools"]["forecast"], result["pools"]["llm"]
        self.assertEqual((forecast["preferred"], llm["preferred"]), ("A", "B"))
        self.assertEqual(forecast["backends"][0], {
            "name": "A", "healthy": True, "draining": False, "healing": False, "inflight": 1, "latency_ms": 210,
            "score": 0.42, "busy_other": False, "last_error": None})
        self.assertEqual(forecast["backends"][1], {
            "name": "B", "healthy": False, "draining": False, "healing": True, "inflight": 0, "latency_ms": None,
            "score": None, "busy_other": None, "last_error": "connection refused"})
        self.assertTrue(llm["backends"][0]["draining"])
        self.assertEqual((llm["backends"][0]["latency_ms"], llm["backends"][0]["score"]), (850, 0.0486))
        self.assertEqual(llm["backends"][1], {
            "name": "B", "healthy": None, "draining": None, "healing": None, "inflight": None, "latency_ms": None,
            "score": None, "busy_other": None, "last_error": None})
        # The last 5 of the 7 decisions, newest last.
        self.assertEqual([decision["kind"] for decision in result["last_decisions"]],
                         ["unhealthy", "failover", "switch", "heal", "route"])
        self.assertEqual(result["last_decisions"][1], {
            "time_utc": datetime.datetime.fromtimestamp(1794550003.1, datetime.timezone.utc).strftime("%H:%M:%S"),
            "pool": "forecast", "backend": "A", "kind": "failover", "reason": "B refused the connection"})
        json.dumps(result, allow_nan=False)

    def test_set_routing_policy(self):
        for policy in ("static", "round_robin", "least_loaded", "adaptive"):
            self.assertEqual(tools.set_routing_policy(policy), {"policy": policy, "node": "router"})
        self.assertEqual([request["body"] for request in self.seen("/router/policy")],
                         [{"policy": "static"}, {"policy": "round_robin"}, {"policy": "least_loaded"},
                          {"policy": "adaptive"}])

    def test_set_routing_policy_refuses_an_unknown_policy_without_calling(self):
        result = tools.set_routing_policy("fastest")
        self.assertIn("one of", result["error"])
        self.assertEqual(ROUTER.seen, [])


class ToolErrorsTest(AgentTestCase):
    """Whatever goes wrong, a tool gives back {"error": ...} and does not raise."""

    def test_an_http_error_carries_the_servers_message_and_the_node(self):
        result = tools.forecast_point(41.9, -87.9, init="2001-01-01T00")
        self.assertEqual(sorted(result), ["error", "node"])
        self.assertIn("HTTP 404", result["error"])
        self.assertIn("init time not held", result["error"])
        self.assertEqual(result["node"], "A")

    def test_an_answer_that_is_not_json(self):
        result = tools.forecast_point(41.9, -87.9, init="not-json")
        self.assertIn("did not answer with JSON", result["error"])

    def test_http_json_reads_empty_answers_and_other_error_shapes(self):
        base = os.environ["ROUTER_URL"]
        self.assertEqual(tools.http_json("GET", base + "/empty"), ({}, None, None))
        data, node, err = tools.http_json("GET", base + "/vllm-error")
        self.assertEqual((data, node), (None, None))
        self.assertEqual(err, "GET %s/vllm-error answered HTTP 404: The model `x` does not exist." % base)
        self.assertIn("HTTP 500: the model fell over", tools.http_json("GET", base + "/openai-error")[2])
        self.assertIn("HTTP 404: no such path", tools.http_json("GET", base + "/nowhere")[2])

    def test_router_down(self):
        os.environ["ROUTER_URL"], old = dead_address(), os.environ["ROUTER_URL"]
        try:
            results = [tools.forecast_point(41.9, -87.9), tools.ohare_tomorrow(), tools.ensemble_range(41.9, -87.9),
                       tools.storm_2019(), tools.cluster_status(), tools.set_routing_policy("static")]
        finally:
            os.environ["ROUTER_URL"] = old
        for result in results:
            self.assertEqual(list(result), ["error"])
            self.assertIn("cannot reach http://127.0.0.1:", result["error"])
            self.assertIn("Is the router running", result["error"])

    def test_a_router_that_never_answers(self):
        with socket.socket() as silent:                 # takes the connection, then says nothing
            silent.bind(("127.0.0.1", 0))
            silent.listen(1)
            url = "http://127.0.0.1:%d/router/status" % silent.getsockname()[1]
            self.assertEqual(tools.http_json("GET", url, timeout=0.3),
                             (None, None, "GET %s gave no answer within 0.3 s" % url))

    def test_a_bad_router_url(self):
        os.environ["ROUTER_URL"], old = "nodeA:8100", os.environ["ROUTER_URL"]
        try:
            result = tools.cluster_status()
        finally:
            os.environ["ROUTER_URL"] = old
        self.assertIn("cannot reach nodeA:8100", result["error"])

    def test_nonsense_arguments_do_not_raise(self):
        self.assertIn("error", tools.forecast_point("north", None, lead_hours=[1]))
        self.assertIn("error", tools.ensemble_range({}, set()))
        self.assertIn("error", tools.set_routing_policy(["adaptive"]))


# ---------------------------------------------------------------- agent/weather_mcp.py, over its stdin and stdout

class McpClient(object):
    """Runs agent/weather_mcp.py as a child process and talks to it the way an MCP client does."""

    def __init__(self, router_url=None):
        env = dict(os.environ, ROUTER_URL=router_url or os.environ["ROUTER_URL"])
        self.process = subprocess.Popen([sys.executable, MCP_SERVER], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.stdout_lines = queue.Queue()       # None marks the end of the stream
        self.stderr_lines = []
        self.next_id = 0
        self.threads = [threading.Thread(target=self._pump, args=(self.process.stdout, self.stdout_lines.put)),
                        threading.Thread(target=self._pump, args=(self.process.stderr, self.stderr_lines.append))]
        for thread in self.threads:
            thread.daemon = True
            thread.start()

    @staticmethod
    def _pump(stream, sink):
        for line in iter(stream.readline, b""):
            sink(line)
        sink(None)

    def send_line(self, text):
        self.process.stdin.write(text.encode("utf-8") + b"\n")
        self.process.stdin.flush()

    def read(self):
        """The next message from the server. Everything on stdout must be one JSON-RPC message per line."""
        try:
            line = self.stdout_lines.get(timeout=20)
        except queue.Empty:
            raise AssertionError("no answer from the server within 20 s. Its log: %r" % (self.stderr_lines,))
        if line is None:
            raise AssertionError("the server closed stdout. Its log: %r" % (self.stderr_lines,))
        message = json.loads(line.decode("utf-8"))
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise AssertionError("not a JSON-RPC message on stdout: %r" % line)
        return message

    def request(self, method, params=None):
        """Send a request and return its answer."""
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send_line(json.dumps(message))
        answer = self.read()
        if answer.get("id") != self.next_id:
            raise AssertionError("expected the answer to request %d, got %r" % (self.next_id, answer))
        return answer

    def call(self, name, arguments=None):
        """tools/call. Returns the result part after checking its shape."""
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        result = self.request("tools/call", params)["result"]
        if sorted(result) != ["content", "isError", "structuredContent"]:
            raise AssertionError("wrong keys in a tools/call result: %r" % result)
        if len(result["content"]) != 1 or result["content"][0]["type"] != "text":
            raise AssertionError("content must be one text block: %r" % result)
        if json.loads(result["content"][0]["text"]) != result["structuredContent"]:
            raise AssertionError("the text is not the structured content as JSON: %r" % result)
        return result

    def finish(self):
        """Close stdin, as a client does to stop the server. Returns (exit code, unread stdout lines)."""
        self.process.stdin.close()
        code = self.process.wait(timeout=20)
        for thread in self.threads:
            thread.join(5)
        left = []
        while not self.stdout_lines.empty():
            line = self.stdout_lines.get()
            if line is not None:
                left.append(line)
        return code, left

    def kill(self):
        """For cleanup: stop the child if a failed test left it running."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


class McpServerTest(AgentTestCase):

    def setUp(self):
        super(McpServerTest, self).setUp()
        self.client = McpClient()
        self.addCleanup(self.client.kill)

    def test_initialize_then_ping(self):
        answer = self.client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                    "clientInfo": {"name": "test", "version": "0"}})
        self.assertEqual(sorted(answer), ["id", "jsonrpc", "result"])
        result = answer["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["capabilities"], {"tools": {}})
        self.assertEqual(sorted(result["serverInfo"]), ["name", "version"])
        # A notification gets no answer: the next thing on stdout is the answer to the ping after it.
        self.client.send_line(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.client.request("ping"), {"jsonrpc": "2.0", "id": 2, "result": {}})
        code, left = self.client.finish()
        self.assertEqual((code, left), (0, []))

    def test_initialize_picks_the_version(self):
        for asked, given in [("2024-11-05", "2024-11-05"), ("2025-03-26", "2025-03-26"),
                             ("2025-11-25", "2025-11-25"), ("2026-07-28", "2025-11-25"),
                             ("1999-01-01", "2025-11-25"), (None, "2025-11-25")]:
            params = {"capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}
            if asked:
                params["protocolVersion"] = asked
            self.assertEqual(self.client.request("initialize", params)["result"]["protocolVersion"], given)

    def test_string_ids_come_back_unchanged(self):
        self.client.send_line(json.dumps({"jsonrpc": "2.0", "id": "abc-1", "method": "ping"}))
        self.assertEqual(self.client.read(), {"jsonrpc": "2.0", "id": "abc-1", "result": {}})

    def test_tools_list(self):
        listed = self.client.request("tools/list")["result"]["tools"]
        self.assertEqual([tool["name"] for tool in listed], TOOL_NAMES)
        for tool, ours in zip(listed, tools.TOOLS):
            self.assertEqual(sorted(tool), ["description", "inputSchema", "name"])
            self.assertEqual(tool["description"], ours["description"])
            self.assertEqual(tool["inputSchema"], ours["input_schema"])
            self.assertEqual(tool["inputSchema"]["type"], "object")

    def test_tools_call_for_each_tool(self):
        before = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=18)).date().isoformat()

        forecast = self.client.call("forecast_point", {"lat": 41.9602, "lon": -87.9316, "lead_hours": 12})
        self.assertFalse(forecast["isError"])
        self.assertEqual(forecast["structuredContent"]["node"], "A")
        self.assertEqual([row["lead_h"] for row in forecast["structuredContent"]["by_lead"]], [6, 12])

        ohare = self.client.call("ohare_tomorrow")
        self.assertFalse(ohare["isError"])
        after = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=18)).date().isoformat()
        content = ohare["structuredContent"]
        self.assertIn(content["date_local"], (before, after))      # tomorrow in Chicago: UTC-6, plus a day
        self.assertEqual((content["low"]["temp_c"], content["high"]["temp_c"]), (-2.6, 6.7))
        self.assertEqual(content["low"]["valid_time_local"], content["date_local"] + " 06:00")
        self.assertEqual(content["near_6am"]["temp_c"], -2.6)
        self.assertEqual(content["node"], "A")

        ensemble = self.client.call("ensemble_range", {"lat": 41.9602, "lon": -87.9316, "members": 8})
        self.assertFalse(ensemble["isError"])
        self.assertEqual([(shard["node"], shard["members"]) for shard in ensemble["structuredContent"]["shards"]],
                         [("A", 4), ("B", 4)])
        self.assertEqual(self.seen("/v1/ensemble")[0]["body"]["members"], 8)

        storm = self.client.call("storm_2019", {})
        self.assertFalse(storm["isError"])
        self.assertEqual(storm["structuredContent"]["error_at_72h_c"], 2.3)

        status = self.client.call("cluster_status")
        self.assertFalse(status["isError"])
        self.assertEqual(status["structuredContent"]["policy"], "adaptive")
        self.assertEqual(len(status["structuredContent"]["last_decisions"]), 5)

        policy = self.client.call("set_routing_policy", {"policy": "least_loaded"})
        self.assertFalse(policy["isError"])
        self.assertEqual(policy["structuredContent"], {"policy": "least_loaded", "node": "router"})
        self.assertEqual(self.seen("/router/policy")[0]["body"], {"policy": "least_loaded"})

        code, left = self.client.finish()
        self.assertEqual((code, left), (0, []))
        # The log went to stderr, one line per call.
        log = b"".join(line for line in self.client.stderr_lines if line).decode("utf-8")
        for name in TOOL_NAMES:
            self.assertIn("weather_mcp: %s ok" % name, log)

    def test_a_failed_tool_is_a_result_with_is_error(self):
        refused = self.client.call("forecast_point", {"lat": "north", "lon": 3})
        self.assertTrue(refused["isError"])
        self.assertIn("'lat' must be a number", refused["structuredContent"]["error"])
        not_held = self.client.call("forecast_point", {"lat": 41.9, "lon": -87.9, "init": "2001-01-01T00"})
        self.assertTrue(not_held["isError"])
        self.assertIn("init time not held", not_held["structuredContent"]["error"])

    def test_protocol_errors(self):
        unknown = self.client.request("resources/list")
        self.assertEqual(unknown, {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601,
                                                                       "message": "Method not found: resources/list"}})
        # The 2026-07-28 revision's probe. "Method not found" tells a newer client to fall back to initialize.
        discover = self.client.request("server/discover",
                                       {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}})
        self.assertEqual(discover["error"]["code"], -32601)

        self.client.send_line("this is not JSON")
        bad_line = self.client.read()
        self.assertEqual(bad_line["error"]["code"], -32700)
        self.assertIsNone(bad_line.get("id"))
        self.client.send_line('{"jsonrpc": "2.0", "id": 9, "method": "ping"')       # cut short
        self.assertEqual(self.client.read()["error"]["code"], -32700)
        self.client.process.stdin.write(b'{"jsonrpc": "2.0", "id": 9, "method": "\xff"}\n')   # not UTF-8
        self.client.process.stdin.flush()
        self.assertEqual(self.client.read()["error"]["code"], -32700)

        self.client.send_line("[1, 2]")
        self.assertEqual(self.client.read()["error"]["code"], -32600)
        self.client.send_line('{"jsonrpc": "2.0", "id": 9}')
        self.assertEqual(self.client.read()["error"]["code"], -32600)

        self.assertEqual(self.client.request("tools/call", {"name": "make_it_rain", "arguments": {}})["error"],
                         {"code": -32602, "message": "Unknown tool: make_it_rain"})
        self.assertEqual(self.client.request("tools/call", {"name": "cluster_status", "arguments": [1]})
                         ["error"]["code"], -32602)
        self.assertEqual(self.client.request("tools/call", ["cluster_status"])["error"]["code"], -32602)

        # Blank lines, an unknown notification and a stray answer are passed over. The server is still up.
        self.client.send_line("")
        self.client.send_line(json.dumps({"jsonrpc": "2.0", "method": "notifications/whatever"}))
        self.client.send_line(json.dumps({"jsonrpc": "2.0", "id": 77, "result": {}}))
        self.assertEqual(self.client.request("ping")["result"], {})
        code, left = self.client.finish()
        self.assertEqual((code, left), (0, []))

    def test_sigterm_and_sigint_exit_with_code_0(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            client = McpClient()
            self.addCleanup(client.kill)
            self.assertEqual(client.request("ping")["result"], {})      # it is up and waiting on stdin
            client.process.send_signal(signum)
            self.assertEqual(client.process.wait(timeout=20), 0)

    def test_sigterm_in_the_middle_of_a_tool_call_exits_with_code_0(self):
        with socket.socket() as silent:                 # a "router" that takes the connection and never answers
            silent.bind(("127.0.0.1", 0))
            silent.listen(1)
            silent.settimeout(20)
            client = McpClient("http://127.0.0.1:%d" % silent.getsockname()[1])
            self.addCleanup(client.kill)
            client.send_line(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": "cluster_status"}}))
            connection, _address = silent.accept()      # the tool is now waiting for the router
            with connection:
                client.process.send_signal(signal.SIGTERM)
                self.assertEqual(client.process.wait(timeout=20), 0)


# ---------------------------------------------------------------- the number guard in agent/desk_agent.py

class NumberGuardTest(unittest.TestCase):

    RESULTS = [{"place": "Chicago O'Hare", "lat": 41.9602, "lon": -87.9316, "date_local": "2026-11-15",
                "local_time": "UTC-6, a fixed offset (Chicago standard time, right for November)",
                "init_time_utc": "2026-11-14T06",
                "low": {"temp_c": -2.6, "valid_time_utc": "2026-11-15T12", "valid_time_local": "2026-11-15 06:00"},
                "high": {"temp_c": 6.7, "valid_time_utc": "2026-11-15T18", "valid_time_local": "2026-11-15 12:00"},
                "near_6am": {"temp_c": -2.6, "wind_mps": 4.3, "pressure_hpa": 1013.2}, "ok": True, "node": "A"}]

    def guard(self, answer, question="What will tomorrow morning be like at O'Hare?", results=None):
        return desk_agent.guard_answer(answer, self.RESULTS if results is None else results, question)

    def assert_kept(self, answer, **more):
        self.assertEqual(self.guard(answer, **more), (answer, []))

    def assert_dropped(self, answer, **more):
        self.assertEqual(self.guard(answer, **more), ("", [answer]))

    def test_supported_numbers_are_kept(self):
        self.assert_kept("Tomorrow morning will be cold. The low is -2.6 C and the high is 6.7 C, with a 4.3 m/s wind.")
        self.assert_kept("No numbers here at all.")
        self.assert_kept("The low is \u22122.6 \u00b0C (that is -2.6\u00b0C). Pressure is 1,013.2 hPa.")

    def test_a_sentence_with_an_invented_number_is_dropped(self):
        text, dropped = self.guard("The low is -2.6 C. Expect 15.5 C by Sunday! The high is 6.7 C.")
        self.assertEqual(text, "The low is -2.6 C. The high is 6.7 C.")
        self.assertEqual(dropped, ["Expect 15.5 C by Sunday!"])
        # One bad number is enough, however many good ones stand next to it.
        self.assert_dropped("The low is -2.6 C, the high 6.7 C and the record 31.1 C.")
        self.assert_dropped("Humidity is 80%.")

    def test_everything_dropped_leaves_nothing(self):
        self.assertEqual(self.guard("It is 99 C. Or 98 C."), ("", ["It is 99 C.", "Or 98 C."]))
        self.assertEqual(self.guard("It is 6.7 C.", results=[]), ("", ["It is 6.7 C."]))

    def test_tolerance_is_0_05(self):
        self.assertEqual(desk_agent.TOLERANCE, 0.05)
        self.assert_kept("The high is 6.74 C.")
        self.assert_kept("The high is 6.75 C.")                 # exactly 0.05 away
        self.assert_kept("The high is 6.65 C.")
        self.assert_dropped("The high is 6.76 C.")
        self.assert_dropped("The high is 6.64 C.")
        self.assert_dropped("The high is 7 C.")                 # rounded too far
        self.assert_kept("The low is -2.63 C.")
        self.assert_dropped("The low is -2.7 C.")
        self.assert_kept("Latitude 41.96, longitude -87.93.")

    def test_numbers_from_the_question_are_not_checked(self):
        question = "What about 36 hours from now at 40.7, -74.0?"
        self.assert_kept("In 36 hours at 40.7, -74.0 it will be -2.6 C.", question=question)
        self.assert_dropped("In 37 hours it will be -2.6 C.", question=question)

    def test_four_digit_years_are_not_checked(self):
        self.assert_kept("The January 2019 cold wave was worse than 1985.")
        self.assert_dropped("It was 2019.5 units.")
        self.assert_dropped("It reached 3000 m.")               # four digits, but not a year

    def test_times_of_day_are_not_checked(self):
        self.assert_kept("The low comes around 5 AM.")
        self.assert_kept("The low comes at 5am, 5 a.m. or 5:30 p.m. local time.")
        self.assert_kept("The low comes at 05:45, which is 11:45 UTC.")
        self.assert_kept("The run starts at 21 UTC (21Z).")
        self.assert_dropped("The low comes after 5 hours.")
        self.assert_dropped("The low is 5 amps.")

    def test_list_numbering_is_not_checked(self):
        answer = "Tomorrow:\n1. Low: -2.6 C\n2. High: 6.7 C\n3) Wind: 4.3 m/s\n(4) Fine.\n**5.** Done."
        self.assert_kept(answer)
        text, dropped = self.guard("1. Low: -2.6 C\n2. High: 9.9 C\n3. Wind: 4.3 m/s")
        self.assertEqual(text, "1. Low: -2.6 C\n3. Wind: 4.3 m/s")
        self.assertEqual(dropped, ["2. High: 9.9 C"])
        self.assert_dropped("7.")                               # a bare number is not a list marker

    def test_each_line_is_checked_on_its_own(self):
        answer = ("Tomorrow at O'Hare:\n\n- Low: **-2.6\u00b0C** at 6 AM\n- High: **8.8\u00b0C**\n"
                  "- Wind: 4.3 m/s\n\nDress warmly.")
        text, dropped = self.guard(answer)
        self.assertEqual(text, "Tomorrow at O'Hare:\n\n- Low: **-2.6\u00b0C** at 6 AM\n- Wind: 4.3 m/s\n\n"
                               "Dress warmly.")
        self.assertEqual(dropped, ["- High: **8.8\u00b0C**"])
        text, dropped = self.guard("The low is -2.6 C. It hit 50 C once.\nThe high is 6.7 C.")
        self.assertEqual(text, "The low is -2.6 C.\nThe high is 6.7 C.")

    def test_minus_signs(self):
        self.assert_kept("It will be 2.6 degrees below zero, near 87.93 W.")    # no sign: either sign matches
        self.assert_dropped("The high is -6.7 C.")                               # an invented minus
        self.assert_dropped("The high is \u22126.7 C.")
        self.assert_kept("Between 4.3-6.7 units, or 4.3\u20136.7.")              # a dash, not a minus
        self.assert_kept("The offset is UTC-6.")
        self.assert_kept("Low:-2.6, high:6.7.")

    def test_digits_inside_names_are_not_numbers(self):
        self.assert_kept("The t2m low from storm_2019 on a GB10 is -2.6 C.")
        self.assert_dropped("The 2 m temperature is -2.6 C.")

    def test_numbers_in_tool_text_count_but_key_names_do_not(self):
        self.assert_kept("On 2026-11-15, that is November 15, the low is at 12 UTC. The run is from the 14th.")
        results = [{"t2m_c": -2.6, "wind10_ms": None, "model": "afno-19m", "reason": "B beat A by 31% for 5.0 s"}]
        self.assert_kept("The afno-19m model says -2.6 C. B beat A by 31% for 5 s.", results=results)
        self.assert_dropped("The wind is 10 m/s.", results=results)
        self.assert_dropped("It is 2 C.", results=results)
        self.assert_dropped("True is not 1.", results=[{"ok": True}])

    def test_find_numbers(self):
        def values(text, prose=True):
            return desk_agent.signed_numbers(text, prose)
        self.assertEqual(values("-2.6 and \u22123 and 4-5 and 1,013.25 and +7"), [-2.6, -3.0, 4.0, 5.0, 1013.25, 7.0])
        self.assertEqual(values("2019-01-27T12", prose=False), [2019.0, 1.0, 27.0, 12.0])
        self.assertEqual(values("2019-01-27T12"), [2019.0, 1.0, 27.0])
        self.assertEqual(values("UTC-6", prose=False), [-6.0])
        self.assertEqual(values("UTC-6"), [6.0])
        self.assertEqual(values("v3.12.15 t2m GB10 6,12,18"), [6.0, 12.0, 18.0])
        self.assertEqual(desk_agent.tool_numbers({"a1": [1, 2.5, "x 3", None, True, {"b": -4}]}), [1.0, 2.5, 3.0, -4.0])


# ---------------------------------------------------------------- the chat loop in agent/desk_agent.py

class DeskAgentTest(AgentTestCase):

    def run_quietly(self, question, model, session, **more):
        """desk_agent.run() against the fake router, with its progress notes kept off the screen."""
        with contextlib.redirect_stderr(io.StringIO()):
            return desk_agent.run(question, os.environ["ROUTER_URL"] + "/v1", model, session, **more)

    def test_the_whole_run_as_a_command(self):
        """A scripted model calls storm_2019, then answers with one true number and one invented one."""
        done = subprocess.run([sys.executable, DESK_AGENT, "How good was the forecast at +72 hours?",
                               "--session", "test-session"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        out, err = done.stdout.decode("utf-8"), done.stderr.decode("utf-8")
        self.assertEqual(done.returncode, 0, err)
        lines = out.splitlines()
        # The true sentence stays. The invented -45.6 goes, and so does the <think> part.
        self.assertEqual(lines[0], "At +72 hours the forecast was -27.9 C.")
        self.assertIn("Dropped 1 sentence(s) with numbers no tool gave.", lines)
        self.assertNotIn("45.6", out)
        self.assertNotIn("think", out)
        facts = out[out.index("Facts from tools"):out.index("Nodes")]
        self.assertIn("storm_2019()", facts)
        self.assertIn("init_time_utc: 2019-01-27T12", facts)
        self.assertIn("lead_h=72, valid_time_utc=2019-01-30T12, forecast_temp_c=-27.9, truth_temp_c=-30.2", facts)
        self.assertIn("error_at_72h_c: 2.3", facts)
        self.assertIn("LLM calls: B, B", out)
        self.assertIn("Tools: storm_2019 on A", out)

        # What the model endpoint saw: the model name it listed, the same session id on every call,
        # the tool list, and then the tool's result as a "tool" message.
        self.assertEqual(self.seen("/v1/models")[0]["session"], "test-session")
        first, second = self.seen("/v1/chat/completions")
        self.assertEqual((first["session"], second["session"]), ("test-session", "test-session"))
        self.assertEqual((first["body"]["model"], first["body"]["stream"]), ("scripted", False))
        self.assertEqual([message["role"] for message in first["body"]["messages"]], ["system", "user"])
        self.assertEqual(first["body"]["messages"][1]["content"], "How good was the forecast at +72 hours?")
        offered = first["body"]["tools"]
        self.assertEqual([tool["function"]["name"] for tool in offered], TOOL_NAMES)
        for tool, ours in zip(offered, tools.TOOLS):
            self.assertEqual(tool, {"type": "function", "function": {
                "name": ours["name"], "description": ours["description"], "parameters": ours["input_schema"]}})
        assistant, tool_message = second["body"]["messages"][2:]
        self.assertEqual(assistant, {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "storm_2019", "arguments": "{}"}}]})
        self.assertEqual((tool_message["role"], tool_message["tool_call_id"]), ("tool", "call_1"))
        self.assertEqual(json.loads(tool_message["content"])["error_at_72h_c"], 2.3)
        self.assertEqual(len(self.seen("/v1/forecast")), 1)

    def test_stops_after_max_rounds(self):
        outcome = self.run_quietly("Is the cluster fine?", "always-tools", "s1", max_rounds=3)
        self.assertIsNone(outcome["answer"])
        self.assertEqual(len(self.seen("/v1/chat/completions")), 3)
        self.assertEqual([call["name"] for call in outcome["calls"]], ["cluster_status"] * 3)
        self.assertEqual(outcome["llm_nodes"], ["B", "B", "B"])
        text = desk_agent.report(outcome, "Is the cluster fine?", 3)
        self.assertIn("no answer", text.splitlines()[0])
        self.assertIn("Dropped 0 sentence(s) with numbers no tool gave.", text)
        self.assertIn("policy: adaptive", text)
        lines = text.splitlines()
        self.assertIn("          name=A, healthy=true, draining=false, healing=false, inflight=1, latency_ms=210, "
                      "score=0.42, busy_other=false", lines)
        self.assertIn("          name=B, healthy=false, draining=false, healing=true, inflight=0, "
                      "last_error=connection refused", lines)         # nulls are left out
        self.assertIn("counters: routed=100, failovers=1, switches=2, heals=0, errors=0", text)
        self.assertIn("Tools: cluster_status on router, cluster_status on router, cluster_status on router", text)

    def test_an_answer_with_no_tool_call(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = desk_agent.main(["Will I need a coat?", "--model", "no-tools", "--session", "s2"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertEqual(text.splitlines()[0], "Bring a coat.")            # "It will be 21.5 C tomorrow." is gone
        self.assertIn("Dropped 1 sentence(s) with numbers no tool gave.", text)
        self.assertIn("none: the model called no tool", text)
        self.assertIn("LLM calls: B", text)
        self.assertIn("Tools: none", text)
        self.assertEqual(self.seen("/v1/models"), [])                      # --model was given: no need to ask

    def test_arguments_that_are_not_json_go_back_to_the_model_as_an_error(self):
        outcome = self.run_quietly("Forecast?", "bad-arguments", "s3")
        self.assertEqual(outcome["answer"], "The tool did not work.")
        self.assertIn("not valid JSON", outcome["calls"][0]["result"]["error"])
        tool_message = self.seen("/v1/chat/completions")[1]["body"]["messages"][-1]
        self.assertIn("not valid JSON", json.loads(tool_message["content"])["error"])
        self.assertEqual(self.seen("/v1/forecast"), [])
        self.assertIn("error: forecast_point: the arguments were not valid JSON",
                      desk_agent.report(outcome, "Forecast?", 6))

    def test_model_endpoint_down(self):
        dead = dead_address() + "/v1"
        with self.assertRaises(desk_agent.AgentError):
            desk_agent.run("Anyone there?", dead, "scripted", "s4")     # it fails before it has anything to note
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = desk_agent.main(["Anyone there?", "--base-url", dead])
        self.assertEqual((code, out.getvalue()), (1, ""))
        self.assertIn("the model call failed: cannot reach", err.getvalue())

    def test_arguments_sent_as_an_object_and_a_call_with_no_id(self):
        outcome = self.run_quietly("Make it static.", "object-arguments", "s5")
        self.assertEqual(outcome["answer"], "Done.")
        self.assertEqual(outcome["calls"], [{"name": "set_routing_policy", "arguments": {"policy": "static"},
                                             "result": {"policy": "static", "node": "router"}}])
        # Sent back in the standard shape: the arguments as JSON text, and an id the tool message repeats.
        assistant, tool_message = self.seen("/v1/chat/completions")[1]["body"]["messages"][2:]
        self.assertEqual(assistant["tool_calls"], [{"id": "call_1", "type": "function", "function": {
            "name": "set_routing_policy", "arguments": '{"policy": "static"}'}}])
        self.assertEqual(tool_message["tool_call_id"], "call_1")

    def test_text_of(self):
        self.assertEqual(desk_agent.text_of({"content": "  Hello.  "}), "Hello.")
        self.assertEqual(desk_agent.text_of({"content": None}), "")
        self.assertEqual(desk_agent.text_of({}), "")
        self.assertEqual(desk_agent.text_of({"content": [{"type": "text", "text": "Hello"},
                                                         {"type": "text", "text": " there."}]}), "Hello there.")
        self.assertEqual(desk_agent.text_of({"content": "<think>Hm.\nSay 5?</think>\nIt is cold."}), "It is cold.")

    def test_facts_for_every_tool(self):
        self.set_clock(2026, 11, 14, 15, 0)
        ROUTER.latest = "2026-11-14T06"
        calls = [{"name": name, "arguments": arguments, "result": tools.call_tool(name, arguments)}
                 for name, arguments in [("forecast_point", dict(OHARE, lead_hours=12)), ("ohare_tomorrow", {}),
                                         ("ensemble_range", dict(OHARE, lead_hours=12)), ("storm_2019", {}),
                                         ("cluster_status", {}), ("set_routing_policy", {"policy": "static"})]]
        text = desk_agent.report({"answer": "Fine.", "calls": calls, "llm_nodes": ["A", None]}, "How is it?", 6)
        for expected in [
                "Fine.",
                "Dropped 0 sentence(s) with numbers no tool gave.",
                '  forecast_point({"lat": 41.9602, "lon": -87.9316, "lead_hours": 12})',
                "    last_lead: lead_h=12, valid_time_utc=2026-11-14T18, temp_c=6.7, wind_mps=4.4, pressure_hpa=1012.8",
                "  ohare_tomorrow()",
                "    low: temp_c=-2.6, valid_time_utc=2026-11-15T12, valid_time_local=2026-11-15 06:00",
                "    near_6am: temp_c=-2.6, wind_mps=4.7, valid_time_utc=2026-11-15T12, "
                "valid_time_local=2026-11-15 06:00",
                "    last_lead: lead_h=12, valid_time_utc=2026-11-14T18, temp_mean_c=6.7, temp_min_c=6.0, "
                "temp_max_c=7.5",
                "      node=A, members=10, ms=1305, retried=false",
                "    error_at_72h_c: 2.3",
                "        preferred: A",
                "    policy: static",
                "  LLM calls: A, unknown",
                "  Tools: forecast_point on A, ohare_tomorrow on A, ensemble_range on A+B, storm_2019 on A, "
                "cluster_status on router, set_routing_policy on router"]:
            self.assertIn(expected, text.splitlines())


if __name__ == "__main__":
    unittest.main()
