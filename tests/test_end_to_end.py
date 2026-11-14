"""End-to-end test: every piece working together on one machine.

It starts the local cluster (two forecast mocks, two LLM mocks, the router) with
tools/dev_cluster.py on ports 19100 to 19112, drives it the way a person would, and
always stops it again. Nothing here needs a GPU, PyTorch or NumPy.

Run it from the repo root (about a minute):

    .venv/bin/python -m unittest tests/test_end_to_end.py -v
    .venv/bin/python -m unittest tests.test_end_to_end.EndToEnd.test_ensemble_is_split_and_merged -v

What it covers: a forecast and an ensemble through the router, chat with and without a
stream, session affinity, the four policies, draining, a slowed node (the adaptive policy
moves once and stays), a dead node (failover, heal, recovery), a node that answers 500,
the status page's shape, the dashboard page, the MCP server, the desk agent and a short
run of the load generator.

If ports 19100 to 19112 are taken, the test fails at the start and says so. The pieces' logs
stay in .dev_cluster_logs/19100/ for a look after a failure (git ignores that folder).

Written at the hackathon (Nov 13-14, 2026).
"""

import atexit
import datetime
import http.client
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from urllib.parse import urlsplit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_CLUSTER = os.path.join(REPO, "tools", "dev_cluster.py")
BASE = 19100
ROUTER = "http://127.0.0.1:%d" % BASE
FORECAST = {"A": "http://127.0.0.1:%d" % (BASE + 1), "B": "http://127.0.0.1:%d" % (BASE + 2)}
LLM = {"A": "http://127.0.0.1:%d" % (BASE + 11), "B": "http://127.0.0.1:%d" % (BASE + 12)}
OHARE = {"lat": 41.9602, "lon": -87.9316}
POLICIES = ("static", "round_robin", "least_loaded", "adaptive")

BACKEND_KEYS = {"name", "url", "probe", "healthy", "draining", "healing", "inflight", "latency_ewma_ms",
                "score", "routed", "failures", "last_error", "probe_data"}
DECISION_KEYS = {"ts", "pool", "backend", "policy", "kind", "reason", "retries"}
DECISION_KINDS = {"route", "failover", "switch", "heal", "unhealthy", "recovered", "drain", "policy", "fanout"}
TOOL_NAMES = ["forecast_point", "ohare_tomorrow", "ensemble_range", "storm_2019", "cluster_status",
              "set_routing_policy"]


def call(method, url, body=None, headers=None, timeout=30.0):
    """One HTTP request. Returns (status, headers with lower-case names, body).

    The body comes back parsed if it is JSON, as bytes if not.
    """
    parts = urlsplit(url)
    send = {"Connection": "close"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        send["Content-Type"] = "application/json"
    send.update(headers or {})
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        conn.request(method, parts.path + ("?" + parts.query if parts.query else ""), body=data, headers=send)
        resp = conn.getresponse()
        raw = resp.read()
        got = dict((name.lower(), value) for name, value in resp.getheaders())
    finally:
        conn.close()
    try:
        return resp.status, got, json.loads(raw.decode("utf-8"))
    except ValueError:
        return resp.status, got, raw


def dev_cluster(command):
    """Run tools/dev_cluster.py up or down for this test's ports. Returns (exit code, what it printed)."""
    done = subprocess.run([sys.executable, DEV_CLUSTER, command, "--ports-from", str(BASE)], cwd=REPO,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    return done.returncode, done.stdout.decode("utf-8", "replace")


STARTED = []        # not empty while this test's cluster may be running


def stop_cluster():
    """Stop this test's cluster, if it was started and not stopped yet. Safe to call twice."""
    if STARTED:
        del STARTED[:]
        dev_cluster("down")


atexit.register(stop_cluster)       # the last resort, for a run that is cut short with Ctrl-C


class EndToEnd(unittest.TestCase):
    """One cluster for the whole class. Each test starts from the same clean state (see setUp)."""

    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(stop_cluster)            # runs whatever happens below, even a failed start
        STARTED.append(True)
        code, printed = dev_cluster("up")
        if code != 0:
            raise AssertionError("the cluster did not start on ports %d to %d (are they taken?):\n%s"
                                 % (BASE, BASE + 12, printed))

    def setUp(self):
        """Back to normal: hooks cleared, nothing drained, policy adaptive, every backend healthy."""
        for url in list(FORECAST.values()) + list(LLM.values()):
            self.assertEqual(call("POST", url + "/admin/reset", {})[0], 200, url)
        for name in "AB":
            self.assertEqual(call("POST", ROUTER + "/router/drain", {"backend": name, "drain": False})[0], 200)
        self.policy("adaptive")
        self.wait_for(self.all_calm, "the router to see every backend healthy and idle again", 20.0)

    # ---- helpers

    def wait_for(self, check, what, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = check()
            if value:
                return value
            time.sleep(0.1)
        self.fail("timed out after %g s waiting for %s" % (timeout, what))

    def status(self):
        code, _, body = call("GET", ROUTER + "/router/status")
        self.assertEqual(code, 200)
        return body

    def row(self, pool, name):
        """One backend's entry in /router/status."""
        return [b for b in self.status()["pools"][pool]["backends"] if b["name"] == name][0]

    def all_calm(self):
        """Every backend healthy and in service, and the last probes show the forecast servers idle
        and undisturbed. The probes come once a second, so after a reset this takes a moment."""
        pools = self.status()["pools"]
        for pool in pools.values():
            for b in pool["backends"]:
                if not b["healthy"] or b["draining"] or b["inflight"]:
                    return False
        for b in pools["forecast"]["backends"]:
            seen = b["probe_data"]
            if seen.get("busy_other") is not False or seen.get("queued_units") != 0 or seen.get("slowdown") != 1.0:
                return False
        return True

    def policy(self, name):
        code, _, body = call("POST", ROUTER + "/router/policy", {"policy": name})
        self.assertEqual((code, body), (200, {"policy": name}))

    def forecast(self, **more):
        """A small forecast through the router. Returns (status, headers, body)."""
        body = {"init": "latest", "lead_hours": 24, "members": 1, "point": OHARE}
        body.update(more)
        return call("POST", ROUTER + "/v1/forecast", body)

    def backends_of(self, count, **more):
        """Send `count` forecasts one after another; they must all work. Returns who answered."""
        seen = []
        for _ in range(count):
            code, headers, body = self.forecast(**more)
            self.assertEqual(code, 200, body)
            seen.append(headers["x-backend"])
        return seen

    def chat(self, session=None, **more):
        body = {"model": "mock-llm", "max_tokens": 8, "messages": [{"role": "user", "content": "Will it snow?"}]}
        body.update(more)
        headers = {"X-Session-Id": session} if session else None
        return call("POST", ROUTER + "/v1/chat/completions", body, headers)

    # ---- forecasts

    def test_forecast_goes_through_the_router(self):
        code, headers, body = self.forecast(request_id="e2e-1")
        self.assertEqual(code, 200, body)
        self.assertIn(headers["x-backend"], ("A", "B"))
        self.assertEqual((headers["x-router-policy"], headers["x-router-retries"]), ("adaptive", "0"))
        self.assertNotIn("x-router-affinity", headers)           # only the llm pool has affinity
        self.assertEqual(headers["access-control-allow-origin"], "*")
        self.assertEqual((body["request_id"], body["node"]), ("e2e-1", headers["x-backend"]))
        self.assertEqual((body["leads_h"], body["units"], body["members"]), ([6, 12, 18, 24], 4, 1))
        self.assertEqual(len(body["valid_times"]), 4)
        for key in ("t2m_c", "t2m_c_mean", "t2m_c_min", "t2m_c_max", "wind10_ms_mean", "msl_hpa_mean"):
            self.assertEqual(len(body["point"][key]), 4, key)
            self.assertTrue(all(isinstance(x, float) for x in body["point"][key]), key)
        self.assertEqual(sorted(body["timing_ms"]), ["gpu", "queue", "total"])
        # A request the server refuses comes back as it is, with the reason.
        code, headers, body = self.forecast(lead_hours=7)
        self.assertEqual(code, 400)
        self.assertIn("multiple of 6", body["error"])
        # The other read-only path of the pool.
        code, headers, body = call("GET", ROUTER + "/v1/info")
        self.assertEqual((code, body["mock"], body["node"]), (200, True, headers["x-backend"]))

    def test_ensemble_is_split_and_merged(self):
        ask = {"init": "latest", "lead_hours": 48, "members": 20, "point": OHARE, "with_truth": True}
        started = time.monotonic()
        code, headers, merged = call("POST", ROUTER + "/v1/ensemble", ask)
        split_s = time.monotonic() - started
        self.assertEqual(code, 200, merged)
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("A+B", "0"))
        shards = merged["shards"]
        self.assertEqual([(s["backend"], s["members"], s["member_offset"], s["retried"]) for s in shards],
                         [("A", 10, 0, False), ("B", 10, 10, False)])
        for shard in shards:
            self.assertEqual(sorted(shard), ["backend", "member_offset", "members", "ms", "retried"])
            self.assertGreater(shard["ms"], 0)
        # The same 20 members from one node alone, with no router in between.
        started = time.monotonic()
        code, _, whole = call("POST", FORECAST["A"] + "/v1/forecast", ask)
        whole_s = time.monotonic() - started
        self.assertEqual(code, 200, whole)
        for key in ("init_time", "members", "member_offset", "leads_h", "valid_times", "units", "model"):
            self.assertEqual(merged[key], whole[key], key)
        self.assertEqual((merged["node"], merged["units"]), ("A+B", 160))
        for key in ("t2m_c", "t2m_c_min", "t2m_c_max", "truth_t2m_c"):       # exact
            self.assertEqual(merged["point"][key], whole["point"][key], key)
        for key in ("t2m_c_mean", "wind10_ms_mean", "msl_hpa_mean"):          # each shard rounded its mean
            self.assertEqual(len(merged["point"][key]), 8, key)
            for ours, theirs in zip(merged["point"][key], whole["point"][key]):
                self.assertAlmostEqual(ours, theirs, delta=0.011, msg=key)
        self.assertLess(split_s, whole_s)              # two nodes at once beat one (0.8 s against 1.6 s)
        self.assertEqual([b["inflight"] for b in self.status()["pools"]["forecast"]["backends"]], [0, 0])

    def test_a_new_analysis_reaches_both_nodes(self):
        """POST /v1/state through the router goes to every forecast server, so "latest" is the same
        everywhere. If the nodes do differ, the router refuses to merge their shards."""
        def later(hours):
            code, _, info = call("GET", FORECAST["A"] + "/v1/info")
            self.assertEqual(code, 200)
            latest = datetime.datetime.strptime(info["latest"], "%Y-%m-%dT%H")
            return (latest + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H")

        def state(when):
            return {"time": when, "npy_b64": "bm90IGEgcmVhbCBzdGF0ZQ=="}        # the mocks do not decode it

        ask = {"init": "latest", "lead_hours": 12, "members": 4, "point": OHARE}
        newer, newest = later(6), later(12)
        self.addCleanup(call, "POST", ROUTER + "/v1/state", state(newest))      # leave both nodes alike
        code, headers, answer = call("POST", ROUTER + "/v1/state", state(newer))
        self.assertEqual((code, headers["x-backend"]), (200, "A+B"), answer)
        self.assertEqual((answer["latest"], answer["left_out"]), (newer, []))
        self.assertEqual([(row["backend"], row["status"], row["latest"]) for row in answer["backends"]],
                         [("A", 200, newer), ("B", 200, newer)])
        for name in "AB":
            self.assertEqual(call("GET", FORECAST[name] + "/v1/info")[2]["latest"], newer)
        code, _, merged = call("POST", ROUTER + "/v1/ensemble", ask)
        self.assertEqual((code, merged["init_time"], len(merged["shards"])), (200, newer, 2))
        # A still newer analysis goes to node B alone, straight to its server and not through the router.
        self.assertEqual(call("POST", FORECAST["B"] + "/v1/state", state(newest))[0], 200)
        code, _, answer = call("POST", ROUTER + "/v1/ensemble", ask)
        self.assertEqual(code, 502)
        self.assertIn("different analyses (A from %s, B from %s)" % (newer, newest), answer["error"])
        self.assertNotIn("point", answer)
        # Sent through the router, both have it and the ensemble works again.
        self.assertEqual(call("POST", ROUTER + "/v1/state", state(newest))[0], 200)
        code, _, merged = call("POST", ROUTER + "/v1/ensemble", ask)
        self.assertEqual((code, merged["init_time"]), (200, newest))

    # ---- chat

    def test_chat_plain_and_streamed(self):
        code, headers, body = self.chat()
        self.assertEqual(code, 200, body)
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertTrue(body["choices"][0]["message"]["content"])
        self.assertIn(headers["x-backend"], ("A", "B"))
        self.assertEqual((headers["x-router-policy"], headers["x-router-retries"], headers["x-router-affinity"]),
                         ("adaptive", "0", "none"))

        # A stream must arrive piece by piece, not in one lump at the end.
        ask = {"model": "mock-llm", "max_tokens": 32, "stream": True,
               "messages": [{"role": "user", "content": "Tell me about the wind."}]}
        conn = http.client.HTTPConnection("127.0.0.1", BASE, timeout=30)
        try:
            started = time.monotonic()
            conn.request("POST", "/v1/chat/completions", body=json.dumps(ask).encode("utf-8"),
                         headers={"Content-Type": "application/json", "X-Session-Id": "e2e-stream"})
            resp = conn.getresponse()
            got = dict((name.lower(), value) for name, value in resp.getheaders())
            arrivals, text = [], b""
            while True:
                piece = resp.read1(65536)
                if not piece:
                    break
                arrivals.append(time.monotonic() - started)
                text += piece
        finally:
            conn.close()
        self.assertEqual((resp.status, got["content-type"]), (200, "text/event-stream"))
        self.assertEqual((got["x-router-affinity"], got["x-router-retries"]), ("hit", "0"))
        self.assertIn(got["x-backend"], ("A", "B"))
        events = [line for line in text.decode("utf-8").split("\n\n") if line.strip()]
        self.assertEqual(events[-1], "data: [DONE]")
        said = "".join(json.loads(event[6:])["choices"][0]["delta"].get("content") or "" for event in events[:-1])
        self.assertEqual(len(said), 32 * 4)            # the mock writes 4 characters per token
        self.assertGreaterEqual(len(arrivals), 5)      # 32 tokens at 40 a second: about 0.8 s, in pieces
        self.assertGreater(arrivals[-1] - arrivals[0], 0.4)

    def test_a_session_sticks_to_one_backend(self):
        homes = {}
        for session in ("alpha", "bravo", "charlie", "delta"):
            seen = set()
            for _ in range(4):
                code, headers, body = self.chat(session)
                self.assertEqual((code, headers["x-router-affinity"]), (200, "hit"), body)
                seen.add(headers["x-backend"])
            self.assertEqual(len(seen), 1, "session %s moved between backends: %s" % (session, seen))
            homes[session] = seen.pop()
        self.assertEqual(set(homes.values()), {"A", "B"}, homes)     # the sessions spread over both nodes
        # The "user" field does the same job as the header.
        seen = set()
        for _ in range(3):
            code, headers, body = self.chat(user="charlie")
            self.assertEqual((code, headers["x-router-affinity"]), (200, "hit"))
            seen.add(headers["x-backend"])
        self.assertEqual(seen, {homes["charlie"]})

    # ---- the router's own controls

    def test_policies_can_be_switched(self):
        for name in POLICIES:
            self.policy(name)
            self.assertEqual(self.status()["policy"], name)
            code, headers, body = self.forecast()
            self.assertEqual((code, headers["x-router-policy"]), (200, name))
        self.policy("static")
        self.assertEqual(self.backends_of(4), ["A"] * 4)
        self.policy("round_robin")
        self.assertEqual(sorted(self.backends_of(4)), ["A", "A", "B", "B"])
        code, _, body = call("POST", ROUTER + "/router/policy", {"policy": "fastest"})
        self.assertEqual(code, 400)
        self.assertIn("unknown policy", body["error"])
        self.assertEqual(self.status()["policy"], "round_robin")     # a refused name changes nothing

    def test_drain_takes_a_node_out_and_puts_it_back(self):
        self.policy("round_robin")
        self.assertEqual(set(self.backends_of(4)), {"A", "B"})
        code, _, body = call("POST", ROUTER + "/router/drain", {"backend": "B", "drain": True})
        self.assertEqual((code, body), (200, {"backend": "B", "drain": True, "pools": ["forecast", "llm"]}))
        self.assertEqual(self.backends_of(4), ["A"] * 4)
        self.assertTrue(self.row("forecast", "B")["draining"] and self.row("llm", "B")["draining"])
        for session in ("alpha", "bravo", "charlie", "delta"):       # sessions that lived on B move too
            self.assertEqual(self.chat(session)[1]["x-backend"], "A")
        code, headers, merged = call("POST", ROUTER + "/v1/ensemble",
                                     {"init": "latest", "lead_hours": 12, "members": 8, "point": OHARE})
        self.assertEqual((code, [(s["backend"], s["members"]) for s in merged["shards"]]), (200, [("A", 8)]))
        code, _, body = call("POST", ROUTER + "/router/drain", {"backend": "B", "drain": False})
        self.assertEqual(code, 200)
        self.assertEqual(set(self.backends_of(4)), {"A", "B"})
        self.assertEqual(call("POST", ROUTER + "/router/drain", {"backend": "Z", "drain": True})[0], 404)

    # ---- trouble on a node

    def test_a_slowed_node_loses_the_preference_once(self):
        """A training job appears beside the preferred node. The adaptive policy moves away after
        the dwell time, and it does not move back and forth after that."""
        before = self.status()
        slowed = before["pools"]["forecast"]["preferred"]
        self.assertIn(slowed, ("A", "B"))
        other = "B" if slowed == "A" else "A"
        self.assertEqual(self.backends_of(3), [slowed] * 3)          # light traffic goes to the preferred node

        def busy_for(seconds, until=None):
            """Send a small forecast every 0.15 s. Returns the seconds gone when until() came true."""
            started = time.monotonic()
            while time.monotonic() - started < seconds:
                self.assertEqual(self.forecast()[0], 200)
                if until is not None and until():
                    return time.monotonic() - started
                time.sleep(0.15)
            return None

        def preferred():
            return self.status()["pools"]["forecast"]["preferred"]

        self.assertEqual(call("POST", FORECAST[slowed] + "/admin/contention", {"factor": 1.8})[0], 200)
        moved_after = busy_for(10.0, until=lambda: preferred() == other)
        self.assertIsNotNone(moved_after, "the preference stayed on the slowed node for 10 s")
        self.assertGreater(moved_after, 2.5)           # dwell_s is 3: not at the first bad probe
        self.assertTrue(self.row("forecast", slowed)["probe_data"]["busy_other"])
        self.assertEqual(self.backends_of(3), [other] * 3)
        busy_for(3.0)                                  # the slow node stays slow: nothing changes
        self.assertEqual(preferred(), other)
        self.assertEqual(call("POST", FORECAST[slowed] + "/admin/contention", {"factor": 1.0})[0], 200)
        busy_for(4.5)                                  # it is as good again, but not better: no way back
        after = self.status()
        self.assertEqual(after["pools"]["forecast"]["preferred"], other)
        self.assertEqual(after["counters"]["switches"] - before["counters"]["switches"], 1)
        switches = [d for d in after["decisions"] if d["kind"] == "switch" and d["pool"] == "forecast"]
        self.assertEqual(switches[-1]["backend"], other)
        self.assertIn("idle score", switches[-1]["reason"])
        self.assertEqual(after["counters"]["errors"], before["counters"]["errors"])

    def test_a_dead_node_is_hidden_healed_and_taken_back(self):
        """Node B's forecast server goes down. No client sees an error, the router marks B
        unhealthy, runs B's heal action after heal_after_s (10 s), and uses B again."""
        self.policy("round_robin")                     # so that B gets every second request
        before = self.status()["counters"]
        self.assertEqual(call("POST", FORECAST["B"] + "/admin/fail", {"mode": "down"})[0], 200)
        started = time.monotonic()
        answers, unhealthy_at, back_at = [], None, None
        while time.monotonic() - started < 25.0 and back_at is None:
            code, headers, body = self.forecast()
            answers.append((code, headers.get("x-backend"), headers.get("x-router-retries")))
            healthy = self.row("forecast", "B")["healthy"]
            if unhealthy_at is None and not healthy:
                unhealthy_at = time.monotonic() - started
            elif unhealthy_at is not None and healthy:
                back_at = time.monotonic() - started
            time.sleep(0.2)
        self.assertEqual(set(code for code, _, _ in answers), {200}, answers)        # no error reached a client
        self.assertIn((200, "A", "1"), answers)                                      # a retry saved a request
        self.assertIsNotNone(unhealthy_at, "B was never marked unhealthy")
        self.assertLess(unhealthy_at, 4.0)
        self.assertIsNotNone(back_at, "B did not come back within 25 s")
        self.assertGreater(back_at, 9.0)               # not before the heal action, which waits 10 s
        after = self.status()
        self.assertGreaterEqual(after["counters"]["failovers"] - before["failovers"], 1)
        self.assertEqual(after["counters"]["heals"] - before["heals"], 1)
        self.assertEqual(after["counters"]["errors"], before["errors"])
        story = [d["kind"] for d in after["decisions"]
                 if d["pool"] == "forecast" and d["backend"] == "B" and d["kind"] in ("unhealthy", "heal", "recovered")]
        self.assertEqual(story[-3:], ["unhealthy", "heal", "recovered"])             # still on the status page
        heal = [d for d in after["decisions"] if d["kind"] == "heal"][-1]
        self.assertIn("attempt 1 of 5 ran: http POST %s/admin/fail answered 200" % FORECAST["B"], heal["reason"])
        self.assertEqual(set(self.backends_of(4)), {"A", "B"})                       # B works again

    def test_a_500_is_passed_on_and_not_held_against_the_node(self):
        """A node that answers 500 while its /health says 200: by the contract the router passes
        the answer on. It does not retry it and does not mark the node unhealthy."""
        self.policy("round_robin")
        before = self.status()["counters"]["errors"]
        self.assertEqual(call("POST", FORECAST["B"] + "/admin/fail", {"mode": "error"})[0], 200)
        answers = [self.forecast() for _ in range(4)]
        by_node = dict((headers["x-backend"], (code, headers["x-router-retries"])) for code, headers, _ in answers)
        self.assertEqual(by_node, {"A": (200, "0"), "B": (500, "0")})
        self.assertIn("error", [body for code, _, body in answers if code == 500][0])
        self.assertTrue(self.row("forecast", "B")["healthy"])
        self.assertEqual(self.status()["counters"]["errors"] - before, 2)

    # ---- pages

    def test_status_has_the_documented_shape(self):
        self.forecast()
        self.chat("alpha")
        code, headers, state = call("GET", ROUTER + "/router/status")
        self.assertEqual((code, headers["access-control-allow-origin"]), (200, "*"))
        self.assertEqual(sorted(state), ["counters", "decisions", "policy", "pools", "uptime_s"])
        self.assertEqual(state["policy"], "adaptive")
        self.assertGreater(state["uptime_s"], 0)
        self.assertEqual(sorted(state["counters"]), ["errors", "failovers", "heals", "routed", "switches"])
        self.assertEqual(sorted(state["pools"]), ["forecast", "llm"])
        for pool_name, probe, numbers in (("forecast", "forecast", {"queued", "queued_units", "gpu_util_pct",
                                                                    "busy_other", "slowdown", "mem_available_gb",
                                                                    "ms_per_unit"}),
                                          ("llm", "llamacpp", {"queued", "running"})):
            pool = state["pools"][pool_name]
            self.assertEqual(sorted(pool), ["backends", "preferred"])
            self.assertIn(pool["preferred"], ("A", "B"))
            self.assertEqual([b["name"] for b in pool["backends"]], ["A", "B"])
            for b in pool["backends"]:
                self.assertEqual(set(b), BACKEND_KEYS)
                self.assertEqual((b["probe"], b["healthy"], b["draining"], b["healing"]), (probe, True, False, False))
                self.assertTrue(numbers <= set(b["probe_data"]), b["probe_data"])
                self.assertGreater(b["score"], 0)
        self.assertLessEqual(len(state["decisions"]), 50)
        for decision in state["decisions"]:
            self.assertEqual(set(decision), DECISION_KEYS)
            self.assertIn(decision["kind"], DECISION_KINDS)
        self.assertEqual(state["decisions"], sorted(state["decisions"], key=lambda d: d["ts"]))   # newest last
        code, _, health = call("GET", ROUTER + "/router/health")
        self.assertEqual((code, health["status"], health["usable"]), (200, "ok", {"forecast": 2, "llm": 2}))

    def test_the_dashboard_is_served_at_slash(self):
        code, headers, page = call("GET", ROUTER + "/")
        self.assertEqual(code, 200)
        self.assertTrue(headers["content-type"].startswith("text/html"))
        with open(os.path.join(REPO, "dashboard", "index.html"), "rb") as f:
            self.assertEqual(page, f.read())
        self.assertIn(b"/router/status", page)
        # What the page does first: read the status, and a forecast server's own numbers.
        code, headers, metrics = call("GET", FORECAST["A"] + "/metrics")
        self.assertEqual((code, headers["access-control-allow-origin"], metrics["node"]), (200, "*", "A"))
        self.assertEqual(sorted(metrics["gpu"]), ["power_w", "proc_mem_mb", "temp_c", "util_pct"])

    # ---- the agent's side and the load generator

    def test_mcp_server_serves_every_tool(self):
        env = dict(os.environ, ROUTER_URL=ROUTER)
        proc = subprocess.Popen([sys.executable, os.path.join(REPO, "agent", "weather_mcp.py")], env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        lines = queue.Queue()

        def read():
            for line in proc.stdout:
                lines.put(line)
        reader = threading.Thread(target=read)
        reader.daemon = True
        reader.start()

        def ask(message, answer=True):
            proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            proc.stdin.flush()
            return json.loads(lines.get(timeout=30).decode("utf-8")) if answer else None

        def tool(number, name, arguments):
            got = ask({"jsonrpc": "2.0", "id": number, "method": "tools/call",
                       "params": {"name": name, "arguments": arguments}})
            self.assertEqual(got["id"], number)
            result = got["result"]
            self.assertEqual(json.loads(result["content"][0]["text"]), result["structuredContent"])
            return result["isError"], result["structuredContent"]

        try:
            hello = ask({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "end-to-end test", "version": "0"}}})
            self.assertEqual(hello["result"]["protocolVersion"], "2025-06-18")
            self.assertEqual(hello["result"]["serverInfo"]["name"], "switchyard")
            ask({"jsonrpc": "2.0", "method": "notifications/initialized"}, answer=False)
            listed = ask({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
            self.assertEqual([t["name"] for t in listed], TOOL_NAMES)

            failed, got = tool(10, "forecast_point", {"lat": 41.96, "lon": -87.93, "lead_hours": 24})
            self.assertFalse(failed, got)
            self.assertEqual((got["last_lead"]["lead_h"], len(got["by_lead"])), (24, 4))
            self.assertIn(got["node"], ("A", "B"))
            failed, got = tool(11, "ohare_tomorrow", {})
            self.assertFalse(failed, got)
            self.assertLessEqual(got["low"]["temp_c"], got["high"]["temp_c"])
            failed, got = tool(12, "ensemble_range", {"lat": 41.96, "lon": -87.93, "lead_hours": 24, "members": 20})
            self.assertFalse(failed, got)
            self.assertEqual(got["node"], "A+B")
            self.assertEqual([(s["node"], s["members"]) for s in got["shards"]], [("A", 10), ("B", 10)])
            self.assertLessEqual(got["last_lead"]["temp_min_c"], got["last_lead"]["temp_max_c"])
            failed, got = tool(13, "storm_2019", {})
            self.assertFalse(failed, got)
            self.assertEqual([c["lead_h"] for c in got["checks"]], [24, 48, 72])
            self.assertIsNotNone(got["error_at_72h_c"])
            failed, got = tool(14, "cluster_status", {})
            self.assertFalse(failed, got)
            self.assertEqual((got["policy"], sorted(got["pools"]), got["node"]),
                             ("adaptive", ["forecast", "llm"], "router"))
            failed, got = tool(15, "set_routing_policy", {"policy": "round_robin"})
            self.assertEqual((failed, got), (False, {"policy": "round_robin", "node": "router"}))
            self.assertEqual(self.status()["policy"], "round_robin")
            failed, got = tool(16, "set_routing_policy", {"policy": "fastest"})
            self.assertTrue(failed)
            failed, got = tool(17, "forecast_point", {"lat": 41.96, "lon": -87.93, "lead_hours": 7})
            self.assertTrue(failed)
            self.assertIn("multiple of 6", got["error"])                 # the server's own words reach the model
        finally:
            proc.stdin.close()
            code = proc.wait(timeout=10)
            reader.join(5.0)
            proc.stdout.close()
        self.assertEqual(code, 0)                                        # it leaves when stdin closes

    def test_desk_agent_answers_through_the_router(self):
        """The LLM mock plays a tool-calling model: one tool call, then an answer that quotes the result."""
        done = subprocess.run(
            [sys.executable, os.path.join(REPO, "agent", "desk_agent.py"),
             "What will tomorrow morning be like at O'Hare?", "--session", "e2e-agent"],
            env=dict(os.environ, ROUTER_URL=ROUTER), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        out, err = done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace")
        self.assertEqual(done.returncode, 0, err)
        self.assertIn("called ohare_tomorrow({}): ok", err)
        self.assertTrue(out.startswith("Based on the tool:"), out)
        self.assertIn("Dropped 0 sentence(s) with numbers no tool gave.", out)
        self.assertIn("Facts from tools\n  ohare_tomorrow()", out)
        llm_line = [line for line in out.splitlines() if line.strip().startswith("LLM calls:")][0]
        self.assertIn(llm_line.split(":", 1)[1].strip(), ("A, A", "B, B"))   # both model calls on one node
        self.assertIn("Tools: ohare_tomorrow on ", out)

    def test_load_generator_runs_a_scenario(self):
        folder = tempfile.mkdtemp(prefix="wx_e2e_")
        self.addCleanup(shutil.rmtree, folder, True)
        out = os.path.join(folder, "run.jsonl")
        done = subprocess.run(
            [sys.executable, os.path.join(REPO, "bench", "loadgen.py"), "--target", ROUTER,
             "--scenario", "contention", "--duration", "6", "--rate", "3", "--seed", "1", "--policy", "round_robin",
             "--admin-a", FORECAST["A"], "--admin-b", FORECAST["B"], "--out", out],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=80)
        printed = done.stdout.decode("utf-8", "replace")
        self.assertEqual(done.returncode, 0, printed)
        with open(out + ".summary.json") as f:
            summary = json.load(f)
        with open(out) as f:
            records = [json.loads(line) for line in f]
        self.assertGreater(summary["requests"], 5)
        self.assertEqual((summary["ok"], summary["errors"]), (summary["requests"], 0), printed)
        self.assertEqual(len(records), summary["requests"])
        self.assertEqual(set(r["status"] for r in records), {200})
        self.assertTrue(set(r["backend"] for r in records) <= {"A", "B"})
        self.assertEqual(sorted(summary["share"]), ["A", "B"])           # round robin uses both nodes
        self.assertEqual([(e["name"], e["ok"]) for e in summary["events"]],
                         [("contention_on", True), ("contention_off", True)])
        self.assertEqual(summary["router"]["delta"]["routed"], summary["requests"])
        self.assertEqual(summary["router"]["delta"]["errors"], 0)
        self.assertEqual(summary["router_policy"], "round_robin")


if __name__ == "__main__":
    unittest.main()
