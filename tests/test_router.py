"""Tests for the router (router/router.py, contract section 3).

The first half drives the choosing logic alone, with a clock the tests move by
hand. The second half runs the real router in this process against tiny fake
backends. Every server binds 127.0.0.1 on a free port and is stopped again.

Run from the repo root:
    .venv/bin/python -m unittest tests/test_router.py -v
    .venv/bin/python -m unittest tests.test_router.ProxyTests -v

Written at the hackathon (Nov 13-14, 2026).
"""

import http.client
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTER_PY = os.path.join(REPO, "router", "router.py")

_spec = importlib.util.spec_from_file_location("wx_router", ROUTER_PY)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

BACKEND_KEYS = {"name", "url", "probe", "healthy", "draining", "healing", "inflight",
                "latency_ewma_ms", "score", "routed", "failures", "last_error", "probe_data"}
DECISION_KEYS = {"ts", "pool", "backend", "policy", "kind", "reason", "retries"}


# ---------------------------------------------------------------- helpers for the logic tests

class Clock(object):
    """A clock that only moves when a test says so."""

    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def add(self, seconds):
        self.t += seconds


def two_by_two(**over):
    """A config with backends A and B in both pools, and no log file."""
    raw = {
        "decision_log": "",
        "pools": {
            "forecast": {"backends": [
                {"name": "A", "url": "http://127.0.0.1:1", "probe": "forecast"},
                {"name": "B", "url": "http://127.0.0.1:2", "probe": "forecast"}]},
            "llm": {"affinity": True, "affinity_break": 3.0, "backends": [
                {"name": "A", "url": "http://127.0.0.1:3", "probe": "llamacpp"},
                {"name": "B", "url": "http://127.0.0.1:4", "probe": "llamacpp"}]}}}
    raw.update(over)
    return R.make_config(raw)


def make_core(policy="adaptive", **over):
    """Returns (core, clock, log): log collects every decision the core makes."""
    clock, log = Clock(), []
    core = R.Core(two_by_two(policy=policy, **over), now=clock, sink=log.append)
    return core, clock, log


def backend(core, pool, name):
    return core.pools[pool].get(name)


def set_latency(core, pool, **ms):
    """set_latency(core, "forecast", A=100, B=90) sets the latency EWMAs by hand."""
    for name, value in ms.items():
        backend(core, pool, name).latency_ewma_ms = float(value)


def set_speed(core, pool, **ms_per_unit):
    """set_speed(core, "forecast", A=200, B=100): what each backend's probe says it needs per unit.

    Nothing is queued, so a backend's idle score is its ms_per_unit in seconds.
    """
    for name, value in ms_per_unit.items():
        core.probe_result(pool, name, True, {"queued_units": 0, "ms_per_unit": float(value)})


def pick(core, pool="forecast", **kw):
    """Route one request and finish it at once. Returns the Choice."""
    choice = core.route(pool, **kw)
    if choice is not None:
        core.finish(pool, choice.backend, None)
    return choice


def kinds(log):
    return [d["kind"] for d in log]


class PolicyTests(unittest.TestCase):
    """The four policies, on the logic class alone."""

    def test_static_takes_the_first_healthy_backend(self):
        core, clock, log = make_core("static")
        picks = [pick(core) for _ in range(3)]
        self.assertEqual([c.backend for c in picks], ["A", "A", "A"])
        self.assertEqual(picks[0].reason, "first healthy")
        self.assertEqual(picks[0].policy, "static")
        core.probe_result("forecast", "A", False, error="down")
        core.probe_result("forecast", "A", False, error="down")
        self.assertEqual(pick(core).backend, "B")
        self.assertEqual(core.status()["pools"]["forecast"]["preferred"], "B")

    def test_round_robin_rotates_over_healthy_backends(self):
        core, clock, log = make_core("round_robin")
        self.assertEqual([pick(core).backend for _ in range(4)], ["A", "B", "A", "B"])
        self.assertEqual(pick(core).reason, "rotation")      # that was A
        core.set_drain("A", True)
        self.assertEqual([pick(core).backend for _ in range(3)], ["B", "B", "B"])
        core.set_drain("A", False)
        self.assertEqual([pick(core).backend for _ in range(2)], ["A", "B"])
        self.assertIsNone(core.status()["pools"]["forecast"]["preferred"])

    def test_least_loaded_takes_the_lowest_score_every_time(self):
        core, clock, log = make_core("least_loaded")
        # Nothing finishes, so each request makes its backend look busier.
        self.assertEqual([core.route("forecast").backend for _ in range(4)], ["A", "B", "A", "B"])
        for name in "ABAB":
            core.finish("forecast", name, None)
        set_latency(core, "forecast", A=300, B=100)
        choice = pick(core)
        self.assertEqual((choice.backend, choice.reason), ("B", "lowest score"))
        self.assertEqual(core.status()["pools"]["forecast"]["preferred"], "B")

    def test_least_loaded_flaps_where_adaptive_holds(self):
        """Two scores that swap places within the margin: 10% apart, margin 25%."""
        results = {}
        for policy in ("least_loaded", "adaptive"):
            core, clock, log = make_core(policy)
            seen = []
            for i in range(40):
                clock.add(1.0)
                if i % 2:
                    set_speed(core, "forecast", A=100, B=90)
                else:
                    set_speed(core, "forecast", A=90, B=100)
                seen.append(pick(core).backend)
            results[policy] = (seen, core.counters["switches"])
        flappy, steady = results["least_loaded"], results["adaptive"]
        self.assertEqual(flappy[0], ["A", "B"] * 20)
        self.assertEqual(flappy[1], 39)                    # every change of mind is counted as a switch
        self.assertEqual(steady[0], ["A"] * 40)
        self.assertEqual(steady[1], 0)

    def test_switches_count_a_change_of_choice_for_static_and_least_loaded(self):
        core, clock, log = make_core("static")
        self.assertEqual([pick(core).backend for _ in range(3)], ["A", "A", "A"])
        self.assertEqual(core.counters["switches"], 0)
        core.set_drain("A", True)                          # static moves to B: one switch
        self.assertEqual([pick(core).backend for _ in range(3)], ["B", "B", "B"])
        self.assertEqual(core.counters["switches"], 1)
        core.set_drain("A", False)                         # and back to A: two
        self.assertEqual(pick(core).backend, "A")
        self.assertEqual(core.counters["switches"], 2)
        # A retry on the other backend is a failover, not a switch.
        choice = core.route("forecast", exclude=["A"], retries=1)
        core.finish("forecast", choice.backend, None)
        self.assertEqual((choice.backend, choice.kind), ("B", "failover"))
        self.assertEqual(pick(core).backend, "A")
        self.assertEqual(core.counters["switches"], 2)
        # A new policy starts with no last choice, and round robin never counts.
        core.set_policy("round_robin")
        self.assertEqual([pick(core).backend for _ in range(4)], ["A", "B", "A", "B"])
        core.set_policy("least_loaded")
        self.assertEqual(pick(core).backend, "A")
        self.assertEqual(core.counters["switches"], 2)
        self.assertEqual(kinds(log).count("switch"), 0)    # counted, but not a row in the decisions

    def test_our_own_traffic_does_not_move_the_preference(self):
        """A is preferred and busy with our requests, B stands idle. That is no reason to switch."""
        core, clock, log = make_core("adaptive")
        calm = {"queued_units": 0, "ms_per_unit": 10.0, "slowdown": 1.0, "busy_other": False}
        slow = {"queued_units": 0, "ms_per_unit": 18.0, "slowdown": 1.8, "busy_other": True}
        core.probe_result("forecast", "A", True, dict(calm))
        core.probe_result("forecast", "B", True, dict(calm))
        for second in range(60):
            clock.add(1.0)
            self.assertEqual(core.route("forecast", units=300).backend, "A")
            self.assertAlmostEqual(core.score(backend(core, "forecast", "A")), 3.01)
            core.probe_result("forecast", "A", True, dict(calm, queued_units=300), units_before=300)
            core.probe_result("forecast", "B", True, dict(calm))
            core.finish("forecast", "A", True, latency_ms=3000.0, units=300)
        self.assertEqual(core.counters["switches"], 0)
        self.assertEqual(core.pools["forecast"].preferred, "A")
        # A training job starts beside A. Now B really is the better box: one switch, after the dwell.
        core.probe_result("forecast", "A", True, dict(slow))
        clock.add(2.5)
        core.probe_result("forecast", "A", True, dict(slow))
        self.assertEqual(core.pools["forecast"].preferred, "A")
        clock.add(0.5)
        core.probe_result("forecast", "A", True, dict(slow))
        self.assertEqual(core.pools["forecast"].preferred, "B")
        self.assertEqual(log[-1]["kind"], "switch")
        self.assertIn("B (idle score 0.010) beat A (idle score 0.027) by more than 25% for 3.0 s",
                      log[-1]["reason"])
        # B now carries the traffic and A is still slow: nothing flips back.
        for second in range(30):
            clock.add(1.0)
            self.assertEqual(core.route("forecast", units=300).backend, "B")
            core.probe_result("forecast", "A", True, dict(slow))
            core.probe_result("forecast", "B", True, dict(calm, queued_units=300), units_before=300)
            core.finish("forecast", "B", True, latency_ms=3000.0, units=300)
        # The job ends. A is as good as B again, but not better: the preference stays where it is.
        for second in range(30):
            clock.add(1.0)
            core.probe_result("forecast", "A", True, dict(calm))
            self.assertEqual(pick(core).backend, "B")
        self.assertEqual(core.counters["switches"], 1)

    def test_a_different_mix_of_requests_does_not_move_the_preference(self):
        """LLM backend A served long answers, B short ones. That says nothing about the two boxes."""
        core, clock, log = make_core("adaptive")
        set_latency(core, "llm", A=2000, B=200)
        a, b = backend(core, "llm", "A"), backend(core, "llm", "B")
        self.assertAlmostEqual(core.score(a), 2.0)                   # the plain score is each one's own
        self.assertAlmostEqual(core.score(b), 0.2)
        self.assertAlmostEqual(core.score(a, idle=True), 1.1)        # the idle score uses the pool's mean
        self.assertAlmostEqual(core.score(b, idle=True), 1.1)
        for _ in range(30):
            clock.add(1.0)
            self.assertEqual(pick(core, "llm").backend, "A")
        self.assertEqual(core.counters["switches"], 0)

    def test_work_from_other_clients_does_move_the_preference(self):
        core, clock, log = make_core("adaptive")
        core.probe_result("forecast", "B", True, {"queued_units": 0, "ms_per_unit": 10.0})
        for second in range(4):                            # somebody loads A directly, not through us
            core.probe_result("forecast", "A", True, {"queued_units": 500, "ms_per_unit": 10.0})
            clock.add(1.0)
        core.tick()
        self.assertEqual(core.pools["forecast"].preferred, "B")
        self.assertEqual(core.counters["switches"], 1)

    def test_adaptive_prefers_one_backend(self):
        core, clock, log = make_core("adaptive")
        choice = pick(core)
        self.assertEqual((choice.backend, choice.reason, choice.kind), ("A", "preferred", "route"))
        self.assertEqual(core.status()["pools"]["forecast"]["preferred"], "A")
        self.assertEqual(log[-1], {"ts": log[-1]["ts"], "pool": "forecast", "backend": "A",
                                   "policy": "adaptive", "kind": "route", "reason": "preferred",
                                   "retries": 0})

    def test_dwell_must_pass_before_a_switch(self):
        core, clock, log = make_core("adaptive")   # hysteresis 0.25, dwell 3 s, cooldown 10 s
        set_speed(core, "forecast", A=200, B=100)         # B's idle score is half of A's
        core.tick()
        clock.add(2.5)
        self.assertEqual(pick(core).backend, "A")          # better, but not for long enough
        self.assertEqual(core.counters["switches"], 0)
        clock.add(0.5)
        choice = pick(core)                                # 3.0 s: now it switches
        self.assertEqual((choice.backend, choice.reason), ("B", "preferred"))
        self.assertEqual(core.counters["switches"], 1)
        switch = [d for d in log if d["kind"] == "switch"][0]
        self.assertEqual((switch["pool"], switch["backend"]), ("forecast", "B"))
        self.assertIn("0.100", switch["reason"])           # both scores are in the reason
        self.assertIn("0.200", switch["reason"])
        self.assertEqual(core.status()["pools"]["forecast"]["preferred"], "B")
        self.assertEqual(core.status()["pools"]["llm"]["preferred"], "A")   # pools are separate

    def test_dwell_starts_over_when_the_lead_is_lost(self):
        core, clock, log = make_core("adaptive")
        set_speed(core, "forecast", A=200, B=100)
        core.tick()
        clock.add(2.0)
        set_speed(core, "forecast", A=200, B=190)          # inside the margin again
        core.tick()
        clock.add(0.5)
        set_speed(core, "forecast", A=200, B=100)          # ahead again, from t = 2.5
        core.tick()
        clock.add(2.5)
        core.tick()                                        # t = 5.0: only 2.5 s in a row
        self.assertEqual(core.counters["switches"], 0)
        clock.add(0.5)
        core.tick()                                        # t = 5.5: 3.0 s in a row
        self.assertEqual(core.counters["switches"], 1)

    def test_no_switch_while_the_margin_is_not_met(self):
        core, clock, log = make_core("adaptive")
        set_speed(core, "forecast", A=100, B=76)           # 24% better: not enough
        for _ in range(30):
            clock.add(1.0)
            self.assertEqual(pick(core).backend, "A")
        set_speed(core, "forecast", A=100, B=74)           # 26% better: enough
        for _ in range(4):
            clock.add(1.0)
            core.tick()
        self.assertEqual(core.status()["pools"]["forecast"]["preferred"], "B")

    def test_cooldown_holds_back_the_next_switch(self):
        core, clock, log = make_core("adaptive")
        set_speed(core, "forecast", A=200, B=100)
        core.tick()
        clock.add(3.0)
        core.tick()                                        # switch to B at t = 3
        self.assertEqual(core.pools["forecast"].preferred, "B")
        set_speed(core, "forecast", A=10, B=100)           # now A is far better
        core.tick()
        clock.add(9.5)                                     # t = 12.5: dwell long over, cooldown not
        self.assertEqual(pick(core).backend, "B")
        self.assertEqual(core.counters["switches"], 1)
        clock.add(0.5)                                     # t = 13.0: 10 s after the last switch
        self.assertEqual(pick(core).backend, "A")
        self.assertEqual(core.counters["switches"], 2)

    def test_an_unhealthy_preferred_backend_is_left_at_once(self):
        core, clock, log = make_core("adaptive")
        set_speed(core, "forecast", A=200, B=100)
        core.tick()
        clock.add(3.0)
        core.tick()                                        # a normal switch to B at t = 3
        clock.add(1.0)                                     # t = 4: deep inside the cooldown
        del log[:]
        core.probe_result("forecast", "B", False, error="health: connection refused")
        self.assertEqual(kinds(log), [])                   # one failure is not enough (fail_n 2)
        core.finish("forecast", "B", False, error="timed out")   # a request counts too
        self.assertEqual(kinds(log), ["unhealthy", "switch"])
        self.assertEqual(log[0]["backend"], "B")
        self.assertIn("timed out", log[0]["reason"])
        self.assertEqual(log[1]["backend"], "A")
        self.assertIn("unhealthy", log[1]["reason"])
        choice = pick(core)
        self.assertEqual((choice.backend, choice.reason), ("A", "preferred"))
        # B comes back: it does not get the preference back just for that.
        clock.add(3.0)
        core.probe_result("forecast", "B", True, {})
        self.assertEqual(kinds(log)[-2:], ["route", "recovered"])
        self.assertEqual(core.pools["forecast"].preferred, "A")

    def test_spill_sends_the_overflow_to_the_next_best(self):
        core, clock, log = make_core("adaptive", adaptive={"spill_inflight": 4})
        first = [core.route("forecast") for _ in range(5)]
        self.assertEqual([(c.backend, c.reason) for c in first], [("A", "preferred")] * 5)
        over = [core.route("forecast") for _ in range(5)]  # A has 5 open: more than 4
        self.assertEqual([(c.backend, c.reason) for c in over], [("B", "spill")] * 5)
        # B is now as loaded as A, so spilling would not help: back to A.
        again = core.route("forecast")
        self.assertEqual((again.backend, again.reason), ("A", "preferred"))
        self.assertEqual(core.pools["forecast"].preferred, "A")     # a spill is not a switch
        self.assertEqual(core.counters["switches"], 0)
        for _ in range(6):
            core.finish("forecast", "A", None)
        for _ in range(5):
            core.finish("forecast", "B", None)
        self.assertEqual(backend(core, "forecast", "A").inflight, 0)
        self.assertEqual(core.route("forecast").reason, "preferred")

    def test_spill_with_the_default_setting(self):
        """spill_inflight is 0: a request never waits on the preferred backend while another one is
        quicker. A forecast server runs one request at a time, so anything in flight means a wait."""
        core, clock, log = make_core("adaptive")
        picks = [core.route("forecast") for _ in range(5)]
        self.assertEqual([(c.backend, c.reason) for c in picks],
                         [("A", "preferred"), ("B", "spill"), ("A", "preferred"), ("B", "spill"), ("A", "preferred")])
        self.assertEqual(core.pools["forecast"].preferred, "A")
        self.assertEqual(core.counters["switches"], 0)
        for name in "ABABA":
            core.finish("forecast", name, None)
        # With the preferred backend free again, everything goes there: one request after another stays on A.
        self.assertEqual([pick(core).backend for _ in range(4)], ["A"] * 4)

    def test_spill_has_nowhere_to_go_with_one_backend(self):
        core, clock, log = make_core("adaptive")
        core.set_drain("B", True)
        picks = [core.route("forecast") for _ in range(8)]
        self.assertEqual(set((c.backend, c.reason) for c in picks), {("A", "preferred")})

    def test_policy_can_change_at_run_time(self):
        core, clock, log = make_core("static")
        self.assertEqual(core.set_policy("round_robin"), "round_robin")
        self.assertEqual(log[-1]["kind"], "policy")
        self.assertEqual(log[-1]["reason"], "static -> round_robin")
        self.assertEqual(log[-1]["policy"], "round_robin")
        self.assertEqual([pick(core).backend for _ in range(2)], ["A", "B"])
        with self.assertRaises(ValueError):
            core.set_policy("fastest")
        self.assertEqual(core.policy, "round_robin")


class ScoreTests(unittest.TestCase):

    def test_general_formula(self):
        core, clock, log = make_core("static")             # static: both requests below go to A
        b = backend(core, "llm", "A")
        self.assertAlmostEqual(core.score(b), 0.2)         # (0 + 0 + 1) x default 200 ms
        core.route("llm")
        core.route("llm")                                  # 2 in flight
        b.latency_ewma_ms = 100.0
        b.probe_data = {"queued": 3, "running": 1, "slowdown": 1.5}
        self.assertAlmostEqual(core.score(b), (2 + 3 + 1) * 0.1 * 1.5)
        b.probe_data = {"queued": None, "running": None}   # unknown counts as 0 and 1.0
        self.assertAlmostEqual(core.score(b), (2 + 0 + 1) * 0.1)

    def test_penalties(self):
        core, clock, log = make_core()
        b = backend(core, "forecast", "A")
        base = core.score(b)
        b.probe_data = {"busy_other": True}
        self.assertAlmostEqual(core.score(b), base * 1.5)
        b.probe_data = {"busy_other": True, "mem_available_gb": 4.0}
        self.assertAlmostEqual(core.score(b), base * 1.5 * 2.0)
        b.probe_data = {"busy_other": False, "mem_available_gb": 101.2}
        self.assertAlmostEqual(core.score(b), base)
        b.probe_data = {"busy_other": None, "mem_available_gb": None, "slowdown": None}
        self.assertAlmostEqual(core.score(b), base)

    def test_forecast_formula_counts_the_work_in_units(self):
        core, clock, log = make_core()
        b = backend(core, "forecast", "A")
        data = {"queued": 3, "queued_units": 120, "ms_per_unit": 10.0, "slowdown": 2.0}
        core.probe_result("forecast", "A", True, dict(data))
        # None of the 120 units is ours, so it is other clients' work. ms_per_unit
        # already holds the slowdown: it is not multiplied in a second time.
        self.assertAlmostEqual(core.score(b), (120 + 1) * 0.010)
        # A request of ours counts at once and with its own size, before any probe sees it.
        core.route("forecast", units=40)
        self.assertAlmostEqual(core.score(b), (40 + 120 + 1) * 0.010)
        # The next probe sees it in the server's queue. It is still ours, not somebody else's.
        core.probe_result("forecast", "A", True, dict(data, queued_units=160))
        self.assertAlmostEqual(core.score(b), (40 + 120 + 1) * 0.010)
        self.assertEqual(core.status()["pools"]["forecast"]["backends"][0]["score"], 1.61)
        # When it ends, its units leave the score at once.
        core.finish("forecast", "A", True, latency_ms=400.0, units=40)
        self.assertAlmostEqual(core.score(b), (120 + 1) * 0.010)
        # The other clients' work goes when a probe says so.
        core.probe_result("forecast", "A", True, dict(data, queued_units=0))
        self.assertAlmostEqual(core.score(b), 0.010)
        # Without ms_per_unit the general formula is used: (0 + 3 + 1) x 0.4 s x 2.0.
        core.probe_result("forecast", "A", True, dict(data, ms_per_unit=None))
        self.assertAlmostEqual(core.score(b), (0 + 3 + 1) * 0.4 * 2.0)

    def test_a_request_that_ends_during_a_probe_is_not_other_work(self):
        core, clock, log = make_core()
        b = backend(core, "forecast", "A")
        core.route("forecast", units=400)
        before = core.open_units("forecast", "A")           # the probe leaves now: 400 units are ours
        self.assertEqual(before, 400)
        core.finish("forecast", "A", True, latency_ms=4000.0, units=400)     # the request ends
        core.probe_result("forecast", "A", True, {"queued_units": 400, "ms_per_unit": 10.0},
                          units_before=before)              # the probe saw it just before it ended
        self.assertAlmostEqual(core.score(b), 0.010)        # an idle backend, not 400 units from others

    def test_the_idle_score_leaves_our_own_work_out(self):
        core, clock, log = make_core()
        b = backend(core, "forecast", "A")
        core.probe_result("forecast", "A", True, {"queued_units": 30, "ms_per_unit": 18.0,
                                                  "slowdown": 1.8, "busy_other": True})
        core.route("forecast", units=400)
        self.assertAlmostEqual(core.score(b), (400 + 30 + 1) * 0.018 * 1.5)
        self.assertAlmostEqual(core.score(b, idle=True), (30 + 1) * 0.018 * 1.5)
        llm = backend(core, "llm", "A")
        llm.latency_ewma_ms = 100.0
        llm.probe_data = {"queued": 3, "running": 1}
        core.route("llm")
        self.assertAlmostEqual(core.score(llm), (1 + 3 + 1) * 0.1)
        self.assertAlmostEqual(core.score(llm, idle=True), 0.1)
        self.assertAlmostEqual(core.score(llm, latency_ms=500.0), (1 + 3 + 1) * 0.5)

    def test_forecast_units_from_the_request_body(self):
        def units(obj, method="POST", path="/v1/forecast"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
            return R.forecast_units(method, path, body)
        self.assertEqual(units({"lead_hours": 24}), 4)                       # members defaults to 1
        self.assertEqual(units({"lead_hours": 120, "members": 20}), 400)
        self.assertEqual(units({"lead_hours": 48, "members": None}), 8)
        self.assertEqual(units({"lead_hours": 72.0, "members": 6}), 72)
        # What the server will refuse counts as 1, and never raises.
        for bad in ({}, {"lead_hours": "24"}, {"lead_hours": 24, "members": True}, {"lead_hours": -6},
                    {"lead_hours": 24, "members": 10 ** 9}, {"lead_hours": float("nan")}, [1, 2],
                    b"not json", b"\xff\xfe", b""):
            self.assertEqual(units(bad), 1, bad)
        self.assertEqual(units({"lead_hours": 3}), 1)                        # less than one step
        self.assertEqual(units(b" " * 70000), 1)                             # too large to parse
        # Nothing else in the pool uses the GPU.
        self.assertEqual(units({"lead_hours": 24}, "GET", "/v1/info"), 0)
        self.assertEqual(units({"time": "2026-11-14T00", "npy_b64": "abc"}, "POST", "/v1/state"), 0)
        self.assertEqual(units({"lead_hours": 24}, "GET"), 0)

    def test_an_unmeasured_backend_is_taken_to_be_like_its_peers(self):
        core, clock, log = make_core("adaptive")
        a, b = backend(core, "llm", "A"), backend(core, "llm", "B")
        self.assertAlmostEqual(core.score(b), 0.2)         # nobody measured: the default, 200 ms
        core.route("llm")
        core.finish("llm", "A", True, latency_ms=5000.0)   # A served one slow answer
        self.assertAlmostEqual(core.score(a), 5.0)
        self.assertAlmostEqual(core.score(b), 5.0)         # B: like A, not 25 times faster
        for _ in range(20):                                # so nothing pulls the preference to B
            clock.add(1.0)
            self.assertEqual(pick(core, "llm").backend, "A")
        self.assertEqual(core.counters["switches"], 0)
        b.latency_ewma_ms = 1000.0                         # once measured, its own number counts
        self.assertAlmostEqual(core.score(b), 1.0)
        self.assertAlmostEqual(core.score(backend(core, "forecast", "B")), 0.2)   # another pool

    def test_latency_ewma(self):
        core, clock, log = make_core()
        core.route("forecast")
        core.finish("forecast", "A", True, latency_ms=100.0)
        self.assertEqual(backend(core, "forecast", "A").latency_ewma_ms, 100.0)
        core.route("forecast")
        core.finish("forecast", "A", True, latency_ms=200.0)
        self.assertAlmostEqual(backend(core, "forecast", "A").latency_ewma_ms, 120.0)
        core.route("forecast")
        core.finish("forecast", "A", None, latency_ms=5000.0)    # a 500 or a hang-up: ignored
        self.assertAlmostEqual(backend(core, "forecast", "A").latency_ewma_ms, 120.0)
        self.assertEqual(backend(core, "forecast", "A").inflight, 0)


def session_keys(core):
    """One session key whose home backend is A, and one whose home is B."""
    found = {}
    for i in range(200):
        key = "session-%d" % i
        found.setdefault(pick(core, "llm", key=key).backend, key)
        if len(found) == 2:
            return found["A"], found["B"]
    raise AssertionError("200 keys all landed on one backend: %r" % found)


class AffinityTests(unittest.TestCase):
    """Session affinity in the llm pool."""

    def test_hit_sticks_to_the_home_backend(self):
        core, clock, log = make_core("adaptive")
        key_a, key_b = session_keys(core)
        for _ in range(3):
            choice = pick(core, "llm", key=key_b)
            self.assertEqual((choice.backend, choice.reason, choice.affinity), ("B", "affinity", "hit"))
            choice = pick(core, "llm", key=key_a)
            self.assertEqual((choice.backend, choice.affinity), ("A", "hit"))
        self.assertEqual(core.pools["llm"].preferred, "A")   # the policy itself would say A

    def test_no_key_means_none(self):
        core, clock, log = make_core("adaptive")
        choice = pick(core, "llm")
        self.assertEqual((choice.backend, choice.reason, choice.affinity), ("A", "preferred", "none"))
        self.assertIsNone(pick(core, "forecast", key="anything").affinity)   # not an affinity pool

    def test_affinity_breaks_when_home_is_too_loaded(self):
        core, clock, log = make_core("adaptive")
        key_a, key_b = session_keys(core)
        core.probe_result("llm", "B", True, {"queued": 1, "running": 1})
        self.assertEqual(pick(core, "llm", key=key_b).affinity, "hit")      # 2x the best: still home
        core.probe_result("llm", "B", True, {"queued": 3, "running": 1})
        choice = pick(core, "llm", key=key_b)                                # 4x the best: break at 3x
        self.assertEqual((choice.backend, choice.reason, choice.affinity), ("A", "preferred", "miss"))
        core.probe_result("llm", "B", True, {"queued": 0, "running": 0})
        self.assertEqual(pick(core, "llm", key=key_b).affinity, "hit")
        # Now A is the loaded one. The policy still sends A's session to A (it is
        # the preferred backend), and a request that lands on its home is a hit.
        core.probe_result("llm", "A", True, {"queued": 3, "running": 1})
        choice = pick(core, "llm", key=key_a)
        self.assertEqual((choice.backend, choice.reason, choice.affinity), ("A", "preferred", "hit"))

    def test_a_home_that_only_looks_slow_is_not_left(self):
        """B's sessions have long prompts, so B's latency is 9 times A's. B is idle all the same."""
        core, clock, log = make_core("adaptive")
        key_a, key_b = session_keys(core)
        set_latency(core, "llm", A=100, B=900)
        for _ in range(3):
            choice = pick(core, "llm", key=key_b)
            self.assertEqual((choice.backend, choice.reason, choice.affinity), ("B", "affinity", "hit"))
        # Load still breaks it: three requests more than the idle backend is over 3x.
        open_on_b = [core.route("llm", key=key_b) for _ in range(3)]
        self.assertEqual([c.backend for c in open_on_b], ["B", "B", "B"])
        self.assertEqual(pick(core, "llm", key=key_b).affinity, "miss")

    def test_home_moves_while_its_backend_is_unhealthy(self):
        core, clock, log = make_core("adaptive")
        key_a, key_b = session_keys(core)
        core.probe_result("llm", "B", False, error="down")
        core.probe_result("llm", "B", False, error="down")
        choice = pick(core, "llm", key=key_b)
        self.assertEqual((choice.backend, choice.affinity), ("A", "hit"))    # A is home for now
        clock.add(3.0)
        core.probe_result("llm", "B", True, {})
        choice = pick(core, "llm", key=key_b)
        self.assertEqual((choice.backend, choice.affinity), ("B", "hit"))    # and back again

    def test_a_retry_away_from_home_is_a_miss(self):
        core, clock, log = make_core("adaptive")
        key_a, key_b = session_keys(core)
        choice = core.route("llm", key=key_b, exclude=["B"], retries=1, after="B failed: status 503")
        self.assertEqual((choice.backend, choice.affinity, choice.kind), ("A", "miss", "failover"))
        self.assertTrue(choice.reason.startswith("B failed: status 503"))
        self.assertEqual(log[-1]["kind"], "failover")
        self.assertEqual(log[-1]["retries"], 1)
        self.assertEqual(core.counters["failovers"], 1)

    def test_affinity_can_be_off(self):
        clock = Clock()
        cfg = two_by_two()
        cfg["pools"]["llm"]["affinity"] = False
        core = R.Core(cfg, now=clock)
        self.assertIsNone(core.route("llm", key="session-1").affinity)


class DrainAndHealthTests(unittest.TestCase):

    def test_a_draining_backend_gets_nothing_new(self):
        core, clock, log = make_core("adaptive")
        open_on_a = core.route("forecast")
        self.assertEqual(open_on_a.backend, "A")
        self.assertEqual(core.set_drain("A", True), ["forecast", "llm"])
        self.assertEqual(kinds(log)[-4:], ["drain", "drain", "switch", "switch"])
        self.assertIn("draining", log[-1]["reason"])
        self.assertEqual(core.pools["forecast"].preferred, "B")    # left at once, no dwell
        for policy in R.POLICIES:
            core.set_policy(policy)
            self.assertEqual(set(pick(core).backend for _ in range(4)), {"B"}, policy)
            self.assertEqual(set(pick(core, "llm", key="k%d" % i).backend for i in range(8)), {"B"})
        # What was already running there still ends normally.
        core.finish("forecast", "A", True, latency_ms=50.0)
        row = core.status()["pools"]["forecast"]["backends"][0]
        self.assertEqual((row["draining"], row["inflight"], row["latency_ewma_ms"]), (True, 0, 50.0))
        self.assertEqual(core.set_drain("B", True), ["forecast", "llm"])
        self.assertIsNone(core.route("forecast"))                  # nobody left
        self.assertEqual(core.health()["status"], "degraded")
        core.set_drain("A", False)
        self.assertEqual(pick(core).backend, "A")
        self.assertEqual(core.set_drain("nobody", True), [])

    def test_unhealthy_after_fail_n_and_back_after_a_good_probe(self):
        core, clock, log = make_core("round_robin")          # fail_n 2, recover_s 3
        core.finish("forecast", "A", False, error="connection refused")
        self.assertTrue(backend(core, "forecast", "A").healthy)
        core.finish("forecast", "A", True)                   # a success in between: start over
        core.probe_result("forecast", "A", False, error="health answered 503 (loading)")
        self.assertTrue(backend(core, "forecast", "A").healthy)
        core.finish("forecast", "A", False, error="status 503")     # two in a row
        row = core.status()["pools"]["forecast"]["backends"][0]
        self.assertEqual((row["healthy"], row["failures"], row["last_error"]), (False, 3, "status 503"))
        self.assertEqual((log[-1]["kind"], log[-1]["backend"]), ("unhealthy", "A"))
        self.assertIn("status 503", log[-1]["reason"])
        self.assertEqual([pick(core).backend for _ in range(3)], ["B", "B", "B"])
        core.probe_result("forecast", "A", False, error="health: timed out")
        self.assertEqual(kinds(log).count("unhealthy"), 1)   # said once, not on every probe
        clock.add(2.5)
        core.probe_result("forecast", "A", True, {})         # too early: recover_s is 3
        self.assertFalse(backend(core, "forecast", "A").healthy)
        clock.add(0.5)
        core.probe_result("forecast", "A", True, {})
        row = core.status()["pools"]["forecast"]["backends"][0]
        self.assertEqual((row["healthy"], row["last_error"]), (True, None))
        self.assertEqual((log[-1]["kind"], log[-1]["backend"]), ("recovered", "A"))
        self.assertEqual(sorted(pick(core).backend for _ in range(2)), ["A", "B"])

    def test_no_backend_left(self):
        core, clock, log = make_core("adaptive")
        for name in "AB":
            core.probe_result("forecast", name, False, error="down")
            core.probe_result("forecast", name, False, error="down")
        self.assertIsNone(core.route("forecast"))
        self.assertIsNone(core.plan_fanout("forecast", 10))
        self.assertEqual(core.counters["routed"], 0)
        self.assertEqual(core.route("llm").backend, "A")     # the other pool does not care


HEAL = {"http": {"url": "http://127.0.0.1:1/admin/fail", "method": "POST", "json": {"mode": "ok"}}}


def make_heal_core():
    """Forecast A has a heal action, B has none. Defaults: after 10 s, back-off 5, 15, 60, 5 tries."""
    clock, log = Clock(), []
    cfg = R.make_config({"decision_log": "", "policy": "static", "pools": {"forecast": {"backends": [
        {"name": "A", "url": "http://127.0.0.1:1", "probe": "forecast", "heal": HEAL},
        {"name": "B", "url": "http://127.0.0.1:2", "probe": "forecast"}]}}})
    return R.Core(cfg, now=clock, sink=log.append), clock, log


def take_down(core, name):
    core.probe_result("forecast", name, False, error="health: connection refused")
    core.probe_result("forecast", name, False, error="health: connection refused")


class HealLogicTests(unittest.TestCase):
    """When a heal action is due. Running it is tested further down."""

    def test_heal_after_s_then_back_off_then_give_up(self):
        core, clock, log = make_heal_core()
        take_down(core, "A")
        take_down(core, "B")                                 # B has no action: never due
        clock.add(9.5)
        self.assertEqual(core.due_heals(), [])
        clock.add(0.5)                                       # unhealthy for 10 s
        self.assertEqual(core.due_heals(), [("forecast", "A", HEAL, 1)])
        self.assertTrue(core.status()["pools"]["forecast"]["backends"][0]["healing"])
        self.assertEqual(core.due_heals(), [])               # not twice while it runs
        clock.add(1.0)
        core.heal_done("forecast", "A", 1, True, "http POST http://127.0.0.1:1/admin/fail answered 200")
        self.assertEqual((log[-1]["kind"], log[-1]["backend"]), ("heal", "A"))
        self.assertEqual(log[-1]["reason"],
                         "attempt 1 of 5 ran: http POST http://127.0.0.1:1/admin/fail answered 200")
        self.assertEqual(core.counters["heals"], 1)
        # The waits between attempts: 5 s, 15 s, 60 s, 60 s, counted from the end of the last one.
        for attempt, wait in ((2, 5.0), (3, 15.0), (4, 60.0), (5, 60.0)):
            clock.add(wait - 0.5)
            self.assertEqual(core.due_heals(), [], "attempt %d came too early" % attempt)
            clock.add(0.5)
            self.assertEqual(core.due_heals(), [("forecast", "A", HEAL, attempt)])
            self.assertTrue(backend(core, "forecast", "A").healing)
            core.heal_done("forecast", "A", attempt, False, "connection refused")
        self.assertIn("attempt 5 of 5 failed", log[-1]["reason"])
        self.assertFalse(backend(core, "forecast", "A").healing)    # out of tries
        clock.add(3600.0)
        self.assertEqual(core.due_heals(), [])               # heal_max_attempts is 5
        self.assertEqual(core.counters["heals"], 5)

    def test_a_recovery_resets_the_attempts(self):
        core, clock, log = make_heal_core()
        take_down(core, "A")
        clock.add(10.0)
        self.assertEqual(len(core.due_heals()), 1)
        core.heal_done("forecast", "A", 1, True, "ok")
        clock.add(5.0)
        self.assertEqual(core.due_heals(), [("forecast", "A", HEAL, 2)])
        core.heal_done("forecast", "A", 2, True, "ok")
        core.probe_result("forecast", "A", True, {})         # it is back
        row = core.status()["pools"]["forecast"]["backends"][0]
        self.assertEqual((row["healthy"], row["healing"]), (True, False))
        clock.add(100.0)
        self.assertEqual(core.due_heals(), [])               # healthy: nothing to heal
        take_down(core, "A")
        clock.add(9.5)
        self.assertEqual(core.due_heals(), [])
        clock.add(0.5)
        self.assertEqual(core.due_heals(), [("forecast", "A", HEAL, 1)])   # counting from 1 again

    def test_a_draining_backend_is_left_alone(self):
        core, clock, log = make_heal_core()
        take_down(core, "A")
        core.set_drain("A", True)
        clock.add(60.0)
        self.assertEqual(core.due_heals(), [])
        core.set_drain("A", False)
        self.assertEqual(core.due_heals(), [("forecast", "A", HEAL, 1)])


class DecisionTests(unittest.TestCase):

    def test_route_decisions_are_sampled_but_all_are_logged(self):
        core, clock, log = make_core("static")
        for _ in range(20):
            pick(core)
        shown = core.status()["decisions"]
        self.assertEqual(len(shown), 5)                      # 5 a second in the status list
        self.assertEqual(len(log), 20)                       # every one in the log
        clock.add(1.0)
        for _ in range(20):
            pick(core)
        self.assertEqual(len(core.status()["decisions"]), 10)
        self.assertEqual(len(log), 40)
        for _ in range(10):                                  # anything but "route" is always kept
            core.finish("forecast", core.route("forecast", exclude=["A"], retries=1).backend, None)
        shown = core.status()["decisions"]
        self.assertEqual([d["kind"] for d in shown], ["route"] * 10 + ["failover"] * 10)
        for decision in shown:
            self.assertEqual(set(decision), DECISION_KEYS)

    def test_route_rows_give_way_before_the_rare_ones(self):
        core, clock, log = make_core("static")
        core.set_drain("B", True)                          # two rare rows, one per pool
        for _ in range(300):                               # over a minute of traffic, 4 route rows a second
            clock.add(0.25)
            pick(core)
        shown = core.status()["decisions"]
        self.assertEqual([d["kind"] for d in shown], ["drain"] * 2 + ["route"] * 48)
        self.assertEqual(shown[-1], log[-1])               # the newest row is last
        self.assertEqual(shown[2:], [d for d in log if d["kind"] == "route"][-48:])
        # Route rows keep half of a full list. Past that, the oldest row goes, whatever its kind.
        for i in range(100):
            clock.add(0.25)
            pick(core)
            core.set_policy("round_robin" if i % 2 == 0 else "static")
        shown = core.status()["decisions"]
        self.assertEqual(len(shown), 50)
        self.assertEqual([d["kind"] for d in shown].count("route"), 25)
        self.assertEqual([d["kind"] for d in shown].count("policy"), 25)
        self.assertEqual(shown[-1], log[-1])
        kept = set(id(d) for d in shown)
        self.assertEqual([d for d in log if id(d) in kept], shown)   # still in the order they happened

    def test_the_status_keeps_the_last_fifty_newest_last(self):
        core, clock, log = make_core("static")
        for i in range(60):
            clock.add(1.0)
            core.set_policy("round_robin" if i % 2 == 0 else "static")
        shown = core.status()["decisions"]
        self.assertEqual(len(shown), 50)
        self.assertEqual(shown[-1]["reason"], "round_robin -> static")
        self.assertEqual(shown, log[-50:])
        self.assertEqual(len(log), 60)

    def test_the_counts_stay_right_under_threads(self):
        core, clock, log = make_core("adaptive")

        def work():
            for i in range(1000):
                pool = "forecast" if i % 2 else "llm"
                choice = core.route(pool, key="session-%d" % (i % 7))
                core.finish(pool, choice.backend, True, latency_ms=10.0 + i % 5)
        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        state = core.status()
        self.assertEqual(state["counters"]["routed"], 4000)
        for pool in state["pools"].values():
            self.assertEqual(sum(b["inflight"] for b in pool["backends"]), 0)
            self.assertEqual(sum(b["routed"] for b in pool["backends"]), 2000)
        self.assertEqual(len(log), 4000 + state["counters"]["switches"])

    def test_a_broken_log_does_not_stop_routing(self):
        def broken(decision):
            raise IOError("disk full")
        core = R.Core(two_by_two(), now=Clock(), sink=broken)
        self.assertEqual(core.route("forecast").backend, "A")


FORECAST_METRICS = {
    "node": "A", "ts": 1794550000.0, "inflight": 1, "queued": 3, "queued_units": 120,
    "served_total": 120, "errors_total": 0, "units_total": 5400, "ms_per_unit": 10.2,
    "latency_ms": {"p50": 210.0, "p95": 900.0, "ewma": 260.0},
    "gpu": {"util_pct": 96, "temp_c": 70, "power_w": 60.0, "proc_mem_mb": 7115},
    "host": {"mem_available_gb": 101.2, "load1": 1.9},
    "busy_other": False, "slowdown": 1.0}

LLAMACPP_METRICS = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 3
"""

VLLM_METRICS = """# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-7B-Instruct"} 5.0
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="Qwen/Qwen2.5-7B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="Qwen/Qwen2.5-7B-Instruct"} 17.0
"""


class ProbeParsingTests(unittest.TestCase):

    def test_forecast_metrics_json(self):
        self.assertEqual(R.parse_forecast_metrics(FORECAST_METRICS), {
            "inflight": 1, "queued": 3, "queued_units": 120, "gpu_util_pct": 96,
            "busy_other": False, "slowdown": 1.0, "mem_available_gb": 101.2, "ms_per_unit": 10.2})

    def test_forecast_metrics_with_nulls_and_gaps(self):
        nothing = dict.fromkeys(("inflight", "queued", "queued_units", "gpu_util_pct", "busy_other",
                                 "slowdown", "mem_available_gb", "ms_per_unit"))
        self.assertEqual(R.parse_forecast_metrics({}), nothing)
        self.assertEqual(R.parse_forecast_metrics(None), nothing)
        self.assertEqual(R.parse_forecast_metrics(["not", "an", "object"]), nothing)
        self.assertEqual(R.parse_forecast_metrics({
            "inflight": None, "queued": None, "queued_units": None, "ms_per_unit": None,
            "gpu": None, "host": {"mem_available_gb": None}, "busy_other": None,
            "slowdown": "fast"}), nothing)
        some = R.parse_forecast_metrics({"queued": "7", "gpu": {"util_pct": None}, "busy_other": True})
        self.assertEqual((some["queued"], some["gpu_util_pct"], some["busy_other"]), (7.0, None, True))

    def test_llamacpp_metrics_text(self):
        self.assertEqual(R.parse_llm_metrics(LLAMACPP_METRICS, "llamacpp"), {"queued": 3, "running": 1})
        labelled = ('llamacpp:requests_processing{model="qwen 2.5 {q4}",slot="0"} 2\n'
                    'llamacpp:requests_deferred{model="qwen 2.5 {q4}"} 0 1794550000000\n')
        self.assertEqual(R.parse_llm_metrics(labelled, "llamacpp"), {"queued": 0, "running": 2})

    def test_vllm_metrics_text_with_labels(self):
        self.assertEqual(R.parse_llm_metrics(VLLM_METRICS, "vllm"), {"queued": 5, "running": 2})
        # The wrong engine name still finds the numbers.
        self.assertEqual(R.parse_llm_metrics(VLLM_METRICS, "llamacpp"), {"queued": 5, "running": 2})
        self.assertEqual(R.parse_llm_metrics(LLAMACPP_METRICS, "vllm"), {"queued": 3, "running": 1})

    def test_missing_metrics_give_null(self):
        self.assertEqual(R.parse_llm_metrics("", "vllm"), {"queued": None, "running": None})
        only_running = "vllm:num_requests_running 4\n"
        self.assertEqual(R.parse_llm_metrics(only_running, "vllm"), {"queued": None, "running": 4})
        junk = "\n\n<html>404</html>\nnot a metric line at all\nname_only\nx{broken 1\nvalue_is NaN\n"
        self.assertEqual(R.parse_llm_metrics(junk, "llamacpp"), {"queued": None, "running": None})

    def test_prometheus_lines(self):
        text = ('a_total 3\n'
                'b{x="1"} 1.5\n'
                'b{x="2"} 2.5\n'          # the same name twice: added up
                '  c{path="/v1/{id}", note="a } b"}   7   1794550000\n'
                '# d 9\n')
        self.assertEqual(R.parse_prometheus(text), {"a_total": 3, "b": 4.0, "c": 7})


class FanoutPlanTests(unittest.TestCase):

    def test_split_in_proportion_to_one_over_score(self):
        self.assertEqual(R.split_members(20, [0.1, 0.1]), [10, 10])
        self.assertEqual(R.split_members(20, [0.1, 0.3]), [15, 5])     # 3x the wait, a third of the work
        self.assertEqual(R.split_members(7, [1.0, 2.0, 4.0]), [4, 2, 1])
        self.assertEqual(R.split_members(40, [0.5]), [40])
        self.assertEqual(R.split_members(1, [0.2, 0.2]), [1, 0])       # the tie goes to the first
        self.assertEqual(R.split_members(1, [0.3, 0.2]), [0, 1])       # else to the better score
        self.assertEqual(R.split_members(3, [1.0, 1.0, 1.0, 1.0]), [1, 1, 1, 0])
        self.assertEqual(R.split_members(10, [0.01, 5.0]), [10, 0])    # a share under half a member
        for total in range(1, 41):
            for scores in ([0.2, 0.2], [0.2, 0.37], [0.11, 0.2, 0.9], [3.0, 0.001]):
                shares = R.split_members(total, scores)
                self.assertEqual(sum(shares), total)
                self.assertTrue(all(s >= 0 for s in shares))

    def test_plan_fanout(self):
        core, clock, log = make_core("static")
        plan = core.plan_fanout("forecast", 20)
        self.assertEqual(plan, [("A", "http://127.0.0.1:1", 10), ("B", "http://127.0.0.1:2", 10)])
        self.assertEqual((log[-1]["kind"], log[-1]["backend"]), ("fanout", "A+B"))
        self.assertEqual(log[-1]["reason"], "20 members: A 10, B 10")
        self.assertEqual(kinds(log), ["fanout"])             # one decision for the whole ensemble
        self.assertEqual(core.counters["routed"], 1)
        self.assertEqual([backend(core, "forecast", n).inflight for n in "AB"], [1, 1])
        core.finish("forecast", "A", True, latency_ms=100.0)
        core.finish("forecast", "B", True, latency_ms=300.0)
        self.assertEqual([p[2] for p in core.plan_fanout("forecast", 20)], [15, 5])
        self.assertEqual([p[2] for p in core.plan_fanout("forecast", 1)], [1])
        take = core.set_drain("B", True)
        self.assertEqual(take, ["forecast", "llm"])
        self.assertEqual(core.plan_fanout("forecast", 20), [("A", "http://127.0.0.1:1", 20)])


def canned_answer(node, members, offset, mean, low, high, t2m=None, truth=None, queue=1.0, gpu=10.0):
    """A forecast server's answer for one shard, with two lead times."""
    return {"request_id": "abc", "node": node, "model": "afno-19m", "init_time": "2026-11-13T00",
            "members": members, "member_offset": offset, "leads_h": [6, 12],
            "valid_times": ["2026-11-13T06", "2026-11-13T12"],
            "point": {"lat": 41.9602, "lon": -87.9316, "t2m_c": t2m,
                      "t2m_c_mean": mean, "t2m_c_min": low, "t2m_c_max": high,
                      "wind10_ms_mean": [5.0, 6.0], "msl_hpa_mean": [1010.0, 1011.0],
                      "truth_t2m_c": truth},
            "units": members * 2, "timing_ms": {"queue": queue, "gpu": gpu, "total": queue + gpu}}


def canned_shard(node, members, offset, *args, **kw):
    return {"backend": node, "members": members, "member_offset": offset,
            "answer": canned_answer(node, members, offset, *args, **kw)}


class MergeTests(unittest.TestCase):

    def test_merge_two_shards(self):
        merged = R.merge_shards([
            canned_shard("B", 8, 12, [4.0, 5.0], [0.5, 3.0], [5.0, 9.0],
                         truth=[None, 2.5], queue=7.0, gpu=80.0),
            canned_shard("A", 12, 0, [2.0, 3.0], [1.0, 2.0], [3.0, 4.0],
                         t2m=[1.5, 2.5], truth=[1.4, None], queue=1.0, gpu=120.0)])
        self.assertEqual((merged["members"], merged["member_offset"]), (20, 0))
        self.assertEqual(merged["node"], "A+B")
        self.assertEqual(merged["request_id"], "abc")
        self.assertEqual(merged["leads_h"], [6, 12])
        point = merged["point"]
        self.assertEqual(point["t2m_c"], [1.5, 2.5])                  # from the shard with member 0
        self.assertEqual(point["t2m_c_mean"], [2.8, 3.8])             # (12 x 2 + 8 x 4) / 20
        self.assertEqual(point["t2m_c_min"], [0.5, 2.0])              # min of mins
        self.assertEqual(point["t2m_c_max"], [5.0, 9.0])              # max of maxes
        self.assertEqual(point["wind10_ms_mean"], [5.0, 6.0])
        self.assertEqual(point["msl_hpa_mean"], [1010.0, 1011.0])
        self.assertEqual(point["truth_t2m_c"], [1.4, 2.5])            # from whichever shard has it
        self.assertEqual((point["lat"], point["lon"]), (41.9602, -87.9316))
        self.assertEqual(merged["units"], 40)
        self.assertEqual(merged["timing_ms"], {"queue": 7.0, "gpu": 200.0, "total": None})

    def test_merge_without_member_zero(self):
        merged = R.merge_shards([canned_shard("A", 5, 5, [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]),
                                 canned_shard("B", 15, 10, [4.0, 6.0], [0.0, 1.5], [9.0, 3.5])])
        self.assertEqual((merged["members"], merged["member_offset"]), (20, 5))
        self.assertIsNone(merged["point"]["t2m_c"])
        self.assertEqual(merged["point"]["t2m_c_mean"], [3.5, 5.0])
        self.assertIsNone(merged["point"]["truth_t2m_c"])

    def test_merge_one_shard_and_nulls(self):
        one = canned_shard("A", 20, 0, [2.0, 3.0], [1.0, 2.0], [3.0, 4.0], t2m=[1.5, 2.5])
        merged = R.merge_shards([one])
        self.assertEqual(merged["node"], "A")
        self.assertEqual(merged["point"]["t2m_c_mean"], [2.0, 3.0])
        self.assertEqual(merged["point"]["t2m_c"], [1.5, 2.5])
        gaps = R.merge_shards([canned_shard("A", 10, 0, [2.0, None], None, [3.0, 4.0], t2m=[1.5, 2.5]),
                               canned_shard("B", 30, 10, [4.0, 6.0], [0.5, 0.5], [None, 5.0])])
        self.assertEqual(gaps["point"]["t2m_c_mean"], [3.5, 6.0])     # a null is left out of the mean
        self.assertEqual(gaps["point"]["t2m_c_min"], [0.5, 0.5])
        self.assertEqual(gaps["point"]["t2m_c_max"], [3.0, 5.0])


class ConfigTests(unittest.TestCase):

    def test_the_example_config_loads(self):
        with open(os.path.join(REPO, "router", "config.example.json")) as f:
            cfg = R.make_config(json.load(f))
        self.assertEqual((cfg["port"], cfg["policy"], cfg["probe_s"], cfg["fail_n"]), (8100, "adaptive", 1.0, 2))
        self.assertEqual(cfg["adaptive"], {"hysteresis": 0.25, "dwell_s": 3.0, "cooldown_s": 10.0,
                                           "spill_inflight": 0})
        self.assertEqual(cfg["adaptive"], R.DEFAULTS["adaptive"])     # the file and the built-in defaults agree
        self.assertEqual(cfg["score"], R.DEFAULTS["score"])
        self.assertEqual(cfg["heal_backoff_s"], [5, 15, 60])
        forecast, llm = cfg["pools"]["forecast"], cfg["pools"]["llm"]
        self.assertEqual([(b["name"], b["url"], b["probe"]) for b in forecast["backends"]],
                         [("A", "http://127.0.0.1:8101", "forecast"), ("B", "http://127.0.0.1:8102", "forecast")])
        self.assertEqual(forecast["backends"][0]["heal"]["http"]["json"], {"mode": "ok"})
        self.assertEqual((llm["first_byte_timeout_s"], llm["affinity"], llm["affinity_break"]), (600, True, 3.0))
        self.assertEqual(llm["backends"][1]["heal"], {"cmd": ["echo", "restart B's LLM here"]})
        core = R.Core(cfg, now=Clock())
        self.assertEqual(core.route("forecast").url, "http://127.0.0.1:8101")

    def test_defaults_are_filled_in(self):
        cfg = R.make_config({"pools": {"llm": {"backends": [{"name": "A", "url": "http://h:8000"}]}},
                             "adaptive": {"dwell_s": 1.0}})
        self.assertEqual(cfg["port"], 8100)
        self.assertEqual(cfg["adaptive"]["dwell_s"], 1.0)
        self.assertEqual(cfg["adaptive"]["hysteresis"], 0.25)
        self.assertEqual(cfg["pools"]["llm"]["backends"][0]["probe"], "llamacpp")
        self.assertEqual(cfg["pools"]["forecast"]["backends"], [])
        self.assertEqual(R.DEFAULTS["adaptive"]["dwell_s"], 3.0)      # the defaults are not touched

    def test_bad_configs_say_what_is_wrong(self):
        def pools(*backends):
            return {"pools": {"forecast": {"backends": list(backends)}}}
        good = {"name": "A", "url": "http://h:8101"}
        bad = [
            ({"policy": "fastest"}, "unknown policy"),
            (pools(good, dict(good)), "used twice"),
            (pools({"name": "A", "url": "h:8101"}), "http://host:port"),
            (pools({"name": "A", "url": "https://h:8101"}), "http://host:port"),
            (pools({"name": "A", "url": "http://h:port"}), "http://host:port"),
            (pools({"name": "A"}), "needs a name and a url"),
            (pools(dict(good, probe="ping")), "unknown probe"),
            (pools(dict(good, heal={"shell": "rm -rf /"})), "heal must be"),
            (pools(dict(good, heal={"cmd": "systemctl restart x"})), "heal must be"),
            ({"pools": {"gpu": {"backends": []}}}, "unknown pool"),
            ({"heal_backoff_s": []}, "heal_backoff_s"),
        ]
        for raw, words in bad:
            with self.assertRaises(ValueError) as caught:
                R.make_config(raw)
            self.assertIn(words, str(caught.exception))

    def test_backend_list_from_the_command_line(self):
        self.assertEqual(R.parse_backend_list("A=http://h1:8101, B=http://h2:8101", "forecast"), [
            {"name": "A", "url": "http://h1:8101", "probe": "forecast"},
            {"name": "B", "url": "http://h2:8101", "probe": "forecast"}])
        with self.assertRaises(ValueError):
            R.parse_backend_list("http://h1:8101", "forecast")

    def test_paths_and_pools(self):
        for path in ("/v1/forecast", "/v1/ensemble", "/v1/state", "/v1/info"):
            self.assertEqual(R.pool_for_path(path), "forecast")
        for path in ("/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings",
                     "/v1/models/some-model"):
            self.assertEqual(R.pool_for_path(path), "llm")
        for path in ("/", "/v1", "/v1/forecasts", "/metrics", "/health", "/admin/fail"):
            self.assertIsNone(R.pool_for_path(path))


# ---------------------------------------------------------------- helpers for the HTTP tests

def reply(h, status, obj, headers=()):
    """Answer a fake backend's request with JSON."""
    data = json.dumps(obj).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    for name, value in headers:
        h.send_header(name, value)
    h.end_headers()
    if h.command != "HEAD":
        h.wfile.write(data)


class FakeBackend(object):
    """A tiny HTTP server on a free port.

    It answers /health and /metrics itself. For any other path a test sets
    routes[path] = function(handler, body). seen lists those other requests.
    """

    def __init__(self, name="A"):
        self.name = name
        self.health_status = 200
        self.metrics = None          # a dict is sent as JSON, a str as text, None gives 404
        self.routes = {}
        self.seen = []               # (method, path with query, headers, body)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                owner.handle(self)

            do_POST = do_HEAD = do_GET

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 256

            def handle_error(self, request, client_address):
                pass                 # the router hanging up on a fake is part of some tests

        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = "http://127.0.0.1:%d" % self.port
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.02,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2.0)

    def handle(self, h):
        length = int(h.headers.get("Content-Length") or 0)
        body = h.rfile.read(length) if length else b""
        path = h.path.split("?")[0]
        if path == "/health":
            ok = self.health_status == 200
            reply(h, self.health_status, {"status": "ok" if ok else "loading", "node": self.name, "ready": ok})
        elif path == "/metrics":
            if self.metrics is None:
                reply(h, 404, {"error": "no metrics here"})
            elif isinstance(self.metrics, str):
                data = self.metrics.encode("utf-8")
                h.send_response(200)
                h.send_header("Content-Type", "text/plain; version=0.0.4")
                h.send_header("Content-Length", str(len(data)))
                h.end_headers()
                h.wfile.write(data)
            else:
                reply(h, 200, self.metrics)
        else:
            headers = dict((k.lower(), v) for k, v in h.headers.items())
            self.seen.append((h.command, h.path, headers, body))
            route = self.routes.get(path)
            if route is None:
                reply(h, 404, {"error": "fake backend %s has no %s" % (self.name, path)})
            else:
                route(h, body)

    def answer_with(self, path, status=200, **fields):
        """Make `path` answer a fixed JSON object that also names this backend."""
        def route(h, body):
            reply(h, status, dict(fields, node=self.name))
        self.routes[path] = route


def spec(name, where, probe="http", **more):
    """A backend entry for the config. `where` is a FakeBackend or a URL."""
    url = where if isinstance(where, str) else where.url
    return dict(more, name=name, url=url, probe=probe)


def closed_port():
    """A port on which nobody listens: connecting to it is refused."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def call(port, method, path, body=None, headers=None, timeout=5.0):
    """One request. Returns (status, headers with lower-case names, body bytes)."""
    send = dict(headers or {})
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode("utf-8")
        send.setdefault("Content-Type", "application/json")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=send)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict((k.lower(), v) for k, v in resp.getheaders()), data
    finally:
        conn.close()


def status_of(app):
    """GET /router/status, as a dict."""
    status, headers, data = call(app.port, "GET", "/router/status")
    assert status == 200, status
    return json.loads(data.decode("utf-8"))


def row(app, pool, name):
    """One backend's entry in /router/status."""
    for entry in status_of(app)["pools"][pool]["backends"]:
        if entry["name"] == name:
            return entry
    raise AssertionError("no backend %s in pool %s" % (name, pool))


class RouterCase(unittest.TestCase):
    """Base class: fake backends and a real router inside the test process."""

    def backend(self, name="A"):
        fake = FakeBackend(name)
        self.addCleanup(fake.stop)
        return fake

    def router(self, forecast=(), llm=(), first_byte_timeout_s=2.0, affinity_break=3.0, **over):
        """Start a router on a free port. Short timers, static policy, no log file."""
        raw = {"decision_log": "", "policy": "static", "probe_s": 0.05, "connect_timeout_s": 0.5,
               "recover_s": 0.2,
               "pools": {"forecast": {"first_byte_timeout_s": first_byte_timeout_s,
                                      "backends": list(forecast)},
                         "llm": {"first_byte_timeout_s": first_byte_timeout_s, "affinity": True,
                                 "affinity_break": affinity_break, "backends": list(llm)}}}
        raw.update(over)
        app = R.Router(R.make_config(raw))
        app.start(port=0)
        self.addCleanup(app.stop)
        return app

    def wait_for(self, check, what, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = check()
            if value:
                return value
            time.sleep(0.01)
        self.fail("timed out waiting for " + what)


class ProxyTests(RouterCase):

    def test_json_post_goes_through_with_the_right_headers(self):
        a = self.backend("A")

        def forecast(h, body):
            reply(h, 200, {"node": "A", "echo": json.loads(body.decode("utf-8")), "path": h.path},
                  [("X-From-Backend", "yes"), ("Keep-Alive", "timeout=5")])
        a.routes["/v1/forecast"] = forecast
        app = self.router(forecast=[spec("A", a)])
        sent = {"init": "latest", "lead_hours": 24, "members": 1}
        status, headers, data = call(app.port, "POST", "/v1/forecast?trace=1", sent,
                                     {"X-Trace": "t-1", "Connection": "keep-alive, X-Hop",
                                      "X-Hop": "not for the backend"})
        self.assertEqual(status, 200)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(answer["echo"], sent)
        self.assertEqual(answer["path"], "/v1/forecast?trace=1")       # path and query as sent
        self.assertEqual(headers["x-backend"], "A")
        self.assertEqual(headers["x-router-policy"], "static")
        self.assertEqual(headers["x-router-retries"], "0")
        self.assertNotIn("x-router-affinity", headers)                 # only the llm pool has it
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["x-from-backend"], "yes")             # end-to-end headers pass
        self.assertEqual(headers["content-length"], str(len(data)))
        self.assertNotIn("transfer-encoding", headers)
        self.assertNotIn("keep-alive", headers)                        # hop-by-hop: dropped
        self.assertEqual(headers["access-control-allow-origin"], "*")
        method, path, got, body = a.seen[0]
        self.assertEqual((method, path), ("POST", "/v1/forecast?trace=1"))
        self.assertEqual(json.loads(body.decode("utf-8")), sent)
        self.assertEqual(got["x-trace"], "t-1")
        self.assertEqual(got["content-type"], "application/json")
        self.assertEqual(got["content-length"], str(len(body)))
        self.assertEqual(got["host"], "127.0.0.1:%d" % a.port)
        self.assertEqual(got["connection"], "close")                   # the router's own, not the client's
        self.assertNotIn("x-hop", got)                                 # named in Connection: dropped
        state = status_of(app)
        self.assertEqual(state["counters"], {"routed": 1, "failovers": 0, "switches": 0,
                                             "heals": 0, "errors": 0})
        entry = state["pools"]["forecast"]["backends"][0]
        self.assertEqual((entry["routed"], entry["inflight"], entry["failures"]), (1, 0, 0))
        self.assertIsInstance(entry["latency_ewma_ms"], float)
        routes = [d for d in state["decisions"] if d["kind"] == "route"]
        self.assertEqual((routes[0]["pool"], routes[0]["backend"], routes[0]["reason"]),
                         ("forecast", "A", "first healthy"))

    def test_a_forecast_in_flight_counts_with_its_units(self):
        a = self.backend("A")
        a.metrics = dict(FORECAST_METRICS, queued_units=0)       # the fake never shows a queue
        release = threading.Event()

        def forecast(h, body):
            release.wait(5.0)
            reply(h, 200, {"node": "A"})
        a.routes["/v1/forecast"] = forecast
        a.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", a, "forecast")])
        self.wait_for(lambda: row(app, "forecast", "A")["probe_data"].get("ms_per_unit"), "a forecast probe")
        self.assertAlmostEqual(row(app, "forecast", "A")["score"], 0.0102, places=4)       # idle: 1 x 10.2 ms
        body = {"init": "latest", "lead_hours": 120, "members": 20}                        # 20 x 20 = 400 units
        thread = threading.Thread(target=call, args=(app.port, "POST", "/v1/forecast", body))
        thread.start()
        try:
            self.wait_for(lambda: row(app, "forecast", "A")["inflight"] == 1, "the forecast to be in flight")
            self.assertAlmostEqual(row(app, "forecast", "A")["score"], 401 * 0.0102, places=4)
            self.assertEqual(call(app.port, "GET", "/v1/info")[0], 200)      # /v1/info costs no units
            time.sleep(0.15)                                                 # a few probes go by
            self.assertAlmostEqual(row(app, "forecast", "A")["score"], 401 * 0.0102, places=4)
        finally:
            release.set()
            thread.join(5.0)
        self.wait_for(lambda: row(app, "forecast", "A")["inflight"] == 0, "the forecast to end")
        self.assertAlmostEqual(row(app, "forecast", "A")["score"], 0.0102, places=4)

    def test_get_and_keep_alive(self):
        a = self.backend("A")
        a.answer_with("/v1/info", model="afno-19m")
        app = self.router(forecast=[spec("A", a)])
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5.0)
        self.addCleanup(conn.close)
        for _ in range(3):                                 # three requests on one connection
            conn.request("GET", "/v1/info")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(resp.read().decode("utf-8")), {"model": "afno-19m", "node": "A"})
        self.assertEqual([(m, p, b) for m, p, _, b in a.seen], [("GET", "/v1/info", b"")] * 3)
        self.assertNotIn("content-length", a.seen[0][2])   # a GET carries no body

    def test_head_requests_and_old_clients(self):
        a = self.backend("A")
        a.answer_with("/v1/models", object="list")

        def until_close(h, body):                          # no Content-Length: the close ends it
            h.send_response(200)
            h.send_header("Content-Type", "text/plain")
            h.send_header("Connection", "close")
            h.end_headers()
            h.wfile.write(b"one ")
            h.wfile.flush()
            h.wfile.write(b"two")
        a.routes["/v1/completions"] = until_close
        app = self.router(llm=[spec("A", a)])
        # HEAD: the headers a GET would get, and no body. The connection stays usable.
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5.0)
        self.addCleanup(conn.close)
        conn.request("HEAD", "/v1/models")
        resp = conn.getresponse()
        self.assertEqual((resp.status, resp.read()), (200, b""))
        self.assertEqual(resp.getheader("X-Backend"), "A")
        self.assertGreater(int(resp.getheader("Content-Length")), 0)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        self.assertEqual(json.loads(resp.read().decode("utf-8")), {"object": "list", "node": "A"})
        self.assertEqual([m for m, _, _, _ in a.seen], ["HEAD", "GET"])
        # An HTTP/1.0 client cannot read chunks: it gets the plain bytes, then the close.
        sock = socket.create_connection(("127.0.0.1", app.port), timeout=5.0)
        self.addCleanup(sock.close)
        sock.sendall(b"POST /v1/completions HTTP/1.0\r\nContent-Length: 2\r\n\r\n{}")
        raw = b""
        while True:
            piece = sock.recv(65536)
            if not piece:
                break
            raw += piece
        head, _, body = raw.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 200"), head)
        self.assertNotIn(b"chunked", head.lower())
        self.assertIn(b"connection: close", head.lower())
        self.assertEqual(body, b"one two")

    def test_client_errors_and_500_are_passed_on_untouched(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/forecast", status=400, error="lead_hours must be a multiple of 6")
        a.answer_with("/v1/info", status=500, error="the model fell over")
        b.answer_with("/v1/forecast")
        b.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        status, headers, data = call(app.port, "POST", "/v1/forecast", {"lead_hours": 7})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data.decode("utf-8"))["error"], "lead_hours must be a multiple of 6")
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("A", "0"))
        status, headers, data = call(app.port, "GET", "/v1/info")
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(data.decode("utf-8"))["error"], "the model fell over")
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("A", "0"))
        self.assertEqual(b.seen, [])                       # no retry for either
        state = status_of(app)
        self.assertEqual((state["counters"]["failovers"], state["counters"]["errors"]), (0, 1))
        entry = state["pools"]["forecast"]["backends"][0]
        self.assertEqual((entry["healthy"], entry["failures"]), (True, 0))

    def test_unknown_paths_and_bad_requests(self):
        app = self.router()
        status, headers, data = call(app.port, "GET", "/v2/anything")
        self.assertEqual(status, 404)
        self.assertIn("no pool serves /v2/anything", json.loads(data.decode("utf-8"))["error"])
        status, headers, data = call(app.port, "POST", "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 503)                      # the llm pool is empty
        self.assertEqual(json.loads(data.decode("utf-8")), {"error": "no healthy backend"})
        status, headers, data = call(app.port, "POST", "/v1/forecast", b"abc",
                                     {"Transfer-Encoding": "chunked"})
        self.assertEqual(status, 411)
        self.assertIn("Content-Length", json.loads(data.decode("utf-8"))["error"])

    def test_llm_requests_carry_the_affinity_header(self):
        a, b = self.backend("A"), self.backend("B")
        for fake in (a, b):
            fake.answer_with("/v1/chat/completions", object="chat.completion")
        # The fakes answer in well under a millisecond, so their latency EWMAs can
        # differ by more than 3x for no reason. This test is not about the break.
        app = self.router(llm=[spec("A", a), spec("B", b)], policy="round_robin", affinity_break=1e6)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        homes = {}
        for session in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"):
            seen = set()
            for _ in range(3):
                status, headers, data = call(app.port, "POST", "/v1/chat/completions", body,
                                             {"X-Session-Id": session})
                self.assertEqual((status, headers["x-router-affinity"]), (200, "hit"))
                self.assertEqual(json.loads(data.decode("utf-8"))["node"], headers["x-backend"])
                seen.add(headers["x-backend"])
            self.assertEqual(len(seen), 1, "session %s moved between backends" % session)
            homes[session] = seen.pop()
        self.assertEqual(set(homes.values()), {"A", "B"})  # sessions spread over both
        # The body's "user" field works like the header.
        status, headers, data = call(app.port, "POST", "/v1/chat/completions", dict(body, user="s1"))
        self.assertEqual((headers["x-router-affinity"], headers["x-backend"]), ("hit", homes["s1"]))
        # No key: the policy decides, and says so.
        picks = []
        for _ in range(2):
            status, headers, data = call(app.port, "POST", "/v1/chat/completions", body)
            self.assertEqual(headers["x-router-affinity"], "none")
            picks.append(headers["x-backend"])
        self.assertEqual(sorted(picks), ["A", "B"])

    def test_failover_on_a_refused_connection(self):
        b = self.backend("B")
        b.answer_with("/v1/forecast", ok=True)
        dead = "http://127.0.0.1:%d" % closed_port()
        # fail_n is huge so that A stays in the list and every request meets it first.
        app = self.router(forecast=[spec("A", dead), spec("B", b)], fail_n=1000)
        status, headers, data = call(app.port, "POST", "/v1/forecast", {"members": 1})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data.decode("utf-8")), {"ok": True, "node": "B"})
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("B", "1"))
        state = status_of(app)
        self.assertEqual((state["counters"]["routed"], state["counters"]["failovers"],
                          state["counters"]["errors"]), (1, 1, 0))
        failover = [d for d in state["decisions"] if d["kind"] == "failover"]
        self.assertEqual(len(failover), 1)
        self.assertEqual((failover[0]["backend"], failover[0]["retries"]), ("B", 1))
        self.assertIn("A failed: connection refused", failover[0]["reason"])
        entry = state["pools"]["forecast"]["backends"][0]
        self.assertGreaterEqual(entry["failures"], 1)
        self.assertIn("refused", entry["last_error"])
        self.assertEqual(len(b.seen), 1)

    def test_failover_on_a_503(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/forecast", status=503, error="loading")
        b.answer_with("/v1/forecast", ok=True)
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000)
        for expected in (1, 2):
            status, headers, data = call(app.port, "POST", "/v1/forecast", {"members": 1})
            self.assertEqual(status, 200)
            self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("B", "1"))
            self.assertEqual(status_of(app)["counters"]["failovers"], expected)
        self.assertEqual((len(a.seen), len(b.seen)), (2, 2))
        self.assertEqual(row(app, "forecast", "A")["last_error"], "status 503")

    def test_failures_from_requests_alone_mark_a_backend_unhealthy(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/forecast", status=502, error="bad gateway")
        b.answer_with("/v1/forecast", ok=True)
        a.metrics = b.metrics = FORECAST_METRICS
        # No probe helps here: one at the start, then none for an hour.
        app = self.router(forecast=[spec("A", a, "forecast"), spec("B", b, "forecast")],
                          probe_s=3600, fail_n=2)
        for name in "AB":
            self.wait_for(lambda: row(app, "forecast", name)["probe_data"].get("ms_per_unit"),
                          "the first probe of " + name)
        retries = [call(app.port, "POST", "/v1/forecast", {})[1]["x-router-retries"] for _ in range(4)]
        self.assertEqual(retries, ["1", "1", "0", "0"])    # after 2 failures A is skipped
        self.assertEqual(len(a.seen), 2)
        self.assertFalse(row(app, "forecast", "A")["healthy"])
        self.assertIn("unhealthy", [d["kind"] for d in status_of(app)["decisions"]])

    def test_a_503_from_the_retry_is_passed_on(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/forecast", status=503, error="A is loading")
        b.answer_with("/v1/forecast", status=503, error="B is loading")
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000)
        status, headers, data = call(app.port, "POST", "/v1/forecast", {"members": 1})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(data.decode("utf-8")), {"error": "B is loading", "node": "B"})
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("B", "1"))
        self.assertEqual((len(a.seen), len(b.seen)), (1, 1))            # once each, no third try
        self.assertEqual(status_of(app)["counters"]["errors"], 1)

    def test_no_healthy_backend_gives_503(self):
        a = self.backend("A")
        a.answer_with("/v1/forecast", status=503, error="loading")
        app = self.router(forecast=[spec("A", a)], fail_n=1000)
        status, headers, data = call(app.port, "POST", "/v1/forecast", {"members": 1})
        self.assertEqual(status, 503)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(answer["error"], "no healthy backend")
        self.assertIn("A failed: status 503", answer["detail"])
        self.assertEqual(headers["access-control-allow-origin"], "*")
        state = status_of(app)
        self.assertEqual((state["counters"]["failovers"], state["counters"]["errors"]), (0, 1))

    def test_two_dead_backends_give_502(self):
        dead = ["http://127.0.0.1:%d" % closed_port() for _ in range(2)]
        app = self.router(forecast=[spec("A", dead[0]), spec("B", dead[1])], fail_n=1000)
        status, headers, data = call(app.port, "POST", "/v1/forecast", {"members": 1})
        self.assertEqual(status, 502)
        detail = json.loads(data.decode("utf-8"))["detail"]
        self.assertIn("A failed: connection refused", detail)
        self.assertIn("B failed: connection refused", detail)
        self.assertEqual(headers["x-router-retries"], "1")

    def test_first_byte_timeout_fails_over(self):
        a, b = self.backend("A"), self.backend("B")
        release = threading.Event()
        self.addCleanup(release.set)
        a.routes["/v1/forecast"] = lambda h, body: (release.wait(5.0), reply(h, 200, {"node": "A"}))
        b.answer_with("/v1/forecast", ok=True)
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000,
                          first_byte_timeout_s=0.3)
        began = time.monotonic()
        status, headers, data = call(app.port, "POST", "/v1/forecast", {"members": 1})
        took = time.monotonic() - began
        self.assertEqual((status, headers["x-backend"], headers["x-router-retries"]), (200, "B", "1"))
        self.assertGreaterEqual(took, 0.3)
        self.assertLess(took, 4.0)
        self.assertEqual(row(app, "forecast", "A")["last_error"], "timed out")

    def test_no_retry_once_bytes_were_passed_on(self):
        a, b = self.backend("A"), self.backend("B")

        def cut_short(h, body):
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", "1000")
            h.end_headers()
            h.wfile.write(b'{"partial":')
            h.wfile.flush()
            h.close_connection = True                      # hang up 989 bytes early
        a.routes["/v1/forecast"] = cut_short
        b.answer_with("/v1/forecast", ok=True)
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000)
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=5.0)
        self.addCleanup(conn.close)
        conn.request("POST", "/v1/forecast", body=b"{}", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)                 # the headers were already on their way
        self.assertEqual((resp.getheader("X-Backend"), resp.getheader("X-Router-Retries")), ("A", "0"))
        with self.assertRaises(http.client.IncompleteRead) as caught:
            resp.read()
        self.assertEqual(caught.exception.partial, b'{"partial":')
        self.assertEqual(b.seen, [])                       # B was never asked
        state = self.wait_for(lambda: status_of(app) if status_of(app)["counters"]["errors"] else None,
                              "the error to be counted")
        self.assertEqual(state["counters"]["failovers"], 0)
        entry = state["pools"]["forecast"]["backends"][0]
        self.assertGreaterEqual(entry["failures"], 1)
        self.assertIn("mid-answer", entry["last_error"])

    def test_a_stream_arrives_piece_by_piece(self):
        for framing in ("chunked", "until close"):
            with self.subTest(backend_framing=framing):
                a = self.backend("A")
                release = threading.Event()
                self.addCleanup(release.set)
                progress = {"last_sent": False}

                def stream(h, body, framing=framing, release=release, progress=progress):
                    def send(piece):
                        if framing == "chunked":
                            piece = b"%x\r\n" % len(piece) + piece + b"\r\n"
                        h.wfile.write(piece)
                        h.wfile.flush()
                    h.send_response(200)
                    h.send_header("Content-Type", "text/event-stream")
                    if framing == "chunked":
                        h.send_header("Transfer-Encoding", "chunked")
                    else:
                        h.send_header("Connection", "close")
                    h.end_headers()
                    send(b"data: one\n\n")
                    release.wait(30.0)                     # the rest waits for the test's go
                    progress["last_sent"] = True
                    send(b"data: two\n\n")
                    if framing == "chunked":
                        h.wfile.write(b"0\r\n\r\n")
                a.routes["/v1/chat/completions"] = stream
                app = self.router(llm=[spec("A", a)])
                conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=10.0)
                self.addCleanup(conn.close)
                conn.request("POST", "/v1/chat/completions", body=b'{"stream": true}',
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()                  # status and headers come at once
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
                self.assertEqual(resp.getheader("Transfer-Encoding"), "chunked")
                self.assertIsNone(resp.getheader("Content-Length"))
                self.assertEqual(resp.getheader("X-Backend"), "A")
                self.assertEqual(resp.getheader("X-Router-Affinity"), "none")
                self.assertEqual(resp.read1(1024), b"data: one\n\n")
                # We hold the first piece and the backend has not sent the last one yet.
                self.assertFalse(progress["last_sent"])
                self.assertEqual(row(app, "llm", "A")["inflight"], 1)
                release.set()
                self.assertEqual(resp.read(), b"data: two\n\n")
                self.wait_for(lambda: row(app, "llm", "A")["inflight"] == 0, "the stream to end")
                self.assertEqual(row(app, "llm", "A")["failures"], 0)

    def test_a_client_that_hangs_up_is_not_a_backend_failure(self):
        a = self.backend("A")
        release, done = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        outcome = {}

        def stream(h, body):
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.send_header("Connection", "close")
            h.end_headers()
            h.wfile.write(b"data: one\n\n")
            h.wfile.flush()
            release.wait(30.0)
            try:
                for _ in range(300):                       # up to 3 s of stream
                    h.wfile.write(b"x" * 1024)
                    h.wfile.flush()
                    time.sleep(0.01)
                outcome["how"] = "sent everything"
            except OSError:
                outcome["how"] = "the router hung up"
            done.set()
        a.routes["/v1/chat/completions"] = stream
        app = self.router(llm=[spec("A", a)])
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=10.0)
        conn.request("POST", "/v1/chat/completions", body=b'{"stream": true}',
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.read1(1024), b"data: one\n\n")
        resp.close()
        conn.close()                                       # the client walks away mid-stream
        release.set()
        self.assertTrue(done.wait(20.0))
        self.assertEqual(outcome["how"], "the router hung up")   # it stopped pulling the stream
        self.wait_for(lambda: row(app, "llm", "A")["inflight"] == 0, "the request to be closed")
        state = status_of(app)
        entry = state["pools"]["llm"]["backends"][0]
        self.assertEqual((entry["healthy"], entry["failures"], entry["last_error"]), (True, 0, None))
        self.assertEqual(state["counters"]["errors"], 0)


class ProbeTests(RouterCase):
    """The probe thread against fake backends."""

    def test_unhealthy_after_failed_probes_and_back_after_a_good_one(self):
        a, b = self.backend("A"), self.backend("B")
        for fake in (a, b):
            fake.metrics = FORECAST_METRICS
            fake.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", a, "forecast"), spec("B", b, "forecast")],
                          fail_n=2, recover_s=0.3)
        self.assertEqual(call(app.port, "GET", "/v1/info")[1]["x-backend"], "A")
        a.health_status = 503                              # A starts answering "loading"
        down_at = time.monotonic()
        self.wait_for(lambda: not row(app, "forecast", "A")["healthy"], "A to be marked unhealthy")
        self.assertEqual(row(app, "forecast", "A")["last_error"], "health answered 503 (loading)")
        headers = call(app.port, "GET", "/v1/info")[1]
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("B", "0"))   # A is skipped
        a.health_status = 200
        self.wait_for(lambda: row(app, "forecast", "A")["healthy"], "A to come back")
        self.assertGreaterEqual(time.monotonic() - down_at, 0.3)     # not before recover_s
        self.assertIsNone(row(app, "forecast", "A")["last_error"])
        seen = [(d["kind"], d["backend"]) for d in status_of(app)["decisions"]]
        self.assertIn(("unhealthy", "A"), seen)
        self.assertIn(("recovered", "A"), seen)
        self.assertLess(seen.index(("unhealthy", "A")), seen.index(("recovered", "A")))
        self.assertEqual(call(app.port, "GET", "/v1/info")[1]["x-backend"], "A")

    def test_probe_numbers_show_in_the_status(self):
        f = self.backend("A")
        f.metrics = dict(FORECAST_METRICS, busy_other=True, slowdown=1.8)
        llama, vllm, plain, bare = [self.backend(n) for n in "ABCD"]
        llama.metrics = LLAMACPP_METRICS
        vllm.metrics = VLLM_METRICS                        # bare has no /metrics at all
        app = self.router(forecast=[spec("A", f, "forecast")],
                          llm=[spec("A", llama, "llamacpp"), spec("B", vllm, "vllm"),
                               spec("C", plain, "http"), spec("D", bare, "llamacpp")])
        self.wait_for(lambda: row(app, "forecast", "A")["probe_data"].get("ms_per_unit"), "a forecast probe")
        entry = row(app, "forecast", "A")
        self.assertEqual(entry["probe_data"], {
            "inflight": 1, "queued": 3, "queued_units": 120, "gpu_util_pct": 96, "busy_other": True,
            "slowdown": 1.8, "mem_available_gb": 101.2, "ms_per_unit": 10.2})
        # (queued_units + 1) x ms_per_unit, times the busy_other penalty. The slowdown is
        # in ms_per_unit already. The 120 units are not ours: no request went through the router.
        self.assertAlmostEqual(entry["score"], 121 * 0.0102 * 1.5, places=3)
        self.wait_for(lambda: "queued" in row(app, "llm", "D")["probe_data"]
                      and row(app, "llm", "A")["probe_data"].get("running") is not None
                      and row(app, "llm", "B")["probe_data"].get("running") is not None, "the llm probes")
        self.assertEqual(row(app, "llm", "A")["probe_data"], {"queued": 3, "running": 1})
        self.assertEqual(row(app, "llm", "B")["probe_data"], {"queued": 5, "running": 2})
        self.assertEqual(row(app, "llm", "C")["probe_data"], {})
        self.assertEqual(row(app, "llm", "D")["probe_data"], {"queued": None, "running": None})
        self.assertTrue(row(app, "llm", "D")["healthy"])   # no /metrics is not a failure
        self.assertAlmostEqual(row(app, "llm", "A")["score"], (0 + 3 + 1) * 0.2)

    def test_probe_backend_says_why(self):
        a = self.backend("A")
        a.metrics = FORECAST_METRICS
        self.assertEqual(R.probe_backend(a.url, "forecast", 0.5),
                         (True, R.parse_forecast_metrics(FORECAST_METRICS), None))
        self.assertEqual(R.probe_backend(a.url, "http", 0.5), (True, {}, None))
        a.health_status = 503
        self.assertEqual(R.probe_backend(a.url, "llamacpp", 0.5),
                         (False, {}, "health answered 503 (loading)"))
        dead = "http://127.0.0.1:%d" % closed_port()
        self.assertEqual(R.probe_backend(dead, "vllm", 0.5), (False, {}, "health: connection refused"))


def forecast_route(fake, fail_first=0, fail_status=503, barrier=None):
    """A fake /v1/forecast whose numbers depend only on the member range asked for."""
    count = {"calls": 0}

    def route(h, body):
        ask = json.loads(body.decode("utf-8"))
        count["calls"] += 1
        if count["calls"] <= fail_first:
            reply(h, fail_status, {"error": "busy"})
            return
        if barrier is not None:
            barrier.wait(3.0)                              # both shards must be here together
        first = ask.get("member_offset", 0) == 0
        reply(h, 200, canned_answer(
            fake.name, ask["members"], ask.get("member_offset", 0),
            mean=[2.0, 3.0] if first else [4.0, 5.0], low=[1.0, 2.0] if first else [0.5, 3.0],
            high=[3.0, 4.0] if first else [5.0, 9.0], t2m=[1.5, 2.5] if first else None,
            truth=[1.4, None] if first else [None, 2.5], queue=1.0, gpu=10.0 * ask["members"]))
    return route


ENSEMBLE = {"init": "latest", "lead_hours": 12, "members": 20,
            "point": {"lat": 41.9602, "lon": -87.9316}, "with_truth": True, "request_id": "abc"}


class EnsembleTests(RouterCase):

    def check_merged(self, answer):
        self.assertEqual((answer["members"], answer["member_offset"]), (20, 0))
        point = answer["point"]
        self.assertEqual(point["t2m_c"], [1.5, 2.5])
        self.assertEqual(point["t2m_c_mean"], [3.0, 4.0])              # 10 members each
        self.assertEqual(point["t2m_c_min"], [0.5, 2.0])
        self.assertEqual(point["t2m_c_max"], [5.0, 9.0])
        self.assertEqual(point["truth_t2m_c"], [1.4, 2.5])
        self.assertEqual(answer["units"], 40)
        self.assertEqual(answer["timing_ms"]["gpu"], 200.0)
        self.assertGreater(answer["timing_ms"]["total"], 0.0)
        self.assertEqual(answer["request_id"], "abc")

    def test_split_run_together_and_merged(self):
        a, b = self.backend("A"), self.backend("B")
        barrier = threading.Barrier(2)
        a.routes["/v1/forecast"] = forecast_route(a, barrier=barrier)
        b.routes["/v1/forecast"] = forecast_route(b, barrier=barrier)
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        status, headers, data = call(app.port, "POST", "/v1/ensemble", ENSEMBLE)
        self.assertEqual(status, 200)
        answer = json.loads(data.decode("utf-8"))
        self.check_merged(answer)
        self.assertEqual(answer["node"], "A+B")
        shards = answer["shards"]
        self.assertEqual([(s["backend"], s["members"], s["member_offset"], s["retried"]) for s in shards],
                         [("A", 10, 0, False), ("B", 10, 10, False)])
        for shard in shards:
            self.assertEqual(sorted(shard), ["backend", "member_offset", "members", "ms", "retried"])
            self.assertGreater(shard["ms"], 0.0)
        self.assertEqual((headers["x-backend"], headers["x-router-policy"], headers["x-router-retries"]),
                         ("A+B", "static", "0"))
        # Each backend got one /v1/forecast: the same request with its own member range.
        for fake, offset in ((a, 0), (b, 10)):
            self.assertEqual(len(fake.seen), 1)
            method, path, _, body = fake.seen[0]
            self.assertEqual((method, path), ("POST", "/v1/forecast"))
            self.assertEqual(json.loads(body.decode("utf-8")), dict(ENSEMBLE, members=10, member_offset=offset))
        state = status_of(app)
        self.assertEqual((state["counters"]["routed"], state["counters"]["failovers"]), (1, 0))
        fanouts = [d for d in state["decisions"] if d["kind"] == "fanout"]
        self.assertEqual(len(fanouts), 1)
        self.assertEqual((fanouts[0]["backend"], fanouts[0]["reason"]), ("A+B", "20 members: A 10, B 10"))
        self.assertEqual([r["inflight"] for r in state["pools"]["forecast"]["backends"]], [0, 0])
        self.assertEqual([r["routed"] for r in state["pools"]["forecast"]["backends"]], [1, 1])

    def test_a_failed_shard_is_run_again_on_the_other_backend(self):
        a, b = self.backend("A"), self.backend("B")
        a.routes["/v1/forecast"] = forecast_route(a, fail_first=1)       # A says 503 once
        b.routes["/v1/forecast"] = forecast_route(b)
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000)
        status, headers, data = call(app.port, "POST", "/v1/ensemble", ENSEMBLE)
        self.assertEqual(status, 200)
        answer = json.loads(data.decode("utf-8"))
        self.check_merged(answer)                                         # nothing is lost
        self.assertEqual(answer["node"], "B")
        self.assertEqual([(s["backend"], s["members"], s["member_offset"], s["retried"])
                          for s in answer["shards"]], [("B", 10, 0, True), ("B", 10, 10, False)])
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("B", "1"))
        self.assertEqual(len(a.seen), 1)
        asked_of_b = sorted(json.loads(body.decode("utf-8"))["member_offset"] for _, _, _, body in b.seen)
        self.assertEqual(asked_of_b, [0, 10])
        state = status_of(app)
        self.assertEqual((state["counters"]["routed"], state["counters"]["failovers"],
                          state["counters"]["errors"]), (1, 1, 0))
        failover = [d for d in state["decisions"] if d["kind"] == "failover"]
        self.assertEqual(len(failover), 1)
        self.assertIn("ensemble shard: A failed (status 503", failover[0]["reason"])
        self.assertEqual([d["kind"] for d in state["decisions"]].count("fanout"), 1)

    def test_shards_count_with_their_units_while_they_run(self):
        a, b = self.backend("A"), self.backend("B")
        release = threading.Event()
        for fake in (a, b):
            inner = forecast_route(fake)

            def held(h, body, inner=inner):
                release.wait(5.0)
                inner(h, body)
            fake.routes["/v1/forecast"] = held
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        thread = threading.Thread(target=call, args=(app.port, "POST", "/v1/ensemble", ENSEMBLE))
        thread.start()
        try:
            self.wait_for(lambda: len(a.seen) == 1 and len(b.seen) == 1, "both shards to arrive")
            # 10 members x 2 steps (lead 12 h) on each backend.
            self.assertEqual([app.core.open_units("forecast", name) for name in "AB"], [20, 20])
        finally:
            release.set()
            thread.join(5.0)
        self.wait_for(lambda: row(app, "forecast", "A")["inflight"] == 0, "the ensemble to end")
        self.assertEqual([app.core.open_units("forecast", name) for name in "AB"], [0, 0])

    def test_shards_from_different_analyses_are_not_merged(self):
        """Node B holds another "latest" than node A. A mix of the two would be no forecast at all."""
        a, b = self.backend("A"), self.backend("B")
        a.routes["/v1/forecast"] = forecast_route(a)

        def another_start(h, body):
            ask = json.loads(body.decode("utf-8"))
            answer = canned_answer("B", ask["members"], ask["member_offset"], [4.0, 5.0], [0.5, 3.0], [5.0, 9.0])
            reply(h, 200, dict(answer, init_time="2019-02-03T18"))
        b.routes["/v1/forecast"] = another_start
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        status, headers, data = call(app.port, "POST", "/v1/ensemble", ENSEMBLE)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(status, 502)
        self.assertIn("different analyses (A from 2026-11-13T00, B from 2019-02-03T18)", answer["error"])
        self.assertIn("POST /v1/state", answer["error"])
        self.assertNotIn("point", answer)
        self.assertEqual([(s["backend"], s["members"]) for s in answer["shards"]], [("A", 10), ("B", 10)])
        self.assertEqual(headers["x-backend"], "A+B")
        self.assertEqual(status_of(app)["counters"]["errors"], 1)

    def test_one_healthy_backend_gets_every_member(self):
        a, b = self.backend("A"), self.backend("B")
        a.routes["/v1/forecast"] = forecast_route(a)
        b.routes["/v1/forecast"] = forecast_route(b)
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        self.assertEqual(call(app.port, "POST", "/router/drain", {"backend": "B", "drain": True})[0], 200)
        status, headers, data = call(app.port, "POST", "/v1/ensemble", dict(ENSEMBLE, members=7))
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual([(s["backend"], s["members"], s["member_offset"]) for s in answer["shards"]],
                         [("A", 7, 0)])
        self.assertEqual((answer["members"], answer["node"], headers["x-backend"]), (7, "A", "A"))
        self.assertEqual(b.seen, [])

    def test_ensemble_errors(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/forecast", status=503, error="loading")
        b.answer_with("/v1/forecast", status=503, error="loading")
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000)
        for bad in (dict(ENSEMBLE, members=0), dict(ENSEMBLE, members="many"), dict(ENSEMBLE, member_offset=-1)):
            status, headers, data = call(app.port, "POST", "/v1/ensemble", bad)
            self.assertEqual(status, 400)
            self.assertIn("whole number", json.loads(data.decode("utf-8"))["error"])
        self.assertEqual(call(app.port, "POST", "/v1/ensemble", b"not json")[0], 400)
        self.assertEqual(a.seen + b.seen, [])
        # Every shard fails on both backends: 502, and the shard list says where it ran.
        status, headers, data = call(app.port, "POST", "/v1/ensemble", ENSEMBLE)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(status, 502)
        self.assertEqual(answer["error"], "an ensemble shard failed twice")
        self.assertEqual(len(answer["shards"]), 2)
        self.assertEqual((len(a.seen), len(b.seen)), (2, 2))
        self.assertEqual([r["inflight"] for r in status_of(app)["pools"]["forecast"]["backends"]], [0, 0])
        # A request the backends reject is not run again: their 400 is passed on.
        a.answer_with("/v1/forecast", status=400, error="lead_hours must be a multiple of 6")
        b.answer_with("/v1/forecast", status=400, error="lead_hours must be a multiple of 6")
        status, headers, data = call(app.port, "POST", "/v1/ensemble", dict(ENSEMBLE, lead_hours=7))
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data.decode("utf-8"))["error"], "lead_hours must be a multiple of 6")
        self.assertEqual((len(a.seen), len(b.seen)), (3, 3))


class StateTests(RouterCase):
    """POST /v1/state must reach every forecast backend, or the nodes hold different "latest" analyses."""

    STATE = {"time": "2026-11-14T00", "npy_b64": "k05VTVBZ"}

    def test_a_new_analysis_goes_to_every_backend(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/state", latest="2026-11-14T00")
        b.answer_with("/v1/state", latest="2026-11-14T00")
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        status, headers, data = call(app.port, "POST", "/v1/state", self.STATE)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(answer["latest"], "2026-11-14T00")
        self.assertEqual(answer["backends"], [
            {"backend": "A", "status": 200, "latest": "2026-11-14T00", "error": None},
            {"backend": "B", "status": 200, "latest": "2026-11-14T00", "error": None}])
        self.assertEqual(answer["left_out"], [])
        self.assertEqual((headers["x-backend"], headers["x-router-retries"]), ("A+B", "0"))
        for fake in (a, b):                                 # each got the body as it was sent
            self.assertEqual(len(fake.seen), 1)
            method, path, sent_headers, body = fake.seen[0]
            self.assertEqual((method, path, json.loads(body.decode("utf-8"))), ("POST", "/v1/state", self.STATE))
        state = status_of(app)
        self.assertEqual(state["counters"]["routed"], 1)
        self.assertEqual([r["inflight"] for r in state["pools"]["forecast"]["backends"]], [0, 0])
        fanout = [d for d in state["decisions"] if d["kind"] == "fanout"]
        self.assertEqual([(d["backend"], d["reason"]) for d in fanout], [("A+B", "an analysis for every node")])

    def test_a_backend_out_of_service_is_named_as_left_out(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/state", latest="2026-11-14T00")
        b.answer_with("/v1/state", latest="2026-11-14T00")
        app = self.router(forecast=[spec("A", a), spec("B", b)])
        app.core.set_drain("B", True)
        status, headers, data = call(app.port, "POST", "/v1/state", self.STATE)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual((status, headers["x-backend"]), (200, "A"))
        self.assertEqual(([row["backend"] for row in answer["backends"]], answer["left_out"]), (["A"], ["B"]))
        self.assertEqual(b.seen, [])
        app.core.set_drain("A", True)                       # nobody left
        status, headers, data = call(app.port, "POST", "/v1/state", self.STATE)
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(data.decode("utf-8"))["left_out"], ["A", "B"])

    def test_a_refused_or_lost_analysis_is_an_error(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/state", latest="2026-11-14T00")
        b.answer_with("/v1/state", status=400, error="time must be at 00, 06, 12 or 18 UTC")
        app = self.router(forecast=[spec("A", a), spec("B", b)], fail_n=1000)
        status, headers, data = call(app.port, "POST", "/v1/state", self.STATE)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(status, 400)                       # the server's own words are passed on
        self.assertEqual(answer["error"], "time must be at 00, 06, 12 or 18 UTC")
        self.assertEqual([row["status"] for row in answer["backends"]], [200, 400])
        # A backend that cannot be reached: the caller must know that only one node has the analysis.
        b.stop()
        status, headers, data = call(app.port, "POST", "/v1/state", self.STATE)
        answer = json.loads(data.decode("utf-8"))
        self.assertEqual(status, 502)
        self.assertEqual(answer["error"], "the analysis did not reach every node")
        self.assertEqual(answer["detail"], "B: connection refused")
        self.assertEqual([(row["backend"], row["status"], row["latest"]) for row in answer["backends"]],
                         [("A", 200, "2026-11-14T00"), ("B", None, None)])
        self.assertEqual(status_of(app)["counters"]["errors"], 1)


class HealTests(RouterCase):

    def test_http_heal_fires_after_heal_after_s_with_back_off(self):
        a = self.backend("A")
        a.health_status = 503                              # down from the start
        calls = []

        def admin_fail(h, body):
            calls.append((time.monotonic(), h.command, h.headers.get("Content-Type"),
                          json.loads(body.decode("utf-8"))))
            reply(h, 200, {"mode": "ok"})
            if len(calls) == 2:                            # the second attempt brings it back,
                timer = threading.Timer(0.15, setattr, (a, "health_status", 200))   # a moment later
                timer.daemon = True
                timer.start()
        a.routes["/admin/fail"] = admin_fail
        heal = {"http": {"url": a.url + "/admin/fail", "method": "POST", "json": {"mode": "ok"}}}
        began = time.monotonic()
        app = self.router(forecast=[spec("A", a, "forecast", heal=heal)], fail_n=2, recover_s=0.1,
                          heal_after_s=0.3, heal_backoff_s=[0.4, 5.0], heal_max_attempts=5)
        self.wait_for(lambda: row(app, "forecast", "A")["healing"], "the first heal attempt")
        self.assertFalse(row(app, "forecast", "A")["healthy"])
        self.wait_for(lambda: row(app, "forecast", "A")["healthy"], "A to come back")
        self.assertEqual(len(calls), 2)
        for when, method, kind, body in calls:
            self.assertEqual((method, kind, body), ("POST", "application/json", {"mode": "ok"}))
        self.assertGreaterEqual(calls[0][0] - began, 0.3)              # not before heal_after_s
        self.assertGreaterEqual(calls[1][0] - calls[0][0], 0.4)        # then the first back-off
        state = status_of(app)
        self.assertEqual(state["counters"]["heals"], 2)
        heals = [d for d in state["decisions"] if d["kind"] == "heal"]
        self.assertEqual([d["backend"] for d in heals], ["A", "A"])
        self.assertEqual(heals[0]["reason"], "attempt 1 of 5 ran: http POST %s/admin/fail answered 200" % a.url)
        self.assertTrue(heals[1]["reason"].startswith("attempt 2 of 5 ran"))
        order = [d["kind"] for d in state["decisions"] if d["kind"] in ("unhealthy", "heal", "recovered")]
        self.assertEqual(order, ["unhealthy", "heal", "heal", "recovered"])
        entry = state["pools"]["forecast"]["backends"][0]
        self.assertEqual((entry["healthy"], entry["healing"]), (True, False))
        time.sleep(0.6)                                    # healthy again: no third attempt
        self.assertEqual(len(calls), 2)

    def test_heal_actions_run_as_configured(self):
        a = self.backend("A")
        a.answer_with("/admin/fail", status=500, error="cannot")
        ok, text = R.run_heal_action({"http": {"url": a.url + "/admin/fail"}})
        self.assertEqual((ok, text), (False, "http POST %s/admin/fail answered 500" % a.url))
        self.assertEqual(a.seen[0][0], "POST")             # POST is the default method
        ok, text = R.run_heal_action({"http": {"url": "http://127.0.0.1:%d/x" % closed_port(), "method": "get"}})
        self.assertFalse(ok)
        self.assertIn("connection refused", text)
        ok, text = R.run_heal_action({"cmd": ["echo", "restart B's LLM here"]})
        self.assertEqual((ok, text), (True, "cmd echo exit 0: restart B's LLM here"))
        script = "import sys; print('no luck ' + 'x' * 500); sys.exit(3)"
        ok, text = R.run_heal_action({"cmd": [sys.executable, "-c", script]})
        self.assertFalse(ok)
        self.assertIn("exit 3: no luck xxx", text)
        self.assertEqual(len(text.split(": ", 1)[1]), 200)             # the output is cut at 200
        ok, text = R.run_heal_action({"cmd": ["/no/such/program", "--now"]})
        self.assertFalse(ok)
        self.assertIn("/no/such/program", text)


class AdminTests(RouterCase):

    def test_status_shape(self):
        f, llm = self.backend("A"), self.backend("A")
        f.metrics, llm.metrics = FORECAST_METRICS, LLAMACPP_METRICS
        f.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", f, "forecast")], llm=[spec("A", llm, "llamacpp")],
                          policy="adaptive")
        self.assertEqual(call(app.port, "GET", "/v1/info")[0], 200)
        self.wait_for(lambda: row(app, "forecast", "A")["probe_data"].get("ms_per_unit")
                      and row(app, "llm", "A")["probe_data"].get("running") is not None, "the probes")
        status, headers, data = call(app.port, "GET", "/router/status")
        self.assertEqual((status, headers["content-type"]), (200, "application/json"))
        state = json.loads(data.decode("utf-8"))
        self.assertEqual(list(state), ["policy", "uptime_s", "counters", "pools", "decisions"])
        self.assertEqual(state["policy"], "adaptive")
        self.assertIsInstance(state["uptime_s"], float)
        self.assertEqual(state["counters"], {"routed": 1, "failovers": 0, "switches": 0,
                                             "heals": 0, "errors": 0})
        self.assertEqual(list(state["pools"]), ["forecast", "llm"])
        for name, pool in state["pools"].items():
            self.assertEqual(list(pool), ["preferred", "backends"])
            self.assertEqual(pool["preferred"], "A")
            for entry in pool["backends"]:
                self.assertEqual(list(entry), [
                    "name", "url", "probe", "healthy", "draining", "healing", "inflight",
                    "latency_ewma_ms", "score", "routed", "failures", "last_error", "probe_data"])
                self.assertEqual((entry["healthy"], entry["draining"], entry["healing"]), (True, False, False))
                self.assertEqual((entry["inflight"], entry["failures"], entry["last_error"]), (0, 0, None))
                self.assertIsInstance(entry["score"], float)
                self.assertIsInstance(entry["probe_data"], dict)
        forecast = state["pools"]["forecast"]["backends"][0]
        self.assertEqual((forecast["name"], forecast["url"], forecast["probe"], forecast["routed"]),
                         ("A", f.url, "forecast", 1))
        self.assertIsInstance(forecast["latency_ewma_ms"], float)
        for key in ("queued", "queued_units", "gpu_util_pct", "busy_other", "slowdown",
                    "mem_available_gb", "ms_per_unit"):
            self.assertIn(key, forecast["probe_data"])
        chat = state["pools"]["llm"]["backends"][0]
        self.assertEqual((chat["probe"], chat["routed"], chat["latency_ewma_ms"]), ("llamacpp", 0, None))
        self.assertEqual(chat["probe_data"], {"queued": 3, "running": 1})
        self.assertIsInstance(state["decisions"], list)
        last = state["decisions"][-1]
        self.assertEqual(list(last), ["ts", "pool", "backend", "policy", "kind", "reason", "retries"])
        self.assertLess(abs(last["ts"] - time.time()), 30.0)
        self.assertEqual(dict(last, ts=0), {"ts": 0, "pool": "forecast", "backend": "A", "policy": "adaptive",
                                            "kind": "route", "reason": "preferred", "retries": 0})

    def test_policy_drain_and_health(self):
        a, b = self.backend("A"), self.backend("B")
        a.answer_with("/v1/info")
        b.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", a), spec("B", b)], policy="round_robin")
        status, headers, data = call(app.port, "GET", "/router/health")
        self.assertEqual(status, 200)
        health = json.loads(data.decode("utf-8"))
        self.assertEqual((health["status"], health["policy"], health["usable"]),
                         ("ok", "round_robin", {"forecast": 2, "llm": 0}))
        # the policy
        status, headers, data = call(app.port, "POST", "/router/policy", {"policy": "static"})
        self.assertEqual((status, json.loads(data.decode("utf-8"))), (200, {"policy": "static"}))
        self.assertEqual(status_of(app)["policy"], "static")
        self.assertEqual(call(app.port, "GET", "/v1/info")[1]["x-router-policy"], "static")
        status, headers, data = call(app.port, "POST", "/router/policy", {"policy": "fastest"})
        self.assertEqual(status, 400)
        self.assertIn("unknown policy 'fastest'", json.loads(data.decode("utf-8"))["error"])
        self.assertEqual(call(app.port, "POST", "/router/policy", b"policy=static")[0], 400)
        self.assertEqual(status_of(app)["policy"], "static")
        # draining
        status, headers, data = call(app.port, "POST", "/router/drain", {"backend": "A", "drain": True})
        self.assertEqual((status, json.loads(data.decode("utf-8"))),
                         (200, {"backend": "A", "drain": True, "pools": ["forecast"]}))
        self.assertTrue(row(app, "forecast", "A")["draining"])
        self.assertEqual(set(call(app.port, "GET", "/v1/info")[1]["x-backend"] for _ in range(4)), {"B"})
        self.assertEqual(call(app.port, "POST", "/router/drain", {"backend": "B", "drain": True})[0], 200)
        status, headers, data = call(app.port, "GET", "/v1/info")
        self.assertEqual((status, json.loads(data.decode("utf-8"))), (503, {"error": "no healthy backend"}))
        self.assertEqual(json.loads(call(app.port, "GET", "/router/health")[2].decode("utf-8"))["status"], "degraded")
        self.assertEqual(call(app.port, "POST", "/router/drain", {"backend": "A", "drain": False})[0], 200)
        self.assertEqual(call(app.port, "GET", "/v1/info")[1]["x-backend"], "A")
        self.assertEqual(call(app.port, "POST", "/router/drain", {"backend": "Z", "drain": True})[0], 404)
        self.assertEqual(call(app.port, "POST", "/router/drain", {"backend": "B"})[0], 400)
        self.assertEqual(call(app.port, "POST", "/router/drain", {"backend": "B", "drain": "yes"})[0], 400)
        # wrong method, wrong path
        self.assertEqual(call(app.port, "GET", "/router/policy")[0], 405)
        self.assertEqual(call(app.port, "POST", "/router/status", {})[0], 405)
        self.assertEqual(call(app.port, "GET", "/router/nothing")[0], 404)
        seen = [d["kind"] for d in status_of(app)["decisions"]]
        self.assertEqual(seen.count("policy"), 1)
        self.assertEqual(seen.count("drain"), 3)

    def test_cors(self):
        a = self.backend("A")
        a.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", a)])
        answers = [call(app.port, "GET", "/router/status"), call(app.port, "GET", "/router/health"),
                   call(app.port, "POST", "/router/policy", {"policy": "static"}),
                   call(app.port, "POST", "/router/policy", {"policy": "nope"}),
                   call(app.port, "POST", "/router/drain", {"backend": "A", "drain": False}),
                   call(app.port, "GET", "/router/nothing")]
        self.assertEqual([a[0] for a in answers], [200, 200, 200, 400, 200, 404])
        for status, headers, data in answers:
            self.assertEqual(headers["access-control-allow-origin"], "*")
        # A browser's preflight is answered by the router itself, for any path.
        for path in ("/router/policy", "/router/drain", "/router/status", "/v1/forecast"):
            status, headers, data = call(app.port, "OPTIONS", path, headers={
                "Origin": "http://localhost:8000", "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"})
            self.assertEqual((status, data), (204, b""))
            self.assertNotIn("content-length", headers)
            self.assertEqual(headers["access-control-allow-origin"], "*")
            self.assertIn("POST", headers["access-control-allow-methods"])
            self.assertIn("Content-Type", headers["access-control-allow-headers"])
        self.assertEqual(a.seen, [])                       # no preflight reached a backend
        # A page may read the router's headers on a proxied answer.
        headers = call(app.port, "GET", "/v1/info")[1]
        self.assertEqual(headers["access-control-allow-origin"], "*")
        self.assertIn("X-Backend", headers["access-control-expose-headers"])

    def test_get_slash_serves_the_dashboard_or_one_line(self):
        self.assertEqual(R.REPO_ROOT, REPO)                # found from router.py's own place
        app = self.router()
        tmp = tempfile.mkdtemp(prefix="wx-router-test-")
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(setattr, R, "REPO_ROOT", R.REPO_ROOT)
        R.REPO_ROOT = tmp                                  # a repo with no dashboard yet
        status, headers, data = call(app.port, "GET", "/")
        self.assertEqual(status, 200)
        self.assertTrue(headers["content-type"].startswith("text/plain"))
        self.assertIn(b"/router/status", data)
        self.assertEqual(data.count(b"\n"), 1)             # one line
        os.mkdir(os.path.join(tmp, "dashboard"))
        page = b"<!doctype html><title>desk</title><p>hello</p>\n"
        with open(os.path.join(tmp, "dashboard", "index.html"), "wb") as f:
            f.write(page)
        status, headers, data = call(app.port, "GET", "/")
        self.assertEqual((status, data), (200, page))
        self.assertTrue(headers["content-type"].startswith("text/html"))
        self.assertEqual(headers["content-length"], str(len(page)))

    def test_decisions_go_to_the_log_file(self):
        tmp = tempfile.mkdtemp(prefix="wx-router-test-")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "decisions.jsonl")
        a = self.backend("A")
        a.answer_with("/v1/info")
        app = self.router(forecast=[spec("A", a)], decision_log=path)
        for _ in range(12):
            self.assertEqual(call(app.port, "GET", "/v1/info")[0], 200)
        self.assertEqual(call(app.port, "POST", "/router/policy", {"policy": "adaptive"})[0], 200)
        with open(path) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual([d["kind"] for d in lines], ["route"] * 12 + ["policy"])   # every one, unsampled
        for decision in lines:
            self.assertEqual(set(decision), DECISION_KEYS)
        self.assertEqual((lines[0]["pool"], lines[0]["backend"], lines[0]["policy"]), ("forecast", "A", "static"))


class LoadTests(RouterCase):

    def test_two_hundred_requests_in_flight(self):
        try:
            import resource
            R.raise_open_file_limit(4096)
            if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 1100:
                self.skipTest("this test needs about 1000 open sockets; the limit here is lower")
        except ImportError:
            pass
        a, b = self.backend("A"), self.backend("B")
        release = threading.Event()
        self.addCleanup(release.set)

        def hold(fake):
            def route(h, body):
                release.wait(30.0)
                reply(h, 200, {"node": fake.name, "n": json.loads(body.decode("utf-8"))["n"]})
            return route
        a.routes["/v1/forecast"] = hold(a)
        b.routes["/v1/forecast"] = hold(b)
        app = self.router(forecast=[spec("A", a), spec("B", b)], policy="round_robin",
                          first_byte_timeout_s=40.0, connect_timeout_s=5.0, fail_n=1000)
        results = [None] * 200

        def client(i):
            try:
                status, headers, data = call(app.port, "POST", "/v1/forecast", {"n": i}, timeout=40.0)
                results[i] = (status, headers.get("x-backend"), json.loads(data.decode("utf-8")))
            except Exception as exc:
                results[i] = exc
        threads = [threading.Thread(target=client, args=(i,)) for i in range(200)]
        for thread in threads:
            thread.daemon = True
            thread.start()

        def in_flight():
            return [r["inflight"] for r in status_of(app)["pools"]["forecast"]["backends"]]
        self.wait_for(lambda: sum(in_flight()) == 200, "200 requests in flight", timeout=30.0)
        self.assertEqual(in_flight(), [100, 100])          # round robin, not one too many anywhere
        release.set()
        for thread in threads:
            thread.join(30.0)
        for i, result in enumerate(results):
            self.assertIsInstance(result, tuple, "request %d: %r" % (i, result))
            status, name, answer = result
            self.assertEqual((status, answer["n"], answer["node"]), (200, i, name))   # no mix-ups
        self.assertEqual(sorted(r[1] for r in results), ["A"] * 100 + ["B"] * 100)
        state = self.wait_for(lambda: status_of(app) if sum(in_flight()) == 0 else None, "all to finish")
        rows = state["pools"]["forecast"]["backends"]
        self.assertEqual([(r["inflight"], r["routed"], r["healthy"]) for r in rows],
                         [(0, 100, True), (0, 100, True)])
        self.assertEqual(state["counters"], {"routed": 200, "failovers": 0, "switches": 0,
                                             "heals": 0, "errors": 0})


class CommandLineTests(unittest.TestCase):
    """The real program: first line on stdout, and a clean exit on a signal."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wx-router-test-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def start(self, *args):
        """Start router.py in the temp folder. Returns (process, first line of stdout)."""
        proc = subprocess.Popen([sys.executable, ROUTER_PY] + list(args), cwd=self.tmp,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def tidy():
            if proc.poll() is None:
                proc.kill()
            proc.wait(10.0)
            proc.stdout.close()
            proc.stderr.close()
        self.addCleanup(tidy)
        watchdog = threading.Timer(30.0, proc.kill)        # never leave it running
        watchdog.daemon = True
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        return proc, proc.stdout.readline().decode("utf-8").rstrip("\n")

    def test_first_line_and_clean_exit_on_sigterm_and_sigint(self):
        a = FakeBackend("A")
        self.addCleanup(a.stop)
        a.answer_with("/v1/info", ok=True)
        dead = "http://127.0.0.1:%d" % closed_port()
        for sig in (signal.SIGTERM, signal.SIGINT):
            proc, first = self.start("--port", "0", "--policy", "round_robin", "--forecast", "A=" + a.url,
                                     "--llm", "A=%s,B=%s" % (dead, dead), "--llm-probe", "vllm")
            self.assertRegex(first, r"^listening on http://127\.0\.0\.1:\d+$")
            port = int(first.rsplit(":", 1)[1])
            status, headers, data = call(port, "GET", "/v1/info")
            self.assertEqual((status, headers["x-backend"], headers["x-router-policy"]), (200, "A", "round_robin"))
            state = json.loads(call(port, "GET", "/router/status")[2].decode("utf-8"))
            self.assertEqual(state["policy"], "round_robin")
            self.assertEqual([b["probe"] for b in state["pools"]["forecast"]["backends"]], ["forecast"])
            self.assertEqual([(b["name"], b["probe"]) for b in state["pools"]["llm"]["backends"]],
                             [("A", "vllm"), ("B", "vllm")])
            proc.send_signal(sig)
            self.assertEqual(proc.wait(10.0), 0)
            self.assertNotIn("Traceback", proc.stderr.read().decode("utf-8"))
        # With no config file the decisions went to the default file, in the working folder.
        with open(os.path.join(self.tmp, "router-decisions.jsonl")) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual([d["kind"] for d in lines].count("route"), 2)
        self.assertEqual(set(lines[0]), DECISION_KEYS)

    def test_config_file_and_overrides(self):
        a = FakeBackend("A")
        self.addCleanup(a.stop)
        path = os.path.join(self.tmp, "config.json")
        with open(path, "w") as f:
            json.dump({"port": 8100, "policy": "static", "decision_log": "", "probe_s": 0.2,
                       "pools": {"forecast": {"backends": [{"name": "A", "url": a.url, "probe": "forecast"}]}}}, f)
        proc, first = self.start("--config", path, "--port", "0", "--policy", "least_loaded")
        self.assertRegex(first, r"^listening on http://127\.0\.0\.1:\d+$")
        port = int(first.rsplit(":", 1)[1])
        health = json.loads(call(port, "GET", "/router/health")[2].decode("utf-8"))
        self.assertEqual((health["status"], health["policy"]), ("ok", "least_loaded"))
        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(10.0), 0)
        self.assertEqual(os.listdir(self.tmp), ["config.json"])       # "" means no log file

    def test_bad_start_says_why(self):
        for args, words in ((["--config", os.path.join(self.tmp, "missing.json")], "cannot start"),
                            ([], "no backends"),
                            (["--forecast", "http://127.0.0.1:8101"], "expected NAME=URL"),
                            (["--forecast", "A=127.0.0.1:8101"], "http://host:port")):
            proc, first = self.start(*args)
            self.assertEqual(first, "")                    # nothing on stdout
            self.assertEqual(proc.wait(10.0), 2)
            self.assertIn(words, proc.stderr.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
