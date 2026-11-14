"""The existing router's switching rules, checked one decision at a time with a fake clock.

Nothing here adds a routing algorithm. These tests pin down what router.py already does:
hysteresis (margin), dwell time (window_s), cooldown, failure detection, recovery and
return to the home box.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import router as rt


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(rt.time, "monotonic", c)
    return c


def make(clock, policy="adaptive", **adaptive):
    cfg = rt.AdaptiveConfig(**{"margin": 0.25, "window_s": 1.0, "cooldown_s": 10.0, **adaptive})
    r = rt.Router([rt.Backend("box-a", "http://a"), rt.Backend("box-b", "http://b")], policy, cfg,
                  timeout_s=5, poll_s=0.25, probe_s=0.5, down_after=2, up_after=2)
    r.last_switch = clock.t - 100  # the start-up cooldown is over
    return r


def speeds(r, a, b):
    r.by_name["box-a"].probe_ms, r.by_name["box-b"].probe_ms = a, b


def tick(r, clock, seconds, step=0.25):
    for _ in range(int(seconds / step)):
        clock.t += step
        r.decide_adaptive()


def test_starts_on_the_first_box(clock):
    assert make(clock).active.name == "box-a"


def test_hysteresis_a_small_advantage_never_switches(clock):
    r = make(clock)
    speeds(r, 100, 80)  # B is 20% faster; the margin is 25%
    tick(r, clock, 30)
    assert r.active.name == "box-a" and r.switches == []


def test_dwell_time_the_advantage_must_last_the_whole_window(clock):
    r = make(clock)
    speeds(r, 100, 50)
    tick(r, clock, 0.75)  # clearly better, but not yet for window_s
    assert r.active.name == "box-a"
    speeds(r, 100, 95)  # the advantage goes away: the timer resets
    tick(r, clock, 0.5)
    speeds(r, 100, 50)
    tick(r, clock, 0.75)
    assert r.active.name == "box-a"
    tick(r, clock, 0.5)
    assert r.active.name == "box-b" and len(r.switches) == 1
    assert "box-b 50ms vs box-a 100ms" in r.switches[0]["reason"]


def test_cooldown_blocks_a_second_switch(clock):
    r = make(clock)
    speeds(r, 100, 50)
    tick(r, clock, 2)
    assert r.active.name == "box-b"
    speeds(r, 20, 100)  # now A is far better
    tick(r, clock, 8)
    assert r.active.name == "box-b", "switched back inside the 10 s cooldown"
    tick(r, clock, 3)
    assert r.active.name == "box-a" and len(r.switches) == 2


def test_no_oscillation_when_the_boxes_trade_places_quickly(clock):
    r = make(clock)
    for i in range(40):  # the faster box flips every 0.5 s for 20 s
        speeds(r, *((100, 50) if i % 2 == 0 else (50, 100)))
        tick(r, clock, 0.5)
    assert len(r.switches) == 0


def test_failure_is_detected_after_down_after_checks_and_switches_at_once(clock):
    r = make(clock)
    speeds(r, 50, 50)
    a = r.by_name["box-a"]
    r.mark(a, ok=False)
    assert a.healthy, "one failed check must not take a box out"
    r.mark(a, ok=False)
    assert not a.healthy
    clock.t += 0.01
    r.decide_adaptive()  # no window, no cooldown for a dead box
    assert r.active.name == "box-b" and r.switches[-1]["reason"] == "active box down"


def test_connection_refused_takes_the_box_out_immediately(clock):
    r = make(clock)
    r.mark(r.by_name["box-a"], ok=False, hard=True)
    assert not r.by_name["box-a"].healthy
    assert next(b.name for b in r.order()) == "box-b"


def test_recovery_needs_up_after_good_checks(clock):
    r = make(clock)
    a = r.by_name["box-a"]
    r.mark(a, ok=False, hard=True)
    r.mark(a, ok=True)
    assert not a.healthy
    r.mark(a, ok=True)
    assert a.healthy


def test_traffic_returns_to_the_home_box_once_it_is_about_as_good(clock):
    r = make(clock, preferred="box-a", return_margin=0.20)
    speeds(r, 50, 50)
    a = r.by_name["box-a"]
    r.mark(a, ok=False, hard=True)
    clock.t += 0.1
    r.decide_adaptive()
    assert r.active.name == "box-b"
    r.mark(a, ok=True)
    r.mark(a, ok=True)
    a.probe_ms = 70  # home is back but 40% slower than B: stay away
    tick(r, clock, 15)
    assert r.active.name == "box-b"
    a.probe_ms = 58  # within the 20% return margin
    tick(r, clock, 2)
    assert r.active.name == "box-a" and "back to preferred box-a" in r.switches[-1]["reason"]
    assert len(r.switches) == 2


def test_without_a_home_box_traffic_stays_where_it_is(clock):
    r = make(clock)
    speeds(r, 50, 50)
    a = r.by_name["box-a"]
    r.mark(a, ok=False, hard=True)
    clock.t += 0.1
    r.decide_adaptive()
    r.mark(a, ok=True)
    r.mark(a, ok=True)
    a.probe_ms = 50
    tick(r, clock, 30)
    assert r.active.name == "box-b" and len(r.switches) == 1


def test_a_stalled_probe_counts_as_a_slow_box(clock):
    r = make(clock)
    speeds(r, 50, 50)
    r.by_name["box-a"].probe_started = clock.t  # a probe went out and never came back
    tick(r, clock, 3)
    assert r.active.name == "box-b"


@pytest.mark.parametrize("policy,expected", [("static", ["box-a", "box-b"]), ("adaptive", ["box-a", "box-b"])])
def test_order_puts_the_chosen_box_first_and_keeps_a_fallback(clock, policy, expected):
    assert [b.name for b in make(clock, policy).order()] == expected


def test_round_robin_alternates_and_least_loaded_follows_load(clock):
    r = make(clock, "round_robin")
    firsts = [r.order()[0].name for _ in range(4)]
    assert sorted(set(firsts)) == ["box-a", "box-b"] and firsts[0] != firsts[1]
    r = make(clock, "least_loaded")
    r.by_name["box-a"].in_flight = 5
    assert r.order()[0].name == "box-b"
