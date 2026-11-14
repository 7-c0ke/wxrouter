"""Pure-function checks for the two-box benchmark driver."""
import loadgen
from inference import cluster_bench as cb


def row(t, box, ms=10.0, status=200):
    return {"t": t, "latency_ms": ms, "status": status, "served_by": box, "id": "x", "kind": "short"}


def test_timeline_counts_per_second():
    rows = [row(0.1, "box-a"), row(0.9, "box-b", 30.0), row(1.2, "box-b"), row(1.5, "none", status=0)]
    tl = cb.timeline(rows, 2.0)
    assert tl[0] == {"t": 0, "requests": 2, "errors": 0, "served_by": {"box-a": 1, "box-b": 1},
                     "p50_ms": 20.0, "p95_ms": 29.0}
    assert tl[1]["errors"] == 1 and tl[1]["served_by"] == {"box-b": 1}


def test_fault_report_shares_and_recovery():
    rows = ([row(t + 0.5, "box-a") for t in range(4)]          # before: all A
            + [row(t + 0.5, "box-b") for t in range(4, 14)]        # slowed, then stopped: all B
            + [row(14.5, "box-b"), row(15.5, "box-a"), row(16.5, "box-a")])
    events = loadgen.parse_events("4:slow:box-a,8:normal:box-a,10:kill:box-a,13:restart:box-a")
    rep = cb.fault_report(cb.timeline(rows, 17), events)
    assert rep["slow"]["share_before_pct"] == 100.0 and rep["slow"]["share_during_pct"] == 0.0
    assert rep["stop"]["errors_while_down"] == 0
    assert rep["stop"]["first_served_again_s"] == 15 and rep["stop"]["seconds_to_serve_again"] == 2


def test_distinct_nodes_needs_proof():
    def place(a, b):
        return {"box-a": {"node": a}, "box-b": {"node": b}}

    assert cb.distinct_nodes(place({"gpu_uuid": "GPU-1"}, {"gpu_uuid": "GPU-2"})) is True
    assert cb.distinct_nodes(place({"gpu_uuid": "GPU-1"}, {"gpu_uuid": "GPU-1"})) is False
    assert cb.distinct_nodes(place({"gpu_uuid": None, "addresses": []}, {"gpu_uuid": "GPU-2"})) is None
    assert cb.distinct_nodes(place({"gpu_uuid": None, "addresses": ["10.0.0.1"]},
                                   {"gpu_uuid": None, "addresses": ["10.0.0.2"]})) is True


def test_case_switches_keeps_only_this_case_and_times_it():
    status = {"policy": "adaptive", "adaptive": {"switches": [
        {"t": 5, "from": "box-a", "to": "box-b", "reason": "old"},
        {"t": 90, "from": "box-a", "to": "box-b", "reason": "slow"}]}}
    samples = [{"t": 0.0, "router": {"active": "box-a"}}, {"t": 0.5, "router": {"active": "box-a"}},
               {"t": 1.0, "router": {"active": "box-b"}}]
    assert cb.case_switches(status, 1, samples) == [
        {"t": 90, "from": "box-a", "to": "box-b", "reason": "slow", "t_case": 1.0}]
    assert cb.case_switches({**status, "policy": "round_robin"}, 0, samples) == []
