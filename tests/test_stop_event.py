"""A fine-tune that was stopped on request is "stopped" in the telemetry and in the Events card, not "failed".

    python -m pytest tests/test_stop_event.py -v

The supervisor calls every run that did not exit with 0 "failed"; weather/finetune.py leaves with 3 after a stop.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ui"))
import events as E                      # noqa: E402
from whatif import telemetry as T       # noqa: E402


def sup(state, code=None, server="running"):
    return {"name": "box-b", "servers": {"forecast": {"state": server, "last_exit": code}},
            "jobs": {"finetune": {"state": state, "exit": code, "last_lines": []}}}


def test_strip_supervisor_names_a_stop():
    for code, want in [(3, "stopped"), (-15, "stopped"), (-9, "stopped"), (1, "failed"), (2, "failed"),
                       (None, "failed"), (True, "failed"), (0, "failed"), ("3", "failed")]:
        assert T.strip_supervisor(sup("failed", code))["jobs"]["finetune"] == want, code
    for state in ("idle", "running", "done"):
        assert T.strip_supervisor(sup(state, 3))["jobs"]["finetune"] == state
    # a SERVER keeps the supervisor's own word, whatever its exit code
    assert T.strip_supervisor({"servers": {"forecast": {"state": "failed", "exit": 3}}, "jobs": {}})["servers"] == {"forecast": "failed"}
    assert T.strip_supervisor({"servers": {"forecast": None}, "jobs": {"finetune": None}}) == {"servers": {"forecast": None}, "jobs": {"finetune": None}}
    assert T.strip_supervisor(None) is None and T.strip_supervisor({}) == {"servers": {}, "jobs": {}}


def lines(*statuses):
    hub = E.EventHub()
    seen = E.TelemetryEvents(hub)
    for i, s in enumerate(statuses):
        seen.observe({"t": 100.0 + i, "router": {"backends": {}}, "jobs": {"box-b": T.strip_supervisor(s)}})
    return [(e["level"], e["text"]) for e in hub.events]


def test_events_card_says_stopped_for_a_stop_and_failed_for_a_failure():
    assert lines(sup("running"), sup("failed", 3)) == [("end", "box-b job finetune: running -> stopped")]
    assert lines(sup("running"), sup("failed", -15)) == [("end", "box-b job finetune: running -> stopped")]
    assert lines(sup("running"), sup("failed", 1)) == [("bad", "box-b job finetune: running -> failed")]
    assert lines(sup("running"), sup("done", 0)) == [("end", "box-b job finetune: running -> done")]
    assert lines(sup("failed", 3), sup("running")) == [("up", "box-b job finetune: stopped -> running")]
    assert lines(sup("idle"), sup("idle", server="stopped")) == [("bad", "box-b server forecast: running -> stopped")]
