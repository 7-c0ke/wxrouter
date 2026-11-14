"""Tests for the Control Room's live Events stream (ui/events.py) and its endpoints in ui/server.py:
GET /api/events, GET /api/telemetry, POST /api/incident/diagnose, POST /api/incident/remediate.

    python -m pytest tests/test_events.py -q

Telemetry samples are in the WhatIf recorder's own format (whatif/telemetry.py); tests/whatif_fakes.py makes
them. The Fuzzball events are real ones, captured from the team's real-stack run on Nov 14
(tests/fixtures_fuzzball_events.json). No stack and no cluster are needed.
"""
import asyncio
import copy
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
for name in ("server", "agent", "events"):
    sys.modules.pop(name, None)
import events as E  # noqa: E402
from whatif_fakes import make_samples  # noqa: E402

FIXTURE = json.loads((ROOT / "tests" / "fixtures_fuzzball_events.json").read_text())["events"]


def texts(hub, src=None):
    return [e["text"] for e in hub.events if src is None or e["src"] == src]


def base_sample(t=100.0):
    return copy.deepcopy(make_samples("box-b:slow", t0=t + 1000, n_pre=1, n_post=0)[0]) | {"t": t}


# -------------------------------------------------------------------- telemetry changes
def test_steady_samples_make_no_events():
    hub = E.EventHub()
    seen = E.TelemetryEvents(hub)
    for s in make_samples("box-b:slow", t0=1000.0, n_pre=10, n_post=0):
        seen.observe(s)
    assert list(hub.events) == []


def test_box_down_and_back_up_from_router_and_health():
    hub, s = E.EventHub(), base_sample()
    seen = E.TelemetryEvents(hub)
    seen.observe(s)
    down = copy.deepcopy(s) | {"t": 101.0}
    down["router"]["backends"]["box-b"]["healthy"] = False
    down["boxes"]["box-b"]["health"]["code"] = 0
    seen.observe(down)
    up = copy.deepcopy(s) | {"t": 102.0}
    seen.observe(up)
    t = texts(hub)
    assert "router marked box-b down: no traffic to it" in t
    assert "box-b /health no answer" in t
    assert "router marked box-b healthy: it gets traffic again" in t
    assert "box-b /health answers again (HTTP 200)" in t
    assert all(e["box"] == "box-b" for e in hub.events) and [e["level"] for e in hub.events] == ["bad", "bad", "up", "up"]


def test_router_lost_and_back_failovers_errors_probe_jobs():
    hub, s = E.EventHub(), base_sample()
    seen = E.TelemetryEvents(hub)
    s["jobs"] = {"box-a": {"servers": {"forecast": "running"}, "jobs": {"finetune": "idle"}}}
    seen.observe(s)
    n = copy.deepcopy(s) | {"t": 101.0}
    n["router"]["failovers"] = 3
    n["traffic"]["err_pct"] = 12.0
    n["probe"] = {"box-a": {"t": 101.0, "code": 0, "compute_ms": None, "total_ms": 3000.0, "timeout": True},
                  "box-b": {"t": 99.0, "pending_ms": 1500.0}}
    n["jobs"] = {"box-a": {"servers": {"forecast": "backoff"}, "jobs": {"finetune": "running"}}}
    seen.observe(n)
    gone = copy.deepcopy(n) | {"t": 102.0, "router": None}
    gone["probe"] = {"box-b": {"t": 99.0, "pending_ms": 2500.0}}   # still stalled: not reported twice
    seen.observe(gone)
    back = copy.deepcopy(s) | {"t": 103.0}
    back["router"]["failovers"] = 3
    seen.observe(back)
    t = texts(hub)
    assert "3 failovers: requests retried on the other box" in t
    assert any(x.startswith("errors 12% of requests") for x in t)
    assert "box-a probe forecast timed out" in t
    assert sum("box-b probe forecast stalled" in x for x in t) == 1
    assert "box-a probe forecast answers again" not in t, "box-a has had no probe news since its timeout"
    assert "box-a server forecast: running -> backoff" in t and "box-a job finetune: idle -> running" in t
    assert "router not answering" in t and "router answering again" in t
    assert "errors back under 1% of requests" in t


# -------------------------------------------------------------------- Fuzzball events
def test_fuzzball_real_events_folded_and_deduplicated():
    hub = E.EventHub()
    fb = E.FuzzballEvents(hub, "cd1cd258", "/bin/true")
    n = fb.ingest(FIXTURE)
    assert 0 < n < len(FIXTURE), "the 12 'reservation held' events of box-a must fold"
    assert texts(hub).count("Holding resources for scheduling") == 1, "12 reservation-held events, one line"
    assert fb.ingest(FIXTURE) == 0, "the same events again add nothing"
    wf = [e for e in hub.events if e["stage_kind"] == "workflow"]
    assert wf and wf[0]["stage"] == "workflow" and wf[0]["level"] == "up"
    assert all(isinstance(e["ts"], float) for e in hub.events)
    assert E.fuzzball_level("STAGE_CANCELED") == "bad" and E.fuzzball_level("CONTAINER_EXITED") == "end"


def _cli(tmp_path, body):
    p = tmp_path / "fuzzball"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_fuzzball_poll_ok_then_unreachable_once_then_back(tmp_path):
    hub = E.EventHub()
    data = tmp_path / "events.json"
    data.write_text(json.dumps({"events": FIXTURE[:3]}))
    fb = E.FuzzballEvents(hub, "cd1cd258", _cli(tmp_path, f'if [ -f "{tmp_path}/down" ]; then echo \'Error: rpc error: code = Unavailable desc = connection refused\' >&2; exit 1; fi\ncat "{data}"\n'))
    asyncio.run(fb.poll_once())
    assert fb.status == "ok" and len(hub.events) >= 1
    (tmp_path / "down").write_text("")
    asyncio.run(fb.poll_once())
    asyncio.run(fb.poll_once())
    assert fb.status == "unreachable" and fb.reason == "Fuzzball API unreachable from this machine"
    assert texts(hub).count("Fuzzball API unreachable from this machine") == 1
    (tmp_path / "down").unlink()
    asyncio.run(fb.poll_once())
    assert fb.status == "ok" and "Fuzzball API reachable again" in texts(hub)


def test_fuzzball_off_without_cli_or_workflow(tmp_path):
    assert E.FuzzballEvents(E.EventHub(), None, "/bin/true").status == "off"
    assert "no fuzzball CLI" in E.FuzzballEvents(E.EventHub(), "x", None).reason
    st = tmp_path / "state.json"
    st.write_text(json.dumps({"real": {"id": "abc", "name": "real-stack"}}))
    assert E.workflow_from_state(st) == "abc"
    assert E.workflow_from_state(tmp_path / "missing.json") is None


# -------------------------------------------------------------------- endpoints
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import server as S
    ops = S.Ops("http://127.0.0.1:9", {"box-a": "http://127.0.0.1:9", "box-b": "http://127.0.0.1:9"})
    app = S.create_app(ops, S.Agent(ops, None, "x", None), telemetry=False)
    with TestClient(app) as c:
        c.app_ = app
        yield c


def test_events_and_telemetry_endpoints(client):
    hub, rec = client.app_.state.hub, client.app_.state.recorder
    hub.add("telemetry", "bad", "router marked box-b down: no traffic to it", box="box-b")
    d = client.get("/api/events").json()
    assert d["last_seq"] == 1 and d["events"][0]["box"] == "box-b"
    assert d["fuzzball"]["status"] == "off" and d["telemetry"]["on"] is False
    assert client.get("/api/events?after=1").json()["events"] == []
    now = time.time()
    rec.buf.extend(make_samples("box-a:down", t0=now - 30, n_pre=35, n_post=30))
    assert 25 <= len(client.get("/api/telemetry?seconds=30").json()["samples"]) <= 31


def test_diagnose_rule_on_recorded_telemetry_and_remedy(client):
    rec, hub = client.app_.state.recorder, client.app_.state.hub
    assert client.post("/api/incident/diagnose", json={"policy": "rule"}).status_code == 409, "no telemetry yet"
    assert client.post("/api/incident/diagnose", json={"policy": "nope"}).status_code == 422
    rec.buf.extend(make_samples("box-b:slow", t0=time.time() - 30, n_pre=35, n_post=30))
    r = client.post("/api/incident/diagnose", json={"policy": "rule"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["diagnosis"]["ranking"][0] == "box-b:slow"
    assert d["can_apply"] and "/router/config" in d["remedy"] and "box-a" in d["remedy"]
    ev = [e for e in hub.events if e["src"] == "incident"][-1]
    assert "most likely box-b:slow" in ev["text"] and ev["trail"] and ev["incident"] == d["incident"]
    # the router at 127.0.0.1:9 does not answer: the remedy is tried and reported as failed, not hidden
    rr = client.post("/api/incident/remediate", json={"incident": d["incident"]}).json()
    assert rr["applied"] is False and rr["result"]["status"] == 0
    assert "FAILED" in [e for e in hub.events if e["src"] == "incident"][-1]["text"]
    assert client.post("/api/incident/remediate", json={"incident": 99}).status_code == 404


def test_down_box_without_supervisor_gets_no_automatic_start(client):
    client.app_.state.recorder.buf.extend(make_samples("box-a:down", t0=time.time() - 30, n_pre=35, n_post=30))
    d = client.post("/api/incident/diagnose", json={"policy": "rule"}).json()
    assert d["diagnosis"]["ranking"][0] == "box-a:down"
    assert d["can_apply"] is False and "no supervisor" in d["remedy"]


def test_a_failing_probe_is_one_line_until_it_answers_again():
    hub, s = E.EventHub(), base_sample()
    seen = E.TelemetryEvents(hub)
    seen.observe(s)
    for i in range(1, 8):    # 503 on every probe for 7 s
        n = copy.deepcopy(s) | {"t": 100.0 + i}
        n["probe"] = {"box-a": {"t": 100.0 + i, "code": 503, "compute_ms": None, "total_ms": 5.0}}
        seen.observe(n)
    ok = copy.deepcopy(s) | {"t": 108.0}
    ok["probe"] = {"box-a": {"t": 108.0, "code": 200, "compute_ms": 3.0, "total_ms": 5.0}}
    seen.observe(ok)
    assert texts(hub) == ["box-a probe forecast failed: HTTP 503", "box-a probe forecast answers again"]


def test_fuzzball_asks_for_the_newest_events_not_the_first_100(tmp_path):
    """The CLI without -n returns the OLDEST 100 events (seen on the cluster: 459 events, only the first 100
    came back), so a busy workflow would freeze. The poller must ask for the newest ones."""
    args = tmp_path / "args.txt"
    data = tmp_path / "events.json"
    data.write_text(json.dumps({"events": FIXTURE[:2]}))
    fb = E.FuzzballEvents(E.EventHub(), "cd1cd258", _cli(tmp_path, f'echo "$@" > "{args}"\ncat "{data}"\n'))
    asyncio.run(fb.poll_once())
    assert args.read_text().split() == ["workflow", "events", "cd1cd258", "-n", "1000", "-o", "json"]
