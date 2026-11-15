"""Tests for the team's GPU view from Fuzzball (ui/fuzzball_gpu.py) and GET /api/cluster/gpus.

    python -m pytest tests/test_fuzzball_gpu.py -q

The JSON is real: `fuzzball node list / node show / workflow list -o json`, captured on the event cluster
on Nov 14 2026 while both team14 GPUs (CIQ-CHI-SPARK27 and -28) were held by a workflow of the team's shared group
(tests/fixtures_fuzzball_nodes.json). The CLI is a small script that answers with that JSON.
"""
import asyncio
import copy
import json
import stat
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "ui", ROOT, ROOT / "tests"):
    sys.path.insert(0, str(p))
for name in ("server", "agent", "events", "fuzzball_gpu"):
    sys.modules.pop(name, None)
import events as E  # noqa: E402
import fuzzball_gpu as G  # noqa: E402
from whatif_fakes import make_samples  # noqa: E402

FIX = json.loads((ROOT / "tests" / "fixtures_fuzzball_nodes.json").read_text())
SHOWS = FIX["node_show"]
S27 = next(s for s in SHOWS.values() if s["hostName"] == "CIQ-CHI-SPARK27")
OTHER_WF = S27["runningJobs"][0]["workflowId"]


def texts(hub):
    return [e["text"] for e in hub.events]


def test_node_view_reads_the_gpu_and_who_holds_it():
    v = G.node_view(S27, ours={})
    assert v["host"] == "CIQ-CHI-SPARK27" and v["health"] == "healthy" and v["status"] == "ready" and v["score"] == 99
    assert v["gpu"]["model"] == "NVIDIA GB10" and v["gpu"]["arch"] == "Blackwell"
    assert v["gpu"]["uuids"] == ["GPU-ef16568f-cc74-340a-f52a-46831263cfd4"] and v["gpu"]["allocated"] is True
    assert v["gpu"]["total"] == 1 and v["gpu"]["free"] == 0
    assert len(v["held_by"]) == 1, "the job without a GPU in the same workflow is merged, not a second holder"
    h = v["held_by"][0]
    assert (h["workflow"], h["ours"], h["gpus"], h["jobs"]) == (OTHER_WF, False, 1, 2)
    assert G.holder_text(h) == f"workflow {OTHER_WF[:8]} (not in the CLI's group)"
    assert G.node_view(S27, ours={OTHER_WF: "real-stack-team14-5"})["held_by"][0]["ours"] is True


def test_changes_between_polls_become_events():
    hub = E.EventHub()
    g = G.FuzzballGPUs(hub, "/bin/true", "team14", {"box-a": "CIQ-CHI-SPARK27"})
    v = G.node_view(S27, {})
    g.update([v])
    first = texts(hub)
    assert first == [f"CIQ-CHI-SPARK27 (box-a): NVIDIA GB10 busy, held by workflow {OTHER_WF[:8]} (not in the CLI's group); healthy, score 99"]
    assert hub.events[-1]["box"] == "box-a" and hub.events[-1]["src"] == "cluster"
    g.update([copy.deepcopy(v)])
    assert len(hub.events) == 1, "nothing changed: no line"
    free = copy.deepcopy(v)
    free["held_by"], free["gpu"]["allocated"], free["gpu"]["free"] = [], False, 1
    g.update([free])
    ours = copy.deepcopy(free)
    ours["held_by"] = [{"workflow": "abc12345-0000", "name": "real-stack-team14-5", "ours": True, "gpus": 1,
                        "cores": 12, "memory_gb": 64, "jobs": 1}]
    ours["gpu"]["allocated"], ours["health"], ours["score"] = True, "degraded", 90
    g.update([ours])
    t = texts(hub)[1:]
    assert f"CIQ-CHI-SPARK27 (box-a): GPU released by workflow {OTHER_WF[:8]} (not in the CLI's group)" in t
    assert "CIQ-CHI-SPARK27 (box-a): GPU free" in t
    assert "CIQ-CHI-SPARK27 (box-a): GPU taken by workflow real-stack-team14-5 (abc12345)" in t
    assert "CIQ-CHI-SPARK27 (box-a): health healthy -> degraded" in t
    assert "CIQ-CHI-SPARK27 (box-a): reliability score 99 -> 90" in t
    levels = {e["text"].split(": ", 1)[1]: e["level"] for e in list(hub.events)[1:]}
    assert levels["health healthy -> degraded"] == "bad" and levels["GPU free"] == "up"


def _fake_cli(tmp_path, down=False):
    (tmp_path / "list.json").write_text(json.dumps(FIX["node_list"]))
    (tmp_path / "wf.json").write_text(json.dumps(FIX["workflow_list"]))
    for nid, show in SHOWS.items():
        (tmp_path / (nid.replace("/", "_") + ".json")).write_text(json.dumps(show))
    (tmp_path / "ctx.json").write_text(json.dumps({"groupName": "team14@fb-hack.local"}))
    script = f'''#!/bin/sh
D="{tmp_path}"
if [ -f "$D/down" ]; then echo 'Error: rpc error: code = Unavailable desc = connection refused' >&2; exit 1; fi
echo "$@" >> "$D/calls.txt"
case "$1 $2" in
  "context show") cat "$D/ctx.json" ;;
  "node list") cat "$D/list.json" ;;
  "workflow list") cat "$D/wf.json" ;;
  "node show") cat "$D/$(echo "$3" | tr / _).json" ;;
esac
'''
    p = tmp_path / "fuzzball"
    p.write_text(script)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_poll_reads_only_the_team_nodes_and_names_holders(tmp_path):
    hub = E.EventHub()
    g = G.FuzzballGPUs(hub, _fake_cli(tmp_path))
    asyncio.run(g.poll_once())
    assert g.status == "ok" and g.group == "team14"
    assert sorted(g.nodes) == ["CIQ-CHI-SPARK27", "CIQ-CHI-SPARK28"], "the node of another team is not read"
    shows = [c for c in (tmp_path / "calls.txt").read_text().splitlines() if c.startswith("node show")]
    assert len(shows) == 2 and all(c.endswith("-o json") for c in shows)
    info = g.info()
    assert all(n["gpu"]["allocated"] and not n["held_by"][0]["ours"] for n in info["nodes"])
    assert len(hub.events) == 2 and all("not in the CLI's group" in t for t in texts(hub))


def test_unreachable_is_one_line_then_back(tmp_path):
    hub = E.EventHub()
    g = G.FuzzballGPUs(hub, _fake_cli(tmp_path), "team14")
    asyncio.run(g.poll_once())
    (tmp_path / "down").write_text("")
    asyncio.run(g.poll_once())
    asyncio.run(g.poll_once())
    assert g.status == "unreachable" and texts(hub).count("GPU view: Fuzzball API unreachable from this machine") == 1
    (tmp_path / "down").unlink()
    asyncio.run(g.poll_once())
    assert g.status == "ok" and "Fuzzball node API reachable again" in texts(hub)


def test_off_without_cli():
    g = G.FuzzballGPUs(E.EventHub(), None)
    assert g.info()["status"] == "off" and "no fuzzball CLI" in g.info()["reason"]
    asyncio.run(g.run())   # returns at once


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import server as S
    ops = S.Ops("http://127.0.0.1:9", {"box-a": "http://127.0.0.1:9", "box-b": "http://127.0.0.1:9"})
    app = S.create_app(ops, S.Agent(ops, None, "x", None), telemetry=False)
    with TestClient(app) as c:
        c.app_ = app
        yield c


def test_endpoint_and_diagnosis_evidence(client):
    assert client.get("/api/cluster/gpus").json()["status"] == "off"
    assert client.get("/api/events").json()["cluster"]["status"] == "off"
    gpus = client.app_.state.gpus
    gpus.box_of = {"CIQ-CHI-SPARK28": "box-b"}
    s28 = next(s for s in SHOWS.values() if s["hostName"] == "CIQ-CHI-SPARK28")
    gpus.update([G.node_view(s28, {})])
    got = client.get("/api/cluster/gpus").json()["nodes"]
    assert got[0]["box"] == "box-b" and got[0]["gpu"]["allocated"] is True
    client.app_.state.recorder.buf.extend(make_samples("box-b:slow", t0=time.time() - 30, n_pre=35, n_post=30))
    d = client.post("/api/incident/diagnose", json={"policy": "rule"}).json()
    assert d["diagnosis"]["ranking"][0] == "box-b:slow"
    assert d["diagnosis"]["trail"][0].startswith("[fuzzball] box-b's node CIQ-CHI-SPARK28: GPU held by workflow")
    assert "not in the CLI's group" in d["diagnosis"]["trail"][0]
