"""The read-only Control Room: the page and forecasts work; nothing that changes the cluster does."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def app_and_ops():
    for name in ("server", "agent", "router_switching.serve_readonly"):
        sys.modules.pop(name, None)
    from router_switching import serve_readonly as ro
    ops = ro.ui.Ops("http://127.0.0.1:9", {"box-a": "http://127.0.0.1:9", "box-b": "http://127.0.0.1:9"},
                    {"box-a": "http://127.0.0.1:9"})
    agent = ro.ui.Agent(ops, None, "x", None)
    return ro, ro.build(ops, agent), ops


def test_page_loads_and_says_it_is_read_only():
    ro, app, ops = app_and_ops()
    with TestClient(app) as c:
        assert c.get("/").status_code == 200 and "wxrouter Control Room" in c.get("/").text
        assert c.get("/api/readonly").json()["read_only"] is True
        assert c.get("/api/v1").status_code == 200
    assert ops.stacks == {}, "no supervisor: no Fine-tune or Kill from this page"


def test_every_change_is_refused_from_the_page():
    ro, app, ops = app_and_ops()
    with TestClient(app) as c:
        for path, body in (("/api/router/config", {"policy": "static"}), ("/api/router/config", {"drain": {"box-a": True}}),
                           ("/api/box/box-a/mode", {"mode": "error"}), ("/api/box/box-a/train", {"action": "start"}),
                           ("/api/box/box-a/kill", None), ("/api/switching/burst", None)):
            r = c.post(path, json=body)
            assert r.status_code == 403 and "read-only" in r.json()["detail"], path


def test_the_chat_cannot_change_the_cluster_either():
    import asyncio
    ro, app, ops = app_and_ops()
    for call in (ops.set_config({"policy": "static"}), ops.set_box_mode("box-a", "error"), ops.finetune("box-a", "start"),
                 ops.kill_forecast("box-a"), ops.set_policy("static")):
        assert "read-only" in asyncio.run(call)["error"]


def test_load_test_is_capped():
    ro, app, ops = app_and_ops()
    with TestClient(app) as c:
        assert c.post("/api/load", json={"rps": 150}).json()["rps"] == ro.MAX_LOAD_RPS
        assert c.post("/api/load", json={"rps": 0}).json()["rps"] == 0
