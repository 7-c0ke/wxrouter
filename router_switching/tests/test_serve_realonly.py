"""The real-only Control Room: a mock or CPU box makes every live endpoint refuse; real GPU boxes pass."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_api_v1 import AFNO, MOCK, ROUTER, TINY, Client  # noqa: E402  answers the real servers gave


def app_for(hosts):
    for name in ("server", "agent", "router_switching.serve_realonly"):
        sys.modules.pop(name, None)
    from router_switching import serve_realonly as ro
    ops = ro.ui.Ops("http://r", {"box-a": "http://a", "box-b": "http://b"})
    app = ro.build(ops, ro.ui.Agent(ops, None, "x", None))
    c = TestClient(app)
    c.__enter__()
    ops.client = Client(hosts)
    return c


def test_real_gpu_boxes_pass():
    c = app_for({"http://r": ROUTER, "http://a": AFNO, "http://b": AFNO})
    assert c.get("/api/realonly").json() == {"real_only": True, "refused": {}, "detail": None}
    assert c.get("/api/v1/gpus").status_code == 200 and c.get("/").status_code == 200


def test_a_mock_box_is_refused_everywhere_including_the_chat():
    c = app_for({"http://r": ROUTER, "http://a": MOCK, "http://b": AFNO})
    for path in ("/api/overview", "/api/v1/status", "/api/switching/live", "/api/model"):
        r = c.get(path)
        assert r.status_code == 503 and "mock server" in r.json()["detail"], path
    assert c.post("/api/chat", json={"messages": [{"role": "user", "content": "weather?"}]}).status_code == 503
    assert c.post("/api/forecast", json={"lat": 1, "lon": 1}).status_code == 503
    assert c.get("/").status_code == 200, "the page itself loads and can say why it shows nothing"


def test_a_cpu_box_is_refused():
    cpu = {**TINY, "/v1/metrics": {**TINY["/v1/metrics"], "device": {"type": "cpu", "name": "cpu"}}}
    c = app_for({"http://r": ROUTER, "http://a": AFNO, "http://b": cpu})
    assert c.get("/api/overview").status_code == 503 and "CPU" in c.get("/api/realonly").json()["detail"]


def test_an_offline_box_is_not_refused():
    c = app_for({"http://r": ROUTER, "http://a": AFNO})
    assert c.get("/api/realonly").json()["refused"] == {}
