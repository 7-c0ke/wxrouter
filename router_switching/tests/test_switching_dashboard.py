"""The Control Room extension: live view from real endpoints, honest labels, and the page still loads."""
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router_switching import dashboard

INFER = {"schema": "wxrouter.inference.metrics/v1", "device": {"type": "cuda", "name": "NVIDIA GB10"},
         "node": {"gpu_uuid": "GPU-1"}, "compute_ms": {"p50": 10.0, "p95": 12.0}, "batches": {"mean_rows": 7.5},
         "config": {"max_batch_rows": 64}, "throughput": {"requests_per_s": 90.0, "rows_per_s": 99.0},
         "gpu_memory": {"allocated_mb": 11.0}, "errors": {"error_rate": 0.0}, "load": {"gpu_util_source": "nvidia-smi"}}
STATS = {"mode": "normal", "gpu_util": 0.4, "in_flight": 1, "queue_depth": 2, "p50_latency_ms": 11.0, "p95_latency_ms": 14.0}


def test_inference_box_on_cuda_is_real_gpu():
    v = dashboard.box_view({"status": "ok", "model_version": "wx-tiny"}, STATS, INFER, None)
    assert v["kind"] == "real GPU" and v["device"] == "NVIDIA GB10" and v["gpu_util"] == 0.4
    assert v["mean_batch_rows"] == 7.5 and v["requests_per_s"] == 90.0 and v["gpu_util_source"] == "nvidia-smi"
    assert v["gpu_memory_device_mb"] is None, "a GB10 reports no device memory; the page must say N/A, not a number"


def test_inference_box_on_cpu_is_labelled_cpu():
    m = {**INFER, "device": {"type": "cpu", "name": "cpu"}}
    v = dashboard.box_view({"status": "ok"}, STATS, m, None)
    assert v["kind"] == "CPU" and "no GPU" in v["gpu_util_source"] and v["gpu_util"] is None


def test_mock_box_is_simulated_and_invents_nothing():
    v = dashboard.box_view({"status": "ok", "model_version": "wx-mock-2026-11-14"}, STATS, None, None)
    assert v["kind"] == "simulated"
    assert v["compute_p50_ms"] is None and v["mean_batch_rows"] is None and v["requests_per_s"] is None


def test_team_server_is_read_from_its_info():
    assert dashboard.box_view({"status": "ok"}, STATS, None, {"device": "cuda", "mock": False, "model": "afno-19m"})["kind"] == "real GPU"
    v = dashboard.box_view({"status": "ok"}, STATS, None, {"device": "none", "mock": True, "model": "mock-1"})
    assert v["kind"] == "simulated" and v["gpu_util"] is None


def test_unreachable_box_is_unknown():
    v = dashboard.box_view(None, None, None, None)
    assert v["kind"] == "unknown" and not v["reachable"] and v["gpu_util"] is None


def test_router_view_reports_the_routers_own_decision():
    status = {"policy": "adaptive", "failovers": 3,
              "adaptive": {"margin": 0.25, "window_s": 1.0, "cooldown_s": 10.0, "return_margin": 0.2, "preferred": "box-a",
                           "active": "box-b", "switches": [{"t": 40.0, "from": "box-a", "to": "box-b", "reason": "active box down"}]},
              "backends": {"box-a": {"healthy": False, "probe_ms": None, "in_flight": 0, "err_rate": 0.5, "health_score": {"score": 99.0}},
                           "box-b": {"healthy": True, "probe_ms": 12.0, "in_flight": 2, "err_rate": 0.0, "health_score": {"score": 8.0}}}}
    metrics = {"now": 46.5, "window_s": 10.0, "overall": {"rps": 20}, "per_box": {"box-a": {"requests": 50}, "box-b": {"requests": 150}}}
    r = dashboard.router_view(status, metrics)
    assert r["last_switch"] == {"t": 40.0, "from": "box-a", "to": "box-b", "reason": "active box down", "age_s": 6.5}
    assert r["share_pct"] == {"box-a": 25.0, "box-b": 75.0}
    assert r["thresholds"] == {"margin": 0.25, "window_s": 1.0, "cooldown_s": 10.0, "return_margin": 0.2}
    assert r["backends"]["box-a"]["health_score"] == 99.0 and not r["backends"]["box-a"]["healthy"]


class FakeClient:
    def __init__(self, table):
        self.table = table

    async def get(self, url, timeout=None):
        class R:
            status_code = 200 if url in self.table else 404

            def json(_self):
                return self.table[url]
        return R()


@pytest.fixture()
def app_client():
    class Ops:
        router, boxes = "http://r", {"box-a": "http://a", "box-b": "http://b"}
        client = FakeClient({
            "http://r/router/status": {"policy": "round_robin", "failovers": 0, "adaptive": {"active": "box-a", "switches": []},
                                       "backends": {"box-a": {"healthy": True}, "box-b": {"healthy": True}}},
            "http://r/router/metrics?window=10": {"now": 5.0, "window_s": 10.0, "overall": {}, "per_box": {}},
            "http://a/health": {"status": "ok", "model_version": "wx-tiny"}, "http://a/stats": STATS, "http://a/v1/metrics": INFER,
            "http://b/health": {"status": "ok", "model_version": "wx-mock-1"}, "http://b/stats": STATS})
    app = FastAPI()
    dashboard.attach(app, Ops)
    return TestClient(app)


def test_live_endpoint_mixes_a_real_and_a_simulated_box(app_client):
    d = app_client.get("/api/switching/live").json()
    assert d["boxes"]["box-a"]["kind"] == "real GPU" and d["boxes"]["box-b"]["kind"] == "simulated"
    assert d["router"]["policy"] == "round_robin" and d["router"]["last_switch"] is None




@pytest.mark.parametrize("ui", ["ui"])
def test_existing_pages_keep_their_script_and_gain_one(ui):
    page = (ROOT / ui / "static" / "index.html").read_text()
    assert '<script src="static/app.js"></script>' in page
    assert page.count("switching/static/switching.js") == 1
    assert "router_switching.dashboard" in (ROOT / ui / "server.py").read_text()


def test_control_room_serves_its_page_and_the_new_endpoints():
    """The team's real ui/server.py app, with the panels attached."""
    sys.path.insert(0, str(ROOT / "ui"))
    for name in ("server", "agent"):
        sys.modules.pop(name, None)
    import server as ui_server

    ops = ui_server.Ops("http://127.0.0.1:9", {"box-a": "http://127.0.0.1:9"})
    with TestClient(ui_server.create_app(ops, ui_server.Agent(ops, None, "x", None))) as c:
        assert c.get("/").status_code == 200 and "wxrouter Control Room" in c.get("/").text
        assert c.get("/static/app.js").status_code == 200
        live = c.get("/api/switching/live").json()
        assert live["router"] is None and live["boxes"] == {}, "no router: no data, not even box numbers"
        assert c.get("/api/overview").status_code == 502  # unchanged: the page's own call still reports the router down
    sys.path.remove(str(ROOT / "ui"))
    for name in ("server", "agent"):
        sys.modules.pop(name, None)


def test_one_place_per_fact_no_benchmarks_card_no_second_inference_form_no_burst(app_client):
    """The cleanup: the page script adds GPU panels and a routing readout, and nothing that repeats the page."""
    assert app_client.get("/switching/static/switching.js").status_code == 200
    assert app_client.get("/switching/static/switching.css").status_code == 200
    for gone in ("/api/switching/results", "/api/switching/demo"):
        assert app_client.get(gone).status_code == 404
    assert app_client.post("/api/switching/burst").status_code in (404, 405)
    js = (ROOT / "router_switching" / "static" / "switching.js").read_text()
    for removed in ("Benchmarks", "Run inference", "sw-burst", "Intelligent switching", "api/switching/results"):
        assert removed not in js, f"{removed!r} is back in the page script"
    css = (ROOT / "router_switching" / "static" / "switching.css").read_text()
    for hidden in ("#model-card", ".kpi-active", "#pill-policy", "#boxes .box-grid"):
        assert hidden in css, f"{hidden} repeats what a section already shows and must stay hidden"
    page = (ROOT / "ui" / "static" / "index.html").read_text()
    assert page.count('id="rng-load"') == 1, "one traffic control"


def test_a_mock_box_never_shows_a_gpu_reading():
    v = dashboard.box_view({"status": "ok", "model_version": "wx-mock-1"}, {**STATS, "gpu_util": 0.25}, None, None)
    assert v["kind"] == "simulated" and v["gpu_util"] is None
