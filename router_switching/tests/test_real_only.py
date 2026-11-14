"""Real GPUs only: mock servers and CPU boxes cannot reach the page's live endpoints.

The app under test is the team's real ui/server.py, built the way its command line builds it by default.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router_switching import dashboard

CUDA = {"schema": "wxrouter.inference.metrics/v1", "device": {"type": "cuda", "name": "NVIDIA GB10"},
        "node": {"gpu_uuid": "GPU-1"}, "compute_ms": {"p50": 10.0, "p95": 12.0, "p99": 13.0, "count": 5},
        "batches": {"mean_rows": 2.0}, "config": {"max_batch_rows": 64}, "throughput": {"requests_per_s": 9.0, "rows_per_s": 9.0},
        "gpu_memory": {"allocated_mb": 11.0}, "errors": {"error_rate": 0.0}, "load": {"gpu_util_source": "nvidia-smi"}}
STATS = {"mode": "normal", "gpu_util": 0.4, "in_flight": 0, "queue_depth": 0, "p50_latency_ms": 11.0, "p95_latency_ms": 14.0,
         "requests_total": 3, "errors_total": 0}
ROUTER = {"http://r/router/status": {"policy": "balanced", "failovers": 0, "adaptive": {"active": "box-a", "switches": []},
                                     "backends": {"box-a": {"healthy": True}, "box-b": {"healthy": True}}},
          "http://r/router/metrics": {"now": 5.0, "window_s": 10.0, "overall": {"rps": 0, "p95_ms": None, "errors": 0}, "per_box": {}}}


def box(url, version, metrics=None, info=None, stats=STATS):
    t = {f"{url}/health": {"status": "ok", "model_version": version}, f"{url}/stats": {**stats, "model_version": version}}
    if metrics:
        t[f"{url}/v1/metrics"] = metrics
    if info:
        t[f"{url}/v1/info"] = info
    return t


class Client:
    """Answers GETs from a table, 404 otherwise; no network."""
    def __init__(self, table):
        self.table = table

    async def get(self, url, timeout=None, params=None):
        key = url.split("?")[0]
        table = self.table

        class R:
            status_code = 200 if key in table else 404
            text = ""

            def json(_self):
                if key not in table:
                    raise ValueError("not JSON")
                return table[key]
        return R()

    async def post(self, url, **kw):
        raise AssertionError(f"a refused page must not send anything on: POST {url}")

    async def aclose(self):
        pass


def app_for(table, real_only=True):
    sys.path.insert(0, str(ROOT / "ui"))
    for name in ("server", "agent"):
        sys.modules.pop(name, None)
    import server as ui
    ops = ui.Ops("http://r", {"box-a": "http://a", "box-b": "http://b"})
    app = ui.create_app(ops, ui.Agent(ops, None, "x", None), real_only=real_only)
    c = TestClient(app)
    c.__enter__()
    ops.client = Client(table)
    sys.path.remove(str(ROOT / "ui"))
    return c


API = ["/api/overview", "/api/model", "/api/switching/live", "/api/router/config"]
MOCK_A = {**ROUTER, **box("http://a", "wx-mock-2026-11-14"), **box("http://b", "wx-tiny-untrained-0.1", CUDA)}


@pytest.mark.parametrize("path", API)
def test_a_mock_box_is_refused_on_every_live_endpoint(path):
    c = app_for(MOCK_A)
    r = c.get(path)
    assert r.status_code == 503 and "mock server" in r.json()["detail"] and list(r.json()["refused"]) == ["box-a"]


@pytest.mark.parametrize("path", ["/api/forecast", "/api/load", "/api/router/config", "/api/chat"])
def test_a_refused_page_sends_no_request_and_starts_no_traffic(path):
    c = app_for(MOCK_A)
    assert c.post(path, json={"lat": 1, "lon": 1, "rps": 5, "messages": []}).status_code == 503


def test_the_weather_servers_own_mock_mode_is_refused_too():
    info = {"device": "none", "mock": True, "model": "mock-1"}
    c = app_for({**ROUTER, **box("http://a", "afno-19m", info=info), **box("http://b", "afno-19m", info=info)})
    assert c.get("/api/overview").status_code == 503 and len(c.get("/api/switching/guard").json()["refused"]) == 2


def test_a_cpu_box_is_refused_no_silent_cpu_fallback():
    cpu = {**CUDA, "device": {"type": "cpu", "name": "cpu"}}
    c = app_for({**ROUTER, **box("http://a", "wx-tiny-untrained-0.1", cpu), **box("http://b", "wx-tiny-untrained-0.1", CUDA)})
    r = c.get("/api/switching/live")
    assert r.status_code == 503 and "runs on CPU" in r.json()["detail"]
    info = {"device": "cpu", "mock": False, "model": "afno-19m"}   # the trained model that fell back to CPU
    c = app_for({**ROUTER, **box("http://a", "afno-19m-abc", info=info), **box("http://b", "afno-19m-abc", info=info)})
    assert c.get("/api/overview").status_code == 503


def test_real_gpu_boxes_pass_and_show_their_own_readings():
    c = app_for({**ROUTER, **box("http://a", "wx-tiny-untrained-0.1", CUDA), **box("http://b", "wx-tiny-untrained-0.1", CUDA)})
    assert c.get("/api/switching/guard").json() == {"real_only": True, "refused": {}, "detail": None}
    live = c.get("/api/switching/live").json()
    a = live["boxes"]["box-a"]
    assert {b["kind"] for b in live["boxes"].values()} == {"real GPU"} and a["state"] == "ONLINE" and a["device_type"] == "cuda"
    assert a["compute_p99_ms"] == 13.0 and a["mean_batch_rows"] == 2.0 and a["requests_total"] == 3 and a["last_ok_ts"]


def test_a_routing_change_is_recorded_with_who_asked_and_a_change_made_elsewhere_is_called_that():
    table = {**ROUTER, **box("http://a", "wx-tiny-untrained-0.1", CUDA), **box("http://b", "wx-tiny-untrained-0.1", CUDA)}
    c = app_for(table)
    first = c.get("/api/switching/live").json()["config_changes"]
    assert len(first) == 1 and first[0]["policy"] == "balanced" and "first reading" in first[0]["source"]
    table["http://r/router/status"] = {**ROUTER["http://r/router/status"], "policy": "static"}   # nobody asked this server
    changes = c.get("/api/switching/live").json()["config_changes"]
    assert changes[-1]["policy"] == "static" and "not through this Control Room" in changes[-1]["source"]
    assert len(c.get("/api/switching/live").json()["config_changes"]) == 2, "one entry per change, not one per poll"


def test_an_offline_box_shows_offline_and_no_numbers():
    dashboard.LAST_OK.clear()
    c = app_for({**ROUTER, **box("http://b", "wx-tiny-untrained-0.1", CUDA)})   # box-a answers nothing
    assert c.get("/api/switching/guard").json()["refused"] == {}, "a silent box is not a mock; it is shown as offline"
    a = c.get("/api/switching/live").json()["boxes"]["box-a"]
    assert a["state"] == "OFFLINE" and a["kind"] == "unknown" and a["last_ok_ts"] is None
    assert all(a[k] is None for k in ("gpu_util", "latency_p50_ms", "compute_p50_ms", "requests_total", "gpu_memory_mb", "in_flight"))


def test_numbers_a_server_did_not_measure_are_none_not_zero():
    busy = {**CUDA, "load": {"gpu_util_source": "busy_slots"}, "compute_ms": {"p50": 0.0, "p95": 0.0, "count": 0}}
    v = dashboard.box_view({"status": "ok"}, {**STATS, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}, busy, None)
    assert v["gpu_util"] is None, "busy slots is not a GPU reading"
    assert v["compute_p50_ms"] is None and v["latency_p50_ms"] is None, "an empty window is not 0 ms"
    w = dashboard.box_view({"status": "ok"}, {**STATS, "gpu_util": 0.0}, None, {"device": "cuda", "mock": False, "model": "afno-19m"})
    assert w["gpu_util"] is None, "the weather server's /stats says 0.0 when it has no reading; only its /metrics counts"
    dashboard.machine_view(w, {"gpu": {"util_pct": 13, "proc_mem_mb": 20070, "temp_c": 50, "power_w": 12.9}})
    assert w["gpu_util"] == 0.13 and w["gpu_proc_mem_mb"] == 20070
    dashboard.machine_view(w, {"gpu": {"util_pct": None}})
    assert w["gpu_util"] is None and w["gpu_proc_mem_mb"] is None


def test_dev_mode_is_the_only_way_to_see_a_mock_and_the_command_line_defaults_to_real_only():
    c = app_for(MOCK_A, real_only=False)
    assert c.get("/api/switching/live").status_code == 200 and c.get("/api/switching/guard").status_code == 404
    src = (ROOT / "ui" / "server.py").read_text()
    assert 'create_app(ops, agent, real_only=not a.dev' in src and '"--dev", action="store_true"' in src
    for stack in ("dev_stack.py", "ui/dev_stack_supervised.py"):
        assert '"--dev"' in (ROOT / stack).read_text(), f"{stack} starts mocks, so it must say so"
    for fz in list((ROOT / "fuzzball").glob("*.fz")) + [ROOT / "deploy" / "inference-serve.fz", ROOT / "deploy" / "inference-demo.fz"]:
        text = fz.read_text()
        run = [l for l in text.splitlines() if l.strip().startswith(("exec python", "python "))]
        assert not any("mock_server.py" in l or "--mock" in l or "mock_llm" in l for l in run), f"{fz.name} starts a mock"


def test_fuzzball_status_is_parsed_from_the_cli_and_a_stage_that_is_not_running_holds_no_node():
    d = dashboard.parse_describe("Name:      real-stack\nStatus:    STAGE_STATUS_STARTED\nCreated:   2026-11-14 02:30:47PM\n"
                                 "Service  | Started  | box-a | 3m55s |\nService  | Pending  | box-b |  |\nJob | Finished | prep | 1s |")
    assert d["status"] == "STARTED" and [(s["name"], s["status"]) for s in d["stages"]] == [("box-a", "STARTED"), ("box-b", "PENDING"), ("prep", "FINISHED")]
    p = dashboard.parse_placement("x [SERVICE] box-a: Resources allocated (node: 192.168.99.37/7331, requested cpu: 12, requested devices: 1, more")
    assert p == {"box-a": {"node": "192.168.99.37", "gpus": 1}}
