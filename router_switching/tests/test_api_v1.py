"""The /api/v1 JSON API: real shapes in, honest values out, and the team's page untouched.

The box answers below are copies of what the real servers returned on SPARK27/SPARK28 on Nov 14, 2026
(the weather team's AFNO server and the inference server), trimmed.
"""
import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router_switching import api, dashboard

AFNO = {
    "/health": {"status": "ok", "box": "box-a", "model_version": "afno-19m-65f9f3bb3270"},
    "/stats": {"box": "box-a", "in_flight": 0, "queue_depth": 0, "gpu_util": 0.0, "p50_latency_ms": 26.6, "p95_latency_ms": 3088.0,
               "requests_total": 244, "errors_total": 0, "mode": "normal", "model_version": "afno-19m-65f9f3bb3270", "mock": False},
    "/v1/info": {"model": "afno-19m", "params": 19128832, "device": "cuda", "mock": False, "latest": "2026-11-14T12", "ms_per_unit": 9.82},
    "/metrics": {"ms_per_unit": 9.08, "gpu": {"util_pct": 13, "temp_c": 50, "power_w": 12.9, "proc_mem_mb": 20070}},
}
TINY = {
    "/health": {"status": "ok", "box": "box-b", "model_version": "wx-tiny-untrained-0.1"},
    "/stats": {"box": "box-b", "in_flight": 2, "queue_depth": 1, "gpu_util": 0.05, "p50_latency_ms": 6.9, "p95_latency_ms": 30.4,
               "requests_total": 31584, "errors_total": 0, "mode": "normal"},
    "/v1/metrics": {"schema": "wxrouter.inference.metrics/v1", "model_version": "wx-tiny-untrained-0.1",
                    "device": {"type": "cuda", "name": "NVIDIA GB10"}, "node": {"gpu_uuid": "GPU-a54b10ad"},
                    "compute_ms": {"count": 2000, "p50": 5.741, "p95": 23.276, "p99": 27.677}, "latency_ms": {"mean": 10.6},
                    "throughput": {"requests_per_s": 11.7, "rows_per_s": 11.7}, "batches": {"mean_rows": 1.13},
                    "config": {"max_batch_rows": 64}, "errors": {"error_rate": 0.0, "timeouts": 0},
                    "gpu_memory": {"allocated_mb": 20.3, "peak_allocated_mb": 20.3, "device_used_mb": None}},
}
MOCK = {"/health": {"status": "ok", "model_version": "wx-mock-2026-11-14"},
        "/stats": {"in_flight": 1, "queue_depth": 0, "gpu_util": 0.25, "p50_latency_ms": 60.0, "p95_latency_ms": 90.0,
                   "requests_total": 9, "errors_total": 0, "mode": "normal"}}
ROUTER = {
    "/router/status": {"policy": "balanced", "failovers": 2, "config": {"defaults": {"policy": "adaptive"}, "drained": []},
                       "adaptive": {"margin": 0.25, "window_s": 1.0, "cooldown_s": 10.0, "return_margin": 0.2, "preferred": None,
                                    "active": "box-b", "switches": [{"t": 4.0, "from": "box-a", "to": "box-b", "reason": "active box down"}]},
                       "backends": {"box-a": {"healthy": True, "probe_ms": 18.0, "in_flight": 0, "err_rate": 0.0, "health_score": {"score": 19.0}},
                                    "box-b": {"healthy": True, "probe_ms": 19.0, "in_flight": 1, "err_rate": 0.0, "health_score": {"score": 16.0}}}},
    "/router/metrics": {"now": 50.0, "window_s": 10.0, "overall": {"requests": 80, "rps": 8.0, "errors": 0, "failovers": 0,
                                                                  "p50_ms": 82.0, "p95_ms": 199.0, "p99_ms": 248.0},
                        "per_box": {"box-a": {"requests": 30, "rps": 3.0, "p95_ms": 170.0}, "box-b": {"requests": 50, "rps": 5.0, "p95_ms": 210.0}}},
}


class Client:
    """Serves the tables above by URL prefix; anything else is a refused connection."""
    def __init__(self, hosts):
        self.hosts, self.posts = hosts, []

    def _find(self, url):
        url = url.split("?")[0]
        for base, table in self.hosts.items():
            if url.startswith(base) and url[len(base):] in table:
                return table[url[len(base):]]
        return None

    async def get(self, url, timeout=None, params=None):
        hit = self._find(url)
        known = any(url.startswith(b) for b in self.hosts)
        if not known:
            raise dashboard.httpx.ConnectError("refused")

        class R:
            status_code = 200 if hit is not None else 404
            text = ""

            def json(_s):
                if hit is None:
                    raise ValueError("no JSON")
                return hit
        return R()

    async def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))

        class R:
            status_code, text = 200, ""
            headers = {"x-served-by": "box-b", "x-router-failover": "false", "x-router-policy": "balanced"}

            def json(_s):
                return {"box": "box-b", "model_version": "afno-19m-65f9f3bb3270", "forecast": [{"lead_hours": 6, "temp_c": 5.53}],
                        "timing": {"compute_ms": 74.1}}
        return R()


def client_for(hosts, boxes=None):
    class Ops:
        router = "http://r"
        client = Client(hosts)
    Ops.boxes = boxes or {"box-a": "http://a", "box-b": "http://b"}
    app = FastAPI()
    dashboard.attach(app, Ops)
    return TestClient(app), Ops


REAL = {"http://r": ROUTER, "http://a": AFNO, "http://b": TINY}


def test_trained_afno_box_reads_gpu_from_its_metrics_not_from_the_zero_in_stats():
    api.LAST_OK.clear()
    c, _ = client_for(REAL)
    a = c.get("/api/v1/gpus/box-a").json()
    assert (a["state"], a["kind"], a["cuda"]) == ("ONLINE", "real GPU", True)
    assert a["model"] == {"version": "afno-19m-65f9f3bb3270", "name": "afno-19m", "params": 19128832, "trained": True, "starts_from": "2026-11-14T12"}
    assert a["gpu"]["utilization"] == 0.13 and "nvidia-smi" in a["gpu"]["utilization_source"]
    assert a["gpu"]["memory_process_mb"] == 20070 and a["gpu"]["temperature_c"] == 50 and a["gpu"]["power_w"] == 12.9
    assert a["gpu"]["memory_total_mb"] is None, "a GB10 reports no device total; the API must not invent one"
    assert a["inference"]["gpu_call_ms"] == {"p50": None, "p95": None, "p99": None}, "this server has no per-call percentiles"
    assert a["inference"]["request_ms"] == {"p50": 26.6, "p95": 3088.0} and a["inference"]["ms_per_model_step"] == 9.08
    assert a["batch"] == {"mean_rows": None, "max_rows": None} and a["requests"]["total"] == 244 and a["telemetry"]["last_ok_ts"]


def test_inference_server_box_reports_batching_and_is_marked_untrained():
    c, _ = client_for(REAL)
    b = c.get("/api/v1/gpus").json()["gpus"]["box-b"]
    assert b["kind"] == "real GPU" and b["cuda"] is True and b["device"] == "NVIDIA GB10" and b["model"]["trained"] is False
    assert b["inference"]["gpu_call_ms"] == {"p50": 5.741, "p95": 23.276, "p99": 27.677}
    assert b["batch"] == {"mean_rows": 1.13, "max_rows": 64} and b["throughput"]["requests_per_s"] == 11.7
    assert b["gpu"]["memory_tensors_mb"] == 20.3 and b["queue"] == {"depth": 1, "in_flight": 2}


def test_unmeasured_is_null_never_a_default():
    busy = {**TINY, "/v1/metrics": {**TINY["/v1/metrics"], "load": {"gpu_util_source": "busy_slots"}, "compute_ms": {"count": 0, "p50": 0.0, "p95": 0.0}},
            "/stats": {**TINY["/stats"], "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}}
    no_smi = {**AFNO, "/metrics": {"gpu": {"util_pct": None}}}
    c, _ = client_for({"http://r": ROUTER, "http://a": no_smi, "http://b": busy})
    g = c.get("/api/v1/gpus").json()["gpus"]
    assert g["box-a"]["gpu"]["utilization"] is None, "/stats said 0.0 with no reading; that is not a measurement"
    assert g["box-b"]["gpu"]["utilization"] is None, "busy slots is not a GPU reading"
    assert g["box-b"]["inference"]["gpu_call_ms"]["p50"] is None and g["box-b"]["inference"]["request_ms"]["p50"] is None


def test_a_mock_is_labelled_simulated_and_has_no_gpu_reading():
    c, _ = client_for({"http://r": ROUTER, "http://a": MOCK, "http://b": TINY})
    a = c.get("/api/v1/gpus/box-a").json()
    assert a["kind"] == "simulated" and a["gpu"]["utilization"] is None and a["cuda"] is None and a["model"]["trained"] is False


def test_offline_box_and_offline_router_have_a_state_and_no_numbers():
    api.LAST_OK.clear()
    c, _ = client_for({"http://b": TINY})   # no router, no box-a
    g = c.get("/api/v1/gpus/box-a").json()
    assert g["state"] == "OFFLINE" and g["kind"] == "unknown" and g["telemetry"] == {"last_ok_ts": None, "age_s": None}
    assert g["gpu"]["utilization"] is None and g["requests"]["total"] is None and g["queue"]["depth"] is None
    assert c.get("/api/v1/routing").json()["state"] == "OFFLINE"
    p = c.get("/api/v1/performance").json()
    assert p["state"] == "OFFLINE" and "rps" not in p
    s = c.get("/api/v1/status").json()
    assert set(s) == {"ts", "gpus", "routing", "performance", "fuzzball"} and s["fuzzball"]["available"] is False


def test_routing_says_which_rule_routes_and_where_requests_went():
    c, _ = client_for(REAL)
    r = c.get("/api/v1/routing").json()
    assert (r["policy"], r["startup_policy"], r["failovers_total"]) == ("balanced", "adaptive", 2)
    assert r["adaptive"]["routing"] is False and r["adaptive"]["active"] == "box-b" and len(r["adaptive"]["switches"]) == 1
    assert r["adaptive"]["thresholds"] == {"margin": 0.25, "window_s": 1.0, "cooldown_s": 10.0, "return_margin": 0.2}
    assert r["boxes"]["box-a"]["share_pct"] == 37.5 and r["boxes"]["box-b"]["rps"] == 5.0 and r["boxes"]["box-b"]["health_score"] == 16.0
    p = c.get("/api/v1/performance?window=60").json()
    assert p["rps"] == 8.0 and p["p99_ms"] == 248.0 and p["errors"] == 0


def test_inference_returns_the_servers_own_answer_and_can_target_one_box():
    c, ops = client_for(REAL)
    d = c.post("/api/v1/inference", json={"lat": 41.98, "lon": -87.9, "hours_ahead": 48}).json()
    assert d["sent_to"] == "router" and d["served_by"] == "box-b" and d["failover"] is False and d["policy"] == "balanced"
    assert d["response"]["forecast"] == [{"lead_hours": 6, "temp_c": 5.53}] and d["response"]["timing"]["compute_ms"] == 74.1
    assert ops.client.posts[-1][0] == "http://r/v1/forecast" and "box" not in ops.client.posts[-1][1]
    c.post("/api/v1/inference", json={"lat": 1, "lon": 1, "box": "box-a"})
    assert ops.client.posts[-1][0] == "http://a/v1/forecast"
    assert c.post("/api/v1/inference", json={"lat": 1, "lon": 1, "box": "nope"}).status_code == 404
    assert c.post("/api/v1/inference", json={"lat": 999, "lon": 1}).status_code == 422
    assert c.get("/api/v1/gpus/nope").status_code == 404


def test_fuzzball_status_gives_nodes_and_turns_offline_into_pending():
    d = api.parse_describe("Name:      real-stack\nStatus:    STAGE_STATUS_STARTED\nCreated:   2026-11-14 02:30:47PM\n"
                           "Service  | Started  | box-a | 3m55s |\nService  | Pending  | box-b |  |\nJob | Finished | prep | 1s |")
    assert d["status"] == "STARTED" and [(s["name"], s["status"]) for s in d["stages"]] == [("box-a", "STARTED"), ("box-b", "PENDING"), ("prep", "FINISHED")]
    assert api.parse_placement("x [SERVICE] box-a: Resources allocated (node: 192.168.99.37/7331, requested cpu: 12, requested devices: 1, y") == \
        {"box-a": {"node": "192.168.99.37", "gpus": 1}}
    gpus = {"box-a": {"state": "ONLINE"}, "box-b": {"state": "OFFLINE"}}
    fb = {"available": True, "stages": [{"name": "box-a", "status": "STARTED", "node": "CIQ-CHI-SPARK28"}, {"name": "box-b", "status": "PENDING", "node": None}]}
    api.place_on_nodes(gpus, fb)
    assert gpus["box-a"]["node"] == "CIQ-CHI-SPARK28" and gpus["box-b"]["state"] == "PENDING" and gpus["box-b"]["node"] is None
    assert asyncio.run(api.read_fuzzball(None))["available"] is False


def test_other_origins_can_read_it_and_the_teams_page_and_older_endpoints_are_untouched():
    c, _ = client_for(REAL)
    assert c.get("/api/v1").headers["access-control-allow-origin"] == "*" and "GET /api/v1/gpus" in c.get("/api/v1").json()["endpoints"]
    assert c.options("/api/v1/inference").status_code == 204
    assert "access-control-allow-origin" not in c.get("/api/switching/live").headers, "only /api/v1 is opened to other origins"
    assert c.get("/api/switching/live").status_code == 200 and c.get("/api/switching/results").status_code == 200
    page = (ROOT / "ui" / "static" / "index.html").read_text()
    assert "switching" not in page, "this API adds no card and no script to the team's page"
