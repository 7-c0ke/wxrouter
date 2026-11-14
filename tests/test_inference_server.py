"""The inference server must honour API.md exactly, and work behind the team's router."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import contract
from contract import ForecastResponse, Health, Stats
from inference import forecast as fc
from inference.config import EngineConfig
from inference.engine import InferenceEngine
from inference.server import create_app

ROOT = Path(__file__).resolve().parent.parent
CFG = {"backend": "numpy", "hidden": 32, "layers": 2, "warmup_runs": 1}


@pytest.fixture()
def client(tmp_path):
    engine = InferenceEngine(EngineConfig(box="box-t", **CFG))
    engine.load()  # ready before the app starts, so the first request is served
    with TestClient(create_app(engine, str(tmp_path / "metrics.json"), 0.05)) as c:
        c.metrics_path = tmp_path / "metrics.json"
        yield c


def test_contract_version_matches():
    assert fc.CONTRACT_VERSION == contract.CONTRACT_VERSION


def test_forecast_matches_contract(client):
    r = client.post("/v1/forecast", json={"lat": 37.77, "lon": -122.42, "hours_ahead": 48, "request_id": "abc"})
    assert r.status_code == 200
    body = ForecastResponse.model_validate(r.json())
    assert body.request_id == "abc" and body.box == "box-t" and not body.low_confidence
    assert [p.lead_hours for p in body.forecast] == list(range(1, 49))
    assert body.members is None
    assert body.timing.compute_ms > 0 and body.timing.total_ms >= body.timing.compute_ms
    assert all(-90 < p.temp_c < 70 and p.wind_ms >= 0 and p.precip_mm >= 0 and 900 < p.pressure_hpa < 1100
               for p in body.forecast)


def test_step_hours_ensemble_and_low_confidence(client):
    r = client.post("/v1/forecast", json={"lat": -10, "lon": 20, "hours_ahead": 96, "step_hours": 6,
                                          "ensemble_members": 4})
    body = ForecastResponse.model_validate(r.json())
    assert body.low_confidence
    assert [p.lead_hours for p in body.forecast] == list(range(6, 97, 6))
    assert [m.member for m in body.members] == [0, 1, 2, 3]
    assert body.members[0].points != body.members[1].points
    temps = [m.points[0].temp_c for m in body.members]
    assert body.forecast[0].temp_c == pytest.approx(sum(temps) / 4, abs=0.06)


def test_same_request_same_answer(client):
    req = {"lat": 51.5, "lon": 0.1, "hours_ahead": 24}
    a, b = client.post("/v1/forecast", json=req).json(), client.post("/v1/forecast", json=req).json()
    assert a["forecast"] == b["forecast"]


def test_invalid_input_is_422(client):
    assert client.post("/v1/forecast", json={"lat": 91, "lon": 0, "hours_ahead": 1}).status_code == 422
    assert client.post("/v1/forecast", json={"lat": 0, "lon": 0, "hours_ahead": 121}).status_code == 422


def test_health_stats_and_metrics(client):
    for _ in range(5):
        client.post("/v1/forecast", json={"lat": 1, "lon": 2, "hours_ahead": 12})
    h = Health.model_validate(client.get("/health").json())
    assert h.status == "ok" and h.box == "box-t"
    s = Stats.model_validate(client.get("/stats").json())
    assert s.requests_total == 5 and s.errors_total == 0 and s.mode == "normal"
    assert s.max_concurrency == 1 and 0 <= s.gpu_util <= 1 and s.p50_latency_ms > 0
    m = client.get("/v1/metrics").json()
    assert m["schema"] == "wxrouter.inference.metrics/v1" and m["latency_ms"]["count"] == 5
    time.sleep(0.2)
    assert json.loads(client.metrics_path.read_text())["box"] == "box-t"


def test_health_is_503_until_loaded():
    engine = InferenceEngine(EngineConfig(box="box-t", **CFG))
    app = create_app(engine)
    engine.start = lambda: _never()  # model never finishes loading
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/health").json()["status"] == "down"
        assert c.post("/v1/forecast", json={"lat": 1, "lon": 2, "hours_ahead": 12}).status_code == 503


async def _never():
    import asyncio
    await asyncio.sleep(3600)


# ------------------------------------------------------------------ integration smoke test
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, ok, timeout: float = 30.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = httpx.get(url, timeout=1)
            if ok(r):
                return r
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"{url} not ready after {timeout}s")


@pytest.mark.parametrize("server", ["inference", "mock"])
def test_router_in_front_of_two_inference_servers(server):
    """Two real inference servers behind the team's unmodified router.py, including failover."""
    pa, pb, pr = free_port(), free_port(), free_port()
    common = (["--backend", "numpy", "--hidden", "32", "--layers", "2"]
              if server == "inference" else ["--base-ms", "1", "--per-hour-ms", "0.1"])
    entry = ["-m", "inference.server"] if server == "inference" else ["mock_server.py"]
    procs = {}

    def spawn(name, args):
        procs[name] = subprocess.Popen([sys.executable, *args], cwd=ROOT,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        spawn("box-a", [*entry, "--name", "box-a", "--port", str(pa), *common])
        spawn("box-b", [*entry, "--name", "box-b", "--port", str(pb), *common])
        spawn("router", ["router.py", "--port", str(pr), "--policy", "round_robin",
                         "--backend", f"box-a=http://127.0.0.1:{pa}", "--backend", f"box-b=http://127.0.0.1:{pb}"])
        base = f"http://127.0.0.1:{pr}"
        wait_for(f"http://127.0.0.1:{pa}/health", lambda r: r.status_code == 200)
        wait_for(f"http://127.0.0.1:{pb}/health", lambda r: r.status_code == 200)
        wait_for(f"{base}/health", lambda r: r.status_code == 200)
        time.sleep(1.0)  # let the router's health poll see both boxes

        req = {"lat": 37.7, "lon": -122.4, "hours_ahead": 24}
        served = set()
        for _ in range(8):
            r = httpx.post(f"{base}/v1/forecast", json=req, timeout=10)
            assert r.status_code == 200
            ForecastResponse.model_validate(r.json())
            served.add(r.headers["X-Served-By"])
        assert served == {"box-a", "box-b"}

        procs["box-b"].kill()
        procs["box-b"].wait()
        for _ in range(6):
            r = httpx.post(f"{base}/v1/forecast", json=req, timeout=10)
            assert r.status_code == 200 and r.headers["X-Served-By"] == "box-a"
    finally:
        for p in procs.values():
            p.kill()
            p.wait()
