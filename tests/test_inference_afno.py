"""The AFNO adapter, checked against the weather team's own rollout with a small stand-in network.

The real model needs NVIDIA PhysicsNeMo and the 77 MB weights, which are not on a laptop. The
adapter only calls `model(states)`, so any network with the same input and output shape exercises
every line of it.
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "weather"))

import forecast_core as core
import wx_common as wx
from fastapi.testclient import TestClient

from contract import ForecastResponse
from inference.afno import AfnoBackend, AfnoService
from inference.config import EngineConfig
from inference.engine import InferenceEngine
from inference.server import create_app

WHEN = datetime.datetime(2026, 11, 14, 0)


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    data = tmp_path_factory.mktemp("bundle")
    (data / "stats.json").write_text(json.dumps({"channels": wx.CHANNEL_NAMES,
                                                 "mean": [280.0] * 20, "std": [10.0] * 20}))
    (data / "states").mkdir()
    rng = np.random.default_rng(0)
    np.save(data / "states" / "2026111400.npy", rng.standard_normal(core.STATE_SHAPE).astype(np.float16) * 0.5)
    torch.manual_seed(0)
    fc = object.__new__(core.Forecaster)  # skip __init__, which builds the real AFNO
    fc.device, fc.batch = torch.device("cpu"), 20
    fc.mean, fc.std = wx.read_stats(str(data))
    fc.store = core.StateStore(str(data))
    fc.model = torch.nn.Sequential(torch.nn.Conv2d(20, 20, 3, padding=1), torch.nn.Tanh()).eval()
    return AfnoBackend(forecaster=fc)


def test_batched_rollout_matches_the_teams_rollout(backend):
    """Rows with different places and lengths in one batch == forecast_core.rollout_point one at a time."""
    state = backend.fc.store.get(WHEN)
    x = core.perturbed_states(state, WHEN, range(3))
    aux = np.array([[41.98, -87.9], [-33.9, 151.2], [89.0, 359.9]], np.float32)
    steps = [4, 2, 3]
    z = backend.rollout(x, 4, steps, aux)
    assert z.shape == (3, 4, 20)
    for i, n in enumerate(steps):
        ref = core.rollout_point(backend.fc.model, "cpu", x[i:i + 1], n, float(aux[i, 0]), float(aux[i, 1]))
        assert np.allclose(z[i, :n], ref[0], atol=1e-5)


def test_server_answers_the_contract_with_the_afno_path(backend):
    engine = InferenceEngine(EngineConfig(box="box-w", warmup_runs=1, max_batch_rows=20), backend=backend)
    engine.load()
    with TestClient(create_app(engine, service=AfnoService(backend))) as c:
        r = c.post("/v1/forecast", json={"lat": 41.98, "lon": -87.9, "hours_ahead": 20, "step_hours": 2,
                                         "ensemble_members": 3, "request_id": "ohare"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["box"] == "box-w" and body["model_version"] == "afno-19m" and body["request_id"] == "ohare"
        assert body["init_time"] == "2026-11-14T00" and body["native_step_hours"] == 6 and body["interpolated"]
        assert [p["lead_hours"] for p in body["forecast"]] == list(range(2, 21, 2))
        assert all(p["precip_mm"] is None for p in body["forecast"])  # the model has no rain output
        assert [m["member"] for m in body["members"]] == [0, 1, 2]
        assert body["units"] == 3 * 4  # 20 h rounds up to four 6-hour steps
        assert body["timing"]["compute_ms"] > 0
        # every field the contract requires is there (precip aside, which this model cannot give)
        for p in body["forecast"]:
            p["precip_mm"] = 0.0
        for m in body["members"]:
            for p in m["points"]:
                p["precip_mm"] = 0.0
        ForecastResponse.model_validate(body)
        assert c.get("/health").json()["model_version"] == "afno-19m"
        assert c.get("/v1/metrics").json()["device"]["backend"] == "afno"


def test_same_numbers_as_the_teams_forecaster(backend):
    """The adapter's mean series equals Forecaster.run() + point_series, the path serve/forecast_server.py uses."""
    fc, state = backend.fc, backend.fc.store.get(WHEN)
    series, _ = fc.run(state, WHEN, 3, 2, 0, 10.0, 20.0, detail=True)
    engine = InferenceEngine(EngineConfig(box="box-w", warmup_runs=1), backend=backend)
    engine.load()
    with TestClient(create_app(engine, service=AfnoService(backend))) as c:
        body = c.post("/v1/forecast", json={"lat": 10.0, "lon": 20.0, "hours_ahead": 18, "step_hours": 6,
                                            "ensemble_members": 2}).json()
    assert [p["temp_c"] for p in body["forecast"]] == pytest.approx(series["t2m_c_mean"], abs=0.011)
    assert [p["pressure_hpa"] for p in body["forecast"]] == pytest.approx(series["msl_hpa_mean"], abs=0.011)
