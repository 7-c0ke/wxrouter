"""Turn an API.md forecast request into model rows, and model output into the response.

Works on plain dicts so this module (and the bench) does not need pydantic.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from .model import FEATURES, MODEL_VERSION

CONTRACT_VERSION = "1.1"  # kept equal to contract.CONTRACT_VERSION by a test
_TWO_PI = 2 * math.pi


def issue_time() -> datetime:
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def build_rows(lat: float, lon: float, members: int, issued: datetime) -> np.ndarray:
    """One row per ensemble member. Member 0 is unperturbed; the rest get a fixed nudge."""
    la, lo = math.radians(lat), math.radians(lon)
    hour = (issued.hour + lon / 15) % 24 / 24 * _TWO_PI
    doy = issued.timetuple().tm_yday / 365.25 * _TWO_PI
    base = np.array([math.sin(la), math.cos(la), math.sin(lo), math.cos(lo),
                     math.sin(hour), math.cos(hour), math.sin(doy), math.cos(doy)], np.float32)
    x = np.tile(base, (members, 1))
    for m in range(1, members):
        x[m] += 0.05 * np.random.default_rng(m).standard_normal(FEATURES).astype(np.float32)
    return x


def to_physical(y: np.ndarray, lat: float, lon: float, issued: datetime, leads: list[int]) -> np.ndarray:
    """Map model output in [-1, 1] to temp_c, wind_ms, precip_mm, pressure_hpa at `leads`.

    y is (members, hours, 4) with one step per hour. Returns (members, len(leads), 4).
    """
    idx = np.array(leads) - 1
    y = y[:, idx, :].astype(np.float64)
    local_hour = (issued.hour + np.array(leads) + lon / 15) % 24
    diurnal = 5 * np.sin((local_hour - 9) / 24 * _TWO_PI)
    out = np.empty_like(y)
    out[..., 0] = 27 - 0.45 * abs(lat) + diurnal + 4 * y[..., 0]
    out[..., 1] = np.maximum(0.0, 4 + 3 * y[..., 1])
    out[..., 2] = np.maximum(0.0, 2 * y[..., 2] - 0.5)
    out[..., 3] = 1013 + 8 * y[..., 3]
    return out.round(1)


def _points(values: np.ndarray, leads: list[int], valid: list[str]) -> list[dict]:
    return [{"lead_hours": lead, "valid_time": vt, "temp_c": t, "wind_ms": w, "precip_mm": p, "pressure_hpa": q}
            for lead, vt, (t, w, p, q) in zip(leads, valid, values.tolist())]


def build_response(req: dict, y: np.ndarray, issued: datetime, box: str, timing: dict) -> dict:
    """`req` is a validated ForecastRequest as a dict; the result matches ForecastResponse."""
    hours, step, members = req["hours_ahead"], req.get("step_hours", 1), req.get("ensemble_members", 1)
    leads = list(range(step, hours + 1, step))
    valid = [(issued + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in leads]
    phys = to_physical(y, req["lat"], req["lon"], issued, leads)
    if members > 1:
        member_list = [{"member": m, "points": _points(phys[m], leads, valid)} for m in range(members)]
        forecast = _points(phys.mean(axis=0).round(1), leads, valid)
    else:
        member_list, forecast = None, _points(phys[0], leads, valid)
    return {
        "request_id": req.get("request_id"), "box": box, "model_version": MODEL_VERSION,
        "contract_version": CONTRACT_VERSION, "issued_at": issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lat": req["lat"], "lon": req["lon"], "hours_ahead": hours, "step_hours": step,
        "low_confidence": hours > 72, "native_step_hours": 1,  # this model steps hourly: nothing is interpolated
        "forecast": forecast, "members": member_list, "timing": timing,
    }
