"""Shared API contract for the forecast servers (real and mock) and the router.

Both the real GPU forecast server and the mock MUST accept and return exactly
these shapes, so the router and load generator never need to know which one
they are talking to. See API.md for the human-readable version.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

CONTRACT_VERSION = "1.1"
MAX_HOURS_AHEAD = 120  # 5 days; most accurate under 72
MAX_ENSEMBLE_MEMBERS = 20


# ---------------------------------------------------------------- forecast
class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    hours_ahead: int = Field(..., ge=1, le=MAX_HOURS_AHEAD)
    step_hours: int = Field(1, ge=1, le=24, description="Spacing between forecast points")
    ensemble_members: int = Field(1, ge=1, le=MAX_ENSEMBLE_MEMBERS)
    request_id: Optional[str] = Field(None, description="Client-chosen id, echoed back")


class ForecastPoint(BaseModel):
    lead_hours: int
    valid_time: str  # ISO-8601 UTC
    temp_c: float
    wind_ms: float
    pressure_hpa: float
    # Optional: not every model outputs these. Clients must handle null.
    precip_mm: Optional[float] = None  # rain in the hour ending at valid_time
    humidity_pct: Optional[float] = None  # relative humidity, 0..100
    cloud_pct: Optional[float] = None  # cloud cover, 0..100
    # The model runs in steps of `model_step_hours` (6). Points between steps are
    # linear interpolations, so they carry no extra information.
    interpolated: bool = False


class EnsembleMember(BaseModel):
    member: int
    points: list[ForecastPoint]


class Timing(BaseModel):
    queue_ms: float  # waiting for a GPU slot
    compute_ms: float  # model run
    total_ms: float  # server-side wall time


class ForecastResponse(BaseModel):
    request_id: Optional[str]
    box: str  # which box produced it, e.g. "box-a"
    model_version: str
    contract_version: str = CONTRACT_VERSION
    issued_at: str  # ISO-8601 UTC
    lat: float
    lon: float
    hours_ahead: int
    step_hours: int
    low_confidence: bool  # true when hours_ahead > 72
    model_step_hours: int = 6  # native time step; other points are interpolated
    forecast: list[ForecastPoint]  # ensemble mean when members > 1
    members: Optional[list[EnsembleMember]] = None
    timing: Timing


# ---------------------------------------------------------------- health/stats
class Health(BaseModel):
    status: Literal["ok", "degraded", "down"]
    box: str
    model_version: str
    contract_version: str = CONTRACT_VERSION
    uptime_s: float


class Stats(BaseModel):
    box: str
    in_flight: int  # requests currently on the GPU
    queue_depth: int  # requests waiting for a GPU slot
    max_concurrency: int
    gpu_util: float  # 0..1
    ewma_latency_ms: float  # server-side total_ms, smoothed
    p50_latency_ms: float  # over the last ~200 requests
    p95_latency_ms: float
    requests_total: int
    errors_total: int
    mode: str  # mock only: normal | slow | error | hang ; real server: "normal" or "training"
