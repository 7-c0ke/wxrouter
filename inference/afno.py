"""Serve the weather team's trained AFNO model through the inference engine.

This is the adapter between two pieces that already exist:

  weather/forecast_core.py   the model, the stored analyses, ensemble noise, units
  serve/forecast_server.py   to_v1(): the team's own API.md response builder

and the engine in this package, which adds batching across requests, timeouts,
retries and the metrics document. A row is one ensemble member's starting state
(20, 120, 240); `aux` carries each row's (lat, lon); one step is 6 hours.

  python -m inference.server --model afno --weights /wx/bundle/weights/afno_event.pt --data /wx/bundle

Needs PyTorch, NumPy and NVIDIA PhysicsNeMo, like the team's server.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from .model import sort_by_horizon, unsort

ROOT = Path(os.environ.get("WX_REPO") or Path(__file__).resolve().parent.parent)  # where weather/ and serve/ are
for _p in (ROOT / "weather", ROOT / "serve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STEP_HOURS = 6


class AfnoBackend:
    """Engine backend around forecast_core.Forecaster. Batches rows from different requests."""

    name = "afno"

    def __init__(self, weights: str | None = None, data_dir: str | None = None, device: str = "auto",
                 forecaster=None):
        import forecast_core
        import torch

        self.core, self.torch = forecast_core, torch
        self.fc = forecaster or forecast_core.Forecaster(weights, data_dir, device)
        self.device = str(self.fc.device).split(":")[0]
        self.device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else self.device
        self.precision = "bf16-autocast" if self.device == "cuda" else "fp32"
        self.model_version = "afno-19m"
        if weights:
            digest = hashlib.sha256()
            with open(weights, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            self.model_version = f"afno-19m-{digest.hexdigest()[:12]}"  # same naming as serve/forecast_server.py

    def warmup_input(self):
        shape = self.core.STATE_SHAPE
        return np.zeros((1, *shape), np.float32), 1, np.zeros((1, 2), np.float32)

    def rollout(self, x: np.ndarray, steps: int, row_steps=None, aux=None) -> np.ndarray:
        """x (rows, 20, 120, 240) z-scores, aux (rows, 2) lat/lon -> z at each row's place, (rows, steps, 20)."""
        torch, core = self.torch, self.core
        order, live = sort_by_horizon(row_steps, steps, x.shape[0])
        if order is not None:
            x, aux = x[order], aux[order]
        rows = x.shape[0]
        corners = [core.grid_weights(float(lat), float(lon)) for lat, lon in aux]
        dev = self.fc.device
        ii = torch.tensor([[c[0] for c in cs] for cs in corners], device=dev)
        jj = torch.tensor([[c[1] for c in cs] for cs in corners], device=dev)
        ww = torch.tensor([[c[2] for c in cs] for cs in corners], dtype=torch.float32, device=dev)
        rr = torch.arange(rows, device=dev)[:, None]
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
        kept = torch.zeros((rows, steps, x.shape[1]), dtype=torch.float32, device=dev)
        try:
            with torch.inference_mode():
                for s in range(steps):
                    n = live[s]
                    if n < t.shape[0]:
                        t = t[:n]  # rows whose forecast has ended drop out of the batch
                    if self.device == "cuda":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out = self.fc.model(t)
                    else:
                        out = self.fc.model(t)
                    t = out.float()  # cast and feed back, as in training and in forecast_core.rollout_point
                    vals = t[rr[:n], :, ii[:n], jj[:n]]  # (n, 4 corners, channels)
                    kept[:n, s] = (vals * ww[:n, :, None]).sum(dim=1)
            return unsort(kept.cpu().numpy(), order)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

    def memory(self) -> dict:
        if self.device != "cuda":
            return {}
        cuda, mb = self.torch.cuda, 1024 * 1024
        return {"allocated_mb": round(cuda.memory_allocated() / mb, 1),
                "reserved_mb": round(cuda.memory_reserved() / mb, 1),
                "peak_allocated_mb": round(cuda.max_memory_allocated() / mb, 1)}

    def reset_peak_memory(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()


class AfnoService:
    """Turns an API.md request into engine rows and the engine's output into the API.md response."""

    def __init__(self, backend: AfnoBackend):
        import forecast_server  # the team's server: only its response builder is used

        self.backend, self.core, self.fs = backend, backend.core, forecast_server

    def prepare(self, req: dict):
        """-> (x, steps, aux, ctx). Raises LookupError when the server holds no analysis to start from."""
        store = self.backend.fc.store
        when = store.latest()
        state = store.get(when) if when is not None else None
        if state is None:
            raise LookupError("no analysis is held; put state files in the data folder's states/")
        members = req.get("ensemble_members", 1)
        steps = (req["hours_ahead"] + STEP_HOURS - 1) // STEP_HOURS
        x = self.core.perturbed_states(state, when, range(members))
        aux = np.tile(np.array([[req["lat"], req["lon"]]], np.float32), (members, 1))
        return x, steps, aux, {"when": when, "state": state, "steps": steps, "t0": time.perf_counter()}

    def respond(self, req: dict, z: np.ndarray, ctx: dict, box: str, timing: dict) -> dict:
        fc, members = self.backend.fc, req.get("ensemble_members", 1)
        series = self.core.point_series(z, 0, fc.mean, fc.std, detail=True)
        each = series.pop("_members")
        start = self.core.start_values(ctx["state"], req["lat"], req["lon"], fc.mean, fc.std)
        answer = {"node": box, "init_time": self.fs.time_text(ctx["when"]), "members": members, "member_offset": 0,
                  "point": {"lat": req["lat"], "lon": req["lon"], **series}, "units": members * ctx["steps"],
                  "timing_ms": {"queue": timing["queue_ms"], "gpu": timing["compute_ms"],
                                "total": round((time.perf_counter() - ctx["t0"]) * 1000, 2)}}
        asked = {"hours_ahead": req["hours_ahead"], "step_hours": req.get("step_hours", 1),
                 "request_id": req.get("request_id")}
        return self.fs.to_v1(answer, asked, ctx["when"], start, each, self.backend.model_version)
