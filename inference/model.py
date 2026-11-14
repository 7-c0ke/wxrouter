"""Tiny autoregressive forecast model with a torch (GPU) and a numpy (CPU) backend.

The weights are seeded random numbers, NOT a trained weather model. The point is
a real GPU workload whose cost grows with forecast length and ensemble size, the
same way the mock's cost model does. Swap in the weather team's model by writing
another backend with the same `rollout()` signature.

One forecast hour is `layers` residual blocks on a `hidden`-wide state. A rollout
of `steps` hours returns one 4-vector in [-1, 1] per hour; forecast.py turns that
into temperature, wind, precipitation and pressure.
"""
from __future__ import annotations

import numpy as np

FEATURES = 8
OUTPUTS = 4
MODEL_VERSION = "wx-tiny-untrained-0.1"


def init_weights(hidden: int, layers: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def mat(i: int, o: int, gain: float = 1.0) -> np.ndarray:
        return (rng.standard_normal((i, o)) * gain / np.sqrt(i)).astype(np.float32)

    w = {"enc_w": mat(FEATURES, hidden), "enc_b": np.zeros(hidden, np.float32)}
    for i in range(layers):
        w[f"w1_{i}"] = mat(hidden, hidden)
        w[f"b1_{i}"] = np.zeros(hidden, np.float32)
        w[f"w2_{i}"] = mat(hidden, hidden, 0.5)
        w[f"b2_{i}"] = np.zeros(hidden, np.float32)
    w["dec_w"] = mat(hidden, OUTPUTS)
    w["dec_b"] = np.zeros(OUTPUTS, np.float32)
    return w


class NumpyBackend:
    """CPU fallback that needs only numpy. Same weights and maths as TorchBackend."""

    name = "numpy"
    device = "cpu"
    device_name = "cpu (numpy)"
    precision = "fp32"

    def __init__(self, hidden: int, layers: int, seed: int):
        self.w = init_weights(hidden, layers, seed)
        self.layers = layers

    def rollout(self, x: np.ndarray, steps: int) -> np.ndarray:
        w = self.w
        h = np.tanh(x @ w["enc_w"] + w["enc_b"])
        out = np.empty((x.shape[0], steps, OUTPUTS), np.float32)
        for t in range(steps):
            for i in range(self.layers):
                h = h + np.tanh(h @ w[f"w1_{i}"] + w[f"b1_{i}"]) @ w[f"w2_{i}"] + w[f"b2_{i}"]
            h = np.tanh(h)
            out[:, t] = np.tanh(h @ w["dec_w"] + w["dec_b"])
        return out

    def memory(self) -> dict:
        return {}

    def reset_peak_memory(self) -> None:
        pass


class TorchBackend:
    name = "torch"

    def __init__(self, hidden: int, layers: int, seed: int, device: str, amp: bool):
        import torch

        self.torch = torch
        self.device = device
        self.layers = layers
        self.amp = amp and device == "cuda"
        self.precision = "fp16-autocast" if self.amp else "fp32"
        if device == "cuda":
            self.device_name = torch.cuda.get_device_name(0)
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.device_name = device
        self.w = {k: torch.from_numpy(v).to(device) for k, v in init_weights(hidden, layers, seed).items()}

    def rollout(self, x: np.ndarray, steps: int) -> np.ndarray:
        torch, w = self.torch, self.w
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                h = torch.tanh(torch.from_numpy(x).to(self.device) @ w["enc_w"] + w["enc_b"])
                out = torch.empty((x.shape[0], steps, OUTPUTS), dtype=h.dtype, device=self.device)
                for t in range(steps):
                    for i in range(self.layers):
                        h = torch.addmm(w[f"b2_{i}"], torch.tanh(torch.addmm(w[f"b1_{i}"], h, w[f"w1_{i}"])),
                                        w[f"w2_{i}"]) + h
                    h = torch.tanh(h)
                    out[:, t] = torch.tanh(torch.addmm(w["dec_b"], h, w["dec_w"]))
            return out.float().cpu().numpy()  # .cpu() waits for the GPU, so timings are honest
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()  # give the retry a chance
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


def make_backend(cfg):
    """Pick the backend and device. Falls back to CPU unless cfg.require_gpu."""
    torch = None
    if cfg.backend in ("auto", "torch"):
        try:
            import torch
        except ImportError:
            if cfg.backend == "torch":
                raise
    cuda = bool(torch and torch.cuda.is_available())
    if (cfg.require_gpu or cfg.device == "cuda") and not cuda:
        raise RuntimeError("CUDA GPU required but not available "
                           f"(torch={'missing' if torch is None else torch.__version__})")
    if torch is None:
        return NumpyBackend(cfg.hidden, cfg.layers, cfg.seed)
    device = cfg.device if cfg.device != "auto" else ("cuda" if cuda else "cpu")
    return TorchBackend(cfg.hidden, cfg.layers, cfg.seed, device, cfg.amp)
