"""A real, controllable load on the same device as the model.

`Burner.set(factor)` starts `round(factor) - 1` threads that run matrix products
on the model's device until told to stop, the way a training job sharing the
GPU would. Nothing is simulated: forecasts get slower because the GPU is busy,
and the slowdown you get is whatever the hardware gives, not exactly `factor`.
"""
from __future__ import annotations

import threading

import numpy as np


class Burner:
    def __init__(self, backend, size: int = 1024):
        self.backend, self.size = backend, size
        self.factor = 1.0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def set(self, factor: float) -> None:
        self.clear()
        self.factor = max(1.0, float(factor))
        self._stop = threading.Event()
        for i in range(max(0, round(self.factor) - 1)):
            t = threading.Thread(target=self._run, args=(self._stop,), name=f"burner-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def clear(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        self.factor = 1.0

    @property
    def active(self) -> bool:
        return bool(self._threads)

    def _run(self, stop: threading.Event) -> None:
        n = self.size
        if getattr(self.backend, "name", "") == "torch":
            torch, device = self.backend.torch, self.backend.device
            a = torch.randn(n, n, device=device)
            while not stop.is_set():
                a = torch.tanh(a @ a / n)
                if device == "cuda":
                    torch.cuda.synchronize()
        else:
            a = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
            while not stop.is_set():
                a = np.tanh(a @ a / n)
