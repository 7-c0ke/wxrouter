"""GPU utilisation and memory from nvidia-smi, sampled on a background thread.

/stats must stay cheap, so readers only ever see the last sample. On machines
without nvidia-smi (laptops) every field is None and callers fall back to
in_flight / max_concurrency.
"""
from __future__ import annotations

import shutil
import subprocess
import threading

QUERY = "utilization.gpu,memory.used,memory.total,name"


def _num(s: str):
    try:
        return float(s)
    except ValueError:  # "[N/A]" on unified-memory boxes such as DGX Spark
        return None


def parse_smi(line: str) -> dict:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return {}
    util = _num(parts[0])
    return {"gpu_util": None if util is None else round(util / 100, 3),
            "memory_used_mb": _num(parts[1]), "memory_total_mb": _num(parts[2]),
            "name": ",".join(parts[3:]).strip()}


def query_smi(timeout: float = 2.0) -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        out = subprocess.run([exe, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=timeout, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    lines = out.strip().splitlines()
    return parse_smi(lines[0]) if lines else {}


class GpuMonitor:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.last: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread or not shutil.which("nvidia-smi"):
            return
        self.last = query_smi()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.last = query_smi() or self.last

    def stop(self) -> None:
        self._stop.set()
