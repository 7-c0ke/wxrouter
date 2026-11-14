"""GPU utilisation and memory from nvidia-smi, sampled on a background thread.

/stats must stay cheap, so readers only ever see the last sample. On machines
without nvidia-smi (laptops) every field is None and callers fall back to
in_flight / max_concurrency.
"""
from __future__ import annotations

import os
import shutil
import socket
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
            self.last = query_smi() or {}   # no reading means no value: an old sample is never kept as current

    def stop(self) -> None:
        self._stop.set()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def node_identity() -> dict:
    """Which machine and which GPU this process is on, to prove two boxes are on different nodes."""
    smi = shutil.which("nvidia-smi")
    uuid = _run([smi, "--query-gpu=uuid", "--format=csv,noheader"]).splitlines() if smi else []
    return {"hostname": socket.gethostname(),
            "addresses": _run(["hostname", "-I"]).split() if shutil.which("hostname") else [],
            "gpu_uuid": uuid[0].strip() if uuid else None}


def pin_to_fast_cores(sysfs: str = "/sys/devices/system/cpu") -> dict:
    """Keep this process on the machine's fastest cores, when its cores are not all alike.

    A GPU call here is hundreds of tiny operations, each launched by one CPU thread, so the
    speed of that thread sets the speed of the call. A DGX Spark (GB10) has ten Cortex-X925
    cores and ten slower Cortex-A725 cores; measured on one, the same call took 9.9 ms on an
    X925 and 56-67 ms on an A725. A scheduler that hands a job cores 0-3 gives it only slow ones.

    Call it first thing in the main thread: threads started later inherit the setting.
    Returns what it found; `pinned` is None when there was nothing to choose or no way to tell.
    """
    if not hasattr(os, "sched_getaffinity"):
        return {"allowed": None, "pinned": None, "note": "no CPU affinity on this OS"}
    allowed = sorted(os.sched_getaffinity(0))
    freq = {}
    for c in allowed:
        try:
            with open(f"{sysfs}/cpu{c}/cpufreq/cpuinfo_max_freq") as f:
                freq[c] = int(f.read().strip())
        except (OSError, ValueError):
            return {"allowed": allowed, "pinned": None, "note": "core speeds not readable"}
    top = max(freq.values())
    fast = [c for c in allowed if freq[c] == top]
    info = {"allowed": allowed, "max_freq_khz": {str(c): freq[c] for c in allowed}}
    if len(fast) == len(allowed):
        return {**info, "pinned": None, "note": "all allowed cores are equally fast"}
    os.sched_setaffinity(0, fast)
    return {**info, "pinned": fast, "note": f"pinned to the {len(fast)} fastest of {len(allowed)} allowed cores"}


def slow_cores_only(info: dict, sysfs: str = "/sys/devices/system/cpu") -> bool:
    """True when the machine has faster cores than any this process may use."""
    mine = [int(v) for v in (info.get("max_freq_khz") or {}).values()]
    if not mine:
        return False
    best = 0
    try:
        for name in os.listdir(sysfs):
            if name.startswith("cpu") and name[3:].isdigit():
                with open(f"{sysfs}/{name}/cpufreq/cpuinfo_max_freq") as f:
                    best = max(best, int(f.read().strip()))
    except (OSError, ValueError):
        return False
    return best > max(mine)
