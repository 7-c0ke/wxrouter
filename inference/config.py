"""Engine settings. Every field is also a CLI flag on the server and the bench."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields


@dataclass
class EngineConfig:
    box: str = "box-a"
    device: str = "auto"  # auto (cuda, else cpu) | cuda | mps | cpu
    backend: str = "auto"  # auto (torch if installed, else numpy) | torch | numpy
    hidden: int = 256  # model width
    layers: int = 4  # residual blocks per forecast hour
    seed: int = 0  # weight seed; both boxes must use the same one
    amp: bool = False  # fp16 autocast on CUDA; off because on the GB10 it was slower and drifted (see README)
    max_batch_rows: int = 64  # rows (request x ensemble member) per GPU call; 1 disables batching
    max_wait_ms: float = 0.0  # extra wait to fill a batch; 0 = only batch what is already queued
    concurrency: int = 1  # GPU calls running at once
    max_queue: int = 256  # waiting requests before 503
    timeout_s: float = 10.0  # per GPU call
    retries: int = 1  # retries after a failed GPU call (not after a timeout)
    retry_backoff_ms: float = 20.0
    warmup_runs: int = 3
    require_gpu: bool = False  # fail at load instead of falling back to CPU
    pin_fast_cores: bool = True  # on machines with fast and slow cores (DGX Spark), stay on the fast ones


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    for f in fields(EngineConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)


def engine_config_from_args(a: argparse.Namespace) -> EngineConfig:
    return EngineConfig(**{f.name: getattr(a, f.name) for f in fields(EngineConfig)})
