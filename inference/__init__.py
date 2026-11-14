"""GPU inference subsystem for wxrouter.

  model.py     tiny autoregressive forecast net; weights + numpy/torch backends
  engine.py    batching, concurrency, warm-up, timeouts, retries
  metrics.py   p50/p95/p99, throughput, error rate, JSON snapshot for the router
  gpu.py       GPU utilisation and memory (nvidia-smi / torch)
  forecast.py  contract.ForecastRequest <-> model rows
  server.py    real forecast server implementing API.md (drop-in for mock_server.py)
  bench.py     cold/warm, batch and concurrency benchmark -> JSON

Run from the repo root: `python -m inference.server`, `python -m inference.bench`.
"""
from .config import EngineConfig
from .engine import InferenceEngine, InferenceError, InferenceTimeout, Overloaded

__all__ = ["EngineConfig", "InferenceEngine", "InferenceError", "InferenceTimeout", "Overloaded"]
