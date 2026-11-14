"""Unit tests for the inference engine, model and metrics. No GPU or network needed."""
import asyncio
import json
import logging
import time

import numpy as np
import pytest

from inference.config import EngineConfig
from inference.engine import (
    InferenceEngine,
    InferenceError,
    InferenceTimeout,
    Overloaded,
)
from inference.gpu import parse_smi
from inference.metrics import (
    JsonFormatter,
    MetricsRecorder,
    latency_summary,
    percentile,
)
from inference.model import FEATURES, OUTPUTS, NumpyBackend, make_backend

SMALL = {"backend": "numpy", "hidden": 32, "layers": 2, "warmup_runs": 1}


def rows(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, FEATURES)).astype(np.float32)


class FakeBackend(NumpyBackend):
    """Numpy model that can be told to fail or stall, and records each call's batch size."""

    def __init__(self, fail=0, delay=0.0):
        super().__init__(16, 1, 0)
        self.fail, self.delay, self.calls = fail, delay, []

    def rollout(self, x, steps, row_steps=None):
        self.calls.append(len(x))
        if self.delay:
            time.sleep(self.delay)
        if self.fail > 0:
            self.fail -= 1
            raise RuntimeError("CUDA error: injected")
        return super().rollout(x, steps, row_steps)


def engine_with(backend, **cfg):
    return InferenceEngine(EngineConfig(**{**SMALL, **cfg}), backend=backend)


async def started(engine):
    await engine.start()
    return engine


# ------------------------------------------------------------------ model
def test_rollout_shape_range_and_determinism():
    a, b = NumpyBackend(32, 2, seed=7), NumpyBackend(32, 2, seed=7)
    y = a.rollout(rows(5), 24)
    assert y.shape == (5, 24, OUTPUTS) and y.dtype == np.float32
    assert np.all(np.abs(y) <= 1) and np.isfinite(y).all()
    assert np.array_equal(y, b.rollout(rows(5), 24))
    assert not np.array_equal(y, NumpyBackend(32, 2, seed=8).rollout(rows(5), 24))


def test_rows_are_independent_of_batch_and_horizon():
    """Batching must not change answers: a row's first k hours are the same alone or in a batch."""
    m = NumpyBackend(32, 2, 0)
    x = rows(6)
    full = m.rollout(x, 48)
    alone = m.rollout(x[2:3], 12)
    assert np.allclose(full[2:3, :12], alone, atol=1e-5)


def test_mixed_horizons_in_one_batch_match_separate_runs():
    """Rows that end early leave the batch; every row still gets exactly its own forecast."""
    m = NumpyBackend(32, 2, 0)
    x, hours = rows(5), [6, 48, 12, 48, 1]
    y = m.rollout(x, 48, hours)
    for i, h in enumerate(hours):
        assert np.allclose(y[i, :h], m.rollout(x[i:i + 1], h)[0], atol=1e-5)
    torch = pytest.importorskip("torch")
    from inference.model import TorchBackend

    yt = TorchBackend(32, 2, 0, "cpu", amp=False).rollout(x, 48, hours)
    for i, h in enumerate(hours):
        assert np.allclose(yt[i, :h], y[i, :h], atol=1e-4)
    assert torch.__version__


def test_torch_backend_matches_numpy():
    pytest.importorskip("torch")
    from inference.model import TorchBackend

    x = rows(4)
    ref = NumpyBackend(32, 2, 3).rollout(x, 24)
    got = TorchBackend(32, 2, 3, "cpu", amp=False).rollout(x, 24)
    assert np.allclose(ref, got, atol=1e-4)


def test_cpu_fallback_and_require_gpu():
    assert make_backend(EngineConfig(**SMALL)).device == "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        with pytest.raises(RuntimeError, match="GPU required"):
            make_backend(EngineConfig(require_gpu=True))


# ------------------------------------------------------------------ metrics
def test_percentiles():
    vals = list(range(1, 101))
    assert percentile(vals, 0.5) == pytest.approx(50.5)
    assert percentile(vals, 0.99) == pytest.approx(99.01)
    assert percentile([], 0.95) == 0.0
    s = latency_summary(vals)
    assert s["count"] == 100 and s["p50"] <= s["p95"] <= s["p99"] <= s["max"] == 100


def test_recorder_error_rate_and_throughput():
    m = MetricsRecorder()
    for _ in range(8):
        m.ok(1.0, 4.0, 5.0, rows=2)
    m.error("timeout")
    m.error("rejected")
    snap = m.snapshot()
    assert snap["errors"] == {"requests_total": 10, "errors_total": 2, "error_rate": 0.2,
                              "timeouts": 1, "rejected": 1, "retries": 0}
    assert snap["latency_ms"]["p99"] == 5.0
    assert snap["throughput"]["requests_per_s"] > 0
    assert snap["throughput"]["rows_per_s"] == pytest.approx(2 * snap["throughput"]["requests_per_s"])


def test_json_log_lines():
    rec = logging.LogRecord("wxrouter.inference", logging.INFO, "", 0, "engine_ready", None, None)
    rec.fields = {"box": "box-a", "load_ms": 1.5}
    out = json.loads(JsonFormatter().format(rec))
    assert out["event"] == "engine_ready" and out["box"] == "box-a" and out["level"] == "INFO"


def test_parse_nvidia_smi():
    assert parse_smi("37, 1024, 16384, NVIDIA L40") == {
        "gpu_util": 0.37, "memory_used_mb": 1024.0, "memory_total_mb": 16384.0, "name": "NVIDIA L40"}
    na = parse_smi("5, [N/A], [N/A], NVIDIA GB10")  # unified-memory boxes
    assert na["gpu_util"] == 0.05 and na["memory_used_mb"] is None
    assert parse_smi("garbage") == {}


# ------------------------------------------------------------------ engine
def test_load_warms_up_and_reports_cold_start():
    e = InferenceEngine(EngineConfig(**{**SMALL, "warmup_runs": 3}))
    assert not e.ready and e.status() == "down"
    e.load()
    assert e.ready and e.status() == "ok"
    assert len(e.cold_start["warmup_ms"]) == 3 and e.cold_start["first_inference_ms"] > 0


def test_infer_matches_direct_call_and_slices_horizons():
    async def go():
        e = await started(InferenceEngine(EngineConfig(**SMALL)))
        xa, xb = rows(3, 1), rows(2, 2)
        (ya, ta), (yb, _) = await asyncio.gather(e.infer(xa, 48), e.infer(xb, 6))
        assert ya.shape == (3, 48, OUTPUTS) and yb.shape == (2, 6, OUTPUTS)
        assert np.allclose(ya, e.infer_sync(xa, 48), atol=1e-5)
        assert np.allclose(yb, e.infer_sync(xb, 6), atol=1e-5)
        assert ta["compute_ms"] > 0 and ta["queue_ms"] >= 0
        await e.stop()

    asyncio.run(go())


def test_concurrent_requests_are_batched():
    async def go():
        fb = FakeBackend(delay=0.02)
        e = await started(engine_with(fb, max_batch_rows=8))
        fb.calls.clear()
        out = await asyncio.gather(*(e.infer(rows(1, i), 4) for i in range(17)))
        assert len(out) == 17
        assert sum(fb.calls) == 17 and max(fb.calls) == 8 and len(fb.calls) <= 4
        assert e.snapshot()["batches"]["mean_rows"] > 1
        await e.stop()

    asyncio.run(go())


def test_batching_off_with_max_batch_rows_1():
    async def go():
        fb = FakeBackend()
        e = await started(engine_with(fb, max_batch_rows=1))
        fb.calls.clear()
        await asyncio.gather(*(e.infer(rows(1, i), 4) for i in range(6)))
        assert fb.calls == [1] * 6
        await e.stop()

    asyncio.run(go())


def test_oversized_request_still_runs():
    async def go():
        e = await started(engine_with(FakeBackend(), max_batch_rows=4))
        y, _ = await e.infer(rows(20), 3)  # a 20-member ensemble is never split
        assert y.shape == (20, 3, OUTPUTS)
        await e.stop()

    asyncio.run(go())


def test_retry_recovers_from_one_failure():
    async def go():
        fb = FakeBackend()
        e = await started(engine_with(fb, retries=1, retry_backoff_ms=1))
        fb.fail = 1
        y, _ = await e.infer(rows(1), 4)
        assert y.shape == (1, 4, OUTPUTS)
        errs = e.snapshot()["errors"]
        assert errs["retries"] == 1 and errs["errors_total"] == 0
        await e.stop()

    asyncio.run(go())


def test_failure_after_retries_raises_and_counts():
    async def go():
        fb = FakeBackend()
        e = await started(engine_with(fb, retries=1, retry_backoff_ms=1))
        fb.fail = 5
        with pytest.raises(InferenceError, match="injected"):
            await e.infer(rows(1), 4)
        errs = e.snapshot()["errors"]
        assert errs["errors_total"] == 1 and errs["error_rate"] == 1.0
        fb.fail = 0
        await e.infer(rows(1), 4)  # engine keeps serving afterwards
        assert e.snapshot()["errors"]["error_rate"] == 0.5
        await e.stop()

    asyncio.run(go())


def test_timeout_raises_and_is_not_retried():
    async def go():
        fb = FakeBackend()
        e = await started(engine_with(fb, timeout_s=0.05, retries=3))
        fb.calls.clear()
        fb.delay = 0.3
        with pytest.raises(InferenceTimeout):
            await e.infer(rows(1), 4)
        assert len(fb.calls) == 1
        errs = e.snapshot()["errors"]
        assert errs["timeouts"] == 1 and errs["retries"] == 0
        await e.stop()

    asyncio.run(go())


def test_queue_full_and_not_ready_are_rejected():
    async def go():
        fb = FakeBackend()
        e = engine_with(fb, max_queue=2, max_batch_rows=1)
        with pytest.raises(Overloaded, match="not ready"):
            await e.infer(rows(1), 4)
        await e.start()
        fb.delay = 0.05
        results = await asyncio.gather(*(e.infer(rows(1, i), 4) for i in range(8)), return_exceptions=True)
        rejected = [r for r in results if isinstance(r, Overloaded)]
        assert rejected and len(rejected) < 8
        assert e.snapshot()["errors"]["rejected"] == len(rejected) + 1
        await e.stop()

    asyncio.run(go())


def test_snapshot_is_json_with_router_fields():
    async def go():
        e = await started(InferenceEngine(EngineConfig(**SMALL)))
        await e.infer(rows(2), 12)
        snap = json.loads(json.dumps(e.snapshot()))
        assert snap["schema"] == "wxrouter.inference.metrics/v1"
        assert snap["availability"] == {**snap["availability"], "status": "ok", "ready": True}
        assert {"in_flight", "queue_depth", "max_concurrency", "gpu_util"} <= snap["load"].keys()
        assert {"p50", "p95", "p99", "ewma"} <= snap["latency_ms"].keys()
        assert {"requests_per_s", "rows_per_s"} <= snap["throughput"].keys()
        assert snap["device"]["type"] == "cpu" and "gpu_memory" in snap
        await e.stop()

    asyncio.run(go())
