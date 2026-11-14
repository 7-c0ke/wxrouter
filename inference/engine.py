"""Inference engine: loads the model, batches requests, runs them on the GPU.

Requests wait in a queue. Each of `concurrency` workers takes whatever is queued
(up to `max_batch_rows` rows), runs it as one GPU call on a thread, and hands
each request its slice. So batches form by themselves under load and a lone
request is never delayed, unless `max_wait_ms` is set.

A GPU call that raises is retried `retries` times. One that exceeds `timeout_s`
is not retried: its thread is still busy, so the caller gets InferenceTimeout
(503 from the server) and the router fails over to the other box.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .config import EngineConfig
from .gpu import GpuMonitor, node_identity
from .metrics import METRICS_SCHEMA, MetricsRecorder, log_event
from .model import FEATURES, MODEL_VERSION, make_backend


class Overloaded(Exception):
    """Queue full or engine not ready. Maps to 503."""


class InferenceTimeout(Exception):
    """GPU call exceeded timeout_s. Maps to 503."""


class InferenceError(Exception):
    """GPU call failed after all retries. Maps to 500."""


@dataclass
class _Job:
    x: np.ndarray  # (rows, FEATURES)
    steps: int
    future: asyncio.Future
    enqueued: float
    aux: np.ndarray | None = None  # per-row extras for backends that need them (AFNO: lat, lon)


class InferenceEngine:
    def __init__(self, cfg: EngineConfig | None = None, backend=None):
        self.cfg = cfg or EngineConfig()
        self.backend = backend  # tests inject fakes here
        self.metrics = MetricsRecorder()
        self.gpu = GpuMonitor()
        self.ready = False
        self.load_error: str | None = None
        self.cold_start: dict = {}
        self.node: dict = {}
        self.cpu: dict = {}  # what pin_to_fast_cores() found; set by the server and the bench
        self.in_flight = 0  # requests inside a GPU call
        self.started = time.monotonic()
        self._queue: asyncio.Queue[_Job] | None = None
        self._workers: list[asyncio.Task] = []
        self._pool: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------ load
    def load(self) -> None:
        """Build the backend and warm it up. Blocking; safe to call from a thread."""
        t0 = time.perf_counter()
        if self.backend is None:
            self.backend = make_backend(self.cfg)
        load_ms = (time.perf_counter() - t0) * 1000
        if hasattr(self.backend, "warmup_input"):
            x, steps, aux = self.backend.warmup_input()
            args = (x, steps, None, aux)
        else:
            args = (np.zeros((1, FEATURES), np.float32), 12)
        warm = []
        for _ in range(max(1, self.cfg.warmup_runs)):
            t = time.perf_counter()
            self.backend.rollout(*args)
            warm.append(round((time.perf_counter() - t) * 1000, 3))
        self.backend.reset_peak_memory()
        self.node = {**node_identity(), "cpu": self.cpu}
        self.cold_start = {"load_ms": round(load_ms, 3), "first_inference_ms": warm[0],
                           "warmup_ms": warm, "ready_ms": round((time.perf_counter() - t0) * 1000, 3)}
        self.ready = True
        log_event("engine_ready", box=self.cfg.box, backend=self.backend.name, device=self.backend.device,
                  device_name=self.backend.device_name, precision=self.backend.precision, **self.cold_start)

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._pool = ThreadPoolExecutor(self.cfg.concurrency + 1, thread_name_prefix="gpu")
        try:
            if not self.ready:
                await asyncio.get_running_loop().run_in_executor(self._pool, self.load)
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            log_event("engine_load_failed", logging.ERROR, box=self.cfg.box, error=self.load_error)
            raise
        self.gpu.start()
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.cfg.concurrency)]

    async def stop(self) -> None:
        self.ready = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self.gpu.stop()
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ inference
    def infer_sync(self, x: np.ndarray, steps: int, aux: np.ndarray | None = None) -> np.ndarray:
        """One direct GPU call, no queue. For offline batch jobs and the bench."""
        x = np.ascontiguousarray(x, np.float32)
        return self.backend.rollout(x, steps) if aux is None else self.backend.rollout(x, steps, None, aux)

    async def infer(self, x: np.ndarray, steps: int, aux: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Queue `x` (rows, FEATURES) for a `steps`-hour rollout.

        Returns (y, timing): y is (rows, steps, OUTPUTS); timing has queue_ms and compute_ms.
        """
        if not self.ready or self._queue is None:
            self.metrics.error("rejected")
            raise Overloaded("engine not ready")
        if self._queue.qsize() >= self.cfg.max_queue:
            self.metrics.error("rejected")
            raise Overloaded("queue full")
        job = _Job(np.ascontiguousarray(x, np.float32), steps,
                   asyncio.get_running_loop().create_future(), time.perf_counter(), aux)
        self._queue.put_nowait(job)
        return await job.future

    async def _worker(self) -> None:
        cfg, queue = self.cfg, self._queue
        carry: _Job | None = None
        while True:
            batch = [carry or await queue.get()]
            carry = None
            rows = len(batch[0].x)
            deadline = time.perf_counter() + cfg.max_wait_ms / 1000
            while rows < cfg.max_batch_rows:
                wait = deadline - time.perf_counter()
                try:
                    nxt = queue.get_nowait() if wait <= 0 else await asyncio.wait_for(queue.get(), wait)
                except (TimeoutError, asyncio.QueueEmpty):
                    break
                if rows + len(nxt.x) > cfg.max_batch_rows:
                    carry = nxt  # starts the next batch
                    break
                batch.append(nxt)
                rows += len(nxt.x)
            batch = [j for j in batch if not j.future.done()]  # drop requests whose client left
            if batch:
                await self._run_batch(batch)

    async def _run_batch(self, batch: list[_Job]) -> None:
        cfg, loop = self.cfg, asyncio.get_running_loop()
        steps = max(j.steps for j in batch)
        x = np.concatenate([j.x for j in batch]) if len(batch) > 1 else batch[0].x
        row_steps = [j.steps for j in batch for _ in range(len(j.x))]
        call = [self.backend.rollout, x, steps, row_steps]
        if batch[0].aux is not None:
            call.append(np.concatenate([j.aux for j in batch]))
        started = time.perf_counter()
        self.in_flight += len(batch)
        self.metrics.batch(len(batch), len(x))
        y, err, kind = None, None, "error"
        try:
            for attempt in range(cfg.retries + 1):
                try:
                    y = await asyncio.wait_for(
                        loop.run_in_executor(self._pool, *call), cfg.timeout_s)
                    break
                except TimeoutError:
                    err, kind = InferenceTimeout(f"GPU call exceeded {cfg.timeout_s}s"), "timeout"
                    log_event("inference_timeout", logging.ERROR, box=cfg.box, rows=len(x), steps=steps)
                    break
                except Exception as e:  # noqa: BLE001 - any backend failure is retryable
                    err = InferenceError(f"{type(e).__name__}: {e}")
                    log_event("inference_failed", logging.WARNING, box=cfg.box, rows=len(x), steps=steps,
                              attempt=attempt + 1, error=str(err))
                    if attempt < cfg.retries:
                        self.metrics.retries += 1
                        await asyncio.sleep(cfg.retry_backoff_ms / 1000 * (attempt + 1))
        finally:
            self.in_flight -= len(batch)
        done = time.perf_counter()
        compute_ms = (done - started) * 1000
        offset = 0
        for j in batch:
            n = len(j.x)
            if y is None:
                self.metrics.error(kind)
                if not j.future.done():
                    j.future.set_exception(err)
            else:
                queue_ms = (started - j.enqueued) * 1000
                self.metrics.ok(queue_ms, compute_ms, (done - j.enqueued) * 1000, n)
                if not j.future.done():
                    j.future.set_result((y[offset:offset + n, :j.steps],
                                         {"queue_ms": queue_ms, "compute_ms": compute_ms,
                                          "batch_requests": len(batch), "batch_rows": len(x)}))
            offset += n
        log_event("batch", logging.DEBUG, box=cfg.box, requests=len(batch), rows=len(x), steps=steps,
                  compute_ms=round(compute_ms, 2), ok=y is not None)

    # ------------------------------------------------------------ metrics
    @property
    def queue_depth(self) -> int:
        return self._queue.qsize() if self._queue else 0

    def gpu_util(self) -> float:
        """NVML utilisation when there is a GPU, else the busy share of GPU slots."""
        util = self.gpu.last.get("gpu_util")
        if util is None:
            calls = min(self.cfg.concurrency, self.in_flight)
            util = calls / self.cfg.concurrency
        return round(float(util), 3)

    def gpu_util_source(self) -> str:
        """Where gpu_util() came from this time: a dashboard must not show "busy_slots" as a GPU reading."""
        return "nvidia-smi" if self.gpu.last.get("gpu_util") is not None else "busy_slots"

    def status(self) -> str:
        if not self.ready:
            return "down"
        return "degraded" if self.queue_depth >= self.cfg.max_queue // 2 else "ok"

    def snapshot(self) -> dict:
        """Everything the router might want, as one JSON-able dict."""
        b = self.backend
        return {
            "schema": METRICS_SCHEMA,
            "box": self.cfg.box,
            "model_version": getattr(b, "model_version", MODEL_VERSION),
            "availability": {"status": self.status(), "ready": self.ready, "load_error": self.load_error,
                             "uptime_s": round(time.monotonic() - self.started, 1)},
            "node": self.node,
            "device": {"backend": getattr(b, "name", None), "type": getattr(b, "device", None),
                       "name": getattr(b, "device_name", None), "precision": getattr(b, "precision", None)},
            "load": {"in_flight": self.in_flight, "queue_depth": self.queue_depth,
                     "max_concurrency": self.cfg.concurrency, "max_queue": self.cfg.max_queue,
                     "gpu_util": self.gpu_util(), "gpu_util_source": self.gpu_util_source()},
            **self.metrics.snapshot(),
            "gpu_memory": {**(b.memory() if b and self.ready else {}),
                           "device_used_mb": self.gpu.last.get("memory_used_mb"),
                           "device_total_mb": self.gpu.last.get("memory_total_mb")},
            "cold_start": self.cold_start,
            "config": asdict(self.cfg),
        }
