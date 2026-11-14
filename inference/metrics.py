"""Latency, throughput and error bookkeeping, plus JSON-lines logging."""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque

METRICS_SCHEMA = "wxrouter.inference.metrics/v1"


def percentile(values, q: float) -> float:
    """Linear-interpolated percentile, q in 0..1. Empty input gives 0."""
    if not values:
        return 0.0
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def latency_summary(values) -> dict:
    vals = list(values)
    return {"count": len(vals),
            "mean": round(sum(vals) / len(vals), 3) if vals else 0.0,
            "p50": round(percentile(vals, 0.50), 3),
            "p95": round(percentile(vals, 0.95), 3),
            "p99": round(percentile(vals, 0.99), 3),
            "max": round(max(vals), 3) if vals else 0.0}


class MetricsRecorder:
    """Counters for one engine. Only touched from the event loop, so no locks."""

    def __init__(self, window: int = 2000, rate_window_s: float = 10.0):
        self.started = time.monotonic()
        self.rate_window_s = rate_window_s
        self.total_ms: deque[float] = deque(maxlen=window)
        self.queue_ms: deque[float] = deque(maxlen=window)
        self.compute_ms: deque[float] = deque(maxlen=window)
        self.done: deque[tuple[float, int]] = deque(maxlen=20000)  # (finish time, rows)
        self.ewma_ms = 0.0
        self.requests_total = 0  # finished, ok or not, plus rejected
        self.errors_total = 0
        self.timeouts = 0
        self.rejected = 0
        self.retries = 0
        self.batches = 0
        self.batch_rows = 0
        self.batch_requests = 0

    def ok(self, queue_ms: float, compute_ms: float, total_ms: float, rows: int) -> None:
        self.requests_total += 1
        self.total_ms.append(total_ms)
        self.queue_ms.append(queue_ms)
        self.compute_ms.append(compute_ms)
        self.done.append((time.monotonic(), rows))
        self.ewma_ms = total_ms if self.ewma_ms == 0 else 0.8 * self.ewma_ms + 0.2 * total_ms

    def error(self, kind: str) -> None:
        self.requests_total += 1
        self.errors_total += 1
        if kind == "timeout":
            self.timeouts += 1
        elif kind == "rejected":
            self.rejected += 1

    def batch(self, requests: int, rows: int) -> None:
        self.batches += 1
        self.batch_requests += requests
        self.batch_rows += rows

    def throughput(self) -> dict:
        now = time.monotonic()
        span = min(self.rate_window_s, max(now - self.started, 1e-6))
        recent = [rows for t, rows in self.done if now - t <= span]
        return {"window_s": round(span, 2),
                "requests_per_s": round(len(recent) / span, 2),
                "rows_per_s": round(sum(recent) / span, 2)}

    def snapshot(self) -> dict:
        n = self.requests_total
        return {
            "latency_ms": {"ewma": round(self.ewma_ms, 3), **latency_summary(self.total_ms)},
            "queue_ms": latency_summary(self.queue_ms),
            "compute_ms": latency_summary(self.compute_ms),
            "throughput": self.throughput(),
            "errors": {"requests_total": n, "errors_total": self.errors_total,
                       "error_rate": round(self.errors_total / n, 5) if n else 0.0,
                       "timeouts": self.timeouts, "rejected": self.rejected, "retries": self.retries},
            "batches": {"count": self.batches,
                        "mean_requests": round(self.batch_requests / self.batches, 2) if self.batches else 0.0,
                        "mean_rows": round(self.batch_rows / self.batches, 2) if self.batches else 0.0},
        }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": round(record.created, 3), "level": record.levelname, "event": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO") -> None:
    """One JSON object per line on stderr, so stdout stays clean for results."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("wxrouter.inference")
    log.handlers[:] = [handler]
    log.setLevel(level.upper())
    log.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    logging.getLogger("wxrouter.inference").log(level, event, extra={"fields": fields})
