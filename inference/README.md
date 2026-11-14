# wxrouter inference subsystem

The real forecast server behind the router: it implements `API.md` on a GPU (PyTorch/CUDA),
with a CPU fallback for laptops. It replaces `mock_server.py` one for one. `router.py`,
`loadgen.py`, `contract.py` and `mock_server.py` are unchanged.

| file | what it is |
|---|---|
| `model.py` | Tiny autoregressive forecast net, torch backend (GPU) and numpy backend (CPU, same weights) |
| `engine.py` | Batching, concurrency, warm-up, timeout, retries |
| `metrics.py` | p50/p95/p99, throughput, error rate; JSON-lines logging |
| `gpu.py` | GPU utilisation and memory from `nvidia-smi` |
| `forecast.py` | `ForecastRequest` → model rows → `ForecastResponse` |
| `server.py` | FastAPI server: `/v1/forecast`, `/health`, `/stats`, plus `/v1/metrics` |
| `bench.py` | Benchmark runner → `bench.json` + `table.md` |
| `../deploy/build_inference_fuzzfile.py` | Generates the Fuzzball workflows |
| `../deploy/collect_bench.py` | Pulls benchmark JSON out of a Fuzzball job log |

**The model is not a trained weather model.** Its weights are seeded random numbers
(`model_version: wx-tiny-untrained-0.1`). It gives the router a real GPU workload whose cost grows
with `hours_ahead` and `ensemble_members`. The forecasts it returns have the right shape and
plausible ranges and mean nothing. To serve the weather team's model, add a backend class with
the same `rollout(x, steps)` method in `model.py`; nothing else changes.

## Install

Python 3.11 or newer.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r inference/requirements.txt
```

`torch` is optional on a laptop: without it the server uses the numpy backend.
Install `torch` separately for GPU inference. Test tools (`pytest`, `ruff`, `pyyaml`)
are optional development dependencies; reuse them when already installed.

## Run locally

```bash
.venv/bin/python -m inference.server --name box-a --port 8001
.venv/bin/python -m inference.server --name box-b --port 8002
.venv/bin/python router.py --backend box-a=http://127.0.0.1:8001 --backend box-b=http://127.0.0.1:8002 --policy adaptive
```

Device choice: `--device auto` uses CUDA when present, else CPU. `--require-gpu` exits instead
of falling back. `--backend numpy` forces the no-torch path. `--device mps` is opt-in.

Main knobs (all flags on both the server and the bench; see `config.py`):

| flag | default | meaning |
|---|---:|---|
| `--max-batch-rows` | 64 | rows (request × ensemble member) per GPU call; `1` turns batching off |
| `--max-wait-ms` | 0 | extra wait to fill a batch; `0` batches only what is already queued |
| `--concurrency` | 1 | GPU calls running at once (`max_concurrency` in `/stats`) |
| `--max-queue` | 256 | waiting requests before `503` |
| `--timeout-s` | 10 | per GPU call; on timeout the request gets `503` so the router fails over |
| `--retries` | 1 | retries after a failed GPU call (timeouts are not retried) |
| `--warmup-runs` | 3 | inferences run before `/health` reports `ok` |

Logs are one JSON object per line on stderr (`engine_ready`, `inference_failed`,
`inference_timeout`, and `batch` at `--log-level DEBUG`).

## For the routing team

Nothing to change: point `--backend` at an inference server instead of a mock.

- `/health` returns `503 {"status":"down"}` until the model is loaded and warmed up, `degraded`
  when the queue is at least half full, else `ok`.
- `/stats` is the contract's `Stats`. `gpu_util` comes from `nvidia-smi` (sampled once a second
  in the background) and falls back to busy GPU slots ÷ `max_concurrency` when there is no GPU.
  `mode` is always `"normal"`; there is no `/admin/mode`.
- `timing.compute_ms` is the wall time of the GPU call that carried the request. With batching,
  requests in the same batch report the same `compute_ms`.
- `GET /v1/metrics` (new, outside the contract) returns everything in one JSON document.
  `--metrics-file PATH` writes the same document to disk every 2 s.

```json
{
  "schema": "wxrouter.inference.metrics/v1",
  "box": "box-a",
  "model_version": "wx-tiny-untrained-0.1",
  "availability": {"status": "ok", "ready": true, "load_error": null, "uptime_s": 31.3},
  "device": {"backend": "torch", "type": "cuda", "name": "...", "precision": "fp16-autocast"},
  "load": {"in_flight": 0, "queue_depth": 0, "max_concurrency": 1, "max_queue": 256, "gpu_util": 0.0},
  "latency_ms": {"ewma": 1.5, "count": 296, "mean": 2.0, "p50": 1.3, "p95": 5.4, "p99": 8.1, "max": 15.4},
  "queue_ms": {"...": "same fields"},
  "compute_ms": {"...": "same fields"},
  "throughput": {"window_s": 10.0, "requests_per_s": 13.8, "rows_per_s": 15.1},
  "errors": {"requests_total": 296, "errors_total": 0, "error_rate": 0.0, "timeouts": 0, "rejected": 0, "retries": 0},
  "batches": {"count": 290, "mean_requests": 1.02, "mean_rows": 1.1},
  "gpu_memory": {"allocated_mb": null, "reserved_mb": null, "peak_allocated_mb": null, "device_used_mb": null, "device_total_mb": null},
  "cold_start": {"load_ms": 23.5, "first_inference_ms": 0.6, "warmup_ms": [0.6, 0.7, 0.7], "ready_ms": 25.5}
}
```

Latency percentiles cover the last 2000 requests. Throughput covers the last 10 s. The
`gpu_memory` torch fields appear only on CUDA.

From Python, without HTTP:

```python
from inference import EngineConfig, InferenceEngine
engine = InferenceEngine(EngineConfig(box="box-a"))
await engine.start()                    # loads and warms up
y, timing = await engine.infer(x, 48)   # x: (rows, 8) float32 from inference.forecast.build_rows
engine.snapshot()                       # the metrics document above
```

## Test

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/ruff check inference tests deploy
```

`tests/test_inference_server.py::test_router_in_front_of_two_inference_servers` is the
integration smoke test: two inference servers behind the unmodified `router.py`, then one box
is killed and the router must keep answering.

## Benchmark

```bash
.venv/bin/python -m inference.bench            # results/inference/<timestamp>/bench.json, table.md
.venv/bin/python -m inference.bench --quick    # a few seconds
```

It measures cold start (torch import, model load, first inference), one GPU call per batch
size, and serving under N concurrent clients with batching off (**baseline**,
`--max-batch-rows 1`) and on. On CUDA it also compares fp32 with fp16 autocast.

### Measured: laptop CPU baseline (MacBook, arm64, torch CPU fp32, 48 h forecasts)

From `results/inference/20261114-110854/`. This is the CPU fallback path, not a GPU number.

| | value |
|---|---:|
| torch import | 791 ms |
| model load | 24 ms |
| first inference (cold) | 1.85 ms |
| warm inference (12 h, 1 row) | 0.68 ms |

One model call per batch:

| batch rows | p50 ms | p95 ms | p99 ms | rows/s |
|---:|---:|---:|---:|---:|
| 1 | 2.45 | 2.82 | 3.00 | 400 |
| 16 | 12.92 | 21.94 | 22.75 | 1098 |
| 64 | 28.48 | 34.22 | 35.83 | 2225 |
| 256 | 68.24 | 91.72 | 101.25 | 3506 |

Serving 400 requests, batching off (baseline) vs on:

| clients | mode | p50 ms | p95 ms | p99 ms | req/s | mean batch |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | 2.64 | 3.48 | 3.81 | 362 | 1.0 |
| 1 | batched | 2.51 | 3.02 | 3.23 | 387 | 1.0 |
| 8 | baseline | 21.07 | 25.55 | 26.40 | 361 | 1.0 |
| 8 | batched | 6.87 | 9.13 | 9.25 | 1125 | 8.0 |
| 32 | baseline | 82.32 | 96.48 | 107.43 | 378 | 1.0 |
| 32 | batched | 17.09 | 18.39 | 18.39 | 1812 | 30.8 |

- The bottleneck without batching is one request per model call: throughput stays near
  370 req/s however many clients there are, and latency grows with the queue.
- Batching what is already queued raised throughput 3.1× at 8 clients and 4.8× at 32, and
  cut p95 from 96 ms to 18 ms at 32 clients. With one client it changes nothing, since
  there is never a second request to batch with.
- No errors in any run. One run per row, so treat differences of a few percent as noise.
- Behind the unmodified router (`least_loaded`, two numpy-backend servers, the 20 s seed-42
  trace at 25 req/s): 505 requests, 0 errors, p50 7.0 ms, p95 11.8 ms, p99 17.4 ms.

## Fuzzball

```bash
.venv/bin/python deploy/build_inference_fuzzfile.py                 # deploy/inference-smoke.fz
.venv/bin/python deploy/build_inference_fuzzfile.py --mode bench    # deploy/inference-bench.fz
.venv/bin/python deploy/build_inference_fuzzfile.py --mode serve    # deploy/inference-serve.fz
```

Regenerate after any change under `inference/`: the sources are embedded in the Fuzzfile, the
same way `build_fuzzfile.py` does it, so the private repo needs no token.

| workflow | what runs | GPU request |
|---|---|---|
| `inference-smoke.fz` | job `gpu-smoke`: `nvidia-smi`, torch CUDA check, `bench --quick` | 1 × `nvidia.com/gpu`, 4 cores, 16 GB, 15 min limit |
| `inference-bench.fz` | job `gpu-bench`: full benchmark | same, 45 min limit |
| `inference-serve.fz` | services `box-a`, `box-b` (inference servers) and `router` | 1 GPU per box; `--exclusive` puts them on separate nodes |

Every GPU stage runs with `--require-gpu`, so a node without a usable GPU fails the job
(exit code 2) and never reports CPU numbers as GPU numbers.

Image: `docker://nvcr.io/nvidia/pytorch:25.09-py3` (the tag exists for amd64 and arm64; override
with `--image`). The serve boxes also `pip install fastapi uvicorn httpx pydantic` at start, so
they need PyPI access. The smoke and bench jobs need no network beyond the image pull.

```bash
fuzzball workflow validate deploy/inference-smoke.fz
fuzzball workflow start -n inference-smoke-team14 deploy/inference-smoke.fz
fuzzball workflow describe <workflow-id>
fuzzball workflow log <workflow-id> gpu-smoke | .venv/bin/python deploy/collect_bench.py --label fuzzball-smoke
```

The last command saves the job's numbers to `results/inference/<timestamp>-fuzzball-smoke/`;
commit that folder to share it. To write results on the cluster as well, pass
`--results-volume <existing persistent volume> --results-dir /results` to the generator.

### Cluster status (fb-hack, 2026-11-14)

- CLI v4.3.1, context `fb-hack`, account `team14@fb-hack.local`. Workflows run under that
  account's own group; `fuzzball group use team14` is refused ("permission denied: owner").
- Hardware: 40 DGX Spark nodes (`CIQ-CHI-SPARK01`–`40`, arm64, 20 cores, 128 GB, one
  `nvidia.com/gpu` each) and two small CPU nodes. `fuzzball node list -a` shows free GPUs;
  38 were free when checked.
- All three workflows above pass `fuzzball workflow validate`.
- Team volume `wxdesk` is mounted by name. The smoke workflow in this repo was generated with
  `--results-volume wxdesk --results-dir /wxdesk`, so its results land in
  `wxdesk:/wxrouter-inference/<timestamp>/`.

Earlier failed workflows and their causes:

| workflow | cause |
|---|---|
| `449d2ce7` wxrouter-team14, `1f8b5234` wxrouter-probe-team14 (09:38–09:39) | Cluster side. The image cache was unreachable: `lookup fuzzball-objectcache on 10.96.0.10:53: no such host`. It failed before any container started. Later runs of the same workflows succeeded, so it has cleared. |
| `76acc6d3` voltest-team14 (Nov 13) | Script bug. `find /datasets ! -readable` under `set -e`: Alpine's BusyBox `find` has no `-readable`, so it exited 1 with stderr discarded. The volume mount itself worked. |

GPU smoke test: submitted as `a2ffa685-4e61-43fa-bee8-c564a5a5c570` (`inference-smoke-team14`)
at 11:03. **No GPU numbers yet**: at the time of writing the workflow was still pulling the
PyTorch image for the first time. Collect the result with the `collect_bench.py` command above.
