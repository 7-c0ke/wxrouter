# wxrouter inference subsystem

A GPU forecast server for the router, with batching, real fault hooks, benchmarks and Fuzzball
workflows. It answers `API.md` v1.0, so `router.py`, `loadgen.py` and the UI work with it unchanged.

Status on 2026-11-14 (all on DGX Spark nodes through Fuzzball; numbers in `router_switching/README.md`):

| | state |
|---|---|
| One-GPU inference and batching | **ran** (`4a0aaf44`, SPARK28) |
| Two boxes on two GPUs behind `router.py`: five policies, then slow / stop / restart | **ran** (`c5ffff9f`, SPARK27 + SPARK28); 0 failed requests in every fault case |
| Trained AFNO weather model behind the engine | **ran** (`664e809a`, SPARK28); answers match the team's server |

## What is here

| file | what it is |
|---|---|
| `engine.py` | Queue, batching across requests, concurrency, warm-up, timeout, retries |
| `model.py` | The tiny stand-in model (torch on GPU, numpy on CPU, same weights) |
| `afno.py` | Adapter that serves the weather team's trained AFNO through the engine |
| `server.py` | FastAPI server: `/v1/forecast`, `/health`, `/stats`, `/v1/metrics`, `/admin/*` |
| `contention.py` | A real second load on the box's GPU, for "slow" mode |
| `metrics.py`, `gpu.py` | p50/p95/p99, throughput, error rate, GPU use and memory, node identity, JSON logs |
| `bench.py` | One-box benchmark: cold start, batch sizes, batching on/off, fp32 vs fp16 |
| `cluster_bench.py` | Two-box benchmark through the router: every policy, plus slow / stop / restart |
| `afno_check.py` | Same AFNO weights behind the team's server and behind the engine: compare, then time |
| `report.py` | Benchmark JSON to one HTML page of charts and tables |
| `../deploy/build_inference_fuzzfile.py` | Generates the Fuzzball workflows |
| `../deploy/run_inference.sh` | One command: generate, validate, start, wait, collect, rebuild the report |
| `../deploy/collect_bench.py` | Pulls a job's result JSON out of its log |

**Two models.** `--model tiny` (default) is a small network with seeded random weights
(`wx-tiny-untrained-0.1`). It is an infrastructure workload: real GPU work whose cost grows with
`hours_ahead` and `ensemble_members`, but its forecasts mean nothing. `--model afno` serves the
weather team's trained model. No weather accuracy number in this folder comes from the tiny model.

## Install and test

Python 3.11 or newer.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt numpy torch pytest ruff pyyaml
.venv/bin/python -m pytest tests/test_inference_engine.py tests/test_inference_server.py \
    tests/test_inference_fuzzfile.py tests/test_cluster_bench.py tests/test_inference_afno.py -q
```

45 tests: engine (batching, mixed forecast lengths, timeout, retry, queue full), the API contract,
admin hooks, a supervised stop and restart, the router in front of two servers with one killed,
Fuzzfile structure, and the AFNO adapter against the weather team's own rollout. `torch` is optional
on a laptop; without it the server uses numpy. (`tests/test_gfs_live.py` and `tests/test_train.py`
belong to the weather code and need `pygrib` and PhysicsNeMo.)

## Run locally

```bash
.venv/bin/python -m inference.server --name box-a --port 8001 --admin --supervise
.venv/bin/python -m inference.server --name box-b --port 8002 --admin --supervise
.venv/bin/python router.py --backend box-a=http://127.0.0.1:8001 --backend box-b=http://127.0.0.1:8002
.venv/bin/python -m inference.cluster_bench --router http://127.0.0.1:8000 \
    --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002 --duration 10 --fault-duration 30
```

| flag | default | meaning |
|---|---:|---|
| `--device` | auto | CUDA when present, else CPU. `--require-gpu` exits instead of falling back |
| `--max-batch-rows` | 64 | rows (request × ensemble member) per GPU call; `1` turns batching off |
| `--max-wait-ms` | 0 | extra wait to fill a batch; `0` batches only what is already queued |
| `--concurrency` | 1 | GPU calls running at once |
| `--max-queue` | 256 | waiting requests before `503` |
| `--timeout-s` | 10 | per GPU call; on timeout the request gets `503` and the router fails over |
| `--retries` | 1 | retries after a failed GPU call (timeouts are not retried) |
| `--amp` | off | fp16 autocast. Off because on the GB10 it was slower and inaccurate (below) |
| `--admin` | off | enable the `/admin` hooks |
| `--supervise` | off | start the server again after `/admin/stop` |

## For the routing and UI teams

Point `--backend` at an inference server instead of a mock. Nothing else changes.

- `/health`: `503 down` until the model is loaded and warm, `degraded` when slowed or the queue is
  half full, else `ok`.
- `/stats`: the contract's `Stats`. `gpu_util` comes from `nvidia-smi`, sampled once a second;
  without a GPU it is busy GPU slots ÷ `max_concurrency`. `mode` is `training` while slow mode's
  GPU load runs, else `normal`.
- `timing.compute_ms` is the GPU call that carried the request. Requests batched together report
  the same value, so the router's adaptive probe keeps working.
- `GET /v1/metrics` (outside the contract): one JSON document, schema
  `wxrouter.inference.metrics/v1`, with `availability`, `node` (hostname, addresses, GPU UUID),
  `device`, `load`, `latency_ms` / `queue_ms` / `compute_ms` (p50, p95, p99), `throughput`, `errors`,
  `batches`, `gpu_memory`, `cold_start`, `config`. `--metrics-file PATH` writes it to disk every 2 s.
- With `--admin`, `POST /admin/mode` takes the same body as the mock (`normal`, `slow`, `error`,
  `hang`), so `loadgen.py run --events` works against real boxes. `slow` does not sleep: it starts
  `round(slow_factor) - 1` threads of matrix products on the same GPU. The slowdown you get is
  whatever the hardware gives; `cluster_bench.py` measures and reports it.
- `POST /admin/stop {"down_s": 15}` ends the process. Under `--supervise` it is started again after
  `down_s`: a real stop (connection refused) and a real cold start.
- `POST /admin/engine {"max_batch_rows": 1}` changes batching while running.

## For the weather team: plugging in a model

The engine calls one method on a backend:

```python
rollout(x, steps, row_steps=None, aux=None) -> np.ndarray   # (rows, steps, outputs)
```

`x` holds one row per ensemble member, from any number of requests. `row_steps` gives each row's
own forecast length (rows that finish early leave the batch). `aux` carries per-row extras.

`afno.py` is that backend for your AFNO. A row is a member's starting state `(20, 120, 240)`, `aux`
is its `(lat, lon)`, a step is 6 hours, and the output is the z-scores at the place. It reuses your
code for everything else: `Forecaster`, `StateStore.latest()`, `perturbed_states`, `grid_weights`,
`point_series`, `start_values`, and `to_v1` from `serve/forecast_server.py` for the response. So the
answer has the same shape as your server's (`precip_mm: null`, `native_step_hours`, `interpolated`).

```bash
python -m inference.server --model afno --weights /wx/bundle/weights/<file>.pt --data /wx/bundle --require-gpu
```

On a GPU with the real weights (`deploy/run_inference.sh afno`, workflow `664e809a`) its answers
matched your server's on 128 forecast points, and batching across requests did **not** raise
throughput: the GPU is already full with one request. It always starts from the newest analysis;
`init`, `member_offset` and `with_truth` are not supported yet.

## Measured on a DGX Spark (NVIDIA GB10, real GPU)

Workflow `4a0aaf44-70ed-4067-a176-0f05beed8d81`, node CIQ-CHI-SPARK28 (192.168.99.37), arm64,
image `nvcr.io/nvidia/pytorch:26.02-py3`, torch 2.11.0a0+nv26.02, CUDA available, fp32, tiny model,
48 h forecasts. Files: `results/inference/20261114-174419-fuzzball-bench/`. Charts:
`results/inference/report.html`.

| cold start | ms |
|---|---:|
| torch import | 1282 |
| model load | 430 |
| first inference | 523 |
| warm inference (12 h, 1 row) | 2.5 |

| batch rows | p50 ms | p95 ms | p99 ms | rows/s | peak GPU MB |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.95 | 10.02 | 10.04 | 101 | 11.1 |
| 4 | 11.29 | 11.44 | 11.55 | 354 | 11.2 |
| 16 | 11.24 | 11.27 | 11.30 | 1424 | 11.2 |
| 64 | 11.81 | 11.86 | 11.88 | 5420 | 11.4 |
| 256 | 10.45 | 10.52 | 10.53 | 24500 | 12.1 |

| clients | mode | p50 ms | p95 ms | p99 ms | req/s | mean batch | errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | no batching | 10.78 | 11.26 | 11.48 | 93 | 1.0 | 0 |
| 1 | batched | 10.53 | 11.38 | 11.56 | 93 | 1.0 | 0 |
| 8 | no batching | 85.33 | 87.97 | 89.05 | 94 | 1.0 | 0 |
| 8 | batched | 12.42 | 12.80 | 13.05 | 650 | 8.0 | 0 |
| 32 | no batching | 336.74 | 342.51 | 343.24 | 95 | 1.0 | 0 |
| 32 | batched | 12.66 | 13.71 | 13.72 | 2401 | 30.8 | 0 |

| precision | rows | p50 ms | rows/s | peak GPU MB | max diff vs fp32 | valid |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 256 | 10.55 | 24170 | 23.2 | | |
| fp16 autocast | 256 | 19.58 | 13072 | 23.1 | 0.365 | no |

- **The bottleneck is the call, not the GPU.** A call takes about 10 ms whether it carries 1 row or
  256, so without batching one GB10 serves about 95 req/s however many clients there are.
- **Batching what is already queued** gave 25× the throughput at 32 clients (95 → 2401 req/s) and
  cut p95 from 343 ms to 14 ms. With one client it changes nothing.
- **fp16 autocast was rejected.** It was 1.9× slower than fp32 and its outputs drifted by 0.37 on a
  −1…1 scale, so `--amp` is off by default.
- One run per row; treat differences of a few percent as noise. `nvidia-smi` reports no memory
  total on a GB10 (unified memory), so GPU memory is PyTorch's own count.
- After this run, mixed-length batches were changed so rows that finish early leave the batch.
  That is covered by tests but has not been re-measured on the GPU.

The smoke run before it (`f673064b-0cd3-4805-93c3-5ecdaac0d18d`, same node, fp16 at the time) is in
`results/inference/20261114-174253-fuzzball-smoke/`. The laptop CPU run is in
`results/inference/20261114-110854/` and is labelled CPU in the report.

## Fuzzball

```bash
deploy/run_inference.sh smoke    # 1 GPU, about 20 s of work
deploy/run_inference.sh bench    # 1 GPU, about 30 s of work
deploy/run_inference.sh afno     # 1 GPU, trained model, needs /wx/bundle, /wx/repo, /wx/pydeps
deploy/run_inference.sh demo     # both team GPUs, about 10 minutes; refuses to start unless both are free
```

Each regenerates `deploy/inference-<mode>.fz` from the current sources (they are embedded in the
Fuzzfile, as `build_fuzzfile.py` does), validates it, starts it, waits, and saves the result JSON
under `results/`. All five Fuzzfiles pass `fuzzball workflow validate` on CLI 4.3.1.

| workflow | what runs | resources |
|---|---|---|
| `inference-smoke.fz`, `inference-bench.fz` | one job: `nvidia-smi`, torch CUDA check, `bench` | 1 GPU, 10 cores, 16 GB |
| `inference-afno.fz` | one job: team server, then engine, same AFNO weights | 1 GPU, 10 cores, 48 GB, 20 min limit |
| `inference-demo.fz` | services `box-a`, `box-b`, `router`; job `demo` runs `cluster_bench.py` | 2 GPUs, 10 cores per box, 6 each for router and job; 30 min limit on the job |
| `inference-serve.fz` | the three services only | 2 GPUs. Add a hold job to keep it up (see `deploy/README.md`) |

What `demo` does, with one seeded trace (seed 42, 120 req/s) for every case:

1. Waits for both boxes and the router. Stops with an error unless both boxes are on CUDA and
   report different GPU UUIDs.
2. Measures how much slow mode slows box A.
3. One GPU without batching, then with batching (straight at box A).
4. Two GPUs through the router: `static`, `round_robin`, `least_loaded`, `adaptive`, `balanced`.
5. The same five with faults: box A slowed at 10 s, back to normal at 25 s, stopped at 35 s,
   restarted at 45 s (60 s in all). Adaptive runs with box A as its home box.
6. Ends. Fuzzball stops the services when the job ends, so nothing is left running.

Results go to `wxdesk:/wxrouter-inference/` and, through the job log, to `results/cluster/`.

Things learned on this cluster:

- Team 14 has **two nodes**: CIQ-CHI-SPARK27 and CIQ-CHI-SPARK28, one GPU each. Jobs from the
  `team14@fb-hack.local` account only land there. `fuzzball node list -a` shows whether their GPUs
  are free; `fuzzball workflow why <workflow> <stage-id>` explains a pending stage.
- Services cannot have `policy.timeout`; only jobs can. A workflow's services stop when its jobs end.
- The image cache fails now and then with `lookup fuzzball-objectcache on 10.96.0.10:53: no such
  host`. It hit workflows at 09:38, 11:03–11:19 and 11:33 and cleared by itself each time. A failed
  start costs nothing; start it again a few minutes later.
- `nvcr.io/nvidia/pytorch:26.02-py3` is cached (9.3 GB), and PyPI, nvcr.io and GitHub are reachable
  from jobs.

## Notes from the cluster runs

- A DGX Spark has ten fast cores (5–9, 15–19) and ten slow ones. The model's GPU call took 9.9 ms
  on a fast core and 56–67 ms on a slow one, and a small Fuzzball job gets slow cores only. The
  server pins itself to the fastest cores it is allowed (`--no-pin-fast-cores` turns that off),
  logs `slow_cores_only` when it was given none, and the workflows ask for 10 cores.
- A large result printed on one line breaks `fuzzball workflow log` (`bufio.Scanner: token too
  long`). `cluster_bench.py` prints its result gzip + base64 in short lines; `collect_bench.py`
  reads both forms.
- The team's GPUs are shared with teammates' jobs. `deploy/run_inference.sh demo` refuses to start
  unless both are free, and nothing here stops anyone else's workflow.
