# router_switching

GPU inference, batching, benchmarking and switching work for wxrouter, in one place, and shown in
the team's Control Room. It **reuses** what already exists: the router is `router.py`, the load
generator is `loadgen.py`, the inference server is `inference/`, the weather model is `weather/`
and `serve/`. Nothing here is a second router or a copy of them.

| file | what it does |
|---|---|
| `dashboard.py`, `static/` | Three cards added to the Control Room: GPU inference, intelligent switching, benchmarks and reliability tests |
| `collect.py` | Gathers every benchmark into `results/summary.json`, each run labelled **real GPU**, **CPU** or **simulated** |
| `threshold_eval.py` | Sweeps the adaptive rule's margin, window and cooldown on mock boxes (simulated) |
| `fuzzball.sh` | `status`, `one-gpu`, `afno`, `two-gpu`, `all`: bounded GPU runs on fb-hack, then refreshes the results |
| `tests/test_switching.py` | 14 tests of `router.py`'s hysteresis, dwell time, cooldown, failure detection, recovery, return home |
| `tests/test_dashboard.py` | 12 tests of the dashboard endpoints, the labels, and that `ui/` still serves its page |
| `results/` | `summary.json` (what the dashboard reads) and the threshold sweeps |

The measuring code itself lives where it is used: `inference/bench.py` (one box),
`inference/cluster_bench.py` (two boxes, every policy, slow / stop / restart),
`inference/afno_check.py` (the trained model) and `deploy/run_inference.sh` (Fuzzball).

## See it

```bash
.venv/bin/python dev_stack.py          # the team's local stack: two mock boxes, router, Control Room
open http://127.0.0.1:8080             # scroll down: the three new cards are under "Routing" and "Events"
```

With one real inference box (CPU on a laptop) next to a mock:

```bash
.venv/bin/python -m inference.server --admin --name box-a --port 8001
.venv/bin/python mock_server.py --name box-b --port 8002
.venv/bin/python router.py --policy adaptive --preferred box-a \
    --backend box-a=http://127.0.0.1:8001 --backend box-b=http://127.0.0.1:8002
.venv/bin/python ui/server.py --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002
```

On the cluster, port-forward the router and boxes of a running workflow and point `ui/server.py`
at them the same way; the cards then show `REAL GPU`.

## What the cards show, and where each number comes from

Nothing is estimated. A value a box does not report is shown as a dash or "unavailable".

**GPU inference** (per box, every 2 s)

| row | source |
|---|---|
| Runs on: `REAL GPU` / `CPU` / `SIMULATED` | `device.type` in the box's `GET /v1/metrics`; for the weather team's server, `GET /v1/info`; a mock is recognised by `mock` in its `model_version` |
| Available, mode | `GET /health`, `GET /stats` |
| GPU utilization | `gpu_util` in `/stats`, shown only for a box on CUDA (it comes from `nvidia-smi`). For CPU and mock boxes it says "unavailable" |
| Active requests, queue depth, request p50 / p95 | `GET /stats` |
| Inference p50 / p95, batch size, throughput, GPU memory | `GET /v1/metrics` (`compute_ms`, `batches`, `throughput`, `gpu_memory`) |

**Intelligent switching** (from the router, every 2 s)

| shown | source |
|---|---|
| Policy, preferred backend, active box, thresholds | `GET /router/status` → `policy`, `adaptive` |
| Health score, probe time, error share per box | `GET /router/status` → `backends` |
| Traffic split | `GET /router/metrics?window=10` → `per_box` |
| Last switch with its reason, the switch list, failovers | `GET /router/status` → `adaptive.switches`, `failovers` |

The reason shown is the router's own text (`box-b 50ms vs box-a 100ms`, `active box down`,
`back to preferred box-a ...`), so every switch on the page is a decision the router made. The
existing Events card shows the same switches.

**Benchmarks and reliability tests**: one tab per run from `results/summary.json`, each with its
label, its source file, latency (p50 / p95 / p99), requests per second, error rate, traffic share
and a table. Fault runs add the six steps (normal, GPU A slowdown, GPU A failure, failover to GPU B,
GPU A recovery, rebalancing) with the measured numbers, and a per-second timeline. "CPU vs GPU"
appears only when a real GPU run exists. What has not been measured on a GPU is listed at the top.

## How it is wired into the page

`ui/server.py` and `ui2/server.py` each gained one guarded block (7 lines) that calls
`router_switching.dashboard.attach(app, ops)`, and each `index.html` gained one `<script>` tag.
If `router_switching/` is missing the page works exactly as before. The cards use the page's own
CSS variables and classes. New endpoints: `GET /api/switching/live`, `GET /api/switching/results`.
No existing endpoint, control or file was changed otherwise.

## Tests

```bash
.venv/bin/python -m pytest router_switching/tests tests/test_inference_engine.py tests/test_inference_server.py \
    tests/test_cluster_bench.py tests/test_inference_afno.py tests/test_inference_fuzzfile.py -q
.venv/bin/python -m router_switching.threshold_eval      # simulated, about 5 minutes
.venv/bin/python -m router_switching.collect             # rebuild results/summary.json
```

## Results

RESULTS_SECTION
