# wxrouter — forecast mock, router, and load generator

A laptop test bed for routing weather-forecast requests across two DGX Spark boxes.
It covers the items from the Slack thread: a fixed API contract, a mock server with
`/health`, `/stats` and slow mode, a router with four policies and failover, and a
seeded load generator that reports p50/p95/p99.

**Project overview:** [docs/PROJECT.md](docs/PROJECT.md) has the status, progress log, diagrams (components,
request flow, deployment flow), routing rules, demo scenes, benchmarks and open questions. It's a copy of
the team's [live project doc](https://claude.ai/artifact/4EQyR9HZiknTTh7FY3CFt3).

| file | what it is |
|---|---|
| `API.md`, `contract.py` | The request/response contract. The real GPU server must match it. |
| `mock_server.py` | Fake GPU forecast server: same API, simulated delays, `/admin/mode` (normal / slow / error / hang) |
| `router.py` | One address in front of box A and box B: `static`, `round_robin`, `least_loaded`, `adaptive`, plus failover |
| `loadgen.py` | `trace` (seeded trace), `run` (replay against any router), `bench` (all policies, same trace, same faults) |

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

## Control Room UI

`python dev_stack.py` starts both mock boxes, the router and a dashboard with an agent chat at http://127.0.0.1:8080. See [ui/README.md](ui/README.md).

## Quick start: compare every policy (about 4 minutes)

```bash
.venv/bin/python loadgen.py bench --seed 42 --duration 60 --rps 25
```

This starts two mocks and a router for each policy, replays the same trace, and runs these
default faults: box A slowed ×4 (as if a training job were running), box A back to normal,
box B killed, box B restarted. Results go to `results/<timestamp>/` as `table.md`,
`summary.csv`, `summary.json`, and `raw-<policy>.jsonl`.

Useful flags: `--events "10:slow:box-b:4,30:normal:box-b"` sets custom faults (scene 3 as
described). `--policies adaptive,least_loaded` runs only some policies. `--margin`, `--window`
and `--cooldown` tune adaptive.

## Run the pieces by hand

```bash
.venv/bin/python mock_server.py --name box-a --port 8001
.venv/bin/python mock_server.py --name box-b --port 8002
.venv/bin/python router.py --backend box-a=http://127.0.0.1:8001 --backend box-b=http://127.0.0.1:8002 --policy adaptive
```

```bash
curl -s -XPOST localhost:8000/v1/forecast -H 'content-type: application/json' -d '{"lat":37.7,"lon":-122.4,"hours_ahead":48}' -D - -o /dev/null
curl -s localhost:8000/router/status
curl -s -XPOST localhost:8000/router/config -H 'content-type: application/json' -d '{"policy":"least_loaded"}'
curl -s -XPOST localhost:8002/admin/mode -H 'content-type: application/json' -d '{"mode":"slow","slow_factor":4}'
```

To point the router at the real boxes, change the `--backend` URLs. Nothing else changes, as
long as the real server implements `API.md`.

## Real inference server (GPU, with CPU fallback)

`python -m inference.server --name box-a --port 8001` is a drop-in replacement for
`mock_server.py`: same API, real model runs, batching, and `/v1/metrics` with p50/p95/p99,
throughput, load and GPU memory. Setup, benchmarks and the Fuzzball GPU workflows are in
[inference/README.md](inference/README.md).

## How adaptive works

- Every 0.5 s the router sends a small probe forecast (12 h) to each box without waiting
  for the reply. The score is the box's **GPU compute time** (`timing.compute_ms`), not the
  round trip. The round trip includes queueing behind the router's own traffic, so the busy
  box always looked worse and adaptive switched for no reason.
- A probe stuck for more than `--stall-ms` (1000), or rejected with 503, counts as a slow box.
- **Leave** the active box when the other is at least `margin` (25%) faster for `window_s`
  (1 s) straight, and at least `cooldown_s` (10 s) after the last switch.
- **Return home:** with `--preferred box-b`, traffic starts on B and returns once B is
  healthy and no more than `return_margin` (20%) slower than the active box for 1 s. The
  25%/20% gap stops it bouncing. Setting `preferred` at runtime:
  `POST /router/config {"preferred": "box-b"}` (`""` clears it). All routing settings, drain and reset are in
  [API.md](API.md#router-routing-config).
- **Dead active box:** the switch is immediate.

Scene 3 on the laptop (`--rps 15 --preferred box-b --events "8:slow:box-b:2,22:normal:box-b"`):
B left once because of slowness about 2 s after training started, and traffic returned to B
about 3 s after it ended. There were 2 switches and 0 errors. With a harsher 4× slowdown at
25 req/s, B's backlog times out first, so the switch reason shows "active box down". The
outcome is the same: one switch away, then back to B.

## The balanced rule ("Both" in the UI)

`--policy balanced` keeps both boxes in play. For each request it computes a **health score** per
box (lower is better) and picks a box at random with weight 1/score², so the better box gets most of
the traffic but the other keeps getting some, which keeps its numbers current:

```
score = (gpu_ms_per_unit × (1 + active) + health_ms) × (1 + 5 × error_rate) × history
```

| input | where it comes from |
|---|---|
| `gpu_ms_per_unit` | the box's own `timing.compute_ms` ÷ work units (a unit is 6 forecast hours × one ensemble member), smoothed; before any traffic, the probe |
| `active` | router in-flight + the box's `queue_depth`, ÷ `max_concurrency` |
| `health_ms` | round trip of `GET /health`, polled every 0.25 s, smoothed |
| `error_rate` | smoothed share of recent requests that errored or timed out (decays as the box serves cleanly) |
| `history` | 0.75 to 1.5: above 1 when the box served more than its share of work units in the last 30 s |

`GET /router/status` shows each box's `health_score` with these parts, and the Control Room shows
them on each box card. In the UI, choosing **Both** in "Preferred box" turns on `balanced`.
Choosing a box turns on `adaptive` with that box as home.

Laptop result (seed 42, 30 s, 20 req/s; box A slowed 4×, then box B killed and restarted):

| rule | errors | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| static | 0 | 192 | 5654 | 5880 |
| round_robin | 0 | 122 | 2478 | 2966 |
| least_loaded | 0 | 98 | 380 | 1090 |
| adaptive | 0 | 117 | 2483 | 3819 |
| **balanced** | 0 | 100 | **350** | **758** |

While box A was slowed, balanced sent it 18% of the traffic (least_loaded sent 29%). After box B
came back, its recent errors kept its share low for a while, and it recovered as they decayed.

## First laptop result (seed 42, 40 s, 25 req/s, 987 requests)

| policy | errors | p50 ms | p95 ms | p99 ms | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 0 | 218 | 5293 | 5809 | 77 | 23 | 81 | |
| round_robin | 0 | 135 | 3654 | 5440 | 56 | 44 | 16 | |
| least_loaded | 0 | 105 | 401 | 1071 | 50 | 50 | 2 | |
| adaptive | 0 | 160 | 498 | 799 | 56 | 44 | 5 | 2 |

- No requests failed under any policy, including when box B was killed. Failover worked.
- While box A was slowed, p95 was 5.8 s for static and 0.47 s for adaptive. Adaptive switched
  once, about 1 s after the slowdown started, and once more when B was killed. It never flapped.
- Least-loaded had the best p50 and p95 because it uses both boxes. Adaptive had the best p99
  because it moves all traffic off a sick box.
- These numbers come from the mock's cost model. Rerun on the real boxes before drawing conclusions.

## Deploying on Fuzzball

See [deploy/README.md](deploy/README.md). `python deploy/fb.py up` deploys `fuzzball/30-real-stack.fz`: the real AFNO model and Nemotron on both boxes, the router and the Control Room. Only the real stack is deployed; mock boxes are for laptop development and tests.

## Where vllm-router fits

[vllm-project/router](https://github.com/vllm-project/router) is a Rust router for
OpenAI-compatible **LLM** endpoints. Its policies are `round_robin`, `random`,
`consistent_hash`, `power_of_two`, `cache_aware` and `kv_aware`, plus `sticky_least_loaded` in
the CLI. It also has retries, a circuit breaker, health checks and Prometheus metrics.

- **Use it for the Nemotron/LLM traffic.** `cache_aware` (prefix reuse) is something this router doesn't do.
- **Its docs don't say it forwards custom paths like `POST /v1/forecast`.** Every example uses
  `/v1/chat/completions`. Until that's tested, keep this router (or a path-based proxy) in front of the weather service.
- **Its health check runs every 60 s by default.** For scene 1 (B fails and nobody notices), set
  `--health-check-interval-secs` to about 1–2 s. Retries (on 5xx/429) will cover requests in the meantime.
- **Nothing in it matches the "adaptive" rule** (switch only when clearly better for N seconds,
  then cool down). `sticky_least_loaded` is the closest. You can benchmark it with this harness:
  `loadgen.py run --router <vllm-router url>` works against any router.
- **Not tested here.** PyPI only has Linux wheels (`manylinux` aarch64/x86_64), so it installs on
  the DGX Sparks but not on macOS without a Rust build. Its docs target vLLM workers. Whether it
  works cleanly in front of llama.cpp's `llama-server` also needs a test on the boxes.

Suggested layout: one entry point that routes `/v1/chat/completions` → vllm-router (LLM) and
`/v1/forecast` → this router (weather).

## Deployment: Fuzzball + VAST (plan, not built yet)

- **Fuzzball (CIQ)** would run the workflows: start forecast servers, the LLM server and the
  routers as services on each box, and run the scene 3 fine-tuning job as a batch step.
  Fuzzball can stop and start services, so it can also drive scene 1.
- **VAST Data** would hold the shared data: model weights and checkpoints (so both boxes load the
  same tuned model during the one-box-at-a-time swap), the NOAA training data, and benchmark
  results from `loadgen.py`.
- **Open questions:** does Fuzzball give each service a stable address the router can use as
  `--backend`? Will both boxes mount VAST through the same path or the same S3 bucket?

## Real model and GPU forecast server (merged from the Switchyard work)

These folders add the real weather model and a GPU server that answers this repo's contract (`API.md` v1.0).
It can stand where `mock_server.py` stands, with no change to `router.py`.

| folder | what it is |
|---|---|
| `weather/` | The model code written at the event: trainer (`train.py`), scorer (`score.py`), live NOAA GFS download (`gfs_live.py`, `gfs_archive.py`), fine-tune job (`finetune.py`), bundle maker (`make_bundle.py`) |
| `serve/forecast_server.py` | Forecast server for one box: the real AFNO model on the GPU, or `--mock` with no model. Answers `POST /v1/forecast`, `GET /health`, `GET /stats`, `POST /admin/mode` |
| `serve/node_stack.py` | A small supervisor for one box. It keeps the forecast server (and an LLM server) alive and starts a fine-tune job on request. Fuzzball gives a box's GPU to one job at a time, so everything that needs the GPU has to live in one service |
| `router/`, `dashboard/`, `agent/`, `bench/`, `tools/`, `serve/mock_llm.py` | A second router, dashboard, agent tools and load generator, written overnight before the two code bases met. Kept as a cross-check. The main router is `router.py` and the main page is `ui/` |
| `fuzzball/` | Fuzzfiles. Those in `fuzzball/quick/` ran on the event cluster (GPU hello, image pull, shared-volume test). The others have not been run yet |
| `tests/` | Unit tests for these folders, standard library only: `python -m unittest tests/test_forecast_server.py` |
| `docs/CONTRACT.md` | The wider interface those pieces were first built against. `API.md` is the team contract |

Try it on a laptop behind `router.py` (no GPU and no model file needed):

```bash
.venv/bin/python serve/forecast_server.py --mock --admin --node box-a --port 8001
.venv/bin/python serve/forecast_server.py --mock --admin --node box-b --port 8002
.venv/bin/python router.py --backend box-a=http://127.0.0.1:8001 --backend box-b=http://127.0.0.1:8002 --policy least_loaded
curl -s -XPOST localhost:8000/v1/forecast -H 'content-type: application/json' -d '{"lat":41.98,"lon":-87.9,"hours_ahead":48,"ensemble_members":3}' -D - -o /dev/null
```

With the model, drop `--mock` and point the server at a bundle made by `weather/make_bundle.py`
(`--weights BUNDLE/weights/afno_event.pt --data BUNDLE`). The model file (77 MB) is not in the repository.

What a caller should know about the real server:

- `precip_mm` is `null` in every point. The model has no rain output.
- The model steps 6 hours at a time. Hourly points are interpolated between its steps; the answer says so
  (`native_step_hours`, `interpolated`).
- A large request takes seconds, not milliseconds: 20 members x 120 h took 3.9 s on one GB10, and 8.4 s while
  a fine-tune ran on the same GPU. Start `router.py` with a `--timeout` above its default of 5 s for those.
- `/stats` reports `mode: training` when something else is using the box's GPU.

Measured so far, all on one GB10 (a team member's own DGX Spark, Nov 13-14, 2026; not yet on the event boxes):

| what | result |
|---|---|
| Model | NVIDIA PhysicsNeMo AFNO, 19.1M parameters, trained from scratch in 5 h 14 min with `weather/train.py` |
| 3-day error, 2020 test year | Z500 301.85 m2/s2 (956 for "same as now"), T850 1.67 K, T2M 1.44 K |
| Server | ready in 5.3 s; one 48-h forecast in 62 ms; 20 members x 120 h in 3.9 s |
| Sharing a GPU with a fine-tune | forecasts 2.2 times slower; the fine-tune 12% slower |
| Fine-tune on one year of NOAA GFS analyses (300 steps, about 10 min) | 24-h errors down 11% (Z500), 18% (T850), 17% (T2M) on 83 held-out forecasts; at 72 h Z500 2% worse, T850 2% and T2M 5% better |
