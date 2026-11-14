# wxrouter — forecast mock, router, and load generator

A laptop test bed for routing weather-forecast requests across two DGX Spark boxes.
It covers the items from the Slack thread: a fixed API contract, a mock server with
`/health`, `/stats` and slow mode, a router with four policies and failover, and a
seeded load generator that reports p50/p95/p99.

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
  `POST /router/config {"preferred": "box-b"}` (`""` clears it).
- **Dead active box:** the switch is immediate.

Scene 3 on the laptop (`--rps 15 --preferred box-b --events "8:slow:box-b:2,22:normal:box-b"`):
B left once because of slowness about 2 s after training started, and traffic returned to B
about 3 s after it ended. There were 2 switches and 0 errors. With a harsher 4× slowdown at
25 req/s, B's backlog times out first, so the switch reason shows "active box down". The
outcome is the same: one switch away, then back to B.

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
