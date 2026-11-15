# wxrouter in detail: the laptop test bed, the routing rules, the benchmark and the server

This is the project's README as it stood until Nov 15, 2026, moved here when the front page was
rewritten as a summary. The text is the same except for the title, the links (adjusted to this folder)
and three corrections: the note under "The first plan for the deployment", the install line under "Tests"
and the folder of the first plan's files. It is in the order the project grew:
the laptop test bed first, then the same benchmark on the event boxes, then the real model and its server.
Paths in code blocks are from the repository root. The summary, the figures and the sponsor table are in
the [README](../README.md).

Team 14, CIQ x NVIDIA Supercomputing Local AI Hackathon (Chicago, Nov 13-15, 2026).

A lab buys two desk-side GPU boxes. It wants to serve models and fine-tune on the same two, with nobody on
call. This project puts one address in front of both. A router sends each request to the box that can
answer it best, so the service keeps answering while one box trains or fails. The load is real: on each box
a weather model (NVIDIA PhysicsNeMo AFNO, 19.1M parameters) and an LLM assistant (NVIDIA Nemotron 3.5
Lightning on llama.cpp), started by CIQ Fuzzball from one stack file.

**Measured on the two event boxes on Nov 14** (two NVIDIA GB10; one run each; method, caveats and raw files
in [results/event-boxes-2026-11-14.md](../results/event-boxes-2026-11-14.md)):

| what | result |
|---|---|
| Five routing rules, the same 1,182 requests each, a real fine-tune on box A and a kill of box B's server | 0 of 5,910 requests failed |
| While box A trained: every request to box A, against least loaded | p95 3.2 s against 0.32 s (the same 534 requests) |
| A real fine-tune beside serving (300 steps, 12 min) | 24-hour errors down 11.1%, 18.5% and 16.9% on 83 held-out forecasts; at 72 hours one of the three scores is 2.0% worse |
| The tuned model goes live, one box at a time, under traffic | 0 of 465 requests failed; each box back in 11 s |
| A forecast server killed under traffic | 0 of 298 requests failed; back by itself in 12 to 14 s |
| One 20-member, 120-hour forecast split across both boxes | 2.0 to 2.1 s, against 4.2 s on one box |
| One box's LLM server killed during a chat (second router) | 0 of 38 chat requests failed |

Checked on one GB10 the same night ([one chat question followed to the model, and what the page's Demo
traffic does to the GPU](../results/one-gb10-2026-11-14-night/README.md); [accuracy, precision, recall and F1
before and after the fine-tune](../results/event-scores-2026-11-14/README.md); [the chat's follow-up answers
before and after a fix, what three reads of that fix found, and its known
limits](../results/chat-follow-ups-2026-11-14/README.md)). That fix has not run on the event boxes yet.

**Project overview:** [docs/PROJECT.md](PROJECT.md) has the diagrams (components, request flow,
deployment flow), the routing rules, the demo scenes and the progress log. It is a copy of the team's
working document, taken on Nov 14 in the afternoon; where it differs from the results file, the results
file is newer.

| where | what it is |
|---|---|
| `router.py` | The router: one address in front of box A and box B. Five rules (`static`, `round_robin`, `least_loaded`, `adaptive`, `balanced`) and failover |
| `serve/`, `weather/` | The forecast server for one box, its supervisor, and the weather model's code (trainer, scorer, fine-tune) |
| `ui/` | The Control Room: the page, its assistant, the demo traffic |
| `router/` | The second router (`router2`): LLM chat to either box, and one ensemble split across both |
| `fuzzball/`, `deploy/` | The stack file that runs all of it on the event boxes, and its launcher |
| `bench/`, `results/` | The measurement scripts and what they returned |
| `API.md`, `contract.py` | The request/response contract of a forecast server |
| `mock_server.py`, `loadgen.py`, `dev_stack.py` | The laptop test bed the project began with: mock boxes with simulated delays, and a seeded load generator (`trace`, `run`, `bench`) |

The rest of this page is in the order the project grew: the laptop test bed first, then the same benchmark
on the event boxes, then the real model and its server.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

## Control Room UI

`python dev_stack.py` starts both mock boxes, the router and a dashboard with an agent chat at http://127.0.0.1:8080. See [ui/README.md](../ui/README.md).

## Quick start on a laptop: compare every policy on mock boxes (about 6 minutes)

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

To point the router at the real boxes, change the `--backend` URLs and give it a longer `--timeout` (the
deployed stack uses 30 s; a large ensemble takes seconds). Nothing else changes, as long as the real server
implements `API.md`.

## A second inference server (GPU, with CPU fallback)

`python -m inference.server --name box-a --port 8001` is a drop-in replacement for
`mock_server.py`: same API, real GPU work, batching, and `/v1/metrics` with p50/p95/p99,
throughput, load and GPU memory. Its default model is a small stand-in with random weights (it exercises
the GPU; its numbers are not a forecast); `--model afno` serves the trained model. The deployed stack uses
`serve/forecast_server.py` (below). Setup, benchmarks and the Fuzzball GPU workflows are in
[inference/README.md](../inference/README.md).

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
  [API.md](../API.md#router-routing-config).
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

*From the first night, on mock boxes. The files of this run were not kept, and mock runs differ from run to
run: in a stored run of the same trace (`results/all/20261114-064333/`) adaptive has the worst p99 of the four, not
the best, and on the real boxes below it is the slowest of the rules that route. Read this table as how the project started; the table that reproduces from
stored rows is the event-box one in the next section.*

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

## The same benchmark on the event boxes (real model, real faults)

On Nov 14 the comparison above was run against the deployed stack: the real weather model on both
GB10 boxes, `router.py` in front of them, and real faults in place of the mock's. `bench/fault_table.py`
replays the same seeded trace as `loadgen.py bench` (seed 42, 150 s, about 8 requests a second, 1,182
requests, the mix above) under each rule and fires the same events each time: a real fine-tune starts on
box A's GPU at 10 s and is stopped at 90 s, and box B's forecast server is killed at 105 s (its
supervisor starts it again).

| rule | errors | p50 ms | p95 ms | p99 ms | p95 ms while box A trained (requests planned at 10 to 80 s) |
|---|---:|---:|---:|---:|---:|
| static | 0 | 3,075 | 26,893 | 28,613 | 3,170 |
| round_robin | 0 | 74 | 509 | 1,220 | 697 |
| least_loaded | 0 | 81 | 387 | 873 | 319 |
| adaptive | 0 | 109 | 1,004 | 2,310 | 1,161 |
| **balanced** | 0 | 82 | 411 | 793 | 411 |

- No request failed under any rule: 0 of 5,910. Box B's server answered again 11 to 14 s after the kill.
- **While box A trained**, sending it every request (static) cost a p95 of 3.2 s, and 227 of 534 requests
  waited more than a second. Least loaded and balanced, with the same requests: 0.32 and 0.41 s, and 0 and
  4 over a second. That is the fine-tune, like for like: 10 times at p95.
- **The whole-run numbers of the static row are larger than that for a second reason.** From 82 s to 127 s
  box A answered five times slower than normal, the queue grew to 214 requests, and 423 requests waited
  more than 10 s. The fine-tune had been stopped by then, and we have not found the cause; a replay on one
  GB10 the same night did not show it (p95 2.3 s over the whole trace). With least loaded, balanced or
  round robin the p95 over the whole run was 0.4 to 0.5 s, and at most 22 requests waited more than a second.
- Adaptive, with one active box, was at 1.0 s: twice it moved to the box with the fine-tune on it, once
  because the other box was slow for a few seconds and its probe stalled.
- One run per rule. The method, the per-window numbers and the raw rows are in
  [results/event-boxes-2026-11-14.md](../results/event-boxes-2026-11-14.md), section 10.

Two more result folders from the evening of Nov 14, both from one GB10 with the real model:
[results/one-gb10-2026-11-14-night](../results/one-gb10-2026-11-14-night/README.md) follows one chat question
through every hop to the model and measures what the page's "Demo traffic" does to the GPU;
[results/event-scores-2026-11-14](../results/event-scores-2026-11-14/README.md) has accuracy, precision, recall
and F1 of the forecasts before and after the fine-tune, for two yes/no questions.

```bash
# inside the running real stack (fuzzball workflow exec <id> box-a -- bash -c "..." < /dev/null)
cd /wx/repo && python3 bench/fault_table.py --router http://router:8000 --out /wx/results
```

## Deploying on Fuzzball

See [deploy/README.md](../deploy/README.md). `python deploy/fb.py up` deploys `fuzzball/30-real-stack.fz`: the real AFNO model and Nemotron on both boxes, the router and the Control Room. Only the real stack is deployed; mock boxes are for laptop development and tests.

## Where vllm-router fits (a note from the first day; not tested)

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

## The first plan for the deployment (Nov 13, kept for the record)

*What was built is in "Deploying on Fuzzball" above and in [fuzzball/README.md](../fuzzball/README.md). Of
the two open questions below, the first was answered on the event cluster: services of one workflow reach
each other by name. For the second: both boxes mount one shared Fuzzball volume at the same path; which
storage system is behind that volume we have not confirmed.*

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
| `router/` | The stack's second router (`router/router.py`, service `router2`): the page's chat goes through it to the LLM on either box, and it splits one ensemble forecast across both boxes |
| `bench/` | The measurement scripts behind `results/` (five-rule fault table, fine-tune scene, model swap, split, pre-flight check) |
| `dashboard/`, `agent/`, `tools/`, `serve/mock_llm.py` | A first dashboard, agent tools and a mock LLM, written overnight before the two code bases met. Kept as a cross-check; the deployed stack does not use them. The main router is `router.py` and the main page is `ui/` |
| `fuzzball/` | Fuzzfiles. `30-real-stack.fz` is the deployed stack; it ran on the event boxes on Nov 14 ([results](../results/event-boxes-2026-11-14.md)). `quick/` holds the one-time jobs (GPU hello, image pull, shared-volume test, model and data downloads). The numbered files 00 to 50 in `fuzzball/first-plan/` are the first night's plan and were not run |
| `tests/` | Unit tests. How to run them is under "Tests" below |
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

Measured on one GB10 (a team member's own DGX Spark, Nov 13-14, 2026). The same server and the same fine-tune
on the two event boxes: [results/event-boxes-2026-11-14.md](../results/event-boxes-2026-11-14.md).

| what | result |
|---|---|
| Model | NVIDIA PhysicsNeMo AFNO, 19.1M parameters, trained from scratch in 5 h 14 min with `weather/train.py` |
| 3-day error, 2020 test year | Z500 301.85 m2/s2 (956 for "same as now"), T850 1.67 K, T2M 1.44 K |
| Server | ready in 5.3 s; one 48-h forecast in 62 ms; 20 members x 120 h in 3.9 s |
| Sharing a GPU with a fine-tune | forecasts 2.2 times slower; the fine-tune 12% slower |
| Fine-tune on one year of NOAA GFS analyses (300 steps, about 10 min) | 24-h errors down 11% (Z500), 18% (T850), 17% (T2M) on 83 held-out forecasts; at 72 h Z500 2% worse, T850 2% and T2M 5% better |

## Tests

```bash
pip install pytest pyyaml numpy
# everything that needs no PyTorch (about 2 minutes, about 770 tests)
python -m pytest tests router_switching --ignore=tests/test_finetune.py --ignore=tests/test_forecast_core.py --ignore=tests/test_train.py --ignore=tests/test_score.py --ignore=tests/test_gfs_live.py --ignore=tests/test_event_scores.py --ignore=tests/test_end_to_end.py -q
# the end-to-end test starts a small cluster on fixed local ports: run it on its own
python -m pytest tests/test_end_to_end.py -q
# the weather model's tests need PyTorch (some also need the model file and skip without it)
python -m unittest tests.test_forecast_core tests.test_score tests.test_gfs_live tests.test_finetune tests.test_event_scores tests.test_forecast_server
```

## What was brought to the event, and what was built there

- **Brought.** Four things, all from one team member's work on this kind of weather model before the
  event: the training recipe (the settings that `weather/train.py` implements); the ERA5 training data, packed
  before the event (`weather/wx_common.py`); reference measurements, among them the scores of a model of the
  same recipe trained on Nov 10, which `weather/score.py` checks itself against (that model is not served
  and is not in this repository); and the container image with PyTorch and PhysicsNeMo that the training ran
  in. The public components under "Credits" were brought as they are.
- **Built at the event.** The code in this repository (first commit Nov 13, 11:33 PM); the weather model's
  code was written after kickoff, from that recipe. The model that the boxes serve was trained from scratch
  overnight on Nov 13-14 with `weather/train.py`, in 5 h 14 min, on that team member's own DGX Spark (one
  GB10, not one of the event's boxes).
- **Run on the event boxes.** Serving, routing, the fine-tune on NOAA analyses, the model swap and every
  measurement in [results/event-boxes-2026-11-14.md](../results/event-boxes-2026-11-14.md). The from-scratch
  training was not run there. Results from the team member's own GB10 are marked as such wherever they appear.

## Credits

- Weather model: NVIDIA PhysicsNeMo's AFNO (`nvidia-physicsnemo` 2.2.2); the weights are ours.
- LLM: NVIDIA Nemotron 3.5 Lightning 30B-A3B (4-bit GGUF), served by llama.cpp (release build b12359).
- Training data: ERA5 (Hersbach et al.), via WeatherBench 2. Generated using Copernicus Climate Change
  Service information [2026]. Neither the European Commission nor ECMWF is responsible for any use made of it.
- Starting weather and fine-tune data: NOAA GFS analyses.
- Platform: CIQ Fuzzball 4.3.1 on the event's NVIDIA GB10 boxes.
- FastAPI, uvicorn, httpx and pydantic for the servers.
- Parts of the code and of the documentation were written with AI coding assistants (Claude Code among
  them; see the commit trailers).
