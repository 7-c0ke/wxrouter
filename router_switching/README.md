# router_switching

GPU inference, batching, benchmarking and switching work for wxrouter, in one place, and shown in
the team's Control Room. It **reuses** what already exists: the router is `router.py`, the load
generator is `loadgen.py`, the inference server is `inference/`, the weather model is `weather/`
and `serve/`. Nothing here is a second router or a copy of them.

| file | what it does |
|---|---|
| `dashboard.py` | APIs added to the Control Room server: `/api/switching/live`, `/api/switching/results`, `/api/switching/burst`. The page shows no cards for them |
| `threshold_eval.py` | Sweeps the adaptive rule's margin, window and cooldown on mock boxes (simulated) |
| `fuzzball.sh` | `status`, `one-gpu`, `afno`, `two-gpu`, `all`: bounded GPU runs on fb-hack, then refreshes the results |
| `tests/test_switching_rules.py` | 14 tests of `router.py`'s hysteresis, dwell time, cooldown, failure detection, recovery, return home |
| `tests/test_switching_dashboard.py` | 12 tests of the dashboard endpoints, the labels, and that `ui/` still serves its page |
| `results/screenshots/` | two screenshots of the real two-GPU runs (the dashboard reads no result files) |

The measuring code itself lives where it is used: `inference/bench.py` (one box),
`inference/cluster_bench.py` (two boxes, every policy, slow / stop / restart),
`inference/afno_check.py` (the trained model) and `deploy/run_inference.sh` (Fuzzball).

## See it

```bash
.venv/bin/python dev_stack.py          # the team's local stack: two mock boxes, router, Control Room
curl -s localhost:8080/api/switching/live      # what each box and the router report now (no cards on the page)
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
at them the same way; `/api/switching/live` then reports `real GPU`.

## See it on the real GPUs

```bash
router_switching/live_demo.sh            # attaches to the running two-GPU workflow (starts one only if both team GPUs are free)
                                         # opens http://127.0.0.1:18180 with the Control Room on the real boxes
```

The APIs return only what the router and the boxes report live. There are no cards on the page,
no scripted demo and no recorded run.

## What the APIs return, and where each number comes from

Nothing is estimated. A value a box does not report is shown as a dash or "unavailable".

**`GET /api/switching/live` → `boxes`** (per box)

| row | source |
|---|---|
| Runs on: `REAL GPU` / `CPU` / `SIMULATED` | `device.type` in the box's `GET /v1/metrics`; for the weather team's server, `GET /v1/info`; a mock is recognised by `mock` in its `model_version` |
| Available, mode | `GET /health`, `GET /stats` |
| GPU utilization | `gpu_util` in `/stats`, shown only for a box on CUDA (it comes from `nvidia-smi`). For CPU and mock boxes it says "unavailable" |
| Active requests, queue depth, request p50 / p95 | `GET /stats` |
| Inference p50 / p95, batch size, throughput, GPU memory | `GET /v1/metrics` (`compute_ms`, `batches`, `throughput`, `gpu_memory`) |

**`GET /api/switching/live` → `router`**

| shown | source |
|---|---|
| Policy, preferred backend, active box, thresholds | `GET /router/status` → `policy`, `adaptive` |
| Health score, probe time, error share per box | `GET /router/status` → `backends` |
| Traffic split | `GET /router/metrics?window=10` → `per_box` |
| Last switch with its reason, the switch list, failovers | `GET /router/status` → `adaptive.switches`, `failovers` |

The reason shown is the router's own text (`box-b 50ms vs box-a 100ms`, `active box down`,
`back to preferred box-a ...`), so every switch on the page is a decision the router made. The
existing Events card shows the same switches.

**`GET /api/switching/results`**: live, never read from disk. The "Live" run is the router's own `GET /router/metrics` over the last 10 s, 1 min and 5 min: latency
(p50 / p95 / p99), requests per second, error rate, traffic share per box, failovers and switches,
labelled real GPU, CPU or simulated from what the boxes report. After `POST /api/switching/burst`, a second run shows that burst.

## How it is wired in

`ui/server.py` has one guarded block (7 lines) that calls
`router_switching.dashboard.attach(app, ops)`. The page loads no script from here and shows no cards.
If `router_switching/` is missing the page works exactly as before. Endpoints: `GET /api/switching/live`, `GET /api/switching/results` and `POST /api/switching/burst`.
No existing endpoint, control or file was changed otherwise.

## Tests

```bash
.venv/bin/python -m pytest router_switching/tests tests/test_inference_engine.py tests/test_inference_server.py \
    tests/test_cluster_bench.py tests/test_inference_afno.py tests/test_inference_fuzzfile.py -q
.venv/bin/python -m router_switching.threshold_eval      # simulated, about 5 minutes
```

## Results

The runs below were measured on Nov 14, 2026. Their result files are no longer in the repo, and
the dashboard does not read any: it shows live numbers only.

### Real GPU: two DGX Sparks (workflow `c5ffff9f-7019-4d0f-b6d7-7c7b159294ec`)

box-a on CIQ-CHI-SPARK27 (GPU `GPU-ef16568f…`), box-b on CIQ-CHI-SPARK28 (GPU `GPU-a54b10ad…`), both NVIDIA
GB10 on CUDA, fp32, tiny stand-in model. One seeded trace at 120 req/s for every case (30 s steady,
60 s with faults). File: `results/cluster/20261114-182549-fuzzball-demo/`.

| case | errors | p50 ms | p95 ms | p99 ms | served/s | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one GPU, no batching | 29 | 906.1 | 3002.7 | 5023.5 | 78.2 | 100 | 0 | | |
| one GPU, batching | 0 | 25.9 | 50.9 | 62.0 | 117.9 | 100 | 0 | | |
| two GPUs, static | 0 | 32.1 | 60.7 | 77.2 | 117.9 | 100 | 0 | 0 | |
| two GPUs, round robin | 0 | 20.9 | 42.7 | 53.6 | 117.9 | 50.0 | 50.0 | 0 | |
| two GPUs, least loaded | 0 | 21.2 | 41.5 | 51.6 | 117.9 | 50.2 | 49.8 | 0 | |
| two GPUs, adaptive | 0 | 30.0 | 56.7 | 68.4 | 117.9 | 41.0 | 59.0 | 0 | 3 |
| two GPUs, balanced | 0 | 21.4 | 45.1 | 56.8 | 117.9 | 49.4 | 50.6 | 0 | |
| faults, static | 0 | 35.1 | 64.1 | 78.8 | 119.9 | 77.1 | 22.9 | 3 | |
| faults, round robin | 0 | 23.2 | 50.5 | 62.6 | 119.9 | 38.4 | 61.6 | 1 | |
| faults, least loaded | 0 | 22.8 | 49.3 | 60.1 | 119.9 | 38.6 | 61.4 | 2 | |
| faults, adaptive (home box-a) | 0 | 34.1 | 67.1 | 109.4 | 119.9 | 41.6 | 58.4 | 3 | 6 |
| faults, balanced | 0 | 26.9 | 54.7 | 71.6 | 119.9 | 40.3 | 59.7 | 1 | |

Faults: box A slowed at 10 s, normal at 25 s, its server stopped at 35 s, started again at 45 s.

What this run shows:

- **Batching is what makes one GPU enough.** At 120 req/s one box without batching fell behind
  (p50 0.9 s, 29 requests lost); with batching it served all of them at p50 26 ms.
- **Failover worked under every policy: 0 of 7,202 requests failed** while box A's server was
  stopped for 10 s. Box A served again 3 to 5 s after its restart (a cold start: process, model
  load, warm-up, then two good health checks).
- **Two GPUs lowered latency by about a fifth** against one (p50 21 ms vs 26 ms, p95 42 ms vs 51 ms)
  with round robin, least loaded or balanced. Throughput is the same because the offered load was
  the same; this trace does not find the two-box ceiling.
- **Adaptive did not help here, and it oscillated.** With no fault at all it switched 3 times in
  30 s, and 6 times in the fault run, about once per cooldown. The reason is in its own log:
  `box-b 4ms vs box-a 11ms`. The box carrying the traffic runs larger batches, so its probe is
  slower than the idle box's by more than the 25% margin, and the rule reads "busy" as "slow". It
  had the worst p99 of the two-box policies (109 ms). This is a property of the rule meeting a
  batching server, not a bug in either; a probe that is not batched with traffic, or a larger
  margin, should be tried before using adaptive with these boxes.
- **The slowdown step was mild.** With the boxes on fast cores, slow mode made a forecast 1.25×
  slower (11.0 → 13.7 ms). Only balanced (53% → 42%) and adaptive shifted traffic visibly. A
  heavier load (`--slow-factor`) is needed to test a training-sized slowdown.

### Real GPU: the same test with a heavy slowdown (workflow `6e7d8135-2ba8-4ff6-b96f-fb08e8644c96`, group `team14`)

Same boxes, nodes, trace and fault times. Only the slowdown differs: `--slow-factor 16`, which made
one forecast on box A **41× slower** (11 → 451 ms). This is the run where the policies separate.
File: `results/cluster/20261114-184739-fuzzball-serve-team14/`. The steady cases repeated the
earlier run to within a few milliseconds.

| faults, policy | errors | p50 ms | p95 ms | p99 ms | served/s | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 1 | 483.0 | 5913.6 | 8815.7 | 89.2 | 78.4 | 21.6 | 114 | |
| round robin | 0 | 237.3 | 3083.6 | 6439.4 | 120.0 | 34.7 | 65.3 | 2 | |
| least loaded | 0 | 24.9 | 55.8 | 678.0 | 119.9 | 27.6 | 72.4 | 1 | |
| adaptive (home box-a) | 0 | 31.8 | 61.7 | 77.8 | 119.9 | 30.3 | 69.7 | 4 | 4 |
| balanced | 0 | 28.3 | 59.3 | 92.9 | 120.0 | 32.0 | 68.0 | 1 | |

- **With a real slowdown, switching matters.** Static stayed on the slow box (p95 5.9 s, one lost
  request). Round robin kept sending it half the traffic (p95 3.1 s). Least loaded, adaptive and
  balanced moved away and kept p95 near 60 ms.
- **Adaptive had the best tail here** (p99 78 ms): box A's share went to 0% while it was slowed,
  and traffic returned home after it recovered. Its four switches each have a cause in the log
  (one early bounce, back home, `active box down`, back home after the restart).
- Least loaded still sent box A 6% during the slowdown, which shows in its p99 (678 ms). Balanced
  sent it 2%.
- Still 0 failed requests while box A's server was stopped, for every policy.
- For static and round robin, box A took almost no traffic between the end of the slowdown and
  the stop. The cause was not investigated (a backlog from the slowdown is the likely one), so
  their recovery timings in this run should not be quoted.

### Real GPU: fast and slow cores on a DGX Spark (workflow `ecff518b-bf5b-45b4-b80b-763e457bbc5c`)

The same 48 h forecast call, pinned to each core in turn:

| cores | type | time per call |
|---|---|---:|
| 5–9, 15–19 | Cortex-X925 (3.9 GHz) | 9.9 ms |
| 0–4 | Cortex-A725 (2.8 GHz) | 64–67 ms |
| 10–14 | Cortex-A725 (2.8 GHz) | 56–57 ms |

A 4-core Fuzzball job gets cores 0–3, all slow, so the first two-box attempt (`33c887ca`, stopped,
not used) ran every GPU call 6.4× slower. The server now pins itself to the fastest cores it is
allowed and the workflows ask for 10 cores. Any GPU service on these nodes has the same exposure.

### Real GPU: the trained AFNO model (workflow `664e809a-db77-4168-930e-ff062e86c380`)

SPARK28, weights `afno_event.pt` (19.1 M parameters), one 48 h forecast per request.
File: `results/afno/20261114-182522-fuzzball-afno/`.

| server | clients | req/s | p50 ms | p95 ms | GPU call p50 ms |
|---|---:|---:|---:|---:|---:|
| team (`serve/forecast_server.py`) | 1 | 8.3 | 120 | 127 | 66 |
| team | 4 | 14.5 | 270 | 308 | 64 |
| team | 16 | 14.3 | 1100 | 1157 | 64 |
| engine (`inference.server --model afno`) | 1 | 12.9 | 77 | 79 | 67 |
| engine | 4 | 16.5 | 240 | 264 | 166 |
| engine | 16 | 13.5 | 1160 | 1327 | 982 |

- The two servers gave the same answers: 128 forecast points compared, largest difference 0.0 °C,
  0.01 m/s, 0.01 hPa.
- **Batching across requests does not raise AFNO throughput.** Both servers level off near
  14–16 req/s; the GPU is already full with one request (about 8 ms per member-step). At 16
  clients the engine was slightly slower (13.5 vs 14.3 req/s).
- With one client the engine answered in 77 ms against 120 ms, with the same GPU time. That
  difference is time outside the model.

### Real GPU: one box, batch sizes and precision (workflow `4a0aaf44-70ed-4067-a176-0f05beed8d81`)

In `inference/README.md`: 95 → 2,401 req/s at 32 clients with batching; fp16 slower and inaccurate.

### Simulated: adaptive thresholds on mock boxes

`results/thresholds-20261114-182852.json`. Mock boxes, 20 req/s, box A slowed 4× then box B killed
and restarted. No requests failed in any setting.

| margin | window s | cooldown s | switches | p50 ms | p95 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1 | 10 | 2 | 120 | 2451 | 3621 |
| 0.25 | 1 | 10 | 2 | 121 | 2567 | 4416 |
| 0.50 | 1 | 10 | 2 | 145 | 4297 | 5657 |
| 0.25 | 0.25 | 10 | 2 | 122 | 1144 | 2996 |
| 0.25 | 3 | 10 | 2 | 164 | 5523 | 5820 |
| 0.25 | 1 | 2 | 2 | 120 | 2384 | 4237 |
| 0.25 | 1 | 30 | 2 | 119 | 2384 | 4236 |
| 0.10 | 0.25 | 2 | 3 | 115 | 1049 | 3131 |

On the mocks the window matters most: p95 was 1.1 s at 0.25 s, 2.6 s at 1 s and 5.5 s at 3 s,
because requests queue on the slow box until the rule moves. Cooldown changed nothing in this
scenario. One run per setting, so small differences are noise. These are mock numbers: they say
how the rule reacts, not how the GPUs behave (see the oscillation above for that).

## Not done

- Adaptive with a larger margin or an unbatched probe on the real boxes (to confirm the fix for
  the oscillation).
- A load high enough to find the two-box ceiling.
- The AFNO model behind the router on two GPUs (it ran on one).
- The cards were connected to the live cluster boxes through port-forwards and showed both as
  `REAL GPU` (`results/screenshots/`); the page's own traffic and fault buttons were not exercised
  against the cluster.

## JSON API for front ends: `/api/v1`

Served by the Control Room server (`ui/server.py`) next to the page. It adds nothing to the page. Code:
`router_switching/api.py`; tests: `router_switching/tests/test_api_v1.py`.

| endpoint | answer |
|---|---|
| `GET /api/v1` | the endpoint list and where each number comes from |
| `GET /api/v1/status` | `gpus`, `routing`, `performance`, `fuzzball` in one answer |
| `GET /api/v1/gpus`, `GET /api/v1/gpus/{box}` | per box: `state`, `kind`, `cuda`, `node`, `model`, `gpu`, `inference`, `throughput`, `batch`, `requests`, `queue`, `telemetry` |
| `GET /api/v1/routing` | `policy`, `startup_policy`, `preferred`, `adaptive` (routing or not, active box, thresholds, switches), `failovers_total`, per-box share, health score, probe, latency |
| `GET /api/v1/performance?window=10` | router `rps`, `p50_ms`, `p95_ms`, `p99_ms`, `errors`, `failovers` over 1 to 300 s |
| `GET /api/v1/fuzzball?workflow=ID` | workflow status, stages, node of each running stage (from the `fuzzball` CLI) |
| `POST /api/v1/inference` | one real forecast. Body: `lat`, `lon`, `hours_ahead` (48), `step_hours` (6), `ensemble_members` (1), optional `box` to skip the router. Returns `served_by`, `failover`, `policy`, `round_trip_ms` and the server's answer unchanged in `response` |

How to read it:

- `null` means nobody reported the value. It is never a default or an estimate.
- `state` is `ONLINE`, `DOWN` (the box answers but says it is not ok), `OFFLINE` (no telemetry), or Fuzzball's
  own word (`PENDING`, `ERRORED`, `CANCELED`, ...) when the box is silent and Fuzzball says its service is not running.
- `kind` is `real GPU`, `CPU`, `simulated` (a mock server) or `unknown`. A simulated box has no `gpu.utilization`.
- `gpu.utilization` is 0 to 1 and present only when it is an nvidia-smi reading; `gpu.utilization_source` says which.
- `requests.total` is the box's own counter and includes the router's probe (one small forecast every 0.5 s).
- GET answers carry `Access-Control-Allow-Origin: *`, so a page served from another port can fetch them.
- Node names and `/fuzzball` need a workflow ID: start the server with `WX_WORKFLOW=<id>` or pass `?workflow=<id>`.

```bash
curl -s localhost:8080/api/v1/status | python3 -m json.tool
curl -s -XPOST localhost:8080/api/v1/inference -H 'content-type: application/json' \
     -d '{"lat": 41.98, "lon": -87.9, "hours_ahead": 48}'
```

```js
const s = await (await fetch("http://127.0.0.1:8080/api/v1/status")).json();
for (const [name, g] of Object.entries(s.gpus))
  console.log(name, g.node, g.state, g.model.version, g.gpu.utilization, g.inference.request_ms.p95);
```

## Connecting a chat agent (Nemotron or any LLM with function calling)

The API describes itself as tools, so an agent needs no wxrouter code:

| step | call |
|---|---|
| get the tools | `GET /api/v1/tools` → `tools` is in OpenAI function-calling format; pass it as `tools` to your LLM |
| run a tool the LLM asked for | `POST /api/v1/tools/{name}` with the LLM's arguments object as the JSON body → `{"tool", "result"}` |

| tool | does | key |
|---|---|---|
| `get_forecast` | one real forecast on a GPU: who served it, failover, timings, first points | no |
| `get_gpu_status` | each box: state, node, model, GPU readings, latency, counters | no |
| `get_routing_status` | rule, traffic share, healthy boxes, switches, failovers | no |
| `get_performance` | router request rate, latency percentiles, errors | no |
| `set_routing_policy` | change the rule | yes |
| `set_box_fault` | make one box answer 503, or put it back | yes |
| `run_switching_check` | forecasts → fail one box → forecasts (the other box must serve) → restore → forecasts; returns who served each request, errors, failovers, and how fast the router reacted. The box is always restored | yes |

The key: start the server with `WX_API_KEY=<key>` and send it as `X-API-Key: <key>` (or `Authorization: Bearer <key>`).
With no key set, the three changing tools answer only to callers on the server's own machine.

For another machine to reach it, start the server on all interfaces:

```bash
WX_API_KEY=<key> WX_WORKFLOW=<workflow-id> python ui/server.py --host 0.0.0.0 --port 18180 \
    --router http://127.0.0.1:28000 --box box-a=http://127.0.0.1:28001 --box box-b=http://127.0.0.1:28002
```

A complete agent loop, standard library only, is in `examples/nemotron_agent.py`:

```bash
python router_switching/examples/nemotron_agent.py --api http://<wxrouter-host>:18180 --llm http://<your-llm-host>:8090/v1 \
    --key <key> "Run the inference switching check and tell me which GPU served the requests in each step."
```

### Two commands that check it on the running workflow

```bash
router_switching/check_switching.sh            # Nemotron runs the switching check and explains it (about 30 s)
router_switching/check_switching.sh --direct   # the same check without the LLM; prints PASS or FAIL
.venv/bin/python router_switching/e2e_check.py # one question traced through every system, each hop PASS or FAIL
```

`check_switching.sh` first finds the running workflow and repairs what is missing (port-forwards, the local
API server, its key in `~/.wxrouter_api_key`). It starts no workflow and stops none. `e2e_check.py` changes
nothing: it asks Nemotron a forecast question, follows the tool call through the API and the router to a GPU,
confirms it in the router's own request log and the box's own counter, and checks that Nemotron's answer
quotes the values the model returned.

### On Fuzzball, with an address: `deploy_api.sh`

```bash
router_switching/deploy_api.sh            # start it next to the running GPU stack, print its addresses
router_switching/deploy_api.sh --status   # show the running one
```

One CPU-only service (2 cores, 2 GB, no GPU) in its own workflow `wxrouter-api`. It runs `ui/server.py` from the
branch on GitHub and reaches the GPU stack's router, boxes and Nemotron by their cluster names
(`router.<stack-id>.<group-id>.fuzzball`). It ends by itself after 4 hours. If the GPU stack is replaced, stop
this workflow and run the script again.

Ways in, as measured on fb-hack on Nov 14:

| way | address | state |
|---|---|---|
| Fuzzball endpoint (DNS) | `https://endpoint-<id>.endpoints.fb-hack.civilfritz.net/` (subdomain) and a `…/endpoints/accounts/…/api/api-path/` path address; scope `group`: a Fuzzball login in the browser, or `FB-Authorization: Bearer <token>` from `fuzzball workflow endpoints generate-token <endpoint-id>` | the name resolves, but the gateway's certificate is issued by the cluster's private `root-ca`, so curl, Python and browsers do not trust it. Not verified through this address. Needs the cluster's CA certificate from the admins |
| port-forward | `fuzzball workflow port-forward <wxrouter-api id> api 18180:18180`, then `http://127.0.0.1:18180` | works; verified |
| node address | `http://<node>:18180` | not reachable from a laptop on the event Wi-Fi |

`/api/v1/fuzzball` and the `node` fields are empty in this deployment: the container has no `fuzzball` CLI.
