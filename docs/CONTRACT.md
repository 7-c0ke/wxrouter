# The interfaces (so every piece can be built and tested apart)

*Written at the hackathon, Fri Nov 13, 2026 (night). All HTTP is plain JSON over HTTP/1.1. Everything except the real weather model runs on the Python standard library alone (Python 3.9 to 3.13), on arm64 Linux (the GB10 boxes) and on a laptop. The real weather model also needs PyTorch, NumPy and NVIDIA PhysicsNeMo.*

## The picture
```
 clients: the agent (LLM + weather tools over MCP), the load generator, the dashboard
                      |
              ROUTER  :8100     one address for both boxes; picks a healthy backend per request
              /                 \       policies: static, round_robin, least_loaded, adaptive
   node A                         node B                      (the two GB10 boxes)
   forecast server :8101          forecast server :8101       weather model: a forecast = members x steps of GPU work
   LLM server      :8000          LLM server      :8000       OpenAI-compatible (llama.cpp or vLLM)
   (maybe a training job on one node: the contention the router must route around)
```
Local testing uses mocks on one machine: forecast mocks on 8101 and 8102, LLM mocks on 8001 and 8002, router on 8100.

## 1. Forecast server (one per node; `serve/forecast_server.py`)
Every response carries `Access-Control-Allow-Origin: *` (the dashboard reads the servers directly), and `OPTIONS` is answered for preflight.

- `GET /health` → `200 {"status":"ok","node":"A","model":"afno-19m","ready":true}`; `503` (same shape, `"ready":false`, a `"status"` saying why) while loading, draining or failed.
- `GET /v1/info` → `{"node","model","params","device","mock":false,"latest":"2026-11-14T00"|null,"ranges":[{"first":"2019-01-20T00","last":"2019-02-03T18"}],"ms_per_unit":10.2}`.
- `GET /metrics` → `200` JSON:
  ```json
  {"node":"A","ts":1794550000.0,"inflight":1,"queued":3,"queued_units":120,"served_total":120,"errors_total":0,
   "units_total":5400,"ms_per_unit":10.2,
   "latency_ms":{"p50":210.0,"p95":900.0,"ewma":260.0},
   "gpu":{"util_pct":96,"temp_c":70,"power_w":60.0,"proc_mem_mb":7115},
   "host":{"mem_available_gb":101.2,"load1":1.9},
   "busy_other":false,"slowdown":1.0}
  ```
  - A **unit** is one member advanced one 6-hour step. `queued_units` counts the work waiting or running; `ms_per_unit` is a running average of GPU time per unit.
  - `busy_other`: another compute process (a training job, say) is on this node's GPU. `slowdown`: current `ms_per_unit` over the best the server has seen (1.0 = undisturbed).
  - On a GB10, `nvidia-smi` reports no total GPU memory (unified memory), so `host.mem_available_gb` comes from `/proc/meminfo` `MemAvailable`, and `gpu.proc_mem_mb` sums `nvidia-smi --query-compute-apps`. Any field may be `null` when unknown; no reader may crash on a `null`.
  - `GET /metrics?format=prom` gives the same numbers as Prometheus text (names prefixed `wx_`).
- `POST /v1/forecast` body:
  ```json
  {"init":"latest","lead_hours":24,"members":1,"member_offset":0,
   "point":{"lat":41.9602,"lon":-87.9316},"with_truth":false,"request_id":"abc"}
  ```
  - `init`: `"latest"` (the newest analysis the server holds), a time such as `"2019-01-27T12"` (from the data the server holds), or `{"npy_b64":"...","time":"2026-11-14T00"}` (a float16 z-score state of shape (20,120,240), base64 of `.npy` bytes: a laptop fetches NOAA's analysis and sends it, so nodes need no internet).
  - `lead_hours`: 6 to 120, a multiple of 6. `members`: 1 to 40. `member_offset` (default 0): this request computes global members `member_offset … member_offset+members-1`. Global member 0 is the unperturbed forecast; member m > 0 adds smoothed noise (0.05 standard deviations per channel) drawn from a seed that depends only on the init time and m, so the same member is the same on every node and an ensemble can be split across nodes.
  - Cost: `members x lead_hours/6` units. That is the load knob.
  - `point`: latitude −90…90, longitude −180…360; values are interpolated bilinearly on the 1.5° grid.
  - Response:
  ```json
  {"request_id":"abc","node":"A","model":"afno-19m","init_time":"2026-11-13T00","members":1,"member_offset":0,
   "leads_h":[6,12,18,24],"valid_times":["2026-11-13T06","2026-11-13T12","2026-11-13T18","2026-11-14T00"],
   "point":{"lat":41.9602,"lon":-87.9316,
            "t2m_c":[3.9,3.5,3.2,3.15],        "comment: member 0's series, or null if member 0 is not in this request": 0,
            "t2m_c_mean":[...],"t2m_c_min":[...],"t2m_c_max":[...],
            "wind10_ms_mean":[...],"msl_hpa_mean":[...],
            "truth_t2m_c":[...]},             "comment: only with with_truth, null where the server has no data": 0,
   "units":4,"timing_ms":{"queue":0.4,"gpu":212.0,"total":214.1}}
  ```
  (The `"comment…"` keys above are notes, not fields.)
  - Errors: `400` bad request, `404` init time not held, `503` not ready, `500` failure; body `{"error":"..."}`.
- `POST /v1/state {"time":"2026-11-14T00","npy_b64":"..."}` stores an analysis; it becomes `"latest"` if it is the newest. `200 {"latest":"2026-11-14T00"}`.
- **Mock mode** (`--mock`: no GPU, no PyTorch, no NumPy): the same API with made-up but smooth and repeatable numbers. Work is emulated by one worker thread sleeping `units x ms_per_unit x slowdown` (default 10 ms per unit, as measured on a GB10), so requests queue as on one GPU. Test hooks, enabled only with `--admin`:
  - `POST /admin/contention {"factor":1.8}`: as if a training job shared the GPU (measured on a GB10: training beside an agent slows 41–47%); sets `busy_other` and `slowdown`. `{"factor":1.0}` clears it.
  - `POST /admin/fail {"mode":"down"|"hang"|"error"|"ok"}`: `down` = every request and `/health` get 503; `hang` = accept and never answer; `error` = work requests get 500 while `/health` stays 200; `ok` = recover.
  - `POST /admin/reset` clears counters and both hooks.

## 2. LLM servers (the teammates' side) and the LLM mock (`serve/mock_llm.py`)
OpenAI-compatible: `POST /v1/chat/completions` (JSON, or server-sent events when `"stream":true`), `GET /v1/models`, plus the engine's `GET /health` and `GET /metrics` (llama.cpp's server and vLLM both expose Prometheus text). The router needs nothing more.

The mock speaks the same API with no model: `--flavor llamacpp|vllm` chooses whose metric names `/metrics` prints, `--tps` the emulated tokens per second, `--slots` how many requests run at once. It emulates a prompt cache (a prompt that extends the last one served by a slot is prefilled faster), so cache-aware routing can be tested. It has the same `--admin` hooks as the forecast mock. With `tools` in the request it can play a scripted tool-calling model (first reply: one tool call; after a tool result: a final answer that quotes it), so the agent runs end to end with no GPU.

## 3. Router (`router/router.py`, port 8100)
- **Pools.** `forecast` (paths `/v1/forecast`, `/v1/ensemble`, `/v1/state`, `/v1/info`) and `llm` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`). Each pool lists backends: `{"name":"A","url":"http://nodeA:8101","probe":"forecast"}`; probe kinds: `forecast` (JSON `/metrics`), `vllm`, `llamacpp` (Prometheus `/metrics` + `/health`), `http` (only `/health`).
- **What it tracks per backend:** health (active probes every `probe_s`, plus passive errors), its own in-flight count and latency EWMA, and the probe's numbers (queue, GPU use, `busy_other`, `slowdown`, free memory).
- **Load score** (lower is better), one formula for every policy that needs it: an estimate of how long a new request would wait, `(inflight + queued + 1) x latency_ewma x slowdown`, plus configurable penalties (GPU busy with another process, little free memory). The weights live in the config file.
- **Policies**, switchable at run time:
  - `static`: always the first healthy backend in the list;
  - `round_robin`: rotate over healthy backends;
  - `least_loaded`: lowest score for every request, no memory of the last choice (it can flap);
  - `adaptive` (the switcher): requests go to a *preferred* backend. The preference changes only when another backend beats it by a margin (`hysteresis`, e.g. 25%) for at least `dwell_s`, and not within `cooldown_s` of the last switch. An unhealthy backend is left at once. While the preferred backend has more than `spill_inflight` requests in flight, the overflow goes to the next-best healthy backend.
- **Session affinity for the `llm` pool** (cache-aware routing, optional): a request with an `X-Session-Id` header (or a `user` field) sticks to one backend, chosen by rendezvous hashing over healthy backends, so the engine's prompt cache is reused. Affinity is broken when that backend is unhealthy or its score exceeds the best by `affinity_break` (e.g. 3x). Response header `X-Router-Affinity: hit|miss|none`.
- **Failover.** If a backend fails before any response byte was sent to the client (refused, timed out, 502/503/504), retry once on another healthy backend. Mark a backend unhealthy after `fail_n` straight failures; probe it back after `recover_s`. Streaming responses are passed through as they arrive, never buffered.
- **Self-healing.** A backend may carry a `heal` action in the config (a command line, or an HTTP call). When the backend has been unhealthy for `heal_after_s`, the router runs it, with back-off and a cap on attempts, and records it. Actions come only from the config file, never from a request.
- **Ensemble fan-out:** `POST /v1/ensemble` takes the body of `/v1/forecast`. The router splits `members` across the healthy forecast backends (in proportion to their free capacity), sends the shards at the same time with different `member_offset`s, re-runs a failed shard on another backend, and merges: `t2m_c` from the shard holding member 0, means weighted by member counts, min of mins, max of maxes. The response adds `"shards":[{"backend":"A","members":20,"member_offset":0,"ms":2100,"retried":false}, ...]`.
- **Headers on every proxied response:** `X-Backend: A`, `X-Router-Policy: adaptive`, `X-Router-Retries: 0`.
- `GET /router/status` → policy, per-pool preferred backend, per-backend state and score, counters (routed, failovers, switches, heals), and the last 50 decisions with reasons. `GET /router/health`. `POST /router/policy {"policy":"adaptive"}`. `POST /router/drain {"backend":"B","drain":true}` (take a node out for maintenance or training; `false` puts it back). CORS open on all `/router/*`.
- `GET /` serves `dashboard/index.html` if it is there.
- Decisions are also appended to a JSON-lines file.

## 4. Load generator and comparison (`bench/loadgen.py`, `bench/compare.py`)
`loadgen.py` runs a named scenario against the router (or straight at a backend) and writes one JSON line per request (`t_start`, `latency_ms`, `status`, `backend`, `retries`, `kind`, `units`) plus a summary (throughput, p50/p95/p99, error rate, share per backend, time to recover after a failure). Arrivals are open-loop (they do not wait for answers) and seeded, so each policy sees the same requests at the same times. Scenarios: `steady`, `contention` (a training job appears on node B a third of the way in and leaves at two thirds), `failure` (node B dies halfway), `recovery` (node B dies, then comes back), `burst`. Timed events are HTTP calls (the mocks' admin hooks) or commands.

`compare.py` runs every policy through every scenario and writes a results folder: `summary.json`, `RESULTS.md` (the table) and one SVG chart per scenario.

`tools/dev_cluster.py up` starts two forecast mocks, two LLM mocks and the router on one machine and prints the addresses; `down` stops exactly the processes it started.

## 5. The agent's tools (`agent/weather_mcp.py`, `agent/desk_agent.py`)
`weather_mcp.py` is a Model Context Protocol server over stdio (JSON-RPC 2.0, one JSON message per line): `forecast_point`, `ohare_tomorrow`, `ensemble_range`, `storm_2019`, `cluster_status`, `set_routing_policy`. Each calls the router. Any MCP client (Claude Code, Codex, an open model's agent loop) can use it. `desk_agent.py` is a small tool loop for an open model behind the router's `llm` pool; code checks that every number in its answer came from a tool.

## 6. Ports and environment
`ROUTER_URL` (default `http://127.0.0.1:8100`); forecast server `--port 8101`; `WX_DATA` (the data folder: `stats.json`, optional `bundle.json` and state files, optional full packed files); `WX_WEIGHTS` (the model file, as written by `weather/train.py`: a `torch.save`d dict whose `"model"` entry is the AFNO's `state_dict`).
