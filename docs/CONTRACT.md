# The interfaces (so every piece can be built and tested apart)

*Written at the hackathon, Fri Nov 13, 2026 (night). All HTTP is plain JSON over HTTP/1.1. Everything here must run with the Python standard library only (plus PyTorch and PhysicsNeMo for the real weather model), on arm64 Linux (the GB10 boxes) and on a laptop.*

## The picture
```
 clients: the agent (through the MCP weather tools), the load generator, the dashboard
                      |
              ROUTER  :8100     picks a healthy backend per request; policies: static, round_robin, adaptive
              /                 \
   node A                         node B                      (the two GB10 boxes)
   forecast server :8101          forecast server :8101       weather model: a forecast = members x steps of GPU work
   LLM server      :8000          LLM server      :8000       OpenAI-compatible (vLLM or llama.cpp), the teammates' side
   (maybe a training or fine-tuning job on one node: the contention the router must route around)
```

## 1. Forecast server (one per node; `serve/forecast_server.py`)
- `GET /health` → `200 {"status":"ok","node":"A","model":"afno-19m","ready":true}`; `503` while loading, draining or failed.
- `GET /metrics` → `200` JSON:
  ```json
  {"node":"A","ts":1794550000.0,"inflight":1,"queued":3,"served_total":120,"errors_total":0,
   "latency_ms":{"p50":210.0,"p95":900.0,"ewma":260.0},
   "gpu":{"util_pct":96,"temp_c":70,"power_w":60.0,"proc_mem_mb":7115},
   "host":{"mem_available_gb":101.2,"load1":1.9},
   "busy_other":false,"slowdown":1.0}
  ```
  - `busy_other`: another compute process (a training job, say) is on this node's GPU.
  - On a GB10, `nvidia-smi` reports no total GPU memory (unified memory), so `host.mem_available_gb` comes from `/proc/meminfo` `MemAvailable`, and `gpu.proc_mem_mb` sums `nvidia-smi --query-compute-apps`. Any field may be `null` when unknown.
  - `GET /metrics?format=prom` gives the same numbers as Prometheus text.
- `POST /v1/forecast` body:
  ```json
  {"init":"latest","lead_hours":24,"members":1,"point":{"lat":41.9602,"lon":-87.9316},"request_id":"abc"}
  ```
  - `init`: `"latest"` (the newest analysis the server holds), a time such as `"2019-01-27T06"` (from the server's packed data, if present), or `{"npy_b64":"..."}` (a float16 z-score state of shape (20,120,240) as base64 of `.npy` bytes: the laptop fetches NOAA's analysis and sends it, so nodes need no internet).
  - `lead_hours`: 6 to 120, a multiple of 6. `members`: 1 to 40 (member 0 unperturbed; the others add smoothed noise at 0.05 of each channel's standard deviation).
  - Cost: `members x lead_hours/6` forward passes. That is the load knob.
  - Response:
  ```json
  {"request_id":"abc","node":"A","model":"afno-19m","init_time":"2026-11-13T00","members":1,
   "leads_h":[6,12,18,24],
   "point":{"lat":41.9602,"lon":-87.9316,"t2m_c":[3.9,3.5,3.2,3.15],"t2m_c_min":[...],"t2m_c_max":[...]},
   "timing_ms":{"queue":0.4,"gpu":212.0,"total":214.1}}
  ```
  - Errors: `400` bad request, `503` not ready, `500` failure; body `{"error":"..."}`.
- **Mock mode** (`--mock`, no GPU, no PyTorch): the same API. Work is emulated by one worker thread sleeping `units x ms_per_unit x slowdown`, so requests queue as on one GPU. Test hooks, enabled only with `--admin`:
  - `POST /admin/contention {"factor":1.8}`: as if a training job shared the GPU (measured on a GB10: training beside an agent slows 41–47%); sets `busy_other` and `slowdown`.
  - `POST /admin/fail {"mode":"down"|"hang"|"error"|"ok"}`: refuse connections' work (503), hang, return 500, or recover.

## 2. LLM servers (the teammates' side)
OpenAI-compatible: `POST /v1/chat/completions` (maybe streamed), `GET /v1/models`, plus the engine's `GET /health` and `GET /metrics` (vLLM and llama.cpp both expose Prometheus text). The router needs nothing more.

## 3. Router (`router/router.py`, port 8100)
- **Pools.** `forecast` (paths under `/v1/forecast`) and `llm` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`). Each pool lists backends: `{"name":"A","url":"http://nodeA:8101","probe":"forecast"}`; probe kinds: `forecast` (JSON `/metrics`), `vllm`, `llamacpp` (Prometheus `/metrics`), `http` (only `/health`).
- **What it tracks per backend:** health (active probes every `probe_s`, plus passive errors), its own in-flight count and latency EWMA, and the probe's numbers (queue, GPU use, `busy_other`, free memory).
- **Policies**, switchable at run time:
  - `static`: always the first healthy backend in the list;
  - `round_robin`: rotate over healthy backends;
  - `adaptive`: lowest load score wins, but the preferred backend changes only when another beats it by a margin (hysteresis) for at least `dwell_s`, and not within `cooldown_s` of the last switch. An unhealthy backend is left at once.
- **Failover.** If a backend fails before any response byte was sent, retry once on another healthy backend. Mark a backend unhealthy after `fail_n` straight failures; probe it back after `recover_s`.
- **Headers on every proxied response:** `X-Backend: A`, `X-Router-Policy: adaptive`, `X-Router-Retries: 0`.
- `GET /router/status` → policy, per-pool preferred backend, per-backend state and score, counters (routed, failovers, switches), and the last 50 decisions with reasons. CORS open (`Access-Control-Allow-Origin: *`) so a dashboard can poll it.
- `POST /router/policy {"policy":"adaptive"}`; `GET /router/health`.
- Decisions are also appended to a JSON-lines file.

## 4. Load generator (`bench/loadgen.py`)
Runs a named scenario against the router (or straight at a backend) and writes one JSON line per request (`t_start`, `latency_ms`, `status`, `backend`, `retries`, `kind`) plus a summary (throughput, p50/p95/p99, error rate, share per backend, time to recover). Seeded, so each policy sees the same arrivals. Scenarios: `steady`, `contention` (a training job appears on node B a third of the way in), `failure` (node B dies halfway), `recovery`.

## 5. MCP weather tools (`mcp/weather_mcp.py`)
A Model Context Protocol server over stdio (JSON-RPC 2.0): `forecast_point`, `ohare_tomorrow`, `ensemble_range`, `storm_2019`, `cluster_status`, `set_routing_policy`. Each calls the router. Any MCP client (Claude Code, Codex, an open model's agent loop) can use it.

## 6. Ports and environment
`ROUTER_URL` (default `http://127.0.0.1:8100`); forecast server `--port 8101`; `WX_DATA` (the data folder); `WX_WEIGHTS` (the model file).
