# Forecast API contract v1.1

The real GPU forecast server and the mock server must both implement this API exactly.
The router exposes the same `POST /v1/forecast`, so clients can't tell whether they are
talking to the router or to a box directly. The pydantic models are in `contract.py`.

## `POST /v1/forecast`

Request:

```json
{
  "lat": 37.77,
  "lon": -122.42,
  "hours_ahead": 48,
  "step_hours": 1,
  "ensemble_members": 1,
  "request_id": "optional-client-id"
}
```

| field | type | rules |
|---|---|---|
| `lat` | float | -90 … 90 |
| `lon` | float | -180 … 180 |
| `hours_ahead` | int | 1 … 120 (most accurate ≤ 72) |
| `step_hours` | int | 1 … 24, default 1 |
| `ensemble_members` | int | 1 … 20, default 1 |
| `request_id` | string? | echoed back |

Response `200`:

```json
{
  "request_id": "optional-client-id",
  "box": "box-a",
  "model_version": "wx-2026-11-14",
  "contract_version": "1.1",
  "issued_at": "2026-11-14T09:00:00Z",
  "lat": 37.77, "lon": -122.42,
  "hours_ahead": 48, "step_hours": 1,
  "low_confidence": false,
  "native_step_hours": 6,
  "forecast": [
    {"lead_hours": 1, "valid_time": "2026-11-14T10:00:00Z",
     "temp_c": 14.2, "wind_ms": 3.1, "pressure_hpa": 1016.4,
     "precip_mm": null, "humidity_pct": 62, "cloud_pct": 40, "interpolated": true},
    {"lead_hours": 6, "valid_time": "2026-11-14T15:00:00Z",
     "temp_c": 16.0, "wind_ms": 3.4, "pressure_hpa": 1016.1,
     "precip_mm": null, "humidity_pct": 55, "cloud_pct": 35, "interpolated": false}
  ],
  "members": null,
  "timing": {"queue_ms": 0.4, "compute_ms": 212.0, "total_ms": 212.6}
}
```

- `forecast` holds one point every `step_hours`, up to `hours_ahead`. When `ensemble_members > 1` it is the ensemble mean, and `members` holds each member's points.
- `low_confidence` is `true` when `hours_ahead > 72`.
- **Model steps:** the model computes every `native_step_hours` (6). Points between steps are straight-line interpolations. A server may mark each point `"interpolated": true/false` (the mock does). The real server (`serve/forecast_server.py`) sends one response-level `interpolated` flag instead, so clients should treat a point as a model step when `lead_hours % native_step_hours == 0`. The UI does this for its 6-hour tables.
- **Optional fields** (may be `null` or missing; clients must handle both): `precip_mm` (rain in the hour ending at `valid_time`), `humidity_pct`, `cloud_pct`. The real AFNO model has none of these: it gives temperature, wind and pressure (plus ensemble `temp_c_min`/`temp_c_max`).
- v1.1 changes from v1.0: `precip_mm` became optional; added `humidity_pct`, `cloud_pct`, `interpolated` and `native_step_hours`. A v1.0 client that reads only the old fields keeps working, as long as it handles a null `precip_mm`.
- `timing` is measured on the server. The router compares it with what the client sees.

Errors: `422` invalid input, `503` overloaded or shutting down (the router retries on the other box), `500` model failure (also retried).

**Safe to retry:** a forecast has no side effects, so the router may send the same request to the other box.

## `GET /health`

Must be cheap and must not touch the GPU. Returns `200` with `{"status": "ok"|"degraded", ...}`, or `503` with `"down"`.

```json
{"status": "ok", "box": "box-a", "model_version": "wx-2026-11-14", "contract_version": "1.1", "uptime_s": 812.3}
```

## `GET /stats`

Must be cheap. The router polls it about every 250 ms.

```json
{
  "box": "box-a",
  "in_flight": 2, "queue_depth": 0, "max_concurrency": 4,
  "gpu_util": 0.55,
  "ewma_latency_ms": 230.1, "p50_latency_ms": 210.0, "p95_latency_ms": 480.0,
  "requests_total": 1532, "errors_total": 0,
  "mode": "normal"
}
```

On the real server, `gpu_util` should come from NVML (`nvidia-smi`). Set `mode` to `"training"` while a fine-tuning job is running.

## Mock-only: `POST /admin/mode`

Used by the load generator to simulate trouble:

```json
{"mode": "slow", "slow_factor": 3.0}
```

| mode | behaviour |
|---|---|
| `normal` | regular delays |
| `slow` | compute × `slow_factor`, and `gpu_util` stays high (like a training job running on the box) |
| `error` | `/v1/forecast` and `/health` return 503 |
| `hang` | `/v1/forecast` never answers (tests the router timeout); `/health` still answers |

Killing the process (connection refused) is done by `bench.py`, which starts and stops the mocks itself.

## Router: `GET /router/metrics?window=10&since=<t>`

Rolling stats for the last `window` seconds, overall and per box (`requests`, `rps`, `errors`, `failovers`, `p50_ms`, `p95_ms`, `p99_ms`), and `now` (router clock in seconds). With `since`, it also returns each request after that time (`t`, `box`, `ms`, `status`, `failover`). The dashboard uses this.

## Router: routing config

`GET /router/config` returns what is set and what may be set:

```json
{"policy": "adaptive", "preferred": "box-b", "drained": [],
 "settings": {"margin": 0.25, "window_s": 1.0, "cooldown_s": 10.0, "return_margin": 0.2, "stall_ms": 1000.0},
 "policies": ["static", "round_robin", "least_loaded", "adaptive", "balanced"],
 "backends": ["box-a", "box-b"],
 "limits": {"margin": {"min": 0, "max": 0.95, "step": 0.05, "unit": "fraction", "help": "..."}, "...": {}},
 "defaults": {"policy": "adaptive", "preferred": "box-b", "margin": 0.25, "...": 0}}
```

`POST /router/config` changes any of these fields; send only the ones to change. It returns `GET /router/status`, which now carries the same `config` block and `drained` per box.

| field | meaning |
|---|---|
| `policy` | the rule |
| `preferred` | home box: adaptive returns to it; static sends everything to it. `""` clears it |
| `margin`, `window_s`, `cooldown_s`, `return_margin` | adaptive switching (ranges in `limits`) |
| `stall_ms` | a probe waiting longer than this counts as a slow box |
| `drain` | `{"box-a": true}`: no new requests to that box under any rule (still the last resort on failover); `false` puts it back |
| `reset` | `true`: back to the start-up settings, then apply the rest of the update |

A value out of range, an unknown rule or an unknown box returns 422 and changes nothing.

### The same operations in both routers

`router.py` (the deployed one, port 8000) also answers the names that `router/router.py` (port 8100, docs/CONTRACT.md section 3) uses, so a script written for either works against `router.py`:

| operation | `router.py` | `router/router.py` |
|---|---|---|
| read settings | `GET /router/config` | (in `GET /router/status`) |
| change rule | `POST /router/config {"policy"}` or `POST /router/policy {"policy"}` | `POST /router/policy` |
| drain a box | `POST /router/config {"drain":{...}}` or `POST /router/drain {"backend","drain"}` | `POST /router/drain` |
| tune adaptive, home box, reset | `POST /router/config` | config file only |
| health | `GET /health`, `GET /router/health` | `GET /router/health` |
| status | `GET /router/status` | `GET /router/status` (other shape) |
| rules | static, round_robin, least_loaded, adaptive, balanced | the same without balanced |
| ensemble split | not yet | `POST /v1/ensemble` |
| box header | `X-Served-By` and `X-Backend` | `X-Backend` |

## Router-added response headers

| header | meaning |
|---|---|
| `X-Served-By` | box that answered, e.g. `box-a` |
| `X-Backend` | the same; the name `router/router.py` uses |
| `X-Router-Retries` | `0`, or `1` after a failover (`router/router.py`'s name) |
| `X-Router-Policy` | `static` / `round_robin` / `least_loaded` / `adaptive` / `balanced` |
| `X-Router-Attempts` | `1`, or `2` if a failover happened |
| `X-Router-Failover` | `true` if the first box failed |
