# Forecast API contract v1.0

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
  "contract_version": "1.0",
  "issued_at": "2026-11-14T09:00:00Z",
  "lat": 37.77, "lon": -122.42,
  "hours_ahead": 48, "step_hours": 1,
  "low_confidence": false,
  "forecast": [
    {"lead_hours": 1, "valid_time": "2026-11-14T10:00:00Z",
     "temp_c": 14.2, "wind_ms": 3.1, "precip_mm": 0.0, "pressure_hpa": 1016.4}
  ],
  "members": null,
  "timing": {"queue_ms": 0.4, "compute_ms": 212.0, "total_ms": 212.6}
}
```

- `forecast` holds one point every `step_hours`, up to `hours_ahead`. When `ensemble_members > 1` it is the ensemble mean, and `members` holds each member's points.
- `low_confidence` is `true` when `hours_ahead > 72`.
- `timing` is measured on the server. The router compares it with what the client sees.

Errors: `422` invalid input, `503` overloaded or shutting down (the router retries on the other box), `500` model failure (also retried).

**Safe to retry:** a forecast has no side effects, so the router may send the same request to the other box.

## `GET /health`

Must be cheap and must not touch the GPU. Returns `200` with `{"status": "ok"|"degraded", ...}`, or `503` with `"down"`.

```json
{"status": "ok", "box": "box-a", "model_version": "wx-2026-11-14", "contract_version": "1.0", "uptime_s": 812.3}
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

## Router-added response headers

| header | meaning |
|---|---|
| `X-Served-By` | box that answered, e.g. `box-a` |
| `X-Router-Policy` | `static` / `round_robin` / `least_loaded` / `adaptive` / `balanced` |
| `X-Router-Attempts` | `1`, or `2` if a failover happened |
| `X-Router-Failover` | `true` if the first box failed |
