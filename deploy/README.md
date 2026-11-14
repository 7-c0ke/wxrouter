# Deploying wxrouter on Fuzzball

One workflow runs three services on the cluster:

| service | what | port |
|---|---|---|
| `box-a` | forecast server (the mock for now) | 8001 |
| `box-b` | forecast server (the mock for now) | 8002 |
| `router` | wxrouter, `adaptive` with `--preferred box-b`; starts once both boxes are RUNNING | 8000 |

The repo is private, so the code is **embedded in the Fuzzfile** (top-level `files:`) instead of
pulled from GitHub or a registry. No tokens or image secrets are needed. The image is stock
`python:3.12-slim` (arm64 on the DGX Sparks), and each service pip-installs FastAPI, uvicorn,
httpx and pydantic at start, so **the nodes need outbound access to PyPI**.

## Deploy

The CLI is at `~/bin/fuzzball` (v4.3.1). Add it to PATH or use the full path.

```bash
.venv/bin/python deploy/build_fuzzfile.py          # regenerate after any code change
fuzzball workflow validate deploy/wxrouter.fz
fuzzball workflow start -n wxrouter-team14 deploy/wxrouter.fz
```

Generator options: `--policy least_loaded`, `--preferred ""` (no home box), `--exclusive`
(each box takes a whole Spark, so A and B land on different nodes; use it for the real GPU
server), `--box-cores`, `--box-memory`, and `--image`.

## Use it

```bash
fuzzball workflow list --order-by -created
fuzzball workflow describe <workflow-id>
fuzzball workflow log <workflow-id> router -f
fuzzball workflow port-forward <workflow-id> router 8000:8000
curl -s localhost:8000/router/status
curl -s -XPOST localhost:8000/v1/forecast -H 'content-type: application/json' -d '{"lat":37.7,"lon":-122.4,"hours_ahead":48}' -D - -o /dev/null
```

Scene 3 on the cluster: port-forward `box-b` (8002) and set slow mode:
`curl -XPOST localhost:8002/admin/mode -H 'content-type: application/json' -d '{"mode":"slow","slow_factor":2}'`.

Stop it when done, since it holds cluster resources: `fuzzball workflow stop <workflow-id>`.

## Status (2026-11-14)

- `deploy/wxrouter.fz` passes `fuzzball workflow validate`.
- The first deploy (`449d2ce7-…`) failed before any container started. The cluster could not
  reach its image cache: `lookup fuzzball-objectcache on 10.96.0.10:53: no such host`. A plain
  `alpine:3.20` probe job failed the same way, so this is a cluster-side outage, not the
  Fuzzfile. Ask the cluster admins, then rerun `fuzzball workflow start`.
- Not yet verified on the cluster: PyPI egress, and whether services reach each other by the
  short names `box-a` / `box-b`. The router's log prints how each name resolves, which tells
  you right away. If the names don't resolve, switch the backends to the service FQDNs.
