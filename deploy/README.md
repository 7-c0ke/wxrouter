# Deploying wxrouter on Fuzzball

## The deploy script: `deploy/fb.py`

One script over the fuzzball CLI. Run it with no arguments for a menu, or name an action:

```bash
python deploy/fb.py                 # menu: pick an action
python deploy/fb.py check           # CLI found? logged in? API reachable? volume there?
python deploy/fb.py setup           # once: repo, packages, AFNO model, llama.cpp, Nemotron (18.9 GB), NOAA year
python deploy/fb.py up              # validate, start fuzzball/30-real-stack.fz, wait until every service runs
python deploy/fb.py open            # Control Room at http://localhost:8080 (port-forward)
python deploy/fb.py logs box-a -f   # follow a service's log
python deploy/fb.py redeploy        # stop, then start again (prep pulls the latest main)
python deploy/fb.py down            # stop the workflow and free both GPUs
```

- `--target real` (default) runs the real stack: real AFNO model and Nemotron on both boxes, the router and
  the Control Room. It needs `setup` first. `--target mock` runs `deploy/wxrouter.fz` (regenerated from this
  checkout): mock boxes, router and Control Room, with nothing needed on the volume.
- `--dry-run` on any action prints the fuzzball commands instead of running them.
- The CLI is found from `$FUZZBALL`, then `PATH`, then `~/bin/fuzzball`. The deployed workflow's id is kept in
  `.fb_deploy.json` (git-ignored). `--id <workflow id>` acts on another one.
- `up` refuses to start a second copy of a running workflow; use `redeploy` or `up --replace`.
- Known CLI errors get a one-line explanation: API refused (network path or outage), image cache down,
  login expired.

There is no config step after the servers start. All services run in one workflow and reach each other by
short name, so the Control Room's addresses are fixed in the Fuzzfile.

---

## The generated mock workflow (`--target mock`)

One workflow runs four services on the cluster:

| service | what | port |
|---|---|---|
| `box-a` | forecast server (the mock for now) | 8001 |
| `box-b` | forecast server (the mock for now) | 8002 |
| `router` | wxrouter, `adaptive` with `--preferred box-b`; starts once both boxes are RUNNING | 8000 |
| `ui` | Control Room dashboard + forecast assistant; starts once the router is RUNNING; published as endpoint `control-room` | 8080 |

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

Generator options: `--llm-base-url http://<llm-host>:8080/v1` (assistant uses an LLM such as Nemotron; without it, the rule-based mode), `--no-ui`, `--no-ui-endpoint`, `--policy least_loaded`, `--preferred ""` (no home box), `--exclusive`
(each box takes a whole Spark, so A and B land on different nodes; use it for the real GPU
server), `--box-cores`, `--box-memory`, and `--image`.

## Open the Control Room

Either through the Fuzzball endpoint, or by port-forwarding (works whatever the endpoint settings are):

```bash
fuzzball endpoint list                                   # URL of `control-room` (scope: group)
fuzzball workflow port-forward <workflow-id> ui 8080:8080   # then open http://localhost:8080
```

The page uses relative URLs, so it works under the endpoint's `/endpoints/<id>/` path. Open it
with the trailing slash. The demo buttons (Training, Fail) work because the boxes run the mock.

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

- **Deployed and worked** as workflow `f81cfa72-2973-478a-b1bb-2b5fbb8cafa4` (`wxrouter-team14`).
  All four services and the `hold` job ran, and PyPI installs worked.
- Services reach each other by short name: the router's log shows `box-a` / `box-b` resolving to
  `<name>.<workflow-id>.<group-id>.fuzzball`.
- Tested through `port-forward` of the `ui` service. Chat forecasts were served by box-b (188 ms).
  In scene 3, training on box-b made adaptive switch to box-a (69 ms vs 1001 ms compute) with 0
  errors.
- **A workflow with only services finishes right away.** Fuzzball ends a workflow when its jobs
  end and stops the services with it. The `hold` job (`--hours`, default 4) keeps it up. Stop early
  with `fuzzball workflow stop <id>`.
- Right after that test, the Fuzzball API started refusing connections
  (`dial tcp 172.16.0.133:443: connection refused`), so the workflow's current state is unknown.
  Box-b may still be in slow mode from the demo, and demo traffic may still be on. When the API is
  back, check `fuzzball workflow describe <id>`, then reset in the UI or stop it.
- Not yet tried: opening the UI through the `control-room` endpoint URL (vs. port-forward),
  `--exclusive` placement, and the LLM mode on the cluster.
- Earlier the same day, an image-cache outage on the cluster
  (`lookup fuzzball-objectcache ... no such host`) blocked every workflow, then recovered.
