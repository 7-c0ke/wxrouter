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

- `up` runs the real stack: real AFNO model and Nemotron on both boxes, the router and the Control Room.
  It needs `setup` first. There is no mock target: what is deployed is always live. Mock boxes are only
  for laptop development (`dev_stack.py`, `ui/dev_stack_supervised.py`) and tests.
- `--dry-run` on any action prints the fuzzball commands instead of running them.
- The CLI is found from `$FUZZBALL`, then `PATH`, then `~/bin/fuzzball`. The deployed workflow's id is kept in
  `.fb_deploy.json` (git-ignored). `--id <workflow id>` acts on another one.
- `up` refuses to start a second copy of a running workflow; use `redeploy` or `up --replace`.
- Known CLI errors get a one-line explanation: API refused (network path or outage), image cache down,
  login expired.

There is no config step after the servers start. All services run in one workflow and reach each other by
short name, so the Control Room's addresses are fixed in the Fuzzfile.

---

## Open the Control Room

Either through the Fuzzball endpoint, or by port-forwarding (works whatever the endpoint settings are):

```bash
fuzzball endpoint list                                   # URL of `control-room` (scope: group)
fuzzball workflow port-forward <workflow-id> ui 8080:8080   # then open http://localhost:8080
```

The page uses relative URLs, so it works under the endpoint's `/endpoints/<id>/` path. Open it
with the trailing slash. On the real boxes each card shows the real buttons only (Fine-tune, Kill server,
Drain); the simulated Slow / Fail buttons appear only on mock boxes during laptop development.

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

*This is the status of the first deployment, with mock boxes, on Nov 14 in the morning; it is kept for the
record. The stack that ran afterwards is `fuzzball/30-real-stack.fz` (`python deploy/fb.py up`), and what it
did on the event boxes is in [results/event-boxes-2026-11-14.md](../results/event-boxes-2026-11-14.md).*

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
