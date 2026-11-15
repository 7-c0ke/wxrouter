# fuzzball/: the whole system as one Fuzzball workflow

`30-real-stack.fz` is one Fuzzball workflow. It starts 2 jobs (`prep`, `hold`) and 5 services (`box-a`,
`box-b`, `router`, `router2`, `ui`): the weather model and the NVIDIA LLM on each box's GPU, two routers in
front of them, and the web page (the Control Room). It ran on the event's two GB10 boxes (Fuzzball nodes
CIQ-CHI-SPARK27 and CIQ-CHI-SPARK28) on Nov 14, 2026, under Fuzzball 4.3.1. The proof is
[`results/event-boxes-2026-11-14.md`](../results/event-boxes-2026-11-14.md): it names five starts of this file
by workflow id (lines 3 to 18 and 459 to 460), and its measurements were taken inside those stacks. The file
grew during the day. Its present revision (6) was started at 4:54 PM and at 5:10 PM (lines 442 and 459 there).

## The workflow

```
prep ──FINISHED──> box-a ─┬─RUNNING──> router  ─┬─RUNNING──> ui ──RUNNING──> hold
                   box-b ─┘            router2 ─┴─RUNNING──────────────────> hold
```

An arrow is a `depends-on` line: the stage on the right waits until the stage on the left has that status.

| Stage | Kind | Image | What it does | Line |
|---|---|---|---|---|
| `prep` | job | `nvcr.io/nvidia/pytorch:26.02-py3` | `git pull` of the team repo on the volume; stops if this copy of the file is older than the repo's; checks that the model files are there; fetches the newest NOAA analysis | 59 |
| `box-a`, `box-b` | service, 1 GPU each | the same NVIDIA image | Our supervisor (`serve/node_stack.py`). It keeps the forecast server and `llama-server` with the LLM alive on the box's GPU, and starts a fine-tune when asked | 151, 194 |
| `router` | service | `python:3.12-slim` | `router.py` in front of the two forecast servers | 237 |
| `router2` | service | `python:3.12-slim` | `router/router.py`: chat to either box's LLM, and one forecast split across both boxes | 272 |
| `ui` | service | `python:3.12-slim` | The Control Room, behind the endpoint `control-room` | 309 |
| `hold` | job | `rockylinux:9` | Sleeps 8 hours, so that Fuzzball keeps the services running: a workflow with only services ends at once (`deploy/README.md:75-77`) | 125 |

## Fuzzball features in use

From here on, a line number with no file name is in `30-real-stack.fz`, and "results" is
`results/event-boxes-2026-11-14.md`.

| Fuzzball feature | What we use it for | Where |
|---|---|---|
| Workflow format `v4` | The stack and every one-time job are plain workflow files | 51; `quick/*.fz` |
| Jobs and services in one workflow | A short job first, five long-running servers, and a job that keeps them up | 59, 125, 151 to 309 |
| `depends-on` with `FINISHED` | A box starts only after `prep` ended well | 157-159, 200-202 |
| `depends-on` with `RUNNING` | The routers wait for both boxes, the page for both routers, `hold` for the page and `router2` | 243-247, 278-282, 318-323, 139-143 |
| `devices: nvidia.com/gpu: 1` | One GPU for each box service. Each node has one GPU, so the two services land on different nodes | 180-181, 223-224 |
| Persistent volume, seen by both nodes | Code, Python packages, the weather model, the 18.9 GB LLM file, NOAA data, logs and results exist once, at `/wx` in every stage but `hold`. The LLM loaded from it in 13.9 s on event box-a (results, line 79) | 53-56; mounts at 62-64 and in every service |
| Service names as addresses | `http://box-a:8001`, `http://router:8000`, `http://router2:8100`. Nothing is configured after the start | 259-260, 293-295, 333-336 |
| `network.ports` | Each service declares the ports it serves | 182-192, 225-235, 266-270, 303-307, 342-346 |
| `network.endpoints` (type `path`, scope `group`) | The Connect button on the workflow's page opens the Control Room for the team's group. The page serves under the endpoint's path | 347-352; `ui/server.py:842` |
| `readiness-probe` (`http-get`) | Connect becomes active only once the page answers | 353-361 |
| `policy.timeout.execute` | 10 minutes for `prep`, 9 hours for `hold`. Every job in `quick/` carries one too | 121-123, 132-134 |
| OCI images (`docker://`) | NVIDIA's NGC PyTorch image for every stage that uses the GPU or the model, `rockylinux:9` for `hold` and small jobs, `python:3.12-slim` for the routers and the page. We built no image | 61, 153, 196; 127; 239, 274, 314 |
| `fuzzball workflow exec` | The benchmark scripts run inside the running stack, in `box-a`, with one command each | 39-49; results, lines 89, 122, 197 |
| The CLI, wrapped in one script | `deploy/fb.py` calls `version`, `workflow list`, `volume list`, `validate`, `start -n`, `describe` (polled until the services run), `events`, `port-forward`, `log` and `stop`. It turns three CLI errors we met into one plain line each | `deploy/fb.py:157-285`, `95-104` |
| `workflow events -o json` | The Control Room's Events card shows the workflow's own stage events, when the page runs beside a logged-in CLI (a laptop). Inside the deployed stack it is off: the `ui` image has no CLI, and the check of 4:55 PM printed "fuzzball off" (`results/event-boxes-2026-11-14/restart-check-1655.txt`, line 6) | `ui/events.py:204-262` |
| `node list`, `node show`, `workflow list`, `context show` (`-o json`) | The Control Room's view of the team's nodes: is each GPU taken, by which workflow, and the node's health. Read-only. The same limit: beside a logged-in CLI only, off inside the deployed stack | `ui/fuzzball_gpu.py:1-17`, `141-155` |
| Inline `files:` with `file://` mounts | The generated workflows of the second inference engine carry their Python sources inside the workflow file, so a job needs no access to the repository (private when they were written). Workflows made this way ran on the event boxes (`inference/README.md:10-12`) | `deploy/inference-serve.fz:3`, `3493-3494`; `deploy/build_inference_fuzzfile.py:3-4` |

The two Control Room views are tested against real CLI output captured on the event cluster
(`tests/test_events.py:7`, `tests/test_fuzzball_gpu.py:5`). Not used in any file that ran: ingress, egress,
secrets, retries, task arrays, multinode jobs, image builds.

## How to run it

The volume `wxdesk` on the team's provisioner must exist first (`quick/03-share-test.fz`, line 3).

```bash
python deploy/fb.py check       # CLI found? logged in? API reachable? volume there?
python deploy/fb.py setup       # once: quick/05, quick/04 and quick/06 fill the volume
python deploy/fb.py up          # validate, start 30-real-stack.fz, wait for box-a, box-b, router and ui
python deploy/fb.py open        # port-forward the Control Room to http://localhost:8080
python deploy/fb.py down        # stop the workflow; both GPUs are free again

# the same with the plain CLI, from the repo root (header of 30-real-stack.fz, lines 21 to 26 and 40)
fuzzball workflow validate fuzzball/30-real-stack.fz
fuzzball workflow start -n real-stack fuzzball/30-real-stack.fz
fuzzball workflow log <id> box-a -f
fuzzball workflow port-forward <id> ui 8080:8080     # or the Connect button on the workflow's page
fuzzball workflow exec <id> box-a -- bash -c "cd /wx/repo && python3 bench/preflight.py" < /dev/null
fuzzball workflow stop <id>
```

`--dry-run` on any `fb.py` action prints the CLI commands and runs nothing. After a stop, wait about 45 s
before the next start (finding 5 below). `fb.py redeploy` does not wait between the two, so for a restart run
`down`, wait, then `up`.

## What we learned about Fuzzball 4.3.1, and what we did about it

Each finding names its source. Where that is the results file, it was measured on the event cluster on
Nov 14, one time unless two are named. Where it is a comment in the stack file or the code, no result file
was kept for it.

1. **A job ends after one hour unless it carries `policy.timeout.execute`.** A stack ended as "Errored"
   ("Job timed out - exceeded maximum runtime") 60 minutes after its `hold` job started (results, lines
   432-435). `hold` now carries `execute: 9h` (lines 129-134); a service takes no such setting. **Not yet
   confirmed:** we have not seen a stack live past one hour with it. The limit was in the stored definition
   of the last two stacks (`restart-check-1655.txt`, line 22, and `redeploy-check-1711.txt`, line 6, in
   `results/event-boxes-2026-11-14/`); the hour had not passed when the results were written (results,
   475-476).
2. **A service starts only if a job depends on it, directly or through another service.** `router2` showed
   "Finished" after 0 s with an empty log while the rest ran. A one-minute, CPU-only workflow with two
   identical services proved the rule: the one that the job named started, the other did not (results,
   423-429; `results/event-boxes-2026-11-14/scripts/svc-rule-test.fz`). So `hold` names `ui` and `router2`
   (lines 135-143).
3. **One GPU goes to one job or service.** So one service per box holds the GPU, and our supervisor runs the
   forecast server, the LLM and a fine-tune inside it (lines 14-15). On one event box the LLM used 18.7 GB
   of GPU memory beside 0.4 GB for the forecast server (results, line 80).
4. **The queue hands over cleanly.** A second copy of the stack, queued behind the first, took both GPUs 23 s
   after the first ended, with no clash over the ports (results, lines 436-438). Its first job does run at
   once, under the stack that is still serving, so `prep` never changes the model link (lines 103-106).
5. **A start right after a stop meets ports that are still held.** A stack that was queued in front of a
   stop failed at once: "could not use tcp port 8002: already used". A start 48 s after that stop worked, and
   so did one 45 s after a later stop (results, lines 439-441 and 459-460).
6. **Every part was up 36 s after a start**, in both of those starts (results, lines 441 and 460).
7. **The Connect button opens the first endpoint of the first running service, in file order.** So the file
   has one endpoint, on `ui` (lines 310-312), and the page serves under the endpoint's path (lines 329-332).
8. **`fuzzball workflow events` returns only the oldest 100 events without `-n`.** The page asks for the
   newest 1,000 each time and drops the ones it has seen (`ui/events.py:210`, `250-252`).
9. **The web editor keeps an old draft.** The commands come from the copy that was started, the code from
   the volume, so an old copy would run old commands with new code. The file carries a revision number;
   `prep` and the supervisor stop an older copy in the first seconds (lines 71-85, 165, 208).

## The one-time jobs in `quick/`

All six ran on the event cluster on Nov 14. The first three have no result file here; their header comments
give what Fuzzball's `workflow describe` showed.

| File | What it does | What ran |
|---|---|---|
| `01-hello-gpu.fz` | Two 1-GPU jobs at the same time, one per box, on `rockylinux:9` | Both finished in 47 s (its line 2) |
| `02-pull-pytorch.fz` | Pulls NVIDIA's PyTorch image (9.3 GB as Fuzzball reports it) and checks that PyTorch sees the GPU | Two starts: the first ended in an error after the image step, the second finished in 35 s (its lines 2-4) |
| `03-share-test.fz` | Two 16-core jobs, too big to share a box, write and read the same volume | Both finished, 1 min 2 s each. The log was not saved, so we give no speed (its lines 4-6) |
| `04-fetch-llm.fz` | Downloads the 18.9 GB LLM file to the volume | The stack loads that file (results, line 79) |
| `05-setup-volume.fz` | Code, Python packages, the weather model and llama.cpp onto the volume | The stack runs from them (`30-real-stack.fz`, lines 17-19) |
| `06-gfs-year.fz` | One year of NOAA analyses for the fine-tune | 1.68 GB on the volume (`results/event-boxes-2026-11-14/preflight-1546.txt`, line 11) |

`setup_volume.sh` and `fetch_assets.sh` in this folder are the scripts that `04` and `05` fetch and run.

## The one-GPU fallback

`31-one-gpu-stack.fz` is the same stack with `box-a` only: one GPU, one router, the page, and a 30-minute
`hold` (its lines 1-2 and 54). A teammate wrote it for times when two GPUs were slow to place. This repository
holds no record of a run of it, so we claim none. A test keeps its revision number equal to the main stack's
(`tests/test_node_stack.py:1073`); `31-one-gpu-stack.yaml` is an identical copy. Its router declares an
endpoint above the page's (its lines 134-137), so by finding 7 Connect would open the router there.

## The first night's plan

[`first-plan/`](first-plan/README.md) holds eight workflow files and a long README from Nov 13. They were
written from the Fuzzball 4.4 documentation before the team had the cluster. They were not run.
