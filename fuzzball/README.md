# fuzzball/: start everything on the two GB10 nodes

These are the job files ("Fuzzfiles") for CIQ's Fuzzball 4.4.0. They start the forecast servers, the LLM
servers, the router, a training job beside the servers, and the comparison, on the team's two machines.
Run every command from the repo root, for example:

    fuzzball workflow start fuzzball/00-hello-gpu.fz
    fuzzball workflow log <workflow-id> hello-1

Written at the hackathon (Nov 13-14, 2026).

**Read this first (Nov 14, evening).** The stack that is deployed is `30-real-stack.fz`: one file with both
boxes, the two routers, the page and a hold job. It ran on the event cluster (Fuzzball 4.3.1) on Nov 14; what
was measured there is in `results/event-boxes-2026-11-14.md`, and what Fuzzball did that we did not expect is in
that file's own comments. `quick/` holds the one-time jobs that ran there too (GPU hello, image pull,
shared-volume test, the LLM, the volume, the year of NOAA data). `31-one-gpu-stack.fz` is a fallback for one
free GPU. The numbered files 00 to 50 below are the first night's plan; they were not run, and where they
differ from `30-real-stack.fz`, that file is right.

About the numbered files: nothing in them has run on a Fuzzball cluster. The syntax comes from the 4.4 documentation
(https://ui.stable.fuzzball.ciq.dev/docs/). What the team saw on 4.3.1 is in
`docs/reference/FUZZBALL-NOTES-4.3.1.md`. Every guess is marked `# UNVERIFIED:` in the files, and
section 3 below lists them all as a checklist for the first hour. What was checked on a laptop: every
file loads as YAML with only documented keys, every script passes `bash -n`, and the scripts were run
once with stand-in programs in place of the servers.

## The files

| File | What it starts | GPU |
|---|---|---|
| `00-hello-gpu.fz` | two small jobs, one per node: GPU, memory, network, volumes, internet. Pulls the big image on both nodes | 1 each |
| `10-setup-volume.fz` | code from GitHub and the missing Python packages onto the team volume | no |
| `11-fetch-weights.fz` | the weights and the data bundle onto the team volume | no |
| `20-node-stack-a.fz`, `20-node-stack-b.fz` | per node: the forecast server (8101) and the LLM server (8000), one service | 1 each |
| `30-router.fz` | the router (8100) with an address for laptops | no |
| `40-train-beside.fz` | a training job beside the servers of one node | uses the stack's |
| `50-bench.fz` | the comparison of the routing policies; results on the volume | no |

Two changes from the first plan, both because of how Fuzzball works:

- `20-node-stack.fz` became two files, one per node. A workflow is stopped as a whole, and one failed job cancels
  its whole workflow. With one workflow per node we can stop or lose node B and node A keeps serving.
- The fetch of weights moved out of `10-setup-volume.fz` into `11-fetch-weights.fz`. Fuzzball copies files in
  ("ingress") when a workflow starts, and a missing file fails the whole workflow. So the code setup, which we
  run often, does not wait for weights.

Everything lives on one persistent volume named `wxdesk`, mounted at `/wx` in every job:

    /wx/repo      the GitHub repository             /wx/nodes     A.addr, B.addr, router.addr (who is where)
    /wx/pydeps    Python packages the image lacks   /wx/control   requests for side jobs
    /wx/weights   final.pt, stage1_final.pt         /wx/run/A|B   per-node caches and the side job's log
    /wx/bundle    the forecast server's data        /wx/results   benchmark results
    /wx/data      the big training data (you put it there; only 40-train-beside.fz needs it)

## 1. Log in (again in every session: logins ran out within a day in practice)

    fuzzball context create default <api host>     # as on the organizers' slide
    fuzzball context use default
    fuzzball context login                         # prints a link; open it in a browser
    fuzzball group list
    fuzzball group use team<n>                     # workflows, group secrets and group objects belong to the group in use
    fuzzball version

- The CLI: web UI (https://<ui-host>/), "Download CLI" in the sidebar. The organizers say the
  web UI has a CLI built in as well.
- If `context create` wants more arguments, the documented form is
  `fuzzball context create default <api host> <auth URL> fuzzball-cli`. Ask the organizers for the auth URL.
- No browser: `fuzzball context login --direct` asks for user and password in the terminal.

## 2. Look around, then create the team volume (once)

    fuzzball node list --available-resources          # expect 2 nodes: cpu/arm64, 20 cores, about 128 GB, nvidia.com/gpu:1
    fuzzball volume provisioner list                  # our own storage provisioner and the shared one (refused? ask for the names)
    fuzzball volume provisioner info <name> -o yaml   # "local: true" under "driver:" means node-local storage (section 12)
    fuzzball volume list                              # the shared volume with models is listed here
    fuzzball volume create <our provisioner> wxdesk --permissions 0770

The files name the volume only by `name: wxdesk`. That works while no other volume we can see has that name.
If Fuzzball says the name is ambiguous, add `use: <our provisioner>` under each `name: wxdesk`.

## 3. First-hour checklist (every UNVERIFIED line, and how to check it)

| # | We assume | Where | Check | If it is not so |
|---|---|---|---|---|
| 1 | Each node shows one GPU, so two GPU jobs never share a node | 00, 20 | `fuzzball node list --available-resources`; the two hello logs show different `gpu=` and addresses | More than one GPU per node: the two servers could be two jobs. Same node twice: the other GPU was busy, ask `fuzzball workflow why <id>` |
| 2 | Our storage is one volume that both nodes see | all | no `local: true` in the provisioner info; last lines of the later hello log | section 12 |
| 3 | Jobs reach the internet | 10 | hello log, "internet" | ask the organizers; pip and git need it |
| 4 | git is in the PyTorch image | 10 | setup log, step 1 | send a tar file of the repository through the object cache (one more ingress entry in 11) and unpack it to `/wx/repo` |
| 5 | The image's pip knows `--dry-run --report`; arm64 wheels exist for physicsnemo's needs and pygrib | 10 | setup log, step 2 and the import check | set `PIP_MODE=user` at the top of 10 and run it again: `pip install --user` into `/wx/pyuser`, the way that worked on 4.3.1. The node stacks look there too |
| 6 | The web UI host hands the CLI to a job (`/api/download/cli/linux/arm64`) | 10, 50 | setup log, step 3 | only `EVENTS=contention` in 50 needs it; run the comparison from a laptop instead (section 10) |
| 7 | Ingress into a persistent volume works | 11 | does `11-fetch-weights.fz` start? | use an ephemeral volume, as the comment in the file says |
| 8 | `host: true` works for a one-node service | 20, 30 | the stack starts, and its log shows the node's real address (compare with `fuzzball node list`) | delete the line in the three files. The ports stay published on the node. A stack then takes its address from the note that `00-hello-gpu.fz` left for its GPU; or set `NODE_A_ADDR` and `NODE_B_ADDR` by hand in 30 |
| 9 | The address of a node's default route is reachable from the other node, and ports 8000, 8100, 8101 are open between the nodes | 20, 30 | the router log starts with four health lines; `200` is good, `000` is no answer | `WX_ADVERTISE` in the stack files, or the two addresses in 30; a firewall: ask the organizers |
| 10 | One job may mount the shared volume and our volume together | 20 | the stack starts with the `shared` lines switched on | fetch the GGUF file onto our volume instead: an ingress line `hf://<org>/<repo>/<file>.gguf` in 11 |
| 11 | Our llama-server flags fit the build (`--metrics`, `--jinja`, ...) | 20 | stack log | change `LLM_EXTRA` |
| 12 | Endpoint: `protocol: http` and `type: subdomain` are right, and the cluster resolves subdomain addresses | 30 | `fuzzball endpoint list`, open the address | port-forward (section 6) |
| 13 | `fuzzball workflow port-forward <id> <service> <local>:<remote>` (seen on 4.3.1; the 4.4 pages only name the command) | README | `fuzzball workflow port-forward --help` | the endpoint |
| 14 | Inside a job, the CLI works with the job's own token, and `workflow start` prints the new id | 50 | bench log, "what the last start ... said" | section 10, from a laptop |
| 15 | Egress from our volume into the object cache | 50 | optional lines in the file | section 7, other ways |
| 16 | 14 cores and 96 GB per stack fit a node | 20 | a stack that stays Pending: `fuzzball workflow why <id>` | lower the numbers |
| 17 | Our jobs land only on our own two nodes | all | the addresses in the hello logs are our nodes' (`fuzzball node list`) | ask the organizers how a team's nodes are chosen |
| 18 | Programs stored on the volume may run (it is not mounted `noexec`) | 10, 20, 50 | hello log: "programs on /wx can run: yes" | packages, llama-server and the CLI cannot live on the volume then: tell the organizers, or bake them into an image (section 9, way 3) |

Also seen on 4.3.1 and worth one look: memory requests were booked but not enforced; a second GPU job waits for
the first; a GPU job that waits can hold back later jobs.

## 4. The order to run

    fuzzball workflow validate fuzzball/00-hello-gpu.fz       # checks a file without running it

    # 1. Hello. Minutes the first time: each node pulls the 20 GB image.
    fuzzball workflow start fuzzball/00-hello-gpu.fz
    fuzzball workflow events <id> --follow
    fuzzball workflow log <id> hello-1
    fuzzball workflow log <id> hello-2

    # 2. Code and packages. First edit REPO_URL at the top of the file.
    fuzzball workflow start fuzzball/10-setup-volume.fz
    fuzzball workflow log <id> setup -f                       # ends with SETUP DONE or SETUP FAILED in: ...

    # 3. Weights and bundle: from the laptop into the object cache, then onto the volume.
    #    (A file you do not have: delete its four lines in 11-fetch-weights.fz first.)
    tar -czf bundle.tar.gz -C /path/to/out bundle
    fuzzball object put ./final.pt         fb://group/wx/final.pt
    fuzzball object put ./stage1_final.pt  fb://group/wx/stage1_final.pt
    fuzzball object put ./bundle.tar.gz    fb://group/wx/bundle.tar.gz
    fuzzball workflow start fuzzball/11-fetch-weights.fz
    fuzzball workflow log <id> place                          # ends with FETCH DONE

    # 4. One stack per node. Each holds its node's GPU until it is stopped.
    fuzzball workflow start fuzzball/20-node-stack-a.fz
    fuzzball workflow start fuzzball/20-node-stack-b.fz
    fuzzball workflow log <id> stack-a -f                     # wait for "listening on ..." and "up."

    # 5. The router. Its log starts with a health line for each of the four servers.
    fuzzball workflow start fuzzball/30-router.fz
    fuzzball workflow log <id> router -f

    # 6. The comparison. RESULTS.md is printed at the end of the log.
    fuzzball workflow start fuzzball/50-bench.fz
    fuzzball workflow log <id> bench -f

    # 7. Training beside serving, on the node named in the file. It needs the training data in /wx/data.
    fuzzball workflow start fuzzball/40-train-beside.fz
    fuzzball workflow log <id> train-beside -f
    fuzzball workflow stop <id>                               # stopping it stops the training

    # Stop everything at the end (the services run until stopped):
    fuzzball workflow list --order-by -created
    fuzzball workflow stop <id>

Rules of thumb:

- While both stacks are up, both GPUs are taken. Do not start another GPU job then (not `00-hello-gpu.fz` either):
  it waits, and a waiting GPU job can hold back other jobs. If it happened: `fuzzball workflow stop <id>`.
- After `10-setup-volume.fz` pulled new code, stop and start the stacks and the router. Running servers keep the old code.
- After new weights (`fuzzball object put --force ...`, then 11 again), stop and start the stacks.
- A stack on the letter A does not have to sit on a fixed machine. Whoever starts second gets the other node.

## 5. See what is going on

    fuzzball workflow list --order-by -created     # newest first
    fuzzball workflow describe <id>                # the stages (volume, image, job, service) and their status
    fuzzball workflow events <id> --follow         # image pulls, the node a stage runs on, why it is blocked
    fuzzball workflow log <id> <job> -f            # -n 50 for the last 50 lines
    fuzzball workflow why <id>                     # why a stage is still waiting
    fuzzball queue show --mine                     # our queued and running stages
    fuzzball workflow exec <id> <job> -- nvidia-smi             # one command inside a running job
    fuzzball workflow exec --tty <id> <job> -- /bin/bash        # a shell inside it (keep the "--")
    fuzzball workflow get <id> --raw               # the file as we sent it (Fuzzball stores a rewritten copy)
    fuzzball node show <node-id>                   # one node: free cores, memory, GPU, and the jobs on it

Status words: Pending (waiting), Started (running), Finished, Errored, Canceled. A workflow is named after its
file, so the list shows which id is which. `<job>` above is one of: `hello-1`, `hello-2`, `setup`, `place`,
`stack-a`, `stack-b`, `router`, `train-beside`, `bench`.

## 6. Reaching ports

What Fuzzball gives us:

- Inside one workflow, a job reaches a service by its name (`http://router:8100`).
- Across workflows, names do not resolve (seen on 4.3.1). A service's declared ports are published on its node,
  so use `http://<node address>:<port>`. That is why each stack writes its node's address to `/wx/nodes/A.addr`
  or `B.addr`, and the router reads them.
- These ports are open on the node address with no login. Anyone who can reach the node can call them.
- Two services that declare the same port never share a node. With the one-GPU rule, that puts stack A and
  stack B on different nodes.

### Reaching the router

| From | Address | Notes |
|---|---|---|
| a job on the cluster | `http://<address in /wx/nodes/router.addr>:8100` | `50-bench.fz` finds it by itself |
| a laptop on the nodes' network | `http://<node address>:8100` | try `curl http://<node address>:8100/router/health`. If it answers, nothing else is needed |
| a laptop, any program | `fuzzball workflow port-forward <router workflow id> router 8100:8100`, then `http://127.0.0.1:8100` | on 4.3.1 a request that sent nothing back for 60 s was cut. Stream LLM replies. The dashboard's polling keeps its own connection busy |
| a browser | the address from `fuzzball endpoint list`, or the Connect button on the workflow's page in the web UI | https, through Fuzzball's proxy. Works when the browser is logged in to the web UI |

The dashboard is the page at `/` of the router, so it has the same address as the router. It also reads each
forecast server's `/metrics` straight from the node addresses the router knows. A laptop outside the nodes'
network cannot reach those, so that part of the page may stay empty there. What comes from the router works.

The load generator takes plain `http://` addresses only, and none of our programs sends a Fuzzball token. From a
laptop, give them the port-forward address, or a node address. To call the https endpoint from a script, get a
token and send it in a header:

    fuzzball endpoint list                                        # ID and URL of each endpoint
    fuzzball endpoint generate-token <ENDPOINT_ID> --expiration 24h
    curl -H "FB-Authorization: Bearer <token>" https://endpoint-<id>.endpoints.<domain>/router/health

Or set `scope: public` in `30-router.fz`: then the https address needs no login and no token. Anyone who has the
address can then use the router, so do that only for the demo.

The same port-forward works for a single server, for example the forecast server of node B:

    fuzzball workflow port-forward <stack-b workflow id> stack-b 18101:8101      # then http://127.0.0.1:18101/health

## 7. Files in and out

In, from a laptop: put the file into the object cache, then name it as an ingress source in a Fuzzfile
(`11-fetch-weights.fz` shows how).

    fuzzball object put ./final.pt fb://group/wx/final.pt          # --force replaces an object that is there
    fuzzball object list fb://group/wx

Out, results:

- The comparison's table is in the log: `fuzzball workflow log <bench workflow id> bench`.
- Small files, through any running job that mounts the volume (the router is always up):
  `fuzzball workflow exec <router workflow id> router -- cat /wx/results/<name>/summary.json > summary.json`
- The whole results folder: switch on the five `egress` lines in `50-bench.fz`. After the run:
  `fuzzball object get fb://group/wx-results/<bench workflow id>.tgz ./` (UNVERIFIED, checklist 15).
- UNVERIFIED alternative: `fuzzball workflow exec <router workflow id> router -- tar -C /wx/results -czf - <name> > <name>.tgz`

An `https://` ingress source is kept in the group's object cache for 7 days under its address (seen on 4.3.1).
A changed file behind the same address is not fetched again: give it a new name.

## 8. Secrets

A job cannot read our laptop's environment. A secret is stored once in Fuzzball and named in the file.

    # a GitHub token for a private repository (10-setup-volume.fz, the GITHUB_TOKEN line)
    printf 'type: value\nsecret:\n  value: %s\n' "<the token>" > gh.yaml
    fuzzball secret create github-token gh.yaml --from-file -s user
    rm gh.yaml
    fuzzball secret list                           # shows secret://user/github-token

    # a Hugging Face token, for an "hf://" ingress source of a gated model
    fuzzball secret create secret://user/hf-token --type hf '{"token":"hf_..."}'

In a file: `- GITHUB_TOKEN=secret://user/github-token` in `env` (must be user scope), `secret: secret://user/hf-token`
under an ingress `source`, `secret:` under `image` for a private registry.

## 9. One GPU, one job: what follows from it

A node reports one GPU. Fuzzball counts GPUs: a job that asks for `nvidia.com/gpu: 1` takes the node's only one,
and the next GPU job waits. A job that does not ask sees no GPU at all. So everything that uses a node's GPU runs
inside one job, the node stack:

- **Both servers share one image.** The stack uses the PyTorch image, because the forecast server needs it. The LLM
  server must therefore be a `llama-server` program that runs in that image. `LLM_BIN` points at it. While it is
  missing, the stack runs the forecast server alone and says so in its log. The same holds if the LLM server
  ends: the forecast server keeps running. If the forecast server ends, the whole stack stops with an error.
  Three ways to get a llama-server:
  1. Build it once on the volume, in a shell inside a running stack (UNVERIFIED: the image has git, cmake and the
     CUDA compiler):

         fuzzball workflow exec --tty <stack workflow id> stack-a -- /bin/bash
         export HOME=/wx/home
         git clone --depth 1 https://github.com/ggml-org/llama.cpp /wx/llama.cpp
         cmake -S /wx/llama.cpp -B /wx/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
         cmake --build /wx/llama.cpp/build --target llama-server -j 12
         /wx/llama.cpp/build/bin/llama-server --version

     Then switch on the `shared` lines, set `LLM_MODEL`, rebuild the -b file (section 11), and restart both stacks.
  2. One image that has both PyTorch 2.11 with CUDA and llama-server (the teammates' registry, `oras://...`).
     Put it as the stack's `image: uri:` and set `LLM_BIN=llama-server`.
  3. Let Fuzzball 4.4 build such an image from the PyTorch image (an `images:` section; not tried).
- **Training beside serving is a side job of the stack.** `40-train-beside.fz` is a small CPU job. It writes a
  request file to `/wx/control/side-<letter>.sh`. The stack on that node picks it up within seconds and runs it in
  its own container, where the GPU is. Stopping the 40 workflow stops the training. By hand, from any job:
  `touch /wx/control/side-B.stop`.
- **vLLM instead of llama.cpp:** the commented lines in the stack file. It needs an image with vLLM and PyTorch,
  and `LLM_PROBE=vllm` in `30-router.fz`.

## 10. Timed events for the comparison

On the real nodes the scenarios that use test hooks (contention, failure, recovery) do not work; `steady` and
`burst` do. `bench/compare.py` runs `--scenarios steady` and takes what happens from `--events FILE.json`, where
an event is a command that starts or stops a job:

    python bench/compare.py --target http://<router>:8100 --duration 360 --rate 4 --out /wx/results/NAME \
        --scenarios steady --events /wx/results/NAME/events.json

`50-bench.fz` writes that file itself for `EVENTS=contention` (start `40-train-beside.fz` at one third, stop it at
two thirds). For other events, save a file like this in the repository (it reaches the volume with the next
`10-setup-volume.fz`) and set `EVENTS=/wx/repo/<path>`:

    [
      {"at_s": 120, "name": "fail",
       "cmd": ["bash", "-c", "/wx/bin/fuzzball workflow stop $(cat /wx/nodes/B.workflow)"]},
      {"at_s": 240, "name": "recover",
       "cmd": ["bash", "-c", "/wx/bin/fuzzball workflow start /wx/repo/fuzzball/20-node-stack-b.fz"]}
    ]

`/wx/nodes/B.workflow` holds the id of the workflow that runs stack B. The events fire in every run, once per
policy. A stopped stack needs a minute or more to serve again, so test failures with one policy per run
(`EXTRA_ARGS=--policies adaptive`) or a long pause.

From a laptop it is the same with the laptop's own CLI, through the port-forward:

    python bench/compare.py --target http://127.0.0.1:8100 --duration 360 --rate 4 --out bench/results/NAME \
        --scenarios steady --events events.json
    # in events.json: ["bash", "-c", "fuzzball workflow start fuzzball/40-train-beside.fz > /tmp/wx-train.out"]
    #            and: ["bash", "-c", "fuzzball workflow stop $(grep -oE '[0-9a-f-]{36}' /tmp/wx-train.out | head -1)"]

The router can heal a node the same way. A heal action is a `cmd` in the router's config file
(`router/config.example.json`); for backend B it can be the two commands above: stop the old stack workflow, then
start `20-node-stack-b.fz`. A new stack cannot start while the old one still holds the GPU and the ports. Give the
config file to the router with the `ROUTER_EXTRA` line in `30-router.fz`.

## 11. Changing the node stack

Edit `20-node-stack-a.fz`. Then make the B file from it, so that the two never drift apart:

    sed -e 's/stack-a/stack-b/g' -e 's/WX_NODE=A/WX_NODE=B/g' fuzzball/20-node-stack-a.fz > fuzzball/20-node-stack-b.fz
    diff fuzzball/20-node-stack-a.fz fuzzball/20-node-stack-b.fz     # only the name and the letter differ

## 12. If our storage is node-local (`local: true`)

Then a volume lives on one node's disk, and Fuzzball places every job that mounts it on that node. One volume for
both nodes does not work. First ask the organizers for storage that both nodes see. Otherwise:

    fuzzball volume create <our provisioner> wxdesk-b --node <id of the other node> --permissions 0770

Put `name: wxdesk-b` in `20-node-stack-b.fz`, run 10 and 11 once more with `name: wxdesk-b`, and set
`NODE_B_ADDR` by hand in `30-router.fz` (the router sees only one of the two volumes). A side job or a bench event
for node B must then come from a job that mounts `wxdesk-b`.

## 13. Fallback without Fuzzball: docker or podman on each box

If Fuzzball gets in the way and we have a shell on the boxes. Each box needs its own copy of the folder that is
`/wx` above (here `$WX`). With podman, write `--device nvidia.com/gpu=all` in place of `--gpus all`.

    IMG=nvcr.io/nvidia/pytorch:26.02-py3
    RUN="docker run --rm --network host --ipc host --user $(id -u):$(id -g) -v $WX:/wx -w /wx/repo \
         -e HOME=/wx/home -e USER=wx -e PYTHONUSERBASE=/wx/pyuser -e PYTHONUNBUFFERED=1"

    # once per box: code and packages. "pip install --user" adds only what the image lacks, so the image's torch stays.
    git clone https://github.com/OWNER/REPO.git $WX/repo && mkdir -p $WX/home $WX/pyuser
    $RUN -e PIP_BREAK_SYSTEM_PACKAGES=1 $IMG python -m pip install --user nvidia-physicsnemo==2.2.2 pygrib
    $RUN $IMG python -c "import torch, physicsnemo.models.afno, pygrib; print(torch.__file__)"   # must not start with /wx

    # forecast server (node letter A or B)
    $RUN --gpus all -d --name wx-forecast $IMG \
        python serve/forecast_server.py --node A --port 8101 --host 0.0.0.0 --weights /wx/weights/final.pt --data /wx/bundle

    # LLM server (its own image here; docker lets two containers share the GPU)
    docker run -d --name wx-llm --gpus all --network host -v /path/to/models:/models LLAMA_IMAGE \
        llama-server --model /models/MODEL.gguf --host 0.0.0.0 --port 8000 --n-gpu-layers 999 --metrics --ctx-size 32768 --parallel 4 --jinja

    # router (on either box; the addresses of the two boxes)
    $RUN -d --name wx-router $IMG python router/router.py \
        --forecast A=http://IP_A:8101,B=http://IP_B:8101 --llm A=http://IP_A:8000,B=http://IP_B:8000 \
        --llm-probe llamacpp --port 8100 --host 0.0.0.0

    # training beside serving (on the box that should get busy)
    $RUN --gpus all --name wx-train $IMG \
        python weather/train.py --stage 2 --init /wx/weights/stage1_final.pt --out /wx/runs/beside --data /wx/data --max-steps 400

    # the comparison (events are then: docker start / docker stop wx-train, docker stop wx-forecast, ...)
    $RUN --name wx-bench $IMG \
        python bench/compare.py --target http://IP_ROUTER:8100 --duration 120 --rate 4 --out /wx/results/NAME --scenarios steady

    docker logs -f wx-forecast        # logs
    docker stop wx-forecast wx-llm wx-router

## 14. What the documentation says, page by page

Checked in the 4.4 documentation (all under https://ui.stable.fuzzball.ciq.dev/docs/):

| Fact | Page |
|---|---|
| File layout: `version: v4`, `volumes`, `jobs`, `services`, `defaults`; a job has `image.uri`, `script` or `command`, `env` (KEY=VALUE), `cwd`, `mounts` (path: volume), `resource` (cpu.cores, memory.size, devices), `policy.timeout.execute`, `policy.retry`, `depends-on`, `multinode`, `task-array` | appendices/workflow-syntax/ |
| Image addresses must start with `docker://`, `oras://`, `fb://` or `fuzzball://`. `cpu/arm64: 1` keeps a job on arm64 nodes and is not used up | creating-fuzzfiles/container-urls/ |
| `nvidia.com/gpu: 1` under `devices`; the driver tools come into any image | advanced-hardware/gpus/ |
| Devices are counted per node (total, requested, available) | nodes/node-details/ |
| Services: `network.ports`, `network.endpoints`, `network.host`, `persist`. Declared ports are published on the node; two services with the same port never share a node. Without `persist: true` a service that nothing depends on is stopped at once. Services take no `policy` | appendices/workflow-syntax/#services |
| A service's name resolves inside its workflow | appendices/workflow-syntax/#services, advanced-features/job-hostnames/ |
| Endpoints: `type` path or subdomain (subdomain is for apps that expect to sit at the root of an address), `scope` user, group, organization or public (default group), tokens, `FB-Authorization` | advanced-features/workflow-endpoints/ |
| The Connect button and `fuzzball workflow connect` | advanced-features/workflow-connect/ |
| Persistent volumes must be created first: `fuzzball volume create PROVISIONER NAME`, `--permissions`, `--node`. `name:` alone finds a volume whose name is unique | volumes/vols-operations/vols-ops-creation/, volume-admin/storage-volumes/sav-workflows/ |
| Node-local storage (`local: true`, shown by `fuzzball volume provisioner info`) pins jobs to one node; one job cannot mount volumes of two storage segments | volume-admin/provisioners/provisioner-driver-types/, volume-admin/provisioners/provisioner-definitions/ |
| Ingress sources: `https://`, `s3://`, `hf://`, `fb://`, `fuzzball://`; destination `file://path` | volumes/vols-operations/vols-ops-ingress/ |
| Object cache: `fuzzball object put / get / list / delete`, `fb://group/...`, `fb://user/...` | object-cache/ |
| Secrets: `fuzzball secret create NAME FILE --from-file -s user`, `secret://user/NAME`, env secrets are user scope | secrets/sec-adding/, secrets/sec-in-workflows/ |
| Every job gets `FB_TOKEN`, `FB_API_HOST`, `FB_WORKFLOW_ID`, `FB_JOB_NAME`; the CLI works inside a job with them; where a job downloads the CLI | advanced-features/in-workflow-api-access/, appendices/environment-variables/ |
| `workflow start`, `stop`, `describe`, `list`, `log`, `validate`, `get --raw` | quick-start/ (the pages under it) |
| `workflow events <id> --follow` | creating-fuzzfiles/building-images/, debugging/ |
| `node list --available-resources`, `node show` | nodes/listing-nodes/, nodes/node-details/ |
| `workflow exec` (jobs run as our user, not root) | advanced-features/shell-exec/ |
| `workflow why`, `queue show --mine` | scheduler-admin/queue-observability/ |
| `context create`, `context login --direct`, `group use` | quick-start/uqs-logging-in/ |
| 4.4 no longer treats `{{ }}` in a file as a template; retry needs `policy.retry.on` | release-notes/v4.4.0/ |
| Images can be built inside a workflow (`images:`) | creating-fuzzfiles/building-images/ |

Read but not used here: task arrays (parallel-and-distributed/parallel/), multinode jobs
(parallel-and-distributed/generic/), autoscaling, readiness probes, retries.

Not in the 4.4 pages, taken from the team's 4.3.1 notes: the `port-forward` arguments and its 60-second cut;
names do not resolve across workflows; a job without a GPU request sees no GPU; memory is booked but not enforced;
`pip install --user` with `PYTHONUSERBASE` on a volume works; one failed job cancels its workflow.
