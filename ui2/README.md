# Control Room, revised copy (ui2)

This folder is a revised copy of `ui/` (by Niravkumar Patel). `ui/` itself is not changed. The copy was
made from commit d3728fd and then brought up to `ui/` as of commit dfcaa4b (the relative page addresses and
the balanced rule with its health score). The page looks and works as the original does. The differences
are below.

Why a copy and not a change to `ui/`: the changes are not small, and `ui/` is a teammate's work. If they are
wanted, the right end state is one folder, with these changes folded into `ui/`.

## What is different from `ui/`

1. **Missing rain does not break it.** Our forecast server sends `"precip_mm": null` on every point (the
   model has no rain output). `ui/server.py` crashes on that. Here the summary's `precip_mm` is `null` when
   no point has a number, and each point carries `p: null`. The chat answer then says nothing about rain,
   the chart draws no rain bars, and the "rain bars" label is hidden. Wind and temperature are guarded the
   same way. (`pressure_hpa` is not read anywhere in `ui/` or here.)
2. **The page says what is simulated.** A box is a mock when `"mock"` is in its `model_version`.
   `/api/overview` now also reads each box's `GET /health` to learn that. Each box card has a badge:
   `SIMULATED`, or `REAL MODEL` plus a short model version. The top bar shows "simulated boxes" when any box
   is a mock. Each forecast answer in the chat has a chip: "simulated data" or "real model". The buttons
   that use `/admin/mode` are now labelled "Training (simulated)" and "Fail (simulated)".
3. **Real training and real failure, when the box's supervisor is known.** `--stack NAME=URL` gives the
   address of `serve/node_stack.py` on a box. For such a box:
   - the card has "Start fine-tune" / "Stop fine-tune" (one button) and "Kill forecast server", in place of
     the three simulated buttons;
   - the card shows the job: `fine-tune: step 120/300 · 1.52 steps/s · 80 s`, then done, stopped or failed;
   - the event log says "Real job: ..." for these actions and "Simulated: ..." for `/admin/mode` actions;
   - the assistant has the tools `start_finetune(box, steps?)`, `stop_finetune(box)` and
     `kill_forecast_server(box)` for these boxes, and `simulate_box` only for boxes without a supervisor.
     Every reply says whether the action was real or simulated.
   A supervisor that is slow or gone never breaks the page: its status is `null` and the card says so.
4. **A "Model" card** under the box cards, from `GET /v1/info` of the first box that answers it: model name,
   parameters in millions, device, newest starting state, speed per member-step. The teammate's mock has no
   `/v1/info`; the card then reads "Simulated boxes: forecast numbers are made up by the mock."
5. **Smaller fixes in the rule-based assistant.** Its replies check the tool's result before they report
   success. "Stop the training job on box B" stops it (the original starts one). A question such as
   "Is box A down?" gets the status and changes nothing. "box" followed by a word starting with a or b
   ("which box are slow") is no longer read as box-a or box-b.
6. **`dev_stack2.py`** starts a local test stack with our forecast server and our supervisor (see below).

## Run

Use a Python that has FastAPI, uvicorn, httpx and pydantic (the same as for `ui/`). Run from the repo root.

**Local test stack with our server** (two supervisors, each with our forecast server in mock mode and a
fake `finetune` job; the router; the page):

```bash
python ui2/dev_stack2.py          # page: http://127.0.0.1:19080   (Ctrl-C stops everything it started)
python ui2/dev_stack2.py --help   # every port can be set
```

Nothing in that stack is real: the forecast numbers are made up, and the "fine-tune" trains nothing. It
prints progress lines in the format of `weather/finetune.py` for about 40 s, so the buttons have a job to
start, watch and stop. Killing the forecast server is a real process kill; the supervisor restarts it.

**With the teammate's mocks** (as `dev_stack.py` does it, but with this page):

```bash
python mock_server.py --name box-a --port 8001 --seed 1 &
python mock_server.py --name box-b --port 8002 --seed 2 &
python router.py --port 8000 --policy adaptive --preferred box-b \
    --backend box-a=http://127.0.0.1:8001 --backend box-b=http://127.0.0.1:8002 &
python ui2/server.py --port 8080 --router http://127.0.0.1:8000 \
    --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002
```

**Against the real boxes:**

```bash
python ui2/server.py --router http://ROUTER:8000 \
    --box box-a=http://NODE_A:8101 --box box-b=http://NODE_B:8101 \
    --stack box-a=http://NODE_A:8109 --stack box-b=http://NODE_B:8109 \
    [--llm-base-url http://NODE_A:8000/v1 --llm-model nemotron]
```

- `--box NAME=URL` is the forecast server of a box. `--stack NAME=URL` is its supervisor
  (`serve/node_stack.py`). `--stack` is optional, per box. A box without it keeps the simulated buttons.
- The page still shows exactly two boxes named `box-a` and `box-b`, as the original does. So the router's
  `--backend` names and the names in `--box` and `--stack` must be `box-a` and `box-b`. The forecast
  server's own `--node` name does not matter to the page.
- The supervisor must run with `--admin`, keep a server named `forecast` and hold a job named `finetune`.
  Without `--admin` it answers 403, and the page shows that text in the event log.
- The simulated buttons need `/admin/mode` on the box: the teammate's mock has it, our server has it only
  when started with `--admin`.

Things to know for the demo:
- "Kill forecast server" is a real SIGKILL. The supervisor starts the server again after 1 s. If you kill
  again within a minute, it waits 2, then 5, 10 and 30 s.
- "Stop fine-tune" answers when the job is gone. The supervisor sends SIGTERM, and SIGKILL 30 s later. It
  calls every run that did not exit with 0 "failed"; the page shows "stopped" when the exit was a signal.
- The real `finetune.py` prints a step line every 20 steps. So the step on the card moves in jumps of 20.
  Before the first step line the card shows the last line the job printed.
- "Start fine-tune" from the button runs the job with its own default number of steps. The API and the
  assistant can pass a number ("start a training job on box B with 200 steps").

## API (for agents and scripts)

| endpoint | purpose |
|---|---|
| `POST /api/chat` `{"messages":[{"role":"user","content":"..."}]}` | assistant reply + `steps` (tool, args, result, ms) |
| `POST /api/forecast` `{"lat","lon","hours_ahead","ensemble_members"}` | forecast + summary + `served_by` + `simulated`; missing numbers are `null` |
| `GET /api/overview?since=<t>` | as before, plus `health` (per box: `box`, `model_version`, `status`, `simulated`), `stacks` (per box with `--stack`: `finetune` and `forecast`, or `null`), `model`, `simulated_boxes` |
| `GET /api/model` | `/v1/info` of the first box that has it (kept for 10 s) |
| `POST /api/router/config` `{"policy","preferred"}` | change the routing rule |
| `POST /api/box/{box}/mode` `{"mode":"normal|slow|error"}` | simulated demo scenes (`/admin/mode`) |
| `POST /api/box/{box}/train` `{"action":"start"|"stop","steps":200}` | real job: the supervisor's `finetune` job (`steps` is optional) |
| `POST /api/box/{box}/kill` | real: SIGKILL to the supervisor's `forecast` server |
| `POST /api/load` `{"rps": 15}` | built-in demo traffic (0 = off) |

`train` and `kill` return the supervisor's own JSON plus `"kind": "real job"`, or `{"error": "..."}`.

## What was tested, and what was not

Tested on a laptop (macOS, Python 3.12), on Nov 14, 2026:
- With the teammate's mocks and router: `/api/overview`, `/api/forecast` (rain numbers are there),
  `/api/chat` "Weather in Chicago next 48 hours" (the reply gives the rain), `/api/box/box-b/mode` slow and
  normal, and the rule-based box, policy and status phrases.
- With `dev_stack2.py`: `/api/overview` (health, supervisor status, model info), `/api/forecast` and
  `/api/chat` with null rain (no rain text, no error), `train` start, a second start (refused by the
  supervisor), stop, a run with `steps`, `kill` (the router failed over on the next request and the server
  was back in about 1 s), and the rule-based phrases for the real actions.
- The page itself in a browser (desktop width), against both stacks: badges, the top-bar pill, the chips,
  the chart with and without rain, the three simulated buttons, Start/Stop fine-tune with the job line,
  Kill forecast server, the event log lines, the Model card. No errors in the console.
- A supervisor that refuses the connection, and one that accepts and never answers: the overview still
  answers (in 1 s at most) with `null` for it.
- The LLM path against a small stub server only: the tool list, the system prompt and the tool calls.
- Ctrl-C and SIGTERM on `dev_stack2.py`: every process it started was gone afterwards.

Not tested:
- A real model. The `REAL MODEL` badge, the "real model" chip and the Model card with real values were
  only drawn from made-up data in the browser.
- The real `weather/finetune.py` under the supervisor. Only the fake job, which prints the same line format.
- A real LLM (Nemotron). So it is not known how well a model follows "say nothing about rain".
- The cluster (Fuzzball, port-forward), phone widths, dark mode.
