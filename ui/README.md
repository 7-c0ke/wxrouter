# Control Room: dashboard + forecast assistant

One page for people and a chat API for agents.

- **Dashboard**: requests/s, p50/p95/p99, errors and failovers (last 10 s); a card per box
  (traffic share, queue, GPU, probe compute time, health, ACTIVE/preferred badges); a live
  60 s latency chart colored by box, with a per-second share strip; routing controls (rule,
  preferred box, demo traffic); and an event log of switches, failovers and demo actions.
- **Demo buttons** on each box card: Normal, Training (slow, 3×), Fail. These work on the
  mock servers, which have `/admin/mode`.
- **Forecast assistant**: ask for weather by city or coordinates. Answers come with a
  **weather card**: conditions icon (sun, moon, partly cloudy, cloud, rain, snow, from cloud
  cover, rain and local day/night), temperature now plus high/low, wind, humidity, rain (only
  when the model provides it), and a strip of 6-hour model steps. A **°C / °F** toggle (also
  km/h ↔ mph) is remembered per browser. The **maximize** button opens a full forecast view
  with a daily table (low–high bars, ensemble range), a table of the 6-hour model steps, and
  which box served it. Chips show which tool ran and which box answered. The assistant can
  also show cluster status, change the routing rule, or start or stop a simulated training job.

## Run

```bash
.venv/bin/python dev_stack.py            # mocks + router + UI → http://127.0.0.1:8080
.venv/bin/python dev_stack.py --no-precip   # mocks return precip_mm = null
```

With an LLM: any OpenAI-compatible endpoint that supports tool calling, for example Nemotron on
llama.cpp (`llama-server --jinja ...`) or vllm-router:

```bash
.venv/bin/python dev_stack.py --llm-base-url http://<box>:8080/v1 --llm-model nemotron
```

Without `--llm-base-url`, the assistant runs a built-in rule-based mode with the same tools, so
the demo works offline. The header shows which mode is on. If the LLM is unreachable, the reply
falls back to offline and says so.

Against the cluster: port-forward the router and boxes
(`fuzzball workflow port-forward <id> router 8000:8000`, and the same for `box-a 8001:8001` and
`box-b 8002:8002`), then run `python ui/server.py`.

## Real model, real jobs (merged from ui2/)

The page tells **simulated** from **real** apart. A box is a mock when `"mock"` is in its
`model_version`. Each box card has a `SIMULATED` or `REAL MODEL <version>` badge, every forecast
reply says which it is, and a Model card (from `GET /v1/info`) shows model, parameters, device,
newest starting state and speed.

**Five buttons per box:**

| button | kind | what it does |
|---|---|---|
| Normal | simulated | turns off slow or fail mode (`/admin/mode`) |
| Slow (sim) | simulated | the box's test hook makes it 3× slower; no job runs |
| Fail (sim) | simulated | the box's test hook answers 503; nothing is killed |
| Fine-tune / Stop tune | **real** | starts or stops the box's `finetune` job through its supervisor (`serve/node_stack.py`); the card shows its progress |
| Kill server | **real** | SIGKILLs the box's forecast server; the supervisor restarts it, and the router fails over meanwhile |

The real buttons need the box's supervisor, given with `--stack NAME=URL`. Without it, they're greyed
out. The assistant has the same split: `simulate_box` for boxes without a supervisor, and
`start_finetune`, `stop_finetune` and `kill_forecast_server` for boxes with one. It always says
whether an action was real or simulated.

Local stack with supervisors (our forecast server in `--mock`, a fake 30–40 s fine-tune job that
trains nothing, and real process kills):

```bash
.venv/bin/python ui/dev_stack_supervised.py          # page: http://127.0.0.1:19080
```

Against the real boxes (this is what `fuzzball/30-real-stack.fz` runs):

```bash
python ui/server.py --router http://ROUTER:8000 \
    --box box-a=http://NODE_A:8001 --box box-b=http://NODE_B:8002 \
    --stack box-a=http://NODE_A:8109 --stack box-b=http://NODE_B:8110 \
    [--llm-base-url http://NODE_A:8090/v1 --llm-model nemotron]
```

## API (for agents and scripts)

| endpoint | purpose |
|---|---|
| `POST /api/chat` `{"messages":[{"role":"user","content":"..."}]}` | assistant reply + `steps` (tool, args, result, ms) |
| `POST /api/forecast` `{"lat","lon","hours_ahead","ensemble_members"}` | forecast + summary + `served_by` |
| `GET /api/overview?since=<t>` | router status, 10 s metrics, box stats, new requests since `t` |
| `GET /api/router/config` | the router's settings, plus the rules, boxes and ranges allowed |
| `POST /api/router/config` `{"policy","preferred","margin","window_s","cooldown_s","return_margin","stall_ms","drain":{box:bool},"reset"}` | change any routing setting (only the fields sent); 422 with the router's reason and nothing changed for a bad value |
| `POST /api/box/{box}/mode` `{"mode":"normal|slow|error"}` | simulated demo scenes (`/admin/mode`) |
| `POST /api/box/{box}/train` `{"action":"start|stop","steps":N}` | real fine-tune job (needs `--stack`) |
| `POST /api/box/{box}/kill` | real kill of the forecast server (needs `--stack`) |
| `GET /api/model` | model info from `GET /v1/info` |
| `POST /api/load` `{"rps": 15}` | built-in demo traffic (0 = off) |

Tested on a laptop with the mocks (desktop and phone widths), and the LLM path against a stub
OpenAI-compatible server. It hasn't run against real Nemotron yet.
