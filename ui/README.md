# Control Room: dashboard + forecast assistant

One page for people and a chat API for agents.

- **Dashboard**: requests/s, p50/p95/p99, errors and failovers (last 10 s); a card per box
  (traffic share, queue, GPU, probe compute time, health, ACTIVE/preferred badges); a live
  60 s latency chart colored by box, with a per-second share strip; routing controls (rule,
  preferred box, demo traffic); and an event log of switches, failovers and demo actions.
- **Demo buttons** on each box card: Normal, Training (slow, 3×), Fail. These work on the
  mock servers, which have `/admin/mode`.
- **Forecast assistant**: ask for weather by city or coordinates. Answers include a
  temperature chart (with an ensemble range band when asked), rain bars, a marker at 72 h,
  and chips showing which tool ran and **which box served it**. It can also show cluster
  status, change the routing rule, or start or stop a simulated training job.

## Run

```bash
.venv/bin/python dev_stack.py            # mocks + router + UI → http://127.0.0.1:8080
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

## API (for agents and scripts)

| endpoint | purpose |
|---|---|
| `POST /api/chat` `{"messages":[{"role":"user","content":"..."}]}` | assistant reply + `steps` (tool, args, result, ms) |
| `POST /api/forecast` `{"lat","lon","hours_ahead","ensemble_members"}` | forecast + summary + `served_by` |
| `GET /api/overview?since=<t>` | router status, 10 s metrics, box stats, new requests since `t` |
| `POST /api/router/config` `{"policy","preferred"}` | change the routing rule |
| `POST /api/box/{box}/mode` `{"mode":"normal|slow|error"}` | demo scenes (mock only) |
| `POST /api/load` `{"rps": 15}` | built-in demo traffic (0 = off) |

Tested on a laptop with the mocks (desktop and phone widths), and the LLM path against a stub
OpenAI-compatible server. It hasn't run against real Nemotron yet.
