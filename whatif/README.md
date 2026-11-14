# WhatIf: incident diagnosis for the two-box stack

A recorder keeps the last 180 s of telemetry (router, both boxes, a small probe forecast every 2 s). When an
alert fires at t0, LLM agents test one hypothesis each with cheap tools that compare [t0-30 s, t0) with
[t0, t0+30 s); a scheduler decides which agents keep going. Ten hypotheses: `box-a|box-b` x
`slow|down|error|hang`, `load:overload`, `router:misconfig`.

| file | what |
|---|---|
| `config.py` | settings from env / `.env`: `LLM_ENDPOINTS`, `LLM_API_KEY`, `MODEL`, `NODE_CAP`, `ROUTER`, `BOX_A/B`, `STACK_A/B` |
| `telemetry.py` | 1 Hz recorder into a ring buffer; drops fault-injector fields (`mode`, `slowdown`, `busy_other`, health text); save/load incidents |
| `tools.py` | `box_health`, `box_stats`, `probe`, `router_status`, `traffic` (1 unit), `box_jobs` (2); budget 8; lines like `box-b.compute_ms: 57 -> 171 (+200%), onset +2s, step` |
| `hypotheses.py` | the ten faults with a runbook each |
| `llm.py` | sticky per incident, least-loaded pick, priority gate (cap `NODE_CAP`), failover, thinking off by default, JSON extraction, `calls.jsonl` |
| `prompts.py`, `scheduler.py` | policies `whatif` (plan 4, prune, spawn up to 5, early stop, thinking judge on ties), `equal`, `single`, `rule` |
| `remediate.py` | one action per diagnosis, dry run unless `--apply` |
| `scenarios.py` | labelled incidents on private local stacks (parallel), or `--real slow:box-a` on the cluster |
| `eval.py`, `summarize.py` | every policy on every incident, interleaved; AC@1/AC@3, cost, paired sign test |
| `cli.py` | `check`, `diagnose`, `eval`, `record` |
| `forward.sh` | port-forward the real stack's LLMs, router, boxes and supervisors to localhost |

```bash
python -m whatif.cli record -n 10 --parallel 4                       # data/incidents/
python -m whatif.cli eval --policies single,equal,whatif,rule         # results/whatif/<time>/summary.md
python -m whatif.cli diagnose --incident data/incidents/<id>.json --policy whatif
whatif/forward.sh <real-stack workflow id> &  python -m whatif.cli check
python -m whatif.cli diagnose --live [--apply]
```
