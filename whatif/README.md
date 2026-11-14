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
| `prompts.py`, `scheduler.py` | policies `whatif` (plan 3, prune, spawn up to 3, early stop at p>=0.6 with a 0.3 lead from round 2, a judge over every branch), `equal`, `single`, `triage` (rule first; agents only when the signal is unclear), `rule` |
| `remediate.py` | one action per diagnosis, dry run unless `--apply` |
| `scenarios.py` | labelled incidents on private local stacks (parallel), or `--real slow:box-a` on the cluster |
| `eval.py`, `summarize.py` | every policy on every incident, interleaved, LLM policies `--repeats` times; AC@1/AC@3 (mean and run range), tokens, AC@1 per 10k tokens, paired sign test; `--split dev\|test` from `data/splits.json` |
| `cli.py` | `check`, `diagnose`, `eval`, `record` |
| `forward.sh` | port-forward the real stack's LLMs, router, boxes and supervisors to localhost |

```bash
python -m whatif.cli record -n 10 --parallel 4                       # data/incidents/
python -m whatif.cli eval --policies single,equal,whatif,triage,rule  # results/whatif/<time>/summary.md
python -m whatif.cli eval --incidents data/incidents-30 --split test --repeats 2 --parallel 2
python -m whatif.cli diagnose --incident data/incidents/<id>.json --policy whatif
whatif/forward.sh <real-stack workflow id> &  python -m whatif.cli check
python -m whatif.cli diagnose --live [--apply]
```

The final ranking of `whatif` and `triage` is the judge's beliefs over every branch the agents explored. A judge
reply that does not parse gets one retry (thinking off, JSON only); with no usable answer the ranking is the rule's
on the same recorded window. Ties always break by the rule's order.

Tune on the `dev` split only; report on `test`.
