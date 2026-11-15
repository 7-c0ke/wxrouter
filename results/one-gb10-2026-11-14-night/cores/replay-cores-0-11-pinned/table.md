# Routing benchmark on the real stack: 20261115-050206

- made by bench/fault_table.py against http://127.0.0.1:19000 (model afno-19m-65f9f3bb3270)
- trace: seed 42, 1182 requests over 150 s (about 8.0 requests a second), the same for every rule
- events, the same for every rule: 10:slow:box-a,90:normal:box-a
- slow = a real fine-tune on that box (the supervisor's job, with --out /night/ft/fault-table); kill = the forecast server's process is killed and its supervisor starts it again
- latency = what the client saw, answered requests only; errors = no answer or not HTTP 200

| policy | reqs | errors | err % | p50 ms | p95 ms | p99 ms | max ms | mean ms | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 1182 | 0 | 0.0 | 277.9 | 2481.4 | 2992.7 | 3187.5 | 747.5 | 100.0 | 0.0 | 0 |  |

p95 by window of the case (requests in brackets). "training" starts at the job's first training step; "box down" ends when the killed server answers again.

| policy | before p95 ms | job starting p95 ms | training p95 ms | between p95 ms |
|---|---:|---:|---:|---:|
| static | 956.2 (89) | 2099.9 (237) | 2857.2 (422) | 462.7 (434) |

**static**: first training step at 40.03 s, job stopped at 94.56 s, killed box down at None s and answering again at None s (the router used it again at None s); the sender was at most 1.8 ms late; 0 unanswered
