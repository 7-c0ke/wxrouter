# Routing benchmark — 20261114-064640

- trace: seed 42, 987 requests over 40s (~25.0 req/s)
- events: 6:slow:box-a:4,18:normal:box-a,24:kill:box-b,30:restart:box-b
- latency = what the client saw, successful requests only; errors = non-200 or no answer

| policy | reqs | errors | err % | p50 ms | p95 ms | p99 ms | max ms | mean ms | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive | 987 | 0 | 0.0 | 175.4 | 673.5 | 2928.0 | 4107.8 | 287.6 | 61.4 | 38.6 | 5 | 2 |

**adaptive switches:**
- 9.12s box-a → box-b (box-b 62ms vs box-a 1005ms)
- 26.15s box-b → box-a (active box down)
