# Routing benchmark — 20261114-064333

- trace: seed 42, 987 requests over 40s (~25.0 req/s)
- events: 6:slow:box-a:4,18:normal:box-a,24:kill:box-b,30:restart:box-b
- latency = what the client saw, successful requests only; errors = non-200 or no answer

| policy | reqs | errors | err % | p50 ms | p95 ms | p99 ms | max ms | mean ms | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 987 | 0 | 0.0 | 205.9 | 5294.3 | 5809.8 | 6063.1 | 825.3 | 76.1 | 23.9 | 77 |  |
| round_robin | 987 | 0 | 0.0 | 127.3 | 3348.5 | 4763.6 | 6136.8 | 575.1 | 57.9 | 42.1 | 7 |  |
| least_loaded | 987 | 0 | 0.0 | 106.5 | 407.8 | 1103.5 | 4071.8 | 176.3 | 51.6 | 48.4 | 3 |  |
| adaptive | 987 | 0 | 0.0 | 195.9 | 4425.1 | 5969.0 | 6268.9 | 639.1 | 63.5 | 36.5 | 42 | 2 |

**adaptive switches:**
- 11.62s box-a → box-b (box-b 58ms vs box-a 161ms)
- 26.14s box-b → box-a (active box down)
