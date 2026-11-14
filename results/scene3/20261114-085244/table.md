# Routing benchmark — 20261114-085244

- trace: seed 42, 671 requests over 45s (~15.0 req/s)
- events: 8:slow:box-b:2,22:normal:box-b
- latency = what the client saw, successful requests only; errors = non-200 or no answer

| policy | reqs | errors | err % | p50 ms | p95 ms | p99 ms | max ms | mean ms | box A % | box B % | failovers | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive | 671 | 0 | 0.0 | 105.0 | 568.6 | 1167.6 | 1499.1 | 173.1 | 31.0 | 69.0 | 0 | 2 |

**adaptive switches:**
- 12.12s box-b → box-a (box-a 57ms vs box-b 96ms)
- 27.14s box-a → box-b (back to preferred box-b (67ms vs 60ms))
