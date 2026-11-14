# Two-box routing benchmark, 2026-11-14T18:26:16Z

Trace: seed 42, 120.0 req/s; steady cases 30.0 s, fault cases 60.0 s.
Fault events: `10:slow:box-a,25:normal:box-a,35:kill:box-a,45:restart:box-a`.

- **box-a**: NVIDIA GB10 (cuda, fp32), node addresses ['10.240.100.7'], GPU GPU-ef16568f-cc74-340a-f52a-46831263cfd4
- **box-b**: NVIDIA GB10 (cuda, fp32), node addresses ['10.240.100.18'], GPU GPU-a54b10ad-647a-9d5b-de98-d141a393fde3
- distinct nodes: **True**

Slow mode (factor 4.0): one 48 h forecast took 11.0 ms idle and 13.71 ms under the GPU load (1.25x).

| case | policy | reqs | errors | p50 ms | p95 ms | p99 ms | ok/s | box A % | box B % | failovers | switches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_gpu_no_batching | direct | 3539 | 29 | 906.1 | 3002.7 | 5023.5 | 78.2 | 100.0 | 0.0 | 0 | 0 |
| single_gpu_batching | direct | 3539 | 0 | 25.9 | 50.9 | 62.0 | 117.9 | 100.0 | 0.0 | 0 | 0 |
| two_gpu_static | static | 3539 | 0 | 32.1 | 60.7 | 77.2 | 117.9 | 100.0 | 0.0 | 0 | 0 |
| two_gpu_round_robin | round_robin | 3539 | 0 | 20.9 | 42.7 | 53.6 | 117.9 | 50.0 | 50.0 | 0 | 0 |
| two_gpu_least_loaded | least_loaded | 3539 | 0 | 21.2 | 41.5 | 51.6 | 117.9 | 50.2 | 49.8 | 0 | 0 |
| two_gpu_adaptive | adaptive | 3539 | 0 | 30.0 | 56.7 | 68.4 | 117.9 | 41.0 | 59.0 | 0 | 3 |
| two_gpu_balanced | balanced | 3539 | 0 | 21.4 | 45.1 | 56.8 | 117.9 | 49.4 | 50.6 | 0 | 0 |
| fault_static | static | 7202 | 0 | 35.1 | 64.1 | 78.8 | 119.9 | 77.1 | 22.9 | 3 | 0 |
| fault_round_robin | round_robin | 7202 | 0 | 23.2 | 50.5 | 62.6 | 119.9 | 38.4 | 61.6 | 1 | 0 |
| fault_least_loaded | least_loaded | 7202 | 0 | 22.8 | 49.3 | 60.1 | 119.9 | 38.6 | 61.4 | 2 | 0 |
| fault_adaptive | adaptive | 7202 | 0 | 34.1 | 67.1 | 109.4 | 119.9 | 41.6 | 58.4 | 3 | 6 |
| fault_balanced | balanced | 7202 | 0 | 26.9 | 54.7 | 71.6 | 119.9 | 40.3 | 59.7 | 1 | 0 |

**fault_static**
- box-a slowed 10-25 s: its share went 100.0% -> 100.0% -> 100.0% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 3.0 s after the restart, share in the last 5 s 100.0%

**fault_round_robin**
- box-a slowed 10-25 s: its share went 50.0% -> 50.0% -> 49.9% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 4.0 s after the restart, share in the last 5 s 50.0%

**fault_least_loaded**
- box-a slowed 10-25 s: its share went 50.0% -> 48.4% -> 50.6% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 3.0 s after the restart, share in the last 5 s 50.6%

**fault_adaptive**
- box-a slowed 10-25 s: its share went 24.2% -> 76.3% -> 34.2% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 5.0 s after the restart, share in the last 5 s 100.0%
- switch at 2.58 s: box-a -> box-b (box-b 4ms vs box-a 11ms)
- switch at 12.35 s: box-b -> box-a (back to preferred box-a (6ms vs 11ms))
- switch at 22.67 s: box-a -> box-b (box-b 4ms vs box-a 10ms)
- switch at 32.45 s: box-b -> box-a (back to preferred box-a (4ms vs 8ms))
- switch at 35.54 s: box-a -> box-b (active box down)
- switch at 50.56 s: box-b -> box-a (back to preferred box-a (4ms vs 10ms))

**fault_balanced**
- box-a slowed 10-25 s: its share went 53.4% -> 41.9% -> 51.9% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 3.0 s after the restart, share in the last 5 s 65.3%
