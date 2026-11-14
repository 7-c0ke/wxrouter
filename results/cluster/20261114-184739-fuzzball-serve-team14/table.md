# Two-box routing benchmark, 2026-11-14T18:47:39Z

Trace: seed 42, 120.0 req/s; steady cases 30.0 s, fault cases 60.0 s.
Fault events: `10:slow:box-a,25:normal:box-a,35:kill:box-a,45:restart:box-a`.

- **box-a**: NVIDIA GB10 (cuda, fp32), node addresses ['10.240.100.9'], GPU GPU-ef16568f-cc74-340a-f52a-46831263cfd4
- **box-b**: NVIDIA GB10 (cuda, fp32), node addresses ['10.240.100.19'], GPU GPU-a54b10ad-647a-9d5b-de98-d141a393fde3
- distinct nodes: **True**

Slow mode (factor 16.0): one 48 h forecast took 11.03 ms idle and 450.95 ms under the GPU load (40.88x).

| case | policy | reqs | errors | p50 ms | p95 ms | p99 ms | ok/s | box A % | box B % | failovers | switches |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_gpu_no_batching | direct | 3539 | 50 | 898.4 | 3151.1 | 6139.9 | 81.9 | 100.0 | 0.0 | 0 | 0 |
| single_gpu_batching | direct | 3539 | 0 | 25.2 | 50.3 | 59.3 | 118.0 | 100.0 | 0.0 | 0 | 0 |
| two_gpu_static | static | 3539 | 0 | 33.2 | 60.0 | 72.3 | 117.9 | 100.0 | 0.0 | 0 | 0 |
| two_gpu_round_robin | round_robin | 3539 | 0 | 19.4 | 40.6 | 51.4 | 117.9 | 50.0 | 50.0 | 0 | 0 |
| two_gpu_least_loaded | least_loaded | 3539 | 0 | 19.7 | 41.0 | 53.0 | 117.9 | 49.2 | 50.8 | 0 | 0 |
| two_gpu_adaptive | adaptive | 3539 | 0 | 31.9 | 62.2 | 83.9 | 117.9 | 41.1 | 58.9 | 0 | 3 |
| two_gpu_balanced | balanced | 3539 | 0 | 22.3 | 47.7 | 61.8 | 117.9 | 48.9 | 51.1 | 0 | 0 |
| fault_static | static | 7202 | 1 | 483.0 | 5913.6 | 8815.7 | 89.2 | 78.4 | 21.6 | 114 | 0 |
| fault_round_robin | round_robin | 7202 | 0 | 237.3 | 3083.6 | 6439.4 | 120.0 | 34.7 | 65.3 | 2 | 0 |
| fault_least_loaded | least_loaded | 7202 | 0 | 24.9 | 55.8 | 678.0 | 119.9 | 27.6 | 72.4 | 1 | 0 |
| fault_adaptive | adaptive | 7202 | 0 | 31.8 | 61.7 | 77.8 | 119.9 | 30.3 | 69.7 | 4 | 4 |
| fault_balanced | balanced | 7202 | 0 | 28.3 | 59.3 | 92.9 | 120.0 | 32.0 | 68.0 | 1 | 0 |

**fault_static**
- box-a slowed 10-25 s: its share went 100.0% -> 92.0% -> 0.0% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 0.0 s after the restart, share in the last 5 s 100.0%

**fault_round_robin**
- box-a slowed 10-25 s: its share went 50.0% -> 49.8% -> 0.9% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 0.0 s after the restart, share in the last 5 s 50.0%

**fault_least_loaded**
- box-a slowed 10-25 s: its share went 50.0% -> 5.7% -> 49.7% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 3.0 s after the restart, share in the last 5 s 48.9%

**fault_adaptive**
- box-a slowed 10-25 s: its share went 29.1% -> 0.0% -> 70.3% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 5.0 s after the restart, share in the last 5 s 100.0%
- switch at 3.07 s: box-a -> box-b (box-b 4ms vs box-a 7ms)
- switch at 29.69 s: box-b -> box-a (back to preferred box-a (10ms vs 18ms))
- switch at 35.82 s: box-a -> box-b (active box down)
- switch at 51.1 s: box-b -> box-a (back to preferred box-a (4ms vs 9ms))

**fault_balanced**
- box-a slowed 10-25 s: its share went 53.8% -> 1.6% -> 62.7% (before, during, after); 0 errors
- box-a stopped at 35 s, restarted at 45 s: 0 errors while down, served again 3.0 s after the restart, share in the last 5 s 60.7%
