# AFNO behind the inference engine, 2026-11-14T18:25:22Z

GPU: NVIDIA GB10. Weights `afno-19m-65f9f3bb3270`, start time 2026-11-14T12.

Same requests to both servers: 128 forecast points compared, largest difference 0.0 C, 0.01 m/s, 0.01 hPa -> agree: **True**.

One 48 h forecast per request (8 model steps):

| server | clients | req/s | p50 ms | p95 ms | p99 ms | GPU call p50 ms | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| team | 1 | 8.29 | 119.92 | 126.56 | 129.92 | 66.0 | 0 |
| team | 4 | 14.46 | 269.94 | 308.26 | 324.12 | 64.2 | 0 |
| team | 16 | 14.33 | 1099.9 | 1157.39 | 1176.51 | 64.45 | 0 |
| engine | 1 | 12.94 | 77.32 | 78.86 | 79.91 | 66.53 | 0 |
| engine | 4 | 16.45 | 240.41 | 263.77 | 274.71 | 166.28 | 0 |
| engine | 16 | 13.5 | 1159.55 | 1326.78 | 1328.98 | 981.7 | 0 |
