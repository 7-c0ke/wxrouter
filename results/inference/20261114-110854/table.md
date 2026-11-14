# Inference benchmark, 2026-11-14T17:08:45Z

Device: **cpu** (torch, fp32), model `wx-tiny-untrained-0.1` hidden=256 layers=4

## Cold start

| torch import ms | model load ms | first inference ms | warm inference ms |
|---:|---:|---:|---:|
| 791.2 | 24.122 | 1.849 | 0.676 |

## One GPU call per batch (48 h forecast)

| batch rows | p50 ms | p95 ms | p99 ms | rows/s | ms/row | peak GPU MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.45 | 2.82 | 3.00 | 400.1 | 2.4991 |  |
| 4 | 3.95 | 4.23 | 4.27 | 1006.8 | 0.9933 |  |
| 16 | 12.92 | 21.94 | 22.75 | 1097.9 | 0.9108 |  |
| 64 | 28.48 | 34.22 | 35.83 | 2224.5 | 0.4495 |  |
| 256 | 68.24 | 91.72 | 101.25 | 3506.4 | 0.2852 |  |

## Serving: baseline (no batching) vs batched

| clients | mode | p50 ms | p95 ms | p99 ms | req/s | mean batch | errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 2.64 | 3.48 | 3.81 | 361.6 | 1.0 | 0 |
| 1 | batched | 2.51 | 3.02 | 3.23 | 387.4 | 1.0 | 0 |
| 8 | baseline | 21.07 | 25.55 | 26.40 | 360.6 | 1.0 | 0 |
| 8 | batched | 6.87 | 9.13 | 9.25 | 1125.3 | 8.0 | 0 |
| 32 | baseline | 82.32 | 96.48 | 107.43 | 378.3 | 1.0 | 0 |
| 32 | batched | 17.09 | 18.39 | 18.39 | 1812.0 | 30.77 | 0 |
