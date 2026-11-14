# Inference benchmark, 2026-11-14T17:42:53Z

Device: **NVIDIA GB10** (torch, fp16-autocast), model `wx-tiny-untrained-0.1` hidden=256 layers=4

## Cold start

| torch import ms | model load ms | first inference ms | warm inference ms |
|---:|---:|---:|---:|
| 1247.6 | 447.034 | 880.58 | 5.57 |

## One GPU call per batch (48 h forecast)

| batch rows | p50 ms | p95 ms | p99 ms | rows/s | ms/row | peak GPU MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.93 | 21.96 | 21.96 | 45.6 | 21.9338 | 11.3 |
| 16 | 18.65 | 18.66 | 18.66 | 857.9 | 1.1657 | 11.3 |
| 64 | 18.78 | 18.91 | 18.93 | 3401.7 | 0.294 | 11.4 |

## Serving: baseline (no batching) vs batched

| clients | mode | p50 ms | p95 ms | p99 ms | req/s | mean batch | errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 22.73 | 23.50 | 50.43 | 41.7 | 1.0 | 0 |
| 1 | batched | 22.85 | 23.52 | 23.64 | 43.7 | 1.0 | 0 |
| 16 | baseline | 360.43 | 360.93 | 361.38 | 44.3 | 1.0 | 0 |
| 16 | batched | 20.17 | 92.95 | 92.95 | 416.3 | 16.0 | 0 |

## Precision

| precision | batch rows | p50 ms | rows/s | peak GPU MB | max diff vs fp32 | valid |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 64 | 11.83 | 5401.2 | 22.5 |  |  |
| fp16-autocast | 64 | 18.71 | 3420.4 | 22.6 | 0.28388 | False |
