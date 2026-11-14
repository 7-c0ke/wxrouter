# Inference benchmark, 2026-11-14T17:44:19Z

Device: **NVIDIA GB10** (torch, fp32), model `wx-tiny-untrained-0.1` hidden=256 layers=4

## Cold start

| torch import ms | model load ms | first inference ms | warm inference ms |
|---:|---:|---:|---:|
| 1282.2 | 430.113 | 522.681 | 2.537 |

## One GPU call per batch (48 h forecast)

| batch rows | p50 ms | p95 ms | p99 ms | rows/s | ms/row | peak GPU MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.95 | 10.02 | 10.04 | 100.6 | 9.9403 | 11.1 |
| 4 | 11.29 | 11.44 | 11.55 | 354.2 | 2.8229 | 11.2 |
| 16 | 11.24 | 11.27 | 11.30 | 1423.8 | 0.7023 | 11.2 |
| 64 | 11.81 | 11.86 | 11.88 | 5420.1 | 0.1845 | 11.4 |
| 256 | 10.45 | 10.52 | 10.53 | 24499.9 | 0.0408 | 12.1 |

## Serving: baseline (no batching) vs batched

| clients | mode | p50 ms | p95 ms | p99 ms | req/s | mean batch | errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 10.78 | 11.26 | 11.48 | 93.2 | 1.0 | 0 |
| 1 | batched | 10.53 | 11.38 | 11.56 | 93.4 | 1.0 | 0 |
| 8 | baseline | 85.33 | 87.97 | 89.05 | 93.7 | 1.0 | 0 |
| 8 | batched | 12.42 | 12.80 | 13.05 | 650.1 | 8.0 | 0 |
| 32 | baseline | 336.74 | 342.51 | 343.24 | 94.9 | 1.0 | 0 |
| 32 | batched | 12.66 | 13.71 | 13.72 | 2401.1 | 30.77 | 0 |

## Precision

| precision | batch rows | p50 ms | rows/s | peak GPU MB | max diff vs fp32 | valid |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 256 | 10.55 | 24169.8 | 23.2 |  |  |
| fp16-autocast | 256 | 19.58 | 13071.6 | 23.1 | 0.36538 | False |
