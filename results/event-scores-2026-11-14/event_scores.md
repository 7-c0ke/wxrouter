# Yes/no scores before and after the fine-tune

83 test forecasts, started 2026-10-01T00 to 2026-11-11T00 (42 days); 28,800 grid points each, weighted by area; truth: the NOAA GFS analysis at the valid time. Made by weather/event_scores.py.

The model's own score is the RMSE (the fine-tune report). These scores answer a yes/no question about the same forecasts. In brackets: the 90% interval of the change from a paired bootstrap over the start days.

## 2-m temperature below 0 C, 24 hours ahead

The event is true in 14.65% of the cases.

| score | base model | tuned model | change | "nothing changes" |
|---|---:|---:|---|---:|
| accuracy | 98.92% | 98.90% | -0.0% (-0.0% to +0.0%) | 98.01% |
| precision | 96.18% | 96.50% | +0.3% (+0.2% to +0.5%) | 93.52% |
| recall | 96.45% | 96.01% | -0.5% (-0.6% to -0.3%) | 92.87% |
| F1 | 96.31% | 96.25% | -0.1% (-0.1% to -0.0%) | 93.19% |
| wrong answers (1 - accuracy) | 1.08% | 1.10% | +1.3% (-0.3% to +2.9%) | 1.99% |

## 2-m temperature below 0 C, 72 hours ahead

The event is true in 14.86% of the cases.

| score | base model | tuned model | change | "nothing changes" |
|---|---:|---:|---|---:|
| accuracy | 98.47% | 98.35% | -0.1% (-0.2% to -0.1%) | 97.29% |
| precision | 94.50% | 94.72% | +0.2% (-0.1% to +0.5%) | 91.76% |
| recall | 95.25% | 94.14% | -1.2% (-1.3% to -1.0%) | 89.85% |
| F1 | 94.87% | 94.43% | -0.5% (-0.6% to -0.3%) | 90.79% |
| wrong answers (1 - accuracy) | 1.53% | 1.65% | +7.9% (+5.4% to +10.6%) | 2.71% |

## 2-m temperature 5 C or more away from the starting analysis, 24 hours ahead

The event is true in 3.70% of the cases.

| score | base model | tuned model | change | "nothing changes" |
|---|---:|---:|---|---:|
| accuracy | 97.05% | 97.56% | +0.5% (+0.5% to +0.6%) | 96.30% |
| precision | 61.74% | 69.37% | +12.4% (+11.3% to +13.6%) | n/a |
| recall | 52.95% | 61.12% | +15.4% (+14.0% to +16.9%) | 0.00% |
| F1 | 57.01% | 64.99% | +14.0% (+13.0% to +15.2%) | 0.00% |
| wrong answers (1 - accuracy) | 2.95% | 2.44% | -17.5% (-18.5% to -16.7%) | 3.70% |

## 2-m temperature 5 C or more away from the starting analysis, 72 hours ahead

The event is true in 7.55% of the cases.

| score | base model | tuned model | change | "nothing changes" |
|---|---:|---:|---|---:|
| accuracy | 94.29% | 94.62% | +0.3% (+0.3% to +0.4%) | 92.45% |
| precision | 61.95% | 63.50% | +2.5% (+2.0% to +3.1%) | n/a |
| recall | 63.17% | 67.47% | +6.8% (+6.0% to +7.7%) | 0.00% |
| F1 | 62.56% | 65.43% | +4.6% (+4.1% to +5.1%) | 0.00% |
| wrong answers (1 - accuracy) | 5.71% | 5.38% | -5.7% (-6.4% to -4.9%) | 7.55% |

## Check against the fine-tune report

| 2-m temperature RMSE (K) | base | tuned | "nothing changes" |
|---|---:|---:|---:|
| 24 h | 1.350 | 1.121 | 2.025 |
| 72 h | 1.754 | 1.671 | 2.812 |
