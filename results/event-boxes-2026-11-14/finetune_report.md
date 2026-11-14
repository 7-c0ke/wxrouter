### Fine-tuning the weather model on GFS analyses: scores on held-out weeks

Base weights: `/wx/bundle/weights/afno_event.pt`. Fine-tuned for 300 steps (batch 16, 4 fed-back steps, peak learning rate 3e-05) on 1212 training samples from GFS analyses between 2025-11-14T00 and 2026-09-13T18.
tuned.pt holds the weights after step 300, the best of 3 validation checks (validation score 0.8508; the base model is 1.0, lower is better).

Test: 83 forecasts started at 00 and 12 UTC from 2026-10-01T00 to 2026-11-11T00, each run to 72 h and compared with the GFS analysis at every valid time.

| score | 6 h | 24 h | 48 h | 72 h |
|---|---|---|---|---|
| Z500 RMSE (m2/s2) | 62.45 -> 48.33 (-22.6%) | 125.08 -> 111.25 (-11.1%) | 213.30 -> 209.95 (-1.6%) | 316.74 -> 323.02 (+2.0%) |
| T850 RMSE (K) | 0.952 -> 0.686 (-27.9%) | 1.194 -> 0.974 (-18.5%) | 1.485 -> 1.342 (-9.6%) | 1.827 -> 1.781 (-2.5%) |
| T2M RMSE (K) | 1.097 -> 0.856 (-22.0%) | 1.350 -> 1.121 (-16.9%) | 1.545 -> 1.396 (-9.6%) | 1.754 -> 1.671 (-4.7%) |
| all 20 channels RMSE (z-units) | 0.1587 -> 0.1284 (-19.1%) | 0.2552 -> 0.2194 (-14.1%) | 0.3419 -> 0.3173 (-7.2%) | 0.4247 -> 0.4124 (-2.9%) |
| O'Hare 2-m temp. error (C) | 0.83 -> 0.75 (-9.7%) | 1.13 -> 0.86 (-23.4%) | 1.27 -> 1.19 (-6.5%) | 1.63 -> 1.58 (-3.1%) |
| persistence: Z500 RMSE (m2/s2) | 245.73 | 620.74 | 842.82 | 932.69 |
| persistence: T850 RMSE (K) | 1.687 | 3.080 | 3.959 | 4.208 |
| persistence: T2M RMSE (K) | 2.625 | 2.025 | 2.585 | 2.812 |

Each cell: base -> tuned (change); lower is better.
RMSE is weighted by cos(latitude). O'Hare is the mean absolute error of 2-m temperature at Chicago O'Hare. Persistence is the forecast "nothing changes", for scale.

In words (in brackets: the change and its 90% interval):

- Z500 at 24 h: **better** (-11.1%; -12.3% to -9.8%)
- Z500 at 72 h: **worse** (+2.0%; +1.0% to +3.0%)
- T850 at 24 h: **better** (-18.5%; -19.7% to -17.2%)
- T850 at 72 h: **better** (-2.5%; -3.6% to -1.5%)
- T2M at 24 h: **better** (-16.9%; -17.6% to -16.2%)
- T2M at 72 h: **better** (-4.7%; -5.5% to -3.8%)

The tuned model is better than the base model in some comparisons and worse in others (of 6 comparisons, 5 better, 1 worse, 0 no clear change). Read the table before using tuned.pt.

"Better" means the whole interval is below zero, "worse" that it is above zero, and "no clear change" that it contains zero. The intervals come from a paired bootstrap: the 42 test days were drawn again with replacement 1000 times, a day's forecasts staying together.

How the test weeks were kept apart: the analyses were split by date before anything ran, training used only those up to 2026-09-13T18, the 14 days before 2026-10-01T00 were used only to pick the best weights, and the weeks from 2026-10-01T00 on were scored once, at the end.

Written by weather/finetune.py on 2026-11-14 (cuda, PyTorch 2.11.0a0+eb65b36914.nv26.02). Every number is in finetune_report.json.
