# Accuracy, precision, recall and F1 before and after the fine-tune (Nov 14, 2026)

A teammate asked for "accuracy, precision, recall and F1 before and after tuning, with the percentage
improvement". Three things to know before using the numbers below.

1. **Which model.** The weather model is NVIDIA PhysicsNeMo's AFNO (19 million parameters). It is the model
   that was fine-tuned. Nemotron is the LLM behind the chat. It was not tuned, so it has no "before" and "after".
2. **Its own score is an error, not an accuracy.** The model forecasts numbers (temperature, pressure, wind
   on a global grid). The standard score for that is the RMSE, and that is the result to quote first:
   `results/event-boxes-2026-11-14/finetune_report.md`. One day ahead the error fell by 11 to 23%.
3. **Accuracy, precision, recall and F1 exist only for a yes/no question.** So `weather/event_scores.py` asks
   two yes/no questions of the same test forecasts. Both questions and their thresholds were fixed before the
   first run and were not changed afterwards.

## The two questions

| | question, at every grid point | how often the answer is yes |
|---|---|---|
| frost | Will the 2-m temperature be below 0 °C? | 15% of the cases |
| big change | Will the 2-m temperature be 5 °C or more away from its value in the starting analysis? | 4% a day ahead, 8% three days ahead |

## The scores

83 test forecasts from 42 days (Oct 1 to Nov 11, 2026) that the fine-tune never trained on and never chose
its setting on. Each of the 28,800 grid points of each forecast is one case (2,390,400 cases per lead time),
weighted by area. The truth is the NOAA GFS analysis at the valid time. In brackets: the 90% interval of the
change, from a paired bootstrap over the 42 start days.

**A change of 5 °C or more, one day ahead**

| score | base model | tuned model | change |
|---|---:|---:|---|
| accuracy | 97.05% | 97.56% | +0.5% (+0.5% to +0.6%) |
| precision | 61.74% | 69.37% | +12.4% (+11.3% to +13.6%) |
| recall | 52.95% | 61.12% | +15.4% (+14.0% to +16.9%) |
| F1 | 57.01% | 64.99% | +14.0% (+13.0% to +15.2%) |
| wrong answers | 2.95% | 2.44% | -17.5% (-18.5% to -16.7%) |

**A change of 5 °C or more, three days ahead**

| score | base model | tuned model | change |
|---|---:|---:|---|
| accuracy | 94.29% | 94.62% | +0.3% (+0.3% to +0.4%) |
| precision | 61.95% | 63.50% | +2.5% (+2.0% to +3.1%) |
| recall | 63.17% | 67.47% | +6.8% (+6.0% to +7.7%) |
| F1 | 62.56% | 65.43% | +4.6% (+4.1% to +5.1%) |
| wrong answers | 5.71% | 5.38% | -5.7% (-6.4% to -4.9%) |

**Frost, one day and three days ahead**

| score | base, 1 day | tuned, 1 day | change | base, 3 days | tuned, 3 days | change |
|---|---:|---:|---|---:|---:|---|
| accuracy | 98.92% | 98.90% | -0.0% | 98.47% | 98.35% | -0.1% |
| precision | 96.18% | 96.50% | +0.3% | 94.50% | 94.72% | +0.2% |
| recall | 96.45% | 96.01% | -0.5% | 95.25% | 94.14% | -1.2% |
| F1 | 96.31% | 96.25% | -0.1% (-0.1% to -0.0%) | 94.87% | 94.43% | -0.5% (-0.6% to -0.3%) |

`event_scores.md` has every number, with the score of the forecast "nothing changes" beside it.

## What it says, in plain words

- **The fine-tune helped where forecasting is hard.** Asked whether the temperature will change by 5 °C or
  more by tomorrow, the tuned model catches 61% of those cases instead of 53%, and is right 69% of the times
  it says yes instead of 62%. F1 rises from 57.0% to 65.0%, a gain of 14%. Three days ahead the gain is 4.6%.
  The forecast "nothing changes" never says yes to this question, so its recall is 0.
- **It did not help with "frost: yes or no".** Both models answer that correctly for almost 99% of the cases,
  because most of the globe is far from 0 °C on any day. The tuned model is equal a day ahead (F1 -0.1%) and
  slightly worse three days ahead (F1 -0.5%). We report it because we said beforehand that we would ask it.
- **Accuracy alone says little here.** When an event is rare, a forecast that always says "no" is already
  96% accurate for the big-change question. Precision, recall and F1 are the scores to read.

## How it was run

- `weather/event_scores.py`, on one NVIDIA GB10 (a separate DGX Spark), Nov 14, 2026, 9:45 PM; 37 s.
- Base weights `afno_event.pt` (SHA-256 `65f9f3bb3270...`). Tuned weights `afno_tuned_641de834.pt`
  (`641de8342da7...`): the fine-tune that ran on an event box on Nov 14 at 2:46 PM.
- NOAA GFS analyses from Nov 14, 2025 to Nov 14, 2026; the same split as `weather/finetune.py`.
- **Check:** the script also computes the 2-m temperature RMSE of the same forecasts. It gives 1.350 K (base)
  and 1.121 K (tuned) one day ahead, and 1.754 K and 1.671 K three days ahead: the numbers of the fine-tune
  report from the event box, to three decimals.
- One run. The intervals describe the 83 forecasts of these six weeks, not another season.

```bash
python weather/event_scores.py --base BUNDLE/weights/afno_event.pt --tuned afno_tuned_641de834.pt \
    --analyses GFS_YEAR --stats BUNDLE --out results/event-scores-2026-11-14
```
