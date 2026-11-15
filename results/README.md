# Results: what is in this folder, and where each thing was measured

Everything here was measured on Nov 14, 2026, unless a row gives another date. Four places appear:

- **Event boxes**: the event's two NVIDIA GB10 machines (Fuzzball nodes CIQ-CHI-SPARK27 and
  CIQ-CHI-SPARK28) with the real weather model, the LLM and the router.
- **Event machines, stand-in model**: the same machines with `wx-tiny-untrained-0.1`, a small model with
  random weights. It makes the GPU work. Its output is not a forecast.
- **One GB10**: a team member's own DGX Spark. It has one GPU, so both "boxes" share it.
- **Laptop, mock boxes**: no GPU. The boxes are [mock_server.py](../mock_server.py), with made-up timings.

## Start here

1. [event-boxes-2026-11-14.md](event-boxes-2026-11-14.md): the main record. 15 numbered sections with
   method, numbers and caveats, and a list of what was not measured.
2. [event-boxes-2026-11-14/finetune_report.md](event-boxes-2026-11-14/finetune_report.md): what a 12-minute
   fine-tune on an event box gained, and where it lost.
3. [figures/skill-by-lead-time-2020.png](figures/skill-by-lead-time-2020.png): the weather model's error
   by lead time on a year it never saw. Its numbers: [figures/data/scores_overnight.json](figures/data/scores_overnight.json).
4. [one-gb10-2026-11-14-night/README.md](one-gb10-2026-11-14-night/README.md): checks made that night on
   one GB10, with a replay of the one event-box result we could not explain.

## Every folder and file

| Path | What it holds | Where it was measured | Headline number, with its caveat |
|---|---|---|---|
| [event-boxes-2026-11-14.md](event-boxes-2026-11-14.md) | The main record: speed, kills under traffic, five routing rules, a fine-tune beside serving, the model swap, the LLM, what Fuzzball did | Event boxes, 11:45 AM to 5:11 PM Central | 0 of 5,910 requests failed under five routing rules, each with a real fine-tune on box A and a kill of box B's server (section 10). One run per rule. The "no routing" row holds 45 seconds in which box A was slow for a reason we have not found. |
| [event-boxes-2026-11-14/](event-boxes-2026-11-14/) | Raw files and small scripts behind sections 6 and 8 to 15 of that record: [the five-rule table](event-boxes-2026-11-14/fault-table-20261114-214821/table.md) with a row for every request, the fine-tune report, the swap logs, the chat checks | Event boxes, 2:39 to 5:11 PM | While box A fine-tuned, the same 534 requests had a p95 of 3.2 s with no routing and 0.32 s with the least-loaded rule. One run. Sections 1 to 5 and 7 have no file here: their numbers were read from the terminal. |
| [event-scores-2026-11-14/](event-scores-2026-11-14/README.md) | Accuracy, precision, recall and F1 of the base and the tuned model for two yes/no questions. "Event" here means a weather event, not the event boxes | One GB10, 9:45 PM. The tuned weights are the ones from the event box | For "will the 2 m temperature change by 5 °C or more within a day?", F1 rose from 57.0% to 65.0% on 83 held-out forecasts. For "frost: yes or no" the tuned model is no better (F1 -0.1% at one day, -0.5% at three). One run, six weeks of one season. |
| [one-gb10-2026-11-14-night/](one-gb10-2026-11-14-night/README.md) | A chat question followed through every hop, what the page's "Demo traffic" does to a GPU, a replay of the "no routing" case, a test of fast and slow CPU cores | One GB10, 9 to 11 PM | The replay: p95 2.3 s, 0 failed, and the 45-second slow spell of the event box did not come back. Its cause is still not known. One GPU, so nothing here is about routing between two machines. |
| [chat-follow-ups-2026-11-14/](chat-follow-ups-2026-11-14/README.md) | A fault in the chat and its fix: a follow-up question was answered by the LLM without asking the weather model. Also what three code reads found, and the known limits | Before the fix: event boxes, 5:45 PM (the three `event-boxes-*` files) and one GB10. After the fix: one GB10 only, 9:45 PM to 1:15 AM on Nov 15 | Follow-ups answered by the weather model: 0 of 10 on the event boxes before the fix; 9 of 9 in twelve of thirteen runs with the fix, 8 of 9 in one. The fix has not run on the event boxes. |
| [afno/](afno/20261114-182522-fuzzball-afno/table.md) | The trained weather model served two ways: by the team's forecast server and by a teammate's inference engine | One event machine, real model, 12:25 PM (the folder stamp is UTC) | 128 forecast points agree (largest difference 0.0 °C, 0.01 m/s, 0.01 hPa). One 48-hour forecast: 77 ms through the engine, 120 ms through the team's server. Batching did not raise throughput for this model: 13.5 and 14.3 requests a second with 16 clients. One run. |
| [cluster/](cluster/) | The routing rules with the inference engine on two GPUs, at 120 requests a second: [a first run](cluster/20261114-182549-fuzzball-demo/table.md) and [one with a heavier slowdown](cluster/20261114-184739-fuzzball-serve-team14/table.md) | Event machines, stand-in model, 12:25 and 12:47 PM (stamps in UTC) | Heavier run, box A made 41 times slower for 15 s and later stopped for 10 s: p95 56 to 62 ms with least loaded, adaptive or balanced, against 3.1 s with round robin and 5.9 s with static, which also lost 1 of 7,202 requests. The slowdown is a made-up GPU load, not a fine-tune, and the model is the stand-in. In the first run the slowdown was only 1.25 times and the rules do not separate. One run each. |
| [inference/](inference/) | Benchmarks of the inference engine alone: [the full run](inference/20261114-174419-fuzzball-bench/table.md), [a smoke run](inference/20261114-174253-fuzzball-smoke/table.md), [a CPU run](inference/20261114-110854/table.md), and [report.html](inference/report.html) with charts of these and of `cluster/` and `afno/` | One event machine, stand-in model, 11:42 and 11:44 AM. The CPU run is from a laptop (its table says "cpu") | With 32 clients, batching raised throughput from 95 to 2,401 requests a second and cut p95 from 343 ms to 14 ms. This is the stand-in model: a GPU call takes 10 ms. The real model needs about 65 ms on the GPU for a 48-hour forecast and gains nothing from batching (see `afno/`). One run. |
| [all/](all/20261114-064333/table.md) | Four routing rules against one seeded trace (987 requests, 25 a second), box A slowed, then box B killed | Laptop, mock boxes | 0 errors under every rule; p95 408 ms with least loaded against 5,294 ms with static. Mock timings: this says how the rules behave, not how fast a GPU is. |
| [window1/](window1/20261114-064640/table.md) | The same trace with the adaptive rule alone, after its window was shortened from 3 s to 1 s (the setting is named in [docs/PROJECT.md](../docs/PROJECT.md), not in the files) | Laptop, mock boxes | p95 674 ms, against 4,425 ms for adaptive in `all/`. One run each; mock runs differ from run to run. |
| [scene3/](scene3/20261114-085244/table.md) | A rehearsal of "a box starts training": adaptive with box B as its home box, box B made 2 times slower from 8 to 22 s, 15 requests a second | Laptop, mock boxes | 2 switches (away from box B, then back), 0 errors, p95 569 ms. The slowdown is the mock's, not a real job. |
| [figures/](figures/) | The figures of the main README, the data behind them and the two scripts that made them | See the next table | |

The result files name no person. By the repository's history, `all/`, `window1/` and `scene3/` came with the
laptop test bed, `inference/`, `cluster/` and `afno/` with the inference engine, and the rest with the weather
model and the stack. The team table in the [main README](../README.md#team) says who built each.

## The figures

| Figure | What it shows | Where its numbers come from |
|---|---|---|
| [skill-by-lead-time-2020.png](figures/skill-by-lead-time-2020.png) | Model error from 6 to 120 hours ahead against two simple baselines. At 3 days, 500 hPa height: 302 m²/s², against 956 for "same as now" and 825 for the long-term average | 362 forecasts of 2020 scored against ERA5 by [weather/score.py](../weather/score.py) on one GB10, early on Nov 14 (the file records the GPU run and the time, not the machine's name). One training run. The file also holds the scores of a model of the same recipe trained on Nov 10, before the event; the scorer checks itself against them, and that model is not served. |
| [cold-wave-2019.png](figures/cold-wave-2019.png), [compact](figures/cold-wave-2019-compact.png) | The cold wave of January 2019 as it happened and as the model forecast it 72 hours earlier. Chicago: -22 °C against -24 °C | The event model run from the ERA5 state of 2019-01-27 06 UTC by [make_case_data.py](figures/make_case_data.py), on a DGX Spark on Nov 15. One forecast of one event: a picture, not a score. |
| [finetune-gain.png](figures/finetune-gain.png) | Change in error after the fine-tune: 11 to 19% lower one day ahead; three days ahead one of three scores is 2% worse | Event boxes: section 8 of the main record and `finetune_report.md`. 83 held-out forecasts, one run, the best of three settings. |
| [router-p95-while-training.png](figures/router-p95-while-training.png) | p95 wait of the same 534 requests under each routing rule while box A fine-tunes: 3.17 s to 0.32 s | Event boxes: section 10 of the main record. One run per rule. |

[make_figures.py](figures/make_figures.py) draws all of them. The first two read [figures/data/](figures/data/);
the two bar charts use numbers typed in from the main record. The map outlines are Natural Earth files
(public domain).

## How to read these numbers

- **Every event-box number is from one run.** None was repeated. Where a number depends on the run's
  conditions (other pages reading the stack, a slow spell with no known cause), the main record says so in
  the same section.
- **Results from one GB10 are marked.** They show that the software does something, with the real model and
  the real LLM. They do not show how two machines share work.
- **Laptop results use mock boxes with made-up timings.** They say nothing about real GPUs. The same is
  true of the stand-in model's speed in `inference/` and `cluster/`: it is not the weather model's.
- **Times in the text are US Central.** Folder names made on the event machines are stamped in UTC
  (`fault-table-20261114-214821` ran at 3:48 PM Central). The laptop folders carry the laptop's own clock.
- Percent changes are relative to the base value. For an error, lower is better. For precision, recall
  and F1, higher is better.
