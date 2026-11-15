# Two questions from the team, checked on one GB10 (Nov 14, 2026, 9 PM)

The event boxes were out of reach that evening. These checks ran on one NVIDIA GB10 (a team member's own
DGX Spark) with the code of `main` (commit `23fc90d`), started the way `fuzzball/30-real-stack.fz` starts
it: two supervisors, two real forecast servers (weights `65f9f3bb3270`, the NOAA analysis of Nov 14,
18 UTC), `router.py`, the second router, the page, and NVIDIA Nemotron on llama.cpp. Both "boxes" share the
one GPU here, so nothing below says anything about routing between two machines. Every number is from one run.

## 1. Does a chat question reach the weather model?

**Yes.** One question, followed through every hop (`night_proof.py chat`, 8:58 PM):

| hop | what was seen |
|---|---|
| The question | "Weather in Chicago for the next 48 hours?" |
| Tools the assistant ran | `geocode`, then `get_forecast` (lat 41.88, lon -87.63, 48 hours) |
| The router | its count of forwarded forecasts went from 0 to 1 |
| The forecast server | answered as `box-b`, `model_version afno-19m-65f9f3bb3270`, in 68 ms |
| The same request sent straight to the router | 62.9 ms of model run on the GPU (8 model steps of 6 hours) |
| The card against that direct answer | 48 hourly values, largest difference 0.0 °C |
| The reply | "Chicago: 2.8 °C to 9.7 °C, wind up to 19 km/h, served by box-b." |

Three things show that the numbers are the model's and not the LLM's:

1. **The same answer on another machine.** The event boxes gave "2.8 °C to 9.7 °C, wind up to 19 km/h" for
   this question that afternoon. Same weights and same starting analysis, a different machine, the same numbers.
2. **Another starting analysis, other numbers.** From 18 UTC: 2.8 to 9.7 °C. From 12 UTC: 2.8 to 8.4 °C.
   From 00 UTC: 1.9 to 8.1 °C.
3. **Other weights, other numbers.** With the fine-tuned weights in place the same question was answered by
   `afno-19m-641de8342da7` with 2.2 to 8.7 °C.

**One exception, found the same day and fixed on the branch `fix/chat-answers-from-the-model`:** a
follow-up question in the same chat ("And what about New York?"). With the code of `main` the LLM answered
it without asking the model: 0 of 9 follow-ups had a forecast, and the numbers were its own
(`proof-followups-main.json`; on the event boxes at 5:45 PM: 0 of 10). The page shows no weather card for
such an answer. With the fix: 9 of 9, in five runs.

## 2. What is "Demo traffic"?

It is a load generator inside the page server (`LoadGen` in `ui/server.py`). The slider sets a rate. At
that rate, at random moments, the page server sends a forecast request to the router's `/v1/forecast`: a
random place on the globe, and a lead time of 6 to 24 hours (60% of the requests), 25 to 72 hours (32%) or
73 to 120 hours (8%). The router picks a box by its rule, and that box's forecast server runs the weather
model on its GPU: one model step for every 6 hours, 1 to 20 steps a request.

So the **requests are made up**, and the **work is real**: every one is a forecast computed by the model,
routed by the router and timed. Nothing is replayed or cached. It runs until someone presses Off, also after
the browser that started it is closed.

Measured with the GPU's own counter (`nvidia-smi`, twice a second), 20 to 30 s for each setting:

| setting | requests the router forwarded | forecasts the two servers computed | GPU busy, mean | p50 wait | p95 wait | errors |
|---|---:|---:|---:|---:|---:|---:|
| Off | 0 | 94 (health probes) | 32% | | | 0 |
| Light, 8 a second | 242 (8.1 a second) | 388 | 54% | 79 ms | 336 ms | 0 |
| Busy, 15 a second | 428 (14.3 a second) | 564 | 63% | 251 ms | 662 ms | 0 |
| Heavy, 25 a second | 626 (20.9 a second) | 652 | 89% | 5.7 s | 6.6 s | 0 |

- The GPU gets busier with every step, and at "Heavy" one GPU cannot keep up: it finishes 21 requests a
  second of the 25 sent, and the wait grows to seconds. That is real queueing on a real GPU.
- On the event stack each box has a GPU of its own, so the same setting loads each box about half as much.
- "Off" is not zero: the router checks each box with a small forecast twice a second, the page's recorder
  every two seconds, and the second router probes too. Here that kept the one GPU 32% busy by `nvidia-smi`'s
  count (the two servers and every probe share it).
- 5 seconds after "Off" the router had forwarded 4 more requests (the queue emptying) and then none.

## 3. The five-rule test's "no routing" case, replayed (10:38 PM)

Why: on the event boxes the static case of `bench/fault_table.py` had a p95 of 26.9 s, and the rows show
that most of it comes from 45 seconds in which box A was five times slower than normal, which the fine-tune
does not explain (`results/event-boxes-2026-11-14.md`, section 10). The question was whether the software
does that by itself.

The same script, trace (seed 42, 1,182 requests in 150 s) and fine-tune, under the static rule, without the
kill of box B (here both servers share one GPU, so a restart of "box B" would disturb "box A"):

```bash
python bench/fault_table.py --rules static --events 10:slow:box-a,90:normal:box-a ...
```

| | event box A (3:48 PM) | one GB10 (10:38 PM) |
|---|---:|---:|
| p50 / p95 / slowest, whole trace | 3,075 / 26,893 / 29,739 ms | 259 / 2,301 / 3,095 ms |
| p95 of the requests planned at 10 to 80 s (the fine-tune runs) | 3,170 ms | 2,289 ms |
| One model step alone / beside the fine-tune | 9.2 to 9.6 / 19 to 20 ms | 8.0 / 17.8 to 18.4 ms |
| One model step from 90 s (the stop is asked) to 127 s | 47 to 48 ms | 8 ms |
| Most requests waiting at the end of a 10-second slice | 214 | 14 |
| Failed | 0 | 0 |

**No.** With the same software a box that trains and gets every request is 2 to 3 seconds late at p95, and
it is back to normal as soon as the job stops. The 45-second spell on event box A did not come back. Its
cause is still not known. The fine-tune was gone for the last 34 of those 45 seconds, and that box's LLM got
no request in those minutes (by its own log).

Files: `static-replay/` (`table.md`, `summary.json`, `summary.csv`, `run.txt`, `raw-rows.tar.gz` with every
request, and `static_timeline.py`, which prints the 10-second timeline from a case's raw rows:
`python3 static_timeline.py <folder with raw-static.jsonl and trace.jsonl> static box-a`).

## Files

- `night_proof.py`: the checks (`chat`, `load`, `followups`). `start_llm.sh`, `start_stack.sh`, `run_stack.sh`:
  how the stack was started on the one machine.
- `proof-chat-main.json`, `proof-load-main.json`, `proof-followups-main.json`: what the checks returned.
