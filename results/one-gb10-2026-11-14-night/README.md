# Two questions from the team, checked on one GB10 (Nov 14, 2026, 9 PM)

The event boxes were out of reach that evening. These checks ran on one NVIDIA GB10 (a separate
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

**One exception, found the same day and fixed since (pull request 23, in `main` from Nov 15; see
`results/chat-follow-ups-2026-11-14/`):** a follow-up question in the same chat ("And what about New
York?"). With the code that `main` had on Nov 14 the LLM answered it without asking the model: 0 of 9
follow-ups had a forecast, and the numbers were its own (`proof-followups-main.json`; on the event boxes at
5:45 PM: 0 of 10). The page shows no weather card for such an answer. With the fix: 9 of 9 in twelve of
thirteen runs on one GB10, and 8 of 9 in one. The fix has not run on the event boxes yet.

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
- Since pull request 23 the generator sends nothing more while 100 of its own requests are waiting, and the
  page says so beside the slider. The "Heavy" row above was measured on `main`, without that limit. With it
  the wait on one GPU stops near 5 s, and the requests that are held back are never sent.

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

## 4. Fast and slow CPU cores: a candidate for that spell, tested (11 PM)

A GB10 has ten fast cores (3.9 GHz) and ten slower ones (2.8 GHz). A teammate had measured on an event box
that the same GPU call of his server took 9.9 ms from a fast core and 56 to 67 ms from a slow one
(`inference/DEMO.md`, 6a). The event stack's box service has cores 0 to 11: five fast (5 to 9) and seven
slow. If the system moved the forecast server's model thread to a slow core, the box would turn slow
without any change on the GPU. Three tests:

**What a slow core costs our server** (`cores/core_test.py`: 15 forecasts of 48 hours each way, the server's
threads moved by hand, then put back):

| the server's threads on | one forecast | model run | a model step |
|---|---:|---:|---:|
| all 20 cores, as started (the busy thread sat on a fast one) | 64.7 ms | 60.4 ms | 7.5 ms |
| one slow core | 157.6 ms | 146.6 ms | 18.3 ms |
| one fast core | 63.7 ms | 60.9 ms | 7.6 ms |
| the ten slow cores | 157.8 ms | 146.1 ms | 18.3 ms |
| the ten fast cores | 64.4 ms | 60.1 ms | 7.5 ms |

So a slow core makes a forecast 2.4 times slower with nothing else on the GPU: the model run is bound by
one CPU thread. Beside a fine-tune (18 to 20 ms a step on a fast core) that would be about 45 ms, which is
what event box A showed from 82 s.

**The replay again, with the event box's cores** (the whole stack and the load generator on cores 0 to 11,
`cores/cores_experiment.sh`), while a small script wrote down each second which core the server's busiest
thread sat on:

| | p50 | p95 | p95 at 10 to 80 s | busy seconds on a fast / slow core |
|---|---:|---:|---:|---|
| all 20 cores (section 3) | 259 ms | 2,301 ms | 2,289 ms | not watched |
| cores 0 to 11, server as it is | 247 ms | 2,319 ms | 2,287 ms | 148 / 0 |
| cores 0 to 11, server pinned to its fast cores 5 to 9 | 278 ms | 2,481 ms | 2,522 ms | 149 / 0 |

- **The spell did not come back, and the thread never left the fast cores.** So this test neither shows nor
  excludes that a slow core caused the 45 seconds on event box A. After the fine-tune had ended a slow core
  alone would give about 20 ms a step, not the 47 to 48 that were seen, so it would not be the whole story.
- **Pinning the server to its fast cores is not adopted.** It worked (the server reported its five cores and
  served as before), but in this test it was no gain: the p95 was about 7% higher, most likely because the
  fine-tune's threads want the same five cores. The change is kept as `cores/forecast-server-pin.patch`.
- **To find out on the event boxes:** `bench/core_watch.py` does the watching (read-only). Run it inside
  box-a while the five-rule test runs. If the model thread sits on a slow core when the box turns slow, that
  is the cause, and `taskset -a -cp 5-9 <pid>` moves it back.

Files: `cores/` (`core_test.py` and its output, the two replays with their per-second core files and raw
rows, the scripts that started them, the patch).

## Files

- `night_proof.py`: the checks (`chat`, `load`, `followups`). `start_llm.sh`, `start_stack.sh`, `run_stack.sh`:
  how the stack was started on the one machine.
- `proof-chat-main.json`, `proof-load-main.json`, `proof-followups-main.json`: what the checks returned.
