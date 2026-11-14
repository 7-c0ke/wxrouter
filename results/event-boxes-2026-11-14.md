# Measurements on the event boxes (Sat Nov 14, 2026)

*Two NVIDIA GB10 boxes (CIQ-CHI-SPARK27 and CIQ-CHI-SPARK28), Fuzzball 4.3.1, workflow `real-stack`
(`115da53c-e361-44dd-8a3b-791441ea4ce9`, started 11:45 from `fuzzball/30-real-stack.fz`). Each box runs one
GPU service: the supervisor `serve/node_stack.py` with the real forecast server (`serve/forecast_server.py`,
AFNO, weights `65f9f3bb...`) inside it. `router.py` is in front; `ui2/` is the page (since merged into `ui/`).
Times are Central. Every number here is from one run. None has been repeated yet.*

*Sections 1 to 5 are from the morning. Sections 6 to 9 are from a second run of the same stack in the
afternoon (workflow `ac7c14cc-b7ed-4dff-bd90-b3aa391fedab`, 2:39 to 3:05 PM, code at commit `3043c4f`). Their
result files are in `results/event-boxes-2026-11-14/`.*

## 1. The real model answers through the router (11:46)
One request for Chicago O'Hare, 48 hours, hourly, sent to the router from inside the workflow: HTTP 200,
served by box-b, `model_version: afno-19m-65f9f3bb3270`, starting state NOAA GFS analysis of 2026-11-14 12 UTC,
48 points. Both boxes report 19,128,832 parameters on `cuda`.

## 2. Speed of a forecast (11:49-11:50)
Requests sent one at a time from inside box-a's service (`urllib`, same point, 6-hour steps).

| where | 1 member, 48 h: wall / GPU | 20 members, 120 h: wall / GPU |
|---|---|---|
| box-a, direct | 78 ms / 66 ms | 4.21 s / 3.83 s |
| box-b, direct | 85 ms / 70 ms | 4.31 s / 3.89 s |
| through the router | 84 ms / 66 ms | 4.18 s / 3.80 s |

Medians of 8 and of 3 requests. "GPU" is the server's own `timing.compute_ms`. The servers report 9.4 and
9.7 ms per member-step. On a team member's own GB10 the same server measured 62 ms, 3.9 s and 9.6 ms on
Nov 14 in the morning: the event boxes match it.

## 3. Killing a forecast server under traffic (11:52-11:53)
Five requests a second through the router (1 member, 48 h, hourly) for 60 s. At 10 s the forecast server on
box-a was killed through its supervisor (`POST /servers/forecast/kill`); at 34 s the one on box-b.

| | |
|---|---|
| requests sent | 298 |
| answered with HTTP 200 | 298 |
| failed | **0** |
| answered by the other box after a failed first try | 2 |
| box-a serving again after | 12.2 s |
| box-b serving again after | 13.7 s |
| latency of the answers | median 83 ms, p95 96 ms, max 1,164 ms |

The supervisor restarted each server by itself; nobody touched Fuzzball. (The same two kills also loaded a
code fix: the restarts doubled as a one-box-at-a-time update.)

## 4. Four routing rules under the same steady traffic (11:56-11:57)
The page's own load generator at 15 requests a second (random places, 6 to 120 hours ahead, hourly points),
both boxes healthy, no faults. Each rule ran for about 25 s; the figures are the router's last 10 s.

| rule | p50 | p95 | p99 | errors |
|---|---|---|---|---|
| adaptive, home box box-b | 182-208 ms | **13-15 s** | 13-16 s | 0 |
| balanced | 75 ms | 197 ms | 772 ms | 0 |
| least loaded | 52-56 ms | 187-200 ms | 278 ms | 0 |
| round robin | 66-68 ms | 179-193 ms | 437 ms | 0 |

What it shows: one GB10 cannot carry 15 of these requests a second alone, so a rule that keeps one active
box (adaptive) lets that box's queue grow, while every rule that uses both boxes stays under 0.2 s at p95.
Caveats: short windows, one run, and each rule started with whatever queue the rule before it left (the
first balanced window still showed the backlog from adaptive and is left out). This is not the fault-injection
benchmark of `loadgen.py bench`; that one has not run on the real boxes yet.

## 5. The LLM on the boxes (12:07-12:09)
NVIDIA Nemotron 3.5 Lightning 30B-A3B, 4-bit GGUF (18.9 GB), on llama.cpp's own release build b12359 for
CUDA on arm64, started by the same supervisor inside each box's GPU service (ports 8090 and 8091).

| | |
|---|---|
| The supervisor started it by itself | on its 45th and 46th try, as soon as the downloaded file was in place |
| Load from the shared volume | 13.9 s (box-a) |
| GPU memory | 18.7 GB for the LLM, 0.4 GB for the forecast server, on the same GB10 |
| A tool call (`get_forecast` with the right arguments), each box | 2.2 to 2.3 s, 55 to 56 tokens a second |

End to end through the Control Room (`ui2`, LLM mode): "What is the weather in Chicago for the next 48 hours?"
was answered in 9.4 s (the LLM called `geocode`, then `get_forecast`; the router sent the forecast to box-b;
the reply gave 2.4 to 8.4 C and said that the model has no rain output). A cluster-status question took 2.8 s;
a 3-day, 10-member forecast for Tokyo 17.3 s. The model's thinking was switched on for all of these.

## 6. One forecast split across both boxes (2:40-2:44 PM)
`bench/scene_split.py`, inside box-a. The request: 20 ensemble members, 120 hours, one place. One box alone
computes all 20 members. The second router (`router/router.py`, `POST /v1/ensemble`) gives a share of the
members to each box, runs the shares at the same time and merges the answers.

| | time |
|---|---|
| One box alone, 22 runs (11 on each box) | median 4.18 s (4.16 to 4.39 s); 3.82 s of it on the GPU |
| Split 10 + 10, 4 runs | 2.03 to 2.05 s |
| Split 9 + 11 or 13 + 7, 3 runs | 2.28 to 2.68 s |
| Split 15 + 5, 4 runs | 3.08 to 3.15 s |

- An even split is 2.05 times as fast as one box. Over all 11 split runs the median is 2.29 s (1.83 times).
- The merged answer is the same forecast: the ensemble mean differs by at most 0.008 C, and member 0, the
  lowest and the highest member are identical.
- Why 7 of 11 splits were uneven: the second router shares the members by each box's load at that moment, and
  it counts small other work. The main router probes each box twice a second, and a teammate's page sent some
  test forecasts to the same stack in these minutes. The second run (6 of the 11) switched off the penalty
  for "busy with other work" and was still uneven in 4 of 6. This is a weakness of the planner, not of the boxes.

## 7. The assistant with the LLM's thinking switched off (2:44 PM)
Four questions to the page's chat (`POST /api/chat`, LLM mode, `--llm-thinking off`), sent from inside the stack.

| question | answer after | with thinking on (section 5) |
|---|---|---|
| Weather in Chicago for the next 48 hours? | 3.9 s, and 2.9 s when asked again | 9.4 s |
| What is the status of the cluster? | 2.2 s | 2.8 s |
| A 3-day forecast for Tokyo with 10 ensemble members | 4.7 s | 17.3 s |

In two of four forecast answers (three above and one more to see the tool result) the LLM wrote a humidity figure ("humidity 71 %", "humidity about 70 %").
The model has no humidity output and the tool result says `humidity_avg_pct: null`: the figure was made up.
Pull request 17 (merged later that afternoon) changes the instructions and cuts such a figure from the reply.

## 8. A real fine-tune on box A while forecasts flow (2:45-3:00 PM)
`bench/scene_finetune.py`, inside box-a. It sends 8 forecast requests a second through the router (lengths
of 6 to 120 hours, mixed), starts the supervisor's real fine-tune job on box-a
(`weather/finetune.py`, 300 steps, batch 16, on one year of NOAA GFS analyses), and measures 45 seconds under
each routing rule while the job trains. The LLM (18.7 GB) stayed loaded on both GPUs.

| phase | p50 ms | p95 ms | p99 ms | requests | failed | served by box-a / box-b |
|---|---:|---:|---:|---:|---:|---|
| before, nothing trains (balanced) | 56 | 167 | 278 | 371 | 0 | 187 / 184 |
| training, static: all traffic stays on box-a | 889 | 1,907 | 2,355 | 371 | 0 | 371 / 0 |
| training, least loaded | 69 | 251 | 363 | 368 | 0 | 143 / 225 |
| training, balanced | 94 | 302 | 442 | 347 | 0 | 129 / 218 |
| training, adaptive with box-a as its home box | 87 | 457 | 1,021 | 362 | 0 | 26 / 336 |

- Without routing, a real training job on the box makes the forecasts on it about 11 times slower at p95
  (0.17 s to 1.9 s). With the router using both boxes, p95 stays at 0.25 to 0.30 s. Adaptive moved 93% of
  the requests to the other box; its window includes the switch itself.
- No request failed in any phase.
- The fine-tune: 300 steps in 690 s of training, 738 s in all (12 min 18 s), with exit code 0. On one GB10
  before the event, alone, the same job took 530 s of training and 564 s in all. So beside serving, and
  with the LLM on the same GPU, it took about 30% longer.
- The script also counts training steps per phase (0.24 steps a second under static, 0.38 to 0.51 under the
  other rules, 0.44 with no traffic). These are rough: a 45-second window holds about 20 steps, and the
  first passes read the training data cold from the shared volume. The job was already on a fast core;
  pinning it to the five fast cores of the service changed nothing.

What the fine-tune gained, from its own report (`finetune_report.md`): 83 forecasts started between
2026-10-01 and 2026-11-11, which the training never saw, scored against the GFS analyses.

| score | 24 h, base to tuned | 72 h, base to tuned |
|---|---|---|
| Z500 RMSE (m2/s2) | 125.08 to 111.25 (-11.1%) | 316.74 to 323.02 (+2.0%, worse) |
| T850 RMSE (K) | 1.194 to 0.974 (-18.5%) | 1.827 to 1.781 (-2.5%) |
| T2M RMSE (K) | 1.350 to 1.121 (-16.9%) | 1.754 to 1.671 (-4.7%) |
| O'Hare 2-m temperature error (C) | 1.13 to 0.86 (-23.4%) | 1.63 to 1.58 (-3.1%, no clear change) |

These gains are the same as in the run on one GB10 before the event (-11.1%, -18.4%, -16.9% at 24 h).

Caveats for this section:
- The row "after the fine-tune" is left out. During the last wait the routing rule was changed to "static"
  from a page connected to the same stack, not by the script, so that phase did not run under the rule its
  label says. The five rows above show the shares that their rules should give.
- A teammate's page was reading the stack during the run (its screenshot at 2:48 PM shows the static phase).
- The script stops waiting for a window's answers after 20 s, so under overload the slowest requests of a
  window can be missing. All 371 requests of the static phase had been answered.

## 9. The tuned model goes live, one box at a time (3:02-3:05 PM)
`bench/swap_weights.py`. Both forecast servers load their weights through the link
`/wx/bundle/weights/serving.pt`. The script sends 5 requests a second through the router (balanced rule),
points the link at the new file, then asks each box's supervisor in turn to kill the forecast server, and waits
until the server is back with a new `model_version` (the first 12 hex digits of the weights' SHA-256).

| | box-a | box-b | requests | failed | answered by the other box | p50 | p95 |
|---|---|---|---:|---:|---:|---:|---:|
| base to tuned (`65f9f3bb3270` to `641de8342da7`) | 11.3 s | 10.9 s | 232 | 0 | 2 | 82 ms | 96 ms |
| tuned back to base | 11.3 s | 11.3 s | 233 | 0 | 1 | 85 ms | 217 ms |

The stack was left on the base model. The tuned file is on the volume at `/wx/ft/box-a/tuned.pt`
(SHA-256 `641de8342da732064455e8e1a1cba855283f8724c9d77a376db962de4009fd8b`).

## Not measured on the event boxes yet
- `loadgen.py bench` with its faults, with the weather model. As written it cannot run on the real boxes
  (the real server refuses a simulated slowdown). The five-rule fault table in `README.md` is from mock servers.
- LLM traffic through a router (the page calls box-a's LLM directly).
- The assistant after pull request 17, against the real LLM.
- A repeat of any of the runs above. Every number here is from one run.
