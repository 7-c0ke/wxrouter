# First measurements on the event boxes (Sat Nov 14, 2026)

*Two NVIDIA GB10 boxes (CIQ-CHI-SPARK27 and CIQ-CHI-SPARK28), Fuzzball 4.3.1, workflow `real-stack`
(`115da53c-e361-44dd-8a3b-791441ea4ce9`, started 11:45 from `fuzzball/30-real-stack.fz`). Each box runs one
GPU service: the supervisor `serve/node_stack.py` with the real forecast server (`serve/forecast_server.py`,
AFNO, weights `65f9f3bb...`) inside it. `router.py` is in front; `ui2/` is the page. Times are Central.
Every number here is from one run. None has been repeated yet.*

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

## Not measured on the event boxes yet
- A real fine-tune beside serving (the training data is not on the volume yet).
- The LLM (its file was still downloading during these runs).
- `loadgen.py bench` with its faults.
