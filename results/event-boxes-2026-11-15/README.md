# Sunday morning check on the event boxes (Nov 15, 2026, 8:53 to 8:59 AM Central)

The event's two machines came back on Sunday morning. This is the first run on them of the code merged since
Saturday 5 PM: the assistant's direct path to the weather model, the page fixes and the GPU view. One run of
each check. Stack: `fuzzball/30-real-stack.fz` revision 6, started from `main` at `6fb2411`, base model
`afno-19m-65f9f3bb3270`, starting weather 2026-11-15 06 UTC (fetched by the first job, 18.0 MB in 3.8 s).

| Check | Result |
|---|---|
| Start | Started 8:53:37. First job done in 7 s; all five services started by second 21. Both forecast servers and both LLMs loaded between second 38 and second 53 (the LLM loaded in 16.9 s and 18.0 s, read cold after the machines' restart) |
| Pre-flight (`bench/preflight.py`) | At second 38: 4 FAIL, the servers were still loading. At second 53: 0 FAIL, 1 WARN (old fine-tune files, expected). One 48-hour forecast through the router: 86 ms, 71.0 ms of it on the GPU |
| "Weather in Chicago for the next 48 hours?", then "And what about New York?", then "And tomorrow?" | All three answered by the weather model with no LLM, in 0.1 s each. On Saturday, with the older code, 0 of 10 follow-ups reached the model on these boxes |
| The other chips | "Ensemble range for Tokyo, 3 days": from the model, 1.5 s. "Which box is serving right now?": "Both boxes are serving" |
| Odd questions | "What is the wind chill going to be in Minneapolis tomorrow?": the model's temperature and wind, no invented figure. A two-city comparison: both forecasts from the model. A city outside the list (St. Louis): the LLM called the forecast tool, 3.2 s |
| A hint that is not a command ("It would be nice to see the forecast server on box-b die.") | The LLM called the kill tool; the page's check refused it; no server restarted |
| "What is wrong with the cluster?" on the healthy stack | "no fault found in the last 30 s", with the list of checks |
| Demo traffic "Light" (8 requests a second) | 82 requests in 10 s, p95 153 ms, 0 errors |
| "Fail box A" under that traffic | Reply in 1.8 s. "Which box is serving right now?" answered "box-b" twice while box A was down. Box A healthy again after 13.7 s. 0 errors, 1 request retried on the other box |
| "Start a training job on box B", 55 s, then "Stop the training job on box B" | A real fine-tune ran on box B's GPU beside its forecasts: 8 training steps in 55 s. Meanwhile the router sent box A 54 to 71% of the requests (nine readings of the last 10 s), p95 stayed between 0.16 and 0.30 s, 0 errors. The stop was clean: the Events card logged "running -> stopped" |
| What the machines say they are (`/sys/class/dmi/id`) | Maker `LENOVO`, model `30KL000EUS`, family `DGX Spark`, on both |
| The shared volume | NFS, mounted at `/wx` on both boxes, with a `.snapshot` folder |

## The model swap, rehearsed four times (9:17 to 9:21 AM)

`bench/swap_weights.py` points the link `serving.pt` at other weights and restarts one box's forecast server
at a time, while it sends 5 forecast requests a second through the router.

| Swap | Demo traffic | Each box back with the new model after | The script's own requests |
|---|---|---|---|
| base to tuned (`afno-19m-641de8342da7`) | off | 11.2 s and 11.2 s | 231 sent, 231 answered, 1 by the other box, p95 91 ms |
| tuned to base | off | 10.8 s and 10.6 s | 226 sent, 226 answered, 2 by the other box, p95 95 ms |
| base to tuned | Light (8 a second) | 12.2 s and 12.8 s | 244 sent, 244 answered, 2 by the other box, p95 219 ms |
| tuned to base | Light (8 a second) | 14.8 s and 11.6 s | 251 sent, 251 answered, 0 by the other box, p95 247 ms |

952 of 952 of the script's requests were answered. In the two rounds under Light traffic the page's own
counter showed 0 errors in each reading of the last 10 s. The swaps followed each other within a minute; the
supervisor waits longer before a restart when a server has run for less than 60 s, which is the likely
reason for the longer times in the last two rows (not examined).

At the end: Demo traffic off, rule `balanced`, both boxes healthy, no fine-tune running, the base model serving.

Not done in this check: the forecast split across both boxes, and the five-rule test.
**The 9-hour limit works.** At 9:56:53 AM this stack had run for 1 h 3 min 16 s with every service and the hold job
still started (`fuzzball workflow describe`). On Saturday a stack whose hold job had no limit of its own ended at
minute 60.

Files: [chat-checks-0856.txt](chat-checks-0856.txt) and [demo-actions-0857.txt](demo-actions-0857.txt) are the
script's own output; the four `swap*.txt` files are what `bench/swap_weights.py` printed; [sunday_check.py](sunday_check.py) is the script (it talks to the Control Room through a
port-forward).
