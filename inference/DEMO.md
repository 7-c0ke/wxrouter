# Demo script and talk track

## Before the demo

```bash
~/bin/fuzzball context use fb-hack && ~/bin/fuzzball context login --direct
~/bin/fuzzball node list -a | grep -E "SPARK27|SPARK28"     # both must show nvidia.com/gpu:1
```

## The run (about 12 minutes, hands off)

```bash
deploy/run_inference.sh bench    # one GPU: batching off and on
deploy/run_inference.sh demo     # two GPUs: every policy, then slow / stop / restart
open results/inference/report.html
```

The second command prints the workflow ID, the two nodes the boxes landed on, and where the JSON
was saved. The report then has the policy table, where requests went, and a timeline per policy
with the faults marked.

To watch it live instead, port-forward the router while `demo` runs:

```bash
~/bin/fuzzball workflow port-forward <workflow-id> router 8000:8000
watch -n1 'curl -s localhost:8000/router/status | python3 -m json.tool | head -40'
```

## Talk track

1. **The problem.** Forecast requests arrive at one address. Behind it are two DGX Sparks. One of
   them may be busy with a training job, or gone. Nobody should have to notice.
2. **One GPU first.** A forecast call costs about 10 ms on a GB10 whether it carries one request or
   256. Served one at a time, the box tops out near 95 requests a second. The engine runs whatever
   is already waiting as one call: 2,400 requests a second at 32 clients, p95 down from 343 ms to
   14 ms. Measured on SPARK28, workflow `4a0aaf44`.
3. **A thing we tried that lost.** Half precision was 1.9× slower here and gave wrong answers, so
   it is off. The benchmark checks that itself.
4. **Two GPUs.** Fuzzball places one box on each Spark; the run refuses to continue unless the two
   boxes report different GPU UUIDs. The router sees only what the boxes report: health, queue,
   GPU use, and the time each GPU call took.
5. **Break it.** We put a real second load on box A's GPU, then stop its server, then start it
   again. The report shows, per policy, where requests went second by second, how many failed, and
   how long box A took to serve again. *(Fill in from the report once the two-GPU run exists. Do not
   quote the laptop numbers as GPU results.)*
6. **The real model.** The weather team's trained AFNO runs behind the same engine through one
   adapter, and its answers are compared with their own server's on the same weights.
   *(State the result of `deploy/run_inference.sh afno` once it has run.)*

## What is real and what is not

| | |
|---|---|
| One-GPU numbers | Real GB10, Fuzzball workflow `4a0aaf44` |
| Slow mode | A real load on the GPU, not a sleep |
| Stop and restart | The server process really exits and cold-starts |
| The tiny model's forecasts | Not weather. Seeded random weights |
| Laptop runs | CPU only; labelled so in the report |
| Two-GPU and AFNO-on-GPU numbers | Not measured yet |
