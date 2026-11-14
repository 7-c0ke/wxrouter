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
5. **Break it.** We put a real second load on box A's GPU, then stop its server for 10 seconds,
   then start it again. Under every routing policy, 0 of 7,202 requests failed, and box A was
   serving again 3 to 5 seconds after its restart. Workflow `c5ffff9f`, SPARK27 and SPARK28.
6. **What surprised us.** (a) A Spark has fast and slow CPU cores, and a small job lands on the
   slow ones: the same GPU call took 64 ms there and 9.9 ms on a fast core. The server now pins
   itself to fast cores. (b) The adaptive rule bounced between boxes about every 10 seconds,
   because the busy box runs bigger batches and so looks slower to the probe. Round robin, least
   loaded and balanced were steadier and faster here.
7. **The real model.** The weather team's trained AFNO runs behind the same engine through one
   adapter and gives the same answers as their server (128 points, differences of 0.01). Batching
   across requests does not speed it up: that GPU is already full with one request.

## What is real and what is not

| | |
|---|---|
| One-GPU numbers | Real GB10, Fuzzball workflow `4a0aaf44` |
| Slow mode | A real load on the GPU, not a sleep |
| Stop and restart | The server process really exits and cold-starts |
| The tiny model's forecasts | Not weather. Seeded random weights |
| Laptop runs | CPU only; labelled so in the report |
| Two-GPU numbers | Real: workflow `c5ffff9f`, box-a on SPARK27, box-b on SPARK28 |
| AFNO numbers | Real: workflow `664e809a`, SPARK28 |
| Adaptive threshold sweep | Simulated: mock boxes |
