"""Does the forecast server get slower when its threads sit on a slow CPU core?  Runs INSIDE the test stack's
container. Takes the server's pid from its supervisor, reads its command line, times forecasts with the
threads (a) as they are, (b) on one slow core, (c) on one fast core, (d) on all slow cores, (e) on all fast
cores, and puts the first affinity back at the end."""
import json, os, statistics as st, sys, time, urllib.request

SUP, SRV = "http://127.0.0.1:19109", "http://127.0.0.1:19001"
def get(u):
    with urllib.request.urlopen(u, timeout=10) as r: return json.loads(r.read().decode())
def forecast(hours=48):
    body = json.dumps({"lat": 41.98, "lon": -87.9, "hours_ahead": hours, "step_hours": 6, "request_id": "core-test"}).encode()
    t = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(SRV + "/v1/forecast", body, {"content-type": "application/json"}), timeout=120) as r:
        d = json.loads(r.read().decode())
    return (time.perf_counter() - t) * 1000, (d.get("timing") or {}).get("compute_ms")
pid = get(SUP + "/status")["servers"]["forecast"]["pid"]
cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
print("pid from the supervisor:", pid, "|", cmd[:150])
assert "forecast_server.py" in cmd and "--node box-a" in cmd, "not the forecast server of box-a"
freq = {}
for c in sorted(os.sched_getaffinity(0)):
    try: freq[c] = int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read())
    except Exception as e: freq[c] = None
print("max kHz per core:", freq)
top = max(v for v in freq.values() if v)
fast = sorted(c for c, v in freq.items() if v == top); slow = sorted(c for c, v in freq.items() if v and v < top)
print("fast cores:", fast, "| slow cores:", slow)
tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
before = {t: os.sched_getaffinity(t) for t in tids}
print(len(tids), "threads; affinity of the main thread before:", sorted(before[pid]))
def where():
    out = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            f = open(f"/proc/{pid}/task/{t}/stat").read().rsplit(")", 1)[1].split()
            out[int(t)] = (int(f[36]), int(f[11]) + int(f[12]))   # last core, cpu ticks
        except Exception: pass
    return out
def run(label, cores, n=15):
    if cores is not None:
        for t in os.listdir(f"/proc/{pid}/task"):
            try: os.sched_setaffinity(int(t), cores)
            except Exception as e: print("  could not set", t, e)
    w0 = where(); forecast(); forecast()
    rows = [forecast() for _ in range(n)]
    w1 = where()
    busy = sorted(((w1[t][1] - w0.get(t, (0, 0))[1], t, w1[t][0]) for t in w1), reverse=True)[:3]
    tot = st.median(r[0] for r in rows); gpu = st.median(r[1] for r in rows if r[1] is not None)
    print(f"{label:34s} total {tot:7.1f} ms | model run {gpu:7.1f} ms | {gpu/8:5.1f} ms a step | busiest threads (ticks, tid, last core): {busy}")
    return tot, gpu
try:
    run("as started (all cores)", None)
    if slow:
        run(f"one slow core ({slow[0]})", {slow[0]})
    run(f"one fast core ({fast[0]})", {fast[0]})
    if slow:
        run(f"all slow cores {slow[0]}-{slow[-1]}", set(slow))
    run(f"all fast cores", set(fast))
finally:
    for t, m in before.items():
        try: os.sched_setaffinity(t, m)
        except Exception: pass
    for t in os.listdir(f"/proc/{pid}/task"):
        try: os.sched_setaffinity(int(t), before[pid])
        except Exception: pass
    print("affinity put back:", sorted(os.sched_getaffinity(pid)))
    run("after putting it back", None, n=8)
