"""Does pinning the forecast server to the box's five fast cores change its speed? Run inside box-a.
The process id comes from the supervisor, and its command line is read before anything is done to it."""
import json
import statistics
import subprocess
import time
import urllib.request

A, B, STACK_A = "http://127.0.0.1:8001", "http://box-b:8002", "http://127.0.0.1:8109"
ASK = {"init": "latest", "lead_hours": 48, "members": 1, "point": {"lat": 41.98, "lon": -87.9}}


def forecast(url):
    req = urllib.request.Request(url + "/v1/forecast", data=json.dumps(ASK).encode(), headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000.0, float((d.get("timing_ms") or {}).get("gpu") or 0.0)


def series(name, url, n=60):
    rows = [forecast(url) for _ in range(n)]
    total, gpu = sorted(r[0] for r in rows), sorted(r[1] for r in rows)
    out = {"p50": statistics.median(total), "p95": total[int(0.95 * (n - 1))], "gpu_p50": statistics.median(gpu)}
    print("%-34s total p50 %6.1f ms  p95 %6.1f ms | GPU p50 %6.1f ms | rest (p50 - GPU p50) %5.1f ms" % (
        name, out["p50"], out["p95"], out["gpu_p50"], out["p50"] - out["gpu_p50"]), flush=True)
    return out


with urllib.request.urlopen(STACK_A + "/status", timeout=5) as r:
    pid = json.loads(r.read())["servers"]["forecast"]["pid"]
cmdline = open("/proc/%d/cmdline" % pid).read().replace("\0", " ")
print("forecast server on box-a: pid %d: %s" % (pid, cmdline[:150]))
if "forecast_server.py" not in cmdline:
    raise SystemExit("that is not the forecast server; nothing was changed")
before = subprocess.run(["taskset", "-cp", str(pid)], capture_output=True, text=True).stdout.strip()
print("its cores before: %s" % before)
cores_before = before.split(":")[-1].strip()

for url in (A, B):
    for _ in range(8):
        forecast(url)
series("box-a, not pinned", A)
series("box-b, not pinned (control)", B)
try:
    out = subprocess.run(["taskset", "-a", "-cp", "5-9", str(pid)], capture_output=True, text=True)
    print("pinned to cores 5-9: exit %s; now: %s" % (out.returncode, subprocess.run(["taskset", "-cp", str(pid)], capture_output=True, text=True).stdout.strip()))
    series("box-a, PINNED to cores 5-9", A)
    series("box-b, not pinned (control)", B)
    series("box-a, PINNED to cores 5-9 (again)", A)
finally:
    out = subprocess.run(["taskset", "-a", "-cp", cores_before, str(pid)], capture_output=True, text=True)
    print("put back to %s: exit %s; now: %s" % (cores_before, out.returncode, subprocess.run(["taskset", "-cp", str(pid)], capture_output=True, text=True).stdout.strip()))
series("box-a, not pinned (again)", A)
