#!/usr/bin/env python3
"""Which CPU core does the forecast server's model thread sit on, second by second?

A GB10 has ten fast cores and ten slower ones, and a box's service gets some of each (the event stack's box
service had cores 0 to 11: five fast, seven slow). A model run is hundreds of small GPU calls launched by
one CPU thread, so the core under that thread sets the pace: on one GB10 the same forecast took 7.5 ms a
model step on a fast core and 18.3 ms on a slow one (results/one-gb10-2026-11-14-night, section 4).

This script only watches. Run it INSIDE the box's own service while something else sends the traffic
(it reads /proc, so it has to be in the same container as the server):

    fuzzball workflow exec <id> box-a -- python3 /wx/repo/bench/core_watch.py --seconds 200 --out /wx/results/core-watch-box-a.txt

It takes the server's pid from the box's supervisor and checks the command line before it reads anything.
Every second it writes one line: seconds since start, the core of the busiest thread, that thread's CPU
time in that second (in ticks, 100 a second), and "fast" or "slow". At the end it prints a summary.
If a box turns slow and the thread sits on a slow core, `taskset -a -cp <fast cores> <pid>` moves it
(results/event-boxes-2026-11-14/scripts/pin_check.py did that by hand).
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def core_rates(sysfs="/sys/devices/system/cpu"):
    """Top clock rate of every core that can be read, in kHz."""
    rates = {}
    for name in os.listdir(sysfs):
        if name.startswith("cpu") and name[3:].isdigit():
            try:
                with open(os.path.join(sysfs, name, "cpufreq", "cpuinfo_max_freq")) as f:
                    rates[int(name[3:])] = int(f.read().strip())
            except (OSError, ValueError):
                pass
    return rates


def threads(pid):
    """{thread id: (core it last ran on, CPU ticks so far)} of one process."""
    out = {}
    for tid in os.listdir("/proc/%d/task" % pid):
        try:
            with open("/proc/%d/task/%s/stat" % (pid, tid)) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            out[int(tid)] = (int(fields[36]), int(fields[11]) + int(fields[12]))
        except (OSError, ValueError, IndexError):
            pass                                    # the thread ended between the listing and the read
    return out


def summary(rows, busy_ticks=20):
    """rows: (second, core, ticks, "fast"/"slow"/"?"). Counts the seconds in which the thread was busy."""
    busy = [r for r in rows if r[2] >= busy_ticks]
    longest = run = 0
    for r in rows:
        run = run + 1 if (r[2] >= busy_ticks and r[3] == "slow") else 0
        longest = max(longest, run)
    cores = {}
    for r in busy:
        cores[r[1]] = cores.get(r[1], 0) + 1
    return {"seconds": len(rows), "busy_seconds": len(busy),
            "busy_on_fast": sum(1 for r in busy if r[3] == "fast"),
            "busy_on_slow": sum(1 for r in busy if r[3] == "slow"),
            "longest_slow_stretch_s": longest, "busy_seconds_by_core": dict(sorted(cores.items()))}


def main(argv=None):
    p = argparse.ArgumentParser(description="Watch which CPU core the forecast server's busiest thread sits on.")
    p.add_argument("--supervisor", default="http://127.0.0.1:8109", help="this box's supervisor (box-b: port 8110)")
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--out", default=None, help="write the per-second lines to this file")
    args = p.parse_args(argv)
    with urllib.request.urlopen(args.supervisor + "/status", timeout=5) as r:
        pid = json.loads(r.read().decode())["servers"]["forecast"]["pid"]
    with open("/proc/%d/cmdline" % pid) as f:
        cmd = f.read().replace("\0", " ")
    if "forecast_server.py" not in cmd:
        sys.exit("pid %s is not a forecast server: %s" % (pid, cmd[:120]))
    rates = core_rates()
    top = max(rates.values()) if rates else None
    allowed = sorted(os.sched_getaffinity(pid))
    print("forecast server pid %d; it may use cores %s; fast cores among them: %s"
          % (pid, allowed, [c for c in allowed if rates.get(c) == top]), flush=True)
    out = open(args.out, "w") if args.out else None
    rows, start, before = [], time.time(), threads(pid)
    while time.time() - start < args.seconds:
        time.sleep(1.0)
        now = threads(pid)
        if not now:
            print("the server's process is gone", flush=True)
            break
        tid = max(now, key=lambda t: now[t][1] - before.get(t, (0, 0))[1])
        core, ticks = now[tid][0], now[tid][1] - before.get(tid, (0, 0))[1]
        kind = "?" if core not in rates else ("fast" if rates[core] == top else "slow")
        rows.append((round(time.time() - start, 1), core, ticks, kind))
        if out:
            out.write("%.1f %d %d %s\n" % rows[-1])
            out.flush()
        before = now
    print(json.dumps(summary(rows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
