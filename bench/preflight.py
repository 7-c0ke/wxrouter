"""One minute of checks before GPU time is spent on a measurement.

Run it inside box-a of the real stack (fuzzball/30-real-stack.fz), before bench/scene_finetune.py,
bench/swap_weights.py and bench/scene_split.py. It changes nothing: it only reads.

    python3 bench/preflight.py
    python3 bench/preflight.py --router http://127.0.0.1:19000 --no-files \
        --box box-a=http://127.0.0.1:19001,http://127.0.0.1:19109 --box box-b=http://127.0.0.1:19002,http://127.0.0.1:19110

Each line starts with ok, WARN or FAIL. A FAIL means that at least one of the three measurements would
fail or give a number that cannot be used; the exit code is then 1. A WARN is something to know first.

Standard library only. Written at the hackathon (Nov 14, 2026).
"""
import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
found = {"FAIL": 0, "WARN": 0}


def say(level, text):
    if level in found:
        found[level] += 1
    print("%-4s  %s" % (level, text), flush=True)


def http(url, body=None, timeout=10.0):
    """GET, or POST of a JSON body. Returns (status, parsed JSON or None, headers, seconds). Never raises."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw, status, headers = answer.read(), answer.status, answer.headers
    except urllib.error.HTTPError as e:
        raw, status, headers = e.read(), e.code, e.headers
    except Exception as e:                       # refused, reset, timed out
        return type(e).__name__, None, {}, time.monotonic() - t0
    try:
        return status, json.loads(raw), headers, time.monotonic() - t0
    except ValueError:
        return status, None, headers, time.monotonic() - t0


def age(path):
    minutes = (time.time() - os.path.getmtime(path)) / 60.0
    return "%.0f min old" % minutes if minutes < 180 else "%.1f h old" % (minutes / 60.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router", default="http://router:8000")
    p.add_argument("--box", action="append", metavar="NAME=FORECAST_URL,STACK_URL",
                   help="a box: its forecast server and its supervisor (default: box-a and box-b of the real stack)")
    p.add_argument("--train-box", default="box-a", help="the box the fine-tune will run on (default box-a)")
    p.add_argument("--link", default="/wx/bundle/weights/serving.pt")
    p.add_argument("--analyses", default="/wx/gfs-year")
    p.add_argument("--train-out", default="/wx/ft/box-a")
    p.add_argument("--no-files", action="store_true", help="skip the checks of files on the volume (for a laptop test)")
    a = p.parse_args()
    boxes = []
    for item in a.box or ["box-a=http://box-a:8001,http://box-a:8109", "box-b=http://box-b:8002,http://box-b:8110"]:
        name, urls = item.split("=", 1)
        forecast_url, stack_url = urls.split(",", 1)
        boxes.append((name, forecast_url.rstrip("/"), stack_url.rstrip("/")))

    # ---- the router
    status, d, _, _ = http(a.router.rstrip("/") + "/router/status", timeout=5.0)
    if status != 200 or not isinstance(d, dict):
        say("FAIL", "router %s does not answer /router/status (%s)" % (a.router, status))
    else:
        backends = d.get("backends") or {}
        sick = [n for n, b in backends.items() if not (isinstance(b, dict) and b.get("healthy"))]
        adaptive = d.get("adaptive") or {}
        say("FAIL" if sick or len(backends) < 2 else "ok",
            "router: rule %s, preferred %s, active %s, backends %s%s" % (
                d.get("policy"), adaptive.get("preferred"), adaptive.get("active"), ", ".join(sorted(backends)) or "none",
                (" | NOT healthy: " + ", ".join(sick)) if sick else ""))

    # ---- the boxes: health, model, starting weather
    versions, latest, mocks = {}, {}, {}
    for name, forecast_url, stack_url in boxes:
        status, d, _, _ = http(forecast_url + "/health", timeout=5.0)
        if status != 200 or not isinstance(d, dict) or d.get("status") != "ok":
            say("FAIL", "%s: forecast server not healthy at %s (%s %s)" % (name, forecast_url, status, json.dumps(d)[:120]))
            continue
        versions[name] = d.get("model_version")
        status, info, _, _ = http(forecast_url + "/v1/info", timeout=5.0)
        if status == 200 and isinstance(info, dict):
            latest[name], mocks[name] = info.get("latest"), bool(info.get("mock"))
            say("WARN" if info.get("mock") else "ok", "%s: %s, model_version %s, device %s, newest starting weather %s%s" % (
                name, info.get("model"), versions[name], info.get("device"), info.get("latest"),
                " | this is a MOCK: no real numbers" if info.get("mock") else ""))
        else:
            say("WARN", "%s: healthy, model_version %s, but no /v1/info (not our forecast server?)" % (name, versions[name]))
    if len(versions) == len(boxes) and len(set(versions.values())) > 1:
        say("WARN", "the boxes serve different models: %s" % versions)
    if len(latest) == len(boxes) and len(set(latest.values())) > 1:
        say("FAIL", "the boxes start from different analyses (%s): the split forecast will be refused. "
                    "Send the newest state to both through the second router (POST /v1/state)." % latest)

    # ---- the supervisors: servers up, no fine-tune running
    pids = {}
    for name, forecast_url, stack_url in boxes:
        status, d, _, _ = http(stack_url + "/status", timeout=5.0)
        if status != 200 or not isinstance(d, dict):
            say("FAIL", "%s: supervisor does not answer at %s (%s)" % (name, stack_url, status))
            continue
        servers, jobs = d.get("servers") or {}, d.get("jobs") or {}
        down = [n for n, s in servers.items() if not (isinstance(s, dict) and s.get("state") == "running")]
        job = jobs.get("finetune") if isinstance(jobs.get("finetune"), dict) else None
        pids[name] = (servers.get("forecast") or {}).get("pid")
        text = "%s: supervisor keeps %s; restarts %s; fine-tune job: %s" % (
            name, ", ".join(sorted(servers)) or "nothing",
            {n: s.get("restarts") for n, s in servers.items() if isinstance(s, dict)}, job.get("state") if job else "NOT DEFINED")
        if down:
            say("FAIL", text + " | not running: " + ", ".join(down))
        elif job is None and name == a.train_box:
            say("FAIL", text + " | the training scene needs a job named finetune on this box")
        elif job and job.get("state") == "running":
            say("FAIL" if name == a.train_box else "WARN", text + " | a fine-tune is running NOW; wait for it or stop it first")
        else:
            say("ok", text)

    # ---- one real request through the router
    status, d, headers, seconds = http(a.router.rstrip("/") + "/v1/forecast",
                                       {"lat": 41.98, "lon": -87.9, "hours_ahead": 48, "request_id": "preflight"}, timeout=60.0)
    if status != 200 or not isinstance(d, dict):
        say("FAIL", "a 48-hour forecast through the router failed: %s %s" % (status, json.dumps(d)[:160]))
    else:
        timing = d.get("timing") or {}
        say("ok", "a 48-hour forecast through the router: %.0f ms (GPU %s ms), served by %s" % (
            seconds * 1000.0, timing.get("compute_ms"), headers.get("x-served-by")))

    # ---- files on the volume
    if not a.no_files:
        if os.path.islink(a.link):
            say("ok", "model link: %s -> %s" % (a.link, os.readlink(a.link)))
        else:
            say("WARN", "model link %s is not there: this stack was started from an old copy of the stack file" % a.link)
        pid = pids.get(a.train_box)
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                command = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
            if os.path.basename(a.link) in command:
                say("ok", "the forecast server here loads its model through the link: the swap can work")
            else:
                say("FAIL", "the forecast server here was started WITHOUT the link (an old copy of the stack file): "
                            "the model swap cannot work in this stack; the other two measurements can")
        except (OSError, TypeError):
            say("WARN", "could not read the forecast server's command line here (run this inside %s)" % a.train_box)
        npy, index = os.path.join(a.analyses, "analyses.npy"), os.path.join(a.analyses, "analyses.json")
        if os.path.isfile(npy) and os.path.isfile(index):
            say("ok", "training data: %s, %.2f GB" % (npy, os.path.getsize(npy) / 1e9))
        else:
            say("FAIL", "training data missing in %s (analyses.npy and analyses.json): the fine-tune would stop at once" % a.analyses)
        if os.path.isdir(a.train_out):
            old = ["%s (%s)" % (n, age(os.path.join(a.train_out, n))) for n in sorted(os.listdir(a.train_out))
                   if n in ("log.jsonl", "tuned.pt", "finetune_report.json", "finetune_report.md")]
            if old:
                say("WARN", "%s holds files of an earlier run: %s. A fine-tune writes over them; a speed read from "
                            "an old log.jsonl would be wrong, so check the times in the result." % (a.train_out, ", ".join(old)))
            else:
                say("ok", "%s is empty of earlier results" % a.train_out)
        else:
            say("ok", "%s does not exist yet (the fine-tune makes it)" % a.train_out)
        for name in ("bench/scene_finetune.py", "bench/swap_weights.py", "bench/scene_split.py", "router/router.py", "weather/finetune.py"):
            if not os.path.isfile(os.path.join(ROOT, name)):
                say("FAIL", "%s is missing in %s: pull the repo on the volume first" % (name, ROOT))
        try:
            usage = os.statvfs(os.path.dirname(a.train_out.rstrip("/")) or "/")
            free_gb = usage.f_bavail * usage.f_frsize / 1e9
            say("ok" if free_gb > 5 else "WARN", "free space for results: %.0f GB" % free_gb)
        except OSError:
            pass

    # ---- the processor cores and the GPU, for the record
    if hasattr(os, "sched_getaffinity"):
        allowed, speeds = sorted(os.sched_getaffinity(0)), {}
        for core in allowed:
            try:
                with open("/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq" % core) as f:
                    speeds[core] = int(f.read().strip())
            except (OSError, ValueError):
                break
        if len(speeds) == len(allowed) and speeds:
            top = max(speeds.values())
            fast = [c for c in allowed if speeds[c] == top]
            say("ok", "cores allowed here: %s; the fastest (%.1f GHz): %s; slower: %s" % (
                allowed, top / 1e6, fast, [c for c in allowed if c not in fast] or "none"))
        else:
            say("ok", "cores allowed here: %s (their speeds cannot be read)" % allowed)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            say("ok", "GPU: %s" % out)
    except (OSError, subprocess.SubprocessError):
        pass

    print("")
    print("%d FAIL, %d WARN" % (found["FAIL"], found["WARN"]))
    raise SystemExit(1 if found["FAIL"] else 0)


if __name__ == "__main__":
    main()
