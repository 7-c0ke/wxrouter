"""Scene 2, measured: one big forecast on two boxes.

A forecast with many ensemble members is one large job. A box alone computes every member. The second
router (router/router.py, POST /v1/ensemble) gives half of the members to each box, runs both halves at
the same time and merges the answers. This script times both ways on the same request, several times,
and checks that the merged answer is the same forecast:

    one box alone   POST <box>/v1/forecast    members=20          (box A, then box B)
    split           POST <router>/v1/ensemble members=20          (10 on each box)

    python bench/scene_split.py                                   # inside the real stack: box-a, box-b
    python bench/scene_split.py --box A=http://127.0.0.1:8101 --box B=http://127.0.0.1:8102 --hours 48

If --split-router is not given, the script starts router/router.py itself on a free local port, in front
of the two boxes, and stops it at the end. Nothing else is started and nothing on the boxes is changed.
It sends no other traffic, so run it while the boxes are otherwise quiet.

Standard library only. Written at the hackathon (Nov 14, 2026).
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WX_REPO") or os.path.dirname(HERE)      # WX_REPO: when this file is run from elsewhere


def http(url, body=None, timeout=300.0):
    """GET, or POST of a JSON body. Returns (status, parsed JSON or None, seconds). Never raises."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw, status = answer.read(), answer.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except Exception as e:                       # refused, reset, timed out
        return type(e).__name__, None, time.monotonic() - t0
    seconds = time.monotonic() - t0
    try:
        return status, json.loads(raw), seconds
    except ValueError:
        return status, None, seconds


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_router(boxes, folder, busy_other_penalty=None):
    """Start router/router.py in front of the boxes. Returns (process, url)."""
    port = free_port()
    config = {"port": port, "policy": "least_loaded", "decision_log": "",
              "pools": {"forecast": {"first_byte_timeout_s": 300,
                                     "backends": [{"name": name, "url": url, "probe": "forecast"}
                                                  for name, url in boxes]}}}
    if busy_other_penalty is not None:
        # The split router gives a box that reports work from somebody else fewer members. In the real stack
        # the main router probes each box twice a second, which is such work, though tiny.
        config["score"] = {"busy_other_penalty": busy_other_penalty}
    path = os.path.join(folder, "split-router.json")
    with open(path, "w") as f:
        json.dump(config, f)
    log = open(os.path.join(folder, "split-router.log"), "w")
    process = subprocess.Popen([sys.executable, os.path.join(ROOT, "router", "router.py"), "--config", path],
                               cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    return process, "http://127.0.0.1:%d" % port


def healthy_backends(router):
    status, d, _ = http(router + "/router/status", timeout=5.0)
    if status != 200 or not isinstance(d, dict):
        return []
    rows = ((d.get("pools") or {}).get("forecast") or {}).get("backends") or []
    return [r.get("name") for r in rows if r.get("healthy")]


def median(values):
    return round(statistics.median(values), 1) if values else None


def series_gap(one, other):
    """The largest difference between two lists of numbers; None where they cannot be compared."""
    if not isinstance(one, list) or not isinstance(other, list) or len(one) != len(other) or not one:
        return None
    try:
        return max(abs(float(a) - float(b)) for a, b in zip(one, other))
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--box", action="append", metavar="NAME=FORECAST_URL",
                   help="a box's forecast server; give two (default: box-a=http://box-a:8001 box-b=http://box-b:8002)")
    p.add_argument("--split-router", help="a running router/router.py; without it the script starts its own")
    p.add_argument("--members", type=int, default=20, help="ensemble members (default 20)")
    p.add_argument("--hours", type=int, default=120, help="forecast length in hours, a multiple of 6 (default 120)")
    p.add_argument("--repeats", type=int, default=5, help="how often each way is timed (default 5)")
    p.add_argument("--settle", type=float, default=2.5,
                   help="quiet seconds before each split, so the router has a fresh reading of both boxes (default 2.5)")
    p.add_argument("--busy-other-penalty", type=float,
                   help="the split router's penalty for a box that reports other work (its default is 1.5; 1.0 "
                        "switches it off, which shows the split itself when the only other work is health probes)")
    p.add_argument("--lat", type=float, default=41.98)
    p.add_argument("--lon", type=float, default=-87.9)
    p.add_argument("--out", help="folder for the result file scene-split-<time>.json")
    a = p.parse_args()

    boxes = []
    for item in a.box or ["box-a=http://box-a:8001", "box-b=http://box-b:8002"]:
        try:
            name, url = item.split("=", 1)
        except ValueError:
            p.error("--box wants NAME=FORECAST_URL, not %r" % item)
        boxes.append((name, url.rstrip("/")))
    if len(boxes) != 2:
        p.error("give exactly two boxes")
    if a.hours % 6 or not 6 <= a.hours <= 120:
        p.error("--hours must be a multiple of 6 from 6 to 120")
    ask = {"init": "latest", "lead_hours": a.hours, "members": a.members, "point": {"lat": a.lat, "lon": a.lon}}

    versions = {}
    for name, url in boxes:
        status, d, _ = http(url + "/health", timeout=5.0)
        if status != 200 or not isinstance(d, dict) or d.get("status") != "ok":
            raise SystemExit("%s is not healthy (%s): %s" % (name, url, status))
        versions[name] = d.get("model_version")
        status, d, seconds = http(url + "/v1/forecast", dict(ask, lead_hours=6, members=1))     # warm-up
        if status != 200:
            raise SystemExit("%s refused a small forecast: HTTP %s %s" % (name, status, json.dumps(d)[:200]))
        print("%s: healthy, model_version %s, small forecast in %.0f ms" % (name, versions[name], seconds * 1000), flush=True)

    folder = tempfile.mkdtemp(prefix="scene-split-")
    process, router = None, a.split_router
    try:
        if not router:
            process, router = start_router(boxes, folder, a.busy_other_penalty)
        router = router.rstrip("/")
        deadline = time.time() + 30
        while sorted(healthy_backends(router)) != sorted(name for name, _ in boxes):
            if process is not None and process.poll() is not None:
                raise SystemExit("the split router ended while starting; see %s" % os.path.join(folder, "split-router.log"))
            if time.time() > deadline:
                raise SystemExit("the split router at %s does not see both boxes as healthy" % router)
            time.sleep(0.5)
        print("split router at %s sees both boxes" % router, flush=True)

        alone = dict((name, []) for name, _ in boxes)       # wall ms per run
        alone_gpu = dict((name, []) for name, _ in boxes)   # the box's own GPU ms
        split, split_shards, gaps, exact = [], [], [], []
        whole = None
        for run in range(1, a.repeats + 1):
            for name, url in boxes:
                status, d, seconds = http(url + "/v1/forecast", ask)
                if status != 200:
                    raise SystemExit("%s alone failed: HTTP %s %s" % (name, status, json.dumps(d)[:200]))
                alone[name].append(seconds * 1000.0)
                alone_gpu[name].append(float((d.get("timing_ms") or {}).get("gpu") or 0.0))
                whole = d
            # The split router reads each box's load once a second. Wait until it has seen both boxes idle
            # again, or it rightly gives the whole job to the box that is not busy with the request above.
            time.sleep(a.settle)
            status, d, seconds = http(router + "/v1/ensemble", ask)
            if status != 200:
                raise SystemExit("the split failed: HTTP %s %s" % (status, json.dumps(d)[:300]))
            split.append(seconds * 1000.0)
            shards = d.get("shards") or []
            split_shards.append([(s.get("backend"), s.get("members"), s.get("ms")) for s in shards])
            if d.get("init_time") == whole.get("init_time"):
                gaps.append(series_gap(d["point"].get("t2m_c_mean"), whole["point"].get("t2m_c_mean")))
                exact.append(all(d["point"].get(k) == whole["point"].get(k) for k in ("t2m_c", "t2m_c_min", "t2m_c_max")))
            print("run %d: alone %s, split %.0f ms %s" % (
                run, ", ".join("%s %.0f ms" % (name, alone[name][-1]) for name, _ in boxes), split[-1],
                " + ".join("%s %s members %.0f ms" % s for s in split_shards[-1])), flush=True)
            time.sleep(a.settle)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    one_box = [ms for name, _ in boxes for ms in alone[name]]
    known_gaps = [g for g in gaps if g is not None]
    result = {
        "what": "bench/scene_split.py", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": ask, "repeats": a.repeats, "model_versions": versions, "busy_other_penalty": a.busy_other_penalty,
        "one_box_ms": {"median": median(one_box), "min": round(min(one_box), 1), "max": round(max(one_box), 1),
                       "by_box": dict((name, median(alone[name])) for name, _ in boxes),
                       "gpu_by_box": dict((name, median(alone_gpu[name])) for name, _ in boxes)},
        "split_ms": {"median": median(split), "min": round(min(split), 1), "max": round(max(split), 1)},
        "shards_by_run": split_shards,
        "even_splits": sum(1 for shards in split_shards if len(shards) == 2 and shards[0][1] == shards[1][1]),
        "speedup": round(statistics.median(one_box) / statistics.median(split), 2),
        "same_forecast": {"compared_runs": len(known_gaps),
                          "largest_gap_in_mean_t2m_c": round(max(known_gaps), 4) if known_gaps else None,
                          "member_0_min_max_identical": all(exact) if exact else None},
    }
    print("")
    print("one box alone: median %(median)s ms (%(min)s to %(max)s)" % result["one_box_ms"])
    print("split 2 boxes: median %(median)s ms (%(min)s to %(max)s)" % result["split_ms"])
    print("speed-up: %.2f times; %d members x %d h; %d runs each way; %d of %d splits were even"
          % (result["speedup"], a.members, a.hours, a.repeats, result["even_splits"], a.repeats))
    same = result["same_forecast"]
    if same["compared_runs"]:
        print("same forecast: mean temperature differs by at most %s deg C; member 0, lowest and highest member identical: %s"
              % (same["largest_gap_in_mean_t2m_c"], same["member_0_min_max_identical"]))
    else:
        print("the answers could not be compared (the boxes started from different analyses)")
    print(json.dumps(result), flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "scene-split-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        print("written: %s" % path)


if __name__ == "__main__":
    main()
