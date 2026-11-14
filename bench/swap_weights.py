"""Swap the weather model's weights, one box at a time, while forecasts keep flowing.

The forecast servers of the real stack load their weights through one link on the shared volume
(fuzzball/30-real-stack.fz: /wx/bundle/weights/serving.pt). This script
  1. starts steady traffic through the router,
  2. points the link at the new weights (an atomic rename, so no server ever sees a half-written link),
  3. for each box in turn: asks its supervisor to kill the forecast server, and waits until the server is
     back, healthy, and reporting a new model_version (the version carries the first 12 hex digits of the
     weights' SHA-256, so a changed version is proof that the new weights are the ones in use),
  4. stops the traffic and reports: per box, old version, new version and the seconds it took; for the
     traffic, how many requests were sent, how many failed, and how many were answered by the other box.

    python bench/swap_weights.py --to /wx/bundle/weights/afno_tuned_641de834.pt      # inside the real stack
    python bench/swap_weights.py --router http://router:8000 \
        --box box-a=http://box-a:8001,http://box-a:8109 --box box-b=http://box-b:8002,http://box-b:8110 \
        --to /wx/bundle/weights/afno_tuned_641de834.pt              # the same, with every address spelled out

Going back is the same command with --to /wx/bundle/weights/afno_event.pt.
Give --to a file that stays as it is. /wx/ft/<box>/tuned.pt is rewritten by every fine-tune on that box, and a
fine-tune that is stopped (also by the end of the stack) leaves its unfinished weights there: on Nov 14 the
finished model of 2:46 PM was replaced that way by one of 61 steps. The finished one is kept as
/wx/bundle/weights/afno_tuned_641de834.pt.
If a box does not come back with the new weights, the script stops and points the link at the old weights
again. The supervisor keeps trying to start that box's server, so its next try loads the old weights; the
other boxes were not touched.
A box that serves a mock (model_version holds "mock") has no weights: there the script only waits for the
server to come back, and says so. That is how it was tried on a laptop.

Standard library only. Written at the hackathon (Nov 14, 2026).
"""
import argparse
import collections
import json
import os
import threading
import time
import urllib.error
import urllib.request


def http(url, body=None, timeout=10.0):
    """GET, or POST of a JSON body. Returns (status, parsed JSON or None, headers). Never raises."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw = answer.read()
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            return answer.status, parsed, answer.headers
    except urllib.error.HTTPError as e:
        try:
            parsed = json.loads(e.read())
        except ValueError:
            parsed = None
        return e.code, parsed, e.headers
    except Exception as e:                       # refused, reset, timed out
        return type(e).__name__, None, {}


class Traffic:
    """One small forecast request every 1/rps seconds through the router, from threads."""

    def __init__(self, router, rps):
        self.url = router.rstrip("/") + "/v1/forecast"
        self.rps = rps
        self.rows = []                           # (start, ms, status, served by, failover)
        self.lock = threading.Lock()
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _one(self, i):
        start = time.time()
        status, _, headers = http(self.url, {"lat": 41.98, "lon": -87.9, "hours_ahead": 48, "request_id": "swap-%d" % i},
                                  timeout=120.0)
        with self.lock:
            self.rows.append((start, (time.time() - start) * 1000.0, status, headers.get("x-served-by"),
                              headers.get("x-router-failover") == "true"))

    def _run(self):
        i = 0
        while not self.stop:
            threading.Thread(target=self._one, args=(i,), daemon=True).start()
            i += 1
            time.sleep(1.0 / self.rps)

    def summary(self):
        with self.lock:
            rows = list(self.rows)
        good = sorted(r[1] for r in rows if r[2] == 200)
        bad = collections.Counter(str(r[2]) for r in rows if r[2] != 200)

        def q(f):
            return round(good[min(len(good) - 1, int(f * len(good)))], 1) if good else None
        return {"requests": len(rows), "ok": len(good), "failed": sum(bad.values()), "errors": dict(bad),
                "failovers": sum(1 for r in rows if r[4]), "p50_ms": q(0.50), "p95_ms": q(0.95),
                "max_ms": round(good[-1], 1) if good else None,
                "served_by": dict(collections.Counter(r[3] for r in rows if r[2] == 200))}


def version(forecast_url):
    status, d, _ = http(forecast_url.rstrip("/") + "/health", timeout=3.0)
    if status != 200 or not isinstance(d, dict):
        return None, None
    return d.get("status"), d.get("model_version")


def point_link(link, target):
    """Make `link` a symbolic link to `target`, replacing what is there in one step."""
    folder = os.path.dirname(link)
    relative = os.path.relpath(target, folder) if os.path.dirname(target) == folder else target
    tmp = link + ".new"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(relative, tmp)
    os.replace(tmp, link)
    return relative


def swap_box(name, forecast_url, stack_url, timeout_s):
    """Kill the box's forecast server through its supervisor and wait for the new one. Returns a row."""
    _, old = version(forecast_url)
    mock = old is not None and "mock" in old.lower()
    t0 = time.time()
    status, d, _ = http(stack_url.rstrip("/") + "/servers/forecast/kill", {}, timeout=10.0)
    if status != 200:
        return {"box": name, "old": old, "new": None, "seconds": None, "ok": False,
                "note": "the supervisor refused the kill: HTTP %s %s" % (status, json.dumps(d)[:120])}
    seen_down = False
    while time.time() - t0 < timeout_s:
        state, now = version(forecast_url)
        if state != "ok":
            seen_down = True
        elif seen_down and (mock or now != old):
            return {"box": name, "old": old, "new": now, "seconds": round(time.time() - t0, 1), "ok": True,
                    "note": "a mock has no weights: only the restart was checked" if mock else ""}
        elif seen_down and now == old:
            return {"box": name, "old": old, "new": now, "seconds": round(time.time() - t0, 1), "ok": False,
                    "note": "the server came back with the SAME model_version: it did not load new weights"}
        time.sleep(0.25)
    return {"box": name, "old": old, "new": None, "seconds": None, "ok": False,
            "note": "not serving again after %d s" % timeout_s}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router", default="http://router:8000")
    p.add_argument("--box", action="append", metavar="NAME=FORECAST_URL,STACK_URL",
                   help="a box: its forecast server and its supervisor. Repeat; the boxes are swapped in this order "
                        "(default: box-a and box-b of fuzzball/30-real-stack.fz)")
    p.add_argument("--link", default="/wx/bundle/weights/serving.pt", help="the link the forecast servers load")
    p.add_argument("--to", required=True, help="the weights file the link should point at")
    p.add_argument("--rps", type=float, default=5.0, help="forecast requests a second during the swap (default 5)")
    p.add_argument("--pause", type=float, default=8.0, help="seconds of traffic before, between and after the boxes")
    p.add_argument("--timeout", type=float, default=180.0, help="longest wait for one box, in seconds")
    p.add_argument("--no-link", action="store_true", help="do not touch the link (restart the servers only)")
    a = p.parse_args()

    boxes = []
    for item in a.box or ["box-a=http://box-a:8001,http://box-a:8109", "box-b=http://box-b:8002,http://box-b:8110"]:
        try:
            name, urls = item.split("=", 1)
            forecast_url, stack_url = urls.split(",", 1)
        except ValueError:
            p.error("--box wants NAME=FORECAST_URL,STACK_URL, not %r" % item)
        boxes.append((name, forecast_url, stack_url))
    a.to = os.path.abspath(a.to)                 # the link must not depend on the folder this was started in
    if not a.no_link:
        if not os.path.isfile(a.to):
            raise SystemExit("the new weights are not there: %s" % a.to)
        if os.path.lexists(a.link) and not os.path.islink(a.link):
            raise SystemExit("%s exists and is not a link; this script will not replace a real file" % a.link)
    for name, forecast_url, _ in boxes:
        state, now = version(forecast_url)
        print("%s: %s, model_version %s" % (name, state, now), flush=True)
        if state != "ok":
            raise SystemExit("%s is not healthy; swap nothing while a box is down" % name)

    traffic = Traffic(a.router, a.rps)
    time.sleep(a.pause)
    before = None
    if not a.no_link:
        before = os.readlink(a.link) if os.path.islink(a.link) else None
        print("link %s: %s -> %s" % (a.link, before, point_link(a.link, a.to)), flush=True)
    rows = []
    for name, forecast_url, stack_url in boxes:
        row = swap_box(name, forecast_url, stack_url, a.timeout)
        rows.append(row)
        print("%s: %s -> %s in %s s%s" % (name, row["old"], row["new"], row["seconds"], (" | " + row["note"]) if row["note"] else ""), flush=True)
        if not row["ok"]:
            if before is not None:
                # The way back: the link as it was. The supervisor's next try then starts this box's server with
                # the old weights, and a box that restarts later (a crash, a kill in the demo) does not pick up
                # the file that just failed.
                back = point_link(a.link, os.path.join(os.path.dirname(a.link), before))
                print("stopping here. The link %s points at %s again; the other boxes keep their weights" % (a.link, back), flush=True)
            else:
                print("stopping here: the other boxes keep their weights", flush=True)
            break
        time.sleep(a.pause)
    traffic.stop = True
    time.sleep(1.5)
    s = traffic.summary()
    print("traffic: %(requests)d requests, %(ok)d answered, %(failed)d failed %(errors)s, %(failovers)d answered by the other box; "
          "p50 %(p50_ms)s ms, p95 %(p95_ms)s ms, max %(max_ms)s ms; served by %(served_by)s" % s, flush=True)
    print(json.dumps({"what": "bench/swap_weights.py", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "to": a.to,
                      "boxes": rows, "traffic": s}), flush=True)
    raise SystemExit(0 if rows and all(r["ok"] for r in rows) and len(rows) == len(boxes) else 1)


if __name__ == "__main__":
    main()
