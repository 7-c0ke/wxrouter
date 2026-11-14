"""The fine-tune scene, measured: what a real training job on one box does to the forecasts, with and
without the router steering around it, and what the forecasts do to the training job.

It talks to the team's router (router.py), to the supervisor of the box that trains (serve/node_stack.py)
and to nothing else. Standard library only, so it runs inside any service of the stack:

    python bench/scene_finetune.py --router http://router:8000 \
        --train-box box-a --train-stack http://box-a:8109 --train-out /wx/ft/box-a --out /wx/results

What it does, in order (the times are the defaults):
  1. before        45 s of traffic with nothing training (rule: balanced)
  2. start the fine-tune on the training box through its supervisor, and wait for its first step
  3. while it trains, 45 s of the same traffic under each rule in --rules. The default list starts with
     "static", which the team's router defines as "everything goes to box-a": with --train-box box-a that is
     the no-routing case, all traffic on the box that trains.
  4. alone         45 s with no traffic at all: the fine-tune's own speed
  5. wait for the job to end (light traffic, rule balanced), then 30 s of traffic again (after)
For every phase it prints the forecasts' p50 / p95 / p99, how the two boxes shared them, the errors, and the
fine-tune's steps per second in that phase (counted from the job's own log.jsonl; when that file cannot be
read, from the step lines the supervisor keeps). The same numbers go to OUT/scene-finetune-<time>.json.

Nothing here is simulated: the job is the supervisor's configured "finetune" command. A stand-in job (as in
ui/dev_stack_supervised.py) is measured the same way, which is how this script was tried on a laptop.

Written at the hackathon (Nov 14, 2026).
"""
import argparse
import collections
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

STEP_LINE = re.compile(r"step\s+(\d+)\s*/\s*(\d+)")


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
    """Forecast requests through the router at a steady rate, from threads, with the answers kept."""

    def __init__(self, router, seed=7):
        self.router = router.rstrip("/")
        self.rng = random.Random(seed)
        self.rps = 0.0
        self.rows = []                           # (start time, ms, status, served by, failover)
        self.lock = threading.Lock()
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _body(self):
        r = self.rng.random()
        if r < 0.60:
            hours = self.rng.randint(6, 24)
        elif r < 0.92:
            hours = self.rng.randint(25, 72)
        else:
            hours = self.rng.randint(73, 120)
        return {"lat": round(self.rng.uniform(-60, 70), 2), "lon": round(self.rng.uniform(-180, 180), 2),
                "hours_ahead": hours, "step_hours": 1, "request_id": "scene"}

    def _one(self, body):
        start = time.time()
        status, _, headers = http(self.router + "/v1/forecast", body, timeout=120.0)
        row = (start, (time.time() - start) * 1000.0, status, headers.get("x-served-by"),
               headers.get("x-router-failover") == "true")
        with self.lock:
            self.rows.append(row)

    def _run(self):
        while not self.stop:
            rps = self.rps
            if rps <= 0:
                time.sleep(0.05)
                continue
            threading.Thread(target=self._one, args=(self._body(),), daemon=True).start()
            time.sleep(self.rng.expovariate(rps))

    def window(self, since, until):
        """What happened to the requests that STARTED in [since, until)."""
        with self.lock:
            rows = [r for r in self.rows if since <= r[0] < until]
        good = sorted(r[1] for r in rows if r[2] == 200)
        bad = collections.Counter(str(r[2]) for r in rows if r[2] != 200)

        def q(f):
            return round(good[min(len(good) - 1, int(f * len(good)))], 1) if good else None
        by = collections.Counter(r[3] for r in rows if r[2] == 200)
        return {"requests": len(rows), "ok": len(good), "errors": dict(bad), "failovers": sum(1 for r in rows if r[4]),
                "p50_ms": q(0.50), "p95_ms": q(0.95), "p99_ms": q(0.99), "served_by": dict(by)}


class Job:
    """The fine-tune job on the training box, through its supervisor."""

    def __init__(self, stack, out_dir):
        self.stack = stack.rstrip("/")
        self.log = os.path.join(out_dir, "log.jsonl") if out_dir else None
        self.report = os.path.join(out_dir, "finetune_report.json") if out_dir else None
        self.started_at = None

    def status(self):
        _, d, _ = http(self.stack + "/status", timeout=5.0)
        return ((d or {}).get("jobs") or {}).get("finetune") or {}

    def start(self, steps):
        body = {"args": ["--steps", str(steps)]} if steps else {}
        self.started_at = time.time()
        return http(self.stack + "/jobs/finetune/start", body, timeout=15.0)

    def steps_done(self):
        """Training steps finished so far: from the job's log.jsonl if this run wrote it, else from the lines
        the supervisor kept. None when neither says."""
        if self.log and self.started_at:
            try:
                if os.path.getmtime(self.log) >= self.started_at - 1:
                    last = 0
                    with open(self.log) as f:
                        for line in f:
                            if '"step"' in line and '"dt"' in line:
                                try:
                                    last = max(last, int(json.loads(line).get("step", 0)))
                                except ValueError:
                                    pass
                    return last
            except OSError:
                pass
        best = None
        for line in self.status().get("last_lines") or []:
            m = STEP_LINE.search(line)
            if m:
                best = int(m.group(1))
        return best


def set_rule(router, rule, preferred):
    body = {"policy": rule}
    if preferred is not None:
        body["preferred"] = preferred
    status, d, _ = http(router.rstrip("/") + "/router/config", body, timeout=5.0)
    return status == 200, d


def phase(name, traffic, job, seconds, settle, rps, note=""):
    """Run traffic at `rps` for settle + seconds; measure the last `seconds`. Returns the row."""
    traffic.rps = rps
    time.sleep(settle)
    t0, s0 = time.time(), job.steps_done()
    time.sleep(seconds)
    t1, s1 = time.time(), job.steps_done()
    time.sleep(0.2)
    # Requests that started in the window may still be on their way: give them a moment, up to 20 s.
    deadline = time.time() + 20
    while time.time() < deadline:
        with traffic.lock:
            started = sum(1 for r in traffic.rows if t0 <= r[0] < t1)
        if rps <= 0 or started >= max(1, int(0.97 * rps * seconds * 0.8)):
            break
        time.sleep(0.5)
    row = traffic.window(t0, t1)
    row.update(phase=name, note=note, seconds=round(t1 - t0, 1), rps_asked=rps,
               job_state=job.status().get("state"),
               train_steps=None if s0 is None or s1 is None else s1 - s0,
               train_steps_per_s=None if s0 is None or s1 is None or s1 < s0 else round((s1 - s0) / (t1 - t0), 3))
    print("%-34s p50 %8s  p95 %8s  p99 %8s ms | %4d requests, errors %s, failovers %d | boxes %s | fine-tune %s steps/s (%s)"
          % (name, row["p50_ms"], row["p95_ms"], row["p99_ms"], row["requests"], row["errors"] or 0, row["failovers"],
             row["served_by"], row["train_steps_per_s"], row["job_state"]), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router", default="http://router:8000")
    p.add_argument("--train-box", default="box-a", help="the box that trains (default box-a, the box 'static' sends everything to)")
    p.add_argument("--train-stack", default="http://box-a:8109", help="that box's supervisor")
    p.add_argument("--train-out", default="/wx/ft/box-a", help="the fine-tune's --out folder, for log.jsonl and the report ('' if not readable from here)")
    p.add_argument("--steps", type=int, default=300, help="training steps to ask for (0: the job's own default)")
    p.add_argument("--rps", type=float, default=8.0, help="forecast requests a second (default 8)")
    p.add_argument("--rules", default="static,least_loaded,balanced,adaptive", help="rules to try while it trains, in this order")
    p.add_argument("--phase-seconds", type=float, default=45.0)
    p.add_argument("--settle", type=float, default=6.0, help="seconds of traffic before each measured window")
    p.add_argument("--wait-end", type=float, default=1500.0, help="longest wait for the job to end, in seconds")
    p.add_argument("--out", default="/wx/results")
    a = p.parse_args()

    router = a.router.rstrip("/")
    status, st, _ = http(router + "/router/status", timeout=5.0)
    if status != 200:
        raise SystemExit("the router does not answer at %s (%s)" % (router, status))
    boxes = sorted((st or {}).get("backends", {}))
    other = [b for b in boxes if b != a.train_box]
    job = Job(a.train_stack, a.train_out)
    first = job.status()
    if not first:
        raise SystemExit("the supervisor at %s does not list a 'finetune' job" % a.train_stack)
    if first.get("state") == "running":
        raise SystemExit("a fine-tune is already running on %s; stop it first" % a.train_box)
    print("router %s, boxes %s, training box %s, %s requests/s, %s steps" % (router, boxes, a.train_box, a.rps, a.steps or "default"), flush=True)

    traffic = Traffic(router)
    rows = []
    set_rule(router, "balanced", "")
    rows.append(phase("before (balanced)", traffic, job, a.phase_seconds, a.settle, a.rps))

    status, d, _ = job.start(a.steps)
    print("start fine-tune on %s: HTTP %s %s" % (a.train_box, status, json.dumps(d)[:160]), flush=True)
    if status != 200:
        traffic.stop = True
        raise SystemExit("the fine-tune did not start")
    t_start = time.time()
    while time.time() - t_start < 300:          # the job loads the model and scores the base weights first
        s = job.steps_done()
        state = job.status().get("state")
        if state != "running":
            traffic.stop = True
            raise SystemExit("the fine-tune ended before its first step (state %s): %s" % (state, job.status().get("last_lines", [])[-3:]))
        if s:
            break
        time.sleep(2)
    print("first training step after %.0f s" % (time.time() - t_start), flush=True)

    for rule in [r for r in a.rules.split(",") if r]:
        preferred = a.train_box if rule == "adaptive" else ""
        ok, _ = set_rule(router, rule, preferred)
        if not ok:
            print("rule %s was refused by the router; skipped" % rule, flush=True)
            continue
        if job.status().get("state") != "running":
            print("the fine-tune ended before rule %s could be measured" % rule, flush=True)
            break
        note = "all traffic on the training box" if rule == "static" and a.train_box == boxes[0] else (
            "home box = the training box" if rule == "adaptive" else "")
        rows.append(phase("training, rule %s" % rule, traffic, job, a.phase_seconds, a.settle, a.rps, note))
    if job.status().get("state") == "running":
        set_rule(router, "balanced", "")
        rows.append(phase("training, no traffic", traffic, job, a.phase_seconds, 3.0, 0.0, "the fine-tune alone"))

    set_rule(router, "balanced", "")
    traffic.rps = min(a.rps, 2.0)
    while time.time() - t_start < a.wait_end and job.status().get("state") == "running":
        time.sleep(5)
    end = job.status()
    print("fine-tune ended: state %s, exit %s, %s s" % (end.get("state"), end.get("exit"), end.get("seconds")), flush=True)
    rows.append(phase("after (balanced)", traffic, job, min(30.0, a.phase_seconds), a.settle, a.rps))
    traffic.stop = True

    result = {"what": "bench/scene_finetune.py", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "router": router,
              "train_box": a.train_box, "other_boxes": other, "rps": a.rps, "steps_asked": a.steps,
              "job_end": {k: end.get(k) for k in ("state", "exit", "seconds")}, "phases": rows}
    if job.report:
        try:
            with open(job.report) as f:
                rep = json.load(f)
            if os.path.getmtime(job.report) >= t_start:
                result["finetune_report"] = {
                    "overall": rep.get("overall"), "training": rep.get("training"), "seconds": rep.get("seconds"),
                    "rows": [r for r in (rep.get("comparison") or {}).get("rows", []) if r.get("lead_h") in (24, 72)]}
                print("fine-tune report: %s" % (rep.get("overall") or {}).get("sentence"), flush=True)
                for r in result["finetune_report"]["rows"]:
                    print("  %-18s %3d h  base %.3f  tuned %.3f  change %+.1f%%  (%s)" % (
                        r.get("score"), r.get("lead_h"), r.get("base"), r.get("tuned"), r.get("change_pct"), r.get("verdict")), flush=True)
        except (OSError, ValueError) as e:
            print("no fine-tune report read (%s)" % e, flush=True)
    try:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, "scene-finetune-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        print("wrote %s" % path, flush=True)
    except OSError as e:
        print("could not write the result file (%s); the lines above are the record" % e, flush=True)


if __name__ == "__main__":
    main()
