"""The five-rule fault table on the REAL stack: the same requests and the same two faults under every routing
rule, with nothing simulated.

loadgen.py's `bench` makes this table on a laptop, with mock boxes that it starts itself. This script makes
the same table against the deployed stack (fuzzball/30-real-stack.fz): the team's router (router.py) in
front of the two forecast servers with the real model. The faults are real as well. Both go through the
supervisors (serve/node_stack.py):

  slow     a real fine-tune starts on box-a (the supervisor's "finetune" job) and trains on the same GPU
  normal   that job is stopped
  kill     the forecast server on box-b is killed, as a crash would; its supervisor starts it again

For each rule in turn: set the rule on the running router, send the trace (each request leaves at its
planned time, whether or not earlier ones were answered), fire the events at their times, wait for the
answers. The next rule gets the same trace and the same events. At the end the router gets back the
settings it had before, and a fine-tune that this script started is stopped.

    python bench/fault_table.py --router http://router:8000 \
        --stack box-a=http://box-a:8109 --stack box-b=http://box-b:8110 --out /wx/results

Standard library only, so it runs inside any service of the stack. The trace is loadgen.py's (same seed,
same mix; tests/test_fault_table.py checks that the two stay equal) and the table has loadgen.py's columns,
so the real table can stand next to the laptop one.

What to know before a run:
  * It takes both boxes for about 3 minutes per rule (15 minutes for the five). Nobody else should change the
    routing rule or start a job in that time. The script watches for it and marks a case as "disturbed".
  * The fine-tune gets its own --out folder (--job-args). Do not point it at /wx/ft/box-a: a stopped
    fine-tune saves the weights it has so far as tuned.pt and would overwrite a finished run there.
  * A real fine-tune loads its data and scores the base model before its first training step (34 s on the
    event boxes). The table's "training" window starts at the first step that the job reports.
  * Each event's answer is checked. A case whose event was refused is reported as such, not as a result.

Written at the hackathon (Nov 14, 2026).
"""
import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

RULES = ["static", "round_robin", "least_loaded", "adaptive", "balanced"]
STEP_LINE = re.compile(r"step\s+(\d+)\s*/\s*(\d+)")
TRAINING_LINE = re.compile(r"^training: \d+ steps")     # weather/finetune.py prints it right before its first step
COLUMNS = [("policy", "policy"), ("requests", "reqs"), ("errors", "errors"), ("error_pct", "err %"),
           ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"), ("p99_ms", "p99 ms"), ("max_ms", "max ms"),
           ("mean_ms", "mean ms"), ("share_a_pct", "box A %"), ("share_b_pct", "box B %"),
           ("failovers", "failovers"), ("switches", "switches")]          # loadgen.py's columns, in its order
WINDOWS = ["before", "job starting", "training", "between", "box down", "after"]


def http(url, body=None, timeout=10.0, post=False):
    """GET, or POST of a JSON body (post=True posts without one). Returns (status, parsed JSON or None,
    headers). Never raises: when there is no HTTP answer, the status is the name of the error."""
    data = None if body is None else json.dumps(body).encode()
    if data is None and post:
        data = b""
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


# ---------------------------------------------------------------- the trace (loadgen.py's, line for line)
def make_trace(seed, duration, rps):
    """Random arrivals with loadgen.py's mix of sizes. The same seed gives the same trace, always, and the
    same one loadgen.py makes: the calls to the random generator are in the same order."""
    rng = random.Random(seed)
    out, t, i = [], 0.0, 0
    while True:
        t += rng.expovariate(rps)
        if t >= duration:
            return out
        r = rng.random()
        if r < 0.60:
            hours, members, kind = rng.randint(6, 24), 1, "short"
        elif r < 0.90:
            hours, members, kind = rng.randint(25, 72), 1, "medium"
        elif r < 0.98:
            hours, members, kind = rng.randint(73, 120), 1, "long"
        else:
            hours, members, kind = rng.randint(24, 72), rng.randint(4, 10), "ensemble"
        out.append({
            "t": round(t, 4), "kind": kind,
            "body": {"lat": round(rng.uniform(-60, 70), 3), "lon": round(rng.uniform(-180, 180), 3),
                     "hours_ahead": hours, "step_hours": 1, "ensemble_members": members,
                     "request_id": "req-%d-%06d" % (seed, i)},
        })
        i += 1


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- the events
Event = collections.namedtuple("Event", "t action box")


def parse_events(text):
    """"10:slow:box-a,90:normal:box-a,105:kill:box-b" -> the events in time order."""
    events = []
    for part in filter(None, (text or "").split(",")):
        f = part.strip().split(":")
        if len(f) != 3 or f[1] not in ("slow", "normal", "kill"):
            raise SystemExit("bad event %r; write T:ACTION:BOX with ACTION slow, normal or kill" % part)
        try:
            events.append(Event(float(f[0]), f[1], f[2]))
        except ValueError:
            raise SystemExit("bad event %r; the time must be a number of seconds" % part)
    return sorted(events)


def default_events(duration, slow_box, kill_box):
    """A fine-tune on one box for about half of the case, then the other box's forecast server is killed.
    Both boxes are hit, so no rule wins just by never using the bad one (loadgen.py's reason, and its order)."""
    d = duration
    return "%g:slow:%s,%g:normal:%s,%g:kill:%s" % (round(d / 15.0, 1), slow_box, round(0.60 * d, 1), slow_box,
                                                    round(0.70 * d, 1), kill_box)


def fire(event, stacks, job_args):
    """Do one event through the box's supervisor. Returns (worked, what the supervisor said, in short)."""
    stack = stacks[event.box]
    if event.action == "slow":
        status, d, _ = http(stack + "/jobs/finetune/start", {"args": job_args} if job_args else {}, timeout=15.0)
    elif event.action == "normal":
        status, d, _ = http(stack + "/jobs/finetune/stop", post=True, timeout=45.0)
    else:
        status, d, _ = http(stack + "/servers/forecast/kill", post=True, timeout=15.0)
    said = json.dumps(d)[:200] if d is not None else ""
    return status == 200, "HTTP %s %s" % (status, said)


# ---------------------------------------------------------------- one case: one rule, the trace, the events
class Watch:
    """Reads the router and the supervisors twice a second while a case runs, and keeps what it saw with the
    time since the case began."""

    def __init__(self, router, stacks, slow_box, kill_box, forecast_url, clock):
        self.router, self.stacks, self.slow_box, self.kill_box = router, stacks, slow_box, kill_box
        self.forecast_url, self.clock = forecast_url, clock
        self.samples = []
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            t = self.clock()
            sample = {"t": round(t, 2)}
            status, d, _ = http(self.router + "/router/status", timeout=2.0)
            if status == 200 and d:
                sample["policy"] = d.get("policy")
                sample["active"] = (d.get("adaptive") or {}).get("active")
                sample["healthy"] = dict((n, b.get("healthy")) for n, b in (d.get("backends") or {}).items())
            if self.slow_box in self.stacks:
                _, d, _ = http(self.stacks[self.slow_box] + "/status", timeout=2.0)
                job = ((d or {}).get("jobs") or {}).get("finetune") or {}
                sample["job"] = job.get("state")
                step, training = None, False
                for line in job.get("last_lines") or []:
                    found = STEP_LINE.search(line)
                    if found:
                        step = int(found.group(1))
                    training = training or bool(found) or bool(TRAINING_LINE.search(line))
                sample["step"], sample["training"] = step, training
            if self.forecast_url:
                status, _, _ = http(self.forecast_url + "/health", timeout=1.5)
                sample["kill_box_answers"] = status == 200
                _, d, _ = http(self.stacks[self.kill_box] + "/status", timeout=2.0)
                sample["pid"] = (((d or {}).get("servers") or {}).get("forecast") or {}).get("pid")
            self.samples.append(sample)
            time.sleep(max(0.05, 0.5 - (self.clock() - t)))

    def end(self):
        self.stop = True
        self.thread.join(timeout=5)
        return self.samples


def marks_of(samples, events, rule):
    """The moments of a case that the table needs, in seconds since the case began (None: not seen)."""
    at = dict((e.action, e.t) for e in events)
    slow, normal, kill = at.get("slow"), at.get("normal"), at.get("kill")
    kill_box = next((e.box for e in events if e.action == "kill"), None)
    first_step = job_seen = job_gone = down = back = router_back = pid_before = None
    for s in samples:
        t = s["t"]
        if slow is not None and t >= slow and job_gone is None:
            if s.get("job") == "running":
                job_seen = t if job_seen is None else job_seen
                if first_step is None and s.get("training"):
                    first_step = t
            elif job_seen is not None:
                job_gone = t
        if kill is not None:
            if t < kill:
                pid_before = s.get("pid") or pid_before
                continue
            new_process = bool(s.get("pid")) and pid_before is not None and s.get("pid") != pid_before
            if down is None and (s.get("kill_box_answers") is False or new_process):
                down = t
            if down is not None and back is None and s.get("kill_box_answers") and (new_process or pid_before is None):
                back = t
            if back is not None and router_back is None and (s.get("healthy") or {}).get(kill_box):
                router_back = t
    other_rules = sorted(set(s["policy"] for s in samples if s.get("policy") and s["policy"] != rule))
    return {"slow_asked_s": slow, "first_training_step_s": first_step, "stop_asked_s": normal,
            "job_stopped_s": job_gone, "kill_asked_s": kill, "kill_box_down_s": down,
            "kill_box_answers_again_s": back, "router_uses_it_again_s": router_back,
            "other_rules_seen": other_rules}


def window_of(t, m, duration):
    """The window of a case that a request planned for second t falls in."""
    slow, kill = m["slow_asked_s"], m["kill_asked_s"]
    if slow is None or t < slow:
        if kill is None or t < kill:
            return "before"
    if slow is not None and t >= slow:
        first, gone = m["first_training_step_s"], m["job_stopped_s"]
        end = gone if gone is not None else (kill if kill is not None else duration)
        if t < end:
            return "training" if first is not None and t >= first else "job starting"
        if kill is None or t < kill:
            return "between"
    back = m["kill_box_answers_again_s"]
    return "box down" if back is None or t < back else "after"


def pct(sorted_vals, q):
    """loadgen.py's percentile: linear interpolation between the two nearest values."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def numbers(rows, boxes):
    """Count, errors and percentiles of some rows, with loadgen.py's meaning of each number."""
    ok = [r for r in rows if r["status"] == 200]
    lat = sorted(r["latency_ms"] for r in ok)
    n = len(rows)
    share = collections.Counter(r["served_by"] for r in ok)
    return {
        "requests": n,
        "errors": n - len(ok),
        "error_pct": round(100.0 * (n - len(ok)) / n, 2) if n else 0.0,
        "mean_ms": round(statistics.fmean(lat), 1) if lat else float("nan"),
        "p50_ms": round(pct(lat, 0.50), 1),
        "p95_ms": round(pct(lat, 0.95), 1),
        "p99_ms": round(pct(lat, 0.99), 1),
        "max_ms": round(lat[-1], 1) if lat else float("nan"),
        "share_a_pct": round(100.0 * share.get(boxes[0], 0) / max(1, len(ok)), 1),
        "share_b_pct": round(100.0 * share.get(boxes[1], 0) / max(1, len(ok)), 1) if len(boxes) > 1 else 0.0,
        "failovers": sum(1 for r in ok if r.get("attempts", 1) > 1),
    }


def run_case(rule, trace, events, a, stacks, boxes, forecast_url, job_args):
    """One rule: send the trace, fire the events, return (rows, what happened)."""
    start = time.perf_counter()

    def clock():
        return time.perf_counter() - start

    rows, lock, threads, fired = [], threading.Lock(), [], []

    def one(item):
        sent = clock()
        status, _, headers = http(a.router + "/v1/forecast", item["body"], timeout=a.client_timeout)
        row = {"id": item["body"]["request_id"], "t": item["t"], "sent_s": round(sent, 4), "kind": item["kind"],
               "status": status if isinstance(status, int) else 0,
               "served_by": headers.get("x-served-by") or "none",
               "attempts": int(headers.get("x-router-attempts") or 0) if isinstance(status, int) else 0,
               "latency_ms": round((clock() - sent) * 1000.0, 2)}
        if not isinstance(status, int):
            row["error"] = status
        with lock:
            rows.append(row)

    def fire_one(event):
        t = clock()
        worked, said = fire(event, stacks, job_args)
        with lock:
            fired.append({"t": round(t, 2), "took_s": round(clock() - t, 2), "action": event.action,
                          "box": event.box, "worked": worked, "said": said})
        print("    event at %5.1f s: %s %s -> %s" % (t, event.action, event.box, said[:120]), flush=True)

    def fire_events():
        # Each event in its own thread: a job stop answers only when the job is gone, and the next event
        # must not wait for that.
        for event in events:
            time.sleep(max(0.0, event.t - clock()))
            thread = threading.Thread(target=fire_one, args=(event,), daemon=True)
            thread.start()
            event_threads.append(thread)

    event_threads = []
    slow_box = next((e.box for e in events if e.action == "slow"), None)
    kill_box = next((e.box for e in events if e.action == "kill"), None)
    watch = Watch(a.router, stacks, slow_box, kill_box, forecast_url, clock)
    events_thread = threading.Thread(target=fire_events, daemon=True)
    events_thread.start()
    for item in trace:
        wait = item["t"] - clock()
        if wait > 0:
            time.sleep(wait)
        thread = threading.Thread(target=one, args=(item,), daemon=True)
        thread.start()
        threads.append(thread)
    events_thread.join(timeout=max(1.0, (events[-1].t if events else 0) - clock() + 5))
    for thread in event_threads:
        thread.join(timeout=60)
    deadline = time.time() + a.client_timeout + 5
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.time()))
    # After a kill, stay until the box answers again (or 60 s), so the next rule starts with two boxes.
    if any(e.action == "kill" for e in events):
        until = time.time() + 60
        while time.time() < until and not (watch.samples and watch.samples[-1].get("kill_box_answers")):
            time.sleep(0.5)
    samples = watch.end()
    with lock:
        rows = sorted(rows, key=lambda r: r["t"])
    unanswered = len(trace) - len(rows)
    marks = marks_of(samples, events, rule)
    late = max((r["sent_s"] - r["t"] for r in rows), default=0.0)
    fired.sort(key=lambda f: f["t"])
    happened = {"events": fired, "marks": marks, "unanswered": unanswered,
                "sender_latest_ms": round(late * 1000.0, 1), "samples": samples,
                "disturbed": (bool(marks["other_rules_seen"]) or any(not f["worked"] for f in fired)
                              or len(fired) < len(events))}
    return rows, happened


# ---------------------------------------------------------------- the router's settings, kept and put back
def settings_now(router):
    status, d, _ = http(router + "/router/config", timeout=5.0)
    if status != 200 or not d:
        return None
    keep = {"policy": d.get("policy"), "preferred": d.get("preferred") or ""}
    for key in ("margin", "window_s", "cooldown_s", "return_margin", "stall_ms"):
        if (d.get("settings") or {}).get(key) is not None:
            keep[key] = d["settings"][key]
    return keep


def set_rule(router, rule, preferred):
    """Set the rule so that every case starts alike. router.py clears the adaptive rule's state (its active
    box, its list of switches) only when the rule CHANGES, and it picks the active box from the home box that
    was set BEFORE the change. So: first another rule together with the home box, then the rule."""
    other = "round_robin" if rule != "round_robin" else "static"
    status, d, _ = http(router + "/router/config", {"policy": other, "preferred": preferred}, timeout=5.0)
    if status == 200:
        status, d, _ = http(router + "/router/config", {"policy": rule}, timeout=5.0)
    return status == 200, d


def wait_healthy(router, seconds=90.0):
    """Wait until the router sees every box as healthy. Returns the names that are still not, after the wait."""
    until = time.time() + seconds
    while True:
        status, d, _ = http(router + "/router/status", timeout=5.0)
        bad = sorted(n for n, b in ((d or {}).get("backends") or {"router": {}}).items() if not b.get("healthy"))
        if not bad or time.time() >= until:
            return bad
        time.sleep(1.0)


def job_state(stack):
    _, d, _ = http(stack + "/status", timeout=5.0)
    return (((d or {}).get("jobs") or {}).get("finetune") or {}).get("state")


def stop_job_and_wait(stack, seconds=60.0):
    """Stop the fine-tune if it runs, and wait until the supervisor says it is gone. True if it is gone."""
    if job_state(stack) != "running":
        return True
    http(stack + "/jobs/finetune/stop", post=True, timeout=45.0)
    until = time.time() + seconds
    while time.time() < until:
        if job_state(stack) != "running":
            return True
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------- the files
def markdown_table(summaries):
    head = "| " + " | ".join(h for _, h in COLUMNS) + " |"
    sep = "|" + "|".join("---" if k == "policy" else "---:" for k, _ in COLUMNS) + "|"
    body = ["| " + " | ".join(str(s[k]) for k, _ in COLUMNS) + " |" for s in summaries]
    return "\n".join([head, sep] + body)


def windows_table(summaries):
    """p95 of each rule in each window of the case: where the rules differ."""
    used = [w for w in WINDOWS if any(s["windows"].get(w, {}).get("requests") for s in summaries)]
    head = "| policy | " + " | ".join("%s p95 ms" % w for w in used) + " |"
    sep = "|---|" + "|".join("---:" for _ in used) + "|"
    body = []
    for s in summaries:
        cells = []
        for w in used:
            n = s["windows"].get(w) or {}
            cells.append("%s (%d)" % (n.get("p95_ms"), n.get("requests", 0)) if n.get("requests") else "")
        body.append("| %s | %s |" % (s["policy"], " | ".join(cells)))
    return "\n".join([head, sep] + body)


def write_outputs(out, summaries, meta):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"meta": meta, "results": summaries}, f, indent=2)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k, _ in COLUMNS])
        w.writeheader()
        w.writerows(dict((k, s[k]) for k, _ in COLUMNS) for s in summaries)
    md = ["# Routing benchmark on the real stack: %s" % meta["started"], "",
          "- made by bench/fault_table.py against %s (model %s)" % (meta["router"], meta.get("model") or "not read"),
          "- trace: seed %s, %d requests over %d s (about %s requests a second), the same for every rule"
          % (meta["seed"], meta["requests"], meta["duration"], meta["rps"]),
          "- events, the same for every rule: %s" % (meta["events"] or "none"),
          "- slow = a real fine-tune on that box (the supervisor's job, with %s); kill = the forecast server's "
          "process is killed and its supervisor starts it again" % (" ".join(meta["job_args"]) or "its own settings"),
          "- latency = what the client saw, answered requests only; errors = no answer or not HTTP 200", "",
          markdown_table(summaries), "",
          "p95 by window of the case (requests in brackets). \"training\" starts at the job's first training "
          "step; \"box down\" ends when the killed server answers again.", "",
          windows_table(summaries), ""]
    for s in summaries:
        m = s["marks"]
        line = "**%s**: first training step at %s s, job stopped at %s s, killed box down at %s s and answering " \
               "again at %s s (the router used it again at %s s); the sender was at most %s ms late; %d unanswered" % (
                   s["policy"], m["first_training_step_s"], m["job_stopped_s"], m["kill_box_down_s"],
                   m["kill_box_answers_again_s"], m["router_uses_it_again_s"], s["sender_latest_ms"], s["unanswered"])
        if s["disturbed"]:
            line += ". DISTURBED: " + "; ".join(
                ["the rule was changed from outside to %s" % ", ".join(m["other_rules_seen"])] * bool(m["other_rules_seen"])
                + ["event %s %s did not work (%s)" % (f["action"], f["box"], f["said"][:80])
                   for f in s["events"] if not f["worked"]])
        md += [line, ""]
        if s.get("switch_log"):
            md += ["%s switches:" % s["policy"]] + ["- %s s %s -> %s (%s)" % (x.get("t"), x.get("from"), x.get("to"), x.get("reason"))
                                                     for x in s["switch_log"]] + [""]
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write("\n".join(md))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router", default="http://router:8000")
    p.add_argument("--stack", action="append", default=[], metavar="BOX=URL",
                   help="a box's supervisor; give both (default: box-a=http://box-a:8109 and box-b=http://box-b:8110)")
    p.add_argument("--rules", default=",".join(RULES), help="rules to try, in this order (default: all five)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, default=150.0, help="seconds of traffic per rule (default 150)")
    p.add_argument("--rps", type=float, default=8.0, help="forecast requests a second (default 8; one box with "
                   "the real model serves about 15 forecasts of 48 h a second)")
    p.add_argument("--trace", help="send this saved trace (a .jsonl of loadgen.py) and not a new one")
    p.add_argument("--events", default=None, help='"T:ACTION:BOX,..." with ACTION slow, normal or kill '
                   "(default: a fine-tune on --slow-box from 1/15 to 60%% of the case, --kill-box killed at 70%%)")
    p.add_argument("--slow-box", default="box-a", help="the box that trains (default box-a: the box 'static' uses)")
    p.add_argument("--kill-box", default="box-b", help="the box whose forecast server is killed (default box-b)")
    p.add_argument("--job-args", default="--out /wx/ft/fault-table",
                   help="extra arguments for the fine-tune. They come after the supervisor's own, so --out here "
                        "wins. ('' for a job that takes none, as the stand-in of ui/dev_stack_supervised.py)")
    p.add_argument("--preferred", default="", help="home box of 'adaptive' during the run ('' = none, as in "
                   "loadgen.py's bench: adaptive then starts on the first box)")
    p.add_argument("--settle", type=float, default=8.0, help="quiet seconds after a rule is set, before its traffic")
    p.add_argument("--client-timeout", type=float, default=60.0)
    p.add_argument("--out", default="/wx/results")
    a = p.parse_args(argv)

    a.router = a.router.rstrip("/")
    stacks = dict(s.split("=", 1) for s in (a.stack or ["box-a=http://box-a:8109", "box-b=http://box-b:8110"]))
    stacks = dict((name, url.rstrip("/")) for name, url in stacks.items())
    rules = [r for r in a.rules.split(",") if r]
    for rule in rules:
        if rule not in RULES:
            raise SystemExit("unknown rule %r; the rules are %s" % (rule, ", ".join(RULES)))
    job_args = a.job_args.split()

    # What is there, and is it free.
    status, st, _ = http(a.router + "/router/status", timeout=5.0)
    if status != 200 or not st:
        raise SystemExit("the router does not answer at %s (%s)" % (a.router, status))
    boxes = list(st.get("backends") or {})
    events = parse_events(a.events if a.events is not None else default_events(a.duration, a.slow_box, a.kill_box))
    for event in events:
        if event.box not in stacks or event.box not in boxes:
            raise SystemExit("event %s names box %r, but the router has %s and --stack has %s"
                             % (event.action, event.box, boxes, sorted(stacks)))
    not_healthy = [n for n, b in st["backends"].items() if not b.get("healthy")]
    if not_healthy:
        raise SystemExit("the router does not see %s as healthy; start with two healthy boxes" % ", ".join(not_healthy))
    for name, url in sorted(stacks.items()):
        code, d, _ = http(url + "/status", timeout=5.0)
        if code != 200 or not d:
            raise SystemExit("the supervisor of %s does not answer at %s (%s)" % (name, url, code))
        if "forecast" not in (d.get("servers") or {}):
            raise SystemExit("the supervisor of %s keeps no server named 'forecast'" % name)
        if any(e.action == "slow" and e.box == name for e in events):
            job = (d.get("jobs") or {}).get("finetune")
            if job is None:
                raise SystemExit("the supervisor of %s has no job named 'finetune'" % name)
            if job.get("state") == "running":
                raise SystemExit("a fine-tune is already running on %s; this script will not touch a job it did "
                                 "not start. Stop it first, or wait for it." % name)
    kills = [e.box for e in events if e.action == "kill"]
    forecast_url = (st["backends"][kills[0]].get("url") or "").rstrip("/") if kills else None
    models = set()
    for name in boxes:                           # the model each box serves: /health has the version with its hash
        url = (st["backends"][name].get("url") or "").rstrip("/")
        _, health, _ = http(url + "/health", timeout=5.0)
        _, info, _ = http(url + "/v1/info", timeout=5.0)
        models.add(str((health or {}).get("model_version") or (info or {}).get("model") or "not read"))
    model = " and ".join(sorted(models))
    drained = [n for n, b in st["backends"].items() if b.get("drained")]
    if drained:
        raise SystemExit("%s is drained on the router (taken out of service); put it back first" % ", ".join(drained))
    before = settings_now(a.router)
    if before is None:
        raise SystemExit("the router's settings cannot be read (GET /router/config); they could not be put back, "
                         "so nothing was started")

    trace = read_jsonl(a.trace) if a.trace else make_trace(a.seed, a.duration, a.rps)
    duration = trace[-1]["t"] if trace else a.duration
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(a.out, "fault-table-" + stamp)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "trace.jsonl"), trace)
    events_text = ",".join("%g:%s:%s" % (e.t, e.action, e.box) for e in events)
    print("router %s, boxes %s, model %s" % (a.router, boxes, model), flush=True)
    print("trace: %d requests over %.0f s (seed %s, about %g a second)" % (len(trace), duration, a.seed, a.rps), flush=True)
    print("events: %s\nresults: %s\nthe router's settings now (put back at the end): %s\n"
          % (events_text or "none", out, json.dumps(before)), flush=True)

    summaries = []
    try:
        for rule in rules:
            print("== %s" % rule, flush=True)
            if not all(stop_job_and_wait(stacks[e.box]) for e in events if e.action == "slow"):
                print("    the fine-tune of the case before did not stop within 60 s; stopping here", flush=True)
                break
            bad = wait_healthy(a.router)
            if bad:
                print("    the router still does not see %s as healthy after 90 s; stopping here" % ", ".join(bad), flush=True)
                break
            ok, _ = set_rule(a.router, rule, a.preferred)
            if not ok:
                print("    the router refused rule %s; skipped" % rule, flush=True)
                continue
            time.sleep(a.settle)
            rows, happened = run_case(rule, trace, events, a, stacks, boxes, forecast_url, job_args)
            _, after, _ = http(a.router + "/router/status", timeout=5.0)
            write_jsonl(os.path.join(out, "raw-%s.jsonl" % rule), rows)
            with open(os.path.join(out, "watch-%s.json" % rule), "w") as f:
                json.dump(happened["samples"], f)
            s = numbers(rows, boxes)
            s["policy"] = rule
            switch_log = ((after or {}).get("adaptive") or {}).get("switches") or []
            s["switches"] = len(switch_log) if rule == "adaptive" else ""
            s["switch_log"] = switch_log if rule == "adaptive" else []
            marks = happened["marks"]
            by_window = collections.defaultdict(list)
            for row in rows:
                by_window[window_of(row["t"], marks, duration)].append(row)
            s["windows"] = dict((w, numbers(by_window[w], boxes)) for w in WINDOWS if by_window[w])
            s.update(marks=marks, events=happened["events"], unanswered=happened["unanswered"],
                     sender_latest_ms=happened["sender_latest_ms"], disturbed=happened["disturbed"])
            summaries.append(s)
            train = s["windows"].get("training") or {}
            down = s["windows"].get("box down") or {}
            print("    p50 %s ms | p95 %s | p99 %s | errors %d | A %s%% B %s%% | failovers %d | while training p95 %s"
                  " | while the box was down p95 %s%s\n"
                  % (s["p50_ms"], s["p95_ms"], s["p99_ms"], s["errors"], s["share_a_pct"], s["share_b_pct"],
                     s["failovers"], train.get("p95_ms"), down.get("p95_ms"),
                     " | DISTURBED, see table.md" if s["disturbed"] else ""), flush=True)
    finally:
        # Leave the stack as it was found: no fine-tune of ours running, the router's own settings back.
        for box in sorted(set(e.box for e in events if e.action == "slow")):
            if not stop_job_and_wait(stacks[box]):
                print("WARNING: the fine-tune on %s is still running; stop it by hand: POST %s/jobs/finetune/stop"
                      % (box, stacks[box]), flush=True)
        status, _, _ = http(a.router + "/router/config", before, timeout=5.0)
        if status != 200:
            status, _, _ = http(a.router + "/router/config",
                                {"policy": before["policy"], "preferred": before["preferred"]}, timeout=5.0)
            print("WARNING: the router refused the full settings; only the rule and the home box were sent", flush=True)
        now = settings_now(a.router)
        print("the router's settings were put back (HTTP %s): %s%s" % (
            status, json.dumps(now), "" if now == before else "  <- NOT what it had before: %s" % json.dumps(before)), flush=True)

    meta = {"started": stamp, "seed": a.seed if not a.trace else "from trace", "duration": round(duration),
            "rps": a.rps if not a.trace else round(len(trace) / max(1.0, duration), 1), "requests": len(trace),
            "events": events_text, "router": a.router, "model": model, "job_args": job_args,
            "preferred": a.preferred, "boxes": boxes, "what": "bench/fault_table.py"}
    write_outputs(out, summaries, meta)
    print(markdown_table(summaries))
    print()
    print(windows_table(summaries))
    print("\nwrote %s: table.md, summary.csv, summary.json, raw-*.jsonl, watch-*.json" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
