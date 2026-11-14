#!/usr/bin/env python3
"""Policy comparison. Runs every routing policy through every scenario with the same seeded load,
then writes the numbers, one table per scenario and one chart per scenario.

Run it from the repo root, with the mocks and the router up (tools/dev_cluster.py up). It needs
only the standard library.

    python bench/compare.py --target http://127.0.0.1:8100 --admin-a http://127.0.0.1:8101 \
        --admin-b http://127.0.0.1:8102 --policies static,round_robin,least_loaded,adaptive \
        --scenarios steady,contention,failure,recovery --duration 60 --rate 4 --seed 1 \
        --kind forecast --out bench/results/NAME

    python bench/compare.py --scenarios contention --duration 20 --out bench/results/quick \
        --admin-a http://127.0.0.1:8101 --admin-b http://127.0.0.1:8102

Written at the hackathon (Nov 13-14, 2026).

What it does:

- First it asks the router to take each policy once, so a misspelled name stops it at the start.
- For each scenario, for each policy: one run of bench/loadgen.py with the same seed. So every
  policy gets the same requests at the same times. The runs go one after another, in the order
  given.
- After each run it resets the admin hooks (a node the scenario left down or slow comes back)
  and waits 3 seconds for queues to drain. At the end it puts back the policy the router had.
- It writes into the --out folder:
    <scenario>_<policy>.jsonl                one line per request
    <scenario>_<policy>.jsonl.summary.json   that run's summary
    summary.json                             the settings, every summary, the charts' numbers
    RESULTS.md                               the settings, then one table per scenario
    <scenario>.svg                           p95 latency in 5-second windows, a line per policy
- Under each table is a one-line reading: which policy had the best p95 and by how much against
  round_robin. It is computed from the rows, not assumed.
- Percentiles use the nearest-rank method, over ok requests only (see loadgen.py).
- On the real machines there are no admin hooks. Use --scenarios steady with --events FILE.json
  (see loadgen.py). The events are then drawn on the chart.
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loadgen  # noqa: E402  (bench/loadgen.py, beside this file)

PAUSE_S = 3.0               # between runs, for queues to drain
WINDOW_S = 5.0              # the charts show the p95 latency of each window this long
BASELINE = "round_robin"    # the reading under each table compares the best policy with this one
LOG_RATIO = 30.0            # a chart whose values span more than this gets a log axis

SCENARIO_NOTES = {
    "steady": "Nothing happens: both nodes stay as they are.",
    "contention": "Node B gets 1.8 times slower a third of the way in, as if a training job shared its GPU, "
                  "and is back to normal at two thirds.",
    "failure": "Node B dies halfway. Nothing brings it back, unless the router's heal action does.",
    "recovery": "Node B dies a third of the way in and comes back at two thirds.",
    "burst": "The arrival rate is 4 times higher between one third and one half of the run.",
}

# Chart colors. One fixed color and marker shape per policy, so a policy looks the same in every chart.
W, H = 900, 360
SURFACE, INK, INK_SOFT, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SHAPES = ("circle", "square", "triangle", "diamond")
KNOWN_POLICIES = ("adaptive", "round_robin", "least_loaded", "static")      # these own the first four colors


# ---------------------------------------------------------------- RESULTS.md

def ms(value):
    return "-" if value is None else "%.1f" % value


def bold(text, yes):
    return "**%s**" % text if yes else text


def p95_of(row):
    return row["summary"]["latency_ms"]["p95"]


def recover_text(summary):
    if summary.get("fail_event_s") is None:
        return "-"
    if summary.get("time_to_recover_s") is None:
        return "not in run"
    return "%.1f" % summary["time_to_recover_s"]


def table_lines(rows):
    """The Markdown table of one scenario: a row per policy. Best p95 and fewest errors in bold."""
    names = sorted(set(name for row in rows for name in (row["summary"].get("share") or {})))
    head = ["policy", "ok", "errors", "p50 ms", "p95 ms", "p99 ms", "mean ms",
            "share " + " / ".join(names) if names else "share", "retries", "failovers", "switches", "recover s"]
    have = [p95_of(row) for row in rows if p95_of(row) is not None]
    best = min(have) if have else None
    fewest = min(row["summary"]["errors"] for row in rows)
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for row in rows:
        s = row["summary"]
        lat = s["latency_ms"]
        share = s.get("share") or {}
        delta = (s.get("router") or {}).get("delta") or {}

        def count(key):
            return "-" if delta.get(key) is None else "%d" % delta[key]

        cells = [row["policy"], "%d" % s["ok"], bold("%d" % s["errors"], s["errors"] == fewest),
                 ms(lat["p50"]), bold(ms(lat["p95"]), lat["p95"] is not None and lat["p95"] == best),
                 ms(lat["p99"]), ms(lat["mean"]),
                 " / ".join("%.0f%%" % (100.0 * share.get(name, 0.0)) for name in names) or "-",
                 "%d" % s["retries"], count("failovers"), count("switches"), recover_text(s)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def reading(rows):
    """One line: which policy had the best p95, and by how much against round_robin. All computed."""
    have = [(p95_of(row), row["policy"]) for row in rows if p95_of(row) is not None]
    if not have:
        return "Reading: no policy got an ok answer, so there is no p95 to compare."
    best_p95, best = min(have, key=lambda pair: pair[0])
    base = dict((policy, value) for value, policy in have).get(BASELINE)
    others = [pair for pair in have if pair[1] != best]
    if best == BASELINE and others:
        value, policy = min(others, key=lambda pair: pair[0])
        text = ("Reading: %s itself had the best p95, %.1f ms. The next best, %s, was %.1f%% higher (%.1f ms)."
                % (BASELINE, best_p95, policy, 100.0 * (value - best_p95) / best_p95 if best_p95 > 0 else 0.0, value))
    elif best == BASELINE:
        text = "Reading: %s was the only policy with ok answers (p95 %.1f ms)." % (BASELINE, best_p95)
    elif base is None or base <= 0:
        text = ("Reading: %s had the best p95, %.1f ms. There is no %s p95 here to compare it with."
                % (best, best_p95, BASELINE))
    else:
        text = ("Reading: %s had the best p95, %.1f ms: %.1f%% lower than %s (%.1f ms)."
                % (best, best_p95, 100.0 * (base - best_p95) / base, BASELINE, base))
    errors = dict((row["policy"], row["summary"]["errors"]) for row in rows)
    fewest = min(errors.values())
    if errors[best] > fewest:
        cleanest = [row["policy"] for row in rows if errors[row["policy"]] == fewest][0]
        text += (" But %s had %d errors (the fewest was %d, %s), and p95 counts ok answers only."
                 % (best, errors[best], fewest, cleanest))
    return text


def results_markdown(settings, runs, written_at):
    """The whole RESULTS.md as text."""
    s = settings
    lines = [
        "# Routing policies compared", "",
        "Written by `bench/compare.py` on %s." % written_at, "",
        "## Settings", "",
        "- target: `%s`" % s["target"],
        "- load: %s requests, %g per second on average (Poisson arrivals), %g s per run, seed %d"
        % (s["kind"], s["rate"], s["duration_s"], s["seed"]),
        "- policies, in run order: %s" % ", ".join(s["policies"]),
        "- scenarios, in run order: %s" % ", ".join(s["scenarios"]),
        "- admin hooks: A `%s`, B `%s`" % (s["admin_a"] or "-", s["admin_b"] or "-"),
        "- per request: timeout %g s; shed when %d are in flight" % (s["timeout_s"], s["max_inflight"]),
    ]
    if s["kind"] != "forecast":
        lines.append("- LLM: %d sessions, %s, model `%s`"
                     % (s["sessions"], "streamed" if s["stream"] else "not streamed", s["model"]))
    if s["events_file"]:
        lines.append("- extra events from `%s`" % s["events_file"])
    lines += [
        "",
        "How to read the tables:", "",
        "- Every policy got the same requests at the same times (same seed, open loop).",
        "- ok is an HTTP 2xx answer. errors is everything else: HTTP errors, timeouts, refused, shed.",
        "- The latency columns are in milliseconds, over ok requests only. Percentiles use the nearest-rank method.",
        "- share is which backend answered. retries is the sum of the `X-Router-Retries` headers.",
        "- failovers and switches are the router's own counters, after the run minus before.",
        "- switches is how often a policy changed its choice of backend: for adaptive its preferred backend, "
        "for least_loaded the lowest score from one request to the next, for static the first healthy one. "
        "round_robin has no choice to change. A request that overflows from adaptive's preferred backend "
        "to the other one is not a switch.",
        "- recover s is the time from the fail event to the first 5 seconds with no error. "
        "It is 0.0 when no request failed at all: the router's retry hid the dead node.",
        "- Bold marks the best p95 and the fewest errors in each table.",
        "- Each chart shows the p95 latency of the ok requests sent in each %g-second window." % WINDOW_S,
        "",
    ]
    for scenario in s["scenarios"]:
        rows = [run for run in runs if run["scenario"] == scenario]
        if not rows:
            continue
        events = (rows[0]["summary"].get("events") or [])
        lines += ["## %s" % scenario, ""]
        if scenario in SCENARIO_NOTES and (scenario != "steady" or not events):
            lines += [SCENARIO_NOTES[scenario], ""]
        if events:
            lines += ["Events (times from the first run): "
                      + "; ".join("%s at %.1f s" % (e["name"], e["t_actual"]) for e in events) + ".", ""]
        lines += table_lines(rows)
        lines += ["", reading(rows), ""]
        notes = ["- Note, %s: %s" % (row["policy"], warning)
                 for row in rows for warning in (row["summary"].get("warnings") or [])]
        if notes:
            lines += notes + [""]
        lines += ["![%s: p95 latency over time](%s.svg)" % (scenario, safe_name(scenario)), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- the SVG chart

def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def nice_step(raw):
    """The next round step (1, 2 or 5 times a power of ten) at or above raw."""
    power = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * power * (1 + 1e-9):
            return m * power
    return 10 * power


def y_axis(values):
    """Pick the y scale for these values: (bottom, top, ticks, log?).

    A plain axis from zero when the values are of one size. A log axis when they span more than
    LOG_RATIO, so one overloaded policy does not flatten the others.
    """
    low, high = min(values), max(max(values), 1e-6)
    if low > 0 and high / low > LOG_RATIO:
        marks = [(m * 10.0 ** k, m) for k in range(int(math.floor(math.log10(low))) - 1, int(math.ceil(math.log10(high))) + 2)
                 for m in (1, 2, 5)]                 # 1, 2, 5, 10, 20, 50, ... around the values
        bottom = max(v for v, _ in marks if v <= low * (1 + 1e-9))
        top = min(v for v, _ in marks if v >= high * (1 - 1e-9))
        ticks = [v for v, _ in marks if bottom <= v <= top]
        if len(ticks) > 10:                          # many powers of ten: label only those, and the two ends
            ticks = [v for v, m in marks if bottom <= v <= top and (m == 1 or v in (bottom, top))]
        return bottom, top, ticks, True
    step = nice_step(high / 5.0)
    count = int(math.ceil(high / step - 1e-9))
    return 0.0, step * count, [step * n for n in range(count + 1)], False


def series_styles(policies):
    """A color and a marker shape for each policy. The four known policies always get the same."""
    spare = list(range(len(KNOWN_POLICIES), len(SERIES_COLORS)))
    styles = {}
    for policy in policies:
        if policy in KNOWN_POLICIES:
            slot = KNOWN_POLICIES.index(policy)
        elif spare:
            slot = spare.pop(0)
        else:
            slot = None                              # out of colors: gray
        color = MUTED if slot is None else SERIES_COLORS[slot]
        styles[policy] = (color, SHAPES[(slot or 0) % len(SHAPES)])
    return styles


def marker(shape, x, y, color):
    """One point of a line: a small filled shape with a ring in the background color."""
    ring = 'fill="%s" stroke="%s" stroke-width="2"' % (color, SURFACE)
    if shape == "square":
        return '<rect x="%.1f" y="%.1f" width="8.4" height="8.4" %s/>' % (x - 4.2, y - 4.2, ring)
    if shape == "triangle":
        corners = ((x, y - 5.6), (x + 5.2, y + 3.6), (x - 5.2, y + 3.6))
    elif shape == "diamond":
        corners = ((x, y - 5.8), (x + 5.8, y), (x, y + 5.8), (x - 5.8, y))
    else:
        return '<circle cx="%.1f" cy="%.1f" r="4.6" %s/>' % (x, y, ring)
    return '<polygon points="%s" %s/>' % (" ".join("%.1f,%.1f" % corner for corner in corners), ring)


def chart_events(rows):
    """The scenario's events as (seconds, label). The time is the mean of the runs' actual times."""
    logs = [row["summary"].get("events") or [] for row in rows]
    events = []
    for n, first in enumerate(logs[0] if logs else []):
        times = [log[n]["t_actual"] for log in logs
                 if len(log) > n and log[n].get("name") == first.get("name") and loadgen.is_number(log[n].get("t_actual"))]
        if times:
            events.append((sum(times) / len(times), str(first.get("label") or first.get("name"))))
    return events


def svg_chart(scenario, rows, settings):
    """One scenario's chart as SVG text: p95 latency per window over time, one line per policy."""
    duration = float(settings["duration_s"])
    left, right, top, bottom = 66.0, W - 22.0, 102.0, H - 50.0
    styles = series_styles([row["policy"] for row in rows])
    values = [w["p95_ms"] for row in rows for w in row["windows"] if w["p95_ms"] is not None]
    low, high, ticks, log = y_axis(values) if values else (0.0, 1.0, [0.0, 1.0], False)
    unit, per_unit = ("s", 1000.0) if high >= 10000 else ("ms", 1.0)

    def x_of(t):
        return left + (right - left) * t / duration

    def y_of(v):
        if log:
            part = (math.log10(v) - math.log10(low)) / (math.log10(high) - math.log10(low))
        else:
            part = (v - low) / (high - low)
        return bottom - (bottom - top) * part

    font = "system-ui, -apple-system, 'Segoe UI', sans-serif"
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
           'font-family="%s" font-size="12" role="img">' % (W, H, W, H, font),
           "<title>%s: p95 latency over time, one line per policy</title>" % esc(scenario),
           '<rect width="%d" height="%d" fill="%s"/>' % (W, H, SURFACE),
           '<text x="16" y="26" font-size="16" font-weight="600" fill="%s">%s: p95 latency in %g-second windows</text>'
           % (INK, esc(scenario), WINDOW_S),
           '<text x="16" y="45" fill="%s">%s requests, %g per second, seed %s. Each point: the p95 of the ok '
           'requests sent in that window.</text>'
           % (INK_SOFT, esc(settings["kind"]), settings["rate"], settings["seed"])]

    # The legend: a line sample with the policy's marker, then its name.
    x = 16.0
    for row in rows:
        color, shape = styles[row["policy"]]
        out.append('<line x1="%.1f" y1="66" x2="%.1f" y2="66" stroke="%s" stroke-width="2" stroke-linecap="round"/>'
                   % (x, x + 26, color))
        out.append(marker(shape, x + 13, 66, color))
        out.append('<text x="%.1f" y="70" fill="%s">%s</text>' % (x + 33, INK_SOFT, esc(row["policy"])))
        x += 33 + 7.0 * len(row["policy"]) + 22

    # The y axis: grid lines and their values.
    for tick in ticks:
        y = y_of(tick)
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
                   % (left, y, right, y, GRID))
        out.append('<text x="%.1f" y="%.1f" text-anchor="end" fill="%s">%s</text>'
                   % (left - 8, y + 4, MUTED, "{:,g}".format(tick / per_unit)))
    out.append('<text transform="translate(16 %.1f) rotate(-90)" text-anchor="middle" fill="%s">'
               'p95 latency (%s%s)</text>' % ((top + bottom) / 2, INK_SOFT, unit, ", log scale" if log else ""))

    # The x axis: seconds since the run began.
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
               % (left, bottom, right, bottom, AXIS))
    step = nice_step(duration / 8.0)
    for n in range(int(duration / step + 1e-9) + 1):
        tx = x_of(n * step)
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
                   % (tx, bottom, tx, bottom + 5, AXIS))
        out.append('<text x="%.1f" y="%.1f" text-anchor="middle" fill="%s">%g</text>' % (tx, bottom + 19, MUTED, n * step))
    out.append('<text x="%.1f" y="%.1f" text-anchor="middle" fill="%s">time since the run began (s)</text>'
               % ((left + right) / 2, H - 12, INK_SOFT))

    # The events: a vertical line each, with a short label above the plot.
    # A label that would run into the one before it goes one step lower, inside the plot.
    # The labels are drawn last, on top of everything, each with a rim in the background color.
    labels = []
    last_end, level = -1e9, 0
    for t, label in chart_events(rows):
        ex = x_of(min(max(t, 0.0), duration))
        width = 6.4 * len(label)
        to_left = ex + 5 + width > right
        start = ex - 5 - width if to_left else ex + 5
        level = (level + 1) % 3 if start < last_end + 8 else 0
        last_end = start + width
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
                   % (ex, top - 18, ex, bottom, MUTED))
        labels.append('<text x="%.1f" y="%.1f" text-anchor="%s" font-size="11" fill="%s" stroke="%s" '
                      'stroke-width="4" stroke-linejoin="round" paint-order="stroke">%s</text>'
                      % (ex - 5 if to_left else ex + 5, top - 8 if level == 0 else top - 2 + 14 * level,
                         "end" if to_left else "start", INK_SOFT, SURFACE, esc(label)))

    if not values:
        out.append('<text x="%.1f" y="%.1f" text-anchor="middle" fill="%s">no ok answers in this scenario</text>'
                   % ((left + right) / 2, (top + bottom) / 2, INK_SOFT))

    # The lines first, then the markers on top of them.
    shapes = []
    for row in rows:
        color, shape = styles[row["policy"]]
        windows = row["windows"]
        every = max(1, int(math.ceil(len(windows) / 30.0)))      # at most about 30 markers per line
        path, pen_down = [], False
        for n, w in enumerate(windows):
            if w["p95_ms"] is None:
                pen_down = False                                  # a gap: no ok request in this window
                continue
            px, py = x_of((w["t0"] + w["t1"]) / 2.0), y_of(w["p95_ms"])
            path.append("%s%.1f %.1f" % ("L" if pen_down else "M", px, py))
            alone = not pen_down and (n + 1 == len(windows) or windows[n + 1]["p95_ms"] is None)
            pen_down = True
            if n % every == 0 or alone or n + 1 == len(windows):
                shapes.append("<g><title>%s, %g to %g s: p95 %.1f ms (%d ok, %d errors)</title>%s</g>"
                              % (esc(row["policy"]), w["t0"], w["t1"], w["p95_ms"], w["ok"], w["errors"],
                                 marker(shape, px, py, color)))
        if path:
            out.append('<path d="%s" fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" '
                       'stroke-linecap="round"/>' % (" ".join(path), color))
    out += shapes + labels
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- running it all

def safe_name(text):
    """A name that is safe inside a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def split_list(text, option):
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise loadgen.SetupError("%s is empty" % option)
    if len(set(names)) != len(names):
        raise loadgen.SetupError("%s lists a name twice: %s" % (option, text))
    return names


def one_line(summary):
    """A run's result in one line, for the progress output."""
    lat = summary["latency_ms"]
    return "ok %d of %d, errors %d, p50 %s ms, p95 %s ms, p99 %s ms" % (
        summary["ok"], summary["requests"], summary["errors"], ms(lat["p50"]), ms(lat["p95"]), ms(lat["p99"]))


def reset_admins(args):
    """Put the mocks back to normal, so a node the scenario left down or slow comes back."""
    for url in (args.admin_a, args.admin_b):
        if url:
            try:
                loadgen.call_json("POST", url.rstrip("/") + "/admin/reset", {})
            except loadgen.SetupError:
                pass                    # not an address at all: the run itself says so


def compare(args):
    """Run every (scenario, policy) pair and write the results folder. Returns the list of runs."""
    policies = split_list(args.policies, "--policies")
    scenarios = split_list(args.scenarios, "--scenarios")
    for scenario in scenarios:
        if scenario not in loadgen.SCENARIOS:
            raise loadgen.SetupError("unknown scenario %r (known: %s)" % (scenario, ", ".join(loadgen.SCENARIOS)))
        if scenario in ("contention", "failure", "recovery") and not args.admin_b:
            raise loadgen.SetupError("the %s scenario posts to node B's admin hooks: give --admin-b URL. "
                                     "On real machines use --scenarios steady with --events FILE.json." % scenario)
    target = args.target.rstrip("/")
    status, problem = loadgen.router_status(target)
    if problem:
        raise loadgen.SetupError("the target %s is not answering: %s" % (target, problem))
    policy_before = status.get("policy") if isinstance(status, dict) else None

    total = len(scenarios) * len(policies)
    print("%d runs of %g s each: about %.1f minutes" % (total, args.duration, total * (args.duration + PAUSE_S + 1) / 60.0),
          flush=True)
    runs = []
    try:
        for policy in policies:         # find a policy the router does not know now, not ten minutes in
            code, _, error = loadgen.call_json("POST", target + "/router/policy", {"policy": policy})
            if not loadgen.is_ok_status(code):
                raise loadgen.SetupError("%s/router/policy refused the policy %r: %s. Nothing was run. "
                                         "Is the target the router?"
                                         % (target, policy, "HTTP %s" % code if loadgen.is_number(code) else error))
        for scenario in scenarios:
            for policy in policies:
                if runs:
                    time.sleep(PAUSE_S)
                print("[%d/%d] %s / %s" % (len(runs) + 1, total, scenario, policy), flush=True)
                opts = argparse.Namespace(**vars(args))
                opts.scenario, opts.policy = scenario, policy
                opts.out = os.path.join(args.out, "%s_%s.jsonl" % (safe_name(scenario), safe_name(policy)))
                try:
                    summary, records = loadgen.run(opts, quiet=True)
                except loadgen.SetupError as e:
                    raise loadgen.SetupError("run %d of %d (%s / %s) could not start: %s. The report was not "
                                             "written." % (len(runs) + 1, total, scenario, policy, e))
                runs.append({"scenario": scenario, "policy": policy, "jsonl": os.path.basename(opts.out),
                             "summary": summary, "windows": loadgen.window_p95(records, args.duration, WINDOW_S)})
                print("        " + one_line(summary), flush=True)
                for warning in summary.get("warnings") or []:
                    print("        WARNING: " + warning, flush=True)
                reset_admins(args)
    finally:                            # also after Ctrl-C or an error: leave the cluster as it was found
        reset_admins(args)
        if isinstance(policy_before, str):
            loadgen.call_json("POST", target + "/router/policy", {"policy": policy_before})

    written_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    settings = {"target": target, "kind": args.kind, "duration_s": args.duration, "rate": args.rate,
                "seed": args.seed, "policies": policies, "scenarios": scenarios, "admin_a": args.admin_a,
                "admin_b": args.admin_b, "timeout_s": args.timeout, "max_inflight": args.max_inflight,
                "sessions": args.sessions, "stream": bool(args.stream), "model": args.model,
                "events_file": args.events, "pause_s": PAUSE_S, "window_s": WINDOW_S}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"written_at": written_at, "settings": settings, "runs": runs}, f, indent=2)
        f.write("\n")
    text = results_markdown(settings, runs, written_at)
    with open(os.path.join(args.out, "RESULTS.md"), "w") as f:
        f.write(text)
    for scenario in scenarios:
        rows = [run for run in runs if run["scenario"] == scenario]
        with open(os.path.join(args.out, "%s.svg" % safe_name(scenario)), "w") as f:
            f.write(svg_chart(scenario, rows, settings))
    print("")
    print(text)
    print("wrote %s: RESULTS.md, summary.json, %d charts (.svg), %d runs (.jsonl and summaries)"
          % (args.out, len(scenarios), len(runs)), flush=True)
    return runs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run every routing policy through every scenario with the same load, then write "
                    "RESULTS.md, summary.json and one SVG chart per scenario.",
        epilog="Example: python bench/compare.py --admin-a http://127.0.0.1:8101 --admin-b http://127.0.0.1:8102 "
               "--scenarios steady,contention --out bench/results/first")
    loadgen.add_run_options(parser)
    parser.add_argument("--policies", default="static,round_robin,least_loaded,adaptive",
                        help="comma-separated, run in this order (default: all four)")
    parser.add_argument("--scenarios", default="steady,contention,failure,recovery",
                        help="comma-separated, from: %s (default: steady,contention,failure,recovery)"
                             % ", ".join(loadgen.SCENARIOS))
    parser.add_argument("--out", default="bench/results/compare", help="the results folder (default bench/results/compare)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        compare(args)
    except loadgen.SetupError as e:
        sys.stderr.write("compare: %s\n" % e)
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("compare: stopped by Ctrl-C. The report was not written; finished runs are in %s\n" % args.out)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
