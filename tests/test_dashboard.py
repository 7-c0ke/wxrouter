"""
Tests for the demo's live view, dashboard/index.html.

What they check:
  - the page is one plain file with no outside resources, and its script parses (node --check);
  - a tiny fake router (in this file) can serve the page at / and a canned /router/status;
  - the script mentions the contract's paths and field names;
  - with node installed, the script is run in node against a fake page and a fake router,
    with good and bad answers. It must never throw, never show "undefined" or "NaN", and its
    buttons must send the bodies the contract asks for. Without node those tests are skipped.
No browser is involved: how the page looks has to be checked by eye.

Run from the repo root:
    .venv/bin/python -m unittest tests/test_dashboard.py -v
    .venv/bin/python -m unittest tests.test_dashboard.TestScriptInNode -v

Written at the hackathon (Nov 13-14, 2026).
"""

import datetime
import http.server
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "dashboard", "index.html")
NODE = shutil.which("node")
DASH = chr(0x2013)                   # the en dash the page shows for a missing number
ROUTER = "http://router.test:8100"   # only a name: the node run fakes fetch, so nothing is sent
OHARE = {"lat": 41.9602, "lon": -87.9316}
POLICIES = ["static", "round_robin", "least_loaded", "adaptive"]

# Paths and field names from docs/CONTRACT.md that the page has to use.
CONTRACT_NAMES = [
    "/router/status", "/router/policy", "/router/drain", "/v1/ensemble", "/v1/forecast", "/metrics",
    "policy", "counters", "pools", "decisions", "preferred", "backends", "healthy", "draining", "healing",
    "inflight", "latency_ewma_ms", "score", "routed", "last_error", "probe_data", "queued_units", "queued",
    "gpu_util_pct", "busy_other", "mem_available_gb", "failovers", "switches", "heals", "errors",
    "util_pct", "temp_c", "power_w", "valid_times", "t2m_c_mean", "t2m_c_min", "t2m_c_max", "truth_t2m_c",
    "shards", "retried", "init_time", "lead_hours", "with_truth",
]


def read_page():
    with open(PAGE, encoding="utf-8") as handle:
        return handle.read()


def page_script():
    """The page's one inline script."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", read_page(), re.S)
    if len(scripts) != 1:
        raise AssertionError("expected one <script> in the page, found %d" % len(scripts))
    return scripts[0]


# ---- Canned answers, shaped as the contract says ----

def backend(name, url, probe, probe_data, **changes):
    """One backend entry of /router/status."""
    entry = {"name": name, "url": url, "probe": probe, "healthy": True, "draining": False, "healing": False,
             "inflight": 1, "latency_ewma_ms": 210.0, "score": 0.42, "routed": 60, "failures": 0,
             "last_error": None, "probe_data": probe_data}
    entry.update(changes)
    return entry


def decision(ts, pool, name, kind, reason):
    return {"ts": ts, "pool": pool, "backend": name, "policy": "adaptive", "kind": kind, "reason": reason,
            "retries": 0}


FORECAST_PROBE = {"queued": 0, "queued_units": 0, "gpu_util_pct": 96, "busy_other": False, "slowdown": 1.0,
                  "mem_available_gb": 99.6, "ms_per_unit": 10.2}
LLM_PROBE = {"queued": 2, "running": 1}

# Node A is fine and preferred. Node B has a training job on its GPU, and its LLM server is drained.
STATUS = {
    "policy": "adaptive", "uptime_s": 120.5,
    "counters": {"routed": 100, "failovers": 1, "switches": 2, "heals": 0, "errors": 0},
    "pools": {
        "forecast": {"preferred": "A", "backends": [
            backend("A", "http://127.0.0.1:8101", "forecast", FORECAST_PROBE),
            backend("B", "http://127.0.0.1:8102", "forecast",
                    dict(FORECAST_PROBE, queued=1, queued_units=40, busy_other=True, slowdown=1.8),
                    inflight=3, latency_ewma_ms=380.4, score=123.456, routed=40)]},
        "llm": {"preferred": "A", "backends": [
            backend("A", "http://127.0.0.1:8001", "llamacpp", LLM_PROBE, latency_ewma_ms=2500.0, score=15.25),
            backend("B", "http://127.0.0.1:8002", "llamacpp", LLM_PROBE, draining=True)]},
    },
    # One decision of every kind, newest last. The newest carries markup, to check it is shown as text.
    "decisions": [
        decision(1794550000.1, "forecast", "A", "route", "preferred"),
        decision(1794550001.0, "forecast", "B", "unhealthy", "3 failures in a row"),
        decision(1794550001.5, "forecast", "A", "failover", "B refused the connection"),
        decision(1794550002.0, "forecast", "B", "heal", "ran the heal action, attempt 1"),
        decision(1794550003.0, "forecast", "B", "recovered", "probe ok"),
        decision(1794550004.0, "llm", "A", "switch", "A beat B by 31% for 5.0 s"),
        decision(1794550005.0, "llm", "B", "drain", "drained by the operator"),
        decision(1794550006.0, None, None, "policy", "policy set to adaptive"),
        decision(1794550007.0, "forecast", None, "fanout", "20 members: A 12, B 8"),
        decision(1794550008.0, "forecast", "A", "route", "newest <b>one</b>"),
    ],
}

# GET /metrics from a forecast server. GPU use and free memory differ from the router's probe above,
# so a test can tell which of the two the page is showing.
METRICS = {
    "node": "A", "ts": 1794550000.0, "inflight": 1, "queued": 3, "queued_units": 120, "served_total": 120,
    "errors_total": 0, "units_total": 5400, "ms_per_unit": 10.2,
    "latency_ms": {"p50": 210.0, "p95": 900.0, "ewma": 260.0},
    "gpu": {"util_pct": 97, "temp_c": 70, "power_w": 60.0, "proc_mem_mb": 7115},
    "host": {"mem_available_gb": 101.2, "load1": 1.9},
    "busy_other": False, "slowdown": 1.0,
}


def forecast_answer(init_time, hours, members, **point):
    """The answer to POST /v1/forecast or /v1/ensemble, with made-up temperatures."""
    steps = list(range(hours // 6))
    start = datetime.datetime.strptime(init_time, "%Y-%m-%dT%H")
    mean = [round(3.0 - 0.4 * i, 2) for i in steps]
    answer = {
        "request_id": "abc", "node": "A", "model": "afno-19m", "init_time": init_time, "members": members,
        "member_offset": 0, "leads_h": [6 * (i + 1) for i in steps],
        "valid_times": [(start + datetime.timedelta(hours=6 * (i + 1))).strftime("%Y-%m-%dT%H") for i in steps],
        "point": dict(OHARE, t2m_c=mean, t2m_c_mean=mean, t2m_c_min=[v - 1.5 for v in mean],
                      t2m_c_max=[v + 2.0 for v in mean], wind10_ms_mean=[4.0 for _ in steps],
                      msl_hpa_mean=[1013.0 for _ in steps]),
        "units": members * len(steps), "timing_ms": {"queue": 0.4, "gpu": 212.0, "total": 214.1},
    }
    answer["point"].update(point)
    return answer


ENSEMBLE = dict(forecast_answer("2026-11-13T00", 48, 20), shards=[
    {"backend": "A", "members": 12, "member_offset": 0, "ms": 1300, "retried": False},
    {"backend": "B", "members": 8, "member_offset": 12, "ms": 2100, "retried": True}])
REPLAY = forecast_answer("2019-01-27T12", 72, 1,
                         truth_t2m_c=[2.5, 1.0, None, -3.0, -8.0, -14.5, -20.0, -24.0, None, -26.5, -22.0, -18.0])

# ---- Answers a careless page would choke on ----
# Each answer is "DOWN" (no answer at all), {"json": ...} or {"raw": "text"}, with an optional "code".

BAD_STATUSES = [
    ("http 500", {"code": 500, "json": {"error": "boom"}}),
    ("not json", {"raw": "<html>oops</html>"}),
    ("empty body", {"raw": ""}),
    ("json null", {"raw": "null"}),
    ("json list", {"raw": "[1, 2]"}),
    ("json number", {"raw": "7"}),
    ("empty object", {"json": {}}),
    ("top level nulls", {"json": dict.fromkeys(STATUS)}),
    ("top level wrong types", {"json": {"policy": 7, "uptime_s": "long", "counters": [1], "pools": [],
                                        "decisions": "none"}}),
    ("null fields", {"json": {
        "policy": "adaptive", "counters": dict.fromkeys(STATUS["counters"]),
        "pools": {"forecast": {"preferred": None, "backends": [
                      dict.fromkeys(STATUS["pools"]["forecast"]["backends"][0]), {"name": "A"},
                      dict(dict.fromkeys(STATUS["pools"]["forecast"]["backends"][0]), name="B",
                           url="http://127.0.0.1:8102", probe_data=dict.fromkeys(FORECAST_PROBE)),
                      None, "A", 7, []]},
                  "llm": {"backends": None}},
        "decisions": [None, {}, 5, "x", [], dict.fromkeys(STATUS["decisions"][0])]}}),
    ("wrong types inside", {"json": {
        "policy": {"name": "x"},
        "counters": {"routed": "12", "failovers": True, "switches": [], "heals": {}, "errors": "NaN"},
        "pools": {"forecast": {"preferred": 3, "backends": {"A": {}}},
                  "llm": {"preferred": ["A"], "backends": [
                      {"name": "A", "url": 5, "healthy": "yes", "draining": 1, "healing": "no", "inflight": "1",
                       "latency_ewma_ms": "fast", "score": [], "routed": {}, "last_error": {"code": 5},
                       "probe_data": "none"},
                      {"name": 5}, {"name": ""}, {"name": None}]}},
        "decisions": [{"ts": "now", "pool": 1, "backend": 2, "kind": 3, "reason": {"a": 1}},
                      {"ts": 1e300, "kind": "switch"}, {"ts": -1e300, "kind": "constructor"},
                      {"ts": 0, "kind": ["heal"], "reason": ""}]}}),
    # What Python's json.dumps prints for float("nan") and float("inf"). It is not JSON.
    ("python nan", {"raw": '{"policy": "adaptive", "counters": {"routed": NaN, "errors": -Infinity},'
                           ' "pools": {"forecast": {"preferred": "A", "backends": [{"name": "A", "healthy": true,'
                           ' "score": Infinity, "latency_ewma_ms": NaN, "routed": 5, "inflight": 2}]}},'
                           ' "decisions": [{"ts": NaN, "kind": "route", "reason": "ok"}]}'}),
    ("states", {"json": {
        "policy": "least_loaded", "counters": {"routed": 7},
        "pools": {"forecast": {"preferred": "A", "backends": [
                      backend("A", "http://127.0.0.1:8101", "forecast", None, healthy=False, failures=3,
                              last_error="connection refused <script>"),
                      backend("B", "http://127.0.0.1:8102", "forecast", FORECAST_PROBE, healthy=False,
                              healing=True, last_error="timed out")]},
                  "llm": {"preferred": "B", "backends": [
                      backend("A", "http://127.0.0.1:8001", "vllm", LLM_PROBE, draining=True), {"name": "B"}]}},
        "decisions": []}}),
    # Round robin prefers nobody: the router sends null, and a name that is in no pool marks nothing.
    ("nobody preferred", {"json": dict(STATUS, policy="round_robin", pools={
        "forecast": dict(STATUS["pools"]["forecast"], preferred=None),
        "llm": dict(STATUS["pools"]["llm"], preferred="C")})}),
    ("odd names", {"json": {
        "policy": "round_robin",
        "pools": {"forecast": {"preferred": "C", "backends": [
                      backend(name, "http://127.0.0.1:8101/", "forecast", FORECAST_PROBE)
                      for name in ("C", "<b>B</b>", "A", "__proto__", "constructor", "it's \"quoted\" & odd")]},
                  "llm": {"preferred": "toString", "backends": [
                      backend("toString", "ftp://nowhere", "http", None)]}},
        "decisions": [decision(1794550009.0, "<i>pool</i>", "<b>B</b>", "<u>kind</u>", "a & b < c")]}}),
    # More decisions than the 50 the feed shows.
    ("many decisions", {"json": dict(STATUS, decisions=[
        decision(1794550000.0 + i, "forecast", "A", "route", "number %d" % i) for i in range(120)])}),
    ("absurd numbers", {"json": {
        "policy": "adaptive", "counters": {"routed": 1e300, "failovers": -1e300, "switches": 1e15, "heals": 2.5},
        "pools": {"forecast": {"preferred": "A", "backends": [
            backend("A", "http://127.0.0.1:8101", "forecast", dict(FORECAST_PROBE, queued_units=1e300),
                    inflight=1e300, latency_ewma_ms=1e300, score=-1e300, routed=1e300),
            backend("B", "http://127.0.0.1:8102", "forecast", FORECAST_PROBE, latency_ewma_ms=9e14, score=9e14,
                    routed=9e14)]}},
        "decisions": [decision(1e300, "forecast", "A", "route", 1e300),
                      decision(9e14, "forecast", "A", "heal", 5)]}}),
    ("router restarted", {"json": dict(STATUS, counters=dict.fromkeys(STATUS["counters"], 0), pools={
        name: {"preferred": "A", "backends": [dict(b, routed=1) for b in pool["backends"]]}
        for name, pool in STATUS["pools"].items()})}),
]

BAD_METRICS = [
    {"json": {}},
    {"json": dict.fromkeys(METRICS)},
    {"json": {"gpu": {"util_pct": None, "temp_c": "hot", "power_w": [], "proc_mem_mb": None},
              "host": "none", "busy_other": "yes"}},
    {"raw": "wx_inflight 1"},
    {"raw": "[1]"},
    {"raw": '{"gpu": {"util_pct": NaN, "temp_c": Infinity}, "host": {"mem_available_gb": NaN}}'},
    {"code": 503, "json": {"error": "loading"}},
    "DOWN",
]

BAD_FORECASTS = [
    ("empty", {"json": {}}),
    ("nulls", {"json": {"init_time": None, "members": None, "valid_times": None, "point": None, "shards": None,
                        "node": None}}),
    ("odd series", {"json": {
        "valid_times": [None, 5, "2019-01-27T18"], "members": "20", "node": {"x": 1}, "init_time": 5,
        "point": {"t2m_c": None, "t2m_c_mean": [None, "warm", 2.5], "t2m_c_min": [None, None, 1.0, 0.5],
                  "t2m_c_max": [3.0], "truth_t2m_c": [None, None, None]},
        "shards": [None, {}, 7, {"backend": 7, "members": "many", "ms": None, "retried": "yes"}]}}),
    ("wrong types", {"json": {"valid_times": "soon", "point": [1, 2], "shards": {"A": 1}, "members": []}}),
    ("one step", {"json": {"valid_times": ["2019-01-27T18"], "point": {
        "t2m_c": [1.5], "t2m_c_mean": [1.5], "t2m_c_min": [1.0], "t2m_c_max": [2.0], "truth_t2m_c": [1.2]}}}),
    ("absurd numbers", {"json": {
        "valid_times": ["2019-01-27T18", "2019-01-28T00"], "members": 1e300,
        "point": {"t2m_c": [1e308, -1e308], "t2m_c_mean": [9e14, -9e14], "t2m_c_min": [-9e14, -1e308],
                  "t2m_c_max": [9e14, 1e308], "truth_t2m_c": [1e300, 2.0]},
        "shards": [{"backend": "A", "members": 1e300, "ms": 1e300}]}}),
    ("not json", {"raw": "<html>"}),
    ("refused", {"code": 503, "json": {"error": "no healthy backend"}}),
    ("down", "DOWN"),
]


def node_cases():
    """What the node run does: a list of status polls, then a list of button presses."""
    good = {"json": STATUS}
    statuses = [
        {"label": "router down", "status": "DOWN", "metrics": "DOWN"},
        {"label": "good first", "status": good, "metrics": {"json": METRICS}},
        # "grow" adds to every backend's routed count before each poll, as if requests were flowing.
        {"label": "traffic", "status": good, "metrics": {"json": METRICS}, "grow": 3, "ticks": 5},
        # Polls 0.1 s apart (a button press adds a poll): they must not add bars to the chart.
        {"label": "polls close together", "status": good, "metrics": {"json": METRICS}, "grow": 1, "ticks": 3,
         "step": 100},
        # Eight polls, so the last good /metrics answer is too old to show.
        {"label": "metrics down", "status": good, "metrics": "DOWN", "ticks": 8},
    ]
    for i, answer in enumerate(BAD_METRICS):
        statuses.append({"label": "bad metrics %d" % i, "status": good, "metrics": answer})
    for i, (label, answer) in enumerate(BAD_STATUSES):
        statuses.append({"label": label, "status": answer, "metrics": BAD_METRICS[i % len(BAD_METRICS)]})
    # The laptop's clock is set back an hour: the chart starts again and still draws.
    statuses.append({"label": "clock went back", "status": good, "metrics": {"json": METRICS}, "grow": 2,
                     "ticks": 3, "jump": -3600000})
    statuses.append({"label": "good", "status": good, "metrics": {"json": METRICS}})

    clicks = [{"label": "policy " + name, "id": "pol-" + name, "answers": {"/router/policy": {"json": {}}}}
              for name in POLICIES]
    clicks += [
        {"label": "drain A", "id": "n0-forecast-drain", "answers": {"/router/drain": {"json": {}}}},
        {"label": "put B back", "id": "n1-llm-drain", "answers": {}},
        {"label": "policy refused", "id": "pol-static",
         "answers": {"/router/policy": {"code": 400, "json": {"error": "unknown policy"}}}},
        {"label": "drain, no answer", "id": "n0-llm-drain", "answers": {"/router/drain": "DOWN"}},
        {"label": "ensemble", "id": "fc-ensemble", "answers": {"/v1/ensemble": {"json": ENSEMBLE}}},
        {"label": "replay", "id": "fc-replay", "answers": {"/v1/forecast": {"json": REPLAY}}},
    ]
    for label, answer in BAD_FORECASTS:
        clicks.append({"label": "ensemble: " + label, "id": "fc-ensemble", "answers": {"/v1/ensemble": answer}})
        clicks.append({"label": "replay: " + label, "id": "fc-replay", "answers": {"/v1/forecast": answer}})
    return {"search": "?router=" + ROUTER + "/&title=Test+title", "statuses": statuses, "clicks": clicks}


# The node side. It gives the page's script a fake page, a fake clock and a fake router, plays the
# cases above, and prints what the page showed (snaps), what it sent (calls) and what went wrong (problems).
HARNESS = r"""
// usage: node harness.js dashboard.js cases.json
const fs = require("fs");
const vm = require("vm");
const code = fs.readFileSync(process.argv[2], "utf8");
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const later = setImmediate;          // kept: the timers the page sees are replaced below
const BAD = /undefined|NaN|\bnull\b|Infinity|\[object/;
const problems = [], calls = [], snaps = {}, els = {};
let drawn = [], rects = 0, tick = null, answers = {}, now = 1794550000000;

// A canvas context that takes any call, remembers the text and counts the bars of the
// last frame (a frame starts with clearRect), and objects to NaN.
const ctx = new Proxy({}, {
  get: (kept, name) => name in kept ? kept[name] : (...args) => {
    if (args.some((a) => typeof a === "number" && !isFinite(a))) {
      problems.push("canvas " + String(name) + "(" + args.join(", ") + ")");
    }
    if (name === "fillText") drawn.push(String(args[0]));
    if (name === "clearRect") rects = 0;
    if (name === "fillRect") rects += 1;
    return name === "measureText" ? {width: 10} : undefined;
  },
  set: (kept, name, value) => { kept[name] = value; return true; },
});

function el(id) {                    // every id the page asks for exists
  if (!els[id]) {
    els[id] = {textContent: "", innerHTML: "", className: "", title: "", disabled: false, onclick: null,
               clientWidth: 640, clientHeight: 300, width: 0, height: 0, getContext: () => ctx};
  }
  return els[id];
}

globalThis.window = {location: {search: cases.search, protocol: "file:", origin: "null"},
                     devicePixelRatio: 2, addEventListener: () => {}};
globalThis.document = {title: "", documentElement: {}, getElementById: el};
globalThis.getComputedStyle = () => ({fontSize: "16px"});
globalThis.setInterval = (fn) => { tick = fn; return 1; };
globalThis.setTimeout = () => 1;
globalThis.clearTimeout = () => {};
if (typeof AbortController === "undefined") {      // node older than 15
  globalThis.AbortController = function () { this.signal = {}; this.abort = () => {}; };
}
Date.now = () => now;
globalThis.fetch = (url, options) => {
  options = options || {};
  calls.push({url: url, method: options.method || "GET", type: (options.headers || {})["Content-Type"] || null,
              body: options.body === undefined ? null : JSON.parse(options.body)});
  const answer = answers[url.replace(/^https?:\/\/[^\/]+/, "")];
  if (!answer || answer === "DOWN") return Promise.reject(new TypeError("fetch failed"));
  const raw = typeof answer.raw === "string" ? answer.raw : JSON.stringify(answer.json);
  return Promise.resolve({status: answer.code || 200, text: () => Promise.resolve(raw)});
};
console.error = (...args) => problems.push("console.error: " + args.map((a) => (a && a.stack) || String(a)).join(" "));
process.on("unhandledRejection", (e) => problems.push("unhandled rejection: " + ((e && e.stack) || e)));

function snapshot(label) {           // keep what the page shows now, and note anything that looks wrong
  const snap = {"document.title": document.title, "canvas": drawn.join(" | ")};
  for (const id of Object.keys(els)) {
    for (const key of ["textContent", "innerHTML", "className", "title"]) {
      if (els[id][key] !== "") snap[id + "." + key] = els[id][key];
    }
    if (els[id].disabled) snap[id + ".disabled"] = "yes";
  }
  for (const key of Object.keys(snap)) {
    if (typeof snap[key] !== "string" || BAD.test(snap[key])) {
      problems.push(label + ": " + key + " shows " + String(snap[key]).slice(0, 160));
    }
  }
  snap.rects = rects;
  snaps[label] = snap;
  drawn = [];
}

const settle = () => new Promise((done) => later(done));
async function step(label, action) { // do one thing a browser would do, then let the promises finish
  try { action(); } catch (e) { problems.push(label + " threw: " + ((e && e.stack) || e)); }
  await settle();
  await settle();
}
function click(id) {
  if (typeof el(id).onclick !== "function") throw new Error("no click handler on #" + id);
  el(id).onclick();
}

(async () => {
  await step("load", () => vm.runInThisContext(code, {filename: "dashboard.js"}));
  snapshot("load");
  for (const c of cases.statuses) {
    answers = {"/router/status": c.status, "/metrics": c.metrics};
    now += c.jump || 0;
    for (let i = 0; i < (c.ticks || 4); i++) {
      now += c.step || 1000;
      if (c.grow) for (const p of Object.values(c.status.json.pools)) for (const b of p.backends) b.routed += c.grow;
      await step(c.label, () => tick());
    }
    snapshot(c.label);
  }
  for (const c of cases.clicks) {
    Object.assign(answers, c.answers);
    await step(c.label, () => click(c.id));
    snapshot(c.label);
  }
  console.log(JSON.stringify({problems: problems, calls: calls, snaps: snaps}));
})();
"""


class TestPageFile(unittest.TestCase):
    """Checks on the file itself. They need nothing but Python."""

    def test_file_is_there_and_says_what_it_is(self):
        self.assertTrue(os.path.isfile(PAGE), "missing " + PAGE)
        page = read_page()
        self.assertTrue(page.lower().startswith("<!doctype html>"))
        self.assertIn("Written at the hackathon (Nov 13-14, 2026).", page)
        self.assertIn("<h1 id=\"title\">Two boxes, one address</h1>", page)
        self.assertEqual(page.count("<canvas"), 2, "one canvas for traffic, one for the forecast")
        self.assertIn("Forecast O'Hare, 20 members", page)
        self.assertIn("Replay January 2019", page)
        self.assertIn("training job on this GPU", page)
        for colour in ("#4da3ff", "#ff9f43"):
            self.assertIn(colour, page, "node colour")
        self.assertLessEqual(page.count("\n"), 720, "the page was meant to stay under about 700 lines")

    def test_no_outside_resources(self):
        page = read_page()
        outside = re.findall(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//[^\s"'>]*""", page, re.I)
        self.assertEqual(outside, [], "src= and href= must not point at another server")
        self.assertEqual(re.findall(r"""url\(\s*["']?\s*(?:https?:)?//""", page, re.I), [], "no url(...) to a server")
        for tag in ("<link", "<iframe", "<img", "<script src", "@import", "@font-face"):
            self.assertNotIn(tag, page.lower(), "the page must be one file with nothing loaded from outside")

    def test_plain_ascii(self):
        # Plain ASCII cannot be garbled by a server that sends the wrong charset.
        with open(PAGE, "rb") as handle:
            lines = handle.read().split(b"\n")
        odd = [number for number, line in enumerate(lines, 1) if any(byte > 126 for byte in line)]
        self.assertEqual(odd, [], "these lines hold non-ASCII characters; use an entity or a backslash-u escape")

    def test_uses_the_contract_names(self):
        script = page_script()
        for name in CONTRACT_NAMES:
            self.assertIn(name, script, "the page's script never mentions %r" % name)
        for name in POLICIES:
            self.assertIn('id="pol-%s"' % name, read_page(), "no button for the policy %r" % name)

    def test_script_passes_node_check(self):
        if NODE is None:
            self.skipTest("node is not on the PATH")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "dashboard.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(page_script())
            done = subprocess.run([NODE, "--check", path], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)


class FakeRouter(http.server.BaseHTTPRequestHandler):
    """The two things the router does for the page: serve it at / and answer GET /router/status."""

    def do_GET(self):
        if self.path == "/":
            with open(PAGE, "rb") as handle:
                self.answer(200, "text/html; charset=utf-8", handle.read())
        elif self.path == "/router/status":
            self.answer(200, "application/json", json.dumps(STATUS).encode("utf-8"))
        else:
            self.answer(404, "application/json", b'{"error": "not found"}')

    def answer(self, code, kind, body):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                         # keep the test output quiet


class TestFakeRouter(unittest.TestCase):
    """The page as a browser would get it from the router."""

    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRouter)    # port 0: the OS picks
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05})
        self.thread.daemon = True
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def get(self, path):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))     # never through a proxy
        with opener.open(self.base + path, timeout=5) as answer:
            return answer.status, answer.headers.get("Content-Type", ""), answer.read()

    def test_page_and_status_are_served(self):
        code, kind, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertTrue(kind.startswith("text/html"), kind)
        with open(PAGE, "rb") as handle:
            self.assertEqual(body, handle.read(), "the page arrives byte for byte")
        self.assertIn(b"/router/status", body, "the page polls the status of the server it came from")

        code, kind, body = self.get("/router/status")
        self.assertEqual(code, 200)
        self.assertTrue(kind.startswith("application/json"), kind)
        status = json.loads(body.decode("utf-8"))
        self.assertEqual(status, STATUS)
        self.assertEqual(sorted(status), ["counters", "decisions", "policy", "pools", "uptime_s"])
        self.assertEqual(sorted(status["pools"]), ["forecast", "llm"])


class TestScriptInNode(unittest.TestCase):
    """Runs the page's script in node, once, then looks at what it showed and what it sent."""

    @classmethod
    def setUpClass(cls):
        if NODE is None:
            raise unittest.SkipTest("node is not on the PATH")
        files = {"harness.js": HARNESS, "dashboard.js": page_script(), "cases.json": json.dumps(node_cases())}
        with tempfile.TemporaryDirectory() as folder:
            for name, content in files.items():
                with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
                    handle.write(content)
            done = subprocess.run([NODE, "harness.js", "dashboard.js", "cases.json"], cwd=folder,
                                  capture_output=True, text=True, timeout=45)
        if done.returncode != 0 or not done.stdout.strip():
            raise AssertionError("the node run failed:\n" + done.stdout + done.stderr)
        result = json.loads(done.stdout)
        cls.problems, cls.calls, cls.snaps = result["problems"], result["calls"], result["snaps"]

    def shown(self, label, key):
        """What element.property held after that case, e.g. shown("good", "c-routed.textContent")."""
        return self.snaps[label].get(key, "")

    def assert_shown(self, label, expected):
        for key, value in expected.items():
            self.assertEqual(self.shown(label, key), value, "%s: %s" % (label, key))

    def test_never_throws_and_never_shows_undefined(self):
        self.assertEqual(self.problems, [])
        self.assertEqual(len(self.snaps), 1 + len(node_cases()["statuses"]) + len(node_cases()["clicks"]))

    def test_title_and_router_come_from_the_address(self):
        self.assert_shown("load", {"title.textContent": "Test title", "document.title": "Test title",
                                   "routerAddr.textContent": ROUTER})

    def test_router_down_is_said_in_the_header(self):
        for label in ("load", "router down"):
            self.assert_shown(label, {"connText.textContent": "cannot reach the router, retrying",
                                      "connText.className": "warn", "connLight.className": "light bad",
                                      "page.className": "stale", "c-routed.textContent": DASH})
        self.assertIn("HTTP 500", self.shown("http 500", "connText.textContent"))
        for label in ("not json", "empty body", "json null", "json list", "json number"):
            self.assertIn("bad answer from the router", self.shown(label, "connText.textContent"), label)
            self.assertEqual(self.shown(label, "page.className"), "stale", label)

    def test_good_status_is_shown(self):
        self.assert_shown("good", {
            "connText.textContent": "connected", "connText.className": "", "connLight.className": "light ok",
            "page.className": "", "policyLabel.textContent": "policy",
            "pol-adaptive.className": "on", "pol-static.className": "", "pol-round_robin.className": "",
            "c-routed.textContent": "100", "c-failovers.textContent": "1", "c-switches.textContent": "2",
            "c-heals.textContent": "0", "c-errors.textContent": "0",
            # node A
            "n0-gpu.textContent": "97", "n0-power.textContent": "60", "n0-temp.textContent": "70",
            "n0-mem.textContent": "101", "n0-badge.className": "badge hide",
            "n0-forecast-light.className": "light ok", "n0-forecast-state.textContent": "healthy",
            "n0-forecast-pref.className": "pref", "n0-forecast-drain.textContent": "drain",
            "n0-forecast-inflight.textContent": "1", "n0-forecast-queued.textContent": "0",
            "n0-forecast-latency.textContent": "210", "n0-forecast-score.textContent": "0.42",
            "n0-llm-light.className": "light ok", "n0-llm-pref.className": "pref",
            "n0-llm-queued.textContent": "2", "n0-llm-latency.textContent": "2500",
            "n0-llm-score.textContent": "15.3",
            # node B: a training job on the GPU, and the LLM server drained
            "n1-badge.className": "badge", "n1-forecast-pref.className": "pref gone",
            "n1-forecast-inflight.textContent": "3", "n1-forecast-queued.textContent": "40",
            "n1-forecast-latency.textContent": "380", "n1-forecast-score.textContent": "123",
            "n1-llm-light.className": "light off", "n1-llm-state.textContent": "draining",
            "n1-llm-pref.className": "pref gone", "n1-llm-drain.textContent": "put back",
        })
        nodes = self.shown("good", "nodes.innerHTML")
        self.assertLess(nodes.index("Node A"), nodes.index("Node B"))
        self.assertIn("--c:#4da3ff", nodes.split("Node A")[0], "node A is blue")
        self.assertIn("--c:#ff9f43", nodes.split("Node A")[1].split("Node B")[0], "node B is orange")
        self.assertIn("queued units", nodes)

    def test_feed_is_newest_first_and_coloured(self):
        feed = self.shown("good", "feed.innerHTML")
        self.assertEqual(feed.count('<div class="row'), len(STATUS["decisions"]))
        self.assertLess(feed.index("newest"), feed.index("policy set to adaptive"))
        self.assertLess(feed.index("policy set to adaptive"), feed.index(">preferred<"))
        for kind in ("failover", "switch", "heal", "unhealthy", "recovered"):
            self.assertIn('class="row hot %s"' % kind, feed)
        for kind in ("route", "drain", "policy", "fanout"):
            self.assertNotIn("hot " + kind, feed)
        self.assertNotIn("<b>one</b>", feed, "text from the router must be shown as text, not as markup")
        self.assertIn("&#60;b&#62;one&#60;/b&#62;", feed)
        self.assertIn(DASH, feed, "a decision with no backend shows a dash")
        self.assertIn("none yet", self.shown("states", "feed.innerHTML"))
        many = self.shown("many decisions", "feed.innerHTML")
        self.assertEqual(many.count('<div class="row'), 50, "the feed shows the newest 50")
        self.assertLess(many.index("number 119<"), many.index("number 70<"))
        self.assertNotIn("number 69<", many)

    def test_health_lights(self):
        self.assert_shown("states", {
            "pol-least_loaded.className": "on", "pol-adaptive.className": "",
            "n0-forecast-light.className": "light bad",
            "n0-forecast-state.textContent": "unhealthy, connection refused <script>",
            "n1-forecast-light.className": "light heal", "n1-forecast-state.textContent": "healing, unhealthy, timed out",
            "n0-llm-light.className": "light off", "n0-llm-state.textContent": "draining",
            "n1-llm-light.className": "light unknown", "n1-llm-state.textContent": "no health report",
            # the marker follows the status: forecast prefers A, llm prefers B
            "n0-forecast-pref.className": "pref", "n1-forecast-pref.className": "pref gone",
            "n0-llm-pref.className": "pref gone", "n1-llm-pref.className": "pref",
            "c-routed.textContent": "7", "c-errors.textContent": DASH,
        })

    def test_preferred_marker_follows_the_status(self):
        self.assert_shown("good", {"n0-forecast-pref.className": "pref", "n1-forecast-pref.className": "pref gone",
                                   "n0-llm-pref.className": "pref", "n1-llm-pref.className": "pref gone"})
        self.assert_shown("nobody preferred", {
            "pol-round_robin.className": "on", "n0-forecast-pref.className": "pref gone",
            "n1-forecast-pref.className": "pref gone", "n0-llm-pref.className": "pref gone",
            "n1-llm-pref.className": "pref gone"})

    def test_missing_numbers_show_a_dash(self):
        for label in ("empty object", "top level nulls", "top level wrong types", "null fields"):
            for key in ("c-routed", "c-errors", "n0-gpu", "n0-power", "n0-temp", "n0-mem", "n0-forecast-inflight",
                        "n0-forecast-queued", "n0-forecast-latency", "n0-forecast-score", "n1-llm-queued"):
                self.assertEqual(self.shown(label, key + ".textContent"), DASH, "%s: %s" % (label, key))
            self.assertEqual(self.shown(label, "connText.textContent"), "connected", label)
        self.assertEqual(self.shown("wrong types inside", "policyLabel.textContent"), 'policy: {"name":"x"}')

    def test_python_nan_is_read_as_missing(self):
        self.assert_shown("python nan", {
            "connText.textContent": "connected", "c-routed.textContent": DASH, "c-errors.textContent": DASH,
            "n0-forecast-inflight.textContent": "2", "n0-forecast-score.textContent": DASH,
            "n0-forecast-latency.textContent": DASH, "n0-forecast-light.className": "light ok"})

    def test_gpu_numbers_fall_back_to_the_routers_probe(self):
        # /metrics answers: its numbers win. /metrics is down: GPU use and free memory come from the probe.
        self.assert_shown("good first", {"n0-gpu.textContent": "97", "n0-mem.textContent": "101",
                                         "n0-power.textContent": "60", "n0-temp.textContent": "70"})
        self.assert_shown("metrics down", {"n0-gpu.textContent": "96", "n0-mem.textContent": "100",
                                           "n0-power.textContent": DASH, "n0-temp.textContent": DASH})

    def test_traffic_chart_draws_bars_from_routed_counts(self):
        # "rects" is the number of bar pieces in the last frame: one per poll, per backend with requests.
        self.assertEqual(self.snaps["good first"]["rects"], 0, "no requests, no bars")
        self.assertEqual(self.snaps["traffic"]["rects"], 5 * 4, "five polls with requests on four backends")
        for words in ("forecast pool", "LLM pool", "last 2 minutes", "10/s", "5/s", " ms"):
            self.assertIn(words, self.shown("traffic", "canvas"))
        self.assertEqual(self.snaps["polls close together"]["rects"], 5 * 4, "polls 0.1 s apart add no bars")
        self.assertEqual(self.snaps["clock went back"]["rects"], 2 * 4, "old bars dropped, new ones drawn")

    def test_absurd_numbers_show_a_dash(self):
        self.assert_shown("absurd numbers", {
            "c-routed.textContent": DASH, "c-failovers.textContent": DASH, "c-switches.textContent": DASH,
            "c-heals.textContent": "3", "n0-forecast-inflight.textContent": DASH,
            "n0-forecast-latency.textContent": DASH, "n0-forecast-score.textContent": DASH,
            "n0-forecast-queued.textContent": DASH, "n1-forecast-latency.textContent": "900000000000000"})

    def test_odd_names_are_shown_as_text(self):
        nodes = self.shown("odd names", "nodes.innerHTML")
        self.assertNotIn("<b>B</b>", nodes)
        self.assertIn("&#60;b&#62;B&#60;/b&#62;", nodes)
        self.assertEqual(nodes.count('class="panel node"'), 7, "one column per name")
        feed = self.shown("odd names", "feed.innerHTML")
        for markup in ("<i>", "<b>", "<u>"):
            self.assertNotIn(markup, feed)

    def test_buttons_send_what_the_contract_says(self):
        posts = [(call["url"], call["body"]) for call in self.calls if call["method"] == "POST"]
        for name in POLICIES:
            self.assertIn((ROUTER + "/router/policy", {"policy": name}), posts)
        self.assertIn((ROUTER + "/router/drain", {"backend": "A", "drain": True}), posts)
        self.assertIn((ROUTER + "/router/drain", {"backend": "B", "drain": False}), posts)
        self.assertIn((ROUTER + "/v1/ensemble",
                       {"init": "latest", "lead_hours": 48, "members": 20, "point": OHARE}), posts)
        self.assertIn((ROUTER + "/v1/forecast", {"init": "2019-01-27T12", "lead_hours": 72, "members": 1,
                                                 "point": OHARE, "with_truth": True}), posts)
        for call in self.calls:
            if call["method"] == "POST":
                self.assertEqual(call["type"], "application/json", call["url"])
        self.assertIn("POST /router/policy failed: HTTP 400: unknown policy",
                      self.shown("policy refused", "connText.textContent"))
        self.assertIn("POST /router/drain failed: no answer", self.shown("drain, no answer", "connText.textContent"))

    def test_polls_the_router_and_the_forecast_servers(self):
        gets = [call["url"] for call in self.calls if call["method"] == "GET"]
        self.assertEqual(sorted(set(gets)), ["http://127.0.0.1:8101/metrics", "http://127.0.0.1:8102/metrics",
                                             ROUTER + "/router/status"])
        self.assertGreater(gets.count(ROUTER + "/router/status"), gets.count("http://127.0.0.1:8101/metrics"))

    def test_forecast_card(self):
        info = self.shown("ensemble", "fcInfo.innerHTML")
        for words in ("2026-11-13T00", "total <b>0 ms</b>", "node A</b> 12 members, 1300 ms",
                      "node B</b> 8 members, 2100 ms (run again after a failure)"):
            self.assertIn(words, info)
        canvas = self.shown("ensemble", "canvas")
        for words in ("mean of 20 members", "band: lowest to highest member", "11-13 06Z", "11-15 00Z", " C"):
            self.assertIn(words, canvas)

        info = self.shown("replay", "fcInfo.innerHTML")
        self.assertIn("2019-01-27T12", info)
        self.assertIn("answered by", info)
        canvas = self.shown("replay", "canvas")
        for words in ("forecast", "what happened"):
            self.assertIn(words, canvas)
        self.assertNotIn("band:", canvas)

        for button in ("ensemble", "replay"):
            self.assertIn("No forecast: HTTP 503: no healthy backend", self.shown(button + ": refused", "fcInfo.innerHTML"))
            self.assertIn("No forecast: no answer", self.shown(button + ": down", "fcInfo.innerHTML"))
            # the node run plays a page opened from a file, so "no answer" comes with the hint
            self.assertIn("open this page from the router", self.shown(button + ": down", "fcInfo.innerHTML"))
            self.assertNotIn("open this page", self.shown(button + ": refused", "fcInfo.innerHTML"))
            self.assertIn("The answer held no temperatures.", self.shown(button + ": empty", "canvas"))
            for label in (button, button + ": down", button + ": odd series"):
                self.assertEqual(self.shown(label, "fc-ensemble.disabled"), "", "the buttons work again: " + label)


if __name__ == "__main__":
    unittest.main()
