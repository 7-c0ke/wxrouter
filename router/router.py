#!/usr/bin/env python3
"""The router: one address in front of both boxes (contract section 3).

It sends each request to a healthy backend, retries once on the other box if
that backend breaks, splits ensembles across the forecast servers, passes a
new analysis (POST /v1/state) to every forecast server, and runs the heal
action of a backend that stays down. Standard library only.

Run it:
    python router/router.py --config router/config.example.json
    python router/router.py --port 8100 --llm-probe llamacpp \\
        --forecast A=http://nodeA:8101,B=http://nodeB:8101 \\
        --llm A=http://nodeA:8000,B=http://nodeB:8000

Then open http://127.0.0.1:8100/router/status (or / for the dashboard).

Written at the hackathon (Nov 13-14, 2026).
"""

import argparse
import collections
import copy
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POLICIES = ("static", "round_robin", "least_loaded", "adaptive")
PROBES = ("forecast", "llamacpp", "vllm", "http")

# Which pool serves which path.
POOL_PATHS = {
    "forecast": ("/v1/forecast", "/v1/ensemble", "/v1/state", "/v1/info"),
    "llm": ("/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings"),
}

DEFAULTS = {
    "port": 8100, "policy": "adaptive", "probe_s": 1.0, "fail_n": 2, "recover_s": 3.0,
    "connect_timeout_s": 2.0,
    "adaptive": {"hysteresis": 0.25, "dwell_s": 3.0, "cooldown_s": 10.0, "spill_inflight": 0},
    "score": {"busy_other_penalty": 1.5, "low_mem_gb": 8, "low_mem_penalty": 2.0,
              "default_latency_ms": 200},
    "heal_after_s": 10, "heal_backoff_s": [5, 15, 60], "heal_max_attempts": 5,
    "decision_log": "router-decisions.jsonl",
}
POOL_DEFAULTS = {
    "forecast": {"first_byte_timeout_s": 120, "affinity": False, "affinity_break": 3.0,
                 "probe": "forecast"},
    "llm": {"first_byte_timeout_s": 600, "affinity": True, "affinity_break": 3.0,
            "probe": "llamacpp"},
}

EWMA_ALPHA = 0.2          # weight of the newest latency sample
KEEP_DECISIONS = 50       # how many decisions /router/status shows
ROUTE_SAMPLES_PER_S = 5   # plain "route" decisions kept per second in that list
HEAL_HTTP_TIMEOUT_S = 10
HEAL_CMD_TIMEOUT_S = 60
RETRY_STATUSES = (502, 503, 504)

# Headers that belong to one connection and are never passed on.
HOP_BY_HOP = ("connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailer", "trailers", "transfer-encoding", "upgrade", "proxy-connection")

# The Prometheus names for (requests running, requests waiting), per engine.
PROM_NAMES = {
    "llamacpp": ("llamacpp:requests_processing", "llamacpp:requests_deferred"),
    "vllm": ("vllm:num_requests_running", "vllm:num_requests_waiting"),
}


# ---------------------------------------------------------------- config

def make_config(raw=None):
    """Fill in the defaults and check the config. Raises ValueError with a plain message."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("the config must be a JSON object")
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in ("adaptive", "score"):
            if not isinstance(value, dict):
                raise ValueError("config '%s' must be an object" % key)
            cfg[key].update(value)
        elif key != "pools":
            cfg[key] = value
    if cfg["policy"] not in POLICIES:
        raise ValueError("unknown policy '%s'; use one of: %s" % (cfg["policy"], ", ".join(POLICIES)))
    steps = cfg["heal_backoff_s"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("heal_backoff_s must be a list of seconds, for example [5, 15, 60]")
    raw_pools = raw.get("pools") or {}
    if not isinstance(raw_pools, dict):
        raise ValueError("config 'pools' must be an object with 'forecast' and/or 'llm'")
    for name, value in raw_pools.items():
        if name not in POOL_PATHS:
            raise ValueError("unknown pool '%s'; the pools are: %s" % (name, ", ".join(POOL_PATHS)))
        if value is not None and not isinstance(value, dict):
            raise ValueError("pool '%s' must be an object with a 'backends' list" % name)
    cfg["pools"] = collections.OrderedDict()
    for name in POOL_PATHS:
        pool = dict(POOL_DEFAULTS[name])
        pool.update(raw_pools.get(name) or {})
        default_probe = pool.pop("probe")
        backends, seen = [], set()
        for spec in pool.get("backends") or []:
            where = "pool '%s'" % name
            if not isinstance(spec, dict) or not spec.get("name") or not spec.get("url"):
                raise ValueError("%s: every backend needs a name and a url" % where)
            spec = dict(spec)
            if spec["name"] in seen:
                raise ValueError("%s: backend name '%s' is used twice" % (where, spec["name"]))
            seen.add(spec["name"])
            try:
                parts = urlsplit(spec["url"])
                good_url = parts.scheme == "http" and bool(parts.hostname) and (parts.port or 80) > 0
            except ValueError:                 # a port that is not a number
                good_url = False
            if not good_url:
                raise ValueError("%s: backend %s: the url must look like http://host:port, got '%s'"
                                 % (where, spec["name"], spec["url"]))
            spec.setdefault("probe", default_probe)
            if spec["probe"] not in PROBES:
                raise ValueError("%s: backend %s: unknown probe '%s'; use one of: %s"
                                 % (where, spec["name"], spec["probe"], ", ".join(PROBES)))
            if spec.get("heal") is not None and not _good_heal(spec["heal"]):
                raise ValueError("%s: backend %s: heal must be {\"http\": {\"url\": \"http://...\"}} "
                                 "or {\"cmd\": [\"program\", \"arg\"]}" % (where, spec["name"]))
            backends.append(spec)
        pool["backends"] = backends
        cfg["pools"][name] = pool
    return cfg


def _good_heal(heal):
    """True for a heal action the router can run: one HTTP call or one command line."""
    if not isinstance(heal, dict):
        return False
    call, cmd = heal.get("http"), heal.get("cmd")
    if isinstance(call, dict) and call.get("url"):
        return urlsplit(str(call["url"])).scheme == "http"
    return isinstance(cmd, list) and bool(cmd) and all(isinstance(x, str) for x in cmd)


def parse_backend_list(text, probe):
    """Turn "A=http://h:8101,B=http://h2:8101" into a list of backend specs."""
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("expected NAME=URL, got '%s'" % item)
        name, url = item.split("=", 1)
        out.append({"name": name.strip(), "url": url.strip(), "probe": probe})
    return out


# ---------------------------------------------------------------- reading probe answers

def num(value):
    """The value as a number, or None if it is missing or not a real number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def parse_forecast_metrics(obj):
    """Pick the numbers the router uses out of a forecast server's /metrics JSON.

    Anything missing or null comes back as None.
    """
    if not isinstance(obj, dict):
        obj = {}
    gpu = obj.get("gpu") if isinstance(obj.get("gpu"), dict) else {}
    host = obj.get("host") if isinstance(obj.get("host"), dict) else {}
    busy = obj.get("busy_other")
    return {
        "inflight": num(obj.get("inflight")),
        "queued": num(obj.get("queued")),
        "queued_units": num(obj.get("queued_units")),
        "gpu_util_pct": num(gpu.get("util_pct")),
        "busy_other": busy if isinstance(busy, bool) else None,
        "slowdown": num(obj.get("slowdown")),
        "mem_available_gb": num(host.get("mem_available_gb")),
        "ms_per_unit": num(obj.get("ms_per_unit")),
    }


def parse_prometheus(text):
    """Read Prometheus text into {metric name: value}.

    Labels in braces are dropped, and lines with the same name are added up.
    Lines that make no sense are skipped.
    """
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line[:line.index("{")].strip()
            rest = line[line.rfind("}") + 1:].split()
        else:
            parts = line.split()
            name, rest = parts[0], parts[1:]
        value = num(rest[0]) if rest else None
        if not name or value is None:
            continue
        out[name] = out.get(name, 0) + value
    return out


def parse_llm_metrics(text, kind):
    """Running and waiting request counts from an LLM engine's /metrics text.

    The engine named by `kind` is looked for first, then the other one, so a
    wrong --llm-probe still finds the numbers. Missing metrics give None.
    """
    metrics = parse_prometheus(text)
    running = waiting = None
    for engine in [kind] + [k for k in PROM_NAMES if k != kind]:
        names = PROM_NAMES.get(engine)
        if not names:
            continue
        if running is None:
            running = metrics.get(names[0])
        if waiting is None:
            waiting = metrics.get(names[1])
    return {"queued": _whole(waiting), "running": _whole(running)}


def _whole(x):
    """2.0 becomes 2; anything else is left alone."""
    if isinstance(x, float) and x == int(x):
        return int(x)
    return x


def forecast_units(method, path, body):
    """How much GPU work a request to the forecast pool asks for, in units.

    A unit is one member advanced one 6-hour step, so POST /v1/forecast costs
    members x lead_hours / 6. Anything else in the pool (/v1/info, /v1/state)
    costs nothing. A body that cannot be read counts as 1: the server will
    refuse it anyway. A very large body (a state sent along as npy_b64) is
    not parsed, to keep the router quick; it counts as 1 as well, and the
    next probe sees the real number.
    """
    if method != "POST" or path != "/v1/forecast":
        return 0
    if len(body) > 65536:
        return 1
    try:
        obj = json.loads(body.decode("utf-8"))
    except ValueError:
        return 1
    if not isinstance(obj, dict):
        return 1
    members, lead = obj.get("members", 1), obj.get("lead_hours")
    if members is None:
        members = 1
    for value in (members, lead):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 1000:
            return 1
    return max(1, int(members) * (int(lead) // 6))


def split_members(total, scores):
    """Share `total` ensemble members out in proportion to 1/score.

    Returns whole numbers, one per score, that add up to total. A backend
    whose share rounds to nothing gets 0 and is left out by the caller.
    """
    weights = [1.0 / max(s, 1e-9) for s in scores]
    whole = sum(weights)
    ideal = [total * w / whole for w in weights]
    shares = [int(x) for x in ideal]
    # The members left over go to the largest remainders (ties: the better score).
    order = sorted(range(len(scores)), key=lambda i: (shares[i] - ideal[i], scores[i], i))
    for i in order[:total - sum(shares)]:
        shares[i] += 1
    return shares


# ---------------------------------------------------------------- the choosing logic

Choice = collections.namedtuple("Choice", "backend url reason affinity kind policy")


class Backend(object):
    """What the router knows about one backend."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.url = spec["url"].rstrip("/")
        self.probe = spec["probe"]
        self.heal = spec.get("heal")
        self.healthy = True          # trusted until probes or requests say otherwise
        self.draining = False
        self.healing = False         # heal actions are being tried and it is not back yet
        self.peers = [self]          # every backend of the same pool (set by the pool)
        self.inflight = 0            # the router's own count of requests it has open here
        self.latency_ewma_ms = None
        self.routed = 0
        self.failures = 0            # all failed probes and requests so far
        self.fail_streak = 0         # failures in a row
        self.last_error = None
        self.probe_data = {}         # the last numbers a probe got
        self.open_units = 0          # forecast work (units) in the requests the router has open here
        self.other_units = 0         # forecast work the last probe saw on top of ours (other clients)
        self.unhealthy_since = None
        self.heal_attempts = 0
        self.heal_running = False
        self.next_heal_at = None


class Pool(object):
    """One pool (forecast or llm): its backends and the adaptive policy's memory."""

    def __init__(self, name, spec):
        self.name = name
        self.backends = [Backend(b) for b in spec.get("backends") or []]
        for b in self.backends:
            b.peers = self.backends
        self.affinity = bool(spec.get("affinity"))
        self.affinity_break = float(spec.get("affinity_break", 3.0))
        self.first_byte_timeout_s = float(spec.get("first_byte_timeout_s", 120))
        self.preferred = self.backends[0].name if self.backends else None
        self.challenger = None        # the backend that is beating the preferred one
        self.challenger_since = None  # ... and since when, without a break
        self.last_switch = None
        self.rr_last = -1             # index of round robin's last pick
        self.last_pick = None         # static and least_loaded: the backend they chose last time

    def get(self, name):
        for b in self.backends:
            if b.name == name:
                return b
        return None


class Core(object):
    """Backend state, the score, the four policies, affinity, draining and health.

    There are no sockets in here. Every method takes the one lock, reads the
    clock through self.now (tests pass their own) and returns at once. The
    HTTP code below calls route() before a request and finish() after it.
    """

    def __init__(self, config, now=time.monotonic, wall=time.time, sink=None):
        self.cfg = config
        self.now = now
        self.wall = wall
        self.sink = sink              # called with every decision (the log file)
        self.lock = threading.RLock()
        self.policy = config["policy"]
        self.pools = collections.OrderedDict(
            (name, Pool(name, spec)) for name, spec in config["pools"].items())
        self.counters = collections.OrderedDict(
            (k, 0) for k in ("routed", "failovers", "switches", "heals", "errors"))
        self.decisions = []           # what /router/status shows: at most KEEP_DECISIONS, newest last
        self._route_times = collections.deque(maxlen=ROUTE_SAMPLES_PER_S)
        self.started = now()

    # ---- score

    def score(self, backend, idle=False, latency_ms=None):
        """Estimated wait in seconds for a new request on this backend. Lower is better.

        Two formulas, and which one is used depends on what the last probe said:

        1. Forecast backends whose probe gave ms_per_unit:
               (open_units + other_units + 1) x ms_per_unit
           Work is counted in units (one member, one 6-hour step), because one
           forecast can be 1 unit or 800. open_units is the work in the requests
           the router has open on the backend right now. The router reads each
           request's units from its body, so this number is always up to date
           and does not wait for the next probe. other_units is what the last
           probe saw on top of ours: work that other clients sent straight to
           the server. ms_per_unit is the server's speed as it is now. It
           already holds any slowdown, so the probe's "slowdown" is not
           multiplied in a second time.

        2. Everything else (the llm pool, or no probe numbers yet):
               (inflight + queued + 1) x latency x slowdown
           inflight is the router's own count, queued and slowdown come from
           the probe (0 and 1.0 when unknown).

        latency is the router's latency EWMA for the backend. A backend that
        has not answered a request yet is taken to be as fast as the measured
        ones in its pool (their mean); if none is measured, the config's
        default_latency_ms is used. Otherwise the box that has done nothing
        yet would look far better, or far worse, than the one doing the work.

        Then the penalties: times busy_other_penalty when another job is on
        the GPU, times low_mem_penalty when free memory is under low_mem_gb.

        idle=True gives the score the backend would have with nothing of ours
        on it: (other_units + 1) x ms_per_unit, or latency x slowdown with the
        pool's mean latency. The adaptive policy compares these, see
        _adaptive_update. latency_ms, if given, replaces the backend's own
        latency in formula 2; the affinity check uses that.

        Why the pool's mean latency in those two places: a backend's own
        latency says more about the requests it happened to get (long prompts,
        large forecasts, time spent in its queue) than about the box. It
        would move the preference and break affinity for no real reason.
        """
        cfg = self.cfg["score"]
        data = backend.probe_data or {}
        ms_per_unit = num(data.get("ms_per_unit"))
        if backend.probe == "forecast" and ms_per_unit and ms_per_unit > 0:
            ours = 0 if idle else backend.open_units
            score = (ours + backend.other_units + 1) * ms_per_unit / 1000.0
        else:
            if latency_ms is None:
                latency_ms = self._pool_latency_ms(backend) if idle else self._latency_ms(backend)
            slowdown = num(data.get("slowdown"))
            if not slowdown or slowdown <= 0:
                slowdown = 1.0
            queued = num(data.get("queued")) or 0
            ahead = 0 if idle else backend.inflight + max(queued, 0)
            score = (ahead + 1) * latency_ms / 1000.0 * slowdown
        if data.get("busy_other") is True:
            score *= cfg["busy_other_penalty"]
        mem = num(data.get("mem_available_gb"))
        if mem is not None and mem < cfg["low_mem_gb"]:
            score *= cfg["low_mem_penalty"]
        return score

    def _pool_latency_ms(self, backend):
        """The mean latency EWMA of the measured backends in this backend's pool, else the default."""
        known = [p.latency_ewma_ms for p in backend.peers if p.latency_ewma_ms is not None]
        return sum(known) / len(known) if known else self.cfg["score"]["default_latency_ms"]

    def _latency_ms(self, backend):
        """The backend's own latency EWMA; if it has none yet, its pool's mean."""
        if backend.latency_ewma_ms is not None:
            return backend.latency_ewma_ms
        return self._pool_latency_ms(backend)

    def _usable(self, pool, exclude=()):
        """Backends that may get a new request, in config order."""
        return [b for b in pool.backends
                if b.healthy and not b.draining and b.name not in exclude]

    def _best(self, backends):
        """Lowest score; the earlier one in the list wins a tie."""
        return min(backends, key=self.score)

    # ---- decisions

    def _decide(self, kind, pool, backend, reason, retries=0):
        """Record one decision: always to the log, and to the status list.

        Plain "route" decisions are sampled into the status list (at most 5 a
        second). When the list is full, the oldest "route" row goes first, as
        long as "route" rows fill more than half of it. So a busy router does
        not push the rare switches and failovers out of the last 50.
        """
        decision = {"ts": round(self.wall(), 3), "pool": pool, "backend": backend,
                    "policy": self.policy, "kind": kind, "reason": reason, "retries": retries}
        keep = True
        if kind == "route":
            t = self.now()
            recent = self._route_times
            if len(recent) == recent.maxlen and t - recent[0] < 1.0:
                keep = False
            else:
                recent.append(t)
        if keep:
            shown = self.decisions
            shown.append(decision)
            if len(shown) > KEEP_DECISIONS:
                routes = [i for i, d in enumerate(shown) if d["kind"] == "route"]
                del shown[routes[0] if len(routes) > KEEP_DECISIONS // 2 else 0]
        if self.sink is not None:
            try:
                self.sink(decision)
            except Exception:  # a broken log must never stop routing
                pass
        return decision

    # ---- the adaptive policy's preferred backend

    def _switch(self, pool, new, now, reason):
        pool.preferred = new.name
        pool.last_switch = now
        pool.challenger = None
        pool.challenger_since = None
        self.counters["switches"] += 1
        self._decide("switch", pool.name, new.name, reason)

    def _adaptive_update(self, pool, now):
        """Move the pool's preferred backend if the rules say so.

        Called before every adaptive choice and after every probe, so "for
        dwell_s without a break" is checked at each of those moments.

        The two backends are compared on their idle scores: what each would
        make a request wait with nothing of ours on it. The preferred backend
        carries our traffic, so on the plain score it always looks worse than
        the one that stands idle, and the preference would flip back and forth
        at every cooldown. On idle scores it moves only when a probe says the
        other box is really the better one: faster per unit, no training job
        beside it, more free memory, less work from other clients. (An LLM
        engine's probe says none of that, so in the llm pool the preference
        moves only when its backend is unhealthy or draining.)
        """
        usable = self._usable(pool)
        if not usable:
            pool.challenger = None
            return
        pref = pool.get(pool.preferred)
        if pref is None:
            pool.preferred = usable[0].name
            return
        if pref not in usable:
            # Unhealthy or draining: leave it at once. Dwell and cooldown do not apply.
            best = self._best(usable)
            why = "is draining" if pref.healthy else "is unhealthy"
            self._switch(pool, best, now, "%s %s; %s takes over (score %.3f; %s had %.3f)" % (
                pref.name, why, best.name, self.score(best), pref.name, self.score(pref)))
            return
        others = [b for b in usable if b is not pref]
        if not others:
            pool.challenger = None
            return
        rival = min(others, key=lambda b: self.score(b, idle=True))
        knobs = self.cfg["adaptive"]
        s_pref, s_rival = self.score(pref, idle=True), self.score(rival, idle=True)
        if not s_rival < s_pref * (1.0 - knobs["hysteresis"]):
            pool.challenger = None      # not better by the margin: the dwell clock starts over
            return
        if pool.challenger != rival.name:
            pool.challenger = rival.name
            pool.challenger_since = now
        held = now - pool.challenger_since
        cooled = pool.last_switch is None or now - pool.last_switch >= knobs["cooldown_s"]
        if held >= knobs["dwell_s"] and cooled:
            self._switch(pool, rival, now, "%s (idle score %.3f) beat %s (idle score %.3f) by more "
                         "than %d%% for %.1f s" % (rival.name, s_rival, pref.name, s_pref,
                                                   round(knobs["hysteresis"] * 100), held))

    def tick(self):
        """Re-check the preferred backends. Probes call this; tests may too."""
        with self.lock:
            if self.policy == "adaptive":
                now = self.now()
                for pool in self.pools.values():
                    self._adaptive_update(pool, now)

    # ---- choosing

    def _moved(self, pool, chosen, retry):
        """static and least_loaded: count a switch when the choice is not the one before.

        That is how often a policy changed its mind, so the counter can be
        compared across policies. A retry is not counted: it is a failover.
        """
        if not retry:
            if pool.last_pick not in (None, chosen.name):
                self.counters["switches"] += 1
            pool.last_pick = chosen.name
        return chosen

    def _pick(self, pool, usable, retry=False):
        """The policy's choice among the usable backends: (backend, reason)."""
        if self.policy == "static":
            return self._moved(pool, usable[0], retry), "first healthy"
        if self.policy == "round_robin":
            count = len(pool.backends)
            for step in range(1, count + 1):
                index = (pool.rr_last + step) % count
                if pool.backends[index] in usable:
                    pool.rr_last = index
                    return pool.backends[index], "rotation"
        if self.policy == "least_loaded":
            return self._moved(pool, self._best(usable), retry), "lowest score"
        # adaptive
        pref = pool.get(pool.preferred)
        if pref is None or pref not in usable:
            return self._best(usable), "best of the rest"
        if pref.inflight > self.cfg["adaptive"]["spill_inflight"]:
            # The preferred backend is full. The overflow goes to the next-best
            # one, but only while that one really scores lower: a burst is then
            # shared out instead of being dumped on the second box.
            others = [b for b in usable if b is not pref]
            if others:
                second = self._best(others)
                if self.score(second) < self.score(pref):
                    return second, "spill"
        return pref, "preferred"

    def _home(self, pool, key):
        """Rendezvous hashing: the usable backend with the highest hash of key + name."""
        best, best_hash = None, None
        for b in self._usable(pool):
            h = hashlib.sha1((key + "\n" + b.name).encode("utf-8")).digest()
            if best_hash is None or h > best_hash:
                best, best_hash = b, h
        return best

    def route(self, pool_name, key=None, exclude=(), retries=0, after=None, units=0):
        """Choose a backend for one request and count it as in flight there.

        key: the session key for affinity (or None). exclude: names not to
        use (the backend that just failed). retries: 0 for a first try, 1 for
        the retry; `after` then says what went wrong, for the decision's reason.
        units: the forecast work in the request (see forecast_units), 0 if none.
        Returns a Choice, or None when no backend can take the request.
        Call finish() with the same units when the request is over.
        """
        with self.lock:
            pool = self.pools[pool_name]
            if self.policy == "adaptive":
                self._adaptive_update(pool, self.now())
            usable = self._usable(pool, exclude)
            if not usable:
                return None
            chosen, reason, affinity, home = None, None, None, None
            if pool.affinity and key:
                # Stay with the session's home backend, unless it just failed
                # (excluded) or its score is over affinity_break times the best.
                # The scores here use one latency for the whole pool, so only
                # the load counts. With each backend's own latency, a session
                # with long prompts would make its own home look slow, leave
                # it, and lose the prompt cache it came for.
                home = self._home(pool, key)
                shared = self._pool_latency_ms(usable[0])
                limit = pool.affinity_break * min(self.score(b, latency_ms=shared) for b in usable)
                if (home is not None and home in usable
                        and self.score(home, latency_ms=shared) <= limit):
                    chosen, reason = home, "affinity"
            if chosen is None:
                chosen, reason = self._pick(pool, usable, retry=bool(retries))
            if pool.affinity:
                # hit: the request goes to its session's home. none: it had no session key.
                affinity = "none" if not key else ("hit" if chosen is home else "miss")
            kind = "route"
            if retries:
                kind = "failover"
                self.counters["failovers"] += 1
                if after:
                    reason = "%s; %s" % (after, reason)
            else:
                self.counters["routed"] += 1
            chosen.inflight += 1
            chosen.routed += 1
            chosen.open_units += units
            self._decide(kind, pool.name, chosen.name, reason, retries)
            return Choice(chosen.name, chosen.url, reason, affinity, kind, self.policy)

    def plan_fanout(self, pool_name, members, steps=1):
        """Split an ensemble across the usable backends in proportion to 1/score.

        Returns [(name, url, members), ...] in config order, each counted as
        one request in flight with members x steps units of work, or None
        when no backend is usable.
        """
        with self.lock:
            pool = self.pools[pool_name]
            usable = self._usable(pool)
            if not usable:
                return None
            shares = split_members(members, [self.score(b) for b in usable])
            plan = []
            for b, share in zip(usable, shares):
                if share > 0:
                    b.inflight += 1
                    b.routed += 1
                    b.open_units += share * steps
                    plan.append((b.name, b.url, share))
            self.counters["routed"] += 1
            self._decide("fanout", pool.name, "+".join(p[0] for p in plan), "%d members: %s" % (
                members, ", ".join("%s %d" % (p[0], p[2]) for p in plan)))
            return plan

    def plan_all(self, pool_name, what):
        """Every usable backend of a pool, for a request that must reach them all.

        Returns ([(name, url), ...], [the names left out]). Each one returned
        is counted as a request in flight: call finish() for it. The list is
        None when no backend is usable. `what` is the decision's reason.
        """
        with self.lock:
            pool = self.pools[pool_name]
            usable = self._usable(pool)
            left_out = [b.name for b in pool.backends if b not in usable]
            if not usable:
                return None, left_out
            for b in usable:
                b.inflight += 1
                b.routed += 1
            self.counters["routed"] += 1
            self._decide("fanout", pool.name, "+".join(b.name for b in usable), what)
            return [(b.name, b.url) for b in usable], left_out

    def finish(self, pool_name, name, ok, latency_ms=None, error=None, units=0):
        """A request to this backend is over.

        ok=True: it answered (the latency feeds the EWMA). ok=False: the
        backend failed. ok=None: neither (the client hung up, or a 500 was
        passed on). units: what route() or plan_fanout() counted for it.
        """
        with self.lock:
            pool = self.pools[pool_name]
            b = pool.get(name)
            if b is None:
                return
            b.inflight = max(0, b.inflight - 1)
            b.open_units = max(0, b.open_units - units)
            if ok is True:
                if b.healthy:
                    b.fail_streak = 0
                if latency_ms is not None:
                    if b.latency_ewma_ms is None:
                        b.latency_ewma_ms = float(latency_ms)
                    else:
                        b.latency_ewma_ms += EWMA_ALPHA * (latency_ms - b.latency_ewma_ms)
            elif ok is False:
                self._failed(pool, b, error or "failed", self.now())

    def count_error(self):
        with self.lock:
            self.counters["errors"] += 1

    # ---- health

    def _failed(self, pool, b, error, now):
        b.failures += 1
        b.fail_streak += 1
        b.last_error = error
        if b.healthy and b.fail_streak >= self.cfg["fail_n"]:
            b.healthy = False
            b.unhealthy_since = now
            b.heal_attempts = 0
            b.healing = False
            b.next_heal_at = now + self.cfg["heal_after_s"]
            self._decide("unhealthy", pool.name, b.name,
                         "%d failures in a row; last: %s" % (b.fail_streak, error))
            if self.policy == "adaptive":
                self._adaptive_update(pool, now)

    def open_units(self, pool_name, name):
        """The forecast work the router has open on a backend right now, in units."""
        with self.lock:
            b = self.pools[pool_name].get(name)
            return b.open_units if b is not None else 0

    def probe_result(self, pool_name, name, ok, data=None, error=None, units_before=0):
        """Take in one probe's answer. A good probe is the only way back to healthy.

        units_before: open_units() as it was just before the probe was sent.
        The server's queued_units minus our own work is other clients' work.
        Our own work is taken as the larger of before and now, so a request
        that ended while the probe was on its way is not mistaken for theirs.
        """
        with self.lock:
            now = self.now()
            pool = self.pools[pool_name]
            b = pool.get(name)
            if b is None:
                return
            if ok:
                b.probe_data = data or {}
                seen = num(b.probe_data.get("queued_units"))
                b.other_units = 0 if seen is None else max(0, seen - max(units_before, b.open_units))
                if b.healthy:
                    b.fail_streak = 0
                elif now - b.unhealthy_since >= self.cfg["recover_s"]:
                    down_s = now - b.unhealthy_since
                    b.healthy = True
                    b.fail_streak = 0
                    b.last_error = None
                    b.unhealthy_since = None
                    b.healing = False
                    b.heal_attempts = 0      # a recovery resets the heal attempts
                    b.next_heal_at = None
                    self._decide("recovered", pool.name, b.name,
                                 "probe ok after %.1f s down" % down_s)
            else:
                self._failed(pool, b, error or "probe failed", now)
            if self.policy == "adaptive":
                self._adaptive_update(pool, now)

    # ---- self-healing

    def due_heals(self):
        """The heal actions to run now: [(pool, backend, action, attempt number), ...].

        A backend is due when it has a heal action, has been unhealthy for
        heal_after_s (later attempts: the back-off since the last one ended),
        is not draining, and has attempts left. Each one returned is marked
        as healing; call heal_done() when its action has run.
        """
        with self.lock:
            now = self.now()
            due = []
            for pool in self.pools.values():
                for b in pool.backends:
                    if b.healthy or b.draining or not b.heal or b.heal_running:
                        continue
                    if b.heal_attempts >= self.cfg["heal_max_attempts"]:
                        continue
                    if b.next_heal_at is None or now < b.next_heal_at:
                        continue
                    b.heal_attempts += 1
                    b.heal_running = True
                    b.healing = True
                    due.append((pool.name, b.name, b.heal, b.heal_attempts))
            return due

    def heal_done(self, pool_name, name, attempt, ok, text):
        """A heal action has run: record it and start the back-off."""
        with self.lock:
            b = self.pools[pool_name].get(name)
            if b is None:
                return
            steps = self.cfg["heal_backoff_s"]
            b.heal_running = False
            b.next_heal_at = self.now() + steps[min(attempt - 1, len(steps) - 1)]
            if attempt >= self.cfg["heal_max_attempts"]:
                b.healing = False     # that was the last try; now only a good probe helps
            self.counters["heals"] += 1
            self._decide("heal", pool_name, name, "attempt %d of %d %s: %s" % (
                attempt, self.cfg["heal_max_attempts"], "ran" if ok else "failed", text))

    # ---- admin

    def set_policy(self, policy):
        """Switch the policy at run time. Raises ValueError for an unknown name."""
        if policy not in POLICIES:
            raise ValueError("unknown policy '%s'; use one of: %s" % (policy, ", ".join(POLICIES)))
        with self.lock:
            old, self.policy = self.policy, policy
            for pool in self.pools.values():
                pool.challenger = None
                pool.challenger_since = None
                pool.last_pick = None
            self._decide("policy", None, None, "%s -> %s" % (old, policy))
        self.tick()
        return policy

    def set_drain(self, name, drain):
        """Take a node out (or put it back) in every pool that has it. Returns those pools."""
        touched = []
        with self.lock:
            for pool in self.pools.values():
                b = pool.get(name)
                if b is None:
                    continue
                b.draining = bool(drain)
                touched.append(pool.name)
                self._decide("drain", pool.name, name,
                             "draining: no new requests" if drain else "back in service")
        self.tick()
        return touched

    def _preferred_now(self, pool):
        """What /router/status shows as preferred, for the policy in use."""
        if self.policy == "adaptive":
            return pool.preferred
        usable = self._usable(pool)
        if not usable or self.policy == "round_robin":
            return None                 # round robin prefers nobody
        if self.policy == "static":
            return usable[0].name
        return self._best(usable).name

    def status(self):
        """The body of GET /router/status."""
        with self.lock:
            pools = collections.OrderedDict()
            for pool in self.pools.values():
                rows = []
                for b in pool.backends:
                    ewma = b.latency_ewma_ms
                    rows.append(collections.OrderedDict([
                        ("name", b.name), ("url", b.url), ("probe", b.probe),
                        ("healthy", b.healthy), ("draining", b.draining), ("healing", b.healing),
                        ("inflight", b.inflight),
                        ("latency_ewma_ms", None if ewma is None else round(ewma, 1)),
                        ("score", round(self.score(b), 4)),
                        ("routed", b.routed), ("failures", b.failures),
                        ("last_error", b.last_error), ("probe_data", dict(b.probe_data))]))
                pools[pool.name] = collections.OrderedDict(
                    [("preferred", self._preferred_now(pool)), ("backends", rows)])
            return collections.OrderedDict([
                ("policy", self.policy),
                ("uptime_s", round(self.now() - self.started, 1)),
                ("counters", dict(self.counters)),
                ("pools", pools),
                ("decisions", list(self.decisions))])

    def health(self):
        """The body of GET /router/health: how many backends each pool can use."""
        with self.lock:
            usable = collections.OrderedDict(
                (pool.name, len(self._usable(pool))) for pool in self.pools.values())
            empty = [name for name, pool in self.pools.items()
                     if pool.backends and not usable[name]]
            return collections.OrderedDict([
                ("status", "degraded" if empty else "ok"), ("policy", self.policy),
                ("uptime_s", round(self.now() - self.started, 1)), ("usable", usable)])


# ---------------------------------------------------------------- small network helpers

NET_ERRORS = (OSError, http.client.HTTPException)


def warn(text):
    try:
        sys.stderr.write("router: %s\n" % text)
        sys.stderr.flush()
    except (OSError, ValueError):
        pass                      # nobody is listening any more; keep running


def explain(exc):
    """A short, plain name for a network error."""
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timed out"
    if isinstance(exc, ConnectionRefusedError):
        return "connection refused"
    if isinstance(exc, http.client.RemoteDisconnected):
        return "closed without an answer"
    if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
        return "connection reset"
    if isinstance(exc, http.client.IncompleteRead):
        return "answer cut short"
    return "%s: %s" % (type(exc).__name__, exc)


def http_call(method, url, body=None, headers=None, connect_timeout_s=2.0, timeout_s=10.0):
    """One whole HTTP request, with the answer read to the end.

    Returns (status, body bytes). Raises one of NET_ERRORS when the other
    side cannot be reached or does not answer in time.
    """
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=connect_timeout_s)
    try:
        conn.connect()
        conn.sock.settimeout(timeout_s)
        send = {"Connection": "close"}
        send.update(headers or {})
        conn.request(method, path, body=body, headers=send)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _why(body):
    """The reason in a JSON error body, as " (reason)", or "" if there is none."""
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return ""
    if not isinstance(obj, dict):
        return ""
    word = obj.get("status") or obj.get("error")
    if isinstance(word, dict):
        word = word.get("message")
    return " (%s)" % str(word)[:80] if word else ""


def probe_backend(url, kind, timeout_s):
    """Probe one backend. Returns (ok, data, error).

    ok is what GET /health said (200 = healthy; llama.cpp says 503 while its
    model loads). data holds the numbers from GET /metrics, None where they
    are unknown; a missing /metrics does not make a backend unhealthy.
    """
    try:
        status, body = http_call("GET", url + "/health", None, None, timeout_s, timeout_s)
    except NET_ERRORS as exc:
        return False, {}, "health: " + explain(exc)
    if status != 200:
        return False, {}, "health answered %d%s" % (status, _why(body))
    data = {}
    if kind == "forecast":
        data = parse_forecast_metrics(None)
        try:
            status, body = http_call("GET", url + "/metrics", None, None, timeout_s, timeout_s)
            if status == 200:
                data = parse_forecast_metrics(json.loads(body.decode("utf-8", "replace")))
        except NET_ERRORS + (ValueError,):
            pass
    elif kind in PROM_NAMES:
        data = parse_llm_metrics("", kind)
        try:
            status, body = http_call("GET", url + "/metrics", None, None, timeout_s, timeout_s)
            if status == 200:
                data = parse_llm_metrics(body.decode("utf-8", "replace"), kind)
        except NET_ERRORS:
            pass
    return True, data, None


def run_heal_action(action):
    """Run one heal action from the config file. Returns (ok, text for the decision)."""
    if action.get("http"):
        spec = action["http"]
        method = str(spec.get("method") or "POST").upper()
        body, headers = None, {}
        if spec.get("json") is not None:
            body = json.dumps(spec["json"]).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            status, _ = http_call(method, spec["url"], body, headers,
                                  HEAL_HTTP_TIMEOUT_S, HEAL_HTTP_TIMEOUT_S)
        except NET_ERRORS as exc:
            return False, "http %s %s: %s" % (method, spec["url"], explain(exc))
        return status < 400, "http %s %s answered %d" % (method, spec["url"], status)
    cmd = [str(x) for x in action["cmd"]]
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=HEAL_CMD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "cmd %s: stopped after %d s" % (cmd[0], HEAL_CMD_TIMEOUT_S)
    except OSError as exc:
        return False, "cmd %s: %s" % (cmd[0], exc)
    output = done.stdout.decode("utf-8", "replace").strip()[:200]
    return done.returncode == 0, "cmd %s exit %d: %s" % (cmd[0], done.returncode, output)


# ---------------------------------------------------------------- merging an ensemble

def _combine(series, weights, how):
    """Combine the shards' lists item by item. None values are skipped.

    how: "mean" (weighted by member counts), "min", "max" or "first".
    """
    lists = [(w, s) for w, s in zip(weights, series) if isinstance(s, list)]
    if not lists:
        return None
    out = []
    for i in range(max(len(s) for _, s in lists)):
        values = [(w, num(s[i])) for w, s in lists if i < len(s) and num(s[i]) is not None]
        if not values:
            out.append(None)
        elif how == "mean":
            total = float(sum(w for w, _ in values))
            out.append(round(sum(w * v for w, v in values) / total, 3))
        elif how == "min":
            out.append(min(v for _, v in values))
        elif how == "max":
            out.append(max(v for _, v in values))
        else:
            out.append(values[0][1])
    return out


def merge_shards(shards):
    """Merge the forecast answers of an ensemble's shards into one answer.

    Each shard is a dict with "members", "member_offset" and "answer" (the
    forecast server's JSON). t2m_c comes from the shard holding global member
    0 (null if no shard does). Series named *_mean are weighted by member
    counts, *_min is the min of mins, *_max the max of maxes. truth_t2m_c is
    taken from whichever shard has a value. units are summed. timing_ms gets
    the longest queue wait and the summed GPU time; the caller sets "total".
    """
    shards = sorted(shards, key=lambda s: s["member_offset"])
    lead = shards[0]
    answers = [s["answer"] if isinstance(s["answer"], dict) else {} for s in shards]
    points = [a.get("point") if isinstance(a.get("point"), dict) else {} for a in answers]
    weights = [s["members"] for s in shards]
    out = dict(answers[0])
    out["members"] = sum(weights)
    out["member_offset"] = lead["member_offset"]
    nodes = []
    for a in answers:
        if a.get("node") is not None and str(a["node"]) not in nodes:
            nodes.append(str(a["node"]))
    if nodes:
        out["node"] = "+".join(nodes)
    point = dict(points[0])
    keys = []
    for p in points:
        keys.extend(k for k in p if k not in keys)
    for key in keys:
        series = [p.get(key) for p in points]
        if key == "t2m_c":
            point[key] = points[0].get(key) if lead["member_offset"] == 0 else None
        elif key.startswith("truth_"):
            point[key] = _combine(series, weights, "first")
        elif key.endswith("_mean"):
            point[key] = _combine(series, weights, "mean")
        elif key.endswith("_min"):
            point[key] = _combine(series, weights, "min")
        elif key.endswith("_max"):
            point[key] = _combine(series, weights, "max")
        elif point.get(key) is None:
            point[key] = next((x for x in series if x is not None), None)
    out["point"] = point
    out["units"] = sum(num(a.get("units")) or 0 for a in answers)
    timings = [a.get("timing_ms") if isinstance(a.get("timing_ms"), dict) else {} for a in answers]
    queues = [num(t.get("queue")) for t in timings if num(t.get("queue")) is not None]
    gpus = [num(t.get("gpu")) for t in timings if num(t.get("gpu")) is not None]
    out["timing_ms"] = {"queue": max(queues) if queues else None,
                        "gpu": round(sum(gpus), 1) if gpus else None, "total": None}
    return out


# ---------------------------------------------------------------- the HTTP side

class DecisionLog(object):
    """Appends every decision to a JSON-lines file. A path of "" or None turns it off."""

    def __init__(self, path):
        self.file = None
        self.lock = threading.Lock()
        if path:
            try:
                self.file = open(path, "a", buffering=1)  # line buffered: one write per line
            except OSError as exc:
                warn("cannot open the decision log %s (%s); going on without it" % (path, exc))

    def write(self, decision):
        with self.lock:
            if self.file is None:
                return
            try:
                self.file.write(json.dumps(decision) + "\n")
            except (OSError, ValueError) as exc:
                warn("cannot write the decision log (%s); going on without it" % exc)
                self.file = None

    def close(self):
        with self.lock:
            if self.file is not None:
                self.file.close()
                self.file = None


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True          # a stuck client never keeps the process alive
    request_queue_size = 256       # room for a burst of new connections
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        """A client that vanishes is normal. Anything else gets one line, not a traceback."""
        exc = sys.exc_info()[1]
        if not isinstance(exc, OSError):
            warn("%s: %s" % (type(exc).__name__, exc))


class Router(object):
    """The server, the probe threads, the heal actions and the ensemble fan-out.

    In a test:  app = Router(make_config({...})); port = app.start(port=0); ...; app.stop()
    """

    def __init__(self, config, now=time.monotonic):
        self.cfg = config
        self.log = DecisionLog(config.get("decision_log"))
        self.core = Core(config, now=now, sink=self.log.write)
        self.stopping = threading.Event()
        self.httpd = None
        self.threads = []
        self.host = None
        self.port = None

    def start(self, host="127.0.0.1", port=None):
        """Bind, then serve and probe on background threads. Returns the port in use."""
        if port is None:
            port = self.cfg["port"]
        self.httpd = RouterServer((host, int(port)), Handler)
        self.httpd.app = self
        self.host, self.port = host, self.httpd.server_address[1]
        self._spawn(self.httpd.serve_forever, 0.1)
        for pool in self.core.pools.values():
            for b in pool.backends:
                self._spawn(self._probe_loop, pool.name, b.name, b.url, b.probe)
        return self.port

    def stop(self):
        """Stop serving and probing, and close the log."""
        self.stopping.set()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.log.close()

    def _spawn(self, target, *args):
        thread = threading.Thread(target=target, args=args)
        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    # ---- probes and heals

    def _probe_loop(self, pool_name, name, url, kind):
        """Probe one backend every probe_s seconds until the router stops."""
        while not self.stopping.is_set():
            try:
                before = self.core.open_units(pool_name, name)
                ok, data, error = probe_backend(url, kind, self.cfg["connect_timeout_s"])
                if self.stopping.is_set():
                    break
                self.core.probe_result(pool_name, name, ok, data, error, before)
                self.run_due_heals()
            except Exception as exc:  # keep probing whatever happens
                warn("probe of %s/%s: %s: %s" % (pool_name, name, type(exc).__name__, exc))
            self.stopping.wait(self.cfg["probe_s"])

    def run_due_heals(self):
        """Start a thread for every heal action that is due."""
        for pool_name, name, action, attempt in self.core.due_heals():
            thread = threading.Thread(target=self._heal, args=(pool_name, name, action, attempt))
            thread.daemon = True
            thread.start()

    def _heal(self, pool_name, name, action, attempt):
        try:
            ok, text = run_heal_action(action)
        except Exception as exc:
            ok, text = False, "%s: %s" % (type(exc).__name__, exc)
        self.core.heal_done(pool_name, name, attempt, ok, text)

    # ---- ensemble fan-out

    def ensemble(self, request):
        """Run POST /v1/ensemble. Returns (status, answer, backends used, shard re-runs)."""
        members = request.get("members", 1)
        base = request.get("member_offset", 0)
        for label, value, least in (("members", members, 1), ("member_offset", base, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                return 400, {"error": "%s must be a whole number, %d or more" % (label, least)}, None, 0
        lead = request.get("lead_hours")       # the forecast servers check it; here it only sizes the work
        steps = 1
        if isinstance(lead, int) and not isinstance(lead, bool) and 6 <= lead <= 6000:
            steps = lead // 6
        plan = self.core.plan_fanout("forecast", members, steps)
        if plan is None:
            self.core.count_error()
            return 503, {"error": "no healthy backend"}, None, 0
        started = time.monotonic()
        shards, offset = [], base
        for name, url, share in plan:
            shards.append({"backend": name, "url": url, "members": share, "member_offset": offset,
                           "units": share * steps, "retried": False, "ms": None, "answer": None,
                           "error": None})
            offset += share
        threads = [threading.Thread(target=self._run_shard, args=(s, request)) for s in shards]
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()
        total_ms = (time.monotonic() - started) * 1000.0
        public = [collections.OrderedDict((k, s[k]) for k in
                                          ("backend", "members", "member_offset", "ms", "retried"))
                  for s in shards]
        names = []
        for s in shards:
            if s["backend"] not in names:
                names.append(s["backend"])
        used = "+".join(names)
        retries = sum(1 for s in shards if s["retried"])
        failed = [s for s in shards if s["error"]]
        if failed:
            status, answer = failed[0]["error"]
            if status >= 500:
                self.core.count_error()
            answer = dict(answer)
            answer["shards"] = public
            return status, answer, used, retries
        starts = ["%s from %s" % (s["backend"], s["answer"].get("init_time")) for s in shards]
        if len(set(s["answer"].get("init_time") for s in shards)) > 1:
            # "latest" is not the same analysis on every node. Mixing the shards would give
            # numbers that belong to no forecast at all, so say so instead.
            self.core.count_error()
            return 502, {"error": "the nodes started from different analyses (%s). Send the newest "
                                  "state to every node (POST /v1/state), or name an init time that "
                                  "all of them hold." % ", ".join(starts),
                         "shards": public}, used, retries
        merged = merge_shards(shards)
        merged["timing_ms"]["total"] = round(total_ms, 1)
        merged["shards"] = public
        return 200, merged, used, retries

    def _run_shard(self, shard, request):
        """Run one shard: its backend first, then once more on another one if that fails."""
        core = self.core
        body = dict(request)
        body["members"] = shard["members"]
        body["member_offset"] = shard["member_offset"]
        payload = json.dumps(body).encode("utf-8")
        timeout_s = core.pools["forecast"].first_byte_timeout_s
        name, url = shard["backend"], shard["url"]
        began = time.monotonic()
        errors = []
        for attempt in (0, 1):
            status, answer, error = None, None, None
            t0 = time.monotonic()
            try:
                status, raw = http_call("POST", url + "/v1/forecast", payload,
                                        {"Content-Type": "application/json"},
                                        self.cfg["connect_timeout_s"], timeout_s)
                try:
                    answer = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    answer = None
                if status != 200:
                    error = "status %d%s" % (status, _why(raw))
                elif not isinstance(answer, dict):
                    error = "the answer was not a JSON object"
            except NET_ERRORS as exc:
                error = explain(exc)
            now = time.monotonic()
            shard["ms"] = round((now - began) * 1000.0, 1)
            if error is None:
                core.finish("forecast", name, True, (now - t0) * 1000.0, units=shard["units"])
                shard["answer"] = answer
                return
            # Only a dead connection or a 502/503/504 counts against the backend's health.
            broke = status is None or status in RETRY_STATUSES
            core.finish("forecast", name, False if broke else None, error=error, units=shard["units"])
            errors.append("%s: %s" % (name, error))
            if status is not None and 400 <= status < 500:
                # The request itself is wrong. Another backend would say the same.
                shard["error"] = (status, answer if isinstance(answer, dict) else {"error": error})
                return
            if attempt == 1:
                shard["error"] = (502, {"error": "an ensemble shard failed twice",
                                        "detail": "; ".join(errors)})
                return
            choice = core.route("forecast", exclude=[name], retries=1, units=shard["units"],
                                after="ensemble shard: %s failed (%s)" % (name, error))
            if choice is None:
                shard["error"] = (503, {"error": "no healthy backend", "detail": "; ".join(errors)})
                return
            name, url = choice.backend, choice.url
            shard["backend"] = name
            shard["retried"] = True

    # ---- an analysis for every node

    def push_state(self, body):
        """Run POST /v1/state: store the analysis on every forecast backend in service.

        Sent to one backend only, the nodes would hold different "latest"
        analyses, and a forecast would depend on where it lands. Returns
        (status, answer, backends used). The answer is the first backend's
        ({"latest": ...}) plus "backends" (how each one answered) and
        "left_out" (the unhealthy and draining ones: they did not get it).
        """
        core = self.core
        plan, left_out = core.plan_all("forecast", "an analysis for every node")
        if plan is None:
            core.count_error()
            return 503, {"error": "no healthy backend", "left_out": left_out}, None
        timeout_s = core.pools["forecast"].first_byte_timeout_s
        rows = [{"backend": name, "url": url, "status": None, "answer": None, "error": None}
                for name, url in plan]

        def send(row):
            try:
                row["status"], raw = http_call("POST", row["url"] + "/v1/state", body,
                                               {"Content-Type": "application/json"},
                                               self.cfg["connect_timeout_s"], timeout_s)
                try:
                    row["answer"] = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    pass
                if row["status"] != 200:
                    row["error"] = "status %d%s" % (row["status"], _why(raw))
            except NET_ERRORS as exc:
                row["error"] = explain(exc)
            broke = row["status"] is None or row["status"] in RETRY_STATUSES
            ok = True if row["error"] is None else (False if broke else None)
            core.finish("forecast", row["backend"], ok, error=row["error"])

        threads = [threading.Thread(target=send, args=(row,)) for row in rows]
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()
        public = []
        for row in rows:
            latest = row["answer"].get("latest") if isinstance(row["answer"], dict) else None
            public.append(collections.OrderedDict([("backend", row["backend"]), ("status", row["status"]),
                                                   ("latest", latest), ("error", row["error"])]))
        used = "+".join(row["backend"] for row in rows)
        failed = [row for row in rows if row["error"]]
        if not failed:
            status = 200
            answer = dict(rows[0]["answer"]) if isinstance(rows[0]["answer"], dict) else {}
        elif (failed[0]["status"] is not None and 400 <= failed[0]["status"] < 500
              and isinstance(failed[0]["answer"], dict)):
            status = failed[0]["status"]          # the request itself is wrong: the server's own words
            answer = dict(failed[0]["answer"])
        else:
            core.count_error()
            status = 502
            answer = {"error": "the analysis did not reach every node",
                      "detail": "; ".join("%s: %s" % (row["backend"], row["error"]) for row in failed)}
        answer["backends"] = public
        answer["left_out"] = left_out
        return status, answer, used


EXPOSED = "X-Backend, X-Router-Policy, X-Router-Retries, X-Router-Affinity"
CORS_HEADERS = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Session-Id"),
    ("Access-Control-Expose-Headers", EXPOSED),
    ("Access-Control-Max-Age", "600"),
)


def pool_for_path(path):
    """The pool that serves this path, or None."""
    for pool, paths in POOL_PATHS.items():
        for known in paths:
            if path == known or path.startswith(known + "/"):
                return pool
    return None


class Handler(BaseHTTPRequestHandler):
    """One client connection. /router/* and / are answered here; the rest is proxied."""

    protocol_version = "HTTP/1.1"
    server_version = "wx-router"
    disable_nagle_algorithm = True   # streamed pieces leave at once
    timeout = 300                    # drop a client that says nothing for this long

    def log_message(self, fmt, *args):
        pass                         # the decision log is the record; no line per request

    def do_GET(self):
        self._started = False
        try:
            self._serve()
        except OSError:
            self.close_connection = True      # the client went away
        except Exception as exc:              # a bug: say so, and keep the server up
            warn("%s %s: %s: %s" % (self.command, self.path, type(exc).__name__, exc))
            self.close_connection = True
            if not self._started:
                try:
                    self._json(500, {"error": "router bug: %s: %s" % (type(exc).__name__, exc)})
                except OSError:
                    pass

    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = do_GET

    def _serve(self):
        path = urlsplit(self.path).path
        body = self._read_body()
        if body is None:
            return
        if self.command == "OPTIONS":         # a browser's preflight
            self._started = True
            self.send_response(204)           # 204 has no body, so no Content-Length either
            for name, value in CORS_HEADERS:
                self.send_header(name, value)
            self.end_headers()
        elif path.startswith("/router/"):
            self._admin(path, body)
        elif path == "/" and self.command in ("GET", "HEAD"):
            self._dashboard()
        elif pool_for_path(path) is None:
            self._json(404, {"error": "no pool serves %s; the router's own pages are under "
                                      "/router/, start with /router/status" % path})
        elif path == "/v1/ensemble" and self.command == "POST":
            self._ensemble(body)
        elif path == "/v1/state" and self.command == "POST":
            self._state(body)                 # to every node, not to one
        else:
            self._proxy(pool_for_path(path), body)

    # ---- small answers

    def _read_body(self):
        """The request body (b"" if there is none), or None after answering an error."""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            self._json(411, {"error": "send a Content-Length; chunked request bodies are not supported"})
            return None
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            self._json(400, {"error": "bad Content-Length"})
            return None
        body = self.rfile.read(length) if length else b""
        if len(body) < length:                # the client hung up half way
            self.close_connection = True
            return None
        return body

    def _json(self, status, obj, extra=()):
        data = json.dumps(obj).encode("utf-8") + b"\n"
        self._started = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in CORS_HEADERS:
            self.send_header(name, value)
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json_body(self, body):
        """The body as a JSON object, or None after answering 400."""
        try:
            obj = json.loads(body.decode("utf-8"))
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            self._json(400, {"error": "the body must be a JSON object"})
            return None
        return obj

    def _dashboard(self):
        try:
            with open(os.path.join(REPO_ROOT, "dashboard", "index.html"), "rb") as f:
                data, kind = f.read(), "text/html; charset=utf-8"
        except OSError:
            data = b"The router is up. There is no dashboard/index.html here; try /router/status\n"
            kind = "text/plain; charset=utf-8"
        self._started = True
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _admin(self, path, body):
        core = self.server.app.core
        reading = self.command in ("GET", "HEAD")
        if path == "/router/status" and reading:
            self._json(200, core.status())
        elif path == "/router/health" and reading:
            self._json(200, core.health())
        elif path == "/router/policy" and self.command == "POST":
            request = self._json_body(body)
            if request is None:
                return
            try:
                self._json(200, {"policy": core.set_policy(request.get("policy"))})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
        elif path == "/router/drain" and self.command == "POST":
            request = self._json_body(body)
            if request is None:
                return
            name, drain = request.get("backend"), request.get("drain")
            if not isinstance(name, str) or not isinstance(drain, bool):
                self._json(400, {"error": 'send {"backend": "B", "drain": true} (false puts it back)'})
                return
            pools = core.set_drain(name, drain)
            if pools:
                self._json(200, {"backend": name, "drain": drain, "pools": pools})
            else:
                self._json(404, {"error": "no backend is named '%s'" % name})
        elif path in ("/router/status", "/router/health"):
            self._json(405, {"error": "use GET for %s" % path})
        elif path in ("/router/policy", "/router/drain"):
            self._json(405, {"error": "use POST for %s" % path})
        else:
            self._json(404, {"error": "unknown path; the router has /router/status, "
                                      "/router/health, /router/policy and /router/drain"})

    def _ensemble(self, body):
        app = self.server.app
        request = self._json_body(body)
        if request is None:
            return
        status, answer, used, retries = app.ensemble(request)
        extra = [("X-Router-Policy", app.core.policy), ("X-Router-Retries", str(retries))]
        if used:
            extra.insert(0, ("X-Backend", used))
        self._json(status, answer, extra)

    def _state(self, body):
        app = self.server.app
        status, answer, used = app.push_state(body)
        extra = [("X-Router-Policy", app.core.policy), ("X-Router-Retries", "0")]
        if used:
            extra.insert(0, ("X-Backend", used))
        self._json(status, answer, extra)

    # ---- proxying

    def _affinity_key(self, pool_name, body):
        """The session key: the X-Session-Id header, else the JSON body's "user", else None."""
        if not self.server.app.core.pools[pool_name].affinity:
            return None
        key = (self.headers.get("X-Session-Id") or "").strip()
        if key:
            return key
        if b'"user"' in body:
            try:
                obj = json.loads(body.decode("utf-8"))
            except ValueError:
                return None
            if isinstance(obj, dict) and obj.get("user") not in (None, ""):
                return str(obj["user"])
        return None

    def _proxy(self, pool_name, body):
        """Send the request to a backend; if it fails before answering, once more to another."""
        core = self.server.app.core
        key = self._affinity_key(pool_name, body)
        units = 0
        if pool_name == "forecast":
            units = forecast_units(self.command, urlsplit(self.path).path, body)
        tried, errors = [], []
        for attempt in (0, 1):
            choice = core.route(pool_name, key=key, exclude=tried, retries=attempt,
                                after=errors[-1] if errors else None, units=units)
            if choice is None:
                core.count_error()
                answer = {"error": "no healthy backend"}
                if errors:
                    answer["detail"] = "; ".join(errors)
                self._json(503, answer, [("X-Router-Policy", core.policy), ("X-Router-Retries", "0")])
                return
            finished = []

            def finish(ok, latency_ms=None, error=None, choice=choice, finished=finished):
                if not finished:          # exactly once per attempt, whatever happens
                    finished.append(True)
                    core.finish(pool_name, choice.backend, ok, latency_ms, error, units)

            try:
                error = self._attempt(pool_name, choice, body, attempt, finish)
            finally:
                finish(None)
            if error is None:
                return
            tried.append(choice.backend)
            errors.append("%s failed: %s" % (choice.backend, error))
        core.count_error()
        self._json(502, {"error": "the backend failed, and so did the retry",
                         "detail": "; ".join(errors)},
                   [("X-Router-Policy", core.policy), ("X-Router-Retries", "1")])

    def _forward_headers(self, host, body):
        """The client's end-to-end headers, plus the ones this hop needs."""
        named = [t.strip().lower() for t in (self.headers.get("Connection") or "").split(",")]
        out = []
        for name, value in self.headers.items():
            low = name.lower()
            if low in HOP_BY_HOP or low in named or low in ("host", "content-length", "expect"):
                continue
            out.append((name, value))
        out.append(("Host", host))
        out.append(("Connection", "close"))
        if body or self.command in ("POST", "PUT", "PATCH"):
            out.append(("Content-Length", str(len(body))))
        return out

    def _attempt(self, pool_name, choice, body, attempt, finish):
        """Try one backend.

        Returns None when an answer was passed on to the client (whole or in
        part), or the error text when the backend failed before any byte was
        passed on; only then may the caller retry.
        """
        app = self.server.app
        pool = app.core.pools[pool_name]
        target = urlsplit(self.path)
        parts = urlsplit(choice.url)
        path = parts.path.rstrip("/") + target.path + ("?" + target.query if target.query else "")
        began = time.monotonic()
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                          timeout=app.cfg["connect_timeout_s"])
        try:
            conn.connect()
            conn.sock.settimeout(pool.first_byte_timeout_s)   # also the longest gap in a stream
            conn.putrequest(self.command, path, skip_host=True, skip_accept_encoding=True)
            for name, value in self._forward_headers(parts.netloc, body):
                conn.putheader(name, value)
            conn.endheaders(body if body else None)
            resp = conn.getresponse()                         # waits for the first byte
        except NET_ERRORS as exc:
            conn.close()
            finish(False, error=explain(exc))
            return explain(exc)
        except ValueError as exc:         # a path or header that cannot be sent on as it is
            conn.close()
            finish(None)
            self._json(400, {"error": "bad request: %s" % exc})
            return None

        status = resp.status
        gateway_error = None
        if status in RETRY_STATUSES:
            gateway_error = "status %d" % status
            if attempt == 0:
                conn.close()
                finish(False, error=gateway_error)
                return gateway_error
            # This was the retry. Pass the backend's own answer on; it still counts against it.

        no_body = self.command == "HEAD" or status in (204, 304) or status < 200
        length = resp.length          # None when the backend gave no Content-Length
        chunked = False
        skip = set(HOP_BY_HOP)
        skip.add("content-length")
        skip.update(t.strip().lower() for t in (resp.getheader("Connection") or "").split(","))
        try:
            self._started = True
            self.send_response_only(status, resp.reason)
            seen = set()
            for name, value in resp.getheaders():
                if name.lower() not in skip:
                    seen.add(name.lower())
                    self.send_header(name, value)
            if no_body:
                given = resp.getheader("Content-Length")
                if given is not None and status >= 200 and status != 204:
                    self.send_header("Content-Length", given)
            elif length is not None:
                self.send_header("Content-Length", str(length))
            elif self.request_version == "HTTP/1.0":
                self.send_header("Connection", "close")       # no chunks for an old client
            else:
                chunked = True
                self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Backend", choice.backend)
            self.send_header("X-Router-Policy", choice.policy)
            self.send_header("X-Router-Retries", str(attempt))
            if pool_name == "llm":
                self.send_header("X-Router-Affinity", choice.affinity or "none")
            if "access-control-allow-origin" not in seen:
                self.send_header("Access-Control-Allow-Origin", "*")
            if "access-control-expose-headers" not in seen:
                self.send_header("Access-Control-Expose-Headers", EXPOSED)
            self.end_headers()
        except OSError:
            conn.close()                  # the client hung up: not the backend's fault
            self.close_connection = True
            finish(None)
            return None

        # The body: piece by piece, as it arrives. Nothing is held back.
        sent, cut, client_gone = 0, None, False
        while not no_body:
            try:
                piece = resp.read1(65536)
            except NET_ERRORS as exc:
                cut = "broke mid-answer: %s" % explain(exc)
                break
            if not piece:
                break
            try:
                if chunked:
                    self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
                else:
                    self.wfile.write(piece)
                self.wfile.flush()
            except OSError:
                client_gone = True
                break
            sent += len(piece)
        if cut is None and not client_gone and not no_body and length is not None and sent < length:
            cut = "broke mid-answer: %d of %d bytes" % (sent, length)
        if chunked and cut is None and not client_gone:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                client_gone = True
        conn.close()
        took_ms = (time.monotonic() - began) * 1000.0
        if client_gone:
            self.close_connection = True  # a client that hangs up is not a backend failure
            finish(None)
        elif cut is not None:
            self.close_connection = True  # the only way left to tell the client it is incomplete
            app.core.count_error()
            finish(False, error=cut)
        elif gateway_error is not None:
            app.core.count_error()
            finish(False, error=gateway_error)
        elif status >= 500:
            app.core.count_error()        # a 500 is passed on as it is: no retry, no health mark
            finish(None)
        else:
            finish(True, took_ms)
        return None


# ---------------------------------------------------------------- command line

def raise_open_file_limit(want=4096):
    """Each request in flight holds two sockets, and macOS starts with a limit of 256."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY:
            want = min(want, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The router: one address in front of both boxes.",
        epilog="Example: python router/router.py --config router/config.example.json")
    parser.add_argument("--config", help="JSON config file (see router/config.example.json)")
    parser.add_argument("--port", type=int, help="port to listen on; 0 picks a free one "
                                                 "(default: the config's port, 8100)")
    parser.add_argument("--policy", choices=POLICIES, help="policy to start with")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on; 0.0.0.0 for every interface (default 127.0.0.1)")
    parser.add_argument("--forecast", metavar="A=URL,B=URL",
                        help="forecast servers, for example A=http://nodeA:8101,B=http://nodeB:8101")
    parser.add_argument("--llm", metavar="A=URL,B=URL",
                        help="LLM servers, for example A=http://nodeA:8000,B=http://nodeB:8000")
    parser.add_argument("--llm-probe", choices=("llamacpp", "vllm", "http"), default="llamacpp",
                        help="how to probe the --llm servers (default llamacpp)")
    args = parser.parse_args(argv)
    try:
        raw = {}
        if args.config:
            with open(args.config) as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("%s must hold a JSON object" % args.config)
        pools = raw.get("pools") or {}
        if not isinstance(pools, dict):
            raise ValueError("config 'pools' must be an object with 'forecast' and/or 'llm'")
        raw["pools"] = pools
        if args.forecast:
            pools.setdefault("forecast", {})["backends"] = parse_backend_list(args.forecast, "forecast")
        if args.llm:
            pools.setdefault("llm", {})["backends"] = parse_backend_list(args.llm, args.llm_probe)
        if args.policy:
            raw["policy"] = args.policy
        if args.port is not None:
            raw["port"] = args.port
        config = make_config(raw)
        if not any(pool["backends"] for pool in config["pools"].values()):
            raise ValueError("no backends; give --config FILE, or --forecast and/or --llm")
    except (OSError, ValueError) as exc:
        warn("cannot start: %s" % exc)
        return 2

    raise_open_file_limit()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    app = Router(config)
    try:
        port = app.start(args.host)
    except OSError as exc:
        warn("cannot listen on %s:%s: %s" % (args.host, config["port"], exc))
        return 1
    print("listening on http://%s:%d" % (args.host, port), flush=True)
    try:
        print("policy %s; %s; decisions go to %s" % (
            config["policy"],
            "; ".join("%s: %s" % (name, ", ".join(b["name"] for b in pool["backends"]) or "none")
                      for name, pool in config["pools"].items()),
            config.get("decision_log") or "nowhere (no log file)"), flush=True)
    except OSError:
        pass                      # whoever started us read the first line and closed the pipe
    while not stop.is_set():
        stop.wait(1.0)
    app.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
