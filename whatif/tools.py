"""Agent tools over a recorded window: compare [t0-30 s, t0) with [t0, t0+30 s].

Every tool answers in short lines an LLM can read, one per metric:

    box-b.compute_ms: 57 -> 171 (+200%), onset +2s, step

"onset" is when the metric first left its baseline (the first 20 s of the window before the alert), relative
to the alert; "step" means it stayed changed, "blip" that it came back, "ramp" that it kept growing.

Tools and their costs (a budget of 8 units per incident; a repeated call is free and gives the same text):
    box_health(box) 1   box_stats(box) 1   probe(box) 1   router_status() 1   traffic() 1   box_jobs(box) 2
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from whatif.config import BOXES

PRE_S, POST_S = 30.0, 30.0
COSTS = {"box_health": 1, "box_stats": 1, "probe": 1, "router_status": 1, "traffic": 1, "box_jobs": 2}
BOX_TOOLS = ("box_health", "box_stats", "probe", "box_jobs")
BUDGET = 8
STALL_MS = 1000.0

TOOL_DOCS = {
    "box_health": "box_health(box): GET /health answers (share ok, codes) and round-trip ms, before vs after",
    "box_stats": "box_stats(box): the box's own /stats: queue_depth, in_flight, gpu_util, p95 latency, errors per s",
    "probe": "probe(box): a small forecast every 2 s: compute_ms (GPU time) vs total_ms (round trip), stalls >1 s, errors",
    "router_status": "router_status(): router policy, preferred/active box, switches, failovers, per-box health and error rate",
    "traffic": "traffic(): user traffic through the router: rps, p95 ms, error %, share per box",
    "box_jobs": "box_jobs(box): the box supervisor: which servers and jobs (e.g. finetune) are running (cost 2)",
}


# -------------------------------------------------------------------- series and comparison
Series = list[tuple[float, float]]


def series(samples: list[dict], get: Callable[[dict], Optional[float]]) -> Series:
    out = []
    for s in samples:
        try:
            v = get(s)
        except (KeyError, TypeError, AttributeError):
            v = None
        if v is not None:
            out.append((s["t"], float(v)))
    return out


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


@dataclass
class Change:
    name: str
    before: Optional[float]
    after: Optional[float]
    onset: Optional[float] = None  # seconds relative to t0
    shape: str = "flat"  # flat | step | blip | ramp | n/a
    extra: str = ""

    @property
    def pct(self) -> Optional[float]:
        if self.before is None or self.after is None or abs(self.before) < 1e-9:
            return None
        return 100.0 * (self.after - self.before) / abs(self.before)

    def line(self) -> str:
        if self.before is None and self.after is None:
            return f"{self.name}: no data"
        s = f"{self.name}: {_fmt(self.before)} -> {_fmt(self.after)}"
        if self.pct is not None and self.shape != "flat":
            s += f" ({self.pct:+.0f}%)"
        if self.shape == "flat":
            s += ", no change"
        elif self.onset is not None:
            s += f", onset {self.onset:+.0f}s, {self.shape}"
        if self.extra:
            s += f", {self.extra}"
        return s


def compare(name: str, pts: Series, t0: float, rel: float = 0.3, abs_eps: float = 1e-6,
            agg: Callable[[list[float]], float] = statistics.median) -> Change:
    """Median before vs after, plus when and how it changed. A point 'deviates' when it is off the baseline by
    more than max(3 sd, rel x |baseline|, abs_eps)."""
    pre = [v for t, v in pts if t0 - PRE_S <= t < t0]
    post = [v for t, v in pts if t0 <= t < t0 + POST_S]
    before = agg(pre) if pre else None
    after = agg(post) if post else None
    ch = Change(name, before, after)
    if before is None or after is None:
        ch.shape = "n/a"
        return ch
    base = [v for t, v in pts if t0 - PRE_S <= t < t0 - 10] or pre
    b = statistics.median(base)
    sd = statistics.pstdev(base) if len(base) > 1 else 0.0
    thr = max(3 * sd, rel * abs(b), abs_eps)
    late = [(t, v) for t, v in pts if t0 - 10 <= t < t0 + POST_S]
    dev = [(t, abs(v - b) > thr) for t, v in late]
    first = next((t for t, d in dev if d), None)
    if first is None:
        return ch  # flat
    ch.onset = round(first - t0, 1)
    after_onset = [(t, v) for t, v in late if t >= first]
    flags = [d for t, d in dev if t >= first]
    frac = sum(flags) / len(flags)
    tail = flags[-max(1, len(flags) // 3):]
    if frac < 0.6 and not any(tail):
        ch.shape = "blip"
    elif frac >= 0.6:
        vals = [v for _, v in after_onset]
        third = max(1, len(vals) // 3)
        head_m, tail_m = statistics.median(vals[:third]), statistics.median(vals[-third:])
        ch.shape = "ramp" if abs(tail_m - b) > 1.5 * abs(head_m - b) and len(vals) >= 6 else "step"
    else:
        ch.shape = "step" if any(tail) else "blip"
    if ch.shape == "blip":
        dur = sum(1 for d in flags if d)
        ch.extra = f"~{dur} samples off baseline"
    return ch


def rate(pts: Series, t0: float) -> Series:
    """Per-second increase of a counter (resets count as 0)."""
    out = []
    for (t1, v1), (t2, v2) in zip(pts, pts[1:]):
        if t2 > t1:
            out.append((t2, max(0.0, (v2 - v1) / (t2 - t1))))
    return out


# -------------------------------------------------------------------- the tools
def _box(s: dict, box: str) -> dict:
    return (s.get("boxes") or {}).get(box) or {}


def _probes(samples: list[dict], box: str) -> list[tuple[float, dict]]:
    """Completed probes by start time, plus stalls seen while pending (one per probe)."""
    done, pending = {}, {}
    for s in samples:
        p = (s.get("probe") or {}).get(box)
        if not p:
            continue
        if "pending_ms" in p:
            pending[p["t"]] = max(pending.get(p["t"], 0.0), p["pending_ms"])
        else:
            done[p["t"]] = p
    for t, ms in pending.items():
        if t not in done:
            done[t] = {"t": t, "code": None, "compute_ms": None, "total_ms": ms, "pending": True}
    return sorted(done.items())


def _window_count(items: list[tuple[float, bool]], t0: float) -> tuple[int, int, int, int]:
    pre = [ok for t, ok in items if t0 - PRE_S <= t < t0]
    post = [ok for t, ok in items if t0 <= t < t0 + POST_S]
    return sum(pre), len(pre), sum(post), len(post)


def box_health(samples: list[dict], t0: float, box: str) -> str:
    codes = [(s["t"], _box(s, box).get("health", {}).get("code")) for s in samples if _box(s, box).get("health")]
    ok = [(t, c == 200) for t, c in codes]
    a, n, b, m = _window_count(ok, t0)
    post_codes = sorted({c for t, c in codes if t0 <= t < t0 + POST_S and c != 200})
    names = {0: "no answer", 503: "503"}
    lines = [f"{box}.health_ok: {a}/{n} -> {b}/{m}" + (f", failures: {', '.join(names.get(c, str(c)) for c in post_codes)}" if post_codes else "")]
    fails = [t for t, good in ok if not good and t >= t0 - 10]
    if fails and b < m:
        lines[0] += f", first failure {fails[0] - t0:+.0f}s"
    lines.append(compare(f"{box}.health_ms", series(samples, lambda s: _box(s, box)["health"]["ms"] if _box(s, box)["health"]["code"] == 200 else None), t0, abs_eps=5).line())
    return "\n".join(lines)


def box_stats(samples: list[dict], t0: float, box: str) -> str:
    st = lambda k: series(samples, lambda s: (_box(s, box).get("stats") or {}).get(k))
    lines = [compare(f"{box}.queue_depth", st("queue_depth"), t0, abs_eps=1.5, agg=statistics.mean).line(),
             compare(f"{box}.in_flight", st("in_flight"), t0, abs_eps=1.5, agg=statistics.mean).line(),
             compare(f"{box}.gpu_util", st("gpu_util"), t0, abs_eps=0.15, agg=statistics.mean).line(),
             compare(f"{box}.p95_latency_ms", st("p95_latency_ms"), t0, abs_eps=20).line(),
             compare(f"{box}.errors_per_s", rate(st("errors_total"), t0), t0, abs_eps=0.5, agg=statistics.mean).line()]
    missing = sum(1 for s in samples if t0 <= s["t"] < t0 + POST_S and _box(s, box) and _box(s, box).get("stats") is None)
    if missing:
        lines.append(f"{box}./stats unanswered in {missing} samples after the alert")
    return "\n".join(lines)


def probe(samples: list[dict], t0: float, box: str) -> str:
    ps = _probes(samples, box)
    comp = [(t, p["compute_ms"]) for t, p in ps if p.get("compute_ms") is not None]
    tot = [(t, p["total_ms"]) for t, p in ps if p.get("code") == 200]
    stalls = [(t, bool(p.get("pending") or p.get("timeout") or (p.get("total_ms") or 0) > STALL_MS)) for t, p in ps]
    errs = [(t, p.get("code") not in (200, None, 0) or (p.get("code") == 0 and not p.get("timeout") and not p.get("pending"))) for t, p in ps]
    sa, sn, sb, sm = _window_count(stalls, t0)
    ea, en, eb, em = _window_count(errs, t0)
    codes = sorted({str(p.get("code")) for t, p in ps if t0 <= t < t0 + POST_S and p.get("code") not in (200, None)})
    lines = [compare(f"{box}.compute_ms", comp, t0, abs_eps=5).line(),
             compare(f"{box}.total_ms", tot, t0, abs_eps=10).line(),
             f"{box}.probe_stalls(>1s): {sa}/{sn} -> {sb}/{sm}",
             f"{box}.probe_errors: {ea}/{en} -> {eb}/{em}" + (f" (codes {', '.join(codes)})" if codes else "")]
    return "\n".join(lines)


def router_status(samples: list[dict], t0: float) -> str:
    rs = [(s["t"], s["router"]) for s in samples if s.get("router")]
    pre = [r for t, r in rs if t0 - PRE_S <= t < t0]
    post = [r for t, r in rs if t0 <= t < t0 + POST_S]
    if not rs:
        return "router: no answer in the whole window"
    if not post:
        return "router: no answer after the alert"

    def mode(rows, key):
        vals = [r.get(key) for r in rows]
        return max(set(vals), key=vals.count) if vals else None

    lines = []
    for key in ("policy", "preferred", "active"):
        a, b = mode(pre, key), mode(post, key)
        changes = [(t - t0) for (t, r), (_, q) in zip(rs, rs[1:]) if r.get(key) != q.get(key) and t0 - 10 <= t < t0 + POST_S]
        lines.append(f"router.{key}: {a} -> {b}" + (f" (changed at {changes[0]:+.0f}s)" if changes else ""))
    sw = series(samples, lambda s: s["router"]["switches"])
    fo = series(samples, lambda s: s["router"]["failovers"])
    win = lambda pts, lo, hi: (max((v for t, v in pts if lo <= t < hi), default=0) - min((v for t, v in pts if lo <= t < hi), default=0))
    lines.append(f"router.switches: {win(sw, t0 - PRE_S, t0):.0f} -> {win(sw, t0, t0 + POST_S):.0f}")
    lines.append(f"router.failovers: {win(fo, t0 - PRE_S, t0):.0f} -> {win(fo, t0, t0 + POST_S):.0f}")
    for box in BOXES:
        hb = series(samples, lambda s: 1.0 if s["router"]["backends"][box]["healthy"] else 0.0)
        a = sum(v for t, v in hb if t0 - PRE_S <= t < t0); n = sum(1 for t, v in hb if t0 - PRE_S <= t < t0)
        b = sum(v for t, v in hb if t0 <= t < t0 + POST_S); m = sum(1 for t, v in hb if t0 <= t < t0 + POST_S)
        lines.append(f"router.{box}.healthy: {a:.0f}/{n} -> {b:.0f}/{m}")
        lines.append(compare(f"router.{box}.err_rate", series(samples, lambda s: s["router"]["backends"][box]["err_rate"]), t0, abs_eps=0.05, agg=statistics.mean).line())
        lines.append(compare(f"router.{box}.probe_ms", series(samples, lambda s: s["router"]["backends"][box]["probe_ms"]), t0, abs_eps=10).line())
    return "\n".join(lines)


def traffic(samples: list[dict], t0: float) -> str:
    tr = lambda k: series(samples, lambda s: s["traffic"][k])
    lines = [compare("traffic.rps", tr("rps"), t0, abs_eps=1.0, agg=statistics.mean).line(),
             compare("traffic.p95_ms", tr("p95_ms"), t0, abs_eps=30).line(),
             compare("traffic.err_pct", tr("err_pct"), t0, abs_eps=1.0, agg=statistics.mean).line()]
    for box in BOXES:
        share = series(samples, lambda s: (s["traffic"]["per_box"][box]["requests"] or 0) / max(1, sum((v["requests"] or 0) for v in s["traffic"]["per_box"].values()))
                       if sum((v["requests"] or 0) for v in s["traffic"]["per_box"].values()) else None)
        lines.append(compare(f"traffic.share_{box}", share, t0, abs_eps=0.2, agg=statistics.mean).line())
    return "\n".join(lines)


def box_jobs(samples: list[dict], t0: float, box: str) -> str:
    js = [(s["t"], s["jobs"][box]) for s in samples if (s.get("jobs") or {}).get(box)]
    if not js:
        return f"{box}.supervisor: not reachable (no supervisor data in the window)"
    pre = [j for t, j in js if t < t0]
    post = [j for t, j in js if t >= t0]
    a = pre[-1] if pre else None
    b = post[-1] if post else None
    lines = []
    for kind in ("servers", "jobs"):
        names = sorted(set((a or {}).get(kind, {})) | set((b or {}).get(kind, {})))
        for n in names:
            x, y = (a or {}).get(kind, {}).get(n), (b or {}).get(kind, {}).get(n)
            lines.append(f"{box}.{kind[:-1]}.{n}: {x} -> {y}")
    return "\n".join(lines) or f"{box}.supervisor: nothing configured"


TOOLS: dict[str, Callable] = {"box_health": box_health, "box_stats": box_stats, "probe": probe,
                              "router_status": router_status, "traffic": traffic, "box_jobs": box_jobs}


# -------------------------------------------------------------------- budgeted access
@dataclass
class ToolResult:
    key: str
    text: str
    cost: int
    cached: bool = False
    error: bool = False


@dataclass
class Toolbox:
    """The tools for one incident: a shared evidence board with a unit budget. A call already made is
    answered from the board for free."""
    samples: list[dict]
    t0: float
    budget: int = BUDGET
    spent: int = 0
    board: dict[str, str] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)

    @staticmethod
    def key(name: str, args: dict | None) -> str:
        box = (args or {}).get("box")
        return f"{name}({box})" if name in BOX_TOOLS else f"{name}()"

    def left(self) -> int:
        return self.budget - self.spent

    def call(self, name: str, args: dict | None = None, who: str = "") -> ToolResult:
        args = args or {}
        if name not in TOOLS:
            return ToolResult(name, f"unknown tool {name!r}; tools: {', '.join(TOOLS)}", 0, error=True)
        if name in BOX_TOOLS and args.get("box") not in BOXES:
            return ToolResult(name, f"{name} needs box = one of {', '.join(BOXES)}", 0, error=True)
        k = self.key(name, args)
        if k in self.board:
            self.log.append({"key": k, "who": who, "cost": 0, "cached": True})
            return ToolResult(k, self.board[k], 0, cached=True)
        cost = COSTS[name]
        if cost > self.left():
            return ToolResult(k, f"budget exhausted: {k} costs {cost}, {self.left()} left", 0, error=True)
        fn = TOOLS[name]
        text = fn(self.samples, self.t0, args["box"]) if name in BOX_TOOLS else fn(self.samples, self.t0)
        self.spent += cost
        self.board[k] = text
        self.log.append({"key": k, "who": who, "cost": cost, "cached": False})
        return ToolResult(k, text, cost)

    def evidence(self) -> str:
        if not self.board:
            return "(no evidence yet)"
        return "\n".join(f"## {k}\n{v}" for k, v in self.board.items())
