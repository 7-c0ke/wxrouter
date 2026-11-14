"""The faults WhatIf can name, each with a short runbook: what it looks like in the tools."""
from __future__ import annotations

from whatif.config import BOXES

KINDS = {
    "slow": ("The box's GPU got slower (a training job beside it, thermal limits, a noisy neighbour). "
             "probe compute_ms rises on that box, total_ms with it, and gpu_util may rise while its own traffic does "
             "not; /health still answers 200 quickly. The router may move traffic off it (switch, share drops)."),
    "down": ("The box's forecast server is gone (crashed or killed). /health gives no answer at all (connection "
             "refused or timeout), /stats too, probes fail at once with no HTTP code; the router marks it unhealthy "
             "and fails over, so user traffic mostly survives. The supervisor shows the server restarting."),
    "error": ("The server is up but refuses work: forecasts and probes come back 503 immediately (fast, not "
              "slow), /health may answer 503 quickly; errors per second and the router's err_rate for that box "
              "rise; the router fails over. Compute time is not higher."),
    "hang": ("The server accepts requests and never answers: probes stall past 1 s and time out, so compute_ms "
             "goes to n/a (nothing completes), not up; /health may still answer 200 and rps does not change. The "
             "router sees timeouts after its per-try timeout and fails over; p95 jumps toward the timeout."),
}
GLOBAL = {
    "load:overload": ("Too much user traffic: traffic.rps clearly up (x1.5 or more, often a ramp), queue_depth and "
                      "in_flight up on the box(es) carrying traffic, total/p95 latency up. Probes there may stall >1 s "
                      "waiting in the queue, but those that finish show NORMAL compute_ms, and health answers 200. "
                      "Nothing changed in the router config. Proving line, from traffic(): `traffic.rps: 8 -> 24 "
                      "(+200%)` with `box-x.queue_depth` / `box-x.in_flight` up beside it."),
    "router:misconfig": ("The router's configuration changed, not the boxes: router.policy or preferred changed near "
                         "the alert, and traffic share moves to a worse box. That box may be slow, but it was ALREADY "
                         "slow before the alert (its compute_ms shows no change at t0); it now queues (probe stalls, "
                         "queue_depth up) because traffic was sent to it. Traffic p95 rises, rps does not. Proving "
                         "line, first in router_status(): `router.policy: adaptive -> round_robin (changed at -5s)` "
                         "(or router.preferred). A config change at t0 outweighs a slow or stalling box."),
}

HYPOTHESES: dict[str, str] = {f"{b}:{k}": text for k, text in KINDS.items() for b in BOXES}
HYPOTHESES.update(GLOBAL)
ORDER = [f"{b}:{k}" for k in ("slow", "down", "error", "hang") for b in BOXES] + list(GLOBAL)


def runbook(h: str) -> str:
    return f"{h}: {HYPOTHESES[h]}"


def all_runbooks() -> str:
    return "\n".join(f"- {runbook(h)}" for h in ORDER)


def _compact(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# Every spelling an LLM writes, squeezed to letters and digits: "box_a:slow", "Box A slow", "slow (box-a)",
# "boxa-slow" -> "boxaslow" or "slowboxa".
_ALIASES: dict[str, str] = {}
for _b in BOXES:
    for _k in KINDS:
        _ALIASES[_compact(_b) + _k] = _ALIASES[_k + _compact(_b)] = f"{_b}:{_k}"
for _a in ("loadoverload", "overload", "load", "overloadload", "trafficoverload", "highload"):
    _ALIASES[_a] = "load:overload"
for _a in ("routermisconfig", "misconfig", "router", "misconfigrouter", "routermisconfiguration",
           "misconfiguration", "routerconfig"):
    _ALIASES[_a] = "router:misconfig"


def normalize(h: object) -> str | None:
    """Any spelling of a hypothesis id -> the canonical id, or None when it names none of the ten."""
    if not isinstance(h, str):
        return None
    s = h.strip().lower()
    if s in HYPOTHESES:
        return s
    return _ALIASES.get(_compact(s))
