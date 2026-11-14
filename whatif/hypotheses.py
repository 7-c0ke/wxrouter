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
    "hang": ("The server accepts requests and never answers: probes stall past 1 s and time out, in_flight "
             "climbs while nothing completes; /health may still answer 200. The router sees timeouts after "
             "its per-try timeout and fails over; p95 jumps to the timeout."),
}
GLOBAL = {
    "load:overload": ("Too much user traffic: traffic.rps clearly up (x1.5 or more), queue_depth and total/p95 "
                      "latency up on the box(es) carrying traffic, but probe compute_ms normal on both boxes and "
                      "health fine. Nothing changed in the router config."),
    "router:misconfig": ("The router's configuration changed, not the boxes: policy or preferred box changed near "
                         "the alert, and traffic share moves to a box that is worse (or a sick box keeps getting "
                         "traffic). Box metrics themselves are steady across the alert; traffic p95 rises."),
}

HYPOTHESES: dict[str, str] = {f"{b}:{k}": text for k, text in KINDS.items() for b in BOXES}
HYPOTHESES.update(GLOBAL)
ORDER = [f"{b}:{k}" for k in ("slow", "down", "error", "hang") for b in BOXES] + list(GLOBAL)


def runbook(h: str) -> str:
    return f"{h}: {HYPOTHESES[h]}"


def all_runbooks() -> str:
    return "\n".join(f"- {runbook(h)}" for h in ORDER)


def normalize(h: object) -> str | None:
    """Accept small variations an LLM writes ("Box B slow", "box_b:slow", "overload") -> a known id."""
    if not isinstance(h, str):
        return None
    s = h.strip().lower().replace("_", "-").replace(" ", "")
    if s in HYPOTHESES:
        return s
    aliases = {"overload": "load:overload", "load": "load:overload", "misconfig": "router:misconfig",
               "router": "router:misconfig", "routermisconfig": "router:misconfig", "loadoverload": "load:overload"}
    if s in aliases:
        return aliases[s]
    for b in BOXES:
        short = b.replace("-", "")
        for k in KINDS:
            if s in (f"{b}{k}", f"{short}:{k}", f"{short}{k}", f"{b}-{k}", f"{k}:{b}"):
                return f"{b}:{k}"
    return None
