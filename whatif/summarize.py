"""Tables from results.jsonl: per policy, per fault kind, and whatif (and triage) paired against each other policy.

With repeated runs (rows carry `rep`), AC@1/AC@3 are the mean over all runs with the range of the per-run means;
the paired tests compare each incident's mean over runs. Every row says what produced it: the LLM (from the run's
config.json) or "rule (no LLM)".

    python -m whatif.summarize results/whatif/<ts>          rewrites summary.md / summary.json from results.jsonl
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

METRICS = ("tokens", "tool_units", "llm_calls", "thinking_calls", "wall_s", "parse_failures")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else 0.0


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value for wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def engine_label(cfg: dict | None) -> str:
    """What answered the LLM calls, for every table and chart."""
    cfg = (cfg or {}).get("config", cfg) or {}
    eps, model = cfg.get("llm_endpoints") or [], cfg.get("model") or "?"
    if any("integrate.api.nvidia.com" in e for e in eps):
        return f"Nemotron via NVIDIA API ({model})"
    return f"{model} at {', '.join(eps) or '?'}"


def policy_label(policy: str, engine: str) -> str:
    if policy == "rule":
        return "rule (no LLM)"
    if policy == "triage":
        return f"rule, then {engine} when unclear"
    return engine


def per_policy(rows: list[dict], policies: list[str]) -> list[dict]:
    out = []
    for p in policies:
        rs = [r for r in rows if r["policy"] == p]
        if not rs:
            continue
        reps = sorted({r.get("rep", 0) for r in rs})
        by_rep = lambda m: [_mean([r[m] for r in rs if r.get("rep", 0) == k]) for k in reps]
        s = {"policy": p, "n": len({r["incident"] for r in rs}), "runs": len(reps),
             "ac1": _mean([r["ac1"] for r in rs]), "ac3": _mean([r["ac3"] for r in rs]),
             "ac1_range": [min(by_rep("ac1")), max(by_rep("ac1"))], "ac3_range": [min(by_rep("ac3")), max(by_rep("ac3"))],
             "errors": sum(1 for r in rs if r.get("error")),
             **{m: _mean([r.get(m, 0) for r in rs]) for m in METRICS}}
        s["ac1_per_10k_tokens"] = s["ac1"] / (s["tokens"] / 1e4) if s["tokens"] else None
        out.append(s)
    return out


def per_kind(rows: list[dict], policies: list[str]) -> dict:
    kinds = sorted({r["kind"] for r in rows})
    return {k: {p: _mean([r["ac1"] for r in rows if r["kind"] == k and r["policy"] == p]) for p in policies} for k in kinds}


def paired(rows: list[dict], a: str, b: str) -> dict | None:
    by: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by.setdefault(r["incident"], {}).setdefault(r["policy"], []).append(r)
    avg = lambda rs: {m: _mean([r.get(m, 0) for r in rs]) for m in ("ac1", "tokens", "wall_s")}
    pairs = [(avg(v[a]), avg(v[b])) for v in by.values() if a in v and b in v]
    if not pairs:
        return None
    wins = sum(1 for x, y in pairs if x["ac1"] > y["ac1"])
    losses = sum(1 for x, y in pairs if x["ac1"] < y["ac1"])
    return {"a": a, "b": b, "n": len(pairs), "wins": wins, "losses": losses, "ties": len(pairs) - wins - losses,
            "p_sign": round(sign_test(wins, losses), 4),
            "d_ac1": _mean([x["ac1"] - y["ac1"] for x, y in pairs]),
            "d_tokens": _mean([x.get("tokens", 0) - y.get("tokens", 0) for x, y in pairs]),
            "d_wall_s": _mean([x.get("wall_s", 0) - y.get("wall_s", 0) for x, y in pairs])}


def _rng(s: dict, k: str) -> str:
    lo, hi = s[f"{k}_range"]
    return f"{s[k]:.2f}" + (f" ({lo:.2f}-{hi:.2f})" if s["runs"] > 1 else "")


def markdown(rows: list[dict], policies: list[str], engine: str = "LLM") -> str:
    pp = per_policy(rows, policies)
    runs = max((s["runs"] for s in pp), default=1)
    lines = [f"# WhatIf evaluation: {len({r['incident'] for r in rows})} incidents", "",
             f"Agents: **{engine}**. Baseline: **rule (no LLM)**."
             + (f" LLM policies ran {runs} times per incident: AC@1/AC@3 are the mean over all runs, with the range of "
                "the per-run means in brackets." if runs > 1 else ""), "",
             "| policy | engine | n | runs | AC@1 | AC@3 | tokens/inc | AC@1 per 10k tok | tool units | LLM calls | thinking calls | parse fails | wall s | errors |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in pp:
        per10k = "n/a (0 tokens)" if s["ac1_per_10k_tokens"] is None else f"{s['ac1_per_10k_tokens']:.3f}"
        lines.append(f"| {s['policy']} | {policy_label(s['policy'], engine)} | {s['n']} | {s['runs']} | {_rng(s, 'ac1')} | "
                     f"{_rng(s, 'ac3')} | {s['tokens']:.0f} | {per10k} | {s['tool_units']:.1f} | {s['llm_calls']:.1f} | "
                     f"{s['thinking_calls']:.2f} | {s['parse_failures']:.2f} | {s['wall_s']:.1f} | {s['errors']} |")
    pk = per_kind(rows, policies)
    lines += ["", "AC@1 by fault kind (mean over runs):", "", "| kind | " + " | ".join(policies) + " |", "|---|" + "---:|" * len(policies)]
    for k, v in pk.items():
        n = len({r["incident"] for r in rows if r["kind"] == k})
        lines.append(f"| {k} (n={n}) | " + " | ".join(f"{v[p]:.2f}" for p in policies) + " |")
    for a in ("whatif", "triage"):
        if a not in policies:
            continue
        lines += ["", f"Paired, {a} vs each (same incidents, each incident's mean over runs; sign test on AC@1, ties dropped):", "",
                  f"| vs | n | {a} wins | losses | ties | p | ΔAC@1 | Δtokens | Δwall s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for b in policies:
            if b == a:
                continue
            c = paired(rows, a, b)
            if c:
                lines.append(f"| {b} | {c['n']} | {c['wins']} | {c['losses']} | {c['ties']} | {c['p_sign']:.3f} | "
                             f"{c['d_ac1']:+.2f} | {c['d_tokens']:+.0f} | {c['d_wall_s']:+.1f} |")
    return "\n".join(lines) + "\n"


def write(rows: list[dict], out: Path, policies: list[str], engine: str = "LLM") -> dict:
    summary = {"engine": engine, "per_policy": per_policy(rows, policies), "per_kind": per_kind(rows, policies),
               "paired": [c for a in ("whatif", "triage") if a in policies for b in policies if b != a
                          for c in [paired(rows, a, b)] if c]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(rows, policies, engine))
    return summary


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else None
    write(rows, out, policies, engine_label(cfg))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
