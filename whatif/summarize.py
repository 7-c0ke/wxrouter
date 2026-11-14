"""Tables from results.jsonl: per policy, per fault kind, and whatif paired against each other policy.

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


def per_policy(rows: list[dict], policies: list[str]) -> list[dict]:
    out = []
    for p in policies:
        rs = [r for r in rows if r["policy"] == p]
        if not rs:
            continue
        out.append({"policy": p, "n": len(rs), "ac1": _mean([r["ac1"] for r in rs]), "ac3": _mean([r["ac3"] for r in rs]),
                    "errors": sum(1 for r in rs if r.get("error")),
                    **{m: _mean([r.get(m, 0) for r in rs]) for m in METRICS}})
    return out


def per_kind(rows: list[dict], policies: list[str]) -> dict:
    kinds = sorted({r["kind"] for r in rows})
    return {k: {p: _mean([r["ac1"] for r in rows if r["kind"] == k and r["policy"] == p]) for p in policies} for k in kinds}


def paired(rows: list[dict], a: str, b: str) -> dict | None:
    by = {}
    for r in rows:
        by.setdefault(r["incident"], {})[r["policy"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    if not pairs:
        return None
    wins = sum(1 for x, y in pairs if x["ac1"] > y["ac1"])
    losses = sum(1 for x, y in pairs if x["ac1"] < y["ac1"])
    return {"a": a, "b": b, "n": len(pairs), "wins": wins, "losses": losses, "ties": len(pairs) - wins - losses,
            "p_sign": round(sign_test(wins, losses), 4),
            "d_ac1": _mean([x["ac1"] - y["ac1"] for x, y in pairs]),
            "d_tokens": _mean([x.get("tokens", 0) - y.get("tokens", 0) for x, y in pairs]),
            "d_wall_s": _mean([x.get("wall_s", 0) - y.get("wall_s", 0) for x, y in pairs])}


def markdown(rows: list[dict], policies: list[str]) -> str:
    pp = per_policy(rows, policies)
    lines = [f"# WhatIf evaluation: {len({r['incident'] for r in rows})} incidents", "",
             "| policy | n | AC@1 | AC@3 | tokens/inc | tool units | LLM calls | thinking calls | parse fails | wall s | errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in pp:
        lines.append(f"| {s['policy']} | {s['n']} | {s['ac1']:.2f} | {s['ac3']:.2f} | {s['tokens']:.0f} | {s['tool_units']:.1f} | "
                     f"{s['llm_calls']:.1f} | {s['thinking_calls']:.2f} | {s['parse_failures']:.2f} | {s['wall_s']:.1f} | {s['errors']} |")
    pk = per_kind(rows, policies)
    lines += ["", "AC@1 by fault kind:", "", "| kind | " + " | ".join(policies) + " |", "|---|" + "---:|" * len(policies)]
    for k, v in pk.items():
        n = len({r["incident"] for r in rows if r["kind"] == k})
        lines.append(f"| {k} (n={n}) | " + " | ".join(f"{v[p]:.2f}" for p in policies) + " |")
    if "whatif" in policies:
        lines += ["", "Paired, whatif vs each (same incidents; sign test on AC@1, ties dropped):", "",
                  "| vs | n | whatif wins | losses | ties | p | ΔAC@1 | Δtokens | Δwall s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for b in policies:
            if b == "whatif":
                continue
            c = paired(rows, "whatif", b)
            if c:
                lines.append(f"| {b} | {c['n']} | {c['wins']} | {c['losses']} | {c['ties']} | {c['p_sign']:.3f} | "
                             f"{c['d_ac1']:+.2f} | {c['d_tokens']:+.0f} | {c['d_wall_s']:+.1f} |")
    return "\n".join(lines) + "\n"


def write(rows: list[dict], out: Path, policies: list[str]) -> dict:
    summary = {"per_policy": per_policy(rows, policies), "per_kind": per_kind(rows, policies),
               "paired": [c for b in policies if b != "whatif" for c in [paired(rows, "whatif", b)] if c]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(rows, policies))
    return summary


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    write(rows, out, policies)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
