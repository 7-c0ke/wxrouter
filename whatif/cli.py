"""WhatIf command line.

    python -m whatif.cli diagnose --live [--apply] [--policy whatif]
        Record the stack for 60 s with the alert at the middle (or --pre/--post), diagnose, print the evidence
        trail and the remedy. --apply carries the remedy out and reports p95 before and 30 s after.
        The recording is saved to data/incidents/live-<time>.json so it can be replayed.
    python -m whatif.cli diagnose --incident data/incidents/<id>.json [--policy rule]
        Diagnose a saved incident (no stack needed).
    python -m whatif.cli eval --incidents data/incidents --policies single,equal,whatif,rule [--out DIR]
        Every policy on every incident; writes results/whatif/<time>/ (results.jsonl, summary.md, trails/).
    python -m whatif.cli record -n 10                       see whatif/scenarios.py
    python -m whatif.cli check                              which endpoints answer (LLM, router, boxes, supervisors)

Settings come from the environment and .env (see whatif/config.py).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

from whatif import config as wcfg
from whatif import remediate, scenarios
from whatif.eval import run_eval
from whatif.llm import LLMClient
from whatif.scheduler import POLICIES, diagnose
from whatif.telemetry import Recorder, load_incident, load_incidents, save_incident

ROOT = wcfg.ROOT


async def check(cfg: wcfg.Config) -> int:
    rows = [("router", f"{cfg.router}/router/status")]
    rows += [(b, f"{u}/health") for b, u in cfg.boxes.items()]
    rows += [(f"{b} supervisor", f"{u}/status") for b, u in cfg.stacks.items()]
    rows += [(f"llm {u}", f"{u}/models") for u in cfg.llm_endpoints]
    bad = 0
    async with httpx.AsyncClient(timeout=5) as c:
        for name, url in rows:
            try:
                key = cfg.key_for(url.removesuffix("/models")) if name.startswith("llm") else None
                r = await c.get(url, headers={"Authorization": f"Bearer {key}"} if key else None)
                extra = ""
                if name.startswith("llm") and r.status_code == 200:
                    extra = " models: " + ", ".join(m.get("id", "?") for m in (r.json().get("data") or [])[:3])
                print(f"  {'ok ' if r.status_code < 400 else 'BAD'} {name:<32} HTTP {r.status_code}{extra}")
                bad += r.status_code >= 400
            except httpx.HTTPError as e:
                print(f"  --  {name:<32} {e.__class__.__name__}")
                bad += 1
    return bad


def print_diagnosis(d, truth: str | None = None) -> None:
    print("\nEvidence trail:")
    for line in d.trail:
        print("  " + line)
    print(f"\nRanking ({d.policy}): " + "  >  ".join(d.ranking[:3]))
    if truth:
        print(f"Truth: {truth}  ({'correct' if d.top() == truth else 'top-3' if truth in d.ranking[:3] else 'missed'})")
    print(f"Cost: {d.tool_units} tool units, {d.llm_calls} LLM calls ({d.thinking_calls} thinking), "
          f"{d.tokens} tokens, {d.parse_failures} parse failures, {d.wall_s:.1f} s")


async def cmd_diagnose(a, cfg: wcfg.Config) -> None:
    if a.incident:
        inc = load_incident(Path(a.incident))
    else:
        rec = Recorder(cfg.router, cfg.boxes, cfg.stacks)
        print(f"Recording {a.pre + a.post:.0f} s ({a.pre:.0f} s before the alert, {a.post:.0f} s after) ...", flush=True)
        start = time.time()
        await rec.run(a.pre + a.post)
        inc = {"id": f"live-{time.strftime('%Y%m%d-%H%M%S', time.localtime(start))}", "truth": None,
               "alert": round(start + a.pre, 3), "started": round(start, 3), "where": "live", "buffers": rec.snapshot()}
        path = save_incident(ROOT / "data" / "incidents" / "live" / f"{inc['id']}.json", inc)
        print(f"saved {path.relative_to(ROOT)}")
    llm = None if a.policy == "rule" else LLMClient(cfg.llm_endpoints, cfg.model, cfg.node_cap, cfg.llm_api_key)
    try:
        d = await diagnose(inc["buffers"], inc["alert"], a.policy, llm, session=inc["id"])
    finally:
        if llm:
            await llm.aclose()
    print_diagnosis(d, inc.get("truth"))
    action = remediate.plan(d.top(), cfg)
    print(f"\nRemedy: {action.describe()}")
    if a.apply:
        if not a.live:
            print("(--apply only acts on a --live diagnosis; not applied)")
            return
        action = await remediate.execute(action, cfg, apply=True, settle_s=a.settle)
        print(f"Applied: {action.applied}  {action.result}")
    elif action.method:
        print("(dry run; add --apply to carry it out)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m whatif.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true")
    src.add_argument("--incident")
    d.add_argument("--policy", choices=POLICIES, default="whatif")
    d.add_argument("--apply", action="store_true")
    d.add_argument("--pre", type=float, default=30.0)
    d.add_argument("--post", type=float, default=30.0)
    d.add_argument("--settle", type=float, default=30.0)
    e = sub.add_parser("eval")
    e.add_argument("--incidents", default=str(ROOT / "data" / "incidents"))
    e.add_argument("--policies", default="single,equal,whatif,rule")
    e.add_argument("--out", default=None)
    e.add_argument("--parallel", type=int, default=2)
    e.add_argument("--limit", type=int, default=None)
    sub.add_parser("record", add_help=False)
    sub.add_parser("check")
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "record":
        return scenarios.main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    cfg = wcfg.load()
    if a.cmd == "check":
        sys.exit(1 if asyncio.run(check(cfg)) else 0)
    if a.cmd == "diagnose":
        asyncio.run(cmd_diagnose(a, cfg))
    elif a.cmd == "eval":
        incs = load_incidents(Path(a.incidents))[: a.limit]
        if not incs:
            raise SystemExit(f"no incidents in {a.incidents}")
        out = Path(a.out) if a.out else ROOT / "results" / "whatif" / time.strftime("%Y%m%d-%H%M%S")
        policies = [p.strip() for p in a.policies.split(",") if p.strip()]
        print(f"{len(incs)} incidents x {', '.join(policies)} -> {out}")
        asyncio.run(run_eval(incs, policies, out, cfg, a.parallel))
        print("\n" + (out / "summary.md").read_text())


if __name__ == "__main__":
    main()
