"""Run every policy on every saved incident and record what each one concluded and what it cost.

Policies are interleaved: incident i runs them in an order rotated by i, so a slow drift in the LLM servers
(warm caches, a teammate's load) does not favour one policy. With repeats > 1 every LLM policy runs that many
times per incident (rows carry `rep`); the rule is deterministic and runs once. Up to `parallel` incidents run at once; inside an
incident the policies run one after another, each with its own session id `<incident>/<policy>`.

Out folder: results.jsonl (one row per incident x policy), calls.jsonl (every LLM call), trails/ (the evidence
trail of each run), config.json, and the summary from whatif/summarize.py.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from whatif import summarize
from whatif.config import Config
from whatif.llm import LLMClient
from whatif.scheduler import POLICIES, diagnose


def score_row(inc: dict, policy: str, d) -> dict:
    truth = inc["truth"]
    return {"incident": inc["id"], "truth": truth, "kind": truth.split(":")[1], "policy": policy,
            "top": d.top(), "ranking": d.ranking[:5], "ac1": int(d.ranking[:1] == [truth]),
            "ac3": int(truth in d.ranking[:3]), "noise": bool((inc.get("fault") or {}).get("noise")),
            **{k: v for k, v in d.as_dict().items() if k not in ("ranking", "trail", "policy")}}


async def evaluate(incidents: list[dict], policies: list[str], out: Path, llm: LLMClient | None,
                   parallel: int = 2, progress=print, repeats: int = 1) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "trails").mkdir(exist_ok=True)
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        raise ValueError(f"unknown policies {bad}")
    if llm is None and any(p != "rule" for p in policies):
        raise ValueError("LLM policies need an LLM client")
    rows: list[dict] = []
    sem = asyncio.Semaphore(parallel)
    res_path = out / "results.jsonl"

    async def one(i: int, inc: dict):
        async with sem:
            k = i % len(policies)
            runs = [(r, p) for r in range(repeats) for p in policies[k:] + policies[:k] if r == 0 or p != "rule"]
            for rep, policy in runs:
                t = time.perf_counter()
                try:
                    d = await diagnose(inc["buffers"], inc["alert"], policy, llm, session=f"{inc['id']}/{policy}/{rep}")
                    row = {**score_row(inc, policy, d), "rep": rep}
                    trail = "\n".join(d.trail)
                except Exception as e:  # keep going: a failed run is a row with the error
                    row = {"incident": inc["id"], "truth": inc["truth"], "kind": inc["truth"].split(":")[1],
                           "policy": policy, "top": None, "ranking": [], "ac1": 0, "ac3": 0,
                           "rep": rep, "error": f"{e.__class__.__name__}: {e}", "wall_s": round(time.perf_counter() - t, 2)}
                    trail = row["error"]
                rows.append(row)
                with res_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                (out / "trails" / f"{inc['id']}__{policy}{'' if rep == 0 else f'__r{rep}'}.txt").write_text(
                    f"truth: {inc['truth']}\npolicy: {policy}\nranking: {row.get('ranking')}\n\n{trail}\n")
                progress(f"  {inc['id'][:40]:40} {policy:7} r{rep} -> {str(row.get('top')):18} "
                         f"{'OK ' if row['ac1'] else 'miss'} truth={inc['truth']:18} "
                         f"tok={row.get('tokens', 0):>6} calls={row.get('llm_calls', 0):>3} "
                         f"units={row.get('tool_units', 0)} {row.get('wall_s', 0):.1f}s" + (f" ERROR {row['error']}" if row.get("error") else ""))

    await asyncio.gather(*(one(i, inc) for i, inc in enumerate(incidents)))
    return rows


async def run_eval(incidents: list[dict], policies: list[str], out: Path, cfg: Config, parallel: int = 2,
                   repeats: int = 1, split: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"config": cfg.redacted(), "policies": policies, "repeats": repeats,
                                                 "split": split, "incidents": [i["id"] for i in incidents]}, indent=2) + "\n")
    llm = None
    if any(p != "rule" for p in policies):
        llm = LLMClient(cfg.llm_endpoints, cfg.model, cfg.node_cap, cfg.llm_api_key, log_path=out / "calls.jsonl")
    try:
        rows = await evaluate(incidents, policies, out, llm, parallel, repeats=repeats)
    finally:
        if llm:
            await llm.aclose()
    return summarize.write(rows, out, policies, summarize.engine_label(cfg.redacted()))


def load_split(splits_path: Path, name: str) -> list[str]:
    """Incident ids of one split ("dev" or "test") from data/splits.json."""
    splits = json.loads(Path(splits_path).read_text())
    if name not in splits:
        raise SystemExit(f"no split {name!r} in {splits_path}; have: {', '.join(k for k in splits if not k.startswith('_'))}")
    return list(splits[name])
