"""Prompts. The system prompt is the same for every agent of every policy, so llama-server's slot cache
reuses it; only the short user message changes per turn. Every reply is one JSON object."""
from __future__ import annotations

import json

from whatif.hypotheses import HYPOTHESES, ORDER, all_runbooks, runbook
from whatif.tools import COSTS, TOOL_DOCS

STACK = """The system: two GPU boxes (box-a, box-b) each run a weather forecast server. A router sends user
forecast requests to them (policy "adaptive": stay on one box, preferred box-b, switch when the other is clearly
faster; every policy retries a failed request once on the other box). An alert fired at t0. Tools compare the
30 s before t0 with the 30 s after it. Lines read: `metric: before -> after (+change%), onset +Ns, shape` where
onset is when it left its baseline relative to t0 (negative = before the alert) and shape is step (stayed),
blip (came back) or ramp (kept growing). Brief blips on a box can be unrelated noise."""

TOOLS_TEXT = "\n".join(f"- {doc} [cost {COSTS[name]}]" for name, doc in TOOL_DOCS.items())

SYSTEM = f"""You are an SRE agent diagnosing an incident on a small forecast service.

{STACK}

Tools (the team shares one budget; a tool already on the evidence board is free to read):
{TOOLS_TEXT}

Possible root causes and what each looks like:
{all_runbooks()}

Rules: reason only from the evidence board, not from the alert wording alone. Ask for at most one tool per
turn, the one that best separates your hypothesis from the others. Reply with ONE JSON object and nothing else."""

AGENT_FORMAT = """Reply format:
{"p": <0..1, probability that YOUR hypothesis is the root cause>,
 "next_tool": {"name": "<tool>", "box": "<box-a|box-b, for box tools>"} or null if no tool would change your mind,
 "suggest": [<other hypothesis ids the evidence points to, at most 2>],
 "reason": "<one short sentence citing evidence lines>"}"""

SINGLE_FORMAT = """Reply format:
{"ranking": [{"h": "<hypothesis id>", "p": <0..1>}, ... the 3 most likely, most likely first],
 "next_tool": {"name": "<tool>", "box": "<box-a|box-b, for box tools>"} or null when you are done,
 "reason": "<one short sentence citing evidence lines>"}"""

PLAN_FORMAT = """Reply format:
{"plan": [<exactly K hypothesis ids worth testing first, most likely first>], "reason": "<one sentence>"}"""


def _ids() -> str:
    return ", ".join(ORDER)


def plan_messages(alert: str, k: int) -> list[dict]:
    user = (f"ALERT at t0:\n{alert}\n\nNo tool has been run yet. Choose the {k} hypotheses most worth testing.\n"
            f"Hypothesis ids: {_ids()}\n\n" + PLAN_FORMAT.replace("K", str(k)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def agent_messages(alert: str, hypothesis: str, evidence: str, budget_left: int, last_p: float | None,
                   others: list[str]) -> list[dict]:
    user = (f"ALERT at t0:\n{alert}\n\nYOUR HYPOTHESIS: {runbook(hypothesis)}\n"
            f"Other agents are testing: {', '.join(others) or 'none'}\n"
            f"Budget left: {budget_left} units." + (f" Your previous p: {last_p:.2f}." if last_p is not None else "") +
            f"\n\nEVIDENCE BOARD:\n{evidence}\n\n{AGENT_FORMAT}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def single_messages(alert: str, evidence: str, budget_left: int) -> list[dict]:
    user = (f"ALERT at t0:\n{alert}\n\nYou test ALL hypotheses yourself. Hypothesis ids: {_ids()}\n"
            f"Budget left: {budget_left} units.\n\nEVIDENCE BOARD:\n{evidence}\n\n{SINGLE_FORMAT}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


JUDGE_FORMAT = """Reply format:
{"ranking": [{"h": "<hypothesis id>", "p": <0..1>}, ... every hypothesis listed above, most likely first],
 "reason": "<one short sentence citing the deciding evidence line>"}"""

JSON_ONLY = "\n\nReply with ONE JSON object only, in the format above: no thinking, no prose, no code fence."


def judge_messages(alert: str, evidence: str, candidates: list[tuple[str, float | None, str]]) -> list[dict]:
    """The final judgement over every branch the agents explored, pruned ones included."""
    lines = "\n".join(f"- {h} (agent's last p={'n/a' if p is None else f'{p:.2f}'}): {why or 'no reason given'}"
                      for h, p, why in candidates)
    user = (f"ALERT at t0:\n{alert}\n\nFINAL JUDGEMENT. The agents explored these hypotheses:\n{lines}\n\n"
            "An agent sometimes rates its own hypothesis high on evidence that fits a different one better (the "
            "other box, or another kind of fault): judge from the evidence lines and the runbooks, not from the "
            f"agents' p.\n\nEVIDENCE BOARD:\n{evidence}\n\nGive each hypothesis above the probability that it "
            "is the root cause.\n" + JUDGE_FORMAT)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def hypothesis_ids() -> list[str]:
    return list(HYPOTHESES)
