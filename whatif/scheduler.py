"""The scheduler: who gets to think next. Four policies over the same tools, budget and agents:

  whatif   a planner picks K_PLAN hypotheses; one agent each (MAX_LIVE at most). Each round every live agent
           reads the shared evidence board and answers {p, next_tool, suggest}. The scheduler runs the tool
           requests of the most promising agents first (and gives them LLM priority), prunes agents whose p fell
           below PRUNE_P, spawns agents for hypotheses the evidence points to, and stops once one hypothesis is
           clearly ahead. Two leaders it cannot separate go to one judge call with thinking on.
  equal    the same planner and agents, but every agent runs every round with equal priority and tool requests
           are served in turn: no pruning, no spawning, no early stop until nobody asks for anything new.
  single   one agent holds every hypothesis and runs the tools itself, one per turn.
  rule     no LLM: fixed thresholds over the window (see rule_diagnose).

All policies return a Diagnosis with a full ranking, so AC@k is defined for each.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from whatif import prompts
from whatif.hypotheses import ORDER, normalize
from whatif.llm import LLMClient, LLMError, extract_json
from whatif.tools import BOX_TOOLS, BUDGET, COSTS, PRE_S, POST_S, STALL_MS, Toolbox, _probes, traffic

K_PLAN, MAX_LIVE, MAX_ROUNDS = 4, 5, 10
PRUNE_P, STOP_P, STOP_GAP, TIE_GAP = 0.15, 0.8, 0.4, 0.15
POLICIES = ("whatif", "equal", "single", "rule")


@dataclass
class Agent:
    h: str
    p: Optional[float] = None
    reason: str = ""
    alive: bool = True
    turns: int = 0
    parse_failures: int = 0
    want: Optional[dict] = None
    suggest: list = field(default_factory=list)


@dataclass
class Diagnosis:
    policy: str
    ranking: list[str]
    scores: dict[str, float]
    trail: list[str]
    tool_units: int = 0
    rounds: int = 0
    llm_calls: int = 0
    thinking_calls: int = 0
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parse_failures: int = 0
    llm_errors: int = 0
    wall_s: float = 0.0

    def top(self) -> str:
        return self.ranking[0] if self.ranking else ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("policy", "ranking", "scores", "trail", "tool_units", "rounds",
                                              "llm_calls", "thinking_calls", "tokens", "prompt_tokens",
                                              "completion_tokens", "parse_failures", "llm_errors", "wall_s")}


def alert_text(samples: list[dict], t0: float) -> str:
    """What the pager says: the user-facing numbers that tripped it. Same for every policy; free."""
    lines = traffic(samples, t0).splitlines()[:3]
    return "wxrouter user-facing forecasts look unhealthy.\n" + "\n".join(lines)


def _tool_req(raw) -> Optional[tuple[str, dict]]:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name") or raw.get("tool")
    if not isinstance(name, str):
        return None
    name = name.strip().split("(")[0]
    args = {}
    if name in BOX_TOOLS:
        box = raw.get("box") or (raw.get("args") or {}).get("box")
        args["box"] = str(box).strip().lower().replace("_", "-") if box else None
    return name, args


def _p(raw) -> Optional[float]:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, v / 100 if 1 < v <= 100 else v))


def _rank(scores: dict[str, float], prior: list[str]) -> list[str]:
    pri = {h: i for i, h in enumerate(prior + [h for h in ORDER if h not in prior])}
    return sorted(ORDER, key=lambda h: (-scores.get(h, -1.0), pri[h]))


class Run:
    """One incident under one LLM policy."""

    def __init__(self, samples: list[dict], t0: float, llm: LLMClient, session: str, policy: str,
                 budget: int = BUDGET, k_plan: int = K_PLAN, max_live: int = MAX_LIVE, max_rounds: int = MAX_ROUNDS,
                 judge_thinking: bool = True):
        self.tb = Toolbox(samples, t0, budget=budget)
        self.alert = alert_text(samples, t0)
        self.llm, self.session, self.policy = llm, session, policy
        self.k_plan, self.max_live, self.max_rounds = k_plan, max_live, max_rounds
        self.judge_thinking = judge_thinking
        self.trail: list[str] = []
        self.parse_failures = 0
        self.rounds = 0

    def note(self, s: str) -> None:
        self.trail.append(s)

    async def ask(self, messages: list[dict], who: str, priority: int = 5, thinking: bool = False) -> Optional[dict]:
        try:
            r = await self.llm.chat(messages, self.session, priority=priority, thinking=thinking, who=who)
        except LLMError as e:
            self.note(f"[{who}] LLM failed: {e}")
            return None
        try:
            obj = extract_json(r.text)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            return obj
        except ValueError as e:
            self.parse_failures += 1
            self.note(f"[{who}] unparseable reply: {str(e)[:100]}")
            return None

    # ------------------------------------------------------------ planning
    async def plan(self) -> list[str]:
        obj = await self.ask(prompts.plan_messages(self.alert, self.k_plan), "planner", priority=0)
        got = []
        for h in (obj or {}).get("plan") or []:
            h = normalize(h)
            if h and h not in got:
                got.append(h)
        for h in ORDER:  # top up with the fixed order when the planner gave too few
            if len(got) >= self.k_plan:
                break
            if h not in got:
                got.append(h)
        self.note(f"[planner] test first: {', '.join(got[:self.k_plan])}" + (f" ({obj.get('reason')})" if obj and obj.get("reason") else ""))
        return got[:self.k_plan]

    # ------------------------------------------------------------ one agent turn
    async def step(self, a: Agent, live: list[Agent], priority: int) -> None:
        others = [x.h for x in live if x is not a and x.alive]
        obj = await self.ask(prompts.agent_messages(self.alert, a.h, self.tb.evidence(), self.tb.left(), a.p, others),
                             f"agent {a.h}", priority=priority)
        a.turns += 1
        a.want, a.suggest = None, []
        if obj is None:
            a.parse_failures += 1
            return
        p = _p(obj.get("p", obj.get("confidence")))
        if p is not None:
            a.p = p
        a.reason = str(obj.get("reason") or "")[:200]
        a.want = _tool_req(obj.get("next_tool"))
        a.suggest = [h for h in (normalize(s) for s in (obj.get("suggest") or [])[:2]) if h and h != a.h]
        self.note(f"[agent {a.h}] p={a.p if a.p is not None else float('nan'):.2f}"
                  + (f" wants {self.tb.key(*a.want)}" if a.want else "") + (f" | {a.reason}" if a.reason else ""))

    def serve(self, requests: list[tuple[Agent, tuple[str, dict]]]) -> int:
        """Run tool requests in the given order. Returns how many were new (not already on the board)."""
        new = 0
        for a, (name, args) in requests:
            r = self.tb.call(name, args, who=a.h)
            if r.cached:
                continue
            if r.error:
                self.note(f"[tool] {a.h}: {r.text}")
                continue
            new += 1
            self.note(f"[tool] {r.key} (cost {r.cost}, {self.tb.left()} left) for {a.h}:\n    " + r.text.replace("\n", "\n    "))
        return new

    # ------------------------------------------------------------ policies
    async def run_agents(self) -> dict[str, float]:
        whatif = self.policy == "whatif"
        plan = await self.plan()
        live = [Agent(h) for h in plan]
        queue = [h for h in ORDER if h not in plan]
        all_agents = list(live)
        for rnd in range(1, self.max_rounds + 1):
            self.rounds = rnd
            alive = [a for a in live if a.alive]
            if not alive:
                break
            if whatif:  # most promising first: lower priority number = served first by the gate
                alive.sort(key=lambda a: -(a.p if a.p is not None else 0.5))
            await asyncio.gather(*(self.step(a, alive, priority=(i if whatif else 5)) for i, a in enumerate(alive)))
            reqs = [(a, a.want) for a in alive if a.want]
            if whatif:
                reqs.sort(key=lambda r: -(r[0].p or 0.0))
            else:  # equal: in turn, starting with a different agent each round
                k = (rnd - 1) % max(1, len(reqs)) if reqs else 0
                reqs = reqs[k:] + reqs[:k]
            new = self.serve(reqs)

            if whatif:
                for a in alive:
                    if a.p is not None and a.p < PRUNE_P and a.turns >= 1 and self.tb.board:
                        a.alive = False
                        self.note(f"[scheduler] prune {a.h} (p={a.p:.2f})")
                suggested = [h for a in sorted(alive, key=lambda a: -(a.p or 0)) for h in a.suggest]
                for h in suggested + queue:
                    if sum(x.alive for x in live) >= self.max_live:
                        break
                    if h in [x.h for x in all_agents]:
                        continue
                    if h not in suggested and not any(x.p is not None and x.p < PRUNE_P for x in all_agents):
                        break  # only fill from the queue when something was pruned
                    ag = Agent(h)
                    live.append(ag); all_agents.append(ag)
                    if h in queue:
                        queue.remove(h)
                    self.note(f"[scheduler] spawn {h}" + (" (suggested)" if h in suggested else ""))
                ps = sorted((a.p for a in live if a.alive and a.p is not None), reverse=True)
                if self.tb.board and ps and ps[0] >= STOP_P and (len(ps) == 1 or ps[1] <= ps[0] - STOP_GAP):
                    self.note(f"[scheduler] stop: a clear leader (p={ps[0]:.2f})")
                    break
            # Both stops need the board unchanged this round, so every live agent has already read it as it is.
            fresh = any(a.alive and a.turns == 0 for a in live)  # a spawned agent still gets its first turn
            if new == 0 and not fresh:
                pending = [a for a in alive if a.alive and a.want and self.tb.key(*a.want) not in self.tb.board]
                if not pending:
                    self.note("[scheduler] stop: no new evidence requested")
                    break
                if self.tb.left() < min(COSTS.values()) or all(COSTS[a.want[0]] > self.tb.left() for a in pending if a.want[0] in COSTS):
                    self.note("[scheduler] stop: budget spent")
                    break

        scores = {a.h: a.p for a in all_agents if a.p is not None}
        if whatif:
            top = sorted(scores.items(), key=lambda kv: -kv[1])
            if len(top) >= 2 and top[0][1] - top[1][1] < TIE_GAP:
                cands = [(h, p, next(a.reason for a in all_agents if a.h == h)) for h, p in top[:3]]
                obj = await self.ask(prompts.judge_messages(self.alert, self.tb.evidence(), cands), "judge",
                                     priority=0, thinking=self.judge_thinking)
                ranked = [normalize((x or {}).get("h")) for x in (obj or {}).get("ranking") or [] if isinstance(x, dict)]
                ranked = [h for h in ranked if h]
                if ranked:
                    self.note(f"[judge] {' > '.join(ranked)}")
                    base = max(scores.values()) + 0.01
                    for i, h in enumerate(ranked):
                        scores[h] = base + (len(ranked) - i) * 0.01
        self._prior = plan
        return scores

    async def run_single(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rnd in range(1, self.max_rounds + 1):
            self.rounds = rnd
            obj = await self.ask(prompts.single_messages(self.alert, self.tb.evidence(), self.tb.left()), "single")
            if obj is None:
                continue
            rk = {}
            for x in obj.get("ranking") or []:
                if isinstance(x, dict):
                    h, p = normalize(x.get("h")), _p(x.get("p"))
                    if h and h not in rk:
                        rk[h] = p if p is not None else 0.5 - 0.01 * len(rk)
            if rk:
                scores = rk
            want = _tool_req(obj.get("next_tool"))
            self.note(f"[single] {', '.join(f'{h}={p:.2f}' for h, p in list(scores.items())[:3])}"
                      + (f" wants {self.tb.key(*want)}" if want else " done"))
            if not want:
                break
            if self.serve([(Agent("single"), want)]) == 0:
                # Nothing new on the board (already there, unknown tool, or no budget): the ranking just given
                # was made from the same evidence, so asking again would not change it.
                self.note("[scheduler] stop: " + ("budget spent" if self.tb.left() < COSTS.get(want[0], 1) else "no new evidence"))
                break
        self._prior = list(scores)
        return scores


async def diagnose(samples: list[dict], t0: float, policy: str, llm: Optional[LLMClient] = None,
                   session: str = "incident", **kw) -> Diagnosis:
    t_start = time.perf_counter()
    if policy == "rule":
        d = rule_diagnose(samples, t0)
        d.wall_s = round(time.perf_counter() - t_start, 3)
        return d
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {', '.join(POLICIES)}")
    if llm is None:
        raise ValueError(f"policy {policy} needs an LLM client")
    run = Run(samples, t0, llm, session, policy, **kw)
    run._prior = []
    scores = await (run.run_single() if policy == "single" else run.run_agents())
    st = llm.stats.get(session)
    d = Diagnosis(policy=policy, ranking=_rank(scores, run._prior), scores={h: round(p, 3) for h, p in scores.items()},
                  trail=run.trail, tool_units=run.tb.spent, rounds=run.rounds, parse_failures=run.parse_failures,
                  wall_s=round(time.perf_counter() - t_start, 3))
    if st:
        d.llm_calls, d.thinking_calls, d.tokens = st.calls, st.thinking_calls, st.tokens
        d.prompt_tokens, d.completion_tokens, d.llm_errors = st.prompt_tokens, st.completion_tokens, st.errors
    return d


# -------------------------------------------------------------------- the deterministic baseline
def rule_diagnose(samples: list[dict], t0: float) -> Diagnosis:
    """health no-answer -> down; probe stalls -> hang; 503s -> error; one box compute >1.5x the other -> slow;
    rps >1.5x -> overload; else misconfig. Each check names the worse box. Fired rules rank in that order."""
    post = [s for s in samples if t0 <= s["t"] < t0 + POST_S]
    pre = [s for s in samples if t0 - PRE_S <= s["t"] < t0]
    boxes = ("box-a", "box-b")
    trail, fired = [], []

    def hcodes(box):
        return [((s.get("boxes") or {}).get(box) or {}).get("health", {}).get("code") for s in post]

    noans = {b: sum(1 for c in hcodes(b) if c == 0) / max(1, len(post)) for b in boxes}
    b = max(boxes, key=noans.get)
    if noans[b] > 0.3:
        fired.append(f"{b}:down"); trail.append(f"[rule] {b} health unanswered {noans[b]:.0%} of samples -> down")

    stalls, errs, comp = {}, {}, {}
    for bx in boxes:
        ps = [(t, p) for t, p in _probes(samples, bx) if t0 <= t < t0 + POST_S]
        stalls[bx] = sum(1 for t, p in ps if p.get("pending") or p.get("timeout") or (p.get("total_ms") or 0) > STALL_MS)
        h503 = sum(1 for c in hcodes(bx) if c == 503)
        errs[bx] = sum(1 for t, p in ps if p.get("code") == 503) + h503
        c = [p["compute_ms"] for t, p in ps if p.get("compute_ms") is not None]
        comp[bx] = statistics.median(c) if c else None
    b = max(boxes, key=stalls.get)
    if stalls[b] >= 2:
        fired.append(f"{b}:hang"); trail.append(f"[rule] {b} probe stalls {stalls[b]} -> hang")
    b = max(boxes, key=errs.get)
    if errs[b] >= 2:
        fired.append(f"{b}:error"); trail.append(f"[rule] {b} 503s {errs[b]} -> error")
    if comp["box-a"] and comp["box-b"]:
        b, o = ("box-a", "box-b") if comp["box-a"] > comp["box-b"] else ("box-b", "box-a")
        if comp[b] > 1.5 * comp[o]:
            fired.append(f"{b}:slow"); trail.append(f"[rule] {b} compute {comp[b]:.0f} vs {comp[o]:.0f} ms -> slow")
    rps = lambda ss: statistics.mean([s["traffic"]["rps"] for s in ss if s.get("traffic") and s["traffic"].get("rps") is not None] or [0])
    r0, r1 = rps(pre), rps(post)
    if r0 > 0 and r1 > 1.5 * r0:
        fired.append("load:overload"); trail.append(f"[rule] rps {r0:.1f} -> {r1:.1f} -> overload")
    if not fired:
        fired.append("router:misconfig"); trail.append("[rule] nothing else fired -> misconfig")
    ranking = list(dict.fromkeys(fired + [h for h in ORDER if h not in fired]))
    scores = {h: round(1.0 - 0.1 * i, 2) for i, h in enumerate(fired)}
    return Diagnosis(policy="rule", ranking=ranking, scores=scores, trail=trail)
