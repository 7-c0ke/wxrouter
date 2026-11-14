"""The scheduler: who gets to think next. Four policies over the same tools, budget and agents:

  whatif   a planner picks K_PLAN hypotheses; one agent each (MAX_LIVE at most). Each round every live agent
           reads the shared evidence board and answers {p, next_tool, suggest}. The scheduler runs the tool
           requests of the most promising agents first (and gives them LLM priority), prunes agents whose p fell
           below PRUNE_P, spawns agents for hypotheses the evidence points to, and stops once one hypothesis is
           clearly ahead. Two leaders it cannot separate go to one judge call with thinking on.
  equal    the same planner and agents, but every agent runs every round with equal priority and tool requests
           are served in turn: no pruning, no spawning, no early stop until nobody asks for anything new.
  single   one agent holds every hypothesis and runs the tools itself, one per turn.
  triage   rule first: when the rule's signal is unambiguous (see rule_signals) answer with no LLM at all;
           otherwise whatif seeded with the rule's top 3 (no planner call) on a smaller budget.
  rule     no LLM: fixed thresholds over the window (see rule_diagnose).

whatif and triage end with a judge over every branch the agents explored. The final ranking is the judge's
beliefs; if the judge gives nothing usable (after one retry), it is the rule's ranking on the same recorded
window. Ties always break by the rule's order, never by the order an LLM happened to list things.

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

K_PLAN, MAX_LIVE, MAX_SPAWN, MAX_ROUNDS = 3, 5, 3, 10
PRUNE_P = 0.15
STOP_P, STOP_GAP, STOP_ROUND = 0.6, 0.3, 2  # early stop: leader >= STOP_P, ahead by STOP_GAP, from round 2
TRIAGE_BUDGET = 6
JUDGE_RETRY_TOKENS = 1200
# Dev split: with thinking on, 4 of 5 judge calls used up their tokens thinking and returned no answer; the
# retry without thinking answered every time. So the judge does not think unless asked to.
JUDGE_THINKING = False
RPS_SURGE = RPS_FLAT = 1.5  # rps after/before: overload at or above; "flat" (no surge) below. Dev: faults x0.96-1.38, overload x2.4+
POLICIES = ("whatif", "equal", "single", "triage", "rule")
BELIEF_ID_KEYS, BELIEF_P_KEYS = ("h", "id", "hypothesis"), ("p", "prob", "probability", "confidence")


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
    return "wxrouter user-facing forecasts look unhealthy.\n" + traffic(samples, t0, alert_only=True)


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


def _rank(scores: dict[str, float], tiebreak: list[str]) -> list[str]:
    """Highest score first; equal scores (and everything unscored) in the rule's order."""
    pos = {h: i for i, h in enumerate(list(tiebreak) + [h for h in ORDER if h not in tiebreak])}
    return sorted(ORDER, key=lambda h: (-scores.get(h, -1.0), pos[h]))


class Run:
    """One incident under one LLM policy."""

    def __init__(self, samples: list[dict], t0: float, llm: LLMClient, session: str, policy: str,
                 budget: int = BUDGET, k_plan: int = K_PLAN, max_live: int = MAX_LIVE, max_rounds: int = MAX_ROUNDS,
                 max_spawn: int = MAX_SPAWN, judge_thinking: bool = JUDGE_THINKING):
        self.tb = Toolbox(samples, t0, budget=budget)
        self.signals = rule_signals(samples, t0)
        self.rule = rule_diagnose(samples, t0, self.signals)  # tie-breaks, and the fallback ranking
        self.alert = alert_text(samples, t0)
        self.llm, self.session, self.policy = llm, session, policy
        self.k_plan, self.max_live, self.max_rounds, self.max_spawn = k_plan, max_live, max_rounds, max_spawn
        self.judge_thinking = judge_thinking
        self.trail: list[str] = []
        self.parse_failures = 0
        self.rounds = 0

    def note(self, s: str) -> None:
        self.trail.append(s)

    async def ask(self, messages: list[dict], who: str, priority: int = 5, thinking: bool = False,
                  max_tokens: int = 600) -> tuple[Optional[dict], bool]:
        """One call. Returns (object or None, whether the reply came back but did not parse)."""
        try:
            r = await self.llm.chat(messages, self.session, priority=priority, thinking=thinking,
                                    max_tokens=max_tokens, who=who)
        except LLMError as e:
            self.note(f"[{who}] LLM failed: {e}")
            return None, False
        try:
            obj = extract_json(r.text)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            return obj, False
        except ValueError as e:
            self.parse_failures += 1
            self.note(f"[{who}] unparseable reply: {str(e)[:100]}")
            return None, True

    async def ask_obj(self, messages: list[dict], who: str, priority: int = 5, thinking: bool = False) -> Optional[dict]:
        return (await self.ask(messages, who, priority, thinking))[0]

    def ids(self, items, who: str) -> list[str]:
        """Hypothesis ids in any spelling -> canonical ids, in order, without repeats. Unknown ones are logged."""
        out, unknown = [], []
        for x in items if isinstance(items, list) else []:
            h = normalize(x)
            if h is None:
                unknown.append(x)
            elif h not in out:
                out.append(h)
        if unknown:
            self.note(f"[{who}] unknown hypothesis ids ignored: {', '.join(repr(u) for u in unknown[:5])}")
        return out

    def beliefs(self, obj: Optional[dict], who: str) -> dict[str, float]:
        """{"ranking": [{"h": id, "p": 0..1}, ...]} (or a plain {id: p} map, or bare ids) -> {canonical id: p}.
        A missing p gets a value below the ones before it. Unknown ids and unexpected keys are logged."""
        items = (obj or {}).get("ranking", (obj or {}).get("beliefs")) or []
        if isinstance(items, dict):
            items = [{"h": k, "p": v} for k, v in items.items()]
        out: dict[str, float] = {}
        unknown, extra = [], set()
        for i, x in enumerate(items if isinstance(items, list) else []):
            if isinstance(x, dict):
                raw = next((x[k] for k in BELIEF_ID_KEYS if k in x), None)
                p = _p(next((x[k] for k in BELIEF_P_KEYS if k in x), None))
                extra |= set(x) - set(BELIEF_ID_KEYS) - set(BELIEF_P_KEYS) - {"reason", "why"}
            else:
                raw, p = x, None
            h = normalize(raw)
            if h is None:
                unknown.append(raw)
            elif h not in out:
                out[h] = p if p is not None else max(0.0, 0.5 - 0.05 * i)
        if unknown:
            self.note(f"[{who}] unknown hypothesis ids ignored: {', '.join(repr(u) for u in unknown[:5])}")
        if extra:
            self.note(f"[{who}] unexpected keys ignored: {', '.join(sorted(map(str, extra))[:5])}")
        return out

    # ------------------------------------------------------------ planning
    async def plan(self) -> list[str]:
        obj = await self.ask_obj(prompts.plan_messages(self.alert, self.k_plan), "planner", priority=0)
        got = self.ids((obj or {}).get("plan") or [], "planner")
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
        obj = await self.ask_obj(prompts.agent_messages(self.alert, a.h, self.tb.evidence(), self.tb.left(), a.p, others),
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
        a.suggest = [h for h in self.ids((obj.get("suggest") or [])[:2], f"agent {a.h}") if h != a.h]
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
    async def run_agents(self, seeds: Optional[list[str]] = None) -> dict[str, float]:
        whatif = self.policy in ("whatif", "triage")
        plan = list(seeds) if seeds is not None else await self.plan()
        live = [Agent(h) for h in plan]
        queue = [h for h in ORDER if h not in plan]
        all_agents = list(live)
        spawned = 0
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
                    if sum(x.alive for x in live) >= self.max_live or spawned >= self.max_spawn:
                        break
                    if h in [x.h for x in all_agents]:
                        continue
                    if h not in suggested and not any(x.p is not None and x.p < PRUNE_P for x in all_agents):
                        break  # only fill from the queue when something was pruned
                    ag = Agent(h)
                    live.append(ag); all_agents.append(ag)
                    spawned += 1
                    if h in queue:
                        queue.remove(h)
                    self.note(f"[scheduler] spawn {h}" + (" (suggested)" if h in suggested else ""))
                ps = sorted((a.p for a in live if a.alive and a.p is not None), reverse=True)
                if (rnd >= STOP_ROUND and self.tb.board and ps and ps[0] >= STOP_P
                        and (len(ps) == 1 or ps[1] <= ps[0] - STOP_GAP)):
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
            return await self.judge(all_agents)
        return scores

    async def judge(self, agents: list[Agent]) -> dict[str, float]:
        """One judge call over every branch. On an unparseable or empty answer, one retry with thinking off, more
        room and a JSON-only instruction. The last answer that gave usable beliefs is the ranking; with none,
        the rule's ranking on the recorded window."""
        cands = [(a.h, a.p, a.reason) for a in sorted(agents, key=lambda a: -(a.p if a.p is not None else -1))]
        msgs = prompts.judge_messages(self.alert, self.tb.evidence(), cands)
        obj, bad = await self.ask(msgs, "judge", priority=0, thinking=self.judge_thinking)
        got = self.beliefs(obj, "judge")
        if not got and (bad or obj is not None):
            self.note("[judge] retry: thinking off, JSON only")
            retry = msgs[:-1] + [{"role": "user", "content": msgs[-1]["content"] + prompts.JSON_ONLY}]
            obj, _ = await self.ask(retry, "judge retry", priority=0, thinking=False, max_tokens=JUDGE_RETRY_TOKENS)
            got = self.beliefs(obj, "judge retry")
        if got:
            self.note("[judge] " + ", ".join(f"{h}={p:.2f}" for h, p in sorted(got.items(), key=lambda kv: -kv[1])))
            return got
        self.note("[judge] no usable answer: ranking by the rule on the recorded window")
        return {}

    async def run_single(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rnd in range(1, self.max_rounds + 1):
            self.rounds = rnd
            obj = await self.ask_obj(prompts.single_messages(self.alert, self.tb.evidence(), self.tb.left()), "single")
            if obj is None:
                continue
            rk = self.beliefs(obj, "single")
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
        return scores

    async def run_triage(self) -> dict[str, float]:
        sig = self.signals
        if sig["unambiguous"]:
            self.note(f"[triage] unambiguous ({sig['why']}): {self.rule.top()}, no LLM")
            self.trail += self.rule.trail
            return dict(self.rule.scores)
        seeds = self.rule.ranking[:3]
        self.note(f"[triage] not clear-cut ({sig['why']}): agents seeded with the rule's top 3: {', '.join(seeds)}")
        return await self.run_agents(seeds=seeds)


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
    if policy == "triage":
        kw.setdefault("budget", TRIAGE_BUDGET)
    run = Run(samples, t0, llm, session, policy, **kw)
    scores = await (run.run_single() if policy == "single" else run.run_triage() if policy == "triage"
                    else run.run_agents())
    st = llm.stats.get(session)
    d = Diagnosis(policy=policy, ranking=_rank(scores, run.rule.ranking), scores={h: round(p, 3) for h, p in scores.items()},
                  trail=run.trail, tool_units=run.tb.spent, rounds=run.rounds, parse_failures=run.parse_failures,
                  wall_s=round(time.perf_counter() - t_start, 3))
    if st:
        d.llm_calls, d.thinking_calls, d.tokens = st.calls, st.thinking_calls, st.tokens
        d.prompt_tokens, d.completion_tokens, d.llm_errors = st.prompt_tokens, st.completion_tokens, st.errors
    return d


# -------------------------------------------------------------------- the deterministic baseline
def _config_change(samples: list[dict], t0: float) -> Optional[str]:
    """The first router policy/preferred change in [t0-10 s, t0+POST_S), as `key: before -> after at +Ns`."""
    rs = [(s["t"], s["router"]) for s in samples if s.get("router")]
    for (t, r), (_, q) in zip(rs, rs[1:]):
        if t0 - 10 <= t < t0 + POST_S:
            for key in ("policy", "preferred"):
                if r.get(key) != q.get(key):
                    return f"router.{key} {r.get(key)} -> {q.get(key)} at {t - t0:+.0f}s"
    return None


def rule_signals(samples: list[dict], t0: float) -> dict:
    """The rule's checks, in priority order: health no-answer -> down; a router policy/preferred change at t0 ->
    misconfig; rps x1.5 or more -> overload; probe stalls -> hang; 503s -> error; one box's compute > 1.5x the
    other's -> slow; nothing -> misconfig. Config and load come before stalls because both make a healthy box
    queue and stall. Also says whether the answer is unambiguous enough to need no LLM (triage)."""
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

    config = _config_change(samples, t0)
    if config:
        fired.append("router:misconfig"); trail.append(f"[rule] {config} -> misconfig")

    rps = lambda ss: statistics.mean([s["traffic"]["rps"] for s in ss if s.get("traffic") and s["traffic"].get("rps") is not None] or [0])
    r0, r1 = rps(pre), rps(post)
    surge = r0 > 0 and r1 >= RPS_SURGE * r0
    rps_flat = r1 < RPS_FLAT * r0 if r0 > 0 else r1 == 0
    if surge:
        fired.append("load:overload"); trail.append(f"[rule] rps {r0:.1f} -> {r1:.1f} -> overload")

    stalls, errs, comp, comp_pre = {}, {}, {}, {}
    for bx in boxes:
        allp = _probes(samples, bx)
        ps = [(t, p) for t, p in allp if t0 <= t < t0 + POST_S]
        stalls[bx] = sum(1 for t, p in ps if p.get("pending") or p.get("timeout") or (p.get("total_ms") or 0) > STALL_MS)
        h503 = sum(1 for c in hcodes(bx) if c == 503)
        errs[bx] = sum(1 for t, p in ps if p.get("code") == 503) + h503
        c = [p["compute_ms"] for t, p in ps if p.get("compute_ms") is not None]
        comp[bx] = statistics.median(c) if c else None
        c0 = [p["compute_ms"] for t, p in allp if t0 - PRE_S <= t < t0 and p.get("compute_ms") is not None]
        comp_pre[bx] = statistics.median(c0) if c0 else None
    b = max(boxes, key=stalls.get)
    if stalls[b] >= 2:
        fired.append(f"{b}:hang"); trail.append(f"[rule] {b} probe stalls {stalls[b]} -> hang")
    b = max(boxes, key=errs.get)
    if errs[b] >= 2:
        fired.append(f"{b}:error"); trail.append(f"[rule] {b} 503s {errs[b]} -> error")
    slow_new = False
    if comp["box-a"] and comp["box-b"]:
        b, o = ("box-a", "box-b") if comp["box-a"] > comp["box-b"] else ("box-b", "box-a")
        if comp[b] > 1.5 * comp[o]:
            fired.append(f"{b}:slow"); trail.append(f"[rule] {b} compute {comp[b]:.0f} vs {comp[o]:.0f} ms -> slow")
            # new at t0, not a box that was already slow (that is misconfig's signature)
            slow_new = not (comp_pre[b] and comp_pre[o] and comp_pre[b] > 1.5 * comp_pre[o])
    if not fired:
        fired.append("router:misconfig"); trail.append("[rule] nothing else fired -> misconfig")

    kind = fired[0].split(":")[1]
    # Clear-cut: health fails, 503s, stalls with flat rps, or a new one-box slowdown with flat rps; and no config
    # change, which would make any of those a consequence of where the router sends traffic.
    clear = {"down": (True, "health unanswered"), "error": (True, "503s"),
             "hang": (rps_flat, "probe stalls" + (" with flat rps" if rps_flat else " while rps rose")),
             "slow": (rps_flat and slow_new, "one box's compute >= 1.5x the other's"
                      + ("" if slow_new else ", but it was already slow before t0") + ("" if rps_flat else ", rps rose"))}
    ok, why = clear.get(kind, (False, {"overload": "rps surge", "misconfig": "no check fired"}.get(kind, kind)))
    if config:
        ok, why = False, f"router config changed ({config})"
    return {"fired": fired, "trail": trail, "rps": (r0, r1), "rps_flat": rps_flat, "config": config,
            "unambiguous": ok, "why": why}


def rule_diagnose(samples: list[dict], t0: float, signals: Optional[dict] = None) -> Diagnosis:
    sig = signals or rule_signals(samples, t0)
    fired = sig["fired"]
    ranking = list(dict.fromkeys(fired + [h for h in ORDER if h not in fired]))
    scores = {h: round(1.0 - 0.1 * i, 2) for i, h in enumerate(fired)}
    return Diagnosis(policy="rule", ranking=ranking, scores=scores, trail=list(sig["trail"]))
