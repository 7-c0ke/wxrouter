"""WhatIf scheduler: whatif / equal / single with a fake LLM, and the rule baseline on synthetic faults."""
import asyncio
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from whatif.llm import LLMClient  # noqa: E402
from whatif import prompts  # noqa: E402
from whatif.scheduler import JUDGE_RETRY_TOKENS, MAX_SPAWN, TRIAGE_BUDGET, diagnose, rule_diagnose, rule_signals  # noqa: E402
from whatif_fakes import T0, FakeLLM, make_samples  # noqa: E402

PROBE_B = {"name": "probe", "box": "box-b"}
JUDGE_B = {"ranking": [{"h": "box-b:slow", "p": 0.9}, {"h": "box-a:slow", "p": 0.05}]}


def judge_requests(fake):
    return [r for r in fake.requests if "FINAL JUDGEMENT" in r["body"]["messages"][-1]["content"]]


def run(policy, fake, samples=None, **kw):
    async def main():
        c = LLMClient(["http://e1/v1", "http://e2/v1"], cap=4, transport=fake.transport())
        try:
            return await diagnose(samples or make_samples("box-b:slow"), T0, policy, c, session="inc", **kw)
        finally:
            await c.aclose()
    return asyncio.run(main())


class WhatIfPolicyTests(unittest.TestCase):
    def test_prune_spawn_and_early_stop(self):
        fake = FakeLLM(agents={
            "box-b:slow": [{"p": 0.5, "next_tool": PROBE_B, "reason": "check compute"},
                           {"p": 0.9, "next_tool": None, "reason": "compute x3, step at +2s"}],
            "box-a:slow": [{"p": 0.05, "next_tool": {"name": "probe", "box": "box-a"}, "suggest": ["box_b hang"]}],
            "box-b:hang": [{"p": 0.1, "next_tool": None}],
        }, judge=JUDGE_B)
        d = run("whatif", fake)
        self.assertEqual(d.top(), "box-b:slow")
        trail = "\n".join(d.trail)
        self.assertIn("[scheduler] prune box-a:slow", trail)
        self.assertIn("[scheduler] spawn box-b:hang (suggested)", trail)
        self.assertLessEqual(trail.count("[scheduler] spawn"), MAX_SPAWN)
        self.assertIn("[scheduler] stop: a clear leader", trail)
        self.assertEqual(d.tool_units, 2)  # two probes, each once
        self.assertEqual(d.rounds, 2)
        self.assertEqual(d.thinking_calls, 0)  # the judge does not think by default
        self.assertEqual(len(d.ranking), 10)

    def test_no_early_stop_in_round_one(self):
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.95, "next_tool": PROBE_B}, {"p": 0.95, "next_tool": None}]},
                       judge=JUDGE_B)
        d = run("whatif", fake)
        self.assertEqual(d.rounds, 2)
        self.assertIn("stop: a clear leader", "\n".join(d.trail))

    def test_final_ranking_is_the_judges_beliefs(self):
        # The agents favour overload; the judge, reading the evidence, puts box-b:slow first. The judge wins.
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.4, "next_tool": PROBE_B}, {"p": 0.4, "next_tool": None}],
                               "load:overload": [{"p": 0.7, "next_tool": {"name": "traffic"}}, {"p": 0.7, "next_tool": None}]},
                       judge={"ranking": [{"h": "Box B slow", "p": 0.8}, {"h": "overload", "p": 0.2}]})
        d = run("whatif", fake, judge_thinking=True)
        self.assertEqual(d.ranking[:2], ["box-b:slow", "load:overload"])
        self.assertEqual(d.thinking_calls, 1)
        j = judge_requests(fake)
        self.assertEqual(len(j), 1)
        self.assertEqual(j[0]["body"]["chat_template_kwargs"], {"enable_thinking": True})
        for h in ("box-b:slow", "load:overload", "box-a:slow"):  # every branch goes to the judge
            self.assertIn(f"- {h} (agent's last p=", j[0]["body"]["messages"][-1]["content"])

    def test_judge_parse_failure_retries_once_without_thinking(self):
        fake = FakeLLM(judge=["", {"ranking": [{"id": "box_b:slow", "p": 0.7, "odd": 1}, {"h": "box-z:slow", "p": 0.3}]}])
        d = run("whatif", fake, judge_thinking=True)
        j = judge_requests(fake)
        self.assertEqual(len(j), 2)
        retry = j[1]["body"]
        self.assertEqual(retry["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(retry["max_tokens"], JUDGE_RETRY_TOKENS)
        self.assertTrue(retry["messages"][-1]["content"].endswith(prompts.JSON_ONLY))
        self.assertEqual(d.top(), "box-b:slow")
        trail = "\n".join(d.trail)
        self.assertIn("[judge retry] unknown hypothesis ids ignored: 'box-z:slow'", trail)
        self.assertIn("[judge retry] unexpected keys ignored: odd", trail)

    def test_no_usable_judge_answer_falls_back_to_the_rule(self):
        # Agents wrongly favour box-a:hang; the judge fails twice; the rule (box-b compute x3) decides.
        fake = FakeLLM(agents={"box-a:slow": [{"p": 0.9, "next_tool": PROBE_B}]}, judge=["no json", "still none"])
        d = run("whatif", fake)
        self.assertEqual(len(judge_requests(fake)), 2)
        self.assertEqual(d.top(), "box-b:slow")
        self.assertIn("ranking by the rule", "\n".join(d.trail))

    def test_ties_break_by_rule_not_list_order(self):
        # The judge lists box-a:slow first but gives both the same p; the rule (box-b slow) breaks the tie.
        fake = FakeLLM(judge={"ranking": [{"h": "box-a:slow", "p": 0.5}, {"h": "box-b:slow", "p": 0.5}]})
        d = run("whatif", fake)
        self.assertEqual(d.ranking[:2], ["box-b:slow", "box-a:slow"])

    def test_budget_is_never_exceeded(self):
        greedy = lambda tool: [{"p": 0.5, "next_tool": tool}]
        fake = FakeLLM(agents={"box-b:slow": greedy({"name": "box_jobs", "box": "box-b"}),
                               "box-a:slow": greedy({"name": "box_jobs", "box": "box-a"}),
                               "load:overload": greedy({"name": "traffic"}),
                               "router:misconfig": greedy({"name": "router_status"})})
        d = run("whatif", fake, budget=4)
        self.assertLessEqual(d.tool_units, 4)
        self.assertIn("stop: budget spent", "\n".join(d.trail))

    def test_unparseable_replies_are_counted_not_fatal(self):
        fake = FakeLLM(plan="I think box b", agents={"box-b:slow": ["sorry, no json"]})
        d = run("whatif", fake)
        self.assertGreaterEqual(d.parse_failures, 2)
        self.assertEqual(len(d.ranking), 10)

    def test_think_wrapped_json_parses(self):
        fake = FakeLLM(wrap_think=True, agents={"box-b:slow": [{"p": 0.95, "next_tool": PROBE_B}, {"p": 0.95, "next_tool": None}]},
                       judge=JUDGE_B)
        d = run("whatif", fake)
        self.assertEqual((d.top(), d.parse_failures), ("box-b:slow", 0))

    def test_priority_follows_confidence(self):
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.7, "next_tool": PROBE_B}, {"p": 0.95, "next_tool": None}],
                               "box-a:slow": [{"p": 0.3, "next_tool": {"name": "probe", "box": "box-a"}}, {"p": 0.2}]},
                       judge=JUDGE_B)
        d = run("whatif", fake)
        self.assertEqual(d.top(), "box-b:slow")
        # round 2 (after the planner and 3 agents): box-b:slow (p=.7) was asked first
        r2 = [r["body"]["messages"][-1]["content"] for r in fake.requests[4:]]
        self.assertIn("YOUR HYPOTHESIS: box-b:slow", r2[0])


class EqualPolicyTests(unittest.TestCase):
    def test_no_prune_no_spawn_all_agents_each_round(self):
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.5, "next_tool": PROBE_B}, {"p": 0.9, "next_tool": None}],
                               "box-a:slow": [{"p": 0.05, "next_tool": None, "suggest": ["box-b:hang"]}]})
        d = run("equal", fake)
        trail = "\n".join(d.trail)
        self.assertNotIn("prune", trail)
        self.assertNotIn("spawn", trail)
        self.assertEqual(d.top(), "box-b:slow")
        self.assertEqual(d.llm_calls, 1 + 3 * d.rounds)  # planner + every agent every round, no judge
        self.assertEqual(d.thinking_calls, 0)


class SinglePolicyTests(unittest.TestCase):
    def test_single_agent_tool_loop(self):
        fake = FakeLLM(single=[{"ranking": [{"h": "load:overload", "p": 0.4}], "next_tool": PROBE_B},
                               {"ranking": [{"h": "box-b:slow", "p": 0.9}, {"h": "box-a:slow", "p": 0.1}], "next_tool": None}])
        d = run("single", fake)
        self.assertEqual(d.ranking[:2], ["box-b:slow", "box-a:slow"])
        self.assertEqual((d.llm_calls, d.tool_units), (2, 1))


class TriagePolicyTests(unittest.TestCase):
    def test_unambiguous_signal_needs_no_llm(self):
        for fault in ("box-a:down", "box-b:error", "box-a:hang", "box-b:slow"):
            fake = FakeLLM()
            d = run("triage", fake, samples=make_samples(fault))
            self.assertEqual((d.top(), d.llm_calls, d.tokens), (fault, 0, 0), fault)
            self.assertEqual(fake.requests, [])

    def test_ambiguous_runs_whatif_seeded_by_the_rule(self):
        fake = FakeLLM(agents={"load:overload": [{"p": 0.8, "next_tool": {"name": "traffic"}}, {"p": 0.9, "next_tool": None}]},
                       judge={"ranking": [{"h": "load:overload", "p": 0.9}]})
        d = run("triage", fake, samples=make_samples("load:overload"))
        trail = "\n".join(d.trail)
        self.assertIn("[triage] not clear-cut (rps surge)", trail)
        self.assertNotIn("[planner]", trail)
        self.assertFalse(any("Choose the" in r["body"]["messages"][-1]["content"] for r in fake.requests))
        first = {re.search(r"YOUR HYPOTHESIS: ([\w-]+:[\w-]+):", r["body"]["messages"][-1]["content"]).group(1)
                 for r in fake.requests[:3]}
        self.assertEqual(first, set(rule_diagnose(make_samples("load:overload"), T0).ranking[:3]))
        self.assertIn("Budget left: 6 units.", fake.requests[0]["body"]["messages"][-1]["content"])
        self.assertEqual(TRIAGE_BUDGET, 6)
        self.assertEqual(d.top(), "load:overload")

    def test_config_change_is_never_clear_cut(self):
        sig = rule_signals(make_samples("router:misconfig"), T0)
        self.assertFalse(sig["unambiguous"])
        self.assertIn("router config changed", sig["why"])


class RuleTests(unittest.TestCase):
    def test_each_fault(self):
        for fault in ("box-a:down", "box-b:down", "box-a:hang", "box-b:error", "box-a:slow", "box-b:slow",
                      "load:overload", "router:misconfig"):
            d = rule_diagnose(make_samples(fault), T0)
            self.assertEqual(d.top(), fault, f"{fault}: {d.trail}")
            self.assertEqual(len(d.ranking), 10)

    def test_queueing_from_load_or_config_is_not_a_hang(self):
        # Stalls on a healthy box plus rps x3 -> overload; stalls plus a policy change -> misconfig.
        over = make_samples("load:overload")
        mis = make_samples("router:misconfig")
        for s in over + mis:
            if s["t"] >= T0 + 2 and "box-b" in s["probe"]:
                s["probe"]["box-b"] = {"t": s["t"], "code": 0, "compute_ms": None, "total_ms": 3000.0, "timeout": True}
        self.assertEqual(rule_diagnose(over, T0).ranking[:2], ["load:overload", "box-b:hang"])
        self.assertEqual(rule_diagnose(mis, T0).ranking[:2], ["router:misconfig", "box-b:hang"])

    def test_rule_policy_via_diagnose(self):
        d = asyncio.run(diagnose(make_samples("box-a:down"), T0, "rule"))
        self.assertEqual((d.top(), d.llm_calls), ("box-a:down", 0))


if __name__ == "__main__":
    unittest.main()
