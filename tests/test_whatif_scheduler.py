"""WhatIf scheduler: whatif / equal / single with a fake LLM, and the rule baseline on synthetic faults."""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from whatif.llm import LLMClient  # noqa: E402
from whatif.scheduler import diagnose, rule_diagnose  # noqa: E402
from whatif_fakes import T0, FakeLLM, make_samples  # noqa: E402

PROBE_B = {"name": "probe", "box": "box-b"}


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
            "box-a:slow": [{"p": 0.3, "next_tool": {"name": "probe", "box": "box-a"}},
                           {"p": 0.05, "next_tool": None, "suggest": ["box-b:hang"]}],
            "box-b:hang": [{"p": 0.1, "next_tool": None}],
        })
        d = run("whatif", fake)
        self.assertEqual(d.top(), "box-b:slow")
        trail = "\n".join(d.trail)
        self.assertIn("[scheduler] prune box-a:slow", trail)
        self.assertIn("[scheduler] spawn box-b:hang (suggested)", trail)
        self.assertIn("[scheduler] stop: a clear leader", trail)
        self.assertEqual(d.tool_units, 2)  # two probes, each once
        self.assertLessEqual(d.rounds, 3)
        self.assertEqual(d.thinking_calls, 0)
        self.assertEqual(len(d.ranking), 10)

    def test_close_call_goes_to_a_thinking_judge(self):
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.6, "next_tool": PROBE_B}, {"p": 0.6, "next_tool": None}],
                               "load:overload": [{"p": 0.55, "next_tool": {"name": "traffic"}}, {"p": 0.55, "next_tool": None}]},
                       judge={"ranking": [{"h": "box-b:slow", "p": 0.8}, {"h": "load:overload", "p": 0.2}]})
        d = run("whatif", fake)
        self.assertEqual(d.ranking[:2], ["box-b:slow", "load:overload"])
        self.assertEqual(d.thinking_calls, 1)
        judge = [r for r in fake.requests if "could not separate" in r["body"]["messages"][-1]["content"]]
        self.assertEqual(judge[0]["body"]["chat_template_kwargs"], {"enable_thinking": True})

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
        fake = FakeLLM(wrap_think=True, agents={"box-b:slow": [{"p": 0.95, "next_tool": PROBE_B}, {"p": 0.95, "next_tool": None}]})
        d = run("whatif", fake)
        self.assertEqual((d.top(), d.parse_failures), ("box-b:slow", 0))

    def test_priority_follows_confidence(self):
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.7, "next_tool": PROBE_B}, {"p": 0.95, "next_tool": None}],
                               "box-a:slow": [{"p": 0.3, "next_tool": {"name": "probe", "box": "box-a"}}, {"p": 0.2}]})
        d = run("whatif", fake)
        self.assertEqual(d.top(), "box-b:slow")
        # round 2: box-b:slow (p=.7) was asked first
        r2 = [r["body"]["messages"][-1]["content"] for r in fake.requests[5:]]
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
        self.assertEqual(d.llm_calls, 1 + 4 * d.rounds)  # planner + every agent every round
        self.assertEqual(d.thinking_calls, 0)


class SinglePolicyTests(unittest.TestCase):
    def test_single_agent_tool_loop(self):
        fake = FakeLLM(single=[{"ranking": [{"h": "load:overload", "p": 0.4}], "next_tool": PROBE_B},
                               {"ranking": [{"h": "box-b:slow", "p": 0.9}, {"h": "box-a:slow", "p": 0.1}], "next_tool": None}])
        d = run("single", fake)
        self.assertEqual(d.ranking[:2], ["box-b:slow", "box-a:slow"])
        self.assertEqual((d.llm_calls, d.tool_units), (2, 1))


class RuleTests(unittest.TestCase):
    def test_each_fault(self):
        for fault in ("box-a:down", "box-b:down", "box-a:hang", "box-b:error", "box-a:slow", "box-b:slow",
                      "load:overload", "router:misconfig"):
            d = rule_diagnose(make_samples(fault), T0)
            self.assertEqual(d.top(), fault, f"{fault}: {d.trail}")
            self.assertEqual(len(d.ranking), 10)

    def test_rule_policy_via_diagnose(self):
        d = asyncio.run(diagnose(make_samples("box-a:down"), T0, "rule"))
        self.assertEqual((d.top(), d.llm_calls), ("box-a:down", 0))


if __name__ == "__main__":
    unittest.main()
