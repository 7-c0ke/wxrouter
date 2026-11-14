"""WhatIf: remediation dry run, evaluation and summary with a fake LLM, and the CLI on a saved incident."""
import asyncio
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from whatif import cli, config, remediate, summarize  # noqa: E402
from whatif.eval import evaluate  # noqa: E402
from whatif.llm import LLMClient  # noqa: E402
from whatif.scenarios import FAULTS, draw_fault  # noqa: E402
from whatif.telemetry import save_incident  # noqa: E402
from whatif_fakes import T0, FakeLLM, make_samples  # noqa: E402

CFG = config.load(env={}, dotenv=None)


class RemediationTests(unittest.TestCase):
    def test_plans(self):
        a = remediate.plan("box-b:slow", CFG)
        self.assertEqual((a.method, a.url, a.body), ("POST", "http://localhost:8000/router/config", {"preferred": "box-a"}))
        self.assertEqual(remediate.plan("box-a:down", CFG).url, "http://localhost:8109/servers/forecast/start")
        self.assertEqual(remediate.plan("router:misconfig", CFG).body, {"policy": "balanced"})
        self.assertIsNone(remediate.plan("load:overload", CFG).method)
        self.assertEqual(remediate.plan("box-b:hang", CFG).url, "http://localhost:8002/admin/mode")

    def test_dry_run_sends_nothing(self):
        sent = []
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200, json={})))

        async def main():
            try:
                return await remediate.execute(remediate.plan("box-b:slow", CFG), CFG, apply=False, client=client)
            finally:
                await client.aclose()
        a = asyncio.run(main())
        self.assertFalse(a.applied)
        self.assertEqual(sent, [])

    def test_apply_measures_p95_before_and_after(self):
        seen = []

        def handler(r):
            seen.append((r.method, r.url.path))
            if r.url.path == "/router/metrics":
                return httpx.Response(200, json={"overall": {"p95_ms": 900.0 if len(seen) == 1 else 300.0}})
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def main():
            try:
                return await remediate.execute(remediate.plan("box-b:slow", CFG), CFG, apply=True, settle_s=0, client=client)
            finally:
                await client.aclose()
        a = asyncio.run(main())
        self.assertTrue(a.applied)
        self.assertEqual(seen, [("GET", "/router/metrics"), ("POST", "/router/config"), ("GET", "/router/metrics")])
        self.assertEqual((a.result["p95_before_ms"], a.result["p95_after_ms"]), (900.0, 300.0))


class ScenarioTests(unittest.TestCase):
    def test_draw_fault_ranges(self):
        import random
        rng = random.Random(3)
        for _ in range(200):
            f = draw_fault(rng)
            self.assertIn(f.kind, FAULTS)
            if f.kind == "slow":
                self.assertTrue(1.5 <= f.severity <= 4.0)
            if f.kind == "overload":
                self.assertTrue(2.0 <= f.severity <= 4.0)
            if f.noise:
                self.assertNotEqual(f.noise["box"], f.box)
                self.assertTrue(3 <= f.noise["dur"] <= 5)
            self.assertIn(f.truth.split(":")[1], ("slow", "error", "hang", "down", "overload", "misconfig"))


class EvalTests(unittest.TestCase):
    def test_eval_and_summary(self):
        incs = [{"id": f"inc{i}", "truth": t, "alert": T0, "fault": {"noise": None}, "buffers": make_samples(t)}
                for i, t in enumerate(["box-b:slow", "box-a:down", "load:overload"])]
        fake = FakeLLM(agents={"box-b:slow": [{"p": 0.9, "next_tool": {"name": "probe", "box": "box-b"}}, {"p": 0.9}]},
                       single=[{"ranking": [{"h": "box-b:slow", "p": 0.8}], "next_tool": None}])

        async def main(out):
            c = LLMClient(["http://e1/v1"], transport=fake.transport())
            try:
                return await evaluate(incs, ["single", "equal", "whatif", "rule"], out, c, parallel=2, progress=lambda s: None)
            finally:
                await c.aclose()

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rows = asyncio.run(main(out))
            self.assertEqual(len(rows), 12)
            self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 12)
            self.assertEqual(len(list((out / "trails").glob("*.txt"))), 12)
            summary = summarize.write(rows, out, ["single", "equal", "whatif", "rule"])
            md = (out / "summary.md").read_text()
        pp = {s["policy"]: s for s in summary["per_policy"]}
        self.assertEqual(pp["rule"]["ac1"], 1.0)
        self.assertAlmostEqual(pp["whatif"]["ac1"], 1 / 3)
        self.assertEqual(pp["rule"]["tokens"], 0)
        self.assertIn("| whatif |", md)
        self.assertIn("Paired, whatif vs each", md)
        # policy order is rotated per incident
        order = [[r["policy"] for r in rows if r["incident"] == i] for i in ("inc0", "inc1")]
        self.assertNotEqual(order[0], order[1])

    def test_sign_test(self):
        self.assertEqual(summarize.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(summarize.sign_test(8, 0), 2 / 256)
        self.assertAlmostEqual(summarize.sign_test(3, 3), 1.0)


class CliTests(unittest.TestCase):
    def test_diagnose_saved_incident_with_rule(self):
        with tempfile.TemporaryDirectory() as d:
            p = save_incident(Path(d) / "x.json", {"id": "x", "truth": "box-a:down", "alert": T0, "buffers": make_samples("box-a:down")})
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["diagnose", "--incident", str(p), "--policy", "rule"])
        out = buf.getvalue()
        self.assertIn("Ranking (rule): box-a:down", out)
        self.assertIn("Truth: box-a:down  (correct)", out)
        self.assertIn("servers/forecast/start", out)
        self.assertIn("(dry run; add --apply", out)


if __name__ == "__main__":
    unittest.main()
