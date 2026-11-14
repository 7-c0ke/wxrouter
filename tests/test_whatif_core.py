"""WhatIf: config, telemetry stripping, window tools, hypotheses, JSON extraction, the gate and the LLM client."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from whatif import config, telemetry, tools  # noqa: E402
from whatif.hypotheses import HYPOTHESES, ORDER, normalize  # noqa: E402
from whatif.llm import LLMClient, LLMError, PriorityGate, extract_json, strip_think  # noqa: E402
from whatif_fakes import T0, FakeLLM, make_samples  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        c = config.load(env={}, dotenv=None)
        self.assertEqual(c.llm_endpoints, ("http://localhost:8090/v1", "http://localhost:8091/v1"))
        self.assertEqual((c.model, c.node_cap, c.thinking), ("nemotron", 4, False))
        self.assertEqual(c.boxes["box-b"], "http://localhost:8002")
        self.assertEqual(c.stacks["box-a"], "http://localhost:8109")

    def test_dotenv_then_env_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text('# c\nexport MODEL="other"\nLLM_ENDPOINTS=http://x/v1/, http://y/v1\nLLM_API_KEY=secret\nNODE_CAP=2\n')
            c = config.load(env={"MODEL": "env-model"}, dotenv=p)
        self.assertEqual(c.model, "env-model")
        self.assertEqual(c.llm_endpoints, ("http://x/v1", "http://y/v1"))
        self.assertEqual(c.node_cap, 2)
        self.assertEqual(c.redacted()["llm_api_key"], "set")
        self.assertNotIn("secret", json.dumps(c.redacted()))
        self.assertEqual(c.key_for("http://y/v1/"), "secret")

    def test_one_key_per_endpoint(self):
        c = config.load(env={"LLM_ENDPOINTS": "http://x/v1,http://y/v1", "LLM_API_KEY": "ka, kb"}, dotenv=None)
        self.assertEqual((c.key_for("http://x/v1"), c.key_for("http://y/v1")), ("ka", "kb"))
        with self.assertRaises(ValueError):
            config.load(env={"LLM_ENDPOINTS": "http://x/v1,http://y/v1", "LLM_API_KEY": "a,b,c"}, dotenv=None)


class StripTests(unittest.TestCase):
    def test_stats_drop_mode_and_extras(self):
        s = telemetry.strip_stats({"in_flight": 1, "queue_depth": 2, "gpu_util": 0.7, "mode": "slow",
                                   "slowdown": 3.0, "busy_other": True, "p95_latency_ms": 9})
        self.assertNotIn("mode", s)
        self.assertNotIn("slowdown", s)
        self.assertNotIn("busy_other", s)
        self.assertEqual(s["queue_depth"], 2)

    def test_router_and_supervisor(self):
        r = telemetry.strip_router({"policy": "adaptive", "failovers": 2,
                                    "adaptive": {"active": "box-b", "preferred": "box-b", "switches": [{}, {}]},
                                    "backends": {"box-a": {"healthy": True, "probe_ms": 50, "url": "x"}}})
        self.assertEqual((r["active"], r["switches"]), ("box-b", 2))
        self.assertNotIn("url", r["backends"]["box-a"])
        j = telemetry.strip_supervisor({"servers": {"forecast": {"state": "running", "pid": 3}}, "jobs": {"finetune": {"state": "idle"}}})
        self.assertEqual(j, {"servers": {"forecast": "running"}, "jobs": {"finetune": "idle"}})

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            inc = {"id": "x", "truth": "box-b:slow", "alert": T0, "buffers": make_samples()}
            telemetry.save_incident(Path(d) / "x.json", inc)
            self.assertEqual(telemetry.load_incidents(Path(d))[0], json.loads(json.dumps(inc)))
            (Path(d) / "bad.json").write_text("{}")
            with self.assertRaises(ValueError):
                telemetry.load_incident(Path(d) / "bad.json")


class ToolTests(unittest.TestCase):
    def test_compare_line_step(self):
        pts = [(T0 + i, 57.0 if i < 2 else 171.0) for i in range(-30, 30)]
        line = tools.compare("box-b.compute_ms", pts, T0).line()
        self.assertEqual(line, "box-b.compute_ms: 57 -> 171 (+200%), onset +2s, step")

    def test_compare_flat_and_blip(self):
        flat = [(T0 + i, 57.0 + (i % 3)) for i in range(-30, 30)]
        self.assertIn("no change", tools.compare("m", flat, T0).line())
        blip = [(T0 + i, 200.0 if 3 <= i < 7 else 57.0) for i in range(-30, 30)]
        ch = tools.compare("m", blip, T0)
        self.assertEqual((ch.shape, ch.onset), ("blip", 3.0))

    def test_probe_tool_sees_slow_box_only(self):
        s = make_samples("box-b:slow")
        self.assertIn("step", tools.probe(s, T0, "box-b").splitlines()[0])
        self.assertIn("no change", tools.probe(s, T0, "box-a").splitlines()[0])

    def test_down_error_hang_signatures(self):
        self.assertIn("no answer", tools.box_health(make_samples("box-a:down"), T0, "box-a"))
        self.assertIn("503", tools.box_health(make_samples("box-a:error"), T0, "box-a"))
        hang = tools.probe(make_samples("box-a:hang"), T0, "box-a")
        self.assertRegex(hang, r"probe_stalls\(>1s\): 0/\d+ -> [1-9]")

    def test_router_and_traffic(self):
        s = make_samples("router:misconfig")
        r = tools.router_status(s, T0)
        self.assertIn("router.policy: adaptive -> static (changed at +", r)
        self.assertIn("traffic.share_box-a: 0 -> 0.93, onset +2s, step", tools.traffic(s, T0))

    def test_box_jobs_without_supervisor(self):
        self.assertIn("not reachable", tools.box_jobs(make_samples(), T0, "box-a"))

    def test_toolbox_budget_and_cache(self):
        tb = tools.Toolbox(make_samples(), T0, budget=3)
        self.assertEqual(tb.call("probe", {"box": "box-b"}).cost, 1)
        again = tb.call("probe", {"box": "box-b"})
        self.assertTrue(again.cached)
        self.assertEqual(again.cost, 0)
        self.assertTrue(tb.call("box_jobs", {"box": "box-a"}).cost == 2)
        self.assertEqual(tb.left(), 0)
        self.assertTrue(tb.call("traffic").error)
        self.assertTrue(tb.call("nope").error)
        self.assertTrue(tb.call("probe", {"box": "box-c"}).error)
        self.assertIn("## probe(box-b)", tb.evidence())


class HypothesisTests(unittest.TestCase):
    def test_ten_and_normalize(self):
        self.assertEqual(len(HYPOTHESES), 10)
        self.assertEqual(set(ORDER), set(HYPOTHESES))
        for raw, want in [("box-b:slow", "box-b:slow"), ("Box_B:Slow", "box-b:slow"), ("boxa:hang", "box-a:hang"),
                          ("overload", "load:overload"), ("misconfig", "router:misconfig"), ("box-c:slow", None), (3, None)]:
            self.assertEqual(normalize(raw), want, raw)


class JsonTests(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(extract_json('{"p": 0.9}'), {"p": 0.9})
        self.assertEqual(extract_json('<think>maybe {"p": 0}</think>Answer: {"p": 0.4, "x": [1,2,],}'), {"p": 0.4, "x": [1, 2]})
        self.assertEqual(extract_json('```json\n{"a": "b}"}\n```'), {"a": "b}"})
        self.assertEqual(extract_json('reasoning...</think>{"ok": true}'), {"ok": True})
        with self.assertRaises(ValueError):
            extract_json("<think>never closed {\"p\": 1}")
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_strip_think(self):
        self.assertEqual(strip_think("<think>a</think> b"), "b")


class GateTests(unittest.TestCase):
    def test_cap_and_priority(self):
        async def main():
            gate = PriorityGate(2)
            order, running, peak = [], 0, 0

            async def job(name, prio, hold):
                nonlocal running, peak
                await gate.acquire(prio)
                running += 1
                peak = max(peak, running)
                order.append(name)
                await asyncio.sleep(hold)
                running -= 1
                gate.release()

            first = [asyncio.create_task(job(f"h{i}", 5, 0.05)) for i in range(2)]
            await asyncio.sleep(0.01)  # both slots taken
            rest = [asyncio.create_task(job("low", 9, 0)), asyncio.create_task(job("high", 0, 0)),
                    asyncio.create_task(job("mid", 5, 0))]
            await asyncio.gather(*first, *rest)
            return order, peak, gate.active

        order, peak, active = asyncio.run(main())
        self.assertEqual(peak, 2)
        self.assertEqual(order[2:], ["high", "mid", "low"])
        self.assertEqual(active, 0)


class ClientTests(unittest.TestCase):
    def run_client(self, fake, endpoints=("http://e1/v1", "http://e2/v1"), log=None, **kw):
        async def main():
            c = LLMClient(list(endpoints), cap=2, transport=fake.transport(), log_path=log)
            try:
                return [await c.chat([{"role": "user", "content": "Choose the"}], "inc-1", **kw) for _ in range(2)], c
            finally:
                await c.aclose()
        return asyncio.run(main())

    def test_sticky_headers_thinking_off(self):
        fake = FakeLLM()
        with tempfile.TemporaryDirectory() as d:
            (r1, r2), c = self.run_client(fake, log=Path(d) / "calls.jsonl")
            rows = (Path(d) / "calls.jsonl").read_text().splitlines()
        self.assertEqual(r1.endpoint, r2.endpoint)
        req = fake.requests[0]
        self.assertEqual(req["headers"]["x-session-id"], "inc-1")
        self.assertEqual(req["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(c.stats["inc-1"].calls, 2)
        self.assertEqual(c.stats["inc-1"].tokens, 240)
        self.assertEqual(len(rows), 2)

    def test_failover_moves_the_incident(self):
        fake = FakeLLM(fail_status={"e1": 503})
        (r1, r2), c = self.run_client(fake)
        self.assertEqual((r1.endpoint, r2.endpoint), ("http://e2/v1", "http://e2/v1"))
        self.assertEqual(c.stats["inc-1"].failovers, 1)

    def test_all_down_raises(self):
        fake = FakeLLM(fail_status={"e1": 500, "e2": 502})
        with self.assertRaises(LLMError):
            self.run_client(fake)

    def test_thinking_flag(self):
        fake = FakeLLM()
        (r1, _), c = self.run_client(fake, thinking=True)
        self.assertEqual(fake.requests[0]["body"]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(c.stats["inc-1"].thinking_calls, 2)

    def test_rate_limit_backs_off_on_one_endpoint(self):
        import httpx
        from whatif import llm
        hits = []

        def handler(req):
            hits.append(req)
            if len(hits) <= 2:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}], "usage": {}})

        async def main():
            c = LLMClient(["http://api/v1"], transport=httpx.MockTransport(handler))
            try:
                return await c.chat([{"role": "user", "content": "x"}], "inc-9"), c
            finally:
                await c.aclose()
        old, llm.RATE_LIMIT_BACKOFF_S = llm.RATE_LIMIT_BACKOFF_S, 0.0
        try:
            r, c = asyncio.run(main())
        finally:
            llm.RATE_LIMIT_BACKOFF_S = old
        self.assertEqual(r.text, '{"ok": 1}')
        self.assertEqual((len(hits), c.stats["inc-9"].rate_limited), (3, 2))


if __name__ == "__main__":
    unittest.main()
