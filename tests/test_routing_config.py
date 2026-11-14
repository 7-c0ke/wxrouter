"""Tests for the routing config API of router.py (the router the Control Room and the real stack use).

    python -m unittest tests/test_routing_config.py -v

No boxes are needed: the app's lifespan (health polls, probes) is not started, and the routing order is
read straight from Router.order().
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from fastapi.testclient import TestClient
    spec = importlib.util.spec_from_file_location("wxrouter_router_under_test", ROOT / "router.py")
    R = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = R
    spec.loader.exec_module(R)
except ImportError:  # fastapi/httpx missing
    R = None


def make(policy="adaptive", preferred=None):
    backends = [R.Backend("box-a", "http://127.0.0.1:1"), R.Backend("box-b", "http://127.0.0.1:2")]
    router = R.Router(backends, policy, R.AdaptiveConfig(preferred=preferred),
                      timeout_s=1, poll_s=1, probe_s=1, down_after=2, up_after=2)
    return router, TestClient(R.create_app(router))


def names(router):
    return [b.name for b in router.order()]


@unittest.skipIf(R is None, "needs fastapi and httpx")
class ConfigApi(unittest.TestCase):

    def test_get_config_lists_what_may_be_set(self):
        router, c = make(preferred="box-b")
        cfg = c.get("/router/config").json()
        self.assertEqual(cfg["policy"], "adaptive")
        self.assertEqual(cfg["preferred"], "box-b")
        self.assertEqual(cfg["backends"], ["box-a", "box-b"])
        self.assertIn("balanced", cfg["policies"])
        self.assertEqual(cfg["drained"], [])
        for k in ("margin", "window_s", "cooldown_s", "return_margin", "stall_ms"):
            self.assertIn(k, cfg["settings"])
            self.assertLessEqual(cfg["limits"][k]["min"], cfg["settings"][k])
            self.assertLessEqual(cfg["settings"][k], cfg["limits"][k]["max"])
        self.assertEqual(cfg["defaults"]["preferred"], "box-b")

    def test_post_sets_every_field(self):
        router, c = make()
        r = c.post("/router/config", json={"policy": "balanced", "margin": 0.4, "window_s": 2,
                                           "cooldown_s": 5, "return_margin": 0.1, "stall_ms": 700,
                                           "drain": {"box-a": True}})
        self.assertEqual(r.status_code, 200)
        cfg = r.json()["config"]
        self.assertEqual(cfg["policy"], "balanced")
        self.assertEqual(cfg["settings"], {"margin": 0.4, "window_s": 2, "cooldown_s": 5,
                                           "return_margin": 0.1, "stall_ms": 700})
        self.assertEqual(cfg["drained"], ["box-a"])
        self.assertTrue(r.json()["backends"]["box-a"]["drained"])

    def test_out_of_range_or_unknown_box_changes_nothing(self):
        router, c = make()
        for body in ({"margin": 2}, {"stall_ms": 5}, {"cooldown_s": -1}, {"policy": "fastest"},
                     {"preferred": "box-z", "cooldown_s": 3}, {"drain": {"box-z": True}, "margin": 0.5}):
            with self.subTest(body=body):
                self.assertEqual(c.post("/router/config", json=body).status_code, 422)
        cfg = c.get("/router/config").json()
        self.assertEqual(cfg["settings"]["cooldown_s"], 10.0)
        self.assertEqual(cfg["settings"]["margin"], 0.25)

    def test_reset_goes_back_to_start_up_settings(self):
        router, c = make(policy="adaptive", preferred="box-b")
        c.post("/router/config", json={"policy": "round_robin", "preferred": "", "cooldown_s": 1,
                                       "stall_ms": 300, "drain": {"box-b": True}})
        cfg = c.post("/router/config", json={"reset": True}).json()["config"]
        self.assertEqual(cfg["policy"], "adaptive")
        self.assertEqual(cfg["preferred"], "box-b")
        self.assertEqual(cfg["settings"]["cooldown_s"], 10.0)
        self.assertEqual(cfg["settings"]["stall_ms"], 1000.0)
        self.assertEqual(cfg["drained"], [])
        self.assertEqual(router.active.name, "box-b")

    def test_teammate_names_work_too(self):
        router, c = make()
        self.assertEqual(c.post("/router/policy", json={"policy": "least_loaded"}).json(), {"policy": "least_loaded"})
        self.assertEqual(router.policy, "least_loaded")
        self.assertEqual(c.post("/router/policy", json={"policy": "fastest"}).status_code, 400)
        r = c.post("/router/drain", json={"backend": "box-a", "drain": True})
        self.assertEqual(r.json(), {"backend": "box-a", "drain": True, "pools": ["forecast"]})
        self.assertTrue(router.by_name["box-a"].drained)
        self.assertEqual(c.post("/router/drain", json={"backend": "box-z", "drain": True}).status_code, 404)
        self.assertEqual(c.post("/router/drain", json={"backend": "box-a"}).status_code, 400)
        h = c.get("/router/health").json()
        self.assertEqual((h["status"], h["usable"]), ("ok", {"forecast": 1}))

    def test_response_headers_name_the_box_both_ways(self):
        router, _ = make()
        h = router.headers(router.by_name["box-b"], ["box-a", "box-b"])
        self.assertEqual((h["X-Served-By"], h["X-Backend"], h["X-Router-Retries"]), ("box-b", "box-b", "1"))


@unittest.skipIf(R is None, "needs fastapi and httpx")
class RoutingOrder(unittest.TestCase):

    def test_static_uses_the_preferred_box(self):
        router, c = make(policy="static")
        self.assertEqual(names(router), ["box-a", "box-b"])
        c.post("/router/config", json={"preferred": "box-b"})
        self.assertEqual(names(router), ["box-b", "box-a"])

    def test_drained_box_is_last_resort_under_every_rule(self):
        for policy in R.POLICIES:
            with self.subTest(policy=policy):
                router, c = make(policy=policy)
                c.post("/router/config", json={"drain": {"box-a": True}})
                router.decide_adaptive()  # adaptive leaves a drained active box at once
                for _ in range(20):
                    self.assertEqual(names(router), ["box-b", "box-a"])

    def test_down_box_comes_after_a_drained_one(self):
        router, c = make(policy="round_robin")
        c.post("/router/config", json={"drain": {"box-a": True}})
        router.by_name["box-b"].healthy = False
        self.assertEqual(names(router), ["box-a", "box-b"])

    def test_adaptive_does_not_go_home_to_a_drained_preferred_box(self):
        router, c = make(policy="adaptive", preferred="box-b")
        c.post("/router/config", json={"drain": {"box-b": True}})
        router.decide_adaptive()
        self.assertEqual(router.active.name, "box-a")
        for b in router.backends:
            b.probe_ms = 50.0
        router.adaptive.window_s = router.adaptive.cooldown_s = 0
        router.decide_adaptive()
        router.decide_adaptive()
        self.assertEqual(router.active.name, "box-a")
        c.post("/router/config", json={"drain": {"box-b": False}})
        router.decide_adaptive()
        router.decide_adaptive()
        self.assertEqual(router.active.name, "box-b")


if __name__ == "__main__":
    unittest.main()
