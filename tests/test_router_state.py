"""POST /v1/state through router.py: a new analysis must reach every box in service.

    python -m unittest tests/test_router_state.py -v

Two mock forecast servers (serve/forecast_server.py --mock) run as subprocesses on free ports. The router runs
in this process through FastAPI's TestClient. The mock accepts any npy_b64, so no NumPy is needed. The mocks keep the newest time pushed to them, so a test
that checks "latest" pushes a time newer than every other test's.
"""
import base64
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from fastapi.testclient import TestClient
    spec = importlib.util.spec_from_file_location("wxrouter_router_state_test", ROOT / "router.py")
    R = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = R
    spec.loader.exec_module(R)
except ImportError:  # fastapi/httpx missing
    R = None

STATE = {"time": "2030-01-01T00", "npy_b64": base64.b64encode(b"not decoded by the mock").decode("ascii")}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock(name):
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "forecast_server.py"), "--mock",
                             "--node", name, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    end = time.time() + 20
    while time.time() < end:
        try:
            urllib.request.urlopen(url + "/health", timeout=1).close()
            return proc, url
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"mock {name} did not start")


@unittest.skipIf(R is None, "needs fastapi and httpx")
class StatePush(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.procs, urls = zip(*(start_mock(n) for n in ("box-a", "box-b")))
        cls.urls = dict(zip(("box-a", "box-b"), urls))

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.kill()
            p.wait()

    def router(self, urls=None):
        urls = urls or self.urls
        backends = [R.Backend(name, url) for name, url in urls.items()]
        # Slow polls: the tests decide which box is up or drained, not the health checks.
        return R.Router(backends, "adaptive", R.AdaptiveConfig(), timeout_s=5, poll_s=600, probe_s=600,
                        down_after=2, up_after=2)

    def test_every_box_gets_the_analysis(self):
        with TestClient(R.create_app(self.router())) as c:
            r = c.post("/v1/state", json=dict(STATE, time="2031-01-01T00"))   # newer than any other test's
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["latest"], "2031-01-01T00")
        self.assertEqual(body["left_out"], [])
        self.assertEqual(sorted(row["backend"] for row in body["backends"]), ["box-a", "box-b"])
        for row in body["backends"]:
            self.assertEqual((row["status"], row["error"]), (200, None))
            self.assertEqual(row["answer"], {"latest": "2031-01-01T00"})

    def test_a_drained_box_is_left_out(self):
        router = self.router()
        router.by_name["box-b"].drained = True
        with TestClient(R.create_app(router)) as c:
            r = c.post("/v1/state", json=dict(STATE, time="2030-01-01T06"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["left_out"], ["box-b"])
        self.assertEqual([row["backend"] for row in r.json()["backends"]], ["box-a"])

    def test_a_bad_body_is_refused_with_the_boxes_status(self):
        with TestClient(R.create_app(self.router())) as c:
            r = c.post("/v1/state", json={"time": "2030-01-01T00"})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertTrue(all(row["status"] == 400 for row in r.json()["backends"]))

    def test_an_unreachable_box_makes_it_502_but_the_other_still_gets_it(self):
        urls = {"box-a": self.urls["box-a"], "box-b": f"http://127.0.0.1:{free_port()}"}
        with TestClient(R.create_app(self.router(urls))) as c:
            r = c.post("/v1/state", json=dict(STATE, time="2030-01-01T12"))
        self.assertEqual(r.status_code, 502, r.text)
        rows = {row["backend"]: row for row in r.json()["backends"]}
        self.assertEqual(rows["box-a"]["status"], 200)
        self.assertIsNone(rows["box-b"]["status"])
        self.assertIn("Connect", rows["box-b"]["error"])
        self.assertIn("latest", r.json())          # box-a's answer is passed on

    def test_no_box_in_service_is_503(self):
        router = self.router()
        for b in router.backends:
            b.drained = True
        with TestClient(R.create_app(router)) as c:
            r = c.post("/v1/state", json=STATE)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["left_out"], ["box-a", "box-b"])


if __name__ == "__main__":
    unittest.main()
