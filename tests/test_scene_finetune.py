"""bench/scene_finetune.py: the parts that need no stack.

    python -m unittest tests.test_scene_finetune -v
"""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("scene_finetune", os.path.join(ROOT, "bench", "scene_finetune.py"))
sf = importlib.util.module_from_spec(spec)
sys.modules["scene_finetune"] = sf
spec.loader.exec_module(sf)

START = {"event": "start", "time": "2026-11-14T20:59:46", "settings": {"steps": 300, "batch": 16}}


def step(n):
    return {"step": n, "loss": 0.1, "lr": 3e-5, "grad_norm": 1.0, "dt": 2.1}


class StepsInLog(unittest.TestCase):
    def count(self, rows):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        try:
            return sf.steps_in_log(f.name)
        finally:
            os.unlink(f.name)

    def test_one_run(self):
        self.assertEqual(self.count([START] + [step(n) for n in range(1, 6)]), 5)

    def test_an_older_run_in_the_same_file_does_not_count(self):
        # finetune.py appends: a second run in the same folder used to read as step 300 from its first second,
        # so every phase of the scene reported 0.0 steps a second.
        older = [START] + [step(n) for n in range(1, 301)] + [{"event": "end", "step": 300, "complete": True}]
        self.assertEqual(self.count(older + [START]), 0)
        self.assertEqual(self.count(older + [START] + [step(n) for n in range(1, 8)]), 7)

    def test_lines_that_are_not_training_steps_do_not_count(self):
        rows = [START, step(1), step(2), {"event": "validation", "step": 2, "score": 0.98},
                {"event": "error", "step": 9, "loss": "nan"}, {"event": "end", "step": 300}]
        self.assertEqual(self.count(rows), 2)

    def test_an_empty_file(self):
        self.assertEqual(self.count([]), 0)


class Config(BaseHTTPRequestHandler):
    answer = (200, {"policy": "adaptive", "preferred": "box-b", "drained": [], "settings": {"margin": 0.25}})

    def do_GET(self):
        status, body = self.answer
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class SettingsNow(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), Config)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_the_rule_and_the_home_box(self):
        Config.answer = (200, {"policy": "adaptive", "preferred": "box-b", "settings": {"margin": 0.25}})
        self.assertEqual(sf.settings_now(self.url + "/"), {"policy": "adaptive", "preferred": "box-b"})

    def test_no_home_box_is_an_empty_text(self):
        Config.answer = (200, {"policy": "balanced", "preferred": None})
        self.assertEqual(sf.settings_now(self.url), {"policy": "balanced", "preferred": ""})

    def test_unreadable_settings_are_none(self):
        Config.answer = (500, {"detail": "no"})
        self.assertIsNone(sf.settings_now(self.url))
        self.assertIsNone(sf.settings_now("http://127.0.0.1:9"))        # nothing listens there


if __name__ == "__main__":
    unittest.main()
