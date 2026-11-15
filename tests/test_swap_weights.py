"""bench/swap_weights.py: the link, which needs no stack.

    python -m unittest tests.test_swap_weights -v
"""
import importlib.util
import os
import sys
import tempfile
import unittest
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("swap_weights", os.path.join(ROOT, "bench", "swap_weights.py"))
sw = importlib.util.module_from_spec(spec)
sys.modules["swap_weights"] = sw
spec.loader.exec_module(sw)


class Link(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.weights = os.path.join(self.tmp.name, "weights")
        self.other = os.path.join(self.tmp.name, "ft")
        os.makedirs(self.weights)
        os.makedirs(self.other)
        for path, text in ((os.path.join(self.weights, "base.pt"), "base"), (os.path.join(self.other, "tuned.pt"), "tuned")):
            with open(path, "w") as f:
                f.write(text)
        self.link = os.path.join(self.weights, "serving.pt")

    def tearDown(self):
        self.tmp.cleanup()

    def read(self):
        with open(self.link) as f:
            return f.read()

    def test_a_file_beside_the_link_is_linked_by_name(self):
        self.assertEqual(sw.point_link(self.link, os.path.join(self.weights, "base.pt")), "base.pt")
        self.assertEqual(os.readlink(self.link), "base.pt")
        self.assertEqual(self.read(), "base")

    def test_there_and_back(self):
        # What main() does when a box does not come back: the link as os.readlink() gave it before the swap.
        sw.point_link(self.link, os.path.join(self.weights, "base.pt"))
        before = os.readlink(self.link)
        sw.point_link(self.link, os.path.join(self.other, "tuned.pt"))
        self.assertEqual(self.read(), "tuned")
        back = sw.point_link(self.link, os.path.join(os.path.dirname(self.link), before))
        self.assertEqual((back, self.read()), ("base.pt", "base"))
        sw.point_link(self.link, os.path.join(self.other, "tuned.pt"))           # and from a link with a full path
        before = os.readlink(self.link)
        sw.point_link(self.link, os.path.join(self.weights, "base.pt"))
        sw.point_link(self.link, os.path.join(os.path.dirname(self.link), before))
        self.assertEqual(self.read(), "tuned")

    def test_no_half_made_link_is_left(self):
        sw.point_link(self.link, os.path.join(self.weights, "base.pt"))
        sw.point_link(self.link, os.path.join(self.other, "tuned.pt"))
        self.assertEqual(sorted(os.listdir(self.weights)), ["base.pt", "serving.pt"])


class Tag(unittest.TestCase):
    def test_the_tag_is_the_start_of_the_files_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.pt")
            with open(path, "wb") as f:
                f.write(b"weights")
            import hashlib
            self.assertEqual(sw.weights_tag(path), hashlib.sha256(b"weights").hexdigest()[:12])

    def test_a_box_that_has_the_weights_is_recognised(self):
        self.assertTrue(sw.serves("afno-19m-641de8342da7", "641de8342da7"))
        self.assertFalse(sw.serves("afno-19m-65f9f3bb3270", "641de8342da7"))
        self.assertFalse(sw.serves(None, "641de8342da7"))
        self.assertFalse(sw.serves("mock-0.1", None))        # --no-link: no file, so no tag


class WholeRun(unittest.TestCase):
    """main() against stand-in boxes: no network, no stack. `sw.http` answers every call from a small table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = self.tmp.name
        for name, text in (("base.pt", b"base"), ("tuned.pt", b"tuned")):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(text)
        self.base, self.tuned = os.path.join(folder, "base.pt"), os.path.join(folder, "tuned.pt")
        self.link = os.path.join(folder, "serving.pt")
        self.kills = []
        self.serving = {"a": "afno-19m-" + sw.weights_tag(self.base), "b": "afno-19m-" + sw.weights_tag(self.base)}
        self.kill_answer = (200, {"action": "kill"}, {})
        for name, value in (("http", self.http), ("Traffic", self.NoTraffic)):
            patcher = unittest.mock.patch.object(sw, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    class NoTraffic:
        def __init__(self, router, rps):
            self.stop = False

        def summary(self):
            return {"requests": 0, "ok": 0, "failed": 0, "errors": {}, "failovers": 0, "p50_ms": None, "p95_ms": None,
                    "max_ms": None, "served_by": {}}

    def http(self, url, body=None, timeout=10.0):
        box = "a" if "//a-" in url else "b"
        if url.endswith("/health"):
            return 200, {"status": "ok", "model_version": self.serving[box]}, {}
        if url.endswith("/servers/forecast/kill"):
            self.kills.append(box)
            return self.kill_answer
        return 404, None, {}

    def run_main(self, *extra):
        import contextlib
        import io
        argv = ["swap_weights.py", "--to", self.tuned, "--link", self.link, "--pause", "0", "--timeout", "1",
                "--box", "box-a=http://a-forecast.test,http://a-stack.test",
                "--box", "box-b=http://b-forecast.test,http://b-stack.test", *extra]
        out = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), \
                unittest.mock.patch.object(sw.time, "sleep", lambda s: None):
            with self.assertRaises(SystemExit) as done:
                sw.main()
        return done.exception.code, out.getvalue()

    def test_boxes_that_already_serve_the_weights_are_left_alone_and_a_wrong_link_is_put_right(self):
        tag = sw.weights_tag(self.tuned)
        self.serving = {"a": "afno-19m-" + tag, "b": "afno-19m-" + tag}
        sw.point_link(self.link, self.base)             # a swap that was cut short: the link points elsewhere
        code, out = self.run_main()
        self.assertEqual((code, self.kills), (0, []))
        self.assertEqual(os.path.realpath(self.link), os.path.realpath(self.tuned))
        self.assertIn("pointed at other weights", out)
        code, out = self.run_main()                     # and again: now there is nothing to change
        self.assertEqual((code, self.kills), (0, []))
        self.assertIn("nothing was changed", out)

    def test_a_refused_kill_is_not_followed_by_a_second_one(self):
        sw.point_link(self.link, self.base)
        self.kill_answer = (500, {"detail": "no"}, {})
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(self.kills, ["a"])             # one try on the first box, and no "one more restart"
        self.assertNotIn("restarted once more", out)
        self.assertEqual(os.path.realpath(self.link), os.path.realpath(self.base))      # the link is back


if __name__ == "__main__":
    unittest.main()
