"""bench/swap_weights.py: the link, which needs no stack.

    python -m unittest tests.test_swap_weights -v
"""
import importlib.util
import os
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
