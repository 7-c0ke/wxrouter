"""weather/live_refresh.py: which cycle to fetch, and when not to.

    python -m unittest tests/test_live_refresh.py -v

No network: gfs_live.py is replaced by a fake fetch.
"""
import datetime
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("live_refresh_under_test", ROOT / "weather" / "live_refresh.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

T = datetime.datetime


class ExpectedCycle(unittest.TestCase):

    def test_newest_cycle_at_least_four_hours_old(self):
        for now, want in ((T(2026, 11, 14, 20, 48), "2026111412"),   # 18Z is only 2:48 old
                          (T(2026, 11, 14, 22, 0), "2026111418"),
                          (T(2026, 11, 14, 21, 59), "2026111412"),
                          (T(2026, 11, 15, 2, 0), "2026111418"),     # across midnight
                          (T(2026, 11, 15, 3, 59), "2026111418"),
                          (T(2026, 11, 15, 4, 0), "2026111500")):
            with self.subTest(now=now):
                self.assertEqual(L.expected_cycle(now), want)


class Step(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.args = types.SimpleNamespace(out=self.dir.name, data="unused", push=[], lag_h=4.0)
        self.asked = []

    def tearDown(self):
        self.dir.cleanup()

    def touch(self, cycle):
        open(os.path.join(self.dir.name, cycle + ".npy"), "wb").close()

    def fetch(self, code):
        def fake(args, cycle):
            self.asked.append(cycle)
            return code
        return fake

    def test_up_to_date_fetches_nothing(self):
        self.touch("2026111412")
        self.assertEqual(L.step(self.args, T(2026, 11, 14, 20, 48), self.fetch(0)), "up to date")
        self.assertEqual(self.asked, [])

    def test_a_new_cycle_is_fetched(self):
        self.touch("2026111412")
        self.assertEqual(L.step(self.args, T(2026, 11, 14, 22, 5), self.fetch(0)), "fetched 2026111418")
        self.assertEqual(self.asked, ["2026111418"])

    def test_not_posted_yet_is_reported_and_tried_again_next_time(self):
        self.assertEqual(L.step(self.args, T(2026, 11, 14, 22, 5), self.fetch(1)), "not yet 2026111418 (exit 1)")
        self.assertEqual(L.step(self.args, T(2026, 11, 14, 22, 15), self.fetch(0)), "fetched 2026111418")
        self.assertEqual(self.asked, ["2026111418", "2026111418"])

    def test_other_files_in_the_folder_are_ignored(self):
        for name in ("notes.txt", "2026111418.npy.part", "latest.npy"):
            open(os.path.join(self.dir.name, name), "wb").close()
        self.assertIsNone(L.newest_on_disk(self.dir.name))
        self.assertIsNone(L.newest_on_disk(os.path.join(self.dir.name, "missing")))


if __name__ == "__main__":
    unittest.main()
