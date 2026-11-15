"""bench/core_watch.py: the summary of what was watched (the watching itself needs a Linux /proc)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import core_watch  # noqa: E402


class SummaryTest(unittest.TestCase):
    def test_counts_busy_seconds_by_kind_of_core(self):
        rows = [(1.0, 5, 90, "fast"), (2.0, 5, 95, "fast"), (3.0, 2, 3, "slow"),      # idle on a slow core: not counted
                (4.0, 2, 80, "slow"), (5.0, 3, 85, "slow"), (6.0, 7, 99, "fast"), (7.0, 1, 70, "slow")]
        s = core_watch.summary(rows)
        self.assertEqual((s["seconds"], s["busy_seconds"], s["busy_on_fast"], s["busy_on_slow"]), (7, 6, 3, 3))
        self.assertEqual(s["longest_slow_stretch_s"], 2)
        self.assertEqual(s["busy_seconds_by_core"], {1: 1, 2: 1, 3: 1, 5: 2, 7: 1})

    def test_nothing_watched(self):
        self.assertEqual(core_watch.summary([])["longest_slow_stretch_s"], 0)


if __name__ == "__main__":
    unittest.main()
