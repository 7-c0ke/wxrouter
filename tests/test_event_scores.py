"""weather/event_scores.py: the yes/no questions, the counts and the four scores. Needs torch (as test_finetune.py)."""
import os
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "weather"))


@unittest.skipIf(torch is None, "needs torch")
class TestEventScores(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import event_scores
        cls.E = event_scores

    def test_the_two_questions(self):
        E = self.E
        start = torch.tensor([[[280.0, 270.0, 273.15, 260.0]]])
        now = torch.tensor([[[272.0, 276.0, 273.15, 265.0]]])
        a = E.answers(now, start)
        self.assertEqual(a["frost"].tolist(), [[[True, False, False, True]]])          # below 0 C; exactly 0 C is not frost
        self.assertEqual(a["big_change"].tolist(), [[[True, True, False, True]]])      # 8, 6, 0 and 5 kelvin away

    def test_counts_are_weighted_by_area(self):
        E = self.E
        said = torch.tensor([[[True, True], [False, False]]])
        truth = torch.tensor([[[True, False], [True, False]]])
        area = torch.tensor([[2.0, 2.0], [0.5, 0.5]])
        self.assertEqual(E.counts(said, truth, area).tolist(), [[2.0, 2.0, 0.5, 0.5]])  # tp, fp, fn, tn

    def test_scores(self):
        s = self.E.scores([30.0, 10.0, 20.0, 40.0])
        self.assertAlmostEqual(s["accuracy"], 0.70)
        self.assertAlmostEqual(s["precision"], 0.75)
        self.assertAlmostEqual(s["recall"], 0.60)
        self.assertAlmostEqual(s["f1"], 2 * 0.75 * 0.60 / (0.75 + 0.60))
        self.assertAlmostEqual(s["base_rate"], 0.50)
        self.assertAlmostEqual(s["wrong"], 0.30)
        never = self.E.scores([0.0, 0.0, 5.0, 95.0])                                   # a forecast that never says yes
        self.assertIsNone(never["precision"])
        self.assertEqual(never["recall"], 0.0)
        self.assertEqual(never["f1"], 0.0)

    def test_the_interval_is_paired_and_by_day(self):
        import numpy as np
        E = self.E
        base = np.array([[5.0, 5.0, 5.0, 85.0]] * 6)
        tuned = np.array([[8.0, 2.0, 2.0, 88.0]] * 6)                                   # better on every start
        low, high = E.interval(base, tuned, ["d1", "d1", "d2", "d2", "d3", "d3"], "f1", resamples=200)
        self.assertGreater(low, 0)
        self.assertAlmostEqual(low, high)                                               # every draw gives the same change
        self.assertAlmostEqual(low, 60.0)                                               # F1 0.5 -> 0.8


if __name__ == "__main__":
    unittest.main()
