"""bench/fault_table.py: the parts that need no stack.

The trace and the table's numbers must stay loadgen.py's, so the real table can stand next to the laptop one.
loadgen.py is in the team repo (it needs httpx); where it cannot be imported, those two tests are skipped.

    python -m unittest tests.test_fault_table -v
"""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ft = load("fault_table", os.path.join(ROOT, "bench", "fault_table.py"))
try:
    loadgen = load("team_loadgen", os.path.join(ROOT, "loadgen.py"))
except Exception:                                # not in this checkout, or httpx is missing
    loadgen = None


class Trace(unittest.TestCase):
    def test_same_seed_same_trace(self):
        self.assertEqual(ft.make_trace(42, 30, 8), ft.make_trace(42, 30, 8))
        self.assertNotEqual(ft.make_trace(42, 30, 8), ft.make_trace(43, 30, 8))

    def test_shape(self):
        trace = ft.make_trace(7, 60, 8)
        self.assertTrue(400 < len(trace) < 560)                      # about 8 a second for 60 s
        self.assertEqual(sorted(trace, key=lambda r: r["t"]), trace)
        for row in trace:
            body = row["body"]
            self.assertTrue(6 <= body["hours_ahead"] <= 120)
            self.assertTrue(1 <= body["ensemble_members"] <= 10)
            self.assertEqual(body["step_hours"], 1)

    @unittest.skipIf(loadgen is None, "loadgen.py cannot be imported here")
    def test_equal_to_loadgen(self):
        for seed, duration, rps in ((42, 60, 25), (42, 150, 8), (7, 40, 20)):
            self.assertEqual(ft.make_trace(seed, duration, rps), loadgen.make_trace(seed, duration, rps))


class Numbers(unittest.TestCase):
    ROWS = ([{"status": 200, "latency_ms": float(ms), "served_by": "box-a", "attempts": 1} for ms in range(1, 81)]
            + [{"status": 200, "latency_ms": 500.0 + ms, "served_by": "box-b", "attempts": 2} for ms in range(15)]
            + [{"status": 502, "latency_ms": 9.0, "served_by": "none", "attempts": 2},
               {"status": 0, "latency_ms": 60000.0, "served_by": "none", "attempts": 0, "error": "timeout"}])

    def test_counts(self):
        n = ft.numbers(self.ROWS, ["box-a", "box-b"])
        self.assertEqual((n["requests"], n["errors"], n["failovers"]), (97, 2, 15))
        self.assertEqual(n["error_pct"], round(100.0 * 2 / 97, 2))
        self.assertEqual((n["share_a_pct"], n["share_b_pct"]), (round(100.0 * 80 / 95, 1), round(100.0 * 15 / 95, 1)))
        self.assertEqual(n["max_ms"], 514.0)

    def test_empty(self):
        n = ft.numbers([], ["box-a", "box-b"])
        self.assertEqual((n["requests"], n["errors"], n["error_pct"]), (0, 0, 0.0))
        self.assertNotEqual(n["p95_ms"], n["p95_ms"])                # NaN

    @unittest.skipIf(loadgen is None, "loadgen.py cannot be imported here")
    def test_equal_to_loadgen(self):
        ours = ft.numbers(self.ROWS, ["box-a", "box-b"])
        theirs = loadgen.summarize("balanced", self.ROWS, None)
        for key, _ in ft.COLUMNS:
            if key not in ("policy", "switches"):
                self.assertEqual(ours[key], theirs[key], key)
        self.assertEqual([c for c in ft.COLUMNS], [c for c in loadgen.COLUMNS])


class Events(unittest.TestCase):
    def test_default(self):
        events = ft.parse_events(ft.default_events(150, "box-a", "box-b"))
        self.assertEqual([(e.t, e.action, e.box) for e in events],
                         [(10.0, "slow", "box-a"), (90.0, "normal", "box-a"), (105.0, "kill", "box-b")])

    def test_sorted_and_checked(self):
        events = ft.parse_events("30:kill:box-b, 5:slow:box-a")
        self.assertEqual([e.action for e in events], ["slow", "kill"])
        for bad in ("5:restart:box-a", "x:slow:box-a", "5:slow", "5:slow:box-a:4"):
            with self.assertRaises(SystemExit):
                ft.parse_events(bad)


def sample(t, **more):
    row = {"t": t, "policy": "balanced", "healthy": {"box-a": True, "box-b": True}, "job": "idle",
           "step": None, "training": False, "kill_box_answers": True, "pid": 100}
    row.update(more)
    return row


class Marks(unittest.TestCase):
    EVENTS = ft.parse_events("10:slow:box-a,90:normal:box-a,105:kill:box-b")

    def samples(self):
        rows = [sample(t) for t in (0, 5)]
        rows += [sample(t, job="running") for t in (11, 30)]                          # loading, scoring the base
        rows += [sample(t, job="running", training=True) for t in (44, 60, 89)]
        rows += [sample(t, job="running", training=True, step=20) for t in (90.5, 92)]  # the stop is on its way
        rows += [sample(94, job="failed", step=20, training=True)]                    # exit 3: stopped by a signal
        rows += [sample(104, job="failed")]
        rows += [sample(105.5, job="failed", kill_box_answers=False, pid=None,
                        healthy={"box-a": True, "box-b": False})]
        rows += [sample(110, job="failed", kill_box_answers=False, pid=222, healthy={"box-a": True, "box-b": False})]
        rows += [sample(118, job="failed", pid=222, healthy={"box-a": True, "box-b": False})]
        rows += [sample(119, job="failed", pid=222)]
        return rows

    def test_moments(self):
        m = ft.marks_of(self.samples(), self.EVENTS, "balanced")
        self.assertEqual((m["first_training_step_s"], m["job_stopped_s"]), (44, 94))
        self.assertEqual((m["kill_box_down_s"], m["kill_box_answers_again_s"], m["router_uses_it_again_s"]),
                         (105.5, 118, 119))
        self.assertEqual(m["other_rules_seen"], [])

    def test_windows(self):
        m = ft.marks_of(self.samples(), self.EVENTS, "balanced")
        want = {3: "before", 10: "job starting", 43.9: "job starting", 44: "training", 93.9: "training",
                94: "between", 104.9: "between", 105: "box down", 117.9: "box down", 118: "after", 149: "after"}
        for t, window in want.items():
            self.assertEqual(ft.window_of(t, m, 150), window, t)

    def test_a_fast_restart_is_still_seen(self):
        # A mock server is back within one reading: only its process id shows that it was killed.
        rows = [sample(0), sample(104), sample(105.4, pid=333), sample(106, pid=333)]
        m = ft.marks_of(rows, ft.parse_events("105:kill:box-b"), "balanced")
        self.assertEqual((m["kill_box_down_s"], m["kill_box_answers_again_s"]), (105.4, 105.4))
        self.assertEqual(ft.window_of(50, m, 150), "before")
        self.assertEqual(ft.window_of(105.2, m, 150), "box down")
        self.assertEqual(ft.window_of(120, m, 150), "after")

    def test_a_rule_changed_from_outside_is_reported(self):
        rows = self.samples()
        rows[5]["policy"] = "static"
        self.assertEqual(ft.marks_of(rows, self.EVENTS, "balanced")["other_rules_seen"], ["static"])

    def test_a_job_that_never_started(self):
        rows = [sample(t) for t in (0, 20, 60, 100)]
        m = ft.marks_of(rows, ft.parse_events("10:slow:box-a,90:normal:box-a"), "balanced")
        self.assertEqual((m["first_training_step_s"], m["job_stopped_s"]), (None, None))
        self.assertEqual(ft.window_of(50, m, 150), "job starting")


class Tables(unittest.TestCase):
    def test_markdown(self):
        s = dict(ft.numbers(Numbers.ROWS, ["box-a", "box-b"]), policy="balanced", switches="",
                 windows={"before": ft.numbers(Numbers.ROWS[:80], ["box-a", "box-b"])})
        table = ft.markdown_table([s]).splitlines()
        self.assertEqual(table[0].count("|"), len(ft.COLUMNS) + 1)
        self.assertTrue(table[2].startswith("| balanced | 97 | 2 |"))
        windows = ft.windows_table([s]).splitlines()
        self.assertEqual(windows[0], "| policy | before p95 ms |")
        self.assertTrue(windows[2].startswith("| balanced | 76.0"))


if __name__ == "__main__":
    unittest.main()
