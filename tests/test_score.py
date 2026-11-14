"""Tests for weather/score.py. Written at the hackathon (Nov 13-14, 2026).

Run them from the repo root (about 15 seconds on a laptop; CPU only, at most 4 threads, about 1.2 GB of memory):

    .venv/bin/python -m unittest tests/test_score.py -v

The tests write a small made-up data folder into a temp folder (about 100 MB, removed at the end) and score
small stand-in models on it: an identity model, a model that adds a fixed amount every step, a broken one.
The made-up numbers are chosen so that the right answers can be worked out by hand.

The made-up main file starts on 2019-01-01, not 1979-01-01 (a real-length one would take 69 GB), so the tests
move wx_common.T0_MAIN to that date while they run. One test builds the real model with random weights, to
check that a weights file written the way train.py writes it loads and runs (33 six-hour steps).
"""
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER = os.path.join(os.path.dirname(HERE), "weather")
SCORE = os.path.join(WEATHER, "score.py")
sys.path.insert(0, WEATHER)

import score            # noqa: E402
import wx_common as wx  # noqa: E402

torch.set_num_threads(4)                         # be gentle with the laptop

VARS = ["z500", "t850", "t2m"]
CHANNEL = {"z500": 14, "t850": 5, "t2m": 2}
MEAN = {"z500": 54000.0, "t850": 270.0, "t2m": 280.0}            # made-up statistics; every product with a
STD = {"z500": 800.0, "t850": 16.0, "t2m": 6.0}                  # multiple of 1/8 is exact in float32
GAP = {"z500": 16.0, "t850": 0.5, "t2m": -0.25}  # truth_2020.npy = packed_2020.npy in physical units + GAP
N_ROWS = 31                                      # rows 0..30: the first four starts of a year and their 120 hours
MAIN_FIRST = "2019-01-01T00"                     # the first time in the made-up main file
STORM_START_ROW, STORM_VALID_ROW = 105, 117      # 2019-01-27T06 and 2019-01-30T06, counted from MAIN_FIRST
BOX = (slice(86, 89), slice(179, 183))           # the 12 grid points around Chicago
# Three patterns over the box that each add up to zero, so the box averages below are round numbers.
PATTERN_A = np.array([[-0.25, 0.0, 0.0, 0.25], [0.125, -0.125, 0.125, -0.125], [0.25, -0.25, 0.0, 0.0]])
PATTERN_B = np.array([[0.5, -0.5, 0.25, -0.25], [0.0, 0.0, 0.0, 0.0], [-0.125, 0.125, 0.375, -0.375]])
PATTERN_C = np.array([[1.0, -1.0, 2.0, -2.0], [0.5, -0.5, 0.0, 0.0], [3.0, -3.0, 1.5, -1.5]])

TMP = None               # set by setUpModule
DATA = None              # the made-up data folder
DATA_NO_TRUTH = None     # the same folder without truth_2020.npy
T0_PATCH = None


def clim_offset(slot, hour_index, v):
    """In the made-up climatology, slots 0..7 are the 2020 truth plus this much (v = 0, 1, 2 for z500, t850, t2m)."""
    return (v + 1) * (slot + 1) + hour_index / 4.0


def start_hours(n):
    """Hours after Jan 1 00Z of a year's first n starts: 00Z and 12Z of every second day (0, 12, 48, 60, ...)."""
    return [48 * (i // 2) + 12 * (i % 2) for i in range(n)]


def make_data(folder):
    """Write the made-up data folder. Z-scores are multiples of 1/8, so float16 and float32 hold them exactly."""
    os.makedirs(folder)
    mean = [10.0 * c for c in range(20)]
    std = [1.0 + c for c in range(20)]
    for var in VARS:
        mean[CHANNEL[var]], std[CHANNEL[var]] = MEAN[var], STD[var]
    with open(os.path.join(folder, "stats.json"), "w") as f:
        json.dump({"channels": wx.CHANNEL_NAMES, "mean": mean, "std": std}, f)

    rng = np.random.default_rng(2026)
    spread = (0.5 + 1.5 * np.arange(120) / 119.0).astype(np.float32)[:, None]    # noisier towards the north pole

    def random_z(n):
        z = rng.standard_normal((n, 20, 120, 240), dtype=np.float32) * spread
        return (np.round(z * 8) / 8).astype(np.float16)

    z2020 = random_z(N_ROWS)
    np.save(os.path.join(folder, "packed_2020.npy"), z2020)
    truth = np.stack([MEAN[var] + STD[var] * z2020[:, CHANNEL[var]].astype(np.float64) + GAP[var] for var in VARS],
                     axis=1)
    np.save(os.path.join(folder, "truth_2020.npy"), truth.astype(np.float32))

    # The main file: 118 rows from MAIN_FIRST. Rows 0..30 are random; the two storm rows are set by hand;
    # the rest is never read and stays zero.
    main = np.lib.format.open_memmap(os.path.join(folder, "packed_1979_2019.npy"), mode="w+", dtype=np.float16,
                                     shape=(STORM_VALID_ROW + 1, 20, 120, 240))
    main[:N_ROWS] = random_z(N_ROWS)
    main[STORM_START_ROW, 5] = 30.0                              # far too warm outside the box
    main[STORM_START_ROW, 5][BOX] = -0.5 + PATTERN_A             # box average -0.5: 270 + 16 * -0.5 = 262 K
    main[STORM_VALID_ROW, 5] = 20.0
    main[STORM_VALID_ROW, 5][BOX] = -1.625 + PATTERN_B           # box average -1.625: 244 K
    main.flush()
    del main

    # The climatology: 30 day slots. Slots 0..7 are the 2020 truth plus clim_offset; slot 29 (Jan 30) is the storm's.
    clim = np.lib.format.open_memmap(os.path.join(folder, "clim_1979_2017.npy"), mode="w+", dtype=np.float32,
                                     shape=(30, 4, 3, 120, 240))
    for row in range(N_ROWS):
        slot, hour_index = divmod(row, 4)
        for v in range(3):
            clim[slot, hour_index, v] = truth[row, v] + clim_offset(slot, hour_index, v)
    clim[29] = 500.0                                             # wrong hour or wrong variable: far off
    clim[29, 1, 1] = 1000.0                                      # right map, outside the box: far off
    clim[29, 1, 1][BOX] = 268.0 + PATTERN_C                      # box average 268 K
    clim.flush()
    del clim


def setUpModule():
    global TMP, DATA, DATA_NO_TRUTH, T0_PATCH
    TMP = tempfile.mkdtemp(prefix="wx-test-score-")
    try:
        DATA = os.path.join(TMP, "data")
        make_data(DATA)
        DATA_NO_TRUTH = os.path.join(TMP, "data_no_truth")
        os.makedirs(DATA_NO_TRUTH)
        for name in ("stats.json", "packed_2020.npy", "packed_1979_2019.npy", "clim_1979_2017.npy"):
            os.symlink(os.path.join(DATA, name), os.path.join(DATA_NO_TRUTH, name))
    except BaseException:
        shutil.rmtree(TMP, ignore_errors=True)   # tearDownModule is not called when this function fails
        raise
    T0_PATCH = mock.patch.object(wx, "T0_MAIN", np.datetime64(MAIN_FIRST, "h"))
    T0_PATCH.start()


def tearDownModule():
    T0_PATCH.stop()
    shutil.rmtree(TMP, ignore_errors=True)


def strict_json(text):
    """Parse JSON, refusing NaN and Infinity (which Python would write but other readers reject)."""
    def refuse(name):
        raise ValueError("scores.json holds %s, which is not JSON" % name)
    return json.loads(text, parse_constant=refuse)


def run_score(args, model=None, out=None):
    """Run score.main in this process, on the CPU, with a fresh --out folder (or the one given).
    Returns (exit code, the content of scores.json or None, what it printed, what it wrote to stderr)."""
    if out is None:
        out = tempfile.mkdtemp(prefix="out-", dir=TMP)
    printed, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(errors):
        code = score.main([str(a) for a in args] + ["--out", out, "--device", "cpu"], model=model)
    path = os.path.join(out, "scores.json")
    scores = None
    if os.path.exists(path):
        with open(path) as f:
            scores = strict_json(f.read())
    return code, scores, printed.getvalue(), errors.getvalue()


def by_definition(forecasts, truths):
    """RMSE straight from the definition, in slow plain steps: for each start, the mean over the grid of
    w * (forecast - truth)^2 with w = cos(latitude) / (its mean); then the root of the mean over starts.
    forecasts and truths are lists of (120, 240) maps, one pair per start."""
    cos_lat = [math.cos(math.radians(-88.5 + 1.5 * i)) for i in range(120)]
    mean_cos = sum(cos_lat) / 120
    total = 0.0
    for forecast, truth in zip(forecasts, truths):
        squared = (np.asarray(forecast, np.float64) - np.asarray(truth, np.float64)) ** 2
        along_lon = squared.mean(axis=1)                         # one number per latitude row
        total += sum(c / mean_cos * m for c, m in zip(cos_lat, along_lon)) / 120
    return math.sqrt(total / len(forecasts))


def physical(packed_row, var):
    """One variable of one packed row (20, 120, 240) in physical units, float64."""
    return MEAN[var] + STD[var] * np.asarray(packed_row[CHANNEL[var]], np.float64)


class AddEachStep(torch.nn.Module):
    """A stand-in model: every six-hour step adds a fixed amount to each channel's z-scores."""

    def __init__(self, per_channel):
        super().__init__()
        self.register_buffer("delta", torch.tensor(per_channel, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return x + self.delta


class Broken(torch.nn.Module):
    """A stand-in model whose forecasts blow up."""

    def forward(self, x):
        return x * float("nan")


class Spy(torch.nn.Module):
    """The identity model, but it notes how it is called, and it answers in float64."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x):
        self.calls.append((tuple(x.shape), x.dtype, torch.is_inference_mode_enabled(), self.training))
        return x.double()


def t850_drift(per_step):
    """A model that changes T850 (channel 5) by per_step z-score units every step and leaves the rest alone."""
    delta = [0.0] * 20
    delta[5] = per_step
    return AddEachStep(delta)


class ArithmeticTests(unittest.TestCase):

    def test_weighted_mse_by_hand(self):
        w = wx.area_weights()
        zero = np.zeros((120, 240))
        # The same error everywhere: the weights average to 1, so the MSE is the error squared.
        self.assertAlmostEqual(float(score.weighted_mse(zero + 3.0, zero, w)), 9.0, places=12)
        # The 120 latitudes are -88.5, -87, ..., 90. Their cosines add up to 1 / tan(0.75 degrees) = 76.39.
        # An error of 2 on the equator row only (row 59, cosine 1) costs 4 * (1 / 76.39) = 4 * tan(0.75 degrees).
        tan = math.tan(math.radians(0.75))
        forecast = zero.copy()
        forecast[59] = 2.0
        self.assertAlmostEqual(float(score.weighted_mse(forecast, zero, w)), 4 * tan, places=12)
        self.assertAlmostEqual(4 * tan, 0.05236, places=5)       # an unweighted mean would give 4 / 120 = 0.0333
        # The same error at 60 N (row 99) has half the weight; at the north pole (row 119) it has none.
        forecast = zero.copy()
        forecast[99] = 2.0
        self.assertAlmostEqual(float(score.weighted_mse(forecast, zero, w)), 2 * tan, places=12)
        forecast = zero.copy()
        forecast[119] = 2.0
        self.assertAlmostEqual(float(score.weighted_mse(forecast, zero, w)), 0.0, places=12)
        # Axes in front of the grid are kept: (2 starts, 3 variables).
        forecasts = np.zeros((2, 3, 120, 240))
        forecasts[0, 1] = 3.0
        forecasts[1, 2, 59] = 2.0
        mse = score.weighted_mse(forecasts, np.zeros((2, 3, 120, 240)), w)
        self.assertEqual(mse.shape, (2, 3))
        np.testing.assert_allclose(mse, [[0, 9, 0], [0, 0, 4 * tan]], atol=1e-12)

    def test_rmse_is_root_of_mean_mse_over_starts(self):
        # Two starts with MSE 1 and 7: RMSE = sqrt(4) = 2. (The mean of the two roots would be 1.82.)
        self.assertEqual(score.rmse_over_starts(np.array([[1.0, 0.0], [7.0, 18.0]])).tolist(), [2.0, 3.0])

    def test_leads_and_known_values(self):
        self.assertEqual(score.LEADS_H, list(range(6, 121, 6)))
        self.assertEqual(score.VARS, VARS)
        self.assertEqual(wx.SCORED_IDX, [14, 5, 2])
        self.assertEqual(score.KNOWN[2020]["model"]["z500"], {24: 102.18, 72: 303.86, 120: 543.81})
        self.assertEqual(score.KNOWN[2020]["persistence"]["t850"], {24: 3.051, 72: 4.326, 120: 4.592})
        self.assertEqual(score.KNOWN[2020]["climatology"]["t2m"], {24: 2.657, 72: 2.657, 120: 2.655})
        self.assertEqual(score.KNOWN[2019]["model"], {"z500": {72: 296.86}, "t850": {72: 1.6607}, "t2m": {72: 1.4431}})

    def fake_year(self, year, model_factor, baseline_factor, n_starts=None):
        """A year's result whose scores are the known values times a factor (1.0 where nothing is known)."""
        full = len(wx.test_starts(year))
        rmse = {}
        for kind in score.KINDS:
            factor = model_factor if kind == "model" else baseline_factor
            rmse[kind] = None
            if factor is not None:
                rmse[kind] = {}
                for var in VARS:
                    values = [1.0] * 20
                    for lead, known in score.KNOWN.get(year, {}).get(kind, {}).get(var, {}).items():
                        values[lead // 6 - 1] = known * factor
                    rmse[kind][var] = values
        result = {"n_starts": full if n_starts is None else n_starts, "n_starts_full": full, "rmse": rmse}
        score.add_verdicts(year, result)
        return result

    def test_verdicts_band_and_baseline_tolerance(self):
        yes = {"z500": True, "t850": True, "t2m": True}
        no = {"z500": False, "t850": False, "t2m": False}
        nothing = {"z500": None, "t850": None, "t2m": None}
        # The band for "same skill" is +-2% at 72 h.
        self.assertEqual(self.fake_year(2020, 1.019, 1.0)["same_skill_72h"], yes)
        self.assertEqual(self.fake_year(2020, 0.981, 1.0)["same_skill_72h"], yes)
        self.assertEqual(self.fake_year(2020, 1.021, 1.0)["same_skill_72h"], no)
        self.assertEqual(self.fake_year(2020, 0.979, 1.0)["same_skill_72h"], no)
        self.assertEqual(self.fake_year(2019, 1.019, 1.0)["same_skill_72h"], yes)
        self.assertEqual(self.fake_year(2019, 1.021, 1.0)["same_skill_72h"], no)
        self.assertEqual(self.fake_year(2020, float("nan"), 1.0)["same_skill_72h"], no)
        self.assertEqual(self.fake_year(2020, None, 1.0)["same_skill_72h"], nothing)         # --no-model
        # A baseline may be 0.5% from its known value, not more.
        self.assertIs(self.fake_year(2020, 1.0, 1.004)["baselines_ok"], True)
        self.assertIs(self.fake_year(2020, 1.0, 0.996)["baselines_ok"], True)
        self.assertIs(self.fake_year(2020, 1.0, 1.006)["baselines_ok"], False)
        self.assertIs(self.fake_year(2020, 1.0, 0.994)["baselines_ok"], False)
        self.assertIsNone(self.fake_year(2019, 1.0, 1.5)["baselines_ok"])                    # nothing known for 2019
        # The rows behind the verdicts.
        result = self.fake_year(2020, 1.01, 1.006)
        self.assertEqual(len(result["compare"]), 27)                                         # 3 kinds x 3 x 3 leads
        rows = dict(((r["kind"], r["variable"], r["lead_h"]), r) for r in result["compare"])
        row = rows[("model", "t850", 72)]
        self.assertEqual((row["known"], row["tolerance_pct"], row["ok"]), (1.6809, 2.0, True))
        self.assertAlmostEqual(row["tonight"], 1.6809 * 1.01)
        self.assertAlmostEqual(row["diff_pct"], 1.0)
        self.assertIsNone(rows[("model", "t850", 24)]["ok"])                                 # only 72 h is judged
        self.assertAlmostEqual(rows[("model", "t850", 24)]["diff_pct"], 1.0)
        row = rows[("persistence", "z500", 120)]
        self.assertEqual((row["known"], row["tolerance_pct"], row["ok"]), (1051.0, 0.5, False))
        # A year cut short by --limit cannot be compared: no verdicts at all.
        result = self.fake_year(2020, 1.0, 1.0, n_starts=4)
        self.assertIs(result["comparable"], False)
        self.assertEqual(result["same_skill_72h"], nothing)
        self.assertIsNone(result["baselines_ok"])
        self.assertTrue(all(r["ok"] is None for r in result["compare"]))
        self.assertAlmostEqual(result["compare"][0]["diff_pct"], 0.0)

    def printed_year(self, year, result):
        """What print_year prints for a fake_year result."""
        starts = wx.test_starts(year)[:result["n_starts"]]
        result.update({"first_start": str(starts[0]), "last_start": str(starts[-1]),
                       "truth": "some_file.npy", "seconds": 95.2})
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            score.print_year(year, result)
        return printed.getvalue()

    def test_table_and_verdict_lines(self):
        # A full year, 1% above the reference model, baselines 0.1% off: the good case of the real run.
        text = self.printed_year(2020, self.fake_year(2020, 1.01, 1.001))
        self.assertIn("2020: 362 of 362 starts, 2020-01-01T00 to 2020-12-26T12; truth from some_file.npy; 95 s", text)
        self.assertIn("  model       z500 m2/s2     103.20    102.18     +1.0%    306.90    303.86     +1.0%"
                      "    549.25    543.81     +1.0%\n", text)
        self.assertIn("  climatology t2m  K         2.6597    2.6570     +0.1%    2.6597    2.6570     +0.1%"
                      "    2.6577    2.6550     +0.1%\n", text)
        self.assertIn('within +-2% of the reference model ("same skill")?  '
                      "z500 yes (+1.0%),  t850 yes (+1.0%),  t2m yes (+1.0%)\n", text)
        self.assertIn("Baselines: all 18 agree with their known values to within 0.5%.\n", text)
        for word in ("NO", "DIFFER", "No verdicts", "cut short"):
            self.assertNotIn(word, text)
        # Three header and nine table lines, then the two verdict lines; every line fits a terminal.
        self.assertEqual(len(text.strip("\n").split("\n")), 3 + 9 + 2)
        self.assertLessEqual(max(len(line) for line in text.split("\n")), 118)
        # 3% above the reference: not the same skill. A baseline 1% off: flagged, row by row.
        text = self.printed_year(2020, self.fake_year(2020, 1.03, 1.01))
        self.assertIn("z500 NO (+3.0%),  t850 NO (+3.0%),  t2m NO (+3.0%)", text)
        self.assertIn("BASELINES DIFFER from their known values by more than 0.5%. The scorer may be wrong:", text)
        self.assertIn("      climatology t850 at 120 h: 3.4986 against 3.4640 (+1.0%)\n", text)
        self.assertEqual(text.count(" against "), 18)
        # 2019 has a reference for the model at 72 h only, and no known baselines.
        text = self.printed_year(2019, self.fake_year(2019, 0.99, 1.0))
        self.assertIn("  model       z500 m2/s2       1.00         -         -    293.89    296.86     -1.0%"
                      "      1.00         -         -\n", text)
        self.assertIn("z500 yes (-1.0%),  t850 yes (-1.0%),  t2m yes (-1.0%)", text)
        self.assertIn("Baselines: there are no known values for 2019.", text)
        # Baselines only (--no-model): no model rows, no 72-hour line.
        text = self.printed_year(2020, self.fake_year(2020, None, 1.0))
        self.assertNotIn("model", text)
        self.assertIn("Baselines: all 18 agree", text)
        # Cut short by --limit: the numbers, but no verdicts.
        text = self.printed_year(2020, self.fake_year(2020, 1.0, 1.0, n_starts=4))
        self.assertIn("2020: 4 of 362 starts (cut short by --limit), 2020-01-01T00 to 2020-01-03T12;", text)
        self.assertIn("Only 4 of 362 starts were scored", text)
        self.assertNotIn("same skill", text)
        self.assertNotIn("Baselines", text)

    def test_describe_checkpoint(self):
        self.assertEqual(score.describe_checkpoint({"stage": 2, "step": 1564, "complete": True}),
                         "stage 2, step 1564, complete")
        self.assertEqual(score.describe_checkpoint({"stage": 2, "step": 3, "complete": False}),
                         "stage 2, step 3, NOT complete")
        self.assertEqual(score.describe_checkpoint({"stage": None, "step": None, "complete": None}), "")

    def test_real_years_fit_the_real_files(self):
        # With the real first time of the main file: every start of 2019 and 2020 and its 120 hours lie inside
        # its file (59,900 and 1,464 rows), and every valid time has a slot in the climatology (366 days x 4).
        with mock.patch.object(wx, "T0_MAIN", np.datetime64("1979-01-01T00", "h")):
            for year, count, which, n_rows in ((2020, 362, "2020", 1464), (2019, 360, "main", 59900)):
                starts = wx.test_starts(year)
                self.assertEqual(len(starts), count)
                self.assertEqual(str(starts[0]), "%d-01-01T00" % year)
                self.assertEqual(str(starts[3]), "%d-01-03T12" % year)
                places = [wx.locate(t) for t in starts]
                self.assertEqual(set(name for name, _ in places), {which})
                self.assertGreaterEqual(min(row for _, row in places), 0)
                self.assertLess(max(row for _, row in places) + score.N_STEPS, n_rows)
                slots = set()
                for t in starts:
                    for k in range(1, score.N_STEPS + 1):
                        slots.add(wx.calendar_slot(t + np.timedelta64(6 * k, "h")))
                self.assertTrue(all(0 <= slot <= 365 and 0 <= hour <= 3 for slot, hour in slots))
                self.assertEqual((59, 0) in slots, year == 2020)                 # Feb 29 exists in 2020 only
            self.assertEqual(wx.locate("2019-01-01T00"), ("main", 58440))
            self.assertEqual(wx.locate(score.STORM_START), ("main", 58545))
            self.assertEqual(wx.calendar_slot("2019-01-30T06"), (29, 1))         # the storm's valid time
        self.assertEqual((wx.LATS[86], wx.LATS[88], wx.LONS[179], wx.LONS[182]), (40.5, 43.5, 268.5, 273.0))

    def test_json_ready(self):
        got = score.json_ready({2020: {72: np.float64(1.5), "bad": float("nan"), "inf": np.float32("inf")},
                                "n": np.int64(3), "ok": np.bool_(True), "t": (1, 2), "none": None, "s": "x"})
        self.assertEqual(got, {"2020": {"72": 1.5, "bad": None, "inf": None}, "n": 3, "ok": True, "t": [1, 2],
                               "none": None, "s": "x"})
        self.assertIs(type(got["n"]), int)
        self.assertIs(type(got["ok"]), bool)
        # Anything that is not a JSON type is written as its text, so the file can always be written.
        self.assertEqual(score.json_ready({"when": np.datetime64("2020-01-01T00", "h")}), {"when": "2020-01-01T00"})


class ScoringTests(unittest.TestCase):

    def test_identity_model_scores_exactly_as_persistence(self):
        # 2019 is scored against the packed file, so "output = input" is the persistence forecast, bit for bit.
        code, scores, _, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 4], model=torch.nn.Identity())
        self.assertEqual(code, 0)
        year = scores["years"]["2019"]
        self.assertEqual(year["rmse"]["model"], year["rmse"]["persistence"])
        self.assertTrue(year["truth"].startswith("packed_1979_2019.npy"))
        for var in VARS:
            self.assertEqual(len(year["rmse"]["model"][var]), 20)
            self.assertTrue(all(v > 0 for v in year["rmse"]["model"][var]))
        # And it is not the climatology's score.
        self.assertNotEqual(year["rmse"]["model"], year["rmse"]["climatology"])

    def test_rollout_is_done_the_way_training_validated_it(self):
        # Eval mode, inference mode, float32 in at every step (the model's float64 answer is cast back),
        # 20 steps for each batch of starts and 12 for the storm.
        spy = Spy()
        self.assertTrue(spy.training)
        code, scores, _, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 3, "--batch", 2, "--storm"],
                                       model=spy)
        self.assertEqual(code, 0)
        shapes = [call[0] for call in spy.calls]
        self.assertEqual(shapes, [(2, 20, 120, 240)] * 20 + [(1, 20, 120, 240)] * 20 + [(1, 20, 120, 240)] * 12)
        self.assertEqual(set(call[1:] for call in spy.calls), {(torch.float32, True, False)})
        self.assertFalse(torch.is_inference_mode_enabled())      # and it is switched off again afterwards
        year = scores["years"]["2019"]
        self.assertEqual(year["rmse"]["model"], year["rmse"]["persistence"])
        # Dropout does nothing in eval mode: it scores as the identity. (In training mode it would not.)
        _, scores, _, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 1], model=torch.nn.Dropout(0.5))
        year = scores["years"]["2019"]
        self.assertEqual(year["rmse"]["model"], year["rmse"]["persistence"])

    def test_cuda_branch_of_the_rollout_runs(self):
        # There is no CUDA GPU on a laptop, so this cannot check bf16 numbers. It only makes sure the lines of
        # the CUDA branch run: without CUDA, torch.autocast("cuda", ...) warns and switches itself off.
        forecaster = score.Forecaster(torch.nn.Identity(), torch.device("cpu"))
        self.assertFalse(forecaster.bf16)
        forecaster.bf16 = True
        forecaster.start(np.full((2, 20, 120, 240), 0.5, np.float16))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = forecaster.step()
            z = forecaster.step()
        self.assertEqual((z.shape, z.dtype), ((2, 3, 120, 240), np.float32))
        self.assertTrue((z == 0.5).all())

    def test_persistence_and_climatology_from_the_definition(self):
        code, scores, _, _ = run_score(["--data", DATA, "--years", 2020, 2019, "--limit", 4, "--no-model"])
        self.assertEqual(code, 0)
        truth_2020 = np.load(os.path.join(DATA, "truth_2020.npy"), mmap_mode="r")
        main = np.load(os.path.join(DATA, "packed_1979_2019.npy"), mmap_mode="r")
        clim = np.load(os.path.join(DATA, "clim_1979_2017.npy"), mmap_mode="r")
        rows = [h // 6 for h in start_hours(4)]                  # 0, 2, 8, 10
        self.assertEqual(scores["years"]["2020"]["truth"], "truth_2020.npy")
        for k in (1, 4, 12, 20):                                 # 6, 24, 72 and 120 hours
            for v, var in enumerate(VARS):
                # 2020: the truth is truth_2020.npy.
                truths = [truth_2020[r + k, v] for r in rows]
                want = by_definition([truth_2020[r, v] for r in rows], truths)
                got = scores["years"]["2020"]["rmse"]["persistence"][var][k - 1]
                self.assertAlmostEqual(got / want, 1.0, places=10, msg="2020 persistence %s step %d" % (var, k))
                maps = [clim[(r + k) // 4, (r + k) % 4, v] for r in rows]
                want = by_definition(maps, truths)
                got = scores["years"]["2020"]["rmse"]["climatology"][var][k - 1]
                self.assertAlmostEqual(got / want, 1.0, places=10, msg="2020 climatology %s step %d" % (var, k))
                # 2019: the truth is the packed file, turned into physical units.
                truths = [physical(main[r + k], var) for r in rows]
                want = by_definition([physical(main[r], var) for r in rows], truths)
                got = scores["years"]["2019"]["rmse"]["persistence"][var][k - 1]
                self.assertAlmostEqual(got / want, 1.0, places=10, msg="2019 persistence %s step %d" % (var, k))
                want = by_definition(maps, truths)               # Jan 1-8 has the same slots in every year
                got = scores["years"]["2019"]["rmse"]["climatology"][var][k - 1]
                self.assertAlmostEqual(got / want, 1.0, places=10, msg="2019 climatology %s step %d" % (var, k))

    def test_climatology_on_made_up_climatology(self):
        # In slots 0..7 the made-up climatology is the 2020 truth plus clim_offset(slot, hour, variable), the same
        # at every grid point. So its RMSE is the root of the mean over starts of that offset squared.
        code, scores, _, _ = run_score(["--data", DATA, "--years", 2020, "--limit", 4, "--no-model"])
        self.assertEqual(code, 0)
        got = scores["years"]["2020"]["rmse"]["climatology"]
        for v, var in enumerate(VARS):
            for k in range(1, 21):
                squares = []
                for h0 in start_hours(4):
                    valid = h0 + 6 * k                           # hours after 2020-01-01T00
                    squares.append(clim_offset(valid // 24, valid % 24 // 6, v) ** 2)
                want = math.sqrt(sum(squares) / 4)
                self.assertAlmostEqual(got[var][k - 1], want, places=9, msg="%s step %d" % (var, k))
        # One of them by hand: 24 hours after the four starts it is Jan 2 00Z, Jan 2 12Z, Jan 4 00Z, Jan 4 12Z,
        # so the T850 offsets are 2*2, 2*2 + 0.5, 2*4, 2*4 + 0.5: RMSE = sqrt((16 + 20.25 + 64 + 72.25) / 4).
        self.assertAlmostEqual(got["t850"][3], math.sqrt(172.5 / 4), places=9)

        # With one start only: 24 hours after Jan 1 00Z is slot 1, hour 0: offsets 2, 4 and 6.
        code, scores, _, _ = run_score(["--data", DATA, "--years", 2020, "--limit", 1, "--no-model"])
        got = scores["years"]["2020"]["rmse"]["climatology"]
        self.assertAlmostEqual(got["z500"][3], 2.0, places=9)
        self.assertAlmostEqual(got["t850"][3], 4.0, places=9)
        self.assertAlmostEqual(got["t2m"][3], 6.0, places=9)

    def test_truth_for_2020_is_the_truth_file_when_it_is_there(self):
        _, with_file, _, _ = run_score(["--data", DATA, "--years", 2020, "--limit", 2, "--no-model"])
        code, without, _, _ = run_score(["--data", DATA_NO_TRUTH, "--years", 2020, "--limit", 2, "--no-model"])
        self.assertEqual(code, 0)
        self.assertEqual(with_file["years"]["2020"]["truth"], "truth_2020.npy")
        self.assertTrue(without["years"]["2020"]["truth"].startswith("packed_2020.npy"))
        # Persistence does not notice (the made-up truth file is the packed file plus a constant) ...
        for var in VARS:
            np.testing.assert_allclose(without["years"]["2020"]["rmse"]["persistence"][var],
                                       with_file["years"]["2020"]["rmse"]["persistence"][var], rtol=1e-10)
        # ... but the climatology does: against the packed file it is off by clim_offset + GAP.
        for v, var in enumerate(VARS):
            for k in (1, 12, 20):
                squares = [(clim_offset((h0 + 6 * k) // 24, (h0 + 6 * k) % 24 // 6, v) + GAP[var]) ** 2
                           for h0 in start_hours(2)]
                self.assertAlmostEqual(without["years"]["2020"]["rmse"]["climatology"][var][k - 1],
                                       math.sqrt(sum(squares) / 2), places=9)
        # The model always starts from the packed file. With the truth file there, the identity model's error
        # is (packed file at the start) - (truth file at the valid time); without it, it is persistence again.
        _, a, _, _ = run_score(["--data", DATA, "--years", 2020, "--limit", 2], model=torch.nn.Identity())
        _, b, _, _ = run_score(["--data", DATA_NO_TRUTH, "--years", 2020, "--limit", 2], model=torch.nn.Identity())
        packed = np.load(os.path.join(DATA, "packed_2020.npy"), mmap_mode="r")
        truth = np.load(os.path.join(DATA, "truth_2020.npy"), mmap_mode="r")
        rows = [h // 6 for h in start_hours(2)]
        for v, var in enumerate(VARS):
            for k in (1, 12, 20):
                want = by_definition([physical(packed[r], var) for r in rows], [truth[r + k, v] for r in rows])
                self.assertAlmostEqual(a["years"]["2020"]["rmse"]["model"][var][k - 1] / want, 1.0, places=10)
        self.assertNotEqual(a["years"]["2020"]["rmse"]["model"], a["years"]["2020"]["rmse"]["persistence"])
        self.assertEqual(b["years"]["2020"]["rmse"]["model"], b["years"]["2020"]["rmse"]["persistence"])

    def test_limit(self):
        truth_2020 = np.load(os.path.join(DATA, "truth_2020.npy"), mmap_mode="r")
        for limit in (1, 3, 4):
            code, scores, printed, _ = run_score(["--data", DATA, "--years", 2020, 2019, "--limit", limit,
                                                  "--no-model"])
            self.assertEqual(code, 0)
            self.assertEqual(scores["limit"], limit)
            for year, full in (("2020", 362), ("2019", 360)):
                self.assertEqual(scores["years"][year]["n_starts"], limit)
                self.assertEqual(scores["years"][year]["n_starts_full"], full)
                self.assertIs(scores["years"][year]["comparable"], False)
                self.assertIsNone(scores["years"][year]["baselines_ok"])
                self.assertEqual(scores["years"][year]["first_start"], year + "-01-01T00")
            last = {1: "2020-01-01T00", 3: "2020-01-03T00", 4: "2020-01-03T12"}[limit]
            self.assertEqual(scores["years"]["2020"]["last_start"], last)
            self.assertIn("%d of 362 starts" % limit, printed)
            self.assertIn("No verdicts", printed)
            # Only the first `limit` starts count: 72 hours of persistence, from the definition.
            rows = [h // 6 for h in start_hours(limit)]
            want = by_definition([truth_2020[r, 0] for r in rows], [truth_2020[r + 12, 0] for r in rows])
            self.assertAlmostEqual(scores["years"]["2020"]["rmse"]["persistence"]["z500"][11] / want, 1.0, places=10)

    def test_progress_lines(self):
        # A long year says how far it is every 20 seconds. Here: after every batch but the last.
        with mock.patch.object(score, "PROGRESS_EVERY_S", 0):
            code, _, printed, _ = run_score(["--data", DATA, "--years", 2020, "--limit", 3, "--batch", 1,
                                             "--no-model"])
        self.assertEqual(code, 0)
        self.assertRegex(printed, r"  2020: 1 of 3 starts done, \d+ s\n")
        self.assertRegex(printed, r"  2020: 2 of 3 starts done, \d+ s\n")
        self.assertNotIn("3 of 3 starts done", printed)
        _, _, printed, _ = run_score(["--data", DATA, "--years", 2020, "--limit", 3, "--batch", 1, "--no-model"])
        self.assertNotIn("starts done", printed)

    def test_small_files_need_limit(self):
        # Without --limit the starts run past the end of the small files: a clear message, no scores.json.
        code, scores, _, errors = run_score(["--data", DATA, "--years", 2020, "--no-model"])
        self.assertEqual(code, 1)
        self.assertIsNone(scores)
        self.assertIn("FAILED", errors)
        self.assertIn("2020-01-05T00", errors)                   # the fifth start is the first that does not fit
        self.assertIn("only 31 rows", errors)
        self.assertIn("--limit", errors)

    def test_problems_with_the_data_show_before_any_scoring(self):
        # Three starts of 2020 fit in the small files; all 360 of 2019 do not (the made-up main file has 118
        # rows). The run must stop before the model is asked for a single step, not after 2020 has been scored.
        real = wx.test_starts
        spy = Spy()
        with mock.patch.object(wx, "test_starts", lambda year: real(year)[:3] if year == 2020 else real(year)):
            code, scores, printed, errors = run_score(["--data", DATA, "--years", 2020, 2019], model=spy)
        self.assertEqual((code, scores, spy.calls), (1, None, []))
        self.assertIn("2019: the start 2019-01-25T12 and its 120 hours need rows 98 to 118, but only 118 rows are "
                      "there (packed_1979_2019.npy). Use --limit to score fewer starts.", errors)
        self.assertNotIn("2020:", printed)
        # The same for a storm that is not in the file (here: moved to mid February, rows 181 to 193).
        spy = Spy()
        with mock.patch.object(score, "STORM_START", "2019-02-15T06"):
            code, scores, _, errors = run_score(["--data", DATA, "--years", 2020, "--limit", 1, "--storm"],
                                                model=spy)
        self.assertEqual((code, scores, spy.calls), (1, None, []))
        self.assertIn("the storm forecast needs rows 181 to 193, but packed_1979_2019.npy has only 118 rows", errors)
        # And for a climatology file that ends too early. The made-up one has 30 day slots: Jan 1 to Jan 30.
        data = score.Data(DATA)
        self.assertEqual(data.check_slot(np.datetime64("2020-01-30T06", "h")), (29, 1))
        with self.assertRaises(score.ScoreError) as caught:
            data.check_slot(np.datetime64("2020-01-31T00", "h"))
        self.assertIn("has 30 day slots, but 2020-01-31T00 needs slot 30", str(caught.exception))
        with mock.patch.object(data, "truth_rows", return_value=10 ** 6):        # as if the packed file were long
            plan = score.plan_year(2020, data, 26)               # the 26th start, Jan 25 12Z, ends on Jan 30
            self.assertEqual((len(plan["starts"]), plan["n_starts_full"], plan["which"]), (26, 362, "2020"))
            self.assertEqual(plan["rows"][:5], [0, 2, 8, 10, 16])
            with self.assertRaises(score.ScoreError) as caught:
                score.plan_year(2020, data, 27)                  # the 27th, Jan 27 00Z, ends on Feb 1
        self.assertIn("2020-02-01T00 needs slot 31", str(caught.exception))

    def test_batch_size_does_not_change_the_scores(self):
        model = AddEachStep([0.125 * (c % 3 - 1) for c in range(20)])
        results = []
        for batch in (1, 2, 16):
            code, scores, _, _ = run_score(["--data", DATA, "--years", 2020, 2019, "--limit", 3, "--batch", batch],
                                           model=model)
            self.assertEqual(code, 0)
            self.assertEqual(scores["batch"], batch)
            results.append(scores)
        for year in ("2020", "2019"):
            for kind in score.KINDS:
                for var in VARS:
                    for other in results[1:]:
                        np.testing.assert_allclose(other["years"][year]["rmse"][kind][var],
                                                   results[0]["years"][year]["rmse"][kind][var], rtol=1e-12)

    def test_model_that_drifts_in_one_channel(self):
        # A model that warms T850 by 1/8 z-score unit per step forecasts, after k steps, the start state plus
        # k * 16 / 8 = 2k kelvin everywhere. Check its score against the definition.
        code, scores, _, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 2], model=t850_drift(0.125))
        self.assertEqual(code, 0)
        main = np.load(os.path.join(DATA, "packed_1979_2019.npy"), mmap_mode="r")
        rows = [h // 6 for h in start_hours(2)]
        for k in (1, 7, 20):
            want = by_definition([physical(main[r], "t850") + 2.0 * k for r in rows],
                                 [physical(main[r + k], "t850") for r in rows])
            self.assertAlmostEqual(scores["years"]["2019"]["rmse"]["model"]["t850"][k - 1] / want, 1.0, places=10)
        # The other two variables were left alone: they still score as persistence.
        for var in ("z500", "t2m"):
            self.assertEqual(scores["years"]["2019"]["rmse"]["model"][var],
                             scores["years"]["2019"]["rmse"]["persistence"][var])
        self.assertNotEqual(scores["years"]["2019"]["rmse"]["model"]["t850"],
                            scores["years"]["2019"]["rmse"]["persistence"]["t850"])

    def test_storm_share_on_made_up_data(self):
        # Box averages of T850: start 262 K, truth three days later 244 K, climatology 268 K.
        # A model that cools the box by 1/16 z-score unit (1 K) per step forecasts 262 - 12 = 250 K:
        # share caught = (250 - 268) / (244 - 268) = 0.75.
        code, scores, printed, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 1, "--storm"],
                                             model=t850_drift(-1.0 / 16))
        self.assertEqual(code, 0)
        storm = scores["storm"]
        self.assertEqual((storm["start"], storm["valid"], storm["lead_h"]), ("2019-01-27T06", "2019-01-30T06", 72))
        self.assertEqual((storm["lat_index"], storm["lon_index"], storm["n_points"]), ([86, 88], [179, 182], 12))
        self.assertEqual((storm["lat_deg"], storm["lon_deg"]), ([40.5, 43.5], [268.5, 273.0]))
        self.assertAlmostEqual(storm["truth_k"], 244.0, places=9)
        self.assertAlmostEqual(storm["climatology_k"], 268.0, places=9)
        self.assertAlmostEqual(storm["forecast_k"], 250.0, places=9)
        self.assertAlmostEqual(storm["share_caught"], 0.75, places=9)
        self.assertEqual(storm["reference_range"], [0.88, 0.97])
        self.assertIs(storm["in_reference_range"], False)
        self.assertIn("0.750", printed)
        self.assertIn("OUTSIDE", printed)

        # Doing nothing (the identity model) keeps 262 K: (262 - 268) / (244 - 268) = 0.25.
        _, scores, _, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 1, "--storm"],
                                    model=torch.nn.Identity())
        self.assertAlmostEqual(scores["storm"]["forecast_k"], 262.0, places=9)
        self.assertAlmostEqual(scores["storm"]["share_caught"], 0.25, places=9)

        # Cooling by 1/12 unit per step gives 246 K: 22/24 = 0.917, inside the reference range.
        _, scores, printed, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 1, "--storm"],
                                          model=t850_drift(-1.0 / 12))
        self.assertAlmostEqual(scores["storm"]["share_caught"], 22.0 / 24, places=5)
        self.assertIs(scores["storm"]["in_reference_range"], True)
        self.assertIn("inside", printed)

        # --limit does not touch the storm, and without a model there is truth and climatology but no share.
        _, scores, printed, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 1, "--storm", "--no-model"])
        storm = scores["storm"]
        self.assertAlmostEqual(storm["truth_k"], 244.0, places=9)
        self.assertAlmostEqual(storm["climatology_k"], 268.0, places=9)
        self.assertEqual((storm["forecast_k"], storm["share_caught"], storm["in_reference_range"]),
                         (None, None, None))
        self.assertIn("no model", printed)

        # Without --storm there is no storm block.
        _, scores, _, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 1, "--no-model"])
        self.assertIsNone(scores["storm"])

    def test_shape_of_scores_json(self):
        # The command line of the overnight driver's smoke run, with a stand-in model.
        weights = os.path.join(TMP, "not-read.pt")
        out = os.path.join(TMP, "out-shape")
        code, scores, printed, errors = run_score(
            ["--weights", weights, "--data", DATA, "--years", 2020, 2019, "--storm", "--limit", 4],
            model=torch.nn.Identity(), out=out)
        self.assertEqual(code, 0, errors)
        self.assertEqual(os.listdir(out), ["scores.json"])                       # nothing else, no .tmp left
        self.assertEqual(set(scores), {
            "what", "written", "weights", "checkpoint", "device", "bf16", "torch", "data", "batch", "limit",
            "seconds", "variables", "units", "leads_h", "years", "storm", "reference", "warnings"})
        self.assertEqual(scores["weights"], weights)
        self.assertEqual((scores["device"], scores["bf16"], scores["batch"], scores["limit"]), ("cpu", False, 16, 4))
        self.assertEqual(scores["torch"], torch.__version__)
        self.assertEqual(scores["data"], DATA)
        self.assertEqual(scores["variables"], VARS)
        self.assertEqual(scores["units"], {"z500": "m2/s2", "t850": "K", "t2m": "K"})
        self.assertEqual(scores["leads_h"], list(range(6, 121, 6)))
        self.assertGreaterEqual(scores["seconds"], 0)
        self.assertEqual(scores["warnings"], [])
        self.assertEqual(list(scores["years"]), ["2020", "2019"])                # in the order asked for
        for name, year in scores["years"].items():
            self.assertEqual(set(year), {
                "n_starts", "n_starts_full", "first_start", "last_start", "leads_h", "truth", "rmse", "seconds",
                "comparable", "compare", "same_skill_72h", "baselines_ok"}, name)
            self.assertEqual(year["n_starts"], 4)
            self.assertEqual(year["leads_h"], list(range(6, 121, 6)))
            self.assertEqual(set(year["rmse"]), {"model", "persistence", "climatology"})
            for kind in year["rmse"]:
                self.assertEqual(list(year["rmse"][kind]), VARS)
                for var in VARS:
                    values = year["rmse"][kind][var]
                    self.assertEqual(len(values), 20)
                    self.assertTrue(all(type(v) is float and math.isfinite(v) and v > 0 for v in values))
            self.assertEqual(len(year["compare"]), 27)
            self.assertEqual(set(year["compare"][0]),
                             {"kind", "variable", "lead_h", "tonight", "known", "diff_pct", "tolerance_pct", "ok"})
            self.assertEqual(year["same_skill_72h"], {"z500": None, "t850": None, "t2m": None})
        self.assertEqual(scores["years"]["2020"]["truth"], "truth_2020.npy")
        self.assertIn("packed_1979_2019.npy", scores["years"]["2019"]["truth"])
        self.assertEqual(set(scores["storm"]), {
            "start", "valid", "lead_h", "variable", "units", "lat_index", "lon_index", "lat_deg", "lon_deg",
            "n_points", "forecast_k", "truth_k", "climatology_k", "share_caught", "reference_range",
            "in_reference_range"})
        reference = scores["reference"]
        self.assertEqual((reference["band_pct"], reference["baseline_tolerance_pct"]), (2.0, 0.5))
        self.assertEqual(reference["storm_share_range"], [0.88, 0.97])
        self.assertEqual(reference["rmse"]["2020"]["model"]["z500"], {"24": 102.18, "72": 303.86, "120": 543.81})
        self.assertEqual(reference["rmse"]["2020"]["persistence"]["t2m"], {"24": 2.035, "72": 2.935, "120": 3.215})
        self.assertEqual(reference["rmse"]["2020"]["climatology"]["z500"], {"24": 824.8, "72": 825.3, "120": 825.9})
        self.assertEqual(reference["rmse"]["2019"]["model"]["t2m"], {"72": 1.4431})
        # The table: one block per year with the three leads, then the storm, then where the file went.
        for text in ("2020: 4 of 362 starts", "2019: 4 of 360 starts", "24 h", "72 h", "120 h", "persistence",
                     "climatology", "303.86", "Storm", "wrote " + os.path.join(out, "scores.json")):
            self.assertIn(text, printed)

    def test_full_year_gets_verdicts(self):
        # Pretend a year has three starts only, so that the small files hold a "full" year.
        real = wx.test_starts
        with mock.patch.object(wx, "test_starts", lambda year: real(year)[:3]):
            code, scores, printed, _ = run_score(["--data", DATA, "--years", 2020, 2019, "--storm"],
                                                 model=torch.nn.Identity())
        self.assertEqual(code, 0)
        for name in ("2020", "2019"):
            year = scores["years"][name]
            self.assertEqual((year["n_starts"], year["n_starts_full"], year["comparable"]), (3, 3, True))
            # Made-up data is nowhere near the reference model: every 72-hour verdict is a no.
            self.assertEqual(year["same_skill_72h"], {"z500": False, "t850": False, "t2m": False})
        self.assertIs(scores["years"]["2020"]["baselines_ok"], False)
        self.assertIsNone(scores["years"]["2019"]["baselines_ok"])
        self.assertEqual(len(scores["warnings"]), 1)
        self.assertIn("2020: baselines differ", scores["warnings"][0])
        self.assertIn("same skill", printed)
        self.assertIn("z500 NO", printed)
        self.assertIn("BASELINES DIFFER", printed)
        self.assertIn("persistence z500 at 24 h", printed)
        self.assertIn("no known values for 2019", printed)
        self.assertIn("WARNING: 2020: baselines differ", printed)
        self.assertNotIn("No verdicts", printed)

    def test_broken_model_gives_null_scores_not_bad_json(self):
        code, scores, printed, _ = run_score(["--data", DATA, "--years", 2019, "--limit", 2, "--storm"],
                                             model=Broken())
        self.assertEqual(code, 0)                                # the scorer did its job; the model is the problem
        year = scores["years"]["2019"]
        self.assertEqual(year["rmse"]["model"]["z500"], [None] * 20)
        self.assertTrue(all(v > 0 for v in year["rmse"]["persistence"]["z500"]))
        self.assertIsNone(scores["storm"]["share_caught"])
        self.assertIs(scores["storm"]["in_reference_range"], False)
        self.assertIn("not finite", scores["warnings"][0])
        self.assertIn("WARNING", printed)

    def test_clear_messages_for_missing_things(self):
        empty = os.path.join(TMP, "empty")
        os.makedirs(empty, exist_ok=True)
        code, scores, _, errors = run_score(["--data", empty, "--years", 2020, "--no-model"])
        self.assertEqual((code, scores), (1, None))
        self.assertIn("stats.json", errors)
        code, _, _, errors = run_score(["--data", os.path.join(TMP, "nowhere"), "--years", 2020, "--no-model"])
        self.assertEqual(code, 1)
        self.assertIn("does not exist", errors)
        # A folder with statistics but no climatology, then one with no packed file for the year.
        partial = os.path.join(TMP, "partial")
        os.makedirs(partial, exist_ok=True)
        os.symlink(os.path.join(DATA, "stats.json"), os.path.join(partial, "stats.json"))
        code, _, _, errors = run_score(["--data", partial, "--years", 2020, "--no-model"])
        self.assertEqual(code, 1)
        self.assertIn("clim_1979_2017.npy", errors)
        os.symlink(os.path.join(DATA, "clim_1979_2017.npy"), os.path.join(partial, "clim_1979_2017.npy"))
        code, _, _, errors = run_score(["--data", partial, "--years", 2020, "--no-model"])
        self.assertEqual(code, 1)
        self.assertIn("packed_2020.npy", errors)
        # A weights file that is not there (the data is fine).
        code, scores, _, errors = run_score(["--data", DATA, "--years", 2020, "--limit", 1,
                                             "--weights", os.path.join(TMP, "no-such.pt")])
        self.assertEqual((code, scores), (1, None))
        self.assertIn("no-such.pt", errors)
        # A year outside the data.
        code, _, _, errors = run_score(["--data", DATA, "--years", 2021, "--no-model"])
        self.assertEqual(code, 1)
        self.assertIn("1979 to 2020", errors)


class CommandLineTests(unittest.TestCase):

    def run_script(self, args, wrapper=None):
        """Run score.py as its own process. Returns the finished process."""
        env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
        cmd = [sys.executable, SCORE] + [str(a) for a in args]
        if wrapper is not None:
            cmd = [sys.executable, "-c", wrapper] + cmd[1:]
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                              env=env, timeout=120, cwd=TMP)

    def test_no_model_does_not_import_torch(self):
        # Run the script the way the shell would, then look at what got imported. (2020 only: a separate
        # process does not see the moved start of the made-up main file.)
        wrapper = ("import runpy, sys\n"
                   "sys.argv = sys.argv[1:]\n"                   # as in a plain run: the script's path comes first
                   "code = 0\n"
                   "try:\n"
                   "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
                   "except SystemExit as e:\n"
                   "    code = e.code or 0\n"
                   "print('EXIT=%s TORCH=%s' % (code, 'torch' in sys.modules))\n")
        out = os.path.join(TMP, "out-no-model")
        p = self.run_script(["--no-model", "--data", DATA, "--out", out, "--years", 2020, "--limit", 4],
                            wrapper=wrapper)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("EXIT=0 TORCH=False", p.stdout)
        with open(os.path.join(out, "scores.json")) as f:
            scores = strict_json(f.read())
        self.assertEqual((scores["weights"], scores["device"], scores["torch"], scores["checkpoint"]),
                         (None, None, None, None))
        year = scores["years"]["2020"]
        self.assertIsNone(year["rmse"]["model"])
        self.assertEqual(len(year["rmse"]["persistence"]["z500"]), 20)
        self.assertEqual(len(year["rmse"]["climatology"]["t2m"]), 20)
        self.assertEqual(len(year["compare"]), 18)               # the two baselines only
        self.assertIn("baselines only", p.stdout)
        self.assertNotIn("model       z500", p.stdout)

    def test_plain_command_line_and_exit_codes(self):
        out = os.path.join(TMP, "out-plain")
        p = self.run_script(["--no-model", "--data", DATA, "--out", out, "--years", 2020, "--limit", 2])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertTrue(os.path.exists(os.path.join(out, "scores.json")))
        self.assertIn("wrote " + os.path.join(out, "scores.json"), p.stdout)
        # --weights is needed when there is a model to score.
        p = self.run_script(["--data", DATA, "--out", out, "--years", 2020])
        self.assertEqual(p.returncode, 2)
        self.assertIn("--weights", p.stderr)
        # A missing data folder: exit code 1 and one clear line, no traceback.
        p = self.run_script(["--no-model", "--data", os.path.join(TMP, "nowhere"), "--out", out, "--years", 2020])
        self.assertEqual(p.returncode, 1)
        self.assertIn("score.py: FAILED:", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        # --help mentions every option of the overnight driver's command line.
        p = self.run_script(["--help"])
        self.assertEqual(p.returncode, 0)
        for option in ("--weights", "--data", "--out", "--years", "--storm", "--limit", "--device", "--batch",
                       "--no-model"):
            self.assertIn(option, p.stdout)


class RealModelTests(unittest.TestCase):

    def test_weights_file_as_train_py_writes_it(self):
        # The real model (19 million weights, random here), saved the way train.py saves final.pt, then scored
        # through --weights for one start and the storm: 32 six-hour steps on the CPU, a few seconds.
        try:
            import physicsnemo.models.afno  # noqa: F401
        except ImportError as e:
            self.skipTest("PhysicsNeMo is not installed here: %s" % e)
        torch.manual_seed(7)
        model = wx.build_afno().eval()
        path = os.path.join(TMP, "final.pt")
        torch.save({"model": model.state_dict(), "step": 3, "seed": 1, "stage": 2, "complete": False, "val": None,
                    "steps_per_s": None, "recipe": {"card": "test", "betas": [0.9, 0.95]}}, path)
        code, scores, printed, errors = run_score(["--weights", path, "--data", DATA, "--years", 2020, "--storm",
                                                   "--limit", 1])
        self.assertEqual(code, 0, errors)
        self.assertEqual(scores["weights"], path)
        self.assertEqual(scores["checkpoint"], {"stage": 2, "step": 3, "complete": False})
        self.assertIn("not complete", scores["warnings"][0])
        self.assertIn("stage 2, step 3, NOT complete", printed)
        for var in VARS:
            values = scores["years"]["2020"]["rmse"]["model"][var]
            self.assertTrue(all(v is not None and v > 0 for v in values), values)
        self.assertIsNotNone(scores["storm"]["forecast_k"])
        # The weights in the file were the ones used: the 6-hour scores match a forward pass done here.
        packed = np.load(os.path.join(DATA, "packed_2020.npy"), mmap_mode="r")
        truth = np.load(os.path.join(DATA, "truth_2020.npy"), mmap_mode="r")
        with torch.inference_mode():
            step1 = model(torch.from_numpy(np.array(packed[0:1])).float())[0].numpy()
        for v, var in enumerate(VARS):
            want = by_definition([physical(step1, var)], [truth[1, v]])
            self.assertAlmostEqual(scores["years"]["2020"]["rmse"]["model"][var][0] / want, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
