"""Tests for weather/finetune.py: the split by time, the scores, the bootstrap, the training loop, the files.

Run them from the repo root (about 10 seconds on a laptop; CPU only, at most 4 threads, well under 1 GB):

    .venv/bin/python -m unittest tests/test_finetune.py -v

Most tests use made-up analyses on a tiny 4 x 8 grid and a model with one number in it (the next state is
a x this state). The made-up weather follows a known law, "the next state is 0.9 x this one", so the right
answer is known: fine-tuning must move a towards 0.9, and the scores must then improve.

The command-line tests write a small analyses folder on the real 120 x 240 grid into a temp folder (about
65 MB, removed at the end). The real AFNO is not built: wx_common.build_afno is swapped for the small model.

One test is skipped unless WX_FINETUNE_SMOKE=1 is set. It runs the real AFNO (random weights) through the
command line: scoring, two training steps, and scoring the saved weights again. It is meant for the GPU
machines, as a check before the real run:

    WX_FINETUNE_SMOKE=1 python -m unittest tests/test_finetune.py -v -k real_model_two_steps

Written at the hackathon (Nov 13-14, 2026).
"""
import contextlib
import datetime
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER = os.path.join(os.path.dirname(HERE), "weather")
FINETUNE = os.path.join(WEATHER, "finetune.py")
sys.path.insert(0, WEATHER)

import finetune         # noqa: E402
import wx_common as wx  # noqa: E402

torch.set_num_threads(4)                         # be gentle with the laptop

T0 = datetime.datetime(2026, 1, 1)               # the first made-up analysis
H6 = datetime.timedelta(hours=6)

# The tiny grid: 4 latitudes x 8 longitudes. Area weights with mean 1, and a "place" between four grid points.
TINY_AREA = np.repeat(np.array([0.4, 1.2, 1.6, 0.8])[:, None], 8, axis=1)
TINY_POINT = [(1, 2, 0.25), (1, 3, 0.25), (2, 2, 0.25), (2, 3, 0.25)]
STD = np.arange(1.0, 21.0)                       # made-up standard deviations of the 20 channels
STD[wx.CH_Z500], STD[wx.CH_T850], STD[wx.CH_T2M] = 100.0, 2.0, 4.0
MEAN = 10.0 * np.arange(20.0)                    # made-up means; no score may depend on them
MEAN[wx.CH_Z500], MEAN[wx.CH_T850], MEAN[wx.CH_T2M] = 54000.0, 270.0, 280.0
RMSE_KEYS = ["z500_rmse", "t850_rmse", "t2m_rmse"]
ALL_KEYS = RMSE_KEYS + ["all_channels_rmse_z", "ohare_t2m_mae_c"]

# The tiny data set: 21 blocks of 3 days. Training gets 15 blocks, validation 3, test 3.
TINY_SPLIT = ["--test-from", "2026-02-24T00", "--val-days", "9", "--lead-hours", "24"]

# The real-grid data set of the command-line tests: 14 days. Train 3 days, gap 1, validation 5, test 5.
CLI_SPLIT = ["--test-from", "2026-01-10T00", "--val-days", "5", "--lead-hours", "24", "--batch", "8"]

TMP = None               # set by setUpModule
CLI_DATA = None          # the analyses folder on the real grid (with stats.json)


class Scale(torch.nn.Module):
    """The one-number model: the next state is a x this state."""

    def __init__(self, a):
        super(Scale, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))

    def forward(self, x):
        return x * self.a


class Same(torch.nn.Module):
    """A model that gives back what it was given."""

    def forward(self, x):
        return x


class PlusOne(torch.nn.Module):
    """A model that adds 1 to everything, every step."""

    def forward(self, x):
        return x + 1.0


class SignalAt(Scale):
    """The one-number model, but it sends this process SIGTERM in the middle of one chosen forward call:
    the n-th call in training mode, or the n-th call in eval mode."""

    def __init__(self, a, call, in_training=True):
        super(SignalAt, self).__init__(a)
        self.call, self.in_training, self.calls = call, in_training, 0

    def forward(self, x):
        if self.training == self.in_training:
            self.calls += 1
            if self.calls == self.call:
                os.kill(os.getpid(), signal.SIGTERM)
        return x * self.a


class Watched(object):
    """The analyses, with a note of every row that is read and of whether the model was in training mode
    or in eval mode (being scored) at that moment."""

    def __init__(self, rows, model):
        self.rows, self.model, self.shape = rows, model, rows.shape
        self.read = {True: [], False: []}        # model.training -> the rows read, in order

    def __getitem__(self, key):
        picked = range(len(self.rows))[key]
        self.read[self.model.training] += [picked] if isinstance(picked, int) else list(picked)
        return self.rows[key]


def six_hourly(n, missing=()):
    """The times of n 6-hourly cycles from T0, without the cycle numbers in `missing`."""
    return [T0 + k * H6 for k in range(n) if k not in missing]


def decay_blocks(factors, block_rows=9, skip=3, shape=(20, 4, 8), seed=0):
    """Made-up analyses in blocks, one block per entry of `factors`. Inside a block every row is factor x the
    row before it, so the truth is "next = factor x current". A block starts from fresh random numbers at
    00 UTC, and `skip` cycles are missing after it, so no sample and no forecast can cross from one block
    into the next. Rows are float16, as in analyses.npy. Returns (rows, times)."""
    rng = np.random.default_rng(seed)
    rows, times = [], []
    for b, factor in enumerate(factors):
        state = rng.standard_normal(shape)
        for k in range(block_rows):
            rows.append(state.astype(np.float16))
            times.append(T0 + (b * (block_rows + skip) + k) * H6)
            state = factor * rows[-1].astype(np.float64)
    return np.stack(rows), times


def tiny_scoring():
    return finetune.Scoring(STD, area=TINY_AREA, point=TINY_POINT)


def strict_json(text):
    """Parse JSON, refusing NaN and Infinity (which Python would write but other readers reject)."""
    def refuse(name):
        raise ValueError("the report holds %s, which is not JSON" % name)
    return json.loads(text, parse_constant=refuse)


def read_report(out):
    with open(os.path.join(out, "finetune_report.json")) as f:
        return strict_json(f.read())


def read_log(out):
    with open(os.path.join(out, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def load_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def new_out():
    return tempfile.mkdtemp(prefix="out-", dir=TMP)


def run_tiny(model, data, times, extra, out=None):
    """Run finetune.finetune() on tiny data that is already in memory. Returns (code, report, printed, out)."""
    out = out or new_out()
    args = finetune.parse_args(["--init", "base.pt", "--analyses", "not-used", "--out", out] + TINY_SPLIT
                               + [str(x) for x in extra])
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code, report = finetune.finetune(args, model, data, times, tiny_scoring())
    return code, report, printed.getvalue(), out


def run_main(extra, model=None, analyses=None, init="base.pt", out=None, split=None):
    """Run the command line in this process, on the CPU. Returns (code, printed, errors, out)."""
    out = out or new_out()
    argv = (["--init", init, "--analyses", analyses or CLI_DATA, "--out", out, "--device", "cpu"]
            + (CLI_SPLIT if split is None else split) + [str(x) for x in extra])
    printed, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(errors):
        code = finetune.main(argv, model=model)
    return code, printed.getvalue(), errors.getvalue(), out


def failure(errors):
    """What finetune.py said when it refused: its one line on stderr, "finetune.py: FAILED: ...". Other
    libraries may have printed warnings before it; those do not count."""
    lines = [line for line in errors.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith("finetune.py: FAILED: ") or "Traceback" in errors:
        raise AssertionError("expected one line that starts with 'finetune.py: FAILED: ', got: %r" % errors)
    return lines[-1]


def make_cli_data(folder, n_rows=56):
    """An analyses folder on the real grid, as weather/gfs_archive.py writes it, plus a stats.json.
    The made-up weather is: next = 0.9 x current + fresh noise, with z-scores of about the right size."""
    os.makedirs(folder)
    rng = np.random.default_rng(2026)
    shape = (wx.N_CHANNELS, wx.N_LAT, wx.N_LON)
    big = np.lib.format.open_memmap(os.path.join(folder, "analyses.npy"), mode="w+", dtype=np.float16,
                                    shape=(n_rows,) + shape)
    state = rng.standard_normal(shape, dtype=np.float32)
    for row in range(n_rows):
        big[row] = state
        state = 0.9 * state + math.sqrt(1.0 - 0.81) * rng.standard_normal(shape, dtype=np.float32)
    big.flush()
    del big
    times = [(T0 + row * H6).strftime("%Y-%m-%dT%H") for row in range(n_rows)]
    with open(os.path.join(folder, "analyses.json"), "w") as f:
        json.dump({"file": "analyses.npy", "step_hours": 6, "first": times[0], "last": times[-1], "times": times}, f)
    write_stats(folder, STD)


def write_stats(folder, std):
    with open(os.path.join(folder, "stats.json"), "w") as f:
        json.dump({"channels": wx.CHANNEL_NAMES, "mean": [float(m) for m in MEAN], "std": [float(s) for s in std]}, f)


def setUpModule():
    global TMP, CLI_DATA
    TMP = tempfile.mkdtemp(prefix="wx-test-finetune-")
    try:
        CLI_DATA = os.path.join(TMP, "gfs")
        make_cli_data(CLI_DATA)
    except BaseException:
        shutil.rmtree(TMP, ignore_errors=True)   # tearDownModule is not called when this function fails
        raise


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ the splits

class SplitTest(unittest.TestCase):
    TEST_FROM = datetime.datetime(2026, 2, 1)
    VAL_FROM = datetime.datetime(2026, 1, 27)    # 5 days before
    GAP_FROM = datetime.datetime(2026, 1, 26)    # 24 hours before that

    def where(self, when):
        """The split a time belongs to, straight from the rules."""
        if when >= self.TEST_FROM:
            return "test"
        if when >= self.VAL_FROM:
            return "validation"
        return "gap" if when >= self.GAP_FROM else "train"

    def test_split_by_time_with_the_gap(self):
        times = six_hourly(160)                  # 40 days, nothing missing
        rows = finetune.split_rows(times, self.TEST_FROM, 5, 24)
        self.assertEqual(rows, {"train": (0, 100), "gap": (100, 104), "validation": (104, 124), "test": (124, 160)})
        self.assertEqual(times[99], datetime.datetime(2026, 1, 25, 18))          # the last training row
        self.assertEqual(times[104], self.VAL_FROM)
        self.assertEqual(times[124], self.TEST_FROM)
        # No training target falls inside validation: the last sample's targets end a day before it.
        samples = finetune.chain_starts(times, 0, 100, 2)
        self.assertEqual(samples, list(range(98)))
        self.assertLessEqual(times[samples[-1] + 2], self.VAL_FROM - datetime.timedelta(hours=24 + 6))

    def test_split_looks_at_times_not_at_row_numbers(self):
        missing = {3, 50, 51, 100, 104, 124, 130}                # three of them sit right on a boundary
        times = six_hourly(160, missing)
        rows = finetune.split_rows(times, self.TEST_FROM, 5, 24)
        self.assertEqual(rows, {"train": (0, 97), "gap": (97, 100), "validation": (100, 119), "test": (119, 153)})
        seen = []
        for name, (first, stop) in rows.items():
            for row in range(first, stop):
                self.assertEqual(name, self.where(times[row]), times[row])
                seen.append(row)
        self.assertEqual(seen, list(range(len(times))))          # every row is in exactly one part

    def test_a_split_can_be_empty(self):
        times = six_hourly(40)
        rows = finetune.split_rows(times, datetime.datetime(2027, 1, 1), 5, 24)
        self.assertEqual(rows, {"train": (0, 40), "gap": (40, 40), "validation": (40, 40), "test": (40, 40)})
        rows = finetune.split_rows(times, T0, 5, 24)
        self.assertEqual(rows, {"train": (0, 0), "gap": (0, 0), "validation": (0, 0), "test": (0, 40)})


class StartsTest(unittest.TestCase):
    def test_forecast_starts_at_00_and_12_that_end_inside_the_split(self):
        times = six_hourly(40)
        starts = finetune.chain_starts(times, 8, 32, 4, (0, 12))                 # rows 8..31, 24-hour forecasts
        self.assertEqual(starts, list(range(8, 28, 2)))                          # row 28 would need row 32
        self.assertTrue(all(times[row].hour in (0, 12) for row in starts))
        self.assertEqual(finetune.chain_starts(times, 8, 12, 4, (0, 12)), [])    # too short for any forecast

    def test_forecast_starts_respect_gaps(self):
        missing = {13}
        cycles = [k for k in range(40) if k not in missing]
        times = six_hourly(40, missing)
        starts = finetune.chain_starts(times, 0, len(times), 4, (0, 12))
        # A 24-hour forecast from cycle c needs cycles c+1 .. c+4. Cycles 10 and 12 would need the missing 13.
        self.assertEqual([cycles[row] for row in starts], [0, 2, 4, 6, 8] + list(range(14, 36, 2)))
        for row in starts:
            for k in range(1, 5):
                self.assertEqual(times[row + k], times[row] + k * H6)

    def test_training_samples_respect_gaps(self):
        missing = {13}
        cycles = [k for k in range(40) if k not in missing]
        times = six_hourly(40, missing)
        samples = finetune.chain_starts(times, 0, len(times), 2)                 # any hour; the 2 rows after it
        self.assertEqual([cycles[row] for row in samples], [c for c in range(38) if c not in (11, 12, 13)])
        # Inside a row range: the targets must be in the range too.
        self.assertEqual(finetune.chain_starts(times, 0, 10, 2), list(range(8)))


def plan_args(*extra):
    return finetune.parse_args(["--init", "base.pt", "--analyses", "not-used", "--out", "not-used"]
                               + [str(x) for x in extra])


class PlanTest(unittest.TestCase):
    def test_plan_counts(self):
        times = six_hourly(240)                  # 60 days
        args = plan_args("--test-from", "2026-02-20T00", "--val-days", 6, "--lead-hours", 24)
        plan = finetune.make_plan(args, times)
        self.assertEqual(plan["rows"], {"train": (0, 172), "gap": (172, 176), "validation": (176, 200),
                                        "test": (200, 240)})
        self.assertEqual(plan["leads_h"], [6, 12, 18, 24])
        self.assertEqual(plan["shown_h"], [6, 24])
        self.assertEqual(plan["samples"], list(range(170)))                      # --rollout 2 by default
        self.assertEqual(plan["starts"]["validation"], list(range(176, 196, 2)))
        self.assertEqual(plan["starts"]["test"], list(range(200, 236, 2)))

    def test_shown_leads_are_6_hours_and_every_24(self):
        times = six_hourly(400)
        args = plan_args("--test-from", "2026-03-20T00", "--lead-hours", 72)
        self.assertEqual(finetune.make_plan(args, times)["shown_h"], [6, 24, 48, 72])

    def test_max_starts_keeps_the_first_n(self):
        times = six_hourly(240)
        args = plan_args("--test-from", "2026-02-20T00", "--val-days", 6, "--lead-hours", 24, "--max-starts", 3)
        plan = finetune.make_plan(args, times)
        self.assertEqual(plan["starts"], {"validation": [176, 178, 180], "test": [200, 202, 204]})
        self.assertEqual(plan["available"], {"validation": 10, "test": 18})

    def test_too_few_forecast_starts_is_refused(self):
        times = six_hourly(240)
        args = plan_args("--test-from", "2026-02-20T00", "--val-days", 2, "--lead-hours", 24)
        with self.assertRaises(finetune.FinetuneError) as caught:
            finetune.make_plan(args, times)
        message = str(caught.exception)
        self.assertIn("validation split (2026-02-18T00 to 2026-02-19T18, 8 rows) gives 2 forecast starts", message)
        self.assertIn("at least 8 are needed", message)
        self.assertNotIn("test split", message)

    def test_no_test_weeks_is_refused(self):
        times = six_hourly(240)                  # January and February; the default test weeks start in October
        with self.assertRaises(finetune.FinetuneError) as caught:
            finetune.make_plan(plan_args(), times)
        self.assertIn("The test split (no rows) gives 0 forecast starts", str(caught.exception))

    def test_too_few_training_samples_is_refused_unless_eval_only(self):
        times = six_hourly(120)                  # 30 days: training gets 13 of them
        split = ["--test-from", "2026-01-21T00", "--val-days", 6, "--lead-hours", 24]
        with self.assertRaises(finetune.FinetuneError) as caught:
            finetune.make_plan(plan_args(*split), times)
        self.assertIn("gives 50 training samples; at least 100 are needed", str(caught.exception))
        plan = finetune.make_plan(plan_args("--eval-only", *split), times)       # scoring needs no training rows
        self.assertEqual(len(plan["samples"]), 50)

    def test_times_out_of_order_are_refused(self):
        times = six_hourly(240)
        times[7], times[8] = times[8], times[7]
        with self.assertRaises(finetune.FinetuneError) as caught:
            finetune.make_plan(plan_args("--test-from", "2026-02-20T00"), times)
        self.assertIn("must increase", str(caught.exception))

    def test_a_bad_test_from_is_refused(self):
        with self.assertRaises(finetune.FinetuneError) as caught:
            finetune.make_plan(plan_args("--test-from", "October 1"), six_hourly(240))
        self.assertIn("must look like 2026-10-01T00", str(caught.exception))

    def test_bad_numbers_on_the_command_line(self):
        for bad in (["--lead-hours", "20"], ["--lead-hours", "18"], ["--steps", "0"], ["--warmup", "400"],
                    ["--lr", "0"], ["--batch", "0"], ["--device", "mps"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit, msg=bad):
                    plan_args(*bad)
        self.assertEqual(plan_args("--eval-only", "--steps", "5").steps, 5)      # nothing is trained: no warm-up to fit


# ------------------------------------------------------------------------------------------------ training pieces

class ScheduleTest(unittest.TestCase):
    def test_learning_rate(self):
        peak, warmup, total = 3e-5, 20, 400
        rate = [finetune.learning_rate(s, peak, warmup, total) for s in range(total + 1)]
        for s in range(warmup):                                  # a straight climb: 1/20, 2/20, ... of the peak
            self.assertTrue(math.isclose(rate[s], peak * (s + 1) / warmup, rel_tol=1e-12), s)
        self.assertTrue(math.isclose(rate[0], 1.5e-6, rel_tol=1e-12))
        self.assertEqual(rate[19], peak)                         # the climb ends exactly at the peak ...
        self.assertEqual(rate[20], peak)                         # ... and the cosine starts from it
        for s in range(warmup, total + 1):
            want = peak * 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / (total - warmup)))
            self.assertTrue(math.isclose(rate[s], want, rel_tol=1e-12, abs_tol=1e-20), s)
        self.assertTrue(math.isclose(rate[210], peak / 2, rel_tol=1e-9))         # halfway down
        for s in range(warmup, total):
            self.assertLess(rate[s + 1], rate[s])                # always falling after the warm-up
        self.assertEqual(rate[400], 0.0)                         # zero at --steps ...
        self.assertGreater(rate[399], 0.0)                       # ... so the last step that runs is just above
        self.assertTrue(all(0.0 < r <= peak for r in rate[:400]))

    def test_learning_rate_without_warmup(self):
        self.assertEqual(finetune.learning_rate(0, 0.5, 0, 10), 0.5)
        self.assertTrue(math.isclose(finetune.learning_rate(5, 0.5, 0, 10), 0.25, rel_tol=1e-12))

    def test_sample_stream(self):
        epochs = [np.random.default_rng([3, e]).permutation(10) for e in range(3)]
        stream = finetune.SampleStream(10, 3)
        first, second = stream.take(10), stream.take(10)
        self.assertEqual(sorted(first), list(range(10)))         # an epoch is every sample once ...
        self.assertEqual(sorted(second), list(range(10)))
        self.assertTrue(np.array_equal(first, epochs[0]) and np.array_equal(second, epochs[1]))
        self.assertFalse(np.array_equal(first, second))          # ... in a fresh order
        # A batch can straddle two epochs, and the same seed gives the same order again.
        stream = finetune.SampleStream(10, 3)
        self.assertTrue(np.array_equal(stream.take(16), np.concatenate([epochs[0], epochs[1][:6]])))
        self.assertTrue(np.array_equal(stream.take(8), np.concatenate([epochs[1][6:], epochs[2][:4]])))
        self.assertFalse(np.array_equal(finetune.SampleStream(10, 4).take(10), epochs[0]))

    def test_loss_by_hand(self):
        # x = 1 everywhere, targets 0, the model doubles: the two fed-back steps give 2 and 4.
        model = Scale(2.0)
        weights = torch.tensor(wx.t7_channel_weights(), dtype=torch.float32).view(1, -1, 1, 1)
        x, targets = torch.ones(3, 20, 4, 8), torch.zeros(3, 2, 20, 4, 8)
        loss = finetune.rollout_loss(model, x, targets, weights, False)
        self.assertAlmostEqual(loss.item(), (4.0 + 16.0) / 2, places=5)          # the weights have mean 1
        # The loss is (a^2 + a^4) / 2, so its slope is a + 2 a^3 = 18. Had the fed-back state been detached,
        # the slope would be 10.
        loss.backward()
        self.assertAlmostEqual(model.a.grad.item(), 18.0, places=4)
        # The channel weights count: an error in Z500 alone weighs 60/26, in another channel 20/26.
        for channel, weight in ((wx.CH_Z500, 60.0 / 26.0), (0, 20.0 / 26.0)):
            targets = torch.ones(1, 1, 20, 4, 8)
            targets[:, :, channel] = 0.0
            loss = finetune.rollout_loss(Same(), torch.ones(1, 20, 4, 8), targets, weights, False)
            self.assertAlmostEqual(loss.item(), weight / 20.0, places=6)

    def test_loss_is_the_one_train_py_uses(self):
        import train
        torch.manual_seed(0)
        model = torch.nn.Conv2d(20, 20, 3, padding=1)
        x, targets = torch.randn(3, 20, 6, 8), torch.randn(3, 2, 20, 6, 8)
        weights = torch.tensor(wx.t7_channel_weights(), dtype=torch.float32).view(1, -1, 1, 1)
        ours = finetune.rollout_loss(model, x, targets, weights, False)
        theirs = train.rollout_loss(model, x, targets, weights, False)
        self.assertAlmostEqual(ours.item(), theirs.item(), places=6)
        ours_grad = torch.autograd.grad(ours, model.weight)[0]
        theirs_grad = torch.autograd.grad(theirs, model.weight)[0]
        self.assertTrue(torch.allclose(ours_grad, theirs_grad, rtol=1e-5, atol=1e-8))
        self.assertEqual((finetune.BETAS, finetune.WEIGHT_DECAY, finetune.CLIP_NORM),
                         (train.BETAS, train.WEIGHT_DECAY, train.CLIP_NORM))


# ------------------------------------------------------------------------------------------------ the scores

class ScoreTest(unittest.TestCase):
    def test_weighted_rmse_by_hand(self):
        # A 2 x 2 grid; the second latitude row weighs three times the first. Two starts, one 6-hour step.
        # The model gives back its input (zeros), so forecast - truth is minus the truth row.
        area = np.array([[0.5, 0.5], [1.5, 1.5]])
        data = np.zeros((4, 20, 2, 2), np.float16)
        data[1, wx.CH_Z500] = [[1, 1], [2, 2]]       # start A: weighted mean of squares (0.5+0.5+6+6)/4 = 3.25
        data[3, wx.CH_Z500] = [[1, -1], [1, -1]]     # start B: (0.5+0.5+1.5+1.5)/4 = 1
        data[1, wx.CH_T850] = [[2, 0], [0, 0]]       # A: 0.5*4/4 = 0.5;  B: 0
        data[1, wx.CH_T2M] = [[0, 0], [1, 1]]        # A: (1.5+1.5)/4 = 0.75
        data[3, wx.CH_T2M] = [[0, 0], [-1, -1]]      # B: 0.75
        scoring = finetune.Scoring(STD, area=area, point=[(1, 0, 0.5), (1, 1, 0.5)])
        errors = finetune.evaluate(Same(), data, [0, 2], 1, "cpu", scoring)
        mse = errors["model"]["mse"]
        self.assertEqual(mse.shape, (2, 1, 20))
        self.assertEqual(mse[:, 0, wx.CH_Z500].tolist(), [3.25, 1.0])
        self.assertEqual(mse[:, 0, wx.CH_T850].tolist(), [0.5, 0.0])
        self.assertEqual(mse[:, 0, wx.CH_T2M].tolist(), [0.75, 0.75])
        self.assertEqual(errors["model"]["point"][:, 0].tolist(), [-1.0, 1.0])   # forecast - truth at the place
        scores = finetune.scores_of(finetune.start_values(errors["model"], scoring.std))
        # RMSE = the root of the mean over starts, times the channel's standard deviation (100, 2 and 4).
        self.assertAlmostEqual(scores["z500_rmse"][0], 100.0 * math.sqrt((3.25 + 1.0) / 2), places=9)
        self.assertAlmostEqual(scores["t850_rmse"][0], 2.0 * math.sqrt(0.5 / 2), places=9)
        self.assertAlmostEqual(scores["t2m_rmse"][0], 4.0 * math.sqrt(0.75), places=9)
        # All 20 channels, z-units: start A (3.25 + 0.5 + 0.75) / 20, start B (1 + 0 + 0.75) / 20.
        self.assertAlmostEqual(scores["all_channels_rmse_z"][0], math.sqrt((0.225 + 0.0875) / 2), places=9)
        self.assertAlmostEqual(scores["ohare_t2m_mae_c"][0], 4.0, places=9)      # |1| z-units x 4 K
        # The root of the mean, not the mean of the roots:
        self.assertNotAlmostEqual(scores["z500_rmse"][0], 100.0 * (math.sqrt(3.25) + 1.0) / 2, places=3)

    def test_fed_back_steps_and_leads(self):
        # The truth never changes (all zeros) and the model adds 1 each step: lead k is off by k everywhere.
        data = np.zeros((6, 20, 4, 8), np.float16)
        scoring = tiny_scoring()
        errors = finetune.evaluate(PlusOne(), data, [0, 1, 2], 3, "cpu", scoring, batch=2)
        scores = finetune.scores_of(finetune.start_values(errors["model"], scoring.std))
        for k in range(3):
            self.assertAlmostEqual(scores["z500_rmse"][k], 100.0 * (k + 1), places=9)
            self.assertAlmostEqual(scores["t850_rmse"][k], 2.0 * (k + 1), places=9)
            self.assertAlmostEqual(scores["t2m_rmse"][k], 4.0 * (k + 1), places=9)
            self.assertAlmostEqual(scores["all_channels_rmse_z"][k], float(k + 1), places=9)
            self.assertAlmostEqual(scores["ohare_t2m_mae_c"][k], 4.0 * (k + 1), places=9)
        still = finetune.scores_of(finetune.start_values(errors["persistence"], scoring.std))
        self.assertEqual(still, dict((key, [0.0, 0.0, 0.0]) for key in ALL_KEYS))

    def test_a_model_that_returns_its_input_scores_exactly_as_persistence(self):
        data = np.random.default_rng(5).standard_normal((30, 20, 4, 8)).astype(np.float16)
        starts = finetune.chain_starts(six_hourly(30), 0, 30, 4, (0, 12))
        self.assertEqual(len(starts), 13)
        scoring = tiny_scoring()
        errors = finetune.evaluate(Same(), data, starts, 4, "cpu", scoring, batch=5)
        for what in ("mse", "point"):
            self.assertTrue(np.array_equal(errors["model"][what], errors["persistence"][what]), what)
            self.assertTrue(np.abs(errors["model"][what]).min() > 0)
        model = finetune.scores_of(finetune.start_values(errors["model"], scoring.std))
        still = finetune.scores_of(finetune.start_values(errors["persistence"], scoring.std))
        self.assertEqual(model, still)
        # Persistence by definition: the start state against the state k steps later.
        k, s = 3, 4
        diff = data[starts[s]].astype(np.float64) - data[starts[s] + k].astype(np.float64)
        want = (TINY_AREA * diff * diff).mean(axis=(1, 2))
        self.assertTrue(np.allclose(errors["persistence"]["mse"][s, k - 1], want, rtol=1e-12))
        # The batch size changes nothing.
        again = finetune.evaluate(Same(), data, starts, 4, "cpu", scoring, batch=16)
        self.assertTrue(np.allclose(errors["model"]["mse"], again["model"]["mse"], rtol=1e-12))

    def test_evaluate_leaves_the_model_as_it_was_and_can_be_stopped(self):
        data = np.zeros((6, 20, 4, 8), np.float16)
        model = Scale(1.0)
        model.train()
        self.assertIsNotNone(finetune.evaluate(model, data, [0], 2, "cpu", tiny_scoring()))
        self.assertTrue(model.training)
        self.assertIsNone(finetune.evaluate(model, data, [0], 2, "cpu", tiny_scoring(), stop=lambda: True))
        self.assertTrue(model.training)
        model.eval()
        finetune.evaluate(model, data, [0], 2, "cpu", tiny_scoring())
        self.assertFalse(model.training)

    def test_area_weights_and_ohare_on_the_real_grid(self):
        scoring = finetune.Scoring(STD)
        cos_lat = [math.cos(math.radians(-88.5 + 1.5 * i)) for i in range(120)]
        mean_cos = sum(cos_lat) / 120
        self.assertEqual(scoring.area.shape, (120, 240))
        for i in (0, 59, 87, 119):
            self.assertTrue(np.allclose(scoring.area[i], max(cos_lat[i], 0.0) / mean_cos, rtol=1e-12))
        self.assertAlmostEqual(float(scoring.area.mean()), 1.0, places=12)
        # An error of 3 along the equator row (59) of Z500 only, nowhere else.
        forecast, truth = torch.zeros(1, 20, 120, 240), torch.zeros(1, 20, 120, 240)
        forecast[0, wx.CH_Z500, 59] = 3.0
        mse, _ = finetune.grid_errors(forecast, truth, torch.as_tensor(scoring.area), scoring.point)
        self.assertAlmostEqual(mse[0, wx.CH_Z500].item(), 9.0 * (cos_lat[59] / mean_cos) / 120, places=12)
        self.assertEqual(mse[0, wx.CH_T2M].item(), 0.0)

        # O'Hare (41.9602 N, 87.9316 W = 272.0684 E) lies between rows 86 and 87 (40.5, 42.0 N) and
        # columns 181 and 182 (271.5, 273.0 E).
        self.assertEqual([(i, j) for i, j, _ in scoring.point], [(86, 181), (86, 182), (87, 181), (87, 182)])
        self.assertAlmostEqual(sum(w for _, _, w in scoring.point), 1.0, places=12)
        # Bilinear weights give back a field that is a straight line in latitude and in longitude.
        field = 2.0 * wx.LATS[:, None] + wx.LONS[None, :]
        at_ohare = sum(w * field[i, j] for i, j, w in scoring.point)
        self.assertAlmostEqual(at_ohare, 2.0 * 41.9602 + (360.0 - 87.9316), places=9)
        forecast = torch.zeros(1, 20, 120, 240)
        forecast[0, wx.CH_T2M] = torch.from_numpy(field).float()
        _, at_point = finetune.grid_errors(forecast, truth, torch.as_tensor(scoring.area), scoring.point)
        self.assertAlmostEqual(at_point.item(), at_ohare, places=3)              # float32 field

    def test_point_weights_at_the_edges(self):
        for lat, lon in ((90.0, 0.0), (-90.0, 359.9), (0.0, -0.1), (41.9602, 272.0684)):
            weights = finetune.point_weights(lat, lon)
            self.assertAlmostEqual(sum(w for _, _, w in weights), 1.0, places=12)
            self.assertTrue(all(0 <= i < 120 and 0 <= j < 240 and w >= 0 for i, j, w in weights))
        east, west = finetune.point_weights(41.9602, 272.0684), finetune.point_weights(41.9602, -87.9316)
        self.assertEqual([(i, j) for i, j, _ in east], [(i, j) for i, j, _ in west])
        self.assertTrue(np.allclose([w for _, _, w in east], [w for _, _, w in west], atol=1e-9))
        # Just west of Greenwich the two columns are the last one and the first one.
        self.assertEqual(sorted(set(j for _, j, _ in finetune.point_weights(0.0, -0.1))), [0, 239])

    def test_selection_score(self):
        leads = [6, 12, 18, 24, 30, 36, 42, 48]                  # 72 h does not exist here
        base = dict((key, [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 4.0]) for key in ALL_KEYS)
        self.assertEqual(finetune.selection_score(base, base, leads), 1.0)
        model = dict(base)
        model["z500_rmse"] = [9.0, 9.0, 9.0, 1.0, 9.0, 9.0, 9.0, 4.0]            # 24 h: half; 48 h: the same
        model["t2m_rmse"] = [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 5.0]             # 24 h: 1.5; 48 h: 1.25
        model["ohare_t2m_mae_c"] = [50.0] * 8                    # not one of the three graded maps
        want = (0.5 + 1.0 + 1.0 + 1.0 + 1.5 + 1.25) / 6
        self.assertAlmostEqual(finetune.selection_score(model, base, leads), want, places=12)


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.days = ["2026-10-%02d" % (1 + n // 2) for n in range(40)]           # 20 days, two starts each
        self.base = rng.uniform(0.5, 1.5, size=(40, 3)) ** 2                     # squared errors, 3 leads

    def test_an_obviously_better_model_gives_an_interval_below_zero(self):
        rng = np.random.default_rng(12)
        tuned = self.base * rng.uniform(0.5, 0.7, size=(40, 3))                  # every error clearly smaller
        low, high = finetune.bootstrap_interval(self.base, tuned, self.days, True, 1)
        self.assertEqual(low.shape, (3,))
        for k in range(3):
            self.assertLess(low[k], high[k])
            self.assertLess(high[k], 0.0)
            self.assertGreater(low[k], -50.0)                    # sqrt(0.5) - 1 = -29% at the very most
            change = 100.0 * (math.sqrt(tuned[:, k].mean() / self.base[:, k].mean()) - 1.0)
            self.assertTrue(low[k] < change < high[k])
            self.assertEqual(finetune.verdict_word(low[k], high[k]), "better")

    def test_identical_models_give_an_interval_that_contains_zero(self):
        low, high = finetune.bootstrap_interval(self.base, self.base.copy(), self.days, True, 1)
        self.assertEqual(low.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(high.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(finetune.verdict_word(0.0, 0.0), "no clear change")

    def test_an_obviously_worse_model_and_a_noisy_one(self):
        low, high = finetune.bootstrap_interval(self.base, self.base * 1.44, self.days, True, 1)
        self.assertTrue(np.allclose(low, 20.0) and np.allclose(high, 20.0))      # RMSE x 1.2 on every draw
        self.assertEqual(finetune.verdict_word(low[0], high[0]), "worse")
        # Half the days much better and half much worse, the same on the whole: which days a draw picks
        # decides its sign, so the interval must reach to both sides of zero.
        base = np.ones(40)
        mixed = np.repeat(np.tile([0.5, 1.5], 10), 2)            # squared errors: day 1 x 0.5, day 2 x 1.5, ...
        for seed in (1, 2, 3):
            low, high = finetune.bootstrap_interval(base, mixed, self.days, True, seed)
            self.assertTrue(-20.0 < low < -3.0 and 3.0 < high < 20.0, (seed, low, high))
            self.assertEqual(finetune.verdict_word(low, high), "no clear change")

    def test_mean_absolute_errors_are_not_rooted(self):
        base = np.sqrt(self.base[:, 0])
        low, high = finetune.bootstrap_interval(base, base * 0.5, self.days, False, 1)
        self.assertAlmostEqual(float(low), -50.0, places=9)
        self.assertAlmostEqual(float(high), -50.0, places=9)
        low, high = finetune.bootstrap_interval(base ** 2, base ** 2 * 0.25, self.days, True, 1)
        self.assertAlmostEqual(float(low), -50.0, places=9)                      # a quarter of the squares

    def test_a_days_forecasts_are_drawn_together(self):
        # On every day one start gets better and the other worse by the same amount, so a day as a whole
        # does not change. Drawing whole days must see no change at all; drawing single starts would see noise.
        base = np.tile([1.0, 2.0], 20)
        tuned = np.tile([2.0, 1.0], 20)
        low, high = finetune.bootstrap_interval(base, tuned, self.days, True, 1)
        self.assertEqual((float(low), float(high)), (0.0, 0.0))
        apart = ["start %d" % n for n in range(40)]              # every start its own "day"
        low, high = finetune.bootstrap_interval(base, tuned, apart, True, 1)
        self.assertTrue(low < -2.0 and high > 2.0)

    def test_against_a_slow_bootstrap(self):
        rng = np.random.default_rng(14)
        days = ["d%d" % n for n in (0, 0, 1, 2, 2, 2, 3, 4, 4, 5)]               # uneven days
        base = rng.uniform(0.5, 2.0, size=10)
        tuned = base * rng.uniform(0.6, 1.2, size=10)
        low, high = finetune.bootstrap_interval(base, tuned, days, True, 7, resamples=200)
        # The same draws, done the slow way: pick 6 days, gather their starts, score both models.
        names = sorted(set(days))
        drawn = np.random.default_rng(7).integers(0, 6, size=(200, 6))
        changes = []
        for draw in drawn:
            picked = [n for d in draw for n in range(10) if days[n] == names[d]]
            rmse_base = math.sqrt(sum(base[n] for n in picked) / len(picked))
            rmse_tuned = math.sqrt(sum(tuned[n] for n in picked) / len(picked))
            changes.append(100.0 * (rmse_tuned / rmse_base - 1.0))
        want_low, want_high = np.percentile(changes, [5, 95])
        self.assertAlmostEqual(float(low), want_low, places=9)
        self.assertAlmostEqual(float(high), want_high, places=9)
        # Seeded: the same seed gives the same interval, another seed another one.
        again = finetune.bootstrap_interval(base, tuned, days, True, 7, resamples=200)
        self.assertEqual((float(low), float(high)), (float(again[0]), float(again[1])))
        other = finetune.bootstrap_interval(base, tuned, days, True, 8, resamples=200)
        self.assertNotEqual(float(low), float(other[0]))

    def test_verdict_words(self):
        self.assertEqual(finetune.verdict_word(-5.0, -0.1), "better")
        self.assertEqual(finetune.verdict_word(0.1, 5.0), "worse")
        self.assertEqual(finetune.verdict_word(-5.0, 5.0), "no clear change")
        self.assertEqual(finetune.verdict_word(-5.0, 0.0), "no clear change")    # zero is inside
        self.assertEqual(finetune.verdict_word(0.0, 5.0), "no clear change")
        self.assertEqual(finetune.verdict_word(float("nan"), float("nan")), "no clear change")


# ------------------------------------------------------------------------------------------------ the whole job

class FineTuneTest(unittest.TestCase):
    """One fine-tuning run on the tiny data, where the truth is "next = 0.9 x current" and the base model
    says "next = 1.0 x current". The tests below look at what it wrote."""

    STEPS = 60
    EXTRA = ["--steps", STEPS, "--lr", 0.02, "--warmup", 5, "--eval-every", 20, "--rollout", 2, "--seed", 1]

    @classmethod
    def setUpClass(cls):
        rows, cls.times = decay_blocks([0.9] * 21)
        path = os.path.join(TMP, "tiny.npy")
        np.save(path, rows)
        cls.data = np.load(path, mmap_mode="r")                  # memory-mapped float16, as the real file is
        cls.model = Scale(1.0)
        cls.code, cls.returned, cls.printed, cls.out = run_tiny(cls.model, cls.data, cls.times, cls.EXTRA)
        cls.report = read_report(cls.out)
        cls.saved = load_weights(os.path.join(cls.out, "tuned.pt"))

    def test_training_moves_the_model_towards_0_9(self):
        self.assertEqual(self.code, 0)
        a = self.saved["model"]["a"].item()
        self.assertLess(abs(a - 0.9), 0.02)                      # it started at 1.0
        self.assertEqual(self.model.a.item(), a)                 # the model is left holding the saved weights
        steps = [line for line in read_log(self.out) if "event" not in line]
        self.assertLess(np.mean([line["loss"] for line in steps[-10:]]), 0.01 * steps[0]["loss"])

    def test_validation_improves(self):
        selection = self.report["selection"]
        self.assertEqual([check["step"] for check in selection["checks"]], [20, 40, 60])
        self.assertEqual(selection["leads_h"], [24])
        self.assertTrue(selection["checked_on_validation"] and selection["beats_base_on_validation"])
        self.assertLess(selection["best_score"], 0.2)            # the base model is 1.0
        self.assertEqual(selection["best_score"], min(check["score"] for check in selection["checks"]))
        best = [check for check in selection["checks"] if check["score"] == selection["best_score"]][0]
        self.assertEqual(selection["best_step"], best["step"])
        validation = self.report["scores"]["validation"]
        for key in ALL_KEYS:
            for k in range(4):
                self.assertLess(validation["tuned"][key][k], 0.2 * validation["base"][key][k], (key, k))
                self.assertEqual(validation["base"][key][k], validation["persistence"][key][k])   # a = 1.0
        # The validation score is what the docstring says it is.
        want = np.mean([validation["tuned"][key][3] / validation["base"][key][3] for key in RMSE_KEYS])
        self.assertAlmostEqual(selection["best_score"], want, places=12)

    def test_the_test_weeks_say_better(self):
        self.assertEqual(self.report["verdicts"],
                         {"z500": {"24h": "better"}, "t850": {"24h": "better"}, "t2m": {"24h": "better"}})
        self.assertEqual(self.report["overall"]["word"], "better")
        self.assertEqual(self.report["overall"]["sentence"], "The tuned model is better than the base model on "
                         "the test weeks (of 3 comparisons, 3 better, 0 worse, 0 no clear change).")
        rows = self.report["comparison"]["rows"]
        self.assertEqual([(row["score"], row["lead_h"]) for row in rows],
                         [(key, lead) for key in ALL_KEYS for lead in (6, 24)])
        test = self.report["scores"]["test"]
        for row in rows:
            k = self.report["leads_h"].index(row["lead_h"])
            self.assertEqual(row["base"], test["base"][row["score"]][k])
            self.assertEqual(row["tuned"], test["tuned"][row["score"]][k])
            self.assertEqual(row["persistence"], test["persistence"][row["score"]][k])
            self.assertAlmostEqual(row["change_pct"], 100.0 * (row["tuned"] / row["base"] - 1.0), places=9)
            low, high = row["ci90_pct"]
            self.assertTrue(low <= high < 0.0, row)
            self.assertTrue(low - 1.0 < row["change_pct"] < high + 1.0, row)
            self.assertEqual(row["verdict"], "better")
        self.assertEqual(self.report["comparison"]["starts"], 9)
        self.assertEqual(self.report["comparison"]["days"], 6)
        self.assertEqual(self.report["comparison"]["resamples"], 1000)
        for line in ("Z500 at 24 h: better", "T850 at 24 h: better", "T2M at 24 h: better",
                     "The tuned model is better than the base model"):
            self.assertIn(line, self.printed)

    def test_the_splits_are_kept_apart(self):
        splits = self.report["splits"]
        self.assertEqual((splits["test_from"], splits["validation_from"], splits["gap_from"]),
                         ("2026-02-24T00", "2026-02-15T00", "2026-02-14T00"))
        self.assertEqual(splits["train"], {"first": "2026-01-01T00", "last": "2026-02-13T18", "rows": 134,
                                           "samples": 104})
        self.assertEqual((splits["gap"]["rows"], splits["gap"]["hours"]), (1, 24))
        self.assertEqual((splits["validation"]["rows"], splits["validation"]["starts"]), (27, 9))
        self.assertEqual((splits["test"]["rows"], splits["test"]["starts"]), (27, 9))
        self.assertEqual(splits["validation"]["start_times"],
                         ["2026-02-%02dT%02d" % (day, hour) for day, hour in
                          ((15, 0), (15, 12), (16, 0), (18, 0), (18, 12), (19, 0), (21, 0), (21, 12), (22, 0))])
        self.assertTrue(all("2026-02-24T00" <= t for t in splits["test"]["start_times"]))
        self.assertEqual(splits["test"]["start_days"], 6)
        for name in ("train", "gap", "validation", "test"):
            self.assertIn("%s to %s" % (splits[name]["first"], splits[name]["last"]), self.printed)

    def test_report_json_has_everything(self):
        report = self.report
        self.assertEqual(sorted(report), sorted([
            "what", "written", "mode", "complete", "stopped_early", "settings", "device", "bf16", "torch",
            "params", "analyses", "splits", "leads_h", "shown_leads_h", "units", "scores", "selection",
            "training", "comparison", "verdicts", "overall", "seconds"]))
        self.assertEqual((report["mode"], report["complete"], report["stopped_early"]), ("finetune", True, False))
        self.assertEqual((report["device"], report["bf16"], report["torch"]), ("cpu", False, torch.__version__))
        self.assertEqual(report["params"], 1)
        self.assertEqual((report["leads_h"], report["shown_leads_h"]), ([6, 12, 18, 24], [6, 24]))
        settings = report["settings"]
        for name, value in (("init", "base.pt"), ("out", self.out), ("test_from", "2026-02-24T00"),
                            ("val_days", 9), ("lead_hours", 24), ("steps", 60), ("lr", 0.02), ("warmup", 5),
                            ("batch", 16), ("rollout", 2), ("eval_every", 20), ("seed", 1), ("max_starts", None),
                            ("eval_only", False), ("compile", False), ("betas", [0.9, 0.95]),
                            ("weight_decay", 0.01), ("clip_norm", 1.0)):
            self.assertEqual(settings[name], value, name)
        self.assertEqual(sorted(report["units"]), sorted(ALL_KEYS))
        for split in ("validation", "test"):
            self.assertEqual(sorted(report["scores"][split]), ["base", "persistence", "tuned"])
            for who in ("base", "persistence", "tuned"):
                scores = report["scores"][split][who]
                self.assertEqual(sorted(scores), sorted(ALL_KEYS))
                for key in ALL_KEYS:
                    self.assertEqual(len(scores[key]), 4)
                    self.assertTrue(all(isinstance(v, float) and v > 0 for v in scores[key]), (split, who, key))
        for phase in ("load", "base_validation", "base_test", "training", "validation_checks", "tuned_test",
                      "bootstrap", "total"):
            self.assertIsInstance(report["seconds"][phase], float, phase)
        training = report["training"]
        self.assertEqual((training["steps_asked"], training["steps_done"], training["samples"]), (60, 60, 104))
        self.assertGreater(training["steps_per_s"], 0)
        self.assertLess(training["mean_loss_last_20"], training["mean_loss_first_20"])
        self.assertEqual(report, json.loads(json.dumps(finetune.json_ready(self.returned))))

    def test_report_markdown(self):
        with open(os.path.join(self.out, "finetune_report.md")) as f:
            text = f.read()
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("### Fine-tuning the weather model"))
        self.assertIn("| score | 6 h | 24 h |", lines)
        self.assertIn("|---|---|---|", lines)
        table = [line for line in lines if line.startswith("| ") and not line.startswith("| score")]
        self.assertEqual([line.split(" | ")[0] for line in table],
                         ["| Z500 RMSE (m2/s2)", "| T850 RMSE (K)", "| T2M RMSE (K)",
                          "| all 20 channels RMSE (z-units)", "| O'Hare 2-m temp. error (C)",
                          "| persistence: Z500 RMSE (m2/s2)", "| persistence: T850 RMSE (K)",
                          "| persistence: T2M RMSE (K)"])
        self.assertTrue(all(line.count("|") == 4 for line in table))
        row = self.report["comparison"]["rows"][1]               # Z500 at 24 h
        self.assertIn("%.2f -> %.2f (%+.1f%%)" % (row["base"], row["tuned"], row["change_pct"]), table[0])
        self.assertIn("- Z500 at 24 h: **better** (", text)
        self.assertIn(self.report["overall"]["sentence"], text)
        self.assertIn("How the test weeks were kept apart: the analyses were split by date before anything ran, "
                      "training used only those up to 2026-02-13T18, the 9 days before 2026-02-24T00 were used "
                      "only to pick the best weights, and the weeks from 2026-02-24T00 on were scored once, at "
                      "the end.", text)
        self.assertIn("Test: 9 forecasts started at 00 and 12 UTC from 2026-02-24T00 to 2026-03-03T00", text)
        self.assertTrue(all(ord(ch) < 128 for ch in text))       # plain ASCII pastes anywhere

    def test_tuned_pt_is_laid_out_like_final_pt(self):
        saved = self.saved                                       # loaded with weights_only=True
        for key in ("model", "step", "seed", "base", "finetune", "report"):
            self.assertIn(key, saved)
        self.assertEqual(sorted(saved["model"]), ["a"])
        Scale(0.0).load_state_dict(saved["model"])
        self.assertEqual(saved["step"], self.report["selection"]["best_step"])
        self.assertEqual((saved["seed"], saved["base"], saved["stage"], saved["complete"]),
                         (1, "base.pt", "finetune", True))
        self.assertEqual(saved["finetune"], self.report["settings"])
        summary = saved["report"]
        self.assertEqual((summary["complete"], summary["steps_done"], summary["checked_on_validation"]),
                         (True, 60, True))
        self.assertEqual(summary["validation_score"], self.report["selection"]["best_score"])
        self.assertEqual(summary["overall"], self.report["overall"])
        self.assertEqual(summary["verdicts"], self.report["verdicts"])
        self.assertEqual(summary["test_rows"], self.report["comparison"]["rows"])
        self.assertEqual(summary["test"], ["2026-02-24T00", "2026-03-04T00"])

    def test_log_has_one_line_per_step(self):
        log = read_log(self.out)
        steps = [line for line in log if "event" not in line]
        self.assertEqual([line["step"] for line in steps], list(range(1, 61)))
        for line in steps:
            want = finetune.learning_rate(line["step"] - 1, 0.02, 5, 60)
            self.assertEqual(line["lr"], want)
            self.assertTrue(math.isfinite(line["loss"]) and math.isfinite(line["grad_norm"]))
        self.assertEqual([line["event"] for line in log if "event" in line],
                         ["start", "validation", "validation", "validation", "end"])
        self.assertEqual(log[0]["samples"], 104)
        self.assertEqual((log[-1]["step"], log[-1]["complete"]), (60, True))
        # One printed line per 20 steps and one per validation check.
        self.assertEqual([line.split("  loss")[0] for line in self.printed.splitlines() if line.startswith("step ")],
                         ["step   20/60", "step   40/60", "step   60/60"])
        self.assertEqual(sum(line.startswith("validation after step") for line in self.printed.splitlines()), 3)

    def test_the_log_is_appended_to(self):
        out = new_out()
        for _ in range(2):
            code, _, _, _ = run_tiny(Scale(1.0), self.data, self.times,
                                     ["--steps", 3, "--eval-every", 3, "--warmup", 0, "--lr", 0.02], out=out)
            self.assertEqual(code, 0)
        self.assertEqual([line["step"] for line in read_log(out) if "event" not in line], [1, 2, 3, 1, 2, 3])

    def test_the_same_seed_gives_the_same_run(self):
        extra = ["--steps", 8, "--eval-every", 4, "--warmup", 2, "--lr", 0.02]
        runs = [run_tiny(Scale(1.0), self.data, self.times, extra + ["--seed", seed]) for seed in (5, 5, 6)]
        a = [load_weights(os.path.join(out, "tuned.pt"))["model"]["a"].item() for _, _, _, out in runs]
        self.assertEqual(a[0], a[1])
        self.assertNotEqual(a[0], a[2])
        self.assertEqual(runs[0][1]["comparison"]["rows"], runs[1][1]["comparison"]["rows"])


class HonestyTest(unittest.TestCase):
    """What must hold for the test weeks to mean anything."""

    def test_training_reads_training_rows_only_and_test_rows_are_scored_twice(self):
        rows, times = decay_blocks([0.9] * 21)
        model = Scale(1.0)
        data = Watched(rows, model)
        code, _, _, _ = run_tiny(model, data, times, ["--steps", 60, "--lr", 0.02, "--warmup", 5,
                                                      "--eval-every", 20])
        self.assertEqual(code, 0)
        train, validation, test = range(0, 134), range(135, 162), range(162, 189)       # row 134 is the gap
        self.assertEqual([times[r].strftime("%Y-%m-%dT%H") for r in (133, 134, 135, 162)],
                         ["2026-02-13T18", "2026-02-14T00", "2026-02-15T00", "2026-02-24T00"])
        # Training read every training row and nothing else: not the gap, not validation, not test.
        self.assertEqual(sorted(set(data.read[True])), list(train))
        self.assertEqual(len(data.read[True]), 60 * 16 * 3)      # 60 steps x 16 samples x (1 input + 2 targets)
        # Each sample is three rows in a row, 6 hours apart: nothing reaches across a missing cycle.
        for n in range(0, len(data.read[True]), 3):
            first, second, third = data.read[True][n:n + 3]
            self.assertEqual((second, third), (first + 1, first + 2))
            self.assertEqual(times[third] - times[first], 2 * H6)
        # Scoring read validation and test rows only. One scoring pass loads 9 starts and their 4 leads.
        scored = data.read[False]
        self.assertTrue(set(scored) <= set(validation) | set(test))
        one_pass = 9 * (1 + 4)
        # Validation: the base model once, then the three checks. Test: the base model, then the best weights.
        self.assertEqual(sum(row in validation for row in scored), (1 + 3) * one_pass)
        self.assertEqual(sum(row in test for row in scored), 2 * one_pass)
        # The test rows were read only before training began and after it ended.
        self.assertTrue(all(row in validation for row in scored[2 * one_pass:(2 + 3) * one_pass]))
        self.assertTrue(all(row in test for row in scored[(2 + 3) * one_pass:]))

    def test_optimizer_settings_and_gradient_clipping(self):
        # The same made-up weather, eight times larger, so that the gradient is far above the clip of 1.
        rows, times = decay_blocks([0.9] * 21)
        rows = rows * np.float16(8.0)
        clips, real_clip, real_adamw = [], torch.nn.utils.clip_grad_norm_, torch.optim.AdamW

        def watched_clip(parameters, max_norm):
            parameters = list(parameters)
            before = real_clip(parameters, max_norm)
            after = math.sqrt(sum(float((p.grad ** 2).sum()) for p in parameters))
            clips.append((max_norm, before.item(), after))
            return before

        with mock.patch.object(torch.nn.utils, "clip_grad_norm_", watched_clip), \
                mock.patch.object(torch.optim, "AdamW", wraps=real_adamw) as adamw:
            code, _, _, out = run_tiny(Scale(1.0), rows, times, ["--steps", 5, "--warmup", 0, "--lr", 0.001,
                                                                 "--eval-every", 5])
        self.assertEqual(code, 0)
        self.assertEqual(adamw.call_count, 1)
        self.assertEqual(adamw.call_args[1]["betas"], (0.9, 0.95))
        self.assertEqual(adamw.call_args[1]["weight_decay"], 0.01)
        self.assertEqual(len(clips), 5)                          # once per step
        steps = [line for line in read_log(out) if "event" not in line]
        for (max_norm, before, after), line in zip(clips, steps):
            self.assertEqual(max_norm, 1.0)
            self.assertGreater(before, 5.0)                      # the gradient as it came ...
            self.assertAlmostEqual(after, 1.0, places=4)         # ... and as the optimizer got it
            self.assertEqual(line["grad_norm"], before)          # the log keeps the norm before clipping

    def test_compile_is_used_for_the_training_steps_only(self):
        # torch.compile is swapped for a wrapper that counts its calls (no compiler is needed for this test).
        rows, times = decay_blocks([0.9] * 21)

        class Counted(torch.nn.Module):
            def __init__(self, inner):
                super(Counted, self).__init__()
                self.inner, self.calls = inner, 0

            def forward(self, x):
                self.calls += 1
                return self.inner(x)

        made = []

        def fake_compile(model):
            made.append(Counted(model))
            return made[-1]

        extra = ["--steps", 6, "--warmup", 0, "--lr", 0.02, "--eval-every", 3, "--rollout", 2]
        with mock.patch.object(torch, "compile", fake_compile):
            code, report, _, out = run_tiny(Scale(1.0), rows, times, extra + ["--compile"])
        self.assertEqual(code, 0)
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].calls, 6 * 2)                   # two fed-back steps per training step; no scoring
        self.assertTrue(report["settings"]["compile"])
        saved = load_weights(os.path.join(out, "tuned.pt"))
        self.assertEqual(sorted(saved["model"]), ["a"])          # the plain model's names, not the wrapper's
        _, plain, _, plain_out = run_tiny(Scale(1.0), rows, times, extra)          # the same run without --compile
        plain_saved = load_weights(os.path.join(plain_out, "tuned.pt"))
        self.assertEqual(saved["model"]["a"].item(), plain_saved["model"]["a"].item())
        self.assertEqual(report["scores"]["test"]["tuned"], plain["scores"]["test"]["tuned"])

    def test_a_compile_that_fails_says_what_to_do(self):
        rows, times = decay_blocks([0.9] * 21)

        class Broken(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("no compiler\nfound here")

        out = new_out()
        with mock.patch.object(torch, "compile", lambda model: Broken()):
            with self.assertRaises(finetune.FinetuneError) as caught:
                run_tiny(Scale(1.0), rows, times, ["--steps", 6, "--warmup", 0, "--compile"], out=out)
        self.assertEqual(str(caught.exception), "the first training step failed with --compile (RuntimeError: "
                         "no compiler found here). Run again without --compile.")
        self.assertFalse(os.path.exists(os.path.join(out, "tuned.pt")))

    def test_the_best_check_is_kept_not_the_last_one(self):
        # Training follows another law (0.8) than validation and test (0.9). Starting from 1.0 the model
        # passes 0.9 early and ends near 0.8: the best validation check is an early one.
        rows, times = decay_blocks([0.8] * 15 + [0.9] * 6)
        model = Scale(1.0)
        code, report, _, out = run_tiny(model, rows, times, ["--steps", 40, "--lr", 0.02, "--warmup", 0,
                                                             "--eval-every", 5])
        self.assertEqual(code, 0)
        checks = report["selection"]["checks"]
        self.assertEqual([check["step"] for check in checks], [5, 10, 15, 20, 25, 30, 35, 40])
        best = min(checks, key=lambda check: check["score"])
        self.assertLess(best["step"], 40)
        self.assertLess(best["score"], 0.5 * checks[-1]["score"])
        self.assertEqual(report["selection"]["best_step"], best["step"])
        self.assertEqual([check["best_so_far"] for check in checks],
                         [check["score"] == min(c["score"] for c in checks[:n + 1]) for n, check in enumerate(checks)])
        saved = load_weights(os.path.join(out, "tuned.pt"))
        a = saved["model"]["a"].item()
        self.assertEqual(saved["step"], best["step"])
        self.assertLess(abs(a - 0.9), 0.05)                      # near 0.9, not near 0.8
        self.assertEqual(model.a.item(), a)                      # the model was put back to the best weights
        self.assertEqual(report["training"]["steps_done"], 40)
        # The test scores in the report are those of the saved weights, not of the last step.
        plan = finetune.make_plan(finetune.parse_args(["--init", "x", "--analyses", "x", "--out", "x"] + TINY_SPLIT),
                                  times)
        errors = finetune.evaluate(Scale(a), rows, plan["starts"]["test"], 4, "cpu", tiny_scoring())
        again = finetune.scores_of(finetune.start_values(errors["model"], STD))
        self.assertEqual(again, report["scores"]["test"]["tuned"])
        errors = finetune.evaluate(Scale(a), rows, plan["starts"]["validation"], 4, "cpu", tiny_scoring())
        again = finetune.scores_of(finetune.start_values(errors["model"], STD))
        self.assertEqual(again, report["scores"]["validation"]["tuned"])


class NotBetterTest(unittest.TestCase):
    def test_a_tuned_model_that_is_worse_is_saved_and_the_verdict_says_so(self):
        # The training weeks follow another law (next = 0.8 x current) than validation and test (0.9), and the
        # base model is already right for those. Fine-tuning can only make it worse.
        rows, times = decay_blocks([0.8] * 15 + [0.9] * 6)
        model = Scale(0.9)
        code, report, printed, out = run_tiny(model, rows, times, ["--steps", 40, "--lr", 0.02, "--warmup", 5,
                                                                   "--eval-every", 20])
        self.assertEqual(code, 0)                                # still a finished run
        saved = load_weights(os.path.join(out, "tuned.pt"))      # and the weights are still saved
        self.assertLess(saved["model"]["a"].item(), 0.88)        # pulled towards 0.8
        self.assertFalse(report["selection"]["beats_base_on_validation"])
        self.assertGreater(report["selection"]["best_score"], 1.0)
        self.assertEqual(report["verdicts"],
                         {"z500": {"24h": "worse"}, "t850": {"24h": "worse"}, "t2m": {"24h": "worse"}})
        self.assertEqual(report["overall"]["word"], "worse")
        for text in (report["overall"]["sentence"], printed, saved["report"]["overall"]["sentence"]):
            self.assertIn("The tuned model is NOT better than the base model on the test weeks: it is worse", text)
            self.assertIn("tuned.pt was saved anyway", text)
        self.assertIn("No validation check beat the base model", printed)
        with open(os.path.join(out, "finetune_report.md")) as f:
            text = f.read()
        self.assertIn("No validation check beat the base model.", text)
        self.assertIn("- Z500 at 24 h: **worse** (", text)
        self.assertIn("NOT better", text)

    def test_a_tuned_model_that_is_the_same_gets_no_clear_change(self):
        # A learning rate too small to change a float32 number: the tuned model is the base model.
        rows, times = decay_blocks([0.9] * 21)
        code, report, printed, out = run_tiny(Scale(1.0), rows, times, ["--steps", 4, "--lr", 1e-12, "--warmup", 0,
                                                                        "--eval-every", 2])
        self.assertEqual(code, 0)
        self.assertEqual(load_weights(os.path.join(out, "tuned.pt"))["model"]["a"].item(), 1.0)
        for row in report["comparison"]["rows"]:
            self.assertEqual((row["change_pct"], row["ci90_pct"], row["verdict"]), (0.0, [0.0, 0.0], "no clear change"))
        self.assertEqual(report["overall"]["word"], "no clear change")
        self.assertIn("NOT measurably better", report["overall"]["sentence"])
        self.assertIn("tuned.pt was saved anyway", printed)
        self.assertEqual(report["selection"]["best_score"], 1.0)
        self.assertFalse(report["selection"]["beats_base_on_validation"])

    def test_a_loss_that_is_not_finite_stops_the_run(self):
        rows, times = decay_blocks([0.9] * 21)
        rows = rows.copy()
        rows[5] = np.inf                                         # one bad analysis among the training rows
        out = new_out()
        with self.assertRaises(finetune.FinetuneError) as caught:
            run_tiny(Scale(1.0), rows, times, ["--steps", 30, "--warmup", 5, "--eval-every", 10], out=out)
        self.assertIn("not finite numbers", str(caught.exception))
        self.assertIn("Try a lower --lr", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(out, "tuned.pt")))

    def test_a_base_model_with_broken_forecasts_is_refused(self):
        rows, times = decay_blocks([0.9] * 21)
        out = new_out()
        with self.assertRaises(finetune.FinetuneError) as caught:
            run_tiny(Scale(float("nan")), rows, times, ["--steps", 4, "--warmup", 0], out=out)
        self.assertIn("the base model on validation", str(caught.exception))
        self.assertEqual(os.listdir(out), [])


class StopTest(unittest.TestCase):
    """SIGTERM: stop after the current step, save the best weights so far, exit with code 3."""

    EXTRA = ["--steps", 40, "--lr", 0.02, "--warmup", 5, "--eval-every", 10, "--rollout", 2]

    def setUp(self):
        self.rows, self.times = decay_blocks([0.9] * 21)
        # If finetune.py did not catch the signal, this would: the test then fails instead of dying.
        self.strays = []
        self.before = signal.signal(signal.SIGTERM, lambda signum, frame: self.strays.append(signum))
        self.ours = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, signal.SIG_DFL if self.before is None else self.before)

    def check_handlers_are_back(self):
        self.assertEqual(self.strays, [])
        self.assertIs(signal.getsignal(signal.SIGTERM), self.ours)

    def test_sigterm_after_a_validation_check_keeps_the_best_weights_so_far(self):
        model = SignalAt(1.0, call=2 * 25 - 1)                   # two forward calls per step: during step 25
        code, report, printed, out = run_tiny(model, self.rows, self.times, self.EXTRA)
        self.check_handlers_are_back()
        self.assertEqual(code, 3)
        self.assertEqual(report["training"]["steps_done"], 25)   # the step it was in was finished
        self.assertEqual([check["step"] for check in report["selection"]["checks"]], [10, 20])
        saved = load_weights(os.path.join(out, "tuned.pt"))
        self.assertEqual(saved["step"], report["selection"]["best_step"])
        self.assertIn(saved["step"], (10, 20))
        self.assertEqual((saved["complete"], saved["report"]["stopped_early"]), (False, True))
        self.assertTrue(saved["report"]["checked_on_validation"])
        self.assertLess(abs(saved["model"]["a"].item() - 0.9), 0.1)
        on_disk = read_report(out)
        self.assertEqual((on_disk["complete"], on_disk["stopped_early"]), (False, True))
        self.assertIsNone(on_disk["comparison"])
        self.assertIsNone(on_disk["scores"]["test"]["tuned"])
        self.assertIsNotNone(on_disk["scores"]["validation"]["tuned"])
        self.assertIn("stopped by a signal after step 25", printed)
        self.assertIn("NOT scored on the test weeks", printed)
        with open(os.path.join(out, "finetune_report.md")) as f:
            text = f.read()
        self.assertIn("The run was stopped early", text)
        self.assertIn("tuned.pt holds the weights after step %d, the best of 2 validation checks" % saved["step"], text)
        self.assertIn("have not been scored on them", text)
        self.assertNotIn("scored once, at the end", text)
        self.assertNotIn("->", text)                             # the table shows the base model only
        log = read_log(out)
        self.assertEqual(len([line for line in log if "event" not in line]), 25)
        self.assertEqual((log[-1]["event"], log[-1]["complete"], log[-1]["stopped_early"]), ("end", False, True))

    def test_sigterm_before_the_first_validation_check_keeps_the_weights_as_they_are(self):
        model = SignalAt(1.0, call=2 * 3 - 1)                    # during step 3
        code, report, printed, out = run_tiny(model, self.rows, self.times, self.EXTRA)
        self.check_handlers_are_back()
        self.assertEqual(code, 3)
        self.assertEqual(report["training"]["steps_done"], 3)
        saved = load_weights(os.path.join(out, "tuned.pt"))
        self.assertEqual((saved["step"], saved["complete"]), (3, False))
        self.assertFalse(saved["report"]["checked_on_validation"])
        self.assertEqual(saved["model"]["a"].item(), model.a.item())
        self.assertLess(saved["model"]["a"].item(), 1.0)
        self.assertIn("before a validation check had passed", printed)
        self.assertIsNone(read_report(out)["scores"]["validation"]["tuned"])
        with open(os.path.join(out, "finetune_report.md")) as f:
            self.assertIn("tuned.pt holds the weights after step 3. They were not scored on validation.", f.read())

    def test_sigterm_during_a_validation_check_does_not_count_that_check(self):
        # Eval-mode calls: 9 starts x 4 leads fit in one batch, so each scoring pass makes 4 calls. The base
        # model takes 8 (validation and test), the check after step 10 takes 4, and call 14 is in the next check.
        model = SignalAt(1.0, call=14, in_training=False)
        code, report, _, out = run_tiny(model, self.rows, self.times, self.EXTRA)
        self.check_handlers_are_back()
        self.assertEqual(code, 3)
        self.assertEqual(report["training"]["steps_done"], 20)
        self.assertEqual([check["step"] for check in report["selection"]["checks"]], [10])
        self.assertEqual(load_weights(os.path.join(out, "tuned.pt"))["step"], 10)

    def test_sigterm_while_the_base_model_is_scored_writes_nothing(self):
        model = SignalAt(1.0, call=2, in_training=False)
        code, _, printed, out = run_tiny(model, self.rows, self.times, self.EXTRA)
        self.check_handlers_are_back()
        self.assertEqual(code, 3)
        self.assertEqual(os.listdir(out), [])
        self.assertIn("nothing was written", printed)

    def test_sigterm_just_before_training_saves_no_weights(self):
        # Eval-mode call 8 is the base model's last step on test: the signal is seen when training would start.
        model = SignalAt(1.0, call=8, in_training=False)
        code, report, printed, out = run_tiny(model, self.rows, self.times, self.EXTRA)
        self.check_handlers_are_back()
        self.assertEqual(code, 3)
        self.assertEqual(report["training"]["steps_done"], 0)
        self.assertEqual(sorted(os.listdir(out)), ["finetune_report.json", "finetune_report.md", "log.jsonl"])
        self.assertIn("no weights were saved", printed)
        with open(os.path.join(out, "finetune_report.md")) as f:
            self.assertIn("Fine-tuned for 0 steps", f.read())
        on_disk = read_report(out)
        self.assertEqual((on_disk["complete"], on_disk["stopped_early"]), (False, True))
        self.assertIsNotNone(on_disk["scores"]["test"]["base"])
        self.assertEqual(model.a.item(), 1.0)

    def test_sigterm_while_the_tuned_model_is_scored_on_test_still_saves_it(self):
        # 8 calls for the base model, 4 for each of the checks after steps 2 and 4, then the test pass.
        model = SignalAt(1.0, call=8 + 4 + 4 + 2, in_training=False)
        code, report, _, out = run_tiny(model, self.rows, self.times,
                                        ["--steps", 4, "--eval-every", 2, "--warmup", 0, "--lr", 0.02])
        self.check_handlers_are_back()
        self.assertEqual(code, 3)
        self.assertEqual(report["training"]["steps_done"], 4)
        saved = load_weights(os.path.join(out, "tuned.pt"))
        self.assertEqual((saved["complete"], saved["report"]["checked_on_validation"]), (False, True))
        self.assertIsNone(read_report(out)["comparison"])


# ------------------------------------------------------------------------------------------------ the command line

def round_trip(case, init, extra):
    """Fine-tune from a weights file through the command line, then score the saved tuned.pt with
    --eval-only, as one would on the other machine. Returns the two reports."""
    with mock.patch.object(finetune, "MIN_SAMPLES", 8):          # the small data set has 10 training samples
        code, printed, errors, out = run_main(extra, init=init)
    case.assertEqual(code, 0, errors + printed[-2000:])
    case.assertEqual(sorted(os.listdir(out)), ["finetune_report.json", "finetune_report.md", "log.jsonl", "tuned.pt"])
    tuned = os.path.join(out, "tuned.pt")
    report = read_report(out)
    case.assertTrue(report["complete"])
    saved = load_weights(tuned)
    case.assertEqual((saved["base"], saved["complete"]), (init, True))
    code, _, errors, check = run_main(["--eval-only"] + extra, init=tuned)
    case.assertEqual(code, 0, errors)
    return report, read_report(check)


class CommandLineTest(unittest.TestCase):
    """finetune.main() on files, on the real 120 x 240 grid. The small model stands in for the AFNO."""

    def save_init(self, a, name):
        """A weights file laid out as weather/train.py writes it."""
        path = os.path.join(TMP, name)
        torch.save({"model": Scale(a).state_dict(), "step": 1564, "seed": 1, "stage": 2, "complete": True}, path)
        return path

    def test_eval_only_writes_the_report_and_no_tuned_pt(self):
        code, printed, errors, out = run_main(["--eval-only"], model=Scale(0.9))
        self.assertEqual(code, 0, errors)
        self.assertEqual(sorted(os.listdir(out)), ["finetune_report.json", "finetune_report.md"])
        report = read_report(out)
        self.assertEqual((report["mode"], report["complete"], report["stopped_early"]), ("eval-only", True, False))
        for name in ("selection", "training", "comparison", "verdicts", "overall"):
            self.assertIsNone(report[name], name)
        self.assertEqual(report["settings"]["stats"], CLI_DATA)  # stats.json was found beside the analyses
        self.assertEqual(report["splits"]["train"]["rows"], 12)
        self.assertEqual(report["splits"]["gap"]["rows"], 4)
        for split in ("validation", "test"):
            self.assertEqual((report["splits"][split]["rows"], report["splits"][split]["starts"]), (20, 8))
            scores = report["scores"][split]
            self.assertIsNone(scores["tuned"])
            for key in ALL_KEYS:
                self.assertEqual(len(scores["base"][key]), 4)
                self.assertTrue(all(v > 0 for v in scores["base"][key]), (split, key))
            for key in RMSE_KEYS + ["all_channels_rmse_z"]:
                # 0.9 is the right number for this made-up weather, so the model beats "nothing changes".
                self.assertLess(scores["base"][key][3], scores["persistence"][key][3], (split, key))
            # Physical RMSE is the z-score RMSE of that channel times its standard deviation, and the
            # made-up noise is the same size in every channel.
            self.assertAlmostEqual(scores["base"]["z500_rmse"][0] / 100.0, scores["base"]["all_channels_rmse_z"][0],
                                   delta=0.02 * scores["base"]["all_channels_rmse_z"][0])
        with open(os.path.join(out, "finetune_report.md")) as f:
            text = f.read()
        self.assertTrue(text.startswith("### The weather model on GFS analyses"))
        self.assertIn("This run trained nothing", text)
        self.assertIn("| score | 6 h | 24 h |", text)
        self.assertNotIn("->", text)
        self.assertIn("base model on test: 8 forecast starts", printed)

    def test_a_model_that_changes_nothing_scores_as_persistence_on_the_real_grid(self):
        code, _, _, out = run_main(["--eval-only"], model=Scale(1.0))
        self.assertEqual(code, 0)
        report = read_report(out)
        for split in ("validation", "test"):
            self.assertEqual(report["scores"][split]["base"], report["scores"][split]["persistence"])
        # The made-up weather is next = 0.9 x current + noise with variance 1: after k steps the RMSE of
        # "nothing changes" is sqrt(2 * (1 - 0.9^k)) z-units, here to within a few percent.
        z = report["scores"]["test"]["persistence"]["all_channels_rmse_z"]
        for k in range(4):
            self.assertAlmostEqual(z[k], math.sqrt(2.0 * (1.0 - 0.9 ** (k + 1))), delta=0.03)

    def test_persistence_on_the_real_grid_by_definition(self):
        """The scores of "nothing changes" worked out again from the files, the long way round: z-scores to
        physical units with stats.json's mean and std, cos(latitude) weights from the definition, and the
        four grid points around O'Hare by hand."""
        code, _, _, out = run_main(["--eval-only"], model=Scale(1.0))
        self.assertEqual(code, 0)
        report = read_report(out)
        data = np.load(os.path.join(CLI_DATA, "analyses.npy"), mmap_mode="r")
        cos_lat = np.array([math.cos(math.radians(-88.5 + 1.5 * i)) for i in range(120)])
        weights = (cos_lat / cos_lat.mean())[:, None]
        starts = list(range(36, 52, 2))          # 00 and 12 UTC from Jan 10; the last 24 hours end at row 54
        self.assertEqual(report["splits"]["test"]["start_times"],
                         [(T0 + row * H6).strftime("%Y-%m-%dT%H") for row in starts])
        fy = (41.9602 + 88.5) / 1.5 - 86         # O'Hare lies this far from row 86 towards row 87 ...
        fx = (360.0 - 87.9316) / 1.5 - 181       # ... and this far from column 181 towards column 182

        def physical(row, channel):
            return data[row, channel].astype(np.float64) * STD[channel] + MEAN[channel]

        def ohare_c(row):
            t = physical(row, wx.CH_T2M) - 273.15
            return ((1 - fy) * (1 - fx) * t[86, 181] + (1 - fy) * fx * t[86, 182]
                    + fy * (1 - fx) * t[87, 181] + fy * fx * t[87, 182])

        got = report["scores"]["test"]["persistence"]
        for k in range(4):                       # leads 6, 12, 18, 24 h
            for key, channel in (("z500_rmse", 14), ("t850_rmse", 5), ("t2m_rmse", 2)):
                mse = [(weights * (physical(row, channel) - physical(row + k + 1, channel)) ** 2).mean()
                       for row in starts]
                want = math.sqrt(sum(mse) / len(mse))
                self.assertTrue(math.isclose(got[key][k], want, rel_tol=1e-9), (key, k, got[key][k], want))
            want = sum(abs(ohare_c(row) - ohare_c(row + k + 1)) for row in starts) / len(starts)
            self.assertTrue(math.isclose(got["ohare_t2m_mae_c"][k], want, rel_tol=1e-9), (k, want))
            mse = [np.mean([(weights * (data[row, c].astype(np.float64) - data[row + k + 1, c]) ** 2).mean()
                            for c in range(20)]) for row in starts]
            want = math.sqrt(sum(mse) / len(mse))
            self.assertTrue(math.isclose(got["all_channels_rmse_z"][k], want, rel_tol=1e-9), (k, want))

    def test_the_weights_file_is_read_the_way_train_py_wrote_it(self):
        init = self.save_init(0.8, "init_08.pt")
        with mock.patch.object(wx, "build_afno", lambda: Scale(1.0)):
            code, _, errors, out = run_main(["--eval-only"], init=init)
        self.assertEqual(code, 0, errors)
        _, _, _, direct = run_main(["--eval-only"], model=Scale(0.8))
        self.assertEqual(read_report(out)["scores"], read_report(direct)["scores"])
        self.assertEqual(read_report(out)["settings"]["init"], init)

    def test_stats_come_from_the_stats_folder_or_from_wx_data(self):
        # A folder with the analyses but no stats.json, and a stats.json elsewhere with every std doubled.
        bare = os.path.join(TMP, "gfs_without_stats")
        os.makedirs(bare)
        for name in ("analyses.npy", "analyses.json"):
            os.symlink(os.path.join(CLI_DATA, name), os.path.join(bare, name))
        elsewhere = os.path.join(TMP, "stats_elsewhere")
        os.makedirs(elsewhere)
        write_stats(elsewhere, 2.0 * STD)
        _, _, _, usual = run_main(["--eval-only", "--max-starts", 2], model=Scale(0.9))
        usual = read_report(usual)["scores"]["test"]["base"]
        with mock.patch.dict(os.environ, {"WX_DATA": elsewhere}):
            code, _, _, out = run_main(["--eval-only", "--max-starts", 2], model=Scale(0.9), analyses=bare)
        self.assertEqual(code, 0)
        report = read_report(out)
        self.assertEqual(report["settings"]["stats"], elsewhere)
        self.assertAlmostEqual(report["scores"]["test"]["base"]["t850_rmse"][3], 2.0 * usual["t850_rmse"][3], places=9)
        self.assertEqual(report["scores"]["test"]["base"]["all_channels_rmse_z"], usual["all_channels_rmse_z"])
        code, _, _, out = run_main(["--eval-only", "--max-starts", 2, "--stats", elsewhere], model=Scale(0.9))
        self.assertEqual(code, 0)
        self.assertEqual(read_report(out)["settings"]["stats"], elsewhere)
        self.assertEqual(read_report(out)["splits"]["test"]["starts"], 2)        # --max-starts
        self.assertEqual(read_report(out)["splits"]["test"]["starts_available"], 8)
        with mock.patch.dict(os.environ, {"WX_DATA": os.path.join(TMP, "nowhere")}):
            code, _, errors, _ = run_main(["--eval-only"], model=Scale(0.9), analyses=bare)
        self.assertEqual(code, 1)
        self.assertIn("there is no stats.json in", errors)
        self.assertIn("Pass --stats DIR", errors)

    def test_fine_tune_from_the_command_line_and_score_the_result_again(self):
        init = self.save_init(1.0, "init_10.pt")
        with mock.patch.object(wx, "build_afno", lambda: Scale(0.0)):
            report, check = round_trip(self, init, ["--steps", 6, "--eval-every", 3, "--warmup", 0, "--lr", 0.02,
                                                   "--batch", 4])
        self.assertEqual(report["splits"]["train"]["samples"], 10)
        self.assertEqual(report["training"]["steps_done"], 6)
        self.assertTrue(report["selection"]["beats_base_on_validation"])         # 1.0 moved towards 0.9
        self.assertEqual(report["overall"]["word"], "better")
        # The saved weights, scored on their own, give the tuned model's scores again.
        for split in ("validation", "test"):
            self.assertEqual(check["scores"][split]["base"], report["scores"][split]["tuned"])
            self.assertEqual(check["scores"][split]["persistence"], report["scores"][split]["persistence"])

    def test_problems_are_reported_in_one_line(self):
        empty = os.path.join(TMP, "empty")
        os.makedirs(empty)
        wrong = os.path.join(TMP, "wrong_shape")
        os.makedirs(wrong)
        np.save(os.path.join(wrong, "analyses.npy"), np.zeros((5, 20, 4, 8), np.float16))
        short = os.path.join(TMP, "times_short")
        os.makedirs(short)
        os.symlink(os.path.join(CLI_DATA, "analyses.npy"), os.path.join(short, "analyses.npy"))
        for folder in (wrong, short):
            with open(os.path.join(folder, "analyses.json"), "w") as f:
                json.dump({"times": ["2026-01-01T00", "2026-01-01T06"]}, f)
        not_weights = os.path.join(TMP, "not_weights.pt")
        torch.save({"weights": Scale(1.0).state_dict()}, not_weights)
        other_model = os.path.join(TMP, "other_model.pt")
        torch.save({"model": torch.nn.Linear(2, 2).state_dict()}, other_model)
        cases = [
            (dict(analyses=empty, model=Scale(1.0)), ["analyses.npy is missing", "weather/gfs_archive.py"]),
            (dict(analyses=wrong, model=Scale(1.0)), ["has shape (5, 20, 4, 8)", "(rows, 20, 120, 240) float16"]),
            (dict(analyses=short, model=Scale(1.0)), ["lists 2 times, but analyses.npy has 56 rows"]),
            (dict(init=os.path.join(TMP, "missing.pt")), ["missing.pt does not exist"]),
            (dict(init=not_weights), ["has no 'model' entry"]),
            (dict(init=other_model), ["do not fit the model"]),
            (dict(split=[], model=Scale(1.0)), ["The test split (no rows) gives 0 forecast starts",
                                                "at least 8 are needed", "--test-from (now 2026-10-01T00)"]),
            (dict(split=CLI_SPLIT[:2] + ["--val-days", "1", "--lead-hours", "24"], model=Scale(1.0)),
             ["The validation split", "at least 8 are needed"]),
        ]
        with mock.patch.object(wx, "build_afno", lambda: Scale(0.0)):
            for kwargs, wanted in cases:
                code, _, errors, out = run_main(["--eval-only"], **kwargs)
                self.assertEqual(code, 1, kwargs)
                said = failure(errors)
                for text in wanted:
                    self.assertIn(text, said)
                self.assertEqual(os.listdir(out), [], kwargs)
        # Fine-tuning (not --eval-only) needs 100 training samples; this folder has 10.
        code, _, errors, out = run_main([], model=Scale(1.0))
        self.assertEqual(code, 1)
        self.assertIn("gives 10 training samples; at least 100 are needed", errors)
        self.assertEqual(os.listdir(out), [])


class ProgramTest(unittest.TestCase):
    """weather/finetune.py run as a program, in its own process."""

    def program(self, *args):
        return subprocess.run([sys.executable, FINETUNE] + [str(a) for a in args], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=120)

    def test_help_names_every_option(self):
        done = self.program("--help")
        self.assertEqual(done.returncode, 0, done.stderr)
        for option in ("--init", "--analyses", "--out", "--stats", "--test-from", "--val-days", "--steps", "--lr",
                       "--warmup", "--batch", "--rollout", "--eval-every", "--lead-hours", "--device", "--compile",
                       "--seed", "--max-starts", "--eval-only"):
            self.assertIn(option, done.stdout)

    def test_a_refusal_is_one_line_and_exit_code_1(self):
        out = os.path.join(TMP, "never_made")
        done = self.program("--init", "missing.pt", "--analyses", os.path.join(TMP, "no_such_folder"), "--out", out)
        self.assertEqual(done.returncode, 1)
        self.assertIn("analyses.npy is missing", failure(done.stderr))
        self.assertFalse(os.path.exists(out))


@unittest.skipUnless(os.environ.get("WX_FINETUNE_SMOKE") == "1",
                     "set WX_FINETUNE_SMOKE=1 to run the real AFNO (a check for the GPU machines)")
class RealModelSmokeTest(unittest.TestCase):
    """The real AFNO with random weights through the command line: scoring, two training steps, the files,
    and scoring the saved weights again. It says nothing about forecast quality. It checks that the real
    model runs through every step of finetune.py on this machine."""

    def test_real_model_two_steps(self):
        torch.manual_seed(0)
        init = os.path.join(TMP, "afno_random.pt")
        torch.save({"model": wx.build_afno().state_dict(), "step": 0, "seed": 0, "stage": 2, "complete": True}, init)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        report, check = round_trip(self, init, ["--steps", 2, "--eval-every", 1, "--warmup", 0, "--batch", 2,
                                                "--max-starts", 2, "--device", device])
        self.assertEqual(report["params"], 19128832)
        self.assertEqual((report["device"], report["bf16"]), (device, device == "cuda"))
        self.assertEqual(report["training"]["steps_done"], 2)
        self.assertEqual(len(report["comparison"]["rows"]), 10)
        before = load_weights(init)["model"]
        after = load_weights(os.path.join(report["settings"]["out"], "tuned.pt"))["model"]
        self.assertEqual(sorted(before), sorted(after))
        self.assertTrue(any(not torch.equal(before[name], after[name]) for name in before))     # it did train
        for key in ALL_KEYS:
            got, want = check["scores"]["test"]["base"][key], report["scores"]["test"]["tuned"][key]
            self.assertTrue(np.allclose(got, want, rtol=1e-3), (key, got, want))


if __name__ == "__main__":
    unittest.main()
