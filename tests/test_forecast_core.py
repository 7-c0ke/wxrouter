"""Tests for weather/forecast_core.py: the ensemble noise, the grid interpolation, the state store, the units,
and a short run of the real model on CPU.

    .venv/bin/python -m unittest tests/test_forecast_core.py -v

All of them are skipped unless NumPy, PyTorch and PhysicsNeMo are installed and .local/ holds the test data:
.local/bundle (stats.json, bundle.json, 60 states from 2019-01-20T00) and the stand-in weights.
Nothing is written into .local/: the tests work in temp folders that are removed afterwards.

Written at the hackathon (Nov 13-14, 2026).
"""
import base64
import datetime
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(REPO, ".local", "bundle")
WEIGHTS = os.environ.get("WX_TEST_WEIGHTS", os.path.join(REPO, ".local", "weights", "model.pt"))
INIT = datetime.datetime(2019, 1, 27, 12)       # the start of the January 2019 cold wave forecast
OHARE = (41.9602, -87.9316)
HOURS_6 = datetime.timedelta(hours=6)


def why_not():
    """Why these tests cannot run on this machine; None if they can."""
    for path in (WEIGHTS, os.path.join(BUNDLE, "bundle.json"), os.path.join(BUNDLE, "stats.json")):
        if not os.path.exists(path):
            return "missing %s" % path
    for module in ("numpy", "torch", "physicsnemo"):
        if importlib.util.find_spec(module) is None:
            return "%s is not installed" % module
    return None


SKIP = why_not()
if SKIP is None:
    import numpy as np
    import torch

    sys.path.insert(0, os.path.join(REPO, "weather"))
    import forecast_core as core
    import wx_common as wx

    torch.set_num_threads(4)                    # be gentle with the laptop


DATA = None     # set by setUpModule: a temp data folder holding links to the bundle's own files


def setUpModule():
    """Other tools put new analyses into .local/bundle/states/. These tests must see the bundle alone, so they
    use a temp folder with links to its files and no states/ folder."""
    global DATA
    if SKIP is None:
        DATA = tempfile.mkdtemp(prefix="forecast_core_test_")
        for name in os.listdir(BUNDLE):
            if os.path.isfile(os.path.join(BUNDLE, name)):
                os.symlink(os.path.join(BUNDLE, name), os.path.join(DATA, name))


def tearDownModule():
    if DATA is not None:
        shutil.rmtree(DATA, True)


def packed(array):
    """An array as the npy_b64 text a client would send."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def temp_folder(test):
    folder = tempfile.mkdtemp(prefix="forecast_core_test_")
    test.addCleanup(shutil.rmtree, folder, True)
    return folder


@unittest.skipIf(SKIP, SKIP)
class NoiseTest(unittest.TestCase):

    def test_same_time_and_member_same_noise(self):
        first = core.member_noise(INIT, 3)
        self.assertEqual((first.shape, first.dtype), ((20, 120, 240), np.float32))
        self.assertTrue(np.array_equal(first, core.member_noise(INIT, 3)))
        self.assertFalse(np.array_equal(first, core.member_noise(INIT, 4)))                 # another member
        self.assertFalse(np.array_equal(first, core.member_noise(INIT + HOURS_6, 3)))       # another init time

    def test_noise_follows_the_recipe(self):
        rng = np.random.default_rng([2019012712, 6])                    # the seed: (YYYYMMDDHH, member), nothing else
        field = core.smooth(rng.standard_normal((20, 120, 240), dtype=np.float32))
        expected = field / field.std(axis=(1, 2), dtype=np.float64)[:, None, None] * 0.05
        self.assertLess(np.abs(core.member_noise(INIT, 6) - expected).max(), 1e-7)

    def test_standard_deviation_is_005_in_every_channel(self):
        for member in (1, 2, 39, 1000):
            noise = core.member_noise(INIT, member)
            spread = noise.std(axis=(1, 2), dtype=np.float64)
            self.assertEqual(spread.shape, (20,))
            self.assertLess(np.abs(spread - 0.05).max(), 1e-6, member)
            self.assertLess(np.abs(noise).max(), 0.5)

    def test_member_0_is_unperturbed(self):
        self.assertFalse(core.member_noise(INIT, 0).any())
        state = core.StateStore(DATA).get(INIT)
        first = core.perturbed_states(state, INIT, [0, 5])
        self.assertEqual((first.shape, first.dtype), ((2, 20, 120, 240), np.float32))
        self.assertTrue(np.array_equal(first[0], np.asarray(state, np.float32)))        # member 0: the analysis itself
        added = (first[1] - first[0]).std(axis=(1, 2), dtype=np.float64)
        self.assertLess(np.abs(added - 0.05).max(), 1e-3)                               # member 5: 0.05 of noise on top

    def test_a_member_is_the_same_in_any_batch(self):
        state = core.StateStore(DATA).get(INIT)
        together = core.perturbed_states(state, INIT, [1, 2, 3])
        alone = core.perturbed_states(state, INIT, [2])
        self.assertTrue(np.array_equal(together[1], alone[0]))

    def test_noise_is_smooth(self):
        noise = core.member_noise(INIT, 7)[wx.CH_T2M]
        east = np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1]
        north = np.corrcoef(noise[:-1].ravel(), noise[1:].ravel())[0, 1]
        around = np.corrcoef(noise[:, 0], noise[:, -1])[0, 1]                               # longitude 358.5 next to 0
        self.assertGreater(min(east, north, around), 0.85)

    def test_smooth_is_three_passes_of_1_4_6_4_1(self):
        def slow(field):                                    # the definition, one grid point at a time
            n_lat, n_lon = field.shape
            kernel = [1, 4, 6, 4, 1]
            for _ in range(3):
                out = np.zeros_like(field)
                for i in range(n_lat):
                    for j in range(n_lon):                  # longitude wraps
                        out[i, j] = sum(kernel[d + 2] * field[i, (j + d) % n_lon] for d in range(-2, 3)) / 16.0
                field, out = out, np.zeros_like(field)
                for i in range(n_lat):
                    for j in range(n_lon):                  # latitude: the edge rows are repeated
                        out[i, j] = sum(kernel[d + 2] * field[min(max(i + d, 0), n_lat - 1), j]
                                        for d in range(-2, 3)) / 16.0
                field = out
            return field

        field = np.random.default_rng(0).standard_normal((2, 7, 9))
        fast = core.smooth(field)
        for k in range(2):
            self.assertLess(np.abs(fast[k] - slow(field[k])).max(), 1e-12)
        flat = np.full((3, 5, 8), 2.5, np.float32)
        self.assertTrue(np.array_equal(core.smooth(flat), flat))                            # nothing to smooth
        self.assertEqual(core.smooth(flat).dtype, np.float32)


@unittest.skipIf(SKIP, SKIP)
class GridTest(unittest.TestCase):

    def setUp(self):
        self.field = np.random.default_rng(1).standard_normal((20, 120, 240)).astype(np.float32)

    def test_at_a_grid_point_it_is_the_grid_value(self):
        for i, j in ((0, 0), (119, 239), (87, 181), (60, 120), (1, 238)):
            lat, lon = float(wx.LATS[i]), float(wx.LONS[j])
            self.assertTrue(np.array_equal(core.at_point(self.field, lat, lon), self.field[:, i, j]), (i, j))
            tensor = core.at_point(torch.from_numpy(self.field), lat, lon)
            self.assertTrue(np.array_equal(tensor.numpy(), self.field[:, i, j]), (i, j))
            self.assertEqual(core.at_point(self.field[2], lat, lon), self.field[2, i, j])       # one channel

    def test_between_grid_points_it_is_bilinear(self):
        lat, lon = float(wx.LATS[40]) + 0.75, float(wx.LONS[10]) + 0.75                         # the middle of a cell
        middle = (self.field[:, 40, 10] + self.field[:, 40, 11] + self.field[:, 41, 10] + self.field[:, 41, 11]) / 4
        self.assertLess(np.abs(core.at_point(self.field, lat, lon) - middle).max(), 1e-6)
        quarter = 0.75 * self.field[:, 40, 10] + 0.25 * self.field[:, 41, 10]           # a quarter of the way north
        there = core.at_point(self.field, float(wx.LATS[40]) + 0.375, float(wx.LONS[10]))
        self.assertLess(np.abs(there - quarter).max(), 1e-6)

    def test_ohare_sits_between_these_four_points(self):
        weights = core.grid_weights(*OHARE)
        self.assertEqual([(i, j) for i, j, _ in weights], [(86, 181), (86, 182), (87, 181), (87, 182)])
        self.assertEqual((wx.LATS[86], wx.LATS[87], wx.LONS[181], wx.LONS[182]), (40.5, 42.0, 271.5, 273.0))
        self.assertAlmostEqual(sum(w for _, _, w in weights), 1.0)
        self.assertGreater(weights[2][2], 0.5)                          # nearest: 42.0 N, 271.5 E

    def test_longitude_wraps(self):
        between = (self.field[:, 50, 239] + self.field[:, 50, 0]) / 2   # halfway from 358.5 back to 0
        self.assertLess(np.abs(core.at_point(self.field, float(wx.LATS[50]), 359.25) - between).max(), 1e-6)
        for lon, same in ((-1.5, 358.5), (360.0, 0.0), (-180.0, 180.0), (-87.9316, 272.0684)):
            a, b = core.at_point(self.field, 12.3, lon), core.at_point(self.field, 12.3, same)
            self.assertLess(np.abs(a - b).max(), 1e-5, lon)

    def test_latitude_stops_at_the_grid_edge(self):
        for lon in (0.0, 123.4):
            south = core.at_point(self.field, -90.0, lon)               # the grid ends at -88.5
            self.assertTrue(np.array_equal(south, core.at_point(self.field, -88.5, lon)))
        self.assertTrue(np.array_equal(core.at_point(self.field, 90.0, 3.0), self.field[:, 119, 2]))
        for lat, lon in ((-90, -180), (90, 360), (0, 0), (89.99, 359.99), (-88.5, 358.5)):
            weights = core.grid_weights(lat, lon)
            self.assertAlmostEqual(sum(w for _, _, w in weights), 1.0)
            for i, j, w in weights:
                self.assertTrue(0 <= i < 120 and 0 <= j < 240 and 0.0 <= w <= 1.0, (lat, lon, i, j, w))


@unittest.skipIf(SKIP, SKIP)
class StoreTest(unittest.TestCase):

    def test_the_bundle(self):
        store = core.StateStore(DATA)
        first, last = datetime.datetime(2019, 1, 20), datetime.datetime(2019, 2, 3, 18)
        self.assertEqual(store.ranges(), [(first, last)])
        self.assertEqual(store.latest(), last)                          # no states/ folder: the bundle's last time
        rows = np.load(os.path.join(DATA, "states_2019-01-20T00_60.npy"), mmap_mode="r")
        state = store.get(INIT)
        self.assertEqual((state.shape, state.dtype), ((20, 120, 240), np.float16))
        self.assertTrue(np.array_equal(state, rows[30]))                # 7.5 days after the first row
        self.assertTrue(np.array_equal(store.get(first), rows[0]))
        self.assertTrue(np.array_equal(store.get(last), rows[59]))
        for when in (first - HOURS_6, last + HOURS_6, INIT + datetime.timedelta(hours=1),
                     datetime.datetime(2026, 11, 14)):
            self.assertIsNone(store.get(when), when)

    def small_store(self, test_rows=3):
        """A data folder with a 3-row bundle, in a temp folder."""
        folder = temp_folder(self)
        rows = np.random.default_rng(2).standard_normal((test_rows, 20, 120, 240)).astype(np.float16)
        np.save(os.path.join(folder, "few.npy"), rows)
        with open(os.path.join(folder, "bundle.json"), "w") as f:
            f.write('{"states": {"file": "few.npy", "first_time": "2026-11-01T00", "step_hours": 6, "rows": %d}}'
                    % test_rows)
        return folder, rows

    def test_pushed_states(self):
        folder, rows = self.small_store()
        store = core.StateStore(folder)
        start = datetime.datetime(2026, 11, 1)
        self.assertEqual(store.latest(), start + 2 * HOURS_6)
        newer, older = datetime.datetime(2026, 11, 14), datetime.datetime(2026, 10, 1, 18)
        self.assertEqual(store.put(newer, rows[1]), newer)              # the newest becomes the latest
        self.assertEqual(store.put(older, rows[2]), newer)              # an older one does not
        self.assertTrue(np.array_equal(store.get(newer), rows[1]))
        self.assertEqual(sorted(os.listdir(os.path.join(folder, "states"))), ["2026100118.npy", "2026111400.npy"])
        self.assertEqual(store.ranges(), [(older, older), (start, start + 2 * HOURS_6), (newer, newer)])
        self.assertEqual(store.put(start + 3 * HOURS_6, rows[0]), newer)
        self.assertEqual(store.ranges()[1], (start, start + 3 * HOURS_6))       # it joins the bundle's stretch

        again = core.StateStore(folder)                                 # a restarted server finds the files
        self.assertEqual(again.latest(), newer)
        self.assertTrue(np.array_equal(again.get(older), rows[2]))
        self.assertEqual(again.ranges(), store.ranges())
        with self.assertRaises(ValueError):
            store.put(newer, rows[1].astype(np.float32))

    def test_latest_prefers_states_over_the_bundle_even_when_older(self):
        folder, rows = self.small_store()
        store = core.StateStore(folder)
        early = datetime.datetime(2026, 10, 1)
        self.assertEqual(store.put(early, rows[0]), early)              # "the newest state from states/ or pushed"

    def test_odd_files_in_states(self):
        folder, rows = self.small_store()
        states = os.path.join(folder, "states")
        os.mkdir(states)
        np.save(os.path.join(states, "2026111400.npy"), rows[0])
        np.save(os.path.join(states, "2026111406.npy"), np.zeros((20, 12, 24), np.float16))     # the wrong shape
        np.save(os.path.join(states, "2026111403.npy"), rows[0])                               # not on a 6-hour mark
        for name in ("notes.txt", "20261114.npy", "2026111400.npy.tmp", "2026139900.npy"):
            open(os.path.join(states, name), "w").close()
        store = core.StateStore(folder)
        self.assertEqual(store.latest(), datetime.datetime(2026, 11, 14, 6))
        self.assertTrue(np.array_equal(store.get(datetime.datetime(2026, 11, 14)), rows[0]))
        self.assertIsNone(store.get(datetime.datetime(2026, 11, 14, 3)))
        with self.assertRaises(ValueError) as caught:                   # a bad file is named, not skipped in silence
            store.get(datetime.datetime(2026, 11, 14, 6))
        self.assertIn("2026111406.npy", str(caught.exception))

    def test_a_folder_that_cannot_be_written(self):
        folder, rows = self.small_store()
        open(os.path.join(folder, "states"), "w").close()               # a file is in the way of the states/ folder
        store = core.StateStore(folder)
        when = datetime.datetime(2026, 11, 14)
        stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            self.assertEqual(store.put(when, rows[0]), when)
            self.assertIn("memory only", sys.stderr.getvalue())
        finally:
            sys.stderr = stderr
        self.assertTrue(np.array_equal(store.get(when), rows[0]))       # kept in memory all the same

    def test_packed_files(self):
        folder = temp_folder(self)
        rows = np.random.default_rng(3).standard_normal((3, 20, 120, 240)).astype(np.float16)
        np.save(os.path.join(folder, "packed_2020.npy"), rows)
        store = core.StateStore(folder)
        first = datetime.datetime(2020, 1, 1)
        self.assertEqual(store.ranges(), [(first, first + 2 * HOURS_6)])
        self.assertEqual(store.latest(), first + 2 * HOURS_6)           # no states/, no bundle: the last packed time
        self.assertTrue(np.array_equal(store.get(first + HOURS_6), rows[1]))
        self.assertIsNone(store.get(first + 3 * HOURS_6))
        empty = core.StateStore(temp_folder(self))
        self.assertEqual((empty.latest(), empty.ranges(), empty.get(first)), (None, [], None))

    def test_a_broken_bundle_says_so(self):
        folder, rows = self.small_store()
        with open(os.path.join(folder, "bundle.json"), "w") as f:
            f.write('{"states": {"file": "few.npy", "first_time": "2026-11-01T00", "step_hours": 6, "rows": 5}}')
        with self.assertRaises(ValueError) as caught:
            core.StateStore(folder)
        self.assertIn("rows", str(caught.exception))
        with open(os.path.join(folder, "bundle.json"), "w") as f:
            f.write('{"clim": {}}')
        with self.assertRaises(ValueError):
            core.StateStore(folder)

    def test_decode_state(self):
        state = np.array(core.StateStore(DATA).get(INIT))
        self.assertTrue(np.array_equal(core.decode_state(packed(state)), state))
        swapped = core.decode_state(packed(state.astype(">f2")))        # big-endian float16 is still float16
        self.assertTrue(np.array_equal(swapped, state))
        self.assertEqual(swapped.dtype, np.float16)
        broken = state.copy()
        broken[3, 4, 5] = np.inf
        for bad, word in ((packed(state.astype(np.float32)), "float16"), (packed(state[:, :, :100]), "shape"),
                          (packed(broken), "NaN"), ("not base64 !!!", "base64"),
                          (base64.b64encode(b"hello").decode(), ".npy"),
                          (packed(np.array(["a"], dtype=object))[:40], ".npy")):
            with self.assertRaises(ValueError) as caught:
                core.decode_state(bad)
            self.assertIn(word, str(caught.exception))


@unittest.skipIf(SKIP, SKIP)
class UnitsTest(unittest.TestCase):

    def setUp(self):
        self.mean, self.std = wx.read_stats(DATA)

    def test_units(self):
        z = np.zeros((2, 3, 20), np.float32)                # every channel at its climate mean ...
        z[1, :, wx.CH_T2M] = 1.0                            # ... but member 1 is one standard deviation warmer
        z[:, 2, wx.CH_MSL] = -2.0                           # and the last step has low pressure
        z[:, 1, wx.CH_U10] = (3.0 - self.mean[wx.CH_U10]) / self.std[wx.CH_U10]     # u = 3 m/s
        z[:, 1, wx.CH_V10] = (-4.0 - self.mean[wx.CH_V10]) / self.std[wx.CH_V10]    # v = -4 m/s
        series = core.point_series(z, 0, self.mean, self.std)
        cold, warm = self.mean[wx.CH_T2M] - 273.15, self.mean[wx.CH_T2M] + self.std[wx.CH_T2M] - 273.15
        self.assertEqual(series["t2m_c"], [round(cold, 2)] * 3)                     # member 0's own series
        self.assertEqual(series["t2m_c_min"], [round(cold, 2)] * 3)
        self.assertEqual(series["t2m_c_max"], [round(warm, 2)] * 3)
        self.assertEqual(series["t2m_c_mean"], [round((cold + warm) / 2, 2)] * 3)
        self.assertEqual(series["wind10_ms_mean"][1], 5.0)                          # a 3-4-5 triangle
        self.assertAlmostEqual(series["msl_hpa_mean"][0], self.mean[wx.CH_MSL] / 100.0, delta=0.006)
        self.assertAlmostEqual(series["msl_hpa_mean"][2], (self.mean[wx.CH_MSL] - 2 * self.std[wx.CH_MSL]) / 100.0,
                               delta=0.006)
        for values in series.values():
            for value in values:
                self.assertIs(type(value), float)                                   # plain floats: JSON can carry them
        self.assertIsNone(core.point_series(z, 3, self.mean, self.std)["t2m_c"])    # member 0 is not in this shard

    def test_numbers(self):
        self.assertEqual(core.numbers(np.array([1.234, -0.001, np.nan, np.inf], np.float32)), [1.23, 0.0, None, None])
        self.assertEqual(str(core.numbers([-0.001])[0]), "0.0")                     # never "-0.0"

    def test_truth(self):
        store = core.StateStore(DATA)
        lat, lon = float(wx.LATS[87]), float(wx.LONS[181])              # the grid point next to O'Hare
        times = [INIT, INIT + HOURS_6, datetime.datetime(2019, 3, 1)]
        truth = core.truth_t2m_c(store, times, lat, lon, self.mean, self.std)
        z = float(store.get(INIT)[wx.CH_T2M, 87, 181])
        self.assertEqual(truth[0], round(z * self.std[wx.CH_T2M] + self.mean[wx.CH_T2M] - 273.15, 2))
        self.assertIsInstance(truth[1], float)
        self.assertIsNone(truth[2])                                     # not held
        cold = core.truth_t2m_c(store, [datetime.datetime(2019, 1, 30, 12)], OHARE[0], OHARE[1], self.mean, self.std)
        self.assertLess(cold[0], -20.0)                                 # the cold wave at O'Hare


@unittest.skipIf(SKIP, SKIP)
class ModelTest(unittest.TestCase):
    """The real model on CPU: a handful of steps, one or two members."""

    @classmethod
    def setUpClass(cls):
        cls.forecaster = core.Forecaster(WEIGHTS, DATA, "cpu", batch=2)
        cls.model = cls.forecaster.model
        cls.state = cls.forecaster.store.get(INIT)
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(cls.state, np.float32)[None])
            cls.step1 = cls.model(x).float()
            cls.step2 = cls.model(cls.step1).float()

    def test_load_model(self):
        self.assertFalse(self.model.training)
        self.assertEqual(self.forecaster.params, 19128832)
        self.assertEqual(str(self.forecaster.device), "cpu")
        with tempfile.NamedTemporaryFile(prefix="forecast_core_test_", suffix=".pt") as f:
            torch.save({"step": 3}, f.name)
            with self.assertRaises(ValueError) as caught:
                core.load_model(f.name, "cpu")
            self.assertIn("'model'", str(caught.exception))

    def test_one_step_looks_like_the_next_analysis(self):
        now = np.asarray(self.state, np.float32)
        truth = np.asarray(self.forecaster.store.get(INIT + HOURS_6), np.float32)
        forecast = self.step1[0].numpy()
        error = float(np.sqrt(((forecast - truth) ** 2).mean()))
        no_change = float(np.sqrt(((now - truth) ** 2).mean()))
        self.assertLess(error, 0.25)                        # in standard deviations, over all channels
        self.assertLess(error, 0.6 * no_change)             # and much better than "nothing changes"

    def test_rollout_feeds_each_output_back(self):
        lat, lon = float(wx.LATS[87]), float(wx.LONS[181])
        first = np.asarray(self.state, np.float32)[None]
        z = core.rollout_point(self.model, "cpu", first, 2, lat, lon)
        self.assertEqual((z.shape, z.dtype), ((1, 2, 20), np.float32))
        self.assertLess(np.abs(z[0, 0] - self.step1[0, :, 87, 181].numpy()).max(), 1e-4)
        self.assertLess(np.abs(z[0, 1] - self.step2[0, :, 87, 181].numpy()).max(), 1e-4)
        between = core.rollout_point(self.model, "cpu", first, 1, *OHARE)
        self.assertLess(np.abs(between[0, 0] - core.at_point(self.step1[0], *OHARE).numpy()).max(), 1e-4)

    def test_members_and_shards(self):
        z, timings = core.forecast_point(self.model, "cpu", self.state, INIT, 2, 3, 0, OHARE[0], OHARE[1], batch=2)
        self.assertEqual((z.shape, z.dtype), ((3, 2, 20), np.float32))
        self.assertEqual([n for n, _ in timings], [2, 1])                           # 3 members, 2 at a time
        self.assertTrue(all(seconds > 0 for _, seconds in timings))
        plain = core.at_point(self.step2[0], *OHARE).numpy()
        self.assertLess(np.abs(z[0, 1] - plain).max(), 1e-3)                        # member 0: the unperturbed forecast
        self.assertGreater(np.abs(z[1] - z[0]).max(), 1e-3)                         # members 1 and 2 differ from it
        self.assertGreater(np.abs(z[2] - z[1]).max(), 1e-3)
        self.assertLess(np.abs(z[1] - z[0]).max(), 1.0)                             # but they are nudged, not thrown
        shard, _ = core.forecast_point(self.model, "cpu", self.state, INIT, 2, 1, 2, OHARE[0], OHARE[1], batch=2)
        self.assertLess(np.abs(shard[0] - z[2]).max(), 1e-3)                        # member 2 alone = member 2 of three

    def test_forecaster_run(self):
        self.assertGreater(self.forecaster.warm_up(), 0.0)
        series, timings = self.forecaster.run(self.state, INIT, 2, 1, 0, OHARE[0], OHARE[1], with_truth=True)
        self.assertEqual(sorted(series), ["msl_hpa_mean", "t2m_c", "t2m_c_max", "t2m_c_mean", "t2m_c_min",
                                          "truth_t2m_c", "wind10_ms_mean"])
        self.assertEqual(len(timings), 1)
        self.assertEqual(series["t2m_c"], series["t2m_c_mean"])
        for forecast, truth in zip(series["t2m_c"], series["truth_t2m_c"]):
            self.assertLess(abs(forecast - truth), 5.0)                             # 6 and 12 hours ahead
        z = core.at_point(self.step1[0], *OHARE).numpy().astype(np.float64)
        celsius = z[wx.CH_T2M] * self.forecaster.std[wx.CH_T2M] + self.forecaster.mean[wx.CH_T2M] - 273.15
        self.assertAlmostEqual(series["t2m_c"][0], celsius, delta=0.02)
        late = datetime.datetime(2019, 2, 3, 18)                                    # the bundle's last analysis
        series, _ = self.forecaster.run(self.forecaster.store.get(late), late, 1, 1, 0, 0.0, 0.0, with_truth=True)
        self.assertEqual(series["truth_t2m_c"], [None])                             # nothing to check it against
        self.assertNotIn("truth_t2m_c", self.forecaster.run(self.state, INIT, 1, 1, 0, 0.0, 0.0)[0])


if __name__ == "__main__":
    unittest.main()
