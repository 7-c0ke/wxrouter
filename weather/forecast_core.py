"""The real weather forecast: the stored states, the ensemble noise, the model rollout and the units.

This is the only part of the forecast server that needs NumPy and PyTorch. It is a library, not a program:
serve/forecast_server.py imports it when it runs without --mock.

    python serve/forecast_server.py --node A --port 8101 --weights /out/stage2/final.pt --data /data
    python -m unittest tests/test_forecast_core.py -v

A state is a float16 array of z-scores, shape (20, 120, 240), channels in wx_common.CHANNEL_NAMES order.
Times are plain datetime objects in UTC. Every state sits on a 6-hour mark (00, 06, 12, 18).

Written at the hackathon (Nov 13-14, 2026).
"""
import base64
import binascii
import datetime
import io
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import wx_common as wx  # noqa: E402

STATE_SHAPE = (wx.N_CHANNELS, wx.N_LAT, wx.N_LON)     # (20, 120, 240)
STEP = datetime.timedelta(hours=wx.STEP_HOURS)        # one model step
GRID_DEG = 1.5
NOISE_STD = 0.05                                      # size of a member's perturbation, in z-score units
CPU_THREADS = 4                                       # PyTorch's CPU threads; plenty when the GPU does the work


def time_number(when):
    """datetime -> 2019012712. This is the noise seed and the file name in states/."""
    return int(when.strftime("%Y%m%d%H"))


# ---------------------------------------------------------------- the model

def load_model(weights_path, device):
    """Build the AFNO and load the weights weather/train.py saved. Returns it in eval mode on `device`."""
    model = wx.build_afno()
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "model" not in saved:
        raise ValueError("%s has no 'model' entry; it must be a file written by weather/train.py" % weights_path)
    model.load_state_dict(saved["model"])
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------- states

def check_state(array, what="the state"):
    """Return `array` as a float16 (20, 120, 240) array, or raise ValueError saying what is wrong with it."""
    if not isinstance(array, np.ndarray):
        raise ValueError("%s is not one array" % what)
    if array.shape != STATE_SHAPE:
        raise ValueError("%s has shape %s; it must be %s" % (what, tuple(array.shape), STATE_SHAPE))
    if array.dtype.kind != "f" or array.dtype.itemsize != 2:
        raise ValueError("%s has type %s; it must be float16 (z-scores)" % (what, array.dtype))
    array = np.ascontiguousarray(array, dtype=np.float16)
    if not np.isfinite(array).all():
        raise ValueError("%s holds NaN or infinite values" % what)
    return array


def decode_state(npy_b64):
    """A state sent over HTTP (base64 of .npy bytes) -> float16 (20, 120, 240). Raises ValueError if it is not one."""
    try:
        raw = base64.b64decode(npy_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("it is not valid base64")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        raise ValueError("it does not hold .npy bytes (make them with numpy.save)")
    return check_state(array)


class StateStore(object):
    """The analyses this server holds, by UTC time. A forecast starts from one; with_truth checks against them.

    It looks in three places under the data folder:
      1. states pushed with POST /v1/state (kept in memory) and files states/YYYYMMDDHH.npy;
      2. the bundle: bundle.json and the state file it names;
      3. the full packed files, if they are there (memory-mapped, so they cost no memory).
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.states_dir = os.path.join(data_dir, "states")
        self.lock = threading.Lock()
        self.pushed = {}            # time -> state, from put()
        self.blocks = []            # (first time, rows): rows is (n, 20, 120, 240), one row every 6 hours
        self.bundle_last = None     # last time in the bundle
        self.packed_last = None     # last time in the packed files
        self.warned = False

        bundle_path = os.path.join(data_dir, "bundle.json")
        if os.path.exists(bundle_path):
            try:
                with open(bundle_path) as f:
                    info = json.load(f)["states"]
                name, n_rows, step = info["file"], int(info["rows"]), info["step_hours"]
                first = datetime.datetime.strptime(info["first_time"], "%Y-%m-%dT%H")
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError("%s is not a bundle description (%s: %s)" % (bundle_path, type(e).__name__, e))
            if step != wx.STEP_HOURS:
                raise ValueError("%s has step_hours %r; it must be %d" % (bundle_path, step, wx.STEP_HOURS))
            rows = np.load(os.path.join(data_dir, name), mmap_mode="r")
            self._add_block(first, rows, name)
            if len(rows) != n_rows:
                raise ValueError("%s has %d rows but bundle.json says %d" % (name, len(rows), n_rows))
            self.bundle_last = first + (len(rows) - 1) * STEP

        for which, first in (("main", wx.T0_MAIN), ("2020", wx.T0_2020)):
            try:
                rows = wx.open_packed(which, data_dir)
            except FileNotFoundError:
                continue
            first = first.astype(datetime.datetime)
            self._add_block(first, rows, "the packed file (%s)" % which)
            self.packed_last = first + (len(rows) - 1) * STEP

    def _add_block(self, first, rows, what):
        if rows.ndim != 4 or rows.shape[1:] != STATE_SHAPE or rows.dtype != np.float16 or len(rows) == 0:
            raise ValueError("%s has shape %s and type %s; it must be (rows, 20, 120, 240) float16"
                             % (what, rows.shape, rows.dtype))
        self.blocks.append((first, rows))

    def _file_times(self):
        """The times of the files in states/."""
        try:
            names = os.listdir(self.states_dir)
        except OSError:
            return []
        times = []
        for name in names:
            if re.fullmatch(r"\d{10}\.npy", name):
                try:
                    when = datetime.datetime.strptime(name[:10], "%Y%m%d%H")
                except ValueError:
                    continue
                if when.hour % wx.STEP_HOURS == 0:
                    times.append(when)
        return times

    def get(self, when):
        """The state at a time, float16 (20, 120, 240), or None if it is not held."""
        with self.lock:
            state = self.pushed.get(when)
        if state is not None:
            return state
        if when.hour % wx.STEP_HOURS == 0 and when.minute == 0:
            path = os.path.join(self.states_dir, "%d.npy" % time_number(when))
            if os.path.exists(path):
                try:
                    array = np.load(path, allow_pickle=False)
                except Exception as e:
                    raise ValueError("cannot read %s (%s: %s)" % (path, type(e).__name__, e))
                return check_state(array, path)
        for first, rows in self.blocks:
            n, rest = divmod(when - first, STEP)
            if not rest and 0 <= n < len(rows):
                return rows[n]
        return None

    def latest(self):
        """The newest state from states/ or pushed. With none of those: the bundle's last time (then the packed
        files' last time). None if the store is empty."""
        with self.lock:
            times = list(self.pushed)
        times += self._file_times()
        if times:
            return max(times)
        return self.bundle_last or self.packed_last

    def ranges(self):
        """The stretches of 6-hourly times held: [(first, last), ...], oldest first."""
        spans = [(first, first + (len(rows) - 1) * STEP) for first, rows in self.blocks]
        with self.lock:
            singles = set(self.pushed)
        singles.update(self._file_times())
        spans += [(when, when) for when in singles]
        merged = []
        for first, last in sorted(spans):
            if merged and first <= merged[-1][1] + STEP:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        return [(first, last) for first, last in merged]

    def put(self, when, state):
        """Keep a pushed state in memory, and write it to states/ if that folder can be written.
        Returns the latest time afterwards."""
        state = check_state(state)
        path = os.path.join(self.states_dir, "%d.npy" % time_number(when))
        with self.lock:
            self.pushed[when] = state
            try:
                os.makedirs(self.states_dir, exist_ok=True)
                with open(path + ".tmp", "wb") as f:
                    np.save(f, state)
                os.replace(path + ".tmp", path)       # so a reader never sees half a file
            except OSError as e:
                if not self.warned:
                    self.warned = True
                    sys.stderr.write("cannot write %s (%s); pushed states stay in memory only\n" % (path, e))
        return self.latest()


# ---------------------------------------------------------------- ensemble noise

def smooth(field):
    """Three passes of the (1, 4, 6, 4, 1)/16 filter along longitude (it wraps) and along latitude (the edge
    rows are repeated). The last two axes of `field` are latitude and longitude."""
    for _ in range(3):
        field = (np.roll(field, 2, -1) + np.roll(field, -2, -1)
                 + 4.0 * (np.roll(field, 1, -1) + np.roll(field, -1, -1)) + 6.0 * field) / 16.0
        rows = np.concatenate([field[..., :1, :]] * 2 + [field] + [field[..., -1:, :]] * 2, axis=-2)
        field = (rows[..., :-4, :] + rows[..., 4:, :]
                 + 4.0 * (rows[..., 1:-3, :] + rows[..., 3:-1, :]) + 6.0 * rows[..., 2:-2, :]) / 16.0
    return field


def member_noise(when, member):
    """What global member `member` adds to the starting state: float32 (20, 120, 240), in z-score units.

    Member 0 adds nothing. Any other member adds smooth noise with standard deviation 0.05 in every channel.
    The seed is (init time, member) and nothing else, so every node makes the same member."""
    if member == 0:
        return np.zeros(STATE_SHAPE, np.float32)
    rng = np.random.default_rng([time_number(when), int(member)])
    field = smooth(rng.standard_normal(STATE_SHAPE, dtype=np.float32))
    spread = field.std(axis=(1, 2), dtype=np.float64)
    return (field * (NOISE_STD / spread)[:, None, None]).astype(np.float32)


def perturbed_states(state, when, members):
    """The starting states of the given global members: float32 (len(members), 20, 120, 240)."""
    base = np.asarray(state, dtype=np.float32)
    return np.stack([base + member_noise(when, m) for m in members])


# ---------------------------------------------------------------- one point on the grid

def grid_weights(lat, lon):
    """The four grid points around a place and their bilinear weights: [(row, column, weight), ...].

    The grid has 120 latitudes from -88.5 to 90 and 240 longitudes from 0 to 358.5. Longitude wraps around;
    a latitude south of -88.5 is moved onto the grid's last row."""
    lat = min(max(float(lat), float(wx.LATS[0])), float(wx.LATS[-1]))
    y = (lat - float(wx.LATS[0])) / GRID_DEG          # 0 .. 119, in rows
    i = min(int(math.floor(y)), wx.N_LAT - 2)
    fy = y - i                                        # how far towards row i + 1
    x = (float(lon) % 360.0) / GRID_DEG               # 0 .. 240, in columns
    fx = x - math.floor(x)                            # how far towards the next column
    j = int(math.floor(x)) % wx.N_LON
    j2 = (j + 1) % wx.N_LON
    return [(i, j, (1.0 - fy) * (1.0 - fx)), (i, j2, (1.0 - fy) * fx),
            (i + 1, j, fy * (1.0 - fx)), (i + 1, j2, fy * fx)]


def at_point(field, lat, lon):
    """Interpolate `field` (a float32 NumPy array or a PyTorch tensor ending in (120, 240)) to one place."""
    total = None
    for i, j, weight in grid_weights(lat, lon):
        part = field[..., i, j] * weight
        total = part if total is None else total + part
    return total


# ---------------------------------------------------------------- the forecast

def rollout_point(model, device, first, n_steps, lat, lon):
    """Advance a batch of states by n_steps 6-hour steps. Returns the z-scores at one place:
    float32 (members, n_steps, 20).

    Each output is cast to float32 and fed back in, as in training. Only the place's values are kept at each
    step, so memory does not grow with the lead time. On CUDA the model runs under bf16 autocast."""
    device = torch.device(device)
    x = torch.from_numpy(np.ascontiguousarray(first, dtype=np.float32)).to(device)
    kept = []
    with torch.inference_mode():
        for _ in range(n_steps):
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
            else:
                out = model(x)
            x = out.float()
            kept.append(at_point(x, lat, lon))
        return torch.stack(kept, dim=1).cpu().numpy()      # .cpu() also waits for the GPU, so timing is honest


def forecast_point(model, device, state, when, n_steps, members, member_offset, lat, lon, batch=20):
    """Run global members member_offset .. member_offset + members - 1 from one state, `batch` at a time.

    Returns (z, timings). z is float32 (members, n_steps, 20): z-scores at the place. timings has one
    (members in the batch, seconds in the model) pair per batch, for the server's speed numbers."""
    parts, timings = [], []
    for a in range(0, members, batch):
        ids = range(member_offset + a, member_offset + min(a + batch, members))
        first = perturbed_states(state, when, ids)
        t0 = time.perf_counter()
        parts.append(rollout_point(model, device, first, n_steps, lat, lon))
        timings.append((len(ids), time.perf_counter() - t0))
    return np.concatenate(parts), timings


def numbers(values):
    """A NumPy vector -> a list of floats rounded to 0.01, ready for JSON. NaN and infinity become None."""
    return [round(float(v), 2) + 0.0 if math.isfinite(v) else None for v in values]


def point_series(z, member_offset, mean, std):
    """z-scores at a place, (members, steps, 20) -> the series of the response, in everyday units.

    2-m temperature: K to C. 10-m wind speed: from u and v, m/s. Mean-sea-level pressure: Pa to hPa.
    t2m_c is member 0's own series; it is None when member 0 is not among these members."""
    z = np.asarray(z, dtype=np.float64)

    def physical(channel):
        return z[:, :, channel] * std[channel] + mean[channel]

    t2m = physical(wx.CH_T2M) - 273.15
    wind = np.hypot(physical(wx.CH_U10), physical(wx.CH_V10))
    msl = physical(wx.CH_MSL) / 100.0
    return {
        "t2m_c": numbers(t2m[0]) if member_offset == 0 else None,
        "t2m_c_mean": numbers(t2m.mean(axis=0)),
        "t2m_c_min": numbers(t2m.min(axis=0)),
        "t2m_c_max": numbers(t2m.max(axis=0)),
        "wind10_ms_mean": numbers(wind.mean(axis=0)),
        "msl_hpa_mean": numbers(msl.mean(axis=0)),
    }


def truth_t2m_c(store, valid_times, lat, lon, mean, std):
    """The analysed 2-m temperature (C) at a place for each valid time; None where the store has no state."""
    out = []
    for when in valid_times:
        state = store.get(when)
        if state is None:
            out.append(None)
            continue
        z = at_point(np.asarray(state[wx.CH_T2M], dtype=np.float32), lat, lon)
        out.append(numbers([float(z) * std[wx.CH_T2M] + mean[wx.CH_T2M] - 273.15])[0])
    return out


class Forecaster(object):
    """The loaded model with its data: what the server's worker thread needs to answer a request."""

    def __init__(self, weights_path, data_dir, device="auto", batch=20):
        torch.set_num_threads(CPU_THREADS)
        self.device = wx.pick_device(device)
        self.batch = batch
        self.mean, self.std = wx.read_stats(data_dir)
        self.store = StateStore(data_dir)
        self.model = load_model(weights_path, self.device)
        self.params = sum(p.numel() for p in self.model.parameters())

    def warm_up(self):
        """Run one member one step, twice, and return the seconds the second run took. The first run pays the
        one-off start-up costs, so it is not the one that is timed."""
        blank = np.zeros((1,) + STATE_SHAPE, np.float32)
        rollout_point(self.model, self.device, blank, 1, 0.0, 0.0)
        t0 = time.perf_counter()
        rollout_point(self.model, self.device, blank, 1, 0.0, 0.0)
        return time.perf_counter() - t0

    def run(self, state, when, n_steps, members, member_offset, lat, lon, with_truth=False):
        """Answer one request. Returns (series, timings): the response's point series, and forecast_point's
        timings."""
        z, timings = forecast_point(self.model, self.device, state, when, n_steps, members, member_offset,
                                    lat, lon, self.batch)
        series = point_series(z, member_offset, self.mean, self.std)
        if with_truth:
            valid_times = [when + (k + 1) * STEP for k in range(n_steps)]
            series["truth_t2m_c"] = truth_t2m_c(self.store, valid_times, lat, lon, self.mean, self.std)
        return series, timings
