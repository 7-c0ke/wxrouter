"""Score a trained weather model: its forecast error from 6 to 120 hours ahead, beside two simple baselines.

The overnight driver (weather/run_overnight.sh) runs this when training ends, in the container, on the GPU:

    python weather/score.py --weights /out/stage2/final.pt --data /data --out /out/scores --years 2020 2019 --storm

Baselines only, with no model and no PyTorch (checks the arithmetic on a machine without a GPU):

    python weather/score.py --no-model --data /data --out /tmp/scores --years 2020

For each year a forecast starts at 00Z and 12Z of every second day (362 starts in 2020, 360 in 2019). The model
is fed its own output for 20 six-hour steps, and Z500, T850 and T2M are compared with what really happened:

    MSE(start, lead) = mean over the grid of  w * (forecast - truth)^2     (w = cos(latitude), scaled to mean 1)
    RMSE(lead)       = square root of the mean over starts of that MSE

Two baselines get the same arithmetic: persistence (the forecast is the truth at the start time) and climatology
(the 1979-2017 average for that day of the year and hour). --storm adds one case: the January 2019 cold wave.

It prints a table for 24, 72 and 120 hours beside the values measured before the event, and writes everything
to <out>/scores.json. Nothing is written outside --out. Exit code 0 means scores.json was written; any other
code comes with a message that says why.

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                     # wx_common.py sits beside this file

import wx_common as wx  # noqa: E402

N_STEPS = 20                                                     # six-hour steps in one forecast
LEADS_H = [wx.STEP_HOURS * k for k in range(1, N_STEPS + 1)]     # 6, 12, ..., 120
SHOWN_H = [24, 72, 120]                                          # the leads in the printed table
VARS = list(wx.SCORED)                # z500, t850, t2m: also the order in truth_2020.npy and the climatology file
UNITS = {"z500": "m2/s2", "t850": "K", "t2m": "K"}
KINDS = ["model", "persistence", "climatology"]

BAND_PCT = 2.0            # "same skill": a 72-hour score within 2% of the reference model's
BASELINE_PCT = 0.5        # a baseline more than 0.5% from its known value means this scorer is wrong
PROGRESS_EVERY_S = 20     # while a year is being scored, say how far it is this often

# Known answers, measured before the event on this data: RMSE by lead hour.
# "model" is the reference model (the same recipe, trained on Nov 10).
KNOWN = {
    2020: {
        "model": {"z500": {24: 102.18, 72: 303.86, 120: 543.81},
                  "t850": {24: 0.9182, 72: 1.6809, 120: 2.6138},
                  "t2m": {24: 0.9396, 72: 1.4495, 120: 2.0730}},
        "persistence": {"z500": {24: 605.3, 72: 956.4, 120: 1051.0},
                        "t850": {24: 3.051, 72: 4.326, 120: 4.592},
                        "t2m": {24: 2.035, 72: 2.935, 120: 3.215}},
        "climatology": {"z500": {24: 824.8, 72: 825.3, 120: 825.9},
                        "t850": {24: 3.468, 72: 3.466, 120: 3.464},
                        "t2m": {24: 2.657, 72: 2.657, 120: 2.655}},
    },
    2019: {
        "model": {"z500": {72: 296.86}, "t850": {72: 1.6607}, "t2m": {72: 1.4431}},
    },
}

# The January 2019 cold wave: T850 around Chicago in the 72-hour forecast.
STORM_START = "2019-01-27T06"
STORM_LEAD_H = 72
STORM_LAT = (86, 88)              # latitude rows 86..88: 40.5 to 43.5 N
STORM_LON = (179, 182)            # longitude columns 179..182: 268.5 to 273 E
STORM_RANGE = (0.88, 0.97)        # the share of the cold this recipe caught before the event

CLIM_NAME = "clim_1979_2017.npy"
TRUTH_2020_NAME = "truth_2020.npy"
GRID = (wx.N_LAT, wx.N_LON)


class ScoreError(Exception):
    """A problem to report in one line (a missing file, a start outside the data). Not a bug."""


# ---------------------------------------------------------------- the data folder

class Data:
    """The data folder's files, memory-mapped. Rows are read from disk only when they are asked for."""

    def __init__(self, folder=None):
        self.folder = wx.data_dir(folder)
        if not os.path.isdir(self.folder):
            raise ScoreError("the data folder %s does not exist" % self.folder)
        try:
            self.mean, self.std = wx.read_stats(self.folder)
        except (OSError, ValueError, KeyError) as e:
            raise ScoreError("cannot use stats.json in %s: %s" % (self.folder, e))
        self.clim = self._open(CLIM_NAME, (4, len(VARS)) + GRID)
        self.truth_2020 = None                   # optional; without it 2020 is checked against packed_2020.npy
        if os.path.exists(os.path.join(self.folder, TRUTH_2020_NAME)):
            self.truth_2020 = self._open(TRUTH_2020_NAME, (len(VARS),) + GRID)
        self._packed = {}

    def _open(self, name, row_shape):
        path = os.path.join(self.folder, name)
        try:
            array = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as e:
            raise ScoreError("cannot open the data file %s: %s" % (path, e))
        return self._checked(array, row_shape)

    def _checked(self, array, row_shape):
        if array.shape[1:] != row_shape:
            raise ScoreError("%s has shape %s; every row should have shape %s"
                             % (array.filename, array.shape, row_shape))
        return array

    def packed(self, which):
        """The z-score file 'main' (1979-2019) or '2020': float16 (time, 20, 120, 240)."""
        if which not in self._packed:
            try:
                array = wx.open_packed(which, self.folder)
            except (OSError, ValueError) as e:
                raise ScoreError("cannot open the packed file for '%s' in %s: %s" % (which, self.folder, e))
            self._packed[which] = self._checked(array, (wx.N_CHANNELS,) + GRID)
        return self._packed[which]

    def has_truth_file(self, which):
        return which == "2020" and self.truth_2020 is not None

    def packed_name(self, which):
        return os.path.basename(str(self.packed(which).filename))

    def truth_name(self, which):
        """Where the truth for this file's times comes from, in words for the output."""
        if self.has_truth_file(which):
            return TRUTH_2020_NAME
        return "%s (z-scores turned into physical units)" % self.packed_name(which)

    def truth_rows(self, which):
        """How many rows can be scored: the packed file's length, or the truth file's if that is shorter."""
        n = len(self.packed(which))
        return min(n, len(self.truth_2020)) if self.has_truth_file(which) else n

    def to_physical(self, z):
        """z-scores of Z500, T850, T2M, shape (..., 3, 120, 240) -> physical units, in float64."""
        z = np.asarray(z, dtype=np.float64)
        return z * self.std[wx.SCORED_IDX][:, None, None] + self.mean[wx.SCORED_IDX][:, None, None]

    def truth(self, which, rows):
        """What really happened at these rows: Z500, T850, T2M in physical units, float64 (len(rows), 3, 120, 240).
        For 2020 this is truth_2020.npy when the folder has it; otherwise the packed z-scores, converted."""
        if self.has_truth_file(which):
            return np.stack([self.truth_2020[r] for r in rows]).astype(np.float64)
        packed = self.packed(which)
        return self.to_physical(np.stack([packed[r, wx.SCORED_IDX] for r in rows]))

    def check_slot(self, when):
        """(day slot, hour index) of a time in the climatology file; a clear message if the file is too short."""
        slot, hour = wx.calendar_slot(when)
        if slot >= len(self.clim):
            raise ScoreError("%s has %d day slots, but %s needs slot %d"
                             % (self.clim.filename, len(self.clim), when, slot))
        return slot, hour

    def climatology(self, times):
        """The 1979-2017 average of Z500, T850, T2M for these valid times: float64 (len(times), 3, 120, 240)."""
        return np.stack([self.clim[self.check_slot(t)] for t in times]).astype(np.float64)


# ---------------------------------------------------------------- the arithmetic

def weighted_mse(forecast, truth, weights):
    """Mean over the grid of weights * (forecast - truth)^2, in float64.
    The last two axes are the grid (120, 240); the result keeps the axes before them."""
    diff = np.asarray(forecast, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return (weights * diff * diff).mean(axis=(-2, -1))


def rmse_over_starts(mse):
    """mse has one row per start. The square root of the mean over starts (not the mean of square roots)."""
    return np.sqrt(np.mean(mse, axis=0))


# ---------------------------------------------------------------- the model

class Forecaster:
    """The model on its device. It is fed its own output, six hours at a time, the way train.py validated it:
    eval mode, inference mode, bf16 autocast on CUDA (plain float32 elsewhere), every output cast to float32
    before it goes back in. No torch.compile."""

    def __init__(self, model, device):
        import torch
        self.torch = torch
        self.device = device
        self.bf16 = device.type == "cuda"
        self.model = model.to(device).eval()
        self.x = None

    def start(self, states):
        """states: z-scores (batch, 20, 120, 240), as stored in the packed files (float16)."""
        self.x = self.torch.from_numpy(np.ascontiguousarray(states)).to(self.device).float()

    def step(self):
        """Move every forecast on by six hours.
        Returns the z-scores of Z500, T850 and T2M as float32 numpy, shape (batch, 3, 120, 240)."""
        torch = self.torch
        with torch.inference_mode():
            if self.bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(self.x)
            else:
                out = self.model(self.x)
            self.x = out.float()
            return self.x[:, wx.SCORED_IDX].cpu().numpy()


def load_model(weights):
    """The AFNO with the weights train.py saved. Returns (model, what the file says about itself)."""
    import torch
    if not os.path.isfile(weights):
        raise ScoreError("the weights file %s does not exist" % weights)
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "model" not in saved:
        raise ScoreError("%s has no 'model' entry; it should be a final.pt written by train.py" % weights)
    model = wx.build_afno()
    try:
        model.load_state_dict(saved["model"])
    except RuntimeError as e:
        raise ScoreError("the weights in %s do not fit the model: %s" % (weights, e))
    return model, dict((key, saved.get(key)) for key in ("stage", "step", "complete"))


def make_forecaster(weights, want_device, model=None):
    """Put the model on its device: the AFNO with the weights file's weights, or the stand-in a test passes.
    Returns (forecaster, what the weights file says about itself, or None for a stand-in)."""
    import torch
    device = wx.pick_device(want_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ScoreError("--device cuda was asked for, but PyTorch finds no CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = True             # the same settings train.py validated with
        torch.backends.cudnn.allow_tf32 = True
    checkpoint = None
    if model is None:
        model, checkpoint = load_model(weights)
    return Forecaster(model, device), checkpoint


def describe_checkpoint(checkpoint):
    """What train.py wrote into the weights file about its run, in a few words: 'stage 2, step 1564, complete'."""
    said = ["%s %s" % (key, checkpoint[key]) for key in ("stage", "step") if checkpoint.get(key) is not None]
    if checkpoint.get("complete") is True:
        said.append("complete")
    elif checkpoint.get("complete") is False:
        said.append("NOT complete")
    return ", ".join(said)


# ---------------------------------------------------------------- one year

def plan_year(year, data, limit):
    """Which starts to score for a year, and where they are in the files.
    Stops with a clear message if a start, its 120 hours or their climatology are not in the data.
    run() does this for every year before any scoring, so that such a problem shows in the first second."""
    all_starts = wx.test_starts(year)
    starts = all_starts[:limit] if limit else all_starts
    which = wx.locate(starts[0])[0]              # the file that holds this year
    n_rows = data.truth_rows(which)
    rows = [wx.time_index(t, which) for t in starts]
    for t, row in zip(starts, rows):
        if row < 0 or row + N_STEPS >= n_rows:
            files = data.packed_name(which) + (" and " + TRUTH_2020_NAME if data.has_truth_file(which) else "")
            raise ScoreError("%d: the start %s and its %d hours need rows %d to %d, but only %d rows are there "
                             "(%s). Use --limit to score fewer starts."
                             % (year, t, LEADS_H[-1], row, row + N_STEPS, n_rows, files))
    last_valid = starts[-1] + np.timedelta64(LEADS_H[-1], "h")
    data.check_slot(last_valid)                  # the latest valid time has the highest day slot
    return {"starts": starts, "n_starts_full": len(all_starts), "which": which, "rows": rows}


def score_year(year, plan, data, forecaster, batch, weights):
    """RMSE at every lead for one year's starts: the model (if there is one), persistence and climatology.
    Returns the year's entry for scores.json."""
    t0 = time.time()
    starts, which, rows = plan["starts"], plan["which"], plan["rows"]
    packed = data.packed(which)
    kinds = [kind for kind in KINDS if kind != "model" or forecaster is not None]
    mse = dict((kind, np.zeros((len(starts), N_STEPS, len(VARS)))) for kind in kinds)     # [start, lead, variable]
    last_note = time.time()
    for a in range(0, len(starts), batch):
        b = min(a + batch, len(starts))
        at_start = data.truth(which, rows[a:b])                  # persistence: the truth at the start time
        if forecaster is not None:
            forecaster.start(np.stack([packed[r] for r in rows[a:b]]))
        for k in range(1, N_STEPS + 1):
            truth = data.truth(which, [r + k for r in rows[a:b]])
            valid = [t + np.timedelta64(k * wx.STEP_HOURS, "h") for t in starts[a:b]]
            mse["persistence"][a:b, k - 1] = weighted_mse(at_start, truth, weights)
            mse["climatology"][a:b, k - 1] = weighted_mse(data.climatology(valid), truth, weights)
            if forecaster is not None:
                forecast = data.to_physical(forecaster.step())
                mse["model"][a:b, k - 1] = weighted_mse(forecast, truth, weights)
        if b < len(starts) and time.time() - last_note >= PROGRESS_EVERY_S:
            print("  %d: %d of %d starts done, %.0f s" % (year, b, len(starts), time.time() - t0), flush=True)
            last_note = time.time()

    rmse = {}
    for kind in KINDS:
        rmse[kind] = None
        if kind in mse:
            table = rmse_over_starts(mse[kind])                  # (20 leads, 3 variables)
            rmse[kind] = dict((var, [float(v) for v in table[:, i]]) for i, var in enumerate(VARS))
    result = {
        "n_starts": len(starts), "n_starts_full": plan["n_starts_full"],
        "first_start": str(starts[0]), "last_start": str(starts[-1]),
        "leads_h": LEADS_H,
        "truth": data.truth_name(which),
        "rmse": rmse,
        "seconds": round(time.time() - t0, 1),
    }
    add_verdicts(year, result)
    return result


def add_verdicts(year, result):
    """Put tonight's RMSE at 24, 72 and 120 h beside the known values, and judge what can be judged:
    the model's 72-hour scores against the +-2% band, the baselines against +-0.5%.
    A year cut short by --limit is not comparable, so it gets no verdicts ("ok" stays None)."""
    known = KNOWN.get(year, {})
    comparable = result["n_starts"] == result["n_starts_full"]
    rows = []
    for kind in KINDS:
        if result["rmse"][kind] is None:
            continue
        for var in VARS:
            for lead in SHOWN_H:
                tonight = result["rmse"][kind][var][LEADS_H.index(lead)]
                value = known.get(kind, {}).get(var, {}).get(lead)
                row = {"kind": kind, "variable": var, "lead_h": lead, "tonight": tonight, "known": value,
                       "diff_pct": None, "tolerance_pct": None, "ok": None}
                if value is not None:
                    row["diff_pct"] = 100.0 * (tonight - value) / value
                    if kind != "model":
                        row["tolerance_pct"] = BASELINE_PCT
                    elif lead == 72:
                        row["tolerance_pct"] = BAND_PCT
                    if comparable and row["tolerance_pct"] is not None:
                        row["ok"] = bool(abs(row["diff_pct"]) <= row["tolerance_pct"])   # a NaN score is not ok
                rows.append(row)
    result["comparable"] = comparable
    result["compare"] = rows
    result["same_skill_72h"] = dict((var, None) for var in VARS)
    for row in rows:
        if row["kind"] == "model" and row["lead_h"] == 72:
            result["same_skill_72h"][row["variable"]] = row["ok"]
    baseline_oks = [row["ok"] for row in rows if row["kind"] != "model" and row["ok"] is not None]
    result["baselines_ok"] = all(baseline_oks) if baseline_oks else None


# ---------------------------------------------------------------- the storm

def plan_storm(data):
    """Where the storm forecast is in the files: (file, start row, steps, start time, valid time).
    Stops with a clear message if it is not in the data."""
    start = np.datetime64(STORM_START, "h")
    valid = start + np.timedelta64(STORM_LEAD_H, "h")
    n_steps = STORM_LEAD_H // wx.STEP_HOURS
    which, row = wx.locate(start)
    n_rows = data.truth_rows(which)
    if row < 0 or row + n_steps >= n_rows:
        raise ScoreError("the storm forecast needs rows %d to %d, but %s has only %d rows"
                         % (row, row + n_steps, data.packed_name(which), n_rows))
    data.check_slot(valid)
    return which, row, n_steps, start, valid


def score_storm(data, forecaster):
    """The January 2019 cold wave: how much of the cold around Chicago did the 72-hour forecast catch?
    share caught = (forecast - climatology) / (truth - climatology), with T850 averaged over 12 grid points."""
    which, row, n_steps, start, valid = plan_storm(data)
    packed = data.packed(which)
    t850 = VARS.index("t850")                    # 1: T850's place among the three scored maps

    def box_mean(field):
        """The plain average of a (120, 240) map over the 12 grid points of the box."""
        box = np.asarray(field, dtype=np.float64)[STORM_LAT[0]:STORM_LAT[1] + 1, STORM_LON[0]:STORM_LON[1] + 1]
        return float(box.mean())

    truth_k = box_mean(data.truth(which, [row + n_steps])[0, t850])
    clim_k = box_mean(data.climatology([valid])[0, t850])
    forecast_k, share, inside = None, None, None
    if forecaster is not None:
        forecaster.start(np.stack([packed[row]]))
        for _ in range(n_steps):
            z = forecaster.step()
        forecast_k = box_mean(data.to_physical(z)[0, t850])
        if truth_k != clim_k:
            share = (forecast_k - clim_k) / (truth_k - clim_k)
            inside = bool(STORM_RANGE[0] <= share <= STORM_RANGE[1])
    return {
        "start": str(start), "valid": str(valid), "lead_h": STORM_LEAD_H,
        "variable": "t850", "units": "K",
        "lat_index": list(STORM_LAT), "lon_index": list(STORM_LON),
        "lat_deg": [float(wx.LATS[i]) for i in STORM_LAT], "lon_deg": [float(wx.LONS[i]) for i in STORM_LON],
        "n_points": (STORM_LAT[1] - STORM_LAT[0] + 1) * (STORM_LON[1] - STORM_LON[0] + 1),
        "forecast_k": forecast_k, "truth_k": truth_k, "climatology_k": clim_k,
        "share_caught": share,
        "reference_range": list(STORM_RANGE), "in_reference_range": inside,
    }


# ---------------------------------------------------------------- printing

def show(var, value):
    """A number for the table: Z500 with 2 decimals, the temperatures with 4, '-' when there is none."""
    if value is None:
        return "-"
    return ("%.2f" if var == "z500" else "%.4f") % value


def show_pct(value):
    return "-" if value is None else "%+.1f%%" % value


def print_year(year, r):
    cut = "" if r["comparable"] else " (cut short by --limit)"
    print()
    print("%d: %d of %d starts%s, %s to %s; truth from %s; %.0f s"
          % (year, r["n_starts"], r["n_starts_full"], cut, r["first_start"], r["last_start"], r["truth"],
             r["seconds"]))
    print(("%-25s" % "  RMSE" + "".join(("%d h" % lead).center(30) for lead in SHOWN_H)).rstrip())
    print("%-25s" % "" + "".join("%10s%10s%10s" % ("tonight", "known", "diff") for lead in SHOWN_H))
    by_key = dict(((row["kind"], row["variable"], row["lead_h"]), row) for row in r["compare"])
    for kind in KINDS:
        if r["rmse"][kind] is None:
            continue
        for var in VARS:
            line = "  %-11s %-4s %-6s" % (kind, var, UNITS[var])
            for lead in SHOWN_H:
                row = by_key[(kind, var, lead)]
                line += "%10s%10s%10s" % (show(var, row["tonight"]), show(var, row["known"]),
                                          show_pct(row["diff_pct"]))
            print(line)

    if not r["comparable"]:
        print("  Only %d of %d starts were scored, so these numbers cannot be compared with the known values. "
              "No verdicts." % (r["n_starts"], r["n_starts_full"]))
        return
    judged = [row for row in r["compare"] if row["kind"] == "model" and row["ok"] is not None]
    if judged:
        parts = ["%s %s (%s)" % (row["variable"], "yes" if row["ok"] else "NO", show_pct(row["diff_pct"]))
                 for row in judged]
        print("  72 h, within +-%g%% of the reference model (\"same skill\")?  %s" % (BAND_PCT, ",  ".join(parts)))
    elif r["rmse"]["model"] is not None:
        print("  72 h: there is no reference model score for %d." % year)
    baselines = [row for row in r["compare"] if row["kind"] != "model" and row["ok"] is not None]
    wrong = [row for row in baselines if not row["ok"]]
    if not baselines:
        print("  Baselines: there are no known values for %d." % year)
    elif not wrong:
        print("  Baselines: all %d agree with their known values to within %g%%." % (len(baselines), BASELINE_PCT))
    else:
        print("  BASELINES DIFFER from their known values by more than %g%%. The scorer may be wrong:" % BASELINE_PCT)
        for row in wrong:
            print("      %s %s at %d h: %s against %s (%s)"
                  % (row["kind"], row["variable"], row["lead_h"], show(row["variable"], row["tonight"]),
                     show(row["variable"], row["known"]), show_pct(row["diff_pct"])))


def print_storm(s):
    print()
    print("Storm (the January 2019 cold wave): the %d-hour forecast from %s for %s."
          % (s["lead_h"], s["start"], s["valid"]))
    forecast = "none (no model)" if s["forecast_k"] is None else "%.2f K" % s["forecast_k"]
    print("  T850 averaged over %d grid points around Chicago: forecast %s, truth %.2f K, climatology %.2f K"
          % (s["n_points"], forecast, s["truth_k"], s["climatology_k"]))
    if s["share_caught"] is not None:
        print("  share of the cold caught = (forecast - climatology) / (truth - climatology) = %.3f"
              % s["share_caught"])
        print("  the reference for this recipe is %g to %g: tonight's is %s"
              % (STORM_RANGE[0], STORM_RANGE[1], "inside" if s["in_reference_range"] else "OUTSIDE"))


# ---------------------------------------------------------------- scores.json

def json_ready(x):
    """Plain JSON types only. A number that is not finite becomes null, so that any JSON reader can load the file."""
    if isinstance(x, dict):
        return dict((str(key), json_ready(value)) for key, value in x.items())
    if isinstance(x, (list, tuple)):
        return [json_ready(value) for value in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x if x is None or isinstance(x, str) else str(x)      # anything else: its text, so writing cannot fail


def write_json(path, obj):
    """Write to .tmp and rename, so a reader never sees half a file."""
    with open(path + ".tmp", "w") as f:
        json.dump(json_ready(obj), f, indent=1, allow_nan=False)
        f.write("\n")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- the run

def run(args, model=None):
    """Do the scoring that the arguments ask for, print the table, write scores.json. Returns its content."""
    t_start = time.time()
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        raise ScoreError("cannot create the output folder %s: %s" % (args.out, e))
    if not os.access(args.out, os.W_OK):
        raise ScoreError("the output folder %s cannot be written to" % args.out)
    data = Data(args.data)
    years = []
    for year in args.years:
        if not 1979 <= year <= 2020:
            raise ScoreError("year %d: the data covers 1979 to 2020" % year)
        if year not in years:
            years.append(year)
    # Find every row that will be read now, so that a missing file or a short one stops the run in its
    # first second, before the model is loaded.
    plans = dict((year, plan_year(year, data, args.limit)) for year in years)
    if args.storm:
        plan_storm(data)

    warnings = []
    forecaster, checkpoint = None, None
    if args.no_model:
        print("score.py: no model (--no-model), baselines only; data %s" % data.folder, flush=True)
    else:
        forecaster, checkpoint = make_forecaster(args.weights, args.device, model)
        about = describe_checkpoint(checkpoint) if checkpoint else ""
        print("score.py: weights %s%s; device %s, %s; batch %d; data %s"
              % (args.weights, " (%s)" % about if about else "", forecaster.device,
                 "bf16 autocast" if forecaster.bf16 else "float32", args.batch, data.folder), flush=True)
        if checkpoint and checkpoint["complete"] is False:
            warnings.append("%s is marked not complete: training stopped early, at step %s"
                            % (args.weights, checkpoint["step"]))
    print('"known" is what was measured before the event; on the model rows it is the reference model '
          "(the same recipe, trained on Nov 10).", flush=True)

    weights = wx.area_weights()
    scored = {}
    for year in years:
        result = score_year(year, plans[year], data, forecaster, args.batch, weights)
        scored[str(year)] = result
        print_year(year, result)
        if result["rmse"]["model"] is not None and not all(
                math.isfinite(v) for var in VARS for v in result["rmse"]["model"][var]):
            warnings.append("%d: some of the model's scores are not finite (they are null in scores.json)" % year)
        if result["baselines_ok"] is False:
            warnings.append("%d: baselines differ from their known values by more than %g%%; the scorer may be "
                            "wrong" % (year, BASELINE_PCT))
        sys.stdout.flush()
    storm = None
    if args.storm:
        storm = score_storm(data, forecaster)
        print_storm(storm)

    seconds = round(time.time() - t_start, 1)
    scores = {
        "what": "RMSE with cos(latitude) weights of forecasts started at 00Z and 12Z every second day; "
                "written by weather/score.py",
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "weights": None if args.no_model else args.weights,
        "checkpoint": checkpoint,
        "device": None if forecaster is None else str(forecaster.device),
        "bf16": None if forecaster is None else forecaster.bf16,
        "torch": None if forecaster is None else forecaster.torch.__version__,
        "data": data.folder,
        "batch": args.batch,
        "limit": args.limit,
        "seconds": seconds,
        "variables": VARS,
        "units": UNITS,
        "leads_h": LEADS_H,
        "years": scored,
        "storm": storm,
        "reference": {
            "note": "measured before the event on this data; 'model' is the reference model (same recipe, "
                    "trained Nov 10); RMSE by lead hour",
            "band_pct": BAND_PCT,
            "baseline_tolerance_pct": BASELINE_PCT,
            "storm_share_range": list(STORM_RANGE),
            "rmse": KNOWN,
        },
        "warnings": warnings,
    }
    path = os.path.join(args.out, "scores.json")
    write_json(path, scores)
    print()
    for line in warnings:
        print("WARNING: %s" % line)
    print("wrote %s (%.0f s in all)" % (path, seconds), flush=True)
    return scores


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def main(argv=None, model=None):
    """The command line. `model` is for the tests: a small torch.nn.Module used in place of the weights file."""
    p = argparse.ArgumentParser(
        description="Score a trained weather model against what really happened, beside persistence and "
                    "climatology.",
        epilog="Example: python weather/score.py --weights /out/stage2/final.pt --data /data --out /out/scores "
               "--years 2020 2019 --storm")
    p.add_argument("--weights", default=None,
                   help="the model file train.py wrote (stage 2's final.pt); not needed with --no-model")
    p.add_argument("--data", default=None, help="the data folder (default: $WX_DATA, or /data)")
    p.add_argument("--out", required=True, help="folder for scores.json (created if missing); nothing is written "
                                                "anywhere else")
    p.add_argument("--years", type=int, nargs="+", default=[2020, 2019], help="years to score (default: 2020 2019)")
    p.add_argument("--storm", action="store_true", help="also score the January 2019 cold wave around Chicago")
    p.add_argument("--limit", type=positive_int, default=None, metavar="N",
                   help="only the first N starts of each year (a quick check; no verdicts)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto",
                   help="where the model runs (default auto: the GPU if there is one)")
    p.add_argument("--batch", type=positive_int, default=16, help="forecasts run together on the GPU (default 16)")
    p.add_argument("--no-model", action="store_true", help="baselines only; PyTorch is not imported")
    args = p.parse_args(argv)
    if model is None and not args.no_model and not args.weights:
        p.error("--weights is needed unless --no-model is given")
    try:
        run(args, model)
    except ScoreError as e:
        print("score.py: FAILED: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
