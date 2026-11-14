"""Fine-tune the weather model on NOAA's GFS analyses, and measure on held-out weeks whether that helped.

The model (weather/train.py) learned from ERA5 reanalysis. A live forecast starts from NOAA's GFS analysis,
which differs a little from ERA5 everywhere (the two humidity maps sit about 0.45 standard deviations higher,
for one). weather/gfs_archive.py stacks about a year of GFS analyses on the model's grid. This script trains
the model a little further on those, then scores the result on weeks it never trained on. A gain counts only
if this script measures it.

    python weather/finetune.py --init /out/stage2/final.pt --analyses /data/gfs --out /out/finetune

Score a weights file without training (for example the same tuned.pt, on the other machine):

    python weather/finetune.py --init /out/finetune/tuned.pt --analyses /data/gfs --out /out/check --eval-only

A quick try of the whole path (too few steps and forecasts for the numbers to mean much):

    python weather/finetune.py --init /out/stage2/final.pt --analyses /data/gfs --out /out/try --steps 40 --max-starts 8

What it does, in order:
  1. Splits the analyses by time, never at random:
         TEST        every analysis at or after --test-from
         VALIDATION  the --val-days days before that
         TRAIN       everything earlier, less a gap of --lead-hours before validation that nothing uses
  2. Picks the forecast starts of validation and of test: every 00 and 12 UTC analysis whose next --lead-hours
     of 6-hourly analyses are all there, in the same split. The data has gaps, so times are checked, not row
     numbers.
  3. Scores the base model (--init) on validation and on test. From each start the model is fed its own
     output, six hours at a time, and every step is compared with the analysis for that time.
  4. Trains the model on TRAIN: --steps steps of --batch samples. A sample is an analysis and the --rollout
     analyses after it. The loss is the one weather/train.py used (Z500, T850 and T2M count three times).
     Every --eval-every steps, and at the end, the model is scored on validation; the best weights are kept.
  5. Scores the best weights on TEST, once, and sets them beside the base model.
  6. Writes the weights and the report.
It refuses to start if validation or test gives fewer than 8 forecast starts, or if TRAIN gives fewer than
100 training samples (with --eval-only nothing is trained, so TRAIN may be small).

The scores (lower is better):
    Z500, T850, T2M   RMSE in physical units with cos(latitude) weights: for each start the weighted mean
                      squared error over the grid, then the square root of the mean over starts
    all 20 channels   the same RMSE over every channel, in z-score units
    O'Hare            the mean absolute error of 2-m temperature at Chicago O'Hare, in C (bilinear on the grid)
    persistence       the same scores for the forecast "nothing changes", for scale
"Best on validation" is the lowest mean of (RMSE / the base model's RMSE) over Z500, T850 and T2M at 24, 48
and 72 hours. The base model scores 1.0 by definition.

The test table gives each change in percent with a 90% interval. The interval is a paired bootstrap: the
days of the test weeks are drawn again with replacement 1,000 times (a day's 00 and 12 UTC forecasts stay
together), both models are scored on every draw, and the 5th and 95th percentiles of the change are taken.
"better" means the whole interval is below zero, "worse" that it is above zero, and "no clear change" that
it contains zero.

Files in --out:
    tuned.pt               the best weights, laid out like train.py's final.pt (the forecast server loads it)
    finetune_report.json   everything: settings, splits, every score, the intervals, the verdicts, timings
    finetune_report.md     a short table to paste into a README
    log.jsonl              one JSON line per training step, and a line for the start, for each validation
                           check and for the end (appended to, so a second run adds to the first)
With --eval-only there is no tuned.pt and no log.jsonl. Give it its own --out, or it overwrites the report.

stats.json (the standard deviations that turn z-scores into physical units) is looked for in --stats; without
that, beside the analyses, and then in $WX_DATA.

It can run beside a forecast server on the same GPU: only one batch is on the GPU at a time. SIGTERM (or
Ctrl-C) stops it after the current step. It then saves the best weights so far as tuned.pt (before the first
validation check: the weights as they are) and exits without scoring them on the test weeks.

Exit codes: 0 = finished, also when the tuned model is not better (the report says so); 1 = refused or
failed, with a message; 2 = a mistake on the command line; 3 = stopped early by a signal.

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import bisect
import contextlib
import datetime
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                     # wx_common.py sits beside this file

import wx_common as wx  # noqa: E402

STEP = datetime.timedelta(hours=wx.STEP_HOURS)   # one model step: 6 hours
START_HOURS = (0, 12)                            # forecasts start at 00 and 12 UTC
OHARE_LAT, OHARE_LON = 41.9602, -87.9316         # Chicago O'Hare
MIN_STARTS = 8                                   # validation and test each need this many forecast starts
MIN_SAMPLES = 100                                # training needs this many samples
SELECT_LEADS_H = (24, 48, 72)                    # "best on validation" looks at these leads
VERDICT_LEADS_H = (24, 72)                       # the leads that get a verdict in words
BETAS = (0.9, 0.95)                              # AdamW, as in weather/train.py
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
RESAMPLES = 1000                                 # bootstrap draws
PRINT_EVERY = 20                                 # training steps between printed lines
CPU_THREADS = 4                                  # PyTorch's CPU threads; plenty when the GPU does the work
EXIT_STOPPED = 3                                 # the exit code after SIGTERM or Ctrl-C

# The scores, in the order they are shown. squares=True is an RMSE: squared errors are averaged over the
# starts and the square root is taken at the end. squares=False is a plain mean of absolute errors.
SCORES = [
    dict(key="z500_rmse", name="Z500 RMSE", unit="m2/s2", head="Z500 m2/s2", format="%.2f", squares=True),
    dict(key="t850_rmse", name="T850 RMSE", unit="K", head="T850 K", format="%.3f", squares=True),
    dict(key="t2m_rmse", name="T2M RMSE", unit="K", head="T2M K", format="%.3f", squares=True),
    dict(key="all_channels_rmse_z", name="all 20 channels RMSE", unit="z-units", head="all 20 z",
         format="%.4f", squares=True),
    dict(key="ohare_t2m_mae_c", name="O'Hare 2-m temp. error", unit="C", head="O'Hare C", format="%.2f",
         squares=False),
]
# The three graded maps: (score key, short name, channel).
PHYSICAL = [("z500_rmse", "z500", wx.CH_Z500), ("t850_rmse", "t850", wx.CH_T850), ("t2m_rmse", "t2m", wx.CH_T2M)]


class FinetuneError(Exception):
    """A problem to report in one line (a missing file, a split that is too small). Not a bug."""


# ---------------------------------------------------------------- times and splits

def parse_time(text, what):
    """A time such as 2026-10-01T00 (UTC) as a datetime. `what` names the value in the error message."""
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%dT%H")
    except (TypeError, ValueError):
        raise FinetuneError("%s is %r; it must look like 2026-10-01T00" % (what, text))


def time_text(when):
    """datetime -> 2026-10-01T00."""
    return when.strftime("%Y-%m-%dT%H")


def one_line(error, limit=300):
    """An exception's text on one line and not too long, to put inside a message."""
    return " ".join(str(error).split())[:limit]


def split_rows(times, test_from, val_days, lead_hours):
    """Cut the rows into train, gap, validation and test by their times. `times` must be in increasing order.

    Returns {name: (first row, the row after the last)}:
        test        the rows at or after test_from
        validation  the val_days days before test_from
        gap         the lead_hours hours before validation: no split uses them
        train       everything earlier
    """
    val_from = test_from - datetime.timedelta(days=val_days)
    gap_from = val_from - datetime.timedelta(hours=lead_hours)
    a = bisect.bisect_left(times, gap_from)
    b = bisect.bisect_left(times, val_from)
    c = bisect.bisect_left(times, test_from)
    return {"train": (0, a), "gap": (a, b), "validation": (b, c), "test": (c, len(times))}


def chain_starts(times, first, stop, n_steps, hours=None):
    """The rows in [first, stop) whose next n_steps rows are in there too, each exactly 6 hours after the one
    before. The data has gaps, so this compares the times, not the row numbers.
    hours: keep only the rows at these UTC hours (forecast starts); None keeps every hour (training samples)."""
    found = []
    for row in range(first, stop - n_steps):
        if hours is not None and times[row].hour not in hours:
            continue
        if all(times[row + k] - times[row] == k * STEP for k in range(1, n_steps + 1)):
            found.append(row)
    return found


def span_text(times, rows):
    """A row range in words: "2026-09-17T00 to 2026-09-30T18, 56 rows", or "no rows"."""
    first, stop = rows
    if stop <= first:
        return "no rows"
    return "%s to %s, %d row%s" % (time_text(times[first]), time_text(times[stop - 1]), stop - first,
                                   "" if stop - first == 1 else "s")


def make_plan(args, times):
    """Steps 1 and 2: the splits, the training samples and the forecast starts.
    Raises FinetuneError if the times are out of order or a split is too small to be of use."""
    for earlier, later in zip(times, times[1:]):
        if later <= earlier:
            raise FinetuneError("the times of the analyses must increase, but %s is followed by %s"
                                % (time_text(earlier), time_text(later)))
    test_from = parse_time(args.test_from, "--test-from")
    n_steps = args.lead_hours // wx.STEP_HOURS
    leads = [wx.STEP_HOURS * k for k in range(1, n_steps + 1)]
    rows = split_rows(times, test_from, args.val_days, args.lead_hours)
    samples = chain_starts(times, rows["train"][0], rows["train"][1], args.rollout)
    starts, available = {}, {}
    for name in ("validation", "test"):
        found = chain_starts(times, rows[name][0], rows[name][1], n_steps, START_HOURS)
        available[name] = len(found)
        starts[name] = found[:args.max_starts] if args.max_starts else found

    problems = []
    for name in ("validation", "test"):
        if available[name] < MIN_STARTS:
            problems.append("The %s split (%s) gives %d forecast starts of %d hours; at least %d are needed."
                            % (name, span_text(times, rows[name]), available[name], args.lead_hours, MIN_STARTS))
    if not args.eval_only and len(samples) < MIN_SAMPLES:
        problems.append("The training split (%s) gives %d training samples; at least %d are needed."
                        % (span_text(times, rows["train"]), len(samples), MIN_SAMPLES))
    if problems:
        whole = span_text(times, (0, len(times)))
        raise FinetuneError("%s The analyses are: %s. Change --test-from (now %s), --val-days or --lead-hours, "
                            "or fetch more analyses." % (" ".join(problems), whole, args.test_from))
    val_from = test_from - datetime.timedelta(days=args.val_days)
    return {
        "test_from": test_from,
        "validation_from": val_from,
        "gap_from": val_from - datetime.timedelta(hours=args.lead_hours),
        "rows": rows,
        "leads_h": leads,
        "shown_h": [leads[0]] + [lead for lead in leads if lead % 24 == 0],     # 6 h, then every 24 h
        "samples": samples,
        "starts": starts,
        "available": available,
    }


def describe_plan(plan, times, args):
    """Print the splits, and return them as the report's "splits" entry."""
    out = {}
    for name in ("train", "gap", "validation", "test"):
        first, stop = plan["rows"][name]
        out[name] = {"first": time_text(times[first]) if stop > first else None,
                     "last": time_text(times[stop - 1]) if stop > first else None,
                     "rows": stop - first}
    out["train"]["samples"] = len(plan["samples"])
    out["gap"]["hours"] = args.lead_hours
    for name in ("test_from", "validation_from", "gap_from"):
        out[name] = time_text(plan[name])
    for name in ("validation", "test"):
        rows = plan["starts"][name]
        out[name]["starts"] = len(rows)
        out[name]["starts_available"] = plan["available"][name]
        out[name]["start_times"] = [time_text(times[row]) for row in rows]
        out[name]["start_days"] = len(set(times[row].date() for row in rows))

    notes = {
        "train": "%d training samples (an analysis and the %d after it)" % (len(plan["samples"]), args.rollout),
        "gap": "used by nothing: it keeps training %d h clear of validation" % args.lead_hours,
    }
    for name in ("validation", "test"):
        used, found = out[name]["starts"], out[name]["starts_available"]
        some = "%d of %d" % (used, found) if used != found else "%d" % found
        notes[name] = "%s forecast starts (00 and 12 UTC, %d h each)%s" % (
            some, args.lead_hours, ", cut by --max-starts" if used != found else "")
    print("analyses: %s" % span_text(times, (0, len(times))))
    print("split by time, never at random:")
    for name in ("train", "gap", "validation", "test"):
        print("  %-11s %-44s %s" % (name, span_text(times, plan["rows"][name]), notes[name]))
    sys.stdout.flush()
    return out


# ---------------------------------------------------------------- the model, the loss, the learning rate

def load_base_model(path):
    """The AFNO with the weights weather/train.py saved. A tuned.pt written by this script loads the same way."""
    if not os.path.isfile(path):
        raise FinetuneError("the weights file %s does not exist" % path)
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise FinetuneError("cannot read the weights file %s (%s: %s)" % (path, type(e).__name__, one_line(e)))
    if not isinstance(saved, dict) or "model" not in saved:
        raise FinetuneError("%s has no 'model' entry; it must be a file written by weather/train.py" % path)
    model = wx.build_afno()
    try:
        model.load_state_dict(saved["model"])
    except RuntimeError as e:
        raise FinetuneError("the weights in %s do not fit the model: %s" % (path, one_line(e)))
    return model


def pick_device(want):
    """auto = the CUDA GPU if PyTorch finds one, else the CPU."""
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        raise FinetuneError("--device cuda was asked for, but PyTorch finds no CUDA GPU")
    return torch.device(want)


def learning_rate(step, peak, warmup, total):
    """The rate for the 0-based step `step`: a straight climb to `peak` over the first `warmup` steps, then
    half a cosine wave down. The wave reaches zero at step `total`; the last step that runs, total - 1, is
    just above zero."""
    if step < warmup:
        return peak * ((step + 1) / warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


class SampleStream(object):
    """The seeded order in which training samples are drawn. Epoch e is a fresh shuffle of 0 .. n-1
    (numpy's default_rng([seed, e]).permutation(n)); the stream runs straight on from one epoch into the next."""

    def __init__(self, n, seed):
        self.n, self.seed = n, seed
        self.epoch, self.pos = 0, 0
        self.order = self._order(0)

    def _order(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def take(self, count):
        """The next `count` sample numbers."""
        pieces = []
        while count > 0:
            if self.pos == self.n:
                self.epoch, self.pos = self.epoch + 1, 0
                self.order = self._order(self.epoch)
            piece = self.order[self.pos:self.pos + count]
            pieces.append(piece)
            self.pos += len(piece)
            count -= len(piece)
        return np.concatenate(pieces)


def autocast(on_cuda):
    """bf16 autocast on CUDA; plain float32 on a CPU."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda else contextlib.nullcontext()


def rollout_loss(model, x, targets, weights, on_cuda):
    """The training loss, as in weather/train.py: the mean over the fed-back 6-hour steps of the mean of
    weights * (forecast - target)^2, in z-score units. targets: (batch, steps, 20, lat, lon).
    Every output is cast to float32 and fed back without detaching, so gradients flow through all the steps."""
    n_steps = targets.shape[1]
    state = x
    total = 0.0
    for k in range(n_steps):
        with autocast(on_cuda):
            out = model(state)
        state = out.float()
        total = total + (weights * (state - targets[:, k]) ** 2).mean() / n_steps
    return total


def copy_state(model):
    """A copy of the model's weights on the CPU, safe from further training."""
    return dict((name, value.detach().to("cpu", copy=True)) for name, value in model.state_dict().items())


# ---------------------------------------------------------------- scoring

def point_weights(lat, lon):
    """The four grid points around a place and their bilinear weights: [(row, column, weight), ...].
    The grid has 120 latitudes from -88.5 to 90 and 240 longitudes from 0 to 358.5; longitude wraps around."""
    degrees = float(wx.LATS[1] - wx.LATS[0])                     # 1.5
    lat = min(max(float(lat), float(wx.LATS[0])), float(wx.LATS[-1]))
    y = (lat - float(wx.LATS[0])) / degrees
    i = min(int(math.floor(y)), wx.N_LAT - 2)
    fy = y - i                                                   # how far towards row i + 1
    x = (float(lon) % 360.0) / degrees
    j = int(math.floor(x)) % wx.N_LON
    fx = x - math.floor(x)                                       # how far towards the next column
    j2 = (j + 1) % wx.N_LON
    return [(i, j, (1.0 - fy) * (1.0 - fx)), (i, j2, (1.0 - fy) * fx),
            (i + 1, j, fy * (1.0 - fx)), (i + 1, j2, fy * fx)]


class Scoring(object):
    """What the scores need besides the forecasts: stats.json's standard deviations (they turn z-scores into
    physical units), the area weights of the grid, and the grid points around the place whose temperature is
    checked. By default the model's grid and Chicago O'Hare. The tests pass a small grid of their own."""

    def __init__(self, std, area=None, point=None):
        self.std = np.asarray(std, np.float64)
        self.area = wx.area_weights() if area is None else np.asarray(area, np.float64)
        self.point = point_weights(OHARE_LAT, OHARE_LON) if point is None else list(point)


def grid_errors(forecast, truth, area, point):
    """Compare a batch of forecasts with the truth, in float64. forecast, truth: (batch, 20, lat, lon) tensors.
    Returns (mse, at_point):
        mse       (batch, 20): the mean over the grid of area * (forecast - truth)^2, per channel
        at_point  (batch,): forecast - truth of 2-m temperature at the place (bilinear)"""
    diff = forecast.double() - truth.double()
    at_point = 0.0
    for i, j, weight in point:
        at_point = at_point + weight * diff[:, wx.CH_T2M, i, j]
    mse = (diff * diff * area).mean(dim=(-2, -1))
    return mse, at_point


def evaluate(model, data, starts, n_steps, device, scoring, batch=16, stop=None):
    """Roll the model forward from every start row and measure each 6-hour step against the analysis for that
    time. data[row] is a state (20, lat, lon); the truth for step k of start row r is data[r + k], which
    chain_starts() has checked to be the right time.

    The model runs in eval mode under torch.inference_mode, with bf16 autocast on CUDA. Every output is cast
    to float32 and fed back in, as in training and in the forecast server.

    Returns {"model": errors, "persistence": errors}; persistence is the forecast "nothing changes".
    Each errors is, in z-score units and float64:
        "mse"    (starts, steps, 20): the area-weighted mean squared error over the grid, per channel
        "point"  (starts, steps): forecast - truth of 2-m temperature at the place
    Returns None if stop() turned true before the end.
    """
    device = torch.device(device)
    on_cuda = device.type == "cuda"
    area = torch.as_tensor(scoring.area, dtype=torch.float64, device=device)
    n_channels = int(data.shape[1])
    out = {}
    for who in ("model", "persistence"):
        out[who] = {"mse": np.zeros((len(starts), n_steps, n_channels)), "point": np.zeros((len(starts), n_steps))}

    def load(rows):
        return torch.from_numpy(np.stack([data[row] for row in rows])).to(device).float()

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for a in range(0, len(starts), batch):
                rows = starts[a:a + batch]
                b = a + len(rows)
                first = load(rows)
                state = first
                for k in range(1, n_steps + 1):
                    if stop is not None and stop():
                        return None
                    with autocast(on_cuda):
                        state = model(state)
                    state = state.float()
                    truth = load([row + k for row in rows])
                    for who, forecast in (("model", state), ("persistence", first)):
                        mse, at_point = grid_errors(forecast, truth, area, scoring.point)
                        out[who]["mse"][a:b, k - 1] = mse.cpu().numpy()
                        out[who]["point"][a:b, k - 1] = at_point.cpu().numpy()
    finally:
        model.train(was_training)
    return out


def start_values(errors, std):
    """What each score averages over the starts: {score key: float64 (starts, leads)}.
    For an RMSE these are squared errors. Physical units are z-score units times the channel's standard
    deviation (the mean cancels in forecast - truth). For O'Hare they are absolute errors in C."""
    mse = errors["mse"]
    values = {}
    for key, _, channel in PHYSICAL:
        values[key] = mse[:, :, channel] * std[channel] ** 2
    values["all_channels_rmse_z"] = mse.mean(axis=2)
    values["ohare_t2m_mae_c"] = np.abs(errors["point"]) * std[wx.CH_T2M]
    return values


def scores_of(values):
    """Average over the starts: {score key: [one number per lead]}.
    An RMSE is the square root of the mean of the squared errors (not the mean of square roots)."""
    scores = {}
    for score in SCORES:
        mean = values[score["key"]].mean(axis=0)
        scores[score["key"]] = [float(v) for v in (np.sqrt(mean) if score["squares"] else mean)]
    return scores


def selection_score(scores, base_scores, leads):
    """How good a model is on validation, in one number: the mean of RMSE / the base model's RMSE over Z500,
    T850 and T2M at 24, 48 and 72 hours (the leads that exist). The base model scores 1.0; lower is better."""
    ratios = []
    for key, _, _ in PHYSICAL:
        for lead in SELECT_LEADS_H:
            if lead in leads:
                k = leads.index(lead)
                ratios.append(scores[key][k] / base_scores[key][k])
    return float(np.mean(ratios))


def bootstrap_interval(base, tuned, days, squares, seed, resamples=RESAMPLES):
    """A 90% interval for the change of one score, tuned against base, in percent: a paired bootstrap by day.

    base, tuned: what the score averages, per forecast start, shape (starts,) or (starts, leads): squared
    errors for an RMSE (squares=True), absolute errors otherwise. days: one label per start; starts with
    the same label are always drawn together.

    Each of the `resamples` draws picks as many days as there are, with replacement, and scores both models
    on the starts of the days it picked. Returns (low, high): the 5th and 95th percentiles of
    100 * (tuned score / base score - 1). They are NaN if a draw's base score is zero.
    """
    base = np.asarray(base, np.float64)
    tuned = np.asarray(tuned, np.float64)
    number = dict((day, n) for n, day in enumerate(sorted(set(days))))
    day_of = np.array([number[day] for day in days])
    base_by_day = np.zeros((len(number),) + base.shape[1:])
    tuned_by_day = np.zeros((len(number),) + tuned.shape[1:])
    np.add.at(base_by_day, day_of, base)                         # each day's sum over its starts
    np.add.at(tuned_by_day, day_of, tuned)

    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(number), size=(resamples, len(number)))
    times_drawn = np.zeros((resamples, len(number)))             # [draw, day]: how often the draw picked the day
    np.add.at(times_drawn, (np.arange(resamples)[:, None], drawn), 1.0)
    base_sum = np.tensordot(times_drawn, base_by_day, axes=1)    # (resamples, ...): sums over the drawn starts
    tuned_sum = np.tensordot(times_drawn, tuned_by_day, axes=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Both models are scored on the same starts, so the ratio of the sums is the ratio of the means.
        ratio = np.where(base_sum > 0, tuned_sum / base_sum, np.nan)
        if squares:
            ratio = np.sqrt(ratio)
        low, high = np.percentile(100.0 * (ratio - 1.0), [5, 95], axis=0)
    return low, high


def verdict_word(low, high):
    """The 90% interval of a change in words: "better" if all of it is below zero, "worse" if all of it is
    above zero, "no clear change" if it contains zero (or could not be worked out)."""
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)):
        return "no clear change"
    if high < 0:
        return "better"
    if low > 0:
        return "worse"
    return "no clear change"


def compare(base, tuned, persistence, days, leads, shown, seed):
    """Base against tuned on the same forecast starts. base, tuned, persistence: start_values() of each.
    Returns one row per score and shown lead: the three scores, the change in percent, its 90% interval,
    and the verdict word."""
    base_scores, tuned_scores, still_scores = scores_of(base), scores_of(tuned), scores_of(persistence)
    rows = []
    for score in SCORES:
        key = score["key"]
        low, high = bootstrap_interval(base[key], tuned[key], days, score["squares"], seed)
        for lead in shown:
            k = leads.index(lead)
            b, t = base_scores[key][k], tuned_scores[key][k]
            rows.append({
                "score": key, "lead_h": lead,
                "persistence": still_scores[key][k], "base": b, "tuned": t,
                "change_pct": 100.0 * (t / b - 1.0) if b > 0 else None,
                "ci90_pct": [float(low[k]), float(high[k])],
                "verdict": verdict_word(float(low[k]), float(high[k])),
            })
    return rows


def verdicts_of(rows):
    """The verdict words for Z500, T850 and T2M at 24 and 72 hours (the comparisons that count), and one
    sentence for the whole result.
    Returns (verdicts, overall): verdicts = {"z500": {"24h": "better", ...}, ...}; overall = {"word", "sentence"}."""
    verdicts = {}
    words = []
    for key, short, _ in PHYSICAL:
        verdicts[short] = {}
        for row in rows:
            if row["score"] == key and row["lead_h"] in VERDICT_LEADS_H:
                verdicts[short]["%dh" % row["lead_h"]] = row["verdict"]
                words.append(row["verdict"])
    better, worse = words.count("better"), words.count("worse")
    counted = "of %d comparisons, %d better, %d worse, %d no clear change" % (
        len(words), better, worse, len(words) - better - worse)
    if better and not worse:
        word = "better"
        sentence = "The tuned model is better than the base model on the test weeks (%s)." % counted
    elif worse and not better:
        word = "worse"
        sentence = ("The tuned model is NOT better than the base model on the test weeks: it is worse (%s). "
                    "tuned.pt was saved anyway." % counted)
    elif better and worse:
        word = "mixed"
        sentence = ("The tuned model is better than the base model in some comparisons and worse in others "
                    "(%s). Read the table before using tuned.pt." % counted)
    else:
        word = "no clear change"
        sentence = ("The tuned model is NOT measurably better than the base model on the test weeks (%s). "
                    "tuned.pt was saved anyway." % counted)
    return verdicts, {"word": word, "sentence": sentence}


# ---------------------------------------------------------------- stopping on a signal

class StopFlag(object):
    """Turns true on SIGTERM or Ctrl-C. The work then stops at the next safe place.
    A second signal stops the process at once."""

    def __init__(self):
        self.signal = None
        self.before = {}

    def __call__(self):
        return self.signal is not None

    def on_signal(self, signum, frame):
        if self.signal is not None:
            raise KeyboardInterrupt
        self.signal = signum

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.before[sig] = signal.signal(sig, self.on_signal)
            except ValueError:                   # not the main thread: signals cannot be caught from here
                pass

    def restore(self):
        for sig, handler in self.before.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        self.before = {}


# ---------------------------------------------------------------- printing

def show(score, value):
    """A score as text in its own number format; "-" when there is none."""
    if value is None or not math.isfinite(value):
        return "-"
    return score["format"] % value


def show_pct(value):
    if value is None or not math.isfinite(value):
        return "-"
    return "%+.1f%%" % value


def show_interval(pair):
    if pair is None or not all(v is not None and math.isfinite(v) for v in pair):
        return "-"
    return "%+.1f%% to %+.1f%%" % (pair[0], pair[1])


def print_scores(title, leads, shown, who):
    """A small table of scores. who: [(label, scores_of() result), ...]."""
    print(title)
    print("  %-12s %5s" % ("", "lead") + "".join("%13s" % score["head"] for score in SCORES))
    for label, scores in who:
        for lead in shown:
            k = leads.index(lead)
            print("  %-12s %3d h" % (label, lead)
                  + "".join("%13s" % show(score, scores[score["key"]][k]) for score in SCORES))
    sys.stdout.flush()


def print_comparison(rows):
    """The test table: one line per score and shown lead."""
    by_key = dict((score["key"], score) for score in SCORES)
    print("  %-34s %5s %12s %10s %10s %8s   %s"
          % ("score", "lead", "persistence", "base", "tuned", "change", "90% interval of the change"))
    for row in rows:
        score = by_key[row["score"]]
        print("  %-34s %3d h %12s %10s %10s %8s   %s"
              % ("%s (%s)" % (score["name"], score["unit"]), row["lead_h"], show(score, row["persistence"]),
                 show(score, row["base"]), show(score, row["tuned"]), show_pct(row["change_pct"]),
                 show_interval(row["ci90_pct"])))


def verdict_lines(rows):
    """What the verdict lines are made of, for Z500, T850 and T2M at 24 and 72 hours:
    [(name, lead, verdict word, the change as text, its 90% interval as text), ...]."""
    lines = []
    for key, short, _ in PHYSICAL:
        for row in rows:
            if row["score"] == key and row["lead_h"] in VERDICT_LEADS_H:
                lines.append((short.upper(), row["lead_h"], row["verdict"],
                              show_pct(row["change_pct"]), show_interval(row["ci90_pct"])))
    return lines


# ---------------------------------------------------------------- files

def json_ready(x):
    """Plain JSON types only. A number that is not finite becomes null, so that any JSON reader can load it."""
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
    return x


def write_text(path, text):
    """Write to .tmp and rename, so a reader never sees half a file."""
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def save_weights(path, saved):
    torch.save(saved, path + ".tmp")
    os.replace(path + ".tmp", path)


def report_markdown(report):
    """The short report: a table to paste into a README, the verdicts, and how the test weeks were kept apart."""
    settings, splits = report["settings"], report["splits"]
    leads, shown = report["leads_h"], report["shown_leads_h"]
    test = report["scores"]["test"]
    tuned = test["tuned"]
    selection = report["selection"]
    lines = []
    if report["mode"] == "eval-only":
        lines += ["### The weather model on GFS analyses: scores on held-out weeks", "",
                  "Weights: `%s`. This run trained nothing; it only scored these weights." % settings["init"]]
    else:
        lines += ["### Fine-tuning the weather model on GFS analyses: scores on held-out weeks", "",
                  "Base weights: `%s`. Fine-tuned for %d steps (batch %d, %d fed-back steps, peak learning rate "
                  "%g) on %d training samples from GFS analyses between %s and %s."
                  % (settings["init"], report["training"]["steps_done"], settings["batch"], settings["rollout"],
                     settings["lr"], splits["train"]["samples"], splits["train"]["first"], splits["train"]["last"])]
        if selection["best_step"] is None:
            lines.append("No weights were saved.")
        elif not selection["checked_on_validation"]:
            lines.append("tuned.pt holds the weights after step %d. They were not scored on validation."
                         % selection["best_step"])
        else:
            lines.append("tuned.pt holds the weights after step %d, the best of %d validation checks "
                         "(validation score %.4f; the base model is 1.0, lower is better)."
                         % (selection["best_step"], len(selection["checks"]), selection["best_score"]))
            if not selection["beats_base_on_validation"]:
                lines.append("No validation check beat the base model.")
    lines.append("")
    lines.append("Test: %d forecasts started at 00 and 12 UTC from %s to %s, each run to %d h and compared "
                 "with the GFS analysis at every valid time."
                 % (splits["test"]["starts"], splits["test"]["start_times"][0], splits["test"]["start_times"][-1],
                    leads[-1]))
    if settings["max_starts"]:
        lines.append("Only the first %d forecast starts of validation and of test were used (--max-starts): "
                     "a quick check, not the full score." % settings["max_starts"])
    if report["stopped_early"]:
        lines.append("")
        lines.append("**The run was stopped early. The tuned weights were not scored on the test weeks; "
                     "the table shows the base model only.**")
    lines.append("")
    lines.append("| score | " + " | ".join("%d h" % lead for lead in shown) + " |")
    lines.append("|---|" + "---|" * len(shown))
    by_cell = {}
    if report["comparison"] is not None:
        by_cell = dict(((row["score"], row["lead_h"]), row) for row in report["comparison"]["rows"])
    for score in SCORES:
        cells = []
        for lead in shown:
            k = leads.index(lead)
            if tuned is None:
                cells.append(show(score, test["base"][score["key"]][k]))
            else:
                row = by_cell[(score["key"], lead)]
                cells.append("%s -> %s (%s)" % (show(score, row["base"]), show(score, row["tuned"]),
                                                show_pct(row["change_pct"])))
        lines.append("| %s (%s) | %s |" % (score["name"], score["unit"], " | ".join(cells)))
    for score in SCORES[:3]:
        cells = [show(score, test["persistence"][score["key"]][leads.index(lead)]) for lead in shown]
        lines.append("| persistence: %s (%s) | %s |" % (score["name"], score["unit"], " | ".join(cells)))
    lines.append("")
    if tuned is None:
        lines.append("Each cell is the %s score; lower is better."
                     % ("model's" if report["mode"] == "eval-only" else "base model's"))
    else:
        lines.append("Each cell: base -> tuned (change); lower is better.")
    lines.append("RMSE is weighted by cos(latitude). O'Hare is the mean absolute error of 2-m temperature at "
                 "Chicago O'Hare. Persistence is the forecast \"nothing changes\", for scale.")
    if report["comparison"] is not None:
        lines.append("")
        lines.append("In words (in brackets: the change and its 90% interval):")
        lines.append("")
        for name, lead, word, change, interval in verdict_lines(report["comparison"]["rows"]):
            lines.append("- %s at %d h: **%s** (%s; %s)" % (name, lead, word, change, interval))
        lines.append("")
        lines.append(report["overall"]["sentence"])
        lines.append("")
        lines.append("\"Better\" means the whole interval is below zero, \"worse\" that it is above zero, and "
                     "\"no clear change\" that it contains zero. The intervals come from a paired bootstrap: "
                     "the %d test days were drawn again with replacement %d times, a day's forecasts "
                     "staying together." % (report["comparison"]["days"], report["comparison"]["resamples"]))
    lines.append("")
    if report["mode"] == "eval-only":
        lines.append("The split is by date and is the one a fine-tuning run uses (test: from %s on), so these "
                     "numbers can be set beside its report." % splits["test_from"])
    elif report["complete"]:
        lines.append("How the test weeks were kept apart: the analyses were split by date before anything ran, "
                     "training used only those up to %s, the %d days before %s were used only to pick the best "
                     "weights, and the weeks from %s on were scored once, at the end."
                     % (splits["train"]["last"], settings["val_days"], splits["test_from"], splits["test_from"]))
    else:
        lines.append("The analyses were split by date before anything ran, and training used only those up to "
                     "%s. The tuned weights have not seen the weeks from %s on, and have not been scored on "
                     "them." % (splits["train"]["last"], splits["test_from"]))
    lines.append("")
    lines.append("Written by weather/finetune.py on %s (%s, PyTorch %s). Every number is in "
                 "finetune_report.json." % (report["written"][:10], report["device"], report["torch"]))
    return "\n".join(lines) + "\n"


def write_report(out, report):
    """finetune_report.json and finetune_report.md."""
    ready = json_ready(report)
    write_text(os.path.join(out, "finetune_report.json"), json.dumps(ready, indent=1, allow_nan=False) + "\n")
    write_text(os.path.join(out, "finetune_report.md"), report_markdown(ready))


# ---------------------------------------------------------------- training

def train(args, model, data, plan, scoring, device, base_scores, stop, log):
    """Step 4: train on TRAIN, score on validation every --eval-every steps and at the end, keep the best.

    base_scores: the base model's validation scores. log(obj) appends one line to log.jsonl.
    Returns {"steps_done", "best", "checks", "steps_per_s", "losses", "train_seconds", "check_seconds"}.
    best is None if no validation check passed; else {"step", "score", "scores", "state"}.
    """
    on_cuda = device.type == "cuda"
    samples, leads = plan["samples"], plan["leads_h"]
    weights = torch.tensor(wx.t7_channel_weights(), dtype=torch.float32, device=device).view(1, -1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, betas=BETAS, weight_decay=WEIGHT_DECAY,
                                  **({"fused": True} if on_cuda else {}))
    forward = torch.compile(model) if args.compile else model   # `model` stays the handle for the weights
    stream = SampleStream(len(samples), args.seed)
    torch.manual_seed(args.seed)
    model.train()

    best, checks, losses = None, [], []
    step = 0
    timed_steps, timed_seconds, train_seconds, check_seconds = 0, 0.0, 0.0, 0.0
    while step < args.steps and not stop():
        t0 = time.perf_counter()
        lr = learning_rate(step, args.lr, args.warmup, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows = [samples[n] for n in stream.take(args.batch)]
        # Each sample is rows r .. r + rollout: the input and its targets. Only this batch goes to the GPU.
        block = torch.from_numpy(np.stack([data[r:r + args.rollout + 1] for r in rows])).to(device).float()
        optimizer.zero_grad(set_to_none=True)
        try:
            loss = rollout_loss(forward, block[:, 0], block[:, 1:], weights, on_cuda)
        except Exception as e:
            if step > 0 or not args.compile:
                raise
            # torch.compile does its work in the first call, and that needs a working compiler on the machine.
            raise FinetuneError("the first training step failed with --compile (%s: %s). Run again without "
                                "--compile." % (type(e).__name__, one_line(e)))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item()    # the norm before clipping
        loss_value = loss.item()
        if not (math.isfinite(loss_value) and math.isfinite(grad_norm)):
            # Checked before optimizer.step(), so the weights are not spoiled; but the run cannot go on.
            log({"event": "error", "step": step + 1, "loss": repr(loss_value), "grad_norm": repr(grad_norm)})
            raise FinetuneError("step %d gave loss %s and gradient norm %s, which are not finite numbers. "
                                "Nothing was saved. Try a lower --lr; if that does not help, look for NaN or "
                                "infinity in the analyses." % (step + 1, loss_value, grad_norm))
        optimizer.step()
        step += 1
        dt = time.perf_counter() - t0
        train_seconds += dt
        if step > 1:                             # the first step pays one-off start-up costs
            timed_steps += 1
            timed_seconds += dt
        losses.append(loss_value)
        log({"step": step, "loss": loss_value, "lr": lr, "grad_norm": grad_norm, "dt": round(dt, 4)})
        if step % PRINT_EVERY == 0 or step == args.steps:
            speed = "%.2f steps/s" % (timed_steps / timed_seconds) if timed_seconds > 0 else "%.1f s for this step" % dt
            print("step %4d/%d  loss %.6f  lr %.2e  %s" % (step, args.steps, loss_value, lr, speed), flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            t0 = time.perf_counter()
            errors = evaluate(model, data, plan["starts"]["validation"], len(leads), device, scoring, args.batch, stop)
            if errors is None:                   # a signal came during the check: this check does not count
                break
            scores = scores_of(start_values(errors["model"], scoring.std))
            score = selection_score(scores, base_scores, leads)
            is_best = math.isfinite(score) and (best is None or score < best["score"])
            if is_best:
                best = {"step": step, "score": score, "scores": scores, "state": copy_state(model)}
            took = time.perf_counter() - t0
            check_seconds += took
            checks.append({"step": step, "score": score, "best_so_far": is_best, "seconds": round(took, 1)})
            log({"event": "validation", "step": step, "score": score, "best_so_far": is_best})
            print("validation after step %d: score %.4f%s  (%.1f s)"
                  % (step, score, "  <- best so far" if is_best else "", took), flush=True)
    return {"steps_done": step, "best": best, "checks": checks, "losses": losses,
            "steps_per_s": timed_steps / timed_seconds if timed_seconds > 0 else None,
            "train_seconds": train_seconds, "check_seconds": check_seconds}


# ---------------------------------------------------------------- the run

def settings_of(args):
    """The settings of this run, for the report and for tuned.pt."""
    return {
        "init": args.init, "analyses": args.analyses, "stats": args.stats, "out": args.out,
        "test_from": args.test_from, "val_days": args.val_days, "lead_hours": args.lead_hours,
        "steps": args.steps, "lr": args.lr, "warmup": args.warmup, "batch": args.batch, "rollout": args.rollout,
        "eval_every": args.eval_every, "device": args.device, "compile": args.compile, "seed": args.seed,
        "max_starts": args.max_starts, "eval_only": args.eval_only,
        "optimizer": "AdamW", "betas": list(BETAS), "weight_decay": WEIGHT_DECAY, "clip_norm": CLIP_NORM,
        "schedule": "linear warm-up over `warmup` steps to `lr`, then a cosine down to zero at `steps`",
        "loss": "mean over the fed-back steps of the channel-weighted squared error in z-score units",
        "channel_weights": [float(w) for w in wx.t7_channel_weights()],
        "start_hours_utc": list(START_HOURS), "bootstrap_resamples": RESAMPLES,
    }


def check_inputs(data, times, scoring):
    """Stop if the analyses, their times and the scoring grid do not fit together."""
    shape = tuple(data.shape)
    if len(shape) != 4 or shape[1] != wx.N_CHANNELS:
        raise FinetuneError("the analyses have shape %s; they must be (rows, %d, latitudes, longitudes)"
                            % (shape, wx.N_CHANNELS))
    if shape[2:] != tuple(scoring.area.shape):
        raise FinetuneError("the analyses are on a %d x %d grid, but the area weights are for %d x %d"
                            % (shape[2:] + tuple(scoring.area.shape)))
    if len(times) != shape[0]:
        raise FinetuneError("there are %d analyses but %d times" % (shape[0], len(times)))
    if len(scoring.std) != wx.N_CHANNELS or not (np.isfinite(scoring.std).all() and (scoring.std > 0).all()):
        raise FinetuneError("stats.json must give %d standard deviations, all above zero" % wx.N_CHANNELS)


def check_scores(who, scores, must_be_positive):
    """Stop if a model's scores are not usable numbers: its forecasts hold NaN or infinity, or (when the
    scores are about to be divided by) an error is exactly zero."""
    for score in SCORES:
        for value in scores[score["key"]]:
            if not math.isfinite(value) or (must_be_positive and value <= 0):
                raise FinetuneError("%s: the score \"%s\" came out as %r. The forecasts hold NaN or infinity, "
                                    "or the weights and the analyses do not belong together."
                                    % (who, score["name"], value))


def new_report(args, model, device, times, plan, splits):
    """The report with everything that is known before any forecast is run. The rest is filled in on the way."""
    return {
        "what": "weather/finetune.py: the weather model fine-tuned on GFS analyses and scored on held-out weeks",
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "eval-only" if args.eval_only else "finetune",
        "complete": False,
        "stopped_early": False,
        "settings": settings_of(args),
        "device": str(device), "bf16": device.type == "cuda", "torch": torch.__version__,
        "params": sum(p.numel() for p in model.parameters()),
        "analyses": {"rows": len(times), "first": time_text(times[0]), "last": time_text(times[-1])},
        "splits": splits,
        "leads_h": plan["leads_h"],
        "shown_leads_h": plan["shown_h"],
        "units": dict((score["key"], score["unit"]) for score in SCORES),
        "scores": {"validation": {"persistence": None, "base": None, "tuned": None},
                   "test": {"persistence": None, "base": None, "tuned": None}},
        "selection": None,
        "training": None,
        "comparison": None,
        "verdicts": None,
        "overall": None,
        "seconds": {},
    }


def save_tuned(path, state, report):
    """tuned.pt: the weights in the layout of train.py's final.pt, with this run's settings and its summary
    numbers. Everything in it is a tensor or a plain Python value, so torch.load(weights_only=True) reads it."""
    settings, selection, splits = report["settings"], report["selection"], report["splits"]
    summary = {
        "complete": report["complete"], "stopped_early": report["stopped_early"],
        "steps_done": report["training"]["steps_done"], "best_step": selection["best_step"],
        "checked_on_validation": selection["checked_on_validation"], "validation_score": selection["best_score"],
        "train": [splits["train"]["first"], splits["train"]["last"]],
        "validation": [splits["validation"]["first"], splits["validation"]["last"]],
        "test": [splits["test"]["first"], splits["test"]["last"]],
        "test_starts": splits["test"]["starts"],
        "overall": report["overall"], "verdicts": report["verdicts"],
        "test_rows": report["comparison"]["rows"] if report["comparison"] is not None else None,
    }
    save_weights(path, {
        "model": state, "step": selection["best_step"], "seed": settings["seed"], "base": settings["init"],
        "finetune": json_ready(settings), "report": json_ready(summary),
        "stage": "finetune", "complete": report["complete"],
    })


def print_outcome(report, out):
    """The end of a fine-tuning run: the test table, the verdicts in words, and what was written."""
    selection, comparison = report["selection"], report["comparison"]
    steps_done, best_step = report["training"]["steps_done"], selection["best_step"]
    if comparison is not None:
        starts = report["splits"]["test"]["start_times"]
        print()
        print("TEST: base against tuned (the weights after step %d), %d forecast starts on %d days, %s to %s"
              % (best_step, comparison["starts"], comparison["days"], starts[0], starts[-1]))
        print_comparison(comparison["rows"])
        print("In words (the change and its 90% interval in brackets):")
        for name, lead, word, change, interval in verdict_lines(comparison["rows"]):
            print("  %s at %d h: %s (%s; %s)" % (name, lead, word, change, interval))
        print(report["overall"]["sentence"])
        if not selection["beats_base_on_validation"]:
            print("No validation check beat the base model (the best score was %.4f; the base model is 1.0)."
                  % selection["best_score"])
    if report["stopped_early"]:
        if best_step is None:
            print("stopped by a signal before the first training step; no weights were saved")
        elif selection["checked_on_validation"]:
            print("stopped by a signal after step %d. tuned.pt holds the best weights so far (step %d). "
                  "They were NOT scored on the test weeks." % (steps_done, best_step))
        else:
            print("stopped by a signal after step %d, before a validation check had passed. tuned.pt holds "
                  "the weights as they were then. They were NOT scored on validation or on the test weeks."
                  % steps_done)
    if best_step is not None:
        print("wrote %s (the weights after step %d of %d)"
              % (os.path.join(out, "tuned.pt"), best_step, report["settings"]["steps"]))
    print("wrote %s and finetune_report.md (%.0f s in all)"
          % (os.path.join(out, "finetune_report.json"), report["seconds"]["total"]), flush=True)


def finetune(args, model, data, times, scoring, device="cpu", started=None):
    """The whole job, on things already in memory. Writes the files; returns (exit code, the report).

    args      from parse_args()
    model     a torch.nn.Module that maps (batch, 20, lat, lon) z-scores to the state 6 hours later.
              It is trained in place: afterwards it holds the best weights.
    data      (rows, 20, lat, lon) z-scores, a NumPy array or a memory map
    times     one datetime (UTC) per row, increasing
    scoring   a Scoring: the standard deviations, the area weights, the place
    started   time.time() when the command began, so that the timings include loading
    """
    started = time.time() if started is None else started
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True             # the settings weather/train.py trained with
        torch.backends.cudnn.allow_tf32 = True

    check_inputs(data, times, scoring)
    plan = make_plan(args, times)
    splits = describe_plan(plan, times, args)
    leads, shown = plan["leads_h"], plan["shown_h"]
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        raise FinetuneError("cannot make the output folder %s (%s)" % (args.out, one_line(e)))
    if not os.access(args.out, os.W_OK):
        raise FinetuneError("the output folder %s cannot be written to" % args.out)

    model = model.to(device)
    report = new_report(args, model, device, times, plan, splits)
    seconds = report["seconds"]
    seconds["load"] = round(time.time() - started, 1)
    print("%s: a model of %s parameters on %s (%s), PyTorch %s"
          % ("scoring only" if args.eval_only else "fine-tuning", "{:,}".format(report["params"]), device,
             "bf16 autocast" if report["bf16"] else "float32", torch.__version__), flush=True)

    stop = StopFlag()

    def score(name):
        """Score the model as it is now on validation or on test. None if a signal stopped it."""
        return evaluate(model, data, plan["starts"][name], len(leads), device, scoring, args.batch, stop)

    stop.install()
    log_file = None
    try:
        # Step 3: the base model on validation and on test.
        base = {}
        for name in ("validation", "test"):
            t0 = time.time()
            errors = score(name)
            if errors is None:
                print("stopped by a signal while the base model was being scored; nothing was written", flush=True)
                report["stopped_early"] = True
                return EXIT_STOPPED, report
            base[name] = dict((who, start_values(errors[who], scoring.std)) for who in ("model", "persistence"))
            seconds["base_" + name] = round(time.time() - t0, 1)
            report["scores"][name]["base"] = scores_of(base[name]["model"])
            report["scores"][name]["persistence"] = scores_of(base[name]["persistence"])
            print_scores("base model on %s: %d forecast starts (%.1f s)"
                         % (name, len(plan["starts"][name]), seconds["base_" + name]), leads, shown,
                         [("base", report["scores"][name]["base"]),
                          ("persistence", report["scores"][name]["persistence"])])
            check_scores("the base model on %s" % name, report["scores"][name]["base"], not args.eval_only)

        if args.eval_only:
            report["complete"] = True
            seconds["total"] = round(time.time() - started, 1)
            write_report(args.out, report)
            print("wrote %s and finetune_report.md (%.0f s in all)"
                  % (os.path.join(args.out, "finetune_report.json"), seconds["total"]), flush=True)
            return 0, report

        # Step 4: train, and keep the weights that do best on validation.
        log_file = open(os.path.join(args.out, "log.jsonl"), "a")

        def log(obj):
            log_file.write(json.dumps(json_ready(obj)) + "\n")
            log_file.flush()

        log({"event": "start", "time": report["written"], "settings": report["settings"], "device": str(device),
             "torch": torch.__version__, "params": report["params"], "samples": len(plan["samples"])})
        print("training: %d steps of %d samples, %d fed-back steps, peak learning rate %g, %d training samples"
              % (args.steps, args.batch, args.rollout, args.lr, len(plan["samples"])))
        print("validation score: the mean of RMSE / the base model's RMSE over Z500, T850 and T2M at %s h. "
              "The base model is 1.0; lower is better."
              % ", ".join(str(lead) for lead in SELECT_LEADS_H if lead in leads), flush=True)
        result = train(args, model, data, plan, scoring, device, report["scores"]["validation"]["base"], stop, log)
        seconds["training"] = round(result["train_seconds"], 1)
        seconds["validation_checks"] = round(result["check_seconds"], 1)
        steps_done, best, losses = result["steps_done"], result["best"], result["losses"]
        stopped = stop()
        checked = best is not None               # False: no validation check has passed
        if not checked and not stopped:
            raise FinetuneError("no validation check gave finite scores: after training, the model's forecasts "
                                "hold NaN or infinity. Nothing was saved. Try a lower --lr.")
        if not checked and steps_done > 0:
            # Stopped before a validation check passed: the weights as they are now are all there is.
            best = {"step": steps_done, "score": None, "scores": None, "state": copy_state(model)}
        report["training"] = {
            "steps_asked": args.steps, "steps_done": steps_done, "samples": len(plan["samples"]),
            "steps_per_s": result["steps_per_s"],
            "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
            "mean_loss_first_20": float(np.mean(losses[:20])) if losses else None,
            "mean_loss_last_20": float(np.mean(losses[-20:])) if losses else None,
        }
        report["selection"] = {
            "rule": "lowest mean of RMSE / the base model's RMSE on validation, over Z500, T850 and T2M at the "
                    "leads below; the base model is 1.0",
            "leads_h": [lead for lead in SELECT_LEADS_H if lead in leads],
            "checks": result["checks"],
            "best_step": best["step"] if best is not None else None,
            "best_score": best["score"] if best is not None else None,
            "checked_on_validation": checked,
            "beats_base_on_validation": bool(checked and best["score"] < 1.0),
        }
        if checked:
            report["scores"]["validation"]["tuned"] = best["scores"]

        # Step 5: the best weights on TEST, once.
        if not stopped:
            model.load_state_dict(best["state"])
            t0 = time.time()
            errors = score("test")
            if errors is None:
                stopped = True
            else:
                tuned = start_values(errors["model"], scoring.std)
                seconds["tuned_test"] = round(time.time() - t0, 1)
                report["scores"]["test"]["tuned"] = scores_of(tuned)
                check_scores("the tuned model on test", report["scores"]["test"]["tuned"], False)
                t0 = time.time()
                days = [str(times[row].date()) for row in plan["starts"]["test"]]       # whole days are drawn
                rows = compare(base["test"]["model"], tuned, base["test"]["persistence"], days, leads, shown,
                               args.seed)
                seconds["bootstrap"] = round(time.time() - t0, 1)
                report["comparison"] = {
                    "split": "test", "starts": len(days), "days": len(set(days)), "resamples": RESAMPLES,
                    "seed": args.seed,
                    "interval": "5th and 95th percentiles of the change in percent, from a paired bootstrap "
                                "over forecast starts grouped by calendar day",
                    "rows": rows,
                }
                report["verdicts"], report["overall"] = verdicts_of(rows)
                report["complete"] = True
        report["stopped_early"] = stopped
        seconds["total"] = round(time.time() - started, 1)

        # Step 6: the files.
        if best is not None:
            save_tuned(os.path.join(args.out, "tuned.pt"), best["state"], report)
        write_report(args.out, report)
        log({"event": "end", "step": steps_done, "complete": report["complete"], "stopped_early": stopped,
             "best_step": report["selection"]["best_step"], "steps_per_s": result["steps_per_s"]})
        print_outcome(report, args.out)
        return (EXIT_STOPPED if stopped else 0), report
    finally:
        stop.restore()
        if log_file is not None:
            log_file.close()


def load_analyses(folder):
    """DIR/analyses.npy, memory-mapped, and the time of each row from DIR/analyses.json (weather/gfs_archive.py)."""
    npy, info = os.path.join(folder, "analyses.npy"), os.path.join(folder, "analyses.json")
    for path in (npy, info):
        if not os.path.isfile(path):
            raise FinetuneError("%s is missing. --analyses must be a folder written by weather/gfs_archive.py "
                                "(analyses.npy and analyses.json)." % path)
    try:
        data = np.load(npy, mmap_mode="r")
    except (OSError, ValueError) as e:
        raise FinetuneError("cannot open %s (%s: %s)" % (npy, type(e).__name__, one_line(e)))
    want = (wx.N_CHANNELS, wx.N_LAT, wx.N_LON)
    if data.ndim != 4 or data.shape[1:] != want or data.dtype != np.float16:
        raise FinetuneError("%s has shape %s and type %s; it must be (rows, %d, %d, %d) float16"
                            % ((npy, tuple(data.shape), data.dtype) + want))
    try:
        with open(info) as f:
            texts = json.load(f)["times"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise FinetuneError("cannot read the times from %s (%s: %s)" % (info, type(e).__name__, one_line(e)))
    if not isinstance(texts, list) or len(texts) != len(data):
        raise FinetuneError("%s lists %s times, but analyses.npy has %d rows. Rebuild both with "
                            "weather/gfs_archive.py --stack-only."
                            % (info, len(texts) if isinstance(texts, list) else "no", len(data)))
    return data, [parse_time(text, "a time in %s" % info) for text in texts]


def load_std(args):
    """The standard deviations from stats.json. Looks in --stats; without it, in the analyses folder and then
    in $WX_DATA. Returns (std, the folder used)."""
    folders = [args.stats] if args.stats else [args.analyses, wx.data_dir(None)]
    for folder in folders:
        if os.path.isfile(os.path.join(folder, "stats.json")):
            try:
                _, std = wx.read_stats(folder)
            except (OSError, ValueError, KeyError) as e:
                raise FinetuneError("cannot use stats.json in %s (%s: %s)" % (folder, type(e).__name__, one_line(e)))
            return std, folder
    raise FinetuneError("there is no stats.json in %s. Pass --stats DIR: the folder with the stats.json that "
                        "the analyses were made with." % " or ".join(folders))


def run(args, model=None):
    """Open what the arguments name, then do the work. Returns (exit code, the report).
    `model` is for the tests: it stands in for the AFNO, and --init is then not read."""
    started = time.time()
    torch.set_num_threads(min(CPU_THREADS, torch.get_num_threads()))
    data, times = load_analyses(args.analyses)
    std, args.stats = load_std(args)
    scoring = Scoring(std)
    check_inputs(data, times, scoring)
    make_plan(args, times)                       # a split that is too small stops the run before the model loads
    device = pick_device(args.device)
    if model is None:
        model = load_base_model(args.init)
    return finetune(args, model, data, times, scoring, device, started)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def whole_number(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def positive_float(text):
    value = float(text)
    if not (value > 0 and math.isfinite(value)):
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def parse_args(argv=None):
    """The command line's options, with their numbers checked."""
    p = argparse.ArgumentParser(description="Fine-tune the weather model on GFS analyses and measure, on "
                                            "held-out weeks, whether that helped.")
    p.add_argument("--init", required=True, metavar="WEIGHTS.pt",
                   help="the weights to start from: a final.pt written by weather/train.py, or a tuned.pt")
    p.add_argument("--analyses", required=True, metavar="DIR",
                   help="the folder weather/gfs_archive.py wrote: analyses.npy and analyses.json")
    p.add_argument("--out", required=True, metavar="OUTDIR",
                   help="folder for tuned.pt, the report and log.jsonl (made if missing)")
    p.add_argument("--stats", default=None, metavar="DIR",
                   help="folder with stats.json (default: the analyses folder, then $WX_DATA)")
    p.add_argument("--test-from", default="2026-10-01T00", metavar="YYYY-MM-DDTHH",
                   help="the test weeks are every analysis from this time on (default 2026-10-01T00)")
    p.add_argument("--val-days", type=positive_int, default=14, metavar="N",
                   help="validation is the N days before the test weeks (default 14)")
    p.add_argument("--steps", type=positive_int, default=400, metavar="N", help="training steps (default 400)")
    p.add_argument("--lr", type=positive_float, default=3e-5, help="peak learning rate (default 3e-5)")
    p.add_argument("--warmup", type=whole_number, default=20, metavar="N",
                   help="steps over which the learning rate climbs to its peak (default 20)")
    p.add_argument("--batch", type=positive_int, default=16, metavar="N",
                   help="samples per training step, and forecasts scored together (default 16)")
    p.add_argument("--rollout", type=positive_int, default=2, metavar="N",
                   help="6-hour steps the model is fed back through in training (default 2)")
    p.add_argument("--eval-every", type=positive_int, default=100, metavar="N",
                   help="score on validation every N steps, and at the end (default 100)")
    p.add_argument("--lead-hours", type=positive_int, default=72, metavar="HOURS",
                   help="how far each scored forecast runs: a multiple of 6, at least 24 (default 72)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="where the model runs (default auto: the GPU if there is one)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model for the training steps")
    p.add_argument("--seed", type=whole_number, default=1,
                   help="seed for the order of the training samples and for the bootstrap (default 1)")
    p.add_argument("--max-starts", type=positive_int, default=None, metavar="N",
                   help="score only the first N forecast starts of validation and of test (a quick check)")
    p.add_argument("--eval-only", action="store_true",
                   help="train nothing: score the --init weights and write the report")
    args = p.parse_args(argv)
    if args.lead_hours % wx.STEP_HOURS or args.lead_hours < 24:
        p.error("--lead-hours must be a multiple of 6, and 24 or more")
    if not args.eval_only and args.warmup >= args.steps:
        p.error("--warmup (%d) must be less than --steps (%d); for a short run add --warmup 0"
                % (args.warmup, args.steps))
    return args


def main(argv=None, model=None):
    """The command line. `model` is for the tests: a small torch.nn.Module used in place of the AFNO
    (it is trained in place)."""
    args = parse_args(argv)
    try:
        code, _ = run(args, model)
    except FinetuneError as e:
        print("finetune.py: FAILED: %s" % e, file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
