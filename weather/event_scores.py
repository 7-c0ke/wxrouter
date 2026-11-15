#!/usr/bin/env python3
"""Yes/no scores of the weather model before and after the fine-tune: accuracy, precision, recall, F1.

The model forecasts fields of numbers, so its own score is an error in physical units: the RMSE in
weather/finetune.py and its report. Accuracy, precision, recall and F1 are the scores of a yes/no
classifier. A forecast has them only after it is turned into a yes/no question. This script asks two such
questions of the SAME test forecasts that the fine-tune report scores, for the base weights and the tuned
ones:

  frost        will the 2-m temperature at this grid point be below 0 C at the valid time?
  big change   will the 2-m temperature at this grid point be 5 C or more away from its value in the
               starting analysis?

Each grid point of each forecast is one case, weighted by the area it stands for (cos latitude). The truth
is the NOAA GFS analysis at the valid time. "persistence" is the forecast "nothing changes": for "big
change" it never says yes, so its recall there is 0.

The two questions and their thresholds were fixed before the first run and were not changed after looking
at a result. The test forecasts start on days that the fine-tune never trained on and never chose its
setting on (the same split as weather/finetune.py: --test-from, --val-days, --lead-hours).

    python weather/event_scores.py --base afno_event.pt --tuned tuned.pt --analyses GFS_YEAR --stats BUNDLE --out OUT

Writes OUT/event_scores.json and OUT/event_scores.md. The interval of each change is from a paired bootstrap
over the days on which the forecasts start (the 5th and 95th percentiles of 1,000 draws), as in the report
of weather/finetune.py.

Written at the hackathon (Nov 14, 2026).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import finetune as ft  # noqa: E402
import wx_common as wx  # noqa: E402

FREEZING_K = 273.15
CHANGE_K = 5.0
RESAMPLES = 1000
QUESTIONS = [("frost", "2-m temperature below 0 C"),
             ("big_change", "2-m temperature 5 C or more away from the starting analysis")]
WHO = ("base", "tuned", "persistence")


def answers(t2m_k, start_k):
    """The yes/no answer to each question, for 2-m temperatures in kelvin: {question: bool tensor}."""
    return {"frost": t2m_k < FREEZING_K, "big_change": (t2m_k - start_k).abs() >= CHANGE_K}


def counts(said, truth, area):
    """Area-weighted hits, false alarms, misses and correct no's per forecast: a (batch, 4) float64 tensor
    in the order tp, fp, fn, tn. said, truth: bool (batch, lat, lon); area: (lat, lon) or (lat, 1) weights."""
    w = area.double()
    parts = [(said & truth), (said & ~truth), (~said & truth), (~said & ~truth)]
    return torch.stack([(p.double() * w).sum(dim=(-2, -1)) for p in parts], dim=1)


def scores(c):
    """accuracy, precision, recall and F1 from summed counts (tp, fp, fn, tn). A score whose denominator is
    0 is None: precision when the forecast never says yes, recall when the event never happens."""
    tp, fp, fn, tn = (float(v) for v in c)
    total = tp + fp + fn + tn
    return {"accuracy": (tp + tn) / total if total else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "base_rate": (tp + fn) / total if total else None,
            "wrong": (fp + fn) / total if total else None}


def roll(model, data, starts, lead_steps, device, mean_k, std_k, batch=16):
    """Run the model from every start row. Returns {step: float32 array (starts, lat, lon)}: the forecast
    2-m temperature in kelvin at each step in lead_steps. The same stepping as finetune.evaluate()."""
    device = torch.device(device)
    on_cuda = device.type == "cuda"
    out = {k: [] for k in lead_steps}
    model = model.to(device).eval()
    with torch.inference_mode():
        for a in range(0, len(starts), batch):
            rows = starts[a:a + batch]
            state = torch.from_numpy(np.stack([data[r] for r in rows])).to(device).float()
            for k in range(1, max(lead_steps) + 1):
                with ft.autocast(on_cuda):
                    state = model(state)
                state = state.float()
                if k in out:
                    out[k].append((state[:, wx.CH_T2M].double() * std_k + mean_k).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def interval(base_c, tuned_c, days, key, seed=1, resamples=RESAMPLES):
    """The 5th and 95th percentiles of the change of one score (tuned - base, in percent of base) when the
    start days are drawn again with replacement. base_c, tuned_c: (starts, 4) counts."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(days))
    where = {d: np.array([i for i, x in enumerate(days) if x == d]) for d in uniq}
    changes = []
    for _ in range(resamples):
        pick = np.concatenate([where[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        b, t = scores(base_c[pick].sum(axis=0))[key], scores(tuned_c[pick].sum(axis=0))[key]
        if b and t is not None:
            changes.append(100.0 * (t - b) / b)
    if not changes:
        return None
    return [float(np.percentile(changes, 5)), float(np.percentile(changes, 95))]


def run(args):
    started = time.time()
    data, times = ft.load_analyses(args.analyses)
    mean, std = wx.read_stats(args.stats)
    mean_k, std_k = float(mean[wx.CH_T2M]), float(std[wx.CH_T2M])
    test_from = ft.parse_time(args.test_from, "--test-from")
    n_steps = args.lead_hours // wx.STEP_HOURS
    rows = ft.split_rows(times, test_from, args.val_days, args.lead_hours)
    starts = ft.chain_starts(times, rows["test"][0], rows["test"][1], n_steps, ft.START_HOURS)
    leads = [int(x) for x in args.leads.split(",")]
    lead_steps = [h // wx.STEP_HOURS for h in leads]
    device = ft.pick_device(args.device)
    print("test forecasts: %d, started %s to %s, on %s" % (len(starts), ft.time_text(times[starts[0]]),
                                                         ft.time_text(times[starts[-1]]), device), flush=True)
    area = torch.as_tensor(np.broadcast_to(wx.area_weights(), (wx.N_LAT, wx.N_LON)).copy(), dtype=torch.float64)
    forecast = {}
    for who, path in (("base", args.base), ("tuned", args.tuned)):
        forecast[who] = roll(ft.load_base_model(path), data, starts, lead_steps, device, mean_k, std_k)
    kelvin = lambda rws: np.stack([data[r][wx.CH_T2M] for r in rws]).astype(np.float64) * std_k + mean_k   # noqa: E731
    start_k = torch.from_numpy(kelvin(starts))
    days = [times[r].date().isoformat() for r in starts]

    result = {"what": "area-weighted yes/no scores of 2-m temperature forecasts on the test starts of weather/finetune.py",
              "base": os.path.abspath(args.base), "tuned": os.path.abspath(args.tuned),
              "test_starts": len(starts), "first_start": ft.time_text(times[starts[0]]),
              "last_start": ft.time_text(times[starts[-1]]), "start_days": len(set(days)),
              "cases_per_forecast": wx.N_LAT * wx.N_LON, "questions": dict(QUESTIONS),
              "interval": "5th and 95th percentiles of %d paired bootstrap draws over the start days" % RESAMPLES,
              "leads_h": leads, "rmse_t2m_k": {}, "scores": {}}
    for hours, k in zip(leads, lead_steps):
        truth_k = torch.from_numpy(kelvin([r + k for r in starts]))
        said_k = {"base": torch.from_numpy(forecast["base"][k]).double(),
                  "tuned": torch.from_numpy(forecast["tuned"][k]).double(), "persistence": start_k}
        result["rmse_t2m_k"][str(hours)] = {
            who: float(torch.sqrt((((said_k[who] - truth_k) ** 2) * area).mean())) for who in WHO}
        truth = answers(truth_k, start_k)
        for q, _ in QUESTIONS:
            per = {who: counts(answers(said_k[who], start_k)[q], truth[q], area).numpy() for who in WHO}
            entry = {who: scores(per[who].sum(axis=0)) for who in WHO}
            entry["change_pct"], entry["change_interval_pct"] = {}, {}
            for key in ("accuracy", "precision", "recall", "f1", "wrong"):
                b, t = entry["base"][key], entry["tuned"][key]
                entry["change_pct"][key] = 100.0 * (t - b) / b if b and t is not None else None
                entry["change_interval_pct"][key] = interval(per["base"], per["tuned"], days, key)
            result["scores"].setdefault(q, {})[str(hours)] = entry
    result["seconds"] = round(time.time() - started, 1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "event_scores.json"), "w") as f:
        json.dump(result, f, indent=1)
    text = markdown(result)
    with open(os.path.join(args.out, "event_scores.md"), "w") as f:
        f.write(text)
    print(text)
    return result


def markdown(r):
    pct = lambda v: "n/a" if v is None else "%.2f%%" % (100 * v)              # noqa: E731
    chg = lambda v: "n/a" if v is None else "%+.1f%%" % v                     # noqa: E731
    span = lambda p: "" if not p else " (%+.1f%% to %+.1f%%)" % tuple(p)      # noqa: E731
    lines = ["# Yes/no scores before and after the fine-tune", "",
             "%d test forecasts, started %s to %s (%d days); %s grid points each, weighted by area; truth: the NOAA GFS "
             "analysis at the valid time. Made by weather/event_scores.py." % (
                 r["test_starts"], r["first_start"], r["last_start"], r["start_days"], format(r["cases_per_forecast"], ",")), "",
             "The model's own score is the RMSE (the fine-tune report). These scores answer a yes/no question about the "
             "same forecasts. In brackets: the 90% interval of the change from a paired bootstrap over the start days.", ""]
    for q, words in QUESTIONS:
        for hours in r["leads_h"]:
            e = r["scores"][q][str(hours)]
            lines += ["## %s, %d hours ahead" % (words, hours), "",
                      "The event is true in %s of the cases." % pct(e["base"]["base_rate"]), "",
                      "| score | base model | tuned model | change | \"nothing changes\" |", "|---|---:|---:|---|---:|"]
            for key, name in (("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
                              ("wrong", "wrong answers (1 - accuracy)")):
                lines.append("| %s | %s | %s | %s%s | %s |" % (
                    name, pct(e["base"][key]), pct(e["tuned"][key]), chg(e["change_pct"][key]),
                    span(e["change_interval_pct"][key]), pct(e["persistence"][key])))
            lines.append("")
    lines += ["## Check against the fine-tune report", "",
              "| 2-m temperature RMSE (K) | base | tuned | \"nothing changes\" |", "|---|---:|---:|---:|"]
    for hours in r["leads_h"]:
        v = r["rmse_t2m_k"][str(hours)]
        lines.append("| %d h | %.3f | %.3f | %.3f |" % (hours, v["base"], v["tuned"], v["persistence"]))
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, metavar="WEIGHTS.pt")
    p.add_argument("--tuned", required=True, metavar="WEIGHTS.pt")
    p.add_argument("--analyses", required=True, metavar="DIR", help="the folder of weather/gfs_archive.py")
    p.add_argument("--stats", required=True, metavar="DIR", help="the folder with stats.json")
    p.add_argument("--out", required=True, metavar="OUTDIR")
    p.add_argument("--test-from", default="2026-10-01T00")
    p.add_argument("--val-days", type=int, default=14)
    p.add_argument("--lead-hours", type=int, default=72)
    p.add_argument("--leads", default="24,72", help="hours ahead to score, comma-separated (default 24,72)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args(argv)
    try:
        run(args)
    except ft.FinetuneError as e:
        print("event_scores: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
