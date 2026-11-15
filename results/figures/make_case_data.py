#!/usr/bin/env python3
"""The January 2019 cold wave, forecast by the model that was trained at the hackathon.

Runs the event model (weights/afno_event.pt, trained on ERA5 1979-2017) from the ERA5 state of
2019-01-27T06 for 18 six-hour steps and writes the forecast and the ERA5 truth for six fields
(2 m temperature, 850 hPa temperature, sea-level pressure, 500 hPa geopotential, 10 m wind u and v)
to one .npz file. January 2019 is not in the training data. Usage: make_case_data.py BUNDLE OUT.npz

Run on Nov 15, 2026 inside the project's container on a DGX Spark, on the CPU (18 steps in 5.4 s), with the repo
at /repo. BUNDLE is the event bundle (weather/make_bundle.py): the weights, stats.json and the ERA5 states of
2019-01-26T00 to 2019-01-31T18. The 2 m temperature part of the output is in data/cold-wave-2019-t2m.npz."""
import sys, json, time
import numpy as np
import torch
sys.path.insert(0, "/repo/weather")
import wx_common as wx
import forecast_core as fc

bundle, out = sys.argv[1], sys.argv[2]
meta = json.load(open(bundle + "/bundle.json"))
mean, std = wx.read_stats(bundle)
states = np.load(bundle + "/" + meta["states"]["file"]).astype(np.float32)          # (24, 20, 120, 240) z-scores
first = np.datetime64(meta["states"]["first_time"], "h")
init = np.datetime64("2019-01-27T06", "h")
i0 = int((init - first) / np.timedelta64(6, "h"))
n = states.shape[0] - 1 - i0
keep = {"t2m": wx.CH_T2M, "t850": wx.CH_T850, "msl": wx.CH_MSL, "z500": wx.CH_Z500, "u10": wx.CH_U10, "v10": wx.CH_V10}
model = fc.load_model(bundle + "/weights/afno_event.pt", "cpu")
x = torch.from_numpy(states[i0][None].copy())
fcst = {k: [] for k in keep}
t0 = time.perf_counter()
with torch.inference_mode():
    for step in range(n):
        x = model(x).float()
        z = x[0].numpy()
        for k, c in keep.items():
            fcst[k].append(z[c] * std[c] + mean[c])
secs = time.perf_counter() - t0
truth = {k: [states[i0 + 1 + s][c] * std[c] + mean[c] for s in range(n)] for k, c in keep.items()}
start = {k: states[i0][c] * std[c] + mean[c] for k, c in keep.items()}
arrays = {}
for k in keep:
    arrays["fcst_" + k] = np.asarray(fcst[k], np.float32)
    arrays["truth_" + k] = np.asarray(truth[k], np.float32)
    arrays["start_" + k] = np.asarray(start[k], np.float32)
np.savez_compressed(out, lats=wx.LATS, lons=wx.LONS, init=str(init), step_hours=6, steps=n,
                    weights_sha256=meta["weights"]["sha256"], **arrays)
w = np.cos(np.deg2rad(wx.LATS))[:, None] * np.ones((1, wx.N_LON))
for lead in (24, 48, 72, 96):
    s = lead // 6 - 1
    if s < n:
        e = np.sqrt((w * (arrays["fcst_t850"][s] - arrays["truth_t850"][s]) ** 2).sum() / w.sum())
        p = np.sqrt((w * (arrays["start_t850"] - arrays["truth_t850"][s]) ** 2).sum() / w.sum())
        print("T850 global RMSE at +%d h: forecast %.2f K, 'same as at the start' %.2f K" % (lead, e, p))
print("wrote", out, "| init", init, "| steps", n, "| %.1f s on the CPU for %d steps" % (secs, n))
