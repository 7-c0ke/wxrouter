#!/usr/bin/env python3
"""Figures for the README and the one-page summary, drawn from stored result files.

    python make_figures.py DATA_DIR OUT_DIR

DATA_DIR (results/figures/data) holds cold-wave-2019-t2m.npz (the 2 m temperature part of what
make_case_data.py wrote with the event model), scores_overnight.json (written by weather/score.py for the
same model on Nov 14, 2026) and four Natural Earth outline files (public domain). The two bar charts use
numbers from results/event-boxes-2026-11-14.md (sections 8 and 10). Needs numpy and matplotlib."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

DATA, OUT = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})
# cold = blue ramp, 0 degrees C = neutral grey, warm = orange ramp (two hues and a neutral midpoint)
COLD_WARM = LinearSegmentedColormap.from_list("cold_warm", [
    (0.00, "#0d366b"), (0.20, "#1c5cab"), (0.36, "#3987e5"), (0.50, "#9ec5f4"), (0.60, "#dfe5ec"),
    (0.611, "#ecebe7"), (0.70, "#f6c7a8"), (0.85, "#eb6834"), (1.00, "#a8401a")])


def lines_of(name):
    out = []
    for f in json.load(open(f"{DATA}/{name}"))["features"]:
        g = f["geometry"]
        if g is None:
            continue
        t, c = g["type"], g["coordinates"]
        parts = [c] if t == "LineString" else c if t in ("MultiLineString", "Polygon") else [r for p in c for r in p] if t == "MultiPolygon" else []
        out += [np.asarray(p, float)[:, :2] for p in parts]
    return out


def interp(field, lats, lons, lat, lon):
    """Value at a place, from the four grid points around it."""
    lon = lon % 360
    j = (lon - lons[0]) / 1.5; i = (lat - lats[0]) / 1.5
    i0, j0 = int(np.floor(i)), int(np.floor(j)); di, dj = i - i0, j - j0
    j1 = (j0 + 1) % len(lons)
    return ((1 - di) * (1 - dj) * field[..., i0, j0] + (1 - di) * dj * field[..., i0, j1]
            + di * (1 - dj) * field[..., i0 + 1, j0] + di * dj * field[..., i0 + 1, j1])


def cold_wave(compact=False):
    """compact=True: the same panels with no title, subtitle or footnote, for a page that has its own text."""
    d = np.load(f"{DATA}/cold-wave-2019-t2m.npz")
    lats, lons = d["lats"], d["lons"]
    s72 = 72 // 6 - 1
    truth, fcst = d["truth_t2m"][s72] - 273.15, d["fcst_t2m"][s72] - 273.15
    west, east, south, north = -128.0, -62.0, 24.0, 62.0
    order = np.argsort(((lons + 180) % 360) - 180)
    lon180 = (((lons + 180) % 360) - 180)[order]
    jj = (lon180 >= west - 1.5) & (lon180 <= east + 1.5); ii = (lats >= south - 1.5) & (lats <= north + 1.5)
    levels = np.arange(-44, 29, 4)
    norm = BoundaryNorm(levels, COLD_WARM.N)
    coast, states, lakes = lines_of("ne_110m_coastline.geojson"), lines_of("ne_50m_admin1_borders_wide.geojson"), lines_of("ne_50m_lakes_greatlakes.geojson")
    chi = (41.88, -87.63)
    fig = plt.figure(figsize=(10.4, 3.45 if compact else 5.3), dpi=200)
    y0, h, yc = (0.235, 0.655, 0.09) if compact else (0.17, 0.60, 0.105)
    if not compact:
      fig.text(0.03, 0.94, "Three days ahead: the polar-vortex cold wave of January 2019", fontsize=15.5, fontweight="bold", color=INK)
      fig.text(0.03, 0.845, "Temperature at 2 m on 30 Jan 2019, 06 UTC. The model was trained from scratch during the hackathon, in 5 h 14 min\n"
               "on a team member's own NVIDIA DGX Spark, on ERA5 weather from 1979 to 2017. It never saw 2019.", fontsize=10, color=INK2, linespacing=1.45)
    axes = [fig.add_axes([0.03, y0, 0.305, h]), fig.add_axes([0.35, y0, 0.305, h])]
    for ax, field, title in ((axes[0], truth, "30 Jan 2019, as it happened (ERA5)" if compact else "What happened (ERA5 reanalysis)"), (axes[1], fcst, "Our forecast, made 72 hours earlier")):
        sub = field[:, order][np.ix_(ii, jj)]
        cs = ax.contourf(lon180[jj], lats[ii], sub, levels=levels, cmap=COLD_WARM, norm=norm, extend="both")
        ax.contour(lon180[jj], lats[ii], sub, levels=[0], colors=[INK], linewidths=0.9, linestyles="--")
        for p in states: ax.plot(p[:, 0], p[:, 1], color="#ffffff", lw=0.35, alpha=0.75)
        for p in coast: ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.7)
        for p in lakes: ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.5)
        v = float(interp(field, lats, lons, *chi))
        ax.plot([chi[1]], [chi[0]], "o", ms=7, mfc="#ffffff", mec=INK, mew=1.5, zorder=5)
        ax.annotate(f"Chicago  {v:.0f} °C".replace("-", "\u2212"), (chi[1], chi[0]), xytext=(-8, -21), textcoords="offset points", ha="right", fontsize=10, fontweight="bold",
                    color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec=INK, lw=0.8), zorder=6)
        ax.set_xlim(west, east); ax.set_ylim(south, north); ax.set_aspect("auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(INK2); sp.set_linewidth(0.6)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=6)
    cax = fig.add_axes([0.10, yc, 0.42, 0.032 if compact else 0.024])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=np.arange(-40, 29, 8))
    cb.ax.tick_params(labelsize=8.5, length=2); cb.outline.set_visible(False)
    cb.ax.set_xticklabels([f"{t}".replace("-", "\u2212") for t in np.arange(-40, 29, 8)])
    fig.text(0.53, yc - 0.002, "°C at 2 m  (dashed: 0 °C)" if compact else "°C  (dashed: 0 °C)", fontsize=8.5, color=INK2)

    ax = fig.add_axes([0.725, y0, 0.255, h])
    hours = 6 * (1 + np.arange(int(d["steps"])))
    t_truth = np.concatenate([[float(interp(d["start_t2m"], lats, lons, *chi))], interp(d["truth_t2m"], lats, lons, *chi)]) - 273.15
    t_fcst = np.concatenate([[t_truth[0] + 273.15], interp(d["fcst_t2m"], lats, lons, *chi)]) - 273.15
    hh = np.concatenate([[0], hours])
    ax.axvline(72, color=GRID, lw=1.2)
    ax.plot(hh, t_truth, color=INK, lw=2, marker="o", ms=3.2, label="what happened")
    ax.plot(hh, t_fcst, color=BLUE, lw=2, marker="o", ms=3.2, label="our forecast")
    ax.set_ylim(-31, 3)
    ax.text(73.5, 1.6, "+72 h", color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, hh[-1] + 2); ax.set_xticks(np.arange(0, 109, 24)); ax.set_xlabel("hours after the start (27 Jan, 06 UTC)", fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}".replace("-", "\u2212")); ax.set_ylabel("°C", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8.8, handlelength=1.6, borderaxespad=0.2)
    ax.set_title("Chicago, every 6 hours", fontsize=11, color=INK, loc="left", pad=6)
    err = np.abs(t_fcst[1:] - t_truth[1:])
    if not compact:
      fig.text(0.03, 0.022, f"Chicago is the value of the 1.5° grid at 41.9° N, 87.6° W. Over the {int(hh[-1])} hours the forecast missed by {err.mean():.1f} °C on average ({err.max():.1f} °C at most). "
               "One forecast of one event:\nthe scores over a whole test year are in skill-by-lead-time-2020.png. Made by results/figures/make_case_data.py and make_figures.py.", fontsize=7.8, color=INK2, va="bottom", linespacing=1.4)
    fig.savefig(f"{OUT}/cold-wave-2019-compact.png" if compact else f"{OUT}/cold-wave-2019.png")
    plt.close(fig)
    if compact:
        return
    print("cold wave: Chicago at +72 h: forecast %.1f, ERA5 %.1f; start %.1f; coldest ERA5 %.1f at +%d h, forecast then %.1f; mean abs miss %.2f, max %.2f" % (
        t_fcst[12], t_truth[12], t_truth[0], t_truth.min(), hh[int(t_truth.argmin())], t_fcst[int(t_truth.argmin())], err.mean(), err.max()))


def skill():
    s = json.load(open(f"{DATA}/scores_overnight.json"))
    y = s["years"]["2020"]; leads = np.asarray(y["leads_h"] if len(y["leads_h"]) == 20 else s["leads_h"])
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.4), dpi=200)
    fig.subplots_adjust(left=0.06, right=0.905, top=0.655, bottom=0.13, wspace=0.62)
    fig.text(0.03, 0.93, "Forecast error by lead time, on a year the model never saw", fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.03, 0.765, f"2020: {y['n_starts']} forecasts, started at 00 and 12 UTC every second day, scored against ERA5 over the whole globe\n"
             "(root-mean-square error, weighted by area). Lower is better. The model was trained and scored on one GB10,\n"
             "a team member's own NVIDIA DGX Spark. One training run.", fontsize=10, color=INK2, linespacing=1.45)
    names = {"z500": ("Pressure pattern (500 hPa)", "m²/s²"), "t850": ("Temperature aloft (850 hPa)", "K"), "t2m": ("Temperature at 2 m", "K")}
    for ax, v in zip(axes, ("z500", "t850", "t2m")):
        r = y["rmse"]
        get = lambda kind: np.asarray([r[kind][v][str(h)] if isinstance(r[kind][v], dict) else r[kind][v][i] for i, h in enumerate(leads)])
        m, p, c = get("model"), get("persistence"), get("climatology")
        ax.plot(leads, c, color=AQUA, lw=2); ax.plot(leads, p, color=ORANGE, lw=2); ax.plot(leads, m, color=BLUE, lw=2.4)
        ax.set_xlim(0, 124); ax.set_ylim(0, max(p.max(), c.max()) * 1.12); ax.set_xticks([24, 48, 72, 96, 120])
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        top = ax.get_ylim()[1]
        labels = sorted([[m[-1], "this model", INK, "bold"], [p[-1], "“same as\nnow”", INK2, "normal"], [c[-1], "long-term\naverage", INK2, "normal"]])
        gap = 0.135 * top                              # keep the three labels apart, in the order of their lines
        for k in (1, 2):
            labels[k][0] = max(labels[k][0], labels[k - 1][0] + gap)
        for yl, text, col, weight in labels:
            ax.text(126, yl, text, color=col, fontsize=8.8, va="center", fontweight=weight, clip_on=False, linespacing=1.1)
        for yy_, col in ((m[-1], BLUE), (p[-1], ORANGE), (c[-1], AQUA)):
            ax.plot([120], [yy_], "o", ms=5, mfc=col, mec=SURFACE, mew=1.2, clip_on=False, zorder=5)
        i72 = list(leads).index(72)
        ax.plot([72], [m[i72]], "o", ms=6, mfc=BLUE, mec=SURFACE, mew=1.5)
        unit = names[v][1]
        ax.annotate(f"3 days:\n{m[i72]:.0f} {unit}" if v == "z500" else f"3 days:\n{m[i72]:.2f} {unit}", (72, m[i72]), xytext=(14, -30), textcoords="offset points",
                    fontsize=8.8, color=INK, ha="left", linespacing=1.15, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        ax.set_title(names[v][0], fontsize=10.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("hours ahead", fontsize=9); ax.set_ylabel(unit, fontsize=9)
        print(f"skill {v}: 24 h {m[3]:.3f} vs persistence {p[3]:.3f} vs climatology {c[3]:.3f}; 72 h {m[i72]:.3f} vs {p[i72]:.3f} vs {c[i72]:.3f}; 120 h {m[-1]:.3f} vs {p[-1]:.3f} vs {c[-1]:.3f}")
    fig.savefig(f"{OUT}/skill-by-lead-time-2020.png")


def router():
    rules = [("static: no routing,\nall requests to the box that trains", 3170), ("adaptive", 1161), ("round robin", 697), ("balanced", 411), ("least loaded", 319)]
    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=200)
    fig.subplots_adjust(left=0.33, right=0.93, top=0.68, bottom=0.17)
    fig.text(0.035, 0.915, "Waiting time while one box fine-tunes", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.035, 0.80, "95th-percentile wait of the same 534 forecast requests under each routing rule,\non the two event boxes, real model, while box A ran a real fine-tune. Lower is better.", fontsize=9.5, color=INK2)
    yy = np.arange(len(rules))[::-1]
    ax.barh(yy, [v / 1000 for _, v in rules], height=0.62, color=[ORANGE] + [BLUE] * 4)
    for y_, (name, v) in zip(yy, rules):
        ax.text(v / 1000 + 0.05, y_, f"{v / 1000:.2f} s", va="center", fontsize=10, color=INK, fontweight="bold" if name == "least loaded" else "normal")
    ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in rules], fontsize=9.5, color=INK)
    ax.set_xlim(0, 3.7); ax.set_xlabel("seconds", fontsize=9); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.text(0.035, 0.025, "0 of 5,910 requests failed in the whole test (five rules, a real fine-tune, a killed server). One run per rule.", fontsize=8.5, color=INK2)
    fig.savefig(f"{OUT}/router-p95-while-training.png")


def finetune():
    rows = [("Pressure pattern (500 hPa)", -11.1, 2.0), ("Temperature aloft (850 hPa)", -18.5, -2.5), ("Temperature at 2 m", -16.9, -4.7)]
    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=200)
    fig.subplots_adjust(left=0.33, right=0.93, top=0.60, bottom=0.11)
    fig.text(0.035, 0.915, "What 12 minutes of fine-tuning on a serving box bought", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.035, 0.735, "Change in forecast error after 300 training steps on ten months of NOAA analyses,\nrun on an event box while it served forecasts. 83 forecasts from six weeks\nthat the tuning never saw. Left of zero is better.", fontsize=9.5, color=INK2, linespacing=1.4)
    yy = np.arange(len(rows))[::-1] * 1.0
    ax.barh(yy + 0.19, [r[1] for r in rows], height=0.34, color=BLUE, label="1 day ahead")
    ax.barh(yy - 0.19, [r[2] for r in rows], height=0.34, color=ORANGE, label="3 days ahead")
    for y_, (_, a, b) in zip(yy, rows):
        ax.text(a - 0.4, y_ + 0.19, f"{a:.1f}%".replace("-", "−"), va="center", ha="right", fontsize=10, color=INK)
        ax.text(b + (0.4 if b > 0 else -0.4), y_ - 0.19, (f"+{b:.1f}%" if b > 0 else f"{b:.1f}%").replace("-", "−"), va="center", ha="left" if b > 0 else "right", fontsize=10, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(-24, 6); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%".replace("-", "−")); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(-0.02, 1.0), ncol=2, handlelength=1.2, columnspacing=1.6)
    fig.savefig(f"{OUT}/finetune-gain.png")


for f in (cold_wave, skill, router, finetune):
    f()
cold_wave(compact=True)
print("figures written to", OUT)
