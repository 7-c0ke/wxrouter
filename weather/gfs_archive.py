"""Build a small dataset of NOAA's GFS analyses on the model's grid, for fine-tuning and for checking forecasts.

The weather model was trained on ERA5 reanalysis, but a live forecast starts from NOAA's GFS analysis.
This script fetches many past analyses (one every 6 hours), turns each into the model's 20 maps exactly as
weather/gfs_live.py does, and stacks them into one file. A year is about 1,460 analyses: about 26 GB to
download, 1.7 GB when stacked.

    python weather/gfs_archive.py --out DIR --data DIR-with-stats.json --first 2025111400 --last 2026111318

Run it again after an interruption: analyses already in DIR/cycles/ are not fetched again.

Files written:
    DIR/cycles/YYYYMMDDHH.npy   one analysis each: float16 z-scores, shape (20, 120, 240)
    DIR/analyses.npy            all of them in time order: shape (N, 20, 120, 240), float16
    DIR/analyses.json           the N times, the cycles that are missing or were refused, and summary numbers

It needs NumPy and pygrib, and internet access to NOAA's open data bucket (no login).

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfs_live  # noqa: E402
import wx_common as wx  # noqa: E402

SHAPE = (wx.N_CHANNELS, wx.N_LAT, wx.N_LON)


def cycles_between(first, last):
    """Every 6-hourly cycle name from first to last, both included."""
    a = datetime.datetime.strptime(first, "%Y%m%d%H")
    b = datetime.datetime.strptime(last, "%Y%m%d%H")
    out = []
    while a <= b:
        out.append(a.strftime("%Y%m%d%H"))
        a += datetime.timedelta(hours=6)
    return out


def load_cycle(path):
    """The saved analysis at `path`, or None if the file is missing or not what we expect."""
    try:
        state = np.load(path)
    except (OSError, ValueError):
        return None
    if state.shape != SHAPE or state.dtype != np.float16 or not np.isfinite(state).all():
        return None
    return state


def fetch_one(job):
    """Fetch, decode, regrid and save one cycle. Runs in a worker process.

    Returns (cycle, status, megabytes fetched, a note). status: "have", "new", "missing" or "bad".
    """
    cycle, folder, mean, std = job
    path = os.path.join(folder, cycle + ".npy")
    if load_cycle(path) is not None:
        return cycle, "have", 0.0, ""
    try:
        records, nbytes = gfs_live.fetch_cycle(cycle)
    except gfs_live.NotPosted:
        return cycle, "missing", 0.0, "not on the server"
    except gfs_live.GfsError as e:
        return cycle, "bad", 0.0, str(e)[:200]
    try:
        fields = np.stack([gfs_live.regrid(gfs_live.decode(raw, key, codes)) * factor
                           for raw, (key, codes, factor) in zip(records, gfs_live.CHANNELS)])
    except gfs_live.GfsError as e:
        return cycle, "bad", nbytes / 1e6, str(e)[:200]
    state = gfs_live.to_state(fields, mean, std)
    mean_z = state.astype(np.float64).mean(axis=(1, 2))
    if not np.isfinite(state).all() or np.abs(mean_z).max() > gfs_live.Z_LIMIT:
        return cycle, "bad", nbytes / 1e6, "mean z-score up to %.2f" % np.abs(mean_z).max()
    with open(path + ".part", "wb") as f:
        np.save(f, state)
    os.replace(path + ".part", path)
    return cycle, "new", nbytes / 1e6, ""


def stack(out, cycles, notes):
    """Write analyses.npy and analyses.json from the per-cycle files that exist."""
    folder = os.path.join(out, "cycles")
    have = [c for c in cycles if os.path.exists(os.path.join(folder, c + ".npy"))]
    if not have:
        raise SystemExit("no analyses in %s; nothing to stack" % folder)
    path = os.path.join(out, "analyses.npy")
    big = np.lib.format.open_memmap(path + ".part", mode="w+", dtype=np.float16, shape=(len(have),) + SHAPE)
    total = np.zeros(wx.N_CHANNELS)
    kept = []
    for cycle in have:
        state = load_cycle(os.path.join(folder, cycle + ".npy"))
        if state is None:
            notes[cycle] = "the saved file could not be read"
            continue
        big[len(kept)] = state
        total += state.astype(np.float64).mean(axis=(1, 2))
        kept.append(cycle)
    big.flush()
    del big
    if len(kept) != len(have):                       # rare: write again without the unreadable ones
        data = np.load(path + ".part", mmap_mode="r")
        small = np.lib.format.open_memmap(path + ".part2", mode="w+", dtype=np.float16, shape=(len(kept),) + SHAPE)
        small[:] = data[:len(kept)]
        small.flush()
        del small, data
        os.replace(path + ".part2", path + ".part")
    os.replace(path + ".part", path)
    times = [gfs_live.iso_time(c) for c in kept]
    steps = [(datetime.datetime.strptime(b, "%Y%m%d%H") - datetime.datetime.strptime(a, "%Y%m%d%H")).total_seconds()
             for a, b in zip(kept, kept[1:])]
    info = {
        "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": gfs_live.BASE_URL + " (NOAA GFS 0.25-degree analyses, f000)",
        "file": "analyses.npy", "shape": [len(kept)] + list(SHAPE), "dtype": "float16 z-scores (stats.json)",
        "step_hours": 6, "first": times[0], "last": times[-1], "times": times,
        "pairs_6h": int(sum(1 for s in steps if s == 6 * 3600)),
        "asked_for": len(cycles), "not_included": {c: notes.get(c, "not fetched") for c in cycles if c not in kept},
        "channels": wx.CHANNEL_NAMES,
        "mean_z_per_channel": [round(float(v), 4) for v in total / len(kept)],
    }
    with open(os.path.join(out, "analyses.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="Fetch a range of GFS analyses and stack them on the model's grid.")
    p.add_argument("--out", required=True, help="folder to write (made if missing)")
    p.add_argument("--data", default=None, help="folder with stats.json (default: $WX_DATA, else /data)")
    p.add_argument("--first", required=True, metavar="YYYYMMDDHH")
    p.add_argument("--last", required=True, metavar="YYYYMMDDHH")
    p.add_argument("--workers", type=int, default=6, help="cycles fetched at the same time (default 6)")
    p.add_argument("--stack-only", action="store_true", help="fetch nothing; rebuild analyses.npy from cycles/")
    args = p.parse_args(argv)
    try:
        gfs_live.check_cycle(args.first)
        gfs_live.check_cycle(args.last)
        mean, std = gfs_live.read_stats(args.data)
    except gfs_live.GfsError as e:
        raise SystemExit("error: %s" % e)
    cycles = cycles_between(args.first, args.last)
    if not cycles:
        raise SystemExit("error: --first is after --last")
    folder = os.path.join(args.out, "cycles")
    os.makedirs(folder, exist_ok=True)

    notes = {}
    if not args.stack_only:
        started = time.time()
        counts = {"have": 0, "new": 0, "missing": 0, "bad": 0}
        megabytes = 0.0
        jobs = [(c, folder, mean, std) for c in cycles]
        print("%d cycles from %s to %s, %d at a time" % (len(cycles), cycles[0], cycles[-1], args.workers), flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, (cycle, status, mb, note) in enumerate(pool.map(fetch_one, jobs, chunksize=1), 1):
                counts[status] += 1
                megabytes += mb
                if note:
                    notes[cycle] = note
                    print("  %s: %s (%s)" % (cycle, status, note), flush=True)
                if n % 25 == 0 or n == len(cycles):
                    elapsed = time.time() - started
                    print("%4d/%d  new %d  had %d  missing %d  refused %d  %.0f MB  %.0f min so far, about %.0f min left"
                          % (n, len(cycles), counts["new"], counts["have"], counts["missing"], counts["bad"],
                             megabytes, elapsed / 60, elapsed / n * (len(cycles) - n) / 60), flush=True)
    info = stack(args.out, cycles, notes)
    print("wrote %s: %d analyses from %s to %s, %d six-hour pairs, %d not included"
          % (os.path.join(args.out, "analyses.npy"), info["shape"][0], info["first"], info["last"],
             info["pairs_6h"], len(info["not_included"])), flush=True)
    worst = int(np.argmax(np.abs(info["mean_z_per_channel"])))
    print("largest mean z-score over the set: %+.2f (%s)" % (info["mean_z_per_channel"][worst], wx.CHANNEL_NAMES[worst]))
    return 0 if info["shape"][0] >= 0.95 * len(cycles) else 1


if __name__ == "__main__":
    sys.exit(main())
