"""Pack what a forecast server needs into one small folder (about 105 MB), so a node needs no big data files.

Written at the hackathon (Nov 13-14, 2026).

    python weather/make_bundle.py --weights /out/stage2/final.pt --data /data --out /out/bundle

The bundle holds:
    weights/afno_event.pt   the trained model, exactly as weather/train.py wrote it
    stats.json              mean and standard deviation of each of the 20 channels (1979-2017)
    states_<first>_<n>.npy  a few days of real weather (float16 z-scores, 6-hourly) for the January 2019 replay
    states/                 put newer analyses here as YYYYMMDDHH.npy (weather/gfs_live.py writes them)
    bundle.json             what is in the bundle, with checksums
    README.md               where the data came from

Start a server on it with:
    python serve/forecast_server.py --node A --port 8101 --weights BUNDLE/weights/afno_event.pt --data BUNDLE
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wx_common as wx  # noqa: E402

README = """# The forecast server's bundle

Made by `weather/make_bundle.py` on {made}.

- `weights/afno_event.pt`: NVIDIA PhysicsNeMo's AFNO (19,128,832 parameters), trained from scratch during the
  hackathon, overnight on Nov 13-14, 2026, on a team member's own NVIDIA DGX Spark (one GB10; not one of the
  event's boxes), with `weather/train.py`
  (18,828 one-step steps, then 1,564 four-step rollout steps). Training data: ERA5 reanalysis, 1979-2017, at
  1.5 degrees, 20 channels, every 6 hours.
- `stats.json`: each channel's mean and standard deviation over 1979-2017. States are stored as z-scores.
- `{states}`: {rows} six-hourly states from {first} to {last} (the polar-vortex cold wave of January 2019,
  a winter the model never saw in training).
- `states/`: newer analyses, one file per time, named `YYYYMMDDHH.npy`.

Data: ERA5 (Hersbach et al.), via WeatherBench 2. Generated using Copernicus Climate Change Service information [2026].
Neither the European Commission nor ECMWF is responsible for any use made of it.
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description="Pack weights, statistics and a few days of weather into one folder.")
    p.add_argument("--weights", required=True, help="the trained model (stage 2's final.pt)")
    p.add_argument("--data", default=None, help="folder with packed_1979_2019.npy and stats.json (default $WX_DATA or /data)")
    p.add_argument("--out", required=True, help="the bundle folder to write (must not hold a bundle already)")
    p.add_argument("--first", default="2019-01-26T00", help="first state to include")
    p.add_argument("--rows", type=int, default=24, help="how many six-hourly states (24 = six days)")
    args = p.parse_args(argv)

    if os.path.exists(os.path.join(args.out, "bundle.json")):
        raise SystemExit(f"{args.out} already holds a bundle; choose a new folder")
    os.makedirs(os.path.join(args.out, "weights"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "states"), exist_ok=True)

    first = np.datetime64(args.first, "h")
    which, row = wx.locate(first)
    data = wx.open_packed(which, args.data)
    if row < 0 or row + args.rows > len(data):
        raise SystemExit(f"{args.first} + {args.rows} rows is outside the {which} file ({len(data)} rows)")
    last = first + np.timedelta64(wx.STEP_HOURS * (args.rows - 1), "h")
    states_name = f"states_{str(first)}_{args.rows}.npy"
    block = np.ascontiguousarray(data[row:row + args.rows])
    if block.dtype != np.float16 or block.shape[1:] != (wx.N_CHANNELS, wx.N_LAT, wx.N_LON):
        raise SystemExit(f"unexpected data: {block.shape} {block.dtype}")
    np.save(os.path.join(args.out, states_name), block)

    wx.read_stats(args.data)                                   # fails loudly if the channel list is not ours
    shutil.copyfile(os.path.join(wx.data_dir(args.data), "stats.json"), os.path.join(args.out, "stats.json"))
    weights_out = os.path.join(args.out, "weights", "afno_event.pt")
    shutil.copyfile(args.weights, weights_out)

    made = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest = {
        "made": made,
        "states": {"file": states_name, "first_time": str(first), "step_hours": wx.STEP_HOURS, "rows": args.rows,
                   "sha256": sha256(os.path.join(args.out, states_name))},
        "weights": {"file": "weights/afno_event.pt", "model": "afno-19m", "source": os.path.abspath(args.weights),
                    "bytes": os.path.getsize(weights_out), "sha256": sha256(weights_out)},
        "stats": {"file": "stats.json", "sha256": sha256(os.path.join(args.out, "stats.json"))},
        "channels": wx.CHANNEL_NAMES,
        "grid": {"lat_first": float(wx.LATS[0]), "lat_last": float(wx.LATS[-1]), "lon_first": 0.0,
                 "lon_last": float(wx.LONS[-1]), "step_deg": 1.5, "shape": [wx.N_LAT, wx.N_LON]},
    }
    with open(os.path.join(args.out, "bundle.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(README.format(made=made, states=states_name, rows=args.rows, first=str(first), last=str(last)))

    total = sum(os.path.getsize(os.path.join(d, name)) for d, _, names in os.walk(args.out) for name in names)
    print(f"wrote {args.out}: {args.rows} states from {first} to {last}, weights {manifest['weights']['bytes'] / 1e6:.1f} MB, "
          f"{total / 1e6:.1f} MB in all")
    print(f"weights sha256 {manifest['weights']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
