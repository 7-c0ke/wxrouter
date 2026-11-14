"""Shared facts about the weather data and model. Written at the hackathon (Fri Nov 13, 2026, after kickoff).

The data files were staged before the event (ERA5 at 1.5 degrees from WeatherBench 2, packed as fp16 z-scores):
  packed_1979_2019.npy  (59900, 20, 120, 240) float16, 6-hourly from 1979-01-01T00
  packed_2020.npy       (1464, 20, 120, 240) float16, 6-hourly from 2020-01-01T00
  stats.json            {"channels": [...20 names...], "mean": [...], "std": [...]}   (1979-2017 statistics)
  clim_1979_2017.npy    (366, 4, 3, 120, 240) float32: day-of-leap-year slot, hour/6, (z500, t850, t2m)
  truth_2020.npy        (1464, 3, 120, 240) float32 physical (z500, t850, t2m)
Grid: 120 latitudes from -88.5 to 90.0 (ascending; the south-pole row was dropped), 240 longitudes from 0 to 358.5.
"""
import json
import os

import numpy as np

CHANNEL_NAMES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure",
    "mean_sea_level_pressure", "temperature850", "u_component_of_wind1000", "v_component_of_wind1000",
    "geopotential1000", "u_component_of_wind850", "v_component_of_wind850", "geopotential850",
    "u_component_of_wind500", "v_component_of_wind500", "geopotential500", "temperature500", "geopotential50",
    "relative_humidity500", "relative_humidity850", "total_column_water_vapour",
]
N_CHANNELS = 20
CH_U10, CH_V10, CH_T2M, CH_SP, CH_MSL, CH_T850 = 0, 1, 2, 3, 4, 5
CH_Z500 = 14
SCORED = {"z500": CH_Z500, "t850": CH_T850, "t2m": CH_T2M}      # the three graded maps, in this order everywhere
SCORED_IDX = [CH_Z500, CH_T850, CH_T2M]

N_LAT, N_LON = 120, 240
LATS = -88.5 + 1.5 * np.arange(N_LAT)
LONS = 1.5 * np.arange(N_LON)
STEP_HOURS = 6

T0_MAIN = np.datetime64("1979-01-01T00", "h")     # first time in packed_1979_2019.npy
T0_2020 = np.datetime64("2020-01-01T00", "h")     # first time in packed_2020.npy
N_TIMES_MAIN = 59900
N_TRAIN_TIMES = 56980                               # 1979-01-01T00 .. 2017-12-31T18
VAL_FIRST, N_VAL_TIMES = 56980, 1460                # the year 2018


def data_dir(path=None):
    return path or os.environ.get("WX_DATA", "/data")


def open_packed(which, path=None):
    """which = 'main' (1979-2019) or '2020'. Returns a read-only float16 memmap (time, 20, 120, 240)."""
    name = "packed_1979_2019.npy" if which == "main" else "packed_2020.npy"
    return np.load(os.path.join(data_dir(path), name), mmap_mode="r")


def read_stats(path=None):
    """(mean, std) as float64 arrays of length 20, from stats.json."""
    with open(os.path.join(data_dir(path), "stats.json")) as f:
        s = json.load(f)
    if s["channels"] != CHANNEL_NAMES:
        raise ValueError("stats.json lists different channels")
    return np.asarray(s["mean"], np.float64), np.asarray(s["std"], np.float64)


def time_index(when, which="main"):
    """Row of a timestamp (numpy datetime64 or ISO string) in the 'main' or '2020' file."""
    t0 = T0_MAIN if which == "main" else T0_2020
    return int((np.datetime64(when, "h") - t0) // np.timedelta64(STEP_HOURS, "h"))


def locate(when):
    """('main' | '2020', row) for any timestamp from 1979 to 2020."""
    t = np.datetime64(when, "h")
    return ("2020", time_index(t, "2020")) if t >= T0_2020 else ("main", time_index(t, "main"))


def build_afno():
    """NVIDIA PhysicsNeMo's AFNO exactly as the recipe card gives it: 19,128,832 parameters."""
    from physicsnemo.models.afno import AFNO
    return AFNO(inp_shape=[N_LAT, N_LON], in_channels=N_CHANNELS, out_channels=N_CHANNELS, patch_size=[4, 4],
                num_blocks=8, embed_dim=512, depth=8)


def t7_channel_weights():
    """The loss weights: 3 for Z500, T850 and T2M, 1 for the rest, divided by their mean (60/26 and 20/26)."""
    w = np.ones(N_CHANNELS, np.float64)
    w[SCORED_IDX] = 3.0
    return w / w.mean()


def area_weights():
    """cos(latitude) weights on the 120 x 240 grid, scaled to mean 1 (for scoring)."""
    w = np.cos(np.deg2rad(LATS)).clip(min=0.0)
    w = np.repeat(w[:, None], N_LON, axis=1)
    return w / w.mean()


def test_starts(year):
    """Forecast start times for scoring a year: 00Z and 12Z every second day from Jan 1 through Dec 26.
    362 starts in 2020, 360 in other years."""
    days = np.arange(np.datetime64(f"{year}-01-01"), np.datetime64(f"{year}-12-27"), np.timedelta64(2, "D"))
    return [np.datetime64(d, "h") + np.timedelta64(h, "h") for d in days for h in (0, 12)]


def calendar_slot(when):
    """(slot, hour_index) into clim_1979_2017.npy: day of year in a leap-year calendar (Jan 1 = 0), hour // 6."""
    t = np.datetime64(when, "h")
    day = t.astype("datetime64[D]")
    slot = int((np.datetime64("2000-" + str(day)[5:]) - np.datetime64("2000-01-01")) // np.timedelta64(1, "D"))
    return slot, int((t - day) // np.timedelta64(1, "h")) // STEP_HOURS


def pick_device(want="auto"):
    import torch
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
