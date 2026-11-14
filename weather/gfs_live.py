"""Today's weather as the model's input: NOAA's newest GFS analysis, saved as the model's 20 maps.

What it does:
  1. finds the newest GFS 0.25-degree analysis on NOAA's open data bucket (no login needed);
  2. downloads only the 20 GRIB records the model needs (about 18 MB), using the bucket's index file;
  3. averages each map onto the model's 1.5-degree grid and turns it into z-scores with stats.json;
  4. saves a float16 array of shape (20, 120, 240) as YYYYMMDDHH.npy;
  5. with --push, also sends it to a forecast server or the router (POST /v1/state), so the nodes
     need no internet.

Run it on a machine with internet, NumPy and pygrib (the laptop):

    python weather/gfs_live.py --out .local/bundle/states --data .local/bundle
    python weather/gfs_live.py --out states --cycle 2026111400 --push http://127.0.0.1:8100

--data is the folder with stats.json (default: $WX_DATA). --push may be given once per server.
Nothing is saved if a channel's mean z-score is beyond +-1.0, because then the data or stats.json
is wrong; --force saves it anyway.

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import base64
import collections
import concurrent.futures
import datetime
import http.client
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import wx_common as wx

BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
USER_AGENT = "Mozilla/5.0 (compatible; python-urllib)"
TIMEOUT_S = 60
TRIES = 3
BACKOFF_S = (2.0, 5.0)          # the waits before the second and the third try
PARALLEL = 4                    # downloads at the same time
WALK_BACK = 8                   # cycles to try, newest first (8 cycles = two days)
Z_LIMIT = 1.0                   # do not save if a channel's mean z-score is beyond this
G = 9.80665                     # geopotential height (m) x G = geopotential (m2/s2)
OHARE = (41.9602, -87.9316)     # Chicago O'Hare: latitude, longitude

FINE_NLAT, FINE_NLON = 721, 1440    # the GFS grid: 90 N to 90 S and 0 E eastwards, every 0.25 degrees
COARSE_DEG = 1.5                    # the model grid's spacing: one coarse cell spans 6 fine steps
EDGE_HALF = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])    # weights of the 7 fine points across a cell

# One row per model channel, in the model's order (wx_common.CHANNEL_NAMES):
#   the record's name in NOAA's index ("variable:level"),
#   what the decoded record must say: (GRIB2 parameter category, parameter number, level in hPa or metres),
#   the factor that turns NOAA's unit into the model's unit.
CHANNELS = [
    ("UGRD:10 m above ground", (2, 2, 10), 1.0),
    ("VGRD:10 m above ground", (2, 3, 10), 1.0),
    ("TMP:2 m above ground", (0, 0, 2), 1.0),
    ("PRES:surface", (3, 0, 0), 1.0),
    ("PRMSL:mean sea level", (3, 1, 0), 1.0),
    ("TMP:850 mb", (0, 0, 850), 1.0),
    ("UGRD:1000 mb", (2, 2, 1000), 1.0),
    ("VGRD:1000 mb", (2, 3, 1000), 1.0),
    ("HGT:1000 mb", (3, 5, 1000), G),
    ("UGRD:850 mb", (2, 2, 850), 1.0),
    ("VGRD:850 mb", (2, 3, 850), 1.0),
    ("HGT:850 mb", (3, 5, 850), G),
    ("UGRD:500 mb", (2, 2, 500), 1.0),
    ("VGRD:500 mb", (2, 3, 500), 1.0),
    ("HGT:500 mb", (3, 5, 500), G),
    ("TMP:500 mb", (0, 0, 500), 1.0),
    ("HGT:50 mb", (3, 5, 50), G),
    ("RH:500 mb", (1, 1, 500), 0.01),                                          # percent -> fraction
    ("RH:850 mb", (1, 1, 850), 0.01),
    ("PWAT:entire atmosphere (considered as a single layer)", (1, 3, 0), 1.0),   # kg/m2, as column water vapour
]

# What every record must say about its grid (pygrib key, value). regrid() relies on it.
GFS_GRID = [
    ("Ni", FINE_NLON), ("Nj", FINE_NLAT), ("scanningMode", 0),
    ("latitudeOfFirstGridPointInDegrees", 90.0), ("latitudeOfLastGridPointInDegrees", -90.0),
    ("longitudeOfFirstGridPointInDegrees", 0.0), ("longitudeOfLastGridPointInDegrees", 359.75),
]

Record = collections.namedtuple("Record", "key first last date")    # one line of the index


class GfsError(Exception):
    """Something went wrong; the message is written for the person at the keyboard."""


class NotPosted(GfsError):
    """The server does not have the file (HTTP 404): the cycle is not posted yet."""


# ---------------------------------------------------------------------------------------------- cycles

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def newest_cycles(now, count):
    """The names (YYYYMMDDHH) of the last `count` 6-hourly cycles at or before `now`, newest first."""
    start = now.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=now.hour % 6)
    return [(start - datetime.timedelta(hours=6 * i)).strftime("%Y%m%d%H") for i in range(count)]


def check_cycle(text):
    """Raise GfsError unless `text` is a cycle name such as 2026111400."""
    try:
        ok = len(text) == 10 and text.isdigit() and datetime.datetime.strptime(text, "%Y%m%d%H").hour % 6 == 0
    except ValueError:
        ok = False
    if not ok:
        raise GfsError("--cycle must look like 2026111400 (YYYYMMDDHH with HH = 00, 06, 12 or 18), not %r" % text)


def iso_time(cycle):
    """2026111400 -> 2026-11-14T00 (the form the servers use)."""
    return "%s-%s-%sT%s" % (cycle[:4], cycle[4:6], cycle[6:8], cycle[8:])


def cycle_url(cycle):
    return "%s/gfs.%s/%s/atmos/gfs.t%sz.pgrb2.0p25.f000" % (BASE_URL, cycle[:8], cycle[8:], cycle[8:])


# ------------------------------------------------------------------------------------------- the index

def parse_index(text):
    """Read NOAA's index text. Returns one Record(key, first, last, date) per line.

    A line looks like "12:345678:d=2026111400:TMP:2 m above ground:anl:". The key is "TMP:2 m above ground".
    The record's bytes run from its offset (first) to the next line's offset minus one (last).
    The last record has last = None: it runs to the end of the file.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(":")
        try:
            if not parts[2].startswith("d="):
                raise ValueError(line)
            rows.append((parts[3] + ":" + parts[4], int(parts[1]), parts[2][2:]))
        except (IndexError, ValueError):
            raise GfsError("cannot read line %d of the index: %r" % (number, line[:100]))
        if len(rows) > 1 and rows[-1][1] <= rows[-2][1]:
            raise GfsError("the offsets in the index do not grow at line %d: %r" % (number, line[:100]))
    lasts = [first - 1 for _, first, _ in rows[1:]] + [None]
    return [Record(key, first, last, date) for (key, first, date), last in zip(rows, lasts)]


def pick_ranges(records, cycle):
    """The byte range (first, last) of each of the model's 20 channels, in the model's order."""
    by_key = {}
    for record in records:
        by_key.setdefault(record.key, []).append(record)
    ranges = []
    for key, _, _ in CHANNELS:
        found = by_key.get(key, [])
        if len(found) != 1:
            raise GfsError("the index of cycle %s has %d records named %r; 1 is needed" % (cycle, len(found), key))
        if found[0].date != cycle:
            raise GfsError("the index of cycle %s dates %r as %s" % (cycle, key, found[0].date))
        ranges.append((found[0].first, found[0].last))
    return ranges


# ------------------------------------------------------------------------------------------- downloads

def http_get(url, first=None, last=None):
    """GET a URL, or only its bytes first..last (last = None: to the end of the file). Returns the bytes.

    Three tries with a wait in between. A 404 raises NotPosted at once; it is not tried again.
    """
    headers = {"User-Agent": USER_AGENT}
    if first is not None:
        headers["Range"] = "bytes=%d-%s" % (first, "" if last is None else last)
    why = ""
    for attempt in range(1, TRIES + 1):
        if attempt > 1:
            print("  try %d of %d for %s (%s)" % (attempt, TRIES, url, why), file=sys.stderr, flush=True)
            time.sleep(BACKOFF_S[attempt - 2])
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                if first is not None and response.status != 206:
                    # Without this check a server that ignores Range would send the whole 500 MB file.
                    raise GfsError("%s answered %d to a byte-range request; 206 is needed" % (url, response.status))
                data = response.read()
        except urllib.error.HTTPError as e:
            e.close()
            if e.code == 404:
                raise NotPosted("not on the server (HTTP 404): %s" % url)
            why = "HTTP %d" % e.code
            continue
        except (OSError, http.client.HTTPException) as e:
            why = "%s: %s" % (type(e).__name__, e)
            continue
        if first is not None and last is not None and len(data) != last - first + 1:
            why = "got %d bytes, wanted %d" % (len(data), last - first + 1)
            continue
        return data
    raise GfsError("could not fetch %s after %d tries (%s)" % (url, TRIES, why))


def fetch_cycle(cycle):
    """Download the model's 20 records of one cycle, a few at a time.

    Returns (the 20 GRIB records as bytes, in the model's order; the number of bytes fetched).
    Raises NotPosted if the cycle is not on the server (yet).
    """
    url = cycle_url(cycle)
    index = http_get(url + ".idx")
    ranges = pick_ranges(parse_index(index.decode("utf-8", "replace")), cycle)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL)
    try:
        jobs = [pool.submit(http_get, url, first, last) for first, last in ranges]
        records = [job.result() for job in jobs]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return records, len(index) + sum(len(r) for r in records)


def fetch_newest():
    """Try the newest cycle, then walk back six hours at a time until one is posted.

    Returns (cycle, records, bytes fetched) as fetch_cycle does.
    """
    cycles = newest_cycles(utc_now(), WALK_BACK)
    for cycle in cycles:
        try:
            records, nbytes = fetch_cycle(cycle)
            return cycle, records, nbytes
        except NotPosted:
            print("cycle %s is not posted yet; trying the one before" % cycle, flush=True)
    raise GfsError("none of the last %d cycles (%s back to %s) is on NOAA's server"
                   % (len(cycles), cycles[0], cycles[-1]))


# ------------------------------------------------------------------------------------- decode and regrid

def decode(raw, key, codes):
    """One GRIB record (bytes) -> its map on the GFS grid: a (721, 1440) float64 array, 90 N first.

    Checks that the record is the one asked for (`codes`, see CHANNELS) and that its grid is the GFS grid.
    """
    if raw[:4] != b"GRIB" or raw[-4:] != b"7777" or int.from_bytes(raw[8:16], "big") != len(raw):
        raise GfsError("the bytes fetched for %r are not one whole GRIB2 record" % key)
    try:
        import pygrib
    except ImportError:
        raise GfsError("pygrib is needed to decode GRIB records; install it (pip install pygrib)")
    try:
        message = pygrib.fromstring(raw)
        said = (message["parameterCategory"], message["parameterNumber"], message["level"])
        grid = [(name, message[name]) for name, _ in GFS_GRID]
        values = np.ma.filled(message["values"], np.nan)
    except Exception as e:      # pygrib raises several kinds of error on a record it cannot read
        raise GfsError("pygrib could not decode the record %r: %s" % (key, e))
    if said != codes:
        raise GfsError("the record fetched for %r says (category, number, level) = %r; %r is needed"
                       % (key, said, codes))
    for (name, value), (_, wanted) in zip(grid, GFS_GRID):
        if abs(value - wanted) > 1e-6:
            raise GfsError("the record %r is not on the GFS 0.25-degree grid: %s is %r, not %r"
                           % (key, name, value, wanted))
    if values.shape != (FINE_NLAT, FINE_NLON) or not np.isfinite(values).all():
        raise GfsError("the record %r has shape %s or missing values" % (key, values.shape))
    return np.asarray(values, np.float64)


def regrid(fine):
    """Area-average one map from the GFS grid (721, 1440; 90 N first) onto the model grid (120, 240;
    -88.5 first).

    Each coarse cell is the mean of the 7 x 7 fine points within 0.75 degrees of its centre. The edge
    points count half (they are shared with the next cell). Each fine row also counts by cos(latitude),
    its share of the area. Longitude wraps. The row at 90 N uses only the points that exist.
    """
    fine = np.asarray(fine, np.float64)
    if fine.shape != (FINE_NLAT, FINE_NLON):
        raise ValueError("regrid() wants a map of shape (721, 1440), not %s" % (fine.shape,))
    # West to east: coarse column j is centred on fine column 6 j and uses fine columns 6 j - 3 ... 6 j + 3.
    # Three extra columns on each side make the longitude wrap.
    wide = np.concatenate([fine[:, -3:], fine, fine[:, :3]], axis=1)
    bands = np.zeros((FINE_NLAT, wx.N_LON))
    for k in range(7):
        bands += EDGE_HALF[k] * wide[:, k:k + 6 * wx.N_LON:6]
    bands /= EDGE_HALF.sum()
    # North to south: coarse row q (counted from 90 N) is centred on fine row 6 q.
    # Three extra rows of weight zero stand for "north of the pole".
    fine_lats = 90.0 - 0.25 * np.arange(FINE_NLAT)
    weight = np.concatenate([np.zeros(3), np.cos(np.deg2rad(fine_lats)).clip(min=0.0)])
    tall = np.concatenate([np.zeros((3, wx.N_LON)), bands])
    total = np.zeros((wx.N_LAT, wx.N_LON))
    weight_sum = np.zeros(wx.N_LAT)
    for k in range(7):
        w = EDGE_HALF[k] * weight[k:k + 6 * wx.N_LAT:6]
        total += w[:, None] * tall[k:k + 6 * wx.N_LAT:6]
        weight_sum += w
    return (total / weight_sum[:, None])[::-1].copy()      # the model's rows run from south to north


def to_state(fields, mean, std):
    """20 maps in the model's units, shape (20, 120, 240) -> the model's input: float16 z-scores."""
    z = (np.asarray(fields, np.float64) - mean[:, None, None]) / std[:, None, None]
    return z.astype(np.float16)


def point_value(field, lat, lon):
    """The value of a (120, 240) map on the model grid at one point, by bilinear interpolation.

    lon may be -180...180 or 0...360; it wraps. South of -88.5 the southernmost row is used.
    """
    y = min(max((lat - wx.LATS[0]) / COARSE_DEG, 0.0), wx.N_LAT - 1.0)
    x = (lon % 360.0) / COARSE_DEG
    i = min(int(y), wx.N_LAT - 2)
    j = int(x) % wx.N_LON
    j_east = (j + 1) % wx.N_LON
    fy, fx = y - i, x - int(x)
    south = (1.0 - fx) * field[i, j] + fx * field[i, j_east]
    north = (1.0 - fx) * field[i + 1, j] + fx * field[i + 1, j_east]
    return float((1.0 - fy) * south + fy * north)


# ---------------------------------------------------------------------------------------- save and push

def npy_bytes(array):
    """The bytes of a .npy file holding `array`."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def push_state(url, cycle, npy):
    """POST the state (the bytes of the .npy file) to url, a server's /v1/state.

    Returns (HTTP status, the answer as text). One try: a server on our own network is there or it is not.
    """
    body = json.dumps({"time": iso_time(cycle), "npy_b64": base64.b64encode(npy).decode("ascii")}).encode("ascii")
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, response.read().decode("utf-8", "replace").strip()
    except urllib.error.HTTPError as e:
        answer = e.read().decode("utf-8", "replace").strip()
        e.close()
        raise GfsError("push to %s failed: HTTP %d %s" % (url, e.code, answer[:300]))
    except (OSError, http.client.HTTPException) as e:
        raise GfsError("push to %s failed: %s: %s" % (url, type(e).__name__, e))


def read_stats(folder):
    """(mean, std) of the 20 channels from stats.json, with a message that says what to do if it fails."""
    try:
        return wx.read_stats(folder)
    except (OSError, ValueError, KeyError) as e:
        raise GfsError("cannot use stats.json in %s (%s: %s). Pass --data DIR, the folder with stats.json, "
                       "or set WX_DATA." % (wx.data_dir(folder), type(e).__name__, e))


def run(args):
    """Do what the command line asks. Returns the exit code; raises GfsError with a message on failure."""
    started = time.time()
    # Catch the cheap mistakes before any download.
    if args.cycle:
        check_cycle(args.cycle)
    for url in args.push:
        if not url.startswith(("http://", "https://")):
            raise GfsError("--push wants a URL such as http://host:8100, not %r" % url)
    mean, std = read_stats(args.data)
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        raise GfsError("cannot make the folder %s (%s)" % (args.out, e))

    # Download the 20 records.
    if args.cycle:
        cycle = args.cycle
        try:
            records, nbytes = fetch_cycle(cycle)
        except NotPosted as e:
            raise GfsError("cycle %s is not on NOAA's server; an analysis is posted 3.5 to 4 hours after "
                           "its hour (%s)" % (cycle, e))
    else:
        cycle, records, nbytes = fetch_newest()
    print("cycle %s (GFS analysis of %s UTC)" % (cycle, iso_time(cycle)), flush=True)

    # Decode each record, put it on the model grid in the model's units, and make z-scores.
    fields = np.stack([regrid(decode(raw, key, codes)) * factor
                       for raw, (key, codes, factor) in zip(records, CHANNELS)])
    state = to_state(fields, mean, std)
    if not np.isfinite(state).all():
        raise GfsError("the state has values that are not finite; nothing saved")

    # Report, and refuse a state that does not look like the model's training data.
    # The mean is the plain mean over the 120 x 240 grid points, as in stats.json.
    mean_z = state.astype(np.float64).mean(axis=(1, 2))
    print("mean z-score of each channel (the research saw all within +-0.5):")
    for name, z in zip(wx.CHANNEL_NAMES, mean_z):
        print("  %-26s %+.2f" % (name, z))
    worst = int(np.argmax(np.abs(mean_z)))
    print("largest: %+.2f (%s)" % (mean_z[worst], wx.CHANNEL_NAMES[worst]))
    t2m_kelvin = state[wx.CH_T2M].astype(np.float64) * std[wx.CH_T2M] + mean[wx.CH_T2M]
    print("O'Hare 2-m temperature: %.2f C" % (point_value(t2m_kelvin, OHARE[0], OHARE[1]) - 273.15))
    if abs(mean_z[worst]) > Z_LIMIT:
        problem = ("the mean z-score of %s is %+.2f, beyond +-%.1f, so the data or stats.json is wrong"
                   % (wx.CHANNEL_NAMES[worst], mean_z[worst], Z_LIMIT))
        if not args.force:
            raise GfsError("nothing saved: %s. Use --force to save it anyway." % problem)
        print("warning: %s; saving it anyway (--force)" % problem)

    # Save, then push the same bytes.
    npy = npy_bytes(state)
    path = os.path.abspath(os.path.join(args.out, cycle + ".npy"))
    try:
        with open(path + ".part", "wb") as f:
            f.write(npy)
        os.replace(path + ".part", path)    # the file appears under its name only when it is whole
    except OSError as e:
        raise GfsError("cannot write %s (%s)" % (path, e))
    print("fetched %.1f MB, took %.1f s" % (nbytes / 1e6, time.time() - started))
    print("wrote %s  (float16, shape %s)" % (path, state.shape), flush=True)

    failed = 0
    for base in args.push:
        url = base.rstrip("/") + "/v1/state"
        try:
            status, answer = push_state(url, cycle, npy)
            print("pushed to %s: %d %s" % (url, status, answer), flush=True)
        except GfsError as e:
            failed += 1
            print("error: %s (the file above is saved)" % e, file=sys.stderr, flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch NOAA's newest GFS analysis and save it as the model's 20 maps: "
                    "float16 z-scores of shape (20, 120, 240) in DIR/YYYYMMDDHH.npy.")
    parser.add_argument("--out", required=True, metavar="DIR", help="folder for YYYYMMDDHH.npy (made if missing)")
    parser.add_argument("--cycle", metavar="YYYYMMDDHH",
                        help="the analysis to fetch, HH = 00, 06, 12 or 18 (default: the newest one posted)")
    parser.add_argument("--data", metavar="DIR", help="folder with stats.json (default: $WX_DATA, else /data)")
    parser.add_argument("--push", metavar="URL", action="append", default=[],
                        help="also POST the state to URL/v1/state, a forecast server or the router, "
                             "such as http://127.0.0.1:8100 (may be given more than once)")
    parser.add_argument("--force", action="store_true", help="save even if a mean z-score is beyond +-1.0")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except GfsError as e:
        print("error: %s" % e, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
