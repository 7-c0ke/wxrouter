"""Keep the forecast servers on NOAA's newest analysis: fetch each new GFS cycle and push it to the router.

weather/gfs_live.py fetches one analysis and stops. This loops around it. Every --check-s seconds it works out
which cycle NOAA should have posted by now (the newest 00/06/12/18 UTC at least --lag-h hours old). If that cycle
is not in --out yet, it runs gfs_live.py for it with --push, so the router sends it to every box. A cycle that is
not posted yet is tried again on the next check.

At start it also pushes the newest file already in --out, once, so that boxes started before that file was
written get it too.

    python weather/live_refresh.py --out /wx/bundle/states --data /wx/bundle --push http://router:8000
    python weather/live_refresh.py --out states --data bundle --push http://127.0.0.1:8000 --once

Needs what gfs_live.py needs: NumPy, pygrib and internet access to NOAA's open data bucket.

Written at the hackathon (Nov 14, 2026).
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GFS_LIVE = os.path.join(HERE, "gfs_live.py")


def expected_cycle(now, lag_h=4.0):
    """The newest cycle NOAA should have posted at `now` (UTC): 00/06/12/18, at least lag_h hours old."""
    t = now - datetime.timedelta(hours=lag_h)
    return t.replace(hour=t.hour - t.hour % 6, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H")


def newest_on_disk(out):
    """The newest YYYYMMDDHH.npy in out, as its cycle name, or None."""
    try:
        names = [n[:-4] for n in os.listdir(out) if n.endswith(".npy") and len(n) == 14 and n[:10].isdigit()]
    except OSError:
        return None
    return max(names) if names else None


def run_gfs_live(args, cycle=None):
    """Run gfs_live.py once. Returns its exit code."""
    cmd = [sys.executable, GFS_LIVE, "--out", args.out, "--data", args.data]
    if cycle:
        cmd += ["--cycle", cycle]
    for url in args.push:
        cmd += ["--push", url]
    return subprocess.call(cmd)


def push_file(args, cycle):
    """Push an analysis already on disk to every --push URL. Returns True if every push worked."""
    sys.path.insert(0, HERE)
    import gfs_live                         # numpy is needed only from here on
    with open(os.path.join(args.out, cycle + ".npy"), "rb") as f:
        npy = f.read()
    ok = True
    for base in args.push:
        try:
            status, answer = gfs_live.push_state(base.rstrip("/") + "/v1/state", cycle, npy)
            log("pushed %s to %s: %d %s" % (cycle, base, status, answer[:200]))
        except gfs_live.GfsError as e:
            log("push of %s to %s failed: %s" % (cycle, base, e))
            ok = False
    return ok


def log(text):
    print("[live-refresh %s] %s" % (datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ"), text),
          flush=True)


def step(args, now, fetch=run_gfs_live):
    """One check. Returns what happened: "up to date", "fetched CYCLE" or "not yet CYCLE (exit N)"."""
    cycle = expected_cycle(now, args.lag_h)
    have = newest_on_disk(args.out)
    if have is not None and have >= cycle:
        return "up to date"
    code = fetch(args, cycle)
    return "fetched %s" % cycle if code == 0 else "not yet %s (exit %d)" % (cycle, code)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="folder of YYYYMMDDHH.npy (the boxes' states folder)")
    p.add_argument("--data", required=True, help="folder with stats.json")
    p.add_argument("--push", action="append", default=[], help="router or server URL; may repeat")
    p.add_argument("--check-s", type=float, default=600.0, help="seconds between checks (default 600)")
    p.add_argument("--lag-h", type=float, default=4.0,
                   help="hours after its time that NOAA has posted a cycle (default 4)")
    p.add_argument("--once", action="store_true", help="one check, then stop")
    args = p.parse_args(argv)

    have = newest_on_disk(args.out)
    if have and args.push:
        push_file(args, have)
    while True:
        log(step(args, datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)))
        if args.once:
            return 0
        time.sleep(args.check_s)


if __name__ == "__main__":
    sys.exit(main())
