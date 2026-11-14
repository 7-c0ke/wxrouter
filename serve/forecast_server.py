"""The forecast server: one per node. It answers weather-forecast requests over HTTP (docs/CONTRACT.md, section 1).

Real mode runs the trained weather model on this machine's GPU. Mock mode (--mock) needs no GPU, no PyTorch
and no NumPy: it sleeps to emulate the work and answers with made-up, repeatable numbers.

    python serve/forecast_server.py --mock --admin --node A --port 8101
    python serve/forecast_server.py --node A --port 8101 --weights /out/stage2/final.pt --data /data
    curl -s localhost:8101/v1/forecast -d '{"init":"latest","lead_hours":24,"point":{"lat":41.96,"lon":-87.93}}'

    GET  /health   /v1/info   /metrics (JSON; Prometheus text with ?format=prom)
    POST /v1/forecast   /v1/state
    POST /admin/contention   /admin/fail   /admin/reset      (test hooks; only with --admin)

The first line it prints is "listening on http://HOST:PORT" (--port 0 picks a free port). Ctrl-C or SIGTERM
stops it. One worker thread does the forecasts one at a time, as one GPU would; requests wait in a queue.
The mock listens on 127.0.0.1 only. The real server listens on every address (0.0.0.0), so that the router
can reach it from the other node. --host changes either.

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import collections
import datetime
import json
import math
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import hashlib
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

REAL_MODEL = "afno-19m"
MAX_BODY = 8 * 1024 * 1024          # bytes. A state sent as npy_b64 is about 1.6 MB.
FORECAST_FIELDS = ("init", "lead_hours", "members", "member_offset", "point", "with_truth", "request_id")
FAIL_MODES = ("down", "hang", "error", "ok")
TIME_FORM = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})(?::00(?::00)?)?Z?")
MOCK_FIRST = datetime.datetime(1979, 1, 1)      # the mock "holds" every 6-hourly time from here to its latest


class ApiError(Exception):
    """An answer that is not 200: the HTTP status and the text for {"error": ...}."""

    def __init__(self, status, message):
        Exception.__init__(self, message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- checking what a client sent

def show(value):
    """A value as short JSON text, for error messages."""
    try:
        text = json.dumps(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def time_text(when):
    """datetime -> "2019-01-27T12"."""
    return "%04d-%02d-%02dT%02d" % (when.year, when.month, when.day, when.hour)


def parse_time(value, name):
    """"2019-01-27T12" (UTC; ":00", ":00:00" and "Z" may follow) -> datetime. Anything else: ApiError(400)."""
    found = TIME_FORM.fullmatch(value) if isinstance(value, str) else None
    if found:
        try:
            when = datetime.datetime(*[int(part) for part in found.groups()])
        except ValueError:
            when = None
        if when is not None and 1900 <= when.year <= 2200:
            return when
    raise ApiError(400, '%s must be a UTC time like "2019-01-27T12", got %s' % (name, show(value)))


def analysis_time(value, name):
    """parse_time, and the hour must be 00, 06, 12 or 18: every state sits on a 6-hour mark."""
    when = parse_time(value, name)
    if when.hour % 6:
        raise ApiError(400, "%s must be at 00, 06, 12 or 18 UTC, got %s" % (name, time_text(when)))
    return when


def whole_number(value, name, low, high=None):
    """`value` as an int from low to high (None: no upper limit), or ApiError(400). 24.0 passes; true, 1.5
    and "24" do not."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        limits = "%d or more" % low if high is None else "from %d to %d" % (low, high)
        raise ApiError(400, "%s must be a whole number %s, got %s" % (name, limits, show(value)))
    return value


def number(value, name, low, high):
    """`value` as a float from low to high, or ApiError(400). true, NaN and text do not pass."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (low <= value <= high):
        raise ApiError(400, "%s must be a number from %g to %g, got %s" % (name, low, high, show(value)))
    return float(value)


def check_forecast_request(body):
    """Check the body of POST /v1/forecast and fill in the defaults. ApiError(400) says what is wrong.

    Only lead_hours and point must be there. In the result, "init" is "latest", a datetime, or
    {"npy_b64": text, "time": datetime}; lat and lon sit at the top level."""
    if not isinstance(body, dict):
        raise ApiError(400, "the body must be a JSON object")
    for key in body:
        if key not in FORECAST_FIELDS:
            raise ApiError(400, "unknown field %s; the fields are: %s" % (show(key), ", ".join(FORECAST_FIELDS)))
    body = dict((key, value) for key, value in body.items() if value is not None)     # null = not given

    if "lead_hours" not in body:
        raise ApiError(400, "lead_hours is missing: 6 to 120, a multiple of 6")
    lead_hours = whole_number(body["lead_hours"], "lead_hours", 6, 120)
    if lead_hours % 6:
        raise ApiError(400, "lead_hours must be a multiple of 6, got %d" % lead_hours)
    members = whole_number(body.get("members", 1), "members", 1, 40)
    member_offset = whole_number(body.get("member_offset", 0), "member_offset", 0)

    point = body.get("point")
    if not isinstance(point, dict) or sorted(point) != ["lat", "lon"]:
        raise ApiError(400, 'point must be {"lat": ..., "lon": ...}, got %s' % show(point))
    lat = number(point["lat"], "point.lat", -90, 90)
    lon = number(point["lon"], "point.lon", -180, 360)

    with_truth = body.get("with_truth", False)
    if not isinstance(with_truth, bool):
        raise ApiError(400, "with_truth must be true or false, got %s" % show(with_truth))
    request_id = body.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 200):
        raise ApiError(400, "request_id must be text of at most 200 characters")

    init = body.get("init", "latest")
    if isinstance(init, dict):
        if sorted(init) != ["npy_b64", "time"] or not isinstance(init["npy_b64"], str) or not init["npy_b64"]:
            raise ApiError(400, 'an init object must be {"npy_b64": "...", "time": "2026-11-14T00"}')
        init = {"npy_b64": init["npy_b64"], "time": analysis_time(init["time"], "init.time")}
    elif init != "latest":
        try:
            init = parse_time(init, "init")
        except ApiError:
            raise ApiError(400, 'init must be "latest", a UTC time like "2019-01-27T12" or '
                                '{"npy_b64": "...", "time": "..."}; got %s' % show(init))
    return {"init": init, "lead_hours": lead_hours, "members": members, "member_offset": member_offset,
            "lat": lat, "lon": lon, "with_truth": with_truth, "request_id": request_id}


# ---------------------------------------------------------------- the team contract's names (API.md, v1.1)
# The same forecast can be asked for with: lat, lon, hours_ahead, step_hours, ensemble_members.
# The model steps 6 hours at a time. hours_ahead is rounded up to a multiple of 6 for the model run, and
# step_hours may be 1 to 24: points between the model's steps are interpolated in time, and the answer says
# so ("native_step_hours": 6, "interpolated"). The model has no rain output, so precip_mm is null.
# init, member_offset and with_truth may be added as optional extras.

V1_ONLY = ("lat", "lon", "hours_ahead", "step_hours", "ensemble_members")
V1_FIELDS = V1_ONLY + ("request_id", "init", "member_offset", "with_truth")
IDLE_BEFORE_BLAME_S = 4.0           # the GPU is "someone else's" only after we have been idle this long
OTHER_HOLD_S = 10.0                 # and the flag stays up this long after it was last seen


def is_v1(body):
    return isinstance(body, dict) and any(key in body for key in V1_ONLY)


def from_v1(body):
    """A body in the team contract's names -> (the body in this server's own names, the request as given). ApiError(422) says what
    is wrong. step_hours may be 1 to 24: values between the model's 6-hour steps are interpolated in time."""
    for key in body:
        if key not in V1_FIELDS:
            raise ApiError(422, "unknown field %s; the fields are: %s" % (show(key), ", ".join(V1_FIELDS)))
    given = dict((key, value) for key, value in body.items() if value is not None)
    for key in ("lat", "lon", "hours_ahead"):
        if key not in given:
            raise ApiError(422, "%s is missing; a forecast needs lat, lon and hours_ahead" % key)
    try:
        hours = whole_number(given["hours_ahead"], "hours_ahead", 1, 120)
        step = whole_number(given.get("step_hours", 1), "step_hours", 1, 24)
        members = whole_number(given.get("ensemble_members", 1), "ensemble_members", 1, 40)
    except ApiError as e:
        raise ApiError(422, e.message)
    own = {"lead_hours": 6 * ((hours + 5) // 6), "members": members,
           "point": {"lat": given["lat"], "lon": given["lon"]}}
    for key in ("request_id", "init", "member_offset", "with_truth"):
        if key in given:
            own[key] = given[key]
    return own, {"hours_ahead": hours, "step_hours": step, "request_id": given.get("request_id")}


def between(values, hour):
    """The value at `hour` from a series given every 6 hours from hour 0 (straight line in between)."""
    k, rest = divmod(hour, 6)
    a = values[k]
    if rest == 0 or a is None:
        return a
    b = values[k + 1]
    return None if b is None else round(a + (b - a) * rest / 6.0, 2) + 0.0


def v1_points(init_time, hours, step, temp, wind, pressure, extra=None):
    """The contract's list of points: one every `step` hours up to `hours`. Each series starts at hour 0."""
    points = []
    for lead in range(step, hours + 1, step):
        point = {"lead_hours": lead,
                 "valid_time": (init_time + datetime.timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "temp_c": between(temp, lead), "wind_ms": between(wind, lead),
                 "precip_mm": None,              # this model has no precipitation; null, not a made-up zero
                 "pressure_hpa": between(pressure, lead)}
        for name, series in (extra or {}).items():
            point[name] = between(series, lead)
        points.append(point)
    return points


def to_v1(answer, asked, init_time, start, members, model_version):
    """This server's answer -> the team contract's shape (API.md, v1.1).

    `start` holds the analysed values at the start time (hour 0); `members` each member's own series.
    lead_hours counts from init_time, the newest analysis, which is some hours before the request."""
    point, hours, step = answer["point"], asked["hours_ahead"], asked["step_hours"]

    def from_zero(first, series):
        return [first] + list(series)

    forecast = v1_points(init_time, hours, step, from_zero(start["t2m_c"], point["t2m_c_mean"]),
                         from_zero(start["wind10_ms"], point["wind10_ms_mean"]),
                         from_zero(start["msl_hpa"], point["msl_hpa_mean"]),
                         {"temp_c_min": from_zero(start["t2m_c"], point["t2m_c_min"]),
                          "temp_c_max": from_zero(start["t2m_c"], point["t2m_c_max"])})
    if "truth_t2m_c" in point:
        for entry in forecast:
            lead = entry["lead_hours"]
            entry["truth_temp_c"] = point["truth_t2m_c"][lead // 6 - 1] if lead % 6 == 0 else None
    each = None
    if answer["members"] > 1:
        each = [{"member": answer["member_offset"] + m,
                 "points": v1_points(init_time, hours, step, from_zero(start["t2m_c"], members["t2m_c"][m]),
                                     from_zero(start["wind10_ms"], members["wind10_ms"][m]),
                                     from_zero(start["msl_hpa"], members["msl_hpa"][m]))}
                for m in range(answer["members"])]
    timing = answer["timing_ms"]
    return {"request_id": asked["request_id"], "box": answer["node"], "model_version": model_version,
            "contract_version": "1.1", "issued_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat": point["lat"], "lon": point["lon"], "hours_ahead": hours, "step_hours": step,
            "low_confidence": hours > 72, "forecast": forecast, "members": each,
            "timing": {"queue_ms": timing["queue"], "compute_ms": timing["gpu"], "total_ms": timing["total"]},
            # Beyond the contract: what a reader needs to interpret the points.
            "init_time": answer["init_time"], "native_step_hours": 6,
            "interpolated": step % 6 != 0 or hours % 6 != 0,
            "ensemble_members": answer["members"], "units": answer["units"]}


STATE_CLOCK_SLACK_H = 1     # a state to keep may be dated this far ahead of this machine's clock, not more


def check_state_request(body, now=None):
    """Check the body of POST /v1/state. Returns (time, npy_b64 text).

    The time must not be in the future. A kept state becomes "latest" when it is the newest, so a state
    with a late date would start every forecast "from latest" at a time that has not come, and it would hide
    every real analysis that arrives after it (it also stays on the shared volume). `now` is for the tests."""
    if (not isinstance(body, dict) or sorted(body) != ["npy_b64", "time"]
            or not isinstance(body["npy_b64"], str) or not body["npy_b64"]):
        raise ApiError(400, 'the body must be {"time": "2026-11-14T00", "npy_b64": "..."}')
    when = analysis_time(body["time"], "time")
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    if when > now + datetime.timedelta(hours=STATE_CLOCK_SLACK_H):
        raise ApiError(400, "time %s is in the future (it is %s UTC now): an analysis is weather that was measured, "
                            "so a state to keep cannot be dated later than now" % (time_text(when), time_text(now)))
    return when, body["npy_b64"]


# ---------------------------------------------------------------- mock mode: made-up weather

def mock_weather(valid, lat, lon):
    """Made-up "true" weather at a place and a valid time: (2-m temperature in C, 10-m wind in m/s,
    sea-level pressure in hPa). Smooth in every argument, and the same on every node."""
    hours = (valid - MOCK_FIRST).total_seconds() / 3600.0
    day_of_year = valid.timetuple().tm_yday + valid.hour / 24.0
    local_hour = (valid.hour + lon / 15.0) % 24.0
    la, lo = math.radians(lat), math.radians(lon)
    wave = 2.0 * math.pi * hours
    t2m = 28.0 - 48.0 * math.sin(la) ** 2                                               # warm equator, cold poles
    t2m -= 22.0 * (lat / 90.0) * math.cos(2.0 * math.pi * (day_of_year - 15) / 365.25)   # seasons; south is opposite
    t2m += 4.0 * math.cos(la) * math.cos(2.0 * math.pi * (local_hour - 15.0) / 24.0)     # warmest in mid-afternoon
    t2m += 3.0 * math.sin(wave / 120.0 + 2.0 * lo + 3.0 * la)                            # a 5-day weather wave
    wind = 5.0 + 3.0 * math.sin(wave / 72.0 + 3.0 * lo + la)
    msl = 1013.0 + 12.0 * math.sin(wave / 144.0 + 2.0 * lo - 3.0 * la)
    return t2m, wind, msl


def mock_series(init, lead_hours, members, member_offset, lat, lon, truth_until=None, detail=False):
    """The made-up point series of a response (the same keys as the real model's).

    A forecast is the made-up truth plus a drift that grows with the lead (it depends on the init time only),
    plus a member's own offset, which also grows with the lead. The offset comes from random.Random seeded
    with (init time, global member number); member 0 has none. So the same member is the same on every node.
    With truth_until (a time), the truth is added for valid times up to it; later ones get None."""
    init_text = time_text(init)
    drift_rng = random.Random(init_text + "|drift")
    drift = [drift_rng.uniform(-0.5, 0.5), drift_rng.uniform(-0.3, 0.3), drift_rng.uniform(-0.8, 0.8)]   # per day
    offsets = []
    for member in range(member_offset, member_offset + members):
        rng = random.Random("%s|%d" % (init_text, member))
        offsets.append([rng.gauss(0.0, 1.0) if member else 0.0 for _ in range(3)])

    def r2(value):
        return round(value, 2) + 0.0            # "+ 0.0" turns -0.0 into 0.0

    series = {"t2m_c": [] if member_offset == 0 else None, "t2m_c_mean": [], "t2m_c_min": [], "t2m_c_max": [],
              "wind10_ms_mean": [], "msl_hpa_mean": []}
    if truth_until is not None:
        series["truth_t2m_c"] = []
    each = {"t2m_c": [[] for _ in offsets], "wind10_ms": [[] for _ in offsets], "msl_hpa": [[] for _ in offsets]}
    for lead in range(6, lead_hours + 6, 6):
        valid = init + datetime.timedelta(hours=lead)
        t2m, wind, msl = mock_weather(valid, lat, lon)
        days = lead / 24.0
        temps = [t2m + drift[0] * days + off[0] * (0.1 + 0.4 * days) for off in offsets]
        winds = [max(0.0, wind + drift[1] * days + off[1] * (0.05 + 0.25 * days)) for off in offsets]
        pressures = [msl + drift[2] * days + off[2] * (0.1 + 0.5 * days) for off in offsets]
        if member_offset == 0:
            series["t2m_c"].append(r2(temps[0]))
        series["t2m_c_mean"].append(r2(sum(temps) / members))
        series["t2m_c_min"].append(r2(min(temps)))
        series["t2m_c_max"].append(r2(max(temps)))
        series["wind10_ms_mean"].append(r2(sum(winds) / members))
        series["msl_hpa_mean"].append(r2(sum(pressures) / members))
        for m in range(members):
            each["t2m_c"][m].append(r2(temps[m]))
            each["wind10_ms"][m].append(r2(winds[m]))
            each["msl_hpa"][m].append(r2(pressures[m]))
        if truth_until is not None:
            series["truth_t2m_c"].append(r2(t2m) if valid <= truth_until else None)
    if detail:
        series["_members"] = each
    return series


class MockEngine(object):
    """Mock mode. No model: it sleeps to emulate one GPU and makes the weather up."""

    model, mock, params, device = "mock", True, 0, "none"
    status, error = "ok", None

    def __init__(self, ms_per_unit, latest, stop):
        self.base_ms = ms_per_unit      # emulated GPU time for one unit, undisturbed
        self.fixed_latest = latest      # --latest, or None to follow the clock
        self.pushed_latest = None       # the newest time sent to /v1/state
        self.factor = 1.0               # the contention hook: 1.8 = a training job makes us 1.8x slower
        self.stop = stop

    def load(self):
        pass

    def latest(self):
        """--latest if given, else the most recent 00/06/12/18 UTC that is at least 4 hours old (about when
        a real analysis would be out). A newer state sent to /v1/state wins."""
        when = self.fixed_latest
        if when is None:
            old = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(hours=4)
            when = old.replace(hour=old.hour - old.hour % 6, minute=0, second=0, microsecond=0)
        if self.pushed_latest is not None and self.pushed_latest > when:
            when = self.pushed_latest
        return when

    def ranges(self):
        return [(MOCK_FIRST, self.latest())]

    def find_init(self, init):
        """The init time of a request (and no state: the mock needs none). 404 if the time is not "held"."""
        if isinstance(init, dict):
            return init["time"], None           # the state itself is not decoded
        latest = self.latest()
        when = latest if init == "latest" else init
        if when.hour % 6 or not (MOCK_FIRST <= when <= latest):
            raise ApiError(404, "init time %s is not held; this server holds %s to %s, every 6 hours"
                           % (time_text(when), time_text(MOCK_FIRST), time_text(latest)))
        return when, None

    def push_state(self, when, npy_b64):
        if self.pushed_latest is None or when > self.pushed_latest:
            self.pushed_latest = when
        return self.latest()

    def set_contention(self, factor):
        self.factor = factor

    def work(self, job):
        req = job.req
        t0 = time.monotonic()
        if self.stop.wait(job.units * self.base_ms * self.factor / 1000.0):     # the emulated GPU work
            raise ApiError(503, "the server is stopping")
        job.gpu_ms = (time.monotonic() - t0) * 1000.0
        detail = bool(req.get("detail"))
        job.series = mock_series(job.init_time, req["lead_hours"], req["members"], req["member_offset"],
                                 req["lat"], req["lon"], self.latest() if req["with_truth"] else None, detail)
        if detail:
            t0m, w0m, p0m = mock_weather(job.init_time, req["lat"], req["lon"])
            job.series["_init"] = {"t2m_c": round(t0m, 2), "wind10_ms": round(w0m, 2), "msl_hpa": round(p0m, 2)}

    def speed(self):
        """(ms per unit now, slowdown)."""
        return self.base_ms * self.factor, self.factor

    def machine(self, busy):
        """Made-up GPU and host numbers, and busy_other. They move with the load and the contention hook."""
        contended = self.factor > 1.0
        hot = busy or contended
        gpu = {"util_pct": 96 if hot else 0, "temp_c": 70 if hot else 45, "power_w": 60.0 if hot else 12.0,
               "proc_mem_mb": 7115}
        host = {"mem_available_gb": 60.0 if contended else 101.2, "load1": 1.9 if hot else 0.3}
        return gpu, host, contended


# ---------------------------------------------------------------- real mode: the model and the machine

def as_number(text):
    """One cell of nvidia-smi's CSV as a float; None for "[N/A]" and the like."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def nvidia_smi(query):
    """Run one nvidia-smi CSV query. Returns the rows as lists of text, or None if it failed."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-" + query, "--format=csv,noheader,nounits"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [[cell.strip() for cell in line.split(",")] for line in done.stdout.splitlines() if line.strip()]


def mem_available_gb(path="/proc/meminfo"):
    """MemAvailable in GiB, or None where there is no /proc (a laptop). A GB10 has unified memory, so this
    is the number that says how much the GPU can still take."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1048576.0, 1)       # the file counts in kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_machine(have_smi):
    """The GPU and host numbers for /metrics, and busy_other: (gpu, host, busy_other).

    Any number may be None: a machine without nvidia-smi or /proc just has no answer.
    busy_other is True when a process other than this one holds more than 500 MB on the GPU."""
    gpu = {"util_pct": None, "temp_c": None, "power_w": None, "proc_mem_mb": None}
    busy_other = False
    if have_smi:
        rows = nvidia_smi("gpu=utilization.gpu,temperature.gpu,power.draw")
        if rows and len(rows[0]) == 3:
            util, temp, power = [as_number(cell) for cell in rows[0]]
            gpu["util_pct"] = None if util is None else int(round(util))
            gpu["temp_c"] = None if temp is None else int(round(temp))
            gpu["power_w"] = None if power is None else round(power, 1)
        apps = nvidia_smi("compute-apps=pid,used_memory")
        if apps is not None:
            used = [(as_number(row[0]), as_number(row[1])) for row in apps if len(row) == 2]
            known = [mb for _, mb in used if mb is not None]
            if known or not used:
                gpu["proc_mem_mb"] = int(round(sum(known)))
            busy_other = any(pid != os.getpid() and mb is not None and mb > 500 for pid, mb in used)
    host = {"mem_available_gb": mem_available_gb(), "load1": None}
    try:
        host["load1"] = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        pass
    return gpu, host, busy_other


class RealEngine(object):
    """Real mode: the trained model. NumPy, PyTorch and weather/forecast_core.py are imported in load(),
    not at the top of this file, so mock mode runs without them."""

    model, mock, factor = REAL_MODEL, False, 1.0

    def __init__(self, weights, data_dir, device, batch, stop):
        self.weights, self.data_dir, self.want_device, self.batch, self.stop = weights, data_dir, device, batch, stop
        self.status, self.error = "loading", None       # then "ok" or "failed"
        self.params = None
        self.device = None
        self.core = None                # the forecast_core module
        self.forecaster = None          # the model, the statistics and the state store
        self.ms_per_unit = None         # running average of model time per unit
        self.slowdown = 1.0
        self.best = {}                  # members in a batch -> the best ms per unit seen for that batch size
        self.gpu = {"util_pct": None, "temp_c": None, "power_w": None, "proc_mem_mb": None}
        self.host = {"mem_available_gb": None, "load1": None}
        self.busy_other = False

    def load(self):
        """Import the heavy parts, open the data and load the model. Runs once, in the worker thread."""
        threading.Thread(target=self.watch, name="machine-watch", daemon=True).start()
        try:
            weather = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weather")
            if weather not in sys.path:
                sys.path.insert(0, weather)
            import forecast_core
            self.core = forecast_core
            self.forecaster = forecast_core.Forecaster(self.weights, self.data_dir, self.want_device, self.batch)
            self.params = self.forecaster.params
            self.device = str(self.forecaster.device)
            self.timed(1, 1, self.forecaster.warm_up() * 1000.0)
            self.status = "ok"
            sys.stderr.write("model ready on %s: %d parameters, %.1f ms for one member-step\n"
                             % (self.device, self.params, self.ms_per_unit))
        except Exception as e:
            traceback.print_exc()
            self.error = "%s: %s" % (type(e).__name__, e)
            self.status = "failed"
            sys.stderr.write("FAILED to load the model (%s). /health now answers 503. Fix it and start again.\n"
                             % self.error)

    def watch(self):
        """Refresh the GPU and host numbers every 2 seconds, until the server stops."""
        have_smi = shutil.which("nvidia-smi") is not None
        while True:
            try:
                self.gpu, self.host, self.busy_other = read_machine(have_smi)
            except Exception:
                traceback.print_exc()
            if self.stop.wait(2.0):
                return

    def timed(self, batch_members, units, ms):
        """Count one batch's model time. ms_per_unit is a running average. slowdown is a running average of
        "this batch's time per unit over the best seen for a batch of the same size", so a mix of small and
        large requests does not look like a slow GPU (a large batch costs less per member)."""
        per_unit = ms / units
        self.ms_per_unit = per_unit if self.ms_per_unit is None else 0.8 * self.ms_per_unit + 0.2 * per_unit
        best = min(self.best.get(batch_members, per_unit), per_unit)
        self.best[batch_members] = best
        self.slowdown = 0.7 * self.slowdown + 0.3 * per_unit / best

    def latest(self):
        return self.forecaster.store.latest()

    def ranges(self):
        return self.forecaster.store.ranges()

    def find_init(self, init):
        """The init time and starting state of a request. 400 for a bad npy_b64; 404 for a time not held."""
        if isinstance(init, dict):
            try:
                return init["time"], self.core.decode_state(init["npy_b64"])
            except ValueError as e:
                raise ApiError(400, "init.npy_b64: %s" % e)
        when = self.latest() if init == "latest" else init
        if when is None:
            raise ApiError(404, "no analysis is held yet; send one with POST /v1/state")
        state = self.forecaster.store.get(when)
        if state is None:
            raise ApiError(404, "init time %s is not held; GET /v1/info lists the ranges" % time_text(when))
        return when, state

    def push_state(self, when, npy_b64):
        try:
            state = self.core.decode_state(npy_b64)
        except ValueError as e:
            raise ApiError(400, "npy_b64: %s" % e)
        return self.forecaster.store.put(when, state)

    def set_contention(self, factor):
        if factor != 1.0:
            raise ApiError(400, "contention is only emulated in mock mode; on a real node, run a real job on the GPU")

    def work(self, job):
        req = job.req
        steps = req["lead_hours"] // 6
        job.series, timings = self.forecaster.run(job.state, job.init_time, steps, req["members"],
                                                  req["member_offset"], req["lat"], req["lon"], req["with_truth"],
                                                  req.get("detail", False))
        for batch_members, seconds in timings:
            self.timed(batch_members, batch_members * steps, seconds * 1000.0)
        job.gpu_ms = sum(seconds for _, seconds in timings) * 1000.0

    def speed(self):
        return self.ms_per_unit, self.slowdown

    def machine(self, busy):
        return self.gpu, self.host, self.busy_other


# ---------------------------------------------------------------- the queue, the counters, the test hooks

class Job(object):
    """One forecast request on its way through the queue."""

    def __init__(self, req, init_time, state):
        self.req = req                  # the checked request
        self.init_time = init_time
        self.state = state              # real mode: the starting state. Mock mode: None.
        self.units = req["members"] * (req["lead_hours"] // 6)
        self.arrived = time.monotonic()
        self.started = None             # when the worker took it
        self.gpu_ms = 0.0
        self.series = None              # the answer: the point's series
        self.error = None               # or an ApiError
        self.done = threading.Event()


def percentile(ordered, percent):
    """Nearest-rank percentile of a sorted list; None if the list is empty."""
    if not ordered:
        return None
    return ordered[max(1, int(math.ceil(percent / 100.0 * len(ordered)))) - 1]


def rounded(value, digits):
    return None if value is None else round(value, digits)


class App(object):
    """What all request threads share: the engine, the one worker and its queue, the counters, the test hooks."""

    def __init__(self, node, engine, admin, stop):
        self.node = node
        self.engine = engine
        self.admin = admin                  # are the /admin/ test hooks on?
        self.stop = stop                    # set when the server is shutting down
        self.lock = threading.Lock()
        self.jobs = queue.Queue()           # first in, first out
        self.inflight = 0                   # requests the worker is computing now (0 or 1)
        self.queued = 0                     # requests waiting for the worker
        self.queued_units = 0               # units waiting or running
        self.fail_mode = "ok"               # the /admin/fail hook
        self.not_hung = threading.Event()   # cleared while fail_mode is "hang"; hung requests wait on it
        self.not_hung.set()
        self.ids = 0
        self.born = time.monotonic()
        self.last_busy = time.monotonic()   # when this server last started or finished a forecast
        self.other_seen = None              # when the GPU was last seen busy while this server was idle
        self.reset_counters()
        self.worker = threading.Thread(target=self.work, name="worker", daemon=True)

    def reset_counters(self):
        self.served_total = 0               # forecasts answered with 200
        self.errors_total = 0               # answers with status 500
        self.units_total = 0                # units computed
        self.latencies = collections.deque(maxlen=200)      # total ms of the last 200 forecasts served
        self.latency_ewma = None

    def work(self):
        """The one worker thread: load the model (real mode), then take requests in the order they came."""
        self.engine.load()
        while True:
            job = self.jobs.get()
            if job is None:                 # main() says stop
                return
            with self.lock:
                self.queued -= 1
                self.inflight += 1
                self.last_busy = time.monotonic()
            job.started = time.monotonic()
            try:
                if self.stop.is_set():
                    raise ApiError(503, "the server is stopping")
                self.engine.work(job)
            except ApiError as e:
                job.error = e
            except Exception as e:
                traceback.print_exc()
                job.error = ApiError(500, "the forecast failed: %s: %s" % (type(e).__name__, e))
            with self.lock:
                self.inflight -= 1
                self.last_busy = time.monotonic()
                self.queued_units -= job.units
                if job.error is None:
                    self.units_total += job.units
            job.done.set()

    def status(self):
        """"ok", or the reason this server takes no work now: stopping, down (test hook), loading, failed."""
        if self.stop.is_set():
            return "stopping"
        if self.fail_mode == "down":
            return "down"
        return self.engine.status

    def check_ready(self):
        status = self.status()
        if status != "ok":
            why = " (%s)" % self.engine.error if status == "failed" else ""
            raise ApiError(503, "not ready: %s%s" % (status, why))

    def health(self, status):
        body = {"status": status, "node": self.node, "model": self.engine.model, "ready": status == "ok",
                "box": self.node, "contract_version": "1.1", "uptime_s": round(time.monotonic() - self.born, 1),
                "model_version": self.model_version() if status == "ok" else self.engine.model}
        if status == "failed":
            body["error"] = self.engine.error
        return body

    def info(self):
        engine = self.engine
        latest = engine.latest()
        return {"node": self.node, "model": engine.model, "params": engine.params, "device": engine.device,
                "mock": engine.mock, "latest": None if latest is None else time_text(latest),
                "ranges": [{"first": time_text(first), "last": time_text(last)} for first, last in engine.ranges()],
                "ms_per_unit": rounded(engine.speed()[0], 2)}

    def metrics(self):
        with self.lock:
            ordered = sorted(self.latencies)
            numbers = {"node": self.node, "ts": round(time.time(), 3), "inflight": self.inflight,
                       "queued": self.queued, "queued_units": self.queued_units,
                       "served_total": self.served_total, "errors_total": self.errors_total,
                       "units_total": self.units_total}
            ewma = self.latency_ewma
        ms_per_unit, slowdown = self.engine.speed()
        gpu, host, busy_other = self.engine.machine(numbers["inflight"] > 0)
        numbers["ms_per_unit"] = rounded(ms_per_unit, 2)
        numbers["latency_ms"] = {"p50": rounded(percentile(ordered, 50), 1), "p95": rounded(percentile(ordered, 95), 1),
                                 "ewma": rounded(ewma, 1)}
        numbers["gpu"] = gpu
        numbers["host"] = host
        if not self.engine.mock:
            # Process ids do not match inside a container, so "another process" is judged by what we can see:
            # the GPU is busy although this server has been idle for a while, or our own work has slowed.
            now = time.monotonic()
            util = gpu.get("util_pct")
            with self.lock:
                idle = self.inflight == 0 and self.queued == 0 and now - self.last_busy >= IDLE_BEFORE_BLAME_S
                if idle and util is not None and util >= 50:
                    self.other_seen = now
                seen = self.other_seen is not None and now - self.other_seen <= OTHER_HOLD_S
                working = self.inflight > 0 or now - self.last_busy <= OTHER_HOLD_S
            # A slow-down only counts while we are working: the average does not move when no request comes.
            busy_other = bool(seen or (working and slowdown >= 1.3))
        numbers["busy_other"] = busy_other
        numbers["slowdown"] = round(slowdown, 2)
        return numbers

    def model_version(self):
        """A short name for the weights in use: the model's name plus the start of the file's SHA-256."""
        if getattr(self, "_version", None) is None:
            version = "mock-1" if self.engine.mock else self.engine.model
            path = getattr(self.engine, "weights", None)
            if not self.engine.mock and path:
                try:
                    digest = hashlib.sha256()
                    with open(path, "rb") as f:
                        for block in iter(lambda: f.read(1 << 20), b""):
                            digest.update(block)
                    version = "%s-%s" % (self.engine.model, digest.hexdigest()[:12])
                except OSError:
                    pass
            self._version = version
        return self._version

    def stats(self):
        """GET /stats in the team contract's shape (API.md, v1.1), plus a few extra numbers at the end."""
        m = self.metrics()
        util, lat = m["gpu"].get("util_pct"), m["latency_ms"]
        return {"box": m["node"], "in_flight": m["inflight"], "queue_depth": m["queued"], "max_concurrency": 1,
                "gpu_util": 0.0 if util is None else round(util / 100.0, 2),
                "ewma_latency_ms": lat["ewma"] or 0.0, "p50_latency_ms": lat["p50"] or 0.0,
                "p95_latency_ms": lat["p95"] or 0.0,
                "requests_total": m["served_total"], "errors_total": m["errors_total"],
                "mode": "training" if m["busy_other"] else "normal",
                "queued_units": m["queued_units"], "ms_per_unit": m["ms_per_unit"], "slowdown": m["slowdown"],
                "model_version": self.model_version(), "mock": self.engine.mock}

    def count_error(self):
        with self.lock:
            self.errors_total += 1

    def forecast(self, body, t0):
        """POST /v1/forecast: check, queue, wait for the worker, build the answer.

        A body in the team contract's names (lat, lon, hours_ahead ...) is answered in that contract's shape."""
        v1, asked = is_v1(body), None
        if v1:
            body, asked = from_v1(body)
        try:
            req = check_forecast_request(body)
        except ApiError as e:
            raise ApiError(422 if v1 and e.status == 400 else e.status, e.message)
        req["detail"] = v1                  # the team contract also wants each member and the start values
        init_time, state = self.engine.find_init(req["init"])
        job = Job(req, init_time, state)
        with self.lock:
            self.queued += 1
            self.queued_units += job.units
            self.ids += 1
            request_id = req["request_id"] or "%s-%d" % (self.node, self.ids)
        self.jobs.put(job)
        while not job.done.wait(0.25):
            if self.stop.is_set():
                raise ApiError(503, "the server is stopping")
        if job.error is not None:
            raise job.error
        total_ms = (time.monotonic() - t0) * 1000.0
        with self.lock:
            self.served_total += 1
            self.latencies.append(total_ms)
            self.latency_ewma = total_ms if self.latency_ewma is None else 0.8 * self.latency_ewma + 0.2 * total_ms
        leads = list(range(6, req["lead_hours"] + 6, 6))
        start, members = job.series.pop("_init", None), job.series.pop("_members", None)
        point = {"lat": req["lat"], "lon": req["lon"]}
        point.update(job.series)
        answer = {"request_id": request_id, "node": self.node, "model": self.engine.model,
                  "init_time": time_text(init_time), "members": req["members"], "member_offset": req["member_offset"],
                  "leads_h": leads,
                  "valid_times": [time_text(init_time + datetime.timedelta(hours=lead)) for lead in leads],
                  "point": point, "units": job.units,
                  "timing_ms": {"queue": round((job.started - job.arrived) * 1000.0, 1), "gpu": round(job.gpu_ms, 1),
                                "total": round(total_ms, 1)}}
        if v1:
            if asked["request_id"] is None:
                asked["request_id"] = request_id
            return to_v1(answer, asked, init_time, start, members, self.model_version())
        return answer

    def store_state(self, body):
        """POST /v1/state: keep an analysis. It becomes "latest" if it is the newest."""
        when, npy_b64 = check_state_request(body)
        return {"latest": time_text(self.engine.push_state(when, npy_b64))}

    def set_fail(self, mode):
        self.fail_mode = mode
        if mode == "hang":
            self.not_hung.clear()
        else:
            self.not_hung.set()             # lets go of the requests that were hanging; they get no answer

    def admin_call(self, path, body):
        """The test hooks under /admin/. Each answers with where the hooks stand: {"mode": ..., "factor": ...}."""
        if not self.admin:
            raise ApiError(403, "the test hooks are off; start the server with --admin")
        body = {} if body is None else body
        if not isinstance(body, dict):
            raise ApiError(400, "the body must be a JSON object")
        if path == "/admin/contention":
            factor = body.get("factor")
            if (sorted(body) != ["factor"] or isinstance(factor, bool) or not isinstance(factor, (int, float))
                    or not (1.0 <= factor <= 100.0)):
                raise ApiError(400, 'the body must be {"factor": 1.8}: a number from 1.0 (no contention) to 100')
            self.engine.set_contention(float(factor))
        elif path == "/admin/fail":
            mode = body.get("mode")
            if sorted(body) != ["mode"] or not isinstance(mode, str) or mode not in FAIL_MODES:
                raise ApiError(400, 'the body must be {"mode": "down" | "hang" | "error" | "ok"}')
            self.set_fail(mode)
        elif path == "/admin/mode":         # the team contract's single switch: normal, slow, error or hang
            mode, factor = body.get("mode"), body.get("slow_factor", 3.0)
            if mode not in ("normal", "slow", "error", "hang"):
                raise ApiError(400, 'the body must be {"mode": "normal" | "slow" | "error" | "hang", "slow_factor": 3.0}')
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not (1.0 <= factor <= 100.0):
                raise ApiError(400, "slow_factor must be a number from 1.0 to 100")
            self.engine.set_contention(float(factor) if mode == "slow" else 1.0)
            self.set_fail({"normal": "ok", "slow": "ok", "error": "down", "hang": "hang"}[mode])
        elif path == "/admin/reset":
            with self.lock:
                self.reset_counters()
            self.engine.set_contention(1.0)
            self.set_fail("ok")
        else:
            raise ApiError(404, "no such test hook: %s" % show(path))
        return {"mode": self.fail_mode, "factor": self.engine.factor}


def prom_text(metrics):
    """The /metrics numbers as Prometheus text. Every name gets the prefix wx_; nested names are joined with
    "_" (wx_gpu_util_pct, wx_latency_ms_p50). A number that is unknown (null) is left out."""
    flat = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat += [("%s_%s" % (key, sub), sub_value) for sub, sub_value in value.items()]
        else:
            flat.append((key, value))
    lines = []
    for name, value in flat:
        if value is None or isinstance(value, str):         # unknown, or the node's name
            continue
        lines.append("# TYPE wx_%s %s" % (name, "counter" if name.endswith("_total") else "gauge"))
        lines.append("wx_%s %s" % (name, int(value) if isinstance(value, bool) else value))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- HTTP

GET_PATHS = ("/health", "/metrics", "/stats", "/v1/info")
POST_PATHS = ("/v1/forecast", "/v1/state", "/admin/contention", "/admin/fail", "/admin/mode", "/admin/reset")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "forecast-server"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")       # on every response, errors included
        BaseHTTPRequestHandler.end_headers(self)

    def answer(self, status, body, content_type="application/json", headers=()):
        if status == 500:
            self.server.app.count_error()
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers:
                self.send_header(name, value)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:                     # the client has gone away
            self.close_connection = True

    def answer_json(self, status, obj, headers=()):
        self.answer(status, json.dumps(obj).encode("utf-8"), headers=headers)

    def send_error(self, code, message=None, explain=None):
        """The base class calls this for a request it cannot read. Answer in JSON like everywhere else."""
        self.close_connection = True
        self.answer_json(code, {"error": message or self.responses.get(code, ("error",))[0]})

    def log_message(self, format, *args):
        pass                                # no line per request; failures are printed where they happen

    def do_GET(self):
        self.safely(self.get)

    def do_POST(self):
        self.safely(self.post)

    def do_OPTIONS(self):
        """The browser's question before a cross-origin POST (the dashboard reads this server directly)."""
        self.no_body()
        self.answer(200, b"", "text/plain", [("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                                             ("Access-Control-Allow-Headers", "*"),
                                             ("Access-Control-Max-Age", "600")])

    def safely(self, handle):
        """Run one request. Whatever goes wrong, the client gets an answer and the server lives on."""
        app = self.server.app
        try:
            if app.fail_mode == "hang" and not self.path.startswith("/admin/"):
                app.not_hung.wait()         # the test hook: accept and never answer
                self.close_connection = True
                return
            handle()
        except ApiError as e:
            self.answer_json(e.status, {"error": e.message})
        except Exception as e:
            traceback.print_exc()
            self.answer_json(500, {"error": "internal error: %s: %s" % (type(e).__name__, e)})

    def no_body(self):
        """A GET or OPTIONS should carry no body. If one is announced, do not read it: close afterwards."""
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0").strip() not in ("", "0"):
            self.close_connection = True

    def read_body(self):
        """The JSON body of a POST as a Python value; None if there is no body. ApiError(400) if unusable."""
        length = self.headers.get("Content-Length", "0").strip()
        if self.headers.get("Transfer-Encoding") or not length.isdigit() or int(length) > MAX_BODY:
            self.close_connection = True    # the body cannot be read, so this connection cannot go on
            if self.headers.get("Transfer-Encoding"):
                raise ApiError(400, "chunked bodies are not supported; send Content-Length")
            if not length.isdigit():
                raise ApiError(400, "Content-Length must be a number, got %s" % show(length))
            raise ApiError(400, "the body is too large: %s bytes; the limit is %d" % (length, MAX_BODY))
        data = self.rfile.read(int(length))
        if len(data) != int(length):
            self.close_connection = True
            raise ApiError(400, "the body is shorter than Content-Length says")
        if not data.strip():
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (ValueError, RecursionError):
            raise ApiError(400, "the body is not valid JSON")

    def wrong_method(self, allowed):
        self.answer_json(405, {"error": "use %s for %s" % (allowed, urlsplit(self.path).path)}, [("Allow", allowed)])

    def get(self):
        app = self.server.app
        self.no_body()
        url = urlsplit(self.path)
        if url.path == "/health":
            status = app.status()
            self.answer_json(200 if status == "ok" else 503, app.health(status))
        elif url.path == "/metrics":
            form = parse_qs(url.query).get("format", ["json"])[-1]
            if form not in ("json", "prom"):
                raise ApiError(400, "format must be json or prom, got %s" % show(form))
            app.check_ready()
            if form == "prom":
                self.answer(200, prom_text(app.metrics()).encode("utf-8"), "text/plain; version=0.0.4; charset=utf-8")
            else:
                self.answer_json(200, app.metrics())
        elif url.path == "/stats":
            app.check_ready()
            self.answer_json(200, app.stats())
        elif url.path == "/v1/info":
            app.check_ready()
            self.answer_json(200, app.info())
        elif url.path in POST_PATHS:
            self.wrong_method("POST")
        else:
            raise ApiError(404, "no such path: %s" % show(url.path))

    def post(self):
        app = self.server.app
        t0 = time.monotonic()
        body = self.read_body()
        path = urlsplit(self.path).path
        if path.startswith("/admin/"):
            self.answer_json(200, app.admin_call(path, body))
        elif path in ("/v1/forecast", "/v1/state"):
            if app.fail_mode == "error":
                raise ApiError(500, "this server is set to fail (test hook /admin/fail)")
            app.check_ready()
            self.answer_json(200, app.forecast(body, t0) if path == "/v1/forecast" else app.store_state(body))
        elif path in GET_PATHS:
            self.wrong_method("GET")
        else:
            raise ApiError(404, "no such path: %s" % show(path))


class Server(ThreadingHTTPServer):
    daemon_threads = True                   # a hung or waiting request never keeps the process alive
    request_queue_size = 128                # a burst of new connections waits here instead of being refused

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return                          # a client that hangs up mid-request is not worth a traceback
        ThreadingHTTPServer.handle_error(self, request, client_address)


def main(argv=None):
    p = argparse.ArgumentParser(description="The forecast server of one node (docs/CONTRACT.md, section 1).")
    p.add_argument("--node", default="A", help="this node's name, as it appears in every answer (default A)")
    p.add_argument("--host", default=None,
                   help="address to listen on (default: 127.0.0.1 with --mock; 0.0.0.0 without, so the router "
                        "can reach the node)")
    p.add_argument("--port", type=int, default=8101, help="default 8101; 0 picks a free port")
    p.add_argument("--mock", action="store_true", help="no model: emulate the work by sleeping, make the numbers up")
    p.add_argument("--admin", action="store_true", help="turn on the test hooks under /admin/")
    p.add_argument("--ms-per-unit", type=float, default=10.0,
                   help="mock mode: emulated GPU time for one member-step, in ms (default 10, as measured on a GB10)")
    p.add_argument("--latest", default=None, metavar="TIME",
                   help='mock mode: the time "latest" stands for, like 2026-11-14T00 (default: follows the clock)')
    p.add_argument("--weights", default=os.environ.get("WX_WEIGHTS"),
                   help="real mode: the model file written by weather/train.py (default $WX_WEIGHTS)")
    p.add_argument("--data", default=os.environ.get("WX_DATA"),
                   help="real mode: the folder with stats.json and the states (default $WX_DATA, else /data)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"],
                   help="real mode: where the model runs (default auto)")
    p.add_argument("--batch", type=int, default=20, help="real mode: how many members advance together (default 20)")
    args = p.parse_args(argv)
    if not 0 <= args.port <= 65535:
        p.error("--port must be from 0 to 65535")
    if args.batch < 1:
        p.error("--batch must be 1 or more")
    if not args.ms_per_unit >= 0:
        p.error("--ms-per-unit must be 0 or more")

    stop = threading.Event()
    if args.mock:
        latest = None
        if args.latest is not None:
            try:
                latest = analysis_time(args.latest, "--latest")
            except ApiError as e:
                p.error(e.message)
        engine = MockEngine(args.ms_per_unit, latest, stop)
    else:
        data_dir = args.data or "/data"
        if args.latest is not None:
            p.error("--latest only works with --mock")
        if not args.weights:
            p.error("real mode needs the model file: give --weights or set WX_WEIGHTS (or use --mock)")
        if not os.path.isfile(args.weights):
            p.error("the model file is not there: %s" % args.weights)
        if not os.path.isfile(os.path.join(data_dir, "stats.json")):
            p.error("no stats.json in the data folder %s: give --data or set WX_DATA" % data_dir)
        engine = RealEngine(args.weights, data_dir, args.device, args.batch, stop)

    app = App(args.node, engine, args.admin, stop)
    host = args.host or ("127.0.0.1" if args.mock else "0.0.0.0")
    try:
        httpd = Server((host, args.port), Handler)
    except OSError as e:
        sys.stderr.write("cannot listen on %s:%d: %s\n" % (host, args.port, e))
        return 2
    httpd.app = app

    def on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    print("listening on http://%s:%d" % (host, httpd.server_address[1]), flush=True)
    app.worker.start()                      # real mode: this is where the model loads
    serving = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, name="http", daemon=True)
    serving.start()

    while not stop.wait(1.0):               # signals arrive in this (the main) thread
        pass
    app.not_hung.set()                      # let go of hung requests
    app.jobs.put(None)                      # and of the worker
    httpd.shutdown()
    httpd.server_close()
    app.worker.join(1.0)
    if app.worker.is_alive():
        # Still inside a model call, or still loading. Its client has its 503 already. Leave now, with code 0.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
