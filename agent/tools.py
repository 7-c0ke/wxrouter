"""The weather desk's tools: plain functions that ask the router for forecasts and for the cluster's state.

Each function returns a small dict with the units in the key names: temp_c (degrees C), wind_mps (metres
per second), pressure_hpa (hPa), lead_h (hours), *_ms (milliseconds). Times ending in _utc are UTC.
"node" is the backend that answered (the router's X-Backend header). A failure comes back as
{"error": "..."}; these functions never raise.

Try one from a shell. The router must be running; set ROUTER_URL if it is not at http://127.0.0.1:8100.

    python agent/tools.py ohare_tomorrow
    python agent/tools.py forecast_point '{"lat": 41.96, "lon": -87.93, "lead_hours": 48}'

weather_mcp.py (the MCP server) and desk_agent.py (the chat loop) both use TOOLS and call_tool() from here.

Written at the hackathon (Nov 13-14, 2026).
"""
import datetime
import functools
import http.client
import json
import math
import os
import socket
import sys
import urllib.error
import urllib.request

DEFAULT_ROUTER_URL = "http://127.0.0.1:8100"
WORK_TIMEOUT_S = 60       # a forecast can wait in a queue behind other work
QUICK_TIMEOUT_S = 10      # /router/* is answered by the router itself
MAX_LEAD_H = 120          # the forecast server's limit (contract section 1)

OHARE_LAT, OHARE_LON = 41.9602, -87.9316
CHICAGO_UTC_OFFSET_H = -6           # Central Standard Time: right for November, wrong in summer (-5)
STORM_INIT = "2019-01-27T12"        # 72 hours before the coldest morning (Jan 30) of the 2019 cold wave
POLICIES = ["static", "round_robin", "least_loaded", "adaptive"]

# Never go through a proxy: the router is on this machine or on the local network.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------- talking to the router

def router_url():
    """The router's address: $ROUTER_URL, or the default. Read on every call, so it can change."""
    return (os.environ.get("ROUTER_URL") or DEFAULT_ROUTER_URL).rstrip("/")


def http_json(method, url, body=None, headers=None, timeout=WORK_TIMEOUT_S):
    """Send one JSON request. Returns (data, node, error).

    data is the answer (a dict; {} for an empty answer). node is the X-Backend header, or None. error is
    None when it worked, otherwise one plain sentence, and data is None.
    """
    send_headers = {"Accept": "application/json"}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        send_headers["Content-Type"] = "application/json"
    send_headers.update(headers or {})
    what = "%s %s" % (method, url)
    node = None
    try:
        request = urllib.request.Request(url, data=payload, headers=send_headers, method=method)
        with _OPENER.open(request, timeout=timeout) as response:
            node = response.headers.get("X-Backend")
            raw = response.read()
    except urllib.error.HTTPError as exc:       # there was an answer, but with an error status
        node = exc.headers.get("X-Backend") if exc.headers is not None else None
        return None, node, "%s answered HTTP %s: %s" % (what, exc.code, _error_text(exc))
    except (OSError, http.client.HTTPException, ValueError) as exc:     # refused, timed out, cut off, bad URL
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, socket.timeout) and getattr(reason, "errno", None) is None:
            return None, node, "%s gave no answer within %s s" % (what, timeout)     # our own time limit
        return None, node, "cannot reach %s (%s). Is the router running there?" % (url, reason)
    if not raw.strip():
        return {}, node, None                   # success with an empty body: nothing to report
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None, node, "%s did not answer with JSON; it sent %r" % (what, raw[:60])
    if not isinstance(data, dict):
        return None, node, "%s answered with JSON that is not an object" % what
    return data, node, None


def _error_text(exc):
    """The message inside an HTTP error answer: its JSON "error" field if it has one, else its text."""
    try:
        text = exc.read().decode("utf-8", "replace").strip()
    except (OSError, http.client.HTTPException):
        text = ""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    message = (data.get("error") or data.get("message")) if isinstance(data, dict) else None
    if isinstance(message, dict):               # OpenAI style: {"error": {"message": "..."}}
        message = message.get("message") or json.dumps(message)
    return str(message or text[:200] or exc.reason)


def _error(message, node=None):
    """The shape of every failure. "node" is added when a backend was named."""
    out = {"error": message}
    if node:
        out["node"] = node
    return out


def _never_raise(function):
    """Make a tool return {"error": ...} whatever goes wrong inside it (an answer in a shape we did not expect)."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            return _error("%s failed: %s: %s" % (function.__name__, type(exc).__name__, exc))
    return wrapper


# ---------------------------------------------------------------- small helpers that put up with nulls

def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _at(series, i):
    """Item i of a list, or None if there is no such item."""
    return series[i] if isinstance(series, list) and 0 <= i < len(series) else None


def _num(value, digits=1):
    """value rounded to `digits` decimals, or None if it is not a finite number (null, text, NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits) + 0.0           # "+ 0.0" turns -0.0 into 0.0


def _whole(value):
    """value rounded to a whole number, or None if it is not a finite number."""
    value = _num(value, 0)
    return None if value is None else int(value)


def _parse_time(text):
    """"2026-11-14T00" as a datetime (UTC, no tzinfo), or None if it does not look like that."""
    try:
        return datetime.datetime.strptime(str(text)[:13], "%Y-%m-%dT%H")
    except ValueError:
        return None


def _fmt_time(when):
    return when.strftime("%Y-%m-%dT%H")


def _utc_now():
    """The time now in UTC, without tzinfo. The tests swap this out."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _forecast(path, lat, lon, lead_hours, members, init, with_truth=False):
    """POST a forecast request (the body of the contract's /v1/forecast). Returns (data, node, error)."""
    body = {"init": init, "lead_hours": lead_hours, "members": members,
            "point": {"lat": lat, "lon": lon}, "with_truth": with_truth}
    data, node, err = http_json("POST", router_url() + path, body)
    if data is not None and not node:
        node = data.get("node")                 # no router in between: the forecast server names itself
    return data, node, err


def _rows(data):
    """One small dict per lead time, from a /v1/forecast or /v1/ensemble answer. Missing values are None.

    The server's names map like this: t2m_c -> temp_c, wind10_ms_mean -> wind_mps, msl_hpa_mean -> pressure_hpa.
    temp_c is member 0 (the forecast with no noise added), or the ensemble mean if member 0 is not there.
    """
    point = _dict(data.get("point"))
    init = _parse_time(data.get("init_time"))
    rows = []
    for i, lead in enumerate(_list(data.get("leads_h"))):
        lead = _whole(lead)
        valid = _at(data.get("valid_times"), i)
        if not isinstance(valid, str) and init is not None and lead is not None:
            valid = _fmt_time(init + datetime.timedelta(hours=lead))
        member0 = _num(_at(point.get("t2m_c"), i))
        mean = _num(_at(point.get("t2m_c_mean"), i))
        rows.append({
            "lead_h": lead,
            "valid_time_utc": valid,
            "temp_c": mean if member0 is None else member0,
            "temp_mean_c": mean,
            "temp_min_c": _num(_at(point.get("t2m_c_min"), i)),
            "temp_max_c": _num(_at(point.get("t2m_c_max"), i)),
            "wind_mps": _num(_at(point.get("wind10_ms_mean"), i)),
            "pressure_hpa": _num(_at(point.get("msl_hpa_mean"), i)),
            "truth_temp_c": _num(_at(point.get("truth_t2m_c"), i)),
        })
    return rows


def _pick(row, keys):
    return dict((key, row.get(key)) for key in keys)


# ---------------------------------------------------------------- the six tools

@_never_raise
def forecast_point(lat, lon, lead_hours=24, members=1, init="latest"):
    """The forecast at one place, every 6 hours out to lead_hours (POST /v1/forecast).

    init is "latest" or a time the cluster holds, such as "2019-01-27T12". With more than one member the
    rows also carry the lowest and highest member.
    """
    data, node, err = _forecast("/v1/forecast", lat, lon, lead_hours, members, init)
    if err:
        return _error(err, node)
    rows = _rows(data)
    if not rows:
        return _error("the forecast answer has no lead times", node)
    keys = ["lead_h", "valid_time_utc", "temp_c", "wind_mps", "pressure_hpa"]
    if members != 1:
        keys[3:3] = ["temp_min_c", "temp_max_c"]
    by_lead = [_pick(row, keys) for row in rows]
    return {
        "lat": lat,
        "lon": lon,
        "init_time_utc": data.get("init_time"),
        "model": data.get("model"),
        "members": members,
        "last_lead": by_lead[-1],
        "by_lead": by_lead,
        "total_ms": _whole(_dict(data.get("timing_ms")).get("total")),
        "node": node,
    }


@_never_raise
def ohare_tomorrow():
    """Tomorrow at Chicago O'Hare: the low, the high and the value nearest 6 AM local time.

    "Tomorrow" is the next calendar day in Chicago, taken as a fixed UTC-6 (right for November). The
    forecast starts from the newest analysis ("latest") and has one value every 6 hours, so the low and
    the high are the lowest and highest of tomorrow's four values (00, 06, 12 and 18 h local).
    """
    offset = datetime.timedelta(hours=CHICAGO_UTC_OFFSET_H)
    now = _utc_now()
    tomorrow = (now + offset).date() + datetime.timedelta(days=1)
    day_start = datetime.datetime.combine(tomorrow, datetime.time(0)) - offset     # tomorrow 00:00 local, as UTC
    day_end = day_start + datetime.timedelta(days=1)
    six_am = day_start + datetime.timedelta(hours=6)

    def lead_to_cover(init):
        """The lead (a multiple of 6 h) whose last value is the last one before tomorrow ends."""
        hours = (day_end - init).total_seconds() / 3600.0
        return 6 * (int(math.ceil(hours / 6.0)) - 1)

    # How old the newest analysis is, we only learn from the answer. Guess 12 hours old. If the answer's
    # init time shows that was not far enough, ask once more with the right lead.
    lead = min(MAX_LEAD_H, lead_to_cover(now - datetime.timedelta(hours=12)))
    data = node = None
    for _attempt in (1, 2):
        data, node, err = _forecast("/v1/forecast", OHARE_LAT, OHARE_LON, lead, 1, "latest")
        if err:
            return _error(err, node)
        init = _parse_time(data.get("init_time"))
        if init is None:
            return _error("the forecast answer has no usable init_time (got %r)" % (data.get("init_time"),), node)
        needed = lead_to_cover(init)
        if needed <= lead:
            break
        if needed > MAX_LEAD_H:
            return _error("the newest analysis the cluster holds (%s) is too old: reaching the end of tomorrow "
                          "needs %d hours of lead and the limit is %d. Send the cluster a newer analysis."
                          % (data.get("init_time"), needed, MAX_LEAD_H), node)
        lead = needed
    else:
        return _error("two forecasts in a row did not reach the end of tomorrow. "
                      "The nodes may hold different newest analyses.", node)

    points = []                                  # (valid time, row) for each of tomorrow's values
    for row in _rows(data):
        valid = _parse_time(row["valid_time_utc"])
        if valid is not None and row["temp_c"] is not None and day_start <= valid < day_end:
            points.append((valid, row))
    if not points:
        return _error("the forecast from %s has no values for %s in Chicago" % (data.get("init_time"), tomorrow), node)

    def describe(point, more=()):
        valid, row = point
        out = {"temp_c": row["temp_c"]}
        out.update(_pick(row, more))
        out["valid_time_utc"] = _fmt_time(valid)
        out["valid_time_local"] = (valid + offset).strftime("%Y-%m-%d %H:%M")
        return out

    low = min(points, key=lambda point: point[1]["temp_c"])
    high = max(points, key=lambda point: point[1]["temp_c"])
    morning = min(points, key=lambda point: abs((point[0] - six_am).total_seconds()))
    return {
        "place": "Chicago O'Hare",
        "lat": OHARE_LAT,
        "lon": OHARE_LON,
        "date_local": tomorrow.isoformat(),
        "local_time": "UTC%+d, a fixed offset (Chicago standard time, right for November)" % CHICAGO_UTC_OFFSET_H,
        "init_time_utc": data.get("init_time"),
        "low": describe(low),
        "high": describe(high),
        "near_6am": describe(morning, ["wind_mps"]),
        "node": node,
    }


@_never_raise
def ensemble_range(lat, lon, lead_hours=24, members=20):
    """An ensemble from the newest analysis, split across the nodes by the router (POST /v1/ensemble).

    Gives the mean, lowest and highest member at the last lead and at every lead, and the shards: which
    node ran how many members in how many milliseconds.
    """
    data, node, err = _forecast("/v1/ensemble", lat, lon, lead_hours, members, "latest")
    if err:
        return _error(err, node)
    rows = _rows(data)
    if not rows:
        return _error("the ensemble answer has no lead times", node)
    by_lead = [_pick(row, ["lead_h", "valid_time_utc", "temp_mean_c", "temp_min_c", "temp_max_c"]) for row in rows]
    shards = []
    for shard in _list(data.get("shards")):
        shard = _dict(shard)
        shards.append({"node": shard.get("backend"), "members": _whole(shard.get("members")),
                       "ms": _whole(shard.get("ms")), "retried": shard.get("retried")})
    names = []
    for shard in shards:
        if isinstance(shard["node"], str) and shard["node"] not in names:
            names.append(shard["node"])
    return {
        "lat": lat,
        "lon": lon,
        "init_time_utc": data.get("init_time"),
        "members": members,
        "last_lead": by_lead[-1],
        "by_lead": by_lead,
        "shards": shards,
        "node": "+".join(names) or node,        # every node that ran a shard, such as "A+B"
    }


@_never_raise
def storm_2019():
    """The January 2019 cold wave at Chicago O'Hare, replayed: a 72-hour forecast from 2019-01-27T12
    next to what really happened (the server's stored truth) at +24, +48 and +72 hours."""
    data, node, err = _forecast("/v1/forecast", OHARE_LAT, OHARE_LON, 72, 1, STORM_INIT, with_truth=True)
    if err:
        return _error(err, node)
    rows = dict((row["lead_h"], row) for row in _rows(data))
    if not rows:
        return _error("the forecast answer has no lead times", node)
    checks = []
    for lead in (24, 48, 72):
        row = rows.get(lead, {})
        checks.append({"lead_h": lead, "valid_time_utc": row.get("valid_time_utc"),
                       "forecast_temp_c": row.get("temp_c"), "truth_temp_c": row.get("truth_temp_c")})
    forecast, truth = checks[-1]["forecast_temp_c"], checks[-1]["truth_temp_c"]
    return {
        "event": "January 2019 cold wave at Chicago O'Hare, replayed",
        "lat": OHARE_LAT,
        "lon": OHARE_LON,
        "init_time_utc": data.get("init_time") or STORM_INIT,
        "checks": checks,
        "error_at_72h_c": None if forecast is None or truth is None else _num(forecast - truth),
        "error_is": "forecast minus truth, in degrees C (above zero: the forecast was too warm)",
        "node": node,
    }


@_never_raise
def cluster_status():
    """GET /router/status, boiled down: the policy, each pool's preferred backend and its backends'
    health and load, the counters, and the last 5 routing decisions (newest last)."""
    data, node, err = http_json("GET", router_url() + "/router/status", timeout=QUICK_TIMEOUT_S)
    if err:
        return _error(err, node)
    pools = {}
    for pool_name, pool in _dict(data.get("pools")).items():
        pool = _dict(pool)
        backends = []
        for backend in _list(pool.get("backends")):
            backend = _dict(backend)
            backends.append({
                "name": backend.get("name"),
                "healthy": backend.get("healthy"),
                "draining": backend.get("draining"),
                "healing": backend.get("healing"),
                "inflight": _whole(backend.get("inflight")),
                "latency_ms": _whole(backend.get("latency_ewma_ms")),
                "score": _num(backend.get("score"), 4),
                "busy_other": _dict(backend.get("probe_data")).get("busy_other"),
                "last_error": backend.get("last_error"),
            })
        pools[pool_name] = {"preferred": pool.get("preferred"), "backends": backends}
    decisions = []
    for decision in _list(data.get("decisions"))[-5:]:
        decision = _dict(decision)
        decisions.append({"time_utc": _clock(decision.get("ts")), "pool": decision.get("pool"),
                          "backend": decision.get("backend"), "kind": decision.get("kind"),
                          "reason": decision.get("reason")})
    return {
        "policy": data.get("policy"),
        "pools": pools,
        "counters": dict((name, _whole(count)) for name, count in _dict(data.get("counters")).items()),
        "last_decisions": decisions,
        "node": node or "router",
    }


def _clock(ts):
    """Seconds since 1970 as "HH:MM:SS" in UTC, or None."""
    try:
        return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%H:%M:%S")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


@_never_raise
def set_routing_policy(policy):
    """Switch the router's policy (POST /router/policy): static, round_robin, least_loaded or adaptive."""
    if policy not in POLICIES:
        return _error("policy must be one of: %s (got %r)" % (", ".join(POLICIES), policy))
    data, node, err = http_json("POST", router_url() + "/router/policy", {"policy": policy}, timeout=QUICK_TIMEOUT_S)
    if err:
        return _error(err, node)
    return {"policy": data.get("policy") or policy, "node": node or "router"}


# ---------------------------------------------------------------- the tool list and the one way to call a tool

def _schema(properties=None, required=()):
    """A JSON schema for a tool's input: an object with these properties and nothing else."""
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


_LAT = {"type": "number", "description": "Latitude in degrees north, -90 to 90."}
_LON = {"type": "number", "description": "Longitude in degrees east, -180 to 360 (west of Greenwich is negative)."}
_LEAD = {"type": "integer", "description": "How far ahead, in hours: 6 to 120, a multiple of 6. Default 24."}

# Each tool: its name, a one-sentence description and a JSON schema for its input.
TOOLS = [
    {"name": "forecast_point",
     "description": "Forecast the temperature (C), wind (m/s) and pressure (hPa) at one place, "
                    "every 6 hours out to lead_hours.",
     "input_schema": _schema({
         "lat": _LAT, "lon": _LON, "lead_hours": _LEAD,
         "members": {"type": "integer", "description": "Ensemble members, 1 to 40. Default 1: one plain forecast."},
         "init": {"type": "string", "description": "Where to start: \"latest\" (the default, the newest analysis) "
                                                   "or a UTC time the cluster holds, such as \"2019-01-27T12\"."},
     }, required=["lat", "lon"])},
    {"name": "ohare_tomorrow",
     "description": "Tomorrow's low, high and 6 AM temperature (C) at Chicago O'Hare, from the newest analysis.",
     "input_schema": _schema()},
    {"name": "ensemble_range",
     "description": "Run an ensemble forecast split across the nodes and give the mean, lowest and highest "
                    "temperature (C) at one place.",
     "input_schema": _schema({
         "lat": _LAT, "lon": _LON, "lead_hours": _LEAD,
         "members": {"type": "integer", "description": "Ensemble members. Default 20."},
     }, required=["lat", "lon"])},
    {"name": "storm_2019",
     "description": "Replay the January 2019 Chicago cold wave: the forecast from 2019-01-27T12 next to what "
                    "really happened, at +24, +48 and +72 hours.",
     "input_schema": _schema()},
    {"name": "cluster_status",
     "description": "Show the cluster behind the router: the routing policy, each backend's health and load, "
                    "the counters and the last 5 routing decisions.",
     "input_schema": _schema()},
    {"name": "set_routing_policy",
     "description": "Change how the router picks a node: static, round_robin, least_loaded or adaptive.",
     "input_schema": _schema({
         "policy": {"type": "string", "enum": POLICIES, "description": "The policy to switch to."},
     }, required=["policy"])},
]
TOOL_NAMES = [tool["name"] for tool in TOOLS]
_FUNCTIONS = {"forecast_point": forecast_point, "ohare_tomorrow": ohare_tomorrow,
              "ensemble_range": ensemble_range, "storm_2019": storm_2019,
              "cluster_status": cluster_status, "set_routing_policy": set_routing_policy}

_TYPE_WORDS = {"number": "a number", "integer": "a whole number", "string": "text", "boolean": "true or false"}


def _describe(value):
    """A JSON value in plain words, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return "the number %r" % (value,)
    if isinstance(value, str):
        return "the text %r" % (value,)
    return "a list" if isinstance(value, list) else "an object"


def _fits(value, kind):
    """Is this JSON value of the JSON-schema type `kind`?"""
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    return True


def check_arguments(schema, arguments):
    """Check a dict of arguments against a tool's input schema: names, required names, types, allowed values.

    Returns (clean, None) when they fit, or (None, "what is wrong"). In clean, a null for an optional name
    is left out (the default applies) and 24.0 for a whole number has become 24.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    unknown = sorted(name for name in arguments if name not in properties)
    if unknown:
        takes = ", ".join(properties) if properties else "no arguments"
        return None, "unknown argument %s. It takes: %s." % (", ".join(repr(name) for name in unknown), takes)
    clean = {}
    for name, spec in properties.items():
        value = arguments.get(name)
        if value is None:
            if name in required:
                return None, "missing required argument %r. %s" % (name, spec.get("description", ""))
            continue
        kind = spec.get("type")
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if not _fits(value, kind):
            return None, "%r must be %s, got %s" % (name, _TYPE_WORDS.get(kind, kind), _describe(value))
        if "enum" in spec and value not in spec["enum"]:
            return None, "%r must be one of: %s (got %s)" % (name, ", ".join(spec["enum"]), _describe(value))
        clean[name] = value
    return clean, None


@_never_raise
def call_tool(name, arguments=None):
    """Run one tool by name with a dict of arguments, after checking them against the tool's schema.

    Always returns a dict. Anything wrong (unknown tool, bad arguments, the router down) is {"error": "..."}.
    """
    schemas = dict((tool["name"], tool["input_schema"]) for tool in TOOLS)
    if not isinstance(name, str) or name not in schemas:
        return _error("unknown tool %r. The tools are: %s." % (name, ", ".join(TOOL_NAMES)))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _error("%s: the arguments must be a JSON object, got %s" % (name, _describe(arguments)))
    clean, problem = check_arguments(schemas[name], arguments)
    if problem:
        return _error("%s: %s" % (name, problem))
    return _FUNCTIONS[name](**clean)


def main(argv):
    """Run one tool from the command line and print its result as JSON."""
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Tools: " + ", ".join(TOOL_NAMES))
        return 0 if argv else 2
    try:
        arguments = json.loads(argv[1]) if len(argv) > 1 else {}
    except ValueError as exc:
        print("The arguments must be one JSON object, such as '{\"lat\": 41.96, \"lon\": -87.93}' (%s)." % exc,
              file=sys.stderr)
        return 2
    result = call_tool(argv[0], arguments)
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
