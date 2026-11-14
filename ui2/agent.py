"""Forecast agent behind the chat panel (ui2: a revised copy of ui/agent.py).

Two modes:
  llm      Any OpenAI-compatible chat endpoint with tool calling: llama.cpp's
           llama-server (start it with --jinja), vllm-router, or vLLM. Set
           --llm-base-url (for example http://box-a:8080/v1) and --llm-model.
  offline  No LLM configured. A small rule-based parser covers the same tools,
           so the demo runs on a laptop. The UI labels which mode answered.

Every tool call is returned as a "step", so the chat can show which tool ran,
which box served it and how long it took. Users see the routing at work.

Changed from ui/agent.py:
  - Rain is left out of the answer when the forecast has no rain numbers (our model has none).
  - A box whose supervisor is known (--stack) gets real actions: start_finetune, stop_finetune,
    kill_forecast_server. A box without one keeps simulate_box. Every reply says which it was.
  - The rule-based replies check the tool's result instead of always reporting success.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

# Approximate city coordinates for the geocode tool (offline; good to ~0.1°).
CITIES: dict[str, tuple[float, float]] = {
    "new york": (40.71, -74.01), "los angeles": (34.05, -118.24), "chicago": (41.88, -87.63),
    "houston": (29.76, -95.37), "phoenix": (33.45, -112.07), "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49), "san diego": (32.72, -117.16), "dallas": (32.78, -96.80),
    "austin": (30.27, -97.74), "san jose": (37.34, -121.89), "san francisco": (37.77, -122.42),
    "seattle": (47.61, -122.33), "denver": (39.74, -104.99), "boston": (42.36, -71.06),
    "miami": (25.76, -80.19), "atlanta": (33.75, -84.39), "washington": (38.91, -77.04),
    "minneapolis": (44.98, -93.27), "detroit": (42.33, -83.05), "portland": (45.52, -122.68),
    "las vegas": (36.17, -115.14), "salt lake city": (40.76, -111.89), "new orleans": (29.95, -90.07),
    "anchorage": (61.22, -149.90), "honolulu": (21.31, -157.86), "toronto": (43.65, -79.38),
    "vancouver": (49.28, -123.12), "montreal": (45.50, -73.57), "mexico city": (19.43, -99.13),
    "london": (51.51, -0.13), "paris": (48.86, 2.35), "berlin": (52.52, 13.40),
    "madrid": (40.42, -3.70), "rome": (41.90, 12.50), "amsterdam": (52.37, 4.90),
    "stockholm": (59.33, 18.07), "oslo": (59.91, 10.75), "zurich": (47.38, 8.54),
    "dubai": (25.20, 55.27), "mumbai": (19.08, 72.88), "delhi": (28.61, 77.21),
    "bangalore": (12.97, 77.59), "ahmedabad": (23.02, 72.57), "singapore": (1.35, 103.82),
    "tokyo": (35.68, 139.69), "seoul": (37.57, 126.98), "beijing": (39.90, 116.41),
    "shanghai": (31.23, 121.47), "hong kong": (22.32, 114.17), "sydney": (-33.87, 151.21),
    "melbourne": (-37.81, 144.96), "auckland": (-36.85, 174.76), "cape town": (-33.92, 18.42),
    "nairobi": (-1.29, 36.82), "cairo": (30.04, 31.24), "sao paulo": (-23.55, -46.63),
    "rio de janeiro": (-22.91, -43.17), "buenos aires": (-34.60, -58.38), "lima": (-12.05, -77.04),
}

REAL_TOOLS = ("start_finetune", "stop_finetune", "kill_forecast_server")


def fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    params = {"type": "object", "properties": properties}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def build_tools(boxes: list[str], stack_boxes: list[str]) -> list[dict]:
    """The tools for the LLM. Boxes with a supervisor get the real actions; the others keep simulate_box."""
    sim_boxes = [b for b in boxes if b not in stack_boxes]
    tools = [
        fn("geocode", "Look up latitude/longitude for a city name.", {"place": {"type": "string"}}, ["place"]),
        fn("get_forecast",
           "Weather forecast for a location from the GPU weather model (via the router). "
           "Accurate up to 72 h, max 120 h. ensemble_members > 1 gives an uncertainty range. "
           "The summary has precip_mm (rain) only when the model gives rain; when the result says rain is "
           "not available, do not mention rain. simulated=true means the numbers come from a mock server.",
           {"lat": {"type": "number"}, "lon": {"type": "number"},
            "hours_ahead": {"type": "integer", "minimum": 1, "maximum": 120},
            "ensemble_members": {"type": "integer", "minimum": 1, "maximum": 20}},
           ["lat", "lon", "hours_ahead"]),
        fn("get_cluster_status",
           "Which boxes are healthy, which one is active, routing policy, and latency. Per box: simulated "
           "(true = a mock server) and, where a real fine-tune job can run, its state.", {}),
        fn("set_routing_policy", "Change the router's rule. preferred = home box for adaptive ('' clears it).",
           {"policy": {"type": "string", "enum": ["static", "round_robin", "least_loaded", "adaptive", "balanced"]},
            "preferred": {"type": "string"}}, ["policy"]),
    ]
    if sim_boxes:
        tools.append(fn(
            "simulate_box",
            "SIMULATED demo control (a test hook; no real job runs and nothing really fails): put a box in "
            "normal, slow (as if a training job were running) or error (as if it were down) mode.",
            {"box": {"type": "string", "enum": sim_boxes},
             "mode": {"type": "string", "enum": ["normal", "slow", "error"]},
             "slow_factor": {"type": "number"}}, ["box", "mode"]))
    if stack_boxes:
        box = {"type": "string", "enum": stack_boxes}
        tools += [
            fn("start_finetune",
               "REAL job: start the fine-tune run on a box through its supervisor. It shares the GPU with the "
               "forecasts, so the box gets slower while it runs. steps is optional (the job has its own default).",
               {"box": box, "steps": {"type": "integer", "minimum": 1}}, ["box"]),
            fn("stop_finetune", "REAL job: stop the fine-tune run on a box.", {"box": box}, ["box"]),
            fn("kill_forecast_server",
               "REAL failure: kill the forecast server process on a box. Its supervisor starts it again by "
               "itself within seconds; the router sends requests to the other box meanwhile.",
               {"box": box}, ["box"]),
        ]
    return tools


def system_prompt(boxes: list[str], stack_boxes: list[str]) -> str:
    sim_boxes = [b for b in boxes if b not in stack_boxes]
    text = (
        "You are the forecast assistant for a two-box DGX Spark cluster. Answer weather questions "
        "with the get_forecast tool (call geocode first for place names). Keep answers short: the "
        "key numbers with units (°C, m/s), any low-confidence warning past 72 h, and which box "
        "served it. Give rain in mm only when the forecast summary has precip_mm. When the result says "
        "rain is not available, the model has no rain output: say nothing about rain, precipitation, "
        "showers or dry weather. Leave out any other number that is missing. When the result has "
        "simulated=true, the numbers come from a mock server and not from a weather model: say so in a "
        "few words. For cluster questions use get_cluster_status. Only change routing, box modes or "
        "jobs when the user asks. ")
    if stack_boxes:
        text += (f"On {', '.join(stack_boxes)} the actions are real: start_finetune and stop_finetune run a "
                 "real fine-tune job, and kill_forecast_server really kills the forecast server (its "
                 "supervisor restarts it by itself). ")
    if sim_boxes:
        text += (f"On {', '.join(sim_boxes)} simulate_box only simulates a training job (slow) or a failure "
                 "(error); nothing real runs or fails. ")
    return text + "Always say whether an action was real or simulated."


class Agent:
    def __init__(self, ops: "Ops", llm_base_url: str | None, llm_model: str | None,
                 llm_api_key: str | None = None):
        self.ops = ops
        self.llm_base_url = llm_base_url.rstrip("/") if llm_base_url else None
        self.llm_model = llm_model or "local"
        self.llm_api_key = llm_api_key

    @property
    def mode(self) -> str:
        return f"llm:{self.llm_model}" if self.llm_base_url else "offline"

    @property
    def boxes(self) -> list[str]:
        return list(self.ops.boxes)

    @property
    def stack_boxes(self) -> list[str]:
        return [b for b in self.ops.boxes if b in self.ops.stacks]

    # ---------------------------------------------------------------- tools
    async def run_tool(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        try:
            if name == "geocode":
                out = geocode(str(args.get("place", "")))
            elif name == "get_forecast":
                out = await self.ops.forecast(float(args["lat"]), float(args["lon"]),
                                              int(args.get("hours_ahead", 24)),
                                              int(args.get("ensemble_members", 1)))
            elif name == "get_cluster_status":
                out = await self.ops.cluster_status()
            elif name == "set_routing_policy":
                out = await self.ops.set_policy(args["policy"], args.get("preferred"))
            elif name == "simulate_box":
                if args["box"] in self.ops.stacks:
                    out = {"error": f"{args['box']} has a supervisor, so its actions are real: use start_finetune, "
                                    "stop_finetune or kill_forecast_server"}
                else:
                    out = await self.ops.set_box_mode(args["box"], args["mode"], float(args.get("slow_factor") or 2))
            elif name == "start_finetune":
                out = await self.ops.finetune(args["box"], "start", int(args["steps"]) if args.get("steps") else None)
            elif name == "stop_finetune":
                out = await self.ops.finetune(args["box"], "stop")
            elif name == "kill_forecast_server":
                out = await self.ops.kill_forecast(args["box"])
            else:
                out = {"error": f"unknown tool {name}"}
        except Exception as e:  # report tool errors to the model/user instead of crashing the chat
            out = {"error": f"{e.__class__.__name__}: {e}"}
        return {"tool": name, "args": args, "result": out, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ---------------------------------------------------------------- chat
    async def chat(self, messages: list[dict]) -> dict:
        if self.llm_base_url:
            try:
                return await self._chat_llm(messages)
            except httpx.HTTPError as e:
                res = await self._chat_offline(messages)
                res["note"] = f"LLM unreachable ({e.__class__.__name__}); answered offline."
                return res
        return await self._chat_offline(messages)

    async def _chat_llm(self, messages: list[dict]) -> dict:
        tools = build_tools(self.boxes, self.stack_boxes)
        convo = [{"role": "system", "content": system_prompt(self.boxes, self.stack_boxes)}] + [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        steps: list[dict] = []
        headers = {"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}
        async with httpx.AsyncClient(timeout=120) as c:
            for _ in range(6):
                r = await c.post(f"{self.llm_base_url}/chat/completions", headers=headers, json={
                    "model": self.llm_model, "messages": convo, "tools": tools, "temperature": 0.2})
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                if not calls:
                    return {"reply": msg.get("content") or "", "steps": steps, "mode": self.mode}
                convo.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for call in calls:
                    f = call["function"]
                    try:
                        args = json.loads(f.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    step = await self.run_tool(f["name"], args)
                    steps.append(step)
                    convo.append({"role": "tool", "tool_call_id": call.get("id", f["name"]),
                                  "content": json.dumps(slim(step["result"]))})
        return {"reply": "I couldn't finish that in a few steps; try asking more specifically.",
                "steps": steps, "mode": self.mode}

    async def _chat_offline(self, messages: list[dict]) -> dict:
        text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = text.lower()
        steps: list[dict] = []

        box = find_box(low, self.boxes)
        policy = next((p for p in ("least_loaded", "round_robin", "static", "adaptive", "balanced")
                       if p.replace("_", " ") in low or p in low or (p == "least_loaded" and "least loaded" in low)), None)

        # "Is box A down?" asks; it must not fail (or really kill) box A. A question about a box gets the status.
        asking = bool(re.match(r"\s*(is|are|was|were|why|what|which|how|does|did|has|have|when|who)\b", low))
        if box and not asking:
            real = box in self.ops.stacks  # a supervisor is known: the actions are real ones
            train = re.search(r"\b(slow|train|fine-?tun)", low)
            halt = re.search(r"\b(stop|cancel|halt|abort|end)\b", low)
            if train and halt:
                return await (self._stop_real(box, steps) if real else self._sim(box, "normal", steps))
            if train:
                return await (self._start_real(box, low, steps) if real else self._sim(box, "slow", steps))
            if re.search(r"\b(fail|kill|down|stop|crash)", low):
                return await (self._kill_real(box, steps) if real else self._sim(box, "error", steps))
            if re.search(r"\b(normal|recover|restore|back|heal|fix)", low):
                return await (self._recover_real(box, steps) if real else self._sim(box, "normal", steps))
        if policy and re.search(r"\b(switch|use|set|change|policy|rule)", low):
            pref = (box or ("box-b" if "box-b" in self.ops.boxes else self.boxes[-1])) if policy == "adaptive" else None
            step = await self.run_tool("set_routing_policy", {"policy": policy, **({"preferred": pref} if pref else {})})
            steps.append(step)
            if err := step["result"].get("error"):
                return self._done(f"The routing rule was not changed: {err}", steps)
            return self._done(f"Routing rule is now **{policy}**" + (f" with {pref} as the preferred box." if pref else "."), steps)
        if (re.search(r"\b(status|health|which box|cluster|active|routing|latency)\b", low) or (box and asking)) \
                and not find_city(low):
            step = await self.run_tool("get_cluster_status", {})
            steps.append(step)
            s = step["result"]
            if "error" in s:
                return self._done(f"I could not read the cluster status: {s['error']}", steps)
            boxes = ", ".join(describe_box(n, b) for n, b in s["boxes"].items())
            p95 = s.get("p95_ms")
            return self._done(f"Policy **{s['policy']}**, active box **{s['active']}**. {boxes}. "
                              f"Last 10 s: {s['rps']} req/s" + (f", p95 {p95:.0f} ms." if p95 else "."), steps)

        # Default: a forecast question.
        city = find_city(low)
        coords = re.search(r"(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)", text)
        if city:
            g = await self.run_tool("geocode", {"place": city})
            steps.append(g)
            lat, lon, place = g["result"]["lat"], g["result"]["lon"], city.title()
        elif coords:
            lat, lon, place = float(coords.group(1)), float(coords.group(2)), f"{coords.group(1)}, {coords.group(2)}"
        else:
            can = ("start or stop the real fine-tune job or kill a forecast server" if self.ops.stacks
                   else "simulate a slow or failed box for the demo")
            return self._done("I can forecast for a city (for example *\"Weather in Chicago for the next 48 hours\"*) "
                              "or coordinates (*\"37.7, -122.4 next 3 days\"*). I can also show cluster status, "
                              f"switch the routing rule, or {can}.", steps)
        hours = parse_hours(low)
        members = 10 if re.search(r"\b(ensemble|range|uncertain|spread|confidence)", low) else 1
        f = await self.run_tool("get_forecast", {"lat": lat, "lon": lon, "hours_ahead": hours, "ensemble_members": members})
        steps.append(f)
        r = f["result"]
        if "error" in r:
            return self._done(f"The forecast failed: {r['error']}", steps)
        return self._done(forecast_text(place, hours, members, r), steps)

    # ------------------------------------------------- box actions of the rule-based mode
    async def _sim(self, box: str, mode: str, steps: list[dict]) -> dict:
        """SIMULATED: the box's /admin/mode test hook."""
        step = await self.run_tool("simulate_box", {"box": box, "mode": mode, **({"slow_factor": 3} if mode == "slow" else {})})
        steps.append(step)
        if err := step["result"].get("error"):
            return self._done(f"The simulation on {box} did not work: {err}", steps)
        if mode == "slow":
            return self._done(f"Started a **simulated** training job on {box} (no real job runs): it's now about "
                              "3× slower. Watch the dashboard: adaptive moves new requests off it within a few "
                              "seconds.", steps)
        if mode == "error":
            return self._done(f"**Simulated** failure: {box} is now failing every request. The router retries on "
                              "the other box, so users shouldn't see errors.", steps)
        return self._done(f"{box} is back to normal (simulation off). If it's the preferred box, traffic returns "
                          "to it shortly.", steps)

    async def _start_real(self, box: str, low: str, steps: list[dict]) -> dict:
        m = re.search(r"(\d+)\s*steps?\b", low)
        n = int(m.group(1)) if m and int(m.group(1)) > 0 else None
        step = await self.run_tool("start_finetune", {"box": box, **({"steps": n} if n else {})})
        steps.append(step)
        if err := step["result"].get("error"):
            return self._done(f"The real fine-tune job on {box} did not start: {err}", steps)
        return self._done(f"Started the **real** fine-tune job on {box} through its supervisor"
                          + (f" ({n} steps)" if n else "") + ". This is not a simulation: it shares the GPU with "
                          "the forecasts. Its progress shows on the box card, and adaptive moves new requests "
                          "off the box if it gets clearly slower.", steps)

    async def _stop_real(self, box: str, steps: list[dict]) -> dict:
        step = await self.run_tool("stop_finetune", {"box": box})
        steps.append(step)
        if err := step["result"].get("error"):
            return self._done(f"The real fine-tune job on {box} was not stopped: {err}", steps)
        status = step["result"].get("status") or {}
        state, code = status.get("state"), status.get("exit")
        if state == "failed" and isinstance(code, int) and code < 0:   # the supervisor's word for "ended by a signal"
            how = "It is stopped now (ended by the stop signal)."
        elif state == "done":
            how = "It has ended (done)."
        elif state == "idle":
            how = "It was not running."
        else:
            how = f"Its state is now: {state}."
        return self._done(f"Asked the supervisor of {box} to stop the **real** fine-tune job. {how}", steps)

    async def _kill_real(self, box: str, steps: list[dict]) -> dict:
        step = await self.run_tool("kill_forecast_server", {"box": box})
        steps.append(step)
        if err := step["result"].get("error"):
            return self._done(f"The forecast server on {box} was not killed: {err}", steps)
        return self._done(f"Killed the forecast server on {box}. This is a **real** process kill, not a "
                          "simulation. The router retries on the other box meanwhile, and the supervisor starts "
                          "the server again by itself within a few seconds.", steps)

    async def _recover_real(self, box: str, steps: list[dict]) -> dict:
        step = await self.run_tool("get_cluster_status", {})
        steps.append(step)
        s = step["result"]
        if "error" in s:
            return self._done(f"I could not read the cluster status: {s['error']}", steps)
        job = ((s.get("boxes") or {}).get(box) or {}).get("finetune") or {}
        if job.get("state") == "running":
            return await self._stop_real(box, steps)
        up = "healthy" if ((s.get("boxes") or {}).get(box) or {}).get("healthy") else "not healthy yet"
        return self._done(f"Nothing to undo by hand on {box}: no real fine-tune job is running there, and its "
                          f"supervisor restarts the forecast server by itself. The router sees {box} as {up}.", steps)

    def _done(self, reply: str, steps: list[dict]) -> dict:
        return {"reply": reply, "steps": steps, "mode": self.mode}


# -------------------------------------------------------------------- helpers
def forecast_text(place: str, hours: int, members: int, r: dict) -> str:
    """The rule-based answer to a forecast question. A number the model did not give is left out:
    with no rain numbers there is no word about rain."""
    s = r.get("summary") or {}
    parts = []
    if s.get("min_c") is not None and s.get("max_c") is not None:
        parts.append(f"{s['min_c']}–{s['max_c']} °C")
    if s.get("precip_mm") is not None:
        parts.append(f"{s['precip_mm']} mm of rain in total")
    if s.get("max_wind_ms") is not None:
        parts.append(f"wind up to {s['max_wind_ms']} m/s")
    if not parts:
        return f"**{place}, next {hours} h:** the forecast came back without numbers."
    warn = " Beyond 72 h the model is less reliable, so treat the later part as a rough guide." if r.get("low_confidence") else ""
    ens = (f" With {members} ensemble members, temperatures span {s['ens_min_c']}–{s['ens_max_c']} °C."
           if members > 1 and s.get("ens_min_c") is not None and s.get("ens_max_c") is not None else "")
    return f"**{place}, next {hours} h:** {', '.join(parts)}.{ens}{warn}"


def describe_box(name: str, b: dict) -> str:
    """One box in the rule-based status answer: health, mode, whether it is a mock, a running real job."""
    notes = [str(b.get("mode"))]
    if b.get("simulated"):
        notes.append("simulated")
    job = b.get("finetune") or {}
    if job.get("state") == "running":
        at = f" at step {job['step']}/{job['total']}" if job.get("step") is not None and job.get("total") is not None else ""
        notes.append(f"real fine-tune job running{at}")
    return f"{name} {'healthy' if b.get('healthy') else 'DOWN'} ({', '.join(notes)})"


def find_box(text: str, names: list[str]) -> str | None:
    """The first configured box that the (lower-case) text names: "box-b", "box b", "boxb", "node b"."""
    for name in names:
        key = name.lower()
        short = re.sub(r"^(?:box|node)[\s_-]*", "", key) or key
        if re.search(rf"\b(?:box|node)[\s_-]?{re.escape(short)}\b", text):
            return name
        if len(key) > 2 and re.search(rf"(?<![\w-]){re.escape(key)}(?![\w-])", text):
            return name
    return None


def geocode(place: str) -> dict:
    key = place.strip().lower()
    if key in CITIES:
        lat, lon = CITIES[key]
        return {"place": place, "lat": lat, "lon": lon, "approximate": True}
    for name, (lat, lon) in CITIES.items():
        if key in name or name in key:
            return {"place": name.title(), "lat": lat, "lon": lon, "approximate": True}
    return {"error": f"unknown place {place!r}; give coordinates instead"}


def find_city(text: str) -> str | None:
    hits = [c for c in CITIES if re.search(rf"\b{re.escape(c)}\b", text)]
    return max(hits, key=len) if hits else None


def parse_hours(text: str) -> int:
    if m := re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)\b", text):
        return max(1, min(120, int(m.group(1))))
    if m := re.search(r"(\d+)\s*(d|day|days)\b", text):
        return max(1, min(120, 24 * int(m.group(1))))
    if "week" in text:
        return 120
    if "tomorrow" in text:
        return 48
    if "tonight" in text or "today" in text:
        return 12
    return 24


def slim(result: Any) -> Any:
    """What the LLM sees: drop the per-hour arrays the UI needs but the model doesn't. Numbers that are
    missing are dropped too, and missing rain is spelled out, so the model has nothing to make a rain
    sentence from."""
    if isinstance(result, dict) and "points" in result:
        out = {k: v for k, v in result.items() if k not in ("points", "members_temp")}
        summary = out.get("summary") or {}
        if summary.get("precip_mm") is None:
            out["rain"] = "not available: this model has no rain output, so say nothing about rain"
        out["summary"] = {k: v for k, v in summary.items() if v is not None}
        return out
    if isinstance(result, dict) and isinstance(result.get("status"), dict) and "last_lines" in result["status"]:
        # a supervisor's answer about a job: the last three lines are enough
        return {**result, "status": {**result["status"], "last_lines": result["status"]["last_lines"][-3:]}}
    return result


Ops = Any  # implemented in server.py: forecast, cluster_status, set_policy, set_box_mode, finetune, kill_forecast
