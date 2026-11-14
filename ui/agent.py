"""Forecast agent behind the chat panel.

Two modes:
  llm      Any OpenAI-compatible chat endpoint with tool calling: llama.cpp's
           llama-server (start it with --jinja), vllm-router, or vLLM. Set
           --llm-base-url (for example http://box-a:8080/v1) and --llm-model.
  offline  No LLM configured. A small rule-based parser covers the same tools,
           so the demo runs on a laptop. The UI labels which mode answered.

Every tool call is returned as a "step", so the chat can show which tool ran,
which box served it and how long it took. Users see the routing at work.
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

TOOLS = [
    {"type": "function", "function": {
        "name": "geocode", "description": "Look up latitude/longitude for a city name.",
        "parameters": {"type": "object", "properties": {"place": {"type": "string"}}, "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Weather forecast for a location from the GPU weather model (via the router). "
                       "Accurate up to 72 h, max 120 h. ensemble_members > 1 gives an uncertainty range.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number"},
            "hours_ahead": {"type": "integer", "minimum": 1, "maximum": 120},
            "ensemble_members": {"type": "integer", "minimum": 1, "maximum": 20}},
            "required": ["lat", "lon", "hours_ahead"]}}},
    {"type": "function", "function": {
        "name": "get_cluster_status",
        "description": "Which boxes are healthy, which one is active, routing policy, and latency.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_routing_policy",
        "description": "Change the router's rule. preferred = home box for adaptive ('' clears it).",
        "parameters": {"type": "object", "properties": {
            "policy": {"type": "string", "enum": ["static", "round_robin", "least_loaded", "adaptive", "balanced"]},
            "preferred": {"type": "string"}}, "required": ["policy"]}}},
    {"type": "function", "function": {
        "name": "simulate_box",
        "description": "Demo control: put a box in normal, slow (a training job is running) or error (down) mode.",
        "parameters": {"type": "object", "properties": {
            "box": {"type": "string", "enum": ["box-a", "box-b"]},
            "mode": {"type": "string", "enum": ["normal", "slow", "error"]},
            "slow_factor": {"type": "number"}}, "required": ["box", "mode"]}}},
]

SYSTEM = (
    "You are the forecast assistant for a two-box DGX Spark cluster. Answer weather questions "
    "with the get_forecast tool (call geocode first for place names). Keep answers short: the "
    "key numbers with units (°C, mm, m/s), any low-confidence warning past 72 h, and which box "
    "served it. For cluster questions use get_cluster_status. Only change routing or box modes "
    "when the user asks."
)


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
                out = await self.ops.set_box_mode(args["box"], args["mode"], float(args.get("slow_factor") or 2))
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
        convo = [{"role": "system", "content": SYSTEM}] + [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        steps: list[dict] = []
        headers = {"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}
        async with httpx.AsyncClient(timeout=120) as c:
            for _ in range(6):
                r = await c.post(f"{self.llm_base_url}/chat/completions", headers=headers, json={
                    "model": self.llm_model, "messages": convo, "tools": TOOLS, "temperature": 0.2})
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                if not calls:
                    return {"reply": msg.get("content") or "", "steps": steps, "mode": self.mode}
                convo.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for call in calls:
                    fn = call["function"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    step = await self.run_tool(fn["name"], args)
                    steps.append(step)
                    convo.append({"role": "tool", "tool_call_id": call.get("id", fn["name"]),
                                  "content": json.dumps(slim(step["result"]))})
        return {"reply": "I couldn't finish that in a few steps; try asking more specifically.",
                "steps": steps, "mode": self.mode}

    async def _chat_offline(self, messages: list[dict]) -> dict:
        text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = text.lower()
        steps: list[dict] = []

        box = "box-a" if re.search(r"\b(box[\s-]?a|\ba\b box)", low) else "box-b" if re.search(r"box[\s-]?b", low) else None
        policy = next((p for p in ("least_loaded", "round_robin", "static", "adaptive", "balanced")
                       if p.replace("_", " ") in low or p in low or (p == "least_loaded" and "least loaded" in low)), None)

        if box and re.search(r"\b(slow|train|fine-?tun)", low):
            steps.append(await self.run_tool("simulate_box", {"box": box, "mode": "slow", "slow_factor": 3}))
            return self._done(f"Started a simulated training job on {box}: it's now about 3× slower. "
                              "Watch the dashboard: adaptive moves new requests off it within a few seconds.", steps)
        if box and re.search(r"\b(fail|kill|down|stop|crash)", low):
            steps.append(await self.run_tool("simulate_box", {"box": box, "mode": "error"}))
            return self._done(f"{box} is now failing every request. The router retries on the other box, "
                              "so users shouldn't see errors.", steps)
        if box and re.search(r"\b(normal|recover|restore|back|heal|fix)", low):
            steps.append(await self.run_tool("simulate_box", {"box": box, "mode": "normal"}))
            return self._done(f"{box} is back to normal. If it's the preferred box, traffic returns to it shortly.", steps)
        if policy and re.search(r"\b(switch|use|set|change|policy|rule)", low):
            pref = "box-b" if policy == "adaptive" else None
            steps.append(await self.run_tool("set_routing_policy", {"policy": policy, **({"preferred": pref} if pref else {})}))
            return self._done(f"Routing rule is now **{policy}**" + (" with box-b as the preferred box." if pref else "."), steps)
        if re.search(r"\b(status|health|which box|cluster|active|routing|latency)\b", low) and not find_city(low):
            step = await self.run_tool("get_cluster_status", {})
            steps.append(step)
            s = step["result"]
            boxes = ", ".join(f"{n} {'healthy' if b['healthy'] else 'DOWN'} ({b['mode']})" for n, b in s["boxes"].items())
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
            return self._done("I can forecast for a city (for example *\"Weather in Chicago for the next 48 hours\"*) "
                              "or coordinates (*\"37.7, -122.4 next 3 days\"*). I can also show cluster status, "
                              "switch the routing rule, or simulate a slow or failed box for the demo.", steps)
        hours = parse_hours(low)
        members = 10 if re.search(r"\b(ensemble|range|uncertain|spread|confidence)", low) else 1
        f = await self.run_tool("get_forecast", {"lat": lat, "lon": lon, "hours_ahead": hours, "ensemble_members": members})
        steps.append(f)
        r = f["result"]
        if "error" in r:
            return self._done(f"The forecast failed: {r['error']}", steps)
        s = r["summary"]
        warn = " Beyond 72 h the model is less reliable, so treat the later part as a rough guide." if r["low_confidence"] else ""
        ens = f" With {members} ensemble members, temperatures span {s['ens_min_c']}–{s['ens_max_c']} °C." if members > 1 else ""
        return self._done(
            f"**{place}, next {hours} h:** {s['min_c']}–{s['max_c']} °C, {s['precip_mm']} mm of rain in total, "
            f"wind up to {s['max_wind_ms']} m/s.{ens}{warn}", steps)

    def _done(self, reply: str, steps: list[dict]) -> dict:
        return {"reply": reply, "steps": steps, "mode": self.mode}


# -------------------------------------------------------------------- helpers
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
    """What the LLM sees: drop the per-hour arrays the UI needs but the model doesn't."""
    if isinstance(result, dict) and "points" in result:
        return {k: v for k, v in result.items() if k not in ("points", "members_temp")}
    return result


Ops = Any  # implemented in server.py: forecast, cluster_status, set_policy, set_box_mode
