"""Forecast agent behind the chat panel of the Control Room (ui/).

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
  - The rule-based mode changes something only on a plain command about one box (see command_verb and
    box_action). A question, a remark, a "do not ..." or a sentence that names two boxes changes nothing.
  - When the LLM fails after a tool has run, the message is not run again through the rules.
  - Infrastructure questions ("why is box-a slow?", "what's wrong with the router?", "diagnose the latency")
    go to WhatIf (whatif/, policy triage: the rule first, the LLM agents only when the signal is unclear),
    in either mode, before the LLM or the rules see them. The diagnosis also goes to the Events card,
    where its remedy can be applied; the chat itself changes nothing.
  - Weather questions (a known city or coordinates) are answered by the weather model itself: geocode, then
    get_forecast through the router, and a reply made only from the model's numbers, with no LLM wording.
  - The LLM is told which tool each box command maps to: it chose drain for "kill box-a" and "recover box-a".
  - In LLM mode a question about the cluster's state is answered from get_cluster_status, never from memory.
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
ROUTING_TOOLS = ("set_routing_config", "set_routing_policy")
# The tools that change something. In LLM mode the model chooses the tools; these run only when the user's
# own last message is a plain command for it (llm_tool_refusal below).
CHANGING_TOOLS = REAL_TOOLS + ROUTING_TOOLS + ("simulate_box",)
ACTION_OF_TOOL = {"start_finetune": ("start",), "stop_finetune": ("stop",), "kill_forecast_server": ("kill",)}
ACTION_OF_MODE = {"slow": ("start",), "error": ("kill",), "normal": ("recover", "stop")}     # simulate_box
MAX_STEPS = 1_000_000   # the most steps a fine-tune start may ask for: the bound of POST /api/box/{box}/train
STOP_EXIT = 3           # weather/finetune.py: "3 = stopped early by a signal" (it caught the signal and left)
DIAG_POLICY = "triage"  # whatif.scheduler policy for questions from the chat: no tokens when the signal is clear


class LLMFailed(Exception):
    """The LLM endpoint gave no usable answer: no connection, an HTTP error, or JSON of another shape."""


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
            "ensemble_members": {"type": "integer", "minimum": 1, "maximum": 20},
            "place": {"type": "string", "description": "Place name to show with the forecast"}},
           ["lat", "lon", "hours_ahead"]),
        fn("get_cluster_status",
           "Which boxes are healthy, which one is active, routing policy, and latency. Per box: simulated "
           "(true = a mock server) and, where a real fine-tune job can run, its state (stopped = it was "
           "stopped on request, failed = it broke). supervisor 'no answer' means the job's state is not "
           "known: do not say that no job is running.", {}),
        fn("set_routing_config",
           "Change the router's settings; send only the fields to change. policy = the rule. preferred = home "
           "box for adaptive and the one box static uses ('' clears it). margin and return_margin are "
           "fractions (0.25 = 25%); window_s and cooldown_s seconds; stall_ms milliseconds. drain = "
           "{box: true} takes a box out of service (no new requests), false puts it back. reset = true goes "
           "back to the start-up settings.",
           {"policy": {"type": "string", "enum": ["static", "round_robin", "least_loaded", "adaptive", "balanced"]},
            "preferred": {"type": "string", "enum": ["", *boxes]},
            "margin": {"type": "number", "minimum": 0, "maximum": 0.95},
            "window_s": {"type": "number", "minimum": 0, "maximum": 30},
            "cooldown_s": {"type": "number", "minimum": 0, "maximum": 120},
            "return_margin": {"type": "number", "minimum": 0, "maximum": 1},
            "stall_ms": {"type": "number", "minimum": 100, "maximum": 10000},
            "drain": {"type": "object", "properties": {b: {"type": "boolean"} for b in boxes}},
            "reset": {"type": "boolean"}}),
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
               {"box": box, "steps": {"type": "integer", "minimum": 1, "maximum": MAX_STEPS}}, ["box"]),
            fn("stop_finetune",
               "REAL job: stop the fine-tune run on a box. In the result, state 'failed' with exit 3 or an "
               "exit below 0 (stopped: true) means it stopped as asked: say stopped, not failed.",
               {"box": box}, ["box"]),
            fn("kill_forecast_server",
               "REAL failure: kill the forecast server process on a box. Its supervisor starts it again by "
               "itself within seconds; the router sends requests to the other box meanwhile.",
               {"box": box}, ["box"]),
        ]
    return tools


def system_prompt(boxes: list[str], stack_boxes: list[str], units: str = "C") -> str:
    sim_boxes = [b for b in boxes if b not in stack_boxes]
    text = (
        "You are the forecast assistant for a two-box DGX Spark cluster. Answer weather questions "
        "with the get_forecast tool (call geocode first for place names, and pass the place name). "
        "The user sees a weather card, so keep the text to one or two sentences: the range and wind in "
        "the user's units (" + ("°F, mph" if units == "F" else "°C, km/h") + "), humidity only when the "
        "forecast summary has a number for humidity_avg_pct (when it is null, say nothing about humidity), any "
        "low-confidence warning past 72 h, and which box served it. The model steps every 6 h; hourly "
        "values in between are interpolated. Give rain in mm only when the forecast summary has precip_mm. When the result says "
        "rain is not available, the model has no rain output: say nothing about rain, precipitation, "
        "showers or dry weather. Leave out any other number that is missing. When the result has "
        "simulated=true, the numbers come from a mock server and not from a weather model: say so in a "
        "few words. For any question about the cluster's state (the active box, the routing rule or its settings, box health, jobs, latency) call get_cluster_status and answer from its result, even when the same question was answered before: never answer it from memory. Only change routing, box modes or "
        "jobs when the user's last message is a plain command for it; a tool that answers \"not run\" "
        "changed nothing, so say that and what the user can type instead. ")
    if stack_boxes:
        text += (f"On {', '.join(stack_boxes)} the actions are real: start_finetune and stop_finetune run a "
                 "real fine-tune job, and kill_forecast_server really kills the forecast server (its "
                 "supervisor restarts it by itself). There, \"kill\", \"fail\" or \"crash\" a box means "
                 "kill_forecast_server; \"recover\" it means nothing to do, its supervisor restarts it. ")
    if sim_boxes:
        text += (f"On {', '.join(sim_boxes)} simulate_box only simulates a training job (slow) or a failure "
                 "(error); nothing real runs or fails. There, \"kill\", \"fail\" or \"crash\" a box means "
                 "simulate_box mode error, \"slow\" or \"train\" means mode slow, and \"recover\", \"heal\" or "
                 "\"back to normal\" means mode normal. ")
    text += ("set_routing_config with drain is only for \"drain\" and \"undrain\"; never use it for kill, fail or "
             "recover. ")
    return text + "Always say whether an action was real or simulated."


GO_AHEAD = "Go ahead: call the tools now, then answer with what they return."
# A reply that only says what the model is about to do. "Let me know ..." is a normal closing line, not that.
ANNOUNCE = re.compile(r"\b(let me (?!know)|i'?ll (check|look|get|fetch|start|call|find)|i will (check|look|get|fetch|"
                      r"start|call|find)|i need to (check|look|get|fetch|find)|i'?m going to|i am going to|"
                      r"first,? i|one moment)", re.I)


def announces_action(text: str) -> bool:
    """True when a reply with no tool call is only an announcement of tool use, not an answer."""
    t = text.strip()
    return bool(t) and len(t) < 400 and bool(ANNOUNCE.search(t))


class Agent:
    def __init__(self, ops: "Ops", llm_base_url: str | None, llm_model: str | None,
                 llm_api_key: str | None = None, llm_thinking: bool = True):
        self.ops = ops
        self.llm_base_url = llm_base_url.rstrip("/") if llm_base_url else None
        self.llm_model = llm_model or "local"
        self.llm_api_key = llm_api_key
        self.units = "C"
        # False: ask the server to skip the model's thinking phase (llama.cpp and vLLM take
        # chat_template_kwargs; a server that does not know the field may refuse the request).
        self.llm_thinking = llm_thinking
        # async (policy, source=...) -> diagnosis dict, set by the page server (ui/server.py run_diagnosis).
        # Without it infrastructure questions go to the LLM or the rules like any other.
        self.diagnoser = None

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
                                              int(args.get("ensemble_members", 1)), args.get("place"))
            elif name == "get_cluster_status":
                out = await self.ops.cluster_status()
            elif name in ("set_routing_config", "set_routing_policy"):  # the old name still works
                fields = ("policy", "preferred", "margin", "window_s", "cooldown_s", "return_margin",
                          "stall_ms", "drain", "reset")
                update = {k: args[k] for k in fields if args.get(k) is not None}
                out = (await self.ops.set_config(update)) if update else {"error": "no setting given"}
            elif name == "simulate_box":
                if args["box"] in self.ops.stacks:
                    out = {"error": f"{args['box']} has a supervisor, so its actions are real: use start_finetune, "
                                    "stop_finetune or kill_forecast_server"}
                else:
                    out = await self.ops.set_box_mode(args["box"], args["mode"], float(args.get("slow_factor") or 2))
            elif name == "start_finetune":
                try:
                    n = clean_steps(args.get("steps"))      # the same bound as the HTTP endpoint's
                except ValueError as e:
                    out = {"error": str(e)}
                else:
                    out = await self.ops.finetune(args["box"], "start", n)
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
    async def chat(self, messages: list[dict], units: str = "C") -> dict:
        self.units = units
        last_user = str(next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "") or "")
        if self.diagnoser and infra_question(last_user):
            return await self._diagnose()
        if weather_question(last_user):
            return await self._weather(last_user)
        if not self.llm_base_url:
            return await self._chat_offline(messages)
        steps: list[dict] = []      # _chat_llm adds each tool step here, so the steps outlive an LLM failure
        try:
            return await self._chat_llm(messages, steps)
        except LLMFailed as e:
            if steps:
                # A tool has run. The rules must not get the same message: with --stack that would be a second
                # real kill, or a second start that then reads "did not start".
                done = "; ".join(step_text(s) for s in steps)
                return {"reply": f"The LLM stopped answering ({e}) before it wrote its reply. What ran before "
                                 f"that: {done}. Nothing was run a second time.",
                        "steps": steps, "mode": self.mode, "note": f"LLM error ({e}); the steps shown did run."}
            res = await self._chat_offline(messages)    # nothing has run yet: the rules may answer
            res["note"] = f"LLM unreachable ({e}); answered offline."
            return res

    async def _chat_llm(self, messages: list[dict], steps: list[dict]) -> dict:
        tools = build_tools(self.boxes, self.stack_boxes)
        convo = [{"role": "system", "content": system_prompt(self.boxes, self.stack_boxes, getattr(self, "units", "C"))}] + [
            {"role": m["role"], "content": m.get("content") or ""} for m in messages
            if m.get("role") in ("user", "assistant")]
        headers = {"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}
        last_user = str(next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "") or "")
        nudged = 0
        async with httpx.AsyncClient(timeout=120) as c:
            for _ in range(6):
                try:
                    body = {"model": self.llm_model, "messages": convo, "tools": tools, "temperature": 0.2}
                    if not self.llm_thinking:
                        body["chat_template_kwargs"] = {"enable_thinking": False}
                    r = await c.post(f"{self.llm_base_url}/chat/completions", headers=headers, json=body)
                    r.raise_for_status()
                    msg = r.json()["choices"][0]["message"]
                    calls = msg.get("tool_calls") or []
                    todo = [(call.get("id"), call["function"]["name"], call["function"].get("arguments"))
                            for call in calls]
                except (httpx.HTTPError, ValueError, LookupError, TypeError, AttributeError) as e:
                    raise LLMFailed(e.__class__.__name__) from e
                if not todo and nudged < 2 and announces_action(strip_think(msg.get("content") or "")):
                    # The model said what it will do ("Let me get their coordinates") but called no tool. Without
                    # this the user gets that sentence as the answer. Ask it once more to act; at most twice.
                    nudged += 1
                    convo.append({"role": "assistant", "content": strip_think(msg.get("content") or "")})
                    convo.append({"role": "user", "content": GO_AHEAD})
                    continue
                if not todo:
                    return {"reply": only_given_numbers(strip_think(msg.get("content") or ""), steps),
                            "steps": steps, "mode": self.mode}
                convo.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for call_id, name, raw in todo:
                    try:
                        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    refusal = llm_tool_refusal(name, args, last_user, self.boxes)
                    if refusal:     # the model asked for a change that the user's own words do not ask for
                        step = {"tool": name, "args": args, "result": {"error": refusal, "not_run": True}, "ms": 0.0}
                    else:
                        step = await self.run_tool(name, args)
                    steps.append(step)
                    convo.append({"role": "tool", "tool_call_id": call_id or name,
                                  "content": json.dumps(slim(step["result"]))})
        return {"reply": "I couldn't finish that in a few steps; try asking more specifically.",
                "steps": steps, "mode": self.mode}

    async def _chat_offline(self, messages: list[dict]) -> dict:
        text = str(next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "") or "")
        low = text.lower()
        steps: list[dict] = []

        places = {n: box_mentions(low, n) for n in self.boxes}
        named = [n for n in self.boxes if places[n]]            # every configured box that the text names
        policy = POLICY.search(low)
        # Only a command changes anything: with --stack a change is a real kill or a real job. A question
        # ("Hey, is box A down?"), a remark ("I think box A is down") or a "do not kill box A" gets the status.
        cmd = command_verb(low)
        two = (f"That names more than one box ({' and '.join(named)}), and I will not guess which one you mean. "
               "Nothing was changed. Say it again with one box only.")

        if cmd and policy and policy_command(cmd, low, policy):
            name = re.sub(r"[\s-]", "_", policy.group(1))
            pref = None
            if name == "adaptive":
                if len(named) > 1:
                    return self._done(two, steps)
                pref = named[0] if named else ("box-b" if "box-b" in self.ops.boxes else self.boxes[-1])
            step = await self.run_tool("set_routing_config", {"policy": name, **({"preferred": pref} if pref else {})})
            steps.append(step)
            if err := step["result"].get("error"):
                return self._done(f"The routing rule was not changed: {err}", steps)
            return self._done(f"Routing rule is now **{name}**" + (f" with {pref} as the preferred box." if pref else "."), steps)

        if cmd and cmd.group(1) in ("drain", "undrain") and named:
            if len(named) > 1:
                return self._done(two, steps)
            on = cmd.group(1) == "drain" and not re.search(r"\bback\b", low)
            step = await self.run_tool("set_routing_config", {"drain": {named[0]: on}})
            steps.append(step)
            if err := step["result"].get("error"):
                return self._done(f"Nothing was changed: {err}", steps)
            return self._done(f"{named[0]} is **{'drained' if on else 'back in service'}**: "
                              + ("it gets no new requests (it is still the last resort if the other box fails)."
                                 if on else "it gets traffic again under the current rule."), steps)

        if cmd and cmd.group(1) == "reset" and re.search(r"\b(?:rout\w*|settings?|config\w*)\b", low):
            step = await self.run_tool("set_routing_config", {"reset": True})
            steps.append(step)
            if err := step["result"].get("error"):
                return self._done(f"Nothing was changed: {err}", steps)
            return self._done(f"Routing is back to its start-up settings: rule **{step['result'].get('policy')}**, "
                              "no box drained.", steps)

        setting = parse_setting(low) if cmd and cmd.group(1) in ("set", "change", "make") else None
        if setting:
            key, value = setting
            step = await self.run_tool("set_routing_config", {key: value})
            steps.append(step)
            if err := step["result"].get("error"):
                return self._done(f"Nothing was changed: {err}", steps)
            return self._done(f"Router setting **{key}** is now **{fmt_setting(key, value)}**.", steps)

        action = box_action(cmd, low, [i for n in named for i in places[n]]) if cmd and named else None
        if action:
            if len(named) > 1:      # the words do not say which box: never act on the wrong one
                return self._done(two, steps)
            box = named[0]
            real = box in self.ops.stacks   # a supervisor is known: the actions are real ones
            if real and "simulat" in low:
                return self._done(f"{box} has a supervisor, so nothing is simulated there: its actions are real. "
                                  f"Nothing was changed. For the real thing say it without \"simulate\", for "
                                  f"example \"fail {box}\" or \"start a training job on {box}\".", steps)
            if action == "start":
                return await (self._start_real(box, low, steps) if real else self._sim(box, "slow", steps))
            if action == "stop":
                return await (self._stop_real(box, steps) if real else self._sim(box, "normal", steps))
            if action == "kill":
                return await (self._kill_real(box, steps) if real else self._sim(box, "error", steps))
            return await (self._recover_real(box, steps) if real else self._sim(box, "normal", steps))

        if (re.search(r"\b(status|health|which box|cluster|active|routing|latency|policy|rule)\b", low)
                or named or policy) and not find_city(low):
            step = await self.run_tool("get_cluster_status", {})
            steps.append(step)
            s = step["result"]
            if "error" in s:
                return self._done(f"I could not read the cluster status: {s['error']}", steps)
            boxes = ", ".join(describe_box(n, b) for n, b in s["boxes"].items())
            p95 = s.get("p95_ms")
            reply = (f"Policy **{s['policy']}**, active box **{s['active']}**. {boxes}. "
                     f"Last 10 s: {s['rps']} req/s" + (f", p95 {p95:.0f} ms." if p95 else "."))
            if named and not QUESTION.search(low):
                # It named a box and was not a question: say plainly that it was not taken as a command.
                reply += (f" Nothing was changed: I act only on a plain command about one box, such as "
                          f"\"fail {named[0]}\" or \"stop the training job on {named[0]}\".")
            return self._done(reply, steps)

        # Default: a forecast question.
        res = await self._forecast(text, steps)
        if res is not None:
            return res
        can = ("start or stop the real fine-tune job or kill a forecast server" if self.ops.stacks
               else "simulate a slow or failed box for the demo")
        return self._done("I can forecast for a city (for example *\"Weather in Chicago for the next 48 hours\"*) "
                          "or coordinates (*\"37.7, -122.4 next 3 days\"*). I can also show cluster status, "
                          f"switch the routing rule, or {can}.", steps)

    async def _forecast(self, text: str, steps: list[dict]) -> dict | None:
        """The weather model's forecast for the place in `text` (a known city or "lat, lon"), as the chat reply;
        None when the text names no place."""
        low = text.lower()
        city = find_city(low)
        coords = COORDS.search(text)
        if city:
            g = await self.run_tool("geocode", {"place": city})
            steps.append(g)
            lat, lon, place = g["result"]["lat"], g["result"]["lon"], city.title()
        elif coords:
            lat, lon, place = float(coords.group(1)), float(coords.group(2)), f"{coords.group(1)}, {coords.group(2)}"
        else:
            return None
        hours = parse_hours(low)
        members = 10 if re.search(r"\b(ensemble|range|uncertain|spread|confidence)", low) else 1
        f = await self.run_tool("get_forecast", {"lat": lat, "lon": lon, "hours_ahead": hours,
                                                 "ensemble_members": members, "place": place})
        steps.append(f)
        r = f["result"]
        if "error" in r:
            return self._done(f"The forecast failed: {r['error']}", steps)
        return self._done(forecast_text(place, hours, members, r, self.units), steps)

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
        m = re.search(r"(\d+)[\s-]*steps?\b", low)
        n = int(m.group(1)) if m else None      # as written: the tool refuses 0 and anything above MAX_STEPS
        step = await self.run_tool("start_finetune", {"box": box, **({"steps": n} if n is not None else {})})
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
        state = status.get("state")
        if was_stopped(status):     # the supervisor says "failed" for it: exit below 0, or finetune.py's own 3
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
        b = (s.get("boxes") or {}).get(box)
        if not isinstance(b, dict):
            return self._done(f"The cluster status says nothing about {box}, so I cannot tell what state it is in. "
                              "Nothing was changed.", steps)
        if b.get("supervisor") == "no answer":
            return self._done(f"The supervisor of {box} did not answer, so I do not know whether a fine-tune job "
                              "is running there or whether its forecast server will be started again. Nothing "
                              "was changed.", steps)
        job = b.get("finetune") or {}
        if job.get("state") == "running":
            return await self._stop_real(box, steps)
        up = "healthy" if b.get("healthy") else "not healthy yet"
        return self._done(f"Nothing to undo by hand on {box}: no real fine-tune job is running there, and its "
                          f"supervisor restarts the forecast server by itself. The router sees {box} as {up}.", steps)

    async def _weather(self, text: str) -> dict:
        """A weather question: the weather model answers (through the router); the reply is its numbers."""
        steps: list[dict] = []
        res = await self._forecast(text, steps)
        if res is None:     # weather_question saw a place, so this does not happen; keep the chat answering
            return self._done("Which place? Name a city or give coordinates.", steps)
        low = text.lower()
        if re.search(r"\b(?:why|how\s+come|explain)\b", low):
            res["reply"] += " The weather model gives the forecast numbers; it does not say why the weather is so."
        res["mode"] = "weather-model"
        served = next((s["result"].get("served_by") for s in steps if s["tool"] == "get_forecast"
                       and isinstance(s["result"], dict)), None)
        res["note"] = ("Answered by the weather model" + (f" on {served}" if served else "")
                       + " through the router: the numbers are the model's, no LLM.")
        return res

    async def _diagnose(self) -> dict:
        """An infrastructure question: answer with a WhatIf diagnosis of the last minute."""
        t0 = time.perf_counter()
        try:
            out = await self.diagnoser(DIAG_POLICY, source="chat")
        except Exception as e:  # not enough telemetry, no live router, the LLM failed
            why = getattr(e, "message", None) or f"{e.__class__.__name__}: {e}"
            step = {"tool": "whatif_diagnose", "args": {"policy": DIAG_POLICY}, "result": {"error": why},
                    "ms": round((time.perf_counter() - t0) * 1000, 1)}
            return {"reply": f"That is a question about the infrastructure, so I asked WhatIf, but it cannot diagnose "
                             f"right now: {why}.", "steps": [step], "mode": "whatif",
                    "note": "Routed to WhatIf incident diagnosis."}
        d = out["diagnosis"]
        step = {"tool": "whatif_diagnose", "args": {"policy": out["policy"]},
                "result": {"incident": out["incident"], "ranking": d["ranking"][:3], "tokens": d["tokens"],
                           "llm_calls": d["llm_calls"]}, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        by = (f"the WhatIf agents on {out.get('engine') or 'the LLM'}, {d['tokens']} tokens in {d['llm_calls']} LLM calls"
              if d["llm_calls"] else "the rule (no LLM), 0 tokens")
        return {"reply": diagnosis_reply(out), "steps": [step], "mode": f"whatif:{out['policy']}",
                "note": f"Answered by WhatIf ({out['policy']}: the rule first, agents only when unclear): {by}."}

    def _done(self, reply: str, steps: list[dict]) -> dict:
        return {"reply": reply, "steps": steps, "mode": self.mode}


# -------------------------------------------------------------------- helpers
# ------------------------------------------------------------------ infrastructure questions -> WhatIf
# Asks for a cause: "why ...", "what's wrong", "diagnose", "root cause". STRONG ones need nothing else; the
# others must also name something in the infrastructure (INFRA).
STRONG_DIAG = re.compile(r"\b(?:diagnos\w*|root[\s-]?causes?|troubleshoot\w*|what(?:['\u2019]?s|\s+is)\s+(?:wrong|going\s+on)|"
                         r"what\s+went\s+wrong|(?:any|some)thing\s+wrong|what\s+happened)\b")
DIAG = re.compile(r"\b(?:why|how\s+come|explain|investigate|cause[ds]?)\b")
INFRA = re.compile(r"\b(?:slow\w*|latenc\w*|p95|lag\w*|errors?|503s?|fail\w*|down|outage|hang\w*|hung|stall\w*|"
                   r"time-?outs?|timing\s+out|unhealthy|degraded|broken|unresponsive|incident|alerts?|router|routing|"
                   r"switch\w*|failover|box[\s_-]?[ab]|boxes|cluster|servers?|gpus?|queue\w*|traffic|requests?|"
                   r"stack|infra\w*|service)\b")
# A weather question ("why is it so cold in Chicago?") stays a weather question unless it names a fault.
WEATHER = re.compile(r"\b(?:weather|temperatures?|rain\w*|wind\w*|humid\w*|snow\w*|sunny|cloud\w*|hot|cold|warm\w*)\b")
FAULT = re.compile(r"\b(?:slow\w*|latenc\w*|errors?|503s?|fail\w*|down|hang\w*|stall\w*|time-?outs?|broken|unhealthy)\b")


def infra_question(text: str) -> bool:
    """True for a question about why the service misbehaves, which WhatIf answers. Commands, status questions
    ("which box is active?") and weather questions are not."""
    low = text.lower().strip()
    if not low or command_verb(low):
        return False
    if not (STRONG_DIAG.search(low) or (DIAG.search(low) and INFRA.search(low))):
        return False
    if (WEATHER.search(low) or find_city(text)) and not FAULT.search(low):
        return False
    return True


COORDS = re.compile(r"(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)")
CLUSTER_WORDS = re.compile(r"\b(?:box[\s_-]?[ab]|boxes|router|routing|cluster|policy|rule|served|serving|gpus?|latency)\b")


def weather_question(text: str) -> bool:
    """True for a weather question the weather model answers: it names a known city or "lat, lon", and is not a
    command or a question about the cluster ("which box served Chicago?")."""
    low = text.lower().strip()
    if not low or command_verb(low) or CLUSTER_WORDS.search(low):
        return False
    return bool(find_city(low) or COORDS.search(text))


def diagnosis_reply(out: dict) -> str:
    """The chat text for a WhatIf diagnosis (server.py run_diagnosis): the cause, the evidence, the remedy."""
    d = out["diagnosis"]
    top, rest = d["ranking"][0], d["ranking"][1:3]
    meaning = f" ({out['meaning'][0].lower()}{out['meaning'][1:]})" if out.get("meaning") else ""
    lines = [f"**WhatIf diagnosis #{out['incident']}**: most likely **{top}**{meaning}"
             + (f"; next {', '.join(rest)}." if rest else ".")]
    evidence = [t.split("] ", 1)[1] for t in d.get("trail") or []
                if t.startswith(("[rule] ", "[judge] ")) and "\n" not in t]
    if evidence:
        lines.append("Evidence: " + "; ".join(evidence[:4]) + ".")
    if out.get("can_apply"):
        lines.append(f"Suggested remedy (not applied): {out['remedy']}. To apply it, use **Apply remedy** on "
                     f"incident #{out['incident']} in the Events card.")
    else:
        lines.append(f"No automatic remedy: {out['remedy']}.")
    return "\n".join(lines)


def forecast_text(place: str, hours: int, members: int, r: dict, units: str = "C") -> str:
    """The rule-based answer to a forecast question, in the user's units. A number the model did
    not give is left out: with no rain numbers there is no word about rain."""
    s = r.get("summary") or {}
    T, W = temp_fmt(units), wind_fmt(units)
    parts = []
    if s.get("min_c") is not None and s.get("max_c") is not None:
        parts.append(f"{T(s['min_c'])} to {T(s['max_c'])}")
    if s.get("max_wind_ms") is not None:
        parts.append(f"wind up to {W(s['max_wind_ms'])}")
    if s.get("humidity_avg_pct") is not None:
        parts.append(f"humidity around {s['humidity_avg_pct']}%")
    if not parts:
        return f"**{place}, next {hours} h:** the forecast came back without numbers."
    rain = "" if s.get("precip_mm") is None else (
        " No rain expected." if s["precip_mm"] == 0 else f" {s['precip_mm']} mm of rain in total.")
    warn = " Beyond 72 h the model is less reliable, so treat the later part as a rough guide." if r.get("low_confidence") else ""
    ens = (f" Across {members} ensemble members: {T(s['ens_min_c'])} to {T(s['ens_max_c'])}."
           if members > 1 and s.get("ens_min_c") is not None and s.get("ens_max_c") is not None else "")
    sim = " (Simulated data from a mock server.)" if r.get("simulated") else ""
    return f"**{place}, next {hours} h:** {', '.join(parts)}.{rain}{ens}{warn}{sim}"


def describe_box(name: str, b: dict) -> str:
    """One box in the rule-based status answer: health, mode, whether it is a mock, a running real job."""
    notes = [str(b.get("mode"))]
    if b.get("simulated"):
        notes.append("simulated")
    if b.get("drained"):
        notes.append("drained: no new requests")
    job = b.get("finetune") or {}
    if b.get("supervisor") == "no answer":
        notes.append("supervisor not reachable")
    elif job.get("state") == "running":
        at = f" at step {job['step']}/{job['total']}" if job.get("step") is not None and job.get("total") is not None else ""
        notes.append(f"real fine-tune job running{at}")
    return f"{name} {'healthy' if b.get('healthy') else 'DOWN'} ({', '.join(notes)})"


def box_mentions(text: str, name: str) -> list[int]:
    """Where the (lower-case) text names this box: "box-b", "box b", "boxb", "node b". The start of each place."""
    key = name.lower()
    short = re.sub(r"^(?:box|node)[\s_-]*", "", key) or key
    hits = [m.start() for m in re.finditer(rf"\b(?:box|node)[\s_-]?{re.escape(short)}\b", text)]
    if len(key) > 2:
        hits += [m.start() for m in re.finditer(rf"(?<![\w-]){re.escape(key)}(?![\w-])", text)]
    return sorted(set(hits))


def find_box(text: str, names: list[str]) -> str | None:
    """The first configured box that the (lower-case) text names."""
    return next((n for n in names if box_mentions(text, n)), None)


# ---------------------------------------------- how the rule-based mode reads a command
# A command begins with its verb, after polite words at most ("please", "can you", "hey" ...). Every
# alternative before the verb starts with a different word and ends in separators, so a long
# "can you please can you please ..." is read in linear time (an optional "please" inside the "can you"
# part would make it exponential).
COMMAND = re.compile(
    r"\W*(?:(?:please|now|ok|okay|then|and|so|hey|just|go\s+ahead\s+and)\W+|(?:can|could|would|will)\s+you\W+)*"
    r"(start|run|launch|begin|train|fine[\s-]?tune|simulate|slow|stop|cancel|halt|abort|end|kill|terminate|pause|"
    r"quit|fail|crash|shut|recover|restore|heal|fix|take|bring|put|set|make|turn|switch|use|change|"
    r"drain|undrain|reset)\b")
# With one of these the sentence is not a plain command: a "not", or a condition the rules cannot check.
HEDGE = re.compile(r"\b(?:not|never|dont|cannot|if|whether|unless|when|once|until)\b|n['\u2019]t\b")
ASKING = re.compile(r"\b(?:is|are|was|were|does|do|did|should|shall|what|why|how|which|who|or)\b")
QUESTION = re.compile(r"\?|^\W*(?:is|are|was|were|why|what|which|how|does|do(?!\s+not\b)|did|has|have|when|who|where|can|"
                      r"could|would|will|should|any)\b")
POLICY = re.compile(r"\b(least[\s_-]loaded|round[\s_-]robin|static|adaptive|balanced)\b")
JOB = re.compile(r"\b(?:slow|train|fine[\s-]?tun|jobs?\b|runs?\b)")    # the training job, or its simulation ("slow mode")
SERVER = re.compile(r"(?:forecast\s+)?(?:server|process|service)\b")
MODEL = re.compile(r"(?:weather\s+)?model\b")
# Between the verb and what it is aimed at: a particle ("take DOWN box A") and words such as "the", "a", "real".
BEFORE_OBJECT = re.compile(r"\s+(?:(?:down|off|on|back|up|out)\s+)?(?:(?:the|a|an|that|this|its|my|our|your|real|"
                           r"simulated|fake|running|current|new|another|short|quick|small|\d+[\s-]*steps?)\s+)*")


def command_verb(low: str) -> re.Match | None:
    """The verb that a plain command begins with (group 1 of the match), or None when the lower-case text is
    not one: it does not begin with a command verb, it has a "not" or an "if", or it asks."""
    m = COMMAND.match(low)
    if not m or HEDGE.search(low):
        return None
    if "?" in low:
        # "Fail box A?" asks. "Can you fail box A?" is a request, unless something more is asked after it.
        polite = re.search(r"\b(?:can|could|would|will)\s+you\b", low[:m.start(1)])
        if not polite or ASKING.search(low, m.end(1)):
            return None
    return m


# "set the cooldown to 5 s", "set margin to 30%", "change the return margin to 0.1", "set stall to 800 ms"
SETTING = re.compile(r"\b(cooldown|return[\s_-]margin|margin|window|stall)(?:[\s_-]?(?:time|ms|s))?\s+"
                     r"(?:to|=|at)\s+(\d+(?:\.\d+)?)\s*(%|ms|milliseconds?|s|secs?|seconds?)?(?!\w)")
SETTING_KEYS = {"cooldown": "cooldown_s", "window": "window_s", "margin": "margin",
                "return margin": "return_margin", "stall": "stall_ms"}


def parse_setting(low: str) -> tuple[str, float] | None:
    """A router setting and its value in the router's units, from a plain "set X to N" command."""
    m = SETTING.search(low)
    if not m:
        return None
    key = SETTING_KEYS[re.sub(r"[\s_-]+", " ", m.group(1))]
    value, unit = float(m.group(2)), (m.group(3) or "")
    if key in ("margin", "return_margin"):
        return key, value / 100 if unit == "%" or value > 1 else value
    if key == "stall_ms":
        return key, value * 1000 if unit.startswith("s") else value
    return key, value / 1000 if unit.startswith("m") else value


def fmt_setting(key: str, value: float) -> str:
    if key in ("margin", "return_margin"):
        return f"{value * 100:.0f}%"
    return f"{value:.0f} ms" if key == "stall_ms" else f"{value:g} s"


def policy_command(cmd: re.Match, low: str, policy: re.Match) -> bool:
    """Is the command a change of the routing rule? Only when the rule's name is what the verb is aimed at:
    "switch to round robin", "set the policy to static", "use adaptive with box A preferred". Not
    "set box B to slow so adaptive moves away"."""
    return (cmd.group(1) in ("switch", "use", "set", "change") and policy.start() >= cmd.end(1) and bool(
        re.fullmatch(r"\s+(?:(?:to|the|a|back|over|routing|router|policy|rule|mode)\s+)*", low[cmd.end(1):policy.start()])))


def command_object(low: str, at: int, mentions: list[int]) -> str | None:
    """What the verb of a command is aimed at: "box", "job", "server" or "model". None when it is something
    else, as in "stop sending traffic to box A", "take a look at box A" or "fail over to box A"."""
    m = BEFORE_OBJECT.match(low, at)
    if not m:
        return None
    if m.end() in mentions:
        return "box"
    return next((kind for kind, pattern in (("job", JOB), ("server", SERVER), ("model", MODEL))
                 if pattern.match(low, m.end())), None)


def box_action(cmd: re.Match, low: str, mentions: list[int]) -> str | None:
    """What a command asks for on the box it names: "start" (the training job), "stop" (the training job),
    "kill" (the forecast server) or "recover". None when the words do not say it clearly; then nothing is
    changed. `mentions` are the places where the text names a box."""
    verb = re.sub(r"[\s-]", "", cmd.group(1))           # "fine-tune", "fine tune" -> "finetune"
    rest = low[cmd.end(1):]
    aim = command_object(low, cmd.end(1), mentions)
    job = bool(JOB.search(rest))
    back = bool(re.search(r"\b(?:normal|back|recover\w*|restore\w*|heal\w*)\b", rest))
    off = bool(re.search(r"\b(?:off|out\s+of)\b", rest))
    down = bool(re.search(r"\b(?:down|fail(?:ed|s|ing|ure)?|kill(?:ed|s|ing)?|crash(?:ed|es|ing)?)\b", rest))
    if verb == "simulate":      # for a box without a supervisor; on one with a supervisor the caller refuses it
        if back:
            return "recover"
        down = down or bool(re.search(r"\berror", rest))
        return None if down == job else "kill" if down else "start"
    if verb in ("start", "run", "launch", "begin"):
        return "start" if aim == "job" else None
    if verb in ("train", "finetune"):
        return "start" if aim in ("box", "model") else None   # not "fine-tune job on box B, is it done"
    if verb == "slow":
        return "start" if aim == "box" else None
    if verb in ("cancel", "abort", "end", "pause", "quit"):   # words for a job, never for a server
        return "stop" if aim in ("job", "box") else None
    if verb in ("stop", "halt", "kill", "terminate", "shut", "fail", "crash"):
        if aim == "job":
            return "stop"       # "kill the training job on box B" stops the job; it kills no server
        if aim == "server":
            return "kill"
        # "kill box B while it is training": the job or the server? Not clear, so nothing.
        return "kill" if aim == "box" and not job else None
    if verb in ("recover", "restore", "heal", "fix"):
        return "recover" if aim in ("box", "server") else None
    # take, bring, put, set, make, turn, switch, change, use: the other words decide
    if aim == "job":
        if off or back:
            return "stop"       # "turn off the training on box B"
        return "start" if verb in ("turn", "switch") and re.match(r"\s+on\b", rest) else None
    if aim == "box":
        if back or (off and job):
            return "stop" if job else "recover"     # "set box B back to normal, it is too slow"
        if down:
            return None if job else "kill"          # "take box A down"
        return "start" if re.search(r"\bslow|\b(?:in|into|to)\s+(?:the\s+)?(?:train|fine[\s-]?tun)", rest) else None
    if aim == "server":
        return "recover" if back else "kill" if down and not job else None
    return None


def llm_tool_refusal(name: str, args: dict, text: str, boxes: list[str]) -> str | None:
    """Why the LLM may not run this tool for the user's last message `text` (the words are for the LLM, which
    passes them on); None when it may.

    In LLM mode the model chooses the tools. For a tool that changes something, that choice is checked
    against the user's own words by the rules of the rule-based mode: the last message must be a plain
    command (it does not ask, and has no "if" and no "not"); for a box it must name exactly that box and
    ask for that action. So a question ("What happens if box A dies?"), a pasted text, or an old command
    further up in the chat never kills a server, starts a job or changes the routing."""
    if name not in CHANGING_TOOLS:
        return None
    low = text.lower()
    how = ('Nothing was changed. Tell the user so, and that a change needs a plain command about one box or '
           'the rule, for example "fail box A", "start a training job on box B" or "switch to balanced".')
    cmd = command_verb(low)
    if not cmd:
        return ("not run: the user's last message is not a plain command (it asks something, or it has an "
                "\"if\" or a \"not\"). " + how)
    places = {n: box_mentions(low, n) for n in boxes}
    named = [n for n in boxes if places[n]]
    if name in ROUTING_TOOLS:
        policy = POLICY.search(low)
        ok = bool((policy and policy_command(cmd, low, policy))
                  or (cmd.group(1) in ("drain", "undrain") and len(named) == 1)
                  or (cmd.group(1) == "reset" and re.search(r"\b(?:rout\w*|settings?|config\w*)\b", low))
                  or (cmd.group(1) in ("set", "change", "make") and parse_setting(low)))
        return None if ok else "not run: the user's last message is not a command to change the routing. " + how
    box = args.get("box")
    if named != [box]:
        return ("not run: the user's last message must name the one box to act on (it names %s; this call is "
                "for %s). " % (" and ".join(named) or "no box", box) + how)
    wanted = ACTION_OF_MODE.get(args.get("mode"), ()) if name == "simulate_box" else ACTION_OF_TOOL[name]
    if box_action(cmd, low, places[box]) not in wanted:
        return "not run: the user's words do not clearly ask for this on %s. " % box + how
    return None


HUMIDITY_FIGURE = re.compile(r"[,;]?\s*(?:and\s+)?(?:relative\s+)?humidity\b[^.,;\n]*?\d+(?:\.\d+)?\s*%", re.I)


def only_given_numbers(reply: str, steps: list[dict]) -> str:
    """The LLM's reply without a humidity figure that no forecast of this turn gave. The real model has no
    humidity output (the summary says humidity_avg_pct: null); a language model sometimes fills the gap
    with a plausible number, and a weather assistant must not state one."""
    summaries = [s["result"].get("summary") for s in steps
                 if s.get("tool") == "get_forecast" and isinstance(s.get("result"), dict)]
    summaries = [x for x in summaries if isinstance(x, dict)]
    if not summaries or any(x.get("humidity_avg_pct") is not None for x in summaries):
        return reply
    return HUMIDITY_FIGURE.sub("", reply)


def was_stopped(job: Any) -> bool:
    """Was this job stopped on request, as opposed to broken? `job` is a supervisor's view of it. The
    supervisor calls every run that did not exit with 0 "failed". Below zero, the stop signal itself ended
    the process; 3 is the code weather/finetune.py leaves with after it caught the signal."""
    if not isinstance(job, dict) or job.get("state") != "failed":
        return False
    code = job.get("exit")
    return isinstance(code, int) and not isinstance(code, bool) and (code < 0 or code == STOP_EXIT)


def clean_steps(value: Any) -> int | None:
    """The `steps` of a fine-tune start: None (the job's own default) or a whole number from 1 to MAX_STEPS,
    the bound of POST /api/box/{box}/train. Anything else raises ValueError."""
    if value is None or value == "":
        return None
    problem = ValueError(f"steps must be a whole number from 1 to {MAX_STEPS}")
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise problem
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        raise problem from None
    if not 1 <= n <= MAX_STEPS:
        raise problem
    return n


def strip_think(text: str) -> str:
    """The model's reply without its <think>...</think> blocks: the reasoning is not for the user."""
    out = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.S | re.I)
    out = re.sub(r"\A.*?</think\s*>", "", out, count=1, flags=re.S | re.I)   # the server cut the opening tag
    out = re.sub(r"<think\b[^>]*>.*\Z", "", out, flags=re.S | re.I)          # the block never closed
    return out.strip()


def step_text(step: dict) -> str:
    """One tool step in words, for the reply after an LLM failure: what ran, on which box, how it ended."""
    args, result = step.get("args") or {}, step.get("result")
    what = str(step.get("tool")) + (f" on {args['box']}" if args.get("box") else "")
    err = result.get("error") if isinstance(result, dict) else None
    return f"{what}: failed ({err})" if err else f"{what}: done"


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


def temp_fmt(units: str):
    return (lambda c: f"{round(c * 9 / 5 + 32)} °F") if units == "F" else (lambda c: f"{round(c)} °C")


def wind_fmt(units: str):
    return (lambda ms: f"{round(ms * 2.237)} mph") if units == "F" else (lambda ms: f"{round(ms * 3.6)} km/h")


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


Ops = Any  # implemented in server.py: forecast, cluster_status, set_config, set_policy, set_box_mode, finetune, kill_forecast
