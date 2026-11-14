"""The desk agent: asks an open model a weather question, lets it call the tools, then checks its numbers.

    python agent/desk_agent.py "What will tomorrow morning be like at O'Hare?"
    python agent/desk_agent.py "Is the cluster healthy?" --base-url http://127.0.0.1:8100/v1 --session demo-1

The model is behind the router's llm pool (OpenAI-compatible /v1/chat/completions). The loop: send the
question with the tool list; run the tool calls the model asks for; send the results back as "tool"
messages; stop at the first answer with no tool call, or after --max-rounds. Every call carries the same
X-Session-Id, so the router keeps the conversation on one node. The model server must have tool calling
switched on (llama.cpp: start llama-server with --jinja; vLLM: --enable-auto-tool-choice and a
--tool-call-parser).

Then the number guard. Every number in the answer must be in some tool result (within 0.05). A sentence
with a number that no tool gave is removed. Not checked: numbers that were in the question, four-digit
years, times of day (6 AM, 06:00, 12 UTC) and list numbering. A number written without a minus sign also
matches a negative tool value, so "30.1 below zero" passes. The guard checks where a number came from,
not what the sentence says about it.

It prints the checked answer, how many sentences were dropped, the key facts from the tools, and which
nodes served the model and the tools. Progress notes go to stderr.

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import json
import re
import sys
import uuid

import tools

TOLERANCE = 0.05          # how far a number in the answer may be from a tool's number
LLM_TIMEOUT_S = 180       # one model call; a model that has just been loaded can be slow

SYSTEM_PROMPT = """You are the assistant at a weather forecast desk. Answer the question with the tools.

Rules:
- Get every number from a tool. Never give a number from memory.
- Copy numbers exactly as the tools give them. Do not round them. Do not convert units (no Fahrenheit,
  no mph). Do not do sums.
- A checker removes every sentence of your answer that has a number the tools did not give.
- temp_c is degrees Celsius, wind_mps is metres per second, pressure_hpa is hectopascals.
- Times ending in _utc are UTC. Times ending in _local are local time.
- If a tool gives an error, say what it says. Do not guess.
- Answer in a few short, plain sentences."""

# The parts of each tool's result that go in the "Facts from tools" block.
KEY_FACTS = {
    "forecast_point": ["init_time_utc", "last_lead"],
    "ohare_tomorrow": ["date_local", "init_time_utc", "low", "high", "near_6am"],
    "ensemble_range": ["init_time_utc", "members", "last_lead", "shards"],
    "storm_2019": ["init_time_utc", "checks", "error_at_72h_c"],
    "cluster_status": ["policy", "pools", "counters"],
    "set_routing_policy": ["policy"],
}


# ---------------------------------------------------------------- the number guard

_DIGITS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")       # 1,013.2 or 24 or 3.15
_MINUS_SIGNS = "-\u2212\u2013"                                            # hyphen, minus sign, en dash
_TIME_OF_DAY = re.compile(
    r"(?<![\d.])\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\b\.?)?"            # 06:00, 6:30 pm
    r"|(?<![\d.])\d{1,2}\s*[ap]\.?m\b\.?"                                 # 6 AM, 6am, 6 a.m.
    r"|(?<![\d.])\d{1,4}\s*(?:UTC|GMT|Z)\b",                              # 12 UTC, 1200 UTC, 12Z
    re.IGNORECASE)
_LIST_MARKER = re.compile(r"[ \t>*#-]*\(?\d{1,3}[.)]\**\s+(?=\S)")        # "1. ", "  2) ", "- (3) ", "**4.** "
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)\]*_]*\s+(?=[^a-z\s])")


def find_numbers(text, prose=True):
    """The numbers in a piece of text, as (value, has_minus, start, end).

    value is never negative; has_minus says a minus sign stood right in front of it. A hyphen after a digit
    is a dash, not a minus: "3-5" and "2019-01-27" hold no negative numbers.

    In prose (the model's answer, the question) digits at the end of a word are not numbers (t2m, GB10,
    storm_2019), and a hyphen after a letter is a dash too (UTC-6). In tool text (prose=False) both count.
    """
    found = []
    for match in _DIGITS.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start > 0 else ""
        if before == "." and start > 1 and text[start - 2].isdigit():
            continue                            # the third part of something like 3.12.15
        if prose and (before.isalpha() or before == "_"):
            continue                            # the end of a name: t2m, GB10, storm_2019
        has_minus = False
        if before and before in _MINUS_SIGNS:
            further = text[start - 2] if start > 1 else ""
            if prose:
                has_minus = not (further.isalnum() or further in (")", "]"))
            else:
                has_minus = not further.isdigit()
        found.append((float(match.group().replace(",", "")), has_minus, start, end))
    return found


def signed_numbers(text, prose=True):
    """The numbers in a piece of text as plain floats, negative where a minus stood in front."""
    return [-value if has_minus else value for value, has_minus, _start, _end in find_numbers(text, prose)]


def tool_numbers(result):
    """Every number in a tool result: its number values, and numbers written inside its text values.

    Key names are not counted.
    """
    if isinstance(result, bool):
        return []
    if isinstance(result, (int, float)):
        try:
            return [float(result)]
        except OverflowError:
            return []
    if isinstance(result, str):
        return signed_numbers(result, prose=False)
    if isinstance(result, dict):
        result = list(result.values())
    if isinstance(result, (list, tuple)):
        return [number for item in result for number in tool_numbers(item)]
    return []


def is_known(value, has_minus, known):
    """Is this number from the answer one of the known numbers, within the tolerance?

    A number written with a minus must match a negative known number. A number written without one may
    match either sign, so "30.1 below zero" and "87.9 W" pass when a tool gave -30.1 and -87.9.
    """
    for number in known:
        gap = abs(-value - number) if has_minus else abs(value - abs(number))
        if gap <= TOLERANCE + 1e-9:
            return True
    return False


def unsupported_numbers(line, known):
    """Where one line of the answer has a number that is not known: a list of positions in the line."""
    skip = [match.span() for match in _TIME_OF_DAY.finditer(line)]
    marker = _LIST_MARKER.match(line)
    if marker:
        skip.append(marker.span())
    bad = []
    for value, has_minus, start, end in find_numbers(line):
        if any(first <= start < last for first, last in skip):
            continue                            # a time of day, or the "2." in front of a list item
        if end - start == 4 and line[start:end].isdigit() and 1900 <= value <= 2100 and not has_minus:
            continue                            # a year
        if not is_known(value, has_minus, known):
            bad.append(start)
    return bad


def sentence_spans(line):
    """(start, end) for each sentence in one line. A list marker such as "2." stays with its sentence."""
    marker = _LIST_MARKER.match(line)
    spans, start = [], 0
    for cut in _SENTENCE_END.finditer(line, marker.end() if marker else 0):
        spans.append((start, cut.end()))
        start = cut.end()
    if start < len(line):
        spans.append((start, len(line)))
    return spans


def guard_answer(answer, tool_results, question=""):
    """Remove every sentence of the answer that has a number no tool gave.

    tool_results is the list of dicts the tools returned. Returns (what is left of the answer, the list of
    removed sentences). Each line of the answer is checked on its own, so one bad bullet does not take
    the whole list with it.
    """
    known = tool_numbers(tool_results) + signed_numbers(question)
    kept_lines, dropped = [], []
    for line in answer.splitlines(True):
        body = line.rstrip("\r\n")
        if not body.strip():
            kept_lines.append(line)             # an empty line stays
            continue
        bad = unsupported_numbers(body, known)
        kept = []
        for start, end in sentence_spans(body):
            if any(start <= position < end for position in bad):
                dropped.append(body[start:end].strip())
            else:
                kept.append(body[start:end])
        if kept:
            kept_lines.append("".join(kept).rstrip() + line[len(body):])
    return re.sub(r"\n{3,}", "\n\n", "".join(kept_lines)).strip(), dropped


# ---------------------------------------------------------------- the chat loop

class AgentError(Exception):
    """The model could not be reached, or its answer has a shape we cannot use."""


def note(text):
    """Progress goes to stderr, so stdout holds only the report."""
    sys.stderr.write("desk_agent: %s\n" % text)
    sys.stderr.flush()


def openai_tools():
    """tools.TOOLS in the shape the OpenAI chat API wants."""
    return [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                              "parameters": tool["input_schema"]}} for tool in tools.TOOLS]


def first_model(base_url, session):
    """The first model the endpoint lists (GET /models), or "default" if it will not say."""
    data, _node, _err = tools.http_json("GET", base_url + "/models", headers={"X-Session-Id": session},
                                        timeout=tools.QUICK_TIMEOUT_S)
    try:
        return str(data["data"][0]["id"])
    except (KeyError, IndexError, TypeError):
        return "default"


def ask_model(base_url, body, session):
    """One chat completion. Returns (the assistant's message, the node that served it)."""
    data, node, err = tools.http_json("POST", base_url + "/chat/completions", body,
                                      headers={"X-Session-Id": session}, timeout=LLM_TIMEOUT_S)
    if err:
        raise AgentError(err)
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        message = None
    if not isinstance(message, dict):
        raise AgentError("the model's answer has no choices[0].message: %s" % json.dumps(data)[:200])
    return message, node


def text_of(message):
    """The text of an assistant message, without any <think>...</think> part."""
    content = message.get("content")
    if isinstance(content, list):               # some servers send a list of parts
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        return ""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def run_tool_call(tool_call, number):
    """Run one tool call from the model. Returns (call id, tool name, arguments as text, arguments, result)."""
    tool_call = tool_call if isinstance(tool_call, dict) else {}
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = function.get("name")
    raw = function.get("arguments")
    call_id = tool_call.get("id") or "call_%d" % number
    if isinstance(raw, str):                    # the usual case: the arguments are JSON in a string
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except ValueError as exc:
            return call_id, name, raw, raw, {"error": "%s: the arguments were not valid JSON (%s)" % (name, exc)}
    else:                                       # some servers send the object itself
        arguments = raw
        raw = json.dumps({} if raw is None else raw)
    return call_id, name, raw, arguments, tools.call_tool(name, arguments)


def run(question, base_url, model, session, max_rounds=6):
    """The tool loop. Returns {"answer": text or None, "calls": [...], "llm_nodes": [...]}.

    answer is None when the model was still asking for tools after max_rounds model calls. Each entry of
    calls is {"name", "arguments", "result"}. llm_nodes has the X-Backend of each model call.
    Raises AgentError if a model call fails.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    calls, llm_nodes = [], []
    for round_number in range(1, max_rounds + 1):
        body = {"model": model, "messages": messages, "tools": openai_tools(), "temperature": 0, "stream": False}
        message, node = ask_model(base_url, body, session)
        llm_nodes.append(node)
        wanted = message.get("tool_calls")
        if not isinstance(wanted, list) or not wanted:
            note("round %d: the model (node %s) answered" % (round_number, node))
            return {"answer": text_of(message), "calls": calls, "llm_nodes": llm_nodes}
        echoed, tool_messages = [], []
        for tool_call in wanted:
            call_id, name, raw, arguments, result = run_tool_call(tool_call, len(calls) + 1)
            calls.append({"name": name, "arguments": arguments, "result": result})
            echoed.append({"id": call_id, "type": "function", "function": {"name": str(name), "arguments": raw}})
            tool_messages.append({"role": "tool", "tool_call_id": call_id, "name": str(name),
                                  "content": json.dumps(result)})
            how = "failed: %s" % result["error"] if "error" in result else "ok, node %s" % result.get("node")
            note("round %d: the model (node %s) called %s(%s): %s" % (round_number, node, name, raw, how))
        content = message.get("content")
        messages.append({"role": "assistant", "content": content if isinstance(content, str) else "",
                         "tool_calls": echoed})
        messages.extend(tool_messages)
    return {"answer": None, "calls": calls, "llm_nodes": llm_nodes}


# ---------------------------------------------------------------- the report

def _nested(value):
    return isinstance(value, (dict, list))


def show(value):
    """A value from a tool result as short text. A small dict becomes "a=1, b=2", without its nulls."""
    if isinstance(value, dict):
        return ", ".join("%s=%s" % (key, show(item)) for key, item in value.items() if item is not None)
    if isinstance(value, list):
        return ", ".join(show(item) for item in value)
    return value if isinstance(value, str) else json.dumps(value)


def fact_lines(key, value, pad):
    """Lines for one part of a tool result. Small things take one line; nested things get a block."""
    if isinstance(value, dict) and any(_nested(item) for item in value.values()):
        lines = [pad + "%s:" % key]
        for inner_key, item in value.items():
            lines.extend(fact_lines(inner_key, item, pad + "  "))
        return lines
    if isinstance(value, list) and any(_nested(item) for item in value):
        return [pad + "%s:" % key] + [pad + "  " + show(item) for item in value]
    return [pad + "%s: %s" % (key, show(value))]


def report(outcome, question, max_rounds):
    """The text to print: the checked answer, the dropped count, the facts and the nodes."""
    calls = outcome["calls"]
    if outcome["answer"] is None:
        text, dropped = "(no answer: the model was still asking for tools after %d rounds)" % max_rounds, []
    else:
        text, dropped = guard_answer(outcome["answer"], [call["result"] for call in calls], question)
        if not text and dropped:
            text = "(nothing left: every sentence had a number no tool gave)"
        elif not text:
            text = "(the model gave an empty answer)"
    lines = [text, "", "Dropped %d sentence(s) with numbers no tool gave." % len(dropped), "", "Facts from tools"]
    served = []
    for call in calls:
        name, result = call["name"], call["result"]
        arguments = json.dumps(call["arguments"]) if call["arguments"] else ""
        lines.append("  %s(%s)" % (name, arguments))
        if "error" in result:
            lines.append("    error: %s" % result["error"])
            served.append("%s failed" % name)
            continue
        for key in KEY_FACTS.get(name, sorted(result)):
            lines.extend(fact_lines(key, result.get(key), "    "))
        served.append("%s on %s" % (name, result.get("node") or "unknown"))
    if not calls:
        lines.append("  none: the model called no tool")
    lines += ["", "Nodes",
              "  LLM calls: %s" % (", ".join(node or "unknown" for node in outcome["llm_nodes"]) or "none"),
              "  Tools: %s" % (", ".join(served) or "none")]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask the weather desk's model a question. It may call the weather tools. "
                    "Every number in its answer is then checked against what the tools returned.")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--base-url", metavar="URL",
                        help="the OpenAI-compatible endpoint, ending in /v1 (default: $ROUTER_URL/v1)")
    parser.add_argument("--model", metavar="NAME",
                        help="the model's name (default: the first one the endpoint lists)")
    parser.add_argument("--session", metavar="ID",
                        help="sent as X-Session-Id so the router keeps the conversation on one node "
                             "(default: a new random id)")
    parser.add_argument("--max-rounds", metavar="N", type=int, default=6,
                        help="stop after this many model calls (default 6)")
    args = parser.parse_args(argv)
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    for stream in (sys.stdout, sys.stderr):     # an odd terminal encoding must not lose the answer
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    base_url = (args.base_url or tools.router_url() + "/v1").rstrip("/")
    session = args.session or "desk-" + uuid.uuid4().hex[:12]
    model = args.model or first_model(base_url, session)
    note("model %s at %s, session %s" % (model, base_url, session))
    try:
        outcome = run(args.question, base_url, model, session, args.max_rounds)
    except AgentError as exc:
        note("the model call failed: %s" % exc)
        return 1
    print(report(outcome, args.question, args.max_rounds))
    return 0 if outcome["answer"] else 1


if __name__ == "__main__":
    sys.exit(main())
