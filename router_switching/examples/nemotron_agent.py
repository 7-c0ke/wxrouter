#!/usr/bin/env python3
"""A chat agent that drives wxrouter through its API: your LLM (Nemotron or any OpenAI-compatible server
with function calling) decides which tool to call, this script runs the tool on the wxrouter API and
hands the result back, until the LLM answers in words.

  python nemotron_agent.py --api http://WXROUTER_HOST:18180 --llm http://YOUR_LLM_HOST:8090/v1 \\
      "Check that inference switching works and tell me which GPU served the requests."

  --key KEY     needed for the tools that change the cluster (run_switching_check, set_box_fault,
                set_routing_policy); ask the wxrouter owner for it
  --model NAME  the model name your LLM server expects (default: nemotron)

Standard library only. The tools come from GET {api}/api/v1/tools, so this script needs no update when
tools are added.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

SYSTEM = ("You operate a GPU inference cluster: two GPU boxes behind a router that switches traffic between them. "
          "Use the tools to answer; never guess numbers. A null in a tool result means the value was not reported. "
          "Tools marked CHANGES THE CLUSTER are only for when the user asks for a test or a change. "
          "When you report, say which box served the requests, whether any failed, and how long things took.")


def post(url, body, headers=None, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
    except OSError as e:
        return {"error": f"could not reach {url}: {e}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--api", default="http://127.0.0.1:18180", help="the wxrouter Control Room server")
    ap.add_argument("--llm", default="http://127.0.0.1:8090/v1", help="OpenAI-compatible base URL of your LLM")
    ap.add_argument("--model", default="nemotron")
    ap.add_argument("--key", default=None, help="wxrouter API key, for the tools that change the cluster")
    ap.add_argument("--max-steps", type=int, default=6)
    a = ap.parse_args()
    with urllib.request.urlopen(f"{a.api}/api/v1/tools", timeout=10) as r:
        tools = json.load(r)["tools"]
    key = {"x-api-key": a.key} if a.key else {}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.prompt}]
    for _ in range(a.max_steps):
        out = post(f"{a.llm}/chat/completions", {"model": a.model, "messages": messages, "tools": tools, "temperature": 0.2})
        if "error" in out:
            sys.exit(f"LLM: {out['error']}")
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            print(msg.get("content") or "")
            return
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name, raw = c["function"]["name"], c["function"].get("arguments") or "{}"
            args = json.loads(raw) if isinstance(raw, str) else raw
            print(f"[tool] {name} {json.dumps(args)}", file=sys.stderr)
            result = post(f"{a.api}/api/v1/tools/{name}", args, key)
            print(f"[result] {json.dumps(result)[:400]}", file=sys.stderr)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(result)[:6000]})
    print("(stopped: the LLM kept calling tools)")


if __name__ == "__main__":
    main()
