#!/usr/bin/env python3
"""A teammate's question (Nov 14, 10 PM): do weather questions go to the weather model, do questions about the
infrastructure go to the WhatIf agents, and does an anomaly get a diagnosis in the chat? Here with WhatIf's
LLM calls going to this machine's own LLM server (no key, no cloud)."""
import json, sys, time, urllib.request
PAGE = "http://127.0.0.1:19080"
def call(path, body=None, timeout=240):
    req = urllib.request.Request(PAGE + path, data=None if body is None else json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
def chat(text, history=()):
    t0 = time.time()
    code, ans = call("/api/chat", {"messages": list(history) + [{"role": "user", "content": text}]})
    secs = round(time.time() - t0, 1)
    print("  > %s" % text)
    if code != 200:
        print("      HTTP %s: %s" % (code, ans)); return None
    print("      %4.1f s | mode %s | tools %s" % (secs, ans.get("mode"), [s["tool"] for s in ans.get("steps") or []]))
    print("      reply: %s" % str(ans.get("reply")).replace("\n", " / ")[:600])
    if ans.get("note"): print("      note:  %s" % ans["note"][:260])
    for s in ans.get("steps") or []:
        if s.get("tool") == "whatif_diagnose":
            print("      whatif: %s" % json.dumps(s.get("result"))[:400])
    return ans
def diagnose(policy):
    t0 = time.time()
    code, out = call("/api/incident/diagnose", {"policy": policy})
    print("  POST /api/incident/diagnose policy=%s -> HTTP %s in %.1f s" % (policy, code, time.time() - t0))
    if code == 200:
        d = out.get("diagnosis") or {}
        print("      ranking %s | LLM calls %s | tokens %s | engine %s" % ((d.get("ranking") or [])[:3], d.get("llm_calls"), d.get("tokens"), out.get("engine")))
        for line in (d.get("trail") or [])[:8]:
            print("      trail: %s" % str(line).replace("\n", " ")[:220])
    else:
        print("      %s" % str(out)[:400])
print("started", time.strftime("%H:%M:%S"), "(waiting until the page has a minute of telemetry)")
for _ in range(40):
    code, ev = call("/api/events")
    if code == 200 and (ev.get("telemetry") or {}).get("seconds", 0) >= 62: break
    time.sleep(3)
print("telemetry:", (ev.get("telemetry") if code == 200 else ev))
print("== 1. weather questions go to the weather model")
chat("Weather in Chicago for the next 48 hours?")
print("== 2. questions about the infrastructure, on a healthy and idle stack")
chat("Why is box-a idle?")
chat("Is anything wrong with the cluster?")
print("== 3. the WhatIf agents themselves, with the LLM (healthy stack)")
diagnose("triage"); diagnose("whatif")
print("== 4. an anomaly: box-a's forecast server is killed, then the chat is asked")
chat("Fail box A")
time.sleep(6)
chat("What happened to box-a?")
diagnose("whatif")
time.sleep(30)
chat("Why did box-a go down?")
print("ended", time.strftime("%H:%M:%S"))
