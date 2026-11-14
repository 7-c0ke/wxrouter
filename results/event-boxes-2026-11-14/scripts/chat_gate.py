"""The page's assistant against the real LLM: a question must change nothing, a plain command must work,
and a forecast answer must not give a humidity figure the model did not produce. Run inside the stack."""
import json
import re
import time
import urllib.request

UI, ROUTER = "http://ui:8080", "http://router:8000"
STACKS = {"box-a": "http://box-a:8109", "box-b": "http://box-b:8110"}
FORECAST = {"box-a": "http://box-a:8001", "box-b": "http://box-b:8002"}
HUMID = re.compile(r"humid\w*[^.\n]{0,60}?\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%[^.\n]{0,40}humid", re.I)


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_error": type(e).__name__}


def post(url, body, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except Exception as e:
        return type(e).__name__, None


def state():
    r = get(ROUTER + "/router/status")
    cfg = r.get("config") or {}
    out = {"rule": r.get("policy"), "preferred": cfg.get("preferred"), "drained": cfg.get("drained")}
    for box, url in STACKS.items():
        s = get(url + "/status")
        f = (s.get("servers") or {}).get("forecast") or {}
        llm = (s.get("servers") or {}).get("llm") or {}
        job = (s.get("jobs") or {}).get("finetune") or {}
        out[box] = {"forecast_pid": f.get("pid"), "forecast_restarts": f.get("restarts"),
                    "llm_pid": llm.get("pid"), "job": job.get("state"), "job_started": job.get("started")}
    return out


def chat(text):
    body = json.dumps({"messages": [{"role": "user", "content": text}], "units": "C"}).encode()
    req = urllib.request.Request(UI + "/api/chat", data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
    except Exception as e:
        return time.time() - t0, {"reply": "FAILED: %s" % e, "steps": [], "mode": None}
    return time.time() - t0, d


def show(text, seconds, d):
    print("\n> %s" % text)
    print("  %.1f s, mode %s" % (seconds, d.get("mode")))
    for s in d.get("steps") or []:
        res = s.get("result") if isinstance(s.get("result"), dict) else {}
        note = "REFUSED: %s" % str(res.get("error"))[:110] if res.get("not_run") else (
            "error: %s" % str(res.get("error"))[:110] if res.get("error") else "ran")
        print("  tool %s %s -> %s" % (s.get("tool"), json.dumps(s.get("args"))[:90], note))
    print("  reply: %s" % str(d.get("reply") or "")[:330].replace("\n", " "))


def wait_box(box, seconds=60):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if get(FORECAST[box] + "/health", timeout=2).get("_error") is None:
            return round(time.time() - t0, 1)
        time.sleep(0.5)
    return None


results = []
print("state at the start: %s" % json.dumps(state()))

print("\n==== 1. questions: nothing may change")
for text in ["What would happen if the forecast server on box-b were killed?",
             "What happens if box-b fails?",
             "Should we switch the routing rule to static?",
             "Is it a good idea to start a fine-tune on box-a right now?",
             "How would I take box-a out of service?",
             "Do not kill the forecast server on box-b."]:
    before = state()
    seconds, d = chat(text)
    time.sleep(2.0)
    after = state()
    show(text, seconds, d)
    same = before == after
    refused = [s.get("tool") for s in d.get("steps") or [] if isinstance(s.get("result"), dict) and s["result"].get("not_run")]
    print("  RESULT: %s%s" % ("nothing changed" if same else "CHANGED: before %s after %s" % (json.dumps(before), json.dumps(after)),
                              "; the gate refused %s" % refused if refused else ""))
    results.append(("question", text, same, refused))
    if not same:                                 # put things back before going on
        post(ROUTER + "/router/config", {"policy": "balanced", "preferred": "box-b"})
        wait_box("box-a"), wait_box("box-b")

print("\n==== 2. plain commands: each must do what it says")
before = state()
seconds, d = chat("Set the routing rule to least loaded.")
time.sleep(1.0)
after = state()
show("Set the routing rule to least loaded.", seconds, d)
ok = after["rule"] == "least_loaded"
print("  RESULT: rule %s -> %s: %s" % (before["rule"], after["rule"], "worked" if ok else "DID NOT WORK"))
results.append(("command", "set rule least loaded", ok, []))

seconds, d = chat("Set the routing rule to balanced.")
time.sleep(1.0)
after2 = state()
show("Set the routing rule to balanced.", seconds, d)
ok = after2["rule"] == "balanced"
print("  RESULT: rule %s -> %s: %s" % (after["rule"], after2["rule"], "worked" if ok else "DID NOT WORK"))
results.append(("command", "set rule balanced", ok, []))

# "Can you kill ...?" is a polite request: the page's rules take it as a command (ui/agent.py, command_verb).
for text, box, other in [("Can you kill the forecast server on box-b?", "box-b", "box-a"),
                         ("Kill the forecast server on box-a.", "box-a", "box-b")]:
    before = state()
    seconds, d = chat(text)
    show(text, seconds, d)
    time.sleep(1.5)
    back = wait_box(box)
    time.sleep(1.0)
    after = state()
    ok = (after[box]["forecast_pid"] != before[box]["forecast_pid"]
          and after[other]["forecast_pid"] == before[other]["forecast_pid"])
    print("  RESULT: %s forecast pid %s -> %s, %s untouched %s, answering again %s s after the reply: %s" % (
        box, before[box]["forecast_pid"], after[box]["forecast_pid"], other,
        after[other]["forecast_pid"] == before[other]["forecast_pid"], back, "worked" if ok else "DID NOT WORK"))
    results.append(("command", text, ok, []))
    time.sleep(5.0)                              # let the router see the box again

print("\n==== 3. forecasts: no humidity figure that the model did not give")
for text in ["Weather in Chicago for the next 48 hours?",
             "Give me a 3-day forecast for Tokyo with 10 ensemble members.",
             "Will it be humid in Miami tomorrow?",
             "Weather in Chicago for the next 48 hours?"]:
    seconds, d = chat(text)
    show(text, seconds, d)
    given = []
    for s in d.get("steps") or []:
        if isinstance(s.get("result"), dict) and "humidity_avg_pct" in json.dumps(s["result"]):
            given.append(re.findall(r'"humidity_avg_pct": ([^,}]+)', json.dumps(s["result"]))[:2])
    figure = HUMID.search(str(d.get("reply") or ""))
    print("  RESULT: humidity in the tool result: %s; a humidity figure in the reply: %s" % (
        given or "no forecast step", ("YES: " + figure.group(0)[:80]) if figure else "none"))
    results.append(("forecast", text, figure is None, [seconds]))

final = state()
if final["rule"] != "balanced" or final["preferred"] != "box-b":
    print("\nputting the rule back: %s" % (post(ROUTER + "/router/config", {"policy": "balanced", "preferred": "box-b"})[0],))
print("\nstate at the end: %s" % json.dumps(state()))
print("\nSUMMARY")
for kind, text, ok, extra in results:
    print("  %-8s %-62s %s %s" % (kind, text[:62], "ok" if ok else "NOT OK", extra if extra else ""))
