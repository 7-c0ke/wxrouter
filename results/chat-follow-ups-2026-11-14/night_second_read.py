#!/usr/bin/env python3
"""The fixes of the second read, on the test stack with the real model and the real LLM (Nov 14, midnight).
For every answer: the tools that ran, the reply, the note, and whether each temperature and wind figure of the
reply is a number of a forecast of that turn (checked here by this script's own small parser)."""
import json, re, sys, time, urllib.request
PAGE, SUP_A = "http://127.0.0.1:19080", "http://127.0.0.1:19109"
def call(url, body=None, timeout=180):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), dict(r.headers)
def chat(messages, units="C"):
    t0 = time.time()
    ans, _ = call(PAGE + "/api/chat", {"messages": messages, "units": units})
    return ans, round(time.time() - t0, 1)
def forecasts(ans):
    return [s for s in ans.get("steps") or [] if s.get("tool") == "get_forecast" and isinstance(s.get("result"), dict) and isinstance(s["result"].get("summary"), dict)]
FIG = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(?:°\s*[CF]?|km\s*/\s*h|mph|m\s*/\s*s)")
def check_numbers(ans):
    given = []
    for s in forecasts(ans):
        m = s["result"]["summary"]
        for k in ("min_c", "max_c", "ens_min_c", "ens_max_c"):
            if isinstance(m.get(k), (int, float)): given += [m[k], m[k] * 9 / 5 + 32]
        if isinstance(m.get("max_wind_ms"), (int, float)): given += [m["max_wind_ms"], m["max_wind_ms"] * 3.6, m["max_wind_ms"] * 2.237]
    # the LLM often writes a minus as a non-breaking hyphen or a dash ("‑4.4 °C", "–0.8 °C")
    text = re.sub("[\u2010\u2011\u2012\u2013\u2212](?=\\d)", "-", str(ans.get("reply") or ""))
    figs = [float(x) for x in FIG.findall(text)]
    bad = [v for v in figs if not any(abs(v - g) <= (0.5 if float(v).is_integer() else 0.051) for g in given)]
    return figs, bad
def asked_figures(q):
    return [float(x) for x in re.findall(r"(?<![\w.,])\d+(?:\.\d+)?(?![\w,])", q)]
RESULTS = []
def show(tag, q, ans, secs, expect=""):
    figs, bad = check_numbers(ans)
    fs = forecasts(ans)
    own = [v for v in bad if v in asked_figures(q)] if fs else []     # "above 10 °C?": the reply may repeat the 10
    bad = [v for v in bad if v not in own]
    places = [str(s["result"].get("place") or (s.get("args") or {}).get("place")) for s in fs]
    model = "; ".join("%s %.1f to %.1f C, wind %.0f km/h" % (p, s["result"]["summary"]["min_c"], s["result"]["summary"]["max_c"], (s["result"]["summary"].get("max_wind_ms") or 0) * 3.6) for p, s in zip(places, fs))
    verdict = "no weather figure in the reply" if not figs else ("ALL %d FIGURES ARE THE MODEL'S" % (len(figs) - len(own)) if not bad else "NOT FROM A FORECAST OF THIS TURN: %s" % bad)
    if own: verdict += " (and %s repeats a figure of the question)" % own
    print("  [%s] %s" % (tag, q))
    print("        %4.1f s | mode %s | tools %s" % (secs, ans.get("mode"), [s["tool"] for s in ans.get("steps") or []]))
    print("        reply: %s" % str(ans.get("reply")).replace("\n", " / ")[:420])
    if ans.get("note"): print("        note:  %s" % ans["note"][:230])
    if model: print("        model: %s" % model)
    print("        check: %s%s" % (verdict, ("   | expected: " + expect) if expect else ""))
    RESULTS.append({"tag": tag, "question": q, "seconds": secs, "mode": ans.get("mode"), "tools": [s["tool"] for s in ans.get("steps") or []],
                    "reply": ans.get("reply"), "note": ans.get("note"), "forecast_places": places, "figures": figs, "figures_not_from_the_model": bad})
    return ans
def U(text): return {"role": "user", "content": text}
def Asst(text): return {"role": "assistant", "content": text}
def restarts(): return call(SUP_A + "/status")[0]["servers"]["forecast"]["restarts"]

print("started", time.strftime("%H:%M:%S"))
print("== A. the kill scene: which box serves, before, during and after")
a, s = chat([U("Which box is serving right now?")]); show("before", "Which box is serving right now?", a, s, "both boxes")
r0 = restarts()
a, s = chat([U("Fail box A")]); show("kill", "Fail box A", a, s, "a real kill of box-a's forecast server")
a, s = chat([U("Which box is serving right now?")]); show("during", "Which box is serving right now?", a, s, "box-b alone, box-a down")
ov, _ = call(PAGE + "/api/overview")
print("        router's view right then:", {n: (b.get("healthy"), b.get("drained")) for n, b in ov["router"]["backends"].items()})
time.sleep(25)
a, s = chat([U("Which box is serving right now?")]); show("after", "Which box is serving right now?", a, s, "both boxes again")
print("        box-a's forecast server restarts: %s -> %s" % (r0, restarts()))

print("== B. a place the list does not know: the LLM asks the model, and the reply's numbers are checked")
for q in ("Weather in St. Louis tomorrow?", "What's the weather like in Reykjavik for the next 3 days?", "Will St. Louis get above 10 °C tomorrow?",
          "Is it over 50 degrees in St. Louis tomorrow?"):
    a, s = chat([U(q)]); show("fresh", q, a, s)
a, s = chat([U("Weather in St. Louis tomorrow?")], "F"); show("fresh, °F", "Weather in St. Louis tomorrow?", a, s)

print("== C. comparisons with a place the list does not know")
for q in ("Is it warmer in Chicago or St. Louis?", "Is it warmer in Knoxville or Chicago?", "Is St. Louis warmer than Chicago?"):
    a, s = chat([U(q)]); show("fresh", q, a, s, "two forecasts, or the model's numbers of what did run")

print("== D. a chain: Chicago, then St. Louis, then a follow-up without a place")
Q1 = U("Weather in Chicago for the next 48 hours?")
a1, s = chat([Q1]); show("1", Q1["content"], a1, s)
h = [Q1, Asst(a1["reply"]), U("And what about St. Louis?")]
a2, s = chat(h); show("2", "And what about St. Louis?", a2, s)
h += [Asst(a2["reply"]), U("And tomorrow?")]
a3, s = chat(h); show("3", "And tomorrow?", a3, s, "St. Louis (or a question back), NOT Chicago")

print("== E. which of these is warmest, after three known cities")
h = [Q1, Asst(a1["reply"]), U("And what about New York?"), Asst("**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h."),
     U("And Tokyo?"), Asst("**Tokyo, next 48 h:** 9 °C to 17 °C, wind up to 20 km/h."), U("Which of these is warmest?")]
a, s = chat(h); show("compare", "Which of these is warmest?", a, s, "three forecasts and a Warmest line, no LLM")

print("== E2. the same question when not every place can be compared by the rules: the LLM gets it")
four = [Q1, Asst(a1["reply"]), U("And what about New York?"), Asst("**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h."),
        U("And Boston tomorrow?"), Asst("**Boston, next 48 h:** 5 °C to 8 °C, wind up to 21 km/h."),
        U("Same for Miami please"), Asst("**Miami, next 48 h:** 20 °C to 25 °C, wind up to 21 km/h."), U("Which of these is coldest?")]
a, s = chat(four); show("compare", "Which of these is coldest?  (after Chicago, New York, Boston, Miami)", a, s, "the LLM; every figure from a forecast of this turn")
kyoto = [Q1, Asst(a1["reply"]), U("And what about New York?"), Asst("**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h."),
         U("And Kyoto?"), Asst("**Kyoto, next 48 h:** 10.8 °C to 16.3 °C, wind up to 24 km/h."), U("Which of these is warmest?")]
a, s = chat(kyoto); show("compare", "Which of these is warmest?  (after Chicago, New York, Kyoto)", a, s, "the LLM; every figure from a forecast of this turn")
a, s = chat([Q1, Asst(a1["reply"]), U("Where is a good hotel near the venue?")]); show("follow-up", "Where is a good hotel near the venue?", a, s, "no forecast: not a weather question")

print("== F. questions about the earlier answer")
base = [Q1, Asst(a1["reply"])]
for q in ("Is that warm for November?", "So should I take a coat?", "Can you give me that in Fahrenheit?", "Thanks!"):
    a, s = chat(base + [U(q)]); show("follow-up", q, a, s)

print("== G. arithmetic and the cluster")
for q in ("What is 20 C in Fahrenheit?", "what is 20c in f", "How hot is the GPU on box A?", "What is the p95 right now?",
          "Is 1,000 steps enough for the fine-tune?"):
    a, s = chat([U(q)]); show("fresh", q, a, s)
h = [U("Ensemble range for Tokyo, 3 days"), Asst("**Tokyo, next 72 h:** 9 °C to 17 °C, wind up to 20 km/h.")]
for q in ("Which box answered the Tokyo question?", "How long did the Tokyo forecast take?"):
    a, s = chat(h + [U(q)]); show("follow-up", q, a, s, "no new forecast for Tokyo")

print("== G2. cases from the readers who tried to break the fixes")
for q, expect in (("And St. Louis, in Fahrenheit this time?", "a forecast for St. Louis; never numbers without one"),
                  ("Will it rain in St. Louis tomorrow?", "the model's numbers; no rain figure; no reply that starts with 'Louis'")):
    a, s = chat(base + [U(q)], "F" if "Fahrenheit" in q else "C"); show("follow-up", q, a, s, expect)
for q, expect in (("Which box is up, and how cold will St. Louis get tonight?", "a forecast, or no weather figure"),
                  ("How is the latency right now?", "the status's numbers, not replaced"),
                  ("Weather in Thunder Bay tomorrow?", "the name stays in the reply (it holds a weather word)"),
                  ("Is it warmer in Chicago or in Thunder Bay tomorrow?", "two forecasts; no number glued to another"),
                  ("Weather in St. Cloud, Minnesota tomorrow?", "the name stays in the reply")):
    a, s = chat([U(q)]); show("fresh", q, a, s, expect)

print("== H. not for real: nothing may be killed")
r1 = restarts()
for q in ("Kill box A (sim)", "Fail box A in sim mode", "Simulate a failure on box A", "Fail box A virtually"):
    a, s = chat([U(q)]); show("gate", q, a, s, "nothing killed")
time.sleep(3)
print("        box-a's forecast server restarts before and after these four: %s -> %s" % (r1, restarts()))

print("== I. the page's files")
for path in ("/", "/static/app.js", "/static/style.css", "/api/events"):
    req = urllib.request.Request(PAGE + path)
    with urllib.request.urlopen(req, timeout=10) as r:
        print("        %-18s Cache-Control: %s" % (path, r.headers.get("cache-control")))
bad = [r for r in RESULTS if r["figures_not_from_the_model"] and r["tag"] not in ("follow-up",) or (r["figures_not_from_the_model"] and r["forecast_places"])]
print("SUMMARY: %d answers; answers with a forecast whose reply has a figure that is not the model's: %d" % (
    len(RESULTS), sum(1 for r in RESULTS if r["forecast_places"] and r["figures_not_from_the_model"])))
print("         answers without a forecast that still show a temperature or wind figure: %s" % [r["question"] for r in RESULTS if not r["forecast_places"] and r["figures"]])
if len(sys.argv) > 1:
    json.dump(RESULTS, open(sys.argv[1], "w"), indent=1)
print("ended", time.strftime("%H:%M:%S"))
