#!/usr/bin/env python3
"""A follow-up without a place, after a question about a place outside the city list: is it answered for that
place? Each chain is run several times, because the LLM does not always choose the same way."""
import json, sys, time, urllib.request
PAGE = "http://127.0.0.1:19080"
def chat(messages):
    req = urllib.request.Request(PAGE + "/api/chat", data=json.dumps({"messages": messages}).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())
def places(ans):
    return [str(s["result"].get("place") or (s.get("args") or {}).get("place")) for s in ans.get("steps") or []
            if s.get("tool") == "get_forecast" and isinstance(s.get("result"), dict) and "error" not in s["result"]]
U = lambda t: {"role": "user", "content": t}
A = lambda t: {"role": "assistant", "content": t}
CHAINS = [("Weather in Chicago for the next 48 hours?", "And what about St. Louis?", "st. louis", ["And tomorrow?", "What about the wind?", "And the weekend?"]),
          ("Weather in Tokyo?", "Weather in Kansas City tomorrow?", "kansas city", ["And the weekend?", "And tonight?"]),
          ("Weather in Chicago for the next 48 hours?", "And Reykjavik?", "reykjavik", ["And tomorrow?", "What about the wind?"])]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
right = wrong = both = none = 0
rows = []
print("started", time.strftime("%H:%M:%S"))
for q1, q2, want, follow_ups in CHAINS:
    a1 = chat([U(q1)])
    for q3 in follow_ups:
        for i in range(N):
            a2 = chat([U(q1), A(a1["reply"]), U(q2)])
            a3 = chat([U(q1), A(a1["reply"]), U(q2), A(a2["reply"]), U(q3)])
            got = [p.lower() for p in places(a3)]
            kind = "none" if not got else "right" if all(want in p for p in got) else "both" if any(want in p for p in got) else "WRONG"
            right += kind == "right"; wrong += kind == "WRONG"; both += kind == "both"; none += kind == "none"
            rows.append({"chain": [q1, q2, q3], "second_answer_places": places(a2), "places": places(a3), "kind": kind, "reply": a3.get("reply"), "mode": a3.get("mode")})
            print("  %-34s > %-18s -> forecast for %-28s %-5s | %s" % (q2[:34], q3, places(a3), kind, str(a3.get("reply")).replace("\n", " / ")[:110]))
print("SUMMARY: %d follow-ups: the place of the question before: %d | that place and another: %d | another place only: %d | no forecast: %d" % (
    right + wrong + both + none, right, both, wrong, none))
if len(sys.argv) > 2:
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
print("ended", time.strftime("%H:%M:%S"))
