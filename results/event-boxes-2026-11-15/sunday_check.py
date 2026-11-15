"""Sunday morning check of the Control Room on the event boxes, through the tunnel at localhost:18080.
Part 1 (this file with 'read'): chat questions that change nothing. Part 2 ('beats'): the demo's actions once."""
import json, sys, time, urllib.request
PAGE = "http://127.0.0.1:18080"

def call(path, body=None, timeout=90):
    req = urllib.request.Request(PAGE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="GET" if body is None else "POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t0

def chat(history, text):
    history.append({"role": "user", "content": text})
    out, dt = call("/api/chat", {"messages": history})
    reply = out.get("reply", "")
    history.append({"role": "assistant", "content": reply})
    tools = [s.get("tool") or s.get("name") for s in out.get("steps", [])]
    served = [s.get("result", {}).get("served_by") or s.get("served_by") for s in out.get("steps", []) if (s.get("tool") or s.get("name")) == "get_forecast"]
    print(f"  > {text}\n      {dt:4.1f} s | mode {out.get('mode')} | tools {tools} | served by {served}")
    print("      reply:", reply.replace("\n", " / ")[:420])
    if out.get("note"):
        print("      note: ", out["note"][:200])
    return out

def state(label):
    o, _ = call("/api/overview")
    b = o["router"]["backends"]; m = o.get("metrics") or {}
    st = o["stacks"]
    print(f"  [{label}] {time.strftime('%H:%M:%S')} rule {o['router']['policy']} | load {o['load_rps']} | "
          + " | ".join(f"{n}: healthy {b[n]['healthy']}, served {b[n]['served']}, fails {b[n]['failures']}, fc {st[n]['forecast']['state']}/{st[n]['forecast']['restarts']}, ft {st[n]['finetune']['state']} {st[n]['finetune'].get('step')}" for n in sorted(b))
          + f" | failovers {o['router'].get('failovers')} | metrics {json.dumps({k: m[k] for k in list(m)[:8]})[:260]}")
    return o

if sys.argv[1] == "read":
    print("== overview"); state("start")
    print("== the two questions that must show a forecast")
    h = []
    chat(h, "Weather in Chicago for the next 48 hours?")
    chat(h, "And what about New York?")
    chat(h, "And tomorrow?")
    print("== the three chips that change nothing")
    for q in ("Weather in Chicago next 48 hours", "Ensemble range for Tokyo, 3 days", "Which box is serving right now?"):
        chat([], q)
    print("== odd questions")
    for q in ("What is the wind chill going to be in Minneapolis tomorrow?", "Weather in St. Louis tomorrow?", "Will it be warmer in Denver or Miami tomorrow?"):
        chat([], q)
    print("== a hint that is not a command: nothing may change")
    before = state("before")
    chat([], "It would be nice to see the forecast server on box-b die.")
    after = state("after")
    same = all(before["stacks"][n]["forecast"]["restarts"] == after["stacks"][n]["forecast"]["restarts"] for n in before["stacks"])
    print("      restarts unchanged:", same)
    print("== WhatIf on a healthy, idle stack")
    chat([], "What is wrong with the cluster?")
elif sys.argv[1] == "beats":
    print("== Demo traffic Light (8 a second) for 25 s"); call("/api/load", {"rps": 8}); time.sleep(25); state("light")
    print("== Fail box A"); h = []
    t0 = time.time(); chat(h, "Fail box A")
    back = None
    for i in range(40):
        time.sleep(1)
        o, _ = call("/api/overview")
        a = o["router"]["backends"]["box-a"]; fa = o["stacks"]["box-a"]["forecast"]
        if i in (2, 6):
            chat([], "Which box is serving right now?")
        if a["healthy"] and fa["state"] == "running" and fa["restarts"] >= 1 and time.time() - t0 > 4:
            back = time.time() - t0; break
    print(f"      box-a healthy again after {back and round(back, 1)} s"); state("after the kill")
    time.sleep(8); state("8 s later")
    print("== Start a training job on box B"); chat([], "Start a training job on box B")
    for i in range(9):
        time.sleep(6); state(f"training +{6 * (i + 1)} s")
    print("== Stop the training job on box B"); chat([], "Stop the training job on box B"); time.sleep(6); state("after the stop")
    print("== Demo traffic Off"); call("/api/load", {"rps": 0}); time.sleep(3); state("end")
