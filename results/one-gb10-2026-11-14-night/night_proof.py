#!/usr/bin/env python3
"""Evidence run on one GB10 (the team's stack on local ports): does the chat reach the model, and is Demo traffic real GPU work?
Standard library only. Usage: night_proof.py chat | load | followups   [out.json]"""
import json, subprocess, sys, threading, time, urllib.request

PAGE, ROUTER = "http://127.0.0.1:19080", "http://127.0.0.1:19000"
BOX = {"box-a": "http://127.0.0.1:19001", "box-b": "http://127.0.0.1:19002"}
STACK = {"box-a": "http://127.0.0.1:19109", "box-b": "http://127.0.0.1:19110"}


def get(url, t=20):
    with urllib.request.urlopen(url, timeout=t) as r:
        return json.loads(r.read().decode())


def post(url, body, t=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=t) as r:
        return json.loads(r.read().decode()), dict(r.headers)


def served():
    st = get(ROUTER + "/router/status")
    return {n: b["served"] for n, b in st["backends"].items()}


def chat(messages):
    t0 = time.time()
    ans, _ = post(PAGE + "/api/chat", {"messages": messages, "units": "C"})
    return ans, round(time.time() - t0, 1)


def fc_steps(ans):
    return [s for s in ans.get("steps") or [] if s.get("tool") == "get_forecast" and isinstance(s.get("result"), dict) and s["result"].get("points")]


def gpu_samples(seconds):
    out = []
    end = time.time() + seconds
    while time.time() < end:
        t0 = time.time()
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
            out.append(float(v))
        except Exception:
            pass
        time.sleep(max(0.0, 0.5 - (time.time() - t0)))
    return out


def mean(v):
    return round(sum(v) / len(v), 1) if v else None


def part_chat(save):
    print("=== 1. One chat question, followed through every hop ===")
    info = {n: get(u + "/v1/info") for n, u in BOX.items()}
    for n, i in info.items():
        print("  %s: model %s, %s parameters, device %s, mock %s, newest starting weather %s" % (n, i["model"], format(i["params"], ","), i["device"], i["mock"], i["latest"]))
    s0 = served()
    q = "Weather in Chicago for the next 48 hours?"
    ans, secs = chat([{"role": "user", "content": q}])
    time.sleep(0.5)
    s1 = served()
    print("  question:", q)
    print("  reply (%.1f s): %s" % (secs, ans.get("reply")))
    print("  tools the assistant ran:", [s["tool"] for s in ans.get("steps") or []], "| mode:", ans.get("mode"))
    f = fc_steps(ans)
    if not f:
        print("  NO forecast step in this answer"); save["chat"] = {"answer": ans}; return
    r, a = f[0]["result"], f[0]["args"]
    print("  forecast step: asked lat %s lon %s hours %s | served by %s | model_version %s | %d points | %s ms through the router" % (
        a.get("lat"), a.get("lon"), a.get("hours_ahead"), r.get("served_by"), r.get("model_version"), len(r["points"]), r.get("latency_ms")))
    print("  router's count of forwarded forecasts: before %s, after %s  (difference %d)" % (s0, s1, sum(s1.values()) - sum(s0.values())))
    direct, hdr = post(ROUTER + "/v1/forecast", {"lat": a["lat"], "lon": a["lon"], "hours_ahead": a["hours_ahead"], "step_hours": 1,
                                               "ensemble_members": a.get("ensemble_members", 1)})
    dt = [p["temp_c"] for p in direct["forecast"]]
    ct = [p["t"] for p in r["points"]]
    diff = max(abs(x - y) for x, y in zip(dt, ct)) if len(dt) == len(ct) else None
    print("  the same request sent straight to the router: served by %s, model_version %s, timing %s" % (hdr.get("x-served-by") or hdr.get("X-Served-By"), direct["model_version"], direct["timing"]))
    print("  card temperatures against that direct answer: %d and %d points, largest difference %s C" % (len(ct), len(dt), diff))
    print("  card: min %.1f C, max %.1f C | direct: min %.1f C, max %.1f C" % (min(ct), max(ct), min(dt), max(dt)))
    model_steps = [p for p in direct["forecast"] if not p.get("interpolated")]
    print("  of the %d hourly points, %d are model steps (every %d h); the rest are straight lines between them" % (len(dt), len(model_steps), direct.get("native_step_hours", 6)))
    save["chat"] = {"question": q, "reply": ans.get("reply"), "seconds": secs, "tools": [s["tool"] for s in ans["steps"]], "served_before": s0, "served_after": s1,
                    "step": {"args": a, "served_by": r.get("served_by"), "model_version": r.get("model_version"), "latency_ms": r.get("latency_ms"), "min_c": min(ct), "max_c": max(ct)},
                    "direct": {"model_version": direct["model_version"], "timing": direct["timing"], "min_c": min(dt), "max_c": max(dt), "max_abs_diff_c": diff}}

    print("\n=== 2. The answer follows the model's input: another starting state gives other numbers ===")
    native = {}
    for init in ("2026-11-14T18", "2026-11-14T12", "2026-11-14T00"):
        try:
            d, _ = post(BOX["box-a"] + "/v1/forecast", {"init": init, "lead_hours": 48, "members": 1, "point": {"lat": a["lat"], "lon": a["lon"]}})
            t = d["point"]["t2m_c"]
            native[init] = {"min_c": round(min(t), 2), "max_c": round(max(t), 2), "gpu_ms": d["timing_ms"].get("gpu"), "units": d.get("units")}
            print("  started from %s: %d model steps, 2-m temperature %.1f to %.1f C, %s ms on the GPU" % (init, len(t) - 1 if len(t) > 8 else len(t), min(t), max(t), d["timing_ms"].get("gpu")))
        except Exception as e:
            print("  started from %s: %r" % (init, e))
    save["starts"] = native

    print("\n=== 3. The answer follows the model's weights: swap in the fine-tuned weights, ask again ===")
    import os
    link = os.path.expanduser("~/wx-event/night/bundle/weights/serving.pt")

    def swap(target):
        tmp = link + ".tmp"
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(target, tmp); os.replace(tmp, link)
        for n, u in STACK.items():
            post(u + "/servers/forecast/kill", {})
        end = time.time() + 120
        while time.time() < end:
            try:
                if all(get(u + "/health", 3).get("ready") for u in BOX.values()):
                    break
            except Exception:
                pass
            time.sleep(1)
        time.sleep(2)
        return {n: get(u + "/health").get("model_version") for n, u in BOX.items()}

    try:
        v = swap("afno_tuned_641de834.pt")
        print("  both forecast servers restarted with the tuned file; they now report:", v)
        ans2, secs2 = chat([{"role": "user", "content": q}])
        f2 = fc_steps(ans2)
        if f2:
            r2 = f2[0]["result"]; ct2 = [p["t"] for p in r2["points"]]
            print("  same question (%.1f s): %s" % (secs2, ans2.get("reply")))
            print("  forecast step: served by %s | model_version %s | min %.1f C, max %.1f C  (base weights gave %.1f to %.1f)" % (r2.get("served_by"), r2.get("model_version"), min(ct2), max(ct2), min(ct), max(ct)))
            save["tuned"] = {"reply": ans2.get("reply"), "model_version": r2.get("model_version"), "min_c": min(ct2), "max_c": max(ct2), "versions": v}
        else:
            print("  NO forecast step in the second answer:", ans2.get("reply")); save["tuned"] = {"reply": ans2.get("reply")}
    finally:
        v = swap("afno_event.pt")
        print("  put back the base weights; the servers report:", v)
        save["restored"] = v


def part_load(save):
    print("=== Demo traffic: what the page sends, and what it costs on the GPU ===")
    rows = []

    def window(label, rps, seconds):
        post(PAGE + "/api/load", {"rps": rps})
        time.sleep(6 if rps else 4)
        s0 = served(); st0 = {n: get(u + "/stats") for n, u in BOX.items()}; t0 = time.time()
        g = gpu_samples(seconds)
        s1 = served(); st1 = {n: get(u + "/stats") for n, u in BOX.items()}; dt = time.time() - t0
        ov = get(PAGE + "/api/overview")
        m = ov["metrics"]["overall"]
        fwd = sum(s1.values()) - sum(s0.values())
        allreq = sum(st1[n]["requests_total"] - st0[n]["requests_total"] for n in BOX)
        row = {"label": label, "set_rps": rps, "seconds": round(dt, 1), "router_forwarded": fwd, "forwarded_per_s": round(fwd / dt, 1),
               "forecasts_computed_by_servers": allreq, "gpu_util_mean": mean(g), "gpu_util_max": max(g) if g else None, "gpu_samples": len(g),
               "router_p50_ms": m.get("p50_ms"), "router_p95_ms": m.get("p95_ms"), "errors": m.get("errors"),
               "ms_per_unit": {n: st1[n].get("ms_per_unit") for n in BOX}, "page_load_rps": ov.get("load_rps")}
        rows.append(row)
        print("  %-22s set %2s/s | router forwarded %5d (%.1f/s) | forecasts the two servers computed %5d | GPU busy mean %s%% max %s%% | p50 %s ms p95 %s ms | errors %s" % (
            label, rps, fwd, fwd / dt, allreq, row["gpu_util_mean"], row["gpu_util_max"], m.get("p50_ms"), m.get("p95_ms"), m.get("errors")))

    try:
        window("Off (probes only)", 0, 20)
        window("Light", 8, 30)
        window("Busy", 15, 30)
        window("Heavy", 25, 30)
    finally:
        post(PAGE + "/api/load", {"rps": 0})
    time.sleep(8)
    s0 = served(); time.sleep(5); s1 = served()
    print("  after Off: router forwarded %d requests in 5 s" % (sum(s1.values()) - sum(s0.values())))
    save["load"] = rows


def part_followups(save):
    print("=== Follow-up questions in one chat: does the assistant ask the weather model again? ===")
    out = []
    Q1 = {"role": "user", "content": "Weather in Chicago for the next 48 hours?"}
    a1, s = chat([Q1])
    print("  fresh: Chicago 48 h        %4.1f s | forecast steps %d | %s" % (s, len(fc_steps(a1)), a1.get("reply")))
    out.append({"tag": "fresh Chicago", "forecast_steps": len(fc_steps(a1)), "reply": a1.get("reply"), "tools": [x["tool"] for x in a1.get("steps") or []]})
    A1 = {"role": "assistant", "content": a1.get("reply", "")}
    for q in ("And what about New York?", "And what about New York?", "And Boston tomorrow?", "Same for Miami please", "What about Tokyo for the next 3 days?",
              "Weather in Denver for the next 48 hours?", "And tomorrow?", "Is that warm for November?", "What about St. Louis?"):
        a, s = chat([Q1, A1, {"role": "user", "content": q}])
        n = len(fc_steps(a))
        print("  follow-up: %-40s %4.1f s | forecast steps %d | tools %s | %s" % (q, s, n, [x["tool"] for x in a.get("steps") or []], str(a.get("reply"))[:150].replace("\n", " ")))
        out.append({"tag": "follow-up", "question": q, "forecast_steps": n, "reply": a.get("reply"), "tools": [x["tool"] for x in a.get("steps") or []], "mode": a.get("mode"), "note": a.get("note")})
    for q in ("Weather in New York for the next 48 hours?", "Is it going to rain in Seattle this week?", "Is it going to rain in Seattle this week?",
              "Is it going to rain in Seattle this week?", "Will it be humid in Miami tomorrow?", "Do I need a coat in Boston tomorrow?", "Weather in St. Louis tomorrow?",
              "What is the weather like in Paris and in London tomorrow?"):
        a, s = chat([{"role": "user", "content": q}])
        n = len(fc_steps(a))
        print("  fresh: %-44s %4.1f s | forecast steps %d | tools %s | %s" % (q, s, n, [x["tool"] for x in a.get("steps") or []], str(a.get("reply"))[:150].replace("\n", " ")))
        out.append({"tag": "fresh", "question": q, "forecast_steps": n, "reply": a.get("reply"), "tools": [x["tool"] for x in a.get("steps") or []], "mode": a.get("mode"), "note": a.get("note")})
    fu = [o for o in out if o["tag"] == "follow-up"]; fr = [o for o in out if o["tag"] == "fresh"]
    print("  SUMMARY: follow-ups with a forecast from the model: %d of %d | fresh questions: %d of %d" % (
        sum(1 for o in fu if o["forecast_steps"]), len(fu), sum(1 for o in fr if o["forecast_steps"]), len(fr)))
    save["followups"] = out


if __name__ == "__main__":
    what = sys.argv[1]
    save = {"what": what, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("started", save["started"])
    {"chat": part_chat, "load": part_load, "followups": part_followups}[what](save)
    save["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if len(sys.argv) > 2:
        json.dump(save, open(sys.argv[2], "w"), indent=1)
        print("saved", sys.argv[2])
    print("ended", save["ended"])
