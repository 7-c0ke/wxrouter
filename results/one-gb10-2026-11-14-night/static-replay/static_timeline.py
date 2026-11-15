"""Per 10 s of a fault_table case: requests sent, median and p95 latency, how many were waiting at the
end of the slice, and the time per model step while the box was busy (gap between two answers of one
box, divided by the model steps of the second request). Usage: static_timeline.py <folder> [rule] [box]"""
import json, os, sys, statistics as st
D = sys.argv[1]; rule = sys.argv[2] if len(sys.argv) > 2 else "static"; box = sys.argv[3] if len(sys.argv) > 3 else "box-a"
trace = {}
for l in open(os.path.join(D, "trace.jsonl")):
    r = json.loads(l)
    body = r.get("body") or r.get("request") or r
    hours = body.get("hours_ahead") or body.get("hours") or 0
    step = body.get("step_hours") or 1
    members = body.get("ensemble_members") or body.get("members") or 1
    trace[r.get("id") or body.get("request_id")] = max(1, -(-int(hours) // 6)) * int(members)
rows = [json.loads(l) for l in open(os.path.join(D, f"raw-{rule}.jsonl"))]
rows = [r for r in rows if r.get("status") == 200]
for r in rows:
    r["done"] = r["sent_s"] + r["latency_ms"] / 1000.0
    r["steps"] = trace.get(r["id"], None)
print(f"{rule}: {len(rows)} answered; steps known for {sum(1 for r in rows if r['steps'])}; total model steps {sum(r['steps'] or 0 for r in rows)}")
mine = sorted((r for r in rows if r["served_by"] == box), key=lambda r: r["done"])
per = {}
for a, b in zip(mine, mine[1:]):
    gap = b["done"] - a["done"]
    if b["sent_s"] < a["done"] and b["steps"]:          # b was already waiting when a finished: the box was busy
        per.setdefault(int(b["done"] // 10), []).append(1000.0 * gap / b["steps"])
def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)
print(" slice    sent  p50 ms   p95 ms  waiting at end  steps finished/s  ms per model step while busy (median, n)")
for s in range(0, 16):
    lo, hi = s * 10, s * 10 + 10
    sent = [r for r in rows if lo <= r["t"] < hi]
    lat = [r["latency_ms"] for r in sent]
    waiting = sum(1 for r in rows if r["sent_s"] < hi <= r["done"])
    fin = sum(r["steps"] or 0 for r in rows if lo <= r["done"] < hi) / 10.0
    g = per.get(s, [])
    print(f" {lo:3d}-{hi:3d} {len(sent):5d} {pct(lat,.5) if lat else 0:7.0f} {pct(lat,.95) if lat else 0:8.0f} {waiting:10d} {fin:14.1f}      " + (f"{st.median(g):6.1f}  (n={len(g)})" if g else "   -"))
