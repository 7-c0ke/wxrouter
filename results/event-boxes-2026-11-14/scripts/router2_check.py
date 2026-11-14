"""The second router (router/router.py) against the real boxes: forecasts and chat through one address,
a chat that stays on one box, and a chat that goes on when that box's LLM server is killed.
Run inside box-a of the stack, with the second router listening on 127.0.0.1:8100."""
import collections
import json
import threading
import time
import urllib.error
import urllib.request

R2 = "http://127.0.0.1:8100"
STACK_B = "http://box-b:8110"
LLM_B = "http://box-b:8091"


def call(url, body=None, headers=None, timeout=120, post=False):
    data = None if body is None else json.dumps(body).encode()
    if data is None and post:
        data = b""
    h = {"content-type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                d = json.loads(raw)
            except ValueError:
                d = None
            return r.status, d, r.headers, time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            d = json.loads(e.read())
        except ValueError:
            d = None
        return e.code, d, e.headers, time.time() - t0
    except Exception as e:
        return type(e).__name__, None, {}, time.time() - t0


def chat(session, text="Reply with the single word: ready", timeout=120):
    body = {"model": "nemotron", "messages": [{"role": "user", "content": text}], "max_tokens": 24,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    status, d, headers, seconds = call(R2 + "/v1/chat/completions", body, {"X-Session-Id": session}, timeout)
    said = ""
    try:
        said = (d["choices"][0]["message"].get("content") or "").strip()[:40]
    except Exception:
        pass
    return {"status": status, "backend": headers.get("X-Backend"), "retries": headers.get("X-Router-Retries"),
            "affinity": headers.get("X-Router-Affinity"), "s": round(seconds, 2), "said": said}


print("== 1. the second router's own view")
for _ in range(30):
    status, d, _, _ = call(R2 + "/router/status", timeout=5)
    pools = (d or {}).get("pools") or {}
    healthy = dict((p, dict((b.get("name"), b.get("healthy")) for b in (v.get("backends") or []))) for p, v in pools.items())
    if status == 200 and healthy and all(all(v.values()) for v in healthy.values()):
        break
    time.sleep(1.0)
print("status %s, policy %s, healthy: %s" % (status, (d or {}).get("policy"), json.dumps(healthy)))
if status != 200:
    print("keys: %s" % (sorted(d) if isinstance(d, dict) else d))
    raise SystemExit("the second router does not answer")

print("\n== 2. a forecast and a split forecast through it")
ask = {"init": "latest", "lead_hours": 48, "members": 1, "point": {"lat": 41.98, "lon": -87.9}}
for i in range(3):
    status, d, headers, seconds = call(R2 + "/v1/forecast", ask)
    print("forecast 48 h: HTTP %s in %.0f ms, by %s (GPU %s ms)" % (
        status, seconds * 1000, headers.get("X-Backend"), ((d or {}).get("timing_ms") or {}).get("gpu")))
big = dict(ask, lead_hours=120, members=20)
for i in range(3):
    status, d, headers, seconds = call(R2 + "/v1/ensemble", big)
    shards = (d or {}).get("shards") or ((d or {}).get("router") or {}).get("shards") or []
    print("ensemble 20 x 120 h: HTTP %s in %.2f s, shards %s" % (
        status, seconds, [(s.get("backend"), s.get("members"), s.get("ms")) for s in shards]))
    time.sleep(2.5)

print("\n== 3. chat through it: one session stays on one box")
by_session = {}
for session in ["s-%02d" % i for i in range(8)]:
    r = chat(session)
    by_session[session] = r["backend"]
    print("session %s: HTTP %s by %s in %s s (affinity %s) said %r" % (session, r["status"], r["backend"], r["s"], r["affinity"], r["said"]))
print("homes: %s" % dict(collections.Counter(by_session.values())))
again = [chat("s-00")["backend"] for _ in range(4)]
print("session s-00 four more times: %s (its home was %s)" % (again, by_session["s-00"]))

on_b = [s for s, b in by_session.items() if b == "box-b"]
if not on_b:
    raise SystemExit("no session landed on box-b; the failover part needs one")
pinned = on_b[0]

print("\n== 4. the LLM server on box-b is killed while session %s (home box-b) keeps chatting" % pinned)
rows, stop = [], [False]
t0 = time.time()


def talker():
    while not stop[0]:
        t = time.time() - t0
        r = chat(pinned, timeout=60)
        r["t"] = round(t, 1)
        rows.append(r)
        time.sleep(max(0.0, 1.0 - (time.time() - t0 - t)))


thread = threading.Thread(target=talker, daemon=True)
thread.start()
time.sleep(8.0)
kill_t = time.time() - t0
status, d, _, _ = call(STACK_B + "/servers/llm/kill", post=True, timeout=15)
print("kill at %.1f s: HTTP %s %s" % (kill_t, status, json.dumps(d)[:140]))
back_t = None
while time.time() - t0 < 150:
    status, _, _, _ = call(LLM_B + "/health", timeout=2)
    if status == 200 and time.time() - t0 > kill_t + 1.5:
        back_t = time.time() - t0
        break
    time.sleep(0.5)
print("box-b's LLM server answers /health again %s" % ("%.1f s after the kill" % (back_t - kill_t) if back_t else "NOT within 150 s"))
time.sleep(15.0)
stop[0] = True
thread.join(timeout=70)
for r in rows:
    print("  t %5.1f  HTTP %s  by %-5s  retries %s  affinity %-5s  %5.2f s  %r" % (
        r["t"], r["status"], r["backend"], r["retries"], r["affinity"], r["s"], r["said"]))
bad = [r for r in rows if r["status"] != 200]
after_kill = [r for r in rows if r["t"] >= kill_t]
print("chat requests: %d, failed: %d; after the kill: %d answered by %s; slowest %.2f s" % (
    len(rows), len(bad), len(after_kill), dict(collections.Counter(r["backend"] for r in after_kill if r["status"] == 200)),
    max([r["s"] for r in rows] or [0])))
status, d, _, _ = call(R2 + "/router/status", timeout=5)
pools = (d or {}).get("pools") or {}
print("at the end, healthy: %s" % json.dumps(dict((p, dict((b.get("name"), b.get("healthy")) for b in (v.get("backends") or []))) for p, v in pools.items())))
print("counters: %s" % json.dumps((d or {}).get("counters")))
