#!/usr/bin/env python3
"""End to end: one question, traced through every system, each hop checked against a second source.

  question -> Nemotron (LLM on a GPU) -> tool call -> wxrouter API -> router -> GPU box -> forecast
           -> back to Nemotron -> answer in words

  python router_switching/e2e_check.py [--api http://127.0.0.1:18180] [--llm http://127.0.0.1:28090/v1]
      [--router http://127.0.0.1:28000] ["question"]

Each line is PASS or FAIL with the evidence. Nothing is changed on the cluster: the only tool offered
to the LLM here is get_forecast. Standard library only.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

FAILS = []


def call(url, body=None, timeout=120):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), round((time.perf_counter() - t) * 1000), None
    except urllib.error.HTTPError as e:
        return None, round((time.perf_counter() - t) * 1000), f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except (OSError, ValueError) as e:
        return None, round((time.perf_counter() - t) * 1000), str(e)


def line(ok, what, evidence):
    if not ok:
        FAILS.append(what)
    print(f"  {'PASS' if ok else 'FAIL'}  {what}\n        {evidence}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", default="What will the temperature be in Chicago (lat 41.98, lon -87.9) over the next 24 hours? Use the forecast tool.")
    ap.add_argument("--api", default="http://127.0.0.1:18180")
    ap.add_argument("--llm", default="http://127.0.0.1:28090/v1")
    ap.add_argument("--router", default="http://127.0.0.1:28000")
    ap.add_argument("--model", default="nemotron")
    a = ap.parse_args()

    print("A. Every system is up")
    st, ms, err = call(f"{a.api}/api/v1/status", timeout=40)
    if not line(st is not None, "wxrouter API answers", err or f"GET /api/v1/status in {ms} ms"):
        sys.exit(1)
    fb = st["fuzzball"]
    if fb.get("available"):
        line(fb.get("status") == "STARTED", "Fuzzball workflow is running", f"{fb.get('name')} {fb.get('id')} status {fb.get('status')}")
    else:   # e.g. the API runs inside a Fuzzball container, which has no fuzzball CLI: not a failure of the system
        print(f"  N/A   Fuzzball workflow status\n        this API server cannot ask Fuzzball: {fb.get('reason')}", flush=True)
    for s in fb.get("stages", []):
        if s["kind"] == "Service":
            line(s["status"] == "STARTED", f"service {s['name']}", f"{s['status']} on {s['node'] or 'no node'}, GPUs {s['gpus']}, up {s['running_for']}")
    for n, g in st["gpus"].items():
        line(g["state"] == "ONLINE" and g["kind"] == "real GPU" and g["cuda"] is True, f"{n} is a real GPU on CUDA",
             f"{g['state']}, {g['kind']}, node {g['node']}, model {g['model']['version']} ({g['model']['params']} params), "
             f"GPU util {g['gpu']['utilization']}, {g['gpu']['memory_process_mb']} MiB, {g['gpu']['temperature_c']} C")
    r = st["routing"]
    line(r["state"] == "ONLINE" and all(b["healthy"] for b in r["boxes"].values()), "router sees both boxes healthy",
         f"rule {r.get('policy')}, healthy {{{', '.join(f'{n}: {b['healthy']}' for n, b in r.get('boxes', {}).items())}}}")
    models, ms, err = call(f"{a.llm}/models", timeout=15)
    line(models is not None, "Nemotron (LLM server) answers", err or f"GET {a.llm}/models in {ms} ms")
    page_ok = True
    try:
        with urllib.request.urlopen(a.api + "/", timeout=10) as p:
            page_ok = p.status == 200 and b"wxrouter" in p.read()
    except OSError:
        page_ok = False
    line(page_ok, "dashboard page loads", a.api + "/")
    if FAILS:
        sys.exit(f"\nStopped: {len(FAILS)} system check(s) failed, so the question was not sent.")

    print(f"\nB. The question\n        \"{a.question}\"")
    tools_doc, _, err = call(f"{a.api}/api/v1/tools")
    tools = [t for t in tools_doc["tools"] if t["function"]["name"] == "get_forecast"]
    box_before = {n: g["requests"]["total"] for n, g in st["gpus"].items()}
    rt0, _, _ = call(f"{a.router}/router/metrics?window=1")
    t_router0 = rt0["now"] if rt0 else None
    messages = [{"role": "system", "content": "Answer weather questions only from the get_forecast tool. Quote the temperatures it returns and say which box served the request."},
                {"role": "user", "content": a.question}]

    print("\nC. Question -> Nemotron -> tool call")
    out, llm1_ms, err = call(f"{a.llm}/chat/completions", {"model": a.model, "messages": messages, "tools": tools, "temperature": 0.1}, 240)
    calls = (out or {}).get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    if not line(bool(calls) and calls[0]["function"]["name"] == "get_forecast", "Nemotron chose the forecast tool",
                err or (f"{llm1_ms} ms; arguments {calls[0]['function']['arguments']}" if calls else f"it answered without a tool: {str((out or {}).get('choices', [{}])[0].get('message', {}).get('content'))[:200]}")):
        sys.exit(1)
    args = json.loads(calls[0]["function"]["arguments"]) if isinstance(calls[0]["function"]["arguments"], str) else calls[0]["function"]["arguments"]

    print("\nD. Tool call -> wxrouter API -> router -> GPU")
    res, api_ms, err = call(f"{a.api}/api/v1/tools/get_forecast", args)
    if not line(res is not None, "the API ran the forecast", err or f"POST /api/v1/tools/get_forecast in {api_ms} ms"):
        sys.exit(1)
    f = res["result"]
    line(f["served_by"] in st["gpus"], "a GPU box served it", f"served by {f['served_by']} on {st['gpus'].get(f['served_by'], {}).get('node')}, rule {f['policy']}, failover {f['failover']}")
    line(isinstance(f["gpu_ms"], (int, float)) and f["gpu_ms"] > 0, "the GPU did the work", f"GPU compute {f['gpu_ms']} ms, round trip {f['round_trip_ms']} ms, model {f['model_version']}")
    temps = [p["temp_c"] for p in f["forecast"]]
    line(len(temps) > 0, "the model returned a forecast", f"{f['points']} points; first: +{f['forecast'][0]['lead_hours']} h {temps[0]} C, {f['forecast'][0]['wind_ms']} m/s, {f['forecast'][0].get('pressure_hpa')} hPa")

    print("\nE. Independent evidence that the request really went through")
    time.sleep(1.5)
    rt, _, _ = call(f"{a.router}/router/metrics?window=10&since={t_router0}") if t_router0 is not None else (None, 0, None)
    rows = [x for x in (rt or {}).get("requests", []) if x["box"] == f["served_by"] and x["status"] == 200]
    line(bool(rows), "the router's own request log has it", f"{len(rows)} request(s) to {f['served_by']} since the question, router-side latency {rows[-1]['ms']} ms" if rows else "no matching row in GET /router/metrics")
    st2, _, _ = call(f"{a.api}/api/v1/gpus", timeout=40)
    after = st2["gpus"][f["served_by"]]["requests"]["total"] if st2 else None
    line(after is not None and after > box_before[f["served_by"]], "the box's own request counter went up",
         f"{f['served_by']}: {box_before[f['served_by']]} -> {after} (includes the router's probes)")

    print("\nF. Result -> Nemotron -> answer")
    messages += [{"role": "assistant", "content": "", "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0].get("id", "get_forecast"), "content": json.dumps(res)[:6000]}]
    out, llm2_ms, err = call(f"{a.llm}/chat/completions", {"model": a.model, "messages": messages, "tools": tools, "temperature": 0.1}, 240)
    answer = ((out or {}).get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    line(bool(answer), "Nemotron answered in words", err or f"{llm2_ms} ms, {len(answer)} characters")
    quoted = [t for t in temps if re.search(rf"(?<![\d.]){re.escape(f'{t:g}')}(?!\d)", answer) or re.search(rf"(?<![\d.]){re.escape(f'{t:.1f}')}(?!\d)", answer)]
    line(bool(quoted), "its answer quotes temperatures the model actually returned", f"{len(quoted)} of {len(temps)} returned values appear in the answer (e.g. {quoted[:3]})" if quoted else f"none of {temps} appear in the answer")
    line(f["served_by"].replace("-", "") in answer.replace("-", "").replace("‑", "").lower(), "its answer names the box that served it", f"expected {f['served_by']}")

    print(f"\nTimings: Nemotron decides {llm1_ms} ms -> API+router+GPU {api_ms} ms (GPU {f['gpu_ms']} ms) -> Nemotron writes {llm2_ms} ms")
    print("\nNemotron's answer:\n" + answer)
    print(f"\nRESULT: {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failed{': ' + '; '.join(FAILS) if FAILS else ''})")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
