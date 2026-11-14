"""Benchmark two real inference boxes behind router.py, including a failure scenario.

One seeded trace (loadgen.make_trace) is replayed for every case, so the cases
are comparable:

  single_gpu_no_batching   straight at one box, max_batch_rows=1
  single_gpu_batching      straight at one box, batching on
  two_gpu_<policy>         through the router, one run per routing policy
  fault_<policy>           through the router while box A is slowed, then
                           stopped, then started again

Faults are real: "slow" puts a second load on that box's GPU, "kill" stops the
server process and "restart" is the supervisor starting it again (cold start).

Runs anywhere the boxes and the router are reachable. It needs only httpx and
loadgen.py, so on Fuzzball it runs as a plain script next to loadgen.py:

  python -m inference.cluster_bench --router http://127.0.0.1:8000 \\
      --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002 --out results/cluster
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


def _root_loadgen():
    """The team's loadgen.py, loaded by path under its own name: bench/loadgen.py is also called
    `loadgen`, and whichever is imported first would otherwise win."""
    import importlib.util

    here = Path(__file__).resolve().parent
    path = next(p for p in (here / "loadgen.py", here.parent / "loadgen.py") if p.exists())
    spec = importlib.util.spec_from_file_location("wxrouter_root_loadgen", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


loadgen = _root_loadgen()

SCHEMA = "wxrouter.cluster.bench/v1"
BEGIN, END = "===WXROUTER_CLUSTER_JSON_BEGIN===", "===WXROUTER_CLUSTER_JSON_END==="
PROBE = {"lat": 40.0, "lon": -100.0, "hours_ahead": 48, "request_id": "calibrate"}


async def get_json(client: httpx.AsyncClient, url: str, timeout: float = 2.0):
    try:
        r = await client.get(url, timeout=timeout)
        return r.status_code, r.json()
    except (httpx.HTTPError, ValueError):
        return 0, None


async def post_json(client: httpx.AsyncClient, url: str, body: dict, timeout: float = 30.0):
    try:
        r = await client.post(url, json=body, timeout=timeout)
        return r.status_code, r.json()
    except (httpx.HTTPError, ValueError) as e:
        return 0, {"error": type(e).__name__}


async def wait_ready(client, boxes: dict[str, str], router: str | None, timeout: float, why: str) -> float:
    """Wait until every box answers /health 200 and the router (if any) calls every box healthy."""
    t0 = time.perf_counter()
    while True:
        codes = [(await get_json(client, f"{u}/health"))[0] for u in boxes.values()]
        ok = all(c == 200 for c in codes)
        if ok and router:
            code, st = await get_json(client, f"{router}/router/status")
            ok = code == 200 and all(b["healthy"] for b in st["backends"].values())
        waited = time.perf_counter() - t0
        if ok:
            return waited
        if waited > timeout:
            raise SystemExit(f"not ready after {timeout:.0f}s ({why}): box health codes {codes}")
        await asyncio.sleep(0.5)


async def placement(client, boxes: dict[str, str]) -> dict:
    out = {}
    for name, url in boxes.items():
        _, m = await get_json(client, f"{url}/v1/metrics", 5)
        m = m or {}
        out[name] = {"node": m.get("node"), "device": m.get("device"), "cold_start": m.get("cold_start"),
                     "model_version": m.get("model_version")}
    return out


def distinct_nodes(place: dict) -> bool | None:
    """True when every box reports a different GPU UUID (or, without one, a different first address)."""
    ids = []
    for p in place.values():
        node = p.get("node") or {}
        ids.append(node.get("gpu_uuid") or (node.get("addresses") or [None])[0])
    if any(i is None for i in ids):
        return None
    return len(set(ids)) == len(ids)


async def median_compute_ms(client, url: str, n: int = 12) -> float | None:
    vals = []
    for _ in range(n):
        code, body = await post_json(client, f"{url}/v1/forecast", PROBE)
        if code == 200:
            vals.append(body["timing"]["compute_ms"])
    return round(statistics.median(vals), 2) if vals else None


async def calibrate_slow(client, url: str, factor: float) -> dict:
    """How much the real GPU load slows one 48 h forecast on an otherwise idle box."""
    normal = await median_compute_ms(client, url)
    await post_json(client, f"{url}/admin/mode", {"mode": "slow", "slow_factor": factor})
    await asyncio.sleep(1.0)
    slow = await median_compute_ms(client, url)
    await post_json(client, f"{url}/admin/mode", {"mode": "normal"})
    return {"slow_factor_requested": factor, "compute_ms_normal": normal, "compute_ms_slow": slow,
            "measured_slowdown": round(slow / normal, 2) if normal and slow else None}


def timeline(rows: list[dict], duration: float) -> list[dict]:
    """Per second of send time: requests, errors, who served them, latency."""
    out = []
    for sec in range(int(duration) + 1):
        rs = [r for r in rows if int(r["t"]) == sec]
        if not rs:
            continue
        ok = sorted(r["latency_ms"] for r in rs if r["status"] == 200)
        by = {}
        for r in rs:
            if r["status"] == 200:
                by[r["served_by"]] = by.get(r["served_by"], 0) + 1
        out.append({"t": sec, "requests": len(rs), "errors": len(rs) - len(ok), "served_by": by,
                    "p50_ms": round(loadgen.pct(ok, 0.5), 1) if ok else None,
                    "p95_ms": round(loadgen.pct(ok, 0.95), 1) if ok else None})
    return out


def fault_report(tl: list[dict], events: list[loadgen.Event]) -> dict:
    """What happened around each fault on the faulted box: its share of traffic, errors, time to return."""
    def share(box: str, t0: float, t1: float) -> float | None:
        n = sum(sum(s["served_by"].values()) for s in tl if t0 <= s["t"] < t1)
        mine = sum(s["served_by"].get(box, 0) for s in tl if t0 <= s["t"] < t1)
        return round(100 * mine / n, 1) if n else None

    def errors(t0: float, t1: float) -> int:
        return sum(s["errors"] for s in tl if t0 <= s["t"] < t1)

    end = (tl[-1]["t"] + 1) if tl else 0
    rep = {}
    by = {(e.action, e.box): e.t for e in events}
    for (action, box), t in by.items():
        if action == "slow":
            t_end = by.get(("normal", box), end)
            rep["slow"] = {"box": box, "from_s": t, "to_s": t_end,
                           "share_before_pct": share(box, 0, t), "share_during_pct": share(box, t + 2, t_end),
                           "share_after_pct": share(box, t_end + 2, by.get(("kill", box), end)),
                           "errors_during": errors(t, t_end)}
        if action == "kill":
            t_back = by.get(("restart", box), end)
            served_again = next((s["t"] for s in tl if s["t"] >= t_back and s["served_by"].get(box, 0) > 0), None)
            rep["stop"] = {"box": box, "stopped_s": t, "restart_s": t_back,
                           "errors_while_down": errors(t, t_back), "errors_after_restart": errors(t_back, end),
                           "first_served_again_s": served_again,
                           "seconds_to_serve_again": None if served_again is None else served_again - t_back,
                           "share_last_5s_pct": share(box, end - 5, end)}
    return rep


def case_switches(status: dict | None, n_before: int, samples: list[dict]) -> list[dict]:
    """The router's switches made during this case. The router stamps them on its own clock, so
    `t_case` (seconds into the case) comes from when our samples saw the active box change."""
    if not status or status["policy"] != "adaptive":
        return []
    new = [dict(s) for s in status["adaptive"]["switches"][n_before:]]
    seen, last = [], None
    for s in samples:
        active = (s.get("router") or {}).get("active")
        if active and last and active != last:
            seen.append(s["t"])
        last = active or last
    if len(seen) == len(new):
        for sw, t in zip(new, seen):
            sw["t_case"] = t
    return new


async def run_case(client, label: str, target: str, trace: list[dict], events: list[loadgen.Event],
                   boxes: dict[str, str], router: str | None, direct_box: str | None, a) -> dict:
    samples: list[dict] = []
    stop = asyncio.Event()
    start = time.perf_counter()

    async def sampler() -> None:
        while not stop.is_set():
            row = {"t": round(time.perf_counter() - start, 2)}
            for name, url in boxes.items():
                code, s = await get_json(client, f"{url}/stats", 1.0)
                row[name] = ({"up": True, "gpu_util": s["gpu_util"], "in_flight": s["in_flight"],
                              "queue_depth": s["queue_depth"], "p50_ms": s["p50_latency_ms"], "mode": s["mode"]}
                             if code == 200 else {"up": False})
            if router:
                code, st = await get_json(client, f"{router}/router/status", 1.0)
                if code == 200:
                    row["router"] = {"active": st["adaptive"]["active"],
                                     "healthy": [n for n, b in st["backends"].items() if b["healthy"]]}
            samples.append(row)
            await asyncio.sleep(a.sample_s)

    async def on_event(_client, ev: loadgen.Event) -> None:
        url = boxes[ev.box]
        if ev.action == "kill":
            back = next((e.t for e in events if e.action == "restart" and e.box == ev.box and e.t > ev.t), ev.t + 10)
            await post_json(client, f"{url}/admin/stop", {"down_s": round(back - ev.t, 2)}, 3)
        elif ev.action == "restart":
            pass  # the box's supervisor starts it again after down_s
        else:
            await post_json(client, f"{url}/admin/mode", {"mode": ev.action, "slow_factor": ev.arg or a.slow_factor}, 10)

    before = (await get_json(client, f"{router}/router/status"))[1] if router else None
    n_before = len(before["adaptive"]["switches"]) if before else 0
    print(f"  {label}: {len(trace)} requests" + (f", events {[e.label() for e in events]}" if events else ""), flush=True)
    task = asyncio.create_task(sampler())
    rows = await loadgen.replay(target, trace, events, on_event, a.client_timeout)
    wall = time.perf_counter() - start
    stop.set()
    await task
    if direct_box:
        for r in rows:
            if r["status"] == 200:
                r["served_by"] = direct_box
    status = (await get_json(client, f"{router}/router/status"))[1] if router else None
    policy = status["policy"] if status else "direct"
    summary = loadgen.summarize(policy if policy == "adaptive" else label, rows, status)
    summary["policy"] = policy
    ok = sum(1 for r in rows if r["status"] == 200)
    summary.update(label=label, wall_s=round(wall, 2), ok_per_s=round(ok / wall, 1),
                   offered_per_s=round(len(trace) / max(trace[-1]["t"], 1e-9), 1))
    tl = timeline(rows, trace[-1]["t"])
    case = {"label": label, "policy": policy, "summary": summary, "timeline": tl, "samples": samples,
            "events": [{"t": e.t, "action": e.action, "box": e.box, "arg": e.arg} for e in events],
            "switches": case_switches(status, n_before, samples),
            "failovers": status["failovers"] if status else None}
    if events:
        case["fault_report"] = fault_report(tl, events)
    case["boxes_after"] = {}
    for name, url in boxes.items():
        _, m = await get_json(client, f"{url}/v1/metrics", 3)
        if m:
            case["boxes_after"][name] = {k: m.get(k) for k in ("load", "compute_ms", "batches", "gpu_memory", "errors")}
    if a.out:
        loadgen.write_jsonl(Path(a.out) / f"raw-{label}.jsonl", sorted(rows, key=lambda r: r["t"]))
    print(f"    -> errors {summary['errors']}, p50 {summary['p50_ms']} ms, p95 {summary['p95_ms']} ms, "
          f"p99 {summary['p99_ms']} ms, {summary['ok_per_s']} ok/s, A {summary['share_a_pct']}% B {summary['share_b_pct']}%",
          flush=True)
    return case


def markdown(res: dict) -> str:
    p = res["params"]
    lines = [f"# Two-box routing benchmark, {res['created_at']}", "",
             f"Trace: seed {p['seed']}, {p['rps']} req/s; steady cases {p['duration']} s, fault cases {p['fault_duration']} s.",
             f"Fault events: `{p['events']}`.", ""]
    for name, pl in res["placement"].items():
        node, dev = pl.get("node") or {}, pl.get("device") or {}
        lines.append(f"- **{name}**: {dev.get('name')} ({dev.get('type')}, {dev.get('precision')}), "
                     f"node addresses {node.get('addresses')}, GPU {node.get('gpu_uuid')}")
    lines += [f"- distinct nodes: **{res['distinct_nodes']}**", ""]
    cal = res.get("calibration")
    if cal:
        lines += [(f"Slow mode (factor {cal['slow_factor_requested']}): one 48 h forecast took "
                  f"{cal['compute_ms_normal']} ms idle and {cal['compute_ms_slow']} ms under the GPU load "
                  f"({cal['measured_slowdown']}x)."), ""]
    lines += ["| case | policy | reqs | errors | p50 ms | p95 ms | p99 ms | ok/s | box A % | box B % | failovers | switches |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in res["cases"]:
        s = c["summary"]
        lines.append(f"| {c['label']} | {c['policy']} | {s['requests']} | {s['errors']} | {s['p50_ms']} | {s['p95_ms']} | "
                     f"{s['p99_ms']} | {s['ok_per_s']} | {s['share_a_pct']} | {s['share_b_pct']} | {s['failovers']} | "
                     f"{len(c['switches'])} |")
    for c in res["cases"]:
        fr = c.get("fault_report")
        if not fr:
            continue
        lines += ["", f"**{c['label']}**"]
        if "slow" in fr:
            f = fr["slow"]
            lines.append(f"- {f['box']} slowed {f['from_s']:g}-{f['to_s']:g} s: its share went {f['share_before_pct']}% -> "
                         f"{f['share_during_pct']}% -> {f['share_after_pct']}% (before, during, after); "
                         f"{f['errors_during']} errors")
        if "stop" in fr:
            f = fr["stop"]
            lines.append(f"- {f['box']} stopped at {f['stopped_s']:g} s, restarted at {f['restart_s']:g} s: "
                         f"{f['errors_while_down']} errors while down, served again "
                         f"{f['seconds_to_serve_again']} s after the restart, share in the last 5 s {f['share_last_5s_pct']}%")
        for sw in c["switches"]:
            lines.append(f"- switch at {sw.get('t_case', '?')} s: {sw['from']} -> {sw['to']} ({sw['reason']})")
    return "\n".join(lines) + "\n"


async def main_async(a) -> dict:
    boxes = dict(b.split("=", 1) for b in a.box)
    names = list(boxes)
    first = names[0]
    events = loadgen.parse_events(a.events)
    steady = loadgen.make_trace(a.seed, a.duration, a.rps)
    faulty = loadgen.make_trace(a.seed, a.fault_duration, a.rps)
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
    async with httpx.AsyncClient(limits=limits) as client:
        waited = await wait_ready(client, boxes, a.router, a.ready_timeout, "start")
        print(f"ready after {waited:.1f}s", flush=True)
        place = await placement(client, boxes)
        distinct = distinct_nodes(place)
        for name, pl in place.items():
            print(f"  {name}: {json.dumps(pl['node'])} {json.dumps(pl['device'])}", flush=True)
        print(f"  distinct nodes: {distinct}", flush=True)
        if a.require_distinct_nodes and distinct is not True:
            raise SystemExit("FAILED: the boxes are not provably on different GPU nodes")
        if a.require_gpu and any((pl["device"] or {}).get("type") != "cuda" for pl in place.values()):
            raise SystemExit("FAILED: a box is not running on CUDA")
        res = {"schema": SCHEMA, "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "params": {"seed": a.seed, "rps": a.rps, "duration": a.duration, "fault_duration": a.fault_duration,
                          "events": ",".join(f"{e.t:g}:{e.action}:{e.box}" + (f":{e.arg:g}" if e.arg else "")
                                             for e in events),
                          "slow_factor": a.slow_factor, "adaptive_home_in_fault_cases": a.preferred, "policies": a.policies, "fault_policies": a.fault_policies},
               "placement": place, "distinct_nodes": distinct, "cases": []}
        if a.admin:
            res["calibration"] = await calibrate_slow(client, boxes[first], a.slow_factor)
            print(f"  slow calibration: {res['calibration']}", flush=True)

        async def settle(router: str | None) -> None:
            if a.admin:
                for url in boxes.values():
                    await post_json(client, f"{url}/admin/mode", {"mode": "normal"})
            await wait_ready(client, boxes, router, a.ready_timeout, "between cases")
            await asyncio.sleep(a.settle_s)

        if a.admin and not a.skip_single:
            one = {first: boxes[first]}
            for label, rows_per_call in (("single_gpu_no_batching", 1), ("single_gpu_batching", a.max_batch_rows)):
                await settle(None)
                await post_json(client, f"{boxes[first]}/admin/engine", {"max_batch_rows": rows_per_call})
                res["cases"].append(await run_case(client, label, boxes[first], steady, [], one, None, first, a))
            await post_json(client, f"{boxes[first]}/admin/engine", {"max_batch_rows": a.max_batch_rows})
        if a.router:
            for prefix, policies, trace, evs in (("two_gpu", a.policies, steady, []),
                                                 ("fault", a.fault_policies if a.admin else [], faulty, events)):
                for policy in policies:
                    await settle(a.router)
                    home = a.preferred if (prefix == "fault" and policy == "adaptive") else ""
                    code, _ = await post_json(client, f"{a.router}/router/config", {"policy": policy, "preferred": home})
                    if code != 200:
                        print(f"  skipping {policy}: router refused it ({code})", flush=True)
                        continue
                    await asyncio.sleep(1.0)
                    res["cases"].append(await run_case(client, f"{prefix}_{policy}", a.router, trace, evs,
                                                       boxes, a.router, None, a))
            await settle(a.router)
    return res


def names(s: str) -> list[str]:
    return [v for v in s.split(",") if v]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", default="", help="router URL; empty runs only the single-box cases")
    ap.add_argument("--box", action="append", required=True, help="NAME=URL; the first one is box A")
    ap.add_argument("--policies", type=names, default=["static", "round_robin", "least_loaded", "adaptive", "balanced"])
    ap.add_argument("--fault-policies", type=names, default=["static", "round_robin", "least_loaded", "adaptive", "balanced"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rps", type=float, default=200.0)
    ap.add_argument("--duration", type=float, default=30.0, help="steady cases, seconds")
    ap.add_argument("--fault-duration", type=float, default=60.0)
    ap.add_argument("--events", default="10:slow:box-a,25:normal:box-a,35:kill:box-a,45:restart:box-a",
                    help="T:ACTION:BOX[:ARG] list for the fault cases")
    ap.add_argument("--preferred", default="box-a", help="adaptive's home box in the fault cases; '' for none")
    ap.add_argument("--slow-factor", type=float, default=4.0)
    ap.add_argument("--max-batch-rows", type=int, default=64)
    ap.add_argument("--no-admin", dest="admin", action="store_false",
                    help="boxes have no /admin hooks: skip single-box and fault cases")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--require-distinct-nodes", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--sample-s", type=float, default=0.5)
    ap.add_argument("--client-timeout", type=float, default=30.0)
    ap.add_argument("--out", default="results/cluster", help="a <timestamp>/ folder is created inside")
    ap.add_argument("--emit-markers", action="store_true")
    a = ap.parse_args()
    a.router = a.router.rstrip("/")
    if a.out:
        a.out = str(Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S"))
        Path(a.out).mkdir(parents=True, exist_ok=True)
    res = asyncio.run(main_async(a))
    md = markdown(res)
    print(md)
    if a.out:
        (Path(a.out) / "cluster_bench.json").write_text(json.dumps(res, indent=2))
        (Path(a.out) / "table.md").write_text(md)
        print(f"wrote {a.out}/cluster_bench.json, table.md and raw-*.jsonl")
    if a.emit_markers:
        print(BEGIN)
        print(json.dumps(res))
        print(END)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
