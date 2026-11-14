"""Load generator and benchmark for the forecast router.

Three subcommands:

  trace   Make a request trace from a seed (same seed = same trace, always).
            python loadgen.py trace --seed 42 --duration 60 --rps 25 --out trace.jsonl

  run     Replay a trace against a running router; optional timed events
          (slow/normal/error/hang through the mock's /admin/mode).
            python loadgen.py run --router http://127.0.0.1:8000 --trace trace.jsonl \
                --box box-a=http://127.0.0.1:8001 --box box-b=http://127.0.0.1:8002 \
                --events "10:slow:box-b:4,30:normal:box-b"

  bench   Start two mocks and a router locally, run the SAME trace and the SAME
          events for every routing policy (fresh processes each time), including
          really killing and restarting a box, then print a comparison table.
            python loadgen.py bench --seed 42 --duration 60 --rps 25

Events: "T:ACTION:BOX[:ARG]" separated by commas, T in seconds from start.
  ACTION = slow (ARG = factor, default 3) | normal | error | hang | kill | restart
  kill/restart only work in `bench`, because they need to own the process.

The open-loop design matters: requests are sent at their scheduled time whether
or not earlier ones have answered, so a slow box shows up as queueing and tail
latency, as it would with real users.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
POLICIES = ["static", "round_robin", "least_loaded", "adaptive"]


# ------------------------------------------------------------------ trace
def make_trace(seed: int, duration: float, rps: float) -> list[dict]:
    """Poisson arrivals with a realistic size mix. Deterministic for a seed."""
    rng = random.Random(seed)
    out, t, i = [], 0.0, 0
    while True:
        t += rng.expovariate(rps)
        if t >= duration:
            return out
        r = rng.random()
        if r < 0.60:
            hours, members, kind = rng.randint(6, 24), 1, "short"
        elif r < 0.90:
            hours, members, kind = rng.randint(25, 72), 1, "medium"
        elif r < 0.98:
            hours, members, kind = rng.randint(73, 120), 1, "long"
        else:
            hours, members, kind = rng.randint(24, 72), rng.randint(4, 10), "ensemble"
        out.append({
            "t": round(t, 4), "kind": kind,
            "body": {"lat": round(rng.uniform(-60, 70), 3), "lon": round(rng.uniform(-180, 180), 3),
                     "hours_ahead": hours, "step_hours": 1, "ensemble_members": members,
                     "request_id": f"req-{seed}-{i:06d}"},
        })
        i += 1


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ events
@dataclass
class Event:
    t: float
    action: str
    box: str
    arg: float | None = None

    def label(self) -> str:
        return f"{self.t:g}s {self.action} {self.box}" + (f" x{self.arg:g}" if self.arg else "")


def parse_events(s: str | None) -> list[Event]:
    events = []
    for part in filter(None, (s or "").split(",")):
        f = part.strip().split(":")
        if len(f) < 3 or f[1] not in ("slow", "normal", "error", "hang", "kill", "restart"):
            raise SystemExit(f"bad event {part!r}; expected T:ACTION:BOX[:ARG]")
        events.append(Event(float(f[0]), f[1], f[2], float(f[3]) if len(f) > 3 else None))
    return sorted(events, key=lambda e: e.t)


def default_events(duration: float) -> str:
    """Training slows box A for a while, then box B is killed and comes back.
    Hitting both boxes means no policy wins just by never touching the bad one."""
    d = duration
    return (f"{0.15 * d:g}:slow:box-a:4,{0.45 * d:g}:normal:box-a,"
            f"{0.60 * d:g}:kill:box-b,{0.75 * d:g}:restart:box-b")


# ------------------------------------------------------------------ replay
async def replay(router: str, trace: list[dict], events: list[Event], on_event, timeout: float) -> list[dict]:
    results: list[dict] = []
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        start = time.perf_counter()

        async def one(item: dict) -> None:
            t0 = time.perf_counter()
            row = {"id": item["body"]["request_id"], "t": item["t"], "kind": item["kind"]}
            try:
                r = await client.post(f"{router}/v1/forecast", json=item["body"])
                row.update(status=r.status_code, served_by=r.headers.get("x-served-by", "?"),
                           attempts=int(r.headers.get("x-router-attempts", "1")))
            except httpx.HTTPError as e:
                row.update(status=0, served_by="none", attempts=0, error=e.__class__.__name__)
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            results.append(row)

        async def fire_events() -> None:
            for ev in events:
                await asyncio.sleep(max(0.0, ev.t - (time.perf_counter() - start)))
                print(f"    event @ {time.perf_counter() - start:5.1f}s: {ev.label()}", flush=True)
                await on_event(client, ev)

        ev_task = asyncio.create_task(fire_events())
        tasks = []
        for item in trace:
            await asyncio.sleep(max(0.0, item["t"] - (time.perf_counter() - start)))
            tasks.append(asyncio.create_task(one(item)))
        await asyncio.gather(*tasks)
        await ev_task
    return results


async def admin_event(client: httpx.AsyncClient, boxes: dict[str, str], ev: Event) -> None:
    if ev.action in ("kill", "restart"):
        raise SystemExit("kill/restart need `bench` (it owns the processes)")
    body = {"mode": ev.action, "slow_factor": ev.arg or 3.0}
    try:
        await client.post(f"{boxes[ev.box]}/admin/mode", json=body, timeout=2)
    except httpx.HTTPError as e:
        print(f"    (event failed: {e.__class__.__name__})")


# ------------------------------------------------------------------ summary
def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q  # linear interpolation
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(policy: str, rows: list[dict], router_status: dict | None) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    lat = sorted(r["latency_ms"] for r in ok)
    n = len(rows)
    share = {}
    for r in ok:
        share[r["served_by"]] = share.get(r["served_by"], 0) + 1
    return {
        "policy": policy,
        "requests": n,
        "errors": n - len(ok),
        "error_pct": round(100 * (n - len(ok)) / n, 2) if n else 0.0,
        "mean_ms": round(statistics.fmean(lat), 1) if lat else float("nan"),
        "p50_ms": round(pct(lat, 0.50), 1),
        "p95_ms": round(pct(lat, 0.95), 1),
        "p99_ms": round(pct(lat, 0.99), 1),
        "max_ms": round(lat[-1], 1) if lat else float("nan"),
        "share_a_pct": round(100 * share.get("box-a", 0) / max(1, len(ok)), 1),
        "share_b_pct": round(100 * share.get("box-b", 0) / max(1, len(ok)), 1),
        "failovers": sum(1 for r in ok if r.get("attempts", 1) > 1),
        "switches": len(router_status["adaptive"]["switches"]) if router_status and policy == "adaptive" else "",
    }


COLUMNS = [("policy", "policy"), ("requests", "reqs"), ("errors", "errors"), ("error_pct", "err %"),
           ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"), ("p99_ms", "p99 ms"), ("max_ms", "max ms"),
           ("mean_ms", "mean ms"), ("share_a_pct", "box A %"), ("share_b_pct", "box B %"),
           ("failovers", "failovers"), ("switches", "switches")]


def markdown_table(summaries: list[dict]) -> str:
    head = "| " + " | ".join(h for _, h in COLUMNS) + " |"
    sep = "|" + "|".join("---" if k == "policy" else "---:" for k, _ in COLUMNS) + "|"
    body = ["| " + " | ".join(str(s[k]) for k, _ in COLUMNS) + " |" for s in summaries]
    return "\n".join([head, sep, *body])


def write_outputs(out: Path, summaries: list[dict], meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"meta": meta, "results": summaries}, indent=2))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k, _ in COLUMNS])
        w.writeheader()
        w.writerows({k: s[k] for k, _ in COLUMNS} for s in summaries)
    md = [f"# Routing benchmark — {meta['started']}", "",
          f"- trace: seed {meta['seed']}, {meta['requests']} requests over {meta['duration']}s (~{meta['rps']} req/s)",
          f"- events: {meta['events'] or 'none'}",
          "- latency = what the client saw, successful requests only; errors = non-200 or no answer", "",
          markdown_table(summaries), ""]
    for s in summaries:
        if s.get("switch_log"):
            md += [f"**{s['policy']} switches:**", *[f"- {x['t']}s {x['from']} → {x['to']} ({x['reason']})"
                                                     for x in s["switch_log"]], ""]
    (out / "table.md").write_text("\n".join(md))


# ------------------------------------------------------------------ bench
class Procs:
    def __init__(self, py: str, base_port: int, seed: int, log_dir: Path):
        self.py, self.seed, self.log_dir = py, seed, log_dir
        self.router_port = base_port
        self.ports = {"box-a": base_port + 1, "box-b": base_port + 2}
        self.mocks: dict[str, subprocess.Popen] = {}
        self.router: subprocess.Popen | None = None
        self.logs = []

    def url(self, box: str) -> str:
        return f"http://127.0.0.1:{self.ports[box]}"

    def _spawn(self, args: list[str], name: str) -> subprocess.Popen:
        log = (self.log_dir / f"{name}.log").open("a")
        self.logs.append(log)
        return subprocess.Popen([self.py, *args], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)

    def start_mock(self, box: str) -> None:
        seed = self.seed + (1 if box == "box-a" else 2)
        self.mocks[box] = self._spawn(["mock_server.py", "--name", box, "--port", str(self.ports[box]),
                                       "--seed", str(seed)], box)

    def kill_mock(self, box: str) -> None:
        p = self.mocks.pop(box, None)
        if p:
            p.kill()
            p.wait()

    def start_router(self, policy: str, extra: list[str]) -> None:
        self.router = self._spawn(["router.py", "--port", str(self.router_port), "--policy", policy,
                                   "--backend", f"box-a={self.url('box-a')}",
                                   "--backend", f"box-b={self.url('box-b')}", *extra], "router")

    def stop_all(self) -> None:
        for b in list(self.mocks):
            self.kill_mock(b)
        if self.router:
            self.router.terminate()
            self.router.wait()
            self.router = None
        for log in self.logs:
            log.close()
        self.logs = []


async def wait_ready(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            try:
                if (await c.get(f"{url}/health", timeout=0.5)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise SystemExit(f"{url} did not become ready; see logs")


async def bench(a) -> None:
    trace = read_jsonl(Path(a.trace)) if a.trace else make_trace(a.seed, a.duration, a.rps)
    duration = trace[-1]["t"] if trace else a.duration
    events_str = a.events if a.events is not None else default_events(a.duration)
    events = parse_events(events_str)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "trace.jsonl", trace)
    policies = a.policies.split(",") if a.policies else POLICIES
    router_extra = ["--margin", str(a.margin), "--window", str(a.window), "--cooldown", str(a.cooldown),
                    "--timeout", str(a.timeout), "--return-margin", str(a.return_margin)]
    if a.preferred:
        router_extra += ["--preferred", a.preferred]
    print(f"trace: {len(trace)} requests over {duration:.0f}s, seed {a.seed}")
    print(f"events: {events_str or 'none'}\nresults: {out}\n")

    summaries = []
    for policy in policies:
        print(f"== {policy}")
        log_dir = out / "logs" / policy
        log_dir.mkdir(parents=True, exist_ok=True)
        procs = Procs(sys.executable, a.base_port, a.seed, log_dir)
        try:
            for box in ("box-a", "box-b"):
                procs.start_mock(box)
            await asyncio.gather(*(wait_ready(procs.url(b)) for b in ("box-a", "box-b")))
            procs.start_router(policy, router_extra)
            router_url = f"http://127.0.0.1:{procs.router_port}"
            await wait_ready(router_url)
            await asyncio.sleep(1.5)  # let probes warm up

            async def on_event(client, ev: Event):
                if ev.action == "kill":
                    procs.kill_mock(ev.box)
                elif ev.action == "restart":
                    procs.start_mock(ev.box)
                else:
                    await admin_event(client, {b: procs.url(b) for b in procs.ports}, ev)

            rows = await replay(router_url, trace, events, on_event, a.client_timeout)
            async with httpx.AsyncClient() as c:
                status = (await c.get(f"{router_url}/router/status", timeout=2)).json()
        finally:
            procs.stop_all()
        write_jsonl(out / f"raw-{policy}.jsonl", sorted(rows, key=lambda r: r["t"]))
        s = summarize(policy, rows, status)
        s["switch_log"] = status["adaptive"]["switches"] if policy == "adaptive" else []
        summaries.append(s)
        print(f"    p50 {s['p50_ms']} ms | p95 {s['p95_ms']} | p99 {s['p99_ms']} | errors {s['errors']}"
              f" | A {s['share_a_pct']}% B {s['share_b_pct']}%\n")

    meta = {"started": stamp, "seed": a.seed, "duration": round(duration), "rps": a.rps,
            "requests": len(trace), "events": events_str}
    write_outputs(out, summaries, meta)
    print(markdown_table(summaries))
    print(f"\nwrote {out / 'table.md'}, summary.csv, summary.json, raw-*.jsonl")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="make a seeded trace")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--duration", type=float, default=60)
    t.add_argument("--rps", type=float, default=25)
    t.add_argument("--out", default="trace.jsonl")

    r = sub.add_parser("run", help="replay a trace against a running router")
    r.add_argument("--router", default="http://127.0.0.1:8000")
    r.add_argument("--trace", required=True)
    r.add_argument("--box", action="append", default=[], help="NAME=URL for admin events")
    r.add_argument("--events", default="")
    r.add_argument("--client-timeout", type=float, default=30)
    r.add_argument("--label", default="run", help="name for this row in the table")
    r.add_argument("--out", default="results")

    b = sub.add_parser("bench", help="compare all policies on local mocks")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--duration", type=float, default=60)
    b.add_argument("--rps", type=float, default=25)
    b.add_argument("--trace", help="reuse a saved trace instead of generating one")
    b.add_argument("--events", default=None, help="default: slow A, then kill and restart B")
    b.add_argument("--policies", default="", help="comma list; default all four")
    b.add_argument("--base-port", type=int, default=18000, help="router; mocks on +1, +2")
    b.add_argument("--margin", type=float, default=0.25)
    b.add_argument("--window", type=float, default=1.0)
    b.add_argument("--cooldown", type=float, default=10.0)
    b.add_argument("--preferred", default="", help="adaptive home box, e.g. box-b")
    b.add_argument("--return-margin", type=float, default=0.20)
    b.add_argument("--timeout", type=float, default=5.0, help="router per-attempt timeout")
    b.add_argument("--client-timeout", type=float, default=30)
    b.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.cmd == "trace":
        tr = make_trace(a.seed, a.duration, a.rps)
        write_jsonl(Path(a.out), tr)
        print(f"wrote {len(tr)} requests to {a.out}")
    elif a.cmd == "run":
        trace = read_jsonl(Path(a.trace))
        boxes = dict(x.split("=", 1) for x in a.box)
        events = parse_events(a.events)
        rows = asyncio.run(replay(a.router, trace, events,
                                  lambda c, ev: admin_event(c, boxes, ev), a.client_timeout))
        try:
            status = httpx.get(f"{a.router}/router/status", timeout=2).json()
        except httpx.HTTPError:
            status = None
        s = summarize(status["policy"] if status else a.label, rows, status)
        s["switch_log"] = status["adaptive"]["switches"] if status and s["policy"] == "adaptive" else []
        out = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "raw.jsonl", sorted(rows, key=lambda r: r["t"]))
        write_outputs(out, [s], {"started": out.name, "seed": "from trace", "duration": round(trace[-1]["t"]),
                                 "rps": round(len(trace) / max(1, trace[-1]["t"]), 1),
                                 "requests": len(trace), "events": a.events})
        print(markdown_table([s]))
    else:
        asyncio.run(bench(a))


if __name__ == "__main__":
    main()
