"""Make labelled incidents: run traffic, inject one fault, record, save.

Local (the default): each incident gets its own stack, two mock_server.py boxes and router.py (adaptive,
preferred box-b, like fuzzball/30-real-stack.fz), on free ports, so incidents can be recorded in parallel and a
box can really be killed. Timeline (seconds from the start of recording):

    0     traffic at the base rate (open loop, Poisson), recorder at 1 Hz
    20    inject one fault, random type, box and severity:
            slow      /admin/mode slow x1.5-4                      truth box-X:slow
            error     /admin/mode error                            truth box-X:error
            hang      /admin/mode hang                             truth box-X:hang
            down      SIGKILL the box's process                    truth box-X:down
            overload  traffic x2-4                                 truth load:overload
            misconfig router policy -> static or round_robin while box-a has been slow (x2-3) since t=0
                                                                   truth router:misconfig
          optional noise: a 3-5 s slow blip (x2-3) on the other box, somewhere in [12 s, 35 s]
    25    the alert (t0)
    60    stop; save {id, truth, alert, fault, noise, buffers} to data/incidents/<id>.json

Real stack (--real, only when the team agrees: it changes the shared stack for a minute):
    slow = supervisor POST /jobs/finetune/start (stopped again at the end), down = POST /servers/forecast/kill.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from whatif import config as wcfg
from whatif.telemetry import Recorder, save_incident

ROOT = wcfg.ROOT
FAULTS = ("slow", "error", "hang", "down", "overload", "misconfig")
INJECT_S, ALERT_DELAY_S, DURATION_S = 20.0, 5.0, 60.0
BASE_RPS = 8.0
HOURS = (24, 48, 72, 120)


@dataclass
class Fault:
    kind: str
    box: str | None = None
    severity: float | None = None
    detail: str = ""
    noise: dict | None = None

    @property
    def truth(self) -> str:
        if self.kind == "overload":
            return "load:overload"
        if self.kind == "misconfig":
            return "router:misconfig"
        return f"{self.box}:{self.kind}"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "box": self.box, "severity": self.severity, "detail": self.detail, "noise": self.noise}


def draw_fault(rng: random.Random, kinds: tuple[str, ...] = FAULTS, noise_p: float = 0.5) -> Fault:
    kind = rng.choice(kinds)
    box = rng.choice(wcfg.BOXES)
    f = Fault(kind, box if kind not in ("overload", "misconfig") else None)
    if kind == "slow":
        f.severity = round(rng.uniform(1.5, 4.0), 2)
    elif kind == "overload":
        f.severity = round(rng.uniform(2.0, 4.0), 2)
    elif kind == "misconfig":
        f.severity = round(rng.uniform(2.0, 3.0), 2)
        f.detail = rng.choice(("static", "round_robin"))
    if rng.random() < noise_p:
        other = [b for b in wcfg.BOXES if b != f.box] if f.box else list(wcfg.BOXES)
        f.noise = {"box": rng.choice(other), "at": round(rng.uniform(12, 35), 1),
                   "dur": round(rng.uniform(3, 5), 1), "factor": round(rng.uniform(2, 3), 2)}
    return f


# -------------------------------------------------------------------- a private local stack
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalStack:
    def __init__(self, seed: int):
        self.seed = seed
        self.ports = {"router": free_port(), "box-a": free_port(), "box-b": free_port()}
        self.procs: dict[str, subprocess.Popen] = {}

    @property
    def router(self) -> str:
        return f"http://127.0.0.1:{self.ports['router']}"

    @property
    def boxes(self) -> dict[str, str]:
        return {b: f"http://127.0.0.1:{self.ports[b]}" for b in wcfg.BOXES}

    def _spawn(self, name: str, args: list[str]) -> None:
        self.procs[name] = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, start_new_session=True)

    async def start(self) -> None:
        for i, b in enumerate(wcfg.BOXES):
            self._spawn(b, ["mock_server.py", "--name", b, "--port", str(self.ports[b]), "--seed", str(self.seed * 10 + i)])
        await self._wait([f"{u}/health" for u in self.boxes.values()])
        self._spawn("router", ["router.py", "--port", str(self.ports["router"]), "--policy", "adaptive",
                               "--preferred", "box-b", *sum((["--backend", f"{b}={u}"] for b, u in self.boxes.items()), [])])
        await self._wait([f"{self.router}/health"])

    async def _wait(self, urls: list[str], timeout: float = 20.0) -> None:
        end = time.monotonic() + timeout
        async with httpx.AsyncClient() as c:
            for u in urls:
                while True:
                    try:
                        if (await c.get(u, timeout=1)).status_code < 500:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > end:
                        raise RuntimeError(f"{u} did not come up")
                    await asyncio.sleep(0.2)

    def kill(self, name: str) -> None:
        p = self.procs.get(name)
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)

    def stop(self) -> None:
        for name, p in self.procs.items():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in self.procs.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


# -------------------------------------------------------------------- traffic
class Traffic:
    """Open-loop Poisson arrivals; `rate` can change while it runs. Rows: (t_sent, ms, status, box)."""

    def __init__(self, router: str, rate: float, rng: random.Random, timeout_s: float = 15.0):
        self.router, self.rate, self.rng, self.timeout_s = router, rate, rng, timeout_s
        self.rows: list[tuple] = []
        self.tasks: set = set()

    async def one(self, client: httpx.AsyncClient) -> None:
        body = {"lat": round(self.rng.uniform(25, 50), 2), "lon": round(self.rng.uniform(-125, -70), 2),
                "hours_ahead": self.rng.choice(HOURS), "step_hours": 6,
                "ensemble_members": 3 if self.rng.random() < 0.1 else 1, "request_id": uuid.uuid4().hex[:8]}
        t0 = time.time()
        try:
            r = await client.post(f"{self.router}/v1/forecast", json=body, timeout=self.timeout_s)
            self.rows.append((round(t0, 3), round((time.time() - t0) * 1000, 1), r.status_code, r.headers.get("x-served-by")))
        except httpx.HTTPError:
            self.rows.append((round(t0, 3), round((time.time() - t0) * 1000, 1), 0, None))

    async def run(self, until: float) -> None:
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=500)) as client:
            while time.time() < until:
                await asyncio.sleep(self.rng.expovariate(self.rate))
                t = asyncio.create_task(self.one(client))
                self.tasks.add(t)
                t.add_done_callback(self.tasks.discard)
            if self.tasks:
                await asyncio.wait(self.tasks, timeout=self.timeout_s)


# -------------------------------------------------------------------- one incident
async def _mode(client: httpx.AsyncClient, url: str, mode: str, factor: float = 3.0) -> None:
    try:
        await client.post(f"{url}/admin/mode", json={"mode": mode, "slow_factor": factor}, timeout=2)
    except httpx.HTTPError:
        pass


async def record_local(fault: Fault, seed: int, out_dir: Path | None, base_rps: float = BASE_RPS,
                       inject_s: float = INJECT_S, duration_s: float = DURATION_S) -> dict:
    rng = random.Random(seed)
    stack = LocalStack(seed)
    await stack.start()
    try:
        rec = Recorder(stack.router, stack.boxes)
        traffic = Traffic(stack.router, base_rps, rng)
        async with httpx.AsyncClient() as c:
            if fault.kind == "misconfig":
                await _mode(c, stack.boxes["box-a"], "slow", fault.severity)
            start = time.time()
            rec_task = asyncio.create_task(rec.run(duration_s))
            tr_task = asyncio.create_task(traffic.run(start + duration_s))

            async def at(t: float, fn):
                await asyncio.sleep(max(0.0, start + t - time.time()))
                await fn()

            async def inject():
                if fault.kind in ("slow", "error", "hang"):
                    await _mode(c, stack.boxes[fault.box], fault.kind, fault.severity or 3.0)
                elif fault.kind == "down":
                    stack.kill(fault.box)
                elif fault.kind == "overload":
                    traffic.rate = base_rps * fault.severity
                elif fault.kind == "misconfig":
                    await c.post(f"{stack.router}/router/config", json={"policy": fault.detail}, timeout=2)

            jobs = [at(inject_s, inject)]
            if fault.noise:
                n = fault.noise
                jobs.append(at(n["at"], lambda: _mode(c, stack.boxes[n["box"]], "slow", n["factor"])))
                # Noise is never on the faulty box. The one box with a standing condition is box-a under
                # misconfig (slow since t=0): the blip ends by going back to that, not to normal.
                if fault.kind == "misconfig" and n["box"] == "box-a":
                    end_blip = lambda: _mode(c, stack.boxes["box-a"], "slow", fault.severity)
                else:
                    end_blip = lambda: _mode(c, stack.boxes[n["box"]], "normal")
                jobs.append(at(n["at"] + n["dur"], end_blip))
            await asyncio.gather(*jobs)
            await rec_task
            await tr_task
    finally:
        stack.stop()
    inc_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(start))}-{fault.truth.replace(':', '-')}-{seed}"
    rows = traffic.rows
    incident = {"id": inc_id, "truth": fault.truth, "alert": round(start + inject_s + ALERT_DELAY_S, 3),
                "injected": round(start + inject_s, 3), "started": round(start, 3), "fault": fault.as_dict(),
                "seed": seed, "where": "local", "buffers": rec.snapshot(),
                "traffic_summary": {"requests": len(rows), "errors": sum(1 for r in rows if r[2] != 200)}}
    if out_dir:
        save_incident(Path(out_dir) / f"{inc_id}.json", incident)
    return incident


async def record_many(n: int, out_dir: Path, seed: int = 1, parallel: int = 4, kinds: tuple[str, ...] = FAULTS,
                      noise_p: float = 0.5, base_rps: float = BASE_RPS, balanced: bool = True) -> list[dict]:
    """n incidents, `parallel` at a time. balanced: cycle through the fault kinds so each gets its share."""
    rng = random.Random(seed)
    faults = []
    for i in range(n):
        f = draw_fault(rng, (kinds[i % len(kinds)],) if balanced else kinds, noise_p)
        faults.append((f, seed * 1000 + i))
    sem = asyncio.Semaphore(parallel)
    done: list[dict] = []

    async def one(f: Fault, s: int):
        async with sem:
            try:
                inc = await record_local(f, s, out_dir, base_rps=base_rps)
                done.append(inc)
                print(f"  recorded {inc['id']}  truth={inc['truth']}  noise={'yes' if f.noise else 'no'}  "
                      f"requests={inc['traffic_summary']['requests']}", flush=True)
            except Exception as e:  # one broken incident must not stop the batch
                print(f"  FAILED {f.truth} seed {s}: {e.__class__.__name__}: {e}", flush=True)

    await asyncio.gather(*(one(f, s) for f, s in faults))
    return done


# -------------------------------------------------------------------- the real stack (opt-in)
async def record_real(kind: str, box: str, cfg: wcfg.Config, out_dir: Path, base_rps: float = 4.0) -> dict:
    """slow = start the fine-tune job on `box`; down = kill its forecast server. Traffic goes through the real
    router; the job is stopped again at the end (the killed server comes back by itself)."""
    if kind not in ("slow", "down"):
        raise ValueError("the real stack supports slow (fine-tune job) and down (kill the forecast server)")
    rng = random.Random()
    rec = Recorder(cfg.router, cfg.boxes, cfg.stacks)
    traffic = Traffic(cfg.router, base_rps, rng, timeout_s=40)
    start = time.time()
    async with httpx.AsyncClient() as c:
        rec_task = asyncio.create_task(rec.run(DURATION_S))
        tr_task = asyncio.create_task(traffic.run(start + DURATION_S))
        await asyncio.sleep(INJECT_S)
        path = "/jobs/finetune/start" if kind == "slow" else "/servers/forecast/kill"
        r = await c.post(f"{cfg.stacks[box]}{path}", timeout=10)
        print(f"  injected: POST {cfg.stacks[box]}{path} -> {r.status_code}")
        await rec_task
        await tr_task
        if kind == "slow":
            await c.post(f"{cfg.stacks[box]}/jobs/finetune/stop", timeout=10)
    truth = f"{box}:{kind}"
    inc_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(start))}-real-{truth.replace(':', '-')}"
    incident = {"id": inc_id, "truth": truth, "alert": round(start + INJECT_S + ALERT_DELAY_S, 3),
                "injected": round(start + INJECT_S, 3), "started": round(start, 3),
                "fault": {"kind": kind, "box": box, "detail": path}, "where": "real", "buffers": rec.snapshot()}
    save_incident(Path(out_dir) / f"{inc_id}.json", incident)
    return incident


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=10, help="incidents to record (default 10)")
    ap.add_argument("--out", default=str(ROOT / "data" / "incidents"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kinds", default=",".join(FAULTS))
    ap.add_argument("--noise", type=float, default=0.5, help="chance of a slow blip on the other box")
    ap.add_argument("--rps", type=float, default=BASE_RPS)
    ap.add_argument("--real", metavar="KIND:BOX", help="one incident on the real stack, e.g. slow:box-a")
    a = ap.parse_args(argv)
    if a.real:
        kind, _, box = a.real.partition(":")
        inc = asyncio.run(record_real(kind, box, wcfg.load(), Path(a.out)))
        print(f"saved {inc['id']}")
        return
    kinds = tuple(k.strip() for k in a.kinds.split(",") if k.strip())
    bad = [k for k in kinds if k not in FAULTS]
    if bad:
        ap.error(f"unknown kinds {bad}; choose from {FAULTS}")
    t = time.time()
    done = asyncio.run(record_many(a.n, Path(a.out), a.seed, a.parallel, kinds, a.noise, a.rps))
    print(f"{len(done)}/{a.n} incidents in {time.time() - t:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
