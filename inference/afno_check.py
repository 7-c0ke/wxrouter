"""Check and time the trained AFNO model behind the inference engine, on one GPU.

Starts two servers on this machine, one after the other, with the same weights:

  team      serve/forecast_server.py   (the weather team's server, one request at a time)
  engine    inference.server --model afno   (this package: batches rows across requests)

For each it sends the same forecasts and compares the answers, then measures
latency and throughput with 1 and N clients at once. Only httpx is needed here;
the servers need PyTorch and PhysicsNeMo.

  python -m inference.afno_check --weights /wx/bundle/weights/afno_event.pt --data /wx/bundle --repo /wx/repo
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent.parent
BEGIN, END = "===WXROUTER_AFNO_JSON_BEGIN===", "===WXROUTER_AFNO_JSON_END==="
CHECKS = [{"lat": 41.98, "lon": -87.9, "hours_ahead": 48, "step_hours": 6},
          {"lat": -33.87, "lon": 151.21, "hours_ahead": 120, "step_hours": 6, "ensemble_members": 4},
          {"lat": 64.1, "lon": -21.9, "hours_ahead": 20, "step_hours": 1}]
LOAD = {"lat": 37.77, "lon": -122.42, "hours_ahead": 48, "step_hours": 6}


def pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


async def wait_health(client: httpx.AsyncClient, url: str, proc: subprocess.Popen, timeout: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"FAILED: server at {url} exited with code {proc.returncode}")
        try:
            if (await client.get(f"{url}/health", timeout=2)).status_code == 200:
                return round(time.perf_counter() - t0, 2)
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    raise SystemExit(f"FAILED: {url} not healthy after {timeout:.0f}s")


async def load(client: httpx.AsyncClient, url: str, clients: int, requests: int) -> dict:
    lat, comp, errors = [], [], 0
    todo = iter(range(requests))

    async def one_client() -> None:
        nonlocal errors
        for _ in todo:
            t = time.perf_counter()
            try:
                r = await client.post(f"{url}/v1/forecast", json=LOAD, timeout=120)
                if r.status_code != 200:
                    errors += 1
                    continue
                lat.append((time.perf_counter() - t) * 1000)
                comp.append(r.json()["timing"]["compute_ms"])
            except httpx.HTTPError:
                errors += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one_client() for _ in range(clients)))
    wall = time.perf_counter() - t0
    return {"clients": clients, "requests": requests, "errors": errors, "requests_per_s": round(len(lat) / wall, 2),
            "p50_ms": pct(lat, 0.5) if lat else None, "p95_ms": pct(lat, 0.95) if lat else None,
            "p99_ms": pct(lat, 0.99) if lat else None, "compute_p50_ms": pct(comp, 0.5) if comp else None}


async def run_server(name: str, cmd: list[str], port: int, a, env: dict) -> dict:
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(cmd, cwd=a.repo, env=env)  # noqa: ASYNC220 - returns at once
    try:
        async with httpx.AsyncClient() as client:
            ready_s = await wait_health(client, url, proc, a.ready_timeout)
            answers = []
            for body in CHECKS:
                r = await client.post(f"{url}/v1/forecast", json=body, timeout=120)
                r.raise_for_status()
                answers.append(r.json())
            await load(client, url, 1, 5)  # settle
            sweeps = [await load(client, url, c, a.requests) for c in a.clients]
            for path in ("/v1/metrics", "/metrics", "/stats"):
                r = await client.get(f"{url}{path}", timeout=5)
                if r.status_code == 200:
                    extra = r.json()
                    break
        out = {"server": name, "ready_s": ready_s, "model_version": answers[0]["model_version"],
               "init_time": answers[0].get("init_time"), "load": sweeps, "metrics": extra, "answers": answers}
        for s in sweeps:
            print(f"  {name}: {s['clients']} clients -> {s['requests_per_s']} req/s, p50 {s['p50_ms']} ms, "
                  f"p95 {s['p95_ms']} ms, GPU call p50 {s['compute_p50_ms']} ms, errors {s['errors']}", flush=True)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def compare(team: dict, engine: dict) -> dict:
    """Largest difference between the two servers' answers to the same requests."""
    worst = {"temp_c": 0.0, "wind_ms": 0.0, "pressure_hpa": 0.0}
    points = 0
    for a, b in zip(team["answers"], engine["answers"]):
        assert [p["lead_hours"] for p in a["forecast"]] == [p["lead_hours"] for p in b["forecast"]]
        series = [(a["forecast"], b["forecast"])]
        for ma, mb in zip(a.get("members") or [], b.get("members") or []):
            series.append((ma["points"], mb["points"]))
        for pa, pb in series:
            for x, y in zip(pa, pb):
                points += 1
                for k, cur in worst.items():
                    worst[k] = max(cur, abs(x[k] - y[k]))
    worst = {k: round(v, 3) for k, v in worst.items()}
    # bf16 autocast on the GPU is not bit-reproducible between batch shapes, so allow a small gap
    return {"points_compared": points, "max_abs_diff": worst, "same_init_time": team["init_time"] == engine["init_time"],
            "agree": worst["temp_c"] <= 0.5 and worst["wind_ms"] <= 0.5 and worst["pressure_hpa"] <= 1.0}


def markdown(res: dict) -> str:
    c, g = res["compare"], res["gpu"]
    lines = [f"# AFNO behind the inference engine, {res['created_at']}", "",
             f"GPU: {g}. Weights `{res['servers'][0]['model_version']}`, start time {res['servers'][0]['init_time']}.", "",
             (f"Same requests to both servers: {c['points_compared']} forecast points compared, largest difference "
             f"{c['max_abs_diff']['temp_c']} C, {c['max_abs_diff']['wind_ms']} m/s, {c['max_abs_diff']['pressure_hpa']} hPa "
             f"-> agree: **{c['agree']}**."), "",
             "One 48 h forecast per request (8 model steps):", "",
             "| server | clients | req/s | p50 ms | p95 ms | p99 ms | GPU call p50 ms | errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in res["servers"]:
        for row in s["load"]:
            lines.append(f"| {s['server']} | {row['clients']} | {row['requests_per_s']} | {row['p50_ms']} | {row['p95_ms']} | "
                         f"{row['p99_ms']} | {row['compute_p50_ms']} | {row['errors']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", default=str(HERE), help="checkout holding serve/ and weather/")
    ap.add_argument("--inference-root", default=str(HERE), help="folder holding the inference/ package")
    ap.add_argument("--clients", type=lambda s: [int(v) for v in s.split(",")], default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--ready-timeout", type=float, default=300)
    ap.add_argument("--out", default="results/afno")
    ap.add_argument("--emit-markers", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, WX_REPO=a.repo, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [a.inference_root, a.repo, env.get("PYTHONPATH", "")]))
    py = sys.executable
    team_cmd = [py, "serve/forecast_server.py", "--node", "team", "--port", "8101", "--host", "127.0.0.1",
                "--weights", a.weights, "--data", a.data]
    engine_cmd = [py, "-m", "inference.server", "--model", "afno", "--name", "engine", "--port", "8102",
                  "--require-gpu", "--timeout-s", "120", "--weights", a.weights, "--data", a.data]
    team = asyncio.run(run_server("team", team_cmd, 8101, a, env))
    engine = asyncio.run(run_server("engine", engine_cmd, 8102, a, env))
    dev = engine["metrics"].get("device", {})
    if dev.get("type") != "cuda":
        raise SystemExit(f"FAILED: the engine ran the model on {dev.get('type')}, not CUDA")
    res = {"schema": "wxrouter.afno.check/v1", "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "gpu": dev.get("name"), "node": engine["metrics"].get("node"), "compare": compare(team, engine),
           "gpu_memory": engine["metrics"].get("gpu_memory"), "servers": [team, engine]}
    for s in res["servers"]:
        s["answers"] = s["answers"][:1]  # keep one full answer as a sample
    md = markdown(res)
    print(md)
    out = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "afno_check.json").write_text(json.dumps(res, indent=2))
    (out / "table.md").write_text(md)
    print(f"wrote {out}/afno_check.json and table.md")
    if a.emit_markers:
        print(BEGIN)
        print(json.dumps(res))
        print(END)
    if not res["compare"]["agree"]:
        raise SystemExit("FAILED: the two servers disagree")


if __name__ == "__main__":
    main()
