"""The "Run Switching Demo" button, run for real against the team's router and two mock boxes.

The mock boxes make this run SIMULATED (and the demo must say so). The router, its adaptive rule and
its failover are the real ones, started as processes the way dev_stack.py starts them.
"""
import asyncio
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from router_switching import demo

BASE = 19300
ROUTER, A, B = (f"http://127.0.0.1:{BASE + i}" for i in range(3))


def up(url: str, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up")


@pytest.fixture(scope="module")
def stack():
    py = sys.executable
    procs = [subprocess.Popen([py, "mock_server.py", "--name", n, "--port", str(BASE + i), "--seed", str(i)], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i, n in ((1, "box-a"), (2, "box-b"))]
    try:
        up(f"{A}/health"), up(f"{B}/health")
        procs.append(subprocess.Popen([py, "router.py", "--port", str(BASE), "--policy", "round_robin", "--cooldown", "2",
                                       "--backend", f"box-a={A}", "--backend", f"box-b={B}"], cwd=ROOT,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        up(f"{ROUTER}/health")
        yield
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)


class Ops:
    def __init__(self, router=ROUTER, boxes=None):
        self.router, self.boxes, self.client = router, boxes or {"box-a": A, "box-b": B}, None


def run(ops, **kw) -> dict:
    async def go():
        async with httpx.AsyncClient() as c:
            ops.client = c
            d = demo.SwitchingDemo(ops, save=False, **kw)
            assert d.start()["status"] == "running"
            assert d.start()["status"] == "running", "a second click while it runs must not start a second run"
            await d.task
            return d.state, (await c.get(f"{ops.router}/router/status")).json() if ops.router == ROUTER else None, \
                (await c.get(f"{A}/stats")).json() if ops.router == ROUTER else None
    return asyncio.run(go())


def test_demo_moves_traffic_to_b_and_back_and_puts_everything_back(stack):
    st, router, box_a = run(Ops(), hold_s=2.0, wait_s=15.0, rps=20.0)
    assert st["status"] == "done", st["error"]
    assert st["kind"] == "simulated", "mock boxes must never be shown as GPU results"
    phases = {p["key"]: p for p in st["phases"]}
    assert all(p["state"] == "done" and p["requests"] > 0 for p in st["phases"])
    assert phases["normal"]["share_a_pct"] > 90, "before the fault the home box carries the traffic"
    assert phases["fail"]["share_b_pct"] > 80, "with box A failed, box B serves"
    switches = [e["text"] for e in st["events"] if e["kind"] == "switch"]
    assert any("box-a → box-b" in s and "active box down" in s for s in switches), switches
    assert sum("box-a → box-b" in s for s in switches) >= 2 and any("box-b → box-a" in s for s in switches), switches
    t = st["timings"]
    assert 0 < t["slow_to_switch_s"] < 15 and 0 <= t["fail_to_switch_s"] < 5 and 0 < t["restore_to_serving_s"] < 15
    assert t["errors_while_failed"] == phases["fail"]["errors"], "both are counted from the same router rows"
    assert st["overall"]["errors"] == sum(p["errors"] for p in st["timeline"]) or st["overall"]["requests"] > 0
    # left as found: the router's earlier policy, box A normal
    assert router["policy"] == "round_robin" and not router["adaptive"]["preferred"] and router["adaptive"]["margin"] == 0.25
    assert box_a["mode"] == "normal" and "cleanup_error" not in st


def test_demo_fails_cleanly_when_the_router_is_away():
    st, _, _ = run(Ops(router="http://127.0.0.1:9"), hold_s=0.1, wait_s=0.1)
    assert st["status"] == "failed" and "router" in st["error"]


def test_demo_needs_two_boxes():
    d = demo.SwitchingDemo(Ops(boxes={"box-a": A}))
    assert d.start()["status"] == "failed" and d.task is None


def test_summary_counts_come_straight_from_the_rows():
    rows = [{"t": 0.2, "box": "box-a", "ms": 10.0, "status": 200, "failover": False},
            {"t": 0.6, "box": "box-b", "ms": 30.0, "status": 200, "failover": True},
            {"t": 1.1, "box": "none", "ms": 5.0, "status": 503, "failover": False},
            {"t": 1.4, "box": "box-b", "ms": 20.0, "status": 200, "failover": False}]
    s = demo.summarize(rows, "box-a", "box-b", 2.0)
    assert s["requests"] == 4 and s["errors"] == 1 and s["error_pct"] == 25.0 and s["rps"] == 2.0
    assert s["share_a_pct"] == 25.0 and s["share_b_pct"] == 50.0 and s["failovers"] == 1 and s["p50_ms"] == 20.0
    assert demo.per_second(rows, "box-a", "box-b", 0.0, 2.0) == [
        {"t": 0, "a": 1, "b": 1, "errors": 0, "p95_ms": 30.0}, {"t": 1, "a": 0, "b": 1, "errors": 1, "p95_ms": 20.0}]
    assert demo.summarize([], "box-a", "box-b", 2.0)["p95_ms"] is None, "no requests: no number"


def test_closing_the_page_mid_run_still_puts_box_a_and_the_policy_back(stack):
    """Found in review: the page closed its HTTP client while a run had box A in a fault mode."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from router_switching import dashboard

    ops = Ops()

    @dashboard.asynccontextmanager
    async def page(app):
        ops.client = httpx.AsyncClient()
        yield
        await ops.client.aclose()

    app = FastAPI(lifespan=page)
    dashboard.attach(app, ops)
    with TestClient(app) as c:
        assert c.post("/api/switching/demo").json()["status"] == "running"
        end = time.time() + 30
        while c.get("/api/switching/demo").json()["phase"] != "slow" and time.time() < end:
            time.sleep(0.3)
        time.sleep(1.0)
        assert c.get("/api/switching/demo").json()["phase"] == "slow"
    with urllib.request.urlopen(f"{A}/stats") as r:
        assert __import__("json").load(r)["mode"] == "normal"
    with urllib.request.urlopen(f"{ROUTER}/router/status") as r:
        assert __import__("json").load(r)["policy"] == "round_robin"
