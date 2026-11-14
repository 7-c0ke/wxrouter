"""Start a local test stack for the Control Room (ui/): OUR forecast server under OUR supervisor (twice), the router, the page.

  python ui/dev_stack_supervised.py                     # page on http://127.0.0.1:19080
  python ui/dev_stack_supervised.py --ui-port 19081 --job-seconds 20

What it starts, each as a child of this script (Ctrl-C stops them all):
  serve/node_stack.py --admin   two supervisors, box-a on port 19109 and box-b on port 19110. Each keeps
                                serve/forecast_server.py --mock --admin (ports 19001 and 19002) alive, and
                                holds a job named "finetune".
  router.py                     port 19000, with the backends box-a and box-b
  ui/server.py                 port 19080, with --box and --stack for both boxes

Nothing here is a real model. The forecast servers run in mock mode (they answer "precip_mm": null, as
the real one does). The "finetune" job is a harmless fake: it trains nothing and writes nothing; for
about 40 s it prints one line a second in the format of weather/finetune.py, so the page has a job to
start, watch and stop. It is this same file, run with --fake-finetune.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STOP_WAIT_S = 20.0      # a supervisor gives its own children 10 s to stop before it kills them


def fake_finetune(argv: list[str]) -> int:
    """The harmless stand-in for weather/finetune.py: the same progress lines, and nothing else."""
    ap = argparse.ArgumentParser(prog="dev_stack_supervised.py --fake-finetune")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step-seconds", type=float, default=1.0)
    a = ap.parse_args(argv)
    rng = random.Random(1)
    print("FAKE fine-tune from ui/dev_stack_supervised.py: nothing is trained and nothing is written", flush=True)
    timed = 0.0
    for step in range(1, a.steps + 1):
        t0 = time.perf_counter()
        time.sleep(a.step_seconds)
        dt = time.perf_counter() - t0
        if step > 1:
            timed += dt
        loss = 0.004 + 0.02 * math.exp(-3.0 * step / a.steps) + rng.uniform(0, 0.0005)
        lr = 3e-5 * min(1.0, step / 5)
        # the print call of weather/finetune.py, word for word
        speed = "%.2f steps/s" % ((step - 1) / timed) if timed > 0 else "%.1f s for this step" % dt
        print("step %4d/%d  loss %.6f  lr %.2e  %s" % (step, a.steps, loss, lr, speed), flush=True)
    print("fake fine-tune finished: no weights were written", flush=True)
    return 0


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-finetune":
        raise SystemExit(fake_finetune(sys.argv[2:]))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ui-port", type=int, default=19080)
    ap.add_argument("--router-port", type=int, default=19000)
    ap.add_argument("--box-a-port", type=int, default=19001)
    ap.add_argument("--box-b-port", type=int, default=19002)
    ap.add_argument("--stack-a-port", type=int, default=19109)
    ap.add_argument("--stack-b-port", type=int, default=19110)
    ap.add_argument("--policy", default="adaptive")
    ap.add_argument("--preferred", default="box-b")
    ap.add_argument("--job-seconds", type=int, default=40,
                    help="how long the fake fine-tune job runs: one step a second (default 40)")
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-model", default="nemotron")
    a = ap.parse_args()

    py, host = sys.executable, "127.0.0.1"
    boxes = {"box-a": (a.box_a_port, a.stack_a_port), "box-b": (a.box_b_port, a.stack_b_port)}
    ports = [a.ui_port, a.router_port, a.box_a_port, a.box_b_port, a.stack_a_port, a.stack_b_port]
    if len(set(ports)) != len(ports):
        ap.error("the six ports must all be different")
    busy = [p for p in ports if not port_is_free(p)]
    if busy:
        raise SystemExit("in use already: port %s. Nothing was started; choose other ports (see --help)."
                         % ", ".join(str(p) for p in busy))
    box_url = {n: f"http://{host}:{bp}" for n, (bp, _) in boxes.items()}
    stack_url = {n: f"http://{host}:{sp}" for n, (_, sp) in boxes.items()}
    router_url = f"http://{host}:{a.router_port}"

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signum == signal.SIGHUP and signal.getsignal(signum) == signal.SIG_IGN:
            continue                                    # started under nohup: a hang-up stays ignored
        signal.signal(signum, lambda *_: stop.set())

    procs: list[tuple[str, subprocess.Popen]] = []      # only what this script started itself

    def start(label: str, cmd: list[str], cwd: Path) -> None:
        procs.append((label, subprocess.Popen(cmd, cwd=cwd)))

    def wait(url: str, timeout: float = 30) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if stop.is_set():
                raise SystemExit("stopped while starting")
            for label, p in procs:
                if p.poll() is not None:
                    raise SystemExit(f"{label} ended with code {p.returncode} while starting")
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        raise SystemExit(f"{url} did not come up")

    grandchildren: dict[int, str] = {}                  # pid -> what it is: the supervisors' own children
    try:
        for name, (box_port, stack_port) in boxes.items():
            forecast = [py, "serve/forecast_server.py", "--mock", "--admin", "--node", name,
                        "--host", host, "--port", str(box_port)]
            job = [py, str(Path(__file__).resolve()), "--fake-finetune", "--steps", str(a.job_seconds)]
            # The supervisor splits each command as a shell would (it never runs one), so quote the same way.
            start(f"supervisor {name}", [py, "serve/node_stack.py", "--name", name, "--admin",
                                         "--host", host, "--port", str(stack_port),
                                         "--serve", "forecast=" + shlex.join(forecast),
                                         "--job", "finetune=" + shlex.join(job)], ROOT)
        for name in boxes:
            wait(f"{stack_url[name]}/health")
            wait(f"{box_url[name]}/health")
        start("router", [py, "router.py", "--port", str(a.router_port), "--policy", a.policy,
                         *(["--preferred", a.preferred] if a.preferred else []),
                         *[x for n in boxes for x in ("--backend", f"{n}={box_url[n]}")]], ROOT)
        wait(f"{router_url}/health")
        ui = [py, "server.py", "--port", str(a.ui_port), "--router", router_url,
              *[x for n in boxes for x in ("--box", f"{n}={box_url[n]}")],
              *[x for n in boxes for x in ("--stack", f"{n}={stack_url[n]}")]]
        if a.llm_base_url:
            ui += ["--llm-base-url", a.llm_base_url, "--llm-model", a.llm_model]
        start("ui", ui, HERE)
        wait(f"http://{host}:{a.ui_port}/")
        print(f"\n  Control Room (ui): http://{host}:{a.ui_port}\n"
              f"  Router:             {router_url}\n"
              f"  Forecast servers:   {', '.join(f'{n} {u}' for n, u in box_url.items())}   (ours, --mock)\n"
              f"  Supervisors:        {', '.join(f'{n} {u}' for n, u in stack_url.items())}\n"
              f"  The finetune job is a fake: about {a.job_seconds} s of progress lines, nothing is trained.\n"
              f"  Ctrl-C to stop\n", flush=True)
        while not stop.wait(0.5):
            for label, p in procs:
                if p.poll() is not None:
                    print(f"{label} ended with code {p.returncode}; stopping the rest", flush=True)
                    stop.set()
    finally:
        # Note the supervisors' children first, to check afterwards that they are gone too.
        for name, url in stack_url.items():
            status = get_json(f"{url}/status") or {}
            for kind in ("servers", "jobs"):
                for child, view in (status.get(kind) or {}).items():
                    if isinstance(view, dict) and isinstance(view.get("pid"), int):
                        grandchildren[view["pid"]] = f"{child} of {name}"
        for _, p in reversed(procs):                    # page, router, then the supervisors
            if p.poll() is None:
                p.terminate()                           # a supervisor stops its own children on SIGTERM
        end = time.time() + STOP_WAIT_S
        for label, p in procs:
            try:
                p.wait(timeout=max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                print(f"{label} (pid {p.pid}) did not stop in {STOP_WAIT_S:.0f} s; killing it", flush=True)
                p.kill()
                p.wait()
        left = {pid: what for pid, what in grandchildren.items() if alive(pid)}
        for pid, what in left.items():                  # only after a supervisor had to be killed
            print(f"still running: {what} (pid {pid}). Stop it with: kill {pid}", flush=True)
        print("stopped" if not left else "stopped, but see the lines above", flush=True)


if __name__ == "__main__":
    main()
