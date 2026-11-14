"""Start the whole demo locally: two mock boxes, the router and the Control Room UI.

  python dev_stack.py                     # UI on http://127.0.0.1:8080
  python dev_stack.py --llm-base-url http://<box>:8080/v1 --llm-model nemotron

Ctrl-C stops everything.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wait(url: str, timeout: float = 20) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise SystemExit(f"{url} did not come up")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ui-port", type=int, default=8080)
    ap.add_argument("--router-port", type=int, default=8000)
    ap.add_argument("--policy", default="adaptive")
    ap.add_argument("--preferred", default="box-b")
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-model", default="nemotron")
    ap.add_argument("--no-precip", action="store_true", help="mocks return precip_mm = null")
    a = ap.parse_args()
    py, rp = sys.executable, a.router_port
    a_url, b_url = f"http://127.0.0.1:{rp + 1}", f"http://127.0.0.1:{rp + 2}"
    extra = ["--no-precip"] if a.no_precip else []
    procs = [
        subprocess.Popen([py, "mock_server.py", "--name", "box-a", "--port", str(rp + 1), "--seed", "1", *extra], cwd=HERE),
        subprocess.Popen([py, "mock_server.py", "--name", "box-b", "--port", str(rp + 2), "--seed", "2", *extra], cwd=HERE),
    ]
    try:
        wait(f"{a_url}/health"); wait(f"{b_url}/health")
        procs.append(subprocess.Popen([py, "router.py", "--port", str(rp), "--policy", a.policy,
                                       *(["--preferred", a.preferred] if a.preferred else []),
                                       "--backend", f"box-a={a_url}", "--backend", f"box-b={b_url}"], cwd=HERE))
        wait(f"http://127.0.0.1:{rp}/health")
        ui = [py, "server.py", "--dev", "--port", str(a.ui_port), "--router", f"http://127.0.0.1:{rp}",
              "--box", f"box-a={a_url}", "--box", f"box-b={b_url}"]
        if a.llm_base_url:
            ui += ["--llm-base-url", a.llm_base_url, "--llm-model", a.llm_model]
        procs.append(subprocess.Popen(ui, cwd=HERE / "ui"))
        wait(f"http://127.0.0.1:{a.ui_port}/")
        print(f"\n  Control Room: http://127.0.0.1:{a.ui_port}\n  Router:       http://127.0.0.1:{rp}\n  Ctrl-C to stop\n", flush=True)
        signal.sigwait({signal.SIGINT, signal.SIGTERM})
    finally:
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
