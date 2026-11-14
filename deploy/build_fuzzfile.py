"""Generate a Fuzzball Fuzzfile (v4) that runs wxrouter on the cluster.

The repo is private, so the cluster can't download the code. Instead, this
script embeds the Python sources in the Fuzzfile's top-level `files:` section.
Each service mounts the files it needs into /app. No registry, image secret or
GitHub token is needed. The image is stock `python:3.12-slim` (multi-arch, so
it runs on the arm64 DGX Sparks), and dependencies are pip-installed at start.

Services:
  box-a, box-b   forecast servers (the mock today)
  router         wxrouter in front of both, started once both boxes are RUNNING
  ui             Control Room dashboard + forecast assistant, started once the router
                 is RUNNING; exposed as a Fuzzball endpoint (and reachable by port-forward)

Jobs:
  hold           sleeps for --hours. A workflow ends when its jobs end and stops its
                 services, so without this job the services stop right after starting.
                 `fuzzball workflow stop <id>` ends it early.

Usage:
  python deploy/build_fuzzfile.py                      # writes deploy/wxrouter.fz
  python deploy/build_fuzzfile.py --exclusive          # each box gets a whole node
  python deploy/build_fuzzfile.py --llm-base-url http://<llm>:8080/v1   # assistant uses an LLM
  python deploy/build_fuzzfile.py --no-ui              # boxes + router only
  fuzzball workflow validate deploy/wxrouter.fz
  fuzzball workflow start deploy/wxrouter.fz
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPS = "fastapi uvicorn httpx pydantic"


def block(text: str, indent: int) -> str:
    pad = " " * indent
    return "\n".join(pad + line if line.strip() else "" for line in text.rstrip("\n").splitlines())


def resource(cores: int, memory: str, exclusive: bool) -> str:
    out = f"""    resource:
      cpu:
        cores: {cores}
      memory:
        size: {memory}"""
    if exclusive:
        out += "\n      exclusive: true"
    return out


def box_service(name: str, port: int, a) -> str:
    script = f"""#!/bin/sh
set -e
echo "[{name}] host=$(hostname) ip=$(hostname -i 2>/dev/null || true)"
pip install --quiet --no-cache-dir --disable-pip-version-check {DEPS}
cd /app
exec python mock_server.py --name {name} --host 0.0.0.0 --port {port} --seed {1 if name == 'box-a' else 2}
"""
    return f"""  {name}:
    image:
      uri: {a.image}
    files:
      /app/contract.py: file://contract-py
      /app/mock_server.py: file://mock-server-py
    script: |
{block(script, 6)}
{resource(a.box_cores, a.box_memory, a.exclusive)}
    network:
      ports:
        - name: http
          port: {port}
          protocol: tcp"""


def router_service(a) -> str:
    preferred = f" --preferred {a.preferred}" if a.preferred else ""
    script = f"""#!/bin/sh
set -e
pip install --quiet --no-cache-dir --disable-pip-version-check {DEPS}
# Show how the boxes resolve from here; useful if routing fails.
for h in box-a box-b; do echo "[router] $h -> $(getent hosts $h || echo unresolved)"; done
cd /app
exec python router.py --host 0.0.0.0 --port 8000 --policy {a.policy}{preferred} \\
  --backend box-a=http://box-a:8001 --backend box-b=http://box-b:8002
"""
    return f"""  router:
    image:
      uri: {a.image}
    files:
      /app/router.py: file://router-py
    depends-on:
      - name: box-a
        status: RUNNING
      - name: box-b
        status: RUNNING
    script: |
{block(script, 6)}
{resource(1, "1GB", False)}
    network:
      ports:
        - name: http
          port: 8000
          protocol: tcp"""


def ui_service(a) -> str:
    llm = f" \\\n  --llm-base-url {a.llm_base_url} --llm-model {a.llm_model}" if a.llm_base_url else ""
    script = f"""#!/bin/sh
set -e
pip install --quiet --no-cache-dir --disable-pip-version-check {DEPS}
cd /app/ui
exec python server.py --host 0.0.0.0 --port 8080 --router http://router:8000 \\
  --box box-a=http://box-a:8001 --box box-b=http://box-b:8002{llm}
"""
    endpoint = """
      endpoints:
        - name: control-room
          port-name: http
          protocol: http""" if a.ui_endpoint else ""
    return f"""  ui:
    image:
      uri: {a.image}
    files:
      /app/ui/server.py: file://ui-server-py
      /app/ui/agent.py: file://ui-agent-py
      /app/ui/static/index.html: file://ui-index-html
      /app/ui/static/app.js: file://ui-app-js
      /app/ui/static/style.css: file://ui-style-css
    depends-on:
      - name: router
        status: RUNNING
    script: |
{block(script, 6)}
{resource(1, "1GB", False)}
    network:
      ports:
        - name: http
          port: 8080
          protocol: tcp{endpoint}"""


def hold_job(a) -> str:
    last = "ui" if not a.no_ui else "router"
    return f"""jobs:
  hold:
    image:
      uri: {a.image}
    command: ["sleep", "{int(a.hours * 3600)}"]
    depends-on:
      - name: {last}
        status: RUNNING
{resource(1, "256MB", False)}"""


def build(a) -> str:
    files = {
        "contract-py": ROOT / "contract.py",
        "mock-server-py": ROOT / "mock_server.py",
        "router-py": ROOT / "router.py",
    }
    if not a.no_ui:
        files.update({
            "ui-server-py": ROOT / "ui" / "server.py",
            "ui-agent-py": ROOT / "ui" / "agent.py",
            "ui-index-html": ROOT / "ui" / "static" / "index.html",
            "ui-app-js": ROOT / "ui" / "static" / "app.js",
            "ui-style-css": ROOT / "ui" / "static" / "style.css",
        })
    parts = ["# Generated by deploy/build_fuzzfile.py. Do not edit by hand; edit the sources and regenerate.",
             "version: v4", "files:"]
    for key, path in files.items():
        parts += [f"  {key}: |", block(path.read_text(), 4)]
    parts += ["services:", box_service("box-a", 8001, a), box_service("box-b", 8002, a), router_service(a)]
    if not a.no_ui:
        parts.append(ui_service(a))
    parts.append(hold_job(a))
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "deploy" / "wxrouter.fz"))
    ap.add_argument("--image", default="docker://python:3.12-slim")
    ap.add_argument("--policy", default="adaptive",
                    choices=["static", "round_robin", "least_loaded", "adaptive"])
    ap.add_argument("--preferred", default="box-b", help="adaptive home box; '' for none")
    ap.add_argument("--exclusive", action="store_true",
                    help="give each box a whole node (puts A and B on different Sparks)")
    ap.add_argument("--box-cores", type=int, default=2)
    ap.add_argument("--box-memory", default="2GB")
    ap.add_argument("--hours", type=float, default=4.0,
                    help="how long the services stay up (a 'hold' job keeps the workflow alive)")
    ap.add_argument("--no-ui", action="store_true", help="leave out the Control Room service")
    ap.add_argument("--no-ui-endpoint", dest="ui_endpoint", action="store_false",
                    help="don't publish a Fuzzball endpoint for the UI (use port-forward)")
    ap.add_argument("--llm-base-url", default=None,
                    help="OpenAI-compatible URL for the assistant, e.g. http://<llm-host>:8080/v1")
    ap.add_argument("--llm-model", default="nemotron")
    a = ap.parse_args()
    Path(a.out).write_text(build(a))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
