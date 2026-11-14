"""Generate Fuzzball Fuzzfiles (v4) for the GPU inference subsystem.

Same approach as build_fuzzfile.py: the repo is private, so the Python sources
are embedded in the Fuzzfile's top-level `files:` section and mounted into /app.

  --mode smoke   one short GPU job: nvidia-smi, torch CUDA check, `bench --quick`   (default)
  --mode bench   one GPU job running the full benchmark
  --mode serve   box-a and box-b as real GPU inference servers, router in front
  --mode afno    one GPU job: the trained AFNO model behind the team's server and behind the
                 engine, answers compared and both timed (needs the team volume set up)
  --mode demo    the same three services plus a job that benchmarks every routing policy and
                 the slow / stop / restart scenario, then ends (which stops the services)

Every job requests `nvidia.com/gpu` and runs with --require-gpu, so a node with
CPUs but no usable GPU fails loudly instead of reporting CPU numbers.

Usage:
  python deploy/build_inference_fuzzfile.py                       # deploy/inference-smoke.fz
  python deploy/build_inference_fuzzfile.py --mode bench
  python deploy/build_inference_fuzzfile.py --mode serve --exclusive
  fuzzball workflow validate deploy/inference-smoke.fz
  fuzzball workflow start -n inference-smoke-team14 deploy/inference-smoke.fz
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_DEPS = "fastapi uvicorn httpx pydantic"
INFERENCE_FILES = sorted(p.name for p in (ROOT / "inference").glob("*.py"))


def key(path: str) -> str:
    return "-".join(part for part in re.split(r"[/_.]+", path) if part)


def block(text: str, indent: int) -> str:
    pad = " " * indent
    return "\n".join(pad + line if line.strip() else "" for line in text.rstrip("\n").splitlines())


def file_mounts(paths: list[str]) -> str:
    return "\n".join(f"      /app/{p}: file://{key(p)}" for p in paths)


def resource(a, gpu: bool, cores: int | None = None, memory: str | None = None) -> str:
    out = f"""    resource:
      cpu:
        cores: {cores or a.cores}
      memory:
        size: {memory or a.memory}"""
    if gpu:
        out += f"\n      devices:\n        {a.gpu_device}: {a.gpus}"
        if a.exclusive:
            out += "\n      exclusive: true"
    return out


GPU_CHECK = """echo "[{name}] host=$(hostname) arch=$(uname -m)"
nvidia-smi || echo "[{name}] nvidia-smi not available"
python -c "import torch; print('[{name}] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
"""


def bench_job(a, quick: bool) -> str:
    name = "gpu-smoke" if quick else "gpu-bench"
    out_dir = f"{a.results_dir}/wxrouter-inference" if a.results_volume else a.results_dir
    script = f"""#!/bin/sh
set -e
{GPU_CHECK.format(name=name)}cd /app
python -m inference.bench {"--quick " if quick else ""}--require-gpu --emit-markers --out {out_dir}
"""
    mounts = f"\n    mounts:\n      {a.results_dir}: results" if a.results_volume else ""
    return f"""  {name}:
    image:
      uri: {a.image}
    files:
{file_mounts(["inference/" + f for f in INFERENCE_FILES])}
    script: |
{block(script, 6)}{mounts}
{resource(a, gpu=True)}
    policy:
      timeout:
        execute: {"15m" if quick else "45m"}"""


def box_service(name: str, port: int, a) -> str:
    extra = " --admin --supervise" if a.mode == "demo" else ""
    script = f"""#!/bin/sh
set -e
{GPU_CHECK.format(name=name)}pip install --quiet --no-cache-dir --disable-pip-version-check {SERVER_DEPS}
cd /app
exec python -m inference.server --name {name} --host 0.0.0.0 --port {port} --require-gpu{extra}
"""
    return f"""  {name}:
    image:
      uri: {a.image}
    files:
{file_mounts(["contract.py"] + ["inference/" + f for f in INFERENCE_FILES])}
    script: |
{block(script, 6)}
{resource(a, gpu=True)}
    network:
      ports:
        - name: http
          port: {port}
          protocol: tcp"""


def router_service(a) -> str:
    preferred = f" --preferred {a.preferred}" if a.preferred else ""
    script = f"""#!/bin/sh
set -e
pip install --quiet --no-cache-dir --disable-pip-version-check {SERVER_DEPS}
for h in box-a box-b; do echo "[router] $h -> $(getent hosts $h || echo unresolved)"; done
cd /app
exec python router.py --host 0.0.0.0 --port 8000 --policy {a.policy}{preferred} \\
  --backend box-a=http://box-a:8001 --backend box-b=http://box-b:8002
"""
    return f"""  router:
    image:
      uri: {a.router_image}
    files:
      /app/router.py: file://router-py
    depends-on:
      - name: box-a
        status: RUNNING
      - name: box-b
        status: RUNNING
    script: |
{block(script, 6)}
{resource(a, gpu=False, cores=2, memory="2GB")}
    network:
      ports:
        - name: http
          port: 8000
          protocol: tcp"""


def demo_job(a) -> str:
    out_dir = f"{a.results_dir}/wxrouter-inference/cluster" if a.results_volume else a.results_dir
    script = f"""#!/bin/sh
set -e
pip install --quiet --no-cache-dir --disable-pip-version-check httpx
cd /app
python cluster_bench.py --router http://router:8000 --box box-a=http://box-a:8001 --box box-b=http://box-b:8002 \\
  --require-gpu --require-distinct-nodes --emit-markers --out {out_dir} {a.demo_args}
"""
    mounts = f"\n    mounts:\n      {a.results_dir}: results" if a.results_volume else ""
    return f"""  demo:
    image:
      uri: {a.router_image}
    files:
      /app/loadgen.py: file://loadgen-py
      /app/cluster_bench.py: file://inference-cluster-bench-py
    depends-on:
      - name: router
        status: RUNNING
    script: |
{block(script, 6)}{mounts}
{resource(a, gpu=False, cores=2, memory="2GB")}
    policy:
      timeout:
        execute: {a.service_timeout}"""


def afno_job(a) -> str:
    """The trained model on one GPU: the team's server and the engine, same weights, compared and timed."""
    script = f"""#!/bin/sh
set -e
{GPU_CHECK.format(name="afno-check")}WEIGHTS=$(ls {a.wx_dir}/bundle/weights/*.pt | head -1)
echo "[afno-check] weights=$WEIGHTS repo=$(cd {a.wx_dir}/repo && git log --oneline -1 2>/dev/null || echo unknown)"
export PYTHONPATH={a.wx_dir}/pydeps
python -c "import fastapi, uvicorn, httpx, pydantic" 2>/dev/null \\
  || pip install --quiet --no-cache-dir --disable-pip-version-check {SERVER_DEPS}
cd /app
python -m inference.afno_check --weights "$WEIGHTS" --data {a.wx_dir}/bundle --repo {a.wx_dir}/repo \\
  --inference-root /app --emit-markers --out {a.wx_dir}/wxrouter-inference/afno
"""
    return f"""  afno-check:
    image:
      uri: {a.image}
    files:
{file_mounts(["contract.py"] + ["inference/" + f for f in INFERENCE_FILES])}
    script: |
{block(script, 6)}
    mounts:
      {a.wx_dir}: results
{resource(a, gpu=True, cores=8, memory="48GB")}
    policy:
      timeout:
        execute: 20m"""


def build(a) -> str:
    paths = ["inference/" + f for f in INFERENCE_FILES]
    if a.mode == "afno":
        paths = ["contract.py", *paths]
    if a.mode in ("serve", "demo"):
        paths = ["contract.py", "router.py", *paths]
    if a.mode == "demo":
        paths.append("loadgen.py")
    parts = ["# Generated by deploy/build_inference_fuzzfile.py. Do not edit by hand; edit the sources and regenerate.",
             "version: v4", "files:"]
    for p in paths:
        parts += [f"  {key(p)}: |", block((ROOT / p).read_text(), 4)]
    if a.mode in ("serve", "demo"):
        if a.mode == "demo" and a.results_volume:
            parts += ["volumes:", "  results:", f"    name: {a.results_volume}"]
        parts += ["services:", box_service("box-a", 8001, a), box_service("box-b", 8002, a), router_service(a)]
        if a.mode == "demo":
            parts += ["jobs:", demo_job(a)]
    elif a.mode == "afno":
        parts += ["volumes:", "  results:", f"    name: {a.results_volume or 'wxdesk'}", "jobs:", afno_job(a)]
    else:
        if a.results_volume:
            parts += ["volumes:", "  results:", f"    name: {a.results_volume}"]
        parts += ["jobs:", bench_job(a, quick=a.mode == "smoke")]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "bench", "serve", "demo", "afno"], default="smoke")
    ap.add_argument("--out", help="default: deploy/inference-<mode>.fz")
    ap.add_argument("--image", default="docker://nvcr.io/nvidia/pytorch:26.02-py3",
                    help="needs python + CUDA torch; the default is multi-arch (amd64 and arm64)")
    ap.add_argument("--router-image", default="docker://python:3.12-slim")
    ap.add_argument("--gpu-device", default="nvidia.com/gpu")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--memory", default="16GB")
    ap.add_argument("--exclusive", action="store_true", help="serve: give each box a whole node")
    ap.add_argument("--results-dir", default="/tmp/results", help="where the bench writes inside the job")
    ap.add_argument("--results-volume", default="",
                    help="existing persistent volume to mount at --results-dir; results go to its\n"
                         "wxrouter-inference/ folder (optional)")
    ap.add_argument("--wx-dir", default="/wx", help="afno: where the team volume is mounted (repo/, bundle/, pydeps/)")
    ap.add_argument("--service-timeout", default="30m", help="demo: hard limit for the demo job; the services stop when it ends")
    ap.add_argument("--demo-args", default="", help="demo: extra flags for inference/cluster_bench.py")
    ap.add_argument("--policy", default="adaptive", choices=["static", "round_robin", "least_loaded", "adaptive"])
    ap.add_argument("--preferred", default="box-b", help="serve: adaptive home box; '' for none")
    a = ap.parse_args()
    out = Path(a.out or ROOT / "deploy" / f"inference-{a.mode}.fz")
    out.write_text(build(a))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
