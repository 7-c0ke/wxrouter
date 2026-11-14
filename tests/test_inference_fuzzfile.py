"""Structural checks on the generated Fuzzfiles. Not a substitute for `fuzzball workflow validate`."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent


def generate(tmp_path, *args) -> dict:
    out = tmp_path / "wf.fz"
    subprocess.run([sys.executable, "deploy/build_inference_fuzzfile.py", "--out", str(out), *args],
                   cwd=ROOT, check=True, capture_output=True)
    return yaml.safe_load(out.read_text())


def check_stage(wf: dict, stage: dict) -> None:
    assert stage["image"]["uri"].startswith("docker://")
    assert ("script" in stage) != ("command" in stage)
    assert stage["resource"]["cpu"]["cores"] > 0 and stage["resource"]["memory"]["size"]
    for path, ref in stage["files"].items():
        assert path.startswith("/app/") and ref.startswith("file://")
        assert ref.removeprefix("file://") in wf["files"], f"{ref} not defined in top-level files"


@pytest.mark.parametrize("mode,job", [("smoke", "gpu-smoke"), ("bench", "gpu-bench")])
def test_job_fuzzfile(tmp_path, mode, job):
    wf = generate(tmp_path, "--mode", mode)
    assert wf["version"] == "v4" and list(wf["jobs"]) == [job]
    stage = wf["jobs"][job]
    check_stage(wf, stage)
    assert stage["resource"]["devices"] == {"nvidia.com/gpu": 1}
    assert "--require-gpu" in stage["script"] and "--emit-markers" in stage["script"]
    assert stage["policy"]["timeout"]["execute"]
    assert "/app/inference/engine.py" in stage["files"]
    # the embedded source is byte-identical to the repo
    assert wf["files"]["inference-engine-py"].rstrip("\n") == (ROOT / "inference/engine.py").read_text().rstrip("\n")


def test_results_volume_is_optional(tmp_path):
    assert "volumes" not in generate(tmp_path)
    wf = generate(tmp_path, "--results-volume", "team14-results", "--results-dir", "/results")
    assert wf["volumes"] == {"results": {"name": "team14-results"}}
    assert wf["jobs"]["gpu-smoke"]["mounts"] == {"/results": "results"}


def test_serve_fuzzfile(tmp_path):
    wf = generate(tmp_path, "--mode", "serve", "--exclusive")
    assert set(wf["services"]) == {"box-a", "box-b", "router"}
    for name, port in (("box-a", 8001), ("box-b", 8002)):
        box = wf["services"][name]
        check_stage(wf, box)
        assert box["resource"]["devices"] == {"nvidia.com/gpu": 1} and box["resource"]["exclusive"] is True
        assert box["network"]["ports"][0]["port"] == port
        assert f"inference.server --name {name}" in box["script"]
    router = wf["services"]["router"]
    check_stage(wf, router)
    assert "devices" not in router["resource"]
    assert {d["name"] for d in router["depends-on"]} == {"box-a", "box-b"}


def test_collect_bench_roundtrip(tmp_path):
    from inference.bench import BEGIN, END

    res = {"schema": "wxrouter.inference.bench/v1", "created_at": "2026-11-14T10:00:00Z",
           "environment": {"device_name": "NVIDIA GB10", "backend": "torch", "precision": "fp16-autocast",
                           "model_version": "m", "hidden": 256, "layers": 4, "torch_import_ms": 1},
           "params": {"hours_ahead": 48},
           "cold_start": {"load_ms": 1, "first_inference_ms": 2, "warm_inference_ms": 3},
           "batch_sweep": [], "serving": [], "precision": []}
    log = tmp_path / "job.log"
    log.write_text(f"noise\n{BEGIN}\n{json.dumps(res)}\n{END}\nmore noise\n")
    subprocess.run([sys.executable, "deploy/collect_bench.py", str(log), "--label", "t", "--out", str(tmp_path)],
                   cwd=ROOT, check=True, capture_output=True)
    saved = tmp_path / "inference" / "20261114-100000-t" / "bench.json"
    assert json.loads(saved.read_text()) == res
