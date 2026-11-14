"""Gather every benchmark result into router_switching/results/summary.json for the dashboard.

  python -m router_switching.collect

It reads what the benchmarks already wrote (it runs nothing):

  results/inference/*/bench.json            one box: batching on/off, batch sizes   (inference/bench.py)
  results/cluster/*/cluster_bench.json      two boxes: policies, slow/stop/restart  (inference/cluster_bench.py)
  results/afno/*/afno_check.json            the trained model, team server vs engine (inference/afno_check.py)
  results/all/*/summary.json                the team's mock comparison              (loadgen.py bench)
  router_switching/results/thresholds-*.json  adaptive threshold sweep on mocks     (threshold_eval.py)

Every run is labelled `real GPU`, `CPU` or `simulated` from what the run itself recorded
(the device it ran on), never from its file name.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "router_switching" / "results" / "summary.json"
REAL, CPU, SIM = "real GPU", "CPU", "simulated"


def kind_of(device: dict | None) -> str:
    return REAL if (device or {}).get("type", (device or {}).get("device")) == "cuda" else CPU


def row(label: str, s: dict, **extra) -> dict:
    return {"label": label, "p50_ms": s.get("p50_ms"), "p95_ms": s.get("p95_ms"), "p99_ms": s.get("p99_ms"),
            "rps": s.get("ok_per_s"), "requests": s.get("requests"), "errors": s.get("errors"),
            "error_pct": s.get("error_pct"), "share_a_pct": s.get("share_a_pct"), "share_b_pct": s.get("share_b_pct"),
            "failovers": s.get("failovers"), **extra}


def from_bench(path: Path) -> list[dict]:
    b = json.loads(path.read_text())
    env = b["environment"]
    kind = REAL if env.get("device") == "cuda" else CPU
    rows = []
    for r in b["serving"]:
        lat = r["latency_ms"]
        rows.append({"label": f"{r['clients']} cl · {'no batch' if r['mode'] == 'baseline' else 'batched'}",
                     "clients": r["clients"], "batched": r["mode"] != "baseline", "p50_ms": lat["p50"],
                     "p95_ms": lat["p95"], "p99_ms": lat["p99"], "rps": r["requests_per_s"],
                     "error_pct": round(100 * r["error_rate"], 2), "mean_batch_rows": r["mean_batch_rows"]})
    base = {"kind": kind, "source": str(path.relative_to(ROOT)), "when": b["created_at"],
            "device": env["device_name"], "model": env["model_version"]}
    out = [{**base, "type": "batching", "title": f"Batching off vs on, one box ({env['device_name']})", "rows": rows,
            "batch_sweep": [{"rows": r["batch_rows"], "p50_ms": r["latency_ms"]["p50"], "rows_per_s": r["rows_per_s"],
                             "peak_gpu_mb": r["gpu_memory"].get("peak_allocated_mb")} for r in b["batch_sweep"]],
            "cold_start": b["cold_start"],
            "precision": [{"precision": r["precision"], "p50_ms": r["latency_ms"]["p50"], "rows_per_s": r["rows_per_s"],
                           "max_abs_diff_vs_fp32": r.get("max_abs_diff_vs_fp32"),
                           "numerically_valid": r.get("numerically_valid")} for r in b.get("precision", [])]}]
    return out


def from_cluster(path: Path) -> list[dict]:
    c = json.loads(path.read_text())
    devices = [(p.get("device") or {}) for p in c["placement"].values()]
    kind = REAL if devices and all(d.get("type") == "cuda" for d in devices) else CPU
    base = {"kind": kind, "source": str(path.relative_to(ROOT)), "when": c["created_at"],
            "device": ", ".join(sorted({d.get("name") or "?" for d in devices})), "params": c["params"],
            "placement": {n: {"gpu_uuid": (p.get("node") or {}).get("gpu_uuid"),
                              "address": ((p.get("node") or {}).get("addresses") or [None])[0]}
                          for n, p in c["placement"].items()},
            "distinct_nodes": c["distinct_nodes"], "calibration": c.get("calibration")}
    groups = {"single_gpu": ("single", "One GPU, same trace"), "two_gpu": ("policies", "Two GPUs, every routing policy"),
              "fault": ("faults", "Two GPUs with box A slowed, stopped and restarted")}
    out = []
    for prefix, (typ, title) in groups.items():
        cases = [x for x in c["cases"] if x["label"].startswith(prefix + "_")]
        if not cases:
            continue
        run = {**base, "type": typ, "title": title,
               "rows": [row(x["label"][len(prefix) + 1:], x["summary"], switches=len(x["switches"])) for x in cases]}
        if typ == "faults":
            run["cases"] = [{"policy": x["policy"], "events": x["events"], "switches": x["switches"],
                             "fault_report": x.get("fault_report"),
                             "timeline": [{"t": s["t"], "a": s["served_by"].get("box-a", 0),
                                           "b": s["served_by"].get("box-b", 0), "errors": s["errors"],
                                           "p95_ms": s["p95_ms"]} for s in x["timeline"]]} for x in cases]
        out.append(run)
    return out


def from_afno(path: Path) -> list[dict]:
    a = json.loads(path.read_text())
    rows = [{"label": f"{s['server']} · {r['clients']} cl", "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"],
             "p99_ms": r["p99_ms"], "rps": r["requests_per_s"],
             "error_pct": round(100 * r["errors"] / max(1, r["requests"]), 2)} for s in a["servers"] for r in s["load"]]
    return [{"kind": REAL, "type": "afno", "title": "Trained AFNO model: team server vs engine",
             "source": str(path.relative_to(ROOT)), "when": a["created_at"], "device": a["gpu"],
             "model": a["servers"][0]["model_version"], "compare": a["compare"], "rows": rows}]


def from_mock(path: Path) -> list[dict]:
    m = json.loads(path.read_text())
    meta = m.get("meta", {})
    rows = [{"label": s["policy"], "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "p99_ms": s["p99_ms"],
             "rps": None, "requests": s["requests"], "errors": s["errors"], "error_pct": s["error_pct"],
             "share_a_pct": s["share_a_pct"], "share_b_pct": s["share_b_pct"], "failovers": s["failovers"],
             "switches": s["switches"] if s["switches"] != "" else None} for s in m["results"]]
    return [{"kind": SIM, "type": "faults-mock", "title": "Mock boxes with faults (the team's loadgen bench)",
             "source": str(path.relative_to(ROOT)), "when": meta.get("started"), "device": "mock_server.py",
             "params": {"events": meta.get("events"), "rps": meta.get("rps"), "duration": meta.get("duration")},
             "rows": rows}]


def from_thresholds(path: Path) -> list[dict]:
    t = json.loads(path.read_text())
    return [{"kind": t["kind"], "type": "thresholds", "title": "Adaptive thresholds: margin, window, cooldown",
             "source": str(path.relative_to(ROOT)), "when": t["created_at"], "device": t["device"],
             "params": t["params"], "rows": t["rows"]}]


def newest(pattern: str, per_kind=None) -> list[Path]:
    """Newest file for the pattern; with per_kind, the newest for each distinct key."""
    files = sorted(ROOT.glob(pattern))
    if not per_kind:
        return files[-1:]
    best = {}
    for f in files:
        best[per_kind(f)] = f
    return list(best.values())


def build() -> dict:
    def bench_kind(p: Path) -> str:
        return json.loads(p.read_text())["environment"].get("device", "cpu")

    def cluster_kind(p: Path) -> str:
        c = json.loads(p.read_text())
        return str(sorted((pl.get("device") or {}).get("type") or "?" for pl in c["placement"].values()))

    runs: list[dict] = []
    for p in newest("results/inference/*/bench.json", bench_kind):
        if json.loads(p.read_text())["params"].get("quick"):
            continue
        runs += from_bench(p)
    for p in newest("results/cluster/*/cluster_bench.json", cluster_kind):
        runs += from_cluster(p)
    for p in newest("results/afno/*/afno_check.json"):
        runs += from_afno(p)
    for p in newest("results/all/*/summary.json"):
        runs += from_mock(p)
    for p in newest("router_switching/results/thresholds-*.json"):
        runs += from_thresholds(p)
    order = {REAL: 0, CPU: 1, SIM: 2}
    types = ["faults", "policies", "single", "batching", "afno", "faults-mock", "thresholds"]
    runs.sort(key=lambda r: (order[r["kind"]], types.index(r["type"])))
    have_gpu = {r["type"] for r in runs if r["kind"] == REAL}
    missing = [what for typ, what in (("batching", "one-GPU batching benchmark"),
                                      ("policies", "two-GPU routing policies"),
                                      ("faults", "two-GPU slowdown, failover and recovery"),
                                      ("afno", "trained AFNO model on a GPU")) if typ not in have_gpu]
    # CPU vs GPU only when both exist for the same benchmark.
    cpu_gpu = None
    b = {r["kind"]: r for r in runs if r["type"] == "batching"}
    if REAL in b and CPU in b:
        def top(run):
            r = max((x for x in run["rows"] if x["batched"]), key=lambda x: x["clients"])
            return {"clients": r["clients"], "rps": r["rps"], "p95_ms": r["p95_ms"], "device": run["device"]}
        cpu_gpu = {"gpu": top(b[REAL]), "cpu": top(b[CPU]),
                   "note": "Different machines: a DGX Spark GPU and a laptop CPU. Batched, same model and forecast length."}
    return {"schema": "wxrouter.switching.summary/v1",
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runs": runs, "cpu_vs_gpu": cpu_gpu, "not_measured_on_gpu": missing}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = build()
    OUT.write_text(json.dumps(s, indent=1))
    for r in s["runs"]:
        print(f"  [{r['kind']:>9}] {r['type']:<12} {r['title']}  <- {r['source']}")
    if s["not_measured_on_gpu"]:
        print("  not measured on a GPU yet:", "; ".join(s["not_measured_on_gpu"]))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
