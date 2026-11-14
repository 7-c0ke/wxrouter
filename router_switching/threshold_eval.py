"""How the adaptive rule's thresholds change its behaviour: margin, window and cooldown.

Runs the team's own `loadgen.py bench` once per setting, with the same seeded trace and the same
faults each time (box A slowed 4x, then box B killed and restarted), and records latency, errors
and the number of switches. It uses the mock boxes, so the result is labelled `simulated`: it
shows how the rule behaves, not how fast a GPU is.

  python -m router_switching.threshold_eval            # about 5 minutes
  python -m router_switching.threshold_eval --quick    # 3 settings, about 1 minute
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = [  # (margin, window_s, cooldown_s); the router's defaults are 0.25, 1, 10
    (0.10, 1.0, 10.0), (0.25, 1.0, 10.0), (0.50, 1.0, 10.0),
    (0.25, 0.25, 10.0), (0.25, 3.0, 10.0),
    (0.25, 1.0, 2.0), (0.25, 1.0, 30.0),
    (0.10, 0.25, 2.0),  # the twitchiest corner
]
QUICK = [(0.10, 0.25, 2.0), (0.25, 1.0, 10.0), (0.50, 3.0, 30.0)]


def run_one(margin: float, window: float, cooldown: float, a) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "loadgen.py", "bench", "--policies", "adaptive", "--seed", str(a.seed),
               "--duration", str(a.duration), "--rps", str(a.rps), "--margin", str(margin), "--window", str(window),
               "--cooldown", str(cooldown), "--base-port", str(a.base_port), "--out", tmp]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, timeout=a.duration * 4 + 120)
        summary = json.loads(next(Path(tmp).glob("*/summary.json")).read_text())
    s = summary["results"][0]
    return {"label": f"m{margin:g} w{window:g} c{cooldown:g}", "margin": margin, "window_s": window,
            "cooldown_s": cooldown, "requests": s["requests"], "errors": s["errors"], "error_pct": s["error_pct"],
            "p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "p99_ms": s["p99_ms"], "share_a_pct": s["share_a_pct"],
            "share_b_pct": s["share_b_pct"], "failovers": s["failovers"], "switches": s["switches"],
            "switch_log": s.get("switch_log", []), "events": summary["meta"]["events"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--rps", type=float, default=20)
    ap.add_argument("--base-port", type=int, default=18300)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rows = []
    for margin, window, cooldown in (QUICK if a.quick else GRID):
        r = run_one(margin, window, cooldown, a)
        rows.append(r)
        print(f"  margin {margin:<4} window {window:<4} cooldown {cooldown:<4} -> switches {r['switches']}, "
              f"errors {r['errors']}, p50 {r['p50_ms']} ms, p95 {r['p95_ms']} ms, p99 {r['p99_ms']} ms", flush=True)
    now = datetime.now(UTC)
    out = ROOT / "router_switching" / "results" / f"thresholds-{now.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"schema": "wxrouter.switching.thresholds/v1", "kind": "simulated",
                               "device": "mock_server.py", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "params": {"seed": a.seed, "duration": a.duration, "rps": a.rps,
                                          "events": rows[0]["events"], "policy": "adaptive"},
                               "rows": rows}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
