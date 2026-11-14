"""Save the benchmark JSON printed by a Fuzzball job into results/inference/.

The smoke and bench jobs print their result between marker lines, so the job
log is all that is needed to get the numbers off the cluster:

  fuzzball workflow log <workflow-id> gpu-smoke | python deploy/collect_bench.py --label fuzzball-smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.bench import BEGIN, END, markdown


def extract(log: str) -> dict:
    if BEGIN not in log or END not in log:
        raise SystemExit("no benchmark JSON in the log (did the job fail before the bench finished?)")
    return json.loads(log.split(BEGIN, 1)[1].split(END, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="log file; default stdin")
    ap.add_argument("--label", default="fuzzball")
    ap.add_argument("--out", default=str(ROOT / "results" / "inference"))
    a = ap.parse_args()
    res = extract(Path(a.log).read_text() if a.log else sys.stdin.read())
    stamp = res["created_at"].replace("-", "").replace(":", "").replace("T", "-").rstrip("Z")
    out = Path(a.out) / f"{stamp}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=2))
    (out / "table.md").write_text(markdown(res))
    print(f"wrote {out}/bench.json and table.md ({res['environment']['device_name']})")


if __name__ == "__main__":
    main()
