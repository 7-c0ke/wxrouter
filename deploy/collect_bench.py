"""Save the JSON a Fuzzball job printed into results/.

The inference jobs print their result between marker lines, so the job log is
all that is needed to get the numbers off the cluster:

  fuzzball workflow log <id> gpu-bench  | python deploy/collect_bench.py --label fuzzball-bench
  fuzzball workflow log <id> demo       | python deploy/collect_bench.py --label two-gpu
  fuzzball workflow log <id> afno-check | python deploy/collect_bench.py --label afno

The kind of result is recognised from its markers. It goes to
results/<inference|cluster|afno>/<timestamp>-<label>/ as JSON plus table.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import afno_check, bench, cluster_bench

KINDS = [(bench.BEGIN, bench.END, "inference", "bench.json", bench.markdown),
         (cluster_bench.BEGIN, cluster_bench.END, "cluster", "cluster_bench.json", cluster_bench.markdown),
         (afno_check.BEGIN, afno_check.END, "afno", "afno_check.json", afno_check.markdown)]


def extract(log: str):
    for begin, end, folder, name, markdown in KINDS:
        if begin in log and end in log:
            return json.loads(log.split(begin, 1)[1].split(end, 1)[0]), folder, name, markdown
    raise SystemExit("no result JSON in the log (did the job fail before it finished?)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="log file; default stdin")
    ap.add_argument("--label", default="fuzzball")
    ap.add_argument("--out", default=str(ROOT / "results"), help="results root; a folder per kind is made inside")
    a = ap.parse_args()
    res, folder, name, markdown = extract(Path(a.log).read_text() if a.log else sys.stdin.read())
    stamp = res["created_at"].replace("-", "").replace(":", "").replace("T", "-").rstrip("Z")
    out = Path(a.out) / folder / f"{stamp}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(res, indent=2))
    (out / "table.md").write_text(markdown(res))
    print(out / name)


if __name__ == "__main__":
    main()
