#!/bin/sh
# Run one inference workflow on Fuzzball from start to saved results.
#
#   deploy/run_inference.sh smoke|bench|demo|afno
#
# smoke, bench, afno use one GPU; demo uses both of the team's GPUs and needs them free.
# It regenerates the Fuzzfile from the current sources, validates it, starts it, waits for
# it to end, saves the JSON under results/, and rebuilds results/inference/report.html.
set -eu
MODE=${1:?usage: deploy/run_inference.sh smoke|bench|demo|afno}
FB=${FUZZBALL:-$HOME/bin/fuzzball}
PY=${PYTHON:-.venv/bin/python}
cd "$(dirname "$0")/.."
case $MODE in
  smoke) JOB=gpu-smoke; ARGS="--results-volume wxdesk --results-dir /wx" ;;
  bench) JOB=gpu-bench; ARGS="--results-volume wxdesk --results-dir /wx" ;;
  demo)  JOB=demo;      ARGS="--results-volume wxdesk --results-dir /wx" ;;
  afno)  JOB=afno-check; ARGS="" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
if [ "$MODE" = demo ]; then
  FREE=$("$FB" node list -a | grep -E "SPARK27|SPARK28" | grep -c "gpu:1" || true)
  [ "$FREE" = 2 ] || { echo "The team's two GPUs (SPARK27, SPARK28) are not both free ($FREE free). Not starting."; exit 1; }
fi
"$PY" deploy/build_inference_fuzzfile.py --mode "$MODE" $ARGS
"$FB" workflow validate "deploy/inference-$MODE.fz"
ID=$("$FB" workflow start -n "inference-$MODE-team14" "deploy/inference-$MODE.fz" | tail -1)
echo "workflow $ID started; waiting"
while ! "$FB" workflow list --order-by -created | grep "$ID" | grep -q -E "Finished|Errored|Canceled"; do sleep 15; done
"$FB" workflow describe "$ID" | sed -n '4p;11p'
"$FB" workflow events "$ID" | grep "Resources allocated" | grep -E "JOB|SERVICE" | cut -c1-140
LOG=$(mktemp)
"$FB" workflow log "$ID" "$JOB" > "$LOG" 2>&1 || true
SAVED=$("$PY" deploy/collect_bench.py "$LOG" --label "fuzzball-$MODE") || { tail -30 "$LOG"; exit 1; }
echo "saved $SAVED"
latest() { ls -d "results/$1"/*fuzzball-"$2"* 2>/dev/null | tail -1; }
set -- --bench "$(latest inference bench)/bench.json"
[ -n "$(latest cluster demo)" ] && set -- "$@" --cluster "$(latest cluster demo)/cluster_bench.json"
[ -n "$(latest afno afno)" ] && set -- "$@" --afno "$(latest afno afno)/afno_check.json"
"$PY" -m inference.report "$@" --out results/inference/report.html
