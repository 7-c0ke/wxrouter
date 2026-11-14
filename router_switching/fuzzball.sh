#!/bin/sh
# GPU validation on fb-hack, in the order that makes sense, then refresh the dashboard's results.
#
#   router_switching/fuzzball.sh status       are the cluster and the team's two GPUs usable right now?
#   router_switching/fuzzball.sh one-gpu      batching benchmark on one DGX Spark
#   router_switching/fuzzball.sh afno         the trained model on one GPU, team server vs engine
#   router_switching/fuzzball.sh two-gpu      two boxes on two Sparks: every policy, then slow / stop / restart
#   router_switching/fuzzball.sh all          one-gpu, afno, two-gpu
#
# Every workload is bounded (job time limits) and ends by itself; nothing is left running.
# It never stops anyone else's workflow. two-gpu refuses to start unless both team GPUs are free.
set -eu
cd "$(dirname "$0")/.."
FB=${FUZZBALL:-$HOME/bin/fuzzball}
PY=${PYTHON:-.venv/bin/python}
status() {
  "$FB" context list | tail -1
  echo "Team 14 nodes (free cores, free GB, free GPUs):"
  "$FB" node list -a | grep -E "SPARK27|SPARK28" | cut -c30-110
  LAST=$("$FB" workflow list --order-by -created | sed -n 2p | cut -d' ' -f1)
  if "$FB" workflow describe "$LAST" | grep -q "fuzzball-objectcache"; then
    echo "BLOCKED: the newest workflow failed on the image cache (fuzzball-objectcache DNS). Wait a few minutes; do not resubmit in a loop."
  else
    echo "Image cache: the newest workflow ($LAST) did not fail on it."
  fi
}
case ${1:-status} in
  status) status ;;
  one-gpu) deploy/run_inference.sh bench ;;
  afno) deploy/run_inference.sh afno ;;
  two-gpu) deploy/run_inference.sh demo ;;
  all) deploy/run_inference.sh bench && deploy/run_inference.sh afno && deploy/run_inference.sh demo ;;
  *) echo "usage: $0 status|one-gpu|afno|two-gpu|all"; exit 2 ;;
esac
[ "${1:-status}" = status ] || "$PY" -m router_switching.collect
