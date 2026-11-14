#!/bin/sh
# Live demo on this machine: the Control Room connected to two real GPU inference servers on fb-hack.
#
#   router_switching/live_demo.sh            attach to the running two-GPU workflow (start one if none), open the page
#   router_switching/live_demo.sh --keep     the same, then keep it alive: when the workflow ends, start the next one
#   router_switching/live_demo.sh --stop     stop the port-forwards and the local page (the workflow is left alone)
#
# It uses the team14 group, so the workflow shows in the Fuzzball web UI. It starts a workflow only
# when both team GPUs are free, never stops anyone's workflow, and each workflow ends by itself
# after HOLD_MINUTES (default 60).
set -u
cd "$(dirname "$0")/.."
FB=${FUZZBALL:-$HOME/bin/fuzzball}; PY=${PYTHON:-.venv/bin/python}
GROUP=65503386-04ad-46dc-91b1-f031f71104a7; NAME=wxrouter-inference-2gpu
UI_PORT=${UI_PORT:-18180}; HOLD=${HOLD_MINUTES:-60}
stop_local() { pkill -f "workflow port-forward" 2>/dev/null; pkill -f "ui/server.py" 2>/dev/null; true; }
running_id() { "$FB" workflow list --order-by -created 2>/dev/null | grep "$NAME" | grep Started | head -1 | cut -d' ' -f1; }
attach() {
  "$FB" group use $GROUP >/dev/null 2>&1
  ID=$(running_id); NEW=""
  if [ -z "$ID" ]; then
    NEW=1
    FREE=$("$FB" node list -a | grep -E "SPARK27|SPARK28" | grep -c "gpu:1")
    [ "$FREE" = 2 ] || { echo "No $NAME workflow is running and the team's two GPUs are not both free ($FREE free). Not starting one."; return 1; }
    "$PY" deploy/build_inference_fuzzfile.py --mode serve --no-bench --hold-minutes "$HOLD" --service-timeout "$((HOLD + 10))m" --out /tmp/wx-live-serve.fz >/dev/null
    "$FB" workflow validate /tmp/wx-live-serve.fz >/dev/null || return 1
    ID=$("$FB" workflow start -n $NAME /tmp/wx-live-serve.fz | tail -1)
    echo "started workflow $ID"
  fi
  pkill -f "workflow port-forward" 2>/dev/null; sleep 1
  for spec in "router 28000:8000" "box-a 28001:8001" "box-b 28002:8002"; do
    set -- $spec; ("$FB" workflow port-forward "$ID" "$1" "$2" >"/tmp/wx-pf-$1.log" 2>&1 &)
  done
  n=0; until curl -s -m 2 -o /dev/null localhost:28000/health || [ $n -ge 90 ]; do sleep 2; n=$((n+1)); done
  # Only a workflow this run just started gets a policy set. An existing one is left exactly as it is:
  # someone may be in the middle of a demo with a box failed on purpose.
  [ -z "$NEW" ] || curl -s -m 3 -XPOST localhost:28000/router/config -H 'content-type: application/json' -d '{"policy":"balanced","preferred":""}' >/dev/null
  pgrep -f "ui/server.py" >/dev/null || ("$PY" ui/server.py --router http://127.0.0.1:28000 --box box-a=http://127.0.0.1:28001 --box box-b=http://127.0.0.1:28002 --port "$UI_PORT" >/tmp/wx-ui.log 2>&1 &)
  sleep 3
  echo "workflow $ID · dashboard http://127.0.0.1:$UI_PORT · $(date +%H:%M:%S)"
}
case ${1:-} in
  --stop) stop_local; echo "local page and port-forwards stopped; the workflow was not touched"; exit 0 ;;
esac
attach || exit 1
[ "${1:-}" = "--keep" ] || { command -v open >/dev/null && open "http://127.0.0.1:$UI_PORT/"; exit 0; }
while true; do
  sleep 20
  [ -n "$(running_id)" ] && [ "$(running_id)" = "$ID" ] && continue
  echo "workflow $ID ended at $(date +%H:%M:%S); starting the next one"
  sleep 10
  attach || { sleep 60; continue; }
done
