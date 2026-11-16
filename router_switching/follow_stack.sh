#!/bin/sh
# Keep the Fuzzball dashboard (and this Mac's) connected to whichever GPU stack the team is running.
#
#   router_switching/follow_stack.sh          check every 30 s until the dashboard workflow ends
#   router_switching/follow_stack.sh --once   check once
#
# The team redeploys its GPU stack (real-stack) often, and each new stack has a new workflow id. This finds
# the newest workflow whose box-a service is running and tells the dashboard workflow (OWNER-wxrouter-api)
# to talk to it, through POST /api/v1/admin/target. The dashboard's link does not change. It also moves this
# Mac's port-forwards (router 28000, boxes 28001/28002, Nemotron 28090) to that stack.
# It starts and stops no Fuzzball workflow.
set -u
cd "$(dirname "$0")/.."
FB=${FUZZBALL:-$HOME/bin/fuzzball}; PY=${PYTHON:-.venv/bin/python}
OWNER=${OWNER:-chandan}; NAME=${OWNER:+$OWNER-}wxrouter-api; KEYFILE=$HOME/.wxrouter_api_key; API=http://127.0.0.1:18181
log() { echo "$(date +%H:%M:%S) $*"; }

newest_stack() {
  for id in $("$FB" workflow list --scope group --order-by -created 2>/dev/null | grep Started | grep -v wxrouter-api | cut -d' ' -f1); do
    "$FB" workflow describe "$id" 2>/dev/null | grep -q -E "Service +\| Started +\| box-a" && { echo "$id"; return; }
  done
}

once() {
  DASH=$("$FB" workflow list --scope group --order-by -created 2>/dev/null | grep Started | grep " $NAME *$" | head -1 | cut -d' ' -f1)
  [ -n "$DASH" ] || { log "no running $NAME workflow: nothing to follow"; return 2; }
  if ! curl -s -m 5 -o /dev/null "$API/api/v1/admin/target"; then
    pgrep -f "port-forward $DASH api" >/dev/null || nohup "$FB" workflow port-forward "$DASH" api 18181:18180 -k 15 >/tmp/wx-pf-api.log 2>&1 </dev/null &
    sleep 4
  fi
  STACK=$(newest_stack)
  [ -n "$STACK" ] || { log "no GPU stack is running right now; the dashboard shows the router as unreachable"; return 0; }
  NOW=$(curl -s -m 8 "$API/api/v1/admin/target" | "$PY" -c "import sys,json; print(json.load(sys.stdin).get('stack') or '')" 2>/dev/null)
  if [ "$NOW" != "$STACK" ]; then
    R=$(curl -s -m 10 -XPOST "$API/api/v1/admin/target" -H 'content-type: application/json' -H "x-api-key: $(cat "$KEYFILE")" -d "{\"stack\": \"$STACK\"}")
    log "dashboard moved from ${NOW:-?} to $STACK: $(echo "$R" | head -c 160)"
  fi
  if ! pgrep -f "port-forward $STACK router 28000" >/dev/null; then
    pgrep -f "fuzzball workflow port-forward" | while read -r p; do ps -o args= -p "$p" | grep -q " api 18181" || kill "$p" 2>/dev/null; done
    for spec in "router 28000:8000" "box-a 28001:8001" "box-b 28002:8002" "box-a 28090:8090"; do
      svc=${spec% *}; ports=${spec#* }
      nohup "$FB" workflow port-forward "$STACK" "$svc" "$ports" -k 15 >"/tmp/wx-pf-${ports%%:*}.log" 2>&1 </dev/null &
    done
    log "this Mac's port-forwards now go to $STACK"
  fi
  return 0
}

[ "${1:-}" = "--once" ] && { once; exit $?; }
log "following the GPU stack for $NAME (every 30 s)"
while once; do sleep 30; done
log "stopped"
