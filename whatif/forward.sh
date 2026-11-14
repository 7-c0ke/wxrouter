#!/usr/bin/env bash
# Port-forward what WhatIf needs from the real stack (fuzzball/30-real-stack.fz) to localhost.
# Same command form as deploy/fb.py `open`:  fuzzball workflow port-forward <wid> <service> <local>:<remote> -k 30
#
#   whatif/forward.sh <workflow-id>        (or with no argument: the "real" id remembered in .fb_deploy.json)
#
# Read-only for the cluster: it only opens tunnels. Ctrl-C closes them all.
set -euo pipefail
cd "$(dirname "$0")/.."
FB="${FUZZBALL:-$(command -v fuzzball || echo "$HOME/bin/fuzzball")}"
WID="${1:-$(python3 -c 'import json;print(json.load(open(".fb_deploy.json"))["real"]["id"])' 2>/dev/null || true)}"
[ -n "$WID" ] || { echo "usage: $0 <workflow-id>   (no 'real' workflow remembered in .fb_deploy.json)"; exit 2; }

# service local:remote
FORWARDS=(
  "box-a 8090:8090"   # Nemotron llama-server, box A
  "box-b 8091:8091"   # Nemotron llama-server, box B
  "router 8000:8000"  # router.py
  "box-a 8001:8001"   # forecast server A
  "box-b 8002:8002"   # forecast server B
  "box-a 8109:8109"   # supervisor A
  "box-b 8110:8110"   # supervisor B
)
pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null' EXIT INT TERM
for f in "${FORWARDS[@]}"; do
  set -- $f
  "$FB" workflow port-forward "$WID" "$1" "$2" -k 30 >"/tmp/whatif-forward-$1-${2%%:*}.log" 2>&1 &
  pids+=($!)
  echo "forwarding $1 $2  (pid $!)"
done
echo "All tunnels up for $WID. Test:  curl -s localhost:8090/v1/models ; python -m whatif.cli check"
wait
