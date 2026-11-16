#!/bin/sh
# Put the Control Room and its /api/v1 on Fuzzball, with an https address, next to the running GPU stack.
#
#   router_switching/deploy_api.sh            start it (or show the one already running), print the address and test it
#   router_switching/deploy_api.sh --status   only show the running one and its address
#   OWNER=chandan router_switching/deploy_api.sh   put a name on it: workflow chandan-wxrouter-api, endpoints chandan-…
#   PUBLIC=1 router_switching/deploy_api.sh        a link anyone on the network can open with no Fuzzball login. The page is
#                                                  then READ-ONLY (router_switching/serve_readonly.py): it shows everything
#                                                  and answers questions, but cannot change routing, fail, drain, fine-tune
#                                                  or kill anything, and the load test is capped at 10 req/s
#
# The page is REAL ONLY (router_switching/serve_realonly.py): if a box is a mock server or runs on CPU,
# it shows nothing and says why, instead of showing those numbers.
# What it starts: ONE small CPU-only service (2 cores, 2 GB, no GPU) named "api", in its own workflow
# "wxrouter-api". It runs ui/server.py from the branch below and talks to the router, the two GPU boxes
# and Nemotron of the GPU stack that is already running, by their cluster names. It takes no GPU, changes
# nothing in the GPU stack, and ends by itself after HOLD_HOURS (default 4).
# If the GPU stack is replaced, this service shows its boxes as OFFLINE: stop it and run this script again.
set -eu
cd "$(dirname "$0")/.."
FB=${FUZZBALL:-$HOME/bin/fuzzball}; PY=${PYTHON:-.venv/bin/python}
OWNER=${OWNER:-}; P=${OWNER:+$OWNER-}; NAME=${P}wxrouter-api; BRANCH=${BRANCH:-feature/dashboard-api}; REPO=${REPO:-7-c0ke/wxrouter}
HOLD=$(( ${HOLD_HOURS:-4} * 3600 )); PORT=18180; KEYFILE=$HOME/.wxrouter_api_key
SCOPE=group; SERVE=../router_switching/serve_realonly.py; [ -n "${PUBLIC:-}" ] && { SCOPE=public; SERVE=../router_switching/serve_readonly.py; }

running() { "$FB" workflow list --scope group --order-by -created 2>/dev/null | grep Started | grep " $1 *$" | head -1 | cut -d' ' -f1; }
show() {   # $1 = our workflow id
  "$FB" workflow endpoints list 2>/dev/null | grep "$1" | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$6); gsub(/ /,"",$7); gsub(/ /,"",$8); gsub(/ /,"",$1); print "    "$4" ("$6", scope "$7"): "$8"   id "$1}'
}

API=$(running $NAME)
if [ -z "$API" ] && [ "${1:-}" = "--status" ]; then echo "No $NAME workflow is running."; exit 1; fi
if [ -z "$API" ]; then
  STACK=""
  for id in $("$FB" workflow list --scope group --order-by -created 2>/dev/null | grep Started | cut -d' ' -f1); do
    if "$FB" workflow describe "$id" 2>/dev/null | grep -q -E "Service +\| Started +\| box-a"; then STACK=$id; break; fi
  done
  [ -n "$STACK" ] || { echo "No running GPU stack (a workflow with a started box-a service) was found. Nothing started."; exit 1; }
  GROUP=$("$FB" group list 2>/dev/null | grep '^\*' | awk -F'|' '{gsub(/ /,"",$2); print $2}')
  [ -n "$GROUP" ] || { echo "Could not read the selected Fuzzball group."; exit 1; }
  SUF="$STACK.$GROUP.fuzzball"
  # the supervisors (Fine-tune, Kill server) only for the team-only link; the public one is read-only
  STACKS="--stack box-a=http://box-a.$SUF:8109 --stack box-b=http://box-b.$SUF:8110"; [ -n "${PUBLIC:-}" ] && STACKS=""
  [ -s "$KEYFILE" ] || { umask 077; "$PY" -c "import secrets; print(secrets.token_urlsafe(18))" > "$KEYFILE"; }
  echo "GPU stack: $STACK   branch: $BRANCH   ends by itself after $((HOLD / 3600)) h"
  cat > /tmp/wxrouter-api.fz <<FZ
version: v4
jobs:
  hold:
    image:
      uri: docker://python:3.12-slim
    command: ["sleep", "$HOLD"]
    depends-on:
      - name: api
        status: RUNNING
    resource:
      cpu:
        cores: 1
      memory:
        size: 256MB
services:
  api:
    image:
      uri: docker://python:3.12-slim
    script: |
      #!/bin/sh
      set -e
      # the cluster's internet access drops now and then (seen Nov 14, 17:40): retry for about two minutes
      for i in 1 2 3 4 5 6 7 8; do pip install --quiet --no-cache-dir --disable-pip-version-check fastapi uvicorn httpx pydantic && break; echo "[api] pip failed (try \$i), retrying"; sleep 15; done
      for i in 1 2 3 4 5 6 7 8; do python -c "import io, tarfile, urllib.request; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz', timeout=120).read())).extractall('/tmp/src')" && break; echo "[api] code download failed (try \$i), retrying"; sleep 15; done
      cd /tmp/src/*/ui
      echo "[api] owner: ${OWNER:-none}   code: $REPO $BRANCH   GPU stack: $STACK"
      export WX_API_KEY='$(cat "$KEYFILE")'
      exec python $SERVE --host 0.0.0.0 --port $PORT --router http://router.$SUF:8000 \\
        --box box-a=http://box-a.$SUF:8001 --box box-b=http://box-b.$SUF:8002 $STACKS \\
        --llm-base-url http://box-a.$SUF:8090/v1 --llm-model nemotron --llm-thinking off
    resource:
      cpu:
        cores: 2
      memory:
        size: 2GB
    network:
      ports:
        - name: http
          port: $PORT
          protocol: tcp
      endpoints:
        # subdomain only: a "path" endpoint hands the app every request with /endpoints/accounts/... in front,
        # and the Control Room answers all of them with 404 {"detail":"Not Found"} (measured on fb-hack, Nov 14)
        - name: ${P}control-room
          port-name: http
          protocol: http
          type: subdomain
          scope: $SCOPE
FZ
  "$FB" workflow validate /tmp/wxrouter-api.fz >/dev/null || { echo "The workflow file did not validate: /tmp/wxrouter-api.fz"; "$FB" workflow validate /tmp/wxrouter-api.fz; exit 1; }
  API=$("$FB" workflow start -n $NAME /tmp/wxrouter-api.fz | tail -1 | grep -o -E '[0-9a-f]{8}-[0-9a-f-]{27}' | head -1)
  [ -n "$API" ] || { echo "The workflow did not start."; exit 1; }
  echo "started workflow $API; waiting for the service (image, pip install, code download)…"
  n=0
  until "$FB" workflow describe "$API" 2>/dev/null | grep -q -E "Service +\| Started +\| api" || [ $n -ge 60 ]; do
    "$FB" workflow describe "$API" 2>/dev/null | grep -q -E "^Status: +STAGE_STATUS_(FAILED|CANCELED)" && { echo "It failed:"; "$FB" workflow events "$API" | grep -i -E "fail|error" | tail -3; exit 1; }
    sleep 3; n=$((n+1))
  done
fi
echo "workflow $NAME: $API"
"$FB" workflow describe "$API" 2>/dev/null | grep -E "^Service|^Job" | cut -c1-70
echo "addresses:"
show "$API"
