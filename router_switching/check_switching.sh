#!/bin/sh
# One command: check inference switching on the two real GPUs, driven by Nemotron.
#
#   router_switching/check_switching.sh              Nemotron runs the check and explains the result
#   router_switching/check_switching.sh --direct     the same check without the LLM (prints the raw result)
#   router_switching/check_switching.sh "question"   ask Nemotron anything about the cluster instead
#
# Before the check it makes sure everything it needs is there, and repairs what is not:
# the running Fuzzball workflow, the port-forwards to it, the local API server, and Nemotron.
# It starts no Fuzzball workflow and stops none. The check makes one box answer 503 for a few
# seconds and always puts it back.
set -u
cd "$(dirname "$0")/.."
FB=${FUZZBALL:-$HOME/bin/fuzzball}; PY=${PYTHON:-.venv/bin/python}; PORT=${UI_PORT:-18180}
KEYFILE=$HOME/.wxrouter_api_key
say() { printf '%s\n' "$*"; }
ok() { curl -s -m "${2:-4}" -o /dev/null "$1"; }

say "1/5 Fuzzball workflow"
ID=$("$FB" workflow list --scope group --order-by -created 2>/dev/null | grep Started | head -1 | cut -d' ' -f1)
[ -n "$ID" ] || { say "    FAILED: no workflow is running on fb-hack (or the fuzzball CLI is not logged in / off the hackathon Wi-Fi)."; exit 1; }
say "    running: $ID"

say "2/5 Port-forwards to it"
if ! pgrep -f "port-forward $ID router" >/dev/null || ! ok localhost:28000/health; then
  say "    (re)connecting…"
  pgrep -f "fuzzball workflow port-forward" | xargs kill 2>/dev/null; sleep 1
  for spec in "router 28000:8000" "box-a 28001:8001" "box-b 28002:8002" "box-a 28090:8090"; do
    svc=${spec% *}; ports=${spec#* }
    nohup "$FB" workflow port-forward "$ID" "$svc" "$ports" -k 15 >"/tmp/wx-pf-${ports%%:*}.log" 2>&1 </dev/null &
  done
  n=0; until ok localhost:28000/health || [ $n -ge 30 ]; do sleep 2; n=$((n+1)); done
  NEWPF=1
fi
ok localhost:28000/health || { say "    FAILED: the router of $ID does not answer on localhost:28000."; exit 1; }
say "    router: $(curl -s -m 4 localhost:28000/health)"

say "3/5 Local API server (port $PORT)"
[ -s "$KEYFILE" ] || { umask 077; "$PY" -c "import secrets; print(secrets.token_urlsafe(18))" > "$KEYFILE"; NEWKEY=1; }
# restart it when the workflow or the key changed under it, so its node names and key are the current ones
if [ -n "${NEWPF:-}${NEWKEY:-}" ] || ! ok "localhost:$PORT/api/v1/tools" || ! curl -s -m 15 "localhost:$PORT/api/v1/fuzzball" | grep -q "$ID" \
   || ! curl -s -m 8 "localhost:$PORT/api/overview" | grep -q '"agent_mode":"llm'; then   # the page's chat must be on Nemotron
  say "    (re)starting…"
  pgrep -f "server.py --host 0.0.0.0 --router http://127.0.0.1:28000" | xargs kill 2>/dev/null
  pgrep -f "server.py --router http://127.0.0.1:28000" | xargs kill 2>/dev/null; sleep 1
  # exec, and every stream away from this script: a server that keeps the script's output open makes a caller wait forever
  (cd ui && WX_WORKFLOW=$ID WX_API_KEY=$(cat "$KEYFILE") exec nohup "../$PY" server.py --host 0.0.0.0 --router http://127.0.0.1:28000 \
      --box box-a=http://127.0.0.1:28001 --box box-b=http://127.0.0.1:28002 \
      --llm-base-url http://127.0.0.1:28090/v1 --port "$PORT" >/tmp/wx-ui.log 2>&1 </dev/null) >/dev/null 2>&1 &
  n=0; until ok "localhost:$PORT/api/v1/tools" || [ $n -ge 20 ]; do sleep 1; n=$((n+1)); done
fi
ok "localhost:$PORT/api/v1/tools" || { say "    FAILED: the API server did not start. Its log: /tmp/wx-ui.log"; tail -5 /tmp/wx-ui.log; exit 1; }
say "    API: http://127.0.0.1:$PORT/api/v1   dashboard: http://127.0.0.1:$PORT/"

say "4/5 The two GPUs"
curl -s -m 25 "localhost:$PORT/api/v1/gpus" | "$PY" -c "
import sys, json
bad = 0
for n, g in json.load(sys.stdin)['gpus'].items():
    print(f\"    {n}: {g['state']} on {g['node'] or 'node N/A'}, {g['kind']}, model {g['model']['version']}\")
    bad += g['state'] != 'ONLINE' or g['kind'] != 'real GPU'
sys.exit(1 if bad else 0)" || { say "    FAILED: both boxes must be ONLINE on a real GPU for this check."; exit 1; }

command -v open >/dev/null && [ -z "${NO_OPEN:-}" ] && open "http://127.0.0.1:$PORT/"

if [ "${1:-}" = "--direct" ]; then
  say "5/5 Switching check, without the LLM (about 10 s)"
  curl -s -m 120 -XPOST "localhost:$PORT/api/v1/tools/run_switching_check" -H 'content-type: application/json' \
       -H "x-api-key: $(cat "$KEYFILE")" -d '{"requests": 6}' | "$PY" -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('result')
if not r:
    print('    FAILED:', d); sys.exit(1)
for s in r['steps']:
    print(f\"    {s['step']:<28} served by {s['served_by']}  errors {s['errors']}  failovers {s['failovers']}\")
print('    router marked it down after', r['steps'][1].get('router_marked_down_after_s'), 's; healthy again after', r['steps'][-1].get('router_marked_healthy_after_s'), 's')
print('    box restored:', r['box_restored'])
print('RESULT:', 'PASS' if r['switching_worked'] and r['box_restored'] else 'FAIL', '-', r['verdict'])
sys.exit(0 if r['switching_worked'] and r['box_restored'] else 1)"
  exit $?
fi

say "5/5 Nemotron runs the check (about 20 to 40 s; progress below)"
ok localhost:28090/v1/models 8 || { say "    FAILED: Nemotron does not answer on localhost:28090 (it runs inside the workflow, on box-a). Try: $0 --direct"; exit 1; }
"$PY" router_switching/examples/nemotron_agent.py --api "http://127.0.0.1:$PORT" --llm http://127.0.0.1:28090/v1 --key "$(cat "$KEYFILE")" \
    "${1:-Run the inference switching check and tell me whether switching worked, which GPU served the requests in each step, and how fast the router reacted.}"
