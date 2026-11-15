#!/usr/bin/env bash
# The static case of bench/fault_table.py with the cores of the event stack's box service (0-11), first with
# the forecast server as it is, then with the server that pins itself to the fast cores.
set -u
N=$HOME/wx-event/night
run_case() {
  docker stop -t 30 wxnight-stack >/dev/null 2>&1; sleep 8
  bash $N/start_stack_cores.sh $N/repo-fix 0-11 >/dev/null
  for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:19080/api/overview >/dev/null 2>&1 && break; sleep 2; done
  sleep 6
  echo "== $1: $(date)"
  echo "start-up line about cores: $(grep -h 'cores:' $N/logs/box-a/forecast.log | tail -1)"
  docker exec wxnight-stack python -c "
import json, os, urllib.request
pid=json.loads(urllib.request.urlopen('http://127.0.0.1:19109/status',timeout=5).read())['servers']['forecast']['pid']
print('forecast server of box-a: pid', pid, 'may use cores', sorted(os.sched_getaffinity(pid)), '| this container:', sorted(os.sched_getaffinity(0)))"
  rm -f $N/out/core-watch.stop
  docker exec -d wxnight-stack python /night/core_watch.py /night/out/core-watch-$1.txt
  docker exec wxnight-stack bash -lc "cd /repo && python bench/fault_table.py --router http://127.0.0.1:19000 --stack box-a=http://127.0.0.1:19109 --stack box-b=http://127.0.0.1:19110 --rules static --events 10:slow:box-a,90:normal:box-a --job-args '--out /night/ft/fault-table' --out /night/out/fault-static-$1" 2>&1 | grep -E "p50|wrote|Traceback|rror"
  touch $N/out/core-watch.stop; sleep 2
}
: > $N/logs/box-a/forecast.log.marker
run_case cores-unpinned
cp $N/repo-fix/serve/forecast_server.py $N/forecast_server.unpinned.py
cp $N/forecast_server.pinned.py $N/repo-fix/serve/forecast_server.py
run_case cores-pinned
echo "done $(date)"
