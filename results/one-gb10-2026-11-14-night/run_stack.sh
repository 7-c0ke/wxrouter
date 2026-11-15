#!/usr/bin/env bash
# Runs inside the container. Same programs and options as fuzzball/30-real-stack.fz, with local addresses.
set -u
cd /repo
B=/night/bundle; H=127.0.0.1
FT="python weather/finetune.py --init $B/weights/afno_event.pt --analyses /gfs --stats $B --steps 300 --rollout 4 --lr 3e-5"
python serve/node_stack.py --name box-a --admin --host $H --port 19109 --log-dir /night/logs/box-a \
  --serve forecast="python serve/forecast_server.py --node box-a --host $H --port 19001 --admin --weights $B/weights/serving.pt --data $B" \
  --job finetune="$FT --out /night/ft/box-a" &
python serve/node_stack.py --name box-b --admin --host $H --port 19110 --log-dir /night/logs/box-b \
  --serve forecast="python serve/forecast_server.py --node box-b --host $H --port 19002 --admin --weights $B/weights/serving.pt --data $B" \
  --job finetune="$FT --out /night/ft/box-b" &
python - <<'PY'
import json, time, urllib.request
end = time.time() + 240
for port in (19001, 19002):
    while time.time() < end:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if json.loads(r.read().decode()).get("ready"):
                    print("[night] forecast server on port", port, "is ready", flush=True); break
        except Exception:
            pass
        time.sleep(1)
PY
python router.py --host $H --port 19000 --policy balanced --preferred box-b --timeout 30 \
  --backend box-a=http://$H:19001 --backend box-b=http://$H:19002 &
python router/router.py --host $H --port 19100 --policy least_loaded \
  --forecast box-a=http://$H:19001,box-b=http://$H:19002 --llm spark=http://$H:18090 --llm-probe llamacpp &
sleep 3
(cd ui && exec python server.py --host $H --port 19080 --router http://$H:19000 \
  --box box-a=http://$H:19001 --box box-b=http://$H:19002 \
  --stack box-a=http://$H:19109 --stack box-b=http://$H:19110 \
  --llm-base-url http://$H:19100/v1 --llm-model nemotron --llm-thinking off) &
echo "[night] all parts started" 
wait
