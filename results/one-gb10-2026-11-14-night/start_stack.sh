#!/usr/bin/env bash
# The team's stack on this one machine, as the event stack file starts it, on local ports 19000-19110.
# usage: start_stack.sh <repo folder>      Stop it with: docker stop -t 30 wxnight-stack
set -u
REPO=$(cd "$1" && pwd)
N=$HOME/wx-event/night
mkdir -p $N/logs/box-a $N/logs/box-b $N/ft
docker run -d --rm --name wxnight-stack --gpus all --ipc=host --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/deps \
  -v "$REPO":/repo:ro -v $N:/night -v $N/pydeps:/deps:ro -v $HOME/wx-event/gfs-year:/gfs:ro \
  -w /repo sc26/afno-r4fd:20261021 bash /night/run_stack.sh
