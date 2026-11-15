#!/usr/bin/env bash
# As start_stack.sh, with the cores of the event stack's box service (0 to 11 of a GB10: five fast, seven slow).
# usage: start_stack_cores.sh <repo folder> <cores, like 0-11>
set -u
REPO=$(cd "$1" && pwd)
N=$HOME/wx-event/night
mkdir -p $N/logs/box-a $N/logs/box-b $N/ft
docker run -d --rm --name wxnight-stack --cpuset-cpus "$2" --gpus all --ipc=host --network host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/deps \
  -v "$REPO":/repo:ro -v $N:/night -v $N/pydeps:/deps:ro -v $HOME/wx-event/gfs-year:/gfs:ro \
  -w /repo sc26/afno-r4fd:20261021 bash /night/run_stack.sh
