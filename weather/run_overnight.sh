#!/usr/bin/env bash
# The overnight run on the DGX Spark: stage 1 -> stage 2 -> scoring, each in Docker, holding the GPU lock throughout.
# Written at the hackathon (Fri Nov 13, 2026).
#
#   run_overnight.sh CODE_DIR OUT_DIR [--smoke]
#
#   CODE_DIR  a copy of this repo on the Spark (mounted read-only at /code)
#   OUT_DIR   where everything is written (created if missing; mounted at /out)
#   --smoke   3 optimizer steps per stage and 4 scoring starts, to check the plumbing in a few minutes.
#             Give it a scratch OUT_DIR: a smoke folder cannot be reused for the real run.
#
# Safe to run again after an interruption: a stage whose final.pt exists is skipped, and an unfinished stage
# resumes from its latest.pt. When the sequence ends, OUT_DIR/DONE lists the times and each step's exit code.

set -u -o pipefail

# The overnight run (Nov 13-14, 2026) used this script with these three set for a team member's own DGX Spark
# (one GB10; not one of the event's boxes).
# Only these lines and the wording of one message were changed afterwards.
LOCK=${WX_LOCK:?set WX_LOCK to a lock file that guards the GPU, for example /tmp/gpu.lock}
DATA=${WX_DATA_DIR:?set WX_DATA_DIR to the folder with packed_1979_2019.npy, packed_2020.npy and stats.json}
IMAGE=${WX_IMAGE:?set WX_IMAGE to a container image with PyTorch 2.11 and PhysicsNeMo 2.2.2}
LOCK_HELD_CODE=75                      # flock's exit code when someone else holds the lock

usage() { echo "usage: $0 CODE_DIR OUT_DIR [--smoke]" >&2; exit 2; }
[ $# -ge 2 ] && [ $# -le 3 ] || usage
SMOKE=0
if [ $# -eq 3 ]; then
    [ "$3" = "--smoke" ] || usage
    SMOKE=1
fi

# Hold the GPU lock for the whole sequence: run this same script again under flock, and wait for it.
if [ "${WX_HOLDS_GPU_LOCK:-}" != "1" ]; then
    WX_HOLDS_GPU_LOCK=1 flock -n -E "$LOCK_HELD_CODE" "$LOCK" bash "$0" "$@"
    rc=$?
    if [ "$rc" -eq "$LOCK_HELD_CODE" ]; then
        echo "The GPU lock ($LOCK) is held by another job. Nothing was started; try again when it is free." >&2
    fi
    exit "$rc"
fi

[ -f "$1/weather/train.py" ] || { echo "no weather/train.py under CODE_DIR ($1)" >&2; exit 2; }
[ -d "$DATA" ] || { echo "data folder $DATA is missing" >&2; exit 2; }
mkdir -p "$2" || exit 2
CODE_DIR=$(cd "$1" && pwd)             # Docker needs absolute paths
OUT_DIR=$(cd "$2" && pwd)

# A smoke run leaves 3-step final.pt files, which a real run would take for finished stages. Keep them apart.
if [ "$SMOKE" -eq 1 ]; then
    if [ -e "$OUT_DIR/stage1" ] && [ ! -e "$OUT_DIR/SMOKE" ]; then
        echo "$OUT_DIR already holds a real run; give --smoke a scratch folder." >&2; exit 2
    fi
    touch "$OUT_DIR/SMOKE"
elif [ -e "$OUT_DIR/SMOKE" ]; then
    echo "$OUT_DIR holds a smoke run; use a fresh OUT_DIR for the real run." >&2; exit 2
fi

mkdir -p "$OUT_DIR/home" "$OUT_DIR/cache/inductor" "$OUT_DIR/cache/triton" "$OUT_DIR/logs" || exit 2
rm -f "$OUT_DIR/DONE"                  # a DONE left by an earlier attempt would be misleading while this one runs

say() { echo "$@" | tee -a "$OUT_DIR/logs/overnight.log"; }

wall() { printf '%dh %02dm %02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# in_docker STEP COMMAND...   Runs the command in the container; output goes to the terminal and logs/STEP.log.
# Sets STEP_RC to the command's exit code.
in_docker() {
    local step=$1; shift
    local t0=$SECONDS
    say "=== $step: started $(date -Iseconds)"
    docker run --rm --name "wx-event-$step" --gpus all --ipc=host --network none \
        --user "$(id -u):$(id -g)" \
        -e HOME=/out/home -e USER=cory \
        -e TORCHINDUCTOR_CACHE_DIR=/out/cache/inductor -e TRITON_CACHE_DIR=/out/cache/triton \
        -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e WX_DATA=/data \
        -v "$DATA":/data:ro -v "$CODE_DIR":/code:ro -v "$OUT_DIR":/out -w /out \
        "$IMAGE" "$@" 2>&1 | tee -a "$OUT_DIR/logs/$step.log"
    STEP_RC=${PIPESTATUS[0]}
    say "=== $step: exit code $STEP_RC, wall time $(wall $((SECONDS - t0)))"
}

START=$(date -Iseconds)
T_START=$SECONDS
MODE=full
TRAIN_EXTRA=""                         # extra arguments, used unquoted below so that empty means none
SCORE_EXTRA=""
if [ "$SMOKE" -eq 1 ]; then
    MODE=smoke
    TRAIN_EXTRA="--max-steps 3"
    SCORE_EXTRA="--limit 4"
fi
say "run_overnight ($MODE): code $CODE_DIR, out $OUT_DIR, started $START"

# 1. Stage 1: one 6-hour step.
if [ -f "$OUT_DIR/stage1/final.pt" ]; then
    say "=== stage1: skipped, stage1/final.pt exists"
    RC1="skipped (final.pt exists)"
else
    in_docker stage1 python /code/weather/train.py --stage 1 --out /out/stage1 $TRAIN_EXTRA
    RC1=$STEP_RC
fi

# 2. Stage 2: four fed-back steps, from stage 1's final weights.
if [ -f "$OUT_DIR/stage2/final.pt" ]; then
    say "=== stage2: skipped, stage2/final.pt exists"
    RC2="skipped (final.pt exists)"
elif [ ! -f "$OUT_DIR/stage1/final.pt" ]; then
    say "=== stage2: not run, stage 1 left no final.pt"
    RC2="not run (no stage1/final.pt)"
else
    in_docker stage2 python /code/weather/train.py --stage 2 --out /out/stage2 --init /out/stage1/final.pt \
        $TRAIN_EXTRA
    RC2=$STEP_RC
fi

# 3. Scoring. Another script's job; if it is missing or fails, the trained weights are still the result.
if [ ! -f "$OUT_DIR/stage2/final.pt" ]; then
    say "=== score: not run, stage 2 left no final.pt"
    RC3="not run (no stage2/final.pt)"
elif [ ! -f "$CODE_DIR/weather/score.py" ]; then
    say "=== score: not run, there is no weather/score.py in $CODE_DIR"
    RC3="not run (no score.py)"
else
    in_docker score python /code/weather/score.py --weights /out/stage2/final.pt --data /data --out /out/scores \
        --years 2020 2019 --storm $SCORE_EXTRA
    RC3=$STEP_RC
    [ "$RC3" = "0" ] || say "=== score: FAILED (exit code $RC3); carrying on, the weights are in stage2/final.pt"
fi

# 4. The record of the run.
{
    echo "mode:   $MODE"
    echo "start:  $START"
    echo "end:    $(date -Iseconds)"
    echo "wall:   $(wall $((SECONDS - T_START)))"
    echo "stage1: $RC1"
    echo "stage2: $RC2"
    echo "score:  $RC3"
} > "$OUT_DIR/DONE"
say "--- $OUT_DIR/DONE"
say "$(cat "$OUT_DIR/DONE")"

if [ "$SMOKE" -eq 1 ]; then
    say "--- stage 1, steps 1-3 of this smoke run (from stage1/log.jsonl):"
    say "$(grep -E '^\{"step": [123],' "$OUT_DIR/stage1/log.jsonl" 2>/dev/null || echo '(no step lines found)')"
    say "--- our research run before the event: step 1 loss 16.459454 grad_norm 48.1035 | step 2 loss 16.363712 grad_norm 47.6809 | step 3 loss 16.184488"
fi

# Training decides success; a scoring failure does not fail the run.
[ -f "$OUT_DIR/stage2/final.pt" ] || exit 1
exit 0
