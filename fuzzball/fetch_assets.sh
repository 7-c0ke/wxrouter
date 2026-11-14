#!/bin/bash
# fetch_assets.sh: put the weather model, the LLM and its server on the team volume (/wx). No GPU needed.
#
# A job started from the Fuzzball web page can fetch this file and run it with three lines:
#
#     mkdir -p /wx/tmp && cd /wx/tmp
#     curl -fsSL -o fetch_assets.sh https://raw.githubusercontent.com/OWNER/REPO/BRANCH/fuzzball/fetch_assets.sh
#     bash fetch_assets.sh
#
# Run it inside the PyTorch image (docker://nvcr.io/nvidia/pytorch:26.02-py3) with the team volume at /wx.
# Every step can be run again: a file that is already there with the right checksum is not fetched again.
# WHAT chooses the steps (default: all four): model, llama, llm, state.
#
#   model  /wx/bundle/weights/afno_event.pt, /wx/bundle/stats.json   the weather model (77 MB)
#   llama  /wx/llama/llama-server                                    llama.cpp's own release build for CUDA on arm64
#   llm    /wx/models/<file>.gguf                                    NVIDIA Nemotron 3.5 Lightning, 4-bit (18.9 GB)
#   state  /wx/bundle/states/YYYYMMDDHH.npy                          today's starting weather from NOAA (needs /wx/repo)
#
# Written at the hackathon (Nov 14, 2026).
WHAT=${WHAT:-model llama llm state}
MODEL_BASE=${MODEL_BASE:-https://github.com/CoryGardner/wxrouter/releases/download/model-v1}
MODEL_SHA256=${MODEL_SHA256:-65f9f3bb327035ecad200c4e1b63c4411fa8d8c71a109c63c61d4291a3328c28}
STATS_SHA256=${STATS_SHA256:-22132a6937913fd5d5eec9901aa9557911d7f16a394a612823978bce5e927886}
LLAMA_BUILD=${LLAMA_BUILD:-b12359}
LLAMA_BASE=${LLAMA_BASE:-https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD}
LLAMA_SHA256=${LLAMA_SHA256:-e12fd3ca9a599349cac5ec767825e17f8cab9dc2dab8d8e5b85d50ce21ef6d7d}
CUDART_SHA256=${CUDART_SHA256:-998f62318999710d398dad40821d1d037731f360791e41628d9dfcde207909d1}
LLM_URL=${LLM_URL:-https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/main/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}
LLM_SHA256=${LLM_SHA256:-61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c}

FAILED=""
note() { echo; echo "--- $*"; }
bad() { echo "PROBLEM ($1): $2"; FAILED="$FAILED $1"; }
wanted() { case " $WHAT " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

mkdir -p /wx/home /wx/tmp /wx/bundle/weights /wx/bundle/states /wx/llama /wx/models /wx/kv \
    || { echo "PROBLEM: cannot write to /wx (the team volume)"; exit 1; }
export HOME=/wx/home TMPDIR=/wx/tmp

# fetch URL FILE SHA256: download (a broken download goes on where it stopped), then check the checksum.
fetch() {
    url=$1; file=$2; want=$3
    if [ -s "$file" ] && [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$want" ]; then
        echo "already there, checksum right: $file"
        return 0
    fi
    if [ -s "$file.part" ] && [ "$(sha256sum "$file.part" | cut -d' ' -f1)" = "$want" ]; then
        mv "$file.part" "$file"
        echo "an earlier download was complete: $file"
        return 0
    fi
    start=$(date +%s)
    if ! curl -fL --retry 5 --retry-delay 5 -C - -sS -o "$file.part" "$url"; then
        echo "download failed: $url"
        return 1
    fi
    got=$(sha256sum "$file.part" | cut -d' ' -f1)
    if [ "$got" != "$want" ]; then
        echo "wrong checksum for $file: got $got, wanted $want. The partial file is removed."
        rm -f "$file.part"
        return 1
    fi
    mv "$file.part" "$file"
    seconds=$(( $(date +%s) - start ))
    echo "fetched $file: $(stat -c %s "$file") bytes in $seconds s (download and checksum)"
}

if wanted model; then
    note "model: the weather model -> /wx/bundle"
    fetch "$MODEL_BASE/afno_event.pt" /wx/bundle/weights/afno_event.pt "$MODEL_SHA256" || bad model "the weights did not arrive"
    fetch "$MODEL_BASE/stats.json" /wx/bundle/stats.json "$STATS_SHA256" || bad model "stats.json did not arrive"
fi

if wanted llama; then
    note "llama: llama.cpp $LLAMA_BUILD (CUDA, arm64) -> /wx/llama"
    cd /wx/llama || exit 1
    a=llama-$LLAMA_BUILD-bin-ubuntu-cuda-13.5-arm64.tar.gz
    b=cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-13.5-arm64.tar.gz
    if fetch "$LLAMA_BASE/$a" "/wx/llama/$a" "$LLAMA_SHA256" && fetch "$LLAMA_BASE/$b" "/wx/llama/$b" "$CUDART_SHA256"; then
        tar -xzf "$a" --strip-components=1 && tar -xzf "$b" --strip-components=1 || bad llama "could not unpack the release files"
        if [ -x /wx/llama/llama-server ]; then
            LD_LIBRARY_PATH=/wx/llama:${LD_LIBRARY_PATH:-} /wx/llama/llama-server --version 2>&1 | head -5 \
                || bad llama "llama-server does not start in this image (a library is missing: see the lines above)"
        else
            bad llama "no llama-server after unpacking; what is there: $(ls /wx/llama | head -20 | tr '\n' ' ')"
        fi
    else
        bad llama "the release files did not arrive"
    fi
fi

if wanted llm; then
    note "llm: $(basename "$LLM_URL") -> /wx/models"
    fetch "$LLM_URL" "/wx/models/$(basename "$LLM_URL")" "$LLM_SHA256" || bad llm "the LLM file did not arrive"
fi

if wanted state; then
    note "state: today's starting weather from NOAA -> /wx/bundle/states"
    if [ ! -f /wx/repo/weather/gfs_live.py ]; then
        bad state "/wx/repo is not there yet: run setup_volume.sh first, then this step again (WHAT=state)"
    elif [ ! -f /wx/bundle/stats.json ]; then
        bad state "/wx/bundle/stats.json is not there: the model step must work first"
    else
        PYTHONPATH=/wx/pydeps PYTHONUSERBASE=/wx/pyuser python /wx/repo/weather/gfs_live.py --out /wx/bundle/states --data /wx/bundle \
            || bad state "gfs_live.py failed (see the lines above)"
    fi
fi

note "what is on the volume now"
for f in /wx/bundle/weights/afno_event.pt /wx/bundle/stats.json /wx/llama/llama-server \
    "/wx/models/$(basename "$LLM_URL")" /wx/repo/serve/forecast_server.py; do
    if [ -e "$f" ]; then echo "  yes      $(ls -la "$f" | awk '{print $5}') bytes  $f"; else echo "  MISSING  $f"; fi
done
echo "  states: $(ls /wx/bundle/states 2>/dev/null | tr '\n' ' ')"

if [ -n "$FAILED" ]; then
    echo; echo "FETCH FAILED in:$FAILED"
    exit 1
fi
echo; echo "FETCH DONE"
