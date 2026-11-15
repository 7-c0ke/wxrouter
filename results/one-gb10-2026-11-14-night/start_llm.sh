#!/usr/bin/env bash
# Our own LLM server for tonight's tests: the same Nemotron file as on the event boxes, on port 18090 of this machine only.
# It does not touch llama-swap (port 8080) or any other service. Stop it with: docker stop wxnight-llm
set -u
docker run -d --rm --name wxnight-llm --gpus all --ipc=host --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v $HOME/models:/models:ro \
  sc26/llama-server-cuda:c25030496 \
  --model /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf --alias nemotron --host 127.0.0.1 --port 18090 \
  --n-gpu-layers 999 --metrics -c 65536 -np 4 -fa on --jinja
