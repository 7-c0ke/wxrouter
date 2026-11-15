# wxrouter on the onprem cluster (step 8)

Once `fuzzball node list` shows both 3090 servers with `nvidia.com/gpu: 1`, wxrouter deploys with the usual
`deploy/fb.py`, after these changes. Each is small; none is made yet.

| what | on the event cluster | onprem | where to change |
|---|---|---|---|
| CLI context | `default` → `api.fb-hack.civilfritz.net` | `onprem` → `api.<METALLB_IP>.nip.io` (README step 4) | `fuzzball context use onprem` before `fb.py` |
| Group | `team14` | the group created in step 4 | `fuzzball group use <group>` |
| Team volume | `wxdesk` in group `team14` (`use: team14`) | must exist on the onprem cluster | `fuzzball/30-real-stack.fz` and `fuzzball/quick/*.fz`: `volumes: wx: use: <group>, name: <volume>`. **Open:** how Fuzzball volumes are backed onprem (storage driver) is not in our notes yet |
| llama.cpp build | `ubuntu-cuda-13.5-arm64` (GB10, aarch64) | **x86_64** build (both servers are AMD x86_64) | `fuzzball/fetch_assets.sh`: pick the asset by `uname -m`, with its own SHA-256 |
| CUDA / driver | GB10, CUDA 13.5 build | RTX 3090 (Ampere, sm_86); llama.cpp's CUDA 13 builds need a recent driver | check `nvidia-smi` on both servers; use a CUDA 12 build if the driver is older |
| GPU memory | GB10 unified memory | **24 GB** per 3090 | measured on GB10: LLM 18.7 GB + forecast 0.4 GB, so it fits; lower `-c 65536` to `-c 32768` in `30-real-stack.fz` for headroom. A fine-tune beside the LLM may not fit |
| Container image | `nvcr.io/nvidia/pytorch:26.02-py3` (arm64) | same tag, amd64 (the image is multi-arch) | none, but each node needs space for it (>10 GB) |
| Disk | volume on the cluster | Nemotron GGUF 18.9 GB + model 77 MB + NOAA data on the volume | `neal-master` has ~28 GB free on `/` |

Then:

```bash
python deploy/fb.py check
python deploy/fb.py setup     # volume: repo, packages, AFNO model, llama.cpp, NOAA analysis; then Nemotron (18.9 GB)
python deploy/fb.py up        # box-a and box-b on the two 3090s, router, router2, Control Room
python deploy/fb.py open      # http://localhost:8080 via port-forward
```

The page should show **REAL MODEL** on both boxes (model_version `afno-19m-...`), and the chat should answer
through Nemotron on `llama-server`.
