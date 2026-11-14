# Non-GPU tasks: results, 14 Nov 2026 (afternoon)

These are the tasks from the 2:10 PM status page that need no GPU. Everything was run on a laptop with mock boxes, at `main` = `2c21a41`.

## Test suite on `main`

`pytest -q`: **630 passed**, 41 skipped, **9 failed, 4 errors**. None of these point at the router or the UI.

| Tests | Cause | Fix |
|---|---|---|
| `test_gfs_live.py` (7 failed) | `pygrib` is not installed on the laptop | Environment only. They pass where `pygrib` is installed (the volume's `/wx/pydeps`). |
| `test_train.py` (4 errors) | `fixture 'tmp' not found`: the tests ask for a `tmp` fixture that no `conftest.py` defines | Add a `tmp` fixture (wrapping `tmp_path`) or rename the parameter. A real test bug, but small. |
| `test_inference_server.py::test_contract_version_matches` | `inference/` still says contract `1.0`; `contract.py` says `1.1` | **Fixed by PR 12** (see below). |
| `test_inference_server.py::test_step_hours_ensemble_and_low_confidence` | Ensemble members 0 and 1 come back identical | Still fails with PR 12. For the `inference/` owner to look at. |

## Task 6: PR 12 review (two-box inference, switching demo)

- Merges into current `main` **with no conflicts**.
- With PR 12: **666 passed** (+36, including `router_switching/tests`), 8 failed, 4 errors. It fixes the contract-version test and adds no new failures.
- Recommendation: **merge**. It still conflicts on the GPUs (the inference demo and the real stack can't both run), so pick one stack per time slot (task 1).

## Task 4: split forecast (one big job on two boxes)

`bench/scene_split.py` with two `serve/forecast_server.py --mock` boxes, 20 members × 48 h, 5 runs each way:

| | Median | Range |
|---|---|---|
| One box alone | 1,607.9 ms | 1,604 to 1,610 ms |
| Split 10 + 10 over two boxes | **811.7 ms** | 811 to 812 ms |

- **Speed-up 1.98×.** All 5 splits were even.
- **Same answer:** mean temperature differs by at most 0.005 °C; member 0 and the lowest and highest members are identical.
- **Recommendation:** use `router/router.py` (it already splits ensembles) as the entry point rather than moving the split into the root `router.py`. Confirm with one real-box run (needs both GPUs about 2 minutes).

## Task 3 (dry run): model swap under traffic

`bench/swap_weights.py --no-link` against two `serve/node_stack.py` supervisors running mock forecast servers, behind `router.py --policy adaptive`:

- box-a restarted in 1.3 s, box-b in 1.3 s.
- **72 requests, 72 answered, 0 failed**, 1 answered by the other box (failover worked). p50 91 ms, p95 100 ms.
- A mock has no weights, so only the restart path is proven. The real swap (`--to /wx/ft/box-a/tuned.pt`) still needs the real stack (about 5 min).

## Task 8: one entry address for chat and forecasts

`router/router.py` already does this; no new code is needed. Tested with two mock forecast servers and two `serve/mock_llm.py`:

```bash
python router/router.py --port 8300 \
  --forecast A=http://box-a:8001,B=http://box-b:8002 \
  --llm A=http://box-a:8090,B=http://box-b:8091 --llm-probe llamacpp
```

- `POST /v1/forecast` and `POST /v1/chat/completions` both answered 200 on the same port.
- Chat stayed on one box (`X-Backend: A`): the llm pool has session affinity on by default (it keeps the KV cache warm). This is expected.
- To use it in the real stack, the UI's `--llm-base-url` becomes `http://router:8000/v1`, and the router service gets the `--llm` pool. **Not tried on real llama.cpp yet.**

## Still needs the GPUs

Task 2 (training scene), task 3 (real swap), task 5 (fault benchmark), task 7 (default rule, from #5), task 9 (chat latency, can use the 30-min `deployment_arin1` slot), task 10 (rehearsal and backup video).

## Feedback notes for the organizers (task 11)

What cost us time on Fuzzball today, in order of hours lost:

1. **Each team has 2 GPU nodes, but the cluster shows 40.** `node list` shows 40 healthy Sparks with free GPUs. Team14's work only runs on provisioner `team14` (SPARK27, SPARK28). When a teammate holds both, a new run says `READY_TO_SCHEDULE: ready to schedule on next tick` for 20+ minutes. That message reads like a scheduler fault, not "your team's GPUs are busy". Ask: say "waiting for provisioner team14 (2 nodes, both busy)" in `workflow why`, or document the per-team node limit.
2. **Two groups per team.** Each account has a personal group (`team14@fb-hack.local`) and the team group (`team14`). The CLI defaulted to the personal one, so CLI runs didn't show on the team dashboard, and dashboard runs weren't visible to the CLI. Ask: default to the team group, or say which group is active in `workflow start` output.
3. **`fuzzball queue show` needs cluster admin.** There is no way for a team to see what's ahead of it.
4. **Docs are for 4.4; the cluster runs 4.3.1.** Some syntax in the examples (for example `cpu/arm64: 1` as a device request) seemed to stall placement on 4.3.1.
5. **GB10 `nvidia-smi` reports memory and power as "Not Supported".** Dashboards that read them show 0 instead of unknown. Worth a note in the hardware sheet.
6. **Image-cache outages** (`lookup fuzzball-objectcache: no such host`) around 9:50 and 11:10 failed workflows in the image stage, with no retry.
