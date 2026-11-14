"""Tests for weather/train.py. Written at the hackathon (Fri Nov 13, 2026).

    python tests/test_train.py            (takes about five minutes on a laptop CPU: it trains the real model)

Plain asserts; each test prints a PASS line. The slow tests run train.py as separate processes, because
"stop, restart, carry on" is the thing being tested. Temporary files go under $TMPDIR and are removed at the end.
"""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER = os.path.join(os.path.dirname(HERE), "weather")
TRAIN = os.path.join(WEATHER, "train.py")
sys.path.insert(0, WEATHER)

import train            # noqa: E402
import wx_common as wx  # noqa: E402

# Holds stats.json and a sparse packed_1979_2019.npy with only the 64 rows that the first optimizer step reads.
FIRSTSTEP = os.environ.get("WX_TEST_FIRSTSTEP", os.path.join(os.path.dirname(HERE), ".local", "firststep"))


def run_train(*args, expect_ok=True):
    """Run train.py on the CPU in its own process. Returns the finished process."""
    cmd = [sys.executable, TRAIN, "--device", "cpu", "--no-compile", *[str(a) for a in args]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"train.py failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
    return p


def read_log(out):
    with open(os.path.join(out, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def step_lines(out):
    """The per-step log lines, keyed by step (a later line for the same step replaces an earlier one)."""
    return {e["step"]: e for e in read_log(out) if "event" not in e}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def max_weight_difference(a, b):
    assert a.keys() == b.keys()
    return max((a[k].double() - b[k].double()).abs().max().item() for k in a)


# ------------------------------------------------------------------------------------------------ fast tests

def test_learning_rates():
    for s in range(18828):
        want = 1e-3 * min(1, (s + 1) / 500) if s < 16004 else 1e-3 * (2824 - (s - 16004)) / 2824
        assert train.lr_stage1(s) == want, s
        assert train.learning_rate({"stage": 1, "total_steps": 18828, "decay_steps": 2824}, s) == want, s
    assert train.lr_stage1(0) == 2e-6 and train.lr_stage1(499) == 1e-3
    assert train.lr_stage1(16003) == 1e-3 and train.lr_stage1(16004) == 1e-3      # the decay starts from the peak
    assert train.lr_stage1(16005) < 1e-3
    assert 0 < train.lr_stage1(18827) == 1e-3 * 1 / 2824

    n = 1564
    for s in range(n):
        want = 1e-4 * (s + 1) / 100 if s < 100 else 1e-4 * (n - s) / (n - 100)
        assert train.lr_stage2(s) == want, s
        assert train.learning_rate({"stage": 2, "total_steps": n, "decay_steps": n - 100}, s) == want, s
    assert train.lr_stage2(0) == 1e-6 and train.lr_stage2(99) == 1e-4 and train.lr_stage2(100) == 1e-4
    assert 0 < train.lr_stage2(n - 1) == 1e-4 * 1 / 1464

    assert train.STAGES[1]["total_steps"] == 18828 and train.STAGES[2]["total_steps"] == 1564
    print("PASS (a) learning rates: both stages match the formulas at every step; the last rates are > 0")


def test_stream():
    n = 56979
    assert wx.N_TRAIN_TIMES - 1 == n
    first = train.StartStream(n, 1).take(32)
    assert np.array_equal(first, np.random.default_rng([1, 0]).permutation(n)[:32])

    # Taking batch after batch is the same as cutting up the concatenated epochs.
    epochs = np.concatenate([np.random.default_rng([1, e]).permutation(n) for e in range(2)])
    stream = train.StartStream(n, 1)
    for i in range(3):
        assert np.array_equal(stream.take(32), epochs[32 * i:32 * i + 32])

    # A batch that straddles the end of epoch 0: 10 rows from epoch 0, then 22 from epoch 1.
    stream = train.StartStream(n, 1, epoch=0, pos=n - 10)
    assert np.array_equal(stream.take(32), epochs[n - 10:n + 22])
    assert stream.state() == (1, 22)

    # A stream restarted from a saved state carries on exactly where the first one was.
    a = train.StartStream(n, 1)
    for _ in range(5):
        a.take(32)
    b = train.StartStream(n, 1, *a.state())
    assert np.array_equal(a.take(32), b.take(32))

    # Tiny n (the synthetic tests): one batch spans several epochs.
    small = np.concatenate([np.random.default_rng([7, e]).permutation(10) for e in range(4)])
    stream = train.StartStream(10, 7)
    assert np.array_equal(stream.take(32), small[:32]) and stream.state() == (3, 2)

    # Stage 2 draws from 56,976 start rows with the same construction.
    assert np.array_equal(train.StartStream(56976, 1).take(32), np.random.default_rng([1, 0]).permutation(56976)[:32])

    # The prefetch thread hands out, for each start row t, rows t .. t + rollout, in stream order, and stops cleanly.
    data = np.arange(50 * 2 * 3, dtype=np.float16).reshape(50, 2, 3)
    for rollout in (1, 4):
        pre = train.Prefetcher(data, train.StartStream(50 - rollout, 1), rollout, n_batches=3, pin=False)
        check = train.StartStream(50 - rollout, 1)
        for _ in range(3):
            block, state = pre.next()
            starts = check.take(32)
            assert block.shape == (32, rollout + 1, 2, 3) and block.dtype == torch.float16
            for j in range(rollout + 1):
                assert np.array_equal(block[:, j].numpy(), data[starts + j])
            assert state == check.state()
        pre.close()
        assert not pre.thread.is_alive()
    # Closing while the thread is blocked on a full queue must not hang.
    pre = train.Prefetcher(data, train.StartStream(49, 1), 1, n_batches=100, pin=False)
    time.sleep(0.3)
    pre.close()
    assert not pre.thread.is_alive()
    print("PASS (b) stream: first 32 indices, epoch boundaries, restart from a saved state, prefetch thread")


def test_t7_weights():
    w = wx.t7_channel_weights()
    assert w.shape == (20,) and abs(w.mean() - 1) < 1e-12
    for c in range(20):
        want = 60 / 26 if c in (14, 5, 2) else 20 / 26
        assert abs(w[c] - want) < 1e-12, c

    # The loss function is the mean over everything of w_c * (pred - target)^2.
    rng = np.random.default_rng(0)
    pred, target = rng.standard_normal((2, 3, 20, 6, 8)).astype(np.float32)
    want = (w[None, :, None, None] * (pred.astype(np.float64) - target) ** 2).mean()
    wt = torch.tensor(w, dtype=torch.float32).view(1, -1, 1, 1)
    got = train.weighted_mse(torch.from_numpy(pred), torch.from_numpy(target), wt).item()
    assert abs(got - want) < 1e-5 * want
    plain = train.weighted_mse(torch.from_numpy(pred), torch.from_numpy(target)).item()
    assert abs(plain - ((pred.astype(np.float64) - target) ** 2).mean()) < 1e-5 * plain
    print("PASS (c) T7 weights: 60/26 for Z500, T850, T2M and 20/26 for the rest; weighted_mse matches numpy")


# ------------------------------------------------------------------------------------------------ slow tests

def test_first_step(tmp):
    if not os.path.exists(os.path.join(FIRSTSTEP, "packed_1979_2019.npy")):
        print(f"SKIP (d) first step: {FIRSTSTEP} is missing")
        return
    out = os.path.join(tmp, "first")
    run_train("--stage", 1, "--data", FIRSTSTEP, "--out", out, "--max-steps", 1)
    line = step_lines(out)[1]
    assert abs(line["loss"] - 16.4625) <= 0.0005, line
    assert abs(line["grad_norm"] - 48.12) <= 0.01, line
    assert line["lr"] == 2e-6
    start = read_log(out)[0]
    assert start["event"] == "start" and start["params"] == 19128832 and start["seed"] == 1 and start["steps"] == 18828
    final = load(os.path.join(out, "final.pt"))
    assert final["complete"] is False and final["val"] is None and final["step"] == 1
    print(f"PASS (d) first step on real data: loss {line['loss']:.6f} (want 16.4625 +/- 0.0005), "
          f"grad norm {line['grad_norm']:.4f} (want 48.12 +/- 0.01)")


def make_tiny_data(folder, fill=None):
    """40 rows of random float16 'z-scores' (or a constant) in the real layout, plus a stats.json."""
    os.makedirs(folder)
    rng = np.random.default_rng(123)
    if fill is None:
        rows = rng.standard_normal((40, wx.N_CHANNELS, wx.N_LAT, wx.N_LON), dtype=np.float32).astype(np.float16)
    else:
        rows = np.full((40, wx.N_CHANNELS, wx.N_LAT, wx.N_LON), fill, np.float16)
    np.save(os.path.join(folder, "packed_1979_2019.npy"), rows)
    with open(os.path.join(folder, "stats.json"), "w") as f:
        json.dump({"channels": wx.CHANNEL_NAMES, "mean": [0.0] * 20, "std": [1.0] * 20}, f)


def test_resume_and_stage2(tmp):
    data = os.path.join(tmp, "tiny")
    make_tiny_data(data)
    # 30 training rows and 10 validation rows; 4 steps in all, the last 2 on the decay.
    tiny = ["--data", data, "--test-train-rows", 30, "--test-val-rows", 10]
    stage1 = ["--stage", 1, *tiny, "--test-total-steps", 4, "--test-decay-steps", 2]

    one_go = os.path.join(tmp, "one_go")
    run_train(*stage1, "--out", one_go)
    a = load(os.path.join(one_go, "final.pt"))
    a_steps = step_lines(one_go)
    assert sorted(a_steps) == [1, 2, 3, 4]
    assert [a_steps[s]["lr"] for s in (1, 2, 3, 4)] == [2e-6, 4e-6, 1e-3, 5e-4]
    assert all(math.isfinite(a_steps[s]["loss"]) and math.isfinite(a_steps[s]["grad_norm"]) for s in a_steps)
    assert set(a) >= {"model", "step", "seed", "recipe", "stage"}
    assert a["step"] == 4 and a["stage"] == 1 and a["seed"] == 1 and a["complete"] is True
    assert math.isfinite(a["val"]["mse_1step"]) and a["val"]["mse_1step"] > 0
    events = [e.get("event") for e in read_log(one_go)]
    assert events[0] == "start" and events[-1] == "end" and "checkpoint" in events
    end = read_log(one_go)[-1]
    assert end["steps_per_s"] > 0 and end["val"] == a["val"]

    # The same run cut in two: 2 steps, a new process, 2 more steps.
    two_parts = os.path.join(tmp, "two_parts")
    run_train(*stage1, "--out", two_parts, "--max-steps", 2)
    part = load(os.path.join(two_parts, "latest.pt"))
    assert part["step"] == 2 and set(part) >= {"model", "optimizer", "step", "stream", "torch_rng", "recipe"}
    assert tuple(part["stream"]) == (2, 6)       # 64 start rows drawn from epochs of 29: 29 + 29 + 6
    assert load(os.path.join(two_parts, "final.pt"))["complete"] is False
    # A short test run pointed at a folder that is already further along is refused.
    p = run_train(*stage1, "--out", two_parts, "--max-steps", 1, expect_ok=False)
    assert p.returncode != 0 and "beyond --max-steps" in p.stderr
    p = run_train(*stage1, "--out", two_parts)
    assert "(resumed)" in p.stdout
    b = load(os.path.join(two_parts, "final.pt"))
    b_steps = step_lines(two_parts)
    assert b["step"] == 4 and b["complete"] is True
    diff = max_weight_difference(a["model"], b["model"])
    assert diff < 1e-6, diff
    for s in (1, 2, 3, 4):
        assert abs(a_steps[s]["loss"] - b_steps[s]["loss"]) < 1e-6 and a_steps[s]["lr"] == b_steps[s]["lr"], s
    assert abs(a["val"]["mse_1step"] - b["val"]["mse_1step"]) < 1e-6
    assert tuple(load(os.path.join(one_go, "latest.pt"))["stream"]) == tuple(load(os.path.join(two_parts, "latest.pt"))["stream"])
    print(f"PASS (e1) stage 1 resume: 4 steps in one process = 2 + restart + 2 (max weight difference {diff:.1e})")

    # Running a finished stage again does nothing.
    p = run_train(*stage1, "--out", two_parts)
    assert "already finished" in p.stdout
    # A folder from a different run is refused rather than silently continued.
    p = run_train("--stage", 1, *tiny, "--test-total-steps", 6, "--test-decay-steps", 2, "--out", two_parts, expect_ok=False)
    assert p.returncode != 0 and "Use a different --out" in p.stderr
    print("PASS (e2) a finished stage is left alone; a mismatched --out is refused")

    # Stage 2: two steps from stage 1's weights, a fresh optimizer, the stage-2 warm-up rates, then validation.
    stage2 = os.path.join(tmp, "stage2")
    p = run_train("--stage", 2, *tiny, "--test-total-steps", 2, "--init", os.path.join(one_go, "final.pt"), "--out", stage2)
    c_steps = step_lines(stage2)
    assert [c_steps[s]["lr"] for s in (1, 2)] == [1e-6, 2e-6] == [train.lr_stage2(0), train.lr_stage2(1)]
    assert all(math.isfinite(c_steps[s]["loss"]) and math.isfinite(c_steps[s]["grad_norm"]) for s in (1, 2))
    c = load(os.path.join(stage2, "final.pt"))
    assert c["stage"] == 2 and c["step"] == 2 and c["complete"] is True and c["recipe"]["rollout"] == 4
    assert math.isfinite(c["val"]["mse_1step"]) and math.isfinite(c["val"]["mse_4step_rollout"])
    assert max_weight_difference(a["model"], c["model"]) > 0          # it did train
    assert tuple(load(os.path.join(stage2, "latest.pt"))["stream"]) == (2, 12)   # 64 start rows from epochs of 26
    # Stage 2 without --init must refuse to start from random weights.
    p = run_train("--stage", 2, *tiny, "--out", os.path.join(tmp, "stage2_noinit"), expect_ok=False)
    assert p.returncode != 0 and "--init" in p.stderr
    print(f"PASS (e3) stage 2: 2 steps from the stage-1 checkpoint, losses {c_steps[1]['loss']:.4f}, "
          f"{c_steps[2]['loss']:.4f}, rates 1e-6 and 2e-6, 4-step validation {c['val']['mse_4step_rollout']:.4f}")
    return stage1, a


def test_interrupt(tmp, stage1, a):
    """SIGTERM (what `docker stop` sends) saves latest.pt after the current step; a restart then ends up with
    the same weights as the uninterrupted run."""
    out = os.path.join(tmp, "interrupted")
    cmd = [sys.executable, TRAIN, "--device", "cpu", "--no-compile", *[str(x) for x in stage1], "--out", out]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    log = os.path.join(out, "log.jsonl")
    deadline = time.time() + 300
    while time.time() < deadline and p.poll() is None:
        if os.path.exists(log) and any('"step": 1,' in line for line in open(log)):
            break
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    stdout, stderr = p.communicate(timeout=300)
    assert p.returncode == 3, (p.returncode, stdout[-1000:], stderr[-2000:])
    stopped_at = load(os.path.join(out, "latest.pt"))["step"]
    assert 1 <= stopped_at < 4 and not os.path.exists(os.path.join(out, "final.pt"))
    run_train(*stage1, "--out", out)
    diff = max_weight_difference(a["model"], load(os.path.join(out, "final.pt"))["model"])
    assert diff < 1e-6, diff
    print(f"PASS (f) SIGTERM at step {stopped_at}: latest.pt saved, exit code 3, the restart matches "
          f"the uninterrupted run (max weight difference {diff:.1e})")


def test_non_finite(tmp):
    data = os.path.join(tmp, "tiny_inf")
    make_tiny_data(data, fill=np.inf)
    out = os.path.join(tmp, "non_finite")
    p = run_train("--stage", 1, "--data", data, "--test-train-rows", 30, "--test-val-rows", 10, "--out", out,
                  expect_ok=False)
    assert p.returncode != 0 and "STOPPED" in p.stderr and "loss" in p.stderr, p.stderr[-2000:]
    assert not os.path.exists(os.path.join(out, "latest.pt")) and not os.path.exists(os.path.join(out, "final.pt"))
    print("PASS (g) a non-finite loss stops the run with a clear error and saves nothing")


if __name__ == "__main__":
    t0 = time.time()
    test_learning_rates()
    test_stream()
    test_t7_weights()
    tmp = tempfile.mkdtemp(prefix="wx-test-train-")
    try:
        test_first_step(tmp)
        stage1, a = test_resume_and_stage2(tmp)
        test_interrupt(tmp, stage1, a)
        test_non_finite(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"ALL PASS ({time.time() - t0:.0f} s)")
