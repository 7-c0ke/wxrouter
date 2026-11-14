"""Train the weather model from the recipe card "N1R-75 with T7". Written at the hackathon (Fri Nov 13, 2026).

  stage 1: learn one 6-hour step.         18,828 optimizer steps, peak learning rate 1e-3.
  stage 2: learn four fed-back steps.      1,564 optimizer steps, peak 1e-4, from stage 1's final weights.

Both stages: AFNO (19.1 M parameters), bf16 autocast + torch.compile on CUDA, AdamW, gradient clip 1.0,
batch 32 as two micro-batches of 16, and the T7 loss (MSE with Z500, T850 and T2M weighted 3x).

    python weather/train.py --stage 1 --out /out/stage1
    python weather/train.py --stage 2 --out /out/stage2 --init /out/stage1/final.pt

Files in --out:
    log.jsonl   one JSON object per line: a start event, one line per optimizer step, checkpoint events, an end event
    latest.pt   everything needed to carry on; a restarted process resumes from it and retraces the same steps
    final.pt    the stage's result ('complete' is False if --max-steps stopped it early)

After a crash log.jsonl can list a step twice (the steps after the last checkpoint are run again); the later
line is the one that counts.
"""
import argparse
import contextlib
import datetime
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time

import numpy as np
import torch

import wx_common as wx

GLOBAL_BATCH = 32            # samples per optimizer step
MICRO_BATCH = 16             # samples per forward/backward pass
ACCUM = GLOBAL_BATCH // MICRO_BATCH
CLIP_NORM = 1.0
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8

STAGES = {
    1: dict(rollout=1, peak_lr=1e-3, warmup_steps=500, total_steps=18828, decay_steps=2824, ckpt_every=1000),
    2: dict(rollout=4, peak_lr=1e-4, warmup_steps=100, total_steps=1564, decay_steps=1464, ckpt_every=250),
}

# A resumed run must agree with the checkpoint on these, or it would not retrace the uninterrupted run.
MUST_MATCH_ON_RESUME = ["stage", "seed", "train_rows", "total_steps", "decay_steps"]


# ---------------------------------------------------------------- learning rate

def lr_stage1(s, total=18828, decay=2824):
    """Rate for 0-based optimizer step s: 500-step warm-up to 1e-3, constant, then linear to 0 over `decay` steps."""
    hold_end = total - decay                     # 16004
    if s < hold_end:
        return 1e-3 * min(1, (s + 1) / 500)
    return 1e-3 * (decay - (s - hold_end)) / decay


def lr_stage2(s, total=1564):
    """Rate for 0-based optimizer step s: 100-step warm-up to 1e-4, then linear decay that would reach 0 at `total`."""
    if s < 100:
        return 1e-4 * (s + 1) / 100
    return 1e-4 * (total - s) / (total - 100)


def learning_rate(recipe, s):
    if recipe["stage"] == 1:
        return lr_stage1(s, recipe["total_steps"], recipe["decay_steps"])
    return lr_stage2(s, recipe["total_steps"])


# ---------------------------------------------------------------- data order

class StartStream:
    """The endless, seeded order of start rows. Epoch e is default_rng([seed, e]).permutation(n), and the
    stream runs straight on from one epoch into the next, so a batch can straddle the boundary."""

    def __init__(self, n, seed, epoch=0, pos=0):
        self.n, self.seed = n, seed
        self.epoch, self.pos = epoch, pos
        self.order = self._order(epoch)

    def _order(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(self.n)

    def take(self, count):
        pieces = []
        while count > 0:
            if self.pos == self.n:
                self.epoch, self.pos = self.epoch + 1, 0
                self.order = self._order(self.epoch)
            piece = self.order[self.pos:self.pos + count]
            pieces.append(piece)
            self.pos += len(piece)
            count -= len(piece)
        return np.concatenate(pieces)

    def state(self):
        return (self.epoch, self.pos)


class Prefetcher:
    """A background thread that reads the next few batches from the memmap, in stream order.

    Each item is (block, stream_state): block is a float16 tensor (32, rollout + 1, 20, 120, 240) holding, for
    every start row t, rows t .. t + rollout; stream_state is where the stream stands after this batch, which
    is what a checkpoint must record (the thread itself has already read further ahead).
    """

    def __init__(self, data, stream, rollout, n_batches, pin, depth=3):
        self.data, self.stream, self.rollout, self.n_batches, self.pin = data, stream, rollout, n_batches, pin
        self.items = queue.Queue(maxsize=depth)
        self.stopping = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._work, name="prefetch", daemon=True)
        self.thread.start()

    def _work(self):
        try:
            for _ in range(self.n_batches):
                starts = self.stream.take(GLOBAL_BATCH)
                block = torch.from_numpy(np.stack([self.data[t:t + self.rollout + 1] for t in starts]))
                if self.pin:
                    block = block.pin_memory()
                item = (block, self.stream.state())
                while not self.stopping.is_set():
                    try:
                        self.items.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        pass
                if self.stopping.is_set():
                    return
        except BaseException as e:               # hand any read error to the training loop instead of hanging it
            self.error = e

    def next(self):
        while True:
            try:
                return self.items.get(timeout=0.2)
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError(f"the prefetch thread failed: {self.error!r}") from self.error
                if not self.thread.is_alive() and self.items.empty():
                    raise RuntimeError("the prefetch thread ended before the training loop did")

    def close(self):
        self.stopping.set()
        self.thread.join()


# ---------------------------------------------------------------- loss

def autocast(amp):
    return torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()


def weighted_mse(pred, target, weights=None):
    """Mean over batch, channels and grid of w_c * (pred - target)^2, in float32. weights: (1, 20, 1, 1) or None."""
    sq = (pred - target) ** 2
    if weights is not None:
        sq = weights * sq
    return sq.mean()


def rollout_loss(model, x, targets, weights, amp):
    """Mean over the fed-back 6-hour steps of the weighted MSE. targets: (batch, steps, 20, 120, 240).

    Stage 1 has one step, stage 2 four. Each output is cast to float32 and fed back without detaching, so
    gradients flow through every step.
    """
    n_steps = targets.shape[1]
    cur = x
    total = 0.0
    for k in range(n_steps):
        with autocast(amp):
            out = model(cur)
        cur = out.float()
        total = total + weighted_mse(cur, targets[:, k], weights) / n_steps
    return total


@torch.no_grad()
def validate(model, data, first, count, rollout, device, amp, batch=32):
    """Unweighted rollout MSE over every start in rows [first, first + count) that has its targets in there too.
    With rollout=1 this is the one-step MSE over count - 1 consecutive pairs (1,459 for 2018)."""
    was_training = model.training
    model.eval()
    n_starts = count - rollout
    total = 0.0
    for a in range(0, n_starts, batch):
        b = min(a + batch, n_starts)
        # The starts are consecutive, so one contiguous read holds the inputs and all their targets.
        rows = torch.from_numpy(np.ascontiguousarray(data[first + a:first + b + rollout])).to(device).float()
        x = rows[:b - a]
        targets = torch.stack([rows[k + 1:k + 1 + b - a] for k in range(rollout)], dim=1)
        total += rollout_loss(model, x, targets, None, amp).item() * (b - a)
    model.train(was_training)
    return total / n_starts


# ---------------------------------------------------------------- files

def save_atomically(obj, path):
    """Write to .tmp and rename, so a crash mid-write never leaves a broken checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def make_recipe(args):
    s = STAGES[args.stage]
    total = args.test_total_steps or s["total_steps"]
    if args.stage == 1:
        decay = args.test_decay_steps or s["decay_steps"]
    else:
        decay = total - s["warmup_steps"]
    train_rows = args.test_train_rows or wx.N_TRAIN_TIMES
    return {
        "card": "N1R-75 with T7", "stage": args.stage, "seed": args.seed,
        "model": "physicsnemo AFNO inp_shape=[120,240] in=20 out=20 patch=[4,4] num_blocks=8 embed_dim=512 depth=8",
        "global_batch": GLOBAL_BATCH, "micro_batch": MICRO_BATCH, "accum": ACCUM,
        "rollout": s["rollout"], "peak_lr": s["peak_lr"], "warmup_steps": s["warmup_steps"],
        "total_steps": total, "decay_steps": decay,
        "betas": list(BETAS), "weight_decay": WEIGHT_DECAY, "eps": ADAM_EPS, "clip_norm": CLIP_NORM,
        "channel_weights": [float(w) for w in wx.t7_channel_weights()],
        "train_rows": train_rows,
        "n_starts": train_rows - s["rollout"],   # start rows whose targets are still training rows
        "init": args.init if args.stage == 2 else None,
    }


# ---------------------------------------------------------------- the run

def run(args):
    recipe = make_recipe(args)
    stage, seed, rollout = recipe["stage"], recipe["seed"], recipe["rollout"]
    total_steps = recipe["total_steps"]
    stop_at = min(total_steps, args.max_steps) if args.max_steps else total_steps
    ckpt_every = args.ckpt_every or STAGES[stage]["ckpt_every"]

    os.makedirs(args.out, exist_ok=True)
    latest_path = os.path.join(args.out, "latest.pt")
    final_path = os.path.join(args.out, "final.pt")

    device = wx.pick_device(args.device)
    on_cuda = device.type == "cuda"
    amp = on_cuda                                # bf16 autocast on CUDA; plain float32 on CPU and MPS
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data = wx.open_packed("main", args.data)
    val_first = wx.VAL_FIRST if args.test_train_rows is None else recipe["train_rows"]
    val_rows = args.test_val_rows or wx.N_VAL_TIMES
    if data.shape[1:] != (wx.N_CHANNELS, wx.N_LAT, wx.N_LON) or data.dtype != np.float16:
        raise SystemExit(f"unexpected data: shape {data.shape}, dtype {data.dtype}")
    if len(data) < val_first + val_rows:
        raise SystemExit(f"the data file has {len(data)} rows; need {val_first + val_rows}")

    # The starting weights depend on these three seeds and on nothing drawing from torch's RNG before the model
    # is built, so the import that build_afno() does is done first.
    import physicsnemo.models.afno  # noqa: F401
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    raw = wx.build_afno()
    raw = raw.to(device)
    n_params = sum(p.numel() for p in raw.parameters())

    optimizer = torch.optim.AdamW(raw.parameters(), lr=0.0, betas=BETAS, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS,
                                  **({"fused": True} if on_cuda else {}))

    step, epoch, pos = 0, 0, 0
    timed_steps, timed_seconds = 0, 0.0          # for steps/s; carried across restarts in the checkpoint
    resumed = os.path.exists(latest_path)
    if resumed:
        ck = torch.load(latest_path, map_location="cpu", weights_only=True)
        for key in MUST_MATCH_ON_RESUME:
            if ck["recipe"][key] != recipe[key]:
                raise SystemExit(f"{latest_path} was written with {key}={ck['recipe'][key]!r}, but this run has "
                                 f"{key}={recipe[key]!r}. Use a different --out.")
        raw.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, (epoch, pos) = ck["step"], ck["stream"]
        timed_steps, timed_seconds = ck["timed_steps"], ck["timed_seconds"]
        recipe["init"] = ck["recipe"]["init"]    # where the weights first came from, not this restart's flags
        torch.set_rng_state(ck["torch_rng"])
        if on_cuda and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    elif stage == 2:
        if not args.init:
            raise SystemExit("stage 2 starts from stage 1's weights: pass --init PATH (stage 1's final.pt)")
        init = torch.load(args.init, map_location="cpu", weights_only=True)
        raw.load_state_dict(init["model"])
        recipe["init_step"] = init.get("step")

    if resumed and step >= total_steps and os.path.exists(final_path):
        print(f"stage {stage} is already finished: {final_path} exists (step {step})", flush=True)
        return 0
    if step > stop_at:
        # Carrying on would write a final.pt marked incomplete into a run that is further along than that.
        raise SystemExit(f"{latest_path} is already at step {step}, beyond --max-steps {args.max_steps}. "
                         f"Use a different --out for a short test run.")

    model = raw
    if on_cuda and not args.no_compile:
        model = torch.compile(raw)               # `raw` stays the handle for state_dict and clipping
    model.train()

    weights = torch.tensor(wx.t7_channel_weights(), dtype=torch.float32, device=device).view(1, -1, 1, 1)

    log_file = open(os.path.join(args.out, "log.jsonl"), "a")

    def log(obj):
        log_file.write(json.dumps(obj) + "\n")
        log_file.flush()

    log({"event": "start", "time": now(), "stage": stage, "torch": torch.__version__, "device": str(device),
         "params": n_params, "seed": seed, "steps": total_steps, "stop_at": stop_at, "resumed_from": step,
         "compiled": model is not raw, "bf16": amp, "recipe": recipe})
    print(f"stage {stage}: {n_params:,} parameters on {device}, torch {torch.__version__}, seed {seed}, "
          f"steps {step} -> {stop_at} of {total_steps}" + (" (resumed)" if resumed else ""), flush=True)

    def save_latest():
        save_atomically({
            "model": raw.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "stream": (epoch, pos),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if on_cuda else None,
            "timed_steps": timed_steps, "timed_seconds": timed_seconds,
            "seed": seed, "stage": stage, "recipe": recipe,
        }, latest_path)
        log({"event": "checkpoint", "time": now(), "step": step, "file": "latest.pt"})

    # Ctrl-C or `docker stop` finishes the current step, saves latest.pt and exits, instead of losing the work
    # since the last checkpoint.
    interrupted = []

    def on_signal(signum, frame):
        if interrupted:                          # asked twice: stop right now
            raise KeyboardInterrupt
        interrupted.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)

    prefetch = Prefetcher(data, StartStream(recipe["n_starts"], seed, epoch, pos), rollout,
                          n_batches=stop_at - step, pin=on_cuda)
    first_of_process = True
    saved_step = step
    try:
        while step < stop_at and not interrupted:
            t0 = time.perf_counter()
            block, stream_after = prefetch.next()
            block = block.to(device, non_blocking=True)

            lr = learning_rate(recipe, step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for k in range(ACCUM):
                chunk = block[k * MICRO_BATCH:(k + 1) * MICRO_BATCH].float()
                loss = rollout_loss(model, chunk[:, 0], chunk[:, 1:], weights, amp) / ACCUM
                loss.backward()
                loss_sum = loss_sum + loss.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), CLIP_NORM).item()   # the norm before clipping
            loss_value = loss_sum.item()

            if not (math.isfinite(loss_value) and math.isfinite(grad_norm)):
                # Checked before optimizer.step(), so latest.pt on disk is still the last good checkpoint.
                log({"event": "error", "time": now(), "step": step + 1, "loss": repr(loss_value),
                     "grad_norm": repr(grad_norm)})
                raise SystemExit(f"STOPPED: step {step + 1} of stage {stage} gave loss {loss_value} and gradient "
                                 f"norm {grad_norm}. Nothing was saved for this step; latest.pt is the last "
                                 f"checkpoint before it.")

            optimizer.step()
            step += 1
            epoch, pos = stream_after
            dt = time.perf_counter() - t0
            if not first_of_process:             # a process's first step includes compilation
                timed_steps += 1
                timed_seconds += dt
            first_of_process = False

            log({"step": step, "loss": loss_value, "lr": lr, "grad_norm": grad_norm, "dt": round(dt, 4)})
            if step <= 3 or step % 100 == 0 or step == stop_at:
                speed = f"{dt:.1f} s (first step of this process)"
                if timed_seconds > 0:
                    rate = timed_steps / timed_seconds
                    speed = f"{rate:.3f} steps/s  {(stop_at - step) / rate / 60:.0f} min left"
                print(f"step {step:6d}/{total_steps}  loss {loss_value:.6f}  grad {grad_norm:.4f}  lr {lr:.3e}  "
                      f"{speed}", flush=True)

            if step % ckpt_every == 0:
                save_latest()
                saved_step = step
    finally:
        prefetch.close()

    if step != saved_step:
        save_latest()                            # the checkpoint "at exit"

    steps_per_s = timed_steps / timed_seconds if timed_seconds > 0 else None
    if interrupted:
        log({"event": "end", "time": now(), "step": step, "complete": False, "interrupted": True,
             "steps_per_s": steps_per_s, "val": None})
        log_file.close()
        print(f"interrupted at step {step}; latest.pt saved. Run the same command again to carry on.", flush=True)
        return 3

    complete = step >= total_steps
    val = None
    if complete:                                 # no validation when --max-steps stopped the run early
        t0 = time.perf_counter()
        val = {"mse_1step": validate(raw, data, val_first, val_rows, 1, device, amp)}
        if stage == 2:
            val["mse_4step_rollout"] = validate(raw, data, val_first, val_rows, 4, device, amp)
        val["seconds"] = round(time.perf_counter() - t0, 1)
        print(f"validation (unweighted, rows {val_first}..{val_first + val_rows - 1}): {val}", flush=True)

    save_atomically({"model": raw.state_dict(), "step": step, "seed": seed, "recipe": recipe, "stage": stage,
                     "complete": complete, "val": val, "steps_per_s": steps_per_s}, final_path)
    log({"event": "end", "time": now(), "step": step, "complete": complete, "steps_per_s": steps_per_s, "val": val})
    rate_text = f"{steps_per_s:.3f} steps/s" if steps_per_s else "no timed steps"
    print(f"stage {stage} {'finished' if complete else 'stopped early by --max-steps'} at step {step} "
          f"({rate_text}); wrote {final_path}", flush=True)
    log_file.close()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the weather model (stage 1: one step; stage 2: four steps).")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--out", required=True, help="folder for log.jsonl, latest.pt and final.pt")
    p.add_argument("--data", default=None, help="folder with packed_1979_2019.npy (default: $WX_DATA or /data)")
    p.add_argument("--init", default=None, help="stage 2: stage 1's final.pt")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop when the step count reaches N (smoke tests); final.pt is then marked incomplete")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=None, help="default: 1000 in stage 1, 250 in stage 2")
    # Test-only: shrink the problem so the tests can run on a tiny synthetic file.
    p.add_argument("--test-train-rows", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-val-rows", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-total-steps", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-decay-steps", type=int, default=None, help=argparse.SUPPRESS)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
