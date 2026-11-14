#!/usr/bin/env python3
"""Deploy wxrouter on Fuzzball: one script over the fuzzball CLI.

Run with no arguments for a menu, or name an action:

  python deploy/fb.py                    menu: pick an action
  python deploy/fb.py check              CLI found? logged in? API reachable? volume there?
  python deploy/fb.py setup [--only volume,llm,gfs]
                                         one-time volume setup jobs (repo, packages, model, llama.cpp,
                                         Nemotron 18.9 GB, a year of NOAA data); waits for each to finish
  python deploy/fb.py up [--replace]
                                         validate, start, wait until every service is running
  python deploy/fb.py status             the deployed workflow's stages
  python deploy/fb.py open [--port 8080] port-forward the Control Room to http://localhost:PORT
  python deploy/fb.py logs SERVICE [-f]  e.g. logs box-a -f
  python deploy/fb.py redeploy           stop, then up again (prep pulls the latest main)
  python deploy/fb.py down               stop the workflow (frees both GPUs)

Target: fuzzball/30-real-stack.fz, the real AFNO model + Nemotron on both boxes, the router and the
Control Room. Needs the volume setup first (`setup`). There is no mock target: what is deployed is
always live (mock boxes are for laptop development only: dev_stack.py).

Add --dry-run to any action to print the fuzzball commands instead of running them.
The CLI is found from $FUZZBALL, then PATH, then ~/bin/fuzzball. The deployed workflow's id is
remembered in .fb_deploy.json (git-ignored), one entry per target.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".fb_deploy.json"

TARGETS = {
    "real": {"file": "fuzzball/30-real-stack.fz", "name": "real-stack",
             "services": ["box-a", "box-b", "router", "ui"]},
}
SETUP_JOBS = {  # one-time jobs that fill the team volume, in the order to run them
    "volume": ("fuzzball/quick/05-setup-volume.fz", "setup-volume", "repo, Python packages, AFNO model, llama.cpp, today's NOAA analysis"),
    "llm": ("fuzzball/quick/04-fetch-llm.fz", "fetch-llm", "Nemotron 3.5 Lightning 30B Q4_0 (18.9 GB, resumable)"),
    "gfs": ("fuzzball/quick/06-gfs-year.fz", "gfs-year", "one year of NOAA GFS analyses for the fine-tune (1.7 GB)"),
}
DONE = ("STAGE_STATUS_FINISHED",)
BAD = ("STAGE_STATUS_FAILED", "STAGE_STATUS_CANCELED", "STAGE_STATUS_CANCELLED")


# ------------------------------------------------------------------ CLI plumbing
class Fuzzball:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.bin = self.find()

    @staticmethod
    def find() -> str | None:
        for cand in (os.environ.get("FUZZBALL"), shutil.which("fuzzball"), str(Path.home() / "bin" / "fuzzball")):
            if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
                return cand
        return None

    def run(self, *args: str, timeout: float = 60, check: bool = True) -> str:
        cmd = [self.bin or "fuzzball", *args]
        if self.dry_run:
            print("  $ " + " ".join(cmd))
            return ""
        if not self.bin:
            raise SystemExit("fuzzball CLI not found: set $FUZZBALL, put it on PATH, or install it to ~/bin/fuzzball")
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"fuzzball did not answer within {timeout:.0f} s: {' '.join(args)}")
        out = (p.stdout or "") + (p.stderr or "")
        if check and p.returncode != 0:
            raise SystemExit(explain(out) or f"fuzzball {' '.join(args)} failed (exit {p.returncode}):\n{out.strip()}")
        return out

    def stream(self, *args: str) -> int:
        cmd = [self.bin or "fuzzball", *args]
        if self.dry_run:
            print("  $ " + " ".join(cmd))
            return 0
        if not self.bin:
            raise SystemExit("fuzzball CLI not found")
        return subprocess.call(cmd, cwd=ROOT)


def explain(out: str) -> str | None:
    """Turn the CLI errors we have met into one actionable line."""
    if "connection refused" in out or "Error while dialing" in out:
        return ("The Fuzzball API refused the connection. Check the network path to the cluster (VPN/Wi-Fi), "
                "or whether the cluster is down. Teammates may still reach it from another network.\n  " + out.strip())
    if "objectcache" in out:
        return "The cluster's image cache is down (fuzzball-objectcache). Ask the cluster admins.\n  " + out.strip()
    if re.search(r"unauthenticated|token.*expired|login", out, re.I):
        return "Not logged in or the login expired: run  fuzzball context login\n  " + out.strip()
    return None


# ------------------------------------------------------------------ parsing CLI text output
def parse_start(out: str) -> str | None:
    """`workflow start` prints the new workflow's id (alone, or as 'Workflow "<id>" started.')."""
    ids = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", out)
    return ids[-1] if ids else None


def parse_describe(out: str) -> dict:
    """`workflow describe` → {"status": "STAGE_STATUS_...", "error": str, "stages": {name: (kind, status)}}."""
    status = re.search(r"^Status:\s+(\S+)", out, re.M)
    error = re.search(r"^Error:[ \t]*(.*)$", out, re.M)
    stages = {}
    for line in out.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 3 and cells[0] in ("Service", "Job", "Image", "Workflow"):
            stages[cells[2]] = (cells[0], cells[1])
    return {"status": status.group(1) if status else None,
            "error": (error.group(1).strip() if error else "") or "", "stages": stages}


def parse_list(out: str) -> list[dict]:
    """`workflow list` → [{"id", "status", "name"}], newest first as the CLI prints them."""
    rows = []
    for line in out.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 5 and re.fullmatch(r"[0-9a-f-]{36}", cells[0]):
            rows.append({"id": cells[0], "status": cells[3], "name": cells[4]})
    return rows


# ------------------------------------------------------------------ state
def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def current_id(args) -> str:
    wid = getattr(args, "id", None) or load_state().get(args.target, {}).get("id")
    if not wid:
        raise SystemExit(f"No deployed '{args.target}' workflow remembered. Run `up`, or pass --id <workflow id>.")
    return wid


# ------------------------------------------------------------------ actions
def act_check(fb: Fuzzball, args) -> None:
    print(f"CLI:      {fb.bin or 'NOT FOUND'}")
    if not fb.bin:
        raise SystemExit("Install the fuzzball CLI (download it from the cluster's web UI) or set $FUZZBALL.")
    print("version:  " + fb.run("version", timeout=15).strip().splitlines()[0])
    print("API:      ", end="", flush=True)
    rows = parse_list(fb.run("workflow", "list", timeout=20))
    print(f"reachable, {len(rows)} workflow(s) visible")
    running = [r for r in rows if r["status"] in ("Started", "Pending")]
    for r in running:
        print(f"          running: {r['name']} {r['id']} ({r['status']})")
    vols = fb.run("volume", "list", timeout=20, check=False)
    names = sorted({line.split("|")[0].strip() for line in vols.splitlines()[1:] if "|" in line})
    print("volumes:  " + (", ".join(names) or "(none listed)"))
    for t in TARGETS.values():
        print(f"file:     {t['file']}: {'ok' if (ROOT / t['file']).exists() else 'MISSING'}")


def wait_for(fb: Fuzzball, wid: str, services: list[str], timeout: float, until_finished: bool = False) -> bool:
    """Poll `workflow describe` until the services run (or the job finishes). Prints changes as they happen."""
    if fb.dry_run:
        print(f"  (would poll `workflow describe {wid}` until {'it finishes' if until_finished else 'services run: ' + ', '.join(services)})")
        return True
    seen: dict = {}
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        d = parse_describe(fb.run("workflow", "describe", wid, timeout=30, check=False))
        for name, (kind, st) in d["stages"].items():
            if seen.get(name) != st:
                print(f"  {time.strftime('%H:%M:%S')}  {kind:<8} {name:<34} {st}")
                seen[name] = st
        if d["status"] in BAD or any(st == "Errored" for _, st in d["stages"].values()):
            print(f"\nFAILED: {d['status']}" + (f"\n  {d['error']}" if d["error"] else ""))
            print("\nLast events:\n" + "\n".join(fb.run("workflow", "events", wid, check=False).splitlines()[-12:]))
            return False
        if until_finished and d["status"] in DONE:
            return True
        if not until_finished and d["status"] in DONE:
            print("\nThe workflow finished on its own: its services stopped. (A workflow ends when its jobs end.)")
            return False
        if not until_finished and services and all(d["stages"].get(s, ("", ""))[1] in ("Started", "Running") for s in services):
            return True
        time.sleep(5)
    print(f"\nTimed out after {timeout:.0f} s. It may still be starting: python deploy/fb.py status")
    return False


def act_up(fb: Fuzzball, args) -> None:
    t = TARGETS[args.target]
    for script in t.get("build", []):
        print(f"Building {t['file']} from this checkout ...")
        if not fb.dry_run:
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print(f"Validating {t['file']} ...")
    fb.run("workflow", "validate", t["file"])
    if not fb.dry_run:
        running = [r for r in parse_list(fb.run("workflow", "list", timeout=20))
                   if r["name"] == t["name"] and r["status"] in ("Started", "Pending")]
        if running and not args.replace:
            ids = ", ".join(r["id"] for r in running)
            raise SystemExit(f"'{t['name']}' is already running ({ids}). Use `redeploy`, or `up --replace`.")
        for r in running:
            print(f"Stopping the running '{t['name']}' {r['id']} ...")
            fb.run("workflow", "stop", r["id"])
    print(f"Starting {t['name']} ...")
    wid = parse_start(fb.run("workflow", "start", "-n", t["name"], t["file"])) or ("<id>" if fb.dry_run else None)
    if not wid:
        raise SystemExit("Started, but no workflow id was printed. See: fuzzball workflow list")
    if not fb.dry_run:
        state = load_state()
        state[args.target] = {"id": wid, "name": t["name"], "file": t["file"], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        save_state(state)
    print(f"Workflow {wid}. Waiting for: {', '.join(t['services'])}")
    ok = wait_for(fb, wid, t["services"], args.timeout)
    if ok:
        print(f"\nUp. Open the Control Room:  python deploy/fb.py open --target {args.target}")
        print(f"Stop it when done (it holds the GPUs):  python deploy/fb.py down --target {args.target}")
    else:
        raise SystemExit(1)


def act_status(fb: Fuzzball, args) -> None:
    wid = current_id(args)
    out = fb.run("workflow", "describe", wid)
    print(out if fb.dry_run else out.rstrip())


def act_open(fb: Fuzzball, args) -> None:
    wid = current_id(args)
    print(f"Control Room: http://localhost:{args.port}   (Ctrl-C to stop forwarding)")
    sys.exit(fb.stream("workflow", "port-forward", wid, "ui", f"{args.port}:8080", "-k", "30"))


def act_logs(fb: Fuzzball, args) -> None:
    wid = current_id(args)
    extra = ["-f"] if args.follow else ["-n", str(args.lines)]
    sys.exit(fb.stream("workflow", "log", wid, args.service, *extra))


def act_down(fb: Fuzzball, args) -> None:
    wid = current_id(args)
    print(f"Stopping {wid} ...")
    fb.run("workflow", "stop", wid)
    if not fb.dry_run:
        state = load_state()
        state.pop(args.target, None)
        save_state(state)
    print("Stopped. Both GPUs are free.")


def act_redeploy(fb: Fuzzball, args) -> None:
    args.replace = True
    act_up(fb, args)


def act_setup(fb: Fuzzball, args) -> None:
    keys = [k.strip() for k in (args.only or ",".join(SETUP_JOBS)).split(",") if k.strip()]
    for k in keys:
        if k not in SETUP_JOBS:
            raise SystemExit(f"unknown setup step {k!r}; choose from: {', '.join(SETUP_JOBS)}")
    for k in keys:
        file, name, what = SETUP_JOBS[k]
        print(f"\n[{k}] {what}\n  {file}")
        fb.run("workflow", "validate", file)
        wid = parse_start(fb.run("workflow", "start", "-n", name, file)) or "<id>"
        print(f"  workflow {wid}")
        if not wait_for(fb, wid, [], args.timeout, until_finished=True):
            raise SystemExit(f"setup step {k} did not finish; see: fuzzball workflow log {wid} <job>")
    print("\nVolume setup done.")


ACTIONS = {
    "check": (act_check, "check the CLI, login, API and volume"),
    "setup": (act_setup, "one-time volume setup (repo, model, LLM, NOAA data)"),
    "up": (act_up, "validate, start and wait for the stack"),
    "status": (act_status, "show the deployed workflow's stages"),
    "open": (act_open, "port-forward the Control Room to localhost"),
    "logs": (act_logs, "show a service's log"),
    "redeploy": (act_redeploy, "stop and start again (pulls latest main)"),
    "down": (act_down, "stop the workflow and free the GPUs"),
}


def menu() -> list[str]:
    print("wxrouter on Fuzzball\n")
    names = list(ACTIONS)
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n:<9} {ACTIONS[n][1]}")
    try:
        pick = input("\nPick a number (Enter to quit): ").strip()
        if not pick:
            raise SystemExit(0)
        action = names[int(pick) - 1]
        argv = [action]
        if action == "logs":
            argv.append(input("Service (box-a, box-b, router, ui) [router]: ").strip() or "router")
        return argv
    except (ValueError, IndexError):
        raise SystemExit("Not a number from the list.")
    except EOFError:
        raise SystemExit(0)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not any(a in ACTIONS for a in argv):  # no action named (maybe just --dry-run): show the menu
        argv = [a for a in argv if a == "--dry-run"] + menu()
    ap = argparse.ArgumentParser(prog="deploy/fb.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print fuzzball commands instead of running them")
    sub = ap.add_subparsers(dest="action", required=True)
    for name, (_, help_) in ACTIONS.items():
        p = sub.add_parser(name, help=help_)
        p.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
        if name in ("up", "redeploy", "status", "open", "logs", "down"):
            p.add_argument("--target", choices=TARGETS, default="real")
            p.add_argument("--id", help="act on this workflow id instead of the remembered one")
        if name in ("up", "redeploy", "setup"):
            p.add_argument("--timeout", type=float, default=1800, help="seconds to wait (default 1800)")
        if name == "up":
            p.add_argument("--replace", action="store_true", help="stop a running workflow of the same name first")
        if name == "setup":
            p.add_argument("--only", help="comma list of: " + ", ".join(SETUP_JOBS))
        if name == "open":
            p.add_argument("--port", type=int, default=8080)
        if name == "logs":
            p.add_argument("service")
            p.add_argument("-f", "--follow", action="store_true")
            p.add_argument("-n", "--lines", type=int, default=50)
    args = ap.parse_args(argv)
    fb = Fuzzball(dry_run=getattr(args, "dry_run", False))
    ACTIONS[args.action][0](fb, args)


if __name__ == "__main__":
    main()
