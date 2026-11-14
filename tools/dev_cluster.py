#!/usr/bin/env python3
"""Start the whole thing on one machine: two forecast mocks, two LLM mocks, the router.

Run it:
    python tools/dev_cluster.py up                # start, print the addresses, return
    python tools/dev_cluster.py up --foreground   # stay up until Ctrl-C, then stop everything
    python tools/dev_cluster.py status            # who is alive and healthy
    python tools/dev_cluster.py down              # stop exactly what "up" started

A second cluster on other ports (for tests), and how to stop only that one:
    python tools/dev_cluster.py up --ports-from 18100
    python tools/dev_cluster.py down --ports-from 18100

Standard ports: router 8100, forecast mocks 8101 and 8102, LLM mocks 8001 and 8002.
With --ports-from N: router N, forecast N+1 and N+2, LLM N+11 and N+12.

What it leaves in the repo root:
    .dev_cluster.json        the PIDs it started, one entry per cluster ("down" removes it)
    .dev_cluster_logs/       one log per piece (shifted clusters: a subfolder named after N)

"down" only signals a PID that is still running the same script on the same
port as when "up" started it. Anything else is left alone.

Standard library only.
Written at the hackathon (Nov 13-14, 2026).
"""

import argparse
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(REPO, ".dev_cluster.json")
LOG_DIR = os.path.join(REPO, ".dev_cluster_logs")
EXAMPLE_CONFIG = os.path.join(REPO, "router", "config.example.json")
HOST = "127.0.0.1"
WAIT_S = 20.0       # how long a piece may take to answer its health check
STOP_S = 5.0        # how long a piece may take to stop before it is killed


# ---------------------------------------------------------------- what to run

def ports_for(base=None):
    """The five ports. No base: the standard ones. A base N shifts them all."""
    if base is None:
        return {"router": 8100, "forecast-A": 8101, "forecast-B": 8102, "llm-A": 8001, "llm-B": 8002}
    return {"router": base, "forecast-A": base + 1, "forecast-B": base + 2, "llm-A": base + 11, "llm-B": base + 12}


def router_config(ports, decision_log="router-decisions.jsonl"):
    """The router's config file for these ports. With the standard ports it is the team's example config."""

    def url(name):
        return "http://%s:%d" % (HOST, ports[name])

    def heal(name):
        return {"http": {"url": url(name) + "/admin/fail", "method": "POST", "json": {"mode": "ok"}}}

    return {
        "port": ports["router"], "policy": "adaptive", "probe_s": 1.0, "fail_n": 2, "recover_s": 3.0,
        "connect_timeout_s": 2.0,
        "adaptive": {"hysteresis": 0.25, "dwell_s": 3.0, "cooldown_s": 10.0, "spill_inflight": 0},
        "score": {"busy_other_penalty": 1.5, "low_mem_gb": 8, "low_mem_penalty": 2.0, "default_latency_ms": 200},
        "heal_after_s": 10, "heal_backoff_s": [5, 15, 60], "heal_max_attempts": 5,
        "decision_log": decision_log,
        "pools": {
            "forecast": {"first_byte_timeout_s": 120, "backends": [
                {"name": "A", "url": url("forecast-A"), "probe": "forecast", "heal": heal("forecast-A")},
                {"name": "B", "url": url("forecast-B"), "probe": "forecast", "heal": heal("forecast-B")}]},
            "llm": {"first_byte_timeout_s": 600, "affinity": True, "affinity_break": 3.0, "backends": [
                {"name": "A", "url": url("llm-A"), "probe": "llamacpp"},
                {"name": "B", "url": url("llm-B"), "probe": "llamacpp",
                 "heal": {"cmd": ["echo", "restart B's LLM here"]}}]}}}


def points_at(config, ports):
    """True if a router config sends both pools to this cluster's mocks."""
    try:
        for pool, wanted in router_config(ports)["pools"].items():
            urls = [backend["url"].rstrip("/") for backend in config["pools"][pool]["backends"]]
            if urls != [backend["url"] for backend in wanted["backends"]]:
                return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def choose_config(base, ports):
    """The router's config file: (path, temp folder or None, a note to print or None).

    The standard cluster uses router/config.example.json, if that file points
    at the local mocks. Otherwise a config for these ports is written to a
    temp folder.
    """
    note = None
    if base is None:
        try:
            with open(EXAMPLE_CONFIG) as handle:
                if points_at(json.load(handle), ports):
                    return EXAMPLE_CONFIG, None, None
            note = "router/config.example.json does not point at the local mocks, so a config was written for them."
        except FileNotFoundError:
            note = "router/config.example.json does not exist yet, so a config was written."
        except (OSError, ValueError) as error:
            note = "router/config.example.json cannot be read (%s), so a config was written." % error
    tmp = tempfile.mkdtemp(prefix="dev_cluster_")
    path = os.path.join(tmp, "router-config.json")
    with open(path, "w") as handle:
        json.dump(router_config(ports, os.path.join(tmp, "router-decisions.jsonl")), handle, indent=1)
        handle.write("\n")
    return path, tmp, note


def plan(ports, config_path):
    """The five pieces, backends first and the router last: what to run and how to ask if it is well."""

    def piece(name, script, args, health):
        path = os.path.join(REPO, *script.split("/"))
        return {"name": name, "script": path, "port": ports[name], "url": "http://%s:%d" % (HOST, ports[name]),
                "health": health, "cmd": [sys.executable, path] + args + ["--port", str(ports[name])]}

    return [
        piece("forecast-A", "serve/forecast_server.py", ["--node", "A", "--mock", "--admin"], "/health"),
        piece("forecast-B", "serve/forecast_server.py", ["--node", "B", "--mock", "--admin"], "/health"),
        piece("llm-A", "serve/mock_llm.py", ["--node", "A", "--flavor", "llamacpp", "--admin"], "/health"),
        piece("llm-B", "serve/mock_llm.py", ["--node", "B", "--flavor", "llamacpp", "--admin"], "/health"),
        piece("router", "router/router.py", ["--config", config_path], "/router/health"),
    ]


def curl_lines(ports):
    """Six things worth trying, as (what it is, the command)."""
    router = "http://%s:%d" % (HOST, ports["router"])
    node_b = "http://%s:%d" % (HOST, ports["forecast-B"])
    post = " -H 'Content-Type: application/json' -d "
    point = '"point":{"lat":41.9602,"lon":-87.9316}'
    return [
        ("a forecast through the router",
         "curl -s %s/v1/forecast%s'{\"init\":\"latest\",\"lead_hours\":24,\"members\":1,%s}'" % (router, post, point)),
        ("an ensemble, split across both nodes",
         "curl -s %s/v1/ensemble%s'{\"init\":\"latest\",\"lead_hours\":48,\"members\":20,%s}'" % (router, post, point)),
        ("a chat completion",
         "curl -s %s/v1/chat/completions%s'{\"model\":\"mock-llm\",\"max_tokens\":32,"
         "\"messages\":[{\"role\":\"user\",\"content\":\"Will it snow tomorrow?\"}]}'" % (router, post)),
        ("the router's status",
         "curl -s %s/router/status" % router),
        ("switch the policy",
         "curl -s %s/router/policy%s'{\"policy\":\"round_robin\"}'" % (router, post)),
        ("break node B's forecast server ({\"mode\":\"ok\"} mends it)",
         "curl -s %s/admin/fail%s'{\"mode\":\"down\"}'" % (node_b, post)),
    ]


# ---------------------------------------------------------------- the state file

def read_state():
    """The clusters on record: {router port as text: {"started", "tmp", "pieces"}}."""
    try:
        with open(STATE_FILE) as handle:
            state = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise SystemExit("dev_cluster: cannot read %s (%s). Look at it, stop its PIDs by hand, then delete it."
                         % (STATE_FILE, error))
    if not isinstance(state, dict):
        raise SystemExit("dev_cluster: %s is not what this tool wrote. Look at it, then delete it." % STATE_FILE)
    return state


def set_cluster(key, cluster):
    """Write one cluster's entry to the state file (None removes it). No clusters left: remove the file.

    The file is read again just before, so other clusters' entries stay as
    they are, even if someone else ran "up" or "down" in the meantime.
    """
    state = read_state()
    state.pop(key, None)
    if cluster is not None:
        state[key] = cluster
    if not state:
        try:
            os.remove(STATE_FILE)
        except FileNotFoundError:
            pass
        return
    scratch = "%s.%d.tmp" % (STATE_FILE, os.getpid())
    with open(scratch, "w") as handle:
        json.dump(state, handle, indent=1)
        handle.write("\n")
    os.replace(scratch, STATE_FILE)


def pieces_of(cluster):
    """The piece entries of one cluster on record. Anything this tool did not write is dropped."""
    pieces = cluster.get("pieces") if isinstance(cluster, dict) else None
    return [piece for piece in pieces if isinstance(piece, dict)] if isinstance(pieces, list) else []


def record(piece):
    """What the state file keeps about one started piece."""
    return dict((key, piece[key]) for key in ("name", "pid", "port", "url", "health", "script", "log"))


# ---------------------------------------------------------------- processes

def command_line(pid):
    """The command line of a running process, or "" if it is gone or cannot be read."""
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as handle:        # Linux
            return handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        pass
    try:                                                            # macOS, and Linux without /proc
        done = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.decode("utf-8", "replace").strip()


def why_not_ours(piece):
    """None if the PID still runs this piece: same script, same port. Else the reason to leave it alone."""
    pid, script, port = piece.get("pid"), piece.get("script"), piece.get("port")
    if type(pid) is not int or pid <= 1 or not isinstance(script, str) or type(port) is not int:
        return "its entry in the state file is not usable"
    if not script.startswith(REPO + os.sep):
        return "its script is not in this repo"
    line = command_line(pid)
    if not line:
        return "it is not running"
    words = line.replace("=", " ").split()
    same_port = any(words[i] == "--port" and words[i + 1] == str(port) for i in range(len(words) - 1))
    if script not in line or not same_port:
        return "pid %d now belongs to something else" % pid
    return None


def still_running(piece):
    proc = piece.get("proc")
    if proc is not None:                    # our own child: this also reaps it
        return proc.poll() is None
    return why_not_ours(piece) is None


def stop_pieces(pieces):
    """SIGTERM every piece that is still ours, wait up to 5 seconds, then SIGKILL the stubborn ones."""
    stopping = []
    for piece in pieces:
        reason = why_not_ours(piece)
        if reason:
            print("  %-11s left alone: %s" % (piece.get("name"), reason), flush=True)
            continue
        try:
            os.kill(piece["pid"], signal.SIGTERM)
        except OSError as error:
            print("  %-11s could not signal pid %d: %s" % (piece["name"], piece["pid"], error), flush=True)
            continue
        stopping.append(piece)
    deadline = time.monotonic() + STOP_S
    left = list(stopping)
    while left and time.monotonic() < deadline:
        time.sleep(0.1)
        left = [piece for piece in left if still_running(piece)]
    for piece in left:
        if why_not_ours(piece) is None:     # check again: never kill a stranger
            try:
                os.kill(piece["pid"], signal.SIGKILL)
            except OSError:
                pass
    for piece in stopping:
        how = "killed (did not stop in %d s)" % STOP_S if piece in left else "stopped"
        print("  %-11s pid %-7d %s" % (piece["name"], piece["pid"], how), flush=True)


def port_taken(port):
    """True if something already listens on the port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex((HOST, port)) == 0
    finally:
        probe.close()


def healthy(url, path):
    """True if GET url+path answers 200 within a second."""
    host, port = url.split("//", 1)[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=1.0)
    try:
        conn.request("GET", path, headers={"Connection": "close"})
        response = conn.getresponse()
        response.read()             # read it all, so closing does not reset the connection
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def launch(piece, log_dir):
    """Start one piece in its own session, output to its log. Sets "proc" and "pid", or "state" saying why not."""
    if not os.path.exists(piece["script"]):
        piece["state"] = "NOT STARTED: %s does not exist yet" % os.path.relpath(piece["script"], REPO)
        return
    if port_taken(piece["port"]):
        piece["state"] = "NOT STARTED: port %d is taken (another cluster? try status or down)" % piece["port"]
        return
    os.makedirs(log_dir, exist_ok=True)
    piece["log"] = os.path.join(log_dir, piece["name"] + ".log")
    with open(piece["log"], "wb") as log:
        piece["proc"] = subprocess.Popen(
            piece["cmd"], cwd=REPO, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)     # own session: it outlives us
    piece["pid"] = piece["proc"].pid


def wait_healthy(pieces):
    """Poll until each piece answers its health check, has died, or 20 seconds are over. Sets "state"."""
    deadline = time.monotonic() + WAIT_S
    waiting = list(pieces)
    while waiting:
        for piece in list(waiting):
            code = piece["proc"].poll()
            if code is not None:
                piece["state"] = "FAILED: exited with code %d" % code
                waiting.remove(piece)
            elif healthy(piece["url"], piece["health"]):
                piece["state"] = "healthy"
                waiting.remove(piece)
        if not waiting or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    for piece in waiting:
        piece["state"] = "NOT HEALTHY after %d s (still running)" % WAIT_S


def log_tail(path, lines=5):
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8", "replace").splitlines()[-lines:]
    except OSError:
        return []


def remove_tmp(folder):
    """Remove a temp folder that "up" made. Anything with another name is left alone."""
    if folder and os.path.basename(folder).startswith("dev_cluster_") and os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------- up, down, status

def up(base, foreground):
    ports = ports_for(base)
    key = str(ports["router"])
    old = read_state().get(key)
    if old is not None:
        alive = [str(piece.get("name")) for piece in pieces_of(old) if why_not_ours(piece) is None]
        if alive:
            print("This cluster is already up (%s). Run \"down\" first." % ", ".join(alive), flush=True)
            return 1
        if isinstance(old, dict):                   # a stale entry: nothing of it runs any more
            remove_tmp(old.get("tmp"))
        set_cluster(key, None)

    config_path, tmp, config_note = choose_config(base, ports)
    log_dir = LOG_DIR if base is None else os.path.join(LOG_DIR, str(base))
    pieces = plan(ports, config_path)
    started = []

    began = time.strftime("%Y-%m-%d %H:%M:%S")

    def save():
        running = [record(piece) for piece in started if piece["proc"].poll() is None]
        set_cluster(key, {"started": began, "tmp": tmp, "pieces": running} if running else None)

    def stop_everything():
        stop_pieces(started)
        set_cluster(key, None)
        remove_tmp(tmp)

    old_handler = signal.signal(signal.SIGTERM, signal.default_int_handler)   # treat SIGTERM like Ctrl-C
    try:
        # Backends first, so the router finds them well on its first look.
        for group in (pieces[:-1], pieces[-1:]):
            for piece in group:
                launch(piece, log_dir)
                if "proc" in piece:
                    started.append(piece)
                    save()
            wait_healthy([piece for piece in group if "proc" in piece])
        save()

        print("%-11s %-24s %-7s %s" % ("piece", "address", "pid", "state"))
        for piece in pieces:
            print("%-11s %-24s %-7s %s" % (piece["name"], piece["url"], piece.get("pid", "-"), piece["state"]))
            if "proc" in piece and piece["state"] != "healthy":
                for line in log_tail(piece["log"]):
                    print("    | " + line)
        shown = os.path.relpath(log_dir, REPO) if log_dir.startswith(REPO + os.sep) else log_dir
        print("logs: %s" % os.path.join(shown, "<piece>.log"))
        print("router config: %s" % config_path)
        if config_note:
            print("  (%s)" % config_note)
        print("")
        print("Try:")
        for what, command in curl_lines(ports):
            print("  # " + what)
            print("  " + command)
        print("")
        good = [piece for piece in pieces if piece["state"] == "healthy"]
        print("%d of %d pieces are up." % (len(good), len(pieces)), flush=True)

        if not foreground:
            if started:
                print("Stop them with: python tools/dev_cluster.py down%s"
                      % ("" if base is None else " --ports-from %d" % base), flush=True)
            else:
                remove_tmp(tmp)
            return 0 if len(good) == len(pieces) else 1

        print("Staying up. Ctrl-C stops everything.", flush=True)
        while True:
            time.sleep(0.5)
            for piece in started:
                if piece["proc"].poll() is not None and not piece.get("mourned"):
                    piece["mourned"] = True
                    print("%s exited with code %d. See %s" % (piece["name"], piece["proc"].returncode, piece["log"]),
                          flush=True)
    except KeyboardInterrupt:
        print("\nStopping:", flush=True)
        stop_everything()
        return 0 if foreground else 130
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def down(base):
    state = read_state()
    if not state:
        print("Nothing to stop: there is no %s." % os.path.basename(STATE_FILE), flush=True)
        return 0
    keys = sorted(state) if base is None else [str(base)]
    for key in keys:
        if key not in state:
            print("No cluster with its router on port %s is on record." % key, flush=True)
            continue
        cluster = state[key]
        print("Stopping the cluster with its router on port %s:" % key, flush=True)
        stop_pieces(pieces_of(cluster))
        if isinstance(cluster, dict):
            remove_tmp(cluster.get("tmp"))
        set_cluster(key, None)
    return 0


def status():
    state = read_state()
    if not state:
        print("Nothing is up: there is no %s." % os.path.basename(STATE_FILE), flush=True)
        return 1
    all_well = True
    for key in sorted(state):
        cluster = state[key]
        began = cluster.get("started") if isinstance(cluster, dict) else None
        print("Cluster with its router on port %s, started %s:" % (key, began))
        print("  %-11s %-24s %-7s %s" % ("piece", "address", "pid", "state"))
        for piece in pieces_of(cluster):
            reason = why_not_ours(piece)
            if reason:
                word = "DEAD (%s)" % reason
            elif healthy("http://%s:%d" % (HOST, piece["port"]), str(piece.get("health") or "/health")):
                word = "alive, healthy"
            else:
                word = "alive, NOT HEALTHY"
            all_well = all_well and word == "alive, healthy"
            print("  %-11s %-24s %-7s %s" % (piece.get("name"), piece.get("url"), piece.get("pid"), word))
    sys.stdout.flush()
    return 0 if all_well else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Start, check or stop the local cluster: two forecast mocks, two LLM mocks, the router.",
        epilog="Examples: python tools/dev_cluster.py up   |   python tools/dev_cluster.py down")
    parser.add_argument("command", choices=("up", "down", "status"),
                        help="up: start everything. down: stop what up started. status: who is alive and healthy")
    parser.add_argument("--foreground", action="store_true",
                        help="with up: stay until Ctrl-C, then stop everything")
    parser.add_argument("--ports-from", type=int, metavar="N",
                        help="use ports N (router), N+1, N+2 (forecast), N+11, N+12 (LLM). "
                             "With down: stop only the cluster whose router is on port N")
    args = parser.parse_args(argv)
    if args.ports_from is not None and not 1024 <= args.ports_from <= 65523:
        parser.error("--ports-from must be 1024 to 65523")
    if args.command == "up":
        return up(args.ports_from, args.foreground)
    if args.command == "down":
        return down(args.ports_from)
    return status()


if __name__ == "__main__":
    sys.exit(main())
