"""The node stack: a small supervisor that keeps one node's servers alive.

It runs inside the node's one Fuzzball service. Fuzzball gives a node's GPU to one job at a time, so the LLM
server, the forecast server and any fine-tune run must live inside that one service. A Fuzzball service that
exits fails its whole workflow, so this process does not exit on its own. When a server dies, it starts it
again: this is the "self-healing node". Standard library only. Linux and macOS (it uses process groups).

    python serve/node_stack.py --name A --config stack.json --admin --log-dir /out/logs
    python serve/node_stack.py --name A --admin \
        --serve forecast="python serve/forecast_server.py --node A --host 0.0.0.0 --port 8101" \
        --serve llm="./llama-server -m model.gguf --host 0.0.0.0 --port 8000" \
        --job finetune="python weather/finetune.py --steps 500"

The config file is JSON and says the same thing. "cwd" (relative to where this supervisor was started) and
"env" (added to this supervisor's own environment) are optional:

    {"servers": {"forecast": {"cmd": ["python", "serve/forecast_server.py", "--node", "A"],
                              "cwd": ".", "env": {"WX_DATA": "/vol/bundle"}},
                 "llm": {"cmd": ["./llama-server", "-m", "model.gguf", "--port", "8000"]}},
     "jobs": {"finetune": {"cmd": ["python", "weather/finetune.py"]}}}

What it does:
  - It starts every server as a child process in its own process group. No shell is ever used (a command that
    needs one names it: ["bash", "-c", "..."]). Each line a child prints, on stdout or stderr, is copied to this
    process's output as "[forecast] ...", and to DIR/forecast.log with --log-dir DIR. Children are started with
    PYTHONUNBUFFERED=1, so the lines of a Python child come at once.
  - When a server exits, for any reason, it is started again after a wait of 1, 2, 5, 10, then 30 seconds. The
    wait goes back to 1 second once the server has stayed up for 60 seconds. A command that cannot be started at
    all (a wrong path, say) is tried again by the same rule; /status shows why under "error".
  - A server is its whole process group: when its main process exits, what it left behind is killed. So a
    server command must stay in the foreground.
  - A job (a fine-tune run, say) is a command that runs once, when asked over HTTP. It is never restarted.
  - On SIGTERM or Ctrl-C it stops the children (SIGTERM to each process group, SIGKILL after 10 seconds) and
    exits with code 0. A hang-up (the terminal closes) does the same, unless it was started under nohup. A wrong
    command line, a bad config file or a port in use ends it at the start, with code 2.

HTTP (default port 8109; the first line printed is "listening on http://HOST:PORT"; --port 0 picks a free port):
    GET  /health                 {"status": "ok", "name": "A"}
    GET  /status                 every server: state (running | backoff | stopped), pid, restarts, last_exit,
                                 up_s, next_start_in_s, error. Every job: state (idle | running | done | failed),
                                 pid, started, seconds, exit, error, last_lines (the last 20 lines it printed).
                                 An exit code below zero is a signal: -9 is SIGKILL, -15 is SIGTERM.
  Only with --admin (for the live demo and for the router's self-healing action):
    POST /servers/<name>/kill    SIGKILL to its process group, as a crash would; it comes back by the usual rule
    POST /servers/<name>/stop    stop it and keep it down
    POST /servers/<name>/start   start it now (this also ends a wait early)
    POST /jobs/<name>/start      run the job once; 409 while it runs. Optional body: {"args": ["--steps", "200"]}
    POST /jobs/<name>/stop       SIGTERM to its process group, SIGKILL after 30 seconds
  Only names from the config are accepted (404 otherwise). A request can add arguments to a job's command; it
  can never supply a command. Without --admin these calls answer 403.

    curl -s http://nodeA:8109/status
    curl -s -X POST http://nodeA:8109/servers/forecast/kill
    curl -s -X POST http://nodeA:8109/jobs/finetune/start -d '{"args": ["--steps", "200"]}'

Written at the hackathon (Nov 13-14, 2026).
"""
import argparse
import collections
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)     # the wait before each new start of a server that keeps dying
STABLE_S = 60.0         # a server that stayed up this long is well again: its next wait is the first one
GRACE_S = 10.0          # stopping a server, or every child at the end: SIGTERM, then SIGKILL after this long
JOB_GRACE_S = 30.0      # stopping a job over HTTP: SIGTERM, then SIGKILL after this long
KILL_WAIT_S = 5.0       # how long to wait for a process to be gone after SIGKILL
START_WAIT_S = 2.0      # how long POST /servers/<name>/start waits to see the start happen
READER_WAIT_S = 2.0     # how long to wait for the last lines of a process that has exited
OUT_WAIT_S = 2.0        # how long a line waits for its turn on stdout before it is dropped
LAST_LINES = 20         # lines of a job's output kept for /status
MAX_LINE = 4000         # a longer line of output is cut into pieces of this many characters
MAX_BODY = 64 * 1024    # bytes in the body of a POST
MAX_ARGS = 100          # extra arguments one request may add to a job
NAME_FORM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
GET_PATHS = ("/health", "/status")
ACTIONS = {"servers": ("kill", "stop", "start"), "jobs": ("start", "stop")}
# The event's stack runs this file from the repo on the team volume. The stack's first job pulls that repo,
# so the code is always the newest; the commands come from whichever copy of fuzzball/30-real-stack.fz was
# started. The stack file says which revision it is (WX_STACK_REV in the box's environment, or --stack-rev),
# so that this program can notice an old copy and stop, instead of running old commands. It is an
# environment variable on purpose: an older copy of THIS file ignores it instead of failing on a new flag.
VOLUME_REPO = "/wx/repo"
STACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fuzzball", "30-real-stack.fz")
EXIT_OLD_STACK = 4


def stack_file_rev(path=STACK_FILE):
    """The STACK_REV that this checkout's fuzzball/30-real-stack.fz declares, or None if it declares none."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                found = re.match(r"\s*STACK_REV=(\d+)\s*$", line)
                if found:
                    return int(found.group(1))
    except OSError:
        pass
    return None


def old_stack_copy(given, repo_rev, on_volume):
    """Why this supervisor must not start, as text for the log, or None when it may start.

    given: the revision the stack file passed (WX_STACK_REV or --stack-rev); None if it passed none (a copy
    from before this check).
    repo_rev: the STACK_REV of the repo's own stack file. on_volume: this file runs from the team volume."""
    if not on_volume or repo_rev is None or (given is not None and given >= repo_rev):
        return None
    return "\n".join((
        "STOP: this stack was started from an OLD copy of fuzzball/30-real-stack.fz.",
        "  The copy that started it is revision %s; the repo's file is revision %d."
        % ("unknown (older than this check)" if given is None else given, repo_rev),
        "  The code on the volume is always the newest, so an old copy runs old commands with new code.",
        "  Start it again from the current file. After git pull:",
        "      fuzzball workflow start -n real-stack fuzzball/30-real-stack.fz      (or: python deploy/fb.py up)",
        "  In the web page, paste the current file from GitHub first: the editor keeps an old draft.",
        "  A workflow file of your own that was copied from it: copy the current one again, then your changes."))


class ConfigError(Exception):
    """The command line or the config file is wrong. The text says what to change."""


class ApiError(Exception):
    """An answer that is not 200: the HTTP status and the text for {"error": ...}."""

    def __init__(self, status, message):
        Exception.__init__(self, message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- reading the command line and the config file

def show(value):
    """A value as short JSON text, for messages."""
    try:
        text = json.dumps(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def check_name(name, where):
    """A server's or job's name: it goes into URLs, log file names and the front of output lines."""
    if not isinstance(name, str) or not NAME_FORM.fullmatch(name):
        raise ConfigError("%s: the name %s will not do; use letters, digits, - and _ (at most 40, not starting "
                          "with - or _)" % (where, show(name)))
    return name


def check_entry(spec, where):
    """One server or job as the config file gives it -> {"cmd": [...], "cwd": text or None, "env": {...}}."""
    if not isinstance(spec, dict):
        raise ConfigError('%s must be an object like {"cmd": ["python", "x.py"]}, got %s' % (where, show(spec)))
    for key in spec:
        if key not in ("cmd", "cwd", "env"):
            raise ConfigError("%s: unknown key %s; the keys are cmd, cwd and env" % (where, show(key)))
    cmd = spec.get("cmd")
    if (not isinstance(cmd, list) or not cmd or not all(isinstance(part, str) and "\0" not in part for part in cmd)
            or not cmd[0]):
        raise ConfigError('%s: cmd must be a list of texts like ["python", "x.py", "--port", "8101"], got %s'
                          % (where, show(cmd)))
    cwd = spec.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd):
        raise ConfigError("%s: cwd must be a folder name, got %s" % (where, show(cwd)))
    env = spec.get("env")
    if env is None:
        env = {}
    if not isinstance(env, dict) or not all(key and "=" not in key and "\0" not in key and isinstance(value, str)
                                            and "\0" not in value for key, value in env.items()):
        raise ConfigError('%s: env must be an object of texts like {"WX_DATA": "/vol/bundle"} (write numbers in '
                          "quotes), got %s" % (where, show(env)))
    return {"cmd": list(cmd), "cwd": cwd, "env": dict(env)}


def no_key_twice(pairs):
    """For json.loads: build one object, and refuse a key that appears twice (a copied block not renamed)."""
    built = {}
    for key, value in pairs:
        if key in built:
            raise ConfigError("the key %s appears twice in one object" % show(key))
        built[key] = value
    return built


def load_config(path):
    """Read the config file -> (servers, jobs), each {name: {"cmd": [...], "cwd": ..., "env": {...}}}."""
    try:
        with open(path, encoding="utf-8") as f:
            config = json.loads(f.read(), object_pairs_hook=no_key_twice)
    except OSError as e:
        raise ConfigError("cannot read the config file %s: %s" % (path, e.strerror or e))
    except ConfigError as e:
        raise ConfigError("%s: %s" % (path, e))
    except (ValueError, RecursionError) as e:
        raise ConfigError("the config file %s is not valid JSON: %s" % (path, e))
    if not isinstance(config, dict):
        raise ConfigError('%s: the top level must be an object like {"servers": {...}, "jobs": {...}}' % path)
    for key in config:
        if key not in ("servers", "jobs"):
            raise ConfigError("%s: unknown key %s; the keys are servers and jobs" % (path, show(key)))
    tables = []
    for kind in ("servers", "jobs"):
        group = config.get(kind)
        if group is None:
            group = {}
        if not isinstance(group, dict):
            raise ConfigError('%s: "%s" must be an object like {"name": {"cmd": [...]}}, got %s'
                              % (path, kind, show(group)))
        table = {}
        for name, spec in group.items():
            where = "%s: %s.%s" % (path, kind, name)
            check_name(name, where)
            table[name] = check_entry(spec, where)
        tables.append(table)
    return tables[0], tables[1]


def named_command(text, flag):
    """'forecast=python serve/forecast_server.py --node A' -> ("forecast", {"cmd": [...], ...}).

    The command is split the way a shell splits words (quotes work), but no shell ever runs it."""
    name, equals, command = text.partition("=")
    if not equals:
        raise ConfigError('%s wants NAME="COMMAND", like %s forecast="python serve/forecast_server.py", got %s'
                          % (flag, flag, show(text)))
    check_name(name, '%s NAME="COMMAND"' % flag)
    try:
        cmd = shlex.split(command)
    except ValueError as e:
        raise ConfigError("%s %s: the command cannot be split into words (%s): %s" % (flag, name, e, show(command)))
    if not cmd or not cmd[0]:
        raise ConfigError("%s %s: the command is empty" % (flag, name))
    return name, {"cmd": cmd, "cwd": None, "env": {}}


def read_table(config, serve, job):
    """Everything to run, from --config and from every --serve and --job -> (servers, jobs), in the order given."""
    servers, jobs = load_config(config) if config else ({}, {})
    for flag, texts, table in (("--serve", serve, servers), ("--job", job, jobs)):
        for text in texts:
            name, spec = named_command(text, flag)
            if name in table:
                raise ConfigError("%s %s: this name is given twice" % (flag, name))
            table[name] = spec
    for name in servers:
        if name in jobs:
            raise ConfigError("the name %s is used for a server and for a job; give them different names" % name)
    if not servers and not jobs:
        raise ConfigError('nothing to run: give --config FILE or --serve NAME="COMMAND" (--job works the same way)')
    return servers, jobs


def seconds_list(text, flag):
    """"0.2,0.5,1" -> [0.2, 0.5, 1.0]."""
    try:
        steps = [float(part) for part in text.split(",")]
    except ValueError:
        steps = []
    if not steps or not all(0 <= step <= 86400 for step in steps):
        raise ConfigError("%s wants seconds separated by commas, like 1,2,5,10,30, got %s" % (flag, show(text)))
    return steps


# ---------------------------------------------------------------- processes

def signal_group(pgid, signum):
    """Send a signal to a whole process group. True if it was sent. A group that is gone is not an error."""
    if pgid <= 1 or pgid == os.getpgrp():       # never the first process of the system, never our own group
        return False
    try:
        os.killpg(pgid, signum)
        return True
    except OSError:                             # no such group any more (macOS can say "not permitted" instead)
        return False


def exit_text(code):
    """An exit code in words. A negative code is how subprocess reports a signal: -9 is SIGKILL."""
    if code >= 0:
        return "exited with code %d" % code
    try:
        return "was killed by %s" % signal.Signals(-code).name
    except ValueError:
        return "was killed by signal %d" % -code


def extra_args(data):
    """The optional body {"args": ["--steps", "200"]} of POST /jobs/<name>/start -> the extra arguments.

    Nothing else in the body is used. ApiError(400) if the body will not do."""
    if not data.strip():
        return []
    try:
        body = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        raise ApiError(400, "the body is not valid JSON")
    if body is None:
        return []
    if not isinstance(body, dict):
        raise ApiError(400, 'the body must be an object like {"args": ["--steps", "200"]}, got %s' % show(body))
    args = body.get("args")
    if args is None:
        return []
    if (not isinstance(args, list) or len(args) > MAX_ARGS
            or not all(isinstance(arg, str) and "\0" not in arg for arg in args)):
        raise ApiError(400, 'args must be a list of at most %d texts like ["--steps", "200"] (write numbers in '
                            "quotes), got %s" % (MAX_ARGS, show(args)))
    return args


class Child(object):
    """One command from the config, and what is known about its process. A server or a job."""

    def __init__(self, name, kind, spec):
        self.name = name
        self.kind = kind                    # "server" or "job"
        self.cmd = list(spec["cmd"])
        self.cwd = spec.get("cwd")
        self.env = dict(spec.get("env") or {})
        self.proc = None                    # the process (a subprocess.Popen) from its start until its end is noted
        self.state = "stopped" if kind == "server" else "idle"
        self.want_up = kind == "server"     # a server: False from a stop on purpose until the next start request
        self.on_purpose = False             # True once this supervisor has told the running process to end
        self.tries = 0                      # how many times a start was tried
        self.restarts = 0                   # a server: how many times it was started again after dying
        self.restart_due = False            # a server: its next start is such a restart
        self.fails = 0                      # a server: which wait comes next (an index into the backoff steps)
        self.next_start = 0.0               # a server: when it may start (time.monotonic)
        self.started_mono = None            # when the current or last process started (time.monotonic)
        self.started_wall = None            # the same moment as time.time()
        self.seconds = None                 # a job: how long its last run took
        self.last_exit = None               # how the last process ended; below zero = killed by that signal
        self.error = None                   # why the last start failed, if it did
        self.lines = collections.deque(maxlen=LAST_LINES)      # a job: the last lines it printed
        self.lines_of = None                # a job: the process those lines came from
        self.log = None                     # its log file, with --log-dir
        self.thread = None                  # a server: its keeper thread. A job: the thread of its last run


class Stack(object):
    """The table of children and everything that happens to them.

    One lock (self.lock) guards the table. self.changed is a condition on that lock: every change to the table
    notifies it, and whoever waits for a change waits on it, which lets go of the lock meanwhile. The lock is
    never held while waiting for a process, and nothing is printed while it is held."""

    def __init__(self, name, servers, jobs, admin=False, backoff=BACKOFF_S, stable_s=STABLE_S, grace_s=GRACE_S,
                 job_grace_s=JOB_GRACE_S):
        self.name = name
        self.admin = admin
        self.backoff = tuple(backoff)
        self.stable_s = stable_s
        self.grace_s = grace_s
        self.job_grace_s = job_grace_s
        self.servers = dict((key, Child(key, "server", spec)) for key, spec in servers.items())    # in config order
        self.jobs = dict((key, Child(key, "job", spec)) for key, spec in jobs.items())
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.out_lock = threading.Lock()    # one line at a time on stdout
        self.log_lock = threading.Lock()    # and in the log files
        self.stalled = False                # True while stdout takes no lines (nobody reads it)
        self.silent = False                 # True at the very end: nothing more is printed
        self.closing = False                # True from the moment this supervisor was told to stop
        self.t0 = time.monotonic()

    def everyone(self):
        return list(self.servers.values()) + list(self.jobs.values())

    # ------------------------------------------------------------ output

    def open_logs(self, folder):
        """With --log-dir: one file per child, <name>.log, added to if it is there. OSError if that fails."""
        os.makedirs(folder, exist_ok=True)
        for child in self.everyone():
            child.log = open(os.path.join(folder, child.name + ".log"), "a", buffering=1, encoding="utf-8",
                             errors="replace")

    def put(self, line, child=None, plain=None):
        """Write one line to stdout and, for a child with a log file, to that file (`plain` instead, if given).

        When nobody reads this process's stdout any more, one thread hangs in its write. The others must not
        queue up behind it for ever, or no server would be started again: a line that cannot have its turn
        within OUT_WAIT_S is dropped (the log files still get it)."""
        if child is not None:
            with self.log_lock:
                if child.log is not None:
                    try:
                        child.log.write((line if plain is None else plain) + "\n")
                    except (OSError, ValueError):
                        child.log = None    # the log file broke (a full disk, say): go on without it
        if self.stalled:
            mine = self.out_lock.acquire(False)
        else:
            mine = self.out_lock.acquire(True, OUT_WAIT_S)
        if not mine:
            self.stalled = True
            return
        try:
            if self.silent:
                return
            if self.stalled:
                self.stalled = False
                sys.stdout.write("[stack %s %s] stdout took no lines for a while; some were dropped\n"
                                 % (self.name, time.strftime("%H:%M:%S")))
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except (OSError, ValueError, AttributeError):
            pass                            # stdout is gone, or was never there. The children still need their keeper.
        finally:
            self.out_lock.release()

    def say(self, text, child=None):
        """One line from the supervisor itself, with the time of day. About a child: also into that child's log."""
        about = "" if child is None else child.name + ": "
        self.put("[stack %s %s] %s%s" % (self.name, time.strftime("%H:%M:%S"), about, text), child)

    def hush(self):
        """Print nothing from now on, and close the log files: the last step before this process leaves.

        Returns False if another thread still hangs in the middle of a line (stdout takes no more)."""
        if self.stalled:
            free = self.out_lock.acquire(False)
        else:
            free = self.out_lock.acquire(True, OUT_WAIT_S)
        self.silent = True
        if free:
            self.out_lock.release()
        with self.log_lock:
            for child in self.everyone():
                if child.log is not None:
                    try:
                        child.log.close()
                    except (OSError, ValueError):
                        pass
                    child.log = None
        return free

    def copy_output(self, child, proc):
        """Copy what one process prints, line by line, until its pipe closes. Runs in its own thread."""
        front = "[%s] " % child.name
        try:
            while True:
                line = proc.stdout.readline(MAX_LINE)
                if not line:
                    break
                text = line.rstrip("\n")
                if child.kind == "job":
                    with self.lock:
                        if child.lines_of is proc:      # not a late line of an earlier run
                            child.lines.append(text)
                self.put(front + text, child, text)
        except (OSError, ValueError):       # the pipe broke
            pass
        finally:
            try:
                proc.stdout.close()
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------ starting, and noting an end

    def spawn(self, child, extra=()):
        """Start the child's command, plus `extra` arguments. Returns None, or the reason it could not start.

        Call with the lock held: starting takes a moment and waits for nothing. The process gets its own
        session, so its process group is its own and its number is the process's pid."""
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")     # a Python child then prints each line when it happens
        env.update(child.env)
        child.tries += 1
        child.on_purpose = False
        child.started_mono = time.monotonic()
        child.started_wall = time.time()
        if child.kind == "job":
            child.lines.clear()
            child.last_exit = None
            child.seconds = None
        try:
            child.proc = subprocess.Popen(child.cmd + list(extra), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, cwd=child.cwd, env=env, start_new_session=True,
                                          encoding="utf-8", errors="replace")
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            child.error = "cannot start: %s" % e
            return str(e)
        child.lines_of = child.proc
        child.error = None
        child.state = "running"
        return None

    def next_wait(self, server, ran_s):
        """How long to wait before this server's next start: 1, 2, 5, 10, 30, 30, ... seconds, and from the
        first step again once it has stayed up for stable_s. Call with the lock held."""
        if ran_s >= self.stable_s:
            server.fails = 0
        wait = self.backoff[min(server.fails, len(self.backoff) - 1)]
        server.fails += 1
        return wait

    def launch(self, server):
        """Start a server now. Returns the line to print about it. Call with the lock held."""
        again = server.restart_due
        if again:
            server.restarts += 1
        server.restart_due = False
        problem = self.spawn(server)
        if problem is None:
            return "started, pid %d%s" % (server.proc.pid, ", restart %d" % server.restarts if again else "")
        wait = self.next_wait(server, 0.0)
        server.state = "backoff"
        server.restart_due = True
        server.next_start = time.monotonic() + wait
        return "cannot start (%s); trying again in %g s" % (problem, wait)

    def ended(self, child, code):
        """Note that the child's process has ended, and decide what comes next. Returns the line to print.
        Call with the lock held."""
        now = time.monotonic()
        ran_s = now - child.started_mono
        text = "%s after %.1f s" % (exit_text(code), ran_s)
        child.proc = None
        child.last_exit = code
        if child.kind == "job":
            child.seconds = ran_s
            child.state = "done" if code == 0 else "failed"
            return text
        if child.on_purpose or self.closing or not child.want_up:
            child.state = "stopped"
            child.next_start = now          # a start request starts it at once
            return text + "; it was asked to stop"
        wait = self.next_wait(child, ran_s)
        child.state = "backoff"
        child.restart_due = True
        child.next_start = now + wait
        return text + "; starting it again in %g s" % wait

    def attend(self, child, proc):
        """Stay with one process until it exits: copy its output, then note how it ended. No lock is held
        while waiting."""
        reader = threading.Thread(target=self.copy_output, args=(child, proc), name="out-" + child.name, daemon=True)
        reader.start()
        code = proc.wait()
        signal_group(proc.pid, signal.SIGKILL)      # what its main process left behind in the group goes too
        reader.join(READER_WAIT_S)                  # its last lines; a stray holder of the pipe cannot stall us
        with self.changed:
            note = self.ended(child, code)
            self.changed.notify_all()
        self.say(note, child)

    # ------------------------------------------------------------ the threads

    def start(self):
        """Start every server, and one keeper thread per server that stays with it from then on."""
        for server in self.servers.values():
            with self.changed:
                note = self.launch(server)
                self.changed.notify_all()
            self.say(note, server)
            server.thread = threading.Thread(target=self.keep, args=(server,), name="keep-" + server.name,
                                             daemon=True)
            server.thread.start()

    def keep(self, server):
        """The life of one server, in its own thread: start it, stay with it until it exits, wait, again."""
        while True:
            try:
                if not self.keep_once(server):
                    return
            except Exception:                       # a mistake in here must not end the keeping
                traceback.print_exc()
                time.sleep(1.0)

    def keep_once(self, server):
        """One round for one server. Returns False when the supervisor is closing and the server is down."""
        note = None
        with self.changed:
            while server.proc is None and note is None:
                if self.closing:
                    return False
                wait = server.next_start - time.monotonic() if server.want_up else None
                if wait is None or wait > 0:
                    self.changed.wait(wait)         # lets go of the lock; any change to the table wakes it
                else:
                    note = self.launch(server)
                    self.changed.notify_all()
            proc = server.proc
        if note is not None:
            self.say(note, server)
        if proc is not None:
            self.attend(server, proc)
        return True

    def run_job(self, job, proc, note):
        """One run of a job, in its own thread."""
        self.say(note, job)
        while True:
            try:
                self.attend(job, proc)
                return
            except Exception:
                traceback.print_exc()
                time.sleep(1.0)

    # ------------------------------------------------------------ ending processes on purpose

    def signal_child(self, proc, signum):
        """Signal the process group of a process that has not been collected yet. Call with the lock held."""
        if proc.returncode is None:
            signal_group(proc.pid, signum)

    def wait_gone(self, pairs, timeout):
        """Wait until the end of each (child, process) pair has been noted. True if all have ended in time."""
        with self.changed:
            return self.changed.wait_for(lambda: all(child.proc is not proc for child, proc in pairs), timeout)

    def end_all(self, pairs, grace_s):
        """End these (child, process) pairs on purpose: SIGTERM to each process group, then SIGKILL to those
        still there after grace_s. Returns when all are gone. Call without the lock."""
        with self.changed:
            pairs = [(child, proc) for child, proc in pairs if child.proc is proc]
            for child, proc in pairs:
                child.on_purpose = True
                self.signal_child(proc, signal.SIGTERM)
        for child, proc in pairs:
            self.say("SIGTERM sent to pid %d" % proc.pid, child)
        if self.wait_gone(pairs, grace_s):
            return
        with self.changed:
            late = [(child, proc) for child, proc in pairs if child.proc is proc]
            for child, proc in late:
                self.signal_child(proc, signal.SIGKILL)
        for child, proc in late:
            self.say("still running %g s after SIGTERM: SIGKILL sent to pid %d" % (grace_s, proc.pid), child)
        self.wait_gone(late, KILL_WAIT_S)

    def close(self):
        """Stop every child on purpose and wait until all are gone. Nothing is started after this.

        Returns the names of the children that are still there: none, unless SIGKILL did not end one."""
        with self.changed:
            self.closing = True
            pairs = [(child, child.proc) for child in self.everyone() if child.proc is not None]
            self.changed.notify_all()
        self.end_all(pairs, self.grace_s)
        end = time.monotonic() + KILL_WAIT_S
        for child in self.everyone():               # let each thread print its last line
            if child.thread is not None:
                try:
                    child.thread.join(max(0.0, end - time.monotonic()))
                except RuntimeError:
                    pass
        with self.lock:
            return [child.name for child in self.everyone() if child.proc is not None]

    # ------------------------------------------------------------ what the HTTP calls do

    def server_view(self, server, now):
        """One server as /status shows it. Call with the lock held."""
        running = server.proc is not None
        waiting = server.state == "backoff"
        return {"state": server.state,
                "pid": server.proc.pid if running else None,
                "restarts": server.restarts,
                "last_exit": server.last_exit,
                "up_s": round(now - server.started_mono, 1) if running else None,
                "next_start_in_s": round(max(0.0, server.next_start - now), 1) if waiting else None,
                "error": server.error}

    def job_view(self, job, now):
        """One job as /status shows it. Call with the lock held."""
        running = job.proc is not None
        seconds = now - job.started_mono if running else job.seconds
        return {"state": job.state,
                "pid": job.proc.pid if running else None,
                "started": None if job.started_wall is None else round(job.started_wall, 3),
                "seconds": None if seconds is None else round(seconds, 1),
                "exit": job.last_exit,
                "error": job.error,
                "last_lines": list(job.lines)}

    def status(self):
        """The answer to GET /status. The servers and the jobs come in the order of the config."""
        with self.lock:
            now = time.monotonic()
            return {"name": self.name,
                    "uptime_s": round(now - self.t0, 1),
                    "servers": dict((server.name, self.server_view(server, now)) for server in self.servers.values()),
                    "jobs": dict((job.name, self.job_view(job, now)) for job in self.jobs.values())}

    def result(self, action, child):
        """The answer to an admin call: what was asked, and the child as /status shows it now."""
        with self.lock:
            now = time.monotonic()
            view = self.server_view(child, now) if child.kind == "server" else self.job_view(child, now)
        return {"action": action, child.kind: child.name, "status": view}

    def find(self, table, kind, name):
        """The configured child of that name, or ApiError(404). Call with the lock held."""
        child = table.get(name)
        if child is None:
            raise ApiError(404, "no %s named %s in the config; the %ss are: %s"
                           % (kind, show(name), kind, ", ".join(table) or "none"))
        return child

    def refuse_if_closing(self):
        if self.closing:
            raise ApiError(503, "this supervisor is stopping; nothing is started any more")

    def ask_kill(self, name):
        """POST /servers/<name>/kill: SIGKILL, as a crash would. The keeper starts it again by its usual rule."""
        with self.changed:
            server = self.find(self.servers, "server", name)
            proc = server.proc
            if proc is None:
                raise ApiError(409, "%s is not running (state: %s), so there is nothing to kill" % (name, server.state))
            self.signal_child(proc, signal.SIGKILL)
        self.say("SIGKILL sent to pid %d on request, as a crash would" % proc.pid, server)
        self.wait_gone([(server, proc)], KILL_WAIT_S)
        return self.result("kill", server)

    def ask_stop(self, name):
        """POST /servers/<name>/stop: stop it and keep it down until a start is asked for."""
        with self.changed:
            server = self.find(self.servers, "server", name)
            server.want_up = False
            server.restart_due = False
            proc = server.proc
            was_waiting = proc is None and server.state == "backoff"
            if proc is None:
                server.state = "stopped"
                self.changed.notify_all()
        if was_waiting:
            self.say("stopped on request while it waited to start", server)
        if proc is not None:
            self.end_all([(server, proc)], self.grace_s)
        return self.result("stop", server)

    def ask_start(self, name):
        """POST /servers/<name>/start: start a stopped server now. This also ends a backoff wait early."""
        with self.changed:
            server = self.find(self.servers, "server", name)
            self.refuse_if_closing()
            server.want_up = True
            if server.proc is None:
                tries = server.tries
                server.next_start = time.monotonic()
                self.changed.notify_all()           # its keeper does the start; wait a moment to see it happen
                self.changed.wait_for(lambda: server.tries != tries or self.closing, START_WAIT_S)
        return self.result("start", server)

    def ask_job_start(self, name, data):
        """POST /jobs/<name>/start: run the job once. 409 while it runs. The body may add arguments."""
        with self.changed:
            job = self.find(self.jobs, "job", name)
            extra = extra_args(data)
            self.refuse_if_closing()
            if job.proc is not None:
                raise ApiError(409, "%s is already running (pid %d); a job runs one at a time" % (name, job.proc.pid))
            problem = self.spawn(job, extra)
            if problem is None:
                note = "started, pid %d%s" % (job.proc.pid, ", extra arguments %s" % show(extra) if extra else "")
                job.thread = threading.Thread(target=self.run_job, args=(job, job.proc, note), name="job-" + name,
                                              daemon=True)
                job.thread.start()
            else:
                job.state = "failed"
            self.changed.notify_all()
        if problem is not None:
            self.say("cannot start (%s)" % problem, job)
            raise ApiError(500, "cannot start %s: %s" % (name, problem))
        return self.result("start", job)

    def ask_job_stop(self, name):
        """POST /jobs/<name>/stop: SIGTERM to the job's process group, SIGKILL after job_grace_s."""
        with self.changed:
            job = self.find(self.jobs, "job", name)
            proc = job.proc
        if proc is not None:
            self.end_all([(job, proc)], self.job_grace_s)
        return self.result("stop", job)

    def admin_call(self, kind, name, action, data):
        """One admin call -> the answer. `kind` and `action` come from ACTIONS, the name from the URL."""
        if kind == "servers":
            return {"kill": self.ask_kill, "stop": self.ask_stop, "start": self.ask_start}[action](name)
        if action == "start":
            return self.ask_job_start(name, data)
        return self.ask_job_stop(name)


# ---------------------------------------------------------------- HTTP

def admin_path(path):
    """"/servers/forecast/kill" -> ("servers", "forecast", "kill"). None if the path is not an admin call."""
    parts = path.split("/")
    if len(parts) == 4 and parts[0] == "" and parts[1] in ACTIONS and parts[3] in ACTIONS[parts[1]]:
        return parts[1], parts[2], parts[3]
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "node-stack"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")       # on every answer, errors included
        BaseHTTPRequestHandler.end_headers(self)

    def answer(self, status, body, content_type="application/json", headers=()):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in headers:
                self.send_header(name, value)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:                     # the client has gone away
            self.close_connection = True

    def answer_json(self, status, obj, headers=()):
        self.answer(status, json.dumps(obj).encode("utf-8"), headers=headers)

    def send_error(self, code, message=None, explain=None):
        """The base class calls this for a request it cannot read. Answer in JSON like everywhere else."""
        self.close_connection = True
        self.answer_json(code, {"error": message or self.responses.get(code, ("error",))[0]})

    def log_message(self, format, *args):
        pass                                # no line per request; the admin calls print what they do

    def do_GET(self):
        self.safely(self.get)

    def do_POST(self):
        self.safely(self.post)

    def do_OPTIONS(self):
        """The browser's question before a cross-origin POST (the dashboard calls this server directly)."""
        self.no_body()
        self.answer(200, b"", "text/plain", [("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                                             ("Access-Control-Allow-Headers", "*"),
                                             ("Access-Control-Max-Age", "600")])

    def safely(self, handle):
        """Run one request. Whatever goes wrong, the client gets an answer and the supervisor lives on."""
        try:
            handle()
        except ApiError as e:
            self.answer_json(e.status, {"error": e.message})
        except Exception as e:
            traceback.print_exc()
            self.answer_json(500, {"error": "internal error: %s: %s" % (type(e).__name__, e)})

    def no_body(self):
        """A GET or OPTIONS should carry no body. If one is announced, do not read it: close afterwards."""
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0").strip() not in ("", "0"):
            self.close_connection = True

    def read_body(self):
        """The body of a POST as bytes; it may be empty. ApiError(400) if it cannot be read."""
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = -1
        if self.headers.get("Transfer-Encoding") or not 0 <= size <= MAX_BODY:
            self.close_connection = True    # the body stays unread, so this connection cannot go on
            raise ApiError(400, "send the body with a Content-Length of at most %d bytes" % MAX_BODY)
        data = self.rfile.read(size)
        if len(data) != size:
            self.close_connection = True
            raise ApiError(400, "the body is shorter than Content-Length says")
        return data

    def wrong_method(self, allowed, path):
        self.answer_json(405, {"error": "use %s for %s" % (allowed, path)}, [("Allow", allowed)])

    def get(self):
        stack = self.server.stack
        self.no_body()
        path = urlsplit(self.path).path
        if path == "/health":
            self.answer_json(200, {"status": "ok", "name": stack.name})
        elif path == "/status":
            self.answer_json(200, stack.status())
        elif admin_path(path):
            self.wrong_method("POST", path)
        else:
            raise ApiError(404, "no such path: %s" % show(path))

    def post(self):
        stack = self.server.stack
        data = self.read_body()
        path = urlsplit(self.path).path
        call = admin_path(path)
        if call is None:
            if path in GET_PATHS:
                return self.wrong_method("GET", path)
            raise ApiError(404, "no such path: %s" % show(path))
        if not stack.admin:
            raise ApiError(403, "this supervisor was started without --admin, so it changes nothing over HTTP")
        kind, name, action = call
        self.answer_json(200, stack.admin_call(kind, name, action, data))


class HttpServer(ThreadingHTTPServer):
    daemon_threads = True                   # a waiting request never keeps the process alive
    request_queue_size = 64

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return                          # a client that hangs up mid-request is not worth a traceback
        ThreadingHTTPServer.handle_error(self, request, client_address)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Keeps one node's servers alive inside its one Fuzzball service, and starts them again when "
                    "they die. The top of this file lists the HTTP calls.")
    p.add_argument("--name", required=True, help="this node's name, as /health and /status show it (A or B)")
    p.add_argument("--config", metavar="FILE", help="the JSON file that lists the servers and the jobs")
    p.add_argument("--serve", action="append", default=[], metavar='NAME="COMMAND"',
                   help="a server to keep alive. The command is split into words as a shell would, but no shell "
                        "runs it. Give it once per server")
    p.add_argument("--job", action="append", default=[], metavar='NAME="COMMAND"',
                   help="a command that runs once each time POST /jobs/NAME/start asks for it (needs --admin)")
    p.add_argument("--host", default="0.0.0.0",
                   help="address to listen on (default 0.0.0.0, so the router and the dashboard can reach it)")
    p.add_argument("--port", type=int, default=8109, help="default 8109; 0 picks a free port")
    p.add_argument("--admin", action="store_true",
                   help="allow the POST calls that kill, stop and start the configured commands. Anyone who can "
                        "reach the port can use them")
    p.add_argument("--log-dir", metavar="DIR", help="also write each child's output to DIR/<name>.log")
    p.add_argument("--stack-rev", type=int, metavar="N",
                   help="the STACK_REV of the stack file that starts this supervisor (default: the environment "
                        "variable WX_STACK_REV). On the team volume the supervisor stops when it is missing "
                        "or lower than the repo's own stack file")
    # Hidden: only the tests use these, to keep their waits short.
    p.add_argument("--test-backoff", help=argparse.SUPPRESS)
    p.add_argument("--test-stable", type=float, default=STABLE_S, help=argparse.SUPPRESS)
    p.add_argument("--test-grace", type=float, default=GRACE_S, help=argparse.SUPPRESS)
    p.add_argument("--test-job-grace", type=float, default=JOB_GRACE_S, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 0 <= args.port <= 65535:
        p.error("--port must be from 0 to 65535")
    try:
        check_name(args.name, "--name")
        servers, jobs = read_table(args.config, args.serve, args.job)
        backoff = seconds_list(args.test_backoff, "--test-backoff") if args.test_backoff else BACKOFF_S
    except ConfigError as e:
        p.error(str(e))

    try:
        sys.stdout.reconfigure(errors="replace")    # a child's odd characters must not cost a line of output
    except (AttributeError, ValueError):
        pass
    on_volume = (os.path.realpath(os.path.dirname(os.path.dirname(STACK_FILE))) == VOLUME_REPO
                 or os.environ.get("WX_STACK_GUARD") == "1")       # the variable is for the tests
    given = args.stack_rev
    if given is None:
        try:
            given = int(os.environ.get("WX_STACK_REV", ""))
        except ValueError:
            given = None
    why = old_stack_copy(given, stack_file_rev(), on_volume)
    if why:
        print(why, flush=True)              # on stdout, where `fuzzball workflow log <id> box-a` shows it
        return EXIT_OLD_STACK
    stack = Stack(args.name, servers, jobs, admin=args.admin, backoff=backoff, stable_s=args.test_stable,
                  grace_s=args.test_grace, job_grace_s=args.test_job_grace)
    if args.log_dir:
        try:
            stack.open_logs(args.log_dir)
        except OSError as e:
            sys.stderr.write("cannot write logs in %s: %s\n" % (args.log_dir, e))
            return 2
    try:
        httpd = HttpServer((args.host, args.port), Handler)
    except OSError as e:
        sys.stderr.write("cannot listen on %s:%d: %s\n" % (args.host, args.port, e))
        return 2
    httpd.stack = stack

    got = []                                # the signals that arrived

    def on_signal(signum, frame):
        got.append(signum)                  # only this: a signal handler must not take a lock

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:   # ignored at the start, as under nohup: it stays so
        signal.signal(signal.SIGHUP, on_signal)
    print("listening on http://%s:%d" % (args.host, httpd.server_address[1]), flush=True)
    stack.say("keeping %d server(s): %s. %d job(s): %s. Admin calls are %s"
              % (len(servers), ", ".join(servers) or "none", len(jobs), ", ".join(jobs) or "none",
                 "on" if args.admin else "off"))
    if jobs and not args.admin:
        stack.say("note: without --admin nothing can start a job")

    def serve():
        while True:
            try:
                httpd.serve_forever(poll_interval=0.1)
            except Exception:
                traceback.print_exc()
                time.sleep(1.0)
            if stack.closing:
                return

    serving = threading.Thread(target=serve, name="http", daemon=True)
    code = 0
    try:
        stack.start()                       # the servers are started before the first request is answered
        serving.start()
        while not got:                      # signals arrive in this (the main) thread
            time.sleep(0.2)
        stack.say("%s received: stopping every child" % signal.Signals(got[0]).name)
    except Exception:                       # the threads could not be started
        traceback.print_exc()
        code = 1
    left = stack.close()
    if serving.is_alive():
        closer = threading.Thread(target=httpd.shutdown, name="http-close", daemon=True)
        closer.start()
        closer.join(KILL_WAIT_S)            # never wait forever for the HTTP server
    httpd.server_close()
    stack.say("%s; leaving with code %d"
              % ("still there after SIGKILL: %s" % ", ".join(left) if left else "every child is gone", code))
    if not stack.hush():
        os._exit(code)                      # a thread hangs in a write to stdout: do not wait for it at the end
    return code


if __name__ == "__main__":
    exit_code = main()
    try:
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        # stdout is gone. Point it at nothing, so that Python's own last flush cannot turn a clean exit into
        # a failed one (a Fuzzball service that exits with a non-zero code fails its workflow).
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError, AttributeError):
            pass
    sys.exit(exit_code)
