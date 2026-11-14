"""Tests for serve/node_stack.py, the supervisor that keeps one node's servers alive.

    .venv/bin/python -m unittest tests/test_node_stack.py -v

Every supervisor here is a child process on 127.0.0.1, on a port the OS picks. The commands it keeps are tiny
Python scripts that print and sleep. Hidden options of the supervisor make its waits short. Each test stops its
supervisor at the end and checks that it left with code 0, printed no traceback and left no child behind. Every
script also ends by itself within about a minute, so nothing can stay running for long whatever happens.

Written at the hackathon (Nov 13-14, 2026).
"""
import http.client
import io
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = os.path.join(REPO, "serve", "node_stack.py")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))       # straight to 127.0.0.1, no proxy
SERVER_KEYS = ["state", "pid", "restarts", "last_exit", "up_s", "next_start_in_s", "error"]
JOB_KEYS = ["state", "pid", "started", "seconds", "exit", "error", "last_lines"]

sys.path.insert(0, os.path.join(REPO, "serve"))
import node_stack as ns  # noqa: E402   (standard library only)

# ---- The commands the supervisor keeps in these tests. Each ends by itself within about a minute.

# A server: prints a line every 0.1 s.
TICK = """
import time
for i in range(600):
    print("tick %d" % i, flush=True)
    time.sleep(0.1)
"""

# Exits at once with code 3.
FAILS = """
import sys
print("about to fail", flush=True)
sys.exit(3)
"""

# Stays up for the seconds given as its argument, then exits with code 0.
BRIEF = """
import sys, time
print("up", flush=True)
time.sleep(float(sys.argv[1]))
"""

# On SIGTERM it creates the file named by its argument, then exits.
POLITE = """
import signal, sys, time
def leave(signum, frame):
    open(sys.argv[1], "w").close()
    sys.exit(0)
signal.signal(signal.SIGTERM, leave)
print("ready", flush=True)
for i in range(600):
    time.sleep(0.1)
"""

# Ignores SIGTERM.
STUBBORN = """
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("ready", flush=True)
for i in range(300):
    time.sleep(0.1)
"""

# Starts a process of its own (same process group, same output pipe) and prints that pid.
TREE = """
import subprocess, sys, time
kid = subprocess.Popen([sys.executable, "-c", "import time\\nfor i in range(300): time.sleep(0.1)"])
print("kid %d" % kid.pid, flush=True)
for i in range(300):
    time.sleep(0.1)
"""

# Starts a process of its own too. On SIGTERM that kid creates the file named by the argument and exits; the
# main process only waits for its kid, then exits. So the file shows that SIGTERM went to the whole group.
FAMILY = """
import signal, subprocess, sys, time
KID = '''
import signal, sys, time
def leave(signum, frame):
    open(sys.argv[1], "w").close()
    sys.exit(0)
signal.signal(signal.SIGTERM, leave)
print("kid ready", flush=True)
for i in range(300):
    time.sleep(0.1)
'''
kid = subprocess.Popen([sys.executable, "-c", KID, sys.argv[1]])
def leave(signum, frame):
    kid.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, leave)
print("ready", flush=True)
for i in range(300):
    time.sleep(0.1)
"""

# A job: prints 30 lines and its extra arguments, without flushing. Then it waits until the file named by its
# first argument is there (15 s at most) and exits with code N if the extra arguments are "--exit N".
WORK = """
import os, sys, time
for i in range(30):
    print("line %d" % i)
print("extra=[%s]" % " ".join(sys.argv[2:]))
for i in range(300):
    if os.path.exists(sys.argv[1]):
        break
    time.sleep(0.05)
print("the end")
sys.exit(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
"""

# Prints its working folder and the WX_DATA it sees.
WHERE = """
import os, time
print("cwd=%s data=%s" % (os.getcwd(), os.environ.get("WX_DATA")), flush=True)
for i in range(600):
    time.sleep(0.1)
"""

# Prints about 600 KB at once, far more than a pipe holds, then goes on like TICK.
CHATTY = """
import time
for i in range(3000):
    print("chatty line %04d " % i + "x" * 180, flush=True)
for i in range(600):
    print("tick %d" % i, flush=True)
    time.sleep(0.1)
"""

# Creates the file named by its argument and exits: proof that it was started.
TOUCH = """
import sys
open(sys.argv[1], "w").close()
"""


def command(script, *args):
    """The command line that runs one of the scripts above, as a list of words."""
    return [sys.executable, "-c", script] + [str(arg) for arg in args]


def named(name, script, *args):
    """The value for --serve or --job: NAME=COMMAND, quoted so that the supervisor splits it into the same words."""
    return "%s=%s" % (name, shlex.join(command(script, *args)))


def alive(pid):
    """True while a process with this pid exists (and is ours to signal)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gone(pid):
    """True once the process has ended. One that is dead but not yet collected by its parent (a zombie) counts
    as gone: that is all a supervisor can do about a process that is not its own child."""
    if not alive(pid):
        return True
    try:
        with open("/proc/%d/stat" % pid) as f:                  # Linux
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        pass
    try:                                                        # macOS
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                               timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return not state or state.startswith("Z")


def wait_until(check, timeout=15.0, what="what the test waits for"):
    """Call check() until it returns something true, and return that. AssertionError after `timeout` seconds."""
    end = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() >= end:
            raise AssertionError("gave up after %g s waiting for %s" % (timeout, what))
        time.sleep(0.02)


def call(url, path, body=None, method=None, timeout=30, raw=None):
    """One HTTP request. Returns (status, headers, body); a JSON body comes back parsed."""
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
    request = urllib.request.Request(url + path, data=data, method=method)
    try:
        with OPENER.open(request, timeout=timeout) as response:
            status, headers, text = response.status, response.headers, response.read()
    except urllib.error.HTTPError as e:
        status, headers, text = e.code, e.headers, e.read()
        e.close()
    text = text.decode("utf-8")
    if headers.get("Content-Type", "").startswith("application/json"):
        return status, headers, json.loads(text)
    return status, headers, text


def post(url, path, body=None, raw=None):
    return call(url, path, body=body, method="POST", raw=raw)


class Running(object):
    """A supervisor running as a child process on 127.0.0.1, on a port the OS picked."""

    def __init__(self, *flags):
        self.errors = tempfile.NamedTemporaryFile(prefix="node_stack_test_", suffix=".log")     # its stderr
        self.stderr_text = ""
        self.lines = []                 # every line it printed after the first
        self.pids = set()               # every child pid that /status has shown
        base = [sys.executable, STACK, "--name", "A", "--host", "127.0.0.1", "--port", "0"]
        self.proc = subprocess.Popen(base + list(flags), cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=self.errors, universal_newlines=True)
        watchdog = threading.Timer(20, self.proc.kill)          # never wait forever for the first line
        watchdog.start()
        self.first_line = self.proc.stdout.readline().rstrip("\n")
        watchdog.cancel()
        self.reader = threading.Thread(target=self.collect, daemon=True)        # so the pipe cannot fill up
        self.reader.start()
        found = re.fullmatch(r"listening on (http://127\.0\.0\.1:(\d+))", self.first_line)
        if not found:
            self.stop()
            raise AssertionError("the supervisor did not start. First line: %r. stderr: %s"
                                 % (self.first_line, self.stderr_text))
        self.url, self.port = found.group(1), int(found.group(2))

    def collect(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))

    def status(self):
        """GET /status, parsed. Remembers every child pid it shows."""
        status, _, body = call(self.url, "/status")
        if status != 200:
            raise AssertionError("GET /status gave %s: %s" % (status, body))
        for group in ("servers", "jobs"):
            for entry in body[group].values():
                if entry["pid"]:
                    self.pids.add(entry["pid"])
        return body

    def wait_status(self, check, timeout=15.0, what="the expected /status"):
        """Ask for /status until check(status) is true. Returns that status."""
        last = {}

        def look():
            last["status"] = self.status()
            return check(last["status"])

        try:
            wait_until(look, timeout, what)
        except AssertionError as e:
            raise AssertionError("%s. The last /status: %s" % (e, json.dumps(last.get("status"))))
        return last["status"]

    def matches(self, pattern):
        """Every match of the pattern in the lines printed so far, in order."""
        return [found for found in (re.search(pattern, line) for line in list(self.lines)) if found]

    def wait_line(self, pattern, timeout=15.0):
        """Wait until a printed line matches. Returns the first match."""
        return wait_until(lambda: self.matches(pattern), timeout, "an output line like %r" % pattern)[0]

    def stop(self, sig=signal.SIGTERM):
        """Stop the supervisor if it still runs. Returns its exit code. Safe to call twice."""
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.reader.join(5)
        self.proc.stdout.close()
        if not self.errors.closed:
            with open(self.errors.name) as f:
                self.stderr_text = f.read()
            self.errors.close()
        return self.proc.returncode

    def left_behind(self):
        """The child pids that still run. They are killed here, so that nothing stays behind in any case."""
        left = [pid for pid in sorted(self.pids) if alive(pid)]
        for pid in left:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        return left


class StackTest(unittest.TestCase):
    """What every test here needs: a temp folder, and supervisors that are stopped and checked at the end."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="node_stack_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def start(self, *flags):
        stack = Running(*flags)
        self.addCleanup(self.finish, stack)
        return stack

    def finish(self, stack):
        try:
            stack.status()              # learn the newest pids, if it still answers
        except Exception:
            pass
        code = stack.stop()
        left = stack.left_behind()
        self.assertEqual(left, [], "children still ran after the supervisor had ended")
        self.assertEqual(code, 0, "the supervisor must leave with code 0. Its stderr: %s" % stack.stderr_text)
        self.assertNotIn("Traceback", stack.stderr_text)

    def both_running(self, stack):
        """Wait until every server runs. Returns {name: pid}."""
        seen = stack.wait_status(lambda s: all(server["state"] == "running" for server in s["servers"].values()),
                                 what="every server to run")
        return dict((name, server["pid"]) for name, server in seen["servers"].items())


class KeepingTest(StackTest):
    """Servers are started, watched and started again."""

    def test_starts_both_servers_and_reports_them(self):
        stack = self.start("--serve", named("one", TICK), "--serve", named("two", TICK), "--job", named("work", FAILS))
        self.assertRegex(stack.first_line, r"^listening on http://127\.0\.0\.1:\d+$")
        at_once = stack.status()["servers"]                     # the very first answer already shows them started
        self.assertEqual([server["state"] for server in at_once.values()], ["running", "running"])

        status, headers, body = call(stack.url, "/health")
        self.assertEqual((status, body), (200, {"status": "ok", "name": "A"}))
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")

        self.both_running(stack)
        status, headers, seen = call(stack.url, "/status")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual(list(seen), ["name", "uptime_s", "servers", "jobs"])
        self.assertEqual(seen["name"], "A")
        self.assertGreaterEqual(seen["uptime_s"], 0)
        self.assertEqual(list(seen["servers"]), ["one", "two"])
        pids = []
        for name in ("one", "two"):
            server = seen["servers"][name]
            self.assertEqual(list(server), SERVER_KEYS)
            self.assertEqual((server["state"], server["restarts"], server["last_exit"], server["next_start_in_s"],
                              server["error"]), ("running", 0, None, None, None))
            self.assertGreaterEqual(server["up_s"], 0)
            self.assertTrue(alive(server["pid"]))
            self.assertEqual(os.getpgid(server["pid"]), server["pid"])          # it leads its own process group
            pids.append(server["pid"])
        self.assertNotEqual(pids[0], pids[1])
        self.assertNotEqual(os.getpgid(pids[0]), os.getpgid(stack.proc.pid))
        self.assertEqual(list(seen["jobs"]["work"]), JOB_KEYS)
        self.assertEqual(seen["jobs"], {"work": {"state": "idle", "pid": None, "started": None, "seconds": None,
                                                 "exit": None, "error": None, "last_lines": []}})
        stack.wait_line(r"^\[one\] tick 1$")                    # each child's lines, with its name in front
        stack.wait_line(r"^\[two\] tick 1$")
        stack.wait_line(r"^\[stack A \d\d:\d\d:\d\d\] one: started, pid %d$" % pids[0])

    def test_killed_server_comes_back(self):
        stack = self.start("--admin", "--test-backoff", "0.2,0.4", "--serve", named("one", TICK),
                           "--serve", named("two", TICK))
        first = self.both_running(stack)

        status, headers, body = post(stack.url, "/servers/one/kill")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertEqual((body["action"], body["server"]), ("kill", "one"))
        self.assertEqual(body["status"]["last_exit"], -9)       # the answer comes when the process is dead
        self.assertFalse(alive(first["one"]))

        seen = stack.wait_status(lambda s: s["servers"]["one"]["state"] == "running")
        one = seen["servers"]["one"]
        self.assertEqual((one["restarts"], one["last_exit"]), (1, -9))
        self.assertNotEqual(one["pid"], first["one"])
        self.assertTrue(alive(one["pid"]))
        self.assertEqual((seen["servers"]["two"]["pid"], seen["servers"]["two"]["restarts"]), (first["two"], 0))

        os.kill(one["pid"], signal.SIGKILL)                     # a crash that the supervisor was not told about
        seen = stack.wait_status(lambda s: s["servers"]["one"]["restarts"] == 2
                                 and s["servers"]["one"]["state"] == "running")
        self.assertNotIn(seen["servers"]["one"]["pid"], (first["one"], one["pid"]))
        self.assertEqual((seen["servers"]["two"]["pid"], seen["servers"]["two"]["restarts"]), (first["two"], 0))
        stack.wait_line(r"one: was killed by SIGKILL after [\d.]+ s; starting it again in 0\.[24] s$")
        stack.wait_line(r"one: started, pid %d, restart 2$" % seen["servers"]["one"]["pid"])

    def test_wait_grows_for_a_server_that_keeps_failing(self):
        stack = self.start("--test-backoff", "0.3,0.7,1.5", "--serve", named("bad", FAILS),
                           "--serve", named("good", TICK))
        good_pid = stack.wait_status(lambda s: s["servers"]["good"]["pid"])["servers"]["good"]["pid"]
        longest = {}        # restarts so far -> the longest wait /status showed with that count
        first = {}          # restarts so far -> (clock just before asking, the wait shown) when first seen waiting
        back = {}           # restarts so far -> clock when that count was first seen
        end = time.monotonic() + 30
        while 4 not in back:
            self.assertLess(time.monotonic(), end, "the failing server was not started 5 times in 30 s")
            asked = time.monotonic()
            bad = stack.status()["servers"]["bad"]
            back.setdefault(bad["restarts"], time.monotonic())
            self.assertIn(bad["state"], ("running", "backoff"))
            if bad["state"] == "backoff":
                self.assertEqual((bad["last_exit"], bad["pid"], bad["up_s"]), (3, None, None))
                longest[bad["restarts"]] = max(longest.get(bad["restarts"], 0.0), bad["next_start_in_s"])
                first.setdefault(bad["restarts"], (asked, bad["next_start_in_s"]))
            time.sleep(0.02)

        for count, step in enumerate([0.3, 0.7, 1.5, 1.5]):     # it stays at the last step
            if count == 0 and count not in longest:
                continue                                        # the first wait is short: a slow machine may miss it
            self.assertIn(count, longest, "never saw the wait before restart %d" % (count + 1))
            self.assertLessEqual(longest[count], step + 0.001)
            asked, shown = first[count]                         # the wait shown was really waited
            self.assertGreaterEqual(back[count + 1] - asked, shown - 0.06)
        self.assertGreater(longest[1], 0.3)                     # and it grew
        self.assertGreater(longest[2], 0.7)
        self.assertGreater(longest[3], 0.7)

        self.assertIsNone(stack.proc.poll())                    # the supervisor is still there
        good = stack.status()["servers"]["good"]
        self.assertEqual((good["state"], good["pid"], good["restarts"]), ("running", good_pid, 0))
        stack.wait_line(r"^\[bad\] about to fail$")
        stack.wait_line(r"bad: exited with code 3 after [\d.]+ s; starting it again in 1\.5 s$")

    def test_wait_goes_back_to_the_first_step_after_staying_up(self):
        # Each run lasts 0.7 s and "stayed up" is set to 0.4 s, so every wait is the first step (0.2 s). If the
        # wait kept growing, the second restart alone would take 20 more seconds.
        stack = self.start("--test-backoff", "0.2,20,20", "--test-stable", "0.4",
                           "--serve", named("brief", BRIEF, 0.7))
        seen = stack.wait_status(lambda s: s["servers"]["brief"]["restarts"] >= 3, timeout=12,
                                 what="three restarts, each after the first wait")
        self.assertEqual(seen["servers"]["brief"]["last_exit"], 0)      # an exit with code 0 is restarted too

    def test_command_that_cannot_start_is_tried_again(self):
        missing = os.path.join(self.tmp, "no-such-program")
        stack = self.start("--test-backoff", "0.2,0.4", "--serve", "ghost=%s --flag" % shlex.quote(missing),
                           "--serve", named("good", TICK))
        seen = stack.wait_status(lambda s: s["servers"]["ghost"]["restarts"] >= 2
                                 and s["servers"]["good"]["state"] == "running")
        ghost = seen["servers"]["ghost"]
        self.assertEqual((ghost["state"], ghost["pid"], ghost["last_exit"]), ("backoff", None, None))
        self.assertIn("cannot start", ghost["error"])
        self.assertIn("no-such-program", ghost["error"])
        good = seen["servers"]["good"]                          # the other server is not disturbed
        self.assertEqual((good["state"], good["restarts"]), ("running", 0))
        self.assertIsNone(stack.proc.poll())
        stack.wait_line(r"ghost: cannot start \(.*no-such-program.*\); trying again in 0\.4 s$")

    def test_what_a_dead_server_left_behind_is_killed(self):
        stack = self.start("--test-backoff", "0.2", "--serve", named("tree", TREE))
        kid = int(stack.wait_line(r"^\[tree\] kid (\d+)$").group(1))
        pid = self.both_running(stack)["tree"]
        self.assertTrue(alive(kid))
        self.assertEqual(os.getpgid(kid), pid)                  # the kid is in the server's process group

        os.kill(pid, signal.SIGKILL)                            # only the main process dies
        wait_until(lambda: gone(kid), 10, "the process that the dead server left behind to be killed")
        stack.wait_status(lambda s: s["servers"]["tree"]["restarts"] == 1
                          and s["servers"]["tree"]["state"] == "running")
        wait_until(lambda: len(stack.matches(r"^\[tree\] kid (\d+)$")) >= 2,
                   what="the restarted server to print its new kid")
        new_kid = int(stack.matches(r"^\[tree\] kid (\d+)$")[1].group(1))
        self.assertNotEqual(new_kid, kid)

        self.assertEqual(stack.stop(), 0)                       # the end takes the whole group along
        wait_until(lambda: gone(new_kid), 10, "the restarted server's kid to end with the supervisor")


class AdminTest(StackTest):
    """The POST calls that --admin turns on."""

    def test_stop_keeps_it_down_and_start_brings_it_back(self):
        stack = self.start("--admin", "--test-backoff", "0.1", "--serve", named("one", TICK),
                           "--serve", named("two", TICK))
        first = self.both_running(stack)

        status, _, body = post(stack.url, "/servers/one/stop")
        self.assertEqual(status, 200)
        self.assertEqual((body["action"], body["server"]), ("stop", "one"))
        self.assertEqual((body["status"]["state"], body["status"]["pid"], body["status"]["last_exit"]),
                         ("stopped", None, -15))
        self.assertFalse(alive(first["one"]))

        time.sleep(0.6)                                         # several waits long: it must stay down
        one = stack.status()["servers"]["one"]
        self.assertEqual((one["state"], one["pid"], one["restarts"], one["next_start_in_s"]),
                         ("stopped", None, 0, None))
        status, _, body = post(stack.url, "/servers/one/stop")  # stopping what is stopped is fine
        self.assertEqual((status, body["status"]["state"]), (200, "stopped"))
        status, _, body = post(stack.url, "/servers/one/kill")  # but there is nothing to kill
        self.assertEqual(status, 409)
        self.assertIn("not running", body["error"])

        status, _, body = post(stack.url, "/servers/one/start")
        self.assertEqual(status, 200)
        self.assertEqual((body["status"]["state"], body["status"]["restarts"]), ("running", 0))
        self.assertNotEqual(body["status"]["pid"], first["one"])
        self.assertTrue(alive(body["status"]["pid"]))
        stack.pids.add(body["status"]["pid"])

        status, _, again = post(stack.url, "/servers/one/start")        # starting what runs changes nothing
        self.assertEqual((status, again["status"]["pid"]), (200, body["status"]["pid"]))
        two = stack.status()["servers"]["two"]
        self.assertEqual((two["state"], two["pid"], two["restarts"]), ("running", first["two"], 0))

    def test_start_ends_a_wait_early_and_stop_ends_it_for_good(self):
        stack = self.start("--admin", "--test-backoff", "30", "--serve", named("bad", FAILS))
        seen = stack.wait_status(lambda s: s["servers"]["bad"]["state"] == "backoff")
        self.assertEqual(seen["servers"]["bad"]["restarts"], 0)
        self.assertGreater(seen["servers"]["bad"]["next_start_in_s"], 20)

        status, _, body = post(stack.url, "/servers/bad/start")         # do not wait the 30 s
        self.assertEqual(status, 200)
        seen = stack.wait_status(lambda s: s["servers"]["bad"]["restarts"] == 1
                                 and s["servers"]["bad"]["state"] == "backoff", timeout=10)
        self.assertGreater(seen["servers"]["bad"]["next_start_in_s"], 20)

        status, _, body = post(stack.url, "/servers/bad/stop")
        self.assertEqual((status, body["status"]["state"], body["status"]["next_start_in_s"]), (200, "stopped", None))
        status, _, body = post(stack.url, "/servers/bad/kill")
        self.assertEqual(status, 409)

    def test_job_runs_once_when_asked(self):
        go = os.path.join(self.tmp, "go")
        stack = self.start("--admin", "--serve", named("one", TICK), "--job", named("work", WORK, go))
        status, _, body = post(stack.url, "/jobs/work/start")
        self.assertEqual(status, 200)
        self.assertEqual((body["action"], body["job"], body["status"]["state"]), ("start", "work", "running"))
        pid = body["status"]["pid"]
        stack.pids.add(pid)
        self.assertTrue(alive(pid))
        self.assertEqual(os.getpgid(pid), pid)

        status, _, body = post(stack.url, "/jobs/work/start")           # a second start while it runs
        self.assertEqual(status, 409)
        self.assertIn("already running", body["error"])

        # Its lines show while it runs, though the job does not flush: the supervisor asks Python not to buffer.
        seen = stack.wait_status(lambda s: "extra=[]" in s["jobs"]["work"]["last_lines"])
        work = seen["jobs"]["work"]
        self.assertEqual((work["state"], work["pid"], work["exit"], work["error"]), ("running", pid, None, None))
        self.assertEqual(work["last_lines"], ["line %d" % i for i in range(11, 30)] + ["extra=[]"])
        self.assertLess(abs(work["started"] - time.time()), 60)
        self.assertGreaterEqual(work["seconds"], 0)

        open(go, "w").close()                                           # lets the job finish
        seen = stack.wait_status(lambda s: s["jobs"]["work"]["state"] != "running")
        work = seen["jobs"]["work"]
        self.assertEqual((work["state"], work["exit"], work["pid"]), ("done", 0, None))
        self.assertEqual(work["last_lines"], ["line %d" % i for i in range(12, 30)] + ["extra=[]", "the end"])
        self.assertGreater(work["seconds"], 0)
        self.assertFalse(alive(pid))
        stack.wait_line(r"^\[work\] the end$")
        time.sleep(0.4)                                                 # a job is never started again
        work = stack.status()["jobs"]["work"]
        self.assertEqual((work["state"], work["pid"]), ("done", None))

        # A second run, now with extra arguments. Nothing else in the body is used: no command comes from there.
        status, _, body = post(stack.url, "/jobs/work/start", {"args": ["--exit", "4"], "cmd": ["echo", "never run"]})
        self.assertEqual(status, 200)
        stack.pids.add(body["status"]["pid"])
        self.assertNotEqual(body["status"]["pid"], pid)
        seen = stack.wait_status(lambda s: s["jobs"]["work"]["state"] != "running")
        work = seen["jobs"]["work"]
        self.assertEqual((work["state"], work["exit"], work["pid"]), ("failed", 4, None))
        self.assertEqual(work["last_lines"][-2:], ["extra=[--exit 4]", "the end"])
        self.assertFalse(any("never run" in line for line in stack.lines))
        one = stack.status()["servers"]["one"]
        self.assertEqual((one["state"], one["restarts"]), ("running", 0))

    def test_job_stop_and_what_a_job_start_refuses(self):
        never = os.path.join(self.tmp, "never")
        stack = self.start("--admin", "--test-job-grace", "0.5", "--job", named("work", WORK, never),
                           "--job", named("stubborn", STUBBORN), "--job", "ghost=%s" % shlex.quote(never))
        for raw in (b"{not json", b"[1, 2]", b'{"args": "--steps 200"}', b'{"args": ["--steps", 200]}'):
            status, _, body = post(stack.url, "/jobs/work/start", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertIn("error", body)
        self.assertEqual(stack.status()["jobs"]["work"]["state"], "idle")

        status, _, body = post(stack.url, "/jobs/work/stop")            # nothing runs: nothing to do
        self.assertEqual((status, body["status"]["state"]), (200, "idle"))

        status, _, body = post(stack.url, "/jobs/work/start", raw=b"")
        self.assertEqual(status, 200)
        pid = body["status"]["pid"]
        stack.pids.add(pid)
        status, _, body = post(stack.url, "/jobs/work/stop")
        self.assertEqual((status, body["action"], body["job"]), (200, "stop", "work"))
        self.assertEqual((body["status"]["state"], body["status"]["exit"], body["status"]["pid"]),
                         ("failed", -15, None))
        self.assertFalse(alive(pid))

        status, _, body = post(stack.url, "/jobs/stubborn/start")       # this one needs SIGKILL
        self.assertEqual(status, 200)
        pid = body["status"]["pid"]
        stack.pids.add(pid)
        stack.wait_line(r"^\[stubborn\] ready$")
        t0 = time.monotonic()
        status, _, body = post(stack.url, "/jobs/stubborn/stop")
        took = time.monotonic() - t0
        self.assertEqual((status, body["status"]["state"], body["status"]["exit"]), (200, "failed", -9))
        self.assertGreaterEqual(took, 0.45)
        self.assertFalse(alive(pid))

        status, _, body = post(stack.url, "/jobs/ghost/start")          # its command is not there
        self.assertEqual(status, 500)
        self.assertIn("cannot start ghost", body["error"])
        ghost = stack.status()["jobs"]["ghost"]
        self.assertEqual((ghost["state"], ghost["pid"], ghost["exit"]), ("failed", None, None))
        self.assertIn("cannot start", ghost["error"])

    def test_admin_calls_answer_403_without_admin(self):
        stack = self.start("--serve", named("one", TICK), "--job", named("work", FAILS))        # no --admin
        first = self.both_running(stack)
        for path in ("/servers/one/kill", "/servers/one/stop", "/servers/one/start", "/jobs/work/start",
                     "/jobs/work/stop", "/servers/nobody/kill", "/jobs/nobody/start"):
            status, headers, body = post(stack.url, path)
            self.assertEqual(status, 403, path)
            self.assertIn("--admin", body["error"])
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        seen = stack.status()
        self.assertEqual((seen["servers"]["one"]["state"], seen["servers"]["one"]["pid"],
                          seen["servers"]["one"]["restarts"]), ("running", first["one"], 0))
        self.assertEqual(seen["jobs"]["work"]["state"], "idle")

    def test_unknown_names_and_paths(self):
        stack = self.start("--admin", "--serve", named("one", TICK), "--job", named("work", FAILS))
        first = self.both_running(stack)
        for path in ("/servers/nobody/kill", "/servers/nobody/stop", "/servers/nobody/start", "/jobs/nobody/start",
                     "/jobs/nobody/stop", "/servers/work/kill", "/jobs/one/start", "/servers/one%20/kill",
                     "/servers/../kill"):
            status, headers, body = post(stack.url, path)
            self.assertEqual(status, 404, path)
            self.assertIn("in the config", body["error"])
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        for path in ("/servers/one/explode", "/jobs/work/kill", "/servers/one", "/servers/one/kill/now", "/nothing"):
            status, _, body = post(stack.url, path)
            self.assertEqual(status, 404, path)
            self.assertIn("no such path", body["error"])
        status, _, body = call(stack.url, "/nothing")
        self.assertEqual(status, 404)

        status, headers, body = call(stack.url, "/servers/one/kill")            # a GET changes nothing
        self.assertEqual((status, headers.get("Allow")), (405, "POST"))
        status, headers, body = post(stack.url, "/status")
        self.assertEqual((status, headers.get("Allow")), (405, "GET"))

        status, headers, body = call(stack.url, "/servers/one/kill", method="OPTIONS")  # a browser's preflight
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))

        seen = stack.status()
        self.assertEqual((seen["servers"]["one"]["state"], seen["servers"]["one"]["pid"],
                          seen["servers"]["one"]["restarts"]), ("running", first["one"], 0))
        self.assertEqual(seen["jobs"]["work"]["state"], "idle")


class LeavingTest(StackTest):
    """How the supervisor ends, and what it does when its surroundings are wrong."""

    def test_sigterm_stops_every_child_and_the_exit_code_is_0(self):
        marker = os.path.join(self.tmp, "polite-got-sigterm")
        kid_marker = os.path.join(self.tmp, "kid-got-sigterm")
        never = os.path.join(self.tmp, "never")
        stack = self.start("--admin", "--test-grace", "0.8", "--serve", named("polite", POLITE, marker),
                           "--serve", named("stubborn", STUBBORN), "--serve", named("tree", TREE),
                           "--serve", named("family", FAMILY, kid_marker), "--job", named("work", WORK, never))
        status, _, body = post(stack.url, "/jobs/work/start")
        self.assertEqual(status, 200)
        stack.wait_line(r"^\[polite\] ready$")
        stack.wait_line(r"^\[stubborn\] ready$")
        stack.wait_line(r"^\[family\] ready$")
        stack.wait_line(r"^\[family\] kid ready$")
        kid = int(stack.wait_line(r"^\[tree\] kid (\d+)$").group(1))
        seen = stack.wait_status(lambda s: all(server["state"] == "running" for server in s["servers"].values())
                                 and s["jobs"]["work"]["state"] == "running")
        pids = [server["pid"] for server in seen["servers"].values()] + [seen["jobs"]["work"]["pid"]]
        self.assertEqual(len(set(pids)), 5)
        for pid in pids + [kid]:
            self.assertTrue(alive(pid))

        t0 = time.monotonic()
        stack.proc.send_signal(signal.SIGTERM)
        code = stack.proc.wait(30)
        took = time.monotonic() - t0
        self.assertEqual(code, 0)
        for pid in pids:
            self.assertFalse(alive(pid), "child %d still runs" % pid)
        wait_until(lambda: gone(kid), 10, "the tree server's kid to be gone")
        self.assertTrue(os.path.exists(marker))                 # the polite one got SIGTERM and time to leave
        self.assertTrue(os.path.exists(kid_marker))             # SIGTERM went to the whole group, not one process
        self.assertGreaterEqual(took, 0.75)                     # the stubborn one needed SIGKILL, after the grace
        self.assertLess(took, 8)
        stack.stop()
        self.assertTrue(stack.matches(r"SIGTERM received: stopping every child$"))
        self.assertTrue(stack.matches(r"stubborn: still running 0\.8 s after SIGTERM: SIGKILL sent to pid \d+$"))
        self.assertTrue(stack.matches(r"polite: exited with code 0 after [\d.]+ s; it was asked to stop$"))
        self.assertRegex(stack.lines[-1], r"every child is gone; leaving with code 0$")

    def start_finding(self, signum, how, *flags):
        """Start a supervisor that finds this signal ignored (signal.SIG_IGN, as under nohup) or not
        (signal.SIG_DFL), whatever this test run itself found: a child process inherits that."""
        before = signal.signal(signum, how)
        try:
            return self.start(*flags)
        finally:
            signal.signal(signum, before)

    def test_ctrl_c_and_hang_up_end_it_the_same_way(self):
        # SIGINT ends it even when it was ignored at the start, as for a background job of a script.
        for signum, found in ((signal.SIGINT, signal.SIG_DFL), (signal.SIGINT, signal.SIG_IGN),
                              (signal.SIGHUP, signal.SIG_DFL)):
            stack = self.start_finding(signum, found, "--serve", named("one", TICK))
            pid = self.both_running(stack)["one"]
            stack.proc.send_signal(signum)
            self.assertEqual(stack.proc.wait(30), 0, signum)
            self.assertFalse(alive(pid))
            stack.stop()
            self.assertTrue(stack.matches(r"%s received: stopping every child$" % signal.Signals(signum).name))

    def test_under_nohup_a_hang_up_changes_nothing(self):
        stack = self.start_finding(signal.SIGHUP, signal.SIG_IGN, "--serve", named("one", TICK))
        pid = self.both_running(stack)["one"]
        stack.proc.send_signal(signal.SIGHUP)
        time.sleep(0.7)
        self.assertIsNone(stack.proc.poll())
        one = stack.status()["servers"]["one"]
        self.assertEqual((one["state"], one["pid"], one["restarts"]), ("running", pid, 0))
        stack.proc.send_signal(signal.SIGTERM)                  # SIGTERM still ends it
        self.assertEqual(stack.proc.wait(30), 0)
        self.assertFalse(alive(pid))

    def test_config_file_with_cwd_env_and_logs(self):
        here = os.path.join(self.tmp, "run here")
        os.mkdir(here)
        logs = os.path.join(self.tmp, "logs")
        config = {"servers": {"where": {"cmd": command(WHERE), "cwd": here, "env": {"WX_DATA": "/vol/bundle"}},
                              "plain": {"cmd": command(TICK)}},
                  "jobs": {"work": {"cmd": command(FAILS)}}}
        path = os.path.join(self.tmp, "stack.json")
        with open(path, "w") as f:
            json.dump(config, f)
        stack = self.start("--config", path, "--admin", "--log-dir", logs, "--job", named("extra", FAILS))
        printed = r"cwd=%s data=/vol/bundle" % re.escape(os.path.realpath(here))
        stack.wait_line(r"^\[where\] %s$" % printed)
        stack.wait_line(r"^\[plain\] tick 1$")
        seen = stack.status()
        self.assertEqual(list(seen["servers"]), ["where", "plain"])
        self.assertEqual(list(seen["jobs"]), ["work", "extra"])         # the config's first, then the command line's

        status, _, body = post(stack.url, "/jobs/work/start")
        self.assertEqual(status, 200)
        work = stack.wait_status(lambda s: s["jobs"]["work"]["state"] != "running")["jobs"]["work"]
        self.assertEqual((work["state"], work["exit"], work["last_lines"]), ("failed", 3, ["about to fail"]))

        def log(name):
            with open(os.path.join(logs, name + ".log")) as f:
                return f.read().splitlines()

        wait_until(lambda: "about to fail" in log("work") and "tick 1" in log("plain"), what="the log files")
        self.assertTrue(any(re.fullmatch(printed, line) for line in log("where")))
        self.assertTrue(any(re.fullmatch(r"\[stack A \d\d:\d\d:\d\d\] where: started, pid \d+", line)
                            for line in log("where")))
        self.assertTrue(any(re.search(r"work: exited with code 3 after", line) for line in log("work")))
        self.assertEqual(sorted(os.listdir(logs)), ["extra.log", "plain.log", "where.log", "work.log"])

    def test_open_connections_do_not_delay_leaving(self):
        stack = self.start("--serve", named("one", TICK))
        pid = self.both_running(stack)["one"]
        kept = http.client.HTTPConnection("127.0.0.1", stack.port, timeout=10)  # a browser keeps its connection
        self.addCleanup(kept.close)
        for path in ("/health", "/status"):                     # two requests on the one connection
            kept.request("GET", path)
            response = kept.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
        half = socket.create_connection(("127.0.0.1", stack.port), timeout=10)  # a request that never ends
        self.addCleanup(half.close)
        half.sendall(b"GET /sta")

        t0 = time.monotonic()
        stack.proc.send_signal(signal.SIGTERM)
        self.assertEqual(stack.proc.wait(30), 0)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertFalse(alive(pid))

    def on_a_pipe(self, *flags):
        """Start a supervisor whose stdout is a pipe that the test holds, and read its first line from it.
        Returns (the process, its URL, the pipe's read end, the list to put child pids in for the clean-up)."""
        errors = tempfile.TemporaryFile()
        reading, writing = os.pipe()
        proc = subprocess.Popen([sys.executable, STACK, "--name", "A", "--host", "127.0.0.1", "--port", "0"]
                                + list(flags), cwd=REPO, stdin=subprocess.DEVNULL, stdout=writing, stderr=errors)
        os.close(writing)
        pids = []

        def end():
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            left = [pid for pid in pids if alive(pid)]
            for pid in left:
                os.killpg(pid, signal.SIGKILL)
            errors.seek(0)
            text = errors.read().decode("utf-8", "replace")
            errors.close()
            self.assertEqual(left, [], "children still ran after the supervisor had ended")
            self.assertNotIn("Traceback", text)

        self.addCleanup(end)
        watchdog = threading.Timer(20, proc.kill)               # never wait forever for the first line
        watchdog.start()
        first = b""
        while not first.endswith(b"\n"):
            byte = os.read(reading, 1)
            if not byte:
                break
            first += byte
        watchdog.cancel()
        found = re.fullmatch(r"listening on (http://127\.0\.0\.1:\d+)\n", first.decode("utf-8"))
        if not found:
            os.close(reading)
            self.fail("the supervisor did not start. First line: %r" % first)
        return proc, found.group(1), reading, pids

    def test_lost_stdout_does_not_stop_it(self):
        proc, url, reading, pids = self.on_a_pipe("--serve", named("one", TICK))
        os.close(reading)                                       # the reader of its output is gone

        def one():
            return call(url, "/status")[2]["servers"]["one"]

        pids.append(wait_until(lambda: one()["pid"], what="the server to start"))
        time.sleep(0.7)                                         # it prints all the while, into a broken pipe
        self.assertEqual((one()["state"], one()["pid"], one()["restarts"]), ("running", pids[0], 0))
        self.assertIsNone(proc.poll())

        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(30), 0)
        self.assertFalse(alive(pids[0]))

    def test_stdout_that_nobody_reads_does_not_stop_the_keeping(self):
        # The pipe stays open but is never read. CHATTY fills it, and the supervisor's next write to it hangs.
        # Servers must still be started again, and SIGTERM must still end everything with code 0.
        proc, url, reading, pids = self.on_a_pipe("--test-backoff", "0.2", "--serve", named("chatty", CHATTY),
                                                  "--serve", named("one", TICK))
        self.addCleanup(os.close, reading)

        def servers():
            return call(url, "/status")[2]["servers"]

        first = wait_until(lambda: all(server["pid"] for server in servers().values()) and servers())
        pids.extend(server["pid"] for server in first.values())
        time.sleep(1.0)                                         # by now the pipe is full
        os.kill(first["one"]["pid"], signal.SIGKILL)
        again = wait_until(lambda: servers()["one"]["restarts"] == 1 and servers()["one"]["state"] == "running"
                           and servers(), 20, "the killed server to come back while stdout takes nothing")
        pids.append(again["one"]["pid"])
        self.assertEqual((again["chatty"]["state"], again["chatty"]["restarts"]), ("running", 0))

        proc.send_signal(signal.SIGTERM)
        self.assertEqual(proc.wait(30), 0)
        for pid in pids:
            self.assertFalse(alive(pid), "child %d still runs" % pid)
        printed = b""                                           # what did fit into the pipe
        while True:
            more = os.read(reading, 65536)
            if not more:
                break
            printed += more
        self.assertGreater(len(printed), 4000)
        self.assertNotIn(b"chatty line 2999", printed)          # the pipe did fill up: the test tested something

    def test_port_in_use_ends_it_at_once_with_code_2(self):
        first = self.start("--serve", named("one", TICK))
        marker = os.path.join(self.tmp, "started")
        done = subprocess.run([sys.executable, STACK, "--name", "B", "--host", "127.0.0.1", "--port", str(first.port),
                               "--serve", named("touch", TOUCH, marker)], cwd=REPO, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(done.returncode, 2)
        self.assertIn("cannot listen on 127.0.0.1:%d" % first.port, done.stderr)
        self.assertEqual(done.stdout, "")
        self.assertFalse(os.path.exists(marker))                # it started nothing

    def test_wrong_command_lines_end_with_code_2_and_start_nothing(self):
        marker = os.path.join(self.tmp, "started")
        good = named("touch", TOUCH, marker)
        cases = [(["--name", "A"], "nothing to run"),
                 (["--name", "A", "--serve", good, "--serve", "no-command"], "NAME="),
                 (["--name", "A", "--serve", good, "--serve", good], "given twice"),
                 (["--name", "A", "--serve", good, "--job", good], "a server and for a job"),
                 (["--name", "A", "--serve", good, "--config", os.path.join(self.tmp, "missing.json")],
                  "cannot read the config file"),
                 (["--name", "A", "--serve", good, "--port", "70000"], "--port"),
                 (["--serve", good], "--name")]
        for flags, expected in cases:
            done = subprocess.run([sys.executable, STACK, "--host", "127.0.0.1", "--port", "0"] + flags, cwd=REPO,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=30)
            self.assertEqual(done.returncode, 2, flags)
            self.assertIn(expected, done.stderr)
            self.assertEqual(done.stdout, "")
            self.assertFalse(os.path.exists(marker), flags)


class PiecesTest(unittest.TestCase):
    """The pieces that can be tested without starting a process: the command line, the config file, the waits."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="node_stack_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def config(self, value):
        path = os.path.join(self.tmp, "stack.json")
        with open(path, "w") as f:
            f.write(value if isinstance(value, str) else json.dumps(value))
        return path

    def test_named_command(self):
        name, spec = ns.named_command("forecast=python serve/forecast_server.py --node A --port 8101", "--serve")
        self.assertEqual(name, "forecast")
        self.assertEqual(spec, {"cmd": ["python", "serve/forecast_server.py", "--node", "A", "--port", "8101"],
                                "cwd": None, "env": {}})
        name, spec = ns.named_command("llm=./llama-server -m 'my model.gguf' --alias a=b", "--serve")
        self.assertEqual((name, spec["cmd"]), ("llm", ["./llama-server", "-m", "my model.gguf", "--alias", "a=b"]))
        for text in ("no-equals-sign", "=python x.py", "a b=python x.py", "../x=python x.py", "name=", "name=   ",
                     "name=python 'unclosed", "-x=python x.py", "n" * 41 + "=python x.py"):
            with self.assertRaises(ns.ConfigError, msg=text):
                ns.named_command(text, "--serve")

    def test_config_file(self):
        good = {"servers": {"forecast": {"cmd": ["python", "serve/forecast_server.py", "--node", "A"], "cwd": ".",
                                         "env": {"WX_DATA": "/vol/bundle"}},
                            "llm": {"cmd": ["./llama-server", "-m", "model.gguf"]}},
                "jobs": {"finetune": {"cmd": ["python", "weather/finetune.py"]}}}
        servers, jobs = ns.load_config(self.config(good))
        self.assertEqual(list(servers), ["forecast", "llm"])
        self.assertEqual(servers["forecast"], {"cmd": ["python", "serve/forecast_server.py", "--node", "A"],
                                               "cwd": ".", "env": {"WX_DATA": "/vol/bundle"}})
        self.assertEqual(servers["llm"], {"cmd": ["./llama-server", "-m", "model.gguf"], "cwd": None, "env": {}})
        self.assertEqual(jobs, {"finetune": {"cmd": ["python", "weather/finetune.py"], "cwd": None, "env": {}}})
        self.assertEqual(ns.load_config(self.config({"servers": {"a": {"cmd": ["x"]}}})),
                         ({"a": {"cmd": ["x"], "cwd": None, "env": {}}}, {}))

        bad = ["{not json", "[]", '{"server": {}}', '{"servers": []}', '{"servers": {"a": ["x"]}}',
               '{"servers": {"a": {}}}', '{"servers": {"a": {"cmd": "python x.py"}}}',
               '{"servers": {"a": {"cmd": []}}}', '{"servers": {"a": {"cmd": ["python", 8101]}}}',
               '{"servers": {"a": {"cmd": ["x"], "command": ["y"]}}}', '{"servers": {"a": {"cmd": ["x"], "cwd": 3}}}',
               '{"servers": {"a": {"cmd": ["x"], "env": {"PORT": 8101}}}}', '{"servers": {"a b": {"cmd": ["x"]}}}',
               '{"servers": {"a": {"cmd": ["x"]}, "a": {"cmd": ["y"]}}}', '{"jobs": {"../j": {"cmd": ["x"]}}}']
        for text in bad:
            with self.assertRaises(ns.ConfigError, msg=text):
                ns.load_config(self.config(text))
        with self.assertRaises(ns.ConfigError):
            ns.load_config(os.path.join(self.tmp, "missing.json"))

    def test_table_from_config_and_command_line(self):
        path = self.config({"servers": {"forecast": {"cmd": ["a"]}}, "jobs": {"finetune": {"cmd": ["b"]}}})
        servers, jobs = ns.read_table(path, ["llm=c d"], ["score=e"])
        self.assertEqual([(name, spec["cmd"]) for name, spec in servers.items()],
                         [("forecast", ["a"]), ("llm", ["c", "d"])])
        self.assertEqual([(name, spec["cmd"]) for name, spec in jobs.items()],
                         [("finetune", ["b"]), ("score", ["e"])])
        for serve, job in ((["forecast=x"], []), ([], ["finetune=x"]), (["finetune=x"], []), ([], ["forecast=x"]),
                           (["llm=x", "llm=y"], [])):
            with self.assertRaises(ns.ConfigError, msg=(serve, job)):
                ns.read_table(path, serve, job)
        with self.assertRaises(ns.ConfigError):
            ns.read_table(None, [], [])
        self.assertEqual(ns.read_table(None, [], ["only=x"]), ({}, {"only": {"cmd": ["x"], "cwd": None, "env": {}}}))

    def test_small_helpers(self):
        self.assertEqual(ns.exit_text(0), "exited with code 0")
        self.assertEqual(ns.exit_text(3), "exited with code 3")
        self.assertEqual(ns.exit_text(-9), "was killed by SIGKILL")
        self.assertEqual(ns.exit_text(-15), "was killed by SIGTERM")
        self.assertEqual(ns.seconds_list("1,2,5,10,30", "--test-backoff"), list(ns.BACKOFF_S))
        for text in ("", "1,,2", "one", "-1", "1,nan"):
            with self.assertRaises(ns.ConfigError, msg=text):
                ns.seconds_list(text, "--test-backoff")
        self.assertEqual(ns.admin_path("/servers/forecast/kill"), ("servers", "forecast", "kill"))
        self.assertEqual(ns.admin_path("/jobs/finetune/start"), ("jobs", "finetune", "start"))
        for path in ("/jobs/finetune/kill", "/servers/forecast", "/servers/a/b/kill", "/status", "servers/a/kill"):
            self.assertIsNone(ns.admin_path(path), path)
        self.assertEqual(ns.extra_args(b""), [])
        self.assertEqual(ns.extra_args(b'{"args": ["--steps", "200"], "other": 1}'), ["--steps", "200"])
        self.assertFalse(ns.signal_group(os.getpgrp(), 0))      # never its own group
        self.assertFalse(ns.signal_group(1, 0))
        self.assertFalse(ns.signal_group(0, 0))

    def test_the_wait_before_each_restart(self):
        # The real steps, without waiting for them: 1, 2, 5, 10, 30, 30 seconds for a server that keeps dying
        # young, and from 1 again after a run of 60 seconds.
        stack = ns.Stack("A", {"s": {"cmd": ["x"], "cwd": None, "env": {}}}, {})
        server = stack.servers["s"]
        self.assertEqual([stack.next_wait(server, 0.5) for _ in range(7)], [1.0, 2.0, 5.0, 10.0, 30.0, 30.0, 30.0])
        self.assertEqual(stack.next_wait(server, 59.9), 30.0)
        self.assertEqual([stack.next_wait(server, 60.0), stack.next_wait(server, 3.0)], [1.0, 2.0])
        self.assertEqual((stack.stable_s, stack.grace_s, stack.job_grace_s), (60.0, 10.0, 30.0))

    def test_printing_never_raises(self):
        stack = ns.Stack("A", {"s": {"cmd": ["x"], "cwd": None, "env": {}}}, {})
        server = stack.servers["s"]
        real = sys.stdout
        try:
            sys.stdout = None                       # as in a process that was started with its stdout closed
            stack.say("nobody hears this")
            stack.put("[s] nor this", server, "nor this")
            sys.stdout = heard = io.StringIO()
            stack.say("started, pid 1", server)
            stack.put("[s] a line", server, "a line")
            self.assertTrue(stack.hush())           # after the end nothing is printed any more
            stack.say("too late")
        finally:
            sys.stdout = real
        printed = heard.getvalue().splitlines()
        self.assertEqual(len(printed), 2)
        self.assertRegex(printed[0], r"^\[stack A \d\d:\d\d:\d\d\] s: started, pid 1$")
        self.assertEqual(printed[1], "[s] a line")


class OldStackCopy(unittest.TestCase):
    """The guard against a stack that was started from an old copy of fuzzball/30-real-stack.fz."""
    SLEEPER = "s=" + shlex.join([sys.executable, "-c", "import time; time.sleep(60)"])   # a server that only waits

    def test_the_rule(self):
        self.assertIsNone(ns.old_stack_copy(None, 3, False))        # not on the team volume: never checked
        self.assertIsNone(ns.old_stack_copy(None, None, True))      # the repo's file declares no revision
        self.assertIsNone(ns.old_stack_copy(3, 3, True))
        self.assertIsNone(ns.old_stack_copy(4, 3, True))            # a newer copy than the repo's is let through
        for given in (None, 2):
            why = ns.old_stack_copy(given, 3, True)
            self.assertIn("OLD copy of fuzzball/30-real-stack.fz", why)
            self.assertIn("the repo's file is revision 3", why)
        self.assertIn("revision unknown (older than this check)", ns.old_stack_copy(None, 3, True))
        self.assertIn("revision 2;", ns.old_stack_copy(2, 3, True))

    def test_the_repos_stack_file_declares_one_revision_and_gives_it_to_both_boxes(self):
        rev = ns.stack_file_rev()
        self.assertIsInstance(rev, int)
        with open(ns.STACK_FILE) as f:
            text = f.read()
        self.assertEqual(len(re.findall(r"(?m)^\s*STACK_REV=\d+\s*$", text)), 1)
        self.assertEqual(re.findall(r"(?m)^\s*export WX_STACK_REV=(\d+)\b", text), [str(rev), str(rev)])

    def test_a_missing_file_or_a_file_without_a_revision_is_no_revision(self):
        self.assertIsNone(ns.stack_file_rev(os.path.join(tempfile.gettempdir(), "no-such-stack-file.fz")))
        with tempfile.NamedTemporaryFile("w", suffix=".fz") as f:
            f.write("# STACK_REV=9 in a comment does not count\n      repo_rev=$(sed -n 's/^ *STACK_REV=x/p')\n")
            f.flush()
            self.assertIsNone(ns.stack_file_rev(f.name))

    def run_guarded(self, rev, flags=()):
        """The supervisor as a child process, told that it runs on the team volume. -> (exit code or None, output)"""
        env = dict(os.environ, WX_STACK_GUARD="1")
        env.pop("WX_STACK_REV", None)
        if rev is not None:
            env["WX_STACK_REV"] = str(rev)
        proc = subprocess.Popen([sys.executable, STACK, "--name", "A", "--host", "127.0.0.1", "--port", "0",
                                 "--serve", self.SLEEPER] + list(flags),
                                cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            out, _ = proc.communicate(timeout=4)
            return proc.returncode, out
        except subprocess.TimeoutExpired:          # it started and is serving: that is the other outcome
            proc.terminate()
            out, _ = proc.communicate(timeout=20)
            return None, out

    def test_an_old_copy_stops_at_once_and_a_current_one_starts(self):
        current = ns.stack_file_rev()
        code, out = self.run_guarded(None)
        self.assertEqual(code, ns.EXIT_OLD_STACK, out)
        self.assertIn("STOP: this stack was started from an OLD copy", out)
        self.assertNotIn("listening on", out)
        code, out = self.run_guarded(current - 1)
        self.assertEqual(code, ns.EXIT_OLD_STACK, out)
        code, out = self.run_guarded("not a number")
        self.assertEqual(code, ns.EXIT_OLD_STACK, out)
        code, out = self.run_guarded(current)
        self.assertIsNone(code, out)
        self.assertIn("listening on http://127.0.0.1:", out)
        code, out = self.run_guarded(None, ["--stack-rev", str(current)])      # the flag does the same
        self.assertIsNone(code, out)

    def test_away_from_the_team_volume_nothing_is_checked(self):
        env = dict(os.environ)
        env.pop("WX_STACK_GUARD", None)
        env.pop("WX_STACK_REV", None)
        proc = subprocess.Popen([sys.executable, STACK, "--name", "A", "--host", "127.0.0.1", "--port", "0",
                                 "--serve", self.SLEEPER], cwd=REPO,
                                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        try:
            first = proc.stdout.readline()
            self.assertIn("listening on http://127.0.0.1:", first)
        finally:
            proc.terminate()
            proc.communicate(timeout=20)


if __name__ == "__main__":
    unittest.main()
