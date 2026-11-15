"""
Tests for the assistant of the Control Room: ui/agent.py, and the parts of ui/server.py it leans on.

With --stack the assistant can start and stop a real fine-tune job and really kill a forecast server, so
these tests pin WHEN it does that. A fake Ops object records every call; nothing is sent anywhere.

What they check:
  - the rule-based mode acts only on a plain command about one box: a phrase table of commands that must
    act (on which box, with which tool) and of questions, remarks and "do not ..." lines that must change
    nothing, each for boxes with a supervisor (real actions) and without one (simulated);
  - a command that names two boxes changes nothing and asks which one;
  - "kill / terminate / turn off / pause / cancel the training job" stops the job: it starts none and
    kills no server;
  - a question never changes the routing rule or the preferred box;
  - a stopped fine-tune job reads as stopped (exit code 3 or a signal), a broken one as failed;
  - "recover box X" says so when the supervisor did not answer;
  - the number of steps of a fine-tune start is bounded as on the HTTP endpoint;
  - LLM mode: after a tool has run, an LLM failure does not send the message through the rules again;
    a <think>...</think> block is cut from the reply;
  - a very long "can you please can you please ..." message is read in well under a second.

Run from the repo root:
    python -m unittest tests/test_ui_agent.py -v
ui/agent.py needs httpx, and ui/server.py needs fastapi, uvicorn and pydantic too. Without them the
tests that need them are skipped.

Written at the hackathon (Nov 13-14, 2026).
"""

import asyncio
import importlib.util
import os
import sys
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI2 = os.path.join(ROOT, "ui")

try:
    import httpx
except ImportError:                     # the tests are skipped, see NEEDS_HTTPX
    httpx = None
NEEDS_HTTPX = unittest.skipIf(httpx is None, "httpx is not installed (ui/agent.py imports it)")


def load(name, path, agent=None):
    """Load one file of ui/ under a name of its own. The repo root has a package named `agent` too, so
    ui/server.py's `from agent import ...` is given ui/agent.py for the time of the import only."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    before = sys.modules.get("agent")
    if agent is not None:
        sys.modules["agent"] = agent
    # ui/server.py also imports its neighbours (events.py) and the repo's whatif package. Without the two
    # folders on the path the import failed, S was None, and every server-side test below was skipped.
    added = [p for p in (UI2, ROOT) if p not in sys.path]
    sys.path[:0] = added
    try:
        spec.loader.exec_module(module)
    finally:
        for p in added:
            sys.path.remove(p)
        if agent is not None:
            if before is None:
                sys.modules.pop("agent", None)
            else:
                sys.modules["agent"] = before
    return module


A = load("ui2_agent_under_test", os.path.join(UI2, "agent.py")) if httpx is not None else None
try:
    S = load("ui2_server_under_test", os.path.join(UI2, "server.py"), agent=A) if A is not None else None
except ImportError:                     # no fastapi, uvicorn or pydantic here
    S = None
NEEDS_SERVER = unittest.skipIf(S is None, "ui/server.py needs httpx, fastapi, uvicorn and pydantic")

BOTH = ("box-a", "box-b")


def job_status(state, code=None):
    """A supervisor's view of its finetune job, as far as the assistant reads it."""
    return {"state": state, "pid": None, "seconds": 12.0, "exit": code, "error": None, "last_lines": []}


class FakeOps:
    """Stands in for server.Ops. `calls` holds every call that would change the cluster, in order."""

    def __init__(self, stacks=(), job="idle", stop_gives=None, no_answer=()):
        self.boxes = {"box-a": "http://box-a.test", "box-b": "http://box-b.test"}
        self.stacks = dict((box, "http://%s-stack.test" % box) for box in stacks)
        self.job = job                                              # the fine-tune job's state in the status
        self.stop_gives = stop_gives or job_status("failed", -15)   # what a stop answers
        self.no_answer = no_answer                                  # boxes whose supervisor does not answer
        self.calls = []
        self.status_reads = 0

    async def forecast(self, lat, lon, hours, members=1, place=None):     # the arguments of server.Ops.forecast
        return {"summary": {"min_c": 1.0, "max_c": 2.0, "precip_mm": None, "max_wind_ms": 3.0},
                "low_confidence": hours > 72, "points": [], "members_temp": None, "served_by": "box-a",
                "simulated": True}

    async def cluster_status(self):
        self.status_reads += 1
        boxes = {}
        for name in self.boxes:
            boxes[name] = {"healthy": True, "mode": "normal", "served_share_10s": 0, "simulated": True}
            if name in self.stacks:
                answers = name not in self.no_answer
                boxes[name]["supervisor"] = "ok" if answers else "no answer"
                boxes[name]["finetune"] = ({"state": self.job, "step": None, "total": None, "speed": None,
                                            "exit": None} if answers else None)
        return {"policy": "adaptive", "active": "box-b", "preferred": "box-b", "rps": 0.0, "p95_ms": None,
                "errors_10s": 0, "boxes": boxes, "recent_switches": []}

    async def set_config(self, update):
        self.calls.append(("set_config", update))
        if "preferred" in update and update["preferred"] not in ("", *self.boxes):
            return {"error": f"unknown backend {update['preferred']!r}", "status": 422}
        return {"policy": update.get("policy", "adaptive"), "preferred": update.get("preferred"), "active": "box-b"}

    async def set_box_mode(self, box, mode, slow_factor=2.0):
        self.calls.append(("simulate", box, mode))
        return {"box": box, "mode": mode, "kind": "simulated"}

    async def finetune(self, box, action, steps=None):
        self.calls.append((action + "_finetune", box, steps))
        status = job_status("running") if action == "start" else self.stop_gives
        return {"action": action, "job": "finetune", "status": status, "kind": "real job"}

    async def kill_forecast(self, box):
        self.calls.append(("kill_forecast_server", box))
        return {"action": "kill", "server": "forecast", "status": {"state": "backoff"}, "kind": "real job"}


def ask(text, ops, **llm):
    """One chat message -> the assistant's answer."""
    agent = A.Agent(ops, llm.get("base_url"), llm.get("model"))
    return asyncio.run(agent.chat([{"role": "user", "content": text}]))


def tools_of(answer):
    return [step["tool"] for step in answer["steps"]]


# What a command must do, for a box with a supervisor and for one without:
#   start   = the training job starts:  start_finetune                 / simulate_box slow
#   stop    = the training job stops:   stop_finetune                  / simulate_box normal
#   kill    = the forecast server dies: kill_forecast_server           / simulate_box error
#   recover = nothing to undo by hand:  only a status read (job idle)  / simulate_box normal
REAL_CALL = {"start": lambda box: [("start_finetune", box, None)],
             "stop": lambda box: [("stop_finetune", box, None)],
             "kill": lambda box: [("kill_forecast_server", box)],
             "recover": lambda box: []}
REAL_TOOLS = {"start": ["start_finetune"], "stop": ["stop_finetune"], "kill": ["kill_forecast_server"],
              "recover": ["get_cluster_status"]}
SIM_MODE = {"start": "slow", "stop": "normal", "kill": "error", "recover": "normal"}

COMMANDS = [
    # the chips on the page, and the README's phrases
    ("Start a training job on box B", "start", "box-b"),
    ("Fail box A", "kill", "box-a"),
    ("Recover box B", "recover", "box-b"),
    ("Stop the training job on box B", "stop", "box-b"),
    # polite words before the verb
    ("Please start a training job on box B", "start", "box-b"),
    ("Can you fail box A?", "kill", "box-a"),
    ("Can you please fail box A?", "kill", "box-a"),
    ("Please can you fail box A", "kill", "box-a"),
    ("Hey, can you please kill box A", "kill", "box-a"),
    ("Could you please stop the fine-tune on box B?", "stop", "box-b"),
    ("Hey, could you please stop the fine-tune on box B?", "stop", "box-b"),
    ("Will you please recover box B?", "recover", "box-b"),
    ("Hey, kill box A", "kill", "box-a"),
    ("OK, fail box A", "kill", "box-a"),
    ("Okay now kill box B", "kill", "box-b"),
    ("Just kill box A", "kill", "box-a"),
    ("Go ahead and kill box A", "kill", "box-a"),
    ("Now recover box B", "recover", "box-b"),
    ("FAIL BOX-A!", "kill", "box-a"),
    ("fail node a", "kill", "box-a"),
    # the forecast server
    ("Kill the forecast server on box A", "kill", "box-a"),
    ("Kill the forecast server on box A while the job runs", "kill", "box-a"),
    ("Stop box B", "kill", "box-b"),
    ("Take box A down", "kill", "box-a"),
    ("Take down box A", "kill", "box-a"),
    ("Shut down box A", "kill", "box-a"),
    ("Crash box A", "kill", "box-a"),
    ("Make box A fail", "kill", "box-a"),
    # back to normal
    ("Bring box A back", "recover", "box-a"),
    ("Bring box A back up", "recover", "box-a"),
    ("Restore box B", "recover", "box-b"),
    ("Make box B healthy again", "recover", "box-b"),
    # starting the job
    ("Start a training job on box B in the background", "start", "box-b"),
    ("Start a fine-tune on box A, I will check back later", "start", "box-a"),
    ("Run the fine-tune on box A", "start", "box-a"),
    ("Fine-tune box B", "start", "box-b"),
    ("Fine-tune the model on box B", "start", "box-b"),
    ("Turn on the training on box B", "start", "box-b"),
    ("Put box B in slow mode", "start", "box-b"),
    ("Make box B slow", "start", "box-b"),
    ("Slow down box B", "start", "box-b"),
    ("Set box B to slow so adaptive moves away", "start", "box-b"),     # not a change of the routing rule
]

# Finding 5: a stop word aimed at the job stops the job. It starts none and kills no server.
STOP_THE_JOB = [
    ("Stop the training job on box B", "box-b"),
    ("Kill the training job on box B", "box-b"),
    ("Terminate the fine-tune on box B", "box-b"),
    ("Turn off the training on box B", "box-b"),
    ("Pause the training job on box B", "box-b"),
    ("Cancel the training job on box B", "box-b"),
    ("Stop the job on box B", "box-b"),
    ("Stop the fine tune on box B", "box-b"),
    ("Pause the fine-tune on box B", "box-b"),
    ("Stop training box B", "box-b"),
    ("End the fine-tune job on box A", "box-a"),
    ("Stop the real fine-tune job on box B", "box-b"),
    ("Abort the run on box B", "box-b"),
    ("Halt the training on box B", "box-b"),
    ("Quit the training job on box B", "box-b"),
    ("Shut down the fine-tune on box B", "box-b"),
    ("Switch off the training job on box B", "box-b"),
    ("Stop slow mode on box B", "box-b"),
    ("Turn off slow mode on box B", "box-b"),
    ("Set box B back to normal, it is too slow", "box-b"),
    ("Put box B back to normal, it is slow", "box-b"),
]

# Messages that name a box and must change nothing: they get the status.
NOT_COMMANDS = [
    # named in the review
    "Hey, is box A down?", "Hey, is box B down?", "Box A down?", "Box A is down?", "Tell me if box A is down",
    "I think box A is down", "Can you tell me if box B is down?", "Can you check if box A is down?",
    "Do not kill box A", "Don't kill box A", "Do we have a training job on box B?",
    "Check whether box B is still training", "box B looks slow, what is going on?",
    "Please show the status of box A, is it down?", "Show me the status of the training job on box B",
    "Could you tell me whether box B is training?", "I think box B looks slow",
    "Do not start training on box B", "Don't start training on box B", "box B should be back to normal soon",
    "Fine-tune status on box B?", "Slow box B?", "Take a look at box A, is it down?",
    "Make sure box A is not down", "Is box A down?", "Is box B slow?", "Why is box B slow?",
    # from the reviewers' phrase lists
    "So is box A down now?", "Did box A crash?", "Any idea why box B is slow?", "Could box A be down?",
    "Show me if the fine-tune on box B is done", "Is the fine-tune on box B finished?",
    "Fine-tune progress on box B?", "Fine-tune job on box B, is it done",
    "Take a look at box A, I think it is down", "Let's make sure box B is not slow",
    "Let's see if box A is down", "Has box A recovered?", "Box A is back, right?",
    "Did the kill on box A work?", "So box B is training now?", "And box A, is it down too?",
    "Then why is box B slow?", "Now is box A down?", "Please tell me if box A is down",
    "Would box B be slow if we train?", "Will box A go down if we kill it?", "box b training?",
    "If I kill box A, what happens?", "Don't fail box A", "Never stop box A", "No need to kill box A",
    "Never mind box A being down", "The training on box B, has it stopped?", "Looks like box A went down",
    "box A is failing, what now", "Just checking, is box A down?", "Stop. Is box A down?",
    "Will you kill box A if I say so?", "Fail box A?", "Box B is slow, bring it back to normal",
    "Restart box A", "Finish the training on box B",
    # a command has to begin with its verb
    "I want to fail box A", "Let's fail box A", "Box B: start a training job", "On box B, start a training job",
    # the verb is aimed at something else, or the words do not say what to do
    "Fail over to box A", "Stop sending traffic to box A", "Stop worrying about box A",
    "Stop, box A is down", "Kill time while box A warms up", "Fine-tune status on box B",
    "Take a note that box B is training", "Put the training job on box B on hold",
    "Kill box B while it is training", "Start box A", "Turn off box B", "Use box B for training",
    "Set box A as the failover box",
    # a condition the rules cannot check or wait for, and a second question after a request
    "Kill box A if it is slow", "Stop the training job on box B when it reaches step 100",
    "Can you kill box A or is that dangerous?", "Start a training job on box B unless it is busy",
]

# Messages that name no box: nothing to act on.
NO_BOX = ["bring the box back", "recover the box", "kill a box", "fail the box", "start a training job",
          "Fail both boxes", "which box is slow?", "which box are slow", "Which box is serving right now?"]

# Finding 2: a command that names two boxes changes nothing and asks which one.
TWO_BOX_COMMANDS = [
    "Kill box B and show that box A takes over",
    "Fail box B so traffic moves to box A",
    "Start a training job on box B and watch box A take the traffic",
    "Stop the training on box B so box A can go home",
    "Recover box B, leave box A alone",
    "Fail box A and box B",
    "Take box B down so box A serves",
    "Use adaptive and prefer box B over box A",
]
# Two boxes, and not a command (or not a clear one): the status, and nothing changes.
TWO_BOX_OTHER = [
    "Show that box A takes over when box B is killed",
    "Let box A take over: kill box B",
    "Move the traffic to box A by starting a training job on box B",
    "Is box B slower than box A?",
    "Fail box A and start training on box B",
]

# Finding 6: only a command changes the routing rule or the preferred box.
ROUTING_COMMANDS = [
    ("Switch to round robin", ("round_robin", None)),
    ("Set the policy to static", ("static", None)),
    ("Change the routing rule to balanced", ("balanced", None)),
    ("Can you switch to least loaded?", ("least_loaded", None)),
    ("Use adaptive with box A preferred", ("adaptive", "box-a")),
    ("Switch to adaptive", ("adaptive", "box-b")),
    ("Use adaptive instead of static", ("adaptive", "box-b")),
    ("Use static routing so box B stops getting traffic", ("static", None)),            # and no kill of box B
    ("Switch to round robin so box A and box B share the load", ("round_robin", None)),
]
ROUTING_QUESTIONS = [
    "Why did adaptive switch to box A?", "Is the policy adaptive?", "How does the adaptive rule work?",
    "What is the routing rule, static or adaptive?", "Which policy is in use, round robin or adaptive?",
    "Should we switch to round robin?", "Switch to static?", "Don't switch to static",
    "Is balanced better than adaptive with box A preferred?", "Can you tell me if the rule is static?",
    "Switch from static to adaptive",       # two rules named: not clear, so nothing
]


@NEEDS_HTTPX
class TestCommandsAct(unittest.TestCase):
    """A plain command about one box acts on that box, with the tool that fits the box."""

    def test_on_boxes_with_a_supervisor_the_action_is_real(self):
        for text, action, box in COMMANDS:
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                answer = ask(text, ops)
                self.assertEqual(ops.calls, REAL_CALL[action](box))
                self.assertEqual(tools_of(answer), REAL_TOOLS[action])
                self.assertEqual(answer["mode"], "offline")

    def test_on_boxes_without_a_supervisor_the_action_is_simulated(self):
        for text, action, box in COMMANDS:
            with self.subTest(text=text):
                ops = FakeOps()
                answer = ask(text, ops)
                self.assertEqual(ops.calls, [("simulate", box, SIM_MODE[action])])
                self.assertEqual(tools_of(answer), ["simulate_box"])

    def test_each_box_gets_its_own_kind_when_only_one_has_a_supervisor(self):
        ops = FakeOps(stacks=("box-b",))
        ask("Fail box A", ops)
        ask("Fail box B", ops)
        ask("Start a training job on box A", ops)
        ask("Start a training job on box B", ops)
        self.assertEqual(ops.calls, [("simulate", "box-a", "error"), ("kill_forecast_server", "box-b"),
                                     ("simulate", "box-a", "slow"), ("start_finetune", "box-b", None)])

    def test_the_reply_says_real_or_simulated(self):
        self.assertIn("**real**", ask("Fail box A", FakeOps(stacks=BOTH))["reply"])
        self.assertIn("**real**", ask("Start a training job on box B", FakeOps(stacks=BOTH))["reply"])
        self.assertIn("**Simulated**", ask("Fail box A", FakeOps())["reply"])
        self.assertIn("**simulated**", ask("Start a training job on box B", FakeOps())["reply"])

    def test_recover_stops_a_running_real_job(self):
        ops = FakeOps(stacks=BOTH, job="running")
        answer = ask("Recover box B", ops)
        self.assertEqual(ops.calls, [("stop_finetune", "box-b", None)])
        self.assertEqual(tools_of(answer), ["get_cluster_status", "stop_finetune"])

    def test_simulate_works_only_where_nothing_is_real(self):
        cases = [("Simulate a slow box B", "box-b", "slow"), ("Simulate a training job on box B", "box-b", "slow"),
                 ("Simulate a failure on box A", "box-a", "error"), ("Simulate a failed box A", "box-a", "error"),
                 ("Start a simulated training job on box B", "box-b", "slow")]
        for text, box, mode in cases:
            with self.subTest(text=text):
                ops = FakeOps()
                ask(text, ops)
                self.assertEqual(ops.calls, [("simulate", box, mode)])
                ops = FakeOps(stacks=BOTH)          # with a supervisor "simulate ..." must not do the real thing
                answer = ask(text, ops)
                self.assertEqual(ops.calls, [])
                self.assertIn("Nothing was changed", answer["reply"])


@NEEDS_HTTPX
class TestStopNeverStarts(unittest.TestCase):

    def test_with_a_supervisor_the_job_is_stopped_and_nothing_else(self):
        for text, box in STOP_THE_JOB:
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH, job="running")
                answer = ask(text, ops)
                self.assertEqual(ops.calls, [("stop_finetune", box, None)])
                self.assertEqual(tools_of(answer), ["stop_finetune"])

    def test_without_a_supervisor_the_simulated_job_is_ended(self):
        for text, box in STOP_THE_JOB:
            with self.subTest(text=text):
                ops = FakeOps()
                ask(text, ops)
                self.assertEqual(ops.calls, [("simulate", box, "normal")])


@NEEDS_HTTPX
class TestOnlyCommandsAct(unittest.TestCase):
    """A question, a remark, a "do not ..." or an unclear line changes nothing, whatever the boxes are."""

    def kinds(self):
        return [("with a supervisor", BOTH), ("without a supervisor", ()), ("one of each", ("box-b",))]

    def test_a_message_that_names_a_box_gets_the_status_and_changes_nothing(self):
        for text in NOT_COMMANDS:
            for kind, stacks in self.kinds():
                with self.subTest(text=text, boxes=kind):
                    ops = FakeOps(stacks=stacks, job="running")
                    answer = ask(text, ops)
                    self.assertEqual(ops.calls, [])
                    self.assertEqual(tools_of(answer), ["get_cluster_status"])
                    self.assertIn("active box", answer["reply"])

    def test_a_message_that_names_no_box_changes_nothing(self):
        for text in NO_BOX:
            for kind, stacks in self.kinds():
                with self.subTest(text=text, boxes=kind):
                    ops = FakeOps(stacks=stacks, job="running")
                    answer = ask(text, ops)
                    self.assertEqual(ops.calls, [])
                    self.assertLessEqual(set(tools_of(answer)), {"get_cluster_status"})

    def test_a_remark_is_told_that_nothing_was_changed_and_a_question_is_not(self):
        self.assertIn("Nothing was changed", ask("I want to fail box A", FakeOps(stacks=BOTH))["reply"])
        self.assertIn("Nothing was changed", ask("Do not kill box A", FakeOps(stacks=BOTH))["reply"])
        self.assertNotIn("Nothing was changed", ask("Hey, is box A down?", FakeOps(stacks=BOTH))["reply"])

    def test_a_forecast_question_is_still_a_forecast(self):
        for text in ("Weather in Chicago next 48 hours", "Ensemble range for Tokyo, 3 days"):
            ops = FakeOps(stacks=BOTH)
            answer = ask(text, ops)
            self.assertEqual(ops.calls, [])
            self.assertEqual(tools_of(answer), ["geocode", "get_forecast"])

    def test_only_the_last_user_message_counts(self):
        ops = FakeOps(stacks=BOTH)
        agent = A.Agent(ops, None, None)
        asyncio.run(agent.chat([{"role": "user", "content": "Fail box A"},
                                {"role": "assistant", "content": "Killed the forecast server on box-a."},
                                {"role": "user", "content": "Is box A down?"}]))
        self.assertEqual(ops.calls, [])


@NEEDS_HTTPX
class TestTwoBoxes(unittest.TestCase):

    def test_a_command_that_names_two_boxes_changes_nothing_and_asks(self):
        for text in TWO_BOX_COMMANDS:
            for stacks in (BOTH, (), ("box-b",)):
                with self.subTest(text=text, stacks=stacks):
                    ops = FakeOps(stacks=stacks, job="running")
                    answer = ask(text, ops)
                    self.assertEqual(ops.calls, [])
                    self.assertEqual(answer["steps"], [])
                    self.assertIn("box-a and box-b", answer["reply"])
                    self.assertIn("Nothing was changed", answer["reply"])

    def test_two_boxes_in_a_question_or_a_remark_get_the_status(self):
        for text in TWO_BOX_OTHER:
            for stacks in (BOTH, ()):
                with self.subTest(text=text, stacks=stacks):
                    ops = FakeOps(stacks=stacks, job="running")
                    answer = ask(text, ops)
                    self.assertEqual(ops.calls, [])
                    self.assertEqual(tools_of(answer), ["get_cluster_status"])


@NEEDS_HTTPX
class TestRoutingRule(unittest.TestCase):

    def test_a_command_changes_the_rule(self):
        for text, (policy, preferred) in ROUTING_COMMANDS:
            for stacks in (BOTH, ()):
                with self.subTest(text=text, stacks=stacks):
                    ops = FakeOps(stacks=stacks)
                    answer = ask(text, ops)
                    want = {"policy": policy, **({"preferred": preferred} if preferred else {})}
                    self.assertEqual(ops.calls, [("set_config", want)])
                    self.assertEqual(tools_of(answer), ["set_routing_config"])

    def test_a_question_never_changes_the_rule_or_the_preferred_box(self):
        for text in ROUTING_QUESTIONS:
            for stacks in (BOTH, ()):
                with self.subTest(text=text, stacks=stacks):
                    ops = FakeOps(stacks=stacks)
                    answer = ask(text, ops)
                    self.assertEqual(ops.calls, [])
                    self.assertEqual(tools_of(answer), ["get_cluster_status"])


@NEEDS_HTTPX
class TestStoppedIsNotFailed(unittest.TestCase):
    """weather/finetune.py leaves with exit code 3 after a stop; the supervisor calls that "failed"."""

    def stop_reply(self, status):
        ops = FakeOps(stacks=BOTH, job="running", stop_gives=status)
        answer = ask("Stop the training job on box B", ops)
        self.assertEqual(ops.calls, [("stop_finetune", "box-b", None)])
        return answer["reply"]

    def test_exit_code_3_reads_as_stopped(self):
        reply = self.stop_reply(job_status("failed", 3))
        self.assertIn("stopped", reply)
        self.assertNotIn("failed", reply)

    def test_a_signal_death_reads_as_stopped(self):
        for code in (-15, -9):
            reply = self.stop_reply(job_status("failed", code))
            self.assertIn("stopped", reply)
            self.assertNotIn("failed", reply)

    def test_a_real_failure_still_reads_as_failed(self):
        for code in (1, 2, None):
            self.assertIn("failed", self.stop_reply(job_status("failed", code)))

    def test_done_and_idle_are_said_as_they_are(self):
        self.assertIn("done", self.stop_reply(job_status("done", 0)))
        self.assertIn("not running", self.stop_reply(job_status("idle")))

    def test_recover_reports_a_stop_with_exit_code_3_as_stopped(self):
        ops = FakeOps(stacks=BOTH, job="running", stop_gives=job_status("failed", 3))
        reply = ask("Recover box B", ops)["reply"]
        self.assertIn("stopped", reply)
        self.assertNotIn("failed", reply)

    def test_was_stopped(self):
        for job, expected in [(job_status("failed", 3), True), (job_status("failed", -15), True),
                              (job_status("failed", -9), True), (job_status("failed", 1), False),
                              (job_status("failed", 2), False), (job_status("failed", None), False),
                              (job_status("failed", True), False), (job_status("done", 0), False),
                              (job_status("running", None), False), (job_status("idle", 3), False),
                              (None, False), ("failed", False)]:
            with self.subTest(job=job):
                self.assertIs(A.was_stopped(job), expected)

    def test_the_llm_is_told_how_to_read_a_stop(self):
        tools = dict((t["function"]["name"], t["function"]) for t in A.build_tools(list(BOTH), list(BOTH)))
        self.assertIn("exit 3", tools["stop_finetune"]["description"])
        self.assertIn("stopped", tools["get_cluster_status"]["description"])


@NEEDS_HTTPX
class TestRecoverWithoutSupervisorAnswer(unittest.TestCase):

    def test_it_says_that_the_supervisor_did_not_answer(self):
        ops = FakeOps(stacks=BOTH, job="running", no_answer=("box-b",))
        answer = ask("Recover box B", ops)
        self.assertEqual(ops.calls, [])
        self.assertIn("supervisor of box-b did not answer", answer["reply"])
        self.assertNotIn("no real fine-tune job is running", answer["reply"])
        self.assertNotIn("restarts the forecast server", answer["reply"])

    def test_the_other_box_is_still_answered_from_what_was_read(self):
        ops = FakeOps(stacks=BOTH, no_answer=("box-b",))
        self.assertIn("no real fine-tune job is running", ask("Recover box A", ops)["reply"])

    def test_the_status_reply_names_the_unreachable_supervisor(self):
        reply = ask("Is box B down?", FakeOps(stacks=BOTH, job="running", no_answer=("box-b",)))["reply"]
        self.assertIn("box-b healthy (normal, simulated, supervisor not reachable)", reply)
        self.assertIn("box-a healthy (normal, simulated, real fine-tune job running)", reply)


@NEEDS_HTTPX
class TestStepsAreBounded(unittest.TestCase):
    """The assistant's `steps` has the bound of POST /api/box/{box}/train: a whole number from 1 to 1000000."""

    def start(self, steps):
        ops = FakeOps(stacks=BOTH)
        step = asyncio.run(A.Agent(ops, None, None).run_tool("start_finetune", {"box": "box-b", "steps": steps}))
        return ops.calls, step["result"]

    def test_a_number_out_of_bounds_starts_nothing(self):
        for steps in (0, -1, -5, 1000001, 99999999999, 2.5, float("inf"), float("nan"), "lots", "2.5", True, [200]):
            with self.subTest(steps=steps):
                calls, result = self.start(steps)
                self.assertEqual(calls, [])
                self.assertIn("steps must be a whole number from 1 to 1000000", result["error"])

    def test_a_number_within_bounds_is_passed_on(self):
        for steps, passed in ((1, 1), (200, 200), (1000000, 1000000), ("200", 200), (200.0, 200), (None, None)):
            with self.subTest(steps=steps):
                calls, result = self.start(steps)
                self.assertEqual(calls, [("start_finetune", "box-b", passed)])
                self.assertNotIn("error", result)

    def test_the_rule_based_mode_passes_the_number_and_takes_the_refusal(self):
        ops = FakeOps(stacks=BOTH)
        self.assertIn("(200 steps)", ask("Start a training job on box B with 200 steps", ops)["reply"])
        self.assertEqual(ops.calls, [("start_finetune", "box-b", 200)])
        for text in ("Start training on box B for 5000000 steps", "Start a fine-tune on box B with 0 steps"):
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                reply = ask(text, ops)["reply"]
                self.assertEqual(ops.calls, [])
                self.assertIn("did not start", reply)

    def test_the_tool_schema_carries_the_bound(self):
        tools = dict((t["function"]["name"], t["function"]) for t in A.build_tools(list(BOTH), list(BOTH)))
        self.assertEqual(tools["start_finetune"]["parameters"]["properties"]["steps"],
                         {"type": "integer", "minimum": 1, "maximum": 1000000})


@NEEDS_HTTPX
class TestLongMessageIsFast(unittest.TestCase):
    """A pattern with an optional "please" inside the "can you" part needs twice the time for each more
    "can you please": 24 of them took 8 s. The lengths grow, so such a pattern fails here early."""

    LIMIT_S = 0.5

    def grow(self, counts):
        """Time the reading of longer and longer messages. A pattern that doubles its time fails at about
        22 repetitions, long before a message that it would never finish."""
        for count in counts:
            for tail in ("tell me about box a", "fail box a", "is the policy adaptive?"):
                text = "can you please " * count + tail
                t0 = time.perf_counter()
                A.command_verb(text)
                took = time.perf_counter() - t0
                self.assertLess(took, self.LIMIT_S, "%d x 'can you please' took %.2f s" % (count, took))

    def test_can_you_please_many_times(self):
        self.grow((4, 8, 12, 16, 18, 20, 22, 24, 200, 5000))

    def test_a_whole_chat_turn_with_a_very_long_message(self):
        self.grow((4, 8, 12, 16, 18, 20, 22, 24))       # first: fail here rather than hang below
        for text, calls in (("can you please " * 5000 + "tell me about box a", []),
                            ("Can you please " * 5000 + "fail box A", [("kill_forecast_server", "box-a")]),
                            ("ok " * 20000 + "is box a down", []),
                            ("stop " + "the " * 20000 + "training job on box b", [("stop_finetune", "box-b", None)]),
                            ("fail" + " " * 100000 + "box a", [("kill_forecast_server", "box-a")])):
            ops = FakeOps(stacks=BOTH)
            t0 = time.perf_counter()
            ask(text, ops)
            took = time.perf_counter() - t0
            self.assertLess(took, self.LIMIT_S, "a message of %d characters took %.2f s" % (len(text), took))
            self.assertEqual(ops.calls, calls)


def tool_call(name, **args):
    """An LLM answer that asks for one tool call."""
    import json
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def says(text):
    """An LLM answer that is the final reply."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class FakeLLM:
    """Stands in for httpx.AsyncClient: each POST gets the next item of a script. An item is a JSON body,
    an HTTP status (an int) or an exception to raise. Nothing is sent anywhere."""

    def __init__(self, *script):
        self.script = list(script)
        self.posts = []

    def __call__(self, *args, **kwargs):            # httpx.AsyncClient(timeout=120)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self.posts.append(kwargs.get("json"))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        request = httpx.Request("POST", url)
        if isinstance(item, int):
            return httpx.Response(item, request=request, text="the model could not parse that")
        if isinstance(item, str):
            return httpx.Response(200, request=request, text=item)      # not JSON
        return httpx.Response(200, request=request, json=item)


@NEEDS_HTTPX
class TestLLMMode(unittest.TestCase):

    def ask_llm(self, text, ops, *script):
        llm = FakeLLM(*script)
        with mock.patch.object(A.httpx, "AsyncClient", llm):
            answer = ask(text, ops, base_url="http://llm.test/v1", model="stub")
        return answer, llm

    def test_a_tool_call_runs_and_the_reply_is_the_models(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm("Fail box A", ops, tool_call("kill_forecast_server", box="box-a"),
                                   says("Killed the forecast server on box-a (a real kill)."))
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])
        self.assertEqual(answer["reply"], "Killed the forecast server on box-a (a real kill).")
        self.assertEqual(answer["mode"], "llm:stub")
        self.assertNotIn("note", answer)
        self.assertEqual(len(llm.posts), 2)

    def test_a_failure_before_any_tool_falls_back_to_the_rules(self):
        failures = [httpx.ConnectError("refused"), 500, "<html>not json</html>", {"choices": []}, {"error": "x"},
                    [], {"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}]
        for failure in failures:
            with self.subTest(failure=failure):
                ops = FakeOps(stacks=BOTH)
                answer, _ = self.ask_llm("Fail box A", ops, failure)
                self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])    # once, by the rules
                self.assertIn("answered offline", answer["note"])
                self.assertEqual(tools_of(answer), ["kill_forecast_server"])

    def test_a_failure_after_a_kill_does_not_kill_again(self):
        failures = [500, httpx.ReadTimeout("slow"), httpx.ConnectError("gone"), "not json", {"choices": []}]
        for failure in failures:
            with self.subTest(failure=failure):
                ops = FakeOps(stacks=BOTH)
                answer, llm = self.ask_llm("Fail box A", ops, tool_call("kill_forecast_server", box="box-a"),
                                           failure)
                self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])    # once, not twice
                self.assertEqual(tools_of(answer), ["kill_forecast_server"])        # the step that ran is shown
                self.assertIn("kill_forecast_server on box-a: done", answer["reply"])
                self.assertIn("LLM", answer["reply"])
                self.assertIn("LLM error", answer["note"])
                self.assertNotIn("offline", answer["note"])
                self.assertEqual(len(llm.posts), 2)

    def test_a_failure_after_a_start_does_not_report_that_it_did_not_start(self):
        ops = FakeOps(stacks=BOTH)
        answer, _ = self.ask_llm("please start the fine-tune on box B", ops,
                                 tool_call("start_finetune", box="box-b", steps=50), 500)
        self.assertEqual(ops.calls, [("start_finetune", "box-b", 50)])
        self.assertIn("start_finetune on box-b: done", answer["reply"])
        self.assertNotIn("did not start", answer["reply"])

    def test_a_failure_after_a_read_only_tool_does_not_run_the_rules_either(self):
        ops = FakeOps(stacks=BOTH)
        answer, _ = self.ask_llm("Fail box A", ops, tool_call("get_cluster_status"), 500)
        self.assertEqual(ops.calls, [])
        self.assertEqual(tools_of(answer), ["get_cluster_status"])

    def test_a_failed_step_is_named_as_failed(self):
        ops = FakeOps(stacks=("box-b",))        # box-a has no supervisor: nothing is real there
        answer, _ = self.ask_llm("Slow box B", ops, tool_call("simulate_box", box="box-b", mode="slow"), 500)
        self.assertEqual(ops.calls, [])
        self.assertIn("simulate_box on box-b: failed (box-b has a supervisor", answer["reply"])

    def test_a_think_block_is_cut_from_the_reply(self):
        ops = FakeOps(stacks=BOTH)
        answer, _ = self.ask_llm("Is box A down?", ops, tool_call("get_cluster_status"),
                                 says("<think>\nThe user asks about box-a. It is healthy.\n</think>\n\nBox A is healthy."))
        self.assertEqual(answer["reply"], "Box A is healthy.")
        self.assertEqual(ops.calls, [])

    def test_strip_think(self):
        for text, expected in [
                ("Box A is healthy.", "Box A is healthy."),
                ("<think>plan</think>Box A is healthy.", "Box A is healthy."),
                ("<think>one</think>First. <THINK>two\nlines</THINK>Second.", "First. Second."),
                ("reasoning without its opening tag</think>\nBox A is healthy.", "Box A is healthy."),
                ("Box A is healthy.<think>and then the model was cut off", "Box A is healthy."),
                ("<think>only thinking, never closed", ""),
                ("", "")]:
            with self.subTest(text=text):
                self.assertEqual(A.strip_think(text), expected)

    def test_steps_out_of_bounds_from_the_llm_start_nothing(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm("Start a training job on box B", ops, tool_call("start_finetune", box="box-b", steps=-5),
                                   says("That number of steps is not allowed."))
        self.assertEqual(ops.calls, [])
        self.assertIn("steps must be a whole number", answer["steps"][0]["result"]["error"])
        self.assertIn("steps must be a whole number", llm.posts[1]["messages"][-1]["content"])  # the model is told


@NEEDS_HTTPX
class TestLLMGate(unittest.TestCase):
    """In LLM mode the model chooses the tools. A tool that changes something runs only when the user's own
    last message is a plain command for it: the rule of the rule-based mode, applied to the model's choice."""

    NOT_COMMANDS = ["Hey, is box A down?", "What happens if box A dies?", "Do not kill box A",
                    "If I kill box A, what happens?", "Should I fail box A?", "Fail box A?",
                    "I think box A should be killed", "kill box A if it is slow", "Here is a log line: kill box A",
                    "Which rule is best?", "Tell me about round robin", "Should we switch to static?"]
    CHANGES = [("kill_forecast_server", {"box": "box-a"}), ("start_finetune", {"box": "box-a", "steps": 50}),
               ("stop_finetune", {"box": "box-a"}), ("set_routing_config", {"policy": "static"}),
               ("set_routing_policy", {"policy": "static"})]

    def ask_llm(self, text, ops, *script):
        llm = FakeLLM(*script)
        with mock.patch.object(A.httpx, "AsyncClient", llm):
            answer = ask(text, ops, base_url="http://llm.test/v1", model="stub")
        return answer, llm

    def refused(self, text, ops, tool, args):
        answer, llm = self.ask_llm(text, ops, tool_call(tool, **args), says("Nothing was changed."))
        self.assertEqual(ops.calls, [], (text, tool))
        step = answer["steps"][0]
        self.assertIs(step["result"].get("not_run"), True, (text, tool))
        self.assertIn("not run", step["result"]["error"])
        self.assertIn("not run", llm.posts[1]["messages"][-1]["content"])      # the model is told why
        self.assertEqual(answer["reply"], "Nothing was changed.")
        return step["result"]["error"]

    def test_a_request_to_simulate_never_runs_the_real_action(self):
        for text, tool, args in (("Simulate a failure on box A", "kill_forecast_server", {"box": "box-a"}),
                                 ("Simulate a training job on box B", "start_finetune", {"box": "box-b"}),
                                 ("Start a fake training job on box B", "start_finetune", {"box": "box-b"}),
                                 ("Kill box A, but only pretend", "kill_forecast_server", {"box": "box-a"}),
                                 ("Kill box A (sim)", "kill_forecast_server", {"box": "box-a"}),
                                 ("Fail box A in sim mode", "kill_forecast_server", {"box": "box-a"}),
                                 ("Kill box A (mocked)", "kill_forecast_server", {"box": "box-a"}),
                                 ("Kill box A (faked)", "kill_forecast_server", {"box": "box-a"}),
                                 ("Fail box A virtually", "kill_forecast_server", {"box": "box-a"}),
                                 ("Fail box A (test mode)", "kill_forecast_server", {"box": "box-a"}),
                                 ("Kill box A hypothetically", "kill_forecast_server", {"box": "box-a"})):
            with self.subTest(text=text):
                self.assertIn("simulated", self.refused(text, FakeOps(stacks=BOTH), tool, args))
        for text in ("Fail box A as a test", "Take box A down for the demo"):    # these are still plain commands
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                self.ask_llm(text, ops, tool_call("kill_forecast_server", box="box-a"), says("Killed."))
                self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])
        ops = FakeOps(stacks=BOTH)                      # the real words still work
        self.ask_llm("Fail box A", ops, tool_call("kill_forecast_server", box="box-a"), says("Killed."))
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])

    def test_one_command_changes_one_thing_once(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm("Fail box A", ops, tool_call("kill_forecast_server", box="box-a"),
                                   tool_call("kill_forecast_server", box="box-a"), says("Killed the forecast server on box-a."))
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])        # once, not twice
        self.assertIs(answer["steps"][1]["result"].get("not_run"), True)
        self.assertIn("already done", answer["steps"][1]["result"]["error"])
        # The same kill with one more argument is the same kill; so is a second start with a step count.
        ops = FakeOps(stacks=BOTH)
        self.ask_llm("Fail box A", ops, tool_call("kill_forecast_server", box="box-a"),
                     tool_call("kill_forecast_server", box="box-a", confirm=True), says("Killed."))
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])
        ops = FakeOps(stacks=BOTH)
        self.ask_llm("Start a training job on box B", ops, tool_call("start_finetune", box="box-b"),
                     tool_call("start_finetune", box="box-b", steps=200), says("Started."))
        self.assertEqual(ops.calls, [("start_finetune", "box-b", None)])

    def test_a_question_or_a_remark_never_changes_anything(self):
        for text in self.NOT_COMMANDS:
            for tool, args in self.CHANGES:
                with self.subTest(text=text, tool=tool):
                    self.refused(text, FakeOps(stacks=BOTH), tool, args)
        for text in self.NOT_COMMANDS:          # and on boxes without a supervisor, where the change is simulated
            for mode in ("slow", "error", "normal"):
                with self.subTest(text=text, mode=mode):
                    self.refused(text, FakeOps(), "simulate_box", {"box": "box-a", "mode": mode})

    def test_a_command_for_another_box_or_another_action_is_not_run(self):
        cases = [("Fail box A", "kill_forecast_server", {"box": "box-b"}, "names box-a"),
                 ("Fail box A and box B", "kill_forecast_server", {"box": "box-a"}, "names box-a and box-b"),
                 ("Switch to balanced", "kill_forecast_server", {"box": "box-a"}, "names no box"),
                 ("Recover box A", "kill_forecast_server", {"box": "box-a"}, "do not clearly ask"),
                 ("Stop the training job on box B", "kill_forecast_server", {"box": "box-b"}, "do not clearly ask"),
                 ("Fail box A", "start_finetune", {"box": "box-a"}, "do not clearly ask"),
                 ("Start a training job on box B", "stop_finetune", {"box": "box-b"}, "do not clearly ask"),
                 ("Fail box A", "set_routing_config", {"policy": "static"}, "not a command to change the routing")]
        for text, tool, args, why in cases:
            with self.subTest(text=text, tool=tool):
                self.assertIn(why, self.refused(text, FakeOps(stacks=BOTH), tool, args))

    def test_plain_commands_still_run(self):
        real = {"start": "start_finetune", "stop": "stop_finetune", "kill": "kill_forecast_server"}
        for text, action, box in COMMANDS:
            with self.subTest(text=text, where="supervisor"):
                if action in real:
                    ops = FakeOps(stacks=BOTH)
                    answer, _ = self.ask_llm(text, ops, tool_call(real[action], box=box), says("Done."))
                    self.assertEqual([(call[0], call[1]) for call in ops.calls], [(real[action], box)])
                    self.assertNotIn("not_run", answer["steps"][0]["result"])
            with self.subTest(text=text, where="no supervisor"):
                ops = FakeOps()
                answer, _ = self.ask_llm(text, ops, tool_call("simulate_box", box=box, mode=SIM_MODE[action]), says("Done."))
                self.assertEqual(ops.calls, [("simulate", box, SIM_MODE[action])])

    def test_routing_commands_run(self):
        for text, update in [("Switch to balanced", {"policy": "balanced"}),
                             ("use adaptive with box A preferred", {"policy": "adaptive", "preferred": "box-a"}),
                             ("Set the cooldown to 5 s", {"cooldown_s": 5}),
                             ("Drain box A", {"drain": {"box-a": True}}),
                             ("Reset the routing", {"reset": True})]:
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                self.ask_llm(text, ops, tool_call("set_routing_config", **update), says("Done."))
                self.assertEqual(ops.calls, [("set_config", update)])

    def test_only_the_last_user_message_counts(self):
        ops = FakeOps(stacks=BOTH)
        llm = FakeLLM(tool_call("kill_forecast_server", box="box-a"), says("Nothing was changed."))
        chat = [{"role": "user", "content": "Fail box A"}, {"role": "assistant", "content": "Killed the server on box-a."},
                {"role": "user", "content": "And how is box A now?"}]
        with mock.patch.object(A.httpx, "AsyncClient", llm):
            answer = asyncio.run(A.Agent(ops, "http://llm.test/v1", "stub").chat(chat))
        self.assertEqual(ops.calls, [])
        self.assertIs(answer["steps"][0]["result"]["not_run"], True)

    def test_tools_that_only_read_run_for_any_message(self):
        ops = FakeOps(stacks=BOTH)
        answer, _ = self.ask_llm("What happens if box A dies?", ops, tool_call("get_cluster_status"),
                                 says("The router sends everything to box B."))
        self.assertEqual((ops.status_reads, ops.calls), (1, []))
        self.assertNotIn("error", answer["steps"][0]["result"])
        # A place the built-in city list does not know: a known city goes to the weather model without the LLM.
        answer, _ = self.ask_llm("Weather in Pune if it rains?", FakeOps(),
                                 tool_call("get_forecast", lat=18.5, lon=73.9, hours_ahead=24, place="Pune"),
                                 says("Pune: 1 to 2 degrees."))
        self.assertIn("summary", answer["steps"][0]["result"])

    def test_no_humidity_figure_that_the_forecast_did_not_give(self):
        none = [{"tool": "get_forecast", "result": {"summary": {"humidity_avg_pct": None}}}]
        given = [{"tool": "get_forecast", "result": {"summary": {"humidity_avg_pct": 60.0}}}]
        for reply, cleaned in [
                ("Chicago: 2 °C to 8 °C, wind up to 18 km/h, humidity 71 %. Served by box-a.",
                 "Chicago: 2 °C to 8 °C, wind up to 18 km/h. Served by box-a."),
                ("11 °C – 17 °C, light wind around 16 km/h, humidity about 70 %. No rain data available.",
                 "11 °C – 17 °C, light wind around 16 km/h. No rain data available."),
                ("2 °C to 8 °C, wind 18 km/h and relative humidity of 65.5%.", "2 °C to 8 °C, wind 18 km/h."),
                ("2 °C to 8 °C, wind up to 18 km/h, humidity not given. Served by box-a.",
                 "2 °C to 8 °C, wind up to 18 km/h, humidity not given. Served by box-a.")]:
            with self.subTest(reply=reply):
                self.assertEqual(A.only_given_numbers(reply, none), cleaned)
                self.assertEqual(A.only_given_numbers(reply, given), reply)     # the forecast gave one: untouched
                self.assertEqual(A.only_given_numbers(reply, []), reply)        # no forecast in this turn
        ops = FakeOps()         # and through the chat: the fake forecast has no humidity
        answer, _ = self.ask_llm("Weather in Pune for the next 48 hours?", ops,    # not a built-in city: the LLM
                                 tool_call("get_forecast", lat=18.5, lon=73.9, hours_ahead=48, place="Pune"),
                                 says("Pune: 1 °C to 2 °C, wind up to 11 km/h, humidity 71 %."))
        self.assertEqual(answer["reply"], "Pune: 1 °C to 2 °C, wind up to 11 km/h.")


class FakeSupervisor:
    """Stands in for the httpx client of server.Ops: answers a POST to a supervisor from a script."""

    def __init__(self, answer=None):
        self.answer = answer
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs.get("json")))
        return httpx.Response(200, request=httpx.Request("POST", url), json=self.answer)


@NEEDS_SERVER
class TestServerSide(unittest.TestCase):
    """ui/server.py: where the job's text for the page and for the assistant is built."""

    def ops(self, answer=None):
        ops = S.Ops("http://router.test", {"box-a": "http://box-a.test", "box-b": "http://box-b.test"},
                    {"box-a": "http://stack-a.test", "box-b": "http://stack-b.test"})
        ops.client = FakeSupervisor(answer)
        return ops

    def test_the_overview_marks_a_stop_as_stopped(self):
        lines = ["step   60/400  loss 0.012000  lr 3.00e-05  1.52 steps/s", "stopped by a signal after step 78."]
        for code, stopped in ((3, True), (-15, True), (-9, True), (1, False), (2, False), (None, False)):
            with self.subTest(exit=code):
                view = S.finetune_view(dict(job_status("failed", code), last_lines=lines))
                self.assertIs(view["stopped"], stopped)
                self.assertEqual((view["state"], view["exit"], view["step"], view["total"]), ("failed", code, 60, 400))
        self.assertIs(S.finetune_view(job_status("done", 0))["stopped"], False)
        self.assertIs(S.finetune_view(job_status("running"))["stopped"], False)

    def test_a_stop_answer_with_exit_code_3_is_marked_stopped(self):
        ops = self.ops({"action": "stop", "job": "finetune", "status": job_status("failed", 3)})
        out = asyncio.run(ops.finetune("box-b", "stop"))
        self.assertIs(out.get("stopped"), True)
        self.assertEqual(out["status"]["state"], "failed")      # the supervisor's own answer is kept
        ops = self.ops({"action": "stop", "job": "finetune", "status": job_status("failed", 1)})
        self.assertNotIn("stopped", asyncio.run(ops.finetune("box-b", "stop")))

    def test_steps_out_of_bounds_never_reach_the_supervisor(self):
        for steps in (0, -5, 1000001, 99999999999, 2.5):
            with self.subTest(steps=steps):
                ops = self.ops({"action": "start", "job": "finetune", "status": job_status("running")})
                out = asyncio.run(ops.finetune("box-b", "start", steps))
                self.assertIn("steps must be a whole number from 1 to 1000000", out["error"])
                self.assertEqual(ops.client.posts, [])

    def test_steps_within_bounds_are_sent_as_an_argument(self):
        ops = self.ops({"action": "start", "job": "finetune", "status": job_status("running")})
        asyncio.run(ops.finetune("box-b", "start", 200))
        asyncio.run(ops.finetune("box-b", "start"))
        self.assertEqual(ops.client.posts, [("http://stack-b.test/jobs/finetune/start", {"args": ["--steps", "200"]}),
                                            ("http://stack-b.test/jobs/finetune/start", None)])

    def test_the_http_endpoint_has_the_same_bound(self):
        field = S.TrainIn.model_fields["steps"]
        self.assertEqual(A.MAX_STEPS, 1000000)
        self.assertIn("le=1000000", repr(field))
        self.assertIn("ge=1", repr(field))

    def test_the_cluster_status_tells_no_answer_from_no_job(self):
        ops = self.ops()

        async def router_status():
            return {"policy": "adaptive", "adaptive": {"active": "box-a", "preferred": "box-b", "switches": []},
                    "backends": {"box-a": {"healthy": True}, "box-b": {"healthy": False}}}

        async def metrics(window, since):
            return {"per_box": {}, "overall": {"rps": 0.0, "p95_ms": None, "errors": 0}}

        async def box_stats():
            return {"box-a": {"mode": "normal"}, "box-b": None}

        async def box_health():
            return {"box-a": {"simulated": False}, "box-b": None}

        async def stack_status():
            return {"box-a": {"finetune": S.finetune_view(job_status("failed", 3)), "forecast": None}, "box-b": None}

        ops.router_status, ops.metrics, ops.box_stats = router_status, metrics, box_stats
        ops.box_health, ops.stack_status = box_health, stack_status
        boxes = asyncio.run(ops.cluster_status())["boxes"]
        self.assertEqual(boxes["box-a"]["supervisor"], "ok")
        self.assertEqual(boxes["box-a"]["finetune"]["state"], "stopped")     # not "failed": it was a stop
        self.assertEqual(boxes["box-a"]["finetune"]["exit"], 3)
        self.assertEqual(boxes["box-b"]["supervisor"], "no answer")
        self.assertIsNone(boxes["box-b"]["finetune"])

    def test_recover_with_the_real_status_shape(self):
        """The assistant's "recover" reply on the status as server.Ops builds it, supervisor gone."""
        ops = self.ops()

        async def cluster_status():
            return {"policy": "adaptive", "active": "box-a", "preferred": "box-b", "rps": 0.0, "p95_ms": None,
                    "errors_10s": 0, "recent_switches": [],
                    "boxes": {"box-a": {"healthy": True, "mode": "normal", "simulated": False, "supervisor": "ok",
                                        "finetune": {"state": "idle"}},
                              "box-b": {"healthy": False, "mode": "unknown", "simulated": None,
                                        "supervisor": "no answer", "finetune": None}}}

        ops.cluster_status = cluster_status
        reply = ask("Recover box B", ops)["reply"]
        self.assertIn("supervisor of box-b did not answer", reply)
        self.assertEqual(ops.client.posts, [])


if __name__ == "__main__":
    unittest.main()


@NEEDS_HTTPX
class TestRoutingSettings(unittest.TestCase):
    """Drain, the tuning settings and reset: plain commands change them; questions do not."""

    COMMANDS = [
        ("drain box-a", {"drain": {"box-a": True}}),
        ("please drain box b", {"drain": {"box-b": True}}),
        ("undrain box-a", {"drain": {"box-a": False}}),
        ("set the cooldown to 5 s", {"cooldown_s": 5.0}),
        ("set margin to 30%", {"margin": 0.3}),
        ("change the return margin to 0.1", {"return_margin": 0.1}),
        ("set window to 2 seconds", {"window_s": 2.0}),
        ("set stall to 800 ms", {"stall_ms": 800.0}),
        ("set stall to 2 s", {"stall_ms": 2000.0}),
        ("reset routing", {"reset": True}),
        ("reset the router settings", {"reset": True}),
    ]
    QUESTIONS = ["should I drain box-a?", "what is the cooldown?", "is box-a drained?",
                 "do not drain box-a", "what if I set the cooldown to 5 s?"]

    def test_commands(self):
        for text, update in self.COMMANDS:
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                answer = ask(text, ops)
                self.assertEqual(ops.calls, [("set_config", update)])
                self.assertEqual(tools_of(answer), ["set_routing_config"])

    def test_questions_change_nothing(self):
        for text in self.QUESTIONS:
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                ask(text, ops)
                self.assertEqual(ops.calls, [])

    def test_drain_naming_two_boxes_changes_nothing(self):
        ops = FakeOps(stacks=BOTH)
        answer = ask("drain box-a and box-b", ops)
        self.assertEqual(ops.calls, [])
        self.assertIn("Nothing was changed", answer["reply"])

    def test_llm_tool_passes_only_known_fields(self):
        ops = FakeOps(stacks=BOTH)
        agent = A.Agent(ops, None, None)
        step = asyncio.run(agent.run_tool("set_routing_config", {"cooldown_s": 3, "bogus": 1, "drain": None}))
        self.assertEqual(ops.calls, [("set_config", {"cooldown_s": 3})])
        self.assertNotIn("error", step["result"])
        step = asyncio.run(agent.run_tool("set_routing_config", {}))
        self.assertIn("error", step["result"])

    def test_tool_schema_lists_the_boxes(self):
        tools = {t["function"]["name"]: t["function"] for t in A.build_tools(["box-a", "box-b"], [])}
        params = tools["set_routing_config"]["parameters"]["properties"]
        self.assertEqual(params["preferred"]["enum"], ["", "box-a", "box-b"])
        self.assertEqual(set(params["drain"]["properties"]), {"box-a", "box-b"})


@NEEDS_HTTPX
class TestAnnouncedButNotDone(unittest.TestCase):
    """Nemotron sometimes answers "Let me get their coordinates" with no tool call. That must not be the reply."""

    def ask_llm(self, text, ops, *script):
        llm = FakeLLM(*script)
        with mock.patch.object(A.httpx, "AsyncClient", llm):
            answer = ask(text, ops, base_url="http://llm.test/v1", model="stub")
        return answer, llm

    # The places in these questions are not in the built-in city list, so the questions reach the LLM
    # (a known city goes to the weather model without it).
    def test_an_announcement_is_followed_up_and_the_tools_run(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm(
            "Will it be warmer in Pune or Nagpur tomorrow at noon?", ops,
            says("I need to check the forecast for both cities. Let me start by getting their coordinates."),
            tool_call("geocode", place="Pune"),
            says("Nagpur will be warmer."))
        self.assertEqual(answer["reply"], "Nagpur will be warmer.")
        self.assertEqual(tools_of(answer), ["geocode"])
        self.assertEqual(len(llm.posts), 3)
        # (the fake keeps a reference to the one conversation list, so look for the follow-up in it)
        self.assertIn({"role": "user", "content": A.GO_AHEAD}, llm.posts[-1]["messages"])

    def test_it_gives_up_after_two_follow_ups(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm("Weather in Pune?", ops,
                                   says("Let me check."), says("Let me check that for you."),
                                   says("Let me look it up."))
        self.assertEqual(len(llm.posts), 3)
        self.assertEqual(answer["reply"], "Let me look it up.")

    def test_a_real_answer_is_not_followed_up(self):
        answer, llm = self.ask_llm("Weather in Pune tomorrow at noon?", FakeOps(stacks=BOTH),
                                   tool_call("get_forecast", lat=18.5, lon=73.9, hours_ahead=24, place="Pune"),
                                   says("Tomorrow at noon Pune is 2 °C. Let me know if you need more."))
        self.assertEqual(len(llm.posts), 2)             # the tool round and the answer: no follow-up
        self.assertEqual(answer["reply"], "Tomorrow at noon Pune is 2 °C. Let me know if you need more.")
        answer, llm = self.ask_llm("Is box A healthy?", FakeOps(stacks=BOTH), says("Box A is healthy and serving traffic."))
        self.assertEqual(len(llm.posts), 1)

    def test_announces_action(self):
        for text, want in (("Let me start by getting their coordinates.", True), ("I'll check box-b.", True),
                           ("One moment.", True), ("Let me know if you need more.", False),
                           ("It will be 4 °C.", False), ("", False), ("Let me check. " * 60, False),
                           # as Nemotron wrote them on the event boxes, with the typographic apostrophe
                           ("I’ll check the forecast for Seattle.", True),
                           ("I’ll check the Seattle forecast for the next 7 days.", True),
                           ("I’m not sure of the exact forecast yet – I’ll look up Seattle’s "
                            "coordinates and pull a 7‑day outlook for you.", True)):
            with self.subTest(text=text[:40]):
                self.assertEqual(A.announces_action(text), want)


@NEEDS_SERVER
class TestPageServerGuards(unittest.TestCase):
    """ui/server.py, found in the review of Nov 14: what the page server says and allows."""

    def ops(self, policy, health=None):
        ops = S.Ops("http://router.test", {"box-a": "http://box-a.test", "box-b": "http://box-b.test"},
                    {"box-a": "http://stack-a.test", "box-b": "http://stack-b.test"})
        ops.client = FakeSupervisor({"mode": "error"})

        async def router_status():
            return {"policy": policy, "adaptive": {"active": "box-a", "preferred": "box-b",
                                                    "switches": [{"t": 1.0, "from": "box-b", "to": "box-a", "reason": "x"}]},
                    "backends": {"box-a": {"healthy": True}, "box-b": {"healthy": True}}}

        async def metrics(window, since):
            return {"per_box": {"box-a": {"requests": 40}, "box-b": {"requests": 44}},
                    "overall": {"rps": 8.4, "p95_ms": 120.0, "errors": 0}}

        async def box_stats():
            return {"box-a": {"mode": "normal"}, "box-b": {"mode": "normal"}}

        async def box_health():
            return health or {"box-a": {"simulated": False, "model_version": "afno-19m-65f9f3bb3270"},
                              "box-b": {"simulated": True, "model_version": "mock-0.1"}}

        async def stack_status():
            return {}

        ops.router_status, ops.metrics, ops.box_stats = router_status, metrics, box_stats
        ops.box_health, ops.stack_status = box_health, stack_status
        return ops

    def test_the_serving_box_follows_health_as_the_router_does(self):
        def st(policy, a=True, b=True, drained=(), preferred="box-b", active="box-a"):
            return {"policy": policy, "adaptive": {"active": active, "preferred": preferred, "switches": []},
                    "backends": {"box-a": {"healthy": a, "drained": "box-a" in drained},
                                 "box-b": {"healthy": b, "drained": "box-b" in drained}}}
        for state, one, words in (
                (st("balanced"), None, "both boxes"),
                (st("balanced", a=False), "box-b", "box-b alone: box-a is down or drained"),     # after "Fail box A"
                (st("least_loaded", drained=("box-b",)), "box-a", "box-a alone: box-b is down or drained"),
                (st("static"), "box-b", "one box at a time: box-b"),
                (st("static", b=False), "box-a", "box-a alone: box-b is down or drained"),      # the home box is down
                (st("static", drained=("box-b",)), "box-a", "box-a alone"),
                (st("static", preferred=None), "box-a", "one box at a time: box-a"),            # no home box: the first
                (st("adaptive"), "box-a", "one box at a time: box-a"),
                (st("adaptive", a=False), "box-b", "box-b alone"),
                (st("balanced", drained=("box-a", "box-b")), None, "both boxes, although both are drained"),
                (st("static", a=False, drained=("box-b",)), "box-b", "one box at a time: box-b"),   # drained, the last resort
                (st("balanced", a=False, drained=("box-b",)), "box-b", "box-b alone"),
                (st("balanced", a=False, b=False), None, "no box")):
            with self.subTest(state=state):
                active, serving = S.serving_now(state)
                self.assertEqual(active, one)
                self.assertTrue(serving.startswith(words), serving)

    def test_the_rule_based_status_names_the_box_that_is_left(self):
        class OneDown(FakeOps):
            async def cluster_status(self):
                s = await super().cluster_status()
                s.update(policy="balanced", active="box-b", serving="box-b alone: box-a is down or drained")
                s["boxes"]["box-a"]["healthy"] = False
                return s
        reply = ask("Which box is serving right now?", OneDown())["reply"]
        self.assertIn("active box **box-b** (the other box is down or drained)", reply)

    def test_one_active_box_only_under_the_rules_that_have_one(self):
        for policy, active in (("adaptive", "box-a"), ("static", "box-b"), ("balanced", None), ("least_loaded", None),
                               ("round_robin", None)):
            with self.subTest(policy=policy):
                s = asyncio.run(self.ops(policy).cluster_status())
                self.assertEqual(s["active"], active)
                self.assertEqual(s["serving"].startswith("both boxes"), active is None)
                self.assertEqual(len(s["recent_switches"]), 1 if policy == "adaptive" else 0)
                self.assertEqual(s["boxes"]["box-b"]["served_share_10s"], 44)

    def test_the_status_reply_does_not_name_an_active_box_under_balanced(self):
        class Balanced(FakeOps):
            async def cluster_status(self):
                s = await FakeOps.cluster_status(self)
                return dict(s, policy="balanced", active=None)
        reply = ask("Which box is serving right now?", Balanced())["reply"]
        self.assertIn("both boxes serve", reply)
        self.assertNotIn("active box", reply)
        self.assertIn("active box **box-b**", ask("Which box is serving right now?", FakeOps())["reply"])    # adaptive

    def test_the_test_hook_is_refused_on_a_box_with_the_real_model(self):
        ops = self.ops("balanced")
        out = asyncio.run(ops.set_box_mode("box-a", "error"))
        self.assertIn("runs the real weather model", out["error"])
        self.assertEqual(ops.client.posts, [])                                  # nothing was sent to the box
        out = asyncio.run(ops.set_box_mode("box-b", "error"))                   # a mock: the hook is its purpose
        self.assertEqual(out.get("kind"), "simulated")
        self.assertEqual(ops.client.posts, [("http://box-b.test/admin/mode", {"mode": "error", "slow_factor": 2.0})])
        stand_in = {"box-a": {"simulated": False, "model_version": "afno-untrained-1"}}
        ops = self.ops("balanced", stand_in)                                    # the inference stand-in keeps its hook
        self.assertEqual(asyncio.run(ops.set_box_mode("box-a", "slow")).get("kind"), "simulated")

    def test_the_load_generator_stops_sending_when_too_many_wait(self):
        sent = []

        class Slow:
            async def post(self, url, **kwargs):
                sent.append(url)
                await asyncio.sleep(0.05)

        async def run():
            ops = S.Ops("http://router.test", {"box-a": "http://box-a.test"})
            ops.client = None                           # the page's own connections must not be used for the load
            gen = S.LoadGen(ops)
            gen.client, gen.MAX_IN_FLIGHT = Slow(), 5
            await asyncio.gather(*[gen._send({"lat": 0, "lon": 0}) for _ in range(12)])
            return gen
        gen = asyncio.run(run())
        self.assertEqual((len(sent), gen.sent, gen.skipped, gen.in_flight), (5, 5, 7, 0))


def chat(messages, ops, **llm):
    """A whole conversation -> the assistant's answer to its last message."""
    agent = A.Agent(ops, llm.get("base_url"), llm.get("model"))
    return asyncio.run(agent.chat(messages))


@NEEDS_HTTPX
class TestWeatherNumbersComeFromTheModel(unittest.TestCase):
    """On the event boxes (Nov 14) the LLM answered every follow-up weather question without calling the
    forecast tool: it copied the form of its earlier answer and wrote new numbers and a box name. No weather
    number may reach the user unless a forecast ran in the same turn."""

    FIRST = [{"role": "user", "content": "Weather in Chicago for the next 48 hours?"},
             {"role": "assistant", "content": "Chicago: 2.8 °C to 9.7 °C, wind up to 19 km/h, served by box-b."}]

    def with_llm(self, messages, ops, *script):
        llm = FakeLLM(*script)
        with mock.patch.object(A.httpx, "AsyncClient", llm):
            answer = chat(messages, ops, base_url="http://llm.test/v1", model="stub")
        return answer, llm

    def test_a_follow_up_that_names_a_known_city_goes_to_the_weather_model(self):
        answer, llm = self.with_llm(self.FIRST + [{"role": "user", "content": "And what about New York?"}], FakeOps())
        self.assertEqual(llm.posts, [])                 # the LLM was not asked at all
        self.assertEqual(tools_of(answer), ["geocode", "get_forecast"])
        self.assertEqual(answer["steps"][1]["args"]["place"], "New York")
        self.assertEqual(answer["steps"][1]["args"]["hours_ahead"], 48)     # the time span of the earlier question
        self.assertEqual(answer["mode"], "weather-model")
        answer, _ = self.with_llm(self.FIRST + [{"role": "user", "content": "And New York tonight?"}], FakeOps())
        self.assertEqual(answer["steps"][1]["args"]["hours_ahead"], 12)     # its own time words win
        answer, _ = self.with_llm(self.FIRST + [{"role": "user", "content": "Weather in Boston?"}], FakeOps())
        self.assertEqual(answer["steps"][1]["args"]["hours_ahead"], 24)     # a new question, not a follow-up

    def test_a_follow_up_without_a_place_gets_the_earlier_place(self):
        for text, hours in (("And tomorrow?", 48), ("What about the wind?", 48), ("And the next 3 days?", 72),
                            ("Will it rain there this weekend?", 120), ("and tonight", 12)):
            with self.subTest(text=text):
                answer, llm = self.with_llm(self.FIRST + [{"role": "user", "content": text}], FakeOps())
                self.assertEqual(llm.posts, [])
                self.assertEqual(tools_of(answer), ["geocode", "get_forecast"])
                self.assertEqual(answer["steps"][1]["args"]["place"], "Chicago")
                self.assertEqual(answer["steps"][1]["args"]["hours_ahead"], hours)
                self.assertIn("earlier question", answer["note"])
        answer, _ = self.with_llm(self.FIRST + [{"role": "user", "content": "Will it rain there tomorrow?"}], FakeOps())
        self.assertIn("no rain", answer["reply"])       # asked about rain: the answer says the model has none

    def test_what_is_not_such_a_follow_up(self):
        first = self.FIRST
        for text in ("What about St. Louis?",               # a place the list does not know: not Chicago's forecast
                     "Is that warm for November?", "And what about it?", "Fail box A", "Which box served that?",
                     "And tomorrow?" + " please" * 12):
            with self.subTest(text=text):
                self.assertIsNone(A.followup_question(first + [{"role": "user", "content": text}]))
        self.assertIsNone(A.followup_question([{"role": "user", "content": "And tomorrow?"}]))     # nothing earlier
        self.assertIsNone(A.followup_question([{"role": "user", "content": "Fail box A"},
                                               {"role": "user", "content": "And tomorrow?"}]))
        self.assertEqual(A.followup_question(first + [{"role": "user", "content": "And tomorrow?"}]),
                         "chicago: And tomorrow?")
        self.assertEqual(A.followup_question(first + [{"role": "user", "content": "What about the wind?"}]),
                         "Weather in Chicago for the next 48 hours? What about the wind?")

    def test_numbers_without_a_forecast_are_sent_back_to_the_llm(self):
        ops = FakeOps()
        answer, llm = self.with_llm(
            self.FIRST + [{"role": "user", "content": "What about St. Louis?"}], ops,
            says("St. Louis: 5 °C to 13 °C, wind up to 22 km/h. Served by box-a."),     # its own numbers
            tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=48, place="St. Louis"),
            says("St. Louis: 1 °C to 2 °C, wind up to 11 km/h. Served by box-a."))
        self.assertEqual(len(llm.posts), 3)
        self.assertIn({"role": "user", "content": A.NOT_FROM_THE_MODEL}, llm.posts[-1]["messages"])
        self.assertEqual(tools_of(answer), ["get_forecast"])
        self.assertEqual(answer["reply"], "St. Louis: 1 °C to 2 °C, wind up to 11 km/h. Served by box-a.")

    def test_numbers_without_a_forecast_never_reach_the_user(self):
        made_up = "St. Louis: 5 °C to 13 °C, wind up to 22 km/h. Served by box-a."
        answer, llm = self.with_llm(self.FIRST + [{"role": "user", "content": "What about St. Louis?"}], FakeOps(),
                                    says(made_up), says(made_up), says(made_up))
        self.assertEqual(len(llm.posts), 3)             # asked, told twice, then no more
        self.assertEqual(answer["reply"], A.NO_FORECAST_REPLY)
        self.assertEqual(answer["steps"], [])
        self.assertIn("not shown", answer["note"])
        self.assertNotIn("13", answer["reply"])

    def test_a_failed_forecast_is_said_plainly(self):
        class Down(FakeOps):
            async def forecast(self, lat, lon, hours, members=1, place=None):
                return {"error": "HTTP 503: all backends failed", "ms": 3.0}
        answer, llm = self.with_llm([{"role": "user", "content": "Weather in Pune?"}], Down(),
                                    tool_call("get_forecast", lat=18.5, lon=73.9, hours_ahead=24, place="Pune"),
                                    says("Pune: 20 °C to 30 °C."), says("Pune: 20 °C to 30 °C."), says("Pune: 20 °C to 30 °C."))
        self.assertEqual(answer["reply"], "The forecast failed (HTTP 503: all backends failed), so I have no numbers to give.")
        self.assertEqual(tools_of(answer), ["get_forecast"])

    def test_unsourced_weather(self):
        # a forecast of 3.2 to 11.0 °C (37.8 to 51.8 °F), wind up to 4.2 m/s (15 km/h, 9 mph)
        ran = [{"tool": "get_forecast", "result": {"summary": {"min_c": 3.2, "max_c": 11.0, "max_wind_ms": 4.2,
                                                               "precip_mm": None}, "points": []}}]
        failed = [{"tool": "get_forecast", "result": {"error": "HTTP 503"}}]
        for reply in ("New York: 3.2\u202f°C to 11.0\u202f°C, wind up to 15\u202fkm/h, served by box\u2011a.",
                      "About 52 degrees tomorrow.", "Wind near 9 mph.", "It is 11° at most.", "Gusts of 4.2 m/s."):
            with self.subTest(reply=reply):
                self.assertTrue(A.unsourced_weather(reply, []))
                self.assertTrue(A.unsourced_weather(reply, failed))
                self.assertFalse(A.unsourced_weather(reply, ran))     # these are the forecast's own numbers
        for reply in ("New York: 5 °C to 13 °C, wind up to 22 km/h.",                    # other numbers
                      "3.2 °C to 11.0 °C here; Boston will be colder, -4 °C to 3 °C.",      # a second, made-up place
                      "3.2 °C to 11.0 °C with a 30% chance of rain.", "11 °C and 5 mm of rain.",
                      "3.2 °C to 11.0 °C, 3 cm of snow on Tuesday.", "Up to 11 °C under 20% cloud cover.",
                      "3.2 °C to 11.0 °C. Humidity will be 65 percent.", "5\u201313 \u2103, wind 22 kmh",
                      "highs near 55 and lows around 41 Fahrenheit", "a high of 13 and a low of 5",
                      "Around 30 knots of wind.", "Wind 22 km per hour.", "Wind: 14 miles per hour.",
                      "Gusts of 6 meters per second.", "Highs in the 50s and lows in the 40s.",
                      "Temperatures in the low teens.", "Single digits overnight."):
            with self.subTest(reply=reply):
                self.assertTrue(A.unsourced_weather(reply, ran))      # after a forecast too: not its numbers
        self.assertFalse(A.unsourced_weather("3 °C to 11 °C, wind up to 15 km/h.", ran))      # rounded to whole numbers
        self.assertFalse(A.unsourced_weather("38 °F to 52 °F, wind up to 9 mph.", ran))       # in the other units
        self.assertFalse(A.unsourced_weather("Wind up to 15 km per hour, or 9 miles an hour.", ran))
        # beside a forecast: the freezing point, and a figure of the user's own question
        self.assertFalse(A.unsourced_weather("No. The low is 3.2 °C, above freezing (0 °C).", ran, "Will it freeze tonight?"))
        self.assertFalse(A.unsourced_weather("No: the low is 38 °F, above the 32 °F freezing point.", ran))
        self.assertFalse(A.unsourced_weather("No. It stays below 30 °C: 3.2 °C to 11.0 °C.", ran, "Will Pune get above 30 °C tomorrow?"))
        self.assertFalse(A.unsourced_weather("Yes, wind up to 9 mph, well under 20 mph.", ran, "Will the wind stay under 20 mph?"))
        self.assertTrue(A.unsourced_weather("No. It stays below 30 °C, at about 25 °C.", ran, "Will Pune get above 30 °C tomorrow?"))
        self.assertTrue(A.unsourced_weather("About 30 °C.", [], "Will Pune get above 30 °C tomorrow?"))    # no forecast: not said
        # negative numbers, with the minus signs an LLM writes
        cold = [{"tool": "get_forecast", "result": {"summary": {"min_c": -4.2, "max_c": -0.1, "max_wind_ms": 4.7}, "points": []}}]
        for reply in ("Akureyri: -4.2 °C to -0.1 °C, wind up to 17 km/h.", "Akureyri: \u20134.2 °C to \u20130.1 °C.",
                      "\u20114.2\u202f°C to \u20110.1\u202f°C, served by box\u2011b.", "\u22124.2 °C to \u22120.1 °C",
                      "Between -4.2 and -0.1 °C.", "24 °F to 32 °F.", "About 4 below zero at night."):
            with self.subTest(reply=reply):
                self.assertFalse(A.unsourced_weather(reply, cold))
        for reply in ("Akureyri: 4.2 °C to 0.1 °C.", "Akureyri: -14.2 °C to -0.1 °C.", "\u20119 °C at night."):
            with self.subTest(reply=reply):
                self.assertTrue(A.unsourced_weather(reply, cold))       # the sign counts, and so does the number
        self.assertFalse(A.unsourced_weather("In the 48 hours to Monday: 3.2 °C to 11.0 °C; in the 1990s it was colder.", ran))
        self.assertFalse(A.unsourced_weather("Yes, 3.2 °C to 11.0 °C is usual for November; take a coat.", ran))
        self.assertFalse(A.unsourced_weather("- Low: 3.2 °C\n- High: 11.0 °C\n- Wind: 15 km/h", ran))
        for reply in ("box-b is serving right now.", "p95 is 120 ms at 8 requests a second.", "Which city do you mean?",
                      "The rule is balanced; 2 switches in the last 10 minutes."):
            with self.subTest(reply=reply):
                self.assertFalse(A.unsourced_weather(reply, []))
        # the user brought the number: a conversion is not a forecast
        self.assertFalse(A.unsourced_weather("20 °C is 68 °F.", [], "What is 20 °C in Fahrenheit?"))
        self.assertFalse(A.unsourced_weather("20 °C is 68 °F.", [], "What is 20 C in Fahrenheit?"))
        self.assertFalse(A.unsourced_weather("20 °C equals 68 °F.", [], "what is 20c in f"))
        self.assertFalse(A.unsourced_weather("68 °F is 20 °C.", [], "Convert 68 fahrenheit to celsius"))
        self.assertFalse(A.unsourced_weather("That is 36 km/h.", [], "How many km/h is 10 meters per second?"))
        self.assertFalse(A.unsourced_weather("Sure: 37 °F to 50 °F.", [], "Can you give me that in Fahrenheit?"))
        for asked in ("What about St. Louis, above 10 °C?", "What about St. Louis in Fahrenheit?",
                      "Is 50 F warm for St. Louis in November?", "Will it stay under 20 mph?"):
            with self.subTest(asked=asked):     # a figure or a unit in the question is not a conversion
                self.assertTrue(A.unsourced_weather("About 13 °C, wind up to 22 km/h.", [], asked))
        # ... but a question about a place or a time is checked even when it holds a figure itself
        for asked in ("Will St. Louis get above 10 °C tomorrow?", "Is it over 50 degrees in St. Louis tomorrow?",
                      "My app says 4 C for Evanston. What does the model say?",
                      "It is 30 mph windy here in Naperville, what about tomorrow?"):
            with self.subTest(asked=asked):
                self.assertTrue(A.unsourced_weather("About 13 °C, wind up to 22 km/h.", [], asked))
        # a number that another tool of the turn gave may be said: a GPU temperature, a place's coordinates
        status = [{"tool": "get_cluster_status", "result": {"boxes": {"box-a": {"gpu": {"temp_c": 54, "util_pct": 63}}}}}]
        self.assertFalse(A.unsourced_weather("The GPU of box-a is at 54 °C, 63% busy.", status, "How hot is the GPU of box A?"))
        self.assertTrue(A.unsourced_weather("The GPU of box-a is at 71 °C.", status, "How hot is the GPU of box A?"))
        place = [{"tool": "geocode", "result": {"place": "Chicago", "lat": 41.88, "lon": -87.63}}]
        self.assertFalse(A.unsourced_weather("Chicago is at 41.88° N.", place, "Where is Chicago?"))
        self.assertFalse(A.unsourced_weather("Chicago (41.88° N, 87.63° W).", place, "Where is Chicago?"))
        self.assertTrue(A.unsourced_weather("Chicago is 4 C now.", []))
        self.assertFalse(A.unsourced_weather("2 Chicago forecasts ran; 3 checks passed.", []))
        self.assertTrue(A.unsourced_weather("It will be humid, with humidity around 78%.", []))    # a made-up humidity too

    def test_after_a_forecast_the_reply_may_state_only_its_numbers(self):
        # FakeOps forecasts 1 to 2 °C, wind up to 3 m/s (11 km/h)
        ask = [{"role": "user", "content": "What about St. Louis?"}]
        call = tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=48, place="St. Louis")
        for made_up in ("St. Louis: 12 °C to 21 °C, wind up to 45 km/h, served by box-a.",
                        "St. Louis: 1 °C to 2 °C. Kansas City will be colder, -4 °C to 0 °C.",
                        "St. Louis: 1 °C to 2 °C, wind up to 11 km/h, with a 30% chance of rain.", ""):
            with self.subTest(reply=made_up):
                answer, llm = self.with_llm(self.FIRST + ask, FakeOps(), call, says(made_up))
                self.assertEqual(len(llm.posts), 2)                 # not sent back: the forecast is there
                self.assertTrue(answer["reply"].startswith("**St. Louis, next 48 h:** 1 °C to 2 °C, wind up to 11 km/h."),
                                answer["reply"])
                self.assertIn("Served by box-a.", answer["reply"])
                for wrong in ("12", "21", "45", "Kansas", "30%", "rain"):
                    self.assertNotIn(wrong, answer["reply"])
                self.assertEqual(answer["note"], A.MODEL_NUMBERS_NOTE)
        for fine in ("St. Louis: 1 °C to 2 °C, wind up to 11 km/h. Served by box-a.",
                     "St. Louis: 34 °F to 36 °F, wind up to 7 mph.",
                     "- Low: 1 °C\n- High: 2 °C\n- Wind: up to 11 km/h"):      # its line breaks stay
            with self.subTest(reply=fine):
                answer, _ = self.with_llm(self.FIRST + ask, FakeOps(), call, says(fine))
                self.assertEqual(answer["reply"], fine)
                self.assertNotIn("note", answer)

    def test_a_figure_in_the_question_does_not_switch_the_check_off(self):
        made_up = "Yes, St. Louis should reach about 13 °C tomorrow afternoon, with wind up to 22 km/h."
        answer, llm = self.with_llm([{"role": "user", "content": "Will St. Louis get above 10 °C tomorrow?"}], FakeOps(),
                                    says(made_up), says(made_up), says(made_up))
        self.assertEqual(len(llm.posts), 3)
        self.assertEqual(answer["reply"], A.NO_FORECAST_REPLY)
        answer, llm = self.with_llm([{"role": "user", "content": "What is 20 C in Fahrenheit?"}], FakeOps(),
                                    says("20 °C is 68 °F."))
        self.assertEqual((len(llm.posts), answer["reply"]), (1, "20 °C is 68 °F."))      # arithmetic is not weather

    def test_an_answer_that_was_sent_back_says_so(self):
        answer, _ = self.with_llm(
            self.FIRST + [{"role": "user", "content": "Is that warm for November?"}], FakeOps(),
            says("Yes: 2.8 °C to 9.7 °C is on the mild side."),                 # the earlier answer's numbers
            tool_call("get_forecast", lat=41.88, lon=-87.63, hours_ahead=48, place="Chicago"),
            says("Chicago: 1 °C to 2 °C, which is cold for November."))
        self.assertEqual(answer["reply"], "Chicago: 1 °C to 2 °C, which is cold for November.")
        self.assertEqual(answer["note"], A.SENT_BACK_NOTE)

    def test_a_number_that_a_tool_gave_may_be_said(self):
        class WithGpu(FakeOps):
            async def cluster_status(self):
                s = await super().cluster_status()
                s["boxes"]["box-a"]["gpu"] = {"temp_c": 54, "util_pct": 63}
                return s
        ask = [{"role": "user", "content": "How hot is the GPU of box A?"}]
        answer, llm = self.with_llm(ask, WithGpu(), tool_call("get_cluster_status"), says("The GPU of box-a is at 54 °C."))
        self.assertEqual((len(llm.posts), answer["reply"]), (2, "The GPU of box-a is at 54 °C."))
        wrong = "The GPU of box-a is at 71 °C."
        answer, llm = self.with_llm(ask, WithGpu(), tool_call("get_cluster_status"), says(wrong), says(wrong), says(wrong))
        self.assertEqual(answer["reply"], A.NO_FORECAST_REPLY)

    def test_what_the_check_reads_as_a_weather_figure(self):
        ran = [{"tool": "get_forecast", "result": {"summary": {"min_c": 3.2, "max_c": 11.0, "max_wind_ms": 4.2,
                                                               "precip_mm": None}, "points": []}}]
        for reply in ("Evanston: 6 \u00baC to 14 \u00baC.", "Evanston: 6 degC to 14 degC.", "Evanston: 6c to 14c.",
                      "Wind up to 22 km/hr.", "Wind up to 22 kmph.", "Wind up to 14 m.p.h.", "Evanston: **6** °C to **14** °C.",
                      "Evanston: 6 up to 10 °C.", "Evanston: 6/10 °C.", "Evanston: 6\u202610 °C.", "Evanston: 6 or 10 °C.",
                      "3 °C to 11 °C; feels like -3 with the wind.", "3 °C to 11 °C, wind chill near -3.", "Gusts to 60.",
                      "Evanston: 10 °C below zero at night.", "Evanston: minus 3 °C to minus 10 °C.",
                      "A frosty 3 °C below freezing.",
                      # rain and the like, in a sentence that names "St. Louis" (the dot is not the sentence's end)
                      "Rain chance in St. Louis is 30%.", "Humidity in St. Louis will be about 65%.",
                      "Snow in St. Paul: 3 cm by morning.", "3 °C to 11 °C, 30 per cent chance of showers.",
                      "3 °C to 11 °C, humidity 65.", "3 °C to 11 °C, RH 65%.", "3 °C to 11 °C, 0.2 in of rain."):
            with self.subTest(reply=reply):
                self.assertTrue(A.unsourced_weather(reply, ran))
        for reply in ("3.2 °C to 11.0 °C in St. Louis. Served by box-b.", "**3.2 °C** to **11.0 °C**, wind up to 15 km/hr.",
                      "3c to 11c, wind 15 kmph.", "Low 3 \u00baC, high 11 \u00baC."):
            with self.subTest(reply=reply):
                self.assertFalse(A.unsourced_weather(reply, ran))
        # statements with no number about what the model does not forecast are cut from a reply
        for said, left in (("3.2 °C to 11.0 °C. Light snow likely overnight.", "3.2 °C to 11.0 °C."),
                           ("3.2 °C to 11.0 °C, mostly sunny.", "3.2 °C to 11.0 °C."),
                           ("3.2 °C to 11.0 °C. Bring an umbrella.", "3.2 °C to 11.0 °C."),
                           ("No rain in St. Louis tomorrow. 3.2 °C to 11.0 °C.", "3.2 °C to 11.0 °C."),
                           ("3.2 °C to 11.0 °C. The model has no rain or cloud output.",
                            "3.2 °C to 11.0 °C. The model has no rain or cloud output.")):
            with self.subTest(said=said):
                self.assertEqual(A.only_given_numbers(said, ran), left)

    def test_a_reply_that_explains_or_closes_is_not_taken_for_an_announcement(self):
        for text in ("Type \"fail box A\" and I'll kill its forecast server for real.", "Killed. I'll stop here.",
                     "I will set the rule back when you ask.", "The forecast server on box-a has been killed."):
            with self.subTest(text=text):
                self.assertFalse(A.announces_action(text))
        self.assertTrue(A.announces_action("I\u2019ll check the forecast for Miami."))
        # an LLM answer whose content is not text does not end the chat with an error
        odd = {"choices": [{"message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}}]}
        answer, _ = self.with_llm([{"role": "user", "content": "hm"}], FakeOps(), odd)
        self.assertIn("ask again", answer["reply"])

    def test_an_empty_reply_and_too_many_rounds_still_give_an_answer(self):
        ops = FakeOps(stacks=BOTH)
        answer, _ = self.with_llm([{"role": "user", "content": "Fail box A"}], ops,
                                  tool_call("kill_forecast_server", box="box-a"), says(""))
        self.assertEqual(answer["reply"], "Done: kill_forecast_server on box-a: done.")
        answer, _ = self.with_llm([{"role": "user", "content": "hm"}], FakeOps(), says(""))
        self.assertIn("ask again", answer["reply"])
        # Eight rounds and no final text, but a forecast ran: the reply is the model's numbers.
        call = tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=24, place="St. Louis")
        answer, llm = self.with_llm([{"role": "user", "content": "What about St. Louis?"}], FakeOps(), *([call] * 8))
        self.assertEqual(len(llm.posts), 8)
        self.assertTrue(answer["reply"].startswith("**St. Louis, next 24 h:** 1 °C to 2 °C"), answer["reply"])
        self.assertEqual(answer["reply"].count("St. Louis"), 1)          # the same forecast is said once

    def test_a_follow_up_after_a_place_the_list_does_not_know_is_not_answered_for_an_older_city(self):
        chain = self.FIRST + [{"role": "user", "content": "And what about St. Louis?"},
                              {"role": "assistant", "content": "St. Louis: 0.5 °C to 10.2 °C, wind up to 7 km/h."}]
        for text in ("And tomorrow?", "What about the wind?"):
            with self.subTest(text=text):
                self.assertIsNone(A.followup_question(chain + [{"role": "user", "content": text}]))
        # ... and the LLM is told which question the follow-up belongs to (the user's own words are not changed)
        self.assertEqual(A.followup_hint(chain + [{"role": "user", "content": "And tomorrow?"}]), "And what about St. Louis?")
        answer, llm = self.with_llm(chain + [{"role": "user", "content": "And tomorrow?"}], FakeOps(),
                                    tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=48, place="St. Louis"),
                                    says("St. Louis: 1 °C to 2 °C, wind up to 11 km/h."))
        sent = [m["content"] for m in llm.posts[0]["messages"] if m["role"] == "user"]
        self.assertEqual(sent[-1], 'And tomorrow?' + A.FOLLOWS.format("And what about St. Louis?"))
        self.assertEqual(sent[:-1], ["Weather in Chicago for the next 48 hours?", "And what about St. Louis?"])
        for msgs in (self.FIRST + [{"role": "user", "content": "And tomorrow?"}],          # the rules know the place
                     chain + [{"role": "user", "content": "Fail box A"}],                   # not a weather follow-up
                     chain + [{"role": "user", "content": "And Reykjavik?"}],               # names its own place
                     [{"role": "user", "content": "And tomorrow?"}]):
            with self.subTest(last=msgs[-1]["content"]):
                self.assertIsNone(A.followup_hint(msgs))
        # a conversion in between names no place: the follow-up still belongs to the Chicago question
        msgs = self.FIRST + [{"role": "user", "content": "What is 20 C in Fahrenheit?"},
                             {"role": "assistant", "content": "20 °C equals 68 °F."}, {"role": "user", "content": "And tomorrow?"}]
        self.assertEqual(A.followup_question(msgs), "chicago: And tomorrow?")
        self.assertIsNone(A.followup_hint(msgs))
        # a command, a status question or thanks in between do not hide the place
        for between, reply in (("Fail box A", "Killed the forecast server on box-a."),
                               ("Which box is serving right now?", "Both boxes are serving."), ("thanks", "You're welcome!")):
            with self.subTest(between=between):
                msgs = self.FIRST + [{"role": "user", "content": between}, {"role": "assistant", "content": reply},
                                     {"role": "user", "content": "And tomorrow?"}]
                self.assertEqual(A.followup_question(msgs), "chicago: And tomorrow?")

    def test_a_second_place_the_list_does_not_know_goes_to_the_llm(self):
        for text in ("Is it warmer in Chicago or St. Louis?", "Weather in Chicago and St. Louis tomorrow?",
                     "Compare Chicago with Knoxville", "Is it warmer in Knoxville or Chicago?",     # named first, too
                     "Is St. Louis warmer than Chicago?", "Chicago vs. Nashville tomorrow"):
            with self.subTest(text=text):
                self.assertFalse(A.weather_question(text))
        for text in ("Will it be warmer in Denver or Miami tomorrow?", "London, Paris and Rome tomorrow",
                     "Will it rain or snow in Chicago?", "Chicago today and tomorrow",
                     "Is it warmer in Denver or Miami right now?", "Highs and lows in Chicago tomorrow",
                     "Do I need a coat or an umbrella in Boston tomorrow?", "Is Chicago colder than Denver or Boston?"):
            with self.subTest(text=text):
                self.assertTrue(A.weather_question(text))

    def test_what_is_not_a_weather_question(self):
        for text in ("Is 1,000 steps enough for the fine-tune?", "How long would 10,000 steps take?",
                     "p95 was 1,200 ms, is that normal?",                       # a thousands comma is not "lat, lon"
                     "Which box answered the Tokyo question?", "How long did the Tokyo forecast take?",
                     "Was the Tokyo forecast handled by one machine?"):
            with self.subTest(text=text):
                self.assertFalse(A.weather_question(text))
        for text in ("p95 was 1,234.5 ms, is that normal?", "Is 99,9 % uptime realistic?",    # not a place on the globe
                     "What happened on Nov 15, 2026?", "Was the Tokyo forecast simulated?",
                     "Did the Chicago forecast come from box-a?"):
            with self.subTest(text=text):
                self.assertFalse(A.weather_question(text))
        for text in ("41.88, -87.63 tomorrow", "Forecast for 37.7, -122.4", "35.68,139.69", "41, 100", "40,-74"):
            with self.subTest(text=text):
                self.assertTrue(A.weather_question(text))
        self.assertIsNone(A.find_coords("lat 95, 200"))
        self.assertEqual(A.find_coords("at -33.9, 151.2 now").group(0), "-33.9, 151.2")
        # a fault question that names a city is for WhatIf, as before this change
        for text in ("Why is the Tokyo forecast timing out?", "Why was the Tokyo forecast slow?",
                     "What happened to the Tokyo forecast?", "Diagnose the Tokyo request"):
            with self.subTest(text=text):
                self.assertTrue(A.infra_question(text))
        for text in ("Why is it so cold in Chicago?", "What's going on in Chicago?", "Explain the weather in Tokyo"):
            with self.subTest(text=text):       # a weather question with "why" stays one
                self.assertFalse(A.infra_question(text))
        self.assertIsNone(A.FAULT.search("why is it so warm in lagos"))      # "lag" is not in "Lagos"
        self.assertTrue(A.FAULT.search("the forecast lags") and A.FAULT.search("it crashed"))

    def test_which_of_these_is_warmest_compares_the_places_asked_before(self):
        class Places(FakeOps):
            async def forecast(self, lat, lon, hours, members=1, place=None):
                top = {"Tokyo": 16.0, "New York": 11.0, "Chicago": 9.7}[place]
                return {"place": place, "hours_ahead": hours, "summary": {"min_c": top - 6, "max_c": top, "precip_mm": None,
                                                                         "max_wind_ms": 5.0},
                        "low_confidence": False, "points": [], "members_temp": None, "served_by": "box-a"}
        msgs = self.FIRST + [{"role": "user", "content": "And what about New York?"},
                             {"role": "assistant", "content": "**New York, next 48 h:** 5 °C to 11 °C, wind up to 18 km/h."},
                             {"role": "user", "content": "Tokyo this weekend?"},
                             {"role": "assistant", "content": "**Tokyo, next 120 h:** 10 °C to 16 °C, wind up to 18 km/h."},
                             {"role": "user", "content": "Which of these is warmest?"}]
        answer, llm = self.with_llm(msgs, Places())
        self.assertEqual(llm.posts, [])                                  # the weather model answers, not the LLM
        self.assertEqual([s["args"]["place"] for s in answer["steps"] if s["tool"] == "get_forecast"],
                         ["Tokyo", "New York", "Chicago"])
        self.assertTrue(answer["reply"].endswith("Warmest: **Tokyo** (up to 16 °C)."), answer["reply"])
        # with one place asked before there is nothing to compare: the LLM (and its check) gets the question
        self.assertIsNone(A.followup_question(self.FIRST + [{"role": "user", "content": "Which of these is warmest?"}]))

    def test_which_of_these_is_compared_only_when_every_place_can_be(self):
        def u(text):
            return {"role": "user", "content": text}

        def a(text):
            return {"role": "assistant", "content": text}
        ny = [u("And what about New York?"), a("**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h.")]
        two = self.FIRST + ny
        for text in ("Which of these is warmest?", "Where is it coldest?", "And which one is windier tomorrow?",
                     "Which is the warmest of the two?", "Which one's warmer?"):
            with self.subTest(text=text):
                self.assertTrue(str(A.followup_question(two + [u(text)])).startswith("new york, chicago: " + text))
        # a second comparison, or one after "And tomorrow?", still compares the same places
        for between, reply in (("Which of these is warmest?", "**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h.\n"
                                                              "**Chicago, next 48 h:** 3 °C to 10 °C, wind up to 19 km/h.\n"
                                                              "Warmest: **New York** (up to 10 °C)."),
                               ("And tomorrow?", "**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h.")):
            with self.subTest(between=between):
                msgs = two + [u(between), a(reply), u("Which is coldest?")]
                self.assertTrue(str(A.followup_question(msgs)).startswith("new york, chicago: "))
        # right after a status question, a command, thanks or a conversion, "which one is hotter?" may mean the
        # boxes: the LLM gets it
        for between, reply in (("Which box is serving right now?", "Both boxes are serving."),
                               ("Fail box A", "Killed the forecast server on box-a."), ("thanks", "You're welcome!"),
                               ("What is 20 C in Fahrenheit?", "20 °C is 68 °F.")):
            with self.subTest(between=between):
                self.assertIsNone(A.followup_question(two + [u(between), a(reply), u("Which one is hotter?")]))
        # Not every place can be compared here: the LLM gets the question (and its numbers are checked).
        st_louis = [u("What about St. Louis?"), a("**St. Louis, next 48 h:** 0.5 °C to 10.2 °C, wind up to 7 km/h.")]
        more = [u("And Boston tomorrow?"), a("**Boston, next 48 h:** 5 °C to 8 °C, wind up to 21 km/h."),
                u("Same for Miami please"), a("**Miami, next 48 h:** 20 °C to 25 °C, wind up to 21 km/h.")]
        by_numbers = [u("Weather at 37.7, -122.4 for the next 3 days"),
                      a("**37.7, -122.4, next 72 h:** 9 °C to 15 °C, wind up to 14 km/h.")]
        for name, msgs in (("a place outside the list, answered by the LLM", two + st_louis),
                           ("the same, asked before the second city", self.FIRST + st_louis + ny),
                           ("four cities: one answer holds three forecasts", two + more),
                           ("a place given as coordinates", by_numbers + two),
                           ("the start of the chat is not here (the page sends the last 12 messages)",
                            [a("**Miami, next 48 h:** 20 °C to 25 °C, wind up to 21 km/h.")] + two),
                           ("12 messages: all the page sends, so an older place may be missing (a failed request "
                            "leaves two user messages in a row)",
                            [u("And Chicago?"), u("And Chicago?"), a("**Chicago, next 48 h:** 3 °C to 10 °C."),
                             u("Thanks!"), a("You're welcome!"), u("Which box is serving right now?"),
                             a("Both boxes are serving."), u("Weather in Chicago?"),
                             a("**Chicago, next 24 h:** 3 °C to 10 °C."), u("And New York?"),
                             a("**New York, next 24 h:** 5 °C to 10 °C.")])):
            with self.subTest(case=name):
                self.assertIsNone(A.followup_question(msgs + [u("Which of these is warmest?")]))
        # One word that is not a comparison word, and it is not this kind of question.
        for text in ("Which is warmer, St. Louis or Nashville?", "Where is it colder, here or in St. Louis?",
                     "Where is a good hotel near the venue?", "Which time window does the forecast cover?",
                     "Where does the wind data come from?", "Which units is the wind in?",
                     "Which cooling do the Sparks use?", "Which day is warmest?", "Which is warmer, today or tomorrow?",
                     "Where can I see the wind on the page?", "Which model needs a warm-up?",
                     "Which days are warmest?", "Which of the next 3 days is the warmest?", "Which hours are windiest?",
                     "Which week is colder?", "Which weekend is warmer?"):
            with self.subTest(text=text):
                self.assertFalse(A.which_question(text.lower()))
                self.assertIsNone(A.followup_question(two + [u(text)]))

    def test_a_closing_line_after_a_change_is_not_sent_back(self):
        ops = FakeOps(stacks=BOTH)
        closing = "Killed the forecast server on box-a. I'll check on it again whenever you ask."
        answer, llm = self.with_llm([{"role": "user", "content": "Fail box A"}], ops,
                                    tool_call("kill_forecast_server", box="box-a"), says(closing))
        self.assertEqual(len(llm.posts), 2)                 # no "go ahead: call the tools now" after the kill
        self.assertEqual(answer["reply"], closing)
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])

    def test_not_for_real_words_stop_a_real_action_without_the_llm_too(self):
        for text in ("Kill box A (sim)", "Kill box A in test mode", "Fail box A (hypothetical)", "Kill box A virtually",
                     "Kill the fake server on box A", "Simulate a failure on box A", "Kill box A, but only pretend",
                     "Kill box A (mocked)", "Stop the training job on box B (sim)", "Start a fake training job on box B",
                     "Kill box A _simulated_", "Kill box A unsimulated"):       # the last two were refused before, too
            with self.subTest(text=text):
                ops = FakeOps(stacks=BOTH)
                self.assertIn("Nothing was changed", ask(text, ops)["reply"])
                self.assertEqual(ops.calls, [])
        ops = FakeOps(stacks=BOTH)
        ask("Fail box A as a test", ops)                    # a plain command: carried out
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])

    def test_a_step_count_with_a_thousands_comma_is_read_whole(self):
        ops = FakeOps(stacks=BOTH)
        ask("Start a training job on box B with 2,000 steps", ops)
        self.assertEqual(ops.calls, [("start_finetune", "box-b", 2000)])
        ops = FakeOps(stacks=BOTH)
        ask("Start a training job on box B, 500 steps", ops)
        self.assertEqual(ops.calls, [("start_finetune", "box-b", 500)])

    def test_how_long_and_how_fast_about_the_weather_or_about_the_service(self):
        for text in ("How fast is the wind in Chicago?", "How fast will the wind be in Chicago tomorrow?",
                     "How long will the rain last in Chicago?", "How long will it stay cold in Chicago?"):
            with self.subTest(text=text):
                self.assertTrue(A.weather_question(text))
        for text in ("How fast was the Tokyo forecast?", "How long does the Tokyo forecast take?",
                     "How long did the Tokyo forecast take?", "Why did the Tokyo forecast take so long?",
                     # a known city linked to a place the list does not know, with a word in between
                     "Weather in Chicago, or St. Louis?", "Weather in Chicago and also St. Louis",
                     "Weather in Chicago and then St. Louis", "Weather in Chicago and what about St. Louis?"):
            with self.subTest(text=text):
                self.assertFalse(A.weather_question(text))
        for text in ("Weather in Chicago and also Miami", "London, Paris and Rome tomorrow",
                     "Weather in Chicago and what about the wind?"):
            with self.subTest(text=text):
                self.assertTrue(A.weather_question(text))

    def test_that_in_fahrenheit_is_arithmetic_only_when_the_message_names_nothing_else(self):
        for text in ("Can you give me that in Fahrenheit?", "What is 20 C in Fahrenheit?", "convert 10 m/s to km/h",
                     "And those in mph please"):
            with self.subTest(text=text):
                self.assertTrue(A.asks_for_arithmetic(text))
        for text in ("Can I get that in Fahrenheit?", "How much is that in Fahrenheit?"):
            with self.subTest(text=text):
                self.assertTrue(A.asks_for_arithmetic(text))
        for text in ("And St. Louis, in Fahrenheit this time?", "Can I get that in Fahrenheit for St. Louis?",
                     "Now those numbers for Nashville in F please", "Convert that to mph and add Kyoto",
                     "Give me that in Fahrenheit, 63103", "Convert that to mph, 東京", "And this in Celsius: Москва"):
            with self.subTest(text=text):               # a place: the reply's numbers must come from a forecast
                self.assertFalse(A.asks_for_arithmetic(text))
                self.assertTrue(A.unsourced_weather("St. Louis: 41 °F to 57 °F, wind up to 9 mph.", [], text))

    def test_the_numbers_of_the_cluster_status_are_not_weather(self):
        status = [{"tool": "get_cluster_status", "result": {
            "policy": "balanced", "rps": 15.0, "errors_10s": 0, "p95_ms": 196.3,
            "config": {"window_s": 1.0, "cooldown_s": 10.0, "min_gain": 0.2},
            "boxes": {"box-a": {"healthy": True, "requests": 75, "gpu": {"temp_c": 54, "util_pct": 63}}}}}]
        asked = "Which box is up, and how cold will St. Louis get tonight?"
        for reply in ("Both boxes are serving. St. Louis will be between 0 °C and 10 °C tonight.",
                      "St. Louis: 1 °C to 10 °C, wind up to 15 km/h.", "The GPU is at 63 °C."):
            with self.subTest(reply=reply):
                self.assertTrue(A.unsourced_weather(reply, status, asked))
        self.assertFalse(A.unsourced_weather("The GPU of box-a is at 54 °C.", status, "How hot is the GPU?"))
        # a count or a time after "high" or "low" is not a temperature
        for reply in ("p95 latency is at a high of 196 ms over the last 10 s.", "Yes. Latency is low at 196 ms (p95).",
                      "Load is at a high of 15 requests per second, all on box-b."):
            with self.subTest(reply=reply):
                self.assertFalse(A.unsourced_weather(reply, status, "How is the latency right now?"))
        self.assertTrue(A.unsourced_weather("St. Louis: a high of 10 and a low of 0.", status, asked))
        place = [{"tool": "geocode", "result": {"place": "Chicago", "lat": 41.88, "lon": -87.63}}]
        self.assertFalse(A.unsourced_weather("Chicago is at 41.88° N, 87.63° W.", place, "Where is Chicago?"))

    def test_a_wind_figure_in_km_per_hour_is_seen_however_it_is_written(self):
        for reply in ("Wind in St. Louis is up to 15 km/hr.", "Wind up to 15 km/hour.", "Wind up to 15 km/hrs.",
                      "Wind up to 15 km/h.", "Wind up to 15 kmph."):
            with self.subTest(reply=reply):
                self.assertEqual(A.weather_values(reply), [15.0])
                self.assertTrue(A.unsourced_weather(reply, [], "And the wind in St. Louis?"))

    def test_a_reply_that_is_cut_to_nothing_becomes_the_models_numbers(self):
        ask_ = [{"role": "user", "content": "Will it be humid in St. Louis tomorrow?"}]
        call = tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=48, place="St. Louis")
        for only in ("Humidity will be around 65%.", "Humidity in St. Louis will be about 65%.", "\u200b", "—"):
            with self.subTest(reply=only):
                answer, _ = self.with_llm(ask_, FakeOps(), call, says(only))
                self.assertTrue(answer["reply"].startswith("**St. Louis, next 48 h:** 1 °C to 2 °C"), answer["reply"])
                self.assertEqual(answer["note"], A.MODEL_NUMBERS_NOTE)
        # a rain sentence that names "St. Louis" is cut whole, not at the dot of "St."
        honest = ("I can't tell you about rain in St. Louis: the model has no rain output. Temperatures will range "
                  "from 1 °C to 2 °C with wind up to 11 km/h.")
        self.assertEqual(A.without_rain_claims(honest), honest)
        self.assertFalse(A.without_rain_claims("No rain is expected in St. Louis tomorrow. 1 °C to 2 °C.").startswith("Louis"))
        # "p.m." ends a sentence when a new one follows: the made-up rain claim after it is still cut
        self.assertEqual(A.without_rain_claims("St. Louis warms from 1 °C to 2 °C by 3 p.m. No rain is expected. "
                                               "Served by box-a."),
                         "St. Louis warms from 1 °C to 2 °C by 3 p.m. Served by box-a.")
        self.assertEqual(A.without_rain_claims("1 °C at 6 a.m. and 2 °C at 3 p.m. on Sunday."),
                         "1 °C at 6 a.m. and 2 °C at 3 p.m. on Sunday.")

    def test_the_comparison_names_a_place_that_the_llm_gave_as_numbers_only(self):
        ask_ = [{"role": "user", "content": "Is St. Louis warmer than Kansas City tonight?"}]
        answer, _ = self.with_llm(ask_, FakeOps(), tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=24),
                                  tool_call("get_forecast", lat=39.1, lon=-94.58, hours_ahead=24),
                                  says("St. Louis is warmer, with a 30% chance of rain in both."))
        self.assertIn("Warmest: **38.63, -90.2**", answer["reply"])
        self.assertNotIn("None", answer["reply"])

    def test_a_fourth_place_is_named_as_not_shown_and_nothing_is_compared(self):
        answer = ask("Which is warmest: London, Paris, Rome or Tokyo?", FakeOps())
        self.assertEqual([s["args"]["place"] for s in answer["steps"] if s["tool"] == "get_forecast"],
                         ["London", "Paris", "Rome"])
        self.assertIn("Not shown: Tokyo (one answer holds 3 forecasts), so no comparison is made.", answer["reply"])
        self.assertNotIn("Warmest", answer["reply"])

    def test_a_place_name_with_a_weather_word_is_not_cut_as_a_statement_about_rain(self):
        for place, reply in (
                ("Thunder Bay", "**Thunder Bay, next 48 h:** 1 °C to 2 °C, wind up to 11 km/h."),
                ("St. Cloud", "Tomorrow in St. Cloud, Minnesota: 1 °C to 2 °C with winds up to 11 km/h. Served by box-a."),
                ("Sunshine Coast", "Sunshine Coast, tomorrow: 1 °C to 2 °C, wind up to 11 km/h."),
                ("Mt. Rainier", "Mt. Rainier tomorrow: 1 °C to 2 °C, wind up to 11 km/h.")):
            with self.subTest(place=place):
                call = tool_call("get_forecast", lat=48.38, lon=-89.25, hours_ahead=48, place=place)
                answer, _ = self.with_llm([{"role": "user", "content": f"Weather in {place}?"}], FakeOps(), call, says(reply))
                self.assertEqual(answer["reply"], reply)
                self.assertNotIn("note", answer)
        # with the name in it a made-up rain statement is still cut, and the name stays
        call = tool_call("get_forecast", lat=48.38, lon=-89.25, hours_ahead=48, place="Thunder Bay")
        answer, _ = self.with_llm([{"role": "user", "content": "Weather in Thunder Bay?"}], FakeOps(), call,
                                  says("Thunder Bay: 1 °C to 2 °C, wind up to 11 km/h. No rain is expected."))
        self.assertEqual(answer["reply"], "Thunder Bay: 1 °C to 2 °C, wind up to 11 km/h.")
        # A weather word that leads into a number and is no place of this turn: nothing can be cut safely
        # (", while Thunder Bay's is -6.3 °C" once came out as "8.6 °C6.3 °C"), so the model's numbers are shown.
        ask_ = [{"role": "user", "content": "Is Seattle warmer than Rainier?"}]
        answer, _ = self.with_llm(ask_, FakeOps(), tool_call("get_forecast", lat=47.6, lon=-122.3, hours_ahead=24),
                                  tool_call("get_forecast", lat=46.85, lon=-121.76, hours_ahead=24),
                                  says("Seattle's range is 1 °C to 2 °C, while Rainier's is 1 °C to 2 °C."))
        self.assertEqual(answer["note"], A.MODEL_NUMBERS_NOTE)
        self.assertNotIn("°C1", answer["reply"])
        # The LLM gave the place another way ("Thunder Bay, Ontario") or not at all: the name is still kept, or
        # the reply becomes the model's numbers; it is never cut down to "Next 48 h: ...".
        named = "**Thunder Bay, next 48 h:** 1 °C to 2 °C, wind up to 11 km/h."
        answer, _ = self.with_llm([{"role": "user", "content": "Weather in Thunder Bay?"}], FakeOps(),
                                  tool_call("get_forecast", lat=48.38, lon=-89.25, hours_ahead=48, place="Thunder Bay, Ontario"),
                                  says(named))
        self.assertEqual(answer["reply"], named)
        answer, _ = self.with_llm([{"role": "user", "content": "Weather in Thunder Bay?"}], FakeOps(),
                                  tool_call("get_forecast", lat=48.38, lon=-89.25, hours_ahead=48), says(named))
        self.assertEqual(answer["note"], A.MODEL_NUMBERS_NOTE)
        self.assertTrue(answer["reply"].startswith("**48.38, -89.25, next 48 h:** 1 °C to 2 °C"), answer["reply"])
        # a real claim that runs into a number is not cut either ("... 2 °C11 km/h" came out before)
        answer, _ = self.with_llm([{"role": "user", "content": "What about St. Louis?"}], FakeOps(),
                                  tool_call("get_forecast", lat=38.63, lon=-90.2, hours_ahead=48, place="St. Louis"),
                                  says("St. Louis: 1 °C to 2 °C, with snow showers and a wind of 11 km/h."))
        self.assertEqual(answer["note"], A.MODEL_NUMBERS_NOTE)
        self.assertNotIn("snow", answer["reply"])

    def test_a_dash_between_two_figures_is_a_range_and_a_dash_before_one_is_a_minus(self):
        self.assertEqual(A.weather_values("St. Louis tomorrow: 0.6°\u20139.2°, wind up to 7 km/h."), [0.6, 9.2, 7.0])
        self.assertEqual(A.weather_values("St. Louis tomorrow: 0.6 \u2103\u20139.2 \u2103."), [0.6, 9.2])
        self.assertEqual(A.weather_values("Reykjavik: \u20114.2 °C to 0.4 °C; Knoxville: \u20130.8 °C \u2013 9.2 °C."),
                         [-4.2, 0.4, -0.8, 9.2])

    def test_a_decade_of_milliseconds_is_not_a_vague_temperature(self):
        status = [{"tool": "get_cluster_status", "result": {"policy": "balanced", "p95_ms": 212.4, "errors_10s": 0}}]
        for reply in ("Latency is in the low 200s right now (p95 212.4 ms).", "p95 is back in the 200s.",
                      "While box-a was down a few requests failed with errors in the 500s (HTTP 503)."):
            with self.subTest(reply=reply):
                self.assertFalse(A.unsourced_weather(reply, status, "How is the latency right now?"))
        for reply in ("Highs in the low 40s.", "It stays in the teens all day.", "Single digits tonight."):
            with self.subTest(reply=reply):
                self.assertTrue(A.unsourced_weather(reply, [], "Weather in St. Louis?"))

    def test_zero_and_32_pass_only_as_the_freezing_point(self):
        ran = [{"tool": "get_forecast", "result": {"summary": {"min_c": 3.2, "max_c": 11.0, "max_wind_ms": 4.2,
                                                               "precip_mm": None}, "points": []}}]
        self.assertFalse(A.unsourced_weather("3 °C to 11 °C: it stays above freezing (0 °C).", ran, "Will it freeze?"))
        self.assertFalse(A.unsourced_weather("38 °F to 52 °F, above the 32 °F freezing point.", ran, "Will it freeze?"))
        self.assertTrue(A.unsourced_weather("Evanston: 0 °C to 11 °C.", ran, "Weather in Evanston?"))
        self.assertFalse(A.unsourced_weather("No. The low is 3.2 °C, so it stays above 0 °C.", ran, "Will it drop below zero?"))
        for made_up in ("0 °C to 11 °C, a freezing start.", "No freeze: 3.2 °C to 11 °C, wind up to 32 km/h.",
                        "A high of 32 °F, right at freezing."):
            with self.subTest(reply=made_up):
                self.assertTrue(A.unsourced_weather(made_up, ran, "Weather in Evanston?"))

    def test_an_announcement_that_opens_the_reply_is_sent_back_even_after_a_change(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.with_llm([{"role": "user", "content": "Fail box A"}], ops,
                                    tool_call("kill_forecast_server", box="box-a"),
                                    says("I'll check the status of box-a now."),
                                    tool_call("get_cluster_status"), says("box-a is restarting."))
        self.assertEqual((len(llm.posts), answer["reply"]), (4, "box-a is restarting."))
        self.assertEqual(ops.calls, [("kill_forecast_server", "box-a")])

    def test_what_a_follow_up_is_told_after_a_comparison_and_after_a_withheld_answer(self):
        def u(text):
            return {"role": "user", "content": text}

        def a(text):
            return {"role": "assistant", "content": text}
        compared = self.FIRST + [u("And what about New York?"), a("**New York, next 48 h:** 5 °C to 10 °C, wind up to 23 km/h."),
                                 u("Which of these is warmest?"),
                                 a("**New York, next 48 h:** 5 °C to 10 °C.\n**Chicago, next 48 h:** 3 °C to 10 °C.\n"
                                   "Warmest: **New York** (up to 10 °C).")]
        self.assertIsNone(A.followup_hint(compared + [u("And tomorrow?")]))        # a comparison names no place
        # the answer about St. Louis was withheld: the follow-up is still about St. Louis, not about Chicago
        withheld = self.FIRST + [u("What about St. Louis?"), a(A.NO_FORECAST_REPLY)]
        self.assertIsNone(A.followup_question(withheld + [u("And tomorrow?")]))
        self.assertEqual(A.followup_hint(withheld + [u("And tomorrow?")]), "What about St. Louis?")
        self.assertFalse(A.which_question("which " * 8000))                        # and at once: the length is tested first
        self.assertIn("only when no other box can", A.describe_box("box-b", {"healthy": True, "mode": "normal", "drained": True}))

    def test_two_places_get_two_forecasts_and_the_comparison(self):
        class Two(FakeOps):
            async def forecast(self, lat, lon, hours, members=1, place=None):
                warm = place == "Miami"
                return {"place": place, "summary": {"min_c": 20.0 if warm else -1.0, "max_c": 27.0 if warm else 6.0,
                                                    "precip_mm": None, "max_wind_ms": 5.0},
                        "low_confidence": False, "points": [], "members_temp": None, "served_by": "box-a"}
        answer, llm = self.with_llm([{"role": "user", "content": "Will it be warmer in Denver or Miami tomorrow?"}], Two())
        self.assertEqual(llm.posts, [])
        self.assertEqual(tools_of(answer), ["geocode", "get_forecast", "geocode", "get_forecast"])
        self.assertEqual([s["args"]["place"] for s in answer["steps"] if s["tool"] == "get_forecast"], ["Denver", "Miami"])
        self.assertIn("**Denver, next 48 h:**", answer["reply"])
        self.assertIn("**Miami, next 48 h:**", answer["reply"])
        self.assertTrue(answer["reply"].endswith("Warmest: **Miami** (up to 27 °C)."), answer["reply"])
        answer, _ = self.with_llm([{"role": "user", "content": "Weather in Paris and London tomorrow?"}], Two())
        self.assertNotIn("Warmest", answer["reply"])    # no comparison was asked for

    def test_no_statement_about_rain_that_the_forecast_did_not_give(self):
        none = [{"tool": "get_forecast", "result": {"summary": {"min_c": 1.0, "precip_mm": None}}}]
        rain = [{"tool": "get_forecast", "result": {"summary": {"min_c": 1.0, "precip_mm": 0.0}}}]
        for reply, cleaned in [     # replies as Nemotron wrote them on the event boxes
                ("Tonight in Chicago the temperature will drop to about 4 °C, with winds around 19 km/h. No rain is forecast.",
                 "Tonight in Chicago the temperature will drop to about 4 °C, with winds around 19 km/h."),
                ("Seattle’s forecast for the next few days: temperatures between 7 °C and 8 °C, light winds around 15 km/h, "
                 "and no rain is expected.",
                 "Seattle’s forecast for the next few days: temperatures between 7 °C and 8 °C, light winds around 15 km/h."),
                ("Denver this weekend: -1 °C to 9.5 °C, light wind around 6 km/h. No rain forecast. Served by box-a.",
                 "Denver this weekend: -1 °C to 9.5 °C, light wind around 6 km/h. Served by box-a."),
                ("Boston tomorrow will be 4 °C to 7 °C with light winds around 21 km/h. No rain is forecast.",
                 "Boston tomorrow will be 4 °C to 7 °C with light winds around 21 km/h."),
                ("Dry and mild, 7 °C to 8 °C.", "7 °C to 8 °C."),
                # true statements about the missing data stay
                ("Temperatures between 7 °C and 8 °C, and no rain mentioned (the model doesn’t provide rain data). "
                 "This came from box-b.",
                 "Temperatures between 7 °C and 8 °C, and no rain mentioned (the model doesn’t provide rain data). "
                 "This came from box-b."),
                ("11 °C to 17 °C. No rain data available.", "11 °C to 17 °C. No rain data available."),
                ("Chicago: 2.8 °C to 9.7 °C, wind up to 19 km/h, served by box-b.",
                 "Chicago: 2.8 °C to 9.7 °C, wind up to 19 km/h, served by box-b.")]:
            with self.subTest(reply=reply):
                self.assertEqual(A.only_given_numbers(reply, none), cleaned)
                self.assertEqual(A.only_given_numbers(reply, rain), reply)      # the forecast has rain numbers: untouched
                self.assertEqual(A.only_given_numbers(reply, []), reply)
        self.assertEqual(A.only_given_numbers("Seattle: 7 °C to 8 °C. No rain is expected.", none, "Will it rain in Seattle?"),
                         "Seattle: 7 °C to 8 °C. The model has no rain output.")

    def test_the_llm_gets_values_ready_to_quote(self):
        r = {"place": "Paris", "served_by": "box-b", "points": [1], "members_temp": None,
             "summary": {"min_c": 2.84, "max_c": 9.66, "precip_mm": None, "max_wind_ms": 5.3, "humidity_avg_pct": None,
                         "ens_min_c": 1.0, "ens_max_c": 11.0}}
        c, f = A.slim(r, "C"), A.slim(r, "F")
        self.assertEqual(c["say"], {"temperature": "2.8 °C to 9.7 °C", "wind_up_to": "19 km/h", "ensemble_range": "1 °C to 11 °C"})
        self.assertEqual(f["say"], {"temperature": "37 °F to 49 °F", "wind_up_to": "12 mph", "ensemble_range": "34 °F to 52 °F"})
        for seen in (c, f):                             # nothing left for the LLM to convert, or to mislabel
            self.assertEqual(seen["summary"], {})
            self.assertNotIn("points", seen)
            self.assertIn("rain", seen)
        self.assertNotIn("say", A.slim({"points": [], "summary": {"min_c": None, "max_c": None}}))

    def test_the_time_span_and_the_members_a_question_asks_for(self):
        for text, hours in (("give me a 3-day forecast for tokyo", 72), ("five-day outlook for paris", 120),
                            ("next 36 hours", 36), ("48-hour forecast", 48), ("three days", 72), ("the next few days", 72),
                            ("tomorrow", 48), ("tonight", 12), ("this week", 120), ("weather in chicago", 24),
                            ("next 10 days", 120)):
            with self.subTest(text=text):
                self.assertEqual(A.parse_hours(text), hours)
        for text, members in (("Give me a 3-day forecast for Tokyo with 10 ensemble members.", 10),
                              ("Chicago, 20 members, 5 days", 20), ("Chicago with 50 members", 20),
                              ("Ensemble range for Tokyo, 3 days", 10), ("Weather in Tokyo", 1)):
            with self.subTest(text=text):
                answer, _ = self.with_llm([{"role": "user", "content": text}], FakeOps())
                self.assertEqual(answer["steps"][-1]["args"]["ensemble_members"], members)
        answer, _ = self.with_llm([{"role": "user", "content": "Give me a 3-day forecast for Tokyo with 10 ensemble members."}], FakeOps())
        self.assertEqual(answer["steps"][-1]["args"]["hours_ahead"], 72)

    def test_geocode_does_not_guess(self):
        self.assertIn("error", A.geocode(""))
        self.assertIn("error", A.geocode("   "))
        self.assertIn("error", A.geocode("St. Louis"))
        self.assertIn("error", A.geocode("Lagos"))      # "la" inside another word is not Los Angeles
        self.assertEqual(A.geocode("LA")["place"], "Los Angeles")
        self.assertEqual(A.geocode("Chicago, IL")["place"], "Chicago")
        self.assertEqual(A.geocode("Paris, France")["lat"], 48.86)
        self.assertEqual(A.geocode("Chicago"), {"place": "Chicago", "lat": 41.88, "lon": -87.63, "approximate": True})
        self.assertEqual(A.geocode("nyc")["lat"], 40.71)

    def test_a_weather_question_with_a_capital_letter_is_not_an_infrastructure_question(self):
        self.assertFalse(A.infra_question("What's going on in Miami tomorrow?"))
        self.assertFalse(A.infra_question("what's going on in miami tomorrow?"))
        self.assertTrue(A.infra_question("Why is box-b slow?"))
