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
    try:
        spec.loader.exec_module(module)
    finally:
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
        answer, _ = self.ask_llm("Weather in Chicago if it rains?", FakeOps(),
                                 tool_call("get_forecast", lat=41.9, lon=-87.6, hours_ahead=24, place="Chicago"),
                                 says("Chicago: 1 to 2 degrees."))
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
        answer, _ = self.ask_llm("Weather in Chicago for the next 48 hours?", ops,
                                 tool_call("get_forecast", lat=41.9, lon=-87.6, hours_ahead=48, place="Chicago"),
                                 says("Chicago: 1 °C to 2 °C, wind up to 11 km/h, humidity 71 %."))
        self.assertEqual(answer["reply"], "Chicago: 1 °C to 2 °C, wind up to 11 km/h.")


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

    def test_an_announcement_is_followed_up_and_the_tools_run(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm(
            "Will it be warmer in Denver or Miami tomorrow at noon?", ops,
            says("I need to check the forecast for both cities. Let me start by getting their coordinates."),
            tool_call("geocode", place="Denver"),
            says("Miami will be warmer."))
        self.assertEqual(answer["reply"], "Miami will be warmer.")
        self.assertEqual(tools_of(answer), ["geocode"])
        self.assertEqual(len(llm.posts), 3)
        # (the fake keeps a reference to the one conversation list, so look for the follow-up in it)
        self.assertIn({"role": "user", "content": A.GO_AHEAD}, llm.posts[-1]["messages"])

    def test_it_gives_up_after_two_follow_ups(self):
        ops = FakeOps(stacks=BOTH)
        answer, llm = self.ask_llm("Weather in Denver?", ops,
                                   says("Let me check."), says("Let me check that for you."),
                                   says("Let me look it up."))
        self.assertEqual(len(llm.posts), 3)
        self.assertEqual(answer["reply"], "Let me look it up.")

    def test_a_real_answer_is_not_followed_up(self):
        for reply in ("Tomorrow at noon Denver is 4 °C. Let me know if you need more.",
                      "Box A is healthy and serving traffic."):
            with self.subTest(reply=reply):
                answer, llm = self.ask_llm("Weather in Denver?", FakeOps(stacks=BOTH), says(reply))
                self.assertEqual(len(llm.posts), 1)

    def test_announces_action(self):
        for text, want in (("Let me start by getting their coordinates.", True), ("I'll check box-b.", True),
                           ("One moment.", True), ("Let me know if you need more.", False),
                           ("It will be 4 °C.", False), ("", False), ("Let me check. " * 60, False)):
            with self.subTest(text=text[:40]):
                self.assertEqual(A.announces_action(text), want)
