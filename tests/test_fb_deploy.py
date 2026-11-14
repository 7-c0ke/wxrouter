"""Tests for deploy/fb.py: parsing of real fuzzball CLI output, and the commands each action runs.

    python -m unittest tests/test_fb_deploy.py

The CLI text below was captured from the event cluster (fuzzball 4.3.1) on Nov 14, 2026.
No cluster is needed: actions run with --dry-run, which prints commands instead of running them.
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
import fb  # noqa: E402

DESCRIBE_RUNNING = """Name:      wxrouter-team14
Status:    STAGE_STATUS_STARTED
Error:
KIND     | STATUS   | NAME                                 | DURATION | PERSISTENT VOLUME | VOLUME REFERENCE | DRIVER ID
Workflow | Started  | f81cfa72-2973-478a-b1bb-2b5fbb8cafa4 | 56s      |                   |                  |
Image    | Finished | docker://python:3.12-slim (arm64)    | 0s       |                   |                  |
Service  | Started  | box-a                                | 55s      |                   |                  |
Service  | Started  | box-b                                | 55s      |                   |                  |
Service  | Started  | router                               | 55s      |                   |                  |
Service  | Started  | ui                                   | 54s      |                   |                  |
Job      | Started  | hold                                 | 54s      |                   |                  |
"""

DESCRIBE_FAILED = """Status:    STAGE_STATUS_FAILED
Error:     failed to probe image cache for "docker://python:3.12-slim": rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp: lookup fuzzball-objectcache on 10.96.0.10:53: no such host"

KIND     | STATUS  | NAME                                 | DURATION | PERSISTENT VOLUME | VOLUME REFERENCE | DRIVER ID
Workflow | Errored | 449d2ce7-7420-45c9-917a-5cc212576471 | 0s       |                   |                  |
Service  | Errored | box-a                                |          |                   |                  |
"""

LIST = """ID                                   | CLUSTER | EMAIL                | STATUS   | NAME
8aaf7d4d-1fa2-43ab-b93b-eb1e50bd3a6f | fb-hack | team14@fb-hack.local | Finished | sat-team14-run4
f81cfa72-2973-478a-b1bb-2b5fbb8cafa4 | fb-hack | team14@fb-hack.local | Started  | wxrouter-team14
"""


def dry(*argv):
    """Run an action with --dry-run; return what it printed. Streaming actions (open, logs) end with
    sys.exit(0) on purpose; that counts as success here."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fb.main(["--dry-run", *argv])
        except SystemExit as e:
            if e.code not in (0, None):
                raise
    return out.getvalue()


class Parsing(unittest.TestCase):
    def test_describe_running(self):
        d = fb.parse_describe(DESCRIBE_RUNNING)
        self.assertEqual(d["status"], "STAGE_STATUS_STARTED")
        self.assertEqual(d["stages"]["router"], ("Service", "Started"))
        self.assertEqual(d["stages"]["hold"], ("Job", "Started"))
        self.assertEqual(d["error"], "")

    def test_describe_failed(self):
        d = fb.parse_describe(DESCRIBE_FAILED)
        self.assertEqual(d["status"], "STAGE_STATUS_FAILED")
        self.assertIn("objectcache", d["error"])
        self.assertEqual(d["stages"]["box-a"], ("Service", "Errored"))

    def test_list(self):
        rows = fb.parse_list(LIST)
        self.assertEqual([r["name"] for r in rows], ["sat-team14-run4", "wxrouter-team14"])
        self.assertEqual(rows[1]["status"], "Started")

    def test_start_id_bare_and_sentence(self):
        wid = "449d2ce7-7420-45c9-917a-5cc212576471"
        self.assertEqual(fb.parse_start(wid + "\n"), wid)
        self.assertEqual(fb.parse_start(f'Workflow "{wid}" started.'), wid)
        self.assertIsNone(fb.parse_start("nothing here"))

    def test_explain_known_errors(self):
        self.assertIn("network path", fb.explain('dial tcp 172.16.0.133:443: connect: connection refused'))
        self.assertIn("image cache", fb.explain("lookup fuzzball-objectcache on 10.96.0.10:53"))
        self.assertIsNone(fb.explain("some other error"))


class DryRuns(unittest.TestCase):
    def setUp(self):
        self._state = fb.STATE
        fb.STATE = Path(self._testMethodName + ".json")  # not created: dry runs never write state

    def tearDown(self):
        fb.STATE = self._state

    def test_up_real_validates_then_starts(self):
        out = dry("up", "--target", "real")
        self.assertIn("workflow validate fuzzball/30-real-stack.fz", out)
        self.assertIn("workflow start -n real-stack fuzzball/30-real-stack.fz", out)
        self.assertLess(out.index("validate"), out.index("start -n"))

    def test_there_is_no_mock_target(self):
        self.assertEqual(list(fb.TARGETS), ["real"])
        with self.assertRaises(SystemExit):
            dry("up", "--target", "mock")

    def test_setup_runs_jobs_in_order(self):
        out = dry("setup")
        self.assertLess(out.index("05-setup-volume.fz"), out.index("04-fetch-llm.fz"))
        self.assertLess(out.index("04-fetch-llm.fz"), out.index("06-gfs-year.fz"))

    def test_setup_only_one(self):
        out = dry("setup", "--only", "llm")
        self.assertIn("04-fetch-llm.fz", out)
        self.assertNotIn("06-gfs-year.fz", out)

    def test_status_open_logs_down_with_id(self):
        wid = "f81cfa72-2973-478a-b1bb-2b5fbb8cafa4"
        self.assertIn(f"workflow describe {wid}", dry("status", "--id", wid))
        self.assertIn(f"workflow log {wid} box-a -f", dry("logs", "box-a", "-f", "--id", wid))
        self.assertIn(f"workflow stop {wid}", dry("down", "--id", wid))
        self.assertIn(f"workflow port-forward {wid} ui 18080:8080", dry("open", "--id", wid, "--port", "18080"))

    def test_no_remembered_id_is_a_clear_error(self):
        with self.assertRaises(SystemExit) as e:
            dry("status", "--target", "real")
        self.assertIn("No deployed 'real' workflow", str(e.exception))

    def test_files_named_in_targets_exist(self):
        root = Path(fb.ROOT)
        for t in fb.TARGETS.values():
            self.assertTrue((root / t["file"]).exists() or "build" in t, t["file"])
        for file, _, _ in fb.SETUP_JOBS.values():
            self.assertTrue((root / file).exists(), file)


if __name__ == "__main__":
    unittest.main()
