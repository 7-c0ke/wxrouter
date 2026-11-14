"""Tests for tools/fb_record.py, against a fake "fuzzball" command.

Run from the repo root (about 15 seconds):

    .venv/bin/python -m unittest tests/test_fb_record.py -v
    .venv/bin/python -m unittest tests.test_fb_record.ScrubTest -v

Written at the hackathon (Nov 13-14, 2026).

No real fuzzball, no cluster and no network are used. The fake is a small Python script in a
temporary folder, started by a two-line launcher that FUZZBALL_BIN points to. It prints canned
text for each command and writes down every command line it is given. The canned text is made
up: the real tool's layout is not known here, and only the secrets in the text matter.

Most tests run the recorder as a program, from a throwaway copy of the repo layout, so nothing
is written into the real repo. Every temporary folder is removed when its test ends.
"""

import base64
import calendar
import contextlib
import io
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "fb_record.py")
sys.path.insert(0, os.path.join(REPO, "tools"))
import fb_record as fb  # noqa: E402

UUID = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
MARK = "[removed]"


def make_jwt():
    """Text shaped like a JSON Web Token. It is built here so that no token-like text sits in the repo."""
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode("utf-8")).decode("ascii").rstrip("=")
    return ".".join([part({"alg": "HS256", "typ": "JWT"}), part({"sub": "nobody", "aud": "tests"}),
                     part("not a real signature")])


# Every made-up secret the fake prints. None of them may appear in anything the recorder writes or prints.
JWT = make_jwt()
SECRETS = {
    "token in JSON": JWT,
    "after Bearer, in a log": "fake-bearer-7f3a",
    "after Bearer, in a script": "fake-bearer-9c1d",
    "FB_TOKEN line": "abc.def",
    "env list": "fake-env-key-11",
    "env list, with spaces": "fake pass phrase 22",
    "script": "fake script word 33",
    "JSON inside JSON": "fake-inner-44",
    "Python arguments": "fake-repr-key-55",
    "Python dict": "fake-dict-key-66",
    "describe text": "fake-text-key-77",
    "error text": "fake-error-key-88",
}

VERSION_OUT = "Client Version: v4.4.0\nServer Version: v4.4.0\n"
NODES_OUT = ("NAME     STATUS   CORES   MEMORY    DEVICES\n"
             "gb10-a   Ready    20      128GiB    nvidia.com/gpu=1\n"
             "gb10-b   Ready    20      128GiB    nvidia.com/gpu=1\n")
EVENTS_OUT = ("15:00:01  forecast  image ready (cached)\n"
              "15:00:02  forecast  started on gb10-a, cores 0-3, nvidia.com/gpu 0\n"
              "15:00:05  llm       started on gb10-b\n"
              "15:00:41  llm       readiness probe passed\n")
DESCRIBE_OUT = ("Name:      node-stack\n"
                "Id:        " + UUID + "\n"
                "Status:    STARTED\n"
                "Jobs:      forecast (STARTED on gb10-a)\n"
                "Services:  llm (STARTED on gb10-b)\n"
                "Environment of llm:\n"
                "  MODEL=qwen3\n"
                "  LLM_API_KEY=" + SECRETS["describe text"] + "\n"
                "  WEIGHTS_TOKEN=secret://user/weights-token\n")
DESCRIBE_CLEAN = DESCRIBE_OUT.replace(SECRETS["describe text"], MARK)
LOG_FORECAST = ("forecast server: node A, model afno-19m, port 8101\n"
                "FB_JOB_NAME=forecast\n"
                "FB_TOKEN=abc.def\n"
                "request from 10.0.0.7 with Authorization: Bearer " + SECRETS["after Bearer, in a log"] + "\n"
                "caller identity " + JWT + " accepted\n"
                "ready\n")
LOG_FORECAST_CLEAN = ("forecast server: node A, model afno-19m, port 8101\n"
                      "FB_JOB_NAME=forecast\n"
                      "FB_TOKEN=[removed]\n"
                      "request from 10.0.0.7 with Authorization: [removed]\n"
                      "caller identity [removed] accepted\n"
                      "ready\n")
LOG_LLM = ("INFO args: Namespace(model='qwen3', api_key='" + SECRETS["Python arguments"] + "', port=8000)\n"
           "INFO settings {'hf_token': '" + SECRETS["Python dict"] + "', 'threads': 4}\n"
           "slot update_slots: id 0 | task 0 | n_tokens = 14\n"
           "eval time = 3069.59 ms / 128 tokens (41.70 tokens per second)\n")
LOG_LLM_CLEAN = ("INFO args: Namespace(model='qwen3', api_key=[removed], port=8000)\n"
                 "INFO settings {'hf_token': '[removed]', 'threads': 4}\n"
                 "slot update_slots: id 0 | task 0 | n_tokens = 14\n"
                 "eval time = 3069.59 ms / 128 tokens (41.70 tokens per second)\n")
SCRIPT = ("#!/bin/sh\n"
          "export LLM_PASSWORD=\"" + SECRETS["script"] + "\"\n"
          "curl -s -H \"Authorization: Bearer " + SECRETS["after Bearer, in a script"] + "\" http://127.0.0.1:8101/\n"
          "curl -s -H \"Cookie: access-token=${FB_TOKEN}\" \"$FB_API_HOST/v1\"\n"
          "exec python serve/forecast_server.py --port 8101\n")
SCRIPT_CLEAN = ("#!/bin/sh\n"
                "export LLM_PASSWORD=[removed]\n"
                "curl -s -H \"Authorization: [removed]\" http://127.0.0.1:8101/\n"
                "curl -s -H \"Cookie: access-token=${FB_TOKEN}\" \"$FB_API_HOST/v1\"\n"
                "exec python serve/forecast_server.py --port 8101\n")
FUZZFILE = ("version: v4\n"
            "services:\n"
            "  stack-a:\n"
            "    image:\n"
            "      uri: docker://registry.example/forecast:1\n"
            "    env:\n"
            "      - WX_DATA=/data\n"
            "      - HF_TOKEN=secret://user/hf-token\n"
            "    script: |\n"
            "      #!/bin/sh\n"
            "      export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'\n"
            "      curl -fsS -H \"Cookie: access-token=${FB_TOKEN}\" \"$FB_API_HOST/v1\" || true\n"
            "      exec python serve/forecast_server.py --port 8101\n")


def describe_tree(clean=False):
    """What the fake answers to "workflow describe -o json"; with clean=True, what should be saved of it."""
    def pick(secret):
        return MARK if clean else SECRETS[secret]
    config = {"client_secret": pick("JSON inside JSON"), "port": 8000}
    return {
        "id": UUID, "name": "node-stack", "status": "STARTED",
        "owner": {"group": "team-switchyard", "token": MARK if clean else JWT},
        "specification": {
            "version": "v4",
            "jobs": {"forecast": {
                "image": {"uri": "docker://registry.example/forecast:1", "secret": "secret://user/registry"},
                "env": ["WX_DATA=/data", "FORECAST_API_KEY=" + pick("env list"),
                        "DB_PASSWORD=" + pick("env list, with spaces")],
                "script": SCRIPT_CLEAN if clean else SCRIPT,
                "resource": {"cpu": {"cores": 4}, "devices": {"nvidia.com/gpu": 1}}}},
            "services": {"llm": {
                "env": ["MODEL=qwen3", "HF_TOKEN=secret://user/hf-token"],
                "files": {"config.json": config if clean else json.dumps(config)},
                "max_tokens": 512}},
        },
    }


def plan(**changes):
    """The fake's answers: one entry per command, each {"out", "err", "code", "sleep"} (all optional)."""
    answers = {
        "version": {"out": VERSION_OUT},
        "describe": {"out": DESCRIBE_OUT},
        "describe-json": {"out": json.dumps(describe_tree(), indent=2) + "\n"},
        "events": {"out": EVENTS_OUT},
        "log forecast": {"out": LOG_FORECAST},
        "log llm": {"out": LOG_LLM},
        "nodes": {"out": NODES_OUT},
    }
    answers.update(changes)
    return answers


EVENTS_FAIL = {"code": 3, "err": "Error: cannot read events: api_key=" + SECRETS["error text"] + " was refused\n"
                                 "Error: the token " + JWT + " is not valid here\n"}

# The fake fuzzball. It answers only the six commands the recorder may run; anything else gets exit code 64.
FAKE = r'''"""A stand-in for the fuzzball command-line tool, written by tests/test_fb_record.py."""
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
with open(os.path.join(here, "calls.jsonl"), "a") as handle:
    handle.write(json.dumps(args) + "\n")
with open(os.path.join(here, "plan.json")) as handle:
    plan = json.load(handle)

key = None
if args == ["version"]:
    key = "version"
elif args == ["node", "list"]:
    key = "nodes"
elif args[:2] == ["workflow", "describe"] and len(args) == 3:
    key = "describe"
elif args[:2] == ["workflow", "describe"] and len(args) == 5 and args[3:] == ["-o", "json"]:
    key = "describe-json"
elif args[:2] == ["workflow", "events"] and len(args) == 3:
    key = "events"
elif args[:2] == ["workflow", "log"] and len(args) == 6 and args[4] == "-n":
    key = "log " + args[3]
answer = plan.get(key)
if answer is None:
    sys.stderr.write("fake fuzzball: no answer for %r\n" % (args,))
    sys.exit(64)
sys.stdout.write(answer.get("out", ""))
sys.stdout.flush()
sys.stderr.write(answer.get("err", ""))
sys.stderr.flush()
time.sleep(answer.get("sleep", 0))
sys.exit(answer.get("code", 0))
'''


def write(path, text):
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def read(path):
    with io.open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def all_text(folder):
    """The names and the contents of every file in a run folder."""
    return dict((name, read(os.path.join(folder, name))) for name in sorted(os.listdir(folder)))


class Sandbox(object):
    """A throwaway copy of the repo layout with the recorder in it, and a fake fuzzball beside it."""

    def __init__(self, answers=None):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="fb_record_test_"))
        self.repo = os.path.join(self.root, "repo")
        self.fake_dir = os.path.join(self.root, "fake")
        for folder in (os.path.join(self.repo, "tools"), os.path.join(self.repo, "fuzzball"), self.fake_dir):
            os.makedirs(folder)
        self.tool = os.path.join(self.repo, "tools", "fb_record.py")
        shutil.copyfile(TOOL, self.tool)
        self.fuzzfile = os.path.join(self.repo, "fuzzball", "20-node-stack-a.fz")
        write(self.fuzzfile, FUZZFILE)
        script = os.path.join(self.fake_dir, "fake_fuzzball.py")
        write(script, FAKE)
        self.fuzzball = os.path.join(self.fake_dir, "fuzzball")
        write(self.fuzzball, "#!/bin/sh\nexec %s %s \"$@\"\n" % (shlex.quote(sys.executable), shlex.quote(script)))
        os.chmod(self.fuzzball, 0o755)
        self.answer(plan() if answers is None else answers)

    def remove(self):
        shutil.rmtree(self.root, True)

    def answer(self, answers):
        """Set what the fake prints from now on."""
        write(os.path.join(self.fake_dir, "plan.json"), json.dumps(answers))

    def calls(self):
        """Every command line the fake was given so far, in order."""
        path = os.path.join(self.fake_dir, "calls.jsonl")
        if not os.path.exists(path):
            return []
        return [json.loads(line) for line in read(path).splitlines()]

    def run(self, *args, **env):
        """Run the recorder as a program, from the sandbox's repo root."""
        environment = dict(os.environ, FUZZBALL_BIN=self.fuzzball, PYTHONDONTWRITEBYTECODE="1")
        environment.update(env)
        return subprocess.run([sys.executable, self.tool] + list(args), cwd=self.repo, env=environment,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=50)

    def folders(self, out=None):
        """The run folders under evidence/ (or under out), oldest first."""
        out = os.path.join(self.repo, "evidence") if out is None else out
        if not os.path.isdir(out):
            return []
        return [os.path.join(out, name) for name in sorted(os.listdir(out))]


def run_main(args, fuzzball, now=None):
    """Call the recorder's main() in this process. Returns (exit status, what it printed, its error text)."""
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ, {"FUZZBALL_BIN": fuzzball}):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                status = fb.main(args, now=now)
            except SystemExit as stopped:           # argparse ends the program on a bad argument
                status = stopped.code
    return status, out.getvalue(), err.getvalue()


class ScrubTest(unittest.TestCase):
    """The filter that takes secrets out of text."""

    def same(self, text):
        self.assertEqual(fb.scrub(text), text)

    def test_json_web_token(self):
        self.assertEqual(fb.scrub("caller identity " + JWT + " accepted"), "caller identity [removed] accepted")
        self.assertEqual(fb.scrub("token=" + JWT), "token=[removed]")
        self.assertEqual(fb.scrub("x-id: " + JWT + ".fourth.fifth end"), "x-id: [removed] end")    # more parts
        header, payload, _ = JWT.split(".")
        self.assertEqual(fb.scrub("unsigned " + header + "." + payload + ". end"), "unsigned [removed] end")
        self.assertEqual(fb.scrub("id eyJa.b.c end"), "id [removed] end")                          # short, same shape
        self.same("eyJ0 and file.tar.gz")

    def test_bearer(self):
        self.assertEqual(fb.scrub("Authorization: Bearer abc123.x-y_z"), "Authorization: [removed]")
        self.assertEqual(fb.scrub("authorization: bearer abc123"), "authorization: [removed]")
        self.assertEqual(fb.scrub('curl -H "FB-Authorization: Bearer abc123" http://x'),
                         'curl -H "FB-Authorization: [removed]" http://x')
        self.assertEqual(fb.scrub("{'Authorization': 'Bearer abc123'}"), "{'Authorization': '[removed]'}")
        self.assertEqual(fb.scrub("AUTH_KEY=Bearer abc123"), "AUTH_KEY=[removed]")
        self.same("the bearer\nof the message")                                                    # not across lines

    def test_env_lines(self):
        self.assertEqual(fb.scrub("FB_TOKEN=abc.def"), "FB_TOKEN=[removed]")
        self.assertEqual(fb.scrub("A=1\nFB_TOKEN=abc.def\r\nB=2\n"), "A=1\nFB_TOKEN=[removed]\r\nB=2\n")
        self.assertEqual(fb.scrub("DB_PASSWORD=correct horse battery staple"), "DB_PASSWORD=[removed]")
        self.assertEqual(fb.scrub("      - HF_TOKEN=two words"), "      - HF_TOKEN=[removed]")
        self.assertEqual(fb.scrub("AWS_SECRET_ACCESS_KEY=a/b+c=="), "AWS_SECRET_ACCESS_KEY=[removed]")
        self.assertEqual(fb.scrub("my.credential-file=/run/x"), "my.credential-file=[removed]")
        for name in ("api_key", "ApiKey", "client_secret", "Password", "x-credential", "hf_token"):
            self.assertEqual(fb.scrub(name + "=value"), name + "=[removed]", name)
        self.same("FB_JOB_NAME=forecast\nPATH=/usr/bin:/bin\nHOME=/")

    def test_name_value_inside_a_line(self):
        self.assertEqual(fb.scrub("15:01:02 env FB_TOKEN=abc.def more words"),
                         "15:01:02 env FB_TOKEN=[removed] more words")
        self.assertEqual(fb.scrub("export HF_TOKEN=abc"), "export HF_TOKEN=[removed]")
        self.assertEqual(fb.scrub('export HF_TOKEN="two words" && run'), "export HF_TOKEN=[removed] && run")
        self.assertEqual(fb.scrub("env HF_TOKEN='two words' ./run --port 8000"),
                         "env HF_TOKEN=[removed] ./run --port 8000")
        self.assertEqual(fb.scrub("run --port 8000 --api-key=abc --threads 4"),
                         "run --port 8000 --api-key=[removed] --threads 4")
        self.assertEqual(fb.scrub("Namespace(model='qwen3', api_key='abc', port=8000)"),
                         "Namespace(model='qwen3', api_key=[removed], port=8000)")
        self.assertEqual(fb.scrub('GET /v1/models?api_key=abc123&user=me HTTP/1.1" 200'),
                         'GET /v1/models?api_key=[removed] HTTP/1.1" 200')
        self.assertEqual(fb.scrub('["sh", "-c", "DB_PASSWORD=two words"]'), '["sh", "-c", "DB_PASSWORD=[removed]"]')
        self.assertEqual(fb.scrub("['MODE=fast', 'DB_PASSWORD=two words']"), "['MODE=fast', 'DB_PASSWORD=[removed]']")
        # Not assignments: there is nothing to take out.
        for text in ("if token == expected:", "key=", "the key = 5", "n_tokens = 14", "nvidia.com/gpu=1",
                     '{"url": "http://h/?token=", "user": "x"}'):
            self.same(text)

    def test_json_values(self):
        before = {"name": "node-stack", "token": "abc", "apiKey": "x y z",
                  "db": {"PASSWORD": "p\"q\\r", "port": 5432}, "max_tokens": 512, "secret_ok": True,
                  "keys": ["a", "b"], "note": "the key is under the mat"}
        after = dict(before, token=MARK, apiKey=MARK, db={"PASSWORD": MARK, "port": 5432})
        for indent in (None, 2):
            self.assertEqual(json.loads(fb.scrub(json.dumps(before, indent=indent))), after)
        self.assertEqual(fb.scrub("{'hf_token': 'abc', 'threads': 4}"), "{'hf_token': '[removed]', 'threads': 4}")
        self.same('{"a": "token", "b": "key"}')                                                    # words as values

    def test_text_inside_json_strings(self):
        self.assertEqual(json.loads(fb.scrub(json.dumps({"script": SCRIPT}))), {"script": SCRIPT_CLEAN})
        inner = {"client_secret": "in\"ner \\ value", "port": 8000, "list": ["API_KEY=a b", "MODE=fast"]}
        inner_clean = {"client_secret": MARK, "port": 8000, "list": ["API_KEY=[removed]", "MODE=fast"]}
        once = fb.scrub(json.dumps({"files": {"config.json": json.dumps(inner)}}))
        self.assertEqual(json.loads(json.loads(once)["files"]["config.json"]), inner_clean)
        twice = fb.scrub(json.dumps({"outer": json.dumps({"files": json.dumps(inner, indent=2)})}, indent=2))
        self.assertEqual(json.loads(json.loads(json.loads(twice)["outer"])["files"]), inner_clean)
        # A string that is not JSON inside (\$ is no JSON escape) still loses a quoted value.
        self.assertEqual(fb.scrub('sh -c "export API_KEY=\\"two words\\"; echo \\$HOME"'),
                         'sh -c "export API_KEY=[removed]; echo \\$HOME"')

    def test_secret_references_are_kept(self):
        for text in ("      - HF_TOKEN=secret://user/hf-token",
                     '      - "HF_TOKEN=secret://user/hf-token"',
                     "export HF_TOKEN=secret://group/hf.token_2",
                     '{"image": {"uri": "docker://r/x:1", "secret": "secret://user/registry"}}',
                     json.dumps({"script": "export HF_TOKEN=\"secret://user/hf-token\"\nrun\n"})):
            self.same(text)
        self.assertEqual(fb.scrub("HF_TOKEN=secret://user/hf-token and more"), "HF_TOKEN=[removed]")
        self.assertEqual(fb.scrub('{"secret": "secret://user/x y"}'), '{"secret": "[removed]"}')

    def test_shell_variables_are_kept(self):
        for text in ('curl -H "Authorization: Bearer ${FB_TOKEN}" "$FB_API_HOST/v1"',
                     "curl -H 'FB-Authorization: Bearer $T' http://x",
                     '      elif curl -fsSL -H "Cookie: fwi.access-token=${FB_TOKEN}" -o /wx/bin/x \\',
                     'export HF_TOKEN="$HF_TOKEN"',
                     "export HF_TOKEN=${HF_TOKEN} MODE=fast",
                     "HF_TOKEN=$HF_TOKEN",
                     "run --api-key='$API_KEY' --port 8000",
                     '["HF_TOKEN=${HF_TOKEN}"]',
                     '{"password": "$DB_PASSWORD"}',
                     json.dumps({"script": "export API_KEY=\"${LLM_KEY}\"\nrun --api-key=$API_KEY\n"})):
            self.same(text)
        # Kept only when the whole value is one variable, with its name in capital letters.
        self.assertEqual(fb.scrub("login user=admin password=$ecret1 host=db"),
                         "login user=admin password=[removed] host=db")
        self.assertEqual(fb.scrub("PASSWORD=$ecret1"), "PASSWORD=[removed]")
        self.assertEqual(fb.scrub('{"password": "$ecret1"}'), '{"password": "[removed]"}')
        self.assertEqual(fb.scrub("Authorization: Bearer $ecret1"), "Authorization: [removed]")
        self.assertEqual(fb.scrub('curl -H "X-Api-Key=$key" http://x'), 'curl -H "X-Api-Key=[removed]" http://x')
        self.assertEqual(fb.scrub("x A_KEY=$A/b y"), "x A_KEY=[removed] y")
        self.assertEqual(fb.scrub("x A_KEY=abc$B y"), "x A_KEY=[removed] y")
        self.assertEqual(fb.scrub("HF_TOKEN=$HF_TOKEN ./run"), "HF_TOKEN=[removed]")

    def test_git_setting_names_are_kept(self):
        self.same("      export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'")
        self.same("GIT_CONFIG_KEY_1=http.extraHeader")
        self.same('["GIT_CONFIG_KEY_12=http.extraHeader"]')
        self.assertEqual(fb.scrub("MY_GIT_CONFIG_KEY_0=x"), "MY_GIT_CONFIG_KEY_0=[removed]")
        self.assertEqual(fb.scrub("x GIT_CONFIG_KEY_0b=y"), "x GIT_CONFIG_KEY_0b=[removed]")

    def test_harmless_text_is_left_alone(self):
        for text in (VERSION_OUT, NODES_OUT, EVENTS_OUT, FUZZFILE,
                     "slot update_slots: id 0 | task 0 | n_tokens = 14\n",
                     "eval time = 3069.59 ms / 128 tokens (41.70 tokens per second)\n",
                     "Avg generation throughput: 31.2 tokens/s, Running: 1 reqs\n",
                     "KeyError: 'forecast'\n",
                     '{"usage": {"prompt_tokens": 14, "completion_tokens": 128}}\n',
                     "policy=adaptive backend=A latency_ms=210 units=4\n"):
            self.same(text)

    def test_the_filter_is_wide_on_purpose(self):
        self.assertEqual(fb.scrub("request 7 max_tokens=512 temperature=0.2"),
                         "request 7 max_tokens=[removed] temperature=0.2")
        self.assertEqual(fb.scrub("TOKENIZERS_PARALLELISM=false"), "TOKENIZERS_PARALLELISM=[removed]")
        self.assertEqual(fb.scrub('{"key": "nvidia.com/gpu", "value": "1"}'), '{"key": "[removed]", "value": "1"}')

    def test_twice_is_the_same_as_once(self):
        for text in (LOG_FORECAST, LOG_LLM, DESCRIBE_OUT, json.dumps(describe_tree(), indent=2),
                     json.dumps(describe_tree()), "Namespace(api_key=None, port=8000)", EVENTS_FAIL["err"]):
            once = fb.scrub(text)
            self.assertEqual(fb.scrub(once), once)
            self.assertEqual(len(once.splitlines()), len(text.splitlines()))       # lines are never added or lost
            for secret in SECRETS.values():
                self.assertNotIn(secret, once)

    def test_random_json_keeps_its_shape_and_loses_its_secrets(self):
        rng = random.Random(20261114)
        pieces = ['"', "\\", " ", "'", "=", "&", "\n", "\t", "\u00e9", "{", "}", ",", ":", "\\n", "/", "#", "$"]
        count = [0]

        def marked(prefix):
            """A text that starts with the prefix, with awkward characters mixed in."""
            count[0] += 1
            word = "%s%dx" % (prefix, count[0])
            return word + "".join(rng.choice(pieces) + word for _ in range(rng.randint(0, 3)))

        def tree(depth):
            node = {}
            for _ in range(rng.randint(1, 5)):
                kind, tag = rng.randint(0, 6), str(rng.randint(0, 9))
                if kind == 0:
                    node[rng.choice(["token", "api_key", "clientSecret", "PASSWORD"]) + tag] = marked("GONE")
                elif kind == 1:
                    node[rng.choice(["name", "image", "note", "status"]) + tag] = "STAY%dx" % rng.randint(0, 99)
                elif kind == 2:
                    node["env" + tag] = ["MODE=STAY1x", rng.choice(["HF_TOKEN", "db_password"]) + "=" + marked("GONE")]
                elif kind == 3:
                    plain = marked("GONE").replace('"', "").replace("\\", "").replace("\n", "")
                    node["script" + tag] = "#!/bin/sh\nexport MY_SECRET=\"%s GONE0x\"\necho STAY2x\n" % plain
                elif kind == 4 and depth < 3:
                    node["jobs" + tag] = tree(depth + 1)
                elif kind == 5 and depth < 2:
                    node["config" + tag] = json.dumps(tree(depth + 1), indent=rng.choice([None, 2]))
                else:
                    node["password" + tag] = rng.choice([5, 1.5, True, None, "secret://user/STAY3x"])
            return node

        def shape(node):
            """The tree with every text replaced by a dot; JSON held in a text is opened first."""
            if isinstance(node, dict):
                return dict((key, shape(value)) for key, value in node.items())
            if isinstance(node, list):
                return [shape(value) for value in node]
            if isinstance(node, str) and node.startswith("{"):
                return shape(json.loads(node))
            return "." if isinstance(node, str) else node

        def texts(node):
            if isinstance(node, dict):
                return "\n".join(texts(value) for value in node.values())
            if isinstance(node, list):
                return "\n".join(texts(value) for value in node)
            if isinstance(node, str) and node.startswith("{"):
                return texts(json.loads(node))
            return node if isinstance(node, str) else ""

        for _ in range(150):
            before = tree(0)
            for indent in (None, 2):
                for ascii_only in (True, False):
                    after = json.loads(fb.scrub(json.dumps(before, indent=indent, ensure_ascii=ascii_only)))
                    self.assertEqual(shape(after), shape(before))
                    self.assertNotIn("GONE", texts(after))
                    self.assertEqual(texts(after).count("STAY"), texts(before).count("STAY"))

    def test_long_input_is_fast(self):
        rng = random.Random(7)
        blob = base64.urlsafe_b64encode(rng.randbytes(300000)).decode("ascii")
        started = time.time()
        for text in (blob, json.dumps({"npy_b64": blob}), "key" * 100000, "eyJ" * 100000, "a=b key=v " * 30000,
                     '"' + "the token and the key " * 20000 + '"'):
            fb.scrub(text)
        self.assertLess(time.time() - started, 10.0)


class NamesTest(unittest.TestCase):
    """Finding job and service names in describe.json."""

    def test_names_at_any_depth(self):
        self.assertEqual(fb.find_job_names(describe_tree()), (["forecast", "llm"], []))
        deep = {"a": [{"b": {"Jobs": {"train": {}}, "c": {"services": {"llm.server": {}, "train": {}}}}}]}
        self.assertEqual(fb.find_job_names(deep), (["train", "llm.server"], []))

    def test_names_in_json_held_in_a_text(self):
        inner = json.dumps({"version": "v4", "services": {"stack-a": {"script": "x"}}, "jobs": {"bench": {}}})
        self.assertEqual(fb.find_job_names({"id": UUID, "specification": inner}), (["stack-a", "bench"], []))
        self.assertEqual(fb.find_job_names({"note": "{not json", "list": "[1, 2"}), ([], []))

    def test_a_list_of_named_things(self):
        tree = {"status": {"jobs": [{"name": "train", "state": "FINISHED"}, {"name": "eval"}, {"state": "x"}, "odd"]}}
        self.assertEqual(fb.find_job_names(tree), (["train", "eval"], []))

    def test_names_that_are_refused(self):
        tree = {"jobs": {"ok-1": {}, "Bad Name": {}, "-n": {}, "../up": {}, "a;b": {}, "": {}}}
        self.assertEqual(fb.find_job_names(tree), (["ok-1"], ["Bad Name", "-n", "../up", "a;b", ""]))

    def test_nothing_to_find(self):
        for tree in ({}, [], "text", 5, None, {"jobs": 3}, {"services": "llm"}, {"job": {"x": {}}}):
            self.assertEqual(fb.find_job_names(tree), ([], []))


class AllowedTest(unittest.TestCase):
    """The recorder runs six commands and no others."""

    def test_the_six_commands(self):
        for args in (["version"], ["node", "list"], ["workflow", "describe", UUID],
                     ["workflow", "describe", UUID, "-o", "json"], ["workflow", "events", UUID],
                     ["workflow", "log", UUID, "forecast", "-n", "200"], ["workflow", "describe", UUID.upper()]):
            self.assertTrue(fb.is_allowed(args), args)

    def test_everything_else(self):
        for args in ([], ["context", "token"], ["context", "login"], ["version", "--verbose"],
                     ["node", "list", "-o", "json"], ["node", "cordon", "gb10-a"], ["secret", "list"],
                     ["workflow", "stop", UUID], ["workflow", "delete", UUID],
                     ["workflow", "describe", "not-a-uuid"], ["workflow", "describe", UUID, "-o", "yaml"],
                     ["workflow", "describe", UUID + "\n"], ["workflow", "events", UUID, "forecast"],
                     ["workflow", "log", UUID, "forecast"], ["workflow", "log", UUID, "forecast", "-n", "0"],
                     ["workflow", "log", UUID, "forecast", "-n", "20; rm"],
                     ["workflow", "log", UUID, "--follow", "-n", "5"],
                     ["workflow", "log", UUID, "Forecast", "-n", "5"],
                     ["workflow", "log", UUID, "forecast", "-f", "5"],
                     ["workflow", "exec", UUID, "forecast", "--", "sh"]):
            self.assertFalse(fb.is_allowed(args), args)

    def test_a_refused_command_is_never_started(self):
        sandbox = Sandbox()
        self.addCleanup(sandbox.remove)
        with mock.patch.dict(os.environ, {"FUZZBALL_BIN": sandbox.fuzzball}):
            for args in (["context", "token"], ["workflow", "stop", UUID]):
                with self.assertRaises(ValueError):
                    fb.run_fuzzball(args)
            self.assertEqual(fb.run_fuzzball(["version"])["out"], VERSION_OUT)
        self.assertEqual(sandbox.calls(), [["version"]])


class RecordedRunTest(unittest.TestCase):
    """One run of the recorder as a program: a Fuzzfile, names from describe.json, one command that fails."""

    @classmethod
    def setUpClass(cls):
        cls.sandbox = Sandbox(plan(events=EVENTS_FAIL))
        cls.addClassCleanup(cls.sandbox.remove)
        cls.done = cls.sandbox.run(UUID, "--name", "node-stack-a", "--fuzzfile", "fuzzball/20-node-stack-a.fz",
                                   "--note", "One service runs both servers of node A.")
        folders = cls.sandbox.folders()
        cls.folder = folders[0] if folders else None
        cls.files = all_text(cls.folder) if cls.folder else {}

    def test_exit_status_and_folder_name(self):
        self.assertEqual(self.done.returncode, 1, self.done.stdout + self.done.stderr)     # one command failed
        self.assertEqual(len(self.sandbox.folders()), 1)
        name = os.path.basename(self.folder)
        self.assertRegex(name, r"^\d{8}-\d{6}-node-stack-a$")
        stamp = calendar.timegm(time.strptime(name[:15], "%Y%m%d-%H%M%S"))                # the name's time is UTC
        self.assertLess(abs(time.time() - stamp), 120)

    def test_folder_contents(self):
        self.assertEqual(sorted(self.files), ["20-node-stack-a.fz", "README.md", "describe.json", "describe.txt",
                                              "log-forecast.txt", "log-llm.txt", "nodes.txt", "version.txt"])
        self.assertEqual(self.files["version.txt"], VERSION_OUT)
        self.assertEqual(self.files["nodes.txt"], NODES_OUT)
        self.assertEqual(self.files["describe.txt"], DESCRIBE_CLEAN)
        self.assertEqual(self.files["log-forecast.txt"], LOG_FORECAST_CLEAN)
        self.assertEqual(self.files["log-llm.txt"], LOG_LLM_CLEAN)
        self.assertEqual(self.files["20-node-stack-a.fz"], FUZZFILE)                       # an exact copy
        saved = json.loads(self.files["describe.json"])
        config = json.loads(saved["specification"]["services"]["llm"]["files"]["config.json"])
        saved["specification"]["services"]["llm"]["files"]["config.json"] = config
        self.assertEqual(saved, describe_tree(clean=True))

    def test_secrets_are_gone_from_every_file(self):
        self.assertEqual(len(self.files), 8)
        for name, text in self.files.items():
            for what, secret in SECRETS.items():
                self.assertNotIn(secret, text, "%s still holds the secret from: %s" % (name, what))
        for text in (self.done.stdout, self.done.stderr):
            for what, secret in SECRETS.items():
                self.assertNotIn(secret, text, "the screen shows the secret from: %s" % what)
        for name in ("describe.txt", "describe.json", "log-forecast.txt", "log-llm.txt", "README.md"):
            self.assertIn(MARK, self.files[name], name)

    def test_readme(self):
        page = self.files["README.md"]
        self.assertTrue(page.startswith("# Fuzzball run record: node-stack-a\n"))
        self.assertIn("\n\nOne service runs both servers of node A.\n\n", page)
        self.assertIn("- Workflow id: `%s`" % UUID, page)
        when = time.strptime(os.path.basename(self.folder)[:15], "%Y%m%d-%H%M%S")
        self.assertIn("- Recorded: %s" % time.strftime("%Y-%m-%d %H:%M:%S UTC", when), page)
        self.assertIn("- Fuzzfile: `fuzzball/20-node-stack-a.fz`, copied into this folder as `20-node-stack-a.fz`\n",
                      page)
        self.assertIn("- Logs asked for: `forecast`, `llm` (found in describe.json)\n", page)
        self.assertIn("- Log lines asked for, per job: 200\n", page)
        for command, result, name in (
                ("version", "ok", "`version.txt`"),
                ("workflow describe %s" % UUID, "ok", "`describe.txt`"),
                ("workflow describe %s -o json" % UUID, "ok", "`describe.json`"),
                ("workflow events %s" % UUID, "FAILED: exit code 3", "none"),
                ("workflow log %s forecast -n 200" % UUID, "ok", "`log-forecast.txt`"),
                ("workflow log %s llm -n 200" % UUID, "ok", "`log-llm.txt`"),
                ("node list", "ok", "`nodes.txt`")):
            self.assertIn("| `fuzzball %s` | %s | %s |\n" % (command, result, name), page)
        self.assertEqual(page.count("\n| `fuzzball "), 7)
        self.assertIn("Values replaced in the saved files: `describe.txt`: 1, `describe.json`: 6, "
                      "`log-forecast.txt`: 3, `log-llm.txt`: 2.", page)
        self.assertNotIn(self.sandbox.root, page)                                          # no private paths

    def test_failed_command_is_recorded_and_the_others_still_ran(self):
        page = self.files["README.md"]
        self.assertIn("`fuzzball workflow events %s` failed: exit code 3. Its error text:\n" % UUID, page)
        self.assertIn("    Error: cannot read events: api_key=[removed] was refused\n", page)
        self.assertIn("    Error: the token [removed] is not valid here\n", page)
        self.assertNotIn("events.txt", self.files)                 # it printed nothing, so there is no file
        self.assertEqual(self.sandbox.calls()[4:], [["workflow", "log", UUID, "forecast", "-n", "200"],
                                                    ["workflow", "log", UUID, "llm", "-n", "200"], ["node", "list"]])
        self.assertIn("FAILED: exit code 3", self.done.stdout)
        self.assertIn("1 of 7 commands failed", self.done.stdout)

    def test_only_the_listed_commands_were_run(self):
        self.assertEqual(self.sandbox.calls(), [
            ["version"],
            ["workflow", "describe", UUID],
            ["workflow", "describe", UUID, "-o", "json"],
            ["workflow", "events", UUID],
            ["workflow", "log", UUID, "forecast", "-n", "200"],
            ["workflow", "log", UUID, "llm", "-n", "200"],
            ["node", "list"]])
        for call in self.sandbox.calls():
            self.assertTrue(fb.is_allowed(call), call)

    def test_last_lines_say_where_and_remind(self):
        lines = self.done.stdout.strip().splitlines()
        place = os.path.join("evidence", os.path.basename(self.folder))
        self.assertEqual(lines[-3], "Saved 8 files in: " + place)
        self.assertTrue(lines[-2].startswith("Read every file in that folder before you commit it."), lines[-2])
        self.assertEqual(lines[-1], 'The lines to look at first:  grep -n -i -E "%s" %s/*' % (fb.LOOK_FOR, place))
        if shutil.which("grep"):                                           # the printed command works as it is
            found = subprocess.run(lines[-1].split(":  ", 1)[1], shell=True, cwd=self.sandbox.repo,
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertIn("log-forecast.txt:3:FB_TOKEN=[removed]", found.stdout)
        self.assertNotIn("The copy of the Fuzzfile is not exact", self.done.stdout)
        self.assertEqual(self.done.stderr, "")


class RecordTest(unittest.TestCase):
    """Other runs of the recorder as a program."""

    def setUp(self):
        self.sandbox = Sandbox()
        self.addCleanup(self.sandbox.remove)

    def test_all_commands_work(self):
        out = os.path.join(self.sandbox.root, "elsewhere", "records")
        self.sandbox.answer(plan(nodes={"out": NODES_OUT, "err": "warning: the login ends in 5 minutes\n"}))
        done = self.sandbox.run(UUID, "--name", "all-good", "--out", out)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("  fuzzball node list ... ok\n      warning: the login ends in 5 minutes\n", done.stdout)
        self.assertEqual(self.sandbox.folders(), [])                                       # nothing under evidence/
        folder, = self.sandbox.folders(out)
        files = all_text(folder)
        self.assertEqual(sorted(files), ["README.md", "describe.json", "describe.txt", "events.txt",
                                         "log-forecast.txt", "log-llm.txt", "nodes.txt", "version.txt"])
        self.assertEqual(files["events.txt"], EVENTS_OUT)
        self.assertIn("None. All 7 commands worked.", files["README.md"])
        self.assertIn("No note was given for this run.", files["README.md"])
        self.assertIn("- Fuzzfile: none was given", files["README.md"])
        self.assertIn("Saved 8 files in: " + folder, done.stdout)                          # outside the working folder
        self.assertNotIn("failed", done.stdout)

    def test_jobs_given_on_the_command_line(self):
        self.sandbox.answer(plan(**{"log train": {"out": "epoch 1\n"}, "log eval.1": {"out": ""}}))
        # The workflow id may also come last, after the job names. --out may be a path from the working folder.
        done = self.sandbox.run("--name", "picked", "--lines", "35", "--out", "records", "--job", "train", "eval.1",
                                "--job", "llm", "train", UUID)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        logs = [call for call in self.sandbox.calls() if call[:2] == ["workflow", "log"]]
        self.assertEqual(logs, [["workflow", "log", UUID, name, "-n", "35"] for name in ("train", "eval.1", "llm")])
        folder, = self.sandbox.folders(os.path.join(self.sandbox.repo, "records"))
        files = all_text(folder)
        self.assertEqual(sorted(name for name in files if name.startswith("log-")),
                         ["log-eval.1.txt", "log-llm.txt", "log-train.txt"])
        self.assertEqual(files["log-train.txt"], "epoch 1\n")
        self.assertEqual(files["log-eval.1.txt"], "")
        self.assertIn("- Logs asked for: `train`, `eval.1`, `llm` (given with --job)", files["README.md"])
        self.assertIn("- Log lines asked for, per job: 35", files["README.md"])
        self.assertIn("eval.1 -n 35` | ok, but it printed nothing | `log-eval.1.txt` |", files["README.md"])
        self.assertIn("Saved 9 files in: " + os.path.join("records", os.path.basename(folder)), done.stdout)

    def test_names_cannot_be_read_from_describe_json(self):
        self.sandbox.answer(plan(**{"describe-json": {"out": "warning: this is not JSON\n"}}))
        done = self.sandbox.run(UUID, "--name", "no-names")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(len(self.sandbox.calls()), 5)                                     # no log was asked for
        files = all_text(self.sandbox.folders()[0])
        self.assertEqual(files["describe.json"], "warning: this is not JSON\n")
        self.assertIn("- Logs asked for: none (describe.json could not be read as JSON)", files["README.md"])
        self.assertIn("No logs were saved: describe.json could not be read as JSON. Run again with --job NAME",
                      done.stdout)

    def test_describe_json_with_names_that_are_refused(self):
        tree = {"id": UUID, "jobs": {"Bad Name": {}, "-n": {}}}
        self.sandbox.answer(plan(**{"describe-json": {"out": json.dumps(tree)}}))
        done = self.sandbox.run(UUID, "--name", "odd-names")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(len(self.sandbox.calls()), 5)
        page = all_text(self.sandbox.folders()[0])["README.md"]
        self.assertIn('- Logs asked for: none (describe.json has no usable names under "jobs" or "services"). '
                      'Names found but not used, because this tool does not accept them: `"Bad Name"`, `"-n"`', page)

    def test_a_failed_command_keeps_what_it_printed(self):
        long_error = "".join("error line %d\n" % number for number in range(1, 41))
        self.sandbox.answer(plan(**{
            "log llm": {"out": "first lines\nFB_TOKEN=abc.def\n", "err": "stream broke\n", "code": 1},
            "describe": {"err": long_error, "code": 4},
            "nodes": {"code": 2}}))
        done = self.sandbox.run(UUID, "--name", "partial")
        self.assertEqual(done.returncode, 1)
        files = all_text(self.sandbox.folders()[0])
        self.assertEqual(files["log-llm.txt"], "first lines\nFB_TOKEN=[removed]\n")
        self.assertNotIn("nodes.txt", files)
        self.assertNotIn("describe.txt", files)
        page = files["README.md"]
        self.assertIn("llm -n 200` | FAILED: exit code 1 | `log-llm.txt` |", page)
        self.assertIn("llm -n 200` failed: exit code 1. Its error text:\n\n    stream broke\n\n"
                      "What it printed before it failed is in `log-llm.txt`.", page)
        self.assertIn("| `fuzzball node list` | FAILED: exit code 2 | none |", page)
        self.assertIn("`fuzzball node list` failed: exit code 2. It printed no error text.", page)
        self.assertIn("failed: exit code 4. The last 30 lines of its error text:\n\n    error line 11\n", page)
        self.assertIn("    error line 40\n", page)
        self.assertNotIn("error line 10\n", page)
        self.assertIn("3 of 7 commands failed", done.stdout)

    def test_every_command_fails(self):
        refused = {"code": 1, "err": "Error: not logged in\n"}
        self.sandbox.answer(dict((key, refused) for key in plan()))
        done = self.sandbox.run(UUID, "--name", "logged-out")
        self.assertEqual(done.returncode, 1)
        self.assertEqual(len(self.sandbox.calls()), 5)
        self.assertEqual(sorted(all_text(self.sandbox.folders()[0])), ["README.md"])
        self.assertIn("Every command failed. Check that fuzzball is logged in", done.stdout)
        self.assertIn("      Error: not logged in", done.stdout)

    def test_fuzzfile_with_a_secret(self):
        write(self.sandbox.fuzzfile, FUZZFILE + "      export LLM_API_KEY=fake-fuzzfile-key-99\n")
        done = self.sandbox.run(UUID, "--name", "leaky", "--fuzzfile", "fuzzball/20-node-stack-a.fz")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = all_text(self.sandbox.folders()[0])
        self.assertEqual(files["20-node-stack-a.fz"], FUZZFILE + "      export LLM_API_KEY=[removed]\n")
        self.assertIn("as `20-node-stack-a.fz`. The copy is not exact. Lines changed by the filter: 14\n",
                      files["README.md"])
        self.assertIn("`20-node-stack-a.fz`: 1.", files["README.md"])
        self.assertIn("The copy of the Fuzzfile is not exact. The filter changed these lines of "
                      "fuzzball/20-node-stack-a.fz:\n    line 14: export LLM_API_KEY=[removed]\n"
                      "If one of them holds a real secret, take it out of fuzzball/20-node-stack-a.fz", done.stdout)
        for text in list(files.values()) + [done.stdout, done.stderr]:
            self.assertNotIn("fake-fuzzfile-key-99", text)
        self.assertIn("fake-fuzzfile-key-99", read(self.sandbox.fuzzfile))                 # the original is not touched

    def test_note_with_a_secret(self):
        done = self.sandbox.run(UUID, "--name", "noted", "--job", "llm",
                                "--note", "  Run with API_KEY=fake-note-key-12 and Bearer fake-note-bearer-13.  ")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        page = all_text(self.sandbox.folders()[0])["README.md"]
        self.assertIn("\n\nRun with API_KEY=[removed] and [removed]\n\n", page)
        self.assertNotIn("fake-note", page + done.stdout + done.stderr)

    def test_fuzzfile_named_like_an_output_file(self):
        write(os.path.join(self.sandbox.repo, "fuzzball", "nodes.txt"), FUZZFILE)
        done = self.sandbox.run(UUID, "--name", "clash", "--fuzzfile", os.path.join("fuzzball", "nodes.txt"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        files = all_text(self.sandbox.folders()[0])
        self.assertEqual(files["nodes.txt"], NODES_OUT)
        self.assertEqual(files["fuzzfile-nodes.txt"], FUZZFILE)

    def test_bad_arguments_stop_before_anything_runs(self):
        outside = os.path.join(self.sandbox.root, "outside.fz")
        write(outside, FUZZFILE)
        os.symlink(outside, os.path.join(self.sandbox.repo, "fuzzball", "link.fz"))
        a_file = os.path.join(self.sandbox.root, "a-file")
        write(a_file, "this is a file, not a folder\n")
        for args, word in (
                (["not-a-uuid", "--name", "x"], "WORKFLOW_UUID"),
                (["--name", "x"], "WORKFLOW_UUID"),
                ([UUID, "--name", "Node Stack"], "--name"),
                ([UUID], "--name"),
                ([UUID, "--name", "x", "--job", "ok", "Not_ok"], "--job"),
                ([UUID, "--name", "x", "--lines", "0"], "--lines"),
                ([UUID, "--name", "x", "--fuzzfile", "fuzzball/missing.fz"], "--fuzzfile"),
                ([UUID, "--name", "x", "--fuzzfile", outside], "--fuzzfile"),
                ([UUID, "--name", "x", "--fuzzfile", "fuzzball/../../outside.fz"], "--fuzzfile"),
                ([UUID, "--name", "x", "--fuzzfile", "fuzzball/link.fz"], "--fuzzfile"),
                ([UUID, "--name", "x", "--fuzzfile", "fuzzball"], "--fuzzfile"),
                ([UUID, "--name", "x", "--out", a_file], "--out")):
            done = self.sandbox.run(*args)
            self.assertEqual(done.returncode, 2, args)
            self.assertIn("fb_record.py: error:", done.stderr, args)
            self.assertIn(word, done.stderr, args)
        self.assertEqual(self.sandbox.calls(), [])
        self.assertEqual(self.sandbox.folders(), [])

    def test_no_fuzzball_tool(self):
        missing = os.path.join(self.sandbox.root, "no-such-fuzzball")
        done = self.sandbox.run(UUID, "--name", "x", FUZZBALL_BIN=missing)
        self.assertEqual(done.returncode, 2)
        self.assertIn("cannot find the fuzzball command-line tool", done.stderr)
        self.assertIn("FUZZBALL_BIN", done.stderr)
        self.assertEqual(self.sandbox.folders(), [])
        not_a_program = os.path.join(self.sandbox.root, "plain-file")
        write(not_a_program, "not a program\n")
        self.assertEqual(self.sandbox.run(UUID, "--name", "x", FUZZBALL_BIN=not_a_program).returncode, 2)
        self.assertEqual(self.sandbox.folders(), [])


class InProcessTest(unittest.TestCase):
    """Cases that need the recorder's clock or its time limit changed: main() is called in this process."""

    def setUp(self):
        self.sandbox = Sandbox()
        self.addCleanup(self.sandbox.remove)
        self.out = os.path.join(self.sandbox.root, "out")
        self.addCleanup(setattr, fb, "TIMEOUT_S", fb.TIMEOUT_S)

    def test_folder_name_is_the_utc_time_and_is_never_reused(self):
        now = calendar.timegm((2026, 11, 14, 15, 4, 5, 0, 0, 0))
        args = [UUID, "--name", "demo.run_1", "--out", self.out, "--job", "llm"]
        status, printed, errors = run_main(args, self.sandbox.fuzzball, now)
        self.assertEqual((status, errors), (0, ""), printed)
        self.assertEqual(os.listdir(self.out), ["20261114-150405-demo.run_1"])
        folder = os.path.join(self.out, "20261114-150405-demo.run_1")
        before = all_text(folder)
        self.assertIn("- Recorded: 2026-11-14 15:04:05 UTC", before["README.md"])
        calls = len(self.sandbox.calls())
        status, printed, errors = run_main(args, self.sandbox.fuzzball, now)       # the same second, the same name
        self.assertEqual(status, 2)
        self.assertIn("is already there", errors)
        self.assertEqual(len(self.sandbox.calls()), calls)                         # nothing was run
        self.assertEqual(all_text(folder), before)

    def test_a_command_that_hangs_is_stopped_and_recorded(self):
        fb.TIMEOUT_S = 3                # every command of this run gets 3 seconds; the fake takes far less
        self.sandbox.answer(plan(events={"out": "first event\nFB_TOKEN=abc.def\n", "sleep": 15}))
        started = time.time()
        status, printed, errors = run_main([UUID, "--name", "slow", "--out", self.out, "--job", "llm"],
                                           self.sandbox.fuzzball)
        self.assertLess(time.time() - started, 12)                                 # it did not wait 15 seconds
        self.assertEqual(status, 1, printed + errors)
        folder, = [os.path.join(self.out, name) for name in os.listdir(self.out)]
        files = all_text(folder)
        self.assertEqual(files["events.txt"], "first event\nFB_TOKEN=[removed]\n")         # what came before the stop
        self.assertIn("| FAILED: no answer within 3 seconds, so it was stopped | `events.txt` |", files["README.md"])
        self.assertEqual(files["nodes.txt"], NODES_OUT)                            # the later commands still ran
        self.assertIn("1 of 6 commands failed", printed)

    def test_bad_arguments(self):
        outside = os.path.join(self.sandbox.root, "outside.fz")
        write(outside, FUZZFILE)
        tail = ["--name", "x", "--out", self.out]
        good = [UUID] + tail
        for args in (["1b4e28ba2fa111d2883f0016d3cca427"] + tail,
                     [UUID[:-1]] + tail,
                     [UUID + "\n"] + tail,
                     [UUID + "0"] + tail,
                     ["../" + UUID] + tail,
                     tail,
                     [UUID, UUID] + tail,
                     [UUID, "--name", "", "--out", self.out],
                     [UUID, "--name", "Caps", "--out", self.out],
                     [UUID, "--name", "a/b", "--out", self.out],
                     [UUID, "--name", "..", "--out", self.out],
                     [UUID, "--name", ".hidden", "--out", self.out],
                     [UUID, "--name", "two words", "--out", self.out],
                     [UUID, "--name", "x\n", "--out", self.out],
                     good + ["--job", "Caps"],
                     good + ["--job", "a/b"],
                     good + ["--job", "ok", "--job", "a;b"],
                     good + ["--job", "-n"],
                     good + ["--job"],
                     tail + ["--job", UUID, UUID],
                     good + ["--lines", "0"],
                     good + ["--lines", "-5"],
                     good + ["--lines", "many"],
                     good + ["--fuzzfile", outside],
                     good + ["--fuzzfile", os.path.join(REPO, "no", "such", "file.fz")],
                     good + ["--fuzzfile", os.path.join(REPO, "tools")],
                     good + ["--no-such-option"]):
            status, printed, errors = run_main(args, self.sandbox.fuzzball)
            self.assertEqual(status, 2, args)
            self.assertIn("error:", errors, args)
        self.assertEqual(self.sandbox.calls(), [])
        self.assertFalse(os.path.exists(self.out))

    def test_the_workflow_id_may_come_first_last_or_in_between(self):
        for args in ([UUID, "--name", "x", "--job", "a", "b", "--job", "c"],
                     ["--name", "x", "--job", "a", "b", "--job", "c", UUID],
                     ["--job", "a", "b", UUID, "--name", "x", "--job", "c"],
                     ["--name", "x", UUID, "--job", "a", "--job", "b", "c", "a"],
                     ["--job", "a", "b", "c", "--lines", "5", UUID, "--name", "x"]):
            options = fb.parse_args(args)
            self.assertEqual((options.workflow, options.name, options.jobs), (UUID, "x", ["a", "b", "c"]), args)
        options = fb.parse_args(["--name", "x", UUID])
        self.assertEqual((options.workflow, options.jobs, options.lines, options.note), (UUID, [], 200, ""))
        self.assertEqual(options.out, os.path.join(REPO, "evidence"))

    def test_help(self):
        status, printed, errors = run_main(["--help"], self.sandbox.fuzzball)
        self.assertEqual((status, errors), (0, ""))
        self.assertTrue(printed.startswith("usage: fb_record.py WORKFLOW_UUID --name SHORT-NAME"), printed[:80])
        self.assertIn("Written at the hackathon (Nov 13-14, 2026).", printed)
        self.assertIn("--fuzzfile fuzzball/20-node-stack-a.fz --note", printed)
        self.assertEqual(self.sandbox.calls(), [])


if __name__ == "__main__":
    unittest.main()
