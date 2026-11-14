#!/usr/bin/env python3
r"""Fuzzball run recorder. Saves what Fuzzball says about one workflow run into the repo.

It asks CIQ's "fuzzball" command-line tool (version 4.4) about one workflow and writes the
answers into a new folder under evidence/. A reader can then see that the system ran on the
event's machines under Fuzzball. It needs only the standard library (Python 3.9 to 3.13).

Run it from the repo root, on a machine where "fuzzball" is logged in to the cluster:

    python tools/fb_record.py WORKFLOW_UUID --name SHORT-NAME [--fuzzfile fuzzball/20-node-stack-a.fz]
        [--job NAME ...] [--lines 200] [--note "one sentence on what this run shows"] [--out evidence]

    python tools/fb_record.py 1b4e28ba-2fa1-11d2-883f-0016d3cca427 --name node-stack-a \
        --fuzzfile fuzzball/20-node-stack-a.fz --note "One service runs both servers of node A."

    python tools/fb_record.py 1b4e28ba-2fa1-11d2-883f-0016d3cca427 --name train-beside \
        --fuzzfile fuzzball/40-train-beside.fz --job train-beside --lines 400

Written at the hackathon (Nov 13-14, 2026).

It writes a new folder, evidence/<UTC time as YYYYMMDD-HHMMSS>-<SHORT-NAME>/, holding:

    README.md       the name, the note, the workflow id, the time, each command and how it went
    version.txt     fuzzball version
    describe.txt    fuzzball workflow describe <UUID>
    describe.json   fuzzball workflow describe <UUID> -o json
    events.txt      fuzzball workflow events <UUID>
    log-<JOB>.txt   fuzzball workflow log <UUID> <JOB> -n <LINES>   (one file per job or service)
    nodes.txt       fuzzball node list
    the Fuzzfile    a copy, if --fuzzfile was given

Things to know:

- Those six commands are the only ones it runs. Each gets 60 seconds. They only read.
- The tool is found as "fuzzball" on the PATH. Set FUZZBALL_BIN to use another path.
- Without --job, the names come from describe.json: every name under a "jobs" or "services"
  key, at any depth. README.md says which names were found.
- A command that fails does not stop the others. README.md records its exit code and its
  error text. A file is written for it only if it printed something. The error text of a
  command that worked (a warning, say) is shown on the screen and not saved.
- The WORKFLOW_UUID may come first or last on the command line.
- Secrets are taken out before anything is written, because the repo is public. Each of these
  is replaced with [removed]:
    * a JSON Web Token (three parts joined by dots, starting with "eyJ");
    * the word Bearer and the token after it;
    * the value of a name that contains TOKEN, SECRET, PASSWORD, KEY or CREDENTIAL, written
      as NAME=value or as "name": "value". Small and capital letters both count.
  Kept, because they name a secret and do not hold one: a Fuzzball secret reference such as
  secret://user/hf-token, and a shell variable in capital letters such as $HF_TOKEN or
  ${HF_TOKEN}. Kept too: GIT_CONFIG_KEY_0, GIT_CONFIG_KEY_1, ... which name a git setting.
- The copy of the Fuzzfile goes through the same filter. If that changes it, the tool shows
  the lines, so you can see whether the Fuzzfile holds a real secret.
- The filter is wide on purpose. It also takes out harmless values such as max_tokens=512.
- The filter does not know every way to write a secret. It does not catch "password: abc"
  (a colon and no quotes), "password = abc" (spaces around the sign), "--api-key abc" (a
  space) or "Authorization: Basic abc". Read every file before you commit. At the end the
  tool prints the folder's path and a grep command that lists the lines to look at first.
- Exit status: 0 if every command worked; 1 if the folder was written but a command failed;
  2 if nothing was written (a bad argument, or no fuzzball tool).
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the folder above tools/
TIMEOUT_S = 60          # each fuzzball command gets this long
ERROR_LINES = 30        # README.md shows at most this many lines of a failed command's error text
MARK = "[removed]"
LOOK_FOR = "token|secret|password|passwd|key|credential|authoriz|bearer|cookie"    # for the reader's own search

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")      # --name and job names; always used with fullmatch
LINES_RE = re.compile(r"[1-9][0-9]*")
JOB_KEYS = ("jobs", "services")                    # keys in describe.json whose value holds names


# ---------------------------------------------------------------------------------------------
# Taking secrets out of text
# ---------------------------------------------------------------------------------------------

_WORDS = r"(?:TOKEN|SECRET|PASSWORD|KEY|CREDENTIAL)"
_NAME_CHAR = r"[A-Za-z0-9_.-]"
# A name with one of the words in it: FB_TOKEN, api-key, aws.secretAccessKey ... The look-ahead
# finds the word first, so a long run of letters is read once and not once per letter.
_NAME = "(?P<name>(?=" + _NAME_CHAR + "*?" + _WORDS + ")" + _NAME_CHAR + "+)"
_DQ = r'"(?:[^"\\\n]|\\.)*"'        # "text" on one line; a \" inside does not end it
_SQ = r"'(?:[^'\\\n]|\\.)*'"        # 'text' on one line
_BSQ = r'\\"(?:(?!\\")[^\n])*\\"'   # \"text\": a quoted value inside a string that could not be decoded
# A value with no quotes. It ends at a space or at a quote. It also ends at \n, \r, \t and \"
# as a JSON string writes them, so it never runs past the end of the string it is in.
_BARE = r"""(?!=)(?:[^\s"'\\]|\\+[^\s"'nrt\\])+"""

_STRING = re.compile(_DQ)
# A JSON Web Token: header.payload.signature, each part in base64url, the header starting "eyJ".
# The signature may be empty, and an encrypted token has more parts: those go too.
_JWT = re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*")
_BEARER = re.compile(r"""\bBearer[ \t]+(?P<value>[^\s"'\\]+)""", re.IGNORECASE)
# "name": "value" as in JSON, or 'name': 'value' as Python prints a dict.
_PAIR = re.compile(
    r"(?P<head>(?:\"(?=[^\"\\\n]*?" + _WORDS + r")[^\"\\\n]*\"|'(?=[^'\\\n]*?" + _WORDS + r")[^'\\\n]*')\s*:\s*)"
    r"(?P<value>" + _DQ + "|" + _SQ + ")", re.IGNORECASE)
# A line that is NAME=value, as "env" prints it: the value is the rest of the line, spaces and all.
_ENV_LINE = re.compile(r"^(?P<head>[ \t]*(?:-[ \t]+)?" + _NAME + r"=)(?P<value>[^\r\n]+)",
                       re.IGNORECASE | re.MULTILINE)
# A quoted string that is NAME=value from its first character, as in a JSON list of environment
# settings: the value runs to the closing quote, spaces and all. One pattern per kind of quote.
_QUOTED_ASSIGN = (
    re.compile(r'(?P<head>(?<!\\)"' + _NAME + r'=)(?P<value>(?:[^"\\\n]|\\.)+)(?P<tail>")', re.IGNORECASE),
    re.compile(r"(?P<head>(?<!\\)'" + _NAME + r"=)(?P<value>(?:[^'\\\n]|\\.)+)(?P<tail>')", re.IGNORECASE),
)
# NAME=value anywhere else: in a command line, in a script, in a URL. A quoted value must not
# run straight into a word: then the quote after "=" was the end of a string ("...?key=", "next"),
# not the start of a value.
_ASSIGN = re.compile(
    r"(?P<head>(?<!" + _NAME_CHAR + ")" + _NAME + "=)"
    r"(?P<value>(?:" + _DQ + "|" + _SQ + "|" + _BSQ + ")(?![A-Za-z0-9_])|" + _BARE + ")", re.IGNORECASE)

# Three things are kept, though they sit where a secret could be:
#   secret://user/hf-token    a Fuzzball secret reference: the name of a stored secret, not its value
#   $HF_TOKEN, ${HF_TOKEN}    a shell variable: the script names it, the value is not there. Only names
#                             in capital letters count, so a password such as $ecret1 is still taken out.
#   GIT_CONFIG_KEY_0=...      git's way to name one of its settings in the environment; the name is no secret
_REFERENCE = re.compile(
    r"""(\\?["']|)(?:secret://[A-Za-z0-9._/-]+|\$[A-Z_][A-Z0-9_]*|\$\{[A-Z_][A-Z0-9_]*\})\1""")
_PLAIN_NAME = re.compile(r"GIT_CONFIG_KEY_[0-9]+")


def _is_kept(match):
    """True if what a NAME=value pattern found is one of the three things above."""
    return (_PLAIN_NAME.fullmatch(match.group("name")) is not None
            or _REFERENCE.fullmatch(match.group("value").strip()) is not None)


def _fix_inside(match):
    """One "..." string that holds escapes: decode it, clean the text inside, encode it again.

    This is how text inside a JSON string gets cleaned: a script, or JSON within JSON. There
    the quotes are written as \\" and the line ends as \\n, which the patterns would misread.
    """
    literal = match.group(0)
    if "\\" not in literal:
        return literal                  # nothing is escaped: the patterns see this text as it is
    try:
        inside = json.loads(literal, strict=False)
    except ValueError:
        return literal                  # not a JSON string after all
    cleaned = scrub(inside)
    if cleaned == inside:
        return literal
    return json.dumps(cleaned, ensure_ascii="\\u" in literal)


def _fix_pair(match):
    value = match.group("value")
    if _REFERENCE.fullmatch(value):
        return match.group(0)
    return match.group("head") + value[0] + MARK + value[0]         # keep the kind of quote that was there


def _fix_bearer(match):
    return match.group(0) if _REFERENCE.fullmatch(match.group("value")) else MARK


def _fix_env_line(match):
    return match.group(0) if _is_kept(match) else match.group("head") + MARK


def _fix_quoted_assign(match):
    return match.group(0) if _is_kept(match) else match.group("head") + MARK + match.group("tail")


def _fix_assign(match):
    if _is_kept(match) or match.group("value").startswith(MARK):    # the second: already taken out
        return match.group(0)
    return match.group("head") + MARK


def scrub(text):
    """Return the text with every secret this tool can recognise replaced by [removed].

    The order matters. Text inside JSON strings goes first. "Bearer abc" goes before NAME=value,
    or AUTH_KEY=Bearer abc would lose only the word "Bearer".
    """
    text = _STRING.sub(_fix_inside, text)
    text = _PAIR.sub(_fix_pair, text)
    text = _BEARER.sub(_fix_bearer, text)
    text = _ENV_LINE.sub(_fix_env_line, text)
    for pattern in _QUOTED_ASSIGN:
        text = pattern.sub(_fix_quoted_assign, text)
    text = _ASSIGN.sub(_fix_assign, text)
    text = _JWT.sub(MARK, text)
    return text


def count_removed(before, after):
    """How many secrets scrub() took out, counted as new [removed] marks."""
    return max(0, after.count(MARK) - before.count(MARK))


# ---------------------------------------------------------------------------------------------
# Running fuzzball
# ---------------------------------------------------------------------------------------------

def fuzzball_path():
    """The fuzzball tool to run: $FUZZBALL_BIN, or "fuzzball" from the PATH."""
    return os.environ.get("FUZZBALL_BIN") or "fuzzball"


def is_allowed(args):
    """True only for the six read-only commands listed at the top of this file."""
    args = list(args)
    if args in (["version"], ["node", "list"]):
        return True
    if len(args) < 3 or args[0] != "workflow" or not UUID_RE.fullmatch(args[2]):
        return False
    if args[1] == "describe":
        return args[3:] in ([], ["-o", "json"])
    if args[1] == "events":
        return args[3:] == []
    if args[1] == "log":
        return (len(args) == 6 and NAME_RE.fullmatch(args[3]) is not None and args[4] == "-n"
                and LINES_RE.fullmatch(args[5]) is not None)
    return False


def run_fuzzball(args, timeout=None):
    """Run one allowed fuzzball command. A command that fails is reported, never raised.

    Returns a dict:
      args     the arguments after "fuzzball"
      out      what it printed (text; secrets are still in it)
      err      what it printed as error text (secrets are still in it)
      code     its exit code, or None if it did not finish
      problem  None if it worked, otherwise a few words saying what went wrong
    """
    args = list(args)
    if not is_allowed(args):
        raise ValueError("fb_record.py does not run this command: fuzzball " + " ".join(args))
    timeout = TIMEOUT_S if timeout is None else timeout
    out, err, code = b"", b"", None
    try:
        done = subprocess.run([fuzzball_path()] + args, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out, err, code = done.stdout, done.stderr, done.returncode
        if code == 0:
            problem = None
        elif code < 0:
            problem = "stopped by signal %d" % -code
        else:
            problem = "exit code %d" % code
    except subprocess.TimeoutExpired as late:           # subprocess.run has already stopped it
        out, err = late.stdout or b"", late.stderr or b""
        problem = "no answer within %g seconds, so it was stopped" % timeout
    except OSError as error:
        problem = "it could not be started (%s)" % (error.strerror or "unknown error")
    return {"args": args, "code": code, "problem": problem,
            "out": out.decode("utf-8", "replace"), "err": err.decode("utf-8", "replace")}


def find_job_names(tree):
    """Find job and service names in a parsed describe.json.

    At any depth, a key called "jobs" or "services" counts if its value is a mapping
    {name: ...} or a list of {"name": name, ...}. A text that is itself JSON is opened too.
    Returns two lists, each in the order found and without repeats: the names this tool
    accepts, and the names it refuses (they could not be used safely in a command or a file name).
    """
    good, bad = [], []

    def note(name):
        if not isinstance(name, str):
            return
        target = good if NAME_RE.fullmatch(name) else bad
        if name not in target:
            target.append(name)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and key.lower() in JOB_KEYS:
                    if isinstance(value, dict):
                        for name in value:
                            note(name)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                note(item.get("name"))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node[:1] in ("{", "["):
            try:
                walk(json.loads(node))
            except ValueError:
                pass

    walk(tree)
    return good, bad


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------

def fuzzfile_problem(path):
    """None if path is an existing file inside this repo, otherwise a sentence saying what is wrong."""
    if not os.path.isfile(path):
        return "--fuzzfile: there is no such file: %s" % path
    root = os.path.realpath(REPO)
    if not os.path.realpath(path).startswith(root + os.sep):
        return "--fuzzfile must be a file inside this repo (%s); %s is not" % (root, path)
    return None


def parse_args(argv):
    """Read and check the command line. A bad argument ends the program with exit status 2."""
    parser = argparse.ArgumentParser(
        prog="fb_record.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="fb_record.py WORKFLOW_UUID --name SHORT-NAME [--fuzzfile FILE] [--job NAME ...] [--lines N] "
              "[--note TEXT] [--out DIR]")
    parser.add_argument("workflow", metavar="WORKFLOW_UUID", nargs="?", help="the id of the workflow to record")
    parser.add_argument("--name", required=True, metavar="SHORT-NAME",
                        help="a short name for this run; the folder's name ends with it "
                             "(small letters, digits, dots, underscores, dashes)")
    parser.add_argument("--fuzzfile", metavar="FILE",
                        help="the Fuzzfile that started the workflow; a copy goes into the folder "
                             "(it must be inside this repo)")
    parser.add_argument("--job", action="append", nargs="+", metavar="NAME",
                        help="a job or service whose log to save; give several names or repeat "
                             "the option (default: every name found in describe.json)")
    parser.add_argument("--lines", type=int, default=200, metavar="N",
                        help="how many log lines to ask for, per job (default: 200)")
    parser.add_argument("--note", default="", metavar="TEXT",
                        help="one sentence on what this run shows; it goes into README.md")
    parser.add_argument("--out", metavar="DIR",
                        help="where to make the new folder (default: evidence/ in this repo)")
    options = parser.parse_args(argv)

    names = [name for group in options.job or [] for name in group]
    if options.workflow is None:
        # "--job train llm WORKFLOW_UUID": the id came last, so it was read as one more job name.
        ids = [name for name in names if UUID_RE.fullmatch(name)]
        if len(ids) != 1:
            parser.error("WORKFLOW_UUID is missing")
        options.workflow = ids[0]
        names.remove(ids[0])
    rule = ("must start with a small letter or a digit, and hold only small letters, digits, dots, "
            "underscores and dashes")
    if not UUID_RE.fullmatch(options.workflow):
        parser.error("WORKFLOW_UUID must look like 1b4e28ba-2fa1-11d2-883f-0016d3cca427; got %r" % options.workflow)
    if not NAME_RE.fullmatch(options.name):
        parser.error("--name %s; got %r" % (rule, options.name))
    options.jobs = []
    for name in names:
        if not NAME_RE.fullmatch(name):
            parser.error("--job: a job name %s; got %r" % (rule, name))
        if name not in options.jobs:
            options.jobs.append(name)
    if options.lines < 1:
        parser.error("--lines must be 1 or more; got %d" % options.lines)
    if options.fuzzfile is not None:
        problem = fuzzfile_problem(options.fuzzfile)
        if problem:
            parser.error(problem)
    if options.out is None:
        options.out = os.path.join(REPO, "evidence")
    options.note = options.note.strip()
    return options


def stop(message):
    """Say why nothing was written. Returns the exit status for that."""
    print("fb_record.py: error: " + message, file=sys.stderr)
    return 2


def shown(args):
    """A command as the screen and README.md show it. FUZZBALL_BIN may hold a private path: it is never shown."""
    return "fuzzball " + " ".join(args)


def error_lines(text):
    """The last lines of a command's error text."""
    return text.strip().splitlines()[-ERROR_LINES:]


def run_step(args, file_name):
    """Run one command, say on the screen how it went, and return its record with secrets removed.

    Added to what run_fuzzball returns:
      text     what it printed, secrets removed
      error    its error text, secrets removed
      removed  how many secrets were taken out of text
      file     the file its output goes to, or None when it failed and printed nothing
    """
    print("  %s ..." % shown(args), end=" ", flush=True)
    step = run_fuzzball(args)
    step["text"] = scrub(step["out"])
    step["error"] = scrub(step["err"])
    step["removed"] = count_removed(step["out"], step["text"])
    step["file"] = file_name if (step["problem"] is None or step["text"]) else None
    print("ok" if step["problem"] is None else "FAILED: %s" % step["problem"], flush=True)
    for line in error_lines(step["error"])[-5:]:        # also the warnings of a command that worked
        print("      " + line, flush=True)
    return step


def gather(options):
    """Run every command for this workflow.

    Returns the steps in the order run, the job names whose logs were asked for, a few words on
    where those names came from, and the names found in describe.json but refused.
    """
    uuid = options.workflow
    steps = [run_step(["version"], "version.txt"),
             run_step(["workflow", "describe", uuid], "describe.txt"),
             run_step(["workflow", "describe", uuid, "-o", "json"], "describe.json"),
             run_step(["workflow", "events", uuid], "events.txt")]
    refused = []
    if options.jobs:
        jobs, source = options.jobs, "given with --job"
    else:
        try:
            jobs, refused = find_job_names(json.loads(steps[2]["out"]))
            source = "found in describe.json"
            if not jobs:
                source = 'describe.json has no usable names under "jobs" or "services"'
        except (ValueError, RecursionError):
            jobs, source = [], "describe.json could not be read as JSON"
    for job in jobs:
        steps.append(run_step(["workflow", "log", uuid, job, "-n", str(options.lines)], "log-%s.txt" % job))
    steps.append(run_step(["node", "list"], "nodes.txt"))
    return steps, jobs, source, refused


def read_fuzzfile(path):
    """Read the Fuzzfile through the filter. Returns the bytes to save and the lines the filter changed.

    The bytes are the file's own, an exact copy, unless the filter found something to replace.
    Each changed line is (its number, the line after the filter).
    """
    with open(path, "rb") as handle:
        original = handle.read()
    text = original.decode("utf-8", "replace")
    clean = scrub(text)
    if clean == text:
        return original, []
    pairs = zip(text.splitlines(), clean.splitlines())      # the filter never adds or drops a line
    changed = [(number, new) for number, (old, new) in enumerate(pairs, 1) if old != new]
    return clean.encode("utf-8", "replace"), changed


def readme(options, stamp, steps, facts, removed):
    """The text of README.md. facts are the list lines about the Fuzzfile and the logs.

    Every piece of captured text that goes in here has been through scrub().
    """
    failed = [step for step in steps if step["problem"] is not None]
    lines = ["# Fuzzball run record: %s" % options.name,
             "",
             scrub(options.note) if options.note else "No note was given for this run.",
             "",
             "- Workflow id: `%s`" % options.workflow,
             "- Recorded: %s" % time.strftime("%Y-%m-%d %H:%M:%S UTC", stamp)]
    lines += facts
    lines += ["",
              "This folder was written by `tools/fb_record.py`. Each file holds what CIQ's `fuzzball` "
              "command-line tool printed, with secrets taken out (see the end of this page).",
              "",
              "## Commands",
              "",
              "| Command | Result | File |",
              "|---|---|---|"]
    for step in steps:
        if step["problem"] is not None:
            result = "FAILED: " + step["problem"]
        elif step["text"]:
            result = "ok"
        else:
            result = "ok, but it printed nothing"
        lines.append("| `%s` | %s | %s |" % (shown(step["args"]), result,
                                             "`%s`" % step["file"] if step["file"] else "none"))
    lines += ["", "## Failed commands", ""]
    if not failed:
        lines += ["None. All %d commands worked." % len(steps), ""]
    for step in failed:
        error = error_lines(step["error"])
        if not error:
            said = "It printed no error text."
        elif len(error) < len(step["error"].strip().splitlines()):
            said = "The last %d lines of its error text:" % len(error)
        else:
            said = "Its error text:"
        lines += ["`%s` failed: %s. %s" % (shown(step["args"]), step["problem"], said), ""]
        if error:
            lines += ["    " + line for line in error] + [""]
        if step["file"]:
            lines += ["What it printed before it failed is in `%s`." % step["file"], ""]
    lines += ["## Secrets",
              "",
              "Before anything was written, the tool replaced each of these with `%s`: a JSON Web Token; "
              "the word Bearer and the token after it; the value of a name that contains TOKEN, SECRET, "
              "PASSWORD, KEY or CREDENTIAL. A Fuzzball secret reference (`secret://...`) and a shell variable "
              "in capital letters (`$NAME`) name a secret and do not hold one, so they are kept. Error text "
              "on this page went through the same filter." % MARK,
              ""]
    if removed:
        lines += ["Values replaced in the saved files: %s." % ", ".join("`%s`: %d" % pair for pair in removed), ""]
    else:
        lines += ["No value had to be replaced in the saved files.", ""]
    return "\n".join(lines)


def assemble(options, stamp, steps, jobs, source, refused, fuzzfile):
    """Decide what goes into the folder.

    Returns the files as (name, bytes), the number of values replaced per file as (name, count),
    and the lines of the Fuzzfile that the filter changed as (number, line after the filter).
    """
    files = [(step["file"], step["text"].encode("utf-8", "replace")) for step in steps if step["file"]]
    removed = [(step["file"], step["removed"]) for step in steps if step["file"] and step["removed"]]
    facts = ["- Fuzzfile: none was given"]
    changed = []
    if fuzzfile is not None:
        data, changed = fuzzfile
        copy_name = os.path.basename(options.fuzzfile)
        if copy_name.lower() in [name.lower() for name, _ in files] + ["readme.md"]:
            copy_name = "fuzzfile-" + copy_name         # never write over one of the files above
        files.append((copy_name, data))
        facts = ["- Fuzzfile: `%s`, copied into this folder as `%s`" % (
            os.path.relpath(os.path.realpath(options.fuzzfile), os.path.realpath(REPO)), copy_name)]
        if changed:
            removed.append((copy_name, sum(max(1, line.count(MARK)) for _, line in changed)))
            facts[0] += ". The copy is not exact. Lines changed by the filter: %s" % ", ".join(
                str(number) for number, _ in changed)
    facts.append("- Logs asked for: %s (%s)" % (", ".join("`%s`" % job for job in jobs) or "none", source))
    if refused:
        facts[-1] += ". Names found but not used, because this tool does not accept them: %s" % ", ".join(
            "`%s`" % scrub(json.dumps(name)) for name in refused)
    facts.append("- Log lines asked for, per job: %d" % options.lines)
    files.append(("README.md", readme(options, stamp, steps, facts, removed).encode("utf-8", "replace")))
    return files, removed, changed


def report(options, folder, steps, jobs, source, files, removed, changed):
    """Say on the screen what was saved, what failed and what to do next."""
    failed = [step for step in steps if step["problem"] is not None]
    print("")
    if not jobs:
        print("No logs were saved: %s. Run again with --job NAME to save a job's log." % source)
    if failed:
        print("%d of %d commands failed. README.md in the folder says which, with the error text."
              % (len(failed), len(steps)))
    if len(failed) == len(steps):
        print("Every command failed. Check that fuzzball is logged in and that the workflow id is right.")
    if removed:
        print("Values replaced with %s: %s." % (MARK, ", ".join("%s: %d" % pair for pair in removed)))
    if changed:
        print("The copy of the Fuzzfile is not exact. The filter changed these lines of %s:" % options.fuzzfile)
        for number, line in changed[:10]:
            print("    line %d: %s" % (number, line.strip()))
        if len(changed) > 10:
            print("    and %d more" % (len(changed) - 10))
        print("If one of them holds a real secret, take it out of %s before you commit that file."
              % options.fuzzfile)
    place = os.path.relpath(folder)
    if place.startswith(os.pardir):
        place = os.path.abspath(folder)
    print("Saved %d files in: %s" % (len(files), place))
    print("Read every file in that folder before you commit it. The filter does not catch every secret "
          "(not \"password: abc\", not \"--api-key abc\"). Look also for names and addresses that should "
          "not be public.")
    print("The lines to look at first:  grep -n -i -E \"%s\" %s/*" % (LOOK_FOR, shlex.quote(place)))


def main(argv=None, now=None):
    """Record one workflow. now is the time to stamp the folder with (seconds; default: the current time)."""
    options = parse_args(argv)
    if shutil.which(fuzzball_path()) is None:
        return stop("cannot find the fuzzball command-line tool (looked for %r). Put it on the PATH, "
                    "or set FUZZBALL_BIN to its full path." % fuzzball_path())
    stamp = time.gmtime(now)
    folder = os.path.join(options.out, time.strftime("%Y%m%d-%H%M%S", stamp) + "-" + options.name)
    if os.path.exists(options.out) and not os.path.isdir(options.out):
        return stop("--out %s is not a folder." % options.out)
    if os.path.exists(folder):
        return stop("%s is already there. Wait a second, or use another --name." % folder)
    fuzzfile = None
    if options.fuzzfile is not None:
        try:
            fuzzfile = read_fuzzfile(options.fuzzfile)
        except OSError as error:
            return stop("cannot read --fuzzfile %s (%s)." % (options.fuzzfile, error.strerror))

    print("Asking fuzzball about workflow %s. Each command gets %d seconds." % (options.workflow, TIMEOUT_S),
          flush=True)
    try:
        steps, jobs, source, refused = gather(options)
    except KeyboardInterrupt:
        print("\nStopped. Nothing was written.", file=sys.stderr)
        return 130

    # Nothing has been written so far. Everything below has been through the filter.
    files, removed, changed = assemble(options, stamp, steps, jobs, source, refused, fuzzfile)
    try:
        os.makedirs(folder)
        for name, data in files:
            with open(os.path.join(folder, name), "wb") as handle:
                handle.write(data)
    except OSError as error:
        return stop("could not write into %s (%s). Remove what is left of that folder." % (folder, error.strerror))
    report(options, folder, steps, jobs, source, files, removed, changed)
    return 1 if any(step["problem"] is not None for step in steps) else 0


if __name__ == "__main__":
    sys.exit(main())
