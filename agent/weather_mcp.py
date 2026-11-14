"""The weather desk's tools as a Model Context Protocol (MCP) server over stdio.

An MCP client (Claude Code, Codex, any agent loop) starts this file and talks to it on stdin and stdout:
JSON-RPC 2.0, one JSON message per line, UTF-8. The tools are the six in tools.py. Each one calls the router
at $ROUTER_URL (default http://127.0.0.1:8100). Only protocol messages go to stdout; logs go to stderr.

Register it with Claude Code (run this in the repo root, or write the full path yourself):

    claude mcp add switchyard -e ROUTER_URL=http://127.0.0.1:8100 -- python3 "$(pwd)/agent/weather_mcp.py"

Any other client, in its "mcpServers" JSON (use the full path):

    {"mcpServers": {"switchyard": {
        "command": "python3",
        "args": ["/full/path/to/agent/weather_mcp.py"],
        "env": {"ROUTER_URL": "http://127.0.0.1:8100"}}}}

Try it by hand (this prints the tool list):

    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python agent/weather_mcp.py

Methods: initialize, notifications/initialized, ping, tools/list, tools/call. It handles one message at a
time. It stops when stdin closes, and exits with code 0 on SIGTERM or SIGINT.

It speaks the MCP revisions that open with "initialize" (2024-11-05 to 2025-11-25). A client built for the
newer 2026-07-28 revision first tries "server/discover", gets "Method not found", and falls back to those.

Written at the hackathon (Nov 13-14, 2026).
"""
import json
import os
import signal
import sys
import time

import tools

# The protocol revisions with the initialize handshake that this server speaks, oldest first.
# A client asking for anything else is offered the last one.
PROTOCOL_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
SERVER_INFO = {"name": "switchyard", "version": "1.0.0"}

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def log(text):
    """Logs go to stderr. stdout carries protocol messages and nothing else."""
    sys.stderr.write("weather_mcp: %s\n" % text)
    sys.stderr.flush()


def result_message(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_message(request_id, code, message):
    """A JSON-RPC error. With no usable request id (a line that is not JSON, say) the id is left out."""
    answer = {"jsonrpc": "2.0"}
    if request_id is not None:
        answer["id"] = request_id
    answer["error"] = {"code": code, "message": message}
    return answer


def handle_line(raw):
    """Handle one line from the client (bytes). Returns the message to send back, or None for no answer."""
    try:
        message = json.loads(raw.decode("utf-8"))
    except ValueError as exc:           # not UTF-8, or not JSON
        return error_message(None, PARSE_ERROR, "Parse error: this line is not JSON (%s)" % exc)
    if not isinstance(message, dict):
        return error_message(None, INVALID_REQUEST, "Invalid request: send one JSON object per line")
    method = message.get("method")
    if not isinstance(method, str):
        if "result" in message or "error" in message:
            return None                 # an answer to a request; this server never asks the client anything
        return error_message(None, INVALID_REQUEST, "Invalid request: there is no method")
    if "id" not in message:
        return None                     # a notification (notifications/initialized, say): never answered
    request_id = message["id"]
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int, float)):
        return error_message(None, INVALID_REQUEST, "Invalid request: the id must be text or a number")
    params = message.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return error_message(request_id, INVALID_PARAMS, "Invalid params: params must be an object")
    try:
        return handle_request(request_id, method, params)
    except Exception as exc:            # a bug here must not take the server down
        log("%s failed: %s: %s" % (method, type(exc).__name__, exc))
        return error_message(request_id, INTERNAL_ERROR, "Internal error: %s: %s" % (type(exc).__name__, exc))


def handle_request(request_id, method, params):
    """Answer one request."""
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1]
        log("initialize: the client asked for protocol %s; speaking %s" % (asked, version))
        return result_message(request_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                                           "serverInfo": SERVER_INFO})
    if method == "ping":
        return result_message(request_id, {})
    if method == "tools/list":
        listed = [{"name": tool["name"], "description": tool["description"], "inputSchema": tool["input_schema"]}
                  for tool in tools.TOOLS]
        return result_message(request_id, {"tools": listed})
    if method == "tools/call":
        return handle_tools_call(request_id, params)
    return error_message(request_id, METHOD_NOT_FOUND, "Method not found: %s" % method)


def handle_tools_call(request_id, params):
    """tools/call. A tool that fails still gives a result (isError true), so the model can read why.

    Only a tool name we do not have, or arguments that are not an object, is a JSON-RPC error.
    """
    name = params.get("name")
    arguments = params.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(name, str) or name not in tools.TOOL_NAMES:
        return error_message(request_id, INVALID_PARAMS, "Unknown tool: %s" % (name,))
    if not isinstance(arguments, dict):
        return error_message(request_id, INVALID_PARAMS, "Invalid params: arguments must be an object")
    started = time.time()
    outcome = tools.call_tool(name, arguments)
    failed = "error" in outcome
    log("%s %s in %d ms, node %s" % (name, "FAILED (%s)" % outcome["error"] if failed else "ok",
                                     (time.time() - started) * 1000, outcome.get("node")))
    return result_message(request_id, {
        "content": [{"type": "text", "text": json.dumps(outcome, ensure_ascii=False)}],
        "structuredContent": outcome,
        "isError": failed,
    })


def encode(message):
    """One message as one line of bytes. json.dumps never puts a raw newline inside."""
    try:
        text = json.dumps(message, allow_nan=False)
    except ValueError as exc:           # a NaN or Infinity slipped into a result; that is not JSON
        text = json.dumps(error_message(message.get("id"), INTERNAL_ERROR, "Internal error: %s" % exc))
    return text.encode("utf-8") + b"\n"


def stop(signum, frame):
    """SIGTERM or SIGINT: leave at once, with exit code 0."""
    raise SystemExit(0)


def main():
    if sys.argv[1:]:                    # it takes no arguments: show how to use it instead of sitting there
        sys.stderr.write(__doc__)
        return 0 if sys.argv[1] in ("-h", "--help") else 2
    out = sys.stdout.buffer             # the protocol stream
    sys.stdout = sys.stderr             # a stray print() anywhere now lands in the log, not in the protocol
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log("ready, waiting for MCP messages on stdin. Router: %s" % tools.router_url())
    try:
        while True:
            raw = sys.stdin.buffer.readline()
            if not raw:
                break                   # the client closed stdin: time to go
            if not raw.strip():
                continue                # a blank line: nothing to answer
            answer = handle_line(raw)
            if answer is not None:
                out.write(encode(answer))
                out.flush()
    except BrokenPipeError:             # the client went away while we were writing
        os.dup2(os.open(os.devnull, os.O_WRONLY), out.fileno())     # so the exit does not complain as well
    return 0


if __name__ == "__main__":
    sys.exit(main())
