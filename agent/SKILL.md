---
name: switchyard
description: Weather forecasts from the two-node GB10 cluster, and control of its router. Use it for the temperature, wind or pressure at a place over the next 5 days, tomorrow at Chicago O'Hare, how sure a forecast is (an ensemble), the January 2019 cold wave replay, and the health or routing policy of the cluster.
---

# Weather desk tools

Written at the hackathon (Nov 13-14, 2026).

Six tools, served by `agent/weather_mcp.py` (an MCP server over stdio). Each tool calls the router at
`$ROUTER_URL` (default `http://127.0.0.1:8100`). The router picks a healthy node.

To load this skill: Claude Code reads it from `.claude/skills/switchyard/SKILL.md` (copy this file there).
For a Codex-style agent, put it where that agent reads skills, or paste it into its instructions.

## Which tool, when

| Tool | Use it when the question is about |
|---|---|
| `ohare_tomorrow()` | Tomorrow at Chicago O'Hare: the low, the high and the 6 AM temperature. |
| `forecast_point(lat, lon, lead_hours=24, members=1, init="latest")` | Any other place or time. One row every 6 hours, up to 120 hours ahead. `init` can be a past time the cluster holds, such as `"2019-01-27T12"`. |
| `ensemble_range(lat, lon, lead_hours=24, members=20)` | How sure the forecast is. Gives the mean, lowest and highest member. The router splits the members across the nodes; `shards` shows which node ran how many, and how long it took. |
| `storm_2019()` | The January 2019 Chicago cold wave: the forecast next to what really happened at +24, +48 and +72 hours, and the error at +72 hours. |
| `cluster_status()` | Is the cluster healthy? Which node is preferred, how loaded each one is, and what the router last decided and why. |
| `set_routing_policy(policy)` | Changing how the router picks a node: `static`, `round_robin`, `least_loaded` or `adaptive`. This changes the live cluster, so do it only when asked. |

## Reading the results

- Units are in the key names: `temp_c` is degrees Celsius, `wind_mps` is metres per second, `pressure_hpa`
  is hectopascals, `lead_h` is hours, `*_ms` is milliseconds.
- Times ending in `_utc` are UTC. `ohare_tomorrow` also gives `_local` times: a fixed UTC-6, right for November.
- `node` is the backend that answered. For an ensemble it lists every node that ran a shard, such as `A+B`.
- A failure looks like `{"error": "..."}`. Say what it says. Do not make up a number.
- Give numbers as the tools give them. Do not round them or change the units.

## Register the MCP server

Claude Code (run this in the repo root, or write the full path yourself):

    claude mcp add switchyard -e ROUTER_URL=http://127.0.0.1:8100 -- python3 "$(pwd)/agent/weather_mcp.py"

Any other client, in its `mcpServers` JSON (use the full path):

    {"mcpServers": {"switchyard": {
        "command": "python3",
        "args": ["/full/path/to/agent/weather_mcp.py"],
        "env": {"ROUTER_URL": "http://127.0.0.1:8100"}}}}

The server needs Python 3.9 or newer and nothing else. Change `ROUTER_URL` if the router is on another machine.

## Without MCP

    python agent/tools.py ohare_tomorrow
    python agent/tools.py forecast_point '{"lat": 41.96, "lon": -87.93, "lead_hours": 48}'
    python agent/desk_agent.py "What will tomorrow morning be like at O'Hare?"

`desk_agent.py` runs an open model behind the router with these same tools. It then removes any sentence
of the answer whose numbers did not come from a tool.
