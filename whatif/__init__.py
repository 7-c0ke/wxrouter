"""WhatIf: diagnose an incident on the two-box forecast stack with a team of small LLM agents.

A recorder keeps the last few minutes of telemetry, agents test one hypothesis each with cheap tools over a
window around the alert, and a scheduler decides which agents keep going. See whatif/README.md.
"""
