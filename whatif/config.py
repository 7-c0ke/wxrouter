"""WhatIf settings: environment variables, with a `.env` file at the repo root filling the gaps.

    LLM_ENDPOINTS   comma list of OpenAI-compatible bases   default http://localhost:8090/v1,http://localhost:8091/v1
    LLM_API_KEY     Bearer token for the LLM (optional; build.nvidia.com needs one). One key for every endpoint,
                    or a comma list with one key per LLM_ENDPOINTS entry, in the same order
    MODEL           model name sent in each request          default nemotron
    NODE_CAP        concurrent LLM calls per endpoint        default 4 (llama-server -np 4)
    THINKING        1 = ask for the model's thinking phase   default 0
    ROUTER          router.py base                           default http://localhost:8000
    BOX_A, BOX_B    forecast servers                         default http://localhost:8001, :8002
    STACK_A, STACK_B  node_stack supervisors (optional)      default http://localhost:8109, :8110

A variable set in the environment wins over the same name in `.env`. Standard library only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOXES = ("box-a", "box-b")


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; `#` comments, blank lines, `export ` prefixes and matching quotes are allowed."""
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


@dataclass(frozen=True)
class Config:
    llm_endpoints: tuple[str, ...]
    model: str
    node_cap: int
    llm_api_key: str | None
    thinking: bool
    router: str
    boxes: dict[str, str]
    stacks: dict[str, str]

    def key_for(self, endpoint: str) -> str | None:
        return key_for(self.llm_api_key, self.llm_endpoints, endpoint)

    def redacted(self) -> dict:
        return {"llm_endpoints": list(self.llm_endpoints), "model": self.model, "node_cap": self.node_cap,
                "llm_api_key": "set" if self.llm_api_key else None, "thinking": self.thinking,
                "router": self.router, "boxes": self.boxes, "stacks": self.stacks}


def key_for(api_key: str | None, endpoints, endpoint: str) -> str | None:
    """The key for one endpoint: a single key serves all; a comma list is matched to endpoints by position."""
    keys = [k.strip() for k in (api_key or "").split(",")]
    if len(keys) == 1:
        return keys[0] or None
    eps = [e.rstrip("/") for e in endpoints]
    if len(keys) != len(eps):
        raise ValueError(f"LLM_API_KEY has {len(keys)} keys for {len(eps)} LLM_ENDPOINTS")
    return keys[eps.index(endpoint.rstrip("/"))] or None


def load(env: dict[str, str] | None = None, dotenv: Path | None = ROOT / ".env") -> Config:
    merged = {**(read_dotenv(dotenv) if dotenv else {}), **(os.environ if env is None else env)}

    def get(key: str, default: str) -> str:
        return (merged.get(key) or default).strip()

    endpoints = tuple(u.strip().rstrip("/") for u in get("LLM_ENDPOINTS", "http://localhost:8090/v1,http://localhost:8091/v1").split(",") if u.strip())
    api_key = merged.get("LLM_API_KEY") or None
    for e in endpoints:
        key_for(api_key, endpoints, e)  # fail early on a key count that does not match
    return Config(
        llm_endpoints=endpoints,
        model=get("MODEL", "nemotron"),
        node_cap=max(1, int(get("NODE_CAP", "4"))),
        llm_api_key=api_key,
        thinking=get("THINKING", "0").lower() in ("1", "true", "yes", "on"),
        router=get("ROUTER", "http://localhost:8000").rstrip("/"),
        boxes={"box-a": get("BOX_A", "http://localhost:8001").rstrip("/"),
               "box-b": get("BOX_B", "http://localhost:8002").rstrip("/")},
        stacks={"box-a": get("STACK_A", "http://localhost:8109").rstrip("/"),
                "box-b": get("STACK_B", "http://localhost:8110").rstrip("/")},
    )
