"""The Control Room chat sends infrastructure questions to WhatIf (ui/agent.py infra_question, Agent._diagnose;
ui/server.py run_diagnosis). Forecasts, status questions and commands stay with the assistant.

    python -m pytest tests/test_ui_whatif_chat.py -q

Telemetry is the WhatIf recorder's own format, made by tests/whatif_fakes.py. Nothing is sent anywhere: the
router and boxes are at 127.0.0.1:9, which does not answer.
"""
import asyncio
import copy
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ui"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
for name in ("server", "agent", "events"):
    sys.modules.pop(name, None)
import agent as A  # noqa: E402
from whatif_fakes import make_samples  # noqa: E402

ROUTED = ["why is box-a slow?", "What's wrong?", "diagnose the latency", "why did the router switch to box-a?",
          "Why are requests failing?", "what happened to box-b", "root cause of the p95 spike?",
          "is something wrong with the cluster?", "why is the forecast slow?", "explain the errors on box-b",
          "how come the latency is so high", "Why is box_b timing out?"]
NOT_ROUTED = ["forecast for Chicago tomorrow", "why is it so cold in Chicago?", "which box is active?", "kill box-a",
              "slow box-a", "why will it rain in Seattle", "Can you fix box-a?", "what's the weather in Paris",
              "is box-b down?", "set the policy to balanced", "", "hello"]


@pytest.mark.parametrize("text", ROUTED)
def test_infrastructure_questions_are_routed(text):
    assert A.infra_question(text), text


@pytest.mark.parametrize("text", NOT_ROUTED)
def test_forecasts_status_and_commands_are_not(text):
    assert not A.infra_question(text), text


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import server as S
    ops = S.Ops("http://127.0.0.1:9", {"box-a": "http://127.0.0.1:9", "box-b": "http://127.0.0.1:9"})
    app = S.create_app(ops, S.Agent(ops, None, "x", None), telemetry=False)
    with TestClient(app) as c:
        c.app_ = app
        yield c


def ask(client, text):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": text}]})
    assert r.status_code == 200, r.text
    return r.json()


def test_chat_question_gets_a_whatif_diagnosis_and_an_incident(client):
    client.app_.state.recorder.buf.extend(make_samples("box-b:slow", t0=time.time() - 30, n_pre=35, n_post=30))
    res = ask(client, "Why is box-b so slow?")
    assert res["mode"] == "whatif:triage"
    assert "most likely **box-b:slow** (the box's GPU got slower);" in res["reply"]
    assert "Evidence: box-b compute 177 vs 59 ms -> slow." in res["reply"]
    assert "Apply remedy** on incident #1" in res["reply"] and "(not applied)" in res["reply"]
    step = res["steps"][0]
    assert step["tool"] == "whatif_diagnose" and step["result"]["ranking"][0] == "box-b:slow"
    assert step["result"]["tokens"] == 0, "a clear-cut signal needs no LLM"
    assert res["note"].endswith("the rule (no LLM), 0 tokens.")
    ev = [e for e in client.app_.state.hub.events if e["src"] == "incident"][-1]
    assert "asked in the chat" in ev["text"] and ev["can_apply"] and ev["incident"] == 1


def test_not_enough_telemetry_says_so_and_changes_nothing(client):
    res = ask(client, "what's wrong with the router?")
    assert res["mode"] == "whatif" and "cannot diagnose right now: not enough telemetry yet" in res["reply"]
    assert res["steps"][0]["result"]["error"].startswith("not enough telemetry")
    assert not [e for e in client.app_.state.hub.events if e["src"] == "incident"]


def test_unreachable_router_is_not_diagnosed(client):
    samples = make_samples("box-b:slow", t0=time.time() - 30, n_pre=35, n_post=30)
    for s in samples:
        s["router"] = None
    client.app_.state.recorder.buf.extend(samples)
    res = ask(client, "why are requests failing?")
    assert "router unreachable" in res["reply"]
    assert client.post("/api/incident/diagnose", json={"policy": "rule"}).status_code == 503


def test_other_messages_still_go_to_the_assistant(client):
    client.app_.state.recorder.buf.extend(make_samples("box-b:slow", t0=time.time() - 30, n_pre=35, n_post=30))
    res = ask(client, "which box is active?")
    assert res["mode"] == "offline"
    assert [s["tool"] for s in res["steps"]] == ["get_cluster_status"]


def test_weather_questions_are_answered_by_the_weather_model(client):
    for text in ("forecast for Chicago tomorrow", "Why is it so cold in Chicago?", "weather at 41.9, -87.6"):
        res = ask(client, text)
        assert res["mode"] == "weather-model", text
        assert [s["tool"] for s in res["steps"]][-1] == "get_forecast", text
        assert "Answered by the weather model" in res["note"] and "no LLM" in res["note"], text
    assert res["steps"][0]["args"]["lat"] == 41.9
    assert "does not say why" in ask(client, "Why is it so cold in Chicago?")["reply"]
    assert "does not say why" not in ask(client, "forecast for Chicago tomorrow")["reply"]


@pytest.mark.parametrize("text", ["forecast for Chicago tomorrow", "Why is it so cold in Chicago?", "weather in Tokyo",
                                  "37.7, -122.4 next 3 days"])
def test_weather_question(text):
    assert A.weather_question(text), text


@pytest.mark.parametrize("text", ["which box served Chicago?", "is box-b down?", "kill box-a", "Set the policy to "
                                  "balanced", "weather in Pune", "hello", ""])
def test_not_a_weather_question(text):
    assert not A.weather_question(text), text


def test_llm_mode_answers_weather_without_the_llm():
    ops = type("Ops", (), {"boxes": {"box-a": "", "box-b": ""}, "stacks": {}})()

    async def forecast(lat, lon, hours, members=1, place=None):
        return {"summary": {"min_c": 1.0, "max_c": 2.0, "precip_mm": None, "max_wind_ms": 3.0}, "points": [],
                "low_confidence": False, "members_temp": None, "served_by": "box-b", "simulated": False}
    ops.forecast = forecast
    ag = A.Agent(ops, "http://127.0.0.1:9/v1", "m")    # nothing listens there: any LLM call would fail
    res = asyncio.run(ag.chat([{"role": "user", "content": "Weather in Chicago for the next 48 hours"}]))
    assert res["mode"] == "weather-model" and "LLM" not in res.get("reply", "")
    assert res["note"] == "Answered by the weather model on box-b through the router: the numbers are the model's, no LLM."


def test_llm_mode_does_not_call_the_llm_for_an_infrastructure_question():
    seen = []

    async def diagnoser(policy, source="page"):
        seen.append((policy, source))
        d = {"ranking": ["box-a:hang", "box-a:slow", "box-b:slow"], "trail": ["[rule] box-a probe stalls 9 -> hang"],
             "tokens": 0, "llm_calls": 0}
        return {"incident": 7, "policy": policy, "diagnosis": d, "remedy": "box-a:hang: POST x", "can_apply": True,
                "engine": "Nemotron via NVIDIA API (m)", "meaning": "The server accepts requests and never answers"}

    ops = type("Ops", (), {"boxes": {"box-a": "", "box-b": ""}, "stacks": {}})()
    ag = A.Agent(ops, "http://127.0.0.1:9/v1", "m")    # nothing listens there: any LLM call would fail
    ag.diagnoser = diagnoser
    t = time.perf_counter()
    res = asyncio.run(ag.chat([{"role": "user", "content": "why is box-a hanging?"}]))
    assert time.perf_counter() - t < 1
    assert seen == [("triage", "chat")]
    assert res["reply"].startswith("**WhatIf diagnosis #7**: most likely **box-a:hang** (the server accepts")
    assert "Evidence: box-a probe stalls 9 -> hang." in res["reply"]


def test_without_a_diagnoser_nothing_changes():
    ops = type("Ops", (), {"boxes": {"box-a": "", "box-b": ""}, "stacks": {}})()
    ag = A.Agent(ops, None, "m")
    assert ag.diagnoser is None
    res = asyncio.run(ag.chat([{"role": "user", "content": "what's wrong?"}]))
    assert res["mode"] == "offline"
