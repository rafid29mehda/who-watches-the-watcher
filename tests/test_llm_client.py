import json

import httpx
import openai
import pytest
from openai._types import NOT_GIVEN

from tests.fakes import FakeTransport, make_completion
from wwtw.budget import BudgetExceeded, BudgetLedger
from wwtw.config import Price
from wwtw.llm_client import (
    CachedChatClient,
    OllamaMonitorBackend,
    OpenAIMonitorBackend,
    ResponseCache,
    cache_key,
    clean_params,
)

PRICES = {
    "gpt-4o-mini-2024-07-18": Price(input=0.15, cached_input=0.075, output=0.60),
    "gpt-5-nano-2025-08-07": Price(input=0.05, cached_input=0.005, output=0.40),
}
MSGS = [{"role": "user", "content": "hi"}]


def _client(tmp_path, transport, cap=1.0, **kw):
    ledger = BudgetLedger(tmp_path / "l.sqlite", cap, PRICES)
    return CachedChatClient(
        cache=ResponseCache(tmp_path / "c.sqlite"),
        ledger=ledger,
        transport=transport,
        sleep=lambda s: None,
        **kw,
    ), ledger


def _rate_limit():
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.RateLimitError("rl", response=httpx.Response(429, request=req), body=None)


def test_clean_params_drops_sentinels():
    assert clean_params({"a": 1, "b": NOT_GIVEN, "c": None}) == {"a": 1, "c": None}


def test_temperature_default_applied_when_agentdojo_passes_not_given(tmp_path):
    t = FakeTransport(lambda p: make_completion("ok"))
    c, _ = _client(tmp_path, t, default_params={"temperature": 0.0})
    c.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", messages=MSGS, temperature=NOT_GIVEN, tools=NOT_GIVEN
    )
    assert t.calls[0]["temperature"] == 0.0 and "tools" not in t.calls[0]


def test_cache_hit_costs_nothing(tmp_path):
    t = FakeTransport(lambda p: make_completion("ok", prompt_tokens=1000, completion_tokens=100))
    c, ledger = _client(tmp_path, t)
    a, cached_a = c.create_with_meta(model="gpt-4o-mini-2024-07-18", messages=MSGS)
    spent = ledger.spent()
    b, cached_b = c.create_with_meta(model="gpt-4o-mini-2024-07-18", messages=MSGS)
    assert len(t.calls) == 1 and not cached_a and cached_b
    assert ledger.spent() == spent > 0
    assert a.choices[0].message.content == b.choices[0].message.content == "ok"


def test_namespaces_isolated(tmp_path):
    t = FakeTransport(lambda p: make_completion("ok"))
    cache = ResponseCache(tmp_path / "c.sqlite")
    ledger = BudgetLedger(tmp_path / "l.sqlite", 1.0, PRICES)
    for ns in ["main", "retest"]:
        CachedChatClient(cache=cache, ledger=ledger, transport=t, namespace=ns).create(
            model="gpt-4o-mini-2024-07-18", messages=MSGS
        )
    assert len(t.calls) == 2
    assert cache_key("main", "openai", {"x": 1}) != cache_key("retest", "openai", {"x": 1})


def test_retries_rate_limit_then_succeeds(tmp_path):
    t = FakeTransport([_rate_limit(), _rate_limit(), make_completion("ok")])
    c, _ = _client(tmp_path, t)
    assert (
        c.create(model="gpt-4o-mini-2024-07-18", messages=MSGS).choices[0].message.content == "ok"
    )
    assert len(t.calls) == 3


def test_non_retryable_error_releases_reservation(tmp_path):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    bad = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    t = FakeTransport([bad])
    c, ledger = _client(tmp_path, t)
    with pytest.raises(openai.BadRequestError):
        c.create(model="gpt-4o-mini-2024-07-18", messages=MSGS)
    assert ledger.spent() == 0.0


def test_budget_exceeded_prevents_call(tmp_path):
    t = FakeTransport(lambda p: make_completion("ok"))
    c, _ = _client(tmp_path, t, cap=0.0000001)
    with pytest.raises(BudgetExceeded):
        c.create(model="gpt-4o-mini-2024-07-18", messages=MSGS)
    assert t.calls == []


def test_openai_monitor_backend_params(tmp_path):
    t = FakeTransport(lambda p: make_completion('{"suspicion": 5}', model="gpt-5-nano-2025-08-07"))
    c, _ = _client(tmp_path, t)
    b = OpenAIMonitorBackend(c, "gpt-5-nano-2025-08-07", "minimal", True, 1000)
    out = b.complete(MSGS)
    p = t.calls[0]
    assert out.text == '{"suspicion": 5}' and not out.cached
    assert p["reasoning_effort"] == "minimal" and p["max_completion_tokens"] == 1000
    assert p["response_format"] == {"type": "json_object"} and "temperature" not in p


def test_ollama_backend_caches_and_unloads(tmp_path):
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content or b"{}")
        seen.append((request.url.path, body))
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {"content": '{"suspicion": 7}'},
                    "prompt_eval_count": 50,
                    "eval_count": 9,
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    b = OllamaMonitorBackend(ResponseCache(tmp_path / "c.sqlite"), "llama3.2:3b", http=http)
    assert b.is_available() and b.has_model()
    r1, r2 = b.complete(MSGS), b.complete(MSGS)
    assert r1.text == r2.text == '{"suspicion": 7}' and r2.cached and not r1.cached
    chat_calls = [s for s in seen if s[0] == "/api/chat"]
    assert len(chat_calls) == 1 and chat_calls[0][1]["options"]["num_ctx"] == 8192
    b.unload()
    assert seen[-1] == ("/api/generate", {"model": "llama3.2:3b", "keep_alive": 0})
