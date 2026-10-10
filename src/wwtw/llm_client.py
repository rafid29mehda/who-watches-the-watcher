"""Cached, budget-capped LLM access.

CachedChatClient mirrors openai.OpenAI().chat.completions.create so AgentDojo's OpenAILLM can use it
unchanged. Monitor backends wrap it (OpenAI) or Ollama's native API (local model).
"""

import contextlib
import hashlib
import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
from openai.types.chat import ChatCompletion

from wwtw.budget import BudgetLedger, Usage

_SENTINELS = {"NotGiven", "Omit"}
RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)"
        )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get(self, key: str) -> dict | None:
        row = self._conn().execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value: dict) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, json.dumps(value), time.time())
        )


def clean_params(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if type(v).__name__ not in _SENTINELS}


def cache_key(namespace: str, backend: str, params: dict) -> str:
    blob = json.dumps(
        {"ns": namespace, "backend": backend, "params": params}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def estimate_input_tokens(params: dict) -> int:
    chars = len(json.dumps(params.get("messages", []), default=str))
    chars += len(json.dumps(params.get("tools", []), default=str))
    return chars // 3 + 50  # deliberately conservative (~4 chars/token in practice)


def usage_of(completion: ChatCompletion) -> Usage:
    u = completion.usage
    if u is None:
        return Usage(0, 0, 0)
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return Usage(u.prompt_tokens, cached, u.completion_tokens)


class CachedChatClient:
    def __init__(
        self,
        *,
        cache: ResponseCache,
        ledger: BudgetLedger,
        namespace: str = "main",
        default_params: dict | None = None,
        transport=None,
        max_attempts: int = 6,
        sleep=time.sleep,
        reserve_output_tokens: int = 2000,
    ):
        self.cache = cache
        self.ledger = ledger
        self.namespace = namespace
        self.default_params = dict(default_params or {})
        self._transport = (
            transport if transport is not None else openai.OpenAI(max_retries=0, timeout=120)
        )
        self.max_attempts = max_attempts
        self._sleep = sleep
        self.reserve_output_tokens = reserve_output_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs) -> ChatCompletion:
        return self.create_with_meta(**kwargs)[0]

    def create_with_meta(self, **kwargs) -> tuple[ChatCompletion, bool]:
        params = {**self.default_params, **clean_params(kwargs)}
        key = cache_key(self.namespace, "openai", params)
        hit = self.cache.get(key)
        if hit is not None:
            return ChatCompletion.model_validate(hit), True
        max_out = params.get("max_completion_tokens") or self.reserve_output_tokens
        rid = self.ledger.reserve(
            params["model"], estimate_input_tokens(params), max_out, self.namespace
        )
        try:
            completion = self._call_with_retries(params)
        except BaseException:
            self.ledger.release(rid)
            raise
        self.ledger.settle(rid, usage_of(completion))
        self.cache.put(key, completion.model_dump(mode="json"))
        return completion, False

    def _call_with_retries(self, params: dict) -> ChatCompletion:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._transport.chat.completions.create(**params)
            except RETRYABLE:
                if attempt == self.max_attempts:
                    raise
                self._sleep(min(60.0, 2.0**attempt) * random.uniform(0.5, 1.0))
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    cached: bool


class OpenAIMonitorBackend:
    def __init__(
        self,
        client: CachedChatClient,
        model: str,
        reasoning_effort: str | None,
        json_mode: bool,
        max_output_tokens: int,
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.max_output_tokens = max_output_tokens

    def complete(self, messages: list[dict]) -> Completion:
        params: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        completion, cached = self.client.create_with_meta(**params)
        return Completion(completion.choices[0].message.content or "", usage_of(completion), cached)


class OllamaMonitorBackend:
    def __init__(
        self,
        cache: ResponseCache,
        model: str,
        num_ctx: int = 8192,
        namespace: str = "main",
        base_url: str = "http://localhost:11434",
        http: httpx.Client | None = None,
        keep_alive: str = "2m",
    ):
        self.cache = cache
        self.model = model
        self.num_ctx = num_ctx
        self.namespace = namespace
        self.http = http if http is not None else httpx.Client(base_url=base_url, timeout=600)
        self.keep_alive = keep_alive

    def complete(self, messages: list[dict]) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": self.num_ctx, "temperature": 0, "seed": 0},
        }
        key = cache_key(self.namespace, "ollama", payload)
        hit = self.cache.get(key)
        if hit is not None:
            return Completion(hit["text"], Usage(*hit["usage"]), True)
        r = self.http.post("/api/chat", json={**payload, "keep_alive": self.keep_alive})
        r.raise_for_status()
        d = r.json()
        usage = Usage(int(d.get("prompt_eval_count", 0)), 0, int(d.get("eval_count", 0)))
        text = d["message"]["content"]
        self.cache.put(key, {"text": text, "usage": [usage.input_tokens, 0, usage.output_tokens]})
        return Completion(text, usage, False)

    def is_available(self) -> bool:
        try:
            return self.http.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self) -> bool:
        models = self.http.get("/api/tags").json().get("models", [])
        return any(m.get("name") == self.model for m in models)

    def unload(self) -> None:
        with contextlib.suppress(httpx.HTTPError):
            self.http.post("/api/generate", json={"model": self.model, "keep_alive": 0})
