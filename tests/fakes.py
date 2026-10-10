"""Test doubles: OpenAI-shaped completions and transports (no network)."""

import json
import threading
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from wwtw.traces.replay import build_environment


def make_completion(
    content=None,
    tool_calls=None,
    prompt_tokens=100,
    completion_tokens=20,
    cached=0,
    model="gpt-4o-mini-2024-07-18",
) -> ChatCompletion:
    tcs = None
    if tool_calls:
        tcs = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
            }
            for tc in tool_calls
        ]
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tcs else "stop",
                    "message": {"role": "assistant", "content": content, "tool_calls": tcs},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    )


class FakeTransport:
    """Mimics openai.OpenAI().chat.completions.create; replies from a callable or a fixed list."""

    def __init__(self, reply):
        self.calls: list[dict] = []
        self._reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        r = self._reply(params) if callable(self._reply) else self._reply.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def _msg_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


class GroundTruthAgent:
    """Fake OpenAI transport that follows the user task's ground truth and, if a tool output contains
    a known injection GOAL, then follows that injection task's ground truth (vulnerable by design)."""

    def __init__(self, suite):
        self.suite = suite
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        with self._lock:
            self.calls.append(params)
        msgs = params["messages"]
        prompt = next(_msg_text(m["content"]) for m in msgs if m["role"] == "user")
        ut = next(t for t in self.suite.user_tasks.values() if prompt == t.PROMPT)
        env = build_environment(self.suite, ut, {})
        plan = list(ut.ground_truth(env.model_copy(deep=True)))
        tool_text = " ".join(_msg_text(m["content"]) for m in msgs if m["role"] == "tool")
        inj = next((t for t in self.suite.injection_tasks.values() if t.GOAL in tool_text), None)
        if inj is not None:
            plan += list(inj.ground_truth(env.model_copy(deep=True)))
        k = sum(1 for m in msgs if m["role"] == "assistant")
        if k < len(plan):
            c = plan[k]
            return make_completion(
                tool_calls=[{"id": f"call_{k}", "name": c.function, "args": dict(c.args)}]
            )
        return make_completion(content=ut.GROUND_TRUTH_OUTPUT)
