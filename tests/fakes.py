"""Test doubles: OpenAI-shaped completions and transports (no network)."""

import json
from types import SimpleNamespace

from openai.types.chat import ChatCompletion


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
