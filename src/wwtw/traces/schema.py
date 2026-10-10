"""Normalized, frozen representation of one agent episode."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    call_id: str | None
    function: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Event:
    kind: str  # "assistant_text" | "tool_call" | "tool_result"
    text: str = ""
    call: ToolCall | None = None
    ordinal: int | None = None  # tool-call ordinal (execution order) for tool_call/tool_result
    error: str | None = None


@dataclass(frozen=True)
class ExecutedCall:
    ordinal: int
    call: ToolCall
    result: Event | None
    event_index: int


@dataclass(frozen=True)
class Episode:
    episode_id: str
    suite: str
    user_task_id: str
    injection_task_id: str | None
    attack: str  # "none" | "important_instructions" | "monitor_aware"
    user_prompt: str
    injections: dict[str, str]
    events: tuple[Event, ...]
    final_output: str
    utility: bool
    security: bool  # AgentDojo semantics: True means the injection goal was achieved
    error: str | None = None

    def executed_calls(self) -> list[ExecutedCall]:
        results = {e.ordinal: e for e in self.events if e.kind == "tool_result"}
        return [
            ExecutedCall(e.ordinal, e.call, results.get(e.ordinal), i)
            for i, e in enumerate(self.events)
            if e.kind == "tool_call"
        ]


def make_episode_id(
    attack: str, suite: str, user_task_id: str, injection_task_id: str | None
) -> str:
    return f"{attack}/{suite}/{user_task_id}/{injection_task_id or 'none'}"


def episode_to_dict(ep: Episode) -> dict:
    return asdict(ep)


def _event_from_dict(d: dict) -> Event:
    call = ToolCall(**d["call"]) if d.get("call") else None
    return Event(kind=d["kind"], text=d["text"], call=call, ordinal=d["ordinal"], error=d["error"])


def episode_from_dict(d: dict) -> Episode:
    return Episode(**{**d, "events": tuple(_event_from_dict(e) for e in d["events"])})
