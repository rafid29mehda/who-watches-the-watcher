"""Convert AgentDojo messages / TraceLogger JSON files into Episodes (events in execution order)."""

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from wwtw.traces.schema import (
    Episode,
    Event,
    ToolCall,
    episode_from_dict,
    episode_to_dict,
    make_episode_id,
)


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join((b.get("content") or "") for b in content if isinstance(b, Mapping))


def _call(tc) -> ToolCall:
    if isinstance(tc, Mapping):
        return ToolCall(tc.get("id"), tc["function"], dict(tc.get("args") or {}))
    return ToolCall(tc.id, tc.function, dict(tc.args))


def events_from_messages(messages: Sequence[Mapping]) -> tuple[str, tuple[Event, ...], str]:
    """Return (user_prompt, events, final_output). Each executed call is followed by its result,
    so the prefix before call k contains exactly what an online gate sees before executing k."""
    user_prompt = ""
    events: list[Event] = []
    pending: list[ToolCall] = []
    ordinal = 0

    def flush_unexecuted() -> None:
        nonlocal ordinal
        for c in pending:
            events.append(Event("tool_call", call=c, ordinal=ordinal))
            ordinal += 1
        pending.clear()

    for m in messages:
        role = m["role"]
        if role == "user":
            flush_unexecuted()
            if not user_prompt:
                user_prompt = _text(m["content"])
        elif role == "assistant":
            flush_unexecuted()
            text = _text(m.get("content"))
            if text.strip():
                events.append(Event("assistant_text", text=text))
            pending.extend(_call(tc) for tc in (m.get("tool_calls") or []))
        elif role == "tool":
            executed = _call(m["tool_call"])
            match = next(
                (
                    i
                    for i, c in enumerate(pending)
                    if c.call_id is not None and c.call_id == executed.call_id
                ),
                None,
            )
            if match is None and pending:
                match = 0
            if match is not None:
                pending.pop(match)
            events.append(Event("tool_call", call=executed, ordinal=ordinal))
            events.append(
                Event(
                    "tool_result",
                    text=_text(m.get("content")),
                    call=executed,
                    ordinal=ordinal,
                    error=m.get("error"),
                )
            )
            ordinal += 1
    flush_unexecuted()
    last = messages[-1] if messages else None
    final = _text(last.get("content")) if last is not None and last["role"] == "assistant" else ""
    return user_prompt, tuple(events), final


def load_episode_file(path: str | Path) -> Episode:
    d = json.loads(Path(path).read_text())
    user_prompt, events, final = events_from_messages(d.get("messages") or [])
    attack = d.get("attack_type") or "none"
    return Episode(
        episode_id=make_episode_id(
            attack, d["suite_name"], d["user_task_id"], d.get("injection_task_id")
        ),
        suite=d["suite_name"],
        user_task_id=d["user_task_id"],
        injection_task_id=d.get("injection_task_id"),
        attack=attack,
        user_prompt=user_prompt,
        injections=d.get("injections") or {},
        events=events,
        final_output=final,
        utility=bool(d.get("utility")),
        security=bool(d.get("security")),
        error=d.get("error"),
    )


def load_run_dir(run_dir: str | Path, attack: str) -> list[Episode]:
    """run_dir = runs/<pipeline_name>. Injection tasks run as user tasks are skipped."""
    files = sorted(Path(run_dir).glob(f"*/user_task_*/{attack}/*.json"))
    return [load_episode_file(f) for f in files]


def write_episodes(episodes: Sequence[Episode], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(episode_to_dict(ep), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_episodes(path: str | Path) -> list[Episode]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [episode_from_dict(json.loads(line)) for line in f if line.strip()]
