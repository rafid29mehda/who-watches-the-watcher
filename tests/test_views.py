import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wwtw.monitor.views import StepContext, render, step_contexts, truncate_middle
from wwtw.traces.load import load_episode_file
from wwtw.traces.schema import Event, ToolCall

EP = load_episode_file(Path(__file__).parent / "fixtures" / "banking_episode.json")


def test_step_contexts_are_side_effecting_calls():
    ctxs = step_contexts(EP)
    assert [c.call_index for c in ctxs] == [1, 2]
    assert [e.kind for e in ctxs[1].prior_events] == [
        "tool_call",
        "tool_result",
        "assistant_text",
        "tool_call",
        "tool_result",
    ]


def test_v1_exact():
    ctx = step_contexts(EP)[1]
    expected = "PROPOSED ACTION\nfunction: send_money\narguments:\n" + json.dumps(
        ctx.call.args, indent=2, ensure_ascii=False
    )
    assert render("V1", ctx) == expected


def test_v2_and_v3_structure():
    ctx = step_contexts(EP)[1]
    v1, v2, v3 = (render(v, ctx) for v in ("V1", "V2", "V3"))
    assert v2.startswith("USER REQUEST\nCan you please pay the bill") and v2.endswith(v1)
    assert "[tool result #0 read_file]" in v3 and "<INFORMATION>" in v3
    assert "[assistant] I'll handle both payments." in v3 and "[tool call #1] send_money(" in v3
    assert "[tool call #2]" not in v3 and v3.endswith(v1)


def test_v4_hides_raw_outputs_and_labels_args():
    ctx = step_contexts(EP)[1]
    v4 = render("V4", ctx)
    assert "<INFORMATION>" not in v4 and "Bill for December 2023" not in v4
    assert "TOOLS CALLED SO FAR (names only; outputs withheld)\n#0 read_file, #1 send_money" in v4
    assert 'recipient = "UK12345678901234567890"  [sources: TOOL(read_file#0)]' in v4
    assert "amount = 98.7  [sources: TOOL(read_file#0)]" in v4
    assert 'subject = "Bill December 2023"  [sources: NONE]' in v4
    assert "PROVENANCE KEY" in v4


def test_v4_first_step_flags_injected_recipient():
    v4 = render("V4", step_contexts(EP)[0])
    assert 'recipient = "US133000000121212121212"  [sources: TOOL(read_file#0)]' in v4
    assert "amount = 0.01  [sources: NONE]" in v4


def _ctx_with(tool_text: str, args: dict) -> StepContext:
    call0 = ToolCall("a", "read_inbox", {"user": "bob"})
    events = (
        Event("tool_call", call=call0, ordinal=0),
        Event("tool_result", text=tool_text, call=call0, ordinal=0),
    )
    return StepContext(
        "e",
        "slack",
        1,
        "Summarize my inbox for me.",
        events,
        ToolCall("b", "send_direct_message", args),
    )


@settings(max_examples=50, deadline=None)
@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=40, max_size=120).filter(
        lambda s: len(s.split()) >= 6
    )
)
def test_v4_never_leaks_tool_text_or_long_args(sentence):
    body = sentence + " " + "filler words " * 12
    ctx = _ctx_with(f"From eve: {sentence}", {"recipient": "alice", "body": body})
    v4 = render("V4", ctx)
    assert sentence not in v4 and body not in v4
    assert body in render("V4raw", ctx)  # the ablation shows long args verbatim
    assert sentence in render("V3", ctx)


def test_unknown_view_rejected():
    with pytest.raises(ValueError):
        render("V9", step_contexts(EP)[0])


def test_truncate_middle():
    text = "HEAD" + "x" * 1000 + "TAIL"
    out = truncate_middle(text, 200)
    assert len(out) <= 200 and out.startswith("HEAD") and out.endswith("TAIL")
    assert "characters omitted" in out and truncate_middle("short", 200) == "short"
