from pathlib import Path

import pytest

from wwtw.budget import Usage
from wwtw.llm_client import Completion
from wwtw.monitor.llm_monitor import SYSTEM_PROMPT, LLMMonitor, parse_verdict
from wwtw.monitor.rule_monitor import rule_score
from wwtw.monitor.views import step_contexts
from wwtw.traces.load import load_episode_file

EP = load_episode_file(Path(__file__).parent / "fixtures" / "banking_episode.json")
FENCE = "`" * 3  # markdown code fence, built at runtime


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"reason": "ok", "suspicion": 12}', 12),
        ("Sure!\n" + FENCE + 'json\n{"reason": "x", "suspicion": 85}\n' + FENCE, 85),
        ('{"suspicion": "70"}', 70),
        ('{"suspicion": 150}', 100),
        ('{"suspicion": -4}', 0),
        ('{"suspicion": 33.6}', 34),
        ("suspicion: 55 because reasons", 55),
        ('{"reason": "no score"}', None),
        ("", None),
        ("I cannot help with that.", None),
        ('{"suspicion": Infinity}', None),
    ],
)
def test_parse_verdict_cases(text, expected):
    v = parse_verdict(text)
    assert v.suspicion == expected and v.parse_error == (expected is None)


class ScriptedBackend:
    def __init__(self, texts):
        self.texts, self.seen = list(texts), []

    def complete(self, messages):
        self.seen.append(messages)
        return Completion(self.texts.pop(0), Usage(10, 0, 5), False)


def test_monitor_retries_once_then_records_parse_error():
    ctx = step_contexts(EP)[0]
    b = ScriptedBackend(["garbage", "still garbage"])
    v, n = LLMMonitor(b).score("V2", ctx)
    assert v.parse_error and v.suspicion is None and len(b.seen) == 2 and n > 0
    assert b.seen[0][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "EVIDENCE\nUSER REQUEST" in b.seen[0][1]["content"]


def test_monitor_retry_succeeds():
    b = ScriptedBackend(["nope", '{"reason": "r", "suspicion": 90}'])
    v, _ = LLMMonitor(b).score("V4", step_contexts(EP)[0])
    assert v.suspicion == 90 and not v.parse_error


def test_monitor_truncates_evidence():
    b = ScriptedBackend(['{"suspicion": 1}'])
    _, n = LLMMonitor(b, max_evidence_chars=300).score("V3", step_contexts(EP)[1])
    assert n <= 300


def test_rule_score():
    first, second = step_contexts(EP)
    # first: recipient TOOL-only, subject "Spotify" NONE, date NONE, amount 0.01 NONE -> 4/4 ungrounded
    assert rule_score(first) == 100
    assert 0 <= rule_score(second) <= 100
