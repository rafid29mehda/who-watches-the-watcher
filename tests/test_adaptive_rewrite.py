"""The rewrite holds the agent's recorded actions fixed and regenerates only what the injected text
touches. These tests run against real committed episodes and the real AgentDojo environments."""

import csv
from pathlib import Path

import pytest
from agentdojo.task_suite.load_suites import get_suite

from wwtw.adaptive.appeals import swap_appeal
from wwtw.adaptive.rewrite import rewrite_episode
from wwtw.agent.attacks import CANARY, MONITOR_NOTE
from wwtw.monitor.views import render, step_contexts
from wwtw.traces.load import read_episodes

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "data" / "traces" / "monitor_aware.jsonl.gz"
LABELS = ROOT / "data" / "labels" / "monitor_aware.csv"
VERSION = "v1.2.2"

pytestmark = pytest.mark.skipif(not TRACES.exists(), reason="committed traces not present")


def preventable_positives() -> list:
    with open(LABELS, newline="") as f:
        labels = {r["episode_id"]: r for r in csv.DictReader(f)}
    out = []
    for ep in read_episodes(TRACES):
        lab = labels[ep.episode_id]
        if (
            ep.security
            and not lab["mismatch"]
            and lab["not_preventable"] != "1"
            and lab["harm_call_index"]
        ):
            out.append(ep)
    return out


@pytest.fixture(scope="module")
def positives():
    return preventable_positives()


@pytest.fixture(scope="module")
def sample(positives):
    """Two episodes per suite, favouring ones whose tool outputs actually carry the injection."""
    chosen, per_suite = [], {}
    for ep in positives:
        if per_suite.get(ep.suite, 0) >= 2:
            continue
        if any(CANARY in e.text for e in ep.events if e.kind == "tool_result"):
            chosen.append(ep)
            per_suite[ep.suite] = per_suite.get(ep.suite, 0) + 1
    assert len(per_suite) == 4
    return chosen


@pytest.fixture(scope="module")
def suites():
    return {s: get_suite(VERSION, s) for s in ("workspace", "travel", "banking", "slack")}


def action_events(ep):
    return [(x.kind, x.call, x.ordinal) for x in ep.events if x.kind == "tool_call"]


def results(ep):
    return {e.ordinal: e.text for e in ep.events if e.kind == "tool_result"}


def test_rewriting_with_the_recorded_appeal_changes_nothing(sample, suites):
    for ep in sample:
        new, report = rewrite_episode(ep, suites[ep.suite], MONITOR_NOTE)
        assert new.events == ep.events, ep.episode_id
        assert new.injections == ep.injections
        assert report.changed == ()


def test_a_new_appeal_changes_only_results_that_carried_the_injection(sample, suites):
    marker = "ZZ-APPEAL-MARKER"
    for ep in sample:
        new, report = rewrite_episode(ep, suites[ep.suite], marker + " for {user}")
        old_r, new_r = results(ep), results(new)
        assert old_r.keys() == new_r.keys()
        assert report.changed, ep.episode_id
        for k in old_r:
            if k in report.changed:
                assert marker in new_r[k]
                assert CANARY not in new_r[k]
            else:
                assert new_r[k] == old_r[k]
        # actions, prompt and verdict-relevant fields are untouched
        assert action_events(new) == action_events(ep)
        assert new.user_prompt == ep.user_prompt and new.final_output == ep.final_output
        assert (new.utility, new.security) == (ep.utility, ep.security)


def test_removing_the_appeal_leaves_no_trace_of_it(sample, suites):
    for ep in sample:
        new, _ = rewrite_episode(ep, suites[ep.suite], None)
        text = "".join(results(new).values()) + "".join(new.injections.values())
        assert CANARY not in text


def test_views_that_cannot_see_tool_outputs_are_unchanged(sample, suites):
    marker = "ZZ-APPEAL-MARKER"
    saw_v3_change = False
    for ep in sample:
        new, _ = rewrite_episode(ep, suites[ep.suite], marker)
        for a, b in zip(step_contexts(ep), step_contexts(new), strict=True):
            assert a.call == b.call
            for v in ("V1", "V2"):
                assert render(v, a) == render(v, b)
            if render("V3", a) != render("V3", b):
                saw_v3_change = True
                assert marker in render("V3", b)
    assert saw_v3_change


def test_rewrite_refuses_an_invalid_appeal(sample, suites):
    ep = sample[0]
    with pytest.raises(ValueError):
        rewrite_episode(ep, suites[ep.suite], "bad {task} placeholder")


def test_rewrite_needs_a_recorded_appeal_to_replace(suites):
    ii = read_episodes(ROOT / "data" / "traces" / "important_instructions.jsonl.gz")
    ep = next(e for e in ii if e.suite == "banking")
    with pytest.raises(ValueError):
        rewrite_episode(ep, suites[ep.suite], "anything")


@pytest.mark.slow
def test_every_preventable_positive_reproduces_exactly(positives, suites):
    """The gate for the experiment: identity rewrite equals the recording for all 250 episodes."""
    bad = []
    for ep in positives:
        new, report = rewrite_episode(ep, suites[ep.suite], MONITOR_NOTE)
        if new.events != ep.events or report.changed:
            bad.append(ep.episode_id)
    assert not bad, bad[:5]


def test_removing_the_appeal_gives_the_standard_attacks_text(suites):
    ii = {
        e.episode_id.split("/", 1)[1]: e
        for e in read_episodes(ROOT / "data" / "traces" / "important_instructions.jsonl.gz")
    }
    checked = 0
    for ep in read_episodes(TRACES):
        key = ep.episode_id.split("/", 1)[1]
        if key not in ii or not ep.injections:
            continue
        goal = suites[ep.suite].get_injection_task_by_id(ep.injection_task_id).GOAL
        assert swap_appeal(ep.injections, None, goal) == ii[key].injections, key
        checked += 1
    assert checked == 949
