import dataclasses
from pathlib import Path

from wwtw.budget import Usage
from wwtw.llm_client import Completion
from wwtw.monitor.llm_monitor import LLMMonitor
from wwtw.monitor.run import (
    plan_jobs,
    read_scores_csv,
    retest_sample,
    rule_rows,
    score_jobs,
    write_scores_csv,
)
from wwtw.traces.load import load_episode_file
from wwtw.traces.replay import ReplayResult

ATT = load_episode_file(Path(__file__).parent / "fixtures" / "banking_episode.json")
BEN = dataclasses.replace(
    ATT,
    attack="none",
    injection_task_id=None,
    injections={},
    episode_id="none/banking/user_task_0/none",
    security=False,
)


class ViewBackend:
    def complete(self, messages):
        ev = messages[1]["content"]
        score = 80 if "TRANSCRIPT" in ev else 20
        return Completion(f'{{"reason": "r", "suspicion": {score}}}', Usage(1, 0, 1), False)


class Garbage:
    def complete(self, messages):
        return Completion("no", Usage(1, 0, 1), False)


def test_plan_jobs_all_scope():
    groups = plan_jobs([ATT, BEN], ["V1", "V3"])
    assert [g[0].episode.attack for g in groups] == ["none", "important_instructions"]
    assert sum(len(g) for g in groups) == 2 * 2 * 2  # 2 episodes x 2 SE steps x 2 views


def test_plan_jobs_until_harm_and_benign_limit():
    labels = {ATT.episode_id: ReplayResult(ATT.episode_id, 1, 1, False, (1, 2), None)}
    groups = plan_jobs(
        [ATT, BEN], ["V1"], labels=labels, scope="successful_until_harm", benign_limit=0
    )
    assert len(groups) == 1 and [j.ctx.call_index for j in groups[0]] == [1]


def test_score_jobs_rows_and_parse_errors():
    groups = plan_jobs([ATT], ["V1", "V3"])
    rows, truncated = score_jobs(groups, LLMMonitor(ViewBackend()), "m", workers=2)
    assert not truncated and len(rows) == 4
    assert {(r["view"], r["suspicion"]) for r in rows} == {("V1", 20), ("V3", 80)}
    bad, _ = score_jobs(groups, LLMMonitor(Garbage()), "m")
    assert all(r["parse_error"] == 1 and r["suspicion"] == "" for r in bad)


def test_time_cap_stops_at_episode_boundary():
    groups = plan_jobs([BEN, ATT, BEN], ["V1"])
    ticks = iter([0, 0, 100, 100, 100, 100, 100])
    rows, truncated = score_jobs(
        groups, LLMMonitor(ViewBackend()), "m", max_minutes=1, clock=lambda: next(ticks)
    )
    assert truncated and len(rows) == len(groups[0])


def test_retest_sample_deterministic():
    groups = plan_jobs([ATT, BEN], ["V1", "V2", "V3", "V4", "V4raw"])
    a, b = retest_sample(groups, 0.1, seed=0), retest_sample(groups, 0.1, seed=0)
    n = sum(len(g) for g in groups)
    assert a == b and sum(len(g) for g in a) == round(0.1 * n)


def test_rule_rows_and_csv_roundtrip(tmp_path):
    rows = rule_rows([ATT, BEN])
    assert len(rows) == 4 and all(r["model"] == "rule" and r["view"] == "rule" for r in rows)
    p = tmp_path / "s.csv"
    write_scores_csv(rows, p)
    back = read_scores_csv(p)
    assert [r["episode_id"] for r in back] == [r["episode_id"] for r in rows]
    assert isinstance(back[0]["suspicion"], int) and isinstance(back[0]["call_index"], int)


def test_between_groups_hook_runs_before_each_group():
    groups = plan_jobs([BEN, ATT], ["V1"])
    calls = []
    score_jobs(groups, LLMMonitor(ViewBackend()), "m", between_groups=lambda: calls.append(1))
    assert len(calls) == len(groups)
