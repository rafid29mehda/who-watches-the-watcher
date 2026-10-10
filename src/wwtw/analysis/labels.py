"""Join episodes with replay labels into per-episode analysis records."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from agentdojo.task_suite.load_suites import get_suite

from wwtw.agent.attacks import CANARY
from wwtw.entities import attacker_entities, normalize_entity
from wwtw.traces.replay import ReplayResult
from wwtw.traces.schema import Episode
from wwtw.traces.side_effects import is_side_effecting


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    attack: str
    suite: str
    user_task_id: str
    injection_task_id: str | None
    utility: bool
    security: bool
    harm: int | None
    strict_harm: int | None
    not_preventable: bool
    mismatch: str | None
    se_calls: tuple[int, ...]
    attacker_entity_calls: tuple[int, ...]
    canary_calls: tuple[int, ...]

    @property
    def cluster(self) -> str:
        return f"{self.suite}/{self.user_task_id}"

    @property
    def kind(self) -> int:
        """0 preventable success, 1 not-preventable success, 2 failed attack."""
        if not self.security:
            return 2
        if self.not_preventable or self.harm is None:
            return 1
        return 0


def _blob(args: dict) -> str:
    return normalize_entity(json.dumps(args, ensure_ascii=False, default=str))


def build_records(
    episodes: Sequence[Episode], labels: dict[str, ReplayResult], benchmark_version: str
) -> list[EpisodeRecord]:
    goals: dict[tuple[str, str], frozenset[str]] = {}
    out = []
    for ep in episodes:
        lab = labels[ep.episode_id]
        ents: frozenset[str] = frozenset()
        if ep.injection_task_id:
            key = (ep.suite, ep.injection_task_id)
            if key not in goals:
                task = get_suite(benchmark_version, ep.suite).get_injection_task_by_id(
                    ep.injection_task_id
                )
                goals[key] = attacker_entities(task.GOAL)
            ents = goals[key]
        ent_calls, canary_calls = [], []
        for ex in ep.executed_calls():
            if not is_side_effecting(ep.suite, ex.call.function):
                continue
            blob = _blob(ex.call.args)
            if ents and any(e in blob for e in ents):
                ent_calls.append(ex.ordinal)
            if CANARY in blob:
                canary_calls.append(ex.ordinal)
        out.append(
            EpisodeRecord(
                ep.episode_id,
                ep.attack,
                ep.suite,
                ep.user_task_id,
                ep.injection_task_id,
                ep.utility,
                ep.security,
                lab.harm_call_index,
                lab.strict_harm_call_index,
                lab.not_preventable,
                lab.mismatch,
                lab.side_effecting_calls,
                tuple(ent_calls),
                tuple(canary_calls),
            )
        )
    return out
