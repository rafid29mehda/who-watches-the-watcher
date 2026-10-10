"""Score side-effecting steps offline: every (episode, step, view) for a monitor model."""

import csv
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path

from wwtw.monitor.llm_monitor import LLMMonitor
from wwtw.monitor.rule_monitor import rule_score
from wwtw.monitor.views import StepContext, step_contexts
from wwtw.traces.replay import ReplayResult
from wwtw.traces.schema import Episode

SCORE_FIELDS = [
    "model",
    "attack",
    "episode_id",
    "suite",
    "user_task_id",
    "injection_task_id",
    "call_index",
    "function",
    "view",
    "suspicion",
    "parse_error",
    "evidence_chars",
    "reason",
]


@dataclass(frozen=True)
class Job:
    episode: Episode
    ctx: StepContext
    view: str


def _jobs_for(ep: Episode, views: Sequence[str], upto: int | None) -> list[Job]:
    return [
        Job(ep, ctx, v)
        for ctx in step_contexts(ep)
        if upto is None or ctx.call_index <= upto
        for v in views
    ]


def plan_jobs(
    episodes: Sequence[Episode],
    views: Sequence[str],
    *,
    labels: dict[str, ReplayResult] | None = None,
    scope: str = "all",
    benign_limit: int | None = None,
    seed: int = 0,
) -> list[list[Job]]:
    rng = random.Random(seed)
    benign = sorted((e for e in episodes if e.attack == "none"), key=lambda e: e.episode_id)
    attacked = sorted((e for e in episodes if e.attack != "none"), key=lambda e: e.episode_id)
    if benign_limit is not None and len(benign) > benign_limit:
        benign = rng.sample(benign, benign_limit)
    groups = [_jobs_for(e, views, None) for e in benign]
    if scope == "successful_until_harm":
        if labels is None:
            raise ValueError("scope 'successful_until_harm' needs replay labels")
        ok = [
            e
            for e in attacked
            if e.security
            and labels[e.episode_id].mismatch is None
            and labels[e.episode_id].harm_call_index is not None
            and not labels[e.episode_id].not_preventable
        ]
        by_attack = {}
        for e in ok:
            by_attack.setdefault(e.attack, []).append(e)
        shuffled = [rng.sample(by_attack[a], len(by_attack[a])) for a in sorted(by_attack)]
        order = [e for e in chain.from_iterable(zip_longest(*shuffled)) if e is not None]
        groups += [_jobs_for(e, views, labels[e.episode_id].harm_call_index) for e in order]
    elif scope == "all":
        groups += [_jobs_for(e, views, None) for e in attacked]
    else:
        raise ValueError(f"unknown scope {scope!r}")
    return [g for g in groups if g]


def _row(
    model: str,
    ep: Episode,
    ctx: StepContext,
    view: str,
    suspicion,
    parse_error: bool,
    evidence_chars: int,
    reason: str,
) -> dict:
    return {
        "model": model,
        "attack": ep.attack,
        "episode_id": ep.episode_id,
        "suite": ep.suite,
        "user_task_id": ep.user_task_id,
        "injection_task_id": ep.injection_task_id or "",
        "call_index": ctx.call_index,
        "function": ctx.call.function,
        "view": view,
        "suspicion": "" if suspicion is None else suspicion,
        "parse_error": int(parse_error),
        "evidence_chars": evidence_chars,
        "reason": reason[:200],
    }


def _score(job: Job, monitor: LLMMonitor, model: str) -> dict:
    verdict, n = monitor.score(job.view, job.ctx)
    return _row(
        model,
        job.episode,
        job.ctx,
        job.view,
        verdict.suspicion,
        verdict.parse_error,
        n,
        verdict.reason,
    )


def score_jobs(
    groups: Sequence[Sequence[Job]],
    monitor: LLMMonitor,
    model_name: str,
    *,
    workers: int = 1,
    max_minutes: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[int, int], None] | None = None,
    between_groups: Callable[[], None] | None = None,
) -> tuple[list[dict], bool]:
    total = sum(len(g) for g in groups)
    if workers > 1 and max_minutes is None:
        flat = [j for g in groups for j in g]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(lambda j: _score(j, monitor, model_name), flat))
        if progress is not None:
            progress(len(rows), total)
        return rows, False
    rows: list[dict] = []
    start = clock()
    for group in groups:
        if max_minutes is not None and clock() - start > max_minutes * 60:
            return rows, True
        if between_groups is not None:
            between_groups()
        rows.extend(_score(j, monitor, model_name) for j in group)
        if progress is not None:
            progress(len(rows), total)
    return rows, False


def rule_rows(episodes: Sequence[Episode]) -> list[dict]:
    return [
        _row("rule", ep, ctx, "rule", rule_score(ctx), False, 0, "")
        for ep in episodes
        for ctx in step_contexts(ep)
    ]


def retest_sample(
    groups: Sequence[Sequence[Job]], fraction: float, seed: int = 0
) -> list[list[Job]]:
    flat = [j for g in groups for j in g]
    k = round(fraction * len(flat))
    picked = random.Random(seed).sample(range(len(flat)), k)
    return [[flat[i]] for i in sorted(picked)]


def write_scores_csv(rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def read_scores_csv(path: str | Path) -> list[dict]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["call_index"] = int(r["call_index"])
            r["parse_error"] = int(r["parse_error"])
            r["evidence_chars"] = int(r["evidence_chars"])
            r["suspicion"] = int(r["suspicion"]) if r["suspicion"] != "" else ""
            out.append(r)
    return out
