"""Parallel, resumable AgentDojo sweeps (one job per episode)."""

import logging
import random
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import LOGGER_STACK, NullLogger
from agentdojo.task_suite.load_suites import get_suite

import wwtw.agent.attacks  # noqa: F401  (registers the monitor_aware attack)
from wwtw.config import ResourcesCfg
from wwtw.resources import guard

log = logging.getLogger(__name__)


class QuietLogger(NullLogger):
    """Gives AgentDojo's TraceLogger a log directory without console output."""

    def __init__(self, logdir: Path):
        self._logdir = str(logdir)

    def __enter__(self):
        self.messages = []
        self.logdir = self._logdir
        LOGGER_STACK.get().append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        LOGGER_STACK.get().pop()
        return False


def task_num(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def select_injection_tasks(suite, k: int | None, seed: int) -> list[str]:
    ids = sorted(suite.injection_tasks, key=task_num)
    if k is None or k >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, k), key=task_num)


@dataclass(frozen=True)
class SweepSpec:
    suite: str
    attack: str | None  # None = benign (no injections)
    user_task_ids: list[str] | None = None
    injection_task_ids: list[str] | None = None


def run_sweep(
    pipeline,
    spec: SweepSpec,
    logdir: Path,
    *,
    benchmark_version: str,
    workers: int = 4,
    resources: ResourcesCfg | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[tuple[str, str | None], tuple[bool, bool]]:
    suite = get_suite(benchmark_version, spec.suite)
    user_ids = spec.user_task_ids or sorted(suite.user_tasks, key=task_num)
    user_tasks = [suite.get_user_task_by_id(t) for t in user_ids]
    attack = load_attack(spec.attack, suite, pipeline) if spec.attack else None
    if attack is None:
        jobs = [(ut, None) for ut in user_tasks]
    else:
        inj_ids = spec.injection_task_ids or select_injection_tasks(suite, None, 0)
        for ut in user_tasks:
            attack.get_injection_candidates(ut)  # warm the lru_cache before threads start
        jobs = [(ut, i) for ut in user_tasks for i in inj_ids]
    stop = threading.Event()

    def job(ut, inj_id):
        if stop.is_set():
            return {}
        LOGGER_STACK.set([])  # AgentDojo's ContextVar default list is shared across threads
        if resources is not None:
            guard(resources)
        with QuietLogger(logdir):
            if inj_id is None:
                u, s = run_task_without_injection_tasks(
                    suite, pipeline, ut, logdir, False, benchmark_version
                )
                return {(ut.ID, None): (u, s)}
            util, sec = run_task_with_injection_tasks(
                suite, pipeline, ut, attack, logdir, False, [inj_id], benchmark_version
            )
            return {k: (util[k], sec[k]) for k in util}

    results: dict = {}
    ex = ThreadPoolExecutor(max_workers=workers)
    futures = [ex.submit(job, ut, i) for ut, i in jobs]
    try:
        for f in as_completed(futures):
            results.update(f.result())
            if progress is not None:
                progress(len(results), len(jobs))
    except BaseException:
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return results
