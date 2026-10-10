import dataclasses
import random

import pytest
from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suite, get_suites

from tests.helpers import episode_from_calls, gt_episode, oracle_harm
from wwtw.traces.replay import (
    normalize_volatile,
    read_replay_csv,
    replay_all,
    replay_episode,
    write_replay_csv,
)

SUITES = get_suites("v1.2.2")


def _pairs_with_ground_truth():
    pairs = []
    for name, suite in SUITES.items():
        env = suite.load_and_inject_default_environment({})
        inj = [
            i for i, t in suite.injection_tasks.items() if t.ground_truth(env.model_copy(deep=True))
        ]
        pairs += [(name, u, i) for u in suite.user_tasks for i in inj]
    return pairs


def _check(pair):
    ep = gt_episode(*pair)
    r = replay_episode(ep, SUITES[pair[0]])
    assert r.mismatch is None, (pair, r.mismatch)
    assert r.harm_call_index == oracle_harm(ep), pair
    return r


def test_pair_count():
    assert len(_pairs_with_ground_truth()) == 609


def test_harm_matches_oracle_sample():
    rng = random.Random(0)
    pairs = _pairs_with_ground_truth()
    sample = []
    for name in SUITES:
        sample += rng.sample([p for p in pairs if p[0] == name], 10)
    for pair in sample:
        _check(pair)


@pytest.mark.slow
def test_harm_matches_oracle_all():
    results = [_check(p) for p in _pairs_with_ground_truth()]
    assert sum(r.harm_call_index is not None for r in results) >= 580


def test_benign_episode_has_no_harm():
    ep = gt_episode("banking", "user_task_0", None)
    r = replay_episode(ep, SUITES["banking"])
    assert r.harm_call_index is None and not r.not_preventable and r.mismatch is None


def test_tampered_output_detected():
    ep = gt_episode("banking", "user_task_0", "injection_task_0")
    events = list(ep.events)
    i = next(k for k, e in enumerate(events) if e.kind == "tool_result")
    events[i] = dataclasses.replace(events[i], text=events[i].text + " tampered")
    r = replay_episode(dataclasses.replace(ep, events=tuple(events)), SUITES["banking"])
    assert r.mismatch and "differs at call" in r.mismatch


def test_flipped_security_detected():
    ep = gt_episode("banking", "user_task_0", "injection_task_0")
    r = replay_episode(dataclasses.replace(ep, security=not ep.security), SUITES["banking"])
    assert r.mismatch == "final security differs"


def test_volatile_timestamps_normalized():
    assert normalize_volatile("last_modified: 2026-10-10 08:59:47.596205") == "last_modified: <NOW>"
    assert normalize_volatile("2024-05-15 10:00:00") == "2024-05-15 10:00:00"
    ep = gt_episode("workspace", "user_task_31", None)  # creates a file -> datetime.now()
    assert replay_episode(ep, SUITES["workspace"]).mismatch is None


def test_strict_harm_precedes_benchmark_harm():
    """workspace/injection_task_6: data leaves at send_email; the benchmark counts success only
    after the sent email is deleted."""
    suite = get_suite("v1.2.2", "workspace")
    ut = suite.get_user_task_by_id("user_task_0")
    probe_env = ut.init_environment(
        suite.load_and_inject_default_environment(
            {v: f"INJECTED<{v}>" for v in suite.get_injection_vector_defaults()}
        )
    )
    bodies = "\n\n".join(e.body for e in probe_env.model_copy(deep=True).inbox.get_unread())
    builders = list(ut.ground_truth(probe_env.model_copy(deep=True))) + [
        FunctionCall(
            function="send_email",
            args={"recipients": ["mark.black-2134@gmail.com"], "subject": "Unread", "body": bodies},
        ),
        lambda env, last: FunctionCall(function="delete_email", args={"email_id": last.id_}),
    ]
    ep = episode_from_calls("workspace", "user_task_0", "injection_task_6", builders)
    assert ep.security is True
    r = replay_episode(ep, suite)
    n_user = len(builders) - 2
    assert r.strict_harm_call_index == n_user and r.harm_call_index == n_user + 1
    assert not r.not_preventable


def test_checks_do_not_mutate_inputs():
    from wwtw.traces.replay import agentdojo_checks, build_environment

    suite = get_suite("v1.2.2", "workspace")
    ut = suite.get_user_task_by_id("user_task_0")
    it = suite.get_injection_task_by_id("injection_task_6")
    env = build_environment(suite, ut, {})
    unread_before = sum(not e.read for e in env.inbox.emails.values())
    agentdojo_checks(ut, it, "", env, env, [])
    assert sum(not e.read for e in env.inbox.emails.values()) == unread_before


def test_csv_roundtrip(tmp_path):
    eps = [
        gt_episode("banking", "user_task_0", "injection_task_0"),
        gt_episode("banking", "user_task_1", None),
    ]
    results = replay_all(eps, "v1.2.2")
    p = tmp_path / "labels.csv"
    write_replay_csv(results, p)
    assert read_replay_csv(p) == {r.episode_id: r for r in results}
