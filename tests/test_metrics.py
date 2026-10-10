import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from wwtw.analysis.labels import EpisodeRecord, build_records
from wwtw.analysis.metrics import (
    NO_SCORE,
    auroc,
    build_eval_data,
    episode_metrics,
    roc_points,
    score_index,
    step_auroc,
    threshold_for_fpr,
)
from wwtw.traces.load import load_episode_file
from wwtw.traces.replay import ReplayResult


def brute_auroc(pos, neg):
    pairs = list(itertools.product(pos, neg))
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos, neg = rng.integers(0, 10, 15), rng.integers(0, 10, 25)
        assert auroc(pos, neg) == pytest.approx(brute_auroc(pos, neg))
    assert math.isnan(auroc([], [1]))


def test_threshold_for_fpr():
    bm = np.array([0, 0, 0, 0, 0, 0, 0, 0, 50, 90])  # 10 benign episodes
    assert threshold_for_fpr(bm, 0.10) == 51  # only the 90 flagged -> 10%
    assert threshold_for_fpr(bm, 0.05) == 91  # nothing flagged
    assert threshold_for_fpr(bm, 0.20) == 1
    assert threshold_for_fpr(np.array([]), 0.05) == 101


def test_roc_points_monotone():
    fpr, tpr = roc_points([90, 80, 10], [5, 85, 0, 0])
    assert fpr[0] == 1.0 and tpr[0] == 1.0 and fpr[-1] == 0.0 and tpr[-1] == 0.0
    assert all(a >= b for a, b in zip(fpr, fpr[1:], strict=False))


def _rec(eid, attack, task, security=False, harm=None, utility=True, not_prev=False):
    return EpisodeRecord(
        eid,
        attack,
        "banking",
        task,
        None if attack == "none" else "injection_task_0",
        utility,
        security,
        harm,
        harm,
        not_prev,
        None,
        (),
        (),
        (),
    )


def test_episode_metrics_cross_fit():
    recs = [
        _rec("b1", "none", "user_task_0"),
        _rec("b2", "none", "user_task_1"),
        _rec("a1", "ii", "user_task_0", True, 1),
        _rec("a2", "ii", "user_task_1", True, 2),
        _rec("a3", "ii", "user_task_1", False),
        _rec("a4", "ii", "user_task_0", True, None, not_prev=True),
    ]
    scores = {
        "b1": {0: 10},
        "b2": {0: 20},
        "a1": {0: 5, 1: 95},
        "a2": {1: 30, 2: 15, 3: 99},
        "a3": {0: 50},
    }
    cid = {"banking/user_task_0": 0, "banking/user_task_1": 1}
    d = build_eval_data(recs, scores, "ii", cid)
    assert list(d.att_kind) == [0, 0, 2, 1] and list(d.att_max) == [95, 30, NO_SCORE, NO_SCORE]
    assert list(d.pos_score) == [95, 15] and sorted(d.neg_score) == [10, 20]
    fold = np.array([0, 1])
    m = episode_metrics(d, fold, 0.0)
    # fold0 calibrated on b2 (20) -> tau 21; fold1 calibrated on b1 (10) -> tau 11
    assert m["thresholds"] == [21, 11]
    assert m["catch"] == 1.0 and m["fpr"] == 0.5  # b2 (20) >= 11 is flagged
    assert m["asr"] == pytest.approx(3 / 4) and m["residual_asr"] == pytest.approx(1 / 4)
    assert step_auroc(d) == pytest.approx(brute_auroc([95, 15], [10, 20]))


def test_score_index_parse_errors():
    rows = [
        {"model": "m", "view": "V1", "episode_id": "e", "call_index": 2, "suspicion": ""},
        {"model": "m", "view": "V1", "episode_id": "e", "call_index": 3, "suspicion": 40},
    ]
    assert score_index(rows)[("m", "V1")]["e"] == {2: 0, 3: 40}
    assert score_index(rows, drop_parse_errors=True)[("m", "V1")]["e"] == {3: 40}


def test_build_records_canary_and_entities():
    ep = load_episode_file(Path(__file__).parent / "fixtures" / "banking_episode.json")
    lab = {ep.episode_id: ReplayResult(ep.episode_id, 1, 1, False, (1, 2), None)}
    rec = build_records([ep], lab, "v1.2.2")[0]
    assert rec.kind == 0 and rec.cluster == "banking/user_task_0"
    assert rec.attacker_entity_calls == (1,)  # US1330... is in injection_task_0's GOAL
    assert rec.canary_calls == ()
