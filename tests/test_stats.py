import numpy as np
import pytest
from scipy.stats import binomtest

from wwtw.analysis.hypotheses import describe, run_hypotheses
from wwtw.analysis.metrics import EvalData
from wwtw.analysis.stats import (
    assign_folds,
    bootstrap_p,
    draws,
    holm,
    mcnemar_exact,
    percentile_ci,
)
from wwtw.config import AnalysisCfg

CFG = AnalysisCfg(
    fpr_targets=[0.05, 0.10],
    primary_fpr=0.05,
    bootstrap_resamples=300,
    descriptive_resamples=200,
    noninferiority_margin=0.05,
)


def test_assign_folds_stratified_and_deterministic():
    keys = [f"banking/user_task_{i}" for i in range(6)] + [f"slack/user_task_{i}" for i in range(4)]
    f1, f2 = assign_folds(keys, 0), assign_folds(keys, 0)
    assert (f1 == f2).all()
    assert f1[:6].sum() == 3 and f1[6:].sum() == 2


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_mcnemar_matches_scipy():
    assert mcnemar_exact(8, 2) == pytest.approx(binomtest(2, 10, 0.5).pvalue)
    assert mcnemar_exact(0, 0) == 1.0


def test_bootstrap_helpers():
    dist = np.linspace(0.1, 0.3, 1001)
    lo, hi = percentile_ci(dist)
    assert 0.10 < lo < 0.11 and 0.29 < hi < 0.30
    assert bootstrap_p(dist, 0.0) == pytest.approx(2 / 1002)
    assert bootstrap_p(dist, 0.2) == pytest.approx(1.0, abs=0.01)
    assert draws(5, 4, 0).shape == (4, 5)


def _synthetic(view_strength, n_tasks=40, seed=0):
    """Positives scored ~view_strength, negatives ~20; one attacked + one benign episode per task."""
    rng = np.random.default_rng(seed)
    tasks = np.arange(n_tasks)
    pos = np.clip(rng.normal(view_strength, 10, n_tasks), 0, 100).astype(int)
    neg = np.clip(rng.normal(20, 10, n_tasks), 0, 100).astype(int)
    return EvalData(
        tasks,
        neg,
        np.ones(n_tasks, bool),
        tasks,
        np.zeros(n_tasks, int),
        pos,
        tasks,
        pos,
        tasks,
        neg,
        True,
    )


def test_hypotheses_detect_planted_effects():
    keys = [f"banking/user_task_{i}" for i in range(40)]
    fold = assign_folds(keys, 0)
    data = {
        ("V1", "important_instructions"): _synthetic(30),
        ("V3", "important_instructions"): _synthetic(80, seed=1),
        ("V4", "important_instructions"): _synthetic(80, seed=2),
        ("V3", "monitor_aware"): _synthetic(35, seed=3),
        ("V4", "monitor_aware"): _synthetic(78, seed=4),
    }
    res = {r.name: r for r in run_hypotheses(data, fold, 40, CFG)}
    assert set(res) == {"H1", "H2", "H3a", "H3b"}
    assert res["H1"].supported and res["H2"].supported and res["H3b"].supported
    assert res["H3a"].supported
    assert all(r.p_holm >= r.p_value for r in res.values())


def test_describe_has_cis():
    keys = [f"banking/user_task_{i}" for i in range(40)]
    out = describe(_synthetic(80), assign_folds(keys, 0), 40, CFG)
    assert out["auroc_ci"][0] <= out["auroc"] <= out["auroc_ci"][1]
    assert set(out) >= {
        "catch_5",
        "catch_5_ci",
        "catch_10",
        "residual_asr_5",
        "utility_retained_5",
        "fpr_5",
    }
