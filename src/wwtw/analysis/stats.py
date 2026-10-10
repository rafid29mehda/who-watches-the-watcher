"""Cluster bootstrap (user tasks as clusters), cross-fitting folds, Holm and McNemar."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from wwtw.analysis.metrics import EvalData


def assign_folds(cluster_keys: Sequence[str], seed: int) -> np.ndarray:
    """Two folds, stratified by suite (the prefix before '/'), deterministic for a seed."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(cluster_keys), dtype=int)
    for suite in sorted({k.split("/")[0] for k in cluster_keys}):
        idx = np.array([i for i, k in enumerate(cluster_keys) if k.split("/")[0] == suite])
        perm = rng.permutation(idx)
        fold[perm[1::2]] = 1
    return fold


def _members(task_arr: np.ndarray, n: int) -> list[np.ndarray]:
    return [np.flatnonzero(task_arr == k) for k in range(n)]


@dataclass(frozen=True)
class Members:
    ben: list
    att: list
    pos: list
    neg: list

    @staticmethod
    def of(d: EvalData, n_clusters: int) -> "Members":
        return Members(
            _members(d.benign_task, n_clusters),
            _members(d.att_task, n_clusters),
            _members(d.pos_task, n_clusters),
            _members(d.neg_task, n_clusters),
        )


def take(members: list, draw) -> np.ndarray:
    parts = [members[k] for k in draw]
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def draws(n_clusters: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_clusters, size=(B, n_clusters))


def percentile_ci(dist, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(dist, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def bootstrap_p(dist, null: float) -> float:
    x = np.asarray(dist, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    lo = int((x <= null).sum())
    hi = int((x >= null).sum())
    return float(min(1.0, 2 * (min(lo, hi) + 1) / (x.size + 1)))


def holm(pvals: Sequence[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [float(a) for a in adj]


def mcnemar_exact(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)
