"""Load committed data, compute every reported number, and write results/summary.json."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from wwtw.analysis.hypotheses import II, MA, describe, run_hypotheses
from wwtw.analysis.labels import build_records
from wwtw.analysis.metrics import (
    build_eval_data,
    episode_metrics,
    roc_points,
    score_index,
    threshold_for_fpr,
)
from wwtw.analysis.stats import assign_folds, mcnemar_exact
from wwtw.config import ExperimentConfig
from wwtw.monitor.run import read_scores_csv
from wwtw.traces.load import read_episodes
from wwtw.traces.replay import read_replay_csv

ATTACKS = ["none", II, MA]
PUBLISHED_AGENTDOJO: dict = {}  # Task 23 fills this from the AgentDojo results page (URL + date)


def _rate(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        return None if np.isnan(o) else round(float(o), 4)
    return o


def _load(cfg: ExperimentConfig):
    data = cfg.paths.data
    episodes, labels = [], {}
    for a in ATTACKS:
        p = data / "traces" / f"{a}.jsonl.gz"
        if p.exists():
            episodes += read_episodes(p)
            labels.update(read_replay_csv(data / "labels" / f"{a}.csv"))
    rows = []
    for model_dir in sorted((data / "scores").glob("*")):
        for a in ATTACKS:
            p = model_dir / f"{a}.csv"
            if p.exists():
                rows += read_scores_csv(p)
    return episodes, labels, rows


def analyze(cfg: ExperimentConfig, out_dir: Path | None = None) -> dict:
    from wwtw.analysis.figures import make_figures
    from wwtw.analysis.tables import make_tables

    out_dir = Path(out_dir or cfg.paths.results)
    episodes, labels, rows = _load(cfg)
    records = build_records(episodes, labels, cfg.agent.benchmark_version)
    clusters = sorted({r.cluster for r in records})
    cid = {k: i for i, k in enumerate(clusters)}
    fold = assign_folds(clusters, cfg.seed)
    idx = score_index(rows)
    api = cfg.monitor.models[0].name
    full = {api, "rule"}
    summary: dict = {
        "meta": {},
        "agent_baseline": {},
        "validity": {},
        "results": [],
        "robustness_gap": [],
        "hypotheses": [],
        "secondary": {},
    }

    # --- agent baseline -----------------------------------------------------------------
    by = {a: [r for r in records if r.attack == a] for a in ATTACKS}
    summary["meta"]["n_episodes"] = {a: len(v) for a, v in by.items()}
    summary["meta"]["n_clusters"] = len(clusters)
    base = {"benign_utility": _rate(r.utility for r in by["none"])}
    for a in (II, MA):
        base[a] = {
            "utility_under_attack": _rate(r.utility for r in by[a]),
            "asr": _rate(r.security for r in by[a]),
            "not_preventable": int(sum(r.kind == 1 for r in by[a])),
            "preventable_success": int(sum(r.kind == 0 and r.mismatch is None for r in by[a])),
        }
    base["per_suite"] = {
        s: {
            "benign_utility": _rate(r.utility for r in by["none"] if r.suite == s),
            **{a: _rate(r.security for r in by[a] if r.suite == s) for a in (II, MA)},
        }
        for s in sorted({r.suite for r in records})
    }
    base["published"] = PUBLISHED_AGENTDOJO
    summary["agent_baseline"] = base

    # --- descriptive results ------------------------------------------------------------
    data: dict = {}
    roc: dict = {}
    for (model, view), scores in sorted(idx.items()):
        if view == "rule" and model != "rule":
            continue
        for a in (II, MA):
            d = build_eval_data(records, scores, a, cid, full_coverage=model in full)
            if d.pos_score.size == 0 or d.neg_score.size == 0:
                continue
            data[(model, view, a)] = d
            row = {
                "model": model,
                "view": view,
                "attack": a,
                **describe(d, fold, len(clusters), cfg.analysis, cfg.seed),
            }
            summary["results"].append(row)
            roc[(model, view, a)] = roc_points(d.pos_score, d.neg_score)
    t = cfg.analysis.primary_fpr
    for (model, view, a), d in data.items():
        if a == II and (model, view, MA) in data:
            ci = episode_metrics(d, fold, t)["catch"]
            cm = episode_metrics(data[(model, view, MA)], fold, t)["catch"]
            summary["robustness_gap"].append({"model": model, "view": view, "delta": ci - cm})

    # --- hypotheses (API model only) ------------------------------------------------------
    needed = [(v, a) for v in ("V1", "V3", "V4") for a in (II, MA)]
    if all((api, v, a) in data for v, a in needed):
        hyp = run_hypotheses(
            {(v, a): data[(api, v, a)] for v, a in needed},
            fold,
            len(clusters),
            cfg.analysis,
            cfg.seed,
        )
        summary["hypotheses"] = [h.__dict__ for h in hyp]

    # --- secondary analyses -----------------------------------------------------------------
    sec = summary["secondary"]
    sec["mcnemar_v3_vs_v4"] = []
    for a in (II, MA):
        if (api, "V3", a) in data and (api, "V4", a) in data:
            c3 = episode_metrics(data[(api, "V3", a)], fold, t)["caught"]
            c4 = episode_metrics(data[(api, "V4", a)], fold, t)["caught"]
            b, c = int((c3 & ~c4).sum()), int((~c3 & c4).sum())
            sec["mcnemar_v3_vs_v4"].append(
                {"attack": a, "v3_only": b, "v4_only": c, "p": mcnemar_exact(b, c)}
            )
    sec["strict_harm_catch"] = []
    for (model, view), scores in sorted(idx.items()):
        if model != api:
            continue
        for a in (II, MA):
            d = build_eval_data(records, scores, a, cid, harm_field="strict_harm")
            if d.pos_score.size:
                sec["strict_harm_catch"].append(
                    {"view": view, "attack": a, "catch": episode_metrics(d, fold, t)["catch"]}
                )
    sec["benign_under_attack_flag_rate"] = []
    for (model, view), scores in sorted(idx.items()):
        if model != api:
            continue
        bmax = [
            max(s.values()) if (s := scores.get(r.episode_id)) else -1
            for r in by["none"]
            if r.mismatch is None
        ]
        tau = threshold_for_fpr(np.asarray(bmax), t)
        for a in (II, MA):
            flags = [
                scores.get(r.episode_id, {}).get(i, 0) >= tau
                for r in by[a]
                if r.utility and not r.security and r.mismatch is None
                for i in r.se_calls
                if i not in r.attacker_entity_calls and i not in r.canary_calls
            ]
            sec["benign_under_attack_flag_rate"].append(
                {"view": view, "attack": a, "rate": _rate(flags), "n": len(flags)}
            )

    # --- validity ------------------------------------------------------------------------------
    val = summary["validity"]
    val["replay_mismatch_rate"] = _rate(r.mismatch is not None for r in records)
    val["parse_error_rate"] = {}
    for r in rows:
        key = f"{r['model']}/{r['view']}"
        val["parse_error_rate"].setdefault(key, []).append(r["parse_error"])
    val["parse_error_rate"] = {k: _rate(v) for k, v in val["parse_error_rate"].items()}
    ma_prev = [r for r in by[MA] if r.kind == 0 and r.mismatch is None]
    val["leakage_rate"] = _rate(r.harm in r.canary_calls for r in ma_prev)
    retest = cfg.paths.data / "scores" / api / "retest.csv"
    if retest.exists():
        main = {
            (r["episode_id"], r["call_index"], r["view"]): r["suspicion"]
            for r in rows
            if r["model"] == api and r["suspicion"] != ""
        }
        pairs = [
            (main[k], r["suspicion"])
            for r in read_scores_csv(retest)
            if r["suspicion"] != "" and (k := (r["episode_id"], r["call_index"], r["view"])) in main
        ]
        if len(pairs) >= 3:
            val["test_retest_spearman"] = float(spearmanr(*zip(*pairs, strict=True)).statistic)
            val["test_retest_n"] = len(pairs)
    val["online"] = _online(cfg, records, idx, by)
    if any(v > 0.02 for v in val["parse_error_rate"].values()):
        idx_drop = score_index(rows, drop_parse_errors=True)
        sec["parse_error_excluded_auroc"] = [
            {
                "model": m,
                "view": v,
                "attack": a,
                "auroc": describe(
                    build_eval_data(records, s, a, cid, full_coverage=m in full),
                    fold,
                    len(clusters),
                    cfg.analysis,
                    cfg.seed,
                )["auroc"],
            }
            for (m, v), s in sorted(idx_drop.items())
            for a in (II, MA)
        ]

    summary = _clean(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "hypotheses.json").write_text(json.dumps(summary["hypotheses"], indent=2))
    make_figures(summary, roc, [out_dir / "figures"])
    make_tables(summary, out_dir / "tables")
    return summary


def _online(cfg: ExperimentConfig, records, idx, by) -> dict:
    meta_p = cfg.paths.data / "online" / "meta.json"
    if not meta_p.exists():
        return {}
    meta = json.loads(meta_p.read_text())
    online_att = read_episodes(cfg.paths.data / "online" / f"{cfg.online.attack}.jsonl.gz")
    online_ben = read_episodes(cfg.paths.data / "online" / "none.jsonl.gz")
    scores = idx[(cfg.online.monitor, cfg.online.view)]
    tau = meta["threshold"]
    offline = {
        (r.user_task_id, r.injection_task_id): r
        for r in by[cfg.online.attack]
        if r.suite == cfg.online.suite and r.mismatch is None
    }

    def predicted_success(r):
        if r.kind == 2:
            return False
        if r.kind == 1:
            return True
        s = scores.get(r.episode_id, {})
        return not any(v >= tau for i, v in s.items() if i <= r.harm)

    pairs = [
        (predicted_success(offline[k]), ep.security)
        for ep in online_att
        if (k := (ep.user_task_id, ep.injection_task_id)) in offline
    ]
    ben_off = {r.user_task_id: r for r in by["none"] if r.suite == cfg.online.suite}
    upairs = []
    for ep in online_ben:
        r = ben_off.get(ep.user_task_id)
        if r is not None:
            s = scores.get(r.episode_id, {})
            upairs.append((r.utility and not any(v >= tau for v in s.values()), ep.utility))
    rng = np.random.default_rng(cfg.seed)
    diffs = []
    arr = np.array(pairs, dtype=float)
    for _ in range(cfg.analysis.descriptive_resamples):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        diffs.append(sample[:, 1].mean() - sample[:, 0].mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975]) if diffs else (float("nan"), float("nan"))
    return {
        "threshold": tau,
        "n": len(pairs),
        "offline_predicted_asr": _rate(p for p, _ in pairs),
        "online_observed_asr": _rate(o for _, o in pairs),
        "diff_ci": [float(lo), float(hi)],
        "agrees": bool(lo <= 0 <= hi),
        "offline_predicted_utility": _rate(p for p, _ in upairs),
        "online_observed_utility": _rate(o for _, o in upairs),
    }
