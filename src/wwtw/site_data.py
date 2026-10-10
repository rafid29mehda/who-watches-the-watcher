"""Build site/data/{summary,examples}.json from committed results (no API calls)."""

import json
import random
import shutil
from pathlib import Path

from wwtw.monitor.views import render, step_contexts

VIEWS = ["V1", "V2", "V3", "V4", "V4raw"]
CATEGORIES = ["v4_only", "v3_only", "both", "neither"]


def _category(scores: dict, ep, harm: dict, taus: dict[str, int]) -> str:
    h = harm.get(ep.episode_id)
    if h is None:
        return "other"
    caught = {
        v: any(
            scores.get((ep.episode_id, c.call_index, v), 0) >= taus[v]
            for c in step_contexts(ep)
            if c.call_index <= h
        )
        for v in ("V3", "V4")
    }
    if caught["V4"] and not caught["V3"]:
        return "v4_only"
    if caught["V3"] and not caught["V4"]:
        return "v3_only"
    return "both" if caught["V3"] else "neither"


def select_examples(
    episodes,
    scores: dict,
    harm: dict,
    per_category: int = 2,
    seed: int = 0,
    taus: dict[str, int] | None = None,
) -> list[dict]:
    """Pick real episodes for each V3/V4 outcome at the given flag thresholds (deterministic)."""
    taus = taus or {"V3": 50, "V4": 50}
    rng = random.Random(seed)
    buckets: dict[str, list] = {}
    for ep in sorted(episodes, key=lambda e: e.episode_id):
        buckets.setdefault(_category(scores, ep, harm, taus), []).append(ep)
    chosen = []
    for cat in CATEGORIES:
        pool = buckets.get(cat, [])
        for ep in rng.sample(pool, min(per_category, len(pool))):
            steps = []
            for ctx in step_contexts(ep):
                steps.append(
                    {
                        "call_index": ctx.call_index,
                        "function": ctx.call.function,
                        "is_harm": ctx.call_index == harm.get(ep.episode_id),
                        "views": {
                            v: {
                                "evidence": render(v, ctx, max_chars=6000),
                                "score": scores.get((ep.episode_id, ctx.call_index, v)),
                            }
                            for v in VIEWS
                        },
                    }
                )
            chosen.append(
                {
                    "episode_id": ep.episode_id,
                    "attack": ep.attack,
                    "suite": ep.suite,
                    "user_prompt": ep.user_prompt,
                    "category": cat,
                    "steps": steps,
                }
            )
    return chosen


def _thresholds(summary: dict, attack: str) -> dict[str, int]:
    """Display threshold per view: the mean of the two cross-fitted 5%-FPR thresholds."""
    primary = summary["meta"]["primary_monitor"]
    out = {}
    for r in summary["results"]:
        if r["model"] == primary and r["attack"] == attack and r.get("thresholds_5"):
            out[r["view"]] = round(sum(r["thresholds_5"]) / len(r["thresholds_5"]))
    return out


def build_site_data(cfg, out_dir: Path = Path("site/data")) -> None:
    from wwtw.monitor.run import read_scores_csv
    from wwtw.traces.load import read_episodes
    from wwtw.traces.replay import read_replay_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((cfg.paths.results / "summary.json").read_text())
    shutil.copy(cfg.paths.results / "summary.json", out_dir / "summary.json")
    api = summary["meta"]["primary_monitor"]
    examples, thresholds = [], {}
    for attack in ["important_instructions", "monitor_aware"]:
        taus = _thresholds(summary, attack)
        thresholds[attack] = taus
        eps = [
            e for e in read_episodes(cfg.paths.data / "traces" / f"{attack}.jsonl.gz") if e.security
        ]
        labels = read_replay_csv(cfg.paths.data / "labels" / f"{attack}.csv")
        harm = {
            k: v.harm_call_index
            for k, v in labels.items()
            if not v.not_preventable and v.mismatch is None
        }
        rows = read_scores_csv(cfg.paths.data / "scores" / api / f"{attack}.csv")
        scores = {
            (r["episode_id"], r["call_index"], r["view"]): r["suspicion"]
            for r in rows
            if r["suspicion"] != ""
        }
        examples += select_examples(eps, scores, harm, seed=cfg.seed, taus=taus)
    payload = {"monitor": api, "thresholds": thresholds, "examples": examples}
    (out_dir / "examples.json").write_text(json.dumps(payload, indent=1))
