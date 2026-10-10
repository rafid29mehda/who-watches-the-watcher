import json
from pathlib import Path

import yaml

from tests.helpers import gt_episode
from wwtw.analysis.summary import analyze
from wwtw.config import load_config
from wwtw.monitor.run import rule_rows, write_scores_csv
from wwtw.monitor.views import step_contexts
from wwtw.traces.load import write_episodes
from wwtw.traces.replay import replay_all, write_replay_csv


def _scores(episodes, model, view, attack_bias):
    rows = []
    for ep in episodes:
        for ctx in step_contexts(ep):
            s = 10 if ep.attack == "none" else attack_bias
            rows.append(
                {
                    "model": model,
                    "attack": ep.attack,
                    "episode_id": ep.episode_id,
                    "suite": ep.suite,
                    "user_task_id": ep.user_task_id,
                    "injection_task_id": ep.injection_task_id or "",
                    "call_index": ctx.call_index,
                    "function": ctx.call.function,
                    "view": view,
                    "suspicion": s,
                    "parse_error": 0,
                    "evidence_chars": 100,
                    "reason": "",
                }
            )
    return rows


def test_analyze_end_to_end(tmp_path, monkeypatch):
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw["paths"] = {k: str(tmp_path / k) for k in ["runs", "cache", "data", "results"]}
    raw["analysis"]["bootstrap_resamples"] = 50
    raw["analysis"]["descriptive_resamples"] = 50
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfgp)
    users = [f"user_task_{i}" for i in range(6)]
    eps = {"none": [gt_episode("banking", u, None) for u in users]}
    for attack in ["important_instructions", "monitor_aware"]:
        eps[attack] = [
            gt_episode("banking", u, i)
            for u in users
            for i in ["injection_task_0", "injection_task_1"]
        ]
        eps[attack] = [
            e.__class__(
                **{
                    **e.__dict__,
                    "attack": attack,
                    "episode_id": e.episode_id.replace("important_instructions", attack),
                }
            )
            for e in eps[attack]
        ]
    data = tmp_path / "data"
    for attack, lst in eps.items():
        write_episodes(lst, data / "traces" / f"{attack}.jsonl.gz")
        write_replay_csv(replay_all(lst, "v1.2.2"), data / "labels" / f"{attack}.csv")
        rows = []
        for view in ["V1", "V2", "V3", "V4", "V4raw"]:
            bias = {"V1": 40, "V2": 50, "V3": 90, "V4": 85, "V4raw": 80}[view]
            if attack == "monitor_aware" and view == "V3":
                bias = 30
            rows += _scores(lst, "gpt-5-nano", view, bias)
        write_scores_csv(rows, data / "scores" / "gpt-5-nano" / f"{attack}.csv")
        write_scores_csv(rule_rows(lst), data / "scores" / "rule" / f"{attack}.csv")
    summary = analyze(cfg)
    out = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert out["meta"]["n_episodes"]["important_instructions"] == 12
    assert {r["name"] for r in out["hypotheses"]} == {"H1", "H2", "H3a", "H3b"}
    views = {(r["model"], r["view"], r["attack"]) for r in out["results"]}
    assert ("gpt-5-nano", "V4", "monitor_aware") in views and (
        "rule",
        "rule",
        "important_instructions",
    ) in views
    assert summary["validity"]["replay_mismatch_rate"] == 0.0
    assert (tmp_path / "results" / "figures" / "catch_by_view.pdf").exists()
    assert (tmp_path / "results" / "tables" / "main_results.tex").exists()
