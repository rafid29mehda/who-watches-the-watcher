import json
from pathlib import Path

import yaml
from agentdojo.task_suite.load_suites import get_suite

from tests.fakes import GroundTruthAgent
from wwtw.analysis.summary import analyze
from wwtw.budget import Usage
from wwtw.cli import Ctx, do_agents, do_monitors, do_replay
from wwtw.llm_client import Completion
from wwtw.pilot import apply_decision_rule, project_total


class FakeBackend:
    def complete(self, messages):
        ev = messages[1]["content"]
        score = 90 if "US133000000121212121212" in ev else 15
        return Completion(f'{{"reason": "r", "suspicion": {score}}}', Usage(5, 0, 5), False)


def _cfg(tmp_path):
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw["paths"] = {k: str(tmp_path / k) for k in ["runs", "cache", "data", "results"]}
    raw["budget"]["ledger_path"] = str(tmp_path / "ledger.sqlite")
    raw["agent"]["suites"] = ["banking"]
    raw["agent"]["workers"] = 2
    raw["analysis"]["bootstrap_resamples"] = 40
    raw["analysis"]["descriptive_resamples"] = 40
    raw["monitor"]["models"] = raw["monitor"]["models"][:1]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_pipeline_end_to_end_with_fakes(tmp_path):
    ctx = Ctx(
        _cfg(tmp_path),
        agent_transport=GroundTruthAgent(get_suite("v1.2.2", "banking")),
        backend_factory=lambda model_cfg, ns: FakeBackend(),
    )
    users = {"banking": [f"user_task_{i}" for i in range(4)]}
    inj = {"banking": ["injection_task_0", "injection_task_1"]}
    do_agents(ctx, user_task_ids=users, injection_task_ids=inj)
    assert (tmp_path / "data/traces/monitor_aware.jsonl.gz").exists()
    do_replay(ctx)
    do_monitors(ctx, retest=True)
    assert (tmp_path / "data/scores/gpt-5-nano/important_instructions.csv").exists()
    assert (tmp_path / "data/scores/rule/none.csv").exists()
    summary = analyze(ctx.cfg)
    assert summary["meta"]["n_episodes"]["important_instructions"] == 8
    assert json.loads((tmp_path / "results/summary.json").read_text())["results"]
    assert (tmp_path / "data/cost_ledger.csv").exists()


def test_projection_and_decision_rule(tmp_path):
    ctx = Ctx(_cfg(tmp_path))
    measured = {
        "agent_cost_per_episode": {"banking": {"benign": 0.002, "attacked": 0.003}},
        "se_steps_per_episode": {"banking": {"benign": 1.2, "attacked": 1.5}},
        "monitor_cost_per_job": {
            "V1": 0.0001,
            "V2": 0.0001,
            "V3": 0.0004,
            "V4": 0.0002,
            "V4raw": 0.0002,
        },
        "spent_so_far": 0.3,
    }
    proj = project_total(ctx.cfg, measured)
    assert proj["total"] > 0.3 and set(proj) >= {"agent", "monitors", "online", "total"}
    d = apply_decision_rule(ctx.cfg, measured)
    assert d["decision"] == "none" and d["max_injection_tasks_per_suite"] is None
    expensive = dict(measured, agent_cost_per_episode={"banking": {"benign": 1.0, "attacked": 1.0}})
    d2 = apply_decision_rule(ctx.cfg, expensive)
    assert d2["decision"] in {
        "subsample_injection_tasks",
        "subsample_and_drop_v4raw",
        "over_budget",
    }
