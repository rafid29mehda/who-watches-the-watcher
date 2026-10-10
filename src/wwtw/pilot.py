"""Pilot: a small stratified run that measures real costs, then projects the full study."""

import json
import random

from agentdojo.task_suite.load_suites import get_suite

from wwtw.agent.run import select_injection_tasks, task_num
from wwtw.config import ExperimentConfig

VIEWS_ALL = ["V1", "V2", "V3", "V4", "V4raw"]


def pilot_selection(cfg: ExperimentConfig) -> dict:
    rng = random.Random(cfg.seed)
    sel = {}
    for s in cfg.agent.suites:
        suite = get_suite(cfg.agent.benchmark_version, s)
        users = sorted(suite.user_tasks, key=task_num)
        sel[s] = {
            "benign": sorted(rng.sample(users, cfg.pilot.benign_per_suite), key=task_num),
            "attacked_users": sorted(
                rng.sample(users, cfg.pilot.user_tasks_per_suite), key=task_num
            ),
            "injection_tasks": select_injection_tasks(
                suite, cfg.pilot.injection_tasks_per_suite, cfg.seed
            ),
        }
    return sel


def _counts(cfg: ExperimentConfig, k: int | None) -> dict:
    out = {}
    for s in cfg.agent.suites:
        suite = get_suite(cfg.agent.benchmark_version, s)
        n_inj = len(suite.injection_tasks) if k is None else min(k, len(suite.injection_tasks))
        out[s] = {"benign": len(suite.user_tasks), "attacked": len(suite.user_tasks) * n_inj}
    return out


def project_total(
    cfg: ExperimentConfig, measured: dict, k: int | None = None, api_views: list[str] | None = None
) -> dict:
    api_views = api_views or cfg.monitor.models[0].views
    counts = _counts(cfg, k)
    n_attacks = len(cfg.agent.attacks)
    agent = monitors = 0.0
    for s, c in counts.items():
        cost = measured["agent_cost_per_episode"][s]
        steps = measured["se_steps_per_episode"][s]
        agent += c["benign"] * cost["benign"] + n_attacks * c["attacked"] * cost["attacked"]
        n_steps = c["benign"] * steps["benign"] + n_attacks * c["attacked"] * steps["attacked"]
        monitors += n_steps * sum(measured["monitor_cost_per_job"][v] for v in api_views)
    monitors *= 1 + cfg.monitor.retest_fraction
    on = cfg.online.suite
    oc = counts.get(on, {"benign": 0, "attacked": 0})
    online = (
        oc["benign"] * measured["agent_cost_per_episode"][on]["benign"]
        + oc["attacked"] * measured["agent_cost_per_episode"][on]["attacked"]
        + (oc["benign"] + oc["attacked"])
        * measured["se_steps_per_episode"][on]["attacked"]
        * measured["monitor_cost_per_job"][cfg.online.view]
    )
    total = agent + monitors + online + measured["spent_so_far"]
    return {
        "agent": agent,
        "monitors": monitors,
        "online": online,
        "spent_so_far": measured["spent_so_far"],
        "total": total,
    }


def apply_decision_rule(cfg: ExperimentConfig, measured: dict) -> dict:
    limit = cfg.pilot.max_projected_usd
    p0 = project_total(cfg, measured)
    if p0["total"] <= limit:
        return {
            "decision": "none",
            "max_injection_tasks_per_suite": None,
            "api_views": cfg.monitor.models[0].views,
            "projection": p0,
        }
    p1 = project_total(cfg, measured, k=4)
    if p1["total"] <= limit:
        return {
            "decision": "subsample_injection_tasks",
            "max_injection_tasks_per_suite": 4,
            "api_views": cfg.monitor.models[0].views,
            "projection": p1,
        }
    views = [v for v in cfg.monitor.models[0].views if v != "V4raw"]
    p2 = project_total(cfg, measured, k=4, api_views=views)
    decision = "subsample_and_drop_v4raw" if p2["total"] <= limit else "over_budget"
    return {
        "decision": decision,
        "max_injection_tasks_per_suite": 4,
        "api_views": views,
        "projection": p2,
    }


def run_pilot(ctx) -> dict:
    from wwtw.agent.run import SweepSpec, run_sweep
    from wwtw.cli import export_traces
    from wwtw.monitor.llm_monitor import LLMMonitor
    from wwtw.monitor.run import plan_jobs, score_jobs
    from wwtw.monitor.views import step_contexts
    from wwtw.traces.load import load_run_dir

    cfg = ctx.cfg
    sel = pilot_selection(cfg)
    pipe = ctx.pipeline()
    measured = {
        "agent_cost_per_episode": {},
        "se_steps_per_episode": {},
        "monitor_cost_per_job": {},
    }
    for s, chosen in sel.items():
        measured["agent_cost_per_episode"][s] = {}
        before = ctx.ledger.spent()
        run_sweep(
            pipe,
            SweepSpec(s, None, chosen["benign"]),
            cfg.paths.runs,
            benchmark_version=cfg.agent.benchmark_version,
            workers=cfg.agent.workers,
            resources=cfg.resources,
        )
        mid = ctx.ledger.spent()
        run_sweep(
            pipe,
            SweepSpec(s, cfg.pilot.attack, chosen["attacked_users"], chosen["injection_tasks"]),
            cfg.paths.runs,
            benchmark_version=cfg.agent.benchmark_version,
            workers=cfg.agent.workers,
            resources=cfg.resources,
        )
        after = ctx.ledger.spent()
        n_att = len(chosen["attacked_users"]) * len(chosen["injection_tasks"])
        measured["agent_cost_per_episode"][s] = {
            "benign": (mid - before) / len(chosen["benign"]),
            "attacked": (after - mid) / n_att,
        }
    export_traces(ctx)
    benign = [e for e in load_run_dir(ctx.run_dir, "none")]
    attacked = [e for e in load_run_dir(ctx.run_dir, cfg.pilot.attack)]
    for s in sel:
        b = [
            len(step_contexts(e))
            for e in benign
            if e.suite == s and e.user_task_id in sel[s]["benign"]
        ]
        a = [len(step_contexts(e)) for e in attacked if e.suite == s]
        measured["se_steps_per_episode"][s] = {
            "benign": sum(b) / max(len(b), 1),
            "attacked": sum(a) / max(len(a), 1),
        }
    api = cfg.monitor.models[0]
    monitor = LLMMonitor(ctx.backend(api), long_threshold=cfg.monitor.long_leaf_chars)
    pilot_eps = [e for e in benign + attacked]
    for v in api.views:
        groups = plan_jobs(pilot_eps, [v])
        n = sum(len(g) for g in groups)
        before = ctx.ledger.spent()
        score_jobs(groups, monitor, api.name, workers=api.workers)
        measured["monitor_cost_per_job"][v] = (ctx.ledger.spent() - before) / max(n, 1)
    measured["spent_so_far"] = ctx.ledger.spent()
    decision = apply_decision_rule(cfg, measured)
    out = {"selection": sel, "measured": measured, **decision}
    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    (cfg.paths.results / "pilot_projection.json").write_text(json.dumps(out, indent=2))
    ctx.export_ledger()
    return out
