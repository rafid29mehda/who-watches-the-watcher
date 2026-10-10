"""Command-line entry point: `wwtw <command>`."""

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import click
import numpy as np
from agentdojo.task_suite.load_suites import get_suite
from dotenv import load_dotenv

from wwtw.agent.pipeline import build_agent_pipeline
from wwtw.agent.run import SweepSpec, run_sweep, select_injection_tasks
from wwtw.analysis.metrics import score_index, threshold_for_fpr
from wwtw.budget import BudgetLedger
from wwtw.config import MonitorModelCfg, load_config
from wwtw.llm_client import (
    CachedChatClient,
    OllamaMonitorBackend,
    OpenAIMonitorBackend,
    ResponseCache,
)
from wwtw.monitor.gate import MonitoredToolsExecutor
from wwtw.monitor.llm_monitor import LLMMonitor
from wwtw.monitor.run import (
    plan_jobs,
    read_scores_csv,
    retest_sample,
    rule_rows,
    score_jobs,
    write_scores_csv,
)
from wwtw.resources import doctor_report, guard, wait_for_ram
from wwtw.traces.load import load_run_dir, read_episodes, write_episodes
from wwtw.traces.replay import read_replay_csv, replay_all, write_replay_csv

ATTACKS_ALL = ["none", "important_instructions", "monitor_aware"]
log = logging.getLogger("wwtw")


class Ctx:
    def __init__(
        self, cfg_path="configs/experiment.yaml", agent_transport=None, backend_factory=None
    ):
        self.cfg = load_config(cfg_path)
        b = self.cfg.budget
        self.ledger = BudgetLedger(b.ledger_path, b.cap_usd, b.prices)
        self.cache = ResponseCache(self.cfg.paths.cache / "llm.sqlite")
        self.agent_transport = agent_transport
        self.backend_factory = backend_factory
        self._ollama_proc: subprocess.Popen | None = None

    def pipeline(self, namespace="main", executor=None, name=None):
        client = CachedChatClient(
            cache=self.cache,
            ledger=self.ledger,
            namespace=namespace,
            default_params={"temperature": self.cfg.agent.temperature},
            transport=self.agent_transport,
        )
        return build_agent_pipeline(client, self.cfg.agent.model, executor, name)

    def backend(self, m: MonitorModelCfg, namespace="main"):
        if self.backend_factory is not None:
            return self.backend_factory(m, namespace)
        if m.backend == "openai":
            client = CachedChatClient(cache=self.cache, ledger=self.ledger, namespace=namespace)
            return OpenAIMonitorBackend(
                client, m.model, m.reasoning_effort, m.json_mode, m.max_output_tokens
            )
        return OllamaMonitorBackend(self.cache, m.model, num_ctx=m.num_ctx, namespace=namespace)

    def export_ledger(self):
        self.ledger.export_csv(self.cfg.paths.data / "cost_ledger.csv")

    @property
    def run_dir(self) -> Path:
        return self.cfg.paths.runs / self.cfg.agent.model


def _progress(label):
    def cb(done, total):
        if done == total or done % 10 == 0:
            log.info("%s: %d/%d", label, done, total)

    return cb


def do_agents(ctx: Ctx, suites=None, attacks=None, user_task_ids=None, injection_task_ids=None):
    cfg = ctx.cfg
    guard(cfg.resources)
    pipe = ctx.pipeline()
    for s in suites or cfg.agent.suites:
        users = (user_task_ids or {}).get(s)
        suite = get_suite(cfg.agent.benchmark_version, s)
        inj = (injection_task_ids or {}).get(s) or select_injection_tasks(
            suite, cfg.agent.max_injection_tasks_per_suite, cfg.seed
        )
        for attack in [None, *(attacks or cfg.agent.attacks)]:
            label = f"{s}/{attack or 'none'}"
            run_sweep(
                pipe,
                SweepSpec(s, attack, users, inj),
                cfg.paths.runs,
                benchmark_version=cfg.agent.benchmark_version,
                workers=cfg.agent.workers,
                resources=cfg.resources,
                progress=_progress(label),
            )
            ctx.export_ledger()
            log.info("%s done; total spend $%.4f", label, ctx.ledger.spent())
    export_traces(ctx)


def export_traces(ctx: Ctx):
    for attack in ATTACKS_ALL:
        eps = load_run_dir(ctx.run_dir, attack)
        if eps:
            write_episodes(eps, ctx.cfg.paths.data / "traces" / f"{attack}.jsonl.gz")


def _episodes(ctx: Ctx, attacks=ATTACKS_ALL):
    out = []
    for a in attacks:
        p = ctx.cfg.paths.data / "traces" / f"{a}.jsonl.gz"
        if p.exists():
            out += read_episodes(p)
    return out


def do_replay(ctx: Ctx, max_mismatch_rate: float = 0.01):
    total = bad = retried = 0
    for a in ATTACKS_ALL:
        p = ctx.cfg.paths.data / "traces" / f"{a}.jsonl.gz"
        if not p.exists():
            continue
        results = replay_all(read_episodes(p), ctx.cfg.agent.benchmark_version)
        write_replay_csv(results, ctx.cfg.paths.data / "labels" / f"{a}.csv")
        total += len(results)
        for r in results:
            if r.mismatch is None:
                continue
            if r.mismatch.startswith("retried:"):
                retried += 1  # AgentDojo re-ran the episode; the log holds only the last attempt
            else:
                bad += 1
    rate = bad / total if total else 0.0
    log.info(
        "replay: %d episodes, %d unexplained mismatches (%.2f%%), %d retried episodes excluded",
        total,
        bad,
        100 * rate,
        retried,
    )
    if rate > max_mismatch_rate:
        raise click.ClickException(
            f"replay mismatch rate {rate:.2%} exceeds {max_mismatch_rate:.0%}"
        )
    return rate


def _labels(ctx: Ctx):
    labels = {}
    for a in ATTACKS_ALL:
        p = ctx.cfg.paths.data / "labels" / f"{a}.csv"
        if p.exists():
            labels.update(read_replay_csv(p))
    return labels


def _ensure_ollama(ctx: Ctx, backend: OllamaMonitorBackend) -> None:
    if backend.is_available():
        return
    exe = shutil.which("ollama")
    if exe is None:
        raise click.ClickException("Ollama is not installed or not on PATH.")
    env = {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1"}
    ctx._ollama_proc = subprocess.Popen(
        [exe, "serve"],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(1)
        if backend.is_available():
            return
    raise click.ClickException("Ollama server did not start within 30 s.")


def do_monitors(ctx: Ctx, models=None, retest=True):
    cfg = ctx.cfg
    guard(cfg.resources)
    episodes = _episodes(ctx)
    labels = _labels(ctx)
    by_attack = {a: [e for e in episodes if e.attack == a] for a in ATTACKS_ALL}
    for a, eps in by_attack.items():
        if eps:
            write_scores_csv(rule_rows(eps), cfg.paths.data / "scores" / "rule" / f"{a}.csv")
    for m in cfg.monitor.models:
        if models and m.name not in models:
            continue
        backend = ctx.backend(m)
        is_local = m.backend == "ollama" and ctx.backend_factory is None
        if is_local:
            _ensure_ollama(ctx, backend)
            if not backend.has_model():
                raise click.ClickException(f"Ollama model {m.model} is not installed.")
        try:
            monitor = LLMMonitor(
                backend,
                long_threshold=cfg.monitor.long_leaf_chars,
                max_evidence_chars=m.max_evidence_chars,
            )
            groups = plan_jobs(
                episodes,
                m.views,
                labels=labels,
                scope=m.attacked_scope,
                benign_limit=m.benign_episode_limit,
                seed=cfg.seed,
            )
            # The local model's own memory counts against "available", so it gets a lower floor.
            floor = (
                cfg.resources.min_available_ram_gb_local
                if m.backend == "ollama"
                else cfg.resources.min_available_ram_gb
            )
            rows, truncated = score_jobs(
                groups,
                monitor,
                m.name,
                workers=m.workers,
                max_minutes=m.max_minutes,
                progress=_progress(m.name),
                between_groups=lambda floor=floor: wait_for_ram(
                    floor, cfg.resources.ram_wait_seconds
                ),
            )
            if truncated:
                log.info("%s: time cap reached; scored %d rows", m.name, len(rows))
            for a in ATTACKS_ALL:
                sub = [r for r in rows if r["attack"] == a]
                if sub:
                    write_scores_csv(sub, cfg.paths.data / "scores" / m.name / f"{a}.csv")
            if retest and m.backend == "openai":
                rmon = LLMMonitor(
                    ctx.backend(m, namespace="retest"), long_threshold=cfg.monitor.long_leaf_chars
                )
                rrows, _ = score_jobs(
                    retest_sample(groups, cfg.monitor.retest_fraction, cfg.seed),
                    rmon,
                    m.name,
                    workers=m.workers,
                )
                write_scores_csv(rrows, cfg.paths.data / "scores" / m.name / "retest.csv")
        finally:
            if is_local:
                backend.unload()
                if ctx._ollama_proc is not None:
                    ctx._ollama_proc.terminate()
                    ctx._ollama_proc = None
            ctx.export_ledger()


def do_validate_online(ctx: Ctx):
    """Run the online suite live with the V4 gate at the offline 5%-FPR threshold."""
    cfg = ctx.cfg
    on = cfg.online
    m = cfg.monitor.get(on.monitor)
    rows = read_scores_csv(cfg.paths.data / "scores" / m.name / "none.csv")
    benign = score_index(rows)[(m.name, on.view)]
    bmax = [max(s.values()) for s in benign.values()]
    bmax += [-1] * (len(read_episodes(cfg.paths.data / "traces" / "none.jsonl.gz")) - len(bmax))
    tau = threshold_for_fpr(np.asarray(bmax), on.fpr)
    monitor = LLMMonitor(
        ctx.backend(m, namespace="online"), long_threshold=cfg.monitor.long_leaf_chars
    )

    def scorer(step_ctx):
        verdict, _ = monitor.score(on.view, step_ctx)
        return verdict.suspicion

    gate = MonitoredToolsExecutor(on.suite, scorer, tau)
    name = f"{cfg.agent.model}-gated"
    pipe = ctx.pipeline(namespace="online", executor=gate, name=name)
    online_runs = cfg.paths.runs / "online"
    for attack in [None, on.attack]:
        run_sweep(
            pipe,
            SweepSpec(on.suite, attack),
            online_runs,
            benchmark_version=cfg.agent.benchmark_version,
            workers=cfg.agent.workers,
            resources=cfg.resources,
            progress=_progress(f"online/{attack or 'none'}"),
        )
    for attack in ["none", on.attack]:
        write_episodes(
            load_run_dir(online_runs / name, attack),
            cfg.paths.data / "online" / f"{attack}.jsonl.gz",
        )
    meta = {"threshold": tau, "suite": on.suite, "view": on.view, "monitor": m.name, "fpr": on.fpr}
    (cfg.paths.data / "online" / "meta.json").write_text(json.dumps(meta, indent=2))
    ctx.export_ledger()


@click.group()
@click.option("--config", default="configs/experiment.yaml", show_default=True)
@click.pass_context
def main(click_ctx, config):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    click_ctx.obj = config


def _ctx(click_ctx) -> Ctx:
    return Ctx(click_ctx.obj)


@main.command()
def doctor():
    click.echo(doctor_report(Path(".")))


@main.command()
@click.pass_context
def budget(click_ctx):
    c = _ctx(click_ctx)
    click.echo(f"spent ${c.ledger.spent():.4f} of ${c.cfg.budget.cap_usd:.2f}")


@main.command()
@click.pass_context
def pilot(click_ctx):
    from wwtw.pilot import run_pilot

    click.echo(json.dumps(run_pilot(_ctx(click_ctx)), indent=2))


@main.command()
@click.option("--suite", "suites", multiple=True)
@click.option("--attack", "attacks", multiple=True)
@click.pass_context
def agents(click_ctx, suites, attacks):
    do_agents(_ctx(click_ctx), list(suites) or None, list(attacks) or None)


@main.command()
@click.pass_context
def replay(click_ctx):
    do_replay(_ctx(click_ctx))


@main.command()
@click.option("--model", "models", multiple=True)
@click.option("--no-retest", is_flag=True)
@click.pass_context
def monitors(click_ctx, models, no_retest):
    do_monitors(_ctx(click_ctx), list(models) or None, retest=not no_retest)


@main.command("validate-online")
@click.pass_context
def validate_online(click_ctx):
    do_validate_online(_ctx(click_ctx))


@main.command()
@click.pass_context
def analyze(click_ctx):
    from wwtw.analysis.summary import analyze as run

    s = run(_ctx(click_ctx).cfg)
    click.echo(json.dumps(s["hypotheses"], indent=2))
