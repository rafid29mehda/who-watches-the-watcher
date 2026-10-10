"""Typed experiment configuration loaded from configs/experiment.yaml."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator

VIEWS = ("V1", "V2", "V3", "V4", "V4raw")


class Price(BaseModel):
    input: float  # USD per 1M uncached input tokens
    cached_input: float
    output: float


class AgentCfg(BaseModel):
    model: str
    temperature: float = 0.0
    benchmark_version: str
    suites: list[str]
    attacks: list[str]
    max_injection_tasks_per_suite: int | None = None
    workers: int = 4


class PilotCfg(BaseModel):
    benign_per_suite: int
    user_tasks_per_suite: int
    injection_tasks_per_suite: int
    attack: str
    max_projected_usd: float


class MonitorModelCfg(BaseModel):
    name: str
    backend: Literal["openai", "ollama"]
    model: str
    views: list[str]
    workers: int = 1
    reasoning_effort: str | None = None
    json_mode: bool = False
    max_output_tokens: int = 1000
    max_minutes: float | None = None
    max_evidence_chars: int | None = None
    benign_episode_limit: int | None = None
    attacked_scope: Literal["all", "successful_until_harm"] = "all"
    num_ctx: int = 8192

    @field_validator("views")
    @classmethod
    def _known_views(cls, v: list[str]) -> list[str]:
        unknown = [x for x in v if x not in VIEWS]
        if unknown:
            raise ValueError(f"unknown views {unknown}; allowed: {VIEWS}")
        return v


class MonitorCfg(BaseModel):
    long_leaf_chars: int = 120
    retest_fraction: float = 0.10
    models: list[MonitorModelCfg]

    def get(self, name: str) -> MonitorModelCfg:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"no monitor model named {name!r}")


class OnlineCfg(BaseModel):
    suite: str
    attack: str
    monitor: str
    view: str
    fpr: float


class AnalysisCfg(BaseModel):
    fpr_targets: list[float]
    primary_fpr: float
    bootstrap_resamples: int
    descriptive_resamples: int
    noninferiority_margin: float


class BudgetCfg(BaseModel):
    cap_usd: float
    ledger_path: Path
    prices: dict[str, Price]


class ResourcesCfg(BaseModel):
    min_free_disk_gb: float = 10
    min_available_ram_gb: float = 3
    min_available_ram_gb_local: float = 1.5  # floor while the local model itself is loaded
    ram_wait_seconds: float = 60


class PathsCfg(BaseModel):
    runs: Path = Path("runs")
    cache: Path = Path(".cache")
    data: Path = Path("data")
    results: Path = Path("results")


class ExperimentConfig(BaseModel):
    seed: int = 0
    agent: AgentCfg
    pilot: PilotCfg
    monitor: MonitorCfg
    online: OnlineCfg
    analysis: AnalysisCfg
    budget: BudgetCfg
    resources: ResourcesCfg = ResourcesCfg()
    paths: PathsCfg = PathsCfg()


def load_config(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    with open(path) as f:
        return ExperimentConfig.model_validate(yaml.safe_load(f))
