from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from wwtw.config import VIEWS, ExperimentConfig, load_config


def test_loads_repo_config():
    cfg = load_config("configs/experiment.yaml")
    assert cfg.agent.model == "gpt-4o-mini-2024-07-18"
    assert cfg.agent.benchmark_version == "v1.2.2"
    assert cfg.budget.cap_usd == 20.0
    assert cfg.budget.prices["gpt-5-nano-2025-08-07"].output == 0.40
    names = [m.name for m in cfg.monitor.models]
    assert names == ["gpt-5-nano", "llama3.2-3b"]
    assert cfg.monitor.models[0].views == list(VIEWS)
    assert cfg.monitor.models[1].max_minutes == 60
    assert cfg.online.view == "V4"


def test_rejects_unknown_view():
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw["monitor"]["models"][0]["views"] = ["V9"]
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_rejects_unknown_backend():
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw["monitor"]["models"][0]["backend"] = "azure"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)
