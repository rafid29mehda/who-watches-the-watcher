import json

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suite

from tests.fakes import GroundTruthAgent
from wwtw.agent.attacks import CANARY
from wwtw.agent.pipeline import build_agent_pipeline
from wwtw.agent.run import SweepSpec, run_sweep, select_injection_tasks
from wwtw.budget import BudgetLedger
from wwtw.config import Price
from wwtw.llm_client import CachedChatClient, ResponseCache
from wwtw.traces.load import load_run_dir
from wwtw.traces.replay import replay_all

MODEL = "gpt-4o-mini-2024-07-18"
SUITE = get_suite("v1.2.2", "banking")
PRICES = {MODEL: Price(input=0.15, cached_input=0.075, output=0.60)}


def _pipeline(tmp_path, fake):
    client = CachedChatClient(
        cache=ResponseCache(tmp_path / "c.sqlite"),
        ledger=BudgetLedger(tmp_path / "l.sqlite", 1.0, PRICES),
        transport=fake,
        default_params={"temperature": 0.0},
    )
    return build_agent_pipeline(client, MODEL)


def test_monitor_aware_attack(tmp_path):
    pipe = _pipeline(tmp_path, GroundTruthAgent(SUITE))
    attack = load_attack("monitor_aware", SUITE, pipe)
    ut, it = (
        SUITE.get_user_task_by_id("user_task_0"),
        SUITE.get_injection_task_by_id("injection_task_0"),
    )
    for text in attack.attack(ut, it).values():
        assert CANARY in text and it.GOAL in text
        assert text.index(CANARY) < text.index("</INFORMATION>")


def test_temperature_zero_sent(tmp_path):
    fake = GroundTruthAgent(SUITE)
    run_sweep(
        _pipeline(tmp_path, fake),
        SweepSpec("banking", None, ["user_task_0"]),
        tmp_path / "runs",
        benchmark_version="v1.2.2",
        workers=1,
    )
    assert fake.calls and all(c["temperature"] == 0.0 for c in fake.calls)


def test_sweep_writes_isolated_replayable_logs(tmp_path):
    fake = GroundTruthAgent(SUITE)
    users = ["user_task_0", "user_task_1", "user_task_2", "user_task_3"]
    spec = SweepSpec(
        "banking", "important_instructions", users, ["injection_task_0", "injection_task_1"]
    )
    res = run_sweep(
        _pipeline(tmp_path, fake), spec, tmp_path / "runs", benchmark_version="v1.2.2", workers=4
    )
    assert len(res) == 8
    run_dir = tmp_path / "runs" / MODEL
    for f in run_dir.glob("banking/user_task_*/important_instructions/*.json"):
        d = json.loads(f.read_text())
        user_msg = next(m for m in d["messages"] if m["role"] == "user")
        assert (
            user_msg["content"][0]["content"] == SUITE.get_user_task_by_id(d["user_task_id"]).PROMPT
        )
        assert f.stem == d["injection_task_id"]
    episodes = load_run_dir(run_dir, "important_instructions")
    assert len(episodes) == 8 and any(e.security for e in episodes)
    assert all(r.mismatch is None for r in replay_all(episodes, "v1.2.2"))


def test_rerun_makes_no_new_calls(tmp_path):
    fake = GroundTruthAgent(SUITE)
    spec = SweepSpec("banking", "important_instructions", ["user_task_0"], ["injection_task_0"])
    pipe = _pipeline(tmp_path, fake)
    run_sweep(pipe, spec, tmp_path / "runs", benchmark_version="v1.2.2", workers=1)
    n = len(fake.calls)
    run_sweep(pipe, spec, tmp_path / "runs", benchmark_version="v1.2.2", workers=1)
    assert len(fake.calls) == n


def test_select_injection_tasks():
    assert select_injection_tasks(SUITE, None, 0) == sorted(
        SUITE.injection_tasks, key=lambda t: int(t.split("_")[-1])
    )
    a, b = select_injection_tasks(SUITE, 4, 0), select_injection_tasks(SUITE, 4, 0)
    assert a == b and len(a) == 4
