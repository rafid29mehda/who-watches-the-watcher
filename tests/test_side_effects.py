from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites

from wwtw.traces.side_effects import BOOKKEEPING, READ_ONLY, SIDE_EFFECTING, is_side_effecting

SUITES = get_suites("v1.2.2")


def test_every_tool_classified_exactly_once():
    for name, suite in SUITES.items():
        tools = {t.name for t in suite.tools}
        assert tools == SIDE_EFFECTING[name] | READ_ONLY[name], name
        assert not (SIDE_EFFECTING[name] & READ_ONLY[name]), name


def test_ground_truth_mutations_are_side_effecting():
    for name, suite in SUITES.items():
        allowed = SIDE_EFFECTING[name] | BOOKKEEPING.get(name, frozenset())
        for task in list(suite.user_tasks.values()) + list(suite.injection_tasks.values()):
            env = suite.load_and_inject_default_environment({})
            if hasattr(task, "PROMPT"):
                env = task.init_environment(env)
            rt = FunctionsRuntime(suite.tools)
            for call in task.ground_truth(env.model_copy(deep=True)):
                before = env.model_dump()
                rt.run_function(env, call.function, call.args)
                if env.model_dump() != before:
                    assert call.function in allowed, f"{name}/{task.ID}: {call.function}"


def test_lookup():
    assert is_side_effecting("banking", "send_money")
    assert not is_side_effecting("banking", "get_balance")
    assert not is_side_effecting("workspace", "get_unread_emails")
    assert is_side_effecting("slack", "get_webpage")
    assert not is_side_effecting("banking", "not_a_tool")
