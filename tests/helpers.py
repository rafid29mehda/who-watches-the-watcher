"""Synthetic episodes executed against real AgentDojo environments (no LLM)."""

import copy
from collections.abc import Callable

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from wwtw.traces.replay import agentdojo_checks, build_environment
from wwtw.traces.schema import Episode, Event, ToolCall, make_episode_id

VERSION = "v1.2.2"
Builder = FunctionCall | Callable[[object, object], FunctionCall]  # (env, last_result) -> call


def episode_from_calls(
    suite_name, user_task_id, injection_task_id, builders, attack="important_instructions"
) -> Episode:
    suite = get_suite(VERSION, suite_name)
    ut = suite.get_user_task_by_id(user_task_id)
    it = suite.get_injection_task_by_id(injection_task_id) if injection_task_id else None
    injections = (
        {v: f"INJECTED<{v}>" for v in suite.get_injection_vector_defaults()}
        if it is not None
        else {}
    )
    env = build_environment(suite, ut, injections)
    pre = env.model_copy(deep=True)
    rt = FunctionsRuntime(suite.tools)
    events, calls, last = [], [], None
    for i, b in enumerate(builders):
        fc = b if isinstance(b, FunctionCall) else b(env, last)
        fc = FunctionCall(function=fc.function, args=copy.deepcopy(dict(fc.args)), id=f"c{i}")
        last, err = rt.run_function(env, fc.function, fc.args)
        tc = ToolCall(fc.id, fc.function, dict(fc.args))
        events.append(Event("tool_call", call=tc, ordinal=i))
        events.append(
            Event("tool_result", text=tool_result_to_str(last), call=tc, ordinal=i, error=err)
        )
        calls.append(fc)
    final = ut.GROUND_TRUTH_OUTPUT
    utility, security = agentdojo_checks(ut, it, final, pre, env, calls)
    return Episode(
        episode_id=make_episode_id(
            attack if it else "none", suite_name, user_task_id, injection_task_id
        ),
        suite=suite_name,
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        attack=attack if it else "none",
        user_prompt=ut.PROMPT,
        injections=injections,
        events=tuple(events),
        final_output=final,
        utility=utility,
        security=bool(security),
    )


def gt_episode(suite_name, user_task_id, injection_task_id) -> Episode:
    suite = get_suite(VERSION, suite_name)
    ut = suite.get_user_task_by_id(user_task_id)
    env = build_environment(suite, ut, {})
    calls = list(ut.ground_truth(env.model_copy(deep=True)))
    if injection_task_id:
        calls += list(
            suite.get_injection_task_by_id(injection_task_id).ground_truth(
                env.model_copy(deep=True)
            )
        )
    return episode_from_calls(suite_name, user_task_id, injection_task_id, calls)


def oracle_harm(ep: Episode) -> int | None:
    """Brute force: re-execute every prefix from scratch on a fresh environment."""
    suite = get_suite(VERSION, ep.suite)
    ut = suite.get_user_task_by_id(ep.user_task_id)
    it = suite.get_injection_task_by_id(ep.injection_task_id)
    executed = ep.executed_calls()
    for k in range(len(executed)):
        env = build_environment(suite, ut, ep.injections)
        pre = env.model_copy(deep=True)
        rt = FunctionsRuntime(suite.tools)
        prefix = [
            FunctionCall(
                function=e.call.function, args=copy.deepcopy(e.call.args), id=e.call.call_id
            )
            for e in executed[: k + 1]
        ]
        for fc in prefix:
            rt.run_function(env, fc.function, fc.args)
        if agentdojo_checks(ut, it, "", pre, env, prefix)[1]:
            return k
    return None
