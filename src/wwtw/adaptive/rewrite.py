"""Re-render an episode with a different appeal while the agent's recorded actions stay fixed.

The injection vectors are rewritten, the benchmark environment is rebuilt, and the recorded tool
calls are executed again, so every tool output comes out exactly as AgentDojo renders it (including
line folding of long strings). Outputs that did not change are kept byte for byte."""

import copy
from dataclasses import dataclass, replace

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime

from wwtw.adaptive.appeals import appeal_user_name, swap_appeal
from wwtw.traces.replay import build_environment, outputs_equal
from wwtw.traces.schema import Episode


@dataclass(frozen=True)
class RewriteReport:
    changed: tuple[int, ...]  # ordinals of tool results whose text differs from the recording


def rewrite_episode(episode: Episode, suite, appeal: str | None) -> tuple[Episode, RewriteReport]:
    if episode.injection_task_id is None or appeal_user_name(episode.injections) is None:
        raise ValueError(f"{episode.episode_id} has no recorded appeal to replace")
    inj_task = suite.get_injection_task_by_id(episode.injection_task_id)
    injections = swap_appeal(episode.injections, appeal, inj_task.GOAL)
    user_task = suite.get_user_task_by_id(episode.user_task_id)
    env = build_environment(suite, user_task, injections)
    runtime = FunctionsRuntime(suite.tools)

    fresh: dict[int, tuple[str, str | None]] = {}
    changed = []
    for ex in episode.executed_calls():
        result, error = runtime.run_function(env, ex.call.function, copy.deepcopy(ex.call.args))
        text = tool_result_to_str(result)
        if outputs_equal(text, ex.result.text) and (error or None) == (ex.result.error or None):
            continue
        fresh[ex.ordinal] = (text, error or None)
        changed.append(ex.ordinal)

    events = tuple(
        replace(e, text=fresh[e.ordinal][0], error=fresh[e.ordinal][1])
        if e.kind == "tool_result" and e.ordinal in fresh
        else e
        for e in episode.events
    )
    return replace(episode, injections=injections, events=events), RewriteReport(tuple(changed))
