"""Online monitor gate: score each side-effecting call right before it executes."""

from ast import literal_eval
from collections.abc import Callable, Sequence

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, is_string_list
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage

from wwtw.monitor.views import StepContext
from wwtw.traces.load import _call, events_from_messages
from wwtw.traces.side_effects import is_side_effecting


def step_context_from_messages(
    messages: Sequence[ChatMessage], assistant, executed: list, results: list, call, suite: str
) -> StepContext:
    """messages ends with the current assistant message; `executed` are the calls of that message
    already run and `results` their tool messages. Mirrors the offline prefix exactly."""
    prefix = [
        *messages[:-1],
        ChatAssistantMessage(
            role="assistant", content=assistant["content"], tool_calls=list(executed)
        ),
        *results,
    ]
    prompt, events, _ = events_from_messages(prefix)
    n_calls = sum(1 for e in events if e.kind == "tool_call")
    return StepContext("online", suite, n_calls, prompt, events, _call(call))


_EMPTY_ENV = EmptyEnv()


def _normalize_list_args(call) -> None:
    for k, v in list(call.args.items()):
        if isinstance(v, str) and is_string_list(v):
            call.args[k] = literal_eval(v)


class MonitoredToolsExecutor(ToolsExecutor):
    def __init__(
        self,
        suite_name: str,
        scorer: Callable[[StepContext], int | None],
        threshold: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.suite_name = suite_name
        self.scorer = scorer
        self.threshold = threshold

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = _EMPTY_ENV,
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ):
        extra_args = {} if extra_args is None else extra_args
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args
        assistant = messages[-1]
        executed: list = []
        results: list = []
        for tc in assistant["tool_calls"]:
            _normalize_list_args(tc)
            if is_side_effecting(self.suite_name, tc.function):
                ctx = step_context_from_messages(
                    messages, assistant, executed, results, tc, self.suite_name
                )
                score = self.scorer(ctx)
                if score is not None and score >= self.threshold:
                    raise AbortAgentError(
                        f"Blocked by security monitor (suspicion {score}).",
                        [*messages, *results],
                        env,
                    )
            single = ChatAssistantMessage(
                role="assistant", content=assistant["content"], tool_calls=[tc]
            )
            _, _, env, out, _ = super().query(
                query, runtime, env, [*messages[:-1], single], extra_args
            )
            results.append(out[-1])
            executed.append(tc)
        return query, runtime, env, [*messages, *results], extra_args
