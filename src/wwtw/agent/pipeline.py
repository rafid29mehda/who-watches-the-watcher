"""The standard AgentDojo tool-calling pipeline, built around our cached, budgeted client."""

from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor


def build_agent_pipeline(
    client, model: str, executor=None, name: str | None = None
) -> AgentPipeline:
    llm = OpenAILLM(client, model, temperature=0.0)  # the client enforces temperature=0.0
    loop = ToolsExecutionLoop([executor if executor is not None else ToolsExecutor(), llm])
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, loop])
    pipeline.name = name or model
    return pipeline
