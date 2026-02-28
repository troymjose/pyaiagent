"""Developer-driven multi-agent orchestration utilities.

These functions complement the LLM-driven ``class Agents:`` handoff
pattern by giving developers explicit control over agent composition:

* :func:`team`     — ad-hoc orchestrator without defining a class
* :func:`pipeline` — sequential chain: output of one agent feeds the next
* :func:`parallel` — concurrent fan-out: all agents run simultaneously
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence, Union

from pyaiagent.core.handoff import HandoffSpec, _is_agent_class

__all__ = ["team", "pipeline", "parallel"]


async def team(
    instruction: str,
    *,
    agents: Union[Sequence[type], Dict[str, Any]],
    input: str,
    session: str | None = None,
    instruction_params: Dict[str, str] | None = None,
    metadata: dict | None = None,
    base_cls: type | None = None,
) -> dict:
    """Create an ad-hoc orchestrator agent and run it.

    This is a convenience function that dynamically creates an agent
    subclass with the given instruction and agents, runs a single
    :meth:`process` call, and cleans up.  Useful for prototyping and
    scripts where defining a class is unnecessary.

    Args:
        instruction: The system instruction for the orchestrator.
        agents:      A list of agent classes (names auto-derived) or a
                     dict mapping tool names to agent classes /
                     :class:`HandoffSpec` instances.
        input:       The user message to process.
        session:     Optional session identifier.
        instruction_params: Template substitutions for the instruction.
        metadata:    Arbitrary metadata dict.
        base_cls:    The provider agent base class for the orchestrator.
                     Defaults to :class:`~pyaiagent.openai.agent.OpenAIAgent`.

    Returns:
        The result dict from :meth:`process`.

    Example::

        from pyaiagent import team

        result = await team(
            "You coordinate research.",
            agents=[Researcher, Writer],
            input="Write about quantum computing",
        )
    """
    if base_cls is None:
        from pyaiagent.openai.agent import OpenAIAgent
        base_cls = OpenAIAgent

    agents_dict: Dict[str, Any] = {}
    if isinstance(agents, dict):
        agents_dict = dict(agents)
    else:
        for entry in agents:
            if isinstance(entry, HandoffSpec):
                name = entry.agent_cls.__name__.lower()
            elif _is_agent_class(entry):
                name = entry.__name__.lower()
            else:
                raise TypeError(
                    f"team() agents entries must be agent subclasses or "
                    f"handoff() specs, got {type(entry).__name__}")
            if name in agents_dict:
                raise ValueError(
                    f"team() duplicate agent name '{name}'. "
                    f"Use a dict to assign explicit names when classes collide.")
            agents_dict[name] = entry

    agents_inner = type("Agents", (), agents_dict)

    team_cls = type("_Team", (base_cls,), {
        "__doc__": instruction,
        "Agents": agents_inner,
    })

    agent = team_cls()
    try:
        return await agent.process(
            input=input,
            session=session,
            instruction_params=instruction_params,
            metadata=metadata,
        )
    finally:
        await agent.aclose()


async def pipeline(
    agents: Sequence[type],
    *,
    input: str,
) -> dict:
    """Run agents sequentially, feeding each agent's output as the next's input.

    Token usage is accumulated across all agents in the pipeline.

    Args:
        agents: An ordered sequence of agent subclasses.
        input:  The initial user message.

    Returns:
        The result dict from the *last* agent, with aggregated token
        counts.

    Example::

        from pyaiagent import pipeline

        result = await pipeline(
            [Researcher, Writer, Editor],
            input="Write about quantum computing",
        )
    """
    if not agents:
        raise ValueError("pipeline() requires at least one agent")

    for i, agent_cls in enumerate(agents):
        if not _is_agent_class(agent_cls):
            raise TypeError(
                f"pipeline() agents[{i}] must be an agent subclass, "
                f"got {type(agent_cls).__name__}")

    current_input = input
    result: dict = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_total_tokens = 0

    for agent_cls in agents:
        agent = agent_cls()
        try:
            result = await agent.process(input=current_input)
            current_input = result["output"]
            total_input_tokens += result["tokens"]["input_tokens"]
            total_output_tokens += result["tokens"]["output_tokens"]
            total_total_tokens += result["tokens"]["total_tokens"]
        finally:
            await agent.aclose()

    result["tokens"] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_total_tokens,
    }
    return result


async def parallel(
    agents: Sequence[type],
    *,
    input: str,
) -> List[dict]:
    """Run agents concurrently and return all results.

    All agents receive the same input and execute in parallel via
    :func:`asyncio.gather`.  If any agent raises, the exception
    propagates and remaining tasks are cancelled.

    Args:
        agents: A sequence of agent subclasses.
        input:  The user message sent to every agent.

    Returns:
        A list of result dicts, one per agent, in the same order as
        the *agents* sequence.

    Example::

        from pyaiagent import parallel

        results = await parallel(
            [FactChecker, Summarizer],
            input="Evaluate this claim...",
        )
    """
    if not agents:
        raise ValueError("parallel() requires at least one agent")

    for i, agent_cls in enumerate(agents):
        if not _is_agent_class(agent_cls):
            raise TypeError(
                f"parallel() agents[{i}] must be an agent subclass, "
                f"got {type(agent_cls).__name__}")

    async def _run(agent_cls: type) -> dict:
        agent = agent_cls()
        try:
            return await agent.process(input=input)
        finally:
            await agent.aclose()

    return list(await asyncio.gather(*[_run(cls) for cls in agents]))
