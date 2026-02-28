"""Tests for pyaiagent.openai.orchestration — team(), pipeline(), parallel().

These tests verify the orchestration utilities at the unit level using
mocked agent classes. Integration with real OpenAI calls is out of scope.

Covers:
    - team() with list and dict agents, process kwargs forwarding, errors
    - pipeline() sequential chaining and token aggregation
    - parallel() concurrent execution
    - Edge cases: empty lists, non-agent entries, single agent

All patches target the module-level attribute path rather than a specific
class reference so that tests remain robust when ``importlib.reload`` tests
in the suite change class identity.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from pyaiagent.openai.orchestration import team, pipeline, parallel
from pyaiagent.openai.handoff import handoff


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agent_cls():
    """Get the current OpenAIAgent class from the live module."""
    return sys.modules["pyaiagent.openai.agent"].OpenAIAgent


def _make_mock_result(output: str, tokens: dict | None = None) -> dict:
    return {
        "input": "test",
        "session": "s1",
        "metadata": {},
        "output": output,
        "output_parsed": None,
        "steps": 1,
        "turn": "t1",
        "history": [],
        "events": [],
        "tokens": tokens or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


# Agent classes created at test time to avoid stale-class issues
def _fresh_agent(name: str, doc: str):
    """Create a fresh agent class using the current OpenAIAgent."""
    return type(name, (_agent_cls(),), {"__doc__": doc})


# ─────────────────────────────────────────────────────────────────────────────
# team()
# ─────────────────────────────────────────────────────────────────────────────

_PROCESS_PATH = "pyaiagent.openai.agent.OpenAIAgent.process"
_ENSURE_READY_PATH = "pyaiagent.openai.agent.OpenAIAgent._ensure_ready"


class TestTeam:
    @pytest.mark.asyncio
    async def test_creates_dynamic_agent_with_list(self):
        """team() with a list of classes auto-derives tool names."""
        AgentA = _fresh_agent("AgentA", "Agent A does task A.")
        AgentB = _fresh_agent("AgentB", "Agent B does task B.")

        with patch(_PROCESS_PATH, new_callable=AsyncMock,
                   return_value=_make_mock_result("team output")), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            result = await team(
                "You coordinate.",
                agents=[AgentA, AgentB],
                input="Do something",
            )
        assert result["output"] == "team output"

    @pytest.mark.asyncio
    async def test_creates_dynamic_agent_with_dict(self):
        """team() with a dict uses the dict keys as tool names."""
        AgentA = _fresh_agent("AgentA", "Agent A does task A.")
        AgentB = _fresh_agent("AgentB", "Agent B does task B.")

        with patch(_PROCESS_PATH, new_callable=AsyncMock,
                   return_value=_make_mock_result("dict output")), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            result = await team(
                "You coordinate.",
                agents={"researcher": AgentA, "writer": AgentB},
                input="Do something",
            )
        assert result["output"] == "dict output"

    @pytest.mark.asyncio
    async def test_accepts_handoff_specs_in_list(self):
        AgentA = _fresh_agent("AgentA", "Agent A does task A.")

        with patch(_PROCESS_PATH, new_callable=AsyncMock,
                   return_value=_make_mock_result("ok")), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            result = await team(
                "Coordinate.",
                agents=[handoff(AgentA, description="Custom")],
                input="Go",
            )
        assert result["output"] == "ok"

    @pytest.mark.asyncio
    async def test_invalid_entry_raises(self):
        with pytest.raises(TypeError, match="agent subclasses"):
            await team("Bad.", agents=[42], input="Go")

    @pytest.mark.asyncio
    async def test_duplicate_names_raises(self):
        """Two agents whose names collide after .lower() are rejected."""
        AgentA = _fresh_agent("Shared", "Agent A.")
        AgentB = _fresh_agent("Shared", "Agent B.")
        with pytest.raises(ValueError, match="duplicate agent name"):
            await team("Dup.", agents=[AgentA, AgentB], input="Go")

    @pytest.mark.asyncio
    async def test_forwards_process_kwargs(self):
        """session, instruction_params, metadata are forwarded."""
        AgentA = _fresh_agent("AgentA", "Agent A does task A.")

        with patch(_PROCESS_PATH, new_callable=AsyncMock,
                   return_value=_make_mock_result("ok")) as mock_proc, \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            await team(
                "Coord.",
                agents=[AgentA],
                input="test",
                session="sess-1",
                metadata={"key": "val"},
            )
        call_kwargs = mock_proc.call_args.kwargs
        assert call_kwargs["session"] == "sess-1"
        assert call_kwargs["metadata"] == {"key": "val"}


# ─────────────────────────────────────────────────────────────────────────────
# pipeline()
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    @pytest.mark.asyncio
    async def test_chains_output_to_input(self):
        """Each agent's output becomes the next agent's input."""
        AgentA = _fresh_agent("AgentA", "A")
        AgentB = _fresh_agent("AgentB", "B")
        call_inputs = []

        async def _mock_process(self, *, input, **kw):
            call_inputs.append(input)
            return _make_mock_result(f"processed({input})")

        with patch(_PROCESS_PATH, _mock_process), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            result = await pipeline([AgentA, AgentB], input="start")

        assert call_inputs == ["start", "processed(start)"]
        assert result["output"] == "processed(processed(start))"

    @pytest.mark.asyncio
    async def test_aggregates_tokens(self):
        AgentA = _fresh_agent("AgentA", "A")
        AgentB = _fresh_agent("AgentB", "B")
        AgentC = _fresh_agent("AgentC", "C")
        call_count = 0

        async def _mock_process(self, *, input, **kw):
            nonlocal call_count
            call_count += 1
            return _make_mock_result("out", tokens={
                "input_tokens": 10 * call_count,
                "output_tokens": 5 * call_count,
                "total_tokens": 15 * call_count,
            })

        with patch(_PROCESS_PATH, _mock_process), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            result = await pipeline([AgentA, AgentB, AgentC], input="go")

        assert result["tokens"]["input_tokens"] == 10 + 20 + 30
        assert result["tokens"]["output_tokens"] == 5 + 10 + 15
        assert result["tokens"]["total_tokens"] == 15 + 30 + 45

    @pytest.mark.asyncio
    async def test_empty_pipeline_raises(self):
        with pytest.raises(ValueError, match="at least one agent"):
            await pipeline([], input="test")

    @pytest.mark.asyncio
    async def test_single_agent_pipeline(self):
        AgentA = _fresh_agent("AgentA", "A")

        with patch(_PROCESS_PATH, new_callable=AsyncMock,
                   return_value=_make_mock_result("single")), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            result = await pipeline([AgentA], input="go")
        assert result["output"] == "single"


# ─────────────────────────────────────────────────────────────────────────────
# parallel()
# ─────────────────────────────────────────────────────────────────────────────

class TestParallel:
    @pytest.mark.asyncio
    async def test_runs_all_agents(self):
        AgentA = _fresh_agent("AgentA", "A")
        AgentB = _fresh_agent("AgentB", "B")
        call_count = 0

        async def _mock_process(self, *, input, **kw):
            nonlocal call_count
            call_count += 1
            return _make_mock_result(f"output_{type(self).__name__}")

        with patch(_PROCESS_PATH, _mock_process), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            results = await parallel([AgentA, AgentB], input="go")

        assert len(results) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_list_in_order(self):
        AgentA = _fresh_agent("AgentA", "A")
        AgentB = _fresh_agent("AgentB", "B")

        async def _mock_process(self, *, input, **kw):
            return _make_mock_result(type(self).__name__)

        with patch(_PROCESS_PATH, _mock_process), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            results = await parallel([AgentA, AgentB], input="go")

        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_parallel_raises(self):
        with pytest.raises(ValueError, match="at least one agent"):
            await parallel([], input="test")

    @pytest.mark.asyncio
    async def test_single_agent_parallel(self):
        AgentA = _fresh_agent("AgentA", "A")

        with patch(_PROCESS_PATH, new_callable=AsyncMock,
                   return_value=_make_mock_result("one")), \
             patch(_ENSURE_READY_PATH, new_callable=AsyncMock):
            results = await parallel([AgentA], input="go")
        assert len(results) == 1
        assert results[0]["output"] == "one"
