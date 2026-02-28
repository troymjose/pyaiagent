"""Tests for pyaiagent.openai.handoff — HandoffSpec, handoff(), and discovery.

Covers:
    - HandoffSpec creation and immutability
    - handoff() convenience function
    - build_handoff_tool_schema() output structure
    - discover_handoffs() with plain classes, HandoffSpec, inheritance,
      missing docstrings, invalid entries, and edge cases
"""
from __future__ import annotations

import pytest
import textwrap

from pyaiagent.openai.handoff import (
    HandoffSpec,
    handoff,
    build_handoff_tool_schema,
    discover_handoffs,
)
from pyaiagent.openai.agent import OpenAIAgent
from pyaiagent.openai.exceptions import OpenAIAgentDefinitionError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — minimal agent classes for testing
# ─────────────────────────────────────────────────────────────────────────────

class _WorkerA(OpenAIAgent):
    """Worker A does research."""

class _WorkerB(OpenAIAgent):
    """Worker B writes articles."""

# Simulates an agent class with no docstring (bypasses __init_subclass__
# which would raise at class creation time).
_NoDocWorkerForTest = type("_NoDocWorkerForTest", (), {
    "__doc__": "",
    "__instruction__": "",
})


# ─────────────────────────────────────────────────────────────────────────────
# HandoffSpec
# ─────────────────────────────────────────────────────────────────────────────

class TestHandoffSpec:
    def test_creation_defaults(self):
        spec = HandoffSpec(agent_cls=_WorkerA)
        assert spec.agent_cls is _WorkerA
        assert spec.description is None

    def test_creation_with_description(self):
        spec = HandoffSpec(agent_cls=_WorkerA, description="Custom desc")
        assert spec.description == "Custom desc"

    def test_frozen(self):
        spec = HandoffSpec(agent_cls=_WorkerA)
        with pytest.raises(AttributeError):
            spec.agent_cls = _WorkerB

    def test_equality(self):
        a = HandoffSpec(agent_cls=_WorkerA, description="x")
        b = HandoffSpec(agent_cls=_WorkerA, description="x")
        assert a == b

    def test_inequality(self):
        a = HandoffSpec(agent_cls=_WorkerA)
        b = HandoffSpec(agent_cls=_WorkerB)
        assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# handoff() convenience function
# ─────────────────────────────────────────────────────────────────────────────

class TestHandoffFunction:
    def test_plain(self):
        spec = handoff(_WorkerA)
        assert isinstance(spec, HandoffSpec)
        assert spec.agent_cls is _WorkerA
        assert spec.description is None

    def test_with_description(self):
        spec = handoff(_WorkerA, description="Research agent")
        assert spec.description == "Research agent"


# ─────────────────────────────────────────────────────────────────────────────
# build_handoff_tool_schema
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHandoffToolSchema:
    def test_schema_structure(self):
        schema = build_handoff_tool_schema("researcher", "Does research.")
        assert schema["type"] == "function"
        assert schema["name"] == "researcher"
        assert schema["description"] == "Does research."
        assert schema["strict"] is True

        params = schema["parameters"]
        assert params["type"] == "object"
        assert "input" in params["properties"]
        assert params["properties"]["input"]["type"] == "string"
        assert params["required"] == ["input"]
        assert params["additionalProperties"] is False

    def test_different_names(self):
        s1 = build_handoff_tool_schema("a", "desc a")
        s2 = build_handoff_tool_schema("b", "desc b")
        assert s1["name"] == "a"
        assert s2["name"] == "b"


# ─────────────────────────────────────────────────────────────────────────────
# discover_handoffs
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscoverHandoffs:
    def test_no_agents_class(self):
        class NoAgents(OpenAIAgent):
            """No agents here."""
        handoffs, schemas = discover_handoffs(NoAgents)
        assert handoffs == {}
        assert schemas == {}

    def test_plain_agent_reference(self):
        class Host(OpenAIAgent):
            """Host agent."""
            class Agents:
                worker = _WorkerA

        handoffs, schemas = discover_handoffs(Host)
        assert "worker" in handoffs
        assert handoffs["worker"].agent_cls is _WorkerA
        assert "worker" in schemas
        assert schemas["worker"]["name"] == "worker"

    def test_handoff_spec_reference(self):
        class Host(OpenAIAgent):
            """Host agent."""
            class Agents:
                worker = handoff(_WorkerA, description="Custom research")

        handoffs, schemas = discover_handoffs(Host)
        assert handoffs["worker"].description == "Custom research"
        assert schemas["worker"]["description"] == "Custom research"

    def test_multiple_agents(self):
        class Host(OpenAIAgent):
            """Host agent."""
            class Agents:
                alpha = _WorkerA
                beta = _WorkerB

        handoffs, schemas = discover_handoffs(Host)
        assert set(handoffs.keys()) == {"alpha", "beta"}
        assert set(schemas.keys()) == {"alpha", "beta"}

    def test_description_from_docstring(self):
        class Host(OpenAIAgent):
            """Host agent."""
            class Agents:
                worker = _WorkerA

        _, schemas = discover_handoffs(Host)
        assert schemas["worker"]["description"] == "Worker A does research."

    def test_custom_description_overrides_docstring(self):
        class Host(OpenAIAgent):
            """Host agent."""
            class Agents:
                worker = handoff(_WorkerA, description="Override")

        _, schemas = discover_handoffs(Host)
        assert schemas["worker"]["description"] == "Override"

    def test_invalid_entry_raises(self):
        with pytest.raises(OpenAIAgentDefinitionError, match="expected an agent subclass"):
            class Host(OpenAIAgent):
                """Host agent."""
                class Agents:
                    bad = 42

    def test_non_agent_class_raises(self):
        with pytest.raises(OpenAIAgentDefinitionError, match="expected an agent subclass"):
            class Host(OpenAIAgent):
                """Host agent."""
                class Agents:
                    bad = str

    def test_handoff_with_non_agent_cls_raises(self):
        with pytest.raises(OpenAIAgentDefinitionError, match="must be an agent subclass"):
            class Host(OpenAIAgent):
                """Host agent."""
                class Agents:
                    bad = handoff(str)

    def test_private_names_skipped(self):
        class Host(OpenAIAgent):
            """Host agent."""
            class Agents:
                _internal = _WorkerA
                public = _WorkerB

        handoffs, _ = discover_handoffs(Host)
        assert "_internal" not in handoffs
        assert "public" in handoffs

    def test_agents_class_inheritance(self):
        """class Agents(Parent.Agents) merges parent entries."""
        class Parent(OpenAIAgent):
            """Parent."""
            class Agents:
                alpha = _WorkerA

        class Child(OpenAIAgent):
            """Child."""
            class Agents(Parent.Agents):
                beta = _WorkerB

        handoffs, schemas = discover_handoffs(Child)
        assert "alpha" in handoffs
        assert "beta" in handoffs

    def test_agent_no_description_raises(self):
        """An agent with no docstring and no explicit description is rejected."""
        with pytest.raises(OpenAIAgentDefinitionError, match="has no docstring"):
            class Host(OpenAIAgent):
                """Host."""
                class Agents:
                    bad = _NoDocWorkerForTest


# ─────────────────────────────────────────────────────────────────────────────
# Integration: __init_subclass__ discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestInitSubclassHandoffIntegration:
    def test_handoffs_stored_on_class(self):
        class Host(OpenAIAgent):
            """Host."""
            class Agents:
                worker = _WorkerA
        assert hasattr(Host, "__handoffs__")
        assert "worker" in Host.__handoffs__
        assert Host.__handoffs__["worker"].agent_cls is _WorkerA

    def test_handoff_names_stored(self):
        class Host(OpenAIAgent):
            """Host."""
            class Agents:
                a = _WorkerA
                b = _WorkerB
        assert set(Host.__handoff_names__) == {"a", "b"}

    def test_handoff_schemas_stored(self):
        class Host(OpenAIAgent):
            """Host."""
            class Agents:
                worker = _WorkerA
        assert len(Host.__handoff_schemas__) == 1
        assert Host.__handoff_schemas__[0]["name"] == "worker"

    def test_no_agents_class_empty_handoffs(self):
        class Plain(OpenAIAgent):
            """Plain agent."""
        assert Plain.__handoffs__ == {}
        assert Plain.__handoff_names__ == ()
        assert Plain.__handoff_schemas__ == ()

    def test_tool_handoff_name_collision_raises(self):
        with pytest.raises(OpenAIAgentDefinitionError, match="Name collision"):
            class Bad(OpenAIAgent):
                """Bad agent."""
                class Agents:
                    search = _WorkerA
                async def search(self, q: str) -> dict:
                    """Search."""
                    return {}

    def test_get_definition_includes_agents(self):
        class Host(OpenAIAgent):
            """Host."""
            class Agents:
                worker = _WorkerA
        defn = Host.get_definition()
        assert "agents" in defn
        assert "worker" in defn["agents"]
        assert defn["agents"]["worker"]["agent_class"] == "_WorkerA"

    def test_get_definition_no_agents_key_when_empty(self):
        class Plain(OpenAIAgent):
            """Plain."""
        defn = Plain.get_definition()
        assert "agents" not in defn

    def test_inherited_handoffs(self):
        class Parent(OpenAIAgent):
            """Parent."""
            class Agents:
                alpha = _WorkerA

        class Child(Parent):
            """Child."""

        assert "alpha" in Child.__handoffs__
        assert Child.__handoffs__["alpha"].agent_cls is _WorkerA

    def test_child_adds_own_handoffs(self):
        class Parent(OpenAIAgent):
            """Parent."""
            class Agents:
                alpha = _WorkerA

        class Child(Parent):
            """Child."""
            class Agents:
                beta = _WorkerB

        assert "alpha" in Child.__handoffs__
        assert "beta" in Child.__handoffs__

    def test_child_overrides_parent_handoff(self):
        class Parent(OpenAIAgent):
            """Parent."""
            class Agents:
                worker = _WorkerA

        class Child(Parent):
            """Child."""
            class Agents:
                worker = handoff(_WorkerB, description="Overridden")

        assert Child.__handoffs__["worker"].agent_cls is _WorkerB
        assert Child.__handoffs__["worker"].description == "Overridden"


# ─────────────────────────────────────────────────────────────────────────────
# _make_handoff_executor
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeHandoffExecutor:
    @pytest.mark.asyncio
    async def test_executor_returns_callable(self):
        spec = HandoffSpec(agent_cls=_WorkerA)
        executor = OpenAIAgent._make_handoff_executor(spec)
        assert callable(executor)
        import asyncio
        assert asyncio.iscoroutinefunction(executor)

    @pytest.mark.asyncio
    async def test_executor_delegates_to_sub_agent(self):
        """The handoff executor creates a sub-agent, calls process(), and returns output."""
        from unittest.mock import AsyncMock, patch

        spec = HandoffSpec(agent_cls=_WorkerA)
        executor = OpenAIAgent._make_handoff_executor(spec)

        mock_result = {
            "output": "sub-agent result",
            "tokens": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
        with patch.object(OpenAIAgent, "process", new_callable=AsyncMock,
                          return_value=mock_result), \
             patch.object(OpenAIAgent, "_ensure_ready", new_callable=AsyncMock):
            result = await executor(input="Do research")

        assert result == {"output": "sub-agent result"}

    @pytest.mark.asyncio
    async def test_executor_handles_sub_agent_error(self):
        """If the sub-agent raises, the executor returns an error dict."""
        from unittest.mock import AsyncMock, patch

        spec = HandoffSpec(agent_cls=_WorkerA)
        executor = OpenAIAgent._make_handoff_executor(spec)

        with patch.object(OpenAIAgent, "process", new_callable=AsyncMock,
                          side_effect=RuntimeError("boom")), \
             patch.object(OpenAIAgent, "_ensure_ready", new_callable=AsyncMock):
            result = await executor(input="fail")

        assert "error" in result
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_executor_handles_constructor_failure(self):
        """If agent_cls() raises, error is returned and no UnboundLocalError."""
        from unittest.mock import patch

        spec = HandoffSpec(agent_cls=_WorkerA)
        executor = OpenAIAgent._make_handoff_executor(spec)

        with patch.object(_WorkerA, "__new__", side_effect=RuntimeError("init failed")):
            result = await executor(input="go")

        assert "error" in result
        assert "init failed" in result["error"]

    @pytest.mark.asyncio
    async def test_executor_closes_sub_agent(self):
        """The sub-agent is always closed, even on success."""
        from unittest.mock import AsyncMock, patch

        spec = HandoffSpec(agent_cls=_WorkerA)
        executor = OpenAIAgent._make_handoff_executor(spec)

        mock_result = {"output": "ok", "tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        with patch.object(OpenAIAgent, "process", new_callable=AsyncMock,
                          return_value=mock_result) as mock_proc, \
             patch.object(OpenAIAgent, "aclose", new_callable=AsyncMock) as mock_close, \
             patch.object(OpenAIAgent, "_ensure_ready", new_callable=AsyncMock):
            await executor(input="go")

        mock_close.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# _ensure_ready with handoffs
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureReadyHandoffs:
    @pytest.mark.asyncio
    async def test_handoff_executors_bound_in_tool_functions(self):
        """_ensure_ready adds handoff executors to _tool_functions."""
        from unittest.mock import patch, AsyncMock, MagicMock

        class Host(OpenAIAgent):
            """Host."""
            class Agents:
                worker = _WorkerA

        agent = Host()

        mock_client = MagicMock()
        with patch("pyaiagent.openai.agent.AsyncOpenAIClient") as MockClient:
            MockClient.return_value.client = mock_client
            await agent._ensure_ready()

        assert "worker" in agent._tool_functions
        import asyncio
        assert asyncio.iscoroutinefunction(agent._tool_functions["worker"])
