"""
Comprehensive unit tests for pyaiagent.openai.agent.

Covers: _resolve_instruction, _ResponseWithParsed, OpenAIAgent.__init_subclass__,
        __new__, __init__, _to_str, _ensure_ready, _create_events,
        _build_static_openai_responses_api_kwargs, _openai_responses_api_call,
        _execute_tool_call, _execute_tool_calls, _format_instruction,
        format_history, format_event, process (full loop: text,
        tool-calls, validation retries, max-steps, errors, token tracking),
        aclose, __aenter__, __aexit__, get_definition.
"""
from __future__ import annotations

import asyncio
import logging
import types
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx
import orjson
import pytest
from pydantic import BaseModel, ValidationError as PydanticValidationError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

from pyaiagent.openai.agent import (
    OpenAIAgent, _ResponseWithParsed, __all__ as agent_all,
)
from pyaiagent.core.agent import _resolve_instruction
from pyaiagent.openai.config import OpenAIAgentConfig
from pyaiagent.openai.exceptions import (
    OpenAIAgentDefinitionError,
    OpenAIAgentProcessError,
    OpenAIAgentClosedError,
    InvalidInputError,
    InvalidSessionError,
    InvalidMetadataError,
    InvalidHistoryError,
    InvalidInstructionParamsError,
    InstructionKeyError,
    ClientError,
    MaxStepsExceededError,
    ValidationRetriesExhaustedError,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=20, total_tokens=30):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class FakeOutputItem:
    """Mimics an SDK output item with model_dump()."""
    def __init__(self, data: dict | None = None):
        self._data = data or {"type": "message", "content": "hello"}

    def model_dump(self):
        return dict(self._data)


class FakeResponse:
    """Minimal fake OpenAI response."""
    def __init__(self, output_text="Hello!", output=None, usage=None,
                 output_parsed=None):
        self.output_text = output_text
        self.output = output if output is not None else [FakeOutputItem()]
        self.usage = usage or FakeUsage()
        self.output_parsed = output_parsed


class SimpleOutput(BaseModel):
    answer: str


def _make_tool_call(name="greet", arguments='{"name": "Alice"}', call_id="call_1"):
    return ResponseFunctionToolCall.model_construct(
        id="fc_" + call_id, call_id=call_id, name=name,
        arguments=arguments, type="function_call",
    )


def _make_output_message(content="Hi"):
    return ResponseOutputMessage.model_construct(
        id="msg_1", type="message", role="assistant", status="completed",
        content=[{"type": "output_text", "text": content}],
    )


def _patch_client():
    """Return (patcher, mock_underlying_client) that prevents real network."""
    mock_client = MagicMock()
    mock_client.responses = MagicMock()
    mock_client.responses.create = AsyncMock()
    mock_client.responses.parse = AsyncMock()
    patcher = patch(
        "pyaiagent.openai.agent.AsyncOpenAIClient",
        return_value=MagicMock(client=mock_client),
    )
    return patcher, mock_client


async def _ready_agent(agent, mock_client=None):
    """Call _ensure_ready with patched client; return the mock client."""
    patcher, mc = _patch_client()
    if mock_client is not None:
        mc = mock_client
    with patcher:
        await agent._ensure_ready()
    return mc


async def _ready_agent_for_validation(agent_cls, text_format, validation_retries):
    """Create an agent pre-configured for validation retry tests.

    Because `text_format` (a class) is callable and gets filtered by
    ConfigResolver, we manually inject it into the config after normal init.
    """
    patcher, mock_client = _patch_client()
    with patcher:
        agent = agent_cls()
        await agent._ensure_ready()
        old = agent._config
        new_kw = {f.name: getattr(old, f.name) for f in old.__dataclass_fields__.values()}
        new_kw["text_format"] = text_format
        new_kw["validation_retries"] = validation_retries
        object.__setattr__(agent, "_config", OpenAIAgentConfig(**new_kw))
        from pyaiagent.openai.agent import to_strict_json_schema
        if to_strict_json_schema is not None:
            object.__setattr__(agent, "_text_format_schema", {
                "format": {
                    "type": "json_schema",
                    "name": text_format.__name__,
                    "schema": to_strict_json_schema(text_format),
                    "strict": True,
                }
            })
    return agent, mock_client


# ── Agent subclasses (at module level for init_subclass) ─────────────────────

class SimpleAgent(OpenAIAgent):
    """You are a helpful assistant."""

class AgentWithConfig(OpenAIAgent):
    """You are helpful."""
    class Config:
        model = "gpt-4o"
        temperature = 0.5
        top_p = 0.9
        seed = 42
        user = "user_123"
        max_output_tokens = 2048
        max_steps = 5
        max_parallel_tools = 3
        tool_timeout = 10.0
        llm_timeout = 60.0
        include_events = False
        include_history = False
        validation_retries = 0

class AgentWithTool(OpenAIAgent):
    """You are helpful. Use tools."""
    async def greet(self, name: str) -> dict:
        """Greet someone by name."""
        return {"greeting": f"Hello, {name}!"}

class AgentWithSyncTool(OpenAIAgent):
    """Agent with sync tool."""
    def compute(self, x: int) -> dict:
        """Compute a value."""
        return {"result": x * 2}

class AgentWithMultipleTools(OpenAIAgent):
    """Agent with multiple tools."""
    async def tool_a(self, x: str) -> dict:
        """Tool A."""
        return {"a": x}
    async def tool_b(self, y: int) -> dict:
        """Tool B."""
        return {"b": y}

class AgentWithTemplateInstruction(OpenAIAgent):
    """Hello {name}, you are in {city}."""

class StrictTemplateAgent(OpenAIAgent):
    """Hello {name}."""
    class Config:
        strict_instruction_params = True

class AgentWithEscapedBraces(OpenAIAgent):
    """Use JSON like {{key}} in your response."""

class MaxStepsAgent(OpenAIAgent):
    """Agent with max_steps=1."""
    class Config:
        max_steps = 1

class ValidationRetryBaseAgent(OpenAIAgent):
    """Return structured output with retries."""
    class Config:
        validation_retries = 2


# ─────────────────────────────────────────────────────────────────────────────
# Module-level exports
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleExports:
    def test_all_contains_openai_agent(self):
        assert "OpenAIAgent" in agent_all

    def test_all_length(self):
        assert len(agent_all) == 1


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_instruction
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveInstruction:
    def test_normal_docstring(self):
        class Cls:
            """Hello world."""
        assert _resolve_instruction(Cls) == "Hello world."

    def test_dedented_docstring(self):
        class Cls:
            """
            Indented
            instruction.
            """
        assert _resolve_instruction(Cls) == "Indented\ninstruction."

    def test_empty_docstring_raises(self):
        class Cls:
            ""
        with pytest.raises(OpenAIAgentDefinitionError, match="Missing class docstring"):
            _resolve_instruction(Cls)

    def test_none_docstring_raises(self):
        Cls = type("Cls", (), {"__doc__": None})
        with pytest.raises(OpenAIAgentDefinitionError, match="Missing class docstring"):
            _resolve_instruction(Cls)

    def test_whitespace_only_raises(self):
        class Cls:
            """   \n\n   """
        with pytest.raises(OpenAIAgentDefinitionError, match="Missing class docstring"):
            _resolve_instruction(Cls)


# ─────────────────────────────────────────────────────────────────────────────
# _ResponseWithParsed
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseWithParsed:
    def test_slots(self):
        assert "_response" in _ResponseWithParsed.__slots__
        assert "output_parsed" in _ResponseWithParsed.__slots__
        assert "validation_error" in _ResponseWithParsed.__slots__

    def test_defaults(self):
        r = _ResponseWithParsed("resp")
        assert r._response == "resp"
        assert r.output_parsed is None
        assert r.validation_error is None

    def test_custom_values(self):
        r = _ResponseWithParsed("resp", output_parsed="parsed", validation_error="err")
        assert r.output_parsed == "parsed"
        assert r.validation_error == "err"

    def test_getattr_delegation(self):
        inner = MagicMock()
        inner.usage = FakeUsage()
        inner.output_text = "hi"
        r = _ResponseWithParsed(inner)
        assert r.usage is inner.usage
        assert r.output_text == "hi"

    def test_getattr_missing_raises(self):
        r = _ResponseWithParsed(object())
        with pytest.raises(AttributeError):
            _ = r.nonexistent_attr


# ─────────────────────────────────────────────────────────────────────────────
# __init_subclass__
# ─────────────────────────────────────────────────────────────────────────────

class TestInitSubclass:
    def test_agent_name_set(self):
        assert SimpleAgent.__agent_name__ == "SimpleAgent"

    def test_instruction_set(self):
        assert SimpleAgent.__instruction__ == "You are a helpful assistant."

    def test_config_kwargs_set(self):
        assert isinstance(SimpleAgent.__config_kwargs__, dict)

    def test_tool_names_tuple(self):
        assert isinstance(SimpleAgent.__tool_names__, tuple)

    def test_tools_schema_tuple(self):
        assert isinstance(SimpleAgent.__tools_schema__, tuple)

    def test_tool_discovery(self):
        assert "greet" in AgentWithTool.__tool_names__

    def test_multiple_tools(self):
        assert "tool_a" in AgentWithMultipleTools.__tool_names__
        assert "tool_b" in AgentWithMultipleTools.__tool_names__

    def test_no_docstring_raises(self):
        with pytest.raises(OpenAIAgentDefinitionError, match="Missing class docstring"):
            type("BadAgent", (OpenAIAgent,), {"__doc__": ""})

    def test_config_merged(self):
        assert AgentWithConfig.__config_kwargs__["model"] == "gpt-4o"
        assert AgentWithConfig.__config_kwargs__["temperature"] == 0.5

    def test_inheritance_preserves_tools(self):
        class Child(AgentWithTool):
            """Child agent."""
        assert "greet" in Child.__tool_names__

    def test_skip_base_class(self):
        assert not hasattr(OpenAIAgent, "__agent_name__")


# ─────────────────────────────────────────────────────────────────────────────
# __new__ and __init__
# ─────────────────────────────────────────────────────────────────────────────

class TestNewAndInit:
    def test_new_initializes_slots(self):
        agent = SimpleAgent()
        assert agent._ready is False
        assert agent._closed is False
        assert agent._config is None
        assert agent._client is None
        assert agent._semaphore is None
        assert agent._tool_functions is None
        assert agent._text_format_schema is None
        assert agent._static_openai_responses_api_kwargs is None

    def test_base_class_raises(self):
        with pytest.raises(TypeError, match="abstract agent class"):
            OpenAIAgent()

    def test_subclass_instantiates(self):
        agent = SimpleAgent()
        assert isinstance(agent, OpenAIAgent)

    def test_ready_lock_is_asyncio_lock(self):
        agent = SimpleAgent()
        assert isinstance(agent._ready_lock, asyncio.Lock)


# ─────────────────────────────────────────────────────────────────────────────
# _to_str
# ─────────────────────────────────────────────────────────────────────────────

class TestToStr:
    def test_string_passthrough(self):
        assert OpenAIAgent._to_str("hello") == "hello"

    def test_dict_serialized(self):
        result = OpenAIAgent._to_str({"key": "value"})
        assert '"key"' in result and '"value"' in result

    def test_list_serialized(self):
        assert OpenAIAgent._to_str([1, 2, 3]) == "[1,2,3]"

    def test_int_serialized(self):
        assert OpenAIAgent._to_str(42) == "42"

    def test_fallback_str(self):
        class Unserializable:
            def __str__(self):
                return "fallback"
        assert OpenAIAgent._to_str(Unserializable()) == "fallback"

    def test_empty_string(self):
        assert OpenAIAgent._to_str("") == ""

    def test_none_serialized(self):
        assert OpenAIAgent._to_str(None) == "null"

    def test_bool_serialized(self):
        assert OpenAIAgent._to_str(True) == "true"

    def test_nested_dict(self):
        result = OpenAIAgent._to_str({"a": {"b": 1}})
        assert '"a"' in result


# ─────────────────────────────────────────────────────────────────────────────
# get_definition
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDefinition:
    def test_returns_dict(self):
        assert isinstance(SimpleAgent.get_definition(), dict)

    def test_agent_name(self):
        assert SimpleAgent.get_definition()["agent_name"] == "SimpleAgent"

    def test_instruction(self):
        assert SimpleAgent.get_definition()["instruction"] == "You are a helpful assistant."

    def test_config(self):
        assert isinstance(SimpleAgent.get_definition()["config"], dict)

    def test_tools_populated(self):
        assert "greet" in AgentWithTool.get_definition()["tools"]

    def test_tools_empty_for_no_tools(self):
        assert SimpleAgent.get_definition()["tools"] == {}

    def test_all_keys_present(self):
        defn = SimpleAgent.get_definition()
        assert set(defn.keys()) == {"agent_name", "instruction", "config", "tools"}


# ─────────────────────────────────────────────────────────────────────────────
# _ensure_ready
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureReady:
    @pytest.mark.asyncio
    async def test_initializes_config(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert agent._config is not None
            assert agent._ready is True

    @pytest.mark.asyncio
    async def test_idempotent(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            config1 = agent._config
            await agent._ensure_ready()
            assert agent._config is config1

    @pytest.mark.asyncio
    async def test_creates_semaphore(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert isinstance(agent._semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_binds_tool_functions(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            assert "greet" in agent._tool_functions
            assert callable(agent._tool_functions["greet"])

    @pytest.mark.asyncio
    async def test_text_format_schema_none_without_retries(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert agent._text_format_schema is None

    @pytest.mark.asyncio
    async def test_text_format_schema_built_when_format_and_retries(self):
        """Manually inject text_format to test the schema-build branch."""
        patcher, _ = _patch_client()
        with patcher:
            agent = ValidationRetryBaseAgent()
            # Override __config_kwargs__ with text_format before _ensure_ready
            saved = dict(agent.__config_kwargs__)
            try:
                agent.__class__.__config_kwargs__ = {**saved, "text_format": SimpleOutput}
                await agent._ensure_ready()
                assert agent._text_format_schema is not None
                assert agent._text_format_schema["format"]["type"] == "json_schema"
                assert agent._text_format_schema["format"]["name"] == "SimpleOutput"
            finally:
                agent.__class__.__config_kwargs__ = saved

    @pytest.mark.asyncio
    async def test_missing_to_strict_json_schema_raises(self):
        patcher, _ = _patch_client()
        with patcher, patch("pyaiagent.openai.agent.to_strict_json_schema", None):
            agent = ValidationRetryBaseAgent()
            saved = dict(agent.__config_kwargs__)
            try:
                agent.__class__.__config_kwargs__ = {**saved, "text_format": SimpleOutput}
                with pytest.raises(ImportError, match="validation_retries requires"):
                    await agent._ensure_ready()
            finally:
                agent.__class__.__config_kwargs__ = saved

    @pytest.mark.asyncio
    async def test_builds_static_kwargs(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert agent._static_openai_responses_api_kwargs is not None
            assert "model" in agent._static_openai_responses_api_kwargs

    @pytest.mark.asyncio
    async def test_concurrent_ensure_ready(self):
        """Two concurrent calls should produce the same config (lock protects)."""
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            r1, r2 = await asyncio.gather(agent._ensure_ready(), agent._ensure_ready())
            assert agent._ready is True


# ─────────────────────────────────────────────────────────────────────────────
# _build_static_openai_responses_api_kwargs
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildStaticKwargs:
    @pytest.mark.asyncio
    async def test_basic_fields(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            kw = agent._static_openai_responses_api_kwargs
            assert kw["model"] == "gpt-4o-mini"
            assert kw["store"] is False
            assert kw["stream"] is False
            assert "tools" in kw

    @pytest.mark.asyncio
    async def test_optional_fields_included(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithConfig()
            await agent._ensure_ready()
            kw = agent._static_openai_responses_api_kwargs
            assert kw["top_p"] == 0.9
            assert kw["seed"] == 42
            assert kw["user"] == "user_123"

    @pytest.mark.asyncio
    async def test_optional_fields_excluded_when_none(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            kw = agent._static_openai_responses_api_kwargs
            assert "top_p" not in kw
            assert "seed" not in kw
            assert "user" not in kw

    @pytest.mark.asyncio
    async def test_tool_schemas_included(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            kw = agent._static_openai_responses_api_kwargs
            assert kw["tools"] == AgentWithTool.__tools_schema__

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_flag(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert "parallel_tool_calls" in agent._static_openai_responses_api_kwargs


# ─────────────────────────────────────────────────────────────────────────────
# _create_events
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateEvents:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithConfig()
            await agent._ensure_ready()
            result = agent._create_events(
                data={"role": "user", "content": "hi"},
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert result == []

    @pytest.mark.asyncio
    async def test_dict_data(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data={"role": "user", "content": "hi"},
                session="s", turn="t", step=1,
                input_tokens=5, output_tokens=10, total_tokens=15,
                metadata={"key": "val"})
            assert len(result) == 1
            msg = result[0]
            assert msg["role"] == "user"
            assert msg["content"] == "hi"
            assert msg["agent"] == "SimpleAgent"
            assert msg["session"] == "s"
            assert msg["tokens"]["input_tokens"] == 5
            assert msg["metadata"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_list_data(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data=[{"role": "a"}, {"role": "b"}],
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_none_metadata_defaults_to_empty_dict(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data={"role": "user"},
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert result[0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_custom_message_id(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data={"role": "user"},
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None, message="custom_msg")
            assert result[0]["message"] == "custom_msg"

    @pytest.mark.asyncio
    async def test_auto_generated_message_id(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data={"role": "user"},
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert len(result[0]["message"]) > 0

    @pytest.mark.asyncio
    async def test_unsupported_data_type_returns_empty(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data=12345,
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert result == []

    @pytest.mark.asyncio
    async def test_response_output_message_sdk_object(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            msg = _make_output_message("test content")
            result = agent._create_events(
                data=msg, session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert len(result) == 1
            assert result[0]["type"] == "message"

    @pytest.mark.asyncio
    async def test_response_function_tool_call_sdk_object(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            tc = _make_tool_call()
            result = agent._create_events(
                data=tc, session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert len(result) == 1
            assert result[0]["type"] == "function_call"

    @pytest.mark.asyncio
    async def test_parsed_arguments_stripped(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            tc = _make_tool_call()
            result = agent._create_events(
                data=tc, session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert "parsed_arguments" not in result[0]

    @pytest.mark.asyncio
    async def test_list_mixed_types_skips_unsupported(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            tc = _make_tool_call()
            result = agent._create_events(
                data=[{"role": "user"}, tc, 12345],
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_list(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data=[], session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# _format_instruction
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatInstruction:
    def test_no_params_returns_as_is(self):
        agent = SimpleAgent()
        assert agent._format_instruction() == "You are a helpful assistant."

    def test_params_substituted(self):
        agent = AgentWithTemplateInstruction()
        result = agent._format_instruction(instruction_params={"name": "Alice", "city": "NYC"})
        assert result == "Hello Alice, you are in NYC."

    def test_missing_param_non_strict_left_as_is(self):
        agent = AgentWithTemplateInstruction()
        result = agent._format_instruction(instruction_params={"name": "Alice"})
        assert "{city}" in result

    def test_missing_param_strict_raises(self):
        agent = StrictTemplateAgent()
        with pytest.raises(InstructionKeyError, match="'name'"):
            agent._format_instruction(instruction_params={})

    def test_escaped_braces(self):
        agent = AgentWithEscapedBraces()
        result = agent._format_instruction()
        assert "{key}" in result

    def test_none_params_non_strict(self):
        agent = AgentWithTemplateInstruction()
        result = agent._format_instruction(instruction_params=None)
        assert "{name}" in result

    def test_empty_params_dict_non_strict(self):
        agent = AgentWithTemplateInstruction()
        result = agent._format_instruction(instruction_params={})
        assert "{name}" in result

    def test_strict_no_params_raises(self):
        agent = StrictTemplateAgent()
        with pytest.raises(InstructionKeyError):
            agent._format_instruction(instruction_params=None)

    def test_json_braces_in_instruction(self):
        class JsonAgent(OpenAIAgent):
            """Return {"key": "value"} always."""
        agent = JsonAgent()
        result = agent._format_instruction()
        assert '{"key": "value"}' in result

    def test_partial_params(self):
        agent = AgentWithTemplateInstruction()
        result = agent._format_instruction(instruction_params={"name": "Bob"})
        assert "Hello Bob" in result
        assert "{city}" in result

    def test_escaped_and_normal_mixed(self):
        class MixedAgent(OpenAIAgent):
            """Name is {name}, JSON {{literal}}."""
        agent = MixedAgent()
        result = agent._format_instruction(instruction_params={"name": "X"})
        assert "Name is X" in result
        assert "{literal}" in result


# ─────────────────────────────────────────────────────────────────────────────
# format_history / format_event
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatMessages:
    def test_format_history_returns_output_text(self):
        agent = SimpleAgent()
        resp = MagicMock(output_text="hi there")
        assert agent.format_history(resp) == "hi there"

    def test_format_history_none_returns_empty(self):
        agent = SimpleAgent()
        resp = MagicMock(output_text=None)
        assert agent.format_history(resp) == ""

    def test_format_event_returns_output_text(self):
        agent = SimpleAgent()
        resp = MagicMock(output_text="hello ui")
        assert agent.format_event(resp) == "hello ui"

    def test_format_event_none_returns_empty(self):
        agent = SimpleAgent()
        resp = MagicMock(output_text=None)
        assert agent.format_event(resp) == ""

    def test_format_history_empty_string(self):
        agent = SimpleAgent()
        resp = MagicMock(output_text="")
        assert agent.format_history(resp) == ""


# ─────────────────────────────────────────────────────────────────────────────
# _execute_tool_call
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_async_tool_success(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="greet", arguments='{"name": "Alice"}', call_id="c1")
            result = await agent._execute_tool_call(tc)
            assert result["type"] == "function_call_output"
            assert result["call_id"] == "c1"
            assert "Hello, Alice!" in result["output"]

    @pytest.mark.asyncio
    async def test_sync_tool_success(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithSyncTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="compute", arguments='{"x": 5}', call_id="c2")
            result = await agent._execute_tool_call(tc)
            assert '"result":10' in result["output"].replace(" ", "")

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="nonexistent", call_id="c3")
            result = await agent._execute_tool_call(tc)
            assert "Tool not found" in result["output"]
            assert result["call_id"] == "c3"

    @pytest.mark.asyncio
    async def test_bad_json_arguments(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="greet", arguments="not json{{{", call_id="c4")
            result = await agent._execute_tool_call(tc)
            assert "Bad tool args JSON" in result["output"]

    @pytest.mark.asyncio
    async def test_empty_string_arguments(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="greet", arguments="", call_id="c5")
            result = await agent._execute_tool_call(tc)
            assert result["type"] == "function_call_output"

    @pytest.mark.asyncio
    async def test_dict_arguments(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = ResponseFunctionToolCall.model_construct(
                id="fc_c6", call_id="c6", name="greet",
                arguments={"name": "Bob"}, type="function_call")
            result = await agent._execute_tool_call(tc)
            assert "Hello, Bob!" in result["output"]

    @pytest.mark.asyncio
    async def test_bytes_arguments(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = ResponseFunctionToolCall.model_construct(
                id="fc_c7", call_id="c7", name="greet",
                arguments=b'{"name": "Bytes"}', type="function_call")
            result = await agent._execute_tool_call(tc)
            assert "Hello, Bytes!" in result["output"]

    @pytest.mark.asyncio
    async def test_bytearray_arguments(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = ResponseFunctionToolCall.model_construct(
                id="fc_c8", call_id="c8", name="greet",
                arguments=bytearray(b'{"name": "BA"}'), type="function_call")
            result = await agent._execute_tool_call(tc)
            assert "Hello, BA!" in result["output"]

    @pytest.mark.asyncio
    async def test_non_dict_return_wrapped(self):
        class StrReturnAgent(OpenAIAgent):
            """Agent returning string."""
            async def tool(self, x: str) -> dict:
                """A tool."""
                return "just a string"

        patcher, _ = _patch_client()
        with patcher:
            agent = StrReturnAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="tool", arguments='{"x": "a"}', call_id="c9")
            result = await agent._execute_tool_call(tc)
            parsed = orjson.loads(result["output"])
            assert parsed["result"] == "just a string"

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        class SlowAgent(OpenAIAgent):
            """Agent with slow tool."""
            class Config:
                tool_timeout = 0.01
            async def slow(self) -> dict:
                """A slow tool."""
                await asyncio.sleep(10)
                return {"done": True}

        patcher, _ = _patch_client()
        with patcher:
            agent = SlowAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="slow", arguments="{}", call_id="c10")
            result = await agent._execute_tool_call(tc)
            assert "timeout" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_tool_exception_logged(self, caplog):
        class FailAgent(OpenAIAgent):
            """Agent with failing tool."""
            async def fail(self) -> dict:
                """A failing tool."""
                raise ValueError("boom")

        patcher, _ = _patch_client()
        with patcher:
            agent = FailAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="fail", arguments="{}", call_id="c11")
            with caplog.at_level(logging.ERROR):
                result = await agent._execute_tool_call(tc)
            assert "Tool execution failed" in result["output"]
            assert "Tool execution failed: fail" in caplog.text

    @pytest.mark.asyncio
    async def test_unknown_argument_type_defaults_empty(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = ResponseFunctionToolCall.model_construct(
                id="fc_c12", call_id="c12", name="greet",
                arguments=12345, type="function_call")
            result = await agent._execute_tool_call(tc)
            assert result["type"] == "function_call_output"

    @pytest.mark.asyncio
    async def test_bad_json_logged(self, caplog):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="greet", arguments="{{bad", call_id="c13")
            with caplog.at_level(logging.ERROR):
                await agent._execute_tool_call(tc)
            assert "Bad tool args JSON: greet" in caplog.text

    @pytest.mark.asyncio
    async def test_none_tool_result(self):
        class NoneAgent(OpenAIAgent):
            """Agent returning None."""
            async def tool(self) -> dict:
                """A tool."""
                return None

        patcher, _ = _patch_client()
        with patcher:
            agent = NoneAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="tool", arguments="{}", call_id="c14")
            result = await agent._execute_tool_call(tc)
            parsed = orjson.loads(result["output"])
            assert parsed["result"] is None

    @pytest.mark.asyncio
    async def test_sync_tool_timeout(self):
        import time
        class SlowSyncAgent(OpenAIAgent):
            """Agent with slow sync tool."""
            class Config:
                tool_timeout = 0.01
            def slow_sync(self) -> dict:
                """A slow sync tool."""
                time.sleep(10)
                return {"done": True}

        patcher, _ = _patch_client()
        with patcher:
            agent = SlowSyncAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="slow_sync", arguments="{}", call_id="c15")
            result = await agent._execute_tool_call(tc)
            assert "timeout" in result["output"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# _execute_tool_calls
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteToolCalls:
    @pytest.mark.asyncio
    async def test_single_call_fast_path(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithTool()
            await agent._ensure_ready()
            tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
            results = await agent._execute_tool_calls([tc])
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_parallel(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithMultipleTools()
            await agent._ensure_ready()
            calls = [
                _make_tool_call(name="tool_a", arguments='{"x": "hello"}', call_id="c1"),
                _make_tool_call(name="tool_b", arguments='{"y": 42}', call_id="c2"),
            ]
            results = await agent._execute_tool_calls(calls)
            assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _openai_responses_api_call
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIResponsesApiCall:
    @pytest.mark.asyncio
    async def test_plain_text_call(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="answer")
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[])
            mc.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_passed_when_non_empty(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[],
                metadata={"k": "v"})
            assert mc.responses.create.call_args[1]["metadata"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_empty_metadata_not_passed(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[],
                metadata={})
            assert "metadata" not in mc.responses.create.call_args[1]

    @pytest.mark.asyncio
    async def test_none_metadata_not_passed(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[])
            assert "metadata" not in mc.responses.create.call_args[1]

    @pytest.mark.asyncio
    async def test_structured_no_retries_uses_parse(self):
        """When text_format set but validation_retries=0, uses responses.parse()."""
        patcher, mc = _patch_client()
        mc.responses.parse.return_value = FakeResponse()
        with patcher:
            agent, mc2 = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=0)
            agent._client = mc
            result = await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[])
            mc.responses.parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_retries_path_success(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text='{"answer":"ok"}')
        with patcher:
            agent, _ = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[])
            assert isinstance(result, _ResponseWithParsed)
            assert result.output_parsed.answer == "ok"
            assert result.validation_error is None

    @pytest.mark.asyncio
    async def test_validation_retries_path_parse_failure(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="not valid json")
        with patcher:
            agent, _ = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[])
            assert isinstance(result, _ResponseWithParsed)
            assert result.output_parsed is None
            assert result.validation_error is not None

    @pytest.mark.asyncio
    async def test_validation_retries_no_output_text(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text=None)
        with patcher:
            agent, _ = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent._openai_responses_api_call(
                instruction="test", current_turn_history=[])
            assert isinstance(result, _ResponseWithParsed)
            assert result.output_parsed is None
            assert result.validation_error is None

    @pytest.mark.asyncio
    async def test_http_error_raises_client_error(self):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock())
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            with pytest.raises(ClientError, match="HTTP error"):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])

    @pytest.mark.asyncio
    async def test_pydantic_error_raises_client_error(self):
        patcher, mc = _patch_client()
        mc.responses.parse.side_effect = PydanticValidationError.from_exception_data(
            title="test", line_errors=[])
        with patcher:
            agent, _ = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=0)
            agent._client = mc
            with pytest.raises(ClientError, match="Structured output validation"):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])

    @pytest.mark.asyncio
    async def test_generic_exception_raises_client_error(self):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = RuntimeError("unexpected")
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            with pytest.raises(ClientError, match="Unexpected error"):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = asyncio.CancelledError()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            with pytest.raises(asyncio.CancelledError):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])

    @pytest.mark.asyncio
    async def test_exception_chaining(self):
        patcher, mc = _patch_client()
        original = httpx.ConnectError("connection refused")
        mc.responses.create.side_effect = original
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            with pytest.raises(ClientError) as exc_info:
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])
            assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_http_error_logged(self, caplog):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = httpx.ConnectError("fail")
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            with caplog.at_level(logging.ERROR), pytest.raises(ClientError):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])
            assert "HTTP error" in caplog.text

    @pytest.mark.asyncio
    async def test_kwargs_include_instruction_and_input(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            await agent._openai_responses_api_call(
                instruction="be nice", current_turn_history=[{"role": "user"}])
            kw = mc.responses.create.call_args[1]
            assert kw["instructions"] == "be nice"
            assert kw["input"] == [{"role": "user"}]


# ─────────────────────────────────────────────────────────────────────────────
# process() — input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessValidation:
    @pytest.mark.asyncio
    async def test_closed_agent_raises(self):
        agent = SimpleAgent()
        await agent.aclose()
        with pytest.raises(OpenAIAgentClosedError):
            await agent.process(input="hi")

    @pytest.mark.asyncio
    async def test_non_string_input(self):
        with pytest.raises(InvalidInputError):
            await SimpleAgent().process(input=123)

    @pytest.mark.asyncio
    async def test_non_string_session(self):
        with pytest.raises(InvalidSessionError):
            await SimpleAgent().process(input="hi", session=123)

    @pytest.mark.asyncio
    async def test_empty_session(self):
        with pytest.raises(InvalidSessionError, match="empty string"):
            await SimpleAgent().process(input="hi", session="   ")

    @pytest.mark.asyncio
    async def test_whitespace_session(self):
        with pytest.raises(InvalidSessionError, match="empty string"):
            await SimpleAgent().process(input="hi", session="\t\n")

    @pytest.mark.asyncio
    async def test_non_dict_metadata(self):
        with pytest.raises(InvalidMetadataError):
            await SimpleAgent().process(input="hi", metadata="not a dict")

    @pytest.mark.asyncio
    async def test_non_list_history(self):
        with pytest.raises(InvalidHistoryError):
            await SimpleAgent().process(input="hi", history="not a list")

    @pytest.mark.asyncio
    async def test_non_dict_instruction_params(self):
        with pytest.raises(InvalidInstructionParamsError):
            await SimpleAgent().process(input="hi", instruction_params="bad")

    @pytest.mark.asyncio
    async def test_none_metadata_allowed(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="hi", metadata=None)
            assert result["metadata"] == {}

    @pytest.mark.asyncio
    async def test_none_session_allowed(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="hi", session=None)
            assert len(result["session"]) > 0

    @pytest.mark.asyncio
    async def test_none_history_allowed(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="hi", history=None)
            assert isinstance(result["history"], list)


# ─────────────────────────────────────────────────────────────────────────────
# process() — text response (no tools)
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessTextResponse:
    @pytest.mark.asyncio
    async def test_basic_response(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="Hello!", output=[_make_output_message("Hello!")])
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert result["output"] == "Hello!"
            assert result["steps"] == 1
            assert result["tokens"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_session_auto_generated(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert len(result["session"]) > 0

    @pytest.mark.asyncio
    async def test_session_preserved(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="Hi", session="my-session")
            assert result["session"] == "my-session"

    @pytest.mark.asyncio
    async def test_metadata_preserved(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="Hi", metadata={"k": "v"})
            assert result["metadata"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_input_preserved(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="test input")
            assert result["input"] == "test input"

    @pytest.mark.asyncio
    async def test_output_parsed_none_for_plain_text(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert result["output_parsed"] is None

    @pytest.mark.asyncio
    async def test_history_populated(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            llm = result["history"]
            assert llm[0] == {"role": "user", "content": "Hi"}
            assert llm[1] == {"role": "assistant", "content": "Reply"}

    @pytest.mark.asyncio
    async def test_history_disabled(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            result = await AgentWithConfig().process(input="Hi")
            assert result["history"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_events_populated(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert len(result["events"]) >= 2

    @pytest.mark.asyncio
    async def test_events_disabled(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            result = await AgentWithConfig().process(input="Hi")
            assert result["events"] == []

    @pytest.mark.asyncio
    async def test_history_shallow_copied(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            original = [{"role": "system", "content": "be nice"}]
            original_len = len(original)
            await SimpleAgent().process(input="Hi", history=original)
            assert len(original) == original_len

    @pytest.mark.asyncio
    async def test_none_output_text_raises_max_steps(self):
        """output_text=None with no tool calls = agent didn't produce a response."""
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text=None, output=[_make_output_message()])
        with patcher:
            with pytest.raises(MaxStepsExceededError):
                await SimpleAgent().process(input="Hi")

    @pytest.mark.asyncio
    async def test_empty_output_text_returns_empty(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="")
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert result["output"] == ""

    @pytest.mark.asyncio
    async def test_turn_is_uuid(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert len(result["turn"]) == 36


# ─────────────────────────────────────────────────────────────────────────────
# process() — tool call loop
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_then_text(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "Alice"}', call_id="c1")
        tool_resp = FakeResponse(output_text=None, output=[tc], usage=FakeUsage(10, 20, 30))
        text_resp = FakeResponse(
            output_text="Greeted!", output=[_make_output_message("Greeted!")],
            usage=FakeUsage(5, 15, 20))
        mc.responses.create.side_effect = [tool_resp, text_resp]
        with patcher:
            result = await AgentWithTool().process(input="Greet Alice")
            assert result["output"] == "Greeted!"
            assert result["steps"] == 2
            assert result["tokens"]["input_tokens"] == 15
            assert result["tokens"]["total_tokens"] == 50

    @pytest.mark.asyncio
    async def test_tool_output_in_history(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "Bob"}', call_id="c1")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc]),
            FakeResponse(output_text="Done!", output=[_make_output_message()]),
        ]
        with patcher:
            result = await AgentWithTool().process(input="Hi")
            llm = result["history"]
            tool_outputs = [m for m in llm if m.get("type") == "function_call_output"]
            assert len(tool_outputs) == 1

    @pytest.mark.asyncio
    async def test_parsed_arguments_stripped_from_llm(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc]),
            FakeResponse(output_text="Done!", output=[_make_output_message()]),
        ]
        with patcher:
            result = await AgentWithTool().process(input="Hi")
            for msg in result["history"]:
                assert "parsed_arguments" not in msg

    @pytest.mark.asyncio
    async def test_tool_calls_with_llm_disabled(self):
        """When include_history=False, tool outputs are NOT added to llm."""
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc]),
            FakeResponse(output_text="Done!", output=[_make_output_message()]),
        ]
        with patcher:
            result = await AgentWithConfig().process(input="Hi")
            llm = result["history"]
            tool_outputs = [m for m in llm if m.get("type") == "function_call_output"]
            assert len(tool_outputs) == 0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_step(self):
        patcher, mc = _patch_client()
        tc1 = _make_tool_call(name="tool_a", arguments='{"x": "h"}', call_id="c1")
        tc2 = _make_tool_call(name="tool_b", arguments='{"y": 1}', call_id="c2")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc1, tc2]),
            FakeResponse(output_text="Both done!", output=[_make_output_message()]),
        ]
        with patcher:
            result = await AgentWithMultipleTools().process(input="Hi")
            assert result["output"] == "Both done!"
            assert result["steps"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# process() — max steps exceeded
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessMaxSteps:
    @pytest.mark.asyncio
    async def test_raises_with_tokens(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
        mc.responses.create.return_value = FakeResponse(
            output_text=None, output=[tc], usage=FakeUsage(100, 200, 300))
        with patcher:
            with pytest.raises(MaxStepsExceededError) as exc_info:
                await MaxStepsAgent().process(input="Hi")
            assert exc_info.value.tokens is not None
            assert exc_info.value.tokens["input_tokens"] == 100
            assert exc_info.value.tokens["total_tokens"] == 300


# ─────────────────────────────────────────────────────────────────────────────
# process() — validation retries (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessValidationRetries:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        patcher, mc = _patch_client()
        bad = FakeResponse(output_text="bad", output=[FakeOutputItem()])
        good = FakeResponse(output_text='{"answer":"ok"}', output=[FakeOutputItem()])
        mc.responses.create.side_effect = [bad, good]
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            assert result["output_parsed"] is not None
            assert result["output_parsed"].answer == "ok"
            assert result["tokens"]["total_tokens"] == 60

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="bad", output=[FakeOutputItem()])
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            with pytest.raises(ValidationRetriesExhaustedError) as exc_info:
                await agent.process(input="test")
            assert exc_info.value.tokens is not None

    @pytest.mark.asyncio
    async def test_retry_cleanup_removes_fix_messages(self):
        patcher, mc = _patch_client()
        bad = FakeResponse(output_text="bad", output=[FakeOutputItem()])
        good = FakeResponse(output_text='{"answer":"ok"}', output=[FakeOutputItem()])
        mc.responses.create.side_effect = [bad, good]
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            llm = result["history"]
            fix_msgs = [m for m in llm if "fix the errors" in m.get("content", "")]
            assert len(fix_msgs) == 0

    @pytest.mark.asyncio
    async def test_retry_with_empty_output_text(self):
        patcher, mc = _patch_client()
        empty = FakeResponse(output_text=None, output=[FakeOutputItem()])
        good = FakeResponse(output_text='{"answer":"ok"}', output=[FakeOutputItem()])
        mc.responses.create.side_effect = [empty, good]
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            assert result["output_parsed"].answer == "ok"

    @pytest.mark.asyncio
    async def test_exhausted_preserves_last_error(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="bad", output=[FakeOutputItem()])
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=1)
            agent._client = mc
            with pytest.raises(ValidationRetriesExhaustedError) as exc_info:
                await agent.process(input="test")
            assert len(exc_info.value.validation_errors) > 0

    @pytest.mark.asyncio
    async def test_first_attempt_valid_no_retry(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text='{"answer":"direct"}', output=[FakeOutputItem()])
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            assert result["output_parsed"].answer == "direct"
            assert mc.responses.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_cleanup_with_include_history(self):
        patcher, mc = _patch_client()
        bad = FakeResponse(output_text="bad", output=[FakeOutputItem()])
        good = FakeResponse(output_text='{"answer":"ok"}', output=[FakeOutputItem()])
        mc.responses.create.side_effect = [bad, good]
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            llm = result["history"]
            assistant_msgs = [m for m in llm if m.get("role") == "assistant"]
            assert len(assistant_msgs) == 1
            assert assistant_msgs[0]["content"] == '{"answer":"ok"}'


# ─────────────────────────────────────────────────────────────────────────────
# process() — error token tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessTokenTracking:
    @pytest.mark.asyncio
    async def test_client_error_has_tokens(self):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = httpx.ConnectError("fail")
        with patcher:
            with pytest.raises(ClientError) as exc_info:
                await SimpleAgent().process(input="Hi")
            assert exc_info.value.tokens is not None
            assert exc_info.value.tokens["input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_tokens_accumulate_across_steps(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc], usage=FakeUsage(10, 20, 30)),
            FakeResponse(output_text="done", output=[_make_output_message()], usage=FakeUsage(5, 10, 15)),
        ]
        with patcher:
            result = await AgentWithTool().process(input="Hi")
            assert result["tokens"]["input_tokens"] == 15
            assert result["tokens"]["output_tokens"] == 30
            assert result["tokens"]["total_tokens"] == 45

    @pytest.mark.asyncio
    async def test_instruction_key_error_no_tokens(self):
        """InstructionKeyError is raised before the API loop, so tokens=None."""
        patcher, mc = _patch_client()
        with patcher:
            with pytest.raises(InstructionKeyError) as exc_info:
                await StrictTemplateAgent().process(input="Hi")
            assert exc_info.value.tokens is None


# ─────────────────────────────────────────────────────────────────────────────
# process() — instruction params integration
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessInstructionParams:
    @pytest.mark.asyncio
    async def test_params_applied(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            await AgentWithTemplateInstruction().process(
                input="Hi", instruction_params={"name": "Alice", "city": "NYC"})
            kw = mc.responses.create.call_args[1]
            assert "Alice" in kw["instructions"]
            assert "NYC" in kw["instructions"]


# ─────────────────────────────────────────────────────────────────────────────
# aclose
# ─────────────────────────────────────────────────────────────────────────────

class TestAclose:
    @pytest.mark.asyncio
    async def test_sets_closed(self):
        agent = SimpleAgent()
        assert agent._closed is False
        await agent.aclose()
        assert agent._closed is True

    @pytest.mark.asyncio
    async def test_idempotent(self):
        agent = SimpleAgent()
        await agent.aclose()
        await agent.aclose()
        assert agent._closed is True

    @pytest.mark.asyncio
    async def test_double_close_no_error(self):
        agent = SimpleAgent()
        await agent.aclose()
        await agent.aclose()

    @pytest.mark.asyncio
    async def test_process_after_close_raises(self):
        agent = SimpleAgent()
        await agent.aclose()
        with pytest.raises(OpenAIAgentClosedError):
            await agent.process(input="hi")


# ─────────────────────────────────────────────────────────────────────────────
# Context manager
# ─────────────────────────────────────────────────────────────────────────────

class TestContextManager:
    @pytest.mark.asyncio
    async def test_aenter_returns_agent(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            async with agent as a:
                assert a is agent

    @pytest.mark.asyncio
    async def test_aexit_closes_agent(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            async with agent:
                pass
            assert agent._closed is True

    @pytest.mark.asyncio
    async def test_aenter_on_closed_raises(self):
        agent = SimpleAgent()
        await agent.aclose()
        with pytest.raises(OpenAIAgentClosedError):
            async with agent:
                pass

    @pytest.mark.asyncio
    async def test_process_inside_context(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            async with SimpleAgent() as a:
                result = await a.process(input="Hi")
                assert result["output"] == "Hello!"

    @pytest.mark.asyncio
    async def test_process_after_context_raises(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse()
        with patcher:
            agent = SimpleAgent()
            async with agent:
                pass
            with pytest.raises(OpenAIAgentClosedError):
                await agent.process(input="Hi")

    @pytest.mark.asyncio
    async def test_exception_in_context_still_closes(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            with pytest.raises(ValueError):
                async with agent:
                    raise ValueError("test")
            assert agent._closed is True


# ─────────────────────────────────────────────────────────────────────────────
# process() — history enabled/disabled with history
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessLlmMessagesHistory:
    @pytest.mark.asyncio
    async def test_history_ignored_when_disabled(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            result = await AgentWithConfig().process(
                input="Hi", history=[{"role": "user", "content": "old"}])
            llm = result["history"]
            assert len(llm) == 1
            assert llm[0]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_history_preserved_when_enabled(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            result = await SimpleAgent().process(
                input="Hi", history=[{"role": "user", "content": "old"}])
            llm = result["history"]
            assert llm[0]["content"] == "old"
            assert llm[1]["content"] == "Hi"


# ─────────────────────────────────────────────────────────────────────────────
# process() — edge cases in event creation with tool calls
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessEventsWithTools:
    @pytest.mark.asyncio
    async def test_tool_results_in_events(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc]),
            FakeResponse(output_text="Done", output=[_make_output_message()]),
        ]
        with patcher:
            result = await AgentWithTool().process(input="Hi")
            ui = result["events"]
            assert len(ui) >= 3  # user + tool_call + tool_result + assistant

    @pytest.mark.asyncio
    async def test_text_response_uses_format_event(self):
        class CustomUiAgent(OpenAIAgent):
            """Custom UI agent."""
            def format_event(self, response) -> str:
                return "CUSTOM"

        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="original", output=[_make_output_message()])
        with patcher:
            result = await CustomUiAgent().process(input="Hi")
            ui = result["events"]
            assistant_ui = [m for m in ui if m.get("role") == "assistant"]
            assert assistant_ui[0]["content"] == "CUSTOM"

    @pytest.mark.asyncio
    async def test_text_response_llm_uses_format_history(self):
        class CustomLlmAgent(OpenAIAgent):
            """Custom LLM agent."""
            def format_history(self, response) -> str:
                return "LLM_CUSTOM"

        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="original", output=[_make_output_message()])
        with patcher:
            result = await CustomLlmAgent().process(input="Hi")
            llm = result["history"]
            assert llm[-1]["content"] == "LLM_CUSTOM"


# ─────────────────────────────────────────────────────────────────────────────
# process() — no text and no tool_calls in a single step
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessNoTextNoTools:
    @pytest.mark.asyncio
    async def test_empty_output_text_with_no_tool_calls(self):
        """Empty string output_text breaks the loop and returns empty output."""
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="", output=[FakeOutputItem()])
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            assert result["output"] == ""

    @pytest.mark.asyncio
    async def test_none_output_text_no_tools_raises(self):
        """None output_text with no tool calls raises MaxStepsExceededError."""
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text=None, output=[FakeOutputItem()])
        with patcher:
            with pytest.raises(MaxStepsExceededError):
                await SimpleAgent().process(input="Hi")

    @pytest.mark.asyncio
    async def test_history_for_empty_text_response(self):
        """Empty text adds items via model_dump in else branch."""
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="", output=[FakeOutputItem()])
        with patcher:
            result = await SimpleAgent().process(input="Hi")
            llm = result["history"]
            non_user = [m for m in llm if m.get("role") != "user"]
            assert len(non_user) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Coverage gap: ImportError fallback for to_strict_json_schema (lines 15-16)
# ─────────────────────────────────────────────────────────────────────────────

class TestToStrictJsonSchemaImportFallback:
    def test_fallback_sets_none_on_import_error(self):
        """Reload the module with a broken import to cover lines 15-16."""
        import importlib
        import sys
        import pyaiagent.openai.agent as agent_mod

        with patch.dict(sys.modules, {"openai.lib._pydantic": None}):
            importlib.reload(agent_mod)
            assert agent_mod.to_strict_json_schema is None

        # Restore module and rebind global references used by other tests
        importlib.reload(agent_mod)
        assert agent_mod.to_strict_json_schema is not None

        global OpenAIAgent, OpenAIAgentConfig
        OpenAIAgent = agent_mod.OpenAIAgent
        OpenAIAgentConfig = agent_mod.OpenAIAgentConfig


# ─────────────────────────────────────────────────────────────────────────────
# Coverage gap: _ensure_ready double-check after lock (line 141)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureReadyDoubleCheck:
    @pytest.mark.asyncio
    async def test_double_check_inside_lock(self):
        """Simulate the scenario where _ready becomes True while waiting for the lock."""
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            # Hold the lock manually
            await agent._ready_lock.acquire()

            # Start _ensure_ready — it passes fast-path (ready=False), then waits for lock
            task = asyncio.create_task(agent._ensure_ready())
            await asyncio.sleep(0)  # let it reach the lock wait

            # Simulate another coroutine having completed init
            agent._ready = True
            agent._config = OpenAIAgentConfig(**agent.__config_kwargs__)
            agent._ready_lock.release()

            await task  # hits line 141: if self._ready: return
            assert agent._ready is True


# ─────────────────────────────────────────────────────────────────────────────
# Coverage gap: CancelledError during tool execution (line 358)
# ─────────────────────────────────────────────────────────────────────────────

class TestToolCallCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_from_tool(self):
        class CancelAgent(OpenAIAgent):
            """Agent with cancellable tool."""
            async def cancel_me(self) -> dict:
                """A tool that gets cancelled."""
                raise asyncio.CancelledError()

        patcher, _ = _patch_client()
        with patcher:
            agent = CancelAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="cancel_me", arguments="{}", call_id="c_cancel")
            with pytest.raises(asyncio.CancelledError):
                await agent._execute_tool_call(tc)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage gap: aclose double-check after lock (line 630)
# ─────────────────────────────────────────────────────────────────────────────

class TestAcloseDoubleCheck:
    @pytest.mark.asyncio
    async def test_double_check_inside_lock(self):
        """Simulate the scenario where _closed becomes True while waiting for the lock."""
        agent = SimpleAgent()
        # Hold the lock manually
        await agent._ready_lock.acquire()

        # Start aclose — it passes fast-path (closed=False), then waits for lock
        task = asyncio.create_task(agent.aclose())
        await asyncio.sleep(0)  # let it reach the lock wait

        # Simulate another coroutine having closed
        agent._closed = True
        agent._ready_lock.release()

        await task  # hits line 630: if self._closed: return
        assert agent._closed is True


# ─────────────────────────────────────────────────────────────────────────────
# Coverage gap: __init_subclass__ cls is OpenAIAgent guard (line 74)
# ─────────────────────────────────────────────────────────────────────────────

class TestInitSubclassBaseGuard:
    def test_base_class_skipped(self):
        """Directly invoke __init_subclass__ with cls=OpenAIAgent to hit the guard."""
        import pyaiagent.openai.agent as agent_mod
        current_cls = agent_mod.OpenAIAgent
        current_cls.__init_subclass__()
        assert not hasattr(current_cls, "__agent_name__")


# ─────────────────────────────────────────────────────────────────────────────
# Additional edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentReuse:
    """Calling process() multiple times on the same agent instance."""

    @pytest.mark.asyncio
    async def test_two_calls_same_agent(self):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = [
            FakeResponse(output_text="First"),
            FakeResponse(output_text="Second"),
        ]
        with patcher:
            agent = SimpleAgent()
            r1 = await agent.process(input="a")
            r2 = await agent.process(input="b")
            assert r1["output"] == "First"
            assert r2["output"] == "Second"
            assert r1["turn"] != r2["turn"]

    @pytest.mark.asyncio
    async def test_reuse_does_not_leak_state(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="Reply")
        with patcher:
            agent = SimpleAgent()
            r1 = await agent.process(input="x", session="s1")
            r2 = await agent.process(input="y", session="s2")
            assert r1["session"] == "s1"
            assert r2["session"] == "s2"
            assert r1["input"] == "x"
            assert r2["input"] == "y"


class TestStrictTemplateSuccess:
    """Strict mode with all placeholders supplied."""

    def test_strict_with_all_params_succeeds(self):
        agent = StrictTemplateAgent()
        result = agent._format_instruction(instruction_params={"name": "Alice"})
        assert result == "Hello Alice."

    def test_strict_with_extra_params_succeeds(self):
        agent = StrictTemplateAgent()
        result = agent._format_instruction(instruction_params={"name": "Bob", "extra": "ignored"})
        assert result == "Hello Bob."


class TestOutputParsedFromParsePath:
    """output_parsed populated via responses.parse() (no validation_retries)."""

    @pytest.mark.asyncio
    async def test_output_parsed_returned_from_parse(self):
        patcher, mc = _patch_client()
        fake_parsed = SimpleOutput(answer="parsed_value")
        parse_resp = FakeResponse(output_text='{"answer":"parsed_value"}', output=[FakeOutputItem()])
        parse_resp.output_parsed = fake_parsed
        mc.responses.parse.return_value = parse_resp
        with patcher:
            agent, _ = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=0)
            agent._client = mc
            result = await agent.process(input="test")
            assert result["output_parsed"] is fake_parsed
            assert result["output_parsed"].answer == "parsed_value"


class TestSyncToolException:
    """Sync tool raising an exception (not async)."""

    @pytest.mark.asyncio
    async def test_sync_tool_exception_returns_error(self, caplog):
        class SyncFailAgent(OpenAIAgent):
            """Agent with failing sync tool."""
            def broken(self) -> dict:
                """A broken sync tool."""
                raise RuntimeError("sync boom")

        patcher, _ = _patch_client()
        with patcher:
            agent = SyncFailAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="broken", arguments="{}", call_id="c_sync_fail")
            with caplog.at_level(logging.ERROR):
                result = await agent._execute_tool_call(tc)
            assert "Tool execution failed" in result["output"]
            assert "Tool execution failed: broken" in caplog.text


class TestNoneAndBoolInput:
    """process() with None or bool input raises InvalidInputError."""

    @pytest.mark.asyncio
    async def test_none_input_raises(self):
        with pytest.raises(InvalidInputError):
            await SimpleAgent().process(input=None)

    @pytest.mark.asyncio
    async def test_bool_input_raises(self):
        with pytest.raises(InvalidInputError):
            await SimpleAgent().process(input=True)

    @pytest.mark.asyncio
    async def test_list_input_raises(self):
        with pytest.raises(InvalidInputError):
            await SimpleAgent().process(input=["hi"])

    @pytest.mark.asyncio
    async def test_empty_string_input_succeeds(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(output_text="ok")
        with patcher:
            result = await SimpleAgent().process(input="")
            assert result["input"] == ""


class TestAdditionalTypeValidation:
    """Edge-case types for metadata, history, instruction_params."""

    @pytest.mark.asyncio
    async def test_list_metadata_raises(self):
        with pytest.raises(InvalidMetadataError):
            await SimpleAgent().process(input="hi", metadata=[1, 2])

    @pytest.mark.asyncio
    async def test_tuple_history_raises(self):
        with pytest.raises(InvalidHistoryError):
            await SimpleAgent().process(input="hi", history=({"role": "user"},))

    @pytest.mark.asyncio
    async def test_int_history_raises(self):
        with pytest.raises(InvalidHistoryError):
            await SimpleAgent().process(input="hi", history=42)

    @pytest.mark.asyncio
    async def test_list_instruction_params_raises(self):
        with pytest.raises(InvalidInstructionParamsError):
            await SimpleAgent().process(input="hi", instruction_params=["a", "b"])

    @pytest.mark.asyncio
    async def test_int_metadata_raises(self):
        with pytest.raises(InvalidMetadataError):
            await SimpleAgent().process(input="hi", metadata=123)

    @pytest.mark.asyncio
    async def test_bool_session_raises(self):
        with pytest.raises(InvalidSessionError):
            await SimpleAgent().process(input="hi", session=True)


class TestPydanticAndGenericErrorLogging:
    """Verify logging for Pydantic and generic exception paths."""

    @pytest.mark.asyncio
    async def test_pydantic_error_logged(self, caplog):
        patcher, mc = _patch_client()
        mc.responses.parse.side_effect = PydanticValidationError.from_exception_data(
            title="test", line_errors=[])
        with patcher:
            agent, _ = await _ready_agent_for_validation(SimpleAgent, SimpleOutput, validation_retries=0)
            agent._client = mc
            with caplog.at_level(logging.ERROR), pytest.raises(ClientError):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])
            assert "Structured output validation" in caplog.text

    @pytest.mark.asyncio
    async def test_generic_error_logged(self, caplog):
        patcher, mc = _patch_client()
        mc.responses.create.side_effect = RuntimeError("kaboom")
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            with caplog.at_level(logging.ERROR), pytest.raises(ClientError):
                await agent._openai_responses_api_call(
                    instruction="test", current_turn_history=[])
            assert "Unexpected error" in caplog.text


class TestEventsResponseOutputMessageInList:
    """ResponseOutputMessage inside a list for _create_events."""

    @pytest.mark.asyncio
    async def test_list_of_output_messages(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            msgs = [_make_output_message("a"), _make_output_message("b")]
            result = agent._create_events(
                data=msgs, session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata=None)
            assert len(result) == 2
            assert all(r["type"] == "message" for r in result)


class TestMultiStepToolChain:
    """Tool→tool→text: 3 API calls, 3 steps."""

    @pytest.mark.asyncio
    async def test_three_step_tool_chain(self):
        patcher, mc = _patch_client()
        tc1 = _make_tool_call(name="greet", arguments='{"name":"A"}', call_id="c1")
        tc2 = _make_tool_call(name="greet", arguments='{"name":"B"}', call_id="c2")
        mc.responses.create.side_effect = [
            FakeResponse(output_text=None, output=[tc1], usage=FakeUsage(10, 10, 20)),
            FakeResponse(output_text=None, output=[tc2], usage=FakeUsage(10, 10, 20)),
            FakeResponse(output_text="All done", output=[_make_output_message()], usage=FakeUsage(5, 5, 10)),
        ]
        with patcher:
            result = await AgentWithTool().process(input="Do both")
            assert result["output"] == "All done"
            assert result["steps"] == 3
            assert result["tokens"]["input_tokens"] == 25
            assert result["tokens"]["total_tokens"] == 50


class TestValidationRetryFallbackErrorMessage:
    """When response has no validation_error attr, fallback message is used."""

    @pytest.mark.asyncio
    async def test_exhausted_uses_fallback_message(self):
        patcher, mc = _patch_client()
        resp = FakeResponse(output_text=None, output=[FakeOutputItem()])
        mc.responses.create.return_value = resp
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=1)
            agent._client = mc
            with pytest.raises(ValidationRetriesExhaustedError) as exc_info:
                await agent.process(input="test")
            assert "output could not be parsed" in exc_info.value.validation_errors


class TestStaticKwargsExplicit:
    """Verify tool_choice, temperature, max_output_tokens in static kwargs."""

    @pytest.mark.asyncio
    async def test_temperature_included(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert "temperature" in agent._static_openai_responses_api_kwargs

    @pytest.mark.asyncio
    async def test_tool_choice_included(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert "tool_choice" in agent._static_openai_responses_api_kwargs

    @pytest.mark.asyncio
    async def test_max_output_tokens_included(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            assert "max_output_tokens" in agent._static_openai_responses_api_kwargs

    @pytest.mark.asyncio
    async def test_config_values_match(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = AgentWithConfig()
            await agent._ensure_ready()
            kw = agent._static_openai_responses_api_kwargs
            assert kw["temperature"] == 0.5
            assert kw["max_output_tokens"] == 2048
            assert kw["model"] == "gpt-4o"


class TestDuplicatePlaceholder:
    """Multiple occurrences of the same placeholder in instruction."""

    def test_duplicate_placeholder_replaced(self):
        class RepeatAgent(OpenAIAgent):
            """Say {word} and {word} again."""
        agent = RepeatAgent()
        result = agent._format_instruction(instruction_params={"word": "hello"})
        assert result == "Say hello and hello again."

    def test_duplicate_placeholder_strict(self):
        class StrictRepeatAgent(OpenAIAgent):
            """Say {word} and {word}."""
            class Config:
                strict_instruction_params = True
        agent = StrictRepeatAgent()
        result = agent._format_instruction(instruction_params={"word": "hi"})
        assert result == "Say hi and hi."


class TestSyncToolCancelledError:
    """CancelledError from sync tool running via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_sync_cancelled_error_propagates(self):
        class SyncCancelAgent(OpenAIAgent):
            """Agent with sync tool that cancels."""
            def sync_cancel(self) -> dict:
                """A sync tool that gets cancelled."""
                raise asyncio.CancelledError()

        patcher, _ = _patch_client()
        with patcher:
            agent = SyncCancelAgent()
            await agent._ensure_ready()
            tc = _make_tool_call(name="sync_cancel", arguments="{}", call_id="c_sc")
            with pytest.raises(asyncio.CancelledError):
                await agent._execute_tool_call(tc)


# ─────────────────────────────────────────────────────────────────────────────
# Validation retry cleanup with history_disabled
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationRetryCleanupLlmDisabled:
    """When validation retries succeed but include_history=False,
    cleanup deletes artifacts but does NOT re-add the assistant message."""

    @pytest.mark.asyncio
    async def test_cleanup_skips_assistant_readd_when_llm_disabled(self):
        class LlmDisabledAgent(OpenAIAgent):
            """Return structured output."""
            class Config:
                include_history = False

        patcher, mc = _patch_client()
        bad = FakeResponse(output_text="bad", output=[FakeOutputItem()])
        good = FakeResponse(output_text='{"answer":"ok"}', output=[FakeOutputItem()])
        mc.responses.create.side_effect = [bad, good]
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                LlmDisabledAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            llm = result["history"]
            assistant_msgs = [m for m in llm if m.get("role") == "assistant"]
            assert len(assistant_msgs) == 0
            assert result["output_parsed"].answer == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# No placeholders in instruction but params provided
# ─────────────────────────────────────────────────────────────────────────────

class TestNoPlaceholdersWithParams:
    """Params passed to an instruction with no placeholders are ignored."""

    def test_params_ignored_when_no_placeholders(self):
        agent = SimpleAgent()
        result = agent._format_instruction(
            instruction_params={"name": "Alice", "city": "NY"})
        assert result == "You are a helpful assistant."


# ─────────────────────────────────────────────────────────────────────────────
# Two validation failures then success on third attempt
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoFailuresThenSuccess:
    """Two validation failures followed by a third-attempt success.
    Exercises history_checkpoint already set (second retry skips checkpoint init)
    and cleanup removing multiple retry artifacts."""

    @pytest.mark.asyncio
    async def test_two_failures_then_success(self):
        patcher, mc = _patch_client()
        bad = FakeResponse(output_text="bad", output=[FakeOutputItem()])
        good = FakeResponse(output_text='{"answer":"ok"}', output=[FakeOutputItem()])
        mc.responses.create.side_effect = [bad, bad, good]
        with patcher:
            agent, _ = await _ready_agent_for_validation(
                SimpleAgent, SimpleOutput, validation_retries=2)
            agent._client = mc
            result = await agent.process(input="test")
            assert result["output_parsed"].answer == "ok"
            assert mc.responses.create.call_count == 3
            llm = result["history"]
            fix_msgs = [m for m in llm if "fix the errors" in m.get("content", "")]
            assert len(fix_msgs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Both format_history and format_event overridden simultaneously
# ─────────────────────────────────────────────────────────────────────────────

class TestBothFormatOverrides:
    """When both format_history and format_event are overridden,
    they produce independent content in history and events."""

    @pytest.mark.asyncio
    async def test_both_overrides_independent(self):
        class DualFormatAgent(OpenAIAgent):
            """Dual format agent."""
            def format_history(self, response):
                return "LLM:" + (response.output_text or "")
            def format_event(self, response):
                return "UI:" + (response.output_text or "")

        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="hello", output=[_make_output_message("hello")])
        with patcher:
            result = await DualFormatAgent().process(input="test")
            llm = result["history"]
            ui = result["events"]
            assistant_llm = [m for m in llm if m.get("role") == "assistant"]
            assert assistant_llm[0]["content"] == "LLM:hello"
            assistant_ui = [m for m in ui if m.get("role") == "assistant"]
            assert assistant_ui[0]["content"] == "UI:hello"


# ─────────────────────────────────────────────────────────────────────────────
# Tool call response with non-None output_text (tools take priority)
# ─────────────────────────────────────────────────────────────────────────────

class TestToolCallWithOutputText:
    """When a response has both tool calls and output_text, tool calls
    take priority — the agent continues to tool execution, not text break."""

    @pytest.mark.asyncio
    async def test_tool_calls_take_priority_over_text(self):
        patcher, mc = _patch_client()
        tc = _make_tool_call(name="greet", arguments='{"name": "X"}', call_id="c1")
        mc.responses.create.side_effect = [
            FakeResponse(output_text="also some text", output=[tc]),
            FakeResponse(output_text="Done!", output=[_make_output_message()]),
        ]
        with patcher:
            result = await AgentWithTool().process(input="Hi")
            assert result["output"] == "Done!"
            assert result["steps"] == 2
            assert mc.responses.create.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# List of all-unsupported items in _create_events
# ─────────────────────────────────────────────────────────────────────────────

class TestAllUnsupportedItemsList:
    """A list containing only unsupported types yields empty events."""

    @pytest.mark.asyncio
    async def test_list_of_only_unsupported_types(self):
        patcher, _ = _patch_client()
        with patcher:
            agent = SimpleAgent()
            await agent._ensure_ready()
            result = agent._create_events(
                data=[12345, 3.14, True],
                session="s", turn="t", step=1,
                input_tokens=0, output_tokens=0, total_tokens=0,
                metadata={})
            assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# process() with empty list as history
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyListLlmMessages:
    """Empty list history is shallow-copied (different code path than None)."""

    @pytest.mark.asyncio
    async def test_empty_list_history(self):
        patcher, mc = _patch_client()
        mc.responses.create.return_value = FakeResponse(
            output_text="Hi!", output=[_make_output_message()])
        with patcher:
            original = []
            result = await SimpleAgent().process(input="test", history=original)
            assert len(original) == 0, "Caller's list must not be mutated"
            llm = result["history"]
            assert any(m.get("role") == "user" for m in llm)
            assert any(m.get("role") == "assistant" for m in llm)
