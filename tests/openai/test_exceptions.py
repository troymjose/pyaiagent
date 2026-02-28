"""
Unit tests for pyaiagent.openai.exceptions.

Covers: hierarchy, __slots__, agent_name attribute, tokens default,
        message formatting, and validation_errors preservation.
"""
import pytest

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

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hierarchy
# ──────────────────────────────────────────────────────────────────────────────

class TestHierarchy:

    def test_definition_error_is_type_error(self):
        assert issubclass(OpenAIAgentDefinitionError, TypeError)

    def test_process_error_is_exception(self):
        assert issubclass(OpenAIAgentProcessError, Exception)

    @pytest.mark.parametrize("cls", [
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
    ])
    def test_all_process_errors_inherit_from_base(self, cls):
        assert issubclass(cls, OpenAIAgentProcessError)

    def test_catch_all_process_errors_with_base(self):
        for cls_factory in [
            lambda: OpenAIAgentClosedError(),
            lambda: InvalidInputError(),
            lambda: InvalidSessionError(),
            lambda: InvalidMetadataError(),
            lambda: InvalidHistoryError(),
            lambda: InvalidInstructionParamsError(),
            lambda: InstructionKeyError(agent_name="A", key="k"),
            lambda: ClientError(agent_name="A", message="fail"),
            lambda: MaxStepsExceededError(),
            lambda: ValidationRetriesExhaustedError(),
        ]:
            exc = cls_factory()
            assert isinstance(exc, OpenAIAgentProcessError)
            assert isinstance(exc, Exception)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Slots
# ──────────────────────────────────────────────────────────────────────────────

class TestSlots:

    def test_definition_error_has_slots(self):
        assert hasattr(OpenAIAgentDefinitionError, "__slots__")

    def test_process_error_has_slots(self):
        assert "agent_name" in OpenAIAgentProcessError.__slots__
        assert "tokens" in OpenAIAgentProcessError.__slots__

    def test_validation_retries_has_slots(self):
        assert "validation_errors" in ValidationRetriesExhaustedError.__slots__

    @pytest.mark.parametrize("cls", [
        OpenAIAgentClosedError,
        InvalidInputError,
        InvalidSessionError,
        InvalidMetadataError,
        InvalidHistoryError,
        InvalidInstructionParamsError,
        InstructionKeyError,
        ClientError,
        MaxStepsExceededError,
    ])
    def test_subclass_slots_empty(self, cls):
        assert cls.__slots__ == ()

    def test_slots_declared_attrs_accessible(self):
        exc = OpenAIAgentProcessError("test")
        assert hasattr(exc, "agent_name")
        assert hasattr(exc, "tokens")

    def test_validation_errors_slot_accessible(self):
        exc = ValidationRetriesExhaustedError(errors="bad")
        assert hasattr(exc, "validation_errors")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Agent name attribute
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentName:

    def test_default_agent_name(self):
        exc = InvalidInputError()
        assert exc.agent_name == "Agent"

    def test_custom_agent_name(self):
        exc = InvalidInputError(agent_name="MyBot")
        assert exc.agent_name == "MyBot"

    @pytest.mark.parametrize("cls,kwargs", [
        (OpenAIAgentClosedError, {}),
        (InvalidInputError, {}),
        (InvalidSessionError, {}),
        (InvalidMetadataError, {}),
        (InvalidHistoryError, {}),
        (InvalidInstructionParamsError, {}),
        (InstructionKeyError, {"key": "x"}),
        (ClientError, {"message": "err"}),
        (MaxStepsExceededError, {}),
        (ValidationRetriesExhaustedError, {}),
    ])
    def test_agent_name_present_on_all(self, cls, kwargs):
        exc = cls(agent_name="TestAgent", **kwargs)
        assert exc.agent_name == "TestAgent"

    def test_base_process_error_agent_name(self):
        exc = OpenAIAgentProcessError("msg", agent_name="Base")
        assert exc.agent_name == "Base"

    def test_base_process_error_default_agent_name(self):
        exc = OpenAIAgentProcessError("msg")
        assert exc.agent_name == "Agent"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Tokens
# ──────────────────────────────────────────────────────────────────────────────

class TestTokens:

    def test_tokens_default_none(self):
        exc = OpenAIAgentProcessError("test")
        assert exc.tokens is None

    def test_tokens_default_on_subclass(self):
        exc = ClientError(agent_name="A", message="fail")
        assert exc.tokens is None

    def test_tokens_assignable(self):
        exc = ClientError(agent_name="A", message="fail")
        exc.tokens = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        assert exc.tokens["input_tokens"] == 10
        assert exc.tokens["output_tokens"] == 20
        assert exc.tokens["total_tokens"] == 30

    def test_tokens_independent_across_instances(self):
        exc1 = InvalidInputError()
        exc2 = InvalidInputError()
        exc1.tokens = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        assert exc2.tokens is None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Message formatting
# ──────────────────────────────────────────────────────────────────────────────

class TestMessageFormatting:

    def test_closed_error_message(self):
        exc = OpenAIAgentClosedError(agent_name="Bot")
        assert str(exc) == "Bot: agent is closed, create a new instance"

    def test_invalid_input_message(self):
        exc = InvalidInputError(agent_name="Bot", received="int")
        assert str(exc) == "Bot: 'input' must be str, not int"

    def test_invalid_session_message(self):
        exc = InvalidSessionError(agent_name="Bot", received="list")
        assert str(exc) == "Bot: 'session' must be a non-empty str, not list"

    def test_invalid_metadata_message(self):
        exc = InvalidMetadataError(agent_name="Bot", received="str")
        assert str(exc) == "Bot: 'metadata' must be dict, not str"

    def test_invalid_history_message(self):
        exc = InvalidHistoryError(agent_name="Bot", received="dict")
        assert str(exc) == "Bot: 'history' must be list, not dict"

    def test_invalid_instruction_params_message(self):
        exc = InvalidInstructionParamsError(agent_name="Bot", received="list")
        assert str(exc) == "Bot: 'instruction_params' must be dict, not list"

    def test_instruction_key_error_message(self):
        exc = InstructionKeyError(agent_name="Bot", key="user_name")
        assert str(exc) == "Bot: missing instruction key 'user_name'"

    def test_client_error_message(self):
        exc = ClientError(agent_name="Bot", message="rate limited")
        assert str(exc) == "Bot: OpenAI API error - rate limited"

    def test_max_steps_message(self):
        exc = MaxStepsExceededError(agent_name="Bot", max_steps=25)
        assert str(exc) == "Bot: exceeded 25 steps without completing"

    def test_validation_retries_message(self):
        exc = ValidationRetriesExhaustedError(
            agent_name="Bot", validation_retries=3, errors="field required")
        msg = str(exc)
        assert "Bot:" in msg
        assert "3 retry attempt(s)" in msg
        assert "field required" in msg

    def test_default_messages_use_default_agent_name(self):
        exc = InvalidInputError()
        assert "Agent:" in str(exc)

    def test_default_received_is_unknown(self):
        exc = InvalidInputError()
        assert "not unknown" in str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# 6. DefinitionError
# ──────────────────────────────────────────────────────────────────────────────

class TestDefinitionError:

    def test_single_error(self):
        exc = OpenAIAgentDefinitionError(cls_name="MyAgent", errors=["bad field"])
        msg = str(exc)
        assert "'MyAgent' has 1 definition error:" in msg
        assert "• bad field" in msg

    def test_multiple_errors(self):
        exc = OpenAIAgentDefinitionError(
            cls_name="MyAgent", errors=["error one", "error two", "error three"])
        msg = str(exc)
        assert "'MyAgent' has 3 definition errors:" in msg
        assert "• error one" in msg
        assert "• error two" in msg
        assert "• error three" in msg

    def test_no_leading_newline(self):
        exc = OpenAIAgentDefinitionError(cls_name="X", errors=["e"])
        assert not str(exc).startswith("\n")

    def test_is_type_error(self):
        exc = OpenAIAgentDefinitionError(cls_name="X", errors=["e"])
        assert isinstance(exc, TypeError)


# ──────────────────────────────────────────────────────────────────────────────
# 7. ValidationRetriesExhaustedError specifics
# ──────────────────────────────────────────────────────────────────────────────

class TestValidationRetriesExhausted:

    def test_validation_errors_stored(self):
        exc = ValidationRetriesExhaustedError(errors="name: required")
        assert exc.validation_errors == "name: required"

    def test_validation_errors_default_empty(self):
        exc = ValidationRetriesExhaustedError()
        assert exc.validation_errors == ""

    def test_validation_retries_in_message(self):
        exc = ValidationRetriesExhaustedError(validation_retries=5)
        assert "5 retry attempt(s)" in str(exc)

    def test_zero_retries_in_message(self):
        exc = ValidationRetriesExhaustedError(validation_retries=0)
        assert "0 retry attempt(s)" in str(exc)
