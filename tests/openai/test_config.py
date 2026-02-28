"""
Comprehensive unit tests for pyaiagent.openai.config.

Covers: OpenAIAgentConfig dataclass, _is_json_safe, all 17 validators,
        _create_config_kwargs (including descriptor filtering),
        _validate_config_kwargs, and resolve().
"""
import math
from typing import Any, Dict
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from pyaiagent.openai.config import (
    OpenAIAgentConfig,
    ConfigResolver,
    _is_json_safe,
    _ALLOWED_FIELDS,
    _MAX_JSON_DEPTH,
)
from pyaiagent.openai.exceptions import OpenAIAgentDefinitionError


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate(field: str, value: Any) -> list[str]:
    """Run a single field validator and return collected errors."""
    errors: list[str] = []
    validator = getattr(ConfigResolver, f"_validate_{field}")
    validator(value, errors)
    return errors


def _no_errors(field: str, value: Any) -> None:
    assert _validate(field, value) == [], f"Unexpected errors for {field}={value!r}"


def _has_error(field: str, value: Any, match: str = "") -> None:
    errors = _validate(field, value)
    assert errors, f"Expected errors for {field}={value!r}"
    if match:
        assert any(match in e for e in errors), f"No error matching '{match}' in {errors}"


# ──────────────────────────────────────────────────────────────────────────────
# 1. OpenAIAgentConfig dataclass
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIAgentConfig:

    def test_defaults(self):
        cfg = OpenAIAgentConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.2
        assert cfg.top_p is None
        assert cfg.seed is None
        assert cfg.max_output_tokens == 4096
        assert cfg.tool_choice == "auto"
        assert cfg.parallel_tool_calls is True
        assert cfg.text_format is None
        assert cfg.max_steps == 10
        assert cfg.max_parallel_tools == 10
        assert cfg.tool_timeout == 30.0
        assert cfg.llm_timeout == 120.0
        assert cfg.include_events is True
        assert cfg.include_history is True
        assert cfg.strict_instruction_params is False
        assert cfg.validation_retries == 0

    def test_frozen(self):
        cfg = OpenAIAgentConfig()
        with pytest.raises(AttributeError):
            cfg.model = "other"

    def test_custom_values(self):
        cfg = OpenAIAgentConfig(model="gpt-4o", temperature=1.0, max_steps=5)
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 1.0
        assert cfg.max_steps == 5

    def test_slots(self):
        assert hasattr(OpenAIAgentConfig, "__slots__")

    def test_field_count_matches_allowed(self):
        from dataclasses import fields as dc_fields
        assert {f.name for f in dc_fields(OpenAIAgentConfig)} == _ALLOWED_FIELDS


# ──────────────────────────────────────────────────────────────────────────────
# 2. _is_json_safe
# ──────────────────────────────────────────────────────────────────────────────

class TestIsJsonSafe:

    # --- Scalars ---

    def test_string(self):
        assert _is_json_safe("hello") is True

    def test_int(self):
        assert _is_json_safe(42) is True

    def test_bool_true(self):
        assert _is_json_safe(True) is True

    def test_bool_false(self):
        assert _is_json_safe(False) is True

    def test_none(self):
        assert _is_json_safe(None) is True

    def test_float_finite(self):
        assert _is_json_safe(3.14) is True

    def test_float_zero(self):
        assert _is_json_safe(0.0) is True

    def test_float_negative(self):
        assert _is_json_safe(-1.5) is True

    def test_float_nan(self):
        assert _is_json_safe(float("nan")) is False

    def test_float_inf(self):
        assert _is_json_safe(float("inf")) is False

    def test_float_neg_inf(self):
        assert _is_json_safe(float("-inf")) is False

    # --- Unsupported types ---

    def test_set_rejected(self):
        assert _is_json_safe({1, 2}) is False

    def test_tuple_rejected(self):
        assert _is_json_safe((1, 2)) is False

    def test_object_rejected(self):
        assert _is_json_safe(object()) is False

    def test_bytes_rejected(self):
        assert _is_json_safe(b"data") is False

    def test_class_rejected(self):
        class Foo: pass
        assert _is_json_safe(Foo) is False

    # --- Lists ---

    def test_empty_list(self):
        assert _is_json_safe([]) is True

    def test_list_of_scalars(self):
        assert _is_json_safe([1, "a", True, None, 3.14]) is True

    def test_list_with_unsafe_item(self):
        assert _is_json_safe([1, object()]) is False

    def test_nested_list(self):
        assert _is_json_safe([[1, 2], [3, 4]]) is True

    def test_list_with_nan(self):
        assert _is_json_safe([1.0, float("nan")]) is False

    # --- Dicts ---

    def test_empty_dict(self):
        assert _is_json_safe({}) is True

    def test_dict_str_keys_scalar_values(self):
        assert _is_json_safe({"a": 1, "b": "c", "d": None}) is True

    def test_dict_non_str_key(self):
        assert _is_json_safe({1: "a"}) is False

    def test_dict_unsafe_value(self):
        assert _is_json_safe({"a": object()}) is False

    def test_nested_dict(self):
        assert _is_json_safe({"a": {"b": {"c": 1}}}) is True

    def test_dict_with_list_value(self):
        assert _is_json_safe({"a": [1, 2, 3]}) is True

    def test_dict_with_nan_value(self):
        assert _is_json_safe({"a": float("nan")}) is False

    # --- Depth ---

    def test_depth_at_limit(self):
        value: Any = "leaf"
        for _ in range(_MAX_JSON_DEPTH):
            value = [value]
        assert _is_json_safe(value) is True

    def test_depth_exceeds_limit(self):
        value: Any = "leaf"
        for _ in range(_MAX_JSON_DEPTH + 2):
            value = [value]
        assert _is_json_safe(value) is False

    def test_depth_exceeds_via_dict(self):
        value: Any = "leaf"
        for _ in range(_MAX_JSON_DEPTH + 2):
            value = {"k": value}
        assert _is_json_safe(value) is False


# ──────────────────────────────────────────────────────────────────────────────
# 3. Validators — model
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateModel:

    def test_valid_model(self):
        _no_errors("model", "gpt-4o-mini")

    def test_empty_string(self):
        _has_error("model", "", "cannot be empty")

    def test_whitespace_only(self):
        _has_error("model", "   ", "cannot be empty")

    def test_leading_whitespace(self):
        _has_error("model", " gpt-4o", "whitespace")

    def test_trailing_whitespace(self):
        _has_error("model", "gpt-4o ", "whitespace")

    def test_not_string(self):
        _has_error("model", 123, "must be a string")

    def test_none(self):
        _has_error("model", None, "must be a string")

    def test_bool(self):
        _has_error("model", True, "must be a string")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Validators — temperature
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateTemperature:

    def test_valid_float(self):
        _no_errors("temperature", 0.5)

    def test_valid_int(self):
        _no_errors("temperature", 1)

    def test_zero(self):
        _no_errors("temperature", 0.0)

    def test_max_boundary(self):
        _no_errors("temperature", 2.0)

    def test_below_zero(self):
        _has_error("temperature", -0.1, "between 0.0 and 2.0")

    def test_above_max(self):
        _has_error("temperature", 2.1, "between 0.0 and 2.0")

    def test_nan(self):
        _has_error("temperature", float("nan"), "finite")

    def test_inf(self):
        _has_error("temperature", float("inf"), "finite")

    def test_neg_inf(self):
        _has_error("temperature", float("-inf"), "finite")

    def test_bool_rejected(self):
        _has_error("temperature", True, "must be a float or int")

    def test_string_rejected(self):
        _has_error("temperature", "0.5", "must be a float or int")

    def test_none_rejected(self):
        _has_error("temperature", None, "must be a float or int")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Validators — top_p
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateTopP:

    def test_none_valid(self):
        _no_errors("top_p", None)

    def test_valid_float(self):
        _no_errors("top_p", 0.9)

    def test_valid_int_zero(self):
        _no_errors("top_p", 0)

    def test_valid_int_one(self):
        _no_errors("top_p", 1)

    def test_zero_float(self):
        _no_errors("top_p", 0.0)

    def test_one_float(self):
        _no_errors("top_p", 1.0)

    def test_below_zero(self):
        _has_error("top_p", -0.1, "between 0.0 and 1.0")

    def test_above_one(self):
        _has_error("top_p", 1.1, "between 0.0 and 1.0")

    def test_nan(self):
        _has_error("top_p", float("nan"), "finite")

    def test_inf(self):
        _has_error("top_p", float("inf"), "finite")

    def test_bool_rejected(self):
        _has_error("top_p", True, "must be a float, int, or None")

    def test_string_rejected(self):
        _has_error("top_p", "0.5", "must be a float, int, or None")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Validators — seed
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateSeed:

    def test_none_valid(self):
        _no_errors("seed", None)

    def test_valid_int(self):
        _no_errors("seed", 42)

    def test_zero(self):
        _no_errors("seed", 0)

    def test_negative(self):
        _no_errors("seed", -1)

    def test_bool_rejected(self):
        _has_error("seed", True, "must be an int or None")

    def test_float_rejected(self):
        _has_error("seed", 1.5, "must be an int or None")

    def test_string_rejected(self):
        _has_error("seed", "42", "must be an int or None")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Validators — max_output_tokens
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateMaxOutputTokens:

    def test_valid(self):
        _no_errors("max_output_tokens", 4096)

    def test_one(self):
        _no_errors("max_output_tokens", 1)

    def test_zero_rejected(self):
        _has_error("max_output_tokens", 0, "must be > 0")

    def test_negative_rejected(self):
        _has_error("max_output_tokens", -1, "must be > 0")

    def test_bool_rejected(self):
        _has_error("max_output_tokens", True, "must be an int")

    def test_float_rejected(self):
        _has_error("max_output_tokens", 100.0, "must be an int")

    def test_string_rejected(self):
        _has_error("max_output_tokens", "4096", "must be an int")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Validators — tool_choice
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateToolChoice:

    def test_auto(self):
        _no_errors("tool_choice", "auto")

    def test_none_str(self):
        _no_errors("tool_choice", "none")

    def test_required(self):
        _no_errors("tool_choice", "required")

    def test_invalid_string(self):
        _has_error("tool_choice", "invalid", "must be one of")

    def test_leading_whitespace(self):
        _has_error("tool_choice", " auto", "whitespace")

    def test_trailing_whitespace(self):
        _has_error("tool_choice", "auto ", "whitespace")

    def test_valid_dict(self):
        _no_errors("tool_choice", {"type": "function", "function": {"name": "search"}})

    def test_empty_dict(self):
        _has_error("tool_choice", {}, "cannot be empty")

    def test_dict_not_json_safe(self):
        _has_error("tool_choice", {"a": object()}, "JSON-serializable")

    def test_dict_with_nan(self):
        _has_error("tool_choice", {"a": float("nan")}, "JSON-serializable")

    def test_not_str_or_dict(self):
        _has_error("tool_choice", 42, "must be a string or dict")

    def test_none_rejected(self):
        _has_error("tool_choice", None, "must be a string or dict")

    def test_bool_rejected(self):
        _has_error("tool_choice", True, "must be a string or dict")

    def test_list_rejected(self):
        _has_error("tool_choice", ["auto"], "must be a string or dict")

    def test_valid_nested_dict(self):
        _no_errors("tool_choice", {"type": "function", "function": {"name": "x", "args": [1, 2]}})


# ──────────────────────────────────────────────────────────────────────────────
# 9. Validators — parallel_tool_calls
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateParallelToolCalls:

    def test_true(self):
        _no_errors("parallel_tool_calls", True)

    def test_false(self):
        _no_errors("parallel_tool_calls", False)

    def test_int_rejected(self):
        _has_error("parallel_tool_calls", 1, "must be a bool")

    def test_string_rejected(self):
        _has_error("parallel_tool_calls", "true", "must be a bool")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Validators — text_format
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateTextFormat:

    def test_none_valid(self):
        _no_errors("text_format", None)

    def test_pydantic_class_valid(self):
        class MyModel(BaseModel):
            name: str
        _no_errors("text_format", MyModel)

    def test_pydantic_instance_rejected(self):
        class MyModel(BaseModel):
            name: str
        _has_error("text_format", MyModel(name="x"), "Pydantic model class")

    def test_non_pydantic_class_rejected(self):
        class Plain:
            pass
        _has_error("text_format", Plain, "Pydantic model class")

    def test_string_rejected(self):
        _has_error("text_format", "MyModel", "Pydantic model class")

    def test_dict_rejected(self):
        _has_error("text_format", {"name": "str"}, "Pydantic model class")

    def test_pydantic_import_error(self):
        """Covers the except ImportError branch when pydantic is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pydantic":
                raise ImportError("No module named 'pydantic'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            errors: list[str] = []
            ConfigResolver._validate_text_format("not_none", errors)
            assert len(errors) == 1
            assert "requires pydantic" in errors[0]


# ──────────────────────────────────────────────────────────────────────────────
# 11. Validators — max_steps
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateMaxSteps:

    def test_valid(self):
        _no_errors("max_steps", 10)

    def test_one(self):
        _no_errors("max_steps", 1)

    def test_zero_rejected(self):
        _has_error("max_steps", 0, "must be > 0")

    def test_negative_rejected(self):
        _has_error("max_steps", -5, "must be > 0")

    def test_bool_rejected(self):
        _has_error("max_steps", True, "must be an int")

    def test_float_rejected(self):
        _has_error("max_steps", 10.0, "must be an int")


# ──────────────────────────────────────────────────────────────────────────────
# 12. Validators — max_parallel_tools
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateMaxParallelTools:

    def test_valid(self):
        _no_errors("max_parallel_tools", 5)

    def test_one(self):
        _no_errors("max_parallel_tools", 1)

    def test_zero_rejected(self):
        _has_error("max_parallel_tools", 0, "must be > 0")

    def test_negative_rejected(self):
        _has_error("max_parallel_tools", -1, "must be > 0")

    def test_bool_rejected(self):
        _has_error("max_parallel_tools", False, "must be an int")

    def test_string_rejected(self):
        _has_error("max_parallel_tools", "10", "must be an int")


# ──────────────────────────────────────────────────────────────────────────────
# 13. Validators — tool_timeout
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateToolTimeout:

    def test_valid_float(self):
        _no_errors("tool_timeout", 30.0)

    def test_valid_int(self):
        _no_errors("tool_timeout", 30)

    def test_zero_rejected(self):
        _has_error("tool_timeout", 0, "must be > 0")

    def test_negative_rejected(self):
        _has_error("tool_timeout", -1.0, "must be > 0")

    def test_nan_rejected(self):
        _has_error("tool_timeout", float("nan"), "finite")

    def test_inf_rejected(self):
        _has_error("tool_timeout", float("inf"), "finite")

    def test_bool_rejected(self):
        _has_error("tool_timeout", True, "must be a number")

    def test_string_rejected(self):
        _has_error("tool_timeout", "30", "must be a number")


# ──────────────────────────────────────────────────────────────────────────────
# 14. Validators — llm_timeout
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateLlmTimeout:

    def test_valid_float(self):
        _no_errors("llm_timeout", 120.0)

    def test_valid_int(self):
        _no_errors("llm_timeout", 60)

    def test_zero_rejected(self):
        _has_error("llm_timeout", 0, "must be > 0")

    def test_negative_rejected(self):
        _has_error("llm_timeout", -5.0, "must be > 0")

    def test_nan_rejected(self):
        _has_error("llm_timeout", float("nan"), "finite")

    def test_inf_rejected(self):
        _has_error("llm_timeout", float("inf"), "finite")

    def test_bool_rejected(self):
        _has_error("llm_timeout", False, "must be a number")

    def test_none_rejected(self):
        _has_error("llm_timeout", None, "must be a number")


# ──────────────────────────────────────────────────────────────────────────────
# 15. Validators — bool fields
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateBoolFields:

    @pytest.mark.parametrize("field", [
        "include_events",
        "include_history",
        "strict_instruction_params",
    ])
    def test_true_valid(self, field):
        _no_errors(field, True)

    @pytest.mark.parametrize("field", [
        "include_events",
        "include_history",
        "strict_instruction_params",
    ])
    def test_false_valid(self, field):
        _no_errors(field, False)

    @pytest.mark.parametrize("field", [
        "include_events",
        "include_history",
        "strict_instruction_params",
    ])
    def test_int_rejected(self, field):
        _has_error(field, 1, "must be a bool")

    @pytest.mark.parametrize("field", [
        "include_events",
        "include_history",
        "strict_instruction_params",
    ])
    def test_string_rejected(self, field):
        _has_error(field, "true", "must be a bool")


# ──────────────────────────────────────────────────────────────────────────────
# 16. Validators — validation_retries
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateValidationRetries:

    def test_zero_valid(self):
        _no_errors("validation_retries", 0)

    def test_max_valid(self):
        _no_errors("validation_retries", ConfigResolver._MAX_VALIDATION_RETRIES)

    def test_mid_range(self):
        _no_errors("validation_retries", 3)

    def test_negative_rejected(self):
        _has_error("validation_retries", -1, "between 0 and")

    def test_above_max_rejected(self):
        _has_error("validation_retries", ConfigResolver._MAX_VALIDATION_RETRIES + 1, "between 0 and")

    def test_bool_rejected(self):
        _has_error("validation_retries", True, "must be an int")

    def test_float_rejected(self):
        _has_error("validation_retries", 3.0, "must be an int")

    def test_string_rejected(self):
        _has_error("validation_retries", "3", "must be an int")


# ──────────────────────────────────────────────────────────────────────────────
# 17. _validate_config_kwargs
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateConfigKwargs:

    def test_empty_kwargs_no_errors(self):
        assert ConfigResolver._validate_config_kwargs({}) == []

    def test_valid_kwargs(self):
        assert ConfigResolver._validate_config_kwargs({"model": "gpt-4o", "temperature": 0.5}) == []

    def test_unknown_key(self):
        errors = ConfigResolver._validate_config_kwargs({"unknown_field": 42})
        assert len(errors) == 1
        assert "Unknown attribute" in errors[0]

    def test_multiple_unknown_keys(self):
        errors = ConfigResolver._validate_config_kwargs({"foo": 1, "bar": 2})
        assert len(errors) == 2

    def test_mixed_valid_and_invalid(self):
        errors = ConfigResolver._validate_config_kwargs({"model": 123, "temperature": 0.5})
        assert len(errors) == 1
        assert "model" in errors[0]

    def test_multiple_validation_errors(self):
        errors = ConfigResolver._validate_config_kwargs({
            "model": 123,
            "temperature": "hot",
            "max_steps": -1,
        })
        assert len(errors) == 3

    def test_all_valid_fields(self):
        kwargs = {
            "model": "gpt-4o",
            "temperature": 0.5,
            "top_p": 0.9,
            "seed": 42,
            "max_output_tokens": 1024,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "text_format": None,
            "max_steps": 10,
            "max_parallel_tools": 5,
            "tool_timeout": 30.0,
            "llm_timeout": 120.0,
            "include_events": True,
            "include_history": True,
            "strict_instruction_params": False,
            "validation_retries": 3,
        }
        assert ConfigResolver._validate_config_kwargs(kwargs) == []

    def test_every_allowed_field_has_validator(self):
        for field_name in _ALLOWED_FIELDS:
            validator = getattr(ConfigResolver, f"_validate_{field_name}", None)
            assert validator is not None, f"Missing validator for field '{field_name}'"

    def test_missing_validator_library_error(self):
        """Covers the 'Library error: missing validator' branch."""
        with patch.object(ConfigResolver, "_validate_model", None, create=False):
            original = ConfigResolver.__dict__.get("_validate_model")
            try:
                delattr(ConfigResolver, "_validate_model")
                errors = ConfigResolver._validate_config_kwargs({"model": "gpt-4o"})
                assert len(errors) == 1
                assert "Library error" in errors[0]
                assert "missing validator" in errors[0]
            finally:
                if original is not None:
                    ConfigResolver._validate_model = original


# ──────────────────────────────────────────────────────────────────────────────
# 18. _create_config_kwargs
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateConfigKwargs:

    def test_no_inner_no_parent(self):
        result = ConfigResolver._create_config_kwargs(None, None)
        assert result == {}

    def test_no_inner_with_parent(self):
        parent = {"model": "gpt-4o", "temperature": 0.5}
        result = ConfigResolver._create_config_kwargs(None, parent)
        assert result == {"model": "gpt-4o", "temperature": 0.5}

    def test_parent_is_copied(self):
        parent = {"model": "gpt-4o"}
        result = ConfigResolver._create_config_kwargs(None, parent)
        result["model"] = "changed"
        assert parent["model"] == "gpt-4o"

    def test_empty_parent_dict(self):
        result = ConfigResolver._create_config_kwargs(None, {})
        assert result == {}

    def test_inner_only(self):
        class Config:
            model = "gpt-4o"
            temperature = 1.0
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert result == {"model": "gpt-4o", "temperature": 1.0}

    def test_inner_overrides_parent(self):
        class Config:
            model = "gpt-4o"
        parent = {"model": "gpt-4o-mini", "temperature": 0.5}
        result = ConfigResolver._create_config_kwargs(Config, parent)
        assert result["model"] == "gpt-4o"
        assert result["temperature"] == 0.5

    # --- Filtering ---

    def test_private_attrs_skipped(self):
        class Config:
            model = "gpt-4o"
            _internal = "hidden"
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "_internal" not in result
        assert result == {"model": "gpt-4o"}

    def test_dunder_attrs_skipped(self):
        class Config:
            model = "gpt-4o"
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "__dict__" not in result
        assert "__module__" not in result

    def test_callable_skipped(self):
        class Config:
            model = "gpt-4o"
            def helper(self): pass
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "helper" not in result
        assert result == {"model": "gpt-4o"}

    def test_lambda_skipped(self):
        class Config:
            model = "gpt-4o"
            compute = lambda: 42
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "compute" not in result

    def test_staticmethod_skipped(self):
        class Config:
            model = "gpt-4o"
            @staticmethod
            def helper(): pass
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "helper" not in result

    def test_classmethod_skipped(self):
        class Config:
            model = "gpt-4o"
            @classmethod
            def helper(cls): pass
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "helper" not in result

    def test_property_skipped(self):
        class Config:
            @property
            def model(self):
                return "gpt-4o"
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "model" not in result

    def test_custom_descriptor_skipped(self):
        class MyDescriptor:
            def __get__(self, obj, objtype=None):
                return "value"
        class Config:
            model = "gpt-4o"
            custom = MyDescriptor()
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "custom" not in result
        assert result == {"model": "gpt-4o"}

    def test_data_descriptor_skipped(self):
        class DataDesc:
            def __get__(self, obj, objtype=None): return "x"
            def __set__(self, obj, value): pass
        class Config:
            model = "gpt-4o"
            desc = DataDesc()
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "desc" not in result

    def test_plain_data_types_accepted(self):
        class Config:
            str_val = "hello"
            int_val = 42
            float_val = 3.14
            bool_val = True
            none_val = None
            list_val = [1, 2]
            dict_val = {"a": 1}
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert result["str_val"] == "hello"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["none_val"] is None
        assert result["list_val"] == [1, 2]
        assert result["dict_val"] == {"a": 1}


# ──────────────────────────────────────────────────────────────────────────────
# 19. resolve()
# ──────────────────────────────────────────────────────────────────────────────

class TestResolve:

    def test_class_without_config(self):
        class Agent:
            pass
        result = ConfigResolver.resolve(Agent)
        assert result == {}

    def test_class_with_valid_config(self):
        class Agent:
            class Config:
                model = "gpt-4o"
                temperature = 1.0
        result = ConfigResolver.resolve(Agent)
        assert result["model"] == "gpt-4o"
        assert result["temperature"] == 1.0

    def test_class_with_invalid_config_raises(self):
        class Agent:
            class Config:
                model = 123
        with pytest.raises(OpenAIAgentDefinitionError, match="model"):
            ConfigResolver.resolve(Agent)

    def test_class_with_unknown_field_raises(self):
        class Agent:
            class Config:
                nonexistent = "value"
        with pytest.raises(OpenAIAgentDefinitionError, match="Unknown attribute"):
            ConfigResolver.resolve(Agent)

    def test_error_includes_class_name(self):
        class MyCustomAgent:
            class Config:
                model = 123
        with pytest.raises(OpenAIAgentDefinitionError, match="MyCustomAgent"):
            ConfigResolver.resolve(MyCustomAgent)

    def test_parent_config_kwargs_merged(self):
        class Agent:
            __config_kwargs__ = {"model": "gpt-4o-mini", "temperature": 0.2}
            class Config:
                temperature = 1.0
        result = ConfigResolver.resolve(Agent)
        assert result["model"] == "gpt-4o-mini"
        assert result["temperature"] == 1.0

    def test_parent_without_inner(self):
        class Agent:
            __config_kwargs__ = {"model": "gpt-4o", "max_steps": 5}
        result = ConfigResolver.resolve(Agent)
        assert result["model"] == "gpt-4o"
        assert result["max_steps"] == 5

    def test_multiple_errors_reported(self):
        class Agent:
            class Config:
                model = 123
                temperature = "hot"
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            ConfigResolver.resolve(Agent)
        msg = str(exc_info.value)
        assert "model" in msg
        assert "temperature" in msg

    def test_result_can_instantiate_config(self):
        class Agent:
            class Config:
                model = "gpt-4o"
                temperature = 0.8
                max_steps = 5
        kwargs = ConfigResolver.resolve(Agent)
        cfg = OpenAIAgentConfig(**kwargs)
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.8
        assert cfg.max_steps == 5

    def test_property_in_config_ignored(self):
        class Agent:
            class Config:
                temperature = 0.5
                @property
                def model(self):
                    return "gpt-4o"
        result = ConfigResolver.resolve(Agent)
        assert "model" not in result
        assert result["temperature"] == 0.5

    def test_empty_config_class(self):
        class Agent:
            class Config:
                pass
        result = ConfigResolver.resolve(Agent)
        assert result == {}

    def test_text_format_class_filtered_by_callable_check(self):
        """Classes are callable, so text_format=<class> in inner Config is
        filtered by the callable(value) guard. text_format must be supplied
        via parent __config_kwargs__ instead."""
        class MyOutput(BaseModel):
            answer: str

        class Agent:
            class Config:
                text_format = MyOutput
        result = ConfigResolver.resolve(Agent)
        assert "text_format" not in result

    def test_text_format_via_parent_config_kwargs(self):
        class MyOutput(BaseModel):
            answer: str

        class Agent:
            __config_kwargs__ = {"text_format": MyOutput}
        result = ConfigResolver.resolve(Agent)
        assert result["text_format"] is MyOutput

    def test_all_non_callable_fields_valid(self):
        class Agent:
            class Config:
                model = "gpt-4o"
                temperature = 0.5
                top_p = 0.9
                seed = 42
                max_output_tokens = 2048
                tool_choice = "required"
                parallel_tool_calls = False
                max_steps = 20
                max_parallel_tools = 5
                tool_timeout = 60.0
                llm_timeout = 180.0
                include_events = False
                include_history = False
                strict_instruction_params = True
                validation_retries = 3

        result = ConfigResolver.resolve(Agent)
        assert len(result) == 15
        cfg = OpenAIAgentConfig(**result)
        assert cfg.model == "gpt-4o"
        assert cfg.validation_retries == 3

    def test_all_fields_via_parent_kwargs(self):
        """All 16 fields including text_format via __config_kwargs__."""
        class MyOutput(BaseModel):
            answer: str

        class Agent:
            __config_kwargs__ = {
                "model": "gpt-4o",
                "temperature": 0.5,
                "top_p": 0.9,
                "seed": 42,
                "max_output_tokens": 2048,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "text_format": MyOutput,
                "max_steps": 20,
                "max_parallel_tools": 5,
                "tool_timeout": 60.0,
                "llm_timeout": 180.0,
                "include_events": False,
                "include_history": False,
                "strict_instruction_params": True,
                "validation_retries": 3,
            }
        result = ConfigResolver.resolve(Agent)
        assert len(result) == 16
        cfg = OpenAIAgentConfig(**result)
        assert cfg.model == "gpt-4o"
        assert cfg.text_format is MyOutput


# ──────────────────────────────────────────────────────────────────────────────
# 20. Edge cases and additional hardening
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Additional edge-case and boundary tests for full hardening."""

    # --- _is_json_safe edge cases ---

    def test_large_int(self):
        assert _is_json_safe(10**100) is True

    def test_negative_int(self):
        assert _is_json_safe(-999) is True

    def test_empty_string(self):
        assert _is_json_safe("") is True

    def test_mixed_list_and_dict_nesting(self):
        assert _is_json_safe({"a": [{"b": [1, 2]}, "c"]}) is True

    def test_list_with_dict_with_non_str_key(self):
        assert _is_json_safe([{1: "a"}]) is False

    def test_deeply_nested_mixed(self):
        value: Any = "leaf"
        for i in range(5):
            value = {"k": [value]} if i % 2 == 0 else [{"k": value}]
        assert _is_json_safe(value) is True

    def test_frozenset_rejected(self):
        assert _is_json_safe(frozenset([1, 2])) is False

    def test_complex_number_rejected(self):
        assert _is_json_safe(1 + 2j) is False

    def test_callable_rejected(self):
        assert _is_json_safe(lambda: 1) is False

    def test_dict_with_bool_key_rejected(self):
        assert _is_json_safe({True: "val"}) is False

    def test_dict_with_none_key_rejected(self):
        assert _is_json_safe({None: "val"}) is False

    # --- Temperature int boundary ---

    def test_temperature_int_zero(self):
        _no_errors("temperature", 0)

    def test_temperature_int_two(self):
        _no_errors("temperature", 2)

    def test_temperature_int_negative(self):
        _has_error("temperature", -1, "between 0.0 and 2.0")

    def test_temperature_int_above_max(self):
        _has_error("temperature", 3, "between 0.0 and 2.0")

    # --- top_p int boundary ---

    def test_top_p_negative_int(self):
        _has_error("top_p", -1, "between 0.0 and 1.0")

    def test_top_p_int_two(self):
        _has_error("top_p", 2, "between 0.0 and 1.0")

    def test_top_p_neg_inf(self):
        _has_error("top_p", float("-inf"), "finite")

    # --- seed large values ---

    def test_seed_large_positive(self):
        _no_errors("seed", 2**63)

    def test_seed_large_negative(self):
        _no_errors("seed", -(2**63))

    # --- max_output_tokens large value ---

    def test_max_output_tokens_large(self):
        _no_errors("max_output_tokens", 10**9)

    # --- tool_choice edge cases ---

    def test_tool_choice_empty_string(self):
        _has_error("tool_choice", "", "must be one of")

    def test_tool_choice_case_sensitive(self):
        _has_error("tool_choice", "Auto", "must be one of")

    def test_tool_choice_dict_nested_nan(self):
        _has_error("tool_choice", {"a": {"b": float("nan")}}, "JSON-serializable")

    def test_tool_choice_dict_nested_object(self):
        _has_error("tool_choice", {"a": [object()]}, "JSON-serializable")

    def test_tool_choice_dict_deep_nesting(self):
        d: Any = "leaf"
        for _ in range(_MAX_JSON_DEPTH + 2):
            d = {"k": d}
        _has_error("tool_choice", d, "JSON-serializable")

    # --- tool_timeout / llm_timeout int edge ---

    def test_tool_timeout_neg_inf(self):
        _has_error("tool_timeout", float("-inf"), "finite")

    def test_llm_timeout_neg_inf(self):
        _has_error("llm_timeout", float("-inf"), "finite")

    def test_tool_timeout_very_large(self):
        _no_errors("tool_timeout", 1e6)

    def test_llm_timeout_very_large(self):
        _no_errors("llm_timeout", 1e6)

    # --- validation_retries boundary ---

    def test_validation_retries_one(self):
        _no_errors("validation_retries", 1)

    def test_validation_retries_exactly_max_minus_one(self):
        _no_errors("validation_retries", ConfigResolver._MAX_VALIDATION_RETRIES - 1)

    def test_validation_retries_none_rejected(self):
        _has_error("validation_retries", None, "must be an int")

    # --- _create_config_kwargs: various descriptor types ---

    def test_non_data_descriptor_skipped(self):
        class NonDataDescriptor:
            def __get__(self, obj, objtype=None):
                return "sneaky"
        class Config:
            temperature = 0.5
            sneaky = NonDataDescriptor()
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "sneaky" not in result

    def test_class_attribute_that_is_a_class_skipped(self):
        """A class is callable, so it should be filtered."""
        class Inner: pass
        class Config:
            temperature = 0.5
            nested_cls = Inner
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "nested_cls" not in result

    def test_multiple_inner_fields(self):
        class Config:
            model = "gpt-4o"
            temperature = 1.0
            max_steps = 20
            tool_choice = "required"
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert result == {
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_steps": 20,
            "tool_choice": "required",
        }

    def test_parent_and_inner_disjoint_keys(self):
        class Config:
            temperature = 1.5
        parent = {"model": "gpt-4o"}
        result = ConfigResolver._create_config_kwargs(Config, parent)
        assert result == {"model": "gpt-4o", "temperature": 1.5}

    # --- resolve edge cases ---

    def test_resolve_with_none_config_attr(self):
        class Agent:
            Config = None
        result = ConfigResolver.resolve(Agent)
        assert result == {}

    def test_resolve_unknown_plus_invalid_errors_accumulated(self):
        class Agent:
            class Config:
                model = 123
                nonexistent = True
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            ConfigResolver.resolve(Agent)
        msg = str(exc_info.value)
        assert "model" in msg
        assert "Unknown attribute" in msg

    def test_resolve_inherits_parent_then_validates(self):
        """Invalid parent values should also be caught."""
        class Agent:
            __config_kwargs__ = {"temperature": "not_a_number"}
        with pytest.raises(OpenAIAgentDefinitionError, match="temperature"):
            ConfigResolver.resolve(Agent)

    def test_resolve_error_is_type_error(self):
        class Agent:
            class Config:
                model = 123
        with pytest.raises(TypeError):
            ConfigResolver.resolve(Agent)

    def test_tool_choice_both_whitespace(self):
        _has_error("tool_choice", "  auto  ", "whitespace")

    def test_model_tab_whitespace(self):
        _has_error("model", "\tgpt-4o", "whitespace")

    # --- Validator independence: each error is independent ---

    def test_two_different_type_errors(self):
        errors = ConfigResolver._validate_config_kwargs({
            "model": [],
            "seed": 1.5,
        })
        assert len(errors) == 2
        assert any("model" in e for e in errors)
        assert any("seed" in e for e in errors)

    # --- int fields: float close to int not accepted ---

    def test_max_steps_float_close_to_int(self):
        _has_error("max_steps", 10.0, "must be an int")

    def test_max_parallel_tools_float_close_to_int(self):
        _has_error("max_parallel_tools", 5.0, "must be an int")

    def test_max_output_tokens_float_close_to_int(self):
        _has_error("max_output_tokens", 4096.0, "must be an int")

    # --- bool is subclass of int: explicitly rejected everywhere ---

    def test_seed_false_rejected(self):
        _has_error("seed", False, "must be an int or None")

    def test_max_output_tokens_false_rejected(self):
        _has_error("max_output_tokens", False, "must be an int")

    def test_max_steps_false_rejected(self):
        _has_error("max_steps", False, "must be an int")

    def test_max_parallel_tools_true_rejected(self):
        _has_error("max_parallel_tools", True, "must be an int")

    def test_validation_retries_false_rejected(self):
        _has_error("validation_retries", False, "must be an int")

    # --- _validate_config_kwargs: only known keys reach validators ---

    def test_unknown_key_with_valid_value_still_error(self):
        errors = ConfigResolver._validate_config_kwargs({"foo": "bar"})
        assert len(errors) == 1
        assert "Unknown attribute" in errors[0]

    def test_unknown_key_error_contains_url(self):
        errors = ConfigResolver._validate_config_kwargs({"xyz": 1})
        assert any("github.com" in e for e in errors)

    # --- OpenAIAgentConfig: equality ---

    def test_config_equality(self):
        a = OpenAIAgentConfig()
        b = OpenAIAgentConfig()
        assert a == b

    def test_config_inequality(self):
        a = OpenAIAgentConfig(model="gpt-4o")
        b = OpenAIAgentConfig(model="gpt-4o-mini")
        assert a != b

    # --- _ALLOWED_FIELDS constant ---

    def test_allowed_fields_is_set(self):
        assert isinstance(_ALLOWED_FIELDS, set)

    def test_allowed_fields_not_empty(self):
        assert len(_ALLOWED_FIELDS) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 21. Missing gap cases
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleExportsAndConstants:
    """Verify module-level constants and __all__ are correct."""

    def test_all_exports(self):
        from pyaiagent.openai import config as mod
        assert "OpenAIAgentConfig" in mod.__all__
        assert "ConfigResolver" in mod.__all__

    def test_valid_tool_choice_strings(self):
        assert ConfigResolver._VALID_TOOL_CHOICE_STRINGS == frozenset({"auto", "none", "required"})

    def test_max_validation_retries_value(self):
        assert ConfigResolver._MAX_VALIDATION_RETRIES == 10

    def test_max_json_depth_value(self):
        assert _MAX_JSON_DEPTH == 10

    def test_json_scalars_tuple(self):
        from pyaiagent.openai.config import _JSON_SCALARS
        assert str in _JSON_SCALARS
        assert int in _JSON_SCALARS
        assert float in _JSON_SCALARS
        assert bool in _JSON_SCALARS
        assert type(None) in _JSON_SCALARS


class TestIsJsonSafeAdditional:
    """Gaps in _is_json_safe not covered by the main section."""

    def test_bool_inside_list(self):
        assert _is_json_safe([True, False]) is True

    def test_bool_inside_dict_value(self):
        assert _is_json_safe({"a": True, "b": False}) is True

    def test_int_zero(self):
        assert _is_json_safe(0) is True

    def test_float_inside_dict(self):
        assert _is_json_safe({"pi": 3.14}) is True

    def test_nan_inside_nested_list(self):
        assert _is_json_safe([[float("nan")]]) is False

    def test_inf_inside_dict_in_list(self):
        assert _is_json_safe([{"v": float("inf")}]) is False

    def test_dict_with_int_key_inside_list(self):
        assert _is_json_safe([{42: "v"}]) is False

    def test_mixed_safe_and_unsafe_in_dict_short_circuits(self):
        assert _is_json_safe({"a": 1, "b": object()}) is False

    def test_mixed_safe_and_unsafe_in_list_short_circuits(self):
        assert _is_json_safe([1, 2, object(), 4]) is False

    def test_single_element_list(self):
        assert _is_json_safe(["x"]) is True

    def test_single_key_dict(self):
        assert _is_json_safe({"k": "v"}) is True

    def test_depth_exactly_at_boundary_with_dict(self):
        value: Any = "leaf"
        for _ in range(_MAX_JSON_DEPTH):
            value = {"k": value}
        assert _is_json_safe(value) is True

    def test_depth_one_over_boundary_with_dict(self):
        value: Any = "leaf"
        for _ in range(_MAX_JSON_DEPTH + 1):
            value = {"k": value}
        assert _is_json_safe(value) is False

    def test_list_of_dicts(self):
        assert _is_json_safe([{"a": 1}, {"b": 2}]) is True

    def test_dict_of_lists(self):
        assert _is_json_safe({"a": [1, 2], "b": [3, 4]}) is True


class TestValidateModelAdditional:

    def test_newline_whitespace(self):
        _has_error("model", "\ngpt-4o", "whitespace")

    def test_carriage_return_whitespace(self):
        _has_error("model", "gpt-4o\r", "whitespace")

    def test_tab_and_space_mixed(self):
        _has_error("model", " \tgpt-4o\t ", "whitespace")

    def test_single_char_model(self):
        _no_errors("model", "a")

    def test_long_model_name(self):
        _no_errors("model", "gpt-4o-2024-11-20-preview")

    def test_model_with_list_value(self):
        _has_error("model", ["gpt-4o"], "must be a string")

    def test_model_with_dict_value(self):
        _has_error("model", {"name": "gpt-4o"}, "must be a string")


class TestValidateTemperatureAdditional:

    def test_epsilon_above_zero(self):
        _no_errors("temperature", 0.001)

    def test_epsilon_below_max(self):
        _no_errors("temperature", 1.999)

    def test_list_rejected(self):
        _has_error("temperature", [0.5], "must be a float or int")

    def test_dict_rejected(self):
        _has_error("temperature", {"v": 0.5}, "must be a float or int")

    def test_int_in_range_passes_nan_check(self):
        """Ints bypass the float NaN/Inf check and go directly to range check."""
        _no_errors("temperature", 1)
        _has_error("temperature", 3, "between 0.0 and 2.0")


class TestValidateTopPAdditional:

    def test_midpoint(self):
        _no_errors("top_p", 0.5)

    def test_list_rejected(self):
        _has_error("top_p", [0.5], "must be a float, int, or None")

    def test_int_range_check_path(self):
        """Int values skip the float NaN/Inf check, hit range check directly."""
        _no_errors("top_p", 0)
        _has_error("top_p", -1, "between 0.0 and 1.0")


class TestValidateTextFormatAdditional:

    def test_int_rejected(self):
        _has_error("text_format", 42, "Pydantic model class")

    def test_bool_rejected(self):
        _has_error("text_format", True, "Pydantic model class")

    def test_list_rejected(self):
        _has_error("text_format", [BaseModel], "Pydantic model class")

    def test_pydantic_subclass_of_subclass_valid(self):
        class Parent(BaseModel):
            x: int
        class Child(Parent):
            y: str
        _no_errors("text_format", Child)

    def test_error_message_contains_hint(self):
        class MyModel(BaseModel):
            name: str
        errors = _validate("text_format", MyModel(name="x"))
        assert any("Hint" in e for e in errors)

    def test_error_message_shows_actual_type(self):
        errors = _validate("text_format", 42)
        assert any("int" in e for e in errors)


class TestValidateToolTimeoutIntPath:
    """Specifically test the int-through-timeout validators branch."""

    def test_tool_timeout_negative_int(self):
        _has_error("tool_timeout", -5, "must be > 0")

    def test_llm_timeout_negative_int(self):
        _has_error("llm_timeout", -10, "must be > 0")

    def test_tool_timeout_int_one(self):
        _no_errors("tool_timeout", 1)

    def test_llm_timeout_int_one(self):
        _no_errors("llm_timeout", 1)


class TestValidateToolChoiceAdditional:

    def test_whitespace_only_string(self):
        """Strips to empty string, caught by whitespace check."""
        _has_error("tool_choice", "   ", "whitespace")

    def test_valid_dict_with_list_value(self):
        _no_errors("tool_choice", {"names": ["a", "b", "c"]})

    def test_valid_dict_with_null_value(self):
        _no_errors("tool_choice", {"type": None})

    def test_valid_dict_single_key(self):
        _no_errors("tool_choice", {"type": "function"})

    def test_dict_with_inf_value(self):
        _has_error("tool_choice", {"v": float("inf")}, "JSON-serializable")

    def test_tuple_rejected(self):
        _has_error("tool_choice", ("auto",), "must be a string or dict")

    def test_set_rejected(self):
        _has_error("tool_choice", {"auto"}, "must be a string or dict")


class TestCreateConfigKwargsAdditional:

    def test_tuple_attribute_accepted(self):
        """Tuples are not callable and don't have __get__, so they pass filters."""
        class Config:
            some_tuple = (1, 2, 3)
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert result["some_tuple"] == (1, 2, 3)

    def test_frozenset_attribute_accepted(self):
        class Config:
            some_set = frozenset({1, 2})
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert result["some_set"] == frozenset({1, 2})

    def test_descriptor_does_not_shadow_parent_kwarg(self):
        """If parent has 'model' and inner Config has a property 'model',
        the property is skipped so parent's value remains."""
        class Config:
            @property
            def model(self):
                return "overridden"
        parent = {"model": "gpt-4o"}
        result = ConfigResolver._create_config_kwargs(Config, parent)
        assert result["model"] == "gpt-4o"

    def test_inner_config_with_only_filtered_attrs(self):
        class Config:
            _private = "x"
            @property
            def model(self): return "y"
            def helper(self): pass
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert result == {}

    def test_inner_class_inherits_from_object(self):
        """Plain inner class — only its own vars are iterated, not object's."""
        class Config:
            temperature = 0.5
        result = ConfigResolver._create_config_kwargs(Config, None)
        assert "temperature" in result
        assert len([k for k in result if not k.startswith("_")]) == 1


class TestResolveAdditional:

    def test_return_type_is_plain_dict(self):
        class Agent:
            class Config:
                model = "gpt-4o"
        result = ConfigResolver.resolve(Agent)
        assert type(result) is dict

    def test_resolve_error_header_format(self):
        """Verify bullet-point format in the error message."""
        class Agent:
            class Config:
                model = 123
                temperature = "bad"
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            ConfigResolver.resolve(Agent)
        msg = str(exc_info.value)
        assert "2 definition errors" in msg
        assert "  \u2022 " in msg

    def test_resolve_single_error_no_plural(self):
        class Agent:
            class Config:
                model = 123
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            ConfigResolver.resolve(Agent)
        msg = str(exc_info.value)
        assert "1 definition error:" in msg
        assert "errors:" not in msg

    def test_resolve_with_no_config_no_parent(self):
        class PlainClass:
            pass
        result = ConfigResolver.resolve(PlainClass)
        assert result == {}
        assert type(result) is dict

    def test_resolve_class_with_config_attr_that_is_int(self):
        """Config attr is not a class; getattr returns int, which is truthy.
        _create_config_kwargs gets inner_config_cls=42, then vars(42) is called."""
        class Agent:
            Config = 42
        with pytest.raises((TypeError, OpenAIAgentDefinitionError)):
            ConfigResolver.resolve(Agent)

    def test_resolve_parent_invalid_unknown_key(self):
        """Unknown key in parent __config_kwargs__ should be caught."""
        class Agent:
            __config_kwargs__ = {"nonexistent": 42}
        with pytest.raises(OpenAIAgentDefinitionError, match="Unknown attribute"):
            ConfigResolver.resolve(Agent)

    def test_inner_overrides_parent_in_resolve(self):
        class Agent:
            __config_kwargs__ = {"model": "parent-model", "max_steps": 5}
            class Config:
                model = "child-model"
        result = ConfigResolver.resolve(Agent)
        assert result["model"] == "child-model"
        assert result["max_steps"] == 5

    def test_resolve_result_does_not_mutate_parent(self):
        parent = {"model": "gpt-4o"}
        class Agent:
            __config_kwargs__ = parent
            class Config:
                temperature = 0.5
        result = ConfigResolver.resolve(Agent)
        assert "temperature" in result
        assert "temperature" not in parent


class TestValidateConfigKwargsAdditional:

    def test_single_valid_field_passes(self):
        assert ConfigResolver._validate_config_kwargs({"seed": 42}) == []

    def test_single_valid_none_field(self):
        assert ConfigResolver._validate_config_kwargs({"top_p": None}) == []

    def test_unknown_key_skips_to_next_not_validator(self):
        """Unknown key should NOT run any validator — only append error."""
        errors = ConfigResolver._validate_config_kwargs({
            "unknown_xyz": "value",
            "model": "gpt-4o",
        })
        assert len(errors) == 1
        assert "Unknown" in errors[0]

    def test_error_accumulation_with_many_fields(self):
        errors = ConfigResolver._validate_config_kwargs({
            "model": 123,
            "temperature": "x",
            "top_p": "y",
            "seed": 1.5,
            "max_output_tokens": False,
        })
        assert len(errors) == 5


# ──────────────────────────────────────────────────────────────────────────────
# 22. None input for validators that don't explicitly handle it
# ──────────────────────────────────────────────────────────────────────────────

class TestNoneRejectedByValidators:
    """None is a valid Python value that can sneak in; verify type name in error."""

    def test_parallel_tool_calls_none(self):
        errors = _validate("parallel_tool_calls", None)
        assert len(errors) == 1
        assert "NoneType" in errors[0]

    def test_max_output_tokens_none(self):
        errors = _validate("max_output_tokens", None)
        assert len(errors) == 1
        assert "NoneType" in errors[0]

    def test_max_steps_none(self):
        errors = _validate("max_steps", None)
        assert len(errors) == 1
        assert "NoneType" in errors[0]

    def test_max_parallel_tools_none(self):
        errors = _validate("max_parallel_tools", None)
        assert len(errors) == 1
        assert "NoneType" in errors[0]

    def test_tool_timeout_none(self):
        errors = _validate("tool_timeout", None)
        assert len(errors) == 1
        assert "NoneType" in errors[0]


# ──────────────────────────────────────────────────────────────────────────────
# 23. Error message content — suggestions and type names
# ──────────────────────────────────────────────────────────────────────────────

class TestErrorMessageContent:
    """Verify that error messages contain useful context beyond just the field name."""

    # --- Whitespace suggestion content ---

    def test_model_whitespace_suggests_stripped(self):
        errors = _validate("model", " gpt-4o ")
        assert len(errors) == 1
        assert "Use 'gpt-4o' instead" in errors[0]

    def test_model_tab_whitespace_suggests_stripped(self):
        errors = _validate("model", "\tgpt-4o\t")
        assert len(errors) == 1
        assert "Use 'gpt-4o' instead" in errors[0]

    def test_tool_choice_whitespace_suggests_stripped(self):
        errors = _validate("tool_choice", " auto ")
        assert len(errors) == 1
        assert "Use 'auto' instead" in errors[0]

    def test_tool_choice_trailing_whitespace_suggests_stripped(self):
        errors = _validate("tool_choice", "required ")
        assert len(errors) == 1
        assert "Use 'required' instead" in errors[0]

    # --- Type names in error messages ---

    def test_model_error_shows_list_type(self):
        errors = _validate("model", [1, 2])
        assert "got list" in errors[0]

    def test_model_error_shows_nonetype(self):
        errors = _validate("model", None)
        assert "got NoneType" in errors[0]

    def test_temperature_error_shows_str_type(self):
        errors = _validate("temperature", "hot")
        assert "got str" in errors[0]

    def test_temperature_error_shows_bool_type(self):
        errors = _validate("temperature", False)
        assert "got bool" in errors[0]

    def test_top_p_error_shows_bool_type(self):
        errors = _validate("top_p", True)
        assert "got bool" in errors[0]

    def test_seed_error_shows_float_type(self):
        errors = _validate("seed", 3.14)
        assert "got float" in errors[0]

    def test_max_output_tokens_error_shows_bool_type(self):
        errors = _validate("max_output_tokens", True)
        assert "got bool" in errors[0]

    def test_tool_choice_error_shows_int_type(self):
        errors = _validate("tool_choice", 42)
        assert "got int" in errors[0]

    def test_parallel_tool_calls_error_shows_int_type(self):
        errors = _validate("parallel_tool_calls", 0)
        assert "got int" in errors[0]

    def test_max_steps_error_shows_float_type(self):
        errors = _validate("max_steps", 3.14)
        assert "got float" in errors[0]

    def test_tool_timeout_error_shows_str_type(self):
        errors = _validate("tool_timeout", "30s")
        assert "got str" in errors[0]

    def test_llm_timeout_error_shows_list_type(self):
        errors = _validate("llm_timeout", [120])
        assert "got list" in errors[0]

    def test_validation_retries_error_shows_float_type(self):
        errors = _validate("validation_retries", 2.5)
        assert "got float" in errors[0]

    def test_text_format_error_shows_int_type(self):
        errors = _validate("text_format", 42)
        assert "got int" in errors[0]

    # --- Specific error message wording ---

    def test_model_empty_message(self):
        errors = _validate("model", "")
        assert errors[0] == "Field 'model' cannot be empty."

    def test_tool_choice_empty_dict_message(self):
        errors = _validate("tool_choice", {})
        assert errors[0] == "Field 'tool_choice' dict cannot be empty."

    def test_temperature_nan_message(self):
        errors = _validate("temperature", float("nan"))
        assert "NaN/Inf" in errors[0]

    def test_max_output_tokens_lte_zero_message(self):
        errors = _validate("max_output_tokens", 0)
        assert errors[0] == "Field 'max_output_tokens' must be > 0."

    def test_missing_validator_error_content(self):
        original = ConfigResolver.__dict__.get("_validate_model")
        try:
            delattr(ConfigResolver, "_validate_model")
            errors = ConfigResolver._validate_config_kwargs({"model": "gpt-4o"})
            assert "pyaiagent bug" in errors[0]
            assert "github.com/troymjose/pyaiagent/issues" in errors[0]
        finally:
            if original is not None:
                ConfigResolver._validate_model = original


# ──────────────────────────────────────────────────────────────────────────────
# 24. _is_json_safe depth boundary with empty containers
# ──────────────────────────────────────────────────────────────────────────────

class TestIsJsonSafeDepthBoundary:
    """Empty containers at depth boundary: all() on empty iterable is True."""

    def test_empty_list_at_max_depth(self):
        assert _is_json_safe([], _MAX_JSON_DEPTH) is True

    def test_empty_dict_at_max_depth(self):
        assert _is_json_safe({}, _MAX_JSON_DEPTH) is True

    def test_empty_list_beyond_max_depth(self):
        assert _is_json_safe([], _MAX_JSON_DEPTH + 1) is False

    def test_empty_dict_beyond_max_depth(self):
        assert _is_json_safe({}, _MAX_JSON_DEPTH + 1) is False

    def test_scalar_at_max_depth(self):
        assert _is_json_safe("x", _MAX_JSON_DEPTH) is True

    def test_scalar_beyond_max_depth(self):
        assert _is_json_safe("x", _MAX_JSON_DEPTH + 1) is False

    def test_nonempty_list_at_max_depth_fails(self):
        """List at max depth, child would be depth+1 which exceeds limit."""
        assert _is_json_safe(["x"], _MAX_JSON_DEPTH) is False

    def test_nonempty_dict_at_max_depth_fails(self):
        assert _is_json_safe({"k": "v"}, _MAX_JSON_DEPTH) is False


# ──────────────────────────────────────────────────────────────────────────────
# 25. OpenAIAgentConfig with unknown kwargs
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenAIAgentConfigUnknownKwargs:
    """Dataclass rejects unknown keyword arguments at construction."""

    def test_unknown_kwarg_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            OpenAIAgentConfig(nonexistent_field=42)

    def test_multiple_unknown_kwargs(self):
        with pytest.raises(TypeError):
            OpenAIAgentConfig(foo=1, bar=2)
