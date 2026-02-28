from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Type, TYPE_CHECKING

from pyaiagent.core.exceptions import AgentDefinitionError
from pyaiagent.core.config import merge_config

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["OpenAIAgentConfig", "ConfigResolver", ]

OpenAiToolChoice = Literal["auto", "none", "required"]


@dataclass(frozen=True, slots=True)
class OpenAIAgentConfig:
    # OpenAI LLM configuration
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float | None = None
    seed: int | None = None
    max_output_tokens: int = 4096
    tool_choice: OpenAiToolChoice | Dict[str, Any] = "auto"
    parallel_tool_calls: bool = True
    text_format: Type[BaseModel] | None = None
    user: str | None = None
    # Runtime configuration
    max_steps: int = 10
    max_parallel_tools: int = 10
    tool_timeout: float = 30.0
    llm_timeout: float = 120.0
    include_events: bool = True
    include_history: bool = True
    strict_instruction_params: bool = False
    validation_retries: int = 0


_ALLOWED_FIELDS = {field.name for field in fields(OpenAIAgentConfig)}
_ALLOWED_FIELDS_STR = "https://github.com/troymjose/pyaiagent#all-configuration-options"
_JSON_SCALARS = (str, int, float, bool, type(None))
_MAX_JSON_DEPTH = 10


def _is_json_safe(value: Any, depth: int = 0) -> bool:
    """Recursively verify a value is JSON-serializable (primitives, lists, dicts with str keys)."""
    if depth > _MAX_JSON_DEPTH:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, _JSON_SCALARS):
        return True
    if isinstance(value, list):
        return all(_is_json_safe(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_safe(v, depth + 1)
            for k, v in value.items())
    return False


class ConfigResolver:

    @staticmethod
    def _validate_model(value, errors):
        if isinstance(value, str):
            if not value.strip():
                errors.append("Field 'model' cannot be empty.")
            elif value != value.strip():
                errors.append(
                    f"Field 'model' has leading/trailing whitespace: '{value}'. Use '{value.strip()}' instead.")
        else:
            errors.append(f"Field 'model' must be a string, got {type(value).__name__}.")

    @staticmethod
    def _validate_temperature(value, errors):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                errors.append("Field 'temperature' must be a finite number, got NaN/Inf.")
            elif not (0.0 <= value <= 2.0):
                errors.append("Field 'temperature' must be between 0.0 and 2.0.")
        else:
            errors.append(f"Field 'temperature' must be a float or int, got {type(value).__name__}.")

    @staticmethod
    def _validate_top_p(value, errors):
        if value is not None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float) and not math.isfinite(value):
                    errors.append("Field 'top_p' must be a finite number, got NaN/Inf.")
                elif not (0.0 <= value <= 1.0):
                    errors.append("Field 'top_p' must be between 0.0 and 1.0.")
            else:
                errors.append(f"Field 'top_p' must be a float, int, or None, got {type(value).__name__}.")

    @staticmethod
    def _validate_seed(value, errors):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"Field 'seed' must be an int or None, got {type(value).__name__}.")

    @staticmethod
    def _validate_max_output_tokens(value, errors):
        if isinstance(value, int) and not isinstance(value, bool):
            if value <= 0:
                errors.append("Field 'max_output_tokens' must be > 0.")
        else:
            errors.append(f"Field 'max_output_tokens' must be an int, got {type(value).__name__}.")

    _VALID_TOOL_CHOICE_STRINGS = frozenset({"auto", "none", "required"})

    @staticmethod
    def _validate_tool_choice(value, errors):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped != value:
                errors.append(
                    f"Field 'tool_choice' has leading/trailing whitespace: '{value}'. Use '{stripped}' instead.")
            elif stripped not in ConfigResolver._VALID_TOOL_CHOICE_STRINGS:
                errors.append(
                    f"Field 'tool_choice' must be one of {{'auto', 'none', 'required'}} or a dict, got '{value}'.")
        elif isinstance(value, dict):
            if not value:
                errors.append("Field 'tool_choice' dict cannot be empty.")
            elif not _is_json_safe(value):
                errors.append(
                    "Field 'tool_choice' dict must be JSON-serializable "
                    "(str keys, values of str/int/float/bool/None/list/dict).")
        else:
            errors.append(f"Field 'tool_choice' must be a string or dict, got {type(value).__name__}.")

    @staticmethod
    def _validate_parallel_tool_calls(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'parallel_tool_calls' must be a bool, got {type(value).__name__}.")

    @staticmethod
    def _validate_text_format(value, errors):
        if value is not None:
            try:
                from pydantic import BaseModel
            except ImportError:
                errors.append("Field 'text_format' requires pydantic. Install it with: pip install pydantic")
                return
            if not (inspect.isclass(value) and issubclass(value, BaseModel)):
                errors.append(
                    f"Field 'text_format' must be a Pydantic model class or None, got {type(value).__name__}. Hint: Did you pass an instance instead of the class?")

    @staticmethod
    def _validate_user(value, errors):
        if value is not None:
            if isinstance(value, str):
                if not value.strip():
                    errors.append("Field 'user' cannot be empty.")
                elif value != value.strip():
                    errors.append(
                        f"Field 'user' has leading/trailing whitespace: '{value}'. Use '{value.strip()}' instead.")
            else:
                errors.append(f"Field 'user' must be a string or None, got {type(value).__name__}.")

    @staticmethod
    def _validate_max_steps(value, errors):
        if isinstance(value, int) and not isinstance(value, bool):
            if value <= 0:
                errors.append("Field 'max_steps' must be > 0.")
        else:
            errors.append(f"Field 'max_steps' must be an int, got {type(value).__name__}.")

    @staticmethod
    def _validate_max_parallel_tools(value, errors):
        if isinstance(value, int) and not isinstance(value, bool):
            if value <= 0:
                errors.append("Field 'max_parallel_tools' must be > 0.")
        else:
            errors.append(f"Field 'max_parallel_tools' must be an int, got {type(value).__name__}.")

    @staticmethod
    def _validate_tool_timeout(value, errors):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                errors.append("Field 'tool_timeout' must be a finite number, got NaN/Inf.")
            elif value <= 0:
                errors.append("Field 'tool_timeout' must be > 0.")
        else:
            errors.append(f"Field 'tool_timeout' must be a number, got {type(value).__name__}.")

    @staticmethod
    def _validate_llm_timeout(value, errors):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                errors.append("Field 'llm_timeout' must be a finite number, got NaN/Inf.")
            elif value <= 0:
                errors.append("Field 'llm_timeout' must be > 0.")
        else:
            errors.append(f"Field 'llm_timeout' must be a number, got {type(value).__name__}.")

    @staticmethod
    def _validate_include_events(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'include_events' must be a bool, got {type(value).__name__}.")

    @staticmethod
    def _validate_include_history(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'include_history' must be a bool, got {type(value).__name__}.")

    @staticmethod
    def _validate_strict_instruction_params(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'strict_instruction_params' must be a bool, got {type(value).__name__}.")

    _MAX_VALIDATION_RETRIES = 10

    @staticmethod
    def _validate_validation_retries(value, errors):
        if isinstance(value, int) and not isinstance(value, bool):
            if not (0 <= value <= ConfigResolver._MAX_VALIDATION_RETRIES):
                errors.append(
                    f"Field 'validation_retries' must be between 0 and "
                    f"{ConfigResolver._MAX_VALIDATION_RETRIES}.")
        else:
            errors.append(f"Field 'validation_retries' must be an int, got {type(value).__name__}.")

    @staticmethod
    def _validate_config_kwargs(config_kwargs: dict[str, Any]) -> list[str]:
        """ Validate merged config kwargs. """

        errors: list[str] = []

        for key, value in config_kwargs.items():
            if key not in _ALLOWED_FIELDS:
                errors.append(
                    f"Unknown attribute '{key}' for inner 'Config' class. Supported attributes: {_ALLOWED_FIELDS_STR}")
                continue

            validator = getattr(ConfigResolver, f"_validate_{key}", None)
            if validator is None:
                errors.append(
                    f"Library error: missing validator for config field '{key}'. "
                    f"This is a pyaiagent bug, not a config mistake. "
                    f"Please report it at https://github.com/troymjose/pyaiagent/issues")
                continue
            validator(value, errors)

        return errors

    @staticmethod
    def _create_config_kwargs(inner_config_cls: type | None, parent_config_kwargs: dict | None) -> dict[str, Any]:
        """ Create merged config kwargs from parent and inner Config class. """
        return merge_config(inner_config_cls, parent_config_kwargs)

    @staticmethod
    def resolve(cls: Type[Any]) -> dict[str, Any]:
        """ Resolve and validate config kwargs for the given class. """
        config_kwargs: dict[str, Any] = ConfigResolver._create_config_kwargs(
            inner_config_cls=getattr(cls, "Config", None),
            parent_config_kwargs=getattr(cls, "__config_kwargs__", None))
        if errors := ConfigResolver._validate_config_kwargs(config_kwargs=config_kwargs):
            raise AgentDefinitionError(cls_name=cls.__name__, errors=errors)
        return config_kwargs
