from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Any, TYPE_CHECKING

from pyaiagent.openai.config import OpenAIAgentConfig
from pyaiagent.openai.exceptions.definition import OpenAIAgentDefinitionError

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["OpenAIAgentConfigManager", ]

_ALLOWED_FIELDS = {f.name for f in fields(OpenAIAgentConfig)}
_ALLOWED_FIELDS_STR = ", ".join(sorted(_ALLOWED_FIELDS))


class OpenAIAgentConfigManager:

    @staticmethod
    def _validate_model(value, errors):
        if isinstance(value, str):
            if not value.strip():
                errors.append("Field 'model' cannot be empty.")
        else:
            errors.append(f"Field 'model' must be a string, got {type(value).__name__}.")

    @staticmethod
    def _validate_temperature(value, errors):
        if isinstance(value, (int, float)):
            if not (0.0 <= value <= 2.0):
                errors.append("Field 'temperature' must be between 0.0 and 2.0.")
        else:
            errors.append(f"Field 'temperature' must be a float or int, got {type(value).__name__}.")

    @staticmethod
    def _validate_max_output_tokens(value, errors):
        if isinstance(value, int):
            if value <= 0:
                errors.append("Field 'max_output_tokens' must be > 0.")
        else:
            errors.append(f"Field 'max_output_tokens' must be an int, got {type(value).__name__}.")

    @staticmethod
    def _validate_tool_choice(value, errors):
        if isinstance(value, (str, dict)):
            if isinstance(value, str) and not value.strip():
                errors.append("Field 'tool_choice' cannot be empty.")
        else:
            errors.append(f"Field 'tool_choice' must be a string or dict, got {type(value).__name__}.")

    @staticmethod
    def _validate_parallel_tool_calls(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'parallel_tool_calls' must be a bool, got {type(value).__name__}.")

    @staticmethod
    def _validate_text_format(value, errors):
        if value is not None:
            # Import at runtime only when needed (rare path)
            from pydantic import BaseModel
            if not (inspect.isclass(value) and issubclass(value, BaseModel)):
                errors.append(
                    f"Field 'text_format' must be a Pydantic model class or None, got {type(value).__name__}. Hint: Did you pass an instance instead of the class?")

    @staticmethod
    def _validate_max_steps(value, errors):
        if isinstance(value, int):
            if value <= 0:
                errors.append("Field 'max_steps' must be > 0.")
        else:
            errors.append(f"Field 'max_steps' must be an int, got {type(value).__name__}.")

    @staticmethod
    def _validate_max_parallel_tools(value, errors):
        if isinstance(value, int):
            if value <= 0:
                errors.append("Field 'max_parallel_tools' must be > 0.")
        else:
            errors.append(f"Field 'max_parallel_tools' must be an int, got {type(value).__name__}.")

    @staticmethod
    def _validate_tool_timeout(value, errors):
        if isinstance(value, (int, float)):
            if value <= 0:
                errors.append("Field 'tool_timeout' must be > 0.")
        else:
            errors.append(f"Field 'tool_timeout' must be a number, got {type(value).__name__}.")

    @staticmethod
    def _validate_llm_timeout(value, errors):
        if isinstance(value, (int, float)):
            if value <= 0:
                errors.append("Field 'llm_timeout' must be > 0.")
        else:
            errors.append(f"Field 'llm_timeout' must be a number, got {type(value).__name__}.")

    @staticmethod
    def _validate_ui_messages_enabled(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'ui_messages_enabled' must be a bool, got {type(value).__name__}.")

    @staticmethod
    def _validate_llm_messages_enabled(value, errors):
        if not isinstance(value, bool):
            errors.append(f"Field 'llm_messages_enabled' must be a bool, got {type(value).__name__}.")

    @staticmethod
    def _validate_config_kwargs(config_kwargs: dict[str, Any]) -> list[str]:
        """ Validate merged config kwargs. """

        errors: list[str] = []

        for key, value in config_kwargs.items():
            # Disallow unknown attrs
            if key not in _ALLOWED_FIELDS:
                errors.append(
                    f"Unknown attribute '{key}' for inner 'Config' class. Supported attributes: {_ALLOWED_FIELDS_STR}")
                continue

            validator = getattr(OpenAIAgentConfigManager, f"_validate_{key}", None)

            if validator is not None:
                validator(value, errors)

        return errors

    @staticmethod
    def _create_config_kwargs(inner_config_cls: type | None, parent_config_kwargs: dict | None) -> dict[str, Any]:
        """ Create merged config kwargs from parent and inner Config class. """

        # Start with parent kwargs (shallow copy only)
        if parent_config_kwargs:
            # dict(...) is a fast shallow copy and usually enough for config
            config_kwargs: dict[str, Any] = dict(parent_config_kwargs)
        else:
            config_kwargs = {}

        # If no inner class is provided, we're done
        if inner_config_cls is None:
            return config_kwargs

        # Merge only “public” attributes (non-dunder).
        # This is shallow and cheap.
        for key, value in vars(inner_config_cls).items():
            if key.startswith("__"):
                continue
            config_kwargs[key] = value

        return config_kwargs

    @staticmethod
    def create(cls) -> dict[str, Any]:
        config_kwargs: dict = OpenAIAgentConfigManager._create_config_kwargs(
            inner_config_cls=getattr(cls, "Config", None),
            parent_config_kwargs=getattr(cls, "__config_kwargs__", None))
        if errors := OpenAIAgentConfigManager._validate_config_kwargs(config_kwargs=config_kwargs):
            raise OpenAIAgentDefinitionError(cls.__name__, errors)
        return config_kwargs
