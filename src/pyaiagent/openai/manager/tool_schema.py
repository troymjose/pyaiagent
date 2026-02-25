from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, get_type_hints

from pydantic import create_model

__all__ = ["ToolSchemaManager"]


def _make_strict(schema: Dict[str, Any]) -> None:
    """Recursively enforce OpenAI strict-mode JSON schema rules in place.

    Strict mode requires: all object properties in "required",
    "additionalProperties": false on every object, and no "default" values.
    """
    schema.pop("default", None)
    schema.pop("title", None)

    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        props = schema.get("properties", {})
        schema["required"] = list(props.keys())
        for prop_schema in props.values():
            _make_strict(prop_schema)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            for sub in schema[key]:
                _make_strict(sub)

    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            _make_strict(def_schema)

    if "items" in schema and isinstance(schema["items"], dict):
        _make_strict(schema["items"])

    if "prefixItems" in schema:
        for item in schema["prefixItems"]:
            _make_strict(item)

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        _make_strict(schema["additionalProperties"])


class ToolSchemaManager:
    """Converts Python function signatures into OpenAI Responses API tool schemas
    using Pydantic for type-to-JSON-Schema conversion."""

    @staticmethod
    def build_function_tool_schema(func_name: str,
                                   func: Any,
                                   description: str,
                                   *,
                                   additional_properties: bool = False,
                                   strict: bool = True,
                                   ) -> Dict[str, Any]:
        """Build a Responses API-compliant tool schema for a Python function.

        Inspects the function's signature and type hints, creates a dynamic
        Pydantic model, and generates a JSON Schema from it.
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func, include_extras=True)

        field_definitions: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name in {"self", "cls"}:
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                continue

            ann = hints.get(name, str)

            if param.default is inspect.Parameter.empty or param.default is ...:
                field_definitions[name] = (ann, ...)
            elif param.default is None:
                field_definitions[name] = (Optional[ann], None)
            else:
                field_definitions[name] = (ann, param.default)

        DynamicModel = create_model(f"{func_name}_params", **field_definitions)
        schema = DynamicModel.model_json_schema()

        if strict:
            _make_strict(schema)
        else:
            schema["additionalProperties"] = additional_properties

        tool: Dict[str, Any] = {
            "type": "function",
            "name": func_name,
            "description": description or f"Tool {func_name}",
            "parameters": schema,
        }

        if strict is not None:
            tool["strict"] = bool(strict)

        return tool
