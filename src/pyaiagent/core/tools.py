"""Tool discovery and schema generation for agent methods.

This module converts Python methods on an agent class into tool schemas.
It runs once per class at definition time (__init_subclass__) and caches
results on class attributes — never on the hot path.

Pipeline:
    1. discover_tools(cls)        — scans the class for public methods with docstrings
    2. _build_function_tool_schema — converts each method's signature → JSON Schema via Pydantic
    3. _make_strict                — post-processes the schema for strict mode
    4. _get_base_classes_tools     — merges inherited tool schemas from the MRO

A method becomes a tool if it is:
    - Public (no _ prefix)
    - Not a framework hook (process, aclose, format_*_message)
    - Has a docstring (used as the tool description)
    - Is a function or coroutine (instance methods — not @staticmethod/@classmethod)

Private methods (prefixed with _) are never exposed, so users can freely
create helper methods without them being registered as tools.
"""
from __future__ import annotations

import sys
import types
import inspect
import textwrap
from pydantic import create_model
from typing import Any, Dict, Union, get_type_hints, get_origin, get_args

from pyaiagent.core.exceptions import AgentDefinitionError

__all__ = ["discover_tools", ]

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

_EXCLUDED_METHODS = frozenset({'process',
                               'aclose',
                               'format_history',
                               'format_event'})

_MAX_STRICT_DEPTH = 64

_UNSUPPORTED_PARAM_KINDS = {
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
    inspect.Parameter.POSITIONAL_ONLY: "positional-only",
}

_NONE_TYPE = type(None)

_UNION_ORIGINS: tuple[Any, ...] = (Union,)
if sys.version_info >= (3, 10):
    _UNION_ORIGINS = (Union, types.UnionType)

# ──────────────────────────────────────────────────────────────────────────────
# Schema post-processing
# ──────────────────────────────────────────────────────────────────────────────


def _make_strict(schema: Dict[str, Any], *,
                 _error_label: str = "tool schema",
                 _visited: set[int] | None = None, _depth: int = 0) -> None:
    """Recursively enforce strict-mode JSON schema rules in place.

    Strict mode requires three invariants on every object in the schema:
        1. "additionalProperties": false  — no extra keys allowed
        2. "required": [all property names] — every field must be present
           (optional fields use anyOf with null instead of omitting the key)
        3. No "default" values

    We also strip "title" (Pydantic adds it; not needed) and sort
    "required" for deterministic output across Python/Pydantic versions.

    Safety: tracks visited schema nodes by id() to handle $ref cycles,
    and enforces a depth cap (_MAX_STRICT_DEPTH) to fail fast on
    pathological schemas rather than silently producing partial results.

    Args:
        schema:       The JSON Schema dict to mutate in place.
        _error_label: Class/tool name for error attribution if depth is exceeded.
        _visited:     Set of already-processed schema node ids (cycle detection).
        _depth:       Current recursion depth.
    """
    if _depth > _MAX_STRICT_DEPTH:
        raise AgentDefinitionError(
            cls_name=_error_label,
            errors=[f"Schema nesting exceeds {_MAX_STRICT_DEPTH} levels during strict-mode enforcement. "
                    f"Simplify the tool parameter types or check for recursive type definitions."])
    if _visited is None:
        _visited = set()
    schema_id = id(schema)
    if schema_id in _visited:
        return
    _visited.add(schema_id)

    schema.pop("default", None)
    schema.pop("title", None)

    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        props = schema.get("properties", {})
        schema["required"] = sorted(props.keys())
        for prop_schema in props.values():
            _make_strict(prop_schema, _error_label=_error_label, _visited=_visited, _depth=_depth + 1)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            for sub in schema[key]:
                _make_strict(sub, _error_label=_error_label, _visited=_visited, _depth=_depth + 1)

    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            _make_strict(def_schema, _error_label=_error_label, _visited=_visited, _depth=_depth + 1)

    if "items" in schema and isinstance(schema["items"], dict):
        _make_strict(schema["items"], _error_label=_error_label, _visited=_visited, _depth=_depth + 1)

    if "prefixItems" in schema:
        for item in schema["prefixItems"]:
            _make_strict(item, _error_label=_error_label, _visited=_visited, _depth=_depth + 1)

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        _make_strict(schema["additionalProperties"], _error_label=_error_label, _visited=_visited, _depth=_depth + 1)

# ──────────────────────────────────────────────────────────────────────────────
# Type annotation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _already_accepts_none(ann: Any) -> bool:
    """Check if a type annotation already includes None.

    Handles both typing-module forms (Optional[T], Union[T, None]) and
    PEP 604 syntax (T | None) on Python 3.10+.

    Used to avoid double-wrapping: if ann is already str | None,
    we don't want to produce Union[str | None, None].
    """
    origin = get_origin(ann)
    if origin in _UNION_ORIGINS:
        return _NONE_TYPE in get_args(ann)
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Schema generation (per-tool)
# ──────────────────────────────────────────────────────────────────────────────


def _build_function_tool_schema(func_name: str,
                                func: Any,
                                description: str,
                                *,
                                cls_name: str = "",
                                additional_properties: bool = False,
                                strict: bool = True,
                                ) -> Dict[str, Any]:
    """Build an API-compliant tool schema for a single Python method.

    Steps:
        1. Resolve type hints (handles forward refs, Annotated, etc.)
        2. Map each parameter to a Pydantic field definition:
           - Required params  → (annotation, ...)      [no default]
           - Optional (=None) → (annotation | None, None)
           - With default     → (annotation, default)
        3. Create a dynamic Pydantic model and export its JSON Schema
        4. Apply strict-mode post-processing if enabled

    Args:
        func_name:   The method name (becomes the tool name in the API).
        func:        The actual function/method object.
        description: Cleaned docstring (becomes the tool description).
        cls_name:    Agent class name for error attribution.
        additional_properties: Whether to allow extra keys (only when strict=False).
        strict:      Enable strict mode (default True).

    Returns:
        A dict matching the tool schema format:
        {"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": ...}

    Raises:
        AgentDefinitionError: On unsupported params, unresolvable hints,
                                     or Pydantic schema generation failure.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise AgentDefinitionError(
            cls_name=cls_name or func_name,
            errors=[f"Tool '{func_name}': failed to resolve type hints — {type(exc).__name__}: {exc}. "
                    f"Check for unresolvable forward references or missing imports."]) from exc

    errors: list[str] = []
    field_definitions: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind in _UNSUPPORTED_PARAM_KINDS:
            errors.append(
                f"Tool '{func_name}': parameter '{name}' uses unsupported "
                f"{_UNSUPPORTED_PARAM_KINDS[param.kind]} syntax. "
                f"Only regular and keyword-only parameters are supported.")
            continue

        ann = hints.get(name, str)

        if param.default is inspect.Parameter.empty or param.default is ...:
            field_definitions[name] = (ann, ...)
        elif param.default is None:
            safe_ann = ann if _already_accepts_none(ann) else Union[ann, _NONE_TYPE]
            field_definitions[name] = (safe_ann, None)
        else:
            field_definitions[name] = (ann, param.default)

    if errors:
        raise AgentDefinitionError(cls_name=cls_name or func_name, errors=errors)

    error_label = cls_name or func_name
    try:
        DynamicModel = create_model(f"{func_name}_params", **field_definitions)
        schema = DynamicModel.model_json_schema()
    except Exception as exc:
        raise AgentDefinitionError(
            cls_name=error_label,
            errors=[f"Tool '{func_name}': schema generation failed — {type(exc).__name__}: {exc}. "
                    f"Check parameter type annotations for unsupported or invalid types."]) from exc

    if strict:
        _make_strict(schema, _error_label=error_label)
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

# ──────────────────────────────────────────────────────────────────────────────
# Tool discovery (per-class)
# ──────────────────────────────────────────────────────────────────────────────


def _get_base_classes_tools(mro: tuple[type, ...]) -> dict[str, dict[str, Any]]:
    """Collect tool schemas from all base classes in the MRO.

    Walks the MRO in reverse (most base first) so that more-derived classes
    override tools from their parents — standard Python inheritance semantics.

    Validates that __tool_names__ and __tools_schema__ have matching lengths
    to prevent silent truncation from zip().
    """
    merged_tools: dict[str, dict[str, Any]] = {}
    for base in reversed(mro[1:]):
        names = getattr(base, "__tool_names__", ())
        schemas = getattr(base, "__tools_schema__", ())
        if len(names) != len(schemas):
            raise AgentDefinitionError(
                cls_name=base.__name__,
                errors=[f"Inconsistent tool metadata: {len(names)} tool names vs {len(schemas)} schemas."])
        for tool_name, tool_schema in zip(names, schemas):
            merged_tools[tool_name] = tool_schema
    return merged_tools


def _is_tool_method(name: str, obj: Any) -> bool:
    """Determine if a class attribute should be registered as a tool.

    A method becomes a tool if it is:
        - Public (name does not start with _)
        - Not a framework hook method (process, aclose, format_*_message)
        - A function or coroutine function (instance methods only)

    Private methods (prefixed with _) are safe for users to define as
    internal helpers without them being exposed to the LLM.
    """
    if name.startswith("_"):
        return False
    if name in _EXCLUDED_METHODS:
        return False
    return inspect.iscoroutinefunction(obj) or inspect.isfunction(obj)


def discover_tools(cls) -> dict[str, dict[str, Any]]:
    """Discover tool methods on an agent class and build their schemas.

    Called once per class in __init_subclass__. Results are cached on class
    attributes (__tool_names__, __tools_schema__) and reused for every
    process() call — this function is never on the hot path.

    Steps:
        1. Scan cls.__dict__ for methods that pass _is_tool_method
        2. Validate each has a docstring (required for tool description)
        3. Build a JSON Schema for each method via _build_function_tool_schema
        4. Merge with inherited tools from base classes (child overrides parent)

    Args:
        cls: The agent class being defined.

    Returns:
        A dict mapping tool names to their tool schema dicts.

    Raises:
        AgentDefinitionError: If any tool method is missing a docstring,
                                     has unsupported parameters, or fails schema generation.
    """
    errors: list[str] = []
    declared_tools: list[dict[str, Any]] = []
    tool_names: list[str] = []
    for name, obj in cls.__dict__.items():
        if not _is_tool_method(name, obj):
            continue
        tool_doc = getattr(obj, "__doc__", "")
        cleaned_tool_doc = textwrap.dedent(tool_doc or "").strip()
        if not cleaned_tool_doc:
            errors.append(
                f"Tool '{name}' is missing docstring. Add a triple-quoted docstring as tool description.")
            continue
        declared_tools.append(_build_function_tool_schema(name, obj, cleaned_tool_doc, cls_name=cls.__name__))
        tool_names.append(name)
    if errors:
        raise AgentDefinitionError(cls_name=cls.__name__, errors=errors)

    merged_tools: dict[str, dict[str, Any]] = _get_base_classes_tools(mro=cls.__mro__)
    for n, sch in zip(tool_names, declared_tools):
        merged_tools[n] = sch

    return merged_tools
