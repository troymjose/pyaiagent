"""
Comprehensive unit tests for pyaiagent.openai.tools.

Covers: _is_tool_method, _already_accepts_none, _build_function_tool_schema,
        _make_strict, _get_base_classes_tools, discover_tools, and integration.
"""
import sys
import inspect
from datetime import datetime
from typing import Any, Annotated, Dict, List, Literal, Optional, Union
from enum import Enum

import pytest
from pydantic import BaseModel

from pyaiagent.openai.exceptions import OpenAIAgentDefinitionError
from pyaiagent.openai.tools import (
    _is_tool_method,
    _already_accepts_none,
    _build_function_tool_schema,
    _make_strict,
    _get_base_classes_tools,
    discover_tools,
    _MAX_STRICT_DEPTH,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared assertion helper
# ──────────────────────────────────────────────────────────────────────────────

def assert_strict_invariants(schema: dict, *, path: str = "$") -> None:
    """Recursively verify OpenAI strict-mode invariants on a JSON Schema dict."""
    assert "default" not in schema, f"'default' found at {path}"
    assert "title" not in schema, f"'title' found at {path}"

    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, (
            f"'additionalProperties' is not False at {path}")
        props = schema.get("properties", {})
        assert schema.get("required") == sorted(props.keys()), (
            f"'required' mismatch at {path}: {schema.get('required')} vs {sorted(props.keys())}")
        for prop_name, prop_schema in props.items():
            assert_strict_invariants(prop_schema, path=f"{path}.properties.{prop_name}")

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            for i, sub in enumerate(schema[key]):
                assert_strict_invariants(sub, path=f"{path}.{key}[{i}]")

    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            assert_strict_invariants(def_schema, path=f"{path}.$defs.{def_name}")

    if "items" in schema and isinstance(schema["items"], dict):
        assert_strict_invariants(schema["items"], path=f"{path}.items")

    if "prefixItems" in schema:
        for i, item in enumerate(schema["prefixItems"]):
            assert_strict_invariants(item, path=f"{path}.prefixItems[{i}]")

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        assert_strict_invariants(schema["additionalProperties"], path=f"{path}.additionalProperties")


# ──────────────────────────────────────────────────────────────────────────────
# 1. TestIsToolMethod
# ──────────────────────────────────────────────────────────────────────────────

class TestIsToolMethod:

    def test_public_sync_function(self):
        def search(self, query: str): ...
        assert _is_tool_method("search", search) is True

    def test_public_async_function(self):
        async def search(self, query: str): ...
        assert _is_tool_method("search", search) is True

    def test_private_method_excluded(self):
        def _helper(self): ...
        assert _is_tool_method("_helper", _helper) is False

    def test_dunder_method_excluded(self):
        def __init__(self): ...
        assert _is_tool_method("__init__", __init__) is False

    @pytest.mark.parametrize("name", ["process", "aclose", "format_history", "format_event"])
    def test_excluded_framework_methods(self, name):
        async def method(self): ...
        assert _is_tool_method(name, method) is False

    def test_string_attribute_not_tool(self):
        assert _is_tool_method("name", "some string") is False

    def test_int_attribute_not_tool(self):
        assert _is_tool_method("count", 42) is False

    def test_none_attribute_not_tool(self):
        assert _is_tool_method("value", None) is False

    def test_staticmethod_descriptor_not_detected(self):
        @staticmethod
        def helper(): ...
        assert _is_tool_method("helper", helper) is False

    def test_classmethod_descriptor_not_detected(self):
        @classmethod
        def helper(cls): ...
        assert _is_tool_method("helper", helper) is False

    def test_lambda_is_function(self):
        fn = lambda self, x: x
        assert _is_tool_method("transform", fn) is True

    def test_class_object_not_tool(self):
        class Nested: ...
        assert _is_tool_method("Nested", Nested) is False

    def test_property_descriptor_not_tool(self):
        @property
        def name(self): return "x"
        assert _is_tool_method("name", name) is False

    def test_list_attribute_not_tool(self):
        assert _is_tool_method("items", [1, 2, 3]) is False

    def test_dict_attribute_not_tool(self):
        assert _is_tool_method("config", {"key": "value"}) is False

    def test_bool_attribute_not_tool(self):
        assert _is_tool_method("enabled", True) is False


# ──────────────────────────────────────────────────────────────────────────────
# 2. TestAlreadyAcceptsNone
# ──────────────────────────────────────────────────────────────────────────────

class TestAlreadyAcceptsNone:

    def test_optional_str(self):
        assert _already_accepts_none(Optional[str]) is True

    def test_union_str_none(self):
        assert _already_accepts_none(Union[str, None]) is True

    def test_union_str_int_none(self):
        assert _already_accepts_none(Union[str, int, None]) is True

    def test_plain_str(self):
        assert _already_accepts_none(str) is False

    def test_plain_int(self):
        assert _already_accepts_none(int) is False

    def test_union_without_none(self):
        assert _already_accepts_none(Union[str, int]) is False

    def test_list_str(self):
        assert _already_accepts_none(list[str]) is False

    def test_any(self):
        assert _already_accepts_none(Any) is False

    def test_dict_type(self):
        assert _already_accepts_none(dict[str, int]) is False

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10+")
    def test_pep604_str_none(self):
        ann = eval("str | None")
        assert _already_accepts_none(ann) is True

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10+")
    def test_pep604_str_int_none(self):
        ann = eval("str | int | None")
        assert _already_accepts_none(ann) is True

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10+")
    def test_pep604_str_int_no_none(self):
        ann = eval("str | int")
        assert _already_accepts_none(ann) is False

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10+")
    def test_pep604_single_type(self):
        assert _already_accepts_none(int) is False

    def test_none_type_itself(self):
        assert _already_accepts_none(type(None)) is False

    def test_none_value(self):
        assert _already_accepts_none(None) is False

    def test_optional_already_optional(self):
        assert _already_accepts_none(Optional[str]) is True


# ──────────────────────────────────────────────────────────────────────────────
# 3. TestBuildFunctionToolSchema
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildFunctionToolSchema:

    # --- Basic signatures ---

    def test_self_only_empty_params(self):
        def tool(self): ...
        schema = _build_function_tool_schema("tool", tool, "A tool.")
        params = schema["parameters"]
        assert params.get("properties", {}) == {}
        assert schema["name"] == "tool"
        assert schema["description"] == "A tool."
        assert schema["type"] == "function"
        assert schema["strict"] is True

    def test_single_required_str(self):
        def search(self, query: str): ...
        schema = _build_function_tool_schema("search", search, "Search.")
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "query" in schema["parameters"]["required"]

    def test_multiple_typed_params(self):
        def multi(self, name: str, age: int, score: float, active: bool): ...
        schema = _build_function_tool_schema("multi", multi, "Multi.")
        props = schema["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["age"]["type"] == "integer"
        assert props["score"]["type"] == "number"
        assert props["active"]["type"] == "boolean"
        assert sorted(schema["parameters"]["required"]) == ["active", "age", "name", "score"]

    def test_optional_param_none_default(self):
        def tool(self, query: str, category: str = None): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        cat_schema = schema["parameters"]["properties"]["category"]
        assert "anyOf" in cat_schema
        type_set = {s.get("type") for s in cat_schema["anyOf"]}
        assert "string" in type_set
        assert "null" in type_set

    def test_param_with_non_none_default(self):
        def tool(self, limit: int = 10): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        props = schema["parameters"]["properties"]
        assert "limit" in props
        assert props["limit"]["type"] == "integer"

    def test_mixed_required_and_optional(self):
        def tool(self, query: str, limit: int = 10, tag: str = None): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert "limit" in props
        assert "tag" in props
        assert len(schema["parameters"]["required"]) == 3

    # --- Type coverage ---

    def test_list_str_type(self):
        def tool(self, items: list[str]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        items_schema = schema["parameters"]["properties"]["items"]
        assert items_schema["type"] == "array"

    def test_list_int_type(self):
        def tool(self, numbers: list[int]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        numbers_schema = schema["parameters"]["properties"]["numbers"]
        assert numbers_schema["type"] == "array"

    def test_bool_type(self):
        def tool(self, flag: bool): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert schema["parameters"]["properties"]["flag"]["type"] == "boolean"

    def test_literal_type(self):
        def tool(self, mode: Literal["fast", "slow"]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        mode_schema = schema["parameters"]["properties"]["mode"]
        assert "enum" in mode_schema
        assert set(mode_schema["enum"]) == {"fast", "slow"}

    def test_optional_int(self):
        def tool(self, count: Optional[int] = None): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        count_schema = schema["parameters"]["properties"]["count"]
        assert "anyOf" in count_schema
        type_set = {s.get("type") for s in count_schema["anyOf"]}
        assert "integer" in type_set
        assert "null" in type_set

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 requires Python 3.10+")
    def test_pep604_optional_no_double_wrap(self):
        ns = {}
        exec("def tool(self, name: str | None = None): ...", ns)
        fn = ns["tool"]
        schema = _build_function_tool_schema("tool", fn, "Tool.")
        name_schema = schema["parameters"]["properties"]["name"]
        assert "anyOf" in name_schema
        type_set = {s.get("type") for s in name_schema["anyOf"]}
        assert "string" in type_set
        assert "null" in type_set
        assert len(name_schema["anyOf"]) == 2

    def test_union_str_int(self):
        def tool(self, value: Union[str, int]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        val_schema = schema["parameters"]["properties"]["value"]
        assert "anyOf" in val_schema
        type_set = {s.get("type") for s in val_schema["anyOf"]}
        assert "string" in type_set
        assert "integer" in type_set

    def test_nested_pydantic_model(self):
        class Address(BaseModel):
            street: str
            city: str

        def tool(self, address: Address): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert "$defs" in schema["parameters"] or "properties" in schema["parameters"]
        assert_strict_invariants(schema["parameters"])

    def test_unannotated_param_defaults_to_string(self):
        ns = {}
        exec("def tool(self, query): ...", ns)
        fn = ns["tool"]
        schema = _build_function_tool_schema("tool", fn, "Tool.")
        assert schema["parameters"]["properties"]["query"]["type"] == "string"

    def test_enum_type(self):
        class Color(str, Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        def tool(self, color: Color): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        color_schema = schema["parameters"]["properties"]["color"]
        # Pydantic emits enums as a $ref to $defs
        if "$ref" in color_schema:
            ref_name = color_schema["$ref"].split("/")[-1]
            enum_def = schema["parameters"]["$defs"][ref_name]
            assert "enum" in enum_def
            assert set(enum_def["enum"]) == {"red", "green", "blue"}
        else:
            assert "enum" in color_schema
            assert set(color_schema["enum"]) == {"red", "green", "blue"}

    # --- Strict mode invariants ---

    def test_strict_mode_invariants(self):
        def tool(self, query: str, limit: int = 10, tag: str = None): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert schema["strict"] is True
        assert_strict_invariants(schema["parameters"])

    def test_strict_mode_nested_model(self):
        class Inner(BaseModel):
            x: int
            y: str

        def tool(self, data: Inner): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert_strict_invariants(schema["parameters"])

    def test_strict_mode_list_of_objects(self):
        class Item(BaseModel):
            name: str
            value: int

        def tool(self, items: list[Item]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert_strict_invariants(schema["parameters"])

    # --- Non-strict mode ---

    def test_non_strict_mode(self):
        def tool(self, query: str): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.", strict=False)
        assert schema["strict"] is False
        assert schema["parameters"].get("additionalProperties") is False

    def test_non_strict_with_additional_properties(self):
        def tool(self, query: str): ...
        schema = _build_function_tool_schema(
            "tool", tool, "Tool.", strict=False, additional_properties=True)
        assert schema["parameters"]["additionalProperties"] is True

    # --- Error paths ---

    def test_var_positional_rejected(self):
        ns = {}
        exec("def tool(self, *args): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError, match=r"\*args"):
            _build_function_tool_schema("tool", fn, "Tool.")

    def test_var_keyword_rejected(self):
        ns = {}
        exec("def tool(self, **kwargs): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError, match=r"\*\*kwargs"):
            _build_function_tool_schema("tool", fn, "Tool.")

    def test_multiple_unsupported_params_all_reported(self):
        ns = {}
        exec("def tool(self, *args, **kwargs): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            _build_function_tool_schema("tool", fn, "Tool.")
        msg = str(exc_info.value)
        assert "*args" in msg
        assert "**kwargs" in msg

    def test_unresolvable_forward_ref(self):
        ns = {}
        exec("def tool(self, x: 'NonExistentType'): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError, match="type hints"):
            _build_function_tool_schema("tool", fn, "Tool.")

    def test_error_includes_cls_name(self):
        ns = {}
        exec("def tool(self, *args): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError, match="MyAgent"):
            _build_function_tool_schema("tool", fn, "Tool.", cls_name="MyAgent")

    def test_error_includes_exception_type(self):
        ns = {}
        exec("def tool(self, x: 'NoSuchType'): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError, match="NameError"):
            _build_function_tool_schema("tool", fn, "Tool.")

    def test_cls_only_skipped(self):
        def tool(cls): ...
        schema = _build_function_tool_schema("tool", tool, "A tool.")
        assert schema["parameters"].get("properties", {}) == {}

    def test_description_fallback(self):
        def tool(self): ...
        schema = _build_function_tool_schema("tool", tool, "")
        assert schema["description"] == "Tool tool"

    def test_positional_only_param_rejected(self):
        ns = {}
        exec("def tool(self, x, /): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError, match="positional-only"):
            _build_function_tool_schema("tool", fn, "Tool.")

    def test_keyword_only_param_accepted(self):
        ns = {}
        exec("def tool(self, *, query: str, limit: int = 5): ...", ns)
        fn = ns["tool"]
        schema = _build_function_tool_schema("tool", fn, "Tool.")
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert "limit" in props

    def test_tuple_param_generates_prefix_items(self):
        def tool(self, coords: tuple[float, float]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        params = schema["parameters"]
        coord_schema = params["properties"]["coords"]
        assert "prefixItems" in coord_schema or coord_schema.get("type") == "array"
        assert_strict_invariants(params)

    def test_dict_str_model_param(self):
        class Score(BaseModel):
            value: int

        def tool(self, scores: dict[str, Score]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert_strict_invariants(schema["parameters"])

    def test_schema_generation_failure_raises(self):
        def tool(self, x: "CompletelyBogusForward"): ...
        with pytest.raises(OpenAIAgentDefinitionError, match="schema generation failed|type hints"):
            _build_function_tool_schema("tool", tool, "Tool.")

    def test_exception_chaining_forward_ref(self):
        ns = {}
        exec("def tool(self, x: 'NoSuchType'): ...", ns)
        fn = ns["tool"]
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            _build_function_tool_schema("tool", fn, "Tool.")
        assert exc_info.value.__cause__ is not None

    def test_optional_of_already_nullable_no_double_wrap(self):
        def tool(self, val: Optional[str] = None): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        val_schema = schema["parameters"]["properties"]["val"]
        assert "anyOf" in val_schema
        assert len(val_schema["anyOf"]) == 2

    def test_ellipsis_default_treated_as_required(self):
        ns = {}
        exec("def tool(self, x: int = ...): ...", ns)
        fn = ns["tool"]
        schema = _build_function_tool_schema("tool", fn, "Tool.")
        assert "x" in schema["parameters"]["required"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. TestMakeStrict
# ──────────────────────────────────────────────────────────────────────────────

class TestMakeStrict:

    # --- Core invariants ---

    def test_removes_default(self):
        schema = {"type": "string", "default": "hello"}
        _make_strict(schema)
        assert "default" not in schema

    def test_removes_title(self):
        schema = {"type": "string", "title": "MyField"}
        _make_strict(schema)
        assert "title" not in schema

    def test_object_gets_additional_properties_false(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        _make_strict(schema)
        assert schema["additionalProperties"] is False

    def test_object_required_equals_sorted_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "zebra": {"type": "string"},
                "apple": {"type": "integer"},
                "mango": {"type": "boolean"},
            },
        }
        _make_strict(schema)
        assert schema["required"] == ["apple", "mango", "zebra"]

    def test_empty_object(self):
        schema = {"type": "object", "properties": {}}
        _make_strict(schema)
        assert schema["required"] == []
        assert schema["additionalProperties"] is False

    # --- Traversal coverage ---

    def test_nested_object_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "title": "Inner",
                    "default": {},
                    "properties": {"x": {"type": "integer", "title": "X", "default": 0}},
                }
            },
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_any_of(self):
        schema = {
            "anyOf": [
                {"type": "object", "title": "A", "properties": {"a": {"type": "string", "default": ""}}},
                {"type": "string", "title": "B", "default": "x"},
            ]
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_one_of(self):
        schema = {
            "oneOf": [
                {"type": "object", "title": "A", "properties": {"a": {"type": "string"}}},
                {"type": "integer", "title": "B"},
            ]
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_all_of(self):
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string", "title": "T"}}},
            ]
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_defs(self):
        schema = {
            "type": "object",
            "properties": {"ref": {"$ref": "#/$defs/MyModel"}},
            "$defs": {
                "MyModel": {
                    "type": "object",
                    "title": "MyModel",
                    "properties": {"val": {"type": "integer", "default": 0}},
                }
            },
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_items_list_schema(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "title": "Item",
                "properties": {"name": {"type": "string", "default": ""}},
            },
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_prefix_items_tuple_schema(self):
        schema = {
            "type": "array",
            "prefixItems": [
                {"type": "string", "title": "First", "default": ""},
                {"type": "object", "title": "Second", "properties": {"x": {"type": "integer"}}},
            ],
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_additional_properties_dict_schema(self):
        # type: "object" forces additionalProperties=False,
        # so test dict-typed additionalProperties on a non-object wrapper
        schema = {
            "additionalProperties": {
                "type": "object",
                "title": "Value",
                "properties": {"v": {"type": "string", "default": ""}},
            },
        }
        _make_strict(schema)
        inner = schema["additionalProperties"]
        assert "title" not in inner
        assert "default" not in inner["properties"]["v"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["v"]

    # --- Safety ---

    def test_cycle_detection_no_infinite_recursion(self):
        schema: dict = {"type": "object", "properties": {}}
        schema["properties"]["self_ref"] = schema
        _make_strict(schema)
        assert schema["additionalProperties"] is False

    def test_depth_cap_raises(self):
        schema: dict = {"type": "object", "properties": {}}
        current = schema
        for _ in range(_MAX_STRICT_DEPTH + 2):
            child: dict = {"type": "object", "properties": {}}
            current["properties"]["nested"] = child
            current = child

        with pytest.raises(OpenAIAgentDefinitionError, match="nesting exceeds"):
            _make_strict(schema)

    def test_depth_cap_includes_error_label(self):
        schema: dict = {"type": "object", "properties": {}}
        current = schema
        for _ in range(_MAX_STRICT_DEPTH + 2):
            child: dict = {"type": "object", "properties": {}}
            current["properties"]["nested"] = child
            current = child

        with pytest.raises(OpenAIAgentDefinitionError, match="MyTool"):
            _make_strict(schema, _error_label="MyTool")

    def test_already_visited_node_skipped(self):
        shared = {"type": "object", "title": "Shared", "properties": {"x": {"type": "string"}}}
        schema = {
            "anyOf": [shared, shared],
        }
        _make_strict(schema)
        assert "title" not in shared

    # --- Determinism ---

    def test_required_sorted(self):
        schema = {
            "type": "object",
            "properties": {
                "z_field": {"type": "string"},
                "a_field": {"type": "integer"},
                "m_field": {"type": "boolean"},
            },
        }
        _make_strict(schema)
        assert schema["required"] == ["a_field", "m_field", "z_field"]

    def test_non_object_schema_untouched(self):
        schema = {"type": "string", "default": "x", "title": "T"}
        _make_strict(schema)
        assert "default" not in schema
        assert "title" not in schema
        assert "additionalProperties" not in schema
        assert "required" not in schema

    def test_object_without_properties_key(self):
        schema = {"type": "object"}
        _make_strict(schema)
        assert schema["additionalProperties"] is False
        assert schema["required"] == []

    def test_items_non_dict_skipped(self):
        schema = {"type": "array", "items": True, "title": "Arr"}
        _make_strict(schema)
        assert "title" not in schema
        assert schema["items"] is True

    def test_depth_overflow_via_anyof(self):
        schema: dict = {"anyOf": []}
        current = schema
        for _ in range(_MAX_STRICT_DEPTH + 2):
            child: dict = {"anyOf": []}
            current["anyOf"].append(child)
            current = child
        with pytest.raises(OpenAIAgentDefinitionError, match="nesting exceeds"):
            _make_strict(schema)

    def test_depth_overflow_via_items(self):
        schema: dict = {"type": "array", "items": {}}
        current = schema["items"]
        for _ in range(_MAX_STRICT_DEPTH + 2):
            child: dict = {"type": "array", "items": {}}
            current["items"] = child
            current = child["items"]
        with pytest.raises(OpenAIAgentDefinitionError, match="nesting exceeds"):
            _make_strict(schema)

    def test_multiple_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/ModelA"},
                "b": {"$ref": "#/$defs/ModelB"},
            },
            "$defs": {
                "ModelA": {
                    "type": "object",
                    "title": "ModelA",
                    "properties": {"x": {"type": "string", "default": ""}},
                },
                "ModelB": {
                    "type": "object",
                    "title": "ModelB",
                    "properties": {"y": {"type": "integer", "title": "Y"}},
                },
            },
        }
        _make_strict(schema)
        assert_strict_invariants(schema)

    def test_depth_overflow_via_prefix_items(self):
        schema: dict = {"prefixItems": []}
        current = schema
        for _ in range(_MAX_STRICT_DEPTH + 2):
            child: dict = {"prefixItems": []}
            current["prefixItems"].append(child)
            current = child
        with pytest.raises(OpenAIAgentDefinitionError, match="nesting exceeds"):
            _make_strict(schema, _error_label="TupleTest")

    def test_depth_overflow_via_defs(self):
        schema: dict = {"$defs": {}}
        current = schema
        for _ in range(_MAX_STRICT_DEPTH + 2):
            child: dict = {"$defs": {}}
            current["$defs"]["nested"] = child
            current = child
        with pytest.raises(OpenAIAgentDefinitionError, match="nesting exceeds"):
            _make_strict(schema)


# ──────────────────────────────────────────────────────────────────────────────
# 5. TestGetBaseClassesTools
# ──────────────────────────────────────────────────────────────────────────────

class TestGetBaseClassesTools:

    def _make_base(self, tool_names=(), tool_schemas=(), name="Base"):
        cls = type(name, (), {
            "__tool_names__": tool_names,
            "__tools_schema__": tool_schemas,
        })
        return cls

    def test_base_with_tools(self):
        base = self._make_base(("search",), ({"name": "search"},))
        mro = (type("Child", (base,), {}), base, object)
        result = _get_base_classes_tools(mro)
        assert "search" in result
        assert result["search"] == {"name": "search"}

    def test_multiple_bases(self):
        base_a = self._make_base(("tool_a",), ({"name": "a"},), "BaseA")
        base_b = self._make_base(("tool_b",), ({"name": "b"},), "BaseB")
        child = type("Child", (base_a, base_b), {})
        mro = (child, base_a, base_b, object)
        result = _get_base_classes_tools(mro)
        assert "tool_a" in result
        assert "tool_b" in result

    def test_later_base_overrides_earlier(self):
        base_a = self._make_base(("tool",), ({"source": "a"},), "BaseA")
        base_b = self._make_base(("tool",), ({"source": "b"},), "BaseB")
        child = type("Child", (base_b, base_a), {})
        mro = (child, base_b, base_a, object)
        result = _get_base_classes_tools(mro)
        assert result["tool"]["source"] == "b"

    def test_base_without_tool_attrs_graceful(self):
        mro = (type("Child", (), {}), object)
        result = _get_base_classes_tools(mro)
        assert result == {}

    def test_mismatched_lengths_raises(self):
        base = self._make_base(("a", "b"), ({"name": "a"},), "Bad")
        mro = (type("Child", (base,), {}), base, object)
        with pytest.raises(OpenAIAgentDefinitionError, match="Inconsistent tool metadata"):
            _get_base_classes_tools(mro)

    def test_empty_tool_attrs(self):
        base = self._make_base((), (), "Empty")
        mro = (type("Child", (base,), {}), base, object)
        result = _get_base_classes_tools(mro)
        assert result == {}

    def test_single_element_mro_object_only(self):
        result = _get_base_classes_tools((object,))
        assert result == {}


# ──────────────────────────────────────────────────────────────────────────────
# 6. TestDiscoverTools
# ──────────────────────────────────────────────────────────────────────────────

class TestDiscoverTools:

    def test_single_async_tool(self):
        class Agent:
            async def search(self, query: str) -> dict:
                """Search the web."""
                ...
        result = discover_tools(Agent)
        assert "search" in result
        assert result["search"]["type"] == "function"
        assert result["search"]["name"] == "search"
        assert result["search"]["description"] == "Search the web."

    def test_single_sync_tool(self):
        class Agent:
            def calculate(self, x: int, y: int) -> int:
                """Add two numbers."""
                ...
        result = discover_tools(Agent)
        assert "calculate" in result

    def test_multiple_tools(self):
        class Agent:
            async def search(self, q: str):
                """Search."""
                ...
            def calc(self, x: int):
                """Calc."""
                ...
        result = discover_tools(Agent)
        assert "search" in result
        assert "calc" in result

    def test_private_helper_not_discovered(self):
        class Agent:
            async def search(self, q: str):
                """Search."""
                ...
            def _helper(self):
                """Helper."""
                ...
        result = discover_tools(Agent)
        assert "search" in result
        assert "_helper" not in result

    def test_no_public_methods_empty(self):
        class Agent:
            def _private(self):
                """Private."""
                ...
        result = discover_tools(Agent)
        assert result == {}

    def test_missing_docstring_raises(self):
        class Agent:
            async def search(self, q: str): ...
        with pytest.raises(OpenAIAgentDefinitionError, match="missing docstring"):
            discover_tools(Agent)

    def test_multiple_missing_docstrings_all_reported(self):
        class Agent:
            async def tool_a(self): ...
            async def tool_b(self): ...
        with pytest.raises(OpenAIAgentDefinitionError) as exc_info:
            discover_tools(Agent)
        msg = str(exc_info.value)
        assert "tool_a" in msg
        assert "tool_b" in msg

    def test_excluded_methods_not_discovered(self):
        class Agent:
            async def process(self):
                """Process."""
                ...
            async def aclose(self):
                """Close."""
                ...
            def format_history(self):
                """Format LLM."""
                ...
            def format_event(self):
                """Format UI."""
                ...
        result = discover_tools(Agent)
        assert result == {}

    def test_schema_output_format(self):
        class Agent:
            async def search(self, query: str):
                """Search the web."""
                ...
        result = discover_tools(Agent)
        tool = result["search"]
        assert tool["type"] == "function"
        assert tool["name"] == "search"
        assert tool["description"] == "Search the web."
        assert "parameters" in tool
        assert tool["strict"] is True

    def test_inheritance_child_overrides_parent_tool(self):
        class Parent:
            __tool_names__ = ("search",)
            __tools_schema__ = ({"name": "search", "source": "parent"},)

        class Child(Parent):
            async def search(self, query: str):
                """Child search."""
                ...

        result = discover_tools(Child)
        assert result["search"]["name"] == "search"
        assert result["search"]["description"] == "Child search."

    def test_inheritance_child_adds_new_tool(self):
        class Parent:
            __tool_names__ = ("search",)
            __tools_schema__ = ({"name": "search", "type": "function"},)

        class Child(Parent):
            async def calculate(self, x: int):
                """Calculate."""
                ...

        result = discover_tools(Child)
        assert "search" in result
        assert "calculate" in result

    def test_class_with_only_non_callable_attrs(self):
        class Agent:
            name = "my_agent"
            count = 42
        result = discover_tools(Agent)
        assert result == {}

    def test_dedented_docstring(self):
        class Agent:
            async def search(self, query: str):
                """
                Search the web for information.
                Returns relevant results.
                """
                ...
        result = discover_tools(Agent)
        assert result["search"]["description"].startswith("Search the web")

    def test_whitespace_only_docstring_treated_as_missing(self):
        class Agent:
            async def search(self, q: str):
                """   \n\t   """
                ...
        with pytest.raises(OpenAIAgentDefinitionError, match="missing docstring"):
            discover_tools(Agent)

    def test_error_includes_class_name(self):
        class MyCustomAgent:
            async def search(self, q: str): ...
        with pytest.raises(OpenAIAgentDefinitionError, match="MyCustomAgent"):
            discover_tools(MyCustomAgent)

    def test_completely_empty_class(self):
        class Agent:
            pass
        result = discover_tools(Agent)
        assert result == {}

    def test_property_not_discovered(self):
        class Agent:
            @property
            def name(self):
                """Agent name."""
                return "test"
            async def search(self, q: str):
                """Search."""
                ...
        result = discover_tools(Agent)
        assert "search" in result
        assert "name" not in result

    def test_staticmethod_not_discovered(self):
        class Agent:
            @staticmethod
            def helper():
                """Helper utility."""
                ...
            async def search(self, q: str):
                """Search."""
                ...
        result = discover_tools(Agent)
        assert "search" in result
        assert "helper" not in result

    def test_classmethod_not_discovered(self):
        class Agent:
            @classmethod
            def factory(cls):
                """Factory method."""
                ...
            async def search(self, q: str):
                """Search."""
                ...
        result = discover_tools(Agent)
        assert "search" in result
        assert "factory" not in result

    def test_none_docstring_treated_as_missing(self):
        class Agent:
            async def search(self, q: str): ...
        Agent.search.__doc__ = None
        with pytest.raises(OpenAIAgentDefinitionError, match="missing docstring"):
            discover_tools(Agent)


# ──────────────────────────────────────────────────────────────────────────────
# 7. TestSchemaIntegration
# ──────────────────────────────────────────────────────────────────────────────

class TestSchemaIntegration:

    def test_complex_agent_tools(self):
        class Location(BaseModel):
            lat: float
            lng: float

        class Agent:
            async def search(
                self,
                query: str,
                location: Location,
                tags: list[str],
                max_results: int = 10,
                category: Optional[str] = None,
                mode: Literal["fast", "thorough"] = "fast",
            ):
                """Search for items near a location."""
                ...

        result = discover_tools(Agent)
        tool = result["search"]

        assert tool["type"] == "function"
        assert tool["strict"] is True
        assert tool["name"] == "search"
        assert tool["description"] == "Search for items near a location."

        params = tool["parameters"]
        assert_strict_invariants(params)

        props = params["properties"]
        assert props["query"]["type"] == "string"
        assert props["tags"]["type"] == "array"
        assert props["max_results"]["type"] == "integer"
        assert "mode" in props

        cat_schema = props["category"]
        assert "anyOf" in cat_schema
        type_set = {s.get("type") for s in cat_schema["anyOf"]}
        assert "null" in type_set

    def test_deterministic_output(self):
        class Agent:
            async def search(self, query: str, limit: int = 10, tag: str = None):
                """Search."""
                ...

        result1 = discover_tools(Agent)
        result2 = discover_tools(Agent)
        assert result1 == result2

    def test_multiple_tools_all_strict(self):
        class Agent:
            async def search(self, query: str):
                """Search."""
                ...
            def calc(self, x: int, y: int):
                """Calculate."""
                ...
            async def fetch(self, url: str, headers: dict[str, str] = None):
                """Fetch URL."""
                ...

        result = discover_tools(Agent)
        assert len(result) == 3
        for name, tool in result.items():
            assert tool["strict"] is True
            assert_strict_invariants(tool["parameters"])

    def test_agent_with_enum_and_nested_model(self):
        class Priority(str, Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        class TaskInfo(BaseModel):
            title: str
            priority: Priority

        class Agent:
            async def create_task(self, task: TaskInfo, notify: bool = True):
                """Create a new task."""
                ...

        result = discover_tools(Agent)
        tool = result["create_task"]
        assert_strict_invariants(tool["parameters"])
        assert tool["description"] == "Create a new task."

    def test_tuple_param_full_pipeline(self):
        class Agent:
            async def set_coords(self, point: tuple[float, float]):
                """Set coordinates."""
                ...

        result = discover_tools(Agent)
        tool = result["set_coords"]
        assert tool["strict"] is True
        assert_strict_invariants(tool["parameters"])

    def test_dict_str_nested_model_full_pipeline(self):
        class Metric(BaseModel):
            name: str
            value: float

        class Agent:
            async def report(self, metrics: dict[str, Metric]):
                """Submit metrics."""
                ...

        result = discover_tools(Agent)
        tool = result["report"]
        assert tool["strict"] is True
        assert_strict_invariants(tool["parameters"])

    def test_deeply_nested_models(self):
        class Level3(BaseModel):
            value: int

        class Level2(BaseModel):
            nested: Level3

        class Level1(BaseModel):
            child: Level2

        class Agent:
            async def deep(self, data: Level1):
                """Deeply nested."""
                ...

        result = discover_tools(Agent)
        tool = result["deep"]
        assert tool["strict"] is True
        assert_strict_invariants(tool["parameters"])

    def test_list_of_tuples_param(self):
        class Agent:
            async def batch(self, points: list[tuple[float, float]]):
                """Process batch of points."""
                ...

        result = discover_tools(Agent)
        tool = result["batch"]
        assert tool["strict"] is True
        assert_strict_invariants(tool["parameters"])

    def test_optional_nested_model(self):
        class Filter(BaseModel):
            field: str
            value: str

        class Agent:
            async def search(self, query: str, filter: Optional[Filter] = None):
                """Search with optional filter."""
                ...

        result = discover_tools(Agent)
        tool = result["search"]
        assert_strict_invariants(tool["parameters"])

    def test_mixed_excluded_and_tools(self):
        class Agent:
            async def process(self):
                """Process."""
                ...
            async def search(self, q: str):
                """Search."""
                ...
            def _internal(self):
                """Internal."""
                ...
            async def calculate(self, x: int):
                """Calculate."""
                ...
            name = "agent"

        result = discover_tools(Agent)
        assert set(result.keys()) == {"search", "calculate"}
        for tool in result.values():
            assert tool["strict"] is True
            assert_strict_invariants(tool["parameters"])


# ──────────────────────────────────────────────────────────────────────────────
# 8. TestEdgeCases — remaining real-world and boundary scenarios
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    # --- Annotated types (common with Pydantic Field) ---

    def test_annotated_str_param(self):
        from pydantic import Field

        def tool(self, query: Annotated[str, Field(description="search query")]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert_strict_invariants(schema["parameters"])

    def test_annotated_optional_param(self):
        from pydantic import Field

        def tool(self, tag: Annotated[Optional[str], Field(description="filter")] = None): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        props = schema["parameters"]["properties"]
        assert "tag" in props
        assert "tag" in schema["parameters"]["required"]
        assert_strict_invariants(schema["parameters"])

    # --- typing.List / typing.Dict (capital-L backward compat) ---

    def test_typing_list_str(self):
        def tool(self, items: List[str]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert schema["parameters"]["properties"]["items"]["type"] == "array"
        assert_strict_invariants(schema["parameters"])

    def test_typing_dict_str_int(self):
        def tool(self, data: Dict[str, int]): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert "data" in schema["parameters"]["properties"]
        assert_strict_invariants(schema["parameters"])

    # --- Any type parameter ---

    def test_any_type_param(self):
        def tool(self, payload: Any): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        assert "payload" in schema["parameters"]["properties"]

    # --- datetime type parameter ---

    def test_datetime_type_param(self):
        def tool(self, created_at: datetime): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        props = schema["parameters"]["properties"]
        assert "created_at" in props
        assert_strict_invariants(schema["parameters"])

    # --- strict=None omits "strict" key ---

    def test_strict_none_omits_key(self):
        def tool(self, query: str): ...
        schema = _build_function_tool_schema("tool", tool, "Tool.", strict=None)
        assert "strict" not in schema

    # --- Async function directly in _build_function_tool_schema ---

    def test_async_function_schema(self):
        async def search(self, query: str, limit: int = 5): ...
        schema = _build_function_tool_schema("search", search, "Search.")
        props = schema["parameters"]["properties"]
        assert "query" in props
        assert "limit" in props
        assert schema["strict"] is True
        assert_strict_invariants(schema["parameters"])

    # --- Return type annotation ignored ---

    def test_return_type_not_in_params(self):
        def tool(self, query: str) -> dict[str, Any]: ...
        schema = _build_function_tool_schema("tool", tool, "Tool.")
        props = schema["parameters"]["properties"]
        assert len(props) == 1
        assert "query" in props
        assert "return" not in props

    # --- Empty dict into _make_strict ---

    def test_make_strict_empty_dict(self):
        schema: dict = {}
        _make_strict(schema)
        assert schema == {}

    # --- Diamond inheritance in _get_base_classes_tools ---

    def test_diamond_inheritance(self):
        grandparent = type("GP", (), {
            "__tool_names__": ("shared",),
            "__tools_schema__": ({"name": "shared", "source": "gp"},),
        })
        parent_a = type("PA", (grandparent,), {
            "__tool_names__": ("shared", "tool_a"),
            "__tools_schema__": ({"name": "shared", "source": "pa"}, {"name": "tool_a"}),
        })
        parent_b = type("PB", (grandparent,), {
            "__tool_names__": ("shared", "tool_b"),
            "__tools_schema__": ({"name": "shared", "source": "pb"}, {"name": "tool_b"}),
        })
        child = type("Child", (parent_a, parent_b), {})
        # Python C3 MRO: Child -> PA -> PB -> GP -> object
        mro = child.__mro__
        result = _get_base_classes_tools(mro)
        assert "shared" in result
        assert "tool_a" in result
        assert "tool_b" in result
        # PA comes before PB in MRO, so PA's "shared" wins (last writer in reversed MRO)
        assert result["shared"]["source"] == "pa"

    # --- Tool with many parameters (stress test) ---

    def test_many_parameters(self):
        params = ", ".join(f"p{i}: str" for i in range(15))
        ns = {}
        exec(f"def tool(self, {params}): ...", ns)
        fn = ns["tool"]
        schema = _build_function_tool_schema("tool", fn, "Tool.")
        props = schema["parameters"]["properties"]
        assert len(props) == 15
        assert schema["parameters"]["required"] == sorted(props.keys())
        assert_strict_invariants(schema["parameters"])
