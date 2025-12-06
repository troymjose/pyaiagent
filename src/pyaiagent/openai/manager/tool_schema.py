from __future__ import annotations
import re
import enum
import inspect
import dataclasses
from uuid import UUID
from pathlib import Path
from decimal import Decimal
import collections.abc as cabc
from functools import lru_cache
from datetime import date, datetime, time
from dataclasses import fields, is_dataclass
from typing import (Any, Dict, List, Union, Optional, get_args, get_origin, ForwardRef, get_type_hints, Literal,
                    Annotated, TypedDict, NotRequired, Required)

__all__ = ["ToolSchemaManager", ]


@lru_cache(maxsize=2048)
def _hints_cached(obj: Any) -> Dict[str, Any]:
    """Cache get_type_hints (expensive when future annotations / forward refs are used)."""
    return get_type_hints(obj, include_extras=True)


def _unwrap_newtype(t: Any) -> Any:
    """If t is a typing.NewType, return its supertype; else t."""
    return getattr(t, "__supertype__", t)


class ToolSchemaManager:
    """
    High-performance converter from Python type hints and function signatures
    to **JSON Schema** fragments suitable for the **OpenAI Responses API** tool
    definitions.

    ------------------------------------------------------------------------
    What this class does
    ------------------------------------------------------------------------
    • Converts Python annotations (including modern typing features) into
      JSON Schema objects that you can embed under a tool's `"parameters"`.
    • Builds complete Responses-API **function tool** specs
      (`{"type":"function", "name": ..., "parameters": {...}, "strict": true}`).
    • Handles a wide range of Python typing constructs safely and quickly,
      with strong defaults and defensive fallbacks.

    ------------------------------------------------------------------------
    Key capabilities (supported mappings)
    ------------------------------------------------------------------------
    • Primitives:           str → {"type":"string"}, bool/int/float/Decimal
    • Temporal/IDs:         datetime/date/time/UUID → {"type":"string","format":...}
    • Bytes:                bytes/bytearray → base64 string
    • Optional / Union:     Optional[T]/Union[...] → {"anyOf":[... , {"type":"null"}]}
    • Literal:              Literal[...] → {"enum":[...]} (+ inferred "type" when homogeneous)
    • Annotated:            Annotated[T, ...] → unwraps to T
    • Collections:
        - Sequence / list / set / Iterable → {"type":"array","items": ...}
          (sets gain `"uniqueItems": true`)
        - tuple / Tuple:
            · Tuple[T, ...] → variadic array with `"items": schema(T)`
            · Tuple[T1, T2, ...] → fixed arity via `"prefixItems"` + min/maxItems
        - Mapping / dict / Dict[str, V] → {"type":"object","additionalProperties": schema(V)}
          (non-string keys fall back to generic object)
    • Enums:                Enum → {"enum":[member.value, ...]} (+ inferred primitive "type")
    • TypedDict:            Emits `"properties"` + `"required"` from fields;
                            honors Required/NotRequired and total=False
    • Dataclasses:          Emits `"properties"` + `"required"` from fields;
                            resolves annotations via get_type_hints (forward-ref safe)
    • NewType:              Unwrapped to underlying supertype
    • ForwardRef / strings: Treated as string types (safe fallback)

    """

    # ---------- small helpers ----------
    @staticmethod
    def _issubclass_safe(cls: Any, parent: Any) -> bool:
        """
        Safely determine whether one object is a **subclass** of another.

        This is a defensive wrapper around Python’s built-in `issubclass()`
        that prevents it from raising `TypeError` or other exceptions when
        given invalid arguments (e.g., non-class objects, typing constructs,
        or dynamically generated proxies).

        The method is used throughout the schema builder to test whether a
        given type hint or annotation originates from a standard container
        abstract base class (e.g. `collections.abc.Mapping`, `Sequence`, `Set`)
        without risking runtime errors during introspection.

        Parameters
        ----------
        cls : Any
            The candidate object to test. Usually this is a class or a
            type returned by `typing.get_origin()`.

        parent : Any
            The parent class or abstract base class (ABC) that we want
            to check inheritance against.

        Returns
        -------
        bool
            • `True`  → if `cls` is a class and a subclass of `parent`
            • `False` → if `cls` is not a class, or if `issubclass()` raises

        Why this exists
        ----------------
        Python’s built-in `issubclass()` only accepts **class objects** as its
        first argument. If you accidentally pass in an instance, a primitive,
        or a special typing object (like `List[int]`, `Dict[str, T]`, etc.),
        it raises a `TypeError`. This helper avoids those crashes.
        """
        try:
            # Check that cls is a class object (not instance or primitive)
            # and safely call issubclass().  If anything goes wrong,
            # return False instead of raising an error.
            return isinstance(cls, type) and issubclass(cls, parent)
        except Exception:
            # Defensive fallback: never let issubclass() exceptions bubble up.
            # This ensures that schema generation continues even if
            # a weird typing construct or proxy is encountered.
            return False

    @staticmethod
    def _is_typed_dict(t: Any) -> bool:
        """
        Safely determine whether a given object represents a **TypedDict type**.

        This helper wraps the built-in `issubclass()` check for `typing.TypedDict`
        inside a defensive guard. It exists because calling `issubclass()` on
        non-class objects (e.g. `int`, `42`, `None`) raises a `TypeError`.
        We use it throughout the schema builder to decide whether a type hint
        (or annotation) should be expanded into a JSON Schema `"object"` with
        `"properties"` inferred from a `TypedDict` definition.

        Parameters
        ----------
        t : Any
            The object or type to test.

            Common examples:
              • A class created with `TypedDict(...)`
              • A subclass of `TypedDict`
              • A dataclass, builtin, or instance (which will safely return False)

        Returns
        -------
        bool
            • `True`  → if `t` is a `TypedDict` **class** (not instance)
            • `False` → otherwise (including plain dicts, dataclasses, and instances)

        Notes
        -----
        - `issubclass()` throws when given non-type arguments. The `try/except`
          ensures this helper **never raises**, preserving stability during
          automatic schema generation.
        - We check `isinstance(t, type)` first to ensure we only test actual
          type objects (classes), not instances or module references.
        - In Python 3.8–3.11, `TypedDict` classes behave like normal classes
          that inherit from the special `typing.TypedDict` base.
          Example:

              class User(TypedDict):
                  name: str
                  age: int

          This method will correctly return `True` for `User`.
        """
        try:
            # `issubclass()` will raise if `t` is not a class.
            # We therefore check `isinstance(t, type)` first.
            return isinstance(t, type) and issubclass(t, TypedDict)  # type: ignore[arg-type]
        except Exception:
            # Defensive fallback — ensure this utility never breaks schema introspection.
            return False

    @staticmethod
    def _is_dataclass_type(t: Any) -> bool:
        """
        Safely determine whether a given object represents a **dataclass type**.

        This helper is a defensive wrapper around Python's built-in
        `dataclasses.is_dataclass()` that avoids raising unexpected exceptions
        when `t` is not a class, is a proxy object, or defines a custom
        `__getattr__` / `__instancecheck__` that might break normal inspection.

        It is used internally during type introspection to decide whether
        an annotation (or nested type) should be expanded into a JSON Schema
        `object` with `"properties"` derived from the dataclass fields.

        Parameters
        ----------
        t : Any
            The object or type to test.

            Common inputs:
              • A dataclass **class**, e.g. `User`, annotated with `@dataclass`
              • A dataclass **instance**, e.g. `User(name="Alice")`
              • A non-dataclass type or value (returns False safely)

        Returns
        -------
        bool
            • `True`  → if `t` is a dataclass **class**
            • `False` → otherwise (including dataclass instances or non-classes)

        Notes
        -----
        - The standard `dataclasses.is_dataclass()` returns True for both
          dataclass classes *and* dataclass **instances**. In our context
          (schema generation), we only want to treat **classes** as dataclass
          types, because instances don’t define reusable type structure.

        - The `isinstance(t, type)` check ensures we only process actual
          type objects, not instances.

        - Any exceptions raised by malformed objects or meta-class conflicts
          are caught and treated as `False`, so that this helper never breaks
          schema generation pipelines.
        """
        try:
            # Confirm it's a class type and then verify dataclass decoration.
            # The second check uses Python's built-in helper.
            return isinstance(t, type) and is_dataclass(t)
        except Exception:
            # Defensive fallback — never allow an unexpected object
            # to propagate exceptions during type introspection.
            return False

    # ---------- main: Python → JSON Schema ----------
    @staticmethod
    def _py_type_to_json_schema(
            t: Any,
            *,
            allow_null: bool = True,
            additional_props_default: Optional[bool] = None,
            max_depth: int = 10,
            _depth: int = 0,
    ) -> Dict[str, Any]:
        return ToolSchemaManager.__py_type_to_json_schema_cached(
            t, allow_null, additional_props_default, max_depth, _depth
        )

    @staticmethod
    @lru_cache(maxsize=4096)
    def __py_type_to_json_schema_cached(
            t: Any,
            allow_null: bool,
            additional_props_default: Optional[bool],
            max_depth: int,
            _depth: int,
    ) -> Dict[str, Any]:
        """
        Convert a single Python type annotation into an equivalent **JSON Schema**
        fragment that can be used in an OpenAI Responses-API `tool` definition.

        This method is the *cached core* of `_py_type_to_json_schema()`, and it is
        responsible for recursively mapping all supported Python type hints
        (`str`, `Optional[...]`, `Dict[str, int]`, `TypedDict`, dataclasses, etc.)
        into valid JSON Schema objects.

        Because type-hint trees can contain many repeated nodes (and because
        introspection of types is relatively expensive), this function is decorated
        with `functools.lru_cache(maxsize=4096)` for high performance.

        Parameters
        ----------
        t : Any
            The Python type annotation to convert.

        allow_null : bool
            Whether to include `"type": "null"` when `None` or `Optional` appears in the annotation.

        additional_props_default : Optional[bool]
            Controls whether `"additionalProperties"` should be added when generating
            object schemas (e.g., for dataclasses or TypedDicts).
            • `False` → disallow extra fields
            • `True`  → allow them
            • `None`  → omit the key entirely (defer to caller)

        max_depth : int
            Maximum recursion depth to guard against self-referential type structures.

        _depth : int
            Internal recursion counter; increments each time this function calls itself.

        Returns
        -------
        Dict[str, Any]
            A valid JSON Schema fragment describing `t`.

        Notes
        -----
        - Caching is per-type-object and configuration flags; repeated calls for the
          same type are effectively free after the first conversion.
        - This function supports:
              • primitives (`str`, `bool`, `int`, `float`)
              • `Optional` / `Union`
              • `Literal`
              • `Annotated`
              • `Enum` subclasses
              • `datetime`, `date`, `time`, `UUID`, `Decimal`, `bytes`
              • `List`, `Tuple`, `Set`, `Dict`, `Mapping`, and plain built-ins (`list`, `dict`, etc.)
              • `TypedDict`
              • `dataclasses`
        - Any unsupported or unknown type is safely represented as `"type": "string"
        """

        # -------------------------------------------------------------------------
        # 1️⃣  Depth guard – prevents runaway recursion on self-referential types
        # -------------------------------------------------------------------------
        if _depth > max_depth:
            return {}

        # -------------------------------------------------------------------------
        # 2️⃣  Normalize common wrappers early
        # -------------------------------------------------------------------------
        # Unwrap typing.NewType → its supertype
        t = _unwrap_newtype(t)

        # Handle trivial "catch-all" and simple atomic cases
        if t is Any:
            return {}  # unrestricted
        if t is type(None):  # type: ignore[comparison-overlap]
            return {"type": "null"} if allow_null else {}
        if isinstance(t, (ForwardRef, str)):
            # Forward references or string annotations (from __future__.annotations)
            return {"type": "string"}

        # Pathlib paths → strings; regex patterns → strings
        if t is Path:
            return {"type": "string"}
        if t is re.Pattern or getattr(t, "__name__", "") == "Pattern":
            return {"type": "string"}

        origin = get_origin(t)
        args = get_args(t)

        # -------------------------------------------------------------------------
        # 3️⃣  Handle compound typing constructs
        # -------------------------------------------------------------------------

        # --- Annotated[T, ...] : just unwrap the underlying type T
        if origin is Annotated and args:
            return ToolSchemaManager.__py_type_to_json_schema_cached(
                args[0], allow_null, additional_props_default, max_depth, _depth + 1
            )

        # --- Literal[...] : enumerate allowed values
        if origin is Literal and args:
            # Only JSON-serializable literal values are allowed; others → str()
            vals = [
                a if isinstance(a, (str, int, float, bool, type(None))) else str(a)
                for a in args
            ]
            schema: Dict[str, Any] = {"enum": vals}

            # Try to infer a primitive "type" if the enum is homogeneous
            non_null = [v for v in vals if v is not None]
            if non_null:
                if all(isinstance(v, str) for v in non_null):
                    schema["type"] = "string"
                elif all(isinstance(v, bool) for v in non_null):
                    schema["type"] = "boolean"
                elif all(isinstance(v, int) for v in non_null):
                    schema["type"] = "integer"
                elif all(isinstance(v, (int, float)) for v in non_null):
                    schema["type"] = "number"
            else:
                schema["type"] = "null"
            return schema

        # --- Union[..., ...] : merge multiple schemas, handle Optional/None
        if origin is Union and args:
            subs = [
                ToolSchemaManager.__py_type_to_json_schema_cached(
                    a, allow_null, additional_props_default, max_depth, _depth + 1
                )
                for a in args
                if not (a is type(None) and not allow_null)
            ]
            # Simplify Union[X] → X when only one branch remains
            return subs[0] if len(subs) == 1 else {"anyOf": subs}

        # -------------------------------------------------------------------------
        # 4️⃣  Primitive scalars
        # -------------------------------------------------------------------------
        if t is str:
            return {"type": "string"}
        if t is bool:
            return {"type": "boolean"}
        if t is int:
            return {"type": "integer"}
        if t is float or t is Decimal:
            return {"type": "number"}

        # Temporal and identifier types → string with format
        if t is datetime:
            return {"type": "string", "format": "date-time"}
        if t is date:
            return {"type": "string", "format": "date"}
        if t is time:
            return {"type": "string", "format": "time"}
        if t is UUID:
            return {"type": "string", "format": "uuid"}
        if t in (bytes, bytearray):
            return {"type": "string", "contentEncoding": "base64"}

        # -------------------------------------------------------------------------
        # 5️⃣  Enums : list of allowed values inferred from members
        # -------------------------------------------------------------------------
        if isinstance(t, type) and ToolSchemaManager._issubclass_safe(t, enum.Enum):
            vals = [e.value for e in t]
            schema: Dict[str, Any] = {"enum": vals}
            if all(isinstance(v, str) for v in vals):
                schema["type"] = "string"
            elif all(isinstance(v, bool) for v in vals):
                schema["type"] = "boolean"
            elif all(isinstance(v, int) for v in vals):
                schema["type"] = "integer"
            elif all(isinstance(v, (int, float)) for v in vals):
                schema["type"] = "number"
            return schema

        # -------------------------------------------------------------------------
        # 6️⃣  Sequences / Sets (including plain list/set) → JSON arrays
        # -------------------------------------------------------------------------
        if (
                (origin is not None and ToolSchemaManager._issubclass_safe(origin,
                                                                           cabc.Sequence) and origin is not tuple)
                or (ToolSchemaManager._issubclass_safe(t, cabc.Sequence) and t not in (str, bytes, bytearray, tuple))
        ):
            item_t = args[0] if args else Any
            schema: Dict[str, Any] = {
                "type": "array",
                "items": ToolSchemaManager.__py_type_to_json_schema_cached(
                    item_t, allow_null, additional_props_default, max_depth, _depth + 1
                ),
            }
            # If it's a set, mark uniqueness
            if (
                    (origin is not None and ToolSchemaManager._issubclass_safe(origin, cabc.Set))
                    or ToolSchemaManager._issubclass_safe(t, cabc.Set)
            ):
                schema["uniqueItems"] = True
            return schema

        # -------------------------------------------------------------------------
        # 7️⃣  Tuples → JSON arrays (fixed or variable length)
        # -------------------------------------------------------------------------
        if origin is tuple or t is tuple:
            if not args:
                return {"type": "array"}
            if len(args) == 2 and args[1] is Ellipsis:
                # Homogeneous tuple (Tuple[T, ...])
                return {
                    "type": "array",
                    "items": ToolSchemaManager.__py_type_to_json_schema_cached(
                        args[0], allow_null, additional_props_default, max_depth, _depth + 1
                    ),
                }
            # Fixed-length tuple (Tuple[T1, T2, ...])
            prefix = [
                ToolSchemaManager.__py_type_to_json_schema_cached(
                    a, allow_null, additional_props_default, max_depth, _depth + 1
                )
                for a in args
            ]
            return {
                "type": "array",
                "prefixItems": prefix,
                "minItems": len(prefix),
                "maxItems": len(prefix),
            }

        # -------------------------------------------------------------------------
        # 8️⃣  Mappings / Dicts → JSON objects
        # -------------------------------------------------------------------------
        if (
                (origin is not None and ToolSchemaManager._issubclass_safe(origin, cabc.Mapping))
                or ToolSchemaManager._issubclass_safe(t, cabc.Mapping)
        ):
            if args and len(args) == 2 and args[0] in (str, Any):
                value_t = args[1]
                return {
                    "type": "object",
                    "additionalProperties": ToolSchemaManager.__py_type_to_json_schema_cached(
                        value_t, allow_null, additional_props_default, max_depth, _depth + 1
                    ),
                }
            # Unknown key/value types or bare dict → generic object
            return {"type": "object"}

        # -------------------------------------------------------------------------
        # 9️⃣  TypedDicts → JSON objects with explicit "properties"/"required"
        # -------------------------------------------------------------------------
        if ToolSchemaManager._is_typed_dict(t):
            hints = _hints_cached(t)
            total = getattr(t, "__total__", True)
            props: Dict[str, Any] = {}
            required: List[str] = []
            for k, at in hints.items():
                req = total
                o = get_origin(at)
                if o is NotRequired:
                    req = False
                    at = get_args(at)[0]
                elif o is Required:
                    req = True;
                    at = get_args(at)[0]
                props[k] = ToolSchemaManager.__py_type_to_json_schema_cached(
                    at, allow_null, additional_props_default, max_depth, _depth + 1
                )
                if req:
                    required.append(k)
            out: Dict[str, Any] = {"type": "object", "properties": props}
            if required:
                out["required"] = required
            if additional_props_default is not None:
                out["additionalProperties"] = additional_props_default
            return out

        # -------------------------------------------------------------------------
        # 🔟  Dataclasses → JSON objects with fields as properties
        # -------------------------------------------------------------------------
        if ToolSchemaManager._is_dataclass_type(t):
            hints = _hints_cached(t)
            props: Dict[str, Any] = {}
            required: List[str] = []
            for f in fields(t):
                at = hints.get(f.name, f.type)
                props[f.name] = ToolSchemaManager.__py_type_to_json_schema_cached(
                    at, allow_null, additional_props_default, max_depth, _depth + 1
                )
                # Required if no default or default_factory defined
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                    required.append(f.name)
            out: Dict[str, Any] = {"type": "object", "properties": props}
            if required:
                out["required"] = required
            if additional_props_default is not None:
                out["additionalProperties"] = additional_props_default
            return out

        # -------------------------------------------------------------------------
        # 11️⃣  Fallback – unknown or unsupported type → treat as string
        # -------------------------------------------------------------------------
        return {"type": "string"}

    # ---------- build OpenAI Responses-API tool ----------
    @staticmethod
    def build_function_tool_schema(func_name: str,
                                   func: Any,
                                   description: str,
                                   *,
                                   additional_properties: bool = False,
                                   strict: bool = True,
                                   ) -> Dict[str, Any]:
        """
        Build a **Responses API-compliant tool schema** for a given Python function.

        This method inspects a Python callable (usually an async tool method)
        and converts its signature + type hints into a JSON Schema that the
        OpenAI **Responses API** can understand when defining `tools=[...]`.
        """
        # ---------------------------------------------------------
        # 1️⃣ Extract function signature and resolved type hints
        # ---------------------------------------------------------
        sig = inspect.signature(func)
        # Cached, forward-ref-safe version of typing.get_type_hints
        hints = _hints_cached(func)

        # JSON Schema components
        props: Dict[str, Any] = {}  # "properties" block
        required: List[str] = []  # "required" field list

        # ---------------------------------------------------------
        # 2️⃣ Iterate through parameters and build schema per arg
        # ---------------------------------------------------------
        for name, param in sig.parameters.items():
            # Skip implicit or unrepresentable parameters
            if name in {"self", "cls"}:
                continue
            if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,  # *args
                    inspect.Parameter.VAR_KEYWORD,  # **kwargs
                    inspect.Parameter.POSITIONAL_ONLY,  # positional-only (rare)
            ):
                continue

            # Determine annotation (default to str if missing)
            ann = hints.get(name, str)
            props[name] = ToolSchemaManager._py_type_to_json_schema(ann)

            # Determine if parameter is required or optional
            # Required if no default value provided
            if param.default is inspect._empty or param.default is ...:
                required.append(name)
            elif param.default is None:
                # Parameter has default=None → implicitly nullable
                sch = props[name]
                # Add "anyOf: [schema, null]" if not already nullable
                if "anyOf" not in sch and sch.get("type") != "null":
                    props[name] = {"anyOf": [sch, {"type": "null"}]}

        # ---------------------------------------------------------
        # 3️⃣ Assemble the "parameters" object (JSON Schema root)
        # ---------------------------------------------------------
        parameters: Dict[str, Any] = {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": additional_properties,
        }

        # ---------------------------------------------------------
        # 4️⃣ Build the final tool schema for the Responses API
        # ---------------------------------------------------------
        tool = {
            "type": "function",  # Required top-level key
            "name": func_name,  # Tool name as invoked by model
            "description": description or f"Tool {func_name}",
            "parameters": parameters,  # JSON Schema from above
        }

        # Add enforcement flag if provided
        if strict is not None:
            tool["strict"] = bool(strict)

        return tool
