"""Tool discovery and schema generation — re-exported from core.

The canonical implementation lives in :mod:`pyaiagent.core.tools`.
This module provides backward-compatible imports for existing code
that imports from ``pyaiagent.openai.tools``.
"""
from pyaiagent.core.tools import (                       # noqa: F401
    discover_tools,
    _make_strict,
    _already_accepts_none,
    _build_function_tool_schema,
    _get_base_classes_tools,
    _is_tool_method,
    _EXCLUDED_METHODS,
    _MAX_STRICT_DEPTH,
    _UNSUPPORTED_PARAM_KINDS,
    _NONE_TYPE,
    _UNION_ORIGINS,
)

# Keep the original __all__ for backward compatibility
from pyaiagent.core.tools import __all__                 # noqa: F401

# Re-export the old exception name so callers that import it transitively
# from this module via `from pyaiagent.openai.tools import ...` still work.
from pyaiagent.openai.exceptions import OpenAIAgentDefinitionError  # noqa: F401
