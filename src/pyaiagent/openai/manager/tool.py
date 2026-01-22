import inspect
import textwrap
from typing import Any
from pyaiagent.openai.manager.tool_schema import ToolSchemaManager
from pyaiagent.openai.exceptions.definition import OpenAIAgentDefinitionError

__all__ = ["OpenAIAgentToolManager", ]

# Framework methods that should never be treated as tools
_EXCLUDED_METHODS = frozenset({
    'process', 'aclose', 'format_llm_message', 'format_ui_message'
})


class OpenAIAgentToolManager:
    """Discovers and builds tool schemas from agent class methods."""

    @staticmethod
    def _get_base_classes_tools(mro: tuple[type, ...]) -> dict[str, dict[str, Any]]:
        """Get all tool schemas from base classes in MRO."""
        merged_tools: dict[str, dict[str, Any]] = {}
        for base in reversed(mro[1:]):
            for tool_name, tool_schema in zip(getattr(base, "__tool_names__", ()),
                                              getattr(base, "__tools_schema__", ())):
                merged_tools[tool_name] = tool_schema
        return merged_tools

    @staticmethod
    def _is_tool_method(name: str, obj: Any) -> bool:
        """
        Check if a method should be treated as a tool.
        
        Tools can be:
        - Async methods (async def) - for I/O-bound work
        - Sync methods (def) - for CPU-bound work (will be run in thread pool)
        
        Excluded:
        - Private methods (starting with _)
        - Framework hook methods (process, aclose, format_*_message)
        """
        if name.startswith("_"):
            return False
        if name in _EXCLUDED_METHODS:
            return False
        # Accept both async and sync methods
        return inspect.iscoroutinefunction(obj) or inspect.isfunction(obj)

    @staticmethod
    def create(cls) -> dict[str, dict[str, Any]]:
        """Discover tools from class and merge with inherited tools."""
        errors: list[str] = []
        declared_tools: list[dict[str, Any]] = []
        tool_names: list[str] = []
        for name, obj in cls.__dict__.items():
            if not OpenAIAgentToolManager._is_tool_method(name, obj):
                continue
            tool_doc = getattr(obj, "__doc__", "")
            # Clean and normalize docstring text
            cleaned_tool_doc = textwrap.dedent(tool_doc or "").strip()
            if not cleaned_tool_doc:
                errors.append(
                    f"Tool '{name}' is missing docstring. Add a triple-quoted docstring as tool description.")
                continue
            declared_tools.append(ToolSchemaManager.build_function_tool_schema(name, obj, cleaned_tool_doc))
            tool_names.append(name)
        if errors:
            raise OpenAIAgentDefinitionError(cls_name=cls.__name__, errors=errors)

        # Get tools from base classes in MRO and merge
        merged_tools: dict[str, dict[str, Any]] = OpenAIAgentToolManager._get_base_classes_tools(mro=cls.__mro__)
        # Add current class's tools to the merged dict, this will override tools from base classes
        for n, sch in zip(tool_names, declared_tools):
            merged_tools[n] = sch

        # Return the final merged tools dictionary
        return merged_tools
