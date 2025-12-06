import inspect
import textwrap
from typing import Any

from pyaiagent.openai.manager.tool_schema import ToolSchemaManager
from pyaiagent.openai.exceptions.definition import OpenAIAgentDefinitionError

__all__ = ["OpenAIAgentToolManager", ]


class OpenAIAgentToolManager:
    """Discovers and builds tool schemas from agent class methods."""

    @staticmethod
    def _get_base_classes_tools(mro: tuple[type, ...]) -> dict[str, dict[str, Any]]:
        """Get all tool schemas from base classes in MRO."""
        merged_tools: dict[str, dict[str, Any]] = {}
        for base in reversed(mro[1:]):
            for tool_name, tool_schema in zip(
                    getattr(base, "__tool_names__", ()),
                    getattr(base, "__tools_schema__", ())
            ):
                merged_tools[tool_name] = tool_schema
        return merged_tools

    @staticmethod
    def create(cls) -> dict[str, dict[str, Any]]:
        """Discover tools from class and merge with inherited tools."""
        errors: list[str] = []
        declared_tools: list[dict[str, Any]] = []
        tool_names: list[str] = []
        for name, obj in cls.__dict__.items():
            if name.startswith("_"):
                continue
            if inspect.iscoroutinefunction(obj):
                tool_doc = getattr(obj, "__doc__", "")
                # Clean and normalize docstring text
                cleaned_tool_doc = textwrap.dedent(tool_doc or "").strip()
                if not cleaned_tool_doc:
                    errors.append(
                        f"Tool '{name}' is missing docstring. Add a triple-quoted docstring as tool description.")
                declared_tools.append(ToolSchemaManager.build_function_tool_schema(name, obj, cleaned_tool_doc))
                tool_names.append(name)
        if errors:
            raise OpenAIAgentDefinitionError(cls.__name__, errors)

        # Get tools from base classes in MRO and merge
        merged_tools: dict[str, dict[str, Any]] = OpenAIAgentToolManager._get_base_classes_tools(mro=cls.__mro__)
        # Add current class's tools to the merged dict, this will override tools from base classes
        for n, sch in zip(tool_names, declared_tools):
            merged_tools[n] = sch
        return merged_tools
