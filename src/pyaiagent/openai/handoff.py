"""Agent handoff specification — re-exported from core.

The canonical implementation lives in :mod:`pyaiagent.core.handoff`.
This module provides backward-compatible imports for existing code
that imports from ``pyaiagent.openai.handoff``.
"""
from pyaiagent.core.handoff import (                     # noqa: F401
    HandoffSpec,
    handoff,
    build_handoff_tool_schema,
    _is_agent_class,
    discover_handoffs,
)

# Keep the original __all__ for backward compatibility
from pyaiagent.core.handoff import __all__                # noqa: F401

# Re-export the old exception name for any transitive imports
from pyaiagent.openai.exceptions import OpenAIAgentDefinitionError  # noqa: F401
