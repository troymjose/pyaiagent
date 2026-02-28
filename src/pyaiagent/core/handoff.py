"""Agent handoff specification and tool schema generation.

A *handoff* lets one agent delegate work to another agent as a tool call.
The orchestrating agent's LLM decides when and what to delegate; the
framework handles instantiation, execution, and cleanup of the sub-agent.

Public API:
    handoff(agent_cls, *, description=None)  — mark an entry for customisation
    HandoffSpec                              — the resulting spec dataclass

Internal helpers called from :mod:`pyaiagent.core.agent`:
    discover_handoffs(cls)              — scan ``class Agents:`` on *cls*
    build_handoff_tool_schema(name, description) — produce a tool schema
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Dict

from pyaiagent.core.exceptions import AgentDefinitionError

__all__ = ["handoff", "HandoffSpec"]


@dataclass(frozen=True, slots=True)
class HandoffSpec:
    """Immutable specification for a single agent handoff.

    Attributes:
        agent_cls:   The agent subclass to delegate to.
        description: Optional tool description override.  When ``None``,
                     the sub-agent's class docstring is used.
    """
    agent_cls: type
    description: str | None = None


def handoff(agent_cls: type, *, description: str | None = None) -> HandoffSpec:
    """Create a :class:`HandoffSpec` for use inside ``class Agents:``.

    Use this when you need to customise the tool description shown to the
    orchestrating LLM.  For the simple case, assigning the agent class
    directly is sufficient.

    Args:
        agent_cls:   An agent subclass.
        description: Custom tool description.  Defaults to the
                     sub-agent's class docstring.

    Returns:
        A frozen :class:`HandoffSpec` instance.

    Example::

        class Orchestrator(OpenAIAgent):
            \"\"\"You coordinate work.\"\"\"

            class Agents:
                simple   = Worker                           # uses Worker docstring
                custom   = handoff(Worker, description="…") # custom description
    """
    return HandoffSpec(agent_cls=agent_cls, description=description)


# ──────────────────────────────────────────────────────────────────────────────
# Schema generation
# ──────────────────────────────────────────────────────────────────────────────

def build_handoff_tool_schema(name: str, description: str) -> Dict[str, Any]:
    """Build a tool schema for a handoff.

    The schema always exposes a single ``input`` string parameter — the
    task description the orchestrating LLM sends to the sub-agent.

    Args:
        name:        Tool name (the attribute name from ``class Agents:``).
        description: Tool description shown to the LLM.

    Returns:
        A dict matching the tool schema format.
    """
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The task or message to delegate to this agent.",
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        "strict": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

def _is_agent_class(value: Any) -> bool:
    """Check if *value* is an agent subclass via duck typing.

    Uses the presence of ``__instruction__`` (set by
    ``PyAiAgent.__init_subclass__``) rather than ``issubclass`` to
    remain robust against ``importlib.reload`` scenarios where the
    base class identity may change.
    """
    return isinstance(value, type) and hasattr(value, "__instruction__")


def discover_handoffs(cls: type) -> tuple[
    Dict[str, HandoffSpec], Dict[str, Dict[str, Any]]
]:
    """Scan ``class Agents:`` on *cls* and return handoff specs + tool schemas.

    Walks the inner ``Agents`` class (if present) and collects:
      - Plain agent class references  → wrapped into a :class:`HandoffSpec`
      - :class:`HandoffSpec` instances → used directly

    Agent class validation uses duck typing (``_is_agent_class``) rather
    than ``issubclass`` to remain robust against ``importlib.reload``.

    Inheritance of the ``Agents`` inner class is supported: if
    ``class Agents(Parent.Agents):`` is used, parent entries are
    visible via normal Python attribute lookup.

    Args:
        cls: The agent class being defined.

    Returns:
        A 2-tuple ``(handoffs, schemas)`` where *handoffs* maps
        tool names to :class:`HandoffSpec` and *schemas* maps tool
        names to tool schema dicts.
    """
    agents_inner = cls.__dict__.get("Agents")
    if agents_inner is None:
        return {}, {}

    errors: list[str] = []
    handoffs: Dict[str, HandoffSpec] = {}
    schemas: Dict[str, Dict[str, Any]] = {}

    for name in dir(agents_inner):
        if name.startswith("_"):
            continue
        value = getattr(agents_inner, name)

        if isinstance(value, HandoffSpec):
            spec = value
            if not _is_agent_class(spec.agent_cls):
                errors.append(
                    f"Agent '{name}': handoff() agent_cls must be an agent subclass, "
                    f"got {type(spec.agent_cls).__name__}.")
                continue
        elif _is_agent_class(value):
            spec = HandoffSpec(agent_cls=value)
        else:
            errors.append(
                f"Agent '{name}': expected an agent subclass or handoff(), "
                f"got {type(value).__name__}.")
            continue

        description = spec.description or textwrap.dedent(spec.agent_cls.__doc__ or "").strip()
        if not description:
            errors.append(
                f"Agent '{name}': agent class '{spec.agent_cls.__name__}' has no docstring. "
                f"Add a docstring or provide a description via handoff().")
            continue

        handoffs[name] = spec
        schemas[name] = build_handoff_tool_schema(name, description)

    if errors:
        raise AgentDefinitionError(cls_name=cls.__name__, errors=errors)

    return handoffs, schemas
