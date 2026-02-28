"""Provider-agnostic base class for declarative AI agents.

:class:`PyAiAgent` is the foundation that all provider-specific agent
classes (``OpenAIAgent``, ``GeminiAgent``, etc.) inherit from.  It handles
everything that does **not** depend on a particular LLM provider:

* Class-level setup (``__init_subclass__``): instruction resolution from
  docstrings, configuration merging, tool discovery, and handoff discovery.
* Lifecycle management: ``aclose()``, async context manager.
* Utility methods: instruction template formatting, JSON serialisation,
  handoff executor creation, and agent definition introspection.

Provider subclasses add their own ``_ensure_ready()``, ``process()``,
LLM-call logic, and response-parsing.

Usage::

    from pyaiagent.openai.agent import OpenAIAgent   # concrete provider

    class MyAgent(OpenAIAgent):
        \"\"\"You are a helpful assistant.\"\"\"

        async def greet(self, name: str) -> dict:
            \"\"\"Greet a user.\"\"\"
            return {"greeting": f"Hello, {name}!"}
"""
from __future__ import annotations

import re
import asyncio
import textwrap
from typing import Any, Dict

import orjson

from pyaiagent.core.config import merge_config
from pyaiagent.core.tools import discover_tools
from pyaiagent.core.handoff import HandoffSpec, discover_handoffs
from pyaiagent.core.exceptions import (
    AgentDefinitionError,
    AgentClosedError,
    InstructionKeyError,
)

__all__ = ["PyAiAgent"]


def _resolve_instruction(cls) -> str:
    """Extract and normalize the agent instruction from the class docstring.

    The docstring is dedented and stripped so that multi-line instructions
    authored with natural Python indentation are passed to the LLM as
    clean, unindented text.

    Args:
        cls: The agent subclass whose ``__doc__`` is read.

    Returns:
        The cleaned instruction string.

    Raises:
        AgentDefinitionError: If the class has no docstring or the
            docstring is empty/whitespace-only.
    """
    instruction = textwrap.dedent(cls.__doc__ or "").strip()
    if not instruction:
        raise AgentDefinitionError(
            cls_name=cls.__name__,
            errors=["Missing class docstring. Add a triple-quoted docstring as agent instruction."])
    return instruction


class PyAiAgent:
    """Provider-agnostic base class for all pyaiagent agents.

    Subclass a concrete provider class (e.g. ``OpenAIAgent``) rather than
    this class directly.  ``PyAiAgent`` cannot make LLM calls on its own.

    Class-level attributes set by ``__init_subclass__``:
        __agent_name__:      Cached ``cls.__name__``.
        __instruction__:     Resolved and dedented docstring.
        __config_kwargs__:   Merged configuration dict.
        __tool_names__:      Tuple of discovered tool method names.
        __tools_schema__:    Tuple of tool schemas.
        __handoffs__:        Dict mapping handoff names to
                             :class:`~pyaiagent.core.handoff.HandoffSpec`.
        __handoff_names__:   Tuple of handoff tool names.
        __handoff_schemas__: Tuple of handoff tool schemas.
    """

    # Marker that prevents __init_subclass__ from running setup on this
    # class.  Any class that declares it directly is treated as abstract
    # (no instruction resolution, no tool discovery, no instantiation).
    _abstract_agent = True

    __slots__ = (
        "_ready_lock",       # asyncio.Lock for double-checked lazy init
        "_ready",            # bool — True after _ensure_ready() completes
        "_closed",           # bool — True after aclose(); blocks process()
        "_config",           # provider-specific config dataclass instance
        "_tool_functions",   # dict[str, Callable] — tool name → bound method / handoff executor
    )

    # ── Class creation ────────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs):
        """Prepare the agent subclass at class-creation time.

        This hook fires once per subclass definition and:

        1. Caches the class name for fast runtime access.
        2. Resolves the instruction template from the docstring.
        3. Merges configuration from the optional inner ``Config`` class.
        4. Discovers tool methods and generates their JSON schemas.
        5. Discovers agent handoffs from ``class Agents:`` and generates
           their tool schemas.

        Abstract intermediate classes (those that define ``_abstract_agent``
        directly) are skipped.
        """
        super().__init_subclass__(**kwargs)

        # Skip abstract intermediates (PyAiAgent, OpenAIAgent, etc.)
        # — they have no docstring-based instruction or tools to discover.
        if "_abstract_agent" in cls.__dict__:
            return

        # ── Step 1: Identity & instruction ──────────────────────────
        cls.__agent_name__ = cls.__name__
        cls.__instruction__ = _resolve_instruction(cls=cls)

        # ── Step 2: Configuration ───────────────────────────────────
        # Delegates to provider-specific _resolve_config() which may
        # add validation (e.g. OpenAI checks field types/ranges).
        cls.__config_kwargs__ = cls._resolve_config()

        # ── Step 3: Tool discovery ──────────────────────────────────
        # Scans cls for public methods with docstrings, builds JSON
        # schemas, and merges with tools inherited from base classes.
        _tools = discover_tools(cls=cls)
        cls.__tool_names__ = tuple(_tools.keys())
        cls.__tools_schema__ = tuple(_tools.values())
        del _tools

        # ── Step 4: Handoff discovery ───────────────────────────────
        # Scans the inner "class Agents:" for agent references and
        # handoff() specs defined directly on this class.
        _own_handoffs, _own_schemas = discover_handoffs(cls)

        # Collect inherited handoffs from all base classes.
        # reversed(mro[1:]) walks most-base-first so child overrides parent.
        _inherited_handoffs: dict[str, HandoffSpec] = {}
        for base in reversed(cls.__mro__[1:]):
            _inherited_handoffs.update(getattr(base, "__handoffs__", {}))

        # Merge: own handoffs override inherited ones with the same name.
        _all_handoffs = {**_inherited_handoffs, **_own_handoffs}

        # ── Step 5: Collision check ─────────────────────────────────
        # A tool method and a handoff cannot share the same name —
        # the LLM would not be able to distinguish between them.
        collisions = set(cls.__tool_names__) & set(_all_handoffs)
        if collisions:
            raise AgentDefinitionError(
                cls_name=cls.__name__,
                errors=[f"Name collision between tool methods and agent handoffs: "
                        f"{', '.join(sorted(collisions))}. "
                        f"Tool methods and agent handoffs must have unique names."])

        # ── Step 6: Merge inherited handoff schemas ─────────────────
        # Handoff schemas are stored separately from HandoffSpecs,
        # so we reconstruct the name→schema mapping from each base.
        _inherited_schemas: dict[str, dict] = {}
        for base in reversed(cls.__mro__[1:]):
            base_handoffs = getattr(base, "__handoffs__", {})
            base_schemas = dict(zip(
                getattr(base, "__handoff_names__", ()),
                getattr(base, "__handoff_schemas__", ())))
            for name in base_handoffs:
                if name in base_schemas:
                    _inherited_schemas[name] = base_schemas[name]

        _all_schemas = {**_inherited_schemas, **_own_schemas}

        # Safety check: every handoff must have a corresponding schema.
        missing = set(_all_handoffs) - set(_all_schemas)
        if missing:
            raise AgentDefinitionError(
                cls_name=cls.__name__,
                errors=[f"Handoff '{n}' has no tool schema — this is a framework bug. "
                        f"Please report it." for n in sorted(missing)])

        # ── Step 7: Cache results on the class ──────────────────────
        cls.__handoffs__ = _all_handoffs
        cls.__handoff_names__ = tuple(_all_handoffs.keys())
        cls.__handoff_schemas__ = tuple(_all_schemas[n] for n in _all_handoffs)

    @classmethod
    def _resolve_config(cls) -> dict[str, Any]:
        """Merge configuration from the inner ``Config`` class.

        Provider subclasses override this to add validation (e.g.
        ``OpenAIAgent`` routes to ``ConfigResolver.resolve()``).
        The default implementation performs a plain merge without
        validation — suitable for providers that haven't built
        their own config resolver yet.

        Uses ``cls.__dict__.get("Config")`` (not ``getattr``) to only
        pick up the ``Config`` class defined directly on *this* class.
        Inherited config is already captured in ``__config_kwargs__``
        from the parent via the MRO.
        """
        return merge_config(
            inner_config_cls=cls.__dict__.get("Config"),
            parent_config_kwargs=getattr(cls, "__config_kwargs__", None),
        )

    # ── Introspection ─────────────────────────────────────────────────

    @classmethod
    def get_definition(cls) -> dict:
        """Return the complete agent definition for inspection or debugging.

        The returned dict contains the resolved instruction, merged
        configuration, and tool schemas.  Useful for logging, testing,
        or building admin dashboards.

        Returns:
            A dict with keys ``agent_name``, ``instruction``, ``config``,
            ``tools``, and optionally ``agents``.

        Raises:
            TypeError: If called on an abstract agent class.

        Example::

            print(MyAgent.get_definition())
        """
        # Abstract classes (PyAiAgent, OpenAIAgent) don't have
        # __agent_name__ because __init_subclass__ skipped them.
        if not hasattr(cls, "__agent_name__"):
            raise TypeError(
                f"{cls.__name__} is an abstract agent class; "
                f"call get_definition() on a concrete subclass.")

        definition: dict = {
            "agent_name": cls.__agent_name__,
            "instruction": cls.__instruction__,
            "config": cls.__config_kwargs__,
            "tools": dict(zip(cls.__tool_names__, cls.__tools_schema__)),
        }

        # Only include "agents" key if handoffs are configured.
        if cls.__handoffs__:
            definition["agents"] = {
                name: {
                    "agent_class": spec.agent_cls.__name__,
                    "description": spec.description or spec.agent_cls.__instruction__,
                }
                for name, spec in cls.__handoffs__.items()
            }
        return definition

    # ── Instance creation ─────────────────────────────────────────────

    def __new__(cls, *args, **kwargs):
        """Allocate a new agent instance and initialize base slot defaults.

        Slot initialization happens here (rather than in ``__init__``)
        so that the base state is always consistent regardless of
        whether subclasses override ``__init__``.  Provider subclasses
        (e.g. ``OpenAIAgent``) chain via ``super().__new__()`` and add
        their own slots in their ``__new__``.
        """
        self = super().__new__(cls)
        self._ready_lock = asyncio.Lock()   # lazily binds to the running loop when first awaited
        self._ready = False                 # flipped to True at the end of _ensure_ready()
        self._closed = False                # flipped to True by aclose()
        self._config = None                 # set by provider's _ensure_ready()
        self._tool_functions = None         # set by provider's _ensure_ready()
        return self

    def __init__(self):
        """Guard against direct instantiation of abstract agent classes.

        Any class that declares ``_abstract_agent = True`` in its own
        ``__dict__`` (i.e. ``PyAiAgent``, ``OpenAIAgent``, or any
        user-defined intermediate) cannot be instantiated directly.

        Raises:
            TypeError: If the concrete type is an abstract agent class.
        """
        if "_abstract_agent" in type(self).__dict__:
            raise TypeError(
                f"{type(self).__name__} is an abstract agent class; "
                f"subclass it before use.")

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _to_str(value: Any) -> str:
        """Serialize *value* to a JSON string, falling back to ``str()``.

        Used to convert tool results (dicts, lists, scalars) into a
        string suitable for the ``output`` field of a
        ``function_call_output`` message.
        """
        if isinstance(value, str):
            return value
        try:
            # orjson is ~3-6x faster than json.dumps for dicts/lists.
            return orjson.dumps(value).decode('utf-8')
        except Exception:
            # Fallback for types orjson can't handle (sets, custom objects, etc.)
            return str(value)

    def _format_instruction(self, instruction_params: Dict[str, str] | None = None) -> str:
        """Substitute ``{placeholder}`` tokens in the instruction template.

        Placeholder syntax:

        * ``{name}`` -- replaced with the value from *instruction_params*.
        * ``{{name}}`` -- escaped; rendered as the literal text ``{name}``.

        Only valid Python identifiers are recognised as placeholders, so
        JSON braces like ``{"key": "value"}`` pass through untouched.

        When ``strict_instruction_params`` is enabled in the agent
        ``Config``, any unresolved placeholder raises
        :class:`~pyaiagent.core.exceptions.InstructionKeyError`.
        In non-strict mode, unresolved placeholders are left as-is.

        Args:
            instruction_params: A mapping of placeholder names to their
                replacement values.

        Returns:
            The fully resolved instruction string.

        Raises:
            InstructionKeyError: In strict mode, if a placeholder has no
                matching key in *instruction_params*.
        """
        instruction = self.__instruction__

        # Read from the class-level config dict (always available), not
        # the instance-level _config (which requires _ensure_ready).
        strict_mode = self.__config_kwargs__.get("strict_instruction_params", False)

        # Fast path: no params and no strict mode → return the raw template.
        if not instruction_params and not strict_mode:
            return instruction

        def replace_match(match: re.Match) -> str:
            # Regex groups: (optional leading {)(identifier)(optional trailing })
            prefix, key, suffix = match.group(1), match.group(2), match.group(3)

            # Escaped placeholder: {{name}} → literal {name}
            if prefix == '{' and suffix == '}':
                return f'{{{key}}}'

            # Substitute if the key exists in the provided params.
            if instruction_params and key in instruction_params:
                return f'{prefix}{instruction_params[key]}{suffix}'

            # Strict mode: missing key is an error.
            if strict_mode:
                raise KeyError(key)

            # Non-strict: leave unresolved placeholders as-is.
            return match.group(0)

        try:
            # Pattern matches: optional "{", then "{identifier}", then optional "}".
            # Only valid Python identifiers are captured, so JSON braces
            # like {"key": "value"} pass through untouched.
            instruction = re.sub(r'(\{?)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(\}?)', replace_match, instruction)
            return instruction
        except KeyError as exc:
            # Convert the raw KeyError into a descriptive framework exception.
            raise InstructionKeyError(agent_name=self.__agent_name__, key=str(exc)) from exc

    @staticmethod
    def _make_handoff_executor(spec: HandoffSpec):
        """Create an async callable that delegates work to a sub-agent.

        The returned coroutine function matches the tool execution
        interface: it accepts keyword arguments (``input``) and returns
        a dict.  The sub-agent is instantiated, run, and closed within
        a single invocation — no state leaks across calls.
        """
        agent_cls = spec.agent_cls

        async def _execute_handoff(input: str = "") -> dict:
            agent = None
            try:
                # Fresh instance per invocation — no shared state.
                agent = agent_cls()
                result = await agent.process(input=input)
                return {"output": result["output"]}
            except Exception as exc:
                # Return errors as tool output instead of crashing the
                # orchestrator.  asyncio.CancelledError (BaseException)
                # is NOT caught here and propagates correctly.
                return {"error": f"Agent '{agent_cls.__name__}' failed: {exc}"}
            finally:
                if agent is not None:
                    await agent.aclose()

        return _execute_handoff

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Mark the agent as closed (idempotent, concurrency-safe).

        After this call, any subsequent :meth:`process` invocation will
        raise :class:`~pyaiagent.core.exceptions.AgentClosedError`.

        Uses a double-checked locking pattern so that concurrent
        ``aclose()`` calls are safe:
        1. Fast path (no lock): return immediately if already closed.
        2. Slow path (under lock): re-check and set the flag atomically.
        """
        if self._closed:                  # fast path — no lock needed
            return
        async with self._ready_lock:
            if self._closed:              # re-check under lock
                return
            self._closed = True

    async def __aenter__(self):
        """Enter the async context manager: initialize and return ``self``.

        Raises:
            AgentClosedError: If the agent was already closed.
        """
        if self._closed:
            raise AgentClosedError(agent_name=self.__agent_name__)
        await self._ensure_ready()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the async context manager: close the agent."""
        await self.aclose()

    # ── Abstract hooks (providers MUST implement) ─────────────────────

    async def _ensure_ready(self) -> None:
        """Lazy one-time initialization — providers **must** override.

        Called automatically on the first :meth:`process` invocation
        (or when entering ``async with``).  Implementations should:

        1. Acquire ``self._ready_lock`` and double-check ``self._ready``.
        2. Build the provider-specific config (``self._config``).
        3. Set up the LLM client and any pre-computed API kwargs.
        4. Bind tool method names to ``self._tool_functions``.
        5. Set ``self._ready = True`` as the last step.

        If initialization fails, ``self._ready`` stays ``False`` so
        the next call retries automatically.
        """
        raise NotImplementedError
