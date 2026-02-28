"""OpenAI Responses API agent built on :class:`~pyaiagent.core.agent.PyAiAgent`.

This module provides :class:`OpenAIAgent`, the concrete provider implementation
that connects the generic :class:`PyAiAgent` framework to the
`OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_.

Key capabilities added on top of ``PyAiAgent``:

* **OpenAI client management** — lazy singleton via
  :class:`~pyaiagent.openai.client.AsyncOpenAIClient`.
* **Structured output with validation retries** — when a ``text_format``
  Pydantic model and ``validation_retries > 0`` are configured, the agent
  automatically re-prompts the LLM to fix malformed JSON.
* **Agentic loop** — the :meth:`OpenAIAgent.process` method runs a multi-step
  loop (tool calls → LLM → tool calls → ...) up to ``max_steps``, accumulating
  token usage and message history.
* **Multi-agent handoffs** — inherited from ``PyAiAgent``; registered
  sub-agents are called as tools by the orchestrating LLM.
* **Customisable message formatting** — override :meth:`format_history`
  and :meth:`format_event` to control what is stored in conversation
  history vs. what is recorded as events.

Lifecycle::

    agent = MyAgent()               # lightweight; no I/O yet
    result = await agent.process(   # first call triggers _ensure_ready()
        input="Hello!",
    )
    await agent.aclose()            # idempotent shutdown

Or as an async context manager::

    async with MyAgent() as agent:
        result = await agent.process(input="Hello!")
"""
from __future__ import annotations

import uuid
import asyncio
import logging
from typing import Any, Awaitable, Dict, List, Optional, Iterable

import httpx
import orjson
from pydantic import ValidationError as PydanticValidationError

try:
    from openai.lib._pydantic import to_strict_json_schema
except ImportError:
    to_strict_json_schema = None
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

from pyaiagent.core.agent import PyAiAgent
from pyaiagent.openai.config import OpenAIAgentConfig, ConfigResolver
from pyaiagent.openai.client import AsyncOpenAIClient
from pyaiagent.openai.exceptions import (
    OpenAIAgentProcessError, InvalidInputError,
    InvalidSessionError, InvalidMetadataError, ClientError,
    MaxStepsExceededError, InvalidHistoryError, InvalidInstructionParamsError,
    OpenAIAgentClosedError, ValidationRetriesExhaustedError,
)

__all__ = ["OpenAIAgent"]

logger = logging.getLogger(__name__)


class _ResponseWithParsed:
    """Transparent proxy that wraps an OpenAI ``Response`` with manual
    Pydantic validation results.

    When validation retries are enabled, we call ``responses.create()``
    (not ``responses.parse()``) so that token usage is always available
    even when the LLM output fails schema validation.  This proxy
    exposes our own ``output_parsed`` / ``validation_error`` while
    forwarding every other attribute (``usage``, ``output``,
    ``output_text``, etc.) to the underlying SDK response via
    ``__getattr__``.

    Attributes:
        output_parsed: The Pydantic model instance if validation
            succeeded, otherwise ``None``.
        validation_error: A string description of the validation failure,
            or ``None`` on success.
    """
    __slots__ = ('_response', 'output_parsed', 'validation_error')

    def __init__(self, response, output_parsed=None, validation_error=None):
        self._response = response
        self.output_parsed = output_parsed
        self.validation_error = validation_error

    def __getattr__(self, name):
        return getattr(self._response, name)


class OpenAIAgent(PyAiAgent):
    """Concrete agent class for the OpenAI Responses API.

    Subclass this to create an agent.  The class docstring becomes the system
    instruction, and any public method with a docstring and type hints is
    automatically registered as a tool.

    Quick start::

        from pyaiagent import OpenAIAgent

        class Greeter(OpenAIAgent):
            \"\"\"You are a friendly greeter.\"\"\"

            async def greet(self, name: str) -> dict:
                \"\"\"Greet a user by name.\"\"\"
                return {"greeting": f"Hello, {name}!"}

        async with Greeter() as agent:
            result = await agent.process(input="Say hi to Alice")

    Configuration (all optional)::

        class MyAgent(OpenAIAgent):
            \"\"\"System instruction here.\"\"\"

            class Config:
                model = "gpt-4o"
                temperature = 0.7
                max_steps = 10
                max_output_tokens = 4096

    Structured output with validation retries::

        from pydantic import BaseModel

        class Answer(BaseModel):
            reasoning: str
            answer: str

        class Analyst(OpenAIAgent):
            \"\"\"Analyse the data and return structured JSON.\"\"\"

            class Config:
                text_format = Answer
                validation_retries = 3

    Instruction templates::

        class PersonalisedAgent(OpenAIAgent):
            \"\"\"Hello {user_name}, you are in {city}.\"\"\"

        result = await PersonalisedAgent().process(
            input="What's the weather?",
            instruction_params={"user_name": "Alice", "city": "London"},
        )

    Multi-agent orchestration::

        class Researcher(OpenAIAgent):
            \"\"\"You research topics thoroughly.\"\"\"

        class ContentTeam(OpenAIAgent):
            \"\"\"You manage content creation. Delegate to the researcher.\"\"\"

            class Agents:
                researcher = Researcher

        result = await ContentTeam().process(input="Research AI in healthcare")
    """

    _abstract_agent = True

    __slots__ = (
        "_client",
        "_semaphore",
        "_static_openai_responses_api_kwargs",
        "_text_format_schema",
    )

    # ── Config resolution (OpenAI-specific validation) ────────────────

    @classmethod
    def _resolve_config(cls) -> dict[str, Any]:
        """Merge and validate config against OpenAI-specific fields."""
        return ConfigResolver.resolve(cls=cls)

    # ── Instance creation ─────────────────────────────────────────────

    def __new__(cls, *args, **kwargs):
        """Allocate and initialize OpenAI-specific slot defaults."""
        self = super().__new__(cls, *args, **kwargs)
        self._client = None
        self._semaphore = None
        self._static_openai_responses_api_kwargs = None
        self._text_format_schema = None
        return self

    def __init__(self):
        super().__init__()

    # ── Lazy initialization ───────────────────────────────────────────

    async def _ensure_ready(self) -> None:
        """Lazy one-time initialization (double-checked locking pattern).

        Called automatically on the first :meth:`process` invocation.
        Subsequent calls are a no-op.  When multiple coroutines race to
        call this concurrently, the lock guarantees initialization
        happens exactly once.

        Initialization steps:

        1. Instantiate :class:`~pyaiagent.openai.config.OpenAIAgentConfig`.
        2. Create a :class:`asyncio.Semaphore` for parallel tool execution.
        3. Obtain an :class:`~openai.AsyncOpenAI` client via
           :class:`~pyaiagent.openai.client.AsyncOpenAIClient`.
        4. Pre-build the static portion of the Responses API kwargs.
        5. If structured output with validation retries is configured,
           pre-compute the strict JSON schema (requires ``openai>=1.40``).
        6. Bind tool method names to bound methods on this instance.

        Raises:
            ImportError: If validation retries are configured but
                ``to_strict_json_schema`` is unavailable.
        """
        if self._ready:
            return
        async with self._ready_lock:
            if self._ready:
                return
            self._config = OpenAIAgentConfig(**self.__config_kwargs__)
            self._semaphore = asyncio.Semaphore(self._config.max_parallel_tools)
            self._client = AsyncOpenAIClient().client
            self._static_openai_responses_api_kwargs = self._build_static_openai_responses_api_kwargs()
            config = self._config
            if config.text_format is not None and config.validation_retries > 0:
                if to_strict_json_schema is None:
                    msg = (
                        "validation_retries requires 'openai.lib._pydantic.to_strict_json_schema' "
                        "which is not available in your installed openai version. "
                        "Please upgrade: pip install 'openai>=1.40.0,<2.0' "
                        "and report this at https://github.com/troymjose/pyaiagent/issues"
                    )
                    logger.critical(msg)
                    raise ImportError(msg)
                self._text_format_schema = {
                    "format": {
                        "type": "json_schema",
                        "name": config.text_format.__name__,
                        "schema": to_strict_json_schema(config.text_format),
                        "strict": True,
                    }
                }
            self._tool_functions = {name: getattr(self, name) for name in self.__tool_names__}
            for name, spec in self.__handoffs__.items():
                self._tool_functions[name] = self._make_handoff_executor(spec)
            self._ready = True

    # ── Events ────────────────────────────────────────────────────────

    def _create_events(
            self,
            data: Any,
            session: str,
            turn: str,
            step: int,
            input_tokens: int,
            output_tokens: int,
            total_tokens: int,
            metadata: Dict[str, str] | None,
            message: str | None = None,
    ) -> list[dict]:
        """Build event dicts from heterogeneous response data.

        Each returned dict contains a common base (agent name, session,
        turn, step, token counts, metadata) merged with the item-specific
        payload.  SDK model objects are serialized via ``model_dump()``
        with ``parsed_arguments`` stripped to avoid leaking internal
        Pydantic artefacts.

        Args:
            data: A dict, an SDK response object, a list of either, or
                any other type (which is silently ignored).
            session: The session identifier for this conversation.
            turn: The turn identifier for this request.
            step: The current step number within the turn.
            input_tokens: Input token count for this step.
            output_tokens: Output token count for this step.
            total_tokens: Total token count for this step.
            metadata: Arbitrary key/value metadata to attach.
            message: Optional explicit message ID; auto-generated if
                ``None``.

        Returns:
            A list of event dicts.  Empty if events are disabled
            or *data* is an unsupported type.
        """
        if not self._config.include_events:
            return []

        metadata = metadata if metadata is not None else {}
        message_id = message or str(uuid.uuid4())

        base = {
            "agent": self.__agent_name__,
            "session": session,
            "message": message_id,
            "turn": turn,
            "step": step,
            "tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "metadata": metadata,
        }

        messages: List[Dict[str, Any]] = []
        append = messages.append

        if isinstance(data, (dict, ResponseOutputMessage, ResponseFunctionToolCall)):
            items: Iterable[Any] = (data,)
        elif isinstance(data, list):
            items = data
        else:
            return messages

        for item in items:
            if isinstance(item, dict):
                merged = base.copy()
                merged.update(item)
                append(merged)
            elif isinstance(item, (ResponseOutputMessage, ResponseFunctionToolCall)):
                item_dict = item.model_dump(mode='python')
                item_dict.pop("parsed_arguments", None)
                merged = base.copy()
                merged.update(item_dict)
                append(merged)

        return messages

    # ── OpenAI API ────────────────────────────────────────────────────

    def _build_static_openai_responses_api_kwargs(self) -> dict:
        """Pre-build the kwargs that remain constant across all API calls.

        Per-call values (``instructions``, ``input``, ``metadata``,
        ``text``/``text_format``) are merged in at call time by
        :meth:`_openai_responses_api_call`.

        Returns:
            A dict ready to be shallow-copied and augmented per request.
        """
        config = self._config
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "tool_choice": config.tool_choice,
            "tools": self.__tools_schema__ + self.__handoff_schemas__,
            "parallel_tool_calls": config.parallel_tool_calls,
            "store": False,
            "stream": False,
        }
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.seed is not None:
            kwargs["seed"] = config.seed
        if config.user is not None:
            kwargs["user"] = config.user
        return kwargs

    async def _openai_responses_api_call(self,
                                         instruction: str,
                                         current_turn_history: List[Dict[str, Any]],
                                         metadata: Dict[str, str] | None = None) -> Any:
        """Execute a single call to the OpenAI Responses API.

        Selects one of three code paths based on configuration:

        1. **Plain text** (``text_format is None``) -- calls
           ``responses.create()`` and returns the raw SDK response.
        2. **Structured output with retries** (``text_format`` set,
           ``validation_retries > 0``) -- calls ``responses.create()``
           with a ``text`` JSON-schema constraint and validates the
           output locally with Pydantic.  Returns a
           :class:`_ResponseWithParsed` proxy so that token usage is
           always available even on validation failure.
        3. **Structured output without retries** (``text_format`` set,
           ``validation_retries == 0``) -- delegates to the SDK's
           ``responses.parse()`` which handles Pydantic validation
           internally.

        All paths enforce a ``llm_timeout`` via :func:`asyncio.wait_for`.

        Args:
            instruction: The system instruction for this call.
            current_turn_history: Conversation history to send as
                the ``input`` parameter.
            metadata: Optional metadata dict forwarded to the API when
                non-empty.

        Returns:
            An SDK ``Response``, a :class:`_ResponseWithParsed` proxy,
            or an SDK ``ParsedResponse`` depending on the path taken.

        Raises:
            ClientError: Wraps any HTTP, Pydantic, or unexpected
                exception with agent context and error chaining.
            asyncio.CancelledError: Re-raised without wrapping.
        """
        try:
            config = self._config
            kwargs = self._static_openai_responses_api_kwargs.copy()
            kwargs["instructions"] = instruction
            kwargs["input"] = current_turn_history
            if metadata:
                kwargs["metadata"] = metadata

            if config.text_format is None:
                return await asyncio.wait_for(self._client.responses.create(**kwargs), timeout=config.llm_timeout)

            if config.validation_retries > 0:
                kwargs["text"] = self._text_format_schema
                response = await asyncio.wait_for(
                    self._client.responses.create(**kwargs), timeout=config.llm_timeout
                )
                output_parsed = None
                validation_error = None
                if response.output_text:
                    try:
                        output_parsed = config.text_format.model_validate_json(response.output_text)
                    except PydanticValidationError as ve:
                        validation_error = str(ve)
                return _ResponseWithParsed(response, output_parsed, validation_error)

            kwargs["text_format"] = config.text_format
            parse_response = await asyncio.wait_for(self._client.responses.parse(**kwargs),
                                                    timeout=config.llm_timeout)
            return parse_response

        except asyncio.CancelledError:
            raise

        except httpx.HTTPError as exc:
            exc_msg = f"HTTP error during OpenAI call. Error: {exc}"
            if logger.isEnabledFor(logging.ERROR):
                logger.exception(exc_msg)
            raise ClientError(agent_name=self.__agent_name__, message=exc_msg) from exc

        except PydanticValidationError as exc:
            exc_msg = f"Structured output validation failed. Error: {exc}"
            if logger.isEnabledFor(logging.ERROR):
                logger.exception(exc_msg)
            raise ClientError(agent_name=self.__agent_name__, message=exc_msg) from exc

        except Exception as exc:
            exc_msg = f"Unexpected error during OpenAI call. Error: {exc}"
            if logger.isEnabledFor(logging.ERROR):
                logger.exception(exc_msg)
            raise ClientError(agent_name=self.__agent_name__, message=exc_msg) from exc

    # ── Tool execution ────────────────────────────────────────────────

    async def _execute_tool_call(self, tool_call: ResponseFunctionToolCall) -> Dict[str, str]:
        """Execute a single tool call and return a ``function_call_output`` dict.

        The semaphore limits concurrent tool executions to
        ``config.max_parallel_tools``.  Async tools are awaited
        directly; sync tools are offloaded via
        :func:`asyncio.to_thread` to avoid blocking the event loop.

        Arguments arrive as JSON strings from the LLM; ``bytes`` /
        ``bytearray`` are decoded first.  If an argument type is
        entirely unexpected, an empty kwargs dict is used as a
        safe fallback.

        Non-dict tool results are automatically wrapped as
        ``{"result": <value>}`` so the output is always JSON-serializable.

        Args:
            tool_call: The SDK ``ResponseFunctionToolCall`` to execute.

        Returns:
            A ``function_call_output`` dict with ``type``, ``call_id``,
            and ``output`` keys.

        Raises:
            asyncio.CancelledError: Re-raised without wrapping.
        """
        async with self._semaphore:
            name, arguments, call_id = tool_call.name, tool_call.arguments, tool_call.call_id
            tool_function = self._tool_functions.get(name)
            if tool_function is None:
                return {"type": "function_call_output",
                        "call_id": call_id,
                        "output": self._to_str({"error": f"Tool not found. Tool: '{name}'"})}
            if isinstance(arguments, (bytes, bytearray)):
                arguments = arguments.decode("utf-8", "ignore")
            if isinstance(arguments, str):
                try:
                    args = orjson.loads(arguments) if arguments else {}
                except Exception:
                    logger.exception("Bad tool args JSON: %s", name)
                    return {"type": "function_call_output",
                            "call_id": call_id,
                            "output": self._to_str({"error": f"Bad tool args JSON. Tool: '{name}'"})}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}
            try:
                if asyncio.iscoroutinefunction(tool_function):
                    tool_result = await asyncio.wait_for(tool_function(**args), timeout=self._config.tool_timeout)
                else:
                    tool_result = await asyncio.wait_for(
                        asyncio.to_thread(tool_function, **args),
                        timeout=self._config.tool_timeout
                    )
                if not isinstance(tool_result, dict):
                    tool_result = {"result": tool_result}
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                return {"type": "function_call_output",
                        "call_id": call_id,
                        "output": self._to_str({
                            "error": f"Tool execution failed after a timeout of {self._config.tool_timeout}. Tool: '{name}'"})}
            except Exception:
                logger.exception("Tool execution failed: %s", name)
                return {"type": "function_call_output",
                        "call_id": call_id,
                        "output": self._to_str({"error": f"Tool execution failed. Tool: '{name}'"})}

            return {"type": "function_call_output", "call_id": call_id, "output": self._to_str(tool_result)}

    async def _execute_tool_calls(self, tool_calls: List[ResponseFunctionToolCall]) -> List[Dict[str, str]]:
        """Execute one or more tool calls, returning results in order.

        A single tool call is awaited directly to avoid
        :func:`asyncio.gather` overhead.  Multiple calls run
        concurrently, bounded by the semaphore in
        :meth:`_execute_tool_call`.

        Args:
            tool_calls: Non-empty list of tool calls from the LLM
                response.

        Returns:
            A list of ``function_call_output`` dicts, one per call.
        """
        if len(tool_calls) == 1:
            return [await self._execute_tool_call(tool_calls[0])]
        tasks: List[Awaitable[Dict[str, str]]] = [self._execute_tool_call(call) for call in tool_calls]
        return await asyncio.gather(*tasks)

    # ── Message formatting ────────────────────────────────────────────

    def format_history(self, response: Any) -> str:
        """
        Format assistant response content for LLM conversation memory.

        Override this method to customize what gets stored in history.
        This is useful for reducing token usage when using structured outputs
        with large fields that don't need to be in conversation history.

        Args:
            response: OpenAI API response object with output_text and output_parsed attributes.

        Returns:
            String content to store as the assistant message.

        Example:
            def format_history(self, response) -> str:
                # Only store agent_response, not the large bot_config
                if response.output_parsed:
                    return response.output_parsed.agent_response
                return response.output_text or ""
        """
        return response.output_text or ""

    def format_event(self, response: Any) -> str:
        """
        Format assistant response content for events (UI/frontend display).

        Override this method to customize what gets stored in events.
        Events are used for session logs, frontend display, and analytics.

        Args:
            response: OpenAI API response object with output_text and output_parsed attributes.

        Returns:
            String content to store in the event record.

        Example:
            def format_event(self, response) -> str:
                # Show a user-friendly summary in UI
                if response.output_parsed:
                    return f"Agent: {response.output_parsed.agent_response}"
                return response.output_text or ""
        """
        return response.output_text or ""

    # ── Agentic loop ──────────────────────────────────────────────────

    async def process(self,
                      *,
                      input: str,
                      session: Optional[str] = None,
                      history: Optional[List[Dict[str, Any]]] = None,
                      instruction_params: Dict[str, str] | None = None,
                      metadata: Optional[dict] = None) -> dict:
        """Run the agent's multi-step processing loop for a single user input.

        This is the primary entry point for interacting with an agent.
        It validates inputs, formats the instruction, and enters an
        iterative loop that alternates between LLM calls and tool
        execution until the LLM produces a final text response or
        ``max_steps`` is exhausted.

        On the first call, :meth:`_ensure_ready` is invoked to perform
        lazy initialization.

        Args:
            input: The user's message (must be a non-empty or empty
                string).
            session: An optional session identifier for grouping turns.
                Auto-generated as a UUID if ``None``.
            history: Optional conversation history to prepend.
                Shallow-copied to avoid mutating the caller's list.
                Ignored when ``include_history`` is ``False``.
            instruction_params: Key-value pairs to substitute into the
                instruction template's ``{placeholder}`` tokens.
            metadata: Arbitrary dict attached to every event.
                Not sent to the OpenAI API.

        Returns:
            A dict with the following structure::

                {
                    "input": str,
                    "session": str,
                    "metadata": dict,
                    "output": str,
                    "output_parsed": BaseModel | None,
                    "steps": int,
                    "turn": str,
                    "history": list[dict],
                    "events": list[dict],
                    "tokens": {
                        "input_tokens": int,
                        "output_tokens": int,
                        "total_tokens": int,
                    },
                }

        Raises:
            OpenAIAgentClosedError: If :meth:`aclose` was already called.
            InvalidInputError: If *input* is not a string.
            InvalidSessionError: If *session* is not a string or is
                blank.
            InvalidMetadataError: If *metadata* is not a dict.
            InvalidHistoryError: If *history* is not a list.
            InvalidInstructionParamsError: If *instruction_params* is
                not a dict.
            InstructionKeyError: In strict mode, if a template
                placeholder is missing from *instruction_params*.
            ClientError: On HTTP or unexpected errors from the OpenAI
                API.
            MaxStepsExceededError: If the loop reaches ``max_steps``
                without a final text response.
            ValidationRetriesExhaustedError: If structured output
                validation fails after all allowed retries.
        """
        if self._closed:
            raise OpenAIAgentClosedError(agent_name=self.__agent_name__)

        # ── Input validation ──────────────────────────────────────────
        if not isinstance(input, str):
            raise InvalidInputError(agent_name=self.__agent_name__, received=type(input).__name__)
        if session is not None:
            if not isinstance(session, str):
                raise InvalidSessionError(agent_name=self.__agent_name__, received=type(session).__name__)
            if not session.strip():
                raise InvalidSessionError(agent_name=self.__agent_name__, received="empty string")
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidMetadataError(agent_name=self.__agent_name__, received=type(metadata).__name__)
        if history is not None and not isinstance(history, list):
            raise InvalidHistoryError(agent_name=self.__agent_name__, received=type(history).__name__)
        if instruction_params is not None and not isinstance(instruction_params, dict):
            raise InvalidInstructionParamsError(agent_name=self.__agent_name__,
                                                received=type(instruction_params).__name__)

        await self._ensure_ready()

        config = self._config
        create_events = self._create_events

        session = session if session is not None else str(uuid.uuid4())
        metadata = metadata if metadata is not None else {}

        instruction = self._format_instruction(instruction_params=instruction_params)
        turn, step = str(uuid.uuid4()), 1
        input_tokens, output_tokens, total_tokens = 0, 0, 0

        current_turn_history = list(
            history) if history is not None and config.include_history else []
        current_turn_history.append({"role": "user", "content": input})
        current_turn_events = create_events(data={"role": "user", "content": input},
                                              session=session,
                                              turn=turn,
                                              step=step,
                                              input_tokens=0,
                                              output_tokens=0,
                                              total_tokens=0,
                                              metadata=metadata,
                                              message=turn)

        assistant_response: Optional[str] = None
        assistant_response_parsed: Optional[Any] = None
        validation_attempts = 0
        validation_retries_enabled = config.text_format is not None and config.validation_retries > 0
        last_validation_error: Optional[str] = None
        history_checkpoint: Optional[int] = None

        try:
            # ── Agentic loop: LLM call → tool execution → repeat ─────
            for _ in range(config.max_steps):
                response = await self._openai_responses_api_call(instruction=instruction,
                                                                 current_turn_history=current_turn_history)

                usage = response.usage
                response_output = response.output

                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                total_tokens += usage.total_tokens

                tool_calls = [item for item in response_output if isinstance(item, ResponseFunctionToolCall)]

                if config.include_history:
                    if not tool_calls and response.output_text:
                        current_turn_history.append(
                            {"role": "assistant", "content": self.format_history(response)})
                    else:
                        for item in response_output:
                            d = item.model_dump()
                            d.pop("parsed_arguments", None)
                            current_turn_history.append(d)

                current_turn_events.extend(create_events(data={"role": "assistant",
                                                                     "content": self.format_event(
                                                                         response)} if not tool_calls and response.output_text else response_output,
                                                                   session=session,
                                                                   turn=turn,
                                                                   step=step,
                                                                   input_tokens=usage.input_tokens,
                                                                   output_tokens=usage.output_tokens,
                                                                   total_tokens=usage.total_tokens,
                                                                   metadata=metadata))

                # ── No tool calls: check for final response ──────────
                if not tool_calls:
                    if validation_retries_enabled:
                        output_parsed = getattr(response, "output_parsed", None)
                        if output_parsed is None:
                            if history_checkpoint is None:
                                history_checkpoint = len(current_turn_history) - 1
                            if validation_attempts < config.validation_retries:
                                validation_attempts += 1
                                error_detail = getattr(response, "validation_error", None) \
                                               or "output could not be parsed into the expected format"
                                last_validation_error = error_detail
                                current_turn_history.append({
                                    "role": "user",
                                    "content": f"fix the errors:\n\n{error_detail}"
                                })
                                step += 1
                                continue
                            raise ValidationRetriesExhaustedError(
                                agent_name=self.__agent_name__,
                                validation_retries=config.validation_retries,
                                errors=last_validation_error or "output_parsed is None"
                            )

                    if validation_attempts > 0 and history_checkpoint is not None:
                        del current_turn_history[history_checkpoint:]
                        if config.include_history:
                            current_turn_history.append(
                                {"role": "assistant", "content": self.format_history(response)})

                    assistant_response = getattr(response, "output_text", None)
                    assistant_response_parsed = getattr(response, "output_parsed", None)
                    break

                # ── Tool calls present: execute and continue loop ────
                step += 1
                tool_execution_results: list = await self._execute_tool_calls(tool_calls=tool_calls)
                if config.include_history:
                    current_turn_history.extend(tool_execution_results)
                current_turn_events.extend(create_events(data=tool_execution_results,
                                                                   session=session,
                                                                   turn=turn,
                                                                   step=step,
                                                                   input_tokens=0,
                                                                   output_tokens=0,
                                                                   total_tokens=0,
                                                                   metadata=metadata))

            if assistant_response is None:
                raise MaxStepsExceededError(agent_name=self.__agent_name__, max_steps=config.max_steps)

        except OpenAIAgentProcessError as exc:
            exc.tokens = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}
            raise

        return {"input": input,
                "session": session,
                "metadata": metadata,
                "output": assistant_response or "",
                "output_parsed": assistant_response_parsed or None,
                "steps": step,
                "turn": turn,
                "history": current_turn_history,
                "events": current_turn_events,
                "tokens": {"input_tokens": input_tokens,
                           "output_tokens": output_tokens,
                           "total_tokens": total_tokens}
                }
