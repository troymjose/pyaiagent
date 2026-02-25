from __future__ import annotations

import re
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

from pyaiagent.openai.config import OpenAIAgentConfig
from pyaiagent.openai.client import AsyncOpenAIClient
from pyaiagent.openai.manager.tool import OpenAIAgentToolManager
from pyaiagent.openai.manager.config import OpenAIAgentConfigManager
from pyaiagent.openai.manager.instruction import OpenAIAgentInstructionManager
from pyaiagent.openai.exceptions.process import OpenAIAgentProcessError, InvalidInputError, InvalidSessionError, \
    InvalidMetadataError, ClientError, \
    MaxStepsExceededError, InvalidLlmMessagesError, InvalidInstructionParamsError, InstructionKeyError, \
    OpenAIAgentClosedError, ValidationRetriesExhaustedError

__all__ = ["OpenAIAgent"]

logger = logging.getLogger(__name__)


class _ResponseWithParsed:
    """Proxy that enriches a responses.create() Response with manual Pydantic validation results.
    Forwards all attribute access to the underlying response (usage, output, output_text, etc.)
    while exposing output_parsed and validation_error from our own validation pass."""
    __slots__ = ('_response', 'output_parsed', 'validation_error')

    def __init__(self, response, output_parsed=None, validation_error=None):
        self._response = response
        self.output_parsed = output_parsed
        self.validation_error = validation_error

    def __getattr__(self, name):
        return getattr(self._response, name)


class OpenAIAgent:
    __slots__ = ("_ready_lock",
                 "_ready",
                 "_closed",
                 "_config",
                 "_client",
                 "_semaphore",
                 "_static_openai_responses_api_kwargs",
                 "_tool_functions",
                 "_text_format_schema")

    def __init_subclass__(cls, **kwargs):
        """ Initialize subclass by preparing required components. """
        # Call super
        super().__init_subclass__(**kwargs)
        # Skip base class
        if cls is OpenAIAgent:
            return
        # Cache class name (avoid repeated __class__.__name__ lookups)
        cls.__agent_name__ = cls.__name__
        # Prepare instruction template
        cls.__instruction__ = OpenAIAgentInstructionManager.create(cls=cls)
        # Prepare config kwargs
        cls.__config_kwargs__ = OpenAIAgentConfigManager.create(cls=cls)
        # Prepare tools
        _tools = OpenAIAgentToolManager.create(cls=cls)
        cls.__tool_names__ = tuple(_tools.keys())
        cls.__tools_schema__ = tuple(_tools.values())
        del _tools

    def __new__(cls, *args, **kwargs):
        # Create the instance
        self = super().__new__(cls)

        # Initialize base state *always*, regardless of subclass __init__
        self._ready_lock = asyncio.Lock()
        self._ready = False
        self._closed = False
        self._config = None
        self._client = None
        self._semaphore = None
        self._static_openai_responses_api_kwargs = None
        self._tool_functions = None
        self._text_format_schema = None

        return self

    @staticmethod
    def _to_str(value: Any) -> str:
        """ Convert value to string representation. """
        if isinstance(value, str):
            return value
        try:
            return orjson.dumps(value).decode('utf-8')
        except Exception:
            return str(value)

    def __init__(self):
        if type(self) is OpenAIAgent:
            raise TypeError("OpenAIAgent is an abstract base; subclass it before use.")

    async def _ensure_ready(self) -> None:
        """ Ensure the agent is ready for processing. """
        # Fast path: already ready (no lock needed)
        if self._ready:
            return
        # Slow path: acquire lock and initialize
        async with self._ready_lock:
            # Double-check after acquiring lock (another coroutine may have initialized)
            if self._ready:
                return
            # Create config
            self._config = OpenAIAgentConfig(**self.__config_kwargs__)
            # Create semaphore
            self._semaphore = asyncio.Semaphore(self._config.max_parallel_tools)
            # Create client
            self._client = AsyncOpenAIClient().client
            # Build static openai responses api kwargs
            self._static_openai_responses_api_kwargs = self._build_static_openai_responses_api_kwargs()
            # Pre-compute strict JSON schema for validation retries (avoids per-call overhead)
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
            # Bind tool functions to self (convert class functions -> bound methods)
            self._tool_functions = {name: getattr(self, name) for name in self.__tool_names__}
            # Mark ready
            self._ready = True

    def _create_ui_messages(
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
        """Create internal message dict(s) from data for UI/DB storage."""
        # Fast guard: if UI messages are disabled, do nothing.
        if not self._config.ui_messages_enabled:
            return []

        # Normalize once
        metadata = metadata if metadata is not None else {}
        message_id = message or str(uuid.uuid4())
        agent_name = self.__agent_name__

        base = {
            "agent": agent_name,
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

        # Local alias for speed
        messages: List[Dict[str, Any]] = []
        append = messages.append  # micro-optimization

        # Normalize data into an iterable of items
        if isinstance(data, (dict, ResponseOutputMessage, ResponseFunctionToolCall)):
            items: Iterable[Any] = (data,)
        elif isinstance(data, list):
            items = data
        else:
            # Unsupported type -> nothing to save
            return messages

        for item in items:
            if isinstance(item, dict):
                # Avoid {**base, **item} to reduce copying
                merged = base.copy()
                merged.update(item)
                append(merged)
            elif isinstance(item, (ResponseOutputMessage, ResponseFunctionToolCall)):
                item_dict = item.model_dump(mode='python')  # Skips JSON conversion step
                merged = base.copy()
                merged.update(item_dict)
                append(merged)

        return messages

    def _build_static_openai_responses_api_kwargs(self) -> dict:
        """Pre-build immutable parts of API kwargs."""
        config = self._config
        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "tool_choice": config.tool_choice,
            "tools": self.__tools_schema__,
            "parallel_tool_calls": config.parallel_tool_calls,
            "store": False,
            "stream": False,
        }
        # Optional parameters - only include if set
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.seed is not None:
            kwargs["seed"] = config.seed
        return kwargs

    async def _openai_responses_api_call(self,
                                         instruction: str,
                                         current_turn_llm_messages: List[Dict[str, Any]],
                                         metadata: Dict[str, str] | None = None) -> Any:
        """Wrapper around Responses API call."""
        try:
            config = self._config
            kwargs = self._static_openai_responses_api_kwargs.copy()
            kwargs["instructions"] = instruction
            kwargs["input"] = current_turn_llm_messages
            # Add metadata if non-empty (already validated as dict in process())
            if metadata:
                kwargs["metadata"] = metadata

            if config.text_format is None:
                return await asyncio.wait_for(self._client.responses.create(**kwargs), timeout=config.llm_timeout)

            # Validation retries path: use create() so the response (with token
            # usage) is always available, even when Pydantic validation fails.
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

            # No retries: use parse() for SDK-managed validation
            kwargs["text_format"] = config.text_format
            parse_response = await asyncio.wait_for(self._client.responses.parse(**kwargs),
                                                    timeout=config.llm_timeout)
            # responses.parse() adds `parsed_arguments` to tool call items.
            # Strip it so model_dump() won't include it when these items are
            # round-tripped back to the API as conversation history.
            for parse_response_model in parse_response.output:
                if hasattr(parse_response_model, "parsed_arguments"):
                    delattr(parse_response_model, "parsed_arguments")
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

    async def _execute_tool_call(self, tool_call: ResponseFunctionToolCall) -> Dict[str, str]:
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
                except Exception as exc:
                    return {"type": "function_call_output",
                            "call_id": call_id,
                            "output": self._to_str(
                                {"error": f"Bad tool args JSON. Tool: '{name}'", "arguments": str(arguments)})}
            elif isinstance(arguments, dict):
                args = arguments
            else:
                args = {}
            try:
                # Check if tool is async or sync
                if asyncio.iscoroutinefunction(tool_function):
                    # Async tool: call directly
                    tool_result = await asyncio.wait_for(tool_function(**args), timeout=self._config.tool_timeout)
                else:
                    # Sync tool: run in thread pool to avoid blocking event loop
                    tool_result = await asyncio.wait_for(
                        asyncio.to_thread(tool_function, **args),
                        timeout=self._config.tool_timeout
                    )
                if not isinstance(tool_result, dict):
                    tool_result = {"result": tool_result}
            except asyncio.CancelledError:
                # Preserve cancellation semantics
                raise
            except asyncio.TimeoutError:
                return {"type": "function_call_output",
                        "call_id": call_id,
                        "output": self._to_str({
                            "error": f"Tool execution failed after a timeout of {self._config.tool_timeout}. Tool: '{name}'"})}
            except Exception as exc:
                return {"type": "function_call_output",
                        "call_id": call_id,
                        "output": self._to_str({"error": f"Tool execution failed. Tool: '{name}'"})}

            return {"type": "function_call_output", "call_id": call_id, "output": self._to_str(tool_result)}

    async def _execute_tool_calls(self, tool_calls: List[ResponseFunctionToolCall]) -> List[Dict[str, str]]:
        """ Execute tool calls and return results. """
        # Fast path for single tool call (common case)
        if len(tool_calls) == 1:
            return [await self._execute_tool_call(tool_calls[0])]
        tasks: List[Awaitable[Dict[str, str]]] = [self._execute_tool_call(call) for call in tool_calls]
        return await asyncio.gather(*tasks)

    def _format_instruction(self, instruction_params: Dict[str, str] | None = None) -> str:
        """ Format instruction template with provided parameters. """
        instruction = self.__instruction__
        strict_mode = self.__config_kwargs__.get("strict_instruction_params", False)

        # Fast path: no params and not strict, return as-is
        if not instruction_params and not strict_mode:
            return instruction

        # Replace only valid {identifier} placeholders, leaving JSON braces untouched
        # Pattern matches: {word} where word is a valid Python identifier
        # Supports escape syntax: {{name}} → {name} (literal braces)
        def replace_match(match: re.Match) -> str:
            prefix, key, suffix = match.group(1), match.group(2), match.group(3)

            # Escaped braces: {{name}} → {name} literally
            if prefix == '{' and suffix == '}':
                return f'{{{key}}}'

            # Normal placeholder: {name} → value
            if instruction_params and key in instruction_params:
                return f'{prefix}{instruction_params[key]}{suffix}'

            # Key looks like a placeholder but wasn't provided
            if strict_mode:
                raise KeyError(key)

            # Non-strict mode: leave unmatched placeholders as-is
            return match.group(0)

        try:
            # Match optional surrounding braces + identifier + optional surrounding braces
            # This handles both {name} and {{name}} (escaped) syntax
            instruction = re.sub(r'(\{?)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(\}?)', replace_match, instruction)
            return instruction
        except KeyError as exc:
            raise InstructionKeyError(agent_name=self.__agent_name__, key=str(exc))

    def format_llm_message(self, response: Any) -> str:
        """
        Format assistant response content for LLM conversation memory.

        Override this method to customize what gets stored in llm_messages.
        This is useful for reducing token usage when using structured outputs
        with large fields that don't need to be in conversation history.

        Args:
            response: OpenAI API response object with output_text and output_parsed attributes.

        Returns:
            String content to store as the assistant message.

        Example:
            def format_llm_message(self, response) -> str:
                # Only store agent_response, not the large bot_config
                if response.output_parsed:
                    return response.output_parsed.agent_response
                return response.output_text or ""
        """
        return response.output_text or ""

    def format_ui_message(self, response: Any) -> str:
        """
        Format assistant response content for UI/frontend display.

        Override this method to customize what gets stored in ui_messages.
        UI messages are used for session logs, frontend display, and analytics.

        Args:
            response: OpenAI API response object with output_text and output_parsed attributes.

        Returns:
            String content to store for UI display.

        Example:
            def format_ui_message(self, response) -> str:
                # Show a user-friendly summary in UI
                if response.output_parsed:
                    return f"Agent: {response.output_parsed.agent_response}"
                return response.output_text or ""
        """
        return response.output_text or ""

    async def process(self,
                      *,
                      input: str,
                      session: Optional[str] = None,
                      llm_messages: Optional[List[Dict[str, Any]]] = None,
                      instruction_params: Dict[str, str] | None = None,
                      metadata: Optional[dict] = None) -> dict:

        if self._closed:
            raise OpenAIAgentClosedError(agent_name=self.__agent_name__)

        # Validate input types before any processing
        if not isinstance(input, str):
            raise InvalidInputError(agent_name=self.__agent_name__, received=type(input).__name__)
        if session is not None:
            if not isinstance(session, str):
                raise InvalidSessionError(agent_name=self.__agent_name__, received=type(session).__name__)
            if not session.strip():
                raise InvalidSessionError(agent_name=self.__agent_name__, received="empty string")
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidMetadataError(agent_name=self.__agent_name__, received=type(metadata).__name__)
        if llm_messages is not None and not isinstance(llm_messages, list):
            raise InvalidLlmMessagesError(agent_name=self.__agent_name__, received=type(llm_messages).__name__)
        if instruction_params is not None and not isinstance(instruction_params, dict):
            raise InvalidInstructionParamsError(agent_name=self.__agent_name__,
                                                received=type(instruction_params).__name__)

        await self._ensure_ready()

        config = self._config
        create_ui_messages = self._create_ui_messages

        # Normalize optional parameters
        session = session if session is not None else str(uuid.uuid4())
        metadata = metadata if metadata is not None else {}

        instruction = self._format_instruction(instruction_params=instruction_params)
        turn, step = str(uuid.uuid4()), 1
        input_tokens, output_tokens, total_tokens = 0, 0, 0

        current_turn_llm_messages = [] if llm_messages is None or not config.llm_messages_enabled else llm_messages
        current_turn_llm_messages.append({"role": "user", "content": input})
        current_turn_ui_messages = create_ui_messages(data={"role": "user", "content": input},
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
        llm_msg_checkpoint: Optional[int] = None

        try:
            for _ in range(config.max_steps):
                response = await self._openai_responses_api_call(instruction=instruction,
                                                                 current_turn_llm_messages=current_turn_llm_messages)

                # Cache frequently accessed attributes
                usage = response.usage
                response_output = response.output

                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                total_tokens += usage.total_tokens

                tool_calls = [item for item in response_output if isinstance(item, ResponseFunctionToolCall)]

                if config.llm_messages_enabled:
                    if not tool_calls and response.output_text:
                        current_turn_llm_messages.append({"role": "assistant", "content": self.format_llm_message(response)})
                    else:
                        current_turn_llm_messages.extend(item.model_dump() for item in response_output)

                current_turn_ui_messages.extend(create_ui_messages(data={"role": "assistant",
                                                                         "content": self.format_ui_message(response)} if not tool_calls and response.output_text else response_output,
                                                                   session=session,
                                                                   turn=turn,
                                                                   step=step,
                                                                   input_tokens=usage.input_tokens,
                                                                   output_tokens=usage.output_tokens,
                                                                   total_tokens=usage.total_tokens,
                                                                   metadata=metadata))

                if not tool_calls:
                    if validation_retries_enabled:
                        output_parsed = getattr(response, "output_parsed", None)
                        if output_parsed is None:
                            if llm_msg_checkpoint is None:
                                llm_msg_checkpoint = len(current_turn_llm_messages) - 1
                            if validation_attempts < config.validation_retries:
                                validation_attempts += 1
                                error_detail = getattr(response, "validation_error", None) \
                                    or "output could not be parsed into the expected format"
                                last_validation_error = error_detail
                                current_turn_llm_messages.append({
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

                    # Clean up retry artifacts from llm messages (keep only the final valid response)
                    if validation_attempts > 0 and llm_msg_checkpoint is not None:
                        del current_turn_llm_messages[llm_msg_checkpoint:]
                        if config.llm_messages_enabled:
                            current_turn_llm_messages.append({"role": "assistant", "content": self.format_llm_message(response)})

                    assistant_response = getattr(response, "output_text", None)
                    assistant_response_parsed = getattr(response, "output_parsed", None)
                    break
                step += 1
                tool_execution_results: list = await self._execute_tool_calls(tool_calls=tool_calls)
                if config.llm_messages_enabled:
                    current_turn_llm_messages.extend(tool_execution_results)
                current_turn_ui_messages.extend(create_ui_messages(data=tool_execution_results,
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
                "messages": {"llm": current_turn_llm_messages, "ui": current_turn_ui_messages},
                "tokens": {"input_tokens": input_tokens,
                           "output_tokens": output_tokens,
                           "total_tokens": total_tokens}
                }

    async def aclose(self) -> None:
        """ Asynchronously close the agent and release resources. """
        # Fast path: already closed (no lock needed)
        if self._closed:
            return
        # Slow path: acquire lock and close
        async with self._ready_lock:
            # Double-check after acquiring lock
            if self._closed:
                return
            self._closed = True

    async def __aenter__(self):
        if self._closed:
            raise OpenAIAgentClosedError(agent_name=self.__agent_name__)
        await self._ensure_ready()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()
