from __future__ import annotations

import uuid
import asyncio
import logging
from typing import Any, Awaitable, Dict, List, Optional, Iterable

import httpx
import orjson
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

from pyaiagent.openai.config import OpenAIAgentConfig
from pyaiagent.openai.client import AsyncOpenAIClient
from pyaiagent.openai.manager.tool import OpenAIAgentToolManager
from pyaiagent.openai.manager.config import OpenAIAgentConfigManager
from pyaiagent.openai.manager.instruction import OpenAIAgentInstructionManager
from pyaiagent.openai.exceptions.process import InvalidInputError, InvalidSessionError, InvalidMetadataError, \
    ClientError, \
    MaxStepsExceededError, InvalidLlmMessagesError, InvalidInstructionParamsError, InstructionKeyError, \
    OpenAIAgentClosedError

__all__ = ["OpenAIAgent"]

logger = logging.getLogger(__name__)


class OpenAIAgent:
    __slots__ = ("_ready_lock",
                 "_ready",
                 "_closed",
                 "_config",
                 "_client",
                 "_semaphore",
                 "_static_openai_responses_api_kwargs",
                 "_tool_functions")

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
        return {
            # Model ID used to generate the response, like gpt-4o or o3.
            # OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points.
            # Refer to the model guide to browse and compare available models.
            "model": config.model,
            # What sampling temperature to use, between 0 and 2.
            # Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            # We generally recommend altering this or top_p but not both.
            "temperature": config.temperature,
            # An upper bound for the number of tokens that can be generated for a response,
            # including visible output tokens and reasoning tokens.
            "max_output_tokens": config.max_output_tokens,
            # How the model should select which tool (or tools) to use when generating a response.
            # none means the model will not call any tool and instead generates a message.
            # auto means the model can pick between generating a message or calling one or more tools.
            # required means the model must call one or more tools.
            "tool_choice": config.tool_choice,
            # An array of tools the model may call while generating a response.
            # You can specify which tool to use by setting the tool_choice parameter.
            "tools": self.__tools_schema__,
            # Whether to allow the model to run tool calls in parallel.
            "parallel_tool_calls": config.parallel_tool_calls,
            # Whether to store the generated model response for later retrieval via API.
            "store": False,
            # If set to true, the model response data will be streamed to the client as it is generated using server-sent events.
            # See the Streaming section below for more information.
            "stream": False,
        }

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

            kwargs["text_format"] = config.text_format
            parse_response = await asyncio.wait_for(self._client.responses.parse(**kwargs),
                                                    timeout=config.llm_timeout)
            for parse_response_model in parse_response.output:
                if hasattr(parse_response_model, "parsed_arguments"):
                    delattr(parse_response_model, "parsed_arguments")
            return parse_response

        except asyncio.CancelledError:
            # Preserve cancellation semantics
            raise

        except httpx.HTTPError as exc:
            exc_msg = f"HTTP error during OpenAI call. Error: {exc}"
            if logger.isEnabledFor(logging.ERROR):
                logger.exception(exc_msg)
            raise ClientError(message=exc_msg) from exc

        except Exception as exc:
            exc_msg = f"Unexpected error during OpenAI call. Error: {exc}"
            if logger.isEnabledFor(logging.ERROR):
                logger.exception(exc_msg)
            raise ClientError(message=exc_msg) from exc

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
                tool_result = await asyncio.wait_for(tool_function(**args), timeout=self._config.tool_timeout)
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
        try:
            instruction: str = self.__instruction__.format(**(instruction_params or {}))
            return instruction
        except KeyError as exc:
            raise InstructionKeyError(str(exc))

    async def process(self,
                      *,
                      input: str,
                      session: Optional[str] = None,
                      llm_messages: Optional[List[Dict[str, Any]]] = None,
                      instruction_params: Dict[str, str] | None = None,
                      metadata: Optional[dict] = None) -> dict:

        if self._closed:
            raise OpenAIAgentClosedError()

        # Validate input types before any processing
        if not isinstance(input, str):
            raise InvalidInputError()
        if session is not None and (not isinstance(session, str) or not session.strip()):
            raise InvalidSessionError()
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidMetadataError()
        if llm_messages is not None and not isinstance(llm_messages, list):
            raise InvalidLlmMessagesError()
        if instruction_params is not None and not isinstance(instruction_params, dict):
            raise InvalidInstructionParamsError()

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
                    # Final text response - simple format
                    current_turn_llm_messages.append({"role": "assistant", "content": response.output_text})
                else:
                    # Has tool calls - preserve full structure for API compatibility
                    current_turn_llm_messages.extend(item.model_dump() for item in response_output)

            # Simple format for final text response, full structure for tool calls
            current_turn_ui_messages.extend(create_ui_messages(data={"role": "assistant",
                                                                     "content": response.output_text or ""} if not tool_calls and response.output_text else response_output,
                                                               session=session,
                                                               turn=turn,
                                                               step=step,
                                                               input_tokens=usage.input_tokens,
                                                               output_tokens=usage.output_tokens,
                                                               total_tokens=usage.total_tokens,
                                                               metadata=metadata))

            if not tool_calls:
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
            raise MaxStepsExceededError()

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
            raise OpenAIAgentClosedError()
        await self._ensure_ready()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()
