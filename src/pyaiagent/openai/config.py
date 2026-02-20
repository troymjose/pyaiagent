from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Type, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["OpenAIAgentConfig", ]

OpenAiToolChoice = Literal["auto", "none", "required"]


@dataclass(frozen=True, slots=True)
class OpenAIAgentConfig:
    # OpenAI LLM configuration
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float | None = None
    seed: int | None = None
    max_output_tokens: int = 4096
    tool_choice: OpenAiToolChoice | Dict[str, Any] = "auto"
    parallel_tool_calls: bool = True
    text_format: Type[BaseModel] | None = None
    # Runtime configuration
    max_steps: int = 10
    max_parallel_tools: int = 10
    tool_timeout: float = 30.0
    llm_timeout: float = 120.0
    ui_messages_enabled: bool = True
    llm_messages_enabled: bool = True
    strict_instruction_params: bool = False
