"""
PyAiAgent is a modern, fast (high-performance), async framework for building AI agents with pythonic code.
"""

from pyaiagent.core.agent import PyAiAgent
from pyaiagent.openai.agent import OpenAIAgent
from pyaiagent.core.handoff import handoff
from pyaiagent.core.orchestration import team, pipeline, parallel
from pyaiagent.openai.client import shutdown, set_default_openai_client, get_default_openai_client, clear_default_openai_client

# Generic exception names (canonical)
from pyaiagent.core.exceptions import (AgentDefinitionError,
                                       AgentProcessError,
                                       AgentClosedError,
                                       InvalidInputError,
                                       InvalidSessionError,
                                       InvalidMetadataError,
                                       InvalidHistoryError,
                                       InvalidInstructionParamsError,
                                       InstructionKeyError,
                                       MaxStepsExceededError,
                                       ValidationRetriesExhaustedError)

# Backward-compatible OpenAI aliases
from pyaiagent.openai.exceptions import (OpenAIAgentDefinitionError,
                                         OpenAIAgentProcessError,
                                         OpenAIAgentClosedError,
                                         ClientError)

__all__ = [
    # Base class
    "PyAiAgent",
    # Provider: OpenAI
    "OpenAIAgent",
    # Multi-agent orchestration
    "handoff",
    "team",
    "pipeline",
    "parallel",
    # Shutdown function
    "shutdown",
    # Client configuration
    "set_default_openai_client",
    "get_default_openai_client",
    "clear_default_openai_client",
    # Generic exceptions
    "AgentDefinitionError",
    "AgentProcessError",
    "AgentClosedError",
    "InvalidInputError",
    "InvalidSessionError",
    "InvalidMetadataError",
    "InvalidHistoryError",
    "InvalidInstructionParamsError",
    "InstructionKeyError",
    "MaxStepsExceededError",
    "ValidationRetriesExhaustedError",
    # Backward-compatible OpenAI aliases
    "OpenAIAgentDefinitionError",
    "OpenAIAgentProcessError",
    "OpenAIAgentClosedError",
    # Provider-specific exceptions
    "ClientError",
]

__version__ = "0.1.6"
