"""
PyAiAgent is a modern, fast (high-performance), async framework for building AI agents with pythonic code.
"""

from pyaiagent.openai.agent import OpenAIAgent
from pyaiagent.openai.shutdown import shutdown
from pyaiagent.openai.client import set_default_openai_client, get_default_openai_client
from pyaiagent.openai.exceptions.process import (OpenAIAgentProcessError,
                                                 OpenAIAgentClosedError,
                                                 InvalidInputError,
                                                 InvalidSessionError,
                                                 InvalidMetadataError,
                                                 InvalidLlmMessagesError,
                                                 InvalidInstructionParamsError,
                                                 InstructionKeyError,
                                                 ClientError,
                                                 MaxStepsExceededError,
                                                 ValidationRetriesExhaustedError)

__all__ = [
    # AI Agent
    "OpenAIAgent",
    # Shutdown function
    "shutdown",
    # Client configuration
    "set_default_openai_client",
    "get_default_openai_client",
    # Exceptions (runtime errors)
    "OpenAIAgentProcessError",
    "OpenAIAgentClosedError",
    "InvalidInputError",
    "InvalidSessionError",
    "InvalidMetadataError",
    "InvalidLlmMessagesError",
    "InvalidInstructionParamsError",
    "InstructionKeyError",
    "ClientError",
    "MaxStepsExceededError",
    "ValidationRetriesExhaustedError",
]

__version__ = "0.1.6"
