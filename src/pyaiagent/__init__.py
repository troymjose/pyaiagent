"""
pyaiagent - A lightweight, high-performance framework for building OpenAI-powered agents in Python.
"""

from pyaiagent.openai.agent import OpenAIAgent
from pyaiagent.openai.config import OpenAIAgentConfig
from pyaiagent.openai.exceptions.definition import OpenAIAgentDefinitionError
from pyaiagent.openai.exceptions.process import (OpenAIAgentProcessError,
                                                 OpenAIAgentClosedError,
                                                 InvalidInputError,
                                                 InvalidSessionError,
                                                 InvalidMetadataError,
                                                 InvalidLlmMessagesError,
                                                 InvalidInstructionParamsError,
                                                 InstructionKeyError,
                                                 ClientError,
                                                 MaxStepsExceededError)

__all__ = [
    # Core
    "OpenAIAgent",
    "OpenAIAgentConfig",
    # Exceptions
    "OpenAIAgentDefinitionError",
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
]

__version__ = "0.1.0"
