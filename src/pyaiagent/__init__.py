"""
PyAiAgent is a modern, fast (high-performance), async framework for building AI agents with pythonic code.
"""

from pyaiagent.openai.agent import OpenAIAgent
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
]

__version__ = "0.1.0"
