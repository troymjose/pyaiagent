"""
PyAiAgent is a modern, fast (high-performance), async framework for building AI agents with pythonic code.
"""

from pyaiagent.openai.agent import OpenAIAgent
from pyaiagent.openai.shutdown import shutdown
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
    # AI Agent
    "OpenAIAgent",
    # Shutdown function
    "shutdown",
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

__version__ = "0.1.3"
