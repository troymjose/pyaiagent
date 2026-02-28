"""OpenAI-specific exception aliases and provider-specific exceptions.

Generic exception classes live in :mod:`pyaiagent.core.exceptions`.
This module re-exports them under backward-compatible ``OpenAI*`` names
and defines provider-specific exceptions like :class:`ClientError`.
"""

from pyaiagent.core.exceptions import (
    AgentDefinitionError,
    AgentProcessError,
    AgentClosedError,
    InvalidInputError,
    InvalidSessionError,
    InvalidMetadataError,
    InvalidHistoryError,
    InvalidInstructionParamsError,
    InstructionKeyError,
    MaxStepsExceededError,
    ValidationRetriesExhaustedError,
)

# Backward-compatible aliases
OpenAIAgentDefinitionError = AgentDefinitionError
OpenAIAgentProcessError = AgentProcessError
OpenAIAgentClosedError = AgentClosedError

__all__ = [
    # Generic names (re-exported from core)
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
    # Provider-specific
    "ClientError",
]


class ClientError(AgentProcessError):
    """Raised when an OpenAI API call fails."""
    __slots__ = ()

    def __init__(self, *, agent_name: str, message: str) -> None:
        super().__init__(f"{agent_name}: OpenAI API error - {message}",
                         agent_name=agent_name)
