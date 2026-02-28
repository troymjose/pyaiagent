"""Provider-agnostic exception hierarchy for the pyaiagent framework.

Definition-time errors (raised when a class is defined):
    AgentDefinitionError   — invalid class setup (missing docstring, bad config, etc.)

Process-time errors (raised during agent execution):
    AgentProcessError      — abstract base for all process-time errors
    ├── AgentClosedError
    ├── InvalidInputError
    ├── InvalidSessionError
    ├── InvalidMetadataError
    ├── InvalidHistoryError
    ├── InvalidInstructionParamsError
    ├── InstructionKeyError
    ├── MaxStepsExceededError
    └── ValidationRetriesExhaustedError

Provider-specific subclasses (e.g. ``ClientError``) live in their
respective provider packages (``pyaiagent.openai``, etc.).
"""

__all__ = [
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
]


class AgentDefinitionError(TypeError):
    """Raised at class-definition time when the agent class is invalid."""
    __slots__ = ()

    def __init__(self, *, cls_name: str, errors: list[str]) -> None:
        n = len(errors)
        header = f"'{cls_name}' has {n} definition error{'s' if n != 1 else ''}"
        details = "\n".join(f"  • {error}" for error in errors)
        super().__init__(f"{header}:\n{details}")


class AgentProcessError(Exception):
    """Base class for errors raised during :meth:`process`."""
    __slots__ = ("agent_name", "tokens")

    def __init__(self, message: str = "", *, agent_name: str = "Agent") -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.tokens: dict[str, int] | None = None


class AgentClosedError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent") -> None:
        super().__init__(f"{agent_name}: agent is closed, create a new instance",
                         agent_name=agent_name)


class InvalidInputError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent", received: str = "unknown") -> None:
        super().__init__(f"{agent_name}: 'input' must be str, not {received}",
                         agent_name=agent_name)


class InvalidSessionError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent", received: str = "unknown") -> None:
        super().__init__(f"{agent_name}: 'session' must be a non-empty str, not {received}",
                         agent_name=agent_name)


class InvalidMetadataError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent", received: str = "unknown") -> None:
        super().__init__(f"{agent_name}: 'metadata' must be dict, not {received}",
                         agent_name=agent_name)


class InvalidHistoryError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent", received: str = "unknown") -> None:
        super().__init__(f"{agent_name}: 'history' must be list, not {received}",
                         agent_name=agent_name)


class InvalidInstructionParamsError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent", received: str = "unknown") -> None:
        super().__init__(f"{agent_name}: 'instruction_params' must be dict, not {received}",
                         agent_name=agent_name)


class InstructionKeyError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str, key: str) -> None:
        super().__init__(f"{agent_name}: missing instruction key '{key}'",
                         agent_name=agent_name)


class MaxStepsExceededError(AgentProcessError):
    __slots__ = ()

    def __init__(self, *, agent_name: str = "Agent", max_steps: int = 10) -> None:
        super().__init__(f"{agent_name}: exceeded {max_steps} steps without completing",
                         agent_name=agent_name)


class ValidationRetriesExhaustedError(AgentProcessError):
    __slots__ = ("validation_errors",)

    def __init__(self, *, agent_name: str = "Agent", validation_retries: int = 0, errors: str = "") -> None:
        self.validation_errors = errors
        super().__init__(
            f"{agent_name}: structured output validation failed after {validation_retries} "
            f"retry attempt(s). Last validation errors:\n{errors}",
            agent_name=agent_name)
