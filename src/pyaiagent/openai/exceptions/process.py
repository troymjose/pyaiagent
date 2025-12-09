__all__ = [
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


class OpenAIAgentProcessError(Exception):
    pass


class OpenAIAgentClosedError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"'{agent_name}' has been closed. Create a new instance to continue.")


class InvalidInputError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"'{agent_name}' requires 'input' to be a string.")


class InvalidSessionError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"'{agent_name}' requires 'session' to be a non-empty string.")


class InvalidMetadataError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"'{agent_name}' requires 'metadata' to be a dict.")


class InvalidLlmMessagesError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"'{agent_name}' requires 'llm_messages' to be a list.")


class InvalidInstructionParamsError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"'{agent_name}' requires 'instruction_params' to be a dict.")


class InstructionKeyError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str, key: str):
        super().__init__(
            f"'{agent_name}' missing instruction key {key}. Provide it via 'instruction_params' in process().")


class ClientError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str, message: str):
        super().__init__(f"'{agent_name}' OpenAI API error: {message}")


class MaxStepsExceededError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", max_steps: int = 10):
        super().__init__(f"'{agent_name}' exceeded {max_steps} steps without completing.")
