__all__ = ["OpenAIAgentProcessError",
           "OpenAIAgentClosedError",
           "InvalidInputError",
           "InvalidSessionError",
           "InvalidMetadataError",
           "InvalidLlmMessagesError",
           "InvalidInstructionParamsError",
           "InstructionKeyError",
           "ClientError",
           "MaxStepsExceededError", ]


class OpenAIAgentProcessError(Exception):
    pass


class OpenAIAgentClosedError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent"):
        super().__init__(f"{agent_name}: agent is closed, create a new instance")


class InvalidInputError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", received: str = "unknown"):
        super().__init__(f"{agent_name}: 'input' must be str, not {received}")


class InvalidSessionError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", received: str = "unknown"):
        super().__init__(f"{agent_name}: 'session' must be a non-empty str, not {received}")


class InvalidMetadataError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", received: str = "unknown"):
        super().__init__(f"{agent_name}: 'metadata' must be dict, not {received}")


class InvalidLlmMessagesError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", received: str = "unknown"):
        super().__init__(f"{agent_name}: 'llm_messages' must be list, not {received}")


class InvalidInstructionParamsError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", received: str = "unknown"):
        super().__init__(f"{agent_name}: 'instruction_params' must be dict, not {received}")


class InstructionKeyError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str, key: str):
        super().__init__(f"{agent_name}: missing instruction key '{key}'")


class ClientError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str, message: str):
        super().__init__(f"{agent_name}: OpenAI API error - {message}")


class MaxStepsExceededError(OpenAIAgentProcessError):
    def __init__(self, *, agent_name: str = "OpenAIAgent", max_steps: int = 10):
        super().__init__(f"{agent_name}: exceeded {max_steps} steps without completing")
