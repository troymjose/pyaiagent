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
    def __init__(self):
        super().__init__("The OpenAIAgent instance has been closed and can no longer process requests.")


class InvalidInputError(OpenAIAgentProcessError):
    def __init__(self):
        super().__init__("Argument 'input' must be a string.")


class InvalidSessionError(OpenAIAgentProcessError):
    def __init__(self):
        super().__init__("Argument 'session' must be a non empty string.")


class InvalidMetadataError(OpenAIAgentProcessError):
    def __init__(self):
        super().__init__("Argument 'metadata' must be a dictionary.")


class InvalidLlmMessagesError(OpenAIAgentProcessError):
    def __init__(self):
        super().__init__(
            "Argument 'llm_messages' should be of type list. It should align with the expected format for messages in the OpenAI Response API.")


class InvalidInstructionParamsError(OpenAIAgentProcessError):
    def __init__(self):
        super().__init__(
            "Argument 'instruction_params' should be of type dict. It should contain key-value pairs that parameterize the agent's instructions.")


class InstructionKeyError(OpenAIAgentProcessError):
    def __init__(self, message):
        super().__init__(
            f"{message} Key Missing. Argument 'instruction_params' passed to process() method should contain all required keys for instruction parameterization.")


class ClientError(OpenAIAgentProcessError):
    def __init__(self, message):
        super().__init__(f"OpenAI Client Error: '{message}'")


class MaxStepsExceededError(OpenAIAgentProcessError):
    def __init__(self):
        super().__init__("Maximum number of steps exceeded without reaching a final response.")
