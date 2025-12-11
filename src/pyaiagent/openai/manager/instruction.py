import textwrap
from pyaiagent.openai.exceptions.definition import OpenAIAgentDefinitionError

__all__ = ["OpenAIAgentInstructionManager", ]


class OpenAIAgentInstructionManager:
    @staticmethod
    def create(cls) -> str:
        """ Return normalized agent instruction text from the class docstring. """
        # Clean and normalize docstring text
        instruction = textwrap.dedent(cls.__doc__ or "").strip()
        if not instruction:
            raise OpenAIAgentDefinitionError(
                cls_name=cls.__name__,
                errors=["Missing class docstring. Add a triple-quoted docstring as agent instruction."])
        return instruction
