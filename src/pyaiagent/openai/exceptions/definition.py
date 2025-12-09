__all__ = ["OpenAIAgentDefinitionError", ]


class OpenAIAgentDefinitionError(TypeError):
    __slots__ = ()

    def __init__(self, cls_name: str, errors: list[str]):
        message = [f"\nInvalid OpenAIAgent Class '{cls_name}'"]
        if errors:
            message.append(
                "================================================== Errors ==================================================")
            message += [f"{index} => {error}" for index, error in enumerate(errors, start=1)]
            message.append(
                "============================================================================================================")
        super().__init__("\n".join(message))
