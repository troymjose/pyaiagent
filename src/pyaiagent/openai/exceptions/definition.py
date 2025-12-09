__all__ = ["OpenAIAgentDefinitionError"]


class OpenAIAgentDefinitionError(TypeError):
    __slots__ = ()

    def __init__(self, *, cls_name: str, errors: list[str]):
        header = f"\n'{cls_name}' has {len(errors)} definition error{'s' if len(errors) != 1 else ''}"
        details = "\n".join(f"  • {error}" for error in errors)
        super().__init__(f"{header}:\n{details}")
