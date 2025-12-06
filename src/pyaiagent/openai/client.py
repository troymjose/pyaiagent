from openai import AsyncOpenAI
from pyaiagent.utils.singleton import PerEventLoopSingleton

__all__ = ["AsyncOpenAIClient", ]


class AsyncOpenAIClient(metaclass=PerEventLoopSingleton):
    """
    Per-event-loop singleton wrapper around `openai.AsyncOpenAI`.

    - One `AsyncOpenAIClient` instance per (class, event loop), enforced by
      the `PerEventLoopSingleton` metaclass.
    - Underlying `AsyncOpenAI` client is reused within the loop to enable
      connection pooling and reduce connection overhead.
    - Call `await AsyncOpenAIClient().aclose()` during shutdown to close
      network resources and remove the instance from the singleton registry.
    """

    __slots__ = ("_client", "_closed")

    def __init__(self, **kwargs):
        self._client: AsyncOpenAI = AsyncOpenAI(**kwargs)
        self._closed: bool = False

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._client.close()
        finally:
            # Always remove from the per-loop singleton registry
            type(self).delete_instance()
