from __future__ import annotations

from openai import AsyncOpenAI
from pyaiagent.utils.singleton import PerEventLoopSingleton

__all__ = ["AsyncOpenAIClient", "set_default_openai_client", "get_default_openai_client", ]

# Global default client storage
_default_openai_client: AsyncOpenAI | None = None


def set_default_openai_client(client: AsyncOpenAI) -> None:
    """
    Set the default AsyncOpenAI client for all agents.
    
    Call this function before creating any agents to provide a custom-configured
    client. This gives you full control over all AsyncOpenAI parameters.
    
    Note: This must be called BEFORE any agent is used for the first time
    in the current event loop. Once the singleton wrapper is created,
    calling this function will not affect existing agent instances.
    
    Args:
        client: A configured AsyncOpenAI client instance.
    
    Example:
        from openai import AsyncOpenAI
        from pyaiagent import set_default_openai_client
        
        # Create a custom client with full control over all parameters
        custom_client = AsyncOpenAI(
            api_key="sk-xxx",
            base_url="https://api.example.com/v1",
            timeout=60.0,
            max_retries=3,
            http_client=my_custom_httpx_client,  # Full flexibility
        )
        
        # Set it as the default
        set_default_openai_client(custom_client)
        
        # Now create and use agents - they'll use this client
        agent = MyAgent()
        result = await agent.process(input="Hello")
    """
    global _default_openai_client
    _default_openai_client = client


def get_default_openai_client() -> AsyncOpenAI | None:
    """
    Get the current default AsyncOpenAI client.
    
    Returns:
        The default client if set, otherwise None.
    """
    return _default_openai_client


class AsyncOpenAIClient(metaclass=PerEventLoopSingleton):
    """
    Per-event-loop singleton wrapper around `openai.AsyncOpenAI`.

    - One `AsyncOpenAIClient` instance per (class, event loop), enforced by
      the `PerEventLoopSingleton` metaclass.
    - Underlying `AsyncOpenAI` client is reused within the loop to enable
      connection pooling and reduce connection overhead.
    - Call `await AsyncOpenAIClient().aclose()` during shutdown to close
      network resources and remove the instance from the singleton registry.
    
    Configuration:
        Set a custom client before using any agents:
        
        ```python
        from openai import AsyncOpenAI
        from pyaiagent import set_default_openai_client
        
        client = AsyncOpenAI(api_key="sk-xxx", base_url="...")
        set_default_openai_client(client)
        ```
    """

    __slots__ = ("_client", "_closed")

    def __init__(self, client: AsyncOpenAI | None = None):
        # Priority: explicit client > global default > create new
        if client is not None:
            self._client = client
        elif _default_openai_client is not None:
            self._client = _default_openai_client
        else:
            self._client = AsyncOpenAI()
        self._closed: bool = False

    @property
    def client(self) -> AsyncOpenAI:
        if self._closed:
            raise RuntimeError("AsyncOpenAIClient is closed. Create a new instance.")
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
