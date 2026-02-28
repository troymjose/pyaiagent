from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI
from pyaiagent.utils import PerEventLoopSingleton

__all__ = [
    "AsyncOpenAIClient",
    "set_default_openai_client",
    "get_default_openai_client",
    "clear_default_openai_client",
    "shutdown",
]

logger = logging.getLogger(__name__)

_default_openai_client: AsyncOpenAI | None = None


def set_default_openai_client(client: AsyncOpenAI) -> None:
    """
    Set the default AsyncOpenAI client for all agents.

    Call this function before creating any agents to provide a custom-configured
    client. This gives you full control over all AsyncOpenAI parameters.

    Note: This must be called BEFORE any agent is used for the first time
    in any event loop. Once an active singleton wrapper exists in any event
    loop across the process, calling this function raises ``RuntimeError``.
    Call ``shutdown()`` on each active event loop first to reconfigure.

    Args:
        client: A configured AsyncOpenAI client instance.

    Raises:
        RuntimeError: If an active AsyncOpenAIClient singleton exists.

    Example:
        from openai import AsyncOpenAI
        from pyaiagent import set_default_openai_client

        custom_client = AsyncOpenAI(
            api_key="sk-xxx",
            base_url="https://api.example.com/v1",
            timeout=60.0,
            max_retries=3,
            http_client=my_custom_httpx_client,
        )
        set_default_openai_client(custom_client)

        agent = MyAgent()
        result = await agent.process(input="Hello")
    """
    n_active = sum(1 for inst in AsyncOpenAIClient.iter_instances() if inst.is_active)
    if n_active:
        logger.info(
            "Refusing to set default client: %d active instance(s)", n_active)
        raise RuntimeError(
            "set_default_openai_client() must be called BEFORE any agent is used. "
            "An active AsyncOpenAIClient singleton exists. "
            "Call shutdown() on each active event loop first to reconfigure.")

    global _default_openai_client
    _default_openai_client = client
    logger.debug("Default OpenAI client set")


def get_default_openai_client() -> AsyncOpenAI | None:
    """
    Get the current default AsyncOpenAI client.

    Returns:
        The default client if set, otherwise None.
    """
    return _default_openai_client


def clear_default_openai_client() -> None:
    """
    Clear the default AsyncOpenAI client reference.

    This does NOT close the client or the active singleton — it only removes
    the default reference. Use ``shutdown()`` to close the active singleton
    and release network resources.
    """
    global _default_openai_client
    _default_openai_client = None
    logger.debug("Default OpenAI client cleared")


class AsyncOpenAIClient(metaclass=PerEventLoopSingleton):
    """
    Per-event-loop singleton wrapper around ``openai.AsyncOpenAI``.

    One ``AsyncOpenAIClient`` instance per (class, event loop), enforced by
    the ``PerEventLoopSingleton`` metaclass. This is a per-event-loop
    singleton, not a process-global one — each event loop gets its own
    instance. ``set_default_openai_client()`` affects future instances across
    all loops but cannot be changed while any loop has an active instance.

    - Underlying ``AsyncOpenAI`` client is reused within the loop to enable
      connection pooling and reduce connection overhead.
    - Call ``await AsyncOpenAIClient().aclose()`` or ``await shutdown()``
      during shutdown to close network resources and remove the instance
      from the singleton registry.
    - If the underlying client was injected (via constructor or global default),
      ``aclose()`` will NOT close it — only internally-created clients are closed.

    Shutdown and in-flight operations:
        Once ``aclose()`` is called, new accesses to ``.client`` raise
        ``RuntimeError``. When ``owns_client`` is ``True``, the
        underlying HTTP transport is also closed, which **may cause
        in-flight requests to fail**. When ``owns_client`` is ``False``
        (injected client), the transport remains open and in-flight
        operations are unaffected.

        For graceful shutdown, stop accepting new work, ``await`` all
        in-flight tasks, then call ``aclose()`` or ``shutdown()``.

    Configuration:
        Set a custom client before using any agents::

            from openai import AsyncOpenAI
            from pyaiagent import set_default_openai_client

            client = AsyncOpenAI(api_key="sk-xxx", base_url="...")
            set_default_openai_client(client)
    """

    __slots__ = ("_client", "_closed", "_owns_client")

    def __init__(self, client: AsyncOpenAI | None = None):
        if client is not None:
            self._client = client
            self._owns_client = False
            logger.debug("AsyncOpenAIClient initialized with explicit client")
        elif _default_openai_client is not None:
            self._client = _default_openai_client
            self._owns_client = False
            logger.debug("AsyncOpenAIClient initialized with default client")
        else:
            self._client = AsyncOpenAI()
            self._owns_client = True
            logger.debug("AsyncOpenAIClient created new internal client")
        self._closed: bool = False

    @property
    def is_active(self) -> bool:
        """Whether this client wrapper is still open and usable."""
        return not self._closed

    @property
    def owns_client(self) -> bool:
        """Whether this wrapper owns (and will close) the underlying client."""
        return self._owns_client

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
            if self._owns_client:
                await asyncio.shield(self._client.close())
                logger.debug("Owned AsyncOpenAI client closed")
            else:
                logger.debug("AsyncOpenAIClient released (external client not closed)")
        finally:
            type(self).delete_instance()


async def shutdown(*, raise_on_error: bool = False) -> None:
    """
    Gracefully close the shared AsyncOpenAIClient for the current event loop.

    - No-op if no client was ever created on this loop.
    - Safe to call multiple times.

    Args:
        raise_on_error: If ``True``, re-raise exceptions from the underlying
            close (for environments where resource leaks must be fatal).
            If ``False`` (default), log the error at ``WARNING`` and continue.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    instance = AsyncOpenAIClient.get_instance(loop)
    if instance is None:
        return

    owns = instance.owns_client
    try:
        await instance.aclose()
        logger.info("AsyncOpenAIClient shutdown complete (owns_client=%s)", owns)
    except Exception:
        logger.warning(
            "Error during AsyncOpenAIClient shutdown (owns_client=%s)",
            owns,
            exc_info=True,
        )
        if raise_on_error:
            raise
