import asyncio

from pyaiagent.openai.client import AsyncOpenAIClient


async def shutdown() -> None:
    """Close shared OpenAI async clients for this event loop."""
    """
    Gracefully close the shared AsyncOpenAIClient for the current event loop.

    - No-op if no client was ever created on this loop.
    - Safe to call multiple times.
    """
    try:
        await AsyncOpenAIClient().aclose()
    except RuntimeError:
        pass

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Called outside of an event loop (e.g., pure sync context) – nothing to close.
        return

    cls = AsyncOpenAIClient

    # Access the singleton registry without creating a new instance.
    try:
        lock = cls._lock
        registry = cls._instances_per_loop
    except AttributeError:
        # If for some reason the metaclass didn't set these, nothing to do.
        return

    # Get the instance for this loop under the lock, then release the lock
    with lock:
        instance: AsyncOpenAIClient = registry.get(loop)

    # No instance, or something unexpected like the _IN_PROGRESS sentinel
    if not isinstance(instance, AsyncOpenAIClient):
        return

    # This will close the underlying AsyncOpenAI client and remove the instance
    await instance.aclose()
