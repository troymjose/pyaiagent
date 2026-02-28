from __future__ import annotations

import asyncio
import threading
from weakref import WeakKeyDictionary

__all__ = ["PerEventLoopSingleton", ]

_MISSING = object()
_IN_PROGRESS = object()


class PerEventLoopSingleton(type):
    """
    Metaclass that ensures ONE instance per event loop
    Any class that uses this metaclass will return the same instance for a given asyncio event loop,
    and a different instance for a different event loop

    Its Thread-safe!
    Safe to use from multiple threads, provided each thread owns its own event loop and
    each event loop is only used from its owning thread (the typical asyncio pattern).

    Requires a running asyncio event loop at instantiation time
    Attempting to instantiate from synchronous code without a running loop will raise RuntimeError.

    Recursive construction (direct or indirect) of the same class within a single
    event loop is explicitly unsupported and will raise RuntimeError.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        cls._instances_per_loop = WeakKeyDictionary()
        cls._lock = threading.Lock()
        return cls

    def __call__(cls, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                f"'{cls.__name__}' class requires a running event loop. Please instantiate it from async code or after loop start.") from exc

        # Cache class attributes locally for faster access
        lock = cls._lock
        registry = cls._instances_per_loop

        with lock:
            existing = registry.get(loop, _MISSING)
            # Detect recursive construction
            if existing is _IN_PROGRESS:
                raise RuntimeError(
                    f"Recursive construction of singleton class '{cls.__name__}' within the same event loop is not supported.")
            # Return existing instance if found
            if existing is not _MISSING:
                return existing
            # Mark as in-progress
            registry[loop] = _IN_PROGRESS

        # Create the instance outside the lock to avoid deadlocks
        try:
            instance = super().__call__(*args, **kwargs)
        except BaseException:
            # Clear the in-progress marker to avoid poisoning the registry
            with lock:
                registry.pop(loop, None)
            raise

        # Store the created instance
        with lock:
            registry[loop] = instance

        return instance

    def delete_instance(cls) -> bool:
        """
        Delete the instance associated with the current event loop.
        Returns True if an instance was deleted, False otherwise (including when no loop is running).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        with cls._lock:
            existing = cls._instances_per_loop.get(loop, _MISSING)
            # Nothing to delete, or construction was in-progress
            if existing in (_MISSING, _IN_PROGRESS):
                return False
            cls._instances_per_loop.pop(loop, None)
            return True

    def get_instance(
        cls, loop: asyncio.AbstractEventLoop | None = None,
    ) -> object | None:
        """
        Get the singleton instance for a specific event loop without creating one.

        Args:
            loop: The event loop to look up. Defaults to the current running loop.

        Returns:
            The singleton instance if one exists, ``None`` otherwise
            (including when construction is in-progress or no loop is running).
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return None
        with cls._lock:
            existing = cls._instances_per_loop.get(loop)
            if existing is _IN_PROGRESS:
                return None
            return existing

    def iter_instances(cls) -> list[object]:
        """
        Return a snapshot list of all registered instances across all event loops.

        Thread-safe. The returned list is a copy, safe to iterate outside the lock.
        Excludes in-progress construction markers.
        """
        with cls._lock:
            return [
                inst for inst in cls._instances_per_loop.values()
                if inst is not _IN_PROGRESS
            ]

    def delete_all_instances(cls) -> None:
        """Testing/maintenance helper: clear the global instance registry."""
        with cls._lock:
            cls._instances_per_loop.clear()
