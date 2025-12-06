import asyncio
import threading
from weakref import WeakKeyDictionary

__all__ = ["PerEventLoopSingleton", ]

_MISSING = object()
_IN_PROGRESS = object()


class PerEventLoopSingleton(type):
    """
    Metaclass that ensures ONE instance per (class, event loop).

    Thread-safe via an internal lock. Safe to use across multiple threads
    that each have their own event loop (e.g. multi-threaded asyncio setups).

    Requires a running asyncio event loop at instantiation time; attempting to
    instantiate from synchronous code without a running loop will raise RuntimeError.

    Recursive construction (direct or indirect) of the same class within a single
    event loop is explicitly unsupported and will raise RuntimeError.
    """

    _instances_per_loop: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[type, object]] = WeakKeyDictionary()
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{cls.__name__} requires a running event loop; "
                "instantiate it from async code or after loop start."
            ) from exc

        with cls._lock:
            per_loop = cls._instances_per_loop.setdefault(loop, {})
            existing = per_loop.get(cls, _MISSING)

            if existing is _IN_PROGRESS:
                raise RuntimeError(
                    f"Recursive construction of singleton {cls.__name__} "
                    "within the same event loop is not supported."
                )
            if existing is not _MISSING:
                return existing

            per_loop[cls] = _IN_PROGRESS

        try:
            instance = super().__call__(*args, **kwargs)
        except BaseException:
            # Clear the in-progress marker to avoid poisoning the registry
            with cls._lock:
                per_loop = cls._instances_per_loop.get(loop)
                if per_loop and per_loop.get(cls) is _IN_PROGRESS:
                    per_loop.pop(cls, None)
            raise

        with cls._lock:
            per_loop = cls._instances_per_loop.setdefault(loop, {})
            per_loop[cls] = instance
        return instance

    def delete_instance(cls) -> bool:
        """
        Delete the instance associated with the current event loop.

        Returns:
            True if an instance was deleted, False otherwise (including when no loop is running).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        with cls._lock:
            per_loop = cls._instances_per_loop.get(loop)
            if not per_loop:
                return False
            removed = per_loop.pop(cls, None) is not None
            if not per_loop:
                cls._instances_per_loop.pop(loop, None)
            return removed

    @classmethod
    def delete_all_instances(cls) -> None:
        """Testing/maintenance helper: clear the global instance registry."""
        with cls._lock:
            cls._instances_per_loop.clear()
