"""Provider-agnostic configuration merging for pyaiagent.

The :func:`merge_config` function reads an optional inner ``Config`` class
from an agent definition and merges it with inherited configuration from
parent classes.  Validation is left to provider-specific resolvers.
"""
from __future__ import annotations

from typing import Any

__all__ = ["merge_config"]


def merge_config(
    inner_config_cls: type | None,
    parent_config_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge an inner ``Config`` class with inherited parent configuration.

    Walks the attributes of *inner_config_cls* (skipping private names,
    callables, and descriptors) and overlays them onto a copy of
    *parent_config_kwargs*.

    Args:
        inner_config_cls:    The inner ``Config`` class defined on the
                             agent subclass, or ``None``.
        parent_config_kwargs: Configuration dict inherited from base classes,
                              or ``None``.

    Returns:
        A merged configuration dict.
    """
    if parent_config_kwargs:
        config_kwargs: dict[str, Any] = dict(parent_config_kwargs)
    else:
        config_kwargs = {}

    if inner_config_cls is None:
        return config_kwargs

    for key, value in vars(inner_config_cls).items():
        if key.startswith("_") or callable(value) or hasattr(value, "__get__"):
            continue
        config_kwargs[key] = value

    return config_kwargs
