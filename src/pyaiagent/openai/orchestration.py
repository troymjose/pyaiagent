"""Multi-agent orchestration — re-exported from core.

The canonical implementation lives in :mod:`pyaiagent.core.orchestration`.
This module provides backward-compatible imports for existing code
that imports from ``pyaiagent.openai.orchestration``.
"""
from pyaiagent.core.orchestration import (               # noqa: F401
    team,
    pipeline,
    parallel,
)

from pyaiagent.core.orchestration import __all__          # noqa: F401
