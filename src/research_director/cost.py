"""Backward-compatible shim — canonical module is :mod:`src.common.llm.cost`.

New code: ``from src.common.llm import CostMeter, CostCapExceeded``.
"""

from __future__ import annotations

from src.common.llm.cost import (
    DEGRADE,
    HALT,
    OK,
    WARN,
    CostCapExceeded,
    CostMeter,
)

__all__ = ["CostCapExceeded", "CostMeter", "OK", "WARN", "DEGRADE", "HALT"]
