"""Shared critic harness — re-exports Scoping-v2 implementation.

All agentic loops (Scoping, Research Director) use the same parameterized
critic (artifact-first → targeted verification; cross-model provider matrix).
"""

from __future__ import annotations

from src.scoping.critic import (  # noqa: F401
    CriticHarness,
    field_persona,
    filter_cross_objections,
    load_duties,
    material_objections,
    resolve_critic_provider,
    review_schema,
)

__all__ = [
    "CriticHarness",
    "field_persona",
    "filter_cross_objections",
    "load_duties",
    "material_objections",
    "resolve_critic_provider",
    "review_schema",
]
