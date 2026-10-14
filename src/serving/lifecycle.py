"""Canonical lifecycle status vocabulary (``docs/ARCHITECTURE.md`` section 2).

Five statuses, deliberately small. Do not invent parallel vocabulary
("champion", "leader", "promoted") in new code -- those remain acceptable
*internal* synonyms only inside Track B's existing Model Lifecycle
implementation during migration (``ALLOWED_TRANSITIONS`` documents where
they conceptually map).
"""
from __future__ import annotations

from enum import Enum


class LifecycleStatus(str, Enum):
    """Status of one trained artifact / experiment within its lifecycle.

    EXPERIMENTAL: Under active development; not evaluated against the
        locked holdout; no promotion claim.
    CANDIDATE: Passed the canonical Validator + Economic Evaluator on a
        held-out (non-locked) window; eligible for promotion review.
        Conceptually maps to Track B's "candidate" in the Custom Test Lab
        leaderboard.
    PRODUCTION: Promoted; currently the trusted model/decision-policy pair
        for its Serving path. Conceptually maps to Track B's "champion" /
        ADR-071 champion-gate outcome.
    DEPRECATED: Formerly PRODUCTION or CANDIDATE, explicitly superseded;
        kept for reference/rollback.
    ARCHIVED: No longer under any active development; historical only
        (e.g. Track A in full, per ``project_manifest.yaml``).
    """

    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


ALLOWED_TRANSITIONS: dict[LifecycleStatus, frozenset[LifecycleStatus]] = {
    LifecycleStatus.EXPERIMENTAL: frozenset({LifecycleStatus.CANDIDATE, LifecycleStatus.ARCHIVED}),
    LifecycleStatus.CANDIDATE: frozenset({
        LifecycleStatus.PRODUCTION, LifecycleStatus.EXPERIMENTAL, LifecycleStatus.ARCHIVED,
    }),
    LifecycleStatus.PRODUCTION: frozenset({LifecycleStatus.DEPRECATED}),
    LifecycleStatus.DEPRECATED: frozenset({LifecycleStatus.ARCHIVED}),
    LifecycleStatus.ARCHIVED: frozenset(),
}
"""Valid forward transitions. Notably: PRODUCTION cannot go directly back to
CANDIDATE/EXPERIMENTAL -- a production model that needs rework is DEPRECATED
first (an explicit, recorded decision), then a *new* CANDIDATE takes its
place. This mirrors the audit's finding that Track B's one-ever-promoted
model (2026-08-05) was never formally deprecated despite paper trading
badly missing its backtest -- there is currently no code path that would
have forced that decision to be recorded."""


def is_valid_transition(current: LifecycleStatus, target: LifecycleStatus) -> bool:
    """Whether ``current -> target`` is an allowed lifecycle transition."""
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())
