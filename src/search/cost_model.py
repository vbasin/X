"""Per-family training cost model for the search stage.

Retained live helper extracted from the archived Model Scout agent
(``archive/legacy_research_director/model_scout.py``, deprecated 2026-07-14 — superseded by
Scoping v2 / ADR-025). Only :func:`estimate_cost` was ever used outside the agent
(``src.search.iterate`` for per-trial USD estimates); the LLM/deterministic proposer and its
journal are archived. This module is deliberately tiny and dependency-light.
"""

from __future__ import annotations

from typing import Any

from src.models.catalog import get_entry

# Rough per-family training cost by cost_class → minutes on the A10 host. Used only for
# cost-cap reasoning + try-ordering, not for billing. Ensembles reuse cached OOF (cheap).
_COST_MINUTES = {"cheap": 3.0, "moderate": 12.0, "expensive": 45.0, "unknown": 15.0}
_ENSEMBLE_MINUTES = 2.0
_A10_USD_PER_HOUR = 1.50            # conservative on-demand A10 estimate; override per call


def _cost_class(family: str) -> str:
    try:
        return get_entry(family).cost_class
    except Exception:  # noqa: BLE001 — unknown family → conservative default
        return "unknown"


def _family_minutes(family: str) -> float:
    return _COST_MINUTES.get(_cost_class(family), _COST_MINUTES["unknown"])


def estimate_cost(families: list[str], ensembles: list[dict[str, Any]] | None = None,
                  *, usd_per_hour: float = _A10_USD_PER_HOUR) -> dict[str, Any]:
    """Conservative (serial upper-bound) cost estimate for a candidate set."""
    minutes = sum(_family_minutes(f) for f in families)
    minutes += _ENSEMBLE_MINUTES * len(ensembles or [])
    return {
        "families": list(families),
        "est_minutes": round(minutes, 1),
        "est_cost_usd": round(minutes / 60.0 * usd_per_hour, 4),
        "usd_per_hour": usd_per_hour,
    }
