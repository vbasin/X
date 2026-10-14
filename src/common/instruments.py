"""Instrument metadata used by Live / Backtest PnL display.

Point values are CME contract multipliers ($ per index point, 1 contract).
"""

from __future__ import annotations

from typing import Any

# CME equity-index futures: $ per full index point (1 contract).
POINT_VALUE_USD: dict[str, float] = {
    "NQ": 20.0,    # E-mini Nasdaq-100
    "MNQ": 2.0,    # Micro E-mini Nasdaq-100
    "ES": 50.0,    # E-mini S&P 500
    "MES": 5.0,    # Micro E-mini S&P 500
    "RTY": 50.0,   # E-mini Russell 2000
    "M2K": 5.0,    # Micro E-mini Russell 2000
    "YM": 5.0,     # E-mini Dow ($5 × index point)
}


def point_value_usd(symbol: str | None, default: float = 1.0) -> float:
    """Return $ per point for *symbol* (case-insensitive), else *default*."""
    if not symbol:
        return float(default)
    return float(POINT_VALUE_USD.get(str(symbol).strip().upper(), default))


def primary_symbol(objective: dict[str, Any] | None) -> str | None:
    """Primary traded symbol from an objective doc's ``traded products`` list."""
    if not objective:
        return None
    products = objective.get("traded products") or objective.get("instruments") or []
    for row in products:
        if not isinstance(row, dict):
            continue
        if row.get("role", "primary") == "primary" and row.get("symbol"):
            return str(row["symbol"])
    # Fallback: first entry with a symbol
    for row in products:
        if isinstance(row, dict) and row.get("symbol"):
            return str(row["symbol"])
    # Last resort: objective_id prefix (e.g. nq-50pt-econ → NQ)
    oid = str(objective.get("objective_id") or "")
    if oid.lower().startswith("nq"):
        return "NQ"
    if oid.lower().startswith("es"):
        return "ES"
    return None


def point_value_from_objective(
    objective: dict[str, Any] | None,
    *,
    default: float = 1.0,
) -> float:
    """$ per point for the objective's primary instrument (NQ → 20, …)."""
    return point_value_usd(primary_symbol(objective), default=default)
