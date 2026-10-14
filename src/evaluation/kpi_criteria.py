"""KPI criteria — objective `requirement` (champion gate) vs `target` (goal).

An objective's ``kpi`` block declares two condition sets:

  kpi.requirement  — the bar a model/ensemble must clear to be CONSIDERED
                     (evaluated by the champion gate).
  kpi.target       — the goal the program aims to reach through iterative
                     experimentation (a flag, not a gate).

Each is a *criteria block* with exactly one of:
  {"any": [cond, ...]}   → satisfied if ANY condition holds (OR)
  {"all": [cond, ...]}   → satisfied if ALL conditions hold (AND)

A *condition* is::

  {metric, direction?, tau?, min, op?, min_signals?}
    metric      : classification — "precision_at_tau" | "recall" | "directional_f1" | "f1"
                  economics     — "net_expectancy" | "hit_rate" | "sharpe" |
                                  "max_drawdown" | "n_signals"   (ADR-027)
    direction   : "up" | "down"   (per-class metrics; omit precision_at_tau → overall)
    tau         : float           (for precision_at_tau)
    min         : float           (threshold; max_drawdown uses a negative floor)
    op          : "gte" (default) | "gt"
    min_signals : int   (precision_at_tau only; default DEFAULT_MIN_SIGNALS)

Metric values are read from a candidate's ``diagnostics`` dict (produced by
``compute_diagnostics``) and its compact ``score`` dict (``canonical_trade_kpi``).
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_MIN_SIGNALS = 10

# Economics scalars live on the compact score (and mirrored under diagnostics.pnl_at_gate).
_ECON_METRICS = frozenset({
    "net_expectancy", "hit_rate", "sharpe", "max_drawdown", "n_signals",
})


# ── Metric lookup ─────────────────────────────────────────────────────────────

def _metric_value(
    cond: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    score: dict[str, Any] | None,
) -> tuple[float | None, int | None]:
    """Return ``(value, n_signals)`` for a condition. n_signals is None when N/A."""
    metric = cond.get("metric")
    direction = cond.get("direction")
    diag = diagnostics or {}
    sc = score or {}

    if metric == "precision_at_tau":
        tau = cond.get("tau")
        # Per-direction: prefer the diagnostics sweep, then score precision_up/down.
        if direction in ("up", "down"):
            for e in diag.get("precision_at_tau_sweep", []):
                if e.get("direction") == direction and abs(float(e.get("tau", -1)) - float(tau or -1)) < 1e-9:
                    return e.get("precision"), int(e.get("n_signals", 0))
            key = "precision_up" if direction == "up" else "precision_down"
            n_key = "n_up" if direction == "up" else "n_down"
            if key in sc:
                return sc.get(key), int(sc.get(n_key, 0))
            return None, None
        # Overall precision at the score's primary tau (no direction in the condition).
        if "precision_at_tau" in sc:
            return sc.get("precision_at_tau"), int(sc.get("n_signals", 0) or 0)
        return None, None

    if metric in ("recall", "f1"):
        rep = diag.get("classification_report", {})
        cell = rep.get(direction, {})
        field = "recall" if metric == "recall" else "f1-score"
        val = cell.get(field)
        return (float(val) if val is not None else None), None

    if metric == "directional_f1":
        val = sc.get("directional_f1", diag.get("f1_macro"))
        return (float(val) if val is not None else None), None

    if metric in _ECON_METRICS:
        # Prefer compact score; fall back to diagnostics.pnl_at_gate for PnL fields.
        if metric == "n_signals":
            # Prefer actual directional trades at the gate (not "any conf>=τ" rows,
            # which include high-confidence flats and inflate n_signals badly).
            trades = diag.get("trades_at_gate") or sc.get("trades_at_tau") or {}
            val = trades.get("taken")
            if val is None:
                # Fall back to directional counts on the compact score.
                n_up = sc.get("n_up")
                n_down = sc.get("n_down")
                if n_up is not None or n_down is not None:
                    val = int(n_up or 0) + int(n_down or 0)
                else:
                    val = sc.get("n_signals")
            return (float(val) if val is not None else None), None
        val = sc.get(metric)
        if val is None:
            pnl = diag.get("pnl_at_gate") or sc.get("pnl") or {}
            val = pnl.get(metric)
        return (float(val) if val is not None else None), None

    return None, None


def _condition_met(
    cond: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    score: dict[str, Any] | None,
    *,
    default_min_signals: int = DEFAULT_MIN_SIGNALS,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate one condition. Returns ``(passed, detail)``."""
    value, n_signals = _metric_value(cond, diagnostics, score)
    min_v = float(cond.get("min", 0.0))
    op = cond.get("op", "gte")

    detail = {
        "metric": cond.get("metric"), "direction": cond.get("direction"),
        "tau": cond.get("tau"), "min": min_v, "op": op,
        "value": value, "n_signals": n_signals,
    }

    if value is None or (isinstance(value, float) and math.isnan(value)):
        detail["passed"] = False
        detail["why"] = "no_value"
        return False, detail

    # Minimum directional signal count guard (precision_at_tau only)
    if cond.get("metric") == "precision_at_tau":
        ms = int(cond.get("min_signals", default_min_signals))
        if n_signals is not None and n_signals < ms:
            detail["passed"] = False
            detail["why"] = f"too_few_signals ({n_signals} < {ms})"
            return False, detail

    passed = (value > min_v) if op == "gt" else (value >= min_v)
    detail["passed"] = bool(passed)
    return bool(passed), detail


# ── Block evaluation ──────────────────────────────────────────────────────────

def evaluate_block(
    block: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None,
    score: dict[str, Any] | None,
    *,
    default_min_signals: int = DEFAULT_MIN_SIGNALS,
    missing_default: bool = False,
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate a criteria block (``{"any": [...]}`` or ``{"all": [...]}``).

    Returns ``(passed, details)``. If *block* is missing/empty, returns
    ``(missing_default, [])``.
    """
    if not block:
        return missing_default, []

    if "any" in block:
        conds, mode = block["any"], "any"
    elif "all" in block:
        conds, mode = block["all"], "all"
    else:
        return missing_default, []

    details = [_condition_met(c, diagnostics, score,
                              default_min_signals=default_min_signals)[1] for c in conds]
    flags = [d["passed"] for d in details]
    passed = any(flags) if mode == "any" else all(flags)
    return passed, details


def meets_requirement(
    objective: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    score: dict[str, Any] | None,
    *,
    default_min_signals: int = DEFAULT_MIN_SIGNALS,
) -> tuple[bool, list[dict[str, Any]]]:
    """Does the candidate clear the objective's champion-gate ``requirement``?

    A missing ``requirement`` block means "no bar" → passes.
    """
    block = (objective.get("kpi") or {}).get("requirement")
    return evaluate_block(block, diagnostics, score,
                          default_min_signals=default_min_signals, missing_default=True)


def meets_target(
    objective: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    score: dict[str, Any] | None,
    *,
    default_min_signals: int = DEFAULT_MIN_SIGNALS,
) -> tuple[bool, list[dict[str, Any]]]:
    """Does the candidate achieve the objective's experimentation ``target``?

    A missing ``target`` block means "no target defined" → not achieved.
    """
    block = (objective.get("kpi") or {}).get("target")
    return evaluate_block(block, diagnostics, score,
                          default_min_signals=default_min_signals, missing_default=False)


# ── Tau helpers (used by the contract + HPO to know which τ to score) ─────────

def requirement_conditions(objective: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``kpi.requirement`` ``any``/``all`` rows (empty if the gate is not defined)."""
    block = ((objective or {}).get("kpi") or {}).get("requirement") or {}
    return list(block.get("any") or []) + list(block.get("all") or [])


def objective_taus(objective: dict[str, Any]) -> list[float]:
    """All τ values referenced by the requirement + target precision conditions,
    unioned with the standard sweep, sorted ascending."""
    taus: set[float] = set()
    kpi = objective.get("kpi") or {}
    if kpi.get("tau") is not None:
        taus.add(round(float(kpi["tau"]), 4))
    for block in (kpi.get("requirement"), kpi.get("target")):
        if not block:
            continue
        for conds in (block.get("any", []), block.get("all", [])):
            for c in conds:
                if c.get("metric") == "precision_at_tau" and c.get("tau") is not None:
                    taus.add(round(float(c["tau"]), 4))
    taus |= {0.5, 0.6, 0.7, 0.8, 0.9}
    return sorted(taus)


def primary_tau(objective: dict[str, Any]) -> float | None:
    """Declared fire-cut τ, or ``None`` when the YAML defines no confidence cut.

    Order: ``kpi.tau`` (optional) → lowest ``precision_at_tau.tau`` on
    ``kpi.requirement``. Never infers 0.8. ``None`` means count every argmax
    up/down (same population as ``confusion_matrix_raw``).
    """
    kpi = objective.get("kpi") or {}
    if kpi.get("tau") is not None:
        return float(kpi["tau"])
    block = kpi.get("requirement") or {}
    taus = [
        float(c["tau"])
        for conds in (block.get("any", []), block.get("all", []))
        for c in conds
        if c.get("metric") == "precision_at_tau" and c.get("tau") is not None
    ]
    return min(taus) if taus else None


def fire_tau(
    objective: dict[str, Any],
    *,
    report_only: bool = False,
) -> float | None:
    """τ used to count trades for this run.

    Dual-binary objectives use ``labels.fire`` (ADR-095) even on Custom Test
    report runs — the event cut is part of the book, not a champion gate.
    3-class Custom Test stays argmax (``report_only`` → None).
    """
    enc = str(((objective or {}).get("labels") or {}).get("encoding") or "")
    if enc == "dual_binary":
        from src.models.dual_binary import dual_binary_fire
        return dual_binary_fire(objective)
    if report_only:
        return None
    return primary_tau(objective)
