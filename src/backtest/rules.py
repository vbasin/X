"""Safe rule DSL for user-defined strategies — structured conditions, never code.

A *rule block* is ``{"all": [...]}`` (AND) or ``{"any": [...]}`` (OR) whose items are
*conditions* or nested blocks. A condition compares one feature column against a constant
(``value``) or another feature column (``feature_rhs``)::

    {feature: vw_vwap_dist, op: lte, value: -20}
    {feature: of_delta,     op: cross_above, value: 0}
    {feature: of_volume_ma_60, op: gt, feature_rhs: rv_realized_vol_60}

Everything is evaluated with vectorised pandas boolean algebra — there is **no eval/exec
and no user-supplied code path** (the spec is data, validated against
``config/strategy_spec.schema.json``). Cross ops fire only on the crossing bar and use the
prior bar *within the same trading day* (day boundaries never produce phantom crosses).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_OPS = ("gt", "lt", "gte", "lte", "eq", "ne", "between",
        "cross_above", "cross_below", "abs_gt", "abs_lt")


def rule_features(block: dict[str, Any] | None) -> set[str]:
    """Every feature column referenced anywhere in *block* (for availability checks)."""
    out: set[str] = set()
    if not block:
        return out
    items = block.get("all") or block.get("any") or []
    for item in items:
        if "feature" in item:                       # condition
            out.add(item["feature"])
            if item.get("feature_rhs"):
                out.add(item["feature_rhs"])
        else:                                       # nested block
            out |= rule_features(item)
    return out


def eval_condition(
    df: pd.DataFrame,
    cond: dict[str, Any],
    *,
    day_id: np.ndarray | None = None,
) -> pd.Series:
    """Evaluate one condition to a boolean Series aligned with *df*.

    ``day_id`` (int array, one id per row) guards cross ops at session boundaries:
    the first bar of each day can never "cross" against the prior day's close.
    """
    feat = cond["feature"]
    op = cond["op"]
    if feat not in df.columns:
        raise KeyError(f"rule references unknown feature column '{feat}'")
    lhs = df[feat]

    if cond.get("feature_rhs") is not None:
        rhs_col = cond["feature_rhs"]
        if rhs_col not in df.columns:
            raise KeyError(f"rule references unknown feature column '{rhs_col}'")
        rhs: pd.Series | float = df[rhs_col]
    else:
        rhs = cond.get("value")

    if op == "gt":
        return lhs > rhs
    if op == "lt":
        return lhs < rhs
    if op == "gte":
        return lhs >= rhs
    if op == "lte":
        return lhs <= rhs
    if op == "eq":
        return lhs == rhs
    if op == "ne":
        return lhs != rhs
    if op == "between":
        lo, hi = rhs  # schema guarantees a 2-element array
        return (lhs >= lo) & (lhs <= hi)
    if op == "abs_gt":
        return lhs.abs() > rhs
    if op == "abs_lt":
        return lhs.abs() < rhs
    if op in ("cross_above", "cross_below"):
        prev_lhs = lhs.shift(1)
        prev_rhs = rhs.shift(1) if isinstance(rhs, pd.Series) else rhs
        if op == "cross_above":
            crossed = (lhs >= rhs) & (prev_lhs < prev_rhs)
        else:
            crossed = (lhs <= rhs) & (prev_lhs > prev_rhs)
        crossed = crossed.fillna(False)
        if day_id is not None and len(day_id) == len(crossed):
            same_day = np.empty(len(day_id), dtype=bool)
            same_day[0] = False
            same_day[1:] = day_id[1:] == day_id[:-1]
            crossed &= same_day
        return crossed
    raise ValueError(f"unknown rule op '{op}' (allowed: {_OPS})")


def eval_rules(
    df: pd.DataFrame,
    block: dict[str, Any] | None,
    *,
    day_id: np.ndarray | None = None,
) -> pd.Series:
    """Evaluate a rule block to a boolean Series. ``None``/empty block => all-True."""
    if not block:
        return pd.Series(True, index=df.index)
    if "all" in block:
        items, combine = block["all"], np.logical_and
        acc = pd.Series(True, index=df.index)
    elif "any" in block:
        items, combine = block["any"], np.logical_or
        acc = pd.Series(False, index=df.index)
    else:
        raise ValueError(f"rule block must have exactly one of 'all'/'any': {block}")
    for item in items:
        sub = (eval_condition(df, item, day_id=day_id) if "feature" in item
               else eval_rules(df, item, day_id=day_id))
        acc = combine(acc, sub.fillna(False))
    return pd.Series(acc, index=df.index)
