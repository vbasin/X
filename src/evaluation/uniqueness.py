"""Sample uniqueness for overlapping labels (López de Prado, *AFML* ch. 4).

A label that spans a window of time is not an independent observation of that
window. At a 5-second decision cadence a 30-minute move is claimed by 360
consecutive decisions, all reading almost the same features and all resolved by the
same price path. Counting them as 360 samples inflates every standard error by
roughly :math:`\\sqrt{360}` and turns a t-statistic into a statement about the
sampling grid rather than about the market.

Two quantities fix it:

``concurrency`` — how many label windows are live at each instant.
``uniqueness``  — each label's average :math:`1/c_t` over its own window, in
``(0, 1]``. A label that never shares its window with another scores 1.

Uniqueness is used twice. As a **training weight** it stops the model spending its
capacity on whichever move happened to be sampled most often. As an **effective
sample size** :math:`n_{\\text{eff}} = \\sum_i u_i` it replaces the raw row count in
every t-statistic, which is the only version of that number worth reporting.

The implementation is a difference array over a second-resolution timeline, so cost
is linear in rows plus the session's span rather than quadratic in overlaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency(start_s: np.ndarray, end_s: np.ndarray) -> np.ndarray:
    """Number of live label windows at each second of a shared timeline.

    *start_s* and *end_s* are inclusive second offsets from a common origin.
    """
    if start_s.size == 0:
        return np.zeros(0, dtype="int64")
    horizon = int(end_s.max()) + 2
    diff = np.zeros(horizon, dtype="int64")
    np.add.at(diff, start_s, 1)
    np.add.at(diff, end_s + 1, -1)
    return np.cumsum(diff)


def uniqueness(start_s: np.ndarray, end_s: np.ndarray) -> np.ndarray:
    """Average ``1 / concurrency`` over each label's own window.

    Computed from the prefix sum of ``1 / c_t``, so every label is one subtraction
    rather than a scan of its window.
    """
    if start_s.size == 0:
        return np.zeros(0, dtype="float64")
    c = concurrency(start_s, end_s).astype("float64")
    inv = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
    prefix = np.concatenate(([0.0], np.cumsum(inv)))
    span = (end_s - start_s + 1).astype("float64")
    return (prefix[end_s + 1] - prefix[start_s]) / np.maximum(span, 1.0)


def label_uniqueness(df: pd.DataFrame, *, ts_col: str = "ts",
                     span_col: str = "t_move", horizon: int = 1800,
                     group_col: str | None = "session") -> np.ndarray:
    """Uniqueness per row, computed within each *group_col* independently.

    Labels in different sessions cannot overlap, so grouping keeps the timeline
    short and the result exact. A row whose span is missing or negative (the move
    never came) is treated as spanning the full *horizon* — it did occupy the
    window, whatever the outcome.
    """
    out = np.ones(len(df), dtype="float64")
    groups = [(None, df)] if group_col is None else df.groupby(group_col, sort=False)
    for _, g in groups:
        ts = pd.to_datetime(g[ts_col], utc=True)
        origin = ts.min()
        start_s = ((ts - origin).dt.total_seconds()).to_numpy().astype("int64")
        span = g[span_col].to_numpy(dtype="float64")
        span = np.where(np.isfinite(span) & (span >= 0), span, horizon)
        end_s = start_s + span.astype("int64")
        out[df.index.get_indexer(g.index)] = uniqueness(start_s, end_s)
    return out


def effective_n(weights: np.ndarray) -> float:
    """Effective sample size — the sum of uniqueness weights."""
    w = np.asarray(weights, dtype="float64")
    return float(w[np.isfinite(w)].sum())


def weighted_tstat(x: np.ndarray, weights: np.ndarray | None = None) -> float:
    """t-statistic of the mean, with overlapping samples discounted.

    With uniform weights this is the ordinary one-sample t. With uniqueness
    weights the denominator uses :math:`n_{\\text{eff}}`, so a run of near-duplicate
    labels stops buying significance it did not earn.
    """
    x = np.asarray(x, dtype="float64")
    ok = np.isfinite(x)
    if weights is None:
        w = np.ones(x.size, dtype="float64")
    else:
        w = np.asarray(weights, dtype="float64")
        ok &= np.isfinite(w)
    x, w = x[ok], w[ok]
    if x.size < 2 or w.sum() <= 0:
        return float("nan")
    mean = float(np.average(x, weights=w))
    var = float(np.average((x - mean) ** 2, weights=w))
    n_eff = effective_n(w)
    if var <= 0 or n_eff <= 1:
        return float("nan")
    return mean / np.sqrt(var / n_eff)
