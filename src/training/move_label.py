"""Directional-move labels: is a move coming, independent of any bracket.

The first-passage label answers "did *this* target get hit before *this* stop",
which couples the label to a bracket choice. One underlying move therefore produces
hundreds of different labels depending on which target/stop pair happened to be
enumerated, and a model trained on realised P&L learns to prefer distant targets
because distance is worth points — conflating "will price move" with "how much is
the move worth".

This module answers the prior question only. For a decision timestamp and a side,
over a fixed horizon:

    y_move = 1  iff  MFE_side >= move_r * R  and  MAE_side < adverse_r * R

"price ran this far my way before it went that far against me". No target, no stop,
no bracket, no P&L. One row per ``(ts, side)``.

``t_move`` records when the favourable threshold was first reached, which is what
the uniqueness weighting in the evaluator needs to discount overlapping labels: at
a 5-second cadence a 30-minute move is claimed by 360 consecutive decisions, and
counting them as independent samples is how a t-statistic becomes fiction.

Excursions are measured on the **print tape**, so a move that happened inside one
second is visible and the order of two barrier touches is observed rather than
assigned by a tie policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.tape import first_cross, rmq, sparse_tables
from src.training import cascade_policy as policy

logger = logging.getLogger("vtech.training.move_label")

LONG, SHORT = 1, -1
SIDES = (LONG, SHORT)


@dataclass(frozen=True)
class MoveConfig:
    """Decision cadence, horizon and what counts as a move."""

    entry_start: str = "08:30"
    entry_end: str = "14:45"
    flatten: str = "15:00"
    decision_grid_seconds: int = policy.DECISION_GRID_SECONDS
    horizon_seconds: int = policy.HORIZON_SECONDS
    # Derived from the Stage 3 target policy so both stages describe one trade.
    move_r: float = policy.MOVE_R          # 0.25 R — the smallest target taken
    adverse_r: float = policy.ADVERSE_R    # 0.125 R — that target's own stop
    timezone: str = "America/Chicago"

    def __post_init__(self) -> None:
        if self.decision_grid_seconds < 1:
            raise ValueError("decision_grid_seconds must be >= 1")
        if self.horizon_seconds < 1:
            raise ValueError("horizon_seconds must be >= 1")
        if self.move_r <= 0 or self.adverse_r <= 0:
            raise ValueError("move_r and adverse_r must be positive")


def decision_times(tape_index: pd.DatetimeIndex, cfg: MoveConfig) -> pd.DatetimeIndex:
    """Decision timestamps on the configured grid inside the entry window."""
    tz = ZoneInfo(cfg.timezone)
    ct = tape_index.tz_convert(tz)
    day = ct[0].date()
    h0, m0 = (int(x) for x in cfg.entry_start.split(":"))
    h1, m1 = (int(x) for x in cfg.entry_end.split(":"))
    start = pd.Timestamp(day.year, day.month, day.day, h0, m0, tz=tz)
    end = pd.Timestamp(day.year, day.month, day.day, h1, m1, tz=tz)
    grid = pd.date_range(start, end, freq=f"{cfg.decision_grid_seconds}s", tz=tz)
    return grid.tz_convert("UTC")


def _sparse_tables(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated alias kept for callers importing from this module."""
    return sparse_tables(px)


def label_moves(tape: pd.DataFrame, cfg: MoveConfig, r_value: float) -> pd.DataFrame:
    """One row per ``(decision ts, side)`` with the directional-move label.

    Returns ``ts``, ``side``, ``y_move``, ``reached`` (favourable threshold cleared,
    ignoring adverse excursion), ``t_move`` (seconds to that threshold, ``-1`` if
    never reached), ``mfe_pts``, ``mae_pts``, ``price``.

    ``y_move`` is the trainable label; ``reached`` exists because measuring a stop
    against ``y_move`` is circular — that label already caps adverse excursion, so
    any stop wider than ``adverse_r`` survives by construction.

    The horizon is clipped at the flatten time, so a decision late in the session
    gets the shorter horizon it would really have. ``t_move`` comes from a
    vectorised binary search on the range-max table: the first print at which the
    running extreme since the decision clears the threshold.
    """
    if tape.empty or not np.isfinite(r_value) or r_value <= 0:
        return pd.DataFrame()

    ts_ns = tape.index.values.astype("int64")
    px = tape["price"].to_numpy(dtype="float64")
    decisions = decision_times(tape.index, cfg)
    if len(decisions) == 0:
        return pd.DataFrame()

    tz = ZoneInfo(cfg.timezone)
    day = tape.index.tz_convert(tz)[0].date()
    fh, fm = (int(x) for x in cfg.flatten.split(":"))
    flat_ns = (pd.Timestamp(day.year, day.month, day.day, fh, fm, tz=tz)
               .tz_convert("UTC").value)

    d_ns = decisions.values.astype("int64")
    start = np.searchsorted(ts_ns, d_ns, side="left")
    horizon_ns = np.minimum(d_ns + cfg.horizon_seconds * 1_000_000_000, flat_ns)
    stop = np.searchsorted(ts_ns, horizon_ns, side="right") - 1

    keep = (start <= stop) & (start < len(px))
    if not keep.any():
        return pd.DataFrame()
    start, stop, d_ns = start[keep], stop[keep], d_ns[keep]
    entry = px[start]

    hi_tbl, lo_tbl = sparse_tables(px)
    seg_hi = rmq(hi_tbl, start, stop, True).astype("float64")
    seg_lo = rmq(lo_tbl, start, stop, False).astype("float64")

    up_thr = cfg.move_r * r_value
    dn_thr = cfg.adverse_r * r_value

    rows = []
    for side in SIDES:
        if side == LONG:
            mfe = np.maximum(seg_hi - entry, 0.0)
            target = entry + up_thr
        else:
            mfe = np.maximum(entry - seg_lo, 0.0)
            target = entry - up_thr

        reached = mfe >= up_thr
        t_mv = np.full(start.size, -1, dtype="int64")
        idx_mv = stop.copy()
        tbl = hi_tbl if side == LONG else lo_tbl
        cross = first_cross(tbl, start, stop, target, side == LONG)
        got = cross >= 0
        if got.any():
            idx_mv[got] = cross[got]
            t_mv[got] = (ts_ns[cross[got]] - d_ns[got]) // 1_000_000_000

        # Adverse excursion is only disqualifying *before* the move completes: a
        # run that clears the threshold and later retraces is still a clean move.
        if side == LONG:
            mae = np.maximum(entry - rmq(lo_tbl, start, idx_mv, False).astype("float64"), 0.0)
        else:
            mae = np.maximum(rmq(hi_tbl, start, idx_mv, True).astype("float64") - entry, 0.0)

        y = (reached & (mae < dn_thr)).astype("int8")
        rows.append(pd.DataFrame({
            "ts": pd.to_datetime(d_ns, utc=True),
            "side": np.int8(side),
            "y_move": y,
            "reached": reached.astype("int8"),
            "t_move": t_mv,
            "mfe_pts": mfe,
            "mae_pts": mae,
            "price": entry,
            "r_value": r_value,
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["ts", "side"],
                                                          ignore_index=True)


def cash_slice(tape: pd.DataFrame, start: str, end: str,
               tz: str = "America/Chicago") -> pd.DataFrame:
    """Prints whose local time is in ``[start, end)``."""
    ct = tape.index.tz_convert(ZoneInfo(tz))
    h0, m0 = (int(x) for x in start.split(":"))
    h1, m1 = (int(x) for x in end.split(":"))
    return tape.loc[(ct.time >= dtime(h0, m0)) & (ct.time < dtime(h1, m1))]


def session_r_from_tape(tape: pd.DataFrame, block_seconds: int = 300) -> float:
    """R for a session: median range of non-overlapping print blocks.

    Same yardstick as the bar-based ``session_r``, measured on prints so that
    excursions inside a second are not invisible.
    """
    if tape.empty:
        return float("nan")
    blocks = tape.index.floor(f"{block_seconds}s")
    g = tape.groupby(blocks)["price"]
    rng = (g.max() - g.min()).to_numpy(dtype="float64")
    rng = rng[np.isfinite(rng) & (rng > 0)]
    return float(np.median(rng)) if rng.size else float("nan")
