"""5s average MBP-10 depth imbalance (``mbp_``).

Each book-changing MBP-10 row already carries the full top-10 book. Per event:

* ``bid_count`` = sum of ``bid_sz_00..09`` (NULL level → 0)
* ``ask_count`` = sum of ``ask_sz_00..09``
* ``imbalance_diff`` = bid_count − ask_count
* ``imbalance_log`` = ln(bid_count) − ln(ask_count) when both > 0, else NaN

The 5s parquet stores the **mean** of those event values over ``[T, T+5s)``.
``imbalance_log`` uses only finite logs (its own denominator). Empty bars are
NaN — they have no occurrences.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common.bar_windows import get_timestep

N_LEVELS = 10
MBP_COLS: tuple[str, ...] = (
    "mbp_bid_count",
    "mbp_ask_count",
    "mbp_imbalance_diff",
    "mbp_imbalance_log",
)
_BID_SZ = tuple(f"bid_sz_{i:02d}" for i in range(N_LEVELS))
_ASK_SZ = tuple(f"ask_sz_{i:02d}" for i in range(N_LEVELS))


def level_sum_sql(side: str) -> str:
    """COALESCE sum of 10 size levels. ``side`` is ``bid`` or ``ask`` (fixed)."""
    if side not in ("bid", "ask"):
        raise ValueError("side must be bid or ask")
    return " + ".join(f"COALESCE({side}_sz_{i:02d}, 0)::float8" for i in range(N_LEVELS))


def event_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event L0–L9 size sums and imbalance (index preserved)."""
    bid = _sum_levels(df, _BID_SZ)
    ask = _sum_levels(df, _ASK_SZ)
    log = np.full(len(bid), np.nan, dtype=np.float64)
    ok = (bid > 0) & (ask > 0)
    log[ok] = np.log(bid[ok]) - np.log(ask[ok])
    return pd.DataFrame(
        {
            "bid_count": bid,
            "ask_count": ask,
            "imbalance_diff": bid - ask,
            "imbalance_log": log,
        },
        index=df.index,
    )


def aggregate_timestep(
    events: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Mean of event-level depth columns; log mean skips NaN (own n)."""
    timestep = timestep or get_timestep()
    if events is None or events.empty:
        return pd.DataFrame(columns=["bid_count", "ask_count", "imbalance_diff", "imbalance_log"])
    src = events.copy()
    if not isinstance(src.index, pd.DatetimeIndex):
        raise ValueError("event_depth frame must have a DatetimeIndex")
    if src.index.tz is None:
        src.index = src.index.tz_localize("UTC")
    else:
        src.index = src.index.tz_convert("UTC")
    return src.resample(timestep).mean()


def compute(
    bars: pd.DataFrame,
    grid: pd.DatetimeIndex | None = None,
    *,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Map aggregated 5s bars onto the feature grid as ``mbp_*`` columns."""
    del timestep  # bars are already bucketed
    cols = list(MBP_COLS)
    if bars is None or bars.empty:
        if grid is None:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(index=grid, columns=cols, dtype=np.float32)

    out = bars.copy()
    if "ts" in out.columns:
        out = out.set_index("ts")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    rename = {
        "bid_count": "mbp_bid_count",
        "ask_count": "mbp_ask_count",
        "imbalance_diff": "mbp_imbalance_diff",
        "imbalance_log": "mbp_imbalance_log",
    }
    out = out.rename(columns=rename)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].astype(np.float32)
    if grid is not None:
        g = grid
        if g.tz is None:
            g = g.tz_localize("UTC")
        else:
            g = g.tz_convert("UTC")
        out = out.reindex(g)
    return out


def build_mbp_patch_frame(
    bars: pd.DataFrame,
    grid: pd.DatetimeIndex,
    *,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Align ``mbp_*`` columns to an existing day parquet index."""
    return compute(bars, grid=grid, timestep=timestep)


def _sum_levels(df: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    parts = []
    n = len(df)
    for c in cols:
        if c in df.columns:
            parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
        else:
            parts.append(np.zeros(n, dtype=np.float64))
    return np.sum(parts, axis=0)


def empty_frame(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    if index is None:
        return pd.DataFrame(columns=list(MBP_COLS))
    return pd.DataFrame(index=index, columns=list(MBP_COLS), dtype=np.float32)
