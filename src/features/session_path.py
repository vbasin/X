"""The 1-second price path and its per-session grid.

These are path shape and calendar primitives, not economics and not a
backtest: they normalise a price frame and lay one session on a contiguous
1-second grid. Label builders, feature builders and the portfolio simulator
all need them, which is why they had ended up imported as *private* names
(``portfolio_sim._session_grid``) from six different modules.

Moved here verbatim from ``src/backtest/portfolio_sim.py``; the behaviour is
unchanged. ``portfolio_sim`` now imports them from this module, so the
simulator owns trade sequencing and account state, and nothing has to reach
into it for path handling.
"""

from __future__ import annotations

import logging
from datetime import time as dtime

import pandas as pd

logger = logging.getLogger(__name__)


def parse_hhmm(s: str) -> dtime:
    """``"08:30"`` -> ``datetime.time(8, 30)``. Seconds, if present, are ignored."""
    h, m = (int(x) for x in str(s).split(":")[:2])
    return dtime(h, m)


def prepare_path(path: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a 1-second OHLC path frame (UTC DatetimeIndex).

    Accepts ``high``/``low`` or falls back to ``close`` only (in which case the caller
    must use ``tie_policy="close_only"`` — anything else would silently invent
    excursions). Duplicate seconds keep the last row.
    """
    if not isinstance(path.index, pd.DatetimeIndex):
        raise ValueError("path must be indexed by a DatetimeIndex")
    df = path.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if "close" not in df.columns:
        if "_close" in df.columns:
            df = df.rename(columns={"_close": "close"})
        else:
            raise ValueError("path requires a 'close' (or '_close') column")
    if "high" not in df.columns or "low" not in df.columns:
        logger.warning("path has no high/low columns — high=low=close; only "
                       "tie_policy='close_only' is meaningful on this input")
        df["high"] = df["close"]
        df["low"] = df["close"]
    keep = ["high", "low", "close"]
    return (df[keep].astype("float64")
            .loc[~df.index.duplicated(keep="last")]
            .sort_index())


def session_grid(day_path: pd.DataFrame) -> pd.DataFrame:
    """Reindex one session onto a contiguous 1s grid.

    A second with no exchange bar had no trade, so it contributes no excursion: close is
    forward-filled and ``high = low = close`` for the synthetic second.
    """
    grid = pd.date_range(day_path.index[0], day_path.index[-1], freq="1s", tz="UTC")
    out = day_path.reindex(grid)
    out["close"] = out["close"].ffill()
    filled = out["high"].isna()
    out.loc[filled, "high"] = out.loc[filled, "close"]
    out.loc[filled, "low"] = out.loc[filled, "close"]
    return out
