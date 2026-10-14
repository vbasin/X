"""Closed-bar visibility for slower-than-5s features (ADR-041).

Left-labeled 1-minute candles (Timescale ``time_bucket`` / pandas default
``resample("1min")``) cover ``[T, T+1m)``. Their OHLC is not known until
``T+1m``. Opening-range high/low is a *running* max/min of **closed** 1-min
candles inside the RTH window, frozen after the window ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.candle_structure import align_to_grid

OR_DURATIONS: tuple[int, ...] = (5, 15, 30)


def shift_closed_1m(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Move a left-labeled 1-min index so values first appear at bar close."""
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx + pd.Timedelta(minutes=1)
    return out


def align_closed_1m_to_grid(
    df: pd.DataFrame,
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Shift 1-min features +1m, then stale-forward onto *grid*."""
    return align_to_grid(shift_closed_1m(df), grid).ffill()


def drop_incomplete_1m(
    candles: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Drop the last 1-min bucket unless *as_of* is at or past its close.

    ``as_of`` defaults to the last index timestamp in *candles* (live: last print
    should be passed explicitly so a left-labeled forming bucket is dropped).
    """
    if candles is None or candles.empty:
        return candles
    idx = candles.index
    if not isinstance(idx, pd.DatetimeIndex):
        return candles
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        candles = candles.copy()
        candles.index = idx
    else:
        candles = candles.copy()
        candles.index = idx.tz_convert("UTC")
        idx = candles.index
    if as_of is None:
        as_of = idx.max()
    else:
        as_of = pd.Timestamp(as_of)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")
    closed = candles.index + pd.Timedelta(minutes=1) <= as_of
    return candles.loc[closed]


def opening_range_running(
    candles: pd.DataFrame,
    grid: pd.DatetimeIndex,
    *,
    market_open: str = "08:30",
    timezone: str = "America/Chicago",
    durations: tuple[int, ...] = OR_DURATIONS,
) -> pd.DataFrame:
    """Running OR high/low from closed 1-min candles; NaN before RTH open.

    A 1-min bar stamped ``T`` (covering ``[T, T+1m)``) contributes only when
    ``t >= T+1m``. After ``open + duration`` the last completed value is held.
    """
    grid = pd.DatetimeIndex(grid)
    if grid.tz is None:
        grid = grid.tz_localize("UTC")
    else:
        grid = grid.tz_convert("UTC")
    cols = [f"tc_or{d}_{side}" for d in durations for side in ("high", "low")]
    out = pd.DataFrame(np.nan, index=grid, columns=cols, dtype="float64")
    if candles is None or candles.empty or "high" not in candles.columns:
        return out.astype(np.float32)

    c = candles.copy()
    if not isinstance(c.index, pd.DatetimeIndex):
        return out.astype(np.float32)
    if c.index.tz is None:
        c.index = c.index.tz_localize("UTC")
    else:
        c.index = c.index.tz_convert("UTC")
    c = c.loc[~c.index.duplicated(keep="last")].sort_index()
    c_ct = c.copy()
    c_ct.index = c_ct.index.tz_convert(timezone)
    grid_ct = grid.tz_convert(timezone)

    open_h, open_m = (int(x) for x in market_open.split(":"))
    day_dates = np.array([t.date() for t in grid_ct])
    for day in sorted(set(day_dates.tolist())):
        open_ct = pd.Timestamp(
            year=day.year, month=day.month, day=day.day,
            hour=open_h, minute=open_m, tz=timezone,
        )
        gmask = day_dates == day
        sub_grid = grid[gmask]
        if len(sub_grid) == 0:
            continue
        for dur in durations:
            win_end_ct = open_ct + pd.Timedelta(minutes=int(dur))
            in_win = np.asarray((c_ct.index >= open_ct) & (c_ct.index < win_end_ct))
            piece = c.iloc[np.flatnonzero(in_win)]
            if piece.empty:
                continue
            vis = piece.copy()
            vis.index = vis.index + pd.Timedelta(minutes=1)
            hi = vis["high"].astype("float64").cummax()
            lo = vis["low"].astype("float64").cummin()
            out.loc[sub_grid, f"tc_or{dur}_high"] = hi.reindex(sub_grid, method="ffill").to_numpy()
            out.loc[sub_grid, f"tc_or{dur}_low"] = lo.reindex(sub_grid, method="ffill").to_numpy()
    return out.astype(np.float32)
