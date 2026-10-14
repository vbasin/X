"""VIX spot from public.tt_candles and 365-day IV rank of iv_atm_mean."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.features.candle_structure import align_to_grid


def compute_vix(vix_1m: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Last 1-minute VIX close, visible only after that minute ends."""
    out = pd.DataFrame(index=pd.DatetimeIndex(grid))
    if vix_1m is None or vix_1m.empty:
        out["iv_vix_close"] = np.nan
        return out.astype(np.float32)
    s = vix_1m.copy()
    if "open_dt_full" in s.columns:
        idx = pd.to_datetime(s["open_dt_full"], utc=True)
        close = pd.Series(s["close_price"].to_numpy(dtype="float64"), index=idx)
    elif isinstance(s.index, pd.DatetimeIndex):
        close = s.iloc[:, 0].astype("float64")
        close.index = (
            s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
        )
    else:
        out["iv_vix_close"] = np.nan
        return out.astype(np.float32)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    # Candle labeled T covers [T, T+1m); visible at T+1m.
    close.index = close.index + pd.Timedelta(minutes=1)
    out["iv_vix_close"] = close.reindex(out.index, method="ffill")
    return out.astype(np.float32)


def compute_iv_rank(
    iv_atm: pd.Series,
    daily_iv: pd.DataFrame,
    session_date: date,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """365-day IV rank of the current ``iv_atm_mean`` (0–100).

    Daily history is prior sessions only; today's running IV is included
    as the current value so the rank can update intraday without leaking
    future bars.
    """
    out = pd.DataFrame(index=iv_atm.index)
    if iv_atm is None or iv_atm.empty:
        out["iv_rank_365"] = np.nan
        return out.astype(np.float32)
    hist = daily_iv
    if hist is None or hist.empty:
        out["iv_rank_365"] = np.nan
        return out.astype(np.float32)
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"]).dt.date
    lo = session_date - timedelta(days=lookback_days)
    prior = h[(h["date"] >= lo) & (h["date"] < session_date)]["iv"].astype("float64")
    prior = prior.dropna()
    iv = iv_atm.astype("float64")
    if prior.empty:
        # First days of the year / no history: rank is undefined.
        out["iv_rank_365"] = np.nan
        return out.astype(np.float32)
    mn = float(prior.min())
    mx = float(prior.max())
    # Include today's running IV in the range (standard IV-rank convention).
    cur_min = np.minimum(iv, mn)
    cur_max = np.maximum(iv, mx)
    width = (cur_max - cur_min).replace(0.0, np.nan)
    out["iv_rank_365"] = (100.0 * (iv - cur_min) / width).clip(0.0, 100.0)
    return out.astype(np.float32)
