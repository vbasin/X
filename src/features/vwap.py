"""VWAP deviation features computed from tick-level TBBO data.

v3: Uses bar_windows(). Added VWAP slope. Removed binary band features
    (redundant with continuous z-score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m, get_timestep
from src.common.market_session import market_session_date


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute VWAP and deviation features.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size.
    timestep : Output resample interval.
    """
    timestep = timestep or get_timestep()
    w2m  = bars_m(2)
    w5m  = bars_m(5)
    w10m = bars_m(10)

    df = tbbo.sort_values("ts_event").set_index("ts_event")
    df["pv"] = df["price"] * df["size"]

    # Cumulative VWAP — reset at the CME Globex session boundary via market_session_date()
    # (4:00 PM CT open, 3:00 PM CT close / settlement; DST-aware).
    # Note: _load_tbbo loads by UTC midnight window (≈19:00–19:00 CT/CDT), so the first
    # ~3 hours of each session (16:00–19:00 CT) are in the previous UTC day's load and
    # are absent here — accumulation starts from the load boundary (~19:00 CT), not 16:00.
    _idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    df["session"] = _idx.map(market_session_date)
    df["cum_pv"] = df.groupby("session")["pv"].cumsum()
    df["cum_vol"] = df.groupby("session")["size"].cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)

    # Resample to timestep
    bar_data = df.resample(timestep).agg(
        vwap=("vwap", "last"),
        close=("price", "last"),
        volume=("size", "sum"),
    )
    # Forward-fill empty bars: a 5s window with no ticks does not change the
    # cumulative VWAP or last traded price.  Without ffill, a single missing
    # bar causes rolling(w).std() (default min_periods=w) to propagate NaN
    # forward for the entire window — poisoning up to 120 output rows per gap.
    bar_data["vwap"] = bar_data["vwap"].ffill()
    bar_data["close"] = bar_data["close"].ffill()

    features = pd.DataFrame(index=bar_data.index)
    features["vw_vwap_dist"] = (bar_data["close"] - bar_data["vwap"]) / bar_data["vwap"].replace(0, np.nan)

    # Rolling z-score of VWAP distance
    for w in (w5m, w10m):
        std = features["vw_vwap_dist"].rolling(w).std().replace(0, np.nan)
        features[f"vw_vwap_zscore_{w}"] = features["vw_vwap_dist"] / std

    # ── VWAP slope (Gap #19) — d(VWAP)/dt ──
    for w in (w2m, w10m):
        features[f"vw_vwap_slope_{w}"] = bar_data["vwap"].diff(w) / bar_data["vwap"].shift(w).replace(0, np.nan)

    return features.astype(np.float32)
