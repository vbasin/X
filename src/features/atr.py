"""ATR on 5-minute and 10-minute bars, stale-forwarded to the 5s grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.candle_structure import align_to_grid


def _wilder_atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = ohlc["close"].shift(1)
    tr = pd.concat([
        (ohlc["high"] - ohlc["low"]).abs(),
        (ohlc["high"] - prev_c).abs(),
        (ohlc["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute(
    candles_1m: pd.DataFrame,
    grid: pd.DatetimeIndex,
    period: int = 14,
) -> pd.DataFrame:
    """Wilder ATR(14) on 5m and 10m OHLC; held until the next bucket close."""
    if candles_1m is None or candles_1m.empty:
        return pd.DataFrame(
            np.nan, index=grid, columns=["cs_atr_5m", "cs_atr_10m"],
        ).astype(np.float32)
    src = candles_1m[["high", "low", "close"]].copy()
    if src.index.tz is None:
        src.index = src.index.tz_localize("UTC")
    else:
        src.index = src.index.tz_convert("UTC")
    parts = {}
    for tf, col in (("5min", "cs_atr_5m"), ("10min", "cs_atr_10m")):
        ohlc = src.resample(tf, label="right").agg(
            {"high": "max", "low": "min", "close": "last"}
        ).dropna(subset=["close"])
        if ohlc.empty:
            parts[col] = pd.Series(np.nan, index=src.index[:0])
        else:
            parts[col] = _wilder_atr(ohlc, period=period)
    raw = pd.DataFrame(parts)
    return align_to_grid(raw, grid).ffill().astype(np.float32)
