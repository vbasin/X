"""Per-head target-geometry features (ADR-082).

``lg_target_v_atr`` / ``lg_target_v_close`` are the long profit target as a
fraction of 10-minute Wilder ATR and of last print. Short columns are the
signed opposite (``-threshold / ATR``, ``-threshold / close``). They are
*not* 1-bar microstructure; they tell each head how large +50 / −50 is in
the current regime.

Causal ATR: Wilder ATR(14) of 10-minute OHLC reconstructed from 15s
``cs_close`` / ``cs_range`` / ``cs_clv``, last *closed* 10m bar, ffilled.
Same identity as 5s ``cs_atr_10m`` but sourced from the native-15s candle,
not the 1-minute resample used by the 5s catalog.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.atr import _wilder_atr

HEAD_LONG_COLS: tuple[str, ...] = ("lg_target_v_atr", "lg_target_v_close")
HEAD_SHORT_COLS: tuple[str, ...] = ("st_target_v_atr", "st_target_v_close")
TARGET_GEOMETRY_COLS: tuple[str, ...] = HEAD_LONG_COLS + HEAD_SHORT_COLS

DEFAULT_MOVE_THRESHOLD = 50.0
ATR_PERIOD = 14
ATR_TF = "10min"


def high_low_from_cs(
    close: np.ndarray | pd.Series,
    rng: np.ndarray | pd.Series,
    clv: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover bar high/low from close, range, CLV.

    ``CLV = (2C − H − L) / (H − L)``, ``range = H − L`` ⇒
    ``H = C + 0.5·range·(1 − CLV)``, ``L = C − 0.5·range·(1 + CLV)``.
    Flat bars (range 0 / CLV NaN) → H = L = C.
    """
    c = np.asarray(close, dtype="float64")
    r = np.asarray(rng, dtype="float64")
    v = np.asarray(clv, dtype="float64")
    r = np.where(np.isfinite(r) & (r > 0.0), r, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)
    high = c + 0.5 * r * (1.0 - v)
    low = c - 0.5 * r * (1.0 + v)
    flat = r <= 0.0
    high = np.where(flat, c, high)
    low = np.where(flat, c, low)
    return high, low


def atr_10m_from_15s(df: pd.DataFrame) -> pd.Series:
    """Causal Wilder ATR(14) of 10-minute OHLC, ffilled onto *df*'s index."""
    idx = _utc_index(df)
    if "cs_close" not in df.columns:
        return pd.Series(np.nan, index=idx, dtype="float32")
    close = pd.to_numeric(df["cs_close"], errors="coerce")
    rng = pd.to_numeric(df["cs_range"], errors="coerce") if "cs_range" in df.columns else 0.0
    clv = pd.to_numeric(df["cs_clv"], errors="coerce") if "cs_clv" in df.columns else np.nan
    high, low = high_low_from_cs(close.to_numpy(), np.asarray(rng, dtype="float64"),
                                 np.asarray(clv, dtype="float64"))
    ohlc = pd.DataFrame(
        {"high": high, "low": low, "close": close.to_numpy()},
        index=idx,
    )
    ohlc = ohlc[np.isfinite(ohlc["close"].to_numpy())]
    if ohlc.empty:
        return pd.Series(np.nan, index=idx, dtype="float32")
    bucket = ohlc.resample(ATR_TF, label="right").agg(
        {"high": "max", "low": "min", "close": "last"},
    ).dropna(subset=["close"])
    if bucket.empty:
        return pd.Series(np.nan, index=idx, dtype="float32")
    atr = _wilder_atr(bucket, period=ATR_PERIOD)
    aligned = atr.reindex(idx, method="ffill")
    return aligned.astype("float32")


def compute_target_geometry(
    df: pd.DataFrame,
    *,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
) -> pd.DataFrame:
    """Four head-private columns; index matches *df*."""
    idx = _utc_index(df)
    thr = float(move_threshold)
    close = (
        pd.to_numeric(df["cs_close"], errors="coerce")
        if "cs_close" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    close = close.reindex(idx)
    atr = atr_10m_from_15s(df).reindex(idx)
    atr_ok = np.where(np.isfinite(atr.to_numpy()) & (atr.to_numpy() > 1e-6),
                      atr.to_numpy(), np.nan)
    close_ok = np.where(np.isfinite(close.to_numpy()) & (np.abs(close.to_numpy()) > 1e-6),
                        close.to_numpy(), np.nan)
    lg_atr = thr / atr_ok
    lg_px = thr / close_ok
    out = pd.DataFrame({
        "lg_target_v_atr": lg_atr,
        "lg_target_v_close": lg_px,
        "st_target_v_atr": -lg_atr,
        "st_target_v_close": -lg_px,
    }, index=idx)
    return out.astype("float32")


def attach_target_geometry(
    df: pd.DataFrame,
    *,
    move_threshold: float = DEFAULT_MOVE_THRESHOLD,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add the four columns if missing (or always when *overwrite*). Idempotent."""
    if df is None or df.empty:
        return df
    have = all(c in df.columns for c in TARGET_GEOMETRY_COLS)
    if have and not overwrite:
        return df
    geo = compute_target_geometry(df, move_threshold=move_threshold)
    out = df.copy()
    for c in TARGET_GEOMETRY_COLS:
        out[c] = geo[c].to_numpy()
    return out


def _utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if "bar_end_ts" in df.columns:
        idx = pd.to_datetime(df["bar_end_ts"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
    else:
        raise ValueError("target geometry needs bar_end_ts or a DatetimeIndex")
    return pd.DatetimeIndex(idx)
