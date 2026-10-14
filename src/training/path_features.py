"""Causal ATR-normalised features from a 1-second OHLC(+volume) path.

Encoder A/B (ADR-053 recovered spec) read ``parquet_path1s``, not the 5s engineered
matrix. Dimensional quantities are divided by a **causal** 10-minute ATR so 2011
and 2026 sit on the same scale. No VIX: it is missing for most of the archive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.session_path import prepare_path

TICK = 0.25
ATR_SECONDS = 600  # 10-minute trailing true-range mean
ATR_MIN_PERIODS = 60
VOL_Z_SECONDS = 600

PATH_FEATURE_COLS: tuple[str, ...] = (
    "log_ret",
    "range_atr",
    "body_atr",
    "close_loc",
    "vol_z",
    "log_atr",
)


def session_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Contiguous 1s grid with volume=0 on silent seconds (no trade ⇒ no flow)."""
    base = prepare_path(raw)
    vol = None
    if "volume" in raw.columns:
        v = raw.copy()
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("path must be indexed by a DatetimeIndex")
        if v.index.tz is None:
            v.index = v.index.tz_localize("UTC")
        else:
            v.index = v.index.tz_convert("UTC")
        vol = v["volume"].astype("float64")
        vol = vol.loc[~vol.index.duplicated(keep="last")].sort_index()
    if "open" in raw.columns:
        o = raw["open"].astype("float64")
        if getattr(o.index, "tz", None) is None:
            o.index = pd.DatetimeIndex(o.index, tz="UTC")
        else:
            o.index = o.index.tz_convert("UTC")
        o = o.loc[~o.index.duplicated(keep="last")].sort_index()
    else:
        o = base["close"]
    grid = pd.date_range(base.index[0], base.index[-1], freq="1s", tz="UTC")
    out = base.reindex(grid)
    out["close"] = out["close"].ffill()
    filled = out["high"].isna()
    out.loc[filled, "high"] = out.loc[filled, "close"]
    out.loc[filled, "low"] = out.loc[filled, "close"]
    opn = o.reindex(grid)
    out["open"] = opn.fillna(out["close"])
    if vol is not None:
        out["volume"] = vol.reindex(grid).fillna(0.0)
    else:
        out["volume"] = 0.0
    return out


def path_ohlc_features(
    raw: pd.DataFrame,
    *,
    warmup: pd.DataFrame | None = None,
    atr_seconds: int = ATR_SECONDS,
    tick: float = TICK,
) -> pd.DataFrame:
    """Return ``PATH_FEATURE_COLS`` on the session 1s grid.

    *warmup* is earlier OHLC (strictly before *raw*) from the continuous
    series — overnight seconds or the tail of a prior session. ATR / vol_z
    roll across that prefix so the session open is not left blank. The
    returned index is the session grid only. With no warmup, leading ATR
    rows stay NaN until ``ATR_MIN_PERIODS`` prior seconds exist.
    """
    session_g = session_ohlcv(raw)
    if warmup is None or warmup.empty:
        return _features_from_grid(session_g, atr_seconds=atr_seconds, tick=tick)
    warm_g = session_ohlcv(warmup)
    warm_g = warm_g.loc[warm_g.index < session_g.index[0]]
    if warm_g.empty:
        return _features_from_grid(session_g, atr_seconds=atr_seconds, tick=tick)
    # Row-stitch only. Do not date_range-fill a multi-hour hole between the
    # last warmup second and the session open — that would invent bars.
    need = int(atr_seconds) + int(ATR_MIN_PERIODS)
    stitched = pd.concat([warm_g.iloc[-need:], session_g])
    feats = _features_from_grid(stitched, atr_seconds=atr_seconds, tick=tick)
    return feats.reindex(session_g.index)


def _features_from_grid(
    g: pd.DataFrame,
    *,
    atr_seconds: int = ATR_SECONDS,
    tick: float = TICK,
) -> pd.DataFrame:
    """ATR-normalised columns on an already-built 1s grid."""
    close = g["close"].to_numpy(dtype="float64")
    high = g["high"].to_numpy(dtype="float64")
    low = g["low"].to_numpy(dtype="float64")
    opn = g["open"].to_numpy(dtype="float64")
    volume = g["volume"].to_numpy(dtype="float64")
    prev_c = np.empty_like(close)
    prev_c[0] = close[0]
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = _causal_mean(tr, atr_seconds, ATR_MIN_PERIODS)
    atr = np.maximum(atr, tick)
    log_close = np.log(np.maximum(close, tick))
    log_ret = np.empty_like(close)
    log_ret[0] = 0.0
    log_ret[1:] = log_close[1:] - log_close[:-1]
    span = np.maximum(high - low, tick)
    log_vol = np.log1p(np.maximum(volume, 0.0))
    vol_mu = _causal_mean(log_vol, VOL_Z_SECONDS, ATR_MIN_PERIODS)
    vol_sd = _causal_std(log_vol, VOL_Z_SECONDS, ATR_MIN_PERIODS)
    vol_sd = np.maximum(vol_sd, 1e-6)
    feats = pd.DataFrame({
        "log_ret": log_ret,
        "range_atr": tr / atr,
        "body_atr": (close - opn) / atr,
        "close_loc": (close - low) / span,
        "vol_z": (log_vol - vol_mu) / vol_sd,
        "log_atr": np.log(atr / tick),
    }, index=g.index)
    return feats


def _causal_mean(x: np.ndarray, win: int, min_periods: int) -> np.ndarray:
    """Trailing mean using values **strictly before** t (shift 1 after rolling)."""
    s = pd.Series(x)
    m = s.rolling(win, min_periods=min_periods).mean().shift(1)
    return m.to_numpy(dtype="float64")


def _causal_std(x: np.ndarray, win: int, min_periods: int) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(win, min_periods=min_periods).std().shift(1)
    return m.to_numpy(dtype="float64")
