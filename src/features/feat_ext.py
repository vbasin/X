"""Assemble the v5.5 feature-extension columns for engine + parquet patch."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.bar_windows import get_timestep
from src.features import atr as atr_mod
from src.features import higher_timeframe
from src.features import stochastic
from src.features.stochastic import SO_COLS
from src.features import trade_location
from src.features import volume_profile
from src.features import vix as vix_mod
from src.features.candle_structure import align_to_grid

EXT_COLS = (
    "ht_ytd_high_dist", "ht_ytd_low_dist",
    "cs_atr_5m", "cs_atr_10m",
    "iv_rank_365", "iv_vix_close",
    "tl_trade_location", "tl_aggressive_pct",
    "tl_loc_ma_30s", "tl_loc_ma_5m", "tl_loc_ma_30m",
    "tl_aggr_30s", "tl_aggr_5m", "tl_aggr_30m",
    "tl_loc_diff_30s_5m", "tl_loc_diff_5m_30m",
    "tl_loc_zscore_30s_5m", "tl_loc_zscore_5m_30m",
    "tl_aggr_diff_30s_5m", "tl_aggr_diff_5m_30m",
    "tl_aggr_zscore_30s_5m", "tl_aggr_zscore_5m_30m",
    "vp_yday_poc_dist", "vp_yday_va_high_dist", "vp_yday_va_low_dist",
    "vp_yday_in_value_area", "vp_yday_poc_slope",
    *SO_COLS,
)

# 1m/60m MAs plus unused increment (series − ma) stats; patch drops these.
TL_RETIRED_COLS = (
    "tl_loc_ma_1m", "tl_loc_ma_60m", "tl_aggr_1m", "tl_aggr_60m",
    "tl_loc_diff_30s", "tl_loc_diff_5m", "tl_loc_diff_30m",
    "tl_loc_zscore_30s", "tl_loc_zscore_5m", "tl_loc_zscore_30m",
    "tl_aggr_diff_30s", "tl_aggr_diff_5m", "tl_aggr_diff_30m",
    "tl_aggr_zscore_30s", "tl_aggr_zscore_5m", "tl_aggr_zscore_30m",
)


def previous_session_date(d: date) -> date:
    """Previous weekday (includes early-close sessions; skips weekends)."""
    cand = d - timedelta(days=1)
    while cand.weekday() >= 5:
        cand -= timedelta(days=1)
    return cand


def write_iv_daily_cache(parquet_dir: Path, out_path: Path) -> pd.DataFrame:
    """Last finite ``iv_atm_mean`` per day file → small cache for IV rank."""
    rows = []
    for f in sorted(Path(parquet_dir).glob("features_*.parquet")):
        day = f.stem.replace("features_", "")
        try:
            s = pd.read_parquet(f, columns=["iv_atm_mean"])["iv_atm_mean"].dropna()
        except Exception:
            continue
        if s.empty:
            continue
        rows.append({"date": day, "iv": float(s.iloc[-1])})
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_parquet(out_path, index=False)
    else:
        df = pd.DataFrame(columns=["date", "iv"])
        df.to_parquet(out_path, index=False)
    return df


def build_ext_frame(
    *,
    grid: pd.DatetimeIndex,
    price: pd.Series,
    candles_1m: pd.DataFrame,
    tbbo: pd.DataFrame,
    prev_tbbo: pd.DataFrame,
    prior_hl: pd.DataFrame,
    vix_1m: pd.DataFrame,
    daily_iv: pd.DataFrame,
    session_date: date,
    timestep: str | None = None,
) -> pd.DataFrame:
    """All v5.5 extension columns aligned to *grid*."""
    timestep = timestep or get_timestep()
    grid = pd.DatetimeIndex(grid)
    grid = grid[~grid.duplicated(keep="last")]
    parts: list[pd.DataFrame] = []

    ytd = higher_timeframe.compute_ytd(prior_hl, candles_1m, price, grid)
    parts.append(ytd)

    parts.append(atr_mod.compute(candles_1m, grid))

    parts.append(vix_mod.compute_vix(vix_1m, grid))

    if tbbo is not None and not tbbo.empty:
        tl = trade_location.compute(tbbo, timestep)
        tl_keep = [
            c for c in tl.columns
            if c in ("tl_trade_location", "tl_aggressive_pct")
            or c.startswith(("tl_loc_", "tl_aggr_"))
        ]
        if tl_keep:
            parts.append(align_to_grid(tl[tl_keep], grid).ffill())

    if prev_tbbo is not None:
        parts.append(volume_profile.compute_prior_session(prev_tbbo, price, timestep))

    if candles_1m is not None and not candles_1m.empty:
        so = stochastic.compute(candles_1m)
        from src.features.closed_bars import align_closed_1m_to_grid
        parts.append(align_closed_1m_to_grid(so, grid))

    if not parts:
        out = pd.DataFrame(index=grid)
    else:
        out = parts[0]
        out = out.loc[~out.index.duplicated(keep="last")]
        for extra in parts[1:]:
            extra = extra.loc[~extra.index.duplicated(keep="last")]
            extra = extra.reindex(out.index)
            out = out.join(extra, how="left")
    for c in EXT_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out


def attach_iv_rank(frame: pd.DataFrame, iv_atm: pd.Series | None, daily_iv: pd.DataFrame,
                   session_date: date) -> pd.DataFrame:
    frame = frame.copy()
    if iv_atm is None or getattr(iv_atm, "empty", True):
        frame["iv_rank_365"] = np.nan
        return frame
    rank = vix_mod.compute_iv_rank(iv_atm.reindex(frame.index), daily_iv, session_date)
    frame["iv_rank_365"] = rank["iv_rank_365"]
    return frame
