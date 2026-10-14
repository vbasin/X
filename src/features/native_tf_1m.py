"""Native 1-minute feature store (ADR-085).

One row per closed Globex 1-minute bar (``bar_end_ts``). Entire 5s-catalog
groups: ``so_`` ``cs_`` ``ps_`` ``vw_`` ``rv_`` ``vp_`` ``ms_``.
Options ``iv_*`` is not on this hive (operator will add selected IV later).

``cs_open`` / ``cs_close`` / … are the **1-minute** candle (same names as the
5s catalog / 15s hive). Rolling windows that the 5s catalog expresses in
minutes are wall-clock on this grid (``rv_realized_vol_60`` = 5 min of 1m
log returns). ``so_`` / 1m ``cs_*`` / ``ps_1m_*`` reuse the existing 1-minute
modules (same values as the 5s parquet at each minute close).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from src.common.cme_holidays import is_trading_day
from src.common.contract_calendar import active_contract
from src.common.market_session import market_session_date
from src.features.bar_features import pandas_freq, timestep_delta
from src.features.native_tf import (
    globex_session_window,
    load_tbbo,
)

logger = logging.getLogger("vtech.features.native_tf_1m")

TIMESTEP = "1m"
BAR_DELTA = pd.Timedelta(minutes=1)

SO_COLS: tuple[str, ...] = (
    "so_fast_d_9_3",
    "so_fast_d_14_3",
    "so_fast_d_40_4",
    "so_full_d_60_10",
    "so_sync_score_all",
    "so_sync_score_9_60",
)
CS_COLS: tuple[str, ...] = (
    "cs_open",
    "cs_high",
    "cs_low",
    "cs_close",
    "cs_volume",
    "cs_range",
    "cs_direction",
    "cs_return",
    "cs_body_range_ratio",
    "cs_body_direction",
    "cs_clv",
    "cs_clv_ma_4",
    "cs_clv_ma_6",
    "cs_clv_ma_10",
    "cs_clv_ma_20",
    "cs_upper_shadow",
    "cs_lower_shadow",
    "cs_range_ma_6",
    "cs_range_ma_12",
    "cs_range_ma_60",
    "cs_range_ma_120",
    "cs_vol_regime",
    "cs_range_chg",
    "cs_15s_clv",
    "cs_30s_clv",
    "cs_5m_clv",
    "cs_10m_clv",
    "cs_15m_body_range",
    "cs_15m_clv",
    "cs_30m_body_range",
    "cs_30m_clv",
    "cs_atr_5m",
    "cs_atr_10m",
)
PS_COLS: tuple[str, ...] = (
    "ps_1m_next_dist_atr",
    "ps_1m_trend",
    "ps_lh_dist_sar",
    "ps_flip",
)
VW_COLS: tuple[str, ...] = (
    "vw_vwap_dist",
    "vw_vwap_zscore_60",
    "vw_vwap_zscore_120",
    "vw_vwap_slope_24",
    "vw_vwap_slope_120",
)
RV_COLS: tuple[str, ...] = (
    "rv_realized_vol_60",
    "rv_realized_vol_120",
    "rv_realized_vol_360",
    "rv_vol_of_vol_120",
    "rv_vol_of_vol_360",
    "rv_vol_ratio",
)
VP_COLS: tuple[str, ...] = (
    "vp_poc_dist",
    "vp_va_high_dist",
    "vp_va_low_dist",
    "vp_in_value_area",
    "vp_poc_slope",
    "vp_yday_poc_dist",
    "vp_yday_va_high_dist",
    "vp_yday_va_low_dist",
    "vp_yday_in_value_area",
    "vp_yday_poc_slope",
)
MS_COLS: tuple[str, ...] = (
    "ms_vpin",
    "ms_kyle_lambda_60",
    "ms_vpin_zscore_60",
    "ms_vpin_zscore_120",
    "ms_amihud_60",
    "ms_amihud_120",
    "ms_amihud_zscore_120",
)
FEATURE_1M_COLS: tuple[str, ...] = (
    *SO_COLS, *CS_COLS, *PS_COLS,
    *VW_COLS, *RV_COLS, *VP_COLS, *MS_COLS,
)


def _utc_idx(idx) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(idx)
    if out.tz is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _shift_to_bar_end(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    out.index = _utc_idx(out.index) + BAR_DELTA
    return out


def candles_1m_from_tbbo(tbbo: pd.DataFrame) -> pd.DataFrame:
    """Left-labeled 1-minute OHLCV. Empty minutes are dropped."""
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if tbbo is None or tbbo.empty or "price" not in tbbo.columns:
        return empty
    ts = pd.to_datetime(tbbo["ts_event"], utc=True)
    px = pd.Series(pd.to_numeric(tbbo["price"], errors="coerce").to_numpy(), index=ts)
    px = px.sort_index()
    px = px[np.isfinite(px.to_numpy())]
    if px.empty:
        return empty
    ohlc = px.resample("1min").ohlc()
    if "size" in tbbo.columns:
        sz = pd.Series(
            pd.to_numeric(tbbo["size"], errors="coerce").to_numpy(dtype="float64"),
            index=ts,
        ).sort_index()
        ohlc["volume"] = sz.resample("1min").sum()
    else:
        ohlc["volume"] = px.resample("1min").count().astype("float64")
    return ohlc.dropna(subset=["close"])


def _cs_ohlcv(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    rng = df["high"] - df["low"]
    out = pd.DataFrame(index=df.index)
    out["cs_open"] = df["open"]
    out["cs_high"] = df["high"]
    out["cs_low"] = df["low"]
    out["cs_close"] = df["close"]
    out["cs_volume"] = df["volume"] if "volume" in df.columns else np.nan
    out["cs_range"] = rng
    out["cs_direction"] = np.sign(df["close"] - df["open"])
    out["cs_return"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    return out.astype("float32")


def _vwap_1m(tbbo: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(VW_COLS), dtype="float32")
    if tbbo is None or tbbo.empty:
        return out
    df = tbbo.sort_values("ts_event").copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("ts_event")
    df["pv"] = df["price"] * df["size"]
    df["session"] = df.index.map(market_session_date)
    cum_pv = df.groupby("session", sort=False)["pv"].cumsum()
    cum_vol = df.groupby("session", sort=False)["size"].cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol
    left = df.resample("1min").agg(vwap=("vwap", "last"), close=("price", "last"))
    left = _shift_to_bar_end(left).reindex(bar_end)
    left["vwap"] = left["vwap"].ffill()
    left["close"] = left["close"].ffill()
    vwap = left["vwap"].replace(0, np.nan)
    dist = (left["close"] - vwap) / vwap
    out["vw_vwap_dist"] = dist
    std5 = dist.rolling("5min").std().replace(0, np.nan)
    std10 = dist.rolling("10min").std().replace(0, np.nan)
    out["vw_vwap_zscore_60"] = dist / std5
    out["vw_vwap_zscore_120"] = dist / std10
    v2 = vwap.shift(freq="2min")
    v10 = vwap.shift(freq="10min")
    out["vw_vwap_slope_24"] = (vwap - v2) / v2.replace(0, np.nan)
    out["vw_vwap_slope_120"] = (vwap - v10) / v10.replace(0, np.nan)
    return out.astype("float32")


def _rv_1m(close: pd.Series, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(RV_COLS), dtype="float32")
    px = pd.to_numeric(close, errors="coerce").reindex(bar_end)
    log_ret = np.log(px / px.shift(1))
    rv5 = log_ret.rolling("5min").std()
    rv10 = log_ret.rolling("10min").std()
    rv30 = log_ret.rolling("30min").std()
    out["rv_realized_vol_60"] = rv5
    out["rv_realized_vol_120"] = rv10
    out["rv_realized_vol_360"] = rv30
    out["rv_vol_of_vol_120"] = rv5.rolling("10min").std()
    out["rv_vol_of_vol_360"] = rv5.rolling("30min").std()
    out["rv_vol_ratio"] = rv5 / rv30.replace(0, np.nan)
    return out.astype("float32")


def _ms_1m(tbbo: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(MS_COLS), dtype="float32")
    if tbbo is None or tbbo.empty:
        return out
    from src.features.microstructure import compute as ms_compute
    raw = ms_compute(tbbo, timestep="1min")
    if raw is not None and not raw.empty and "ms_vpin" in raw.columns:
        vpin = _shift_to_bar_end(raw[["ms_vpin"]]).reindex(bar_end)["ms_vpin"]
        out["ms_vpin"] = vpin
        for col, win in (("ms_vpin_zscore_60", "5min"), ("ms_vpin_zscore_120", "10min")):
            mu = vpin.rolling(win).mean()
            sd = vpin.rolling(win).std().replace(0, np.nan)
            out[col] = (vpin - mu) / sd
    df = tbbo.sort_values("ts_event").copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    signed = np.where(df["side"].to_numpy() == "B", df["size"], -df["size"])
    df = df.set_index("ts_event")
    df["signed"] = signed
    df["dollar"] = df["price"] * df["size"]
    bar = df.resample("1min").agg(
        first=("price", "first"),
        last=("price", "last"),
        signed=("signed", "sum"),
        dollar=("dollar", "sum"),
    )
    bar = _shift_to_bar_end(bar).reindex(bar_end)
    ret = bar["last"] / bar["first"].replace(0, np.nan) - 1.0
    sv = np.sign(bar["signed"]) * np.sqrt(bar["signed"].abs())
    cov = ret.rolling("5min").cov(sv)
    var = sv.rolling("5min").var().replace(0, np.nan)
    out["ms_kyle_lambda_60"] = cov / var
    amihud = ret.abs() / bar["dollar"].replace(0, np.nan)
    a5 = amihud.rolling("5min").mean()
    a10 = amihud.rolling("10min").mean()
    out["ms_amihud_60"] = a5
    out["ms_amihud_120"] = a10
    sd = a5.rolling("10min").std().replace(0, np.nan)
    out["ms_amihud_zscore_120"] = (a5 - a5.rolling("10min").mean()) / sd
    return out.astype("float32")


def _vp_window(
    ts: np.ndarray,
    ticks: np.ndarray,
    sz: np.ndarray,
    t0: np.datetime64,
    t1: np.datetime64,
    tick_size: float,
    va_pct: float,
) -> tuple[float, float, float, float]:
    a = int(np.searchsorted(ts, t0, side="right"))
    b = int(np.searchsorted(ts, t1, side="right"))
    if b <= a:
        return (np.nan, np.nan, np.nan, np.nan)
    w_ticks = ticks[a:b]
    w_sz = sz[a:b]
    ok = np.isfinite(w_ticks) & np.isfinite(w_sz) & (w_sz > 0)
    if not ok.any():
        return (np.nan, np.nan, np.nan, np.nan)
    uniq, inv = np.unique(w_ticks[ok], return_inverse=True)
    vol = np.bincount(inv, weights=w_sz[ok])
    poc = float(uniq[int(vol.argmax())])
    total = float(vol.sum())
    if total <= 0:
        return (poc, poc, poc, np.nan)
    order = np.argsort(vol)[::-1]
    cum = np.cumsum(vol[order])
    keep = cum <= total * va_pct
    if not keep.any():
        return (poc, poc, poc, poc)
    va = uniq[order[keep]]
    return (poc, float(va.max()), float(va.min()), poc)


def _vp_1m(
    tbbo: pd.DataFrame,
    prev_tbbo: pd.DataFrame | None,
    bar_end: pd.DatetimeIndex,
    close: pd.Series,
    tick_size: float = 0.25,
    va_pct: float = 0.70,
) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(VP_COLS), dtype="float32")
    n = len(bar_end)
    if n == 0:
        return out
    poc = np.full(n, np.nan)
    va_h = np.full(n, np.nan)
    va_l = np.full(n, np.nan)
    if tbbo is not None and not tbbo.empty:
        ts = pd.to_datetime(tbbo["ts_event"], utc=True).to_numpy(dtype="datetime64[ns]")
        px = pd.to_numeric(tbbo["price"], errors="coerce").to_numpy(dtype="float64")
        sz = pd.to_numeric(tbbo["size"], errors="coerce").to_numpy(dtype="float64")
        ticks = np.round(px / tick_size) * tick_size
        order = np.argsort(ts)
        ts, ticks, sz = ts[order], ticks[order], sz[order]
        be = bar_end.to_numpy(dtype="datetime64[ns]")
        lo = be - np.timedelta64(30, "m")
        for i in range(n):
            poc[i], va_h[i], va_l[i], _ = _vp_window(
                ts, ticks, sz, lo[i], be[i], tick_size, va_pct,
            )
    px = pd.to_numeric(close, errors="coerce").reindex(bar_end).to_numpy(dtype="float64")
    den = np.where(np.isfinite(px) & (px != 0.0), px, np.nan)
    out["vp_poc_dist"] = (px - poc) / den
    out["vp_va_high_dist"] = (px - va_h) / den
    out["vp_va_low_dist"] = (px - va_l) / den
    in_va = (px >= va_l) & (px <= va_h) & np.isfinite(va_l) & np.isfinite(va_h)
    out["vp_in_value_area"] = np.where(np.isfinite(va_l), in_va.astype("float64"), np.nan)
    poc_s = pd.Series(poc, index=bar_end)
    prev = poc_s.shift(freq="2min")
    out["vp_poc_slope"] = (poc_s - prev) / prev.replace(0, np.nan)

    from src.features.volume_profile import session_profile
    prof = session_profile(prev_tbbo if prev_tbbo is not None else pd.DataFrame())
    out["vp_yday_poc_dist"] = (px - prof["poc"]) / den
    out["vp_yday_va_high_dist"] = (px - prof["va_high"]) / den
    out["vp_yday_va_low_dist"] = (px - prof["va_low"]) / den
    y_in = (px >= prof["va_low"]) & (px <= prof["va_high"])
    out["vp_yday_in_value_area"] = np.where(
        prof["va_low"] == prof["va_low"], y_in.astype("float64"), np.nan,
    )
    out["vp_yday_poc_slope"] = prof["poc_slope"]
    return out.astype("float32")


def _ps_1m(candles: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(PS_COLS), dtype="float32")
    from src.features.psar import _lh_dist, closed_ohlc, sar_on_closed
    closed = closed_ohlc(candles, "1min")
    if closed.empty:
        return out
    raw = sar_on_closed(closed)
    raw = raw.reindex(bar_end)
    closed = closed.reindex(bar_end)
    out["ps_1m_next_dist_atr"] = raw["next_dist_atr"]
    out["ps_1m_trend"] = raw["trend"]
    prev_sar = raw["next_sar"].shift(1)
    prev_trend = raw["trend"].shift(1)
    low = closed["low"].to_numpy(dtype="float64")
    high = closed["high"].to_numpy(dtype="float64")
    dist = _lh_dist(low, high, prev_sar.to_numpy(dtype="float64"),
                    prev_trend.to_numpy(dtype="float64"))
    out["ps_lh_dist_sar"] = dist
    out["ps_flip"] = (np.isfinite(dist) & (dist < 0.0)).astype("float32")
    return out.astype("float32")


def compute_1m_features(
    tbbo: pd.DataFrame,
    *,
    prev_tbbo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Closed-bar 1m frame with ``bar_end_ts`` + :data:`FEATURE_1M_COLS`."""
    empty = pd.DataFrame(columns=["bar_end_ts", *FEATURE_1M_COLS])
    candles = candles_1m_from_tbbo(tbbo)
    if candles.empty:
        return empty

    from src.features import atr as atr_mod
    from src.features import candle_structure, stochastic

    so = _shift_to_bar_end(stochastic.compute(candles))
    cs_shape = _shift_to_bar_end(candle_structure.compute(candles))
    cs_ohlc = _shift_to_bar_end(_cs_ohlcv(candles))
    cs_mtf = _shift_to_bar_end(candle_structure.compute_multitimeframe(candles))
    bar_end = _utc_idx(cs_ohlc.index)
    cs_sub = candle_structure.compute_subminute_clv(
        tbbo, timestep="1min", grid=bar_end,
    )
    atr = atr_mod.compute(candles, bar_end)
    ps = _ps_1m(candles, bar_end)
    vw = _vwap_1m(tbbo, bar_end)
    rv = _rv_1m(cs_ohlc["cs_close"], bar_end)
    vp = _vp_1m(tbbo, prev_tbbo, bar_end, cs_ohlc["cs_close"])
    ms = _ms_1m(tbbo, bar_end)

    parts = [so, cs_shape, cs_ohlc, cs_mtf, cs_sub, atr, ps, vw, rv, vp, ms]
    frame = pd.DataFrame(index=bar_end)
    for p in parts:
        if p is None or p.empty:
            continue
        p = p.copy()
        p.index = _utc_idx(p.index)
        p = p.loc[~p.index.duplicated(keep="last")]
        for c in p.columns:
            frame[c] = p.reindex(bar_end)[c]
    for c in FEATURE_1M_COLS:
        if c not in frame.columns:
            frame[c] = np.nan
    frame = frame.loc[:, list(FEATURE_1M_COLS)].astype("float32")
    frame.insert(0, "bar_end_ts", frame.index)
    return frame.reset_index(drop=True)


def prev_session_date(session: date) -> date:
    d = session - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def build_session_frame(
    conn: psycopg.Connection,
    session: date,
    *,
    root: str = "NQ",
) -> pd.DataFrame:
    """One Globex session of native 1m features (no options IV)."""
    symbol = active_contract(root, session)
    start, end = globex_session_window(session)
    ticks = load_tbbo(conn, start, end, symbol)
    prev = prev_session_date(session)
    p0, p1 = globex_session_window(prev)
    prev_ticks = load_tbbo(conn, p0, p1, active_contract(root, prev))
    out = compute_1m_features(ticks, prev_tbbo=prev_ticks)
    if not out.empty:
        out["_meta_nq_contract"] = symbol
    return out


def session_job(
    payload: tuple[str, str, str],
) -> tuple[str, int, int, str, str]:
    """Worker: ``(session_iso, dsn, staging)``."""
    session_iso, dsn, staging = payload
    session = date.fromisoformat(session_iso)
    tmp = Path(staging) / f"session_{session_iso}.parquet"
    try:
        with psycopg.connect(dsn) as conn:
            conn.execute("SET statement_timeout = '180s'")
            frame = build_session_frame(conn, session)
            n_bars = int(len(frame))
            if frame.empty:
                if tmp.exists():
                    tmp.unlink()
                return session_iso, 0, 0, "", ""
            tmp.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(tmp, index=False, compression="zstd")
            return session_iso, 0, n_bars, str(tmp), ""
    except Exception as exc:  # noqa: BLE001
        return session_iso, 0, 0, "", f"{type(exc).__name__}: {exc}"


def timeframe_freq() -> str:
    return pandas_freq(TIMESTEP)


def bar_delta() -> pd.Timedelta:
    return timestep_delta(TIMESTEP)
