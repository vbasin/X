"""Native 5-minute feature store (ADR-090).

One row per closed Globex 5-minute bar (``bar_end_ts``). Groups: ``cs_``
(OHLCV + Wilder ATR(5)), ``vp_``, ``vw_`` (session VWAP + 2σ bands),
``ps_`` (SAR on closed 5m), ``zz_`` (Thinkorswim ZigZagHighLow factory
params), ``vf_`` (5m wick/body footprint + candle POC), ``rv_``.

``cs_open`` / ``cs_close`` are the **5-minute** candle. Rolling windows
are native 5m bars (``rv_realized_vol_5`` = 5 closed 5m log returns).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from src.common.contract_calendar import active_contract
from src.features.bar_features import pandas_freq, timestep_delta
from src.features.native_tf import (
    globex_session_window,
    load_tbbo,
)
from src.features.native_tf_1m import _vp_window, prev_session_date
from src.features.volume_footprint import POC_EMA_SPAN, _REGION_SUFFIXES

logger = logging.getLogger("vtech.features.native_tf_5m")

TIMESTEP = "5m"
BAR_DELTA = pd.Timedelta(minutes=5)
BB_STDEV = 2.0

CS_COLS: tuple[str, ...] = (
    "cs_open",
    "cs_high",
    "cs_low",
    "cs_close",
    "cs_volume",
    "cs_range",
    "cs_atr_5",
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
VW_COLS: tuple[str, ...] = (
    "vw_vwap_dist",
    "vw_vwap_zscore_5",
    "vw_vwap_zscore_10",
    "vw_vwap_slope_2",
    "vw_vwap_slope_10",
    "vw_bb_upper_dist",
    "vw_bb_lower_dist",
)
PS_COLS: tuple[str, ...] = (
    "ps_5m_next_dist_atr",
    "ps_5m_trend",
    "ps_lh_dist_sar",
    "ps_flip",
)
ZZ_COLS: tuple[str, ...] = (
    "zz_signal",
    "zz_state",
    "zz_swing_high_dist",
    "zz_swing_low_dist",
    "zz_reversal_pts",
    "zz_reversal_frac",
    "zz_new_max",
    "zz_new_min",
)
VF_COLS: tuple[str, ...] = (
    *(f"vf_5m_{s}" for s in _REGION_SUFFIXES),
    "vf_5m_poc_loc",
    "vf_5m_poc_ema_5",
)
RV_COLS: tuple[str, ...] = (
    "rv_realized_vol_5",
    "rv_realized_vol_10",
    "rv_realized_vol_30",
    "rv_vol_of_vol_10",
    "rv_vol_of_vol_30",
    "rv_vol_ratio",
)
FEATURE_5M_COLS: tuple[str, ...] = (
    *CS_COLS, *VP_COLS, *VW_COLS, *PS_COLS,
    *ZZ_COLS, *VF_COLS, *RV_COLS,
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


def candles_5m_from_tbbo(tbbo: pd.DataFrame) -> pd.DataFrame:
    """Left-labeled 5-minute OHLCV. Empty buckets are dropped."""
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if tbbo is None or tbbo.empty or "price" not in tbbo.columns:
        return empty
    ts = pd.to_datetime(tbbo["ts_event"], utc=True)
    px = pd.Series(pd.to_numeric(tbbo["price"], errors="coerce").to_numpy(), index=ts)
    px = px.sort_index()
    px = px[np.isfinite(px.to_numpy())]
    if px.empty:
        return empty
    ohlc = px.resample("5min").ohlc()
    if "size" in tbbo.columns:
        sz = pd.Series(
            pd.to_numeric(tbbo["size"], errors="coerce").to_numpy(dtype="float64"),
            index=ts,
        ).sort_index()
        ohlc["volume"] = sz.resample("5min").sum()
    else:
        ohlc["volume"] = px.resample("5min").count().astype("float64")
    return ohlc.dropna(subset=["close"])


def _cs_ohlcv(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    out = pd.DataFrame(index=df.index)
    out["cs_open"] = df["open"]
    out["cs_high"] = df["high"]
    out["cs_low"] = df["low"]
    out["cs_close"] = df["close"]
    out["cs_volume"] = df["volume"] if "volume" in df.columns else np.nan
    out["cs_range"] = df["high"] - df["low"]
    return out.astype("float32")


def _atr_5(candles: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.Series:
    """Wilder ATR(5) on closed 5-minute OHLC (TOS / Pine ``ta.rma`` seed)."""
    from src.features.psar import _rma, _true_range

    closed = _shift_to_bar_end(candles[["high", "low", "close"]]).reindex(bar_end)
    high = closed["high"].to_numpy(dtype="float64")
    low = closed["low"].to_numpy(dtype="float64")
    close = closed["close"].to_numpy(dtype="float64")
    atr = _rma(_true_range(high, low, close), 5)
    return pd.Series(atr, index=bar_end, name="cs_atr_5", dtype="float64")


def _vwap_5m(tbbo: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    """Session VWAP, bar-count z-score / slope, and VWAP ± 2σ bands."""
    from src.common.market_session import market_session_date

    out = pd.DataFrame(index=bar_end, columns=list(VW_COLS), dtype="float32")
    if tbbo is None or tbbo.empty:
        return out
    df = tbbo.sort_values("ts_event").copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("ts_event")
    px = pd.to_numeric(df["price"], errors="coerce")
    sz = pd.to_numeric(df["size"], errors="coerce")
    df["pv"] = px * sz
    df["p2v"] = (px ** 2) * sz
    df["session"] = df.index.map(market_session_date)
    cum_pv = df.groupby("session", sort=False)["pv"].cumsum()
    cum_p2v = df.groupby("session", sort=False)["p2v"].cumsum()
    cum_vol = df.groupby("session", sort=False)["size"].cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol
    var = (cum_p2v / cum_vol) - df["vwap"] ** 2
    df["vwap_std"] = np.sqrt(np.clip(var.to_numpy(dtype="float64"), 0.0, None))
    left = df.resample("5min").agg(
        vwap=("vwap", "last"),
        close=("price", "last"),
        vwap_std=("vwap_std", "last"),
    )
    left = _shift_to_bar_end(left).reindex(bar_end)
    left["vwap"] = left["vwap"].ffill()
    left["close"] = left["close"].ffill()
    left["vwap_std"] = left["vwap_std"].ffill()
    vwap = left["vwap"].replace(0, np.nan)
    dist = (left["close"] - vwap) / vwap
    std = left["vwap_std"]
    upper = vwap + BB_STDEV * std
    lower = vwap - BB_STDEV * std
    out["vw_vwap_dist"] = dist
    std5 = dist.rolling(5, min_periods=5).std().replace(0, np.nan)
    std10 = dist.rolling(10, min_periods=10).std().replace(0, np.nan)
    out["vw_vwap_zscore_5"] = dist / std5
    out["vw_vwap_zscore_10"] = dist / std10
    out["vw_vwap_slope_2"] = (vwap - vwap.shift(2)) / vwap.shift(2).replace(0, np.nan)
    out["vw_vwap_slope_10"] = (vwap - vwap.shift(10)) / vwap.shift(10).replace(0, np.nan)
    out["vw_bb_upper_dist"] = (left["close"] - upper) / vwap
    out["vw_bb_lower_dist"] = (left["close"] - lower) / vwap
    return out.astype("float32")


def _rv_5m(close: pd.Series, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(RV_COLS), dtype="float32")
    px = pd.to_numeric(close, errors="coerce").reindex(bar_end)
    log_ret = np.log(px / px.shift(1))
    rv5 = log_ret.rolling(5, min_periods=5).std()
    rv10 = log_ret.rolling(10, min_periods=10).std()
    rv30 = log_ret.rolling(30, min_periods=30).std()
    out["rv_realized_vol_5"] = rv5
    out["rv_realized_vol_10"] = rv10
    out["rv_realized_vol_30"] = rv30
    out["rv_vol_of_vol_10"] = rv5.rolling(10, min_periods=10).std()
    out["rv_vol_of_vol_30"] = rv5.rolling(30, min_periods=30).std()
    out["rv_vol_ratio"] = rv5 / rv30.replace(0, np.nan)
    return out.astype("float32")


def _vp_5m(
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
        px_t = pd.to_numeric(tbbo["price"], errors="coerce").to_numpy(dtype="float64")
        sz = pd.to_numeric(tbbo["size"], errors="coerce").to_numpy(dtype="float64")
        ticks = np.round(px_t / tick_size) * tick_size
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
    prev = poc_s.shift(2)
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


def _ps_5m(candles: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=bar_end, columns=list(PS_COLS), dtype="float32")
    from src.features.psar import _lh_dist, closed_ohlc, sar_on_closed
    closed = closed_ohlc(candles, "5min")
    if closed.empty:
        return out
    raw = sar_on_closed(closed)
    raw = raw.reindex(bar_end)
    closed = closed.reindex(bar_end)
    out["ps_5m_next_dist_atr"] = raw["next_dist_atr"]
    out["ps_5m_trend"] = raw["trend"]
    prev_sar = raw["next_sar"].shift(1)
    prev_trend = raw["trend"].shift(1)
    low = closed["low"].to_numpy(dtype="float64")
    high = closed["high"].to_numpy(dtype="float64")
    dist = _lh_dist(
        low, high,
        prev_sar.to_numpy(dtype="float64"),
        prev_trend.to_numpy(dtype="float64"),
    )
    out["ps_lh_dist_sar"] = dist
    out["ps_flip"] = (np.isfinite(dist) & (dist < 0.0)).astype("float32")
    return out.astype("float32")


def _zz_5m(candles: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    """Causal TOS ZigZagHighLow scalars on closed 5-minute OHLC (factory knobs)."""
    from src.external_models.catalog import factory_params
    from src.external_models.zigzag import zzhl_on_closed
    from src.features.psar import closed_ohlc

    out = pd.DataFrame(index=bar_end, columns=list(ZZ_COLS), dtype="float32")
    closed = closed_ohlc(candles, "5min")
    if closed.empty:
        return out
    p = factory_params("zigzag")
    raw = zzhl_on_closed(
        closed,
        percentage_reversal=float(p["percentage_reversal"]),
        absolute_reversal=float(p["absolute_reversal"]),
        atr_length=int(p["atr_length"]),
        atr_reversal=float(p["atr_reversal"]),
        tick_reversal=int(p["tick_reversal"]),
        tick_size=float(p["tick_size"]),
    )
    raw = raw.reindex(bar_end)
    close = closed.reindex(bar_end)["close"]
    den = close.replace(0, np.nan)
    out["zz_signal"] = raw["signal"]
    out["zz_state"] = raw["state"]
    out["zz_swing_high_dist"] = (close - raw["max_price_h"]) / den
    out["zz_swing_low_dist"] = (close - raw["min_price_l"]) / den
    need = pd.to_numeric(raw["reversal_pts"], errors="coerce")
    out["zz_reversal_pts"] = need
    sig = pd.to_numeric(raw["signal"], errors="coerce")
    frac = np.full(len(bar_end), np.nan, dtype="float64")
    need_a = need.to_numpy(dtype="float64")
    close_a = close.to_numpy(dtype="float64")
    max_h = pd.to_numeric(raw["max_price_h"], errors="coerce").to_numpy(dtype="float64")
    min_l = pd.to_numeric(raw["min_price_l"], errors="coerce").to_numpy(dtype="float64")
    sig_a = sig.to_numpy(dtype="float64")
    ok = np.isfinite(need_a) & (need_a != 0.0) & np.isfinite(close_a)
    up = ok & (sig_a > 0) & np.isfinite(max_h)
    dn = ok & (sig_a < 0) & np.isfinite(min_l)
    frac[up] = (max_h[up] - close_a[up]) / need_a[up]
    frac[dn] = (close_a[dn] - min_l[dn]) / need_a[dn]
    out["zz_reversal_frac"] = frac
    out["zz_new_max"] = raw["new_max"].astype("float64")
    out["zz_new_min"] = raw["new_min"].astype("float64")
    return out.astype("float32")


def _vf_5m(tbbo: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    from src.features.volume_footprint import compute_footprints

    out = pd.DataFrame(index=bar_end, columns=list(VF_COLS), dtype="float32")
    fps = compute_footprints(tbbo, "5min")
    if not fps:
        return out
    idx = pd.DatetimeIndex([fp.close_ts for fp in fps], tz="UTC")
    rows = []
    for fp in fps:
        row = {f"vf_5m_{k}": fp.features[k] for k in _REGION_SUFFIXES}
        row["vf_5m_poc_loc"] = fp.features["poc_loc"]
        rows.append(row)
    raw = pd.DataFrame(rows, index=idx)
    raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
    raw["vf_5m_poc_ema_5"] = (
        raw["vf_5m_poc_loc"]
        .ewm(span=POC_EMA_SPAN, adjust=False, min_periods=POC_EMA_SPAN)
        .mean()
    )
    raw = raw.reindex(bar_end)
    for c in VF_COLS:
        if c in raw.columns:
            out[c] = raw[c]
    return out.astype("float32")


def compute_5m_features(
    tbbo: pd.DataFrame,
    *,
    prev_tbbo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Closed-bar 5m frame with ``bar_end_ts`` + :data:`FEATURE_5M_COLS`."""
    empty = pd.DataFrame(columns=["bar_end_ts", *FEATURE_5M_COLS])
    candles = candles_5m_from_tbbo(tbbo)
    if candles.empty:
        return empty

    cs_ohlc = _shift_to_bar_end(_cs_ohlcv(candles))
    bar_end = _utc_idx(cs_ohlc.index)
    cs_ohlc = cs_ohlc.reindex(bar_end)
    atr = _atr_5(candles, bar_end)
    vp = _vp_5m(tbbo, prev_tbbo, bar_end, cs_ohlc["cs_close"])
    vw = _vwap_5m(tbbo, bar_end)
    ps = _ps_5m(candles, bar_end)
    zz = _zz_5m(candles, bar_end)
    vf = _vf_5m(tbbo, bar_end)
    rv = _rv_5m(cs_ohlc["cs_close"], bar_end)

    frame = pd.DataFrame(index=bar_end)
    for p in (cs_ohlc, vp, vw, ps, zz, vf, rv):
        if p is None or p.empty:
            continue
        p = p.copy()
        p.index = _utc_idx(p.index)
        p = p.loc[~p.index.duplicated(keep="last")]
        for c in p.columns:
            frame[c] = p.reindex(bar_end)[c]
    frame["cs_atr_5"] = atr.reindex(bar_end)
    for c in FEATURE_5M_COLS:
        if c not in frame.columns:
            frame[c] = np.nan
    frame = frame.loc[:, list(FEATURE_5M_COLS)].astype("float32")
    frame.insert(0, "bar_end_ts", frame.index)
    return frame.reset_index(drop=True)


def build_session_frame(
    conn: psycopg.Connection,
    session: date,
    *,
    root: str = "NQ",
) -> pd.DataFrame:
    """One Globex session of native 5m features."""
    symbol = active_contract(root, session)
    start, end = globex_session_window(session)
    ticks = load_tbbo(conn, start, end, symbol)
    prev = prev_session_date(session)
    p0, p1 = globex_session_window(prev)
    prev_ticks = load_tbbo(conn, p0, p1, active_contract(root, prev))
    out = compute_5m_features(ticks, prev_tbbo=prev_ticks)
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
