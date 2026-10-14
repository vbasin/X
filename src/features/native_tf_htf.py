"""Native 1h / 1d HTF context stores (ADR-091).

One row per closed Globex hour or Globex session. Built from
``databento.tbbo_1h`` / ``tbbo_1d`` caggs (lead-month NQ) — no TBBO tick
scan. Groups: ``cs_`` (OHLCV + ATR(14)), ``tr_`` (EMA / slope / gap
regime), ``ps_`` (SAR), ``zz_`` (EXT ZigZag), ``rv_`` (vol regime),
``sr_`` (Donchian, prior-period H/L/C, floor pivots, Fib of last ZZ
swing).

``bar_end_ts`` is the close stamp: 1h left-closed ``[T, T+1h)`` labeled
``T+1h``; 1d Globex ``[T-1 17:00, T 17:00)`` CT labeled at session end.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg

from src.common.contract_calendar import active_contract
from src.features.native_tf import globex_session_date, globex_session_window

logger = logging.getLogger("vtech.features.native_tf_htf")

CHI = ZoneInfo("America/Chicago")

CAGG_VIEW = {"1h": "databento.tbbo_1h", "1d": "databento.tbbo_1d"}
SUPPORTED_TF = frozenset(CAGG_VIEW)

CS_COLS: tuple[str, ...] = (
    "cs_open",
    "cs_high",
    "cs_low",
    "cs_close",
    "cs_volume",
    "cs_range",
    "cs_atr_14",
    "cs_clv",
)
TR_COLS: tuple[str, ...] = (
    "tr_ema8_dist",
    "tr_ema21_dist",
    "tr_ema50_dist",
    "tr_ema_stack",
    "tr_slope_10",
    "tr_slope_20",
    "tr_gap",
)
PS_COLS: tuple[str, ...] = (
    "ps_trend",
    "ps_next_dist_atr",
    "ps_flip",
)
ZZ_COLS: tuple[str, ...] = (
    "zz_signal",
    "zz_state",
    "zz_swing_high_dist",
    "zz_swing_low_dist",
)
RV_COLS: tuple[str, ...] = (
    "rv_realized_vol_5",
    "rv_realized_vol_10",
    "rv_realized_vol_20",
    "rv_vol_of_vol_10",
    "rv_vol_ratio",
    "rv_atr_ratio",
)
SR_COLS: tuple[str, ...] = (
    "sr_donch_high_20_dist",
    "sr_donch_low_20_dist",
    "sr_donch_high_50_dist",
    "sr_donch_low_50_dist",
    "sr_pos_in_range_20",
    "sr_pos_in_range_50",
    "sr_prior_high_dist",
    "sr_prior_low_dist",
    "sr_prior_close_dist",
    "sr_pivot_dist",
    "sr_r1_dist",
    "sr_s1_dist",
    "sr_r2_dist",
    "sr_s2_dist",
    "sr_fib_382_dist",
    "sr_fib_500_dist",
    "sr_fib_618_dist",
)
FEATURE_HTF_COLS: tuple[str, ...] = (
    *CS_COLS, *TR_COLS, *PS_COLS, *ZZ_COLS, *RV_COLS, *SR_COLS,
)
FEATURE_1H_COLS = FEATURE_HTF_COLS
FEATURE_1D_COLS = FEATURE_HTF_COLS

_CAGG_SQL = """
SELECT bucket, symbol, open, high, low, close, volume
FROM {view}
WHERE bucket >= %s AND bucket < %s
  AND symbol ~ '^NQ[HMUZ][0-9]+$'
ORDER BY bucket, symbol
"""
_SPAN_SQL = "SELECT min(bucket), max(bucket) FROM {view}"


def _utc_idx(idx) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(idx)
    if out.tz is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _require_tf(timeframe: str) -> str:
    tf = str(timeframe).strip().lower()
    if tf not in SUPPORTED_TF:
        raise ValueError(f"native HTF timeframe must be 1h or 1d, got {timeframe!r}")
    return tf


def bar_end_from_left(bucket, timeframe: str) -> pd.DatetimeIndex:
    """Left cagg label → closed-bar ``bar_end_ts``.

    1h: unix hour ``T`` covers ``[T, T+1h)``.
    1d: Globex session open (17:00 CT) covers ``[open, next 17:00 CT)``;
    session length is 23/24/25 UTC hours around DST, so the close is the
    next Chicago 17:00, not UTC+24h.
    """
    tf = _require_tf(timeframe)
    b = _utc_idx(bucket)
    if len(b) == 0:
        return b
    if tf == "1h":
        return b + pd.Timedelta(hours=1)
    chi = b.tz_convert(CHI)
    end = pd.DatetimeIndex(chi + pd.DateOffset(days=1)).tz_convert("UTC")
    return end


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["bar_end_ts", *FEATURE_HTF_COLS])


def _dist(close: pd.Series, level) -> pd.Series:
    px = pd.to_numeric(close, errors="coerce")
    lv = pd.to_numeric(level, errors="coerce")
    den = px.replace(0, np.nan)
    return (px - lv) / den


def _session_of_closed(bar_end: pd.DatetimeIndex) -> list[date]:
    """Globex session T that the closed bar belongs to (last instant of the bar)."""
    out: list[date] = []
    delta = pd.Timedelta(seconds=1)
    for t in bar_end:
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out.append(globex_session_date((ts - delta).to_pydatetime()))
    return out


def select_lead_month(raw: pd.DataFrame, *, root: str = "NQ") -> pd.DataFrame:
    """One row per ``bucket``: lead-month NQ, else max-volume symbol that hour/day."""
    need = ("bucket", "symbol", "open", "high", "low", "close")
    empty = pd.DataFrame(columns=list(need) + ["volume"])
    if raw is None or raw.empty:
        return empty
    df = raw.copy()
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    df = df.dropna(subset=["close", "bucket"])
    if df.empty:
        return empty
    sess = [
        globex_session_date(pd.Timestamp(t).to_pydatetime())
        for t in df["bucket"]
    ]
    df["_sess"] = sess
    want = {s: active_contract(root, s) for s in set(sess)}
    df["_want"] = df["_sess"].map(want)
    lead = df[df["symbol"] == df["_want"]]
    have = set(lead["bucket"].tolist())
    leftover = df[~df["bucket"].isin(have)]
    if not leftover.empty:
        vol = leftover["volume"].fillna(0.0)
        leftover = leftover.assign(_vol=vol)
        idx = leftover.groupby("bucket")["_vol"].idxmax()
        fallback = leftover.loc[idx]
        lead = pd.concat([lead, fallback], ignore_index=True)
    lead = lead.sort_values("bucket").drop_duplicates(subset=["bucket"], keep="last")
    cols = ["bucket", "symbol", "open", "high", "low", "close", "volume"]
    return lead.loc[:, cols].reset_index(drop=True)


def _rolling_linreg_slope(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype="float64")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return pd.Series(np.nan, index=close.index)

    def _sl(y: np.ndarray) -> float:
        if y.size < window or not np.isfinite(y).all():
            return np.nan
        yc = y - y.mean()
        return float(np.dot(x, yc) / denom)

    return close.rolling(window, min_periods=window).apply(_sl, raw=True)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    from src.features.psar import _rma, _true_range

    tr = _true_range(
        high.to_numpy(dtype="float64"),
        low.to_numpy(dtype="float64"),
        close.to_numpy(dtype="float64"),
    )
    return pd.Series(_rma(tr, period), index=close.index)


def _floor_pivots(h: pd.Series, l: pd.Series, c: pd.Series) -> dict[str, pd.Series]:
    pp = (h + l + c) / 3.0
    rng = h - l
    return {
        "pp": pp,
        "r1": 2.0 * pp - l,
        "s1": 2.0 * pp - h,
        "r2": pp + rng,
        "s2": pp - rng,
    }


def _prior_levels(
    candles: pd.DataFrame,
    timeframe: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prior H/L/C for distances, and H/L/C used for floor pivots.

    1h: prior Globex session H/L/C for both.
    1d: prior session H/L/C for distances; rolling 5-session H/L + last
    close for weekly-style pivots.
    """
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]
    if timeframe == "1d":
        ph = high.shift(1)
        pl = low.shift(1)
        pc = close.shift(1)
        piv_h = ph.rolling(5, min_periods=3).max()
        piv_l = pl.rolling(5, min_periods=3).min()
        piv_c = pc
        return ph, pl, pc, piv_h, piv_l, piv_c

    sess = _session_of_closed(pd.DatetimeIndex(candles.index))
    g = pd.DataFrame(
        {"high": high, "low": low, "close": close, "session": sess},
        index=candles.index,
    )
    daily = g.groupby("session", sort=True).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    prior = daily.shift(1)
    mapped_h = g["session"].map(prior["high"])
    mapped_l = g["session"].map(prior["low"])
    mapped_c = g["session"].map(prior["close"])
    mapped_h.index = g.index
    mapped_l.index = g.index
    mapped_c.index = g.index
    return mapped_h, mapped_l, mapped_c, mapped_h, mapped_l, mapped_c


def _cs(candles: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=candles.index)
    out["cs_open"] = candles["open"]
    out["cs_high"] = candles["high"]
    out["cs_low"] = candles["low"]
    out["cs_close"] = candles["close"]
    out["cs_volume"] = candles["volume"] if "volume" in candles.columns else np.nan
    rng = candles["high"] - candles["low"]
    out["cs_range"] = rng
    clv_den = rng.replace(0, np.nan)
    out["cs_clv"] = (
        (candles["close"] - candles["low"]) - (candles["high"] - candles["close"])
    ) / clv_den
    out["cs_atr_14"] = _atr(candles["high"], candles["low"], candles["close"], 14)
    return out.astype("float32")


def _tr(candles: pd.DataFrame, atr14: pd.Series, timeframe: str) -> pd.DataFrame:
    out = pd.DataFrame(index=candles.index, columns=list(TR_COLS), dtype="float32")
    close = pd.to_numeric(candles["close"], errors="coerce")
    den = close.replace(0, np.nan)
    e8 = close.ewm(span=8, adjust=False, min_periods=8).mean()
    e21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    e50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["tr_ema8_dist"] = (close - e8) / den
    out["tr_ema21_dist"] = (close - e21) / den
    out["tr_ema50_dist"] = (close - e50) / den
    stack = np.full(len(close), np.nan, dtype="float64")
    e8a, e21a, e50a = e8.to_numpy(), e21.to_numpy(), e50.to_numpy()
    ok = np.isfinite(e8a) & np.isfinite(e21a) & np.isfinite(e50a)
    stack[ok] = 0.0
    stack[ok & (e8a > e21a) & (e21a > e50a)] = 1.0
    stack[ok & (e8a < e21a) & (e21a < e50a)] = -1.0
    out["tr_ema_stack"] = stack
    atr = atr14.replace(0, np.nan)
    out["tr_slope_10"] = _rolling_linreg_slope(close, 10) / atr
    out["tr_slope_20"] = _rolling_linreg_slope(close, 20) / atr
    if timeframe == "1d":
        prev = close.shift(1).replace(0, np.nan)
        out["tr_gap"] = (candles["open"] - close.shift(1)) / prev
    else:
        sess = _session_of_closed(pd.DatetimeIndex(candles.index))
        g = pd.DataFrame({"open": candles["open"], "session": sess}, index=candles.index)
        sess_open = g.groupby("session", sort=False)["open"].transform("first")
        out["tr_gap"] = (close - sess_open) / sess_open.replace(0, np.nan)
    return out.astype("float32")


def _ps(candles: pd.DataFrame) -> pd.DataFrame:
    from src.features.psar import sar_on_closed

    out = pd.DataFrame(index=candles.index, columns=list(PS_COLS), dtype="float32")
    raw = sar_on_closed(candles[["high", "low", "close"]])
    if raw.empty:
        return out
    raw = raw.reindex(candles.index)
    out["ps_trend"] = raw["trend"]
    out["ps_next_dist_atr"] = raw["next_dist_atr"]
    out["ps_flip"] = raw["flip"]
    return out.astype("float32")


def _zz(candles: pd.DataFrame) -> pd.DataFrame:
    from src.external_models.catalog import factory_params
    from src.external_models.zigzag import zzhl_on_closed

    out = pd.DataFrame(index=candles.index, columns=list(ZZ_COLS), dtype="float32")
    p = factory_params("zigzag")
    raw = zzhl_on_closed(
        candles[["high", "low", "close"]],
        percentage_reversal=float(p["percentage_reversal"]),
        absolute_reversal=float(p["absolute_reversal"]),
        atr_length=int(p["atr_length"]),
        atr_reversal=float(p["atr_reversal"]),
        tick_reversal=int(p["tick_reversal"]),
        tick_size=float(p["tick_size"]),
    )
    if raw.empty:
        return out
    raw = raw.reindex(candles.index)
    close = candles["close"]
    out["zz_signal"] = raw["signal"]
    out["zz_state"] = raw["state"]
    out["zz_swing_high_dist"] = _dist(close, raw["max_price_h"])
    out["zz_swing_low_dist"] = _dist(close, raw["min_price_l"])
    return out.astype("float32")


def _rv(close: pd.Series, atr5: pd.Series, atr20: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index, columns=list(RV_COLS), dtype="float32")
    px = pd.to_numeric(close, errors="coerce")
    log_ret = np.log(px / px.shift(1))
    rv5 = log_ret.rolling(5, min_periods=5).std()
    rv10 = log_ret.rolling(10, min_periods=10).std()
    rv20 = log_ret.rolling(20, min_periods=20).std()
    out["rv_realized_vol_5"] = rv5
    out["rv_realized_vol_10"] = rv10
    out["rv_realized_vol_20"] = rv20
    out["rv_vol_of_vol_10"] = rv5.rolling(10, min_periods=10).std()
    out["rv_vol_ratio"] = rv5 / rv20.replace(0, np.nan)
    out["rv_atr_ratio"] = atr5 / atr20.replace(0, np.nan)
    return out.astype("float32")


def _sr(candles: pd.DataFrame, timeframe: str, zz: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=candles.index, columns=list(SR_COLS), dtype="float32")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    close = pd.to_numeric(candles["close"], errors="coerce")
    dh20 = high.rolling(20, min_periods=5).max()
    dl20 = low.rolling(20, min_periods=5).min()
    dh50 = high.rolling(50, min_periods=10).max()
    dl50 = low.rolling(50, min_periods=10).min()
    out["sr_donch_high_20_dist"] = _dist(close, dh20)
    out["sr_donch_low_20_dist"] = _dist(close, dl20)
    out["sr_donch_high_50_dist"] = _dist(close, dh50)
    out["sr_donch_low_50_dist"] = _dist(close, dl50)
    w20 = (dh20 - dl20).replace(0, np.nan)
    w50 = (dh50 - dl50).replace(0, np.nan)
    out["sr_pos_in_range_20"] = (close - dl20) / w20
    out["sr_pos_in_range_50"] = (close - dl50) / w50

    ph, pl, pc, piv_h, piv_l, piv_c = _prior_levels(candles, timeframe)
    out["sr_prior_high_dist"] = _dist(close, ph)
    out["sr_prior_low_dist"] = _dist(close, pl)
    out["sr_prior_close_dist"] = _dist(close, pc)
    piv = _floor_pivots(piv_h, piv_l, piv_c)
    out["sr_pivot_dist"] = _dist(close, piv["pp"])
    out["sr_r1_dist"] = _dist(close, piv["r1"])
    out["sr_s1_dist"] = _dist(close, piv["s1"])
    out["sr_r2_dist"] = _dist(close, piv["r2"])
    out["sr_s2_dist"] = _dist(close, piv["s2"])

    sh = pd.to_numeric(zz["zz_swing_high_dist"], errors="coerce")
    sl = pd.to_numeric(zz["zz_swing_low_dist"], errors="coerce")
    # Reconstruct swing prices from distances already on zz.
    swing_h = close - sh * close.replace(0, np.nan)
    swing_l = close - sl * close.replace(0, np.nan)
    rng = (swing_h - swing_l).replace(0, np.nan)
    fib382 = swing_l + 0.382 * rng
    fib500 = swing_l + 0.500 * rng
    fib618 = swing_l + 0.618 * rng
    out["sr_fib_382_dist"] = _dist(close, fib382)
    out["sr_fib_500_dist"] = _dist(close, fib500)
    out["sr_fib_618_dist"] = _dist(close, fib618)
    return out.astype("float32")


def compute_htf_features(
    candles: pd.DataFrame,
    *,
    timeframe: str,
    symbol: pd.Series | None = None,
) -> pd.DataFrame:
    """Closed-bar HTF frame. *candles* index is the **left** cagg bucket."""
    tf = _require_tf(timeframe)
    empty = empty_frame()
    if candles is None or candles.empty:
        return empty
    src = candles.copy()
    src.index = _utc_idx(src.index)
    src = src.loc[~src.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low", "close"):
        if c not in src.columns:
            return empty
        src[c] = pd.to_numeric(src[c], errors="coerce")
    if "volume" not in src.columns:
        src["volume"] = np.nan
    else:
        src["volume"] = pd.to_numeric(src["volume"], errors="coerce")
    src = src.dropna(subset=["close"])
    if src.empty:
        return empty

    bar_end = bar_end_from_left(src.index, tf)
    closed = src.copy()
    closed.index = bar_end
    closed = closed.loc[~closed.index.duplicated(keep="last")].sort_index()
    bar_end = pd.DatetimeIndex(closed.index)

    cs = _cs(closed)
    atr5 = _atr(closed["high"], closed["low"], closed["close"], 5)
    atr20 = _atr(closed["high"], closed["low"], closed["close"], 20)
    tr = _tr(closed, cs["cs_atr_14"], tf)
    ps = _ps(closed)
    zz = _zz(closed)
    rv = _rv(closed["close"], atr5, atr20)
    sr = _sr(closed, tf, zz)

    frame = pd.DataFrame(index=bar_end)
    for part in (cs, tr, ps, zz, rv, sr):
        for c in part.columns:
            frame[c] = part.reindex(bar_end)[c]
    for c in FEATURE_HTF_COLS:
        if c not in frame.columns:
            frame[c] = np.nan
    frame = frame.loc[:, list(FEATURE_HTF_COLS)].astype("float32")
    frame.insert(0, "bar_end_ts", frame.index)
    if symbol is not None:
        sym = pd.Series(symbol, index=_utc_idx(getattr(symbol, "index", src.index)))
        sym.index = bar_end_from_left(sym.index, tf)
        sym = sym.loc[~sym.index.duplicated(keep="last")]
        frame["_meta_nq_contract"] = sym.reindex(bar_end).astype("object").to_numpy()
    elif "symbol" in src.columns:
        sym = src["symbol"]
        sym.index = bar_end
        frame["_meta_nq_contract"] = sym.reindex(bar_end).astype("object").to_numpy()
    return frame.reset_index(drop=True)


def cagg_span(conn: psycopg.Connection, timeframe: str) -> tuple[datetime, datetime]:
    tf = _require_tf(timeframe)
    row = conn.execute(_SPAN_SQL.format(view=CAGG_VIEW[tf])).fetchone()
    if not row or row[0] is None or row[1] is None:
        raise RuntimeError(f"{CAGG_VIEW[tf]} is empty")
    t0, t1 = row[0], row[1]
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    return t0, t1


def load_cagg(
    conn: psycopg.Connection,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    tf = _require_tf(timeframe)
    sql = _CAGG_SQL.format(view=CAGG_VIEW[tf])
    with conn.cursor() as cur:
        cur.execute(sql, (start, end))
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    if not rows:
        return pd.DataFrame(columns=["bucket", "symbol", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=names)
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    return df


def build_htf_frame(
    conn: psycopg.Connection,
    timeframe: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    root: str = "NQ",
) -> pd.DataFrame:
    """Load cagg → lead-month OHLC → HTF features for *timeframe*."""
    tf = _require_tf(timeframe)
    t0, t1 = cagg_span(conn, tf)
    if start is not None:
        t0 = start
    if end is not None:
        t1 = end
    raw = load_cagg(conn, tf, t0, t1)
    lead = select_lead_month(raw, root=root)
    if lead.empty:
        return empty_frame()
    candles = lead.set_index("bucket")[["open", "high", "low", "close", "volume"]]
    return compute_htf_features(
        candles, timeframe=tf, symbol=lead.set_index("bucket")["symbol"],
    )


def clip_bar_end(
    df: pd.DataFrame,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Keep rows whose Globex session date is in ``[start, end]`` inclusive."""
    if df is None or df.empty or (start is None and end is None):
        return df
    ts = pd.to_datetime(df["bar_end_ts"], utc=True)
    sess = _session_of_closed(pd.DatetimeIndex(ts))
    mask = np.ones(len(df), dtype=bool)
    if start is not None:
        mask &= np.array([s >= start for s in sess])
    if end is not None:
        mask &= np.array([s <= end for s in sess])
    return df.loc[mask].reset_index(drop=True)


def globex_bounds(start: date | None, end: date | None) -> tuple[datetime | None, datetime | None]:
    """UTC window covering Globex sessions ``[start, end]`` plus lookback slack."""
    t0 = t1 = None
    if start is not None:
        t0, _ = globex_session_window(start)
        t0 = t0 - pd.Timedelta(days=90).to_pytimedelta()
    if end is not None:
        _, t1 = globex_session_window(end)
    return t0, t1
