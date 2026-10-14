"""Native-timeframe bar features (15s hive).

1-bar tape columns are statistics of a single bar of length *timestep*.
ADR-083 adds RTH-window cumulatives (CVD / cum OFI) and 5-bar slope / ATR.
No 5s-session clip and no ffill from a slower grid.

Source ticks are Databento TBBO (one row per print, with BBO at the print).
Index alignment is left-closed ``[T, T+timestep)``; the emitted timestamp is
the closed-bar label ``bar_end_ts = T + timestep``.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHI = ZoneInfo("America/Chicago")

# Inclusive bar_end_ts clock in America/Chicago (CST/CDT).
# 08:30:00 = last closed 15s bar as cash open prints ([08:29:45, 08:30:00)).
# 15:00:00 = last closed 15s bar at 15:00 ([14:59:45, 15:00:00)).
RTH_FLOW_START_S = 8 * 3600 + 30 * 60
RTH_FLOW_END_S = 15 * 3600
RTH_FLOW_LOOKBACK = 5

# Order matches FEATURE_CATALOG_15s.md (datetime first, then features).
ONE_BAR_COLS: tuple[str, ...] = (
    "of_ofi",
    "of_delta",
    "cs_return",
    "tl_trade_location",
    "tl_aggressive_pct",
    "ss_buy_run_max",
    "ss_sell_run_max",
    "cs_clv",
    "sd_tick_count",
    "sd_bid_sz_delta",
    "sd_ask_sz_delta",
    "sd_price_impact_per_tick",
    "cs_range",
    "cs_close",
)
RTH_FLOW_COLS: tuple[str, ...] = (
    "of_cvd",
    "of_ofi_cum",
    "of_ofi_cum_slope_5",
    "cs_return_slope_5",
    "cs_atr_5",
)
BAR_FEATURE_COLS: tuple[str, ...] = (
    "of_ofi",
    "of_delta",
    "of_cvd",
    "of_ofi_cum",
    "of_ofi_cum_slope_5",
    "cs_return",
    "cs_return_slope_5",
    "tl_trade_location",
    "tl_aggressive_pct",
    "ss_buy_run_max",
    "ss_sell_run_max",
    "cs_clv",
    "sd_tick_count",
    "sd_bid_sz_delta",
    "sd_ask_sz_delta",
    "sd_price_impact_per_tick",
    "cs_range",
    "cs_close",
    "cs_atr_5",
)

TICK_COLS: tuple[str, ...] = (
    "ts_event", "price", "size", "side",
    "bid_px", "ask_px", "bid_sz", "ask_sz",
)


def pandas_freq(timestep: str) -> str:
    """``15s`` / ``1m`` / ``5min`` → pandas offset alias for ``floor`` / ``Timedelta``."""
    raw = str(timestep).strip().lower()
    if raw.endswith("ms"):
        return raw
    if raw.endswith("min"):
        return raw
    if raw.endswith("s") and not raw.endswith("ms"):
        return raw
    if raw.endswith("m") and raw[:-1].isdigit():
        return f"{raw[:-1]}min"
    return raw


def timestep_delta(timestep: str) -> pd.Timedelta:
    return pd.Timedelta(pandas_freq(timestep))


def utc_bar_end(df: pd.DataFrame) -> pd.DatetimeIndex:
    """UTC closed-bar timestamps from ``bar_end_ts`` or a DatetimeIndex."""
    if "bar_end_ts" in df.columns:
        ts = pd.to_datetime(df["bar_end_ts"], utc=True)
        return pd.DatetimeIndex(ts)
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")
    raise ValueError("need bar_end_ts or a DatetimeIndex")


def rth_cash_window_mask(ts: pd.DatetimeIndex) -> np.ndarray:
    """True when *ts* (UTC bar_end) is in [08:30:00, 15:00:00] America/Chicago."""
    chi = pd.DatetimeIndex(ts).tz_convert(CHI)
    tod = (
        chi.hour.astype("int64") * 3600
        + chi.minute.astype("int64") * 60
        + chi.second.astype("int64")
    )
    return (tod >= RTH_FLOW_START_S) & (tod <= RTH_FLOW_END_S)


def globex_session_index(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Globex session date as Chicago midnight: hour≥17 rolls to the next day."""
    chi = pd.DatetimeIndex(ts).tz_convert(CHI)
    midnight = chi.normalize()
    return midnight.where(chi.hour < 17, midnight + pd.Timedelta(days=1))


def _high_low_from_cs(
    close: np.ndarray,
    rng: np.ndarray,
    clv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover high/low from close, range, CLV (same identity as target_geometry)."""
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


def _rel_slope(values: pd.Series, sess: pd.Series, n: int = RTH_FLOW_LOOKBACK) -> pd.Series:
    prev = values.groupby(sess, sort=False).shift(n)
    den = prev.abs().replace(0.0, np.nan)
    return (values - prev) / den


def attach_rth_session_features(
    df: pd.DataFrame,
    *,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Add CVD, cum OFI, 5-bar slopes, and 5-bar ATR (ADR-083).

    ``of_cvd`` / ``of_ofi_cum`` accumulate only on closed bars with
    ``bar_end_ts`` in [08:30:00, 15:00:00] America/Chicago and reset each
    Globex session. Outside that window the columns are NaN. 5-bar slope is
    ``(v[t] − v[t−5]) / abs(v[t−5])`` (zero denominator → NaN), grouped so
    a weekend gap does not leak. ``cs_return_slope_5`` and ``cs_atr_5`` use
    every Globex bar in the session (not RTH-clipped). ``cs_atr_5`` is the
    SMA of 5 15s true ranges, not Wilder and not 5-minute ``cs_atr_5m``.
    """
    if df is None or df.empty:
        return df
    have = all(c in df.columns for c in RTH_FLOW_COLS)
    if have and not overwrite:
        return df
    if "of_delta" not in df.columns or "of_ofi" not in df.columns:
        raise ValueError("attach_rth_session_features needs of_delta and of_ofi")

    out = df.copy()
    ts = utc_bar_end(out)
    n = len(out)
    order = np.argsort(ts.to_numpy(), kind="mergesort")
    sorted_ts = ts[order]
    in_win = rth_cash_window_mask(sorted_ts)
    sess = pd.Series(globex_session_index(sorted_ts), index=np.arange(n))

    def _sorted_col(name: str) -> pd.Series:
        raw = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype="float64")
        return pd.Series(raw[order], index=np.arange(n), dtype="float64")

    delta = _sorted_col("of_delta").where(pd.Series(in_win, index=np.arange(n)))
    ofi = _sorted_col("of_ofi").where(pd.Series(in_win, index=np.arange(n)))
    cvd = delta.groupby(sess, sort=False).cumsum()
    ofi_cum = ofi.groupby(sess, sort=False).cumsum()
    ofi_slope = _rel_slope(ofi_cum, sess)
    ret_slope = _rel_slope(_sorted_col("cs_return"), sess)

    close = _sorted_col("cs_close").to_numpy(dtype="float64")
    if high is None or low is None:
        rng = (
            _sorted_col("cs_range").to_numpy(dtype="float64")
            if "cs_range" in out.columns
            else np.zeros(n, dtype="float64")
        )
        clv = (
            _sorted_col("cs_clv").to_numpy(dtype="float64")
            if "cs_clv" in out.columns
            else np.full(n, np.nan)
        )
        hi, lo = _high_low_from_cs(close, rng, clv)
    else:
        hi = np.asarray(high, dtype="float64")[order]
        lo = np.asarray(low, dtype="float64")[order]
        if hi.shape[0] != n or lo.shape[0] != n:
            raise ValueError("high/low length must match df")

    prev_c = pd.Series(close, index=np.arange(n)).groupby(sess, sort=False).shift(1)
    rng_bar = hi - lo
    tr = rng_bar.copy()
    pc = prev_c.to_numpy(dtype="float64")
    ok = np.isfinite(pc)
    tr[ok] = np.maximum.reduce([
        rng_bar[ok],
        np.abs(hi[ok] - pc[ok]),
        np.abs(lo[ok] - pc[ok]),
    ])
    atr = (
        pd.Series(tr, index=np.arange(n))
        .groupby(sess, sort=False)
        .rolling(RTH_FLOW_LOOKBACK, min_periods=RTH_FLOW_LOOKBACK)
        .mean()
    )
    if isinstance(atr.index, pd.MultiIndex):
        atr = atr.reset_index(level=0, drop=True)
    atr = atr.reindex(np.arange(n))

    def _scatter(s: pd.Series) -> np.ndarray:
        arr = np.full(n, np.nan, dtype="float64")
        arr[order] = s.to_numpy(dtype="float64")
        return arr

    out["of_cvd"] = _scatter(cvd)
    out["of_ofi_cum"] = _scatter(ofi_cum)
    out["of_ofi_cum_slope_5"] = _scatter(ofi_slope)
    out["cs_return_slope_5"] = _scatter(ret_slope)
    out["cs_atr_5"] = _scatter(atr)
    return out


def compute_bar_features(
    tbbo: pd.DataFrame,
    timestep: str = "15s",
) -> pd.DataFrame:
    """Compute 1-bar features from TBBO prints.

    Parameters
    ----------
    tbbo : DataFrame with ``ts_event``, ``price``, ``size``, ``side``,
        ``bid_px``, ``ask_px``, ``bid_sz``, ``ask_sz``.
    timestep : Bar width (``15s``, ``1m``, …). One bar = one output row
        for every interval that contains at least one print.

    Returns
    -------
    DataFrame with column ``bar_end_ts`` (UTC, closed-bar) plus
    :data:`BAR_FEATURE_COLS` (1-bar tape plus ADR-083 RTH/5-bar columns).
    Empty input → empty frame with those columns.
    """
    cols = ["bar_end_ts", *BAR_FEATURE_COLS]
    empty = pd.DataFrame(columns=cols)
    if tbbo is None or tbbo.empty or "ts_event" not in tbbo.columns:
        return empty
    needed = {"price", "size", "side", "bid_px", "ask_px", "bid_sz", "ask_sz"}
    if not needed.issubset(tbbo.columns):
        missing = needed - set(tbbo.columns)
        raise ValueError(f"tbbo missing columns: {sorted(missing)}")

    df = tbbo.loc[:, list(TICK_COLS)].copy()
    ts = pd.to_datetime(df["ts_event"], utc=True)
    keep = ts.notna()
    df = df.loc[keep].copy()
    ts = ts.loc[keep]
    if df.empty:
        return empty
    order = np.argsort(ts.to_numpy(), kind="mergesort")
    df = df.iloc[order].reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(df["ts_event"], utc=True))

    freq = pandas_freq(timestep)
    delta = timestep_delta(timestep)
    bar_start = ts.floor(freq)

    price = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype="float64")
    size = pd.to_numeric(df["size"], errors="coerce").to_numpy(dtype="float64")
    bid_px = pd.to_numeric(df["bid_px"], errors="coerce").to_numpy(dtype="float64")
    ask_px = pd.to_numeric(df["ask_px"], errors="coerce").to_numpy(dtype="float64")
    bid_sz = pd.to_numeric(df["bid_sz"], errors="coerce").to_numpy(dtype="float64")
    ask_sz = pd.to_numeric(df["ask_sz"], errors="coerce").to_numpy(dtype="float64")
    side = df["side"].to_numpy()
    n = len(df)
    if n == 0:
        return empty

    # Cont 2014 tick OFI. First print has no previous quote → NaN (dropped in sum).
    prev_bid = np.empty(n, dtype="float64")
    prev_ask = np.empty(n, dtype="float64")
    prev_bid_sz = np.empty(n, dtype="float64")
    prev_ask_sz = np.empty(n, dtype="float64")
    prev_bid[0] = np.nan
    prev_ask[0] = np.nan
    prev_bid_sz[0] = np.nan
    prev_ask_sz[0] = np.nan
    if n > 1:
        prev_bid[1:] = bid_px[:-1]
        prev_ask[1:] = ask_px[:-1]
        prev_bid_sz[1:] = bid_sz[:-1]
        prev_ask_sz[1:] = ask_sz[:-1]
    ofi = (
        np.where(bid_px >= prev_bid, bid_sz, 0.0)
        - np.where(bid_px <= prev_bid, prev_bid_sz, 0.0)
        - np.where(ask_px <= prev_ask, ask_sz, 0.0)
        + np.where(ask_px >= prev_ask, prev_ask_sz, 0.0)
    )
    # np.where(nan cmp, …) → False, but False * nan prev size must stay nan on row 0.
    ofi[0] = np.nan

    half = (ask_px - bid_px) / 2.0
    mid = (bid_px + ask_px) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = (price - mid) / np.where(half == 0.0, np.nan, half)
    loc_size = loc * size
    sweep = ((price > ask_px) | (price < bid_px)).astype("float64") * size

    # Intra-bar same-side runs: a run cannot cross a bar boundary.
    bar_codes, _ = pd.factorize(bar_start, sort=False)
    chg = np.empty(n, dtype=bool)
    chg[0] = True
    if n > 1:
        chg[1:] = (side[1:] != side[:-1]) | (bar_codes[1:] != bar_codes[:-1])
    run_id = np.cumsum(chg)
    run_len = pd.Series(run_id, copy=False).groupby(run_id, sort=False).cumcount().to_numpy() + 1
    buy_run = np.where(side == "B", run_len, 0)
    sell_run = np.where(side == "A", run_len, 0)
    signed = np.where(side == "B", size, np.where(side == "A", -size, 0.0))

    work = pd.DataFrame({
        "bar_start": bar_start,
        "ofi": ofi,
        "price": price,
        "size": size,
        "signed": signed,
        "loc_size": loc_size,
        "sweep": sweep,
        "bid_sz": bid_sz,
        "ask_sz": ask_sz,
        "buy_run": buy_run,
        "sell_run": sell_run,
    })
    g = work.groupby("bar_start", sort=True)
    agg = g.agg(
        of_ofi=("ofi", "sum"),
        of_delta=("signed", "sum"),
        cs_open=("price", "first"),
        cs_high=("price", "max"),
        cs_low=("price", "min"),
        cs_close=("price", "last"),
        sd_tick_count=("price", "count"),
        bid_first=("bid_sz", "first"),
        bid_last=("bid_sz", "last"),
        ask_first=("ask_sz", "first"),
        ask_last=("ask_sz", "last"),
        loc_size=("loc_size", "sum"),
        sweep=("sweep", "sum"),
        size=("size", "sum"),
        ss_buy_run_max=("buy_run", "max"),
        ss_sell_run_max=("sell_run", "max"),
    )
    if agg.empty:
        return empty

    open_ = agg["cs_open"].replace(0.0, np.nan)
    rng = agg["cs_high"] - agg["cs_low"]
    den_sz = agg["size"].replace(0.0, np.nan)
    ticks = agg["sd_tick_count"].replace(0.0, np.nan)

    out = pd.DataFrame(index=agg.index)
    out["bar_end_ts"] = pd.DatetimeIndex(agg.index) + delta
    out["of_ofi"] = agg["of_ofi"]
    out["of_delta"] = agg["of_delta"]
    out["cs_return"] = (agg["cs_close"] - agg["cs_open"]) / open_
    out["tl_trade_location"] = agg["loc_size"] / den_sz
    out["tl_aggressive_pct"] = agg["sweep"] / den_sz
    out["ss_buy_run_max"] = agg["ss_buy_run_max"].fillna(0)
    out["ss_sell_run_max"] = agg["ss_sell_run_max"].fillna(0)
    out["cs_clv"] = (2.0 * agg["cs_close"] - agg["cs_high"] - agg["cs_low"]) / rng.replace(0.0, np.nan)
    out["sd_tick_count"] = agg["sd_tick_count"]
    out["sd_bid_sz_delta"] = agg["bid_last"] - agg["bid_first"]
    out["sd_ask_sz_delta"] = agg["ask_last"] - agg["ask_first"]
    out["sd_price_impact_per_tick"] = rng / ticks
    out["cs_range"] = rng
    out["cs_close"] = agg["cs_close"]

    out = out.reset_index(drop=True)
    out["bar_end_ts"] = pd.to_datetime(out["bar_end_ts"], utc=True)
    out = attach_rth_session_features(
        out,
        high=agg["cs_high"].to_numpy(dtype="float64"),
        low=agg["cs_low"].to_numpy(dtype="float64"),
        overwrite=True,
    )
    for c in BAR_FEATURE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    return out.loc[:, cols]
