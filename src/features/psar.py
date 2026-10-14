"""Parabolic SAR features for the 5s parquet.

Internal engine is the operator Pine SAR / nextBarSAR state machine on
**closed 1-minute** OHLC (start=0.02, increment=0.02, maximum=0.2, ATR(8)).
Only four columns are written:

* ``ps_1m_next_dist_atr`` — ``(1m close − nextBarSAR) / ATR(8)`` at 1m close,
  held for the following minute
* ``ps_1m_trend`` — ``+1`` / ``−1`` from that 1m close, held 1m
* ``ps_lh_dist_sar`` — 5s low−SAR (uptrend) or SAR−5s high (downtrend),
  vs the in-effect trail (``nextBarSAR`` of the last closed 1m)
* ``ps_flip`` — 1 on a 5s bar with ``ps_lh_dist_sar < 0`` and the next
  11 bars (12 × 5s = 60s)

SAR, nextBarSAR, AF, EP, 1m dist, bars-in-trend are computed but not stored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import get_timestep
from src.features.candle_structure import align_to_grid

DEFAULT_START = 0.02
DEFAULT_INCREMENT = 0.02
DEFAULT_MAXIMUM = 0.2
DEFAULT_ATR_PERIOD = 8
FLIP_BARS = 12  # 12 × 5s = 60s

PS_COLS: tuple[str, ...] = (
    "ps_1m_next_dist_atr",
    "ps_1m_trend",
    "ps_lh_dist_sar",
    "ps_flip",
)
PS_RETIRED_COLS: tuple[str, ...] = (
    "ps_1m_sar",
    "ps_1m_next_sar",
    "ps_1m_af",
    "ps_1m_dist_atr",
    "ps_1m_bars",
    "ps_1m_flip",
)


def tf_label(timeframe: str) -> str:
    """``1min`` → ``1m``, ``5min`` → ``5m``, ``15s`` → ``15s``."""
    tf = normalize_timeframe(timeframe)
    if tf.endswith("min"):
        return tf[:-3] + "m"
    return tf


def normalize_timeframe(timeframe: str) -> str:
    """Canonical pandas offset: ``1m`` → ``1min``. Rejects empty/unknown aliases."""
    tf = str(timeframe).strip().lower()
    if not tf:
        raise ValueError("timeframe is empty")
    if tf.endswith("m") and not tf.endswith("min") and not tf.endswith("ms") and tf[:-1].isdigit():
        tf = f"{tf[:-1]}min"
    pd.tseries.frequencies.to_offset(tf)
    return tf


def _ohlc_frame(candles: pd.DataFrame) -> pd.DataFrame:
    if candles is None or candles.empty:
        return pd.DataFrame(columns=["high", "low", "close"])
    need = [c for c in ("high", "low", "close") if c in candles.columns]
    if len(need) < 3:
        return pd.DataFrame(columns=["high", "low", "close"])
    cols = ["high", "low", "close"]
    if "open" in candles.columns:
        cols = ["open", "high", "low", "close"]
    src = candles[cols].copy()
    idx = src.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
        src.index = idx
    elif idx.tz is None:
        src.index = idx.tz_localize("UTC")
    else:
        src.index = idx.tz_convert("UTC")
    src = src.loc[~src.index.duplicated(keep="last")].sort_index()
    for c in ("high", "low", "close"):
        src[c] = pd.to_numeric(src[c], errors="coerce")
    src = src.dropna(subset=["high", "low", "close"])
    return src


def _native_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) >= 2:
        step = pd.Timedelta(index.to_series().diff().median())
        if pd.notna(step) and step > pd.Timedelta(0):
            return step
    return pd.Timedelta("1min")


def closed_ohlc(
    candles: pd.DataFrame,
    timeframe: str = "1min",
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Left-labeled 1-min (or native TF) OHLC → right-labeled closed buckets.

    * Same TF as the source: shift the index by one bucket (ADR-041).
    * Slower TF: ``resample(tf, label="right", closed="left")``. Incomplete
      last buckets (close stamp after the source coverage end) are dropped.
    """
    src = _ohlc_frame(candles)
    tf = normalize_timeframe(timeframe)
    if src.empty:
        return src
    native = _native_delta(src.index)
    delta = pd.to_timedelta(tf)
    src_close_end = src.index.max() + native
    if abs(native - delta) <= pd.Timedelta("1s"):
        out = src.copy()
        out.index = out.index + delta
    elif delta > native:
        agg = {"high": "max", "low": "min", "close": "last"}
        if "open" in src.columns:
            agg["open"] = "first"
        out = src.resample(tf, label="right", closed="left").agg(agg)
        out = out.dropna(subset=["close"])
    else:
        raise ValueError(
            f"cannot upsample candles (native {native}) to {tf}"
        )
    out = out.loc[out.index <= src_close_end]
    if as_of is not None:
        t = pd.Timestamp(as_of)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        out = out.loc[out.index <= t]
    return out


def _rma(tr: np.ndarray, period: int) -> np.ndarray:
    """Wilder RMA with SMA seed (TradingView ``ta.rma`` / ``ta.atr``)."""
    n = int(tr.size)
    out = np.full(n, np.nan, dtype="float64")
    if n < period or period < 1:
        return out
    out[period - 1] = float(np.mean(tr[:period]))
    alpha = 1.0 / float(period)
    for i in range(period, n):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev = np.empty_like(close)
    prev[0] = close[0]
    prev[1:] = close[:-1]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev), np.abs(low - prev)),
    )


def sar_on_closed(
    ohlc: pd.DataFrame,
    *,
    start: float = DEFAULT_START,
    increment: float = DEFAULT_INCREMENT,
    maximum: float = DEFAULT_MAXIMUM,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> pd.DataFrame:
    """Run the Pine SAR engine on already-closed OHLC (index = close stamp)."""
    cols = (
        "sar", "next_sar", "trend", "af",
        "dist_atr", "next_dist_atr", "bars", "flip",
    )
    if ohlc is None or ohlc.empty:
        return pd.DataFrame(columns=list(cols))
    high = ohlc["high"].to_numpy(dtype="float64")
    low = ohlc["low"].to_numpy(dtype="float64")
    close = ohlc["close"].to_numpy(dtype="float64")
    n = int(close.size)
    sar = np.full(n, np.nan)
    next_sar = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    af_arr = np.full(n, np.nan)
    bars = np.full(n, np.nan)
    flip = np.zeros(n, dtype="float64")

    if n < 2:
        return pd.DataFrame(
            {c: np.full(n, np.nan) for c in cols},
            index=ohlc.index,
        )

    uptrend = False
    ep = np.nan
    af = float(start)
    next_bar = np.nan
    bars_in = 0

    for i in range(1, n):
        first_trend_bar = False
        cur = next_bar
        if i == 1:
            if close[1] > close[0]:
                uptrend = True
                ep = high[1]
                prev_sar = low[0]
                prev_ep = high[1]
            else:
                uptrend = False
                ep = low[1]
                prev_sar = high[0]
                prev_ep = low[1]
            first_trend_bar = True
            cur = prev_sar + float(start) * (prev_ep - prev_sar)
            af = float(start)
            bars_in = 0

        flipped = (uptrend and cur > low[i]) or ((not uptrend) and cur < high[i])
        if flipped:
            first_trend_bar = True
            uptrend = not uptrend
            cur = min(ep, low[i]) if uptrend else max(ep, high[i])
            ep = high[i] if uptrend else low[i]
            af = float(start)
            bars_in = 0
            flip[i] = 1.0

        if not first_trend_bar:
            if uptrend and high[i] > ep:
                ep = high[i]
                af = min(af + float(increment), float(maximum))
            if (not uptrend) and low[i] < ep:
                ep = low[i]
                af = min(af + float(increment), float(maximum))

        if uptrend:
            cur = min(cur, low[i - 1])
            if i > 1:
                cur = min(cur, low[i - 2])
        else:
            cur = max(cur, high[i - 1])
            if i > 1:
                cur = max(cur, high[i - 2])

        nxt = cur + af * (ep - cur)
        bars_in += 1
        sar[i] = cur
        next_sar[i] = nxt
        trend[i] = 1.0 if uptrend else -1.0
        af_arr[i] = af
        bars[i] = float(bars_in)
        next_bar = nxt

    tr = _true_range(high, low, close)
    atr = _rma(tr, int(atr_period))
    atr_ok = np.where(np.isfinite(atr) & (atr > 0.0), atr, np.nan)
    dist = (close - sar) / atr_ok
    next_dist = (close - next_sar) / atr_ok
    return pd.DataFrame(
        {
            "sar": sar,
            "next_sar": next_sar,
            "trend": trend,
            "af": af_arr,
            "dist_atr": dist,
            "next_dist_atr": next_dist,
            "bars": bars,
            "flip": flip,
        },
        index=ohlc.index,
    )


def _price_series(tbbo: pd.DataFrame) -> pd.Series:
    if tbbo is None or getattr(tbbo, "empty", True) or "price" not in tbbo.columns:
        return pd.Series(dtype="float64")
    if "ts_event" in tbbo.columns:
        ts = pd.to_datetime(tbbo["ts_event"], utc=True)
    elif isinstance(tbbo.index, pd.DatetimeIndex):
        ts = tbbo.index
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    else:
        return pd.Series(dtype="float64")
    px = pd.to_numeric(tbbo["price"], errors="coerce")
    s = pd.Series(px.to_numpy(dtype="float64"), index=pd.DatetimeIndex(ts), name="price")
    s = s[np.isfinite(s.to_numpy())]
    s = s.sort_index()
    return s.loc[~s.index.duplicated(keep="last")]


def ohlc_5s(
    tbbo: pd.DataFrame,
    grid: pd.DatetimeIndex,
    timestep: str = "5s",
) -> pd.DataFrame:
    """Left-labeled 5s high/low from TBBO prints. Silent bars stay NaN."""
    grid = pd.DatetimeIndex(grid)
    if grid.tz is None:
        grid = grid.tz_localize("UTC")
    else:
        grid = grid.tz_convert("UTC")
    empty = pd.DataFrame(
        np.nan, index=grid, columns=["high", "low"],
    )
    px = _price_series(tbbo)
    if px.empty:
        return empty
    bar = px.resample(timestep).agg(high="max", low="min")
    return bar.reindex(grid)


def _as_of(grid: pd.DatetimeIndex | None) -> pd.Timestamp | None:
    if grid is None or len(grid) == 0:
        return None
    t = pd.Timestamp(pd.DatetimeIndex(grid).max())
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _empty(grid: pd.DatetimeIndex | None) -> pd.DataFrame:
    idx = grid if grid is not None else pd.DatetimeIndex([], tz="UTC")
    return pd.DataFrame(index=idx, columns=list(PS_COLS), dtype=np.float32)


def _lh_dist(low: np.ndarray, high: np.ndarray, sar: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """Uptrend: low − SAR. Downtrend: SAR − high. Negative = pierce."""
    out = np.full(trend.shape, np.nan, dtype="float64")
    up = trend > 0
    dn = trend < 0
    out[up] = low[up] - sar[up]
    out[dn] = sar[dn] - high[dn]
    return out


def _sticky_flip(dist: np.ndarray, n_bars: int = FLIP_BARS) -> np.ndarray:
    """1 on a pierce bar and the following ``n_bars - 1`` bars (12 × 5s = 60s)."""
    pierce = (np.isfinite(dist) & (dist < 0.0)).astype("float64")
    if pierce.size == 0:
        return pierce
    return (
        pd.Series(pierce)
        .rolling(int(n_bars), min_periods=1)
        .max()
        .to_numpy(dtype="float64")
    )


def compute(
    candles: pd.DataFrame,
    grid: pd.DatetimeIndex | None = None,
    *,
    tbbo: pd.DataFrame | None = None,
    timestep: str | None = None,
    start: float = DEFAULT_START,
    increment: float = DEFAULT_INCREMENT,
    maximum: float = DEFAULT_MAXIMUM,
    atr_period: int = DEFAULT_ATR_PERIOD,
    flip_bars: int = FLIP_BARS,
) -> pd.DataFrame:
    """Four parquet columns on *grid* (5s). 1m SAR state is internal."""
    timestep = timestep or get_timestep()
    if grid is None:
        return _empty(grid)
    grid = pd.DatetimeIndex(grid)
    if grid.tz is None:
        grid = grid.tz_localize("UTC")
    else:
        grid = grid.tz_convert("UTC")

    ohlc = closed_ohlc(candles, "1min", as_of=_as_of(grid))
    raw = sar_on_closed(
        ohlc,
        start=start,
        increment=increment,
        maximum=maximum,
        atr_period=atr_period,
    )
    slow = pd.DataFrame(index=raw.index)
    if not raw.empty:
        slow["ps_1m_next_dist_atr"] = raw["next_dist_atr"]
        slow["ps_1m_trend"] = raw["trend"]
        slow["_next_sar"] = raw["next_sar"]
    slow = align_to_grid(slow, grid).ffill()

    bar5 = ohlc_5s(tbbo if tbbo is not None else pd.DataFrame(), grid, timestep)
    sar = slow["_next_sar"].to_numpy(dtype="float64") if "_next_sar" in slow.columns else np.full(len(grid), np.nan)
    trend = slow["ps_1m_trend"].to_numpy(dtype="float64") if "ps_1m_trend" in slow.columns else np.full(len(grid), np.nan)
    low = bar5["low"].to_numpy(dtype="float64") if "low" in bar5.columns else np.full(len(grid), np.nan)
    high = bar5["high"].to_numpy(dtype="float64") if "high" in bar5.columns else np.full(len(grid), np.nan)
    dist = _lh_dist(low, high, sar, trend)
    flip = _sticky_flip(dist, n_bars=int(flip_bars))

    out = pd.DataFrame(index=grid)
    if "ps_1m_next_dist_atr" in slow.columns:
        out["ps_1m_next_dist_atr"] = slow["ps_1m_next_dist_atr"]
    else:
        out["ps_1m_next_dist_atr"] = np.nan
    if "ps_1m_trend" in slow.columns:
        out["ps_1m_trend"] = slow["ps_1m_trend"]
    else:
        out["ps_1m_trend"] = np.nan
    out["ps_lh_dist_sar"] = dist
    out["ps_flip"] = flip
    return out[list(PS_COLS)].astype(np.float32)


def build_ps_patch_frame(
    candles_1m: pd.DataFrame,
    grid: pd.DatetimeIndex,
    tbbo: pd.DataFrame | None = None,
    *,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Align ``ps_*`` columns to an existing day parquet index (patch path)."""
    return compute(candles_1m, grid=grid, tbbo=tbbo, timestep=timestep)
