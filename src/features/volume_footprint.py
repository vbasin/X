"""Volume candle footprint: wick / body volume of a *closed* candle.

Each closed 1-min / 5-min bucket (left-closed, right-labeled — ADR-041) is
split at the candle body (open and close inclusive) into upper wick, body,
and lower wick. Per region the module stores:

* share of candle volume
* delta = ask volume − bid volume
* scaled delta = vol_frac × (delta / region volume)

Ask volume is buy-aggressor size (engine ``side=='B'``, traded at the ask);
bid volume is sell-aggressor size (``side=='A'``, traded at the bid).

1-minute bars also emit POC location in the candle range and a 5-bar EMA of
that location (span=5 on closed 1-min POC, then stale-forwarded).

``CandleFootprint.to_viz()`` / ``viz_payload()`` keep the tick ladder plus
wick/body bands for a later 1-min / 5-min overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.features.candle_structure import align_to_grid

NQ_TICK = 0.25
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("1min", "5min")
POC_EMA_SPAN = 5
MAX_LADDER_TICKS = 8_000

# Shared by 1m and 5m.
_REGION_SUFFIXES: tuple[str, ...] = (
    "upper_vol_frac",
    "body_vol_frac",
    "lower_vol_frac",
    "upper_delta",
    "body_delta",
    "lower_delta",
    "upper_delta_scaled",
    "body_delta_scaled",
    "lower_delta_scaled",
)
# 1-minute only (ADR-056).
_1M_ONLY: tuple[str, ...] = ("poc_loc", "poc_ema_5")


def tf_label(timeframe: str) -> str:
    """``1min`` → ``1m``, ``5min`` → ``5m``, ``15s`` → ``15s``."""
    tf = str(timeframe).strip().lower()
    if tf.endswith("min"):
        return tf[:-3] + "m"
    return tf


def feature_names(timeframes: Iterable[str] = DEFAULT_TIMEFRAMES) -> tuple[str, ...]:
    """Parquet column names for *timeframes* (stable order)."""
    names: list[str] = []
    for tf in timeframes:
        lab = tf_label(tf)
        for suffix in _REGION_SUFFIXES:
            names.append(f"vf_{lab}_{suffix}")
        if lab == "1m":
            for suffix in _1M_ONLY:
                names.append(f"vf_1m_{suffix}")
    return tuple(names)


VF_COLS: tuple[str, ...] = feature_names()


@dataclass
class CandleFootprint:
    """Closed-candle tick ladder plus wick/body volume features."""

    close_ts: pd.Timestamp
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    body_low: float
    body_high: float
    prices: np.ndarray
    ask: np.ndarray
    bid: np.ndarray
    total: np.ndarray
    poc: float
    features: dict[str, float] = field(default_factory=dict)

    def to_viz(self) -> dict[str, Any]:
        """JSON-serializable overlay for a 1-min / 5-min candle chart."""
        ts = pd.Timestamp(self.close_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return {
            "t_close": ts.isoformat().replace("+00:00", "Z"),
            "timeframe": self.timeframe,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "body_low": float(self.body_low),
            "body_high": float(self.body_high),
            "poc": float(self.poc) if self.poc == self.poc else None,
            "prices": [float(x) for x in np.asarray(self.prices, dtype="float64")],
            "ask": [float(x) for x in np.asarray(self.ask, dtype="float64")],
            "bid": [float(x) for x in np.asarray(self.bid, dtype="float64")],
            "total": [float(x) for x in np.asarray(self.total, dtype="float64")],
            "features": {
                k: (None if v != v else float(v)) for k, v in self.features.items()
            },
        }


def _empty_region_feats() -> dict[str, float]:
    return {s: float("nan") for s in _REGION_SUFFIXES} | {
        "poc_loc": float("nan"),
        "poc": float("nan"),
        "open": float("nan"),
        "high": float("nan"),
        "low": float("nan"),
        "close": float("nan"),
        "body_low": float("nan"),
        "body_high": float("nan"),
    }


def _region_pack(vol: float, ask: float, bid: float, candle_vol: float) -> tuple[float, float, float]:
    """vol_frac, delta (ask−bid), scaled delta for one wick/body region."""
    if candle_vol <= 0.0:
        return float("nan"), float("nan"), float("nan")
    frac = float(vol / candle_vol)
    delta = float(ask - bid)
    if vol <= 0.0:
        return frac, delta, 0.0
    return frac, delta, float(frac * (delta / vol))


def describe_regions(
    ticks: np.ndarray,
    size: np.ndarray,
    ask: np.ndarray,
    bid: np.ndarray,
) -> dict[str, float]:
    """Wick/body volume shares, deltas, scaled deltas, and POC location.

    Body is ``[min(open, close), max(open, close)]`` inclusive. Upper wick is
    strictly above the body; lower wick strictly below. Open/close are first
    and last prints (already tick-rounded).
    """
    feats = _empty_region_feats()
    ticks = np.asarray(ticks, dtype="float64")
    size = np.asarray(size, dtype="float64")
    ask = np.asarray(ask, dtype="float64")
    bid = np.asarray(bid, dtype="float64")
    if ticks.size == 0:
        return feats
    candle_vol = float(size.sum())
    if candle_vol <= 0.0:
        return feats

    open_px = float(ticks[0])
    close_px = float(ticks[-1])
    high = float(ticks.max())
    low = float(ticks.min())
    body_lo = min(open_px, close_px)
    body_hi = max(open_px, close_px)
    feats["open"] = open_px
    feats["close"] = close_px
    feats["high"] = high
    feats["low"] = low
    feats["body_low"] = body_lo
    feats["body_high"] = body_hi

    upper = ticks > body_hi
    lower = ticks < body_lo
    body = ~upper & ~lower

    for name, mask in (("upper", upper), ("body", body), ("lower", lower)):
        vol = float(size[mask].sum()) if bool(mask.any()) else 0.0
        a = float(ask[mask].sum()) if bool(mask.any()) else 0.0
        b = float(bid[mask].sum()) if bool(mask.any()) else 0.0
        frac, delta, scaled = _region_pack(vol, a, b, candle_vol)
        feats[f"{name}_vol_frac"] = frac
        feats[f"{name}_delta"] = delta
        feats[f"{name}_delta_scaled"] = scaled

    rng = high - low
    # POC = tick with the most volume (lowest price wins a tie).
    uniq, inv = np.unique(ticks, return_inverse=True)
    by_tick = np.zeros(uniq.size, dtype="float64")
    np.add.at(by_tick, inv, size)
    poc = float(uniq[int(np.argmax(by_tick))])
    feats["poc"] = poc
    feats["poc_loc"] = 0.5 if rng == 0.0 else float((poc - low) / rng)
    return feats


def _dense_ladder(
    tick_px: np.ndarray,
    size: np.ndarray,
    ask: np.ndarray,
    bid: np.ndarray,
    tick_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if tick_px.size == 0:
        return None
    lo = float(np.round(tick_px.min() / tick_size) * tick_size)
    hi = float(np.round(tick_px.max() / tick_size) * tick_size)
    n = int(round((hi - lo) / tick_size)) + 1
    n = max(n, 1)
    if n > MAX_LADDER_TICKS:
        return None
    prices = lo + np.arange(n, dtype="float64") * tick_size
    idx = np.round((tick_px - lo) / tick_size).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    total = np.zeros(n, dtype="float64")
    a = np.zeros(n, dtype="float64")
    b = np.zeros(n, dtype="float64")
    np.add.at(total, idx, size)
    np.add.at(a, idx, ask)
    np.add.at(b, idx, bid)
    return prices, a, b, total


def _aggressor_volumes(size: np.ndarray, side: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    """Split *size* into ask (buy aggressor) and bid (sell aggressor) volume."""
    n = int(size.size)
    ask = np.zeros(n, dtype="float64")
    bid = np.zeros(n, dtype="float64")
    if side is None or n == 0:
        return ask, bid
    raw = pd.Series(side).astype("string").str.upper().str.strip()
    raw = raw.fillna("")
    is_ask = raw.isin(("B", "BUY", "1")).to_numpy()
    is_bid = raw.isin(("A", "SELL", "-1")).to_numpy()
    ask = np.where(is_ask, size, 0.0)
    bid = np.where(is_bid, size, 0.0)
    return ask, bid


def _prepare_prints(tbbo: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["ts", "price", "size", "ask", "bid"])
    if tbbo is None or tbbo.empty or "price" not in tbbo.columns:
        return empty
    df = tbbo
    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    else:
        return empty
    price = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype="float64")
    if "size" in df.columns:
        size = pd.to_numeric(df["size"], errors="coerce").to_numpy(dtype="float64")
        size = np.where(np.isfinite(size) & (size > 0.0), size, 0.0)
    else:
        size = np.ones(len(df), dtype="float64")
    ok = np.isfinite(price) & (size > 0.0)
    if not bool(ok.any()):
        return empty
    side = df["side"] if "side" in df.columns else None
    if side is not None:
        side = side.iloc[np.flatnonzero(ok)] if hasattr(side, "iloc") else pd.Series(side)[ok]
    else:
        side = None
    price = price[ok]
    size = size[ok]
    ts = pd.DatetimeIndex(ts)[ok]
    ask, bid = _aggressor_volumes(size, side)
    out = pd.DataFrame(
        {"ts": ts, "price": price, "size": size, "ask": ask, "bid": bid},
    )
    return out.sort_values("ts", kind="mergesort")


def _right_bucket(ts: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Stamp prints in ``[T, T+freq)`` at the close ``T+freq`` (ADR-041)."""
    left = ts.floor(freq)
    return left + pd.to_timedelta(freq)


def _footprint_from_group(
    g: pd.DataFrame,
    *,
    close_ts: pd.Timestamp,
    timeframe: str,
    tick_size: float,
) -> CandleFootprint | None:
    px = g["price"].to_numpy(dtype="float64")
    size = g["size"].to_numpy(dtype="float64")
    ask = g["ask"].to_numpy(dtype="float64")
    bid = g["bid"].to_numpy(dtype="float64")
    ticks = np.round(px / tick_size) * tick_size
    feats = describe_regions(ticks, size, ask, bid)
    ladder = _dense_ladder(ticks, size, ask, bid, tick_size)
    if ladder is None:
        prices = np.array([], dtype="float64")
        a = b = total = prices
    else:
        prices, a, b, total = ladder
    return CandleFootprint(
        close_ts=pd.Timestamp(close_ts),
        timeframe=timeframe,
        open=float(feats["open"]),
        high=float(feats["high"]),
        low=float(feats["low"]),
        close=float(feats["close"]),
        volume=float(size.sum()),
        body_low=float(feats["body_low"]),
        body_high=float(feats["body_high"]),
        prices=prices,
        ask=a,
        bid=b,
        total=total,
        poc=float(feats["poc"]),
        features={k: feats[k] for k in (*_REGION_SUFFIXES, "poc_loc")},
    )


def compute_footprints(
    tbbo: pd.DataFrame,
    timeframe: str = "1min",
    *,
    tick_size: float = NQ_TICK,
) -> list[CandleFootprint]:
    """One ``CandleFootprint`` per *closed* *timeframe* bucket (right-labeled)."""
    prints = _prepare_prints(tbbo)
    if prints.empty:
        return []
    ts = pd.DatetimeIndex(prints["ts"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    prints = prints.copy()
    prints["bucket"] = _right_bucket(ts, timeframe)
    out: list[CandleFootprint] = []
    for close_ts, g in prints.groupby("bucket", sort=True):
        fp = _footprint_from_group(
            g, close_ts=close_ts, timeframe=timeframe, tick_size=tick_size,
        )
        if fp is not None:
            out.append(fp)
    return out


def viz_payload(
    tbbo: pd.DataFrame,
    timeframe: str = "1min",
    *,
    tick_size: float = NQ_TICK,
) -> dict[str, Any]:
    """Session overlay: closed-candle ladders + wick/body volume."""
    fps = compute_footprints(tbbo, timeframe, tick_size=tick_size)
    return {
        "timeframe": timeframe,
        "label": tf_label(timeframe),
        "tick": float(tick_size),
        "closed_bar": True,
        "n": len(fps),
        "candles": [fp.to_viz() for fp in fps],
    }


def _features_frame(
    footprints: list[CandleFootprint],
    timeframe: str,
) -> pd.DataFrame:
    lab = tf_label(timeframe)
    cols = list(feature_names((timeframe,)))
    if not footprints:
        return pd.DataFrame(columns=cols)
    rows = []
    index = []
    emit_poc = lab == "1m"
    for fp in footprints:
        row = {f"vf_{lab}_{k}": fp.features[k] for k in _REGION_SUFFIXES}
        if emit_poc:
            row["vf_1m_poc_loc"] = fp.features["poc_loc"]
        rows.append(row)
        index.append(fp.close_ts)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(index, tz="UTC"))
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    if emit_poc and "vf_1m_poc_loc" in out.columns:
        # EMA on closed 1-min POC loc (not on the 5s ffill).
        out["vf_1m_poc_ema_5"] = (
            out["vf_1m_poc_loc"]
            .ewm(span=POC_EMA_SPAN, adjust=False, min_periods=POC_EMA_SPAN)
            .mean()
        )
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].astype(np.float32)


def compute(
    tbbo: pd.DataFrame,
    grid: pd.DatetimeIndex | None = None,
    *,
    timeframes: tuple[str, ...] | list[str] = DEFAULT_TIMEFRAMES,
    tick_size: float = NQ_TICK,
) -> pd.DataFrame:
    """Closed-candle footprint scalars, optionally stale-forwarded onto *grid*.

    Default timeframes: 1-min and 5-min. Each value first appears at that
    bucket's close (right label), then holds until the next close of the
    same timeframe. ``vf_1m_poc_ema_5`` is a 5-bar EMA of closed 1-min
    ``vf_1m_poc_loc`` (span=5), then ffilled.
    """
    tfs = tuple(timeframes) if timeframes else DEFAULT_TIMEFRAMES
    names = feature_names(tfs)
    as_of = None
    if grid is not None and len(grid):
        as_of = pd.Timestamp(pd.DatetimeIndex(grid).max())
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")
    parts: list[pd.DataFrame] = []
    for tf in tfs:
        fps = compute_footprints(tbbo, tf, tick_size=tick_size)
        if as_of is not None:
            fps = [fp for fp in fps if pd.Timestamp(fp.close_ts) <= as_of]
        parts.append(_features_frame(fps, tf))
    if not parts:
        idx = grid if grid is not None else pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(index=idx, columns=list(names), dtype=np.float32)

    out = parts[0]
    for extra in parts[1:]:
        if extra.empty:
            continue
        if out.empty:
            out = extra
            continue
        out = out.join(extra, how="outer")
    if grid is not None:
        out = align_to_grid(out, grid).ffill()
    for c in names:
        if c not in out.columns:
            out[c] = np.nan
    if out.empty:
        idx = grid if grid is not None else pd.DatetimeIndex([], tz="UTC")
        return pd.DataFrame(index=idx, columns=list(names), dtype=np.float32)
    return out[list(names)].astype(np.float32)


def build_vf_patch_frame(
    tbbo: pd.DataFrame,
    grid: pd.DatetimeIndex,
    *,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    tick_size: float = NQ_TICK,
) -> pd.DataFrame:
    """Align ``vf_*`` columns to an existing day parquet index (patch path)."""
    return compute(tbbo, grid=grid, timeframes=timeframes, tick_size=tick_size)
