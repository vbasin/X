"""Candle timeframes for the EXT book and per-engine stack legs.

Book ``timeframe`` is the display / ATR / PT / SL bar. Each stacked engine
may run on a coarser or finer closed bar; signals are aligned onto the book
index without look-ahead (ADR-064 / ADR-065).
"""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.psar import normalize_timeframe

CANDLE_TIMEFRAMES: tuple[str, ...] = ("1min", "5min", "15min", "1hour", "1day")
_TF_SECONDS = {
    "1min": 60, "5min": 300, "15min": 900,
    "1hour": 3600, "1day": 86400,
}
_PANDAS_FREQ = {
    "1min": "1min", "5min": "5min", "15min": "15min", "1hour": "1h",
}
_TF_ALIASES = {
    "1m": "1min", "1min": "1min",
    "5m": "5min", "5min": "5min",
    "15m": "15min", "15min": "15min",
    "1h": "1hour", "1hour": "1hour", "60m": "1hour", "60min": "1hour",
    "h": "1hour", "hour": "1hour",
    "1d": "1day", "1day": "1day", "d": "1day", "day": "1day",
}
_CT = ZoneInfo("America/Chicago")
_DAY_LABEL = pd.Timedelta(hours=8, minutes=30)
_WARMUP_DAYS = {"1min": 2, "5min": 2, "15min": 2, "1hour": 10, "1day": 40}


def coerce_candle_tf(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in _TF_ALIASES:
        return _TF_ALIASES[raw]
    tf = normalize_timeframe(raw) if raw else ""
    if tf not in CANDLE_TIMEFRAMES:
        raise ValueError(f"candle timeframe must be one of {list(CANDLE_TIMEFRAMES)}")
    return tf


def tf_seconds(timeframe: str) -> int:
    tf = coerce_candle_tf(timeframe)
    return int(_TF_SECONDS[tf])


def warmup_days_for(*timeframes: Any) -> int:
    """Days of 1-minute lead-in so HTF engines (EMA-20 on 1day) can warm up."""
    n = 2
    for raw in timeframes:
        if raw is None or str(raw).strip() == "":
            continue
        try:
            tf = coerce_candle_tf(raw)
        except ValueError:
            continue
        n = max(n, int(_WARMUP_DAYS.get(tf, 2)))
    return n


def _utc_index(idx) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def _bar_starts(idx: pd.DatetimeIndex, timeframe: str) -> pd.DatetimeIndex:
    """Left-edge label for each timestamp on *timeframe* (UTC).

    Intraday TFs floor on the UTC clock (CT hour boundaries match). ``1day``
    is the Chicago calendar day labeled 08:30 CT so the bar falls inside the
    default RTH session window (ADR-065).
    """
    tf = coerce_candle_tf(timeframe)
    idx = _utc_index(idx)
    if tf == "1day":
        local = idx.tz_convert(_CT)
        labels = local.normalize() + _DAY_LABEL
        return labels.tz_convert("UTC")
    return idx.floor(_PANDAS_FREQ[tf])


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Left-labeled closed bars from a finer (usually 1-minute) OHLC frame."""
    if df is None or df.empty:
        return df
    tf = coerce_candle_tf(timeframe)
    if tf == "1min":
        return df
    agg: dict[str, str] = {}
    for col, how in (
        ("open", "first"), ("high", "max"), ("low", "min"),
        ("close", "last"), ("volume", "sum"), ("symbol", "last"),
    ):
        if col in df.columns:
            agg[col] = how
    if not agg:
        return df
    src = df.copy()
    src.index = _utc_index(src.index)
    if tf == "1day":
        out = src.groupby(_bar_starts(src.index, tf), sort=True).agg(agg)
        out.index = pd.DatetimeIndex(out.index, tz="UTC")
        out.index.name = df.index.name
    else:
        out = src.resample(_PANDAS_FREQ[tf], label="left", closed="left").agg(agg)
    if "close" in out.columns:
        out = out.dropna(subset=["close"])
    return out


def align_signal_frame(
    frame: pd.DataFrame,
    book_index: pd.DatetimeIndex,
    src_tf: str,
    book_tf: str,
) -> pd.DataFrame:
    """Project an engine frame onto the book index without look-ahead.

    Slower engine (e.g. 15min → 1min): the left-labeled source bar at *T*
    covers ``[T, T+src)`` and is only known at ``T + src − book``. That
    value is then forward-filled. ``tagged`` and ``signal_fresh`` pulse on
    the availability bar only.

    Faster engine (e.g. 1min → 15min): last value in each book bucket.
    Equal timeframes: reindex.
    """
    src_tf = coerce_candle_tf(src_tf)
    book_tf = coerce_candle_tf(book_tf)
    cols = list(frame.columns)
    empty = pd.DataFrame(index=book_index)
    for col in cols:
        if col == "signal":
            empty[col] = np.zeros(len(book_index), dtype="int8")
        elif col in ("tagged", "exit_at_sar", "signal_fresh"):
            empty[col] = np.zeros(len(book_index), dtype=bool)
        else:
            empty[col] = np.nan
    empty["signal_fresh"] = np.zeros(len(book_index), dtype=bool)
    if frame is None or frame.empty or book_index is None or len(book_index) == 0:
        empty.attrs.update(getattr(frame, "attrs", {}) or {})
        empty.attrs["timeframe"] = src_tf
        empty.attrs["book_timeframe"] = book_tf
        return empty

    src_sec = tf_seconds(src_tf)
    book_sec = tf_seconds(book_tf)
    src = frame.copy()
    src.index = _utc_index(src.index)
    src = src.sort_index()
    src = src.loc[~src.index.duplicated(keep="last")]

    if src_sec == book_sec:
        aligned = src.reindex(book_index)
        aligned["signal_fresh"] = aligned.index.isin(src.index)
    elif src_sec < book_sec:
        agg: dict[str, Any] = {}
        for col in src.columns:
            if col in ("tagged", "exit_at_sar"):
                agg[col] = "max"
            else:
                agg[col] = "last"
        grouped = src.groupby(_bar_starts(src.index, book_tf), sort=True).agg(agg)
        grouped.index = pd.DatetimeIndex(grouped.index, tz="UTC")
        aligned = grouped.reindex(book_index)
        aligned["signal_fresh"] = True
    else:
        delay = pd.Timedelta(seconds=src_sec - book_sec)
        moved = src.copy()
        moved.index = moved.index + delay
        pulse_cols = {"tagged"}
        value_cols = [c for c in moved.columns if c not in pulse_cols]
        aligned = moved[value_cols].reindex(book_index, method="ffill")
        fresh = np.zeros(len(book_index), dtype=bool)
        tagged = np.zeros(len(book_index), dtype=bool)
        book_ns = book_index.asi8
        moved_ns = moved.index.asi8
        loc = np.searchsorted(book_ns, moved_ns, side="left")
        tag_src = (
            moved["tagged"].to_numpy(dtype=bool)
            if "tagged" in moved.columns
            else np.zeros(len(moved), dtype=bool)
        )
        for j, pos in enumerate(loc):
            if pos >= len(book_index):
                continue
            fresh[pos] = True
            if bool(tag_src[j]):
                tagged[pos] = True
        aligned["signal_fresh"] = fresh
        aligned["tagged"] = tagged

    if "signal" in aligned.columns:
        aligned["signal"] = pd.to_numeric(aligned["signal"], errors="coerce").fillna(0).astype("int8")
    for col in ("tagged", "exit_at_sar", "signal_fresh"):
        if col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0).astype(bool)
    aligned.attrs.update(getattr(frame, "attrs", {}) or {})
    aligned.attrs["timeframe"] = src_tf
    aligned.attrs["book_timeframe"] = book_tf
    return aligned


def retag_sar_on_book(frame: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """Recompute SAR ``tagged`` from the ffilled trail vs book high/low."""
    if frame is None or frame.empty or "prev_sar" not in frame.columns:
        return frame
    if ohlc is None or ohlc.empty:
        return frame
    n = len(frame)
    high = ohlc["high"].to_numpy(dtype="float64")
    low = ohlc["low"].to_numpy(dtype="float64")
    if len(high) != n:
        return frame
    prev_sar = frame["prev_sar"].to_numpy(dtype="float64")
    prev_trend = (
        frame["prev_trend"].to_numpy(dtype="float64")
        if "prev_trend" in frame.columns
        else np.full(n, np.nan)
    )
    tagged = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(prev_sar[i]) or not np.isfinite(prev_trend[i]):
            continue
        if prev_trend[i] > 0:
            tagged[i] = np.isfinite(low[i]) and low[i] <= prev_sar[i]
        elif prev_trend[i] < 0:
            tagged[i] = np.isfinite(high[i]) and high[i] >= prev_sar[i]
    out = frame.copy()
    out["tagged"] = tagged
    return out
