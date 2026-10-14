"""Timed one-shot PSAR entry (ADR-096).

One new position per session: at a Chicago clock time or N bars after
session open, enter in the direction of the last **closed** 1-hour or
1-day Parabolic SAR. The side is snapshotted at the trigger and held
until session end so the shared book can manage PT / SL / flatten
without a signal-exit. A later HTF SAR flip does not reverse.

Signals are computed from finest 1-minute OHLC (so the HTF trail has
the bars it needs) then placed on the book index. ``once_per_session``
is always on — this is a single-trade rule, not a re-entry book.
"""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.cme_holidays import is_trading_day
from src.external_models.catalog import (
    coerce_book_params,
    coerce_params,
    _hhmm_minutes,
)
from src.external_models.pine_sar_ema import normalize_ohlc
from src.external_models.timeframe import (
    align_signal_frame,
    coerce_candle_tf,
    resample_ohlc,
)
from src.features.psar import sar_on_closed


def _empty_frame(index, params: dict[str, Any]) -> pd.DataFrame:
    n = len(index)
    out = pd.DataFrame({
        "signal": np.zeros(n, dtype="int8"),
        "trigger": np.zeros(n, dtype=bool),
        "htf_side": np.zeros(n, dtype="int8"),
        "sar": np.full(n, np.nan),
        "trend": np.full(n, np.nan),
        "tagged": np.zeros(n, dtype=bool),
        "exit_at_sar": np.zeros(n, dtype=bool),
        "prev_sar": np.full(n, np.nan),
        "prev_trend": np.full(n, np.nan),
    }, index=index)
    out.attrs["model_id"] = "timed_psar"
    out.attrs["engine"] = "timed_psar"
    out.attrs["params"] = params
    out.attrs["once_per_session"] = True
    return out


def _htf_side_frame(finest: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    direction_tf = coerce_candle_tf(params["direction_tf"])
    htf = resample_ohlc(finest, direction_tf)
    n = len(htf)
    empty = pd.DataFrame({
        "signal": np.zeros(n, dtype="int8"),
        "sar": np.full(n, np.nan),
        "trend": np.full(n, np.nan),
        "tagged": np.zeros(n, dtype=bool),
    }, index=htf.index if n else finest.index[:0])
    if n < 2:
        return empty
    sar = sar_on_closed(
        htf,
        start=float(params["start"]),
        increment=float(params["increment"]),
        maximum=float(params["maximum"]),
    )
    trend = sar["trend"].to_numpy(dtype="float64")
    side = np.zeros(n, dtype="int8")
    side[trend > 0] = 1
    side[trend < 0] = -1
    return pd.DataFrame({
        "signal": side,
        "sar": sar["sar"].to_numpy(dtype="float64"),
        "trend": trend,
        "tagged": np.zeros(n, dtype=bool),
    }, index=htf.index)


def timed_psar_signals(
    finest: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    book_ohlc: pd.DataFrame | None = None,
    book: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Book-aligned one-shot side: ``+1`` / ``-1`` from the trigger through session.

    *finest* should be 1-minute (or finer) OHLC so 1-hour / 1-day SAR can
    warm up. *book_ohlc* is the display / PT / SL bar; defaults to *finest*.
    """
    p = coerce_params("timed_psar", params)
    book_p = coerce_book_params(book)
    src = normalize_ohlc(finest)
    book_df = normalize_ohlc(book_ohlc) if book_ohlc is not None else src
    out = _empty_frame(book_df.index, p)
    if src.empty or book_df.empty:
        return out

    direction_tf = coerce_candle_tf(p["direction_tf"])
    book_tf = coerce_candle_tf(book_p.get("timeframe") or "1min")
    htf = _htf_side_frame(src, p)
    aligned = align_signal_frame(htf, book_df.index, direction_tf, book_tf)
    htf_side = aligned["signal"].to_numpy(dtype="int8")
    sar_a = (
        aligned["sar"].to_numpy(dtype="float64")
        if "sar" in aligned.columns else np.full(len(book_df), np.nan)
    )
    trend_a = (
        aligned["trend"].to_numpy(dtype="float64")
        if "trend" in aligned.columns else np.full(len(book_df), np.nan)
    )

    tz = ZoneInfo(str(book_p.get("timezone") or "America/Chicago"))
    local = book_df.index.tz_convert(tz)
    minutes = (local.hour.to_numpy() * 60 + local.minute.to_numpy()).astype(np.int32)
    dates = np.array([t.date() for t in local], dtype=object)
    sess0 = _hhmm_minutes(str(book_p["session_start"]))
    sess1 = _hhmm_minutes(str(book_p["session_end"]))
    in_session = (minutes >= sess0) & (minutes < sess1)
    mode = str(p.get("entry_mode") or "clock")
    entry_min = _hhmm_minutes(str(p.get("entry_time") or book_p["session_start"]))
    bars_from = int(p.get("bars_from_open") or 0)

    n = len(book_df)
    signal = np.zeros(n, dtype="int8")
    trigger = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        d = dates[i]
        j = i + 1
        while j < n and dates[j] == d:
            j += 1
        if not is_trading_day(d):
            i = j
            continue
        sess_idx = [k for k in range(i, j) if bool(in_session[k])]
        if not sess_idx:
            i = j
            continue
        t_i: int | None
        if mode == "bars_from_open":
            if bars_from < 0 or bars_from >= len(sess_idx):
                i = j
                continue
            t_i = sess_idx[bars_from]
        else:
            t_i = None
            for k in sess_idx:
                if int(minutes[k]) >= entry_min:
                    t_i = k
                    break
            if t_i is None:
                i = j
                continue
        side = int(htf_side[t_i])
        if side == 0:
            i = j
            continue
        last = sess_idx[-1]
        signal[t_i:last + 1] = np.int8(side)
        trigger[t_i] = True
        i = j

    out["signal"] = signal
    out["trigger"] = trigger
    out["htf_side"] = htf_side
    out["sar"] = sar_a
    out["trend"] = trend_a
    out.attrs["direction_tf"] = direction_tf
    out.attrs["book_timeframe"] = book_tf
    return out
