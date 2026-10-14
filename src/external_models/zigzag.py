"""Thinkorswim ZigZagHighLow replica (ADR-063).

The ThinkScript study has two layers:

* **Causal ``state``** (init → undefined → uptrend / downtrend). A pullback
  from the swing high of ``prevMaxH * hlPivot + absReversal`` flips to
  downtrend; a bounce from the swing low of the symmetric distance flips
  to uptrend. ``hlPivot`` is ``percentageReversal/100``, plus
  ``WildersAverage(TrueRange) / close * atrReversal`` when ATR reversal is
  on. This layer does **not** look ahead and is the trading signal.
* **Plot ``ZZ``** — TOS confirms which bar of a swing was the last extreme
  with a forward ``fold`` / ``GetValue(..., -i)`` (the line *repaints* on
  the open swing). This port writes the same confirmed vertices when the
  opposite trend starts, plus the open swing's current extreme, so a
  historical chart matches TOS without using future bars as signals.

Per-bar side: ``+1`` uptrend, ``-1`` downtrend, ``0`` init/undefined.
Chart arrows use that side on every bar (green up / red down).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.psar import _rma, _true_range

STATE_INIT = 0
STATE_UNDEF = 1
STATE_UP = 2
STATE_DN = 3


def _abs_reversal(params: dict[str, Any]) -> float:
    absolute = float(params.get("absolute_reversal") or 0.0)
    if absolute != 0.0:
        return absolute
    ticks = float(params.get("tick_reversal") or 0)
    tick_size = float(params.get("tick_size") or 0.25)
    return ticks * tick_size


def zzhl_on_closed(
    ohlc: pd.DataFrame,
    *,
    percentage_reversal: float = 2.5,
    absolute_reversal: float = 0.0,
    atr_length: int = 5,
    atr_reversal: float = 1.0,
    tick_reversal: int = 0,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """TOS ZigZagHighLow state machine on already-closed OHLC."""
    cols = (
        "signal", "state", "zz", "max_price_h", "min_price_l",
        "hl_pivot", "reversal_pts", "atr", "new_max", "new_min",
    )
    if ohlc is None or ohlc.empty:
        return pd.DataFrame(columns=list(cols))
    high = ohlc["high"].to_numpy(dtype="float64")
    low = ohlc["low"].to_numpy(dtype="float64")
    close = ohlc["close"].to_numpy(dtype="float64")
    n = int(close.size)
    empty = {c: np.full(n, np.nan, dtype="float64") for c in cols}
    empty["signal"] = np.zeros(n, dtype="int8")
    empty["state"] = np.full(n, STATE_INIT, dtype="int8")
    empty["new_max"] = np.zeros(n, dtype=bool)
    empty["new_min"] = np.zeros(n, dtype=bool)
    if n == 0:
        return pd.DataFrame(empty, index=ohlc.index)

    atr_len = max(1, int(atr_length))
    atr = _rma(_true_range(high, low, close), atr_len)
    pct = float(percentage_reversal) / 100.0
    atr_rev = float(atr_reversal)
    abs_rev = _abs_reversal({
        "absolute_reversal": absolute_reversal,
        "tick_reversal": tick_reversal,
        "tick_size": tick_size,
    })

    signal = np.zeros(n, dtype="int8")
    state_a = np.full(n, STATE_INIT, dtype="int8")
    zz = np.full(n, np.nan, dtype="float64")
    max_h = np.full(n, np.nan, dtype="float64")
    min_l = np.full(n, np.nan, dtype="float64")
    hl_a = np.full(n, np.nan, dtype="float64")
    need_a = np.full(n, np.nan, dtype="float64")
    new_max = np.zeros(n, dtype=bool)
    new_min = np.zeros(n, dtype=bool)

    prev_state = STATE_INIT
    prev_max = np.nan
    prev_min = np.nan
    last_high_i = 0
    last_low_i = 0
    started = False

    for i in range(n):
        px_h = high[i]
        px_l = low[i]
        if atr_rev != 0.0:
            c = close[i]
            a = atr[i]
            if np.isfinite(a) and np.isfinite(c) and c != 0.0:
                hl = pct + (a / c) * atr_rev
            else:
                hl = np.nan
        else:
            hl = pct
        hl_a[i] = hl
        finite_hl = np.isfinite(hl)
        need = (np.nan if not finite_hl else abs_rev)  # filled after extremes

        if prev_state == STATE_INIT:
            cur_max = px_h
            cur_min = px_l
            is_new_max = True
            is_new_min = True
            cur_state = STATE_UNDEF
        elif prev_state == STATE_UNDEF:
            if np.isfinite(px_h) and np.isfinite(prev_max) and px_h >= prev_max:
                cur_state = STATE_UP
                cur_max = px_h
                cur_min = prev_min
                is_new_max = True
                is_new_min = False
            elif np.isfinite(px_l) and np.isfinite(prev_min) and px_l <= prev_min:
                cur_state = STATE_DN
                cur_max = prev_max
                cur_min = px_l
                is_new_max = False
                is_new_min = True
            else:
                cur_state = STATE_UNDEF
                cur_max = prev_max
                cur_min = prev_min
                is_new_max = False
                is_new_min = False
        elif prev_state == STATE_UP:
            reverse = (
                finite_hl
                and np.isfinite(px_l)
                and np.isfinite(prev_max)
                and px_l <= prev_max - prev_max * hl - abs_rev
            )
            if reverse:
                cur_state = STATE_DN
                cur_max = prev_max
                cur_min = px_l
                is_new_max = False
                is_new_min = True
            else:
                cur_state = STATE_UP
                if np.isfinite(px_h) and np.isfinite(prev_max) and px_h >= prev_max:
                    cur_max = px_h
                    is_new_max = True
                else:
                    cur_max = prev_max
                    is_new_max = False
                cur_min = prev_min
                is_new_min = False
        else:
            reverse = (
                finite_hl
                and np.isfinite(px_h)
                and np.isfinite(prev_min)
                and px_h >= prev_min + prev_min * hl + abs_rev
            )
            if reverse:
                cur_state = STATE_UP
                cur_max = px_h
                cur_min = prev_min
                is_new_max = True
                is_new_min = False
            else:
                cur_state = STATE_DN
                cur_max = prev_max
                is_new_max = False
                if np.isfinite(px_l) and np.isfinite(prev_min) and px_l <= prev_min:
                    cur_min = px_l
                    is_new_min = True
                else:
                    cur_min = prev_min
                    is_new_min = False

        if finite_hl:
            if cur_state == STATE_UP and np.isfinite(cur_max):
                need = cur_max * hl + abs_rev
            elif cur_state == STATE_DN and np.isfinite(cur_min):
                need = cur_min * hl + abs_rev
            elif np.isfinite(cur_max):
                need = cur_max * hl + abs_rev
            else:
                need = abs_rev
        need_a[i] = need

        if is_new_max:
            last_high_i = i
        if is_new_min:
            last_low_i = i

        if not started and cur_state in (STATE_UP, STATE_DN):
            started = True
            if cur_state == STATE_UP and np.isfinite(cur_min):
                zz[last_low_i] = cur_min
            elif cur_state == STATE_DN and np.isfinite(cur_max):
                zz[last_high_i] = cur_max
        elif prev_state == STATE_UP and cur_state == STATE_DN:
            if np.isfinite(cur_max):
                zz[last_high_i] = cur_max
        elif prev_state == STATE_DN and cur_state == STATE_UP:
            if np.isfinite(cur_min):
                zz[last_low_i] = cur_min

        state_a[i] = cur_state
        max_h[i] = cur_max
        min_l[i] = cur_min
        new_max[i] = is_new_max
        new_min[i] = is_new_min
        if cur_state == STATE_UP:
            signal[i] = 1
        elif cur_state == STATE_DN:
            signal[i] = -1
        else:
            signal[i] = 0
        prev_state = cur_state
        prev_max = cur_max
        prev_min = cur_min

    if n:
        if prev_state == STATE_UP and np.isfinite(prev_max):
            zz[last_high_i] = prev_max
        elif prev_state == STATE_DN and np.isfinite(prev_min):
            zz[last_low_i] = prev_min

    return pd.DataFrame({
        "signal": signal,
        "state": state_a,
        "zz": zz,
        "max_price_h": max_h,
        "min_price_l": min_l,
        "hl_pivot": hl_a,
        "reversal_pts": need_a,
        "atr": atr,
        "new_max": new_max,
        "new_min": new_min,
    }, index=ohlc.index)


def zzhl_signals(ohlc: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-bar ZigZag side: ``+1`` uptrend, ``-1`` downtrend, ``0`` undefined."""
    from src.external_models.catalog import coerce_params
    from src.external_models.pine_sar_ema import normalize_ohlc

    p = coerce_params("zigzag", params)
    df = normalize_ohlc(ohlc)
    n = len(df)
    if n == 0:
        out = pd.DataFrame({
            "signal": np.zeros(0, dtype="int8"),
            "state": np.zeros(0, dtype="int8"),
            "zz": np.zeros(0, dtype="float64"),
            "max_price_h": np.zeros(0, dtype="float64"),
            "min_price_l": np.zeros(0, dtype="float64"),
            "hl_pivot": np.zeros(0, dtype="float64"),
            "reversal_pts": np.zeros(0, dtype="float64"),
            "atr": np.zeros(0, dtype="float64"),
            "tagged": np.zeros(0, dtype=bool),
            "exit_at_sar": np.zeros(0, dtype=bool),
            "prev_sar": np.zeros(0, dtype="float64"),
            "prev_trend": np.zeros(0, dtype="float64"),
        }, index=df.index)
        out.attrs["model_id"] = "zigzag"
        out.attrs["engine"] = "zigzag"
        out.attrs["params"] = p
        return out
    core = zzhl_on_closed(
        df,
        percentage_reversal=float(p["percentage_reversal"]),
        absolute_reversal=float(p["absolute_reversal"]),
        atr_length=int(p["atr_length"]),
        atr_reversal=float(p["atr_reversal"]),
        tick_reversal=int(p["tick_reversal"]),
        tick_size=float(p["tick_size"]),
    )
    out = pd.DataFrame({
        "signal": core["signal"].to_numpy(dtype="int8"),
        "state": core["state"].to_numpy(dtype="int8"),
        "zz": core["zz"].to_numpy(dtype="float64"),
        "max_price_h": core["max_price_h"].to_numpy(dtype="float64"),
        "min_price_l": core["min_price_l"].to_numpy(dtype="float64"),
        "hl_pivot": core["hl_pivot"].to_numpy(dtype="float64"),
        "reversal_pts": core["reversal_pts"].to_numpy(dtype="float64"),
        "atr": core["atr"].to_numpy(dtype="float64"),
        "tagged": np.zeros(n, dtype=bool),
        "exit_at_sar": np.zeros(n, dtype=bool),
        "prev_sar": np.full(n, np.nan),
        "prev_trend": np.full(n, np.nan),
    }, index=df.index)
    out.attrs["model_id"] = "zigzag"
    out.attrs["engine"] = "zigzag"
    out.attrs["params"] = p
    return out
