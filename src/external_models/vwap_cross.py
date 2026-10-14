"""VWAP crossover engine for /NQ.

Strategy (long side — short is mirror):

* On the close of candle ``N`` (the *crossing candle*), evaluate two
  session-anchored VWAPs:

  - ``vwap_prev`` — VWAP at the close of candle ``N-1`` (i.e. VWAP "known" at
    the moment candle ``N`` opened);
  - ``vwap_now``  — VWAP at the close of candle ``N`` (updated by candle N's
    own volume-weighted contribution).

* A long *cross-up* is armed when:

  1. ``open[N]  <  vwap_prev``  — the candle opened below the known VWAP;
  2. ``close[N] >  vwap_now``   — the candle closed above the just-updated VWAP;
  3. ``close[N-1] > open[N-1]`` — the preceding candle was green;
  4. VWAP slope over the last ``vwap_slope_lookback`` bars is
     ``>= min_vwap_slope`` (default 0). Slope is **backward-looking**
     for every entry mode so the strategy is live-safe:

     - ``close_confirm``  → ``(V[N] − V[N − lookback]) / lookback``
     - ``early_cross`` / ``close_confirm_early_fill``
                        → ``(V[N-1] − V[N-1 − lookback]) / lookback``

     Default ``lookback = 1`` = the two-point instantaneous slope
     ``V[t] − V[t-1]``.

* Entry is filled at ``open[N+1]``.
* The stop-loss can be:

  - ``crossing_open`` — the opening price of candle N;
  - ``vwap_offset``   — ``vwap_now - vwap_stop_offset`` (points below VWAP);
  - ``worse_of`` (default) — whichever of the two is farther from entry,
    so the trade only fires if both invalidation points agree.

* Profit target is either an absolute point distance or a multiple of the
  stop-loss range (``profit_target_mode``).

Chop guard: two independent knobs are exposed so the operator can dial the
strategy out of whipsaw sessions:

- ``max_session_crosses`` — a hard per-session cap on the number of VWAP
  cross events (open/close disagree with VWAP) that have already occurred
  by candle N; once exceeded, no more entries fire that day.
- ``chop_lookback`` / ``max_recent_crosses`` — a rolling window; if too
  many cross events have already fired inside ``chop_lookback`` candles
  ending at N-1, the signal on N is suppressed. Set the cap to a large
  number to disable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CT = ZoneInfo("America/Chicago")
DEFAULT_POINT_VALUE = 20.0
TICK = 0.25

STOP_MODES: tuple[str, ...] = ("crossing_open", "vwap_offset", "worse_of")
PT_MODES: tuple[str, ...] = ("absolute", "stop_multiple")


def _hm_minutes(hhmm: str) -> int:
    h, m = str(hhmm).split(":")[:2]
    return int(h) * 60 + int(m)


def _ct_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    df = ohlc.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    local = df.index.tz_convert(CT)
    df["_ct_min"] = local.hour * 60 + local.minute
    df["_date"] = local.date
    return df


def _session_vwap(df: pd.DataFrame, *, session_start_min: int) -> np.ndarray:
    """Session-anchored VWAP series aligned with ``df.index``.

    VWAP resets at ``session_start_min`` CT each Chicago day. Bars outside the
    session contribute nothing (kept as NaN forwarded from the last in-session
    value so post-session bars still see the day's closing VWAP).
    """
    typ = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy("float64")
    vol = df["volume"].fillna(0.0).to_numpy("float64")
    ct_min = df["_ct_min"].to_numpy()
    dates = df["_date"].to_numpy()
    out = np.full(len(df), np.nan, dtype="float64")
    cur_date = None
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(len(df)):
        d = dates[i]
        if d != cur_date:
            cur_date = d
            cum_pv = 0.0
            cum_v = 0.0
        if int(ct_min[i]) < session_start_min:
            continue
        v = vol[i]
        if math.isfinite(v) and v > 0 and math.isfinite(typ[i]):
            cum_pv += typ[i] * v
            cum_v += v
        if cum_v > 0:
            out[i] = cum_pv / cum_v
    return out


@dataclass
class _Trade:
    session_date: Any
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    entry_px: float
    exit_px: float
    stop_px: float
    target_px: float
    stop_pts: float
    target_pts: float
    exit_reason: str
    mae_points: float
    mfe_points: float
    holding_bars: int
    target_ref: str = "absolute"


def _resolve_stop(
    *, side: int, entry_px: float, crossing_open: float, vwap_now: float,
    stop_mode: str, vwap_offset: float,
) -> float:
    if stop_mode == "crossing_open":
        stop = crossing_open
    elif stop_mode == "vwap_offset":
        stop = vwap_now - side * vwap_offset
    else:  # worse_of
        by_open = crossing_open
        by_vwap = vwap_now - side * vwap_offset
        # farther from entry = safer/looser invalidation
        stop = min(by_open, by_vwap) if side > 0 else max(by_open, by_vwap)
    return float(stop)


def _resolve_target(
    *, side: int, entry_px: float, stop_px: float,
    pt_mode: str, pt_pts: float, pt_mult: float,
) -> float:
    if pt_mode == "absolute":
        dist = float(pt_pts)
    else:
        dist = abs(entry_px - stop_px) * float(pt_mult)
    return float(entry_px + side * dist)


def _round_tick(px: float) -> float:
    return round(px / TICK) * TICK


def _wilder_atr(bars: pd.DataFrame, length: int) -> np.ndarray:
    """Wilder-smoothed ATR over ``length`` bars, aligned with ``bars.index``."""
    h = bars["high"].to_numpy("float64")
    l = bars["low"].to_numpy("float64")
    c = bars["close"].to_numpy("float64")
    n = len(bars)
    if n == 0:
        return np.array([], dtype="float64")
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.full(n, np.nan, dtype="float64")
    if n < length:
        return atr
    atr[length - 1] = float(np.mean(tr[:length]))
    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr


def _ema(x: np.ndarray, length: int) -> np.ndarray:
    """Standard EMA (alpha = 2/(length+1)), seeded by an SMA of the first ``length`` values."""
    n = len(x)
    out = np.full(n, np.nan, dtype="float64")
    if n == 0 or length < 2:
        return out
    alpha = 2.0 / (length + 1.0)
    if n < length:
        return out
    seed = float(np.mean(x[:length]))
    out[length - 1] = seed
    prev = seed
    for i in range(length, n):
        prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _median_bar_seconds(df: pd.DataFrame) -> int:
    if len(df) < 3:
        return 60
    diffs = np.diff(df.index.asi8) // 1_000_000_000
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 60
    return int(np.median(positive))


def _to_1min_view(fine: pd.DataFrame) -> pd.DataFrame:
    """Return a 1-min OHLCV view of ``fine`` (idempotent if already 1-min)."""
    bar_s = _median_bar_seconds(fine)
    if bar_s >= 60:
        return fine
    grouped = fine.groupby(pd.Grouper(freq="1min"))
    agg = pd.DataFrame({
        "open":   grouped["open"].first(),
        "high":   grouped["high"].max(),
        "low":    grouped["low"].min(),
        "close":  grouped["close"].last(),
        "volume": grouped["volume"].sum(),
    }).dropna(subset=["open", "high", "low", "close"])
    local = agg.index.tz_convert(CT)
    agg["_ct_min"] = local.hour * 60 + local.minute
    agg["_date"] = np.array([t.date() for t in local], dtype=object)
    return agg


def _fine_starts_for_1m(bars_1m: pd.DataFrame, fine: pd.DataFrame) -> np.ndarray:
    """Return the fine-series index of the first bar of each 1-min bucket."""
    fine_i64 = fine.index.asi8
    starts = np.searchsorted(fine_i64, bars_1m.index.asi8, side="left")
    return starts.astype("int64")


def _bar_volume_profile_local(
    bars: pd.DataFrame, *, va_pct: float = 0.70, bin_pts: float = 1.0,
) -> tuple[float, float, float]:
    """POC / VAH / VAL from OHLCV bars (volume spread across each bar's range).

    Duplicated from ``orb.py`` to keep the vwap_cross engine standalone.
    """
    if bars is None or bars.empty:
        return (float("nan"),) * 3
    lo = float(bars["low"].min())
    hi = float(bars["high"].max())
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return (float("nan"),) * 3
    n_bins = max(2, min(600, int(round((hi - lo) / bin_pts))))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = np.zeros(n_bins, dtype="float64")
    b_lo = bars["low"].to_numpy("float64")
    b_hi = bars["high"].to_numpy("float64")
    b_v = bars["volume"].to_numpy("float64")
    for lv, hv, v in zip(b_lo, b_hi, b_v):
        if not (math.isfinite(lv) and math.isfinite(hv) and math.isfinite(v)) or v <= 0:
            continue
        i0 = int(np.searchsorted(edges, lv, side="right")) - 1
        i1 = int(np.searchsorted(edges, hv, side="left"))
        i0 = max(0, min(n_bins - 1, i0))
        i1 = max(i0 + 1, min(n_bins, i1))
        hist[i0:i1] += v / float(i1 - i0)
    total = float(hist.sum())
    if total <= 0:
        return (float("nan"),) * 3
    poc_i = int(np.argmax(hist))
    poc = float(centers[poc_i])
    target = total * va_pct
    lo_i = hi_i = poc_i
    acc = float(hist[poc_i])
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1.0
        up = hist[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            acc += float(hist[hi_i])
        else:
            lo_i -= 1
            acc += float(hist[lo_i])
    return poc, float(centers[hi_i]), float(centers[lo_i])


def _prior_session_vp_levels(
    bars_1m: pd.DataFrame, *, session_start_m: int, session_end_m: int, va_pct: float,
) -> dict[Any, dict[str, float]]:
    """Prior-session Lo / VAL / POC / VAH / Hi keyed by *this* session's date."""
    ct = bars_1m["_ct_min"].to_numpy()
    dates = bars_1m["_date"].to_numpy()
    in_sess_mask = (ct >= session_start_m) & (ct < session_end_m)
    session_dates: list[Any] = []
    seen: set = set()
    for d in dates:
        if d not in seen:
            seen.add(d)
            session_dates.append(d)
    out: dict[Any, dict[str, float]] = {}
    for i, d in enumerate(session_dates):
        if i == 0:
            continue
        pd_ = session_dates[i - 1]
        rows = bars_1m[(dates == pd_) & in_sess_mask]
        if rows.empty:
            continue
        hi = float(rows["high"].max())
        lo = float(rows["low"].min())
        poc, vah, val = _bar_volume_profile_local(rows, va_pct=va_pct)
        levels = {"low": lo, "val": val, "poc": poc, "vah": vah, "high": hi}
        levels = {k: v for k, v in levels.items() if math.isfinite(v)}
        if levels:
            out[d] = levels
    return out


def _resolve_vp_target(
    *, side: int, entry_px: float, levels: dict[str, float] | None, fallback_pts: float,
) -> tuple[float, str]:
    if not levels:
        return float(entry_px + side * fallback_pts), "fallback"
    if side > 0:
        cands = [(name, px) for name, px in levels.items() if px > entry_px + TICK]
        if not cands:
            return float(entry_px + side * fallback_pts), "fallback"
        # nearest above
        name, px = min(cands, key=lambda kv: kv[1])
        return float(px), name
    cands = [(name, px) for name, px in levels.items() if px < entry_px - TICK]
    if not cands:
        return float(entry_px + side * fallback_pts), "fallback"
    name, px = max(cands, key=lambda kv: kv[1])
    return float(px), name


def _clamp_stop(
    *, side: int, entry_px: float, stop_px: float, max_stop_pts: float,
) -> float:
    if max_stop_pts <= 0:
        return stop_px
    if abs(entry_px - stop_px) <= max_stop_pts:
        return stop_px
    return float(entry_px - side * max_stop_pts)


def simulate_vwap_cross(
    ohlc: pd.DataFrame, params: dict[str, Any],
) -> dict[str, Any]:
    """Bar-by-bar simulator for the VWAP crossover strategy.

    ``ohlc`` may be 1-min or a finer cadence (e.g. 15s). The signal loop
    always runs on a 1-min view derived from the input; stop / target /
    session_flat fills are resolved on the finest cadence available.
    """
    p = dict(params)
    fine = _ct_frame(ohlc)
    if fine.empty:
        return {"trades": pd.DataFrame(), "skipped": {"empty": 1}, "n_bars": 0,
                "params": p, "vwap": np.array([], dtype="float64")}

    session_start_m = _hm_minutes(p["session_start"])
    session_end_m = _hm_minutes(p["session_end"])
    trade_start_m = _hm_minutes(p["trade_start"])
    trade_end_m = _hm_minutes(p["trade_end"])
    slope_lb = max(1, int(p["vwap_slope_lookback"]))
    slope_direction = str(p.get("vwap_slope_direction") or "backward").lower()
    slope_forward = slope_direction == "forward"
    min_slope = float(p["min_vwap_slope"])
    min_body_clv = float(p.get("min_close_range_frac") or 0.0)
    use_min_clv = bool(p.get("use_min_signal_clv", True))
    use_slope = bool(p.get("use_vwap_slope_filter", True))
    use_preceder = bool(p.get("use_preceding_bar_color", True))
    use_bar_size = bool(p.get("use_bar_size_filter", False))
    bar_size_mult = float(p.get("bar_size_atr_mult") or 0.0)
    atr_lb = max(2, int(p.get("atr_lookback") or 14))
    use_ema_filter = bool(p.get("use_ema_cross_filter", False))
    ema_length = max(2, int(p.get("ema_length") or 6))
    ema_cross_length = max(2, int(p.get("ema_cross_length") or 3))
    min_bars_ema_pt = max(0, int(p.get("min_bars_before_ema_pt") or 0))
    ema_shrink_pct = float(p.get("ema_vwap_shrink_pct") or 0.0)
    trade_long = bool(p["trade_long"])
    trade_short = bool(p["trade_short"])
    stop_mode = str(p["stop_mode"])
    vwap_offset = float(p["vwap_stop_offset"])
    max_stop_pts = float(p.get("max_stop_pts") or 0.0)
    pt_mode = str(p["profit_target_mode"])
    pt_pts = float(p["profit_target_pts"])
    pt_mult = float(p["profit_target_stop_mult"])
    vp_va_pct = float(p.get("vp_va_pct") or 0.70)
    vp_fallback_pts = float(p.get("vp_fallback_pts") or pt_pts)
    max_sess_x = int(p["max_session_crosses"])
    chop_lb = max(0, int(p["chop_lookback"]))
    max_recent_x = int(p["max_recent_crosses"])
    slip = float(p["slippage_points"])
    cost_pts = float(p["cost_per_trade"])
    qty = max(1, int(p["qty"]))
    pv = float(p["point_value"])
    entry_mode = str(p.get("entry_mode") or "close_confirm")
    entry_ticks = max(0, int(p.get("entry_ticks") or 0))
    entry_bar_delay = max(0, int(p.get("entry_bar_delay") or 0))
    # Day-of-week gate: 7-char string "MTWTFSS", "1" = trade, "0" = skip.
    dow_mask_raw = str(p.get("trade_dow_mask") or "1111100")
    dow_mask_raw = (dow_mask_raw + "1111100")[:7]  # pad/truncate to 7
    dow_allowed = tuple(c == "1" for c in dow_mask_raw)

    # Per-side overrides — default to the shared value so unset params inherit.
    long_max_sess_x = int(p.get("long_max_session_crosses", max_sess_x))
    short_max_sess_x = int(p.get("short_max_session_crosses", max_sess_x))
    long_chop_lb = max(0, int(p.get("long_chop_lookback", chop_lb)))
    short_chop_lb = max(0, int(p.get("short_chop_lookback", chop_lb)))
    long_max_recent = int(p.get("long_max_recent_crosses", max_recent_x))
    short_max_recent = int(p.get("short_max_recent_crosses", max_recent_x))
    long_stop_off = float(p.get("long_vwap_stop_offset", vwap_offset))
    short_stop_off = float(p.get("short_vwap_stop_offset", vwap_offset))
    long_max_stop = float(p.get("long_max_stop_pts", max_stop_pts))
    short_max_stop = float(p.get("short_max_stop_pts", max_stop_pts))
    long_dow_str = (str(p.get("long_trade_dow_mask") or dow_mask_raw) + "1111100")[:7]
    short_dow_str = (str(p.get("short_trade_dow_mask") or dow_mask_raw) + "1111100")[:7]
    long_dow_allowed = tuple(c == "1" for c in long_dow_str)
    short_dow_allowed = tuple(c == "1" for c in short_dow_str)
    trigger_off = entry_ticks * TICK
    is_close_confirm = entry_mode == "close_confirm"
    is_early = entry_mode == "early_cross"
    is_early_fill = entry_mode == "close_confirm_early_fill"

    bar_seconds = _median_bar_seconds(fine)
    bars_1m = _to_1min_view(fine)
    if bars_1m.empty:
        return {"trades": pd.DataFrame(), "skipped": {"empty_1m": 1}, "n_bars": int(len(fine)),
                "params": p, "vwap": np.array([], dtype="float64"),
                "bar_seconds": bar_seconds}

    # Signal-side arrays (1-min)
    v1m = _session_vwap(bars_1m, session_start_min=session_start_m)
    o1 = bars_1m["open"].to_numpy("float64")
    h1 = bars_1m["high"].to_numpy("float64")
    l1 = bars_1m["low"].to_numpy("float64")
    c1 = bars_1m["close"].to_numpy("float64")
    ct1 = bars_1m["_ct_min"].to_numpy()
    d1 = bars_1m["_date"].to_numpy()
    idx1 = bars_1m.index
    n1 = len(bars_1m)

    # Fine-side arrays (for exit + intra-bar entry timing)
    vfine = _session_vwap(fine, session_start_min=session_start_m)
    of = fine["open"].to_numpy("float64")
    hf = fine["high"].to_numpy("float64")
    lf = fine["low"].to_numpy("float64")
    cf = fine["close"].to_numpy("float64")
    ctf = fine["_ct_min"].to_numpy()
    df_ = fine["_date"].to_numpy()
    idxf = fine.index
    nf = len(fine)
    # Map each 1-min bucket to the index of its first fine bar.
    fine_starts = _fine_starts_for_1m(bars_1m, fine)
    # Inverse map: for each fine bar j, the 1-min index it belongs to.
    fine_to_1m = np.searchsorted(bars_1m.index.asi8, fine.index.asi8, side="right") - 1
    fine_to_1m = np.clip(fine_to_1m, 0, n1 - 1).astype("int64")

    def fine_range_1m(i: int) -> tuple[int, int]:
        start = int(fine_starts[i])
        if i + 1 < n1:
            end = int(fine_starts[i + 1])
        else:
            end = nf
        return start, min(end, nf)

    # cross events on 1-min bars
    signs = np.zeros(n1, dtype="int8")
    for i in range(n1):
        if session_start_m <= int(ct1[i]) < session_end_m and math.isfinite(v1m[i]):
            if c1[i] > v1m[i]:
                signs[i] = 1
            elif c1[i] < v1m[i]:
                signs[i] = -1
    cross_flags = np.zeros(n1, dtype="int8")
    last_sign_by_day: dict[Any, int] = {}
    for i in range(n1):
        if signs[i] == 0:
            continue
        prev = last_sign_by_day.get(d1[i])
        if prev is not None and prev != 0 and prev != signs[i]:
            cross_flags[i] = 1
        last_sign_by_day[d1[i]] = signs[i]
    session_cross_count: dict[Any, int] = {}
    cum_cross = np.zeros(n1, dtype="int32")
    for i in range(n1):
        session_cross_count[d1[i]] = session_cross_count.get(d1[i], 0) + int(cross_flags[i])
        cum_cross[i] = session_cross_count[d1[i]]

    prior_vp = _prior_session_vp_levels(
        bars_1m, session_start_m=session_start_m,
        session_end_m=session_end_m, va_pct=vp_va_pct,
    ) if pt_mode == "vp_level" else {}

    # ATR + EMA precompute — only when the corresponding gate / exit is active.
    atr1 = _wilder_atr(bars_1m, atr_lb) if use_bar_size else None
    need_ema = use_ema_filter or pt_mode == "ema_vwap_convergence"
    # Convergence EMA (slower) — used ONLY by the ema_vwap_convergence PT
    # mode's gap calculation. Not consulted by any entry filter.
    ema1 = _ema(c1, ema_length) if pt_mode == "ema_vwap_convergence" else None
    # Crossover EMA (quicker) — feeds the EMA/VWAP state and cross-event flags.
    ema_fast = _ema(c1, ema_cross_length) if use_ema_filter else None

    # Precompute EMA/VWAP fresh-cross flags on the Crossover EMA, reset per day.
    ema_cross_up_flags = np.zeros(n1, dtype=bool)
    ema_cross_dn_flags = np.zeros(n1, dtype=bool)
    if use_ema_filter and ema_fast is not None:
        last_ema_sign_by_day: dict[Any, int] = {}
        for i in range(n1):
            if not (math.isfinite(ema_fast[i]) and math.isfinite(v1m[i])):
                continue
            sgn = 1 if ema_fast[i] > v1m[i] else (-1 if ema_fast[i] < v1m[i] else 0)
            if sgn == 0:
                continue
            prev = last_ema_sign_by_day.get(d1[i], 0)
            if prev != 0 and prev != sgn:
                if sgn > 0:
                    ema_cross_up_flags[i] = True
                else:
                    ema_cross_dn_flags[i] = True
            last_ema_sign_by_day[d1[i]] = sgn

    trades: list[_Trade] = []
    skipped: dict[str, int] = {}
    # Per-session latch of the last confirmed STRICT price cross-bar direction
    # (+1 up, -1 down). Reset at each new session day.
    price_cross_side_by_day: dict[Any, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    entry_i = 1
    while entry_i < n1 - 1:
        i = entry_i
        entry_i += 1
        m_i = int(ct1[i])
        if not (session_start_m <= m_i < session_end_m):
            continue
        dow = idx1[i].weekday() if hasattr(idx1[i], "weekday") else 0
        long_dow_ok = long_dow_allowed[dow] and trade_long
        short_dow_ok = short_dow_allowed[dow] and trade_short
        if not (long_dow_ok or short_dow_ok):
            skip("skipped_dow")
            continue
        needs_early = is_early or is_early_fill
        # Entry-candle-window gate:
        #   close_confirm — the entry candle N+1 must fall in [trade_start, trade_end)
        #   early_cross / close_confirm_early_fill — candle N must fall in [trade_start, trade_end)
        if needs_early:
            if not (trade_start_m <= m_i < trade_end_m):
                continue
        else:
            if i + 1 >= n1 or d1[i + 1] != d1[i]:
                continue
            m_next = int(ct1[i + 1])
            if not (trade_start_m <= m_next < trade_end_m):
                continue

        v_prev = v1m[i - 1]
        v_now = v1m[i]
        if not (math.isfinite(v_prev) and math.isfinite(v_now)):
            continue
        if d1[i - 1] != d1[i]:
            continue

        # Session-cumulative cross count: close_confirm fires at N+1 open so
        # N's own cross-flag is known; early modes fire intra-bar in N so use
        # the count *through N-1* only. Both counters use bars <= entry time.
        cross_count = int(cum_cross[i] if is_close_confirm else cum_cross[i - 1])
        long_chop_ok = cross_count <= long_max_sess_x
        short_chop_ok = cross_count <= short_max_sess_x
        if long_chop_lb > 0:
            lo_l = max(0, i - long_chop_lb)
            recent_l = int(cross_flags[lo_l:i].sum())
            if recent_l > long_max_recent:
                long_chop_ok = False
        if short_chop_lb > 0:
            lo_s = max(0, i - short_chop_lb)
            recent_s = int(cross_flags[lo_s:i].sum())
            if recent_s > short_max_recent:
                short_chop_ok = False
        if not (long_chop_ok or short_chop_ok):
            skip("chop_session_or_recent")
            continue

        # VWAP slope filter — backward (live-safe, default) or forward
        # (uses bars N+1..N+lookback → look-ahead, research-only).
        if use_slope:
            if slope_forward:
                # Original forward-looking behaviour: bars AFTER the reference bar.
                slope_start_i = i if is_close_confirm else i - 1
                slope_end_i = slope_start_i + slope_lb
                if slope_end_i >= n1 or d1[slope_end_i] != d1[i]:
                    continue
            else:
                if is_close_confirm:
                    slope_end_i = i          # V[N] — known at bar close
                else:
                    slope_end_i = i - 1      # V[N-1] — last final VWAP intra-bar
                slope_start_i = slope_end_i - slope_lb
                if slope_start_i < 0 or d1[slope_start_i] != d1[i]:
                    continue
            v_slope_end = v1m[slope_end_i]
            v_slope_start = v1m[slope_start_i]
            if not (math.isfinite(v_slope_end) and math.isfinite(v_slope_start)):
                continue
            slope = (v_slope_end - v_slope_start) / float(slope_lb)
        else:
            slope = 0.0

        # Close-location filter on the crossing candle (live-safe, uses only bar N).
        bar_range = h1[i] - l1[i]
        if bar_range > 0 and math.isfinite(bar_range):
            long_clv = (c1[i] - l1[i]) / bar_range
            short_clv = (h1[i] - c1[i]) / bar_range
        else:
            long_clv = short_clv = 0.5

        # Bar-size (min-ATR) filter — reject small-body signals when enabled.
        if use_bar_size:
            atr_ref_i = i if is_close_confirm else i - 1
            atr_val = atr1[atr_ref_i] if atr1 is not None and atr_ref_i >= 0 else float("nan")
            if not math.isfinite(atr_val) or atr_val <= 0:
                continue
            if bar_range < bar_size_mult * atr_val:
                skip("bar_too_small")
                continue

        # Preceder / slope / body gates (same for every entry mode).
        preceder_long_ok = (c1[i - 1] > o1[i - 1]) if use_preceder else True
        preceder_short_ok = (c1[i - 1] < o1[i - 1]) if use_preceder else True
        slope_long_ok = (slope >= min_slope) if use_slope else True
        slope_short_ok = ((-slope) >= min_slope) if use_slope else True

        # Standard price-cross-bar test — strict "open on one side, close on
        # the other". For early_cross the analogous test is intra-bar
        # penetration past V_prev.
        if is_early:
            price_cross_up = o1[i] < v_prev and h1[i] >= v_prev + trigger_off
            price_cross_dn = o1[i] > v_prev and l1[i] <= v_prev - trigger_off
        else:
            price_cross_up = o1[i] < v_prev and c1[i] > v_now
            price_cross_dn = o1[i] > v_prev and c1[i] < v_now

        # Latch the last confirmed strict price cross-bar direction for this
        # session. Once a strict cross-bar has occurred in either direction,
        # subsequent bars may fire on the EMA cross alone (second-cross rule)
        # as long as the close side still agrees.
        if price_cross_up:
            price_cross_side_by_day[d1[i]] = 1
        elif price_cross_dn:
            price_cross_side_by_day[d1[i]] = -1
        latched_side = price_cross_side_by_day.get(d1[i], 0)

        # EMA filter on: second-cross rule. Fire when the Crossover EMA and
        # the close are both on the same side of VWAP at bar i AND at least
        # ONE of the two crosses has just occurred at bar i — either a strict
        # price cross-bar (open one side, close the other) or a fresh EMA/
        # VWAP cross. The strict price cross must have occurred at some point
        # this session (checked via the latched side) so an EMA-only whipsaw
        # cannot fire without a real price cross-bar earlier.
        if use_ema_filter:
            if is_early:
                ema_ref_i = i - 1
            else:
                ema_ref_i = i
            ema_val = ema_fast[ema_ref_i] if ema_fast is not None and ema_ref_i >= 0 else float("nan")
            vwap_ref = v1m[ema_ref_i] if ema_ref_i >= 0 else float("nan")
            if not (math.isfinite(ema_val) and math.isfinite(vwap_ref)):
                skip("ema_warmup")
                continue
            ema_up = ema_val > vwap_ref
            ema_dn = ema_val < vwap_ref
            if is_early:
                # early_cross keeps its intra-bar semantic: strict price cross
                # AND EMA already on the same side (state on last-final bar).
                long_trigger_ok = price_cross_up and ema_up
                short_trigger_ok = price_cross_dn and ema_dn
            else:
                close_up = c1[i] > v_now
                close_dn = c1[i] < v_now
                ema_up_at_N = bool(ema_cross_up_flags[i])
                ema_dn_at_N = bool(ema_cross_dn_flags[i])
                long_trigger_ok = (
                    latched_side == 1
                    and ema_up and close_up
                    and (price_cross_up or ema_up_at_N)
                )
                short_trigger_ok = (
                    latched_side == -1
                    and ema_dn and close_dn
                    and (price_cross_dn or ema_dn_at_N)
                )
        else:
            long_trigger_ok = price_cross_up
            short_trigger_ok = price_cross_dn

        long_ok = (
            trade_long
            and long_dow_ok
            and long_chop_ok
            and long_trigger_ok
            and preceder_long_ok
            and slope_long_ok
            and (not use_min_clv or long_clv >= min_body_clv)
        )
        short_ok = (
            trade_short
            and short_dow_ok
            and short_chop_ok
            and short_trigger_ok
            and preceder_short_ok
            and slope_short_ok
            and (not use_min_clv or short_clv >= min_body_clv)
        )
        if not (long_ok or short_ok):
            continue
        side = 1 if long_ok else -1

        # Resolve entry price + entry_bar (in the fine series).
        fine_lo_i, fine_hi_i = fine_range_1m(i)
        if is_close_confirm:
            # entry_bar_delay > 0 shifts the fill by that many 1-min bars — used
            # to research the live-safe equivalent of a forward-slope backtest:
            # wait N bars before entering so that V[i+N] is legitimately known.
            entry_1m_idx = i + 1 + entry_bar_delay
            if entry_1m_idx >= n1 or d1[entry_1m_idx] != d1[i]:
                skip("no_delayed_entry_bar")
                continue
            entry_bar_fine = int(fine_starts[entry_1m_idx])
            entry_px = _round_tick(of[entry_bar_fine] + side * slip)
            v_at_entry = v_now
        elif is_early:
            trigger = v_prev + side * trigger_off
            found = -1
            for j in range(fine_lo_i, fine_hi_i):
                touched = (hf[j] >= trigger) if side > 0 else (lf[j] <= trigger)
                if touched:
                    found = j
                    break
            if found < 0:
                skip("no_intrabar_trigger")
                continue
            gap = (of[found] >= trigger) if side > 0 else (of[found] <= trigger)
            fill_px = of[found] if gap else trigger
            entry_bar_fine = found
            entry_px = _round_tick(fill_px + side * slip)
            v_at_entry = v_prev
        else:  # close_confirm_early_fill
            trigger = v_prev + side * trigger_off
            found = -1
            for j in range(fine_lo_i, fine_hi_i):
                touched = (hf[j] >= trigger) if side > 0 else (lf[j] <= trigger)
                if touched:
                    found = j
                    break
            if found >= 0:
                gap = (of[found] >= trigger) if side > 0 else (of[found] <= trigger)
                fill_px = of[found] if gap else trigger
                entry_bar_fine = found
                entry_px = _round_tick(fill_px + side * slip)
                v_at_entry = v_prev
            else:
                # Bar never touched the trigger intra-bar → fall back to the
                # close_confirm behaviour so the trade population matches exactly.
                entry_bar_fine = int(fine_starts[i + 1]) if i + 1 < n1 else fine_lo_i
                if entry_bar_fine >= nf:
                    skip("no_next_bar")
                    continue
                entry_px = _round_tick(of[entry_bar_fine] + side * slip)
                v_at_entry = v_now

        side_vwap_offset = long_stop_off if side > 0 else short_stop_off
        side_max_stop = long_max_stop if side > 0 else short_max_stop
        stop_px = _round_tick(_resolve_stop(
            side=side, entry_px=entry_px,
            crossing_open=o1[i], vwap_now=v_at_entry,
            stop_mode=stop_mode, vwap_offset=side_vwap_offset,
        ))
        stop_px = _round_tick(_clamp_stop(
            side=side, entry_px=entry_px, stop_px=stop_px, max_stop_pts=side_max_stop,
        ))
        if side > 0 and stop_px >= entry_px:
            skip("stop_beyond_entry")
            continue
        if side < 0 and stop_px <= entry_px:
            skip("stop_beyond_entry")
            continue

        if pt_mode == "vp_level":
            target_val, target_ref = _resolve_vp_target(
                side=side, entry_px=entry_px,
                levels=prior_vp.get(d1[i]), fallback_pts=vp_fallback_pts,
            )
            target_px = _round_tick(target_val)
        elif pt_mode == "ema_vwap_convergence":
            # No fixed price target — the convergence rule runs inside the exit walk.
            target_px = float("inf") if side > 0 else float("-inf")
            target_ref = "ema_vwap_convergence"
        else:
            target_px = _round_tick(_resolve_target(
                side=side, entry_px=entry_px, stop_px=stop_px,
                pt_mode=pt_mode, pt_pts=pt_pts, pt_mult=pt_mult,
            ))
            target_ref = pt_mode

        stop_pts = abs(entry_px - stop_px)
        if math.isfinite(target_px):
            target_pts = abs(target_px - entry_px)
            if target_pts <= 0:
                skip("zero_target")
                continue
        else:
            target_pts = 0.0

        # Exit scan runs on the FINE series from entry_bar_fine + 1 onwards.
        exit_scan_start = entry_bar_fine + 1
        exit_i_fine = min(exit_scan_start, nf - 1)
        exit_px = cf[exit_i_fine]
        exit_reason = "session_flat"
        mae = 0.0
        mfe = 0.0
        j = exit_scan_start
        entry_date = d1[i]
        entry_1m_i = int(fine_to_1m[entry_bar_fine])
        prev_1m_i = entry_1m_i
        peak_gap = 0.0
        use_convergence = pt_mode == "ema_vwap_convergence"
        while j < nf and df_[j] == entry_date and int(ctf[j]) < session_end_m:
            hi_j, lo_j = hf[j], lf[j]
            if math.isfinite(hi_j) and math.isfinite(lo_j):
                if side > 0:
                    mae = max(mae, entry_px - lo_j)
                    mfe = max(mfe, hi_j - entry_px)
                else:
                    mae = max(mae, hi_j - entry_px)
                    mfe = max(mfe, entry_px - lo_j)
            hit_stop = (lo_j <= stop_px) if side > 0 else (hi_j >= stop_px)
            hit_target = math.isfinite(target_px) and (
                (hi_j >= target_px) if side > 0 else (lo_j <= target_px)
            )
            if hit_stop:
                gap = (of[j] <= stop_px) if side > 0 else (of[j] >= stop_px)
                exit_px = of[j] if gap else stop_px
                exit_reason = "stop_loss"
                exit_i_fine = j
                break
            if hit_target:
                gap = (of[j] >= target_px) if side > 0 else (of[j] <= target_px)
                exit_px = of[j] if gap else target_px
                exit_reason = "profit_target"
                exit_i_fine = j
                break
            # EMA/VWAP convergence PT — checked at each 1-min boundary using
            # the just-closed 1-min bar's EMA and VWAP.
            if use_convergence:
                cur_1m_i = int(fine_to_1m[j])
                if cur_1m_i > prev_1m_i and cur_1m_i > 0:
                    closed_i = cur_1m_i - 1
                    if closed_i >= entry_1m_i + min_bars_ema_pt and ema1 is not None:
                        e_v = ema1[closed_i]
                        w_v = v1m[closed_i]
                        if math.isfinite(e_v) and math.isfinite(w_v):
                            gap_mag = abs(e_v - w_v)
                            if gap_mag > peak_gap:
                                peak_gap = gap_mag
                            elif (peak_gap > 0
                                  and gap_mag < peak_gap * (1.0 - ema_shrink_pct / 100.0)):
                                exit_px = of[j]
                                exit_reason = "ema_convergence"
                                exit_i_fine = j
                                break
                prev_1m_i = cur_1m_i
            j += 1
        else:
            exit_i_fine = min(j, nf - 1)
            exit_px = cf[exit_i_fine]
            exit_reason = "session_flat"

        exit_px_adj = _round_tick(exit_px - side * slip)
        holding_bars_1m = max(0, int((idxf[exit_i_fine] - idxf[entry_bar_fine]).total_seconds() // 60))
        trades.append(_Trade(
            session_date=d1[i],
            entry_ts=idxf[entry_bar_fine],
            exit_ts=idxf[exit_i_fine],
            side="long" if side > 0 else "short",
            entry_px=entry_px, exit_px=exit_px_adj,
            stop_px=stop_px, target_px=target_px,
            stop_pts=stop_pts, target_pts=target_pts,
            exit_reason=exit_reason,
            mae_points=float(mae), mfe_points=float(mfe),
            holding_bars=holding_bars_1m,
            target_ref=str(target_ref),
        ))
        # Advance the signal cursor past the 1-min bar of the exit so a new
        # signal can only fire *after* this position has closed. Prevents
        # overlapping positions on adjacent bars.
        exit_1m_i = int(fine_to_1m[exit_i_fine]) if 0 <= exit_i_fine < nf else n1 - 1
        entry_i = max(entry_i, exit_1m_i + 1)

    if not trades:
        ledger = pd.DataFrame()
    else:
        rows = []
        for t in trades:
            pts = (t.exit_px - t.entry_px) * (1 if t.side == "long" else -1)
            gross = pts * qty
            usd = gross * pv - cost_pts * pv * qty
            rows.append({
                "session_date": t.session_date,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "side": t.side,
                "qty": qty,
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "stop_px": t.stop_px,
                "target_px": t.target_px,
                "points": pts,
                "pnl_points": gross,
                "pnl_usd": usd,
                "risk_pts": t.stop_pts,
                "target_pts": t.target_pts,
                "r_multiple": pts / t.stop_pts if t.stop_pts > 0 else 0.0,
                "mae_points": t.mae_points,
                "mfe_points": t.mfe_points,
                "holding_bars": t.holding_bars,
                "exit_reason": t.exit_reason,
                "setup": "vwap_cross",
                "target_ref": t.target_ref,
            })
        ledger = pd.DataFrame(rows)
    return {
        "trades": ledger,
        "skipped": skipped,
        "n_bars": int(nf),
        "n_bars_1m": int(n1),
        "bar_seconds": int(bar_seconds),
        "params": p,
        "vwap": v1m,
    }


# ── Backtest-tab payload ─────────────────────────────────────────────────

def _iso(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def vwap_cross_payload(
    res: dict[str, Any], *, source: str, start: str | None, end: str | None,
    n_sessions: int, card: dict[str, Any],
) -> dict[str, Any]:
    """Shape the VWAP-cross result like the other external-model backtests."""
    ledger: pd.DataFrame = res["trades"]
    p = res["params"]
    pv = float(p.get("point_value") or DEFAULT_POINT_VALUE)
    title = card.get("title") or "VWAP Crossover"
    config: dict[str, Any] = {
        "strategy_id": "vwap_cross",
        "objective_id": None,
        "tau": 0.0,
        "confidence_gate": False,
        "sides": ["up", "down"],
        "max_concurrent_positions": 1,
        "target_points": float(p["profit_target_pts"]),
        "stop_points": None,
        "max_holding": f"flatten {p['session_end']}",
        "point_value": pv,
        "instrument": "NQ",
        "window": {
            "start": start,
            "end": end,
            "n_sessions": int(n_sessions),
            "n_bars": int(res["n_bars"]),
        },
        "prediction_mode": "external",
        "fill_path": "vwap_cross_1m",
        "execution": "vwap_cross_candle",
        "source_kind": "external",
        "source": source,
        "run_label": title,
        "model_path": source,
        "candidate_id": "vwap_cross",
        "model_family": "external_vwap_cross",
        "kind": "external_vwap_cross",
        "title": title,
        "timeframe": "1min",
        "bar_seconds": 60,
        "candle_table": "databento.tbbo_1min",
        "params": p,
        "book": None,
        "stack": [{
            "model": "vwap_cross", "hardened": None, "params": p,
            "timeframe": "1min", "title": title,
        }],
        "hardened": None,
        "live_data": False,
        "tbbo_data": False,
        "oco": False,
        "price_mode": "close",
        "weights_note": (
            f"VWAP crossover ({p.get('entry_mode', 'close_confirm')}"
            f"{', ' + str(p.get('entry_ticks', 0)) + ' ticks' if p.get('entry_mode') == 'early_cross' else ''}) · "
            f"session {p['session_start']}–{p['session_end']} CT · "
            f"stop={p['stop_mode']} (vwap±{p['vwap_stop_offset']}pt) · "
            f"pt={p['profit_target_mode']}"
            f"{'=' + str(p['profit_target_pts']) + 'pt' if p['profit_target_mode'] == 'absolute' else '×' + str(p['profit_target_stop_mult'])}"
            f" · chop cap session={p['max_session_crosses']}, "
            f"{p['max_recent_crosses']}/{p['chop_lookback']}bars · "
            f"slope≥{p['min_vwap_slope']} over {p['vwap_slope_lookback']} bars"
        ),
    }
    labels = ["down", "flat", "up"]
    empty_keys = ("gross_profit", "gross_loss", "net_pnl", "win_rate", "expectancy",
                  "profit_factor", "max_drawdown", "sharpe_daily", "avg_win", "avg_loss")
    if ledger is None or ledger.empty:
        return {
            "config": config,
            "summary": {"n_trades": 0, "n_wins": 0, "n_losses": 0,
                        "point_value": pv, "net_value": 0.0,
                        "by_exit_reason": {}, "n_sessions_traded": 0,
                        **{k: None for k in empty_keys}},
            "equity_curve": [],
            "confusion_matrix": {"labels": labels,
                                 "rows_true_cols_pred": [[0] * 3 for _ in range(3)]},
            "trades": [], "skipped": dict(res["skipped"]), "archetypes": [],
        }
    pnl = ledger["pnl_points"].to_numpy("float64")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    cum = np.cumsum(pnl)
    hwm = np.maximum.accumulate(np.maximum(cum, 0.0))
    dd = cum - hwm
    daily = ledger.groupby("session_date")["pnl_points"].sum().to_numpy("float64")

    def _sharpe(x: np.ndarray) -> float | None:
        if len(x) < 2:
            return None
        s = float(x.std(ddof=1))
        return float(x.mean() / s) if s else None

    summary = {
        "n_trades": int(len(ledger)),
        "n_wins": int((pnl > 0).sum()),
        "n_losses": int((pnl <= 0).sum()),
        "gross_profit": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss": float(losses.sum()) if len(losses) else 0.0,
        "net_pnl": float(pnl.sum()),
        "net_value": float(pnl.sum() * pv),
        "win_rate": float((pnl > 0).mean()),
        "expectancy": float(pnl.mean()),
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if len(losses) and float(losses.sum()) != 0.0 else None),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "sharpe_daily": _sharpe(daily),
        "sharpe_trade": _sharpe(pnl),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "avg_holding_bars": float(ledger["holding_bars"].mean()),
        "avg_mae": float(ledger["mae_points"].mean()),
        "avg_mfe": float(ledger["mfe_points"].mean()),
        "avg_r": float(ledger["r_multiple"].mean()),
        "n_sessions_traded": int(ledger["session_date"].nunique()),
        "by_exit_reason": {str(k): int(v)
                           for k, v in ledger["exit_reason"].value_counts().items()},
        "point_value": pv,
    }
    entry_ts = pd.to_datetime(ledger["entry_ts"], utc=True)
    exit_ts = pd.to_datetime(ledger["exit_ts"], utc=True)
    equity_curve = [{"t": _iso(entry_ts.iloc[0]), "equity": 0.0}]
    equity_curve += [{"t": _iso(exit_ts.iloc[j]), "equity": float(cum[j])}
                     for j in range(len(ledger))]
    lab_ix = {l: i for i, l in enumerate(labels)}
    cm = [[0] * 3 for _ in range(3)]
    for side, pts in zip(ledger["side"].to_numpy(), pnl):
        pred = "up" if side == "long" else "down"
        true = pred if pts > 0 else ("down" if pred == "up" else "up")
        cm[lab_ix[true]][lab_ix[pred]] += 1
    trades_out = []
    for j, row in ledger.reset_index(drop=True).iterrows():
        trades_out.append({
            "entry_ts": _iso(row["entry_ts"]),
            "exit_ts": _iso(row["exit_ts"]),
            "side": row["side"],
            "qty": int(row["qty"]),
            "entry_px": float(row["entry_px"]),
            "exit_px": float(row["exit_px"]),
            "stop_px": float(row["stop_px"]),
            "target_px": (float(row["target_px"])
                          if math.isfinite(float(row["target_px"])) else None),
            "points": float(row["points"]),
            "pnl_points": float(row["pnl_points"]),
            "pnl_usd": float(row["pnl_usd"]),
            "r_multiple": float(row["r_multiple"]),
            "risk_pts": float(row["risk_pts"]),
            "target_pts": float(row["target_pts"]),
            "cum_pnl": float(cum[j]),
            "mae_points": float(row["mae_points"]),
            "mfe_points": float(row["mfe_points"]),
            "holding_bars": int(row["holding_bars"]),
            "exit_reason": str(row["exit_reason"]),
            "setup": str(row["setup"]),
            "target_ref": str(row.get("target_ref", "")),
            "n_present": None, "n_total": None,
        })
    return {
        "config": config, "summary": summary, "equity_curve": equity_curve,
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm},
        "trades": trades_out, "skipped": dict(res["skipped"]), "archetypes": [],
    }
