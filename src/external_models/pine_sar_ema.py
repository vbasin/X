"""External-model signal engines + shared book (ADR-058 / ADR-062 / ADR-063 / ADR-096).

Engines, each emitting a per-bar side in ``{-1, 0, +1}``:

* **Parabolic SAR** — follow the closed-bar Wilder trend
  (:func:`src.features.psar.sar_on_closed`, ADR-055). Optional
  ``exit_at_sar`` treats a tagged incoming trail as a stop fill.
* **20 EMA** — dual slope filters (established L + steep S) from the
  original Pine overlay, now a standalone long/short book.
* **ZigZag High/Low** — Thinkorswim ZigZagHighLow causal state
  (:func:`src.external_models.zigzag.zzhl_on_closed`, ADR-063).
* **Timed PSAR** — one entry per session at a clock time or N bars
  after session open, side = last closed 1-hour or 1-day SAR
  (:mod:`src.external_models.timed_psar`, ADR-096).

A stacked backtest ANDs those sides: matching signals enter, any
disagreement or a zero stands down / exits. Session, trade window, ATR
gate, PT/SL (points, × ATR, or PT as × stop), sides, signal-lock, and
costs live on the shared **book**, not on a card. Each engine may run
on its own candle timeframe (`1min`/`5min`/`15min`/`1hour`/`1day`);
the book timeframe is display + ATR/PT/SL bars (ADR-064 / ADR-065).

Fills:
* Market entries at the signal bar's close ± slippage.
* ``strategy.exit`` from average entry: profit target / stop loss as
  NQ points, a multiple of Wilder ATR(period) at entry, or (PT only)
  a multiple of the resolved stop distance. ``0`` disables that side
  independently. High/low first-touch on bars after entry; same-bar
  stop+target → stop. Gapped-through levels fill at the open.
* Optional ``exit_at_sar`` (SAR engine): fill at the tagged trail
  (gap-through at the open) instead of the intersection-bar close.
* New entries only between ``trade_start`` and ``trade_end``.
* Session flatten at ``session_end`` (default 14:45 America/Chicago).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.common.cme_holidays import is_trading_day
from src.external_models.catalog import (
    coerce_book_params,
    coerce_params,
    compat_stack_for,
    model_engine,
    _hhmm_minutes,
)
from src.features.psar import (
    DEFAULT_ATR_PERIOD,
    _rma,
    _true_range,
    sar_on_closed,
)


def tv_ema(close: np.ndarray, length: int) -> np.ndarray:
    """TradingView ``ta.ema``: SMA seed, then α = 2/(length+1)."""
    n = int(close.size)
    out = np.full(n, np.nan, dtype="float64")
    length = int(length)
    if length < 1 or n < length:
        return out
    alpha = 2.0 / (float(length) + 1.0)
    out[length - 1] = float(np.mean(close[:length]))
    one_m = 1.0 - alpha
    for i in range(length, n):
        out[i] = alpha * float(close[i]) + one_m * out[i - 1]
    return out


def ema_established_slope(ema: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """``ema20Lup`` / ``ema20Ldn``: EMA vs SMA of the last ``lookback`` EMA values."""
    s = pd.Series(ema, dtype="float64")
    avg = s.rolling(int(lookback), min_periods=int(lookback)).mean().to_numpy()
    lup = (ema > avg) & np.isfinite(avg) & np.isfinite(ema)
    ldn = (ema < avg) & np.isfinite(avg) & np.isfinite(ema)
    return lup, ldn


def ema_steep_slope(ema: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Pine loop ``for i = 1 to ema20Sbars``: last S steps strictly monotonic."""
    n = int(ema.size)
    s = int(lookback)
    up = np.zeros(n, dtype=bool)
    dn = np.zeros(n, dtype=bool)
    if s < 1 or n <= s:
        return up, dn
    for i in range(s, n):
        inc = True
        dec = True
        for k in range(1, s + 1):
            a = ema[i - k]
            b = ema[i - k + 1]
            if not (np.isfinite(a) and np.isfinite(b)):
                inc = False
                dec = False
                break
            inc = inc and a < b
            dec = dec and a > b
        up[i] = inc
        dn[i] = dec
    return up, dn


def _tighter_stop(
    direction: int,
    candidates: list[tuple[float, str]],
) -> tuple[float, str] | None:
    """Adverse stop that is hit first: long = higher level, short = lower."""
    valid = [(float(px), reason) for px, reason in candidates if np.isfinite(px)]
    if not valid:
        return None
    if direction > 0:
        return max(valid, key=lambda item: item[0])
    return min(valid, key=lambda item: item[0])


def _bracket_fill(
    direction: int,
    entry_px: float,
    target_pts: float,
    stop_pts: float,
    o: float,
    h: float,
    l: float,
    *,
    sar_stop: float = np.nan,
) -> tuple[float, str] | None:
    """Pine ``strategy.exit`` limit+stop from ``position_avg_price``.

    Optional ``sar_stop`` is the in-effect SAR trail (tagged this bar).
    When both the OCO stop and SAR tag, the tighter (first-touch) stop
    fills. Same-bar target+stop → stop (conservative), unless the open
    gapped through the target without gapping the stop. An open that
    gaps through a level fills at the open.
    """
    use_tgt = np.isfinite(target_pts) and float(target_pts) > 0
    use_stop = np.isfinite(stop_pts) and float(stop_pts) > 0
    use_sar = np.isfinite(sar_stop)
    if not np.isfinite(entry_px) or not (use_tgt or use_stop or use_sar):
        return None
    if direction > 0:
        target = entry_px + float(target_pts) if use_tgt else np.nan
        chosen = _tighter_stop(direction, [
            (entry_px - float(stop_pts), "LG Stop Loss") if use_stop else (np.nan, ""),
            (float(sar_stop), "sar_stop") if use_sar else (np.nan, ""),
        ])
        stop, stop_reason = (chosen if chosen is not None else (np.nan, "LG Stop Loss"))
        hit_stop = np.isfinite(stop) and np.isfinite(l) and l <= stop
        hit_tgt = use_tgt and np.isfinite(h) and h >= target
        gap_stop = np.isfinite(stop) and np.isfinite(o) and o <= stop
        gap_tgt = use_tgt and np.isfinite(o) and o >= target
        if hit_stop and hit_tgt:
            if gap_tgt and not gap_stop:
                return float(o), "LG Profit Target"
            return (float(o) if gap_stop else float(stop)), stop_reason
        if hit_stop:
            return (float(o) if gap_stop else float(stop)), stop_reason
        if hit_tgt:
            return (float(o) if gap_tgt else float(target)), "LG Profit Target"
        return None
    target = entry_px - float(target_pts) if use_tgt else np.nan
    chosen = _tighter_stop(direction, [
        (entry_px + float(stop_pts), "ST Stop Loss") if use_stop else (np.nan, ""),
        (float(sar_stop), "sar_stop") if use_sar else (np.nan, ""),
    ])
    stop, stop_reason = (chosen if chosen is not None else (np.nan, "ST Stop Loss"))
    hit_stop = np.isfinite(stop) and np.isfinite(h) and h >= stop
    hit_tgt = use_tgt and np.isfinite(l) and l <= target
    gap_stop = np.isfinite(stop) and np.isfinite(o) and o >= stop
    gap_tgt = use_tgt and np.isfinite(o) and o <= target
    if hit_stop and hit_tgt:
        if gap_tgt and not gap_stop:
            return float(o), "ST Profit Target"
        return (float(o) if gap_stop else float(stop)), stop_reason
    if hit_stop:
        return (float(o) if gap_stop else float(stop)), stop_reason
    if hit_tgt:
        return (float(o) if gap_tgt else float(target)), "ST Profit Target"
    return None


def normalize_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """UTC DatetimeIndex, open/high/low/close, last-wins on duplicate stamps."""
    df = ohlc.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    need = ["high", "low", "close"]
    if any(c not in df.columns for c in need):
        raise ValueError("ohlc must have high, low, close")
    if "open" not in df.columns:
        df["open"] = df["close"]
    return df


def psar_signals(ohlc: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-bar SAR side: ``+1`` uptrend, ``-1`` downtrend, ``0`` warmup."""
    p = coerce_params("psar", params)
    df = normalize_ohlc(ohlc)
    n = len(df)
    empty = pd.DataFrame({
        "signal": np.zeros(n, dtype="int8"),
        "sar": np.full(n, np.nan),
        "next_sar": np.full(n, np.nan),
        "trend": np.full(n, np.nan),
        "prev_sar": np.full(n, np.nan),
        "prev_trend": np.full(n, np.nan),
        "tagged": np.zeros(n, dtype=bool),
        "exit_at_sar": np.zeros(n, dtype=bool),
    }, index=df.index)
    if n == 0:
        empty.attrs["model_id"] = "psar"
        empty.attrs["engine"] = "psar"
        empty.attrs["params"] = p
        return empty
    sar = sar_on_closed(
        df,
        start=float(p["start"]),
        increment=float(p["increment"]),
        maximum=float(p["maximum"]),
    )
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    trend = sar["trend"].to_numpy(dtype="float64")
    next_sar = sar["next_sar"].to_numpy(dtype="float64")
    prev_sar = np.full(n, np.nan, dtype="float64")
    prev_trend = np.full(n, np.nan, dtype="float64")
    if n > 1:
        prev_sar[1:] = next_sar[:-1]
        prev_trend[1:] = trend[:-1]
    signal = np.zeros(n, dtype="int8")
    signal[trend > 0] = 1
    signal[trend < 0] = -1
    tagged = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(prev_sar[i]) or not np.isfinite(prev_trend[i]):
            continue
        if prev_trend[i] > 0:
            tagged[i] = np.isfinite(low[i]) and low[i] <= prev_sar[i]
        elif prev_trend[i] < 0:
            tagged[i] = np.isfinite(high[i]) and high[i] >= prev_sar[i]
    out = pd.DataFrame({
        "signal": signal,
        "sar": sar["sar"].to_numpy(dtype="float64"),
        "next_sar": next_sar,
        "trend": trend,
        "prev_sar": prev_sar,
        "prev_trend": prev_trend,
        "tagged": tagged,
        "exit_at_sar": np.full(n, bool(p.get("exit_at_sar", False))),
    }, index=df.index)
    out.attrs["model_id"] = "psar"
    out.attrs["engine"] = "psar"
    out.attrs["params"] = p
    return out


def ema_signals(ohlc: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Per-bar EMA-slope side: ``+1`` up, ``-1`` down, ``0`` disagree / warmup."""
    p = coerce_params("ema20", params)
    df = normalize_ohlc(ohlc)
    n = len(df)
    close = df["close"].to_numpy(dtype="float64") if n else np.array([], dtype="float64")
    ema_len = int(p.get("ema_length") or 0)
    ema = tv_ema(close, ema_len) if ema_len >= 2 else np.full(n, np.nan)
    l_bars = int(p.get("ema20_l_bars") or 0)
    s_bars = int(p.get("ema20_s_bars") or 0)
    use_l = l_bars > 0
    use_s = s_bars > 0
    if use_l:
        lup, ldn = ema_established_slope(ema, l_bars)
    else:
        lup = np.ones(n, dtype=bool)
        ldn = np.ones(n, dtype=bool)
    if use_s:
        sup, sdn = ema_steep_slope(ema, s_bars)
    else:
        sup = np.ones(n, dtype=bool)
        sdn = np.ones(n, dtype=bool)
    long_ok = (not use_l or lup) & (not use_s or sup) & np.isfinite(ema)
    short_ok = (not use_l or ldn) & (not use_s or sdn) & np.isfinite(ema)
    signal = np.zeros(n, dtype="int8")
    signal[long_ok & ~short_ok] = 1
    signal[short_ok & ~long_ok] = -1
    out = pd.DataFrame({
        "signal": signal,
        "ema": ema,
        "lup": lup,
        "ldn": ldn,
        "sup": sup,
        "sdn": sdn,
        "tagged": np.zeros(n, dtype=bool),
        "exit_at_sar": np.zeros(n, dtype=bool),
        "prev_sar": np.full(n, np.nan),
        "prev_trend": np.full(n, np.nan),
    }, index=df.index)
    out.attrs["model_id"] = "ema20"
    out.attrs["engine"] = "ema"
    out.attrs["params"] = p
    return out


def build_signals(
    model_id: str,
    ohlc: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    book_ohlc: pd.DataFrame | None = None,
    book: dict[str, Any] | None = None,
) -> pd.DataFrame:
    engine = model_engine(model_id)
    if engine == "psar" or model_id == "psar":
        frame = psar_signals(ohlc, params)
        frame.attrs["model_id"] = model_id
        return frame
    if engine == "ema" or model_id == "ema20":
        frame = ema_signals(ohlc, params)
        frame.attrs["model_id"] = model_id
        return frame
    if engine == "zigzag" or model_id == "zigzag":
        from src.external_models.zigzag import zzhl_signals
        frame = zzhl_signals(ohlc, params)
        frame.attrs["model_id"] = model_id
        return frame
    if engine == "timed_psar" or model_id == "timed_psar":
        from src.external_models.timed_psar import timed_psar_signals
        frame = timed_psar_signals(
            ohlc, params, book_ohlc=book_ohlc, book=book,
        )
        frame.attrs["model_id"] = model_id
        return frame
    raise ValueError(f"no signal engine for external model {model_id}")


def consensus_signal(frames: list[pd.DataFrame]) -> np.ndarray:
    """AND of per-bar sides. Any zero or a disagreement → 0."""
    if not frames:
        return np.array([], dtype="int8")
    sigs = [f["signal"].to_numpy(dtype="int8") for f in frames]
    out = sigs[0].copy()
    for s in sigs[1:]:
        if s.shape != out.shape:
            raise ValueError("stacked signal frames must be the same length")
        out = np.where((out == s) & (out != 0), out, np.int8(0)).astype("int8")
    return out


def _sar_stop_arrays(frames: list[pd.DataFrame]) -> list[dict[str, np.ndarray]]:
    out: list[dict[str, np.ndarray]] = []
    for frame in frames:
        n = len(frame)
        if "prev_sar" not in frame.columns or "prev_trend" not in frame.columns:
            continue
        out.append({
            "exit_at_sar": (
                frame["exit_at_sar"].to_numpy(dtype=bool)
                if "exit_at_sar" in frame.columns else np.zeros(n, dtype=bool)
            ),
            "prev_sar": frame["prev_sar"].to_numpy(dtype="float64"),
            "prev_trend": frame["prev_trend"].to_numpy(dtype="float64"),
            "tagged": (
                frame["tagged"].to_numpy(dtype=bool)
                if "tagged" in frame.columns else np.zeros(n, dtype=bool)
            ),
        })
    return out


def _sar_stop_level(sar_arrs: list[dict[str, np.ndarray]], i: int, direction: int) -> float:
    """Incoming SAR trail if a stacked SAR engine wants a stop fill this bar."""
    for arr in sar_arrs:
        if not bool(arr["exit_at_sar"][i]) or not bool(arr["tagged"][i]):
            continue
        prev_sar = float(arr["prev_sar"][i])
        prev_trend = float(arr["prev_trend"][i])
        if not np.isfinite(prev_sar) or not np.isfinite(prev_trend):
            continue
        if (direction > 0 and prev_trend > 0) or (direction < 0 and prev_trend < 0):
            return prev_sar
    return float("nan")


def _signal_exit_reason(model_ids: list[str], pos_dir: int, desired: int) -> str:
    stacked = len(model_ids) > 1
    if desired == -pos_dir:
        if not stacked and model_ids[0] == "psar":
            return "sar_up_reverse" if desired > 0 else "sar_dn_reverse"
        if not stacked and model_ids[0] == "ema20":
            return "ema_up_reverse" if desired > 0 else "ema_dn_reverse"
        if not stacked and model_ids[0] == "zigzag":
            return "zz_up_reverse" if desired > 0 else "zz_dn_reverse"
        if not stacked and model_ids[0] == "timed_psar":
            return "timed_up_reverse" if desired > 0 else "timed_dn_reverse"
        return "stack_reverse"
    if not stacked and model_ids[0] == "psar":
        return "sar_dn_close" if pos_dir > 0 else "sar_up_close"
    if not stacked and model_ids[0] == "ema20":
        return "ema_dn_close" if pos_dir > 0 else "ema_up_close"
    if not stacked and model_ids[0] == "zigzag":
        return "zz_dn_close" if pos_dir > 0 else "zz_up_close"
    if not stacked and model_ids[0] == "timed_psar":
        return "timed_dn_close" if pos_dir > 0 else "timed_up_close"
    return "stack_disagree"


@dataclass
class SimResult:
    trades: pd.DataFrame
    n_bars: int
    skipped: dict[str, int]
    params: dict[str, Any]
    frames: list[pd.DataFrame] = field(default_factory=list)


def _qty_in_points(value: float, unit: str, atr_at_entry: float) -> float:
    """Book distance: NQ points, or ``value × ATR`` when unit is ``atr``.

    Unit ``stop`` is resolved in :func:`simulate_book` after the stop
    distance is known (profit target only).
    """
    v = float(value)
    if v <= 0:
        return 0.0
    if str(unit or "points") == "atr":
        if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            return 0.0
        return v * float(atr_at_entry)
    return v


def simulate_book(
    ohlc: pd.DataFrame,
    frames: list[pd.DataFrame],
    book: dict[str, Any] | None = None,
    *,
    model_ids: list[str] | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> SimResult:
    """Execute consensus signals under the shared book knobs."""
    p = coerce_book_params(book)
    df = normalize_ohlc(ohlc)
    ids = list(model_ids or [str(f.attrs.get("model_id") or "ext") for f in frames])
    if df.empty or not frames:
        return SimResult(pd.DataFrame(), 0, {"empty": 1}, p, frames)
    n = len(df)
    for frame in frames:
        if len(frame) != n:
            raise ValueError("signal frames must align with ohlc")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    opn = df["open"].to_numpy(dtype="float64")
    tr = _true_range(high, low, close)
    atr = _rma(tr, int(p["atr_period"]) or DEFAULT_ATR_PERIOD)
    min_atr_v = float(p["min_atr"])
    min_atr_unit = str(p.get("min_atr_unit") or "points")
    if min_atr_unit == "atr":
        if min_atr_v <= 0:
            atr_ok = np.isfinite(atr)
        else:
            atr_ok = np.isfinite(atr) & np.isfinite(tr) & (tr >= (min_atr_v * atr))
    else:
        atr_ok = np.isfinite(atr) & (atr > min_atr_v)
    desired_all = consensus_signal(frames)
    sig_rows = [f["signal"].to_numpy(dtype="int8") for f in frames]
    sar_arrs = _sar_stop_arrays(frames)

    tz = ZoneInfo(str(p["timezone"]))
    local = df.index.tz_convert(tz)
    minutes = (local.hour.to_numpy() * 60 + local.minute.to_numpy()).astype(np.int32)
    sess0 = _hhmm_minutes(str(p["session_start"]))
    sess1 = _hhmm_minutes(str(p["session_end"]))
    trade0 = _hhmm_minutes(str(p["trade_start"]))
    trade1 = _hhmm_minutes(str(p["trade_end"]))
    in_session = (minutes >= sess0) & (minutes < sess1)
    can_enter = in_session & (minutes >= trade0) & (minutes < trade1)
    exit_time = minutes >= sess1
    chicago_dates = np.array([t.date() for t in local], dtype=object)
    holiday = np.array([not is_trading_day(d) for d in chicago_dates], dtype=bool)

    t_lo = pd.Timestamp(str(window_start), tz=tz).tz_convert("UTC") if window_start else None
    t_hi = (
        (pd.Timestamp(str(window_end), tz=tz) + pd.Timedelta(days=1)).tz_convert("UTC")
        if window_end else None
    )

    qty = int(p["qty"])
    slip = float(p["slippage_points"])
    cost = float(p["cost_per_trade"])
    pt_raw = float(p["profit_target_pts"])
    sl_raw = float(p["stop_loss_pts"])
    pt_unit = str(p.get("profit_target_unit") or "points")
    sl_unit = str(p.get("stop_loss_unit") or "points")
    signal_lock = bool(p.get("signal_lock", False))
    allow_long = bool(p.get("trade_long", True))
    allow_short = bool(p.get("trade_short", True))
    ts = df.index.to_numpy()

    pos = 0
    entry_i = -1
    entry_px = np.nan
    mae = 0.0
    mfe = 0.0
    lock_dir = 0
    trades: list[dict[str, Any]] = []
    skipped = {"holiday": 0, "warmup": 0, "outside_window": 0,
               "outside_trade_window": 0, "signal_lock": 0, "side_blocked": 0,
               "once_per_session": 0}
    once = any(bool((f.attrs or {}).get("once_per_session")) for f in frames)
    session_fills: set[object] = set()
    _lock_reasons = (
        "LG Profit Target", "LG Stop Loss", "ST Profit Target", "ST Stop Loss",
        "sar_stop",
    )

    def _in_window(i: int) -> bool:
        t = pd.Timestamp(ts[i])
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        if t_lo is not None and t < t_lo:
            return False
        if t_hi is not None and t >= t_hi:
            return False
        return True

    def _close_trade(i: int, px: float, reason: str) -> None:
        nonlocal pos, entry_i, entry_px, mae, mfe, lock_dir
        if pos == 0 or entry_i < 0:
            pos = 0
            return
        direction = 1 if pos > 0 else -1
        fill = float(px) - slip * direction
        pnl = (fill - entry_px) * direction * abs(pos) - cost
        hi_win = high[entry_i:i + 1]
        lo_win = low[entry_i:i + 1]
        if direction > 0:
            mae = float(np.nanmax(entry_px - lo_win))
            mfe = float(np.nanmax(hi_win - entry_px))
        else:
            mae = float(np.nanmax(hi_win - entry_px))
            mfe = float(np.nanmax(entry_px - lo_win))
        trades.append({
            "entry_ts": ts[entry_i],
            "exit_ts": ts[i],
            "side": "long" if direction > 0 else "short",
            "entry_px": float(entry_px),
            "exit_px": float(fill),
            "pnl_points": float(pnl),
            "mae_points": max(0.0, mae),
            "mfe_points": max(0.0, mfe),
            "holding_bars": int(i - entry_i),
            "exit_reason": reason,
            "entry_idx": int(entry_i),
            "exit_idx": int(i),
            "session_date": str(chicago_dates[entry_i]),
            "qty": int(abs(pos)),
        })
        pos = 0
        entry_i = -1
        entry_px = np.nan
        mae = 0.0
        mfe = 0.0
        if signal_lock and reason in _lock_reasons:
            lock_dir = direction
        if reason == "flatten_1445":
            lock_dir = 0

    def _want(direction: int) -> bool:
        if direction > 0 and not allow_long:
            skipped["side_blocked"] += 1
            return False
        if direction < 0 and not allow_short:
            skipped["side_blocked"] += 1
            return False
        if signal_lock and lock_dir != 0:
            if (direction > 0 and lock_dir > 0) or (direction < 0 and lock_dir < 0):
                skipped["signal_lock"] += 1
                return False
        return True

    def _bracket_pts(ei: int) -> tuple[float, float]:
        a = float(atr[ei]) if ei >= 0 else float("nan")
        stop_pts = _qty_in_points(sl_raw, sl_unit, a)
        if str(pt_unit) == "stop":
            target_pts = (
                float(pt_raw) * stop_pts
                if stop_pts > 0 and float(pt_raw) > 0 else 0.0
            )
        else:
            target_pts = _qty_in_points(pt_raw, pt_unit, a)
        return target_pts, stop_pts

    def _open_trade(i: int, direction: int, px: float) -> None:
        nonlocal pos, entry_i, entry_px, mae, mfe
        if not _want(direction):
            return
        if not _in_window(i):
            skipped["outside_window"] += 1
            return
        if not bool(can_enter[i]):
            skipped["outside_trade_window"] += 1
            return
        if once and chicago_dates[i] in session_fills:
            skipped["once_per_session"] += 1
            return
        pos = int(direction) * qty
        entry_i = i
        entry_px = float(px) + slip * direction
        mae = 0.0
        mfe = 0.0
        if once:
            session_fills.add(chicago_dates[i])

    merged_params = dict(p)
    for frame in frames:
        mid = str(frame.attrs.get("model_id") or "")
        mp = frame.attrs.get("params") if isinstance(frame.attrs.get("params"), dict) else {}
        for k, v in mp.items():
            merged_params[f"{mid}.{k}" if k in merged_params else k] = v
        merged_params.update({k: v for k, v in mp.items() if k not in merged_params})

    for i in range(n):
        if i > 0 and chicago_dates[i] != chicago_dates[i - 1]:
            lock_dir = 0

        closed_intrabar = False
        if pos != 0 and entry_i >= 0 and i > entry_i:
            direction = 1 if pos > 0 else -1
            sar_level = _sar_stop_level(sar_arrs, i, direction)
            target_pts, stop_pts = _bracket_pts(entry_i)
            hit = _bracket_fill(
                direction, float(entry_px), target_pts, stop_pts,
                opn[i], high[i], low[i],
                sar_stop=sar_level,
            )
            if hit is not None:
                fill_px, reason = hit
                _close_trade(i, fill_px, reason)
                closed_intrabar = True

        if exit_time[i]:
            if pos != 0:
                _close_trade(i, close[i], "flatten_1445")
            continue

        if closed_intrabar:
            continue

        if holiday[i] or not in_session[i]:
            if holiday[i] and in_session[i]:
                skipped["holiday"] += 1
            continue

        desired = int(desired_all[i])
        if lock_dir > 0 and desired < 0:
            lock_dir = 0
        elif lock_dir < 0 and desired > 0:
            lock_dir = 0
        if desired > 0 and not allow_long:
            skipped["side_blocked"] += 1
            desired = 0
        elif desired < 0 and not allow_short:
            skipped["side_blocked"] += 1
            desired = 0
        if desired == 0 and pos == 0:
            if all(int(row[i]) == 0 for row in sig_rows):
                skipped["warmup"] += 1
            continue

        ok = bool(atr_ok[i])
        if pos != 0:
            pos_dir = 1 if pos > 0 else -1
            if desired == pos_dir:
                continue
            reason = _signal_exit_reason(ids, pos_dir, desired)
            _close_trade(i, close[i], reason)
            if desired == -pos_dir and ok:
                _open_trade(i, desired, close[i])
            continue

        if desired != 0 and ok:
            _open_trade(i, desired, close[i])

    if pos != 0:
        _close_trade(n - 1, close[n - 1], "end_of_data")

    ledger = pd.DataFrame(trades)
    return SimResult(ledger, n, skipped, merged_params, frames)


def _expand_model_ids(model_id: str | None, stack: list[str] | None) -> list[str]:
    if stack:
        return [str(x) for x in stack]
    if not model_id:
        return ["psar"]
    compat = compat_stack_for(model_id)
    if compat:
        return compat
    return [model_id]


def simulate(
    ohlc: pd.DataFrame,
    params: dict[str, Any] | None = None,
    *,
    model_id: str | None = None,
    stack: list[str] | None = None,
    trade_start: str | None = None,
    trade_end: str | None = None,
) -> SimResult:
    """Run one engine or a stack on *ohlc* (UTC index, open/high/low/close).

    *params* may mix book + model keys; extras for the other engine are
    ignored. Default single engine is ``psar``. ``model_id='psar_ema20'``
    expands to ``psar`` + ``ema20``.
    """
    if ohlc is None or ohlc.empty:
        book = coerce_book_params(params)
        return SimResult(pd.DataFrame(), 0, {"empty": 1}, book, [])
    df = normalize_ohlc(ohlc)
    ids = _expand_model_ids(model_id, stack)
    book = coerce_book_params(params)
    frames = [
        build_signals(mid, df, params, book_ohlc=df, book=book)
        if mid == "timed_psar" else build_signals(mid, df, params)
        for mid in ids
    ]
    return simulate_book(
        df, frames, book, model_ids=ids,
        window_start=trade_start, window_end=trade_end,
    )
