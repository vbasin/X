"""OHLC + SAR / EMA / ATR / ZigZag series for the Backtest Chart tab.

Uses the same PostgreSQL ``databento.tbbo_1min`` lead-month path as the
external-model book. Overlays are computed on the **full** loaded series
(including warmup / Globex) so 08:30 values match the backtest; the
payload then drops CME holiday calendar dates (never displayed) and,
when ``session_only``, non-session bars. ATR is Wilder RMA of true range
using the book ``atr_period`` (same series as the min-ATR entry gate).
ZigZag High/Low (ADR-063) adds per-bar trend and confirmed pivot vertices
when that engine is in the stack.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src.common.cme_holidays import is_cme_holiday
from src.external_models.candles import CANDLE_TABLE, load_lead_ohlc
from src.external_models.catalog import coerce_chart_params, _hhmm_minutes
from src.external_models.pine_sar_ema import normalize_ohlc, tv_ema
from src.external_models.runner import (
    is_external_source,
    parse_stack,
    resolve_book_params,
    resolve_model_params,
)
from src.external_models.timeframe import (
    align_signal_frame,
    coerce_candle_tf,
    resample_ohlc,
    warmup_days_for,
)
from src.external_models.zigzag import zzhl_on_closed
from src.external_models.vwap_cross import _ct_frame as _vwap_ct_frame, _session_vwap
from src.features.psar import _rma, _true_range, sar_on_closed
from zoneinfo import ZoneInfo

_MAX_BARS = 80_000
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _unix(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def _finite(x: float) -> float | None:
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, 4)


def build_chart_payload(
    *,
    source: str | None = None,
    start: str | None = None,
    end: str | None = None,
    params: dict[str, Any] | None = None,
    book: dict[str, Any] | None = None,
    stack: list[dict[str, Any]] | None = None,
    session_only: bool = True,
    ohlc: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return compact column-arrays of candles + SAR / EMA / ATR / ZigZag."""
    for d in (start, end):
        if d and not _DATE_RE.match(str(d)):
            raise ValueError("dates must be YYYY-MM-DD")
    ext = bool(stack) or is_external_source(source)
    want_sar = True
    want_ema = False
    want_zz = False
    want_vwap = False
    sar_p = coerce_chart_params(params)
    ema_p = None
    zz_p = None
    vwap_p = None
    sar_tf = None
    ema_tf = None
    zz_tf = None
    book_p = coerce_chart_params(params)
    if ext:
        try:
            legs = parse_stack(source, stack)
        except ValueError:
            legs = []
        book_p = resolve_book_params(book, params)
        ids = [leg["model"] for leg in legs] if legs else []
        want_sar = (not ids) or any(mid in ("psar", "timed_psar") for mid in ids)
        want_ema = any(mid == "ema20" for mid in ids)
        want_zz = any(mid == "zigzag" for mid in ids)
        want_vwap = any(mid == "vwap_cross" for mid in ids)
        book_tf_default = coerce_candle_tf(book_p.get("timeframe") or "1min")
        for leg in legs:
            leg_tf = coerce_candle_tf(leg.get("timeframe") or book_tf_default)
            if leg["model"] == "psar" and sar_tf is None:
                sar_tf = leg_tf
                sar_p = {
                    **book_p,
                    **resolve_model_params(
                        "psar", hardened=leg.get("hardened"),
                        params=leg.get("params") or params,
                    ),
                }
            elif leg["model"] == "timed_psar" and sar_tf is None:
                timed_p = resolve_model_params(
                    "timed_psar", hardened=leg.get("hardened"),
                    params=leg.get("params") or params,
                )
                try:
                    sar_tf = coerce_candle_tf(timed_p.get("direction_tf") or "1day")
                except ValueError:
                    sar_tf = "1day"
                sar_p = {**book_p, **timed_p}
            elif leg["model"] == "ema20" and ema_tf is None:
                ema_tf = leg_tf
                ema_p = resolve_model_params(
                    "ema20", hardened=leg.get("hardened"),
                    params=leg.get("params") or params,
                )
            elif leg["model"] == "zigzag" and zz_tf is None:
                zz_tf = leg_tf
                zz_p = resolve_model_params(
                    "zigzag", hardened=leg.get("hardened"),
                    params=leg.get("params") or params,
                )
            elif leg["model"] == "vwap_cross" and vwap_p is None:
                vwap_p = resolve_model_params(
                    "vwap_cross", hardened=leg.get("hardened"),
                    params=leg.get("params") or params,
                )
        # vwap_cross exposes its own EMA + VWAP overlays for chart inspection.
        if vwap_p is not None and not want_ema:
            want_ema = True
            ema_p = vwap_p
            ema_tf = "1min"
        tf = book_tf_default
    else:
        tf = "1min"
        want_sar = True
        want_ema = False
        want_zz = False

    root = "NQ"

    if not start or not end:
        raise ValueError("chart requires start and end dates")

    if ohlc is None:
        finest = load_lead_ohlc(
            start, end, root=root, timeframe="1min",
            warmup_days=warmup_days_for(tf, sar_tf, ema_tf, zz_tf),
        )
    else:
        finest = normalize_ohlc(ohlc)
    if finest is None or finest.empty:
        raise ValueError(f"no {CANDLE_TABLE} rows for {start}→{end}")
    ohlc = resample_ohlc(finest, tf) if tf != "1min" else finest

    n = len(ohlc)
    atr_period = max(1, int(book_p.get("atr_period") or 8))
    min_atr = book_p.get("min_atr")
    try:
        min_atr_f = float(min_atr) if min_atr is not None else None
    except (TypeError, ValueError):
        min_atr_f = None
    if min_atr_f is not None and not math.isfinite(min_atr_f):
        min_atr_f = None
    high_a = ohlc["high"].to_numpy(dtype="float64")
    low_a = ohlc["low"].to_numpy(dtype="float64")
    close_a = ohlc["close"].to_numpy(dtype="float64")
    atr_full = _rma(_true_range(high_a, low_a, close_a), atr_period)

    def _engine_ohlc(engine_tf: str | None) -> pd.DataFrame:
        etf = coerce_candle_tf(engine_tf or tf)
        if etf == "1min":
            return finest
        if etf == tf:
            return ohlc
        return resample_ohlc(finest, etf)

    def _align_df(src: pd.DataFrame, engine_tf: str | None) -> pd.DataFrame:
        etf = coerce_candle_tf(engine_tf or tf)
        if etf == tf and len(src) == len(ohlc) and src.index.equals(ohlc.index):
            return src
        return align_signal_frame(src, ohlc.index, etf, tf)

    sar_df = None
    if want_sar:
        src_ohlc = _engine_ohlc(sar_tf)
        sar_raw = sar_on_closed(
            src_ohlc,
            start=float(sar_p.get("start") or 0.02),
            increment=float(sar_p.get("increment") or 0.02),
            maximum=float(sar_p.get("maximum") or 0.2),
            atr_period=atr_period,
        )
        sar_df = _align_df(sar_raw, sar_tf)
    ema_a: list[float | None]
    if want_ema:
        src_ohlc = _engine_ohlc(ema_tf)
        close = src_ohlc["close"].to_numpy(dtype="float64")
        length = int((ema_p or {}).get("ema_length") or 20)
        ema_src = tv_ema(close, length) if length >= 2 else np.full(len(src_ohlc), np.nan)
        ema_frame = pd.DataFrame({"ema": ema_src}, index=src_ohlc.index)
        ema_aligned = _align_df(ema_frame, ema_tf)
        ema_a = [_finite(x) for x in ema_aligned["ema"].to_numpy(dtype="float64")]
    else:
        ema_a = [None] * n
    zz_df = None
    if want_zz:
        zp = zz_p or {}

        def _num(key: str, default: float, cast=float):
            raw = zp.get(key)
            if raw is None:
                return cast(default)
            return cast(raw)

        src_ohlc = _engine_ohlc(zz_tf)
        zz_raw = zzhl_on_closed(
            src_ohlc,
            percentage_reversal=_num("percentage_reversal", 2.5),
            absolute_reversal=_num("absolute_reversal", 0.0),
            atr_length=_num("atr_length", 5, int),
            atr_reversal=_num("atr_reversal", 1.0),
            tick_reversal=_num("tick_reversal", 0, int),
            tick_size=_num("tick_size", 0.25),
        )
        zz_df = _align_df(zz_raw, zz_tf)

    vwap_full = None
    if want_vwap:
        vwap_start = _hhmm_minutes(str((vwap_p or {}).get("session_start") or book_p.get("session_start") or "08:30"))
        # VWAP is anchored to session_start on the finest series (1min) so the
        # cumulative sum matches the strategy exactly; then aligned to the
        # display timeframe if the book is coarser.
        src = _vwap_ct_frame(finest)
        v_arr = _session_vwap(src, session_start_min=vwap_start)
        v_frame = pd.DataFrame({"vwap": v_arr}, index=finest.index)
        vwap_full = _align_df(v_frame, "1min")

    tz = ZoneInfo(str(book_p.get("timezone") or "America/Chicago"))
    idx = ohlc.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(tz)
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    sess0 = _hhmm_minutes(str(book_p.get("session_start") or "08:30"))
    sess1 = _hhmm_minutes(str(book_p.get("session_end") or "14:45"))
    in_sess = (minutes >= sess0) & (minutes < sess1)
    d0 = pd.Timestamp(str(start)).date()
    d1 = pd.Timestamp(str(end)).date()
    days = np.array([t.date() for t in local], dtype=object)
    in_window = (days >= d0) & (days <= d1)
    holiday_ok: dict[object, bool] = {}
    not_holiday = np.empty(len(days), dtype=bool)
    for i, d in enumerate(days):
        flag = holiday_ok.get(d)
        if flag is None:
            flag = not is_cme_holiday(d)
            holiday_ok[d] = flag
        not_holiday[i] = flag

    keep = in_window & not_holiday
    note = None
    if session_only:
        keep = keep & in_sess
    n_keep = int(np.count_nonzero(keep))
    if n_keep > _MAX_BARS and not session_only:
        keep = in_window & not_holiday & in_sess
        n_keep = int(np.count_nonzero(keep))
        note = f"showing session hours only ({n_keep} bars; cap {_MAX_BARS})"
        session_only = True
    if n_keep > _MAX_BARS:
        raise ValueError(
            f"chart window has {n_keep} bars (cap {_MAX_BARS}). "
            "Pick a shorter date range.")

    ix = np.flatnonzero(keep)
    sl = ohlc.iloc[ix]
    t = [_unix(x) for x in sl.index]
    o = [_finite(x) for x in sl["open"].to_numpy(dtype="float64")]
    h = [_finite(x) for x in sl["high"].to_numpy(dtype="float64")]
    l = [_finite(x) for x in sl["low"].to_numpy(dtype="float64")]
    c = [_finite(x) for x in sl["close"].to_numpy(dtype="float64")]
    vol = [int(v) if math.isfinite(v) else 0
           for v in sl["volume"].to_numpy(dtype="float64")] if "volume" in sl.columns else [0] * len(sl)
    if sar_df is not None:
        ss = sar_df.iloc[ix] if len(sar_df) == len(ohlc) else sar_df.reindex(sl.index)
        sar_a = [_finite(x) for x in ss["sar"].to_numpy(dtype="float64")] if "sar" in ss.columns else [None] * len(sl)
        nsar_a = [_finite(x) for x in ss["next_sar"].to_numpy(dtype="float64")] if "next_sar" in ss.columns else [None] * len(sl)
        trend_a = []
        if "trend" in ss.columns:
            for x in ss["trend"].to_numpy(dtype="float64"):
                if not math.isfinite(x):
                    trend_a.append(0)
                else:
                    trend_a.append(1 if x > 0 else -1)
        else:
            trend_a = [0] * len(sl)
    else:
        sar_a = [None] * len(sl)
        nsar_a = [None] * len(sl)
        trend_a = [0] * len(sl)
    ema_keep = [ema_a[int(i)] for i in ix] if ema_a else [None] * len(sl)
    atr_a = [_finite(atr_full[int(i)]) for i in ix]
    if zz_df is not None:
        zs = zz_df.iloc[ix] if len(zz_df) == len(ohlc) else zz_df.reindex(sl.index)
        zz_line = [_finite(x) for x in zs["zz"].to_numpy(dtype="float64")]
        raw_trend = [int(x) for x in zs["signal"].to_numpy(dtype="int8")]
        engine_tf = coerce_candle_tf(zz_tf or tf)
        if "signal_fresh" in zs.columns and engine_tf != tf:
            fr = zs["signal_fresh"].to_numpy(dtype=bool)
            zz_trend = [t if bool(fr[i]) else 0 for i, t in enumerate(raw_trend)]
        else:
            zz_trend = raw_trend
        zz_max = [_finite(x) for x in zs["max_price_h"].to_numpy(dtype="float64")]
        zz_min = [_finite(x) for x in zs["min_price_l"].to_numpy(dtype="float64")]
        zz_need = [_finite(x) for x in zs["reversal_pts"].to_numpy(dtype="float64")]
    else:
        zz_line = [None] * len(sl)
        zz_trend = [0] * len(sl)
        zz_max = [None] * len(sl)
        zz_min = [None] * len(sl)
        zz_need = [None] * len(sl)

    if vwap_full is not None:
        vs = vwap_full.iloc[ix] if len(vwap_full) == len(ohlc) else vwap_full.reindex(sl.index)
        vwap_a = [_finite(x) for x in vs["vwap"].to_numpy(dtype="float64")]
    else:
        vwap_a = [None] * len(sl)

    return {
        "table": CANDLE_TABLE,
        "timeframe": tf,
        "timezone": str(book_p.get("timezone") or "America/Chicago"),
        "session_start": str(book_p.get("session_start") or "08:30"),
        "session_end": str(book_p.get("session_end") or "14:45"),
        "trade_start": str(book_p.get("trade_start") or book_p.get("session_start") or "08:30"),
        "trade_end": str(book_p.get("trade_end") or book_p.get("session_end") or "14:45"),
        "session_only": bool(session_only),
        "holidays_omitted": True,
        "n": len(t),
        "t": t,
        "o": o,
        "h": h,
        "l": l,
        "c": c,
        "v": vol,
        "sar": sar_a,
        "next_sar": nsar_a,
        "trend": trend_a,
        "ema": ema_keep,
        "atr": atr_a,
        "atr_period": atr_period,
        "min_atr": None if min_atr_f is None else round(min_atr_f, 4),
        "min_atr_unit": str(book_p.get("min_atr_unit") or "points"),
        "zz": zz_line,
        "zz_trend": zz_trend,
        "zz_max": zz_max,
        "zz_min": zz_min,
        "zz_need": zz_need,
        "has_sar": bool(want_sar),
        "has_ema": bool(want_ema),
        "has_zz": bool(want_zz),
        "has_vwap": bool(want_vwap),
        "vwap": vwap_a,
        "note": note,
        "source": source,
        "window": {"start": start, "end": end},
    }
