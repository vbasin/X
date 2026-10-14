"""Live Chart: NQ candlesticks from TBBO + optional TBBO feature overlays.

Instrument is always /NQ (lead-month). Candles come from Databento TBBO
caggs where they exist; 5s is ``time_bucket`` on ticks; 15min is resampled
from 5m. CVD / delta use cagg ``buy_volume`` − ``sell_volume`` for the
whole lookback (reset at session / BOD / RTH). OFI and VWAP still load
prints (capped). Volume Profile overlays (single / recurring) are tick
aggregates, not hive catalog columns. Multi-EMA overlays are close-based
span-N lines (current bar included in the lookback). Session timeframe
may be Custom (From/To CT clocks). Single VP pins a session-end calendar
date.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg

from src.common.contract_calendar import active_contract, active_contracts_in_range
from src.common.db import get_dsn
from src.features.tape import tick_value_area

CT = ZoneInfo("America/Chicago")
ROOT = "NQ"
MAX_BARS = 50_000
MAX_TICK_SECONDS = 36 * 3600
MAX_VP_SECONDS = 20 * 24 * 3600
MAX_PRINTS = 8_000
PRINT_MIN_DEFAULT = 10
PRINT_MIN_MAX = 10_000
MAX_VP_BINS = 8_000
MAX_VP_SPECS = 8
MAX_VP_EXPANDED = 32
MAX_EMA_SPECS = 8
EMA_LOOKBACK_MIN = 1
EMA_LOOKBACK_MAX = 500
EMA_LOOKBACK_DEFAULT = 21
EMA_PRESETS: tuple[int, ...] = (9, 21, 50, 100, 200)
MAX_STOCH_SPECS = 8
STOCH_K_MIN = 1
STOCH_K_MAX = 200
STOCH_D_MIN = 1
STOCH_D_MAX = 50
STOCH_K_DEFAULT = 9
STOCH_D_DEFAULT = 3
# (k_period, d_period) presets — TOS StochasticFast studies.
STOCH_PRESETS: tuple[tuple[int, int], ...] = ((9, 3), (14, 3), (40, 4))

# Parabolic SAR AF knobs — same defaults/bounds as PSAR_PARAM_SPEC (ext model).
SAR_START_DEFAULT = 0.02
SAR_INCREMENT_DEFAULT = 0.02
SAR_MAXIMUM_DEFAULT = 0.2
SAR_AF_MIN = 0.001
SAR_AF_MAX = 1.0
SAR_MAXIMUM_MIN = 0.01
SAR_MAXIMUM_MAX = 1.0
# VWAP Bollinger bands: volume-weighted σ around session VWAP (ADR-090 / native 5m).
VWAP_BB_STDEV = 2.0
VWAP_BB_KEYS: tuple[str, ...] = ("vw_bb_upper", "vw_bb_lower")
_VWAP_SERIES_KEYS: tuple[str, ...] = ("vw_vwap", "vw_vwap_dist", "vw_bb_upper", "vw_bb_lower")

TICK_SIZE = 0.25
VA_PCT = 0.70
MFI_PERIOD = 14
RTH_START_HM = (8, 30)
RTH_END_HM = (15, 0)
RTH_SQL = (
    " AND ((ts_event AT TIME ZONE 'America/Chicago')::time >= TIME '08:30:00'"
    " AND (ts_event AT TIME ZONE 'America/Chicago')::time < TIME '15:00:00')"
)
STMT_MS = 45_000
ORIGINS: tuple[str, ...] = ("session", "bod", "mkt")
SVP_ORIGINS: tuple[str, ...] = ("session", "bod", "mkt", "premarket")
ORIGIN_LABELS = {
    "session": "Session (5:00pm CT)",
    "bod": "BOD (midnight CT)",
    "mkt": "MKT open (8:30am CT)",
    "premarket": "Premarket (5:00pm–8:30am CT)",
}
SESSION_TFS: tuple[str, ...] = ("full_session", "cash", "premarket", "bod", "custom")
SESSION_TF_LABELS = {
    "full_session": "Full session (5:00pm–4:00pm CT)",
    "cash": "Cash session (8:30am–3:00pm CT)",
    "premarket": "Premarket (8:00am–8:30am CT)",
    "bod": "BOD (midnight–midnight CT)",
    "custom": "Custom",
}
_SESSION_TF_ALIASES = {
    "full_session": "full_session", "full": "full_session",
    "session": "full_session", "globex": "full_session", "eth": "full_session",
    "cash": "cash", "cash_session": "cash", "rth": "cash", "mkt": "cash",
    "premarket": "premarket", "pre": "premarket",
    "bod": "bod", "calendar": "bod", "midnight": "bod", "day": "bod",
    "custom": "custom", "from_to": "custom", "user": "custom",
}
VP_KINDS: tuple[str, ...] = ("current", "recurring", "level_blend")
VP_KIND_LABELS = {
    "current": "Single",
    "recurring": "Recurring",
    "level_blend": "Level Blend",
}
# ADR-126 LevelConfig defaults — lookback / KDE resolution for Level Blend VP.
LEVEL_BLEND_LOOKBACK_DEFAULT = 20
LEVEL_BLEND_LOOKBACK_MIN = 1
LEVEL_BLEND_LOOKBACK_MAX = 120
LEVEL_BLEND_SIGMA_DEFAULT = 5.0
LEVEL_BLEND_BUFFER_DEFAULT = 1.0
DEFAULT_CUSTOM_FROM: tuple[int, int] = (8, 30)
DEFAULT_CUSTOM_TO: tuple[int, int] = (15, 0)
_HM_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::\d{2})?$")
VP_PLACEMENTS: tuple[str, ...] = ("overlay", "left_frame", "right_frame")
VP_PLACEMENT_LABELS = {
    "overlay": "Overlay",
    "left_frame": "Left",
    "right_frame": "Right",
}
VP_EXTEND: tuple[int, ...] = (0, 1, 2, 3)
VP_EXTEND_LABELS = {0: "Off", 1: "1 sess", 2: "2 sess", 3: "All sess"}
VP_SIDES: tuple[str, ...] = ("left", "right")
VP_RECUR: tuple[str, ...] = ("daily", "weekly")
VP_RECUR_LABELS = {"daily": "Daily", "weekly": "Weekly"}
VP_HIST_STYLES: tuple[str, ...] = ("volume", "binned")
VP_HIST_STYLE_LABELS = {
    "volume": "Tick histogram",
    "binned": "Grouped ticks",
}
GROUP_TICKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
GROUP_TICKS_DEFAULT = 4
VP_DISPLAY_MODES: tuple[str, ...] = (
    "total", "total_split", "total_delta_split", "buy", "sell", "delta", "none",
)
VP_DISPLAY_MODE_LABELS = {
    "total": "Total volume",
    "total_split": "Total volume split",
    "total_delta_split": "Total Volume delta split",
    "buy": "Buy volume",
    "sell": "Sell volume",
    "delta": "Delta volume",
    "none": "None",
}
HIST_WIDTH_DEFAULT = 0.25
LVN_PCT_DEFAULT = 0.30
LVN_PCT_MIN = 0.10
LVN_PCT_MAX = 0.50
LVN_GROUPS_DEFAULT = 1
LVN_GROUPS_MIN = 1
LVN_GROUPS_MAX = 20
HVN_METHODS: tuple[str, ...] = ("off", "adaptive_kde")
HVN_METHOD_DEFAULT = "off"
KDE_BW_DEFAULT = 2.0
KDE_BW_MIN = 1.0
KDE_BW_MAX = 12.0
KDE_PROM_DEFAULT = 0.05
KDE_PROM_MIN = 0.02
KDE_PROM_MAX = 0.50
KDE_VALLEY_DEFAULT = 0.05
KDE_VALLEY_MIN = 0.02
KDE_VALLEY_MAX = 0.50
KDE_VALLEY_WIDTH_DEFAULT = 2
KDE_VALLEY_WIDTH_MIN = 1
KDE_VALLEY_WIDTH_MAX = 20
KDE_VOL_DEFAULT = 0.05
KDE_VOL_MIN = 0.01
KDE_VOL_MAX = 0.50
KDE_SEP_DEFAULT = 2
KDE_SEP_MIN = 1
KDE_SEP_MAX = 20
KDE_MAX_ZONES_DEFAULT = 5
KDE_MAX_ZONES_MIN = 1
KDE_MAX_ZONES_MAX = 10
_ORIGIN_ALIASES = {
    "session": "session", "globex": "session", "eth": "session",
    "bod": "bod", "midnight": "bod", "day": "bod", "calendar": "bod",
    "mkt": "mkt", "rth": "mkt", "open": "mkt", "mkt_open": "mkt",
    "market": "mkt", "cash": "mkt",
    "premarket": "premarket", "pre": "premarket", "overnight": "premarket",
    "eth_pre": "premarket",
}

TIMEFRAMES: tuple[str, ...] = (
    "5s", "15s", "1min", "5min", "15min", "1hour", "1day",
)
_TF_ALIASES = {
    "5s": "5s", "5sec": "5s",
    "15s": "15s", "15sec": "15s",
    "1m": "1min", "1min": "1min",
    "5m": "5min", "5min": "5min",
    "15m": "15min", "15min": "15min",
    "1h": "1hour", "1hour": "1hour", "60min": "1hour",
    "1d": "1day", "1day": "1day",
}
_PANDAS_FREQ = {
    "5s": "5s", "15s": "15s", "1min": "1min", "5min": "5min",
    "15min": "15min", "1hour": "1h", "1day": "1D",
}
_CAGG = {
    "15s": "databento.tbbo_15s",
    "1min": "databento.tbbo_1min",
    "5min": "databento.tbbo_5m",
    "1hour": "databento.tbbo_1h",
    "1day": "databento.tbbo_1d",
}
_LOOKBACK = {
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "session": None,
    "1d": timedelta(days=1),
    "2d": timedelta(days=2),
    "3d": timedelta(days=3),
    "5d": timedelta(days=5),
    "7d": timedelta(days=7),
    "10d": timedelta(days=10),
    "20d": timedelta(days=20),
    "30d": timedelta(days=30),
    "60d": timedelta(days=60),
    "90d": timedelta(days=90),
    "1y": timedelta(days=365),
}
_LOOKBACK_LABELS = {
    "30m": "30 min", "1h": "1 hour", "2h": "2 hours", "4h": "4 hours",
    "8h": "8 hours", "12h": "12 hours", "session": "session",
    "1d": "1 day", "2d": "2 days", "3d": "3 days", "5d": "5 days",
    "7d": "7 days", "10d": "10 days", "20d": "20 days", "30d": "30 days",
    "60d": "60 days", "90d": "90 days", "1y": "1 year",
}
# No per-TF lookback caps (ADR-128). Window length is bounded only by the
# 5-year parse ceiling; responses still clip to MAX_BARS (newest retained).
# 5s still scans databento.tbbo; other TFs are caggs.
MAX_LOOKBACK_DELTA = timedelta(days=1825)
_DURATION_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*"
    r"(m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|"
    r"d|day|days|"
    r"w|wk|wks|week|weeks|"
    r"y|yr|yrs|year|years)$",
    re.IGNORECASE,
)
# Inclusive CT calendar dates: 2026-08-01..2026-08-15 or "2026-08-01 to 2026-08-15".
_RANGE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s*(?:\.\.|to)\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_UNIT_FAMILY = {
    "m": "minutes", "min": "minutes", "mins": "minutes",
    "minute": "minutes", "minutes": "minutes",
    "h": "hours", "hr": "hours", "hrs": "hours",
    "hour": "hours", "hours": "hours",
    "d": "days", "day": "days", "days": "days",
    "w": "weeks", "wk": "weeks", "wks": "weeks",
    "week": "weeks", "weeks": "weeks",
    "y": "years", "yr": "years", "yrs": "years",
    "year": "years", "years": "years",
}
_FAMILY_SUFFIX = {
    "minutes": "m", "hours": "h", "days": "d", "weeks": "w", "years": "y",
}

# pane: price = overlaid on candles; osc = own pane below volume (ADR-112);
# canvas = drawn on the price pane. of_cvd / of_delta use cagg buy−sell
# (needs=flow), not the 36h tick cap. zero_line = dashed y=0 on that pane.
OVERLAYS: tuple[dict[str, str], ...] = (
    {"key": "of_delta", "label": "Delta Volume", "group": "Order flow",
     "pane": "osc", "needs": "flow", "zero_line": "1"},
    {"key": "of_cvd", "label": "CVD", "group": "Order flow",
     "pane": "osc", "needs": "flow", "params": "cvd", "zero_line": "1"},
    {"key": "of_ofi", "label": "OFI", "group": "Order flow",
     "pane": "osc", "needs": "ticks", "zero_line": "1"},
    {"key": "of_ofi_cum", "label": "OFI_CUM", "group": "Order flow",
     "pane": "osc", "needs": "flow", "params": "ofi", "zero_line": "1"},
    {"key": "vw_vwap", "label": "VWAP", "group": "Price Action",
     "pane": "price", "needs": "ticks"},
    {"key": "vw_vwap_dist", "label": "VWAP distance", "group": "Price Action",
     "pane": "osc", "needs": "ticks", "zero_line": "1"},
    {"key": "cs_clv", "label": "CLV", "group": "Price Action",
     "pane": "osc", "needs": "ohlc", "zero_line": "1"},
    {"key": "mf_mfi", "label": "MFI (14)", "group": "Money flow",
     "pane": "osc", "needs": "ohlc"},
    {"key": "ad_line", "label": "A/D line", "group": "Money flow",
     "pane": "osc", "needs": "ohlc", "params": "ad", "zero_line": "1"},
    {"key": "so_sync_score_all", "label": "Stoch sync (all)", "group": "Stochastic",
     "pane": "osc", "needs": "ohlc", "zero_line": "1"},
    {"key": "so_sync_score_9_60", "label": "Stoch sync (fast+slow)", "group": "Stochastic",
     "pane": "osc", "needs": "ohlc", "zero_line": "1"},
    {"key": "ps_sar", "label": "Parabolic SAR", "group": "Models",
     "pane": "price", "needs": "ohlc", "params": "sar"},
    {"key": "lt_prints", "label": "Large prints", "group": "Order flow",
     "pane": "canvas", "needs": "prints", "params": "prints"},
    {"key": "lt_mbo_orders", "label": "Order map (MBO)", "group": "Order flow",
     "pane": "canvas", "needs": "mbo", "params": "mbo"},
)

_OVERLAY_BY_KEY = {o["key"]: o for o in OVERLAYS}


def coerce_tf(value: str | None) -> str:
    raw = str(value or "1min").strip().lower()
    tf = _TF_ALIASES.get(raw)
    if tf not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {list(TIMEFRAMES)}")
    return tf


def pandas_freq(tf: str) -> str:
    return _PANDAS_FREQ[coerce_tf(tf)]


def parse_bool(value: Any, default: bool = True) -> bool:
    """Query/JSON flag. ``None`` / empty → *default*."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(default)


def rth_bar_mask(buckets: pd.DatetimeIndex) -> np.ndarray:
    """True when the bar *open* is inside cash 08:30–15:00 America/Chicago."""
    if buckets is None or len(buckets) == 0:
        return np.zeros(0, dtype=bool)
    local = pd.DatetimeIndex(pd.to_datetime(buckets, utc=True)).tz_convert(CT)
    minutes = local.hour * 60 + local.minute
    start_m = RTH_START_HM[0] * 60 + RTH_START_HM[1]
    end_m = RTH_END_HM[0] * 60 + RTH_END_HM[1]
    return (minutes >= start_m) & (minutes < end_m)


def globex_overnight_windows(
    start: datetime, end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Half-open Globex overnight windows overlapping ``[start, end)``.

    Each window is ``[D 17:00 America/Chicago, D+1 08:30)``. Live Chart
    night **Shade** paints these instants on the time axis (not bar
    centers / barSpacing).
    """
    start_u = _as_utc(start).to_pydatetime()
    end_u = _as_utc(end).to_pydatetime()
    if start_u.tzinfo is None:
        start_u = start_u.replace(tzinfo=timezone.utc)
    else:
        start_u = start_u.astimezone(timezone.utc)
    if end_u.tzinfo is None:
        end_u = end_u.replace(tzinfo=timezone.utc)
    else:
        end_u = end_u.astimezone(timezone.utc)
    d0 = start_u.astimezone(CT).date() - timedelta(days=1)
    d1 = end_u.astimezone(CT).date()
    out: list[tuple[datetime, datetime]] = []
    d = d0
    while d <= d1:
        a = datetime(d.year, d.month, d.day, 17, 0, tzinfo=CT).astimezone(timezone.utc)
        nxt = d + timedelta(days=1)
        b = datetime(nxt.year, nxt.month, nxt.day, 8, 30, tzinfo=CT).astimezone(timezone.utc)
        if b > start_u and a < end_u:
            out.append((a, b))
        d += timedelta(days=1)
    return out


def non_rth_runs(buckets: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index runs outside cash 08:30–15:00 CT.

    Used when hiding overnight candles (Display off). Shade itself uses
    :func:`globex_overnight_windows` (17:00–08:30 CT).
    """
    if buckets is None or len(buckets) == 0:
        return []
    night = ~rth_bar_mask(buckets)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_night in enumerate(night):
        if bool(is_night):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(len(night))))
    return runs


def filter_rth_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    if ohlc is None or ohlc.empty or "bucket" not in ohlc.columns:
        return ohlc if ohlc is not None else pd.DataFrame()
    idx = pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True))
    return ohlc.loc[rth_bar_mask(idx)].copy()


def filter_rth_ticks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ts_event" not in df.columns:
        return df if df is not None else pd.DataFrame()
    ts = pd.to_datetime(df["ts_event"], utc=True)
    local = ts.dt.tz_convert(CT)
    minutes = local.dt.hour * 60 + local.dt.minute
    start_m = RTH_START_HM[0] * 60 + RTH_START_HM[1]
    end_m = RTH_END_HM[0] * 60 + RTH_END_HM[1]
    return df.loc[(minutes >= start_m) & (minutes < end_m)].copy()


def coerce_origin(
    value: str | None,
    *,
    default: str = "session",
    allowed: tuple[str, ...] | None = None,
) -> str:
    raw = str(value if value is not None else default).strip().lower()
    if not raw:
        raw = default
    key = _ORIGIN_ALIASES.get(raw, raw)
    ok = allowed if allowed is not None else ORIGINS
    if key not in ok:
        raise ValueError(f"origin must be one of {list(ok)}")
    return key


def origin_period_key(idx: pd.DatetimeIndex, origin: str) -> pd.DatetimeIndex:
    """Midnight CT of the origin period that contains each timestamp.

    session = last 17:00 CT, bod = last 00:00 CT, mkt = last 08:30 CT.
    """
    origin = coerce_origin(origin, allowed=SVP_ORIGINS)
    local = pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert(CT)
    if origin == "bod":
        cut_h, cut_m = 0, 0
    elif origin == "mkt":
        cut_h, cut_m = 8, 30
    else:
        # session and premarket both reset at 17:00 CT
        cut_h, cut_m = 17, 0
    before = (local.hour < cut_h) | ((local.hour == cut_h) & (local.minute < cut_m))
    midnight = local.normalize()
    shifted = midnight.where(~before, midnight - pd.DateOffset(days=1))
    return pd.DatetimeIndex(shifted)


def profile_window(
    ohlc: pd.DataFrame | None,
    end: datetime,
    origin: str,
) -> tuple[datetime, datetime]:
    """SVP window: origin of the last candle through *end*.

    Wall-clock ``now`` after the Globex close (or on the weekend) would
    open an empty session. Anchor to the last bar so the profile is the
    session the chart is actually showing.
    """
    origin = coerce_origin(origin, allowed=SVP_ORIGINS)
    ref = end
    if ohlc is not None and not ohlc.empty and "bucket" in ohlc.columns:
        last = _as_utc(ohlc["bucket"].iloc[-1]).to_pydatetime()
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        else:
            last = last.astimezone(timezone.utc)
        ref = last
    start = origin_start(ref, "session" if origin == "premarket" else origin)
    stop = end
    if origin == "premarket":
        chi = start.astimezone(CT)
        nxt = chi.date() + timedelta(days=1)
        rth = datetime(nxt.year, nxt.month, nxt.day, 8, 30, tzinfo=CT).astimezone(timezone.utc)
        if rth < stop:
            stop = rth
    if stop <= start:
        stop = start + timedelta(seconds=1)
    return start, stop


def origin_start(ts: datetime | pd.Timestamp, origin: str) -> datetime:
    """UTC datetime of the most recent origin clock ≤ *ts*."""
    origin = coerce_origin(origin, allowed=SVP_ORIGINS)
    t = _as_utc(ts)
    key = origin_period_key(pd.DatetimeIndex([t]), origin)[0]
    chi = key.tz_convert(CT)
    if origin == "bod":
        start = chi
    elif origin == "mkt":
        start = chi.replace(hour=8, minute=30, second=0, microsecond=0)
    else:
        start = chi.replace(hour=17, minute=0, second=0, microsecond=0)
    return start.tz_convert("UTC").to_pydatetime()


def coerce_session_tf(value: str | None, *, default: str = "full_session") -> str:
    raw = str(value if value is not None else default).strip().lower()
    if not raw:
        raw = default
    key = _SESSION_TF_ALIASES.get(raw, raw)
    if key not in SESSION_TFS:
        raise ValueError(f"session_tf must be one of {list(SESSION_TFS)}")
    return key


def coerce_vp_extend(item: dict[str, Any] | None) -> int:
    """0 = hist only, 1 = this session forward, 2 = two sessions forward, 3 = all visible forward."""
    item = item or {}
    raw = item.get("extend")
    if raw is None:
        raw = item.get("extend_span")
    if raw is not None and raw != "":
        if isinstance(raw, bool):
            return 1 if raw else 0
        key = str(raw).strip().lower().replace("_", " ")
        aliases = {
            "0": 0, "off": 0, "none": 0, "false": 0, "no": 0,
            "1": 1, "1 sess": 1, "1sess": 1, "session": 1, "single": 1,
            "true": 1, "yes": 1, "on": 1,
            "2": 2, "2 sess": 2, "2sess": 2, "two": 2,
            "3": 3, "all": 3, "all sess": 3, "allsess": 3, "*": 3,
        }
        if key in aliases:
            return aliases[key]
        try:
            n = int(float(key))
        except ValueError as exc:
            raise ValueError("extend must be 0, 1, 2, or 3") from exc
        if n not in VP_EXTEND:
            raise ValueError("extend must be 0, 1, 2, or 3")
        return n
    lines = item.get("extend_lines")
    va = item.get("extend_va")
    if lines is None and va is None:
        return 1
    on = False
    if lines is not None:
        on = on or parse_bool(lines, default=True)
    if va is not None:
        on = on or parse_bool(va, default=False)
    return 1 if on else 0


def parse_vp_n(raw: Any, *, fallback: int) -> int:
    if raw is None or raw == "":
        return max(1, int(fallback))
    try:
        n = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("n must be a positive integer") from exc
    if n < 1:
        raise ValueError("n must be a positive integer")
    return n


def format_session_hm(hm: tuple[int, int] | None) -> str | None:
    if hm is None:
        return None
    return f"{int(hm[0]):02d}:{int(hm[1]):02d}"


def parse_session_hm(
    value: Any,
    *,
    field: str = "session_from",
    default: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Parse a CT clock ``HH:MM``. Tuples and datetime-local strings are accepted."""
    if value is None or value == "":
        return default
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        hour, minute = int(value[0]), int(value[1])
    else:
        raw = str(value).strip()
        if "T" in raw:
            raw = raw.split("T", 1)[1]
        raw = raw.strip()
        if len(raw) >= 5 and raw[2] == ":":
            raw = raw[:5]
        matched = _HM_RE.match(raw)
        if not matched:
            raise ValueError(f"{field} must be HH:MM")
        hour, minute = int(matched.group(1)), int(matched.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{field} must be a valid clock time")
    return (hour, minute)


def parse_session_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(CT).date() if raw.tzinfo else raw.date()
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("session_date must be YYYY-MM-DD") from exc


def _chi_dt(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=CT).astimezone(timezone.utc)


def _custom_window_for_end_date(
    end_date: date,
    from_hm: tuple[int, int],
    to_hm: tuple[int, int],
) -> tuple[datetime, datetime]:
    fh, fm = from_hm
    th, tm = to_hm
    if fh * 60 + fm < th * 60 + tm:
        return _chi_dt(end_date, fh, fm), _chi_dt(end_date, th, tm)
    prev = end_date - timedelta(days=1)
    return _chi_dt(prev, fh, fm), _chi_dt(end_date, th, tm)


def session_tf_window_for_end_date(
    end_date: date,
    session_tf: str,
    *,
    from_hm: tuple[int, int] | None = None,
    to_hm: tuple[int, int] | None = None,
) -> tuple[datetime, datetime]:
    """Half-open [start, end) whose session-end calendar date is *end_date* (CT)."""
    session_tf = coerce_session_tf(session_tf)
    if session_tf == "full_session":
        prev = end_date - timedelta(days=1)
        return _chi_dt(prev, 17, 0), _chi_dt(end_date, 16, 0)
    if session_tf == "cash":
        return _chi_dt(end_date, 8, 30), _chi_dt(end_date, 15, 0)
    if session_tf == "premarket":
        return _chi_dt(end_date, 8, 0), _chi_dt(end_date, 8, 30)
    if session_tf == "custom":
        return _custom_window_for_end_date(
            end_date,
            from_hm or DEFAULT_CUSTOM_FROM,
            to_hm or DEFAULT_CUSTOM_TO,
        )
    return _chi_dt(end_date, 0, 0), _chi_dt(end_date + timedelta(days=1), 0, 0)


def session_tf_window(
    ref: datetime | pd.Timestamp,
    session_tf: str,
    *,
    from_hm: tuple[int, int] | None = None,
    to_hm: tuple[int, int] | None = None,
) -> tuple[datetime, datetime]:
    """Session window that contains *ref*, or the most recent completed one."""
    session_tf = coerce_session_tf(session_tf)
    chi = _as_utc(ref).to_pydatetime().astimezone(CT)
    hm = (chi.hour, chi.minute, chi.second, chi.microsecond)
    kw = {"from_hm": from_hm, "to_hm": to_hm}
    if session_tf == "full_session":
        if hm >= (17, 0, 0, 0):
            end_d = chi.date() + timedelta(days=1)
        else:
            end_d = chi.date()
        return session_tf_window_for_end_date(end_d, session_tf, **kw)
    if session_tf == "bod":
        return session_tf_window_for_end_date(chi.date(), session_tf, **kw)
    if session_tf == "cash":
        end_d = chi.date() - timedelta(days=1) if (chi.hour, chi.minute) < (8, 30) else chi.date()
        return session_tf_window_for_end_date(end_d, session_tf, **kw)
    if session_tf == "custom":
        fh, fm = from_hm or DEFAULT_CUSTOM_FROM
        th, tm = to_hm or DEFAULT_CUSTOM_TO
        chi_min = chi.hour * 60 + chi.minute
        from_min = fh * 60 + fm
        to_min = th * 60 + tm
        if from_min < to_min:
            end_d = chi.date() - timedelta(days=1) if chi_min < from_min else chi.date()
        else:
            end_d = chi.date() + timedelta(days=1) if chi_min >= from_min else chi.date()
        return session_tf_window_for_end_date(end_d, session_tf, **kw)
    end_d = chi.date() - timedelta(days=1) if (chi.hour, chi.minute) < (8, 0) else chi.date()
    return session_tf_window_for_end_date(end_d, session_tf, **kw)


def iter_daily_session_windows(
    start: datetime, end: datetime, session_tf: str,
    *,
    from_hm: tuple[int, int] | None = None,
    to_hm: tuple[int, int] | None = None,
) -> list[tuple[datetime, datetime]]:
    session_tf = coerce_session_tf(session_tf)
    d0 = start.astimezone(CT).date() - timedelta(days=1)
    d1 = end.astimezone(CT).date() + timedelta(days=1)
    out: list[tuple[datetime, datetime]] = []
    d = d0
    while d <= d1:
        s, e = session_tf_window_for_end_date(
            d, session_tf, from_hm=from_hm, to_hm=to_hm,
        )
        if e > start and s < end:
            out.append((s, e))
        d += timedelta(days=1)
    return out


def globex_week_bounds(ref: datetime | pd.Timestamp) -> tuple[datetime, datetime]:
    """CME week: Sunday 17:00 CT → Friday 16:00 CT."""
    chi = _as_utc(ref).to_pydatetime().astimezone(CT)
    wd = chi.weekday()
    if wd == 6:
        sun = chi.date() if (chi.hour, chi.minute) >= (17, 0) else chi.date() - timedelta(days=7)
    elif wd == 5:
        sun = chi.date() - timedelta(days=6)
    else:
        sun = chi.date() - timedelta(days=wd + 1)
    fri = sun + timedelta(days=5)
    return _chi_dt(sun, 17, 0), _chi_dt(fri, 16, 0)


def weekly_session_parts(
    ref: datetime, session_tf: str,
    *,
    from_hm: tuple[int, int] | None = None,
    to_hm: tuple[int, int] | None = None,
) -> tuple[datetime, datetime, list[tuple[datetime, datetime]]]:
    session_tf = coerce_session_tf(session_tf)
    kw = {"from_hm": from_hm, "to_hm": to_hm}
    if session_tf == "full_session":
        s, e = globex_week_bounds(ref)
        return s, e, [(s, e)]
    if session_tf == "custom":
        fh, fm = from_hm or DEFAULT_CUSTOM_FROM
        th, tm = to_hm or DEFAULT_CUSTOM_TO
        if fh * 60 + fm >= th * 60 + tm:
            sun_s, fri_e = globex_week_bounds(ref)
            sun = sun_s.astimezone(CT).date()
            fri = fri_e.astimezone(CT).date()
            s = _chi_dt(sun, fh, fm)
            e = _chi_dt(fri, th, tm)
            return s, e, [(s, e)]
    chi = _as_utc(ref).to_pydatetime().astimezone(CT)
    mon = chi.date() - timedelta(days=chi.weekday())
    if session_tf == "bod":
        s = _chi_dt(mon, 0, 0)
        e = _chi_dt(mon + timedelta(days=7), 0, 0)
        return s, e, [(s, e)]
    parts = [
        session_tf_window_for_end_date(mon + timedelta(days=i), session_tf, **kw)
        for i in range(5)
    ]
    return parts[0][0], parts[-1][1], parts


def iter_weekly_session_windows(
    start: datetime, end: datetime, session_tf: str,
    *,
    from_hm: tuple[int, int] | None = None,
    to_hm: tuple[int, int] | None = None,
) -> list[tuple[datetime, datetime, list[tuple[datetime, datetime]]]]:
    session_tf = coerce_session_tf(session_tf)
    out: list[tuple[datetime, datetime, list[tuple[datetime, datetime]]]] = []
    seen: set[tuple[datetime, datetime]] = set()
    t = start
    guard = 0
    while t < end and guard < 64:
        guard += 1
        s, e, parts = weekly_session_parts(
            t, session_tf, from_hm=from_hm, to_hm=to_hm,
        )
        key = (s, e)
        if key in seen:
            t = e + timedelta(days=3)
            continue
        seen.add(key)
        if e > start and s < end:
            out.append((s, e, parts))
        t = e + timedelta(seconds=1)
        if t <= s:
            t = s + timedelta(days=7)
    return out


def parse_ema_lookback(value: Any, *, field: str = "lookback") -> int:
    """Bars in the EMA window, inclusive of the bar being evaluated."""
    if value is None or value == "":
        return int(EMA_LOOKBACK_DEFAULT)
    try:
        n = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if n < EMA_LOOKBACK_MIN or n > EMA_LOOKBACK_MAX:
        raise ValueError(
            f"{field} must be between {EMA_LOOKBACK_MIN} and {EMA_LOOKBACK_MAX}"
        )
    return n


def parse_emas(raw: str | None) -> list[dict[str, Any]]:
    """Parse the Live Chart ``emas`` JSON (multi EMA overlay specs).

    Each spec is ``{id, n, lookback}``. ``lookback`` is the number of bars in
    the EMA window **including** the bar whose EMA value is reported
    (standard span-N EMA via ``ewm(span=N)``).
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("emas must be JSON") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("emas must be a JSON array")
    if len(data) > MAX_EMA_SPECS:
        raise ValueError(f"at most {MAX_EMA_SPECS} EMAs")
    out: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_ns: set[int] = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"emas[{i}] must be an object")
        lookback = parse_ema_lookback(
            item.get("lookback", item.get("period", item.get("length"))),
            field=f"emas[{i}].lookback",
        )
        n_raw = item.get("n") or item.get("ema_n")
        try:
            n = int(n_raw) if n_raw not in (None, "") else (i + 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"emas[{i}].n must be an integer") from exc
        if n < 1:
            raise ValueError(f"emas[{i}].n must be >= 1")
        if n in used_ns:
            raise ValueError(f"emas[{i}].n {n} is duplicated")
        used_ns.add(n)
        raw_id = item.get("id")
        eid = str(raw_id).strip() if raw_id not in (None, "") else f"ema{n}"
        if not eid:
            raise ValueError(f"emas[{i}].id must be non-empty")
        if eid in used_ids:
            raise ValueError(f"emas[{i}].id {eid!r} is duplicated")
        used_ids.add(eid)
        key = f"ema_{n}"
        out.append({
            "id": eid,
            "n": n,
            "lookback": lookback,
            "key": key,
            "label": f"EMA{n} ({lookback})",
        })
    return out


def parse_stoch_period(
    value: Any,
    *,
    field: str,
    default: int,
    lo: int,
    hi: int,
) -> int:
    if value is None or value == "":
        return int(default)
    try:
        n = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if n < lo or n > hi:
        raise ValueError(f"{field} must be between {lo} and {hi}")
    return n


def parse_stochs(raw: str | None) -> list[dict[str, Any]]:
    """Parse Live Chart ``stochs`` JSON (multi StochasticFast %D specs).

    Each spec is ``{id, n, k_period, d_period}``. ``k_period`` is the Lane %K
    lookback; ``d_period`` is the Wilder smooth on %K (TOS StochasticFast).
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("stochs must be JSON") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("stochs must be a JSON array")
    if len(data) > MAX_STOCH_SPECS:
        raise ValueError(f"at most {MAX_STOCH_SPECS} stochastics")
    out: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_ns: set[int] = set()
    used_kd: set[tuple[int, int]] = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"stochs[{i}] must be an object")
        k_period = parse_stoch_period(
            item.get("k_period", item.get("k", item.get("period"))),
            field=f"stochs[{i}].k_period",
            default=STOCH_K_DEFAULT,
            lo=STOCH_K_MIN,
            hi=STOCH_K_MAX,
        )
        d_period = parse_stoch_period(
            item.get("d_period", item.get("d", item.get("smooth"))),
            field=f"stochs[{i}].d_period",
            default=STOCH_D_DEFAULT,
            lo=STOCH_D_MIN,
            hi=STOCH_D_MAX,
        )
        kd = (k_period, d_period)
        if kd in used_kd:
            raise ValueError(f"stochs[{i}] Fast({k_period},{d_period}) is duplicated")
        used_kd.add(kd)
        n_raw = item.get("n") or item.get("stoch_n")
        try:
            n = int(n_raw) if n_raw not in (None, "") else (i + 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"stochs[{i}].n must be an integer") from exc
        if n < 1:
            raise ValueError(f"stochs[{i}].n must be >= 1")
        if n in used_ns:
            raise ValueError(f"stochs[{i}].n {n} is duplicated")
        used_ns.add(n)
        raw_id = item.get("id")
        eid = str(raw_id).strip() if raw_id not in (None, "") else f"stoch{n}"
        if not eid:
            raise ValueError(f"stochs[{i}].id must be non-empty")
        if eid in used_ids:
            raise ValueError(f"stochs[{i}].id {eid!r} is duplicated")
        used_ids.add(eid)
        key = f"stoch_{n}"
        out.append({
            "id": eid,
            "n": n,
            "k_period": k_period,
            "d_period": d_period,
            "key": key,
            "label": f"Stoch{n} Fast %D ({k_period},{d_period})",
        })
    return out


def _parse_sar_af(
    value: Any,
    *,
    field: str,
    default: float,
    lo: float,
    hi: float,
) -> float:
    if value in (None, ""):
        return float(default)
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(n):
        raise ValueError(f"{field} must be finite")
    if n < lo or n > hi:
        raise ValueError(f"{field} must be between {lo} and {hi}")
    return float(n)


def parse_sar_params(raw: str | dict[str, Any] | None) -> dict[str, float]:
    """Parse Live Chart ``sar`` JSON: ``{start, increment, maximum}``.

    Defaults and bounds match the Parabolic SAR ext-model knobs
    (``PSAR_PARAM_SPEC`` / ``sar_on_closed``).
    """
    data: dict[str, Any]
    if raw is None or raw == "":
        data = {}
    elif isinstance(raw, dict):
        data = raw
    else:
        text = str(raw).strip()
        if not text:
            data = {}
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("sar must be JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("sar must be a JSON object")
            data = parsed
    start = _parse_sar_af(
        data.get("start", data.get("af_start")),
        field="sar.start",
        default=SAR_START_DEFAULT,
        lo=SAR_AF_MIN,
        hi=SAR_AF_MAX,
    )
    increment = _parse_sar_af(
        data.get("increment", data.get("af_increment")),
        field="sar.increment",
        default=SAR_INCREMENT_DEFAULT,
        lo=SAR_AF_MIN,
        hi=SAR_AF_MAX,
    )
    maximum = _parse_sar_af(
        data.get("maximum", data.get("af_maximum", data.get("max"))),
        field="sar.maximum",
        default=SAR_MAXIMUM_DEFAULT,
        lo=SAR_MAXIMUM_MIN,
        hi=SAR_MAXIMUM_MAX,
    )
    return {"start": start, "increment": increment, "maximum": maximum}


def ema_on_close(close: pd.Series, lookback: int) -> pd.Series:
    """Standard EMA of *close* with span = *lookback* bars.

    The bar being evaluated is included in the lookback window
    (``ewm(span=N, adjust=False, min_periods=N)``). Values before the first
    full window are NaN. Callers that want a finite value at the first
    visible bar must prepend contiguous earlier closes (ADR-137).
    """
    n = parse_ema_lookback(lookback)
    s = pd.to_numeric(close, errors="coerce").astype("float64")
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


# Bars of history a lookback overlay needs before the first plotted bar.
# CLV is per-bar. CVD / OFI / VWAP / A-D keep their origin resets.
_LOOKBACK_OVERLAY_BARS: dict[str, int] = {
    "mf_mfi": 14,
    "so_fast_d_9_3": 12,
    "so_fast_d_14_3": 17,
    "so_fast_d_40_4": 44,
    "so_full_d_60_10": 70,
    "so_sync_score_all": 70,
    "so_sync_score_9_60": 70,
    "ps_sar": 20,
}
_TF_SECONDS: dict[str, int] = {
    "5s": 5, "15s": 15, "1min": 60, "5min": 300,
    "15min": 900, "1hour": 3600, "1day": 86400,
}


def overlay_warmup_bars(
    keys: list[str] | None,
    ema_specs: list[dict[str, Any]] | None,
    stoch_specs: list[dict[str, Any]] | None = None,
) -> int:
    """How many bars before the visible window lookback overlays need."""
    n = 0
    for key in keys or []:
        n = max(n, _LOOKBACK_OVERLAY_BARS.get(key, 0))
    for spec in ema_specs or []:
        try:
            n = max(n, int(spec.get("lookback") or 0))
        except (TypeError, ValueError):
            continue
    for spec in stoch_specs or []:
        try:
            k = int(spec.get("k_period") or 0)
            d = int(spec.get("d_period") or 0)
        except (TypeError, ValueError):
            continue
        n = max(n, k + d)
    return n


def warmup_start(start: datetime, timeframe: str, n_bars: int) -> datetime:
    """UTC instant ``n_bars`` (+1) before *start* on *timeframe*."""
    if n_bars <= 0:
        return start
    tf = coerce_tf(timeframe)
    secs = _TF_SECONDS.get(tf, 60) * (int(n_bars) + 1)
    return start - timedelta(seconds=secs)


def prepend_ohlc_warmup(display: pd.DataFrame, prior: pd.DataFrame | None) -> pd.DataFrame:
    """Stack *prior* bars strictly before the first displayed bucket."""
    if prior is None or prior.empty or "bucket" not in getattr(prior, "columns", []):
        return display
    if display is None or display.empty or "bucket" not in display.columns:
        return prior
    pre = prior.copy()
    disp = display.copy()
    pre["bucket"] = pd.to_datetime(pre["bucket"], utc=True)
    disp["bucket"] = pd.to_datetime(disp["bucket"], utc=True)
    cut = disp["bucket"].iloc[0]
    pre = pre.loc[pre["bucket"] < cut]
    if pre.empty:
        return disp
    return pd.concat([pre, disp], ignore_index=True)


def parse_profiles(raw: str | None) -> list[dict[str, Any]]:
    """Parse the Live Chart ``profiles`` JSON (volume-profile specs)."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("profiles must be JSON") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("profiles must be a JSON array")
    if len(data) > MAX_VP_SPECS:
        raise ValueError(f"at most {MAX_VP_SPECS} volume profiles")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"profiles[{i}] must be an object")
        kind = str(item.get("kind") or "current").strip().lower()
        legacy_custom = kind == "custom"
        if kind == "custom":
            kind = "current"
        if kind not in VP_KINDS:
            raise ValueError(f"profiles[{i}].kind must be one of {list(VP_KINDS)}")
        session_tf = coerce_session_tf(item.get("session_tf") or item.get("session"))
        placement = str(item.get("placement") or "overlay").strip().lower()
        if placement not in VP_PLACEMENTS:
            raise ValueError(f"profiles[{i}].placement must be one of {list(VP_PLACEMENTS)}")
        side = str(item.get("overlay_side") or item.get("side") or "left").strip().lower()
        if side not in VP_SIDES:
            raise ValueError(f"profiles[{i}].overlay_side must be left or right")
        extend = coerce_vp_extend(item)
        n = parse_vp_n(item.get("n") or item.get("vp_n"), fallback=i + 1)
        hist_style = str(item.get("hist_style") or item.get("histogram") or "volume").strip().lower()
        if hist_style == "lvn_nodes":
            hist_style = "binned"
        if hist_style not in VP_HIST_STYLES:
            raise ValueError(f"profiles[{i}].hist_style must be one of {list(VP_HIST_STYLES)}")
        group_ticks = item.get("group_ticks")
        if group_ticks is None or group_ticks == "":
            group_ticks = GROUP_TICKS_DEFAULT
        try:
            group_ticks = int(group_ticks)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profiles[{i}].group_ticks must be an integer") from exc
        if group_ticks not in GROUP_TICKS:
            raise ValueError(f"profiles[{i}].group_ticks must be one of {list(GROUP_TICKS)}")
        inward_left = placement == "right_frame" or (placement == "overlay" and side == "right")
        right_display = str(item.get("right_display") or ("none" if inward_left else "total_split")).strip().lower()
        left_display = str(item.get("left_display") or ("total_split" if inward_left else "none")).strip().lower()
        if right_display not in VP_DISPLAY_MODES:
            raise ValueError(f"profiles[{i}].right_display must be one of {list(VP_DISPLAY_MODES)}")
        if left_display not in VP_DISPLAY_MODES:
            raise ValueError(f"profiles[{i}].left_display must be one of {list(VP_DISPLAY_MODES)}")
        def coerce_opacity(field: str) -> float:
            raw_opacity = item.get(field)
            if raw_opacity is None or raw_opacity == "":
                return 100.0
            try:
                opacity = float(raw_opacity)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"profiles[{i}].{field} must be a number") from exc
            if not math.isfinite(opacity) or opacity < 0 or opacity > 100:
                raise ValueError(f"profiles[{i}].{field} must be between 0 and 100")
            return opacity
        right_opacity = coerce_opacity("right_opacity")
        left_opacity = coerce_opacity("left_opacity")
        hist_width = item.get("hist_width")
        if hist_width is None or hist_width == "":
            hist_width = HIST_WIDTH_DEFAULT
        try:
            hist_width = float(hist_width)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profiles[{i}].hist_width must be a number") from exc
        if not math.isfinite(hist_width) or hist_width <= 0:
            raise ValueError(f"profiles[{i}].hist_width must be greater than zero")
        chart_labels = parse_bool(item.get("chart_labels"), default=False)
        axis_labels = parse_bool(item.get("axis_labels"), default=False)
        plot_variable_bounds = parse_bool(item.get("plot_variable_bounds"), default=False)
        lvn_pct = item.get("lvn_pct")
        if lvn_pct is None:
            lvn_pct = LVN_PCT_DEFAULT
        try:
            lvn_pct = float(lvn_pct)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profiles[{i}].lvn_pct must be a number") from exc
        if lvn_pct > 1.0:
            lvn_pct = lvn_pct / 100.0
        lvn_pct = min(LVN_PCT_MAX, max(LVN_PCT_MIN, lvn_pct))
        show_lvn = item.get("show_lvn")
        if show_lvn is None:
            show_lvn = False
        else:
            show_lvn = parse_bool(show_lvn, default=False)
        extend_lvn = item.get("extend_lvn")
        if extend_lvn is None:
            extend_lvn = bool(show_lvn)
        else:
            extend_lvn = parse_bool(extend_lvn, default=False)
        lvn_groups = item.get("lvn_min_groups")
        try:
            lvn_groups = LVN_GROUPS_DEFAULT if lvn_groups in (None, "") else int(lvn_groups)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profiles[{i}].lvn_min_groups must be an integer") from exc
        if lvn_groups < LVN_GROUPS_MIN or lvn_groups > LVN_GROUPS_MAX:
            raise ValueError(
                f"profiles[{i}].lvn_min_groups must be between {LVN_GROUPS_MIN} and {LVN_GROUPS_MAX}"
            )
        hvn_method = str(item.get("hvn_method") or HVN_METHOD_DEFAULT).strip().lower()
        if hvn_method not in HVN_METHODS:
            raise ValueError(f"profiles[{i}].hvn_method must be one of {list(HVN_METHODS)}")
        def kde_float(field: str, default: float, lo: float, hi: float) -> float:
            raw = item.get(field)
            try:
                value = default if raw is None or raw == "" else float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"profiles[{i}].{field} must be a number") from exc
            if field in ("kde_prominence", "kde_valley_depth", "kde_volume") and value >= 1.0:
                value /= 100.0
            if not math.isfinite(value) or value < lo or value > hi:
                raise ValueError(f"profiles[{i}].{field} must be between {lo} and {hi}")
            return value
        def kde_int(field: str, default: int, lo: int, hi: int) -> int:
            raw = item.get(field)
            try:
                value = default if raw is None or raw == "" else int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"profiles[{i}].{field} must be an integer") from exc
            if value < lo or value > hi:
                raise ValueError(f"profiles[{i}].{field} must be between {lo} and {hi}")
            return value
        kde_bandwidth = kde_float("kde_bandwidth", KDE_BW_DEFAULT, KDE_BW_MIN, KDE_BW_MAX)
        kde_prominence = kde_float("kde_prominence", KDE_PROM_DEFAULT, KDE_PROM_MIN, KDE_PROM_MAX)
        kde_valley_depth = kde_float("kde_valley_depth", KDE_VALLEY_DEFAULT, KDE_VALLEY_MIN, KDE_VALLEY_MAX)
        kde_volume = kde_float("kde_volume", KDE_VOL_DEFAULT, KDE_VOL_MIN, KDE_VOL_MAX)
        kde_separation = kde_int("kde_separation", KDE_SEP_DEFAULT, KDE_SEP_MIN, KDE_SEP_MAX)
        kde_valley_width = kde_int("kde_valley_width", KDE_VALLEY_WIDTH_DEFAULT, KDE_VALLEY_WIDTH_MIN, KDE_VALLEY_WIDTH_MAX)
        kde_max_zones = kde_int("kde_max_zones", KDE_MAX_ZONES_DEFAULT, KDE_MAX_ZONES_MIN, KDE_MAX_ZONES_MAX)
        recur = str(item.get("recur") or item.get("type") or "daily").strip().lower()
        if kind == "recurring" and recur not in VP_RECUR:
            raise ValueError(f"profiles[{i}].recur must be one of {list(VP_RECUR)}")
        pid = str(item.get("id") or f"vp{i+1}").strip() or f"vp{i+1}"
        from_raw = item.get("range_start") or item.get("vp_start")
        to_raw = item.get("range_end") or item.get("vp_end")
        maybe_from = item.get("from")
        maybe_to = item.get("to")
        if from_raw is None and maybe_from is not None:
            text = str(maybe_from).strip()
            if "T" in text or len(text) > 8:
                from_raw = maybe_from
        if to_raw is None and maybe_to is not None:
            text = str(maybe_to).strip()
            if "T" in text or len(text) > 8:
                to_raw = maybe_to
        range_start = parse_vp_bound(from_raw, field="vp_start")
        range_end = parse_vp_bound(to_raw, field="vp_end", date_is_end=True)
        session_from = parse_session_hm(
            item.get("session_from") or item.get("from_time") or item.get("from_hm"),
            field="session_from",
            default=DEFAULT_CUSTOM_FROM,
        )
        session_to = parse_session_hm(
            item.get("session_to") or item.get("to_time") or item.get("to_hm"),
            field="session_to",
            default=DEFAULT_CUSTOM_TO,
        )
        session_date = parse_session_date(item.get("session_date") or item.get("date"))
        if legacy_custom:
            if session_tf == "full_session":
                session_tf = "custom"
            if range_start is not None:
                chi_s = range_start.astimezone(CT)
                session_from = (chi_s.hour, chi_s.minute)
            if range_end is not None:
                chi_e = range_end.astimezone(CT)
                session_to = (chi_e.hour, chi_e.minute)
                if session_date is None:
                    session_date = chi_e.date()
            elif range_start is not None and session_date is None:
                session_date = range_start.astimezone(CT).date()
        lookback_sessions = item.get("lookback_sessions")
        if lookback_sessions is None or lookback_sessions == "":
            lookback_sessions = LEVEL_BLEND_LOOKBACK_DEFAULT
        try:
            lookback_sessions = int(lookback_sessions)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profiles[{i}].lookback_sessions must be an integer") from exc
        if (lookback_sessions < LEVEL_BLEND_LOOKBACK_MIN
                or lookback_sessions > LEVEL_BLEND_LOOKBACK_MAX):
            raise ValueError(
                f"profiles[{i}].lookback_sessions must be between "
                f"{LEVEL_BLEND_LOOKBACK_MIN} and {LEVEL_BLEND_LOOKBACK_MAX}"
            )
        if kind == "level_blend":
            # Model levels are cash-session only (ADR-126 / ADR-129).
            session_tf = "cash"
            hist_style = "volume"
            show_lvn = False
            extend_lvn = False
            hvn_method = "off"
            right_display = "none"
            left_display = "none"
            plot_variable_bounds = False
        elif kind != "current":
            session_date = None
        out.append({
            "id": pid,
            "n": n,
            "kind": kind,
            "session_tf": session_tf,
            "placement": placement,
            "overlay_side": side,
            "extend": extend,
            "extend_lines": extend >= 1,
            "extend_va": extend >= 1,
            "hist_style": hist_style,
            "group_ticks": group_ticks,
            "right_display": right_display,
            "left_display": left_display,
            "right_opacity": right_opacity,
            "left_opacity": left_opacity,
            "hist_width": hist_width,
            "chart_labels": chart_labels,
            "axis_labels": axis_labels,
            "plot_variable_bounds": plot_variable_bounds,
            "show_lvn": bool(show_lvn),
            "lvn_pct": lvn_pct,
            "extend_lvn": bool(extend_lvn),
            "lvn_min_groups": lvn_groups,
            "hvn_method": hvn_method,
            "kde_bandwidth": kde_bandwidth,
            "kde_prominence": kde_prominence,
            "kde_valley_depth": kde_valley_depth,
            "kde_volume": kde_volume,
            "kde_separation": kde_separation,
            "kde_valley_width": kde_valley_width,
            "kde_max_zones": kde_max_zones,
            "lookback_sessions": lookback_sessions,
            "recur": recur if kind == "recurring" else None,
            "session_from": session_from,
            "session_to": session_to,
            "session_date": session_date,
            "range_start": None,
            "range_end": None,
        })
    return out


def parse_vp_bound(
    raw: str | None,
    *,
    field: str = "vp_start",
    date_is_end: bool = False,
) -> datetime | None:
    """Parse a VP bound. Naive ISO / YYYY-MM-DD is America/Chicago wall time.

    Date-only *end* bounds are exclusive midnight of the next CT day, so
    ``To = 2026-08-28`` includes that Friday's session through 24:00 CT.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    date_only = len(text) <= 10
    try:
        if date_only:
            wall = datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=CT)
            if date_is_end:
                wall = wall + timedelta(days=1)
            ts = wall.astimezone(timezone.utc)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                ts = parsed.replace(tzinfo=CT).astimezone(timezone.utc)
            else:
                ts = parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO datetime or YYYY-MM-DD") from exc
    return ts


def parse_vp_start(raw: str | None, *, end: datetime | None = None) -> datetime | None:
    """Custom VP window start. Naive ISO is America/Chicago wall time."""
    ts = parse_vp_bound(raw, field="vp_start")
    if ts is not None and end is not None and ts >= end:
        raise ValueError("vp_start must be before the chart end")
    return ts


def value_area(
    prices: np.ndarray,
    volumes: np.ndarray,
    va_pct: float = VA_PCT,
) -> tuple[float, float, float]:
    """POC, VA high, VA low — contiguous expansion from POC until *va_pct*."""
    if prices is None or volumes is None or len(prices) == 0:
        return (float("nan"), float("nan"), float("nan"))
    return tick_value_area(prices, volumes, va_pct)


def volume_nodes(
    prices: np.ndarray,
    volumes: np.ndarray,
    buy: np.ndarray | None = None,
    sell: np.ndarray | None = None,
    *,
    lvn_pct: float = LVN_PCT_DEFAULT,
    tick_size: float = TICK_SIZE,
    lvn_min_groups: int = LVN_GROUPS_DEFAULT,
) -> list[dict[str, Any]]:
    """Merge contiguous ticks into body / LVN rectangles.

    CME / Steidlmayer specify POC and the 70% value area, not LVN. VP
    platforms treat an LVN as a **valley**: a contiguous run of prices
    whose volume is ≤ ``lvn_pct`` of POC volume (default 30%) and that
    does not contain the POC. Adjacent above-threshold ticks merge into
    body nodes (volume shelves / HVN mass). ``lo``/``hi`` are inclusive
    tick prices; expand by ``tick_size/2`` when drawing.
    """
    if prices is None or volumes is None or len(prices) == 0:
        return []
    px_all = np.asarray(prices, dtype="float64")
    vol_all = np.asarray(volumes, dtype="float64")
    n0 = int(px_all.size)
    buy_all = np.zeros(n0, dtype="float64") if buy is None else np.asarray(buy, dtype="float64")
    sell_all = np.zeros(n0, dtype="float64") if sell is None else np.asarray(sell, dtype="float64")
    if buy_all.size != n0:
        buy_all = np.zeros(n0, dtype="float64")
    if sell_all.size != n0:
        sell_all = np.zeros(n0, dtype="float64")
    ok = np.isfinite(px_all) & np.isfinite(vol_all) & (vol_all >= 0)
    if not bool(ok.any()):
        return []
    px = px_all[ok]
    vol = vol_all[ok]
    b = np.where(np.isfinite(buy_all[ok]), buy_all[ok], 0.0)
    s = np.where(np.isfinite(sell_all[ok]), sell_all[ok], 0.0)
    order = np.argsort(px, kind="mergesort")
    px = px[order]
    vol = vol[order]
    b = b[order]
    s = s[order]
    i_poc = int(np.argmax(vol))
    poc_vol = float(vol[i_poc])
    if poc_vol <= 0:
        return []
    pct = float(lvn_pct)
    if pct > 1.0:
        pct = pct / 100.0
    pct = min(LVN_PCT_MAX, max(LVN_PCT_MIN, pct))
    thr = pct * poc_vol
    is_lvn = vol <= thr
    is_lvn[i_poc] = False
    n = int(vol.size)
    min_bars = max(LVN_GROUPS_MIN, min(LVN_GROUPS_MAX, int(lvn_min_groups)))
    if min_bars > 1:
        qualified = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            j = i + 1
            while j < n and bool(is_lvn[j]) == bool(is_lvn[i]):
                j += 1
            if bool(is_lvn[i]) and j - i >= min_bars:
                qualified[i:j] = True
            i = j
        is_lvn = qualified
    nodes: list[dict[str, Any]] = []
    i = 0
    ts = float(tick_size or TICK_SIZE)
    while i < n:
        flag = bool(is_lvn[i])
        j = i + 1
        while j < n and bool(is_lvn[j]) == flag:
            j += 1
        if flag and i == 0 and j == n:
            flag = False
        nodes.append({
            "lo": float(px[i]),
            "hi": float(px[j - 1]),
            "v": float(vol[i:j].sum()),
            "b": float(b[i:j].sum()),
            "s": float(s[i:j].sum()),
            "lvn": flag,
            "tick_size": ts,
        })
        i = j
    return nodes


def grouped_tick_bins(
    prices: np.ndarray,
    volumes: np.ndarray,
    buy: np.ndarray | None = None,
    sell: np.ndarray | None = None,
    *,
    tick_size: float = TICK_SIZE,
    group_ticks: int = GROUP_TICKS_DEFAULT,
) -> list[dict[str, Any]]:
    """Aggregate tick volume into fixed, tick-aligned price groups."""
    if prices is None or volumes is None or len(prices) == 0:
        return []
    if group_ticks not in GROUP_TICKS:
        raise ValueError(f"group_ticks must be one of {list(GROUP_TICKS)}")
    px_all = np.asarray(prices, dtype="float64")
    vol_all = np.asarray(volumes, dtype="float64")
    n0 = int(px_all.size)
    buy_all = np.zeros(n0, dtype="float64") if buy is None else np.asarray(buy, dtype="float64")
    sell_all = np.zeros(n0, dtype="float64") if sell is None else np.asarray(sell, dtype="float64")
    if buy_all.size != n0:
        buy_all = np.zeros(n0, dtype="float64")
    if sell_all.size != n0:
        sell_all = np.zeros(n0, dtype="float64")
    ok = np.isfinite(px_all) & np.isfinite(vol_all) & (vol_all >= 0)
    if not bool(ok.any()):
        return []
    px = px_all[ok]
    vol = vol_all[ok]
    b = np.where(np.isfinite(buy_all[ok]), buy_all[ok], 0.0)
    s = np.where(np.isfinite(sell_all[ok]), sell_all[ok], 0.0)
    ts = float(tick_size or TICK_SIZE)
    ticks = np.rint(px / ts).astype("int64")
    groups = np.floor_divide(ticks, group_ticks)
    out: list[dict[str, Any]] = []
    for i in range(int(groups.min()), int(groups.max()) + 1):
        blo = i * group_ticks * ts
        bhi = blo + group_ticks * ts
        mask = groups == i
        out.append({
            "lo": float(blo),
            "hi": float(bhi),
            "p": float((blo + bhi) / 2.0),
            "v": float(vol[mask].sum()) if bool(mask.any()) else 0.0,
            "b": float(b[mask].sum()) if bool(mask.any()) else 0.0,
            "s": float(s[mask].sum()) if bool(mask.any()) else 0.0,
        })
    return out


def lvn_nodes_from_bins(
    bins: list[dict[str, Any]],
    *,
    lvn_pct: float = LVN_PCT_DEFAULT,
    lvn_min_groups: int = LVN_GROUPS_DEFAULT,
) -> list[dict[str, Any]]:
    """Merge adjacent equal-width bins whose volume is ≤ ``lvn_pct`` of the max bin."""
    if not bins:
        return []
    vol = np.array([float(b.get("v") or 0.0) for b in bins], dtype="float64")
    i_poc = int(np.argmax(vol))
    poc_vol = float(vol[i_poc])
    if poc_vol <= 0:
        return []
    pct = float(lvn_pct)
    if pct > 1.0:
        pct = pct / 100.0
    pct = min(LVN_PCT_MAX, max(LVN_PCT_MIN, pct))
    thr = pct * poc_vol
    is_lvn = vol <= thr
    is_lvn[i_poc] = False
    n = len(bins)
    min_bars = max(LVN_GROUPS_MIN, min(LVN_GROUPS_MAX, int(lvn_min_groups)))
    if min_bars > 1:
        qualified = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            j = i + 1
            while j < n and bool(is_lvn[j]) == bool(is_lvn[i]):
                j += 1
            if bool(is_lvn[i]) and j - i >= min_bars:
                qualified[i:j] = True
            i = j
        is_lvn = qualified
    nodes: list[dict[str, Any]] = []
    i = 0
    while i < n:
        flag = bool(is_lvn[i])
        j = i + 1
        while j < n and bool(is_lvn[j]) == flag:
            j += 1
        if flag and i == 0 and j == n:
            flag = False
        chunk = bins[i:j]
        nodes.append({
            "lo": float(chunk[0]["lo"]),
            "hi": float(chunk[-1]["hi"]),
            "v": float(sum(float(c.get("v") or 0.0) for c in chunk)),
            "b": float(sum(float(c.get("b") or 0.0) for c in chunk)),
            "s": float(sum(float(c.get("s") or 0.0) for c in chunk)),
            "lvn": flag,
            "edge": True,
        })
        i = j
    return nodes


def adaptive_kde_hvn_zones(
    bins: list[dict[str, Any]],
    *,
    tick_size: float = TICK_SIZE,
    bandwidth: float = KDE_BW_DEFAULT,
    prominence: float = KDE_PROM_DEFAULT,
    valley_depth: float = KDE_VALLEY_DEFAULT,
    volume_fraction: float = KDE_VOL_DEFAULT,
    min_separation: int = KDE_SEP_DEFAULT,
    valley_width: int = KDE_VALLEY_WIDTH_DEFAULT,
    max_zones: int = KDE_MAX_ZONES_DEFAULT,
) -> list[dict[str, Any]]:
    """Find separated high-volume zones from a smoothed weighted profile.

    KDE identifies modes and significant valleys; the source histogram remains
    authoritative for prices and integrated zone volume.
    """
    if not bins:
        return []
    rows = sorted(
        (b for b in bins if math.isfinite(float(b.get("p") or 0))
         and math.isfinite(float(b.get("v") or 0)) and float(b.get("v") or 0) >= 0),
        key=lambda b: float(b["p"]),
    )
    if len(rows) < 3:
        return []
    prices = np.array([float(b["p"]) for b in rows], dtype="float64")
    volumes = np.array([float(b.get("v") or 0) for b in rows], dtype="float64")
    total = float(volumes.sum())
    peak_volume = float(volumes.max())
    if total <= 0 or peak_volume <= 0:
        return []
    sigma = max(KDE_BW_MIN, min(KDE_BW_MAX, float(bandwidth)))
    radius = max(1, int(math.ceil(sigma * 3.0)))
    grid = np.arange(-radius, radius + 1, dtype="float64")
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(volumes, kernel, mode="same")
    candidates = [i for i in range(1, len(density) - 1)
                  if density[i] >= density[i - 1] and density[i] >= density[i + 1]]
    candidates.sort(key=lambda i: density[i], reverse=True)
    selected: list[int] = []
    for i in candidates:
        if any(abs(i - j) < int(min_separation) for j in selected):
            continue
        selected.append(i)
    selected.sort()
    if not selected:
        return []
    max_density = float(max(density.max(), 1e-12))
    valley_candidates: list[dict[str, Any]] = []
    for left, right in zip(selected, selected[1:]):
        if right - left < max(2, int(valley_width)):
            continue
        valley = left + int(np.argmin(density[left:right + 1]))
        depth = (min(float(density[left]), float(density[right])) - float(density[valley])) / max_density
        if depth < float(valley_depth):
            continue
        valley_candidates.append({
            "index": valley,
            "price": float(prices[valley]),
            "depth": depth,
            "width": int(right - left),
        })
    boundaries = [0] + [int(v["index"]) for v in valley_candidates] + [len(rows) - 1]
    zones: list[dict[str, Any]] = []
    for pos in range(len(boundaries) - 1):
        lo_i = boundaries[pos]
        hi_i = boundaries[pos + 1]
        peak = lo_i + int(np.argmax(density[lo_i:hi_i + 1]))
        if hi_i < lo_i or not (lo_i <= peak <= hi_i):
            continue
        left_base = float(density[lo_i:peak + 1].min())
        right_base = float(density[peak:hi_i + 1].min())
        prom = float(density[peak] - max(left_base, right_base)) / max_density
        zone_volume = float(volumes[lo_i:hi_i + 1].sum())
        if prom < float(prominence) or zone_volume / total < float(volume_fraction):
            continue
        zones.append({
            "lo": float(prices[lo_i] - tick_size / 2),
            "hi": float(prices[hi_i] + tick_size / 2),
            "peak": float(prices[peak]),
            "peak_volume": float(volumes[peak]),
            "volume": zone_volume,
            "volume_fraction": zone_volume / total,
            "prominence": prom,
            "tick_size": float(tick_size),
            "left_valley": float(prices[lo_i]) if pos == 0 else float(prices[boundaries[pos]]),
            "right_valley": float(prices[hi_i]) if pos >= len(valley_candidates) else float(prices[boundaries[pos + 1]]),
        })
    zones.sort(key=lambda z: z["peak_volume"], reverse=True)
    return zones[:max(1, min(KDE_MAX_ZONES_MAX, int(max_zones)))]


def _as_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _query(sql: str, params: tuple, *, timeout_ms: int = STMT_MS) -> pd.DataFrame:
    with psycopg.connect(
        get_dsn(),
        autocommit=True,
        connect_timeout=5,
        options=f"-c statement_timeout={int(timeout_ms)}",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def globex_session_start(now: datetime | None = None) -> datetime:
    """Open of the current Globex session: previous 17:00 America/Chicago."""
    t = now or datetime.now(timezone.utc)
    return origin_start(t, "session")


def _canon_num(n: float) -> str:
    if n == int(n):
        return str(int(n))
    return f"{n:g}"


def _td_token(td: timedelta) -> str:
    secs = int(td.total_seconds())
    if secs % 86400 == 0:
        days = secs // 86400
        if days % 365 == 0 and days >= 365:
            return f"{days // 365}y"
        return f"{days}d"
    if secs % 3600 == 0:
        return f"{secs // 3600}h"
    if secs % 60 == 0:
        return f"{secs // 60}m"
    return f"{secs}s"


def _ct_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=CT)


def range_window(start_d: date, end_d: date) -> tuple[datetime, datetime]:
    """Inclusive CT calendar dates → UTC ``[start, end)`` window."""
    start = _ct_midnight(start_d).astimezone(timezone.utc)
    end = (_ct_midnight(end_d) + timedelta(days=1)).astimezone(timezone.utc)
    return start, end


def parse_lookback_range(lookback: str | None) -> tuple[date, date] | None:
    """Inclusive CT From/To dates, or ``None`` when *lookback* is not a range token."""
    raw = re.sub(r"\s+", " ", str(lookback or "").strip().lower())
    m = _RANGE_RE.fullmatch(raw)
    if m is None:
        return None
    try:
        start_d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        end_d = datetime.strptime(m.group(2), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("lookback range dates must be YYYY-MM-DD") from exc
    if end_d < start_d:
        raise ValueError("lookback range To must be on or after From")
    start, end = range_window(start_d, end_d)
    if end - start > MAX_LOOKBACK_DELTA:
        raise ValueError("lookback duration is at most 5 years")
    return start_d, end_d


def parse_lookback(lookback: str | None) -> tuple[str, timedelta | None]:
    """Canonical lookback token and duration (``None`` = current Globex session).

    Duration tokens (``7d``) are a lower bound relative to the as-of instant.
    Range tokens (``YYYY-MM-DD..YYYY-MM-DD``) pin both edges to CT calendar
    dates; the returned timedelta is the exclusive window length.
    """
    raw = re.sub(r"\s+", " ", str(lookback or "session").strip().lower())
    if not raw:
        raw = "session"
    if raw in ("session", "globex", "eth"):
        return "session", None
    if raw in _LOOKBACK:
        return raw, _LOOKBACK[raw]
    rng = parse_lookback_range(raw)
    if rng is not None:
        start_d, end_d = rng
        token = f"{start_d.isoformat()}..{end_d.isoformat()}"
        start, end = range_window(start_d, end_d)
        return token, end - start
    m = _DURATION_RE.fullmatch(raw)
    if m is None:
        raise ValueError(
            "lookback must be a duration like 6h, 3d, 2w, 90d, 1y, "
            "a date range YYYY-MM-DD..YYYY-MM-DD, "
            f"or one of {list(_LOOKBACK)}"
        )
    n = float(m.group(1))
    family = _UNIT_FAMILY[m.group(2).lower()]
    if not math.isfinite(n) or n <= 0:
        raise ValueError("lookback duration must be positive")
    if family == "minutes":
        delta = timedelta(minutes=n)
    elif family == "hours":
        delta = timedelta(hours=n)
    elif family == "days":
        delta = timedelta(days=n)
    elif family == "weeks":
        delta = timedelta(weeks=n)
    else:
        delta = timedelta(days=n * 365.0)
    if delta < timedelta(seconds=1):
        raise ValueError("lookback duration must be at least 1 second")
    if delta > MAX_LOOKBACK_DELTA:
        raise ValueError("lookback duration is at most 5 years")
    token = f"{_canon_num(n)}{_FAMILY_SUFFIX[family]}"
    if token in _LOOKBACK:
        return token, _LOOKBACK[token]
    return token, delta


def window_for(lookback: str, *, timeframe: str, end: datetime | None = None) -> tuple[datetime, datetime]:
    coerce_tf(timeframe)  # validate; no per-TF window shrink (ADR-128)
    as_of = (end or datetime.now(timezone.utc)).astimezone(timezone.utc)
    token, delta = parse_lookback(lookback)
    rng = parse_lookback_range(token)
    if rng is not None:
        start, end_dt = range_window(*rng)
        if as_of < end_dt:
            end_dt = as_of
    elif token == "session":
        start = globex_session_start(as_of)
        end_dt = as_of
    else:
        assert delta is not None
        end_dt = as_of
        start = end_dt - delta
    if end_dt <= start:
        raise ValueError("empty window")
    return start, end_dt


def parse_print_min(value: Any, default: int = PRINT_MIN_DEFAULT) -> int:
    """Minimum TBBO fill size (contracts) for Large prints."""
    if value is None or value == "":
        return int(default)
    try:
        n = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("min_size must be an integer") from exc
    if n < 1:
        raise ValueError("min_size must be >= 1")
    return min(int(PRINT_MIN_MAX), n)


def parse_overlay_keys(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for part in str(raw).split(","):
        k = part.strip()
        if not k:
            continue
        if k not in _OVERLAY_BY_KEY:
            raise ValueError(f"unknown overlay {k!r}")
        if k not in out:
            out.append(k)
        if len(out) > 8:
            raise ValueError("at most 8 overlays")
    return out


def _unix(ts) -> int:
    return int(_as_utc(ts).timestamp())


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _lead_filter(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    local = df["bucket"].dt.tz_convert(CT)
    chicago_day = np.array([d.date() for d in local], dtype=object)
    d0 = start.astimezone(CT).date() - timedelta(days=1)
    d1 = end.astimezone(CT).date()
    lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0, d1)}
    keep = np.array(
        [lead.get(d) == str(s) for d, s in zip(chicago_day, df["symbol"].astype(str))],
        dtype=bool,
    )
    df = df.loc[keep]
    df = df.drop_duplicates(subset=["bucket"], keep="last")
    return df.sort_values("bucket")


def _load_cagg(table: str, start: datetime, end: datetime) -> pd.DataFrame:
    d0 = start.astimezone(CT).date() - timedelta(days=1)
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0, d1)})
    if not symbols:
        return pd.DataFrame()
    df = _query(
        f"SELECT bucket, symbol, open, high, low, close, volume, "
        f"buy_volume, sell_volume, bid_px, ask_px, bid_sz, ask_sz "
        f"FROM {table} "
        f"WHERE bucket >= %s AND bucket < %s AND symbol = ANY(%s) "
        f"ORDER BY bucket ASC",
        (start, end, symbols),
    )
    if df.empty:
        return df
    for col in (
        "open", "high", "low", "close", "volume",
        "buy_volume", "sell_volume", "bid_px", "ask_px", "bid_sz", "ask_sz",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _lead_filter(df, start, end)


def _load_5s(start: datetime, end: datetime) -> pd.DataFrame:
    d0 = start.astimezone(CT).date() - timedelta(days=1)
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0, d1)})
    if not symbols:
        return pd.DataFrame()
    df = _query(
        "SELECT time_bucket('5 seconds', ts_event) AS bucket, symbol, "
        "first(price, ts_event) AS open, max(price) AS high, min(price) AS low, "
        "last(price, ts_event) AS close, sum(size) AS volume, "
        "sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume, "
        "sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume, "
        "last(bid_px, ts_event) AS bid_px, last(ask_px, ts_event) AS ask_px, "
        "last(bid_sz, ts_event) AS bid_sz, last(ask_sz, ts_event) AS ask_sz "
        "FROM databento.tbbo "
        "WHERE ts_event >= %s AND ts_event < %s AND symbol = ANY(%s) "
        "GROUP BY 1, 2 ORDER BY 1",
        (start, end, symbols),
        timeout_ms=90_000,
    )
    if df.empty:
        return df
    for col in (
        "open", "high", "low", "close", "volume",
        "buy_volume", "sell_volume", "bid_px", "ask_px", "bid_sz", "ask_sz",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return _lead_filter(df, start, end)


def load_ohlc(timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    tf = coerce_tf(timeframe)
    empty = pd.DataFrame(columns=[
        "bucket", "open", "high", "low", "close", "volume",
        "buy_volume", "sell_volume", "bid_px", "ask_px", "bid_sz", "ask_sz",
        "symbol",
    ])
    if tf == "5s":
        df = _load_5s(start, end)
    elif tf == "15min":
        df = _load_cagg(_CAGG["5min"], start, end)
        if df.empty:
            return empty
        src = df.set_index("bucket")
        from src.external_models.timeframe import resample_ohlc
        ohlc_cols = [c for c in ("open", "high", "low", "close", "volume") if c in src.columns]
        out = resample_ohlc(src[ohlc_cols], "15min")
        if out is None or out.empty:
            return empty
        if "buy_volume" in src.columns:
            flow = src[["buy_volume", "sell_volume"]].copy()
            flow.index = pd.DatetimeIndex(pd.to_datetime(flow.index, utc=True))
            flow = flow.resample("15min", label="left", closed="left").sum()
            out = out.join(flow, how="left")
        bbo_cols = [c for c in ("bid_px", "ask_px", "bid_sz", "ask_sz") if c in src.columns]
        if bbo_cols:
            bbo = src[bbo_cols].copy()
            bbo.index = pd.DatetimeIndex(pd.to_datetime(bbo.index, utc=True))
            bbo = bbo.resample("15min", label="left", closed="left").last()
            out = out.join(bbo, how="left")
        out = out.reset_index()
        if "bucket" not in out.columns:
            out = out.rename(columns={out.columns[0]: "bucket"})
        return out
    else:
        df = _load_cagg(_CAGG[tf], start, end)
    if df.empty:
        return empty
    return df.reset_index(drop=True)


def load_large_prints(
    start: datetime,
    end: datetime,
    min_size: int,
    *,
    rth_only: bool = False,
) -> pd.DataFrame:
    """TBBO fills with ``size >= min_size`` in ``[start, end)`` (lead-month).

    Size filter is in SQL so this is not a full-tape load. Window is the
    chart lookback (caller), not the 36h OFI/VWAP tick cap.
    """
    min_size = parse_print_min(min_size)
    if end <= start:
        return pd.DataFrame(columns=["ts_event", "symbol", "price", "size", "side"])
    d0 = start.astimezone(CT).date()
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)})
    if not symbols:
        return pd.DataFrame(columns=["ts_event", "symbol", "price", "size", "side"])
    rth = RTH_SQL if rth_only else ""
    df = _query(
        "SELECT ts_event, symbol, price, size, side "
        "FROM databento.tbbo "
        "WHERE ts_event >= %s AND ts_event < %s AND symbol = ANY(%s) "
        "AND size >= %s AND price IS NOT NULL"
        f"{rth} "
        "ORDER BY ts_event DESC "
        "LIMIT %s",
        (start, end, symbols, min_size, MAX_PRINTS + 1),
        timeout_ms=90_000,
    )
    if df.empty:
        return pd.DataFrame(columns=["ts_event", "symbol", "price", "size", "side"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    local = df["ts_event"].dt.tz_convert(CT)
    chicago_day = np.array([d.date() for d in local], dtype=object)
    lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)}
    keep = np.array(
        [lead.get(d) == str(s) for d, s in zip(chicago_day, df["symbol"].astype(str))],
        dtype=bool,
    )
    df = df.loc[keep].copy()
    for c in ("price", "size"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts_event", "price", "size"])
    df = df.sort_values("ts_event")
    return df.reset_index(drop=True)


def pack_prints(df: pd.DataFrame) -> tuple[list[dict[str, Any]], bool]:
    """Serialize large-print rows. Most-recent ``MAX_PRINTS`` if over cap."""
    capped = False
    if df is None or df.empty:
        return [], False
    if len(df) > MAX_PRINTS:
        df = df.iloc[-MAX_PRINTS:].copy()
        capped = True
    out: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        px = _finite(r.price)
        sz = _finite(r.size)
        if px is None or sz is None:
            continue
        ts = _as_utc(r.ts_event)
        side = str(getattr(r, "side", "") or "").strip().upper()[:1]
        if side not in ("B", "A", "N"):
            side = "N"
        iso = ts.isoformat(timespec="microseconds")
        if iso.endswith("+00:00"):
            iso = iso[:-6] + "Z"
        out.append({
            "t": float(ts.timestamp()),
            "ts": iso,
            "p": px,
            "sz": int(round(sz)),
            "side": side,
        })
    return out, capped


def prints(
    *,
    timeframe: str = "1min",
    lookback: str = "session",
    end: str | None = None,
    min_size: int | str | None = None,
    eth_aggregate: bool = True,
    since: str | None = None,
) -> dict[str, Any]:
    """Large TBBO fills for the same window as ``series()``."""
    tf = coerce_tf(timeframe)
    eth_aggregate = parse_bool(eth_aggregate, default=True)
    rth_only = (not eth_aggregate) and tf != "1day"
    min_size = parse_print_min(min_size)
    end_dt = datetime.now(timezone.utc) if not end else datetime.fromisoformat(
        end.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    start, end_dt = window_for(lookback, timeframe=tf, end=end_dt)
    since_dt = None
    if since and str(since).strip():
        since_dt = datetime.fromisoformat(str(since).replace("Z", "+00:00")).astimezone(timezone.utc)
        if since_dt > start:
            start = since_dt
    df = load_large_prints(start, end_dt, min_size, rth_only=rth_only)
    rows, capped = pack_prints(df)
    return {
        "instrument": "/NQ",
        "timeframe": tf,
        "lookback": parse_lookback(lookback)[0],
        "min_size": min_size,
        "eth_aggregate": bool(eth_aggregate),
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end_dt.isoformat().replace("+00:00", "Z"),
        "n": len(rows),
        "capped": capped,
        "prints": rows,
    }


def load_ticks(start: datetime, end: datetime) -> pd.DataFrame:
    span = (end - start).total_seconds()
    if span > MAX_TICK_SECONDS:
        start = end - timedelta(seconds=MAX_TICK_SECONDS)
    d0 = start.astimezone(CT).date()
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)})
    if not symbols:
        return pd.DataFrame()
    df = _query(
        "SELECT ts_event, symbol, price, size, side, bid_px, ask_px, bid_sz, ask_sz "
        "FROM databento.tbbo "
        "WHERE ts_event >= %s AND ts_event < %s AND symbol = ANY(%s) "
        "ORDER BY ts_event",
        (start, end, symbols),
        timeout_ms=90_000,
    )
    if df.empty:
        return df
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    local = df["ts_event"].dt.tz_convert(CT)
    chicago_day = np.array([d.date() for d in local], dtype=object)
    lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)}
    keep = np.array(
        [lead.get(d) == str(s) for d, s in zip(chicago_day, df["symbol"].astype(str))],
        dtype=bool,
    )
    df = df.loc[keep].copy()
    for c in ("price", "size", "bid_px", "ask_px", "bid_sz", "ask_sz"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_vwap_bars(start: datetime, end: datetime, freq: str) -> pd.DataFrame:
    """Aggregate lead-contract price×size, price²×size, and size for every chart bucket."""
    interval = {
        "5s": "5 seconds", "15s": "15 seconds", "1min": "1 minute",
        "5min": "5 minutes", "15min": "15 minutes", "1hour": "1 hour",
        "1day": "1 day",
    }.get(coerce_tf(freq))
    if interval is None:
        raise ValueError(f"unsupported VWAP frequency: {freq}")
    d0 = start.astimezone(CT).date()
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)})
    if not symbols:
        return pd.DataFrame(columns=["ts_event", "symbol", "pv", "p2v", "size", "close"])
    df = _query(
        "SELECT time_bucket(%s::interval, ts_event) AS ts_event, "
        "symbol, SUM(price * size) AS pv, SUM(price * price * size) AS p2v, "
        "SUM(size) AS size, "
        "last(price, ts_event) AS close "
        "FROM databento.tbbo "
        "WHERE ts_event >= %s AND ts_event < %s AND symbol = ANY(%s) "
        "GROUP BY 1, 2 ORDER BY 1",
        (interval, start, end, symbols),
        timeout_ms=90_000,
    )
    if df.empty:
        return df
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    local = df["ts_event"].dt.tz_convert(CT)
    chicago_day = np.array([d.date() for d in local], dtype=object)
    lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)}
    keep = np.array(
        [lead.get(d) == str(s) for d, s in zip(chicago_day, df["symbol"].astype(str))],
        dtype=bool,
    )
    df = df.loc[keep].copy()
    for c in ("pv", "p2v", "size", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts_event", "pv", "size"])


def _bar_delta(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        buy = pd.to_numeric(df["buy_volume"], errors="coerce").fillna(0.0)
        sell = pd.to_numeric(df["sell_volume"], errors="coerce").fillna(0.0)
        return (buy - sell).astype("float64")
    return pd.Series(0.0, index=df.index, dtype="float64")


def cvd_on_bars(
    bars: pd.DataFrame,
    origin: str,
    *,
    warmup: pd.DataFrame | None = None,
) -> pd.Series:
    """Grouped cumulative signed volume, optionally offset by *warmup* bars."""
    if bars is None or bars.empty:
        return pd.Series(dtype="float64")
    origin = coerce_origin(origin)
    idx = pd.DatetimeIndex(pd.to_datetime(bars["bucket"], utc=True))
    delta = _bar_delta(bars)
    delta.index = idx
    key = origin_period_key(idx, origin)
    cvd = delta.groupby(key, sort=False).cumsum()
    offsets: dict[pd.Timestamp, float] = {}
    if warmup is not None and not warmup.empty:
        w = warmup.copy()
        w["bucket"] = pd.to_datetime(w["bucket"], utc=True)
        w = w.loc[w["bucket"] < idx[0]]
        if not w.empty:
            w_idx = pd.DatetimeIndex(w["bucket"])
            w_delta = _bar_delta(w)
            w_delta.index = w_idx
            w_key = origin_period_key(w_idx, origin)
            w_cvd = w_delta.groupby(w_key, sort=False).cumsum()
            last = w_cvd.groupby(w_key, sort=False).last()
            offsets = {pd.Timestamp(k): float(v) for k, v in last.items()}
    if offsets:
        add = np.array([offsets.get(pd.Timestamp(k), 0.0) for k in key], dtype="float64")
        cvd = cvd + add
    return cvd


def _cvd_opening_values(cvd: pd.Series, origin: str) -> pd.Series:
    """CVD opening values, with zero at an origin-boundary candle."""
    close = pd.to_numeric(cvd, errors="coerce")
    if close.empty:
        return close
    idx = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    key = origin_period_key(idx, origin)
    reset = np.empty(len(close), dtype=bool)
    reset[0] = origin_start(idx[0], origin) == idx[0].to_pydatetime()
    reset[1:] = key[1:] != key[:-1]
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    open_.iloc[reset] = 0.0
    return open_


def cvd_ohlc_from_closes(cvd: pd.Series, origin: str | None = None) -> pd.DataFrame:
    """OHLC from a close-only CVD series (one print per chart bar)."""
    close = pd.to_numeric(cvd, errors="coerce")
    if close.empty:
        return pd.DataFrame(columns=["o", "h", "l", "c"])
    if origin is None:
        open_ = close.shift(1).where(close.shift(1).notna(), close)
    else:
        open_ = _cvd_opening_values(close, origin)
    hi = np.fmax(open_.to_numpy(dtype="float64"), close.to_numpy(dtype="float64"))
    lo = np.fmin(open_.to_numpy(dtype="float64"), close.to_numpy(dtype="float64"))
    return pd.DataFrame({"o": open_.to_numpy(), "h": hi, "l": lo, "c": close.to_numpy()}, index=close.index)


def cvd_ohlc_from_fine(
    fine_cvd: pd.Series,
    buckets: pd.DatetimeIndex,
    origin: str | None = None,
) -> pd.DataFrame:
    """OHLC at *buckets* from a finer CVD path (e.g. 1-minute inside 5-minute bars)."""
    empty = pd.DataFrame(index=buckets, columns=["o", "h", "l", "c"])
    if fine_cvd is None or fine_cvd.empty or buckets is None or len(buckets) == 0:
        return empty
    s = pd.to_numeric(fine_cvd, errors="coerce").dropna()
    if s.empty:
        return empty
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, utc=True))
    s = s.sort_index()
    bidx = pd.DatetimeIndex(pd.to_datetime(buckets, utc=True))
    pos = bidx.searchsorted(s.index, side="right") - 1
    valid = pos >= 0
    if not bool(valid.any()):
        return empty
    grouped = pd.DataFrame({"c": s.to_numpy()[valid]}, index=bidx[pos[valid]])
    if origin is None:
        agg = grouped.groupby(level=0)["c"].agg(o="first", h="max", l="min", c="last")
    else:
        open_ = _cvd_opening_values(grouped["c"], origin)
        grouped["o"] = open_
        grouped["h"] = np.fmax(grouped["o"].to_numpy(), grouped["c"].to_numpy())
        grouped["l"] = np.fmin(grouped["o"].to_numpy(), grouped["c"].to_numpy())
        agg = grouped.groupby(level=0).agg(
            o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
        )
    return agg.reindex(bidx)


def _pack_cvd_ohlc(df: pd.DataFrame, n: int) -> dict[str, list[float | None]]:
    if df is None or df.empty:
        return {k: [None] * n for k in ("o", "h", "l", "c")}
    out: dict[str, list[float | None]] = {}
    for col in ("o", "h", "l", "c"):
        if col in df.columns:
            out[col] = [_finite(x) for x in df[col].to_numpy()]
        else:
            out[col] = [None] * n
        if len(out[col]) < n:
            out[col] = out[col] + [None] * (n - len(out[col]))
        elif len(out[col]) > n:
            out[col] = out[col][:n]
    return out


def _bar_ofi(df: pd.DataFrame) -> pd.Series:
    """Cont-2014 OFI on consecutive bar-end BBO (one event per bar)."""
    need = ("bid_px", "ask_px", "bid_sz", "ask_sz")
    if df is None or df.empty or any(c not in df.columns for c in need):
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(df["bucket"], utc=True))
    bid = pd.to_numeric(df["bid_px"], errors="coerce")
    ask = pd.to_numeric(df["ask_px"], errors="coerce")
    bid_sz = pd.to_numeric(df["bid_sz"], errors="coerce")
    ask_sz = pd.to_numeric(df["ask_sz"], errors="coerce")
    prev_bid, prev_ask = bid.shift(1), ask.shift(1)
    prev_bid_sz, prev_ask_sz = bid_sz.shift(1), ask_sz.shift(1)
    e = (
        (bid >= prev_bid).astype("float64") * bid_sz.fillna(0.0)
        - (bid <= prev_bid).astype("float64") * prev_bid_sz.fillna(0.0)
        - (ask <= prev_ask).astype("float64") * ask_sz.fillna(0.0)
        + (ask >= prev_ask).astype("float64") * prev_ask_sz.fillna(0.0)
    )
    e = e.where(prev_bid.notna() & prev_ask.notna(), 0.0)
    out = pd.Series(e.to_numpy(dtype="float64"), index=idx)
    return out.fillna(0.0)


def origin_cumsum(
    values: pd.Series,
    origin: str,
    warmup: pd.Series | None = None,
) -> pd.Series:
    """Cumulative *values* that reset at *origin* (session / BOD / MKT)."""
    origin = coerce_origin(origin)
    if values is None or values.empty:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(values.index, utc=True))
    val = pd.to_numeric(values, errors="coerce").fillna(0.0)
    val.index = idx
    key = origin_period_key(idx, origin)
    acc = val.groupby(key, sort=False).cumsum()
    offsets: dict[pd.Timestamp, float] = {}
    if warmup is not None and not warmup.empty:
        w = pd.to_numeric(warmup, errors="coerce").fillna(0.0)
        w.index = pd.DatetimeIndex(pd.to_datetime(w.index, utc=True))
        w = w.loc[w.index < idx[0]]
        if not w.empty:
            w_key = origin_period_key(w.index, origin)
            last = w.groupby(w_key, sort=False).cumsum().groupby(w_key, sort=False).last()
            offsets = {pd.Timestamp(k): float(v) for k, v in last.items()}
    if offsets:
        add = np.array([offsets.get(pd.Timestamp(k), 0.0) for k in key], dtype="float64")
        acc = acc + add
    return acc


def load_vbp(
    start: datetime, end: datetime, *, tick_size: float = TICK_SIZE, rth_only: bool = False,
) -> pd.DataFrame:
    span = (end - start).total_seconds()
    if span > MAX_VP_SECONDS:
        start = end - timedelta(seconds=MAX_VP_SECONDS)
    d0 = start.astimezone(CT).date()
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)})
    if not symbols:
        return pd.DataFrame(columns=["px", "vol", "buy", "sell"])
    df = _query(
        "SELECT (ts_event AT TIME ZONE 'America/Chicago')::date AS chi_day, "
        "symbol, "
        "(ROUND((price / %s)::numeric, 0) * %s)::double precision AS px, "
        "SUM(size)::double precision AS vol, "
        "SUM(CASE WHEN side = 'B' THEN size ELSE 0 END)::double precision AS buy, "
        "SUM(CASE WHEN side = 'A' THEN size ELSE 0 END)::double precision AS sell "
        "FROM databento.tbbo "
        "WHERE ts_event >= %s AND ts_event < %s AND symbol = ANY(%s) "
        "AND price IS NOT NULL AND size IS NOT NULL"
        f"{RTH_SQL if rth_only else ''} "
        "GROUP BY 1, 2, 3 ORDER BY 3",
        (tick_size, tick_size, start, end, symbols),
        timeout_ms=90_000,
    )
    if df.empty:
        return pd.DataFrame(columns=["px", "vol", "buy", "sell"])
    lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)}
    day = pd.to_datetime(df["chi_day"]).dt.date
    keep = np.array(
        [lead.get(d) == str(s) for d, s in zip(day, df["symbol"].astype(str))],
        dtype=bool,
    )
    df = df.loc[keep]
    if df.empty:
        return pd.DataFrame(columns=["px", "vol", "buy", "sell"])
    for col in ("px", "vol", "buy", "sell"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["px"])
    df = df.groupby("px", as_index=False)[["vol", "buy", "sell"]].sum().sort_values("px")
    if len(df) > MAX_VP_BINS:
        coarse = 1.0
        df = df.copy()
        df["px"] = (df["px"] / coarse).round() * coarse
        df = df.groupby("px", as_index=False)[["vol", "buy", "sell"]].sum()
        tick_size = coarse
    df.attrs["tick_size"] = tick_size
    return df


def load_vbp_windows(
    windows: list[tuple[datetime, datetime]],
    *,
    tick_size: float = TICK_SIZE,
    rth_only: bool = False,
) -> list[pd.DataFrame]:
    """Volume-at-price for each ``[start, end)`` window (one TBBO scan)."""
    empty = pd.DataFrame(columns=["px", "vol", "buy", "sell"])
    if not windows:
        return []
    if len(windows) == 1:
        return [load_vbp(
            windows[0][0], windows[0][1], tick_size=tick_size, rth_only=rth_only,
        )]
    wmin = min(s for s, _ in windows)
    wmax = max(e for _, e in windows)
    if wmax <= wmin:
        return [empty.copy() for _ in windows]
    d0 = wmin.astimezone(CT).date()
    d1 = wmax.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)})
    if not symbols:
        return [empty.copy() for _ in windows]
    window_parts = []
    window_params: list[Any] = []
    for i, (s, e) in enumerate(windows):
        window_parts.append("(%s, %s, %s)")
        window_params.extend([i, s, e])
    windows_sql = ", ".join(window_parts)
    sql = (
        f"WITH vp_windows(wid, start_ts, end_ts) AS (VALUES {windows_sql}) "
        "SELECT vp_windows.wid AS wid, "
        f"(ts_event AT TIME ZONE 'America/Chicago')::date AS chi_day, "
        f"symbol, "
        f"(ROUND((price / %s)::numeric, 0) * %s)::double precision AS px, "
        f"SUM(size)::double precision AS vol, "
        f"SUM(CASE WHEN side = 'B' THEN size ELSE 0 END)::double precision AS buy, "
        f"SUM(CASE WHEN side = 'A' THEN size ELSE 0 END)::double precision AS sell "
        f"FROM databento.tbbo "
        "JOIN vp_windows ON ts_event >= vp_windows.start_ts AND ts_event < vp_windows.end_ts "
        f"WHERE ts_event >= %s AND ts_event < %s AND symbol = ANY(%s) "
        f"AND price IS NOT NULL AND size IS NOT NULL "
        f"{RTH_SQL if rth_only else ''} "
        f"GROUP BY 1, 2, 3, 4"
    )
    params = tuple(window_params + [tick_size, tick_size, wmin, wmax, symbols])
    df = _query(sql, params, timeout_ms=90_000)
    out = [empty.copy() for _ in windows]
    if df.empty:
        return out
    lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)}
    day = pd.to_datetime(df["chi_day"]).dt.date
    keep = np.array(
        [lead.get(d) == str(s) for d, s in zip(day, df["symbol"].astype(str))],
        dtype=bool,
    )
    df = df.loc[keep]
    if df.empty:
        return out
    for col in ("wid", "px", "vol", "buy", "sell"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wid", "px"])
    grouped = df.groupby(["wid", "px"], as_index=False)[["vol", "buy", "sell"]].sum()
    for wid, part in grouped.groupby("wid"):
        i = int(wid)
        if i < 0 or i >= len(out):
            continue
        chunk = part[["px", "vol", "buy", "sell"]].sort_values("px")
        if len(chunk) > MAX_VP_BINS:
            chunk = chunk.copy()
            chunk["px"] = (chunk["px"] / 1.0).round() * 1.0
            chunk = chunk.groupby("px", as_index=False)[["vol", "buy", "sell"]].sum()
        chunk.attrs["tick_size"] = tick_size
        out[i] = chunk.reset_index(drop=True)
    return out


def load_vbp_progression(
    start: datetime,
    end: datetime,
    timeframe: str,
    buckets: pd.DatetimeIndex,
    *,
    rth_only: bool = False,
) -> list[dict[str, Any]]:
    """Return causal profile bounds at each chart bucket in one session window."""
    if end <= start or buckets is None or len(buckets) == 0:
        return []
    interval = {
        "5s": "5 seconds", "15s": "15 seconds", "1min": "1 minute",
        "5min": "5 minutes", "15min": "15 minutes", "1hour": "1 hour",
        "1day": "1 day",
    }.get(coerce_tf(timeframe))
    if interval is None:
        return []
    d0 = start.astimezone(CT).date()
    d1 = end.astimezone(CT).date()
    symbols = sorted({sym for _, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)})
    if not symbols:
        return []
    df = _query(
        "SELECT time_bucket(%s::interval, ts_event) AS bucket, "
        "(ts_event AT TIME ZONE 'America/Chicago')::date AS chi_day, symbol, "
        "(ROUND((price / %s)::numeric, 0) * %s)::double precision AS px, "
        "SUM(size)::double precision AS vol "
        "FROM databento.tbbo WHERE ts_event >= %s AND ts_event < %s "
        "AND symbol = ANY(%s) AND price IS NOT NULL AND size IS NOT NULL "
        f"{RTH_SQL if rth_only else ''} "
        "GROUP BY 1, 2, 3, 4 ORDER BY 1, 4",
        (interval, TICK_SIZE, TICK_SIZE, start, end, symbols),
        timeout_ms=90_000,
    )
    grouped: dict[pd.Timestamp, dict[float, float]] = {}
    if not df.empty:
        df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
        lead = {d: sym for d, sym in active_contracts_in_range(ROOT, d0 - timedelta(days=1), d1)}
        for row in df.itertuples(index=False):
            if lead.get(pd.Timestamp(row.chi_day).date()) != str(row.symbol):
                continue
            grouped.setdefault(pd.Timestamp(row.bucket), {})[float(row.px)] = (
                grouped.setdefault(pd.Timestamp(row.bucket), {}).get(float(row.px), 0.0)
                + float(row.vol or 0)
            )
    visible = pd.DatetimeIndex(pd.to_datetime(buckets, utc=True))
    visible = visible[(visible >= pd.Timestamp(start)) & (visible < pd.Timestamp(end))]
    cumulative: dict[float, float] = {}
    out: list[dict[str, Any]] = []
    for bucket in visible:
        for px, vol in grouped.get(pd.Timestamp(bucket), {}).items():
            cumulative[px] = cumulative.get(px, 0.0) + vol
        if not cumulative:
            continue
        prices = np.array(sorted(cumulative), dtype="float64")
        volumes = np.array([cumulative[p] for p in prices], dtype="float64")
        poc, va_high, va_low = value_area(prices, volumes)
        out.append({
            "t": int(pd.Timestamp(bucket).timestamp()),
            "high": float(prices.max()),
            "low": float(prices.min()),
            "poc": _finite(poc),
            "va_high": _finite(va_high),
            "va_low": _finite(va_low),
        })
    return out


def merge_vbp(frames: list[pd.DataFrame], *, tick_size: float = TICK_SIZE) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        empty = pd.DataFrame(columns=["px", "vol", "buy", "sell"])
        empty.attrs["tick_size"] = tick_size
        return empty
    df = pd.concat(parts, ignore_index=True)
    for col in ("px", "vol", "buy", "sell"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["px"])
    df = df.groupby("px", as_index=False)[["vol", "buy", "sell"]].sum().sort_values("px")
    ts = float(parts[0].attrs.get("tick_size", tick_size) or tick_size)
    if len(df) > MAX_VP_BINS:
        df = df.copy()
        df["px"] = (df["px"] / 1.0).round() * 1.0
        df = df.groupby("px", as_index=False)[["vol", "buy", "sell"]].sum()
        ts = 1.0
    df.attrs["tick_size"] = ts
    return df.reset_index(drop=True)


def _chart_ref(ohlc: pd.DataFrame | None, end: datetime) -> datetime:
    ref = end
    if ohlc is not None and not ohlc.empty and "bucket" in ohlc.columns:
        last = _as_utc(ohlc["bucket"].iloc[-1]).to_pydatetime()
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        else:
            last = last.astimezone(timezone.utc)
        ref = last
    return ref


def _clip_end(stop: datetime, now: datetime) -> datetime:
    return stop if stop <= now else now


def _clip_wins_rth(
    wins: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """Intersect tape windows with cash 08:30–15:00 CT."""
    out: list[tuple[datetime, datetime]] = []
    for s, e in wins:
        if e <= s:
            continue
        for a, b in iter_daily_session_windows(s, e, "cash"):
            lo, hi = max(s, a), min(e, b)
            if hi > lo:
                out.append((lo, hi))
    return out


def _recurring_jobs_with_visible_candles(
    jobs: list[dict[str, Any]],
    ohlc: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Keep recurring VP jobs only when their windows overlap visible candles."""
    if ohlc is None or ohlc.empty or "bucket" not in ohlc.columns:
        return jobs
    buckets = pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True)).sort_values().unique()
    if not len(buckets):
        return jobs
    out: list[dict[str, Any]] = []
    for job in jobs:
        if job["kind"] != "recurring":
            out.append(job)
            continue
        visible = False
        for start, end in job.get("win") or []:
            pos = buckets.searchsorted(pd.Timestamp(start), side="left")
            if pos < len(buckets) and buckets[pos] < pd.Timestamp(end):
                visible = True
                break
        if visible:
            out.append(job)
    return out


def _limit_recurring_jobs_per_profile(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply recurring VP safety limits independently to each profile spec."""
    # Non-recurring kinds (Single, Level Blend, …) pass through unchanged.
    pinned = [job for job in jobs if job["kind"] != "recurring"]
    by_spec: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        if job["kind"] == "recurring":
            by_spec.setdefault(str(job.get("spec_id") or job["id"]), []).append(job)
    kept = pinned[:]
    for spec_jobs in by_spec.values():
        recent = spec_jobs[-MAX_VP_EXPANDED:]
        span = 0.0
        profile_kept: list[dict[str, Any]] = []
        for job in reversed(recent):
            job_span = sum(max(0.0, (end - start).total_seconds()) for start, end in job["win"])
            if profile_kept and span + job_span > MAX_VP_SECONDS:
                continue
            span += job_span
            profile_kept.append(job)
        kept.extend(reversed(profile_kept))
    return sorted(kept, key=lambda job: (job["start"], str(job["id"])))


def expand_vp_jobs(
    specs: list[dict[str, Any]],
    *,
    ohlc: pd.DataFrame | None,
    chart_start: datetime,
    chart_end: datetime,
    now: datetime,
    rth_only: bool = False,
) -> list[dict[str, Any]]:
    """Expand VP specs into concrete tape windows (before the VBP query)."""
    ref = _chart_ref(ohlc, chart_end)
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        kind = spec["kind"]
        stf = spec["session_tf"]
        hm_kw = {
            "from_hm": spec.get("session_from"),
            "to_hm": spec.get("session_to"),
        }
        if kind in ("current", "level_blend"):
            pinned_date = spec.get("session_date")
            # Level Blend always anchors to the cash window of the eval date
            # (ADR-126 closed-session levels; blank Date = last candle session).
            use_tf = "cash" if kind == "level_blend" else stf
            if pinned_date is not None:
                s, e = session_tf_window_for_end_date(pinned_date, use_tf, **hm_kw)
            else:
                s, e = session_tf_window(ref, use_tf, **hm_kw)
            e = _clip_end(e, now)
            if e <= s:
                e = s + timedelta(seconds=1)
            jobs.append({**spec, "start": s, "end": e, "win": [(s, e)]})
        else:
            if spec.get("recur") == "weekly":
                weeks = iter_weekly_session_windows(
                    chart_start, chart_end, stf, **hm_kw,
                )
                for ws, we, parts in weeks:
                    ee = _clip_end(we, now)
                    if ee <= ws:
                        continue
                    clipped = [
                        (a, min(b, ee)) for a, b in parts if min(b, ee) > a
                    ]
                    jobs.append({
                        **spec,
                        "id": f"{spec['id']}:{int(ws.timestamp())}",
                        "spec_id": spec["id"],
                        "start": ws, "end": ee,
                        "win": clipped or [(ws, ee)],
                    })
            else:
                for s, e in iter_daily_session_windows(
                    chart_start, chart_end, stf, **hm_kw,
                ):
                    ee = _clip_end(e, now)
                    if ee <= s:
                        continue
                    jobs.append({
                        **spec,
                        "id": f"{spec['id']}:{int(s.timestamp())}",
                        "spec_id": spec["id"],
                        "start": s, "end": ee,
                        "win": [(s, ee)],
                    })
    jobs = _recurring_jobs_with_visible_candles(jobs, ohlc)
    jobs = _limit_recurring_jobs_per_profile(jobs)
    if rth_only:
        clipped: list[dict[str, Any]] = []
        for job in jobs:
            wins = _clip_wins_rth(list(job.get("win") or []))
            if not wins:
                job = {**job, "win": [], "start": job["start"], "end": job["end"]}
                clipped.append(job)
                continue
            job = {**job, "win": wins, "start": wins[0][0], "end": wins[-1][1]}
            clipped.append(job)
        jobs = clipped
    return jobs


def profiles_payloads(
    jobs: list[dict[str, Any]], *, ohlc: pd.DataFrame | None = None,
    rth_only: bool = False, timeframe: str = "1min",
) -> list[dict[str, Any]]:
    """Run VBP for expanded jobs and attach layout fields."""
    flat: list[tuple[datetime, datetime]] = []
    index: list[tuple[int, int]] = []
    for ji, job in enumerate(jobs):
        if job.get("kind") == "level_blend":
            continue  # no tape histogram — KDE bounds from session_levels SSOT
        for pj, w in enumerate(job["win"]):
            if w[1] > w[0]:
                flat.append(w)
                index.append((ji, pj))
    frames = load_vbp_windows(flat, rth_only=rth_only) if flat else []
    grouped: dict[int, list[pd.DataFrame]] = {i: [] for i in range(len(jobs))}
    for fr, (ji, _pj) in zip(frames, index):
        grouped[ji].append(fr)
    out: list[dict[str, Any]] = []
    for ji, job in enumerate(jobs):
        if job.get("kind") == "level_blend":
            out.append(level_blend_payload(job, ohlc=ohlc))
            continue
        vbp = merge_vbp(grouped.get(ji, []))
        if job["kind"] == "current" and (vbp is None or vbp.empty):
            hm_kw = {
                "from_hm": job.get("session_from"),
                "to_hm": job.get("session_to"),
            }
            pinned_date = job.get("session_date")
            if pinned_date is not None:
                prev_s, prev_e = session_tf_window_for_end_date(
                    pinned_date - timedelta(days=1), job["session_tf"], **hm_kw,
                )
            else:
                prev_s, prev_e = session_tf_window(
                    job["start"] - timedelta(seconds=1), job["session_tf"], **hm_kw,
                )
            vbp = load_vbp(prev_s, prev_e, rth_only=rth_only)
            prev_wins = _clip_wins_rth([(prev_s, prev_e)]) if rth_only else [(prev_s, prev_e)]
            if prev_wins:
                job = {**job, "start": prev_wins[0][0], "end": prev_wins[-1][1], "win": prev_wins}
            else:
                job = {**job, "start": prev_s, "end": prev_e, "win": [(prev_s, prev_e)]}
        payload = profile_payload(
            vbp if vbp is not None else pd.DataFrame(),
            start=job["start"], end=job["end"], origin=job["session_tf"],
            lvn_pct=float(job.get("lvn_pct") or LVN_PCT_DEFAULT),
            lvn_min_groups=int(job.get("lvn_min_groups") or LVN_GROUPS_DEFAULT),
            group_ticks=int(job.get("group_ticks") or GROUP_TICKS_DEFAULT),
            hvn_method=job.get("hvn_method") or HVN_METHOD_DEFAULT,
            kde_bandwidth=float(job.get("kde_bandwidth") or KDE_BW_DEFAULT),
            kde_prominence=float(job.get("kde_prominence") or KDE_PROM_DEFAULT),
            kde_valley_depth=float(job.get("kde_valley_depth") or KDE_VALLEY_DEFAULT),
            kde_volume=float(job.get("kde_volume") or KDE_VOL_DEFAULT),
            kde_separation=int(job.get("kde_separation") or KDE_SEP_DEFAULT),
            kde_valley_width=int(job.get("kde_valley_width") or KDE_VALLEY_WIDTH_DEFAULT),
            kde_max_zones=int(job.get("kde_max_zones") or KDE_MAX_ZONES_DEFAULT),
        )
        _attach_profile_layout(payload, job, ohlc=ohlc)
        payload["plot_variable_bounds"] = bool(job.get("plot_variable_bounds"))
        if payload["plot_variable_bounds"] and ohlc is not None and not ohlc.empty:
            chart_buckets = pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True))
            variable: list[dict[str, Any]] = []
            for win_start, win_end in job.get("win") or []:
                variable.extend(load_vbp_progression(
                    win_start, win_end, timeframe, chart_buckets, rth_only=rth_only,
                ))
            payload["variable_bounds"] = variable
        else:
            payload["variable_bounds"] = []
        out.append(payload)
    return out


def _attach_profile_layout(
    payload: dict[str, Any],
    job: dict[str, Any],
    *,
    ohlc: pd.DataFrame | None = None,
) -> None:
    """Shared layout / card fields for histogram and Level Blend payloads."""
    sess_date = job.get("session_date")
    if sess_date is None and job.get("start") is not None:
        # Derive session-end date from the cash/full window end for overlays.
        end = job["start"] if job.get("end") is None else job["end"]
        chi = _as_utc(end).to_pydatetime().astimezone(CT)
        # Exclusive end at 15:00 / 16:00 → session-end calendar date is that day
        # when end is noon-ish; for midnight BOD end it is the previous day.
        if (chi.hour, chi.minute) == (0, 0) and chi.second == 0:
            sess_date = (chi - timedelta(seconds=1)).date()
        else:
            sess_date = chi.date()
    extend = int(job.get("extend") if job.get("extend") is not None else (
        1 if job.get("extend_lines") else 0
    ))
    if extend not in VP_EXTEND:
        extend = 1 if extend else 0
    n = parse_vp_n(job.get("n"), fallback=1)
    payload.update({
        "id": job["id"],
        "spec_id": job.get("spec_id") or job["id"],
        "n": n,
        "label": f"VP{n}",
        "kind": job["kind"],
        "session_tf": job["session_tf"],
        "placement": job["placement"],
        "overlay_side": job["overlay_side"],
        "extend": extend,
        "extend_lines": extend >= 1,
        "extend_va": extend >= 1,
        "hist_style": job.get("hist_style") or "volume",
        "group_ticks": int(job.get("group_ticks") or GROUP_TICKS_DEFAULT),
        "right_display": job.get("right_display") or "total_split",
        "left_display": job.get("left_display") or "none",
        "right_opacity": float(job.get("right_opacity") if job.get("right_opacity") is not None else 100),
        "left_opacity": float(job.get("left_opacity") if job.get("left_opacity") is not None else 100),
        "hist_width": float(job.get("hist_width") or HIST_WIDTH_DEFAULT),
        "chart_labels": bool(job.get("chart_labels")),
        "axis_labels": bool(job.get("axis_labels")),
        "show_lvn": bool(job.get("show_lvn")),
        "lvn_pct": float(job.get("lvn_pct") or LVN_PCT_DEFAULT),
        "lvn_min_groups": int(job.get("lvn_min_groups") or LVN_GROUPS_DEFAULT),
        "extend_lvn": bool(job.get("extend_lvn")),
        "hvn_method": job.get("hvn_method") or HVN_METHOD_DEFAULT,
        "kde_bandwidth": float(job.get("kde_bandwidth") or KDE_BW_DEFAULT),
        "kde_prominence": float(job.get("kde_prominence") or KDE_PROM_DEFAULT),
        "kde_valley_depth": float(job.get("kde_valley_depth") or KDE_VALLEY_DEFAULT),
        "kde_volume": float(job.get("kde_volume") or KDE_VOL_DEFAULT),
        "kde_separation": int(job.get("kde_separation") or KDE_SEP_DEFAULT),
        "kde_valley_width": int(job.get("kde_valley_width") or KDE_VALLEY_WIDTH_DEFAULT),
        "kde_max_zones": int(job.get("kde_max_zones") or KDE_MAX_ZONES_DEFAULT),
        "lookback_sessions": int(
            job.get("lookback_sessions") or LEVEL_BLEND_LOOKBACK_DEFAULT
        ),
        "recur": job.get("recur"),
        "session_from": format_session_hm(job.get("session_from")),
        "session_to": format_session_hm(job.get("session_to")),
        "session_date": sess_date.isoformat() if isinstance(sess_date, date) else (
            sess_date if sess_date is not None else None
        ),
    })
    hi_body, lo_body = profile_body_extremes(ohlc, job.get("win") or [])
    payload["hi_body"] = hi_body
    payload["lo_body"] = lo_body


def _eval_day_range(
    ohlc: pd.DataFrame | None,
    start: datetime,
    end: datetime,
) -> tuple[float | None, float | None]:
    """High/low of chart bars inside ``[start, end)`` for lookback extension."""
    if ohlc is None or ohlc.empty or "bucket" not in ohlc.columns:
        return None, None
    buck = pd.to_datetime(ohlc["bucket"], utc=True)
    mask = (buck >= pd.Timestamp(start)) & (buck < pd.Timestamp(end))
    if not bool(mask.any()):
        return None, None
    hi = pd.to_numeric(ohlc.loc[mask, "high"], errors="coerce")
    lo = pd.to_numeric(ohlc.loc[mask, "low"], errors="coerce")
    if not hi.notna().any() or not lo.notna().any():
        return None, None
    return float(hi.max()), float(lo.min())


def level_blend_payload(
    job: dict[str, Any],
    *,
    ohlc: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """KDE-clustered prior-session bounds for a Level Blend VP (ADR-126).

    Uses the cascade session-profile SSOT (``session_levels.parquet``, ADR-129
    tick parity / ADR-134). Prior *lookback_sessions* closed cash sessions
    before the eval date are merged; when the eval day's price range sits
    outside the clustered span the lookback grows by ``extend_step`` until it
    covers the day (same policy as :func:`levels_for_range`).
    """
    from src.features.cash_session_profile import LevelConfig, levels_for_range
    from src.features.session_profile import history_before, load_levels

    start = job["start"]
    end = job["end"]
    sess_date = job.get("session_date")
    if sess_date is None:
        chi = _as_utc(end).to_pydatetime().astimezone(CT)
        if (chi.hour, chi.minute) == (0, 0) and chi.second == 0:
            sess_date = (chi - timedelta(seconds=1)).date()
        else:
            sess_date = chi.date()
    lookback = int(job.get("lookback_sessions") or LEVEL_BLEND_LOOKBACK_DEFAULT)
    cfg = LevelConfig(
        lookback_sessions=lookback,
        sigma_pts=LEVEL_BLEND_SIGMA_DEFAULT,
        buffer_pts=LEVEL_BLEND_BUFFER_DEFAULT,
    )
    day_hi, day_lo = _eval_day_range(ohlc, start, end)
    bounds: list[Any] = []
    sessions_used = 0
    err: str | None = None
    try:
        levels = load_levels()
        hist = history_before(levels, sess_date, cfg.max_lookback_sessions)
        if not hist:
            err = "no prior session levels"
        else:
            # Prefer the eval day's observed range; fall back to mid of newest
            # prior session so a blank lookback day still builds something.
            if day_hi is None or day_lo is None:
                probe = 0.5 * (hist[0].lo + hist[0].hi)
                day_lo = day_hi = probe
            bounds, sessions_used = levels_for_range(hist, day_lo, day_hi, cfg)
    except FileNotFoundError as exc:
        err = str(exc)
    except Exception as exc:  # noqa: BLE001 — surface to UI, keep chart alive
        err = f"level blend failed: {exc}"

    blend_bounds: list[dict[str, Any]] = []
    for b in bounds:
        members_out: list[dict[str, Any]] = []
        for member in (b.members or []):
            if len(member) >= 3:
                kind, price, sess = member[0], member[1], member[2]
            else:
                kind, price, sess = member[0], member[1], None
            sess_iso = None
            if sess is not None and not (isinstance(sess, float) and math.isnan(sess)):
                try:
                    sess_iso = pd.Timestamp(sess).date().isoformat()
                except (TypeError, ValueError):
                    sess_iso = None
            members_out.append({
                "kind": kind,
                "price": float(price),
                "session": sess_iso,
            })
        lo_session = hi_session = None
        if members_out:
            priced = [
                m for m in members_out
                if m.get("price") is not None and math.isfinite(float(m["price"]))
            ]
            if priced:
                lo_m = min(
                    priced,
                    key=lambda m: (float(m["price"]), str(m.get("session") or "")),
                )
                hi_m = max(
                    priced,
                    key=lambda m: (float(m["price"]), str(m.get("session") or "")),
                )
                lo_session = lo_m.get("session")
                hi_session = hi_m.get("session")
        blend_bounds.append({
            "lo": float(b.lo),
            "hi": float(b.hi),
            "strength": float(b.strength),
            "n_levels": int(b.n_levels),
            "kinds": dict(b.kinds),
            "members": members_out,
            "lo_session": lo_session,
            "hi_session": hi_session,
        })
    if blend_bounds:
        hi = max(b["hi"] for b in blend_bounds)
        lo = min(b["lo"] for b in blend_bounds)
    else:
        hi = lo = float("nan")
    payload = profile_payload(
        pd.DataFrame(), start=start, end=end, origin="cash",
    )
    payload.update({
        "bins": [],
        "nodes": [],
        "coarse_bins": [],
        "coarse_nodes": [],
        "high": _finite(hi),
        "low": _finite(lo),
        "poc": None,
        "va_high": None,
        "va_low": None,
        "level_blend": True,
        "bounds": blend_bounds,
        "sessions_used": int(sessions_used),
        "lookback_sessions": lookback,
        "eval_high": _finite(day_hi) if day_hi is not None else None,
        "eval_low": _finite(day_lo) if day_lo is not None else None,
        "error": err,
    })
    _attach_profile_layout(payload, {**job, "session_date": sess_date}, ohlc=ohlc)
    payload["right_display"] = "none"
    payload["left_display"] = "none"
    payload["show_lvn"] = False
    payload["extend_lvn"] = False
    payload["hvn_method"] = "off"
    payload["plot_variable_bounds"] = False
    payload["variable_bounds"] = []
    return payload


def profile_body_extremes(
    ohlc: pd.DataFrame | None,
    windows: list[tuple[datetime, datetime]],
) -> tuple[float | None, float | None]:
    """Return body levels for the candles setting a profile's high and low."""
    if ohlc is None or ohlc.empty or "bucket" not in ohlc.columns:
        return None, None
    required = ("open", "high", "low", "close")
    if any(col not in ohlc.columns for col in required):
        return None, None
    buckets = pd.to_datetime(ohlc["bucket"], utc=True)
    in_window = pd.Series(False, index=ohlc.index)
    for start, end in windows:
        in_window |= (buckets >= pd.Timestamp(start)) & (buckets < pd.Timestamp(end))
    candles = ohlc.loc[in_window, list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if candles.empty:
        return None, None
    hi_row = candles.loc[candles["high"].idxmax()]
    lo_row = candles.loc[candles["low"].idxmin()]
    return _finite(max(hi_row["open"], hi_row["close"])), _finite(min(lo_row["open"], lo_row["close"]))


def profile_payload(
    vbp: pd.DataFrame,
    *,
    start: datetime,
    end: datetime,
    origin: str | None = None,
    tick_size: float = TICK_SIZE,
    lvn_pct: float = LVN_PCT_DEFAULT,
    lvn_min_groups: int = LVN_GROUPS_DEFAULT,
    group_ticks: int = GROUP_TICKS_DEFAULT,
    hvn_method: str = HVN_METHOD_DEFAULT,
    kde_bandwidth: float = KDE_BW_DEFAULT,
    kde_prominence: float = KDE_PROM_DEFAULT,
    kde_valley_depth: float = KDE_VALLEY_DEFAULT,
    kde_volume: float = KDE_VOL_DEFAULT,
    kde_separation: int = KDE_SEP_DEFAULT,
    kde_valley_width: int = KDE_VALLEY_WIDTH_DEFAULT,
    kde_max_zones: int = KDE_MAX_ZONES_DEFAULT,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    coarse: list[dict[str, Any]] = []
    coarse_nodes: list[dict[str, Any]] = []
    hvn_zones: list[dict[str, Any]] = []
    hvn_peaks: list[dict[str, Any]] = []
    lvn_valleys: list[dict[str, Any]] = []
    if group_ticks not in GROUP_TICKS:
        group_ticks = GROUP_TICKS_DEFAULT
    if vbp is None or vbp.empty:
        poc = va_h = va_l = hi = lo = float("nan")
        bins: list[dict[str, float]] = []
        total = 0.0
        ts = tick_size
    else:
        ts = float(vbp.attrs.get("tick_size", tick_size) or tick_size)
        poc, va_h, va_l = value_area(vbp["px"].to_numpy(), vbp["vol"].to_numpy())
        px = pd.to_numeric(vbp["px"], errors="coerce")
        hi = float(px.max()) if px.notna().any() else float("nan")
        lo = float(px.min()) if px.notna().any() else float("nan")
        total = float(pd.to_numeric(vbp["vol"], errors="coerce").fillna(0).sum())
        bins = [
            {
                "p": _finite(r.px) or 0.0,
                "v": _finite(r.vol) or 0.0,
                "b": _finite(getattr(r, "buy", 0.0)) or 0.0,
                "s": _finite(getattr(r, "sell", 0.0)) or 0.0,
            }
            for r in vbp.itertuples(index=False)
        ]
        buy_col = vbp["buy"].to_numpy() if "buy" in vbp.columns else None
        sell_col = vbp["sell"].to_numpy() if "sell" in vbp.columns else None
        nodes = volume_nodes(
            vbp["px"].to_numpy(), vbp["vol"].to_numpy(), buy_col, sell_col,
            lvn_pct=lvn_pct, tick_size=ts,
        )
        coarse = grouped_tick_bins(
            vbp["px"].to_numpy(), vbp["vol"].to_numpy(), buy_col, sell_col,
            tick_size=ts, group_ticks=group_ticks,
        )
        coarse_nodes = lvn_nodes_from_bins(coarse, lvn_pct=lvn_pct, lvn_min_groups=lvn_min_groups)
        if hvn_method == "adaptive_kde":
            source = coarse if group_ticks > 1 else bins
            hvn_zones = adaptive_kde_hvn_zones(
                source, tick_size=ts * (group_ticks if group_ticks > 1 else 1),
                bandwidth=kde_bandwidth, prominence=kde_prominence,
                valley_depth=kde_valley_depth, volume_fraction=kde_volume,
                min_separation=kde_separation, valley_width=kde_valley_width,
                max_zones=kde_max_zones,
            )
            for zone in hvn_zones:
                zone["va_relation"] = (
                    "inside" if zone["lo"] >= float(va_l) and zone["hi"] <= float(va_h)
                    else "outside" if zone["hi"] < float(va_l) or zone["lo"] > float(va_h)
                    else "intersects"
                )
            ordered_hvn = sorted(hvn_zones, key=lambda z: z["peak"])
            for idx, peak in enumerate(ordered_hvn, 1):
                peak["label"] = f"HVN {idx}"
            hvn_peaks = [
                {"label": z["label"], "price": z["peak"],
                 "prominence": z["prominence"], "volume_fraction": z["volume_fraction"]}
                for z in ordered_hvn
            ]
            lvn_valleys = []
            for idx, (left, right) in enumerate(zip(ordered_hvn, ordered_hvn[1:]), 1):
                price = (float(left["right_valley"]) + float(right["left_valley"])) / 2.0
                lvn_valleys.append({"label": f"LVN {idx}", "price": price})
        else:
            hvn_peaks = []
            lvn_valleys = []
    return {
        "origin": origin,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "start_unix": int(start.timestamp()),
        "end_unix": int(end.timestamp()),
        "tick_size": ts,
        "poc": _finite(poc),
        "va_high": _finite(va_h),
        "va_low": _finite(va_l),
        "high": _finite(hi),
        "low": _finite(lo),
        "total_volume": total,
        "lvn_pct": float(lvn_pct),
        "lvn_min_groups": int(lvn_min_groups),
        "group_ticks": group_ticks,
        "bins": bins,
        "nodes": nodes,
        "coarse_bins": coarse,
        "coarse_nodes": coarse_nodes,
        "hvn_method": hvn_method,
        "kde_bandwidth": float(kde_bandwidth),
        "kde_prominence": float(kde_prominence),
        "kde_valley_depth": float(kde_valley_depth),
        "kde_volume": float(kde_volume),
        "kde_separation": int(kde_separation),
        "kde_valley_width": int(kde_valley_width),
        "kde_max_zones": int(kde_max_zones),
        "hvn_zones": hvn_zones,
        "hvn_peaks": hvn_peaks,
        "lvn_valleys": lvn_valleys,
    }


def money_flow_index(ohlc: pd.DataFrame, period: int = MFI_PERIOD) -> pd.Series:
    """Wilder MFI: 100 - 100 / (1 + posMF/negMF) over *period* bars.

    Raw money flow = typical price × volume. Typical price =
    (high + low + close) / 3. Positive when typical price rises vs the
    prior bar, negative when it falls.
    """
    if ohlc is None or ohlc.empty:
        return pd.Series(dtype="float64")
    h = pd.to_numeric(ohlc["high"], errors="coerce")
    low = pd.to_numeric(ohlc["low"], errors="coerce")
    c = pd.to_numeric(ohlc["close"], errors="coerce")
    vol = pd.to_numeric(ohlc["volume"], errors="coerce").fillna(0.0) if "volume" in ohlc.columns else 0.0
    tp = (h + low + c) / 3.0
    raw = tp * vol
    dtp = tp.diff()
    pos = raw.where(dtp > 0, 0.0)
    neg = raw.where(dtp < 0, 0.0)
    pos_s = pos.rolling(int(period), min_periods=int(period)).sum()
    neg_s = neg.rolling(int(period), min_periods=int(period)).sum()
    ratio = pos_s / neg_s.replace(0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    mfi = mfi.mask(neg_s.eq(0) & pos_s.gt(0), 100.0)
    mfi = mfi.mask(pos_s.eq(0) & neg_s.gt(0), 0.0)
    idx = pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True))
    mfi.index = idx
    return mfi.astype("float64")


def clv_volume(ohlc: pd.DataFrame) -> pd.Series:
    """Per-bar A/D increment: CLV × volume."""
    if ohlc is None or ohlc.empty:
        return pd.Series(dtype="float64")
    h = pd.to_numeric(ohlc["high"], errors="coerce")
    low = pd.to_numeric(ohlc["low"], errors="coerce")
    c = pd.to_numeric(ohlc["close"], errors="coerce")
    vol = pd.to_numeric(ohlc["volume"], errors="coerce").fillna(0.0) if "volume" in ohlc.columns else 0.0
    rng = (h - low).replace(0, np.nan)
    clv = (2.0 * c - h - low) / rng
    out = clv.fillna(0.0) * vol
    idx = pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True))
    out.index = idx
    return out.astype("float64")


def _align(series: pd.Series, buckets: pd.DatetimeIndex) -> list[float | None]:
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s[~s.index.duplicated(keep="last")].reindex(buckets)
    return [_finite(v) for v in s.to_numpy()]


def _overlays_from_ohlc(
    ohlc: pd.DataFrame,
    keys: list[str],
    *,
    sar_params: dict[str, float] | None = None,
) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    if ohlc.empty or not keys:
        return out
    idx = pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True))
    want_stoch = any(k.startswith("so_") for k in keys)
    want_sar = "ps_sar" in keys
    want_clv = "cs_clv" in keys
    want_mfi = "mf_mfi" in keys
    candles = pd.DataFrame(
        {
            "open": ohlc["open"].to_numpy(dtype="float64"),
            "high": ohlc["high"].to_numpy(dtype="float64"),
            "low": ohlc["low"].to_numpy(dtype="float64"),
            "close": ohlc["close"].to_numpy(dtype="float64"),
            "volume": ohlc["volume"].to_numpy(dtype="float64")
            if "volume" in ohlc.columns else np.nan,
        },
        index=idx,
    )
    if want_clv:
        rng = (candles["high"] - candles["low"]).replace(0, np.nan)
        clv = (2.0 * candles["close"] - candles["high"] - candles["low"]) / rng
        out["cs_clv"] = [_finite(v) for v in clv.to_numpy()]
    if want_mfi:
        mfi = money_flow_index(ohlc)
        out["mf_mfi"] = [_finite(v) for v in mfi.to_numpy()]
    if want_stoch:
        from src.features.stochastic import compute as so_compute
        so = so_compute(candles)
        for k in keys:
            if k.startswith("so_") and k in so.columns:
                out[k] = [_finite(v) for v in so[k].to_numpy()]
    if want_sar:
        from src.features.psar import sar_on_closed
        sp = sar_params or parse_sar_params(None)
        raw = sar_on_closed(
            candles,
            start=float(sp["start"]),
            increment=float(sp["increment"]),
            maximum=float(sp["maximum"]),
        )
        if raw.empty:
            out["ps_sar"] = [None] * len(ohlc)
        else:
            out["ps_sar"] = _align(raw["sar"], idx)
    return out


def _overlays_from_ticks(
    tbbo: pd.DataFrame, buckets: pd.DatetimeIndex, freq: str, keys: list[str],
    *, vwap_origin: str = "session",
) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    if tbbo is None or tbbo.empty or not keys:
        return {k: [None] * len(buckets) for k in keys}
    df = tbbo.sort_values("ts_event").copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    if "of_ofi" in keys:
        from src.features.order_flow import _compute_ofi
        ofi = _compute_ofi(df, freq)
        if ofi is not None and "of_ofi" in ofi.columns:
            out["of_ofi"] = _align(ofi["of_ofi"], buckets)
        else:
            out["of_ofi"] = [None] * len(buckets)
    vwap_keys = [k for k in keys if k in _VWAP_SERIES_KEYS]
    if vwap_keys:
        ticks = df.set_index("ts_event")
        px = pd.to_numeric(ticks["price"], errors="coerce")
        sz = pd.to_numeric(ticks["size"], errors="coerce")
        ticks["pv"] = px * sz
        ticks["p2v"] = (px ** 2) * sz
        vwap_origin = coerce_origin(vwap_origin)
        ticks["origin_period"] = origin_period_key(ticks.index, vwap_origin)
        cum_vol = ticks.groupby("origin_period", sort=False)["size"].cumsum().replace(0, np.nan)
        ticks["vwap"] = ticks.groupby("origin_period", sort=False)["pv"].cumsum() / cum_vol
        want_bands = "vw_bb_upper" in vwap_keys or "vw_bb_lower" in vwap_keys
        if want_bands:
            cum_p2v = ticks.groupby("origin_period", sort=False)["p2v"].cumsum()
            var = (cum_p2v / cum_vol) - (ticks["vwap"] ** 2)
            ticks["vwap_std"] = np.sqrt(np.clip(var.to_numpy(dtype="float64"), 0.0, None))
            last = ticks.resample(freq).agg(
                vwap=("vwap", "last"), close=("price", "last"), vwap_std=("vwap_std", "last"),
            )
            last["vwap_std"] = last["vwap_std"].ffill()
        else:
            last = ticks.resample(freq).agg(vwap=("vwap", "last"), close=("price", "last"))
        last["vwap"] = last["vwap"].ffill()
        last["close"] = last["close"].ffill()
        if "vw_vwap" in vwap_keys:
            out["vw_vwap"] = _align(last["vwap"], buckets)
        if "vw_vwap_dist" in vwap_keys:
            dist = (last["close"] - last["vwap"]) / last["vwap"].replace(0, np.nan)
            out["vw_vwap_dist"] = _align(dist, buckets)
        if want_bands:
            std = last["vwap_std"]
            if "vw_bb_upper" in vwap_keys:
                out["vw_bb_upper"] = _align(last["vwap"] + VWAP_BB_STDEV * std, buckets)
            if "vw_bb_lower" in vwap_keys:
                out["vw_bb_lower"] = _align(last["vwap"] - VWAP_BB_STDEV * std, buckets)
    return out


def _overlays_from_vwap_bars(
    bars: pd.DataFrame, buckets: pd.DatetimeIndex, keys: list[str],
    *, vwap_origin: str = "session",
) -> dict[str, list[float | None]]:
    """Session/origin VWAP (+ optional ±2σ volume-weighted Bollinger bands).

    σ matches the native 5m hive: ``sqrt(E[price²] − VWAP²)`` over the same
    origin window as VWAP (ADR-090). Bands are price levels
    ``VWAP ± VWAP_BB_STDEV·σ`` for the price pane.
    """
    if bars is None or bars.empty or not keys:
        return {k: [None] * len(buckets) for k in keys}
    df = bars.sort_values("ts_event").copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df["ts_event"], utc=True))
    origin_period = origin_period_key(idx, vwap_origin)
    cum_vol = df["size"].groupby(origin_period, sort=False).cumsum().replace(0, np.nan)
    df["vwap"] = df["pv"].groupby(origin_period, sort=False).cumsum() / cum_vol
    vwap = pd.Series(df["vwap"].to_numpy(), index=idx)
    close = pd.Series(df["close"].to_numpy(), index=idx)
    out: dict[str, list[float | None]] = {}
    if "vw_vwap" in keys:
        out["vw_vwap"] = _align(vwap, buckets)
    if "vw_vwap_dist" in keys:
        dist = (close - vwap) / vwap.replace(0, np.nan)
        out["vw_vwap_dist"] = _align(dist, buckets)
    want_bands = "vw_bb_upper" in keys or "vw_bb_lower" in keys
    if want_bands:
        if "p2v" in df.columns:
            p2v = pd.to_numeric(df["p2v"], errors="coerce")
        else:
            # Fallback when only pv/size are present (tests / older callers).
            p2v = pd.Series(np.nan, index=df.index, dtype="float64")
        cum_p2v = p2v.groupby(origin_period, sort=False).cumsum()
        # Keep arithmetic on row-aligned arrays (avoid DatetimeIndex outer-join).
        vwap_arr = vwap.to_numpy(dtype="float64")
        vol_arr = cum_vol.to_numpy(dtype="float64")
        p2v_arr = cum_p2v.to_numpy(dtype="float64")
        var = (p2v_arr / vol_arr) - (vwap_arr ** 2)
        std = pd.Series(
            np.sqrt(np.clip(var, 0.0, None)),
            index=idx,
        )
        if "vw_bb_upper" in keys:
            out["vw_bb_upper"] = _align(vwap + VWAP_BB_STDEV * std, buckets)
        if "vw_bb_lower" in keys:
            out["vw_bb_lower"] = _align(vwap - VWAP_BB_STDEV * std, buckets)
    return out


def meta() -> dict[str, Any]:
    from src.external_models.candles import candle_span
    span = candle_span(ROOT)
    lead = active_contract(ROOT, datetime.now(timezone.utc).astimezone(CT).date())
    return {
        "instrument": "/NQ",
        "root": ROOT,
        "contract": lead,
        "timeframes": list(TIMEFRAMES),
        "lookbacks": list(_LOOKBACK),
        "lookback_labels": dict(_LOOKBACK_LABELS),
        "lookback_custom": True,
        "lookback_range": True,
        "tf_max_lookback": {},
        "max_bars": MAX_BARS,
        "overlays": [dict(o) for o in OVERLAYS],
        "origins": list(ORIGINS),
        "svp_origins": list(SVP_ORIGINS),
        "origin_labels": dict(ORIGIN_LABELS),
        "session_tfs": list(SESSION_TFS),
        "session_tf_labels": dict(SESSION_TF_LABELS),
        "vp_kinds": list(VP_KINDS),
        "vp_kind_labels": dict(VP_KIND_LABELS),
        "vp_placements": list(VP_PLACEMENTS),
        "vp_placement_labels": dict(VP_PLACEMENT_LABELS),
        "vp_sides": list(VP_SIDES),
        "vp_recur": list(VP_RECUR),
        "vp_recur_labels": dict(VP_RECUR_LABELS),
        "vp_hist_styles": list(VP_HIST_STYLES),
        "vp_hist_style_labels": dict(VP_HIST_STYLE_LABELS),
        "group_ticks": list(GROUP_TICKS),
        "group_ticks_default": GROUP_TICKS_DEFAULT,
        "vp_display_modes": list(VP_DISPLAY_MODES),
        "vp_display_mode_labels": dict(VP_DISPLAY_MODE_LABELS),
        "vp_extend": list(VP_EXTEND),
        "vp_extend_labels": {str(k): v for k, v in VP_EXTEND_LABELS.items()},
        "level_blend_lookback_default": LEVEL_BLEND_LOOKBACK_DEFAULT,
        "level_blend_lookback_min": LEVEL_BLEND_LOOKBACK_MIN,
        "level_blend_lookback_max": LEVEL_BLEND_LOOKBACK_MAX,
        "level_blend_sigma_default": LEVEL_BLEND_SIGMA_DEFAULT,
        "level_blend_buffer_default": LEVEL_BLEND_BUFFER_DEFAULT,
        "lvn_pct_default": LVN_PCT_DEFAULT,
        "hvn_methods": list(HVN_METHODS),
        "hvn_method_default": HVN_METHOD_DEFAULT,
        "kde_defaults": {
            "bandwidth": KDE_BW_DEFAULT,
            "prominence": KDE_PROM_DEFAULT,
            "valley_depth": KDE_VALLEY_DEFAULT,
            "volume": KDE_VOL_DEFAULT,
            "separation": KDE_SEP_DEFAULT,
            "valley_width": KDE_VALLEY_WIDTH_DEFAULT,
            "max_zones": KDE_MAX_ZONES_DEFAULT,
        },
        "ema_presets": list(EMA_PRESETS),
        "ema_lookback_min": EMA_LOOKBACK_MIN,
        "ema_lookback_max": EMA_LOOKBACK_MAX,
        "ema_lookback_default": EMA_LOOKBACK_DEFAULT,
        "ema_max": MAX_EMA_SPECS,
        "stoch_presets": [{"k_period": k, "d_period": d} for k, d in STOCH_PRESETS],
        "stoch_k_min": STOCH_K_MIN,
        "stoch_k_max": STOCH_K_MAX,
        "stoch_d_min": STOCH_D_MIN,
        "stoch_d_max": STOCH_D_MAX,
        "stoch_k_default": STOCH_K_DEFAULT,
        "stoch_d_default": STOCH_D_DEFAULT,
        "stoch_max": MAX_STOCH_SPECS,
        "sar_defaults": {
            "start": SAR_START_DEFAULT,
            "increment": SAR_INCREMENT_DEFAULT,
            "maximum": SAR_MAXIMUM_DEFAULT,
        },
        "sar_bounds": {
            "start": {"min": SAR_AF_MIN, "max": SAR_AF_MAX, "step": 0.01},
            "increment": {"min": SAR_AF_MIN, "max": SAR_AF_MAX, "step": 0.01},
            "maximum": {"min": SAR_MAXIMUM_MIN, "max": SAR_MAXIMUM_MAX, "step": 0.01},
        },
        "span": span,
        "defaults": {
            "timeframe": "1min", "lookback": "session", "live": True,
            "cvd_origin": "session", "vwap_origin": "session", "ofi_origin": "session",
            "ad_origin": "session", "svp_origin": "session",
            "cvd_style": "line",
            "vwap_bands": False,
            "vwap_bb_stdev": VWAP_BB_STDEV,
            "eth_display": True, "eth_shade": False, "eth_aggregate": True,
            "text_size": 12,
            "print_min": PRINT_MIN_DEFAULT,
            "replay": False,
            "ema_lookback": EMA_LOOKBACK_DEFAULT,
            "stoch_k_period": STOCH_K_DEFAULT,
            "stoch_d_period": STOCH_D_DEFAULT,
            "sar": {
                "start": SAR_START_DEFAULT,
                "increment": SAR_INCREMENT_DEFAULT,
                "maximum": SAR_MAXIMUM_DEFAULT,
            },
            "rth": {"start": "08:30", "end": "15:00", "tz": "America/Chicago"},
        },
    }


def series(
    *,
    timeframe: str = "1min",
    lookback: str = "session",
    overlays: str | None = None,
    end: str | None = None,
    cvd_origin: str = "session",
    vwap_origin: str = "session",
    ofi_origin: str | None = None,
    ad_origin: str | None = None,
    svp_origin: str | None = None,
    vp_start: str | None = None,
    vp_end: str | None = None,
    profiles: str | None = None,
    emas: str | None = None,
    stochs: str | None = None,
    sar: str | None = None,
    vwap_bands: bool = False,
    eth_aggregate: bool = True,
    asof: str | None = None,
) -> dict[str, Any]:
    """OHLCV + selected overlays for /NQ on *timeframe*."""
    tf = coerce_tf(timeframe)
    eth_aggregate = parse_bool(eth_aggregate, default=True)
    vwap_bands = parse_bool(vwap_bands, default=False)
    rth_only = (not eth_aggregate) and tf != "1day"
    cvd_origin = coerce_origin(cvd_origin)
    vwap_origin = coerce_origin(vwap_origin)
    ofi_origin = coerce_origin(ofi_origin) if ofi_origin not in (None, "") else cvd_origin
    ad_origin = coerce_origin(ad_origin) if ad_origin not in (None, "") else cvd_origin
    svp_key = None if not (svp_origin and str(svp_origin).strip()) else coerce_origin(
        svp_origin, allowed=SVP_ORIGINS,
    )
    specs = parse_profiles(profiles)
    ema_specs = parse_emas(emas)
    stoch_specs = parse_stochs(stochs)
    sar_params = parse_sar_params(sar)
    end_dt = datetime.now(timezone.utc) if not end else datetime.fromisoformat(
        str(end).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    asof_dt = None
    if asof not in (None, ""):
        try:
            asof_dt = datetime.fromisoformat(str(asof).strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("asof must be an ISO datetime") from exc
        if asof_dt.tzinfo is None:
            asof_dt = asof_dt.replace(tzinfo=timezone.utc)
        asof_dt = asof_dt.astimezone(timezone.utc)
    token, _delta = parse_lookback(lookback)
    start, end_dt = window_for(lookback, timeframe=tf, end=end_dt)
    keys = [k for k in parse_overlay_keys(overlays) if _OVERLAY_BY_KEY[k].get("needs") not in ("prints", "mbo")]
    range_start = parse_vp_bound(vp_start, field="vp_start")
    range_stop = parse_vp_bound(vp_end, field="vp_end", date_is_end=True)
    ohlc = load_ohlc(tf, start, end_dt)
    bars_capped = False
    if len(ohlc) > MAX_BARS:
        ohlc = ohlc.iloc[-MAX_BARS:].copy()
        bars_capped = True
    # Compat: lookback_capped used to mean per-TF window shrink; now it
    # mirrors bars_capped (MAX_BARS truncates; newest retained).
    lookback_capped = bars_capped
    t = o = h = l = c = v = []
    contract = active_contract(ROOT, end_dt.astimezone(CT).date())
    if not ohlc.empty:
        t = [_unix(x) for x in ohlc["bucket"]]
        o = [_finite(x) for x in ohlc["open"]]
        h = [_finite(x) for x in ohlc["high"]]
        l = [_finite(x) for x in ohlc["low"]]
        c = [_finite(x) for x in ohlc["close"]]
        v = [_finite(x) for x in ohlc["volume"]]
        if "symbol" in ohlc.columns and len(ohlc):
            contract = str(ohlc["symbol"].iloc[-1])
    ov: dict[str, list[float | None]] = {}
    cvd_ohlc_df: pd.DataFrame | None = None
    tick_keys = [k for k in keys if _OVERLAY_BY_KEY[k]["needs"] == "ticks"]
    ohlc_keys = [k for k in keys if _OVERLAY_BY_KEY[k]["needs"] == "ohlc"]
    flow_keys = [k for k in keys if _OVERLAY_BY_KEY[k]["needs"] == "flow"]
    # Lookback overlays (EMA/MFI/stoch/SAR) always see calendar-adjacent bars,
    # including overnight and bars before the visible window (ADR-137).
    # Aggregate still filters origin-reset flow and VP tape.
    lookback_keys = [k for k in ohlc_keys if k in _LOOKBACK_OVERLAY_BARS]
    other_ohlc_keys = [k for k in ohlc_keys if k not in _LOOKBACK_OVERLAY_BARS]
    warm_n = overlay_warmup_bars(lookback_keys, ema_specs, stoch_specs)
    ohlc_cont = ohlc
    if warm_n > 0 and ohlc is not None and not ohlc.empty:
        prior = load_ohlc(tf, warmup_start(start, tf, warm_n), start)
        ohlc_cont = prepend_ohlc_warmup(ohlc, prior)
    ohlc_calc = filter_rth_ohlc(ohlc) if rth_only else ohlc
    full_buckets = (
        pd.DatetimeIndex(pd.to_datetime(ohlc["bucket"], utc=True))
        if not ohlc.empty else pd.DatetimeIndex([], tz="UTC")
    )
    calc_buckets = (
        pd.DatetimeIndex(pd.to_datetime(ohlc_calc["bucket"], utc=True))
        if ohlc_calc is not None and not ohlc_calc.empty else pd.DatetimeIndex([], tz="UTC")
    )

    def _to_full(values) -> list[float | None]:
        if values is None:
            return [None] * len(full_buckets)
        if isinstance(values, pd.Series):
            return _align(values, full_buckets)
        arr = list(values)
        if not rth_only or len(arr) == len(full_buckets):
            if len(arr) < len(full_buckets):
                arr = arr + [None] * (len(full_buckets) - len(arr))
            return arr[:len(full_buckets)]
        s = pd.Series(arr, index=calc_buckets)
        return _align(s, full_buckets)

    cont_buckets = (
        pd.DatetimeIndex(pd.to_datetime(ohlc_cont["bucket"], utc=True))
        if ohlc_cont is not None and not ohlc_cont.empty else pd.DatetimeIndex([], tz="UTC")
    )

    def _from_cont(values) -> list[float | None]:
        if values is None:
            return [None] * len(full_buckets)
        if isinstance(values, pd.Series):
            return _align(values, full_buckets)
        s = pd.Series(list(values), index=cont_buckets[:len(list(values))])
        return _align(s, full_buckets)

    if other_ohlc_keys and ohlc_calc is not None and not ohlc_calc.empty:
        raw = _overlays_from_ohlc(ohlc_calc, other_ohlc_keys, sar_params=sar_params)
        for k, vals in raw.items():
            ov[k] = _to_full(vals)
    if lookback_keys and ohlc_cont is not None and not ohlc_cont.empty:
        raw = _overlays_from_ohlc(ohlc_cont, lookback_keys, sar_params=sar_params)
        for k, vals in raw.items():
            ov[k] = _from_cont(vals)
    ema_meta: list[dict[str, Any]] = []
    if ema_specs and ohlc_cont is not None and not ohlc_cont.empty:
        close = pd.to_numeric(ohlc_cont["close"], errors="coerce")
        for spec in ema_specs:
            series_ema = ema_on_close(close, int(spec["lookback"]))
            series_ema.index = cont_buckets
            key = str(spec["key"])
            ov[key] = _from_cont(series_ema)
            ema_meta.append({
                "id": spec["id"],
                "n": spec["n"],
                "lookback": spec["lookback"],
                "key": key,
                "label": spec["label"],
            })
    elif ema_specs:
        for spec in ema_specs:
            key = str(spec["key"])
            ov[key] = [None] * len(t)
            ema_meta.append({
                "id": spec["id"],
                "n": spec["n"],
                "lookback": spec["lookback"],
                "key": key,
                "label": spec["label"],
            })
    stoch_meta: list[dict[str, Any]] = []
    if stoch_specs and ohlc_cont is not None and not ohlc_cont.empty:
        from src.features.stochastic import _fast_d
        high = pd.to_numeric(ohlc_cont["high"], errors="coerce")
        low = pd.to_numeric(ohlc_cont["low"], errors="coerce")
        close_s = pd.to_numeric(ohlc_cont["close"], errors="coerce")
        high.index = cont_buckets
        low.index = cont_buckets
        close_s.index = cont_buckets
        for spec in stoch_specs:
            series_st = _fast_d(
                high, low, close_s,
                int(spec["k_period"]), int(spec["d_period"]),
            )
            key = str(spec["key"])
            ov[key] = _from_cont(series_st)
            stoch_meta.append({
                "id": spec["id"],
                "n": spec["n"],
                "k_period": spec["k_period"],
                "d_period": spec["d_period"],
                "key": key,
                "label": spec["label"],
            })
    elif stoch_specs:
        for spec in stoch_specs:
            key = str(spec["key"])
            ov[key] = [None] * len(t)
            stoch_meta.append({
                "id": spec["id"],
                "n": spec["n"],
                "k_period": spec["k_period"],
                "d_period": spec["d_period"],
                "key": key,
                "label": spec["label"],
            })
    if flow_keys and ohlc_calc is not None and not ohlc_calc.empty:
        delta = _bar_delta(ohlc_calc)
        delta.index = calc_buckets
        if "of_delta" in flow_keys:
            ov["of_delta"] = _to_full(delta)
        if "of_cvd" in flow_keys:
            first = calc_buckets[0].to_pydatetime()
            warm_from = origin_start(first, cvd_origin)
            warmup = None
            fine_cvd = None
            if tf not in ("5s", "15s", "1min"):
                fine_bars = _load_cagg(_CAGG["1min"], warm_from, end_dt)
                if rth_only and fine_bars is not None and not fine_bars.empty:
                    fine_bars = filter_rth_ohlc(fine_bars)
                if fine_bars is not None and not fine_bars.empty:
                    fine_cvd = cvd_on_bars(fine_bars, cvd_origin)
                    warmup = fine_bars.loc[
                        pd.to_datetime(fine_bars["bucket"], utc=True) < calc_buckets[0]
                    ]
            elif warm_from < first:
                warmup = _load_cagg(_CAGG["1min"], warm_from, first)
                if rth_only and warmup is not None and not warmup.empty:
                    warmup = filter_rth_ohlc(warmup)
            cvd = cvd_on_bars(ohlc_calc, cvd_origin, warmup=warmup)
            ov["of_cvd"] = _to_full(cvd)
            if fine_cvd is not None and not fine_cvd.empty:
                cvd_ohlc_df = cvd_ohlc_from_fine(fine_cvd, calc_buckets, cvd_origin)
            else:
                cvd_ohlc_df = cvd_ohlc_from_closes(cvd, cvd_origin)
            if rth_only and cvd_ohlc_df is not None and not cvd_ohlc_df.empty:
                cvd_ohlc_df = cvd_ohlc_df.reindex(full_buckets)
        if "of_ofi_cum" in flow_keys:
            first = calc_buckets[0].to_pydatetime()
            warm_from = origin_start(first, ofi_origin)
            ofi = _bar_ofi(ohlc_calc)
            ofi = ofi.reindex(calc_buckets).fillna(0.0)
            warmup_ofi = None
            if warm_from < first:
                warm_bars = _load_cagg(_CAGG["1min"], warm_from, first)
                if rth_only and warm_bars is not None and not warm_bars.empty:
                    warm_bars = filter_rth_ohlc(warm_bars)
                warmup_ofi = _bar_ofi(warm_bars)
            ofi_cum = origin_cumsum(ofi, ofi_origin, warmup_ofi)
            ov["of_ofi_cum"] = _to_full(ofi_cum.reindex(calc_buckets))
    if "ad_line" in keys and ohlc_calc is not None and not ohlc_calc.empty:
        inc = clv_volume(ohlc_calc)
        first = calc_buckets[0].to_pydatetime()
        warm_from = origin_start(first, ad_origin)
        warmup_ad = None
        if warm_from < first:
            warm_bars = _load_cagg(_CAGG["1min"], warm_from, first)
            if rth_only and warm_bars is not None and not warm_bars.empty:
                warm_bars = filter_rth_ohlc(warm_bars)
            if warm_bars is not None and not warm_bars.empty:
                warmup_ad = clv_volume(warm_bars)
        ad = origin_cumsum(inc, ad_origin, warmup_ad)
        ov["ad_line"] = _to_full(ad.reindex(calc_buckets))
    tick_keys = [k for k in tick_keys if k not in _VWAP_SERIES_KEYS]
    if tick_keys:
        tick_start = start
        if (end_dt - start).total_seconds() > MAX_TICK_SECONDS:
            tick_start = end_dt - timedelta(seconds=MAX_TICK_SECONDS)
        ticks = load_ticks(tick_start, end_dt)
        if rth_only:
            ticks = filter_rth_ticks(ticks)
        raw = _overlays_from_ticks(
            ticks, calc_buckets if rth_only else full_buckets, pandas_freq(tf), tick_keys,
            vwap_origin=vwap_origin,
        )
        for k, vals in raw.items():
            ov[k] = _to_full(vals) if rth_only else vals
    vwap_keys = [k for k in keys if k in ("vw_vwap", "vw_vwap_dist")]
    if vwap_bands:
        for bk in VWAP_BB_KEYS:
            if bk not in vwap_keys:
                vwap_keys.append(bk)
    if vwap_keys:
        vwap_bars = load_vwap_bars(start, end_dt, tf)
        if rth_only:
            vwap_bars = filter_rth_ticks(vwap_bars)
        raw = _overlays_from_vwap_bars(
            vwap_bars, calc_buckets if rth_only else full_buckets, vwap_keys,
            vwap_origin=vwap_origin,
        )
        for k, vals in raw.items():
            ov[k] = _to_full(vals) if rth_only else vals
    for k in keys:
        if k not in ov:
            ov[k] = [None] * len(t)
    now = datetime.now(timezone.utc)
    profile_end = asof_dt if asof_dt is not None else end_dt
    if asof_dt is not None:
        if asof_dt > end_dt:
            asof_dt = end_dt
            profile_end = end_dt
        if asof_dt < start:
            asof_dt = start
            profile_end = start
    ohlc_asof = ohlc
    if asof_dt is not None and ohlc is not None and not ohlc.empty and "bucket" in ohlc.columns:
        buck = pd.to_datetime(ohlc["bucket"], utc=True)
        ohlc_asof = ohlc.loc[buck <= pd.Timestamp(asof_dt)].copy()
    profile_list: list[dict[str, Any]] = []
    svp = None
    vp = None
    if specs:
        jobs = expand_vp_jobs(
            specs, ohlc=ohlc_asof, chart_start=start, chart_end=profile_end, now=profile_end,
            rth_only=rth_only,
        )
        profile_list = profiles_payloads(jobs, ohlc=ohlc_asof, rth_only=rth_only, timeframe=tf)
        for p in profile_list:
            if svp is None and p.get("kind") == "current":
                svp = p
        if svp is None and profile_list:
            svp = profile_list[0]
        if vp is None:
            vp = svp
    else:
        if svp_key is not None:
            svp_from, svp_to = profile_window(ohlc_asof, profile_end, svp_key)
            vbp = load_vbp(svp_from, svp_to, rth_only=rth_only)
            if vbp is None or vbp.empty:
                prev_from, prev_to = profile_window(
                    None, svp_from - timedelta(seconds=1), svp_key,
                )
                vbp = load_vbp(prev_from, prev_to, rth_only=rth_only)
                svp_from, svp_to = prev_from, prev_to
            svp = profile_payload(
                vbp if vbp is not None else pd.DataFrame(),
                start=svp_from, end=svp_to, origin=svp_key,
            )
            svp.update({
                "id": "legacy-svp", "spec_id": "legacy-svp", "kind": "current",
                "session_tf": "full_session" if svp_key == "session" else (
                    "cash" if svp_key == "mkt" else svp_key
                ),
                "n": 1, "label": "VP1",
                "placement": "left_frame", "overlay_side": "left",
                "extend": 1, "extend_lines": True, "extend_va": True,
                "hist_style": "volume", "group_ticks": GROUP_TICKS_DEFAULT,
                "right_display": "total_split", "left_display": "none",
                "show_lvn": False, "lvn_pct": LVN_PCT_DEFAULT, "extend_lvn": False,
                "recur": None,
                "session_from": None, "session_to": None, "session_date": None,
            })
            profile_list.append(svp)
        if range_stop is not None and range_start is None:
            raise ValueError("vp_start is required when vp_end is set")
        if range_start is not None:
            vp_to = range_stop if range_stop is not None else profile_end
            if vp_to > profile_end:
                vp_to = profile_end
            if vp_to <= range_start:
                raise ValueError("To (CT) must be after From (CT)")
            chi_s = range_start.astimezone(CT)
            chi_e = vp_to.astimezone(CT)
            legacy_spec = {
                "id": "legacy-vp", "kind": "current",
                "session_tf": "custom",
                "session_from": (chi_s.hour, chi_s.minute),
                "session_to": (chi_e.hour, chi_e.minute),
                "session_date": chi_e.date(),
                "n": 1, "label": "VP1",
                "placement": "right_frame", "overlay_side": "right",
                "extend": 1, "extend_lines": True, "extend_va": True,
                "hist_style": "volume", "group_ticks": GROUP_TICKS_DEFAULT,
                "right_display": "total_split", "left_display": "none",
                "show_lvn": False, "lvn_pct": LVN_PCT_DEFAULT, "extend_lvn": False,
                "recur": None,
            }
            jobs = expand_vp_jobs(
                [legacy_spec], ohlc=ohlc_asof, chart_start=start,
                chart_end=profile_end, now=profile_end, rth_only=rth_only,
            )
            extra = profiles_payloads(jobs, ohlc=ohlc_asof, rth_only=rth_only, timeframe=tf)
            if extra:
                vp = extra[0]
                profile_list.append(vp)
    return {
        "instrument": "/NQ",
        "contract": contract,
        "timeframe": tf,
        "lookback": token,
        "lookback_capped": lookback_capped,
        "bars_capped": bars_capped,
        "max_bars": MAX_BARS,
        "cvd_origin": cvd_origin,
        "vwap_origin": vwap_origin,
        "ofi_origin": ofi_origin,
        "ad_origin": ad_origin,
        "vwap_bands": bool(vwap_bands),
        "vwap_bb_stdev": VWAP_BB_STDEV if vwap_bands else None,
        "eth_aggregate": bool(eth_aggregate),
        "asof": asof_dt.isoformat().replace("+00:00", "Z") if asof_dt is not None else None,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end_dt.isoformat().replace("+00:00", "Z"),
        "n": len(t),
        "t": t,
        "o": o, "h": h, "l": l, "c": c, "v": v,
        "overlays": ov,
        "cvd_ohlc": _pack_cvd_ohlc(cvd_ohlc_df, len(t)) if cvd_ohlc_df is not None else None,
        "svp": svp,
        "vp": vp,
        "profiles": profile_list,
        "emas": ema_meta,
        "stochs": stoch_meta,
        "sar": dict(sar_params) if "ps_sar" in keys else None,
    }
