"""Load lead-month NQ OHLC from PostgreSQL ``databento.tbbo_1min``."""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg

from src.common.contract_calendar import active_contract, active_contracts_in_range
from src.common.db import get_dsn
from src.external_models.timeframe import coerce_candle_tf, resample_ohlc

logger = logging.getLogger("vtech.external_models.candles")

CANDLE_TABLE = "databento.tbbo_1min"
DEFAULT_ROOT = "NQ"
_CT = ZoneInfo("America/Chicago")
_CONNECT_TIMEOUT_S = 5
_SPAN_TTL_S = 60.0
_SPAN_LOCK = threading.Lock()
_SPAN_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _query(sql: str, params: tuple, *, timeout_ms: int = 60_000) -> pd.DataFrame:
    # libpq ``options`` is session-level. ``set_config(..., local=true)`` is a
    # no-op under autocommit (each execute is its own transaction), which is
    # why the External Models tab used to hang past the intended 15s cap.
    with psycopg.connect(
        get_dsn(),
        autocommit=True,
        connect_timeout=_CONNECT_TIMEOUT_S,
        options=f"-c statement_timeout={int(timeout_ms)}",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def clear_span_cache() -> None:
    with _SPAN_LOCK:
        _SPAN_CACHE.clear()


def lead_symbols(root: str, start: date, end: date) -> list[str]:
    pairs = active_contracts_in_range(root, start, end)
    seen: list[str] = []
    for _, sym in pairs:
        if sym not in seen:
            seen.append(sym)
    return seen


def candle_span(root: str = DEFAULT_ROOT) -> dict[str, Any]:
    """Min/max ``tbbo_1min`` coverage for current-ish NQ lead symbols.

    Not used by the External Models tab catalog. Callers: Backtest/chart when
    the operator omitted dates. Per-symbol ``ORDER BY bucket LIMIT 1`` index
    probes (no ``COUNT(*)``) — a full cagg aggregate was a multi-second scan
    of every matching chunk and could stall the 20s browser abort.
    """
    now = time.monotonic()
    with _SPAN_LOCK:
        cached = _SPAN_CACHE.get(root)
        if cached and now - cached[0] < _SPAN_TTL_S:
            return dict(cached[1])

    today = datetime.now(timezone.utc).astimezone(_CT).date()
    start = date(today.year - 4, 1, 1)
    symbols = lead_symbols(root, start, today)
    miss: dict[str, Any] = {
        "available": False, "error": "no lead-month symbols", "start": None,
        "end": None, "n_rows": None, "table": CANDLE_TABLE,
    }
    if not symbols:
        return miss
    try:
        # Skip-scan: 2 index lookups per quarterly symbol, not a COUNT of
        # ~600k 1m rows across 50+ chunks.
        df = _query(
            f"SELECT MIN(lo) AS lo, MAX(hi) AS hi FROM ("
            f" SELECT"
            f"  (SELECT bucket FROM {CANDLE_TABLE} WHERE symbol = s"
            f"   ORDER BY bucket ASC LIMIT 1) AS lo,"
            f"  (SELECT bucket FROM {CANDLE_TABLE} WHERE symbol = s"
            f"   ORDER BY bucket DESC LIMIT 1) AS hi"
            f" FROM unnest(%s::text[]) AS s"
            f") t",
            (symbols,),
            timeout_ms=8_000,
        )
    except Exception as exc:  # noqa: BLE001 — surface DB errors to the UI
        logger.warning("tbbo_1min span query failed: %s", exc)
        miss["error"] = str(exc)
        return miss
    if df.empty or df.iloc[0]["lo"] is None:
        miss["error"] = "no rows in tbbo_1min for lead NQ"
        return miss
    lo = pd.Timestamp(df.iloc[0]["lo"])
    hi = pd.Timestamp(df.iloc[0]["hi"])
    if lo.tzinfo is None:
        lo = lo.tz_localize("UTC")
    else:
        lo = lo.tz_convert("UTC")
    if hi.tzinfo is None:
        hi = hi.tz_localize("UTC")
    else:
        hi = hi.tz_convert("UTC")
    out = {
        "available": True,
        "error": None,
        "start": lo.tz_convert(_CT).date().isoformat(),
        "end": hi.tz_convert(_CT).date().isoformat(),
        "n_rows": None,
        "table": CANDLE_TABLE,
        "symbols": symbols,
    }
    with _SPAN_LOCK:
        _SPAN_CACHE[root] = (time.monotonic(), out)
    return dict(out)


def load_lead_ohlc(
    start: str | date,
    end: str | date,
    *,
    root: str = DEFAULT_ROOT,
    timeframe: str = "1min",
    warmup_days: int = 2,
    source_table: str | None = None,
) -> pd.DataFrame:
    """Lead-month OHLC indexed by UTC bucket (left-labeled 1-minute by default).

    Rows whose Chicago date is a weekend are kept (Globex) but callers
    typically mask by session. Contract rolls stitch the lead symbol per
    Chicago date; a roll does **not** back-adjust prices.

    ``source_table`` overrides ``databento.tbbo_1min`` — pass
    ``databento.tbbo_15s`` for a finer intra-minute path used by engines
    that need accurate stop / target / session_flat fills. When a finer
    source is used the returned frame retains its native bucket cadence
    unless ``timeframe`` explicitly requests a coarser resample.
    """
    d0 = _as_date(start)
    d1 = _as_date(end)
    if d1 < d0:
        raise ValueError("end date is before start date")
    load0 = d0 - timedelta(days=max(0, int(warmup_days)))
    load1 = d1 + timedelta(days=1)  # exclusive
    symbols = lead_symbols(root, load0, d1)
    t0 = datetime(load0.year, load0.month, load0.day, tzinfo=_CT).astimezone(timezone.utc)
    t1 = datetime(load1.year, load1.month, load1.day, tzinfo=_CT).astimezone(timezone.utc)
    table = source_table or CANDLE_TABLE
    df = _query(
        f"SELECT bucket, symbol, open, high, low, close, volume "
        f"FROM {table} "
        f"WHERE bucket >= %s AND bucket < %s AND symbol = ANY(%s) "
        f"ORDER BY bucket ASC, symbol ASC",
        (t0, t1, symbols),
        timeout_ms=120_000,
    )
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "symbol"])

    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    local = df["bucket"].dt.tz_convert(_CT)
    chicago_day = np.array([d.date() for d in local], dtype=object)
    lead = {d: sym for d, sym in active_contracts_in_range(root, load0, d1)}
    keep = np.array([lead.get(d) == s for d, s in zip(chicago_day, df["symbol"].astype(str))],
                    dtype=bool)
    df = df.loc[keep].copy()
    df = df.drop_duplicates(subset=["bucket"], keep="last")
    df = df.set_index("bucket").sort_index()
    df = df[["open", "high", "low", "close", "volume", "symbol"]]
    df = df.dropna(subset=["high", "low", "close"])
    tf = coerce_candle_tf(timeframe)
    # Only resample on the 1-min source path; finer sources return native cadence.
    if source_table is None and tf != "1min":
        df = resample_ohlc(df, tf)
    return df
