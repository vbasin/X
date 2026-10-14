"""Real-time Databento live streaming service.

Subscribes to GLBX.MDP3 for NQ/ES futures (parent) and the five session
option chains (0DTE, 1DTE, QNE1, QNE2, front quarterly) as raw_symbol,
routes records to TimescaleDB via callbacks, and archives raw DBN to disk.

Also maintains the 0DTE Greeks dual-write path (same compute as the historical
backfill):
  * ``databento.nq_option_greeks_1s`` — every two-sided 0DTE option-second
  * ``databento.nq_option_surface_1s`` + today's surface parquet partition

Run as: python -m src.acquisition.live_streamer
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
import calendar
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import pandas as pd
from databento_dbn import FIXED_PRICE_SCALE, UNDEF_PRICE, ErrorCode, SystemCode

from src.acquisition.options_universe import (
    live_option_session_date,
    resolve_session_option_universe,
)
from src.acquisition.definition_acquisition import (
    DefinitionAcquirer, VendorDefinition, DEFINITION_UPSERT_TAIL,
)
from src.acquisition.schemas import DATASET
from src.common.db import close_pool, get_pool
from src.acquisition.rotating_stream import RotatingDbnStream
from src.common.data_paths import options_surface_dir, raw_dir
from src.common.market_session import cme_weekend_closed

logger = logging.getLogger("vtech.live")

_PX_SCALE = float(FIXED_PRICE_SCALE)
_CT = ZoneInfo("America/Chicago")
_NQ_FUT_RE = re.compile(r"^NQ[FGHJKMNQUVXZ][0-9]$")
# DBN encodes an absent price as UNDEF_PRICE (2^63-1 nanos); it is positive and so
# survives a ``px > 0`` filter. Every price read here must bound it from above.
_UNDEF_PX = 9.0e8
_GREEKS_ENABLED = os.environ.get("VTECH_LIVE_GREEKS", "1").strip() != "0"
_BBO_INSERT_COLS_SQL = "(ts_recv, instrument_id, symbol, price, size, bid_px, ask_px, bid_sz, ask_sz)"
_BBO_INSERT_PARAMS = 9
_BBO_INSERT_MAX_ROWS = 7000
_TBBO_INSERT_COLS_SQL = (
    "(ts_event, ts_recv, instrument_id, symbol, price, size, side, action, "
    "flags, sequence, bid_px, ask_px, bid_sz, ask_sz, bid_ct, ask_ct, ts_in_delta)"
)
_TBBO_INSERT_PARAMS = 17
# order matches ingest_mbo.py's CSV_COLS (batch/historical path) minus `dataset`,
# which the table defaults to 'GLBX.MDP3' — keep both paths writing the same shape.
_MBO_INSERT_COLS_SQL = (
    "(ts_recv, ts_event, rtype, publisher_id, instrument_id, action, side, "
    "price, size, channel_id, order_id, flags, ts_in_delta, sequence, symbol)"
)
_DEF_INSERT_COLS_SQL = (
    "(ts_event, instrument_id, symbol, instrument_class, strike_price, "
    "expiration, underlying, exchange, currency, min_price_increment, multiplier, "
    "source)"
)
_DEF_INSERT_PARAMS = 12
# Rows reconstructed from a quote symbol carry source='synthetic' so an
# authoritative vendor definition can later correct them (see _DEF_UPSERT_TAIL).
_DEF_SOURCE_SYNTHETIC = "synthetic"
_DEF_SOURCE_VENDOR = "databento"
# Vendor rows overwrite synthetic ones; nothing overwrites a vendor row.
#
# Exception: CME's definition record carries NO usable contract terms for these
# options -- `multiplier` is NULL and `min_price_increment` is NaN or a leaked
# DBN UNDEF_PRICE sentinel, on 100% of vendor option rows. Overwriting blindly
# would replace correct reconstructed values (20, 0.25) with garbage, so those
# two columns keep whatever usable value is already stored.
_DEF_UPSERT_TAIL = DEFINITION_UPSERT_TAIL
# NQ weekly/daily options are PM-settled at 15:00 CT.
_WEEKLY_OPTION_SETTLE_HOUR_CT = 15
# Sentinel "nth" meaning end-of-month rather than an ordinal weekday (QNE chains).
_EOM_NTH = -1
_WEEKLY_MONTH_CODE_TO_MONTH = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
_WEEKLY_WEEKDAY_CODE_TO_DAY = {
    "A": calendar.MONDAY,
    "B": calendar.TUESDAY,
    "C": calendar.WEDNESDAY,
    "D": calendar.THURSDAY,
}
# QNE is the end-of-month chain and must be listed before QN[1-4] would otherwise
# be tried; it expires on the last business day of the month rather than on an
# nth weekday. Omitting it silently blacked out the live Greeks/surface writes on
# every month-end expiry, because _register_resolved_weekly_from_bbo returns early
# when the symbol does not parse and the instrument then never enters _0dte_ids.
_WEEKLY_RESOLVED_SYMBOL_RE = re.compile(
    r"^(?P<root>Q[1-5][A-D]|QNE|QN[1-4])(?P<month>[FGHJKMNQUVXZ])(?P<year>\d)\s+(?P<class>[CP])(?P<strike>\d+)$"
)


def _last_business_day_of_month(year: int, month: int) -> date | None:
    """Last Mon-Fri of the month — the CME end-of-month (QNE) option expiry.

    Exchange holidays are not modelled here, matching _nth_weekday_of_month.
    """
    try:
        last = date(year, month, calendar.monthrange(year, month)[1])
    except ValueError:
        return None
    while last.weekday() >= calendar.SATURDAY:
        last -= timedelta(days=1)
    return last


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> date | None:
    weeks = calendar.monthcalendar(year, month)
    days = [week[weekday] for week in weeks if week[weekday] != 0]
    if nth <= 0 or nth > len(days):
        return None
    return date(year, month, days[nth - 1])


# NQ futures list quarterly months only (Mar/Jun/Sep/Dec); each expires the 3rd
# Friday of its month. Serial and weekly options therefore settle into the first
# quarterly contract that has NOT yet expired on the option's own expiry date --
# an Oct weekly settles into Z, and a Sep-25 daily settles into Z, not U.
_QUARTERLY_MONTH_TO_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}


# The quarterly future stops trading at its 08:30 CT final-settlement fixing on
# the 3rd Friday. An option expiring 15:00 CT that same Friday therefore OUTLIVES
# it and settles into the next quarterly -- comparing dates alone gets this wrong
# on exactly four days a year.
_QUARTERLY_FUTURE_SETTLE_HOUR_CT = 8
_QUARTERLY_FUTURE_SETTLE_MINUTE_CT = 30


def _quarterly_future_expiry(year: int, month: int) -> datetime | None:
    """Final-settlement instant of the NQ quarterly future for ``year``/``month``."""
    day = _nth_weekday_of_month(year, month, calendar.FRIDAY, 3)
    if day is None:
        return None
    return datetime(
        day.year,
        day.month,
        day.day,
        _QUARTERLY_FUTURE_SETTLE_HOUR_CT,
        _QUARTERLY_FUTURE_SETTLE_MINUTE_CT,
        tzinfo=_CT,
    ).astimezone(timezone.utc)


def _underlying_future_for_expiry(expiration: datetime, *, root: str = "NQ") -> str | None:
    """Resolve the futures contract an option expiring at ``expiration`` settles into.

    Returns the first quarterly contract still trading at ``expiration`` -- i.e.
    whose own final settlement is at or after it. ``None`` if unresolvable.
    """
    if expiration.tzinfo is None:
        expiration = expiration.replace(tzinfo=timezone.utc)
    for year in (expiration.year, expiration.year + 1):
        for month in sorted(_QUARTERLY_MONTH_TO_CODE):
            fut_expiry = _quarterly_future_expiry(year, month)
            if fut_expiry is not None and fut_expiry >= expiration:
                return f"{root}{_QUARTERLY_MONTH_TO_CODE[month]}{year % 10}"
    return None


def _resolve_year_digit(
    *,
    year_digit: int,
    month: int,
    weekday: int,
    nth: int,
    reference_date: date,
) -> date | None:
    ref_decade = reference_date.year // 10
    best_expiry: date | None = None
    best_rank: tuple[int, int] | None = None
    for decade in range(ref_decade - 1, ref_decade + 2):
        year = (decade * 10) + year_digit
        expiry = (
            _last_business_day_of_month(year, month)
            if nth == _EOM_NTH
            else _nth_weekday_of_month(year, month, weekday, nth)
        )
        if expiry is None:
            continue
        delta_days = (expiry - reference_date).days
        rank = (abs(delta_days), 0 if delta_days >= 0 else 1)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_expiry = expiry
    return best_expiry


def _parse_resolved_weekly_option_symbol(
    symbol: str,
    *,
    reference: date | datetime | None = None,
) -> dict | None:
    match = _WEEKLY_RESOLVED_SYMBOL_RE.match(symbol.strip())
    if match is None:
        return None
    root = match.group("root")
    month_code = match.group("month")
    month = _WEEKLY_MONTH_CODE_TO_MONTH.get(month_code)
    if month is None:
        return None
    year_digit = int(match.group("year"))
    option_class = match.group("class")
    strike = float(match.group("strike"))
    if root == "QNE":
        nth = _EOM_NTH            # sentinel: resolve to the last business day
        weekday = calendar.FRIDAY
    elif root.startswith("QN"):
        nth = int(root[2])
        weekday = calendar.FRIDAY
    else:
        nth = int(root[1])
        weekday = _WEEKLY_WEEKDAY_CODE_TO_DAY[root[2]]

    if reference is None:
        reference_date = datetime.now(timezone.utc).date()
    elif isinstance(reference, datetime):
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        reference_date = reference.astimezone(timezone.utc).date()
    else:
        reference_date = reference

    expiry_date = _resolve_year_digit(
        year_digit=year_digit,
        month=month,
        weekday=weekday,
        nth=nth,
        reference_date=reference_date,
    )
    if expiry_date is None:
        return None

    # NQ weekly/daily options are PM-settled at 15:00 America/Chicago. Localize
    # rather than hardcoding a UTC hour: the UTC offset is -5 under CDT and -6
    # under CST, so a fixed UTC hour is wrong for half the year.
    expiration = datetime(
        expiry_date.year,
        expiry_date.month,
        expiry_date.day,
        _WEEKLY_OPTION_SETTLE_HOUR_CT,
        0,
        0,
        tzinfo=_CT,
    ).astimezone(timezone.utc)

    # The month code in the symbol is the OPTION's contract month, not the
    # underlying future's -- deriving `underlying` from it mislabels every
    # serial/post-roll expiry and invents contracts that do not exist (NQV6).
    underlying = _underlying_future_for_expiry(expiration)
    if underlying is None:
        return None

    return {
        "instrument_class": option_class,
        "strike_price": strike,
        "expiration": expiration,
        "expiry": expiry_date,
        "underlying": underlying,
        "exchange": "XCME",
        "currency": "USD",
        "min_price_increment": 0.25,
        "multiplier": 20.0,
    }


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid_env_int", extra={"name": name, "value": raw, "default": default})
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid_env_float", extra={"name": name, "value": raw, "default": default})
        return default
    return max(minimum, value)


def subscribe_live_feeds(
    client: db.Live,
    option_symbols: Sequence[str] | None = None,
) -> None:
    """Register futures (parent) plus the session's five option chains (raw_symbol).

    Option BBO/definitions are *only* the resolved 0DTE / 1DTE / QNE1 / QNE2 /
    front-quarterly strikes. An empty ``option_symbols`` list is fail-closed:
    futures still stream, extra option parents do not.
    """
    client.subscribe(dataset=DATASET, schema="tbbo",
                     symbols="NQ.FUT", stype_in="parent")
    client.subscribe(dataset=DATASET, schema="bbo-1s",
                     symbols="NQ.FUT", stype_in="parent")
    # L3 full order book — DISABLED 2026-09-29: the live gateway rejects this
    # subscription ("Not authorized for mbo schema") and, worse, that rejection
    # tears down the *entire* client connection (tbbo/bbo-1s/statistics/options
    # all share it), which crash-looped the whole live streamer within a minute
    # of deploy. Live MBO needs a separate real-time entitlement from Databento
    # that this account does not currently have — historical MBO access (which
    # is what backfill_mbo.py uses) does not imply live access. Re-enable only
    # after confirming entitlement with Databento; see ADR-146.
    # client.subscribe(dataset=DATASET, schema="mbo",
    #                  symbols="NQ.FUT", stype_in="parent")
    symbols = list(option_symbols or ())
    if symbols:
        client.subscribe(dataset=DATASET, schema="definition",
                         symbols=symbols, stype_in="raw_symbol")
        client.subscribe(dataset=DATASET, schema="bbo-1s",
                         symbols=symbols, stype_in="raw_symbol")
    client.subscribe(dataset=DATASET, schema="statistics",
                     symbols="NQ.FUT", stype_in="parent")
    # ES futures trades feed tbbo_1min → cross_asset (ca_*) features.
    client.subscribe(dataset=DATASET, schema="tbbo",
                     symbols="ES.FUT", stype_in="parent")


def _ns_to_dt(ns: int) -> datetime | None:
    """Convert nanosecond unix timestamp to timezone-aware datetime."""
    if ns is None or ns == 0:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


def _px(raw: int | float) -> float | None:
    """Convert Databento fixed-point price to float."""
    if raw is None:
        return None
    return float(raw) / _PX_SCALE


def _mbo_px(raw: int | float) -> float | None:
    """Like _px, but MBO price fields use UNDEF_PRICE (2**63-1) for 'no price'
    (e.g. Clear/None-side records) — same sentinel ingest_mbo.py's CSV path
    maps to SQL NULL."""
    if raw is None or int(raw) >= UNDEF_PRICE:
        return None
    return float(raw) / _PX_SCALE


def _instrument_class(raw: object) -> str | None:
    """Normalize Databento's InstrumentClass wrapper for SQL/classification."""
    if raw is None:
        return None
    value = getattr(raw, "value", raw)
    return str(value)


def _enum_int(raw: object) -> int | None:
    """Normalize Databento integer enum wrappers for SQL columns."""
    if raw is None:
        return None
    return int(getattr(raw, "value", raw))


class _BufferedTableWriter:
    """Background batched Postgres writer for one live-streamed schema.

    Generalizes the resilience shape originally built for bbo-1s (see git history
    on this file) so tbbo and mbo get the same treatment: the record-callback
    thread only calls ``enqueue`` (a queue.put, near-zero cost, never blocks on
    Postgres); a dedicated thread batches rows by size/time and flushes them.
    This decouples the shared live-connection callback thread from DB latency —
    without it, a slow/contended commit on one schema's write can stall the next
    record for *every* schema sharing that connection (this was true of tbbo's
    old per-record synchronous insert before this class existed).

    Data-loss policy on a persistent DB outage: drop the batch and keep
    streaming rather than raise. The raw .dbn archive is the system of record
    for replay, so staying connected (and keeping the archive growing) matters
    more than one writer's rows landing in real time; escalate (stop the live
    session) only if Postgres stays unreachable past ``degraded_max_s``.
    """

    def __init__(self, *, name: str, pool, flush_batch, batch_size: int,
                queue_max_rows: int, flush_interval_s: float, enqueue_timeout_s: float,
                flush_retries: int, retry_backoff_s: float, degraded_max_s: float,
                on_persistent_failure=None) -> None:
        self.name = name
        self._pool = pool
        self._flush_batch = flush_batch    # (conn, batch: list[tuple]) -> None; raises on failure
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._enqueue_timeout_s = enqueue_timeout_s
        self._flush_retries = flush_retries
        self._retry_backoff_s = retry_backoff_s
        self._degraded_max_s = degraded_max_s
        self._on_persistent_failure = on_persistent_failure
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max_rows)
        self._stop_token = object()
        self._thread: threading.Thread | None = None
        self._exc: BaseException | None = None
        self._degraded_since: float | None = None
        self._degraded_from_ts: datetime | None = None
        self._dropped_rows = 0
        self._degraded_last_log_s = 0.0
        self._backpressure_last_log_s = 0.0

    def start(self) -> None:
        self._exc = None
        self._thread = threading.Thread(
            target=self._loop, name=f"{self.name}-db-writer", daemon=True)
        self._thread.start()
        logger.info(f"{self.name}_writer_started", extra={
            "batch_size": self._batch_size, "queue_max_rows": self._queue.maxsize,
            "flush_interval_s": self._flush_interval_s})

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            while True:
                self._raise_if_unhealthy()
                try:
                    self._queue.put(self._stop_token, timeout=self._enqueue_timeout_s)
                    break
                except queue.Full:
                    now = time.monotonic()
                    if now - self._backpressure_last_log_s >= 1.0:
                        logger.warning(f"{self.name}_shutdown_backpressure", extra={
                            "queued": self._queue.qsize(), "capacity": self._queue.maxsize})
                        self._backpressure_last_log_s = now
            thread.join()
        self._thread = None
        if self._exc is not None:
            raise RuntimeError(f"{self.name}_writer_failed") from self._exc

    def _raise_if_unhealthy(self) -> None:
        if self._exc is not None:
            raise RuntimeError(f"{self.name}_writer_failed") from self._exc
        thread = self._thread
        if thread is not None and not thread.is_alive():
            raise RuntimeError(f"{self.name}_writer_stopped")

    def enqueue(self, row: tuple) -> None:
        self._raise_if_unhealthy()
        while True:
            try:
                self._queue.put(row, timeout=self._enqueue_timeout_s)
                return
            except queue.Full:
                self._raise_if_unhealthy()
                now = time.monotonic()
                if now - self._backpressure_last_log_s >= 1.0:
                    logger.warning(f"{self.name}_queue_backpressure", extra={
                        "queued": self._queue.qsize(), "capacity": self._queue.maxsize})
                    self._backpressure_last_log_s = now

    def _loop(self) -> None:
        batch: list[tuple] = []
        next_flush = time.monotonic() + self._flush_interval_s
        try:
            while True:
                timeout = self._flush_interval_s if not batch else max(0.0, next_flush - time.monotonic())
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    if batch and self._flush_resilient(batch):
                        batch.clear()
                    next_flush = time.monotonic() + self._flush_interval_s
                    continue

                if item is self._stop_token:
                    if batch:
                        self._flush_resilient(batch)
                        batch.clear()
                    return

                batch.append(item)
                if len(batch) >= self._batch_size:
                    if self._flush_resilient(batch):
                        batch.clear()
                    next_flush = time.monotonic() + self._flush_interval_s
        except BaseException as exc:  # noqa: BLE001
            self._exc = exc
            logger.exception(f"{self.name}_writer_failed")
            if self._on_persistent_failure is not None:
                try:
                    self._on_persistent_failure()
                except Exception:
                    logger.exception(f"live_stop_failed_after_{self.name}_writer_error")

    def _flush_resilient(self, batch: list[tuple]) -> bool:
        """Returns True if the caller should clear the batch (written, or
        deliberately dropped after exhausting retries). A fresh pooled
        connection is taken per attempt so a poisoned/aborted connection
        cannot wedge the writer forever."""
        if not batch:
            return True
        last_exc: BaseException | None = None
        for attempt in range(self._flush_retries):
            try:
                with self._pool.connection() as conn:
                    self._flush_batch(conn, batch)
                self._note_recovered()
                return True
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < self._flush_retries:
                    time.sleep(self._retry_backoff_s * (attempt + 1))
        self._note_degraded(len(batch), last_exc)
        return True

    def _note_degraded(self, n_rows: int, exc: BaseException | None) -> None:
        now = time.monotonic()
        if self._degraded_since is None:
            self._degraded_since = now
            self._degraded_from_ts = datetime.now(timezone.utc)
            logger.critical(f"{self.name}_writer_degraded_enter",
                            extra={"from_ts": str(self._degraded_from_ts), "error": repr(exc)})
        self._dropped_rows += n_rows
        if now - self._degraded_last_log_s >= 5.0:
            self._degraded_last_log_s = now
            logger.critical(f"{self.name}_writer_degraded", extra={
                "degraded_s": round(now - self._degraded_since, 1),
                "dropped_rows": self._dropped_rows, "error": repr(exc)})
        # Escalate only if the DB never comes back: a permanently broken writer
        # should surface as a service failure rather than silently drop forever.
        if now - self._degraded_since >= self._degraded_max_s:
            raise RuntimeError(
                f"{self.name}_writer_degraded_for_{int(now - self._degraded_since)}s") from exc

    def _note_recovered(self) -> None:
        if self._degraded_since is None:
            return
        logger.critical(f"{self.name}_writer_degraded_exit", extra={
            "gap_from": str(self._degraded_from_ts), "gap_to": str(datetime.now(timezone.utc)),
            "degraded_s": round(time.monotonic() - self._degraded_since, 1),
            "dropped_rows": self._dropped_rows,
            "backfill_hint": f"re-ingest the session .dbn via ingest_{self.name}"})
        self._degraded_since = None
        self._degraded_from_ts = None
        self._dropped_rows = 0


class LiveStreamer:
    """Persistent live streaming service with auto-reconnect."""

    # Class-level defaults for the bbo-writer resilience knobs. __init__ replaces
    # these with env-tunable per-instance values; they exist so the writer paths
    # stay usable on instances built via __new__ (the pattern used throughout
    # tests/test_live_options_subscription.py) without every such construction
    # having to know about each new tuning attribute.
    _bbo_flush_retries: int = 3
    _bbo_retry_backoff_s: float = 0.5
    _bbo_degraded_max_s: float = 900.0
    _bbo_degraded_since: float | None = None
    _bbo_degraded_from_ts: datetime | None = None
    _bbo_dropped_rows: int = 0
    _bbo_degraded_last_log_s: float = 0.0

    def __init__(self) -> None:
        # Persistent connection pool — shared across all callbacks.
        # Eliminates per-record TCP handshake + auth overhead that previously
        # caused service instability during high-frequency RTH tbbo bursts.
        self._pool = get_pool(min_size=2, max_size=5)
        self._record_count = 0
        self._data_dir = Path(os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))
        self._client: db.Live | None = None
        self._archive: RotatingDbnStream | None = None
        # Current-session instrument_id → symbol cache. Databento reuses IDs
        # across sessions, so historical IDs must never seed this mapping.
        # Live definitions and SymbolMappingMsg records populate it.
        self._sym_cache: dict[int, str] = {}
        # Stable raw symbols are safe to bootstrap and prevent redundant writes
        # when a live definition is received more than once.
        self._known_symbols: set[str] = set()
        # Symbols that exist ONLY as a reconstructed (synthetic) row. These must
        # stay writable so an authoritative vendor definition can correct them.
        self._synthetic_symbols: set[str] = set()
        self._vendor_definitions: dict[str, VendorDefinition] = {}
        self._definition_acquirer: DefinitionAcquirer | None = None

        # ── 0DTE Greeks live path ────────────────────────────────
        # instrument_id → {symbol, strike, is_call, expiry, expiration}
        self._opt_meta: dict[int, dict] = {}
        self._0dte_ids: set[int] = set()
        self._0dte_session: date | None = None  # CT date the 0DTE set is for
        self._nq_mid: float | None = None      # last outright seen (logging only)
        self._nq_sym: str = "NQ.c.0"
        # Mid per NQ outright. A single "the NQ price" is wrong: several quarterlies
        # quote at once, hundreds of points apart, and an option must be priced
        # against the one it settles into (ADR-153).
        self._fut_mids: dict[str, float] = {}
        # Buffer of two-sided 0DTE option quotes for the current second
        self._greeks_sec: datetime | None = None
        self._greeks_buf: dict[int, tuple[float, float]] = {}  # id → (bid, ask)
        self._surface_parquet_rows: list[dict] = []
        self._greeks_enabled = _GREEKS_ENABLED

        self._bbo_batch_size = _env_int("VTECH_LIVE_BBO_BATCH_SIZE", 2_000, 1)
        self._bbo_queue_max_rows = _env_int("VTECH_LIVE_BBO_QUEUE_MAX_ROWS", 100_000, self._bbo_batch_size)
        self._bbo_flush_interval_s = _env_float("VTECH_LIVE_BBO_FLUSH_INTERVAL_S", 0.25, 0.01)
        self._bbo_enqueue_timeout_s = _env_float("VTECH_LIVE_BBO_ENQUEUE_TIMEOUT_S", 1.0, 0.05)
        # Resilience: a DB write failure must not tear down the raw DBN archive,
        # which is the replay source used to backfill whatever the DB missed.
        # Retry a failing batch this many times before dropping it and entering
        # degraded mode; escalate (kill the session) only if the DB stays broken
        # past _bbo_degraded_max_s.
        self._bbo_flush_retries = _env_int("VTECH_LIVE_BBO_FLUSH_RETRIES", 3, 1)
        self._bbo_retry_backoff_s = _env_float("VTECH_LIVE_BBO_RETRY_BACKOFF_S", 0.5, 0.05)
        self._bbo_degraded_max_s = _env_float("VTECH_LIVE_BBO_DEGRADED_MAX_S", 900.0, 30.0)
        self._bbo_degraded_since: float | None = None
        self._bbo_degraded_from_ts: datetime | None = None
        self._bbo_dropped_rows = 0
        self._bbo_degraded_last_log_s = 0.0
        self._bbo_write_mode = "insert"
        self._bbo_insert_sql_cache: dict[int, str] = {}
        self._definition_insert_sql_cache: dict[int, str] = {}
        self._bbo_queue: queue.Queue = queue.Queue(maxsize=self._bbo_queue_max_rows)
        self._bbo_writer: threading.Thread | None = None
        self._bbo_writer_exc: BaseException | None = None
        self._bbo_stop_token = object()
        self._bbo_backpressure_last_log_s = 0.0

        # tbbo and mbo share the _BufferedTableWriter shape (see that class'
        # docstring). tbbo's tick rate is modest — bbo-1s-sized tuning. mbo's L3
        # event rate is far higher (~200-300/s sustained across a session,
        # bursting well above that at the open), so it gets a bigger batch/queue.
        self._tbbo_batch_size = _env_int("VTECH_LIVE_TBBO_BATCH_SIZE", 500, 1)
        self._tbbo_queue_max_rows = _env_int("VTECH_LIVE_TBBO_QUEUE_MAX_ROWS", 20_000, self._tbbo_batch_size)
        self._tbbo_flush_interval_s = _env_float("VTECH_LIVE_TBBO_FLUSH_INTERVAL_S", 0.25, 0.01)
        self._tbbo_enqueue_timeout_s = _env_float("VTECH_LIVE_TBBO_ENQUEUE_TIMEOUT_S", 1.0, 0.05)
        self._tbbo_flush_retries = _env_int("VTECH_LIVE_TBBO_FLUSH_RETRIES", 3, 1)
        self._tbbo_retry_backoff_s = _env_float("VTECH_LIVE_TBBO_RETRY_BACKOFF_S", 0.5, 0.05)
        self._tbbo_degraded_max_s = _env_float("VTECH_LIVE_TBBO_DEGRADED_MAX_S", 900.0, 30.0)
        self._tbbo_insert_sql_cache: dict[int, str] = {}
        self._tbbo_writer: _BufferedTableWriter | None = None

        self._mbo_batch_size = _env_int("VTECH_LIVE_MBO_BATCH_SIZE", 3000, 1)
        self._mbo_queue_max_rows = _env_int("VTECH_LIVE_MBO_QUEUE_MAX_ROWS", 300_000, self._mbo_batch_size)
        self._mbo_flush_interval_s = _env_float("VTECH_LIVE_MBO_FLUSH_INTERVAL_S", 0.25, 0.01)
        self._mbo_enqueue_timeout_s = _env_float("VTECH_LIVE_MBO_ENQUEUE_TIMEOUT_S", 2.0, 0.05)
        self._mbo_flush_retries = _env_int("VTECH_LIVE_MBO_FLUSH_RETRIES", 3, 1)
        self._mbo_retry_backoff_s = _env_float("VTECH_LIVE_MBO_RETRY_BACKOFF_S", 0.5, 0.05)
        self._mbo_degraded_max_s = _env_float("VTECH_LIVE_MBO_DEGRADED_MAX_S", 900.0, 30.0)
        self._mbo_writer: _BufferedTableWriter | None = None
        self._pending_weekly_definitions: dict[str, tuple] = {}
        self._pending_weekly_definitions_lock = threading.Lock()
        self._option_symbols: set[str] = set()
        self._option_session_date: date | None = None
        self._session_stop = threading.Event()
        self._option_refresh_thread: threading.Thread | None = None
        self._option_refresh_s = _env_float("VTECH_LIVE_OPTION_REFRESH_S", 300.0, 30.0)

    # ── public ───────────────────────────────────────────────────

    def start(self) -> None:
        """Run live sessions until CME weekend close or a fatal shutdown.

        systemd starts this unit on boot (weekday resize/reboot) and at
        Sunday 5pm CT (``vtech-live-streamer.timer``). Friday 4pm CT the
        stop timer takes the unit down. If a boot or a missed Persistent
        timer fires while CME is shut, exit 0 immediately — do not open
        Databento, do not write ``raw/live/`` — and leave the next Sunday
        timer to start us. ``Restart=on-failure`` will not bounce an
        exit-0 weekend skip.

        Each session archives under the Chicago option-session date (next
        trading day on weekends), never the UTC calendar date.
        """
        try:
            if cme_weekend_closed():
                logger.info("live_cme_weekend_skip")
                return
            while True:
                if cme_weekend_closed():
                    logger.info("live_cme_weekend_skip")
                    return
                try:
                    self._run_session()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    logger.exception("live_session_failed")
                    time.sleep(15)
        finally:
            close_pool()

    def _resolve_option_universe(self):
        try:
            return resolve_session_option_universe()
        except Exception:
            logger.exception("live_option_resolve_failed")
            return None

    @staticmethod
    def _open_archive() -> RotatingDbnStream:
        """The raw DBN archive writer for a weekly session.

        Rooted at the resolver-backed live archive directory and driven by the
        same session calendar the subscriptions roll on, so the archive's
        one-directory-per-session-date layout survives a connection that no
        longer restarts at each boundary.
        """
        return RotatingDbnStream(
            root=raw_dir() / "live",
            session_date_fn=live_option_session_date,
        )

    def _run_session(self) -> None:
        """Run one live session: CME Sunday open through Friday close.

        One ``db.Live`` connection for the whole week. Databento documents no
        session lifetime and recommends reusing a session rather than opening
        new ones, and the only scheduled CME Globex gateway restart is Saturday
        02:00 CT — inside the weekend halt this process is already stopped for.
        Verified against the archive: a single connection already spanned the
        daily 16:00-17:00 CT halt (169,834 rows before, 318,944 after).

        The session date still advances at 16:05 CT; it just no longer tears the
        connection down to do it. The archive keeps one directory per session
        date through :class:`RotatingDbnStream`, which rotates on the same
        calendar the subscriptions roll on.
        """
        # Archive layout is unchanged — one directory per Chicago session date —
        # but the file now rotates inside the live session rather than being
        # fixed at connect time. add_stream has no removal counterpart, so a
        # week-long session would otherwise write one file for the week.
        archive = self._open_archive()
        self._archive = archive

        universe = self._resolve_option_universe()
        option_symbols = list(universe.symbols) if universe is not None else []
        self._option_symbols = set(option_symbols)
        self._option_session_date = (
            universe.session_date if universe is not None else live_option_session_date()
        )
        if universe is not None and not universe.complete:
            logger.critical(
                "live_option_universe_incomplete",
                extra={"session_date": str(universe.session_date),
                       "missing": list(universe.missing),
                       "n_symbols": len(option_symbols)},
            )

        client = db.Live(
            reconnect_policy="reconnect",
            heartbeat_interval_s=15,
            slow_reader_behavior="warn",  # never skip records; lag over loss
        )
        self._client = client

        # Futures stay on parent symbology. Options are exact raw_symbol
        # strikes for 0DTE / 1DTE / QNE1 / QNE2 / front quarterly (ADR-034).
        subscribe_live_feeds(client, option_symbols=option_symbols)

        client.add_callback(
            record_callback=self._handle_record,
            exception_callback=self._handle_error,
        )
        client.add_stream(archive, exception_callback=self._handle_error)
        client.add_reconnect_callback(reconnect_callback=self._handle_reconnect)

        self._bootstrap_definitions()
        if self._greeks_enabled:
            try:
                from src.features.options_greeks_pipeline import ensure_tables
                with self._pool.connection() as conn:
                    ensure_tables(conn)
                self._refresh_0dte_set()
                logger.info("live_greeks_enabled", extra={"n_0dte": len(self._0dte_ids)})
            except Exception:
                logger.exception("live_greeks_init_failed")
                self._greeks_enabled = False
        bbo_started = False
        bbo_shutdown_exc: Exception | None = None
        self._tbbo_writer = _BufferedTableWriter(
            name="tbbo", pool=self._pool, flush_batch=self._flush_tbbo_batch,
            batch_size=self._tbbo_batch_size, queue_max_rows=self._tbbo_queue_max_rows,
            flush_interval_s=self._tbbo_flush_interval_s, enqueue_timeout_s=self._tbbo_enqueue_timeout_s,
            flush_retries=self._tbbo_flush_retries, retry_backoff_s=self._tbbo_retry_backoff_s,
            degraded_max_s=self._tbbo_degraded_max_s, on_persistent_failure=self._stop_live_client)
        self._mbo_writer = _BufferedTableWriter(
            name="mbo", pool=self._pool, flush_batch=self._flush_mbo_batch,
            batch_size=self._mbo_batch_size, queue_max_rows=self._mbo_queue_max_rows,
            flush_interval_s=self._mbo_flush_interval_s, enqueue_timeout_s=self._mbo_enqueue_timeout_s,
            flush_retries=self._mbo_flush_retries, retry_backoff_s=self._mbo_retry_backoff_s,
            degraded_max_s=self._mbo_degraded_max_s, on_persistent_failure=self._stop_live_client)
        self._session_stop.clear()
        try:
            self._definition_acquirer = DefinitionAcquirer(self._pool)
            self._definition_acquirer.start()
            self._submit_option_definitions()
            self._start_option_refresh(client)
            self._start_bbo_writer()
            bbo_started = True
            self._tbbo_writer.start()
            self._mbo_writer.start()
            logger.info(
                "live_start",
                extra={
                    "dataset": DATASET,
                    "archive_root": str(raw_dir() / "live"),
                    "option_session": str(self._option_session_date),
                    "n_option_symbols": len(self._option_symbols),
                },
            )
            client.start()
            client.block_for_close()
        finally:
            self._session_stop.set()
            self._stop_option_refresh()
            if self._definition_acquirer is not None:
                try:
                    self._definition_acquirer.stop()
                except Exception as exc:
                    bbo_shutdown_exc = bbo_shutdown_exc or exc
                    logger.exception("definition_acquisition_shutdown_failed")
                self._definition_acquirer = None
            for writer in (self._tbbo_writer, self._mbo_writer):
                try:
                    writer.stop()
                except Exception as exc:  # noqa: BLE001
                    bbo_shutdown_exc = bbo_shutdown_exc or exc
                    logger.exception(f"{writer.name}_writer_shutdown_failed")
            if bbo_started:
                try:
                    self._stop_bbo_writer()
                except Exception as exc:  # noqa: BLE001
                    bbo_shutdown_exc = bbo_shutdown_exc or exc
                    logger.exception("bbo_writer_shutdown_failed")
            if self._greeks_enabled:
                self._flush_greeks_second(force=True)
                self._flush_surface_parquet()
            try:
                archive.close()
                logger.info("dbn_archive_closed", extra={
                    "files_opened": archive.files_opened,
                    "rotations": archive.rotations,
                    "records_written": archive.records_written,
                })
            except Exception:  # noqa: BLE001 — never mask a shutdown error
                logger.exception("dbn_archive_close_failed")
            self._archive = None
            self._client = None
            self._tbbo_writer = None
            self._mbo_writer = None
            if bbo_shutdown_exc is not None:
                raise bbo_shutdown_exc

    def _stop_live_client(self) -> None:
        if self._client is not None:
            self._client.stop()

    def _start_option_refresh(self, client: db.Live) -> None:
        self._option_refresh_thread = threading.Thread(
            target=self._option_refresh_loop,
            args=(client,),
            name="vtech-live-option-refresh",
            daemon=True,
        )
        self._option_refresh_thread.start()

    def _stop_option_refresh(self) -> None:
        thread = self._option_refresh_thread
        self._option_refresh_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _option_refresh_loop(self, client: db.Live) -> None:
        """Re-resolve the five chains: add new strikes, roll the session date.

        The loop runs for the life of the weekly session. It no longer restarts
        the client on a date roll — see :meth:`_roll_session`.
        """
        while not self._session_stop.wait(self._option_refresh_s):
            try:
                self._refresh_option_subscriptions(client)
            except Exception:
                logger.exception("live_option_refresh_failed")

    def _roll_session(self, session_date) -> None:
        """Advance to a new session date without dropping the connection.

        Called from the refresh loop when the session calendar advances, which
        happens at 16:05 CT — inside the 16:01-16:44 CT window where no quote is
        written, so nothing in flight is lost.

        Per-session buffers are flushed here rather than at connection teardown,
        because there is no longer a teardown per session. Without this the
        whole week's surface rows would land in Friday's parquet partition.
        """
        previous = self._option_session_date
        self._option_session_date = session_date
        if self._greeks_enabled:
            try:
                self._flush_greeks_second(force=True)
                self._flush_surface_parquet()
            except Exception:  # noqa: BLE001 — a flush fault must not stop the stream
                logger.exception("session_roll_flush_failed")
        logger.info(
            "live_option_session_roll",
            extra={
                "from": str(previous),
                "to": str(session_date),
                "archive_files": getattr(self._archive, "files_opened", None),
                "n_subscribed": len(self._option_symbols),
            },
        )

    def _refresh_option_subscriptions(self, client: db.Live) -> None:
        if cme_weekend_closed():
            logger.info("live_cme_weekend_close")
            client.stop()
            return
        # Pending definition acquisition does not depend on a successful
        # refresh resolve or on newly added subscriptions.
        self._submit_option_definitions()
        universe = self._resolve_option_universe()
        if universe is None:
            return

        if universe.session_date != self._option_session_date:
            # Session roll. The connection stays up: the new session's chains are
            # subscribed additively below, and the archive rotates itself on the
            # same calendar. Tearing the client down here used to cost 10-50s of
            # unrecoverable quotes (measured 25s/50s/42s/10s across one week's
            # boundaries) because the roll was at Chicago midnight, in live
            # Asian-session trading, and nothing replays a live gap.
            self._roll_session(universe.session_date)

        new_symbols = [s for s in universe.symbols if s not in self._option_symbols]
        if new_symbols:
            client.subscribe(
                dataset=DATASET, schema="definition",
                symbols=new_symbols, stype_in="raw_symbol",
            )
            client.subscribe(
                dataset=DATASET, schema="bbo-1s",
                symbols=new_symbols, stype_in="raw_symbol",
            )
            self._option_symbols.update(new_symbols)
            logger.info(
                "live_option_symbols_added",
                extra={"n_new": len(new_symbols), "n_total": len(self._option_symbols)},
            )

        # Definition presence is a separate question from subscription state,
        # and must be checked even when nothing new was subscribed. Diffing on
        # ``_option_symbols`` alone is exactly why a strike whose definition
        # never arrived stayed invisible forever: it *was* subscribed, so the
        # "nothing new, return" shortcut this replaced skipped it on every later
        # refresh. Measured
        # consequence — 74 of 246 quoted NQZ6 strikes held no definition row on
        # every session from the 2026-09-18 roll onward, and because
        # ``_load_bbo_1s`` resolves instrument ids *from* ``definitions`` they
        # were silently absent from every options feature.
        self._submit_option_definitions()
        self._report_definition_gaps(universe)

    def definition_gaps(self, symbols) -> list[str]:
        """Symbols still awaiting committed authoritative definition acquisition.

        Legacy synthetic rows remain pending. Acquisition results refresh the
        view independently of live callbacks and subscription differences.
        """
        authoritative = set(self._known_symbols)
        acquirer = getattr(self, "_definition_acquirer", None)
        if acquirer is not None:
            if acquirer.database_checked:
                authoritative = set(acquirer.definitions)
            else:
                authoritative.update(acquirer.definitions)
        return sorted(set(symbols) - authoritative)

    def _report_definition_gaps(self, universe) -> None:
        """Expose pending metadata while normal acquisition continues in parallel."""
        acquirer = getattr(self, "_definition_acquirer", None)
        if getattr(self, "_definitions_view_degraded", False) and (
            acquirer is None or not acquirer.database_checked
        ):
            # Every symbol would look missing; that is a read failure, not a
            # data gap, and reporting it would bury the real signal.
            logger.warning(
                "option_definitions_gap_check_skipped",
                extra={"reason": "definitions_view_degraded",
                       "n_resolved": len(universe.symbols)},
            )
            return
        gaps = self.definition_gaps(universe.symbols)
        if not gaps:
            return
        by_root: dict[str, int] = {}
        for symbol in gaps:
            root = symbol.split(" ", 1)[0]
            by_root[root] = by_root.get(root, 0) + 1
        logger.warning(
            "option_definitions_missing",
            extra={
                "session_date": str(universe.session_date),
                "n_missing": len(gaps),
                "n_resolved": len(universe.symbols),
                "by_root": by_root,
                "sample": gaps[:5],
            },
        )

    # ── callbacks ────────────────────────────────────────────────

    def _handle_record(self, record: db.DBNRecord) -> None:
        self._record_count += 1
        if self._record_count % 10_000 == 0:
            logger.info("live_progress", extra={"records": self._record_count})

        if isinstance(record, db.TBBOMsg):
            self._insert_tbbo(record)
        elif isinstance(record, db.MBOMsg):
            self._insert_mbo(record)
        elif isinstance(record, db.BBOMsg):
            self._insert_bbo_1s(record)
        elif isinstance(record, db.InstrumentDefMsg):
            self._upsert_definition(record)
        elif isinstance(record, db.StatMsg):
            self._insert_statistics(record)
        elif isinstance(record, db.SymbolMappingMsg):
            self._handle_symbol_mapping(record)
        elif isinstance(record, db.SystemMsg):
            if not record.is_heartbeat():
                code = SystemCode.from_int(int(record.code))
                if code == SystemCode.SLOW_READER_WARNING:
                    logger.warning("slow_reader_warning", extra={"sys_msg": record.msg})
                else:
                    logger.debug("system_msg", extra={"code": code.name, "sys_msg": record.msg})
        elif isinstance(record, db.ErrorMsg):
            _FATAL = {
                ErrorCode.AUTH_FAILED, ErrorCode.API_KEY_DEACTIVATED,
                ErrorCode.CONNECTION_LIMIT_EXCEEDED, ErrorCode.INVALID_SUBSCRIPTION,
                ErrorCode.INTERNAL_ERROR,
            }
            ec = ErrorCode.from_int(int(record.code))
            if ec in _FATAL:
                logger.critical("gateway_fatal_error",
                                extra={"err": record.err, "code": ec.name,
                                       "is_last": record.is_last})
            else:
                logger.warning("gateway_error",
                               extra={"err": record.err, "code": ec.name,
                                      "is_last": record.is_last})

    def _handle_error(self, exc: Exception) -> None:
        logger.exception("callback_error", exc_info=exc)

    def _handle_reconnect(self, start, end) -> None:
        gap_s = (end - start).total_seconds() if hasattr((end - start), "total_seconds") else float(end - start) / 1e9
        logger.warning("reconnect_gap", extra={"start": str(start), "end": str(end), "gap_s": round(gap_s, 3)})

    def _handle_symbol_mapping(self, rec: db.SymbolMappingMsg) -> None:
        """Update the in-memory symbol cache from a live SymbolMappingMsg.

        The gateway sends these after session start to map instrument_id
        to text symbols.  We store ``stype_out_symbol`` — the resolved OUTRIGHT
        instrument symbol (e.g. "NQU6", "NQU6 P36750") — NOT ``stype_in_symbol``,
        which for our parent subscriptions is only the parent ("NQ.FUT"/"NQ.OPT")
        and would clobber the correct outright the feature engine filters on.
        """
        instrument_id = int(rec.instrument_id)
        sym = str(rec.stype_out_symbol) if rec.stype_out_symbol else ""
        if sym and sym != self._sym_cache.get(instrument_id):
            getattr(self, "_opt_meta", {}).pop(instrument_id, None)
            getattr(self, "_0dte_ids", set()).discard(instrument_id)
            getattr(self, "_greeks_buf", {}).pop(instrument_id, None)
            self._sym_cache[instrument_id] = sym
            logger.debug("symbol_mapped", extra={"instrument_id": instrument_id, "symbol": sym})

    # ── helpers ──────────────────────────────────────────────────

    def _ensure_weekly_definition_state(self) -> None:
        if not hasattr(self, "_pending_weekly_definitions"):
            self._pending_weekly_definitions = {}
        if not hasattr(self, "_pending_weekly_definitions_lock"):
            self._pending_weekly_definitions_lock = threading.Lock()
        if not hasattr(self, "_definition_insert_sql_cache"):
            self._definition_insert_sql_cache = {}
        if not hasattr(self, "_known_symbols"):
            self._known_symbols = set()
        if not hasattr(self, "_synthetic_symbols"):
            self._synthetic_symbols = set()
        if not hasattr(self, "_opt_meta"):
            self._opt_meta = {}
        if not hasattr(self, "_0dte_ids"):
            self._0dte_ids = set()

    def _submit_option_definitions(self) -> None:
        """Run metadata acquisition beside subscriptions, including retry gaps."""
        acquirer = getattr(self, "_definition_acquirer", None)
        if acquirer is not None:
            acquirer.submit(self._option_symbols)

    def _bind_option_definition(
        self, instrument_id: int, symbol: str, ts: datetime | None = None,
    ) -> bool:
        """Attach vendor terms by symbol to this live ID; never use historical IDs."""
        self._ensure_weekly_definition_state()
        definition = getattr(self, "_vendor_definitions", {}).get(symbol)
        acquirer = getattr(self, "_definition_acquirer", None)
        decision = ts or datetime.now(timezone.utc)
        if acquirer is not None:
            acquired = acquirer.get(symbol)
            if acquired is not None and acquired.option_ready and acquired.ts_event <= decision:
                if (definition is None or not definition.option_ready
                        or definition.ts_event > decision
                        or acquired.ts_event > definition.ts_event):
                    definition = acquired
        if definition is None or not definition.option_ready or definition.ts_event > decision:
            self._opt_meta.pop(instrument_id, None)
            self._0dte_ids.discard(instrument_id)
            getattr(self, "_greeks_buf", {}).pop(instrument_id, None)
            return False
        expiration = definition.expiration
        assert expiration is not None
        expiry = expiration.astimezone(_CT).date()
        self._opt_meta[instrument_id] = {
            "symbol": symbol, "strike": definition.strike_price,
            "is_call": definition.instrument_class == "C",
            "expiry": expiry, "expiration": expiration,
            "underlying": definition.underlying,
        }
        self._known_symbols.add(symbol)
        self._synthetic_symbols.discard(symbol)
        if expiry == decision.astimezone(_CT).date() and expiration > decision:
            self._0dte_ids.add(instrument_id)
        else:
            self._0dte_ids.discard(instrument_id)
        return True

    def _register_resolved_weekly_from_bbo(
        self, *, instrument_id: int, symbol: str, ts: datetime | None,
    ) -> None:
        """Compatibility hook: bind authoritative terms, never reconstruct rows."""
        self._bind_option_definition(instrument_id, symbol, ts)

    def _flush_pending_weekly_definitions(self, conn) -> None:
        """Retained compatibility hook; synthetic inserts are disabled."""
        return None

    def _resolve_symbol(self, instrument_id: int) -> str:
        """Resolve instrument_id to symbol.

        Lookup order:
          1. In-memory cache populated only from current live records
          2. Live client's symbology_map (set from SymbolMappingMsg records by the client)
          3. Log a warning and return "" — definition message will arrive shortly
             via the live feed and the next lookup for this id will succeed.
        """
        sym = self._sym_cache.get(instrument_id)
        if sym:
            return sym
        if self._client:
            live_sym = self._client.symbology_map.get(instrument_id, "")
            if live_sym:
                self._sym_cache[instrument_id] = str(live_sym)  # promote to cache
                return str(live_sym)
        logger.warning("symbol_not_found", extra={"instrument_id": instrument_id})
        return ""

    # ── database inserts ─────────────────────────────────────────

    def _bbo_table_has_conflict_arbiter(self) -> bool:
        sql = """
            WITH target AS (
                SELECT c.oid
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'databento' AND c.relname = 'bbo_1s'
                LIMIT 1
            )
            SELECT (
                EXISTS (
                    SELECT 1
                    FROM target t
                    JOIN pg_constraint c ON c.conrelid = t.oid
                    WHERE c.contype IN ('p', 'u', 'x')
                ) OR EXISTS (
                    SELECT 1
                    FROM target t
                    JOIN pg_index i ON i.indrelid = t.oid
                    WHERE i.indisunique
                )
            )
        """
        with self._pool.connection() as conn:
            row = conn.execute(sql).fetchone()
        return bool(row and row[0])

    def _start_bbo_writer(self) -> None:
        has_conflict_arbiter = self._bbo_table_has_conflict_arbiter()
        self._bbo_write_mode = "insert" if has_conflict_arbiter else "copy"
        if self._bbo_write_mode == "insert" and self._bbo_batch_size > _BBO_INSERT_MAX_ROWS:
            logger.warning(
                "bbo_batch_size_clamped",
                extra={"requested": self._bbo_batch_size, "max_insert_rows": _BBO_INSERT_MAX_ROWS},
            )
            self._bbo_batch_size = _BBO_INSERT_MAX_ROWS
        self._bbo_writer_exc = None
        self._bbo_writer = threading.Thread(
            target=self._bbo_writer_loop,
            name="bbo-1s-db-writer",
            daemon=True,
        )
        self._bbo_writer.start()
        logger.info(
            "bbo_writer_started",
            extra={
                "mode": self._bbo_write_mode,
                "batch_size": self._bbo_batch_size,
                "queue_max_rows": self._bbo_queue.maxsize,
                "flush_interval_s": self._bbo_flush_interval_s,
            },
        )

    def _stop_bbo_writer(self) -> None:
        writer = self._bbo_writer
        if writer is None:
            return
        if writer.is_alive():
            while True:
                self._raise_if_bbo_writer_unhealthy()
                try:
                    self._bbo_queue.put(self._bbo_stop_token, timeout=self._bbo_enqueue_timeout_s)
                    break
                except queue.Full:
                    now = time.monotonic()
                    if now - self._bbo_backpressure_last_log_s >= 1.0:
                        logger.warning(
                            "bbo_shutdown_backpressure",
                            extra={"queued": self._bbo_queue.qsize(), "capacity": self._bbo_queue.maxsize},
                        )
                        self._bbo_backpressure_last_log_s = now
            writer.join()
        self._bbo_writer = None
        if self._bbo_writer_exc is not None:
            raise RuntimeError("bbo_1s_writer_failed") from self._bbo_writer_exc

    def _raise_if_bbo_writer_unhealthy(self) -> None:
        if self._bbo_writer_exc is not None:
            raise RuntimeError("bbo_1s_writer_failed") from self._bbo_writer_exc
        writer = self._bbo_writer
        if writer is not None and not writer.is_alive():
            raise RuntimeError("bbo_1s_writer_stopped")

    def _enqueue_bbo_row(self, row: tuple) -> None:
        self._raise_if_bbo_writer_unhealthy()
        while True:
            try:
                self._bbo_queue.put(row, timeout=self._bbo_enqueue_timeout_s)
                return
            except queue.Full:
                self._raise_if_bbo_writer_unhealthy()
                now = time.monotonic()
                if now - self._bbo_backpressure_last_log_s >= 1.0:
                    logger.warning(
                        "bbo_queue_backpressure",
                        extra={"queued": self._bbo_queue.qsize(), "capacity": self._bbo_queue.maxsize},
                    )
                    self._bbo_backpressure_last_log_s = now

    def _bbo_writer_loop(self) -> None:
        batch: list[tuple] = []
        next_flush = time.monotonic() + self._bbo_flush_interval_s
        try:
            while True:
                timeout = self._bbo_flush_interval_s if not batch else max(0.0, next_flush - time.monotonic())
                try:
                    item = self._bbo_queue.get(timeout=timeout)
                except queue.Empty:
                    if batch and self._flush_bbo_batch_resilient(batch):
                        batch.clear()
                    next_flush = time.monotonic() + self._bbo_flush_interval_s
                    continue

                if item is self._bbo_stop_token:
                    if batch:
                        self._flush_bbo_batch_resilient(batch)
                        batch.clear()
                    return

                batch.append(item)
                if len(batch) >= self._bbo_batch_size:
                    if self._flush_bbo_batch_resilient(batch):
                        batch.clear()
                    next_flush = time.monotonic() + self._bbo_flush_interval_s
        except BaseException as exc:  # noqa: BLE001
            self._bbo_writer_exc = exc
            logger.exception("bbo_writer_failed")
            if self._client is not None:
                try:
                    self._client.stop()
                except Exception:
                    logger.exception("live_stop_failed_after_bbo_writer_error")

    def _flush_bbo_batch_resilient(self, batch: list[tuple]) -> bool:
        """Flush ``batch``, surviving transient DB outages.

        Returns True if the caller should clear the batch (written, or
        deliberately dropped). A fresh pooled connection is taken per attempt so
        a poisoned/aborted connection cannot wedge the writer forever.

        Data-loss policy: the raw DBN archive is the system of record for replay,
        so when the DB is unreachable we keep the stream (and therefore the
        archive) alive and drop DB rows, logging the exact gap window and row
        count for later backfill from the .dbn. Raising here would kill the
        archive too and make the outage unrecoverable.
        """
        if not batch:
            return True
        last_exc: BaseException | None = None
        for attempt in range(self._bbo_flush_retries):
            try:
                # Retry wraps ONLY the bbo write. Both writers commit on success
                # and roll back on error, so a retried attempt cannot duplicate
                # rows -- important because copy mode has no conflict arbiter.
                with self._pool.connection() as conn:
                    if self._bbo_write_mode == "copy":
                        self._flush_bbo_batch_copy(conn, batch)
                    else:
                        self._flush_bbo_batch_insert(conn, batch)
                self._note_bbo_recovered()
                return True
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < self._bbo_flush_retries:
                    time.sleep(self._bbo_retry_backoff_s * (attempt + 1))

        # Persistent failure: drop this batch, stay alive, keep the archive.
        self._note_bbo_degraded(len(batch), last_exc)
        return True

    def _note_bbo_degraded(self, n_rows: int, exc: BaseException | None) -> None:
        now = time.monotonic()
        if self._bbo_degraded_since is None:
            self._bbo_degraded_since = now
            self._bbo_degraded_from_ts = datetime.now(timezone.utc)
            logger.critical(
                "bbo_writer_degraded_enter",
                extra={"from_ts": str(self._bbo_degraded_from_ts), "error": repr(exc)},
            )
        self._bbo_dropped_rows += n_rows
        if now - self._bbo_degraded_last_log_s >= 5.0:
            self._bbo_degraded_last_log_s = now
            logger.critical(
                "bbo_writer_degraded",
                extra={
                    "degraded_s": round(now - self._bbo_degraded_since, 1),
                    "dropped_rows": self._bbo_dropped_rows,
                    "error": repr(exc),
                },
            )
        # Escalate only if the DB never comes back: a permanently broken writer
        # should surface as a service failure rather than silently drop forever.
        if now - self._bbo_degraded_since >= self._bbo_degraded_max_s:
            raise RuntimeError(
                f"bbo_1s_writer_degraded_for_{int(now - self._bbo_degraded_since)}s"
            ) from exc

    def _note_bbo_recovered(self) -> None:
        if self._bbo_degraded_since is None:
            return
        logger.critical(
            "bbo_writer_degraded_exit",
            extra={
                "gap_from": str(self._bbo_degraded_from_ts),
                "gap_to": str(datetime.now(timezone.utc)),
                "degraded_s": round(time.monotonic() - self._bbo_degraded_since, 1),
                "dropped_rows": self._bbo_dropped_rows,
                "backfill_hint": "re-ingest the session .dbn via ingest_bbo_1s",
            },
        )
        self._bbo_degraded_since = None
        self._bbo_degraded_from_ts = None
        self._bbo_dropped_rows = 0

    def _flush_bbo_batch(self, conn, batch: list[tuple]) -> None:
        if not batch:
            return
        if self._bbo_write_mode == "copy":
            self._flush_bbo_batch_copy(conn, batch)
        else:
            self._flush_bbo_batch_insert(conn, batch)

    def _flush_bbo_batch_insert(self, conn, batch: list[tuple]) -> None:
        n_rows = len(batch)
        sql = self._bbo_insert_sql_cache.get(n_rows)
        if sql is None:
            values = ",".join([f"({','.join(['%s'] * _BBO_INSERT_PARAMS)})"] * n_rows)
            sql = (
                "INSERT INTO databento.bbo_1s "
                f"{_BBO_INSERT_COLS_SQL} VALUES {values} "
                "ON CONFLICT DO NOTHING"
            )
            self._bbo_insert_sql_cache[n_rows] = sql
        params = [value for row in batch for value in row]
        try:
            conn.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _flush_bbo_batch_copy(self, conn, batch: list[tuple]) -> None:
        sql = (
            "COPY databento.bbo_1s "
            f"{_BBO_INSERT_COLS_SQL} "
            "FROM STDIN"
        )
        try:
            with conn.cursor() as cur:
                with cur.copy(sql) as copy:
                    for row in batch:
                        copy.write_row(row)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _insert_tbbo(self, rec: db.TBBOMsg) -> None:
        lvl = rec.levels[0] if rec.levels else None
        self._tbbo_writer.enqueue((
            _ns_to_dt(rec.ts_event), _ns_to_dt(rec.ts_recv), rec.instrument_id,
            self._resolve_symbol(rec.instrument_id),
            _px(rec.price), int(rec.size),
            str(rec.side) if hasattr(rec, "side") else None,
            str(rec.action) if hasattr(rec, "action") else None,
            int(getattr(rec, "flags", 0)), getattr(rec, "sequence", None),
            _px(lvl.bid_px) if lvl else None,
            _px(lvl.ask_px) if lvl else None,
            int(lvl.bid_sz) if lvl else None,
            int(lvl.ask_sz) if lvl else None,
            int(lvl.bid_ct) if lvl else None,
            int(lvl.ask_ct) if lvl else None,
            int(getattr(rec, "ts_in_delta", 0)),
        ))

    def _flush_tbbo_batch(self, conn, batch: list[tuple]) -> None:
        n_rows = len(batch)
        sql = self._tbbo_insert_sql_cache.get(n_rows)
        if sql is None:
            values = ",".join([f"({','.join(['%s'] * _TBBO_INSERT_PARAMS)})"] * n_rows)
            sql = (
                "INSERT INTO databento.tbbo "
                f"{_TBBO_INSERT_COLS_SQL} VALUES {values} "
                "ON CONFLICT DO NOTHING"
            )
            self._tbbo_insert_sql_cache[n_rows] = sql
        params = [value for row in batch for value in row]
        try:
            conn.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _insert_mbo(self, rec: db.MBOMsg) -> None:
        self._mbo_writer.enqueue((
            _ns_to_dt(rec.ts_recv), _ns_to_dt(rec.ts_event), int(rec.rtype),
            int(rec.publisher_id), int(rec.instrument_id),
            str(rec.action) if hasattr(rec, "action") else None,
            str(rec.side) if hasattr(rec, "side") else None,
            _mbo_px(rec.price), int(rec.size), int(rec.channel_id),
            int(rec.order_id), int(rec.flags), int(rec.ts_in_delta), int(rec.sequence),
            self._resolve_symbol(rec.instrument_id),
        ))

    def _flush_mbo_batch(self, conn, batch: list[tuple]) -> None:
        # No unique constraint on databento.mbo by design (same-price sweep
        # fills yield legitimate duplicate tuples — see create_mbo_table.sql),
        # so a plain COPY is correct and, at mbo's volume, faster than a
        # multi-row INSERT — same choice ingest_mbo.py makes for the batch path.
        sql = f"COPY databento.mbo {_MBO_INSERT_COLS_SQL} FROM STDIN"
        try:
            with conn.cursor() as cur:
                with cur.copy(sql) as copy:
                    for row in batch:
                        copy.write_row(row)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _insert_bbo_1s(self, rec: db.BBOMsg) -> None:
        lvl = rec.levels[0] if rec.levels else None
        ts = _ns_to_dt(rec.ts_recv)
        bid = _px(lvl.bid_px) if lvl else None
        ask = _px(lvl.ask_px) if lvl else None
        bid_sz = int(lvl.bid_sz) if lvl else None
        ask_sz = int(lvl.ask_sz) if lvl else None
        symbol = self._resolve_symbol(rec.instrument_id)
        is_option = " " in symbol and not symbol.startswith("UD:")
        metadata_ready = not is_option or self._bind_option_definition(
            int(rec.instrument_id), symbol, ts,
        )
        self._enqueue_bbo_row((
            ts,
            int(rec.instrument_id),
            symbol,
            _px(rec.price),
            int(rec.size),
            bid, ask, bid_sz, ask_sz,
        ))
        if self._greeks_enabled and ts is not None and metadata_ready:
            self._on_bbo_for_greeks(
                ts=ts,
                instrument_id=int(rec.instrument_id),
                symbol=symbol,
                bid=bid, ask=ask, bid_sz=bid_sz, ask_sz=ask_sz,
            )

    def _upsert_definition(self, rec: db.InstrumentDefMsg) -> None:
        """Write a new instrument definition to DB and cache.

        The definition subscription cannot be cancelled mid-session. Current
        updates arrive live; the subscription companion obtains missing terms
        from the database or a cost-checked historical definition request.
        Instrument IDs are reused across sessions, so definition deduplication
        must use the stable raw symbol. Only genuinely new symbols reach the DB,
        while every live definition refreshes the current ID-to-symbol mapping.
        """
        instrument_id = int(rec.instrument_id)

        # Prefer raw_symbol from the definition itself (always populated);
        # fall back to cache/symbology_map for any edge case.
        sym = str(rec.raw_symbol) if rec.raw_symbol else self._resolve_symbol(instrument_id)
        # A symbol present only as a synthetic reconstruction still needs the
        # vendor definition written so the upsert can correct it.
        already = bool(sym) and sym in self._known_symbols
        definition = VendorDefinition.from_record(rec)
        iclass = definition.instrument_class
        strike = definition.strike_price
        expiration = definition.expiration
        underlying = definition.underlying

        if not already:
            sql = (
                "INSERT INTO databento.definitions "
                f"{_DEF_INSERT_COLS_SQL} "
                f"VALUES ({','.join(['%s'] * _DEF_INSERT_PARAMS)}) "
                f"{_DEF_UPSERT_TAIL}"
            )
            with self._pool.connection() as conn:
                conn.execute(sql, (
                    _ns_to_dt(rec.ts_event), instrument_id,
                    sym,
                    iclass,
                    strike,
                    expiration,
                    underlying,
                    getattr(rec, "exchange", None),
                    getattr(rec, "currency", "USD"),
                    definition.min_price_increment,
                    definition.multiplier,
                    _DEF_SOURCE_VENDOR,
                ))
            if sym:
                self._known_symbols.add(sym)
                self._synthetic_symbols.discard(sym)
                logger.info("definition_new", extra={"instrument_id": instrument_id, "symbol": sym})

        if sym:
            if sym != self._sym_cache.get(instrument_id):
                getattr(self, "_opt_meta", {}).pop(instrument_id, None)
                getattr(self, "_0dte_ids", set()).discard(instrument_id)
                getattr(self, "_greeks_buf", {}).pop(instrument_id, None)
            self._sym_cache[instrument_id] = sym

        # Historical and live definitions normalize the same vendor fields.
        # Contract size is unit_of_measure_qty, not an absent multiplier field.
        if not hasattr(self, "_vendor_definitions"):
            self._vendor_definitions = {}
        self._vendor_definitions[sym] = definition
        self._bind_option_definition(instrument_id, sym)

    def _bootstrap_definitions(self) -> None:
        """Pre-load stable symbols for live definition deduplication.

        Instrument IDs are session-scoped and reusable, so current ID mappings
        come only from live mapping/definition records. Option terms are joined
        separately by raw symbol from authoritative stored or acquired rows.

        Postgres being down must not prevent the raw DBN archive from opening.
        Retry a few times, then start degraded (empty known-symbol set).
        """
        last_exc: BaseException | None = None
        rows = None
        for attempt in range(1, 9):
            try:
                with self._pool.connection() as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT symbol, source FROM databento.definitions"
                        " WHERE symbol IS NOT NULL AND symbol != ''"
                    ).fetchall()
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "definitions_bootstrap_retry",
                    extra={"attempt": attempt, "err": str(exc)},
                )
                time.sleep(min(30, 2 ** attempt))
        self._sym_cache = {}
        self._opt_meta = {}
        self._vendor_definitions = {}
        if rows is None:
            logger.error(
                "definitions_bootstrap_degraded",
                extra={"err": str(last_exc)},
            )
            self._known_symbols = set()
            self._synthetic_symbols = set()
            # The in-memory view of the table is unusable, so it cannot be used
            # to tell "this symbol has no definition row" from "we never read
            # the table". Gap reporting is suppressed while this is set.
            self._definitions_view_degraded = True
        else:
            self._definitions_view_degraded = False
            self._known_symbols = {
                str(r[0]) for r in rows if str(r[1]) != _DEF_SOURCE_SYNTHETIC
            }
            self._synthetic_symbols = {
                str(r[0]) for r in rows if str(r[1]) == _DEF_SOURCE_SYNTHETIC
            } - self._known_symbols
        self._ensure_weekly_definition_state()
        with self._pending_weekly_definitions_lock:
            self._pending_weekly_definitions.clear()
        logger.info("definitions_bootstrapped", extra={
            "n_symbols": len(self._known_symbols),
            "degraded": rows is None,
        })

    # ── 0DTE Greeks live path ────────────────────────────────────

    def _refresh_0dte_set(self) -> None:
        """Rebuild the set of instrument_ids that expire today (CT)."""
        today = datetime.now(_CT).date()
        self._0dte_session = today
        self._0dte_ids = {
            iid for iid, m in self._opt_meta.items() if m["expiry"] == today
        }
        logger.info("0dte_set_refreshed", extra={"date": str(today), "n": len(self._0dte_ids)})

    def _on_bbo_for_greeks(
        self,
        *,
        ts: datetime,
        instrument_id: int,
        symbol: str,
        bid: float | None,
        ask: float | None,
        bid_sz: int | None,
        ask_sz: int | None,
    ) -> None:
        """Update NQ mid / buffer 0DTE option quotes; flush on second boundary."""
        # Roll 0DTE set at CT midnight
        today = ts.astimezone(_CT).date()
        if self._0dte_session != today:
            self._refresh_0dte_set()

        sec = ts.replace(microsecond=0)
        if self._greeks_sec is not None and sec != self._greeks_sec:
            self._flush_greeks_second()

        self._greeks_sec = sec

        # Track front-month NQ future mid for the underlying
        if symbol and _NQ_FUT_RE.match(symbol) and bid and ask and bid > 0 and ask > 0:
            # Bound above too: the DBN UNDEF_PRICE sentinel is positive.
            if bid < _UNDEF_PX and ask < _UNDEF_PX:
                mid = 0.5 * (bid + ask)
                self._fut_mids[symbol] = mid
                self._nq_mid = mid
                self._nq_sym = symbol
            return

        if instrument_id not in self._0dte_ids:
            return
        if not (bid and ask and bid > 0 and ask > 0 and (bid_sz or 0) > 0 and (ask_sz or 0) > 0):
            return
        self._greeks_buf[instrument_id] = (float(bid), float(ask))

    def _flush_greeks_second(self, force: bool = False) -> None:
        """Compute + write Greeks/surface for the buffered second."""
        if self._greeks_sec is None or not self._greeks_buf:
            if force:
                self._greeks_buf.clear()
            return
        if not self._fut_mids:
            self._greeks_buf.clear()
            return

        try:
            from src.features.options_greeks_pipeline import (
                compute_second_batch,
                insert_second,
            )
            quotes = pd.DataFrame(
                [
                    {"instrument_id": iid, "bid_px": ba[0], "ask_px": ba[1]}
                    for iid, ba in self._greeks_buf.items()
                ]
            )
            greeks, surface_row = compute_second_batch(
                quotes,
                self._opt_meta,
                nq_price=dict(self._fut_mids),
                underlying_sym=self._nq_sym,
                ts=self._greeks_sec,
            )
            if not greeks.empty or surface_row is not None:
                with self._pool.connection() as conn:
                    insert_second(conn, greeks, surface_row)
                if surface_row is not None:
                    self._surface_parquet_rows.append(surface_row)
                    # Flush parquet every ~60 surface rows to limit memory
                    if len(self._surface_parquet_rows) >= 60:
                        self._flush_surface_parquet()
        except Exception:
            logger.exception("greeks_flush_failed", extra={"sec": str(self._greeks_sec)})
        finally:
            self._greeks_buf.clear()

    def _flush_surface_parquet(self) -> None:
        """Append accumulated surface rows to today's parquet partition."""
        if not self._surface_parquet_rows:
            return
        try:
            from src.features.options_greeks_pipeline import (
                _SURFACE_DB_COLS,
                write_surface_parquet,
            )
            df = pd.DataFrame(self._surface_parquet_rows)
            for c in _SURFACE_DB_COLS:
                if c not in df.columns:
                    df[c] = float("nan")
            df = df[_SURFACE_DB_COLS]
            # Merge with existing partition if present
            d = datetime.now(_CT).date()
            out_dir = options_surface_dir()
            path = out_dir / f"features_{d.isoformat()}.parquet"
            if path.exists():
                prev = pd.read_parquet(path)
                if prev.index.name == "ts" or "ts" not in prev.columns:
                    prev = prev.reset_index()
                if "ts" in prev.columns:
                    df = pd.concat([prev, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["ts"], keep="last")
            write_surface_parquet(df, d, out_dir)
            self._surface_parquet_rows.clear()
        except Exception:
            logger.exception("surface_parquet_flush_failed")

    def _insert_statistics(self, rec: db.StatMsg) -> None:
        sql = """
            INSERT INTO databento.statistics
                (ts_event, ts_recv, instrument_id, symbol, stat_type,
                 price, quantity, sequence, ts_ref, update_action, stat_flags)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """
        with self._pool.connection() as conn:
            conn.execute(sql, (
                _ns_to_dt(rec.ts_event), _ns_to_dt(rec.ts_recv), rec.instrument_id,
                self._resolve_symbol(rec.instrument_id),
                int(rec.stat_type),
                _px(rec.price) if hasattr(rec, "price") else None,
                int(rec.quantity) if hasattr(rec, "quantity") else None,
                getattr(rec, "sequence", None),
                _ns_to_dt(getattr(rec, "ts_ref", None)),
                _enum_int(getattr(rec, "update_action", None)),
                int(rec.stat_flags) if hasattr(rec, "stat_flags") else None,
            ))


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    from src.common.logging import setup_logging
    setup_logging()
    # Enable Databento's internal logger so gateway events appear in logs
    # (disabled by default per Databento docs recommendation).
    logging.getLogger("databento").setLevel(logging.WARNING)
    LiveStreamer().start()


if __name__ == "__main__":
    main()
