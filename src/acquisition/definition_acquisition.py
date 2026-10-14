"""Vendor definition acquisition alongside live raw-symbol subscriptions.

No historical ID is used to identify a live quote. Definitions are keyed by raw
symbol; the streamer attaches their terms to the current gateway mapping.
Dependencies are injectable so the lifecycle can be tested without a database.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import astuple, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import databento as db
import pandas as pd
from databento_dbn import FIXED_PRICE_SCALE, UNDEF_PRICE, UNDEF_TIMESTAMP

from src.acquisition.schemas import DATASET
from src.common.vm_config import get_vm_config

logger = logging.getLogger("vtech.definition_acquisition")

DEFINITION_COLUMNS = (
    "ts_event", "instrument_id", "symbol", "instrument_class", "strike_price",
    "expiration", "underlying", "exchange", "currency", "min_price_increment",
    "multiplier", "source",
)
_COLUMNS_SQL = "(" + ",".join(DEFINITION_COLUMNS) + ")"

# Keep established vendor-over-synthetic provenance rules. Python normalization
# removes NaN/sentinels before SQL; the SQL guard also protects older callers.
DEFINITION_UPSERT_TAIL = """
    ON CONFLICT (symbol) DO UPDATE SET
        ts_event = EXCLUDED.ts_event,
        instrument_id = EXCLUDED.instrument_id,
        instrument_class = EXCLUDED.instrument_class,
        strike_price = EXCLUDED.strike_price,
        expiration = EXCLUDED.expiration,
        underlying = EXCLUDED.underlying,
        exchange = EXCLUDED.exchange,
        currency = EXCLUDED.currency,
        min_price_increment = CASE
            WHEN EXCLUDED.min_price_increment IS NULL
              OR EXCLUDED.min_price_increment <> EXCLUDED.min_price_increment
              OR EXCLUDED.min_price_increment = 'NaN'::float8
              OR EXCLUDED.min_price_increment > 1e6
            THEN databento.definitions.min_price_increment
            ELSE EXCLUDED.min_price_increment END,
        multiplier = COALESCE(EXCLUDED.multiplier, databento.definitions.multiplier),
        source = EXCLUDED.source
    WHERE (databento.definitions.source = 'synthetic'
      OR (databento.definitions.source = 'databento'
          AND EXCLUDED.ts_event >= databento.definitions.ts_event
          AND (databento.definitions.instrument_class IS NULL
               OR databento.definitions.instrument_class NOT IN ('C', 'P')
               OR databento.definitions.strike_price IS NULL
               OR databento.definitions.strike_price <= 0
               OR databento.definitions.strike_price = 'NaN'::float8
               OR databento.definitions.strike_price = 'Infinity'::float8
               OR databento.definitions.expiration IS NULL
               OR databento.definitions.underlying IS NULL
               OR databento.definitions.underlying = '')))
      AND EXCLUDED.source <> 'synthetic'
"""


def _timestamp(value: Any) -> datetime | None:
    if value is None or int(value) in (0, UNDEF_TIMESTAMP):
        return None
    return pd.Timestamp(int(value), tz="UTC").to_pydatetime(warn=False)


def _price(value: Any, *, positive: bool = False) -> float | None:
    if value is None or not math.isfinite(float(value)) or float(value) >= UNDEF_PRICE:
        return None
    result = float(value) / float(FIXED_PRICE_SCALE)
    if not math.isfinite(result) or (positive and (result <= 0 or result > 1e6)):
        return None
    return result


@dataclass(frozen=True)
class VendorDefinition:
    ts_event: datetime
    instrument_id: int
    symbol: str
    instrument_class: str | None
    strike_price: float | None
    expiration: datetime | None
    underlying: str | None
    exchange: str | None
    currency: str | None
    min_price_increment: float | None
    multiplier: float | None
    source: str = "databento"

    @classmethod
    def from_record(cls, record: Any) -> VendorDefinition:
        """Normalize a DBN definition; absent tick/size remain absent."""
        symbol = str(record.raw_symbol).strip()
        ts = _timestamp(record.ts_event)
        if not symbol or ts is None:
            raise ValueError("definition requires raw_symbol and publication timestamp")
        raw_class = getattr(record, "instrument_class", None)
        iclass = str(getattr(raw_class, "value", raw_class)).upper() if raw_class is not None else None
        size = getattr(record, "unit_of_measure_qty", getattr(record, "multiplier", None))
        return cls(
            ts, int(record.instrument_id), symbol, iclass,
            _price(getattr(record, "strike_price", None)),
            _timestamp(getattr(record, "expiration", None)),
            getattr(record, "underlying", None) or None,
            getattr(record, "exchange", None) or None,
            getattr(record, "currency", None) or None,
            _price(getattr(record, "min_price_increment", None), positive=True),
            _price(size, positive=True),
        )

    @classmethod
    def from_row(cls, row: tuple) -> VendorDefinition:
        """Read a persisted row in DEFINITION_COLUMNS order."""
        return cls(*row)

    def row(self) -> tuple:
        """Parameterized SQL values; every acquisition write is vendor-backed."""
        if self.source != "databento":
            raise ValueError("definition acquisition cannot write synthetic rows")
        return astuple(self)

    @property
    def option_ready(self) -> bool:
        """Minimum vendor terms required for live option pricing."""
        return (
            self.source == "databento" and self.instrument_class in ("C", "P")
            and self.strike_price is not None and math.isfinite(self.strike_price)
            and self.strike_price > 0 and self.expiration is not None
            and bool(self.underlying)
        )


def read_definitions(pool: Any, symbols: Iterable[str]) -> dict[str, VendorDefinition]:
    """Indexed raw-symbol lookup; return only committed authoritative rows."""
    names = sorted(set(symbols))
    if not names:
        return {}
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT " + ",".join(DEFINITION_COLUMNS)
            + " FROM databento.definitions WHERE symbol = ANY(%s) AND source = %s",
            (names, "databento"),
        ).fetchall()
    result = {}
    for row in rows:
        definition = VendorDefinition.from_row(row)
        if definition.option_ready:
            result[definition.symbol] = definition
        else:
            logger.warning("definition_acquisition_terms_incomplete", extra={"symbol": definition.symbol})
    return result


class DefinitionAcquirer:
    """Nonblocking subscription companion with bounded, zero-cost-only fetches."""

    def __init__(
        self, pool: Any, *, client_factory: Callable[[], Any] | None = None,
        workers: int | None = None, retry_s: float = 300.0, batch_size: int = 100,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._client_factory = client_factory or db.Historical
        self._workers = workers if workers is not None else get_vm_config().workers_for(mem_per_worker_gb=0.5)
        if self._workers < 1 or not 1 <= batch_size <= 2000 or retry_s <= 0:
            raise ValueError("invalid definition acquisition worker/batch/retry settings")
        self._batch_size = batch_size
        self._retry_s = retry_s
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._definitions: dict[str, VendorDefinition] = {}
        self._pending: dict[str, float] = {}
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._stopped = False
        self.database_checked = False
        self._thread: threading.Thread | None = None

    def get(self, symbol: str) -> VendorDefinition | None:
        """Lookup by symbol without copying the complete population per quote."""
        with self._condition:
            return self._definitions.get(symbol)

    @property
    def definitions(self) -> dict[str, VendorDefinition]:
        with self._condition:
            return dict(self._definitions)

    @property
    def pending(self) -> frozenset[str]:
        with self._condition:
            return frozenset(self._pending)

    def submit(self, symbols: Iterable[str]) -> None:
        """Schedule exact symbols; repeated subscription refreshes coalesce."""
        with self._condition:
            if self._stopped:
                return
            for symbol in symbols:
                if symbol not in self._definitions:
                    self._pending.setdefault(symbol, 0.0)
            self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="vtech-definition-acquisition", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError("definition_acquisition_shutdown_timeout")

    def acquire(self, symbols: Iterable[str]) -> dict[str, VendorDefinition]:
        """One acquisition pass; injectable APIs/DB make this independently testable."""
        names = sorted(set(symbols))
        existing = read_definitions(self._pool, names)
        with self._condition:
            self.database_checked = True
            self._definitions.update(existing)
        missing = [s for s in names if s not in existing]
        batches = [missing[i:i + self._batch_size] for i in range(0, len(missing), self._batch_size)]
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            futures = [executor.submit(self._fetch_batch, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    acquired = future.result()
                    existing.update(acquired)
                    with self._condition:
                        self._definitions.update(acquired)
                except Exception:
                    logger.exception("definition_acquisition_batch_failed")
        with self._condition:
            self._definitions.update(existing)
        return existing

    def _fetch_batch(self, symbols: list[str]) -> dict[str, VendorDefinition]:
        # Recheck immediately before querying: a live definition or external
        # acquisition may already have committed. Release the DB connection
        # before all network calls.
        if self._stopped:
            return {}
        existing = read_definitions(self._pool, symbols)
        names = [s for s in symbols if s not in existing]
        if not names:
            return existing
        client = self._client_factory()
        # Instance endpoint knobs from the installed SDK; do not change its
        # global HTTP defaults or slow unrelated historical jobs.
        client.metadata.TIMEOUT = 10
        client.timeseries.TIMEOUT = 10
        if self._stopped:
            return existing
        availability = client.metadata.get_dataset_range(dataset=DATASET)
        schema_range = availability.get("schema_ranges", {}).get("definition", {})
        available_end = pd.Timestamp(schema_range.get("end") or availability["end"]).to_pydatetime()
        end = min(available_end, self._clock()).astimezone(timezone.utc)
        # Cover the latest partial UTC day and its preceding daily snapshot.
        start = end.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        available_start = schema_range.get("start") or availability.get("start")
        if available_start:
            start = max(start, pd.Timestamp(available_start).to_pydatetime())
        if start >= end:
            return existing
        request = dict(dataset=DATASET, schema="definition", symbols=names,
                       stype_in="raw_symbol", start=start.isoformat(), end=end.isoformat())
        if self._stopped:
            return existing
        quoted = client.metadata.get_cost(**request)
        try:
            cost = Decimal(str(quoted))
            free = cost.is_finite() and cost == Decimal(0)
        except (InvalidOperation, ValueError):
            free = False
        if not free:
            logger.warning("definition_acquisition_cost_blocked", extra={"cost_usd": str(quoted), "n_symbols": len(names)})
            return existing
        if self._stopped:
            return existing
        logger.info("definition_acquisition_fetch", extra={"n_symbols": len(names), "cost_usd": "0.00", "start": start.isoformat(), "end": end.isoformat()})
        store = client.timeseries.get_range(**request)
        if self._stopped:
            return existing
        latest: dict[str, VendorDefinition] = {}
        allowed = set(names)
        for record in store:
            if not hasattr(record, "raw_symbol"):
                continue
            if str(record.raw_symbol).strip() not in allowed:
                continue
            definition = VendorDefinition.from_record(record)
            received = _timestamp(getattr(record, "ts_recv", None)) or definition.ts_event
            if received >= end or definition.ts_event >= end:
                continue
            if not definition.option_ready:
                logger.warning("definition_acquisition_terms_incomplete", extra={"symbol": definition.symbol})
                continue
            previous = latest.get(definition.symbol)
            if previous is None or definition.ts_event >= previous.ts_event:
                latest[definition.symbol] = definition
        if latest:
            # No transaction spans a vendor fetch. A concurrent vendor write
            # wins over this historical insert through the provenance guard.
            with self._write_lock:
                if self._stopped:
                    return existing
                with self._pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT set_config('statement_timeout', %s, true), "
                            "set_config('lock_timeout', %s, true)",
                            ("15000", "2000"),
                        )
                        cur.executemany(
                            "INSERT INTO databento.definitions " + _COLUMNS_SQL
                            + " VALUES (" + ",".join(["%s"] * len(DEFINITION_COLUMNS)) + ") "
                            + DEFINITION_UPSERT_TAIL,
                            [definition.row() for definition in latest.values()],
                        )
            # Reread after commit: attach persisted winning terms, not stale
            # historical values when a live callback won the race.
            existing.update(read_definitions(self._pool, latest))
        logger.info("definition_acquisition_result", extra={"n_requested": len(names), "n_vendor": len(latest), "n_pending": len(set(names) - set(existing))})
        return existing

    def _run(self) -> None:
        import time

        while True:
            with self._condition:
                if self._stopped:
                    return
                now = time.monotonic()
                due = [symbol for symbol, deadline in self._pending.items() if deadline <= now]
                if not due:
                    delay = min((deadline - now for deadline in self._pending.values()), default=self._retry_s)
                    self._condition.wait(timeout=max(0.01, delay))
                    continue
            try:
                acquired = self.acquire(due)
            except Exception:
                acquired = {}
                logger.exception("definition_acquisition_failed")
            with self._condition:
                for symbol in due:
                    if symbol in acquired:
                        self._pending.pop(symbol, None)
                    else:
                        self._pending[symbol] = time.monotonic() + self._retry_s
                logger.info("definition_acquisition_coverage", extra={"n_available": len(self._definitions), "n_pending": len(self._pending)})
