"""Bounded, append-only DBN recovery with full-payload multiset reconciliation.

Stage the entire archive before comparing occurrences. Reconciliation per decode
chunk would lose identical events split across chunks. Existing live timestamps
use Python's float-second conversion; use that same rounding at the DB boundary
while retaining exact nanoseconds in the immutable vendor archive.
"""
from __future__ import annotations

import csv
import io
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import databento as db
import numpy as np
import pandas as pd
from databento_dbn import UNDEF_PRICE, UNDEF_TIMESTAMP
from databento.common.error import BentoWarning
from psycopg import sql

from src.acquisition.gap_recovery import RecoveryRequest
from src.ingestion.loader import _BBO_DB_COLS, _TBBO_DB_COLS, get_dsn

STATS_COLUMNS = ["ts_event", "ts_recv", "instrument_id", "symbol", "stat_type",
                 "price", "quantity", "sequence", "ts_ref", "update_action", "stat_flags"]
LEGACY_UNDEF_PRICE = float(UNDEF_PRICE) / 1e9
LEGACY_UNDEF_REF = datetime.fromtimestamp(UNDEF_TIMESTAMP / 1e9, timezone.utc)


@dataclass(frozen=True)
class SchemaSpec:
    table: str
    columns: tuple[str, ...]
    time_column: str


SPECS = {
    "tbbo": SchemaSpec("databento.tbbo", tuple(_TBBO_DB_COLS) + ("dataset",), "ts_event"),
    "bbo-1s": SchemaSpec("databento.bbo_1s", tuple(_BBO_DB_COLS) + ("dataset",), "ts_recv"),
    "statistics": SchemaSpec("databento.statistics", tuple(STATS_COLUMNS) + ("dataset",), "ts_event"),
}


def enable_strict_dbn_validation() -> None:
    """Install the process-wide integrity guard before starting worker threads.

    SDK iteration otherwise warns and returns an apparently successful prefix
    when a DBN ends in an incomplete record. Scoped warning contexts race across
    Python 3.12 threads, so orchestration installs this specific policy once.
    """
    warnings.filterwarnings(
        "error", message=r"^DBN file is truncated or contains an incomplete record$",
        category=BentoWarning,
    )


def _wait_for_merge_lock(
    cur: Any, key: str, *, timeout_seconds: float = 600.0,
    progress: Callable[[str, int], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for another recovery writer without raising relation lock timeout.

    Nonblocking advisory probes leave the two-second relation lock protection
    intact. The Python deadline also bounds waits independently of SQL timeout.
    """
    if timeout_seconds <= 0:
        raise ValueError("positive recovery merge-lock deadline required")
    started = monotonic()
    deadline, next_progress = started + timeout_seconds, started + 5.0
    while True:
        cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0))", (key,))
        if cur.fetchone()[0]:
            return
        now = monotonic()
        if now >= deadline:
            raise TimeoutError(f"recovery merge lock wait exceeded {timeout_seconds:g}s: {key}")
        if progress and now >= next_progress:
            progress("waiting_merge", int(now - started))
            next_progress = now + 5.0
        sleep(min(0.15, deadline - now))


def _live_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True).dt.round("us")
    valid = values.notna() & values.ne(0) & values.ne(UNDEF_TIMESTAMP)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    floating = values.loc[valid].astype("float64") / 1e9
    seconds = np.floor(floating).astype("int64")
    micros = np.rint((floating - seconds) * 1e6).astype("int64")
    result.loc[valid] = pd.to_datetime(seconds, unit="s", utc=True) + pd.to_timedelta(micros, unit="us")
    return result


def prepare_frame(frame: pd.DataFrame, schema: str, *, fixed_prices: bool = False) -> pd.DataFrame:
    """Normalize source payload, including null sentinels and live DB precision."""
    spec = SPECS[schema]
    data = frame.reset_index().rename(columns={
        "bid_px_00": "bid_px", "ask_px_00": "ask_px",
        "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz",
        "bid_ct_00": "bid_ct", "ask_ct_00": "ask_ct",
    })
    data["dataset"] = "GLBX.MDP3"
    for name in ("ts_event", "ts_recv", "ts_ref"):
        if name in spec.columns:
            if name not in data:
                data[name] = pd.NaT
            data[name] = _live_timestamps(data[name])
    for name in ("price", "bid_px", "ask_px"):
        if name in spec.columns:
            values = pd.to_numeric(data[name], errors="coerce")
            if fixed_prices:
                values = values.mask(values.ge(UNDEF_PRICE)).astype("float64") / 1e9
            else:
                values = values.mask(values.eq(LEGACY_UNDEF_PRICE))
            data[name] = values
    if "quantity" in data:
        # Preserve integers above 2**53 when an undefined neighbour becomes NULL.
        # Plain int64.mask() promotes the entire column to lossy float64.
        quantities = data["quantity"].astype("Int64")
        data["quantity"] = quantities.mask(quantities.eq(UNDEF_PRICE))
    for name in ("side", "action"):
        if name in spec.columns:
            data[name] = data[name].astype("string").str[:1]
    for name in spec.columns:
        if name not in data:
            data[name] = None
    if data[spec.time_column].isna().any() or data["ts_recv"].isna().any():
        raise ValueError("source record has no valid event/receive timestamp")
    if data["symbol"].isna().any() or data["symbol"].astype(str).eq("").any():
        raise ValueError("archive contains an unmapped symbol; static ID fallback is forbidden")
    return data.loc[:, list(spec.columns)]


def _validate_receive_window(frame: pd.DataFrame, start: pd.Timestamp,
                             end: pd.Timestamp) -> None:
    """Check vendor nanoseconds before lossy PostgreSQL/live rounding.

    A received event one nanosecond before the exclusive end can round to that
    end in PostgreSQL. It is still part of this exact vendor request.
    """
    received = (pd.Series(frame.index.to_numpy()) if frame.index.name == "ts_recv"
                else frame["ts_recv"])
    if not pd.api.types.is_integer_dtype(received):
        raise ValueError("recovery requires exact integer receive timestamps")
    if (received.lt(start.value) | received.ge(end.value)).any():
        raise ValueError("DBN delivered a record outside the requested receive-time interval")


@dataclass(frozen=True)
class IngestReceipt:
    request_id: str
    records: int
    inserted: int
    matched: int
    segments: int
    first_timestamp: str | None
    last_timestamp: str | None


def _identifier(name: str) -> sql.Identifier:
    pieces = name.split(".")
    if len(pieces) != 2 or not all(piece.replace("_", "").isalnum() for piece in pieces):
        raise ValueError("table must be a qualified identifier")
    return sql.Identifier(*pieces)


def _target_projection(columns: tuple[str, ...]) -> sql.Composed:
    projected = []
    for name in columns:
        identifier = sql.Identifier(name)
        if name in ("price", "bid_px", "ask_px"):
            projected.append(sql.SQL("CASE WHEN {} = {} THEN NULL ELSE {} END").format(
                identifier, sql.Literal(LEGACY_UNDEF_PRICE), identifier))
        elif name == "quantity":
            projected.append(sql.SQL("CASE WHEN {} = {} THEN NULL ELSE {} END").format(
                identifier, sql.Literal(UNDEF_PRICE), identifier))
        elif name == "ts_ref":
            projected.append(sql.SQL("CASE WHEN {} = {} THEN NULL ELSE {} END").format(
                identifier, sql.Literal(LEGACY_UNDEF_REF), identifier))
        else:
            projected.append(identifier)
    return sql.SQL(",").join(projected)


def ingest_recovery_file(
    path: Path | str, request: RecoveryRequest, *,
    connection_factory: Callable[[], Any] | None = None,
    chunk_size: int = 200_000, segment_seconds: int = 3600,
    target_override: str | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> IngestReceipt:
    """COPY into a session TEMP table, then append missing payload occurrences.

    Commits are bounded by disjoint time segments. Retrying after a partial
    completion reconciles already committed segments. No target rows, indexes,
    compression settings, or existing raw files are removed or rewritten.
    """
    import psycopg

    spec = SPECS[request.schema]
    if chunk_size < 1 or segment_seconds < 1:
        raise ValueError("positive ingestion bounds required")
    target = _identifier(target_override or spec.table)
    stage = sql.Identifier("recovery_stage_" + uuid4().hex)
    columns = sql.SQL(",").join(map(sql.Identifier, spec.columns))
    store = db.DBNStore.from_file(path)
    if str(store.schema) != request.schema or str(store.dataset) != request.dataset:
        raise ValueError("DBN schema/dataset differs from the reviewed request")
    count = inserted = segments = 0
    existing = sql.Identifier("recovery_existing_" + uuid4().hex)
    start, end = pd.Timestamp(request.start), pd.Timestamp(request.end)
    factory = connection_factory or (lambda: psycopg.connect(get_dsn()))
    with factory() as conn:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '2s'")
            cur.execute("SET statement_timeout = '180s'")
            cur.execute("SET work_mem = '128MB'")
            cur.execute(sql.SQL("CREATE TEMP TABLE {} AS SELECT {} FROM {} WITH NO DATA").format(stage, columns, target))
            for frame in store.to_df(schema=request.schema, map_symbols=True,
                                     pretty_ts=False, price_type="fixed", count=chunk_size):
                _validate_receive_window(frame, start, end)
                prepared = prepare_frame(frame, request.schema, fixed_prices=True)
                prepared["dataset"] = request.dataset
                if request.stype_in == "raw_symbol" and not prepared["symbol"].isin(request.symbols).all():
                    raise ValueError("DBN delivered an unrequested raw symbol")
                # NumPy formats UTC timestamps in C. Pandas' default CSV path
                # constructs a Python Timestamp string for every BBO row.
                # The microsecond representation is identical at the SQL
                # boundary; source nanoseconds remain in the raw archive.
                csv_frame = prepared.copy(deep=False)
                for name in ("ts_event", "ts_recv", "ts_ref"):
                    if name not in spec.columns:
                        continue
                    timestamps = prepared[name].values.astype("datetime64[us]")
                    rendered = np.datetime_as_string(timestamps, unit="us", timezone="UTC").astype(object)
                    rendered[np.isnat(timestamps)] = None
                    csv_frame[name] = rendered
                buffer = io.StringIO()
                csv_frame.to_csv(buffer, header=False, index=False, na_rep="", quoting=csv.QUOTE_MINIMAL)
                with cur.copy(sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV)").format(stage, columns)) as copy:
                    copy.write(buffer.getvalue())
                count += len(prepared)
                if progress:
                    progress("staging", count)
            cur.execute(sql.SQL("CREATE INDEX ON {} ({})").format(stage, sql.Identifier(spec.time_column)))
            cur.execute(sql.SQL("ANALYZE {}").format(stage))
            cur.execute(sql.SQL("SELECT min({}),max({}) FROM {}").format(
                sql.Identifier(spec.time_column), sql.Identifier(spec.time_column), stage))
            first, last = cur.fetchone()
            conn.commit()
            if not count:
                return IngestReceipt(request.request_id, 0, 0, 0, 0, None, None)
            # bbo_1s deliberately has no time index. Read each relevant target
            # chunk once, rather than re-scanning it for every hourly segment.
            # Serialize recovery merges for this target; parallel TEMP staging
            # still uses available cores, and live inserts continue normally.
            _wait_for_merge_lock(cur, "gap_recovery:" + (target_override or spec.table),
                                 progress=progress)
            cur.execute(sql.SQL("SELECT array_agg(DISTINCT symbol) FROM {}").format(stage))
            all_names = cur.fetchone()[0]
            cur.execute(sql.SQL("CREATE TEMP TABLE {} AS SELECT ").format(existing)
                        + _target_projection(spec.columns)
                        + sql.SQL(" FROM {} WHERE {} >= %s AND {} <= %s AND symbol = ANY(%s) AND dataset = %s").format(
                            target, sql.Identifier(spec.time_column), sql.Identifier(spec.time_column)),
                        (first, last, all_names, request.dataset))
            cur.execute(sql.SQL("CREATE INDEX ON {} ({})").format(existing, sql.Identifier(spec.time_column)))
            cur.execute(sql.SQL("ANALYZE {}").format(existing))
            conn.commit()
            left = pd.Timestamp(first).floor(f"{segment_seconds}s").to_pydatetime()
            while left <= last:
                right = left + pd.Timedelta(seconds=segment_seconds)
                # The session advisory lock already serializes all recovery
                # segments for this target; live inserts remain independent.
                cur.execute(sql.SQL("SELECT count(*),array_agg(DISTINCT symbol) FROM {} WHERE {} >= %s AND {} < %s").format(
                    stage, sql.Identifier(spec.time_column), sql.Identifier(spec.time_column)), (left, right))
                staged_count, names = cur.fetchone()
                if not staged_count:
                    conn.commit()
                    left = right
                    continue
                where = sql.SQL("{} >= %s AND {} < %s AND symbol = ANY(%s) AND dataset = %s").format(
                    sql.Identifier(spec.time_column), sql.Identifier(spec.time_column))
                cur.execute(sql.SQL("SELECT count(*) FROM {} WHERE ").format(existing) + where,
                            (left, right, names, request.dataset))
                existing_count = cur.fetchone()[0]
                source = sql.SQL("SELECT {} FROM {} WHERE {} >= %s AND {} < %s").format(
                    columns, stage, sql.Identifier(spec.time_column), sql.Identifier(spec.time_column))
                params: tuple = (left, right)
                if existing_count:
                    source = sql.SQL("(") + source + sql.SQL(") EXCEPT ALL (SELECT ") + columns + sql.SQL(" FROM ") + existing + sql.SQL(" WHERE ") + where + sql.SQL(")")
                    params += (left, right, names, request.dataset)
                statement = sql.SQL("INSERT INTO {} ({}) ").format(target, columns) + source
                # A statistics key with conflicting payload must fail loudly;
                # ON CONFLICT DO NOTHING would falsely claim completeness.
                cur.execute(statement, params)
                inserted += cur.rowcount
                segments += 1
                conn.commit()
                if progress:
                    progress("merged", inserted)
                left = right
            cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))",
                        ("gap_recovery:" + (target_override or spec.table),))
            conn.commit()
    return IngestReceipt(request.request_id, count, inserted, count - inserted, segments,
                         first.isoformat(), last.isoformat())
