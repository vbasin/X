"""DBN file → TimescaleDB loader.

Optimized for large files (100M+ rows):
- COPY-based bulk loading via staging table (10-50x faster than executemany)
- Vectorized pandas → CSV (no Python row loops)
- Chunked DataFrameIterator: never loads full file into memory
- ON CONFLICT DO NOTHING for idempotent re-runs
- Live progress bar with rows/s and ETA
"""
from __future__ import annotations

import csv
import io
import os
import queue
import sys
import threading
import time
import logging
from pathlib import Path

import databento as db
import psycopg
from dotenv import load_dotenv

from src.common.vm_config import get_vm_config

load_dotenv("/opt/vtech/.env")
logger = logging.getLogger("vtech.loader")

# Empirical compressed bytes-per-row for progress estimation
_BYTES_PER_ROW_EST = {"bbo-1s": 12, "tbbo": 25}


def _mem_settings() -> tuple[str, str]:
    """Deprecated shim — use ``get_vm_config()`` directly.

    Returns (work_mem, maintenance_work_mem) for backward compatibility.
    """
    cfg = get_vm_config()
    return cfg.work_mem, cfg.maintenance_work_mem


def get_dsn():
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ.get('PGHOST','localhost')}:{os.environ.get('PGPORT','5432')}"
        f"/{os.environ.get('PGDATABASE','vtech_market')}"
    )


def _estimate_total(path: str, schema: str) -> int | None:
    """Estimate total rows from compressed file size."""
    bpr = _BYTES_PER_ROW_EST.get(schema)
    if bpr:
        return os.path.getsize(path) // bpr
    return None


def _progress_bar(current: int, total: int | None, t0: float,
                  new_rows: int, width: int = 40) -> str:
    """Format a live terminal progress bar."""
    elapsed = time.time() - t0
    rate = current / elapsed if elapsed > 0 else 0

    if total and total > 0:
        pct = min(current / total, 1.0)
        filled = int(width * pct)
        bar = '█' * filled + '░' * (width - filled)
        eta = (total - current) / rate if rate > 0 else 0
        eta_m, eta_s = divmod(int(eta), 60)
        return (
            f"\r  [{bar}] {pct:6.1%}  "
            f"{current:>12,} / ~{total:,}  "
            f"({new_rows:,} new)  "
            f"{rate:,.0f} rows/s  ETA {eta_m}m{eta_s:02d}s"
        )
    else:
        m, s = divmod(int(elapsed), 60)
        return (
            f"\r  {current:>12,} rows  "
            f"({new_rows:,} new)  "
            f"{rate:,.0f} rows/s  {m}m{s:02d}s elapsed"
        )


def _copy_chunk(cur, conn, table: str, db_cols: list[str], df_prepped) -> int:
    """COPY a DataFrame chunk directly into target table.

    Bypasses staging table — safe when no UNIQUE constraint exists on target.
    10-50x faster than executemany(); uses PG bulk COPY protocol with
    vectorized pandas CSV generation (no Python row loops).
    """
    cols_sql = ", ".join(db_cols)

    # Vectorized CSV generation (pandas C engine)
    buf = io.BytesIO()
    df_prepped.to_csv(
        buf, header=False, index=False, na_rep='',
        quoting=csv.QUOTE_MINIMAL, encoding='utf-8',
    )
    buf.seek(0)

    # Direct COPY into target — no staging round-trip
    with cur.copy(
        f"COPY {table} ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '')"
    ) as copy:
        while block := buf.read(1 << 18):  # 256 KB blocks
            copy.write(block)

    inserted = cur.rowcount
    conn.commit()
    return inserted


# ── Schema-specific column prep (vectorized, no iterrows) ──────

_TBBO_RENAME = {
    'bid_px_00': 'bid_px', 'ask_px_00': 'ask_px',
    'bid_sz_00': 'bid_sz', 'ask_sz_00': 'ask_sz',
    'bid_ct_00': 'bid_ct', 'ask_ct_00': 'ask_ct',
}
_TBBO_DB_COLS = [
    'ts_event', 'ts_recv', 'instrument_id', 'symbol', 'price', 'size',
    'side', 'action', 'flags', 'sequence',
    'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 'bid_ct', 'ask_ct', 'ts_in_delta',
]


def _prep_tbbo(df):
    """Vectorized TBBO DataFrame preparation."""
    df = df.reset_index().rename(columns=_TBBO_RENAME)
    df = df[df['ts_event'].notna()].copy()
    if 'side' in df.columns:
        df['side'] = df['side'].str[:1]
    if 'action' in df.columns:
        df['action'] = df['action'].str[:1]
    if 'ts_in_delta' not in df.columns:
        df['ts_in_delta'] = None
    return df[_TBBO_DB_COLS]


_BBO_RENAME = {
    'bid_px_00': 'bid_px', 'ask_px_00': 'ask_px',
    'bid_sz_00': 'bid_sz', 'ask_sz_00': 'ask_sz',
}
_BBO_DB_COLS = [
    'ts_recv', 'instrument_id', 'symbol',
    'price', 'size',
    'bid_px', 'ask_px', 'bid_sz', 'ask_sz',
]


def _prep_bbo_1s(df):
    """Vectorized BBO-1s DataFrame preparation."""
    df = df.reset_index().rename(columns=_BBO_RENAME)
    return df[_BBO_DB_COLS]


# ── Generic COPY-based ingestion engine ─────────────────────────

def _ingest_chunked(dbn_path: str, table: str, db_cols: list[str],
                    prep_fn, label: str, schema: str,
                    total_hint: int | None = None,
                    row_filter=None) -> int:
    """COPY-based chunked ingestion with live progress bar.

    Uses a producer-consumer threading model:
      - Main thread: decompress + pandas prep (CPU-bound, one core)
      - DB worker thread: COPY + commit (I/O-bound, network to PG)
    These two phases run concurrently — DB writes overlap with decompression
    of the next chunk instead of alternating serially.
    """
    store = db.DBNStore.from_file(dbn_path)
    total_est = total_hint or _estimate_total(dbn_path, schema)
    _chunk_size = get_vm_config().ingest_chunk_size(
        bytes_per_row=_BYTES_PER_ROW_EST.get(schema, 100)
    )
    chunks = store.to_df(
        pretty_ts=True, map_symbols=True, price_type="float", count=_chunk_size,
    )

    # Static instrument_id → symbol fallback: bypasses map_symbols date-boundary
    # gaps caused by mid-file CME definition updates or end_date exclusivity.
    # CME never reuses instrument_ids, so date-agnostic lookup is safe.
    _id_to_sym: dict[int, str] = {
        int(iv['symbol']): sym
        for sym, intervals in store.symbology.get('mappings', {}).items()
        for iv in intervals
    }

    total_read = 0
    inserted_counter = [0]  # shared mutable; written only by db_worker
    t0 = time.time()

    est_str = f" (~{total_est:,} est)" if total_est else ""
    logger.info(f"Starting {label} ingestion{est_str}: {dbn_path}")
    print(f"\n  {label} ingestion{est_str}")

    # ── DB worker thread ──────────────────────────────────────────
    # Bounded queue (maxsize=2): main thread stays at most 2 chunks ahead,
    # providing backpressure so memory stays bounded regardless of chunk size.
    write_q: queue.Queue = queue.Queue(maxsize=2)
    db_exc: list[BaseException | None] = [None]
    _cfg = get_vm_config()
    _work_mem = _cfg.work_mem

    def _db_worker() -> None:
        try:
            # autocommit: each COPY is its own transaction, so the connection is
            # plain "idle" between chunks instead of "idle in transaction". A
            # lingering transaction here holds a snapshot, which stops VACUUM from
            # reclaiming space anywhere in the database — including the chunks a
            # concurrent backfill process just folded. There is nothing to make
            # atomic across chunks (the loader already commits per chunk), so the
            # open transaction bought nothing and blocked reclaim.
            with psycopg.connect(get_dsn(), autocommit=True) as conn:
                cur = conn.cursor()
                cur.execute("SET synchronous_commit = off")
                cur.execute(f"SET work_mem = '{_work_mem}'")
                while True:
                    item = write_q.get()
                    if item is None:  # sentinel — all chunks delivered
                        write_q.task_done()
                        break
                    n = _copy_chunk(cur, conn, table, db_cols, item)
                    inserted_counter[0] += n
                    write_q.task_done()
        except BaseException as exc:  # noqa: BLE001
            db_exc[0] = exc
            # Drain queue so the main thread doesn't deadlock on a full queue
            while True:
                try:
                    write_q.get_nowait()
                    write_q.task_done()
                except queue.Empty:
                    break

    worker = threading.Thread(target=_db_worker, name="tbbo-db-writer", daemon=True)
    worker.start()

    # ── Main thread: decompress + prep + enqueue ──────────────────
    try:
        for df in chunks:
            prepped = prep_fn(df)
            if 'symbol' in prepped.columns and _id_to_sym:
                blank = prepped['symbol'] == ''
                if blank.any():
                    prepped.loc[blank, 'symbol'] = (
                        prepped.loc[blank, 'instrument_id']
                        .map(_id_to_sym)
                        .fillna('')
                    )
            # Apply symbol-based row filter AFTER the patch so blank rows are
            # correctly classified before any inclusion/exclusion decision.
            if row_filter is not None:
                prepped = prepped[row_filter(prepped)].reset_index(drop=True)

            total_read += len(df)

            if len(prepped) == 0:
                continue

            # Enqueue with timeout loop so we can detect worker death
            while True:
                if db_exc[0] is not None:
                    raise db_exc[0]
                try:
                    write_q.put(prepped, timeout=1.0)
                    break
                except queue.Full:
                    continue

            bar = _progress_bar(total_read, total_est, t0, inserted_counter[0])
            sys.stdout.write(bar)
            sys.stdout.flush()
    finally:
        write_q.put(None)  # sentinel — always sent so worker can exit cleanly
        worker.join()

    if db_exc[0] is not None:
        raise db_exc[0]

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    rate = total_read / elapsed if elapsed > 0 else 0
    sys.stdout.write('\n')
    summary = (
        f"{label} done: {total_read:,} read, {inserted_counter[0]:,} new "
        f"in {m}m{s:02d}s ({rate:,.0f} rows/s)"
    )
    logger.info(summary)
    print(f"  {summary}")
    return inserted_counter[0]


def ingest_tbbo(dbn_path: str, batch_size: int = 10_000,
                total_hint: int | None = None,
                include_spreads: bool = False) -> int:
    """Load a TBBO .dbn.zst file into databento.tbbo.

    Args:
        include_spreads: If False (default), calendar-spread instruments
            (symbols containing '-', e.g. 'NQM6-NQU6') are dropped before
            insert.  Spreads are never queried by any feature module so
            excluding them saves disk space without any functional impact.

    Drops idx_tbbo_symbol_ts before ingest and rebuilds it after.
    Batch index build is 3-5x faster than per-row incremental updates.
    """
    _row_filter = None
    if not include_spreads:
        # Applied after the symbol-patch step so blank-symbol rows are
        # correctly resolved before the spread check.
        _row_filter = lambda df: ~df['symbol'].str.contains('-', na=False)  # noqa: E731
        print("  Spread filter active: calendar-spread rows will be dropped.")
    # Drop index + set UNLOGGED before ingest:
    #   - No index: incremental B-tree updates during 80M+ row ingest are slow;
    #     a single post-ingest CREATE INDEX is 3-5x faster.
    #   - UNLOGGED: skips WAL entirely during bulk load (2-3x write speedup);
    #     safe here because the source file is always available for re-ingest.
    print("  Dropping idx_tbbo_symbol_ts + setting UNLOGGED for bulk load...")
    with psycopg.connect(get_dsn()) as conn:
        conn.execute("DROP INDEX IF EXISTS databento.idx_tbbo_symbol_ts")
        conn.execute("ALTER TABLE databento.tbbo SET UNLOGGED")
        conn.commit()

    try:
        result = _ingest_chunked(
            dbn_path, 'databento.tbbo', _TBBO_DB_COLS,
            _prep_tbbo, 'TBBO', 'tbbo', total_hint,
            row_filter=_row_filter,
        )
    finally:
        # Always restore LOGGED even on failure — unlogged tables are truncated
        # on crash recovery, so leaving it unlogged after ingest would be unsafe.
        print("  Restoring databento.tbbo to LOGGED...")
        with psycopg.connect(get_dsn()) as conn:
            conn.execute("ALTER TABLE databento.tbbo SET LOGGED")
            conn.commit()

    # Rebuild index with generous maintenance_work_mem for a fast sort-based build
    _maint_mem = get_vm_config().maintenance_work_mem
    print(f"  Rebuilding idx_tbbo_symbol_ts (maintenance_work_mem={_maint_mem})...")
    t_idx = time.time()
    with psycopg.connect(get_dsn()) as conn:
        conn.execute(f"SET maintenance_work_mem = '{_maint_mem}'")
        conn.execute(
            "CREATE INDEX idx_tbbo_symbol_ts "
            "ON databento.tbbo (symbol, ts_event DESC)"
        )
        conn.commit()
    print(f"  Index rebuilt in {time.time() - t_idx:.1f}s")
    return result


def ingest_bbo_1s(dbn_path: str, batch_size: int = 10_000,
                  total_hint: int | None = None) -> int:
    """Load a BBO-1s .dbn.zst file into databento.bbo_1s."""
    return _ingest_chunked(
        dbn_path, 'databento.bbo_1s', _BBO_DB_COLS,
        _prep_bbo_1s, 'BBO-1s', 'bbo-1s', total_hint,
    )


def ingest_definition(dbn_path: str, batch_size: int = 5000) -> int:
    """Load a Definition .dbn.zst file into databento.definitions table."""
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df(pretty_ts=True, map_symbols=True, price_type="float")

    logger.info(f"Loaded {len(df):,} definition records from {dbn_path}")
    df = df.reset_index()

    # source='databento': an authoritative vendor row, which the upsert lets
    # overwrite a row the live streamer reconstructed from a quote symbol.
    insert_sql = """
        INSERT INTO databento.definitions
            (ts_event, instrument_id, symbol, instrument_class, strike_price,
             expiration, underlying, exchange, currency, min_price_increment,
             multiplier, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'databento')
        ON CONFLICT (symbol) DO UPDATE SET
            ts_event            = EXCLUDED.ts_event,
            instrument_id       = EXCLUDED.instrument_id,
            instrument_class    = EXCLUDED.instrument_class,
            strike_price        = EXCLUDED.strike_price,
            expiration          = EXCLUDED.expiration,
            underlying          = EXCLUDED.underlying,
            exchange            = EXCLUDED.exchange,
            currency            = EXCLUDED.currency,
            min_price_increment = CASE
                WHEN EXCLUDED.min_price_increment IS NULL
                  OR EXCLUDED.min_price_increment <> EXCLUDED.min_price_increment
                  OR EXCLUDED.min_price_increment > 1e6
                THEN databento.definitions.min_price_increment
                ELSE EXCLUDED.min_price_increment END,
            multiplier          = COALESCE(EXCLUDED.multiplier,
                                           databento.definitions.multiplier),
            source              = EXCLUDED.source
        WHERE databento.definitions.source = 'synthetic'
    """

    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            rows = []
            for i, row in df.iterrows():
                rows.append((
                    row.get('ts_event'),
                    int(row.get('instrument_id', 0)),
                    str(row.get('symbol', '')),
                    str(row.get('instrument_class', ''))[:1] if row.get('instrument_class') else None,
                    float(row['strike_price']) if row.get('strike_price') else None,
                    row.get('expiration'),
                    str(row['underlying']) if row.get('underlying') else None,
                    str(row['exchange']) if row.get('exchange') else None,
                    str(row.get('currency', 'USD')),
                    float(row['min_price_increment']) if row.get('min_price_increment') else None,
                    float(row['multiplier']) if row.get('multiplier') else None,
                ))

                if len(rows) >= batch_size:
                    cur.executemany(insert_sql, rows)
                    conn.commit()
                    logger.info(f"  Inserted batch ({i+1:,} / {len(df):,})")
                    rows = []

            if rows:
                cur.executemany(insert_sql, rows)
                conn.commit()

    logger.info(f"Ingestion complete: {len(df):,} definition records")
    return len(df)


def ingest_statistics(dbn_path: str, batch_size: int = 5000) -> int:
    """Load a Statistics .dbn.zst file into databento.statistics table."""
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df(pretty_ts=True, map_symbols=True, price_type="float")

    logger.info(f"Loaded {len(df):,} statistics records from {dbn_path}")
    df = df.reset_index()

    insert_sql = """
        INSERT INTO databento.statistics
            (ts_event, ts_recv, instrument_id, symbol, stat_type, price,
             quantity, sequence, ts_ref, update_action, stat_flags)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """

    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor() as cur:
            rows = []
            for i, row in df.iterrows():
                rows.append((
                    row.get('ts_event'),
                    row.get('ts_recv'),
                    int(row.get('instrument_id', 0)),
                    str(row.get('symbol', '')),
                    int(row.get('stat_type', 0)),
                    float(row['price']) if row.get('price') else None,
                    int(row['quantity']) if row.get('quantity') else None,
                    int(row['sequence']) if row.get('sequence') else None,
                    row.get('ts_ref'),
                    int(row['update_action']) if row.get('update_action') is not None else None,
                    int(row['stat_flags']) if row.get('stat_flags') is not None else None,
                ))

                if len(rows) >= batch_size:
                    cur.executemany(insert_sql, rows)
                    conn.commit()
                    logger.info(f"  Inserted batch ({i+1:,} / {len(df):,})")
                    rows = []

            if rows:
                cur.executemany(insert_sql, rows)
                conn.commit()

    logger.info(f"Ingestion complete: {len(df):,} statistics records")
    return len(df)


# ── Dispatcher ──────────────────────────────────────────────────

_INGEST_MAP = {
    "tbbo":       ingest_tbbo,
    "bbo-1s":     ingest_bbo_1s,
    "definition": ingest_definition,
    "statistics": ingest_statistics,
}


def ingest_file(path: str | Path, schema: str, batch_size: int = 5000,
                total_hint: int | None = None) -> int:
    """Dispatch ingestion to the correct function based on schema name.

    Returns the number of rows ingested.
    """
    func = _INGEST_MAP.get(schema)
    if func is None:
        raise ValueError(f"Unknown schema '{schema}'. Expected one of: {sorted(_INGEST_MAP)}")
    kwargs: dict = {"batch_size": batch_size}
    if total_hint is not None and schema in ("tbbo", "bbo-1s"):
        kwargs["total_hint"] = total_hint
    return func(str(path), **kwargs)


if __name__ == "__main__":
    import argparse as _ap
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    p = _ap.ArgumentParser(description="Ingest a .dbn.zst file into TimescaleDB")
    p.add_argument("schema", choices=sorted(_INGEST_MAP))
    p.add_argument("path", help="Path to .dbn.zst file")
    p.add_argument("--total", type=int, default=None,
                   help="Known total row count (improves progress ETA)")
    args = p.parse_args()

    n = ingest_file(args.path, args.schema, total_hint=args.total)
    print(f"\nIngested {n:,} rows")
