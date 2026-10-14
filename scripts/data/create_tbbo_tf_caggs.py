#!/usr/bin/env python3
"""Create and backfill NQ TBBO 15s / 5m / 1h / 1d continuous aggregates.

    python scripts/data/create_tbbo_tf_caggs.py
    python scripts/data/create_tbbo_tf_caggs.py --ddl-only
    python scripts/data/create_tbbo_tf_caggs.py --refresh-only

15s: ``time_bucket('15 seconds')`` on ``databento.tbbo`` ticks (same
columns as ``tbbo_1min``).
5m / 1h: nested caggs on ``databento.tbbo_1min`` (clock 5 minutes / hours).
1d: ``time_bucket`` on ``tbbo`` with timezone ``America/Chicago`` and
origin 17:00 CT — session T is [T-1 17:00, T 17:00) CT, left-labeled at
session open. Globex is dark 16:00–17:00 CT so the last trade is at/before
16:00 CT.

NQ outrights only: ``symbol ~ '^NQ[HMUZ][0-9]+$'`` (no ES, no spreads).

Default refresh: earliest tbbo chunk 2025-04-06 00:00 UTC through the
close of session 2026-08-25 (exclusive end 2026-08-25 17:00 CDT =
2026-08-25 22:00 UTC).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg  # noqa: E402

from src.common.db import get_dsn  # noqa: E402

NQ_OUTRIGHT = r"^NQ[HMUZ][0-9]+$"
REFRESH_START = datetime(2025, 4, 6, tzinfo=timezone.utc)
# Exclusive end: 2026-08-25 17:00 CDT = session-2026-08-25 close boundary.
REFRESH_END = datetime(2026, 8, 25, 22, 0, tzinfo=timezone.utc)
DAY_ORIGIN = "TIMESTAMPTZ '2000-01-01 17:00:00-06'"

DDL = {
    "tbbo_15s": f"""
CREATE MATERIALIZED VIEW databento.tbbo_15s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 seconds', ts_event) AS bucket,
    instrument_id,
    symbol,
    first(price, ts_event)  AS open,
    max(price)              AS high,
    min(price)              AS low,
    last(price, ts_event)   AS close,
    sum(size)               AS volume,
    sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume,
    sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume,
    count(*)                AS trade_count,
    last(bid_px, ts_event)  AS bid_px,
    last(ask_px, ts_event)  AS ask_px,
    last(bid_sz, ts_event)  AS bid_sz,
    last(ask_sz, ts_event)  AS ask_sz
FROM databento.tbbo
WHERE symbol ~ '{NQ_OUTRIGHT}'
GROUP BY 1, 2, 3
WITH NO DATA
""",
    "tbbo_5m": f"""
CREATE MATERIALIZED VIEW databento.tbbo_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', bucket) AS bucket,
    instrument_id,
    symbol,
    first(open, bucket)  AS open,
    max(high)            AS high,
    min(low)             AS low,
    last(close, bucket)  AS close,
    sum(volume)          AS volume,
    sum(buy_volume)      AS buy_volume,
    sum(sell_volume)     AS sell_volume,
    sum(trade_count)     AS trade_count,
    last(bid_px, bucket) AS bid_px,
    last(ask_px, bucket) AS ask_px,
    last(bid_sz, bucket) AS bid_sz,
    last(ask_sz, bucket) AS ask_sz
FROM databento.tbbo_1min
WHERE symbol ~ '{NQ_OUTRIGHT}'
GROUP BY 1, 2, 3
WITH NO DATA
""",
    "tbbo_1h": f"""
CREATE MATERIALIZED VIEW databento.tbbo_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', bucket) AS bucket,
    instrument_id,
    symbol,
    first(open, bucket)  AS open,
    max(high)            AS high,
    min(low)             AS low,
    last(close, bucket)  AS close,
    sum(volume)          AS volume,
    sum(buy_volume)      AS buy_volume,
    sum(sell_volume)     AS sell_volume,
    sum(trade_count)     AS trade_count,
    last(bid_px, bucket) AS bid_px,
    last(ask_px, bucket) AS ask_px,
    last(bid_sz, bucket) AS bid_sz,
    last(ask_sz, bucket) AS ask_sz
FROM databento.tbbo_1min
WHERE symbol ~ '{NQ_OUTRIGHT}'
GROUP BY 1, 2, 3
WITH NO DATA
""",
    "tbbo_1d": f"""
CREATE MATERIALIZED VIEW databento.tbbo_1d
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(
        '1 day'::interval,
        ts_event,
        'America/Chicago',
        {DAY_ORIGIN}
    ) AS bucket,
    instrument_id,
    symbol,
    first(price, ts_event)  AS open,
    max(price)              AS high,
    min(price)              AS low,
    last(price, ts_event)   AS close,
    sum(size)               AS volume,
    sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume,
    sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume,
    count(*)                AS trade_count,
    last(bid_px, ts_event)  AS bid_px,
    last(ask_px, ts_event)  AS ask_px,
    last(bid_sz, ts_event)  AS bid_sz,
    last(ask_sz, ts_event)  AS ask_sz
FROM databento.tbbo
WHERE symbol ~ '{NQ_OUTRIGHT}'
GROUP BY 1, 2, 3
WITH NO DATA
""",
}

POLICIES = (
    (
        "databento.tbbo_15s",
        "INTERVAL '3 hours'",
        "INTERVAL '15 seconds'",
        "INTERVAL '1 minute'",
    ),
    (
        "databento.tbbo_5m",
        "INTERVAL '3 days'",
        "INTERVAL '5 minutes'",
        "INTERVAL '5 minutes'",
    ),
    (
        "databento.tbbo_1h",
        "INTERVAL '3 days'",
        "INTERVAL '1 hour'",
        "INTERVAL '15 minutes'",
    ),
    (
        "databento.tbbo_1d",
        "INTERVAL '3 days'",
        "INTERVAL '1 hour'",
        "INTERVAL '1 hour'",
    ),
)

FIVE_VIEW = "databento.tbbo_5m"
HOUR_VIEW = "databento.tbbo_1h"


def _bar(frac: float, width: int = 20) -> str:
    n = max(0, min(width, int(round(width * frac))))
    return "█" * n + "░" * (width - n)


def _loadavg() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def _wait_load(limit: float) -> None:
    while True:
        load = _loadavg()
        if load <= limit:
            return
        print(f"  load {load:.2f} > {limit:.1f}, sleep 30s", flush=True)
        time.sleep(30)


def existing_caggs(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT view_name FROM timescaledb_information.continuous_aggregates "
        "WHERE view_schema = 'databento'"
    ).fetchall()
    return {r[0] for r in rows}


def create_views(conn: psycopg.Connection) -> None:
    have = existing_caggs(conn)
    for name, sql in DDL.items():
        if name in have:
            print(f"  exists databento.{name}", flush=True)
            continue
        conn.execute(sql)
        print(f"  created databento.{name}", flush=True)


def add_policies(conn: psycopg.Connection) -> None:
    for view, start_off, end_off, sched in POLICIES:
        conn.execute(
            "SELECT add_continuous_aggregate_policy("
            f"%s, start_offset => {start_off}, end_offset => {end_off}, "
            f"schedule_interval => {sched}, if_not_exists => TRUE)",
            (view,),
        )
        print(f"  policy {view}", flush=True)


def _refresh(conn: psycopg.Connection, view: str, start: datetime, end: datetime) -> None:
    conn.execute(
        "CALL refresh_continuous_aggregate(%s, %s::timestamptz, %s::timestamptz)",
        (view, start.isoformat(), end.isoformat()),
    )


def _session_settings(conn: psycopg.Connection) -> None:
    conn.execute("SET statement_timeout = '15min'")
    conn.execute("SET work_mem = '256MB'")
    conn.execute("SET maintenance_work_mem = '1GB'")
    conn.execute("SET max_parallel_workers_per_gather = 0")
    try:
        conn.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
    except Exception:
        pass


def day_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=1), end)
        out.append((cur, nxt))
        cur = nxt
    return out


def week_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """1d cagg refresh needs ≥ two daily buckets per window."""
    out: list[tuple[datetime, datetime]] = []
    cur = start
    step = timedelta(days=7)
    min_span = timedelta(days=2)
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) < min_span:
        a, _ = out[-2]
        _, b = out[-1]
        out[-2:] = [(a, b)]
    elif len(out) == 1 and (out[0][1] - out[0][0]) < min_span:
        out[0] = (out[0][1] - min_span, out[0][1])
    return out


def refresh_nested(conn: psycopg.Connection, view: str, start: datetime, end: datetime) -> None:
    t0 = time.monotonic()
    print(f"⏳ Refreshing {view} {start.date()} → {end.isoformat()}", flush=True)
    _refresh(conn, view, start, end)
    dt = time.monotonic() - t0
    print(f"✅ {view} refresh in {dt:.0f}s", flush=True)


def _refresh_windows(
    conn: psycopg.Connection,
    view: str,
    windows: list[tuple[datetime, datetime]],
    load_limit: float,
) -> None:
    n = len(windows)
    t0 = time.monotonic()
    print(
        f"⏳ Refreshing {view} in {n} windows "
        f"{windows[0][0].date()} → {windows[-1][1].isoformat()}",
        flush=True,
    )
    for i, (a, b) in enumerate(windows, 1):
        _wait_load(load_limit)
        w0 = time.monotonic()
        _refresh(conn, view, a, b)
        elapsed = time.monotonic() - t0
        frac = i / n
        print(
            f"[{_bar(frac)}] {100 * frac:5.1f}% — {view} {a.date()} "
            f"{time.monotonic() - w0:.0f}s  elapsed {elapsed / 60:.1f}m",
            flush=True,
        )
    dt = time.monotonic() - t0
    print(f"✅ {view} refresh in {dt / 60:.1f}m ({n} windows)", flush=True)


def refresh_tick_views(
    conn: psycopg.Connection,
    start: datetime,
    end: datetime,
    load_limit: float,
) -> None:
    _refresh_windows(conn, "databento.tbbo_15s", day_windows(start, end), load_limit)
    _refresh_windows(conn, "databento.tbbo_1d", week_windows(start, end), load_limit)


def enable_15s_compression(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        ALTER MATERIALIZED VIEW databento.tbbo_15s SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'instrument_id',
            timescaledb.compress_orderby = 'bucket ASC'
        )
        """
    )
    conn.execute(
        "SELECT add_compression_policy("
        "'databento.tbbo_15s', INTERVAL '7 days', if_not_exists => TRUE)"
    )
    print("  tbbo_15s compression policy (7d)", flush=True)


def summarize(conn: psycopg.Connection) -> None:
    for view in ("tbbo_15s", "tbbo_5m", "tbbo_1h", "tbbo_1d"):
        row = conn.execute(
            f"SELECT MIN(bucket), MAX(bucket), COUNT(*) FROM databento.{view}"  # noqa: S608
        ).fetchone()
        bad = conn.execute(
            f"SELECT COUNT(*) FROM databento.{view} "  # noqa: S608
            "WHERE symbol !~ %s",
            (NQ_OUTRIGHT,),
        ).fetchone()[0]
        print(
            f"  {view}: {row[2]:,} rows  {row[0]} → {row[1]}  "
            f"non-NQ={bad}",
            flush=True,
        )
    sample = conn.execute(
        """
        SELECT bucket,
               bucket AT TIME ZONE 'America/Chicago' AS bucket_ct,
               symbol, open, high, low, close, volume
        FROM databento.tbbo_1d
        WHERE bucket = TIMESTAMPTZ '2026-08-24 22:00:00+00'
        ORDER BY volume DESC
        LIMIT 5
        """
    ).fetchall()
    print("  session 2026-08-25 daily bars (label 2026-08-24 17:00 CDT):", flush=True)
    for r in sample:
        print(f"    {r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ddl-only", action="store_true")
    ap.add_argument("--refresh-only", action="store_true")
    ap.add_argument("--start", default=REFRESH_START.isoformat())
    ap.add_argument("--end", default=REFRESH_END.isoformat())
    ap.add_argument("--skip-5m", action="store_true")
    ap.add_argument("--skip-1h", action="store_true")
    ap.add_argument("--skip-ticks", action="store_true")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--load-limit", type=float, default=8.0)
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    dsn = get_dsn()
    with psycopg.connect(dsn, autocommit=True) as conn:
        _session_settings(conn)
        if not args.refresh_only:
            print("Creating caggs (WITH NO DATA) …", flush=True)
            create_views(conn)
            add_policies(conn)
        if args.ddl_only:
            return 0
        if not args.skip_5m:
            refresh_nested(conn, FIVE_VIEW, start, end)
        if not args.skip_1h:
            refresh_nested(conn, HOUR_VIEW, start, end)
        if not args.skip_ticks:
            refresh_tick_views(conn, start, end, args.load_limit)
        if not args.no_compress:
            try:
                enable_15s_compression(conn)
            except Exception as exc:
                print(f"  compression skipped: {exc}", flush=True)
        print("Summary:", flush=True)
        summarize(conn)
    return 0


if __name__ == "__main__":
    if "PGUSER" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv("/opt/vtech/.env")
    raise SystemExit(main())
