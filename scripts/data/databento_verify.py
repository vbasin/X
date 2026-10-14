#!/usr/bin/env python3
"""Data verification: record counts, gap detection, date coverage.

Usage:
    python scripts/databento_verify.py --date 2025-07-15
    python scripts/databento_verify.py --summary
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

import psycopg

from src.common.db import get_dsn
from src.ingestion.quality import check_gaps, record_counts


def summary() -> None:
    """Print overall data summary across all tables."""
    dsn = get_dsn()
    with psycopg.connect(dsn) as conn:
        for table in ["databento.tbbo", "databento.bbo_1s", "databento.definitions", "databento.statistics"]:
            row = conn.execute(
                f"SELECT count(*), min(ts_event)::date, max(ts_event)::date FROM {table}"  # noqa: S608
            ).fetchone()
            print(f"{table:<30} {row[0]:>12,} rows  {row[1]} → {row[2]}")

        # Symbols
        symbols = conn.execute("SELECT DISTINCT symbol FROM databento.tbbo ORDER BY symbol").fetchall()
        print(f"\nTBBO symbols: {', '.join(r[0] for r in symbols)}")

        # Continuous aggregate
        row = conn.execute("SELECT count(*) FROM databento.tbbo_1min").fetchone()
        print(f"\ntbbo_1min continuous aggregate: {row[0]:,} rows")


def verify_date(target: date) -> None:
    counts = record_counts(target)
    for table, n in counts.items():
        status = "OK" if n > 0 else "MISSING"
        print(f"  {table:<30} {n:>10,}  [{status}]")

    gaps = check_gaps(target)
    print(f"  TBBO gaps >5s: {gaps['gap_count']}  max_gap: {gaps['max_gap_seconds']:.1f}s  records: {gaps['total_records']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ingested Databento data")
    parser.add_argument("--date", help="Verify a specific date (YYYY-MM-DD)")
    parser.add_argument("--summary", action="store_true", help="Print overall summary")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if args.summary:
        summary()
    elif args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
        print(f"Verifying {d}:")
        verify_date(d)
    else:
        parser.error("Provide --date or --summary")


if __name__ == "__main__":
    main()
