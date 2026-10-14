#!/usr/bin/env python3
"""Cost estimation for Databento historical downloads.

Usage:
    python scripts/databento_cost_check.py
    python scripts/databento_cost_check.py --symbols NQ.FUT --schema tbbo --start 2025-04-01 --end 2026-04-01
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import databento as db

from src.acquisition.schemas import BACKFILL_JOBS, DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description="Databento download cost estimator")
    parser.add_argument("--symbols", help="Override symbols (e.g. NQ.FUT)")
    parser.add_argument("--schema", help="Override schema (e.g. tbbo)")
    parser.add_argument(
        "--stype-in", default="parent",
        choices=["parent", "raw_symbol", "continuous", "instrument_id"],
        help="Symbology for --symbols. Use raw_symbol for a single contract (NQU6); "
             "parent only resolves root symbols like NQ.FUT (default: parent)",
    )
    today = date.today()
    parser.add_argument("--start", default=(today - timedelta(days=365)).isoformat())
    parser.add_argument("--end", default=today.isoformat())
    parser.add_argument("--all", action="store_true", help="Check all backfill jobs")
    args = parser.parse_args()

    client = db.Historical()
    total = 0.0

    if args.symbols and args.schema:
        jobs = [{"symbols": args.symbols, "schema": args.schema,
                 "stype_in": args.stype_in, "priority": 0}]
    elif args.all:
        jobs = BACKFILL_JOBS
    else:
        jobs = BACKFILL_JOBS

    print(f"{'Symbols':<15} {'Schema':<12} {'Priority':>8}  {'Cost':>10}")
    print("-" * 50)

    for job in jobs:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=job["symbols"],
            schema=job["schema"],
            stype_in=job["stype_in"],
            start=args.start,
            end=args.end,
        )
        total += cost
        print(f"{job['symbols']:<15} {job['schema']:<12} P{job['priority']:>7}  ${cost:>9.2f}")

    print("-" * 50)
    print(f"{'TOTAL':<37}  ${total:>9.2f}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/opt/vtech/.env")
    main()
