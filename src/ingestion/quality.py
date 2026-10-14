"""Data quality checks and gap detection for ingested data."""

from __future__ import annotations

import logging
from datetime import date

import psycopg

from src.common.db import get_dsn

logger = logging.getLogger("vtech.ingestion.quality")


def check_gaps(target_date: date, symbol_prefix: str = "NQ") -> dict:
    """Check for time gaps in TBBO data for a given date.

    Returns dict with gap_count, max_gap_seconds, and total_records.
    """
    sql = """
        WITH deltas AS (
            SELECT ts_event,
                   ts_event - LAG(ts_event) OVER (ORDER BY ts_event) AS gap
            FROM databento.tbbo
            WHERE ts_event::date = %s
              AND symbol LIKE %s
        )
        SELECT
            count(*) FILTER (WHERE gap > INTERVAL '5 seconds') AS gap_count,
            EXTRACT(EPOCH FROM max(gap)) AS max_gap_s,
            count(*) AS total_records
        FROM deltas
    """
    with psycopg.connect(get_dsn()) as conn:
        row = conn.execute(sql, (target_date, f"{symbol_prefix}%")).fetchone()

    result = {
        "date": str(target_date),
        "gap_count": row[0] or 0,
        "max_gap_seconds": float(row[1]) if row[1] else 0.0,
        "total_records": row[2] or 0,
    }
    logger.info("gap_check", extra=result)
    return result


def record_counts(target_date: date) -> dict[str, int]:
    """Return record counts per table for a given date."""
    tables = ["databento.tbbo", "databento.bbo_1s", "databento.definitions", "databento.statistics"]
    counts = {}
    with psycopg.connect(get_dsn()) as conn:
        for table in tables:
            ts_col = "ts_recv" if table == "databento.bbo_1s" else "ts_event"
            row = conn.execute(
                f"SELECT count(*) FROM {table} WHERE {ts_col}::date = %s",  # noqa: S608
                (target_date,),
            ).fetchone()
            counts[table] = row[0] if row else 0
    return counts
