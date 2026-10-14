#!/usr/bin/env python3
"""Build (or refresh) the data_capability.json catalog artifact.

Queries TimescaleDB via **chunk metadata** (sub-second per table — no full scans),
the Postgres definitions table, and the local parquet file index, then writes
``data/ops/catalog/data_capability.json``.

Run:
    python scripts/build_data_capability.py [--out PATH]

The artifact is consumed by:
  * ``src/scoping/runner._gap_check``  — field/range-aware gap comparison
  * ``src/research_director/context_assembler`` — DATA CAPABILITY section in every agent's pack
  * ``src/research_director/tools.data_capability_catalog()`` — agent tool for deep questions

Refresh cadence: nightly (added to vtech-feature-builder.service ExecStartPost) and
on-demand before any Stage-0 scoping run.

Design: Option A from the data-capability design debate (2026-07-03):
  *  Pre-computed artifact, not a live DB query at inference time
  *  Chunk-metadata pattern for dates (fast even on compressed hypertables)
  *  Instruments classified: futures/equity/index kept; granular option strikes summarised
  *  Parquet feature groups discovered from column names of the most-recent parquet file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

log = logging.getLogger("vtech.build_data_capability")

from src.common.data_paths import catalog_dir, native_tf_dir
_DEFAULT_OUT = catalog_dir() / "data_capability.json"

# --- Hypertable spec: (schema, table, ts_col, sym_col, data_type, granularity) ---
_HYPERTABLES = [
    ("databento", "tbbo",        "ts_recv",  "symbol", "tick_bid_offer",    "tick"),
    ("databento", "bbo_1s",      "ts_recv",  "symbol", "options_bbo_1s",    "1s"),
    ("databento", "ohlcv_1s",    "ts_event", "symbol", "ohlcv_1s",          "1s"),
    ("databento", "statistics",  "ts_recv",  "symbol", "statistics_daily",  "daily"),
    ("databento", "equity_ohlcv","ts_event", "symbol", "equity_ohlcv",      "1d"),
]

# Instrument classes to include as first-class instruments (no granular strikes)
_INCLUDE_CLASSES = {"F", "S", "X"}   # futures, stocks/ETFs, indices


def build_dataset_entry(conn, schema: str, table: str, ts_col: str, sym_col: str,
                         data_type: str, granularity: str) -> dict:
    """Build one dataset entry from chunk metadata + definitions + column info."""
    entry: dict = {
        "table": f"{schema}.{table}",
        "data_type": data_type,
        "granularity": granularity,
    }

    # Date range from chunk metadata (no full scan)
    dr = conn.execute("""
        SELECT MIN(range_start)::date, MAX(range_end)::date, count(*) AS n_chunks
        FROM timescaledb_information.chunks
        WHERE hypertable_schema = %s AND hypertable_name = %s
    """, (schema, table)).fetchone()
    if dr and dr[0]:
        entry["date_range"] = {"first": str(dr[0]), "last": str(dr[1])}
        entry["n_chunks"] = int(dr[2])
    else:
        entry["date_range"] = None
        entry["n_chunks"] = 0

    # Column names
    cols = [r[0] for r in conn.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table)).fetchall()]
    entry["fields"] = cols

    # Approx row count from pg_stat
    rc = conn.execute("""
        SELECT COALESCE(n_live_tup, 0)
        FROM pg_stat_user_tables
        WHERE schemaname = %s AND relname = %s
    """, (schema, table)).fetchone()
    entry["approx_rows"] = int(rc[0]) if rc else 0

    # Instruments: sample from the actual table's first + last chunks (fast, table-specific)
    # Then enrich with class/exchange from definitions where available.
    instruments: list[dict] = []
    sampled_symbols: set[str] = set()
    for position in ("ASC", "DESC"):
        chunk = conn.execute(f"""
            SELECT chunk_schema, chunk_name
            FROM timescaledb_information.chunks
            WHERE hypertable_schema = %s AND hypertable_name = %s
            ORDER BY range_start {position} LIMIT 1
        """, (schema, table)).fetchone()
        if chunk:
            cs, cn = chunk
            rows = conn.execute(
                f'SELECT DISTINCT {sym_col} FROM "{cs}"."{cn}" LIMIT 60'
            ).fetchall()
            for (sym,) in rows:
                sym_str = str(sym)
                # Exclude granular option strikes (symbol contains space + C or P + digits)
                if (" C" in sym_str or " P" in sym_str) and sym_str[-1].isdigit():
                    continue
                sampled_symbols.add(sym_str)

    # Enrich with instrument class from definitions if available
    if sampled_symbols:
        try:
            sym_list = list(sampled_symbols)
            defs_rows = conn.execute("""
                SELECT symbol, instrument_class, exchange, dataset, underlying
                FROM databento.definitions
                WHERE symbol = ANY(%s)
            """, (sym_list,)).fetchall()
            def_map = {r[0]: r for r in defs_rows}
            for sym_str in sorted(sampled_symbols):
                if sym_str in def_map:
                    _, cls, exch, ds, underlying = def_map[sym_str]
                    instruments.append({"symbol": sym_str, "class": cls,
                                        "exchange": exch, "dataset": ds,
                                        "underlying": underlying})
                else:
                    instruments.append({"symbol": sym_str, "class": "unknown"})
        except Exception:
            instruments = [{"symbol": s, "class": "unknown"} for s in sorted(sampled_symbols)]

    entry["instruments"] = instruments
    # Compact options summary (count, underlying) — don't list all strikes
    options_count = 0
    options_underlyings: set[str] = set()
    try:
        opts = conn.execute("""
            SELECT COUNT(DISTINCT symbol), COUNT(DISTINCT underlying)
            FROM databento.definitions
            WHERE instrument_class IN ('C', 'P')
        """).fetchone()
        if opts:
            options_count = int(opts[0])
            unds = conn.execute("""
                SELECT DISTINCT underlying FROM databento.definitions
                WHERE instrument_class IN ('C', 'P') AND underlying IS NOT NULL
            """).fetchall()
            options_underlyings = {r[0] for r in unds if r[0]}
    except Exception:
        pass
    if options_count:
        entry["options_summary"] = {
            "total_distinct_strikes": options_count,
            "underlyings": sorted(options_underlyings),
            "note": "granular option strikes excluded from instruments list",
        }

    return entry


def build_parquet_entry() -> dict:
    """Discover the 5s hive feature matrix from the local filesystem."""
    pdir = native_tf_dir("5s")
    if not pdir.exists():
        return {"available": False}

    files = sorted(pdir.glob("year=*/month=*/features_5s_part*.parquet"))
    if not files:
        return {"available": False}

    first_date: str | None = None
    last_date: str | None = None
    skipped_date_files = 0
    for f in files:
        try:
            values = pd.read_parquet(f, columns=["bar_end_ts"])["bar_end_ts"].dropna()
            if values.empty:
                continue
            dates = pd.to_datetime(values, utc=True).dt.date.astype(str)
            lo = str(dates.min())
            hi = str(dates.max())
            first_date = lo if first_date is None else min(first_date, lo)
            last_date = hi if last_date is None else max(last_date, hi)
        except Exception:
            skipped_date_files += 1

    entry: dict = {
        "available": True,
        "directory": "data/sources/features/timeframe=5s/symbol=NQ",
        "n_files": len(files),
        "date_range": {
            "first": first_date or "unknown",
            "last": last_date or "unknown",
        },
    }

    # Discover feature groups from the column names of the most-recent file
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(files[-1])
        feature_cols = [c for c in schema.names if not c.startswith("_")]
        prefixes = sorted({c.split("_")[0] + "_" for c in feature_cols
                           if "_" in c and not c.startswith("label")})
        entry["feature_groups"] = prefixes
        entry["n_feature_columns"] = len(feature_cols)
        # Sample of all columns for field-level queries
        entry["feature_columns_sample"] = feature_cols[:50]
    except Exception as exc:
        entry["feature_groups"] = []
        entry["note"] = f"column discovery failed: {exc}"
    if skipped_date_files:
        note = entry.get("note")
        suffix = f"date discovery skipped {skipped_date_files} unreadable hive part(s)"
        entry["note"] = f"{note}; {suffix}" if note else suffix

    return entry


def build_catalog(out_path: Path) -> dict:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)

    from common.db import get_pool
    pool = get_pool()

    datasets = []
    with pool.connection() as conn:
        log.info("building dataset entries from chunk metadata…")
        for schema, table, ts_col, sym_col, data_type, granularity in _HYPERTABLES:
            try:
                entry = build_dataset_entry(conn, schema, table, ts_col, sym_col,
                                             data_type, granularity)
                datasets.append(entry)
                log.info("  %s.%s: %s → %s  (%d instruments)",
                         schema, table,
                         (entry.get("date_range") or {}).get("first", "?"),
                         (entry.get("date_range") or {}).get("last", "?"),
                         len(entry.get("instruments", [])))
            except Exception as exc:
                log.warning("  SKIP %s.%s: %s", schema, table, exc)

    log.info("scanning parquet feature matrix…")
    parquet = build_parquet_entry()
    if parquet.get("available"):
        log.info("  parquet: %s → %s  (%d files, %d feature columns)",
                 parquet["date_range"]["first"], parquet["date_range"]["last"],
                 parquet["n_files"], parquet.get("n_feature_columns", 0))

    catalog = {
        "$schema": "data_capability/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "parquet_features": parquet,
        "summary": _build_summary(datasets, parquet),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2, default=str))
    log.info("wrote %s", out_path)
    return catalog


def _build_summary(datasets: list[dict], parquet: dict) -> dict:
    """Human/LLM-readable one-page summary for quick context injection."""
    lines = []
    for ds in datasets:
        dr = ds.get("date_range") or {}
        instr = [i["symbol"] for i in ds.get("instruments", [])
                 if i.get("class") in _INCLUDE_CLASSES or i.get("class") == "unknown"]
        opts = ds.get("options_summary", {})
        opts_note = (f" + {opts['total_distinct_strikes']} option strikes on "
                     f"{opts['underlyings']}" if opts else "")
        lines.append(
            f"- {ds['table']} ({ds['data_type']}, {ds['granularity']}): "
            f"{dr.get('first', '?')} → {dr.get('last', '?')}, "
            f"instruments: {instr[:10]}{opts_note}"
        )
    pq = parquet
    if pq.get("available"):
        lines.append(
            f"- parquet_features: {pq['date_range']['first']} → {pq['date_range']['last']}, "
            f"{pq['n_files']} files, {pq.get('n_feature_columns',0)} feature columns, "
            f"groups: {pq.get('feature_groups', [])}"
        )
    else:
        lines.append("- parquet_features: not available")
    return {"text": "\n".join(lines)}


def load_catalog(path: Path | None = None) -> dict:
    """Load the pre-built catalog (or return empty sentinel if absent)."""
    p = path or _DEFAULT_OUT
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the data capability catalog artifact")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help="Output JSON path")
    args = ap.parse_args()
    catalog = build_catalog(Path(args.out))
    print(catalog["summary"]["text"])
