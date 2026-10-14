#!/usr/bin/env python3
"""Unified .dbn.zst ingestion script — consolidates all ingest_*.py scripts.

Supported tables (--table):
  bbo_1s    databento.bbo_1s   — BBO 1-second data (replaces ingest_nq_opt.py)
  ohlcv_1s  databento.ohlcv_1s — OHLCV 1-second data (replaces ingest_ohlcv_1s.py)
  tbbo      databento.tbbo     — TBBO tick data (replaces ingest_tbbo_chunked.py,
                                  modernised to COPY instead of row-at-a-time INSERT)

File discovery (choose one):
  --data-dir DIR     Recursively scan DIR for *.dbn.zst files
  --glob PATTERN     Custom glob relative to --data-dir (default: "**/*.dbn.zst")
  --files F [F ...]  Explicit file list (overrides --data-dir)

Progress is tracked in a JSON checkpoint file. Re-runs are safe — completed
files are skipped. The DB UNIQUE constraint is the safety net for any overlap.

Usage:
    # Dry run — show file list, row estimates, worker count
    python scripts/ingest_dbn.py --table ohlcv_1s \\
        --data-dir /opt/vtech/data/sources/raw/backfill --dry-run

    # Ingest bbo_1s from NQ-opt download directory
    python scripts/ingest_dbn.py --table bbo_1s \\
        --data-dir /opt/vtech/data/sources/raw/backfill/nq_opt \\
        --checkpoint /opt/vtech/data/legacy/checkpoints/ingest_bbo_1s.json

    # Ingest ohlcv_1s (both NQ + ES directories)
    python scripts/ingest_dbn.py --table ohlcv_1s \\
        --data-dir /opt/vtech/data/sources/raw/backfill/nq_fut_ohlcv_1s \\
        --data-dir /opt/vtech/data/sources/raw/backfill/es_fut_ohlcv_1s

    # Ingest tbbo files (replaces ingest_tbbo_chunked.py)
    python scripts/ingest_dbn.py --table tbbo \\
        --data-dir /opt/vtech/data/sources/raw/backfill/tbbo

    # Reset checkpoint and re-ingest
    python scripts/ingest_dbn.py --table ohlcv_1s --data-dir ... --reset-checkpoint

Worker model:
    ProcessPoolExecutor — CPU-bound DBN zstd decompress + parse
    Workers are sized by vm_config.workers_for(mem_per_worker_gb=MEM) per table.
    Default: mem_per_worker_gb=0.5 for bbo_1s/ohlcv_1s, 1.5 for tbbo.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import databento as db
import psycopg
from dotenv import load_dotenv

load_dotenv("/opt/vtech/.env")

sys.path.insert(0, "/opt/vtech")
from src.common.vm_config import get_vm_config, fmt_bytes

# ---------------------------------------------------------------------------
# Per-table configuration
# ---------------------------------------------------------------------------
#
# Each entry:
#   table       : fully-qualified DB table name
#   copy_cols   : columns to COPY (must match DB DDL order / names, excluding GENERATED/DEFAULT)
#   df_rename   : dict of DBN DataFrame column → DB column name
#   mem_gb      : estimated peak RAM per worker (for worker sizing)
#   unique_key  : columns forming the natural unique key (used only for doc/diagnostics)
#
_TABLE_CONFIGS: dict[str, dict] = {
    "bbo_1s": {
        "table":      "databento.bbo_1s",
        "copy_cols":  ["ts_recv", "instrument_id", "symbol",
                       "price", "size",
                       "bid_px", "ask_px", "bid_sz", "ask_sz"],
        "df_rename":  {"bid_px_00": "bid_px", "ask_px_00": "ask_px",
                       "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz"},
        "mem_gb":     0.5,
        "unique_key": ["instrument_id", "ts_recv"],
    },
    "ohlcv_1s": {
        "table":      "databento.ohlcv_1s",
        "copy_cols":  ["ts_event", "instrument_id", "symbol",
                       "open", "high", "low", "close", "volume"],
        "df_rename":  {},
        "mem_gb":     0.5,
        "unique_key": ["instrument_id", "ts_event"],
    },
    "tbbo": {
        "table":      "databento.tbbo",
        "copy_cols":  ["ts_event", "ts_recv", "instrument_id", "symbol",
                       "price", "size", "side", "action",
                       "flags", "sequence",
                       "bid_px", "ask_px", "bid_sz", "ask_sz",
                       "bid_ct", "ask_ct", "ts_in_delta"],
        "df_rename":  {"bid_px_00": "bid_px", "ask_px_00": "ask_px",
                       "bid_sz_00": "bid_sz", "ask_sz_00": "ask_sz",
                       "bid_ct_00": "bid_ct", "ask_ct_00": "ask_ct"},
        "mem_gb":     1.5,
        "unique_key": ["instrument_id", "ts_recv"],
    },
}

ALL_TABLES = list(_TABLE_CONFIGS.keys())
CKPT_DEFAULT_DIR = Path("/opt/vtech/data/legacy/checkpoints")
COPY_BLOCK = 1 << 16   # 64 KB write blocks to psycopg COPY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_dbn")

# ---------------------------------------------------------------------------
# DSN
# ---------------------------------------------------------------------------

def _dsn() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ.get('PGHOST', 'localhost')}:{os.environ.get('PGPORT', '5432')}"
        f"/{os.environ.get('PGDATABASE', 'vtech_market')}"
    )

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_ckpt(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_ckpt(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

# ---------------------------------------------------------------------------
# Worker (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _ingest_file(
    file_path_str: str,
    table_name: str,
    copy_cols: list[str],
    df_rename: dict[str, str],
) -> dict:
    """Ingest one .dbn.zst file via psycopg COPY.

    This function is invoked in a subprocess — it re-loads .env and imports
    everything it needs locally to avoid fork-safety issues.
    """
    from dotenv import load_dotenv
    load_dotenv("/opt/vtech/.env")
    import csv as _csv
    import io as _io
    import os as _os
    import time as _time

    import databento as _db
    import psycopg as _psycopg
    import pandas as _pd

    def _get_dsn() -> str:
        return (
            f"postgresql://{_os.environ['PGUSER']}:{_os.environ['PGPASSWORD']}"
            f"@{_os.environ.get('PGHOST', 'localhost')}:{_os.environ.get('PGPORT', '5432')}"
            f"/{_os.environ.get('PGDATABASE', 'vtech_market')}"
        )

    BLOCK = 1 << 16
    t0 = _time.time()
    result = {
        "file": file_path_str,
        "status": "ok",
        "rows": 0,
        "error": "",
        "elapsed_s": 0.0,
    }

    try:
        store = _db.DBNStore.from_file(file_path_str)
        # Use count= to avoid loading all rows into RAM at once for large files
        dfs = store.to_df(pretty_ts=True, map_symbols=True, price_type="float", count=500_000)

        cols_sql = ", ".join(copy_cols)
        total_rows = 0

        with _psycopg.connect(_get_dsn()) as conn:
            cur = conn.cursor()
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET work_mem = '256MB'")

            for df in dfs:
                if isinstance(df, _pd.DataFrame):
                    chunk_df = df
                else:
                    # to_df can return a generator or a single DataFrame
                    chunk_df = df
                    if chunk_df is None or len(chunk_df) == 0:
                        continue

                if len(chunk_df) == 0:
                    continue

                chunk_df = chunk_df.reset_index().rename(columns=df_rename)

                # Select only the target COPY columns (drop extras like spread)
                available = [c for c in copy_cols if c in chunk_df.columns]
                missing   = [c for c in copy_cols if c not in chunk_df.columns]
                if missing:
                    # Fill missing numeric columns with None; this will surface
                    # as NULL in the DB and the caller can investigate.
                    for mc in missing:
                        chunk_df[mc] = None
                    available = copy_cols

                prepped = chunk_df[available]

                buf = _io.BytesIO()
                prepped.to_csv(buf, header=False, index=False, na_rep="",
                               quoting=_csv.QUOTE_MINIMAL, encoding="utf-8")
                buf.seek(0)

                with cur.copy(
                    f"COPY {table_name} ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '')"
                ) as copy:
                    while block := buf.read(BLOCK):
                        copy.write(block)

                total_rows += len(prepped)
                conn.commit()

        result["rows"] = total_rows
        result["elapsed_s"] = _time.time() - t0

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["elapsed_s"] = _time.time() - t0

    return result

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_files(data_dirs: list[Path], glob: str) -> list[Path]:
    """Recursively discover .dbn.zst files under one or more directories."""
    files: list[Path] = []
    for d in data_dirs:
        if not d.exists():
            logger.warning("Data directory does not exist: %s", d)
            continue
        found = sorted(d.rglob(glob) if "**" in glob else d.glob(glob))
        logger.info("  %s: found %d files", d, len(found))
        files.extend(found)
    return files

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Unified .dbn.zst → PostgreSQL COPY ingestor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--table", required=True, choices=ALL_TABLES,
        help=f"Target DB table. Choices: {{{', '.join(ALL_TABLES)}}}",
    )
    parser.add_argument(
        "--data-dir", dest="data_dirs", action="append", type=Path, default=[],
        metavar="DIR",
        help="Directory to scan for .dbn.zst files. Repeat for multiple dirs.",
    )
    parser.add_argument(
        "--glob", default="**/*.dbn.zst",
        help="File glob relative to --data-dir (default: '**/*.dbn.zst')",
    )
    parser.add_argument(
        "--files", nargs="+", type=Path, default=None,
        help="Explicit file list (overrides --data-dir / --glob)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker count (default: vm_config adaptive)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint JSON path (default: data/legacy/checkpoints/ingest_{table}.json)",
    )
    parser.add_argument(
        "--reset-checkpoint", action="store_true",
        help="Clear checkpoint and re-ingest all files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without ingesting any data",
    )
    args = parser.parse_args()

    cfg = _TABLE_CONFIGS[args.table]
    table_name = cfg["table"]
    copy_cols  = cfg["copy_cols"]
    df_rename  = cfg["df_rename"]
    mem_gb     = cfg["mem_gb"]

    # Worker count
    vm = get_vm_config()
    n_workers = args.workers or vm.workers_for(mem_per_worker_gb=mem_gb)

    # Checkpoint
    ckpt_path = args.checkpoint or (CKPT_DEFAULT_DIR / f"ingest_{args.table}.json")
    if args.reset_checkpoint and ckpt_path.exists():
        ckpt_path.unlink()
        logger.info("Checkpoint reset: %s", ckpt_path)
    ckpt = _load_ckpt(ckpt_path)

    # File list
    if args.files:
        all_files = [Path(f) for f in args.files]
    elif args.data_dirs:
        all_files = _discover_files(args.data_dirs, args.glob)
    else:
        parser.error("Provide --data-dir or --files")
        return

    if not all_files:
        logger.error("No .dbn.zst files found. Check --data-dir / --glob.")
        sys.exit(1)

    # Split into pending vs already done
    done_files   = [f for f in all_files if ckpt.get(str(f)) == "done"]
    pending_files = [f for f in all_files if ckpt.get(str(f)) != "done"]

    tag = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n{tag}ingest_dbn — table: {table_name}")
    print(f"  Files total   : {len(all_files)}")
    print(f"  Already done  : {len(done_files)}")
    print(f"  To ingest     : {len(pending_files)}")
    print(f"  Workers       : {n_workers}  (mem/worker ≈ {mem_gb} GB)")
    print(f"  Checkpoint    : {ckpt_path}")
    print()

    if args.dry_run or not pending_files:
        if not pending_files:
            print("Nothing to do — all files already ingested.")
        return

    total_rows = 0
    total_errors = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_ingest_file, str(f), table_name, copy_cols, df_rename): f
            for f in pending_files
        }
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            f = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"file": str(f), "status": "error", "rows": 0,
                       "error": str(exc), "elapsed_s": 0.0}

            status = res["status"]
            flag = "✅" if status == "ok" else "❌"
            rows = res["rows"]
            elapsed = res["elapsed_s"]
            total_rows += rows

            pct = done_count / len(pending_files) * 100
            bar_w = int(pct / 5)
            bar = "█" * bar_w + "░" * (20 - bar_w)
            fname = Path(res["file"]).name

            print(
                f"  {flag} [{done_count:03d}/{len(pending_files)}] [{bar}] {pct:5.1f}%"
                f"  {fname}  {rows:,} rows  {elapsed:.1f}s"
            )

            if status == "ok":
                ckpt[str(f)] = "done"
                _save_ckpt(ckpt_path, ckpt)
            else:
                total_errors += 1
                logger.error("  ERROR: %s — %s", fname, res.get("error", ""))

    elapsed_total = time.time() - t_start
    ok_count = len(pending_files) - total_errors
    print()
    print(f"{'='*60}")
    print(f"  {tag}Ingest complete")
    print(f"  Files:  {ok_count} ok | {total_errors} errors")
    print(f"  Rows:   {total_rows:,}")
    print(f"  Time:   {elapsed_total/60:.1f} min  ({elapsed_total:.0f}s)")
    print(f"{'='*60}")

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
