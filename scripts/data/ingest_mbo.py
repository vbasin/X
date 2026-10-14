#!/usr/bin/env python3
"""Ingest Databento MBO `.dbn.zst` day files into `databento.mbo`.

Copy of ingest_mbp10.py (same FIFO/COPY design, same idempotency); only the table,
column list and checkpoint differ.

Fast path — no Python per-record loop, no multi-GB temp CSV:

    .dbn.zst --[DBN Rust CSV encoder]--> FIFO --[COPY FROM STDIN]--> hypertable

The Rust encoder's CSV header is a name-for-name match with the table's columns
(`ts_recv, ts_event, rtype, ..., order_id, flags, ts_in_delta, sequence, symbol`),
so the COPY needs no transformation step. Two properties were verified against
the encoder before relying on them (see docs/architecture/DATABASE_SCHEMA.md):

  * `UNDEF_PRICE` (2^63-1) is emitted as an *empty* field, so absent book levels
    land as SQL NULL rather than a bogus 9.22e9 price.
  * `pretty_px=True` is lossless for the CME price grid — every tick is a
    multiple of 0.25 and |px_nanos| ~ 2.4e13 << 2^53.

Known, deliberate fidelity limit: DBN timestamps are nanosecond, TIMESTAMPTZ is
microsecond, so ts_recv/ts_event are rounded half-up to the nearest microsecond
(<=500 ns of error). Record order is still recoverable from `sequence`. The
`.dbn.zst` files remain the lossless archive of record.

order_id is uint64 in DBN but stored as BIGINT; COPY errors on >= 2^63 (never
observed — CME ids are ~7e12).

Idempotency: the ts_recv range covered by each file is taken from the file's own
DBN metadata (not its name), then deleted before COPY. Re-running a file is safe.
Compressed chunks are decompressed first, and recompressed after.

Usage:
    # what would run, with per-file row estimates
    python scripts/data/ingest_mbo.py --data-dir /opt/vtech/data/sources/raw/backfill/nq_fut_mbo --dry-run

    # single day first, then the rest
    python scripts/data/ingest_mbo.py --files /path/glbx-mdp3-20260721.mbo.dbn.zst
    python scripts/data/ingest_mbo.py --data-dir /opt/vtech/data/sources/raw/backfill/nq_fut_mbo --workers 3
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

TABLE = "databento.mbo"

# Exact output order of the DBN Rust CSV encoder for the mbo schema
# (verified against a real NQZ6 record 2026-09-25).
CSV_COLS: list[str] = [
    "ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id", "action", "side",
    "price", "size", "channel_id", "order_id", "flags", "ts_in_delta", "sequence", "symbol",
]

CKPT_DEFAULT = Path("/opt/vtech/data/legacy/checkpoints/ingest_mbo.json")
READ_BLOCK = 1 << 20   # 1 MiB reads off the FIFO


def _dsn() -> str:
    from dotenv import load_dotenv
    load_dotenv("/opt/vtech/.env")
    return (
        f"host={os.environ.get('PGHOST', '127.0.0.1')} "
        f"port={os.environ.get('PGPORT', '5432')} "
        f"dbname={os.environ['PGDATABASE']} "
        f"user={os.environ['PGUSER']} "
        f"password={os.environ['PGPASSWORD']}"
    )


def _file_range_ns(path: Path) -> tuple[int, int]:
    """(start_ns, end_ns) covered by a DBN file, from its own metadata."""
    import databento as db
    md = db.DBNStore.from_file(path).metadata
    return int(md.start), int(md.end)


def _ingest_file(path_str: str, dsn: str) -> dict:
    """Ingest one .dbn.zst file. Runs in a worker process."""
    import databento as db
    import psycopg

    path = Path(path_str)
    t0 = time.time()
    start_ns, end_ns = _file_range_ns(path)
    # Snap to whole UTC days. show_chunks() only returns chunks that fall *entirely*
    # within (newer_than, older_than), so mid-day bounds match nothing and the
    # post-COPY compression would silently no-op. Batch files use split_duration=day,
    # so one file == one chunk == one day, and day bounds are also the right
    # DELETE range for idempotent re-runs.
    lo = datetime.fromtimestamp(start_ns / 1e9, timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    hi = datetime.fromtimestamp(end_ns / 1e9, timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    if hi <= lo:
        hi = lo + timedelta(days=1)

    cols_sql = ", ".join(f'"{c}"' for c in CSV_COLS)
    copy_sql = (
        f"COPY {TABLE} ({cols_sql}) FROM STDIN "
        f"WITH (FORMAT csv, HEADER true, NULL '')"
    )

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            # A chunk from a previous run may already be compressed — DELETE and COPY
            # both need it back in row form. decompress_chunk is DDL-ish so it stays
            # outside the DELETE+COPY transaction below.
            cur.execute(
                "SELECT decompress_chunk(c, if_compressed => true) "
                "FROM show_chunks(%s, newer_than => %s, older_than => %s) c",
                (TABLE, lo, hi),
            )

        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET maintenance_work_mem = '2GB'")

            store = db.DBNStore.from_file(path)
            tmpdir = tempfile.mkdtemp(prefix="mbo_", dir="/tmp")
            fifo = os.path.join(tmpdir, "csv.pipe")
            os.mkfifo(fifo)
            err: list[BaseException] = []

            def _write_csv() -> None:
                try:
                    store.to_csv(fifo, pretty_px=True, pretty_ts=True,
                                 map_symbols=True, mode="w")
                except BaseException as exc:      # noqa: BLE001 - propagated below
                    err.append(exc)
                    # Unblock a reader that is still waiting on the pipe.
                    try:
                        with open(fifo, "wb"):
                            pass
                    except OSError:
                        pass

            writer = threading.Thread(target=_write_csv, daemon=True)
            writer.start()
            try:
                # DELETE + COPY in one transaction so each file is all-or-nothing. If the
                # CSV encoder dies mid-stream, COPY sees a short but *structurally valid*
                # stream and would otherwise commit a truncated day that looks complete.
                with conn.transaction():
                    cur.execute(
                        f"DELETE FROM {TABLE} WHERE ts_recv >= %s AND ts_recv < %s",
                        (lo, hi),
                    )
                    deleted = cur.rowcount
                    with cur.copy(copy_sql) as copy:
                        with open(fifo, "rb") as pipe:
                            while block := pipe.read(READ_BLOCK):
                                copy.write(block)
                    rows = cur.rowcount
                    writer.join(timeout=300)
                    if err:
                        raise RuntimeError(
                            f"CSV encoder failed for {path.name}") from err[0]
                    if writer.is_alive():
                        raise RuntimeError(
                            f"CSV encoder still running after COPY drained {path.name}"
                            " — refusing to commit a possibly truncated day")
            finally:
                writer.join(timeout=60)
                os.unlink(fifo)
                os.rmdir(tmpdir)

        # Compress immediately: an uncompressed MBO day is ~1-2 GB.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM ("
                "  SELECT compress_chunk(c, if_not_compressed => true) "
                "  FROM show_chunks(%s, newer_than => %s, older_than => %s) c) s",
                (TABLE, lo, hi),
            )
            n_chunks = cur.fetchone()[0]

    return {
        "file": path.name, "rows": rows, "deleted": deleted, "chunks": n_chunks,
        "start": lo.isoformat(), "end": hi.isoformat(),
        "secs": round(time.time() - t0, 1),
    }


def _load_ckpt(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"done": {}}


def _save_ckpt(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest MBO .dbn.zst files into databento.mbo")
    ap.add_argument("--data-dir", type=Path, action="append", default=[])
    ap.add_argument("--glob", default="**/*.dbn.zst")
    ap.add_argument("--files", type=Path, nargs="+", default=[])
    ap.add_argument("--workers", type=int, default=3,
                    help="Concurrent files. Each holds ~1-2 GB of uncompressed chunk "
                         "until its post-COPY compression finishes — keep this low.")
    ap.add_argument("--checkpoint", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--reset-checkpoint", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(args.files) if args.files else sorted(
        {p for d in args.data_dir for p in d.glob(args.glob)}
    )
    if not files:
        raise SystemExit("no .dbn.zst files found — pass --files or --data-dir")

    ckpt = {"done": {}} if args.reset_checkpoint else _load_ckpt(args.checkpoint)
    pending = [p for p in files if p.name not in ckpt["done"]]

    print(f"{len(files)} file(s) found, {len(pending)} pending, workers={args.workers}",
          flush=True)
    if args.dry_run:
        total = 0
        for p in pending:
            sz = p.stat().st_size
            total += sz
            lo, hi = _file_range_ns(p)
            print(f"  {p.name:<44} {sz/1e6:>8.1f} MB  "
                  f"{datetime.fromtimestamp(lo/1e9, timezone.utc):%Y-%m-%d %H:%M} -> "
                  f"{datetime.fromtimestamp(hi/1e9, timezone.utc):%Y-%m-%d %H:%M}")
        print(f"  {'TOTAL':<44} {total/1e6:>8.1f} MB compressed")
        return

    dsn = _dsn()
    t0, done_rows = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_ingest_file, str(p), dsn): p for p in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:                     # noqa: BLE001
                print(f"[{i}/{len(pending)}] FAIL {p.name}: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            done_rows += res["rows"]
            ckpt["done"][p.name] = res
            _save_ckpt(args.checkpoint, ckpt)
            rate = res["rows"] / max(res["secs"], 1e-9)
            print(f"[{i}/{len(pending)}] {res['file']:<44} "
                  f"{res['rows']:>12,} rows  {res['secs']:>7.1f}s  "
                  f"{rate:>8,.0f} rows/s  chunks={res['chunks']}"
                  + (f"  (replaced {res['deleted']:,})" if res["deleted"] else ""),
                  flush=True)

    mins = (time.time() - t0) / 60
    print(f"\ningested {done_rows:,} rows in {mins:.1f} min "
          f"({done_rows/max(mins*60,1e-9):,.0f} rows/s aggregate)")


if __name__ == "__main__":
    main()
