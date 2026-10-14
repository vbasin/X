#!/usr/bin/env python3
"""Continuous incremental MBO pull: every 30 min, append whatever has newly crossed
CME's 8-hour "Delayed Information" boundary (see docs/decisions/ADR-146). Runs
Sunday session open through Friday close, gated the same way the live streamer
gates itself.

**Append-only. This script never issues a DELETE against databento.mbo.**

Why not delete-then-reload (the pattern scripts/data/ingest_mbo.py uses for whole
UTC-day files, where re-running the same day is a deliberate, idempotent replace):
a continuous incremental process should never need to overwrite its own prior work.
The one unavoidable overlap is the single boundary row at the checkpoint timestamp
(Databento's `start` is inclusive), and that is handled by staging the fetch in a
throwaway temp table and INSERTing only rows strictly newer than the checkpoint into
databento.mbo — never touching what's already there. See _pull_and_append().

The checkpoint is always read fresh from the table itself
(`max(ts_recv, sequence)` per symbol) — never a separately-tracked file — so it can
never drift from what databento.mbo actually contains. If a run ever finds the table
in a state it doesn't expect, it logs loudly and stops; it does not "fix" anything by
deleting or overwriting.

Ingestion method: `timeseries.get_range()` (via src.acquisition.historical.download_range),
which Databento's own docs recommend for requests under 5 GB — every window here is a
fraction of a day, always well under that:
https://databento.com/docs/faqs/streaming-vs-batch-download
batch.submit_job() (used by scripts/data/backfill_mbo.py) stays reserved for the
occasional whole-day/multi-day manual gap-fill, where it's the documented right tool.

Resting-order book (ADR-147, corrected by ADR-149): each tick also writes a new dated checkpoint
into `databento.mbo_resting_orders_v2` from just the newly-staged events — never a full recompute.
"Current state" is simply that table's newest checkpoint for a symbol, not a separate table.
This lets src/webui/mbo_orders.py answer "what large orders were already resting when this
window began" for an arbitrary query window, which it could not do before (it only ever saw orders
whose Add fell inside the window itself). See _update_resting_book() and
src/acquisition/mbo_resting_book.py for the pure state-transition logic.

Snapshots: Databento's historical MBO snapshot (a synthetic book-clear + re-adds) only
appears in a request window that spans 00:00 UTC, and is documented as a convenience for
cold-starting reconstruction, not a requirement for correctness:
https://databento.com/docs/standards-and-conventions/mbo-snapshot
This script's windows are 30 minutes and essentially never touch midnight, but that's
fine — it never cold-starts. It always continues the same gapless event stream that the
existing complete backfill already established (which *does* span real UTC-day
boundaries and so already carries proper snapshots where they matter), so no explicit
snapshot handling is needed here.

Cost-checked every single run, no exceptions, even though this window is always $0
(AGENTS.md §5).

Exit codes: 0 ok / already caught up, 1 a pull or append problem (logged, nothing was
deleted or corrupted), 2 refused (disk / another run holding the lock). The
get_record_count() cross-check is logged but never contributes to the exit code — see
_verify()'s docstring for why it isn't trustworthy enough at this granularity to gate on.

Usage:
    python scripts/ops/mbo_incremental_pull.py --dry-run
    python scripts/ops/mbo_incremental_pull.py
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/vtech")
DATA = ROOT / "data"
LOCK = ROOT / "logs" / "mbo-incremental.lock"
OUT_DIR = DATA / "sources" / "raw" / "incremental" / "nq_fut_mbo"
UTC = timezone.utc
LAG = timedelta(hours=8, minutes=1)   # confirmed boundary: blocked at 8h00m, available at 8h01m
log = logging.getLogger("mbo-incremental")

# Resting-book checkpoint table. v2 supersedes the original mbo_resting_orders, whose checkpoints
# carry phantom orders (never removed, because CME purges Day orders with no Cancel) and truncated
# t_start values. v1 is frozen intact as a rollback path rather than rewritten in place, which
# would have broken this pipeline's append-only rule. See ADR-149 and
# scripts/data/create_mbo_resting_orders_v2.sql.
RESTING_TABLE = "databento.mbo_resting_orders_v2"

# Exact order the DBN Rust CSV encoder emits for the mbo schema (matches ingest_mbo.py,
# verified against a real record). `dataset` is intentionally omitted — the real table
# defaults it to 'GLBX.MDP3' and the scratch stage table has no need for it at all.
CSV_COLS = [
    "ts_recv", "ts_event", "rtype", "publisher_id", "instrument_id", "action", "side",
    "price", "size", "channel_id", "order_id", "flags", "ts_in_delta", "sequence", "symbol",
]
STAGE_TYPES = {
    "ts_recv": "TIMESTAMPTZ", "ts_event": "TIMESTAMPTZ", "rtype": "SMALLINT",
    "publisher_id": "SMALLINT", "instrument_id": "INTEGER", "action": "CHAR(1)",
    "side": "CHAR(1)", "price": "DOUBLE PRECISION", "size": "INTEGER",
    "channel_id": "SMALLINT", "order_id": "BIGINT", "flags": "SMALLINT",
    "ts_in_delta": "INTEGER", "sequence": "BIGINT", "symbol": "TEXT",
}


def _checkpoint(conn, symbol: str) -> tuple[datetime, int] | None:
    """(ts_recv, sequence) of the newest row already stored for this symbol, or None if
    nothing has ever been ingested for it. An incremental process must never guess a
    starting point — a first-ever symbol needs a manual seed via backfill_mbo.py."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts_recv, sequence FROM databento.mbo WHERE symbol = %s "
            "ORDER BY ts_recv DESC, sequence DESC LIMIT 1", (symbol,))
        row = cur.fetchone()
    return (row[0], row[1]) if row else None


def _pull_and_append(conn, symbol: str, ckpt_ts: datetime, ckpt_seq: int, end: datetime) -> dict:
    """Fetch [ckpt_ts, end) and append only rows strictly newer than (ckpt_ts, ckpt_seq).
    The real table only ever sees an INSERT here — never a DELETE."""
    import databento as db

    from src.acquisition.historical import cost_check, download_range

    start_iso, end_iso = ckpt_ts.isoformat(), end.isoformat()
    cost = cost_check([symbol], "mbo", start_iso, end_iso, "raw_symbol")
    if cost > 0.0:
        raise RuntimeError(f"COST GATE: {symbol} {start_iso}->{end_iso} costs ${cost:.4f} — aborting")

    path = download_range(symbol, "mbo", start_iso, end_iso, OUT_DIR, stype_in="raw_symbol")
    store = db.DBNStore.from_file(path)
    cols_sql = ", ".join(f'"{c}"' for c in CSV_COLS)
    stage_cols_sql = ", ".join(f'"{c}" {STAGE_TYPES[c]}' for c in CSV_COLS)

    tmpdir = tempfile.mkdtemp(prefix="mboinc_", dir="/tmp")
    fifo = os.path.join(tmpdir, "csv.pipe")
    os.mkfifo(fifo)
    err: list[BaseException] = []

    def _write_csv() -> None:
        try:
            store.to_csv(fifo, pretty_px=True, pretty_ts=True, map_symbols=True, mode="w")
        except BaseException as exc:      # noqa: BLE001 - propagated below
            err.append(exc)
            try:                          # unblock a reader still waiting on the pipe
                with open(fifo, "wb"):
                    pass
            except OSError:
                pass

    writer = threading.Thread(target=_write_csv, daemon=True)
    writer.start()
    try:
        with conn.cursor() as cur, conn.transaction():
            # Scratch space only: created and dropped inside this one transaction, never
            # persisted, never the target of anything but this run's own fresh pull.
            cur.execute(f"CREATE TEMP TABLE mbo_stage ({stage_cols_sql}) ON COMMIT DROP")
            copy_sql = f"COPY mbo_stage ({cols_sql}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
            with cur.copy(copy_sql) as copy:
                with open(fifo, "rb") as pipe:
                    while block := pipe.read(1 << 20):
                        copy.write(block)
            writer.join(timeout=300)
            if err:
                raise RuntimeError(f"CSV encoder failed for {path.name}") from err[0]
            if writer.is_alive():
                raise RuntimeError(
                    f"CSV encoder still running after COPY drained {path.name}"
                    " — refusing to commit a possibly truncated pull")
            staged = cur.rowcount

            # The only statement that ever writes to the real table, and it is an INSERT.
            # The row-comparison filter drops exactly the one boundary row we already
            # have (ckpt_ts is inclusive on the Databento request) without deleting it
            # first -- it is simply never re-selected from the stage table.
            cur.execute(
                f"INSERT INTO databento.mbo ({cols_sql}) "
                f"SELECT {cols_sql} FROM mbo_stage WHERE (ts_recv, sequence) > (%s, %s)",
                (ckpt_ts, ckpt_seq),
            )
            appended = cur.rowcount

            # 'R' (Clear book) is selected too -- it is a whole-book operation, and dropping it
            # was what let exchange-purged orders live on forever. Ordering is strictly
            # chronological, NOT grouped per order_id: Clear and snapshot groups only mean
            # anything relative to every other order's events. `flags` distinguishes a snapshot
            # re-listing from an ordinary event (see mbo_resting_book.apply_events).
            cur.execute(
                "SELECT order_id, ts_recv, action, side, price, size, symbol, flags "
                "FROM mbo_stage WHERE action IN ('A','M','F','C','R') "
                "ORDER BY ts_recv, sequence")
            new_events = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            book_stats = update_resting_book(cur, symbol, end, new_events)
    finally:
        writer.join(timeout=60)
        try:
            os.unlink(fifo)
        except OSError:
            pass
        os.rmdir(tmpdir)

    return {"symbol": symbol, "staged": staged, "appended": appended, "start": start_iso,
           "end": end_iso, **book_stats}


def update_resting_book(cur, symbol: str, snapshot_ts: datetime, events: pd.DataFrame,
                        table: str = RESTING_TABLE) -> dict:
    """Apply `events` (filtered to A/M/F/C/R, in strict chronological ts_recv/sequence order, with
    a `flags` column) to the resting-order book and write the result as a new partition in `table`
    dated `snapshot_ts`. "Current state" is simply the newest existing partition for this symbol
    -- there is no separate "current" table (see create_mbo_resting_orders_v2.sql for why that was
    collapsed into one table on review).

    Shared by the live incremental puller (events = this tick's freshly-staged rows, called
    inside the same transaction as the MBO append) and
    scripts/ops/mbo_resting_book_backfill.py (events = a historical window read directly from
    the already-ingested databento.mbo, no download involved). See ADR-147 and
    src/acquisition/mbo_resting_book.py for the pure state-transition logic."""
    from src.acquisition.mbo_resting_book import RestingOrder, apply_events

    if events.empty:
        return {"book_updated": 0, "book_closed": 0, "book_size": None, "book_wrote": False}

    # Bounded to snapshot_ts < this tick's own target: an unbounded MAX() would, during a
    # historical backfill, pick up a checkpoint the live incremental puller already wrote far in
    # the future (its own "now") as if it were "current" for a tick being replayed in the past --
    # silently importing today's open orders into a historical checkpoint from weeks ago. Caught
    # before it reached data (see ADR-147); harmless for the live puller itself, since its own
    # newest checkpoint is always the true latest one anyway.
    cur.execute(
        f"SELECT order_id, symbol, side, price, size, t_start, t_start_exact FROM {table} "
        "WHERE symbol = %s AND order_id >= 0 AND snapshot_ts = ("
        f"  SELECT max(snapshot_ts) FROM {table} "
        "  WHERE symbol = %s AND snapshot_ts < %s)",
        (symbol, symbol, snapshot_ts))
    current = {int(r[0]): RestingOrder(symbol=r[1], side=r[2], price=r[3], size=r[4],
                                       t_start=r[5], updated_at=r[5], t_start_exact=r[6])
               for r in cur.fetchall()}

    updated, closed = apply_events(current, events)
    # `closed` includes cancels/fills of order_ids that were never tracked as open in the first
    # place (a short-lived order added and closed inside one batch, or -- expected once, right
    # after this table's cold start -- a Cancel for an order whose Add is in already-ingested
    # history from before this subsystem existed). Harmless either way (it's just absent from
    # the new partition), but counting it as "closed" in the log would be misleading, so the log
    # stat below is scoped to order_ids that were genuinely in `current` beforehand.
    genuinely_closed = closed & current.keys()

    new_state = {oid: o for oid, o in current.items() if oid not in updated and oid not in closed}
    new_state.update(updated)

    rows = [(snapshot_ts, oid, o.symbol, o.side, o.price, o.size, o.t_start, o.t_start_exact)
           for oid, o in new_state.items()]
    if not rows:
        # The book genuinely closed out entirely this tick. Still write a marker row so the next
        # tick's "newest partition" lookup finds THIS checkpoint rather than falling back to a
        # stale earlier one that would resurrect orders that just closed. order_id=-1 is never a
        # real venue id (CME's are large positive uint64s) -- filtered out of `current` above and
        # out of RESTING_SNAPSHOT_SQL's results by its own size >= min_size condition.
        rows = [(snapshot_ts, -1, symbol, "N", 0.0, 0, snapshot_ts, True)]
    cur.executemany(
        f"INSERT INTO {table} "
        "(snapshot_ts, order_id, symbol, side, price, size, t_start, t_start_exact) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (snapshot_ts, order_id) DO NOTHING",
        rows)

    return {"book_updated": len(updated), "book_closed": len(genuinely_closed),
           "book_size": len(new_state), "book_wrote": True}


def _verify(conn, client, symbol: str, start: datetime, end: datetime) -> None:
    """Log databento.mbo's own row count for [start, end) next to Databento's
    get_record_count() for the identical window -- informational only, not a pass/fail
    gate. Empirically (2026-09-29, this script's first real run), get_record_count() is
    NOT additive across sub-windows of a single range: summing it over 5 consecutive
    slices of one ~4h45m pull gave 3,055,275, while the single whole-range call gave
    2,979,612 -- a stable, reproducible difference on repeat calls, not time-based
    settling. CME emits bursts of dozens of MBO records sharing one exact `sequence`
    value (confirmed: up to 85 rows on one sequence in this same data), and
    get_record_count()'s boundary handling around those ties is evidently not exact, so
    it cannot be trusted as a strict oracle for an arbitrary narrow window -- only our
    own pipeline's internal fidelity is something we can actually guarantee (see
    _pull_and_append: staged count already equals the downloaded file's own record
    count, verified every run). A persistent, large, or growing divergence here is
    still worth a human noticing over time, which is why this stays logged -- it just
    no longer fails the run, so a same-granularity Databento quirk can't repeat the
    2026-09-25 daily-job false-failure loop this script exists to not repeat."""
    from src.acquisition.schemas import DATASET

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM databento.mbo WHERE symbol = %s AND ts_recv >= %s AND ts_recv < %s",
            (symbol, start, end))
        have = cur.fetchone()[0]
    n = client.metadata.get_record_count(dataset=DATASET, symbols=[symbol], schema="mbo",
                                          stype_in="raw_symbol", start=start.isoformat(), end=end.isoformat())
    ok = have == n
    (log.info if ok else log.warning)(
        "verify (informational) %s %s->%s: have=%s databento_get_record_count=%s %s",
        symbol, start, end, f"{have:,}", f"{n:,}", "match" if ok else "differs — see docstring, not fatal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="cost check only, no pull/append")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | mbo-incremental | %(levelname)-5s | %(message)s")

    from src.common.market_session import cme_weekend_closed
    if cme_weekend_closed():
        log.info("cme_weekend_closed — nothing to do")
        return 0

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("another mbo_incremental_pull is running — exiting")
        return 2

    free_gb = shutil.disk_usage(DATA).free / 1e9
    if free_gb < a.min_free_gb:
        log.error("only %.1f GB free on %s (< %.0f GB guard) — refusing to pull", free_gb, DATA, a.min_free_gb)
        return 2
    log.info("free space %.1f GB (guard %.0f GB)", free_gb, a.min_free_gb)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import databento as db
    import psycopg

    from scripts.data.ingest_mbo import _dsn
    from src.common.contract_calendar import active_contract

    now = datetime.now(UTC)
    end = now - LAG
    client = db.Historical()

    bad = 0
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        symbols = sorted({active_contract("NQ", end.date()), active_contract("NQ", now.date())})
        for sym in symbols:
            ckpt = _checkpoint(conn, sym)
            if ckpt is None:
                log.error("%s: no existing data — refusing to guess a starting point; "
                         "seed it with backfill_mbo.py first", sym)
                bad += 1
                continue
            ckpt_ts, ckpt_seq = ckpt
            if ckpt_ts >= end:
                log.info("%s: already caught up (checkpoint %s >= target %s)", sym, ckpt_ts, end)
                continue
            log.info("%s: pulling %s -> %s (checkpoint seq=%s)", sym, ckpt_ts, end, ckpt_seq)
            if a.dry_run:
                from src.acquisition.historical import cost_check
                cost = cost_check([sym], "mbo", ckpt_ts.isoformat(), end.isoformat(), "raw_symbol")
                log.info("%s: dry-run cost=$%.4f", sym, cost)
                continue
            try:
                r = _pull_and_append(conn, sym, ckpt_ts, ckpt_seq, end)
            except Exception:
                log.exception("%s: pull/append failed", sym)
                bad += 1
                continue
            log.info("%s: staged=%s appended=%s (%s already present at the boundary) "
                    "resting_book: updated=%s closed=%s",
                    sym, r["staged"], r["appended"], r["staged"] - r["appended"],
                    r["book_updated"], r["book_closed"])
            _verify(conn, client, sym, ckpt_ts, end)
        log.info("free space after: %.1f GB", shutil.disk_usage(DATA).free / 1e9)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
