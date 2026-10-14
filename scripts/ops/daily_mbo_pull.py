#!/usr/bin/env python3
"""Manual MBO catch-up: pull NQ lead-contract MBO days the live streamer missed, $0-gated,
ingest, verify.

Historical/manual tool only — as of 2026-09-29 MBO is captured live (LiveStreamer subscribes
to schema="mbo" alongside tbbo/bbo-1s; see ADR-145/146), the same architecture as tbbo. There is
no recurring timer for this anymore (vtech-mbo-daily.timer was retired); run this by hand after
an outage, a missed session, or before the live-streaming cutover date for the one-time gap-fill.
Everything heavy is reused:

    scripts/data/backfill_mbo.py   cost gate (checked twice) -> day-split batch -> download -> ingest
    scripts/data/ingest_mbo.py     DBN -> FIFO -> COPY into databento.mbo, compress per day

This script only decides WHAT to pull and proves it landed:

  1. Disk guard: refuse to start below --min-free-gb on /opt/vtech/data (a full volume crashed
     PostgreSQL on 2026-09-25).
  2. First candidate day = the last UTC day in databento.mbo (re-checked against Databento's
     record count, so a partial last day is re-pulled), last = the newest UTC day fully covered
     by the feed (`get_dataset_range` mbo end). This script only fills *whole* days; a gap-fill
     that must stop mid-day at an exact timestamp (e.g. live streaming's first captured row) is
     a job for `backfill_mbo.py SYMBOL:day:HH:MM:SS` directly — see ADR-146.
  3. Per day the symbols are {lead(D), lead(D+1)} via `active_contract`, so a roll day pulls both
     contracts. Days with 0 records (weekends, holidays) are skipped.
  4. Consecutive days per symbol become one segment for backfill_mbo.py; a non-$0.00 cost aborts
     with nothing submitted (AGENTS.md §5).
  5. Verification: databento.mbo rows per (symbol, day) must equal Databento's record count —
     logged as a warning on mismatch, not fatal (see the note in main() on the 2026-09-25 day
     that never resolved: Databento's own get_record_count vs. batch-delivery discrepancy).

Exit codes: 0 ok / up to date (verification mismatches included — see above), 1 pull/ingest
command failure, 2 refused (disk / another run already holding the lock).
The Databento free L2/L3 window is ~1 month rolling, so a gap missed for that long cannot be
recovered at $0.

Usage:
    python scripts/ops/daily_mbo_pull.py --dry-run
    python scripts/ops/daily_mbo_pull.py
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path("/opt/vtech")
DATA = ROOT / "data"
LOCK = ROOT / "logs" / "mbo-daily.lock"
UTC = timezone.utc
log = logging.getLogger("mbo-daily")


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    a = datetime(d.year, d.month, d.day, tzinfo=UTC)
    return a, a + timedelta(days=1)


def plan(client, conn, dataset: str, first_pull_day: date | None = None
         ) -> tuple[list[tuple[str, date, int]], date | None]:
    """Days still to pull: [(symbol, day, databento_record_count)], plus the last covered day."""
    from src.common.contract_calendar import active_contract

    end = client.metadata.get_dataset_range(dataset)["schema"]["mbo"]["end"]
    avail_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    last_full = (avail_end - timedelta(days=1)).date()        # D fully covered iff D+1 00:00Z <= end
    if avail_end < datetime(last_full.year, last_full.month, last_full.day, tzinfo=UTC) + timedelta(days=1):
        last_full -= timedelta(days=1)

    with conn.cursor() as cur:
        cur.execute("SELECT max(ts_recv) FROM databento.mbo")
        mx = cur.fetchone()[0]
    start = first_pull_day or (mx.astimezone(UTC).date() if mx else last_full)   # re-check DB's last day
    log.info("db max ts_recv=%s  feed end=%s  candidate days %s..%s", mx, end, start, last_full)

    todo: list[tuple[str, date, int]] = []
    d = start
    while d <= last_full:
        lo, hi = _day_bounds(d)
        for sym in sorted({active_contract("NQ", d), active_contract("NQ", d + timedelta(days=1))}):
            n = client.metadata.get_record_count(dataset=dataset, symbols=[sym], schema="mbo",
                                                 stype_in="raw_symbol", start=lo.isoformat(),
                                                 end=hi.isoformat())
            if n == 0:
                continue
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM databento.mbo WHERE ts_recv >= %s AND ts_recv < %s "
                            "AND symbol = %s", (lo, hi, sym))
                have = cur.fetchone()[0]
            if have != n:
                todo.append((sym, d, n))
                log.info("need %s %s: db has %s of %s records", sym, d, f"{have:,}", f"{n:,}")
        d += timedelta(days=1)
    return todo, last_full


def segments(todo: list[tuple[str, date, int]]) -> list[str]:
    """Merge consecutive days per symbol into SYMBOL:START:END(exclusive) for backfill_mbo.py."""
    segs, by_sym = [], {}
    for sym, d, _ in sorted(todo, key=lambda t: (t[0], t[1])):
        by_sym.setdefault(sym, []).append(d)
    for sym, days in by_sym.items():
        s = prev = days[0]
        for d in days[1:] + [None]:
            if d is not None and d == prev + timedelta(days=1):
                prev = d
                continue
            segs.append(f"{sym}:{s.isoformat()}:{(prev + timedelta(days=1)).isoformat()}")
            if d is not None:
                s = prev = d
    return segs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="plan + cost only, no download/ingest")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-days", type=int, default=10, help="refuse a plan larger than this")
    ap.add_argument("--from-day", type=date.fromisoformat, default=None,
                    help="manual catch-up: first UTC day to consider (default: DB's last day)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | mbo-daily | %(levelname)-5s | %(message)s")

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.error("another daily_mbo_pull is running — exiting")
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
    from src.acquisition.schemas import DATASET

    client = db.Historical()
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        todo, last_full = plan(client, conn, DATASET, a.from_day)
        if not todo:
            log.info("nothing to pull (feed fully covers through %s; databento.mbo already matches)", last_full)
            return 0
        if len({d for _, d, _ in todo}) > a.max_days:
            log.error("plan spans %d days (> --max-days %d) — refusing, run backfill_mbo.py by hand",
                      len({d for _, d, _ in todo}), a.max_days)
            return 2
        segs = segments(todo)
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        log.info("plan: %s", segs)
        cmd = [sys.executable, str(ROOT / "scripts/data/backfill_mbo.py"), "--workers", str(a.workers), *segs]
        if a.dry_run:
            subprocess.run(cmd[:2] + ["--dry-run", *segs], check=True, cwd=ROOT,
                           env=env)
            return 0
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        if rc != 0:
            log.error("backfill_mbo.py exited %s", rc)
            return 1

        # A verify mismatch no longer hard-fails the run (exit 1) — it used to,
        # and on 2026-09-25 got stuck permanently mismatched (databento's
        # get_record_count for that day reports 810 more rows than three
        # separate re-pulls of the identical [symbols, schema, stype_in, range]
        # ever delivered; the discrepancy is on Databento's side, not ours, and
        # re-pulling does not change it). That stuck day blocked every later
        # day from ever being recognized as caught-up while this ran nightly.
        # Now that this is a manual/occasional tool rather than a timer,
        # mismatches are surfaced clearly but only a pull/ingest command
        # failure (handled above) or a DB error here counts as exit 1.
        mismatches = []
        for sym, d, n in todo:
            lo, hi = _day_bounds(d)
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM databento.mbo WHERE ts_recv >= %s AND ts_recv < %s "
                            "AND symbol = %s", (lo, hi, sym))
                have = cur.fetchone()[0]
            ok = have == n
            if not ok:
                mismatches.append((sym, d, have, n))
            log.log(logging.INFO if ok else logging.WARNING, "verify %s %s: db=%s databento=%s %s",
                    sym, d, f"{have:,}", f"{n:,}", "OK" if ok else "MISMATCH")
        if mismatches:
            log.warning("%d day(s) verified short of Databento's reported count — pulled and ingested "
                        "regardless; re-run this tool by hand later to retry, or accept as a known "
                        "get_record_count/delivery discrepancy if it persists across re-runs: %s",
                        len(mismatches),
                        ", ".join(f"{s} {d} ({h:,}/{n:,})" for s, d, h, n in mismatches))
        log.info("free space after: %.1f GB", shutil.disk_usage(DATA).free / 1e9)
        return 0


if __name__ == "__main__":
    sys.exit(main())
