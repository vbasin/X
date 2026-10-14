#!/usr/bin/env python3
"""Backfill 0DTE NQ option Greeks + compact surface (DB + parquet).

For each trading day with a 0DTE chain in ``databento.definitions``:
  1. Load two-sided bbo_1s for those instruments (day-chunk pruned)
  2. Join NQ.c.0 1s close; compute full BS Greeks once
  3. COPY into ``databento.nq_option_greeks_1s``
  4. Collapse to 8-leg delta-bucket surface → ``databento.nq_option_surface_1s``
  5. Write ``data/sources/features/parquet_options_surface/features_YYYY-MM-DD.parquet``

Resumable (skips days that already have surface rows). Use ``--force`` to rebuild.

Usage::

    python scripts/backfill_options_greeks.py --start 2025-04-07 --end 2026-05-01
    python scripts/backfill_options_greeks.py --dates 2026-04-17          # single day smoke
    python scripts/backfill_options_greeks.py --start 2026-04-01 --end 2026-04-30 --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.common.cme_holidays import is_trading_day  # noqa: E402
from src.common.vm_config import get_vm_config  # noqa: E402
from src.features.options_greeks_pipeline import (  # noqa: E402
    ensure_tables,
    process_day,
)

logger = logging.getLogger("vtech.backfill.options_greeks")


def _free_gb(path: str = "/opt/vtech/data") -> float:
    import shutil
    return shutil.disk_usage(path).free / 1e9


def _worker(args: tuple) -> dict:
    """Process one day in a child process (fresh DB connection)."""
    d_iso, force, write_db, write_parquet, write_greeks_pq, pq_dir, write_greeks_db = args
    d = date.fromisoformat(d_iso)
    try:
        return process_day(
            d,
            force=force,
            write_db=write_db,
            write_greeks_db=write_greeks_db,
            write_parquet=write_parquet,
            write_greeks_parquet=write_greeks_pq,
            parquet_dir=Path(pq_dir),
        )
    except Exception as exc:  # noqa: BLE001 — report per-day failure, keep going
        return {"date": d_iso, "skipped": False, "error": str(exc)}


def _iter_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def main() -> int:
    cfg = get_vm_config()
    ap = argparse.ArgumentParser(description="Backfill 0DTE NQ option Greeks + surface")
    ap.add_argument("--start", help="inclusive YYYY-MM-DD")
    ap.add_argument("--end", help="inclusive YYYY-MM-DD")
    ap.add_argument("--dates", nargs="+", help="explicit date list (overrides start/end)")
    ap.add_argument("--workers", type=int, default=cfg.workers_for(mem_per_worker_gb=8.0),
                    help="parallel day workers (default: auto from VM RAM)")
    ap.add_argument("--force", action="store_true", help="rebuild days that already have surface rows")
    ap.add_argument("--no-db", action="store_true", help="skip Postgres writes (parquet only)")
    ap.add_argument("--no-greeks-db", action="store_true",
                    help="skip the nq_option_greeks_1s COPY (77%% of a day's runtime; "
                         "nothing in this repo reads that table)")
    ap.add_argument("--no-parquet", action="store_true", help="skip surface parquet writes")
    ap.add_argument("--greeks-parquet", action="store_true",
                    help="also write full-greeks day parquet (doubles disk; off by default)")
    ap.add_argument("--min-free-gb", type=float, default=8.0,
                    help="abort remaining days if free disk on data volume drops below this")
    ap.add_argument("--parquet-dir", default="/opt/vtech/data/sources/features/parquet_options_surface")
    ap.add_argument("--ddl-only", action="store_true", help="create tables and exit")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    print(cfg, flush=True)

    ensure_tables()
    if args.ddl_only:
        print("DDL applied — tables ready.", flush=True)
        return 0

    if args.dates:
        dates = [date.fromisoformat(x) for x in args.dates]
    else:
        if not args.start or not args.end:
            ap.error("provide --start/--end or --dates")
        dates = _iter_dates(date.fromisoformat(args.start), date.fromisoformat(args.end))

    if not dates:
        print("No trading days in range.", flush=True)
        return 0

    work = [
        (d.isoformat(), args.force, not args.no_db, not args.no_parquet,
         args.greeks_parquet, args.parquet_dir, not args.no_greeks_db)
        for d in dates
    ]
    print(
        f"===== OPTIONS GREEKS BACKFILL — {len(work)} days, workers={args.workers} "
        f"greeks_parquet={args.greeks_parquet} greeks_db={not args.no_greeks_db} "
        f"min_free={args.min_free_gb}GB "
        f"free_now={_free_gb():.1f}GB =====",
        flush=True,
    )

    t0 = time.time()
    done = skipped = failed = aborted = 0
    n_greeks = n_surface = 0
    results: list[dict] = []

    # Sequential-or-parallel with disk gate. Parallel still checks free disk
    # before submitting each new day.
    if args.workers <= 1:
        for w in work:
            if _free_gb() < args.min_free_gb:
                print(f"  ABORT free disk {_free_gb():.1f} GB < {args.min_free_gb} GB "
                      f"(stopping before {w[0]})", flush=True)
                aborted = len(work) - len(results)
                break
            results.append(_worker(w))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            wi = 0
            while wi < len(work) or pending:
                while wi < len(work) and len(pending) < args.workers:
                    if _free_gb() < args.min_free_gb:
                        print(f"  ABORT free disk {_free_gb():.1f} GB < {args.min_free_gb} GB",
                              flush=True)
                        aborted = len(work) - wi - len(pending)
                        wi = len(work)  # stop submitting
                        break
                    w = work[wi]
                    pending[pool.submit(_worker, w)] = w[0]
                    wi += 1
                if not pending:
                    break
                # Wait for any completion
                from concurrent.futures import wait, FIRST_COMPLETED
                done_futs, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    d_iso = pending.pop(fut)
                    r = fut.result()
                    results.append(r)
                    # live progress line
                    if r.get("error"):
                        print(f"  FAIL {r['date']}: {r['error']} | free {_free_gb():.1f}GB",
                              flush=True)
                    elif r.get("skipped"):
                        print(f"  skip {r['date']}: {r.get('reason', '')}", flush=True)
                    else:
                        print(
                            f"  ok   {r['date']}: greeks={r.get('n_greeks', 0):,} "
                            f"surface={r.get('n_surface', 0):,} "
                            f"legsμ={r.get('n_legs_mean', 0):.1f} "
                            f"({r.get('elapsed_s', 0)}s) | free {_free_gb():.1f}GB",
                            flush=True,
                        )

    # Stable summary
    results.sort(key=lambda r: r.get("date", ""))
    for r in results:
        if r.get("error"):
            failed += 1
        elif r.get("skipped"):
            skipped += 1
        else:
            done += 1
            n_greeks += int(r.get("n_greeks", 0))
            n_surface += int(r.get("n_surface", 0))

    elapsed = time.time() - t0
    print(
        f"===== DONE — ok={done} skipped={skipped} failed={failed} aborted={aborted} "
        f"greeks_rows={n_greeks:,} surface_rows={n_surface:,} "
        f"elapsed={elapsed / 60:.1f}m free={_free_gb():.1f}GB =====",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
