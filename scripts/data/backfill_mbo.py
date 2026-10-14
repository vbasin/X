#!/usr/bin/env python3
"""Free-window MBO backfill: cost gate -> day-split batch job -> download -> ingest.

Reuses src.acquisition.historical (cost_check / submit_batch / poll / download) and
scripts/data/ingest_mbo.py. Adds only the segment plan and a hard $0.00 gate:
the cost is re-checked for every segment immediately before submit_job, and any
non-zero cost aborts with nothing submitted (AGENTS.md §5).

Segments are given as SYMBOL:START:END (END exclusive; START on a UTC midnight so Databento
includes a synthetic book snapshot at each day start; END may be a timestamp such as
2026-09-25T21:00:00 when the feed has not reached the next midnight yet):

    python scripts/data/backfill_mbo.py --dry-run \\
        NQU6:2026-08-23:2026-09-13 NQZ6:2026-09-13:2026-09-25
    python scripts/data/backfill_mbo.py NQU6:2026-08-23:2026-09-13 NQZ6:2026-09-13:2026-09-25

Raw .dbn.zst files are retained (the lossless archive of record).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import databento as db

from src.acquisition.historical import cost_check, download_batch, poll_batch_until_done, submit_batch

OUT_DIR = Path("/opt/vtech/data/sources/raw/backfill/nq_fut_mbo")


def _free_gb(path: Path) -> float:
    """Free space (GB, decimal) on the filesystem holding `path`."""
    return shutil.disk_usage(path).free / 1e9


def _assert_key(expected: str | None) -> None:
    """Fail closed if the run would bill a different Databento account than intended."""
    if not expected:
        return
    got = os.environ.get("DATABENTO_API_KEY", "")
    if got != expected:
        sys.exit("API KEY GATE: DATABENTO_API_KEY does not match --require-key "
                 f"(env key ends {got[-4:]!r}, required ends {expected[-4:]!r}) — aborting")
    print(f"api key gate OK (…{got[-4:]})", flush=True)


def _billable(sym: str, start: str, end: str) -> tuple[int, int]:
    from src.acquisition.schemas import DATASET
    kw = dict(dataset=DATASET, symbols=[sym], schema="mbo", stype_in="raw_symbol",
              start=start, end=end)
    c = db.Historical().metadata
    return c.get_record_count(**kw), c.get_billable_size(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", nargs="+", help="SYMBOL:START:END (END exclusive)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="cost/size check only")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-cost", type=float, default=0.0, metavar="USD",
                    help="Cumulative USD budget for the whole run. The default 0.0 keeps the "
                         "AGENTS.md §5 $0.00 gate; any value > 0 must have been explicitly "
                         "approved by the user for this specific pull.")
    ap.add_argument("--require-key", default=None, metavar="KEY",
                    help="Abort unless DATABENTO_API_KEY is exactly this value.")
    ap.add_argument("--min-free-gb", type=float, default=15.0, metavar="GB",
                    help="Refuse to start (and to begin each later segment) when free space "
                         "on --out-dir's filesystem is below this.")
    ap.add_argument("--submit-all", action="store_true",
                    help="Submit every segment's batch job up front so Databento prepares them "
                         "in parallel, then download each in turn and overlap its ingest with the "
                         "next download. Fastest end to end; bills the whole plan at the start.")
    ap.add_argument("--ingest-each", action="store_true",
                    help="Ingest each segment immediately after its own download, instead of "
                         "downloading every segment first. Keeps peak raw-on-disk bounded and "
                         "surfaces an ingest failure before the rest is billed.")
    args = ap.parse_args()

    _assert_key(args.require_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    free0 = _free_gb(args.out_dir)
    print(f"free space on {args.out_dir}: {free0:.1f} GB (floor {args.min_free_gb:.1f} GB)", flush=True)
    if free0 < args.min_free_gb:
        sys.exit(f"DISK GATE: {free0:.1f} GB free < --min-free-gb {args.min_free_gb:.1f} — aborting")

    segs = [tuple(s.split(":", 2)) for s in args.segments]   # END may carry a time (2026-09-25T21:00:00)

    # Gate 1: the whole plan must fit --max-cost before ANY job is submitted.
    # With the default --max-cost 0.0 this is the unconditional $0.00 gate of AGENTS.md §5.
    total = 0.0
    est_zst = est_db = 0.0
    for sym, a, b in segs:
        cost = cost_check([sym], "mbo", a, b, "raw_symbol")
        n, size = _billable(sym, a, b)
        total += cost
        # 3.13x zstd on the 56 B/record DBN wire format and 14.4 B/record in the
        # compressed hypertable, both measured over the 2026-08-23..09-30 load.
        est_zst += n * 56 / 3.13 / 1e9
        est_db += n * 14.4 / 1e9
        print(f"{sym} {a} -> {b}  cost=${cost:.4f}  records={n:,}  "
              f"uncompressed={size/1e9:.1f} GB  (running total ${total:.2f})", flush=True)
        if total > args.max_cost:
            sys.exit(f"COST GATE: cumulative ${total:.4f} exceeds --max-cost "
                     f"${args.max_cost:.2f} at {sym} {a}->{b} — aborting, nothing submitted")
    need = est_zst + est_db
    print(f"\nplan: ${total:.2f} of ${args.max_cost:.2f} budget | "
          f"raw .dbn.zst ~{est_zst:.1f} GB + hypertable ~{est_db:.1f} GB = ~{need:.1f} GB, "
          f"{free0:.1f} GB free", flush=True)
    if free0 - need < args.min_free_gb:
        sys.exit(f"DISK GATE: plan needs ~{need:.1f} GB and would leave "
                 f"{free0 - need:.1f} GB < --min-free-gb {args.min_free_gb:.1f} — aborting")
    if args.dry_run:
        print(f"dry-run OK, total ${total:.4f}")
        return

    spent = 0.0

    if args.submit_all:
        # Databento prepares batch jobs server-side, and that wait dominates a
        # multi-hundred-million-record pull. Submitting all of them first makes those
        # waits concurrent instead of serial; ingest of segment N then overlaps the
        # download of N+1. Only one ingest_mbo.py runs at a time (max_workers=1), so
        # peak uncompressed chunk footprint stays at --workers x ~4.3 GB.
        jobs = []
        for i, (sym, a, b) in enumerate(segs, 1):
            cost = cost_check([sym], "mbo", a, b, "raw_symbol")
            if spent + cost > args.max_cost:
                sys.exit(f"COST GATE (pre-submit): {sym} {a}->{b} at ${cost:.4f} would take the run "
                         f"to ${spent + cost:.4f}, over --max-cost ${args.max_cost:.2f} — aborting "
                         f"with {len(jobs)} of {len(segs)} jobs submitted (${spent:.2f})")
            job_id = submit_batch([sym], "mbo", a, b, stype_in="raw_symbol", split_duration="day")
            spent += cost
            jobs.append((job_id, sym, a, b))
            print(f"[{i}/{len(segs)}] submitted {job_id} ({sym} {a}->{b}) "
                  f"${cost:.2f}, ${spent:.2f} spent", flush=True)

        from concurrent.futures import ThreadPoolExecutor
        futs = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest") as pool:
            for i, (job_id, sym, a, b) in enumerate(jobs, 1):
                free = _free_gb(args.out_dir)
                if free < args.min_free_gb:
                    sys.exit(f"DISK GATE: {free:.1f} GB free before downloading segment "
                             f"{i}/{len(jobs)} ({sym} {a}->{b}) < --min-free-gb "
                             f"{args.min_free_gb:.1f} — aborting; jobs stay on Databento and "
                             f"raw files already on disk are ingestable by ingest_mbo.py")
                poll_batch_until_done(job_id)
                paths = download_batch(job_id, args.out_dir)
                print(f"[{i}/{len(jobs)}] downloaded {len(paths)} files, "
                      f"{sum(p.stat().st_size for p in paths)/1e9:.2f} GB, "
                      f"{_free_gb(args.out_dir):.1f} GB free", flush=True)
                if not args.skip_ingest:
                    futs.append((i, pool.submit(_ingest, args, _dbn_only(paths))))
            for i, f in futs:
                f.result()          # re-raise any ingest failure
                print(f"[{i}/{len(jobs)}] ingested", flush=True)
        print(f"\ndone: ${spent:.2f} spent of ${args.max_cost:.2f} budget, "
              f"{_free_gb(args.out_dir):.1f} GB free", flush=True)
        return

    for i, (sym, a, b) in enumerate(segs, 1):
        # Gate 2: re-check against the remaining budget right before each submit.
        cost = cost_check([sym], "mbo", a, b, "raw_symbol")
        if spent + cost > args.max_cost:
            sys.exit(f"COST GATE (pre-submit): {sym} {a}->{b} at ${cost:.4f} would take the run "
                     f"to ${spent + cost:.4f}, over --max-cost ${args.max_cost:.2f} — aborting")
        free = _free_gb(args.out_dir)
        if free < args.min_free_gb:
            sys.exit(f"DISK GATE: {free:.1f} GB free before segment {i}/{len(segs)} "
                     f"({sym} {a}->{b}) < --min-free-gb {args.min_free_gb:.1f} — aborting, "
                     f"${spent:.2f} spent so far")
        job_id = submit_batch([sym], "mbo", a, b, stype_in="raw_symbol", split_duration="day")
        spent += cost
        print(f"[{i}/{len(segs)}] submitted {job_id} ({sym} {a}->{b}) "
              f"${cost:.2f}, ${spent:.2f} spent, {free:.1f} GB free", flush=True)
        poll_batch_until_done(job_id)
        paths = download_batch(job_id, args.out_dir)
        print(f"[{i}/{len(segs)}] downloaded {len(paths)} files, "
              f"{sum(p.stat().st_size for p in paths)/1e9:.2f} GB", flush=True)
        if args.ingest_each and not args.skip_ingest:
            _ingest(args, _dbn_only(paths))
            print(f"[{i}/{len(segs)}] ingested; {_free_gb(args.out_dir):.1f} GB free", flush=True)

    if not args.ingest_each and not args.skip_ingest:
        _ingest(args, None)
    print(f"\ndone: ${spent:.2f} spent of ${args.max_cost:.2f} budget, "
          f"{_free_gb(args.out_dir):.1f} GB free", flush=True)


def _dbn_only(paths) -> list[str]:
    """Day files only. A batch job also ships condition/manifest/metadata/symbology
    .json, which ingest_mbo.py cannot parse (it would log one FAIL per job)."""
    return [str(p) for p in paths if p.name.endswith(".dbn.zst")]


def _ingest(args, files: list[str] | None) -> None:
    """Hand off to ingest_mbo.py (its own checkpoint makes re-runs idempotent)."""
    import subprocess
    cmd = [sys.executable, str(Path(__file__).with_name("ingest_mbo.py")),
           "--workers", str(args.workers)]
    cmd += ["--files", *files] if files else ["--data-dir", str(args.out_dir)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/opt/vtech/.env")
    main()
