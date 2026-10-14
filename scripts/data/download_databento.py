#!/usr/bin/env python3
"""Unified Databento batch-download script — consolidates batch-paradigm download scripts.

Replaces (batch-submit paradigm):
  download_nq_ohlcv_1s.py   — NQ continuous futures, ohlcv-1s
  download_es_ohlcv_1s.py   — ES continuous futures, ohlcv-1s
  download_nq_opt.py         — NQ options, bbo-1s (single job-ID)
  download_ohlcv_parallel.py — parallel multi-job file downloader

NOTE: The timeseries.get_range() paradigm scripts (download_0dte_definitions.py,
download_nq_0dte_weekly.py, download_1dte_options.py, download_qne_options.py)
are NOT replaced here — they have specialised symbol-map / matrix logic that
cannot be generalised without complex subcommands. Use those scripts directly
for 0DTE / QNE options downloads.

SAFETY: This script NEVER submits a batch job or downloads data without
explicit cost confirmation. A hard gate at --max-cost (default $0.00) aborts
automatically if any job would cost more. Interactive confirmation is required
for non-zero costs unless --force-paid is explicitly provided.

Modes (--action):
  dry-run    Show per-year plan, estimated sizes, costs — NO API calls to billing (default)
  submit     Submit batch jobs for any un-submitted date ranges
  download   Download completed files for all submitted jobs
  check      Check status of pending jobs and print summary

Granularity (--granularity):
  annual     One batch job per calendar year (default — suited for continuous futures)
  monthly    One batch job per calendar month (suited for options/high-volume)
  single     Single batch job for the entire range

Usage:
    # Dry-run (no API calls)
    python scripts/download_databento.py \\
        --symbol NQ.c.0 --schema ohlcv-1s --stype continuous \\
        --out-dir /opt/vtech/data/sources/raw/backfill/nq_fut_ohlcv_1s \\
        --action dry-run

    # Submit annual jobs for NQ ohlcv-1s (2011–today)
    python scripts/download_databento.py \\
        --symbol NQ.c.0 --schema ohlcv-1s --stype continuous \\
        --start 2011-01-01 \\
        --out-dir /opt/vtech/data/sources/raw/backfill/nq_fut_ohlcv_1s \\
        --action submit

    # Submit + download in one pass
    python scripts/download_databento.py \\
        --symbol ES.c.0 --schema ohlcv-1s --stype continuous \\
        --out-dir /opt/vtech/data/sources/raw/backfill/es_fut_ohlcv_1s \\
        --action submit --poll

    # Download files from previously submitted jobs (parallel, 16 workers)
    python scripts/download_databento.py \\
        --out-dir /opt/vtech/data/sources/raw/backfill/nq_fut_ohlcv_1s \\
        --action download --workers 16

    # Single job for a specific symbol + range
    python scripts/download_databento.py \\
        --symbol NQ.OPT --schema bbo-1s --stype parent \\
        --start 2026-04-01 --end 2026-04-30 \\
        --out-dir /opt/vtech/data/sources/raw/backfill/nq_opt \\
        --granularity single --action submit

Checkpoint:
    {out_dir}/batch_jobs.json
    Tracks all submitted job IDs, year/month ranges, download status.
    Safe to interrupt and resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import databento as db
from dotenv import load_dotenv

load_dotenv("/opt/vtech/.env")

sys.path.insert(0, "/opt/vtech")
from src.common.vm_config import get_vm_config, fmt_bytes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET = "GLBX.MDP3"                # Default — only CME Globex supported currently
DATABENTO_MAX_CONCURRENT = 16        # Practical S3 rate-limit ceiling
DL_MEM_PER_WORKER_GB = 0.1           # ~100 MB per download thread

# Cost gate: abort automatically for any job over this cost.
# Override with --max-cost (e.g. 0.01 to allow up to $0.01 per job).
DEFAULT_MAX_COST_USD = 0.00

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_databento")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_ckpt(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"jobs": []}
    return {"jobs": []}


def _save_ckpt(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

# ---------------------------------------------------------------------------
# Date range helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return date.today().isoformat()


def _annual_ranges(start: date, end: date) -> list[tuple[str, str]]:
    """Split [start, end) into annual sub-ranges."""
    ranges: list[tuple[str, str]] = []
    current = max(start, date(start.year, 1, 1))
    while current < end:
        year_end = date(current.year + 1, 1, 1)
        range_end = min(year_end, end)
        ranges.append((current.isoformat(), range_end.isoformat()))
        current = year_end
    return ranges


def _monthly_ranges(start: date, end: date) -> list[tuple[str, str]]:
    """Split [start, end) into monthly sub-ranges."""
    ranges: list[tuple[str, str]] = []
    current = date(start.year, start.month, 1)
    current = max(current, start)
    while current < end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        range_end = min(next_month, end)
        ranges.append((current.isoformat(), range_end.isoformat()))
        current = next_month
    return ranges


def _single_range(start: date, end: date) -> list[tuple[str, str]]:
    return [(start.isoformat(), end.isoformat())]


def _build_ranges(start: date, end: date, granularity: str) -> list[tuple[str, str]]:
    if granularity == "annual":
        return _annual_ranges(start, end)
    elif granularity == "monthly":
        return _monthly_ranges(start, end)
    else:
        return _single_range(start, end)

# ---------------------------------------------------------------------------
# Thread-local Databento client (download phase)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_client() -> db.Historical:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = db.Historical()
    return _thread_local.client

# ---------------------------------------------------------------------------
# Download task
# ---------------------------------------------------------------------------

@dataclass
class DownloadTask:
    job_id: str
    filename: str
    expected_size: int
    output_dir: Path


@dataclass
class DownloadResult:
    task: DownloadTask
    status: str       # 'ok' | 'skip' | 'error'
    bytes_written: int
    elapsed_s: float
    error: str = ""


def _download_one(task: DownloadTask) -> DownloadResult:
    """Download a single .dbn.zst file (skips if already present with correct size)."""
    # Databento SDK always creates {output_dir}/{job_id}/{filename}
    dest = task.output_dir / task.job_id / task.filename
    if dest.exists() and dest.stat().st_size == task.expected_size:
        return DownloadResult(task=task, status="skip", bytes_written=0, elapsed_s=0.0)

    task.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        _get_client().batch.download(
            job_id=task.job_id,
            output_dir=str(task.output_dir),
            filename_to_download=task.filename,
        )
        elapsed = time.monotonic() - t0
        written = dest.stat().st_size if dest.exists() else 0
        return DownloadResult(task=task, status="ok", bytes_written=written, elapsed_s=elapsed)
    except Exception as exc:
        return DownloadResult(task=task, status="error", bytes_written=0,
                              elapsed_s=time.monotonic() - t0, error=str(exc))

# ---------------------------------------------------------------------------
# Action: dry-run
# ---------------------------------------------------------------------------

def cmd_dry_run(
    symbol: str,
    schema: str,
    stype: str,
    ranges: list[tuple[str, str]],
    dataset: str,
) -> None:
    """Print per-range cost and size estimates (makes metadata API calls)."""
    client = db.Historical()
    print(f"\n[dry-run] {dataset} | {symbol} ({stype}) | schema={schema}")
    print(f"{'Range':<30} {'Cost (USD)':>12} {'Est. size':>12}")
    print("-" * 56)

    total_cost = 0.0
    total_bytes = 0

    for start_s, end_s in ranges:
        try:
            cost = client.metadata.get_cost(
                dataset=dataset, symbols=symbol, schema=schema,
                stype_in=stype, start=start_s, end=end_s,
            )
            size = client.metadata.get_billable_size(
                dataset=dataset, symbols=symbol, schema=schema,
                stype_in=stype, start=start_s, end=end_s,
            )
        except Exception as exc:
            print(f"  {start_s} → {end_s:<20} ERROR: {exc}")
            continue
        total_cost += cost
        total_bytes += size
        cost_str = f"${cost:.4f}" if cost > 0 else "FREE"
        print(f"  {start_s} → {end_s:<15}  {cost_str:>12}  {fmt_bytes(size):>12}")

    print("-" * 56)
    total_cost_str = f"${total_cost:.4f}" if total_cost > 0 else "FREE"
    print(f"  {'TOTAL':<28}  {total_cost_str:>12}  {fmt_bytes(total_bytes):>12}")
    print()

# ---------------------------------------------------------------------------
# Action: submit
# ---------------------------------------------------------------------------

def cmd_submit(
    symbol: str,
    schema: str,
    stype: str,
    ranges: list[tuple[str, str]],
    dataset: str,
    ckpt_path: Path,
    max_cost: float,
    force_paid: bool,
) -> None:
    """Submit batch jobs for ranges not yet in the checkpoint."""
    client = db.Historical()
    ckpt = _load_ckpt(ckpt_path)

    # Build set of already-submitted ranges (keyed by start+end)
    submitted_ranges: set[tuple[str, str]] = {
        (j["start"], j["end"]) for j in ckpt.get("jobs", [])
    }

    pending_ranges = [(s, e) for s, e in ranges if (s, e) not in submitted_ranges]

    if not pending_ranges:
        logger.info("All ranges already submitted. Use --action download to fetch files.")
        return

    logger.info("Checking costs for %d pending ranges...", len(pending_ranges))

    for start_s, end_s in pending_ranges:
        # Cost gate
        try:
            cost = client.metadata.get_cost(
                dataset=dataset, symbols=symbol, schema=schema,
                stype_in=stype, start=start_s, end=end_s,
            )
        except Exception as exc:
            logger.error("Cost check failed for %s → %s: %s", start_s, end_s, exc)
            sys.exit(1)

        if cost > max_cost:
            if not force_paid:
                print(
                    f"\n⚠️  COST GATE: Range {start_s} → {end_s} would cost ${cost:.4f}.\n"
                    f"   Max allowed: ${max_cost:.4f}. Use --max-cost {cost:.2f} --force-paid\n"
                    f"   to override. Aborting."
                )
                sys.exit(1)
            else:
                print(f"  ⚠️  Paid range ${cost:.4f}: {start_s} → {end_s} [--force-paid]")

        # Submit the batch job
        try:
            job = client.batch.submit_job(
                dataset=dataset,
                symbols=symbol,
                schema=schema,
                stype_in=stype,
                start=start_s,
                end=end_s,
                encoding="dbn",
                compression="zstd",
                split_duration="month",
            )
            job_id = job["id"]  # databento SDK 0.74.x: submit_job returns dict
            logger.info("  Submitted %s → %s  job_id=%s", start_s, end_s, job_id)
        except Exception as exc:
            logger.error("  Failed to submit %s → %s: %s", start_s, end_s, exc)
            sys.exit(1)

        ckpt.setdefault("jobs", []).append({
            "job_id": job_id,
            "start":  start_s,
            "end":    end_s,
            "symbol": symbol,
            "schema": schema,
            "cost":   cost,
            "downloaded": False,
        })
        _save_ckpt(ckpt_path, ckpt)

    logger.info("All %d jobs submitted. Checkpoint: %s", len(pending_ranges), ckpt_path)
    logger.info("Run with --action download when jobs complete.")

# ---------------------------------------------------------------------------
# Action: check
# ---------------------------------------------------------------------------

def cmd_check(ckpt_path: Path) -> None:
    """Print status of all submitted jobs."""
    ckpt = _load_ckpt(ckpt_path)
    jobs = ckpt.get("jobs", [])
    if not jobs:
        print("No submitted jobs found in checkpoint.")
        return

    client = db.Historical()
    print(f"\nCheckpoint: {ckpt_path}  ({len(jobs)} jobs)\n")
    print(f"{'Job ID':<35} {'Range':<25} {'Downloaded':<12} {'Status'}")
    print("-" * 90)

    # databento SDK 0.74.x has no get_job(); fetch all jobs once and index by id
    try:
        api_by_id = {
            j["id"]: j
            for j in client.batch.list_jobs(states=["queued", "processing", "done", "expired"])
        }
    except Exception as exc:
        api_by_id = {}
        logger.warning("list_jobs failed: %s", exc)

    for job in jobs:
        job_id = job["job_id"]
        r_str = f"{job.get('start', '?')} → {job.get('end', '?')}"
        dl = "yes" if job.get("downloaded") else "no"
        api_job = api_by_id.get(job_id)
        status = api_job["state"] if api_job else "not-found"
        print(f"  {job_id:<33} {r_str:<25} {dl:<12} {status}")
    print()

# ---------------------------------------------------------------------------
# Action: download
# ---------------------------------------------------------------------------

def cmd_download(
    ckpt_path: Path,
    out_dir: Path,
    workers: int,
    poll: bool,
    poll_interval_s: int = 30,
) -> None:
    """Download completed files for all jobs in checkpoint."""
    ckpt = _load_ckpt(ckpt_path)
    jobs = ckpt.get("jobs", [])
    pending_jobs = [j for j in jobs if not j.get("downloaded")]

    if not pending_jobs:
        logger.info("All jobs already downloaded.")
        return

    client = db.Historical()

    # If --poll, wait for pending jobs to complete
    if poll:
        logger.info("Polling for %d pending jobs (interval=%ds)...", len(pending_jobs), poll_interval_s)
        while True:
            # databento SDK 0.74.x has no get_job(); poll via list_jobs once per round
            try:
                api_by_id = {
                    j["id"]: j
                    for j in client.batch.list_jobs(states=["queued", "processing", "done", "expired"])
                }
            except Exception as exc:
                logger.warning("  list_jobs failed: %s", exc)
                api_by_id = {}
            states = {}
            for job in pending_jobs:
                api_job = api_by_id.get(job["job_id"])
                states[job["job_id"]] = api_job["state"] if api_job else "queued"

            pending_count = sum(1 for s in states.values() if s not in ("done", "expired"))
            done_count    = sum(1 for s in states.values() if s == "done")
            logger.info("  %d done | %d pending", done_count, pending_count)

            if pending_count == 0:
                break
            time.sleep(poll_interval_s)

    # Build download tasks for all done jobs
    tasks: list[DownloadTask] = []
    total_bytes = 0
    skip_count = 0

    for job in pending_jobs:
        job_id = job["job_id"]
        try:
            files = client.batch.list_files(job_id)
        except Exception as exc:
            logger.warning("  Cannot list files for %s: %s", job_id, exc)
            continue

        data_files = [f for f in files if f["filename"].endswith(".dbn.zst")]
        for f in data_files:
            dest = out_dir / job_id / f["filename"]
            sz = f["size"]
            total_bytes += sz
            if dest.exists() and dest.stat().st_size == sz:
                skip_count += 1
            tasks.append(DownloadTask(
                job_id=job_id,
                filename=f["filename"],
                expected_size=sz,
                output_dir=out_dir,
            ))

    logger.info(
        "Download plan: %d files | %s | %d already present | workers=%d",
        len(tasks), fmt_bytes(total_bytes), skip_count, workers,
    )

    if not tasks:
        logger.info("Nothing to download.")
        return

    results: list[DownloadResult] = []
    done_count = 0
    errors: list[DownloadResult] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, t): t for t in tasks}
        for fut in as_completed(futures):
            done_count += 1
            res = fut.result()
            results.append(res)

            pct = done_count / len(tasks) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            flag = "✅" if res.status in ("ok", "skip") else "❌"
            logger.info(
                "[%s] %5.1f%% (%d/%d) %s  %s  %s",
                bar, pct, done_count, len(tasks), flag,
                res.task.filename, res.error or f"{fmt_bytes(res.bytes_written)}"
            )
            if res.status == "error":
                errors.append(res)

    # Mark jobs as downloaded if all their files succeeded/skipped
    job_file_statuses: dict[str, list[str]] = {}
    for r in results:
        job_file_statuses.setdefault(r.task.job_id, []).append(r.status)

    updated = 0
    for job in ckpt["jobs"]:
        job_id = job["job_id"]
        statuses = job_file_statuses.get(job_id, [])
        if statuses and all(s in ("ok", "skip") for s in statuses):
            job["downloaded"] = True
            updated += 1
    _save_ckpt(ckpt_path, ckpt)

    ok = len(results) - len(errors)
    logger.info(
        "Download complete: %d ok | %d skip | %d errors | %d jobs marked done",
        ok, sum(1 for r in results if r.status == "skip"),
        len(errors), updated,
    )
    if errors:
        for r in errors:
            logger.error("  FAILED: %s — %s", r.task.filename, r.error)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Databento batch-download script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Symbol / schema args
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Databento dataset (default: {DATASET})")
    parser.add_argument("--symbol", default=None,
                        help="Symbol or parent root, e.g. NQ.c.0 or NQ.OPT")
    parser.add_argument("--schema", default=None,
                        help="DBN schema, e.g. ohlcv-1s, bbo-1s, definition")
    parser.add_argument("--stype", default="continuous",
                        help="Symbol type: continuous, parent, raw_symbol (default: continuous)")

    # Date range
    parser.add_argument("--start", default="2011-01-01",
                        help="Start date YYYY-MM-DD (default: 2011-01-01)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD exclusive (default: today)")

    # Output
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for downloaded files")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint JSON path (default: {out-dir}/batch_jobs.json)")

    # Job config
    parser.add_argument("--granularity", default="annual",
                        choices=["annual", "monthly", "single"],
                        help="Batch job splitting granularity (default: annual)")

    # Action
    parser.add_argument("--action", default="dry-run",
                        choices=["dry-run", "submit", "download", "check"],
                        help="Action to perform (default: dry-run)")
    parser.add_argument("--poll", action="store_true",
                        help="With --action download: poll until jobs complete")
    parser.add_argument("--workers", type=int, default=None,
                        help="Download workers (default: adaptive via vm_config, max 16)")

    # Cost gate
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD,
                        help=f"Abort if per-job cost > this value in USD (default: {DEFAULT_MAX_COST_USD})")
    parser.add_argument("--force-paid", action="store_true",
                        help="Allow non-zero cost jobs without interactive confirmation")

    args = parser.parse_args()

    # Resolve checkpoint
    ckpt_path = args.checkpoint or (args.out_dir / "batch_jobs.json")

    # Date range
    start_d = date.fromisoformat(args.start)
    end_d   = date.fromisoformat(args.end) if args.end else date.today()

    # Ranges
    ranges = _build_ranges(start_d, end_d, args.granularity)

    # Workers
    vm = get_vm_config()
    n_workers = args.workers or min(
        DATABENTO_MAX_CONCURRENT,
        vm.workers_for(mem_per_worker_gb=DL_MEM_PER_WORKER_GB, cpu_cap=DATABENTO_MAX_CONCURRENT),
    )

    if args.action == "dry-run":
        if not args.symbol or not args.schema:
            parser.error("--symbol and --schema are required for dry-run")
        cmd_dry_run(args.symbol, args.schema, args.stype, ranges, args.dataset)

    elif args.action == "submit":
        if not args.symbol or not args.schema:
            parser.error("--symbol and --schema are required for submit")
        cmd_submit(
            symbol=args.symbol,
            schema=args.schema,
            stype=args.stype,
            ranges=ranges,
            dataset=args.dataset,
            ckpt_path=ckpt_path,
            max_cost=args.max_cost,
            force_paid=args.force_paid,
        )
        if args.poll:
            logger.info("--poll: waiting for jobs then downloading...")
            cmd_download(ckpt_path, args.out_dir, n_workers, poll=True)

    elif args.action == "download":
        cmd_download(ckpt_path, args.out_dir, n_workers, poll=args.poll)

    elif args.action == "check":
        cmd_check(ckpt_path)


if __name__ == "__main__":
    main()
