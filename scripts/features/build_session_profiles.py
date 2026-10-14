#!/usr/bin/env python3
"""Build the daily volume-profile SSOT from the print tape.

Sessions are independent — each one's profile is derived from its own prints —
so this fans out across every core with no warmup and no cross-shard state. Both
cascade builders then read the result instead of recomputing prior-session
levels inside their own loops.

Usage::

    python scripts/features/build_session_profiles.py --start 2025-05-01 --end 2026-06-30
    python scripts/features/build_session_profiles.py --all --workers 6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data_paths import parquet_tape_dir  # noqa: E402
from src.common.vm_config import get_vm_config  # noqa: E402
from src.features.session_profile import (  # noqa: E402
    build_one, session_profile_dir, write_levels, write_profile,
)
from src.features.tape import load_tape  # noqa: E402
from src.training.auction_levels import ProfileConfig  # noqa: E402

logger = logging.getLogger("vtech.scripts.build_session_profiles")


def _session_date(p: Path) -> date:
    return date.fromisoformat(p.stem.split("_", 1)[1])


def run_chunk(days: list[str], skip_histogram: bool) -> list[dict]:
    """Profiles for a contiguous block of sessions; returns their levels rows."""
    cfg = ProfileConfig()
    rows: list[dict] = []
    for iso in days:
        d = date.fromisoformat(iso)
        tape = load_tape(d)
        if tape is None or tape.empty:
            continue
        built = build_one(tape, d, cfg)
        if built is None:
            continue
        row, hist = built
        if not skip_histogram:
            write_profile(d, hist)
        rows.append(row)
    return rows


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / max(total, 1))
    return "█" * filled + "░" * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--all", action="store_true", help="every tape session")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: get_vm_config().workers_for(mem_per_worker_gb=2.0)")
    ap.add_argument("--skip-histogram", action="store_true",
                    help="levels only; skips the per-session tick parquet")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.all and not (args.start and args.end):
        ap.error("pass --all or both --start and --end")

    from src.features.tape import list_tape_files
    files = list_tape_files()
    if not args.all:
        lo, hi = date.fromisoformat(args.start), date.fromisoformat(args.end)
        files = [f for f in files if lo <= _session_date(f) <= hi]
    if not files:
        logger.error("no tape sessions selected")
        return 1

    vm = get_vm_config()
    workers = args.workers or vm.workers_for(mem_per_worker_gb=2.0)
    days = [_session_date(f).isoformat() for f in files]
    chunks = [list(c) for c in np.array_split(np.array(days, dtype=object),
                                              min(workers, len(days)))]
    chunks = [c for c in chunks if c]
    logger.info("session profiles: %d sessions | %d workers | %d chunks -> %s",
                len(days), workers, len(chunks), session_profile_dir())

    t0 = time.time()
    rows: list[dict] = []
    if len(chunks) == 1:
        rows = run_chunk(chunks[0], args.skip_histogram)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_chunk, c, args.skip_histogram): i
                    for i, c in enumerate(chunks)}
            for done, fut in enumerate(as_completed(futs), 1):
                rows.extend(fut.result())
                logger.info("[%s] %d/%d chunks | %d sessions",
                            _bar(done, len(chunks)), done, len(chunks), len(rows))

    if not rows:
        logger.error("no sessions produced a profile")
        return 1
    path = write_levels(rows)
    logger.info("✅ %d sessions in %.1fs -> %s", len(rows), time.time() - t0, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
