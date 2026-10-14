#!/usr/bin/env python3
"""Build native 15s 1-bar feature parquets (ADR-075).

Coverage is the full ``databento.tbbo_15s`` span (Globex hours, lead-month
NQ), not the 5s 08:00–11:00 CT session clip. Output::

    $VTECH_DATA_DIR/sources/features/timeframe=15s/symbol=NQ/year=YYYY/month=MM/
        features_15s_part1.parquet
        features_15s_part2.parquet   # only if a month exceeds --max-part-mb

    python scripts/features/build_native_tf_15s.py
    python scripts/features/build_native_tf_15s.py --start 2025-04-07 --end 2025-04-30
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.db import get_dsn  # noqa: E402
from src.common.vm_config import get_vm_config  # noqa: E402
from src.features.native_tf import (  # noqa: E402
    DEFAULT_TIMESTEP,
    session_dates_inclusive,
    session_job,
    staging_dir,
    tbbo_15s_span,
    write_hive_months,
)

_BAR = 20


def _progress(done: int, total: int, label: str, t0: float) -> None:
    pct = min(done / total, 1.0) if total else 1.0
    filled = int(_BAR * pct)
    bar = "█" * filled + "░" * (_BAR - filled)
    elapsed = time.time() - t0
    eta = (elapsed / done * (total - done)) if done else 0.0
    sys.stderr.write(
        f"\r  [{bar}] {pct:5.0%} — {label}  {done}/{total}  "
        f"elapsed {elapsed:.0f}s  eta {eta:.0f}s"
    )
    sys.stderr.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD session date (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD session date (inclusive)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-part-mb", type=float, default=100.0)
    ap.add_argument("--timestep", default=DEFAULT_TIMESTEP)
    ap.add_argument(
        "--keep-staging",
        action="store_true",
        help="Leave per-session temp parquets under ops/tmp/",
    )
    args = ap.parse_args()

    cfg = get_vm_config()
    workers = (
        args.workers
        if args.workers is not None
        else cfg.workers_for(mem_per_worker_gb=2.0)
    )
    workers = max(1, int(workers))
    print(cfg)
    dsn = get_dsn()

    import psycopg

    with psycopg.connect(dsn) as conn:
        conn.execute("SET statement_timeout = '15s'")
        t0_span, t1_span = tbbo_15s_span(conn)
    print(f"tbbo_15s span: {t0_span} → {t1_span}")

    sessions = session_dates_inclusive(t0_span, t1_span)
    if args.start:
        sessions = [d for d in sessions if d >= date.fromisoformat(args.start)]
    if args.end:
        sessions = [d for d in sessions if d <= date.fromisoformat(args.end)]
    if not sessions:
        print("no Globex weekday sessions in range")
        return 0

    stage = staging_dir(args.timestep)
    if stage.exists() and not args.keep_staging:
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)

    jobs = [
        (d.isoformat(), dsn, args.timestep, str(stage))
        for d in sessions
    ]
    print(
        f"⏳ Native {args.timestep} 1-bar features: {len(jobs)} sessions, "
        f"workers={workers}"
    )
    t0 = time.time()
    paths: list[str] = []
    errors: list[tuple[str, str]] = []
    n_ticks = n_bars = done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(session_job, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            iso, ticks, bars, path, err = fut.result()
            done += 1
            n_ticks += ticks
            n_bars += bars
            if err:
                errors.append((iso, err))
            elif path:
                paths.append(path)
            _progress(done, len(jobs), iso, t0)
    sys.stderr.write("\n")

    if errors:
        print(f"⚠️  {len(errors)} session errors (first 8):")
        for iso, err in errors[:8]:
            print(f"    {iso}: {err}")

    print(f"concat {len(paths)} session files ({n_bars:,} bars, {n_ticks:,} ticks)")
    import pandas as pd

    frames = []
    for p in paths:
        frames.append(pd.read_parquet(p))
    if not frames:
        print("no bars produced")
        return 1 if errors else 0
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("bar_end_ts").drop_duplicates(
        subset=["bar_end_ts"], keep="last",
    )
    from src.features.bar_features import attach_rth_session_features
    from src.features.target_geometry import attach_target_geometry
    all_df = attach_rth_session_features(all_df, overwrite=True)
    all_df = attach_target_geometry(all_df)
    max_bytes = int(args.max_part_mb * 1024 * 1024)
    written = write_hive_months(
        all_df, timeframe=args.timestep, max_bytes=max_bytes,
    )
    if not args.keep_staging:
        shutil.rmtree(stage, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.0f}s — {len(all_df):,} bars → {len(written)} part files")
    for p in written:
        mb = p.stat().st_size / (1024 * 1024)
        print(f"    {p}  {mb:.1f} MB")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
