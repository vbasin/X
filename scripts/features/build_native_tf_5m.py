#!/usr/bin/env python3
"""Build native 5-minute feature parquets (ADR-090).

Coverage is the full ``databento.tbbo_15s`` span (Globex hours, lead-month
NQ). Groups: cs_ (OHLCV + ATR(5)) / vp_ / vw_ / ps_ / zz_ / vf_ / rv_.
Output::

    $VTECH_DATA_DIR/sources/features/timeframe=5m/symbol=NQ/year=YYYY/month=MM/
        features_5m_part1.parquet

    python scripts/features/build_native_tf_5m.py
    python scripts/features/build_native_tf_5m.py --start 2025-04-07 --end 2025-04-30
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
    session_dates_inclusive,
    staging_dir,
    tbbo_15s_span,
    write_hive_months,
)
from src.features.native_tf_5m import FEATURE_5M_COLS, TIMESTEP, session_job  # noqa: E402

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
        else cfg.workers_for(mem_per_worker_gb=3.0)
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

    stage = staging_dir(TIMESTEP)
    if stage.exists() and not args.keep_staging:
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=True)

    jobs = [
        (d.isoformat(), dsn, str(stage))
        for d in sessions
    ]
    print(
        f"⏳ Native {TIMESTEP} features: {len(jobs)} sessions, "
        f"workers={workers}"
    )
    t0 = time.time()
    paths: list[str] = []
    errors: list[tuple[str, str]] = []
    n_bars = done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(session_job, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            iso, _ticks, bars, path, err = fut.result()
            done += 1
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

    print(f"concat {len(paths)} session files ({n_bars:,} bars)")
    import pandas as pd

    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        print("no bars produced")
        return 1 if errors else 0
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("bar_end_ts").drop_duplicates(
        subset=["bar_end_ts"], keep="last",
    )
    max_bytes = int(args.max_part_mb * 1024 * 1024)
    written = write_hive_months(
        all_df, timeframe=TIMESTEP, max_bytes=max_bytes,
        feature_cols=FEATURE_5M_COLS,
    )
    elapsed = time.time() - t0
    print(f"✅ {len(written)} hive parts  {len(all_df):,} bars  in {elapsed:.0f}s")
    for p in written[:6]:
        print(f"    {p}  {p.stat().st_size / 1e6:.1f} MB")
    if len(written) > 6:
        print(f"    … {len(written) - 6} more")
    if not args.keep_staging:
        shutil.rmtree(stage, ignore_errors=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
