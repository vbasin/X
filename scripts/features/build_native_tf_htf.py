#!/usr/bin/env python3
"""Build native 1h / 1d HTF context parquets (ADR-091).

Source is ``databento.tbbo_1h`` / ``tbbo_1d`` (lead-month NQ), not TBBO
ticks. Groups: cs_ / tr_ / ps_ / zz_ / rv_ / sr_ (45 columns). Output::

    $VTECH_DATA_DIR/sources/features/timeframe={1h|1d}/symbol=NQ/year=YYYY/month=MM/
        features_{tf}_part1.parquet

    python scripts/features/build_native_tf_htf.py
    python scripts/features/build_native_tf_htf.py --tf 1h
    python scripts/features/build_native_tf_htf.py --tf 1d --start 2025-04-07 --end 2026-08-28
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.db import get_dsn  # noqa: E402
from src.common.vm_config import get_vm_config  # noqa: E402
from src.features.native_tf import write_hive_months  # noqa: E402
from src.features.native_tf_htf import (  # noqa: E402
    FEATURE_HTF_COLS,
    SUPPORTED_TF,
    build_htf_frame,
    clip_bar_end,
    globex_bounds,
)

_BAR = 20


def _progress(label: str, frac: float, t0: float) -> None:
    pct = min(max(frac, 0.0), 1.0)
    filled = int(_BAR * pct)
    bar = "█" * filled + "░" * (_BAR - filled)
    elapsed = time.time() - t0
    sys.stderr.write(f"\r  [{bar}] {pct:5.0%} — {label}  elapsed {elapsed:.0f}s")
    sys.stderr.flush()


def _parse_tfs(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in str(raw or "").split(",") if p.strip()]
    out: list[str] = []
    for p in parts:
        if p not in SUPPORTED_TF:
            raise SystemExit(f"unknown timeframe {p!r}; expected 1h, 1d")
        if p not in out:
            out.append(p)
    if not out:
        raise SystemExit("no timeframes")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tf", default="1h,1d", help="1h, 1d, or comma-separated")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD session date (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD session date (inclusive)")
    ap.add_argument("--max-part-mb", type=float, default=100.0)
    args = ap.parse_args()

    cfg = get_vm_config()
    print(cfg)
    tfs = _parse_tfs(args.tf)
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    t0_win, t1_win = globex_bounds(start, end)
    dsn = get_dsn()
    max_bytes = int(args.max_part_mb * 1024 * 1024)

    import psycopg

    t0 = time.time()
    rc = 0
    n_tf = len(tfs)
    with psycopg.connect(dsn) as conn:
        conn.execute("SET statement_timeout = '120s'")
        for i, tf in enumerate(tfs):
            label = f"native {tf} HTF"
            print(f"⏳ {label}")
            _progress(f"{label} load+compute", (i + 0.3) / n_tf, t0)
            frame = build_htf_frame(conn, tf, start=t0_win, end=t1_win)
            frame = clip_bar_end(frame, start, end)
            if frame.empty:
                print(f"\n⚠️  {tf}: no bars produced")
                rc = 1
                continue
            written = write_hive_months(
                frame, timeframe=tf, max_bytes=max_bytes,
                feature_cols=FEATURE_HTF_COLS,
            )
            _progress(f"{label} write", (i + 1.0) / n_tf, t0)
            sys.stderr.write("\n")
            elapsed = time.time() - t0
            print(
                f"✅ {tf}: {len(written)} hive parts  {len(frame):,} bars  "
                f"({elapsed:.0f}s so far)"
            )
            for p in written[:4]:
                print(f"    {p}  {p.stat().st_size / 1e6:.2f} MB")
            if len(written) > 4:
                print(f"    … {len(written) - 4} more")
    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.0f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
