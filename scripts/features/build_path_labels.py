#!/usr/bin/env python3
"""Label the 1s path cache — honest first-touch labels + dense hazard/excursion targets.

Streams ``parquet_path1s`` in chronological order through
:func:`src.training.path_labels.iter_session_labels`, so the volatility yardstick ``R``
is always drawn from **prior** sessions. Two modes:

* ``--summary-only`` (default) — print per-year label geography and write a CSV. This is
  the report that decides which eras can supervise a geometry at all.
* ``--out DIR`` — also persist one label parquet per session for training.

Examples::

    # which years can supervise a vol-normalised 50/6-equivalent geometry?
    python scripts/features/build_path_labels.py --start 2011-01-03 --end 2026-06-30

    # the same question for the fixed +50/-6 geometry (expect zeros before 2018)
    python scripts/features/build_path_labels.py --mode fixed --start 2011-01-03 --end 2026-06-30

    # materialise training labels for the usable era
    python scripts/features/build_path_labels.py --start 2018-01-02 --end 2026-06-30 \
        --out data/sources/features/parquet_pathlabels
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data_paths import parquet_path1s_dir  # noqa: E402
from src.training.path_labels import (  # noqa: E402
    PathLabelConfig, iter_session_labels, summarise,
)

logger = logging.getLogger("vtech.scripts.build_path_labels")

_BAR = 20


def _progress(done: int, total: int, started: float, note: str = "") -> None:
    frac = done / max(1, total)
    fill = int(frac * _BAR)
    el = time.time() - started
    sys.stderr.write(
        f"\r[{'█' * fill}{'░' * (_BAR - fill)}] {frac * 100:5.1f}% "
        f"— {done}/{total} sessions, {el / 60:.1f}m {note}   ")
    sys.stderr.flush()


def _sessions(path_dir: Path, start: str, end: str):
    files = sorted(f for f in path_dir.glob("path_*.parquet")
                   if start <= f.stem.removeprefix("path_") <= end)
    for f in files:
        yield f.stem.removeprefix("path_"), pd.read_parquet(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--path-dir", default=None)
    ap.add_argument("--out", default=None, help="persist per-session label parquets here")
    ap.add_argument("--summary-csv", default="/tmp/path_label_geography.csv")

    g = ap.add_argument_group("geometry")
    g.add_argument("--mode", default="vol_normalized", choices=("vol_normalized", "fixed"))
    g.add_argument("--target-points", type=float, default=50.0)
    g.add_argument("--stop-points", type=float, default=6.0)
    g.add_argument("--target-r", type=float, default=0.743)
    g.add_argument("--stop-r", type=float, default=0.0891)
    g.add_argument("--horizon-seconds", type=int, default=600)
    g.add_argument("--n-buckets", type=int, default=20)
    g.add_argument("--entry-start", default="08:30")
    g.add_argument("--entry-end", default="11:00")
    g.add_argument("--vol-lookback-sessions", type=int, default=5)
    g.add_argument("--min-stop-ticks", type=float, default=8.0)
    g.add_argument("--tie-policy", default="stop_first",
                   choices=("stop_first", "target_first"))
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = PathLabelConfig(
        mode=a.mode, target_points=a.target_points, stop_points=a.stop_points,
        target_r=a.target_r, stop_r=a.stop_r, horizon_seconds=a.horizon_seconds,
        n_buckets=a.n_buckets, entry_start=a.entry_start, entry_end=a.entry_end,
        vol_lookback_sessions=a.vol_lookback_sessions, min_stop_ticks=a.min_stop_ticks,
        tie_policy=a.tie_policy)

    path_dir = Path(a.path_dir) if a.path_dir else parquet_path1s_dir()
    total = len(sorted(f for f in path_dir.glob("path_*.parquet")
                       if a.start <= f.stem.removeprefix("path_") <= a.end))
    logger.info("labelling %d sessions %s → %s, mode=%s, entries %s–%s CT",
                total, a.start, a.end, cfg.mode, cfg.entry_start, cfg.entry_end)

    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    frames, done, started = [], 0, time.time()
    for key, frame in iter_session_labels(_sessions(path_dir, a.start, a.end), cfg):
        frames.append(frame)
        if out_dir:
            frame.to_parquet(out_dir / f"labels_{key}.parquet", index=False,
                             compression="zstd")
        done += 1
        if done % 50 == 0:
            _progress(done, total, started, f"last={key}")
    _progress(done, total, started, "done")
    sys.stderr.write("\n")

    if not frames:
        logger.error("no sessions labelled — check the date range and path cache")
        return 1

    summary = summarise(frames)
    summary.to_csv(a.summary_csv, index=False)
    pd.set_option("display.width", 200)
    print(f"\n── per-year label geography ({cfg.mode}, entries {cfg.entry_start}–"
          f"{cfg.entry_end} CT, horizon {cfg.horizon_seconds}s) ──")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {a.summary_csv}")
    if out_dir:
        logger.info("wrote %d label parquets to %s", done, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
