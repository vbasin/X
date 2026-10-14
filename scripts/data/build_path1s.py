#!/usr/bin/env python3
"""Build the 1-second OHLC **fill-resolution path** cache → ``parquet_path1s/``.

Why this exists: the 5s feature parquets carry only ``_close``, and both the label
generator (:mod:`src.training.labels`) and the bar backtester
(:mod:`src.backtest.engine`) therefore test the stop barrier on *closes only*. With a
6-point stop that is an optimistic bias — a bar that traded 9 points against the entry
and closed +1 is scored as a clean winner. :mod:`src.backtest.portfolio_sim` resolves
first-touch on this 1-second high/low path instead, so the bias becomes measurable
(``tie_policy="close_only"`` reproduces the old behaviour).

This is a *path* dataset, not a feature dataset: raw OHLC passthrough, no transforms,
no look-ahead question to answer. One file per trading day::

    data/sources/features/parquet_path1s/path_YYYY-MM-DD.parquet
        ts (UTC index) | open | high | low | close | volume

Rows are the exchange's own 1-second bars — seconds with no trade are absent and are
reindexed onto a contiguous grid by the consumer (no trade ⇒ no excursion).

Usage::

    python scripts/data/build_path1s.py --start 2023-01-03 --end 2026-07-14
    python scripts/data/build_path1s.py --start 2025-06-02 --end 2026-05-18 --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data_paths import parquet_path1s_dir  # noqa: E402
from src.common.db import get_dsn  # noqa: E402

logger = logging.getLogger("vtech.scripts.build_path1s")

_CT = ZoneInfo("America/Chicago")
_UTC = ZoneInfo("UTC")

# Wider than any trade window: the simulator needs pre-window context for warmup and
# post-window bars to resolve a position opened just before the cutoff.
_SESSION_START = (7, 0)
_SESSION_END = (15, 15)
_MIN_ROWS = 1_000           # skip holidays / dead days

_SQL = """
SELECT ts_event, open, high, low, close, volume
FROM databento.ohlcv_1s
WHERE symbol = %s AND ts_event >= %s AND ts_event < %s
ORDER BY ts_event
"""


def _session_utc(d: date) -> tuple[datetime, datetime]:
    s = datetime(d.year, d.month, d.day, *_SESSION_START, tzinfo=_CT)
    e = datetime(d.year, d.month, d.day, *_SESSION_END, tzinfo=_CT)
    return s.astimezone(_UTC), e.astimezone(_UTC)


def build_day(d_iso: str, out_dir: str, symbol: str = "NQ.c.0", *, overwrite: bool = False) -> str:
    """Build one day's path parquet. Returns ``ok`` | ``skip:<reason>`` | ``error:<msg>``."""
    import psycopg

    out = Path(out_dir) / f"path_{d_iso}.parquet"
    if out.exists() and not overwrite:
        return "skip:exists"
    start_utc, end_utc = _session_utc(date.fromisoformat(d_iso))
    try:
        with psycopg.connect(get_dsn()) as conn:
            df = pd.read_sql(_SQL, conn, params=(symbol, start_utc, end_utc))
    except Exception as exc:  # noqa: BLE001 — one bad day must not kill the sweep
        return f"error:db:{exc}"

    if len(df) < _MIN_ROWS:
        return f"skip:rows={len(df)}"

    df = (df.rename(columns={"ts_event": "ts"})
            .assign(ts=lambda x: pd.to_datetime(x["ts"], utc=True))
            .drop_duplicates(subset="ts", keep="last")
            .set_index("ts")
            .sort_index()
            .astype({"open": "float64", "high": "float64", "low": "float64",
                     "close": "float64", "volume": "int64"}))

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, compression="zstd")
    tmp.replace(out)
    return "ok"


def _weekdays(start: date, end: date) -> list[str]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--symbol", default="NQ.c.0")
    ap.add_argument("--out", default=None, help="output dir (default: parquet_path1s_dir())")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out) if args.out else parquet_path1s_dir()
    days = _weekdays(date.fromisoformat(args.start), date.fromisoformat(args.end))
    logger.info("path1s build: %d weekdays %s → %s, symbol=%s, out=%s",
                len(days), args.start, args.end, args.symbol, out_dir)
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    tally: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_day, d, str(out_dir), args.symbol,
                            overwrite=args.overwrite): d for d in days}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            key = res.split(":")[0] if res.startswith(("skip", "error")) else res
            tally[key] = tally.get(key, 0) + 1
            if res.startswith("error"):
                logger.warning("%s → %s", futs[fut], res)
            if i % 100 == 0:
                logger.info("%d/%d %s", i, len(days), tally)
    logger.info("done: %s", tally)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
