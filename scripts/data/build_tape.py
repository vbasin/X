#!/usr/bin/env python3
"""Build the print-level **TBBO tape** cache → ``parquet_tape/``.

Why this exists: ``parquet_path1s`` carries 1-second OHLCV bars, which force three
approximations the label pipeline has been paying for.

1. **Volume-at-price is smeared.** ``session_profile`` spreads a bar's volume
   uniformly across every bin its ``[low, high]`` spans, so the POC and value-area
   edges — the inputs to every bound, target and stop — are built on a reconstruction.
   Prints give exact volume at exact prices.
2. **Aggressor side is absent.** 1s bars have no side, so delta arrives via
   :func:`allocate_tf_delta`, which redistributes a *15-second* hive aggregate down
   onto 1s rows by volume share. The sign is real; the intrabar price attribution is
   not. Prints carry side directly.
3. **First touch resolves only to the second.** When target and stop both trade
   inside the same second, the outcome is decided by ``tie_policy``, a policy rather
   than an observation. The print sequence resolves the true order.

New sessions are written under the instrument root, one file per trading
day so a new session does not rewrite history::

    data/sources/price_tape/<root>/tape_YYYY-MM-DD.parquet
        ts (UTC index) | price float64 | size int32 | side int8
        bid_px float64 | ask_px float64 | bid_sz int32 | ask_sz int32

``<root>`` is the instrument whose prices are in the file (``NQ``), not the
lead month. ``scripts/data/build_tape.py`` reads Timescale ``databento.tbbo``
for that session's lead outright (``active_contract``; calendar spreads
excluded) from 07:00 to 15:15 America/Chicago. The stored series after
compaction is ``sources/price_tape/NQ/partNNN.parquet``; existing
``sources/features/parquet_tape/tape_*.parquet`` files stay until that
compaction is run. Readers check the new directory first.

``side`` is the aggressor: ``+1`` buyer lifted the offer, ``-1`` seller hit the bid,
``0`` unknown (Databento ``'N'``, rare). Signed size is therefore ``size * side``.
The BBO columns are the quote **at the instant of the print**, which is what the
Cont (2014) order-flow-imbalance estimator consumes; without them ``of_ofi`` would
collapse onto ``of_delta`` and the model's strongest feature would be lost.

**Symbol.** ``databento.tbbo`` stores outrights (``NQU6``), never the continuous
``NQ.c.0`` that ``parquet_path1s`` is built from, so the lead contract is resolved per
session through :func:`active_contract`. Calendar spreads (``NQM6-NQU6``) are excluded
by the exact-symbol match — a ``LIKE 'NQ%'`` predicate would admit them.

Usage::

    python scripts/data/build_tape.py --start 2025-04-07 --end 2026-06-30
    python scripts/data/build_tape.py --start 2025-05-01 --end 2026-06-30 --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.contract_calendar import active_contract  # noqa: E402
from src.common.data_paths import price_tape_dir  # noqa: E402
from src.common.db import get_dsn  # noqa: E402
from src.common.vm_config import get_vm_config  # noqa: E402

logger = logging.getLogger("vtech.scripts.build_tape")

_CT = ZoneInfo("America/Chicago")
_UTC = ZoneInfo("UTC")

# Matches parquet_path1s so the two caches are interchangeable per session: warmup
# context before the cash open, runway after the 15:00 flatten.
_SESSION_START = (7, 0)
_SESSION_END = (15, 15)
_MIN_ROWS = 1_000           # skip holidays / dead days

_SQL = """
SELECT ts_event, price, size, side, bid_px, ask_px, bid_sz, ask_sz
FROM databento.tbbo
WHERE symbol = %s AND ts_event >= %s AND ts_event < %s
ORDER BY ts_event, sequence
"""

_SIDE_MAP = {"B": 1, "A": -1, "N": 0}


def _session_utc(d: date) -> tuple[datetime, datetime]:
    s = datetime(d.year, d.month, d.day, *_SESSION_START, tzinfo=_CT)
    e = datetime(d.year, d.month, d.day, *_SESSION_END, tzinfo=_CT)
    return s.astimezone(_UTC), e.astimezone(_UTC)


def build_day(d_iso: str, out_dir: str, root: str = "NQ", *,
              overwrite: bool = False) -> str:
    """Build one session's tape parquet. ``ok`` | ``skip:<reason>`` | ``error:<msg>``."""
    import psycopg

    out = Path(out_dir) / f"tape_{d_iso}.parquet"
    if out.exists() and not overwrite:
        return "skip:exists"
    d = date.fromisoformat(d_iso)
    symbol = active_contract(root, d)
    start_utc, end_utc = _session_utc(d)
    try:
        with psycopg.connect(get_dsn()) as conn, conn.cursor() as cur:
            cur.execute(_SQL, (symbol, start_utc, end_utc))
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 — one bad day must not kill the sweep
        return f"error:db:{exc}"

    if len(rows) < _MIN_ROWS:
        return f"skip:rows={len(rows)}"

    df = pd.DataFrame(rows, columns=["ts", "price", "size", "side",
                                     "bid_px", "ask_px", "bid_sz", "ask_sz"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ("price", "bid_px", "ask_px"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("size", "bid_sz", "ask_sz"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    df["side"] = (df["side"].map(_SIDE_MAP).fillna(0).astype("int8"))
    df = df.dropna(subset=["price"])
    # Order is the point of this cache: ts_event ties are broken by exchange sequence,
    # which is what makes first-touch resolvable. Never sort on ts alone here.
    df = df.set_index("ts")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, compression="zstd")
    tmp.replace(out)
    return f"ok:{len(df)}:{symbol}"


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
    ap.add_argument("--root", default="NQ", help="contract root")
    ap.add_argument("--out", default=None,
                    help="output dir (default: sources/price_tape/<root>/)")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: get_vm_config().workers_for()")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out) if args.out else price_tape_dir(args.root)
    days = _weekdays(date.fromisoformat(args.start), date.fromisoformat(args.end))
    workers = args.workers or get_vm_config().workers_for()
    logger.info("tape build: %d weekdays %s → %s, root=%s, workers=%d, out=%s",
                len(days), args.start, args.end, args.root, workers, out_dir)
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    n_rows = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(build_day, d, str(out_dir), args.root,
                            overwrite=args.overwrite): d for d in days}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            kind = res.split(":", 1)[0]
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "ok":
                n_rows += int(res.split(":")[1])
            elif kind == "error":
                logger.warning("%s → %s", futs[fut], res)
            if i % 25 == 0 or i == len(days):
                pct = int(20 * i / len(days))
                logger.info("[%s%s] %d/%d | %s | %s rows",
                            "#" * pct, "." * (20 - pct), i, len(days),
                            dict(sorted(counts.items())), f"{n_rows:,}")

    logger.info("done: %s, %s prints", dict(sorted(counts.items())), f"{n_rows:,}")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
