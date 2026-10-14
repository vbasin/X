#!/usr/bin/env python3
"""Aggregate existing NQ path1s files, preserving close availability and provenance.

No database access or normalization. Existing source files are never modified.
The output directory MUST NOT exist. Historical discovery defaults through 2024;
the July 2026 locked holdout cannot be opened by this command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.data_paths import parquet_path1s_dir
from src.common.vm_config import get_vm_config

PRICE_COLUMNS = ["open", "high", "low", "close"]
PROVIDER_SPEC = "https://databento.com/docs/schemas-and-data-formats/ohlcv"


def sha256_file(path: Path) -> str:
    """Hash actual bytes, not filenames or mutable contract identifiers."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_session(frame: pd.DataFrame, session: str) -> pd.DataFrame:
    """Aggregate left-labelled one-second bars into right-labelled full minutes.

    ts is availability, never the minute's opening instant. Empty minutes remain
    absent. No backfill, interpolation, future scaling, or synthetic executions.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("path1s requires a timezone-aware DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("path1s timestamps must be sorted and unique")
    if frame.empty:
        raise ValueError("empty source session")
    frame = frame.copy()
    frame.index = frame.index.tz_convert("UTC")
    if (frame.index.asi8 % 1_000_000_000).any():
        raise ValueError("timestamps are not one-second bar opens")
    local = frame.index.tz_convert("America/Chicago")
    if set(local.strftime("%Y-%m-%d")) != {session}:
        raise ValueError("source crosses its named Chicago session date")
    values = frame[PRICE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("non-finite/non-positive OHLC")
    if (frame.high < frame[PRICE_COLUMNS].max(axis=1)).any() or (
        frame.low > frame[PRICE_COLUMNS].min(axis=1)
    ).any():
        raise ValueError("inconsistent OHLC")
    if not np.isfinite(frame.volume).all() or (frame.volume < 0).any():
        raise ValueError("invalid volume")
    # Input ts marks the START of the 1s interval. A minute can only be known
    # after all constituent [second, second+1s) intervals have finished.
    grouped = frame.resample("1min", closed="left", label="right")
    bars = grouped.agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"})
    bars["observed_seconds"] = grouped.close.count()
    bars = bars.loc[bars.observed_seconds.gt(0)].copy()
    bars.index.name = "ts"
    bars["bar_open_ts"] = bars.index - pd.Timedelta(minutes=1)
    bars["session"] = session
    local_open = bars.bar_open_ts.dt.tz_convert("America/Chicago")
    bars["minute_ct"] = local_open.dt.hour * 60 + local_open.dt.minute
    return bars


def build_one(path_string: str) -> tuple[pd.DataFrame, dict]:
    """Read, validate and aggregate one source; retain byte-level inventory."""
    path = Path(path_string)
    before = path.stat()
    source_hash = sha256_file(path)
    frame = pd.read_parquet(path)
    bars = aggregate_session(frame, path.stem.removeprefix("path_"))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while reading: {path}")
    local = frame.index.tz_convert("America/Chicago")
    minute = local.hour * 60 + local.minute
    cash = bars.loc[bars.minute_ct.between(510, 899)]
    inventory = {
        "session": bars.session.iloc[0], "source_path": str(path),
        "source_sha256": source_hash, "source_bytes": before.st_size,
        "source_rows": len(frame), "bar_rows": len(bars),
        "first_source_ts": frame.index.min().isoformat(),
        "last_source_ts": frame.index.max().isoformat(),
        "first_minute_ct": int(minute.min()), "last_minute_ct": int(minute.max()),
        "cash_minutes": len(cash), "missing_cash_minutes": 390 - len(cash),
        "max_source_gap_seconds": float(frame.index.to_series().diff().dt.total_seconds().max()),
        "max_abs_second_change": float(frame.close.diff().abs().max()),
        "source_off_tick_prices": int((np.abs(frame[PRICE_COLUMNS].to_numpy() * 4
                                               - np.round(frame[PRICE_COLUMNS].to_numpy() * 4)) > 1e-7).sum()),
    }
    return bars, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start > end or end >= pd.Timestamp("2026-07-01"):
        parser.error("Require start <= end < 2026-07-01; locked holdout is forbidden")
    sources = sorted(path for path in parquet_path1s_dir().glob("path_*.parquet")
                     if args.start <= path.stem[5:] <= args.end)
    if not sources:
        parser.error("No source sessions in requested dates")
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    workers = get_vm_config().workers_for(mem_per_worker_gb=2.0)
    frames, inventory = [], []
    print(f"[{'░' * 20}] aggregate {len(sources)} sessions; workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_one, str(path)) for path in sources]
        last_progress = t0
        for i, future in enumerate(as_completed(futures), 1):
            frame, record = future.result()
            frames.append(frame)
            inventory.append(record)
            if time.monotonic() - last_progress >= 5 or i == len(sources):
                complete = int(20 * i / len(sources))
                print(f"[{'█' * complete}{'░' * (20-complete)}] aggregate {i}/{len(sources)}", flush=True)
                last_progress = time.monotonic()
    bars = pd.concat(frames).sort_index()
    inv = pd.DataFrame(inventory).sort_values("session").reset_index(drop=True)
    bars_path = args.out / "bars_1m.parquet"
    inv_path = args.out / "source_inventory.parquet"
    bars.to_parquet(bars_path, compression="zstd")
    inv.to_parquet(inv_path, compression="zstd", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_resolver": "src.common.data_paths.parquet_path1s_dir",
        "requested_start": args.start, "requested_end": args.end,
        "sessions": len(inv), "rows": len(bars), "workers": workers,
        "source_start": inv.session.min(), "source_end": inv.session.max(),
        "sessions_by_year": inv.session.str[:4].value_counts().sort_index().to_dict(),
        "source_inventory_sha256": sha256_file(inv_path),
        "bars_sha256": sha256_file(bars_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "source_builder_sha256": sha256_file(Path(__file__).parents[1] / "data/build_path1s.py"),
        "availability": "ts = minute close, bar_open_ts = minute open; no future bar joined",
        "provider_timestamp_spec": PROVIDER_SPEC,
        "normalization": "none; raw OHLCV aggregated within each session",
        "contract": "NQ.c.0 calendar front-month, unadjusted; source files lack contract IDs",
        "caveats": ["Source begins 07:00 Chicago; no overnight path or true overnight VWAP",
                    "Cash windows missing minutes can be holidays, absent trades or data gaps",
                    "No quote spread or intrabar touch ordering in this dataset",
                    "No overnight returns or cross-roll absolute levels should be inferred",
                    "Minute close is theoretical availability; actual publication latency unmeasured"],
        "quality": {"sessions_with_incomplete_cash": int(inv.missing_cash_minutes.gt(0).sum()),
                    "off_tick_price_count": int(inv.source_off_tick_prices.sum()),
                    "maximum_source_gap_seconds": float(inv.max_source_gap_seconds.max())},
    }
    (args.out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    elapsed = time.monotonic() - t0
    print(f"✅ Done in {int(elapsed)//60}m {int(elapsed)%60}s: {bars_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
