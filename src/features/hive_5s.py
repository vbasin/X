"""Migrate flat 5s day files into the native-TF hive (ADR-079).

Source (left in place until the operator deletes it)::

    sources/features/parquet_5s/features_YYYY-MM-DD.parquet

Destination::

    sources/features/timeframe=5s/symbol=NQ/year=YYYY/month=MM/features_5s_partN.parquet

Monthly parts split at :data:`src.common.data_paths.NATIVE_TF_PART_MAX_BYTES`
(~100 MB), same writer as 15s. Datetime column is ``bar_end_ts``. Duplicate
timestamps keep the row from the source file with the latest mtime.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from src.common.data_paths import (
    NATIVE_TF_PART_MAX_BYTES,
    NATIVE_TF_SYMBOL_DEFAULT,
    native_tf_dir,
    native_tf_part_files,
    parquet_5s_dir,
)
from src.features.native_tf import split_by_utc_month, write_month_parts

_DAY_STEM = re.compile(r"^features_(\d{4}-\d{2}-\d{2})$")
_TIMEFRAME = "5s"
_MTIME_COL = "_src_mtime_ns"
_ORD_COL = "_src_ord"


def list_5s_day_files(src_dir: Path | None = None) -> list[Path]:
    """Dated ``features_YYYY-MM-DD.parquet`` only (skips ``oof_*.parquet``)."""
    d = Path(src_dir) if src_dir is not None else parquet_5s_dir()
    if not d.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(d.glob("features_*.parquet")):
        if _DAY_STEM.match(p.stem):
            out.append(p)
    return out


def frame_with_bar_end_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose first column is UTC ``bar_end_ts`` (not an index)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["bar_end_ts"])
    out = df
    if "bar_end_ts" in out.columns:
        pass
    elif "ts" in out.columns:
        out = out.rename(columns={"ts": "bar_end_ts"})
    else:
        out = out.reset_index()
        first = str(out.columns[0])
        if first != "bar_end_ts":
            out = out.rename(columns={first: "bar_end_ts"})
    drop = [c for c in ("__index_level_0__", "index", "level_0") if c in out.columns and c != "bar_end_ts"]
    if drop:
        out = out.drop(columns=drop)
    out = out.copy(deep=False)
    out["bar_end_ts"] = pd.to_datetime(out["bar_end_ts"], utc=True)
    cols = ["bar_end_ts", *[c for c in out.columns if c != "bar_end_ts"]]
    return out.loc[:, cols]


def concat_keep_latest(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-concat and keep the latest-mtime row per ``bar_end_ts``."""
    nonempty = [f for f in frames if f is not None and not f.empty]
    if not nonempty:
        return pd.DataFrame(columns=["bar_end_ts"])
    df = pd.concat(nonempty, ignore_index=True, sort=False)
    df = frame_with_bar_end_ts(df)
    if _MTIME_COL not in df.columns:
        df[_MTIME_COL] = 0
    if _ORD_COL not in df.columns:
        df[_ORD_COL] = range(len(df))
    df = df.sort_values(["bar_end_ts", _MTIME_COL, _ORD_COL], kind="mergesort")
    df = df.drop_duplicates(subset=["bar_end_ts"], keep="last")
    return df.drop(columns=[_MTIME_COL, _ORD_COL]).reset_index(drop=True)


def _read_day_file(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    frame = frame_with_bar_end_ts(raw)
    if frame.empty:
        return frame
    frame[_MTIME_COL] = int(path.stat().st_mtime_ns)
    frame[_ORD_COL] = range(len(frame))
    return frame


def migrate_parquet_5s(
    *,
    src_dir: Path | None = None,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    max_bytes: int = NATIVE_TF_PART_MAX_BYTES,
    workers: int = 4,
    dry_run: bool = False,
    progress=None,
) -> list[Path]:
    """Convert every dated 5s day file into hive month parts. Does not delete *src_dir*."""
    files = list_5s_day_files(src_dir)
    if not files:
        return []

    workers = max(1, int(workers))
    n = len(files)
    buckets: dict[tuple[int, int], list[pd.DataFrame]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(_read_day_file, files)):
            done += 1
            if progress:
                progress(done, n, "read day files")
            if frame is None or frame.empty:
                continue
            # Later dated files win ties on mtime (pool.map preserves input order).
            frame[_ORD_COL] = i * 10_000_000 + frame[_ORD_COL].to_numpy()
            for ym, part in split_by_utc_month(frame).items():
                buckets.setdefault(ym, []).append(part)

    written: list[Path] = []
    months = sorted(buckets)
    total_m = len(months)
    for i, ym in enumerate(months, start=1):
        year, month = ym
        if progress:
            progress(i, total_m, f"write {year}-{month:02d}")
        merged = concat_keep_latest(buckets[ym])
        if dry_run or merged.empty:
            continue
        dest = native_tf_dir(_TIMEFRAME, symbol=symbol, year=year, month=month)
        dest.mkdir(parents=True, exist_ok=True)
        written.extend(write_month_parts(
            merged, dest, timeframe=_TIMEFRAME, max_bytes=max_bytes,
        ))
    return written


def load_session_day(
    target_date: date | str,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
) -> pd.DataFrame:
    """One Chicago session date from the 5s hive (UTC DatetimeIndex named ``ts``)."""
    from src.features.native_tf import load_hive_frame

    d = str(target_date)[:10]
    try:
        return load_hive_frame(_TIMEFRAME, d, d, symbol=symbol)
    except FileNotFoundError:
        return pd.DataFrame()


def persist_session_day(
    frame: pd.DataFrame,
    *,
    mtime_ns: int | None = None,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    max_bytes: int = NATIVE_TF_PART_MAX_BYTES,
) -> list[Path]:
    """Merge *frame* into 5s hive month parts under the current data root.

    Creates the month tree if needed. Duplicate ``bar_end_ts`` keep the
    latest-mtime row. Honours ``VTECH_DATA_DIR`` (engine unit tests set it).
    """
    if frame is None or frame.empty:
        return []
    day = frame_with_bar_end_ts(frame)
    if day.empty:
        return []
    day[_MTIME_COL] = int(mtime_ns if mtime_ns is not None else time.time_ns())
    day[_ORD_COL] = range(len(day))
    written: list[Path] = []
    for (year, month), part_df in sorted(split_by_utc_month(day).items()):
        existing: list[pd.DataFrame] = []
        for p in native_tf_part_files(_TIMEFRAME, year, month, symbol=symbol):
            old = pd.read_parquet(p)
            old = frame_with_bar_end_ts(old)
            if old.empty:
                continue
            old[_MTIME_COL] = int(p.stat().st_mtime_ns)
            old[_ORD_COL] = range(len(old))
            existing.append(old)
        merged = concat_keep_latest(existing + [part_df])
        dest = native_tf_dir(_TIMEFRAME, symbol=symbol, year=year, month=month)
        dest.mkdir(parents=True, exist_ok=True)
        written.extend(write_month_parts(
            merged, dest, timeframe=_TIMEFRAME, max_bytes=max_bytes,
        ))
    return written


def maybe_upsert_day(
    frame: pd.DataFrame,
    *,
    mtime_ns: int | None = None,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    max_bytes: int = NATIVE_TF_PART_MAX_BYTES,
) -> list[Path]:
    """Alias for :func:`persist_session_day` (ADR-081 hive write path)."""
    return persist_session_day(
        frame, mtime_ns=mtime_ns, symbol=symbol, max_bytes=max_bytes,
    )
