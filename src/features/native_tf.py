"""Native-timeframe feature store: 1-bar compute → hive month parts.

15s product (ADR-075): every Globex hour that has lead-month TBBO, not the
5s 08:00–11:00 CT clip. Monthly directories hold ``features_{tf}_partN.parquet``
capped at :data:`src.common.data_paths.NATIVE_TF_PART_MAX_BYTES` (~100 MB).

5s research matrix uses the same hive + part writer (ADR-079) via
:mod:`src.features.hive_5s`; this module owns the generic month-part I/O.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg

from src.common.contract_calendar import active_contract
from src.common.data_paths import (
    NATIVE_TF_PART_MAX_BYTES,
    NATIVE_TF_SYMBOL_DEFAULT,
    data_root,
    iter_hive_part_files,
    native_tf_dir,
    native_tf_part_name,
)
from src.features.bar_features import (
    BAR_FEATURE_COLS,
    TICK_COLS,
    compute_bar_features,
)

CHI = ZoneInfo("America/Chicago")
DEFAULT_TIMESTEP = "15s"

_TBBO_SQL = """
SELECT ts_event, price, size, side, bid_px, ask_px, bid_sz, ask_sz
FROM databento.tbbo
WHERE ts_event >= %s AND ts_event < %s
  AND symbol = %s
ORDER BY ts_event
"""

_SPAN_SQL = "SELECT min(bucket), max(bucket) FROM databento.tbbo_15s"


def globex_session_date(ts: datetime) -> date:
    """Globex session T for a timestamp: ``[T-1 17:00, T 17:00)`` America/Chicago."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    chi = ts.astimezone(CHI)
    if chi.hour >= 17:
        return (chi + timedelta(days=1)).date()
    return chi.date()


def globex_session_window(session: date) -> tuple[datetime, datetime]:
    """UTC ``[session-1 17:00 CT, session 17:00 CT)``."""
    end = datetime(session.year, session.month, session.day, 17, 0, tzinfo=CHI)
    start = end - timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def tbbo_15s_span(conn: psycopg.Connection) -> tuple[datetime, datetime]:
    """Light min/max of ``databento.tbbo_15s.bucket`` (no count)."""
    row = conn.execute(_SPAN_SQL).fetchone()
    if not row or row[0] is None or row[1] is None:
        raise RuntimeError("databento.tbbo_15s is empty")
    t0, t1 = row[0], row[1]
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    return t0, t1


def session_dates_inclusive(t0: datetime, t1: datetime) -> list[date]:
    """Weekday Globex session dates covering ``[t0, t1]`` (Sun 17:00 ∈ Monday)."""
    d0 = globex_session_date(t0)
    d1 = globex_session_date(t1)
    out: list[date] = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def load_tbbo(
    conn: psycopg.Connection,
    start: datetime,
    end: datetime,
    symbol: str,
) -> pd.DataFrame:
    """Lead-contract TBBO prints in ``[start, end)``. Chunk-pruned on ``ts_event``."""
    with conn.cursor() as cur:
        cur.execute(_TBBO_SQL, (start, end, symbol))
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    if not rows:
        return pd.DataFrame(columns=list(TICK_COLS))
    df = pd.DataFrame(rows, columns=names)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    for c in ("price", "size", "bid_px", "ask_px", "bid_sz", "ask_sz"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_session_frame(
    conn: psycopg.Connection,
    session: date,
    *,
    timestep: str = DEFAULT_TIMESTEP,
    root: str = "NQ",
) -> pd.DataFrame:
    """1-bar features for one Globex session (lead-month NQ)."""
    symbol = active_contract(root, session)
    start, end = globex_session_window(session)
    ticks = load_tbbo(conn, start, end, symbol)
    if ticks.empty:
        return compute_bar_features(ticks, timestep)
    out = compute_bar_features(ticks, timestep)
    if out.empty:
        return out
    out["_meta_nq_contract"] = symbol
    return out


def _rows_per_part(df: pd.DataFrame, max_bytes: int) -> int:
    """Probe-write a prefix to estimate how many rows fit under *max_bytes*."""
    n = len(df)
    if n == 0:
        return 1
    probe_n = min(max(256, n // 8), n, 8000)
    buf = io.BytesIO()
    df.iloc[:probe_n].to_parquet(buf, index=False, compression="zstd")
    bpr = max(1, len(buf.getvalue()) / probe_n)
    return max(1, int(max_bytes / bpr * 0.92))


def write_month_parts(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    timeframe: str = DEFAULT_TIMESTEP,
    max_bytes: int = NATIVE_TF_PART_MAX_BYTES,
    compression: str = "zstd",
    feature_cols: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Replace ``features_{tf}_part*.parquet`` in *out_dir* with size-capped parts.

    Rows are ordered by ``bar_end_ts``. The datetime column is written first
    (never as a pandas index). Existing parts for this timeframe are unlinked
    first so a rebuild cannot mix schemas.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"features_{timeframe}_part*.parquet"):
        old.unlink()

    if df is None or df.empty:
        return []

    if "bar_end_ts" not in df.columns:
        raise ValueError("write_month_parts requires a bar_end_ts column")

    frame = df.copy(deep=False)
    frame["bar_end_ts"] = pd.to_datetime(frame["bar_end_ts"], utc=True)
    frame = frame.sort_values("bar_end_ts").reset_index(drop=True)
    others = [c for c in frame.columns if c != "bar_end_ts"]
    if feature_cols is not None:
        ordered = [c for c in feature_cols if c in others]
        tail = [c for c in others if c not in ordered]
        others = ordered + tail
    frame = frame.loc[:, ["bar_end_ts", *others]]

    rows_per = _rows_per_part(frame, max_bytes)
    written: list[Path] = []
    part = 1
    start = 0
    n = len(frame)
    while start < n:
        chunk = frame.iloc[start:start + rows_per]
        path = out_dir / native_tf_part_name(timeframe, part)
        chunk.to_parquet(path, index=False, compression=compression)
        size = path.stat().st_size
        if size > max_bytes and len(chunk) > 1:
            path.unlink()
            rows_per = max(1, len(chunk) // 2)
            continue
        written.append(path)
        start += len(chunk)
        part += 1
    return written


def month_key(ts: pd.Timestamp) -> tuple[int, int]:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.year), int(t.month)


def split_by_utc_month(df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    if df is None or df.empty:
        return {}
    ts = pd.to_datetime(df["bar_end_ts"], utc=True)
    keys = list(zip(ts.dt.year.to_numpy(), ts.dt.month.to_numpy()))
    out: dict[tuple[int, int], list[int]] = {}
    for i, k in enumerate(keys):
        out.setdefault((int(k[0]), int(k[1])), []).append(i)
    return {k: df.iloc[idx].reset_index(drop=True) for k, idx in out.items()}


def write_hive_months(
    df: pd.DataFrame,
    *,
    timeframe: str = DEFAULT_TIMESTEP,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    max_bytes: int = NATIVE_TF_PART_MAX_BYTES,
    feature_cols: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Bin *df* by UTC month of ``bar_end_ts`` and write size-split parts."""
    written: list[Path] = []
    cols = feature_cols
    if cols is None and timeframe == DEFAULT_TIMESTEP:
        from src.features.target_geometry import TARGET_GEOMETRY_COLS
        cols = (*BAR_FEATURE_COLS, *TARGET_GEOMETRY_COLS)
    for (year, month), part_df in sorted(split_by_utc_month(df).items()):
        dest = native_tf_dir(timeframe, symbol=symbol, year=year, month=month)
        dest.mkdir(parents=True, exist_ok=True)
        written.extend(write_month_parts(
            part_df, dest, timeframe=timeframe, max_bytes=max_bytes,
            feature_cols=cols,
        ))
    return written


def concat_hive_parquets(
    files: list[Path],
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Concat hive parts into a UTC DatetimeIndex frame (index name ``ts``).

    Requires column ``bar_end_ts``. Duplicate timestamps keep the last row.
    *start*/*end* are inclusive calendar dates on the UTC date of ``bar_end_ts``.
    """
    frames: list[pd.DataFrame] = []
    for p in files:
        part = pd.read_parquet(p)
        if part is None or part.empty:
            continue
        frames.append(part)
    if not frames:
        raise FileNotFoundError("hive parts are empty")
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "bar_end_ts" not in df.columns:
        raise ValueError("parquet missing bar_end_ts")
    df["bar_end_ts"] = pd.to_datetime(df["bar_end_ts"], utc=True)
    df = df.sort_values("bar_end_ts").drop_duplicates(subset=["bar_end_ts"], keep="last")
    df = df.set_index("bar_end_ts")
    df.index.name = "ts"
    if start is not None or end is not None:
        ts = df.index
        mask = pd.Series(True, index=ts)
        if start is not None:
            d0 = pd.Timestamp(start)
            if d0.tzinfo is None:
                d0 = d0.tz_localize("UTC")
            mask &= ts >= d0.normalize()
        if end is not None:
            d1 = pd.Timestamp(end)
            if d1.tzinfo is None:
                d1 = d1.tz_localize("UTC")
            mask &= ts < (d1.normalize() + pd.Timedelta(days=1))
        df = df.loc[mask.to_numpy()]
    return df


def load_hive_frame(
    timeframe: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
) -> pd.DataFrame:
    """Load hive monthly parts into a UTC DatetimeIndex frame (index name ``ts``).

    *start*/*end* are inclusive calendar dates on the UTC date of ``bar_end_ts``.
    Duplicate timestamps keep the last row in file order.
    """
    files = iter_hive_part_files(timeframe, symbol=symbol, start=start, end=end)
    if not files:
        raise FileNotFoundError(
            f"no {timeframe} hive parquet files under {native_tf_dir(timeframe, symbol=symbol)}"
        )
    return concat_hive_parquets(files, start, end)


def staging_dir(timestep: str = DEFAULT_TIMESTEP) -> Path:
    p = data_root() / "ops" / "tmp" / f"native_tf_{timeframe_tag(timestep)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def timeframe_tag(timestep: str) -> str:
    return str(timestep).strip().lower()


def session_job(payload: tuple[str, str, str, str]) -> tuple[str, int, int, str, str]:
    """Worker: ``(session_iso, dsn, timestep, staging)`` → iso, ticks, bars, path, err."""
    session_iso, dsn, timestep, staging = payload
    session = date.fromisoformat(session_iso)
    tmp = Path(staging) / f"session_{session_iso}.parquet"
    try:
        with psycopg.connect(dsn) as conn:
            conn.execute("SET statement_timeout = '180s'")
            start, end = globex_session_window(session)
            symbol = active_contract("NQ", session)
            ticks = load_tbbo(conn, start, end, symbol)
            ticks_n = int(len(ticks))
            frame = compute_bar_features(ticks, timestep)
            if not frame.empty:
                frame["_meta_nq_contract"] = symbol
            if frame.empty:
                if tmp.exists():
                    tmp.unlink()
                return session_iso, ticks_n, 0, "", ""
            tmp.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(tmp, index=False, compression="zstd")
            return session_iso, ticks_n, int(len(frame)), str(tmp), ""
    except Exception as exc:  # noqa: BLE001 — worker must not kill the pool
        return session_iso, 0, 0, "", f"{type(exc).__name__}: {exc}"
