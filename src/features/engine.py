"""Feature engine v3: orchestrates all feature modules.

Loads source data from TimescaleDB, runs each feature module, concatenates
the result into a single feature matrix, and caches to Parquet.

v3 changes:
  - 5s timestep (configurable via VTECH_FEATURES_TIMESTEP)
  - 8:00–11:00 CST time filter applied to output
  - Removed macro_sentiment (VIXY) and equity_context (NVDA/TSLA/XLK/SMH)
  - Added 10 new feature modules
  - Uses bar_windows() for all window conversions

Run as:
    python -m src.features.engine --date=2025-07-15          # patch (default)
    python -m src.features.engine --date=today --rebuild     # replace file
    python -m src.features.engine --start=2025-04-01 --end=2025-07-01
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from src.common.data_paths import native_tf_dir, native_tf_features_root
from src.common.bar_windows import get_timestep
from src.common.cme_holidays import is_trading_day
from src.common.contract_calendar import active_contract, active_contracts_in_range
from src.common.db import get_dsn
from src.common.market_session import rth_open_utc
from src.features import (
    book_pressure,
    candle_structure,
    cross_asset,
    daily_context,
    dealer_gex,
    economic_calendar,
    atr as atr_mod,
    hawkes_clustering,
    higher_timeframe,
    large_trade_cvd,
    mbp_depth,
    psar,
    stochastic,
    microstructure,
    options_surface,
    order_flow,
    realized_vol,
    same_side,
    sub_bar_dynamics,
    time_context,
    trade_arrival,
    trade_location,
    volume_profile,
    volume_footprint,
    vwap,
    wavelets,
)

logger = logging.getLogger("vtech.features.engine")

# Rows pulled per server-side fetch. Large enough to amortize round-trips,
# small enough that one batch of Python tuples stays a few tens of MB.
_QUERY_BATCH = 100_000


def _progress_bar(current: int, total: int, label: str = "",
                  t0: float | None = None, width: int = 20) -> str:
    """Format a live terminal progress bar with ETA."""
    pct = min(current / total, 1.0) if total > 0 else 0
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    parts = [f"\r  [{bar}] {pct:5.0%}"]
    if label:
        parts.append(f" — {label}")
    if t0 is not None:
        elapsed = time.time() - t0
        if pct > 0 and pct < 1.0:
            eta = elapsed / pct * (1 - pct)
            eta_m, eta_s = divmod(int(eta), 60)
            parts.append(f"  ETA {eta_m}m{eta_s:02d}s")
    return "".join(parts)


def _emit(msg: str, newline: bool = False) -> None:
    """Write to stderr for live progress (stdout may be piped)."""
    sys.stderr.write(msg + ("\n" if newline else ""))
    sys.stderr.flush()


def timestep_seconds(timestep: str | None = None) -> int:
    """Parse a pandas-style timestep string (e.g. ``5s``) into seconds."""
    raw = (timestep or get_timestep()).strip().lower()
    if raw.endswith("ms"):
        return max(1, int(raw[:-2]) // 1000)
    if raw.endswith("s"):
        return max(1, int(raw[:-1]))
    if raw.endswith("min"):
        return max(1, int(raw[:-3]) * 60)
    if raw.endswith("m"):
        return max(1, int(raw[:-1]) * 60)
    return max(1, int(raw))


def last_closed_bar_ts(
    asof: pd.Timestamp | datetime | None = None,
    *,
    timestep: str | None = None,
) -> pd.Timestamp:
    """Return the label of the latest *completed* bar at or before *asof*.

    With left-labeled ``resample(timestep)`` bars, the bar labeled ``T`` covers
    ``[T, T+step)``.  It is complete only when wall-clock is at or past ``T+step``.
    """
    step = timestep_seconds(timestep)
    if asof is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        now = pd.Timestamp(asof)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
    epoch = int(now.timestamp())
    # Start of the bar that contains *now* (left edge).
    bar_start = epoch - (epoch % step)
    # That bar is incomplete until *now* reaches bar_start+step; last complete
    # is always the previous left edge.
    last = bar_start - step
    return pd.Timestamp(last, unit="s", tz="UTC")


def merge_feature_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append *new_rows* after *existing*, keeping prior values (no recompute)."""
    if new_rows is None or new_rows.empty:
        return existing
    if existing is None or existing.empty:
        out = new_rows.copy()
    else:
        # Align columns: union, prefer new schema for missing cols
        cols = list(dict.fromkeys(list(existing.columns) + list(new_rows.columns)))
        ex = existing.reindex(columns=cols)
        nw = new_rows.reindex(columns=cols)
        # Drop any overlapping index from new (should not recompute history)
        nw = nw.loc[~nw.index.isin(ex.index)]
        if nw.empty:
            return ex
        out = pd.concat([ex, nw], axis=0)
    out = out.loc[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _series_kind(s: pd.Series) -> str:
    """Compact storage kind for one query column. ``unknown`` means all-null."""
    if s.isna().all():
        return "unknown"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "dt"
    sample = s.dropna().iloc[0]
    if isinstance(sample, str):
        return "str"
    if isinstance(sample, datetime):
        return "dt"
    return "obj"


def _encode_strings(s: pd.Series, mapping: dict[str, int], cats: list[str]) -> np.ndarray:
    """Map a string batch onto one shared category dictionary. NA → -1."""
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    remap = np.full(len(uniques), -1, dtype=np.int32)
    for i, raw in enumerate(uniques):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        key = raw if isinstance(raw, str) else str(raw)
        idx = mapping.get(key)
        if idx is None:
            idx = len(cats)
            mapping[key] = idx
            cats.append(key)
        remap[i] = idx
    out = np.full(len(codes), -1, dtype=np.int32)
    ok = codes >= 0
    out[ok] = remap[codes[ok]]
    return out


def _datetime_ns(s: pd.Series) -> np.ndarray:
    """UTC nanoseconds. ``Series.to_numpy()`` on a tz-aware column returns objects."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, utc=True)
    elif getattr(s.dtype, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s.to_numpy(dtype="datetime64[ns]", copy=True)


class _ColAccum:
    """Append query batches into columnar pieces without keeping the tuples."""

    def __init__(self) -> None:
        self.kind: str | None = None
        self.parts: list[np.ndarray] = []
        self.mapping: dict[str, int] | None = None
        self.cats: list[str] | None = None

    def add(self, s: pd.Series) -> None:
        kind = _series_kind(s)
        if kind == "unknown":
            if self.kind is None:
                self.parts.append(s.to_numpy(dtype=object, copy=True))
                return
            kind = self.kind
        if self.kind is None:
            # A leading all-null batch cannot stay int64 once real values arrive.
            if self.parts and kind == "int":
                kind = "float"
            self.kind = kind
            if self.parts and kind != "obj":
                self.parts = [self._coerce_unknown(p, kind) for p in self.parts]
        elif self.kind != kind:
            self._promote(kind)
            kind = self.kind
        if kind == "str":
            if self.mapping is None or self.cats is None:
                self.mapping, self.cats = {}, []
            self.parts.append(_encode_strings(s, self.mapping, self.cats))
        elif kind == "dt":
            self.parts.append(_datetime_ns(s))
        elif kind == "float":
            self.parts.append(
                pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", copy=True)
            )
        elif kind == "int":
            if s.isna().any():
                self._promote("float")
                self.parts.append(
                    pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", copy=True)
                )
            else:
                self.parts.append(s.to_numpy(dtype="int64", copy=True))
        elif kind == "bool":
            self.parts.append(s.to_numpy(dtype=bool, copy=True))
        else:
            self.parts.append(s.to_numpy(dtype=object, copy=True))

    def _promote(self, kind: str) -> None:
        if {self.kind, kind} <= {"int", "float"}:
            self.kind = "float"
            self.parts = [
                p.astype("float64", copy=False) if getattr(p.dtype, "kind", "") in "iu" else p
                for p in self.parts
            ]
            return
        # A later batch disagreed (rare for a SQL column). Keep object arrays.
        if self.kind == "str" and self.cats is not None:
            decoded: list[np.ndarray] = []
            cats = self.cats
            for codes in self.parts:
                vals = np.empty(len(codes), dtype=object)
                for i, code in enumerate(codes):
                    vals[i] = None if code < 0 else cats[int(code)]
                decoded.append(vals)
            self.parts = decoded
            self.mapping = None
            self.cats = None
        self.kind = "obj"

    @staticmethod
    def _coerce_unknown(part: np.ndarray, kind: str) -> np.ndarray:
        if kind == "str":
            return np.full(len(part), -1, dtype=np.int32)
        if kind == "dt":
            return np.full(len(part), np.datetime64("NaT"), dtype="datetime64[ns]")
        if kind == "float":
            return np.full(len(part), np.nan, dtype="float64")
        if kind == "int":
            return np.full(len(part), np.nan, dtype="float64")  # nulls cannot stay int
        if kind == "bool":
            return np.zeros(len(part), dtype=bool)
        return part

    def finish(self) -> pd.Series:
        if not self.parts:
            return pd.Series(dtype="object")
        if self.kind == "str" and self.cats is not None:
            codes = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
            return pd.Series(pd.Categorical.from_codes(codes, categories=self.cats))
        if self.kind == "dt":
            arr = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
            return pd.Series(arr, dtype="datetime64[ns, UTC]")
        if self.kind in ("float", "int", "bool"):
            arr = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
            if self.kind == "int":
                # An all-null prefix was stored as float NaN.
                if arr.dtype.kind == "f":
                    return pd.Series(arr)
            return pd.Series(arr)
        arr = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
        return pd.Series(arr)


def frame_from_cursor(cur, columns: list[str], *, batch: int = _QUERY_BATCH) -> pd.DataFrame:
    """Drain a server-side cursor into one DataFrame, batch by batch."""
    acc = {col: _ColAccum() for col in columns}
    n = 0
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        n += len(rows)
        frame = pd.DataFrame.from_records(rows, columns=columns)
        del rows
        for col in columns:
            acc[col].add(frame[col])
        del frame
    if n == 0:
        return pd.DataFrame(columns=columns)
    if n >= 1_000_000:
        logger.info("query_streamed rows=%d cols=%d", n, len(columns))
    return pd.DataFrame({col: acc[col].finish() for col in columns})


def _utc_feature_index(df: pd.DataFrame) -> pd.DataFrame:
    """Copy *df* with a UTC DatetimeIndex (no-op on empty)."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        return out
    if idx.tz is None:
        out.index = idx.tz_localize("UTC")
    else:
        out.index = idx.tz_convert("UTC")
    return out


def patch_feature_frames(existing: pd.DataFrame, built: pd.DataFrame) -> pd.DataFrame:
    """Merge *built* into *existing* without clobbering finite live values.

    * New timestamps (missing rows) are taken from *built*.
    * New columns are taken from *built*.
    * Existing cells that are NaN are filled from *built*.
    * Finite existing cells are kept (live path wins over a shorter rebuild).
    """
    existing = _utc_feature_index(existing)
    built = _utc_feature_index(built)
    if existing is None or existing.empty:
        return built
    if built is None or built.empty:
        return existing
    cols = list(dict.fromkeys(list(existing.columns) + list(built.columns)))
    idx = existing.index.union(built.index).sort_values()
    ex = existing.reindex(index=idx, columns=cols)
    nw = built.reindex(index=idx, columns=cols)
    return ex.combine_first(nw)


class FeatureEngine:
    """Orchestrates feature computation from TimescaleDB sources."""

    def __init__(
        self,
        timestep: str | None = None,
        data_dir: str | None = None,
        timezone: str | None = None,
        training_start: str | None = None,
        training_end: str | None = None,
        market_open: str | None = None,
    ) -> None:
        self.timestep = timestep or get_timestep()
        self.data_dir = Path(data_dir or os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))
        self.timezone = timezone or os.environ.get("VTECH_TIMEZONE", "America/Chicago")
        self.training_start = training_start or os.environ.get("VTECH_TRAINING_HOURS_START", "08:00")
        self.training_end = training_end or os.environ.get("VTECH_TRAINING_HOURS_END", "11:00")
        # CME RTH open (NYSE open) — distinct from the data-window start (08:00).
        # All time_context features that reference "market open" use this value.
        self.market_open = market_open or os.environ.get("VTECH_MARKET_OPEN", "08:30")
        self._dsn = get_dsn()
        # Live runtime sets this so each SQL does not pay a fresh TCP/auth handshake.
        self._keep_conn: bool = False
        self._conn: psycopg.Connection | None = None

    # ── public API ───────────────────────────────────────────────

    # Rolling lookback for incremental extends (covers cs_* 120×1m and rv_*).
    _INCREMENTAL_LOOKBACK = pd.Timedelta(hours=2)
    # If the parquet lags more than this, fall back to a full day rebuild.
    _MAX_INCREMENTAL_GAP = pd.Timedelta(hours=2)

    def _load_day_features(self, target_date: date) -> pd.DataFrame:
        """Session-day frame from the 5s hive (empty if that day is missing)."""
        from src.features.hive_5s import load_session_day

        return _utc_feature_index(load_session_day(target_date))

    def _persist_day_features(self, frame: pd.DataFrame) -> Path:
        """Merge *frame* into 5s hive month parts. Returns a written part path."""
        from src.features.hive_5s import persist_session_day

        written = persist_session_day(frame)
        if written:
            return written[-1]
        return native_tf_dir("5s")

    def extend_date(
        self,
        target_date: date,
        *,
        asof: pd.Timestamp | datetime | None = None,
    ) -> Path:
        """Append newly closed 5s feature rows without recomputing prior bars.

        Live path: for each completed bar, compute **only** that bar's feature
        row (using a lookback window of raw ticks for rolling features) and
        merge it into the 5s hive month parts. Prior rows are kept as-is.

        Bootstrap / safety:
        * missing or empty hive day → full :meth:`build_date` (bootstrap)
        * already caught up to the last closed bar → no-op (return path)
        * any lag (including multi-hour) → windowed recompute + append only
          (never full-day recompute of prior rows on the live path)
        """
        existing = self._load_day_features(target_date)
        last_closed = last_closed_bar_ts(asof, timestep=self.timestep)

        if existing.empty:
            logger.info(
                "extend_date_bootstrap date=%s reason=empty",
                target_date,
            )
            return self.build_date(target_date)

        last_ex = existing.index.max()
        last_ex = pd.Timestamp(last_ex)
        if last_ex.tzinfo is None:
            last_ex = last_ex.tz_localize("UTC")
        else:
            last_ex = last_ex.tz_convert("UTC")

        if last_ex >= last_closed:
            logger.info(
                "extend_date_uptodate date=%s last=%s closed=%s",
                target_date, last_ex.isoformat(), last_closed.isoformat(),
            )
            return self._persist_day_features(existing)

        gap = last_closed - last_ex
        logger.info(
            "extend_date_start date=%s from=%s to=%s n_existing=%d",
            target_date,
            (last_ex + pd.Timedelta(seconds=timestep_seconds(self.timestep))).isoformat(),
            last_closed.isoformat(),
            len(existing),
        )
        t0 = time.time()
        new_frame = self._compute_feature_frame(
            target_date,
            data_start=last_ex - self._INCREMENTAL_LOOKBACK,
            data_end=last_closed + pd.Timedelta(seconds=timestep_seconds(self.timestep)),
            quiet=True,
        )
        if new_frame is None or new_frame.empty:
            logger.warning("extend_date_no_features date=%s", target_date)
            return self._persist_day_features(existing)

        # Keep only bars strictly after the last persisted row, up to last closed.
        idx = new_frame.index
        if idx.tz is None:
            # compute frame may be tz-aware or naive; normalize compare in UTC
            new_frame = new_frame.copy()
            new_frame.index = idx.tz_localize("UTC")
            idx = new_frame.index
        else:
            new_frame = new_frame.copy()
            new_frame.index = idx.tz_convert("UTC")

        new_rows = new_frame.loc[(new_frame.index > last_ex) & (new_frame.index <= last_closed)]
        if new_rows.empty:
            logger.info(
                "extend_date_no_new_rows date=%s last_ex=%s last_closed=%s",
                target_date, last_ex.isoformat(), last_closed.isoformat(),
            )
            return self._persist_day_features(existing)

        combined = merge_feature_rows(existing, new_rows)
        out_path = self._persist_day_features(combined)
        logger.info(
            "extend_date_done date=%s appended=%d total=%d elapsed_s=%.2f",
            target_date, len(new_rows), len(combined), time.time() - t0,
        )
        return out_path

    def _parse_hhmm(self, hhmm: str) -> tuple[int, int]:
        parts = str(hhmm).strip().split(":")
        return int(parts[0]), int(parts[1] if len(parts) > 1 else 0)

    def _hhmm_le(self, a: str, b: str) -> bool:
        ah, am = self._parse_hhmm(a)
        bh, bm = self._parse_hhmm(b)
        return (ah, am) <= (bh, bm)

    def _expected_bar_index(self, target_date: date, start_hhmm: str, end_hhmm: str) -> pd.DatetimeIndex:
        """UTC 5s labels for [start, end) in *self.timezone* on *target_date*."""
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(self.timezone)
        sh, sm = self._parse_hhmm(start_hhmm)
        eh, em = self._parse_hhmm(end_hhmm)
        start = pd.Timestamp(
            datetime(target_date.year, target_date.month, target_date.day, sh, sm, tzinfo=tz)
        ).tz_convert("UTC")
        end = pd.Timestamp(
            datetime(target_date.year, target_date.month, target_date.day, eh, em, tzinfo=tz)
        ).tz_convert("UTC")
        step = pd.Timedelta(seconds=timestep_seconds(self.timestep))
        if end <= start:
            return pd.DatetimeIndex([], tz="UTC")
        return pd.date_range(start, end - step, freq=self.timestep, tz="UTC")

    def _widest_patch_hours(
        self, target_date: date, existing: pd.DataFrame,
    ) -> tuple[str, str]:
        """Session [start, end) covering engine defaults, live objective, and existing file."""
        start, end = self.training_start, self.training_end
        try:
            from src.webui.live import active_objective_id, objective_trade_params
            trade = objective_trade_params(active_objective_id())
            obj_start = trade.get("training_start") or trade.get("trading_start")
            obj_end = trade.get("training_end") or trade.get("trading_end")
            if obj_start and self._hhmm_le(str(obj_start), start):
                start = str(obj_start)
            if obj_end and self._hhmm_le(end, str(obj_end)):
                end = str(obj_end)
        except Exception as exc:  # noqa: BLE001
            logger.debug("patch_objective_hours_skip: %s", exc)
        if existing is not None and not existing.empty:
            ex = _utc_feature_index(existing)
            local = ex.index.tz_convert(self.timezone)
            first, last = local.min(), local.max()
            ex_start = f"{first.hour:02d}:{first.minute:02d}"
            nxt = last + pd.Timedelta(seconds=timestep_seconds(self.timestep))
            ex_end = f"{nxt.hour:02d}:{nxt.minute:02d}"
            if self._hhmm_le(ex_start, start):
                start = ex_start
            if self._hhmm_le(end, ex_end):
                end = ex_end
        return start, end

    def patch_date(self, target_date: date) -> Path:
        """Patch a day's parquet: missing file, missing 5s rows, or NaN cells.

        Never deletes finite existing values (so a live 08:30–14:45 file is
        not replaced by an 08:00–11:00 research rebuild). New timestamps and
        NaN cells are filled from a targeted compute.
        """
        existing = self._load_day_features(target_date)
        start_hhmm, end_hhmm = self._widest_patch_hours(target_date, existing)
        expected = self._expected_bar_index(target_date, start_hhmm, end_hhmm)

        if existing.empty:
            logger.info("patch_date_missing_file date=%s → full build", target_date)
            saved_start, saved_end = self.training_start, self.training_end
            self.training_start, self.training_end = start_hhmm, end_hhmm
            try:
                return self.build_date(target_date)
            finally:
                self.training_start, self.training_end = saved_start, saved_end

        missing = expected.difference(existing.index)
        # Feature-like columns only (skip all-NaN meta on purpose still fillable)
        feat_cols = [c for c in existing.columns if not str(c).startswith("_")]
        nan_cells = int(existing[feat_cols].isna().sum().sum()) if feat_cols else 0
        if missing.empty and nan_cells == 0:
            logger.info(
                "patch_date_complete date=%s rows=%d window=%s-%s",
                target_date, len(existing), start_hhmm, end_hhmm,
            )
            return self._persist_day_features(existing)

        logger.info(
            "patch_date_start date=%s missing_rows=%d nan_cells=%d "
            "existing_rows=%d window=%s-%s",
            target_date, len(missing), nan_cells, len(existing),
            start_hhmm, end_hhmm,
        )
        t0 = time.time()
        saved_start, saved_end = self.training_start, self.training_end
        self.training_start, self.training_end = start_hhmm, end_hhmm
        try:
            # Suffix-only hole: compute from last existing bar (lookback) to end.
            suffix_only = (
                not missing.empty
                and nan_cells == 0
                and missing.min() > existing.index.max()
            )
            if suffix_only:
                built = self._compute_feature_frame(
                    target_date,
                    data_start=existing.index.max() - self._INCREMENTAL_LOOKBACK,
                    data_end=expected.max() + pd.Timedelta(
                        seconds=timestep_seconds(self.timestep)
                    ),
                    quiet=True,
                )
            else:
                built = self._compute_feature_frame(target_date, quiet=False)
        finally:
            self.training_start, self.training_end = saved_start, saved_end

        if built is None or built.empty:
            logger.warning("patch_date_compute_empty date=%s — keeping existing", target_date)
            return self._persist_day_features(existing)

        combined = patch_feature_frames(existing, built)
        n_before, n_after = len(existing), len(combined)
        nan_after = int(combined[feat_cols].isna().sum().sum()) if feat_cols else 0
        out_path = self._persist_day_features(combined)
        logger.info(
            "patch_date_done date=%s rows %d→%d nan_cells %d→%d elapsed_s=%.1f",
            target_date, n_before, n_after, nan_cells, nan_after, time.time() - t0,
        )
        _emit(
            f"\n✅ {target_date} patched — {n_after:,} rows "
            f"(+{n_after - n_before:,}), NaN cells {nan_cells}→{nan_after}",
            newline=True,
        )
        return out_path

    def build_date(self, target_date: date) -> Path:
        """Build all features for a single trading day and save to Parquet."""
        logger.info("build_date_start", extra={"date": str(target_date)})
        t0_total = time.time()
        _emit(f"\n⏳ Building features for {target_date}...", newline=True)

        combined = self._compute_feature_frame(target_date, quiet=False)
        if combined is None or combined.empty:
            logger.warning("no_features", extra={"date": str(target_date)})
            return Path()

        # Do not clobber a longer live session (e.g. 08:30–14:45) with the
        # research 08:00–11:00 rebuild.
        existing = self._load_day_features(target_date)
        if not existing.empty and not combined.empty:
            try:
                ex_last = pd.Timestamp(existing.index.max())
                nw_last = pd.Timestamp(combined.index.max())
                if ex_last.tzinfo is None:
                    ex_last = ex_last.tz_localize("UTC")
                else:
                    ex_last = ex_last.tz_convert("UTC")
                if nw_last.tzinfo is None:
                    nw_last = nw_last.tz_localize("UTC")
                else:
                    nw_last = nw_last.tz_convert("UTC")
                if ex_last > nw_last or len(existing) > len(combined):
                    logger.warning(
                        "build_date_skip_overwrite date=%s existing_last=%s "
                        "existing_rows=%d new_last=%s new_rows=%d",
                        target_date, ex_last.isoformat(), len(existing),
                        nw_last.isoformat(), len(combined),
                    )
                    return self._persist_day_features(existing)
            except Exception as exc:  # noqa: BLE001
                logger.warning("build_date_existing_check_failed: %s", exc)
        out_path = self._persist_day_features(combined)

        elapsed = time.time() - t0_total
        el_m, el_s = divmod(int(elapsed), 60)
        _emit(f"\n✅ {target_date} done in {el_m}m{el_s:02d}s — "
              f"{combined.shape[1]} features, {len(combined):,} rows, "
              f"{combined.isna().mean().mean():.1%} NaN", newline=True)

        logger.info("build_date_done", extra={
            "date": str(target_date),
            "features": combined.shape[1],
            "rows": len(combined),
            "nan_pct": float(combined.isna().mean().mean()),
            "path": str(out_path),
        })
        return out_path

    def _write_feature_parquet(self, out_path: Path, combined: pd.DataFrame) -> None:
        """Atomic parquet write + best-effort manifest sidecar."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        combined.to_parquet(tmp, engine="pyarrow", compression="zstd")
        tmp.replace(out_path)
        try:
            from src.common.data_manifest import write_manifest
            write_manifest(out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("manifest_write_failed", extra={
                "path": str(out_path), "error": str(exc),
            })

    def _compute_feature_frame(
        self,
        target_date: date,
        *,
        data_start: pd.Timestamp | datetime | None = None,
        data_end: pd.Timestamp | datetime | None = None,
        quiet: bool = False,
        loaded: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Compute the feature matrix for *target_date* (optionally time-bounded).

        When *data_start* / *data_end* are set, source loads are restricted to
        that window (used by :meth:`extend_date` so only new bars are derived
        from a lookback of raw ticks).  Output is still filtered to training
        hours.

        When *loaded* is provided (warm live path), DB loaders are skipped and
        the preloaded frames are used as-is:
        ``tbbo``, ``bbo_1s`` (or split ``bbo_fut``/``bbo_opt``), ``definitions``,
        ``statistics``, ``candles_1m``, optional ``mbp_depth`` (5s AVG of MBP-10).
        """
        def emit(msg: str, newline: bool = False) -> None:
            if not quiet:
                _emit(msg, newline=newline)

        t0_total = time.time()

        # ── Phase 0: Resolve lead contracts via CME roll calendar ──
        self._primary_nq = active_contract("NQ", target_date)
        self._primary_es = active_contract("ES", target_date)
        emit(f"  Lead contracts: NQ={self._primary_nq}  ES={self._primary_es}", newline=True)

        # Normalize optional bounds to UTC timestamps.
        t0 = None
        t1 = None
        if data_start is not None:
            t0 = pd.Timestamp(data_start)
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            else:
                t0 = t0.tz_convert("UTC")
        if data_end is not None:
            t1 = pd.Timestamp(data_end)
            if t1.tzinfo is None:
                t1 = t1.tz_localize("UTC")
            else:
                t1 = t1.tz_convert("UTC")

        # Incremental path: always use primary-futures BBO (and primary-filtered
        # options when present) to avoid multi-10M-row full-chain scans.
        incremental = t0 is not None or t1 is not None or loaded is not None
        _skip_options = os.environ.get("VTECH_SKIP_OPTIONS", "").strip() == "1"

        if loaded is not None:
            emit("  Using warm preloaded frames (no DB full reload)", newline=True)
            tbbo = loaded.get("tbbo", pd.DataFrame())
            defs = loaded.get("definitions", pd.DataFrame())
            stats = loaded.get("statistics", pd.DataFrame())
            candles_1m = loaded.get("candles_1m", pd.DataFrame())
            mbp_bars = loaded.get("mbp_depth", pd.DataFrame())
            if "bbo_fut" in loaded or "bbo_opt" in loaded:
                bbo_fut = loaded.get("bbo_fut", pd.DataFrame())
                bbo_opt = loaded.get("bbo_opt", pd.DataFrame())
                # skip split below
                _pre_split = True
            else:
                bbo_all = loaded.get("bbo_1s", pd.DataFrame())
                _pre_split = False
        else:
            _pre_split = False
            if incremental and not _skip_options:
                # Same full module set as research; still primary-filter chains to
                # avoid multi-10M-row unfiltered scans. Lookback uses data_start.
                _bbo_loader = lambda: self._load_bbo_1s(
                    target_date, t_start=t0, t_end=t1, primary_only=True,
                )
            elif _skip_options:
                _bbo_loader = lambda: self._load_bbo_fut_primary(
                    target_date, self._primary_nq, t_start=t0, t_end=t1,
                )
            else:
                _bbo_loader = lambda: self._load_bbo_1s(target_date, t_start=t0, t_end=t1)

            load_steps = [
                ("tbbo",        lambda: self._load_tbbo(target_date, t_start=t0, t_end=t1)),
                ("bbo_1s",      _bbo_loader),
                ("definitions", lambda: self._load_definitions(target_date)),
                ("statistics",  lambda: self._load_statistics(target_date)),
                ("candles_1m",  lambda: self._load_candles_1min(target_date, t_start=t0, t_end=t1)),
                ("mbp_depth",   lambda: self._load_mbp10_depth_bars(target_date, t_start=t0, t_end=t1)),
            ]
            loaded_local: dict[str, pd.DataFrame] = {}
            for i, (name, loader) in enumerate(load_steps):
                emit(_progress_bar(i, len(load_steps), f"loading {name}", t0_total))
                t_q = time.time()
                loaded_local[name] = loader()
                rows = len(loaded_local[name])
                dt = time.time() - t_q
                emit(_progress_bar(i + 1, len(load_steps),
                                    f"{name}: {rows:,} rows in {dt:.1f}s", t0_total))
            emit("", newline=True)

            tbbo = loaded_local["tbbo"]
            bbo_all = loaded_local["bbo_1s"]
            defs = loaded_local["definitions"]
            stats = loaded_local["statistics"]
            candles_1m = loaded_local["candles_1m"]
            mbp_bars = loaded_local["mbp_depth"]

        if not _pre_split:
            if not defs.empty and "instrument_class" in defs.columns:
                opt_symbols = set(defs.loc[defs["instrument_class"].isin(("C", "P")), "symbol"])
            else:
                opt_symbols = set()
            if not bbo_all.empty and opt_symbols:
                bbo_fut = bbo_all[~bbo_all["symbol"].isin(opt_symbols)]
                bbo_opt = bbo_all[bbo_all["symbol"].isin(opt_symbols)]
                del bbo_all
            else:
                bbo_fut = bbo_all
                bbo_opt = pd.DataFrame()

        if not bbo_fut.empty and "symbol" in bbo_fut.columns:
            bbo_fut = bbo_fut[bbo_fut["symbol"] == self._primary_nq].copy()

        und_prices = pd.Series(dtype=np.float64)
        if not bbo_fut.empty:
            und_series = bbo_fut.set_index("ts_recv").sort_index()
            und_prices = ((und_series["bid_px"] + und_series["ask_px"]) / 2).resample(self.timestep).last().ffill()
        if und_prices.empty and not tbbo.empty:
            nq_primary = tbbo[tbbo["symbol"] == self._primary_nq]
            if not nq_primary.empty:
                und_series = nq_primary.set_index("ts_recv").sort_index()
                und_prices = ((und_series["bid_px"] + und_series["ask_px"]) / 2).resample(self.timestep).last().ffill()

        all_features: list[pd.DataFrame] = []
        modules: list[tuple[str, callable]] = []

        if not bbo_fut.empty:
            modules.append(("book_pressure", lambda: book_pressure.compute(bbo_fut, self.timestep)))
        if not tbbo.empty:
            modules.append(("order_flow", lambda: order_flow.compute(tbbo, self.timestep)))
        if not tbbo.empty:
            modules.append(("trade_location", lambda: trade_location.compute(tbbo, self.timestep)))
        if not tbbo.empty:
            modules.append(("same_side", lambda: same_side.compute(tbbo, self.timestep)))
        if not tbbo.empty:
            modules.append(("large_trade_cvd", lambda: large_trade_cvd.compute(tbbo, self.timestep)))
        if not tbbo.empty:
            modules.append(("trade_arrival", lambda: trade_arrival.compute(tbbo, self.timestep)))
        if not _skip_options and not bbo_opt.empty and not defs.empty and not und_prices.empty:
            modules.append(("options_surface", lambda: options_surface.compute(bbo_opt, defs, und_prices, self.timestep)))
        if not _skip_options and not bbo_opt.empty and not defs.empty and not und_prices.empty:
            modules.append(("dealer_gex", lambda: dealer_gex.compute(bbo_opt, defs, und_prices, self.timestep)))
        if not stats.empty and not und_prices.empty:
            modules.append(("daily_context", lambda: daily_context.compute(stats, und_prices, self.timestep)))
        if not tbbo.empty:
            modules.append(("microstructure", lambda: microstructure.compute(tbbo, self.timestep)))
        if not und_prices.empty:
            modules.append(("realized_vol", lambda: realized_vol.compute(und_prices, self.timestep)))
        if not bbo_fut.empty or not tbbo.empty:
            modules.append(("sub_bar_dynamics", lambda: sub_bar_dynamics.compute(
                bbo_fut if not bbo_fut.empty else pd.DataFrame(),
                tbbo if not tbbo.empty else pd.DataFrame(),
                self.timestep,
            )))
        if not tbbo.empty:
            modules.append(("volume_footprint", lambda: volume_footprint.compute(
                tbbo,
                grid=und_prices.index if not und_prices.empty else None,
            )))
        if not tbbo.empty and not und_prices.empty:
            modules.append(("volume_profile", lambda: volume_profile.compute(tbbo, und_prices, self.timestep)))
            prev_d = target_date
            from datetime import timedelta as _td
            prev_d = target_date - _td(days=1)
            while prev_d.weekday() >= 5:
                prev_d -= _td(days=1)
            prev_tbbo = self._load_tbbo(prev_d)
            modules.append(("volume_profile_yday",
                            lambda: volume_profile.compute_prior_session(
                                prev_tbbo, und_prices, self.timestep)))
        if not und_prices.empty:
            modules.append(("hawkes_clustering", lambda: hawkes_clustering.compute(und_prices, self.timestep)))
        if not candles_1m.empty and len(candles_1m) >= 64:
            modules.append(("wavelets", lambda: wavelets.compute(candles_1m)))
        if not candles_1m.empty:
            modules.append(("candle_structure", lambda: candle_structure.compute(candles_1m)))
        if not candles_1m.empty and len(candles_1m) >= 15:
            modules.append(("candle_multitf", lambda: candle_structure.compute_multitimeframe(candles_1m)))
        if not tbbo.empty:
            modules.append(("candle_submin",
                            lambda: candle_structure.compute_subminute_clv(tbbo, self.timestep)))
        if not candles_1m.empty:
            modules.append(("stochastic", lambda: stochastic.compute(candles_1m)))
        if not candles_1m.empty:
            modules.append(("atr", lambda: atr_mod.compute(
                candles_1m, candles_1m.index)))
        if not candles_1m.empty:
            modules.append(("psar", lambda: psar.compute(
                candles_1m,
                grid=und_prices.index if not und_prices.empty else None,
                tbbo=tbbo if not tbbo.empty else None,
                timestep=self.timestep,
            )))
        if not tbbo.empty:
            modules.append(("vwap", lambda: vwap.compute(tbbo, self.timestep)))
        modules.append(("mbp_depth", lambda: mbp_depth.compute(
            mbp_bars,
            grid=und_prices.index if not und_prices.empty else None,
            timestep=self.timestep,
        )))

        live_fast = bool(loaded is not None and loaded.get("_live_fast"))
        if live_fast:
            # Hot 5s path: skip modules that are O(lookback) heavy and not
            # required to refresh every bar (IV/GEX already skipped via empty
            # bbo_opt). Values persist via ffill from the last full refresh.
            _skip = {"hawkes_clustering", "wavelets", "sub_bar_dynamics"}
            modules = [(n, fn) for n, fn in modules if n not in _skip]

        t0_feat = time.time()
        n_mod = len(modules)
        emit(f"  Computing {n_mod} feature modules...", newline=True)

        ONE_MIN_MODULES = {
            "wavelets", "candle_structure", "candle_multitf",
            "stochastic", "atr",
        }
        # 1-min OHLC consumers: left-labeled T is visible at T+1m (ADR-041).
        # wavelets already exclude the current 1-min bar; atr/multitf are
        # right-labeled slower bars — do not extra-shift those.
        ONE_MIN_CLOSED_SHIFT = {"candle_structure", "stochastic"}
        # Prefixes produced only at 1m (or slower) — or refreshed on a slower
        # live cadence (iv_/gx_) — that must be ffilled onto the 5s grid.
        _SLOW_FEATURE_PREFIXES = (
            "cs_", "wv_", "iv_", "gx_", "ca_", "ht_", "hk_", "so_", "vf_",
        )

        for i, (mod_name, compute_fn) in enumerate(modules):
            emit(_progress_bar(i, n_mod, mod_name, t0_feat))
            t_m = time.time()
            result = compute_fn()
            if result is not None and not result.empty:
                if mod_name in ONE_MIN_MODULES and self.timestep != "1min":
                    # Normalize tz so date_range does not fail on Etc/UTC vs UTC.
                    idx = result.index
                    if getattr(idx, "tz", None) is not None:
                        result = result.copy()
                        result.index = idx.tz_convert("UTC")
                    end = result.index.max()
                    if t1 is not None:
                        t1u = pd.Timestamp(t1)
                        if t1u.tzinfo is None:
                            t1u = t1u.tz_localize("UTC")
                        else:
                            t1u = t1u.tz_convert("UTC")
                        if t1u > end:
                            end = t1u
                    new_idx = pd.date_range(
                        result.index.min(), end,
                        freq=self.timestep, tz="UTC",
                    )
                    if mod_name in ONE_MIN_CLOSED_SHIFT:
                        from src.features.closed_bars import align_closed_1m_to_grid
                        result = align_closed_1m_to_grid(result, new_idx)
                    else:
                        result = result.reindex(new_idx, method="ffill")
                all_features.append(result)
            dt = time.time() - t_m
            cols = result.shape[1] if result is not None and not result.empty else 0
            emit(_progress_bar(i + 1, n_mod,
                                f"{mod_name}: {cols} cols in {dt:.1f}s", t0_feat))

        if not live_fast:
            emit(_progress_bar(n_mod, n_mod + 1, "cross_asset (loading ES)", t0_feat))
            es_candles = self._load_es_candles_1min(target_date, t_start=t0, t_end=t1)
            if not candles_1m.empty and not es_candles.empty:
                from src.features.closed_bars import shift_closed_1m
                nq_closed = shift_closed_1m(candles_1m[["close"]])
                es_closed = shift_closed_1m(es_candles[["close"]])
                nq_resampled = nq_closed.resample(self.timestep).last().ffill()
                es_resampled = es_closed.resample(self.timestep).last().ffill()
                ca_feats = cross_asset.compute(nq_resampled, es_resampled, self.timestep)
                if not ca_feats.empty:
                    all_features.append(ca_feats)

        if all_features:
            emit(_progress_bar(n_mod, n_mod + 1, "time_context", t0_feat))
            idx = all_features[0].index
            close_ser = None
            if not candles_1m.empty and "close" in candles_1m.columns:
                close_ser = candles_1m["close"].resample(self.timestep).last().ffill().reindex(idx)
            all_features.append(
                time_context.compute(
                    idx, candles_1m if not candles_1m.empty else None,
                    close_series=close_ser,
                    training_start=self.training_start,
                    training_end=self.training_end,
                    market_open=self.market_open,
                )
            )

        if all_features:
            emit(_progress_bar(n_mod, n_mod + 1, "economic_calendar", t0_feat))
            idx = all_features[0].index
            all_features.append(economic_calendar.compute(idx, self.timezone))

        if all_features and not live_fast:
            emit(_progress_bar(n_mod, n_mod + 1, "higher_timeframe", t0_feat))
            idx = all_features[0].index
            daily_ohlc = self._load_daily_ohlc(target_date)
            all_features.append(higher_timeframe.compute(daily_ohlc, target_date, idx))
            ytd_hl = self._load_ytd_hl(target_date)
            px = und_prices if not und_prices.empty else (
                candles_1m["close"] if not candles_1m.empty else pd.Series(dtype="float64")
            )
            all_features.append(higher_timeframe.compute_ytd(
                ytd_hl, candles_1m, px, idx))
            vix_1m = self._load_vix_1m(target_date)
            from src.features import vix as vix_mod
            all_features.append(vix_mod.compute_vix(vix_1m, idx))

        emit(_progress_bar(n_mod + 1, n_mod + 1, "all modules done", t0_feat), newline=True)

        if not all_features:
            return pd.DataFrame()

        emit("  Concatenating...", newline=False)
        deduped = [f.loc[~f.index.duplicated(keep="first")] for f in all_features]
        combined = pd.concat(deduped, axis=1)
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()
        # 5s modules (order flow, etc.) often extend past the last completed 1m
        # candle. Without ffill, cs_*/wv_* are NaN on that trailing edge — Live
        # then reports e.g. 25/43 model features (all 18 cs_* missing).
        slow_cols = [
            c for c in combined.columns
            if isinstance(c, str)
            and c.startswith(_SLOW_FEATURE_PREFIXES)
            and c not in candle_structure.CS_5S_OHLC_COLS
        ]
        if slow_cols:
            combined[slow_cols] = combined[slow_cols].ffill()
        combined = self._filter_training_hours(combined)

        if not tbbo.empty and "price" in tbbo.columns:
            close_5s = (
                tbbo.set_index("ts_event").sort_index()["price"]
                .resample(self.timestep)
                .last()
                .ffill()
            )
            combined["_close"] = close_5s.reindex(combined.index)
        elif not candles_1m.empty and "close" in candles_1m.columns:
            close_resampled = candles_1m["close"].resample(self.timestep).last().ffill()
            combined["_close"] = close_resampled.reindex(combined.index)

        if "_close" in combined.columns:
            ohlc5 = candle_structure.compute_5s_ohlc(
                tbbo if not tbbo.empty else None,
                grid=combined.index,
                timestep=self.timestep,
                close=combined["_close"],
            )
            if ohlc5 is not None and not ohlc5.empty:
                drop = [c for c in ohlc5.columns if c in combined.columns]
                if drop:
                    combined = combined.drop(columns=drop)
                combined = combined.join(ohlc5)

        combined["_meta_nq_contract"] = self._primary_nq
        combined["_meta_es_contract"] = self._primary_es
        try:
            from src.features.feat_ext import attach_iv_rank
            cache = Path(os.environ.get(
                "VTECH_IV_DAILY_CACHE",
                str(native_tf_features_root() / "iv_daily_atm.parquet"),
            ))
            daily_iv = pd.read_parquet(cache) if Path(cache).exists() else pd.DataFrame()
            iv_src = combined["iv_atm_mean"] if "iv_atm_mean" in combined.columns else None
            combined = attach_iv_rank(combined, iv_src, daily_iv, target_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("iv_rank_attach_failed: %s", exc)
            if "iv_rank_365" not in combined.columns:
                combined["iv_rank_365"] = np.nan
        return combined


    def build_range(self, start: date, end: date, skip_existing: bool = True) -> list[Path]:
        """Build features for every trading day in [start, end)."""
        # Collect regular trading days (excludes weekends + CME holidays/early-close)
        days = []
        current = start
        while current < end:
            if is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)

        t0_range = time.time()
        n = len(days)
        have: set[str] = set()
        if skip_existing:
            from src.backtest.runner import _available_dates
            have = set(_available_dates(native_tf_dir("5s")))
            to_build = [d for d in days if d.isoformat() not in have]
            _emit(f"\n⏳ Building features for {len(to_build)}/{n} trading days ({start} → {end}); {n - len(to_build)} already exist.", newline=True)
        else:
            to_build = days
            _emit(f"\n⏳ Building features for {n} trading days ({start} → {end})...", newline=True)

        paths = []
        m = len(to_build)
        for i, d in enumerate(to_build):
            _emit(_progress_bar(i, m, str(d), t0_range), newline=True)
            paths.append(self.build_date(d))

        elapsed = time.time() - t0_range
        el_m, el_s = divmod(int(elapsed), 60)
        ok = sum(1 for p in paths if p != Path())
        _emit(f"\n✅ Range complete: {ok}/{m} days in {el_m}m{el_s:02d}s", newline=True)
        return [p for p in paths if p != Path()]

    # ── time filter ──────────────────────────────────────────────

    def _filter_training_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to rows within training_start–training_end in local timezone."""
        if df.empty:
            return df

        idx = df.index
        if idx.tz is None:
            idx_local = idx.tz_localize("UTC").tz_convert(self.timezone)
        else:
            idx_local = idx.tz_convert(self.timezone)

        start_h, start_m = (int(x) for x in self.training_start.split(":"))
        end_h, end_m = (int(x) for x in self.training_end.split(":"))

        bar_minutes = idx_local.hour * 60 + idx_local.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        mask = (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)
        return df.loc[mask]

    # ── data loaders ─────────────────────────────────────────────

    def _query(self, sql: str, params: tuple) -> pd.DataFrame:
        """Run a read and return a DataFrame without buffering every cell as a Python object.

        A client cursor's ``fetchall`` of one options-BBO day (~42M rows) holds the
        whole result as tuples and is what pushed this process past the web UI
        cgroup cap. The named cursor streams batches; :func:`frame_from_cursor`
        compacts each batch before the next one arrives.
        """
        own = False
        if self._keep_conn:
            if self._conn is None or self._conn.closed:
                # autocommit: the kept connection is read-only, and psycopg's
                # default (autocommit=False) would hold an implicit transaction
                # open for the life of the process — blocking Timescale
                # compression behind an idle-in-transaction snapshot.
                self._conn = psycopg.connect(self._dsn, autocommit=True)
            conn = self._conn
        else:
            conn = psycopg.connect(self._dsn, autocommit=True)
            own = True
        try:
            # Named cursors need a transaction. It lasts for this read only.
            with conn.transaction():
                name = f"vtech_q_{time.monotonic_ns()}"
                with conn.cursor(name=name) as cur:
                    cur.itersize = _QUERY_BATCH
                    cur.execute(sql, params)
                    if cur.description is None:
                        return pd.DataFrame()
                    columns = [desc[0] for desc in cur.description]
                    return frame_from_cursor(cur, columns)
        finally:
            if own:
                conn.close()

    def _refresh_cagg(self, d_start: date, d_end: date) -> None:
        """Refresh tbbo_1min continuous aggregate for a date range."""
        logger.info("Refreshing tbbo_1min cagg for %s → %s", d_start, d_end)
        with psycopg.connect(self._dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CALL refresh_continuous_aggregate("
                    "'databento.tbbo_1min', %s::timestamptz, %s::timestamptz)",
                    (f"{d_start} 00:00:00+00", f"{d_end} 00:00:00+00"),
                )

    def _load_tbbo(
        self,
        d: date,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load tbbo rows for the lead-month NQ contract on date `d`.

        Optional *t_start*/*t_end* (UTC) restrict the query for incremental extends.
        """
        d_next = d + timedelta(days=1)
        start = t_start if t_start is not None else d
        end = t_end if t_end is not None else d_next
        return self._query(
            "SELECT * FROM databento.tbbo "
            "WHERE ts_event >= %s AND ts_event < %s "
            "AND symbol = %s ORDER BY ts_event",
            (start, end, self._primary_nq),
        )

    def _hhmm_utc(self, d: date, hhmm: str) -> datetime:
        from zoneinfo import ZoneInfo
        h, m = (int(x) for x in str(hhmm).split(":")[:2])
        local = datetime(d.year, d.month, d.day, h, m, tzinfo=ZoneInfo(self.timezone))
        return local.astimezone(timezone.utc)

    def _load_mbp10_depth_bars(
        self,
        d: date,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """5s AVG of MBP-10 L0–L9 size sums (server-side; no raw event pull).

        Empty / missing table → empty frame. Callers map onto the 5s grid via
        ``mbp_depth.compute``.
        """
        from src.features.mbp_depth import level_sum_sql

        step = f"{timestep_seconds(self.timestep)} seconds"
        if t_start is not None:
            start = t_start
        else:
            start = self._hhmm_utc(d, self.training_start) - timedelta(seconds=5)
        if t_end is not None:
            end = t_end
        else:
            end = self._hhmm_utc(d, self.training_end) + timedelta(seconds=5)
        sym = symbol or getattr(self, "_primary_nq", None)
        if not sym:
            return pd.DataFrame()
        bid = level_sum_sql("bid")
        ask = level_sum_sql("ask")
        sql = (
            "SELECT time_bucket(%s::interval, ts_recv) AS ts, "
            f"AVG({bid}) AS bid_count, "
            f"AVG({ask}) AS ask_count, "
            f"AVG(({bid}) - ({ask})) AS imbalance_diff, "
            f"AVG(LN({bid}) - LN({ask})) "
            f"FILTER (WHERE ({bid}) > 0 AND ({ask}) > 0) AS imbalance_log "
            "FROM databento.mbp_10 "
            "WHERE ts_recv >= %s AND ts_recv < %s AND symbol = %s "
            "GROUP BY 1 ORDER BY 1"
        )
        try:
            return self._query(sql, (step, start, end, sym))
        except Exception as exc:  # noqa: BLE001 — MBP-10 is a 1-month window
            logger.warning("mbp_10 depth bars skipped date=%s: %s", d, exc)
            if self._keep_conn and self._conn is not None and not self._conn.closed:
                try:
                    self._conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            return pd.DataFrame()

    def _load_bbo_window(
        self,
        d: date,
        *,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Short-window BBO-1s load for live deltas (no day-wide primary-id scan)."""
        _COLS = "ts_recv, symbol, bid_px, ask_px, bid_sz, ask_sz"
        if symbols:
            return self._query(
                f"SELECT {_COLS} FROM databento.bbo_1s "
                "WHERE ts_recv >= %s AND ts_recv < %s AND symbol = ANY(%s) "
                "ORDER BY ts_recv",
                (t_start, t_end, list(symbols)),
            )
        return self._query(
            f"SELECT {_COLS} FROM databento.bbo_1s "
            "WHERE ts_recv >= %s AND ts_recv < %s ORDER BY ts_recv",
            (t_start, t_end),
        )

    def _load_bbo_fut_primary(
        self,
        d: date,
        symbol: str,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load BBO-1s for the primary futures contract only.

        Used when ``VTECH_SKIP_OPTIONS=1`` to avoid loading the full options
        chain data.  Reduces the bbo_1s load from ~35M rows (~13 GB peak RSS)
        to ~100-500K rows (~200 MB) per worker.
        """
        d_next = d + timedelta(days=1)
        start = t_start if t_start is not None else d
        end = t_end if t_end is not None else d_next
        _COLS = "ts_recv, symbol, bid_px, ask_px, bid_sz, ask_sz"
        return self._query(
            f"SELECT {_COLS} FROM databento.bbo_1s "
            "WHERE ts_recv >= %s AND ts_recv < %s AND symbol = %s ORDER BY ts_recv",
            (start, end, symbol),
        )

    def _load_bbo_1s(
        self,
        d: date,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
        primary_only: bool | None = None,
    ) -> pd.DataFrame:
        """Load BBO-1s for the day, optionally filtered to live NQ option chains.

        Unless ``VTECH_BBO_PRIMARY_ONLY=0`` is set (or *primary_only* is False),
        the loader restricts bbo_1s to:

        * all futures (``instrument_class = 'F'``), plus
        * NQ options from exactly the five ADR-034 session chains — 0DTE, 1DTE,
          the next two QNE end-of-month expiries, and the front quarterly.

        That is the same universe ``vtech-live-streamer`` subscribes, and it is
        the set the feature modules actually consume:

        * ``options_surface`` v4 builds every surface feature from the two QNE
          chains (30-day constant-maturity) plus 0DTE/1DTE. ``iv_term_slope`` is
          30d CM-IV − iv_1dte_atm; it no longer reads a quarterly (v3 did).
        * ``dealer_gex`` selects ``0 < DTE <= 90``, which reaches the front
          quarterly and deliberately excludes back-dated quarterlies (168+ DTE).

        Earlier revisions scoped this to "any chain not yet expired" (documented
        variously as a 100-day or 400-day window, while the SQL had no bound at
        all). On 2026-08-20 that matched 10,300 option instruments against the
        3,202 these five chains need — a 3.2x over-read on every load, which is
        the likely driver of the live-features OOM restarts under the old
        hardcoded MemoryMax=32G (now a fraction of host RAM; ADR-060).

        Optional *t_start*/*t_end* (UTC) restrict the query for incremental extends.
        """
        d_next = d + timedelta(days=1)
        start = t_start if t_start is not None else d
        end = t_end if t_end is not None else d_next
        # Only the 6 columns consumed by feature modules; SELECT * on 30M+ rows
        # wastes unused columns and pushes peak RSS well above safe limits.
        # instrument_id dropped: split/join uses symbol instead.
        # ts_event dropped: sub_bar_dynamics indexes on ts_recv.
        _COLS = "ts_recv, symbol, bid_px, ask_px, bid_sz, ask_sz"
        if primary_only is None:
            # Defaults ON: the five-chain scope is the production contract
            # (ADR-034), so batch and live must agree. Previously this defaulted
            # OFF, which meant vtech-feature-builder (no env var) loaded the
            # entire bbo_1s day unfiltered while vtech-live-features (env var
            # set) loaded the narrow set — the same session yielded different
            # features depending on which service built it.
            # Set VTECH_BBO_PRIMARY_ONLY=0 for a deliberate full-universe load.
            primary_only = os.environ.get("VTECH_BBO_PRIMARY_ONLY", "1").strip() != "0"
        if not primary_only:
            return self._query(
                f"SELECT {_COLS} FROM databento.bbo_1s "
                "WHERE ts_recv >= %s AND ts_recv < %s ORDER BY ts_recv",
                (start, end),
            )

        # Resolve instrument_ids: all futures + options from the five ADR-034
        # session chains (0DTE, 1DTE, QNE1, QNE2, front quarterly) — the exact
        # set vtech-live-streamer subscribes, so the consumer contract here
        # matches the producer contract and the two cannot drift apart.
        #
        # Matched on the exact chain root (the leading token of the strike
        # symbol, e.g. 'QNEQ6' in 'QNEQ6 C34000'). NOT a symbol-prefix range
        # filter: NQ 0DTE options use CME parent roots like Q1A.OPT / Q2A.OPT /
        # QN1.OPT whose strike symbols start with 'Q...', not 'NQ...', so a
        # symbol < 'NR' bound would drop them ('Q' > 'N'). databento.definitions
        # is ~11k rows, so the split_part scan is negligible.
        roots = self._session_chain_roots(d)
        if roots:
            ids_df = self._query(
                "SELECT DISTINCT instrument_id FROM databento.definitions "
                "WHERE instrument_class = 'F' "
                "OR (instrument_class IN ('C','P') "
                "    AND split_part(symbol, ' ', 1) = ANY(%s))",
                (roots,),
            )
        else:
            # Non-trading date or unresolvable calendar: fall back to the former
            # not-yet-expired scope rather than silently loading futures only.
            logger.warning("bbo_chain_scope_unresolved", extra={"date": str(d)})
            ids_df = self._query(
                "SELECT DISTINCT instrument_id FROM databento.definitions "
                "WHERE instrument_class = 'F' "
                "OR (instrument_class IN ('C','P') AND expiration::date >= %s)",
                (d,),
            )
        ids: list[int] = ids_df["instrument_id"].astype(int).tolist() if not ids_df.empty else []
        if not ids:
            logger.warning("bbo_primary_filter_empty", extra={"date": str(d)})
            return self._query(
                f"SELECT {_COLS} FROM databento.bbo_1s "
                "WHERE ts_recv >= %s AND ts_recv < %s ORDER BY ts_recv",
                (start, end),
            )

        logger.info("bbo_1s_primary_filter", extra={
            "date": str(d), "primary_nq": self._primary_nq,
            "n_ids": len(ids), "chain_roots": roots,
        })
        return self._query(
            f"SELECT {_COLS} FROM databento.bbo_1s "
            "WHERE ts_recv >= %s AND ts_recv < %s "
            "AND instrument_id = ANY(%s) "
            "ORDER BY ts_recv",
            (start, end, ids),
        )

    @staticmethod
    def _session_chain_roots(d: date) -> list[str]:
        """Symbol roots of the five ADR-034 live option chains for ``d``.

        Pure calendar math via ``options_universe.session_option_targets`` — the
        same resolver ``vtech-live-streamer`` uses, so feature loading and live
        capture stay in lockstep. No network call, no dependency on the local
        ``definitions`` table being current.

        Root rather than expiration date, deliberately: other series can share an
        expiry with a target chain (e.g. the Sep weekly ``QN3U6`` expires on the
        same Friday as the ``NQU6`` quarterly). ``dealer_gex`` selects purely on
        DTE, so an expiry-based scope would let those co-expiring series into its
        gamma aggregation. Root matches exactly what the streamer subscribes.

        Returns ``[]`` when ``d`` is not a CME trading day (the five chains are
        only defined for a session) or if resolution fails; the caller then falls
        back to the previous not-yet-expired scope.
        """
        try:
            from src.acquisition.options_universe import (
                is_trading_day,
                session_option_targets,
            )
        except Exception:  # noqa: BLE001
            logger.exception("session_chain_roots_import_failed")
            return []
        if not is_trading_day(d):
            # Expected for weekends/holidays — not an error, so no traceback.
            logger.info("session_chain_roots_non_trading_day", extra={"date": str(d)})
            return []
        try:
            return sorted({t.root for t in session_option_targets(d)})
        except Exception:  # noqa: BLE001
            logger.exception("session_chain_roots_failed", extra={"date": str(d)})
            return []

    def _load_definitions(self, d: date) -> pd.DataFrame:
        """Load most recent definition for each instrument active on session date `d`.

        Futures (class 'F') are returned unconditionally.
        Options (class 'C'/'P') are returned for any chain that has not yet
        expired — expiration::date >= d.  No ts_event filter: monthly QNE
        chains are ingested with ts_event near expiry-eve, so a ts_event
        guard would hide them for most of their trading life.
        """
        return self._query(
            "SELECT DISTINCT ON (instrument_id) * "
            "FROM databento.definitions "
            "WHERE instrument_class = 'F' "
            "OR (instrument_class IN ('C','P') AND expiration::date >= %s) "
            "ORDER BY instrument_id, ts_event DESC",
            (d,),
        )

    def _load_statistics(self, d: date) -> pd.DataFrame:
        """Load statistics per-date filtered to each day's active NQ contract.

        Uses a 7-day lookback so that OI change (daily diff) has a prior
        reference value. Selects only needed columns — ts_ref contains
        Databento sentinel timestamps (year 48113+) that overflow psycopg's
        parser.

        Per-date filtering resolves the active contract for each of the 7
        lookback days via the CME roll calendar. On/near roll days the
        filter switches contracts mid-window; consumers (e.g. dc_oi_chg)
        may see NaN at the roll boundary.
        """
        d_start = d - timedelta(days=7)
        d_next = d + timedelta(days=1)
        df = self._query(
            "SELECT ts_event, ts_recv, instrument_id, symbol, stat_type, price, quantity "
            "FROM databento.statistics "
            "WHERE ts_event >= %s AND ts_event < %s "
            "AND symbol NOT LIKE '%%-%%' "
            "ORDER BY ts_event",
            (d_start, d_next),
        )
        if df.empty:
            return df
        # Per-date filter: keep only rows whose symbol matches that day's
        # CME-calendar-resolved active contract.
        df["_stat_date"] = df["ts_event"].dt.date
        expected = df["_stat_date"].map(lambda x: active_contract("NQ", x))
        df = df.loc[df["symbol"] == expected].drop(columns=["_stat_date"]).copy()
        return df

    def _load_vix_1m(
        self,
        d: date,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """VIX 1-minute candles from ``public.tt_candles`` (dxFeed ``VIX{=m}``)."""
        d_next = d + timedelta(days=1)
        start = t_start if t_start is not None else d
        end = t_end if t_end is not None else d_next
        return self._query(
            "SELECT DISTINCT ON (open_dt_full) open_dt_full, close_price "
            "FROM public.tt_candles "
            "WHERE event_symbol = 'VIX{=m}' "
            "  AND open_dt_full >= %s - interval '45 minutes' "
            "  AND open_dt_full < %s "
            "ORDER BY open_dt_full, event_dt_full DESC",
            (start, end),
        )

    def _load_ytd_hl(self, d: date) -> pd.DataFrame:
        """Completed-session high/low from Jan 1 through yesterday."""
        year_start = date(d.year, 1, 1)
        symbols_needed = sorted({
            sym for _, sym in active_contracts_in_range("NQ", year_start, d)
        })
        if not symbols_needed:
            return pd.DataFrame(columns=["date", "high", "low"])
        df_s = self._query(
            "SELECT ts_recv, symbol, stat_type, price, "
            "  CASE "
            "    WHEN stat_type = 3 THEN (ts_ref AT TIME ZONE 'UTC')::date "
            "    ELSE (ts_event AT TIME ZONE 'America/Chicago' + "
            "          CASE WHEN EXTRACT(HOUR FROM (ts_event AT TIME ZONE "
            "               'America/Chicago')) >= 15 "
            "               THEN INTERVAL '1 day' ELSE INTERVAL '0' END)::date "
            "  END AS session_date "
            "FROM databento.statistics "
            "WHERE ts_event >= %s AND ts_event < %s "
            "  AND stat_type IN (4, 5) "
            "  AND symbol = ANY(%s) "
            "ORDER BY ts_recv",
            (year_start, d, symbols_needed),
        )
        if df_s.empty:
            return pd.DataFrame(columns=["date", "high", "low"])
        df_s = df_s.rename(columns={"session_date": "date"})
        df_s = df_s.sort_values("ts_recv")
        pivoted = (
            df_s.groupby(["symbol", "date", "stat_type"])["price"]
            .last()
            .unstack("stat_type")
            .rename(columns={4: "low", 5: "high"})
            .reset_index()
        )
        pivoted.columns.name = None
        for col in ("high", "low"):
            if col not in pivoted.columns:
                pivoted[col] = np.nan
        pivoted["_active"] = pivoted["date"].map(lambda x: active_contract("NQ", x))
        active_rows = pivoted[pivoted["symbol"] == pivoted["_active"]].copy()
        return active_rows[["date", "high", "low"]]

    def _load_candles_1min(
        self,
        d: date,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
        refresh_cagg: bool = True,
    ) -> pd.DataFrame:
        """Load 1-minute candles for the lead-month NQ contract on date `d`.

        *refresh_cagg*: when False (live warm path), skip expensive cagg
        refresh on empty — the continuous aggregate is kept current by normal
        ingestion; live deltas often query a sub-minute window with 0 new 1m bars.
        """
        d_next = d + timedelta(days=1)
        start = t_start if t_start is not None else d
        end = t_end if t_end is not None else d_next
        sql = (
            "SELECT * FROM databento.tbbo_1min "
            "WHERE bucket >= %s AND bucket < %s "
            "AND symbol = %s ORDER BY bucket"
        )
        df = self._query(sql, (start, end, self._primary_nq))
        if df.empty and refresh_cagg:
            # Continuous aggregate may not be materialized for historical dates
            self._refresh_cagg(d, d_next)
            df = self._query(sql, (start, end, self._primary_nq))
        if not df.empty:
            df = df.set_index("bucket").sort_index()
        return df

    def _load_es_candles_1min(
        self,
        d: date,
        *,
        t_start: pd.Timestamp | None = None,
        t_end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load ES 1-minute candles for the lead-month ES contract on date `d`."""
        d_next = d + timedelta(days=1)
        start = t_start if t_start is not None else d
        end = t_end if t_end is not None else d_next
        df = self._query(
            "SELECT * FROM databento.tbbo_1min "
            "WHERE bucket >= %s AND bucket < %s "
            "AND symbol = %s ORDER BY bucket",
            (start, end, self._primary_es),
        )
        if not df.empty:
            df = df.set_index("bucket").sort_index()
        return df

    def _load_daily_ohlc(self, d: date) -> pd.DataFrame:
        """Back-adjusted daily OHLC built from CME statistics + RTH open.

        Data sources
        ------------
        close  — CME official settlement (stat_type=3), keyed by ts_ref date.
        high   — CME session high (stat_type=5), keyed by ts_ref date.
        low    — CME session low  (stat_type=4), keyed by ts_ref date.
        open   — first tbbo_1min ``open`` at/after 8:30 AM CT on *target_date*
                 only (used solely for ``ht_gap_pct``; prior days' open is NaN).

        Session definition: session D spans 4:00 PM CT on D-1 through
        3:00 PM CT on D.  The *latest* statistic record (by ts_recv) within
        each session is used, matching the most-confirmed CME value.
        Only stat_type=3 (Settlement) has a valid ts_ref; stat_type=4,5
        (session H/L) carry sentinel ts_ref values (year 48113+) that would
        overflow psycopg.  A server-side CASE expression computes session_date
        safely for all three types: ts_ref::date for settlement, and
        ts_event-shifted by the 3 PM CT session boundary for H/L.

        Back-adjustment: same cumulative close-delta algorithm as before, now
        using settlement prices at roll boundaries.  H−L ranges are invariant
        to additive shifts, so ``ht_daily_atr_ratio`` is unaffected.
        """
        start = d - timedelta(days=35)
        d_next = d + timedelta(days=1)

        # Identify every contract active anywhere in the lookback window.
        symbols_needed = sorted({
            sym for _, sym in active_contracts_in_range("NQ", start, d)
        })

        # Settlement (3), session low (4), session high (5).
        # Only stat_type=3 has a valid ts_ref; stat_type=4/5 carry sentinel
        # values (year 48113+).  A server-side CASE computes the session date
        # safely for all three types — psycopg only sees the resulting DATE.
        #   stat_type=3: (ts_ref AT TIME ZONE 'UTC')::date (CME reference date)
        #   stat_type=4,5: ts_event in Chicago + 1 day if hour >= 15 (session
        #                  boundary at 3 PM CT)
        df_s = self._query(
            "SELECT ts_recv, symbol, stat_type, price, "
            "  CASE "
            "    WHEN stat_type = 3 THEN (ts_ref AT TIME ZONE 'UTC')::date "
            "    ELSE (ts_event AT TIME ZONE 'America/Chicago' + "
            "          CASE WHEN EXTRACT(HOUR FROM (ts_event AT TIME ZONE 'America/Chicago')) >= 15 "
            "               THEN INTERVAL '1 day' ELSE INTERVAL '0' END)::date "
            "  END AS session_date "
            "FROM databento.statistics "
            "WHERE ts_event >= %s AND ts_event < %s "
            "  AND stat_type IN (3, 4, 5) "
            "  AND symbol = ANY(%s) "
            "ORDER BY ts_recv",
            (start, d_next, symbols_needed),
        )

        if df_s.empty:
            return pd.DataFrame()

        # session_date is a Python date returned directly from PostgreSQL (DATE type).
        df_s = df_s.rename(columns={"session_date": "date"})

        # Latest stat per (symbol, date, stat_type) by ts_recv.
        df_s = df_s.sort_values("ts_recv")
        pivoted = (
            df_s.groupby(["symbol", "date", "stat_type"])["price"]
            .last()
            .unstack("stat_type")
            .rename(columns={3: "close", 4: "low", 5: "high"})
            .reset_index()
        )
        pivoted.columns.name = None
        # A day queried intraday can carry high/low (stat_type 4/5) but not yet the
        # settlement (stat_type 3 → "close", published ~3pm CT); an all-provisional
        # window may lack a stat_type entirely, so unstack won't create that column.
        # Guarantee the OHLC columns exist so downstream selection never KeyErrors —
        # missing values stay NaN (higher_timeframe already tolerates NaN daily rows).
        for _col in ("close", "high", "low"):
            if _col not in pivoted.columns:
                pivoted[_col] = np.nan
        # open is populated only for target_date from tbbo_1min below.
        pivoted["open"] = np.nan

        # Map each date to its calendar-resolved active contract.
        pivoted["_active"] = pivoted["date"].map(lambda x: active_contract("NQ", x))
        active_rows = pivoted[pivoted["symbol"] == pivoted["_active"]].copy()

        # Fallback: fill any missing trading days from any available contract.
        missing_dates = set(pivoted["date"].unique()) - set(active_rows["date"].unique())
        if missing_dates:
            fallback = (
                pivoted[pivoted["date"].isin(missing_dates)]
                .sort_values(["date", "symbol"])
                .drop_duplicates(subset=["date"], keep="first")
            )
            active_rows = pd.concat([active_rows, fallback], ignore_index=True)
        active_rows = active_rows.sort_values("date").reset_index(drop=True)

        # Back-adjustment: cumulative close-delta at each roll boundary.
        # Walk newest→oldest; at a contract change add the settlement-price
        # jump to cumulative_delta before recording the older segment's offset,
        # so the last day of the old contract aligns with the continuous series.
        if active_rows["symbol"].nunique() > 1:
            close_by_sym_date = (
                pivoted.set_index(["symbol", "date"])["close"].sort_index()
            )
            segments = active_rows[["date", "symbol"]].drop_duplicates(
                subset=["date"], keep="first"
            )
            prev_symbol: str | None = None
            cumulative_delta = 0.0
            segment_adjustments: list[tuple[date, float]] = []
            for _, row in segments.iloc[::-1].iterrows():
                if prev_symbol is not None and row["symbol"] != prev_symbol:
                    try:
                        new_close = close_by_sym_date.loc[(prev_symbol, row["date"])]
                        old_close = close_by_sym_date.loc[(row["symbol"], row["date"])]
                        cumulative_delta += float(new_close) - float(old_close)
                    except KeyError:
                        pass
                segment_adjustments.append((row["date"], cumulative_delta))
                prev_symbol = row["symbol"]

            adj_map = dict(segment_adjustments)
            for col in ("open", "high", "low", "close"):
                active_rows[col] = active_rows.apply(
                    lambda r: float(r[col]) + adj_map.get(r["date"], 0.0),
                    axis=1,
                )

        # RTH open: first tbbo_1min candle at/after 8:30 AM CT on target date.
        # Handles CDT (13:30 UTC) and CST (14:30 UTC) automatically.
        rth_open = rth_open_utc(d)
        open_row = self._query(
            "SELECT open FROM databento.tbbo_1min "
            "WHERE bucket >= %s AND bucket < %s AND symbol = %s "
            "ORDER BY bucket LIMIT 1",
            (rth_open, rth_open + pd.Timedelta(minutes=30), self._primary_nq),
        )
        today_open = float(open_row["open"].iloc[0]) if not open_row.empty else np.nan

        # Update the target-date row's open (may already exist from statistics).
        today_mask = active_rows["date"] == d
        if today_mask.any():
            active_rows.loc[today_mask, "open"] = today_open
        else:
            today_row = pd.DataFrame({
                "date": [d], "open": [today_open],
                "high": [np.nan], "low": [np.nan], "close": [np.nan],
            })
            active_rows = pd.concat([active_rows, today_row], ignore_index=True)
            active_rows = active_rows.sort_values("date").reset_index(drop=True)

        return active_rows[["date", "open", "high", "low", "close"]].copy()


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Build or patch feature matrices")
    parser.add_argument("--date", help="Single date (YYYY-MM-DD) or 'today'")
    parser.add_argument("--start", help="Start date for range build")
    parser.add_argument("--end", help="End date for range build")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace the day file (default is patch: fill missing rows/NaNs only)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    engine = FeatureEngine()

    if args.date:
        d = date.today() if args.date == "today" else datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.rebuild:
            engine.build_date(d)
        else:
            engine.patch_date(d)
    elif args.start and args.end:
        engine.build_range(
            datetime.strptime(args.start, "%Y-%m-%d").date(),
            datetime.strptime(args.end, "%Y-%m-%d").date(),
        )
    else:
        parser.error("Provide --date or --start/--end")


if __name__ == "__main__":
    main()
