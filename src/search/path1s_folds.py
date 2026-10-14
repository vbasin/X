"""Walk-forward folds on the 1s path + path labels (ADR-053 Encoder A/B).

Does **not** touch the 5s engineered matrix. Each session carries the full
ATR-normalised 1s feature path and the entry-window labels joined on ``ts``.
Windows are cut at fit time (see :mod:`src.models.path_windows`) so overnight
boundaries never leak into a TCN window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.data_paths import parquet_path1s_dir, parquet_pathlabels_dir
from src.models.heads import encode_cr_sequence, encode_pathlabel_frame
from src.validation.walk_forward import (
    DEFAULT_PURGE_SECONDS,
    schedule_from_eval,
)
from datetime import date, timedelta

from src.training.path_features import (
    ATR_MIN_PERIODS,
    ATR_SECONDS,
    PATH_FEATURE_COLS,
    path_ohlc_features,
)

logger = logging.getLogger("vtech.search.path1s_folds")


@dataclass
class PathSession:
    """One trading day of 1s features + labelled entries."""

    date: str
    X: np.ndarray          # (T, F) float32
    ts: np.ndarray         # (T,) datetime64[ns]
    entry_idx: np.ndarray  # (E,) int64 into X
    y: np.ndarray
    y_up: np.ndarray
    y_down: np.ndarray
    y_quantile: np.ndarray
    r_value: np.ndarray
    cr_up: np.ndarray | None = None       # (E, 20) competing-risks classes
    cr_down: np.ndarray | None = None
    cr_mask_up: np.ndarray | None = None  # (E, 20) right-censor mask
    cr_mask_down: np.ndarray | None = None
    feature_cols: tuple[str, ...] = PATH_FEATURE_COLS


def _session_dates_on_disk(path_dir: Path, lab_dir: Path, start: str, end: str) -> list[str]:
    paths = {p.stem.removeprefix("path_") for p in path_dir.glob("path_*.parquet")}
    labs = {p.stem.removeprefix("labels_") for p in lab_dir.glob("labels_*.parquet")}
    days = sorted(paths & labs)
    return [d for d in days if start <= d <= end]


def _read_path(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "ts" in raw.columns:
            raw = raw.set_index("ts")
        else:
            raise ValueError(f"{path.name}: path parquet has no DatetimeIndex or ts column")
    raw.index = pd.to_datetime(raw.index, utc=True)
    return raw.sort_index()


def prior_path_warmup(path_dir: Path, day: str, *, seconds: int | None = None) -> pd.DataFrame | None:
    """Tail of the newest earlier ``path_*.parquet`` before *day*.

    Walks back up to a week so a holiday gap still finds the last stored
    session. Rows are the continuous series that exists on disk — no
    synthetic overnight fill.
    """
    need = int(seconds if seconds is not None else ATR_SECONDS + ATR_MIN_PERIODS)
    try:
        d0 = date.fromisoformat(day)
    except ValueError:
        return None
    for back in range(1, 8):
        prev = (d0 - timedelta(days=back)).isoformat()
        path = path_dir / f"path_{prev}.parquet"
        if not path.exists():
            continue
        raw = _read_path(path)
        if raw.empty:
            continue
        return raw.iloc[-need:]
    return None


def load_session(
    day: str,
    *,
    path_dir: Path,
    lab_dir: Path,
    usable_only: bool = True,
    entry_stride: int = 1,
) -> PathSession | None:
    """Load one day. Returns None if the join is empty (holiday / unusable)."""
    raw = _read_path(path_dir / f"path_{day}.parquet")
    feats = path_ohlc_features(raw, warmup=prior_path_warmup(path_dir, day))
    lab = pd.read_parquet(lab_dir / f"labels_{day}.parquet")
    lab = lab.set_index(pd.to_datetime(lab["ts"], utc=True))
    if usable_only and "usable" in lab.columns:
        lab = lab.loc[lab["usable"].to_numpy()]
    if lab.empty:
        return None
    feats.index = pd.to_datetime(feats.index, utc=True)
    joined = feats.join(lab, how="inner")
    joined = joined.dropna(subset=list(PATH_FEATURE_COLS) + ["label"])
    if joined.empty:
        return None
    if entry_stride > 1:
        joined = joined.iloc[:: int(entry_stride)]
        if joined.empty:
            return None
    packed = encode_pathlabel_frame(joined.reset_index(drop=True))
    cr_up, mask_up = encode_cr_sequence(joined["event_up"], joined["bucket_up"])
    cr_dn, mask_dn = encode_cr_sequence(joined["event_down"], joined["bucket_down"])
    # entry positions in the **full** feature grid (not the inner-join slice)
    full_ts = feats.dropna().index
    full_X = feats.loc[full_ts, list(PATH_FEATURE_COLS)].to_numpy(dtype="float32")
    entry_ts = joined.index
    pos = full_ts.get_indexer(entry_ts)
    if (pos < 0).any():
        raise RuntimeError(f"{day}: label ts missing from feature grid")
    r = (joined["r_value"].to_numpy(dtype="float64")
         if "r_value" in joined.columns else np.full(len(joined), np.nan))
    return PathSession(
        date=day,
        X=full_X,
        ts=full_ts.values,
        entry_idx=pos.astype(np.int64),
        y=joined["label"].to_numpy(dtype="int32"),
        y_up=packed["y_up"].astype("int32"),
        y_down=packed["y_down"].astype("int32"),
        y_quantile=packed.get("y_quantile", np.zeros((len(joined), 4), dtype="float32")).astype("float32"),
        r_value=r,
        cr_up=cr_up, cr_down=cr_dn, cr_mask_up=mask_up, cr_mask_down=mask_dn,
    )


def _load_all(
    days: list[str],
    *,
    path_dir: Path,
    lab_dir: Path,
    usable_only: bool,
    entry_stride: int,
) -> list[PathSession]:
    out: list[PathSession] = []
    for i, day in enumerate(days):
        try:
            sess = load_session(
                day, path_dir=path_dir, lab_dir=lab_dir,
                usable_only=usable_only, entry_stride=entry_stride,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("path1s session %s failed: %s", day, exc)
            continue
        if sess is not None:
            out.append(sess)
        if (i + 1) % 50 == 0:
            logger.info("path1s loaded %d/%d days (%d sessions kept)",
                        i + 1, len(days), len(out))
    return out


def _purge_last_session(sessions: list[PathSession], purge_seconds: int) -> list[PathSession]:
    if not sessions or purge_seconds <= 0:
        return sessions
    last = sessions[-1]
    ts_e = pd.to_datetime(last.ts[last.entry_idx], utc=True)
    cutoff = ts_e.max() - pd.Timedelta(seconds=int(purge_seconds))
    # pandas may return a Series or a ndarray depending on Index vs DatetimeIndex.
    keep = np.asarray(ts_e <= cutoff).reshape(-1)
    if keep.all():
        return sessions
    if not keep.any():
        return sessions[:-1]
    idx = np.flatnonzero(keep)
    purged = PathSession(
        date=last.date, X=last.X, ts=last.ts,
        entry_idx=last.entry_idx[idx],
        y=last.y[idx], y_up=last.y_up[idx], y_down=last.y_down[idx],
        y_quantile=last.y_quantile[idx], r_value=last.r_value[idx],
        cr_up=None if last.cr_up is None else last.cr_up[idx],
        cr_down=None if last.cr_down is None else last.cr_down[idx],
        cr_mask_up=None if last.cr_mask_up is None else last.cr_mask_up[idx],
        cr_mask_down=None if last.cr_mask_down is None else last.cr_mask_down[idx],
        feature_cols=last.feature_cols,
    )
    return sessions[:-1] + [purged]


def _in_range(day: str, start: str, end: str) -> bool:
    return start <= day <= end


def _concat_entries(sessions: list[PathSession], field: str) -> np.ndarray:
    if not sessions:
        return np.array([])
    return np.concatenate([getattr(s, field) for s in sessions])


def _fold_arrays(sessions: list[PathSession]) -> dict[str, np.ndarray]:
    """Flat entry-level arrays for uniqueness/KPI; plugin uses ``sessions``."""
    if not sessions:
        return {
            "y": np.zeros(0, dtype=np.int32),
            "y_up": np.zeros(0, dtype=np.int32),
            "y_down": np.zeros(0, dtype=np.int32),
            "y_quantile": np.zeros((0, 4), dtype=np.float32),
            "r_value": np.zeros(0, dtype=np.float64),
            "ts": np.array([], dtype="datetime64[ns]"),
        }
    ts = np.concatenate([s.ts[s.entry_idx] for s in sessions])
    return {
        "y": _concat_entries(sessions, "y"),
        "y_up": _concat_entries(sessions, "y_up"),
        "y_down": _concat_entries(sessions, "y_down"),
        "y_quantile": _concat_entries(sessions, "y_quantile"),
        "r_value": _concat_entries(sessions, "r_value"),
        "ts": ts,
    }


def build_path1s_folds(objective: dict[str, Any]) -> list[dict[str, Any]]:
    """Calendar walk-forward folds of :class:`PathSession` lists."""
    eval_cfg = objective.get("eval") or {}
    dr = eval_cfg.get("date_range") or {}
    start = str(dr.get("start") or "2018-01-02")
    end = str(dr.get("end") or "2026-06-30")
    search = objective.get("search") or {}
    usable_only = bool(search.get("usable_only", True))
    entry_stride = int(search.get("entry_stride") or 5)
    wf = eval_cfg.get("walk_forward") or {}
    purge_seconds = int(wf.get("purge_seconds") or DEFAULT_PURGE_SECONDS)

    path_dir = parquet_path1s_dir()
    lab_dir = parquet_pathlabels_dir()
    days = _session_dates_on_disk(path_dir, lab_dir, start, end)
    if not days:
        logger.warning("path1s folds: no overlapping path/label sessions in %s–%s", start, end)
        return []
    logger.info("path1s folds: loading %d candidate days %s→%s stride=%d usable_only=%s",
                len(days), start, end, entry_stride, usable_only)
    sessions = _load_all(
        days, path_dir=path_dir, lab_dir=lab_dir,
        usable_only=usable_only, entry_stride=entry_stride,
    )
    if not sessions:
        logger.warning("path1s folds: every session empty after usable/join filter")
        return []

    by_date = {s.date: s for s in sessions}
    if str(wf.get("mode") or "") != "expanding_calendar":
        logger.warning("path1s folds require eval.walk_forward.mode=expanding_calendar")
        return []
    schedule = schedule_from_eval(eval_cfg)
    folds: list[dict[str, Any]] = []
    for spec in schedule:
        tr, oof = spec["train"], spec["oof"]
        train_s = [by_date[d] for d in sorted(by_date) if _in_range(d, tr["start"], tr["end"])]
        oof_s = [by_date[d] for d in sorted(by_date) if _in_range(d, oof["start"], oof["end"])]
        train_s = _purge_last_session(train_s, purge_seconds)
        val_s: list[PathSession] = []
        val_spec = spec.get("val")
        if val_spec:
            val_s = [by_date[d] for d in sorted(by_date)
                     if _in_range(d, val_spec["start"], val_spec["end"])]
            cut = str(spec.get("val_cut_from") or "train")
            val_dates = {s.date for s in val_s}
            if cut == "train":
                train_s = [s for s in train_s if s.date not in val_dates]
            else:
                oof_s = [s for s in oof_s if s.date not in val_dates]
        n_tr = sum(len(s.entry_idx) for s in train_s)
        n_oof = sum(len(s.entry_idx) for s in oof_s)
        if n_tr < 20 or n_oof < 1:
            logger.warning("path1s fold %s skipped (train_entries=%d oof_entries=%d)",
                           spec.get("fold"), n_tr, n_oof)
            continue
        tr_flat = _fold_arrays(train_s)
        oof_flat = _fold_arrays(oof_s)
        val_flat = _fold_arrays(val_s)
        folds.append({
            "sessions_train": train_s,
            "sessions_oof": oof_s,
            "sessions_val": val_s,
            "y_train": tr_flat["y"],
            "y_oof": oof_flat["y"],
            "y_val": val_flat["y"],
            "y_up_train": tr_flat["y_up"],
            "y_down_train": tr_flat["y_down"],
            "y_quantile_train": tr_flat["y_quantile"],
            "r_value_train": tr_flat["r_value"],
            "y_up_oof": oof_flat["y_up"],
            "y_down_oof": oof_flat["y_down"],
            "y_quantile_oof": oof_flat["y_quantile"],
            "r_value_oof": oof_flat["r_value"],
            "ts_train": tr_flat["ts"],
            "ts_oof": oof_flat["ts"],
            "feature_cols": list(PATH_FEATURE_COLS),
            "fold": int(spec.get("fold", len(folds))),
            "fold_meta": {
                "fold": int(spec.get("fold", len(folds))),
                "split": "expanding_calendar_path1s",
                "train": {**tr, "n_sessions": len(train_s), "n_entries": n_tr},
                "oof": {**oof, "n_sessions": len(oof_s), "n_entries": n_oof},
                "purge_seconds": purge_seconds,
                "entry_stride": entry_stride,
            },
            # Placeholder 2-D X so catalog subsample/view code no-ops harmlessly
            # if invoked; the plugin reads sessions_* .
            "X_train": np.zeros((n_tr, 1), dtype="float32"),
            "X_oof": np.zeros((n_oof, 1), dtype="float32"),
            "X_val": np.zeros((len(val_flat["y"]), 1), dtype="float32"),
        })
    logger.info("path1s folds: %d non-empty calendar folds", len(folds))
    return folds
