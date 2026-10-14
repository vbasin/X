"""ATR-scaled first-touch TBM (preliminary nq-atr-auction objective).

Barriers at t are ``target_mult * ATR[t]`` and ``stop_mult * ATR[t]``, floored
in points so a quiet bar cannot have a sub-bar stop. Same first-touch rule as
``generate_labels_tbm``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.training.labels import BIG_DOWN, BIG_UP, NO_SIGNAL


def atr_barrier_series(
    atr: pd.Series,
    *,
    target_mult: float,
    stop_mult: float,
    min_target_pts: float = 12.0,
    min_stop_pts: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    a = pd.to_numeric(atr, errors="coerce").to_numpy(dtype="float64")
    tgt = np.maximum(float(target_mult) * a, float(min_target_pts))
    stp = np.maximum(float(stop_mult) * a, float(min_stop_pts))
    tgt = np.where(np.isfinite(tgt), tgt, np.nan)
    stp = np.where(np.isfinite(stp), stp, np.nan)
    return tgt.astype("float64"), stp.astype("float64")


def classify_tbm_atr(
    prices: np.ndarray,
    target_pts: np.ndarray,
    stop_pts: np.ndarray,
    window: int,
) -> np.ndarray:
    """Vectorised first-touch TBM with per-row barriers."""
    n = len(prices)
    win = max(1, int(window))
    sentinel = np.int32(win + 1)
    first_up_tgt = np.full(n, sentinel, dtype=np.int32)
    first_dn_tgt = np.full(n, sentinel, dtype=np.int32)
    first_up_stp = np.full(n, sentinel, dtype=np.int32)
    first_dn_stp = np.full(n, sentinel, dtype=np.int32)
    p = np.asarray(prices, dtype="float64")
    tgt = np.asarray(target_pts, dtype="float64")
    stp = np.asarray(stop_pts, dtype="float64")
    for k in range(1, win + 1):
        if k >= n:
            break
        m = n - k
        diff = p[k:] - p[:m]
        t = tgt[:m]
        s = stp[:m]
        ok = np.isfinite(t) & np.isfinite(s) & np.isfinite(diff)
        for arr, cond in (
            (first_up_tgt, ok & (diff >= t)),
            (first_dn_tgt, ok & (diff <= -t)),
            (first_up_stp, ok & (diff >= s)),
            (first_dn_stp, ok & (diff <= -s)),
        ):
            head = arr[:m]
            hit = (head == sentinel) & cond
            if hit.any():
                head[hit] = k
    label = np.full(n, NO_SIGNAL, dtype=np.int32)
    label[(first_up_tgt <= win) & (first_up_tgt < first_dn_stp)] = BIG_UP
    label[(first_dn_tgt <= win) & (first_dn_tgt < first_up_stp)] = BIG_DOWN
    return label


def first_of_run(label: np.ndarray, session_id: np.ndarray | None = None) -> np.ndarray:
    """Keep BIG_UP / BIG_DOWN only on the first bar of a consecutive run."""
    y = np.asarray(label, dtype=np.int32).reshape(-1)
    out = np.full(y.size, NO_SIGNAL, dtype=np.int32)
    if y.size == 0:
        return out
    prev = np.roll(y, 1)
    prev[0] = NO_SIGNAL
    if session_id is not None:
        sid = np.asarray(session_id)
        same = sid == np.roll(sid, 1)
        same[0] = False
        prev = np.where(same, prev, NO_SIGNAL)
    onset = (y != NO_SIGNAL) & (y != prev)
    out[onset] = y[onset]
    return out


def generate_labels_tbm_atr(
    close: pd.Series,
    atr: pd.Series,
    *,
    target_mult: float,
    stop_mult: float,
    window: int,
    min_target_pts: float = 12.0,
    min_stop_pts: float = 8.0,
    onset_only: bool = False,
) -> pd.DataFrame:
    close = pd.Series(close)
    atr = pd.Series(atr).reindex(close.index)
    tgt, stp = atr_barrier_series(
        atr, target_mult=target_mult, stop_mult=stop_mult,
        min_target_pts=min_target_pts, min_stop_pts=min_stop_pts,
    )
    y = classify_tbm_atr(
        close.to_numpy(dtype="float64"), tgt, stp, window=window,
    )
    if onset_only:
        idx = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
        ct = idx.tz_convert("America/Chicago")
        sid = np.asarray(ct.strftime("%Y-%m-%d"))
        y = first_of_run(y, sid)
    return pd.DataFrame({"label": y}, index=close.index)


def atr_label_params(objective: dict[str, Any]) -> dict[str, Any]:
    lbl = objective.get("labels") or {}
    return {
        "target_mult": float(lbl.get("move_threshold", 2.5)),
        "stop_mult": float(lbl.get("mae_tolerance", 1.5)),
        "atr_feature": str(lbl.get("atr_feature") or "cs_atr_10m"),
        "min_target_pts": float(lbl.get("min_target_pts", 12.0)),
        "min_stop_pts": float(lbl.get("min_stop_pts", 8.0)),
        "onset_only": bool(lbl.get("onset_only", False)),
    }
