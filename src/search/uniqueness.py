"""Uniqueness for overlapping TBM labels (train subsample + inference cooldown).

5-second bars with a 10-minute triple-barrier window are not i.i.d.: adjacent
labels share almost the same forward path. Train keeps at most one row per
``uniqueness_bars``. Scoring a live book must also space *trades*
(``apply_trade_cooldown``, ADR-047) or n and MDD count the same path many times.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.search.uniqueness")

_BAR_SECONDS = 5


def keep_mask(ts: np.ndarray, uniqueness_bars: int, *, bar_seconds: int = _BAR_SECONDS) -> np.ndarray:
    """True where a TRAIN row is kept. *ts* must be monotonic in time."""
    n = len(ts)
    bars = max(1, int(uniqueness_bars))
    if n == 0:
        return np.zeros(0, dtype=bool)
    gap_ns = int(bars * bar_seconds) * 1_000_000_000
    t = pd.to_datetime(ts, utc=True).asi8
    keep = np.zeros(n, dtype=bool)
    last = t[0] - gap_ns
    for i in range(n):
        if t[i] - last >= gap_ns:
            keep[i] = True
            last = t[i]
    return keep


def apply_trade_cooldown(ts: np.ndarray, uniqueness_bars: int = 120) -> np.ndarray:
    """Inference-time uniqueness: keep trades ≥ *uniqueness_bars* apart.

    Train uniqueness (``apply_train_uniqueness``) thins *fit* rows. Scoring
    every overlapping 5s OOF bar still double-counts the same 10-minute TBM
    path. Apply this to the proposed trade timestamps, in time order.
    """
    return keep_mask(ts, uniqueness_bars)


def apply_train_uniqueness(
    folds: list[dict[str, Any]],
    uniqueness_bars: int,
    *,
    bar_seconds: int = _BAR_SECONDS,
) -> list[dict[str, Any]]:
    """Subsample each fold's ``X_train`` / ``y_train`` (and ``ts_train`` if present)."""
    bars = max(1, int(uniqueness_bars))
    out: list[dict[str, Any]] = []
    for f in folds:
        g = dict(f)
        ts = f.get("ts_train")
        n = len(f["y_train"])
        if ts is None or len(ts) != n:
            # Regular 5s grid fallback (ignores session gaps).
            idx = np.arange(0, n, bars)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
        else:
            mask = keep_mask(ts, bars, bar_seconds=bar_seconds)
        n_keep = int(mask.sum())
        logger.info(
            "train_uniqueness bars=%d fold_train %d → %d (%.1f%%)",
            bars, n, n_keep, 100.0 * n_keep / max(n, 1),
        )
        if n_keep < 8:
            logger.warning("train_uniqueness kept %d rows — leaving fold train unchanged", n_keep)
            out.append(g)
            continue
        out.append(slice_train_rows(f, mask))
    return out


def onset_keep_mask(
    ts: np.ndarray,
    y: np.ndarray,
    duration_s: np.ndarray | None,
    *,
    horizon_s: float = 600.0,
) -> np.ndarray:
    """Keep the first TBM event, then skip until that path hits target/stop/time.

    Flats after the onset are dropped until *duration_s* (seconds) elapses.
    The next kept directional bar is a new onset. OOF is not thinned.
    """
    from src.models.dual_binary import as_dual_binary_y

    n = len(ts)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    yy = as_dual_binary_y(y)
    t = pd.to_datetime(ts, utc=True).asi8
    dur = (np.full(n, np.nan, dtype="float64")
           if duration_s is None else np.asarray(duration_s, dtype="float64"))
    blocked_until = t[0] - 1
    hor = float(horizon_s) if horizon_s and horizon_s > 0 else 600.0
    for i in range(n):
        if t[i] < blocked_until:
            keep[i] = False
            continue
        event = (yy[i, 0] >= 0.5) or (yy[i, 1] >= 0.5)
        if not event:
            continue
        d = dur[i] if i < dur.size else np.nan
        if not np.isfinite(d) or d <= 0:
            d = hor
        blocked_until = t[i] + int(d * 1_000_000_000)
    return keep


def _onset_duration_s(fold: dict[str, Any], y: np.ndarray) -> np.ndarray | None:
    from src.training.aux_labels import AUX_LABEL_COLS

    aux = fold.get("y_aux_train")
    if aux is None or not getattr(aux, "size", 0):
        return None
    a = np.asarray(aux)
    if a.ndim != 2 or a.shape[0] != len(y):
        return None
    col = {c: i for i, c in enumerate(AUX_LABEL_COLS)}
    i_l, i_s = col.get("Time_to_TGT_long"), col.get("Time_to_TGT_short")
    if i_l is None or i_s is None or a.shape[1] <= max(i_l, i_s):
        return None
    from src.models.dual_binary import as_dual_binary_y
    yy = as_dual_binary_y(y)
    dur = np.full(len(yy), np.nan, dtype="float64")
    long_e = yy[:, 0] >= 0.5
    short_e = yy[:, 1] >= 0.5
    dur[long_e] = a[long_e, i_l]
    dur[short_e] = a[short_e, i_s]
    return dur


def apply_train_onset_uniqueness(
    folds: list[dict[str, Any]],
    *,
    horizon_s: float = 600.0,
    drop_rows: bool = True,
) -> list[dict[str, Any]]:
    """Train-only: one labelled onset per TBM path (ADR-099).

    *drop_rows* True (trees) slices train arrays. False (LSTM) writes
    ``w_onset_train`` (1 = keep, 0 = skip) so sequences stay contiguous.
    """
    out: list[dict[str, Any]] = []
    for f in folds:
        g = dict(f)
        y = f["y_train"]
        ts = f.get("ts_train")
        n = len(y)
        if ts is None or len(ts) != n:
            logger.warning("train_onset_uniqueness: missing ts_train — skip fold")
            out.append(g)
            continue
        dur = _onset_duration_s(f, y)
        mask = onset_keep_mask(ts, y, dur, horizon_s=horizon_s)
        n_keep = int(mask.sum())
        logger.info(
            "train_onset_uniqueness fold_train %d → %d (%.1f%%) drop_rows=%s",
            n, n_keep, 100.0 * n_keep / max(n, 1), drop_rows,
        )
        g["w_onset_train"] = mask.astype("float64")
        if drop_rows:
            if n_keep < 8:
                logger.warning("train_onset_uniqueness kept %d — leaving fold train", n_keep)
                out.append(g)
                continue
            g = slice_train_rows(g, mask)
            g["w_onset_train"] = np.ones(n_keep, dtype="float64")
        out.append(g)
    return out


def slice_train_rows(fold: dict[str, Any], selector: np.ndarray) -> dict[str, Any]:
    """Index every train-aligned ndarray (``*_train`` of length ``len(y_train)``).

    Used by uniqueness and the search row-cap so hazard extras (`y_up_train`,
    `r_value_train`, …) stay aligned with ``X_train`` / ``y_train``.
    """
    n = len(fold["y_train"])
    g = dict(fold)
    for k, v in fold.items():
        if not isinstance(v, np.ndarray):
            continue
        if not k.endswith("_train") or len(v) != n:
            continue
        g[k] = v[selector]
    return g
