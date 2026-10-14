"""Sliding-window helpers for sequence families (TCN, LSTM, CNN, Transformer).

The search fold builder emits 2-D arrays ``(N, F)``. Sequence models need
``(N - seq_len + 1, seq_len, F)``. Feature width ``F`` is whatever the objective
view assigned — never a hardcoded catalog ``n_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert 2-D ``(N, F)`` → 3-D ``(N-seq_len+1, seq_len, F)``.

    Already 3-D arrays are returned unchanged.
    """
    if X.ndim == 3:
        return X
    if X.ndim != 2:
        raise ValueError(f"make_windows expected 2-D or 3-D array, got shape {X.shape}")
    n, _f = X.shape
    seq_len = int(seq_len)
    if n < seq_len:
        raise ValueError(
            f"make_windows: only {n} rows but seq_len={seq_len}. "
            "Reduce seq_len or increase the training window.")
    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
    return X[idx]


def window_contiguous_mask(
    ts: np.ndarray | None,
    seq_len: int,
    *,
    max_gap_s: float = 45.0,
) -> np.ndarray | None:
    """True for windows whose consecutive bars never jump more than *max_gap_s*.

    ``make_windows`` treats the 2-D matrix as a continuous tape. A session
    clip that concatenates 08:30–11:00 mornings (ADR-099) leaves ~21.5 h
    holes that look like 15-second steps and poison LSTM lookback. Mask
    those windows out of the loss.
    """
    if ts is None:
        return None
    seq_len = int(seq_len)
    t = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    n = len(t)
    if n < seq_len:
        return np.zeros(0, dtype=bool)
    delta = np.diff(t.asi8) / 1e9
    bad = delta > float(max_gap_s)
    if not bad.any():
        return np.ones(n - seq_len + 1, dtype=bool)
    # window ending at i (label row i) uses bars [i-seq_len+1, i]
    # gaps inside: delta[i-seq_len+1 : i]
    c = np.concatenate([[0], np.cumsum(bad.astype("int32"))])
    end = np.arange(seq_len - 1, n)
    start = end - (seq_len - 1)
    # gaps in [start, end) → c[end] - c[start]
    n_bad = c[end] - c[start]
    return n_bad == 0


def trim_warmup(arr: np.ndarray | None, seq_len: int) -> np.ndarray | None:
    """Drop the first ``seq_len-1`` rows so labels/timestamps match ``make_windows``."""
    if arr is None:
        return None
    seq_len = int(seq_len)
    if len(arr) < seq_len:
        return arr[:0]
    return arr[seq_len - 1:]
