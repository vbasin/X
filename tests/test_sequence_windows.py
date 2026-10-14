"""Sliding windows + gap mask (session-clip lookback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.sequence_windows import make_windows, window_contiguous_mask


def test_window_contiguous_mask_flags_session_holes():
    t0 = pd.Timestamp("2026-02-12 16:00", tz="America/Chicago")
    morning = pd.date_range(t0, periods=8, freq="15s")
    nxt = pd.Timestamp("2026-02-13 08:30", tz="America/Chicago")
    later = pd.date_range(nxt, periods=8, freq="15s")
    ts = morning.append(later).tz_convert("UTC").to_numpy()
    seq_len = 4
    mask = window_contiguous_mask(ts, seq_len, max_gap_s=45)
    assert mask is not None
    assert mask.dtype == bool
    assert len(mask) == len(ts) - seq_len + 1
    # Window that sits entirely in the first session is ok; one that
    # straddles 16:00 → next 08:30 is not.
    X = np.arange(len(ts), dtype="float64").reshape(-1, 1)
    W = make_windows(X, seq_len)
    assert len(W) == len(mask)
    assert bool(mask[0]) is True
    # last window of first block ends at index 7; first straddling
    # window ends at index 8 (first bar of day 2).
    assert bool(mask[8 - seq_len + 1]) is False
    assert bool(mask[-1]) is True


def test_window_contiguous_mask_all_ok_on_regular_tape():
    ts = pd.date_range("2026-02-13 08:30", periods=20, freq="15s", tz="UTC")
    mask = window_contiguous_mask(ts.to_numpy(), 5, max_gap_s=45)
    assert mask is not None
    assert bool(mask.all()) is True
