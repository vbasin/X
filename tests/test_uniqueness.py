"""Train uniqueness subsample (overlapping TBM windows)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.search.uniqueness import (
    apply_trade_cooldown, apply_train_onset_uniqueness, apply_train_uniqueness,
    keep_mask, onset_keep_mask,
)


def test_keep_mask_respects_gap():
    ts = pd.date_range("2026-01-01", periods=20, freq="5s", tz="UTC").to_numpy()
    mask = keep_mask(ts, uniqueness_bars=4)  # 20s
    kept = np.where(mask)[0]
    assert kept[0] == 0
    assert np.all(np.diff(kept) >= 4)


def test_apply_trade_cooldown_is_keep_mask():
    ts = pd.date_range("2026-01-01", periods=20, freq="5s", tz="UTC").to_numpy()
    assert np.array_equal(apply_trade_cooldown(ts, 4), keep_mask(ts, 4))


def test_apply_train_uniqueness_leaves_oof():
    n = 100
    ts = pd.date_range("2026-01-01", periods=n, freq="5s", tz="UTC").to_numpy()
    folds = [{
        "X_train": np.arange(n * 2, dtype="float32").reshape(n, 2),
        "y_train": np.zeros(n, dtype="int32"),
        "ts_train": ts,
        "X_oof": np.ones((n, 2), dtype="float32"),
        "y_oof": np.ones(n, dtype="int32"),
    }]
    out = apply_train_uniqueness(folds, 10)
    assert len(out[0]["X_train"]) == 10
    assert len(out[0]["X_oof"]) == 100


def test_onset_keep_mask_skips_until_duration():
    ts = pd.date_range("2026-01-05 14:30", periods=20, freq="15s", tz="UTC").to_numpy()
    y = np.zeros((20, 2), dtype="float32")
    y[2, 0] = 1.0
    y[3, 0] = 1.0
    y[4, 0] = 1.0
    y[12, 1] = 1.0
    dur = np.full(20, np.nan)
    dur[2] = 45.0  # skip 3 bars of 15s
    mask = onset_keep_mask(ts, y, dur, horizon_s=600)
    assert mask[2]
    assert not mask[3] and not mask[4]
    assert mask[5]
    assert mask[12]


def test_apply_train_onset_uniqueness_lstm_keeps_rows():
    n = 20
    ts = pd.date_range("2026-01-05 14:30", periods=n, freq="15s", tz="UTC").to_numpy()
    y = np.zeros((n, 2), dtype="float32")
    y[1, 0] = 1.0
    y[2, 0] = 1.0
    folds = [{
        "X_train": np.arange(n * 2, dtype="float32").reshape(n, 2),
        "y_train": y,
        "ts_train": ts,
        "X_oof": np.ones((n, 2), dtype="float32"),
        "y_oof": y,
    }]
    kept = apply_train_onset_uniqueness(folds, horizon_s=30.0, drop_rows=False)
    assert len(kept[0]["X_train"]) == n
    assert kept[0]["w_onset_train"][1] == 1.0
    assert kept[0]["w_onset_train"][2] == 0.0
    dropped = apply_train_onset_uniqueness(folds, horizon_s=30.0, drop_rows=True)
    assert len(dropped[0]["X_train"]) < n
    assert len(dropped[0]["X_oof"]) == n
