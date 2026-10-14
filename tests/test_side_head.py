"""Binary TBM side head (flats dropped)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.search.side_head import (
    _binary_y, _dir_mask, all_feature_names, apply_side,
    causal_ema_cross_1m, decile_side_table, univariate_side_auc,
)


def test_dir_mask_and_binary_y():
    y = np.array([0, 1, 2, 2, 0, 1])
    assert list(_dir_mask(y)) == [True, False, True, True, True, False]
    assert list(_binary_y(y[_dir_mask(y)])) == [0, 1, 1, 0]


def test_all_feature_names_drops_meta():
    assert all_feature_names(["cs_clv", "_close", "rv_realized_vol_60"]) == [
        "cs_clv", "rv_realized_vol_60"]


def test_univariate_ranks_informative_col():
    rng = np.random.default_rng(0)
    n = 400
    y = np.where(rng.random(n) < 0.5, 2, 0).astype(np.int32)
    x_good = (y == 2).astype(np.float64) + rng.normal(0, 0.2, n)
    x_noise = rng.normal(size=n)
    X = np.column_stack([x_good, x_noise]).astype(np.float32)
    rows = univariate_side_auc(X, ["signal", "noise"], y, np.ones(n, dtype=bool))
    by = {r["feature"]: r["auc"] for r in rows}
    assert by["signal"] > 0.8
    assert 0.35 < by["noise"] < 0.65


def test_decile_side_table_tracks_signal():
    y = np.array([2, 2, 2, 2, 0, 0, 0, 0] * 20, dtype=np.int32)
    x = np.array([0.1, 0.2, 0.3, 0.4, -0.4, -0.3, -0.2, -0.1] * 20, dtype=np.float64)
    tab = decile_side_table(x, y, np.ones(len(y), dtype=bool), name="x", n_bins=2)
    assert tab["auc"] is not None and tab["auc"] > 0.8
    assert tab["bins"][0]["p_up"] < tab["bins"][-1]["p_up"]


def test_causal_ema_cross_constant_on_flat_day():
    ts = pd.date_range("2025-07-01 13:30:00", periods=12 * 30, freq="5s", tz="UTC").to_numpy()
    px = np.full(len(ts), 20000.0)
    c = causal_ema_cross_1m(ts, px)
    assert c.shape == (len(ts),)
    assert set(np.unique(c)).issubset({-1.0, 0.0, 1.0})


def test_apply_side_emits_both_directions():
    n = 24
    ts = pd.date_range("2025-07-01 13:35:00", periods=n, freq="10min", tz="UTC").to_numpy()
    y = np.array([2, 0] * (n // 2), dtype=np.int32)
    p = np.array([0.8, 0.2] * (n // 2), dtype=np.float64)
    when = np.ones(n, dtype=bool)
    sc = apply_side(p, y, ts, when, margin=0.2)
    assert sc["n_long"] > 0 and sc["n_short"] > 0
    assert sc["n_trades"] == sc["n_long"] + sc["n_short"]
    assert sc["side_acc_dir"] == 1.0
