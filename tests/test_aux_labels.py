"""Path-quality auxiliary labels (ADR-082)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.aux_labels import (
    AUX_LABEL_COLS, LONG_AUX_MAP, SHORT_AUX_MAP, SIDE_RANK_NAME,
    STOP_FIRST_LONG, STOP_FIRST_SHORT,
    aux_from_objective, aux_sample_weight, compute_aux_labels,
    near_miss_sample_weight, pack_aux_outputs, path_side_rank_target, zscore_fit,
)
from src.training.dual_binary_15s import tbm_dual_binary_15s


def _series(vals, freq="5s"):
    idx = pd.date_range("2026-01-05 14:00", periods=len(vals), freq=freq, tz="UTC")
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def test_mfe_mae_and_time_to_tgt_long():
    # Entry 100; dip 5 then rally +60. Horizon 20 bars × 5s = 100s, threshold 50.
    prices = _series([100, 98, 95, 96, 160] + [160] * 20)
    aux = compute_aux_labels(prices, move_threshold=50.0, horizon_s=100, src_step_s=5)
    row = aux.iloc[0]
    assert row["MFE_long_10m"] == pytest.approx(60.0)
    assert row["MAE_b4_TGTorMFE_long_10m"] == pytest.approx(5.0)
    assert row["Time_to_TGT_long"] == pytest.approx(20.0)  # 4 bars × 5s
    assert row["MFE_short_10m"] == pytest.approx(5.0)
    assert np.isnan(row["Time_to_TGT_short"])  # no short target → NaN, not horizon
    assert row["MAE_before_target_or_MFE"] == pytest.approx(5.0)
    packed = pack_aux_outputs(aux.to_numpy(dtype="float32"))
    w = aux_sample_weight(aux.to_numpy(dtype="float32"))
    assert w["Time_to_TGT_short"][0] == 0.0
    assert w["Time_to_TGT_long"][0] == 1.0
    assert packed["Time_to_TGT_short"][0, 0] == 0.0


def test_time_to_tgt_timeout_is_never_horizon():
    prices = _series([100.0] * 150)
    aux = compute_aux_labels(prices, move_threshold=50.0, horizon_s=600, src_step_s=5)
    assert aux["Time_to_TGT_long"].isna().all()
    assert aux["Time_to_TGT_short"].isna().all()
    assert not (aux["Time_to_TGT_long"] == 600).any()


def test_sanitize_stale_600_timeout_not_hit():
    from src.training.aux_labels import sanitize_time_to_tgt
    from src.training.labels import BIG_DOWN, BIG_UP

    n = 4
    idx = pd.date_range("2026-01-05 14:00", periods=n, freq="15s", tz="UTC")
    df = pd.DataFrame({
        "y_long": [1, 0, 0, 1],
        "y_short": [0, 1, 0, 0],
        "Time_to_TGT_long": [280.0, 600.0, 600.0, 600.0],
        "Time_to_TGT_short": [600.0, 90.0, 600.0, np.nan],
    }, index=idx)
    out = sanitize_time_to_tgt(df, horizon_s=600)
    assert out["Time_to_TGT_long"].iloc[0] == pytest.approx(280.0)
    assert np.isnan(out["Time_to_TGT_long"].iloc[1])
    assert np.isnan(out["Time_to_TGT_long"].iloc[2])
    assert out["Time_to_TGT_long"].iloc[3] == pytest.approx(600.0)  # last-bar hit
    assert np.isnan(out["Time_to_TGT_short"].iloc[0])
    assert out["Time_to_TGT_short"].iloc[1] == pytest.approx(90.0)
    assert np.isnan(out["Time_to_TGT_short"].iloc[2])
    lab = pd.DataFrame({
        "label": [BIG_UP, BIG_DOWN, 1, BIG_UP],
        "Time_to_TGT_long": [600.0, 600.0, 600.0, 250.0],
        "Time_to_TGT_short": [600.0, 40.0, 600.0, 600.0],
    }, index=idx)
    s = sanitize_time_to_tgt(lab, horizon_s=600)
    assert s["Time_to_TGT_long"].iloc[0] == pytest.approx(600.0)
    assert np.isnan(s["Time_to_TGT_long"].iloc[1])
    assert np.isnan(s["Time_to_TGT_long"].iloc[2])
    assert s["Time_to_TGT_short"].iloc[1] == pytest.approx(40.0)
    assert np.isnan(s["Time_to_TGT_short"].iloc[0])


def test_forward_returns_are_point_deltas():
    prices = _series(list(range(100, 230)))  # +1 pt / 5s, 130 bars
    aux = compute_aux_labels(prices, move_threshold=50.0, horizon_s=600, src_step_s=5)
    assert aux["forward_return_1m"].iloc[0] == pytest.approx(12.0)
    assert aux["forward_return_5m"].iloc[0] == pytest.approx(60.0)
    assert aux["forward_return_10m"].iloc[0] == pytest.approx(120.0)


def test_15s_dual_binary_carries_aux():
    prices = _series([100, 120, 150, 160, 100, 60, 20, 10, 0] + [0] * 40)
    out = tbm_dual_binary_15s(prices, move_threshold=50.0, mae_tolerance=6.0, horizon_s=100)
    assert "y_long" in out.columns and "y_short" in out.columns
    for c in AUX_LABEL_COLS:
        assert c in out.columns
    assert int(out.iloc[0]["y_long"]) == 1
    assert float(out.iloc[0]["MFE_long_10m"]) > 0


def test_aux_from_objective_is_opt_in():
    assert aux_from_objective({}) is None
    assert aux_from_objective({"labels": {"encoding": "dual_binary"}}) is None
    assert aux_from_objective({"labels": {"aux": []}}) is None
    full = aux_from_objective({"labels": {"aux": True}})
    assert full == AUX_LABEL_COLS
    subset = aux_from_objective({"labels": {"aux": ["MFE_long_10m", "Time_to_TGT_short"]}})
    assert subset == ("MFE_long_10m", "Time_to_TGT_short")
    with pytest.raises(ValueError, match="unknown"):
        aux_from_objective({"labels": {"aux": ["not_a_col"]}})


def test_pack_aux_zscore_and_nan_weight():
    n = 8
    aux = np.zeros((n, len(AUX_LABEL_COLS)), dtype="float32")
    aux[:, 0] = np.arange(n, dtype="float32")
    aux[0, 0] = np.nan
    mu, sd = zscore_fit(aux)
    packed = pack_aux_outputs(aux, mu=mu, sd=sd)
    assert packed["MFE_long_10m"].shape == (n, 1)
    only = pack_aux_outputs(aux, mu=mu, sd=sd, columns=("MFE_long_10m",))
    assert set(only) == {"MFE_long_10m", SIDE_RANK_NAME}
    assert packed["MFE_long_10m"][0, 0] == 0.0
    w = aux_sample_weight(aux)
    assert w["MFE_long_10m"][0] == 0.0
    assert w["MFE_long_10m"][1] == 1.0
    assert SIDE_RANK_NAME in w
    assert "long_MAE_before_target_or_MFE" not in packed
    assert "short_MAE_before_target_or_MFE" not in packed


def test_pack_short_forward_return_is_negated():
    n = 6
    aux = np.zeros((n, len(AUX_LABEL_COLS)), dtype="float32")
    i = AUX_LABEL_COLS.index("forward_return_1m")
    aux[:, i] = np.array([10, -4, 0, 8, -12, 3], dtype="float32")
    mu, sd = zscore_fit(aux)
    packed = pack_aux_outputs(aux, mu=mu, sd=sd)
    np.testing.assert_allclose(
        packed["short_forward_return_1m"],
        -packed["long_forward_return_1m"],
    )
    assert dict((k, s) for k, _, s in LONG_AUX_MAP)["long_forward_return_1m"] == 1.0
    assert dict((k, s) for k, _, s in SHORT_AUX_MAP)["short_forward_return_1m"] == -1.0


def test_path_side_rank_target_mfe_winner():
    n = 4
    aux = np.zeros((n, len(AUX_LABEL_COLS)), dtype="float32")
    i_l = AUX_LABEL_COLS.index("MFE_long_10m")
    i_s = AUX_LABEL_COLS.index("MFE_short_10m")
    aux[:, i_l] = np.array([60.0, 5.0, 10.0, np.nan], dtype="float32")
    aux[:, i_s] = np.array([10.0, 40.0, 10.0, 3.0], dtype="float32")
    y, w = path_side_rank_target(aux)
    np.testing.assert_array_equal(y.reshape(-1), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(w, [1.0, 35.0 / 50.0, 0.0, 0.0])
    packed = pack_aux_outputs(aux)
    np.testing.assert_array_equal(packed[SIDE_RANK_NAME], y)
    ww = aux_sample_weight(aux)
    np.testing.assert_allclose(ww[SIDE_RANK_NAME], w)


def test_pack_stop_first_from_mae():
    n = 4
    aux = np.zeros((n, len(AUX_LABEL_COLS)), dtype="float32")
    i = AUX_LABEL_COLS.index("MAE_b4_TGTorMFE_long_10m")
    j = AUX_LABEL_COLS.index("MAE_b4_TGTorMFE_short_10m")
    aux[:, i] = np.array([6.0, 5.9, 12.0, np.nan], dtype="float32")
    aux[:, j] = np.array([0.0, 6.0, 1.0, 20.0], dtype="float32")
    packed = pack_aux_outputs(aux)
    np.testing.assert_array_equal(packed[STOP_FIRST_LONG].reshape(-1), [1, 0, 1, 0])
    np.testing.assert_array_equal(packed[STOP_FIRST_SHORT].reshape(-1), [0, 1, 0, 1])
    slim = pack_aux_outputs(aux, columns=("MFE_long_10m",))
    assert STOP_FIRST_LONG not in slim


def test_near_miss_upweights_almost_50():
    n = 4
    y = np.array([1, 0, 0, 0], dtype="int32")
    aux = np.zeros((n, len(AUX_LABEL_COLS)), dtype="float32")
    aux[:, AUX_LABEL_COLS.index("MFE_long_10m")] = [50.0, 40.0, 2.0, 0.0]
    w = near_miss_sample_weight(y, aux, side="long")
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(1.8)
    assert w[2] == pytest.approx(1.04)
