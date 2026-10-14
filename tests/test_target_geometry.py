"""Per-head target-geometry features (ADR-082)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.target_geometry import (
    HEAD_LONG_COLS, HEAD_SHORT_COLS, TARGET_GEOMETRY_COLS,
    attach_target_geometry, compute_target_geometry, high_low_from_cs,
)
from src.models.dual_binary import head_feature_spec, side_feature_index, split_head_columns


def test_high_low_from_clv_roundtrip():
    close = np.array([105.0])
    high, low = high_low_from_cs(close, np.array([20.0]), np.array([0.5]))
    # H = 105 + 0.5*20*(1-0.5) = 110; L = 105 - 0.5*20*(1.5) = 90
    assert high[0] == pytest.approx(110.0)
    assert low[0] == pytest.approx(90.0)


def test_flat_bar_high_equals_close():
    high, low = high_low_from_cs(np.array([100.0]), np.array([0.0]), np.array([np.nan]))
    assert high[0] == pytest.approx(100.0)
    assert low[0] == pytest.approx(100.0)


def test_signed_target_over_close_and_atr():
    idx = pd.date_range("2026-01-05 14:00", periods=40, freq="15s", tz="UTC")
    # Constant 20-pt range, close 20_000. ATR of 10m bars ≈ 20 after warmup.
    df = pd.DataFrame({
        "cs_close": np.full(40, 20000.0),
        "cs_range": np.full(40, 20.0),
        "cs_clv": np.zeros(40),
    }, index=idx)
    geo = compute_target_geometry(df, move_threshold=50.0)
    assert list(geo.columns) == list(TARGET_GEOMETRY_COLS)
    # 50 / 20000 = 0.0025
    assert geo["lg_target_v_close"].iloc[-1] == pytest.approx(50.0 / 20000.0)
    assert geo["st_target_v_close"].iloc[-1] == pytest.approx(-50.0 / 20000.0)
    # ATR may still be warming; when finite, short is exact negation of long
    ok = np.isfinite(geo["lg_target_v_atr"].to_numpy())
    if ok.any():
        np.testing.assert_allclose(
            geo.loc[ok, "st_target_v_atr"], -geo.loc[ok, "lg_target_v_atr"],
        )


def test_attach_is_idempotent():
    idx = pd.date_range("2026-01-05 14:00", periods=8, freq="15s", tz="UTC")
    df = pd.DataFrame({
        "cs_close": np.full(8, 18000.0),
        "cs_range": np.full(8, 5.0),
        "cs_clv": np.zeros(8),
        "of_ofi": np.ones(8),
    }, index=idx)
    once = attach_target_geometry(df)
    twice = attach_target_geometry(once)
    for c in TARGET_GEOMETRY_COLS:
        np.testing.assert_array_equal(once[c].to_numpy(), twice[c].to_numpy())


def test_head_feature_spec_puts_htf_in_last_bar_ctx():
    cols = ["of_ofi", "cs_close", "h1_tr_gap", "d1_zz_signal", *HEAD_LONG_COLS[:1], *HEAD_SHORT_COLS[:1]]
    spec = head_feature_spec(cols)
    shared_names = [cols[i] for i in spec["shared"]]
    long_names = [cols[i] for i in spec["long"]]
    short_names = [cols[i] for i in spec["short"]]
    assert "of_ofi" in shared_names and "cs_close" in shared_names
    assert "h1_tr_gap" not in shared_names
    assert "d1_zz_signal" not in shared_names
    assert "h1_tr_gap" in long_names and "h1_tr_gap" in short_names
    assert "d1_zz_signal" in long_names and "d1_zz_signal" in short_names


def test_head_feature_split_hides_opposite_side():
    cols = ["of_ofi", "cs_close", *HEAD_LONG_COLS, *HEAD_SHORT_COLS]
    spec = head_feature_spec(cols)
    assert spec["shared"] == [0, 1]
    X = np.arange(12, dtype="float32").reshape(2, 6)
    xs, xl, xh = split_head_columns(X, spec)
    assert xs.shape == (2, 2)
    assert xl.shape == (2, 2) and xh.shape == (2, 2)
    li = side_feature_index(cols, "long")
    si = side_feature_index(cols, "short")
    long_names = [cols[i] for i in li]
    short_names = [cols[i] for i in si]
    assert "st_target_v_atr" not in long_names
    assert "lg_target_v_atr" not in short_names
    assert "of_ofi" in long_names and "of_ofi" in short_names
