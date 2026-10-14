"""Shared serving bundle load + predict_canonical (Live/Backtest SSOT)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.serving import direction_from_proba, is_serving_bundle, predict_canonical


def test_direction_from_proba_scalar():
    assert direction_from_proba(0.1, 0.2, 0.9, 0.5) == ("up", 0.9)
    assert direction_from_proba(0.9, 0.05, 0.05, 0.5) == ("down", 0.9)
    assert direction_from_proba(0.3, 0.5, 0.2, 0.5) == ("flat", 0.5)
    # tie at threshold: up wins when equal to down and both ≥ thr
    assert direction_from_proba(0.6, 0.1, 0.6, 0.5) == ("up", 0.6)
    # down needs strictly greater than up
    assert direction_from_proba(0.6, 0.1, 0.5, 0.5) == ("down", 0.6)


def test_direction_from_proba_vector():
    d, c = direction_from_proba(
        np.array([0.9, 0.1, 0.2]),
        np.array([0.05, 0.5, 0.1]),
        np.array([0.05, 0.4, 0.7]),
        0.5,
    )
    assert list(d) == ["down", "flat", "up"]
    assert list(c) == pytest.approx([0.9, 0.5, 0.7])


def test_predict_canonical_mock_bundle():
    class _Booster:
        def predict_proba(self, X):
            n = len(X)
            # down / flat / up
            out = np.zeros((n, 3))
            out[:, 1] = 0.5
            out[0, 0], out[0, 1] = 0.8, 0.1
            out[1, 2], out[1, 1] = 0.85, 0.1
            return out

    idx = pd.date_range("2026-07-15", periods=3, freq="5s", tz="UTC")
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 0.0, 0.0]}, index=idx)
    bundle = {
        "feature_cols": ["f1", "f2"],
        "booster": _Booster(),
        "scaler": None,
        "dir_thr": 0.5,
    }
    pred = predict_canonical(df, bundle)
    assert list(pred.columns) == [
        "ts", "direction", "confidence", "p_down", "p_flat", "p_up",
    ]
    assert list(pred["direction"]) == ["down", "up", "flat"]
    assert pred["confidence"].iloc[0] == pytest.approx(0.8)
    assert pred["confidence"].iloc[1] == pytest.approx(0.85)
    assert pred["confidence"].iloc[2] == pytest.approx(0.5)


def test_predict_primary_meta_keeps_primary_confidence():
    class _Primary:
        def predict_proba(self, X):
            n = len(X)
            out = np.zeros((n, 3))
            out[:, 2] = 0.91
            out[:, 1] = 0.05
            out[:, 0] = 0.04
            return out

    idx = pd.date_range("2026-07-15", periods=4, freq="5s", tz="UTC")
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    bundle = {
        "primary_meta": True,
        "feature_cols": ["f1"],
        "bases": [{
            "family": "xgboost", "kind": "tabular",
            "artifact": _Primary(), "scaler": None, "feature_cols": ["f1"],
        }],
        "meta_estimator": {"kind": "constant", "estimator": 1.0},
        "meta": {
            "topology": "primary_meta",
            "feature_names": ["primary_confidence", "is_up"],
            "approve_threshold": 0.5,
            "min_primary_confidence": 0.5,
            "impute_values": {},
        },
    }
    pred = predict_canonical(df, bundle)
    assert list(pred["direction"]) == ["up"] * 4
    assert pred["confidence"].tolist() == pytest.approx([0.91] * 4)

    bundle["meta_estimator"] = {"kind": "constant", "estimator": 0.0}
    pred2 = predict_canonical(df, bundle)
    assert list(pred2["direction"]) == ["flat"] * 4
    assert pred2["confidence"].tolist() == pytest.approx([0.0] * 4)


def test_predict_canonical_dual_binary_pair():
    class _Pair:
        def predict_pair(self, X):
            n = len(X)
            p_long = np.zeros(n)
            p_short = np.zeros(n)
            p_long[0] = 0.8
            p_short[1] = 0.9
            p_long[2] = 0.4
            p_short[2] = 0.3
            return p_long, p_short

    idx = pd.date_range("2026-07-15", periods=3, freq="15s", tz="UTC")
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=idx)
    pred = predict_canonical(df, {
        "feature_cols": ["f1"],
        "booster": _Pair(),
        "scaler": None,
        "dir_thr": 0.5,  # ignored for dual-binary (ADR-089)
    })
    assert list(pred["direction"]) == ["up", "down", "up"]
    assert pred["p_long"].iloc[0] == pytest.approx(0.8)
    assert pred["p_short"].iloc[1] == pytest.approx(0.9)
    gated = predict_canonical(df, {
        "feature_cols": ["f1"],
        "booster": _Pair(),
        "scaler": None,
        "dir_thr": 0.5,
        "dual_binary_fire": 0.5,
    })
    assert list(gated["direction"]) == ["up", "down", "flat"]
    argmax = predict_canonical(df, {
        "feature_cols": ["f1"],
        "booster": _Pair(),
        "scaler": None,
        "dual_binary_fire": 0.0,
    })
    assert list(argmax["direction"]) == ["up", "down", "up"]


def test_predict_sequence_duplicate_timestamps():
    """Live next-session spill duplicates a ts; loc-by-index used to 400 Backtest."""
    class _Seq:
        def predict(self, X, verbose=0):
            n = len(X)
            out = np.zeros((n, 3), dtype="float64")
            out[:, 1] = 0.7
            out[:, 2] = 0.2
            out[:, 0] = 0.1
            return out

    idx = pd.to_datetime([
        "2026-08-25 19:44:50+00:00",
        "2026-08-26 13:02:00+00:00",
        "2026-08-26 13:02:00+00:00",
        "2026-08-26 13:02:05+00:00",
        "2026-08-26 13:02:10+00:00",
        "2026-08-26 13:02:15+00:00",
    ])
    df = pd.DataFrame({"f1": np.arange(6, dtype="float64")}, index=idx)
    bundle = {
        "family": "attention_lstm",
        "kind": "sequence",
        "feature_cols": ["f1"],
        "booster": _Seq(),
        "scaler": None,
        "seq_len": 3,
        "dir_thr": 0.5,
    }
    pred = predict_canonical(df, bundle)
    assert len(pred) == 6
    assert list(pred["direction"].iloc[:2]) == ["flat", "flat"]
    assert list(pred["direction"].iloc[2:]) == ["flat"] * 4
    assert pred["p_flat"].iloc[2] == pytest.approx(0.7)


def test_predict_canonical_lstm_head_context():
    """ADR-082 3-input LSTM: serving must pack input + long_ctx + short_ctx."""
    from tensorflow import keras
    from src.models.attention_lstm import build_attention_lstm

    keras.backend.clear_session()
    model = build_attention_lstm(
        (4, 3), lstm_units=[4], attention_heads=0, dense_units=[4],
        dropout=0.0, head="dual_binary", n_long_ctx=2, n_short_ctx=2,
    )
    cols = [
        "a", "b", "c",
        "lg_target_v_atr", "lg_target_v_close",
        "st_target_v_atr", "st_target_v_close",
    ]
    n = 10
    idx = pd.date_range("2026-07-15", periods=n, freq="15s", tz="UTC")
    df = pd.DataFrame({c: np.arange(n, dtype="float64") for c in cols}, index=idx)
    pred = predict_canonical(df, {
        "family": "attention_lstm",
        "kind": "sequence",
        "feature_cols": cols,
        "booster": model,
        "scaler": None,
        "seq_len": 4,
        "dir_thr": 0.5,
    })
    assert len(pred) == n
    assert "p_long" in pred.columns and "p_short" in pred.columns
    assert pred["p_long"].iloc[:3].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert np.isfinite(pred["p_long"].iloc[3:]).all()
    keras.backend.clear_session()


def test_is_serving_bundle_false_on_missing(tmp_path):
    assert is_serving_bundle(tmp_path) is False
    (tmp_path / "feature_schema.json").write_text("{}")
    assert is_serving_bundle(tmp_path) is False
