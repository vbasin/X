"""Dual-binary head helpers (ADR-076)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dual_binary import (
    as_dual_binary_y, dual_binary_to_canonical, true_direction_from_dual,
)
from src.training.labels import BIG_DOWN, BIG_UP


def test_as_dual_binary_from_3class():
    y = np.array([BIG_DOWN, 1, BIG_UP, 1], dtype="int32")
    got = as_dual_binary_y(y)
    np.testing.assert_array_equal(got[:, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(got[:, 1], [1, 0, 0, 0])


def test_as_dual_binary_passthrough_2d():
    y = np.array([[1, 0], [0, 1]], dtype="float32")
    got = as_dual_binary_y(y)
    np.testing.assert_array_equal(got, y)


def test_true_direction_never_both():
    y = np.array([[1, 0], [0, 1], [0, 0]], dtype="float32")
    d = true_direction_from_dual(y)
    assert list(d) == ["up", "down", "flat"]


def test_xgboost_dual_binary_oof_has_p_long():
    from src.models.catalog import get_entry
    from src.models.base import ModelSpec
    from src.models.tabular_plugins import XGBoostPlugin

    rng = np.random.default_rng(0)
    n, f = 80, 4
    X = rng.normal(size=(n, f)).astype("float32")
    y = np.zeros((n, 2), dtype="float32")
    y[:10, 0] = 1
    y[10:20, 1] = 1
    ts = pd.date_range("2026-01-02", periods=n, freq="15s", tz="UTC")
    fold = {
        "X_train": X[:60], "y_train": y[:60],
        "X_val": X[60:], "y_val": y[60:],
        "X_oof": X[60:], "y_oof": y[60:],
        "ts_oof": ts[60:].to_numpy(),
        "feature_cols": [f"c{i}" for i in range(f)],
    }
    spec = ModelSpec(family="xgboost", params={
        **get_entry("xgboost").params_default, "head": "dual_binary",
        "n_estimators": 8, "max_depth": 2,
    })
    plug = XGBoostPlugin()
    model = plug.build(spec, {"labels": {"encoding": "dual_binary"}})
    fitted = plug.fit(model, [fold], objective={"labels": {"encoding": "dual_binary"}})
    assert fitted.oof_preds is not None
    assert "p_long" in fitted.oof_preds.columns
    assert "p_short" in fitted.oof_preds.columns
    assert set(fitted.oof_preds["direction"]).issubset({"up", "down", "flat"})


def test_stack_head_predict_dict():
    from src.models.dual_binary import stack_head_predict
    pred = {
        "p_long": np.array([[0.8], [0.1]], dtype="float32"),
        "p_short": np.array([[0.2], [0.9]], dtype="float32"),
        "side_rank": np.array([[0.8], [0.1]], dtype="float32"),
    }
    got = stack_head_predict(pred)
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got[:, 0], [0.8, 0.1])
    np.testing.assert_allclose(got[:, 1], [0.2, 0.9])


def test_compute_diagnostics_dual_binary_labels():
    from src.evaluation._kpi import compute_diagnostics

    n = 8
    oof = pd.DataFrame({
        "ts": pd.date_range("2026-01-02", periods=n, freq="15s", tz="UTC"),
        "direction": ["up", "down", "flat", "flat", "up", "flat", "down", "flat"],
        "confidence": [0.8, 0.7, 0.2, 0.3, 0.6, 0.1, 0.9, 0.4],
        "p_long":     [0.8, 0.1, 0.2, 0.3, 0.6, 0.1, 0.05, 0.4],
        "p_short":    [0.1, 0.7, 0.15, 0.2, 0.2, 0.1, 0.9, 0.3],
        "direction_true": ["up", "down", "flat", "up", "flat", "flat", "down", "flat"],
    })
    diag = compute_diagnostics(oof, oof[["direction_true"]], {
        "labels": {"move_threshold": 50, "mae_tolerance": 6, "encoding": "dual_binary"},
    }, gate_tau=None)
    assert diag["head"] == "dual_binary"
    assert diag["labels"] == ["long", "short"]
    assert diag["canonical_labels"] == ["down", "flat", "up"]
    assert "long" in diag["heads"] and "short" in diag["heads"]
    assert diag["heads"]["long"]["n_pos"] == 2
    assert diag["heads"]["short"]["n_pos"] == 2
    assert diag["confusion_matrix_raw"]["operating_point"] == "dual_binary"
    assert diag["confusion_matrix_raw"]["labels"] == ["down", "flat", "up"]


def test_tcn_dual_binary_named_heads():
    from tensorflow import keras
    from src.models.tcn_plugin import _build_tcn

    keras.backend.clear_session()
    model = _build_tcn(
        seq_len=8, n_features=4,
        p={"filters": 4, "kernel_size": 2, "dilations": [1], "dropout": 0.0,
           "head": "dual_binary"},
    )
    names = list(getattr(model, "output_names", []) or [])
    assert "p_long" in names and "p_short" in names
    pred = model.predict(np.zeros((2, 8, 4), dtype="float32"), verbose=0)
    assert set(pred) == {"p_long", "p_short"}
    keras.backend.clear_session()


def test_cnn_and_transformer_dual_binary_named_heads():
    from tensorflow import keras
    from src.models.base import ModelSpec
    from src.models.cnn_plugin import CNNPlugin
    from src.models.transformer_plugin import TransformerPlugin

    keras.backend.clear_session()
    cnn = CNNPlugin().build(ModelSpec(family="cnn", params={
        "head": "dual_binary", "seq_len": 8, "n_features": 4,
        "filters": 4, "n_blocks": 1, "dropout": 0.0,
    }), {})
    assert "p_long" in cnn.output_names and "p_short" in cnn.output_names
    keras.backend.clear_session()
    tr = TransformerPlugin().build(ModelSpec(family="transformer", params={
        "head": "dual_binary", "seq_len": 8, "n_features": 4,
        "d_model": 8, "n_heads": 2, "n_layers": 1, "dropout": 0.0,
    }), {})
    assert "p_long" in tr.output_names and "p_short" in tr.output_names
    keras.backend.clear_session()


def test_lstm_dual_binary_aux_and_head_context():
    from tensorflow import keras
    from src.models.attention_lstm import build_attention_lstm
    from src.training.aux_labels import (
        LONG_AUX_MAP, SHORT_AUX_MAP, SIDE_RANK_NAME,
        STOP_FIRST_LONG, STOP_FIRST_SHORT,
    )

    keras.backend.clear_session()
    model = build_attention_lstm(
        (8, 4), lstm_units=[8], attention_heads=0, dense_units=[4],
        dropout=0.0, head="dual_binary", n_long_ctx=2, n_short_ctx=2, aux=True,
    )
    names = set(getattr(model, "output_names", []) or [])
    assert "p_long" in names and "p_short" in names
    assert SIDE_RANK_NAME in names
    assert STOP_FIRST_LONG in names and STOP_FIRST_SHORT in names
    for k, *_ in LONG_AUX_MAP + SHORT_AUX_MAP:
        assert k in names
    assert "long_MAE_before_target_or_MFE" not in names
    assert "short_MAE_before_target_or_MFE" not in names
    n = 6
    x = {
        "input": np.zeros((n, 8, 4), dtype="float32"),
        "long_ctx": np.ones((n, 2), dtype="float32"),
        "short_ctx": -np.ones((n, 2), dtype="float32"),
    }
    pred = model.predict(x, verbose=0)
    assert pred["p_long"].shape[0] == n
    y = {"p_long": np.array([[1], [0], [0], [1], [0], [0]], dtype="float32"),
         "p_short": np.array([[0], [1], [0], [0], [1], [0]], dtype="float32"),
         SIDE_RANK_NAME: np.array([[1], [0], [0], [1], [0], [0]], dtype="float32")}
    for k, *_ in LONG_AUX_MAP + SHORT_AUX_MAP:
        y[k] = np.zeros((n, 1), dtype="float32")
    y[STOP_FIRST_LONG] = np.array([[1], [0], [0], [1], [0], [0]], dtype="float32")
    y[STOP_FIRST_SHORT] = np.array([[0], [1], [0], [0], [1], [0]], dtype="float32")
    hist = model.fit(x, y, epochs=1, verbose=0)
    assert "loss" in hist.history
    keras.backend.clear_session()


def test_dual_binary_fire_from_objective():
    from src.models.dual_binary import DUAL_BINARY_FIRE, dual_binary_fire
    assert dual_binary_fire(None) == DUAL_BINARY_FIRE
    assert dual_binary_fire({"labels": {"fire": 0.40}}) == pytest.approx(0.40)
    from src.objectives.loader import load_objective
    assert dual_binary_fire(load_objective("nq-50pt-econ-lg-st")) == pytest.approx(0.40)


def test_lstm_dual_binary_named_heads():
    from tensorflow import keras
    from src.models.attention_lstm import build_attention_lstm

    keras.backend.clear_session()
    model = build_attention_lstm(
        (8, 4), lstm_units=[8], attention_heads=0, dense_units=[4],
        dropout=0.0, head="dual_binary",
    )
    names = list(getattr(model, "output_names", []) or [])
    assert "p_long" in names and "p_short" in names
    x = np.zeros((3, 8, 4), dtype="float32")
    pred = model.predict(x, verbose=0)
    assert set(pred) == {"p_long", "p_short"}
    assert pred["p_long"].shape[0] == 3
    y = {"p_long": np.array([1.0, 0.0, 0.0], dtype="float32"),
         "p_short": np.array([0.0, 1.0, 0.0], dtype="float32")}
    hist = model.fit(x, y, epochs=1, verbose=0)
    assert "loss" in hist.history
    keras.backend.clear_session()


def test_window_pack_for_model_splits_lg_st_ctx():
    from src.models.dual_binary import window_pack_for_model

    class _T:
        def __init__(self, name):
            self.name = name

    class _M:
        inputs = [_T("input"), _T("long_ctx"), _T("short_ctx")]

    cols = [
        "a", "b", "c",
        "lg_target_v_atr", "lg_target_v_close",
        "st_target_v_atr", "st_target_v_close",
    ]
    X = np.arange(10 * 7, dtype="float32").reshape(10, 7)
    got = window_pack_for_model(X, 4, cols, _M())
    assert set(got) == {"input", "long_ctx", "short_ctx"}
    assert got["input"].shape == (7, 4, 3)
    assert got["long_ctx"].shape == (7, 2)
    assert got["short_ctx"].shape == (7, 2)
    np.testing.assert_array_equal(got["input"][-1, -1], X[-1, :3])
    np.testing.assert_array_equal(got["long_ctx"][-1], X[-1, 3:5])
    np.testing.assert_array_equal(got["short_ctx"][-1], X[-1, 5:7])


def test_window_pack_htf_is_last_bar_not_sequence():
    from src.models.dual_binary import window_pack_for_model

    class _T:
        def __init__(self, name):
            self.name = name

    class _M:
        inputs = [_T("input"), _T("long_ctx"), _T("short_ctx")]

    cols = ["cs_close", "of_ofi", "h1_tr_gap", "d1_zz_signal",
            "lg_target_v_atr", "st_target_v_atr"]
    X = np.arange(8 * 6, dtype="float32").reshape(8, 6)
    got = window_pack_for_model(X, 4, cols, _M())
    assert got["input"].shape == (5, 4, 2)
    # HTF + side geometry → 3 ctx cols per head (lg/st + h1 + d1)
    assert got["long_ctx"].shape == (5, 3)
    assert got["short_ctx"].shape == (5, 3)
    np.testing.assert_array_equal(got["input"][-1, -1], X[-1, :2])


def test_window_pack_for_model_single_input_keeps_ndarray():
    from src.models.dual_binary import window_pack_for_model

    class _Seq:
        def predict(self, X, verbose=0):
            return np.zeros((len(X), 3))

    X = np.arange(12, dtype="float32").reshape(6, 2)
    got = window_pack_for_model(X, 3, ["a", "b"], _Seq())
    assert isinstance(got, np.ndarray)
    assert got.shape == (4, 3, 2)


def test_canonical_fire_and_probs():
    pl = np.array([0.8, 0.2, 0.4, 0.3, 0.05])
    ps = np.array([0.1, 0.9, 0.3, 0.3, 0.02])
    ts = pd.date_range("2026-01-02", periods=5, freq="15s", tz="UTC")
    y = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]], dtype="float32")
    df = dual_binary_to_canonical(pl, ps, ts=ts, y_true=y)
    # ADR-089: side = argmax, event = max(p) ≥ 0.1. 0.4>0.3 is up; 0.05>0.02 is flat.
    assert list(df["direction"]) == ["up", "down", "up", "flat", "flat"]
    assert list(df["direction_true"]) == ["up", "down", "flat", "flat", "flat"]
    np.testing.assert_allclose(df["p_long"], pl)
    np.testing.assert_allclose(df["p_short"], ps)
    old = dual_binary_to_canonical(pl[:3], ps[:3], fire=0.5)
    assert list(old["direction"]) == ["up", "down", "flat"]
