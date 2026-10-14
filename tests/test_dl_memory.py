"""GPU-memory helpers used by sequence DL plugins (LSTM, TCN)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dl_memory import PREDICT_CHUNK, VAL_CAP, predict_chunked, stride_cap
from src.models.sequence_windows import make_windows, trim_warmup


def test_stride_cap_noop_when_under_cap():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    Xs, ys = stride_cap(X, y, 50)
    assert Xs.shape == X.shape
    assert np.array_equal(ys, y)


def test_stride_cap_keeps_endpoints():
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    Xs, ys = stride_cap(X, y, 10)
    assert len(Xs) == 10
    assert ys[0] == 0
    assert ys[-1] == 99


def test_predict_chunked_empty():
    class _M:
        def predict(self, X, verbose=0):
            raise AssertionError("should not be called on empty X")

    out = predict_chunked(_M(), np.empty((0, 4, 3)))
    assert out.shape == (0, 3)


def test_predict_chunked_concatenates():
    class _M:
        def predict(self, X, verbose=0):
            return np.full((len(X), 3), len(X), dtype=np.float32)

    X = np.zeros((5, 2, 2), dtype=np.float32)
    out = predict_chunked(_M(), X, chunk=2)
    assert out.shape == (5, 3)
    # chunks of 2, 2, 1 → first four rows are 2.0, last is 1.0
    assert float(out[0, 0]) == 2.0
    assert float(out[-1, 0]) == 1.0


def test_window_and_trim_align():
    X = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    y = np.arange(20)
    Xw = make_windows(X, 8)
    yt = trim_warmup(y, 8)
    assert Xw.shape == (13, 8, 3)
    assert len(yt) == 13
    assert yt[0] == 7


def test_env_caps_are_positive():
    assert VAL_CAP > 0
    assert PREDICT_CHUNK > 0


def test_lstm_fit_rebuilds_each_fold_with_scaler_and_sequence():
    """LSTM matches TCN: Sequence-fed fit, RobustScaler, fresh weights per fold."""
    from src.models.attention_lstm_plugin import AttentionLSTMPlugin

    rebuilds: list[int] = []
    calls: list = []

    class _Dummy:
        input_shape = (None, 8, 4)

        def fit(self, x, validation_data=None, epochs=1, verbose=0, callbacks=None):
            calls.append({"x": x, "callbacks": callbacks})

        def predict(self, X, verbose=0):
            return np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (len(X), 1))

    plugin = AttentionLSTMPlugin()
    plugin._params = {"seq_len": 8, "epochs": 1, "batch_size": 4,
                      "class_balance": "sqrt"}

    def _rebuild(n):
        rebuilds.append(n)
        return _Dummy()

    plugin._rebuild = _rebuild
    n, f = 40, 4
    X = np.random.default_rng(1).normal(size=(n, f)).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    ts = np.arange(n)
    fold = {
        "X_train": X, "y_train": y,
        "X_val": X[:24], "y_val": y[:24],
        "X_oof": X, "y_oof": y, "ts_oof": ts,
    }
    fitted = plugin.fit(_Dummy(), [fold, dict(fold)])
    assert rebuilds == [4, 4]
    assert len(calls) == 2
    seq = calls[0]["x"]
    assert hasattr(seq, "__getitem__") and hasattr(seq, "__len__")
    xb, yb = seq[0][0], seq[0][1]
    assert xb.ndim == 3 and xb.shape[0] <= 4 and xb.shape[1] == 8 and xb.shape[2] == 4
    assert len(seq[0]) == 3  # sample weights when class_balance is set
    assert calls[0]["callbacks"]
    assert fitted.meta.get("scaler") is not None
    assert fitted.oof_preds is not None and len(fitted.oof_preds) > 0


def test_predict_chunked_dual_dict():
    class _M:
        def predict(self, X, verbose=0):
            n = len(X)
            return {
                "p_long": np.full((n, 1), 0.7, dtype=np.float32),
                "p_short": np.full((n, 1), 0.2, dtype=np.float32),
            }

    X = np.zeros((5, 2, 2), dtype=np.float32)
    out = predict_chunked(_M(), X, chunk=2)
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out[:, 0], 0.7)
    np.testing.assert_allclose(out[:, 1], 0.2)


def test_lstm_dual_binary_fit_uses_head_dict_and_weights():
    from src.models.attention_lstm_plugin import AttentionLSTMPlugin

    calls: list = []

    class _Dummy:
        def fit(self, x, validation_data=None, epochs=1, verbose=0, callbacks=None):
            calls.append(x)

        def predict(self, X, verbose=0):
            n = len(X)
            return np.column_stack([
                np.full(n, 0.6, dtype=np.float32),
                np.full(n, 0.2, dtype=np.float32),
            ])

    plugin = AttentionLSTMPlugin()
    plugin._params = {"seq_len": 8, "epochs": 1, "batch_size": 4,
                      "class_balance": "sqrt", "head": "dual_binary"}
    plugin._head = "dual_binary"
    plugin._rebuild = lambda n: _Dummy()
    n, f = 40, 4
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    y[:6, 0] = 1
    y[6:12, 1] = 1
    fitted = plugin.fit(_Dummy(), [{
        "X_train": X, "y_train": y,
        "X_val": X[:24], "y_val": y[:24],
        "X_oof": X, "y_oof": y, "ts_oof": np.arange(n),
    }])
    seq = calls[0]
    batch = seq[0]
    assert len(batch) == 3  # X, y-dict, sample_weight dict
    yb, sw = batch[1], batch[2]
    assert isinstance(yb, dict) and set(yb) == {"p_long", "p_short"}
    assert isinstance(sw, dict) and set(sw) == {"p_long", "p_short"}
    assert fitted.oof_preds is not None
    assert "p_long" in fitted.oof_preds.columns
    assert "p_short" in fitted.oof_preds.columns
    assert set(fitted.oof_preds["direction"]).issubset({"up", "down", "flat"})
    assert (fitted.oof_preds["direction"] == "up").any()


def test_tcn_dual_binary_fit_uses_head_dict():
    from src.models.tcn_plugin import TCNPlugin

    calls: list = []

    class _Dummy:
        def fit(self, x, validation_data=None, epochs=1, verbose=0, callbacks=None):
            calls.append(x)

        def predict(self, X, verbose=0):
            n = len(X)
            return np.column_stack([
                np.full(n, 0.6, dtype=np.float32),
                np.full(n, 0.2, dtype=np.float32),
            ])

    plugin = TCNPlugin()
    plugin._params = {"seq_len": 8, "epochs": 1, "batch_size": 4,
                      "class_balance": "sqrt", "head": "dual_binary"}
    plugin._head = "dual_binary"
    plugin._rebuild = lambda n: _Dummy()
    n, f = 40, 4
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    y[:6, 0] = 1
    y[6:12, 1] = 1
    fitted = plugin.fit(_Dummy(), [{
        "X_train": X, "y_train": y,
        "X_val": X[:24], "y_val": y[:24],
        "X_oof": X, "y_oof": y, "ts_oof": np.arange(n),
    }])
    batch = calls[0][0]
    assert isinstance(batch[1], dict) and set(batch[1]) == {"p_long", "p_short"}
    assert isinstance(batch[2], dict) and set(batch[2]) == {"p_long", "p_short"}
    assert "p_long" in fitted.oof_preds.columns
    assert (fitted.oof_preds["direction"] == "up").any()


def test_lstm_fit_imports_val_cap_and_stride_caps_val():
    """Regression: 20260828T095411Z NameError VAL_CAP in dual-binary LSTM fit."""
    import src.models.attention_lstm_plugin as mod
    from src.models.attention_lstm_plugin import AttentionLSTMPlugin
    from src.training.aux_labels import AUX_LABEL_COLS

    assert hasattr(mod, "VAL_CAP")

    val_ns: list[int] = []

    class _Dummy:
        def fit(self, x, validation_data=None, epochs=1, verbose=0, callbacks=None):
            val_ns.append(0 if validation_data is None else int(validation_data.n))

        def predict(self, X, verbose=0):
            n = (len(X) if not isinstance(X, dict)
                 else len(next(iter(X.values()))))
            return np.column_stack([
                np.full(n, 0.6, dtype=np.float32),
                np.full(n, 0.2, dtype=np.float32),
            ])

    plugin = AttentionLSTMPlugin()
    plugin._params = {"seq_len": 8, "epochs": 1, "batch_size": 4,
                      "class_balance": "sqrt", "head": "dual_binary"}
    plugin._head = "dual_binary"
    plugin._rebuild = lambda n: _Dummy()
    plugin._obj_doc = {"labels": {"aux": list(AUX_LABEL_COLS)}}
    n, f = 80, 6
    rng = np.random.default_rng(5)
    cols = [
        "of_ofi", "cs_close",
        "lg_target_v_atr", "lg_target_v_close",
        "st_target_v_atr", "st_target_v_close",
    ]
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    y[:8, 0] = 1
    y[8:16, 1] = 1
    y_aux = rng.normal(size=(n, len(AUX_LABEL_COLS))).astype(np.float32)
    orig = mod.VAL_CAP
    try:
        mod.VAL_CAP = 10
        plugin.fit(_Dummy(), [{
            "X_train": X, "y_train": y,
            "X_val": X[:60], "y_val": y[:60],
            "X_oof": X, "y_oof": y, "ts_oof": np.arange(n),
            "feature_cols": cols,
            "y_aux_train": y_aux, "y_aux_val": y_aux[:60],
        }])
    finally:
        mod.VAL_CAP = orig
    # 60 val rows → 53 windows; stride-capped to VAL_CAP=10
    assert val_ns == [10]


def test_lstm_class_balance_none_omits_sample_weight():
    from src.models.attention_lstm_plugin import AttentionLSTMPlugin

    calls: list = []

    class _Dummy:
        def fit(self, x, validation_data=None, epochs=1, verbose=0, callbacks=None):
            calls.append(x)

        def predict(self, X, verbose=0):
            return np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (len(X), 1))

    plugin = AttentionLSTMPlugin()
    plugin._params = {"seq_len": 8, "epochs": 1, "batch_size": 4, "class_balance": None}
    plugin._rebuild = lambda n: _Dummy()
    n, f = 40, 4
    X = np.random.default_rng(2).normal(size=(n, f)).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    plugin.fit(_Dummy(), [{
        "X_train": X, "y_train": y,
        "X_val": X[:24], "y_val": y[:24],
        "X_oof": X, "y_oof": y, "ts_oof": np.arange(n),
    }])
    assert len(calls[0][0]) == 2  # X, y only — no sample_weight


def test_tcn_fit_feeds_keras_sequence_not_full_array():
    """Regression: full numpy windows on model.fit OOM the A10 (_EagerConst)."""
    from src.models.tcn_plugin import TCNPlugin

    calls: list = []

    class _Dummy:
        input_shape = (None, 128, 4)

        def fit(self, x, validation_data=None, epochs=1, verbose=0, callbacks=None):
            calls.append(x)

        def predict(self, X, verbose=0):
            return np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (len(X), 1))

    plugin = TCNPlugin()
    plugin._params = {"seq_len": 8, "epochs": 1, "batch_size": 4, "class_balance": "sqrt"}
    plugin._rebuild = lambda n: _Dummy()  # skip Keras so this stays a unit test
    n, f = 40, 4
    X = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    y = np.zeros(n, dtype=np.int32)
    ts = np.arange(n)
    fitted = plugin.fit(_Dummy(), [{
        "X_train": X, "y_train": y,
        "X_val": X[:24], "y_val": y[:24],
        "X_oof": X, "y_oof": y, "ts_oof": ts,
    }])
    assert calls, "fit was not called"
    seq = calls[0]
    assert hasattr(seq, "__getitem__") and hasattr(seq, "__len__")
    batch = seq[0]
    xb, yb = batch[0], batch[1]
    assert xb.ndim == 3
    assert xb.shape[0] <= 4
    assert xb.shape[1] == 8
    assert xb.shape[2] == 4
    # sample weights present when class_balance is set
    assert len(batch) == 3
    assert fitted.meta.get("scaler") is not None


def test_describe_walk_forward_folds_equal_count():
    from src.search.fold_cache import describe_walk_forward_folds

    n = 40
    ts = pd.date_range("2026-01-01", periods=n, freq="5s", tz="UTC")
    y = np.array([1] * 30 + [0] * 5 + [2] * 5)
    folds = describe_walk_forward_folds(ts.to_numpy(), y, n_folds=3)
    assert len(folds) == 3
    assert folds[0]["train"]["n_bars"] == 10
    assert folds[0]["oof"]["n_bars"] == 10
    assert folds[2]["train"]["n_bars"] == 30
    assert folds[0]["train"]["start"].startswith("2026-01-01")
    assert folds[2]["oof"]["end"].startswith("2026-01-01")
