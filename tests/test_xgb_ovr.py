"""XGBoost one-vs-rest head → 3-class proba."""

from __future__ import annotations

import numpy as np

from src.models.base import ModelSpec
from src.models.tabular_plugins import OvrDirectional, XGBoostPlugin


def test_ovr_predict_proba_shape_and_simplex():
    rng = np.random.default_rng(0)
    n, f = 80, 4
    X = rng.normal(size=(n, f)).astype("float32")
    y = rng.choice([0, 1, 2], size=n, p=[0.15, 0.7, 0.15]).astype("int32")
    plugin = XGBoostPlugin()
    spec = ModelSpec(family="xgboost", params={
        "head": "ovr", "n_estimators": 8, "max_depth": 2,
        "class_balance": "sqrt", "dir_threshold": 0.55,
    })
    plugin.build(spec, {})
    fitted = plugin.fit(None, [{
        "X_train": X, "y_train": y,
        "X_val": X[:20], "y_val": y[:20],
        "X_oof": X, "y_oof": y, "ts_oof": np.arange(n),
    }])
    assert isinstance(fitted.artifact, OvrDirectional)
    proba = fitted.artifact.predict_proba(X)
    assert proba.shape == (n, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert fitted.oof_preds is not None
    assert set(fitted.oof_preds["direction"].unique()) <= {"up", "down", "flat"}
