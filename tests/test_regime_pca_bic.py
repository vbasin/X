"""Regime plugins — opt-in PCA + BIC state-count selection (added for the regime
feature-set A/B, config/regime_ab.yaml).

Covers: GMM native-BIC K selection on per-bar (2-D) input; HMM PCA dimensionality
reduction + manual-BIC K selection on flattened-sequence (3-D) input; and backward
compatibility (no pca_variance / no select_k ⇒ scaler-only, fixed K).
"""

from __future__ import annotations

import numpy as np

from src.models.base import ModelSpec
from src.models.regime_plugins import GMMPlugin, HMMPlugin


def _folds_2d(n: int = 400, d: int = 8, seed: int = 0) -> list[dict]:
    """Two well-separated Gaussian blobs so a mixture model finds real structure."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(-3.0, 0.5, size=(half, d))
    b = rng.normal(+3.0, 0.5, size=(n - half, d))
    X = np.vstack([a, b]).astype(np.float32)
    y = np.concatenate([np.zeros(half, dtype=int), np.full(n - half, 2, dtype=int)])
    return [{
        "X_train": X, "y_train": y,
        "X_val": X[::4], "y_val": y[::4],
        "feature_cols": [f"f{i}" for i in range(d)],
    }]


def _folds_3d(n: int = 300, seq: int = 16, d: int = 6, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, seq, d)).astype(np.float32)
    y = rng.integers(0, 3, size=n)
    return [{
        "X_train": X, "y_train": y,
        "X_val": X[:60], "y_val": y[:60],
        "feature_cols": [f"f{i}" for i in range(d)],
    }]


def test_gmm_bic_selects_k_and_emits_canonical_oof():
    plugin = GMMPlugin()
    spec = ModelSpec(family="gmm", params={"select_k": [2, 6], "covariance_type": "diag"})
    fitted = plugin.fit(plugin.build(spec, {}), _folds_2d())

    assert fitted.oof_preds is not None
    assert set(fitted.oof_preds["direction"].dropna().unique()) <= {"up", "down", "flat"}
    # A real K was BIC-selected within the sweep range and recorded.
    info = fitted.artifact["select_info"]
    assert info is not None and 2 <= info["selected_k"] <= 6
    assert fitted.artifact["model"].n_components == info["selected_k"]
    assert fitted.artifact["pca"] is None          # GMM path requested no PCA


def test_hmm_pca_reduces_dims_then_predicts():
    plugin = HMMPlugin()
    spec = ModelSpec(family="hmm",
                     params={"pca_variance": 0.9, "select_k": [2, 4], "n_iter": 20})
    folds = _folds_3d()
    fitted = plugin.fit(plugin.build(spec, {}), folds)

    pca = fitted.artifact["pca"]
    assert pca is not None
    # flattened window = seq(16) * d(6) = 96 dims → PCA keeps strictly fewer.
    assert 1 <= pca.n_components_ < 16 * 6
    assert fitted.artifact["select_info"]["selected_k"] in (2, 3, 4)

    preds = plugin.predict(fitted, folds[0]["X_train"][:20])
    assert len(preds) == 20
    assert set(preds["direction"].dropna().unique()) <= {"up", "down", "flat"}
    assert preds["confidence"].between(0.0, 1.0).all()


def test_regime_defaults_are_backward_compatible():
    """No pca_variance / no select_k ⇒ scaler-only, fixed K, canonical OOF."""
    plugin = GMMPlugin()
    spec = ModelSpec(family="gmm", params={"n_components": 3, "covariance_type": "diag"})
    fitted = plugin.fit(plugin.build(spec, {}), _folds_2d())

    assert fitted.artifact["pca"] is None
    assert fitted.artifact["select_info"] is None
    assert fitted.artifact["model"].n_components == 3
    assert fitted.oof_preds is not None
