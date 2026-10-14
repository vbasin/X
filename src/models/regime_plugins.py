"""Regime model plugins (unsupervised) — CPU family.

HMMPlugin   — Hidden Markov Model (hmmlearn GaussianHMM)
GMMPlugin   — Gaussian Mixture Model (sklearn GaussianMixture)

Both follow the same pattern:
  1. Fit on the *feature distribution* only (no labels).
  2. Assign each bar to a regime (state / component).
  3. Derive directional bias per regime from the *empirical* fraction of
     up/down bars in that regime on the training data.
  4. At inference time: assign regime → look up bias → emit canonical predictions.

This makes them directionally usable through the same canonical contract as
supervised models, and useful as regime-gated routers (e.g. regime → specialist
supervised model per regime in the ensemble layer).

The "confidence" is the posterior probability of the assigned regime:
  HMM: posterior from Viterbi / forward-backward pass (model.predict_proba)
  GMM: component posterior (model.predict_proba)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.models.base import (
    CANONICAL_CONFIDENCE, CANONICAL_DIRECTION, CANONICAL_TS,
    FittedModel, ModelSpec, validate_canonical,
)

logger = logging.getLogger("vtech.models.regime_plugins")

_IDX_TO_DIR = {0: "down", 1: "flat", 2: "up"}
_DIR_FROM_LABEL = {0: "down", 1: "flat", 2: "up"}


# ── Shared regime-model scaffold ──────────────────────────────────────────────

class _RegimeBase:
    """Mixin for unsupervised regime models.

    The artifact is a dict::

        {
            "model":       fitted HMM or GMM,
            "scaler":      fitted RobustScaler,
            "regime_bias": {regime_id: "up"|"down"|"flat"},
            "regime_conf": {regime_id: float},  # historical precision
        }
    """

    name: str

    def _make_model(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        model = self._make_model(spec.params)
        # Stash opt-in extras that _make_model ignores but fit() needs. base_params
        # (without the extras) lets the BIC sweep rebuild K-variants of the model.
        model._vtech_cfg = {
            "pca_variance": spec.params.get("pca_variance"),
            "select_k": spec.params.get("select_k"),
            "base_params": {k: v for k, v in dict(spec.params).items()
                            if k not in ("pca_variance", "select_k")},
        }
        return model

    def _select_k_by_bic(
        self, base_params: dict[str, Any], X: np.ndarray, select_k: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Fit the regime model for each K in the sweep range; keep the lowest-BIC one."""
        lo, hi = _k_range(select_k)
        best_bic, best_model, best_k = np.inf, None, None
        for k in range(lo, hi + 1):
            try:
                m = self._make_model({**base_params, "n_components": k})
                m.fit(X)
                bic = _regime_bic(m, X, k)
            except Exception as exc:  # noqa: BLE001 — a bad K must not kill the sweep
                logger.warning("%s BIC sweep K=%d failed: %s", self.name, k, exc)
                continue
            if np.isfinite(bic) and bic < best_bic:
                best_bic, best_model, best_k = bic, m, k
        if best_model is None:            # every K failed — fall back to a fixed build
            m = self._make_model(base_params)
            try:
                m.fit(X)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s fixed-K fallback fit failed: %s", self.name, exc)
            return m, {"selected_k": base_params.get("n_components"), "bic": None}
        logger.info("%s BIC-selected K=%d (bic=%.1f) over [%d,%d]",
                    self.name, best_k, best_bic, lo, hi)
        return best_model, {"selected_k": best_k, "bic": round(float(best_bic), 2)}

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        # Use all training data from all folds (expanding window)
        X_all = np.vstack([_flatten(f["X_train"]) for f in train_folds])
        y_all = np.concatenate([f["y_train"] for f in train_folds])

        cfg          = getattr(model, "_vtech_cfg", {})
        pca_variance = cfg.get("pca_variance")
        select_k     = cfg.get("select_k")
        base_params  = cfg.get("base_params", {})

        scaler = RobustScaler()
        # float64: covariance estimation (GMM/HMM) is numerically fragile in float32
        X_scaled = scaler.fit_transform(X_all).astype(np.float64)

        # Optional PCA: decorrelate + shrink dimensionality before the (diag-cov)
        # regime model. Essential for HMM, whose flattened sequence input is wide and
        # highly collinear; also fixes GMM ill-conditioning on wide matrices.
        pca = None
        if pca_variance:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=float(pca_variance), svd_solver="full",
                      random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
            logger.info("%s PCA: %d → %d dims (%.0f%% variance)", self.name,
                        pca.n_features_in_, pca.n_components_, float(pca_variance) * 100)

        # Fit the regime model (no labels) — optionally BIC-selecting the state count.
        select_info: dict[str, Any] | None = None
        if select_k:
            model, select_info = self._select_k_by_bic(base_params, X_scaled, select_k)
        else:
            try:
                model.fit(X_scaled)
            except Exception as exc:  # noqa: BLE001
                logger.warning("regime_fit_failed: %s — using random regime assignment", exc)

        # Derive regime bias from training labels
        regime_bias, regime_conf = _compute_regime_bias(model, X_scaled, y_all)

        artifact = {
            "model": model,
            "scaler": scaler,
            "pca": pca,
            "regime_bias": regime_bias,
            "regime_conf": regime_conf,
            "select_info": select_info,
        }

        # Build OOF frame from the last fold
        last  = train_folds[-1]
        X_oof = _transform_regime(artifact, last.get("X_oof", last["X_val"]))
        y_oof = last.get("y_oof", last["y_val"])
        oof   = _predict_from_artifact(artifact, X_oof, ts=last.get("ts_oof"), y_true=y_oof)

        return FittedModel(spec=ModelSpec(family=self.name), artifact=artifact, oof_preds=oof)

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        art = fitted.artifact
        X_t = _transform_regime(art, X)
        df  = _predict_from_artifact(art, X_t)
        validate_canonical(df, plugin_name=self.name)
        return df


# ── HMM ───────────────────────────────────────────────────────────────────────

class HMMPlugin(_RegimeBase):
    name: str = "hmm"

    def _make_model(self, params: dict[str, Any]) -> Any:
        from hmmlearn.hmm import GaussianHMM
        p = dict(params)
        return GaussianHMM(
            n_components=p.get("n_components", 4),
            covariance_type=p.get("covariance_type", "diag"),  # diag is more stable than full
            n_iter=p.get("n_iter", 100),
            random_state=42,
        )


# ── GMM ───────────────────────────────────────────────────────────────────────

class GMMPlugin(_RegimeBase):
    name: str = "gmm"

    def _make_model(self, params: dict[str, Any]) -> Any:
        from sklearn.mixture import GaussianMixture
        p = dict(params)
        return GaussianMixture(
            n_components=p.get("n_components", 4),
            covariance_type=p.get("covariance_type", "diag"),  # full is ill-conditioned on wide float32
            reg_covar=p.get("reg_covar", 1e-4),
            max_iter=p.get("max_iter", 200),
            random_state=42,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


def _transform_regime(art: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Apply the fitted scaler (and optional PCA) to raw fold input, in that order.

    Shared by the OOF build and :meth:`predict` so training and inference transform
    identically. Old artifacts without a ``pca`` key transform scaler-only.
    """
    Xs = art["scaler"].transform(_flatten(X))
    pca = art.get("pca")
    if pca is not None:
        Xs = pca.transform(Xs)
    return Xs


def _k_range(select_k: Any, default: tuple[int, int] = (3, 9)) -> tuple[int, int]:
    """Normalise a ``select_k`` param to an inclusive (lo, hi) state-count sweep range."""
    if select_k is True:
        return default
    if isinstance(select_k, (list, tuple)) and len(select_k) == 2:
        return int(select_k[0]), int(select_k[1])
    if isinstance(select_k, int) and not isinstance(select_k, bool):
        return default[0], int(select_k)
    return default


def _regime_bic(model: Any, X: np.ndarray, k: int) -> float:
    """Bayesian Information Criterion for a fitted regime model (lower = better).

    ``GaussianMixture`` exposes a native ``.bic()``. For hmmlearn ``GaussianHMM`` we
    compute it from the log-likelihood and the free-parameter count
    (start + transition + emission mean/covariance parameters).
    """
    if hasattr(model, "bic"):                      # sklearn GaussianMixture
        return float(model.bic(X))
    n, d = X.shape                                 # hmmlearn GaussianHMM
    log_l = float(model.score(X))
    cov_type = getattr(model, "covariance_type", "diag")
    cov_params = {
        "diag": k * d,
        "full": k * d * (d + 1) / 2.0,
        "tied": d * (d + 1) / 2.0,
        "spherical": float(k),
    }.get(cov_type, k * d)
    # start (k-1) + transitions k*(k-1) + means k*d + covariances
    n_params = (k - 1) + k * (k - 1) + k * d + cov_params
    return -2.0 * log_l + n_params * np.log(max(n, 1))



def _assign_regimes(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (regime_ids, posterior_probs) for each row of X."""
    try:
        # sklearn GaussianMixture
        proba = model.predict_proba(X)
        regimes = proba.argmax(axis=1)
        confs   = proba.max(axis=1)
        return regimes, confs
    except AttributeError:
        pass
    try:
        # hmmlearn HMM — use predict for states, score_samples for log-likelihood
        regimes = model.predict(X)
        # Build a pseudo-posterior from state occupancy
        n_states = model.n_components
        confs = np.full(len(regimes), 1.0 / n_states)  # uniform fallback
        return regimes, confs
    except Exception as exc:
        logger.warning("regime_assign_failed: %s", exc)
        n = len(X)
        return np.zeros(n, dtype=int), np.full(n, 0.25)


def _compute_regime_bias(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[int, str], dict[int, float]]:
    """Compute per-regime empirical directional bias from training labels."""
    regimes, _ = _assign_regimes(model, X)

    # Map raw label 0→down, 1→flat, 2→up
    label_map = {0: "down", 1: "flat", 2: "up"}

    bias: dict[int, str] = {}
    conf: dict[int, float] = {}

    for r in np.unique(regimes):
        mask = regimes == r
        ys   = y[mask]
        if len(ys) == 0:
            bias[int(r)] = "flat"
            conf[int(r)] = 0.33
            continue
        counts = {d: int((ys == i).sum()) for i, d in label_map.items()}
        dominant = max(counts, key=counts.get)
        total    = len(ys)
        bias[int(r)] = dominant
        conf[int(r)] = round(counts[dominant] / total, 4)

    return bias, conf


def _predict_from_artifact(
    artifact: dict[str, Any],
    X_scaled: np.ndarray,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    regimes, regime_confs = _assign_regimes(artifact["model"], X_scaled)
    bias = artifact["regime_bias"]
    conf_map = artifact["regime_conf"]

    directions  = [bias.get(int(r), "flat") for r in regimes]
    # Use min(posterior, regime_precision) as confidence signal
    confidences = [min(float(regime_confs[i]), conf_map.get(int(regimes[i]), 0.33))
                   for i in range(len(regimes))]

    df = pd.DataFrame({
        CANONICAL_TS:         pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION:  directions,
        CANONICAL_CONFIDENCE: confidences,
    })
    if y_true is not None:
        _DIR_FROM = {0: "down", 1: "flat", 2: "up"}
        df["direction_true"] = [_DIR_FROM.get(int(y), "flat") for y in y_true]
    return df
