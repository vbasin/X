"""Tabular / tree-ensemble CPU plugins.

Provides four model plugins that share a common sklearn-style scaffold:
  XGBoostPlugin, LightGBMPlugin, RandomForestPlugin, LogisticRegressionPlugin

All supervised; they expect flat 2-D feature arrays (no sequence dimension).
The controller passes ``X_train`` with shape ``(n_bars, n_features)`` directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.models.base import (
    CANONICAL_CONFIDENCE, CANONICAL_DIRECTION, CANONICAL_TS,
    FittedModel, ModelSpec, validate_canonical,
)

logger = logging.getLogger("vtech.models.tabular_plugins")

# 3-class label index → canonical direction string
_IDX_TO_DIR = {0: "down", 1: "flat", 2: "up"}
_DIR_FROM_LABEL = {0: "down", 1: "flat", 2: "up"}


def _tree_gpu_params(framework: str) -> dict[str, Any]:
    """GPU device params for tree frameworks, gated by ``VTECH_TREE_DEVICE``.

    Returns an empty dict on the CPU search path (env unset / "cpu"), so the
    CPU VM's behaviour is unchanged. The GPU runner service sets
    ``VTECH_TREE_DEVICE=cuda`` to route xgboost/lightgbm onto the A10.

      xgboost 3.x : device="cuda" + tree_method="hist"  (gpu_hist is deprecated)
      lightgbm 4.x: device="gpu"  (requires a GPU-enabled lightgbm build)
    """
    dev = os.environ.get("VTECH_TREE_DEVICE", "").strip().lower()
    if dev in ("", "cpu"):
        return {}
    if framework == "xgboost":
        return {"device": "cuda", "tree_method": "hist"}
    if framework == "lightgbm":
        return {"device": "gpu"}
    return {}


# ── Shared scaffold ───────────────────────────────────────────────────────────

class _TabularBase:
    """Mixin for supervised, flat-feature (non-sequence) models."""

    name: str  # set by subclass

    # Plugin-level knobs — consumed here, NOT passed to the estimator constructor:
    #   class_balance : None | "balanced" | "sqrt"  → per-sample weights in fit() (B)
    #   dir_threshold : None | float in (0,1)        → directional firing threshold (A):
    #                   predict "up"/"down" when P(dir) >= dir_threshold even if flat's
    #                   prob is higher, decoupling "take a trade" from "beat the flat class".
    #   head          : "multiclass" | "ovr"         → 3-class softmax vs two binary heads
    _PLUGIN_KEYS = ("class_balance", "dir_threshold", "head")
    _class_balance: Any = None
    _dir_threshold: Any = None

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        self._class_balance = spec.params.get("class_balance")
        self._dir_threshold = spec.params.get("dir_threshold")
        self._head = str(spec.params.get("head") or "multiclass")
        self._spec_params = dict(spec.params)
        self._obj_doc = objective
        clean = {k: v for k, v in spec.params.items() if k not in self._PLUGIN_KEYS}
        if self._head == "ovr":
            return None
        return self._make_estimator(clean)

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        """Train on the last fold's data (expanding-window: last fold = biggest train set)."""
        if objective is not None:
            self._obj_doc = objective
        oof_frames: list[pd.DataFrame] = []
        final_model = model
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start

        if getattr(self, "_head", "multiclass") == "dual_binary":
            from src.training.cost_matrix import custom_optimization_enabled
            if custom_optimization_enabled(getattr(self, "_obj_doc", None)):
                logger.warning(
                    "custom_optimization ignored for %s dual_binary head "
                    "(3x3 cost matrix is 3-class only)", self.name)
            return self._fit_dual_binary(train_folds)

        if getattr(self, "_head", "multiclass") == "ovr":
            from src.training.cost_matrix import custom_optimization_enabled
            if custom_optimization_enabled(getattr(self, "_obj_doc", None)):
                logger.warning(
                    "custom_optimization ignored for %s ovr head "
                    "(3x3 cost matrix is 3-class only)", self.name)
            return self._fit_ovr(train_folds)

        from src.training.cost_matrix import cost_matrix_array
        cost_on = cost_matrix_array(getattr(self, "_obj_doc", None)) is not None
        if cost_on and self.name not in ("xgboost", "lightgbm"):
            logger.warning(
                "custom_optimization ignored for family=%s "
                "(no 3-class cost-matrix train obj; using default loss)",
                self.name,
            )
            cost_on = False

        for k, fold in enumerate(train_folds):
            # Flatten sequences if the controller passed 3-D arrays
            X_tr = _flatten(fold["X_train"])
            y_tr = fold["y_train"]
            X_oof = _flatten(fold.get("X_oof", fold["X_val"]))
            y_oof = fold.get("y_oof", fold["y_val"])

            # Scale per-fold (fit on train, apply to oof)
            scaler = RobustScaler()
            X_tr  = scaler.fit_transform(X_tr)
            X_oof = scaler.transform(X_oof)

            note_train_start(k, n_folds)
            final_model = self._clone(model)
            sw = None if cost_on else _make_sample_weight(y_tr, self._class_balance)
            if sw is not None:
                final_model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                final_model.fit(X_tr, y_tr)

            oof_proba = _safe_predict_proba(final_model, X_oof)
            oof_frames.append(_proba_to_canonical(
                oof_proba, y_true=y_oof, dir_threshold=self._dir_threshold,
                ts=fold.get("ts_oof")))
            checkpoint_fold(k, n_folds, oof_frames[-1])

        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name), artifact=final_model, oof_preds=oof,
            # Serving contract: the estimator was fit on RobustScaler-transformed
            # features — persist the final fold's scaler so a finalized bundle can
            # reproduce inference-time preprocessing (consumed by search.finalize).
            meta={"scaler": scaler},
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        X_flat = _flatten(X)
        art = fitted.artifact
        if isinstance(art, DualBinaryPair) or (fitted.meta or {}).get("head") == "dual_binary":
            from src.models.dual_binary import dual_binary_fire, dual_binary_to_canonical
            p_long, p_short = art.predict_pair(X_flat)
            df = dual_binary_to_canonical(
                p_long, p_short,
                fire=dual_binary_fire(getattr(self, "_obj_doc", None)))
            validate_canonical(df, plugin_name=self.name)
            return df
        proba = _safe_predict_proba(art, X_flat)
        df = _proba_to_canonical(proba, dir_threshold=self._dir_threshold)
        validate_canonical(df, plugin_name=self.name)
        return df

    def _clone(self, model: Any) -> Any:
        """Return a fresh unfitted copy of *model* with the same params."""
        from src.training.cost_matrix import cost_matrix_array
        # Callable custom objectives often fail sklearn clone; rebuild instead.
        if cost_matrix_array(getattr(self, "_obj_doc", None)) is not None:
            clean = {k: v for k, v in (self._spec_params or {}).items()
                     if k not in self._PLUGIN_KEYS}
            return self._make_estimator(clean)
        from sklearn.base import clone as sklearn_clone
        try:
            return sklearn_clone(model)
        except Exception:
            return self._make_estimator(model.get_params())

    def _make_binary_estimator(self) -> Any:
        raise NotImplementedError("ovr head is only implemented for xgboost")

    def _fit_ovr(self, train_folds: list[dict[str, np.ndarray]]) -> FittedModel:
        """Two binary heads: P(up vs rest) and P(down vs rest) → 3-class proba."""
        oof_frames: list[pd.DataFrame] = []
        last: OvrDirectional | None = None
        scaler_last = None
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start
        for k, fold in enumerate(train_folds):
            X_tr = _flatten(fold["X_train"])
            y_tr = np.asarray(fold["y_train"])
            X_oof = _flatten(fold.get("X_oof", fold["X_val"]))
            y_oof = fold.get("y_oof", fold["y_val"])
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_oof = scaler.transform(X_oof)
            y_up = (y_tr == 2).astype("int32")
            y_dn = (y_tr == 0).astype("int32")
            up = self._make_binary_estimator()
            dn = self._make_binary_estimator()
            note_train_start(k, n_folds)
            sw_up = _make_sample_weight(y_up, self._class_balance)
            sw_dn = _make_sample_weight(y_dn, self._class_balance)
            if sw_up is not None:
                up.fit(X_tr, y_up, sample_weight=sw_up)
            else:
                up.fit(X_tr, y_up)
            if sw_dn is not None:
                dn.fit(X_tr, y_dn, sample_weight=sw_dn)
            else:
                dn.fit(X_tr, y_dn)
            last = OvrDirectional(up, dn)
            scaler_last = scaler
            oof_proba = last.predict_proba(X_oof)
            oof_frames.append(_proba_to_canonical(
                oof_proba, y_true=y_oof, dir_threshold=self._dir_threshold,
                ts=fold.get("ts_oof")))
            checkpoint_fold(k, n_folds, oof_frames[-1])
        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name, params=getattr(self, "_spec_params", {})),
            artifact=last, oof_preds=oof, meta={"scaler": scaler_last, "head": "ovr"},
        )

    def _fit_dual_binary(self, train_folds: list[dict[str, np.ndarray]]) -> FittedModel:
        """Two independent binaries on y_long / y_short (ADR-076)."""
        from src.models.dual_binary import as_dual_binary_y, dual_binary_to_canonical

        oof_frames: list[pd.DataFrame] = []
        last: DualBinaryPair | None = None
        scaler_last = None
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start
        for k, fold in enumerate(train_folds):
            X_tr = _flatten(fold["X_train"])
            y_tr = as_dual_binary_y(fold["y_train"])
            X_oof = _flatten(fold.get("X_oof", fold["X_val"]))
            y_oof = as_dual_binary_y(fold.get("y_oof", fold["y_val"]))
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_oof = scaler.transform(X_oof)
            cols = list(fold.get("feature_cols") or [])
            from src.models.dual_binary import side_feature_index
            li = side_feature_index(cols, "long") if cols and len(cols) == X_tr.shape[-1] else None
            si = side_feature_index(cols, "short") if cols and len(cols) == X_tr.shape[-1] else None
            X_tr_l = X_tr[:, li] if li is not None else X_tr
            X_tr_s = X_tr[:, si] if si is not None else X_tr
            X_oof_l = X_oof[:, li] if li is not None else X_oof
            X_oof_s = X_oof[:, si] if si is not None else X_oof
            up = self._make_binary_estimator()
            dn = self._make_binary_estimator()
            note_train_start(k, n_folds)
            sw_up = _combine_sw(
                _make_sample_weight(y_tr[:, 0].astype("int32"), self._class_balance),
                fold.get("y_aux_train"), y_tr[:, 0], "long",
            )
            sw_dn = _combine_sw(
                _make_sample_weight(y_tr[:, 1].astype("int32"), self._class_balance),
                fold.get("y_aux_train"), y_tr[:, 1], "short",
            )
            if sw_up is not None:
                up.fit(X_tr_l, y_tr[:, 0].astype("int32"), sample_weight=sw_up)
            else:
                up.fit(X_tr_l, y_tr[:, 0].astype("int32"))
            if sw_dn is not None:
                dn.fit(X_tr_s, y_tr[:, 1].astype("int32"), sample_weight=sw_dn)
            else:
                dn.fit(X_tr_s, y_tr[:, 1].astype("int32"))
            last = DualBinaryPair(up, dn, long_idx=li, short_idx=si)
            scaler_last = scaler
            p_long, p_short = last.predict_pair(X_oof)
            from src.models.dual_binary import dual_binary_fire
            oof_frames.append(dual_binary_to_canonical(
                p_long, p_short, ts=fold.get("ts_oof"), y_true=y_oof,
                fire=dual_binary_fire(getattr(self, "_obj_doc", None))))
            checkpoint_fold(k, n_folds, oof_frames[-1])
        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name, params=getattr(self, "_spec_params", {})),
            artifact=last, oof_preds=oof,
            meta={"scaler": scaler_last, "head": "dual_binary"},
        )


class DualBinaryPair:
    """Two independent binary classifiers → (p_long, p_short)."""

    def __init__(
        self,
        long_clf: Any,
        short_clf: Any,
        long_idx: list[int] | None = None,
        short_idx: list[int] | None = None,
    ) -> None:
        self.long_clf = long_clf
        self.short_clf = short_clf
        self.long_idx = long_idx
        self.short_idx = short_idx

    def predict_pair(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xl = X[:, self.long_idx] if self.long_idx is not None else X
        Xs = X[:, self.short_idx] if self.short_idx is not None else X
        pu = np.asarray(self.long_clf.predict_proba(Xl), dtype="float64")
        ps = np.asarray(self.short_clf.predict_proba(Xs), dtype="float64")
        p_long = pu[:, 1] if pu.ndim == 2 and pu.shape[1] > 1 else pu.reshape(-1)
        p_short = ps[:, 1] if ps.ndim == 2 and ps.shape[1] > 1 else ps.reshape(-1)
        return np.clip(p_long, 0.0, 1.0), np.clip(p_short, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_long, p_short = self.predict_pair(X)
        return np.column_stack([p_long, p_short])


class OvrDirectional:
    """Pair of binary classifiers exposing 3-class ``predict_proba`` [down, flat, up]."""

    def __init__(self, up: Any, down: Any) -> None:
        self.up = up
        self.down = down

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pu = np.asarray(self.up.predict_proba(X), dtype="float64")
        pdn = np.asarray(self.down.predict_proba(X), dtype="float64")
        p_up = pu[:, 1] if pu.ndim == 2 and pu.shape[1] > 1 else pu.reshape(-1)
        p_dn = pdn[:, 1] if pdn.ndim == 2 and pdn.shape[1] > 1 else pdn.reshape(-1)
        p_up = np.clip(p_up, 0.0, 1.0)
        p_dn = np.clip(p_dn, 0.0, 1.0)
        s = p_up + p_dn
        over = s > 1.0
        p_up = np.where(over, p_up / np.maximum(s, 1e-12), p_up)
        p_dn = np.where(over, p_dn / np.maximum(s, 1e-12), p_dn)
        p_flat = np.clip(1.0 - p_up - p_dn, 0.0, 1.0)
        return np.column_stack([p_dn, p_flat, p_up])


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostPlugin(_TabularBase):
    name: str = "xgboost"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        from xgboost import XGBClassifier
        p = dict(params)
        p.setdefault("n_estimators",  500)
        p.setdefault("max_depth",     6)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("subsample",     0.8)
        p.setdefault("colsample_bytree", 0.8)
        p.setdefault("eval_metric",   "mlogloss")
        p.setdefault("use_label_encoder", False)
        p.setdefault("verbosity",     0)
        p.setdefault("n_jobs",        -1)
        p.setdefault("num_class",     3)
        p.setdefault("objective",     "multi:softprob")
        for k, v in _tree_gpu_params("xgboost").items():
            p.setdefault(k, v)
        drop = ("use_label_encoder", "class_balance", "dir_threshold", "head")
        from src.training.cost_matrix import apply_tree_custom_objective
        p = apply_tree_custom_objective(
            p, getattr(self, "_obj_doc", None), family="xgboost")
        return XGBClassifier(**{k: v for k, v in p.items() if k not in drop})

    def _make_binary_estimator(self) -> Any:
        from xgboost import XGBClassifier
        p = {k: v for k, v in (getattr(self, "_spec_params", {}) or {}).items()
             if k not in self._PLUGIN_KEYS}
        p.setdefault("n_estimators", 150)
        p.setdefault("max_depth", 4)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("subsample", 0.8)
        p.setdefault("colsample_bytree", 0.8)
        p.setdefault("verbosity", 0)
        p.setdefault("n_jobs", -1)
        p["objective"] = "binary:logistic"
        p["eval_metric"] = "logloss"
        p.pop("num_class", None)
        p.pop("use_label_encoder", None)
        for k, v in _tree_gpu_params("xgboost").items():
            p.setdefault(k, v)
        return XGBClassifier(**p)


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LightGBMPlugin(_TabularBase):
    name: str = "lightgbm"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        from lightgbm import LGBMClassifier
        p = dict(params)
        p.setdefault("n_estimators",  500)
        p.setdefault("max_depth",     6)
        p.setdefault("learning_rate", 0.05)
        p.setdefault("subsample",     0.8)
        p.setdefault("num_leaves",    63)
        p.setdefault("verbose",      -1)
        p.setdefault("n_jobs",        -1)
        for k, v in _tree_gpu_params("lightgbm").items():
            p.setdefault(k, v)
        from src.training.cost_matrix import apply_tree_custom_objective
        p = apply_tree_custom_objective(
            p, getattr(self, "_obj_doc", None), family="lightgbm")
        return LGBMClassifier(**p)


# ── Random Forest ─────────────────────────────────────────────────────────────

class RandomForestPlugin(_TabularBase):
    name: str = "random_forest"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        p = dict(params)
        p.setdefault("n_estimators", 300)
        p.setdefault("max_depth",    12)
        p.setdefault("n_jobs",       -1)
        p.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**p)


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticRegressionPlugin(_TabularBase):
    name: str = "logistic_regression"

    def _make_estimator(self, params: dict[str, Any]) -> Any:
        from sklearn.linear_model import LogisticRegression
        p = dict(params)
        p.setdefault("C",            1.0)
        p.setdefault("max_iter",     500)
        p.setdefault("class_weight", "balanced")
        p.setdefault("solver",       "lbfgs")
        # multi_class removed in sklearn 1.8 (solver determines it automatically)
        p.pop("multi_class", None)
        return LogisticRegression(**p)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _flatten(X: np.ndarray) -> np.ndarray:
    """Collapse a 3-D sequence array (N, T, F) → 2-D (N, T*F)."""
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


def _combine_sw(
    class_w: np.ndarray | None,
    aux: Any,
    y_side: np.ndarray,
    side: str,
) -> np.ndarray | None:
    """Class-balance × near-miss MFE weights (ADR-095)."""
    from src.training.aux_labels import near_miss_sample_weight
    miss = near_miss_sample_weight(y_side, aux, side=side)
    if class_w is None:
        return miss
    if miss is None:
        return class_w
    return np.asarray(class_w, dtype="float64") * miss


def _make_sample_weight(
    y: np.ndarray, mode: Any, mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Per-sample weights for class rebalancing (B).

    mode:
      None / falsy → no weighting (returns None)
      "balanced"   → inverse class frequency (sklearn 'balanced' scheme)
      "sqrt"       → sqrt of inverse frequency (gentler; avoids over-firing)

    *mask* (bool, same length as *y*) restricts frequency to loss-active
    rows. Masked-out rows get weight 0.
    """
    if not mode:
        return None
    y = np.asarray(y)
    m = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    if m is not None and m.shape[0] != y.shape[0]:
        raise ValueError(f"sample-weight mask length {m.shape[0]} != y {y.shape[0]}")
    y_fit = y[m] if m is not None and int(m.sum()) >= 8 else y
    classes, counts = np.unique(y_fit, return_counts=True)
    n = max(int(len(y_fit)), 1)
    freq = {c: cnt / n for c, cnt in zip(classes, counts)}
    if mode == "sqrt":
        w_of = {c: float(np.sqrt(1.0 / freq[c])) for c in classes}
    else:  # "balanced"
        k = max(len(classes), 1)
        w_of = {c: float(1.0 / (k * freq[c])) for c in classes}
    w = np.array([w_of.get(v, 1.0) for v in y], dtype="float64")
    if m is not None:
        w = np.where(m, w, 0.0)
    return w


def _as_simplex(proba: np.ndarray) -> np.ndarray:
    """LightGBM custom objectives return raw scores; softmax them to a simplex."""
    p = np.asarray(proba, dtype="float64")
    if p.ndim != 2 or p.shape[1] != 3:
        return p
    row_sum = p.sum(axis=1, keepdims=True)
    in_unit = bool(p.min() >= -1e-8 and p.max() <= 1.0 + 1e-8)
    if in_unit and np.allclose(row_sum, 1.0, atol=1e-3):
        return np.clip(p, 0.0, 1.0)
    from src.training.cost_matrix import softmax_rows
    return softmax_rows(p)


def _safe_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return (N, 3) probability array; pad if the model only knows 2 classes."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Cannot compute class probabilities or labels",
        )
        proba = model.predict_proba(X)
    if proba.shape[1] == 3:
        return _as_simplex(proba)
    # Binary model: insert a zero column for the missing class
    n = proba.shape[0]
    out = np.zeros((n, 3), dtype="float64")
    classes = list(model.classes_)
    for i, c in enumerate(classes):
        if 0 <= c <= 2:
            out[:, int(c)] = proba[:, i]
    return _as_simplex(out)


def _proba_to_canonical(
    proba: np.ndarray,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    dir_threshold: float | None = None,
) -> pd.DataFrame:
    if dir_threshold is None:
        # Default: argmax over {down, flat, up}
        idx  = proba.argmax(axis=1)
        directions = np.array([_IDX_TO_DIR[i] for i in idx], dtype=object)
        conf = proba.max(axis=1).astype("float64")
    else:
        # Directional firing threshold (A): fire up/down when its class prob
        # clears the bar, even if flat has higher probability. confidence =
        # the directional class prob (so the eval τ-gate still applies on top).
        p_down, p_flat, p_up = proba[:, 0], proba[:, 1], proba[:, 2]
        up_mask   = (p_up >= dir_threshold) & (p_up >= p_down)
        down_mask = (p_down >= dir_threshold) & (p_down > p_up) & ~up_mask
        directions = np.where(up_mask, "up", np.where(down_mask, "down", "flat")).astype(object)
        conf = np.where(up_mask, p_up, np.where(down_mask, p_down, p_flat)).astype("float64")
    df = pd.DataFrame({
        CANONICAL_TS:         pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION:  directions,
        CANONICAL_CONFIDENCE: conf,
        "p_down": proba[:, 0].astype("float64"),
        "p_flat": proba[:, 1].astype("float64"),
        "p_up":   proba[:, 2].astype("float64"),
    })
    if y_true is not None:
        df["direction_true"] = [_DIR_FROM_LABEL.get(int(y), "flat") for y in y_true]
    return df
