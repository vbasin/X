"""SVM plugin — CPU, supervised.

Uses sklearn SVC with Platt probability calibration (probability=True).
For large datasets (>50k bars) consider LinearSVC + CalibratedClassifierCV which
is significantly faster; set params_default {"kernel": "linear", "large_scale": True}
to switch to that path automatically.
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

logger = logging.getLogger("vtech.models.svm_plugin")

_IDX_TO_DIR      = {0: "down", 1: "flat", 2: "up"}
_DIR_FROM_LABEL  = {0: "down", 1: "flat", 2: "up"}


class SVMPlugin:
    name: str = "svm"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        p = dict(spec.params)
        large_scale = p.pop("large_scale", False)
        if large_scale or p.get("kernel") == "linear":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            base = LinearSVC(
                C=p.get("C", 1.0),
                max_iter=p.get("max_iter", 2000),
                class_weight=p.get("class_weight", "balanced"),
            )
            return CalibratedClassifierCV(base, cv=3, method="sigmoid")
        else:
            from sklearn.svm import SVC
            return SVC(
                C=p.get("C", 1.0),
                kernel=p.get("kernel", "rbf"),
                gamma=p.get("gamma", "scale"),
                class_weight=p.get("class_weight", "balanced"),
                probability=True,
                max_iter=p.get("max_iter", -1),
            )

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        from sklearn.base import clone as sklearn_clone
        oof_frames: list[pd.DataFrame] = []
        final_model = model
        for fold in train_folds:
            X_tr  = _flatten(fold["X_train"])
            y_tr  = fold["y_train"]
            X_oof = _flatten(fold.get("X_oof", fold["X_val"]))
            y_oof = fold.get("y_oof", fold["y_val"])
            scaler = RobustScaler()
            X_tr  = scaler.fit_transform(X_tr)
            X_oof = scaler.transform(X_oof)
            final_model = sklearn_clone(model)
            final_model.fit(X_tr, y_tr)
            proba = _safe_proba(final_model, X_oof)
            oof_frames.append(_proba_to_canonical(proba, y_true=y_oof))
        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(spec=ModelSpec(family=self.name), artifact=final_model, oof_preds=oof)

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        proba = _safe_proba(fitted.artifact, _flatten(X))
        df = _proba_to_canonical(proba)
        validate_canonical(df, plugin_name=self.name)
        return df


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


def _safe_proba(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 3:
        return proba.astype("float64")
    out = np.zeros((len(X), 3), "float64")
    for i, c in enumerate(model.classes_):
        if 0 <= int(c) <= 2:
            out[:, int(c)] = proba[:, i]
    return out


def _proba_to_canonical(
    proba: np.ndarray,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    idx  = proba.argmax(axis=1)
    conf = proba.max(axis=1).astype("float64")
    df = pd.DataFrame({
        CANONICAL_TS:         pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION:  [_IDX_TO_DIR[i] for i in idx],
        CANONICAL_CONFIDENCE: conf,
    })
    if y_true is not None:
        df["direction_true"] = [_DIR_FROM_LABEL.get(int(y), "flat") for y in y_true]
    return df
