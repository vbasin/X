"""TCN encoder + shared hazard heads (ADR-053 Encoder A).

Same dilated-causal trunk as :class:`src.models.tcn_plugin.TCNPlugin`; the 3-class
softmax is replaced by :mod:`src.models.heads`. Fold extras ``y_up`` / ``y_down``
come from ``labels.scheme: path_vol_normalized``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import FittedModel, ModelSpec, validate_canonical
from src.models.hazard_fit import compile_on_embedding, fit_hazard_folds, predict_hazard
from src.models.tcn_plugin import tcn_embedding


class TCNHazardPlugin:
    name: str = "tcn_hazard"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        self._params = dict(spec.params or {})
        p = self._params
        n_features = int(p.get("n_features") or 1)
        return self._rebuild(n_features, None)

    def _rebuild(self, n_features: int, feature_cols: list[str] | None) -> Any:
        p = getattr(self, "_params", {})
        seq_len = int(p.get("seq_len", 128))
        inputs, emb = tcn_embedding(seq_len=seq_len, n_features=int(n_features), p=p)
        return compile_on_embedding(
            inputs, emb, include_quantile=bool(p.get("include_quantile", True)),
        )

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        p = getattr(self, "_params", {})
        return fit_hazard_folds(
            rebuild=self._rebuild,
            train_folds=train_folds,
            params=p,
            plugin_name=self.name,
            spec=ModelSpec(family=self.name, params=p),
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        df = predict_hazard(
            fitted, X, plugin_name=self.name,
            params=getattr(self, "_params", {}) or (fitted.spec.params if fitted.spec else {}),
        )
        validate_canonical(df, plugin_name=self.name)
        return df
