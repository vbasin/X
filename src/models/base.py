"""Model plugin protocol — the linchpin of Phase 23.

Every model family (LSTM, TCN, Transformer, XGBoost, HMM, …) implements three
methods behind this common interface.  The search controller treats each as a
black box and the Frozen Evaluation Contract scores them identically.

Canonical prediction frame
--------------------------
``predict()`` must return a ``pandas.DataFrame`` with at minimum:

    ts          datetime64[ns, UTC]  bar timestamp
    direction   str                  "up" | "down" | "flat"
    confidence  float64              [0, 1]

Additional columns (e.g. ``size``) are allowed.

Usage example
-------------
::

    from src.models.base import ModelSpec, ModelPlugin
    from src.models.attention_lstm_plugin import AttentionLSTMPlugin

    spec   = ModelSpec(family="attention_lstm", params={"lstm_units": 64})
    plugin = AttentionLSTMPlugin()
    model  = plugin.build(spec, objective)
    fitted = plugin.fit(model, train_folds)
    preds  = plugin.predict(fitted, X_test)   # → canonical DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ── Canonical column names (all plugins must honour these) ──────────────────
CANONICAL_TS         = "ts"
CANONICAL_DIRECTION  = "direction"
CANONICAL_CONFIDENCE = "confidence"
CANONICAL_DIRECTIONS = frozenset({"up", "down", "flat"})

CANONICAL_SCHEMA: dict[str, str] = {
    CANONICAL_TS:         "datetime64[ns, UTC]",
    CANONICAL_DIRECTION:  "object",
    CANONICAL_CONFIDENCE: "float64",
}


def validate_canonical(df: pd.DataFrame, plugin_name: str = "") -> None:
    """Raise ``ValueError`` if *df* does not satisfy the canonical contract."""
    missing = [c for c in (CANONICAL_TS, CANONICAL_DIRECTION, CANONICAL_CONFIDENCE)
               if c not in df.columns]
    if missing:
        raise ValueError(f"{plugin_name}: canonical columns missing: {missing}")
    bad = set(df[CANONICAL_DIRECTION].dropna().unique()) - CANONICAL_DIRECTIONS
    if bad:
        raise ValueError(f"{plugin_name}: invalid direction values: {bad}")
    if not (df[CANONICAL_CONFIDENCE].between(0.0, 1.0, inclusive="both").all()):
        raise ValueError(f"{plugin_name}: confidence values outside [0, 1]")


# ── ModelSpec ────────────────────────────────────────────────────────────────
@dataclass
class ModelSpec:
    """Declarative description of one candidate.  Immutable after construction.

    Attributes
    ----------
    family:
        Key matching a ``catalog.py`` entry, e.g. ``"attention_lstm"``.
    params:
        Hyperparameter dict.  Plugins define allowed keys.
    meta:
        Free-form provenance bag (git_sha, image_digest, …).  Not used by plugins.
    """

    family: str
    params: dict[str, Any] = field(default_factory=dict)
    meta:   dict[str, Any] = field(default_factory=dict)


# ── Fitted model wrapper ─────────────────────────────────────────────────────
@dataclass
class FittedModel:
    """Opaque result of ``ModelPlugin.fit()``.

    Plugins store whatever they need in ``artifact``; the controller only ever
    passes a ``FittedModel`` to ``predict()``.
    """

    spec:     ModelSpec
    artifact: Any                           # the raw trained object (Keras model, sklearn, …)
    oof_preds: pd.DataFrame | None = None   # out-of-fold predictions (set by fit if walk-forward)
    meta: dict[str, Any] = field(default_factory=dict)


# ── ModelPlugin protocol ─────────────────────────────────────────────────────
@runtime_checkable
class ModelPlugin(Protocol):
    """Interface every model family must satisfy.

    All three methods are synchronous.  The search controller calls them in order
    for each candidate: ``build → fit → predict``.
    """

    name: str  # must equal the catalog family key

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        """Instantiate (but do not train) a model from *spec* + *objective*.

        Parameters
        ----------
        spec:
            Hyperparameters for this candidate.
        objective:
            The full parsed Trade Objective dict (from ``src.objectives.loader``).
            Plugins may read ``labels``, ``horizon``, ``session``, etc. to size
            their architecture or loss function.

        Returns
        -------
        Any
            An untrained model object (e.g. a Keras ``Model``, an sklearn estimator).
        """
        ...

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        """Train *model* on *train_folds* and return a ``FittedModel``.

        Parameters
        ----------
        model:
            Output of ``build()``.
        train_folds:
            List of fold dicts, each with keys ``X_train``, ``y_train``,
            ``X_val``, ``y_val``, ``X_oof``, ``y_oof`` (numpy arrays).
            Full walk-forward: every fold receives a *complete* training run.
        objective:
            Optional pass-through for loss / label construction inside the plugin.

        Returns
        -------
        FittedModel
            Wraps the trained artifact and (optionally) the OOF prediction matrix.
        """
        ...

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        """Return the canonical prediction frame for *X*.

        The returned DataFrame must satisfy ``validate_canonical()``; any plugin
        that does not will be rejected by the evaluation contract.

        Parameters
        ----------
        fitted:
            Output of ``fit()``.
        X:
            Feature array, shape ``(n_bars, *feature_dims)``.

        Returns
        -------
        pd.DataFrame
            Columns: ``ts`` (datetime64 UTC), ``direction`` ("up"/"down"/"flat"),
            ``confidence`` ([0, 1]).  Extra columns are allowed.
        """
        ...
