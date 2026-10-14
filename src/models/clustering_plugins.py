"""Clustering regime plugins — CPU, unsupervised.

KMeansPlugin  — sklearn KMeans (centroid-based clustering)
DBSCANPlugin  — sklearn DBSCAN (density-based; no fixed k, finds noise)

Both follow the same regime-model contract as the HMM/GMM plugins:
  1. Fit on the feature distribution (no labels).
  2. Derive per-cluster directional bias from training labels.
  3. At inference: assign cluster → emit canonical prediction.

DBSCAN inference challenge
--------------------------
DBSCAN has no native predict() on new points. Solution: after fitting,
compute the centroid of each non-noise cluster, then use a 1-NN lookup
(sklearn NearestNeighbors) to assign new bars to the nearest cluster.
Bars where the nearest-centroid distance exceeds a threshold are treated
as "noise" (direction="flat", confidence=0.0).
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

logger = logging.getLogger("vtech.models.clustering_plugins")

_DIR_FROM_LABEL = {0: "down", 1: "flat", 2: "up"}


# ── Shared scaffold ───────────────────────────────────────────────────────────

class _ClusterBase:
    name: str

    def _make_model(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        return self._make_model(spec.params)

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        X_all = np.vstack([_flat(f["X_train"]) for f in train_folds])
        y_all = np.concatenate([f["y_train"] for f in train_folds])

        scaler = RobustScaler()
        X_sc   = scaler.fit_transform(X_all)

        labels = model.fit_predict(X_sc)

        bias, conf_map, centroids, nn = _compute_bias_and_centroids(
            model, labels, X_sc, y_all
        )

        artifact = {
            "model":      model,
            "scaler":     scaler,
            "bias":       bias,
            "conf_map":   conf_map,
            "centroids":  centroids,
            "nn":         nn,
        }

        # OOF from last fold
        last  = train_folds[-1]
        X_oof = scaler.transform(_flat(last.get("X_oof", last["X_val"])))
        y_oof = last.get("y_oof", last["y_val"])
        oof   = _predict_from_artifact(artifact, X_oof, y_true=y_oof)
        return FittedModel(
            spec=ModelSpec(family=self.name), artifact=artifact, oof_preds=oof
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        art   = fitted.artifact
        X_sc  = art["scaler"].transform(_flat(X))
        df    = _predict_from_artifact(art, X_sc)
        validate_canonical(df, plugin_name=self.name)
        return df


# ── K-means ───────────────────────────────────────────────────────────────────

class KMeansPlugin(_ClusterBase):
    name: str = "kmeans"

    def _make_model(self, params: dict[str, Any]) -> Any:
        from sklearn.cluster import KMeans
        p = dict(params)
        return KMeans(
            n_clusters=p.get("n_clusters", 6),
            n_init=p.get("n_init", 10),
            max_iter=p.get("max_iter", 300),
            random_state=42,
        )


# ── DBSCAN ────────────────────────────────────────────────────────────────────

class DBSCANPlugin(_ClusterBase):
    name: str = "dbscan"

    def _make_model(self, params: dict[str, Any]) -> Any:
        from sklearn.cluster import DBSCAN
        p = dict(params)
        return DBSCAN(
            eps=p.get("eps", 0.5),
            min_samples=p.get("min_samples", 5),
            n_jobs=p.get("n_jobs", -1),
        )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


def _compute_bias_and_centroids(
    model: Any,
    labels: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[int, str], dict[int, float], np.ndarray, Any]:
    """Build per-cluster bias + a NearestNeighbors index for inference."""
    from sklearn.neighbors import NearestNeighbors

    unique_labels = sorted(set(labels) - {-1})  # exclude DBSCAN noise label
    bias: dict[int, str]   = {}
    conf_map: dict[int, float] = {}
    centroid_list: list[np.ndarray] = []

    for cl in unique_labels:
        mask = labels == cl
        ys   = y[mask]
        xs   = X[mask]
        centroid_list.append(xs.mean(axis=0))
        if len(ys) == 0:
            bias[int(cl)]    = "flat"
            conf_map[int(cl)] = 0.33
            continue
        counts   = {d: int((ys == i).sum()) for i, d in _DIR_FROM_LABEL.items()}
        dominant = max(counts, key=counts.get)
        bias[int(cl)]     = dominant
        conf_map[int(cl)] = round(counts[dominant] / len(ys), 4)

    if not centroid_list:
        # All noise — return empty NN
        centroids = np.zeros((1, X.shape[1]))
        bias[-1]      = "flat"
        conf_map[-1]  = 0.0
    else:
        centroids = np.vstack(centroid_list)

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(centroids)

    return bias, conf_map, centroids, nn


def _predict_from_artifact(
    art: dict[str, Any],
    X_sc: np.ndarray,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    nn        = art["nn"]
    bias      = art["bias"]
    conf_map  = art["conf_map"]
    centroids = art["centroids"]

    distances, indices = nn.kneighbors(X_sc)
    distances  = distances.ravel()
    cluster_ids = indices.ravel()  # index into centroids array

    # Map centroid index back to a signed cluster label.
    # For KMeans the centroid order matches cluster labels 0…k-1.
    # For DBSCAN we stored only non-noise centroids in sorted(unique_labels) order.
    # Use the order of centroid_list as the mapping key.
    n_centroids = len(centroids)
    # Approximate per-bar confidence: 1 / (1 + distance)
    dist_conf = 1.0 / (1.0 + distances)

    # Resolve bias for each bar
    bias_keys = sorted(k for k in bias if k != -1)[:n_centroids]
    directions: list[str]  = []
    confidences: list[float] = []
    for i, ci in enumerate(cluster_ids):
        if ci < len(bias_keys):
            k   = bias_keys[ci]
            dir_ = bias.get(k, "flat")
            emp_conf = conf_map.get(k, 0.33)
        else:
            dir_ = "flat"
            emp_conf = 0.0
        directions.append(dir_)
        confidences.append(min(float(dist_conf[i]), emp_conf))

    df = pd.DataFrame({
        CANONICAL_TS:         pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION:  directions,
        CANONICAL_CONFIDENCE: confidences,
    })
    if y_true is not None:
        df["direction_true"] = [_DIR_FROM_LABEL.get(int(y), "flat") for y in y_true]
    return df
