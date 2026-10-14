"""Primary + meta trade/no-trade composition (López de Prado) — ADR-027 / W3.

Given a primary model's canonical OOF (direction + confidence + direction_true),
train a **cross-fitted** binary meta-gate that predicts whether acting on the
primary side would have won under the objective's labels. A composed prediction
fires only when both:

  1. the primary proposes a side with confidence ≥ ``min_primary_confidence``
  2. the meta model assigns P(win) ≥ ``approve_threshold``

Leakage guard: meta predictions for each row come from a model trained on the
*other* folds only (sklearn KFold, time-ordered splits). Never fit the meta
model on the same rows it is scored on.

Context features (optional) come from a named feature view joined by timestamp
onto the OOF — typically ``regime_core`` so the meta layer sees vol/trend state
beyond the primary's confidence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.training.meta_labels import generate_meta_labels

logger = logging.getLogger("vtech.search.meta_compose")

_DEFAULT_META = {
    "family": "xgboost",
    "context_view": None,
    "approve_threshold": 0.55,
    "min_primary_confidence": 0.50,
    "n_folds": 5,
}


def topology_cfg(objective: dict[str, Any]) -> dict[str, Any]:
    """Return the effective meta topology config (empty dict if mode is not primary_meta)."""
    topo = (objective.get("search") or {}).get("topology") or {}
    if topo.get("mode") != "primary_meta":
        return {}
    meta = {**_DEFAULT_META, **(topo.get("meta") or {})}
    return meta


def compose_primary_meta(
    primary_oof: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    approve_threshold: float = 0.55,
    min_primary_confidence: float = 0.50,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cross-fit a binary meta-gate and return a composed canonical OOF frame.

    Parameters
    ----------
    primary_oof:
        Columns ``ts``, ``direction``, ``confidence``, and usually
        ``direction_true`` (required for meta labels + scoring).
    context:
        Optional feature frame indexed by ``ts`` (or with a ``ts`` column).
        Joined left onto directional primary rows.
    approve_threshold / min_primary_confidence / n_folds:
        See module docstring.

    Returns
    -------
    pd.DataFrame
        Same rows as *primary_oof*, with ``direction`` set to ``flat`` where the
        meta gate rejects (or primary was already flat / below confidence).
        ``confidence`` becomes the meta P(win) on approved rows (else 0).
        Extra columns ``meta_prob`` and ``primary_direction`` are retained for
        diagnostics.
    """
    if primary_oof is None or len(primary_oof) == 0:
        raise ValueError("primary_oof is empty")
    need = {"ts", "direction", "confidence"}
    missing = need - set(primary_oof.columns)
    if missing:
        raise ValueError(f"primary_oof missing columns: {sorted(missing)}")
    if "direction_true" not in primary_oof.columns:
        raise ValueError("primary_oof needs direction_true to build meta labels")

    oof = primary_oof.copy()
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    oof = oof.sort_values("ts").reset_index(drop=True)

    # Map direction_true → raw labels for generate_meta_labels.
    _true_to_raw = {"down": 0, "flat": 1, "up": 2}
    raw = oof["direction_true"].map(_true_to_raw).astype("float")
    if raw.isna().any():
        raise ValueError("direction_true contains values outside {up,down,flat}")
    raw_s = pd.Series(raw.to_numpy(dtype=np.int8), index=oof.index, name="raw")
    meta_y = generate_meta_labels(oof["direction"], raw_s)

    # Training universe: directional primary proposals above min confidence.
    conf = oof["confidence"].to_numpy(dtype=float)
    side = oof["direction"].astype(str).to_numpy()
    train_mask = (
        np.isin(side, ("up", "down"))
        & (conf >= float(min_primary_confidence))
        & np.isfinite(conf)
    )
    if int(train_mask.sum()) < max(20, n_folds * 4):
        logger.warning(
            "meta_compose: too few primary proposals (%d) — returning all-flat gate",
            int(train_mask.sum()),
        )
        return _all_flat(oof)

    X, _names, _impute = _design_matrix(oof, train_mask, context)
    y = meta_y.to_numpy(dtype=np.int8)[train_mask]
    meta_prob_all = np.zeros(len(oof), dtype=float)

    # Time-ordered K-fold: contiguous blocks so later bars are never training on future.
    idx = np.where(train_mask)[0]
    folds = _time_folds(len(idx), n_folds=n_folds)
    for te_local, tr_local in folds:
        tr_idx = idx[tr_local]
        te_idx = idx[te_local]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        # Map local positions in the train_mask-reduced X back via range indices.
        # X rows correspond 1:1 with idx order.
        pos = {int(i): p for p, i in enumerate(idx)}
        tr_pos = [pos[int(i)] for i in tr_idx]
        te_pos = [pos[int(i)] for i in te_idx]
        clf = _fit_binary(X[tr_pos], y[tr_pos], random_state=random_state)
        meta_prob_all[te_idx] = _predict_proba(clf, X[te_pos])

    # Compose: meta is a hard trade/no-trade filter. Keep the primary's side AND
    # its confidence when approved so the objective's τ-gate still works (meta_prob
    # alone typically peaks ~0.5–0.6 and would zero out all fires at τ=0.8).
    out = oof.copy()
    out["primary_direction"] = out["direction"]
    out["primary_confidence"] = conf
    out["meta_prob"] = meta_prob_all
    approved = train_mask & (meta_prob_all >= float(approve_threshold))
    new_dir = np.where(approved, side, "flat")
    new_conf = np.where(approved, conf, 0.0)
    out["direction"] = new_dir
    out["confidence"] = new_conf
    logger.info(
        "meta_compose: proposals=%d approved=%d rate=%.3f thr=%.2f",
        int(train_mask.sum()), int(approved.sum()),
        float(approved.sum() / max(train_mask.sum(), 1)),
        float(approve_threshold),
    )
    return out


def fit_final_primary_meta(
    primary_oof: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    min_primary_confidence: float = 0.50,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit one binary gate on all eligible primary fires (serve-time weights).

    Cross-fitted OOF scores stay in :func:`compose_primary_meta`. This refit is
    what Custom Test persists under ``models/meta__<primary>/``.
    """
    if primary_oof is None or len(primary_oof) == 0:
        return _constant_meta(0.0)
    oof = primary_oof.copy()
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    oof = oof.sort_values("ts").reset_index(drop=True)
    if "direction_true" not in oof.columns:
        return _constant_meta(0.0)

    _true_to_raw = {"down": 0, "flat": 1, "up": 2}
    raw = oof["direction_true"].map(_true_to_raw).astype("float")
    if raw.isna().any():
        return _constant_meta(0.0)
    raw_s = pd.Series(raw.to_numpy(dtype=np.int8), index=oof.index, name="raw")
    meta_y = generate_meta_labels(oof["direction"], raw_s)
    conf = oof["confidence"].to_numpy(dtype=float)
    side = oof["direction"].astype(str).to_numpy()
    train_mask = (
        np.isin(side, ("up", "down"))
        & (conf >= float(min_primary_confidence))
        & np.isfinite(conf)
    )
    if int(train_mask.sum()) < 20:
        return _constant_meta(0.0)
    X, names, impute = _design_matrix(oof, train_mask, context)
    y = meta_y.to_numpy(dtype=np.int8)[train_mask]
    kind, est = _fit_binary(X, y, random_state=random_state)
    return {
        "kind": kind,
        "estimator": est,
        "feature_names": names,
        "impute_values": impute,
        "n_train": int(train_mask.sum()),
    }


def _constant_meta(p: float) -> dict[str, Any]:
    return {
        "kind": "constant",
        "estimator": float(p),
        "feature_names": ["primary_confidence", "is_up"],
        "impute_values": {},
        "n_train": 0,
    }


def predict_meta_p_win(fitted: Any, X: np.ndarray) -> np.ndarray:
    """P(win) from a :func:`fit_final_primary_meta` payload or a raw estimator."""
    if isinstance(fitted, dict) and "kind" in fitted:
        return _predict_proba((fitted["kind"], fitted["estimator"]), X)
    if isinstance(fitted, tuple) and len(fitted) == 2:
        return _predict_proba(fitted, X)
    if hasattr(fitted, "predict_proba"):
        proba = np.asarray(fitted.predict_proba(X))
        classes = list(getattr(fitted, "classes_", [0, 1]))
        if 1 in classes:
            return proba[:, classes.index(1)].astype(float)
        return proba[:, -1].astype(float)
    return np.zeros(len(X), dtype=float)


def meta_design_from_scored(
    *,
    direction: np.ndarray,
    confidence: np.ndarray,
    ts: pd.DatetimeIndex,
    context: pd.DataFrame | None,
    feature_names: list[str],
    impute_values: dict[str, float] | None = None,
) -> np.ndarray:
    """Serve-time design matrix aligned with :func:`_design_matrix` columns."""
    conf = np.asarray(confidence, dtype=float).reshape(-1, 1)
    is_up = (np.asarray(direction).astype(str) == "up").astype(float).reshape(-1, 1)
    colmap: dict[str, np.ndarray] = {
        "primary_confidence": conf[:, 0],
        "is_up": is_up[:, 0],
    }
    impute_values = impute_values or {}
    if context is not None and len(context) and len(ts):
        ctx = context.copy()
        if "ts" in ctx.columns:
            ctx["ts"] = pd.to_datetime(ctx["ts"], utc=True)
            ctx = ctx.set_index("ts")
        if ctx.index.tz is None:
            ctx.index = ctx.index.tz_localize("UTC")
        else:
            ctx.index = ctx.index.tz_convert("UTC")
        num = ctx.select_dtypes(include=[np.number])
        joined = num.reindex(pd.DatetimeIndex(ts))
        for c in joined.columns:
            arr = joined[c].to_numpy(dtype=float)
            med = float(impute_values.get(c, np.nanmedian(arr) if np.isfinite(arr).any() else 0.0))
            arr[~np.isfinite(arr)] = med
            colmap[str(c)] = arr
    n = len(confidence)
    cols = []
    for name in feature_names:
        cols.append(colmap.get(name, np.zeros(n, dtype=float)))
    if not cols:
        cols = [conf[:, 0], is_up[:, 0]]
    return np.column_stack(cols).astype(np.float32)


def _all_flat(oof: pd.DataFrame) -> pd.DataFrame:
    out = oof.copy()
    out["primary_direction"] = out["direction"]
    out["primary_confidence"] = out["confidence"]
    out["meta_prob"] = 0.0
    out["direction"] = "flat"
    out["confidence"] = 0.0
    return out


def _design_matrix(
    oof: pd.DataFrame,
    train_mask: np.ndarray,
    context: pd.DataFrame | None,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Build meta features for rows where *train_mask* is True (row order = oof order filtered)."""
    conf = oof.loc[train_mask, "confidence"].to_numpy(dtype=float).reshape(-1, 1)
    side = oof.loc[train_mask, "direction"].astype(str).to_numpy()
    is_up = (side == "up").astype(float).reshape(-1, 1)
    parts = [conf, is_up]
    names = ["primary_confidence", "is_up"]
    impute: dict[str, float] = {}

    if context is not None and len(context):
        ctx = context.copy()
        if "ts" in ctx.columns:
            ctx["ts"] = pd.to_datetime(ctx["ts"], utc=True)
            ctx = ctx.set_index("ts")
        elif not isinstance(ctx.index, pd.DatetimeIndex):
            raise ValueError("context must be ts-indexed or have a ts column")
        if ctx.index.tz is None:
            ctx.index = ctx.index.tz_localize("UTC")
        else:
            ctx.index = ctx.index.tz_convert("UTC")
        # Numeric only; drop all-NaN columns.
        num = ctx.select_dtypes(include=[np.number])
        joined = num.reindex(oof.loc[train_mask, "ts"].to_numpy())
        if joined.shape[1]:
            arr = joined.to_numpy(dtype=float)
            # Column-wise median impute (meta must not drop rows for sparse context).
            for j, col_name in enumerate(joined.columns):
                col = arr[:, j]
                med = float(np.nanmedian(col) if np.isfinite(col).any() else 0.0)
                col[~np.isfinite(col)] = med
                arr[:, j] = col
                names.append(str(col_name))
                impute[str(col_name)] = med
            parts.append(arr)

    return np.hstack(parts).astype(np.float32), names, impute


def _time_folds(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous time folds: each fold is (test_local_idx, train_local_idx)."""
    n_folds = max(2, min(int(n_folds), n // 4 if n >= 8 else 2))
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n)
    for k in range(n_folds):
        te = all_idx[edges[k]:edges[k + 1]]
        tr = np.concatenate([all_idx[:edges[k]], all_idx[edges[k + 1]:]])
        if len(te) and len(tr):
            folds.append((te, tr))
    return folds


def _fit_binary(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> Any:
    """Fit a small binary classifier. Prefer XGBoost; fall back to logistic regression."""
    # Degenerate: single class → constant predictor.
    if len(np.unique(y)) < 2:
        return ("constant", float(y.mean()) if len(y) else 0.0)
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=1, random_state=random_state, verbosity=0,
        )
        clf.fit(X, y)
        return ("xgb", clf)
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta xgboost unavailable (%s) — logistic fallback", exc)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=random_state)
        clf.fit(X, y)
        return ("sklearn", clf)


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    kind, obj = model
    if kind == "constant":
        return np.full(len(X), float(obj), dtype=float)
    if kind == "xgb":
        # XGBClassifier.predict_proba
        return obj.predict_proba(X)[:, 1].astype(float)
    proba = obj.predict_proba(X)
    # Positive class column.
    classes = list(obj.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)].astype(float)
    return proba[:, -1].astype(float)
