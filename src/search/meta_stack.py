"""Meta-stack — binary LR on signed base evidence (walk-forward or cross-fit).

Semantics distinct from ``output_meta``:

* ``output_meta`` — bases resolve a shared side (unanimous directional agreement);
  meta says APPROVE / REJECT for that side. Meta cannot override the direction.
* ``meta_stack`` — each base contributes a signed scalar ``s_i`` (sign = direction,
  magnitude = confidence). A binary LR learns per-base weights and a bias:

      z = w_xgb · s_xgb + w_tcn · s_tcn + b
      p_up = sigmoid(z)                   # P(true direction is up)

  Fire rule (symmetric around 0.5, threshold in [0.5, 1.0]):

      if p_up >= approve_threshold:              direction=up,   confidence=p_up
      elif p_up <= 1 - approve_threshold:        direction=down, confidence=1-p_up
      else:                                      direction=flat, confidence=0

  Meta target: y = 1 if direction_true == "up" ; 0 if "down" ; MASKED if "flat".

Signed feature options:

* ``linear_weights`` (default) — ``s = p_up - p_down``, in [-1, +1].
* ``log_weights``              — ``s = log((p_up + eps) / (p_down + eps))`` (log-odds
  contributed by each base). Enables Bayesian log-linear pooling: the LR's linear
  combination of log-odds is the optimal aggregate under a naive-Bayes independence
  assumption between bases.

OOF prediction schemes:

* ``walk_forward_OOF`` (default) — Split concatenated OOF into ``n_folds`` contiguous
  time blocks. Block 0 is warmup (no meta fires). For k=1..n_folds-1: fit LR on
  rows [0, edges[k] - embargo); predict block k. Causally identical to live: every
  prediction uses only past labels. This is the "learns and updates its weights as
  more OOF arrives" scheme.
* ``cross_fit`` — 5-block time cross-fit with symmetric embargo (matches
  ``output_meta``). Uses more data per prediction but a fold-0 row can be scored by
  a model that saw fold-2 data. Standard ML literature, weaker live-parity guarantee.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.search.output_meta import _base_vector, align_on_ts

logger = logging.getLogger("vtech.search.meta_stack")

_EPS = 1e-6

_DEFAULT_META: dict[str, Any] = {
    "family": "logistic_regression",
    "scheme": "walk_forward_OOF",   # 'walk_forward_OOF' | 'cross_fit'
    "feature": "linear_weights",    # 'linear_weights' (p_up-p_down) | 'log_weights' (log-odds)
    "approve_threshold": 0.55,
    "embargo_bars": 0,
    "n_folds": 5,
}


def topology_meta_stack(objective: dict[str, Any]) -> dict[str, Any] | None:
    """Return ``{bases, meta}`` or None if mode is not ``meta_stack``."""
    topo = (objective.get("search") or {}).get("topology") or {}
    if topo.get("mode") != "meta_stack":
        return None
    bases = [str(b) for b in (topo.get("bases") or []) if str(b).strip()]
    if not bases:
        return None
    meta = {**_DEFAULT_META, **(topo.get("meta") or {})}
    return {"bases": bases, "meta": meta}


# ── Features ──────────────────────────────────────────────────────────────────

def signed_features(
    aligned: pd.DataFrame, bases: list[str], feature: str,
) -> tuple[np.ndarray, list[str]]:
    """Per-base signed scalar. One column per base, shape (n, |bases|)."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for b in bases:
        p_down, _p_flat, p_up, _conf, _is_up, _is_dn = _base_vector(aligned, b)
        if feature == "log_weights":
            s = np.log((p_up + _EPS) / (p_down + _EPS))
        else:  # linear_weights (default) — bounded in [-1, +1]
            s = p_up - p_down
        cols.append(s.astype("float64"))
        names.append(f"{b}__s_{feature}")
    return np.column_stack(cols), names


# ── LR fit / predict ──────────────────────────────────────────────────────────

def _fit_binary_lr(X: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X, y)
    return clf


def _predict_p_up(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


# ── OOF schemes ───────────────────────────────────────────────────────────────

def _walk_forward_oof(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray, *, n_folds: int, embargo: int,
) -> np.ndarray:
    """Sequential: fit on rows [0, edges[k]-embargo), predict block k. Block 0 = warmup."""
    n = len(X)
    n_folds = max(2, int(n_folds))
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    proba = np.full(n, np.nan, dtype="float64")
    emb = max(0, int(embargo))
    for k in range(1, n_folds):
        te_lo, te_hi = int(edges[k]), int(edges[k + 1])
        tr_hi = max(0, te_lo - emb)
        tr_idx = np.where(mask[:tr_hi])[0]
        if len(tr_idx) < 20 or len(np.unique(y[tr_idx])) < 2:
            logger.warning(
                "meta_stack walk_forward_OOF block=%d skipped: %d training rows / %d classes",
                k, len(tr_idx), len(np.unique(y[tr_idx])),
            )
            continue
        clf = _fit_binary_lr(X[tr_idx], y[tr_idx])
        proba[te_lo:te_hi] = _predict_p_up(clf, X[te_lo:te_hi])
    return proba


def _cross_fit_oof(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray, *, n_folds: int, embargo: int,
) -> np.ndarray:
    """Time-block cross-fit with symmetric embargo (mirrors output_meta._embargo_time_folds)."""
    n = len(X)
    n_folds = max(2, min(int(n_folds), n // 4 if n >= 8 else 2))
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    proba = np.full(n, np.nan, dtype="float64")
    emb = max(0, int(embargo))
    for k in range(n_folds):
        te_lo, te_hi = int(edges[k]), int(edges[k + 1])
        lo = max(0, te_lo - emb)
        hi = min(n, te_hi + emb)
        tr_rng = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        tr_idx = tr_rng[mask[tr_rng]]
        if len(tr_idx) < 20 or len(np.unique(y[tr_idx])) < 2:
            continue
        clf = _fit_binary_lr(X[tr_idx], y[tr_idx])
        proba[te_lo:te_hi] = _predict_p_up(clf, X[te_lo:te_hi])
    return proba


# ── Public API ────────────────────────────────────────────────────────────────

def compose(
    base_oofs: dict[str, pd.DataFrame],
    *,
    bases: list[str],
    scheme: str = "walk_forward_OOF",
    feature: str = "linear_weights",
    approve_threshold: float = 0.55,
    embargo_bars: int = 0,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Cross-fit or walk-forward binary LR on signed base evidence.

    Returns canonical OOF frame ``ts, direction, confidence, direction_true``,
    plus diagnostic columns ``meta_p_up`` and ``meta_finite`` (False for warmup /
    degenerate blocks). Attaches ``attrs['meta_feature_names']`` for logging.
    """
    aligned = align_on_ts(base_oofs, bases)
    if aligned.empty:
        raise ValueError("meta_stack: no overlapping ts across bases")
    if "direction_true" not in aligned.columns:
        raise ValueError("meta_stack: base OOFs missing direction_true")

    X, feat_names = signed_features(aligned, bases, feature)

    truth = np.asarray(aligned["direction_true"].astype(str))
    # y=1 for up, 0 for down, -1 (masked) for flat/other.
    y = np.where(truth == "up", 1, np.where(truth == "down", 0, -1)).astype(np.int8)
    train_mask = y != -1

    if int(train_mask.sum()) < max(20, int(n_folds) * 4):
        logger.warning(
            "meta_stack: only %d directional labels — returning all-flat OOF",
            int(train_mask.sum()),
        )
        return _all_flat(aligned)

    if scheme == "cross_fit":
        proba = _cross_fit_oof(X, y, train_mask, n_folds=int(n_folds), embargo=int(embargo_bars))
    else:  # walk_forward_OOF — default
        proba = _walk_forward_oof(X, y, train_mask, n_folds=int(n_folds), embargo=int(embargo_bars))

    finite = np.isfinite(proba)
    p_up = np.where(finite, proba, 0.5)              # warmup / degenerate → neutral (no fire)
    thr = float(approve_threshold)
    approved_up = finite & (p_up >= thr)
    approved_dn = finite & (p_up <= 1.0 - thr)

    direction = np.where(approved_up, "up",
                np.where(approved_dn, "down", "flat")).astype(object)
    confidence = np.where(approved_up, p_up,
                 np.where(approved_dn, 1.0 - p_up, 0.0)).astype("float64")

    out = pd.DataFrame({
        "ts": aligned["ts"].to_numpy(),
        "direction": direction,
        "confidence": confidence,
        "meta_p_up": p_up,
        "meta_finite": finite,
        "direction_true": truth,
    })
    out.attrs["meta_feature_names"] = feat_names
    n_up = int(np.sum(direction == "up"))
    n_dn = int(np.sum(direction == "down"))
    logger.info(
        "meta_stack scheme=%s feature=%s n=%d fired_up=%d fired_down=%d thr=%.2f",
        scheme, feature, len(out), n_up, n_dn, thr,
    )
    return out


def fit_final(
    base_oofs: dict[str, pd.DataFrame],
    *,
    bases: list[str],
    feature: str = "linear_weights",
) -> dict[str, Any]:
    """Fit one LR on all directional-label rows (used by finalize)."""
    aligned = align_on_ts(base_oofs, bases)
    if "direction_true" not in aligned.columns:
        raise ValueError("meta_stack finalize: base OOFs missing direction_true")
    X, feat_names = signed_features(aligned, bases, feature)
    truth = np.asarray(aligned["direction_true"].astype(str))
    y = np.where(truth == "up", 1, np.where(truth == "down", 0, -1)).astype(np.int8)
    mask = y != -1
    if int(mask.sum()) < 8 or len(np.unique(y[mask])) < 2:
        raise ValueError("meta_stack finalize: not enough directional labels to fit LR")
    clf = _fit_binary_lr(X[mask], y[mask])
    weights = {name: float(w) for name, w in zip(feat_names, clf.coef_.ravel())}
    return {
        "estimator": clf,
        "feature_names": feat_names,
        "family": "logistic_regression",
        "bases": list(bases),
        "feature": feature,
        "weights": weights,
        "bias": float(clf.intercept_[0]),
    }


def _all_flat(aligned: pd.DataFrame) -> pd.DataFrame:
    n = len(aligned)
    truth = (aligned["direction_true"].astype(str).to_numpy()
             if "direction_true" in aligned.columns else np.array([None] * n))
    return pd.DataFrame({
        "ts": aligned["ts"].to_numpy(),
        "direction": np.full(n, "flat", dtype=object),
        "confidence": np.zeros(n, dtype="float64"),
        "meta_p_up": np.full(n, 0.5, dtype="float64"),
        "meta_finite": np.zeros(n, dtype=bool),
        "direction_true": truth,
    })
