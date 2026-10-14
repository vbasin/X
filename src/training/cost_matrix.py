"""3-class expected-cost train loss from the Trade Objective YAML (ADR-072).

When ``custom_optimization: true``, 3-class Keras / XGBoost / LightGBM heads
minimize ``E[cost] = C[true_class] · p`` instead of cross-entropy. ``C`` is the
YAML ``cost_matrix`` (rows = true class, cols = predicted class; lower is
better). Other families keep CE. Sample-weight ``class_balance`` is skipped
while this loss is active — the matrix already encodes class importance.

Serving is unchanged: argmax of the 3-class softmax.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("vtech.training.cost_matrix")

LABELS: tuple[str, ...] = ("down", "flat", "up")
N_CLASS = 3
_HESS_FLOOR = 1e-6


def custom_optimization_enabled(objective: dict[str, Any] | None) -> bool:
    """True only when the YAML opts in with ``custom_optimization: true``."""
    return (objective or {}).get("custom_optimization") is True


def cost_matrix_array(objective: dict[str, Any] | None) -> np.ndarray | None:
    """Return the 3×3 cost matrix, or ``None`` when the switch is off.

    Raises
    ------
    ValueError
        If the flag is true but the matrix is missing / malformed. The
        objective loader should have caught this; this is a second fence
        for callers that skip the loader.
    """
    if not custom_optimization_enabled(objective):
        return None
    return parse_cost_matrix((objective or {}).get("cost_matrix"))


def parse_cost_matrix(block: dict[str, Any] | None) -> np.ndarray:
    """Parse ``cost_matrix.costs`` into a (3, 3) float array in down/flat/up order."""
    if not isinstance(block, dict):
        raise ValueError(
            "custom_optimization: true requires a cost_matrix block "
            "(rows = true class, cols = predicted class)"
        )
    costs = block.get("costs")
    if not isinstance(costs, dict):
        raise ValueError("cost_matrix.costs must be a mapping of true-class rows")
    labels = list(block.get("labels") or LABELS)
    if [str(x) for x in labels] != list(LABELS):
        raise ValueError(
            f"cost_matrix.labels must be {list(LABELS)} in that order; got {labels}"
        )
    out = np.zeros((N_CLASS, N_CLASS), dtype=np.float64)
    for i, true_lab in enumerate(LABELS):
        if true_lab not in costs:
            raise ValueError(f"cost_matrix.costs missing row '{true_lab}'")
        out[i] = _parse_row(costs[true_lab], true_lab)
    if not np.isfinite(out).all():
        raise ValueError("cost_matrix.costs must be finite numbers")
    return out


def _parse_row(row: Any, true_lab: str) -> np.ndarray:
    if isinstance(row, dict):
        missing = [k for k in LABELS if k not in row]
        if missing:
            raise ValueError(
                f"cost_matrix.costs.{true_lab} missing keys {missing}"
            )
        return np.array([float(row[k]) for k in LABELS], dtype=np.float64)
    if isinstance(row, (list, tuple)) and len(row) == N_CLASS:
        return np.array([float(x) for x in row], dtype=np.float64)
    raise ValueError(
        f"cost_matrix.costs.{true_lab} must be a 3-vector or "
        f"{{{', '.join(LABELS)}}} mapping"
    )


def expected_cost_per_row(proba: np.ndarray, y: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """``(C[y] * p).sum(axis=1)`` — the quantity Keras / trees minimize."""
    p = np.asarray(proba, dtype=np.float64)
    yy = np.asarray(y, dtype=int).ravel()
    return (costs[yy] * p).sum(axis=1)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def expected_cost_grad_hess(
    logits: np.ndarray,
    y: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax expected-cost gradient and a positive Hessian floor (trees)."""
    p = softmax_rows(logits)
    yy = np.asarray(y, dtype=int).ravel()
    rows = costs[yy]
    mu = (rows * p).sum(axis=1, keepdims=True)
    grad = p * (rows - mu)
    hess = p * (1.0 - p) * np.maximum(np.abs(rows - mu), _HESS_FLOOR)
    hess = np.maximum(hess, _HESS_FLOOR)
    return grad, hess


class ExpectedCostXgbObj:
    """Picklable XGBoost custom objective (sklearn *or* native DMatrix call)."""

    def __init__(self, costs: list[list[float]] | np.ndarray) -> None:
        self.costs = np.asarray(costs, dtype=np.float64)
        if self.costs.shape != (N_CLASS, N_CLASS):
            raise ValueError(f"costs must be 3x3, got {self.costs.shape}")

    def __call__(self, a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
        y, logits = _split_xgb_args(a, b)
        n = y.size
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(n, N_CLASS)
        elif logits.shape[0] != n:
            logits = logits.reshape(n, N_CLASS)
        return expected_cost_grad_hess(logits, y, self.costs)


class ExpectedCostLgbObj:
    """Picklable LightGBM custom objective. ``y_pred`` is ``(n * 3,)`` logits."""

    def __init__(self, costs: list[list[float]] | np.ndarray) -> None:
        self.costs = np.asarray(costs, dtype=np.float64)
        if self.costs.shape != (N_CLASS, N_CLASS):
            raise ValueError(f"costs must be 3x3, got {self.costs.shape}")

    def __call__(self, y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_true, dtype=int).ravel()
        n = y.size
        logits = np.asarray(y_pred, dtype=np.float64).reshape(n, N_CLASS)
        grad, hess = expected_cost_grad_hess(logits, y, self.costs)
        return grad.reshape(-1), hess.reshape(-1)


def _split_xgb_args(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    """Accept ``(predt, DMatrix)`` or sklearn ``(y_true, y_pred)``."""
    if hasattr(b, "get_label"):
        return np.asarray(b.get_label(), dtype=int).ravel(), np.asarray(a)
    if hasattr(a, "get_label"):
        return np.asarray(a.get_label(), dtype=int).ravel(), np.asarray(b)
    return np.asarray(a, dtype=int).ravel(), np.asarray(b)


def keras_compile_loss(objective: dict[str, Any] | None) -> Any:
    """Keras ``compile(loss=...)`` value for a 3-class softmax head."""
    C = cost_matrix_array(objective)
    if C is None:
        return "sparse_categorical_crossentropy"
    from src.models.keras_layers import ExpectedCostLoss
    return ExpectedCostLoss(costs=C.tolist())


def apply_tree_custom_objective(
    params: dict[str, Any],
    objective: dict[str, Any] | None,
    *,
    family: str,
) -> dict[str, Any]:
    """Inject a 3-class expected-cost obj into XGB / LGBM constructor kwargs."""
    C = cost_matrix_array(objective)
    if C is None:
        return params
    out = dict(params)
    if family == "xgboost":
        out["objective"] = ExpectedCostXgbObj(C)
        out["num_class"] = N_CLASS
        out.setdefault("eval_metric", "mlogloss")
    elif family == "lightgbm":
        out["objective"] = ExpectedCostLgbObj(C)
        out["num_class"] = N_CLASS
        out.setdefault("verbosity", -1)
    else:
        logger.warning(
            "custom_optimization ignored for family=%s (no 3-class cost-matrix obj)",
            family,
        )
    return out
