"""Evaluation for 3-class detection models: {BIG_DOWN=0, NO_SIGNAL=1, BIG_UP=2}.

Note (Phase 23 W0): the trade-KPI scorer lives in ``src/evaluation/__init__.py``,
which re-exports ``evaluate_model`` from here.  Import from either path — both work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logger = logging.getLogger("vtech.training.evaluate")

CLASS_NAMES = ["BIG_DOWN", "NO_SIGNAL", "BIG_UP"]


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate a 3-class detection model.

    ``y_test`` contains raw labels {0=BIG_DOWN, 1=NO_SIGNAL, 2=BIG_UP} — all bars.
    Returns standard classification metrics plus a Precision@tau sweep over
    BIG_DOWN and BIG_UP for tau in [0.50, 0.55, ..., 0.95].
    """
    if class_names is None:
        class_names = CLASS_NAMES
    num_classes = len(class_names)
    labels_list = list(range(num_classes))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)   # shape (N, 3)
    y_pred = probs.argmax(axis=1)

    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels_list, zero_division=0).tolist()
    cm = confusion_matrix(y_test, y_pred, labels=labels_list).tolist()

    report = classification_report(
        y_test, y_pred, target_names=class_names, labels=labels_list,
        output_dict=True, zero_division=0,
    )

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_f1_macro": f1_macro,
        "test_f1_per_class": dict(zip(class_names, f1_per_class)),
        "confusion_matrix": cm,
        "classification_report": report,
        "precision_at_tau": _precision_at_tau_sweep(probs, y_test),
    }
    logger.info("evaluation", extra={
        "accuracy": float(accuracy), "f1_macro": f1_macro,
    })
    return metrics


def _precision_at_tau_sweep(
    probs: np.ndarray,
    y_true: np.ndarray,
    tau_range: tuple[float, float, float] = (0.50, 0.96, 0.05),
) -> list[dict]:
    """Precision@tau sweep for BIG_DOWN (class 0) and BIG_UP (class 2).

    For each threshold tau in [0.50, 0.95], a signal is fired when the
    model's softmax probability for the class exceeds tau.  Because
    P(DOWN)+P(NEUTRAL)+P(UP)=1 and tau>=0.5, DOWN and UP signals are
    mutually exclusive by construction.

    Returns a list of records, one per (tau, direction) pair.
    """
    records = []
    taus = np.arange(*tau_range)
    for tau in taus:
        tau_val = float(round(tau, 2))
        for cls_idx, cls_name in ((0, "BIG_DOWN"), (2, "BIG_UP")):
            mask = probs[:, cls_idx] >= tau_val
            n_signals = int(mask.sum())
            if n_signals > 0:
                precision = float((y_true[mask] == cls_idx).mean())
            else:
                precision = float("nan")
            records.append({
                "tau": tau_val,
                "direction": cls_name,
                "n_signals": n_signals,
                "precision": precision,
            })
    return records


def save_metrics(metrics: dict, save_dir: str | Path, filename: str = "training_result.json") -> Path:
    """Write metrics to a JSON file."""
    path = Path(save_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path
