"""Preregistered quality gates for the small synthetic classification service.

This objective measures deployment readiness for its explicit synthetic task.
It does not assert market edge or replace any trading promotion policy. Gate
thresholds are frozen before training, and the final test is scored only once.
"""
from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiment.workflow_data import classification_metrics, validate_probabilities

OBJECTIVE_ID = "synthetic-regime-service"
GATE_KEYS = frozenset({
    "min_accuracy", "min_macro_f1", "min_class_recall", "min_rows",
    "min_class_rows", "min_prior_log_loss_improvement",
    "persistence_accuracy_margin", "max_log_loss", "max_runtime_seconds",
})


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete gate schema; unknown or missing settings fail closed."""
    if not isinstance(policy, Mapping) or set(policy) != GATE_KEYS:
        raise ValueError("gate policy must contain exactly the supported gate keys")
    result = dict(policy)
    integer_bounds = {
        "min_rows": (1, 10000),
        "min_class_rows": (1, 10000),
        "max_runtime_seconds": (1, 1740),
    }
    for name, (low, high) in integer_bounds.items():
        value = result[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if result["min_class_rows"] * 3 > result["min_rows"]:
        raise ValueError("min_rows must accommodate the three minimum class counts")
    for name in GATE_KEYS - integer_bounds.keys():
        value = result[name]
        upper = 10.0 if name == "max_log_loss" else 1.0
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= upper):
            raise ValueError(f"{name} must be finite in [0, {upper}]")
    if result["max_log_loss"] <= 0:
        raise ValueError("max_log_loss must be positive")
    return result


def load_objective(path: Path) -> dict[str, Any]:
    """Read a complete, explicit non-trading objective and its training recipe."""
    from src.experiment.transformer_workflow import WorkflowConfig

    value = json.loads(Path(path).read_text())
    expected = {"objective_id", "revision", "purpose", "synthetic_only", "gates", "model_config"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("objective must contain exactly the supported objective keys")
    if value["objective_id"] != OBJECTIVE_ID or value["synthetic_only"] is not True:
        raise ValueError("only the explicit synthetic-regime-service objective is supported")
    if type(value["revision"]) is not int or value["revision"] < 1:
        raise ValueError("objective revision must be a positive integer")
    if not isinstance(value["purpose"], str) or not value["purpose"].strip():
        raise ValueError("objective purpose must be a nonempty string")
    value["gates"] = validate_policy(value["gates"])
    config = value["model_config"]
    expected_config = {field.name for field in fields(WorkflowConfig)} | {"objective_id"}
    if not isinstance(config, dict) or set(config) != expected_config:
        raise ValueError("model_config must explicitly resolve every WorkflowConfig field")
    if config["objective_id"] != OBJECTIVE_ID:
        raise ValueError("model_config objective_id must match the evaluation objective")
    # The recipe has one owner. This loader adds objective consistency checks,
    # while the existing runner validates architecture/training limits.
    recipe = {key: val for key, val in config.items() if key != "objective_id"}
    recipe["layers"] = tuple(recipe["layers"])
    WorkflowConfig(**recipe)
    if config["max_runtime_seconds"] != value["gates"]["max_runtime_seconds"]:
        raise ValueError("model and gate runtime budgets must match")
    if config["min_accuracy"] != value["gates"]["min_accuracy"]:
        raise ValueError("model and gate accuracy floors must match")
    return value


def _labels(labels: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(labels)
    if (values.ndim != 1 or len(values) != n_rows or values.dtype.kind == "b"
            or not np.isin(values, (0, 1, 2)).all()):
        raise ValueError("labels must contain one class in {0, 1, 2} per prediction")
    return values.astype(np.int64)


def _sensor_bins(frame: pd.DataFrame) -> np.ndarray:
    if "wf_context" not in frame or len(frame) == 0:
        raise ValueError("persistence baseline requires nonempty wf_context")
    sensor = frame["wf_context"].to_numpy(dtype=float)
    if sensor.ndim != 1 or not np.isfinite(sensor).all():
        raise ValueError("wf_context must be a finite vector")
    return np.digitize(sensor, (-0.5, 0.5))


def fit_persistence_baseline(train_frame: pd.DataFrame) -> np.ndarray:
    """Fit P(future class | current sensor bin) on training labels only.

    The observable sensor has fixed down/flat/up bins at -0.5 and +0.5.
    One additive count per class avoids zero probabilities. No thresholds or
    statistics are estimated from calibration, validation or final-test rows.
    """
    if "label" not in train_frame:
        raise ValueError("persistence baseline requires training labels")
    bins = _sensor_bins(train_frame)
    labels = _labels(train_frame["label"].to_numpy(), len(bins))
    counts = np.ones((3, 3), dtype=float)
    np.add.at(counts, (bins, labels), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def persistence_probabilities(frame: pd.DataFrame, table: np.ndarray) -> np.ndarray:
    """Apply the frozen training table using only contemporaneous features."""
    probabilities = validate_probabilities(table)
    if probabilities.shape != (3, 3):
        raise ValueError("persistence table must have shape (3, 3)")
    return probabilities[_sensor_bins(frame)]


def evaluate_gate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    prior_probability: np.ndarray,
    persistence_probability: np.ndarray,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate immutable predictive criteria and retain all passing/failing checks.

    The prior may be one three-class vector or an aligned probability matrix.
    Persistence predictions must be aligned with the same decision population.
    Runtime and deployment acceptance are separate supervisor/promotion gates;
    the runtime budget is retained here as part of the complete frozen policy.
    The persistence check is noninferiority within the declared accuracy margin,
    never a claim that the Transformer outperforms the simple baseline.
    """
    resolved = validate_policy(policy)
    predicted = validate_probabilities(probabilities)
    targets = _labels(labels, len(predicted))
    prior = np.asarray(prior_probability, dtype=float)
    if prior.shape == (3,):
        prior = np.tile(prior, (len(predicted), 1))
    prior = validate_probabilities(prior)
    persistence = validate_probabilities(persistence_probability)
    if prior.shape != predicted.shape or persistence.shape != predicted.shape:
        raise ValueError("baseline predictions must match the model decision population")
    model_metrics = classification_metrics(targets, predicted)
    prior_metrics = classification_metrics(targets, prior)
    persistence_metrics = classification_metrics(targets, persistence)
    counts = np.bincount(targets, minlength=3)
    correct_counts = np.bincount(targets[predicted.argmax(axis=1) == targets], minlength=3)
    recall = correct_counts / np.maximum(counts, 1)
    # A zero-loss reference cannot be improved upon; avoid nonfinite evidence.
    relative_gain = (1.0 - model_metrics["log_loss"] / prior_metrics["log_loss"]
                     if prior_metrics["log_loss"] > 0 else 0.0)
    persistence_difference = model_metrics["accuracy"] - persistence_metrics["accuracy"]
    checks = {
        "minimum_rows": bool(len(predicted) >= resolved["min_rows"]),
        "minimum_class_rows": bool(np.all(counts >= resolved["min_class_rows"])),
        "accuracy": bool(model_metrics["accuracy"] >= resolved["min_accuracy"]),
        "macro_f1": bool(model_metrics["macro_f1"] >= resolved["min_macro_f1"]),
        "class_recall": bool(np.all(recall >= resolved["min_class_recall"])),
        "maximum_log_loss": bool(model_metrics["log_loss"] <= resolved["max_log_loss"]),
        "prior_log_loss_improvement": bool(relative_gain >= resolved["min_prior_log_loss_improvement"]),
        "persistence_accuracy_noninferiority": bool(
            persistence_difference >= -resolved["persistence_accuracy_margin"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {**model_metrics, "class_recall": recall.tolist(),
                    "prior_log_loss_improvement": float(relative_gain),
                    "persistence_accuracy_difference": float(persistence_difference)},
        "baselines": {"class_prior": prior_metrics, "sensor_persistence": persistence_metrics},
        "class_counts": counts.tolist(),
        "n_rows": len(predicted),
        "policy": resolved,
        "scope": "synthetic_classification_service_only",
    }
