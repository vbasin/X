"""ADR-072: YAML 3×3 expected-cost train loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.objectives.loader import _validate_custom_optimization, load_objective
from src.training.cost_matrix import (
    ExpectedCostXgbObj,
    cost_matrix_array,
    custom_optimization_enabled,
    expected_cost_grad_hess,
    expected_cost_per_row,
    keras_compile_loss,
    parse_cost_matrix,
    softmax_rows,
)


def _tbm_opt_cost_on():
    """Live YAML currently has custom_optimization false (CE probe). Tests that
    need the expected-cost path flip the flag on a copy."""
    obj = dict(load_objective("nq-50pt-tbm-opt"))
    obj["custom_optimization"] = True
    return obj


def test_nq_50pt_tbm_opt_loads():
    obj = load_objective("nq-50pt-tbm-opt")
    assert obj["objective_id"] == "nq-50pt-tbm-opt"
    assert obj["labels"]["scheme"] == "triple_barrier"
    assert obj["labels"]["move_threshold"] == 50
    assert obj["labels"]["mae_tolerance"] == 6
    C = parse_cost_matrix(obj.get("cost_matrix"))
    assert C.shape == (3, 3)
    assert C[0, 0] == 0.0 and C[2, 2] == 0.0 and C[1, 1] == 0.0
    assert C[0, 1] == C[2, 1] == 50.0
    assert C[0, 2] == C[2, 0] == 56.0
    assert C[1, 0] == C[1, 2] == 6.0
    # Flag off → CE; matrix still present for a later flip to true.
    assert obj.get("custom_optimization") is not True
    assert cost_matrix_array(obj) is None
    assert keras_compile_loss(obj) == "sparse_categorical_crossentropy"
    assert custom_optimization_enabled(load_objective("nq-50pt-econ")) is False
    assert cost_matrix_array(load_objective("nq-50pt-econ")) is None
    assert cost_matrix_array(load_objective("nq-50pt-tbm")) is None


def test_flag_true_requires_matrix():
    with pytest.raises(ValueError, match="cost_matrix"):
        _validate_custom_optimization(
            {"custom_optimization": True}, Path("dummy.yaml"))


def test_parse_row_list_or_map():
    C = parse_cost_matrix({
        "labels": ["down", "flat", "up"],
        "costs": {
            "down": [0, 8, 50],
            "flat": {"down": 50, "flat": 3, "up": 50},
            "up": {"down": 50, "flat": 8, "up": 0},
        },
    })
    np.testing.assert_allclose(C[0], [0, 8, 50])
    np.testing.assert_allclose(C[1], [50, 3, 50])


def test_expected_cost_prefers_correct_direction():
    C = np.array([[0, 8, 50], [50, 3, 50], [50, 8, 0]], dtype=np.float64)
    y = np.array([2, 2, 1])  # up, up, flat
    p_hit = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    p_miss = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    hit = expected_cost_per_row(p_hit, y, C)
    miss = expected_cost_per_row(p_miss, y, C)
    assert hit[0] == 0.0 and miss[0] == 8.0
    assert miss[1] == 50.0
    assert hit.mean() < miss.mean()


def test_softmax_grad_points_at_cheapest_class():
    C = np.array([[0, 8, 50], [50, 3, 50], [50, 8, 0]], dtype=np.float64)
    logits = np.zeros((1, 3), dtype=np.float64)
    y = np.array([2])  # true up — cheapest cell is p_up
    grad, hess = expected_cost_grad_hess(logits, y, C)
    assert grad[0, 2] < 0  # increase p_up
    assert grad[0, 0] > 0  # decrease p_down (wrong fire)
    assert (hess > 0).all()
    # finite-diff check on class 2
    p = softmax_rows(logits)
    eps = 1e-5
    logits_up = logits.copy()
    logits_up[0, 2] += eps
    p_up = softmax_rows(logits_up)
    fd = (expected_cost_per_row(p_up, y, C)[0] - expected_cost_per_row(p, y, C)[0]) / eps
    assert abs(fd - grad[0, 2]) < 1e-3


def test_keras_compile_loss_switch():
    assert keras_compile_loss(None) == "sparse_categorical_crossentropy"
    assert keras_compile_loss({"custom_optimization": False}) == "sparse_categorical_crossentropy"
    obj = _tbm_opt_cost_on()
    loss = keras_compile_loss(obj)
    from src.models.keras_layers import ExpectedCostLoss
    assert isinstance(loss, ExpectedCostLoss)
    assert loss.costs[2][2] == 0.0
    assert loss.costs[1] == [6.0, 0.0, 6.0]


def test_xgb_custom_obj_tiny_fit():
    pytest.importorskip("xgboost")
    from src.models.base import ModelSpec
    from src.models.tabular_plugins import XGBoostPlugin

    obj = _tbm_opt_cost_on()
    plugin = XGBoostPlugin()
    spec = ModelSpec(family="xgboost", params={
        "n_estimators": 8, "max_depth": 2, "learning_rate": 0.3,
        "verbosity": 0, "n_jobs": 1, "class_balance": "sqrt",
    })
    model = plugin.build(spec, obj)
    assert isinstance(model.get_params()["objective"], ExpectedCostXgbObj)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=48)
    folds = [{
        "X_train": X[:36], "y_train": y[:36],
        "X_val": X[36:], "y_val": y[36:],
        "X_oof": X[36:], "y_oof": y[36:],
    }]
    fitted = plugin.fit(model, folds, objective=obj)
    assert fitted.oof_preds is not None
    assert set(fitted.oof_preds["direction"].unique()) <= {"up", "down", "flat"}
    conf = fitted.oof_preds["confidence"].to_numpy()
    assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_lgbm_custom_obj_tiny_fit():
    pytest.importorskip("lightgbm")
    from src.models.base import ModelSpec
    from src.models.tabular_plugins import LightGBMPlugin

    obj = _tbm_opt_cost_on()
    plugin = LightGBMPlugin()
    spec = ModelSpec(family="lightgbm", params={
        "n_estimators": 8, "max_depth": 2, "verbose": -1, "n_jobs": 1,
        "min_child_samples": 1, "num_leaves": 4,
    })
    model = plugin.build(spec, obj)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(48, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=48)
    folds = [{
        "X_train": X[:36], "y_train": y[:36],
        "X_val": X[36:], "y_val": y[36:],
        "X_oof": X[36:], "y_oof": y[36:],
    }]
    fitted = plugin.fit(model, folds, objective=obj)
    assert fitted.oof_preds is not None
    conf = fitted.oof_preds["confidence"].to_numpy()
    assert conf.min() >= 0.0 and conf.max() <= 1.0
