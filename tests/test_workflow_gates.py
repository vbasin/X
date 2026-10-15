"""Synthetic service quality gates, independent of MLflow or TensorFlow fits."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment.workflow_gates import (
    evaluate_gate, fit_persistence_baseline, load_objective,
    persistence_probabilities, validate_policy,
)

pytestmark = pytest.mark.ml_workflow
OBJECTIVE = Path(__file__).resolve().parents[1] / "config/workflows/synthetic_regime_objective.json"


@pytest.fixture
def policy():
    return json.loads(OBJECTIVE.read_text())["gates"]


@pytest.fixture
def population():
    labels = np.tile(np.arange(3), 100)
    probabilities = np.full((len(labels), 3), 0.05)
    probabilities[np.arange(len(labels)), labels] = 0.9
    return labels, probabilities


def test_quality_pass_retains_all_comparative_evidence(policy, population):
    labels, probabilities = population
    report = evaluate_gate(labels, probabilities, np.full(3, 1 / 3), probabilities, policy)
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["class_counts"] == [100, 100, 100]
    assert report["metrics"]["class_recall"] == [1, 1, 1]
    assert report["metrics"]["persistence_accuracy_difference"] == 0
    assert report["baselines"]["sensor_persistence"]["accuracy"] == 1
    assert report["policy"] == policy
    json.dumps(report, allow_nan=False)


def test_prior_and_shuffled_targets_fail_the_exact_same_policy(policy, population):
    labels, probabilities = population
    prior = np.full((len(labels), 3), 1 / 3)
    prior_control = evaluate_gate(labels, prior, prior, probabilities, policy)
    shuffled = labels[np.random.default_rng(732).permutation(len(labels))]
    shuffled_control = evaluate_gate(shuffled, probabilities, prior, probabilities, policy)
    for report in (prior_control, shuffled_control):
        assert not report["passed"]
        assert not report["checks"]["accuracy"]
        assert not report["checks"]["prior_log_loss_improvement"]


def test_persistence_noninferiority_is_enforced_even_above_accuracy_floor(policy, population):
    labels, perfect = population
    degraded = perfect.copy()
    degraded[:30] = np.roll(degraded[:30], 1, axis=1)
    report = evaluate_gate(labels, degraded, np.full(3, 1 / 3), perfect, policy)
    assert report["checks"]["accuracy"]
    assert report["checks"]["macro_f1"]
    assert not report["checks"]["persistence_accuracy_noninferiority"]
    assert not report["passed"]


def test_class_imbalance_cannot_hide_an_unlearned_class(policy):
    labels = np.r_[np.zeros(125, dtype=int), np.ones(125, dtype=int), np.full(50, 2)]
    probabilities = np.full((len(labels), 3), 0.05)
    guesses = labels.copy()
    guesses[-50:] = 0
    probabilities[np.arange(len(labels)), guesses] = 0.9
    report = evaluate_gate(labels, probabilities, np.full(3, 1 / 3), probabilities, policy)
    assert report["checks"]["accuracy"]
    assert not report["checks"]["class_recall"]
    assert not report["passed"]


@pytest.mark.parametrize("rows", [30, 199])
def test_too_few_decisions_or_class_examples_fail(policy, population, rows):
    labels, probabilities = population
    report = evaluate_gate(labels[:rows], probabilities[:rows], np.full(3, 1 / 3), probabilities[:rows], policy)
    assert not report["passed"]
    assert not report["checks"]["minimum_rows"]


def test_missing_class_fails_instead_of_dropping_class_from_denominator(policy):
    labels = np.tile([0, 1], 150)
    probabilities = np.full((300, 3), 0.05)
    probabilities[np.arange(300), labels] = 0.9
    report = evaluate_gate(labels, probabilities, np.full(3, 1 / 3), probabilities, policy)
    assert not report["checks"]["minimum_class_rows"]
    assert not report["checks"]["class_recall"]
    assert report["metrics"]["class_recall"][2] == 0


def test_sensor_baseline_uses_only_training_confusion_counts():
    train = pd.DataFrame({"wf_context": [-1.0, -0.6, 0.0, 0.4, 0.6, 1.0], "label": [0, 0, 1, 1, 2, 2]})
    table = fit_persistence_baseline(train)
    np.testing.assert_allclose(table, np.eye(3) * 0.4 + 0.2)
    # Inference needs no labels; any supplied target values are ignored.
    inference = pd.DataFrame({"wf_context": [0.8, -0.8, 0.2], "label": [0, 2, 0]})
    expected = table[[2, 0, 1]]
    np.testing.assert_allclose(persistence_probabilities(inference, table), expected)
    np.testing.assert_allclose(persistence_probabilities(inference.drop(columns="label"), table), expected)
    assert np.all(table > 0)


def test_unseen_training_sensor_bins_have_smoothed_uniform_predictions():
    train = pd.DataFrame({"wf_context": [-1.0, -1.0], "label": [0, 0]})
    table = fit_persistence_baseline(train)
    np.testing.assert_allclose(table[1:], 1 / 3)


@pytest.mark.parametrize("change", [
    {"ignored_gate": 1}, {"min_accuracy": float("nan")},
    {"min_rows": True}, {"min_class_rows": 100},
    {"max_runtime_seconds": 1800}, {"max_log_loss": 0},
    {"persistence_accuracy_margin": -0.01},
])
def test_policy_rejects_unknown_nonfinite_and_inconsistent_settings(policy, change):
    with pytest.raises(ValueError):
        validate_policy({**policy, **change})


def test_policy_rejects_missing_setting(policy):
    incomplete = dict(policy)
    incomplete.pop("min_class_recall")
    with pytest.raises(ValueError):
        validate_policy(incomplete)


@pytest.mark.parametrize("bad", [np.array([np.nan, 0.2, 0.8]), np.array([0.1, 0.1, 0.1]), np.ones((2, 3)) / 3])
def test_malformed_or_misaligned_baseline_fails_closed(policy, population, bad):
    labels, probabilities = population
    with pytest.raises(ValueError):
        evaluate_gate(labels, probabilities, bad, probabilities, policy)


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"wf_context": [], "label": []}),
    pd.DataFrame({"wf_context": [float("inf")], "label": [0]}),
    pd.DataFrame({"wf_context": [0.0], "label": [3]}),
])
def test_bad_training_baseline_data_rejected(frame):
    with pytest.raises(ValueError):
        fit_persistence_baseline(frame)


def test_objective_is_fully_resolved_and_explicitly_non_trading():
    objective = load_objective(OBJECTIVE)
    assert objective["objective_id"] == "synthetic-regime-service"
    assert objective["synthetic_only"] is True
    assert objective["model_config"]["seed"] == 20261015
    assert objective["gates"]["min_accuracy"] == 0.8
    assert objective["gates"]["max_runtime_seconds"] == 900


@pytest.mark.parametrize("mutation", [
    lambda obj: obj.update(unknown=True),
    lambda obj: obj.update(synthetic_only=False),
    lambda obj: obj.update(objective_id="nq-50pt"),
    lambda obj: obj["model_config"].update(objective_id="nq-50pt"),
    lambda obj: obj["model_config"].pop("epochs"),
    lambda obj: obj["model_config"].update(max_runtime_seconds=800),
    lambda obj: obj["model_config"].update(min_accuracy=0.5),
])
def test_objective_refuses_ambiguous_or_inconsistent_contract(tmp_path, mutation):
    objective = deepcopy(json.loads(OBJECTIVE.read_text()))
    mutation(objective)
    path = tmp_path / "objective.json"
    path.write_text(json.dumps(objective))
    with pytest.raises(ValueError):
        load_objective(path)
