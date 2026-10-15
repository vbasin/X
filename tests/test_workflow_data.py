"""DB-free causality, separation, calibration and cost contracts."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment.workflow_data import (
    FEATURE_COLUMNS,
    PARTITION_NAMES,
    apply_temperature,
    calibration_report,
    chronological_partitions,
    classification_metrics,
    evaluate_synthetic_economics,
    fit_temperature,
    make_synthetic_dataset,
    validate_probabilities,
)

pytestmark = pytest.mark.ml_workflow
COST_CONFIG = Path(__file__).resolve().parents[1] / "config" / "execution_costs.yaml"


def test_synthetic_features_are_prefix_invariant_and_labels_follow_forward_path():
    short = make_synthetic_dataset(120, seed=19, horizon=4)
    long = make_synthetic_dataset(360, seed=19, horizon=4)
    pd.testing.assert_frame_equal(short, long.iloc[:120])
    assert short.index.name == "ts"
    assert str(short.index.tz) == "UTC"
    assert all(name.startswith("wf_") for name in FEATURE_COLUMNS)
    assert np.isfinite(short[list(FEATURE_COLUMNS)].to_numpy()).all()
    np.testing.assert_allclose(short["future_price"], long["price"].iloc[4:124])
    np.testing.assert_array_equal(short["future_end"], long.index[4:124])
    delta = short["future_price"] - short["price"]
    np.testing.assert_array_equal(short["label"], np.where(delta < -0.6, 0, np.where(delta > 0.6, 2, 1)))
    # Sensor updates become usable at the publication instant, then carry forward.
    for start in range(0, len(short), 4):
        assert short["wf_context"].iloc[start:start + 4].nunique() == 1


def test_partition_labels_and_sequence_context_do_not_cross_boundaries():
    data = make_synthetic_dataset(1000)
    parts = chronological_partitions(data, sequence_length=16)
    assert tuple(parts) == PARTITION_NAMES
    prior = None
    for name, frame in parts.items():
        assert frame.attrs["partition"] == name
        assert len(frame) > 16
        if prior is not None:
            assert prior.index[-1] < frame.index[0]
            assert prior["future_end"].max() < frame.index[0]
            assert prior.index.intersection(frame.index).empty
        # Context is built from this partition, and the first scored endpoint
        # is row 15. No prior partition needs to be read to construct a sample.
        endpoints = np.arange(15, len(frame))
        assert (endpoints - 15 >= 0).all()
        prior = frame
    assert set(parts["holdout"].index).isdisjoint(parts["calibration"].index)


@pytest.mark.parametrize("change", ["unsorted", "duplicate", "naive", "bad_horizon", "too_short"])
def test_partition_rejects_invalid_temporal_contract(change):
    data = make_synthetic_dataset(400)
    if change == "unsorted":
        data = data.iloc[::-1]
    elif change == "duplicate":
        data.index = pd.DatetimeIndex([data.index[0]] * len(data))
    elif change == "naive":
        data.index = data.index.tz_localize(None)
    elif change == "bad_horizon":
        data.loc[data.index[0], "future_end"] = data.index[0]
    else:
        data = data.iloc[:50]
    with pytest.raises(ValueError):
        chronological_partitions(data)


def test_calibration_uses_supplied_rows_and_serving_application_is_identical():
    # Deliberately overconfident calibration predictions: tempering helps.
    raw = np.tile([0.98, 0.01, 0.01], (100, 1))
    labels = np.r_[np.zeros(70), np.ones(15), np.full(15, 2)].astype(int)
    temperature = fit_temperature(raw, labels)
    assert temperature > 1
    calibrated = apply_temperature(raw, temperature)
    assert classification_metrics(labels, calibrated)["log_loss"] < classification_metrics(labels, raw)["log_loss"]
    # Holdout labels are never passed to the calibrator, and mutation cannot
    # change its already frozen temperature or serving transformation.
    holdout = make_synthetic_dataset(500).iloc[-50:].copy()
    holdout.loc[:, "label"] = 2
    assert fit_temperature(raw, labels) == temperature
    np.testing.assert_array_equal(apply_temperature(raw[:3], temperature), calibrated[:3])
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)


@pytest.mark.parametrize("bad", [
    np.array([[np.nan, 0.5, 0.5]]),
    np.array([[0.2, 0.2, 0.2]]),
    np.array([[1.1, -0.1, 0.0]]),
    np.empty((0, 3)),
    np.ones((3, 2)),
])
def test_probability_contract_rejects_invalid_predictions(bad):
    with pytest.raises(ValueError):
        validate_probabilities(bad)


def test_classification_metrics_keep_three_class_semantics_with_one_class():
    metrics = classification_metrics(np.array([1, 1]), np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["log_loss"] == pytest.approx(-np.log(0.8))


def test_calibration_report_is_canonical_and_keeps_empty_bin_counts():
    from src.experiment.artifacts import BY_CLASS, validate_content

    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    labels = np.array([0, 2, 2])
    report = calibration_report(probabilities, labels, 1.0, 0.55)
    assert not validate_content(BY_CLASS["calibration"], report)
    assert report["n_samples"] == 3
    assert len(report["bucket_edges"]) == 11
    assert len(report["bucket_counts"]) == 10
    assert sum(report["bucket_counts"]) == 3
    assert report["brier_score"] == pytest.approx((0.3**2 + 0.8**2 + 0.4**2) / 3)
    assert report["temperature"] == 1.0
    assert report["confidence_threshold"] == 0.55
    for count, predicted, observed in zip(report["bucket_counts"], report["predicted_rate"], report["observed_rate"]):
        if count == 0:
            assert predicted == observed == 0
    softened = calibration_report(probabilities, labels, 2.0, 0.55)
    calibrated_confidence = apply_temperature(probabilities, 2.0).max(axis=1)
    assert np.dot(softened["bucket_counts"], softened["predicted_rate"]) / 3 == pytest.approx(calibrated_confidence.mean())


def test_calibration_report_places_certainty_in_last_bin_and_rejects_invalid_policy():
    report = calibration_report(np.eye(3), np.arange(3), 1.0, 0.5)
    assert report["bucket_counts"][-1] == 3
    assert report["brier_score"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match="confidence_threshold"):
        calibration_report(np.eye(3), np.arange(3), 1.0, float("nan"))


def test_economic_ledger_is_nonoverlapping_and_charges_real_profile_costs():
    frame = make_synthetic_dataset(160)
    # Oracle predictions only exercise economics; no claimed model quality.
    probabilities = np.eye(3)[frame["label"].to_numpy()]
    report = evaluate_synthetic_economics(frame, probabilities, cost_config=COST_CONFIG)
    trades = report["ledger"]
    assert trades
    assert report["synthetic_only"] is True
    assert report["assumptions"]["economic_contract_unit"] == "MNQ_equivalent"
    assert report["assumptions"]["underlying"] == "SYNTHETIC_NQ_SCALE"
    assert report["assumptions"]["contract_multiplier"] == 2.0
    assert all(t["commission"] == 1.03 for t in trades)
    for previous, current in zip(trades, trades[1:]):
        assert pd.Timestamp(previous["exit_ts"]) <= pd.Timestamp(current["entry_ts"])
    for trade in trades:
        timestamp = pd.Timestamp(trade["entry_ts"])
        delta = frame.loc[timestamp, "future_price"] - frame.loc[timestamp, "price"]
        expected = (trade["direction"] * delta - 0.25) * 2.0 - 1.03
        assert trade["net_dollars"] == pytest.approx(expected)
    base = report["summary"]["net_dollars"]
    assert report["sensitivity"]["double_commission"]["net_dollars"] == pytest.approx(base - 1.03 * len(trades))
    assert report["sensitivity"]["adverse_slippage"]["net_dollars"] == pytest.approx(base - 1.0 * len(trades))
    assert report["summary"]["assumptions"] == report["assumptions"]
    assert report["sensitivity"]["double_commission"]["assumptions"]["commission_per_trade"] == 2.06
    assert report["sensitivity"]["adverse_slippage"]["assumptions"]["slippage_points_per_side"] == 0.25


def test_abstaining_policy_produces_json_safe_empty_economics():
    import json

    frame = make_synthetic_dataset(30)
    probabilities = np.tile([0.1, 0.8, 0.1], (len(frame), 1))
    report = evaluate_synthetic_economics(frame, probabilities, cost_config=COST_CONFIG)
    assert report["ledger"] == []
    assert report["summary"]["n_trades"] == 0
    assert report["summary"]["mean_net_dollars_per_trade"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("probabilities,threshold,expected", [
    ([0.4, 0.5, 0.1], 0.3, -1),
    ([0.3, 0.4, 0.3], 0.3, 1),
    ([0.5, 0.0, 0.5], 0.5, 1),
])
def test_economic_policy_matches_serving_at_low_thresholds_and_ties(probabilities, threshold, expected):
    frame = make_synthetic_dataset(1)
    report = evaluate_synthetic_economics(
        frame, np.array([probabilities]), cost_config=COST_CONFIG, confidence_threshold=threshold,
    )
    assert len(report["ledger"]) == 1
    assert report["ledger"][0]["direction"] == expected
