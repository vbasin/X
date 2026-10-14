"""Tests for src.experiment.schemas."""
import pytest

from src.experiment.schemas import (
    OOF_PREDICTION_COLUMNS,
    CalibrationResult,
    FoldMetrics,
    RunMetrics,
    SelectedPopulation,
    validate_oof_columns,
)


# --- RunMetrics (the metrics.json artifact) -----------------------------------

def test_run_metrics_requires_a_population():
    """A metric without the population it was computed over cannot be
    compared against another run's."""
    with pytest.raises(ValueError):
        RunMetrics(n_samples=0, metrics={"auc": 0.61})


def test_run_metrics_requires_at_least_one_metric():
    with pytest.raises(ValueError):
        RunMetrics(n_samples=10, metrics={})


def test_run_metrics_rejects_impossible_positive_count():
    with pytest.raises(ValueError):
        RunMetrics(n_samples=100, metrics={"auc": 0.61}, n_positive=101)


def test_run_metrics_positive_count_is_optional():
    assert RunMetrics(n_samples=100, metrics={"auc": 0.61}).n_positive is None


# --- OOF column contract ------------------------------------------------------

def test_oof_contract_matches_the_frame_backtest_runner_already_writes():
    assert OOF_PREDICTION_COLUMNS == ("ts", "direction", "confidence", "direction_true")


def test_validate_oof_columns_accepts_extra_columns():
    """Real frames carry model-specific scores; rejecting them would break
    existing producers for no benefit."""
    validate_oof_columns(["ts", "direction", "confidence", "direction_true", "p_up"])


def test_validate_oof_columns_names_the_missing_column():
    with pytest.raises(ValueError, match="direction_true"):
        validate_oof_columns(["ts", "direction", "confidence"])


def test_fold_metrics_requires_at_least_one_metric():
    with pytest.raises(ValueError):
        FoldMetrics(fold_id="f0", train_start="2025-04", train_end="2025-07",
                    test_start="2025-08", test_end="2025-09", n_train=100, n_test=50,
                    metrics={})


def test_fold_metrics_rejects_negative_counts():
    with pytest.raises(ValueError):
        FoldMetrics(fold_id="f0", train_start="2025-04", train_end="2025-07",
                    test_start="2025-08", test_end="2025-09", n_train=-1, n_test=50,
                    metrics={"auc": 0.52})


def test_fold_metrics_valid_round_trip_fields():
    fm = FoldMetrics(fold_id="fold0", train_start="2025-04", train_end="2025-07",
                     test_start="2025-08", test_end="2025-09", n_train=686668, n_test=383124,
                     metrics={"weighted_auc": 0.5266})
    assert fm.metrics["weighted_auc"] == pytest.approx(0.5266)


def test_calibration_result_rejects_mismatched_bucket_lengths():
    with pytest.raises(ValueError):
        CalibrationResult(method="isotonic", n_samples=1000,
                          bucket_edges=(0.0, 0.5, 1.0),
                          predicted_rate=(0.1,),
                          observed_rate=(0.1, 0.2),  # mismatched length
                          brier_score=0.2)


def test_calibration_result_rejects_wrong_edge_count():
    with pytest.raises(ValueError):
        CalibrationResult(method="isotonic", n_samples=1000,
                          bucket_edges=(0.0, 1.0),  # should be 3 edges for 2 buckets
                          predicted_rate=(0.1, 0.2),
                          observed_rate=(0.1, 0.2),
                          brier_score=0.2)


def test_calibration_result_rejects_out_of_range_brier():
    with pytest.raises(ValueError):
        CalibrationResult(method="isotonic", n_samples=100, bucket_edges=(0.0, 1.0),
                          predicted_rate=(0.5,), observed_rate=(0.5,), brier_score=1.5)


def test_calibration_result_none_method_is_valid_uncalibrated_marker():
    cr = CalibrationResult(method="none", n_samples=100, bucket_edges=(0.0, 0.5, 1.0),
                           predicted_rate=(0.2, 0.7), observed_rate=(0.25, 0.65),
                           brier_score=0.18)
    assert cr.method == "none"


def test_selected_population_rejects_selected_exceeding_raw():
    with pytest.raises(ValueError):
        SelectedPopulation(n_raw=100, n_selected=150, selection_rule="p_move>=tau")


def test_selected_population_selection_rate():
    pop = SelectedPopulation(n_raw=1_890_412, n_selected=86_124, selection_rule="p_move>=tau_0.95")
    assert pop.selection_rate == pytest.approx(86_124 / 1_890_412)


def test_selected_population_zero_raw_gives_nan_rate_not_crash():
    pop = SelectedPopulation(n_raw=0, n_selected=0, selection_rule="none")
    import math
    assert math.isnan(pop.selection_rate)


def test_selected_population_overlap_none_is_not_zero_overlap():
    pop = SelectedPopulation(n_raw=100, n_selected=10, selection_rule="x")
    assert pop.overlap_row_ids is None  # "not computed", distinct from frozenset()


def test_selected_population_overlap_can_be_recorded():
    pop = SelectedPopulation(n_raw=100, n_selected=10, selection_rule="x",
                             overlap_row_ids=frozenset({"r1", "r2"}))
    assert pop.overlap_row_ids == {"r1", "r2"}
