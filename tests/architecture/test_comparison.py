"""Tests for src.experiment.comparison.

These are deliberately adversarial: the module's job is to REFUSE to certify
unsound comparisons, so most tests assert that controlled_comparison is False.
"""
from src.experiment.comparison import compare_runs
from src.experiment.manifest import (
    CodeIdentity,
    DatasetRef,
    RunManifest,
    ValidationDesign,
)


def _manifest(run_id: str, **overrides) -> RunManifest:
    base = dict(
        run_id=run_id,
        objective_id="nq-50pt-econ",
        hypothesis="baseline",
        responsibility_versions={"feature_builder": "fb@1"},
        code=CodeIdentity(git_sha="abc123", dirty=False),
        datasets=(DatasetRef(logical_id="labels_15s", contract_id="c1"),),
        config_snapshot={"labels": {"move_threshold": 50, "mae_tolerance": 6}},
        validation=ValidationDesign(split="expanding_equal_count", n_folds=3),
        seeds={"numpy": 42},
        execution_assumptions={"slippage_points_per_side": 1.0},
        # Fully evidenced on purpose: every compared dimension that CAN be
        # empty is treated as missing evidence rather than as agreement, so a
        # partially-filled fixture can never produce a controlled comparison.
        decision_policy={"gate": "p_move>=0.95"},
        calibration={"method": "none"},
    )
    base.update(overrides)
    return RunManifest(**base)


def test_single_dimension_change_is_controlled():
    a = _manifest("a")
    b = _manifest("b", responsibility_versions={"feature_builder": "fb@2"})
    r = compare_runs(a, b)
    assert r.controlled_comparison
    assert r.uncontrolled_dimensions == ("responsibility_versions",)
    assert r.attributable_dimension == "responsibility_versions"
    assert r.warning == ""


def test_two_dimensions_changing_is_not_controlled():
    """The failure the previous implementation certified as attributable."""
    a = _manifest("a")
    b = _manifest(
        "b",
        responsibility_versions={"feature_builder": "fb@2"},
        datasets=(DatasetRef(logical_id="labels_15s", contract_id="c2"),),
    )
    r = compare_runs(a, b)
    assert not r.controlled_comparison
    assert set(r.uncontrolled_dimensions) == {"responsibility_versions", "datasets"}
    assert "No single-cause claim" in r.warning
    assert r.attributable_dimension is None


def test_responsibility_change_with_different_seeds_is_not_controlled():
    a = _manifest("a")
    b = _manifest("b", responsibility_versions={"feature_builder": "fb@2"}, seeds={"numpy": 7})
    r = compare_runs(a, b)
    assert not r.controlled_comparison
    assert "seeds" in r.uncontrolled_dimensions


def test_responsibility_change_with_different_costs_is_not_controlled():
    """Comparing a model change across different cost assumptions is the
    documented backtest-vs-paper failure mode."""
    a = _manifest("a")
    b = _manifest(
        "b",
        responsibility_versions={"feature_builder": "fb@2"},
        execution_assumptions={"slippage_points_per_side": 0.0},
    )
    r = compare_runs(a, b)
    assert not r.controlled_comparison
    assert "execution_assumptions" in r.uncontrolled_dimensions


def test_unrecorded_dimension_is_missing_evidence_not_agreement():
    """Two runs that both omit seeds have not been shown to share a seed."""
    a = _manifest("a", seeds={})
    b = _manifest("b", seeds={}, responsibility_versions={"feature_builder": "fb@2"})
    r = compare_runs(a, b)
    assert not r.controlled_comparison
    assert "seeds" in r.missing_evidence
    assert "seeds" not in r.uncontrolled_dimensions
    assert "unknown, not identical" in r.warning


def test_all_evidence_missing_is_reported():
    blank = dict(
        datasets=(), validation=None, seeds={}, execution_assumptions={},
        config_snapshot={}, decision_policy={}, calibration={},
    )
    r = compare_runs(_manifest("a", **blank), _manifest("b", **blank))
    assert not r.controlled_comparison
    assert set(r.missing_evidence) == {
        "datasets", "validation", "seeds", "execution_assumptions",
        "config_snapshot", "decision_policy", "calibration",
    }


def test_identical_runs_warn_about_unexplained_difference():
    r = compare_runs(_manifest("a"), _manifest("b"))
    assert not r.controlled_comparison
    assert r.uncontrolled_dimensions == ()
    assert "unexplained" in r.warning


def test_config_diff_is_recursive_not_top_level_only():
    """A single nested key change must be reported at its leaf path."""
    a = _manifest("a")
    b = _manifest("b", config_snapshot={"labels": {"move_threshold": 75, "mae_tolerance": 6}})
    r = compare_runs(a, b)
    assert r.controlled_comparison
    assert r.differences["config_snapshot"] == {"labels.move_threshold": (50, 75)}


def test_dataset_contract_change_is_detected_at_leaf():
    a = _manifest("a")
    b = _manifest("b", datasets=(DatasetRef(logical_id="labels_15s", contract_id="c2"),))
    r = compare_runs(a, b)
    assert r.differences["datasets"] == {"[0].contract_id": ("c1", "c2")}


def test_objective_change_is_a_dimension():
    """Comparing runs across two research objectives is not a comparison."""
    a = _manifest("a")
    b = _manifest("b", objective_id="nq-atr-auction")
    r = compare_runs(a, b)
    assert "objective_id" in r.uncontrolled_dimensions


def test_dirty_tree_difference_is_detected():
    a = _manifest("a")
    b = _manifest("b", code=CodeIdentity(git_sha="abc123", dirty=True))
    r = compare_runs(a, b)
    assert r.differences["code"] == {"dirty": (False, True)}


def test_validation_engine_change_is_detected():
    a = _manifest("a")
    b = _manifest("b", validation=ValidationDesign(split="expanding_calendar", n_folds=3))
    r = compare_runs(a, b)
    assert "validation" in r.uncontrolled_dimensions
    assert r.differences["validation"]["split"] == (
        "expanding_equal_count", "expanding_calendar"
    )


# --- the adversarial-review hole: silence must be silence for EVERY dimension --

def test_unrecorded_decision_policy_is_missing_evidence_not_agreement():
    """Counterexample from the adversarial review.

    Two runs use different tau thresholds but both leave decision_policy
    empty. Before the fix this produced no leaf diff, so the dimension was
    reported as NOT varying, and a single unrelated difference elsewhere was
    certified as an attributable cause.
    """
    a = _manifest("a", decision_policy={})
    b = _manifest("b", decision_policy={}, seeds={"numpy": 7})
    r = compare_runs(a, b)
    assert not r.controlled_comparison
    assert "decision_policy" in r.missing_evidence
    assert r.attributable_dimension is None


def test_unrecorded_calibration_is_missing_evidence():
    a = _manifest("a", calibration={})
    b = _manifest("b", calibration={}, seeds={"numpy": 7})
    assert "calibration" in compare_runs(a, b).missing_evidence


def test_unrecorded_config_snapshot_is_missing_evidence():
    a = _manifest("a", config_snapshot={})
    b = _manifest("b", config_snapshot={}, seeds={"numpy": 7})
    assert "config_snapshot" in compare_runs(a, b).missing_evidence


def test_explicitly_recording_no_calibration_is_evidence_not_silence():
    """`{"method": "none"}` says "we did not calibrate"; `{}` says "we did
    not write down whether we calibrated". Only the first supports a
    controlled comparison."""
    a = _manifest("a", calibration={"method": "none"})
    b = _manifest("b", calibration={"method": "none"}, seeds={"numpy": 7})
    r = compare_runs(a, b)
    assert "calibration" not in r.missing_evidence
    assert r.controlled_comparison
    assert r.attributable_dimension == "seeds"


def test_every_emptiable_compared_dimension_requires_evidence():
    """Guard against a dimension being added to COMPARED_DIMENSIONS without
    being added to EVIDENCE_REQUIRED -- the way the hole above appeared."""
    from src.experiment.comparison import (
        COMPARED_DIMENSIONS,
        EVIDENCE_REQUIRED,
    )

    # These two cannot be empty: RunManifest.__post_init__ refuses them.
    # (``code`` used to be here; it may now be None, so it requires evidence.)
    enforced_non_empty = {"objective_id", "responsibility_versions"}
    assert set(COMPARED_DIMENSIONS) - enforced_non_empty == set(EVIDENCE_REQUIRED)


# --- result comparison (setup vs results are reported separately) -------------

def _write(d, name, obj):
    import json
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj))


def test_result_diff_is_reported_at_leaf(tmp_path):
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "metrics.json", {"n_samples": 100, "metrics": {"auc": 0.61}})
    _write(b, "metrics.json", {"n_samples": 100, "metrics": {"auc": 0.64}})
    r = compare_results(a, b, _manifest("a"), _manifest("b"))
    assert r.differences["metrics.json"] == {"metrics.auc": (0.61, 0.64)}
    assert r.one_sided == {}


def test_artifact_present_in_only_one_run_is_reported(tmp_path):
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "metrics.json", {"n_samples": 10, "metrics": {"auc": 0.61}})
    b.mkdir()
    r = compare_results(a, b, _manifest("a"), _manifest("b"))
    assert r.one_sided == {"metrics.json": "a"}


def test_artifacts_are_not_required_for_steps_that_did_not_occur(tmp_path):
    """A run with no calibration/economics step is not 'missing' those
    artifacts -- requiring them would manufacture findings."""
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    no_econ = dict(execution_assumptions={}, decision_policy={}, validation=None)
    r = compare_results(a, b, _manifest("a", **no_econ), _manifest("b", **no_econ))
    # calibration.method is "none" in the fixture, and no run declares an
    # economic_evaluation responsibility -- so neither is expected.
    assert set(r.not_applicable) == {
        "fold_metrics.json", "selected_population.json", "calibration.json",
        "economic_metrics.json", "cost_sensitivity.json",
    }
    assert r.missing_expected == (("metrics.json", "a"), ("metrics.json", "b"))


def test_expected_artifact_absent_is_reported_as_missing(tmp_path):
    """The fixture's manifest records a fold schedule, so fold_metrics.json
    is expected."""
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    ok = {"n_samples": 10, "metrics": {"auc": 0.6}}
    _write(a, "metrics.json", ok); _write(b, "metrics.json", ok)
    r = compare_results(a, b, _manifest("a"), _manifest("b"))
    assert ("fold_metrics.json", "a") in r.missing_expected
    assert ("fold_metrics.json", "b") in r.missing_expected


def test_two_runs_both_lacking_code_identity_do_not_agree_on_it():
    """code may be None now; silence must not read as agreement."""
    a = _manifest("a", code=None)
    b = _manifest("b", code=None, seeds={"numpy": 7})
    r = compare_runs(a, b)
    assert "code" in r.missing_evidence
    assert not r.controlled_comparison


def test_unreadable_expected_artifact_is_invalid_evidence_not_a_clean_pass(tmp_path):
    """Adversarial: a corrupt metrics.json on one side must not make the
    comparison look clean. Previously it was silently skipped."""
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "metrics.json", {"n_samples": 10, "metrics": {"auc": 0.61}})
    b.mkdir()
    (b / "metrics.json").write_text("{not valid json")
    r = compare_results(a, b, _manifest("a"), _manifest("b"))
    assert [(art, side) for art, side, _ in r.invalid_evidence] == [("metrics.json", "b")]
    assert "metrics.json" not in r.differences
    assert not r.clean


# --- invalid evidence: one-sided, and schema-invalid --------------------------

def test_corrupt_artifact_is_invalid_even_when_the_other_run_lacks_it(tmp_path):
    """Adversarial, one-sided: the corrupt file is the ONLY copy. Previously
    the presence check returned first and the file was never read."""
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "metrics.json").write_text("{truncated")
    r = compare_results(a, b, _manifest("a"), _manifest("b"))
    assert [(art, side) for art, side, _ in r.invalid_evidence] == [("metrics.json", "a")]
    assert ("metrics.json", "b") in r.missing_expected
    assert not r.clean


def test_readable_but_schema_invalid_artifact_is_invalid_evidence(tmp_path):
    """Parseable JSON lacking the canonical minimum content is not evidence."""
    from src.experiment.comparison import compare_results
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "metrics.json", {"auc": 0.61})           # no n_samples
    _write(b, "metrics.json", {"n_samples": 10, "metrics": {"auc": 0.6}})
    r = compare_results(a, b, _manifest("a"), _manifest("b"))
    assert r.invalid_evidence[0][:2] == ("metrics.json", "a")
    assert "n_samples" in r.invalid_evidence[0][2]
