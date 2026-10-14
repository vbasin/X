"""Tests for src.experiment.manifest.

No DB, no dataset dependency -- constructs manifests in tmp_path only.
"""
import json

import pytest

from src.experiment.manifest import (
    CodeIdentity,
    DatasetRef,
    RunManifest,
    ValidationDesign,
)


def _manifest(**overrides) -> RunManifest:
    base = dict(
        run_id="run-001",
        objective_id="nq-50pt-econ",
        hypothesis="Adding flow-state features improves selective precision.",
        responsibility_versions={"feature_builder": "cascade/es_fs_st@abc123"},
        code=CodeIdentity(git_sha="abc123", dirty=False),
        datasets=(DatasetRef(logical_id="labels_15s", contract_id="1aa87aaf223d0f7a"),),
        config_snapshot={"labels": {"move_threshold": 50}},
    )
    base.update(overrides)
    return RunManifest(**base)


# --- required-field validation ------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("run_id", ""),
    ("objective_id", ""),
    ("responsibility_versions", {}),
])
def test_required_fields_rejected_when_empty(field, value):
    with pytest.raises(ValueError):
        _manifest(**{field: value})


def test_dataset_ref_requires_logical_id():
    with pytest.raises(ValueError):
        DatasetRef(logical_id="")


def test_code_identity_requires_git_sha():
    with pytest.raises(ValueError):
        CodeIdentity(git_sha="", dirty=False)


def test_code_identity_dirty_is_not_optional():
    """An unqualified SHA is an unverifiable claim -- dirty must be stated."""
    with pytest.raises(TypeError):
        CodeIdentity(git_sha="abc123")  # type: ignore[call-arg]


def test_validation_design_rejects_bad_fold_count():
    with pytest.raises(ValueError):
        ValidationDesign(split="expanding_calendar", n_folds=0)


# --- round trip ---------------------------------------------------------------

def test_write_read_round_trip(tmp_path):
    m = _manifest(
        validation=ValidationDesign(split="expanding_equal_count", n_folds=3),
        seeds={"numpy": 42},
        execution_assumptions={"slippage_points_per_side": 1.0},
    )
    m.write(tmp_path)
    back = RunManifest.read(tmp_path)
    assert back.run_id == m.run_id
    assert back.objective_id == "nq-50pt-econ"
    assert back.code == m.code
    assert back.validation == m.validation
    assert back.datasets[0].contract_id == "1aa87aaf223d0f7a"
    assert back.scientific_fingerprint() == m.scientific_fingerprint()


def test_written_json_embeds_fingerprint(tmp_path):
    _manifest().write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert len(data["scientific_fingerprint"]) == 16


# --- scientific immutability --------------------------------------------------

def test_identical_rewrite_is_allowed(tmp_path):
    """A retried job rewriting the same manifest must not fail."""
    m = _manifest()
    m.write(tmp_path)
    m.write(tmp_path)


def test_rewrite_with_different_run_id_is_refused(tmp_path):
    _manifest().write(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _manifest(run_id="run-002").write(tmp_path)


def test_pass_inventory_does_not_fork_the_run(tmp_path):
    """Model pointers accumulate. The objective snapshot does not."""
    _manifest().write(tmp_path)
    _manifest(passes=({"pass": 1, "role": "directional_screen", "models": []},)).write(tmp_path)
    back = RunManifest.read(tmp_path)
    assert back.passes[0]["role"] == "directional_screen"


def test_empty_validation_may_be_filled_once(tmp_path):
    _manifest().write(tmp_path)
    _manifest(validation=ValidationDesign(
        split="expanding_month_blocks", n_folds=3, test_months=2,
    )).write(tmp_path)
    with pytest.raises(FileExistsError):
        _manifest(validation=ValidationDesign(
            split="expanding_calendar", n_folds=4,
        )).write(tmp_path)


def test_rewrite_with_changed_science_under_same_run_id_is_refused(tmp_path):
    """The core immutability guarantee: a run_id always meant one experiment."""
    _manifest().write(tmp_path)
    changed = _manifest(config_snapshot={"labels": {"move_threshold": 75}})
    with pytest.raises(FileExistsError, match="different scientific fingerprint"):
        changed.write(tmp_path)


def test_bookkeeping_change_does_not_break_immutability(tmp_path):
    """created_at/artifacts/extra are not scientific identity."""
    _manifest(created_at="2026-01-01T00:00:00+00:00").write(tmp_path)
    _manifest(
        created_at="2026-09-22T00:00:00+00:00",
        artifacts={"metrics": ["metrics.json"]},
        extra={"note": "rerun after disk full"},
    ).write(tmp_path)


def test_fingerprint_changes_with_each_scientific_dimension():
    base = _manifest().scientific_fingerprint()
    assert _manifest(seeds={"numpy": 7}).scientific_fingerprint() != base
    assert _manifest(parent_run_id="run-000").scientific_fingerprint() != base
    assert _manifest(
        execution_assumptions={"slippage_points_per_side": 1.0}
    ).scientific_fingerprint() != base
    assert _manifest(
        validation=ValidationDesign(split="expanding_calendar", n_folds=6)
    ).scientific_fingerprint() != base


# --- schema versioning --------------------------------------------------------

def test_future_schema_version_is_refused_not_silently_misread(tmp_path):
    """run.json has already drifted into two incompatible shapes on disk with
    no version field; this manifest must never repeat that."""
    _manifest().write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["schema_version"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="newer than this code supports"):
        RunManifest.read(tmp_path)


# --- location is traceability, not identity ----------------------------------

def test_dataset_location_does_not_change_the_fingerprint():
    """DatasetRef.location documents itself as "never the identity", so it
    must not enter the hash that DEFINES identity. Found by adversarial
    review: leaving it in meant a store migration or a write-to-temp-then-swap
    changed the fingerprint without changing the science."""
    here = _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1",
                                          location="sources/labels/old/x"),))
    moved = _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1",
                                           location="sources/labels/new/x"),))
    assert here.scientific_fingerprint() == moved.scientific_fingerprint()


def test_moved_dataset_does_not_block_an_idempotent_rewrite(tmp_path):
    """The end-to-end consequence of the fix."""
    _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1",
                                   location="sources/labels/old/x"),)).write(tmp_path)
    _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1",
                                   location="sources/labels/new/x"),)).write(tmp_path)


def test_row_count_IS_identity():
    """n_rows is deliberately not stripped: a different row count is a
    different dataset."""
    a = _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1", n_rows=100),))
    b = _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1", n_rows=101),))
    assert a.scientific_fingerprint() != b.scientific_fingerprint()


def test_contract_id_change_is_identity():
    a = _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c1"),))
    b = _manifest(datasets=(DatasetRef(logical_id="l", contract_id="c2"),))
    assert a.scientific_fingerprint() != b.scientific_fingerprint()


# --- identity classification (docs/EXPERIMENTS.md §1) -------------------------

def test_comparison_context_fields_are_not_scientific_identity():
    """hypothesis / experiment_family / baseline_run_id are comparison
    context. Rewording a hypothesis or relabelling a family must not fork a
    run's identity -- it previously did, so a typo fix tripped the
    immutability guard as though the science had changed."""
    base = _manifest().scientific_fingerprint()
    assert _manifest(hypothesis="reworded, same experiment").scientific_fingerprint() == base
    assert _manifest(experiment_family="regrouped").scientific_fingerprint() == base
    assert _manifest(baseline_run_id="run-000").scientific_fingerprint() == base


def test_parent_run_id_IS_scientific_identity():
    """Derivation is real lineage: a meta layer trained on a different base
    run is a different experiment."""
    assert _manifest(parent_run_id="run-000").scientific_fingerprint() != _manifest().scientific_fingerprint()


def test_hypothesis_is_context_not_a_reproducibility_requirement():
    """A run with no hypothesis is still constructible AND still able to be
    CANONICAL_COMPLETE.

    Everything in REQUIRED_FOR_CANONICAL_COMPLETE is something you need in order
    to re-run the experiment and get the same answer. A sentence of English is
    not, which is exactly why hypothesis is excluded from the scientific
    fingerprint. Gating completeness on it made every run INCOMPLETE for a
    reason with no bearing on whether the evidence is sound."""
    m = _manifest(hypothesis="")
    assert "hypothesis" not in RunManifest.REQUIRED_FOR_CANONICAL_COMPLETE
    assert "hypothesis" not in m.missing_evidence()
    # ...and it is still excluded from scientific identity, as before.
    assert "hypothesis" in RunManifest.NON_FINGERPRINTED_FIELDS
    with_text = _manifest(hypothesis="a directional screen adds edge")
    assert with_text.scientific_fingerprint() == m.scientific_fingerprint()


# --- evidence completeness ----------------------------------------------------

def _complete(**overrides) -> RunManifest:
    base = dict(
        validation=ValidationDesign(split="expanding_equal_count", n_folds=3),
        seeds={"numpy": 42},
        execution_assumptions={"slippage_points_per_side": 1.0},
        # A non-empty config_snapshot is not itself proof of resolution --
        # this fixture explicitly claims the (unfabricated) stronger fact.
        config_snapshot_resolved=True,
        # A contract_id alone (this file's _manifest() default) states what
        # the dataset MEANS, not WHICH realization was used -- tenth pass,
        # docs/STORAGE_CONTRACT.md §5a. "Fully evidenced" needs the stronger
        # build/realization identity too.
        datasets=(DatasetRef(logical_id="labels_15s", contract_id="1aa87aaf223d0f7a",
                             build_id="b1"),),
    )
    base.update(overrides)
    return _manifest(**base)


def test_fully_evidenced_run_is_canonical_complete():
    m = _complete()
    assert m.missing_evidence() == ()
    assert m.evidence_level == "CANONICAL_COMPLETE"


def test_missing_evidence_names_every_gap():
    m = _complete(seeds={}, execution_assumptions={}, validation=None)
    assert set(m.missing_evidence()) == {"seeds", "execution_assumptions", "validation"}
    assert m.evidence_level == "INCOMPLETE"


def test_evidence_level_is_persisted_and_recomputed_not_trusted(tmp_path):
    """A hand-edited file must not be able to assert an evidence level its
    own contents do not support."""
    _complete(seeds={}).write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["evidence_level"] == "INCOMPLETE"
    assert "seeds" in data["missing_evidence"]

    data["evidence_level"] = "CANONICAL_COMPLETE"
    data["missing_evidence"] = []
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    assert RunManifest.read(tmp_path).evidence_level == "INCOMPLETE"


# --- DatasetRef identity ------------------------------------------------------

def test_dataset_ref_requires_content_identity_not_a_path():
    with pytest.raises(ValueError, match="no scientific identity"):
        DatasetRef(logical_id="move_matrix_v4", location="sources/labels/move_matrix_v4")


def test_dataset_ref_accepts_either_identity_mechanism():
    DatasetRef(logical_id="l", contract_id="1aa87aaf223d0f7a")
    DatasetRef(logical_id="l", content_sha256="deadbeef")


# --- config_snapshot completeness is an explicit claim, not an inference ------

def test_nonempty_config_snapshot_alone_does_not_satisfy_evidence():
    """A partial/raw objective dict handed straight through must not be
    read as proof the effective run configuration was fully resolved."""
    m = _complete(config_snapshot={"objective_id": "x"}, config_snapshot_resolved=False)
    assert "config_snapshot" in m.missing_evidence()
    assert m.evidence_level == "INCOMPLETE"


def test_config_snapshot_resolved_defaults_to_false():
    """Silence must not read as a resolution claim nobody made."""
    m = _manifest(config_snapshot={"labels": {"move_threshold": 50}})
    assert m.config_snapshot_resolved is False
    assert "config_snapshot" in m.missing_evidence()


def test_explicitly_resolved_config_snapshot_satisfies_evidence():
    m = _complete(config_snapshot={"labels": {"move_threshold": 50}},
                  config_snapshot_resolved=True)
    assert "config_snapshot" not in m.missing_evidence()


def test_config_snapshot_resolved_is_not_scientific_identity():
    """Whether the caller's claim of resolution is trustworthy does not
    change what experiment ran -- only config_snapshot's CONTENT does."""
    same_content = {"labels": {"move_threshold": 50}}
    a = _manifest(config_snapshot=same_content, config_snapshot_resolved=False)
    b = _manifest(config_snapshot=same_content, config_snapshot_resolved=True)
    assert a.scientific_fingerprint() == b.scientific_fingerprint()
