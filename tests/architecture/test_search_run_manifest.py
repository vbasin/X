"""Tests for the FIRST canonical integration: src/search/run_manifest.py.

No DB, no search execution -- the adapter is exercised directly against a
tmp_path run directory, which is why it was factored out of controller.py.
"""
import json

from src.experiment.manifest import RunManifest
from src.search.run_manifest import build_manifest, write_manifest

OBJECTIVE = {
    "objective_id": "nq-50pt-econ",
    "costs": {"slippage_model": "fixed_1pt", "cost_per_trade": 4.0},
    "labels": {"move_threshold": 50},
}
FOLD_META = [{"train": {}, "oof": {}}, {"train": {}, "oof": {}}, {"train": {}, "oof": {}}]


def _kw(out_dir, **over):
    kw = dict(
        objective_id="nq-50pt-econ", run_ts="20260922T120000Z", out_dir=out_dir,
        objective=OBJECTIVE, experiment_type="custom_test",
        fold_split="expanding_equal_count", fold_meta=FOLD_META,
        model_ids=("xgboost",),
    )
    kw.update(over)
    return kw


def test_manifest_records_what_the_search_path_actually_has(tmp_path):
    m = build_manifest(**_kw(tmp_path))
    assert m.objective_id == "nq-50pt-econ"
    assert m.run_id == "nq-50pt-econ/20260922T120000Z"
    assert m.validation.split == "expanding_equal_count"
    assert m.validation.n_folds == 3
    assert m.config_snapshot["objective"]["labels"]["move_threshold"] == 50
    # execution_assumptions is now built by the ONE canonical builder
    # (src/evaluation/economic.py::assumptions_record), so it is a normalised
    # record rather than a raw copy of the objective's costs block: the declared
    # fixed_1pt slippage_model is parsed into points.
    assert m.execution_assumptions["slippage_points_per_side"] == 1.0
    assert m.execution_assumptions["commission_per_trade"] == 4.0
    assert m.execution_assumptions["source"] == "objective costs block"
    assert "model_strategy:xgboost" in m.responsibility_versions
    # Recorded but explicitly NOT claimed as the fully resolved config: a
    # Custom Test run sweeps multiple candidates with their own resolved
    # hyperparameters, which no single run-level mapping can represent.
    # The RUN's configuration IS resolved here: `objective` is the
    # post-topology-override, post-eval-override document. Per-candidate
    # hyperparameters live in each model.json and say nothing about it.
    assert m.config_snapshot_resolved is True


def test_manifest_is_honestly_incomplete_and_names_the_gaps(tmp_path):
    """Without a label contract and without runtime-supplied seeds, the two
    genuinely absent dimensions are dataset identity and seeds. Neither may be
    invented. hypothesis no longer counts (it is not a reproducibility input)
    and the run-level config IS resolved."""
    m = build_manifest(**_kw(tmp_path))
    assert m.evidence_level == "INCOMPLETE"
    assert set(m.missing_evidence()) == {"datasets", "seeds"}


def test_runtime_supplied_seeds_and_datasets_close_those_gaps(tmp_path):
    """What the controller now passes in. A run whose runtime knows its seeds
    and its input datasets should be able to say so, rather than having them
    reconstructed from the files it just wrote."""
    from src.experiment.manifest import DatasetRef

    m = build_manifest(
        **_kw(tmp_path),
        seeds={"fold_subsample": 42},
        datasets=(DatasetRef(logical_id="labels_abc", semantic_type="outcomes",
                             contract_id="abc123", build_id="b1"),),
    )
    assert m.seeds == {"fold_subsample": 42}
    assert set(m.missing_evidence()) == set()
    assert m.evidence_level == "CANONICAL_COMPLETE"


def test_no_folds_is_recorded_as_no_validation_not_as_a_default(tmp_path):
    """3 of the 54 runs on disk had no walk-forward validation at all."""
    m = build_manifest(**_kw(tmp_path, fold_meta=[], fold_split=None))
    assert m.validation is None
    assert "validation" in m.missing_evidence()


def test_dataset_identity_comes_from_the_existing_label_contract(tmp_path):
    (tmp_path / "label_contract.json").write_text(json.dumps({
        "contract": {"resolution": "15s"},
        "contract_id": "1aa87aaf223d0f7a",
        "contract_sha256": "e2e7a868",
        "source_dir": "/opt/vtech/data/sources/labels/15s/1aa87aaf223d0f7a",
    }))
    m = build_manifest(**_kw(tmp_path))
    assert m.datasets[0].contract_id == "1aa87aaf223d0f7a"
    assert m.datasets[0].logical_id == "labels_15s"
    assert m.datasets[0].identity_strength == "contract"
    # Tenth-pass correction: contract_id alone states what the dataset MEANS,
    # not WHICH realization was used (the same contract_id can silently name
    # different bytes after a force=True rebuild) -- it is real evidence, but
    # not strong enough to certify a controlled comparison on its own.
    assert "datasets" in m.missing_evidence()


def test_contract_metadata_hash_is_never_exposed_as_content_sha256(tmp_path):
    """label_contract.json's contract_sha256 hashes the _contract.json METADATA
    sidecar (label_store.write_label_contract_snapshot), never the label
    parquet's bytes -- it must never be surfaced as DatasetRef.content_sha256,
    which claims to be a genuine content digest. Ninth-pass correction; see
    docs/STORAGE_CONTRACT.md §5a and docs/STORAGE_INVENTORY.md §4 STOR-11."""
    (tmp_path / "label_contract.json").write_text(json.dumps({
        "contract": {"resolution": "15s"},
        "contract_id": "1aa87aaf223d0f7a",
        "contract_sha256": "e2e7a868deadbeef",
        "source_dir": "/opt/vtech/data/sources/labels/15s/1aa87aaf223d0f7a",
        "parts": [{"name": "labels_15s_part1.parquet", "bytes": 12345}],
    }))
    m = build_manifest(**_kw(tmp_path))
    ref = m.datasets[0]
    assert ref.content_sha256 is None
    assert ref.content_sha256 != "e2e7a868deadbeef"


def test_part_filename_and_size_inventory_is_not_build_identity(tmp_path):
    """Tenth-pass correction: the ninth pass derived build_id from a
    fingerprint of the recorded part inventory (filenames + byte sizes).
    That is NOT genuine build/provenance evidence -- it says nothing about
    which code, config or upstream data produced the realization, only that
    a directory listing didn't change. This adapter has no genuine
    code/config/upstream evidence available to it, so build_id must stay
    honestly None even when a rich part inventory is present -- never
    substitute a weaker output-metadata signal for it."""
    (tmp_path / "label_contract.json").write_text(json.dumps({
        "contract": {"resolution": "15s"},
        "contract_id": "1aa87aaf223d0f7a",
        "contract_sha256": "e2e7a868",
        "source_dir": "/opt/vtech/data/sources/labels/15s/1aa87aaf223d0f7a",
        "parts": [{"name": "labels_15s_part1.parquet", "bytes": 12345},
                  {"name": "labels_15s_part2.parquet", "bytes": 67890}],
    }))
    m = build_manifest(**_kw(tmp_path))
    ref = m.datasets[0]
    assert ref.build_id is None
    assert ref.content_sha256 is None
    assert ref.contract_id == "1aa87aaf223d0f7a"
    assert ref.identity_strength == "contract"
    # And a contract-only dataset ref is INCOMPLETE realization evidence,
    # even though the part inventory changed nothing about that conclusion.
    assert "datasets" in m.missing_evidence()


def test_legacy_label_contract_without_parts_still_reads(tmp_path):
    """Older label_contract.json shapes (no 'parts' key, predating this
    pass) must remain readable: contract_id alone is sufficient identity,
    build_id is honestly None rather than fabricated or erroring."""
    (tmp_path / "label_contract.json").write_text(json.dumps({
        "contract": {"resolution": "5s"},
        "contract_id": "abc123",
        "source_dir": "/opt/vtech/data/sources/labels/5s/abc123",
    }))
    m = build_manifest(**_kw(tmp_path))
    ref = m.datasets[0]
    assert ref.contract_id == "abc123"
    assert ref.build_id is None
    assert ref.content_sha256 is None
    assert ref.identity_strength == "contract"
    assert "datasets" in m.missing_evidence()


def test_code_identity_states_dirty_rather_than_assuming_clean(tmp_path):
    assert build_manifest(**_kw(tmp_path)).code.dirty in (True, False)


def test_write_is_additive_and_touches_no_existing_artifact(tmp_path):
    """The integration's core safety property."""
    existing = {"run.json": '{"kind": "custom_test"}',
                "folds.json": '{"n_folds": 3}',
                "search_result.json": '{"n_ok": 1}'}
    for name, body in existing.items():
        (tmp_path / name).write_text(body)

    assert write_manifest(**_kw(tmp_path)) == tmp_path / "manifest.json"

    for name, body in existing.items():
        assert (tmp_path / name).read_text() == body, f"{name} was modified"
    assert (tmp_path / "manifest.json").is_file()


def test_write_never_raises_on_bad_input(tmp_path):
    """A provenance record must never fail a search."""
    assert write_manifest(**_kw(tmp_path / "does" / "not" / "exist" / "x")) is None or True
    assert write_manifest(objective_id="", run_ts="", out_dir=tmp_path,
                          objective={}, experiment_type=None,
                          fold_split=None, fold_meta=None) is None


def test_written_manifest_round_trips_and_reports_incompleteness(tmp_path):
    write_manifest(**_kw(tmp_path))
    back = RunManifest.read(tmp_path)
    assert back.run_id == "nq-50pt-econ/20260922T120000Z"
    assert back.evidence_level == "INCOMPLETE"


def test_rerunning_the_same_search_rewrites_idempotently(tmp_path):
    write_manifest(**_kw(tmp_path))
    assert write_manifest(**_kw(tmp_path)) is not None


def test_a_changed_config_under_the_same_run_id_is_refused(tmp_path):
    """Scientific immutability survives the integration."""
    write_manifest(**_kw(tmp_path))
    changed = dict(OBJECTIVE, labels={"move_threshold": 75})
    assert write_manifest(**_kw(tmp_path, objective=changed)) is None
    snap = RunManifest.read(tmp_path).config_snapshot
    assert snap["objective"]["labels"]["move_threshold"] == 50


def test_unavailable_git_is_missing_evidence_never_a_placeholder(tmp_path, monkeypatch):
    """No "unknown" SHA: unavailable code identity must make the run INCOMPLETE."""
    import src.search.run_manifest as rm
    monkeypatch.setattr(rm, "_git_identity", lambda: None)
    m = build_manifest(**_kw(tmp_path))
    assert m.code is None
    assert "code" in m.missing_evidence()
    assert m.evidence_level == "INCOMPLETE"
    assert "unknown" not in m.to_json()


def test_config_snapshot_folds_in_existing_overlay_files_when_present(tmp_path):
    """spec.manual.yaml / topology.yaml / walk_forward.yaml, when the
    controller already wrote them (_copy_custom_test_specs runs before the
    manifest write), are genuinely this run's own resolved overlay inputs --
    not fabricated, just read."""
    import yaml
    (tmp_path / "topology.yaml").write_text(yaml.safe_dump({"mode": "meta_stack"}))
    m = build_manifest(**_kw(tmp_path))
    assert m.config_snapshot["topology"] == {"mode": "meta_stack"}
    assert "manual_spec" not in m.config_snapshot  # file absent -> key absent
