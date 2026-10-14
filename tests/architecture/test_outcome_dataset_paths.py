"""Canonical outcome-dataset path/filename syntax (tenth pass, Storage
Implementation Tranche 1). DB-free: exercises src/common/data_paths.py and
src/training/label_store.py's new track-neutral helpers directly, with no
training, backtests, DB access or Track C data rebuild.

docs/STORAGE_CONTRACT.md §17 (path), §18 (filenames), §19 (Track C semantic
contract fields) are the corresponding documentation.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from src.common.data_paths import outcome_dataset_dir
from src.training.label_store import (
    BUILD_NAME,
    CONTRACT_NAME,
    label_contract_id,
    label_set_dir,
    move_outcome_contract,
    normalize_contract,
    normalize_label_contract,
    outcome_contract_id,
    outcome_part_name,
    outcome_set_dir,
    target_outcome_contract,
    write_build_sidecar,
    write_contract,
    write_outcome_contract,
)


# --- 1. exact canonical directory syntax --------------------------------------

def test_outcome_dataset_dir_is_family_then_resolution_then_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    p = outcome_dataset_dir("directional_first_passage", "5s", "abc123")
    assert p == tmp_path / "sources" / "labels" / "directional_first_passage" / "5s" / "abc123"


def test_outcome_dataset_dir_refuses_empty_dimensions():
    with pytest.raises(ValueError, match="outcome_family"):
        outcome_dataset_dir("", "5s", "abc123")
    with pytest.raises(ValueError, match="resolution"):
        outcome_dataset_dir("directional_first_passage", "", "abc123")
    with pytest.raises(ValueError, match="contract_id"):
        outcome_dataset_dir("directional_first_passage", "5s", "")


def test_resolution_and_outcome_family_are_independent_dimensions(tmp_path, monkeypatch):
    """Same family, two resolutions -> two different directories with the
    SAME family segment; same resolution, two families -> two different
    directories with the SAME resolution segment. Neither dimension
    collapses into the other."""
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    a = outcome_dataset_dir("triple_barrier", "5s", "c1")
    b = outcome_dataset_dir("triple_barrier", "15s", "c1")
    c = outcome_dataset_dir("candidate_level_reachability", "5s", "c1")
    assert a.parent.parent == b.parent.parent  # same family root
    assert a.name == b.name == c.name == "c1"
    assert a.parent.name == c.parent.name == "5s"
    assert a.parent.parent.name != c.parent.parent.name


# --- 2. track-neutral contract hashing -----------------------------------------

def test_outcome_contract_id_matches_outcome_set_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    mc = move_outcome_contract(decision_grid_seconds=5, horizon_seconds=1800,
                               move_r=0.25, adverse_r=0.125, r_lookback_sessions=5)
    d = outcome_set_dir("directional_first_passage", "5s", mc)
    assert d == outcome_dataset_dir("directional_first_passage", "5s", outcome_contract_id(mc))


def test_outcome_contract_id_does_not_apply_track_b_target_observation_default():
    """normalize_label_contract() (Track B) injects target_observation=
    candle_close when absent; normalize_contract() (track-neutral) must not,
    since that field means nothing outside Track B's TBM family."""
    c = {"family": "directional_first_passage", "move_r": 0.25}
    assert "target_observation" not in normalize_contract(c)
    assert "target_observation" in normalize_label_contract(c)
    assert outcome_contract_id(c) != label_contract_id(c)


def test_track_b_label_contract_id_is_unchanged_by_the_generalization(tmp_path, monkeypatch):
    """Regression guard: generalizing normalize_label_contract() to call the
    new normalize_contract() must not change label_contract_id()'s output
    for Track B's existing production contracts (label_set_dir()'s physical
    paths must stay byte-identical -- no migration performed)."""
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    from src.common.data_paths import labels_dir
    c = {"scheme": "triple_barrier", "move_threshold": 50, "mae_tolerance": 6,
         "horizon_s": 600, "encoding": "3class"}
    # Known-good hash of the normalized (TBM-defaulted) dict via an
    # independent sha256 computation, not the function under test.
    import hashlib
    expected_norm = dict(c, target_observation="candle_close")
    raw = json.dumps(expected_norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert label_contract_id(c) == hashlib.sha256(raw.encode()).hexdigest()[:16]
    assert label_set_dir("15s", c) == labels_dir() / "15s" / label_contract_id(c)


def test_outcome_contracts_are_stable_and_family_scoped():
    mc = move_outcome_contract(decision_grid_seconds=5, horizon_seconds=1800,
                               move_r=0.25, adverse_r=0.125, r_lookback_sessions=5)
    tc = target_outcome_contract(horizon_seconds=1800, max_targets=5, stop_ratio=0.5,
                                 max_stop_r=0.25, resting_entry=True, max_fill_wait=900,
                                 lookback_sessions=40, r_lookback_sessions=5)
    assert mc["family"] == "directional_first_passage"
    assert tc["family"] == "candidate_level_reachability"
    assert outcome_contract_id(mc) != outcome_contract_id(tc)
    # deterministic
    assert outcome_contract_id(mc) == outcome_contract_id(dict(mc))


# --- 3. exact canonical filename syntax ----------------------------------------

def test_outcome_part_name_zero_pads_size_split_parts():
    assert outcome_part_name(1) == "part001.parquet"
    assert outcome_part_name(10) == "part010.parquet"
    with pytest.raises(ValueError):
        outcome_part_name(0)


def test_outcome_part_name_session_keyed_variant():
    assert outcome_part_name(date(2026, 1, 1)) == "part_2026-01-01.parquet"


def test_outcome_part_names_carry_no_track_vocabulary():
    for p in (outcome_part_name(1), outcome_part_name(date(2026, 1, 1))):
        assert "move" not in p and "target" not in p and "stage" not in p


def test_contract_and_build_sidecar_filenames_are_fixed():
    assert CONTRACT_NAME == "_contract.json"
    assert BUILD_NAME == "_build.json"


# --- 4. contract + build sidecar writers ---------------------------------------

def test_write_outcome_contract_roundtrips_and_matches_directory_hash(tmp_path):
    mc = move_outcome_contract(decision_grid_seconds=5, horizon_seconds=1800,
                               move_r=0.25, adverse_r=0.125, r_lookback_sessions=5)
    d = tmp_path / "outcome_dir"
    cid = write_outcome_contract(d, mc)
    assert cid == outcome_contract_id(mc)
    stored = json.loads((d / CONTRACT_NAME).read_text())
    assert stored["contract_id"] == cid
    assert stored["schema_version"] == 1
    assert "target_observation" not in stored


def test_write_build_sidecar_requires_genuine_code_identity(tmp_path, monkeypatch):
    """No fabricated build_id when code identity is unavailable -- an absent
    sidecar is truthful, a fabricated one is not."""
    monkeypatch.setattr("src.common.code_identity.git_identity", lambda: None)
    out = write_build_sidecar(tmp_path, population_config={})
    assert out is None
    assert not (tmp_path / BUILD_NAME).exists()


def test_write_build_sidecar_records_upstream_lineage_and_population_config(tmp_path, monkeypatch):
    monkeypatch.setattr("src.common.code_identity.git_identity", lambda: ("deadbeef", False))
    upstream = [{"logical_id": "move_scores", "semantic_type": "predictions",
                "location": "sources/labels/directional_first_passage/5s/abc/scores.parquet"}]
    out = write_build_sidecar(tmp_path, population_config={"tau_q": 0.95}, upstream=upstream)
    assert out is not None
    doc = json.loads(out.read_text())
    assert doc["code_sha"] == "deadbeef" and doc["code_dirty"] is False
    assert doc["population_config"] == {"tau_q": 0.95}
    assert doc["upstream"] == upstream
    assert doc["build_id"]
    # Deterministic: same inputs -> same build_id.
    (tmp_path / BUILD_NAME).unlink()
    out2 = write_build_sidecar(tmp_path, population_config={"tau_q": 0.95}, upstream=upstream)
    assert json.loads(out2.read_text())["build_id"] == doc["build_id"]
