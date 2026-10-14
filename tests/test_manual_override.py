"""Tests for the USER-OVERRIDE (manual candidates) mode — Phase 23.

Covers the manual-spec loader/validation, the explicit feature-selection helpers, and an
end-to-end dry-run through the search controller. No GPU VM or real training required.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.catalog import CatalogEntry, get_entry
from src.models.feature_views import (
    apply_feature_selection,
    resolve_selection_indices,
)
from src.search.manual_spec import load_manual_spec


# ── fixtures ────────────────────────────────────────────────────────────────

def _folds():
    """Two synthetic folds with 2-D (tabular) and 3-D (sequence) X arrays."""
    cols = ["vw_a", "vw_b", "rv_x", "bp_1", "dc_p"]
    n = 6
    x2d = np.arange(n * 5, dtype="float32").reshape(n, 5)
    x3d = np.arange(n * 4 * 5, dtype="float32").reshape(n, 4, 5)
    fold = {
        "X_train": x2d.copy(),
        "y_train": np.zeros(n, dtype="int32"),
        "X_oof":   x3d.copy(),
        "y_oof":   np.zeros(n, dtype="int32"),
        "feature_cols": cols,
    }
    return [fold, dict(fold)]


# ── manual-spec loader ──────────────────────────────────────────────────────

def test_load_example_spec_builds_entries():
    spec = load_manual_spec("config/manual_candidates.example.yaml", objective_id="nq-50pt")
    assert spec.objective_id == "nq-50pt"
    assert spec.ensembles is False
    assert [e.name for e in spec.entries] == [
        "xgb_seqbase", "xgb_microstructure", "lgbm_handpicked", "rf_full"]

    # Two candidates share the xgboost family but keep distinct ids.
    xgb1, xgb2, lgbm, rf = spec.entries
    assert xgb1.family_key == "xgboost" and xgb1.plugin_family == "xgboost"
    assert xgb2.family_key == "xgboost"
    assert lgbm.family_key == "lightgbm"
    assert rf.family_key == "random_forest"

    # params merged over catalog defaults; user keys win.
    base = get_entry("xgboost")
    assert xgb1.params_default["n_estimators"] == 400          # user override
    assert xgb1.params_default["eval_metric"] == base.params_default["eval_metric"]  # inherited

    # feature selection carried verbatim; manual mode clears search_space + row cap.
    assert xgb1.feature_selection == {"view": "sequence_baseline"}
    assert xgb2.feature_selection == {"prefixes": ["bp_", "of_", "vp_"]}
    assert lgbm.feature_selection == {"columns": ["vw_dist_bps", "rv_5m",
                                                  "ht_atr_regime", "dc_pivot_dist"]}
    assert rf.feature_selection is None                       # full matrix
    assert xgb1.search_space == {} and xgb1.search_max_rows is None


def test_xgboost_ovr_alias_keeps_xgboost_plugin():
    spec = load_manual_spec({
        "candidates": [{
            "id": "my_ovr",
            "family": "xgboost_ovr",
            "params": {"head": "ovr", "dir_threshold": 0.55},
        }]
    })
    e = spec.entries[0]
    assert e.name == "my_ovr"
    assert e.family_key == "xgboost"
    assert e.params_default["head"] == "ovr"
    assert e.params_default["dir_threshold"] == 0.55


def test_load_keeps_explicit_search_space():
    spec = load_manual_spec({
        "candidates": [{
            "family": "xgboost",
            "params": {"max_depth": 6},
            "search_space": {
                "max_depth": {"type": "int", "low": 4, "high": 12},
            },
        }]
    })
    e = spec.entries[0]
    assert e.params_default["max_depth"] == 6
    assert set(e.search_space) == {"max_depth"}
    assert e.search_space["max_depth"]["high"] == 12


def test_load_rejects_bad_search_space_type():
    with pytest.raises(ValueError, match="search_space"):
        load_manual_spec({
            "candidates": [{
                "family": "xgboost",
                "search_space": {"max_depth": {"type": "bogus", "low": 1, "high": 2}},
            }]
        })


def test_load_from_dict():
    spec = load_manual_spec({"candidates": [{"family": "xgboost"}]})
    assert len(spec.entries) == 1
    e = spec.entries[0]
    assert e.name == "xgboost" and e.plugin_family == "xgboost"  # id defaults to family


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        load_manual_spec({"candidates": [{"family": "not_a_model"}]})


def test_duplicate_id_raises():
    with pytest.raises(ValueError, match="duplicate manual candidate id"):
        load_manual_spec({"candidates": [
            {"family": "xgboost"}, {"family": "xgboost"}]})  # both default id -> "xgboost"


def test_objective_id_mismatch_raises():
    with pytest.raises(ValueError, match="wrong objective"):
        load_manual_spec({"objective_id": "es-30pt", "candidates": [{"family": "xgboost"}]},
                         objective_id="nq-50pt")


def test_schema_rejects_two_feature_keys():
    pytest.importorskip("jsonschema")
    with pytest.raises(ValueError, match="schema validation"):
        load_manual_spec({"candidates": [
            {"family": "xgboost", "features": {"view": "regime_core", "prefixes": ["vw_"]}}]})


# ── feature selection helpers ───────────────────────────────────────────────

def test_resolve_indices_view_prefixes_columns():
    cols = ["vw_a", "vw_b", "rv_x", "bp_1", "dc_p"]
    assert resolve_selection_indices({"prefixes": ["vw_"]}, cols) == [0, 1]
    assert resolve_selection_indices({"columns": ["rv_x", "dc_p"]}, cols) == [2, 4]
    # microstructure view includes the "bp_" prefix -> selects bp_1 (index 3).
    assert 3 in resolve_selection_indices({"view": "microstructure"}, cols)


def test_resolve_indices_unknown_view_raises():
    with pytest.raises(ValueError, match="unknown feature view"):
        resolve_selection_indices({"view": "no_such_view"}, ["vw_a"])


def test_resolve_indices_bad_shape_raises():
    with pytest.raises(ValueError):
        resolve_selection_indices({"view": "x", "prefixes": ["y"]}, ["a"])  # two keys


def test_apply_feature_selection_masks_2d_and_3d():
    folds = apply_feature_selection(_folds(), {"prefixes": ["vw_"]})
    f = folds[0]
    assert f["feature_cols"] == ["vw_a", "vw_b"]
    assert f["X_train"].shape == (6, 2)      # 2-D tabular masked on last axis
    assert f["X_oof"].shape == (6, 4, 2)     # 3-D sequence masked on last axis
    # values correspond to the first two feature columns of the original matrix
    np.testing.assert_array_equal(f["X_train"][:, 0], np.arange(6) * 5)


def test_apply_feature_selection_close_logret():
    from src.models.feature_views import apply_feature_selection, close_logret_1d
    r = close_logret_1d(np.array([100.0, 100.0, np.e * 100.0]))
    assert r.shape == (3, 1)
    assert float(r[0, 0]) == 0.0
    np.testing.assert_allclose(r[2, 0], 1.0, rtol=1e-5)
    folds = [{
        "X_train": np.zeros((4, 5), dtype="float32"),
        "X_oof": np.zeros((3, 5), dtype="float32"),
        "X_val": np.zeros((0, 5), dtype="float32"),
        "close_train": np.array([10.0, 10.0, 20.0, 20.0]),
        "close_oof": np.array([20.0, 40.0, 40.0]),
        "feature_cols": ["vw_a", "rv_x", "bp_1", "dc_p", "ht_z"],
    }]
    out = apply_feature_selection(folds, {"columns": ["_close"]})
    assert out[0]["feature_cols"] == ["_close_logret"]
    assert out[0]["X_train"].shape == (4, 1)
    assert out[0]["X_oof"].shape == (3, 1)
    np.testing.assert_allclose(out[0]["X_train"][2, 0], np.log(2.0), rtol=1e-5)


def test_apply_feature_selection_no_match_falls_back():
    original = _folds()
    out = apply_feature_selection(original, {"prefixes": ["zzz_"]})
    assert out[0]["feature_cols"] == ["vw_a", "vw_b", "rv_x", "bp_1", "dc_p"]  # unchanged
    assert out[0]["X_train"].shape == (6, 5)


# ── controller integration (dry-run) ────────────────────────────────────────

def test_run_search_manual_dry_run():
    from src.search.controller import run_search
    summary = run_search(
        "nq-50pt",
        manual_spec="config/manual_candidates.example.yaml",
        dry_run=True,
        ensembles=False,
    )
    assert summary["n_candidates"] == 4
    ids = {c["candidate_id"] for c in summary["candidates"]}
    assert ids == {"xgb_seqbase", "xgb_microstructure", "lgbm_handpicked", "rf_full"}
    fams = {c["family"] for c in summary["candidates"]}
    assert fams == {"xgboost", "lightgbm", "random_forest"}


def test_manual_and_families_mutually_exclusive():
    from src.search.controller import run_search
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_search("nq-50pt", families=["xgboost"],
                   manual_spec={"candidates": [{"family": "xgboost"}]}, dry_run=True)

