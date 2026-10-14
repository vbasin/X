"""Tests for the autonomous iterate-until-target loop — Phase 23.

Uses a fake ``run_search`` (monkeypatched) for the loop-control tests so nothing trains and
no GPU VM is contacted; one dry-run test exercises the real controller path end-to-end.
"""

from __future__ import annotations

import pytest

from src.search.iterate import iterate_until_target, load_iterate_spec



# ── spec loading ────────────────────────────────────────────────────────────

def test_load_example_iterate_spec():
    cfg = load_iterate_spec("config/iterate_spec.example.yaml")
    assert cfg.objective_id == "nq-50pt"
    assert cfg.candidate["family"] == "xgboost"
    assert "n_estimators" in cfg.search_space and "max_depth" in cfg.search_space
    assert len(cfg.feature_choices) == 3
    assert cfg.budget["max_trials"] == 40


def test_default_search_space_from_catalog():
    # xgboost has a catalog search_space -> omitting it in the spec falls back to that.
    cfg = load_iterate_spec({"objective_id": "nq-50pt",
                             "candidate": {"family": "xgboost"}})
    assert cfg.search_space                      # non-empty (inherited from catalog)


def test_missing_search_space_raises():
    # tcn has no catalog search_space and none provided -> must raise.
    with pytest.raises(ValueError, match="no search_space"):
        load_iterate_spec({"objective_id": "nq-50pt",
                           "candidate": {"family": "tcn"}})


def test_bad_schema_raises():
    pytest.importorskip("jsonschema")
    with pytest.raises(ValueError, match="schema validation"):
        load_iterate_spec({"candidate": {}})     # missing required 'family'


# ── loop control (fake run_search) ────────────────────────────────────────────

def _fake_summary(cid: str, *, precision: float, target: bool) -> dict:
    return {
        "candidates": [{"candidate_id": cid, "family": "xgboost",
                        "score": {"precision_at_tau": precision}}],
        "gate_result": {"target_ids": [cid] if target else []},
    }


def test_target_met_stops_early(monkeypatch):
    calls = {"n": 0}

    def fake_run_search(obj_id, *, manual_spec, ensembles, dry_run, parquet_dir):
        calls["n"] += 1
        cid = manual_spec["candidates"][0]["id"]
        # meet the target on the 2nd trial
        return _fake_summary(cid, precision=0.9, target=(calls["n"] == 2))

    monkeypatch.setattr("src.search.controller.run_search", fake_run_search)

    report = iterate_until_target({
        "objective_id": "nq-50pt",
        "candidate": {"id": "xgb_t", "family": "xgboost"},
        "search_space": {"max_depth": {"type": "int", "low": 3, "high": 10}},
        "budget": {"max_trials": 20},
    })
    assert report["target_met"] is True
    assert report["stopped_reason"] == "target_met"
    assert report["n_trials"] == 2
    assert report["best"]["meets_target"] is True


def test_max_trials_budget_stop(monkeypatch):
    def fake_run_search(obj_id, *, manual_spec, ensembles, dry_run, parquet_dir):
        cid = manual_spec["candidates"][0]["id"]
        return _fake_summary(cid, precision=0.4, target=False)   # never meets target

    monkeypatch.setattr("src.search.controller.run_search", fake_run_search)

    report = iterate_until_target({
        "objective_id": "nq-50pt",
        "candidate": {"id": "xgb_t", "family": "xgboost"},
        "search_space": {"max_depth": {"type": "int", "low": 3, "high": 10}},
        "budget": {"max_trials": 5},
    })
    assert report["target_met"] is False
    assert report["stopped_reason"] == "max_trials_reached"
    assert report["n_trials"] == 5


def test_cost_budget_stop(monkeypatch):
    def fake_run_search(obj_id, *, manual_spec, ensembles, dry_run, parquet_dir):
        cid = manual_spec["candidates"][0]["id"]
        return _fake_summary(cid, precision=0.4, target=False)

    monkeypatch.setattr("src.search.controller.run_search", fake_run_search)

    # xgboost is cost_class "moderate" (~$0.30/iter), so a $0.20 budget stops after 1 trial.
    report = iterate_until_target({
        "objective_id": "nq-50pt",
        "candidate": {"id": "xgb_t", "family": "xgboost"},
        "search_space": {"max_depth": {"type": "int", "low": 3, "high": 10}},
        "budget": {"max_trials": 50, "cost_budget_usd": 0.20},
    })
    assert report["stopped_reason"] == "cost_budget_exhausted"
    assert report["n_trials"] == 1


def test_feature_choices_are_searched(monkeypatch):
    seen_features = []

    def fake_run_search(obj_id, *, manual_spec, ensembles, dry_run, parquet_dir):
        cand = manual_spec["candidates"][0]
        seen_features.append(cand.get("features"))
        return _fake_summary(cand["id"], precision=0.4, target=False)

    monkeypatch.setattr("src.search.controller.run_search", fake_run_search)

    report = iterate_until_target({
        "objective_id": "nq-50pt",
        "candidate": {"id": "xgb_t", "family": "xgboost"},
        "search_space": {"max_depth": {"type": "int", "low": 3, "high": 10}},
        "feature_choices": [{"view": "sequence_baseline"}, {"prefixes": ["vw_"]}],
        "budget": {"max_trials": 6},
    })
    # the loop must have picked from the provided feature choices
    assert all(f in ({"view": "sequence_baseline"}, {"prefixes": ["vw_"]}) for f in seen_features)
    assert report["n_trials"] == 6


# ── real controller path (dry-run) ────────────────────────────────────────────

def test_iterate_dry_run_end_to_end():
    report = iterate_until_target(
        {
            "objective_id": "nq-50pt",
            "candidate": {"id": "xgb_dry", "family": "xgboost"},
            "search_space": {"max_depth": {"type": "int", "low": 3, "high": 8}},
            "budget": {"max_trials": 2},
        },
        dry_run=True,
    )
    assert report["n_trials"] == 2
    assert report["target_met"] is False          # dry-run produces no real scores
    from pathlib import Path
    assert Path(report["winner_spec_path"]).exists()
    assert Path(report["report_path"]).exists()
