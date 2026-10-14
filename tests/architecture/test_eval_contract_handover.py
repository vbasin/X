"""The hold-out pin's handover from evaluation/<objective>/contract_vN.json to
the run manifest (step 8). DB-free; no scoring.

docs/new_file_map.md keeps the file as the live pin "until that loader reads the
manifest". This pins the sequencing: the manifest can supply the pin, the file
still works unchanged, and two disagreeing sources are refused rather than
resolved by precedence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.contract import ContractHashMismatch, EvalContract


def _run_with_pin(root: Path, *, objective: str, run_ts: str,
                  dates: list[str], digest: str) -> Path:
    run = root / objective / run_ts
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": f"{objective}/{run_ts}",
        "objective_id": objective,
        "schema_version": 4,
        "validation": {"split": "expanding_month_blocks", "n_folds": 5,
                       "hold_out_dates": dates},
        "config_snapshot": {"hold_out_hash": digest, "tau_thresholds": [0.5, 0.9]},
        "passes": [],
    }))
    return run


def test_manifest_supplies_the_pin(tmp_path):
    run = _run_with_pin(tmp_path, objective="obj", run_ts="20260923T000000Z",
                        dates=["2026-02-02", "2026-02-03"], digest="a" * 64)
    got = EvalContract.from_manifest(run)
    assert got is not None
    assert got.objective_id == "obj"
    assert got.hold_out_dates == ["2026-02-02", "2026-02-03"]
    assert got.hold_out_hash == "a" * 64
    # Versioned by the run that established it, not by a file counter.
    assert got.contract_version == "20260923T000000Z"
    assert got.tau_thresholds == [0.5, 0.9]


def test_a_manifest_without_a_pin_reports_absence(tmp_path):
    """An absent pin must never be reconstructed from the fold design."""
    run = tmp_path / "obj" / "t"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "obj/t", "objective_id": "obj", "schema_version": 4,
        "validation": {"split": "expanding_month_blocks", "n_folds": 5},
        "config_snapshot": {}, "passes": [],
    }))
    assert EvalContract.from_manifest(run) is None
    assert EvalContract.from_manifest(tmp_path / "does" / "not" / "exist") is None


def test_the_file_remains_the_pin_when_no_run_dir_is_given():
    """The four live Track B objectives must keep resolving exactly as before."""
    for objective, expected in (
        ("nq-50pt", "6bfc59cae30f"),
        ("nq-50pt-econ", "e68510982d9f"),
        ("nq-50pt-tbm", "e68510982d9f"),
        ("nq-50pt-xgb-tcn", "7f1672a4bfff"),
    ):
        contract = EvalContract.load_latest(objective)
        if contract is None:
            pytest.skip(f"no contract file on disk for {objective}")
        assert contract.hold_out_hash.startswith(expected)
        assert contract.contract_version == "1"


def test_disagreeing_sources_are_refused_not_ranked(tmp_path, monkeypatch):
    """Two live sources of truth for one hold-out window is how a model gets
    scored against the wrong pin. A divergence raises."""
    eval_root = tmp_path / "evaluation"
    (eval_root / "obj").mkdir(parents=True)
    EvalContract(
        objective_id="obj", contract_version="1", hold_out_hash="b" * 64,
        hold_out_dates=["2026-02-02"], created_at="2026-07-01T00:00:00Z",
    ).save(eval_root / "obj" / "contract_v1.json")
    monkeypatch.setattr("src.evaluation.contract._EVAL_ROOT", eval_root)

    run = _run_with_pin(tmp_path / "runs", objective="obj",
                        run_ts="20260923T000000Z",
                        dates=["2026-03-02"], digest="c" * 64)

    # No run_dir: the file is the pin, unchanged.
    assert EvalContract.load_latest("obj").hold_out_hash == "b" * 64
    # With run_dir and a genuine disagreement: refuse.
    with pytest.raises(ContractHashMismatch):
        EvalContract.load_latest("obj", run_dir=run)


def test_agreeing_sources_resolve_to_the_manifest(tmp_path, monkeypatch):
    eval_root = tmp_path / "evaluation"
    (eval_root / "obj").mkdir(parents=True)
    EvalContract(
        objective_id="obj", contract_version="1", hold_out_hash="d" * 64,
        hold_out_dates=["2026-02-02"], created_at="2026-07-01T00:00:00Z",
    ).save(eval_root / "obj" / "contract_v1.json")
    monkeypatch.setattr("src.evaluation.contract._EVAL_ROOT", eval_root)

    run = _run_with_pin(tmp_path / "runs", objective="obj",
                        run_ts="20260923T000000Z",
                        dates=["2026-02-02"], digest="d" * 64)
    got = EvalContract.load_latest("obj", run_dir=run)
    assert got.hold_out_hash == "d" * 64
    assert got.contract_version == "20260923T000000Z"   # read from the run


def test_a_new_objective_with_no_file_uses_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.contract._EVAL_ROOT", tmp_path / "evaluation")
    run = _run_with_pin(tmp_path / "runs", objective="nq-Rscaled-mkt",
                        run_ts="20260923T000000Z",
                        dates=["2026-06-01"], digest="e" * 64)
    got = EvalContract.load_latest("nq-Rscaled-mkt", run_dir=run)
    assert got is not None and got.hold_out_hash == "e" * 64
