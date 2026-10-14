"""Leaderboard artifacts live under data/models/leaders/<objective_id>/."""

from __future__ import annotations

import pytest

from src.common.data_paths import leaders_dir, resolve_data_path
from src.evaluation.leaderboard import get_leaders, load_leaderboard, record_result


def test_leaders_dir_is_models_leaders(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    assert leaders_dir() == tmp_path / "models" / "leaders"


def test_record_result_writes_under_models_leaders(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    path = record_result(
        objective_id="nq-50pt-econ",
        contract_version="1",
        candidate_id="xgb_test",
        model_family="xgboost",
        score_result={
            "primary_precision": 0.8,
            "primary_dir_f1": 0.5,
            "primary_tau": 0.8,
            "per_tau": [{}],
        },
    )
    expected = tmp_path / "models" / "leaders" / "nq-50pt-econ" / "leaderboard_v1.jsonl"
    assert path == expected
    assert path.exists()
    assert (tmp_path / "models" / "leaders" / "nq-50pt-econ" / "latest_leaders.json").exists()
    assert not (tmp_path / "nq-50pt-econ").exists()
    assert not (tmp_path / "evaluation" / "nq-50pt-econ").exists()
    rows = load_leaderboard("nq-50pt-econ", "1")
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "xgb_test"
    assert get_leaders("nq-50pt-econ", "1")[0]["candidate_id"] == "xgb_test"


def test_resolve_data_path_models_leaders(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    # Use an id that is not on the live volume so the exists-as-given short-circuit
    # does not return cwd-relative data/models/leaders/….
    got = resolve_data_path("data/models/leaders/nq-50pt-missing/leaderboard_v1.jsonl")
    assert got == tmp_path / "models" / "leaders" / "nq-50pt-missing" / "leaderboard_v1.jsonl"
    assert resolve_data_path("data/search/nq-50pt-missing/run") == (
        tmp_path / "research" / "search" / "nq-50pt-missing" / "run"
    )
    assert resolve_data_path("data/evaluation/episodes/nq-50pt-missing") == (
        tmp_path / "research" / "episodes" / "nq-50pt-missing"
    )
    assert resolve_data_path("data/evaluation/nq-50pt-missing/contract_v1.json") == (
        tmp_path / "evaluation" / "nq-50pt-missing" / "contract_v1.json"
    )


def test_resolve_oof_source_missing_run_names_the_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    (tmp_path / "research" / "runs").mkdir(parents=True)
    from src.backtest.runner import _resolve_oof_source
    with pytest.raises(FileNotFoundError, match="research/runs") as exc:
        _resolve_oof_source("missing-run", "nq-50pt")
    assert "search_root" not in str(exc.value)


def test_resolve_oof_source_finds_a_canonical_model_oof(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    model = (tmp_path / "research" / "runs" / "nq-Rscaled-mkt"
             / "20260101T000000Z" / "pass_2" / "model_abc")
    model.mkdir(parents=True)
    oof = model / "oof_predictions.parquet"
    oof.write_bytes(b"")
    from src.backtest.runner import _resolve_oof_source
    assert _resolve_oof_source("20260101T000000Z", "nq-Rscaled-mkt") == oof


def test_resolve_oof_source_prefers_a_run_level_oof(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    run = tmp_path / "research" / "runs" / "nq-50pt-tbm" / "20260101T000000Z"
    model = run / "pass_1" / "model_xgboost"
    model.mkdir(parents=True)
    root = run / "oof_xgboost.parquet"
    root.write_bytes(b"root")
    (model / "oof_predictions.parquet").write_bytes(b"nested")
    from src.backtest.runner import _resolve_oof_source
    assert _resolve_oof_source("20260101T000000Z", "nq-50pt-tbm") == root


def test_resolve_oof_source_uses_search_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    run = tmp_path / "research" / "search" / "nq-50pt-econ" / "20260101T000000Z"
    run.mkdir(parents=True)
    oof = run / "oof_xgboost.parquet"
    oof.write_bytes(b"")
    from src.backtest.runner import _resolve_oof_source
    assert _resolve_oof_source("20260101T000000Z", "nq-50pt-econ") == oof
    assert _resolve_oof_source("20260101T000000Z", "nq-50pt-econ") != (
        tmp_path / "nq-50pt-econ" / "20260101T000000Z" / "oof_xgboost.parquet"
    )
