"""Run-layout paths, model ids, and the R-scaled objective ids."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiment.artifacts import classify_existing
from src.experiment.model_identity import (
    allocate_model_id,
    oof_core_columns,
    validate_model_id,
    write_model_json,
)
from src.common.run_layout import (
    hpo_json_path,
    merge_run_json,
    model_dir,
    oof_predictions_path,
    population_cache_path,
    trials_summary_path,
    previous_oof_core,
    read_previous_oof,
    resolve_model,
    read_run,
    training_matrix_path,
)
from src.training.cascade_run_spec import (
    move_feature_spec,
    target_feature_spec,
    xgb_hyperparameters,
)
from src.objectives.loader import load_objective
from src.training.label_store import (
    move_outcome_contract,
    outcome_contract_id,
    write_outcome_contract,
)

REPO = Path(__file__).resolve().parents[1]


def test_population_cache_is_shared_by_the_pass():
    """One optional cache per pass. The legacy helper still names the old per-model path."""
    mid = allocate_model_id(
        "nq-Rscaled-mkt", "20260922T000000Z", 1, "xgboost",
        {"max_depth": 6, "n_estimators": 300})
    cache = population_cache_path("nq-Rscaled-mkt", "20260922T000000Z", 1)
    legacy = training_matrix_path("nq-Rscaled-mkt", "20260922T000000Z", 1, mid)
    oof = oof_predictions_path("nq-Rscaled-mkt", "20260922T000000Z", 1, mid)
    assert cache.name == "training_matrix.parquet"
    assert cache.parent.name == "pass_1"
    assert legacy.parent.name == f"model_{mid}"
    assert oof.parent.name == f"model_{mid}"
    assert "xgb" not in oof.parent.name


def test_hpo_trials_share_one_run_and_one_cache():
    """Ten hyperparameter tweaks are ten model folders under one pass and one fold."""
    run_ts = "20260923T000000Z"
    spec = {"view": "numeric_non_leak"}
    shallow = allocate_model_id("obj", run_ts, 1, "xgboost", {"max_depth": 4}, spec, "y_move")
    deeper = allocate_model_id("obj", run_ts, 1, "xgboost", {"max_depth": 6}, spec, "y_move")
    assert shallow != deeper
    assert model_dir("obj", run_ts, 1, shallow).parent == model_dir("obj", run_ts, 1, deeper).parent
    assert population_cache_path("obj", run_ts, 1) == model_dir("obj", run_ts, 1, shallow).parent / "training_matrix.parquet"
    assert hpo_json_path("obj", run_ts).name == "hpo.json"
    assert trials_summary_path("obj", run_ts).name == "trials_summary.parquet"
    assert resolve_model("obj", run_ts, 1, shallow)["trial"].name == "trial.json"
    assert classify_existing("hpo.json") == ("canonical", "hpo_study")
    assert classify_existing(f"pass_1/model_{shallow}/trial.json") == ("canonical", None)


def test_model_type_is_not_a_model_id():
    with pytest.raises(ValueError):
        model_dir("nq-Rscaled-mkt", "20260922T000000Z", 2, "xgb")
    with pytest.raises(ValueError):
        validate_model_id("xgboost", "xgboost")
    assert validate_model_id("a1b2c3d4e5f6") == "a1b2c3d4e5f6"


def test_allocate_model_id_depends_on_the_pass():
    hyper = {"max_depth": 6, "n_estimators": 300}
    a = allocate_model_id("nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", hyper)
    b = allocate_model_id("nq-Rscaled-mkt", "20260923T005724Z", 2, "xgboost", hyper)
    assert a != b
    assert len(a) == 12


def test_allocate_model_id_is_the_configuration():
    """Same config hashes the same. A different fit in the same pass does not."""
    depth6 = xgb_hyperparameters(
        n_estimators=300, max_depth=6, learning_rate=0.05, seed=42)
    depth10 = xgb_hyperparameters(
        n_estimators=500, max_depth=10, learning_rate=0.05, seed=42)
    features = ["es_b", "es_a"]
    same = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost",
        {"seed": 42, "max_depth": 6, "n_estimators": 300, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"},
        feature_spec={"columns": list(features)}, target="y_move")
    same_columns = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", depth6,
        feature_spec={"columns": ["es_a", "es_b"]}, target="y_move")
    deeper = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", depth10,
        feature_spec={"columns": features}, target="y_move")
    other_features = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", depth6,
        feature_spec={"columns": ["es_a", "es_c"]}, target="y_move")
    other_target = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", depth6,
        feature_spec={"columns": features}, target="y_reach")
    assert same == same_columns
    assert len({same, deeper, other_features, other_target}) == 4
    assert deeper != same
    assert other_features != same


def test_feature_view_changes_the_model_id():
    hyper = xgb_hyperparameters(
        n_estimators=300, max_depth=6, learning_rate=0.05, seed=42)
    kept = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", hyper,
        feature_spec=move_feature_spec(drop_order_flow=False), target="y_move")
    dropped = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost", hyper,
        feature_spec=move_feature_spec(drop_order_flow=True), target="y_move")
    curated = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 2, "xgboost", hyper,
        feature_spec=target_feature_spec(context="curated"), target="y_reach")
    wider = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 2, "xgboost", hyper,
        feature_spec=target_feature_spec(context="all"), target="y_reach")
    assert kept != dropped
    assert curated != wider


def test_explicit_model_id_is_not_rehashed():
    allocated = allocate_model_id(
        "nq-Rscaled-mkt", "20260923T005724Z", 1, "xgboost",
        {"max_depth": 6, "n_estimators": 300},
        feature_spec=["es_a"], target="y_move")
    explicit = "abc123abc123"
    chosen = explicit or allocated
    assert validate_model_id(chosen) == explicit
    assert chosen != allocated


def test_oof_core_columns_drop_features_and_keep_predictions():
    cols = oof_core_columns(
        ["ts", "y_move", "p_move", "es_x", "tau_0.95"],
        ["ts", "y_move", "es_x", "p_move"],
        ["ts", "y_move"],
    )
    assert cols == ["ts", "y_move", "p_move", "tau_0.95"]


def test_merge_keeps_the_pass_role(monkeypatch, tmp_path):
    monkeypatch.setattr("src.common.run_layout.runs_dir", lambda: tmp_path)
    merge_run_json("obj", "t", {
        "pass": 1,
        "role": "directional_screen",
        "purpose": "Score the decision grid.",
        "models": [{"model_id": "abc123abc123", "model_type": "xgboost"}],
    })
    merge_run_json("obj", "t", {
        "pass": 1,
        "models": [{
            "model_id": "abc123abc123",
            "model_type": "xgboost",
            "oof_predictions": "pass_1/model_abc123abc123/oof_predictions.parquet",
        }],
    })
    assert not (tmp_path / "obj" / "t" / "run.json").exists()
    doc = read_run("obj", "t")
    assert doc["passes"][0]["role"] == "directional_screen"
    assert doc["passes"][0]["purpose"] == "Score the decision grid."
    assert doc["passes"][0]["models"][0]["oof_predictions"].endswith("oof_predictions.parquet")


def test_legacy_run_files_remain_readable(monkeypatch, tmp_path):
    monkeypatch.setattr("src.common.run_layout.runs_dir", lambda: tmp_path)
    root = tmp_path / "obj" / "t"
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({
        "objective_id": "obj",
        "run_ts": "t",
        "passes": [{"pass": 1, "role": "directional_screen",
                    "models": [{"model_id": "abc123abc123"}]}],
    }))
    (root / "contract.json").write_text(json.dumps(
        {"objective_snapshot": {"objective_id": "obj", "labels": {"entry": "market"}}}))
    (root / "folds.json").write_text(json.dumps(
        {"split": "expanding_month_blocks", "n_folds": 2}))
    doc = read_run("obj", "t")
    assert doc["passes"][0]["role"] == "directional_screen"
    assert doc["config_snapshot"]["labels"]["entry"] == "market"
    assert doc["validation"]["n_folds"] == 2
    assert not (root / "manifest.json").exists()


def test_previous_oof_is_the_core_columns_only(monkeypatch, tmp_path):
    monkeypatch.setattr("src.common.run_layout.runs_dir", lambda: tmp_path)
    mid = "abc123abc123"
    directory = model_dir("obj", "t", 1, mid)
    directory.mkdir(parents=True)
    pd.DataFrame({
        "ts": [1], "y_move": [1], "es_x": [0.2], "p_move": [0.4],
    }).to_parquet(directory / "oof_predictions.parquet", index=False)
    write_model_json(
        directory, model_id=mid, model_type="xgboost", hyperparameters={"max_depth": 6},
        oof_core_columns=["ts", "y_move", "p_move"], label_contract_id="contract123",
    )
    merge_run_json("obj", "t", {
        "pass": 1,
        "role": "directional_screen",
        "purpose": "Score the decision grid.",
        "label_contract_id": "contract123",
        "models": [{"model_id": mid, "model_type": "xgboost"}],
    })
    _path, cols = previous_oof_core("obj", "t", 2)
    assert cols == ["ts", "y_move", "p_move"]
    frame = read_previous_oof("obj", "t", 2)
    assert list(frame.columns) == ["ts", "y_move", "p_move"]
    # report/ is created by the step that writes a report, not by model.json.
    assert not (directory / "report").exists()


def test_a_run_references_a_label_contract_without_rewriting_it(monkeypatch, tmp_path):
    """The contract stays reusable. The manifest stores the id."""
    monkeypatch.setattr("src.common.run_layout.runs_dir", lambda: tmp_path / "runs")
    labels = tmp_path / "labels"
    contract = move_outcome_contract(
        decision_grid_seconds=5, horizon_seconds=1800,
        move_r=0.25, adverse_r=0.125, r_lookback_sessions=5)
    cid = write_outcome_contract(labels, contract)
    before = (labels / "_contract.json").read_text()
    merge_run_json("nq-Rscaled-mkt", "20260923T005724Z", {
        "pass": 1,
        "role": "directional_screen",
        "purpose": "Score the decision grid.",
        "label_contract_id": cid,
        "upstream": [{
            "kind": "labels",
            "label_contract_id": cid,
            "location": "sources/labels/example",
            "semantic_type": "outcomes",
        }],
        "models": [{"model_id": "abc123abc123", "model_type": "xgboost"}],
    })
    assert (labels / "_contract.json").read_text() == before
    stored = json.loads(before)
    assert "applies_to" not in stored
    assert stored["contract_id"] == cid == outcome_contract_id(contract)
    doc = read_run("nq-Rscaled-mkt", "20260923T005724Z")
    assert doc["passes"][0]["label_contract_id"] == cid


def test_rscaled_objectives_validate():
    mkt = load_objective("nq-Rscaled-mkt")
    limit = load_objective("nq-Rscaled-limit")
    fixed = load_objective("nq-Rscaled-fixedSL")
    firstpass = load_objective("mnq-auction-firstpass")
    assert mkt["labels"]["entry"] == "market"
    assert limit["labels"]["entry"] == "limit"
    assert "stop_rule" not in fixed["labels"]
    assert fixed["labels"]["mae_tolerance"] == 0.45
    assert mkt["labels"]["stop_rule"]["max"] == 0.25
    assert firstpass["objective_id"] == "mnq-auction-firstpass"
    assert firstpass["labels"]["scheme"] == "candidate_exit_surface"
    assert firstpass["labels"]["outcome"]["target"] == "pnl_r"
    assert firstpass["horizon"]["max_trade_horizon"] == "session_close"
    assert "move_threshold" not in firstpass["labels"]
    assert not (REPO / "config" / "objectives" / "nq-Rscaled-adaptiveSL.yaml").exists()
    assert not (REPO / "config" / "objectives" / "nq-r-first-passage-30m.yaml").exists()


def test_run_layout_is_only_a_path_resolver():
    """Step 3 guard. A path module that knows model types or strategy pass
    roles has become a second metadata system. Model identity lives in
    src/experiment/model_identity.py; the cascade's pass roles and
    feature views live in src/training/cascade_run_spec.py."""
    import src.common.run_layout as rl

    source = Path(rl.__file__).read_text()
    body = source.split('"""', 2)[-1]  # skip the module docstring
    forbidden = (
        "xgboost", "xgb", "lgbm", "lightgbm", "catboost", "keras",
        "directional_screen", "target_reach", "move_feature", "target_feature",
        "CASCADE_PASS",
    )
    hits = [token for token in forbidden if token in body]
    assert hits == [], f"model/strategy vocabulary leaked back into run_layout: {hits}"

    # Importing validate_model_id to keep a bad id out of a directory name is
    # path safety. DEFINING model identity here is the drift this guards.
    defined_here = {
        name for name, value in vars(rl).items()
        if getattr(value, "__module__", None) == rl.__name__
    }
    assert not defined_here & {
        "allocate_model_id", "validate_model_id", "write_model_json",
        "oof_core_columns", "xgb_hyperparameters", "move_feature_spec",
        "target_feature_spec",
    }, "run_layout must not define model identity or strategy vocabulary"


def test_model_json_does_not_create_an_empty_report_directory(tmp_path):
    """STORAGE_CONTRACT §21: files meaningful only after a step ran are written
    by that step, not as placeholders. That includes report/."""
    write_model_json(
        tmp_path, model_id="a1b2c3d4e5f6", model_type="xgboost",
        hyperparameters={"max_depth": 6}, target="y_move",
    )
    assert (tmp_path / "model.json").is_file()
    assert not (tmp_path / "report").exists()
