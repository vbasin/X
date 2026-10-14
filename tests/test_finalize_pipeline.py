"""Tests: ADR-024 pipeline fixes — results-outbox cleanup, executed-config echo,
MLflow wave logging, and the champion-finalize bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.oci_queue import LocalMessageQueue
from src.common.object_store import LocalBackend
from src.gpu_runner.dispatch import collect_result, publish_result


# ── results/ outbox cleanup (Deviation B) ─────────────────────────────────────

def test_collect_result_deletes_outbox_after_download(tmp_path):
    store = LocalBackend(tmp_path / "store")
    queue = LocalMessageQueue(tmp_path / "q")
    out = tmp_path / "job_out"
    out.mkdir()
    (out / "oof_x.parquet").write_bytes(b"PARQ")
    (out / "diagnostics_x.json").write_text(json.dumps({"f1_macro": 0.5}))

    publish_result(queue, store, objective_id="obj", run_id="r1",
                   code_digest="d", ran_digest="d",
                   result={"status": "ok", "family": "xgboost",
                           "spec_params": {"max_depth": 9},
                           "feature_selection": {"view": "regime_min"}},
                   out_dir=out)
    assert store.exists("results/r1/oof_x.parquet")

    msgs = queue.get(channel="results.obj", max_msgs=5, wait_s=0)
    body = msgs[0].body
    oof_path, diag = collect_result(store, body, tmp_path / "collected")
    assert oof_path and Path(oof_path).read_bytes() == b"PARQ"
    assert diag == {"f1_macro": 0.5}
    # outbox cleaned after verified download
    assert not store.exists("results/r1/oof_x.parquet")
    assert not store.exists("results/r1/diagnostics_x.json")


def test_result_body_echoes_executed_config(tmp_path):
    store = LocalBackend(tmp_path / "store")
    queue = LocalMessageQueue(tmp_path / "q")
    publish_result(queue, store, objective_id="obj", run_id="r2",
                   code_digest="d", ran_digest="d",
                   result={"status": "ok", "family": "xgboost",
                           "spec_params": {"max_depth": 9, "n_estimators": 50},
                           "feature_selection": {"prefixes": ["rv_"]}},
                   out_dir=tmp_path / "empty")
    body = queue.get(channel="results.obj", max_msgs=1, wait_s=0)[0].body
    assert body["spec_params"] == {"max_depth": 9, "n_estimators": 50}
    assert body["feature_selection"] == {"prefixes": ["rv_"]}


def test_collector_prefers_echoed_params_over_catalog_defaults(tmp_path):
    from src.models.catalog import get_entry
    from src.search.controller import _collect_gpu_result

    store = LocalBackend(tmp_path / "store")
    entry = get_entry("xgboost")
    body = {"run_id": "r3", "status": "ok", "score": {"precision_at_tau": 0.9},
            "spec_params": {"max_depth": 11}, "feature_selection": {"view": "regime_min"},
            "result_files": [], "elapsed_s": 1.0}
    res = _collect_gpu_result(store, body, entry, tmp_path / "out")
    assert res.spec_params == {"max_depth": 11}          # echoed, not catalog default
    assert res.feature_selection == {"view": "regime_min"}

    body_legacy = dict(body, spec_params=None, feature_selection=None)
    res2 = _collect_gpu_result(store, body_legacy, entry, tmp_path / "out2")
    assert res2.spec_params == dict(entry.params_default)  # legacy fallback


# ── MLflow wave logging ───────────────────────────────────────────────────────

def test_write_serve_bundle_is_promotable(tmp_path):
    from sklearn.dummy import DummyClassifier
    from src.search.finalize import write_serve_bundle
    from src.serving.bundle import is_serving_bundle

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0.0], [1.0]], [0, 1])
    dest = tmp_path / "xgboost"
    man = write_serve_bundle(
        dest,
        artifact=clf,
        feature_cols=["f1", "f2"],
        objective_id="nq-50pt-econ",
        candidate_id="xgboost",
        family="xgboost",
        run_ts="20260815T202407Z",
        extra_manifest={
            "weights_source": "last_walk_forward_fold",
            "inference_policy": {
                "tau": 0.65, "trading_start": "08:30", "trading_end": "10:00",
                "sides": ["up"], "cooldown_bars": 120,
            },
        },
    )
    assert is_serving_bundle(dest)
    assert (dest / "model.pkl").exists()
    assert (dest / "feature_schema.json").exists()
    assert (dest / "inference_policy.json").exists()
    assert man["weights_source"] == "last_walk_forward_fold"
    assert man["version"] == "xgboost_20260815T202407Z"
    assert man["inference_policy"]["tau"] == 0.65


def test_log_wave_to_mlflow_sets_run_ids(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    from src.search.controller import CandidateResult, _log_wave_to_mlflow

    results = [
        CandidateResult(candidate_id="xgb_a", family="xgboost",
                        spec_params={"max_depth": 4}, status="ok",
                        score={"precision_at_tau": 0.8, "n_signals": 10,
                               "nested": {"x": 1}, "nan_metric": float("nan")},
                        feature_selection={"view": "regime_min"}),
        CandidateResult(candidate_id="broken", family="hmm",
                        spec_params={}, status="failed"),
    ]
    _log_wave_to_mlflow(results, {"objective_id": "test-obj"}, "20990101T000000Z")

    assert results[0].mlflow_run_id            # ok candidate got a run id
    assert results[1].mlflow_run_id is None    # failed candidate skipped

    run = mlflow.get_run(results[0].mlflow_run_id)
    assert run.data.tags["candidate_id"] == "xgb_a"
    assert run.data.tags["experiment_type"] == "model_selection"
    assert run.data.params["p_max_depth"] == "4"
    assert run.data.metrics["precision_at_tau"] == 0.8
    assert "nan_metric" not in run.data.metrics        # NaN dropped


def test_log_wave_to_mlflow_skips_unreachable_server(monkeypatch, caplog):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    from src.search.controller import CandidateResult, _log_wave_to_mlflow

    results = [CandidateResult(candidate_id="xgb_a", family="xgboost",
                               spec_params={}, status="ok",
                               score={"precision_at_tau": 0.8})]
    with caplog.at_level("WARNING", logger="vtech.search.controller"):
        _log_wave_to_mlflow(results, {"objective_id": "test-obj"}, "20990101T000000Z")
    assert results[0].mlflow_run_id is None
    assert any("mlflow_wave_log_skipped" in r.message for r in caplog.records)


# ── finalize bridge ───────────────────────────────────────────────────────────

def _tiny_search_env(tmp_path, monkeypatch):
    """Synthetic parquet dir + search run dir with an xgboost champion."""
    import src.search.finalize as fin
    import src.search.fold_cache as fc
    import src.common.data_paths as dp

    monkeypatch.setattr(fin, "_DATA_ROOT", tmp_path)
    monkeypatch.setattr(fc, "_CACHE_ROOT", tmp_path / "_fold_cache")
    monkeypatch.setattr(dp, "model_staging_dir", lambda: tmp_path / "model_staging")
    monkeypatch.setattr(fin, "model_staging_dir", lambda: tmp_path / "model_staging")

    pdir = tmp_path / "parquet"
    pdir.mkdir()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-05 14:00", periods=900, freq="5s", tz="UTC")
    close = 20000 + np.cumsum(rng.normal(0, 12, size=900))
    pd.DataFrame({"rv_a": rng.normal(size=900), "vw_b": rng.normal(size=900),
                  "_close": close}, index=idx).to_parquet(pdir / "features_2026-01-05.parquet")

    run_dir = tmp_path / "search" / "t-obj" / "20990101T000000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "search_result.json").write_text(json.dumps({
        "objective_id": "t-obj", "run_ts": "20990101T000000Z", "champion": "xgb_win",
        "gate_result": {"status": "champion"},
        "candidates": [
            {"candidate_id": "xgb_win", "family": "xgboost", "status": "ok",
             "spec_params": {"n_estimators": 10, "max_depth": 2},
             "feature_selection": {"prefixes": ["rv_", "vw_"]},
             "score": {"precision_at_tau": 0.9}},
            {"candidate_id": "ens", "family": "ensemble_vote", "status": "ok",
             "spec_params": {}, "score": {}},
        ]}))
    obj = {"objective_id": "t-obj",
           "labels": {"move_threshold": 50, "mae_tolerance": 6, "scheme": "triple_barrier"},
           "horizon": {"max_trade_horizon": "10m"}, "eval": {"walk_forward_folds": 2},
           "kpi": {"primary": "precision_at_tau",
                   "requirement": {"any": [{"metric": "precision_at_tau", "direction": "up",
                                            "tau": 0.8, "min": 0.85}]},
                   "target": {"all": [{"metric": "precision_at_tau", "direction": "up",
                                       "tau": 0.9, "min": 0.95}]}},
           "traded products": [{"symbol": "NQ", "role": "primary"}],
           "description": "synthetic finalize test objective"}
    monkeypatch.setattr("src.objectives.loader.load_objective", lambda _id: obj)
    return pdir, run_dir


def test_finalize_champion_produces_promotable_bundle(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    from src.search.finalize import finalize_champion

    pdir, run_dir = _tiny_search_env(tmp_path, monkeypatch)
    manifest = finalize_champion("t-obj", run_dir=run_dir, parquet_dir=pdir, stage=True)

    version = manifest["version"]
    sdir = tmp_path / "model_staging" / "t-obj" / version
    # staging only (training_runs dual-write removed)
    assert (sdir / manifest["model_file"]).exists()
    assert (sdir / "feature_schema.json").exists()
    assert (sdir / "finalize_manifest.json").exists()
    assert not (tmp_path / "training_runs" / "t-obj" / version).exists()
    schema = json.loads((sdir / "feature_schema.json").read_text())
    assert set(schema["feature_cols"]) == {"rv_a", "vw_b"}   # selection applied
    assert manifest["spec_params"] == {"n_estimators": 10, "max_depth": 2}
    assert manifest["has_scaler"] is True                    # tabular plugin scaler persisted
    assert (sdir / "scaler.pkl").exists()
    assert manifest["mlflow_run_id"]                         # model browsable in MLflow


def test_finalize_rejects_ensembles_and_dry_runs(tmp_path, monkeypatch):
    from src.search.finalize import finalize_champion
    pdir, run_dir = _tiny_search_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="ensemble"):
        finalize_champion("t-obj", run_dir=run_dir, candidate_id="ens", parquet_dir=pdir)
    m = finalize_champion("t-obj", run_dir=run_dir, parquet_dir=pdir, dry_run=True)
    assert m["dry_run"] and m["candidate_id"] == "xgb_win"
