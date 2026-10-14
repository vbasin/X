"""SHAP explainer for XGBoost serving bundles — planted-signal unit test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.evaluation.shap_explain import (
    explain_bundle, default_shap_out_dir, experiment_run_dir,
    shap_values_for_booster, _is_dual_binary_pair,
)


def _write_bundle(tmp_path, feat_cols, model, scaler):
    d = tmp_path / "bundle"
    d.mkdir()
    model.save_model(str(d / "model.ubj"))
    import joblib
    joblib.dump(scaler, d / "scaler.pkl")
    (d / "feature_schema.json").write_text(json.dumps({
        "feature_cols": feat_cols,
        "n_features": len(feat_cols),
    }))
    (d / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": "nq-50pt-econ",
        "candidate_id": "xgb_shap_unit",
        "family": "xgboost",
        "version": "xgb_shap_unit_test",
        "spec_params": {"dir_threshold": 0.5},
        "finalized_at": "2026-06-02T00:00:00+00:00",
    }))
    return d


def _write_days(tmp_path, feat_cols, X, start="2026-06-01"):
    from src.features.hive_5s import persist_session_day

    idx = pd.date_range(f"{start} 13:30", periods=len(X), freq="5s", tz="UTC")
    df = pd.DataFrame(X, columns=feat_cols, index=idx)
    mid = len(df) // 2
    persist_session_day(df.iloc[:mid])
    persist_session_day(df.iloc[mid:])
    return tmp_path


def test_default_shap_out_dir_is_shap_subdir_of_run(tmp_path):
    run = tmp_path / "research" / "custom_test" / "nq-50pt-econ" / "20260821T012808Z"
    (run / "models" / "xgboost").mkdir(parents=True)
    (run / "search_result.json").write_text("{}")
    (run / "run.json").write_text("{}")
    assert experiment_run_dir(run) == run.resolve()
    assert default_shap_out_dir(run) == run.resolve() / "SHAP"
    nested = run / "models" / "xgboost"
    (nested / "model.ubj").write_bytes(b"x")
    assert experiment_run_dir(nested) == run.resolve()
    assert default_shap_out_dir(nested) == run.resolve() / "SHAP"


def test_default_shap_out_dir_staging_bundle(tmp_path):
    bundle = tmp_path / "model_staging" / "nq-50pt-econ" / "xgb_tbm_topgroups_20260715T234308Z"
    bundle.mkdir(parents=True)
    (bundle / "model.ubj").write_bytes(b"x")
    assert experiment_run_dir(bundle) is None
    assert default_shap_out_dir(bundle) == bundle.resolve() / "SHAP"


def test_explain_bundle_ranks_planted_signal(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    from pathlib import Path
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    n = 900
    feat_cols = ["signal", "noise_a", "noise_b"]
    signal = rng.normal(size=n)
    X = np.column_stack([
        signal,
        rng.normal(size=n),
        rng.normal(size=n),
    ]).astype("float64")
    # 3-class labels driven almost only by `signal`.
    y = np.ones(n, dtype=int)
    y[signal > 0.7] = 2
    y[signal < -0.7] = 0

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    model = XGBClassifier(
        n_estimators=40, max_depth=3, learning_rate=0.2,
        subsample=1.0, colsample_bytree=1.0, eval_metric="mlogloss",
        n_jobs=1, verbosity=0,
    )
    model.fit(Xs, y)

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    bundle = _write_bundle(tmp_path, feat_cols, model, scaler)
    _write_days(tmp_path, feat_cols, X)
    monkeypatch.setattr("src.serving.bundle.validate_bundle_path",
                        lambda p, **k: Path(p).resolve())
    out = tmp_path / "shap_out"
    meta = explain_bundle(
        bundle,
        out_dir=out,
        start="2026-06-01",
        end="2026-06-02",
        sample=400,
        tau=0.5,
        seed=1,
    )
    assert meta["n_rows_loaded"] == n
    assert meta["n_features"] == 3
    assert (out / "shap_importance.csv").exists()
    assert (out / "shap_groups.csv").exists()
    assert (out / "SHAP_SUMMARY.md").exists()
    assert (out / "shap_meta.json").exists()
    assert (out / "shap_values_sample.npz").exists()

    rank = pd.read_csv(out / "shap_importance.csv")
    assert rank.iloc[0]["feature"] == "signal"
    assert float(rank.loc[rank["feature"] == "signal", "share_global"].iloc[0]) > 0.5


def test_shap_values_for_dual_binary_pair_scatters_head_private_cols():
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier
    from src.models.tabular_plugins import DualBinaryPair

    rng = np.random.default_rng(1)
    n = 400
    long_s = rng.normal(size=n)
    short_s = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = np.column_stack([long_s, short_s, noise]).astype("float64")
    y_long = (long_s > 0.3).astype("int32")
    y_short = (short_s < -0.3).astype("int32")
    long_clf = XGBClassifier(
        n_estimators=20, max_depth=2, learning_rate=0.3,
        eval_metric="logloss", n_jobs=1, verbosity=0, objective="binary:logistic",
    )
    short_clf = XGBClassifier(
        n_estimators=20, max_depth=2, learning_rate=0.3,
        eval_metric="logloss", n_jobs=1, verbosity=0, objective="binary:logistic",
    )
    long_clf.fit(X[:, [0]], y_long)
    short_clf.fit(X[:, [1]], y_short)
    pair = DualBinaryPair(long_clf, short_clf, long_idx=[0], short_idx=[1])
    assert _is_dual_binary_pair(pair)

    sv, ev, kind = shap_values_for_booster(pair, X)
    assert kind == "dual_binary"
    assert sv.shape == (n, 3, 3)
    # unused feature / unused head slots stay 0
    assert np.allclose(sv[:, 2, :], 0.0)
    assert np.allclose(sv[:, 1, 2], 0.0)  # short col has no long-head SHAP
    assert np.allclose(sv[:, 0, 0], 0.0)  # long col has no short-head SHAP
    assert np.allclose(sv[:, :, 1], 0.0)  # flat unused
    assert float(np.abs(sv[:, 0, 2]).mean()) > float(np.abs(sv[:, 1, 2]).mean())
    assert float(np.abs(sv[:, 1, 0]).mean()) > float(np.abs(sv[:, 0, 0]).mean())
    assert len(ev) == 3 and ev[1] == 0.0


def _write_dual_bundle(tmp_path, feat_cols, pair, scaler):
    import joblib
    d = tmp_path / "dual_bundle"
    d.mkdir()
    joblib.dump(pair, d / "model.pkl")
    joblib.dump(scaler, d / "scaler.pkl")
    (d / "feature_schema.json").write_text(json.dumps({
        "feature_cols": feat_cols,
        "n_features": len(feat_cols),
    }))
    (d / "finalize_manifest.json").write_text(json.dumps({
        "objective_id": "nq-atr-auction",
        "candidate_id": "xgboost",
        "family": "xgboost",
        "version": "xgb_dual_shap_unit",
        "spec_params": {"dir_threshold": 0.5, "head": "dual_binary"},
        "finalized_at": "2026-06-02T00:00:00+00:00",
    }))
    return d


def test_explain_bundle_dual_binary_pair(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    from pathlib import Path
    from xgboost import XGBClassifier
    from src.models.tabular_plugins import DualBinaryPair

    rng = np.random.default_rng(2)
    n = 600
    feat_cols = ["long_signal", "short_signal", "noise"]
    long_s = rng.normal(size=n)
    short_s = rng.normal(size=n)
    X = np.column_stack([
        long_s, short_s, rng.normal(size=n),
    ]).astype("float64")
    y_long = (long_s > 0.4).astype("int32")
    y_short = (short_s < -0.4).astype("int32")

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    long_clf = XGBClassifier(
        n_estimators=30, max_depth=3, learning_rate=0.25,
        eval_metric="logloss", n_jobs=1, verbosity=0, objective="binary:logistic",
    )
    short_clf = XGBClassifier(
        n_estimators=30, max_depth=3, learning_rate=0.25,
        eval_metric="logloss", n_jobs=1, verbosity=0, objective="binary:logistic",
    )
    long_clf.fit(Xs, y_long)
    short_clf.fit(Xs, y_short)
    pair = DualBinaryPair(long_clf, short_clf)

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    bundle = _write_dual_bundle(tmp_path, feat_cols, pair, scaler)
    _write_days(tmp_path, feat_cols, X)
    monkeypatch.setattr("src.serving.bundle.validate_bundle_path",
                        lambda p, **k: Path(p).resolve())
    out = tmp_path / "shap_dual_out"
    meta = explain_bundle(
        bundle,
        out_dir=out,
        start="2026-06-01",
        end="2026-06-02",
        sample=300,
        tau=0.5,
        seed=1,
    )
    assert meta["head"] == "dual_binary"
    assert meta["n_rows_loaded"] == n
    assert meta["class_map"]["up"] == "long"
    rank = pd.read_csv(out / "shap_importance.csv")
    assert "xgb_gain_long" in rank.columns
    assert "xgb_gain_short" in rank.columns
    # Long signal should dominate the long (up) head; short the down head.
    up = rank.set_index("feature")["mean_abs_up"]
    dn = rank.set_index("feature")["mean_abs_down"]
    assert up["long_signal"] > up["noise"]
    assert dn["short_signal"] > dn["noise"]
    summary = (out / "SHAP_SUMMARY.md").read_text()
    assert "dual-binary" in summary.lower()
