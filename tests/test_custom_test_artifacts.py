"""Custom Test lab layout: nested models/, GPU collect, meta_stack bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.oci_queue import LocalMessageQueue
from src.common.object_store import LocalBackend
from src.gpu_runner.dispatch import collect_result, iter_job_artifacts, publish_result, upload_job_artifacts
from src.search.finalize import (
    lab_model_dir, write_meta_stack_lab_bundle, write_primary_meta_lab_bundle,
    write_serve_bundle,
)
from src.serving.bundle import is_serving_bundle, load_serving_bundle


def _oof_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-02", periods=n, freq="5s", tz="UTC")
    side = rng.choice(["up", "down", "flat"], size=n, p=[0.3, 0.3, 0.4])
    conf = rng.uniform(0.4, 0.95, size=n)
    p_up = np.where(side == "up", conf, rng.uniform(0.05, 0.4, size=n))
    p_down = np.where(side == "down", conf, rng.uniform(0.05, 0.4, size=n))
    p_flat = np.clip(1.0 - p_up - p_down, 0.0, 1.0)
    truth = rng.choice(["up", "down", "flat"], size=n, p=[0.25, 0.25, 0.5])
    return pd.DataFrame({
        "ts": ts, "direction": side, "confidence": conf,
        "p_down": p_down, "p_flat": p_flat, "p_up": p_up,
        "direction_true": truth,
    })


def _mk_last_fold_dir(d: Path, family: str, oof: pd.DataFrame) -> Path:
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0.0], [1.0]], [0, 1])
    write_serve_bundle(
        d, artifact=clf, feature_cols=["f1"],
        objective_id="nq-50pt-econ", candidate_id=family, family=family,
        run_ts="20260820T000000Z", extra_manifest={"weights_source": "last_walk_forward_fold"},
    )
    oof.to_parquet(d / "oof.parquet", index=False)
    return d


def test_publish_collect_nested_models(tmp_path):
    store = LocalBackend(tmp_path / "store")
    queue = LocalMessageQueue(tmp_path / "q")
    out = tmp_path / "job_out"
    dest = lab_model_dir(out, "xgboost")
    dest.mkdir(parents=True)
    (dest / "model.ubj").write_bytes(b"UBJ")
    (dest / "feature_schema.json").write_text("{}")
    (dest / "oof.parquet").write_bytes(b"OOF")
    (dest / "diagnostics.json").write_text(json.dumps({"ok": True}))

    publish_result(queue, store, objective_id="obj", run_id="r_ct",
                   code_digest="d", ran_digest="d",
                   result={"status": "ok", "family": "xgboost"},
                   out_dir=out)
    assert store.exists("results/r_ct/models/xgboost/oof.parquet")
    assert store.exists("results/r_ct/models/xgboost/model.ubj")

    collected = tmp_path / "collected"
    oof_path, diag = collect_result(store, queue.get(channel="results.obj", max_msgs=1, wait_s=0)[0].body,
                                    collected)
    assert Path(oof_path).read_bytes() == b"OOF"
    assert diag == {"ok": True}
    assert (collected / "models" / "xgboost" / "model.ubj").read_bytes() == b"UBJ"
    assert not store.exists("results/r_ct/models/xgboost/oof.parquet")


def test_write_meta_stack_lab_bundle(tmp_path):
    oof_x = _oof_frame(60, 1)
    oof_l = _oof_frame(60, 2)
    bx = _mk_last_fold_dir(tmp_path / "xgboost", "xgboost", oof_x)
    bl = _mk_last_fold_dir(tmp_path / "attention_lstm", "attention_lstm", oof_l)
    composed = oof_x.copy()
    dest = tmp_path / "run" / "models" / "meta_stack__xgboost_attention_lstm"
    man = write_meta_stack_lab_bundle(
        dest,
        base_bundle_dirs={"xgboost": bx, "attention_lstm": bl},
        composed_oof=composed,
        objective_id="nq-50pt-econ",
        candidate_id="meta_stack__xgboost_attention_lstm",
        run_ts="20260820T000000Z",
        search_run_dir=tmp_path / "run",
        search_score={"net_expectancy": 1.0},
        spec_params={
            "bases": ["xgboost", "attention_lstm"],
            "meta": {"feature": "linear_weights", "approve_threshold": 0.55},
        },
        diagnostics={"note": "test"},
    )
    assert is_serving_bundle(dest)
    assert (dest / "meta.pkl").exists()
    assert (dest / "oof.parquet").exists()
    assert (dest / "bases" / "xgboost" / "feature_schema.json").exists()
    assert (dest / "diagnostics.json").exists()
    assert man["family"] == "meta_stack"
    assert man["weights_source"] == "last_walk_forward_fold"
    bundle = load_serving_bundle(dest, validate=False)
    assert bundle.get("meta_stack") is True
    assert bundle.get("family") == "meta_stack"
    assert {b["family"] for b in bundle["bases"]} == {"xgboost", "attention_lstm"}


def test_lab_model_dir():
    assert lab_model_dir(Path("/r"), "xgboost") == Path("/r/models/xgboost")


def test_write_primary_meta_lab_bundle(tmp_path):
    oof_x = _oof_frame(80, 3)
    bx = _mk_last_fold_dir(tmp_path / "xgboost", "xgboost", oof_x)
    dest = tmp_path / "run" / "models" / "meta__xgboost"
    composed = oof_x.copy()
    composed["primary_direction"] = composed["direction"]
    composed["primary_confidence"] = composed["confidence"]
    composed["meta_prob"] = 0.7
    man = write_primary_meta_lab_bundle(
        dest,
        primary_bundle_dir=bx,
        primary_id="xgboost",
        fitted_meta={
            "kind": "constant", "estimator": 0.9,
            "feature_names": ["primary_confidence", "is_up"],
            "impute_values": {}, "n_train": 40,
        },
        composed_oof=composed,
        objective_id="nq-50pt-econ",
        candidate_id="meta__xgboost",
        run_ts="20260821T012808Z",
        search_run_dir=tmp_path / "run",
        search_score={"net_expectancy": 1.0},
        spec_params={
            "primary": "xgboost",
            "bases": ["xgboost"],
            "meta": {"family": "xgboost", "approve_threshold": 0.5,
                     "min_primary_confidence": 0.5},
        },
        diagnostics={"note": "test"},
        context_view="regime_core",
    )
    assert is_serving_bundle(dest)
    assert (dest / "meta.pkl").exists()
    assert (dest / "meta.json").exists()
    assert (dest / "finalize_manifest.json").exists()
    assert (dest / "feature_schema.json").exists()
    assert (dest / "bases" / "xgboost" / "model.pkl").exists() or (
        dest / "bases" / "xgboost" / "feature_schema.json").exists()
    assert man["family"] == "primary_meta"
    bundle = load_serving_bundle(dest, validate=False)
    assert bundle.get("primary_meta") is True
    assert bundle.get("family") == "primary_meta"
    assert bundle["bases"][0]["family"] == "xgboost"


def test_collect_lists_prefix_when_result_files_empty(tmp_path):
    """Stale GPU runner publishes scores with result_files=[] after nesting OOF under models/."""
    store = LocalBackend(tmp_path / "store")
    run_id = "r_stale"
    store.put(f"results/{run_id}/models/xgboost/model.ubj", b"UBJ")
    store.put(f"results/{run_id}/models/xgboost/feature_schema.json", b"{}")
    store.put(f"results/{run_id}/models/xgboost/finalize_manifest.json", b"{}")
    store.put(f"results/{run_id}/models/xgboost/scaler.pkl", b"SCL")
    store.put(f"results/{run_id}/models/xgboost/oof.parquet", b"OOF")
    store.put(f"results/{run_id}/oof_xgboost.parquet", b"OOF2")
    dest = tmp_path / "collected"
    oof_path, _diag = collect_result(
        store, {"run_id": run_id, "result_files": []}, dest)
    assert (dest / "models" / "xgboost" / "model.ubj").read_bytes() == b"UBJ"
    assert (dest / "models" / "xgboost" / "scaler.pkl").read_bytes() == b"SCL"
    assert (dest / "models" / "xgboost" / "feature_schema.json").exists()
    assert (dest / "oof_xgboost.parquet").read_bytes() == b"OOF2"
    assert Path(oof_path).exists()
    assert not store.exists(f"results/{run_id}/models/xgboost/model.ubj")


def test_iter_job_artifacts_includes_lab_bundle_and_root_oof(tmp_path):
    out = tmp_path / "job"
    lab = lab_model_dir(out, "xgboost")
    lab.mkdir(parents=True)
    (lab / "model.ubj").write_bytes(b"UBJ")
    (lab / "oof.parquet").write_bytes(b"NESTED")
    (out / "oof_xgboost.parquet").write_bytes(b"ROOT")
    (out / "diagnostics_xgboost.json").write_text("{}")
    (out / "progress.json").write_text("{}")
    (out / "job.log").write_text("epoch fold=1/6 epoch=1/15\n")
    folds = out / "folds"
    folds.mkdir()
    (folds / "fold_00.json").write_text("{}")
    rels = {p.relative_to(out).as_posix() for p in iter_job_artifacts(out)}
    assert "models/xgboost/model.ubj" in rels
    assert "oof_xgboost.parquet" in rels
    assert "diagnostics_xgboost.json" in rels
    assert "progress.json" in rels
    assert "job.log" in rels
    assert "folds/fold_00.json" in rels
    store = LocalBackend(tmp_path / "store")
    files = upload_job_artifacts(store, "r_up", out)
    assert "models/xgboost/model.ubj" in files
    assert store.exists("results/r_up/models/xgboost/model.ubj")


def test_resolve_experiment_type_webui_spec(tmp_path):
    from src.search.controller import resolve_experiment_type

    spec = tmp_path / "ops" / "webui" / "custom_tests" / "nq-50pt-econ" / "x.manual.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text("candidates: []\n")
    topo = spec.with_name("x.topology.yaml")
    assert resolve_experiment_type(None, spec) == "custom_test"
    assert resolve_experiment_type("", spec) == "custom_test"
    assert resolve_experiment_type(None, None, topo) == "custom_test"
    assert resolve_experiment_type("manual_override", spec) == "manual_override"
    assert resolve_experiment_type(None, "config/manual_candidates.example.yaml") == "manual_override"
    assert resolve_experiment_type(None, {"candidates": []}) == "manual_override"
    assert resolve_experiment_type(None, None) == "model_selection"
    assert resolve_experiment_type("custom_test", None) == "custom_test"
