"""ADR-100: tf_oof_meta serving bundle + asof predict for Backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.objectives.loader import load_objective
from src.search.finalize import write_serve_bundle
from src.search.tf_oof_meta import (
    pack_lab_tf_oof_meta, select_meta_features, write_tf_oof_meta_serving_bundle,
)
from src.serving.bundle import is_serving_bundle, load_serving_bundle
from src.serving.predict import predict_canonical


class _ConstPair:
    def __init__(self, p_long: float, p_short: float) -> None:
        self.p_long = float(p_long)
        self.p_short = float(p_short)

    def predict_pair(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        return np.full(n, self.p_long), np.full(n, self.p_short)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pl, ps = self.predict_pair(X)
        return np.column_stack([pl, ps])


def _base_dir(root: Path, cid: str, family: str, cols: list[str], *, seq_len=None) -> Path:
    d = root / cid
    extra = {"kind": "sequence" if seq_len else "tabular", "native_tf": cid.split("_")[-1]}
    if seq_len:
        extra["seq_len"] = seq_len
        extra["spec_params"] = {"seq_len": seq_len}
    write_serve_bundle(
        d, artifact=_ConstPair(0.2, 0.1), feature_cols=cols,
        objective_id="nq-50pt-econ-lg-st", candidate_id=cid, family=family,
        run_ts="20260829T182514Z", extra_manifest=extra,
    )
    return d


def _objective():
    return load_objective("nq-50pt-econ-lg-st")


def test_write_serving_bundle_is_listed(tmp_path):
    obj = _objective()
    b15 = _base_dir(tmp_path, "lstm_15s", "attention_lstm", ["of_ofi", "cs_close"], seq_len=4)
    b1 = _base_dir(tmp_path, "xgb_1m", "xgboost_ovr", ["so_fast_d_9_10", "h1_tr_gap"])
    b5 = _base_dir(tmp_path, "xgb_5m", "xgboost_ovr", ["cs_range", "svp_poc_dist"])
    dest = tmp_path / "meta"
    names = [
        "lstm_15s__p_long", "lstm_15s__p_short", "xgb_1m__p_long", "xgb_1m__p_short",
        "xgb_5m__p_long", "xgb_5m__p_short", "tod_min",
    ]
    bases = [
        {"id": "lstm_15s", "family": "attention_lstm", "native_tf": "15s"},
        {"id": "xgb_1m", "family": "xgboost_ovr", "native_tf": "1m"},
        {"id": "xgb_5m", "family": "xgboost_ovr", "native_tf": "5m"},
    ]
    man = write_tf_oof_meta_serving_bundle(
        dest, objective=obj, candidate_id="tf_oof_meta__lstm_15s_xgb_1m_xgb_5m",
        run_ts="20260829T182514Z", search_run_dir=tmp_path,
        bases=bases,
        base_bundle_dirs={"lstm_15s": b15, "xgb_1m": b1, "xgb_5m": b5},
        fitted={"estimator": _ConstPair(0.41, 0.10), "feature_names": names, "n_train": 9},
    )
    assert is_serving_bundle(dest)
    assert (dest / "meta.pkl").exists()
    assert (dest / "meta.json").exists()
    assert (dest / "feature_schema.json").exists()
    assert (dest / "bases" / "lstm_15s" / "model.pkl").exists()
    assert man["serving"] is True
    assert man["inference_policy"]["tau"] == pytest.approx(0.40)
    bundle = load_serving_bundle(dest, validate=False)
    assert bundle.get("tf_oof_meta") is True
    assert bundle["family"] == "tf_oof_meta"
    assert bundle["native_tf"] == "15s"
    assert {b["id"] for b in bundle["bases"]} == {"lstm_15s", "xgb_1m", "xgb_5m"}


def test_predict_asof_fire_040(tmp_path, monkeypatch):
    obj = _objective()
    cols15 = ["of_ofi"]
    cols1 = ["so_fast_d_9_10"]
    cols5 = ["cs_range"]
    b15 = _base_dir(tmp_path, "lstm_15s", "attention_lstm", cols15)
    b1 = _base_dir(tmp_path, "xgb_1m", "xgboost_ovr", cols1)
    b5 = _base_dir(tmp_path, "xgb_5m", "xgboost_ovr", cols5)
    dest = tmp_path / "meta"
    names = [
        "lstm_15s__p_long", "lstm_15s__p_short",
        "xgb_1m__p_long", "xgb_1m__p_short",
        "xgb_5m__p_long", "xgb_5m__p_short",
        "tod_min", "session_min", "dow",
    ]
    bases = [
        {"id": "lstm_15s", "family": "xgboost_ovr", "native_tf": "15s"},
        {"id": "xgb_1m", "family": "xgboost_ovr", "native_tf": "1m"},
        {"id": "xgb_5m", "family": "xgboost_ovr", "native_tf": "5m"},
    ]
    write_tf_oof_meta_serving_bundle(
        dest, objective=obj, candidate_id="tf_oof_meta__lstm_15s_xgb_1m_xgb_5m",
        run_ts="20260829T182514Z", search_run_dir=tmp_path,
        bases=bases,
        base_bundle_dirs={"lstm_15s": b15, "xgb_1m": b1, "xgb_5m": b5},
        fitted={"estimator": _ConstPair(0.41, 0.10), "feature_names": names, "n_train": 4},
    )
    bundle = load_serving_bundle(dest, validate=False)
    ts = pd.date_range("2026-01-02 14:30:00", periods=8, freq="15s", tz="UTC")
    df15 = pd.DataFrame({"of_ofi": np.arange(8, dtype=float)}, index=ts)

    def _load(store, start, end, feature_cols=None):
        if store == "1m":
            return pd.DataFrame({"so_fast_d_9_10": [1.0] * len(ts)}, index=ts)
        if store == "5m":
            return pd.DataFrame({"cs_range": [1.0] * len(ts)}, index=ts)
        return df15

    monkeypatch.setattr(
        "src.search.tf_oof_meta.load_store_features_for_serve", _load)
    pred = predict_canonical(df15, bundle)
    assert list(pred["direction"].unique()) == ["up"]
    assert pred["confidence"].iloc[0] == pytest.approx(0.41)

    write_tf_oof_meta_serving_bundle(
        dest, objective=obj, candidate_id="tf_oof_meta__lstm_15s_xgb_1m_xgb_5m",
        run_ts="20260829T182514Z", search_run_dir=tmp_path,
        bases=bases,
        base_bundle_dirs={"lstm_15s": b15, "xgb_1m": b1, "xgb_5m": b5},
        fitted={"estimator": _ConstPair(0.39, 0.10), "feature_names": names, "n_train": 4},
    )
    cold = load_serving_bundle(dest, validate=False)
    pred2 = predict_canonical(df15, cold)
    assert list(pred2["direction"].unique()) == ["flat"]


def test_select_meta_features_order():
    df = pd.DataFrame({"b": [1.0, 2.0], "a": [3.0, 4.0], "ts": [0, 1]})
    X = select_meta_features(df, ["a", "missing", "b"])
    assert X.shape == (2, 3)
    assert X[0, 0] == 3.0
    assert X[0, 1] == 0.0
    assert X[0, 2] == 1.0


def test_pack_lab_from_siblings(tmp_path):
    obj = _objective()
    run = tmp_path / "nq-50pt-econ-lg-st" / "20260829T182514Z"
    models = run / "models"
    b15 = _base_dir(models, "lstm_15s", "attention_lstm", ["of_ofi"])
    b1 = _base_dir(models, "xgb_1m", "xgboost_ovr", ["so_fast_d_9_10"])
    b5 = _base_dir(models, "xgb_5m", "xgboost_ovr", ["cs_range"])
    dest = models / "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m"
    dest.mkdir(parents=True)
    names = ["lstm_15s__p_long", "lstm_15s__p_short"]
    import joblib
    joblib.dump(
        {"estimator": _ConstPair(0.5, 0.1), "feature_names": names, "n_train": 2},
        dest / "meta_estimator.pkl",
    )
    (dest / "compose.json").write_text(json.dumps({
        "topology": "tf_oof_meta",
        "serving": False,
        "feature_names": names,
        "bases": [
            {"id": "lstm_15s", "family": "attention_lstm", "native_tf": "15s"},
            {"id": "xgb_1m", "family": "xgboost_ovr", "native_tf": "1m"},
            {"id": "xgb_5m", "family": "xgboost_ovr", "native_tf": "5m"},
        ],
    }))
    (run / "run.json").write_text(json.dumps({
        "objective_id": "nq-50pt-econ-lg-st", "run_ts": "20260829T182514Z",
    }))
    packed = pack_lab_tf_oof_meta(run, objective=obj)
    assert packed == dest
    assert is_serving_bundle(dest)
    meta = json.loads((dest / "meta.json").read_text())
    assert meta["serving"] is True
    assert meta["fire"] == pytest.approx(0.40)


def test_list_custom_test_includes_meta(tmp_path, monkeypatch):
    from src.webui import datasources as ds

    obj = _objective()
    run = tmp_path / "nq-50pt-econ-lg-st" / "20260829T182514Z"
    models = run / "models"
    b15 = _base_dir(models, "lstm_15s", "attention_lstm", ["of_ofi"])
    b1 = _base_dir(models, "xgb_1m", "xgboost_ovr", ["so_fast_d_9_10"])
    b5 = _base_dir(models, "xgb_5m", "xgboost_ovr", ["cs_range"])
    dest = models / "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m"
    write_tf_oof_meta_serving_bundle(
        dest, objective=obj, candidate_id="tf_oof_meta__lstm_15s_xgb_1m_xgb_5m",
        run_ts="20260829T182514Z", search_run_dir=run,
        bases=[
            {"id": "lstm_15s", "family": "attention_lstm", "native_tf": "15s"},
            {"id": "xgb_1m", "family": "xgboost_ovr", "native_tf": "1m"},
            {"id": "xgb_5m", "family": "xgboost_ovr", "native_tf": "5m"},
        ],
        base_bundle_dirs={"lstm_15s": b15, "xgb_1m": b1, "xgb_5m": b5},
        fitted={"estimator": _ConstPair(0.5, 0.1),
                "feature_names": ["lstm_15s__p_long"], "n_train": 1},
    )
    monkeypatch.setattr(ds, "custom_test_dir", lambda: tmp_path)
    found = ds._list_custom_test_bundles(limit=40)
    paths = {b["path"] for b in found}
    assert str(dest.resolve()) in paths
    rec = next(b for b in found if b["path"] == str(dest.resolve()))
    assert rec["candidate_id"] == "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m"
    assert rec.get("inference_policy", {}).get("tau") == pytest.approx(0.40)
