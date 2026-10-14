"""Feature store registry (ADR-076)."""

from __future__ import annotations

import pytest

from src.features.stores import (
    STORE_IDS, feature_store_id, parquet_dir_for_store, require_store, views_for_store,
)
from src.objectives.loader import load_objective
from src.training.aux_labels import aux_from_objective
from src.search.custom_test import compile_board, default_board, feature_bank, parse_feature_catalog
from src.search.tf_oof_meta import assert_tf_oof_meta_ready


def test_store_ids_and_15s_views():
    assert "5s" in STORE_IDS and "15s" in STORE_IDS
    assert require_store("15s") == "15s"
    with pytest.raises(ValueError):
        require_store("10s")
    v = views_for_store("15s")
    assert "native_15s" in v
    assert "of_" in v["native_15s"]


def test_parquet_dir_for_store_is_timeframe_tree(tmp_path, monkeypatch):
    from src.common.data_paths import native_tf_dir

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    assert parquet_dir_for_store("5s") == native_tf_dir("5s")
    assert parquet_dir_for_store("15s") == native_tf_dir("15s")
    assert parquet_dir_for_store("5s").parent.name == "timeframe=5s"


def test_lg_st_defaults_to_15s_dual_binary():
    obj = load_objective("nq-50pt-econ-lg-st")
    req = aux_from_objective(obj)
    assert req is not None
    assert "MFE_long_10m" in req and "MFE_short_10m" in req
    assert "forward_return_10m" in req
    assert "Time_to_TGT_long" in req and "Time_to_TGT_short" in req
    assert "MAE_before_target_or_MFE" not in req  # parquet sidecar only (ADR-086)
    assert obj["labels"]["fire"] == 0.40
    assert aux_from_objective(load_objective("nq-50pt-econ")) is None
    assert feature_store_id(obj) == "15s"
    board = default_board(obj)
    assert board["feature_store"] == "15s"
    assert board["head"] == "dual_binary"
    assert board["meta"]["enabled"] is False
    fams = {m["family"] for m in board["models"]}
    assert "attention_lstm" in fams and "xgboost_ovr" in fams


def test_feature_bank_15s_catalog():
    bank = feature_bank("15s")
    assert bank["store"] == "15s"
    names = {f["name"] for f in bank["features"]}
    assert "of_ofi" in names and "ss_buy_run_max" in names
    assert "bp_spread_mean" not in names
    prefixes = {g["prefix"] for g in bank["groups"]}
    assert prefixes == {"of_", "cs_", "tl_", "ss_", "sd_", "lg_", "st_"}
    assert "lg_target_v_atr" in names and "st_target_v_close" in names
    views = {v["name"] for v in bank["views"] if v["source"] == "catalog"}
    assert "native_15s" in views
    assert "regime_core" not in views


def test_compile_15s_board_puts_store_on_topology():
    compiled = compile_board({
        "objective_id": "nq-50pt-econ-lg-st",
        "feature_store": "15s",
        "models": [{
            "family": "attention_lstm",
            "head": "dual_binary",
            "items": [{"kind": "feature", "key": "of_ofi", "ablation": False}],
        }],
        "meta": {"enabled": False},
    })
    assert compiled.topology["mode"] == "none"
    assert compiled.topology["feature_store"] == "15s"
    assert compiled.candidates[0]["params"]["head"] == "dual_binary"


def test_custom_test_lifts_fail_closed_when_15s_available():
    obj = load_objective("nq-50pt-econ-lg-st")
    obj = dict(obj)
    obj["search"] = dict(obj["search"])
    obj["search"]["topology"] = {"mode": "none", "feature_store": "15s"}
    # Native_tf + mode none (single-store Custom Test overlay) is not joint search:
    with pytest.raises(NotImplementedError, match="nq-50pt-econ-lg-st"):
        assert_tf_oof_meta_ready(obj)
    # Custom Test allowed if 15s files exist:
    from src.features.stores import store_available
    if store_available("15s"):
        assert_tf_oof_meta_ready(obj, experiment_type="custom_test")


def test_parse_15s_catalog():
    parsed = parse_feature_catalog(store="15s")
    names = {f["name"] for f in parsed["features"]}
    assert "cs_close" in names and "tl_aggressive_pct" in names
