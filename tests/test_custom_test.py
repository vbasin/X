"""Custom Test compiler — board → manual spec + topology overlay."""

from __future__ import annotations

import pytest

from src.models.feature_views import FEATURE_VIEWS
from src.objectives.loader import load_objective
from src.search.custom_test import (
    DEFAULT_META,
    compile_board,
    compile_feature_selection,
    default_board,
    delete_custom_view,
    expand_items_to_features,
    expand_slot,
    feature_bank,
    list_views,
    parse_feature_catalog,
    persist_compiled,
    preview_board,
    save_custom_view,
)
from src.search.controller import (
    _apply_eval_override, _apply_topology_override, custom_test_run_doc,
)
from src.validation.walk_forward import DEFAULT_STOP, DEFAULT_START
from src.search.manual_spec import load_manual_spec


@pytest.fixture(autouse=True)
def _isolate_saved_views(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.search.custom_test.saved_views_dir",
        lambda: tmp_path / "custom_feature_views",
    )


def test_feature_catalog_parse():
    parsed = parse_feature_catalog()
    prefixes = {g["prefix"] for g in parsed["groups"]}
    assert "bp_" in prefixes and "vw_" in prefixes and "tc_" in prefixes
    assert "vf_" in prefixes
    assert "ps_" in prefixes
    assert len(parsed["groups"]) >= 20
    names = {f["name"] for f in parsed["features"]}
    assert "bp_spread_mean" in names
    assert "vw_vwap_dist" in names or any(n.startswith("vw_") for n in names)
    assert "vf_1m_poc_loc" in names and "vf_5m_upper_vol_frac" in names
    assert "vf_1m_poc_ema_5" in names
    assert "ps_1m_next_dist_atr" in names and "ps_lh_dist_sar" in names
    assert "ps_flip" in names and "ps_1m_trend" in names
    assert "ps_1m_sar" not in names
    assert len(parsed["features"]) >= 160
    assert "_close" not in names


def test_feature_bank_has_views_and_models():
    bank = feature_bank()
    view_names = {v["name"] for v in bank["views"]}
    assert set(FEATURE_VIEWS) <= view_names
    core = next(v for v in bank["views"] if v["name"] == "regime_core")
    assert core["source"] == "catalog"
    assert core["n_features"] >= 10
    assert all(any(c.startswith(p) for p in core["prefixes"]) for c in core["features"])
    fams = {m["name"] for m in bank["models"]}
    assert {"xgboost", "tcn", "logistic_regression"} <= fams
    assert bank["feature_builder"]["available"] is False
    xgb = next(m for m in bank["models"] if m["name"] == "xgboost")
    assert xgb["params_default"]["n_estimators"] == 150
    assert xgb["params_default"]["colsample_bytree"] == 0.8
    assert xgb["search_space"]["max_depth"]["type"] == "int"
    assert xgb["hpo_available"] is True
    tcn = next(m for m in bank["models"] if m["name"] == "tcn")
    assert tcn["params_default"]["filters"] == 64
    assert tcn["hpo_available"] is False
    lstm = next(m for m in bank["models"] if m["name"] == "attention_lstm")
    assert lstm["params_default"]["class_balance"] == "sqrt"
    assert lstm["params_default"]["epochs"] == 15
    assert "n_features" not in lstm["params_default"]
    assert "seq_len" in lstm["params_help"]
    assert lstm["search_space"]["class_balance"]["choices"] == [None, "balanced", "sqrt"]
    assert lstm["hpo_available"] is True
    assert lstm["search_max_rows"] == 100_000


def test_default_board_xgb_tcn():
    obj = load_objective("nq-50pt-xgb-tcn")
    board = default_board(obj)
    assert [m["family"] for m in board["models"]] == ["xgboost", "tcn"]
    by_id = {m["id"]: m for m in board["models"]}
    assert by_id["xgboost"]["items"][0]["key"] == "regime_core"
    assert by_id["tcn"]["items"][0]["key"] == "sequence_baseline"
    meta = board["meta"]
    assert meta["enabled"] is True
    assert meta["mode"] == "meta_stack"
    assert meta["family"] == "logistic_regression"
    assert meta["scheme"] == "walk_forward_OOF"
    assert meta["feature"] == "linear_weights"
    assert meta["approve_threshold"] == 0.55
    assert meta["n_folds"] == 5
    assert meta["embargo_bars"] == 120
    wf = board["walk_forward"]
    assert wf["enabled"] is True
    assert wf["start"] == DEFAULT_START
    assert wf["stop"] == DEFAULT_STOP
    assert wf["train_n"] == 2 and wf["train_unit"] == "months"
    assert wf["test_n"] == 2 and wf["test_unit"] == "months"
    assert wf["val"]["enabled"] is False


def test_default_board_nq50pt_meta_off():
    obj = load_objective("nq-50pt")
    board = default_board(obj)
    assert board["meta"]["enabled"] is False
    assert board["meta"]["mode"] == DEFAULT_META["mode"]
    assert board["feature_store"] == "5s"
    assert board["head"] == "multiclass"


def test_ablation_is_multiplicative():
    slot = {
        "id": "xgboost",
        "family": "xgboost",
        "items": [
            {"kind": "view", "key": "regime_core", "ablation": False},
            {"kind": "group", "key": "iv_", "ablation": True},
            {"kind": "feature", "key": "bp_spread_mean", "ablation": True},
        ],
    }
    cands = expand_slot(slot)
    assert len(cands) == 4
    assert cands[0]["is_full"] is True and cands[0]["id"] == "xgboost"
    tags = {c["ablation_tag"] for c in cands}
    assert "full" in tags
    omitted = [c for c in cands if not c["is_full"]]
    assert len(omitted) == 3


def test_compile_board_to_manual_spec():
    board = {
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [
            {"id": "xgboost", "family": "xgboost",
             "items": [{"kind": "view", "key": "regime_core", "ablation": False}]},
            {"id": "tcn", "family": "tcn",
             "items": [{"kind": "view", "key": "sequence_baseline", "ablation": False}]},
        ],
        "meta": {
            "enabled": True, "mode": "meta_stack",
            "family": "logistic_regression", "scheme": "walk_forward_OOF",
            "feature": "linear_weights", "approve_threshold": 0.55,
            "n_folds": 5, "embargo_bars": 120,
        },
    }
    compiled = compile_board(board)
    assert compiled.n_combos == 2
    spec = load_manual_spec(compiled.manual_spec, objective_id="nq-50pt-xgb-tcn")
    assert [e.name for e in spec.entries] == ["xgboost", "tcn"]
    assert spec.entries[0].feature_selection == {"view": "regime_core"}
    assert spec.entries[0].search_max_rows is None          # trees: full matrix
    assert spec.entries[1].search_max_rows is None          # GPU sequence: cap off unless set
    assert compiled.topology["mode"] == "meta_stack"
    assert compiled.topology["bases"] == ["xgboost", "tcn"]


def test_compile_disables_objective_meta_when_off():
    board = {
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "regime_core"}]}],
        "meta": {"enabled": False},
    }
    compiled = compile_board(board)
    assert compiled.topology["mode"] == "none"
    assert compiled.topology.get("feature_store") == "5s"


def test_preview_ablation_count():
    board = {
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{
            "family": "xgboost",
            "items": [
                {"kind": "view", "key": "regime_core", "ablation": False},
                {"kind": "group", "key": "iv_", "ablation": True},
            ],
        }],
        "meta": {"enabled": False},
    }
    prev = preview_board(board)
    assert prev["n_candidates"] == 2
    ids = {c["id"] for c in prev["candidates"]}
    assert "xgboost" in ids
    assert any("no_" in c["id"] for c in prev["candidates"])


def test_compile_feature_selection_shapes():
    assert compile_feature_selection(
        [{"kind": "view", "key": "regime_core"}]) == {"view": "regime_core"}
    sel = compile_feature_selection([
        {"kind": "view", "key": "regime_core"},
        {"kind": "group", "key": "iv_"},
    ])
    assert sel["prefixes"][0] in FEATURE_VIEWS["regime_core"]
    assert "iv_" in sel["prefixes"]
    mixed = compile_feature_selection([
        {"kind": "group", "key": "vw_"},
        {"kind": "feature", "key": "bp_spread_mean"},
    ])
    assert mixed["columns"][-1] == "bp_spread_mean" or "bp_spread_mean" in mixed["columns"]


def test_expand_view_and_group_to_feature_chips():
    bank = feature_bank()
    expanded = expand_items_to_features(
        [{"kind": "view", "key": "regime_core"},
         {"kind": "group", "key": "iv_"}],
        bank=bank,
    )
    assert expanded
    assert all(it["kind"] == "feature" for it in expanded)
    names = [it["key"] for it in expanded]
    assert names == list(dict.fromkeys(names))
    assert any(n.startswith("rv_") for n in names)
    assert any(n.startswith("iv_") for n in names)
    again = expand_items_to_features(expanded, bank=bank)
    assert [it["key"] for it in again] == names


def test_save_custom_view_roundtrip_and_compile():
    rec = save_custom_view("my_lstm_subset", ["vw_vwap_dist", "bp_spread_mean"])
    assert rec["name"] == "my_lstm_subset"
    assert rec["features"] == ["vw_vwap_dist", "bp_spread_mean"]
    views = list_views()
    saved = next(v for v in views if v["name"] == "my_lstm_subset")
    assert saved["source"] == "saved"
    assert saved["features"] == ["vw_vwap_dist", "bp_spread_mean"]
    sel = compile_feature_selection(
        [{"kind": "view", "key": "my_lstm_subset"}],
        views_index={v["name"]: v for v in views},
    )
    assert sel == {"columns": ["vw_vwap_dist", "bp_spread_mean"]}
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "my_lstm_subset"}]}],
        "meta": {"enabled": False},
    })
    assert compiled.candidates[0]["features"] == {
        "columns": ["vw_vwap_dist", "bp_spread_mean"]}
    with pytest.raises(FileExistsError):
        save_custom_view("my_lstm_subset", ["vw_vwap_dist"])
    rec2 = save_custom_view(
        "my_lstm_subset", ["vw_vwap_dist"], overwrite=True)
    assert rec2["features"] == ["vw_vwap_dist"]
    assert rec2["created_at"] == rec["created_at"]
    delete_custom_view("my_lstm_subset")
    assert all(v["name"] != "my_lstm_subset" for v in list_views())


def test_save_custom_view_rejects_catalog_name_and_unknown_feature():
    with pytest.raises(ValueError, match="catalog view"):
        save_custom_view("regime_core", ["vw_vwap_dist"])
    with pytest.raises(ValueError, match="unknown feature"):
        save_custom_view("not_real", ["not_a_column"])
    with pytest.raises(ValueError, match="at least one"):
        save_custom_view("empty_view", [])


def test_unknown_family_and_empty_rejected():
    with pytest.raises(Exception):
        compile_board({"objective_id": "nq-50pt", "models": []})
    with pytest.raises(ValueError, match="unknown model"):
        compile_board({"objective_id": "nq-50pt",
                       "models": [{"family": "not_a_model"}]})


def test_topology_override_replaces_search_block():
    obj = load_objective("nq-50pt-xgb-tcn")
    assert (obj.get("search") or {}).get("topology", {}).get("mode") == "meta_stack"
    off = _apply_topology_override(obj, {"mode": "none"})
    assert off["search"]["topology"] == {"mode": "none"}
    # original not mutated
    assert (obj.get("search") or {}).get("topology", {}).get("mode") == "meta_stack"
    on = _apply_topology_override(obj, {
        "mode": "meta_stack", "bases": ["xgboost"],
        "meta": {"family": "logistic_regression"},
    })
    assert on["search"]["topology"]["bases"] == ["xgboost"]


def test_topology_none_keeps_feature_store_for_fold_builder():
    """GPU job_exec reloads YAML; without this overlay lg-st trains 15s (ADR-076)."""
    from src.features.stores import feature_store_id

    obj = load_objective("nq-50pt-econ-lg-st")
    assert feature_store_id(obj) == "15s"
    over = _apply_topology_override(obj, {"mode": "none", "feature_store": "1m"})
    assert over["search"]["topology"] == {"mode": "none", "feature_store": "1m"}
    assert feature_store_id(over) == "1m"
    assert feature_store_id(obj) == "15s"


def test_compile_params_override_and_hpo_subset():
    board = {
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{
            "family": "xgboost",
            "params": {"max_depth": 6, "colsample_bytree": 1.0, "n_estimators": "200"},
            "hpo": True,
            "hpo_params": ["max_depth", "learning_rate"],
            "items": [{"kind": "view", "key": "regime_core"}],
        }],
        "meta": {"enabled": False},
        "tune_trials": 12,
    }
    compiled = compile_board(board)
    assert compiled.tune_trials == 12
    spec = load_manual_spec(compiled.manual_spec, objective_id="nq-50pt-xgb-tcn")
    e = spec.entries[0]
    assert e.params_default["max_depth"] == 6
    assert e.params_default["colsample_bytree"] == 1.0
    assert e.params_default["n_estimators"] == 200
    assert e.params_default["subsample"] == 0.8  # catalog default kept
    assert set(e.search_space) == {"max_depth", "learning_rate"}
    assert e.search_space["max_depth"]["high"] == 12
    cand = compiled.candidates[0]
    assert cand["params"]["max_depth"] == 6
    assert set(cand["search_space"]) == {"max_depth", "learning_rate"}
    assert any("HPO:" in w and "12" in w for w in compiled.warnings)


def test_compile_hpo_defaults_all_catalog_knobs_and_trials():
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost", "hpo": True,
                    "items": [{"kind": "view", "key": "regime_core"}]}],
        "meta": {"enabled": False},
    })
    assert compiled.tune_trials == 20
    spec = load_manual_spec(compiled.manual_spec)
    from src.models.catalog import get_entry
    assert set(spec.entries[0].search_space) == set(get_entry("xgboost").search_space)


def test_compile_hpo_off_clears_search_space_even_if_trials_set():
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "regime_core"}]}],
        "meta": {"enabled": False},
        "tune_trials": 50,
    })
    assert compiled.tune_trials == 0
    spec = load_manual_spec(compiled.manual_spec)
    assert spec.entries[0].search_space == {}


def test_compile_hpo_unknown_param_rejected():
    with pytest.raises(ValueError, match="hpo_params"):
        compile_board({
            "objective_id": "nq-50pt-xgb-tcn",
            "models": [{
                "family": "xgboost", "hpo": True,
                "hpo_params": ["not_a_real_knob"],
                "items": [{"kind": "view", "key": "regime_core"}],
            }],
            "meta": {"enabled": False},
        })


def test_compile_hpo_without_catalog_space_warns():
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "tcn", "hpo": True,
                    "items": [{"kind": "view", "key": "sequence_baseline"}]}],
        "meta": {"enabled": False},
    })
    assert compiled.tune_trials == 0
    assert any("no catalog search_space" in w for w in compiled.warnings)
    spec = load_manual_spec(compiled.manual_spec)
    assert spec.entries[0].search_space == {}


def test_compile_hpo_with_ablation_warns_per_variant():
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{
            "family": "xgboost", "hpo": True,
            "hpo_params": ["max_depth"],
            "items": [
                {"kind": "view", "key": "regime_core", "ablation": False},
                {"kind": "group", "key": "iv_", "ablation": True},
            ],
        }],
        "meta": {"enabled": False},
        "tune_trials": 8,
    })
    assert compiled.n_combos == 2
    assert all(c.get("search_space") for c in compiled.candidates)
    assert any("ablation variant" in w for w in compiled.warnings)


def test_compile_walk_forward_overlay_and_eval_apply():
    board = {
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "momentum_edge"}]}],
        "meta": {"enabled": False},
        "walk_forward": {
            "enabled": True,
            "start": "2025-05-01", "stop": "2026-06-30",
            "train_n": 2, "train_unit": "months",
            "test_n": 2, "test_unit": "months",
            "val": {"enabled": False, "n": 10, "unit": "days", "cut_from": "train"},
        },
    }
    compiled = compile_board(board)
    ov = compiled.walk_forward
    assert ov["walk_forward_folds"] == 6
    assert ov["date_range"]["end"] == "2026-06-30"
    assert ov["walk_forward"]["mode"] == "expanding_calendar"
    obj = load_objective("nq-50pt-econ")
    patched = _apply_eval_override(obj, ov)
    assert patched["eval"]["walk_forward"]["mode"] == "expanding_calendar"
    assert patched["eval"]["walk_forward_folds"] == 6
    assert (obj.get("eval") or {}).get("walk_forward") is None


def test_compile_walk_forward_disabled_no_overlay():
    compiled = compile_board({
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "momentum_edge"}]}],
        "meta": {"enabled": False},
        "walk_forward": {"enabled": False, "start": "2025-05-01", "stop": "2026-06-30",
                         "train_n": 2, "train_unit": "months",
                         "test_n": 2, "test_unit": "months"},
    })
    assert compiled.walk_forward is None


def test_custom_test_run_doc_is_report_without_gate(tmp_path):
    doc = custom_test_run_doc(
        objective_id="nq-50pt-econ", run_ts="20260827T000000Z",
        out_dir=tmp_path, n_candidates=2, n_ok=2, n_failed=0,
    )
    assert doc["kind"] == "custom_test"
    assert doc["status"] == "report"
    assert "gate_result" not in doc
    assert "champion" not in doc
    assert doc["n_candidates"] == 2


def test_persist_writes_three_files(tmp_path):
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "regime_core"}]}],
        "meta": {"enabled": True, "mode": "meta_stack"},
    })
    paths = persist_compiled(compiled, tmp_path, "20260815T000000Z")
    assert paths["board"].is_file()
    assert paths["manual_spec"].is_file()
    assert paths["topology"].is_file()
    spec = load_manual_spec(paths["manual_spec"], objective_id="nq-50pt-xgb-tcn")
    assert spec.entries[0].family_key == "xgboost"


def test_persist_writes_walk_forward_when_enabled(tmp_path):
    compiled = compile_board({
        "objective_id": "nq-50pt-xgb-tcn",
        "models": [{"family": "xgboost",
                    "items": [{"kind": "view", "key": "regime_core"}]}],
        "meta": {"enabled": False},
        "walk_forward": {
            "enabled": True, "start": "2025-05-01", "stop": "2026-06-30",
            "train_n": 2, "train_unit": "months",
            "test_n": 2, "test_unit": "months",
        },
    })
    paths = persist_compiled(compiled, tmp_path, "20260821T000000Z")
    assert paths["walk_forward"].is_file()
    text = paths["walk_forward"].read_text()
    assert "expanding_calendar" in text
    assert "2025-05-01" in text


def test_compile_lstm_omitted_max_rows_is_full_train():
    compiled = compile_board({
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "attention_lstm",
                    "items": [{"kind": "group", "key": "of_"}]}],
        "meta": {"enabled": False},
    })
    spec = load_manual_spec(compiled.manual_spec, objective_id="nq-50pt-econ")
    assert spec.entries[0].search_max_rows is None


def test_compile_lstm_explicit_max_rows():
    compiled = compile_board({
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "attention_lstm", "max_rows": 50000,
                    "items": [{"kind": "group", "key": "of_"}]}],
        "meta": {"enabled": False},
    })
    spec = load_manual_spec(compiled.manual_spec, objective_id="nq-50pt-econ")
    assert spec.entries[0].search_max_rows == 50_000
    assert compiled.candidates[0]["max_rows"] == 50_000


def test_compile_lstm_null_max_rows_forces_full_matrix():
    compiled = compile_board({
        "objective_id": "nq-50pt-econ",
        "models": [{"family": "attention_lstm", "max_rows": None,
                    "items": [{"kind": "group", "key": "of_"}]}],
        "meta": {"enabled": False},
    })
    spec = load_manual_spec(compiled.manual_spec, objective_id="nq-50pt-econ")
    assert spec.entries[0].search_max_rows is None
