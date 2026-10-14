"""Proof tests for ADR-027 economics objective (W1 ranking + W3 meta + W4 views).

No full walk-forward training — validates the objective schema, champion ranking
by net_expectancy, meta composition, and sparse feature-view resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.evaluation.kpi_criteria import meets_requirement, meets_target, primary_tau
from src.models.feature_views import FEATURE_VIEWS, apply_feature_view
from src.objectives.loader import load_objective
from src.search.gate import (
    apply_champion_gate, champion_gate_enabled, resolve_champion_gate, _rank_key,
)
from src.search.meta_compose import (
    compose_primary_meta, fit_final_primary_meta, topology_cfg,
)
from src.search.tf_oof_meta import assert_tf_oof_meta_ready, topology_tf_oof_meta


# ── Objective loads & schema ──────────────────────────────────────────────────

def test_nq_50pt_xgb_unique_loads_and_validates():
    obj = load_objective("nq-50pt-xgb-unique")
    assert obj["objective_id"] == "nq-50pt-xgb-unique"
    assert obj["kpi"]["primary"] == "net_expectancy"
    assert obj["labels"]["scheme"] == "triple_barrier"
    assert obj["costs"]["cost_per_trade"] == 1.0
    assert obj["eval"]["train_uniqueness_bars"] == 24
    assert obj["eval"]["walk_forward"]["mode"] == "expanding_calendar"
    assert obj["eligible_model_families"] == ["xgboost", "xgboost_ovr"]
    assert obj["search"]["default_feature_view"] == "tbm_topgroups"
    assert obj["search"]["feature_views"]["xgboost"] == "tbm_topgroups"
    assert obj["search"]["feature_views"]["xgboost_ovr"] == "tbm_topgroups"
    assert obj["search"]["topology"]["mode"] == "primary_meta"
    assert primary_tau(obj) == 0.65
    assert "tbm_topgroups" in FEATURE_VIEWS


def test_nq_50pt_econ_loads_and_validates():
    obj = load_objective("nq-50pt-econ")
    assert obj["objective_id"] == "nq-50pt-econ"
    assert obj["kpi"]["primary"] == "net_expectancy"
    assert obj["labels"]["scheme"] == "triple_barrier"
    assert float(obj["costs"]["cost_per_trade"] or 0) == 0.0
    assert obj["search"]["default_feature_view"] == "momentum_edge"
    assert obj["search"]["topology"]["mode"] == "primary_meta"
    assert obj["kpi"]["champion_gate"] is True
    assert primary_tau(obj) is None
    assert "momentum_edge" in FEATURE_VIEWS
    assert topology_cfg(obj)["approve_threshold"] == 0.50


def test_nq_50pt_econ_lg_st_loads_and_validates():
    obj = load_objective("nq-50pt-econ-lg-st")
    assert obj["objective_id"] == "nq-50pt-econ-lg-st"
    assert obj["labels"]["encoding"] == "dual_binary"
    assert obj["kpi"]["primary"] == "net_expectancy"
    assert obj["eval"]["walk_forward_folds"] == 3
    assert obj["eval"]["walk_forward"]["train"] == {"n": 4, "unit": "months"}
    assert obj["eval"]["walk_forward"]["test"] == {"n": 4, "unit": "months"}
    assert obj["search"]["input"] == "native_tf"
    assert obj["search"]["decision_timestep"] == "15s"
    assert obj["session"]["training_start"] == "08:30"
    assert obj["session"]["training_end"] == "11:00"
    assert obj["session"]["trading_end"] == "11:00"
    assert obj["eval"]["train_onset_uniqueness"] is True
    assert obj["search"]["topology"]["session_vp"] is True
    cfg = topology_tf_oof_meta(obj)
    assert cfg is not None
    assert [b["native_tf"] for b in cfg["bases"]] == ["15s", "1m", "5m"]
    assert [b["family"] for b in cfg["bases"]] == ["attention_lstm", "xgboost_ovr", "xgboost_ovr"]
    assert obj["labels"]["fire"] == 0.40
    assert "Time_to_TGT_long" in obj["labels"]["aux"]
    assert "Time_to_TGT_short" in obj["labels"]["aux"]
    assert "forward_return_1m" not in (obj["labels"].get("aux") or [])
    assert "forward_return_10m" in obj["labels"]["aux"]
    assert cfg["bases"][0]["params"]["seq_len"] == 60
    assert cfg["bases"][1]["params"]["max_depth"] == 12
    assert cfg["bases"][2]["params"]["max_depth"] == 8
    assert [c["native_tf"] for c in cfg["context"]] == ["1h", "1d"]
    assert cfg["outputs"] == ["p_final_long", "p_final_short"]
    assert cfg["meta"]["n_folds"] == 3
    assert "approve_threshold" not in (obj["search"]["topology"].get("meta") or {})
    assert "long_threshold" not in (obj["search"]["topology"].get("meta") or {})
    req = obj["kpi"]["requirement"]["all"]
    tgt = obj["kpi"]["target"]["all"]
    assert all(c["metric"] != "precision_at_tau" for c in req + tgt)
    from src.search.tf_oof_meta import missing_tf_oof_stores
    missing = missing_tf_oof_stores(obj)
    if missing:
        with pytest.raises(NotImplementedError, match="nq-50pt-econ-lg-st"):
            assert_tf_oof_meta_ready(obj)
    else:
        assert_tf_oof_meta_ready(obj)


def test_tbm_topgroups_view_keeps_cs_drops_orderflow():
    rng = np.random.default_rng(0)
    cols = [
        "cs_15m_clv", "rv_realized_vol_60", "vp_poc_dist", "iv_atm_mean",
        "of_delta_1", "bp_imb_5", "tc_session_progress",
    ]
    X = rng.normal(size=(20, len(cols))).astype(np.float32)
    folds = [{"X_train": X, "X_test": X[:5], "feature_cols": cols, "y_train": np.zeros(20)}]
    out = apply_feature_view(folds, "tbm_topgroups")
    kept = out[0]["feature_cols"]
    assert kept == ["cs_15m_clv", "rv_realized_vol_60", "vp_poc_dist", "iv_atm_mean"]
    assert "of_delta_1" not in kept
    assert out[0]["X_train"].shape[1] == 4


def test_momentum_edge_view_masks_folds():
    rng = np.random.default_rng(0)
    cols = [
        "of_delta_1", "bp_imb_5", "vw_vwap_dist", "rv_realized_vol_60",
        "ht_gap_pct", "dc_prev_settle_dist", "sd_tick_count", "tc_session_progress",
        "tl_aggressive_pct", "ss_run_len",
        "iv_atm_mean", "wv_energy_1", "ec_minutes_to_event",  # should be dropped
    ]
    X = rng.normal(size=(40, len(cols))).astype(np.float32)
    folds = [{"X_train": X, "X_test": X[:10], "feature_cols": cols, "y_train": np.zeros(40)}]
    out = apply_feature_view(folds, "momentum_edge")
    kept = out[0]["feature_cols"]
    assert "iv_atm_mean" not in kept
    assert "wv_energy_1" not in kept
    assert "of_delta_1" in kept and "vw_vwap_dist" in kept
    assert out[0]["X_train"].shape[1] == len(kept) < len(cols)


# ── KPI economics conditions ──────────────────────────────────────────────────

def test_econ_requirement_reads_net_expectancy_and_n_signals():
    obj = load_objective("nq-50pt-econ")
    # Passer: strong expectancy, enough trades, hit rate, mild DD.
    # n_signals prefers trades_at_gate.taken when diagnostics present.
    score = {
        "net_expectancy": 26.0,
        "n_signals": 80,
        "precision_at_tau": 0.60,
        "max_drawdown": -40.0,
        "hit_rate": 0.60,
        "directional_f1": 0.1,
        "n_up": 40,
        "n_down": 40,
    }
    diag = {"trades_at_gate": {"taken": 80}}
    ok, details = meets_requirement(obj, diagnostics=diag, score=score)
    assert ok, details

    # Fail: high hit rate but expectancy too low (the W1 point).
    score_bad = {**score, "net_expectancy": 2.0}
    ok2, _ = meets_requirement(obj, diagnostics=diag, score=score_bad)
    assert not ok2


def test_overall_precision_without_direction():
    """precision_at_tau condition with no direction uses overall score precision."""
    obj = {
        "kpi": {
            "primary": "net_expectancy",
            "requirement": {"all": [
                {"metric": "precision_at_tau", "tau": 0.8, "min": 0.7, "min_signals": 10},
            ]},
            "target": {"all": [{"metric": "net_expectancy", "min": 99}]},
        }
    }
    ok, _ = meets_requirement(obj, {}, {"precision_at_tau": 0.75, "n_signals": 40})
    assert ok
    ok2, _ = meets_requirement(obj, {}, {"precision_at_tau": 0.60, "n_signals": 40})
    assert not ok2


# ── Champion gate ranks by net_expectancy ─────────────────────────────────────

@dataclass
class _Fake:
    candidate_id: str
    score: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    meets_target: bool = False


def test_gate_ranks_by_net_expectancy_not_precision():
    obj = load_objective("nq-50pt-econ")
    diag = {"trades_at_gate": {"taken": 60}}
    # A: lower precision, higher expectancy → should win under economics ranking
    a = _Fake("econ_better", {
        "net_expectancy": 28.0, "precision_at_tau": 0.60, "directional_f1": 0.05,
        "n_signals": 60, "n_up": 30, "n_down": 30,
        "max_drawdown": -50.0, "hit_rate": 0.62,
    }, diagnostics=diag)
    # B: higher precision, lower (but still-passing) expectancy
    b = _Fake("prec_better", {
        "net_expectancy": 21.0, "precision_at_tau": 0.90, "directional_f1": 0.08,
        "n_signals": 60, "n_up": 30, "n_down": 30,
        "max_drawdown": -30.0, "hit_rate": 0.58,
    }, diagnostics=diag)
    # Sanity: rank key prefers A
    assert _rank_key(a, obj) > _rank_key(b, obj)

    gr = apply_champion_gate([a, b], obj)
    assert gr.status == "promoted"
    assert gr.champion_id == "econ_better"
    assert "econ_better" in gr.passed_ids and "prec_better" in gr.passed_ids


def test_gate_precision_primary_unchanged_for_legacy_objective():
    """nq-50pt still ranks by precision (no regression on legacy objectives)."""
    obj = load_objective("nq-50pt")
    a = _Fake("high_prec", {
        "precision_at_tau": 0.90, "directional_f1": 0.1,
        "precision_up": 0.90, "n_up": 40,
        "precision_down": 0.88, "n_down": 40,
        "n_signals": 80,
    })
    # Build diagnostics so per-direction precision_at_tau@0.8 conditions can pass.
    diag = {
        "precision_at_tau_sweep": [
            {"tau": 0.8, "direction": "up", "n_signals": 40, "precision": 0.90},
            {"tau": 0.8, "direction": "down", "n_signals": 40, "precision": 0.88},
            {"tau": 0.9, "direction": "up", "n_signals": 20, "precision": 0.92},
            {"tau": 0.9, "direction": "down", "n_signals": 20, "precision": 0.91},
        ]
    }
    a.diagnostics = diag
    b = _Fake("low_prec", {
        "precision_at_tau": 0.70, "directional_f1": 0.2,
        "n_signals": 80,
    }, diagnostics={
        "precision_at_tau_sweep": [
            {"tau": 0.8, "direction": "up", "n_signals": 40, "precision": 0.70},
            {"tau": 0.8, "direction": "down", "n_signals": 40, "precision": 0.70},
        ]
    })
    assert _rank_key(a, obj) > _rank_key(b, obj)


# ── Meta composition ──────────────────────────────────────────────────────────

def _synthetic_primary(n: int = 200, hit_rate: float = 0.7, seed: int = 0) -> pd.DataFrame:
    """Primary that is correct ``hit_rate`` of the time on directional fires.

    Confidence correlates with correctness so the meta model has a learnable signal.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-02-02 15:00", periods=n, freq="5s", tz="UTC")
    # 60% directional proposals
    is_dir = rng.random(n) < 0.6
    side = np.where(is_dir, rng.choice(["up", "down"], size=n), "flat")
    correct = rng.random(n) < hit_rate
    true = []
    conf = []
    for i in range(n):
        if side[i] == "flat":
            true.append(rng.choice(["up", "down", "flat"]))
            conf.append(0.3)
            continue
        if correct[i]:
            true.append(side[i])
            conf.append(float(rng.uniform(0.7, 0.95)))
        else:
            true.append("flat" if rng.random() < 0.7 else ("down" if side[i] == "up" else "up"))
            conf.append(float(rng.uniform(0.5, 0.75)))
    return pd.DataFrame({
        "ts": ts, "direction": side, "confidence": conf, "direction_true": true,
    })


def test_compose_primary_meta_raises_precision_via_filter():
    primary = _synthetic_primary(n=300, hit_rate=0.75, seed=1)
    composed = compose_primary_meta(
        primary, context=None,
        approve_threshold=0.55, min_primary_confidence=0.5, n_folds=5,
    )
    assert "meta_prob" in composed.columns
    assert "primary_direction" in composed.columns
    assert "primary_confidence" in composed.columns
    # Approved subset should be a (weakly) higher-precision subset of primary fires.
    p_dir = primary[(primary["direction"].isin(["up", "down"])) & (primary["confidence"] >= 0.5)]
    c_dir = composed[composed["direction"].isin(["up", "down"])]
    assert len(c_dir) <= len(p_dir)
    # Meta is a hard filter: approved rows keep the primary's confidence (so τ-gates work).
    if len(c_dir):
        assert (c_dir["confidence"] > 0).all()
        # Confidence must not be replaced by raw meta_prob alone (often < 0.8).
        assert (c_dir["confidence"] == c_dir["primary_confidence"]).all()
    if len(c_dir) >= 10 and len(p_dir) >= 10:
        p_prec = (p_dir["direction"] == p_dir["direction_true"]).mean()
        c_prec = (c_dir["direction"] == c_dir["direction_true"]).mean()
        # Soft: meta should not destroy precision (allow small noise on tiny samples)
        assert c_prec >= p_prec - 0.05


def test_fit_final_primary_meta_returns_payload():
    primary = _synthetic_primary(n=300, hit_rate=0.75, seed=1)
    fitted = fit_final_primary_meta(primary, min_primary_confidence=0.5)
    assert fitted["kind"] in ("xgb", "sklearn", "constant")
    assert "primary_confidence" in fitted["feature_names"]
    assert fitted["n_train"] > 0


def test_compose_primary_meta_too_few_proposals_all_flat():
    primary = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC"),
        "direction": ["flat"] * 5,
        "confidence": [0.9] * 5,
        "direction_true": ["up"] * 5,
    })
    out = compose_primary_meta(primary)
    assert (out["direction"] == "flat").all()


def test_topology_cfg_inactive_for_legacy():
    assert topology_cfg(load_objective("nq-50pt")) == {}
    assert topology_cfg(load_objective("nq-50pt-tbm")) == {}


def test_primary_tau_from_econ_requirement():
    # nq-50pt-econ has no declared confidence cut — do not infer 0.8.
    assert primary_tau(load_objective("nq-50pt-econ")) is None
    # Unique declares precision_at_tau.tau: 0.65 on requirement.
    assert primary_tau(load_objective("nq-50pt-xgb-unique")) == pytest.approx(0.65)
    # Optional kpi.tau wins over requirement rows.
    assert primary_tau({"kpi": {"tau": 0.55, "requirement": {"all": [
        {"metric": "precision_at_tau", "tau": 0.8, "min": 0.7},
    ]}}}) == pytest.approx(0.55)


def test_champion_gate_true_without_requirement_fails_load():
    from pathlib import Path
    from src.objectives.loader import _validate_champion_gate
    with pytest.raises(ValueError, match="champion_gate"):
        _validate_champion_gate(
            {"kpi": {"champion_gate": True, "requirement": {}}},
            Path("x.yaml"),
        )


def test_champion_gate_is_yaml_opt_in_and_skipped_for_custom_test():
    obj = load_objective("nq-50pt-econ")
    assert champion_gate_enabled(obj, "model_selection") is True
    assert champion_gate_enabled(obj, "custom_test") is False
    assert champion_gate_enabled({"kpi": {"requirement": {"all": []}}}, "model_selection") is False
    assert champion_gate_enabled({"kpi": {"champion_gate": False}}, "model_selection") is False
    a = _Fake("econ_better", {
        "net_expectancy": 28.0, "precision_at_tau": 0.60, "directional_f1": 0.05,
        "n_signals": 60, "n_up": 30, "n_down": 30,
        "max_drawdown": -50.0, "hit_rate": 0.62,
    }, diagnostics={"trades_at_gate": {"taken": 60}})
    assert resolve_champion_gate("custom_test", [a], obj) is None
    assert resolve_champion_gate("model_selection", [a], {"kpi": {}}) is None
    gr = resolve_champion_gate("model_selection", [a], obj)
    assert gr is not None and gr.status == "promoted"
