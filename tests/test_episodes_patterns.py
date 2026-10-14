"""Tests for outcome episodes + pattern mining — Phase 23 (ADR-021).

Covers episode outcome classification, market-context joins (both feature-frame and
fold-array paths), the deterministic win/loss/missed separation stats, bucket lift,
suggestion generation, persistence round-trips, and the program→objective id fallback the
Research Director relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.episodes import (
    build_episodes,
    latest_outcome_reports,
    load_episodes,
    record_episodes,
)
from src.evaluation.patterns import (
    analyze_and_write,
    mine_patterns,
    separation_table,
)

_OBJ = {"objective_id": "test-obj", "session": {"timezone": "America/Chicago"},
        "kpi": {"requirement": {"any": [
            {"metric": "precision_at_tau", "direction": "up", "tau": 0.8, "min": 0.5}]}}}


def _oof(n=400, seed=0):
    """Synthetic canonical OOF where 'ctx_bad' is elevated on losing signals and
    'ctx_blind' is elevated on missed true moves — so the miner has real structure."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-06-02 09:00", periods=n, freq="5s", tz="America/Chicago").tz_convert("UTC")
    true = rng.choice(["up", "down", "flat"], size=n, p=[0.15, 0.15, 0.70])
    conf = rng.uniform(0.4, 1.0, size=n)
    direction = np.array(true, dtype=object)
    # Introduce wrong-direction + false-positive noise on some fired rows
    flip = rng.random(n) < 0.25
    direction[flip] = rng.choice(["up", "down"], size=int(flip.sum()))
    # Some true moves get low confidence (=> missed at tau 0.8)
    missed_mask = (np.isin(true, ["up", "down"])) & (rng.random(n) < 0.4)
    conf[missed_mask] = 0.55
    df = pd.DataFrame({"ts": ts, "direction": direction, "confidence": conf,
                       "direction_true": true})
    return df


def _features_for(oof, seed=1):
    rng = np.random.default_rng(seed)
    n = len(oof)
    fired = (oof["confidence"] >= 0.8) & oof["direction"].isin(["up", "down"])
    win = fired & (oof["direction"] == oof["direction_true"])
    ctx_bad = rng.normal(0, 1, n) + np.where(fired & ~win, 2.0, 0.0)   # elevated on losses
    ctx_neutral = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"rv_ctx_bad": ctx_bad, "rv_ctx_neutral": ctx_neutral},
        index=oof["ts"].values)


# ── episode building ──────────────────────────────────────────────────────────

def test_build_episodes_outcomes():
    oof = _oof()
    feats = _features_for(oof)
    ep = build_episodes(oof, features=feats, objective=_OBJ, tau=0.8)
    assert ep is not None and not ep.empty
    assert set(ep["outcome"].unique()) <= {"win", "wrong_direction",
                                            "false_positive_flat", "missed"}
    # context + time columns attached
    assert "rv_ctx_bad" in ep.columns
    assert "hour" in ep.columns and "day_of_week" in ep.columns


def test_build_episodes_from_fold_arrays():
    oof = _oof()
    ts_arr = oof["ts"].to_numpy()
    feature_cols = ["rv_a", "rv_b"]
    X = np.random.default_rng(2).normal(size=(len(oof), 2)).astype("float32")
    ep = build_episodes(oof, X=X, ts=ts_arr, feature_cols=feature_cols,
                        objective=_OBJ, tau=0.8, context_cols=feature_cols)
    assert ep is not None
    assert "rv_a" in ep.columns and "rv_b" in ep.columns


def test_build_episodes_empty_input():
    assert build_episodes(pd.DataFrame(), objective=_OBJ) is None


def test_missed_downsampling_cap():
    oof = _oof(n=2000)
    ep = build_episodes(oof, objective=_OBJ, tau=0.8, max_missed=10)
    assert (ep["outcome"] == "missed").sum() <= 10
    assert ep.attrs.get("n_missed_total", 0) >= (ep["outcome"] == "missed").sum()


# ── pattern mining ─────────────────────────────────────────────────────────────

def test_mine_patterns_finds_loss_separation():
    oof = _oof()
    feats = _features_for(oof)
    ep = build_episodes(oof, features=feats, objective=_OBJ, tau=0.8)
    report = mine_patterns(ep)
    assert report["n_episodes"] > 0
    feats_flagged = {p["feature"] for p in report["patterns"]}
    # rv_ctx_bad was constructed to separate losses => should surface
    assert "rv_ctx_bad" in feats_flagged
    assert isinstance(report["summary_text"], str) and report["summary_text"]


def test_mine_patterns_suggestions_present():
    oof = _oof()
    feats = _features_for(oof)
    ep = build_episodes(oof, features=feats, objective=_OBJ, tau=0.8)
    report = mine_patterns(ep)
    actions = {s["action"] for s in report["suggestions"]}
    assert actions & {"add_filter_or_regime_gate", "add_feature_or_data",
                      "avoid_time_window", "rebalance_direction"}


def test_separation_table_full():
    oof = _oof()
    feats = _features_for(oof)
    ep = build_episodes(oof, features=feats, objective=_OBJ, tau=0.8)
    sep = separation_table(ep)
    assert "win_vs_loss" in sep and "caught_vs_missed" in sep
    assert "rv_ctx_bad" in sep["win_vs_loss"]


def test_mine_patterns_handles_empty():
    report = mine_patterns(pd.DataFrame())
    assert report["n_episodes"] == 0
    assert report["patterns"] == []


# ── persistence ─────────────────────────────────────────────────────────────

def test_record_and_load_episodes(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    oof = _oof()
    ep = build_episodes(oof, objective=_OBJ, tau=0.8)
    record_episodes("test-obj", "cand_x", "20260704T000000Z", ep, kind="model")
    loaded = load_episodes("test-obj", source_id="cand_x")
    assert loaded is not None and len(loaded) == len(ep)


def test_analyze_and_write_report(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    oof = _oof()
    feats = _features_for(oof)
    ep = build_episodes(oof, features=feats, objective=_OBJ, tau=0.8)
    analyze_and_write("test-obj", "cand_x", "20260704T000000Z", ep,
                      meta={"kind": "model"})
    reports = latest_outcome_reports("test-obj", limit=3)
    assert reports and reports[0]["source_id"] == "cand_x"
    assert "summary_text" in reports[0]


def test_latest_outcome_reports_program_id_fallback(tmp_path, monkeypatch):
    """RD keys by program_id (nq-50pt-08-15-cst); episodes key by objective_id (nq-50pt)."""
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    oof = _oof()
    ep = build_episodes(oof, objective=_OBJ, tau=0.8)
    analyze_and_write("nq-50pt", "cand_x", "20260704T000000Z", ep, meta={"kind": "model"})
    # program id has the objective id as a prefix
    reports = latest_outcome_reports("nq-50pt-08-15-cst", limit=3)
    assert reports and reports[0]["objective_id"] == "nq-50pt"
