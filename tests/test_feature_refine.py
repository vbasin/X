"""Tests for the agent-driven feature add/remove refinement loop — Phase 23.

Covers the deterministic proposer (add outcome-separating features, remove dead weight,
respect caps + the minimum-feature floor), the run-ranking key, spec column resolution, and
an end-to-end loop driven by a stubbed ``run_search`` (no real training / GPU).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.search.feature_refine import (
    _rank_key,
    _resolve_columns,
    propose_mutation,
    refine_features,
)

_OBJ = {
    "objective_id": "test-obj",
    "kpi": {
        "requirement": {"any": [
            {"metric": "precision_at_tau", "direction": "up", "tau": 0.8, "min": 0.5,
             "min_signals": 1}]},
        "target": {"all": [
            {"metric": "precision_at_tau", "direction": "up", "tau": 0.8, "min": 0.95,
             "min_signals": 1}]},
    },
}


# ── deterministic proposer ────────────────────────────────────────────────────

def test_propose_adds_separating_features():
    current = ["rv_a", "rv_b"]
    available = ["rv_a", "rv_b", "of_c", "of_d", "ms_e"]
    sep = {"win_vs_loss": {"of_c": 0.9, "of_d": 0.02, "rv_a": 0.01, "rv_b": 0.0},
           "caught_vs_missed": {"ms_e": 0.7}}
    prop = propose_mutation(current, sep, available, max_add=4, max_remove=4)
    assert "of_c" in prop["add"]          # strong win/loss separation
    assert "ms_e" in prop["add"]          # strong caught/missed separation
    assert "of_d" not in prop["add"]      # too weak


def test_propose_removes_dead_weight():
    current = ["rv_a", "rv_b", "rv_c", "rv_d", "rv_e", "rv_f"]
    available = list(current)
    # all weak => removable, but the min-features floor (5) caps removals to 1
    sep = {"win_vs_loss": {c: 0.01 for c in current}, "caught_vs_missed": {}}
    prop = propose_mutation(current, sep, available, max_add=0, max_remove=4)
    assert len(prop["remove"]) == 1       # 6 current - 1 = 5 floor
    assert prop["add"] == []


def test_propose_respects_max_add_cap():
    current = ["rv_a"]
    available = ["rv_a"] + [f"of_{i}" for i in range(10)]
    sep = {"win_vs_loss": {f"of_{i}": 0.9 for i in range(10)}, "caught_vs_missed": {}}
    prop = propose_mutation(current, sep, available, max_add=3, max_remove=0)
    assert len(prop["add"]) == 3


def test_propose_only_uses_available_columns():
    current = ["rv_a"]
    available = ["rv_a", "of_present"]
    sep = {"win_vs_loss": {"of_absent": 0.9, "of_present": 0.8}, "caught_vs_missed": {}}
    prop = propose_mutation(current, sep, available, max_add=4, max_remove=0)
    assert "of_absent" not in prop["add"]
    assert "of_present" in prop["add"]


# ── ranking ───────────────────────────────────────────────────────────────────

def _diag(prec_up, n=50):
    return {"precision_at_tau_sweep": [
        {"tau": 0.8, "direction": "up", "precision": prec_up, "n_signals": n},
        {"tau": 0.8, "direction": "down", "precision": 0.0, "n_signals": 0}]}


def test_rank_key_prefers_target_then_requirement_then_precision():
    hi = {"diagnostics": _diag(0.96), "score": {"precision_at_tau": 0.96}}
    mid = {"diagnostics": _diag(0.70), "score": {"precision_at_tau": 0.70}}
    lo = {"diagnostics": _diag(0.40), "score": {"precision_at_tau": 0.40}}
    khi, kmid, klo = (_rank_key(hi, _OBJ), _rank_key(mid, _OBJ), _rank_key(lo, _OBJ))
    assert khi > kmid > klo
    assert khi[0] is True          # target met
    assert kmid[0] is False and kmid[1] is True   # requirement met, target not


def test_rank_key_signal_floor_discards_thin_precision():
    thin = {"diagnostics": _diag(0.99, n=0), "score": {"precision_at_tau": float("nan")}}
    key = _rank_key(thin, _OBJ)
    assert key[1] is False         # requirement not met (0 signals)


# ── column resolution ─────────────────────────────────────────────────────────

def test_resolve_columns_variants():
    avail = ["rv_a", "rv_b", "of_c", "vw_d"]
    assert set(_resolve_columns(None, avail)) == set(avail)
    assert _resolve_columns({"columns": ["rv_a", "of_c"]}, avail) == ["rv_a", "of_c"]
    assert set(_resolve_columns({"prefixes": ["rv_"]}, avail)) == {"rv_a", "rv_b"}


# ── end-to-end loop with a stubbed run_search ─────────────────────────────────

def test_refine_loop_improves_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    # parquet dir + a fake fold cache so _available_columns + episode context resolve
    pdir = tmp_path / "parquet"
    pdir.mkdir()

    all_cols = ["rv_a", "rv_b", "of_good", "of_bad", "ms_e"]

    # Stub the fold arrays used by _available_columns and _analyze_and_propose
    n = 300
    ts = pd.date_range("2025-06-02 09:00", periods=n, freq="5s",
                       tz="America/Chicago").tz_convert("UTC").to_numpy()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, len(all_cols))).astype("float32")

    import src.search.feature_refine as fr
    monkeypatch.setattr(fr, "_available_columns", lambda obj, pd_: all_cols)

    def _fake_fold(obj, pdir_, **kw):
        return X, np.zeros(n, dtype="int32"), ts, all_cols
    monkeypatch.setattr("src.search.fold_cache.build_or_load_fold_arrays", _fake_fold)

    # Stub run_search: precision improves when 'of_good' is in the feature set.
    calls = {"n": 0}

    def _fake_run_search(objective_id, *, manual_spec, ensembles, parquet_dir, dry_run):
        calls["n"] += 1
        cols = manual_spec["candidates"][0]["features"]["columns"]
        prec = 0.9 if "of_good" in cols else 0.6
        oof_dir = tmp_path / f"run_{calls['n']}"
        oof_dir.mkdir()
        # write an OOF + diagnostics so the analyzer can build episodes
        m = 300
        rng2 = np.random.default_rng(calls["n"])
        oof = pd.DataFrame({
            "ts": ts[:m],
            "direction": rng2.choice(["up", "down", "flat"], size=m),
            "confidence": rng2.uniform(0.5, 1.0, size=m),
            "direction_true": rng2.choice(["up", "down", "flat"], size=m,
                                          p=[0.2, 0.2, 0.6]),
        })
        oof_path = oof_dir / "oof_cand.parquet"
        oof.to_parquet(oof_path, index=False)
        diag = {"precision_at_tau_sweep": [
            {"tau": 0.8, "direction": "up", "precision": prec, "n_signals": 50},
            {"tau": 0.8, "direction": "down", "precision": 0.3, "n_signals": 40}]}
        (oof_dir / "diagnostics_cand.json").write_text(json.dumps(diag))
        return {
            "objective_id": objective_id,
            "candidates": [{
                "candidate_id": "cand", "status": "ok",
                "score": {"precision_at_tau": prec, "directional_f1": 0.5, "n_signals": 90},
                "oof_path": str(oof_path),
            }],
        }

    spec = {
        "schema_version": "v1", "objective_id": "test-obj", "ensembles": False,
        "candidates": [{"id": "cand", "family": "xgboost",
                        "features": {"columns": ["rv_a", "rv_b"]}}],
    }

    # Force the deterministic proposer to add of_good by making it separate outcomes.
    # We stub separation_table so the loop deterministically proposes of_good.
    import src.evaluation.patterns as pat
    monkeypatch.setattr(pat, "separation_table", lambda ep, cols=None: {
        "win_vs_loss": {"of_good": 0.9, "of_bad": 0.0, "rv_a": 0.0, "rv_b": 0.0},
        "caught_vs_missed": {}})

    # Bypass objective/manual-spec loaders (use dicts directly). These are
    # function-local imports in refine_features, so patch the source modules.
    monkeypatch.setattr("src.objectives.loader.load_objective", lambda oid: _OBJ)

    class _MS:
        raw = spec
    monkeypatch.setattr("src.search.manual_spec.load_manual_spec",
                        lambda s, objective_id=None: _MS())

    report = refine_features(
        "test-obj", spec, candidate_id="cand", max_iters=4, patience=2,
        parquet_dir=str(pdir), run_search_fn=_fake_run_search)

    assert report["n_iterations"] >= 2
    assert "of_good" in report["best_features"]     # the improving feature was kept
    assert report["winner_spec_path"] is not None
