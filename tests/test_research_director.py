"""Research Director model-improvement loop — schema, dry-run + stubbed loop, validation bridge.

Offline: StubLLM (no network/spend), hash embedder, memory/RAG writes degrade without Postgres.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"



# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _pack(baseline: dict | None = None):
    return types.SimpleNamespace(
        render=lambda: "## EXISTING MODEL & RESULT PATTERNS\nconfusion_matrix: FLAT recall 0.30",
        baseline=baseline if baseline is not None
        else {"training_result": {"precision_at_tau": 0.40,
                                  "test_folds": [0.38, 0.41, 0.42],
                                  "n_signals": {"BU": 60, "BD": 55}}},
        objective_progress=None,
    )


def _change(cid="c1", *, auto=True, scope="loss"):
    c = {"id": cid, "title": f"change {cid}", "rank": 1, "scope": scope,
         "targeted_failure_mode": "over-predicts BIG_UP on FLAT",
         "mechanism": "penalize confident wrong-direction calls in flat regimes",
         "expected_effect": {"metric": "precision_at_tau", "direction": "increase", "magnitude": "+0.03"},
         "evidence": [{"tier": "T2", "source_ref": "confusion_matrix[BIG_UP->FLAT]"}],
         "feasibility": "trainable today", "auto_validatable": auto}
    if auto:
        c["executable_candidate"] = {"family": "xgboost", "params": {"scale_pos_weight": 3}}
    else:
        c["validation_method"] = "acquire ES depth feed, then re-eval"
    return c


def _plan(top="c1", changes=None):
    return {"objective_id": "obj-x", "program_id": "prog-x",
            "generated_at": "2026-07-15T00:00:00Z",
            "dominant_failure_modes": [
                {"summary": "over-predicts BIG_UP when truth is FLAT",
                 "evidence": [{"tier": "T2", "source_ref": "confusion_matrix"}]}],
            "changes": changes or [_change("c1")],
            "top_recommendation": top}


# ── schema ───────────────────────────────────────────────────────────────────

def _validator():
    import jsonschema
    from src.common.config_paths import schema_path
    schema = json.loads(schema_path("model_improvement_plan.schema.json").read_text())
    return jsonschema.Draft202012Validator(schema)


def test_schema_accepts_valid_plan():
    assert not list(_validator().iter_errors(_plan()))


def test_schema_conditional_requires_executable_candidate():
    v = _validator()
    # auto_validatable=true but NO executable_candidate → invalid
    bad = _change("c1", auto=True)
    del bad["executable_candidate"]
    assert list(v.iter_errors(_plan(changes=[bad])))
    # auto_validatable=false but NO validation_method → invalid
    bad2 = _change("c2", auto=False)
    del bad2["validation_method"]
    assert list(v.iter_errors(_plan(top="c2", changes=[bad2])))


# ── dry-run loop (no LLM) ────────────────────────────────────────────────────────

def test_dry_run_loop_all_stages(tmp_path):
    from src.research_director.orchestrator import run_rd_loop
    state = run_rd_loop("prog-x", "obj-x", _pack(), dry_run=True, out_root=tmp_path)
    assert list(state["stages"]) == ["diagnose", "cross_review", "synthesis", "validate", "handoff"]
    assert state["validation"]["verdict"] == "HANDOFF_REQUIRED"       # dry run never trains
    run_dir = next((tmp_path / "prog-x").iterdir())
    plan = json.loads((run_dir / "model_improvement_plan.json").read_text())
    assert not list(_validator().iter_errors(plan))                  # dry plan is schema-valid
    assert (run_dir / "improvement_report.md").exists()


# ── stubbed-LLM full loop ────────────────────────────────────────────────────────

class StubLLM:
    """Deterministic provider double: routes on the requested schema title / agent name."""

    def __init__(self):
        self._critic_seen: dict[str, int] = {}

    def chat(self, *, system, user, agent, **kw):
        return "NO_REVISIONS" if "Critique your own draft" in user else f"revised by {agent}"

    def chat_tools(self, *, system, user, agent, registry, **kw):
        trace = [{"tool": "read_attribution", "args": {}, "output": {"top": ["gx_"]}}]
        return f"DIAGNOSIS+PRESCRIPTION by {agent}", trace

    def chat_json(self, *, system, user, agent, schema, **kw):
        title = schema.get("title", "")
        if title == "CriticReview":
            return self._critic_review(agent)
        if title == "ModelImprovementPlan":
            # top pick is HANDOFF (acquisition) so the loop stays offline (no run_search).
            return _plan(top="c1", changes=[_change("c1", auto=False, scope="acquisition")]) \
                | {"reconciliation": "plans agreed" if agent == "rd_judge" else None,
                   "open_questions": []}
        if title == "ScoredResearchLibrary":
            return {"objective_id": "obj-x", "traits": {}, "publications": []}
        raise AssertionError(f"unexpected schema {title}")

    def cost_tally(self):
        return {"usd": 0.0}

    def _critic_review(self, agent):
        n = self._critic_seen[agent] = self._critic_seen.get(agent, 0) + 1
        if "cross_review" in agent:
            return {"reviewer": {"role": "cross_review", "field": agent.split("_cross")[0]},
                    "target": "t", "verdict": "revise",
                    "objections": [
                        {"target_claim": "in-scope", "severity": "major", "kind": "feasibility",
                         "evidence": [{"source_ref": "cm"}], "interface_assumption": "signal cadence"},
                        {"target_claim": "off", "severity": "major", "kind": "out_of_domain",
                         "evidence": [{"source_ref": "x"}], "interface_assumption": "y"}],
                    "verification": {"checks_completed": ["read"]}}
        if n == 1:  # first round objects, second clears → convergence
            return {"reviewer": {"role": "critic", "field": "?"}, "target": "d", "verdict": "revise",
                    "objections": [{"target_claim": "local patch?", "severity": "major",
                                    "kind": "methodology", "evidence": [{"source_ref": "cm"}]}],
                    "verification": {"checks_completed": ["citation_grounding"]}}
        return {"reviewer": {"role": "critic", "field": "?"}, "target": "d",
                "verdict": "no_material_objections", "objections": [],
                "verification": {"checks_completed": ["a", "b", "c"]}}


def test_stub_llm_full_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_EMBED_BACKEND", "hash")
    from src.research_director.orchestrator import run_rd_loop
    stub = StubLLM()
    state = run_rd_loop("prog-x", "obj-x", _pack(), llm=stub, critic_llm=stub, out_root=tmp_path)

    run_dir = next((tmp_path / "prog-x").iterdir())
    plan = json.loads((run_dir / "model_improvement_plan.json").read_text())
    assert not list(_validator().iter_errors(plan))
    assert plan["program_id"] == "prog-x" and plan["objective_id"] == "obj-x"
    assert plan["reconciliation"] == "plans agreed"
    assert plan["provenance"]["critic_rounds"]                       # rounds recorded
    # top pick is handoff-required → validation skipped, not trained
    assert state["validation"]["verdict"] == "HANDOFF_REQUIRED"
    # critic converged on round 2 (objected once, then cleared)
    assert (run_dir / "diagnose_quant_trading_critic_r2.json").exists()
    # cross-review dropped the out-of-domain objection
    cross = json.loads((run_dir / "cross_review_of_ml_data_scientist.json").read_text())
    assert [o["target_claim"] for o in cross["objections"]] == ["in-scope"]
    report = (run_dir / "improvement_report.md").read_text()
    assert "Dominant failure modes" in report and "HANDOFF_REQUIRED" in report


# ── validation bridge ────────────────────────────────────────────────────────────

def _session(tmp_path, plan):
    from src.research_director.orchestrator import RDSession
    sess = RDSession("prog-x", "obj-x", _pack(), llm=object(), critic_llm=object(),
                     cfg={"fields": ["quant_trading", "ml_data_scientist"],
                          "caps": {}, "parquet_dir": "data/parquet",
                          "delta_min": 0.0, "class_prior": 0.09},
                     out_root=tmp_path)
    sess.plan = plan
    return sess


def test_validate_autovalidatable_runs_search_and_maps_verdict(tmp_path, monkeypatch):
    from src.research_director import orchestrator as orch
    import src.search.controller as ctrl

    captured = {}

    def fake_run_search(objective_id, *, parquet_dir=None, manual_spec=None, ensembles=False, tune_trials=0):
        captured["objective_id"] = objective_id
        captured["manual_spec"] = manual_spec
        # candidate scores BETTER than baseline (0.40) on all folds → Evaluator PASS
        return {"run_ts": "rt", "champion": "xgboost", "candidates": [
            {"candidate_id": "c1", "family": "xgboost",
             "score": {"precision_at_tau": 0.46, "test_folds": [0.44, 0.47, 0.48],
                       "n_signals": {"BU": 70, "BD": 60}}}]}

    monkeypatch.setattr(ctrl, "run_search", fake_run_search)
    # baseline carries per-fold metrics so the Evaluator's multi-fold/CI gates can PASS.
    pk = _pack({"training_result": {"precision_at_tau": 0.40, "test_folds": [0.38, 0.40, 0.41],
                                    "n_signals": {"BU": 60, "BD": 55}}})
    sess = _session(tmp_path, _plan(changes=[_change("c1", auto=True)]))
    sess.baseline_tr = pk.baseline["training_result"]

    orch._stage_validate(sess)
    assert captured["objective_id"] == "obj-x"
    assert captured["manual_spec"]["objective_id"] == "obj-x"        # objective stamped
    assert captured["manual_spec"]["candidates"][0]["family"] == "xgboost"
    assert sess.validation["verdict"] == "VALIDATED_IMPROVED"
    assert sess.validation["primary_delta"] is not None and sess.validation["primary_delta"] > 0


def test_validate_handoff_skips_search(tmp_path, monkeypatch):
    from src.research_director import orchestrator as orch
    import src.search.controller as ctrl
    monkeypatch.setattr(ctrl, "run_search",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not train")))
    sess = _session(tmp_path, _plan(top="c1", changes=[_change("c1", auto=False, scope="acquisition")]))
    orch._stage_validate(sess)
    assert sess.validation["verdict"] == "HANDOFF_REQUIRED"
    assert sess.validation["validation_method"]


def test_validate_search_error_is_inconclusive(tmp_path, monkeypatch):
    from src.research_director import orchestrator as orch
    import src.search.controller as ctrl
    monkeypatch.setattr(ctrl, "run_search",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("parquet missing")))
    sess = _session(tmp_path, _plan(changes=[_change("c1", auto=True)]))
    orch._stage_validate(sess)
    assert sess.validation["verdict"] == "INCONCLUSIVE"
    assert "parquet missing" in sess.validation["reason"]


# ── unit helpers ─────────────────────────────────────────────────────────────────

def test_run_metrics_preserves_precision_at_tau_key():
    from src.research_director.orchestrator import _run_metrics_from_candidate
    m = _run_metrics_from_candidate({"score": {"precision_at_tau": 0.44, "directional_f1": 0.5,
                                               "n_signals": {"BU": 10, "BD": 9}, "note": "x"}})
    assert m["precision_at_tau"] == 0.44          # NOT remapped to val_precision_at_tau
    assert m["n_signals"] == {"BU": 10, "BD": 9}
    assert "note" not in m
    # int n_signals normalizes to a dict for the Evaluator's cold-start floor
    m2 = _run_metrics_from_candidate({"score": {"precision_at_tau": 0.4, "n_signals": 5}})
    assert m2["n_signals"] == {"total": 5}


def test_rd_signature_stable_and_scoped():
    from src.research_director.orchestrator import rd_signature
    a = rd_signature(_change("c1"))
    b = rd_signature(_change("c1"))
    c = rd_signature(_change("c1", scope="feature"))
    assert a == b and a != c and len(a) == 16


def test_normalize_feature_selection_strips_extras_and_prefix_columns():
    from src.research_director.orchestrator import (
        _normalize_feature_selection, _normalize_executable_candidate,
    )
    # Extra keys + view → keep view only
    assert _normalize_feature_selection({
        "view": "momentum_edge",
        "context_view": ["rv_regime_state"],
        "meta_inputs": ["primary_confidence"],
    }) == {"view": "momentum_edge"}
    # note + columns with prefixes mixed in
    assert _normalize_feature_selection({
        "columns": ["meta_confidence", "bp_depth_total", "vw_", "rv_", "iv_vix_close"],
        "note": "PROVISIONAL",
    }) == {"columns": ["meta_confidence", "bp_depth_total", "iv_vix_close"]}
    # pure prefix-like columns → prefixes mode
    assert _normalize_feature_selection({
        "columns": ["vw_", "rv_", "of_"],
    }) == {"prefixes": ["vw_", "rv_", "of_"]}
    # invented key recovery
    assert _normalize_feature_selection({
        "meta_context_prefixes": ["rv_", "bp_"],
    }) == {"prefixes": ["rv_", "bp_"]}
    assert _normalize_feature_selection(None) is None

    cand = _normalize_executable_candidate({
        "id": "meta_gate_v1",
        "family": "xgboost",
        "params": {"max_depth": 4},
        "features": {"view": "regime_core", "columns": ["should_drop"]},
        "ensembles": True,
        "junk": "drop_me",
    })
    assert cand == {
        "family": "xgboost", "id": "meta_gate_v1", "params": {"max_depth": 4},
        "ensembles": True, "features": {"view": "regime_core"},
    }


def test_validate_normalizes_features_before_search(tmp_path, monkeypatch):
    from src.research_director import orchestrator as orch
    import src.search.controller as ctrl

    captured = {}

    def fake_run_search(objective_id, *, parquet_dir=None, manual_spec=None, ensembles=False, tune_trials=0):
        captured["manual_spec"] = manual_spec
        return {"run_ts": "rt", "champion": "xgboost", "candidates": [
            {"candidate_id": "c1", "family": "xgboost",
             "score": {"precision_at_tau": 0.46, "test_folds": [0.44, 0.47, 0.48],
                       "n_signals": {"BU": 70, "BD": 60}}}]}

    monkeypatch.setattr(ctrl, "run_search", fake_run_search)
    change = _change("c1", auto=True)
    change["executable_candidate"] = {
        "family": "xgboost", "id": "c1",
        "features": {"view": "momentum_edge", "context_view": ["rv_"], "note": "x"},
    }
    pk = _pack({"training_result": {"precision_at_tau": 0.40, "test_folds": [0.38, 0.40, 0.41],
                                    "n_signals": {"BU": 60, "BD": 55}}})
    sess = _session(tmp_path, _plan(changes=[change]))
    sess.baseline_tr = pk.baseline["training_result"]
    orch._stage_validate(sess)
    feats = captured["manual_spec"]["candidates"][0]["features"]
    assert feats == {"view": "momentum_edge"}
    assert sess.validation["verdict"] == "VALIDATED_IMPROVED"
    # Raw pre-normalize artifact preserved for audit
    run_dir = next((tmp_path / "prog-x").iterdir())
    raw = json.loads((run_dir / "validation_manual_spec_raw.json").read_text())
    assert "context_view" in raw["candidates"][0]["features"]
