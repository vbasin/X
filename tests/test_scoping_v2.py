"""Scoping v2 (ADR-025) — state machine, RAG, tiers, critic, gap differ, journal gate.

DB-backed RAG tests use the real Postgres (repo convention) and skip cleanly when it is
unreachable. LLM-dependent paths run against a StubLLM — no network, no spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"


# ── chunker + hash embedder ────────────────────────────────────────────────────

def test_chunker_packs_and_splits():
    from src.scoping.rag import chunk_text
    text = "para one\n\npara two\n\n" + "x" * 3000
    chunks = chunk_text(text, max_chars=1200, overlap=150)
    assert chunks[0] == "para one\n\npara two"
    assert all(len(c) <= 1200 for c in chunks)
    assert chunk_text(text, max_chars=1200, overlap=150) == chunks   # deterministic
    assert chunk_text("") == []


def test_hash_embedder_deterministic_normalized():
    from src.scoping.embeddings import Embedder
    emb = Embedder("hash", hash_dim=64)
    v1, v2 = emb.embed(["regime gating precision"] * 2)
    assert v1 == v2 and len(v1) == 64
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9                  # L2-normalized
    assert emb.model_id == "hash:64"
    assert emb.embed([]) == []


# ── T3 journal write gate ──────────────────────────────────────────────────────

def _valid_entry() -> dict:
    return {
        "objective_id": "test-obj", "mlflow_run_id": "abc123",
        "model": {"family": "xgboost", "candidate_id": "xgb1"},
        "live_period": {"start": "2026-06-01", "end": "2026-07-01"},
        "realized_metrics": {"precision_at_tau": 0.86, "n_signals": 41},
        "success_rationale": "Cleared the 0.85 precision gate live over a full month with n=41 fires.",
        "user_confirmed_at": "2026-07-14T00:00:00+00:00",
    }


def test_journal_write_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    from src.scoping import journal

    with pytest.raises(PermissionError):                    # unconfirmed = refused
        journal.add_entry("test-obj", _valid_entry())
    assert journal.load_journal("test-obj") == []

    p = journal.add_entry("test-obj", _valid_entry(), confirmed=True)
    assert p.exists()
    entries = journal.load_journal("test-obj")
    assert len(entries) == 1 and entries[0]["model"]["family"] == "xgboost"

    bad = _valid_entry()
    del bad["realized_metrics"]                             # schema-invalid even if confirmed
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        journal.add_entry("test-obj", bad, confirmed=True)


# ── critic: objection utilities + anti-sycophancy schema rule ──────────────────

def test_filter_cross_objections_drops_untagged_and_out_of_domain():
    from src.scoping.critic import filter_cross_objections
    review = {"reviewer": {"field": "quant_trading"}, "objections": [
        {"target_claim": "a", "severity": "major", "kind": "feasibility",
         "evidence": [{"source_ref": "x"}], "interface_assumption": "VIX candle timeliness"},
        {"target_claim": "b", "severity": "major", "kind": "methodology",
         "evidence": [{"source_ref": "y"}], "interface_assumption": None},
        {"target_claim": "c", "severity": "minor", "kind": "out_of_domain",
         "evidence": [{"source_ref": "z"}], "interface_assumption": "anything"},
    ]}
    kept, dropped = filter_cross_objections(review)
    assert [o["target_claim"] for o in kept] == ["a"] and dropped == 2


def test_material_objections():
    from src.scoping.critic import material_objections
    r = {"objections": [{"severity": s} for s in ("blocker", "major", "minor")]}
    assert len(material_objections(r)) == 2


def test_critic_schema_no_material_requires_completed_checks():
    import jsonschema
    from src.common.config_paths import schema_path
    schema = json.loads(schema_path("critic_objection.schema.json").read_text())
    base = {"reviewer": {"role": "critic", "field": "quant_trading"},
            "target": "pass_a r1", "objections": [],
            "verification": {"checks_completed": ["citation_grounding"]}}
    with pytest.raises(jsonschema.ValidationError):         # 1 check < 3 required
        jsonschema.validate({**base, "verdict": "no_material_objections"}, schema)
    jsonschema.validate({**base, "verdict": "revise"}, schema)             # revise = fine
    ok = {**base, "verdict": "no_material_objections",
          "verification": {"checks_completed": ["citation_grounding",
                                                "omission_probe:regimes",
                                                "counter_evidence:tbm"]}}
    jsonschema.validate(ok, schema)


# ── gap check v2 ───────────────────────────────────────────────────────────────

def _plan(candidates, ensembles=()):
    return {"objective_id": "nq-50pt-tbm", "generated_at": "now",
            "model_plan": {"candidates": list(candidates), "ensembles": list(ensembles)},
            "data_requirements": {"data_classes": []}}


def _cand(family, role="primary", features=None):
    c = {"family": family, "role": role, "rationale": "r",
         "evidence": [{"tier": "T2", "source_ref": "s"}]}
    if features:
        c["features"] = features
    return c


def test_gap_check_v2_model_and_feature_coverage(tmp_path):
    from src.objectives.loader import load_objective
    from src.scoping.gap_check import gap_check_v2
    objective = load_objective("nq-50pt-tbm")

    plan = _plan(
        [_cand("xgboost"),
         _cand("no_such_family"),
         _cand("hmm", role="gate", features={"view": "no_such_view"})],
        ensembles=[{"kind": "gate_veto", "members": ["xgboost", "hmm"], "rationale": "r"},
                   {"kind": "quantum_blend", "members": ["a", "b"], "rationale": "r"}])
    report = gap_check_v2(objective, plan, tmp_path)

    mc = {m["family"]: m["status"] for m in report["model_coverage"]}
    assert mc["xgboost"] == "have" and mc["no_such_family"] == "missing"
    assert mc["gate_veto"] == "have" and mc["quantum_blend"] == "missing"
    fc = {f["selector"]: f["status"] for f in report["feature_coverage"]}
    assert fc["view:no_such_view"] == "missing"
    assert report["summary"]["recommendation"] == "revise_plan"
    assert report["summary"]["n_model_missing"] == 2

    ok_plan = _plan([_cand("xgboost")])
    ok = gap_check_v2(objective, ok_plan, tmp_path)
    assert ok["summary"]["recommendation"] != "revise_plan"
    assert ok["model_coverage"][0]["status"] == "have"
    # criticality ranking: items sorted descending
    crits = [i.get("criticality", 1) for i in report["items"]]
    assert crits == sorted(crits, reverse=True)


def test_manual_candidates_draft_validates_against_schema():
    import jsonschema
    from src.scoping.orchestrator import _draft_manual_candidates
    plan = _plan([_cand("xgboost", features={"view": "trees_full"}),
                  _cand("attention_lstm", role="member")])
    plan["model_plan"]["candidates"][0]["params_hint"] = {"max_depth": 8}
    plan["model_plan"]["candidates"][0]["id"] = "xgb_tuned"
    draft = _draft_manual_candidates(plan)
    from src.common.config_paths import schema_path
    schema = json.loads(schema_path("manual_candidates.schema.json").read_text())
    jsonschema.validate(draft, schema)
    assert draft["candidates"][0] == {"family": "xgboost", "id": "xgb_tuned",
                                      "params": {"max_depth": 8},
                                      "features": {"view": "trees_full"}}


# ── orchestrator: dry-run state machine ────────────────────────────────────────

def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setenv("VTECH_RESULTS_INDEX_LOCAL", str(tmp_path / "ridx"))
    monkeypatch.setenv("VTECH_SCOUT_JOURNAL_LOCAL", str(tmp_path / "scout"))


def test_orchestrator_dry_run_all_stages(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    from src.scoping.orchestrator import STAGES, run_scoping_v2
    state = run_scoping_v2("nq-50pt-tbm", dry_run=True, out_root=tmp_path / "scoping")
    assert set(state["stages"]) == set(STAGES)

    run_dir = next((tmp_path / "scoping" / "nq-50pt-tbm").iterdir())
    for name in ("pass_a_quant_trading.md", "pass_a_ml_data_scientist.md",
                 "pass_b_proven_evidence.json",
                 "scoping_plan.json", "data_gap_report.json",
                 "manual_candidates.draft.yaml", "final_research_report.md", "state.json"):
        assert (run_dir / name).exists(), name
    gap = json.loads((run_dir / "data_gap_report.json").read_text())
    assert gap["summary"]["recommendation"] == "revise_plan"       # empty plan flagged


# ── orchestrator: stub-LLM full loop (critic convergence, cross-review, synthesis) ──

class StubLLM:
    """Deterministic provider double: routes on the requested schema title / agent name."""

    def __init__(self, critic_always_objects: bool = False):
        self.critic_always_objects = critic_always_objects
        self.calls: list[str] = []
        self._critic_seen: dict[str, int] = {}

    # duck-typed provider contract ------------------------------------------------
    def chat(self, *, system, user, agent, **kw):
        self.calls.append(f"chat:{agent}")
        if "Critique your own draft" in user:
            return "NO_REVISIONS"
        return f"revised draft by {agent}"

    def chat_tools(self, *, system, user, agent, registry, **kw):
        self.calls.append(f"tools:{agent}")
        trace = [{"tool": "search_research", "args": {"query": "tbm"},
                  "output": [{"source_ref": "pub1", "similarity": 0.9}]}]
        return f"draft by {agent} citing pub1", trace

    def chat_json(self, *, system, user, agent, schema, **kw):
        self.calls.append(f"json:{agent}")
        title = schema.get("title", "")
        if title == "CriticReview":
            return self._critic_review(agent)
        if title == "ScopingPlan":
            return _plan([_cand("xgboost", features={"view": "trees_full"}),
                          _cand("hmm", role="gate")],
                         ensembles=[{"kind": "gate_veto",
                                     "members": ["xgboost", "hmm"], "rationale": "r"}]) \
                | {"reconciliation": "plans agreed" if agent == "scoping_judge" else None,
                   "open_questions": []}
        if title == "ScoredResearchLibrary":
            return {"objective_id": "nq-50pt-tbm", "traits": {}, "publications": [
                {"id": f"pub{i}", "title": f"t{i}", "source_type": "academic",
                 "scores": {"relevance": 4, "complexity": 2, "reliability": 4,
                            "insightfulness": 3}} for i in range(10)]}
        raise AssertionError(f"unexpected schema {title}")

    def cost_tally(self):
        return {"usd": 0.0}

    def _critic_review(self, agent):
        n = self._critic_seen[agent] = self._critic_seen.get(agent, 0) + 1
        if "cross_review" in agent:
            return {"reviewer": {"role": "cross_review", "field": agent.split("_cross")[0]},
                    "target": "pass_b", "verdict": "revise",
                    "objections": [
                        {"target_claim": "in-scope", "severity": "major", "kind": "feasibility",
                         "evidence": [{"source_ref": "pub1"}],
                         "interface_assumption": "granularity learnable"},
                        {"target_claim": "off-scope", "severity": "major", "kind": "out_of_domain",
                         "evidence": [{"source_ref": "pub2"}], "interface_assumption": "x"}],
                    "verification": {"checks_completed": ["read"]}}
        object_now = self.critic_always_objects or n == 1
        if object_now:
            return {"reviewer": {"role": "critic", "field": "?"}, "target": "d",
                    "verdict": "revise",
                    "objections": [{"target_claim": "uncited", "severity": "major",
                                    "kind": "unsupported_claim",
                                    "evidence": [{"source_ref": "pub1"}]}],
                    "verification": {"checks_completed": ["citation_grounding"]}}
        return {"reviewer": {"role": "critic", "field": "?"}, "target": "d",
                "verdict": "no_material_objections", "objections": [],
                "verification": {"checks_completed": ["citation_grounding",
                                                      "omission_probe:x",
                                                      "counter_evidence:y"]}}


def test_orchestrator_stub_llm_full_loop(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    from src.scoping.orchestrator import run_scoping_v2
    stub = StubLLM()
    state = run_scoping_v2("nq-50pt-tbm", llm=stub, critic_llm=stub,
                           out_root=tmp_path / "scoping")
    run_dir = next((tmp_path / "scoping" / "nq-50pt-tbm").iterdir())

    # Pass B has no user-confirmed proven-success journal → internal integration is skipped,
    # but the interface cross-review still runs and synthesis still produces a plan.
    proven = json.loads((run_dir / "pass_b_proven_evidence.json").read_text())
    assert proven["entries"] == []
    assert state["stages"]["pass_b"].get("internal_integration") == "skipped"
    assert (run_dir / "cross_review_of_quant_trading.json").exists()

    plan = json.loads((run_dir / "scoping_plan.json").read_text())
    assert plan["objective_id"] == "nq-50pt-tbm"
    assert plan["reconciliation"] == "plans agreed"                 # replication reconciled
    assert (run_dir / "replication_plan.json").exists()
    assert plan["provenance"]["critic_rounds"]                     # rounds recorded

    # critic converged on round 2 (first round objects, second clears)
    assert (run_dir / "pass_a_quant_trading_critic_r2.json").exists()

    # cross-review: out-of-domain objection dropped, in-scope kept
    cross = json.loads((run_dir / "cross_review_of_quant_trading.json").read_text())
    assert [o["target_claim"] for o in cross["objections"]] == ["in-scope"]
    assert cross["n_dropped_out_of_domain"] == 1

    # retrieval traces captured (critic's required input)
    trace = json.loads((run_dir / "pass_a_quant_trading_trace.json").read_text())
    assert trace["trace"][0]["tool"] == "search_research"

    # machine-consumable handoff
    import yaml
    manual = yaml.safe_load((run_dir / "manual_candidates.draft.yaml").read_text())
    assert {c["family"] for c in manual["candidates"]} == {"xgboost", "hmm"}
    gap = json.loads((run_dir / "data_gap_report.json").read_text())
    assert gap["summary"]["recommendation"] != "revise_plan"        # runnable plan

    report = (run_dir / "final_research_report.md").read_text()
    assert "xgboost" in report and "HUMAN GATE" in report

    # cumulative T1 library write-back
    lib = json.loads((tmp_path / "scoping" / "_library" / "research_library.json").read_text())
    assert len(lib["publications"]) == 10                           # deduped across 2 fields


def test_critic_convergence_cap(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    from src.scoping.orchestrator import load_scoping_cfg, run_scoping_v2
    cfg = load_scoping_cfg()
    cfg["caps"]["critic_rounds"] = 2
    stub = StubLLM(critic_always_objects=True)                      # never satisfied
    run_scoping_v2("nq-50pt-tbm", llm=stub, critic_llm=stub,
                   out_root=tmp_path / "scoping", cfg=cfg)
    run_dir = next((tmp_path / "scoping" / "nq-50pt-tbm").iterdir())
    assert (run_dir / "pass_a_quant_trading_critic_r2.json").exists()      # hit the cap…
    assert not (run_dir / "pass_a_quant_trading_critic_r3.json").exists()  # …and stopped


# ── RAG round-trip (live DB — skips when Postgres is unreachable) ──────────────

def _db_available() -> bool:
    try:
        import psycopg
        from src.common.db import get_dsn
        with psycopg.connect(get_dsn(), connect_timeout=3):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="Postgres unreachable")
def test_rag_roundtrip_live():
    import psycopg
    from src.common.db import get_dsn
    from src.scoping.embeddings import Embedder
    from src.scoping.rag import citation_exists, ensure_schema, index_document, search_research

    ensure_schema()
    emb = Embedder("hash", hash_dim=64)
    try:
        n = index_document("pytest:rag-doc",
                           "Triple barrier stop-loss barriers.\n\nRegime veto gating lifts precision.",
                           tier="t1", objective_id="_pytest", embedder=emb)
        assert n >= 1
        # dedupe: second index is a no-op
        assert index_document("pytest:rag-doc",
                              "Triple barrier stop-loss barriers.\n\nRegime veto gating lifts precision.",
                              tier="t1", objective_id="_pytest", embedder=emb) == 0
        hits = search_research("regime veto precision", k=5, embedder=emb)
        assert hits and hits[0]["source_ref"] == "pytest:rag-doc"
        assert search_research("regime veto precision", tier="t3", k=5, embedder=emb) == []
        assert citation_exists("pytest:rag-doc")["exists"] is True
        assert citation_exists("pytest:not-a-source")["exists"] is False
    finally:
        with psycopg.connect(get_dsn()) as conn:
            conn.execute("DELETE FROM scoping.research_chunks WHERE source_ref LIKE 'pytest:%'")
            conn.commit()
