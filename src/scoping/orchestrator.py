"""Scoping v2 orchestrator — the 5-stage deterministic state machine (ADR-025).

    Objective digest
      → Pass A   external research (T1)               [dual persona, blank slate]
      → Pass B   internal evidence (B1 auto T2 + B2 user-gated T3) + interface cross-review
      → Synthesis (judge + ONE independent replication → reconciled ScopingPlan)
      → Pass C   deterministic gap check v2            [code, no LLM]
      → Final Research Report → HUMAN GATE → search controller

Workflow-over-autonomy: the stage graph is fixed; agents reason only inside bounded rounds
(author self-critique ≤ cap → parameterized critic ≤ cap, converging on "no material
objections"). Every artifact is schema-validated and written to
``data/research/scoping/<objective_id>/<run_ts>/``; every author round logs its RETRIEVAL TRACE
(the critic's required input). "LLM proposes, controller disposes": the emitted plan is a
starting-point hypothesis set — the gap check, the human gate, and the full-fidelity search
controller remain the arbiters.

Observability: MLflow experiment ``scoping/<objective_id>`` — parent run per session,
nested run per stage, all artifacts attached (best-effort, never fatal).

CLI::

    python -m src.scoping.orchestrator --objective nq-50pt-tbm [--dry-run] [--budget 30]
"""

from __future__ import annotations

from src.common.data_paths import data_root, scoping_dir

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from src.common.config_paths import runtime_path, schema_path
from typing import Any

import yaml

logger = logging.getLogger("vtech.scoping.orchestrator")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "config"

STAGES = ("pass_a", "pass_b", "synthesis", "gap_check", "final_report")
_DRAFT_CLIP = 20000       # chars of a draft re-injected into follow-up prompts (≈ max draft size:
                          # author rounds emit up to 8192 tokens — clipping below that loses tails)
_AUTHOR_MAX_TOKENS = 8192  # draft/revision output budget (Opus supports far more; avoids truncation)
_NO_REVISIONS = "NO_REVISIONS"


def load_scoping_cfg() -> dict[str, Any]:
    """The scoping config: ``config/scoping.yaml`` (canonical since 2026-07-14), falling
    back to the legacy ``research_director.yaml → scoping:`` block, then defaults."""
    cfg: dict[str, Any] = {}
    try:
        own = runtime_path("scoping.yaml")
        if own.exists():
            cfg = yaml.safe_load(own.read_text()) or {}
        else:
            cfg = (yaml.safe_load(runtime_path("research_director.yaml").read_text()) or {}).get("scoping") or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("scoping_cfg_unreadable: %s — using defaults", exc)
    caps = cfg.setdefault("caps", {})
    caps.setdefault("self_critique", 3)
    caps.setdefault("critic_rounds", 4)
    caps.setdefault("synthesis_replications", 1)
    caps.setdefault("max_tool_iters", 10)
    cfg.setdefault("fields", ["quant_trading", "ml_data_scientist"])
    cfg.setdefault("providers", {"author": "auto", "critic": "other"})
    cfg.setdefault("cost", {"usd_cap_session": 30.0, "per_agent_usd_cap": 10.0})
    return cfg


# ── session ───────────────────────────────────────────────────────────────────

class ScopingSession:
    """Holds one run's objective, LLMs, config, artifact dir, and rolling state."""

    def __init__(self, objective_id: str, *, dry_run: bool = False,
                 llm: Any = None, critic_llm: Any = None,
                 out_root: Path | None = None, cfg: dict[str, Any] | None = None) -> None:
        from src.objectives.loader import load_objective
        self.objective = load_objective(objective_id)
        self.objective_id = objective_id
        self.cfg = cfg or load_scoping_cfg()
        self.dry_run = dry_run
        self.run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        root = out_root or scoping_dir()
        self.out_dir = root / objective_id / self.run_ts
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.duties = self._load_duties()
        self.llm = llm
        self.critic_llm = critic_llm
        self.author_provider = "none"
        if not dry_run and self.llm is None:
            self.llm, self.critic_llm, self.author_provider = self._make_llms()
        elif self.llm is not None:
            self.author_provider = type(self.llm).__name__
            self.critic_llm = critic_llm or self.llm

        self.state: dict[str, Any] = {"objective_id": objective_id, "run_ts": self.run_ts,
                                      "dry_run": dry_run, "stages": {}}
        self.drafts: dict[str, str] = {}                 # field → final draft text
        self.traces: dict[str, list[dict[str, Any]]] = {}  # field → retrieval trace
        self.cross_objections: dict[str, list[dict[str, Any]]] = {}
        self.library_additions: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.gap_report: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    # ── LLM wiring ─────────────────────────────────────────────────────────────
    def _make_llms(self) -> tuple[Any, Any, str]:
        from src.common.llm_factory import make_llm
        from src.scoping.critic import resolve_critic_provider
        cost = self.cfg.get("cost") or {}
        cap, per = float(cost.get("usd_cap_session", 30.0)), float(cost.get("per_agent_usd_cap", 10.0))
        rates = cost.get("model_rates") or {}          # real $/1k rates — makes the cap LIVE
        prov = (self.cfg.get("providers") or {}).get("author", "auto")
        author = make_llm(usd_cap_cycle=cap, per_agent_usd_cap=per, model_rates=rates,
                          provider=None if prov == "auto" else prov)
        author_provider = "anthropic" if type(author).__name__ == "AnthropicLLM" else "openai"
        providers = self.cfg.get("providers") or {}
        critic_pol = providers.get("critic", "other")
        critic_model = (providers.get("critic_model") or "").strip() or None
        critic_prov = resolve_critic_provider(author_provider, critic_pol)
        author_model = getattr(author, "_model", None)
        same_backend = critic_prov in (author_provider, None) and (
            critic_model is None or critic_model == author_model)
        if same_backend:
            critic = author
        else:
            kw: dict[str, Any] = {"usd_cap_cycle": cap, "per_agent_usd_cap": per,
                                  "model_rates": rates, "provider": critic_prov}
            if critic_model:
                kw["model"] = critic_model
            if critic_prov == "openai":
                # Dedicated critic endpoint (e.g. the OpenAI platform for GPT-5.6): the
                # designated key env is REQUIRED — never drift onto the shared RD_LLM_*
                # endpoint with a platform-only model (it would 404 at call time).
                key_env = providers.get("critic_api_key_env", "OPENAI_API_KEY")
                base_env = providers.get("critic_base_url_env", "OPENAI_BASE_URL")
                kw["api_key"] = os.environ.get(key_env)
                # ALWAYS pin the base_url: without this, LLMClient inherits RD_LLM_BASE_URL
                # (e.g. GitHub Models) and sends the platform key to the wrong endpoint → 401.
                kw["base_url"] = os.environ.get(base_env) or "https://api.openai.com/v1"
                if providers.get("critic_reasoning_effort"):
                    kw["reasoning_effort"] = providers["critic_reasoning_effort"]
                kw["web_tools_enabled"] = bool(providers.get("critic_web", False))
            try:
                if critic_prov == "openai" and not kw.get("api_key"):
                    raise ValueError(f"critic endpoint key missing — set "
                                     f"{providers.get('critic_api_key_env', 'OPENAI_API_KEY')} in .env")
                critic = make_llm(**kw)
            except Exception as exc:  # noqa: BLE001 — a missing critic endpoint degrades, never blocks
                logger.warning("critic_llm_unavailable (%s) — falling back to the author client "
                               "(decorrelation lost this session)", exc)
                critic = author
        logger.info("scoping_llms author=%s(%s) critic=%s(%s)",
                    author_provider, author_model,
                    critic_prov or author_provider, getattr(critic, "_model", "?"))
        return author, critic, author_provider

    def _load_duties(self) -> dict[str, Any]:
        from src.scoping.critic import load_duties
        try:
            return load_duties()
        except Exception as exc:  # noqa: BLE001
            logger.warning("scoping_duties_unreadable: %s", exc)
            return {"duties": {}, "red_flags": {}}

    def duty(self, key: str) -> str:
        return (self.duties.get("duties") or {}).get(key, "")

    def persona(self, field: str) -> str:
        """Role-neutral domain-expertise head of the field persona (RD-loop protocol stripped)."""
        from src.scoping.critic import field_persona
        return field_persona(field)

    # ── artifact IO ────────────────────────────────────────────────────────────
    def write(self, name: str, content: str | dict[str, Any],
              schema_name: str | None = None) -> Path:
        path = self.out_dir / name
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_text(json.dumps(content, indent=2, default=str))
            if schema_name:
                try:
                    import jsonschema
                    schema = json.loads(schema_path(schema_name).read_text())
                    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(content))
                    if errs:
                        logger.warning("artifact_schema_errors file=%s n=%d first=%s",
                                       name, len(errs), errs[0].message)
                except Exception:  # noqa: BLE001
                    pass
        return path

    def mark(self, stage: str, **extra: Any) -> None:
        self.state["stages"][stage] = {"status": "done",
                                       "at": datetime.now(timezone.utc).isoformat(), **extra}
        self.write("state.json", self.state)


# ── public entry point ─────────────────────────────────────────────────────────

def run_scoping_v2(
    objective_id: str,
    *,
    dry_run: bool = False,
    llm: Any = None,
    critic_llm: Any = None,
    out_root: Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full Scoping-v2 loop; returns the final state dict (see module docstring)."""
    sess = ScopingSession(objective_id, dry_run=dry_run, llm=llm, critic_llm=critic_llm,
                          out_root=out_root, cfg=cfg)
    logger.info("scoping_v2_start objective=%s run_ts=%s dry_run=%s out=%s",
                objective_id, sess.run_ts, dry_run, sess.out_dir)
    t0 = time.time()
    stage_fns = (("pass_a", _stage_pass_a), ("pass_b", _stage_pass_b),
                 ("synthesis", _stage_synthesis), ("gap_check", _stage_gap_check),
                 ("final_report", _stage_final_report))
    try:
        for name, fn in stage_fns:
            fn(sess)
    except Exception as exc:
        # Any mid-run failure (cost-cap halt, provider error, schema fail after repair):
        # record it, persist state + MLflow so the partial run stays observable, then raise.
        failed = next((n for n, _ in stage_fns if n not in sess.state["stages"]), "?")
        sess.state["stages"][failed] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                        "at": datetime.now(timezone.utc).isoformat()}
        sess.state["elapsed_s"] = round(time.time() - t0, 1)
        if sess.llm is not None and hasattr(sess.llm, "cost_tally"):
            try:
                sess.state["cost"] = sess.llm.cost_tally()
            except Exception:  # noqa: BLE001
                pass
        sess.write("state.json", sess.state)
        _log_to_mlflow(sess)
        logger.error("scoping_v2_failed stage=%s: %s", failed, exc)
        raise
    sess.state["elapsed_s"] = round(time.time() - t0, 1)
    if sess.llm is not None and hasattr(sess.llm, "cost_tally"):
        try:
            sess.state["cost"] = sess.llm.cost_tally()
        except Exception:  # noqa: BLE001
            pass
    sess.write("state.json", sess.state)
    _log_to_mlflow(sess)
    _print_summary(sess)
    return sess.state


# ── stage A: external research (T1, blank slate) ──────────────────────────────

def _stage_pass_a(sess: ScopingSession) -> None:
    t0 = time.time()
    for field in sess.cfg["fields"]:
        if sess.dry_run:
            draft, trace = f"(dry run) Pass A draft — {field}", []
        else:
            registry = _pass_a_registry(sess)
            draft, trace = _author_round(
                sess, field, registry,
                duty=sess.duty("pass_a"),
                user=_objective_block(sess) + "\n\n" + sess.duty("pass_a"),
                agent=f"{field}_pass_a")
            draft = _self_critique(sess, field, draft, agent=f"{field}_pass_a")
            draft, n_rounds, reviews = _critic_loop(sess, field, draft, trace, target=f"pass_a_{field}",
                                                    registry=registry, duty=sess.duty("pass_a"))
            sess.metrics[f"pass_a_{field}_critic_rounds"] = n_rounds
            for i, r in enumerate(reviews):
                sess.write(f"pass_a_{field}_critic_r{i + 1}.json", r, "critic_objection.schema.json")
            _harvest_library(sess, field, draft)
        sess.drafts[field] = draft
        sess.traces[field] = trace
        sess.write(f"pass_a_{field}.md", draft)
        sess.write(f"pass_a_{field}_trace.json", {"field": field, "trace": trace})
    sess.mark("pass_a", elapsed_s=round(time.time() - t0, 1))


# ── stage B: internal evidence (T2 auto + T3 gated) + cross-review ────────────

def _stage_pass_b(sess: ScopingSession) -> None:
    t0 = time.time()
    from src.scoping.journal import load_proven_evidence

    # ADR-025 + 2026-07-15 directive: Pass B integrates ONLY user-confirmed PROVEN-SUCCESS
    # research (the T3 journal, matched across base-key-sibling objectives). Un-confirmed
    # experimental data (results_index / leaderboard / MLflow "champions") is a starting
    # base model, not a finalized success, and is NOT a Pass-B source. When no proven-
    # success entry exists, there is no legitimate internal evidence — the Pass-A drafts
    # stand and the internal-integration author round is SKIPPED.
    proven = [] if sess.dry_run else load_proven_evidence(sess.objective_id)
    sess.write("pass_b_proven_evidence.json",
               {"tier": "T3", "objective_id": sess.objective_id, "entries": proven})

    if not proven:
        reason = "dry_run" if sess.dry_run else "no_user_confirmed_proven_success"
        logger.info("pass_b_internal_integration_skipped objective=%s reason=%s",
                    sess.objective_id, reason)
        # The dual-perspective interface cross-review still runs — it reconciles the two
        # experts' Pass-A plans and needs no internal evidence.
        if not sess.dry_run:
            _cross_review(sess)
        sess.mark("pass_b", status="skipped" if not sess.dry_run else "done",
                  elapsed_s=round(time.time() - t0, 1),
                  internal_integration="skipped", reason=reason, proven_entries=0)
        return

    evidence_block = (
        "## T3 PROVEN-SUCCESS JOURNAL (user-confirmed live results — the ONLY internal\n"
        "evidence tier; includes base-key-sibling objectives. Each entry's `success_rationale`\n"
        "states WHY it is a proven success; treat it as authoritative unless its `_match` is\n"
        "'base' and this objective's requirements differ materially.)\n"
        + _clip(json.dumps(proven, default=str), 12000))

    for field in sess.cfg["fields"]:
        registry = _pass_b_registry(sess)
        draft, trace = _author_round(
            sess, field, registry,
            duty=sess.duty("pass_b"),
            user=(_objective_block(sess) + "\n\n" + sess.duty("pass_b")
                  + "\n\n## YOUR PASS-A DRAFT\n" + _clip(sess.drafts[field])
                  + "\n\n" + evidence_block),
            agent=f"{field}_pass_b")
        sess.traces[field] = (sess.traces.get(field) or []) + trace
        draft = _self_critique(sess, field, draft, agent=f"{field}_pass_b")
        draft, n_rounds, reviews = _critic_loop(sess, field, draft, sess.traces[field],
                                                target=f"pass_b_{field}",
                                                registry=registry, duty=sess.duty("pass_b"))
        sess.metrics[f"pass_b_{field}_critic_rounds"] = n_rounds
        for i, r in enumerate(reviews):
            sess.write(f"pass_b_{field}_critic_r{i + 1}.json", r, "critic_objection.schema.json")
        sess.drafts[field] = draft
        sess.write(f"pass_b_{field}.md", draft)
        sess.write(f"pass_b_{field}_trace.json", {"field": field, "trace": sess.traces[field]})

    _cross_review(sess)
    sess.mark("pass_b", elapsed_s=round(time.time() - t0, 1))


def _cross_review(sess: ScopingSession) -> None:
    """Interface-scoped cross-review (ADR-025 §4.4): each expert attacks ONLY the interface
    assumptions of the other's plan; untagged/out-of-domain objections are dropped."""
    from src.scoping.critic import filter_cross_objections, material_objections, review_schema
    pairs = [("quant_trading", "ml_data_scientist", "cross_review_quant"),
             ("ml_data_scientist", "quant_trading", "cross_review_ml")]
    for reviewer, author, duty_key in pairs:
        if reviewer not in sess.drafts or author not in sess.drafts:
            continue
        try:
            review = sess.llm.chat_json(
                system=sess.persona(reviewer) + "\n\n" + sess.duty(duty_key),
                user=_with_schema(
                    _objective_block(sess)
                    + f"\n\n## THE {author} DRAFT UNDER INTERFACE REVIEW\n"
                    + _clip(sess.drafts[author])
                    + "\n\n## TASK\nOutput the CriticReview JSON (reviewer.role='cross_review'). "
                      "EVERY objection must set interface_assumption.",
                    review_schema()),
                schema=review_schema(), agent=f"{reviewer}_cross_review",
                temperature=0.2, max_tokens=16000)
        except Exception as exc:  # noqa: BLE001 — cross-review degrading is acceptable
            logger.warning("cross_review_failed reviewer=%s: %s", reviewer, exc)
            continue
        review.setdefault("reviewer", {})
        review["reviewer"].update({"role": "cross_review", "field": reviewer})
        kept, dropped = filter_cross_objections(review)
        sess.cross_objections[author] = kept
        sess.metrics[f"cross_objections_{author}"] = len(kept)
        sess.write(f"cross_review_of_{author}.json",
                   {**review, "objections": kept, "n_dropped_out_of_domain": dropped},
                   "critic_objection.schema.json")
        material = [o for o in kept if o.get("severity") in ("blocker", "major")]
        if material:
            revised = sess.llm.chat(
                system=sess.persona(author) + "\n\n" + sess.duty("pass_b"),
                user=("## YOUR CURRENT DRAFT\n" + _clip(sess.drafts[author])
                      + "\n\n## INTERFACE OBJECTIONS FROM THE OTHER EXPERT (address each)\n"
                      + json.dumps(material, indent=2)
                      + "\n\nRevise your draft to resolve or explicitly rebut each objection. "
                        "Output the full revised draft."),
                agent=f"{author}_cross_revision", temperature=0.3, max_tokens=_AUTHOR_MAX_TOKENS)
            if revised.strip():
                sess.drafts[author] = revised
                sess.write(f"pass_b_{author}.md", revised)


# ── stage: synthesis (judge + one replication → reconciled plan) ──────────────

_SYNTHESIS_MAX_TOKENS = 32000   # ScopingPlan JSON is large (8k truncated live); the JSON path
                                # streams, so this is a ceiling, not a spend — the judge emits
                                # what the plan needs and stops. Scoping is course-setting; do
                                # not cap it to save tokens. See anthropic_llm._JSON_MAX_TOKENS_CEILING.


def _model_catalog_digest() -> Any:
    """The families the search controller can actually BUILD today (src/models/catalog.py),
    injected at SYNTHESIS (2026-07-15 directive): Pass A/B gather evidence; the judge —
    exercising ML-data-science judgement — maps the evidence to concrete implementable model
    candidates/combos and marks anything not yet catalogued as build-required. Archetypes +
    coarse attributes only (no tuned params — that would anchor the plan)."""
    try:
        from src.models.catalog import get_catalog
        return [{"name": e.name, "display_name": e.display_name, "family_group": e.family_group,
                 "needs_labels": e.needs_labels, "needs_sequence": e.needs_sequence,
                 "compute_class": e.compute_class, "cost_class": e.cost_class,
                 "notes": e.notes} for e in get_catalog()]
    except Exception as exc:  # noqa: BLE001 — advisory; absence is itself information
        return {"error": str(exc)}


def _stage_synthesis(sess: ScopingSession) -> None:
    t0 = time.time()
    if sess.dry_run:
        sess.plan = _empty_plan(sess)
        sess.write("scoping_plan.json", sess.plan, "scoping_plan.schema.json")
        sess.mark("synthesis", elapsed_s=round(time.time() - t0, 1), replications=0)
        return

    schema = json.loads(schema_path("scoping_plan.schema.json").read_text())
    inputs = (_objective_block(sess)
              + "\n\n## EXPERT DRAFT — quant_trading\n" + _clip(sess.drafts.get("quant_trading", ""))
              + "\n\n## EXPERT DRAFT — ml_data_scientist\n" + _clip(sess.drafts.get("ml_data_scientist", ""))
              + "\n\n## SURVIVING CROSS-REVIEW OBJECTIONS\n"
              + json.dumps(sess.cross_objections, indent=2, default=str)[:4000]
              + "\n\n## IMPLEMENTED MODEL CATALOG (families the search controller can build today)\n"
              + json.dumps(_model_catalog_digest(), indent=2, default=str)[:4000])

    primary = sess.llm.chat_json(system=sess.duty("synthesis"), user=_with_schema(inputs, schema),
                                 schema=schema, agent="scoping_judge",
                                 temperature=0.2, max_tokens=_SYNTHESIS_MAX_TOKENS)
    n_reps = int(sess.cfg["caps"].get("synthesis_replications", 1))
    if n_reps > 0:
        try:
            replication = sess.critic_llm.chat_json(
                system=sess.duty("replication"), user=_with_schema(inputs, schema), schema=schema,
                agent="scoping_replication", temperature=0.3, max_tokens=_SYNTHESIS_MAX_TOKENS)
            sess.write("replication_plan.json", replication, "scoping_plan.schema.json")
            final = sess.llm.chat_json(
                system=sess.duty("reconciliation"),
                user=_with_schema(
                    "## PRIMARY PLAN\n" + json.dumps(primary, default=str)[:12000]
                    + "\n\n## INDEPENDENT REPLICATION\n" + json.dumps(replication, default=str)[:12000],
                    schema),
                schema=schema, agent="scoping_judge", temperature=0.2, max_tokens=_SYNTHESIS_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001 — replication is a second opinion, not a dependency
            logger.warning("synthesis_replication_failed: %s — keeping primary plan", exc)
            final = primary
            final.setdefault("reconciliation", f"replication failed ({exc}) — primary plan unreconciled")
    else:
        final = primary

    final["objective_id"] = sess.objective_id
    final.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    final["run_ts"] = sess.run_ts
    final.setdefault("provenance", {})
    final["provenance"].update({
        "author_provider": sess.author_provider,
        "critic_rounds": {k: v for k, v in sess.metrics.items() if k.endswith("_critic_rounds")},
        "cross_objections": {k: v for k, v in sess.metrics.items() if k.startswith("cross_objections")},
    })
    sess.plan = final
    sess.write("scoping_plan.json", final, "scoping_plan.schema.json")
    sess.mark("synthesis", elapsed_s=round(time.time() - t0, 1), replications=n_reps)


# ── stage: deterministic gap check v2 ─────────────────────────────────────────

def _stage_gap_check(sess: ScopingSession) -> None:
    t0 = time.time()
    from src.scoping.gap_check import gap_check_v2
    sess.gap_report = gap_check_v2(sess.objective, sess.plan, sess.out_dir)
    sess.write("data_gap_report.json", sess.gap_report, "data_gap_report.schema.json")
    sess.mark("gap_check", elapsed_s=round(time.time() - t0, 1),
              recommendation=sess.gap_report["summary"].get("recommendation"))


# ── stage: final report + machine-consumable outputs + write-back ─────────────

def _stage_final_report(sess: ScopingSession) -> None:
    t0 = time.time()
    manual = _draft_manual_candidates(sess.plan)
    sess.write("manual_candidates.draft.yaml", yaml.dump(manual, sort_keys=False))
    patch = _objective_patch(sess.plan, sess.gap_report)
    sess.write("objective_patch.suggested.yaml", yaml.dump(patch, sort_keys=False))
    sess.write("final_research_report.md", _render_report(sess, manual, patch))
    if not sess.dry_run:
        _write_back_library(sess)
    sess.mark("final_report", elapsed_s=round(time.time() - t0, 1))


# ── author round machinery ─────────────────────────────────────────────────────

def _author_round(sess: ScopingSession, field: str, registry: Any, *,
                  duty: str, user: str, agent: str) -> tuple[str, list[dict[str, Any]]]:
    """One bounded plan→act→observe loop; returns (draft, retrieval_trace).

    Guard: if the tool loop ends with an empty/near-empty draft (e.g. max_iters exhausted
    mid-tool-use), one plain follow-up chat materializes the draft from the same context.
    """
    system = sess.persona(field) + "\n\n" + duty
    draft, trace = sess.llm.chat_tools(
        system=system, user=user, agent=agent, registry=registry,
        max_iters=int(sess.cfg["caps"].get("max_tool_iters", 10)),
        temperature=0.3, max_tokens=_AUTHOR_MAX_TOKENS)
    if len(draft.strip()) < 200:
        logger.warning("author_draft_empty field=%s agent=%s — materializing without tools", field, agent)
        trace_txt = json.dumps(trace, default=str)[:8000]
        draft = sess.llm.chat(
            system=system,
            user=(user + "\n\n## YOUR RESEARCH TRACE (tool calls already made)\n" + trace_txt
                  + "\n\nProduce the FULL draft now from this research — no more tool use."),
            agent=agent, temperature=0.3, max_tokens=_AUTHOR_MAX_TOKENS) or draft
    return draft, trace


def _self_critique(sess: ScopingSession, field: str, draft: str, *, agent: str) -> str:
    """Author self-critique pre-pass (≤ cap): surface slips only — reply NO_REVISIONS to stop."""
    system = sess.persona(field)
    for _ in range(int(sess.cfg["caps"].get("self_critique", 3))):
        flaws = sess.llm.chat(
            system=system,
            user=("## YOUR DRAFT\n" + _clip(draft)
                  + "\n\nCritique your own draft: list up to 5 CONCRETE flaws (uncited claims, "
                    "internal contradictions, missing objective constraints). If none, reply "
                    f"exactly {_NO_REVISIONS}."),
            agent=agent, temperature=0.2, max_tokens=1024)
        if _NO_REVISIONS in flaws.upper() or not flaws.strip():
            break
        draft = sess.llm.chat(
            system=system,
            user=("## YOUR DRAFT\n" + _clip(draft) + "\n\n## YOUR OWN CRITIQUE\n" + flaws
                  + "\n\nOutput the full revised draft fixing these flaws."),
            agent=agent, temperature=0.3, max_tokens=_AUTHOR_MAX_TOKENS) or draft
    return draft


def _critic_loop(sess: ScopingSession, field: str, draft: str,
                 trace: list[dict[str, Any]], *, target: str,
                 registry: Any, duty: str,
                 objective_summary: str | None = None,
                 critic_registry: Any = None,
                 ) -> tuple[str, int, list[dict[str, Any]]]:
    """Parameterized-critic rounds (≤ cap): converge on no material objections.

    ``objective_summary`` / ``critic_registry`` default to this module's scoping context
    (``_objective_block`` / ``_critic_registry``); a sibling loop (e.g. the Research Director's
    model-improvement orchestrator) passes its own evidence block + verification registry so the
    convergence logic — round cap, tool-enabled re-draft, in-place trace accumulation — stays
    single-sourced here rather than duplicated.

    The between-round revision is TOOL-ENABLED (2026-07-15 directive): a critic objection
    that names an unexplored concept or an over-hedged (all-provisional) conclusion re-opens a
    bounded research round on the field's registry, so the author can ACQUIRE NEW EVIDENCE —
    not just re-word the existing draft. New retrieval steps are appended to ``trace`` IN PLACE
    so the next round's critic sees the added evidence and the persisted trace reflects the
    full (non-frozen) research. The caps.critic_rounds ceiling still bounds the loop; each
    round may now spend a tool loop (bounded by the per-agent/session USD caps)."""
    from src.scoping.critic import CriticHarness, material_objections
    if objective_summary is None:
        objective_summary = _objective_block(sess)
    if critic_registry is None:
        critic_registry = _critic_registry(sess)
    harness = CriticHarness(field, llm=sess.critic_llm, duties=sess.duties,
                            max_tool_iters=6)
    reviews: list[dict[str, Any]] = []
    rounds = 0
    system = sess.persona(field) + "\n\n" + duty
    for rounds in range(1, int(sess.cfg["caps"].get("critic_rounds", 4)) + 1):
        try:
            review = harness.review(_clip(draft), trace, target=f"{target} r{rounds}",
                                    objective_summary=objective_summary,
                                    registry=critic_registry)
        except Exception as exc:  # noqa: BLE001 — a failing critic never blocks the pass
            logger.warning("critic_round_failed field=%s round=%d: %s", field, rounds, exc)
            break
        reviews.append(review)
        material = material_objections(review)
        if review.get("verdict") == "no_material_objections" or not material:
            break
        # TOOL-ENABLED revision: re-enter research to close the gaps the critic named, then
        # re-draft. Fresh retrieval accumulates onto `trace` (not frozen).
        new_draft, rev_trace = sess.llm.chat_tools(
            system=system,
            user=("## YOUR DRAFT\n" + _clip(draft)
                  + "\n\n## CRITIC OBJECTIONS (blocker/major — address each)\n"
                  + json.dumps(material, indent=2, default=str)
                  + "\n\nYou have your research tools again. For any objection flagging an "
                    "UNEXPLORED concept or an OVER-HEDGED (all-provisional) conclusion, run "
                    "TARGETED searches to close that evidence gap, THEN revise. Commit to a "
                    "rationally chosen set; attach an explicit validation flag where a choice "
                    "is genuinely open rather than converting everything to hypotheticals. "
                    "Rebut with evidence what the critic got wrong. Output the FULL revised draft."),
            agent=f"{field}_revision", registry=registry,
            max_iters=int(sess.cfg["caps"].get("max_tool_iters", 10)),
            temperature=0.3, max_tokens=_AUTHOR_MAX_TOKENS)
        if new_draft.strip():
            draft = new_draft
        if rev_trace:
            trace.extend(rev_trace)   # ACCUMULATE — visible to next round's critic + trace file
    return draft, rounds, reviews


# ── tool registries ────────────────────────────────────────────────────────────

def _pass_a_registry(sess: ScopingSession) -> Any:
    """Pass A: blank slate — internal RESEARCH corpus + external web only (server tools),
    plus T1-scoped RAG. Deliberately NO T2/T3/inventory tools (anchoring guard)."""
    from src.research_director import tools as rd_tools
    from src.research_director.registry import ToolRegistry, _obj
    from src.scoping.agent import _tool_estimate_acquisition_cost, _tool_read_objective, _truncate

    internal = rd_tools.ResearchTool(external_enabled=False)
    reg = ToolRegistry()
    reg.add("read_objective", "The Trade Objective (products, keywords, KPI, horizon, labels).",
            _obj(), partial(_tool_read_objective, sess.objective))
    reg.add("internal_research", "Search the internal research corpus (research_cases/papers, docs/decisions).",
            _obj({"query": {"type": "string"}}, ["query"]),
            lambda query: _truncate(internal.search(query)))
    reg.add("read_reference", "Read one repo doc by relative path (truncated).",
            _obj({"rel_path": {"type": "string"}}, ["rel_path"]),
            lambda rel_path: _truncate(rd_tools.read_reference(rel_path)))
    reg.add("search_research",
            "Semantic search over the accumulated EXTERNAL research corpus (T1). "
            "Returns top-k chunks with source refs — cite these.",
            _obj({"query": {"type": "string"}, "k": {"type": "integer", "default": 8}}, ["query"]),
            partial(_tool_search_research, tier="t1"))
    reg.add("estimate_acquisition_cost",
            "Advisory cost/scale to acquire a data class. A human gates any acquisition.",
            _obj({"vendor": {"type": "string"}, "dataset": {"type": "string"},
                  "date_range": {"type": "string"}}, ["vendor", "dataset"]),
            _tool_estimate_acquisition_cost)
    return reg


def _pass_b_registry(sess: ScopingSession) -> Any:
    """Pass B: the objective + tiered-corpus RAG. The authoritative internal evidence — the
    T3 proven-success journal — is injected directly into the prompt (see `_stage_pass_b`);
    un-confirmed experimental adapters (results_index / leaderboard / MLflow) are deliberately
    NOT exposed here (2026-07-15 directive: a Scoping "champion" is a base model, not a
    finalized success, so it is not Pass-B evidence)."""
    from src.research_director.registry import ToolRegistry, _obj
    from src.scoping.agent import _tool_read_objective

    reg = ToolRegistry()
    reg.add("read_objective", "The Trade Objective (products, keywords, KPI, horizon, labels).",
            _obj(), partial(_tool_read_objective, sess.objective))
    reg.add("search_research", "Semantic search over the tiered research corpus (t1/t3 filter).",
            _obj({"query": {"type": "string"}, "tier": {"type": "string"},
                  "k": {"type": "integer", "default": 8}}, ["query"]),
            _tool_search_research)
    return reg


def _critic_registry(sess: ScopingSession) -> Any:
    """Critic verification tools: deterministic citation check + omission/counter-evidence search."""
    from src.research_director.registry import ToolRegistry, _obj
    reg = ToolRegistry()
    reg.add("citation_check", "DETERMINISTIC: does this cited source_ref exist in the corpus/index?",
            _obj({"source_ref": {"type": "string"}}, ["source_ref"]), _tool_citation_check)
    reg.add("search_research",
            "Semantic corpus search — use for omission probes and counter-evidence queries.",
            _obj({"query": {"type": "string"}, "tier": {"type": "string"},
                  "k": {"type": "integer", "default": 8}}, ["query"]),
            _tool_search_research)
    return reg


def _tool_search_research(query: str, tier: str | None = None, k: int = 8) -> Any:
    try:
        from src.agentic.rag import research_search
        return research_search(query, tier=tier, k=k, allow_substring_fallback=True)
    except Exception as exc:  # noqa: BLE001 — index absent = degraded, not fatal
        return {"error": f"RAG index unavailable: {exc}"}


def _tool_citation_check(source_ref: str) -> Any:
    try:
        from src.scoping.rag import citation_exists
        return citation_exists(source_ref)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"citation check unavailable: {exc}"}


# ── library write-back (T1 corpus compounding) ────────────────────────────────

def _harvest_library(sess: ScopingSession, field: str, draft: str) -> None:
    """Extract the publications an author actually used into research-library entries.

    The strict library schema demands ≥10 publications (a full-research quota); enforcing
    that on a per-field harvest would incentivize the model to PAD with fabricated
    citations — which would permanently poison the RAG corpus. The harvest therefore uses
    a relaxed copy (minItems 1) and an explicit no-padding instruction.
    """
    try:
        schema = json.loads(schema_path("research_library.schema.json").read_text())
        schema.get("properties", {}).get("publications", {})["minItems"] = 1   # anti-fabrication
        lib = sess.llm.chat_json(
            system=sess.persona(field),
            user=_with_schema(
                "## YOUR RESEARCH DRAFT\n" + _clip(draft)
                + "\n\n## TASK\nOutput a ScoredResearchLibrary JSON of the publications you "
                  "ACTUALLY consulted in this session — every source your draft cites, and "
                  "nothing else. Do NOT pad the list or invent entries to reach a count; "
                  "fabricated citations poison the research corpus. Real URLs/ids only. "
                  "Include the 4-axis 0-5 scores and data_feature_ideas. JSON only.",
                schema),
            schema=schema, agent=f"{field}_library", temperature=0.2, max_tokens=8000)
        pubs = lib.get("publications") or []
        for p in pubs:
            p.setdefault("origin", "pass_a")
            p.setdefault("added_at", datetime.now(timezone.utc).isoformat())
        sess.library_additions.extend(pubs)
        sess.write(f"research_library_{field}.json", lib, "research_library.schema.json")
    except Exception as exc:  # noqa: BLE001 — library harvest is additive, never blocking
        logger.warning("library_harvest_failed field=%s: %s", field, exc)


def _write_back_library(sess: ScopingSession) -> None:
    """Append new publications to the cumulative T1 library + index them (ADR-025 §5)."""
    if not sess.library_additions:
        return
    root = sess.out_dir.parents[1] / "_library"          # data/research/scoping/_library/
    root.mkdir(parents=True, exist_ok=True)
    lib_path = root / "research_library.json"
    lib: dict[str, Any] = {"objective_id": "_cumulative", "publications": []}
    if lib_path.exists():
        try:
            lib = json.loads(lib_path.read_text())
        except Exception:  # noqa: BLE001
            pass
    known = {p.get("id") for p in lib.get("publications") or []}
    new: list[dict[str, Any]] = []
    for p in sess.library_additions:                     # dedupe vs file AND within the batch
        pid = p.get("id")
        if pid and pid not in known:
            known.add(pid)
            new.append(p)
    if not new:
        return
    lib.setdefault("publications", []).extend(new)
    lib["generated_at"] = datetime.now(timezone.utc).isoformat()
    lib.setdefault("traits", {})
    lib_path.write_text(json.dumps(lib, indent=2, default=str))
    logger.info("library_write_back n_new=%d total=%d", len(new), len(lib["publications"]))
    try:
        from src.scoping.rag import _pub_text, index_document
        for p in new:
            index_document(p.get("id") or p.get("url", "?"), _pub_text(p), tier="t1",
                           objective_id=sess.objective_id)
            p["indexed"] = True
        lib_path.write_text(json.dumps(lib, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        logger.warning("library_index_failed: %s — run `python -m src.scoping.rag index`", exc)


# ── machine-consumable outputs + report ───────────────────────────────────────

def _draft_manual_candidates(plan: dict[str, Any]) -> dict[str, Any]:
    """ScopingPlan → draft ManualCandidatesSpec for the search-controller handoff."""
    mp = plan.get("model_plan") or {}
    cands = []
    for c in mp.get("candidates") or []:
        entry: dict[str, Any] = {"family": c.get("family")}
        if c.get("id") and c["id"] != c.get("family"):
            entry["id"] = c["id"]
        if c.get("params_hint"):
            entry["params"] = c["params_hint"]
        if c.get("features"):
            entry["features"] = c["features"]
        cands.append(entry)
    return {"schema_version": "v1", "objective_id": plan.get("objective_id"),
            "ensembles": bool(mp.get("ensembles")), "candidates": cands}


def _objective_patch(plan: dict[str, Any], gap: dict[str, Any]) -> dict[str, Any]:
    """Suggested objective-doc patch: pin runnable families + per-family feature views."""
    runnable = {m["family"] for m in gap.get("model_coverage") or []
                if m.get("status") == "have" and m.get("role") != "ensemble"}
    views = {c["family"]: c["features"]["view"]
             for c in ((plan.get("model_plan") or {}).get("candidates")) or []
             if c.get("family") in runnable and (c.get("features") or {}).get("view")}
    patch: dict[str, Any] = {"eligible_model_families": sorted(runnable)}
    if views:
        patch["search"] = {"feature_views": views}
    return patch


def _render_report(sess: ScopingSession, manual: dict[str, Any],
                   patch: dict[str, Any]) -> str:
    plan, gap = sess.plan, sess.gap_report
    s = gap.get("summary", {})
    mp = plan.get("model_plan") or {}
    lines = [
        f"# Final Research Report — {sess.objective_id}",
        f"*Scoping v2 session `{sess.run_ts}` (ADR-025){' — DRY RUN' if sess.dry_run else ''}*",
        "",
        f"**Gate recommendation (advisory): `{s.get('recommendation', 'n/a').upper()}`** — the human decides.",
        "",
        "## Proposed starting point (hypotheses for the search controller — not verdicts)",
        "",
        "| candidate | family | role | features | rationale |",
        "|---|---|---|---|---|",
    ]
    for c in mp.get("candidates") or []:
        feats = json.dumps(c.get("features")) if c.get("features") else "full matrix"
        lines.append(f"| {c.get('id') or c.get('family')} | {c.get('family')} | {c.get('role')} "
                     f"| {feats} | {_one_line(c.get('rationale', ''))} |")
    if not mp.get("candidates"):
        lines.append("| *(none — degenerate plan; see gap report)* | | | | |")
    for e in mp.get("ensembles") or []:
        lines.append(f"| {e.get('kind')}({', '.join(e.get('members') or [])}) | ensemble | combo "
                     f"| — | {_one_line(e.get('rationale', ''))} |")
    lines += ["", "### Evidence trail (tier-tagged: T3 proven-live > T2 internal > T1 external)", ""]
    for c in mp.get("candidates") or []:
        for ev in c.get("evidence") or []:
            lines.append(f"- **{c.get('id') or c.get('family')}** ← [{ev.get('tier')}] "
                         f"`{ev.get('source_ref')}` {ev.get('note', '')}")
    kpi = plan.get("kpi_sanity") or {}
    if kpi:
        lines += ["", "## KPI sanity", "",
                  f"{kpi.get('assessment', '')}",
                  f"- **Binding constraint**: {kpi.get('binding_constraint', 'n/a')}"]
        lines += [f"- Risk: {r}" for r in kpi.get("risks") or []]
    lines += ["", "## Gaps (criticality-ranked)", "",
              "| item | status | required | criticality | note |", "|---|---|---|---|---|"]
    for it in gap.get("items") or []:
        lines.append(f"| {it.get('data_class')} | {it.get('status')} | {it.get('required')} "
                     f"| {it.get('criticality', '-')} | {_one_line(it.get('gap') or '')} |")
    for m in gap.get("model_coverage") or []:
        if m.get("status") != "have":
            lines.append(f"| model:{m.get('candidate')} | {m.get('status')} | — "
                         f"| {m.get('criticality', '-')} | {_one_line(m.get('note', ''))} |")
    for f in gap.get("feature_coverage") or []:
        if f.get("status") != "have":
            lines.append(f"| features:{f.get('candidate')} {f.get('selector')} | {f.get('status')} | — "
                         f"| {f.get('criticality', '-')} | {_one_line(f.get('note', ''))} |")
    if s.get("total_est_cost_usd"):
        lines += ["", f"**Estimated acquisition cost:** ${s['total_est_cost_usd']:.2f}"]
    oq = plan.get("open_questions") or []
    if oq:
        lines += ["", "## Open questions for the operator", ""] + [f"- {q}" for q in oq]
    if plan.get("reconciliation"):
        lines += ["", "## Synthesis reconciliation (primary vs independent replication)", "",
                  str(plan["reconciliation"])]
    lines += [
        "", "## Next actions (HUMAN GATE)", "",
        "1. Review the gaps above; acquire data via the operator flow if `ACQUIRE` "
        "(cost check is mandatory before any download).",
        "2. To adopt the starting point: copy `manual_candidates.draft.yaml` → "
        "`config/manual_candidates.yaml` (review first) and run the search controller with it.",
        "3. Optional objective patch (eligible families / feature views): see "
        "`objective_patch.suggested.yaml`.",
        "", "```yaml", "# manual_candidates.draft.yaml",
        yaml.dump(manual, sort_keys=False).rstrip(), "```",
        "", "```yaml", "# objective_patch.suggested.yaml",
        yaml.dump(patch, sort_keys=False).rstrip(), "```",
        "", "---",
        f"*Provenance: author={sess.author_provider}; "
        f"metrics={json.dumps(sess.metrics, default=str)}; artifacts in `{sess.out_dir}`.*",
    ]
    return "\n".join(lines) + "\n"


# ── observability ─────────────────────────────────────────────────────────────

def _log_to_mlflow(sess: ScopingSession) -> None:
    """Best-effort MLflow lineage: experiment scoping/<objective_id>, parent run per session,
    nested run per stage, all artifacts attached (ADR-024 'verify everything in one UI')."""
    try:
        import mlflow
        from src.training.config import MLflowConfig
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLflowConfig().tracking_uri))
        mlflow.set_experiment(f"scoping/{sess.objective_id}")
        with mlflow.start_run(run_name=f"scoping_{sess.run_ts}"):
            mlflow.set_tags({"objective_id": sess.objective_id, "run_ts": sess.run_ts,
                             "experiment_type": "scoping", "dry_run": str(sess.dry_run),
                             "author_provider": sess.author_provider})
            caps = sess.cfg.get("caps") or {}
            mlflow.log_params({f"cap_{k}": v for k, v in caps.items()})
            if sess.metrics:
                mlflow.log_metrics({k: float(v) for k, v in sess.metrics.items()})
            gate = (sess.gap_report.get("summary") or {}).get("recommendation")
            if gate:
                mlflow.set_tag("gate_recommendation", gate)
            for stage, info in sess.state.get("stages", {}).items():
                with mlflow.start_run(run_name=stage, nested=True):
                    mlflow.set_tags({"objective_id": sess.objective_id, "stage": stage,
                                     "run_ts": sess.run_ts})
                    if isinstance(info.get("elapsed_s"), (int, float)):
                        mlflow.log_metric("elapsed_s", float(info["elapsed_s"]))
            mlflow.log_artifacts(str(sess.out_dir))
        logger.info("mlflow_scoping_logged experiment=scoping/%s", sess.objective_id)
    except Exception as exc:  # noqa: BLE001 — observability must never fail the run
        logger.warning("mlflow_scoping_log_failed: %s", exc)


# ── misc helpers ──────────────────────────────────────────────────────────────

def _objective_block(sess: ScopingSession) -> str:
    return "## TRADE OBJECTIVE\n```yaml\n" + yaml.dump(sess.objective, sort_keys=False) + "```"


def _empty_plan(sess: ScopingSession) -> dict[str, Any]:
    return {"objective_id": sess.objective_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_ts": sess.run_ts,
            "model_plan": {"candidates": [], "ensembles": []},
            "data_requirements": {"data_classes": []},
            "open_questions": ["dry run — no research performed"],
            "reconciliation": None}


def _with_schema(user: str, schema: dict[str, Any]) -> str:
    """Append the target JSON schema to a chat_json prompt — the clients validate locally
    but never transmit the schema, so without this the model must GUESS the field structure
    (the top live-run failure mode found in pre-flight review)."""
    return (user + "\n\n## OUTPUT JSON SCHEMA (conform EXACTLY — required fields, enums, nesting)\n"
            + "```json\n" + json.dumps(schema, indent=1) + "\n```")


def _clip(text: str, limit: int = _DRAFT_CLIP) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[clipped]"


def _one_line(s: str, limit: int = 140) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= limit else s[:limit] + "…"


def _print_summary(sess: ScopingSession) -> None:
    rec = (sess.gap_report.get("summary") or {}).get("recommendation", "n/a")
    n_c = len(((sess.plan.get("model_plan") or {}).get("candidates")) or [])
    print(f"\n{'─' * 62}\n  Scoping v2 complete  {sess.run_ts}"
          f"\n  Candidates proposed : {n_c}"
          f"\n  Gate recommendation : {rec.upper()}  (advisory — human decides)"
          f"\n  Report              : {sess.out_dir / 'final_research_report.md'}"
          f"\n{'─' * 62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Scoping v2 — dual-perspective agentic research loop (ADR-025)")
    p.add_argument("--objective", required=True)
    p.add_argument("--dry-run", action="store_true", help="No LLM: exercise the state machine only")
    p.add_argument("--budget", type=float, default=None, help="Override usd_cap_session")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg = load_scoping_cfg()
    if args.budget is not None:
        cfg.setdefault("cost", {})["usd_cap_session"] = args.budget
    run_scoping_v2(args.objective, dry_run=args.dry_run, cfg=cfg)


if __name__ == "__main__":
    _cli()
