"""Research Director orchestrator — deep model-improvement loop (sibling of Scoping v2).

The RD's job is NOT to scope a new objective or pick a champion family. It is to IMPROVE AN
EXISTING model: two field-expert agents (quant_trading, ml_data_scientist) study the existing
model and the **patterns in its results** (the Context Pack — confusion matrix, per-class F1,
precision@τ, feature-group importance, integrated gradients, mined win/loss/missed outcome
patterns, distance-to-target, data-capability catalog) and use their expertise to identify the
**single highest-leverage change** that will optimize it — of arbitrary scope (a feature edit, a
new correlated symbol, an overlay/auxiliary model run in conjunction, a loss/penalty change, a
label redesign, an ensemble). A cross-model **critic** forces the agents to think deeper.

Stage graph (deterministic; agents reason only inside bounded author→self-critique→critic rounds,
reusing the Scoping-v2 primitives verbatim):

    diagnose     per-lens Diagnose+Prescribe (author round → self-critique → critic rounds)
      → cross_review   interface-scoped dual-perspective critique of each prescription
      → synthesis      judge merges → one ranked ModelImprovementPlan (+ replication + reconcile)
      → validate       AUTO-VALIDATE the top pick: train+score it vs the baseline (GPU→CPU fallback)
      → handoff        report + journal + RAG write-back; changes needing new data/code hand off

Artifacts to ``data/research_director/<program_id>/<run_ts>/``; schema-validated; MLflow lineage.
"""

from __future__ import annotations

from src.common.data_paths import data_root, model_registry_dir, native_tf_dir, research_director_dir

import hashlib
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

# Reused Scoping-v2 loop primitives (LLM-agnostic — they read only the duck-typed session surface).
from src.scoping.orchestrator import (
    _author_round, _self_critique, _critic_loop, _harvest_library, _write_back_library,
    _with_schema, _clip, _one_line, _model_catalog_digest,
)
from src.scoping.critic import (
    filter_cross_objections, review_schema, resolve_critic_provider,
)

logger = logging.getLogger("vtech.src.research_director.orchestrator")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "config"

STAGES = ("diagnose", "cross_review", "synthesis", "validate", "handoff")
_LENS_KEY = {"quant_trading": "quant", "ml_data_scientist": "ml"}
_AUTHOR_MAX_TOKENS = 8192
_SYNTHESIS_MAX_TOKENS = 32000
_PLAN_SCHEMA = "model_improvement_plan.schema.json"


def load_rd_cfg() -> dict[str, Any]:
    """The RD loop config: ``config/research_director.yaml`` ``rd_loop:`` block + defaults
    (mirrors ``scoping.load_scoping_cfg``)."""
    cfg: dict[str, Any] = {}
    try:
        rd = yaml.safe_load(runtime_path("research_director.yaml").read_text()) or {}
        cfg = dict(rd.get("rd_loop") or {})
        # Execution knobs live in the top-level `execution:`/`objective_id` — surface them here.
        cfg.setdefault("parquet_dir", (rd.get("execution") or {}).get("parquet_dir", str(native_tf_dir("5s"))))
        cfg.setdefault("tune_trials", int((rd.get("execution") or {}).get("tune_trials", 0)))
        cfg.setdefault("delta_min", float(rd.get("delta_min", 0.0)))
        cfg.setdefault("class_prior", float(rd.get("class_prior", 0.09)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("rd_cfg_unreadable: %s — using defaults", exc)
    caps = cfg.setdefault("caps", {})
    caps.setdefault("self_critique", 3)
    caps.setdefault("critic_rounds", 4)
    caps.setdefault("synthesis_replications", 1)
    caps.setdefault("max_tool_iters", 10)
    cfg.setdefault("fields", ["quant_trading", "ml_data_scientist"])
    cfg.setdefault("providers", {"author": "auto", "critic": "other"})
    cfg.setdefault("cost", {"usd_cap_session": 20.0, "per_agent_usd_cap": 5.0})
    return cfg


def load_rd_duties() -> dict[str, Any]:
    try:
        return yaml.safe_load((_CONFIG / "agents" / "rd_duties.yaml").read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("rd_duties_unreadable: %s", exc)
        return {"duties": {}, "red_flags": {}}


# ── session ───────────────────────────────────────────────────────────────────

class RDSession:
    """One RD cycle's state — duck-types the surface the reused Scoping loop functions read
    (``.persona/.llm/.critic_llm/.cfg["caps"]/.write/.dry_run/.duties/.drafts/.traces/
    .cross_objections/.metrics/.library_additions/.out_dir``)."""

    def __init__(self, program_id: str, objective_id: str, pack: Any, *,
                 dry_run: bool = False, llm: Any = None, critic_llm: Any = None,
                 recent_signatures: list[str] | None = None,
                 out_root: Path | None = None, cfg: dict[str, Any] | None = None) -> None:
        self.program_id = program_id
        self.objective_id = objective_id
        self.pack = pack
        self.evidence = pack.render() if pack is not None else "(no context pack)"
        self.baseline_tr = ((getattr(pack, "baseline", None) or {}).get("training_result")) or {}
        self.recent_signatures = recent_signatures or []
        self.cfg = cfg or load_rd_cfg()
        self.dry_run = dry_run
        self.run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        root = out_root or research_director_dir()
        self.out_dir = root / program_id / self.run_ts
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.duties = load_rd_duties()
        self.llm = llm
        self.critic_llm = critic_llm
        self.author_provider = "none"
        if not dry_run and self.llm is None:
            self.llm, self.critic_llm, self.author_provider = _make_llms(self.cfg)
        elif self.llm is not None:
            self.author_provider = type(self.llm).__name__
            self.critic_llm = critic_llm or self.llm

        self.state: dict[str, Any] = {"program_id": program_id, "objective_id": objective_id,
                                      "run_ts": self.run_ts, "dry_run": dry_run, "stages": {}}
        self.drafts: dict[str, str] = {}
        self.traces: dict[str, list[dict[str, Any]]] = {}
        self.cross_objections: dict[str, list[dict[str, Any]]] = {}
        self.library_additions: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.validation: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    def duty(self, key: str) -> str:
        return (self.duties.get("duties") or {}).get(key, "")

    def persona(self, field: str) -> str:
        from src.scoping.critic import field_persona
        return field_persona(field)

    def write(self, name: str, content: str | dict[str, Any], schema_name: str | None = None) -> Path:
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


def _make_llms(cfg: dict[str, Any]) -> tuple[Any, Any, str]:
    """Wire author + cross-model critic LLMs from the loop cfg (mirrors ScopingSession._make_llms)."""
    from src.common.llm import make_llm
    cost = cfg.get("cost") or {}
    cap, per = float(cost.get("usd_cap_session", 20.0)), float(cost.get("per_agent_usd_cap", 5.0))
    rates = cost.get("model_rates") or {}
    providers = cfg.get("providers") or {}
    prov = providers.get("author", "auto")
    author = make_llm(usd_cap_cycle=cap, per_agent_usd_cap=per, model_rates=rates,
                      provider=None if prov == "auto" else prov)
    author_provider = "anthropic" if type(author).__name__ == "AnthropicLLM" else "openai"
    critic_prov = resolve_critic_provider(author_provider, providers.get("critic", "other"))
    critic_model = (providers.get("critic_model") or "").strip() or None
    author_model = getattr(author, "_model", None)
    same = critic_prov in (author_provider, None) and (critic_model is None or critic_model == author_model)
    if same:
        return author, author, author_provider
    kw: dict[str, Any] = {"usd_cap_cycle": cap, "per_agent_usd_cap": per, "model_rates": rates,
                          "provider": critic_prov}
    if critic_model:
        kw["model"] = critic_model
    if critic_prov == "openai":
        key_env = providers.get("critic_api_key_env", "OPENAI_API_KEY")
        base_env = providers.get("critic_base_url_env", "OPENAI_BASE_URL")
        kw["api_key"] = os.environ.get(key_env)
        kw["base_url"] = os.environ.get(base_env) or "https://api.openai.com/v1"
        if providers.get("critic_reasoning_effort"):
            kw["reasoning_effort"] = providers["critic_reasoning_effort"]
        kw["web_tools_enabled"] = bool(providers.get("critic_web", False))
    try:
        if critic_prov == "openai" and not kw.get("api_key"):
            raise ValueError(f"critic endpoint key missing — set "
                             f"{providers.get('critic_api_key_env', 'OPENAI_API_KEY')} in .env")
        critic = make_llm(**kw)
    except Exception as exc:  # noqa: BLE001 — degrade to same-model, never block
        logger.warning("critic_llm_unavailable (%s) — falling back to author client "
                       "(decorrelation lost this cycle)", exc)
        critic = author
    logger.info("rd_llms author=%s(%s) critic=%s(%s)", author_provider, author_model,
                critic_prov or author_provider, getattr(critic, "_model", "?"))
    return author, critic, author_provider


# ── public entry point ─────────────────────────────────────────────────────────

def run_rd_loop(program_id: str, objective_id: str, pack: Any, *,
                recent_signatures: list[str] | None = None, dry_run: bool = False,
                llm: Any = None, critic_llm: Any = None,
                cfg: dict[str, Any] | None = None, out_root: Path | None = None) -> dict[str, Any]:
    """Run the full RD model-improvement loop; returns the final state dict (incl. `plan` +
    `validation`). The caller (director) records the outcome to the journal."""
    sess = RDSession(program_id, objective_id, pack, dry_run=dry_run, llm=llm, critic_llm=critic_llm,
                     recent_signatures=recent_signatures, out_root=out_root, cfg=cfg)
    logger.info("rd_loop_start program=%s objective=%s run_ts=%s dry_run=%s out=%s",
                program_id, objective_id, sess.run_ts, dry_run, sess.out_dir)
    t0 = time.time()
    stage_fns = (("diagnose", _stage_diagnose), ("cross_review", _stage_cross_review),
                 ("synthesis", _stage_synthesis), ("validate", _stage_validate),
                 ("handoff", _stage_handoff))
    try:
        for _name, fn in stage_fns:
            fn(sess)
    except Exception as exc:
        failed = next((n for n, _ in stage_fns if n not in sess.state["stages"]), "?")
        sess.state["stages"][failed] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                        "at": datetime.now(timezone.utc).isoformat()}
        sess.state["elapsed_s"] = round(time.time() - t0, 1)
        _stamp_cost(sess)
        sess.write("state.json", sess.state)
        _log_to_mlflow(sess)
        logger.error("rd_loop_failed stage=%s: %s", failed, exc)
        raise
    sess.state["elapsed_s"] = round(time.time() - t0, 1)
    sess.state["plan"] = sess.plan
    sess.state["validation"] = sess.validation
    _stamp_cost(sess)
    sess.write("state.json", sess.state)
    _log_to_mlflow(sess)
    _print_summary(sess)
    return sess.state


def _stamp_cost(sess: RDSession) -> None:
    if sess.llm is not None and hasattr(sess.llm, "cost_tally"):
        try:
            sess.state["cost"] = sess.llm.cost_tally()
        except Exception:  # noqa: BLE001
            pass


# ── stage 1: diagnose + prescribe (per lens) ────────────────────────────────────

def _stage_diagnose(sess: RDSession) -> None:
    t0 = time.time()
    for field in sess.cfg["fields"]:
        if sess.dry_run:
            draft, trace = f"(dry run) diagnosis+prescription — {field}", []
        else:
            registry = _diagnose_registry(sess)
            duty = sess.duty(f"diagnose_prescribe_{_LENS_KEY.get(field, field)}")
            recall = _recall(sess, field)
            user = _evidence_block(sess) + recall + "\n\n" + duty
            draft, trace = _author_round(sess, field, registry, duty=duty, user=user,
                                         agent=f"{field}_diagnose")
            draft = _self_critique(sess, field, draft, agent=f"{field}_diagnose")
            draft, n_rounds, reviews = _critic_loop(
                sess, field, draft, trace, target=f"diagnose_{field}",
                registry=registry, duty=duty,
                objective_summary=sess.evidence,
                critic_registry=_rd_critic_registry(sess))
            sess.metrics[f"diagnose_{field}_critic_rounds"] = n_rounds
            for i, r in enumerate(reviews):
                sess.write(f"diagnose_{field}_critic_r{i + 1}.json", r, "critic_objection.schema.json")
            _harvest_library(sess, field, draft)
        sess.drafts[field] = draft
        sess.traces[field] = trace
        sess.write(f"diagnose_{field}.md", draft)
        sess.write(f"diagnose_{field}_trace.json", {"field": field, "trace": trace})
    sess.mark("diagnose", elapsed_s=round(time.time() - t0, 1))


# ── stage 2: interface-scoped dual-perspective cross-review ──────────────────────

def _stage_cross_review(sess: RDSession) -> None:
    t0 = time.time()
    if sess.dry_run:
        sess.mark("cross_review", elapsed_s=round(time.time() - t0, 1), status="dry_run")
        return
    pairs = [("quant_trading", "ml_data_scientist", "cross_review_quant"),
             ("ml_data_scientist", "quant_trading", "cross_review_ml")]
    for reviewer, author, duty_key in pairs:
        if reviewer not in sess.drafts or author not in sess.drafts:
            continue
        try:
            review = sess.llm.chat_json(
                system=sess.persona(reviewer) + "\n\n" + sess.duty(duty_key),
                user=_with_schema(
                    f"## THE {author} PRESCRIPTION UNDER INTERFACE REVIEW\n"
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
                system=sess.persona(author) + "\n\n"
                       + sess.duty(f"diagnose_prescribe_{_LENS_KEY.get(author, author)}"),
                user=("## YOUR CURRENT DIAGNOSIS+PRESCRIPTION\n" + _clip(sess.drafts[author])
                      + "\n\n## INTERFACE OBJECTIONS FROM THE OTHER EXPERT (address each)\n"
                      + json.dumps(material, indent=2)
                      + "\n\nRevise to resolve or explicitly rebut each objection. Output the full "
                        "revised draft."),
                agent=f"{author}_cross_revision", temperature=0.3, max_tokens=_AUTHOR_MAX_TOKENS)
            if revised.strip():
                sess.drafts[author] = revised
                sess.write(f"diagnose_{author}.md", revised)
    sess.mark("cross_review", elapsed_s=round(time.time() - t0, 1))


# ── stage 3: synthesis → one ranked ModelImprovementPlan ─────────────────────────

def _stage_synthesis(sess: RDSession) -> None:
    t0 = time.time()
    if sess.dry_run:
        sess.plan = _empty_plan(sess)
        sess.write("model_improvement_plan.json", sess.plan, _PLAN_SCHEMA)
        sess.mark("synthesis", elapsed_s=round(time.time() - t0, 1), replications=0)
        return

    schema = json.loads(schema_path(_PLAN_SCHEMA).read_text())
    inputs = (_evidence_block(sess)
              + "\n\n## EXPERT DIAGNOSIS+PRESCRIPTION — quant_trading\n"
              + _clip(sess.drafts.get("quant_trading", ""))
              + "\n\n## EXPERT DIAGNOSIS+PRESCRIPTION — ml_data_scientist\n"
              + _clip(sess.drafts.get("ml_data_scientist", ""))
              + "\n\n## SURVIVING CROSS-REVIEW OBJECTIONS\n"
              + json.dumps(sess.cross_objections, indent=2, default=str)[:4000]
              + "\n\n## IMPLEMENTED MODEL CATALOG (families the search controller can build today)\n"
              + json.dumps(_model_catalog_digest(), indent=2, default=str)[:4000])

    primary = sess.llm.chat_json(system=sess.duty("synthesis"), user=_with_schema(inputs, schema),
                                 schema=schema, agent="rd_judge",
                                 temperature=0.2, max_tokens=_SYNTHESIS_MAX_TOKENS)
    n_reps = int(sess.cfg["caps"].get("synthesis_replications", 1))
    final = primary
    if n_reps > 0:
        try:
            replication = sess.critic_llm.chat_json(
                system=sess.duty("replication"), user=_with_schema(inputs, schema), schema=schema,
                agent="rd_replication", temperature=0.3, max_tokens=_SYNTHESIS_MAX_TOKENS)
            sess.write("replication_plan.json", replication, _PLAN_SCHEMA)
            final = sess.llm.chat_json(
                system=sess.duty("reconciliation"),
                user=_with_schema(
                    "## PRIMARY PLAN\n" + json.dumps(primary, default=str)[:12000]
                    + "\n\n## INDEPENDENT REPLICATION\n" + json.dumps(replication, default=str)[:12000],
                    schema),
                schema=schema, agent="rd_judge", temperature=0.2, max_tokens=_SYNTHESIS_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001 — replication is a second opinion, not a dependency
            logger.warning("synthesis_replication_failed: %s — keeping primary plan", exc)
            final = primary
            final.setdefault("reconciliation", f"replication failed ({exc}) — primary unreconciled")

    final["objective_id"] = sess.objective_id
    final["program_id"] = sess.program_id
    final.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    final["run_ts"] = sess.run_ts
    final.setdefault("provenance", {})
    final["provenance"].update({
        "author_provider": sess.author_provider,
        "critic_rounds": {k: v for k, v in sess.metrics.items() if k.endswith("_critic_rounds")},
        "cross_objections": {k: v for k, v in sess.metrics.items() if k.startswith("cross_objections")},
    })
    sess.plan = final
    sess.write("model_improvement_plan.json", final, _PLAN_SCHEMA)
    sess.mark("synthesis", elapsed_s=round(time.time() - t0, 1), replications=n_reps,
              n_changes=len(final.get("changes") or []))


# ── stage 4: auto-validate the top recommendation ───────────────────────────────

def _normalize_feature_selection(features: Any) -> dict[str, Any] | None:
    """Coerce LLM-emitted feature picks into manual_candidates.schema.json shape.

    Agents often emit invalid extras (``note``, ``context_view``, ``meta_inputs``) or put
    group prefixes inside ``columns``. Manual-spec validation is strict oneOf
    (view | prefixes | columns) with ``additionalProperties: false`` — without this
    sanitizer every auto-validate cycle fails before training.
    """
    if not features or not isinstance(features, dict):
        return None

    view = features.get("view")
    if isinstance(view, str) and view.strip():
        # Prefer named view when present; drop any sibling keys the LLM invented.
        return {"view": view.strip()}

    prefixes_raw = features.get("prefixes")
    if isinstance(prefixes_raw, list):
        prefixes = [str(p).strip() for p in prefixes_raw if str(p).strip()]
        if prefixes:
            return {"prefixes": prefixes}

    columns_raw = features.get("columns")
    if isinstance(columns_raw, list):
        cols = [str(c).strip() for c in columns_raw if str(c).strip()]
        if not cols:
            return None
        # Group prefixes conventionally end with "_" (vw_, rv_, of_). Exact matrix
        # column names do not. Convert pure-prefix lists; drop prefix-like noise from
        # mixed lists so columns stay exact-name matches.
        prefix_like = [c for c in cols if c.endswith("_")]
        exact = [c for c in cols if not c.endswith("_")]
        if prefix_like and not exact:
            return {"prefixes": prefix_like}
        if exact:
            return {"columns": exact}
        return {"prefixes": prefix_like} if prefix_like else None

    # Some drafts put prefix lists under invented keys — recover when possible.
    for alt in ("meta_context_prefixes", "context_prefixes", "feature_prefixes"):
        alt_raw = features.get(alt)
        if isinstance(alt_raw, list):
            prefixes = [str(p).strip() for p in alt_raw if str(p).strip()]
            if prefixes:
                return {"prefixes": prefixes}
    return None


def _normalize_executable_candidate(cand: dict[str, Any]) -> dict[str, Any]:
    """Return a manual-spec-safe candidate (family/params/features/id/max_rows only)."""
    out: dict[str, Any] = {"family": cand["family"]}
    if cand.get("id"):
        out["id"] = cand["id"]
    if isinstance(cand.get("params"), dict):
        out["params"] = cand["params"]
    if "max_rows" in cand:
        out["max_rows"] = cand["max_rows"]
    if "ensembles" in cand:
        out["ensembles"] = bool(cand["ensembles"])
    feats = _normalize_feature_selection(cand.get("features"))
    if feats is not None:
        out["features"] = feats
    return out


def _stage_validate(sess: RDSession) -> None:
    t0 = time.time()
    top = _top_change(sess.plan)

    if (sess.dry_run or sess.cfg.get("skip_validate") or top is None
            or not top.get("auto_validatable") or not top.get("executable_candidate")):
        reason = ("dry_run" if sess.dry_run else
                  "skip_validate (operator flag)" if sess.cfg.get("skip_validate") else
                  "no top recommendation" if top is None else
                  "top recommendation is handoff-required (new data / new model code)")
        sess.validation = {"status": "skipped", "verdict": "HANDOFF_REQUIRED", "reason": reason,
                           "top_id": (top or {}).get("id"),
                           "validation_method": (top or {}).get("validation_method")}
        sess.mark("validate", status="skipped", verdict="HANDOFF_REQUIRED", reason=reason)
        return

    raw_cand = dict(top["executable_candidate"])
    cand = _normalize_executable_candidate(raw_cand)
    cand.setdefault("id", top["id"])
    if raw_cand.get("features") and cand.get("features") != raw_cand.get("features"):
        logger.info("rd_features_normalized top=%s from=%s to=%s",
                    top.get("id"), raw_cand.get("features"), cand.get("features"))
    elif raw_cand.get("features") and "features" not in cand:
        logger.warning("rd_features_dropped top=%s raw=%s — using full matrix",
                       top.get("id"), raw_cand.get("features"))
    ensembles = bool(cand.pop("ensembles", False))
    manual_spec = {"schema_version": "v1", "objective_id": sess.objective_id,
                   "ensembles": ensembles, "candidates": [cand]}
    sess.write("validation_manual_spec.json", manual_spec)
    sess.write("validation_manual_spec_raw.json",
               {"schema_version": "v1", "objective_id": sess.objective_id,
                "ensembles": ensembles, "candidates": [raw_cand]})

    try:
        from src.search.controller import run_search
        summary = run_search(sess.objective_id, parquet_dir=sess.cfg.get("parquet_dir"),
                             manual_spec=manual_spec, ensembles=ensembles,
                             tune_trials=int(sess.cfg.get("tune_trials", 0)))
    except Exception as exc:  # noqa: BLE001 — training/controller failure → inconclusive, never fatal
        sess.validation = {"status": "error", "verdict": "INCONCLUSIVE", "reason": str(exc),
                           "top_id": top["id"]}
        sess.mark("validate", status="error", verdict="INCONCLUSIVE", reason=str(exc)[:300])
        return

    row = _find_candidate_row(summary, cand)
    run_metrics = _run_metrics_from_candidate(row)
    metric = (top.get("expected_effect") or {}).get("metric") or "precision_at_tau"

    from src.research_director.evaluator import Evaluator
    ev = Evaluator()
    inp = ev.build_input(run_metrics, sess.baseline_tr or None, metric=metric,
                         delta_min=float(sess.cfg.get("delta_min", 0.0)),
                         class_prior=float(sess.cfg.get("class_prior", 0.09)))
    v = ev.evaluate(inp)
    verdict = {"PASS": "VALIDATED_IMPROVED", "FAIL": "VALIDATED_NO_IMPROVEMENT",
               "INCONCLUSIVE": "INCONCLUSIVE"}[v.gate]
    sess.validation = {"status": "done", "verdict": verdict, "gate": v.gate,
                       "metric": metric, "primary_delta": v.primary_delta, "reason": v.reason,
                       "run_primary": inp.run_primary, "baseline_primary": inp.baseline_primary,
                       "n_folds": v.n_folds, "top_id": top["id"],
                       "search_run_ts": summary.get("run_ts"), "champion": summary.get("champion")}
    sess.write("validation_result.json", sess.validation)
    sess.mark("validate", verdict=verdict, primary_delta=v.primary_delta, metric=metric,
              elapsed_s=round(time.time() - t0, 1))


# ── stage 5: handoff — report + RAG write-back ───────────────────────────────────

def _stage_handoff(sess: RDSession) -> None:
    t0 = time.time()
    sess.write("improvement_report.md", _render_report(sess))
    if not sess.dry_run:
        _write_back_library(sess)
        _remember(sess)
    sess.mark("handoff", elapsed_s=round(time.time() - t0, 1))


# ── registries + evidence ────────────────────────────────────────────────────────

def _diagnose_registry(sess: RDSession) -> Any:
    """Deep-research toolset for diagnosing the existing model.

    Uses the shared agentic registry builder: context tools + unified
    ``research_search`` / ``search_research`` (pgvector primary).
    """
    from src.research_director.registry import build_default_registry
    return build_default_registry(
        registry_dir=(sess.cfg.get("registry_dir") or str(model_registry_dir())),
        program_id=sess.program_id,
        model_name=(sess.cfg.get("model_name") or "model"),
        external_research=bool(sess.cfg.get("external_search_enabled", True)))


def _rd_critic_registry(sess: RDSession) -> Any:
    from src.research_director.registry import build_critic_registry
    return build_critic_registry(
        registry_dir=(sess.cfg.get("registry_dir") or str(model_registry_dir())),
        program_id=sess.program_id,
        model_name=(sess.cfg.get("model_name") or "model"))


def _mem(sess: RDSession, field: str) -> Any:
    from src.research_director.memory import AgentMemory
    mcfg = sess.cfg.get("memory") or {}
    return AgentMemory(field, recall_top_k=int(mcfg.get("recall_top_k", 6)),
                       recall_min_similarity=float(mcfg.get("recall_min_similarity", 0.7)))


def _recall(sess: RDSession, field: str) -> str:
    """Per-lens cross-cycle memory of this agent's prior diagnoses/outcomes (pgvector).
    Best-effort — a cold or unavailable store just yields no recall."""
    try:
        text = _mem(sess, field).recall_text(
            f"prior model-improvement findings and outcomes for {sess.program_id}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("rd_memory_recall_failed field=%s: %s", field, exc)
        return ""
    if not text or text == "(no relevant prior memory)":
        return ""
    return "\n\n## YOUR PRIOR MEMORY (semantic recall of past cycles)\n" + text


def _remember(sess: RDSession) -> None:
    """Persist each lens's prescription + the validation verdict for cross-cycle learning."""
    verdict = (sess.validation or {}).get("verdict", "n/a")
    top = _top_change(sess.plan) or {}
    for field in sess.cfg["fields"]:
        draft = sess.drafts.get(field)
        if not draft:
            continue
        try:
            _mem(sess, field).write(
                _clip(draft, 1500), kind="finding",
                metadata={"lens": field, "run_ts": sess.run_ts,
                          "top_recommendation": top.get("id"),
                          "validation_verdict": verdict})
        except Exception as exc:  # noqa: BLE001 — memory write is additive, never blocking
            logger.warning("rd_memory_write_failed field=%s: %s", field, exc)


def _evidence_block(sess: RDSession) -> str:
    """The existing model + patterns in its results (the Context Pack), plus the recent-journal
    dedup guard so the agent does not re-propose a DISPROVEN idea."""
    block = "## EXISTING MODEL & RESULT PATTERNS (diagnose this)\n" + sess.evidence
    if sess.recent_signatures:
        block += ("\n\n## ALREADY-PROPOSED CHANGE SIGNATURES (do NOT re-propose these)\n"
                  + json.dumps(sess.recent_signatures[:40]))
    return block


# ── validation helpers ───────────────────────────────────────────────────────────

def _top_change(plan: dict[str, Any]) -> dict[str, Any] | None:
    changes = plan.get("changes") or []
    top_id = plan.get("top_recommendation")
    return next((c for c in changes if c.get("id") == top_id), (changes[0] if changes else None))


def _find_candidate_row(summary: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any] | None:
    cands = summary.get("candidates") or []
    return next((c for c in cands
                 if c.get("candidate_id") == cand.get("id") or c.get("family") == cand.get("family")),
                (cands[0] if cands else None))


def _run_metrics_from_candidate(row: dict[str, Any] | None) -> dict[str, Any]:
    """Shape a search candidate row into the training_result-shaped dict the Evaluator consumes.

    NOTE: keys stay as ``precision_at_tau`` etc. — NO remap to ``val_*`` — so the run and the
    baseline ``training_result`` resolve the same primary key. ``n_signals`` is normalized to the
    dict shape the Evaluator's cold-start floor expects. Per-fold arrays (``test_folds``) are used
    if the score exposes them; absent them the Evaluator returns INCONCLUSIVE rather than a false
    PASS (the multi-fold + bootstrap-CI gates need them) — an honest verdict, not a fabricated one.
    """
    if not row:
        return {}
    score = dict(row.get("score") or {})
    out: dict[str, Any] = {k: v for k, v in score.items()
                           if k not in ("n_signals", "note")}
    ns = score.get("n_signals")
    if isinstance(ns, dict):
        out["n_signals"] = {k: int(v) for k, v in ns.items() if isinstance(v, (int, float))}
    elif isinstance(ns, (int, float)):
        out["n_signals"] = {"total": int(ns)}
    return out


# ── report + observability ───────────────────────────────────────────────────────

def _render_report(sess: RDSession) -> str:
    plan, val = sess.plan, sess.validation
    lines = [
        f"# Model Improvement Report — {sess.program_id}",
        f"*Research Director cycle `{sess.run_ts}`{' — DRY RUN' if sess.dry_run else ''} "
        f"(objective `{sess.objective_id}`)*",
        "",
        "## Dominant failure modes (diagnosed from the model's results)",
        "",
    ]
    for fm in plan.get("dominant_failure_modes") or []:
        anchors = ", ".join(f"[{e.get('tier')}] `{e.get('source_ref')}`" for e in fm.get("evidence") or [])
        lines.append(f"- **{_one_line(fm.get('summary', ''))}** — {anchors}")
    lines += ["", "## Prescribed changes (ranked; highest-leverage first)", "",
              "| rank | id | scope | change | targets | expected effect | auto-validate |",
              "|---|---|---|---|---|---|---|"]
    for c in sorted(plan.get("changes") or [], key=lambda x: x.get("rank", 999)):
        eff = c.get("expected_effect") or {}
        eff_s = f"{eff.get('metric', '?')} {eff.get('direction', '')} {eff.get('magnitude', '')}".strip()
        mark = "✓" if c.get("auto_validatable") else "handoff"
        top = " ⭐" if c.get("id") == plan.get("top_recommendation") else ""
        lines.append(f"| {c.get('rank', '-')} | {c.get('id')}{top} | {c.get('scope')} "
                     f"| {_one_line(c.get('title', ''))} | {_one_line(c.get('targeted_failure_mode', ''))} "
                     f"| {_one_line(eff_s)} | {mark} |")
    lines += ["", "## Top recommendation — auto-validation", ""]
    if val:
        lines += [
            f"- **Verdict: `{val.get('verdict')}`** ({val.get('status')})",
            f"- {val.get('reason', '')}",
        ]
        if val.get("primary_delta") is not None:
            lines.append(f"- Measured Δ{val.get('metric', '')} = {val.get('primary_delta'):+.4f} "
                         f"(run {val.get('run_primary')}, baseline {val.get('baseline_primary')}, "
                         f"{val.get('n_folds', 0)} folds)")
        if val.get("validation_method"):
            lines.append(f"- Handoff validation method: {val['validation_method']}")
    else:
        lines.append("- (not run)")
    if plan.get("reconciliation"):
        lines += ["", "## Synthesis reconciliation (primary vs replication)", "", str(plan["reconciliation"])]
    oq = plan.get("open_questions") or []
    if oq:
        lines += ["", "## Open questions for the operator", ""] + [f"- {q}" for q in oq]
    lines += ["", "---",
              f"*Provenance: author={sess.author_provider}; metrics={json.dumps(sess.metrics, default=str)}; "
              f"artifacts in `{sess.out_dir}`.*"]
    return "\n".join(lines) + "\n"


def _log_to_mlflow(sess: RDSession) -> None:
    try:
        import mlflow
        from src.training.config import MLflowConfig
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLflowConfig().tracking_uri))
        mlflow.set_experiment(f"research_director/{sess.program_id}")
        with mlflow.start_run(run_name=f"rd_{sess.run_ts}"):
            mlflow.set_tags({"program_id": sess.program_id, "objective_id": sess.objective_id,
                             "run_ts": sess.run_ts, "experiment_type": "research_director",
                             "dry_run": str(sess.dry_run), "author_provider": sess.author_provider,
                             "validation_verdict": (sess.validation or {}).get("verdict", "n/a")})
            if sess.metrics:
                mlflow.log_metrics({k: float(v) for k, v in sess.metrics.items()})
            for stage, info in sess.state.get("stages", {}).items():
                with mlflow.start_run(run_name=stage, nested=True):
                    mlflow.set_tags({"program_id": sess.program_id, "stage": stage, "run_ts": sess.run_ts})
                    if isinstance(info.get("elapsed_s"), (int, float)):
                        mlflow.log_metric("elapsed_s", float(info["elapsed_s"]))
            mlflow.log_artifacts(str(sess.out_dir))
        logger.info("mlflow_rd_logged experiment=research_director/%s", sess.program_id)
    except Exception as exc:  # noqa: BLE001 — observability never fails the run
        logger.warning("mlflow_rd_log_failed: %s", exc)


# ── misc ─────────────────────────────────────────────────────────────────────────

def rd_signature(change: dict[str, Any]) -> str:
    """Stable signature of a proposed change for cross-cycle 'don't re-propose' dedup."""
    key = json.dumps({"scope": change.get("scope"), "title": change.get("title"),
                      "candidate": change.get("executable_candidate")}, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _empty_plan(sess: RDSession) -> dict[str, Any]:
    return {"objective_id": sess.objective_id, "program_id": sess.program_id,
            "generated_at": datetime.now(timezone.utc).isoformat(), "run_ts": sess.run_ts,
            "dominant_failure_modes": [{"summary": "dry run — no diagnosis performed",
                                        "evidence": [{"tier": "T2", "source_ref": "dry_run"}]}],
            "changes": [{"id": "noop", "title": "dry run", "scope": "hparam",
                         "targeted_failure_mode": "none", "mechanism": "dry run",
                         "expected_effect": {"metric": "precision_at_tau", "direction": "increase"},
                         "evidence": [{"tier": "T2", "source_ref": "dry_run"}],
                         "feasibility": "n/a", "auto_validatable": False,
                         "validation_method": "dry run"}],
            "top_recommendation": "noop", "reconciliation": None}


def _print_summary(sess: RDSession) -> None:
    top = _top_change(sess.plan) or {}
    verdict = (sess.validation or {}).get("verdict", "n/a")
    print(f"\n{'─' * 62}\n  Research Director cycle complete  {sess.run_ts}"
          f"\n  Program / objective : {sess.program_id} / {sess.objective_id}"
          f"\n  Changes proposed    : {len(sess.plan.get('changes') or [])}"
          f"\n  Top recommendation  : {_one_line(top.get('title', 'n/a'))}"
          f"\n  Validation verdict  : {verdict}"
          f"\n  Report              : {sess.out_dir / 'improvement_report.md'}"
          f"\n{'─' * 62}\n")
