# ADR-012 — Gated Model Promotion + MLflow Lineage Spine

**Date**: 2026-06-30  
**Status**: Accepted (design)  
**Relates to**: [RESEARCH_DIRECTOR.md](../architecture/RESEARCH_DIRECTOR.md),
[PHASE-19-RESEARCH-DIRECTOR.md](../roadmap/PHASE-19-RESEARCH-DIRECTOR.md)  
**Supersedes**: nothing (first ADR on automated promotion + research-loop MLflow lineage)

---

## Context

Phase 18 stood up an MLflow tracking server (Postgres backend, OCI S3-compat artifact root,
experiment `nq-50pt-intraday`) and wired it into [trainer.py](../../src/training/trainer.py)
for **manual** training runs only. The Research Director research loop logs to its own
Postgres `research_journal` but **not** to MLflow, so an automated experiment's lineage
(objective → finding → spec → run → metrics → registry) is not reconstructable from one
system. Separately, the automated loop will stage and could promote models into
`data/model_registry/` (flipping `LATEST_STABLE.json`), an effectively irreversible action,
with no human checkpoint and no statistically rigorous gate.

## Decision

### 1. MLflow is the lineage spine for the research loop
- Each research **iteration** opens a nested **parent run**; each GPU **training attempt**
  is a **child run** (`mlflow.parent_run_id`). Params = `config_diff` + spec fields;
  metrics = eval metrics + per-iteration cost (`llm_usd`, `gpu_usd`, `iter_usd`);
  artifacts = expert findings, plans, verdict commentary, diagrams.
- `experiment_id ↔ mlflow_run_id` is persisted in `research_journal` and mirrored into
  `provenance.json` (already supported by [promote_model.py](../../scripts/promote_model.py)).
- Staged models register as MLflow Model Registry versions (stage `Staging`); promotion
  transitions the version to `Production`, kept in lockstep with `LATEST_STABLE.json`.

### 2. Promotion is gated
- **Stage unconditionally**: every completed run's model is copied to `data/model_staging/`
  with full provenance — cheap, auditable, reversible.
- **Promote only on a deterministic PASS** vs the current `LATEST_STABLE` baseline:
  (a) primary metric Δ ≥ `delta_min` on the walk-forward **test** set; (b) guardrail
  metrics (BD precision, calibration) do not regress beyond ε; (c) improvement holds on
  **≥ 2 of 3 folds** (motivated directly by the v010 fold-1 78.6% BU P@0.8 vs near-zero
  elsewhere); (d) 95% bootstrap CI on the aggregate Δ excludes 0.
- **Cold-start path** (no baseline): absolute thresholds (`precision@0.8 > 2×class_prior`,
  `n_signals > 50` per direction) instead of relative deltas.
- The `LATEST_STABLE` flip requires an explicit **human acknowledgement**; the LLM verdict
  commentary is advisory and never sets the gate state.

## Consequences
- Full forensic reconstruction of every automated decision from MLflow + journal.
- A single irreversible action (registry flip) is protected by a cheap human checkpoint.
- Requires building the Evaluator/Registrar (not yet implemented) and adding MLflow
  parent/child nesting to the Director — tracked in PHASE-19.
