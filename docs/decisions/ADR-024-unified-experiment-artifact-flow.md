# ADR-024: Unified Experiment Artifact Flow — MLflow Lineage, Champion-Finalize Bridge, Durable Results

## Status
Accepted — refines [ADR-012](ADR-012-gated-promotion-mlflow-lineage.md) (gated promotion + MLflow lineage spine, previously unwired for search) and [ADR-018](ADR-018-objective-doc-single-source-labels.md) (objective doc as single source; extended here to artifact keying). Complements [ADR-022](ADR-022-oci-queue-code-bundle-autonomous-gpu.md) (the `results/` prefix it introduced is now formally a transient outbox).

## Date
2026-07-14 (post-TBM three-model test storage review)

---

## Context

An end-to-end audit of the `nq-50pt-tbm` runs against the Phase-18 storage design found the
run **locations** correct but the **pipeline plumbing** fragmented:

1. **No durable copy of the authoritative results.** `data/search/` (OOF, diagnostics, gate
   verdicts, analysis reports) and `data/evaluation/` (frozen contracts, leaderboards) had no
   `oci_sync.conf` targets. The only object-storage copies were the GPU plane's transient
   `results/<run_id>/` handoff uploads — partial (GPU-trained base candidates only; no gate
   verdicts, no ensembles, no contracts) and never cleaned.
2. **The search dead-ended at `champion_id`.** Search deliberately persists OOF only (its k
   fold-models per candidate exist to rank, not to serve), but nothing bridged a selected
   champion into `training_runs/` → `model_staging/` → `model_registry/`. The
   requirement-meeting tuned XGBoost had to be re-derived by hand.
3. **Provenance loss.** The GPU result collector rebuilt candidates from **catalog defaults**,
   discarding the executed (tuned) `spec_params` and `feature_selection`; the tune wave's
   in-memory Optuna study made the winning params unrecoverable. Leaderboard rows carried
   `mlflow_run_id: null` — ADR-012's lineage spine was never wired for search.
4. **Fragmented exploration taxonomy.** `research_runs/` (HPO/ablation; keeps every trial's
   model) vs `data/search/` (family selection; keeps no models) are the same category —
   comparative sweeps on the frozen contract — with contradictory retention and no unified
   view for a human to verify what was run.
5. **Inconsistent keying**: `program=<id>-<session>` (training/staging/registry) vs
   `objective_id/run_ts` (scoping/search/evaluation).

## Decision

1. **MLflow is the unified verification surface.** Every search wave logs to experiment
   `search/<objective_id>`: one parent run per wave (`wave_<run_ts>`), one **nested run per
   candidate** with tags (`objective_id`, `run_ts`, `experiment_type`, `candidate_id`,
   `family`, `compute_class`), params (executed `spec_params` + `feature_selection`) and the
   full KPI metric set. `experiment_type ∈ {model_selection, manual_override, hpo, ablation,
   feature_refine, finalize}` tags unify the exploration taxonomy inside MLflow without a
   disruptive filesystem migration. Implemented in `controller._log_wave_to_mlflow`
   (best-effort — MLflow outage never fails a search); leaderboard rows now carry the real
   `mlflow_run_id`; `src/search/mlflow_backfill.py` replays historical
   `search_result.json` files (idempotent on the `run_ts` tag) — 75 historical waves
   backfilled at adoption.
2. **Champion-finalize bridge** (`src/search/finalize.py`): reads a search run's champion (or
   named candidate) with its **executed** config, refits it once on the full walk-forward
   matrix (validation tail for a sanity score), and persists a promotable serving bundle —
   `model.<keras|ubj|pkl>` + `scaler.pkl` (tabular plugins now return their fitted scaler in
   `FittedModel.meta`) + `feature_schema.json` + `finalize_manifest.json` (search run dir,
   scores, params, git sha, `mlflow_run_id`) — to `data/training_runs/<objective_id>/<ver>/`
   and copies it to `data/model_staging/<objective_id>/<ver>/` for the existing
   `promote_model.py` gate. The finalized model + artifacts are logged to the same MLflow
   experiment (`finalize_<ver>`), so selection → finalized model is one lineage chain.
   Ensembles are rejected (finalize their base members).
3. **Executed-config echo (provenance).** `job_exec` writes `spec_params` +
   `feature_selection` into `result.json`; `publish_result` carries them on the results
   channel; the collector prefers them over catalog defaults. Tuned params can no longer be
   lost.
4. **`results/<run_id>/` is a transient outbox.** `collect_result` deletes each object after
   a verified download; the durable copy is the collected `data/search/` run dir.
5. **Durability via sync targets** (`oci_sync.conf`): `data/search → vtech_archive/search_runs/`,
   `data/evaluation → vtech_archive/evaluation/`, and the auxiliary dataset declaration
   `data/parquet_raw1s → vtech_hot/parquet_raw1s/` (CPU→S3 push; the GPU side self-pulls via
   the job `data_s3_prefix` — complementary halves of one mechanism). Legacy
   `vtech_hot/gpu_jobs/` (pre-ADR-022 marker FSM) purged after manifest capture.
6. **Keying converges on `objective_id`** for all NEW artifacts (finalize writes
   `training_runs/<objective_id>/…`, `model_staging/<objective_id>/…`). Existing
   `program=<session>` dirs are left in place (append-only history); the convention retires
   by attrition.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Merge `data/search` into `training_runs` | one folder | conflates k throwaway fold-models with deliberate serving models; storage blow-up |
| Persist every candidate's fitted model | no refit cost | most candidates are discarded; DL artifacts are large; retention policy inverts |
| Physical `experiments/` mega-migration now | clean tree | disruptive to append-only history; MLflow tagging achieves the unified *view* without it |
| **OOF-only search + champion refit bridge + MLflow taxonomy** (chosen) | cheap sweeps, single promotable artifact, one browsable lineage | one extra refit per champion (bounded; plugin may later persist its final-fold model to skip it) |

## Consequences

- **+** A human can verify every model/test in one place: MLflow experiments
  `search/<objective_id>` (all waves, all candidates, params + KPIs) and the finalize runs
  holding the actual model artifacts.
- **+** `search → finalize → staging → promote` is a complete, provenance-carrying chain;
  leaderboard ↔ MLflow ids cross-reference.
- **+** Results survive VM loss (archive sync); hot bucket stops accumulating handoff files.
- **−** One extra full-matrix refit per champion (minutes for tabular; bounded for DL via the
  existing memory-safe plugin paths).
- **−** Historical tuned params from before the echo fix are unrecoverable (the 2026-07-11
  tune wave's exact 0.804 config); re-running tuning under the new plumbing re-derives and
  persists them.
- **Constraint**: MLflow logging is best-effort by design — a tracking-server outage degrades
  observability, never correctness; `mlflow_backfill` reconciles gaps afterwards.

Config: `config/oci_sync.conf` (3 new targets)
Implementation: `src/search/finalize.py`, `src/search/mlflow_backfill.py`,
`src/search/controller.py` (`_log_wave_to_mlflow`, `CandidateResult.feature_selection/mlflow_run_id`,
executed-config collection), `src/gpu_runner/dispatch.py` (outbox cleanup, config echo),
`src/gpu_runner/job_exec.py` (config echo), `src/models/tabular_plugins.py` (scaler in meta)
Tests: `tests/test_finalize_pipeline.py`
See also: [ADR-012](ADR-012-gated-promotion-mlflow-lineage.md), [ADR-018](ADR-018-objective-doc-single-source-labels.md), [ADR-022](ADR-022-oci-queue-code-bundle-autonomous-gpu.md), [ADR-023](ADR-023-triple-barrier-label-scheme.md)
