# ARCHITECTURE.md — Canonical Research Architecture

> **Public export:** [PUBLIC_SHOWCASE.md](PUBLIC_SHOWCASE.md) defines the tested
> offline entry point and integration boundaries. Deployment state and historical
> research metrics below are inherited source context, not claims about a running
> public service. `src/demo.py` composes existing research primitives without
> changing production feature, label, loss, or serving semantics.

> **2026-09-24 causal correction:** the older blanket causal assurance for
> Track C below is superseded for `_order_flow` in `entry_state.py`:
> `es_cvd_*`, `es_delta_*`, and `es_ofi_cum` normalize by maxima across
> requested session decisions and can depend on later rows. Do not treat
> affected historical matrices as leakage-free until rebuilt and revalidated.
> Independent experiment `nq-selective-intraday` uses new closed-bar/raw-tape
> features in `src/features/nq_selective.py`, quote-event outcomes in
> `src/backtest/nq_selective.py`, and nested evaluation via
> `scripts/research/train_nq_selective.py`. Status: **EXPERIMENTAL, rejected
> on development OOF**; no serving integration or promotion. Shared paths,
> causal joins, economics and manifests are reused. See
> [measured results](research/NQ_SELECTIVE_RESULTS.md) and
> [ADR-138](decisions/ADR-138-nq-selective-causal-research.md).

> **2026-09-24 TBBO profile-reclaim result:** a separate raw-tape experiment
> uses exact prior cash profiles, closed-minute tape features and 250-ms
> quote-event replay in `src/features/nq_profile_reclaim.py`,
> `src/backtest/nq_profile_reclaim_quotes.py`, and
> `scripts/research/run_nq_profile_reclaim.py`. Status: **EXPERIMENTAL,
> rejected**. Its discovery grid and frozen April–June 2026 validation did not
> meet economic or risk gates; the July–September reserve remains unread. See
> [measured results](research/NQ_TBBO_PROFILE_RECLAIM_RESULTS.md) and
> [ADR-140](decisions/ADR-140-nq-tbbo-profile-reclaim-rejection.md).

**Status: CANONICAL, CURRENT.** This document describes what the durable architecture of this platform *is*, independent of which historical track (A/B/C) happens to implement each piece today. It supersedes `docs/architecture/*.md` as the required entry point; those older docs remain accurate implementation detail for Track A/B and are cited below where still relevant, but a fresh agent should start here, not there.

See `docs/decisions/ADR-136-canonical-research-architecture.md` for why this document exists and `docs/MIGRATION_MANIFEST.md` for the file-level migration record.

## 0. How this document was derived

Per ADR-136, the responsibilities below were derived from the project's actual trading objective (selective, favorable-R:R intraday NQ signals surviving realistic execution costs) and an independent, code-verified audit of the repository — **not** by renaming Track B's or Track C's existing module boundaries. Track-specific terms ("Stage 1", "gate", "meta model", "champion", "objective") are implementation vocabulary for *one* strategy inside a responsibility below, not architecture. Do not assume a future predictive approach needs a cascade, a gate, or a meta layer just because Track C or B currently has one.

> **Revision (2026-09-22, third pass).** ADR-136 defined twelve responsibilities *before* a bottom-up audit of the code they were meant to describe. A subsequent full audit of all 21 `src/` packages (~85k lines) established that **the twelve are a decomposition of the research problem, not of this repository**: eight research-layer responsibilities and thirteen operational capabilities had no owner. The model is revised below into **two planes**. Evidence for every item is in `docs/FUNCTIONAL_COVERAGE.md`; nothing here is aspirational.

## 1. Two planes

The platform has a **research plane** (what produces and validates a trading edge) and a **platform plane** (what makes running the research plane possible at all). Both are durable. The second pass modelled only the first, which is why a third of the codebase had no architectural home.

```
RESEARCH PLANE
  DATA CONTRACTS → DATASET REGISTRY → FEATURE BUILDER → OUTCOME BUILDER
    → MODEL STRATEGY → VALIDATOR → CALIBRATOR → DECISION POLICY
    → ECONOMIC EVALUATOR → EXPERIMENT → MODEL LIFECYCLE → SERVING
  TRADE DEFINITION and COST MODEL are cross-cutting inputs, not stages.
  POPULATION & ELIGIBILITY gates which rows any of the above may use.

PLATFORM PLANE
  ACQUISITION & INGESTION · STORAGE & SYNC · COMPUTE & RESOURCES
  JOB ORCHESTRATION & STATE · CODE IDENTITY · OBSERVABILITY
  INTERFACE SURFACE (WebUI/API) · RESEARCH AUTOMATION
```

### 1.0 Responsibilities added by the third-pass audit

These are **not** new code. Each names a capability that already exists and was previously unowned. See `docs/FUNCTIONAL_COVERAGE.md` §2 and §12.

| Responsibility | Plane | Current implementation | Why it is not a sub-part of an existing one |
|---|---|---|---|
| **Dataset Registry & Identity** | Research | `src/features/stores.py` (~47 call sites), `src/training/label_store.py` | Resolves store id → directory → catalog → column view → loader, and defines *content-addressed dataset identity*. Builds nothing, identifies no run. Highest-fan-in module in the repo. |
| **Trade Definition** | Research | `src/training/cascade_policy.py` | Defines what "the same trade" is. Both Outcome Builder and Decision Policy derive parameters from it; it is upstream of both. |
| **Population & Eligibility** | Research | `src/training/path_labels.py::session_r`, `min_stop_ticks`; selection/gating populations | Decides which rows/sessions are *admissible* for a given trade geometry — distinct from label correctness and from model quality. Also the home for cascade gate-population accounting. |
| **Cost Model** | Research (cross-cutting) | `src/training/cost_matrix.py`, objective `costs:` blocks, `EconomicAssumptions` | Costs enter both the **training objective** (`cost_matrix.py:216,368,410`) and post-hoc evaluation. Owning it in Economic Evaluator alone misses the path that changes what the model learns. |
| **Acquisition & Ingestion** | Platform | `src/acquisition/`, `src/ingestion/` | Live streaming, chain subscription, reconnection, vendor cost governance, bulk COPY, gap detection. Normal vendor definition acquisition runs alongside subscriptions (`src/acquisition/definition_acquisition.py`); rotated DBN files retain symbol mappings (ADR-156). Bounded historical recovery uses `src/acquisition/gap_recovery.py` and `src/ingestion/gap_recovery.py`: exact zero-cost acquisition, immutable receipts, and full-payload multiset append reconciliation (ADR-157). The October recovery published accepted futures/context and option products; broad 5-second feature restoration remains gated by timing and memory findings, recorded in the recovery results and DATA_PIPELINE. Data Contracts is schema/causality, not a runtime service. |
| **Storage & Sync** | Platform | `src/common/{data_paths,object_store}.py`, `src/gpu_runner/parquet_sync.py` | Logical path resolution and the 20-target OCI durability layer. |
| **Compute & Resources** | Platform | `src/common/{vm_config,gpu_provider,gpu_exec,oci_queue}.py`, `src/gpu_runner/` | Machine sizing (mandatory per AGENTS.md §9, 28 dependents), billable GPU lifecycle, queueing, admission control, fallback, health watchdog. |
| **Job Orchestration & State** | Platform | `src/webui/jobs.py`, `src/gpu_runner/dispatch.py` | Survives service restarts; the only job-state store. |
| **Code Identity** | Platform | `src/common/code_bundle.py`, `src/common/repro.py` | sha256-verified bundling guaranteeing the GPU executes the dispatched bytes. Experiment *records* identity; this *enforces* it. |
| **Observability** | Platform | `src/training/job_progress.py`, `src/common/logging.py` | In-flight heartbeats, per-fold checkpointing, crash forensics. Currently the weakest area: structlog has 1 caller against 92 `basicConfig` users, and the only alerting design (`research_director/sentinel.py`) has zero callers. |
| **Interface Surface** | Platform | `src/webui/` (~62 endpoints, 11,930 LOC) | The only UI for Track C. **Not a presentation layer** — `src/features/live_runtime.py`, `src/features/engine.py` and `src/execution/gateway.py` import from it, and it produces the live signal contract. |
| **Research Automation** | Platform | `src/research_director/`, `src/scoping/`, `src/agentic/`, `src/common/llm/` (43 files, ~7,100 LOC) | LLM cost metering, T1/T2/T3 evidence tiers with a human-gated write barrier, pgvector RAG, durable agent memory, and a deterministic promotion gate stricter than anything in `src/validation`. |

**Deliberately not promoted to a responsibility** (recorded so the decision is not silently revisited): *falsified-hypothesis preservation* (`src/training/stop_rule.py`) is a **property of the Experiment record**, not a component — but it is a real obligation: a module retained as executable evidence of a negative result must not be deleted as "unused". *Process liveness arbitration* (`live_runtime.py` ↔ `webui/app.py` heartbeat) is an implementation detail of Interface Surface + Serving, flagged here because it is an undocumented cross-process contract.

## 2. The research-plane responsibilities

### 1.1 Data Contracts
**Responsibility**: define, once, what the raw and intermediate data mean — schemas, timestamps, session/RTH conventions, contract-roll methodology, and the causal ("as-of") semantics every downstream responsibility must respect.
**Canonical module**: `src/common/causal.py` (`as_of_index`). There is no `src/data_contracts/` package. Feature, context, and upstream-prediction joins in `src/experiment/analysis.py` use `as_of_index` and refuse a value knowable only after the decision timestamp. Full column schemas remain documented in `docs/DATA_CONTRACTS.md`.
**Current implementations**: TimescaleDB `tbbo`/`bbo_1s`/`definitions`/`statistics` hypertables (`src/ingestion/`, `src/acquisition/`); continuous-contract roll logic (`src/common/contract_calendar.py`); Track C's tick print tape (`src/features/tape.py`).
**Status**: `ACTIVE`.

### 1.2 Feature Builder
**Responsibility**: deterministic, causal transformation from raw data into a decision-time state representation. Must guarantee "no information from after T." Datasets attached at fit time go through the as-of join in `src/experiment/analysis.py` (a closed bar at its close; a session summary after that session closes).
**Canonical module**: none separate from the implementations below. `src/features/` is Track B's feature engine, not an empty interface package.
**Current implementations**: Track B's `src/features/engine.py` (`FeatureEngine`, ~256-column catalog, `momentum_edge`/`sequence_baseline` views); Track C's `src/training/entry_state.py` + `src/features/flow_state.py` + `src/features/structural_transitions.py` + `src/features/cash_session_profile.py` (`es_`/`fs_`/`st_` families). Both are legitimate, independently-verified-causal strategies for the same responsibility — neither is canonical over the other yet.
**Status**: `ACTIVE` (both current implementations).

### 1.3 Outcome Builder
**Responsibility**: deterministic transformation from a decision point + forward price path into a trainable outcome — barrier/path-dependent target-before-stop logic, horizon, censoring. The outcome may use that forward path. The path is not a feature.
**Canonical module**: `src/outcomes/` (interface only).
**Current implementations**: Track B's `src/training/labels.py` (`_classify_triple_barrier`, fixed 50pt/6pt barrier); Track C's `src/training/move_label.py` (volatility-adaptive R-multiple barrier). Barrier width/horizon should be treated as a *parameter* of one Outcome Builder strategy, not two architecturally distinct systems.
**Status**: `PRODUCTION_WIRED_WEAK` (Track B — see `docs/CONFIG_AUDIT.md` on the fixed-barrier/majority-class-collapse correlation), `ACTIVE` (Track C).

### 1.4 Model Strategy
**Responsibility**: pluggable (features, outcomes) → (trained artifact, predictions) implementations. Includes single classifiers, sequence models, and composed strategies (cascades, ensembles/stacking).
**Canonical module**: `src/models/` (interface only).
**Current implementations**: Track B's model catalog (`src/models/catalog.py`: xgboost, attention_lstm, tcn_close, tcn_so_mbp_vix, meta__xgboost, meta_stack__attention_lstm); Track C's two sequential XGBoost classifiers (Track C's own "Stage 1"/"Stage 3" naming — a **sequential-gated composition strategy**, not a mandatory topology).
**Status**: `PRODUCTION_WIRED_WEAK` (Track B — meta-stack variants produce 0 signals in every run inspected; sequence models underperform XGBoost by a wide margin), `ACTIVE` (Track C).

### 1.5 Validator
**Responsibility**: fold construction (walk-forward, expanding/rolling), purge/embargo sized to the outcome horizon, sample-uniqueness/effective-sample-size weighting for overlapping labels.
**Canonical module**: `src/validation/` — real schedule (`folds_from_objective`, `walk_forward.py`). Track B's equal-count numpy split still lives in `src/search/controller.py::_build_folds`.
**Current implementations**: `src/validation/folds_from_objective` is the schedule both Track C evaluators apply. `eval_move_model.py` and `eval_target_model.py` turn each fold row into train/test frames with `_fold_masks`. Splits are `expanding_calendar`, `expanding_month_blocks`, and `last_range`. The manifest records the split and windows that schedule returned. Track B search still builds numpy folds in `src/search/controller.py::_build_folds`: calendar when the objective asks for it, otherwise the equal-count split (`fold_size = n // (n_folds + 1)`), which does not read the calendar settings. Sample-uniqueness weighting stays in `src/evaluation/uniqueness.py`. The historical two-engine disagreement is in `docs/CONFIG_AUDIT.md` (of 54 on-disk `folds.json`, `n_folds` is 3 in 38 runs, 6 in 12, 4 in 1, 1 in 3).

### 1.6 Calibrator
**Responsibility**: post-hoc score→probability mapping (isotonic/Platt) so downstream thresholding/EV math is justified.
**Canonical module**: none. `src/calibration/` does not exist. No calibrator is implemented on either track.
**Status**: `NOT_IMPLEMENTED`.

### 1.7 Decision Policy
**Responsibility**: turn (calibrated) model output into an actionable trade decision — thresholding, population selection/gating, fill-mode assumptions (immediate vs. resting), position/book construction.
**Canonical module**: none. `src/decision_policy/` does not exist. The behavior lives in the implementations below.
**Current implementations**: Track C's `p_move >= tau_q` gate + target-adaptive stop (`scripts/features/build_target_matrix.py`, ADR-130) + 1-position-book reweighting; Track B's `tau` confidence gate + champion-gate requirement floors.
**Status**: `ACTIVE`. Note: Track C's gate currently has no tooling to isolate whether a downstream metric change is caused by the gate's population shift vs. a genuine downstream model change — flagged as a critical open finding in `project_manifest.yaml`.

### 1.8 Economic Evaluator
**Responsibility**: turn a decision-policy output into realistic PnL — explicit slippage/commission/spread/fill-model/contract-multiplier assumptions, trade ledger, cost-sensitivity, drawdown. **This is the most duplicated and least trustworthy responsibility in the current repository.**
**Canonical module**: `src/evaluation/economic.py` — **real, tested** (`compute_trade_ledger`, `EconomicAssumptions`, `evaluate_cost_sensitivity`). `portfolio_sim` and Track C's persisted book in `eval_target_model.py` both price through it. Other paths still have their own arithmetic.
**Current implementations** (several still disagree — full enumeration in `docs/FUNCTIONAL_COVERAGE.md` §10): slippage, contract multiplier, and Sharpe annualization are not one number across the repo. Representative paths that do not yet go through `economic.py`: Track B's Tier A proxy (`src/evaluation/_kpi.py`); Track B's Tier B backtest (`src/backtest/engine.py` — reads an unset `slippage_points` key, silently ignoring the declared `slippage_model`); Track C's WebUI-facing `src/backtest/cascade_backtest.py`; plus `paper/engine`, `execution/ledger`, `external_models/*`, the WebUI chart interpreter, and `training/cost_matrix.py` (which puts costs into the training objective itself). Track C's persisted ledger does: entry is the fill (`price`, otherwise `fill_px`), exit is `tgt_level` when `y_reach == 1` and `stop_px` otherwise, including a row censored at the horizon. `decision_price` is not the fill. The console print in `eval_target_model.py` still shows an R sensitivity beside that ledger.
**Canonical execution contract** (`src/evaluation/economic.py`): a leg with `resolved_fill_price` is used verbatim and is not slipped again. Otherwise MARKET/STOP legs fill at the executable bid/ask (never midpoint; a STOP's price is a trigger, then MARKET semantics); LIMIT fills at its limit, improved only when upstream fill logic reports it. **Spread is not slippage** — bid/ask pricing captures it; slippage is only further adverse movement (assumed 0 unless a scenario sets it). One all-in commission per completed trade × quantity. Assumptions live in `config/execution_costs.yaml` (NQ $3.36; MNQ-equivalent $1.03 as an independent assumption, multiplier 0.1× NQ, priced off NQ data). Inference DEFERRED.
**Status**: `PRODUCTION_WIRED_BUGGY` (Track B), `ACTIVE_INCONSISTENT` (Track C). **Do not trust a headline expectancy/PnL number from either track without cross-checking `docs/CONFIG_AUDIT.md`.**

### 1.9 Experiment
**Responsibility**: durable, comparable identity for one research run — manifest, config snapshot, code/environment identity, artifact contract. Explicitly *not* a version-suffixed filename (`move_matrix_v3`, `_v4`) — those describe a dataset build, not a controlled experiment, and cannot answer "what exactly changed between these two runs."
**Canonical module**: `src/experiment/` — **real, tested**: `manifest.py` (`RunManifest`), `artifacts.py` (filenames and the join/OOF rules), `analysis.py` (runtime join, OOF analysis view, SHAP report), `config_validation.py`, `comparison.py`, `schemas.py`. **Track B search** still writes `manifest.json` beside `run.json` via `src/search/run_manifest.py` (additive; that path reports `evidence_level: INCOMPLETE` rather than fabricating a hypothesis or a seed). **Objective runs** write `manifest.json` as the only run-level file (`passes`, resolved objective, fold design). `model.json` holds one model's type, hyperparameters, target, feature columns, and join spec. See `docs/MIGRATION_MANIFEST.md` §20 and `docs/STORAGE_CONTRACT.md` §21.
**Current implementations**: new runs, including Optuna studies from `scripts/search/hpo_model_search.py`, write `data/research/runs/<objective_id>/<run_ts>/`. `research/runs/` is the primary new-run root. `data/research/custom_test/` has been migrated and removed. `data/research/search/` remains a compatibility root and is not the HPO layout. MLflow tracking (`mlflow.tracking_uri` in `config.yaml`) is still used by Track B search. The model id hashes that configuration. Training joins the sources named in `model.json` as-of each row's decision timestamp: a feature, context, or upstream prediction is attached only when it was knowable at or before that `ts`. `pass_<n>/training_matrix.parquet` is an optional cache of one row population. OOF files do not carry feature columns; `scripts/research/build_oof_analysis.py` rebuilds the wide table with the same as-of join. Ground-truth labels stay in `sources/labels/` with an immutable `_contract.json`. Older runs that still have `run.json`, `contract.json`, and `folds.json` remain readable. `src/backtest/runner.py::_resolve_oof_source` looks under `run_roots()` for `oof_*.parquet` or `pass_<n>/model_<id>/oof_predictions.parquet`.
**Status**: `PARTIAL_IMPLEMENTATION` — objective runs and Track B search both write `RunManifest`. Comparison, calibration, and a full canonical walk-forward class are not yet the only implementations.

### 1.10 Model Lifecycle / Registry
**Responsibility**: explicit status vocabulary and promotion gate for trained artifacts — which model is trusted for which purpose, and why.
**Canonical module**: `src/serving/lifecycle.py` — **real, tested** (`LifecycleStatus` enum: `EXPERIMENTAL, CANDIDATE, PRODUCTION, DEPRECATED, ARCHIVED`).
**Current implementations**: Track B's `data/models/model_registry/`, `data/models/leaders/`, `data/models/model_staging/`, ADR-071 optional champion gate, `src/serving/promote.py` (manual promotion CLI). Track C has no lifecycle/registry integration at all — it is exposed only as a WebUI backtest-comparison overlay.
**Status**: `PRODUCTION_WIRED` (Track B infra is real; only one model has ever been promoted, 2026-08-05, and it has not been re-promoted since despite continued research).

### 1.11 Serving (Paper / Live)
**Responsibility**: invoke a (production-status model, decision policy) pair on live/replay data; log paper trades; eventually route live orders.
**Canonical module**: `src/serving/` — the Track B serving implementation (`bundle.py`, `lifecycle.py`, `promote.py`), not an empty interface package.
**Current implementations**: `src/serving/`, `src/paper/` (`vtech-paper.service`), `src/execution/` (ADR-032 risk-aware gateway) — all Track B only, resolved via `config.yaml`'s `objective_id` (`src/webui/live.py`). Paper trading has been inactive since 2026-08-05. Track C has no serving path.
**Status**: `WIRED_BUT_INACTIVE`.

### 1.12 Orchestrator
**Responsibility**: propose and sequence experiments across responsibilities 1–11 (HPO, feature/model search, autonomous research loop). A consumer of the chain above, not a parallel reimplementation of it.
**Canonical module**: none. `src/orchestration/` does not exist. HPO and the research director are the implementations below.
**Current implementations**: `src/research_director/`, `src/scoping/` (ADR-025 Scoping v2), `scripts/search/*` (HPO). `hpo_model_search.py` scores every trial on the folds `_build_folds` produced for that study (the objective's walk-forward, after `--folds`) and writes those windows onto `manifest.validation`. It does not stamp `last_range` / `n_folds=1` unless that is the schedule that ran. Trials still share that fold set, so the test windows are reused across the study. An HPO score is a search signal; the selected configuration is re-validated by an ordinary run.
**Status**: `ACTIVE`.

## 3. Lifecycle status vocabulary

Defined once, canonically, in `src/serving/lifecycle.py`:

| Status | Meaning |
|---|---|
| `EXPERIMENTAL` | Under active development; not evaluated against the locked holdout; no promotion claim. |
| `CANDIDATE` | Passed the canonical Validator + Economic Evaluator on a held-out (non-locked) window; eligible for promotion review. |
| `PRODUCTION` | Promoted (Model Lifecycle §1.10); currently the trusted model/decision-policy pair for its Serving path. |
| `DEPRECATED` | Formerly PRODUCTION or CANDIDATE, explicitly superseded; kept for reference/rollback. |
| `ARCHIVED` | No longer under any active development; historical only (e.g. Track A in full). |

Do not invent parallel vocabulary ("champion", "leader", "promoted") in new code — those remain acceptable *internal* synonyms only inside Track B's existing Model Lifecycle implementation during migration.

## 3a. Where the canonical implementations live (2026-09-23 consolidation)

`src/canonical/` no longer exists. It was a parallel tree: eight of its twelve
packages (`features`, `outcomes`, `models`, `validation`, `calibration`,
`decision_policy`, `serving`, `orchestration`) contained **zero** functions or
classes — only docstrings restating the responsibility definitions in §1.1–§1.12
above. A package that documents is a document, so those definitions now live
here only, and the empty packages are gone.

The four packages with real implementations moved to the obvious existing home
for their responsibility, so each responsibility has one place rather than a
canonical/legacy pair:

| Responsibility | Permanent home |
|---|---|
| Experiment identity, artifact contract, model identity, as-of analysis joins, comparison, config validation, objective schema, CLI | `src/experiment/` |
| Economic evaluation (execution pricing, commission, multiplier, PnL, ledger, cost + execution sensitivity) | `src/evaluation/economic.py` |
| Strict as-of causality primitives | `src/common/causal.py` |
| Lifecycle status vocabulary and transition graph | `src/serving/lifecycle.py`, beside `promote.py`, the implementation it governs |

The CLI is `python -m src.experiment.cli`. The DB-free architecture test suite is
`tests/architecture/`.

Nothing was removed for being new: manifest/run identity, the artifact contract,
deterministic analysis joins, model identity, strict causality and economic
evaluation are all intact, with the same tests. What was removed is the second
tree they were sitting in.

## 4. What is NOT canonical (see ADR-136 §5 for full reasoning)

- "Stage 1"/"Stage 3" as mandatory topology — one Model Strategy/Decision Policy composition, not architecture.
- "meta model" as a required layer — currently non-functional (0 signals in every inspected run); an optional Model Strategy shape only.
- "gate" as hardwired two-script plumbing — the concept survives as a Decision Policy strategy; the plumbing does not.
- "objective_id" as a single monolithic config axis — see `docs/CONFIG_AUDIT.md` for how this exact conflation caused a silent cost-modeling bug.
- "Track A/B/C" as an ongoing organizing principle — historical experiment-family labels only, tracked in `project_manifest.yaml` during migration, not meant to persist indefinitely.
- Version-suffixed dataset/script names (`_v3`, `_v4`) — superseded by Experiment (§1.9) run IDs once that responsibility is fully wired.

## 5. Reading further

**Correction (2026-09-22, second restructuring pass)**: an earlier version of this section classified all of `docs/architecture/*.md` as "historical reference, superseded by this doc as the entry point." That was wrong for most of that directory and has been corrected below — do not re-introduce the blanket claim.

`docs/architecture/*.md` follows an existing "Present/Supersedes/Next" convention (ADR-028) and, per those files' own headers, is **actively maintained and current** — not historical:

| Doc | Status | Scope note |
|---|---|---|
| `SYSTEM.md` | **CURRENT, authoritative** | VM/services/stack — shared infra, track-agnostic. This is what `AGENTS.md` §8 already points to. |
| `DATA_LAYOUT.md` | **Snapshot, 2026-09-03, with a 2026-09-23 run-root correction** | Runtime `$VTECH_DATA_DIR` layout. New runs write `research/runs/`. The 2026-09-03 map still describes `research/custom_test/` as it was then. `docs/STORAGE_CONTRACT.md` is the current storage contract. |
| `TREE_INVENTORY.md` | **CURRENT, authoritative** (dated 2026-08-05) | Path-by-path disposition table; states the same "delete only with explicit operator approval" rule as `AGENTS.md` §4. Has partial, slightly stale Track C coverage (mentions the original unversioned `move_matrix/`/`target_matrix/` paths from ADR-127, not the `_v2/_v3/_v4` versions that exist on disk today — see `docs/STORAGE_CONTRACT.md`). |
| `DATA_PIPELINE.md` | **CURRENT, authoritative** | Acquisition/ingestion/DB/Parquet three-tier architecture — shared infra; Track C's tape also reads the tables this describes. |
| `DATABASE_SCHEMA.md` | **CURRENT** (schema/DDL) **but row-count table is a stale snapshot** ("as of 2026-05-19") | Schema is shared infra; don't quote the row-count table as current without re-verifying. |
| `FEATURE_ENGINE.md` | **CURRENT, authoritative, but scoped to Track B's `FeatureEngine` only** | Does not describe Track C's `es_`/`fs_`/`st_` feature engine at all — see `docs/reference/STAGE1_FEATURES.md`/`STAGE3_FEATURES.md` for that. |
| `ML_MODEL.md` | **CURRENT, authoritative, but scoped to Track B only** | Explicitly states "Live pin is `nq-50pt-econ`"; does not mention the cascade/Track C anywhere. Don't read this doc's silence on Track C as Track C being unimportant — it's a scope gap in this doc, not a judgment. |
| `RESEARCH_DIRECTOR.md` | **CURRENT, authoritative** (Present: 2026-08-12) | Track B's Orchestrator implementation; explicitly notes the RD timer is operator-disabled. |
| `WORKFLOW.md` | **CURRENT, authoritative** | "Start here" doc per `docs/README.md`; end-to-end planes — Track B + shared infra; silent on Track C (same gap as `ML_MODEL.md`). |

None of the above is "historical" — the correct characterization is **current, authoritative, but with a Track C blind spot** (all nine were written/maintained before or without reference to the cascade). Track C's implementation depth lives in `docs/reference/STAGE1_FEATURES.md`, `docs/reference/STAGE3_FEATURES.md`, and ADR-127/130/134 instead. Track B's additional implementation depth: `docs/reference/FEATURE_CATALOG*.md`.

- Why decisions were made historically: `docs/decisions/ADR-*`. See `docs/decisions/ADR_INDEX.md` for a status/topic index — do not read all 136 cold; that index tells you which are relevant to a given area and which are confirmed superseded.
- Full config field-by-field truth (current runtime behavior): `docs/CONFIG_AUDIT.md`. For the target composable config architecture this is migrating toward, see `docs/CONFIG_AUDIT.md` §6 "Current vs. Target."
- Canonical storage/artifact layout: `docs/STORAGE_CONTRACT.md` (extends `DATA_LAYOUT.md`/`TREE_INVENTORY.md`, does not replace them).
- Environment setup, logging expectations, fast verification loop: `docs/ENVIRONMENT.md`.
- Script-by-script classification (canonical entry point / experiment-specific / diagnostic / deprecated): `docs/SCRIPT_INVENTORY.md`.
- Causality/schema contracts: `docs/DATA_CONTRACTS.md`.

## Independent NQ continuation research — 2026-09-24 (ADR-139)

Status: **EXPERIMENTAL**, separate from Track B/C and not serving-wired.
Measured disposition and exact run: [NQ retry results](research/NQ_RETRY_RESULTS.md).
The learned ADR-138 approach remains rejected. The retry tests a prior-session
empirical volatility envelope and a directly specified trading policy; an ML
cascade is not an architectural requirement.

| Responsibility | Experimental implementation |
|---|---|
| Data preparation | `scripts/research/build_nq_long_bars.py`; additive, zero-cost-only repair in `repair_nq_discovery_gaps.py` |
| Causal state | `src/features/nq_noise_area.py`: cash open, prior 14 eligible sessions' same-clock absolute returns, current closed-bar VWAP in execution engines |
| Policy and historical outcomes | `src/backtest/nq_cadence_area.py`: cadence-confirmed exits, optional fixed opposite-band emergency stop |
| Quote execution | `src/backtest/nq_cadence_quotes.py`: observed bid/ask, latency, adverse slippage, scheduled flatten |
| Chronological selection | `scripts/research/run_nq_cadence_area.py`: discovery-only parameter selection, immutable frozen policy, later validation |
| Compatibility/reserve evaluation | `scripts/research/eval_nq_cadence_quotes.py`: source/fee preflight, context provenance, final-reserve gate |
| Verification/reporting | `check_nq_cadence_causality.py`, `audit_nq_cadence_marks.py`, fixed benchmark scripts, `report_nq_cadence.py` |

Predecessor `nq_noise_area.py`/`nq_noise_quotes.py` and their runners remain
preserved with the rejected minute-ratcheted variation. All runs use existing
canonical data/run path resolution. Quote ledgers reconcile to canonical
`src/evaluation/economic.py`; manifests use `src/experiment/manifest.py`.
The objective-label schema cannot honestly encode this no-fixed-target policy,
so there is no fabricated objective YAML or production-loader integration.
See [ADR-139](decisions/ADR-139-nq-cadence-policy-research.md),
[protocol](research/NQ_CADENCE_AREA_PROTOCOL.md), and
[data repair evidence](research/NQ_DISCOVERY_DATA_REPAIR.md).


## Portable synthetic Transformer showcase — 2026-10-15

`src/showcase.py` composes a file-backed profile in `src/webui/showcase.py` with
`src/showcase_data.py`'s checksummed fixture and MLflow import. Its console tabs
present recorded data/features, two models, synthetic replay, evaluation,
promotion and artifacts without initializing operational services. Preparation
uses existing canonical data/run path functions under an explicit output root.
The optional `src/experiment/transformer_workflow.py` path performs genuine
bounded training, calibration, comparison, evaluation, serving and scoped
synthetic-service promotion using `workflow_data/gates/tracking` and
`src/serving/workflow_model/http/promotion`. This is an experimental non-trading
example; trading-track lifecycle status is unchanged. See
[implementation and validation](examples/TRANSFORMER_WORKFLOW.md) and
[ADR-162](decisions/ADR-162-portable-transformer-showcase.md).
