# STORAGE_CONTRACT.md — Canonical Storage / Artifact Layout

**Status: CANONICAL, CURRENT.** `docs/architecture/DATA_LAYOUT.md` is a 2026-09-03 snapshot: its §2 "Present inventory" is now stale (e.g. it lists `parquet_5s/` as present-but-idle; that directory is confirmed **absent from disk** as of 2026-09-22, and its `search/` size of "14G" is `research/search/`'s current 40G), and its §4 "Target shape" (`market/`, `experiments/`, `_legacy/`) was never adopted — the reorg that actually happened kept `sources/`, `research/`, `legacy/` (§7 below explains why this document does not propose renaming them either). Treat `DATA_LAYOUT.md` §1 (design principle: data root outside git, resolver-mediated) as still correct; treat its §2/§4 as historical. This document, extended by the eighth pass (2026-09-22, "repository-wide storage and artifact architecture pass"), is the current source of truth, together with `docs/STORAGE_INVENTORY.md` for the full verified filesystem snapshot, producer/consumer tables and migration matrix that are too large to inline here.

This document covers: **Track C's actual current storage layout** (§1), the **complete producer→resolver→path→consumer map for every persistent and intermediate artifact class** (§2), **where the canonical Experiment/run-artifact concept physically lives** (§3), the **version-suffix identity gap** (§4), **logical identity vs. physical location** (§5), three distinct version concepts (§5a), confirmation that **no physical migration was performed** (§6), **canonical terminology** (§7), the **canonical target tree** (§8) and the **live/`ops/` resolution** (§8a), **filename architecture** (§9), **format rules** (§10), **partitioning rules** (§11), the **lifecycle/mutability classification** (§12), the **resolver contract and this pass's resolver changes** (§13), the **risk register** (§14, full detail in `STORAGE_INVENTORY.md` §4), a **clean-sheet self-review** (§15), **unresolved decisions / prioritized sequence** (§16), the **canonical outcome-dataset architecture** (§17), the **canonical filename/format/content contract** (§18), **Track C's semantic contract fields** (§19), and the **final authoritative canonical-syntax table** (§20).

No data was moved, rebuilt, copied or mutated to produce this document. Every path below was verified read-only against the live filesystem (`ls`/`find`/`du`) and against source, including a fresh bounded inventory pass on 2026-09-22 (seven parallel read-only inventory passes, one per major subtree, cross-checked against this document's pre-existing claims — see `STORAGE_INVENTORY.md` for method and full findings). **No database was queried or mutated.**

## 0. The resolver is `src/common/data_paths.py` — extend it, never parallel it

`src/common/data_paths.py` (675 lines, **97 importing modules**) is a mature logical-path resolver: a logical-name→path map (`_LAYOUT`, 36 keys), legacy alias resolution (`_LEGACY_ALIASES`), CLI-tolerant `resolve_data_path()`, hive partition helpers, and `_prefer()` for migration-tolerant lookup across an old and a new location.

**It is the canonical path layer.** `src/` contains no path resolution and must not acquire any. The correct canonical move for any new artifact class is to add a `_LAYOUT` key here. This reverses the second restructuring pass's direction of travel; see `docs/FUNCTIONAL_COVERAGE.md` §3.1.

**Path-status vocabulary used throughout this document:**

| Status | Meaning |
|---|---|
| `CURRENT-ACTIVE` | code writes/reads it today through a resolver |
| `CURRENT-HARDCODED` | code writes/reads it today, bypassing the resolver |
| `COMPAT` | resolved via `_LEGACY_ALIASES`/`_prefer()` for migration tolerance |
| `LEGACY` | present on disk, no live writer |
| `TARGET-UNADOPTED` | documented as a "target shape" but no code supports it |
| `PROPOSED-THIS-PASS` | introduced by this restructuring; nothing writes it yet |

A path that only exists in a document is not a path. **`experiments/` is `TARGET-UNADOPTED`** — it does not exist on disk, has no `_LAYOUT` key, appears nowhere in code, and its only source (`docs/architecture/DATA_LAYOUT.md` §4) is titled "Target shape (recommended)". Two successive restructuring passes proposed paths under it; both are retracted.

## 1. Track C storage layout (missing/stale in the existing docs)

`TREE_INVENTORY.md`'s `data/` row mentions Track C's *original* ADR-127 paths but not the versioned datasets on disk today; `DATA_LAYOUT.md`/`ML_MODEL.md`/`WORKFLOW.md` don't mention Track C at all (see `docs/ARCHITECTURE.md` §5). Verified current state:

| CONCEPT | CURRENT LOCATION(S) | RESOLVER | IDENTITY | STATUS |
|---|---|---|---|---|
| Raw print tape (tick-level, lead contract) | `data/sources/features/parquet_tape/tape_<YYYY-MM-DD>.parquet` | `_LAYOUT["parquet_tape"]` | session date | `CURRENT-ACTIVE` — already canonical |
| "Move matrix" (Stage-1-equivalent outcome+feature dataset) | `data/sources/labels/move_matrix{,_v2,_v3,_v4}/move_*.parquet` (`_v4` = 313 files, current) | **none** — hand-built | directory suffix only | `CURRENT-HARDCODED`; anti-pattern |
| "Move scores" (Stage-1-equivalent OOF) | `data/sources/labels/move_scores{,_v2,_v3,_v4}.parquet` | **none** | filename suffix only | `CURRENT-HARDCODED` |
| "Target matrix" (Stage-3-equivalent dataset) | `data/sources/labels/target_matrix{,_adaptive,_context,_context_rest,_v3,_v4}/` | **none** | **two naming semantics mixed**: `_adaptive`/`_context` are *variants*, `_v3`/`_v4` are *versions*. A fresh agent cannot tell them apart from the name | `CURRENT-HARDCODED`; worst case in repo |
| "Target scores" (Stage-3-equivalent OOF) | `data/sources/labels/target_scores{,_adaptive,_context,_context_rest,_v3,_v4}.parquet` | **none** | same mixed naming | `CURRENT-HARDCODED` |
| Stage 1/3 SHAP diagnostics | `data/evaluation/{move_analysis,context_analysis}/` (`SHAP_SUMMARY.md`, `shap_top20_*.csv`, `shap_bar_*.png`, `shap_beeswarm_*.png`) | `_LAYOUT["evaluation"]` | variant suffix in filename | `CURRENT-ACTIVE` — the closest thing Track C has to a persisted report, but it describes a *dataset variant*, not a controlled run |
| Cascade backtest output (WebUI-facing) | **none persisted** — `src/backtest/cascade_backtest.py` computes on request | n/a | n/a | **MISSING producer**; destination is defined: `<run>/economic_metrics.json` per `docs/EXPERIMENTS.md` §2.1 |
| Cost-sensitivity results | **none persisted** — console output of `scripts/features/eval_target_model.py` only | n/a | n/a | **MISSING producer**; destination is defined: `<run>/cost_sensitivity.json` per `docs/EXPERIMENTS.md` §2.1 |

**`data/sources/labels/` is the single largest resolver gap**: 30 entries, Track C's entire outcome storage plus the `firstpass` family, and **no `_LAYOUT` key**. Hand-constructed in `src/webui/datasources.py:827,837,922` and `src/gpu_runner/job_exec.py:75`; `src/training/label_store.py` and 4 Track C builder/eval scripts (`build_move_matrix.py`, `build_target_matrix.py`, `eval_move_model.py`, `eval_target_model.py`) now route through `labels_dir()` (sixth pass, path-identical). `parquet_labels_15s_dir()`'s `_prefer(layout(...), sources/labels/...)` fallback is **not** a bug -- it is the resolver's standard canonical/legacy pattern, used identically 23 other times in the same file. Classified `COMPAT`.

## 2. Full persistence / intermediate-data map

Every persistent and reusable data class in the repository. `PRODUCER` and `CONSUMER` are traced from source, not inferred from filenames.

### 2.1 Market data and feature datasets

| CONCEPT | PRODUCER | CONSUMER | PHYSICAL PATH | RESOLVER | LOGICAL IDENTITY | MUTABILITY | CANONICAL FOR NEW WORK | MIGRATION |
|---|---|---|---|---|---|---|---|---|
| Raw vendor/normalized market data | `src/ingestion/`, `src/acquisition/live_streamer.py` | TimescaleDB hypertables; `src/features/` | `data/sources/raw/`, `raw_dir()/live/<day>/`; TimescaleDB `tbbo`/`bbo_1s`/`definitions`/`statistics` | `_LAYOUT["raw"]` | symbol + date + vendor | append-only; archival | unchanged | `CURRENT-ACTIVE` |
| Feature datasets by timeframe (hive) | `src/features/engine.py`, `native_tf.py::write_month_parts` | `stores.py` (~47 call sites), all trainers | `sources/features/timeframe=/symbol=/year=/month=/features_{tf}_part*.parquet` | `native_tf_dir()` | store id → catalog → column view | **replace-in-place, destructive by default** (`native_tf.py:161-162` unlinks all existing parts first — a crashed rebuild leaves the store *empty*, not stale) | unchanged; add a write-to-temp-then-swap | `CURRENT-ACTIVE` |
| Fixed-cadence feature parquet (5s/1s/tape/options) | `src/features/` builders | trainers, WebUI | `sources/features/parquet_{5s,raw1s,path1s,tape,options_surface,options_greeks}/` | `_LAYOUT` keys | timeframe + date | replaceable | unchanged | `CURRENT-ACTIVE` |
| 1-second path / first-touch datasets (ADR-052/053) | `src/features/` path builders | `src/training/path_labels.py` | `sources/features/parquet_path1s/`, `parquet_pathlabels/` | `_LAYOUT` keys | date-partitioned | replaceable | unchanged — **this is the unclassified fourth track**; it has storage but no architectural owner beyond Population & Eligibility | `CURRENT-ACTIVE` |
| Session structural levels | `src/features/session_profile.py:170-172` | `build_move_matrix.py`, `build_target_matrix.py` | `session_levels.parquet` | hand-built | **NONE — merged newest-build-wins with no run id, version or timestamp column** | silently mutable | **must gain a build id column**; cascade's structural SSOT (`:1224`) | **BROKEN** — `open_critical_findings/session-levels-has-no-run-identity` |
| Contract-addressed label sets | `src/training/label_store.py:79-131` | `src/search/`, native_tf path | `sources/labels/{5s,15s}/<sha256[:16]>/` + `_contract.json` | `label_set_dir()` | **`label_contract_id()` = sha256 of the normalized SEMANTIC contract** — not a content hash; see §5a | idempotent by default (`:117-119` returns existing parts unless `force`); **NOT content-immutable** — `write_label_parts(..., force=True)` replaces every part beneath an unchanged `contract_id` directory, so the same key can silently name different bytes over time | **BEST IN REPO — adopt as the canonical CONTRACT-addressing pattern** | `CURRENT-ACTIVE` |
| Track C outcome/label datasets | `scripts/features/build_{move,target}_matrix.py` | `eval_{move,target}_model.py` | see §1 | none | directory suffix | overwritten per version | migrate to `label_store`-style contract addressing (§5a, §17) | `CURRENT-HARDCODED` |
| Joined training matrices / fold caches | in-memory in `src/search/controller.py` | same process | not persisted | n/a | n/a | ephemeral | leave ephemeral — persisting them would multiply storage with no reproducibility gain, since `label_contract.json` + `folds.json` already determine them | `CURRENT-ACTIVE` |

### 2.1a Canonical input contract (`DatasetRef`)

Every experiment input recorded in a manifest meets these criteria. `DatasetRef` is the *record*; resolution stays with the existing registries (`data_paths`, `stores.py`, `label_store.py`).

| # | Criterion | Enforced by |
|---|---|---|
| 1 | **Semantic type** ∈ `raw_source`, `features`, `outcomes`, `context`, `objective` | `DATASET_SEMANTIC_TYPES`; unknown values refused |
| 2 | **Logical identity** — stable `logical_id`, never a version-suffixed directory name | convention + review |
| 3 | **Physical location** — data-root-**relative**, from `data_paths.logical_path()`; absolute machine paths refused | `DatasetRef.__post_init__` |
| 4 | **Version identity**, strongest available, ≥1 required: `content_sha256` (content) > `build_id` (realization) > `contract_id` (`label_store`, semantic) > `store_id` (`stores.py`, logical only) — three genuinely different questions, not degrees of one; see §5a. **Only `build`/`content` satisfy `missing_evidence()`'s `datasets` check** (tenth pass) — `contract`-only is truthful but INCOMPLETE for a controlled comparison | `identity_strength`; none → omit the ref → missing evidence |
| 5 | **Resolution** (`15s`, `5s`, …) where it changes the dataset | `DatasetRef.resolution` |
| 6 | **No fabricated identity** — a path never substitutes for provenance | criteria 3 + 4 |
| 7 | **No duplicate vocabulary** — existing names kept (`labels`, `store_id`, `contract_id`) | review |

`store_id` is the weakest: several stores are rebuilt in place (`native_tf.py:161-162`), so it names a dataset without pinning its version.

**Resolver changes this pass (additive, path-identical):** `_LAYOUT["labels"] = "sources/labels"` plus `labels_dir()` and `logical_path()` added to `src/common/data_paths.py`; `src/training/label_store.py::label_set_dir` now resolves through `labels_dir()` (a test asserts the path is byte-identical). **Remaining gap:** the other 13+ hand-built `sources/labels` sites (`webui/datasources.py`, `gpu_runner/job_exec.py`, `scripts/features/*`) and `data_paths.py:192`'s disagreeing fallback were not touched — each is executable Track B/C code and none was needed for the manifest contract.

**Ninth pass correction (2026-09-22):** `DatasetRef` gained a fourth field, `build_id` (§5a). `src/search/run_manifest.py::_datasets_from_run_dir()` previously surfaced `label_contract.json`'s `contract_sha256` — a hash of the `_contract.json` metadata sidecar, not of the label parquet content — as `DatasetRef.content_sha256`. That was false provenance (`docs/STORAGE_INVENTORY.md` §4 STOR-11) and is fixed: `content_sha256` is now always `None` on this path (no genuine content digest is computed anywhere in Track B today), and `build_id` is derived instead from the already-recorded part inventory (`{name, bytes}` per part — a fingerprint of the realization, not the bytes themselves). See §5a for the full three-tier model.

**[CORRECTED, tenth pass]** The part-inventory `build_id` mechanism described immediately above was itself retired: a fingerprint of OUTPUT filenames/byte sizes is not genuine build/provenance evidence (it says nothing about which code, config or upstream data produced the realization). `src/search/run_manifest.py::_datasets_from_run_dir()` now always records `build_id=None` (no genuine build-process evidence is available on that adapter's path), and `contract_id`-only `DatasetRef`s now correctly count as missing evidence for `datasets` rather than complete. Genuine `build_id`s are populated by `label_store.write_build_sidecar()` (§17, §18), from code identity + builder config + upstream lineage. See §5a's current text for the corrected model.

### 2.2 Run artifacts (research plane)

**Present (2026-09-23).** New research runs, including Track B search and Optuna HPO, write `research/runs/<objective_id>/<run_ts>/` via `run_layout.new_run_dir()`. `data/research/custom_test/` is gone. `research/search/` remains a compatibility root for runs already there; it is not the new-run root and it is not the HPO layout. Rows below that still name `custom_test_dir()` describe the pre-migration layout.

| CONCEPT | PRODUCER | CONSUMER | PHYSICAL PATH | RESOLVER | LOGICAL IDENTITY | MUTABILITY | CANONICAL FOR NEW WORK | MIGRATION |
|---|---|---|---|---|---|---|---|---|
| Candidate search run dir | `src/search/controller.py` (`new_run_dir`) | `src/search/finalize.py`, WebUI, leaderboard | `research/runs/<objective_id>/<run_ts>/` | `runs_dir()` | objective + run timestamp | append-only per run | **new-run root** | `CURRENT-ACTIVE` |
| Strategy/backtest run dir | `src/backtest/runner.py` | `src/backtest/archetypes.py` | `research/search/<objective_id>/strategy_<sid>_<ts>/` | `search_dir()` | objective + strategy + ts | append-only | same contract as above | `CURRENT-ACTIVE` |
| Run record | `controller.py:504` | WebUI, finalize | `<run>/run.json` | — | — | written once | **superseded by `manifest.json`** — `run.json` exists on disk in **two incompatible shapes with no version field** | `CURRENT-ACTIVE`, to be complemented |
| Fold schedule | search engines | eval, comparison | `<run>/folds.json` | — | `split` + `n_folds` | written once | retained; mirrored into `manifest.validation` | `CURRENT-ACTIVE` |
| Resolved spec / topology | `controller.py` | reproduction | `<run>/spec.manual.yaml`, `topology.yaml` | — | — | written once | retained; mirrored into `manifest.config_snapshot` | `CURRENT-ACTIVE` |
| OOF predictions | `controller.py`, `backtest/runner.py:143` | `archetypes.py`, `runner.py` replay | `<run>/oof_<model>.parquet` | — | model id within run | immutable | **retain the existing column contract `ts/direction/confidence/direction_true`** (`runner.py:14`) | `CURRENT-ACTIVE` |
| Dataset lineage snapshot | `label_store.py:138-155` | reproduction | `<run>/label_contract.json` | — | contract sha256 + part inventory | immutable | retained; mirrored into `manifest.datasets` | `CURRENT-ACTIVE` |
| Per-model diagnostics | `controller.py` | WebUI | `<run>/diagnostics_<model>.json` | — | model id | written once | folded into `metrics.json` | `CURRENT-ACTIVE` |
| In-flight progress | `src/training/job_progress.py` | WebUI | `<run>/progress[_<model>].json` | — | run + model | **mutated during the run** | operational state, not an experiment artifact — keep separate | `CURRENT-ACTIVE` |
| Feature importance / SHAP | `src/evaluation/shap_explain.py:595` | humans, `docs/` | `<run>/SHAP/{shap_importance.csv,shap_groups.csv,shap_meta.json}` | — | run + model | immutable | **reuse as canonical — do not re-specify** | `CURRENT-ACTIVE` |
| Trained models / checkpoints | trainers | promote, serving | `<run>/models/<candidate>/`; `models/training_runs/`; `legacy/checkpoints/` | `training_runs_dir()`, `_LAYOUT["checkpoints"]` | candidate id | immutable per run | unchanged | `CURRENT-ACTIVE` / `LEGACY` |
| Run manifest | `src/search/controller.py` via `src/search/run_manifest.py` (Track B, additive) and `RunManifest.merge_pass` (objective runs) | `cli run-inspect`, `run-compare`, objective-run readers | `<run>/manifest.json` | `custom_test_dir()` / `runs_dir()` | **`RunManifest.run_id` + `scientific_fingerprint()`** | **scientifically immutable** once a field is set; an empty field may be filled; pass inventory may grow | `CURRENT-ACTIVE` | **WIRED.** Track B still writes it beside `run.json`. Objective runs use it as the only run-level file (schema v4) |
| Metrics / fold metrics | partially `search_result.json` | comparison | `<run>/{metrics.json,fold_metrics.json}` | `custom_test_dir()` | run id | immutable | `schemas.FoldMetrics` | `PROPOSED-THIS-PASS` |
| Selected population | *(nothing)* | comparison | `<run>/selected_population.json` | `custom_test_dir()` | run id | immutable | `schemas.SelectedPopulation` — **closes the `cascade-population-shift` gap** | `PROPOSED-THIS-PASS` |
| Calibration | *(nothing — absent from both tracks)* | decision policy | `<run>/calibration.json` | `custom_test_dir()` | run id | immutable | `schemas.CalibrationResult` | `PROPOSED-THIS-PASS` |
| Trade ledger | `backtest/runner.py` | KPI, reports | `<run>/trades.parquet` | `search_dir()` | run id + trade id | immutable | retained; `TradeLedger` binds trades+economics+assumptions | `CURRENT-ACTIVE` |
| Economic results / cost sensitivity | `src/evaluation/economic.py` via `portfolio_sim` and `eval_target_model.py` | comparison | `<run>/{economic_metrics.json,cost_sensitivity.json}` | `runs_dir()` | run id + `EconomicAssumptions.label` | immutable | **assumptions persisted alongside results, never separated** | `CURRENT-ACTIVE` for those two callers; other PnL paths remain separate |
| Run report | *(nothing standardized)* | humans | `<run>/report.md` | `custom_test_dir()` | run id | immutable | closes the "conclusions live only in console output and chat" failure mode | `PROPOSED-THIS-PASS` |

### 2.3 Model lifecycle

| CONCEPT | PRODUCER | CONSUMER | PHYSICAL PATH | RESOLVER | LOGICAL IDENTITY | MUTABILITY | MIGRATION |
|---|---|---|---|---|---|---|---|
| Leaderboard | `src/evaluation/leaderboard.py:82-97,155,160` | promotion, WebUI | `models/leaders/` | `leaders_dir()` | `candidate_id` (+ frequently-null `mlflow_run_id`) | **rows mutate via max-precision dedupe** | **BROKEN** — writes no `run_ts`, so a row cannot be traced to its run dir, and an older run can permanently mask a newer one (`open_critical_findings/leaderboard-rows-not-traceable-to-runs`) |
| Frozen eval contract | `src/evaluation/contract.py:55` | `_kpi.py`, controller | `evaluation/<objective>/contract_v1.json` | `evaluation_dir()` | objective + contract version | frozen by intent | **BROKEN** — `objective_snapshot` holds only `['note','hold_out_source','primary']`, so `_kpi.py:34-39` defaults `mae_tolerance=5` against a declared 6-pt stop; the hold-out hash is never checked (`score()` enforces it only when `data_hash is not None`, and `controller._record_all` never passes it) |
| Staging models | `src/serving/promote.py:58,83` | promotion gate | `models/model_staging/` | `model_staging_dir()` | candidate id | staged then moved | `CURRENT-ACTIVE` |
| Production registry | `src/serving/promote.py:127,206` | serving | `models/model_registry/<obj>/LATEST.json` | `model_registry_dir()` | objective + champion id | pointer mutated on promotion | `CURRENT-ACTIVE`; lifecycle vocabulary (`src/serving/lifecycle.py`) not yet enforced here |
| Episodes | `src/evaluation/episodes.py:53` | evaluation | `research/episodes/` | `episodes_dir()` | episode id | append-only | `CURRENT-ACTIVE` |

### 2.4 Platform plane

| CONCEPT | PRODUCER | CONSUMER | PHYSICAL PATH | RESOLVER | MUTABILITY | MIGRATION |
|---|---|---|---|---|---|---|
| MLflow tracking DB | `mlflow` via 8 `set_tracking_uri` sites | MLflow UI, `mlflow_backfill.py` | `data/mlflow.db`, `data/ops/mlruns/` | **none** | mutable | `CURRENT-HARDCODED`; logging is best-effort with swallowed exceptions, so `mlflow_run_id` is frequently null |
| MLflow artifacts | same | same | `data/ops/mlflow_artifacts/` **and** `data/mlflow_artifacts/` (both populated, both receiving writes) | `mlflow_artifacts_dir()` via `_prefer()` + `_LEGACY_ALIASES` -- but **no code calls `mlflow_artifacts_dir()` at all**; MLflow itself pins `artifact_location` per-experiment in its Postgres backend store at experiment-creation time and never consults the resolver | append-only | `CURRENT-ACTIVE` + `COMPAT`, **confirmed split-brain, not merely handled** (eighth pass): the top-level dir's newest write is 2026-09-17, five days before this pass, for the pre-reorg default experiment (`nq-50pt-intraday`) whose `artifact_location` was pinned before the `--default-artifact-root` flag was repointed at `ops/`. Fix requires an MLflow admin action (repoint or recreate that experiment's `artifact_location`), not a code or doc change -- out of scope for a no-DB-writes pass. See `docs/STORAGE_INVENTORY.md` §4 STOR-3 |
| Optuna HPO studies | `scripts/search/hpo_model_search.py` | HPO | `research/runs/<objective_id>/<hpo_run_ts>/` (`hpo.json`, `trials_summary.parquet`, per-trial `model.json`). Legacy `scripts/search/hpo_search.py` still has `data/optuna.db` | `runs_dir()` | mutable study files under the run | `CURRENT-ACTIVE` for `hpo_model_search.py`; `optuna.db` is legacy |
| Scoping / Research Director outputs | `src/scoping/{runner,journal,rag}.py`, `src/research_director/{orchestrator,tools}.py` | RD gate, humans | `research/scoping/`, `research/research_director/` | `scoping_dir()`, `research_director_dir()` | append-only | `CURRENT-ACTIVE` |
| GPU job state / health | `src/gpu_runner/{runner,job_exec,dispatch}.py` | dispatcher, WebUI | `ops/gpu_jobs/`, `ops/gpu_health/` | `gpu_jobs_dir()`, `gpu_health_dir()` | mutable operational state | `CURRENT-ACTIVE` |
| Generic job state | `src/webui/jobs.py` | WebUI (only job-state store; survives restarts) | `ops/job_state/` (28 entries, incl. `.bak-repair3` files) | **none** | mutable | `CURRENT-HARDCODED` |
| Paper trading state | `src/paper/{ledger,policy}.py` | paper engine | `ops/paper/` | `paper_dir()` | append-only ledger | `CURRENT-ACTIVE`; inactive since 2026-08-05 |
| Execution state / events | `src/execution/{ledger,config}.py` | gateway | `ops/execution/` | `execution_dir()` | append-only | `CURRENT-ACTIVE` |
| **Live signal contract** | `src/webui/live.py:1798-1890` | `src/features/live_runtime.py`, `src/execution/gateway.py`, `src/execution/config.py` | `data/live/signals.jsonl`, `data/live/heartbeat.json` | `live_dir()` (`_LAYOUT["live"]`, **added this pass** -- `webui/live.py`, `webui/regmodel.py` and `execution/config.py` each independently hardcoded `data_root()/"live"` before; all three now route through the resolver, path-identical) | append-only | `CURRENT-ACTIVE` (was `CURRENT-HARDCODED`, H13, fixed eighth pass); the serving↔WebUI coupling itself is unchanged, tracked in `open_critical_findings/serving-depends-on-webui` |
| Feature catalog | `src/features/stores.py:23-24,62-67` | runtime store resolution | `docs/reference/FEATURE_CATALOG_*.md`, `data/exports/feature_catalog.csv` | **none** | regenerated | `CURRENT-HARDCODED` — **documentation on the runtime path**; a doc edit can change runtime behavior |
| Data catalog | ingestion | WebUI | `ops/catalog/` | `catalog_dir()` | mutable | `CURRENT-ACTIVE` |
| Logs | 92 `basicConfig` users; `ops/logs/` | humans | `ops/logs/`, `/tmp/*.log` | **none** | rotating | `CURRENT-HARDCODED` — Track C's run evidence lives in `/tmp`, outside `data/` entirely |
| Scratch / temp | various | various | `ops/tmp/` | `_LAYOUT["tmp"]` | ephemeral | `CURRENT-ACTIVE` |
| Quarantine | ingestion | humans | `ops/quarantine/` | **none** | append-only | `CURRENT-HARDCODED` |
| Archival / legacy | — | — | `legacy/{checkpoints,hpo,research_runs,features_hive}` | `_LAYOUT` keys | frozen | `LEGACY` (`features_hive` removed from disk; key retained for tools that still look) |
| Non-artifact roots | — | — | `data/postgresql/`, `data/swapfile`, `data/mbp10_job.json` | n/a | n/a | out of scope — infrastructure, not research artifacts |

### 2.5 Hardcoded paths bypassing the resolver (carried from `docs/FUNCTIONAL_COVERAGE.md` §3.4)

| ID | Location | Issue |
|---|---|---|
| H1 | `src/training/config.py:190` | `_load_yaml(config_path or Path("/opt/vtech/config.yaml"))` — absolute machine path in the config loader. **23 importers. Highest blast radius.** |
| H2–H4 | `trainer.py:405`, `xgb_trainer.py:219`, `stack_trainer.py:342` | `Path(save_dir or "checkpoints/latest")` — resolves against process CWD |
| H5 | `stack_trainer.py:569,578,587,600` | CLI defaults pointing at ARCHIVED `legacy/checkpoints/model_v015_fold*` |
| H9 | `engine.py:205`, `trainer.py:449`, `xgb_trainer.py:221`, `stack_trainer.py:344` | `os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")` re-implemented 4× — `data_paths.data_root()` exists for exactly this |
| H10 | `src/features/stores.py:23-24,62-67` | builds `docs/reference/FEATURE_CATALOG_*.md` paths — documentation on the runtime path |
| H11 | `src/training/data_pipeline.py:151,169` | `date=`-partitioned snapshot layout defined nowhere else, no resolver |
| H12 | `feat_ext.py:58`, `hive_5s.py:47`, `data_pipeline.py:77,88`, `stores.py:78,159` | the `features_*.parquet` glob re-implemented in 5 places |
| H13 | ~~`src/webui/regmodel.py:653`~~ | ~~`data_root()/"live"/"signals.jsonl"` — the live signal contract has no `_LAYOUT` key~~ **FIXED eighth pass**: `_LAYOUT["live"]` + `live_dir()` added; `regmodel.py`, `webui/live.py`, `execution/config.py` all now route through it |
| H14 | ~~`src/features/session_profile.py:57-58`~~ | ~~`session_profile_dir()` hardcoded `(root or data_root())/"sources"/"features"/"session_profile"` — feeds the cascade's structural SSOT (`session_levels.parquet`) and had no `_LAYOUT` key at all~~ **FIXED eighth pass**: `_LAYOUT["session_profile"]` added; the default (`root=None`) branch now resolves through `layout()`, path-identical. The `root` override branch (used by tests) is unchanged |
| H15 | ~~`scripts/search/regime_analysis.py:21,86`~~ | ~~`Path("/opt/vtech")/"data/research/research_director/regimes"` — a fully hardcoded absolute path, doesn't even honour `VTECH_DATA_DIR`~~ **FIXED eighth pass**: now `research_director_dir()/"regimes"`, path-identical under the default root. (Note: the same script's line 39 also hardcodes `REPO/"data/sources/features/parquet_5s"` as a fold-cache input — that path is confirmed absent from disk (§2.1 above); left unfixed, since choosing a replacement store is a model-input decision, not a pure path-resolver substitution) |

Repo-wide, **59 files outside `data_paths.py` reference `/opt/vtech/data` literally** (count as of the sixth pass; three call sites closed this pass, see H13–H15).

## 3. Where the Experiment/run-artifact concept physically lives

The directory **leaf is `run_ts`** (physical location) — not `run_id` (logical identity, `docs/EXPERIMENTS.md` §1.2); for Track B, `run_id = <objective_id>/<run_ts>`:

```
$VTECH_DATA_DIR/research/runs/<objective_id>/<run_ts>/
```

resolved through `run_layout.runs_dir()` / `new_run_dir()`. `research/search/<objective_id>/<run_ts>/` is a compatibility root only. `research/custom_test/` has been migrated and removed. A new run's `manifest.json` is the run-level record. Older runs may still carry `run.json`, `folds.json`, and `search_result.json` beside it.

No directory was created preemptively for the `PROPOSED-THIS-PASS` artifacts, consistent with "don't create storage structure for a responsibility that isn't wired yet."

## 4. The version-suffix identity gap (cross-cutting)

Every Track C dataset in §1 uses a version-suffixed directory or filename as its only identity. The *scientific* consequence is in `docs/EXPERIMENTS.md` §1. The *storage* consequence: **there is no way to tell from the filesystem which `move_matrix_v4`/`target_matrix_v4` pairing is mutually consistent** — i.e. that `target_matrix_v4` was built from `move_scores_v4.parquet` and not an earlier version — without reading a build log. Until `manifest.datasets` is wired, treat any `_v4` dataset's upstream lineage as **not self-describing**.

## 5. Logical identity vs. physical location — the rule

**Structured metadata is authoritative identity; physical location aids discoverability only.** A run's identity is `RunManifest.run_id` plus its `scientific_fingerprint()`, not the directory it happens to sit in. A dataset's SEMANTIC identity is `label_contract_id()`, not its directory suffix — but semantic identity is not the whole of "identity"; see §5a immediately below for what it does and does not establish. Track B partially achieves this (objective-keyed paths are real logical identity; `<run_ts>` directory names are still location-as-identity); Track C does not achieve it at all.

## 5a. Three distinct version concepts

The architecture previously blurred three genuinely different questions under one word, "identity" or "content-addressed." A **ninth-pass correction** (2026-09-22) formalizes them, because the blur produced a real bug: `label_contract.json`'s `contract_sha256` (a hash of contract METADATA) was recorded as `DatasetRef.content_sha256` (which claims to be a hash of dataset CONTENT) — see `docs/STORAGE_INVENTORY.md` §4 STOR-11 for the full defect and its fix.

| Question | Name | Existing mechanism | What it does NOT establish |
|---|---|---|---|
| **What should this dataset MEAN?** | Semantic contract identity | `label_store.label_contract_id()` (Track B) / `label_store.outcome_contract_id()` (track-neutral, §17) — sha256 of a normalized contract dict: label rule, horizon, stop/target semantics, resolution, configured build semantics | Whether two realizations sharing this id have the same BYTES. `write_label_parts(..., force=True)` can replace every part beneath an unchanged `contract_id` directory — the directory is idempotent-by-default, not content-immutable. **Not strong enough alone to certify a controlled comparison** (tenth-pass correction) — a `contract_id`-only `DatasetRef` is truthful and constructible but counted as missing evidence by `RunManifest.missing_evidence()`, since the same contract_id can silently name different bytes over time |
| **Which deterministic build produced THIS realization?** | Build / provenance identity | `DatasetRef.build_id` — populated ONLY from genuine build-process evidence: code revision (`src.common.code_identity.git_identity()`), effective builder config, upstream source-dataset version references — recorded in `label_store.write_build_sidecar()`'s `_build.json` (§17, §18) | The actual dataset bytes — it is a fingerprint of build-time PROCESS evidence (what code, what config, what inputs), not of the parquet content itself. **Tenth-pass correction**: the ninth pass populated this field from a fingerprint of the recorded OUTPUT part inventory (filenames + byte sizes) instead. That was retired — a directory listing not changing says nothing about which code, config or upstream data produced the bytes, a materially weaker and differently-shaped claim than genuine build identity. `DatasetRef.build_id` is `None` wherever genuine process evidence is unavailable, never filled from output metadata |
| **What are the actual materialized BYTES?** | Content identity | `DatasetRef.content_sha256` | Nothing is currently computed as a genuine content digest anywhere in Track B or Track C. This field is `None` everywhere in this codebase today, and stays `None` rather than being filled with a substitute — hashing existing multi-GB datasets is deliberately out of scope for a documentation/resolver pass |

**Ranking** (`DatasetRef.identity_strength`): `content` > `build` > `contract` > `store`. `store_id` remains the weakest — it is not a version concept at all, only a mutable logical pointer (`native_tf.py:161-162` rebuilds several stores in place). **Only `build` or `content` identity satisfies `RunManifest.missing_evidence()`'s `datasets` check** (tenth pass) — `contract`-only and `store`-only are both real, truthful evidence, just not strong enough to certify a controlled comparison on their own, since neither pins WHICH realization was used.

**Rule:** never rename a contract hash, a part-inventory/output-metadata fingerprint, or any other non-build, non-content digest into `build_id` or `content_sha256` to make evidence look stronger. If no genuine build-process evidence exists, leave `build_id=None`; if no genuine content digest exists, leave `content_sha256=None`; record the strongest tier that is genuinely available (`build_id`, then `contract_id`, then `store_id`) — an absent or weaker claim is truthful; a promoted one is not.

## 6. No physical migration performed

Nothing above was moved, copied, rebuilt, renamed or deleted. This document maps current-state-as-verified to target-state-as-designed, for a human or future agent to execute later.

---

## 7. Canonical terminology

One term per durable concept. Where an existing repository term is already correct, it is kept — this section maps ambiguous *current* usage onto it, it does not invent new words for their own sake.

| Term | Meaning | Notes |
|---|---|---|
| **dataset** | Any durable, reusable tabular artifact addressed by logical identity + version identity | Umbrella term; always qualified below |
| **source dataset** | Raw or normalized vendor market data (`sources/raw/`) | Not "raw data" alone — "raw" is also used loosely for pre-feature-engineering data; this pass keeps `sources/raw` as the physical name (already precise) and uses "source dataset" only in prose |
| **feature dataset** | Decision-time state representation (`sources/features/`) | Includes the native-TF hive and the flat caches (tape, path1s, options surface) |
| **outcome dataset** | Labels / outcome data (`sources/labels/`) | Covers both Track B's contract-addressed sets (§5a) and Track C's move/target matrices — one term for both, per §9 of the task's own instruction not to fork Track B/C vocabulary; unified new-write pattern in §17 |
| **context dataset** | Reference/structural data feeding decisions but not itself a feature or outcome (e.g. `session_profile`, `session_levels.parquet`) | Matches `DatasetRef.semantic_type = "context"`, already defined in `manifest.py` |
| **dataset store** | A *logical*, potentially-rebuilt-in-place location for a dataset (`store_id` in `stores.py`) | Names WHICH dataset, not WHICH version — see §5 |
| **dataset contract** | A normalized, hashed semantic definition of what a dataset means (`label_contract_id()`) | The strongest non-content identity mechanism in the repo |
| **run artifact** | Anything written into `research/<workflow>/<objective_id>/<run_ts>/` | One run root, all workflows (search, custom_test) |
| **model artifact** | A trained model file/bundle (`model.ubj`, `model.keras`, etc.) inside a serving bundle | Never used for the bundle itself — see "serving bundle" |
| **serving bundle** | A directory of everything needed to load and run one trained model (`models/model_staging/`, `models/model_registry/`) | Distinct from a bare model artifact; a bundle contains one or more model artifacts plus schema/scaler/manifest files |
| **cache** | A derived, fully reproducible artifact whose loss costs recompute time only, never scientific evidence | `_fold_cache/`, `native_tf` hive parts (rebuildable from raw + code), `ops/tmp` (in principle) |
| **operational state** | Mutable, platform-plane data describing the *current* state of a running system, not a scientific result | `ops/gpu_jobs/`, `ops/job_state/`, `ops/webui/jobs/`, `live/`, `ops/paper/` |
| **temporary/scratch** | Genuinely ephemeral, safe to delete at any time with no durable consequence | `ops/tmp/` *by design* (§14 notes it is not honoring this today) |
| **legacy/archive** | Frozen, pre-reorg or superseded content with no live writer (or a documented, isolated exception) | `legacy/*`, the `firstpass` family under `sources/labels/` |

**Track C terminology correction (carried forward, re-verified this pass):** the underlying market/data is always **NQ**; economic sizing is **MNQ-equivalent** (`config/execution_costs.yaml`'s `MNQ_equivalent` profile: 0.1× NQ's $20/pt multiplier, an independent $1.03 commission). `src/backtest/cascade_backtest.py`'s `run_cascade_inmemory(instrument="MNQ", ...)` parameter and output field repeated exactly the anti-pattern `EconomicAssumptions.economic_contract_unit`'s docstring already warns against ("wrongly implied ... that an MNQ contract is bought"). **Fixed this pass**: renamed to `economic_contract_unit`, default `"MNQ_equivalent"`, matching the established canonical field name; verified as the label's only use (assembled into an output payload, never used in any calculation) and its only caller (`src/webui/datasources.py:1199`, which does not pass it explicitly).

No new aliases were created for elegance. `run_id` and `run_ts` (existing terms) remain the only two identifiers for a run; no third one is introduced, per the task's own instruction.

## 8. Canonical target tree

**Top-level roots are not renamed.** `docs/architecture/DATA_LAYOUT.md` §4 proposed `market/`, `experiments/`, `_legacy/`; that proposal was never adopted (`docs/FUNCTIONAL_COVERAGE.md` §5.2 already retracted `experiments/` specifically). The reorg that *did* happen — `sources/`, `research/`, `models/`, `evaluation/`, `ops/`, `legacy/` — already satisfies the task's own test ("what semantic distinction does this directory encode, and why should it exist physically rather than only in metadata?"): each root is a genuine plane/lifecycle boundary (input data vs. run outputs vs. serving lifecycle vs. frozen contracts vs. platform state vs. archive), each already has resolver ownership, and renaming ~108G of directories for vocabulary alone would satisfy none of the task's justification criteria (ownership, lifecycle, discoverability, partitioning, access pattern, retention, reproducibility, resolver behavior — a rename changes none of them). This section therefore documents the **existing top-level roots as canonical**, and specifies resolver ownership only for the subordinate patterns this pass found undocumented or ambiguous.

| Semantic class | Owner (resolver) | Path pattern | Required dimensions | Optional dimensions | Identity in path? | Identity in metadata? | Mutability | Partitioning |
|---|---|---|---|---|---|---|---|---|
| Source data (raw) | `raw_dir()` | `sources/raw/{live,backfill}/...` | lifecycle (live/backfill), content class | session date, job_id | session date (discoverability only) | none today (gap, STOR-5) | append-only per file | by content class, then date/job |
| Feature dataset (hive) | `native_tf_dir()` | `sources/features/timeframe={tf}/symbol=/year=/month=/` | timeframe, symbol, year, month | part number | timeframe/symbol/year/month (all real partitioning, not identity) | store id (mutable — §5) | destructive rebuild per month part | hive: timeframe → symbol → year → month |
| Feature dataset (flat cache) | `parquet_tape_dir()`, `parquet_path1s_dir()`, etc. | `sources/features/<name>/` | dataset name | session date | none | none (gap for `parquet_raw1s`, `session_profile` before this pass) | replaceable per session file | flat, one file per session |
| Outcome dataset (Track B) | `label_store.label_set_dir()` | `sources/labels/{5s,15s}/<contract_id>/` | resolution, contract hash | — | **contract hash — semantic identity only (§5a); NOT content-immutable, `force=True` can replace parts beneath it** | `build_id` (part-inventory fingerprint, §5a) when a `label_contract.json` snapshot exists | idempotent by default, not content-immutable | resolution → contract hash |
| Outcome dataset (Track C) | none (gap, STOR-4) | `sources/labels/{move,target}_matrix<suffix>/` | — | — | **directory suffix (anti-pattern)** | none (gap) | overwritten per version | resolution implicit, no formal partitioning |
| Run artifact root | `custom_test_dir()`, `search_dir()` | `research/<workflow>/<objective_id>/<run_ts>/` | workflow, objective_id, run_ts | — | objective_id (real), run_ts (location, not identity — `run_id` is metadata) | `RunManifest.run_id` + `scientific_fingerprint()` | append-only per run | objective_id → run_ts |
| Reproducible cache | `search_dir() / "_fold_cache"` (no dedicated key — acceptable, see §13) | `research/search/_fold_cache/<input-hash>/` | input hash | — | input hash (correct, cache-key semantics) | — | write-only, unbounded (STOR-1) | flat by hash |
| Serving bundle | `model_staging_dir()`, `model_registry_dir()` | `models/{model_staging,model_registry}/<objective_id>/<candidate>_<run_ts>/` | objective_id, candidate/run_ts | — | objective_id + candidate/run_ts | `finalize_manifest.json` fields (not yet `RunManifest`-linked, STOR-7) | copy-once by convention, not enforced | objective_id → candidate |
| Leaderboard | `leaders_dir()` | `models/leaders/<objective_id>/` | objective_id | contract_version | objective_id | candidate_id (weak — no FK to a serving bundle, STOR-7) | append-only (`leaderboard_v*.jsonl`) + mutable pointer (`latest_leaders.json`) | objective_id |
| Frozen eval contract | `evaluation_dir()` | `evaluation/<objective_id>/contract_v<N>.json` | objective_id, version | — | objective_id + version | hold-out hash | frozen by intent | objective_id |
| Platform operational state | `gpu_jobs_dir()`, `webui_dir()`, `paper_dir()`, `execution_dir()`, etc. | `ops/<subsystem>/...` | subsystem | run_id / objective_id where applicable | subsystem-specific | subsystem-specific | mutable, live | subsystem |
| Platform operational state — **documented exception** | `live_dir()` | top-level `live/` (current, stable — no auto-cutover) — `ops/live/` (semantic target, requires an explicit future migration, §8a) | — | — | — | — | mutable, live (append-only signal stream) | flat |
| Legacy/archive | `checkpoints_dir()`, `_LAYOUT["research_runs"]`, `_LAYOUT["training_runs"]` (legacy variant) | `legacy/{checkpoints,model_staging,research_runs,training_runs}/` | none required | — | frozen snapshot name | none | frozen (with named exceptions — STOR-10) | none |

## 8a. Live-signal architecture — resolving the `ops/` contradiction

The eighth pass simultaneously said (a) operational state canonically belongs under `ops/<subsystem>/`, and (b) added `_LAYOUT["live"] = "live"` pointing at the *top-level* `live/` directory — two statements that cannot both be canonical. Resolved this pass (tenth pass, refining the ninth pass's own resolution):

**Decision: `ops/live/` is the correct canonical FUTURE target by the same rule every other operational-state class already follows** (§8's table) — `live/`'s actively-written, append-only, mutable signal stream is the same lifecycle class as `ops/execution/`, `ops/paper/`, `ops/gpu_jobs/`. The precedent that settles this: `ops/execution/bridge/` is *already* the established pattern for an external-facing file bridge (`docs/architecture/DATA_LAYOUT.md` §1.1's `VTECH_EXECUTION_BRIDGE_DIR`, meant to be mounted for a broker/SMB share) and it lives under `ops/`, not the data root — so "this directory is read by external, non-Python broker tooling (NinjaScript / cAlgo, per `scripts/ops/tail_live_signals.py`'s docstring)" does not, by itself, justify a top-level exception; the codebase's own execution-bridge design already proves external consumption and `ops/`-nesting are compatible.

**Adopting the target requires an explicit cutover, not a filesystem probe (tenth-pass correction).** The ninth pass had `live_dir()` auto-prefer `ops/live/` the instant it was physically populated, reasoning that a future migration would then require no caller change. That auto-cutover is reverted: whether the live-signal transport has actually switched directories is an operational fact (has every external consumer — NinjaScript/cAlgo, `scripts/ops/tail_live_signals.py` — been repointed? is anything still writing/reading the old path?) that a resolver cannot safely infer merely from a directory existing. A stray `mkdir ops/live` (an operator's typo, a test fixture, a half-finished migration attempt) must not silently redirect the live signal contract.

**What this pass does, given no physical migration is permitted:**

1. `live_dir()` (`src/common/data_paths.py`) always returns `layout("live")` — the current, active top-level path. It does **not** check for or prefer `ops/live/` under any condition. `ops/live/` remains the documented FUTURE canonical target; adopting it is a dedicated future code change (this resolver function is updated deliberately, not automatically), not something that happens by populating a directory.
2. `scripts/ops/tail_live_signals.py`, missed by the eighth pass's H13 sweep, hardcoded a fourth independent `data_root() / "live"` construction — routed through `live_dir()` too (§13), unaffected by this pass's reversion (still resolves to the same top-level path).
3. §8's table marks `live_dir()` as a **documented exception**, not a silent contradiction: current, stable physical location top-level `live/` (no auto-cutover); semantic/future target `ops/live/`, migration deferred to an explicit future pass (§16).

No files moved. The contradiction was in the documentation asserting two canonical claims at once, not in the running system.

## 9. Filename architecture

Rules derived from what already exists and works, made explicit:

1. **A canonical filename describes semantic content, never implementation history.** `oof_predictions.parquet`, `metrics.json`, `manifest.json` (existing, `artifacts.py`) are correct; `move_matrix_v4/`, `target_matrix_context_rest/` are the anti-pattern this pass documents but does not fix (STOR-4).
2. **A version suffix is never canonical identity.** Where a version must be visible for human discoverability (e.g. an eval contract's `contract_v<N>.json`), the version is an explicit, incrementing, single-purpose integer tied to one axis of change — not a free-form suffix that silently mixes "different build" with "different variant" (the exact failure in `target_matrix_v3`/`target_matrix_context`).
3. **Identity belongs in metadata (a sidecar or manifest field), not the filename, once a dataset has more than one axis of change.** Track B's `_contract.json` sidecar is the model to copy; a filename-only scheme cannot represent two independent axes (version × variant) without ambiguity, which is exactly what happened to `target_matrix`.
4. **Directory partitions carry real partitioning dimensions only** (symbol, timeframe, year, month, objective_id, run_ts) — never a stage number, model family, or temporary experiment label (already enforced by `artifacts.py`'s filename-criteria comment and tests for run artifacts; this pass finds no counterexample in the run-artifact tree, only in the outcome-dataset tree).
5. **One canonical name per semantic class**, with existing large files referenced from a manifest rather than renamed or duplicated (`artifacts.py`'s `COMPATIBILITY_ARTIFACTS` pattern — already the right idiom, reused rather than re-invented).

## 10. Canonical formats

| Class | Format | Why |
|---|---|---|
| Large typed tabular data (features, outcomes, OOF, trade ledgers) | Parquet | Already universal in the repo; columnar, typed, splittable — correct, unchanged |
| Small machine-readable contracts, manifests, results | JSON | Already universal for `manifest.json`, `*_contract.json`, `metrics.json` — correct, unchanged |
| Human-authored configuration | YAML | `config/objectives/*.yaml`, `spec.manual.yaml`, `topology.yaml` — correct, unchanged |
| Append-only event streams | JSONL | `signals.jsonl`, `leaderboard_v*.jsonl`, `trades.jsonl` — correct, already the pattern in use |
| Trained model artifacts | Model-native binary (`.ubj`, `.keras`, `.pkl` for scalers) | Already the pattern; no case found for a universal serialization format |
| Human reports | Markdown | `report.md` (proposed, not yet produced), `SHAP_SUMMARY.md` — correct; never authoritative runtime state, and no case found where it is used as one |
| Numeric cache arrays | `.npy` | `_fold_cache/`'s `X.npy`/`y.npy`/`ts.npy` — appropriate for a pure numeric cache with a JSON `meta.json` sidecar for the (non-numeric) key |

**Formats already appropriate, confirmed this pass, no change proposed.** The one material problem found is not a format problem — it is STOR-4 (Track C identity), which no format change would fix.

## 11. Partitioning rules

- **Hive-partition by symbol → timeframe → year → month** for feature datasets with a genuine multi-year time axis and a small symbol set (`native_tf_dir()` — already correct, confirmed this pass across all 6 timeframes despite a stale docstring claiming only 5s/15s use monthly parts; verified all six do).
- **Partition by session date, flat, one file per session** for datasets whose natural grain is one trading session and whose total file count stays in the low thousands (`parquet_tape`, `parquet_path1s`, `session_profile` — correct, unchanged; `parquet_path1s` alone holds 3,919 files spanning back to 2011 with no discoverability problem).
- **Partition by content/contract hash, flat, one directory per version** for outcome datasets where mutual consistency between input and output must be verifiable without a build log (`label_store.py` — the target pattern for Track C, not executed this pass).
- **Partition by objective_id → run_ts** for run artifacts — the one partitioning scheme every workflow (search, custom_test, and once wired, Track C) should share, and already does.
- **Do not partition operational/cache state by anything but its own key** — `_fold_cache/<input-hash>/` and `ops/gpu_jobs/<run_id>/` are correctly flat-by-key; adding objective_id/date partitioning to either would only fragment lookups that are already O(1) by key.
- **Avoid giant monolithic files**: confirmed not a problem anywhere in this inventory — the native-TF hive already splits at ~100MB (`NATIVE_TF_PART_MAX_BYTES`), and no single-file class exceeds a few GB.
- **Avoid uncontrolled tiny-file proliferation**: the one active risk is `_fold_cache/`'s unbounded key growth (STOR-1) — not a partitioning-scheme problem, an eviction problem, correctly out of this pass's scope.

## 12. Lifecycle / mutability classification

Every major class in §2 of this document and §2/§4 of `STORAGE_INVENTORY.md` falls into exactly one of:

| Classification | Examples | Backup/sync implication |
|---|---|---|
| Immutable durable scientific data | `sources/labels/{5s,15s}/<contract>/`, `evaluation/<obj>/contract_v*.json`, `<run>/oof_*.parquet` once written | Must be backed up / OCI-synced; loss is unrecoverable evidence loss |
| Append-only durable data | `sources/raw/live/`, `models/leaders/leaderboard_v*.jsonl`, `ops/paper/trades.jsonl`, `live/signals.jsonl` | Must be backed up; safe to sync incrementally |
| Mutable authoritative state | `models/model_registry/<obj>/LATEST.json`, `ops/webui/jobs/`, `session_levels.parquet` (newest-build-wins, a known pre-existing gap) | Point-in-time snapshot only; the *current* value is authoritative, history is not guaranteed |
| Pure reproducible cache | `research/search/_fold_cache/` | **Must never be the sole provenance record** for anything (§14 of the task) — confirmed true: `fold_cache.py`'s own module docstring and cache-key derivation depend only on already-durable file inputs (parquet + objective YAML), no database dependency; no sync/backup requirement |
| Derived feature store, **conditionally** reproducible | `sources/features/timeframe=*/` hive parts and every other `sources/features/*` class (`parquet_tape`, `parquet_path1s`, `parquet_options_surface`, `session_profile`, etc.) | **Corrected this pass** — see below. NOT a pure cache: `src/features/engine.py`'s own module docstring states it "loads source data from TimescaleDB" (confirmed via its `psycopg` usage), not from `sources/raw/` files directly. Reproducing the hive genuinely requires TimescaleDB to still hold the historical `tbbo`/`bbo_1s`/`definitions` rows — and `docs/STORAGE_INVENTORY.md` §2.1 already found raw-file retention is **not uniform** (`backfill_options_daily.py` deletes each day's raw file after ingest). Do **not** claim "no backup requirement": TimescaleDB's own backup/retention posture is unverified by this (or any prior) documentation pass, since querying it is out of scope. A backup/retention audit of TimescaleDB itself is an unresolved decision (§16), not a settled "safe" classification |
| Ephemeral scratch | `ops/tmp/` (by design; not honored in practice today — §14 of `STORAGE_INVENTORY.md`, STOR-10) | No backup; safe to delete; **currently not actually ephemeral**, flagged not fixed |
| Operational state | `ops/gpu_jobs/`, `ops/job_state/`, `ops/gpu_health/` | Point-in-time only; loss degrades monitoring, not science |
| Compatibility | files matching `COMPATIBILITY_ARTIFACTS` in `artifacts.py`; the `_prefer()` fallback branches throughout `data_paths.py` | Same durability class as what they realise; the compatibility label is about naming, not lifecycle |
| Legacy/archive | `legacy/*`, `sources/labels/firstpass*` | Frozen; safe to leave, unsafe to delete without the explicit operator approval AGENTS.md §4 requires |

A cache accidentally becoming the sole provenance record was checked explicitly for `_fold_cache/` (confirmed a pure function of already-durable file inputs — safe) and for `sources/features` hive parts (**correction, this pass**: previously asserted "rebuildable from `sources/raw/`" without verifying `FeatureEngine`'s actual upstream; it reads TimescaleDB, and raw-file retention beneath it is confirmed non-uniform, so the hive's true reproducibility is conditional on TimescaleDB state this pass did not and could not verify — no DB queries permitted). The one class confirmed to have no such conditional dependency is `_fold_cache`.

## 13. Resolver contract and this pass's changes

Per-path classification vocabulary (`CANONICAL_RESOLVER_BACKED` / `COMPATIBILITY` / `HARDCODED_MIGRATION_CANDIDATE` / `LEGACY` / `OPERATIONAL_SPECIAL_CASE`) is applied throughout `docs/STORAGE_INVENTORY.md` §2. `src/common/data_paths.py` remains the sole path SSOT; `src/` continues to hold no path resolution of its own (`cli.py:75` is still the only consumer, confirmed again this pass).

**Resolver changes made this pass** (all additive, all path-identical, all verified with `python -m pytest tests/architecture/ tests/test_tape_cascade.py` — 226 passed, no regressions):

1. `_LAYOUT["live"] = "live"` + `live_dir()` added. `src/webui/live.py::_live_dir()`, `src/webui/regmodel.py`, and `src/execution/config.py` each independently constructed `data_root() / "live"` before; all three now call `live_dir()`. Closes H13.
2. `_LAYOUT["session_profile"] = "sources/features/session_profile"` added. `src/features/session_profile.py::session_profile_dir()`'s default branch (`root=None`) now resolves through `layout("session_profile")`; the `root`-override branch (used by tests) is unchanged. Closes H14 — this directory feeds `session_levels.parquet`, the cascade's structural SSOT.
3. `scripts/search/regime_analysis.py`'s `regimes/` output path, previously `Path("/opt/vtech") / "data/research/research_director/regimes"` (a fully hardcoded absolute path that did not honour `VTECH_DATA_DIR`), now resolves via `research_director_dir() / "regimes"`. Closes H15.
4. `src/experiment/artifacts.py`'s `"resolved_config"` artifact class renamed to `"config_snapshot"` (§9 of the task): the class name now matches the `RunManifest` field it realises; whether that snapshot is genuinely *resolved* remains the separate, explicit `config_snapshot_resolved` claim, never implied by the artifact-class name. `docs/EXPERIMENTS.md` §2.1's table row updated to match. No other reference to the old name existed outside a same-purpose test function name in `test_experiment_manifest.py`, which was already about the `RunManifest` field, not the artifact-class string, and needed no change.
5. `src/backtest/cascade_backtest.py::run_cascade_inmemory`'s `instrument: str = "MNQ"` parameter (and the `"instrument"` key in its two output-payload dicts) renamed to `economic_contract_unit: str = "MNQ_equivalent"` (§7 above). Verified as the label's only use and its only caller before renaming.

**Deliberately not changed this pass** (each would exceed "small, unambiguous, behavior-preserving"): the 9+ hardcoded `sources/raw/backfill` producer scripts (STOR-5, too many independent call sites for one pass); Track C's `move_matrix`/`target_matrix` stale consumer defaults (STOR-2, changing a default changes which dataset a script uses — a behavior change, not a path-resolver substitution); the MLflow artifact split-brain (STOR-3, requires a database admin action, and this pass may not query or mutate any database); `_fold_cache`'s missing LRU eviction (STOR-1, a runtime behavior change, already designed and deferred in `docs/operations/DISK_RECLAIM_2026_08.md`); `training_runs_dir()`'s dead resolver entry (removing it or wiring a producer are both behavior decisions, not documentation facts).

## 14. Risk register

Full detail, including consequence and remedy for each, is `docs/STORAGE_INVENTORY.md` §4 (STOR-1 through STOR-10). Ranked summary: `_fold_cache` unbounded growth inside a never-delete root (STOR-1); Track C consumer scripts silently defaulting to stale dataset versions (STOR-2, the only risk in this list with live, present-tense correctness consequences today); the MLflow artifact split-brain (STOR-3); Track C's version-suffix-only identity (STOR-4, the storage-architecture root cause behind STOR-2); systemic resolver bypass in raw-data backfill producers (STOR-5); an unconfirmed-reader 2.5G archive (STOR-6); a model-lineage chain that breaks at the same point on both the leaderboard side and the search-run side (STOR-7); Track B's `RunManifest` integration existing only in the uncommitted working tree, with zero `manifest.json` files on disk anywhere (STOR-8); orphaned top-level loose files (STOR-9); `ops/tmp` and `legacy/checkpoints` each mixing live and dead content under one directory name (STOR-10).

## 15. Clean-sheet self-review

Run after §1–§14 were drafted, against the task's own ten challenge questions, using the same adversarial-review convention `docs/FUNCTIONAL_COVERAGE.md` §16 established.

| Question | Finding |
|---|---|
| Is every directory level necessary? | The native-TF hive's `symbol=` level is currently single-valued (`NQ` only) — technically unnecessary today, but it is real partitioning infrastructure already built for a documented future (cross-asset context series already land under `sources/raw` per-symbol; extending the hive to a second symbol requires no schema change). Kept — this is the one case in the tree where "not needed yet" is a deliberate, cheap hedge rather than premature structure. No other unnecessary level was found. |
| Is any semantic concept represented by two canonical names? | Yes, found and documented, not renamed: `sources/labels/` holds both "labels" (the repository's long-standing term) and what this document's own terminology table calls "outcome dataset" — kept as synonyms deliberately (§7 explicitly retains `labels` as the physical/code term and uses "outcome dataset" only in cross-track prose), rather than forcing a rename the task itself warns against pre-deciding. |
| Is any physical path being used as scientific identity? | Yes — this is STOR-4, already the top storage-architecture finding, not newly discovered by the review. |
| Does any filename encode model topology unnecessarily? | No new instance found beyond what `docs/EXPERIMENTS.md` already governs for run artifacts (`artifacts.py`'s filename criteria explicitly test for and forbid this). Track C's dataset filenames encode *build generation*, not topology — a different, already-documented problem (STOR-4), not this one. |
| Are immutable scientific outputs clearly separated from mutable operational state? | Mostly yes. One confirmed leak found by the review and not previously documented: `evaluation/<objective>/` directories physically also hold `latest_leaders.json` (mutable, rewritten) beside `contract_v*.json` (frozen by intent) — added to `STORAGE_INVENTORY.md` §2.5 and the migration matrix as `NEEDS_DECISION` during this review pass, not left for a future one, since it was a direct answer to this question. |
| Are caches clearly non-authoritative? | Yes for `_fold_cache` and the feature hive (both confirmed rebuildable, §12). `ops/tmp` is the one exception — nominally a cache/scratch class but not actually evicted, which is exactly why STOR-10 exists rather than being silently accepted. |
| Can a fresh agent locate the producer and consumer for every major class? | For 33 of the ~35 classes inventoried, yes, with file:line citations in `STORAGE_INVENTORY.md` §2. Two exceptions, both already flagged: `parquet_options_surface`'s consumer (STOR-6) and `_fold_cache`'s relationship to the "never-delete" policy it sits inside (STOR-1) — both are findings, not gaps in this document. |
| Can a future one-stage, multi-stage, ensemble or non-ML strategy use the same architecture? | Yes for the run-artifact and serving-bundle contracts (already topology-neutral by construction, `artifacts.py`'s own stated design goal). Not yet for Track C's outcome datasets (STOR-4) — but that gap is about identity, not topology-coupling; nothing in the outcome-dataset path names a stage or model family. |
| Are Track B and Track C accommodated without track-specific filesystem architecture? | Yes at the top level (`sources/labels/` is one directory for both; `research/<workflow>/<objective_id>/<run_ts>/` is one convention for both, once Track C is wired). The one track-specific divergence is Track C's version-suffix identity scheme itself (STOR-4) — which is the defect, not a designed accommodation. |
| Is any proposed root missing resolver ownership? | `_fold_cache` has no dedicated `_LAYOUT` key, but is not "missing ownership" in the harmful sense: it is built as `search_dir() / "_fold_cache"`, i.e. it already inherits `search_dir()`'s resolver backing and honours `VTECH_DATA_DIR`. Reviewed and accepted as sufficient — adding a redundant top-level key for a private implementation detail of one module would violate this document's own "avoid unnecessary nesting" principle (§8). |
| Does any proposed standard require large duplicated files? | No. Every recommendation in §8–§11 either changes metadata/sidecars or is a no-op confirmation of an existing pattern; none requires copying or duplicating existing multi-GB datasets. |

No material problems were found that changed the architecture as drafted; one real gap (the `evaluation/` leaderboard-file duplication) was found and folded into the risk register and migration matrix during the review itself, per the task's own instruction to revise before finalizing.

## 16. Unresolved decisions / prioritized implementation sequence

Every `NEEDS_DECISION` disposition in `docs/STORAGE_INVENTORY.md` §3 requires a human call this pass deliberately does not make. In priority order for a future implementation pass:

1. **Track C outcome-dataset identity (STOR-4/STOR-2).** Highest value, highest cost: adopt the unified contract-addressing pattern §17 defines (generalized from `label_store.py`) for `move_matrix`/`target_matrix`, which requires defining Track C's semantic contract fields and rebuilding/hashing existing data — the mechanism is designed this pass (§17), execution is explicitly excluded by the task's own "do not hash massive existing datasets" instruction. Until then, STOR-2's stale-default risk is live.
2. **`_fold_cache` LRU eviction (STOR-1).** Already designed (`docs/operations/DISK_RECLAIM_2026_08.md` Stage 3); implementation is a runtime code change, not a documentation task.
3. **MLflow artifact split-brain (STOR-3).** Requires an MLflow admin action against the Postgres backend store; this pass may not query or mutate any database.
4. **`legacy/checkpoints`'s live ingest-JSON slice (STOR-10).** Splitting it into `ops/job_state/` (the already-partially-executed consolidation `DATA_LAYOUT.md` itself names) touches 6+ active ingestion scripts — a multi-file migration, not a small additive one.
5. **`evaluation/` leaderboard-file duplication** (found during §15's self-review). Requires confirming why the duplicate writer still fires before deciding whether to stop it — a code-behavior question, not answerable from storage architecture alone.
6. **`sources/raw/backfill` resolver bypass (STOR-5).** Lowest risk, highest file count (9+ scripts) — mechanical but not small.

---

## 17. Canonical outcome-dataset architecture (Track B + Track C, unified)

**Ninth-pass addition, tenth-pass finalization (2026-09-22, Storage Implementation Tranche 1).** The eighth pass's migration matrix proposed `sources/labels/outcomes/<sha256>/` as Track C's target shape while Track B already lives at `sources/labels/{5s,15s}/<contract_id>/`. The ninth pass generalized this into one `<logical_dataset_id>/<contract_id>/` slot, but that slot itself overloaded two different things: Track B's `<logical_dataset_id>` values (`5s`, `15s`) are RESOLUTIONS of one outcome family (triple-barrier direction); Track C's would-be values (`move`, `target`) are two entirely DIFFERENT outcome families, not resolutions of each other or of Track B's. This section replaces that one slot with two explicit, orthogonal dimensions and settles the exact syntax for implementation — this is the final canonical syntax, not another proposal.

### 17.1 The pattern

```
sources/labels/<outcome_family>/<resolution>/<contract_id>/
    _contract.json              # semantic contract: what this dataset MEANS (§5a, §19)
    _build.json                 # realization/provenance, when genuine evidence exists (§5a, §19)
    part001.parquet             # size-split parts, zero-padded (Track B compatibility naming; §18)
    part_2026-01-01.parquet     # OR session-keyed parts, for a session-grain producer (§18)
```

**`<outcome_family>`** states WHAT KIND of outcome this is, at a category level — a stable, topology-neutral name, never a Track C implementation nickname ("move", "target"), a stage number, a model family or a version. It is the genuinely new dimension this pass adds (the ninth pass's `<logical_dataset_id>` conflated it with resolution; separating them is the fix):

| Outcome family | Meaning | Current producer |
|---|---|---|
| `triple_barrier` | 3-class label: did price hit the upper/lower barrier or time out, from a fixed decision point, both directions at once | Track B, `label_store.py`/`tbm_contract_from_objective()` — matches the contract's own existing `"scheme": "triple_barrier"` field, reused rather than inventing a second name for the same concept |
| `directional_first_passage` | Binary label: from a decision point AND a chosen side, did price move `move_r` favorably before `adverse_r` adversely | Track C, `build_move_matrix.py` (`y_move` — "Stage 1" in prose, not in this vocabulary) |
| `candidate_level_reachability` | Binary label: given an already-gated directional decision, did price reach THIS specific candidate structural level before its own derived stop | Track C, `build_target_matrix.py` (`y_reach` — "Stage 3" in prose, not in this vocabulary) |

A future outcome definition (a different holding-period rule, a portfolio-level outcome, a non-ML strategy's own success criterion) adds a new family value here — never a new directory level, never a change to this pattern.

**`<resolution>`** is the decision-time cadence the outcome was computed at (`5s`, `15s`, …) — kept as a DISTINCT dimension from `<outcome_family>`, per the rule that motivated this section: the same family can exist at more than one resolution (Track B's `triple_barrier` already does, at `5s` and `15s`), and the same resolution says nothing about which family it belongs to. Track C's `directional_first_passage` and `candidate_level_reachability` outcomes both currently use `5s` (`MoveConfig.decision_grid_seconds` / `cascade_policy.DECISION_GRID_SECONDS`) — a genuine, physical partitioning value, not a placeholder.

**`<contract_id>`** is `label_store.outcome_contract_id()` — the same sha256-of-normalized-contract-dict mechanism `label_contract_id()` uses (§5a), applied without Track B's TBM-specific `target_observation` default (which means nothing for Track C's families). Track B's existing `label_contract_id()`/`label_set_dir()` are UNCHANGED and remain the compatibility path for its existing on-disk directories — `outcome_contract_id()`/`outcome_set_dir()` (`src/common/data_paths.py`, `src/training/label_store.py`) are the new, track-neutral entry points for every other family. §19 defines exactly which of Track C's builder CLI fields belong in the hashed contract.

**Build/realization identity** (§5a's `build_id`) stays in the `_build.json` sidecar, never in the path — a `force`-style rebuild under an unchanged semantic contract is a new *realization*, not a new *meaning*, and the two questions have different lifecycles (a contract changes when the label rule changes; a realization changes every rebuild). `_build.json` also carries lineage: a `candidate_level_reachability` dataset is built FROM a `directional_first_passage` dataset's gated rows AND from that upstream move model's own score PREDICTIONS (`--scores`, distinct semantic type, §19) — both are recorded as separate upstream `DatasetRef`-shaped entries in `_build.json`'s `upstream` list, reusing the existing `DatasetRef` shape rather than inventing a second lineage concept, and never folded into each other or hidden inside this dataset's own `build_id`.

### 17.2 Compatibility mapping (no moves performed)

| Existing path | Compatibility status | Canonical new-write pattern |
|---|---|---|
| `sources/labels/{5s,15s}/<contract_id>/` (Track B TBM) | Already canonical — matches §17.1 exactly, with `triple_barrier` as the (currently implicit) `<outcome_family>` | Unchanged; nothing to migrate. `outcome_set_dir("triple_barrier", "15s", contract)` resolves to a path that is a peer of, not identical to, the existing `label_set_dir("15s", contract)` — the two are NOT required to converge in this pass (no rebuild); Track B's `_contract.json`/parquet parts stay exactly where they are |
| `sources/labels/move_matrix{,_v2,_v3,_v4}/` | `KEEP_COMPATIBILITY` — existing readers/writers (`build_move_matrix.py`, `eval_move_model.py`, `build_target_matrix.py`) continue exactly as today, and now require an explicit `--dir`/`--move-dir` rather than a silent stale default (§9 below) | `sources/labels/directional_first_passage/5s/<contract_id>/`, written by `build_move_matrix.py --canonical-write` (implemented this pass, not executed — §8 of the task) |
| `sources/labels/target_matrix{,_v3,_v4,_adaptive,_context,_context_rest}/` | `KEEP_COMPATIBILITY` | `sources/labels/candidate_level_reachability/5s/<contract_id>/`, written by `build_target_matrix.py --canonical-write` (implemented this pass, not executed), referencing its upstream move dataset + move-score predictions in `_build.json` |
| `sources/labels/firstpass*/` (7 dirs, superseded) | `LEGACY_PRESERVE` (§16 of `STORAGE_INVENTORY.md`) | None — superseded track, not migrated forward |
| `sources/labels/1s/` (dead ADR-107 artifact) | `LEGACY_PRESERVE` | None |

### 17.3 What this closes from the provenance list

Track C's identity must determine: which move dataset was used (→ the target build's companion `DatasetRef` for its move-outcome input, `_build.json.upstream`); which move-score realization gated the target build (→ a SEPARATE `DatasetRef` with `semantic_type="predictions"` for `--scores`, never conflated with the move-outcome dataset itself, §19); which target dataset was built (→ the target's own `contract_id` + `build_id`); effective build parameters (→ `_build.json.population_config`, §19's build/population-configuration tier); source dataset versions (→ the `upstream` list, one entry per input, same `DatasetRef` shape `RunManifest.datasets` already uses); code/build identity (→ `_build.json.code_sha`/`code_dirty`, from `src.common.code_identity`, shared with `RunManifest.code` rather than duplicated); semantic contract (→ `contract_id`). No new top-level concept beyond §5a's three-tier model and the existing `DatasetRef` shape — this section fixes the SYNTAX and WIRES the write path (`label_store.write_outcome_contract`/`write_build_sidecar`, `--canonical-write` on both builders); it does not execute a build or backfill Track C's existing datasets into it.

## 18. Canonical filename, format and content contract

One table per major durable class, reusing established contracts (`artifacts.py`, `schemas.py`, `label_store.py`) rather than re-specifying hundreds of feature columns. For classes with wide, variable column sets (feature/outcome matrices), only the identity/index columns are given here; the authoritative column list is the cited module.

| Semantic class | Canonical directory | Canonical filename/part pattern | Format | Grain | Primary logical key | Required identity metadata | Timestamp/TZ convention | Schema version | Mutability |
|---|---|---|---|---|---|---|---|---|---|
| Raw/source files | `sources/raw/{live,backfill}/...` | vendor-native (`*.dbn[.zst]`) | Databento binary | per streamer segment / per API batch | session date (live) / `job_id` (backfill) | none today (STOR-5 gap) | UTC in DBN payload; directory keyed by Chicago session date (live) | n/a (vendor format) | append-only per file (uneven retention, §2.1 STOR-5) |
| Native-TF feature parts | `sources/features/timeframe={tf}/symbol=/year=/month=/` | `features_{tf}_part{N}.parquet` (`native_tf_part_name()`) | Parquet | monthly, size-split (~100MB cap, `NATIVE_TF_PART_MAX_BYTES`) | `bar_end_ts` (UTC) | `store_id` only (mutable, §5a) | `bar_end_ts`, UTC | none (schema drift tracked only informally) | destructive rebuild per month part (`native_tf.py:161-162`) |
| Session/tape/path feature datasets | `sources/features/{parquet_tape,parquet_path1s,session_profile}/` | `{tape,path,profile}_<YYYY-MM-DD>.parquet` | Parquet | one file per session | session date | none | session date; intra-file `ts` UTC | none | replaceable per session file |
| Outcome dataset (contract-addressed) | `sources/labels/<outcome_family>/<resolution>/<contract_id>/` (§17, §19) | `_contract.json` + `_build.json` (when genuine build evidence exists) + `part{NNN}.parquet` (size-split) or `part_<session-date>.parquet` (session-keyed) — `label_store.outcome_part_name()` | JSON sidecar ×2 + Parquet | per outcome-family/resolution, size-split OR one part per session | `ts` (UTC) | `contract_id` (required); `build_id` (recommended, in `_build.json`, genuine code/config/upstream evidence only — never derived from the parts themselves, §5a) | `ts`, UTC (`_as_utc_index()`) | `_contract.json.schema_version` (explicit integer field, `label_store.CONTRACT_SCHEMA_VERSION`; excluded from the `contract_id` hash) | idempotent by default, NOT content-immutable (§5a) |
| Dataset contract sidecar | `<outcome_set_dir>/_contract.json` (canonical, §17), `<label_set_dir>/_contract.json` (Track B compat), `<run>/label_contract.json` (run-side snapshot) | fixed name | JSON | one per contract dir / one per run | `contract_id` | `contract_id`, `created_at_utc`, `schema_version`; run-side snapshot adds `source_dir`, `parts` (§5a) | `created_at_utc`, UTC ISO-8601 | `schema_version` field (contract dir only, tenth pass) | mutable only via `write_contract`/`write_outcome_contract` (preserves original `created_at_utc`/`schema_version`); run-side snapshot immutable once written |
| Dataset build/provenance sidecar | `<outcome_set_dir>/_build.json` (`label_store.write_build_sidecar()`, §17, §19) | fixed name | JSON | one per contract dir, written only when genuine evidence exists | `contract_id` (parent dir) | `code_sha`, `code_dirty` (`src.common.code_identity`), `population_config`, `upstream` (list of `DatasetRef`-shaped dicts), `build_id` (derived from the above, never from output metadata) | `created_at_utc`, UTC ISO-8601 | none (new this pass; versioned via the parent contract dir's own `schema_version` if it ever needs one) | written once per realization; a rebuild under the same `contract_id` writes a new `_build.json` alongside the (possibly replaced) parts |
| OOF predictions | `<run>/oof_<model>.parquet` (compat) / `oof_predictions.parquet` (canonical, `artifacts.py`) — run-scoped | as above | Parquet | one row per decision `ts` | `ts` | `fold_id` (recommended) | `ts`, UTC | via `RunManifest.schema_version` (run-level, not per-file) | immutable once written |
| Predictions dataset (cross-stage, e.g. Track C move scores) | Track C compat: `sources/labels/move_scores{,_v2,_v3,_v4}.parquet` (`--scores`, `build_target_matrix.py`) | unchanged (compat); no canonical new-write path defined this pass — the producer is `eval_move_model.py --out-scores`, an eval-script artifact, not yet a canonical outcome/prediction writer | Parquet | one row per scored decision | `ts` | none today (gap) | `ts`, UTC | none | overwritten per eval run |
| Run metrics / fold metrics | `<run>/{metrics.json,fold_metrics.json}` (`artifacts.py`) | fixed names | JSON | one per run / one per fold | `run_id` / `fold_id` | `n_samples` (metrics), per-fold bounds (fold_metrics) | run-level, no per-record timestamp | via `RunManifest.schema_version` | immutable once written |
| Selected population | `<run>/selected_population.json` (`artifacts.py`) | fixed name | JSON | one per run | `run_id` | `n_raw`, `n_selected`, `selection_rule` | n/a | via `RunManifest.schema_version` | immutable once written |
| Trade ledger | `<run>/trades.parquet` (compat) / `trade_ledger.parquet` (canonical) | as above | Parquet | one row per completed trade | `trade_id` | `trade_id`, `entry_ts`/`exit_ts` | `entry_ts`/`exit_ts`, UTC | via `RunManifest.schema_version` | immutable once written |
| Economic outputs | `<run>/{economic_metrics.json,cost_sensitivity.json}` (`artifacts.py`) | fixed names | JSON | one per run | `run_id` | `assumptions` (incl. `economic_contract_unit`, `underlying`) — never separated from results | n/a | via `RunManifest.schema_version` | immutable once written |
| Model/serving bundle | `models/{model_staging,model_registry}/<objective_id>/<candidate>_<run_ts>/` | `model.{ubj,keras}`, `scaler.pkl`, `feature_schema.json`, `finalize_manifest.json` | native binary + JSON + pickle | one bundle per candidate realization | `objective_id` + `candidate`/`run_ts` | `finalize_manifest.json` fields (not `RunManifest`-linked, STOR-7) | `finalized_at`, UTC | `finalize_manifest.json`'s own (informal) | copy-once by convention (not enforced, §2.5 STOR-7) |
| Live/operational event streams | `live/signals.jsonl`, `live/heartbeat.json`, `ops/paper/trades.jsonl` | fixed names | JSONL (append) / JSON (point-in-time) | one JSON object per line (signals/trades) / single overwritten object (heartbeat) | `ts` per line | `objective_id`, `session_date` per record | `ts`, UTC ISO-8601 (`Z` suffix) | none | append-only (event streams) / mutable-overwrite (heartbeat) |

Column-level schema for wide matrices (feature hive, `move_matrix`/`target_matrix`, OOF predictions) is intentionally not reproduced here — see the cited producer module (`native_tf.py`, `build_move_matrix.py`, `build_target_matrix.py`, `src/experiment/schemas.py::OOF_PREDICTION_COLUMNS`) for the authoritative column list, per the task's own instruction to reuse established contracts rather than re-derive them.

---

## 19. Track C semantic contract fields (tenth pass, Storage Implementation Tranche 1)

Every CLI parameter of `build_move_matrix.py`/`build_target_matrix.py` classified into exactly one tier — not "hash every CLI argument," per the task's own instruction. **Semantic** fields change what the outcome MEANS (they enter `contract_id`, via `move_outcome_contract()`/`target_outcome_contract()`, `src/training/label_store.py`). **Build/population** fields change which rows/columns are realized WITHOUT redefining the outcome (they enter `_build.json.population_config`, never the hash). **Execution-only** fields carry no scientific content at all (workers, output path, dry-run).

### 19.1 `build_move_matrix.py` (`directional_first_passage`)

| CLI field | Tier | Why |
|---|---|---|
| `--decision-grid` (`decision_grid_seconds`) | Semantic | Defines the sampling cadence of the outcome itself — part of what a "decision" is |
| `--horizon` (`horizon_seconds`) | Semantic | The forward window within which the move must occur — core to the label definition |
| `--move-r` | Semantic | The reward barrier — core |
| `--adverse-r` | Semantic | The risk barrier — core |
| `--r-lookback-sessions` | Semantic | Defines how the R normalization unit itself is computed; changes what "big enough" means for the same raw point distance, not merely which rows are included |
| `--lookback-sessions` | Build/population | Feeds `es_`/structural-bound FEATURE columns (entry-state history depth), not the label — changes realized feature content, not what `y_move` means |
| `--start`/`--end` | Build/population | Which sessions are included — realization coverage, not outcome meaning |
| `--workers`, `--out`, `--dry-run`, `--canonical-write` | Execution-only | No scientific content |

### 19.2 `build_target_matrix.py` (`candidate_level_reachability`)

| CLI field | Tier | Why |
|---|---|---|
| `--horizon` | Semantic | The window within which reach must occur |
| `--max-targets` | Semantic | How many candidate bounds are considered — structural to the label's row grain |
| `--stop-ratio`, `--max-stop-r` | Semantic | Define the per-target stop-placement rule (ADR-130) — core to what `y_reach` measures |
| `--resting-entry`, `--max-fill-wait` | Semantic | Change HOW the entry fill (and thus the stop/target basis) is computed — geometry of the label itself (ADR-126), not merely which rows are included |
| `--lookback-sessions` | Semantic | Determines which structural bounds are even CANDIDATE targets — changes what `y_reach` can be true OF, unlike the same-named field in `build_move_matrix.py` (there it only feeds features) |
| `--r-lookback-sessions` | Semantic | Same reasoning as §19.1 |
| `--tau-q` | Build/population | Selects which move-scored rows are gated INTO the target build — changes the realized row population, not how `y_reach`/`t_reach` are computed for an included row |
| `--context`, `--move-dir` | Build/population | Which upstream FEATURE columns ride along — changes realized column content, not label meaning. `--move-dir` is also upstream LINEAGE (§17.3), recorded as a `DatasetRef` in `_build.json.upstream`, not hashed into `contract_id` |
| `--scores` | Neither — **upstream input**, not a builder parameter at all | The move-model's OOF predictions this build gates on. Recorded as its own `DatasetRef` with `semantic_type="predictions"` in `_build.json.upstream` (§17.3) — a prediction is a model's claim, not ground truth, and must never be hidden inside `build_id` or conflated with the move-outcome dataset itself |
| `--workers`, `--out`, `--canonical-write` | Execution-only | No scientific content |

Implemented in `src/training/label_store.py::move_outcome_contract()`/`target_outcome_contract()` (semantic tier only) and `build_move_matrix.py`/`build_target_matrix.py --canonical-write` (wires `_build.json.population_config`/`upstream` from the build/population and lineage tiers). Not executed this pass — no Track C data was rebuilt.

---

## 20. Final canonical storage syntax (authoritative)

One compact table, sufficient on its own to answer "where should this new dataset/output be written, and what exactly should its files be named?" without reading Track B/C code. Detail sections are cited for anyone who needs the reasoning; this table is the answer.

| Semantic class | Canonical directory syntax | Canonical filenames | Identity |
|---|---|---|---|
| Raw source | `sources/raw/{live,backfill}/...` (§18) | vendor-native `*.dbn[.zst]` | session date / `job_id` (gap, STOR-5) |
| Native features | `sources/features/timeframe={tf}/symbol=/year=/month=/` (§18) | `features_{tf}_part{N}.parquet` (`native_tf_part_name()`) | `store_id` (mutable, §5a) |
| Session/tape features | `sources/features/{parquet_tape,parquet_path1s,session_profile}/` (§18) | `{tape,path,profile}_<YYYY-MM-DD>.parquet` | session date |
| **Price tape** | `sources/price_tape/<instrument_root>/` (§21; `price_tape_dir()`) | `tape_<YYYY-MM-DD>.parquet` — one file per session so a day appends without rewriting history. This class is session-grained, so the session date IS its identity and belongs in the filename; that is why it does not use outcome datasets' `part{NNN}.parquet` form. Compaction to `partNNN.parquet` is a **deferred option**, not the canonical name | session date + `<instrument_root>` (the root whose prices are in the file, `NQ`, never the outright month) |
| **Outcome datasets** | `sources/labels/<outcome_family>/<resolution>/<contract_id>/` (§17.1) | `_contract.json` + `_build.json` (if genuine build evidence exists) + `part{NNN}.parquet` / `part_<session-date>.parquet` (`outcome_part_name()`) | `contract_id` (semantic, required) / `build_id` (realization, recommended) / `content_sha256` (content, never fabricated) — §5a |
| **Predictions** | `research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/` (§21) — a model's predictions are run-scoped output, never a shared source dataset, so they have no `sources/` path | `oof_predictions.parquet` (compact: keys, fold, outcomes, predictions, thresholds; **no feature columns**) | the producing `run_id` + `pass_n` + `model_id`. A consumer records it as an upstream `DatasetRef(semantic_type="predictions")`; it never carries an outcome's `contract_id` as its own identity |
| Research runs | `research/<workflow>/<objective_id>/<run_ts>/` (§8) | `manifest.json`, `run.json`, `folds.json`, `oof_<model>.parquet`, … (`artifacts.py`) | `run_id` (`= <objective_id>/<run_ts>`) + `RunManifest.scientific_fingerprint()` |
| Model bundles | `models/{model_staging,model_registry}/<objective_id>/<candidate>_<run_ts>/` (§18) | `model.{ubj,keras}`, `scaler.pkl`, `feature_schema.json`, `finalize_manifest.json` | `objective_id` + `candidate`/`run_ts` (not yet `RunManifest`-linked, STOR-7) |
| Operational/live | `ops/<subsystem>/...` (§8); `live/` is a **documented exception** — current, stable, no auto-cutover; `ops/live/` is the future target (§8a) | subsystem-specific, mostly fixed names (`signals.jsonl`, `heartbeat.json`, `trades.jsonl`) | subsystem-specific; none for `live/` today |

**An outcome dataset's home is not a run's cache.** Ground-truth labels live once, under `sources/labels/<outcome_family>/<resolution>/<contract_id>/`, and are reusable across every run that references that `contract_id`. `pass_<n>/training_matrix.parquet` is a cache of a *join* — labels joined to features and context for one row population — and is deletable without losing evidence. A label dataset is therefore never "migrated into" a run's matrix: a run references the contract, and the matrix may cache the joined result. Any row of `docs/new_file_map.md` that maps a `sources/labels/` directory to a `training_matrix.parquet` describes the *cache that run materialized*, not a relocation of the labels, and the source directory stays.

**Version/identity, in one line:** `contract_id` = what it MEANS (hash of the semantic contract, `label_contract_id()`/`outcome_contract_id()`); `build_id` = which deterministic BUILD produced it (genuine code/config/upstream evidence only, `_build.json`, never derived from output filenames/sizes); `content_sha256` = the actual materialized BYTES (not computed anywhere in this codebase today, never fabricated); `store_id` = a mutable logical pointer only (weakest tier). Only `build`/`content` satisfy `RunManifest.missing_evidence()`'s `datasets` check; `contract`/`store` are real but insufficient for a controlled comparison (§5a).

---

## 21. Objective run layout

New objective runs are written here. `src/common/run_layout.py` is the path helper. `docs/new_file_map.md` is the old-path to new-path checklist, including what has been copied.

```text
sources/labels/<dataset>/_contract.json          immutable label definition; runs reference contract_id
sources/price_tape/<instrument>/tape_<YYYY-MM-DD>.parquet
research/runs/<objective_id>/<run_ts>/manifest.json
research/runs/<objective_id>/<hpo_run_ts>/hpo.json                         Optuna study; absent on a single-model run
research/runs/<objective_id>/<hpo_run_ts>/trials_summary.parquet           one row per trial
research/runs/<objective_id>/<run_ts>/pass_<n>/training_matrix.parquet   optional population cache
research/runs/<objective_id>/<hpo_run_ts>/pass_<n>/model_<model_id>/trial.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/oof_predictions.parquet
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/model.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/model.ubj
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/metrics.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/fold_metrics.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/selected_population.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/trade_ledger.parquet
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/economic_metrics.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/cost_sensitivity.json
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/summary.md
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_importance.csv
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_bar.png
research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_beeswarm.png
research/runs/<objective_id>/index.json
```

`manifest.json` is the only run-level file a new objective run writes. It is a `RunManifest` (`schema_version` 4). `passes` holds each pass's role, input spec, and model pointers. `config_snapshot` is the resolved objective. `validation` is the fold design. `run.json`, `contract.json`, and `folds.json` are read for runs that already have them and are not written for new runs.

**Module ownership.** Filename constants and the artifact/join/OOF contract are `src/experiment/artifacts.py`. Model identity and `model.json` content are `src/experiment/model_identity.py` (`allocate_model_id`, `validate_model_id`, `write_model_json`, `oof_core_columns`). The cascade's own pass roles and feature views are `src/training/cascade_run_spec.py` (`CASCADE_PASS`, `xgb_hyperparameters`, `move_feature_spec`, `target_feature_spec`) — strategy vocabulary, deliberately outside the canonical layer per ADR-136. `src/common/run_layout.py` only resolves paths and holds no model-type or pass-role vocabulary; `tests/test_run_layout.py::test_run_layout_is_only_a_path_resolver` enforces that.

**The dataset inventory grows with the pass list.** A cascade is assembled one pass at a time, and pass 2 names inputs pass 1 never saw — its own outcome contract and the upstream predictions it gates on. `merge_pass` therefore re-derives `datasets` from the merged pass list and unions it with what is already recorded, and `RunManifest.write` accepts that as an additive change rather than a rewrite of history. Changing or dropping a ref that is already recorded still raises: every existing entry must survive byte-identically.

**Reading a run.** `read_run_record()` is the single run-level reader for every shape: a new objective run (`manifest.json`), an older objective run (the legacy trio), and a Track B `search`/`custom_test` run (`run.json` + `search_result.json`). A Track B run's candidate list is surfaced as `candidates` and is deliberately **not** promoted into `passes` — deciding which candidate is pass 1 and which is a meta layer is a migration decision (this file map's pass rule), not a read.

**Track B writes this layout too, and writes it live.** `src/search/canonical_run.py::emit_canonical_run` turns a Track B run directory into the canonical tree: one `pass_<n>/model_<model_id>/` per fitted candidate, `model.json`, a compact OOF, metrics, the report set, and `manifest.json` via `merge_pass`. `src/search/controller.py` calls it when a search run finishes, so a new Track B run lands in the canonical layout as part of the run; `scripts/research/migrate_custom_test.py` calls the same function for a historical one. Pass assignment follows one rule — `meta__*`, `meta_stack__*`, `tf_oof_meta*` are pass 2, everything else in the run is pass 1 — and the historical `candidate_id` IS the model id, so existing leaderboard, `compose.json` and `finalize_manifest.json` pointers still resolve. Weight and bundle files are **hard-linked** inside the run: the trainer's `models/<candidate>/` view and the `pass_<n>/model_<id>/` view share bytes. `data/research/custom_test/` has been migrated and removed. New runs are created under `research/runs/`.

A Track B model joins its features from the native-timeframe store at fit time and writes **no training matrix**. That is the canonical layout's no-cache case, not a gap: `model.json`'s `inputs` records the join with an explicit `availability` per source (`bar_close` plus the bar's own duration for a native-timeframe store; `timestamp` for calendar columns derived from the decision timestamp), and a meta model's `inputs` splits its fitted columns into `predictions` refs naming each upstream `model_id` and `context` refs for its own higher-timeframe columns.

**Evidence is closed where it is known and explained where it is not.** `merge_pass` accepts `seeds`, `execution_assumptions`, `code` and `extra`; each fills an empty field and never restates a recorded one. `execution_assumptions` is built in one place — `src/evaluation/economic.py::assumptions_record` — from the objective's `costs` block, resolving `execution_cost_profile` through `config/execution_costs.yaml`. When a run demonstrably applied a different cost than the objective declared, both are kept and `declared_not_applied` names the divergence; a `slippage_model` that no evaluator translated must stay visible rather than be flattened to one number. For a **migrated historical run**, `code.git_sha` comes from the source run's own `finalize_manifest.json`, never from the migrating machine's HEAD — stamping a run with a commit it never saw is a false provenance claim. Missing evidence is never reconstructed: a gap is reported in `missing_evidence` and its reason recorded in `extra.evidence`, and `tests/architecture/test_both_tracks_one_architecture.py` fails on any gap that has no recorded reason.

**Legacy artifacts are retired only after their last consumer moves.** `run.json`, `search_result.json`, `folds.json`, `feature_schema.json`, `finalize_manifest.json` and `compose.json` all still have real readers (`src/serving/bundle.py`, `src/search/tf_oof_meta.py`, `src/evaluation/shap_explain.py`, `src/search/mlflow_backfill.py`, `read_run_record`). `tests/architecture/test_legacy_retirement_gate.py` pins both halves per file: a consumer still reads it, so it must not be deleted; and the canonical replacement exists, so migrating the consumer is actually possible. When the last reader of a file moves, that file's row fails — and the failure is the signal to retire the file and the row together, in one change.

**The economic boundary is closed, and order intent is preserved.** `src/backtest/portfolio_sim.py` owns trade sequencing, path first-touch resolution, position and margin capacity, the session loss halt and account equity. `src/evaluation/economic.py` is the authority for pricing, commission, multiplier, PnL, the ledger and both kinds of sensitivity.

An order's **intent** (`OrderType`: MARKET / LIMIT / STOP) and its **resolved fill** are now independent. `Leg.resolved_fill_price` is valid on every order type and must name its `fill_source`; when present it is used verbatim and no slippage is added, because whatever produced it already accounted for execution. A resolved STOP therefore stays a STOP and a market entry stays MARKET — the earlier adapter rewrote both as LIMIT legs purely to stop slippage being applied twice, which discarded the order semantics and made slippage sensitivity silently ineffective. `Leg.role` (`ENTRY`/`EXIT_TARGET`/`EXIT_STOP`/`EXIT_TIME`) selects the per-leg slippage, so `entry_slippage_points`, `stop_slippage_points` and `target_slippage_points` genuinely participate in pricing instead of being recorded and ignored; a leg with no explicit value falls back to `slippage_points_per_side`, so the single-number model is unchanged. `TradeResult` records each leg's order type, role, fill source and the slippage actually applied (0.0 for a resolved fill), so a ledger states its own execution provenance.

Two sensitivity questions are now distinct, because they are different questions:

| | What varies | What stays | Function |
|---|---|---|---|
| **Fixed-fill cost sensitivity** | commission, contract multiplier | the fills | `evaluate_cost_sensitivity` |
| **Execution sensitivity** | slippage, fill model — so the fills move | the signals and the path | `evaluate_execution_sensitivity` |

Varying slippage against already-resolved fills cannot change any price, so it would return an identical ledger under a label claiming otherwise. `evaluate_cost_sensitivity` refuses that by name and points at the other function (`strict=False` opts out). `portfolio_sim.execution_sensitivity()` supplies the re-derivation by re-simulating the path under each scenario's assumptions.

This is the seam a higher-fidelity execution source plugs into without another architecture change: a Databento TBBO replay returns legs carrying `quote` and/or `resolved_fill_price` with `fill_source=FILL_SOURCE_TBBO`, keeps its real order type and role, and nothing else in either module changes.

Historical behaviour is unchanged: the simulator hands over its own resolved fills, so the blotter's `net_usd` and the KPI block are identical, and `tests/test_portfolio_sim.py::test_canonical_ledger_agrees_with_the_simulator_blotter` still pins the equality per trade and in aggregate. A target exit is a LIMIT only when no target slippage is assumed; with `target_slippage_ticks > 0` the sim is modelling a fill *through* the level, which a limit order cannot do, so the leg is MARKET — the order type follows the fill model rather than being asserted over it. The sim's `$5.00` round trip is still not replaced by the `$3.36` NQ profile; that takes an explicit `SimConfig.from_assumptions()` call.

**The hold-out pin.** `evaluation/<objective_id>/contract_v<N>.json` remains the live pin that `EvalContract.load_latest` reads. A run whose manifest records `validation.hold_out_dates` plus `config_snapshot.hold_out_hash` can supply it instead: `EvalContract.from_manifest(run_dir)`, or `load_latest(objective_id, run_dir=...)`. When both exist and agree, the manifest pin is returned; when they **disagree**, `ContractHashMismatch` is raised rather than resolved by precedence — two live sources of truth for one hold-out window is exactly how a model gets scored against the wrong pin. A caller that passes no `run_dir` gets the previous behaviour unchanged, so the four pinned `nq-50pt*` objectives are untouched.

Ground-truth labels stay in `sources/labels/` with `_contract.json`. That file is the reusable outcome definition. A run records `contract_id` on the pass and in `model.json`. It does not append itself to the contract.

The training frame is a join, not a required file. `model.json` names the label source, feature sources, optional context, join keys, filters, the selected feature columns, and the target. The trainer joins those at fit time. Every feature, context, and upstream-prediction join is as-of the decision `ts`: the value must have been knowable at or before that timestamp. A closed bar is knowable at its close. A completed session profile is knowable at that session's close. The label row may still carry an outcome defined from the forward path. `pass_<n>/training_matrix.parquet` may exist as one cache of a row population that is expensive to rebuild. A second feature subset is another `model.json` over that same population. A separate file is justified only when the row population itself differs (pass 2's candidate rows are not pass 1's decision-grid rows).

`oof_predictions.parquet` is compact: row keys, fold, the outcome fields later passes and reports need, predictions, and train-derived thresholds. It does not store the feature columns. Pass 2 reads pass 1's OOF for the gate, then joins pass 2's own labels, features, and context. `scripts/research/build_oof_analysis.py` rebuilds the wide table from the manifest and `model.json`, and can write a temporary `analysis_view.parquet` for Data Wrangler. After the OOF file is complete, eval writes SHAP under that model's `report/` from the OOF rows plus the feature columns named in `model.json`.

`model_<model_id>` uses a unique id for that model configuration. The id is not the algorithm. When `--model-id` is omitted, `allocate_model_id` hashes a sorted JSON document of `objective_id`, `run_ts`, `pass_n`, `model_type`, the resolved hyperparameters, the resolved feature view, and the target name. The same configuration always gets the same id. A different depth, tree count, feature view, or target in the same pass does not. Output paths and clock times other than `run_ts` are not inputs. `--model-id` overrides the hash. `model.ubj` is written only when a single fitted artifact is saved. The walk-forward evaluators do not write one.

An Optuna HPO study is one objective run. `research/search/` is a separate workflow. The writer is `scripts/search/hpo_model_search.py`, and it writes under `research/runs/`:

```text
research/runs/<objective_id>/<hpo_run_ts>/
    manifest.json
    hpo.json
    trials_summary.parquet
    pass_<n>/
        model_<model_id>/
            trial.json
            model.json
            metrics.json
```

`manifest.validation` records the folds the study scored: `split`, `n_folds`, and one `fold_windows` entry per window. Those come from the objective's walk-forward (`_build_folds`, then `--folds`). The writer does not replace them with `last_range` / `n_folds=1`. `config_snapshot.study.kind` is `hpo`. The snapshot is the loaded objective, and `config_snapshot_resolved` is true for that document. Each trial's hyperparameters stay in its `model.json`.

`hpo.json` records the Optuna sampler, seed, search space, metric, trial count, `pass_n`, and `selected_model_id`. `trials_summary.parquet` has one row per trial (`trial_number`, `model_id`, `status`, `score`, `params`). `trial.json` stores that trial's number, status, and score. `metrics.json` `n_samples` is the count of `y_oof` rows across every scored fold (`y_test` only when `y_oof` is absent). The model id comes from `allocate_model_id` on that trial's hyperparameters. The selected model is the pointer in `hpo.json`. The study does not write a per-trial OOF, a population cache, a weight file, SHAP, or a trade book. A later refit of the winner is an ordinary run at a new `run_ts`; its `model.json` records `hpo_run_ts` and `source_model_id`.

Track B custom-test bundles carry more than `model.ubj`. A migrated model folder also keeps whichever weight file the fit wrote (`model.keras`, `model.pkl`, `meta.pkl`, `meta_estimator.pkl`), the train-fitted `scaler.pkl`, and, until `src/serving/bundle.py` reads `model.json`, `feature_schema.json` and `finalize_manifest.json`. Pass 2 is a meta candidate (`meta__*`, `meta_stack__*`, `tf_oof_meta*`) and keeps `compose.json`. Historical `candidate_id` is that folder's model id. `docs/new_file_map.md` lists the `research/custom_test/` disposition and the `evaluation/nq-50pt*`, `portfolio_sim`, `move_analysis`, and `context_analysis` files. Fitted custom_test runs now live under `research/runs/`, and `data/research/custom_test/` is gone. `research/search/` remains a compatibility root. New runs write `research/runs/`. `shap_explain.py` writes its multi-class report (`shap_bar_global.png`, `shap_bar_signals.png`, per-class beeswarms, `shap_groups.csv`, `shap_meta.json`, `SHAP_SUMMARY.md`) into that model's `report/`.

Pass numbers stay generic. `fold_metrics.json` is the per-fold result and lives with the model. The copied v4 run still has `run.json`, `contract.json`, `folds.json`, and per-model matrices; readers accept that layout.

Files that are only meaningful after a step ran are written by that step, not as empty placeholders: `selected_population.json` when a gate selects rows, `trade_ledger.parquet` when taken trades exist, `economic_metrics.json` and `cost_sensitivity.json` when a cost result is computed. XGBoost SHAP is written by `analyze_move_scores.py` / `analyze_context_scores.py` into that model's `report/` when given `--objective`, `--run-ts`, and `--model-id`. The prose summary is `report/summary.md`, written by the eval script. There is no separate `evaluation/` tree for new writes.

`index.json` lists staged, promoted, and leaderboard pointers at `pass_<n>/model_<model_id>/`.

The price tape is built by `scripts/data/build_tape.py` from Timescale `databento.tbbo`. Each row is one CME print for the session's lead outright (`active_contract`; calendar spreads excluded), 07:00–15:15 America/Chicago: `ts`, `price`, `size`, `side`, `bid_px`, `ask_px`, `bid_sz`, `ask_sz`. `<instrument>` is the root whose prices are in the file (`NQ`), not the outright month. New sessions write `tape_<YYYY-MM-DD>.parquet` under `sources/price_tape/NQ/` so a day can be appended without rewriting history. Readers (`src/features/tape.py`) prefer that directory and fall back to `sources/features/parquet_tape/`. Compaction of the existing daily files into `partNNN.parquet` is listed in the file map and has not been run.

`build_move_matrix.py` and `build_target_matrix.py` write one optional population cache at `pass_<n>/training_matrix.parquet` when given `--objective`, and they record the join spec on the manifest and in `model.json`. With `--objective` and `--run-ts`, the second pass reads the previous pass's compact OOF for the gate and names that pass's population as its feature source. It does not copy those feature columns into its own rows. `eval_move_model.py` and `eval_target_model.py` join the spec when `--dir` is omitted. `--out` and `--dir` remain explicit overrides. The silent `move_matrix_v4` / unsuffixed `target_matrix` output defaults are gone.
