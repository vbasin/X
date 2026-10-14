# SCRIPT_INVENTORY.md — Script Hygiene Classification

**Status: CANONICAL, CURRENT, but directory-level not individually-verified for every file.** ~130 scripts exist under `scripts/`. This document classifies by directory-level pattern (verified via listing + the entry points this session actually used/read) rather than claiming to have individually read all 130 — where a specific script's classification is asserted with confidence (this session directly ran, read, or modified it), it's named explicitly. Everything else inherits its directory's default classification, flagged as such.

Classification vocabulary (per the original task spec):

- **CANONICAL ENTRY POINT** — the intended way to invoke a responsibility today. Shared experiment commands are `python -m src.experiment.cli`. There is no `src/canonical` package.
- **IMPLEMENTATION-SPECIFIC** — one track's concrete implementation of a canonical responsibility (not itself the interface).
- **EXPERIMENT-SPECIFIC** — a one-off analysis/comparison script, not meant to be re-run as part of a standard workflow.
- **DIAGNOSTIC** — inspection/reporting tooling, doesn't change any state.
- **MIGRATION/UTILITY** — data reshaping, patching, or one-time fix scripts.
- **DEPRECATED** — superseded; kept for reference only.

No script was moved, renamed, or deleted to produce this document.

## 1. `scripts/features/` (38 files) — Track B + Track C feature/label/eval pipelines

| Script | Classification | Note |
|---|---|---|
| `build_move_matrix.py` | **CANONICAL ENTRY POINT** (Track C Outcome+Feature Builder) | This session's own verified working entry point for Stage-1-equivalent dataset builds |
| `build_target_matrix.py` | **CANONICAL ENTRY POINT** (Track C Decision Policy + Outcome Builder) | Verified this session; note `--resting-entry` defaults False (immediate-fill, optimistic) |
| `eval_move_model.py` | **CANONICAL ENTRY POINT** (Track C Model Strategy + Validator) | Verified this session; walk-forward folds, `n_eff` reporting |
| `eval_target_model.py` | **CANONICAL ENTRY POINT** (Track C Model Strategy + Economic Evaluator, informal) | Verified this session; console-only cost-sensitivity block (`docs/ARCHITECTURE.md` §1.8) |
| `analyze_move_scores.py`, `analyze_context_scores.py` | **DIAGNOSTIC** | SHAP analysis; verified/extended this session |
| `build_session_profiles.py` | **IMPLEMENTATION-SPECIFIC** (Track C Feature Builder input) | Verified this session |
| `build_native_tf_{15s,1m,5m,htf}.py`, `build_training_matrix.py` | **IMPLEMENTATION-SPECIFIC** (Track B Feature Builder) | Per `docs/architecture/FEATURE_ENGINE.md` — native-TF hive builders |
| `build_labels_5s.py`, `build_dual_binary_15s.py`, `build_first_passage.py`, `build_path_labels.py` | **IMPLEMENTATION-SPECIFIC** (Track B Outcome Builder, multiple objective variants) | `build_labels_5s.py` defaults `--objective nq-50pt-econ` per this session's audit |
| `export_feature_catalog_csv.py` | **DIAGNOSTIC** | |
| `explain_trade.py` | **DIAGNOSTIC** | |
| `report_first_passage.py` | **DIAGNOSTIC** | |
| `probe_stop_rule.py`, `validate_delta_proxy.py` | **DIAGNOSTIC** | |
| `patch_15s_rth_flow.py`, `patch_15s_target_geometry.py`, `patch_iv_fast.py`, `patch_parquets.py` | **MIGRATION/UTILITY** | Name pattern (`patch_*`) is self-describing |
| `migrate_legacy_labels_5s.py`, `migrate_parquet_5s_hive.py`, `convert_legacy_label_contracts.py` | **MIGRATION/UTILITY** | |
| `backfill_t_stop.py`, `cache_large_prints.py`, `monitor_parquet_rebuild.py`, `build_data_capability.py` | **MIGRATION/UTILITY** or **DIAGNOSTIC** (unread individually this pass — name-inferred) | Flagged, not individually verified |

## 2. `scripts/search/` (15 files) — Track B Orchestrator

| Script | Classification | Note |
|---|---|---|
| `hpo_model_search.py` | **CANONICAL ENTRY POINT** (Track B Orchestrator) | Verified this session: scores every trial against the same single OOF split by default — see `project_manifest.yaml`'s open findings |
| `hpo_search.py` | **DEPRECATED relative to `hpo_model_search.py`** | Verified this session: older Optuna path keyed to a different `program_id`, distinct 70/15/15 split mechanism, not the one used for current `nq-50pt-econ` runs |
| `gate_sweep.py`, `feature_group_ablation.py`, `permutation_importance.py`, `regime_analysis.py`, `regime_filter_eval.py`, `run_ablation.py`, `run_feature_subset.py`, `analyze_tbm_combos.py`, `integrated_gradients.py` | **EXPERIMENT-SPECIFIC / DIAGNOSTIC** (unread individually this pass — name-inferred) | Flagged, not individually verified; `integrated_gradients.py` is confirmed to reference `data/legacy/checkpoints/model_v01*` (Track A) per this session's audit |
| `run_portfolio_sim.py` | **IMPLEMENTATION-SPECIFIC** (Track B Economic Evaluator, one of the divergent implementations — `docs/ARCHITECTURE.md` §1.8) | |
| `wait_gpu_result.py` | **MIGRATION/UTILITY** | |
| `hpo_monitor.py` | **DIAGNOSTIC** | |

## 3. `scripts/serving/` (3 files) — Track B Serving

| Script | Classification | Note |
|---|---|---|
| `promote_model.py` | **CANONICAL ENTRY POINT** (Track B Model Lifecycle) | Verified this session: wraps `src/serving/promote.py`, manual/human-invoked |
| `live_predict_display.py` | **DIAGNOSTIC** | Verified this session: standalone, defaults to `nq-50pt-tbm` objective (not `nq-50pt-econ`), not wired to any running service |

## 4. `scripts/data/` (29 files) — data acquisition/ingestion, all tracks' shared substrate

**Not individually re-verified this pass.** Directory-level default: **IMPLEMENTATION-SPECIFIC** (data ingestion layer, shared across all tracks) for `download_*`/`ingest_*`/`backfill_*`/`build_tape.py`/`build_path1s.py`/`create_tbbo_tf_caggs.py`; **MIGRATION/UTILITY** for `patch_*`/`repair_*`/`purge_*`/`recompress_*`. `databento_cost_check.py`/`databento_verify.py` are **DIAGNOSTIC**. `build_tape.py` specifically is confirmed **CANONICAL ENTRY POINT** for Track C's raw print tape (`docs/STORAGE_CONTRACT.md` §1) — deliberately extends sessions to 15:15 CT.

## 5. `scripts/ops/` — operational tooling

**Not individually re-verified this pass.** `apply_vm_resource_limits.py` referenced by `docs/architecture/SYSTEM.md` — **CANONICAL ENTRY POINT** for VM resource sizing. `export_snapshot.py`, `tail_live_signals.py` — **DIAGNOSTIC** (name-inferred).

## 6. `scripts/agents/` — Research Director tooling

**Not individually re-verified this pass.** `run_rd_on_model.py`, `rd_dashboard.py`, `backfill_mlflow.py` — **IMPLEMENTATION-SPECIFIC** (Track B Orchestrator), name-inferred, not individually read.

## 7. `scripts/archive/` (27 files)

**DEPRECATED by directory convention** — the directory name itself is the classification. Confirmed consistent with `docs/architecture/TREE_INVENTORY.md`'s own "archive/ — only true supersessions" rule.

## 8. Reading further

For the responsibility each canonical entry point maps to, see `docs/ARCHITECTURE.md`. For which entry point corresponds to which stage of the canonical workflow, see `docs/RESEARCH_WORKFLOW.md` §1's table (already lists the highest-confidence entries from this document in workflow order).
