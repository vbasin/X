# FUNCTIONAL_COVERAGE.md — Bottom-Up Functional & Persistence Audit

**Status: CANONICAL, CURRENT (2026-09-22, third restructuring pass).**

This document is the *bottom-up* counterpart to `docs/ARCHITECTURE.md`. `ARCHITECTURE.md`
states the intended responsibility decomposition; this file records what the code
**actually does**, derived by tracing imports, call paths, config reads and output
writes — not by inferring from filenames.

It exists because the second restructuring pass designed a canonical architecture
*before* auditing the code it was meant to describe, and consequently proposed
storage paths and identity mechanisms that either already existed in better form
or had never been adopted. See §6 for the corrections that produced this file.

**Method and its limits.** Findings below were produced by four scoped read-only
audits plus direct inspection. Every material claim cites `file:line`. Claims
marked UNVERIFIED were not confirmed and must not be relied on. No database was
queried and no file under `data/` was modified, moved or rebuilt to produce this
document.

---

## 1. Scope coverage

| Scope | Files | Status |
|---|---|---|
| `src/features/` + `src/training/` | 78 | Audited (§2, §3) |
| `src/search/`, `src/models/`, `src/evaluation/`, `src/objectives/` | 60 | Audited (§2, §5, §10) |
| `src/backtest/`, `src/paper/`, `src/execution/`, `src/serving/`, `src/external_models/` | 39 | Audited (§2, §10) |
| `src/common/`, `src/webui/`, `src/research_director/`, `src/scoping/`, `src/gpu_runner/`, `src/acquisition/`, `src/ingestion/`, `src/agentic/` | 93 | Audited (§2, §11) |

All 21 `src/` packages (~85k lines) covered.

---

## 2. Responsibilities the 12-responsibility model does NOT cover

This is the most consequential result of the audit. `docs/ARCHITECTURE.md`'s twelve
responsibilities (Data Contracts, Feature Builder, Outcome Builder, Model Strategy,
Validator, Calibrator, Decision Policy, Economic Evaluator, Experiment, Model
Lifecycle, Serving, Orchestrator) are a reasonable decomposition of the *research*
problem. They are **not a complete decomposition of this repository**. Real,
load-bearing capabilities have no owner under that model.

Per the task rule "if meaningful existing functionality does not fit naturally into
the current 12 responsibilities, revise the architecture rather than forcing it into
an existing box", these are recorded as gaps in the model, not as misplaced code.

| # | Uncovered responsibility | Evidence | Why no existing slot fits |
|---|---|---|---|
| U1 | **Dataset / Store Registry** | `src/features/stores.py` (~47 call sites across search, backtest, gpu_runner, webui, evaluation) | Resolves store id → hive dir → catalog doc → column view → loader, gated on objective keys (`stores.py:131,142`). It builds nothing (not Feature Builder), identifies no run (not Experiment), and is not schema/causality (not Data Contracts). Highest-fan-in module in its scope. |
| U2 | **Run Observability / In-flight Telemetry** | `src/training/job_progress.py:51-56,183-186,215-222` | Live heartbeat, per-fold OOF checkpointing, crash forensics, dual-write to an SSH-visible durable dir. Not Experiment (run identity/config), not Orchestrator (scheduling). It is what makes a long training run debuggable at all. |
| U3 | **Process liveness / compute arbitration** | `src/features/live_runtime.py:139-149` publishes a heartbeat that `src/webui/app.py:307,353` reads to decide whether to skip its own `extend_date` | A leader-election/authority contract between two processes. No responsibility covers "which process owns the current bar". |
| U4 | **Trade / Contract Definition** | `src/training/cascade_policy.py` (48 lines) | Defines what "the same trade" *is*; both the Outcome Builder (`move_label.py:39`) and the Decision Policy (`build_target_matrix.py:63`) derive parameters from it. Upstream of both, owned by neither. Has zero config surface (all literals), so config validation cannot see it. |
| U5 | **Sample / Era Eligibility** | `src/training/path_labels.py::session_r`, `min_stop_ticks` (docstring `:2332-2335`) | Marks whole sessions `usable=False` when the normalised stop falls inside the spread. Not a label, not invalid data, not model quality — it is *data eligibility for a geometry*. |
| U6 | **Cost-aware training objective** | `src/training/cost_matrix.py:216,368,410` | Injects the cost matrix into the Keras loss / XGB / LGBM custom objective. Economic Evaluator scores *after the fact*; this changes what the model **learns**. It is one of the ≥10 divergent economic implementations enumerated in §10, and **the only one that alters training** rather than post-hoc scoring — which is why Cost Model is a cross-cutting responsibility in its own right (`docs/ARCHITECTURE.md` §1.0) rather than a part of Economic Evaluator. |
| U7 | **Query generation inside feature code** | `src/features/mbp_depth.py:786 level_sum_sql`, injected at `engine.py:1050` | A feature module emits a SQL fragment the engine splices into a TimescaleDB query — a data-access-layer concern embedded in Feature Builder. |
| U8 | **Falsified-hypothesis preservation** | `src/training/stop_rule.py` docstring `:2402-2409` | A measured refutation of its own premise, deliberately retained "so the finding stays reproducible rather than asserted". The model has Experiment (running one) but nothing for retaining a negative result as executable evidence. Would be destroyed by any naive "delete unused modules" pass. |

**Boundary-crossing parameter (not a missing responsibility, but unowned):**
`src/features/target_geometry.py` encodes the Decision Policy's profit target
(`DEFAULT_MOVE_THRESHOLD = 50.0`) as an *input feature* (`lg_target_v_atr`), and
`stores.py:214` attaches it automatically on every 15s load — so a policy parameter
leaks into the feature matrix invisibly.

**Package-boundary violations** (any directory-keyed mapping will mis-assign these):
`src/training/entry_state.py` and `src/training/auction_levels.py` are feature
builders living in `src/training/` (`project_manifest.yaml:46` already lists
`entry_state.py` under *features*); conversely `src/training/path_features.py:2293`
imports `src.backtest.portfolio_sim`.

**Unclassified fourth track.** The 1-second path lineage (ADR-052/053) —
`src/training/path_labels.py`, `path_features.py` → `src/search/path1s_folds.py` →
`src/models/heads.py` (hazard / competing risks) — is not Track A, not the
`nq-50pt-econ` objectives loop, and not the MNQ cascade. `project_manifest.yaml`
has no entry for it.

---

## 3. Persistence & path findings

### 3.1 The canonical resolver already exists

`src/common/data_paths.py` (675 lines, **97 importing modules**) is a mature logical
path resolver: logical-name→path map (`_LAYOUT`), legacy alias resolution
(`_LEGACY_ALIASES`), CLI-tolerant `resolve_data_path()`, hive partition helpers, and
`_prefer()` for migration-tolerant lookup.

**Conclusion: do NOT build a parallel path layer under `src/`.** The
correct canonical move is to *extend* `data_paths.py`. This reverses the second
pass's direction of travel.

### 3.2 The largest gap in the resolver: `sources/labels/`

`data/sources/labels/` holds 30 entries — Track C's entire outcome storage plus the
`firstpass` family — and **has no `_LAYOUT` key**. As of the sixth pass it was
hand-constructed in fewer sites: 4 Track C builder/eval entry points now route
through the new `labels_dir()` (`scripts/features/build_move_matrix.py`,
`build_target_matrix.py`, `eval_move_model.py`, `eval_target_model.py` --
byte-identical paths, verified). **Still hand-constructed**:
`src/webui/datasources.py:827,837,922`, `src/gpu_runner/job_exec.py:75`,
`src/training/label_store.py` (routes through `labels_dir()` since the third
pass; see §4).

**CORRECTED (sixth pass):** an earlier version of this section called
`data_paths.py:215`'s `parquet_labels_15s_dir()` a bug -- "the layout key and
its fallback disagree" -- because it resolves `layout("parquet_labels_15s")`
(`sources/features/...`) first and `sources/labels/parquet_labels_15s` second.
That is **not** a disagreement: `_prefer(canonical, legacy)` is the
resolver's own established convention, used identically in 23 other
accessors in the same file (e.g. `parquet_5s_dir()`, `paper_dir()`,
`execution_dir()`). Reclassified: `COMPAT` -- an intentional legacy fallback,
consistent with every other `_prefer()` call site, not an anomaly.

**EIGHTH PASS (2026-09-22 storage architecture pass) — full re-verification.** A dedicated bounded inventory pass (`docs/STORAGE_INVENTORY.md` §1.3, §2.3) re-confirmed everything above and added: `move_matrix`'s version is bumped by editing the hardcoded output literal in `build_move_matrix.py`'s source and rerunning, not a CLI flag; `eval_move_model.py` and `build_target_matrix.py --move-dir` both default to `move_matrix_v3`, one generation behind the producer's current `move_matrix_v4` default, so a bare invocation of either today silently uses stale input (`docs/STORAGE_INVENTORY.md` §4 STOR-2 — a live, present-tense risk, not just historical debt); `target_matrix`'s `_v3`/`_v4` (version) vs `_adaptive`/`_context`/`_context_rest` (variant) naming really does mix two axes in one namespace, independently re-confirmed; `sources/labels/1s/` (467M) is a dead ADR-107 migration artifact with zero live readers; and `docs/roadmap/ROADMAP.md:83` independently confirms the entire `firstpass` family (7 directories, 624M, including 4 with zero code references left at all) superseded by ADR-127.

**NINTH PASS (2026-09-22 targeted correction) — Track B was mislabeled content-addressed; it is contract-addressed.** `label_contract_id()` hashes `normalize_label_contract(contract)` — the semantic contract (label rule, horizon, stop/target semantics, resolution) — never the label parquet's bytes; `write_label_parts(..., force=True)` can replace every part beneath an unchanged `contract_id` directory, so the mechanism is idempotent by default, not content-immutable. Every "content-addressed"/"immutable" claim about Track B's label store in this document and `docs/STORAGE_CONTRACT.md`/`docs/STORAGE_INVENTORY.md` is corrected. A real consequence of the blur was found and fixed: `src/search/run_manifest.py` had been surfacing `label_contract.json`'s `contract_sha256` (a hash of the `_contract.json` metadata sidecar) as `DatasetRef.content_sha256` (which claims to hash dataset content) — false provenance, now fixed (`docs/STORAGE_INVENTORY.md` §4 STOR-11). `DatasetRef` gained a `build_id` field for the middle tier this blur was missing (§5a of `STORAGE_CONTRACT.md`): `content` (genuine bytes hash, never fabricated) > `build` (realization fingerprint) > `contract` (semantic hash) > `store` (mutable logical name).

**TENTH PASS (2026-09-22, Storage Implementation Tranche 1) — the ninth pass's `build_id` mechanism and outcome-path pattern were both retired and replaced.** `build_id` had been derived from a fingerprint of the OUTPUT part inventory (filenames + byte sizes) — not genuine build/provenance evidence, since it says nothing about which code, config or upstream data produced the realization. `src/search/run_manifest.py::_datasets_from_run_dir()` now always records `build_id=None`; a genuine mechanism, `label_store.write_build_sidecar()` (code SHA/dirty via new `src.common.code_identity`, builder config, upstream lineage), populates it elsewhere. `RunManifest.missing_evidence()`'s `datasets` check now also flags contract-only identity, not just store-only — a `contract_id` alone states what a dataset MEANS, not WHICH realization was used. `RunManifest.schema_version` bumped 2→3 for this meaning change (v1/v2 still readable). Separately, the ninth pass's `sources/labels/<logical_dataset_id>/<contract_id>/` outcome-path pattern conflated Track B's resolution values (`5s`/`15s`) with Track C's outcome-role nicknames (`move`/`target`) in one slot — finalized this pass into two explicit dimensions, `sources/labels/<outcome_family>/<resolution>/<contract_id>/` (`docs/STORAGE_CONTRACT.md` §17.1), with Track C's semantic contract fields classified (§19) and opt-in `--canonical-write` support wired into both builders (not executed). The `eval_move_model.py`/`build_target_matrix.py --move-dir`/`eval_target_model.py` stale-default risk this document's EIGHTH PASS entry (above) flagged is now FIXED: all three require an explicit dataset path and fail clearly rather than silently selecting `move_matrix_v3`/unsuffixed `target_matrix`.

### 3.3 Existing storage conventions, ranked

| Convention | Location | Verdict |
|---|---|---|
| **Contract-addressed label sets** | `src/training/label_store.py:44-46,79-89,111-131` → `sources/labels/{5s,15s}/<sha256[:16]>/` + `_contract.json` | **BEST IN REPO — adopt as the canonical pattern.** `label_contract_id()` hashes the normalized semantic contract, never the parquet content — see §4 and the ninth pass's correction, `docs/STORAGE_CONTRACT.md` §5a. |
| Hive feature store | `sources/features/timeframe=/symbol=/year=/month=/` via `native_tf_dir()` | Strong. Atomic per-file replace (`engine.py:520-521`). Retain. |
| Objective-keyed run dirs | `research/runs/<objective_id>/<run_ts>/` via `run_layout.runs_dir()` | **Current new-run root.** `research/custom_test/` is gone. `research/search/` is a compatibility root. |
| Version-suffixed dirs | `sources/labels/move_matrix{,_v2,_v3,_v4}`, `target_matrix{,_adaptive,_context,_context_rest,_v3,_v4}` | **ANTI-PATTERN.** Filename is not scientific identity. See §5.2. |

### 3.4 Hardcoded paths bypassing the resolver

| ID | Location | Issue |
|---|---|---|
| H1 | `src/training/config.py:190` | `_load_yaml(config_path or Path("/opt/vtech/config.yaml"))` — absolute machine path baked into the config loader. **23 importers.** Highest blast radius. |
| H2–H4 | `trainer.py:405`, `xgb_trainer.py:219`, `stack_trainer.py:342` | `Path(save_dir or "checkpoints/latest")` — resolves against process CWD |
| H5 | `stack_trainer.py:569,578,587,600` | CLI defaults pointing at `data/legacy/checkpoints/model_v015_fold*` (ARCHIVED artifacts) |
| H9 | `engine.py:205`, `trainer.py:449`, `xgb_trainer.py:221`, `stack_trainer.py:344` | The `os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")` literal re-implemented 4× — `data_paths.data_root()` exists for exactly this |
| H10 | `src/features/stores.py:23-24,62-67` | Builds `docs/reference/FEATURE_CATALOG_*.md` paths — **documentation on the runtime path** |
| H11 | `src/training/data_pipeline.py:151,169` | `date=`-partitioned snapshot layout defined nowhere else, no resolver |
| H12 | `feat_ext.py:58`, `hive_5s.py:47`, `data_pipeline.py:77,88`, `stores.py:78,159` | The `features_*.parquet` glob re-implemented in 5 places |
| H13 | ~~`src/webui/regmodel.py:653`~~ | ~~The live signal contract had no `_LAYOUT` key~~ **FIXED eighth pass**: `_LAYOUT["live"]` + `live_dir()` added; `regmodel.py`, `webui/live.py`, `execution/config.py` (3 independent hardcoded sites) all route through it now |
| H14 | ~~`src/features/session_profile.py:57-58`~~ | ~~`session_profile_dir()` had no `_LAYOUT` key, feeds the cascade's structural SSOT~~ **FIXED eighth pass**: `_LAYOUT["session_profile"]` added |
| H15 | ~~`scripts/search/regime_analysis.py:86`~~ | ~~`Path("/opt/vtech")/"data/..."` — absolute, ignored `VTECH_DATA_DIR`~~ **FIXED eighth pass**: now `research_director_dir()/"regimes"` |
| STOR-5 | 9+ scripts under `scripts/data/*.py` | Resolver bypass is the norm, not the exception, for `sources/raw/backfill` producers — too many independent call sites for one small pass; see `docs/STORAGE_INVENTORY.md` §4 |

Repo-wide, 59 files outside `data_paths.py` reference `/opt/vtech/data` literally.

### 3.5 Mutability hazards

- **`src/features/native_tf.py:161-162` is destructive by default** — `write_month_parts` unlinks every existing `features_{tf}_part*.parquet` before writing. A crashed rebuild leaves the store *empty* rather than stale. This is the normal write path for 5 timeframes.
- **`session_levels.parquet` has no run identity** — `src/features/session_profile.py:170-172` merges newest-build-wins with no run id, version or timestamp column. It is called the cascade's structural SSOT (`:1224`) and feeds `build_move_matrix.py` / `build_target_matrix.py`. **Two Track C experiments run days apart can silently rest on different structural inputs with no way to detect it afterwards** — this is a concrete, previously-unidentified mechanism behind `project_manifest.yaml`'s `cascade-population-shift` finding.
- **EIGHTH PASS: `research/search/_fold_cache/` grew from 14G (2026-08, `docs/operations/DISK_RECLAIM_2026_08.md`) to 39G (2026-09-22) with no eviction** — a pure, content-keyed, fully reproducible cache (confirmed no evidence-loss risk), but it is now ~89% of `research/`, the single largest directory in `/opt/vtech/data`, and sits nested inside the directory that same operations doc separately designates "never delete — OCI-synced audit trail." The designed LRU eviction (that document's own deferred "Stage 3") was never implemented. See `docs/STORAGE_INVENTORY.md` §4 STOR-1.

---

## 4. Dataset identity: the pattern already in the repo

`src/training/label_store.py` implements, today, what the second pass proposed to
invent:

```python
label_contract_id(contract)      # sha256 of the NORMALISED semantic contract, 16 hex
normalize_label_contract(...)    # strips non-semantic fields (contract_id, created_at)
label_set_dir(resolution, c)     # physical path DERIVED from logical identity
write_contract(d, contract)      # self-describing `_contract.json` sidecar
contract_matches(stored, want)   # verify before use
write_label_contract_snapshot()  # writes contract + sha256 + part inventory INTO the run dir
```

Properties: contract-addressed (**not** content-addressed — `label_contract_id()` hashes
the normalized semantic contract, never the parquet bytes; `docs/STORAGE_CONTRACT.md` §5a),
idempotent by default (`:117-119` returns existing parts unless `force=True`, which replaces
every part beneath an unchanged `contract_id` -- **not content-immutable**), self-describing,
and lineage-recording. This is the **canonical dataset-identity mechanism** and
`RunManifest.data_reference` should be structured
after it rather than remaining a free-form mapping.

The contrast is stark: Track B solved dataset identity properly and Track C, the
more actively developed track, encodes identity in directory suffixes (`_v3`, `_v4`).
**The repo already contains the cure for its own worst identity problem.**

---

## 5. Run identity: what already exists

### 5.1 The adopted convention

Present: new runs are `data/research/runs/<objective_id>/<run_ts>/`, written by `src/search/controller.py` through `run_layout.new_run_dir()`. `data/research/custom_test/` is gone. The artifact list below is what a Track B run still contains inside that directory:

| Artifact | Contents (verified from real run dirs) |
|---|---|
| `run.json` | `kind`, `objective_id`, `run_ts`, `out_dir`, `champion`, `gate_result{status,champion_id,champion_precision,champion_dir_f1,champion_meets_target,passed_ids,target_ids,failed_ids,reasons}`, `n_candidates` |
| `search_result.json` | above plus `experiment_type`, `n_ok`, `n_failed`, `folds`, `candidates` |
| `folds.json` | `split` (`expanding_equal_count` \| `expanding_calendar`), `n_folds`, `note`, per-fold `train`/`oof` with `start`, `end`, `n_bars`, `n_days`, `n_down`, `n_flat`, `n_up` |
| `spec.manual.yaml`, `topology.yaml` | resolved experiment spec |
| `oof_<model>.parquet` | out-of-fold predictions |
| `diagnostics_<model>.json`, `progress[_<model>].json` | per-model diagnostics, in-flight telemetry |
| `label_contract.json` | dataset lineage (`label_store.py:138-155`), when `search.input == native_tf` |

This is the Track B run record. `manifest.json` is written with it. The directory root is `research/runs/`.

### 5.2 Corrections to previously-recorded path claims

1. **`experiments/` is not a real path.** It does not exist on disk, has no `_LAYOUT` key, and appears nowhere in code as a path. `docs/architecture/DATA_LAYOUT.md` §4, its only source, is titled **"Target shape (recommended)"** — an unadopted proposal. The second pass's "path correction" replaced an invented path with a *different* unadopted one. Classification: **TARGET/PLANNED BUT NEVER ADOPTED**.
2. **The `eval.walk_forward_folds` claim was wrong twice; only the enumeration below is reliable.** First claim ("real runs use `n_folds=1` in every case") — false. Second claim ("`n_folds` is never 1") — also false, from a misread `sort | uniq -c`. Full enumeration of all **54** `folds.json` while those runs lived under `data/research/custom_test` (now `research/runs/`): `n_folds` is 3 in 38 runs, 6 in 12, 4 in 1, and **1 in 3** runs (`nq-50pt-econ`, 2026-08-22). The verified statement is a *two-engine* one: `expanding_equal_count` (`src/search/controller.py::_build_folds`) reads `eval.walk_forward_folds` as an INPUT; `expanding_calendar` (`src/validation/walk_forward.py::schedule_from_eval`) never reads it and reports its derived count as an OUTPUT. `src/search/walk_forward_calendar.py` does not exist. Track C no longer has `eval_*_model.py::_folds()`. The three single-fold runs used `expanding_calendar` and therefore had **no walk-forward validation at all** while their objective doc declared `walk_forward_folds: 3`. Full correction history in `docs/CONFIG_AUDIT.md`.

### 5.3 Reproducibility gap

`src/common/repro.py` provides `get_git_sha()`, `set_global_seed()`, `hash_dict()`,
`log_experiment()`. Of these:

- `set_global_seed()` has exactly three callers, all inside `src/training/`: `trainer.py:403`, `xgb_trainer.py:217`, `stack_trainer.py:340`. **Zero calls from `src/search/`, `src/models/` or `src/evaluation/`** — i.e. none from the production search/candidate path, whose tabular plugins pass no `random_state` either.
- `get_git_sha()` has **zero callers**. The only git SHA that reaches a run directory is `finalize_manifest.json`'s, written by a helper with no dirty-tree guard.
- `log_experiment()` / `hash_dict()` are called only by `src/training/trainer.py`. That module is **not dead** — it is reached from `src/training/walk_forward.py:30` and from `scripts/search/{hpo_search,feature_group_ablation}.py` — but it is not on the `src/search/controller.py` production path.

**No live run path records both a seed and its code revision.** Both primitives exist and are unwired. See `project_manifest.yaml` → `no-run-records-seed-or-code-revision`.

---

## 6. Duplication findings

Ranked by risk, not count.

| ID | Duplication | Risk |
|---|---|---|
| DUP-12 | **Six outcome/label builders** — `labels.py`, `atr_labels.py`, `dual_binary_15s.py`, `path_labels.py`, `first_passage.py`, `move_label.py` | `path_labels.py:2316-2319` documents that `labels.py` **mislabels** (compares 5s *closes* to both barriers, so a bar trading 9pt against entry and closing +1 is a "clean winner", ADR-052). `labels.py` remains the PRODUCTION_WIRED Track B outcome builder. |
| DUP-9 | **Two `merge_asof` implementations + ~10 inlined `searchsorted(...,"right")-1` idioms** (`htf_context.py:174`, `entry_state.py:134`, `flow_state.py:292,328`, `structural_transitions.py:111,159`, `entry_state.py:182,337`, `move_label.py:120-122`, `stop_rule.py:141,207`) | AGENTS.md §7 mandates `src/common/causal.py::as_of_index()`. **Grep across `src/features/` + `src/training/`: zero hits.** The canonical primitive has zero adoption. |
| DUP-13 | **Two feature loaders with different output schemas** — `stores.py:186-219` silently attaches `attach_rth_session_features` + `attach_target_geometry` for 15s (`:210-215`); `data_pipeline.py:91-135` does not | A frame loaded via `data_pipeline` is **missing columns** a frame loaded via `stores` has. No assertion guards it. Silent train/eval schema divergence. |
| DUP-2 | **~9 independent session-boundary definitions** (`native_tf.py:60-62`, `bar_features.py:24-25`, `auction_structure.py:19-20`, `cash_session_profile.py:62`, `time_context.py:26`, `move_label.py:51`, `path_labels.py:89`, `engine.py:211`, `session_vp.py:5`) | `src/common/market_session.py` exists but is used by only 3 modules in scope. `time_context.py:26` carries a docstring recording a *previously shipped bug* where this value was 08:00. |
| DUP-1 | **Six volume-profile / value-area implementations**; three independent `VA_PCT`/`tick_size` constant pairs | Two modules acknowledge the conflict in prose rather than resolving it (`session_profile.py:18`, `prior_volume_profile.py:3`). |
| DUP-3 | **Five ATR derivations** | `target_geometry.py` docstring states two columns claim the same quantity from different inputs, both shipped. |
| DUP-6 | **Three near-identical native-TF session builders** (`native_tf_1m.py:455`, `native_tf_5m.py:443`, `native_tf.py:320`) | ~1,500 lines of structurally parallel code. |
| DUP-7 | **Two feature-orchestration paths** — `engine.py:660-745` (batch) vs `live_runtime.py` (warm/incremental) | Train/serve parity is **asserted, not enforced** by a shared driver. |
| DUP-14 | `FEATURE_GROUP_PREFIXES` copied in `trainer.py:177`, `xgb_trainer.py:55`, `stack_trainer.py:72` | Two of the three are dead. |
| DUP-15 | Two walk-forward validators — equal-count in `search/controller.py::_build_folds`, calendar in `src/validation/walk_forward.py`. `search/walk_forward_calendar.py` is gone | — |

---

## 7. Config findings (additions to `docs/CONFIG_AUDIT.md`)

- **`features.enabled_groups` is very likely dead.** Defined at `src/training/config.py:35-57`, merged at `:196-200`; **zero hits inside `src/features/engine.py`**. The engine gates each module on data availability (`engine.py:660-745`), never on this key. `archive/root_docs/config copy.yaml:80,86` still carries per-group toggles, suggesting they were once honoured. Same class as the confirmed `attention_heads`/`slippage_model` divergences. *Basis: absence-of-grep in `engine.py` only — a dynamic lookup elsewhere would falsify it.*
- **Embedded reference data with an expiry.** `src/features/economic_calendar.py:379-398` hardcodes `_FOMC_DATES`/`_CPI_DATES` ending **2026-12**, engine-wired at `:836`. After that date every `ec_` feature silently degrades to "no events" with no error. No contract, config or test guards it.
- **Documentation is on the runtime path.** `src/features/stores.py:23-24,62-67` reads `docs/reference/FEATURE_CATALOG*.md` at runtime, consumed by `src/search/custom_test.py:87`. Editing a doc file can change runtime behaviour.

---

## 8. Unreachable code (report-only)

Confirmed zero-caller: `xgb_trainer.py` (442 lines), `stack_trainer.py` (639),
`walk_forward.py` (127), `equity_context.py` (85), `macro_sentiment.py` (63).
Built-but-unwired: `cvd_structure.py` (185), `prior_volume_profile.py` (418) — both
are cited in ADR-122 and `docs/roadmap/ROADMAP.md:216-217` as delivered while being
functionally unreachable. Probe-only: `stop_rule.py` (276, and see U8 — it is
deliberately retained evidence).

Total ≈2,235 of 20,983 lines (10.7%) unreachable from any production entry point in
this scope. **Per AGENTS.md §4/§11 none may be removed without explicit user
approval. This is a report, not a removal plan.**

---

## 9. UNVERIFIED items

- The enclosing condition of `src/features/engine.py:748` `_skip = {"hawkes_clustering","wavelets","sub_bar_dynamics"}` — if it is the default path, three engine-wired modules are effectively dead despite their prefixes appearing in three feature-group maps and in `stores.py:1352` STORE_VIEWS.
- Whether `features.enabled_groups` is read via a dynamic lookup outside `engine.py`.
- Whether `walk_forward.py`'s `--out` default is supplied by an operator shell script outside `src/` and `scripts/`.
- Whether the declared `eval.hold_out` pin (`"e68510982d9f"`) is actually excluded from HPO trial scoring — `DEFERRED — REQUIRES DATA-SIDE VALIDATION`.

---

## 10. Economic evaluation: the real inventory

`project_manifest.yaml`'s `economic-evaluation-inconsistency` finding records **4**
divergent implementations. The audited count is **at least 10**:

`src/backtest/engine.py` · `src/backtest/cascade_backtest.py` ·
`src/backtest/portfolio_sim.py` · `src/backtest/weights_score.py` (two sub-modes:
`_resolve_signal_oco`, `_resolve_signal_on_close`) · `src/paper/engine.py` ·
`src/execution/ledger.py` · `src/evaluation/_kpi.py` ·
`src/external_models/{pine_sar_ema,orb,vwap_cross}.py` ·
`src/webui/chart_models/interpreter.py` · plus `src/training/cost_matrix.py`, which is
economics injected into the **training objective** rather than post-hoc scoring (U6).

### 10.1 Where they materially disagree

| # | Divergence | Consequence |
|---|---|---|
| 1 | **Slippage: 0 vs 1.0 pt/side vs 2 ticks on stops** | `engine.py`, `weights_score.py`, `cascade_backtest.py` apply **zero**; `paper/engine.py` applies 1.0/side (`policy.py:84-86`); `portfolio_sim.py` 1 tick entry / 2 ticks stop. On a 6-pt stop, 1 pt/side is **33% of the risk unit** — enough to flip the sign of a marginal strategy. Mechanical driver of the backtest-vs-paper gap. |
| 2 | **Contract multiplier: 1.0 vs 2.0 vs 20.0 vs 20×qty** | `engine.py:347` defaults **1.0** (so `total_value` is points mislabelled as dollars); `cascade_backtest.py:250` hardcodes **2.0** (MNQ) and the WebUI never passes it; `weights_score.py:1074`, `portfolio_sim.py:95`, `execution/ledger.py:46` use **20.0** (NQ). **Track B and Track C numbers shown in the same UI are 10× apart for identical points.** |
| 3 | **`sharpe_daily` un-annualized vs ×√252** | `engine.py:468` vs `cascade_backtest.py:202` — a **15.87×** discrepancy on the same-named UI field. |
| 4 | **`sharpe_trade` is a Sharpe in one place and a t-statistic in another** | `cascade_backtest.py:205` computes `mean/std×√n` and labels it `sharpe_trade`. It scales with √n: a 400-trade run reports ~20× `engine.py`'s value for identical per-trade economics. |
| 5 | **PnL source: realized fill vs fixed barrier vs label-implied** | `weights_score.py:840,847` books `±stop/target` exactly while recording an unused real `exit_px`; `cascade_backtest.py:318` and `_kpi.py:81` are barrier/label-implied. Fixed-barrier PnL systematically removes adverse fill dispersion. |
| 6 | **`exposure_frac` denominators differ** | `engine.py:494` uses `holding_bars/n_bars`; `cascade_backtest.py:196-197` uses the **label horizon** (`n_days × 1800s`). Not comparable. |
| 7 | **ddof inconsistent inside one function** | `cascade_backtest.py:201` (pandas, ddof=1) vs `:205` (numpy, ddof=0). |

### 10.2 Statistical inference

Repo-wide grep for `ttest|t_stat|p_value|binomtest|bootstrap|deflated` across
`src/backtest/`, `src/paper/`, `src/execution/`, `src/serving/`,
`src/external_models/` returns **zero hits**. `src/evaluation/overfitting.py:96`
(`deflated_sharpe_ratio`) and `src/evaluation/uniqueness.py:89` (`weighted_tstat`)
exist but are called from **no** backtest/paper/serving path.

`src/research_director/evaluator.py` is the single exception and is **stricter than
anything in `src/validation`**: it requires delta ≥ delta_min AND guardrail
non-regression AND improvement on ≥⌈⅔⌉ folds AND a bootstrap 95% CI on the aggregate
fold delta excluding zero. It lives inside the LLM-orchestration package.

**Implication for `src/evaluation/economic.py`:** its IID `t_stat_iid` is
not a regression against repo practice — it is the *only* trade-level statistic
anywhere. That is precisely why it must stay descriptive and not render verdicts
(see the inference note in that module).

### 10.3 Other confirmed defects (report-only — all are behavior changes)

- `src/backtest/weights_score.py` applies **no commission and no slippage** in either sub-mode; its classic path delegates to `engine.run_backtest`, which reads the never-set `costs.slippage_points`.
- `src/external_models/runner.py:345` multiplies by `point_value × qty` while `pine_sar_ema.py:545` has already scaled PnL by `abs(pos)` — `net_value` therefore scales as **qty²**. Harmless at the default `qty=1`; the UI exposes 1–20. `orb.py` and `vwap_cross.py` do not have this bug.
- `src/backtest/cascade_backtest.py` contains zero occurrences of `cost`, `slip`, `commission` or `fee`, is the WebUI-facing Track C backtest, and has **no test coverage**.
- `cascade_backtest.py:326,330`: `entry_ts` is the decision bar while `entry_px` is the resting fill `t_fill` seconds later — two different instants in one persisted row.
- `src/backtest/weights_score.py.orig` (40 KB) is a committed merge-conflict backup, imported by nothing.

---

## 11. Run identity: seven uncomposed mechanisms

Extending §5. Seven identity mechanisms exist; **none composes with the others.**

| # | Mechanism | Captures | Critical defect |
|---|---|---|---|
| A1 | `run.json` (`controller.py:2269-2288`) | `objective_id` + wall-clock `run_ts` | **Schema drift already real**: newer runs carry `status/n_ok/n_failed`; older ones carry `champion/gate_result` instead. **No `schema_version` field** distinguishes them. Captures no git SHA, config, dataset hash, seed, parent or hypothesis. |
| A2 | `label_contract.json` (`label_store.py:136-157`) | `contract_id`, `contract_sha256`, `source_dir`, part inventory | **The strongest artifact in the repo** — but `label_store.py:138` returns early unless `search.input == "native_tf"`. The config-pinned production objective `nq-50pt-econ` is a 5s-store objective, so **none of its runs has one.** The best identity mechanism is unavailable to the primary track. |
| A3 | `finalize_manifest.json` (`finalize.py:120-143`) | `git_sha`, `version`, `search_run_ts`, `spec_params`, `mlflow_run_id` | **The only place a git SHA reaches a run directory.** `_git_sha` (`controller.py:826-838`) is best-effort with **no dirty-tree guard** (ADR-022 removed it) — a manifest can claim a clean SHA while the tree diverges. |
| A4 | Frozen eval contract (`contract.py:85-93`) | `hold_out_hash`, `hold_out_dates`, `objective_snapshot` | **Broken for the production objective**: `data/evaluation/nq-50pt-econ/contract_v1.json`'s `objective_snapshot` has only `['note','hold_out_source','primary']`. `_kpi.py:34-39` then defaults to `mae_tolerance=5` — so contract-scored `nq-50pt-econ` candidates are priced at a **5-pt stop instead of the declared 6-pt**. Also: `score()` only checks the hash `if data_hash is not None`, and `controller._record_all` never passes it — the comparability guarantee is **inert**. |
| A5 | Seeds | — | **Absent.** `repro.set_global_seed` is called only from the three legacy trainers. Zero calls from `src/search/`, `src/models/`, `src/evaluation/`. The four production tabular plugins pass no `random_state` (`tabular_plugins.py:371,390,413,428`). **No run artifact records a seed; Track B runs are not deterministic.** |
| A6 | `compose.json` (`tf_oof_meta.py:1084-1107`) | `bases[]`, `base_runs{}`, `git_sha` | **Real working child→parent lineage** — limited to the `tf_oof_meta` topology. No other run kind records a parent, and no run records a *baseline it should be compared against*. |
| A7 | Leaderboard rows (`leaderboard.py:82-97`) | `candidate_id`, scores, `mlflow_run_id` | **No `run_ts`** — a leaderboard row cannot be traced to the run directory that produced it, only to an `mlflow_run_id` that is frequently `null` (MLflow logging is best-effort and swallowed, `controller.py:2151-2153,2188-2189`). `get_leaders` dedupes by `candidate_id` keeping max precision, so **an older run can permanently mask a newer one.** |

Plus `results_index.py:128-141`, whose trait key hashes only `traded products` +
`strategy_keywords` — **two objectives differing in labels, costs, session and feature
view collide into the same key**; it is gated off for every Custom Test run anyway
(`controller.py:512`).

---

## 12. Platform-layer capabilities with no responsibility owner

Nineteen operational capabilities were inventoried; **thirteen have no home** in the
twelve. Full detail in §2 of `docs/ARCHITECTURE.md` (revised). Highlights:

- **Market-data acquisition & live streaming** (`src/acquisition/`) — the 24/7 service, chain subscription, reconnection and weekend-halt handling. `ARCHITECTURE.md` names these under Data Contracts, but Data Contracts is defined as schema/causality.
- **Vendor cost governance** (`acquisition/historical.py::cost_check`) — the mechanical enforcement of AGENTS.md §5's "$0.01 → stop and ask" rule.
- **Bulk ingestion, backfill and data-quality/gap detection** (`src/ingestion/`).
- **Environment & resource management** (`common/vm_config.py`) — AGENTS.md §9 makes `workers_for()` mandatory repo-wide; 28 modules depend on it.
- **Remote GPU lifecycle, job queue, admission control, GPU→CPU fallback, health watchdog** (`common/gpu_provider.py`, `common/oci_queue.py`, `gpu_runner/*`).
- **Object storage / OCI sync** (`common/object_store.py`, `gpu_runner/parquet_sync.py`, 20 sync targets).
- **Code identity / content-addressed bundling** (`common/code_bundle.py`) — sha256-verified so the GPU provably executes the dispatched bytes.
- **Job scheduling & state across restarts** (`webui/jobs.py`) — the only job-state store.
- **WebUI / HTTP API** (~62 endpoints, 11,930 LOC) — **33% of the infra scope by LOC, and the only UI for Track C.**
- **LLM research automation** (`research_director/` + `scoping/` + `agentic/` + `common/llm/` = 43 files, ~7,100 LOC) — Orchestrator nominally covers this in one paragraph; it does not cover LLM cost metering, the T1/T2/T3 evidence tiers with a human-gated write barrier, pgvector RAG, per-agent durable memory, or the deterministic promotion gate.

### 12.1 Structural hazards in the platform layer

1. **Serving depends on the WebUI.** `src/features/live_runtime.py:232,263,679`, `src/features/engine.py:359` and `src/execution/gateway.py:167-168` import from `src.webui.live`/`regmodel`; the live signal contract (`data/live/signals.jsonl`) is produced by `src/webui/live.py:1798-1890`. **Any reorganization treating `src/webui/` as a presentation layer will break serving and execution.**
2. **Config loading is not centralized** — at least 6 independent paths (`common/config.py:20`, `objectives/loader.py:21-23`, `webui/live.py:126`, `webui/datasources.py:166`, `research_director/config.py:53-57`, `training/config.py:188-190`). `src/experiment/config_validation.py` validates the *file*, not the *resolution*, so it cannot detect divergence between these paths.
3. **`research_director.agent_memory` is declared durable primary state** (`rag_memory.py:12-14`) and is **absent from all 20 OCI sync targets**. Postgres/pgBackRest is its only durability.
4. **`cme_holidays._HOLIDAYS` ends 2026-12-25** (`:46`) — every `is_trading_day` caller silently treats 2027 holidays as trading days. Same class as the `economic_calendar` expiry (§7).
5. **`gpu_runner/runner.py::ResourceGovernor` re-implements hardware detection** (`:112-137`) instead of calling `get_vm_config()`, contrary to AGENTS.md §9.
6. **`common/logging.py` (structlog) has 1 caller; 92 files use `logging.basicConfig`.** Standardized logging effectively does not exist.
7. **`research_director/sentinel.py`, the only alerting design in the repo, has zero callers.**
8. Dead/orphaned in the platform layer (grep-verified zero callers): `common/gpu_provider.py::JobSpec`+`run_job` (documented consumer `research_director/conductor.py` **does not exist**), `research_director/{sentinel,cost,llm}.py`, `scoping/{anthropic_llm,web_research}.py` (the latter being the only SSRF/prompt-injection-hardened fetcher), `agentic/{memory,critic,embeddings}.py`, `common/config.py::AppConfig`, `common/data_manifest.py::{assert_schema_compatible,diff_schemas,manifest_for_parquet}`.

---

## 13. Reverse-coverage validation

Two checks, run in both directions. This is the gate that had to pass before any
behavior-preserving integration begins.

### 13.1 Functional reverse coverage — responsibility → implementation

For every canonical responsibility in `project_manifest.yaml` (24 after the
two-plane revision), which existing implementation fulfils it?

| Responsibility | Fulfilled by | Verdict |
|---|---|---|
| data_contracts | `src/ingestion/`, `src/acquisition/`, `common/contract_calendar.py`, `features/tape.py`, `src/common/causal.py` | covered. No `src/data_contracts/` package |
| dataset_registry | `features/stores.py`, `training/label_store.py` | covered; **existing implementation is canonical** |
| features | `features/engine.py` (B); `training/entry_state.py` + `features/{flow_state,structural_transitions,cash_session_profile}.py` (C) | covered twice, deliberately |
| outcomes | `training/labels.py` (B); `training/move_label.py` (C) | covered twice |
| trade_definition | `training/cascade_policy.py` | covered |
| population_eligibility | `training/path_labels.py::session_r`, `min_stop_ticks`; cascade gate accounting | **partially covered — no single owner, no persisted artifact.** Gap |
| model_strategy | `models/catalog.py` (B); `scripts/features/eval_{move,target}_model.py` (C); `external_models/` (rule-based baselines) | covered |
| validation | Track C: `src/validation/folds_from_objective`. Track B calendar: `src/validation/walk_forward.py`. Equal-count fallback: `search/controller.py::_build_folds` | covered. Track C has no private `_folds()`. The equal-count fallback remains |
| calibration | *(nothing)* | **NOT covered.** Net-new responsibility; `schemas.CalibrationResult` defines the artifact only |
| decision_policy | `scripts/features/build_target_matrix.py` (τ gate, resting/immediate entry) | covered (C only) |
| cost_model | `training/cost_matrix.py` (training objective), objective `costs:` blocks, `EconomicAssumptions` (post-hoc) | **covered in two disconnected places.** Gap |
| economic_evaluation | `src/evaluation/economic.py`, wired into `portfolio_sim` and Track C's persisted book. Other PnL paths remain (§10) | partially consolidated. Not one implementation |
| experiment | `src/experiment/` (`manifest.py`, `artifacts.py`, `analysis.py`). New runs under `research/runs/` | covered. No `src/canonical/` package |
| model_lifecycle | `src/serving/lifecycle.py`, `serving/promote.py`, ADR-071 gate, `models/model_registry/` | covered. Vocabulary is not yet enforced on the registry |
| serving | `src/serving/`, `src/paper/`, `src/execution/` | covered; **depends on `src/webui/`** (§12.1) |
| orchestration | `research_director/`, `scoping/`, `scripts/search/` | covered |
| acquisition_ingestion · storage_sync · compute_resources · job_orchestration_state · code_identity · observability · interface_surface · research_automation | see §12 | all covered by existing code; `observability` is WEAK |

For every meaningful `src/` package, which responsibility owns it?

| Package | Files / lines | Owning responsibility |
|---|---|---|
| `src/features/` | 54 / 14,190 | features, dataset_registry, data_contracts, serving |
| `src/webui/` | 14 / 11,930 | interface_surface, job_orchestration_state, **serving (unexpectedly)** |
| `src/search/` | 24 / 11,716 | validation, experiment, orchestration |
| `src/training/` | 24 / 6,793 | outcomes, trade_definition, population_eligibility, cost_model, dataset_registry, observability |
| `src/external_models/` | 12 / 6,183 | model_strategy (rule-based baselines), economic_evaluation (each carries its own cost math — §10) |
| `src/models/` | 24 / 5,756 | model_strategy |
| `src/backtest/` | 9 / 4,762 | economic_evaluation, decision_policy |
| `src/common/` | 25 / 4,046 | storage_sync, compute_resources, code_identity, observability, data_contracts |
| `src/research_director/` | 20 / 3,350 | research_automation, orchestration |
| `src/scoping/` | 12 / 3,092 | research_automation |
| `src/evaluation/` | 10 / 2,910 | economic_evaluation, model_lifecycle (leaderboard, contract) |
| `src/gpu_runner/` | 8 / 1,954 | compute_resources, job_orchestration_state, storage_sync |
| `src/acquisition/` | 5 / 1,830 | acquisition_ingestion |
| `src/` | 21 / 1,827 | the canonical layer itself |
| `src/paper/` | 7 / 1,353 | serving |
| `src/serving/` | 5 / 1,087 | serving, model_lifecycle |
| `src/execution/` | 6 / 701 | serving |
| `src/ingestion/` | 3 / 559 | acquisition_ingestion |
| `src/agentic/` | 6 / 259 | research_automation (largely dead — §12.1) |
| `src/objectives/` | 2 / 127 | experiment — **objective identity + JSON-Schema validation; the mechanism `objective_id` rests on** |

**No `src/` package is unowned.** Total ≈ 84,400 lines across 20 non-canonical packages.

### 13.2 Persistence reverse coverage

Every artifact class in `docs/STORAGE_CONTRACT.md` §2 has a stated producer,
path, resolver (or an explicit "none"), consumer and mutability. Outcomes:

- **35 artifact classes** mapped end-to-end.
- **8** have no resolver and are hand-constructed (`sources/labels/*`, `data/live/*`, `ops/job_state`, `ops/logs`, `ops/quarantine`, `mlflow.db`/`mlruns`, `optuna*.db`, feature catalog).
- **7** are `PROPOSED-THIS-PASS` and nothing writes them yet.

### 13.3 Required flag categories

| # | Category | Findings |
|---|---|---|
| 1 | Unmapped functionality | **None.** Every package maps to ≥1 responsibility (§13.1) |
| 2 | Unmapped outputs/intermediates | **None unmapped**, but 8 classes bypass the resolver and 2 (`cascade_backtest` results, cost sensitivity) are **not persisted at all** |
| 3 | Duplicate implementations | economic evaluation (≥10, §10); two walk-forward engines; ≥6 config-loading paths; 5 re-implementations of the `features_*.parquet` glob; 4 of `VTECH_DATA_DIR`; DUP-1…DUP-15 (§6) |
| 4 | Duplicate/inconsistent path logic | `data_paths.py:192` fallback disagrees with its own `_LAYOUT` key; MLflow artifacts split across 2 roots (handled by `_prefer`, not resolved) |
| 5 | Hardcoded paths bypassing shared resolvers | H1–H13 (`STORAGE_CONTRACT.md` §2.5); **59 files** reference `/opt/vtech/data` literally |
| 6 | Ambiguous ownership | `population_eligibility` and `cost_model` have no single owner; `src/webui/` is simultaneously interface_surface and part of serving; **`src/research_director/evaluator.py` is classified by directory** — its content (bootstrap 95% CI on the aggregate fold delta, guardrail non-regression, improvement on ≥⌈2/3⌉ of folds, `:19-21`) is a Validator / Model Lifecycle concern sitting under `research_automation`, and is stricter than anything in `src/validation` |
| 7 | Canonical abstractions with no justified implementation | `src/{features,outcomes,models,validation,decision_policy,serving,orchestration}` are **interface-only**. Justified as *contracts* only while they stay interfaces — see §13.4 |
| 8 | Existing superior logic omitted by the canonical layer | **Two found and corrected this pass**: (a) dataset identity — `label_store.py`'s contract addressing is better than the free-form `data_reference` originally proposed; `DatasetRef` now reuses it. (b) objective structural validation — `config/objectives/objective.schema.json` (enforced by `src/objectives/loader.py`) already sets `additionalProperties: false`; the canonical validator's hand-written `KNOWN_OBJECTIVE_KEYS` allowlist duplicated it **and was wrong** (omitted `custom_optimization`, `cost_matrix`, producing false warnings). Removed; the check now reads the real schema. |

### 13.4 Are the interface-only canonical packages still justified?

Reconsidered explicitly, because "an interface with no implementation" is the
textbook symptom of architecture-for-symmetry.

| Package | Verdict |
|---|---|
| `canonical/experiment/` | **JUSTIFIED — real, tested, non-duplicative.** Nothing else in the repo records implementation identity, a scientific fingerprint, or a conservative comparison |
| `canonical/evaluation/` | **JUSTIFIED** — it is the only implementation that persists cost assumptions with results; the ≥10 existing ones do not |
| `canonical/data_contracts/causal.py` | **JUSTIFIED** — real, tested |
| `canonical/lifecycle/status.py` | **JUSTIFIED** — names transitions nothing else names |
| `canonical/calibration/` | **JUSTIFIED** — the responsibility genuinely has no implementation anywhere |
| `canonical/{features,outcomes,models,validation,decision_policy}/` | **CONDITIONALLY JUSTIFIED** — as *documented contracts* for responsibilities with two competing legitimate implementations each. They must not acquire implementations that re-do `src/features/engine.py` or `src/search/`. If they remain empty after the first integration, delete them rather than fill them |
| `canonical/serving/`, `canonical/orchestration/` | **NOT JUSTIFIED — recommend removal.** `src/serving/` + `src/paper/` + `src/execution/` and `src/research_director/` + `src/scoping/` are mature, wired, and have no competing implementation. These two exist only to make the responsibility list symmetric. *(Not removed in this pass: removal is a destructive operation reserved for the user.)* |
| **No path resolution anywhere in `src/`** | correct — `src/common/data_paths.py` is canonical (§3.1) |

## 14. Functional Coverage Summary

1. **Meaningful modules/packages reviewed**: 20 non-canonical `src/` packages (≈84,400 lines), plus `scripts/` entry points inventoried in `docs/SCRIPT_INVENTORY.md`, `config.yaml`, `config/objectives/*.yaml` and the `docs/architecture/*` set.
2. **Fully mapped to a responsibility**: 20 / 20.
3. **Ambiguous ownership**: 2 responsibilities (`population_eligibility`, `cost_model`); 1 package with dual ownership (`src/webui/`).
4. **Duplicate implementations found**: economic evaluation (≥10), walk-forward engines (2), config loaders (≥6), plus DUP-1…DUP-15.
5. **Legacy/deprecated**: Track A (`legacy/checkpoints/`), `legacy/{hpo,research_runs,features_hive}`, and the zero-caller modules in §12.1 item 8 — **report-only; nothing removed.**
6. **Uncovered functionality**: none.
7. **Existing implementations reused as canonical rather than re-implemented**: `data_paths.py` (paths), `label_store.py` (dataset identity), `objective.schema.json` + `objectives/loader.py` (objective structure), `shap_explain.py` (feature importance), `runner.py`'s OOF column contract, `custom_test_dir()` run layout, `code_bundle.py` (code identity), `vm_config.py` (resources).
8. **Canonical implementations actually required**: 5 — `experiment/{manifest,comparison,config_validation,schemas}.py`, `evaluation/economic.py`, `data_contracts/causal.py`, `lifecycle/status.py`, `cli.py`.
9. **Interface-only packages**: 7 justified conditionally, **2 recommended for removal** (§13.4).

## 15. Persistence / Path Coverage Summary

1. **Persistent/intermediate artifact classes identified**: 35.
2. **Fully mapped producer→path→consumer chains**: 35 / 35 (`STORAGE_CONTRACT.md` §2).
3. **Existing canonical path conventions retained**: `_LAYOUT`'s 36 logical keys, hive partitioning, `label_set_dir()` contract addressing, objective-keyed run dirs, `_prefer()`/`_LEGACY_ALIASES` migration tolerance — **all retained unchanged.**
4. **New path conventions actually required**: **zero new roots.** Only new *files inside the already-adopted run directory* (`manifest.json`, `metrics.json`, `fold_metrics.json`, `selected_population.json`, `calibration.json`, `economic_metrics.json`, `cost_sensitivity.json`, `report.md`).
5. **Hardcoded-path exceptions**: H1–H13; 59 files referencing `/opt/vtech/data` literally.
6. **Unresolved paths (no `_LAYOUT` key)**: 8 classes (§13.2).
7. **Proposed-but-never-adopted historical schemes**: `experiments/` (`DATA_LAYOUT.md` §4 "Target shape"), `artifacts/runs/` (invented by pass 2) — **both retracted**; `legacy/features_hive` (key retained, data removed).
8. **Outputs still lacking a canonical destination**: 0. Track C's cascade backtest results and cost-sensitivity output now resolve to `<run>/economic_metrics.json` and `<run>/cost_sensitivity.json` (`docs/EXPERIMENTS.md` §2.1). What they lack is a *producer that writes them*, not a destination.
9. **EIGHTH PASS (2026-09-22) — full-tree filesystem re-inventory, `docs/STORAGE_INVENTORY.md`.** Item 5's H13 fixed (`live_dir()` added); H14/H15 newly found and fixed in the same pass (`session_profile`, `research_director/regimes`) — item 5 is now H1–H15, with H13–H15 closed. Item 7's "both retracted" verdict for `experiments/` now extends to the whole of `DATA_LAYOUT.md` §4: the reorg that actually happened used `sources/`/`research/`/`legacy/`, not `market/`/`experiments/`/`_legacy/`, and `docs/STORAGE_CONTRACT.md` §8 formally adopts the existing top-level names as canonical rather than proposing a rename. Ten new storage-architecture risks (STOR-1…STOR-10, `docs/STORAGE_INVENTORY.md` §4) were found; none are path-resolver gaps of the kind items 4–6 track — they are version-identity (STOR-2/4), operational (STOR-1/3/10), and lineage (STOR-7/8) findings, out of this list's scope but tracked in the risk register.
10. **TENTH PASS (2026-09-22, Storage Implementation Tranche 1) — outcome-dataset path/filename syntax finalized, `build_id` mechanism corrected, STOR-2 fixed.** Item 4's new-path count stays zero for run artifacts, but a genuinely new pattern is now finalized for outcome datasets: `sources/labels/<outcome_family>/<resolution>/<contract_id>/` (`docs/STORAGE_CONTRACT.md` §17.1), plus two new sidecar-filename constants (`_build.json`) and a new part-naming convention (`part{NNN}.parquet`/`part_<date>.parquet`), implemented in `src/common/data_paths.py::outcome_dataset_dir()` and `src/training/label_store.py`. STOR-2 (item 5's "eval_move_model.py/build_target_matrix.py default to `move_matrix_v3`") is FIXED, not merely flagged — see the addendum above. `DatasetRef.semantic_type` gained `"predictions"`, distinct from `"outcomes"`.

**No file under `data/` was created, moved, copied, rebuilt or deleted in producing any of the above.**


---

## 16. Independent adversarial review (fresh context, read-only)

Run after the sections above were written, per the task's requirement for an
independent challenge before finalizing. The reviewer read the canonical docs and
all of `src/`, and spot-checked ~15 `file:line` citations against source.

**Confirmed accurate by independent verification**: `src/webui/` = 11,930 LOC and
~62 route decorators; `logging.basicConfig` in exactly 92 files vs `structlog` in 1;
`cascade_backtest.py:205`'s `sharpe_trade` is a t-statistic mislabelled as a Sharpe;
`native_tf.py:161-162` destructive unlink; `session_profile.py:170-172`
newest-build-wins merge; `label_store.py:79-95` contract write; no parallel path
layer anywhere in `src/` (only `cli.py:75` consumes `data_paths`).

**Defects it found, all since fixed:**

| # | Defect | Fix |
|---|---|---|
| R1 | **`compare_runs` could certify an unsound comparison.** `EVIDENCE_REQUIRED` covered 4 of 10 compared dimensions. Two runs with different decision-policy thresholds that both left `decision_policy={}` produced no leaf diff, so the dimension read as *not varying*; one unrelated difference elsewhere was then certified as an attributable cause. The module's own "silence is not agreement" principle was implemented for 4 dimensions and silently violated for the rest | `EVIDENCE_REQUIRED` now covers all 7 compared dimensions that can legitimately be empty (the other 3 are refused by `__post_init__`). A guard test asserts the two tuples stay in sync. Fixtures were under-specified and were corrected — the module was right, the tests were wrong |
| R2 | **`DatasetRef.location` entered `scientific_fingerprint()`** although its own docstring says "for human traceability only. Never the identity". A dataset moved between runs changed the fingerprint without changing the science, so `write()` would refuse a legitimate idempotent retry — the exact failure it exists to prevent | `NON_SCIENTIFIC_LEAVES` strips `location` from the fingerprint payload. `n_rows` is deliberately kept: a different row count is a different dataset |
| R3 | **`MIGRATION_MANIFEST.md` §10's Git-safety evidence sentence was false** and contradicted its own §2 table | Replaced with a correction pointing at the accurate per-file split in §16 of that document |
| R4 | **Stale test count** ("76 passed") | Corrected to the current verified figure |
| R5 | **Four stale `cli` citations** said the subcommand is `test`; it is `verify` | Fixed in `MIGRATION_MANIFEST.md` and `RESEARCH_WORKFLOW.md` |
| R6 | **`objective_id` durability was overstated** — demonstrated for Track B, asserted for Track C, which has no objective document at all | `docs/EXPERIMENTS.md` §1 now states the limit explicitly. **SUPERSEDED (sixth pass):** an objective document exists. **Corrected:** `mnq-auction-firstpass.yaml` is the ADR-123/126 candidate-exit objective, not the cascade. The cascade document is `nq-Rscaled-mkt.yaml`. Neither script reads its file yet |
| R7 | **`research_director/evaluator.py` classified by directory** | Recorded in §13.3 and `project_manifest.yaml`; reassignment deferred to when that file is next touched |

**Findings accepted but deliberately not acted on:** `extra` remaining unconstrained
(acceptable only because it is excluded from the fingerprint, and recorded as a
residual gap); the `serving`/`interface_surface` plane tension (already an open
critical finding — the plane separation describes the intended boundary, which the
code does not currently respect, and saying so is the point); and the observation
that a package existing only to host a docstring is unusual scaffolding (covered by
§13.4's recommendation to delete rather than fill them if they stay empty).

R1 and R2 were live defects in code this pass had written and tested. Both were in
modules whose tests passed. That is the argument for the independent review step.
