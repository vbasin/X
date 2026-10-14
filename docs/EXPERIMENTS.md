# EXPERIMENTS.md — Experiment Identity, Artifact Contract, Comparison

**Status: CANONICAL, CURRENT** for the `src/experiment` and `src/evaluation` contracts. **Wired for Track B**: `src/search/controller.py` writes `manifest.json` for every Custom Test search run (`src/search/run_manifest.py`, additive/best-effort — `docs/MIGRATION_MANIFEST.md` §20). **Wired for objective runs**: `manifest.json` is the only run-level file. It holds the pass list, the resolved objective snapshot, and the fold design. The cascade objective is `config/objectives/nq-Rscaled-mkt.yaml`. `mnq-auction-firstpass.yaml` is the superseded candidate-exit objective, not the cascade.

Path/design corrections made while producing this document are recorded in `docs/MIGRATION_MANIFEST.md` §3, not here. This file states the contract as it now stands.

## 1. What identifies an experiment

A version-suffixed dataset or script name (`move_matrix_v3`, `_v4`) is **not** an experiment identity — it tells you a build happened, not what was controlled, what changed, or what it should be compared against. The canonical identity is a `RunManifest` (`src/experiment/manifest.py`, `schema_version: 4`. v3 added the corrected `build_id` meaning; v4 adds `passes` and the objective-run fold-design fields. v1–v3 files remain readable).

**Structurally required** (construction fails without them): `run_id`, `objective_id`, `responsibility_versions`. `code` may be `None` when git identity is genuinely unavailable — it is then reported as missing evidence (INCOMPLETE), **never** filled with a placeholder such as `"unknown"`.

| Field | Meaning |
|---|---|
| `run_id` | unique, generated — not hand-chosen, not a version suffix |
| `objective_id` | the durable research/economic objective (e.g. `nq-50pt-econ`). Derived from the *existing* `config.yaml:objective_id` → `config/objectives/<id>.yaml` (JSON-Schema-validated by `src/objectives/loader.py`) → `data/research/runs/<id>/` chain. `research/custom_test/` is gone. `research/search/` is a compatibility root. **The cascade has an explicit objective document** (`config/objectives/nq-Rscaled-mkt.yaml`, identity copied from `cascade_policy.py` / `move_label.py` / `r_before()` and test-asserted). `config/objectives/mnq-auction-firstpass.yaml` is the superseded ADR-123/126 candidate-exit objective (60s grid, session-close barrier, resting entry, `pnl_r` regression) and is not that cascade document. Objective-run writers now put this id on `manifest.json`. |
| `hypothesis` | one human-written sentence stating what this run tests. **Evidence, not a construction gate** — existing runs record none, and refusing to construct their manifest would mean fabricating one or recording nothing |
| `responsibility_versions` | responsibility → implementation identifier actually used (`{"feature_builder": "cascade/es_fs_st@<rev>", ...}`). This is what makes two runs comparable or not |
| `code` | `CodeIdentity(git_sha, dirty, bundle_sha256)`. `dirty` has **no default** — an unqualified SHA is an unverifiable claim. `bundle_sha256` links to `src/common/code_bundle.py`, which already *enforces* what this field only records |
| `datasets` | tuple of `DatasetRef(logical_id, semantic_type, contract_id, build_id, content_sha256, store_id, resolution, location, n_rows)`. `semantic_type` includes `"predictions"` (tenth pass) — a model's OOF/scored output consumed as a downstream input (e.g. `build_target_matrix.py --scores`) is a CLAIM about the future, not ground truth, and must never be recorded as `"outcomes"` or hidden inside another dataset's `build_id`. **At least one of `contract_id`/`build_id`/`content_sha256`/`store_id` is required** — a location alone is not identity, and `DatasetRef` refuses to be constructed without one. `identity_strength` ranks them `content` > `build` > `contract` > `store` — three genuinely different questions (content identity, realization/build identity, semantic identity), not degrees of one; see `docs/STORAGE_CONTRACT.md` §5a. **Only `build`/`content` identity satisfies `RunManifest.missing_evidence()`'s `datasets` check (tenth pass)** — `store_id` is *logical* identity only (several stores are rebuilt in place, e.g. `native_tf.py:161-162`, so it names WHICH dataset, not WHICH version) and `contract_id` alone is *semantic* identity only (states what the dataset MEANS, not WHICH realization was used — the same `contract_id` can silently name different bytes after a `force=True` rebuild); both are truthful, useful evidence, just not strong enough for `CANONICAL_COMPLETE`. `contract_id` reuses `src/training/label_store.py::label_contract_id()`/`outcome_contract_id()` — a hash of the SEMANTIC contract, never dataset content (Track B is **contract-addressed, not content-addressed**; `force=True` can replace parts beneath an unchanged `contract_id`). `build_id` must come from genuine code/config/upstream-version evidence (`label_store.write_build_sidecar()`, `src.common.code_identity`) — **never from a fingerprint of the dataset's own output filenames/byte sizes**, a mechanism the ninth pass used and the tenth pass retired as insufficiently genuine build evidence. `content_sha256` is `None` everywhere in this codebase today, since no genuine content digest is computed anywhere; never promote a weaker hash into it. Where no mechanism exists at all, omit the `DatasetRef`: an absent reference is reported as missing evidence; a location-only one would claim an identity it does not have |
| `config_snapshot` | the configuration actually used (post-defaults, post-validation). **A non-empty mapping is not itself proof of resolution** — `config_snapshot_resolved: bool` (default `False`) is the caller's explicit, unfabricated claim that this genuinely IS the fully resolved config; `missing_evidence()` requires both. Track B's search adapter always leaves this `False`: a Custom Test run sweeps multiple candidates with their own resolved hyperparameters, which no single run-level mapping can truthfully represent |

**Conditional** (`None`/empty is a recorded *absence of evidence*, never an assertion of agreement — see §3): `validation` (`ValidationDesign`: `split`, `n_folds`, `purge_bars`, `embargo_bars`, `weighting`, and for objective runs `test_months`, `min_train_months`, `purge_sessions`; realized `fold_windows` and `hold_out_dates` are recorded but stripped from the fingerprint), `experiment_family`, `parent_run_id`, `baseline_run_id`, `seeds`, `execution_assumptions`, `decision_policy`, `calibration`, `artifacts`, `passes`, `extra`.

`passes` is the run's pass list: role, input/join spec, and model pointers. It accumulates as each pass is trained, so it is not part of the scientific fingerprint. Each model's feature columns and hyperparameters live in that model's `model.json`, not in the pass list.

**Evidence level.** `missing_evidence()` returns the `REQUIRED_FOR_CANONICAL_COMPLETE` provenance a run did not record; `evidence_level` is `CANONICAL_COMPLETE` when it is empty and `INCOMPLETE` otherwise. Both are persisted into `manifest.json` and **recomputed on read**, so a hand-edited file cannot assert a completeness its contents do not support. An INCOMPLETE manifest is a valid record — it is simply not admissible on its own as evidence for a controlled comparison.

**Identity classification.** These fields are deliberately *not* part of scientific identity: `hypothesis`, `experiment_family`, and `baseline_run_id` are **comparison context**; `created_at`, `artifacts`, `extra`, and `passes` are **bookkeeping**. `passes` grows as models are registered. `parent_run_id` *is* identity — derivation is real lineage, since a meta layer trained on a different base run is a different experiment.

**Scientific immutability.** `scientific_fingerprint()` is a sha256 over the 12 fields in `RunManifest.SCIENTIFIC_FIELDS`, with `DatasetRef.location` stripped (`NON_SCIENTIFIC_LEAVES`) and with `validation.fold_windows` / `hold_out_dates` stripped (those are the design applied to one population). `write()` permits an idempotent rewrite, and it permits a previously empty scientific field to be filled once (the builder opens the manifest before the fold design exists; eval then records it). It raises `FileExistsError` when the same directory is rewritten with a different `run_id` or with a different value in a field that was already set. `read()` refuses a `schema_version` newer than the running code.

Topology-neutrality is a hard requirement: nothing in the manifest names a stage, gate, cascade or meta layer.

## 1.1 What an `objective_id` means

`objective_id` = the stable, topology-independent identifier for the trading/research problem a run serves. It is the repository's existing term (`config/objectives/<objective_id>.yaml`); the canonical layer briefly called it `research_objective_id` — that alias is gone. Manifests written with it (schema v1) are still read and normalised; a file carrying both keys with different values is refused.

The objective YAML holds both what the id **means** and experiment **defaults**. Which is which is declared once, in `config/objectives/objective.schema.json`, by `x-objective-role` on every property (read by `src/experiment/objective.py`):

| Role | Fields | A change means |
|---|---|---|
| `identity` | `objective_id`, `traded products` (underlying — but not `traded products[].note`, see below), `directions`, `economic_intent`, `horizon`, `session.trading_start/trading_end/timezone/flatten` (eligible population), `labels` outcome definition (`scheme`, `move_threshold`, `mae_tolerance` or `stop_rule`, `unit`, `unit_definition`, `target_observation`, …) | normally a **new** `objective_id` |
| `default` | `session.training_*`, `session.decision_grid_seconds` (sampling cadence, not the problem), `labels.encoding/fire/aux`, `kpi` including `kpi.primary` (a **ranking metric** for the champion gate — not the durable economic goal, which is `economic_intent`), `eval`, `search`, `eligible_model_families`, `ensemble`, `custom_optimization`, `cost_matrix`, `costs`, `risk_return`, `data_availability` | same objective; the manifest's `config_snapshot` records the resolved value each run used |
| `meta` | `schema_version`, `description`, `strategy_keywords`, `traded products[].note` | nothing |

Role resolution is **recursive**, not "one tag covers the whole subtree": a `note` nested inside an `identity`-tagged `traded products` entry is still excluded, because the nested tag overrides the container's. Untagged children of an `identity` container *inherit* identity (e.g. `traded products[].symbol`); untagged children of a `default`/`mixed` container default to `default` (the pre-existing `session`/`labels`/`kpi` convention). `objective_identity_sha256(obj)` hashes only the resulting identity-bearing subset, so a changed default or a changed `note` leaves it stable and a changed outcome definition does not.

Schema generalisations made for Track C (minimal; all 9 pre-existing objectives still validate): `labels.scheme += first_passage_r`, `labels.target_observation += print_tape`, `labels.stop_rule` as an alternative to `mae_tolerance` (a target-adaptive stop has no single true adverse bound), `labels.unit_definition` generalised to accept a structured, machine-checkable object (see below) as well as free text, `session.flatten`, `session.decision_grid_seconds`, top-level `directions` and `economic_intent` (topology-neutral durable goal, separated from `kpi.primary`'s ranking role), `costs.execution_cost_profile` (a reference into `config/execution_costs.yaml`), and `kpi` now requires only `primary` (acceptance bars are defaults, and the cascade has none to state truthfully). The candidate-exit objective adds `labels.scheme += candidate_exit_surface`, `labels.target_observation += path_1s_hilo`, `labels.unit_definition.per_session_metric += median_1s_block_range`, and identity objects `outcome`, `candidates`, `entry_rule`, and `selection` (account size and point value inside `selection` stay defaults). `kpi.primary` may be `t_stat`. A `candidate_exit_surface` document does not require `move_threshold` or a stop.

**The cascade objective** is `config/objectives/nq-Rscaled-mkt.yaml`: NQ; long and short; market entry; 08:30–14:45 CT entries, 15:00 flatten, 5 s decisions (a *default*, not identity — see the role table above); any target ≥ 0.25 R reached before its own stop `min(0.5 × target, 0.25 R)` within 30 min, observed on the print tape; `economic_intent: positive_selective_net_expectancy_after_costs`. **R is the median of the 5 most recently COMPLETED prior sessions' own `session_r_from_tape`** (`src/features/session_profile.py::r_before`) — the current session contributes nothing (`labels.unit_definition`: `source: prior_session_median`, `lookback_sessions: 5`, `excludes_current_session: true`, `per_session_metric: median_print_block_range`, `per_session_block_seconds: 300`). A test asserts those fields against `cascade_policy.py` / `move_label.py`. **Not wired** — no cascade script reads the file yet. NQ is the underlying; default economic sizing is `MNQ_equivalent`, referenced via `costs.execution_cost_profile`.

**The superseded first-passage objective** is `config/objectives/mnq-auction-firstpass.yaml` (ADR-123 amended by ADR-126). It is not the cascade. Decisions are every 60s. Entry is a resting order at the far edge of the containing bound (`entry_offset_pts` 1.0 between bounds), cancelled after 900s if unfilled, and an unfilled order emits no row. The barrier race starts at the fill and ends at the 15:00 flatten, observed on the 1s high/low with `tie_policy: stop_first`; a censored path exits at the flatten close. Targets and stops are the cartesian product of KDE-bound near edges plus an R ladder (`0.25, 0.5, 0.75, 1.0, 1.5, 2.0`), not one 0.25 R target with stop `min(0.5 × target, 0.25 R)`. The fitted target is `pnl_r` (regression). The book takes the argmax predicted `pnl_r` per decision bar, one position busy for `t_fill + t_event`, and keeps the train-quantile point with the best t-stat and at least 150 trades. The printed book is 1 MNQ and applies no costs. R is the median of the prior 5 sessions' `session_r` (median 600s high/low range on the 1s grid, window through 15:00). The label builder's `--candidates` default is still `profile` (market fill); this document describes `--candidates bounds`. No script reads the file. Historical `sources/labels/firstpass*` directories are not migrated.

## 1.2 Run identity

| Term | Meaning | Example |
|---|---|---|
| `objective_id` | the trading problem | `nq-50pt-econ` |
| `run_ts` | existing timestamp; the physical directory **leaf** only | `20260922T145336Z` |
| `run_id` | globally unique **logical** id of one execution; Track B: `<objective_id>/<run_ts>` | `nq-50pt-econ/20260922T145336Z` |

Physical location: `research/<workflow>/<objective_id>/<run_ts>/`. Identity and location are related, not interchangeable. There is no third term.

## 2. Artifact contract

**Canonical run root** is `research/runs/`, resolved through `src.common.run_layout.runs_dir()` / `new_run_dir()`. The directory **leaf is `run_ts`** (physical location), not `run_id` (logical identity — §1.2); `run_id = <objective_id>/<run_ts>`:

```
$VTECH_DATA_DIR/research/runs/<objective_id>/<run_ts>/
```

`research/search/` remains only so existing bundles and OOFs still resolve. Nothing new is written there. `research/custom_test/` has been removed.

Do not create a new top-level data root for experiment artifacts. If a genuinely new grouping is ever needed, add a `_LAYOUT` key to `data_paths.py` so it is resolvable — **a path that only exists in a document is not a path.**

### 2.1 Canonical output contract

The single source of truth is **`src/experiment/artifacts.py`** (`ARTIFACTS`). Each class has one canonical filename, a minimum-content rule, and an applicability rule computed from the run's own manifest — so a class whose responsibility did not run is *not applicable*, never "missing".

| Class | Canonical file | Applies when (from the manifest) | Minimum content |
|---|---|---|---|
| manifest | `manifest.json` | always | identity, code, datasets, config_snapshot, validation, seeds, execution_assumptions, responsibility_versions, artifacts, evidence_level, missing_evidence |
| config_snapshot | embedded: `manifest.json#config_snapshot` | always | configuration recorded for the run — never a pointer to the YAML; no second file. Class name matches the manifest field; whether it is fully *resolved* is the separate `config_snapshot_resolved` claim (§1), not implied by the class name |
| metrics | `metrics.json` | always | `n_samples` (the population), `metrics` |
| fold_metrics | `fold_metrics.json` | `validation` recorded | `folds` of `FoldMetrics` |
| oof_predictions | `oof_predictions.parquet` | `validation` recorded | Track B backtest frames: `ts, direction, confidence, direction_true`. Objective-run frames are compact: keys, fold, outcomes, predictions, train-derived thresholds. Feature columns are not stored |
| model | `model.json` | the manifest `passes` list names a model | `model_id`, `model_type`, `hyperparameters`, `target`, plus `feature_columns` and `inputs` (the as-of join spec). One file per `pass_<n>/model_<id>/` |
| hpo_study | `hpo.json` | `config_snapshot.study.kind` is `hpo` | sampler, seed, search space, metric, `n_trials`, `pass_n`, `selected_model_id`. One Optuna study. `validation` records the folds that study scored |
| trials_summary | `trials_summary.parquet` | same | one row per trial: `trial_number`, `model_id`, `status`, `score`, `params` |
| selected_population | `selected_population.json` | `decision_policy` recorded | `n_raw, n_selected, selection_rule` |
| calibration | `calibration.json` | `calibration.method` not `none` | `CalibrationResult` fields |
| trade_ledger | `trade_ledger.parquet` | `economic_evaluation` in `responsibility_versions` | order types, execution prices, quantity, timestamps, gross/commission/net |
| economic_metrics | `economic_metrics.json` | same | `assumptions` (incl. `economic_contract_unit`, `underlying`), totals |
| cost_sensitivity | `cost_sensitivity.json` | same | scenario → ledger summary |
| feature_importance | realised by `SHAP/` | `feature_importance` in `responsibility_versions` | reused from `shap_explain.py` |
| effective_sample_size | `effective_sample_size.json` | never yet (DEFERRED) | `n_trades, n_eff, method` |
| report | `report.md` | recommended, not required | — |

Feature and context sources in `inputs` declare `availability` (`timestamp`, `bar_close`, or `session_close`). The training join and `build_oof_analysis` attach a row only when that availability is at or before the decision `ts`. See `docs/DATA_CONTRACTS.md` §3.

Economics is keyed on a declared `economic_evaluation` responsibility, **not** on `execution_assumptions`: a Track B custom_test search records its objective's cost assumptions as configuration while performing no economic evaluation, and must not be told it is missing a trade ledger.

**Filename criteria** (test-enforced): semantic content only; valid for any topology; no stage, model family, version suffix or temporary label; one name per class; listed in the manifest's `artifacts` map when produced.

**Existing files are classified, not renamed** (`COMPATIBILITY_ARTIFACTS`): `run.json`, `contract.json`, `folds.json` (older objective runs and Track B), `search_result.json`, `spec.manual.yaml`, `topology.yaml` → compatibility, still consumed. New objective runs do not write `run.json`, `contract.json`, or `folds.json`. `training_matrix.parquet` is an optional population cache, not a required class. `analysis_view.parquet` is a temporary inspection file. `trial.json` sits inside an HPO trial's model folder and is classified by name. An Optuna study is `research/runs/<objective_id>/<hpo_run_ts>/` with `hpo.json` beside `manifest.json`; `research/search/` is a separate workflow. `label_contract.json`, `SHAP/` → canonical; `oof_<model>.parquet` → compatibility that **realises** `oof_predictions`; `trades.parquet`, `backtest_report.json` → superseded migration candidates realising `trade_ledger` / `economic_metrics`; `diagnostics_*.json`, `progress*.json`, `models/` → implementation-specific supplemental, realising no canonical class. The manifest's `artifacts` is a map `canonical class → existing files`, so a large existing artifact is referenced, never duplicated for naming symmetry — **and `RunManifest.__post_init__` enforces this**: a key that is not a real canonical class (e.g. a compatibility status like `"compatibility_consumed"` or `"implementation_supplemental"`) is refused outright. A file that realises no canonical class is simply absent from `artifacts`; its classification lives in `classify_existing()` / `COMPATIBILITY_ARTIFACTS`, not smuggled into the manifest under its status name.

**Result evidence.** `compare_results` reads and content-validates every existing comparable artifact on each side *before* comparing presence. Unparseable, or parseable but missing its minimum content → `invalid_evidence`; the comparison is then never `clean`.

One contract, used by whichever Model Strategy/Decision Policy combination the run actually used. Do not invent a second, incompatible artifact set for a different track.

## 3. Comparing two runs

`run-compare` reports **SETUP** and **RESULTS** as two separate sections. A result difference between runs whose setup comparison is uncontrolled is an *observation*, not a finding, and the output says so.

### 3.1 Setup comparison

Implemented in `src/experiment/comparison.py` (`compare_runs`), exercised by 12 adversarial tests in `tests/architecture/test_comparison.py`, and surfaced by `python -m src.experiment.cli run-compare <a> <b>`.

**Compared dimensions** (`COMPARED_DIMENSIONS`): `objective_id`, `responsibility_versions`, `config_snapshot`, `datasets`, `validation`, `seeds`, `code`, `execution_assumptions`, `decision_policy`, `calibration`. Diffs are computed **recursively to the leaf** (`config_snapshot` → `labels.move_threshold`, `datasets` → `[0].contract_id`), not top-level-only.

**Conservatism rules — the module's job is to refuse to certify an unsound comparison:**

1. `controlled_comparison` is `True` only when **exactly one** dimension varies **and** no required evidence is missing.
2. `EVIDENCE_REQUIRED` covers **all 8 compared dimensions that can legitimately be empty** — `datasets`, `validation`, `seeds`, `execution_assumptions`, `config_snapshot`, `decision_policy`, `calibration`, `code`. (`code` moved into this set once the "unknown" SHA placeholder was removed — an unavailable git identity is now `None`, not fabricated, so it must be evidence-required like the rest. Only `objective_id` and `responsibility_versions` are refused outright by `RunManifest.__post_init__` and so can never be silently empty.) If **both** runs leave one unrecorded it is reported in `missing_evidence`; **an unrecorded dimension is unknown, not identical.** A run that genuinely performed no calibration records `calibration={"method": "none"}` — that is evidence; `{}` is silence.
3. `attributable_dimension` is `None` unless rule 1 holds. A single differing `responsibility_versions` entry does **not** license a causal claim on its own.
4. Two manifests that differ in nothing still produce `controlled_comparison: False` with an "unexplained difference" warning — identical inputs should not have produced two runs.
5. Population overlap remains the one comparison input not yet computable from manifests alone; it requires `selected_population.json` from both sides (§2.1). This is the exact check missing from Track C's v3→v4 comparison.

## 4. Reproducing a historical run

No historical run has a `manifest.json`, so reproduction is still evidence-reconstruction:

- **Track B**: a run under `data/research/runs/<objective>/<run_ts>/` still carries `spec.manual.yaml`, `topology.yaml`, and `folds.json` beside `manifest.json`. Those files are the search record. `manifest.json` is the run-level identity. `research/custom_test/` is not a current root.
- **Track C**: re-run the `scripts/features/build_move_matrix.py` / `eval_move_model.py` / `build_target_matrix.py` / `eval_target_model.py` invocation recorded in that session's `/tmp/*.log` or `docs/changelog/CHANGELOG.md` entry. No structured record exists.

Consequence: **no run in this repository is currently reproducible to the standard §1 defines.** That is the gap `RunManifest` exists to close, and the reason integration — not further design — is the next step.

### 3.2 Result comparison

`compare_results(dir_a, dir_b, manifest_a, manifest_b)` diffs the persisted result artifacts to the leaf: `metrics.json`, `fold_metrics.json`, `selected_population.json`, `economic_metrics.json`, `cost_sensitivity.json`.

Each artifact is expected only under the manifest condition that implies the step ran (`RESULT_ARTIFACTS`): fold metrics only when `validation` is recorded, a selected population only when `decision_policy` is, economics only when `execution_assumptions` are. **A run that performed no economic evaluation is not "missing" `economic_metrics.json`** — requiring artifacts for steps that did not occur would manufacture findings. Artifacts neither run was expected to produce are listed under `not_applicable` so their absence is visibly a non-finding. An artifact a run's *own manifest* implies it should have produced, but did not, is reported in `missing_expected`.

## ADR-139 experimental policy artifacts

NQ continuation research persists canonical `RunManifest` objects and resolved
protocol snapshots under the existing `research/runs/` tree. The policy learns
no supervised first-passage target, and the current objective YAML schema cannot
represent its realized-policy-P&L outcome. These are explicitly experimental
runner IDs, not a claim that `load_objective()` or production serving accepts
those policies. A truthful schema extension is required before such integration.

`source_hashes.json`, `code/`, upstream source inventories, `discovery_trials.json`,
`frozen_policy.json`, validation ledgers and gate, and separate quote-phase source
locks establish provenance. Later quote-phase artifacts do not rewrite the
historical manifest. Report exports include all discovery trials, selected trade
CSVs, costs, uncertainty and separate evidence cohorts. Data preparation and
failed/aborted earlier attempts remain retained; see
[retry report](research/NQ_RETRY_RESULTS.md) and ADR-139.
