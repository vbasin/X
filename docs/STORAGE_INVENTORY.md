# STORAGE_INVENTORY.md — Verified Filesystem Inventory & Migration Matrix

**Status: 2026-09-22 snapshot, not the live tree.** Measured read-only on that date. Since then, fitted `research/custom_test/` runs were migrated to `research/runs/` and that directory was removed. `research/search/` remains a compatibility root. New runs write `research/runs/`. Sizes and "0 found" counts below are the 2026-09-22 measurement. This file is the evidence
base for `docs/STORAGE_CONTRACT.md` §7–§16 (the eighth restructuring pass, "repository-wide
`/opt/vtech/data` storage and artifact architecture pass"). It lives separately because the
full producer→consumer inventory and current→canonical migration matrix span roughly 60 rows
across 7 major subtrees — embedding them in `STORAGE_CONTRACT.md` would bury the canonical
rules and rationale under raw tables. `STORAGE_CONTRACT.md` states the target contract and
cites this file for the measured evidence and the full mapping.

Every number below was measured with `du`/`find`/`ls`/bounded parquet-schema reads against
the live filesystem on 2026-09-22. **No file under `data/` was created, moved, copied,
rebuilt, renamed or deleted to produce this document, and no database was queried.**
`docs/architecture/DATA_LAYOUT.md` §2/§4 is a 2026-09-03 snapshot and target-shape proposal;
where it disagrees with the numbers or structure below, this document is the current one —
see `STORAGE_CONTRACT.md` §0 for the disposition of that proposal.

---

## 1. Real filesystem inventory (2026-09-22)

`/opt/vtech/data` total: **111G**.

| Root | Size | Role |
|---|---:|---|
| `sources/` | 63G | Market data + derived features + outcome/label datasets |
| `research/` | 44G | Track B/C run outputs (search, custom_test, episodes, scoping, research_director) |
| `legacy/` | 538M | Pre-reorg archive (checkpoints, model_staging, research_runs, training_runs) |
| `ops/` | 73M | Platform/operational state |
| `models/` | 23M | Serving lifecycle (staging, registry, leaders) |
| `evaluation/` | 3.6M | Frozen eval contracts + ad hoc diagnostics |
| `mlflow_artifacts/` (top-level) | 2.5M | Pre-reorg MLflow artifact store (still receiving writes — §4 STOR-3) |
| `live/` | 84K | Live-signal file bridge (unmanaged before this pass — now `live_dir()`) |
| `exports/` | 36K | Manually regenerated CSV exports |
| `postgresql/`, `swapfile` | 12K / 4G | Infrastructure, not research data — out of scope |
| `mbp10_job.json`, `mlflow.db`, `optuna.db`, `optuna_ablation.db` | ~1.5M | Loose top-level files (§4 STOR-9) |

### 1.1 `sources/` breakdown (63G)

| Path | Size |
|---|---:|
| `sources/raw/live` | 28G |
| `sources/raw/backfill` | 27G |
| `sources/features/*` | 5.5G (`parquet_options_surface` 2.5G, `parquet_path1s` 1.1G, `parquet_tape` 901M, `timeframe=5s` 674M, `parquet_raw1s` 128M, `timeframe=1m` 113M, `timeframe=15s` 89M, `session_profile` 4.4M, `timeframe=5m` 20M, `timeframe=1h` 2.2M, `timeframe=1d` 656K, `parquet_options_greeks` 4K) |
| `sources/labels/*` | 3.5G across 22 entries — see §1.3 |

### 1.2 `research/` breakdown (44G)

| Path | Size |
|---|---:|
| `research/search/_fold_cache` | **39G** — reproducible cache, no eviction (§4 STOR-1) |
| `research/search/<objective>/` (run dirs, excl. `_fold_cache`) | ~1.8G across 8 objectives |
| `research/custom_test/<objective>/` | 3.0G across 5 objectives |
| `research/episodes/<objective>/` | 568M |
| `research/research_director/` | 21M |
| `research/scoping/` | 1.7M |

### 1.3 `sources/labels/` breakdown (3.5G, 22 entries — the messiest area in the tree)

| Path | Size | Status |
|---|---:|---|
| `move_matrix_v4` | 560M | current Stage-1 output (producer's own default) |
| `move_matrix_v3` | 544M | **live by accident** — both consumer scripts still default here |
| `move_matrix_v2` | 481M | superseded |
| `move_matrix` | 420M | superseded (oldest generation, not a symlink) |
| `1s` | 467M | dead — orphaned ADR-107 migration output, zero live readers |
| `firstpass` | 155M | whole track superseded by ADR-127 (roadmap-confirmed) |
| `firstpass_clvdelta_superseded` | 155M | dead, zero code references |
| `firstpass_bounds_v2` | 112M | superseded, zero code references |
| `firstpass_bounds` | 85M | superseded, zero code references |
| `firstpass_bounds_matrix` | 48M | superseded, zero code references |
| `5s` | 46M | **live** — Track B contract-addressed store |
| `firstpass_matrix` | 35M | superseded (script still runs, track is dead) |
| `firstpass_bounds_v2_matrix` | 34M | superseded, zero code references |
| `target_matrix_v3` | 33M | live (surfaced in WebUI) |
| `target_matrix_v4` | 31M | live, most feature-complete |
| `target_matrix_context_rest` | 26M | live variant |
| `target_matrix_context` | 26M | live variant |
| `15s` | 18M | **live** — Track B contract-addressed store |
| `target_matrix_adaptive` | 11M | superseded variant |
| `target_matrix` | 11M | **live by accident** — `eval_target_model.py`'s unsuffixed default |

---

## 2. Producer → artifact → path → consumer inventory

Condensed from seven bounded, read-only inventory passes (filesystem + source grep + schema
sampling) against the live tree and current code. `STORAGE_CONTRACT.md` §2 already covers the
Track B/run-artifact side in detail and is not repeated here; this section adds what that
document did not yet cover, verified fresh.

### 2.1 `sources/raw/` (54G)

| Class | Producer | Consumer | Path pattern | Identity | Resolver |
|---|---|---|---|---|---|
| Live wire-format archive (Databento Live capture, replay/DR safety net — DB write is inline, not from this archive) | `src/acquisition/live_streamer.py:380` | `src/webui/live.py:71,1089` (monitoring only) | `sources/raw/live/<session-date>/stream_<HHMMSS>.dbn[.zst]`; daily zstd compression via `scripts/ops/compress_raw_live.sh` (systemd timer) | session date + connect time; **no content/version identity** | producer `raw_dir()`-backed; consumer hardcodes `data_root()/"sources"/"raw"/"live"` |
| Backfill historical/batch downloads (futures/options/cross-asset tbbo/bbo_1s/ohlcv_1s/mbp10) | 9 scripts under `scripts/data/*.py` | `scripts/data/ingest_dbn.py`, `src/ingestion/loader.py` → TimescaleDB | `sources/raw/backfill/<content_class>/GLBX-<date>-<hash>/*.dbn.zst` | Databento `job_id` (opaque, submission-derived) — **not content version** | **mostly hardcoded** — only 3 of ~12 backfill scripts use `raw_dir()` |
| Reference/contract metadata (definitions) | `scripts/data/download_ingest_definitions.py:68` | contract-roll/options-surface lookups | `sources/raw/backfill/definitions/` (flat) **and** `dataset=GLBX.MDP3/schema=definition/...` (hive) — duplicate content, two coexisting layouts | instrument root only | hardcoded (`DEF_OUT_DIR`) |
| Cross-asset reference + ad hoc gap-fills | assorted one-off scripts | gap-detection follow-up | `<symbol>_ohlcv_1s/`, `gaps_<range>/`, `live_gap_<date>/` | none, single-use | hardcoded per-script |

**Retention is not uniform**: `databento_backfill.py`'s docstring claims raw `.dbn.zst` is "ALWAYS retained," but `backfill_options_daily.py` deletes each day's raw file after ingest — confirmed by `nq_opt_daily/`'s near-empty contents. Do not assume any `backfill/` subtree still holds its original files.

### 2.2 `sources/features/` (5.5G)

| Class | Producer | Consumer | Identity/mutability | Resolver |
|---|---|---|---|---|
| `timeframe={5s,15s,1m,5m,1h,1d}` hive | `engine.py`/`hive_5s.py` (5s, live), `build_native_tf_{15s,1m,5m,htf}.py` → `native_tf.py::write_month_parts` | `stores.py` (~47 call sites), trainers, `fold_cache.py`, WebUI | monthly `features_{tf}_part*.parquet`; **destructive rebuild** (`native_tf.py:161-162` unlinks all parts first) | fully resolver-backed |
| `parquet_5s` ("leftover day files") | none — **confirmed absent from disk** | 3 legacy tools still target it (`patch_parquets.py`, `migrate_parquet_5s_hive.py`, `patch_iv_fast.py`) | dead | resolver key retained, points nowhere |
| `parquet_options_surface` (2.5G, largest subdir) | `live_streamer.py`, `options_greeks_pipeline.py::write_surface_parquet` | **none confirmed** — not in `stores.py`'s `STORE_IDS`; the in-hive `iv_`/`gx_` features come from a separate DB-computed path | write-only archive candidate | resolver-backed |
| `parquet_options_greeks` (4K, empty) | `options_greeks_pipeline.py` intends to, but a name-shadowing bug (`write_greeks_parquet` bool param shadows the module function at the call site, `:567,631`) makes the write always throw; the exception is swallowed per-day by the backfill script | none (empty) | broken producer, not a storage-architecture gap — flagged for a pipeline-code pass, not fixed here | resolver-backed, write path never completes |
| `parquet_path1s` (1.1G) | `scripts/data/build_path1s.py` | 8 consumers, largest fan-in of non-hive classes | per-session, immutable-in-practice | resolver-backed |
| `parquet_tape` (901M) | `scripts/data/build_tape.py` | `build_target_matrix.py`, `build_move_matrix.py`, `build_session_profiles.py` | per-session | resolver-backed |
| `parquet_raw1s` (128M) | `scripts/features/build_raw1s_dataset.py` hardcodes its output as the literal string `"data/sources/features/parquet_raw1s"` | `finalize.py --parquet-dir` manual override | single-experiment dataset | `parquet_raw1s_dir()` exists, **zero callers** |
| `session_profile` (4.4M) | `session_profile.py::session_profile_dir()` — **fixed this pass**, now resolver-backed (`_LAYOUT["session_profile"]`) | `webui/live_chart.py` | feeds the cascade's structural SSOT (`session_levels.parquet`, still newest-build-wins with no run id — pre-existing, unfixed gap, `FUNCTIONAL_COVERAGE.md` §3.5) | now resolver-backed |
| `parquet_labels_15s`/`parquet_labels_5s` | none — directories do not exist under `sources/features/` | n/a | stale `_LAYOUT` keys; real label data lives under `sources/labels/{15s,5s}/` (contract-addressed) | keys retained, COMPAT classification unchanged from the sixth pass |

### 2.3 `sources/labels/` (3.5G) — see §1.3 for the full per-directory breakdown

Two co-resident, incompatible addressing regimes:

- **Track B — contract-addressed, not content-addressed** (`src/training/label_store.py`; corrected this pass, §4 STOR-11): `labels_dir()/{5s,15s}/<sha256[:16]>/` + `_contract.json`. `label_contract_id()` hashes `normalize_label_contract(contract)` — the normalized SEMANTIC contract (scheme, move_threshold, mae_tolerance, horizon_s, target_observation, encoding, aux_labels, time_to_tgt) — never the label parquet's bytes. Idempotent by default (`write_label_parts` returns existing parts unless `force=True`), but **not content-immutable**: `force=True` replaces every part beneath an unchanged `contract_id` directory. Still **BEST IN REPO** for what it actually is — a strong, self-describing semantic contract — confirmed again this pass, no change needed to the mechanism itself; only its description was wrong.
- **Track C — directory-suffix-addressed** (`build_move_matrix.py`, `build_target_matrix.py`): no sidecar/manifest anywhere in any move_matrix*/target_matrix* directory. The suffix is 100% of version identity, and it is applied **inconsistently**:
  - `move_matrix`'s version is bumped by **editing the hardcoded output path in source and rerunning** — not a CLI flag. `eval_move_model.py` and `build_target_matrix.py --move-dir` both still default to `move_matrix_v3`, one generation behind the producer's current `move_matrix_v4` default. **A bare invocation of either consumer today silently uses stale input** — this is a live, present-tense correctness risk, not just historical debt.
  - `target_matrix` mixes two naming semantics in one flat namespace: `_v3`/`_v4` are Stage-1-input version bumps; `_adaptive`/`_context`/`_context_rest` are feature/policy variants at a fixed version. `eval_target_model.py` defaults to the unsuffixed `target_matrix`, also stale relative to the newest variants.
  - The `firstpass` family (7 directories, 624M) is confirmed superseded in its entirety by ADR-127 per the project's own `docs/roadmap/ROADMAP.md:83`; 4 of 7 have zero code references left at all.

### 2.4 `research/` (44G)

| Class | Producer | Path convention observed | manifest.json present? |
|---|---|---|---|
| Search runs | `src/search/controller.py::run()` | `research/search/<objective_id>/<run_ts>/`, but also `iterate_*`/`strategy_<name>_*`/`refine_*`-prefixed leaves | **0 found anywhere on disk**, despite `controller.py` (currently uncommitted, `git status: M`) writing one via `run_manifest.py` |
| `_fold_cache/` (39G) | `src/search/fold_cache.py::build_or_load_fold_arrays` | `research/search/_fold_cache/<sha256-of-inputs>/{X.npy,y.npy,ts.npy,meta.json}` | n/a — pure cache, not a run |
| Custom Test runs | same `controller.py::run()` | `research/custom_test/<objective_id>/<run_ts>/`; one manually-archived `OLD/<run_ts>/` (operator-created, not code-produced) | **0 found** |
| Episodes | `src/evaluation/episodes.py::build_episodes` | `research/episodes/<objective_id>/<run_ts>__<source_id>.*` — flat filenames, no `<run_ts>/` subdirectory (drift from the two-level convention) | n/a (analytical dataset, not a run root) |
| Research Director | `src/research_director/orchestrator.py` | `research/research_director/<objective_id>/<run_ts>/`, **microsecond**-precision `run_ts` (other producers use second precision) | n/a (orchestration, out of RunManifest scope) |
| `research_director/regimes/` | `scripts/search/regime_analysis.py` | previously a fully hardcoded absolute path bypassing `VTECH_DATA_DIR` entirely — **fixed this pass**, now `research_director_dir()/"regimes"` | n/a |
| Scoping | `src/scoping/runner.py` (second-precision) **and** `src/scoping/orchestrator.py` (microsecond-precision) — two producers, same nominal convention, different `run_ts` grain | `research/scoping/<objective_id>/<run_ts>/` | n/a |

`research/search/_fold_cache/` was measured at 14G in `docs/operations/DISK_RECLAIM_2026_08.md` (2026-08); it is 39G today (2026-09-22) — **~2.8x growth in six weeks**, confirming the LRU eviction that document deferred ("Stage 3, DEFERRED") was never implemented. It is a pure, content-keyed, fully reproducible cache with zero evidence-loss risk, but it now accounts for ~89% of `research/` and sits nested inside the directory `docs/operations/DISK_RECLAIM_2026_08.md` separately designates "never delete — OCI-synced search audit trail." An underscore-prefixed cache inside a never-touch root risks being swept into that policy by a future operator despite being fully discardable.

### 2.5 `evaluation/` + `models/`

| Class | Finding |
|---|---|
| `evaluation/<objective>/contract_v*.json` | Correct, resolver-backed. **But** `nq-50pt`, `nq-50pt-econ`, `nq-50pt-tbm` also physically hold `latest_leaders.json`/`leaderboard_v1.jsonl` — duplicating what `leaders_dir()`'s own docstring says lives exclusively under `models/leaders/<objective>/`. Real drift, not just stale docs. |
| `evaluation/context_analysis/`, `evaluation/move_analysis/` | Ad hoc SHAP diagnostic dumps from manually-run scripts, hardcoded path construction (bypasses `evaluation_dir()`), sitting as siblings of real per-objective directories — an agent or operator could mistake either for an `objective_id`. |
| `evaluation/portfolio_sim/adr047_stop_first/` | Hand-labeled run directory, no `objective_id` in the path, weak backlink to its source OOF (a string embedded in `sim_kpis.json`, not a structured reference). |
| `models/model_staging/`, `models/model_registry/` | Resolver-backed. `finalize_manifest.json` is the de-facto run manifest — **not** the canonical `RunManifest`. Lineage traces `model_staging → search_run_dir`, but that search run directory has no `manifest.json` either (§2.4) — the lineage chain breaks at the same point industry-wide across this pass's findings. |
| `models/leaders/` | Leaderboard rows carry **no foreign key** back to a `model_staging`/`model_registry` version at all — the weakest link in the lineage chain (`nq-50pt-xgb-tcn` has a full leaderboard and zero staging/registry presence). |
| `models/training_runs/` | **Confirmed dead**: directory does not exist on disk, `training_runs_dir()` has zero callers besides its own definition. |

### 2.6 `ops/` + misc top-level

| Class | Finding |
|---|---|
| `ops/quarantine/` (35M) | Durable operator-created holding pen for corrupt hive parts (one entry, documented via a `README.md` inside it). No resolver key, no code references — this is intentional manual triage, not a pipeline output. |
| `ops/tmp/` (26M) | `tmp_dir()` has **zero call sites** anywhere in the codebase. Contents are files up to 4+ months old with no cleanup mechanism — durable clutter under a "tmp" label. |
| `mlflow_artifacts` split-brain | **Not** a simple legacy-vs-current split. `ops/mlflow_artifacts` is the systemd-configured `--default-artifact-root`, but the top-level `mlflow_artifacts/` is still actively receiving writes (newer mtime, 2026-09-17) for the default `nq-50pt-intraday` MLflow experiment, because MLflow pins `artifact_location` per-experiment at creation time and the 2026-08 reorg never migrated the pre-existing experiment's pointer. Needs an explicit MLflow admin fix (update or recreate that experiment), not a documentation-only resolution. `mlflow.db` (top-level) is confirmed stale — the live backend store is Postgres. |
| `ops/mlruns/` (2.9M) | Incidental MLflow local file-store fallback from one HPO campaign run with a stale/unset tracking URI — not deliberately wired, zero code references. |
| `live/` vs `execution_dir()` | `ops/execution/` **does not exist on disk at all** — `execution_dir()` is vestigial. The functioning live-signal transport is the top-level `data/live/` directory, previously constructed independently in three places (`webui/live.py`, `webui/regmodel.py`, `execution/config.py`) — **fixed this pass**: `_LAYOUT["live"]` + `live_dir()` added, all three call sites now route through it. |
| `ops/webui/*`, `ops/gpu_health/`, `ops/gpu_jobs/`, `ops/paper/`, `ops/catalog/` | All resolver-backed, correctly scoped, no material findings. |
| `ops/job_state/` | Ingest/backfill checkpoint state, hardcoded via `data_path("ops","job_state")` in 2 scripts (not `_LAYOUT`-keyed) — a named, partially-executed consolidation target per `DATA_LAYOUT.md` (checkpoint JSON was meant to move out of `legacy/checkpoints/` here; some new writers target it directly, old writers still target `legacy/checkpoints`). |
| `mbp10_job.json` (root, 838B) | Orphaned manual copy of a Databento batch-job receipt; no code reads it; the real job-state convention (`ops/job_state/`) is one directory over. Migration candidate, not fixed this pass (would move a file). |

### 2.7 `legacy/` (538M)

| Subtree | Finding | Disposition |
|---|---|---|
| `checkpoints/` (180M) | **Not homogeneous** — (1) ~15 old Keras fold-CV model bundles, no production reader, diagnostic-script-only; (2) `hpo`/`hpo_ablation`, deprecated HPO script output, dead; (3) ~40 `ingest_*.json` files — **actively written today** via `checkpoints_dir()` from `src/common/repro.py` and ~6 live ingestion scripts | **NEEDS_DECISION** — live and dead content share one directory |
| `model_staging/` (179M) | Pre-reorg flat layout, not in `model_artifact_roots()`, zero references anywhere | `LEGACY_PRESERVE` |
| `research_runs/` (107M) | `program=` naming (pre-`objective_id` terminology); mostly dead, but `feature_group_ablation.py` (deprecated, diagnostic-only per `SCRIPT_INVENTORY.md`) would still write here if rerun | `NEEDS_DECISION` |
| `training_runs/` (73M) | Old Keras walk-forward output, zero live references (the analogous non-legacy path `scripts/ops/launch_v015.sh` targets doesn't exist on disk) | `LEGACY_PRESERVE` |

---

## 3. Current → canonical migration matrix

**Complete** (ninth-pass correction — the eighth pass's "representative rows only" disclaimer
is retracted). Every persistent class identified in §1–§2 above has a disposition below;
truly homogeneous paths are grouped into one row per group, named explicitly, rather than
repeated one row per subdirectory. No filesystem discovery was repeated to produce this —
every row traces to a class already inventoried in §1 or §2. Dispositions: `KEEP_CANONICAL`,
`KEEP_COMPATIBILITY`, `ADOPT_FOR_NEW_WRITES`, `MIGRATE_LATER`, `OPERATIONAL_KEEP`,
`LEGACY_PRESERVE`, `NEEDS_DECISION`.

### 3.1 `sources/raw/` (§2.1)

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sources/raw/live/<date>/stream_*.dbn[.zst]` | live wire-format archive | `live_streamer.py` | `webui/live.py` (monitoring) | Databento binary | append-only per file, compressed daily | session date + connect time, no version id | source dataset | source dataset (live archive) | *unchanged* | *unchanged* | none (gap) | `KEEP_CANONICAL` — DB write is inline; this is DR/replay only |
| `sources/raw/backfill/<class>/GLBX-<date>-<hash>/*.dbn.zst` | backfill batch download | 9+ scripts, mostly hardcoded | `ingest_dbn.py` → TimescaleDB | Databento binary | uneven retention (STOR-5) | Databento `job_id` (opaque) | source dataset | source dataset (backfill) | *unchanged* | *unchanged* | none (gap) | `NEEDS_DECISION` — resolver bypass (STOR-5) and non-uniform retention are both real, neither fixed this pass |
| `sources/raw/backfill/definitions/` + `dataset=GLBX.MDP3/schema=definition/...` | reference/contract metadata (2 coexisting layouts) | `download_ingest_definitions.py` (hardcoded) | contract-roll/options-surface lookups | Databento binary | static per pull | instrument root only | context dataset | reference/contract metadata | *unchanged* | *unchanged* | none (gap) | `NEEDS_DECISION` — duplicate layout, not consolidated this pass |
| `sources/raw/backfill/{<symbol>_ohlcv_1s,gaps_*,live_gap_*}/` | cross-asset + ad hoc gap-fills | one-off scripts | gap-detection follow-up | Databento binary | static, single-use | none | source dataset (ad hoc) | source dataset (ad hoc) | *unchanged* | *unchanged* | none | `OPERATIONAL_KEEP` — legitimate one-off remediation record, not a recurring class |

### 3.2 `sources/features/` (§2.2)

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sources/features/timeframe={tf}/symbol=/year=/month=/` | native-TF hive | `engine.py`/`native_tf.py` | `stores.py` (~47 sites), trainers | parquet | destructive rebuild per part; **conditionally reproducible, not a pure cache (§12 correction, STOR-12)** | `store_id` (mutable) | feature dataset | feature dataset (hive) | *unchanged* | *unchanged* | `store_id` (unchanged; no stronger mechanism exists) | `KEEP_CANONICAL` |
| `sources/features/parquet_5s/` | leftover 5s day files | **none — confirmed absent from disk** | 3 legacy tools still target it | n/a | dead | n/a | dead resolver entry | dead resolver entry | n/a | n/a | n/a | `LEGACY_PRESERVE` — key retained for the 3 tools, no data exists to migrate |
| `sources/features/parquet_options_surface/` | 0DTE options surface cache | `live_streamer.py`, `options_greeks_pipeline.py` | **none confirmed** | parquet | replaceable per session | date | feature dataset (unconfirmed reader) | feature dataset | *unchanged* | *unchanged* | none | `NEEDS_DECISION` — confirm a reader exists before treating as prunable (STOR-6); not investigated further this pass |
| `sources/features/parquet_options_greeks/` | full greeks archive | intends to write, blocked by a code defect (name-shadowing bug, `options_greeks_pipeline.py:567,631`) | none (empty) | parquet | n/a (write path never completes) | n/a | feature dataset (broken producer) | feature dataset | *unchanged* | *unchanged* | none | `NEEDS_DECISION` — a pipeline-code bug, not a storage-architecture defect; out of this pass's scope to fix |
| `sources/features/{parquet_tape,parquet_path1s}/` | tape / 1s-path caches | `build_tape.py`, `build_path1s.py` | 8+ consumers (largest fan-in) | parquet, per-session | replaceable per session | session date | feature dataset (flat) | feature dataset (flat) | *unchanged* | *unchanged* | session date (unchanged) | `KEEP_CANONICAL` |
| `sources/features/parquet_raw1s/` | raw1s+VIX single-experiment dataset | `build_raw1s_dataset.py` (hardcodes its own output path) | `finalize.py --parquet-dir` manual override | parquet | replaceable | date | feature dataset (single-purpose) | feature dataset | *unchanged* | *unchanged* | none | `MIGRATE_LATER` — resolver accessor exists and is unused; low value, one script |
| `sources/features/session_profile/` | session structural levels | `session_profile.py` | `build_move_matrix.py`, `build_target_matrix.py`, `webui/live_chart.py` | parquet, per-session | newest-build-wins, no run id (`session-levels-has-no-run-identity`, pre-existing, unfixed) | none | context dataset | context dataset | *unchanged* | *unchanged* | none (gap, unfixed) | `ADOPT_FOR_NEW_WRITES` — resolver key **added this pass**; the run-identity gap itself is unchanged |
| `sources/features/{parquet_labels_15s,parquet_labels_5s}/` | stale label-cache keys | none — directories absent | none | n/a | n/a | n/a | dead resolver entry (COMPAT) | dead resolver entry | n/a | n/a | n/a | `LEGACY_PRESERVE` — real label data lives under `sources/labels/{15s,5s}/`; keys retained per the sixth pass's `_prefer()` analysis |

### 3.3 `sources/labels/` (§1.3, §2.3) — Track B contract-addressed vs. Track C outcome datasets

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sources/labels/{5s,15s}/<contract_id>/` | contract-addressed label set | `label_store.py` | `fold_cache.py`, GPU sync | parquet + JSON sidecar | idempotent by default, **not content-immutable** (`force=True`, §5a) | `label_contract_id()` — semantic contract sha256 | outcome dataset | outcome dataset, contract-addressed | *unchanged* — already matches `STORAGE_CONTRACT.md` §17.1 with `triple_barrier` as the implicit `<outcome_family>` | *unchanged* | contract hash (semantic) + `build_id` (realization, recommended) | `KEEP_CANONICAL` |
| `sources/labels/move_matrix{,_v2,_v3,_v4}/` | "move matrix" | `build_move_matrix.py` (hardcoded literal, bumped by source edit) | `eval_move_model.py`, `build_target_matrix.py --move-dir` (tenth pass: both now REQUIRE an explicit path — no silent `_v3` default, §4 STOR-2) | parquet, 75–117 cols across generations | overwritten per version | directory suffix only | outcome dataset | outcome dataset, contract-addressed | `sources/labels/directional_first_passage/5s/<contract_id>/` (`STORAGE_CONTRACT.md` §17.1, §19.1) — write support implemented this pass (`build_move_matrix.py --canonical-write`), not executed | parquet + JSON sidecar (`_contract.json` + `_build.json`) | contract hash (`move_outcome_contract()`, §19.1) + `build_id` (genuine code/config evidence, `_build.json`) | `NEEDS_DECISION` for the actual rebuild — mechanism designed AND implemented this pass (§17, §19), execution/backfill of existing 2G+ explicitly out of scope |
| `sources/labels/target_matrix{,_v3,_v4,_adaptive,_context,_context_rest}/` | "target matrix" (2 naming semantics mixed) | `build_target_matrix.py` (`--out`) | `eval_target_model.py` (tenth pass: now REQUIRES an explicit path), WebUI (`_cascade_oof_paths`) | parquet, 22–66 cols across generations | overwritten per version | directory suffix, ambiguous (version vs variant) | outcome dataset (+ upstream move-dataset AND move-score-predictions references) | outcome dataset, contract-addressed | `sources/labels/candidate_level_reachability/5s/<contract_id>/`, referencing its move-dataset input's `contract_id`/`build_id` AND its `--scores` predictions input as separate upstream `DatasetRef`s (`STORAGE_CONTRACT.md` §17.1, §19.2) — write support implemented this pass, not executed | parquet + JSON sidecar (`_contract.json` + `_build.json`) | contract hash (`target_outcome_contract()`, §19.2) + `build_id` + upstream `DatasetRef` list (move dataset + move-score predictions, distinct semantic types) | `NEEDS_DECISION` for the actual rebuild — same as above |
| `sources/labels/firstpass*` (7 dirs, 624M) | "firstpass" family | `build_first_passage.py` et al. | mostly none (4/7 zero references) | parquet | static | directory suffix | superseded track | superseded track | *unchanged* (no physical move) | *unchanged* | n/a | `LEGACY_PRESERVE` — project's own roadmap already declares it superseded; archival requires explicit operator approval per AGENTS.md §4 |
| `sources/labels/1s/` | ADR-107 migration output | `convert_legacy_label_contracts.py` (one-time) | none | parquet, hive `year=` partitions | static | `label_contract_id` (orphaned) | dead migration artifact | dead migration artifact | *unchanged* | *unchanged* | n/a | `LEGACY_PRESERVE` |

### 3.4 `research/` (§2.4)

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `research/search/<objective_id>/<run_ts>/` | search run | `src/search/controller.py` | WebUI, `finalize.py`, `mlflow_backfill.py` | mixed (json/yaml/parquet) | append-only | `run_id = <objective_id>/<run_ts>` (adopted, but no `manifest.json` on disk yet, STOR-8) | run artifact root | run artifact root | *unchanged* | *unchanged* | `RunManifest.scientific_fingerprint()` once wired | `KEEP_CANONICAL` |
| `research/search/_fold_cache/<hash>/` | fold cache | `fold_cache.py` | search/HPO trials | `.npy` + `meta.json` | write-only, no eviction | sha256 of parquet fingerprint + label params | pure reproducible cache | reproducible cache | *unchanged*, or relocated out from under the "never-delete" `search/` root | *unchanged* | cache key (unchanged) | `OPERATIONAL_KEEP` — flagged for LRU eviction / relocation in a future pass, not this one (runtime behavior change) |
| `research/custom_test/<objective_id>/<run_ts>/` | custom test run | `src/search/controller.py` | WebUI, `serving/bundle.py`, `canonical/cli.py` | mixed | append-only | `run_id = <objective_id>/<run_ts>` (adopted, no `manifest.json` on disk yet) | run artifact root | run artifact root | *unchanged* | *unchanged* | `RunManifest.scientific_fingerprint()` once wired | `KEEP_CANONICAL` |
| `research/custom_test/<objective_id>/OLD/<run_ts>/` | manually archived run | operator (`mv`), not code | none | mixed | frozen | same as above, extra `OLD/` level | run artifact root, archived | run artifact root | drop the extra `OLD/` level going forward (do not move existing) | *unchanged* | *unchanged* | `LEGACY_PRESERVE` |
| `research/episodes/<objective>/<run_ts>__<source>.*` | episode | `episodes.py::build_episodes` | `patterns.py` (Research Director) | parquet + json | append-only | `<run_ts>__<source_id>` flat filename | analytical dataset | analytical dataset (episodes) | *unchanged* | *unchanged* | *unchanged* | `KEEP_CANONICAL` |
| `research/research_director/<objective_id>/<run_ts>/` | RD session state | `research_director/orchestrator.py` | RD gate, humans | JSON | append-only | `<objective_id>/<run_ts>`, microsecond-precision (drift from second-precision elsewhere) | analytical run record | analytical run record | *unchanged* | *unchanged* | *unchanged* | `OPERATIONAL_KEEP` — the precision drift is cosmetic, not a correctness risk |
| `research/research_director/regimes/` | RD regime output | `scripts/search/regime_analysis.py` (was hardcoded absolute path) | RD training filter | parquet + txt | overwritten | none | analytical cache | analytical cache | `research/research_director/regimes/` (unchanged path, resolver-backed now) | *unchanged* | *unchanged* | `ADOPT_FOR_NEW_WRITES` — **done this pass** |
| `research/scoping/<objective_id>/<run_ts>/` | scoping session | `scoping/runner.py` (second-precision) **and** `scoping/orchestrator.py` (microsecond-precision) — 2 producers, same nominal convention | `webui/datasources.py`, `scoping/rag.py` | JSON | append-only | `<objective_id>/<run_ts>` | analytical run record | analytical run record | *unchanged* | *unchanged* | *unchanged* | `OPERATIONAL_KEEP` — two producers with different `run_ts` grain, cosmetic drift not fixed this pass |

### 3.5 `evaluation/` + `models/` (§2.5)

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `evaluation/<objective>/contract_v*.json` | frozen eval contract | `src/evaluation/contract.py` | `backtest/runner.py`, `search/controller.py` | JSON | frozen by intent | objective + integer version | frozen contract | frozen contract | *unchanged* | *unchanged* | *unchanged* | `KEEP_CANONICAL` |
| `evaluation/<objective>/latest_leaders.json` (duplicate copy) | leaderboard (duplicate) | stale writer / operator copy | none confirmed | JSON | mutable | candidate_id | leaderboard | leaderboard | `models/leaders/<objective>/` only | *unchanged* | *unchanged* | `NEEDS_DECISION` — duplicate should stop being written; existing copies left in place (no deletion this pass) |
| `evaluation/{context_analysis,move_analysis}/` | ad hoc SHAP dumps | `scripts/features/analyze_{context,move}_scores.py` (hardcoded) | none (human-read) | csv/png/md | clobbered per run | none | diagnostic artifact, not an objective | diagnostic artifact | *unchanged* path, route through `evaluation_dir()` | *unchanged* | run_ts recommended, not enforced | `MIGRATE_LATER` — low value, single-script fix, deferred |
| `evaluation/portfolio_sim/<label>/` | portfolio simulation output | `scripts/search/run_portfolio_sim.py` | none confirmed (manual) | parquet + JSON | frozen | hand-chosen `--out-dir` label, no `objective_id` in path | analytical run record | analytical run record | *unchanged* | *unchanged* | none (gap) | `MIGRATE_LATER` — no `objective_id`/run identity in path; low volume, not fixed this pass |
| `models/model_staging/<objective>/<candidate>_<run_ts>/` | serving bundle (staged) | `src/search/finalize.py` | `src/serving/promote.py`, Live/Backtest resolve | mixed (bin + json + parquet) | mutable in practice (mixed mtimes within one dir) | `<objective>/<candidate>_<run_ts>` | serving bundle | serving bundle | *unchanged* | *unchanged* | `finalize_manifest.json`'s fields (not yet `RunManifest`-linked, STOR-7) | `KEEP_CANONICAL` |
| `models/model_registry/<objective>/<version>/` + `LATEST.json` | serving bundle (promoted) | `src/serving/promote.py` | serving | mixed (bin + json + parquet) | `LATEST.json` mutated on promote; versioned dir copy-once by convention | `<objective>/<version>` | serving bundle (promoted) | serving bundle (promoted) | *unchanged* | *unchanged* | `LATEST.json`'s fields (not yet `RunManifest`-linked) | `KEEP_CANONICAL` |
| `models/leaders/<objective>/{leaderboard_v*.jsonl,latest_leaders.json}` | leaderboard | `src/evaluation/leaderboard.py` | promotion, WebUI | JSONL (append) + JSON (pointer) | append-only + mutable pointer | `candidate_id` (weak, no FK to a serving bundle, STOR-7) | leaderboard | leaderboard | *unchanged* | *unchanged* | *unchanged* | `KEEP_CANONICAL` — the mechanism is right; the missing FK to a serving bundle is a lineage gap (STOR-7), not a path problem |
| `models/training_runs/` | training runs | **none — directory does not exist** | **none — zero callers of `training_runs_dir()`** | n/a | n/a | n/a | dead resolver entry | dead resolver entry | n/a | n/a | n/a | `NEEDS_DECISION` — remove the key or wire a producer; not touched this pass (would be a behavior decision, not a documentation fact) |

### 3.6 `ops/` + misc top-level (§2.6)

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ops/quarantine/` | quarantined corrupt data | operator, manual | none (pipeline avoidance) | parquet + README | frozen pending remediation | filename + README | operational record | operational record | *unchanged* | *unchanged* | none | `OPERATIONAL_KEEP` |
| `ops/tmp/` | scratch | various, `tmp_dir()` has zero call sites | none confirmed | mixed | **not actually ephemeral** (files 4+ months old, STOR-10) | none | ephemeral scratch (misclassified in practice) | ephemeral scratch | *unchanged* | *unchanged* | none | `NEEDS_DECISION` — needs an actual TTL/purge mechanism, a runtime change out of this pass's scope |
| `ops/mlflow_artifacts/` **and** top-level `mlflow_artifacts/` | MLflow artifact store (split-brain) | MLflow server (Postgres-backed) | MLflow UI, `mlflow_backfill.py` | MLflow-native | append-only | MLflow `experiment_id`/`run_id` | platform tracking store | platform tracking store | `ops/mlflow_artifacts/` only, once the default experiment's `artifact_location` is repointed | *unchanged* | *unchanged* | `NEEDS_DECISION` — requires an MLflow admin action, out of this pass's scope (no DB writes) |
| `ops/mlruns/` | incidental MLflow file-store fallback | one HPO campaign run with a stale tracking URI | none | MLflow-native | append-only, orphaned | MLflow `experiment_id` | platform tracking store (incidental) | platform tracking store | *unchanged* | *unchanged* | *unchanged* | `LEGACY_PRESERVE` |
| `ops/webui/{custom_tests,jobs,external_models,regmodel,custom_feature_views}/` | WebUI operational state | `src/webui/*` (10 call sites, all `webui_dir()`-backed) | WebUI backend | JSON/YAML | mutable | subsystem-specific | operational state | operational state | *unchanged* | *unchanged* | subsystem-specific | `KEEP_CANONICAL` |
| `ops/{gpu_health,gpu_jobs}/` | GPU orchestration state | `src/gpu_runner/*` | dispatcher, WebUI, `health_monitor.py` | JSON/JSONL | mutable, live | `run_id` | operational state | operational state | *unchanged* | *unchanged* | `run_id` (unchanged) | `KEEP_CANONICAL` |
| `ops/job_state/` | ingest/backfill checkpoint state | 2 scripts, hardcoded (`data_path("ops","job_state")`, not `_LAYOUT`-keyed) | resumable ingest jobs | JSON (sharded) | mutable | job-specific | operational checkpoint | operational checkpoint | *unchanged*, gains a `_LAYOUT` key | *unchanged* | *unchanged* | `MIGRATE_LATER` — a clean, small resolver addition; not made this pass to keep this pass's code changes to the set already verified against tests |
| `ops/paper/<objective>/{trades.jsonl,...}` | paper-trading blotter | `src/paper/ledger.py` | paper engine | JSONL + JSON | append-only ledger | objective_id | operational state | operational state | *unchanged* | *unchanged* | *unchanged* | `KEEP_CANONICAL` — inactive since 2026-08-05, mechanism unchanged |
| `ops/catalog/data_capability.json` | data-availability catalog | ingestion, `research_director/tools.py` | scoping, RD | JSON | mutable | none | operational state | operational state | *unchanged* | *unchanged* | none | `KEEP_CANONICAL` |
| `ops/logs/` | shell-redirected log files | manual (no `logs_dir()` accessor, zero code references) | humans | text | rotating (manually) | none | vestigial operational output | vestigial operational output | *unchanged* | *unchanged* | none | `LEGACY_PRESERVE` |
| `live/{signals.jsonl,heartbeat.json}` | live signal bridge | `src/webui/live.py` | `src/execution/config.py`, `src/features/live_runtime.py`, `scripts/ops/tail_live_signals.py` | JSONL + JSON | append-only | objective_id + session_date + ts | live signal bridge (documented `ops/` exception, §8a) | live signal bridge | top-level `live/` current; `ops/live/` semantic target, not migrated | *unchanged* | *unchanged* | `ADOPT_FOR_NEW_WRITES` — resolver **extended this pass** (`_prefer(ops/live, live)`, path-identical today); 4th hardcoded call site (`tail_live_signals.py`) also fixed this pass |
| `exports/feature_catalog.csv` | feature dictionary export | `scripts/features/export_feature_catalog_csv.py` (hardcoded) | humans | CSV | manually regenerated | none | reference export | reference export | *unchanged* | *unchanged* | none | `OPERATIONAL_KEEP` |
| `mbp10_job.json` (data root) | orphaned batch-job receipt | manual copy, no code writer | none — no code reads it | JSON | static | none | orphaned operator artifact | orphaned operator artifact | `ops/job_state/` (matches the sharded job-JSON convention) | *unchanged* | none | `MIGRATE_LATER` — a single loose file, zero consumers, low priority |

### 3.7 `legacy/` (§2.7)

| CURRENT PATH | CURRENT TERM | PRODUCER | CONSUMER | FORMAT | MUTABILITY | CURRENT IDENTITY | CANONICAL CLASS | CANONICAL TERM | CANONICAL PATH | CANONICAL FORMAT | CANONICAL IDENTITY | DISPOSITION |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `legacy/checkpoints/{model_v*,hpo,hpo_ablation}/` | old Keras models + deprecated HPO dumps | none (dead) | diagnostic scripts only | mixed (Keras + JSON) | frozen | snapshot name | archive | archive | *unchanged* | *unchanged* | n/a | `LEGACY_PRESERVE` |
| `legacy/checkpoints/ingest_*.json` | ingest checkpoint | 6 ingestion scripts + `src/common/repro.py` | resumable ingest jobs | JSON | mutable | job-specific | operational checkpoint | operational checkpoint | `ops/job_state/` (per `DATA_LAYOUT.md`'s own, partially-executed consolidation) | JSON | *unchanged* | `NEEDS_DECISION` — a real writer targets a directory named "legacy"; not moved this pass |
| `legacy/{model_staging,training_runs}/` | pre-reorg archive | none (dead) | none | mixed | frozen | `program=` snapshot name | archive | archive | *unchanged* | *unchanged* | n/a | `LEGACY_PRESERVE` |
| `legacy/research_runs/program=nq-50pt-08-11-cst/` | pre-reorg HPO/ablation hive | `feature_group_ablation.py` (deprecated, would still write here) | none | mixed | frozen except for one live writer | `program=` snapshot name | archive (with one live writer) | archive | *unchanged* | *unchanged* | n/a | `NEEDS_DECISION` |

---

## 4. Confirmed storage-architecture risks (ranked)

Format: problem → concrete consequence → canonical remedy. Only genuine storage/data
integrity risks; code-correctness bugs found incidentally (the `options_greeks_pipeline.py`
name-shadowing bug) are named but not counted here — they belong to a pipeline-code pass, not
a storage-architecture one.

1. **STOR-1 — `_fold_cache/` unbounded growth (39G, ~89% of `research/`).** *Consequence*: the single largest directory in the whole tree, growing ~2.8x in six weeks with a deferred eviction mechanism that was never implemented; it sits inside a directory explicitly protected by an operational "never delete" policy, risking that policy sweeping in fully-discardable cache bytes. *Remedy*: implement the already-designed LRU eviction (`docs/operations/DISK_RECLAIM_2026_08.md` Stage 3) or relocate the cache out from under `research/search/` — both are runtime behavior changes, correctly out of this documentation/resolver pass.
2. **STOR-2 — Track C dataset version drift was live, not just historical. FIXED this pass (tenth).** *Was*: `eval_move_model.py`, `build_target_matrix.py --move-dir`, and `eval_target_model.py` all defaulted to one generation behind their upstream producer's current output; a bare invocation of any of them silently trained/evaluated against stale input with no error. *Remedy, applied*: `eval_move_model.py --dir` and `eval_target_model.py --dir` are now `required=True` (no default at all); `build_target_matrix.py --move-dir` now fails clearly at startup ("--move-dir is required when --context != none") rather than silently defaulting to `move_matrix_v3`. No consumer picks a version-suffixed historical dataset implicitly anymore — the caller must name one explicitly, or use `--context none`. The producers' own OWN version-suffixed output literals (`build_move_matrix.py`'s `move_matrix_v4` default, `build_target_matrix.py`'s unsuffixed `target_matrix` default) are unchanged — that is the producer's own self-versioning (STOR-4), not a consumer silently selecting a stale input, and is superseded by `--canonical-write` (§17, §19) for future writes.
3. **STOR-3 — MLflow `mlflow_artifacts` split-brain.** *Consequence*: the resolver-configured `ops/mlflow_artifacts/` is not actually where the default experiment's runs land; new artifacts for the most-used experiment continue writing to the pre-reorg top-level directory because MLflow pins `artifact_location` per-experiment at creation and the 2026-08 reorg never repointed it. *Remedy*: an explicit MLflow admin action (repoint or recreate the experiment) — out of this pass's no-DB-writes scope.
4. **STOR-4 — Track C outcome datasets have no version-identity mechanism beyond a directory suffix, applied inconsistently** (source-hardcoded for `move_matrix`, CLI-discipline-only and semantically overloaded for `target_matrix`). *Consequence*: no way to tell from the filesystem which `move_matrix`/`target_matrix` pairing is mutually consistent without reading a build log. *Remedy*: adopt Track B's `label_store.py` CONTRACT-addressing pattern tree-wide, generalized in `docs/STORAGE_CONTRACT.md` §17. **[UPDATED, tenth pass]**: the pattern is now finalized (§17.1: `<outcome_family>/<resolution>/<contract_id>/`, replacing the ninth pass's overloaded single-slot proposal), Track C's semantic contract fields are classified (§19), and write support is implemented (`build_move_matrix.py`/`build_target_matrix.py --canonical-write`, `label_store.outcome_set_dir()`/`write_outcome_contract()`/`write_build_sidecar()`) — but rebuilding the existing 2G+ of Track C data under the new paths remains explicitly out of scope, still `NEEDS_DECISION`.
5. **STOR-5 — Resolver bypass is the norm, not the exception, for `sources/raw/backfill` producers.** At least 9 of ~12 backfill scripts hardcode `/opt/vtech/data/sources/raw/...`, silently breaking under a non-default `VTECH_DATA_DIR`. *Remedy*: route through `raw_dir()` — deferred; 9 independent scripts exceeds this pass's "small, unambiguous" bar for a single change.
6. **STOR-6 — `parquet_options_surface` (2.5G, the single largest `sources/features/` subdirectory) has no confirmed in-repo reader.** *Consequence*: possible write-only archive; safe to leave, unsafe to assume prunable without confirming no external (notebook/ad hoc) consumer exists.
7. **STOR-7 — Model/leaderboard lineage chain breaks at the same point repeatedly.** `models/leaders/` rows carry no foreign key to `model_staging`/`model_registry`; `model_staging`'s `finalize_manifest.json` points at a `search_run_dir` that itself has no `manifest.json`. *Consequence*: no run in the model-promotion path is traceable end-to-end today. *Remedy*: wiring `RunManifest` end-to-end is the designed fix and is explicitly out of scope for this pass (Track C manifest integration is the next pass; Track B's own integration is written but not yet exercised against a real run — see STOR-8).
8. **STOR-8 — Track B's `RunManifest` integration exists only in the working tree.** *Consequence*: zero `manifest.json` files exist anywhere under `research/search/` or `research/custom_test/` despite `src/search/controller.py` (currently `git status: M`, uncommitted) calling `run_manifest.py` to write one. *Remedy*: none needed from this pass — this is a deployment/commit-state fact, not an architecture defect; recorded so the next pass does not assume Track B manifests are already flowing. **[SCOPE CLARIFIED, 2026-09-23]** This finding is specifically about `research/search/` and `research/custom_test/`, and remains true for both. It is NOT a claim that no `manifest.json` exists anywhere: the objective-run tree now has one on disk at `research/runs/nq-Rscaled-mkt/20260923T132829Z/manifest.json`, written by `merge_pass()` and read by `read_run_record()`. Track B's own integration is still unexercised against a real Track B run.
9. **STOR-9 — Loose top-level files with no consistent home** (`mbp10_job.json`, and arguably `mlflow.db`/`optuna*.db`). *Consequence*: minor discoverability cost only. *Remedy*: migration candidates into `ops/job_state/`/`ops/`, not moved this pass (would mutate file locations).
10. **STOR-10 — `ops/tmp/` and `legacy/checkpoints/` each mix live and dead content under one directory name.** `tmp_dir()` has zero call sites yet holds 4+-month-old files with no cleanup; `legacy/checkpoints/` holds genuinely dead old-architecture model bundles alongside actively-written ingest-job JSON. *Consequence*: a directory's name is not a reliable signal of its mutability/lifecycle class; an operator cannot safely treat either as uniformly safe-to-clear or uniformly frozen. *Remedy*: `NEEDS_DECISION` — splitting `legacy/checkpoints/`'s live ingest-JSON slice out to `ops/job_state/` is the designed fix (already partially underway per `DATA_LAYOUT.md`) but touches active writers in 6+ scripts, deferred.
11. **STOR-11 — `DatasetRef.content_sha256` was fabricated from contract metadata, not dataset content. FIXED (ninth pass); its own remedy corrected (tenth pass).** `src/search/run_manifest.py::_datasets_from_run_dir()` surfaced `label_contract.json`'s `contract_sha256` field — `hashlib.sha256((source/"_contract.json").read_bytes())`, a hash of the SEMANTIC CONTRACT SIDECAR — as `DatasetRef.content_sha256`, which claims to be a hash of the label PARQUET content. The two never diverge together: `_contract.json`'s bytes are entirely a function of the normalized contract + `created_at_utc` (preserved across rewrites), so `contract_sha256` stays identical across a `force=True` rebuild that changes every actual data byte. *Consequence (if unfixed)*: any future run-comparison or evidence-completeness check trusting `content_sha256` would have silently certified "same content" for two runs that could have used differently-rebuilt label parts. *Remedy, ninth pass*: `content_sha256` set to always `None` on this path; a new `DatasetRef.build_id` field was populated from a fingerprint of the already-recorded part inventory (`{name, bytes}`). **[CORRECTED, tenth pass]**: that `build_id` mechanism was itself not genuine build/provenance evidence — a fingerprint of OUTPUT filenames/sizes says nothing about which code, config or upstream data produced the realization. `build_id` on this adapter's path is now always `None` too (no genuine evidence is available to it); `contract_id`-only `DatasetRef`s now correctly count as missing `datasets` evidence rather than complete (`RunManifest.missing_evidence()`). Genuine `build_id`s are populated elsewhere, by `label_store.write_build_sidecar()` (§17 of `STORAGE_CONTRACT.md`), from real code/config/upstream evidence. See `docs/STORAGE_CONTRACT.md` §5a.
12. **STOR-12 — The native-TF feature hive was misclassified as a pure reproducible cache with "no backup requirement." CORRECTED this pass.** `src/features/engine.py`'s own module docstring states it "loads source data from TimescaleDB" (confirmed via its `psycopg` usage) — not from `sources/raw/` files directly — and STOR-5 already found `sources/raw/` retention is non-uniform. *Consequence*: reproducing the hive genuinely depends on TimescaleDB still holding the historical `tbbo`/`bbo_1s`/`definitions` rows, a durability property this (and every prior) documentation pass has not verified, since querying the database is out of scope. Calling this "safe, no backup needed" was an unverified claim stated as fact. *Remedy*: reclassified as "derived feature store, conditionally reproducible" (`docs/STORAGE_CONTRACT.md` §12), distinct from `_fold_cache` (confirmed genuinely pure — file-only inputs, no DB dependency). A TimescaleDB backup/retention audit is listed as an unresolved decision (§16 of `STORAGE_CONTRACT.md`), not performed this pass (no DB queries permitted).

Risks already named and fixed by the eighth/ninth passes' resolver additions (§13 of
`STORAGE_CONTRACT.md`): the `live/` resolver gap (H13), `session_profile`'s missing
`_LAYOUT` key, `research_director/regimes`'s hardcoded absolute path, and (ninth pass)
`scripts/ops/tail_live_signals.py`'s independent fourth hardcoded `data_root()/"live"`
construction, plus STOR-11 above.

**[CORRECTED, tenth pass]** The line above previously described `live_dir()` as having
"a forward-compatible `ops/live` preference." That auto-preference was reverted this pass
(`docs/STORAGE_CONTRACT.md` §8a): `live_dir()` always resolves to the current top-level
`live/` path; adopting `ops/live/` requires an explicit future cutover, not a filesystem
probe. STOR-2 (Track C stale-default drift) is also now FIXED, not merely flagged — see
its entry above.
