# Data Layout — Runtime Root (`VTECH_DATA_DIR`)

> **Run root (2026-09-23)**: new runs write `research/runs/<obj>/<run_ts>/`. `research/custom_test/` is gone. `research/search/` is a compatibility root, not the HPO layout. The 2026-09-03 paragraph below is the snapshot from that date.
> **Snapshot (2026-09-03)**: Operator reorg on disk + code path SSOT in `src/common/data_paths.py`. Custom Test last-fold bundles then lived under `research/custom_test/<obj>/<run_ts>/models/<cand>/` (ADR-036). Automated search was OOF-only under `research/search/`. Search leaderboards live under `models/leaders/<objective_id>/` (moved off the data-root `nq-50pt*` folders). `parquet_path1s` + `parquet_pathlabels` are live ADR-052/053 caches. Feature hive: 5s research matrix monthly parts under `sources/features/timeframe=5s/` (ADR-079); 15s 1-bar monthly parts under `timeframe=15s/` (ADR-075); 1m monthly parts under `timeframe=1m/` (ADR-085); 5m monthly parts under `timeframe=5m/` (ADR-090); 1h / 1d HTF-context monthly parts under `timeframe=1h/` and `timeframe=1d/` (ADR-091). Label sidecars are contract-addressed under `sources/labels/{5s,15s}/<contract_id>/` (ADR-107); old daily label files remain for audit only. GPU dispatch and OCI sync mirror the selected contract prefix to `vtech_hot`. Live GPU job heartbeats: `ops/gpu_jobs/<run_id>/` (ADR-073). External Model defaults: `ops/webui/external_models/` (ADR-058/062, including `_book`). Custom Test saved views: `ops/webui/custom_feature_views/` (ADR-074).
> **Supersedes**: Flat `data/parquet`, `data/model_staging`, `data/search`, …  
> **Next**: Operator verify + delete leftover `parquet_5s/` day files (ADR-079 / ADR-081). Serving asof for `tf_oof_meta` (ADR-093). Optional cleanup of leftover `vtech_hot/parquet/`. Push GPU `timeframe=1m/` / `5m/` / `1h/` / `1d/` (ADR-078).
> **Rule**: **Bulk bytes stay on disk / OCI — not in git.** Repo owns the **layout contract** only.

---

## 1. Design principle

| Layer | Location | In git? |
|-------|----------|---------|
| Code / config schemas | worktree or `/opt/vtech` checkout | Yes |
| Runtime data root | `VTECH_DATA_DIR` (default `/opt/vtech/data`) | **No** (`data/*` gitignored) |
| Hot mirrors | OCI `vtech_hot` / `vtech_archive` | No |

Every code worktree should set `VTECH_DATA_DIR=/opt/vtech/data` (or a shared volume) so staging/search/parquet are **shared**, not copied per branch.

### 1.1 Live layout (as reorganized)

```text
$VTECH_DATA_DIR/
  sources/raw/
  sources/features/parquet_5s|parquet_raw1s|parquet_path1s|parquet_pathlabels|parquet_labels_15s|parquet_labels_5s|parquet_options_surface|parquet_options_greeks/  # legacy label dirs retained
  sources/labels/{5s,15s}/<contract_id>/{_contract.json,labels_<resolution>_partN.parquet}
  sources/features/timeframe={5s,15s,1m,5m,1h,1d}/symbol=NQ/year=YYYY/month=MM/   # hive parts live (ADR-079/075/085/090/091)
  models/model_staging|model_registry|training_runs|leaders/
  research/search|custom_test|scoping|research_director|episodes/
  evaluation/                    # frozen contracts (leaderboards: models/leaders/)
  ops/catalog|gpu_health|gpu_jobs|webui|tmp|mlflow_artifacts|paper|execution/
  # ops/webui/external_models/ — ADR-058/062 saved defaults + hardened snapshots
  #   <id>/defaults.json, <id>/hardened/*.json, _book/defaults.json
  # ops/webui/custom_feature_views/<name>.json — ADR-074 Custom Test saved views
  legacy/checkpoints|hpo|research_runs/
```

Live paper (P2): ``ops/paper/<objective_id>/{trades.jsonl,events.jsonl,summary.json,open_state.json}``
— limit+OCO fills from Databento TBBO (``--stream`` or ``--replay-db``), not bar-close backtest.

Broker execution (ADR-032): `ops/execution/state/` holds cursor/decision state and
`ops/execution/bridge/{outbox,inbox,processed}/` is the local test/default transport.
Production may point `VTECH_EXECUTION_BRIDGE_DIR` at a private SMB-mounted directory.
Outbox commands and inbox events are immutable, one JSON document per ID.

Code resolves via **`src.common.data_paths`** (prefers new paths; falls back to old flat names if present).

---

## 2. Present inventory (live disk)

| Path | ~Size | Role today | Code activity |
|------|------:|------------|---------------|
| `raw/` | **29G** | Tier-1 DBN archive (backfill + live) | **Active** ingest/archive; **never delete without explicit approval** |
| `swapfile` | **17G** | OS swap on data volume | Infra; not ML workflow |
| `search/` | **14G** | Search waves, OOF, `_fold_cache` | **Active** `src/search/*`, WebUI backtest |
| `parquet_options_surface/` | **2.1G** | 0DTE surface day parquets | **Active** greeks pipeline + live dual-write |
| `parquet_5s/` (`parquet/`) | **~765M** | Leftover 5s day files (not written; operator may delete after verify) | Idle |
| `features/` | **466M** | Hive snapshot tree (Phase 18 export) | **Legacy-ish** — 288 days, **subset of parquet**; promote/tools still reference |
| `model_staging/` | **182M** | Serve candidates by objective | **Active** finalize + Live resolve |
| `checkpoints/` | **180M** | Old trainer folds + **ingest job JSON** | **Mixed** — models legacy; ingest ckpts still used by scripts |
| `parquet_raw1s/` | **128M** | Raw 1s + VIX dataset (193 days) | **Active** LSTM/raw search path + OCI `vtech_hot/parquet_raw1s/` |
| `sources/features/parquet_path1s/` | **1.1G** | 1s OHLC fill path (3,919 sessions, 2011-01-03→2026-06-30) | **Active** ADR-052 simulator + ADR-053 labels; OCI `vtech_hot/parquet_path1s/` |
| `sources/features/parquet_pathlabels/` | **554M** | Honest first-touch labels + dense targets (3,906 sessions) | **Legacy audit** — converted to `sources/labels/1s/<contract_id>/` (ADR-107) |
| `sources/labels/1s/<contract_id>/` | **live** | 1s pathlabels by contract (`c432d21a2903c0d6`); yearly subfolders (`year=YYYY`) + `_geography.csv` | **Active** `convert_legacy_label_contracts.py`; contract-addressed (ADR-107) |
| `sources/labels/5s/<contract_id>/` | **live** | 5s 3-class TBM labels + aux targets; compact, contract-addressed parts | **Active** `build_labels_5s.py`; CPU/GPU resolve the objective contract (ADR-107) |
| `sources/labels/15s/<contract_id>/` | **live** | Dual-binary 15s labels from the 5s TBM path; compact, contract-addressed parts | **Active** `build_dual_binary_15s.py`; CPU/GPU resolve the objective contract (ADR-107) |
| `sources/features/parquet_labels_{5s,15s}/` | legacy | Mutable daily label files retained for audit only | Converted to contract-addressed 5s sets (ADR-107) |
| `evaluation/` | **110M** | Frozen eval contracts (`contract_v*.json`); older leaderboard copies may remain | **Active** contract load; live leaderboard writer is `models/leaders/` |
| `models/leaders/` | **192K** | Per-objective `leaderboard_v*.jsonl` + `latest_leaders.json` (was `data/<objective_id>/`) | **Active** `src/evaluation/leaderboard.py` |
| `research_runs/` | **107M** | Phase-18 HPO/ablation hive | **Legacy** — OCI archive target; not modern search path |
| `training_runs/` | **76M** | Finalize refits + old program= trees | **Active** for econ finalize; old program= mostly archive |
| `tmp/` | **26M** | Logs, probes, junk | **Active** scratch; purgeable |
| `research_director/` | **11M** | RD session artifacts + library | **Active** RD orchestrator |
| `mlflow_artifacts/` | **4.7M** | Local MLflow file artifacts | **Active** if tracking uses local root; config prefers S3 |
| `scoping/` | **1.7M** | Scoping v2 sessions + `_library` | **Active** scoping |
| `mlflow.db` / `optuna*.db` | ~1.5M | Local tracking DBs | **Legacy/local** — MLflow server uses Postgres |
| `hpo/` | **76K** | Small HPO run dirs | **Low** — superseded by search/ |
| `gpu_health/` | **32K** | GPU monitor alerts | **Active** health_monitor |
| `ops/gpu_jobs/` | **live** | GPU job heartbeat + per-fold OOF (ADR-073) | **Active** `job_exec` / health_monitor scrape |
| `catalog/` | **28K** | `data_capability.json` | **Active** scoping/RD tools |
| `webui/` | **24K** | Live model selection + jobs + custom-test specs + saved feature views (`custom_feature_views/`, ADR-074) | **Active** webui |
| `model_registry/` | **0 files** | Production promote target | **Active path, empty** — nothing promoted yet |
| `parquet_options_greeks/` | **empty** | Optional full greeks day archive | Code supports; disk empty |
| `sources/features/timeframe=5s/…` | **~710M** | 5s research hive (ADR-079): monthly `features_5s_partN.parquet`, `bar_end_ts` | **Active** engine/store readers, data capability, snapshot export, HPO, live prediction display; OCI `vtech_hot/timeframe=5s/` |
| `sources/features/timeframe=15s/…` | **~54M** | Native 15s hive (ADR-075 / ADR-083): 14 1-bar + 5 RTH/5-bar flow + 4 `lg_*`/`st_*`, ~1.95 M bars, Globex hours, `bar_end_ts` | **Active** `build_native_tf_15s.py` / `patch_15s_rth_flow.py`; OCI `vtech_hot/timeframe=15s/` |
| `sources/features/timeframe=1m/…` | **live** | Native 1m hive (ADR-085): 71 cols so_/cs_/ps_/vw_/rv_/vp_/ms_ (no options IV), Globex hours, `bar_end_ts` | **Active** `build_native_tf_1m.py` |
| `sources/features/timeframe=5m/…` | **live** | Native 5m hive (ADR-090): 53 cols cs_/vp_/vw_/ps_/zz_/vf_/rv_ (VWAP ±2σ; ATR(5) on the 5m bar), Globex hours, `bar_end_ts` | **Active** `build_native_tf_5m.py`; OCI `vtech_hot/timeframe=5m/` |
| `sources/features/timeframe=1h/…` | **live** | Native 1h HTF context (ADR-091): 45 cols cs_/tr_/ps_/zz_/rv_/sr_ from `tbbo_1h`, `bar_end_ts` | **Active** `build_native_tf_htf.py --tf 1h`; OCI `vtech_hot/timeframe=1h/` |
| `sources/features/timeframe=1d/…` | **live** | Native 1d HTF context (ADR-091): 45 cols cs_/tr_/ps_/zz_/rv_/sr_ from Globex `tbbo_1d`, `bar_end_ts` | **Active** `build_native_tf_htf.py --tf 1d`; OCI `vtech_hot/timeframe=1d/` |
| `postgresql/` | ~empty | Leftover PG data dir on volume? | **Not app workflow** — leave to DBA |
| `lost+found` | — | fsck | Ignore |

### 2.1 Duplicate / overlapping content

| Overlap | Assessment |
|---------|------------|
| `parquet/` (333 days) ⊃ `features/` hive (288 days) | **Same feature product, two layouts.** Search/FE use flat `parquet/`. Hive is Phase-18 snapshot for OCI “Gold”. **Keep one canonical local cache (`parquet/`); treat hive as publish/export view or drop local copy after OCI verified.** |
| Phase-18 `legacy/features_hive` vs `sources/features/timeframe=` | **Different products.** Phase-18 is a 5s export snapshot. ADR-070 native-TF hive now has 15s/1m/5m/1h/1d parts. Do not merge. |
| `checkpoints/*` models ≈ copies under `model_staging/nq-50pt-08-*` | Phase-18 copied trainer folds into staging. **Models duplicated.** Prefer staging+training_runs as serve history; freeze checkpoints as cold archive. |
| `training_runs/program=*` vs `training_runs/<objective>/` | Two naming eras. **Canonical = objective_id keys** (finalize). |
| `research_runs/` vs `search/` | research_runs = old HPO hive; search = current controller. **No merge needed — archive research_runs.** |
| `hpo/` vs `search/` | hpo tiny/old. **Archive.** |
| Ingest JSON inside `checkpoints/` | Not models — **job resume state**. Should move to `ops/job_state/` so “checkpoints” is not overloaded. |

---

## 3. Active workflow → path map

```text
Live streamer ──► raw/live + TimescaleDB (+ optional parquet_options_surface)
Feature engine ──► timeframe=5s/…/features_5s_partN.parquet  (build/patch/extend, ADR-081)
Native TF hive ──► sources/features/timeframe={5s,15s,1m,5m,1h,1d}/symbol=NQ/year=/month=/
                 5s/15s: features_{tf}_partN.parquet; other TFs dirs only
Search/iterate ──► research/search/<objective>/<run_ts>/  (OOF + diagnostics; no weights)
Leaderboard    ──► models/leaders/<objective>/  (JSONL + latest_leaders.json)
Custom Test    ──► research/custom_test/<objective>/<run_ts>/models/<cand>/
                 (last-fold serve bundle + oof.parquet; meta_stack is its own folder)
                 ◄── timeframe=5s (or 15s / raw1s)
Finalize       ──► models/model_staging/<objective>/<candidate>_<run_ts>/
                 (single serve bundle + finalize_manifest; no training_runs dual-write)
Promote        ──► models/model_registry/<objective>/<version>/
                 + models/model_registry/<objective>/LATEST.json
Live/WebUI     ◄── registry LATEST (preferred) else staging
RD baseline    ◄── registry LATEST **only** (must promote before RD improves)
Scoping        ──► research/scoping/<objective>/<run_ts>/
RD artifacts   ──► research/research_director/<objective>/<run_ts>/
Paper          ──► ops/paper/<objective>/
Execution      ──► ops/execution/state + bridge/{outbox,inbox}/
```

**Model lifecycle:**

| Path | Role |
|------|------|
| `research/search/<obj>/…` | Automated search / iterate / refine: OOF lab |
| `research/custom_test/<obj>/…` | User Custom Test: `models/<cand>/` last-fold bundle + OOF (ADR-036) |
| `models/leaders/<obj>/…` | Search/backtest leaderboard JSONL + `latest_leaders.json` |
| `models/model_staging/<obj>/…` | Serve bundles after finalize; **stay here if not promoted** |
| `models/model_registry/<obj>/LATEST.json` | Production pin; Live prefers; **RD baseline only** |
| `models/training_runs/` | **Legacy** — finalize no longer dual-writes |

**RD targets:** every objective with registry `LATEST` **and** RD enabled
(``ops/webui/rd_objectives.json``; toggle in WebUI Objectives tab).

---

## 4. Target shape (recommended)

Group by **plane**, not by historical phase. Names stay short; code gains `src.common.data_paths`.

```text
$VTECH_DATA_DIR/
│
├── market/                          # regenerable from raw/DB where possible
│   ├── raw/                         # ← raw/  (UNCHANGED content; NEVER bulk-delete)
│   │   ├── backfill/
│   │   └── live/
│   └── cache/                       # ML-ready matrices
│       ├── features_5s/             # ← parquet/   (canonical 5s FE)
│       ├── features_raw1s/          # ← parquet_raw1s/
│       └── options_surface/         # ← parquet_options_surface/
│       └── options_greeks/          # ← parquet_options_greeks/ (optional archive)
│
├── experiments/                     # research artifacts (large, archive to OCI)
│   ├── search/                      # ← search/
│   ├── evaluation/                  # ← evaluation/
│   ├── scoping/                     # ← scoping/
│   └── research_director/           # ← research_director/
│
├── models/                          # serve lifecycle
│   ├── staging/                     # ← model_staging/
│   ├── registry/                    # ← model_registry/
│   ├── leaders/                     # search leaderboards (present: models/leaders/)
│   └── training_runs/               # ← training_runs/ (objective-keyed only going forward)
│
├── ops/                             # not market, not models
│   ├── catalog/                     # ← catalog/
│   ├── gpu_health/                  # ← gpu_health/
│   ├── gpu_jobs/                    # ← live GPU progress (ADR-073)
│   ├── webui/                       # ← webui/
│   ├── job_state/                   # ← ingest/pipeline JSON currently in checkpoints/
│   ├── tmp/                         # ← tmp/
│   └── mlflow_artifacts/            # ← mlflow_artifacts/ (if still local)
│
└── _legacy/                         # cold; no active writers after cutover
    ├── checkpoints_models/          # model.keras era copies (from checkpoints/ minus job JSON)
    ├── features_hive/               # ← features/  (or delete local if OCI gold verified)
    ├── research_runs/               # ← research_runs/
    ├── hpo/                         # ← hpo/
    ├── training_runs_program/       # program= trees split out of training_runs
    └── dbs/                         # mlflow.db, optuna*.db if unused
```

**Infra left outside the ML layout (do not nest under market/experiments):**

| Path | Action |
|------|--------|
| `swapfile` | Keep at volume root or system swap location; document only |
| `postgresql/` | Do not treat as app data; confirm with PG install path |
| `lost+found` | Ignore |

---

## 5. Old → new map (including consolidations)

| Current | Target | Action |
|---------|--------|--------|
| `raw/` | `market/raw/` | **Move** (or symlink `raw` → `market/raw` during transition) |
| `parquet/` | `market/cache/features_5s/` | **Move** + update FE/search/GPU defaults |
| `parquet_raw1s/` | `market/cache/features_raw1s/` | **Move** |
| `parquet_options_surface/` | `market/cache/options_surface/` | **Move** |
| `parquet_options_greeks/` | `market/cache/options_greeks/` | **Move** (empty OK) |
| `features/` | ` _legacy/features_hive/` **or delete local** | **Consolidate**: not needed if `parquet/` + OCI `features/` verified; else cold archive |
| `search/` | `experiments/search/` | **Move** |
| `evaluation/` | `experiments/evaluation/` | **Move** |
| `scoping/` | `experiments/scoping/` | **Move** |
| `research_director/` | `experiments/research_director/` | **Move** |
| `model_staging/` | `models/staging/` | **Move** |
| `model_registry/` | `models/registry/` | **Move** (empty) |
| `training_runs/<objective>/` | `models/training_runs/<objective>/` | **Move** |
| `training_runs/program=*` | `_legacy/training_runs_program/` | **Move** cold |
| `checkpoints/*.keras` trees | `_legacy/checkpoints_models/` | **Move** cold (dup of old staging) |
| `checkpoints/ingest_*.json`, pipeline ckpts | `ops/job_state/` | **Move** (active writers) |
| `checkpoints/_experiments.jsonl` | `ops/job_state/` or MLflow only | **Move** or stop writing |
| `research_runs/` | `_legacy/research_runs/` | **Move** cold |
| `hpo/` | `_legacy/hpo/` | **Move** cold |
| `catalog/` | `ops/catalog/` | **Move** |
| `gpu_health/` | `ops/gpu_health/` | **Move** |
| *(new)* | `ops/gpu_jobs/` | **Active** GPU heartbeats (ADR-073) |
| `webui/` | `ops/webui/` | **Move** |
| `tmp/` | `ops/tmp/` | **Move**; enable TTL purge |
| `mlflow_artifacts/` | `ops/mlflow_artifacts/` | **Move** if kept |
| `mlflow.db`, `optuna*.db` | `_legacy/dbs/` | **Move** if Postgres MLflow is SoR |
| `swapfile`, `postgresql/`, `lost+found` | *(stay / out of tree)* | **No ML reorg** |

### Deletions (recommendations only — **not raw**)

| Candidate | Condition before delete |
|-----------|-------------------------|
| Local `features/` hive (466M) | OCI `vtech_hot/features/` complete **and** no code path requires local hive (after path helper) |
| Duplicate `checkpoints` model dirs already under `model_staging` | Spot-check a few dirs; then `_legacy` + age-out |
| `tmp/*` logs/pngs/scripts | Safe purge anytime (recreatable) |
| Empty `parquet_options_greeks/` | Keep empty dir after reorg (code may write later) |
| Local `mlflow.db` / optuna DBs | Only if confirmed unused vs Postgres/server |
| **Never** | `raw/**` without explicit operator approval after ingest safety confirmation |

---

## 6. Migration principles (when executing)

1. **Compat aliases first** — `data_paths.py` maps logical names → new paths; optional symlinks  
   `parquet → market/cache/features_5s` so old hardcodes keep working one release.  
2. **Move, don’t copy** (or copy-then-verify-then-remove) for large trees; prefer `mv` on same filesystem.  
3. **No raw deletes** in reorg.  
4. **Update** `oci_sync.conf`, GPU `sync_parquet.py` dest, finalize/live/serving defaults in same change.  
5. **Worktrees**: only documentation + path code; still point `VTECH_DATA_DIR` at the shared volume.  
6. **model_registry empty**: after reorg, promote the econ champion once so Live prefers registry over a sea of legacy staging folds.

---

## 7. Why this is optimal

| Goal | How target helps |
|------|------------------|
| Clear workflow | market → experiments → models → ops matches code planes |
| Less confusion | One feature cache; staging/registry/training_runs not mixed with ingest JSON |
| Disk headroom | Collapse parquet/features dup; cold-box legacy; purge tmp |
| Safe multi-worktree | Shared `VTECH_DATA_DIR`; no per-branch model copies |
| Aligns with ADR-028 | Layout contract in repo; bytes outside git |

---

## 8. Related

- [ADR-029](../decisions/ADR-029-data-layout-reorganization.md) — decision record  
- [WORKFLOW.md](WORKFLOW.md) — planes  
- [DATA_PIPELINE.md](DATA_PIPELINE.md) — three-tier market data  
- [TREE_INVENTORY.md](TREE_INVENTORY.md) — code tree  
- OCI: `config/runtime/oci_sync.conf`
