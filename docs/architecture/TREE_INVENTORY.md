# Tree Inventory — Architecture Consolidation (P0)

> **Public export supplement:** `src/demo.py`, the synthetic release test,
> portable CI, and publication documentation are additive. The private runtime
> is preserved. Public inclusion/exclusion is governed by
> [PUBLICATION_SCOPE.md](../PUBLICATION_SCOPE.md), not the historical operational
> KEEP dispositions below. No archived data is included in this export.

> **Present supplement — 2026-09-24:** additive independent research only.
> `src/features/nq_selective.py` (causal state),
> `src/backtest/nq_selective.py` (quote-event outcomes),
> `scripts/research/{train_nq_selective,audit_nq_selective}.py`, three
> `tests/architecture/test_nq_selective_*.py`,
> `config/objectives/nq-selective-intraday.yaml`, research protocol/results
> and ADR-138 are **KEEP** as reproducible rejected-hypothesis evidence.
> Artifacts use existing `data/research/runs/nq-selective-intraday/`.
> **Supersedes:** no existing path. **Next:** causal legacy rebuild/revalidation;
> no promotion. No existing path moved or removed.

> **Present**: Inventory of repository paths with role, owner plane, and disposition.  
> **Supersedes**: Ad-hoc layout growth through Phases 17–23.  
> **Next**: Execute dispositions under [ADR-028](../decisions/ADR-028-structure-first-layout-and-docs-workflow.md); keep this file updated as moves land.  
> **Date**: 2026-08-05. Data-path disposition for objective runs, Optuna HPO studies, Track B `custom_test`, and the named `evaluation/` trees is `docs/new_file_map.md` (2026-09-23). This inventory's code-tree dispositions are unchanged.  
> **Rule**: **Delete only with explicit operator approval.** Prefer keep / move / merge / archive.

**Disposition codes**

| Code | Meaning |
|------|---------|
| **KEEP** | Correct home; leave in place |
| **KEEP→DOC** | Keep code; living docs must describe it (currently missing/stale) |
| **MOVE** | Relocate under target layout (no logic change) |
| **MERGE** | Fold into another module/package |
| **ARCHIVE** | Superseded; move under `archive/` |
| **DEFER** | Needed later (P1–P5); note only |
| **NOISE** | Non-source clutter; out of active trees |
| **MISSING** | Needed for G2/G3; not present yet |

---

## 1. Repository root

| Path | Role | Plane | Disposition | Notes |
|------|------|-------|-------------|-------|
| `AGENTS.md` | Agent operating rules | ops/docs | **KEEP** | Extend with past/present/future doc rule (ADR-028) |
| `README.md` | Project entry | docs | **KEEP** | Already reflects multi-plane stack |
| `config.yaml` | Non-secret runtime knobs | config | **KEEP** | SSOT with env overrides (ADR-006) |
| `pyproject.toml` | Package metadata; `where=["src"]` | build | **KEEP** | Align entrypoints with `src.*` imports |
| `requirements.lock` | Pinned deps | build | **KEEP** | |
| `Makefile` | Operator shortcuts | ops | **KEEP→** expand as CLI surface | Prefer Makefile over script sprawl |
| `compose.yaml` | Compose (if used) | ops | **KEEP** | Verify still needed |
| `Dockerfile.gpu` | GPU image | gpu | **KEEP** | |
| `.env` | Secrets | config | **KEEP** (never commit) | |
| `src/` | Application code | — | see §2 | |
| `config/` | Objectives, agents, schemas, run specs | config | see §3 | Split schemas/specs/runtime |
| `docs/` | Living + reference + ADRs + roadmap | docs | see §4 | |
| `scripts/` | Operator CLIs / batch jobs | ops | see §5 | Group by plane |
| `tests/` | Automated tests | quality | **KEEP** | Mirror `src/` planes over time |
| `systemd/` | Service units | ops | **KEEP** | RD unit present, not enabled on host |
| `deploy/` | GPU bootstrap, webui sudoers | ops | **KEEP** | Deprecate git-review scripts later |
| `research_cases/` | RAG/evidence corpus | agents | **KEEP** | Index path must be documented |
| `archive/` | Superseded code/docs | archive | **KEEP** | Only true supersessions |
| `data/` | Runtime artifacts | data | **KEEP** (not source architecture) | On disk / not always in git; search leaderboards: `data/models/leaders/<objective>/`. Feature hive: `$VTECH_DATA_DIR/sources/features/timeframe={5s,15s,1m,5m,1h,1d}/symbol=NQ/year=/month=` (ADR-070 / ADR-079 / ADR-081 / ADR-085 / ADR-090 / ADR-091); files `features_{tf}_partN.parquet`; leftover day files in `parquet_5s/` until operator delete. **Print tape** (ADR-127): `sources/features/parquet_tape/tape_YYYY-MM-DD.parquet`, one lead-contract session per file, 94.1M prints over 319 sessions — the label substrate, replacing `parquet_path1s/` for the cascade path. Cascade labels: `sources/labels/move_matrix/move_*.parquet` (Stage 1), `sources/labels/target_matrix/target_*.parquet` (Stage 3), `sources/labels/move_scores.parquet` (OOF gate scores). |

---

## 2. `src/` packages (as-built)

| Path | Role | Plane | Disposition | Notes |
|------|------|-------|-------------|-------|
| `src/acquisition/` | Databento hist + **live streamer** | data | **KEEP** | Live API already implemented |
| `src/ingestion/` | DBN → TimescaleDB | data | **KEEP** | Optional future nest under `src/data/` |
| `src/features/` | Feature engine + modules + greeks pipeline | features | **KEEP** | includes `mbp_depth.py` (`mbp_`, ADR-057), `psar.py` (`ps_`, ADR-055), `volume_footprint.py` (`vf_`, ADR-056); greeks dual-write path; `bar_features.py` / `native_tf.py` / `native_tf_1m.py` / `native_tf_5m.py` / `native_tf_htf.py` / `hive_5s.py` (ADR-075 / ADR-079 / ADR-083 / ADR-085 / ADR-090 / ADR-091); `stores.py` Custom Test feature-store registry (ADR-076); `target_geometry.py` `lg_*`/`st_*` (ADR-082); `htf_context.py` slim `regime_v1` pack (ADR-095); `session_vp.py` fold-cache `svp_*` (ADR-099) |
| `src/objectives/` | Objective YAML loader | objectives | **KEEP** | YAML stays in `config/objectives/` |
| `src/training/cost_matrix.py` | 3×3 expected-cost train loss (ADR-072) | models | **KEEP** | Opt-in via `custom_optimization` |
| `src/training/job_progress.py` | GPU/CPU job heartbeat + per-fold OOF (ADR-073) | gpu / training | **KEEP** | `ops/gpu_jobs/<run_id>/` |
| `src/models/` | Plugin catalog, views, DL/tree plugins | models | **KEEP** | `keras_layers.py` — LastTimestep / GatherFeatures / first-party `.keras` load (ADR-059 / ADR-077) |
| `src/external_models/` | Pine/rule replicas (SAR + 20 EMA + ZigZag + Timed PSAR; book + stack; per-leg TF) | models (external) | **KEEP** | ADR-058 / ADR-062 / ADR-063 / ADR-064 / ADR-065 / ADR-096; not a catalog family |
| `src/training/` | Labels, fit primitives, walk-forward, xgb | training | **KEEP→DOC** | Boundary: fit primitives vs search orchestration; `aux_labels.py` (ADR-082 / ADR-086) |
| `src/evaluation/` | Frozen contract, KPI, episodes, patterns, **shap_explain** | evaluation | **KEEP** | ADR-014/021/026/027; `make shap MODEL=…` |
| `src/search/` | Controller, gate, iterate, refine, meta, finalize, **custom_test**, **tf_oof_meta** | search | **KEEP** | Primary experiment executor; Custom Test compiler (ADR-035 / ADR-074 saved views); native-TF dual-head composer (ADR-092) + Backtest serving asof (ADR-103) |
| `src/backtest/` | Rule DSL + prediction path BT | backtest | **KEEP** | Reuse semantics for paper brackets (P2); `cascade_backtest.py` bridges the TBBO cascade OOF (`target_scores.parquet`) to the WebUI Backtest tab (ADR-127) |
| `src/serving/` | Bundle load + canonical predict | serving | **KEEP→DOC** | Extend to **policy bundle** (P2); Keras via `load_keras_model` (ADR-059 / ADR-077) |
| `src/execution/` | Risk gateway, atomic broker protocol, broker event ledger | execution | **KEEP** | ADR-032; disabled by default |
| `src/scoping/` | Stage-0 dual-perspective research | agents | **KEEP** → later nest under `agents/` | ADR-025 |
| `src/research_director/` | Model-improvement orchestrator | agents | **KEEP** → later nest; **import fix** | As-built: synthesizer loop, not deleted Phase-20 files |
| `src/gpu_runner/` | OCI queue worker + job_exec + health_monitor | gpu | **KEEP** | ADR-022 / ADR-073 live tee + progress scrape; ADR-078 hive-relative parquet push/pull |
| `src/webui/` | Console (search, custom test, BT, live snapshot, **Live Chart**, **Notes**, **External Models**) | ui | **KEEP** | Custom Test: `#/objectives/<id>/test` (ADR-074 saved views); Live Chart: `#/chart` + `live_chart.py` (ADR-087 / ADR-097 / ADR-105 lookback / ADR-124 date-range lookback / ADR-108 nested overlay params + Volume Profile SSOT / ADR-109 viewport fit + ETH display / ADR-111 large prints / ADR-112 stacked panes / ADR-114 replay / ADR-132 multi-EMA / ADR-135 Level Blend VP / ADR-141 SAR AF params + `#/notes` scenario snapshots via `chart_notes.py` / ADR-142 VWAP ±2σ bands / ADR-143 multi StochasticFast / ADR-144 compact scale tags, hover resize, bound arrows; `#/border` alias; ADR-148 drops the MBP-10 ladder and `/api/book/*`); External: `#/external` (ADR-058/062/063/064/096) |
| `src/common/` | config, db, vm, oci, repro, **llm/**, config_paths | common | **KEEP** | LLM stack consolidated under `common/llm/` |
| `src/common/llm/` | factory, OpenAI client, Anthropic client, CostMeter | common | **KEEP** | Canonical agent LLM substrate |
| `src/common/llm_factory.py` | shim → `common.llm` | common | **KEEP** (shim) | Backward-compatible re-export |
| `src/research_director/llm.py`, `cost.py` | shims → `common.llm` | agents | **KEEP** (shim) | Do not grow new logic here |
| `src/scoping/anthropic_llm.py` | shim → `common.llm.anthropic_client` | agents | **KEEP** (shim) | Anthropic is not scoping-specific |
| `src/paper/` | Live paper limit+OCO on Databento TBBO | paper | **KEEP** | engine/stream/signals/ledger; not bar-backtest |
| `src/agents/common/` | Shared RAG/critic/LLM/tools/cost | agents | **MISSING** | P1 consolidation target |

### 2.1 Research Director internals (as-built vs docs)

| Path | Status | Disposition |
|------|--------|-------------|
| `orchestrator.py` | Present — diagnose→cross_review→synthesis→validate→handoff | **KEEP** — living doc must match |
| `director.py`, `run.py`, `governor.py`, `journal.py`, `evaluator.py` | Present | **KEEP** |
| `rag_memory.py`, `memory.py`, `tools.py`, `registry.py`, `skills/` | Present | **MERGE path** with scoping RAG in P1 |
| `agent.py`, `conductor.py`, `executor.py`, `policy_engine.py`, `registrar.py`, `schemas.py` | **Deleted** in working tree (git `D`) | Docs still cite them — **ARCHIVE narrative only**; do not resurrect without ADR |
| `config/experiment_spec.schema.json` | **Deleted** | Replaced by `model_improvement_plan.schema.json` |

**Import SSOT (done 2026-08-05):** RD uses `from src.research_director…`; entry `python -m src.research_director.run`; systemd `PYTHONPATH=/opt/vtech`.

---

## 3. `config/`

| Path | Role | Disposition |
|------|------|-------------|
| `objectives/*.yaml` + `objective.schema.json` | Trade objectives SSOT | **KEEP** |
| `external_models/*.yaml` | External model factory cards (`psar`, `ema20`, `zigzag`, `timed_psar`; hidden `psar_ema20`) | **KEEP** | ADR-058 / ADR-062 / ADR-063 / ADR-064 / ADR-065 / ADR-096; operator overlays in `ops/webui/external_models/` |
| `agents/*.md`, `*_duties.yaml` | Personas + duties | **KEEP** (personas only) |
| `agents/*.png` | Accidental screenshot | **NOISE** → `docs/archive/scratch/` |
| `schemas/*.schema.json` | JSON schemas (incl. `custom_test.schema.json`) | **MOVED** (done) |
| `specs/*` | Run/candidate YAMLs | **MOVED** (done); loaders resolve via `config_paths` |
| `runtime/*` | RD/scoping/meta/oci_sync | **MOVED** (done) |
| `oci_sync.conf` | OCI sync manifest | **MOVED** → `runtime/` |

Resolvers: `src.common.config_paths` (`schema_path` / `runtime_path` / `resolve_user_path`).

---

## 4. `docs/`

| Path | Role | Disposition |
|------|------|-------------|
| `architecture/*.md` | Living present-state | **KEEP** — rewrite stale sections (SYSTEM layout, RD) |
| `architecture/WORKFLOW.md` | Single workflow map | **ADDED** (P0) |
| `architecture/TREE_INVENTORY.md` | This inventory | **ADDED** (P0) |
| `decisions/ADR-*` | Why (immutable) | **KEEP** append-only |
| `roadmap/ROADMAP.md` | Future only | **KEEP** — strip archaeology to CHANGELOG over time |
| `roadmap/BACKLOG.md` | Deferred | **KEEP→** refresh stale rows (e.g. MLflow) |
| `roadmap/PHASE-*.md` | Phase build plans | **KEEP** as historical plans; mark status |
| `changelog/` | Past | **KEEP** |
| `reference/` | Catalogs | **KEEP** |
| `operations/` | Runbooks | **KEEP** |
| `archive/` | Old architecture snapshots | **KEEP** |
| Root `Screenshot*`, `chrome_*.png` | Noise | **NOISE** → `docs/archive/scratch/` |
| `plan/`, `doc-gen/` | Empty/low value | **KEEP** or fold later |

---

## 5. `scripts/` (plane tags) — **MOVED 2026-08-05**

Grouped under `scripts/{data,features,search,agents,serving,ops,gpu}/` — see `scripts/README.md`.
Legacy absolute paths in ops docs may still say `scripts/foo.py`; prefer the plane subdir.

---

## 6. `tests/`, `systemd/`, `deploy/`, `archive/`, `research_cases/`

| Path | Disposition | Notes |
|------|-------------|-------|
| `tests/test_*.py` | **KEEP** | Add tests when import/layout moves land |
| `systemd/vtech-*.service|timer` | **KEEP** | CME start/stop timers for live-streamer (ADR-060); MemoryMax=% of host RAM |
| `deploy/gpu/*` | **KEEP** | Bootstrap + idle shutdown matter for P4 |
| `deploy/ninjatrader/*` | **KEEP** | Manually installed Windows-side strategy source; SIM default |
| `archive/legacy_research_director/` | **KEEP** | Superseded RD pieces |
| `archive/deprecated_personas/` | **KEEP** | |
| `research_cases/papers/` | **KEEP** | RAG tier source |

---

## 7. Missing pieces that block G2/G3 (not structure noise)

| Missing | Goal | Priority |
|---------|------|----------|
| Shared agentic package (RAG/critic/LLM/tools) | G2 | P1 |
| Single import convention enforced | G4 | P0 |
| Living RD doc = orchestrator topology | G4 | P0 |
| Policy serving bundle (primary+meta+barriers) | G3 | P2 |
| `src/paper/` live limit+OCO on TBBO stream/replay | G3 | P2 engine **done** |
| Incremental live features (optional after paper) | G3 | P2+ |
| Storage capacity policy | ops | P5 |

---

## 8. Execution order for dispositions

1. Docs truth + noise relocation + import SSOT — **done**  
2. LLM consolidation under `src/common/llm/` — **done** (plane shims retained)  
3. `config/{schemas,specs,runtime}` + loaders — **done**  
4. `scripts/<plane>/` + Makefile — **done**  
5. Optional later: remove LLM shims after a quiet period; physical `src/agents/` nest stays **P1**  
6. Residual: ROADMAP Phase-17 archaeology trim (incremental)

No mass delete of domain code. No paid downloads.


## 2026-10-06 — normal live option definitions (ADR-156)

| Path | Role | Disposition |
|------|------|-------------|
| `src/acquisition/definition_acquisition.py` | Platform: vendor terms alongside subscriptions | **KEEP**, active service companion |
| `src/acquisition/live_streamer.py` | Platform: live subscription, mapping, capture and calculation | **KEEP**, vendor-only terms |
| `src/acquisition/rotating_stream.py` | Platform: daily DBN archive with mapping context | **KEEP**, retained mappings replayed |
| `tests/test_definition_acquisition.py` | Isolated acquisition lifecycle gate | **KEEP** |
| `archive/code_snapshots/20261006_normal_subscription/` | Preserved pre-edit source/tests/docs | **KEEP**, no original path removed |
| `docs/operations/NORMAL_OPTION_DEFINITION_LIFECYCLE.md` | Current lifecycle and deployment evidence | **KEEP** |

## 2026-10-06 — capture-gap recovery (ADR-157)

| Path | Responsibility | Disposition |
| --- | --- | --- |
| `src/acquisition/gap_recovery.py` | Exact-cost immutable raw recovery | KEEP, bounded recovery entry point |
| `src/ingestion/gap_recovery.py` | Payload occurrence reconciliation | KEEP, append-only adapter |
| `scripts/data/recover_databento_gaps.py` | Recovery execution and receipts | KEEP |
| `scripts/data/rebuild_gap_outputs.py` | Candidate rebuild and preserved publication | KEEP |
| `data/sources/raw/backfill/gaps_20261002_20261006/` | Vendor archives and execution receipts | External data, preserve |
| `archive/code_snapshots/20261006_gap_recovery/` | Originals preserved before authorized edits | KEEP |

The recovery inventory also includes `scripts/data/validate_gap_recovery.py` and four explicit acquisition-gate test modules (`test_gap_recovery_acquisition`, `test_gap_recovery_ingestion`, `test_gap_recovery_validation`, `test_gap_rebuild_outputs`). Their generated run/candidate/preserved artifacts are external data under `data/tmp/gap_outputs_20261006_*`, resolved through the existing data-plane paths; no preexisting source path is moved or removed.

Completed downstream publication evidence is retained under `data/tmp/gap_outputs_20261006_options_publication_retry/`: original DB/file snapshots, hourly commit receipts, independent canonical-file validation snapshots and exact/normalized contract receipts. Both feature gate attempts remain under their original `data/tmp/` paths; no feature Parquet was generated. All artifacts are preserved external data.
