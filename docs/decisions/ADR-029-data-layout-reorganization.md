# ADR-029: Runtime Data Layout Reorganization (`VTECH_DATA_DIR`)

## Status
Accepted — design 2026-08-05. **Migration not yet executed.** Inventory and target:
[DATA_LAYOUT.md](../architecture/DATA_LAYOUT.md). Part of Architecture Consolidation (P0 / ADR-028).

## Date
2026-08-05

---

## Context

The runtime root `/opt/vtech/data` (~64G+ including swap) grew phase-by-phase:

- Flat `parquet*` caches alongside a Phase-18 hive `features/` that **duplicates** most of the same days.
- Model history in **three** places: `checkpoints/`, `model_staging/`, `training_runs/` (plus empty `model_registry/`).
- Ingest job resume JSON co-located with Keras checkpoints under `checkpoints/`.
- Active research paths (`search/`, `evaluation/`, `scoping/`) beside legacy `research_runs/` and `hpo/`.
- Infra (`swapfile`, optional `postgresql/`) mixed with ML artifacts.

Code (structure-p0 branch) actively reads/writes a **subset** of these paths; several large trees are cold copies or publish mirrors. Git correctly ignores `data/*`; worktrees must not each carry multi‑GB staging — they share `VTECH_DATA_DIR`.

Without a target layout, P0 “architecture cleanup” stops at source code while the operational surface remains convoluted and disk pressure recurs.

---

## Decision

1. **Layout contract in repo; bytes on shared volume**  
   Document and implement a plane-oriented tree under `$VTECH_DATA_DIR`:  
   `market/` · `experiments/` · `models/` · `ops/` · `_legacy/`.  
   Full map: [DATA_LAYOUT.md](../architecture/DATA_LAYOUT.md).

2. **Canonical local feature cache = flat 5s matrix** (today `parquet/`)  
   Hive `features/` is a publish/export layout (OCI gold), not a second source of truth on disk. Prefer one local cache + OCI mirror.

3. **Serve lifecycle = staging → registry (+ training_runs for finalize)**  
   Do not grow new models under `checkpoints/`. Split ingest job state out of `checkpoints/`.

4. **Never bulk-delete `raw/`** without explicit operator approval after ingest safety confirmation (AGENTS.md).

5. **Migration via path helper + temporary symlinks**, then `mv`, then remove aliases — not rsync of data into git worktrees.

6. **Empty `model_registry/` is a process gap**, not a layout gap — promote the current econ champion after path cutover so Live prefers registry.

---

## Consequences

**Easier**

- Operators and agents map folders to workflow planes.
- Safer disk reclamation (legacy box + tmp TTL) without touching raw.
- Multi-worktree development shares one data root.

**Harder**

- One coordinated change: `data_paths`, oci_sync, GPU parquet dest, FE/search/finalize/live.
- Temporary dual paths during migration.

**Out of scope for first migrate PR**

- Compressing TimescaleDB or changing raw DBN layout on object storage.
- Deleting raw archives.

---

## Implementation checklist (when executing)

- [ ] `src/common/data_paths.py` — logical keys → paths; env override  
- [ ] Symlink or dual-write window for `parquet`, `model_staging`, `search`  
- [ ] `mv` trees per DATA_LAYOUT map  
- [ ] Update `oci_sync.conf`, GPU sync, docs  
- [ ] Promote staged econ champion → registry  
- [ ] Optional: remove local `features/` hive after OCI verify  
- [ ] ROADMAP: tick data reorg under Architecture Consolidation  

---

## Related

- ADR-028 structure-first code/docs  
- ADR-018 objective SSOT · ADR-024 artifact flow · Phase 18 OCI redesign (partially superseded local paths)
- [ADR-070](ADR-070-native-tf-hive-layout.md) — native TF hive under `sources/features/timeframe=`; 5s remains flat `parquet_5s/`
