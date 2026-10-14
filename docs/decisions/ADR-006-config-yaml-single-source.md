# ADR-006: config.yaml as Single Source of Truth

## Status
Accepted — refined in part by [ADR-018](ADR-018-objective-doc-single-source-labels.md): objective-level params (labels/session/horizon/KPIs) are delegated to the trade objective doc; `config.yaml` references it via `objective_id` and no longer carries those values.

## Date
2026-04-20 (Phase 17 / v5 architecture)

---

## Context

Before v5, all runtime configuration was scattered across environment variables set in `.env`, `os.environ` calls throughout the codebase, and magic constants in individual module files. Examples:

- `VTECH_FEATURES_TIMESTEP` in `.env`
- `SEQUENCE_LENGTH = 128` hardcoded in `data_pipeline.py`
- `LABEL_THRESHOLD = 50` in `labels.py`
- `FORWARD_WINDOWS = [60, 120, 180, 360]` in multiple files
- `ADAPTIVE_ATR_MULT = 3.0` in `labels.py`
- `SEED = 42` in `trainer.py`

This caused config drift: changing a training parameter required editing 3–5 files. Reproducing an older training run required reconstructing the original environment. The HPO search used a different set of parameters than the trainer. Feature engine and trainer had their own incompatible representations of the same window sizes.

Additionally, secrets (`DATABENTO_API_KEY`, `PGPASSWORD`) were co-located in `.env` with non-secret operational parameters — any config change required touching the secrets file.

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Status quo (env vars only) | Zero migration | Config drift continues; hard to reproduce runs |
| **`config.yaml` + env override fallback** (chosen) | Human-readable; version-controlled; env vars still work | Requires loader code; one new file to maintain |
| MLflow / W&B for config management | Full experiment tracking | Requires daemon, external service, significant migration |
| `pyproject.toml` `[tool.vtech]` | Everything in one file | Non-standard; tooling assumptions for TOML may conflict |
| Python config module (`config.py` constants) | Type-checked | Must change source code to change config; hard to use from shell scripts |

---

## Decision

Create **`/opt/vtech/config.yaml`** as the single source of truth for all non-secret runtime configuration. Secrets (`DATABENTO_API_KEY`, `PG*`) remain in `.env` only.

Design:
1. `config.yaml` is committed to version control — config changes are tracked as git diffs
2. `src/common/config.py` `AppConfig.load()` reads YAML first, then applies `VTECH_*` env-var overrides on top (backwards compatible with existing scripts)
3. All modules import from `AppConfig` — no more `os.getenv()` with inline defaults scattered across files
4. `.env` contains only secrets (permissions: 600). Operational variables (`VTECH_BBO_PRIMARY_ONLY=1`, `VTECH_ADAPTIVE_THRESHOLD=1`) are set at invocation time, not stored in `.env`

The env-var override mechanism preserves backwards compatibility and allows per-invocation overrides for special cases (e.g. `VTECH_BBO_PRIMARY_ONLY=1 python -m src.features.engine` for OOM dates) without changing `config.yaml`.

---

## Consequences

- **+** Reproducing any training run requires only committing `config.yaml` + `requirements.lock` + git SHA (all three are now captured in `_experiments.jsonl` via `repro.py`)
- **+** `hash_dict(config)` in `repro.py` provides a stable config fingerprint for experiment comparison
- **+** Single file to review when asking "what were the parameters for this run?"
- **+** Secrets are completely separated from operational config
- **−** Legacy env vars still work (by design) — a developer could set `VTECH_FEATURES_TIMESTEP=10s` in `.env` and silently override the YAML. Document clearly: **env vars override YAML; only use env vars for per-invocation overrides, not persistent config**.
- **Constraint**: `config.yaml` changes that affect Parquet schema (e.g. changing `features.timestep`) require rebuilding the entire Parquet cache and retraining. The `feature_schema.json` in each checkpoint directory records the exact feature set used.

Implementation: `config.yaml` (repo root), `src/common/config.py` `AppConfig.load()`  
See also: `src/common/repro.py` `log_experiment()` — captures `config_hash` in every training run
