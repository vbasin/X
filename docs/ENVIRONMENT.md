# ENVIRONMENT.md — Reproducibility, Logging, Fast Verification

**Status: CANONICAL, CURRENT.**

> **Public export:** start with `pip install -e ".[dev]"`, then `vtech-demo
> --output .demo-output/run` and `make test-publication`. No environment file or
> external service is needed for that path. The historical operator environment
> below describes optional platform integrations, not public prerequisites.

## 1. Clone → environment → validate → test → experiment

```
1. Python >=3.10 (pyproject.toml requires-python); venv at /opt/vtech/venv
2. cd /opt/vtech && source venv/bin/activate && set -a && source .env && set +a
3. Dependencies: pyproject.toml (+ optional extras: training, serving, agents)
   pinned lock: requirements.lock (431 lines, per CHANGELOG Phase 17D)
4. Validate config (read-only, no DB): python -m src.experiment.cli validate-config
5. Fast verification (no DB, ~seconds): python -m pytest tests/architecture/ -q
6. Broader verification: select explicit component test paths; do not run the full suite.
7. Run an experiment: see docs/RESEARCH_WORKFLOW.md's entry-point table (Track B/C
   scripts) — no unified canonical experiment command exists yet, see docs/EXPERIMENTS.md
```

## 2. Environment variables and secrets

- `.env` (repo root, never committed) — DB credentials, `MLFLOW_S3_ENDPOINT_URL`, Databento API key. Never print its values; never modify without asking (`AGENTS.md` §4).
- `VTECH_DATA_DIR` — runtime data root, default `/opt/vtech/data` (`docs/architecture/DATA_LAYOUT.md` §1). Every worktree should set this to the same shared volume, not a per-branch copy.
- `VTECH_BBO_PRIMARY_ONLY` — documented OOM workaround for high-volume dates (`docs/architecture/DATA_LAYOUT.md` history); not a normal-path variable.
- `MLFLOW_TRACKING_URI` — set on the GPU instance to point at the CPU instance's MLflow server (`config.yaml` comment).

## 3. Determinism / seeds

- Track B: `config.yaml`'s `training.seed: 42` — but note this block is confirmed **dead** for the actual live trainer (`docs/CONFIG_AUDIT.md`); the Custom Test Lab catalog (`src/models/catalog.py`) sets seeds per-model-family instead, not centrally.
- Track C: `random_state=42` hardcoded in `scripts/features/eval_move_model.py`/`eval_target_model.py`'s XGBoost construction.
- `src/experiment/manifest.py`'s `RunManifest.seed` field is designed to record whatever seeds a run actually used, per-library, once wired — see `docs/EXPERIMENTS.md` §1.

## 4. External service dependencies

| Service | Role | Reproducibility note |
|---|---|---|
| PostgreSQL 18.3 + TimescaleDB 2.26.1 | raw tick/quote storage | `docs/architecture/DATABASE_SCHEMA.md` (schema current; row-count snapshot stale) |
| MLflow (`vtech-mlflow.service`) | experiment tracking, Track B | `config.yaml`'s `mlflow.tracking_uri`/`artifact_root` |
| Databento | market data vendor | cost-gated — see `AGENTS.md` §5; `scripts/data/databento_cost_check.py` |
| systemd units (`systemd/*.service`) | live ingestion, WebUI, MLflow, (inactive) paper trading | current live/inactive state: `project_manifest.yaml`'s `tracks` section, re-verify with `systemctl` rather than trusting a stale doc |

## 5. Hardware assumptions

VM specs vary — the operator resizes vCPU/RAM. Never hardcode worker counts or quote specs from a doc; always re-check live:
```
python3 -c "from src.common.vm_config import get_vm_config; print(get_vm_config())"
```
See `AGENTS.md` §8-9 and `docs/architecture/SYSTEM.md` for the full standing rule.

## 6. Logging expectations

- **Decision-relevant results must be persisted as a durable artifact, not left in console output.** Track C's persisted book is `trade_ledger.parquet` / `economic_metrics.json` from `eval_target_model.py` through `src/evaluation/economic.py`. The console R print beside that ledger is not the ledger. Track B per-candidate metrics live on the run under `research/runs/<objective>/<run_ts>/`. `research/custom_test/` is gone.
- New canonical code (`src/*`) should not log to stdout for anything a future comparison/report might need — return structured data (dataclasses, as `economic.py`/`manifest.py` already do) that a caller can choose to print, persist, or both.
- Progress indicators for long operations (>5s) follow `AGENTS.md` §10's existing convention — unrelated to *result* logging, which is the concern here.

## 7. Fast agent-safe verification path

`python -m pytest tests/architecture/ -q` — the one command a fresh agent can run immediately, with zero setup beyond an activated venv, zero database dependency, and a runtime under a second (39+ tests as of this writing — see `docs/MIGRATION_MANIFEST.md` for the current count). This is the verification loop for anything touching `src/*`. It does **not** verify Track B/C's own existing pipelines — those have their own test files under `tests/` (e.g. `tests/test_entry_state.py` for Track C's feature causality, per this session's own prior working pattern) which may need real fixtures or take longer; run the specific relevant test file for a Track B/C change, not the full suite, per this project's own established practice (see `AGENTS.md` and this session's history of running only `tests/test_entry_state.py` for cascade changes, not the whole suite).
