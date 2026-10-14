# ADR-036: Custom Test lab runs persist last-fold bundles under `research/custom_test/`

## Status
Accepted — implemented 2026-08-20.

## Date
2026-08-20

## Context

ADR-024 keeps automated search **OOF-only** (GPU collect ships `oof_*.parquet` +
diagnostics; weights exist after finalize → staging → registry). ADR-035 said
Custom Test should write a last-fold serving folder next to OOF so operators can
backtest without promote. Production Custom Tests dispatch every candidate to the
GPU VM, so last-fold weights died on ephemeral GPU disk. Search-run folders
therefore looked empty of models (`20260815T223220Z`). Backtest listed only
staging/registry.

Custom Test is a **user-driven experiment**, not a catalog sweep. Persisting every
iterate/search trial's LSTM would invert ADR-024 retention.

## Decision

1. **Separate tree, objective-keyed:**
   `research/custom_test/<objective>/<run_ts>/models/<candidate_id>/`.
   Automated search stays `research/search/<objective>/<run_ts>/` (OOF-only).
2. **Per-candidate folder** holds last-fold weights (`model.ubj` / `model.keras` /
   `model.pkl`), `feature_schema.json`, `finalize_manifest.json`
   (`weights_source=last_walk_forward_fold`), `oof.parquet`, `diagnostics.json`.
3. **`meta_stack` is a third serving bundle** at
   `models/meta_stack__<bases>/` (`meta.pkl` + copied last-fold `bases/<family>/` +
   combo `oof.parquet`). Meta uses the full (no-ablation) variant of each selected
   family (ADR-035).
4. **GPU `publish_result` / `collect_result`** transfer nested `models/**` when
   `--experiment-type custom_test` (`persist_bundle` on the job payload). Iterate /
   automated search do not set the flag.
5. **Backtest** lists Custom Test bundles in the dropdown. Live still uses
   registry then staging only. Promote remains a separate human gate.
6. WebUI launch scratch stays `ops/webui/custom_tests/`; the controller copies
   spec + topology into the run dir.

## Consequences

- Custom Test → Backtest no longer requires SHAP/finalize as a save path.
- Automated search disk use unchanged.
- Serving must load standalone `model.keras` and the meta_stack fire rule
  (symmetric LR on signed evidence). Keras 3 Python-lambda `Lambda` layers
  in TCN `pooling=last` graphs are loaded via ADR-059 (`safe_mode=False`
  on first-party artifacts; new graphs use `LastTimestep`).
- GPU code bundle must be republished so `job_exec` honours `persist_bundle`.

## Related

- [ADR-024](ADR-024-unified-experiment-artifact-flow.md) — OOF-only search (unchanged for sweeps)
- [ADR-035](ADR-035-webui-custom-test.md) — Custom Test compiler (artifact path superseded here)
- `src/search/controller.py`, `src/gpu_runner/dispatch.py`, `src/search/finalize.py`
