# ADR-093: Joint native-TF dual-head search

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

[ADR-066](ADR-066-native-tf-dual-head-meta.md) declared `tf_oof_meta`.
[ADR-092](ADR-092-tf-oof-meta-composer.md) composed meta from **already
trained** Custom Test OOF because the fold cache is one feature store per
job. The operator asked for a full-scale end-to-end run from the 15s
parquet through the meta model. That requires the controller to train
each topology base on its own hive, then call the ADR-092 composer in
the same wave.

## Decision

1. **Candidate set = `topology.bases`**, not `eligible_model_families` ∩
   catalog. Each base is a `CatalogEntry` with `name = id`,
   `plugin_family` from YAML `family`, `params.head` from YAML (default
   `dual_binary`), `search_max_rows = None`.
2. **Per-base store.** `objective_for_store` sets
   `topology.feature_store` so the fold cache key and hive loader use
   15s / 1m / 5m independently. Labels stay `parquet_labels_15s`
   inner-joined on `ts` (1m/5m keep bars that land on the 15s TBM grid).
3. **Walk-forward** is the objective `eval` block: expanding calendar
   2025-05-01 → 2026-06-30, 4-month train / 4-month test, 3 folds,
   `purge_seconds: 600` (ADR-066).
4. **Stage 2e** after bases: `compose()` + last-block holdout
   `fit_final()`. Report-only `meta_estimator.pkl` (still not a serving
   bundle).
5. Tree bases train **locally** even if `GPU_INSTANCE_OCID` is set
   (same reason as the Custom Test XGB wave). A later LSTM 15s base
   would dispatch to GPU; this YAML’s bases are all `xgboost_ovr`.
6. Automated search on `input: native_tf` with `mode: none` stays
   fail-closed. Custom Test single-store (`mode: none`) is unchanged
   (ADR-076).
7. Live pin stays `nq-50pt-econ`. No champion gate on lg-st.

## Consequences

- `python -m src.search.controller --objective nq-50pt-econ-lg-st --no-ensemble`
  is the full-scale path.
- OOF-only CLI (`python -m src.search.tf_oof_meta`) remains for
  recomposing named Custom Test runs without retraining.
- 15s family is Attention-LSTM as of [ADR-094](ADR-094-15s-lstm-temporal-base.md).
  `20260828T212130Z` used XGB on 15s and is not the sa.md stack.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md)
- [ADR-092](ADR-092-tf-oof-meta-composer.md)
- Controller: `src/search/controller.py` (`_run_tf_oof_base_layer`,
  `_run_tf_oof_meta_layer`)
- Objective: `config/objectives/nq-50pt-econ-lg-st.yaml`
