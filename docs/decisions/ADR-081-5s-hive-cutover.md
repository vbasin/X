# ADR-081: 5s research + live readers use `timeframe=5s/` only

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

Amends [ADR-079](ADR-079-5s-hive-monthly-parts.md) clauses 4–5 (day-file
live write + dual GPU `parquet/` prefix) and [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md)
(“5s GPU jobs: prefix `parquet/`”).

## Context

ADR-079 migrated 5s into the same monthly-part tree as 15s
(`timeframe=5s/symbol=NQ/year=/month=/features_5s_partN.parquet`,
`bar_end_ts`). Research, Custom Test, GPU, and OCI still defaulted to
`parquet_5s/` / `vtech_hot/parquet/` in several readers. Live
`extend_date` still appended a day file. SHAP, side-head, eval contract,
RD yaml, paper, and label builders still globbed
`features_YYYY-MM-DD.parquet`.

## Decision

1. **Canonical 5s store** is `native_tf_dir("5s")`.
   `parquet_dir_for_store` / `load_feature_frame` do not fall back to
   `parquet_5s/`. `STORE_IDS` is the store list; `HIVE_TIMEFRAMES` stays
   in `data_paths.py` for `native_tf_dir` validation only.
2. **Feature engine** `build_date` / `patch_date` / `extend_date` and
   live_runtime persist via `persist_session_day` (merge into month
   parts, latest `bar_end_ts` wins). They do not write
   `parquet_5s/features_YYYY-MM-DD.parquet`.
3. **Live, paper, SHAP, side-head, eval contract, RD, label builders**
   read the 5s store (`load_session_day` / `load_feature_frame`). SHAP
   uses `feature_store_id(objective)` so a 15s bundle loads 15s parts.
4. **GPU / OCI** drop `parquet_5s → parquet/`. Prefix is
   `timeframe={tf}/` for every store in `HIVE_TIMEFRAMES`. A leftover
   `parquet_5s` path passed as `parquet_dir` remaps to `timeframe=5s/`.
5. **Tmp/test dirs** with dated `features_YYYY-MM-DD.parquet` still load
   as day files (`DataPipeline`). Production defaults do not.
6. **Leftover `parquet_5s/` on disk is not deleted** (operator verifies,
   then deletes). Migration script remains the one-shot converter.

## Consequences

- Live extend rewrites the UTC-month part (~30–50 MiB) instead of a ~2 MB
  day file. Acceptable vs dual layout.
- GPU VMs after this deploy no longer pull `vtech_hot/parquet/`.
- Custom Test 5s hydrates the same way as 15s.
- Live pin unchanged.

## Related

- [ADR-079](ADR-079-5s-hive-monthly-parts.md) — 5s monthly parts
- [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md) — GPU hive-root prefix
- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — Custom Test store
