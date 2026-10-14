# ADR-079: 5s feature hive (`timeframe=5s/` monthly parts)

## Status
Accepted — implemented 2026-08-27. Live pin unchanged.

## Date
2026-08-27

## Context

The 5s research matrix lived as one parquet per session day
(`parquet_5s/features_YYYY-MM-DD.parquet`) with the timestamp on the pandas
index (pyarrow `__index_level_0__`). Native 15s already uses Hive partitions
and monthly size-split parts with a `bar_end_ts` column (ADR-070 / ADR-075).
Keeping 5s as a flat sibling made GPU prefixes, fold-cache globs, and
dataset readers special-case the 5s store.

ADR-070 clause 2 said "5s stays flat. Do not add `timeframe=5s`." That is
superseded here. The 5s *product* is unchanged: 08:00–11:00 CT session clip,
full feature catalog (~200–270 cols), not the 13-col Globex 1-bar 15s store.

## Decision

1. **Canonical 5s cache** is the same hive as native TF:

   ```
   $VTECH_DATA_DIR/sources/features/timeframe=5s/symbol=NQ/year=YYYY/month=MM/
       features_5s_partN.parquet
   ```

   `year`/`month` are UTC of `bar_end_ts`. Parts use the 15s writer
   (`write_month_parts`): append until ~100 MB, then `partN+1`.

2. **Datetime column** is `bar_end_ts` (UTC, first column, `index=False`).
   Loaders expose a DatetimeIndex named `ts` for existing training code.

3. **Duplicates.** Same `bar_end_ts` in more than one source file (live
   next-session spill onto yesterday's day file, rebuilt days): keep the
   row from the source file with the latest mtime. Ties: later dated file,
   then later row in that file.

4. **Old day files stay** at `parquet_5s/` until the operator verifies the
   hive and deletes them. Migration does not delete.
   **Amended by [ADR-081](ADR-081-5s-hive-cutover.md):** live
   `extend_date` / `build_date` / `patch_date` write hive parts only.

5. **Readers.** `load_feature_frame("5s")` loads hive parts.
   **Amended by [ADR-081](ADR-081-5s-hive-cutover.md):** no
   `parquet_5s/` fallback; GPU/OCI prefix is `timeframe=5s/` only.

6. **Path SSOT.** `native_tf_dir("5s")` is valid. `HIVE_TIMEFRAMES` =
   `("5s",) + NATIVE_TF_TIMEFRAMES`. `NATIVE_TF_TIMEFRAMES` stays the
   1-bar native product (`15s`/`1m`/`5m`/`1h`/`1d`).

## Consequences

- Schema union across days: monthly parts outer-concat missing columns
  (feature catalog grew over 2025–2026).
- `oof_*.parquet` under `parquet_5s/` is not a feature day file and is
  not migrated.
- Search fold cache fingerprints hive relative paths once
  `parquet_dir_for_store("5s")` points at the hive.
- Live pin unchanged.

## Related

- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys; clause 2 superseded
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s monthly parts / `bar_end_ts`
- [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md) — GPU hive-root prefix
- [ADR-029](ADR-029-data-layout-reorganization.md) — data-root layout
- Builder: `scripts/features/migrate_parquet_5s_hive.py`
