# ADR-070: Native-TF feature hive (`timeframe=/symbol=/year=/month=`)

## Status
Accepted — dirs prepared 2026-08-27. **Amended 2026-08-27 (ADR-075):** 15s
writes monthly `features_15s_partN.parquet` (≤100 MB), not day files.
**Amended 2026-08-27 (ADR-079):** clause 2 ("5s stays flat") is superseded;
5s uses `timeframe=5s/` monthly parts. **Amended 2026-08-28 (ADR-085):** 1m
writes monthly `features_1m_partN.parquet`. **Amended 2026-08-28 (ADR-090):**
5m writes monthly `features_5m_partN.parquet`. **Amended 2026-08-28 (ADR-091):**
1h / 1d write monthly `features_{tf}_partN.parquet` (HTF context from
`tbbo_1h` / `tbbo_1d`). Live pin unchanged.

## Date
2026-08-27

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs native 15s / 1m / 5m / 1h / 1d feature
stores. The 5s cache is a flat day file (`parquet_5s/features_YYYY-MM-DD.parquet`).
ADR-029 kept a Phase-18 hive as export-only. Native TFs are a new product: five
resolutions, NQ root, year/month prune.

## Decision

1. Canonical native-TF cache is Hive partitions under
   `$VTECH_DATA_DIR/sources/features/` (the existing `features/` root):

   ```
   timeframe={15s|1m|5m|1h|1d}/
     symbol=NQ/
       year=YYYY/
         month=MM/
           features_{tf}_partN.parquet   # 15s (ADR-075); other TFs TBD
           # (placeholder had been features_YYYY-MM-DD.parquet)
   ```

2. **5s stays flat** `parquet_5s/`. Do not add `timeframe=5s`.
   **Superseded by [ADR-079](ADR-079-5s-hive-monthly-parts.md).**
3. Dataset readers start at `timeframe={tf}/`. Do not scan
   `sources/features/` as one hive — `parquet_5s/` and other flat caches are
   siblings.
4. Path SSOT: `src.common.data_paths.native_tf_dir` /
   `native_tf_parquet_path` / `ensure_native_tf_dirs`.
5. Month dirs for 2025-01 through 2026-12 are created empty. This is not the
   Phase-18 `legacy/features_hive` snapshot.

## Consequences

- Search remains fail-closed until the composer is wired (ADR-066).
- 15s files: [ADR-075](ADR-075-native-tf-15s-bar-features.md).
- GPU Object Storage prefix is `timeframe={tf}/` with hive-relative keys
  ([ADR-078](ADR-078-gpu-native-tf-hive-mirror.md)). Not `{pdir.name}/`.
- Live pin unchanged.

## Related

- [ADR-029](ADR-029-data-layout-reorganization.md) — original 5s-flat layout
- [ADR-079](ADR-079-5s-hive-monthly-parts.md) — 5s joins this hive
- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — objective that consumes these stores
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — candle substrate
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s 1-bar files (monthly parts)
- [ADR-085](ADR-085-native-tf-1m-features.md) — 1m files
- [ADR-090](ADR-090-native-tf-5m-features.md) — 5m files
- [ADR-091](ADR-091-native-tf-1h-1d-htf-context.md) — 1h / 1d HTF context files
