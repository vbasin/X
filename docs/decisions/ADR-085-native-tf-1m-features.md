# ADR-085: Native 1-minute feature parquet

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs native 1m features after the 15s 1-bar
hive (ADR-075). Hive dirs exist (ADR-070) as empty
`timeframe=1m/symbol=NQ/year=/month=` partitions. The 5s cache is an
08:00–11:00 CT clip and ffill-holds 1-minute modules (`so_`, 1m `cs_*`,
`ps_1m_*`) onto 5s; a native 1m store is one row per closed minute.

Requested product: entire 5s-catalog groups `so_` `cs_` `ps_` `vw_`
`rv_` `vp_` `ms_`. Options `iv_*` is **not** on this hive (operator will
patch selected IV later). 71 columns.

## Decision

1. **Closed-bar timestamp.** Datetime column is `bar_end_ts` = left-closed
   `[T, T+1m)` labeled `T+1m` (UTC), same convention as native 15s.

2. **Full TBBO window.** Lead-month NQ, every Globex session that has TBBO
   (`[T-1 17:00, T 17:00)` America/Chicago). Minutes with no prints are
   omitted. Range is `min/max(bucket)` from `databento.tbbo_15s` (same span
   as ADR-075).

3. **Column names** stay the 5s catalog names. `cs_open` / `cs_close` are
   the **1-minute** candle. Rolling suffixes that the 5s catalog documents
   as minutes (`rv_realized_vol_60` = 5 min) are wall-clock on this grid.
   `so_*`, 1m candle-shape `cs_*`, and `ps_1m_*` reuse the existing 1-minute
   modules (same values as the 5s parquet at each minute close).

4. **No options IV.** The 1m hive does not load `databento.bbo_1s` or
   run `options_surface`. Selected `iv_*` columns can be patched later.

5. **Monthly size-split parts**:

   ```
   timeframe=1m/symbol=NQ/year=YYYY/month=MM/features_1m_partN.parquet
   ```

   Append until ~100 MB. Amends ADR-070 for **1m**.

6. **Catalog.** [FEATURE_CATALOG_1m.md](../reference/FEATURE_CATALOG_1m.md).
   Custom Test store `1m` / view `native_1m`.

## Consequences

- Automated search `tf_oof_meta` is still fail-closed; OOF-only composer:
  [ADR-092](ADR-092-tf-oof-meta-composer.md). Custom Test can pick the 1m
  store with meta off.
- 5s parquet schema is unchanged.
- GPU Object Storage prefix is `timeframe=1m/` (ADR-078). Push after the
  hive is written.
- Live pin unchanged.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — consumer objective
- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s 1-bar files
- [ADR-038](ADR-038-so-d-only-full-60.md) / [ADR-055](ADR-055-parabolic-sar-closed-bars.md)
- Builder: `scripts/features/build_native_tf_1m.py`
- [ADR-090](ADR-090-native-tf-5m-features.md) — native 5m sibling store
