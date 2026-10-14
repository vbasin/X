# ADR-075: Native 15s 1-bar feature parquet

## Status
Accepted — implemented 2026-08-27. Live pin unchanged.

## Date
2026-08-27

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs native 15s features. Hive dirs exist
(ADR-070) as empty `timeframe=15s/symbol=NQ/year=/month=/` partitions whose
placeholder filename was a 5s-style day file. The 5s cache is an 08:00–11:00
CT clip (ADR-002) and cannot be resampled into overnight 15s bars: OFI, trade
location, intra-bar runs, and tick counts need TBBO prints.

`databento.tbbo_15s` (ADR-067) covers Globex from 2025-04-06 22:00 UTC
through the live edge and is the date-range SSOT. It has OHLCV + last BBO
but not tick OFI, size-weighted location, or intra-bar run length.

## Decision

1. **1-bar model.** Features are statistics of one bar of width `timestep`
   (`src/features/bar_features.py`). The 15s store emits exactly:

   `of_ofi`, `cs_return`, `tl_trade_location`, `tl_aggressive_pct`,
   `ss_buy_run_max`, `ss_sell_run_max`, `cs_clv`, `sd_tick_count`,
   `sd_bid_sz_delta`, `sd_ask_sz_delta`, `sd_price_impact_per_tick`,
   `cs_range`, `cs_close`.

   No rolling z-score / MA. `ss_buy_run_max` / `ss_sell_run_max` are new
   (not on 5s): max consecutive same-side prints **inside** the bar; a run
   does not cross the bar boundary.

2. **Closed-bar timestamp.** Datetime column is `bar_end_ts` = left-closed
   bucket `[T, T+15s)` labeled `T+15s` (UTC).

3. **Full TBBO window.** Lead-month NQ, every Globex session that has TBBO
   (`[T-1 17:00, T 17:00)` America/Chicago). Not the 5s 08:00–11:00 CT clip.
   Bars with no prints are omitted (same as `tbbo_15s` GROUP BY). Range is
   `min/max(bucket)` from `databento.tbbo_15s`.

4. **Monthly size-split parts**, not day files:

   ```
   timeframe=15s/symbol=NQ/year=YYYY/month=MM/features_15s_partN.parquet
   ```

   Append until ~100 MB, then start `partN+1`. `year`/`month` are UTC of
   `bar_end_ts`. This amends the ADR-070 placeholder `features_YYYY-MM-DD.parquet`
   for **15s only**. 1m parts: [ADR-085](ADR-085-native-tf-1m-features.md).
   5m/1h/1d still unspecified until those builders land.

5. **Compute path.** Pull the eight TBBO columns for the lead contract
   (`idx_tbbo_symbol_ts` on `ts_event`) and aggregate in process. A
   server-side window+GROUP BY was slower (~1.5s vs ~1.0s per Globex
   session on 2026-08-26). Formulas match the 5s modules where they exist
   (`of_ofi`, size-weighted `tl_*`, OHLC identities). `sd_*` size deltas
   use TBBO first/last (not BBO-1s) so overnight is defined.

6. **Catalog.** [FEATURE_CATALOG_15s.md](../reference/FEATURE_CATALOG_15s.md)
   — same parseable tables as the 5s catalog. Custom Test still reads the
   5s catalog.

## Consequences

- Automated search `tf_oof_meta` is still fail-closed; OOF-only composer: [ADR-092](ADR-092-tf-oof-meta-composer.md)
  (ADR-066). The 15s feature store is now on disk.
- 5s parquet schema is unchanged (no `ss_buy_run_max`).
- Live pin unchanged.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — consumer objective
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — `tbbo_15s` span
- [ADR-068](ADR-068-tbbo-event-ofi.md) — OFI formula
- [ADR-069](ADR-069-cs-5s-ohlc.md) — `cs_return` / `cs_range` / `cs_close`
- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys; 15s parts amend day files
- [ADR-037](ADR-037-size-weighted-tl-zscore-diff.md) — `tl_*` bar identities
- Builder: `scripts/features/build_native_tf_15s.py`
- [ADR-083](ADR-083-15s-rth-cvd-ofi-slope-atr.md) — RTH CVD / cum OFI / 5-bar slope / ATR on this hive (amends 1-bar-only)
