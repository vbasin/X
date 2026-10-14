# ADR-090: Native 5-minute feature parquet

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs native 5m features after the 15s 1-bar
hive (ADR-075) and the 1m hive (ADR-085). Hive dirs exist (ADR-070) as empty
`timeframe=5m/symbol=NQ/year=/month=` partitions. The 5s cache ffill-holds
some 5-minute modules (`vf_5m_*`, `cs_atr_5m`) onto 5s; a native 5m store
is one row per closed 5-minute bar.

Requested product: entire volume-profile group; VWAP plus two new distances
to session VWAP ± 2σ bands; Parabolic SAR on the 5m bar; ZigZag High/Low
from the EXT engine (ADR-063 factory knobs); volume footprint; Wilder ATR(5)
on the 5m bar; realized vol. 53 columns.

## Decision

1. **Closed-bar timestamp.** Datetime column is `bar_end_ts` = left-closed
   `[T, T+5m)` labeled `T+5m` (UTC), same convention as native 15s / 1m.

2. **Full TBBO window.** Lead-month NQ, every Globex session that has TBBO
   (`[T-1 17:00, T 17:00)` America/Chicago). Empty 5-minute buckets are
   omitted. Range is `min/max(bucket)` from `databento.tbbo_15s` (same span
   as ADR-075 / ADR-085).

3. **Groups** (`src/features/native_tf_5m.py`):

   | Prefix | Contents |
   |--------|----------|
   | `cs_` | 5-minute OHLCV + range + Wilder ATR(5) on this bar |
   | `vp_` | Same 10 columns as 5s/1m (30m tick window + yesterday) |
   | `vw_` | Session VWAP dist / 5–10 bar z-score / 2–10 bar slope + `vw_bb_upper_dist` / `vw_bb_lower_dist` |
   | `ps_` | Pine SAR on closed 5m (`ps_5m_next_dist_atr`, `ps_5m_trend`, pierce, flip) |
   | `zz_` | Causal TOS ZigZagHighLow (`zzhl_on_closed`, factory YAML knobs) |
   | `vf_` | 5m wick/body footprint + `vf_5m_poc_loc` / `vf_5m_poc_ema_5` |
   | `rv_` | Realized vol on 5/10/30 native 5m bars |

4. **VWAP bands.** Session volume-weighted std of price around VWAP:
   `σ = sqrt(E[price²] − VWAP²)`. Upper / lower = `VWAP ± 2σ`. Distances
   are `(close − band) / VWAP`. These two columns do not exist on the 5s
   or 1m hives.

5. **ZigZag** is the EXT engine (`src/external_models/zigzag.py`), not a
   new study. Factory params, not operator-saved Backtest overlays, so a
   hive rebuild is deterministic.

6. **Monthly size-split parts**:

   ```
   timeframe=5m/symbol=NQ/year=YYYY/month=MM/features_5m_partN.parquet
   ```

   Append until ~100 MB. Amends ADR-070 for **5m**.

7. **Catalog.** [FEATURE_CATALOG_5m.md](../reference/FEATURE_CATALOG_5m.md).
   Custom Test store `5m` / view `native_5m`.

## Consequences

- Automated search `tf_oof_meta` is still fail-closed; OOF-only composer: [ADR-092](ADR-092-tf-oof-meta-composer.md)
  exists (ADR-066). 1h/1d context files: [ADR-091](ADR-091-native-tf-1h-1d-htf-context.md).
  Custom Test can pick the 5m store once files are on disk.
- 5s / 15s / 1m parquet schemas are unchanged. 5s `vf_5m_*` still omits
  candle POC (POC stays 1m-only on that hive).
- GPU Object Storage prefix is already `timeframe=5m/` (ADR-078 /
  `oci_sync.conf`). Building the hive is a **CPU** job from TBBO; GPU
  only pulls the prefix when a Custom Test / search job uses store `5m`.
- Live pin unchanged.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — consumer objective
- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) / [ADR-085](ADR-085-native-tf-1m-features.md)
- [ADR-055](ADR-055-parabolic-sar-closed-bars.md) / [ADR-056](ADR-056-volume-footprint-wick-body.md) / [ADR-063](ADR-063-zigzag-high-low.md)
- Builder: `scripts/features/build_native_tf_5m.py`
