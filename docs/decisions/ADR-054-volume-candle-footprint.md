# ADR-054: Volume candle footprint (`vf_`) — concentrations and gaps of a closed candle

## Status
Superseded — column set replaced by [ADR-056](ADR-056-volume-footprint-wick-body.md) (2026-08-22). Closed-bar TBBO module, `vf_` prefix, and `viz_payload` remain.

## Date
2026-08-21

## Context

Session volume profile (`vp_`) answers *where volume sat today*. Sub-bar
dynamics (`sd_`) answers *when volume arrived inside a 5s bar*. Neither
describes **where volume clustered inside a completed 1-minute or 5-minute
candle**, nor the **empty ticks** (air pockets / jumps) between that candle's
high and low.

That per-candle ladder is the usual *volume footprint* / cluster chart: a
closed bar's buy/sell volume at each NQ tick, plus the high-volume nodes and
gaps a trader would draw on the candle. The Border Visualizer (`#/border`)
needs that description to overlay 1-min and 5-min candles; the feature matrix
needs the same description as compact scalars so Custom Test / search can use
it without storing a variable-length ladder in Parquet.

A forming candle's ladder is not known until the bucket closes (ADR-041).

## Decision

1. **Module** `src/features/volume_footprint.py`, prefix `vf_`. Input is TBBO
   prints (`ts_event`, `price`, `size`, `side`). Tick size 0.25. `side` matches
   order-flow: `B` = buy aggressor, `A` = sell aggressor.

2. **Closed-bar buckets.** Every timeframe is left-closed, right-labeled:
   prints in `[T, T+tf)` become visible at `T+tf`, then stale-forward on the 5s
   grid. Default timeframes: **1-min** and **5-min**. Adding 15-min later is the
   same `compute(..., timeframes=...)` call.

3. **Per closed candle:**
   - Dense tick ladder from traded low to high (zeros on untraded ticks).
   - **Gap** = contiguous zero-volume run in that range.
   - **Concentration / HVN** = contiguous ticks with volume ≥ the mean of
     nonzero ticks (merge adjacent). If none, the POC tick is the sole node.
   - **POC** = max-volume tick. **Poor high** = no sell volume at the high;
     **poor low** = no buy volume at the low (unfinished auction).

4. **Parquet scalars (19 per timeframe, 38 total).** Ranked top-2 HVNs
   (`loc`, `width`, `vol` fraction) and top-2 gaps (`loc`, `width`), plus
   summary: `poc_loc`, `poc_vol_frac`, `hvn_count`, `gap_count`, `gap_frac`,
   `delta_frac`, `buy_sell_loc_diff`, `poor_high`, `poor_low`. `loc`/`width`
   are in `[0, 1]` of the candle range (0 = low). Missing ranks are NaN.

5. **Visualization contract.** `CandleFootprint.to_viz()` / `viz_payload()`
   return the full ladder plus concentrations and gaps. That is the overlay
   payload for 1-min / 5-min charts (Border Visualizer next). Not stored in
   the 5s parquet.

6. **Views / serving.** Named view `candle_footprint` = `["vf_"]`. Existing
   views are unchanged so champions keep their column set. Extra parquet
   columns are ignored at serve (`ignore_extra_at_serve=True`).

7. **Retroactive.** `patch_parquets.py --modules vf` (TBBO `price/size/side`
   only). Live engine path computes `vf_` on the warm TBBO buffer.

## Consequences

- 5s rows inside a 1-min (5-min) bucket see the **previous** closed footprint,
  ffilled, until that bucket closes.
- `vf_5m_*` is NaN until the first 5-min close in the day file.
- Historical parquets lack `vf_*` until `--modules vf` or a rebuild.
- Visualization can be added without a second derivation: call
  `viz_payload(tbbo, timeframe="1min"|"5min")` on the same prints.

## Related

- [ADR-056](ADR-056-volume-footprint-wick-body.md) — wick/body volume + 1m POC EMA (current `vf_` set)
- [ADR-041](ADR-041-closed-bar-slower-features.md) — closed-bar visibility
- [ADR-055](ADR-055-parabolic-sar-closed-bars.md) — same variable-TF closed-bar pattern (`ps_`)
- `src/features/volume_profile.py` — session (not per-candle) POC / VA
- `src/features/candle_structure.py` — OHLC shape; no volume-at-price
