# ADR-056: Volume footprint wick/body volume (`vf_`) for closed 1m and 5m candles

## Status
Accepted — implemented 2026-08-22.

## Date
2026-08-22

## Context

[ADR-054](ADR-054-volume-candle-footprint.md) stored ranked HVNs and interior
volume gaps per closed 1-min / 5-min candle. That is a useful *drawing* of the
ladder, but not the volume split a candle-footprint overlay and a tree model
actually want: **how much of the bar traded in the wicks vs the body**, and
whether that volume was ask- or bid-initiated.

The operator metrics are:

* % of candle volume in the upper wick, body (open and close inclusive), and
  lower wick
* delta volume (ask − bid) in each of those three regions
* scaled delta = (region volume / candle volume) × (region delta / region volume)
* 1-min POC position in the candle range
* 5-bar EMA of that 1-min POC position (five closed one-minute bars)

HVN ranks and gap geometry are dropped from the parquet. The tick ladder stays
on `CandleFootprint` / `viz_payload` for visualization.

## Decision

1. **Same module and closed-bar rule** as ADR-054: `src/features/volume_footprint.py`,
   prefix `vf_`, TBBO prints, tick 0.25, left-closed right-labeled 1-min and
   5-min buckets, then ffill onto 5s.

2. **Regions** (tick-rounded OHLC; first print = open, last = close):
   - Body = `[min(open, close), max(open, close)]` inclusive.
   - Upper wick = ticks strictly above the body; lower wick = strictly below.
   - A close-on-high bar has no upper wick; a doji with high=low is 100% body.

3. **Ask / bid.** Ask volume = buy-aggressor size (`side=='B'`, traded at the
   ask). Bid volume = sell-aggressor size (`side=='A'`, traded at the bid).
   `delta = ask − bid`. Empty wick: vol_frac = 0, delta = 0, scaled = 0.

4. **Parquet columns (20).** Both TFs emit nine region columns
   (`upper|body|lower` × `vol_frac`, `delta`, `delta_scaled`). **1-min only:**
   `vf_1m_poc_loc` ∈ [0, 1] and `vf_1m_poc_ema_5` = `ewm(span=5, adjust=False,
   min_periods=5)` of closed 1-min `poc_loc` (computed on the 1-min series,
   then ffilled — not an EMA of the 5s repeats).

5. **Patch** `--modules vf` drops every existing `vf_*` column (ADR-054 names
   included) and writes the 20 new ones.

6. Named view `candle_footprint` = `["vf_"]` unchanged.

## Consequences

- Catalog 38 → 20 `vf_` columns (v5.11 260 → **242**).
- Historical parquets that already have ADR-054 `vf_*` must be patched or those
  old names linger next to the new ones.
- `vf_1m_poc_ema_5` is NaN until five closed 1-min bars.
- Scaled delta equals `region_delta / candle_volume` when the region has
  volume; it is 0 when the region is empty.

## Supersedes

[ADR-054](ADR-054-volume-candle-footprint.md) parquet column set (HVN / gap
ranks). Closed-bar TBBO footprint module and viz ladder remain.

## Related

- [ADR-041](ADR-041-closed-bar-slower-features.md) — closed-bar visibility
- [ADR-054](ADR-054-volume-candle-footprint.md) — prior HVN/gap encoding
- `src/features/candle_structure.py` — OHLC shape; no volume-at-price
