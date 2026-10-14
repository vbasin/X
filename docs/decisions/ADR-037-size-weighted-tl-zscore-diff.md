# ADR-037: Size-weighted `tl_trade_location` / `tl_aggressive_pct` + pair z-score / short−long

## Status
Accepted — implemented 2026-08-20.

## Date
2026-08-20

## Context

v5.5 `tl_loc_ma_*` / `tl_aggr_*` used print-weighted tick windows. That was
aligned to bar-window means of the 5s series (ADR-001 / `bar_windows.py`): each
5s bar counts once. Two remaining gaps:

1. **Bar value** was still an unweighted mean of prints, so a 1-lot cancelled a
   10-lot (location) or a 1-lot inside print cancelled a 10-lot sweep (aggression).
2. Nested SMAs of that series left XGBoost without a short-vs-long encoding.
   1m and 60m MAs were unused extra collinear smooths.

## Decision

16 `tl_` columns: **4 levels × 2 families**, plus **2 pair z-scores and 2 pair
short−longs per family**.

1. **Both 5s bar values are size-weighted:**
   - `tl_trade_location = sum(size × loc) / sum(size)`
     with `loc = (price − mid) / half_spread`
   - `tl_aggressive_pct = sum(size × sweep) / sum(size)`
     with `sweep = 1` iff `price > ask` or `price < bid`
   - Missing `size` is treated as 1.0.
2. **Four levels per family** (the 5s bar plus three equal-weight MAs):
   5s, 30s (`bars(30)` = 6), 5m (`bars_m(5)` = 60), 30m (`bars_m(30)` = 360).
   **1m and 60m MAs are removed.** Location MA names stay `tl_loc_ma_{w}`;
   aggression MA names stay `tl_aggr_{w}`.
3. **Pair stats only** (no `series − ma` increment z-score / diff columns),
   for both location and aggression, pairs 30s vs 5m and 5m vs 30m:
   - `tl_{loc|aggr}_diff_{short}_{long} = ma_short − ma_long`
   - `tl_{loc|aggr}_zscore_{short}_{long} = (ma_short − ma_long) / std_long`
     (`std_long` is full-window `rolling(long).std()` of the 5s series;
     zero std → NaN)

## Consequences

- A large print moves the 5s location and sweep fraction (and everything derived)
  more than a 1-lot.
- Pair z-scores are NaN until the long window has elapsed (60 / 360 bars @ 5s).
- Existing parquets hold print-mean values and 1m/60m columns until
  `patch_parquets --modules ext` or a rebuild.
- `tl_` column count 12 → 16.

## Related

- [ADR-001](ADR-001-5s-timestep.md) — 5s bars; windows in real time via `bar_windows`
- `src/features/trade_location.py`, `docs/reference/FEATURE_CATALOG.md`
