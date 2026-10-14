# ADR-038: Stochastic `%D`-only features + Full(60, slowing=1, D=10)

## Status
Accepted — implemented 2026-08-20.

## Date
2026-08-20

## Context

v5.5/v5.6 stored both Fast %K and Fast %D for four 1-minute studies, named
`so_{kPeriod}_{k|d}` (e.g. `so_9_k` / `so_9_d`). That hid the D length, stored
a %K line that %D already contains as its input, and computed the slowest
study as Fast(60, Wilder 20) even though the module header described TOS
`StochasticFull(80, 20, 60, 10, …, 1, EXPONENTIAL)`.

ThinkorSwim:

- **StochasticFast**: FastK = raw Lane %K; FastD = MA(FastK, dPeriod).
- **StochasticFull**: FullK = MA(raw %K, slowingPeriod); FullD = MA(FullK, dPeriod).
  Slowing = 1 makes FullK identical to FastK; FullD is then a second MA (here
  exponential, length 10), not a Wilder FastD.

## Decision

1. **Emit only %D** (plus `so_sync_score`, which is already a function of the
   four %D series). %K is computed inside the module and discarded.
2. **Studies** (1-minute OHLC, Wilder FastD / exponential FullD):
   - `StochasticFast(80, 20, 9, 3, HIGH, LOW, CLOSE, WILDERS, On FastD)`
   - `StochasticFast(80, 20, 14, 3, HIGH, LOW, CLOSE, WILDERS, On FastD)`
   - `StochasticFast(80, 20, 40, 4, HIGH, LOW, CLOSE, WILDERS, On FastD)`
   - `StochasticFull(80, 20, 60, 1, HIGH, LOW, CLOSE, 10, EXPONENTIAL, On FullD)`
     (kPeriod=60, slowing=1, dPeriod=10).
3. **Names**: `so_{fast|full}_{d|k}_{kPeriod}_{dPeriod}`. All four are the `d`
   line, so `so_fast_d_9_3`, `so_fast_d_14_3`, `so_fast_d_40_4`,
   `so_full_d_60_10`. Slowing is not in the name (Full slowing is 1).
4. Drop pre-v5.7 columns (`so_9_k`, `so_9_d`, … `so_60_d` and the older
   `so_fast9_*` / `so_full60_*` names) on parquet patch.

## Consequences

- `so_` 9 → 5 columns; catalog 221 → 217 (v5.7).
- Full(60) is no longer Fast Wilder-20. With slowing=1 it is EMA(raw %K, 10).
- Existing parquets keep old names/values until `patch_parquets --modules ext`
  or a rebuild.
- Serving bundles trained on `so_9_d` / `so_60_d` need a new feature view /
  retrain; schema identity will fail on the renamed columns.

## Related

- `src/features/stochastic.py`, `docs/reference/FEATURE_CATALOG.md`
- [ADR-037](ADR-037-size-weighted-tl-zscore-diff.md) — prior v5.6 catalog bump
