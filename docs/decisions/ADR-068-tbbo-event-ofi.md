# ADR-068: Tick-level order-flow imbalance on the 5s parquet (`of_ofi`)

## Status
Accepted — implemented 2026-08-27.

## Date
2026-08-27

## Context

`of_delta` / CVD measure **trade** aggressor volume (TBBO `side`). They miss
quote-size changes that never print. Cont et al. (2014) order-flow imbalance
is the event contribution of each TBBO quote update, summed onto the 5s bar.

## Decision

1. Two columns on the 5s grid, still prefix `of_` (not a new family):
   - `of_ofi` — sum of tick OFI in `[T, T+5s)`
   - `of_ofi_zscore_60` — z-score vs the rolling 5-minute window
     (`bars_m(5)` = 60 bars at 5s)

2. Tick event (Cont 2014):

   ```
   e_n = I(bid_px ≥ prev_bid) * bid_sz
       − I(bid_px ≤ prev_bid) * prev_bid_sz
       − I(ask_px ≤ prev_ask) * ask_sz
       + I(ask_px ≥ prev_ask) * prev_ask_sz
   ```

   Bar = `sum(e_n)`. The first tick has no previous quote and is dropped.
   Same-price size changes count (both bid indicators true → `Δbid_sz`).

3. Distinct from `of_delta`: quote events vs prints. Engine
   `order_flow.compute` emits both. Retroactive patch:
   `python scripts/features/patch_parquets.py --modules ofi`

4. Named views that select `of_` (`microstructure`, `momentum_edge`) pick
   these up automatically. No isolated view (unlike `mbp_depth`).

## Consequences

- Catalog v5.13 246 → **v5.14 248**. `of_` 15 → 17.
- `of_ofi_zscore_60` is NaN until 60 bars (and when rolling std is 0).
- `_collect_dates` skips non-trading leftover files unless `--dates`.
- Live pin unchanged.

## Related

- `src/features/order_flow.py` (`_compute_ofi`, `OFI_COLS`)
- [FEATURE_CATALOG.md](../reference/FEATURE_CATALOG.md) §2
- [ADR-057](ADR-057-mbp10-5s-mean-depth-imbalance.md) — L2 depth (different source)
