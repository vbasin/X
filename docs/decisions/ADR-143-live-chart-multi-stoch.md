# ADR-143: Live Chart multi StochasticFast + sync pair

## Status
Accepted — implemented 2026-09-25. Live pin unchanged.

## Date
2026-09-25

## Context

Live Chart exposed a single catalog Fast %D (9,3) plus `so_sync_score_all`.
Operators wanted EMA-style multi-instance StochasticFast oscillators (editable
K / D), while keeping the hive sync scores as separate toggles: quartet
`so_sync_score_all` and pair `so_sync_score_9_60` (Fast 9,3 + Full 60,10).

## Decision

1. **Multi Stoch Fast %D** (not a fixed `OVERLAYS` row). `#/chart` gains a
   **Stochastic Fast** group: enable checkbox, preset/Custom Add, cards with
   `k_period` + `d_period`, Apply. Specs in `localStorage` (`lc_stoch`,
   `lc_stoch_specs`). Request `stochs` JSON → `overlays.stoch_{n}`.
2. **Sync catalog rows.** Stochastic group keeps **Stoch sync (all)** and
   adds **Stoch sync (fast+slow)** (`so_sync_score_9_60`). Both use ±1 pane
   scale with a zero line.
3. **Math.** Each card is TOS StochasticFast %D via `_fast_d` (Wilder on
   Lane %K). Sync scores still come from `src.features.stochastic.compute`.
4. **Migration.** A stored `so_fast_d_9_3` overlay seeds one Stoch card (9,3)
   and is removed from `lc_overlays`.

## Consequences

- Operators can stack Fast(9,3) / (14,3) / (40,4) / custom on separate panes.
- Chart-Model / Notes may carry `stochs`.

## Related

- [ADR-038](ADR-038-stochastic-d-only-full-60-10.md) — %D-only + Full(60,10)
- [ADR-039](ADR-039-stochastic-sync-exponential-envelope.md) — sync scores
- [ADR-132](ADR-132-live-chart-multi-ema-overlay.md) — multi-instance UX
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
