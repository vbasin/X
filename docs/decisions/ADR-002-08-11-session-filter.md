# ADR-002: 08:00–11:00 CST Session Filter

## Status
Accepted

## Date
2025-09-01 (Phase 1 baseline)

---

## Context

NQ futures trade nearly 24 hours per day on Globex. However, the majority of volume and directional moves are concentrated in the US regular-hours session. The system must choose which portion of the trading day to include in feature computation and model training.

Options for session hours:
- Full Globex session (18:00 CST prior day → 17:00 CST current day, ~23 hours)
- RTH (Regular Trading Hours): 08:30–15:00 CST (6.5 hours)
- CME Open: 08:00–15:00 CST or narrower morning window

Constraints:
- Model target: predict 5–30 minute directional moves with >23% precision at τ=0.6
- The CME equity open at 08:30 CST (NQ follows SPX open dynamics) is the most consistently volatile and liquid period
- Overnight session (18:00–08:00) has thin liquidity and different market dynamics (Asia/Europe-driven)
- Training data volume must remain manageable on a 4-vCPU CPU trainer

---

## Options Considered

| Option | Bars/day (5s) | Pros | Cons |
|--------|-------------|------|------|
| 08:00–11:00 CST (chosen) | **1,080** | Captures open auction + first 3 hours of US session; high volume, directional moves, options activity; manageable data volume | Misses afternoon session |
| 08:00–15:00 CST | 5,040 | Full RTH coverage | 4.7× more data; afternoon is lower vol; VWAP/session dynamics harder to model |
| Full Globex | ~16,560 | Maximum data | Overnight dynamics are regime-different; options IV is near-useless overnight; 15× data volume |

Additionally, 0DTE weekly options have their most active/volatile period in the morning session. The IV and GEX features (`iv_*`, `gx_*`) are most meaningful and non-null during 08:00–11:00 CST.

---

## Decision

Use the **08:00–11:00 CST** session window exclusively.

The three-hour morning window captures the full CME equity open, the peak-volume hour (08:30–10:00), and the mid-morning continuation/reversal period. This is where the prediction target (BIG_UP / BIG_DOWN / CONSOLIDATION over 5–30 min) has the most signal relative to noise.

Excluded: `tc_block_overnight` and `tc_block_premarket` features — both would be constant 0 under this filter (removed in v3).

---

## Consequences

- **+** 1,080 bars/day — manageable training data for CPU
- **+** All features are computed on a homogeneous liquidity regime
- **+** `iv_*` and `gx_*` features are most meaningful in morning session
- **−** Does not generalize to afternoon or overnight prediction — separate model needed if afternoon signal is desired
- **Constraint**: The session boundaries must be consistent between feature engine and training pipeline. Both read from `features.trading_hours_start/end` in `config.yaml`.

Config key: `features.trading_hours_start: "08:00"`, `features.trading_hours_end: "11:00"`, `features.timezone: America/Chicago`
