# ADR-001: 5-Second Feature Timestep

## Status
Accepted

## Date
2025-09-01 (Phase 1 baseline; confirmed Phase 10 migration)

---

## Context

The feature engine must choose a bar granularity that balances signal resolution, data volume, and training time. NQ futures are highly liquid during the 08:00–11:00 CST session, with ~100–300 trades per minute at typical activity levels.

The original system used a **10-second** timestep (v1/v2 architecture, 138 features). The session produced 540 bars/day at 10s. Models trained on 10s bars showed adequate signal on rolling averages but lost information about sub-10s microstructure changes.

Constraints:
- No GPU — training on 4-vCPU AMD EPYC. Must complete in <60 min/day.
- TimescaleDB query latency for 1 day of BBO-1s data is ~2–4 min at 5s resolution.
- Feature engine must complete before 22:00 UTC (before next trading session data arrives).

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **1s bars** | Maximum resolution; captures all BBO updates | 10,800 bars/session; training becomes impractical on CPU; DB queries too heavy |
| **5s bars** (chosen) | Good microstructure resolution; 1,080 bars/session; BBO-1s data naturally aggregates to 5s without aliasing | 2× more data than 10s; training ~2× slower than 10s |
| **10s bars** | Faster training; smaller data | Misses 5-10s microstructure; sub-bar dynamics features less meaningful |
| **1-min bars** | Fast training; simple | Loses all intraday microstructure; too coarse for short-horizon prediction |

---

## Decision

Use **5-second bars** as the fundamental timestep.

The primary reason: the prediction targets (5-min and 10-min directional moves) require enough bars-per-session that the LSTM has a meaningful context window. At 5s, a 128-bar sequence covers 10.7 minutes of context, which is the right order of magnitude for the 5–30 min prediction horizons. At 10s, the same 128-bar window covers only 21 minutes — similar, but the sub-bar dynamics features (`sd_*`) become less informative.

Additionally, the options IV solver and GEX modules already operate at 30s cadence and forward-fill to the bar grid. Going finer than 5s would not improve these features.

---

## Consequences

- **+** `sd_*` (Sub-Bar Dynamics) features are meaningful — `sd_price_impact_per_tick`, `sd_volume_front_load` etc. capture genuine intrabar activity
- **+** `bp_*` (Book Pressure) features capture 5s changes in bid/ask depth accurately
- **−** Training data volume is 2× what it would be at 10s (63K sequences on 66-day window vs ~31K)
- **−** Feature engine runtime is ~5m13s/day vs ~2.5m at 10s
- **Constraint**: All rolling window configs are expressed in bars (not time). `bar_windows.py` centralizes the `timestep → bars` conversion to ensure consistency across modules.

Config key: `features.timestep: 5s`
