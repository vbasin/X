# ADR-055: Closed-bar Parabolic SAR (`ps_`) — 5s pierce vs 1m trail

## Status
Accepted — implemented 2026-08-21; **column set amended 2026-08-22**.

## Date
2026-08-21 (amendment 2026-08-22)

## Context

An operator Pine strategy (`Parabolic SAR & 20EMA`) plots `SAR` and
`nextBarSAR` from Wilder's trail (start=0.02, increment=0.02, maximum=0.2)
plus EMA-slope and `ATR(8)>15` entry filters, with session flatten.

The **strategy** (position, stop-replace, 14:45 flatten, EMA/ATR gates) is
not a feature. The **SAR state machine** is not in the 218–256 col matrix:
an accelerating trail from extremes, AF as trend maturity, and
distance-to-trail. VWAP slope / 9/21 EMA cross already failed as side
features ([ADR-050](ADR-050-hypothesis-side-features.md)); this ADR only
adds the trail, not those filters.

The trail must be computed on **closed** candles of a chosen timeframe
([ADR-041](ADR-041-closed-bar-slower-features.md)). 5-second OHLC would
chatter. Engine default is **1-minute** `tbbo_1min` stale-forwarded onto
the 5s parquet; the module accepts any slower pandas offset (`5min`,
`15min`, …) without a second implementation.

`SAR` on bar T is the in-effect stop after T's close (previous
`nextBarSAR`, then same-bar flip / 1–2 bar clamp). `nextBarSAR` on T is
the stop **for T+1**, computed from T only — not T+1 prices.

## Decision

1. **Module** `src/features/psar.py`, prefix `ps_`. 1-minute SAR / nextBarSAR
   / AF / EP are computed internally (Pine start=0.02, increment=0.02,
   maximum=0.2, ATR(8) Wilder RMA with SMA seed). They are **not** parquet
   columns.

2. **Emitted columns (4):**
   - `ps_1m_next_dist_atr` — `(1m close − nextBarSAR) / ATR(8)` at 1m close,
     held 1 minute on the 5s grid.
   - `ps_1m_trend` — `+1` / `−1` at that 1m close, held 1 minute.
   - `ps_lh_dist_sar` — vs the in-effect trail (`nextBarSAR` of the last
     closed 1m): **uptrend** `5s_low − SAR`, **downtrend** `SAR − 5s_high`.
     Negative = that 5s candle pierced the trail (wick counts; close is
     not the test). Silent 5s bars stay NaN.
   - `ps_flip` — `1` on a 5s bar with `ps_lh_dist_sar < 0` and the next
     11 bars (12 × 5s = 60s). Overlapping pierces extend the hold.

3. **Closed 1-min.** Left-labeled `[T, T+1m)` is visible at `T+1m`
   (ADR-041). 5s high/low are left-labeled `[T, T+5s)` from TBBO prints.

4. **Views / serving.** Named view `psar` = `["ps_"]`. Existing views
   unchanged. Do not concat-ffill `ps_` (that would smear 5s dist/flip).

5. **Retroactive.** `patch_parquets.py --modules ps` (1-min candles + TBBO
   `ts_event, price`). Drops retired `ps_1m_sar` / `next_sar` / `af` /
   `dist_atr` / `bars` / `ps_1m_flip` if present.

## Consequences

- `ps_1m_trend` / `next_dist_atr` change only at 1m close. Intra-minute
  pierces show up in `ps_lh_dist_sar` / `ps_flip` without moving 1m trend.
- First 1m trend is NaN until the second closed 1-min bar; `next_dist_atr`
  is NaN until ATR(8) warms up.
- Live 2h lookback re-inits AF/EP at the window start.
- Historical parquets lack `ps_*` until `--modules ps` or a rebuild.

## Related

- [ADR-041](ADR-041-closed-bar-slower-features.md) — closed-bar visibility
- [ADR-050](ADR-050-hypothesis-side-features.md) — EMA/VWAP side model rejected
- [ADR-054](ADR-054-volume-candle-footprint.md) — same variable-TF closed-bar pattern
- `src/features/atr.py` — 5m/10m Wilder ATR(14), not used here
- [ADR-058](ADR-058-external-models-pine-sar-ema.md) — Pine **strategy** replica (orders, session flatten) as an External Model; this ADR remains features-only
