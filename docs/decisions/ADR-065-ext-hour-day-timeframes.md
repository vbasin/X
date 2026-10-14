# ADR-065: EXT 1-hour and 1-day candle timeframes

## Status
Accepted — implemented 2026-08-26.

## Date
2026-08-26

## Context

[ADR-064](ADR-064-ext-trade-window-tf-units.md) listed book and per-engine
candle timeframes as `1min` / `5min` / `15min`, resampled from
`databento.tbbo_1min` with `label="left", closed="left"`. The operator
asked to add **1hour** and **1day** to the same Backtest dropdowns.

Naive pandas `resample("1D")` on the UTC index labels bars at 00:00 UTC
(19:00 CT in daylight time). The book session filter is a Chicago clock
on the bar label (`08:30 ≤ t < 14:45`), so those daily bars would never
be in session. Hourly UTC bars are fine: CT is a whole-hour offset, so
clock hours coincide.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Pandas UTC `1h` / `1D` | Matches 5m/15m resample | Daily bars miss the session window |
| B. CME Globex day 17:00–16:00 CT | Matches ETH daily charts | Label 17:00 is outside RTH; holiday dates are Chicago calendar days |
| C. Clock `1hour` + Chicago calendar `1day` labeled 08:30 CT (chosen) | Hourly matches 15m semantics; daily bars sit in session; causal delay still `src − book` | Daily OHLC is midnight–midnight CT, not the 17:00 Globex session |

## Decision

1. Canonical names: `1hour`, `1day` (aliases `1h` / `60min`, `1d`).
2. **1hour**: pandas `1h`, left-labeled UTC clock hours (same as CT hours).
3. **1day**: group 1-minute bars by **Chicago calendar date**, label at
   **08:30 CT**. OHLC includes that Chicago date's Globex + RTH minutes
   (same “resample then session-filter” pattern as 15min).
4. Causal alignment is unchanged (ADR-064): a slower bar labeled `T` is
   first visible on a 1min book at `T + src − 1min` (hour: `T+59m`;
   day: next session `08:29` CT). `tagged` pulses on that bar.
5. Load warmup from `tbbo_1min`: 10 days when any leg/book is `1hour`,
   40 days when any is `1day` (covers EMA-20 on daily), else 2 days.

Book `timeframe` is still display + ATR / PT / SL. A 1day **book** is one
bar per session (label 08:30); flatten at 14:45 has no later same-day bar,
so open trades hold into the next daily bar unless PT/SL/signal exits.

## Consequences

- Dropdowns on Backtest Parameters and each stacked engine show
  `1min`, `5min`, `15min`, `1hour`, `1day`.
- Live pin and ML backtest are unchanged.
- Extends ADR-064; does not change stacked AND consensus.

## Related

- [ADR-064](ADR-064-ext-trade-window-tf-units.md) — trade window, per-engine TF, units
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — warehouse Globex-session `tbbo_1d` (not EXT resample)
- Package: `src/external_models/timeframe.py`
