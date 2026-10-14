# ADR-063: ZigZag High/Low external model (Thinkorswim ZigZagHighLow)

## Status
Accepted — implemented 2026-08-26.

## Date
2026-08-26

## Context

The External Models plane ([ADR-058](ADR-058-external-models-pine-sar-ema.md),
[ADR-062](ADR-062-ext-model-split-and-stack.md)) already has Parabolic SAR and
20 EMA engines. The operator asked for a third engine from Thinkorswim
`ZigZagHighLow`: percentage / absolute / ATR / tick reversal inputs, a
causal uptrend/downtrend state, and **per-bar chart arrows** (green up =
uptrend, red down = downtrend).

The TOS study also plots `ZZ` with a forward `fold` / `GetValue(..., -i)`
that **repaints** the open swing (it only knows which bar was the last
extreme after later bars). That look-ahead must not be a trading signal.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Causal `state` as the book; confirmed pivots + arrows on the chart (chosen) | Matches TOS reversals without leaking future bars into fills | Open-swing ZZ vertex is tentative (same as TOS) |
| B. Use the look-ahead `ZZ` plot as the signal | Bit-identical line | Looks ahead; not tradable |
| C. Arrows only at confirmed pivots | Cleaner chart | Operator asked for an indicator on every bar |

## Decision

1. **Catalog** — `config/external_models/zigzag.yaml`, engine `zigzag`,
   kind `thinkscript_replica`. Knobs (TOS names in help text):
   `percentage_reversal` (factory 2.5), `absolute_reversal` (0),
   `atr_length` (5), `atr_reversal` (1.0), `tick_reversal` (0),
   `tick_size` (0.25, NQ). At least one of percentage / absolute / ATR /
   tick reversal must be non-zero. `absReversal` is the absolute value when
   that is set, else `tick_reversal * tick_size`. `hlPivot` is
   `percentage/100`, plus `Wilder ATR(true range)/close * atr_reversal`
   when ATR reversal ≠ 0. Wilder ATR here is the study's `atrLength`,
   independent of Backtest Parameters `atr_period` (min-ATR entry gate).

2. **Signals** — per closed bar `{-1, 0, +1}` from causal TOS `state`:
   `+1` uptrend, `-1` downtrend, `0` init/undefined. Same stack AND
   consensus as SAR / EMA. Shared book (session, min ATR, PT/SL, sides,
   costs) is unchanged.

3. **Chart** — when `zigzag` is in the Backtest stack, `POST /api/backtest/chart`
   returns `zz_trend` (every bar), confirmed `zz` pivot vertices, and
   hover fields (`zz_need`, `zz_max`, `zz_min`). The Chart tab draws a
   green up-arrow below every uptrend bar and a red down-arrow above every
   downtrend bar, plus a sparse ZZ line through confirmed pivots. Trade
   entry/exit markers stay on the candle series.

4. **No look-ahead in the book.** Confirmed ZZ vertices are written when
   the opposite trend starts (plus the open swing's current extreme for
   the line). Live pin and ML catalog are unchanged.

TOS factory 2.5% + 1×ATR(5) is hundreds of NQ points on a 1-minute chart.
That is the replica default; operators who want 1m swings set percentage
to 0 and use ATR and/or absolute points.

## Consequences

- `#/external` lists **ZigZag High/Low** next to SAR and 20 EMA.
- Stacking zigzag with SAR or EMA still ANDs sides.
- Tests: `tests/test_zigzag.py`.

## Related

- [ADR-058](ADR-058-external-models-pine-sar-ema.md) — External Models plane
- [ADR-062](ADR-062-ext-model-split-and-stack.md) — split engines + stack
- Package: `src/external_models/zigzag.py`
