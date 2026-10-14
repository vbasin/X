# ADR-096: Timed PSAR external model (one-shot session rule)

## Status
Accepted — implemented 2026-08-29.

## Date
2026-08-29

## Context

External Models Stage 0 already has continuous-signal engines: Parabolic SAR,
20 EMA, and ZigZag High/Low ([ADR-058](ADR-058-external-models-pine-sar-ema.md),
[ADR-062](ADR-062-ext-model-split-and-stack.md),
[ADR-063](ADR-063-zigzag-high-low.md)). Those emit a side on every closed bar
and reverse when the indicator flips.

The operator asked for a **small single-trade rule** with:

1. Book bar timeframe (existing `1min` / `5min` / `15min` / `1hour` / `1day`).
2. ATR length in bars (existing book `atr_period`).
3. Profit target: absolute points, × ATR, or a **multiple of the stop**.
4. Stop loss: absolute points or × ATR.
5. Entry timing: exact Chicago clock, or N bars from session open.
6. Entry **direction** from Parabolic SAR on a 1-hour or 1-day basis.

That is not a fourth continuous trail. It is one fill per session at a
scheduled bar, side snapshotted from the last **closed** HTF SAR, then the
shared book’s PT / SL / flatten.

5s / 30s book bars are out of scope. EXT candles still come from
`databento.tbbo_1min` (finest 1 minute). Live Chart already has 5s/15s for
display ([ADR-087](ADR-087-live-chart.md)); that path is not this book.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Encode as ADR-020 YAML | One backtest DSL | No HTF SAR snapshot, no ×-stop PT, no bars-from-open |
| B. Stack “clock pulse” + `psar@1day` | Reuses engines | Pulse is 0 after the trigger bar, so AND consensus **exits** the next bar |
| C. New engine + book PT unit `stop` (chosen) | Fits the catalog; one-shot hold; PT × stop is available to every EXT book | New card |

## Decision

1. **Card** `config/external_models/timed_psar.yaml` — engine `timed_psar`.
   Knobs: `entry_mode` (`clock` \| `bars_from_open`), `entry_time` (CT),
   `bars_from_open` (0 = first in-session book bar), `direction_tf`
   (`1hour` \| `1day`), SAR `start` / `increment` / `maximum`.

2. **Signals**. On the book index, 0 until the trigger; then hold the
   snapshotted HTF side (`+1` / `−1`) through the last in-session bar.
   Frame attr `once_per_session=True` so the book will not re-enter that
   Chicago date after a fill. A later HTF SAR flip does **not** reverse.

3. **Causal HTF SAR** (same delay as [ADR-064](ADR-064-ext-trade-window-tf-units.md)).
   Finest 1-minute lead is resampled to `direction_tf`, SAR is run on those
   closed bars, then aligned onto the book without look-ahead.
   - `1hour` at 08:30 CT is the 07:00–08:00 hour (the 08:00 hour is not closed).
   - `1day` is yesterday’s Chicago calendar day, first visible at 08:29 CT.

4. **Book**. Bar timeframe, ATR length, session vs trade window, sides, and
   costs stay on Backtest Parameters. Profit target gains unit `stop`
   (multiple of the resolved stop-loss distance at entry). Stop loss and min
   ATR stay `points` \| `atr`. `0` still disables that side.

5. **Fills** unchanged: market at the trigger bar’s close ± slippage; PT/SL
   first-touch on later book bars; flatten at session end. Session open for
   `bars_from_open` is **Session start**, not Globex 17:00.

6. **Chart**. When this engine is in the stack and no `psar` leg precedes it,
   the SAR overlay is the direction-timeframe trail.

7. Live pin, ML catalog, and `ps_*` features are unchanged.

## Consequences

- Operators set (1)–(4) on Backtest Parameters and (5)–(6) on the Timed PSAR
  card (`#/external`), then run as EXT Model `external:timed_psar`.
- Stacking still ANDs. A Timed PSAR hold of `+1` with a flat EMA does not
  enter. Solo is the intended use.
- 5s / 30s remain Live Chart / feature-hive timeframes, not EXT book bars.

## Related

- [ADR-058](ADR-058-external-models-pine-sar-ema.md) — External Models plane
- [ADR-062](ADR-062-ext-model-split-and-stack.md) — engines + book + stack
- [ADR-064](ADR-064-ext-trade-window-tf-units.md) — trade window, TF, PT/SL units
- [ADR-065](ADR-065-ext-hour-day-timeframes.md) — `1hour` / `1day`
- Package: `src/external_models/timed_psar.py`
- UI: `#/external` card **Timed PSAR**; `#/backtest` EXT Model
