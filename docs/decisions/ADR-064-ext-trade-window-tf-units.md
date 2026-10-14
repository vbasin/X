# ADR-064: EXT trade window, per-engine timeframe, PT/SL units

## Status
Accepted — implemented 2026-08-26.

## Date
2026-08-26

## Context

[ADR-062](ADR-062-ext-model-split-and-stack.md) put shared knobs on Backtest
**Parameters**: session start/end gated **both** new entries and signal
management, one candle timeframe ran every stacked engine plus ATR/PT/SL, and
profit target / stop / min ATR were NQ points only.

The operator asked for:

1. A **trade start / trade end** clock window for new entries, leaving
   session start/end as the trading session (session end = last possible
   exit / flatten).
2. A **candle timeframe next to each stacked EXT engine**, so the same
   model can appear twice at different TFs. Book candle timeframe stays
   the display bar and the ATR / PT / SL bar.
3. Independent **points vs × ATR** toggles for profit target, stop loss,
   and min ATR entry (ATR = Wilder of true range over **ATR bars**).
4. Parameters grouped so the box is scannable.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Keep one session clock; PT/SL points-only; one TF (ADR-062) | Simple | Cannot cut off entries at 11:00 and hold to 14:45; cannot mix 15m SAR with 1m ATR |
| B. Trade window + per-leg TF + unit toggles (chosen) | Matches the request; causal HTF alignment | Book vs engine TF must not look ahead |
| C. Compute everything on 1m and sample indicators | Path-accurate PT/SL | 15m SAR would not match a 15m chart |

## Decision

1. **Session vs trade window** (Chicago clock, on the book)
   - `session_start` / `session_end`: bars used as the trading session.
     `session_end` force-flats remaining positions (last possible exit).
   - `trade_start` / `trade_end`: new entries only. Open trades still
     take PT/SL, signal exits, and SAR-trail stops until session end.
   - Factory both windows are 08:30–14:45 (same as ADR-062).
   - Validation: session_end > session_start; trade_end > trade_start;
     trade window ⊆ session.

2. **Candle timeframes**
   - Book `timeframe` (`1min` / `5min` / `15min` / `1hour` / `1day`;
     [ADR-065](ADR-065-ext-hour-day-timeframes.md)): display candles,
     Wilder ATR for the entry gate, and PT/SL first-touch bars.
   - Each stack leg has its own `timeframe` (default = book). The same
     engine may be stacked twice (e.g. SAR 15min AND SAR 5min).
   - Signals are computed on the engine's closed bars (same
     `label="left", closed="left"` resample as today) then aligned onto
     the book index **without look-ahead**: a 15min bar labeled `T` is
     first visible at `T + 14min` on a 1min book. `tagged` pulses on
     that availability bar; the SAR trail is then re-tested against
     book high/low when the engine TF ≠ book TF.
   - Chart overlays use the **first** stacked instance of each engine,
     on that leg's TF, aligned the same way. ZigZag arrows on a faster
     book only paint the HTF close bar (not every 1min of the hold).

3. **Units** (`points` \| `atr`)
   - Profit target / stop loss: NQ points from average entry, or that
     many × ATR(period) **at entry** (level fixed for the trade). `0` =
     off. Independent toggles.
   - Min ATR entry, **points**: skip unless ATR(period) > value (ADR-062).
   - Min ATR entry, **× ATR**: skip unless this bar's true range ≥
     value × ATR(period) (expansion bar). `0` = off.
   - Saved `profit_target_pts` / `stop_loss_pts` overlays still load
     (unit defaults to `points`).

4. **UI**
   - Parameters box sections: Session, Entries, Exits, Size & costs.
   - Each stacked row: engine, candle timeframe, snapshot.
   - ＋ may duplicate the last engine so a second timeframe can be set.

## Consequences

- A 15min SAR + 1min book no longer enters at the 15min **open** (that
  would use future 1min bars). Entries land on the 15min close.
- Live pin, ML backtest, and `ps_*` features are unchanged.
- Extends ADR-062 book knobs; does not replace stacked AND consensus.

## Related

- [ADR-062](ADR-062-ext-model-split-and-stack.md) — split engines + book
- [ADR-063](ADR-063-zigzag-high-low.md) — ZigZag High/Low engine
- Package: `src/external_models/` (`timeframe.py` align/resample)
- UI: `#/backtest` EXT Model **Backtest Parameters**
- [ADR-065](ADR-065-ext-hour-day-timeframes.md) — adds `1hour` / `1day`
