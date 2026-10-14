# ADR-058: External Models plane — Pine SAR+EMA Stage 0

## Status
Accepted — implemented 2026-08-22. Combined card `psar_ema20` is **hidden**
after [ADR-062](ADR-062-ext-model-split-and-stack.md) (separate SAR / EMA
engines, book knobs, stacked consensus). The plane (non-ML, candles from
`databento.tbbo_1min`, operator overlays) still stands.

## Date
2026-08-22

## Context

An operator TradingView strategy (`Parabolic SAR & 20EMA`) is a **trend-follow
book**: Wilder SAR trail, dual EMA(20) slope filters, `ATR(8) > minATR`,
`strategy.exit` profit target / stop loss from average entry (defaults 50 / 10),
session 08:30–14:45 America/Chicago, flatten at 14:45. That is a different
product from `nq-50pt-*` TBM +50/−6 labels.

ADR-055 already extracted the SAR **state machine** as `ps_*` features. The
**strategy** (positions, stop-replace entries at `nextBarSAR`, EMA/ATR gates,
flatten) is not a feature and is not expressible in the ADR-020 rule DSL
(fixed target/stop, no accelerating trail, no rest orders).

The operator asked for Stage 0 of a merge path: run the Pine book as it is,
with editable knobs, then harden a parameter set and backtest it — without
folding it into the ML plugin catalog or the 50/6 objective.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| A. Encode as ADR-020 YAML | Reuses Backtest engine | DSL cannot express SAR trail, stop-entries, dual EMA loops |
| B. Train XGB to imitate entries | Same serving contract | Wrong labels; ADR-050 already rejected EMA-slope as a 50/6 side model |
| C. Separate external-model plane (chosen) | Faithful Pine replica; own UI; candles from PG; no catalog pollution | New package + Backtest payload adapter |

## Decision

1. **New plane** `src/external_models/` — not a `src.models` family, not a
   serving bundle, not an objective. Factory cards live in
   `config/external_models/*.yaml`. First card: `psar_ema20`.

2. **Data** is `databento.tbbo_1min` (lead-month NQ via the CME contract
   calendar). Not the 5s feature parquet. Optional resample to 5m/15m.

3. **Engine** `pine_sar_ema.simulate` reuses `sar_on_closed` (ADR-055) and
   ports the Pine `barstate.isconfirmed` order table. Documented fix: the
   downtrend+long flatten branch closes `LG`, not `ST` (Pine copy-paste bug).
   Market entries fill at 1m close; stop-entries rest at `nextBarSAR` until a
   later bar tags the level. Dual EMA slope filters are **opt-out**
   (`ema_gate`, factory on). Off: SAR trend + ATR only; SAR flip reverses.
   Optional signal entry lock (no same-direction re-entry after TP/SL until
   the opposite SAR signal) and long-only / short-only side switches.
   `strategy.exit` from average entry: profit
   target / stop loss (factory 50 / 10 pts; **0 disables that side**),
   high/low first-touch on bars after entry; same-bar both → stop.
   Optional ``exit_at_sar`` (factory off) fills SAR-intersection exits at
   the tagged trail instead of the bar close — [ADR-061](ADR-061-psar-ema-exit-at-sar.md).

4. **WebUI** tab `#/external` (**External Models**): edit Pine knobs, **Save
   as defaults** (operator overlay under
   `$VTECH_DATA_DIR/ops/webui/external_models/<id>/defaults.json`), **Reset to
   factory**, and **Save hardened copy** (named snapshot). **Backtest this
   configuration** hands off to `#/backtest?external=<id>` via `sessionStorage`
   (working params) and lists `external:<id>` / `external:<id>:hardened:<name>`
   in the Backtest model dropdown. Listing the tab is YAML + overlays
   only; `databento.tbbo_1min` is read when a Backtest (or chart) **runs**,
   not to paint the knob form.

5. Backtest of an external source uses the existing results UI (summary,
   blotter, equity) but **not** ML τ / TBBO OCO / 50-6 labels. The Pine book
   has its own `strategy.exit` +50/−10 (editable). PnL is on the candle
   timeframe, NQ $20/point.

6. Live pin, ML catalog, and `ps_*` features are unchanged. Stage 1+ (HPO,
   ML gate) is out of scope.

## Consequences

- Operators can tune and score the Pine book on the same console as ML
  backtests without contaminating 50/6 labels.
- Historical coverage is whatever `tbbo_1min` has (currently 2025-04-01 →
  present on this host), including the afternoon session the 5s research
  parquet often lacks.
- Confusion-matrix "true side" for this path is win/loss of the Pine trade,
  not a TBM label.
- 1s path-realised fills (ADR-052) are not used at Stage 0; that is a later
  honesty pass.

## Related

- [ADR-055](ADR-055-parabolic-sar-closed-bars.md) — SAR trail as features
- [ADR-020](ADR-020-rule-strategy-backtest.md) — rule DSL (not used here)
- [ADR-050](ADR-050-hypothesis-side-features.md) — EMA slope failed as 50/6 side
- Package: `src/external_models/`
- UI: `#/external`, Backtest source `external:*`
- [ADR-061](ADR-061-psar-ema-exit-at-sar.md) — optional exit at tagged SAR trail
- [ADR-062](ADR-062-ext-model-split-and-stack.md) — split SAR / EMA; book knobs; stack AND
