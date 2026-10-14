# ADR-062: Split SAR / EMA external models, book knobs, stacked consensus

## Status
Accepted — implemented 2026-08-26. Extended by
[ADR-064](ADR-064-ext-trade-window-tf-units.md) (trade window, per-engine
candle timeframe, PT/SL/min-ATR units) and
[ADR-065](ADR-065-ext-hour-day-timeframes.md) (`1hour` / `1day` TFs).

## Date
2026-08-26

## Context

Stage 0 shipped one combined card `psar_ema20` ([ADR-058](ADR-058-external-models-pine-sar-ema.md)):
Wilder SAR trail plus dual EMA(20) slope as a single Pine order table, with
session / ATR / PT-SL / sides on that card's Parameters form. Backtest mixed
those sources into the same staged-weights dropdown as ML bundles.

The operator asked to:

1. Break the EMA trading rule out so **Parabolic SAR** and **20 EMA** are
   separate external models, each with its own long/short entry/exit signals.
2. On the Backtest tab, choose **ML model** vs **EXT Model** first. ML keeps
   the previous staged/registry / Custom Test flow. EXT shows a dropdown of
   engines, a **DEFAULT** / hardened-snapshot dropdown beside it, and a ＋
   control to stack further engines.
3. Stacked engines **narrow** the book: matching sides enter; opposing or
   flat signals do not enter, and they exit an open position.
4. Move shared knobs (Longs / Shorts / signal lock / ATR period / min ATR /
   profit target / stop loss / session start / session end / contracts /
   round-trip cost / slippage / candle timeframe) off the SAR Parameters
   form onto the Backtest tab as **Backtest Parameters**, shown when
   **EXT Model** is selected.
5. Keep EMA-only knobs on the EMA card. SAR-only knobs stay on SAR
   (`exit_at_sar` remains there — [ADR-061](ADR-061-psar-ema-exit-at-sar.md)).
   ATR period is **not** required for Wilder SAR math (ADR-055 uses it only
   for `dist_atr` feature columns), so it lives solely on the book as the
   min-ATR entry gate.

The combined Pine order table (rest stop-entries at `nextBarSAR`, EMA as a
gate on SAR) is **not** the stack. Stacking is a new composition rule.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Keep one card, hide EMA via `ema_gate` | Replica of Pine | Not two models; cannot stack independently |
| B. Two engines + book + AND consensus (chosen) | Matches the request; book knobs apply once | Combined Pine rest-orders are gone |
| C. Encode stack in the ADR-020 rule DSL | One backtest engine | DSL still cannot express SAR trail or EMA slope |

## Decision

1. **Catalog**
   - `config/external_models/psar.yaml` — engine `psar`. Knobs: `start`,
     `increment`, `maximum`, `exit_at_sar`.
   - `config/external_models/ema20.yaml` — engine `ema`. Knobs: `ema_length`,
     `ema20_l_bars`, `ema20_s_bars`.
   - `psar_ema20.yaml` remains as a **hidden** compatibility card
     (`listed: false`). `external:psar_ema20` expands to stack `[psar, ema20]`.
   - Saved overlays under `ops/webui/external_models/_book/defaults.json`
     for book knobs. A one-shot migrate splits a pre-existing
     `psar_ema20/defaults.json` into `psar` / `ema20` / `_book` (the old
     file is not deleted).

2. **Signals** (per closed bar, `{-1, 0, +1}`)
   - SAR: `+1` while the closed-bar Wilder trend is up, `-1` while down,
     `0` during warmup.
   - EMA: `+1` when established (L) and steep (S) slopes agree up, `-1`
     when they agree down, `0` on disagreement or warmup. Look-back `0`
     disables that filter.

3. **Backtest Parameters** (shared book, `#/backtest` when type is **EXT Model**)
   - Longs, Shorts, signal entry lock, ATR period, min ATR, profit target,
     stop loss, session start, **session end** (renamed from flatten),
     contracts, round-trip cost, slippage per side, candle timeframe.
   - Not shown on `#/external` (that tab is engine knobs + hardened
     snapshots only). Saved overlays still live under
     `ops/webui/external_models/_book/defaults.json`.
   - Entries at close ± slippage. PT/SL first-touch on later bars (0 =
     off). Optional SAR-trail stop when a stacked `psar` leg has
     `exit_at_sar` and the incoming trail is tagged. Flatten at session end.

4. **Stack consensus**
   - AND of sides. All engines `+1` → long; all `-1` → short; otherwise
     flat (no new entry; exit if in a position). A reverse (long → short
     with ATR ok) is a close + open on the same bar. Intrabar `sar_stop`
     skips same-bar re-entry.

5. **Backtest UI**
   - Initial dropdown: **ML model** | **EXT Model**.
   - ML: previous staged/registry / Browse / τ / OCO / on-close. EXT
     sources are **not** mixed into `/api/backtest/models`.
   - EXT: one or more rows `[engine] [DEFAULT | hardened…]`, ＋ under the
     last row, − on extra rows, then **Backtest Parameters**. Run posts
     `stack` + `book`.

6. Live pin, `ps_*` features, and the ML catalog are unchanged.

## Consequences

- Scoring SAR-only vs EMA-only vs SAR+EMA AND is a Backtest stack choice,
  not an `ema_gate` toggle.
- Historical `external:psar_ema20` URLs still run, as the two-engine stack,
  which is **not** bit-identical to the ADR-058 Pine order table.
- Operators save engine knobs per card and book knobs once.

## Related

- [ADR-058](ADR-058-external-models-pine-sar-ema.md) — External Models plane
  (combined card superseded for UI)
- [ADR-061](ADR-061-psar-ema-exit-at-sar.md) — `exit_at_sar` now on `psar`
- [ADR-055](ADR-055-parabolic-sar-closed-bars.md) — SAR state machine
- Package: `src/external_models/`
- UI: `#/external`, Backtest type **EXT Model**
