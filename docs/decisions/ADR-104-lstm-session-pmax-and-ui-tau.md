# ADR-104: LSTM session p_max and Backtest UI τ as the event cut

## Status
Accepted — implemented 2026-08-29. Live pin unchanged.  
Last-fold `predict_canonical` also uses UI τ as the event cut (was still baking `labels.fire` 0.40).

Amends [ADR-099](ADR-099-lg-st-morning-onset-session-loss.md) (onset
uniqueness on sequence models), [ADR-095](ADR-095-regime-context-fire-aux.md)
(HTF on the 15s tape), [ADR-101](ADR-101-lstm-globex-tape-session-loss.md)
(τ-off fallback). Related: [ADR-089](ADR-089-dual-binary-argmax-fire.md),
[ADR-102](ADR-102-gpu-htf-context-hydrate.md).

## Date
2026-08-29

## Context

Custom Test LSTM `20260829T204718Z` (Globex tape + 13 HTF cols, sqrt,
onset weights) listed on `#/backtest` but:

1. **Only 0% confidence (gate off) produced trades.** Session
   `p_max` maxed at **0.39**. Book `labels.fire` 0.40 baked OOF
   `direction` all-flat in RTH (the two 0.40 rows were overnight).
   The UI τ then filtered an empty book. Gate-off fell back to fire
   0.1 → 5.6k 2026 H1 fills, Sharpe **−0.48**.
2. Label-implied session NE **rises** with τ (0.10 → +1.4, 0.35 →
   +4.0, n=686). The model ranks; the event cut never saw RTH mass
   at 0.40.
3. `175240Z` (17-col tape, overlapping TBM labels, no HTF in the
   sequence) had `p_max` 0.51. `204718Z` put `h1_*`/`d1_*` in the
   60-step 15s window (piecewise constant over 15 min), computed
   `class_balance` on Globex including overnight flats, and
   zero-weighted overlapping onsets (ADR-099). Those three stacked
   to compress session `p_max` below the book cut.

## Decision

1. **OOF replay event cut is the UI τ** when the confidence gate is
   on, else library fire 0.1. Stop baking `labels.fire` before the
   dropdown. 35% confidence on `204718Z` is 686 session events, not
   “must uncheck enable”.
2. **HTF last-bar context, not the 15s sequence.** `h1_*` / `d1_*`
   join `lg_*`/`st_*` as `long_ctx`/`short_ctx`. The tape stays the
   native 15s columns.
3. **Sequence families skip train-onset uniqueness.** Overlapping
   15s TBM labels are the path of an open 50/6; zeroing them starved
   the event heads. Trees still drop overlapping onsets.
4. **`class_balance` frequencies use loss-active rows** (session ×
   contiguous-window mask). Overnight flats no longer set √(1/f).

Live pin unchanged. Do not promote. Retrain `lstm_15s` after this
lands; do not reuse `204718Z` as the Globex+HTF-in-sequence run.

## Consequences

- `#/backtest` confidence % maps onto `p_max` for dual-binary OOF.
- Listing sets `recommended_tau` from `labels.fire` (0.40) and
  `span_*` from `oof.parquet` (ends 2026-06-30). The 3-class **80%**
  default plus **MTD on the hive through August** showed 0 directional
  calls even when OOF had 50k up/down rows.
- 1m/5m trees unchanged (onset drop + HTF in the row).
- Serving `window_pack_for_model` follows the same HTF-as-ctx split.

## Related

- `src/models/dual_binary.py` (`head_feature_spec`, `dual_sample_weight`)
- `src/models/attention_lstm_plugin.py`, `src/search/controller.py`
- `src/backtest/runner.py` (`_recanonicalize_dual_oof`)
