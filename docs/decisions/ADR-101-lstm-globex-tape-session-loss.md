# ADR-101: 15s LSTM keeps Globex tape; session is a loss mask

## Status
Accepted — implemented 2026-08-29. Amended by [ADR-104](ADR-104-lstm-session-pmax-and-ui-tau.md) (UI τ is the OOF event cut).

## Date
2026-08-29

## Context

`nq-50pt-econ-lg-st` LSTM `20260828T175240Z` (sqrt, seq_len=60) had
`p_max` 0.51 and **207k** directional OOF rows at fire 0.1. After
ADR-099 morning clip, `20260829T182514Z` has `p_max` **0.38** and
**0** `direction` rows at `labels.fire` 0.40. That is not sqrt: both
runs used `class_balance=sqrt`.

Fold cache concatenated 08:30–11:00 CT mornings. `ts` diffs showed
157 holes of **77415 s (21.5 h)** — 11:00 → next 08:30 treated as one
15s step — plus weekend holes. LSTM `make_windows` has no gap check.
Every session-open lookback was the previous afternoon, so the encoder
went cold. Trees (no sequence) only shrank because of morning rows +
fire 0.40 vs the old 0.1 OOF `direction` cut.

`labels.fire` 0.40 is also baked into OOF `direction`. Heads still emit
`p_long`/`p_short`; the parquet looks “not predicting.”

## Decision

1. **15s fold cache** keeps the Globex tape. Session clock is
   `w_session` (loss mask), not a row drop. 1m/5m trees still clip.
2. Cache key adds `|tape=globex` so the clipped 15s entry is not reused.
3. LSTM `fit` multiplies sample weights by `w_session_train` and by a
   **contiguous-window mask** (no window may span a gap > 45 s).
4. Dual-binary canonical frames include `p_max` and `spread`. Historical
   Backtest **re-derives** `direction` from `p_long`/`p_short`. If the
   book fire admits nothing and τ is off, replay at library fire 0.1
   (the 175240Z operating point). Amended: [ADR-102](ADR-102-gpu-htf-context-hydrate.md)
   treats fewer than 10 book fires the same as none.
5. Next LSTM train (GPU) must use the new 15s cache. Live pin unchanged.

## Consequences

- Next `lstm_15s` Custom Test / joint search rebuilds 15s fold cache
  (~Globex row count, not 123k mornings).
- Existing `182514Z` LSTM OOF still has `p_max` 0.38 until retrained.
  Backtest with τ **off** now shows fire-0.1 directional rows from the
  same `p_long`/`p_short`.
- 1m/5m fire 0.40 is still sparse vs Globex-era fire 0.1 OOF; that is
  the book cut, not a missing-column bug.

## Related

- [ADR-099](ADR-099-lg-st-morning-onset-session-vp.md) — morning train
  for **trees**; 15s tape supersedes the 15s row drop
- [ADR-089](ADR-089-dual-binary-argmax-fire.md), [ADR-095](ADR-095-regime-context-fire-aux.md)
- `src/search/fold_cache.py`, `src/models/sequence_windows.py`,
  `src/models/attention_lstm_plugin.py`, `src/backtest/runner.py`
