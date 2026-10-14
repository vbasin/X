# ADR-131: Large-print taxonomy redesign (continuation vs reversal, threshold-normalised)

## Status
Accepted — 2026-09-17. Amends the `fs_` large-print taxonomy introduced in the
"cash-session levels + auction-state confirmation" spec revision (ROADMAP,
2026-09-11) and consumed by [ADR-127](ADR-127-tbbo-tape-and-cascade-architecture.md)'s
Stage 3 CONTEXT pack. **Code-only**: `data/sources/labels/move_matrix*`,
`target_matrix*`, and `target_scores*.parquet` still reflect the prior
taxonomy until a rebuild + retrain is run. That rebuild is deliberately
deferred — more feature changes are coming first.

## Date
2026-09-17

## Context

### 1. The old four-state taxonomy conflated two different questions

`accept` / `absorb_trend` / `absorb_counter` / `reject` classified a large
print by aggressor side, candle direction, and **body-vs-wick** location
only. That mixes two distinct claims a trader would separate — "does this
print support the candle continuing" and "does this print mark a reversal
of it" — into one location boundary (body/wick), and the near-wick vs body
split for absorption was arbitrary relative to where the print actually sat
in the candle's range.

### 2. Continuation and reversal need different geometry, not the same wick/body cut

A continuation read should weigh where the print sits relative to the
**candle's own trading range** — a print in the bottom half of an up candle
(anywhere from the low through the body midpoint) that gets absorbed without
turning the candle down is a stronger read than the old near-wick/body
membership test. A reversal read should weigh how far the print pushed
**into the wick relative to the body that failed to follow it** — a large
wick on a small body is a materially stronger rejection than the same wick on
a large body, and the old taxonomy could not express that at all.

### 3. Volume normalisation hides large prints in high-volume regimes

Dividing every large-print sum by rolling traded volume made a large print
read as unremarkable whenever volume was already high — exactly when a large
print is easiest to miss and most likely to matter. Normalising each
qualifying print's contract size against the large-print threshold
(`large_order_size`, default 20) instead keeps a large print's contribution
constant in scale regardless of the surrounding volume regime.

### 4. Stage 3 was training on the four raw states directly

`fs_accept_1m`, `fs_absorb_counter_1m`, and `fs_reject_1m` rode into the
Stage 3 CONTEXT pack individually. Four separate, sparsely-firing signed
columns (a large print is a rare event) are a weaker training signal than
one composite reading of net large-print flow quality, at two lookback
horizons.

## Decision

1. **Four categories, redefined on candle geometry.** All are computed at
   print level in `classify_large_prints()` (`src/features/flow_state.py`)
   against the OHLC bar the print closed into, mirrored for down candles:
   - `accept_continue` — trend-side print anywhere in the candle body.
   - `absorb_continue` — counter-side print in the **bottom half of the
     candle**, split at the body midpoint (low through midpoint on an up
     candle; midpoint through high on a down candle) — not the body/wick
     boundary.
   - `absorb_reversal` (renamed from `absorb_counter`) — trend-side print in
     the far wick (upper wick on an up candle, lower wick on a down candle).
   - `reject_reversal` (renamed from `reject`) — counter-side print in the
     far wick.
2. **Continuation and reversal each get their own scale factor.**
   Continuation contributions are multiplied by `body_range / full_range`;
   reversal contributions by `wick_range / body_range`, capped at
   `cfg.max_reversal_scale` (default 10.0) so a near-zero body cannot blow
   the ratio up unbounded. Both guard their denominator (`full_range <= 0`
   or `body_range <= 0` yields a zero scale, not a division error).
3. **Print size is normalised by the large-print threshold, not by volume.**
   Every qualifying print contributes `size / cfg.large_order_size` before
   scaling — a 20-contract print contributes `1.0`, a 40-contract print
   contributes `2.0`, regardless of the bar's traded volume. The rolling
   volume division on this family is removed entirely (`fs_large_share_*`,
   which serves an unrelated purpose, is untouched).
4. **Two composite features replace the four raw ones as Stage 3 inputs.**
   - `fs_flow_score` — sum of all four categories, rolled over
     `cfg.fast_window` (60 s / "1 bar").
   - `fs_flow_score_5bar` — the same combined signal rolled over
     `cfg.slow_window` (300 s = 5 × `fast_window`) — the aggregate of the
     last 5 one-minute `fs_flow_score` reads.
   - The four raw categories (`fs_accept_continue`, `fs_absorb_continue`,
     `fs_absorb_reversal`, `fs_reject_reversal`) remain queryable on the
     feature frame for diagnostics but are **not** in Stage 3's CONTEXT pack.
5. **A reversal price level is carried but not yet consumed.**
   `fs_last_reversal_level` holds the raw execution price of the most recent
   `absorb_reversal` or `reject_reversal` print, forward-filled for
   `cfg.slow_window` seconds (5 bars) and reset to NaN once older. It is an
   absolute price — excluded from `DIRECTIONAL` so it is never sign-mirrored
   for a short — and reserved for a future entry/stop-level calculation
   (not yet specified).
6. **`CONTEXT` in `build_target_matrix.py` updated.** `fs_accept_1m`,
   `fs_absorb_counter_1m`, `fs_reject_1m` are replaced by `fs_flow_score`,
   `fs_flow_score_5bar` — CONTEXT drops from 28 to 27 columns, total Stage 3
   inputs from 39 to 38.
7. **The `st_bounce_accept` / structural-transitions flow tags follow the
   rename.** `build_structural_transitions()`'s `accept` / `reject`
   parameters are now fed `fs_accept_continue` / `fs_reject_reversal`
   (previously `fs_accept_1m` / `fs_reject_1m`) in `build_move_matrix.py`.

## Consequences

**Not yet measured against outcomes.** This ADR changes feature
*construction* only; no move/target matrix has been rebuilt and no model
retrained against the new columns. `y_reach`, `y_move`, target geometry,
and every other Stage 1/3 feature are unaffected. The independence guard
(`corr(p_move, p_reach)`) and the SHAP ranking in
[data/evaluation/context_analysis/SHAP_SUMMARY.md](../../data/evaluation/context_analysis/SHAP_SUMMARY.md)
were computed against the *prior* taxonomy and do not yet reflect this
change.

**Files:**
- `src/features/flow_state.py` — `classify_large_prints()` rewritten;
  `FlowConfig` gains `max_reversal_scale`; `build_flow_state()`'s large-print
  section rewritten (new columns, size/threshold normalisation, the two
  composite features, the reversal-level tracker); `DIRECTIONAL_SUFFIXES`
  updated.
- `scripts/features/build_move_matrix.py` — structural-transitions call
  site updated to the renamed accept/reject columns.
- `scripts/features/build_target_matrix.py` — `CONTEXT` tuple updated.
- `src/features/structural_transitions.py` — docstring reference updated.
- `tests/test_cash_session_flow.py` — taxonomy tests rewritten for the new
  categories and scale factors; new tests for `fs_flow_score` linearity and
  `fs_last_reversal_level` expiry.
- `docs/reference/STAGE3_FEATURES.md` — `fs_` table rewritten with a "Used
  in Stage 3 model?" column; header counts corrected to 27/38.
- `docs/roadmap/ROADMAP.md` — historical taxonomy bullet annotated with the
  2026-09-17 revision (not moved to changelog; the phase is not complete).

**Open concerns / deferred work (explicitly, per this request):**
- Rebuild `move_matrix` and `target_matrix` (context + context_rest
  variants) against the new taxonomy, then retrain and re-evaluate Stage 3
  before any promotion decision — deferred until further feature changes
  land, per the user's explicit instruction.
- `fs_last_reversal_level`'s intended use in entry/stop calculations is not
  yet specified; no gate or model consumes it today.
- Whether `fs_flow_score` / `fs_flow_score_5bar` actually outperform the
  four raw categories they replace is unmeasured until the rebuild runs.
