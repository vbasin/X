# ADR-133: EMA/VWAP trend-agreement features and es_delta promotion

## Status
Accepted — 2026-09-17. Extends the `fs_` Flow and Trend State family and
Stage 3's `CONTEXT` pack (see
[docs/reference/STAGE3_FEATURES.md](../reference/STAGE3_FEATURES.md)).
**Code-only**: `data/sources/labels/move_matrix*`, `target_matrix*`, and
`target_scores*.parquet` still reflect the prior `CONTEXT` shape until a
rebuild + retrain is run. That rebuild is deliberately deferred, same as
[ADR-131](ADR-131-large-print-taxonomy-redesign.md) — more feature changes
are coming first.

## Date
2026-09-17

## Context

1. **Only one EMA length and one VWAP-slope window existed.** `fs_ema6_slope_r`
   (20s-bar EMA over 6 one-minute closes) and `fs_vwap_velocity_r` (60-second
   VWAP slope) gave Stage 3 a single fast read of trend. There was no medium
   (20-bar) or slow (60-bar) EMA, and no slower VWAP-slope window (5-minute,
   15-minute) to distinguish a brief wiggle from a sustained trend shift.
2. **No feature expressed EMA/VWAP agreement or divergence.** `fs_vwap_ema_spread_r`
   existed (EMA6 minus VWAP, value only) but was never in `CONTEXT`, and there
   was no equivalent for the two new EMA lengths, no comparison of *slopes*
   (is the EMA moving faster or slower than VWAP), and no read on whether that
   slope gap is widening or closing.
3. **`es_delta_1m`/`es_delta_5m` were computed but unused.** `entry_state.py`
   already produces rolling signed aggressor-volume delta at 1-minute and
   5-minute windows, mirrored for side, but neither column was ever selected
   into `CONTEXT` — a built feature was invisible to the model it was built for.

## Decision

1. **Two additional EMA lengths, same 1-minute-close basis as the existing
   6-bar EMA.** `FlowConfig` gains `ema_bars_20` (default 20) and
   `ema_bars_60` (default 60); `build_flow_state()`'s EMA section is now a
   loop over the three spans so all three share one code path. Each emits
   `fs_ema{20,60}_slope_r` — the 60-second slope of that EMA, divided by `R`,
   same convention as the existing `fs_ema6_slope_r`.
2. **Two additional VWAP-slope windows.** `FlowConfig` gains
   `vwap_vel_window_5m` (300s) and `vwap_vel_window_15m` (900s), emitting
   `fs_vwap_velocity_5m_r` and `fs_vwap_velocity_15m_r` — the same
   `vwap.diff(window) / R` construction as `fs_vwap_velocity_r`, just over
   longer windows.
3. **Nine EMA-vs-VWAP agreement features, one triad per EMA length.** For
   each of EMA6/20/60, all measured against the existing 60-second VWAP
   slope (`fs_vwap_velocity_r`):
   - **Value diff**: `(EMA - VWAP) / R`. EMA6's keeps the pre-existing
     `fs_vwap_ema_spread_r` name (now promoted into `CONTEXT`); EMA20/60 get
     `fs_ema20_vwap_diff_r` / `fs_ema60_vwap_diff_r`.
   - **Slope diff**: `fs_ema{N}_slope_r - fs_vwap_velocity_r` — is this EMA
     moving faster or slower than VWAP over the same window —
     `fs_ema{6,20,60}_slope_vwap_diff_r`.
   - **Slope-of-slope-diff**: the 60-second slope of the slope-diff itself —
     is the EMA/VWAP momentum gap widening or closing —
     `fs_ema{6,20,60}_slope_vwap_diff_accel_r`.
   All intermediate quantities are computed in raw price space and divided by
   `R` exactly once at the point of writing the output column, matching the
   existing `fs_vwap_velocity_r`/`fs_vwap_accel_r` convention.
4. **`es_delta_1m` and `es_delta_5m` promoted into `CONTEXT`.** No new
   calculation — both already existed in `entry_state.py`. This is a
   selection change only.
5. **`CONTEXT` grows from 27 to 42 columns** (15 additions: 13 new `fs_`
   columns plus `es_delta_1m`/`es_delta_5m`); total Stage 3 inputs go from 38
   to 53. `DIRECTIONAL_SUFFIXES` in `flow_state.py` gains all 12 new signed
   `fs_` columns (the two `es_` additions were already directional).

## Consequences

**Not yet measured against outcomes.** Same caveat as ADR-131: this changes
feature *construction and selection* only. No move/target matrix has been
rebuilt and no model retrained against the new columns, so their actual
contribution to `p_reach` is unknown. The correlation guard and SHAP ranking
in [data/evaluation/context_analysis/SHAP_SUMMARY.md](../../data/evaluation/context_analysis/SHAP_SUMMARY.md)
predate both this change and ADR-131 and reflect neither.

**Files:**
- `src/features/flow_state.py` — `FlowConfig` gains `ema_bars_20`,
  `ema_bars_60`, `vwap_vel_window_5m`, `vwap_vel_window_15m` (with
  validation); `build_flow_state()`'s VWAP/EMA section rewritten as a loop
  over the three EMA spans; `DIRECTIONAL_SUFFIXES` gains 12 entries.
- `scripts/features/build_target_matrix.py` — `CONTEXT` tuple gains 13 new
  `fs_` columns and `es_delta_1m`/`es_delta_5m` (27 → 42 columns).
- `tests/test_cash_session_flow.py` — new columns added to the
  column-existence check; new tests for the slope-diff/accel algebraic
  relationships and the 5m/15m VWAP-velocity windows.
- `docs/reference/STAGE3_FEATURES.md` — new `es_`/`fs_` rows; header counts
  corrected to 42/53.

**Open concerns / deferred work:**
- Rebuild `move_matrix` and `target_matrix` (context + context_rest
  variants) against the new `CONTEXT` shape, then retrain and re-evaluate
  Stage 3 — deferred until further feature changes land, per the user's
  explicit instruction (same deferral as ADR-131).
- Whether the 9 EMA-vs-VWAP agreement features add signal beyond the
  existing single-EMA/single-VWAP-window reads is unmeasured until the
  rebuild runs; some may prove redundant with each other (e.g. the three
  slope-diff columns share `fs_vwap_velocity_r` as a common term).
- Correlation guard and SHAP ranking need regenerating against the final
  `CONTEXT` shape once all pending feature changes (this ADR + ADR-131) are
  combined into one rebuild.
