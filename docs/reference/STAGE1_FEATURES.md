# Stage 1 Cascade Feature Reference

> **2026-09-24 correction to the causality claim below:** `_order_flow` in
> `src/training/entry_state.py` uses a full-request maximum to normalize
> `es_cvd_*`, `es_delta_*`, and `es_ofi_cum`. Earlier decisions change when
> later values change. Existing affected matrices require causal rebuild and
> revalidation; the independent `nq-selective-intraday` experiment does not
> consume them. [Evidence](../research/NQ_SELECTIVE_RESULTS.md).

> **Scope**: the numeric inputs used by `scripts/features/eval_move_model.py` for the Stage 1 move-detector classifier.
>
> **Current build**: `data/sources/labels/move_matrix_v4/` (`move_*.parquet`, one file per session), written by `scripts/features/build_move_matrix.py`. `move_matrix`/`move_matrix_v2` are frozen historical snapshots predating the tape-causality fixes below; `move_matrix_v3` predates the clustered-bound `es_` columns added in `v4` (see *Multi-session clustered bounds*, below).
>
> **Row grain**: one row per decision timestamp **and side** — long and short each get their own row on the same decision bar. Unlike Stage 3 there is no candidate/target dimension; Stage 1 asks only "does a move happen," not "which level."
>
> **Feature selection is uncurated.** `eval_move_model.py` takes every numeric column on the move matrix except the label/identity set in `LEAK` (and `r_value`) — there is no `CONTEXT`-style allowlist like Stage 3's. Every `es_`/`fs_`/`st_` column the builders emit is a model input unless explicitly excluded below.
>
> **Causality**: all feature values are measured at or before the decision instant. `y_move` and its supporting columns describe the future path after the decision and are excluded from the feature set. Where a feature needs "the current price" or a session-to-date profile/high/low, it reads the raw print tape directly (last print at or before the decision instant) rather than the 1-second bar covering the decision — `tape_to_path` floor-labels bars by their start, so a bar's own close/high/low can carry a print up to ~1s *after* the decision instant, which the tape-level lookups avoid by construction. `fs_delta_*`/`fs_ofi_*`/`fs_delta_ratio_*`/`fs_delta_per_body_*`/`fs_absorption_*`/`fs_impact_efficiency_*` were rebuilt in `move_matrix_v3` off individual prints (causal running-total window-diffs) rather than the floor-labelled bar grid's `delta`/`ofi` columns, closing that exposure too.

## Impact Scope

| Scope | Meaning |
|---|---|
| **Model input** | Passed to the Stage 1 `XGBClassifier` as a numeric feature. The model uses it to estimate `p_move`. |
| **Label** | Used to construct `y_move`; not a model input. |
| **Carried metadata** | Stored in the move parquet for later gates, analysis, or replay, but not passed to `.fit()`. |
| **Ablation-only drop** | Still a model input by default, but removed under `--no-order-flow` (drops every `fs_` column plus any column whose name contains `cvd`, `ofi`, or `delta`) — the flag that shows what price alone, with no order-flow read, can do. |

Stage 1 trains `XGBClassifier` on `y_move` only. Sample weights are uniqueness weights derived from `t_move` and the label horizon (`src/evaluation/uniqueness.py`), not a model input.

## Context Features

These three column families are merged onto every decision row: `es_` from the decision bar itself (`src/training/entry_state.py`), `fs_` from the same 1-second path (`src/features/flow_state.py`), and `st_` from the prior closed session's profile (`src/features/structural_transitions.py`, [ADR-134](../decisions/ADR-134-cascade-ssot-structural-transitions-policy.md)). All three feed Stage 3's curated `CONTEXT` too — see [STAGE3_FEATURES.md](STAGE3_FEATURES.md) — but Stage 1 sees the **full** column set from each module, not Stage 3's subset.

### Auction State: `es_` (42 columns)

Causal entry-state measurements from `src/training/entry_state.py`. Order-flow columns (`es_cvd_*`, `es_ofi_cum`, `es_delta_*`) come from the native 15s hive and exist only from 2025-04-01; sessions without hive coverage carry NaN rather than a price-only proxy (ADR-123 measured a CLV-based proxy at r≈0.00 against true delta-at-price).

| Feature | Meaning and value determination | Directional? | Value range | Impact |
|---|---|---|---|---|
| `es_vs_dev_poc` | `(price - developing-profile POC) / R`. Profile is exact volume-at-tick from every print strictly up to the decision instant (`tape_levels`'s tick-exact convention, not a bar-smeared one), recomputed from scratch per decision. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_dev_vah` | `(price - developing-profile VAH) / R`. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_dev_val` | `(price - developing-profile VAL) / R`. | Yes | `(-inf, +inf)` | Model input |
| `es_va_width_r` | Developing value-area width (`VAH - VAL`, floored at one tick), divided by `R`. | No | `[0, +inf)` | Model input |
| `es_in_value_area` | `1` when price is inside the developing `[VAL, VAH]` band, else `0`. | No | `{0, 1}` | Model input |
| `es_profile_skew` | `(developing POC - session midrange) / session range` — positive means the developing profile's mass sits above the midpoint of the range traded so far. | Yes | Approximately `[-1, 1]` | Model input |
| `es_session_range_r` | Running session high minus running session low from the tape (last print at or before the decision instant), divided by `R`. | No | `[0, +inf)` | Model input |
| `es_vs_session_open` | `(price - first print of the session) / R`. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_session_high` | `(price - running session high) / R`; zero or negative by construction. | Yes | `(-inf, 0]` | Model input |
| `es_vs_session_low` | `(price - running session low) / R`; zero or positive by construction. | Yes | `[0, +inf)` | Model input |
| `es_vs_prior_poc` | `(price - prior closed session's POC) / R`. NaN if no prior session is available. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_prior_vah` | `(price - prior session's VAH) / R`. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_prior_val` | `(price - prior session's VAL) / R`. | Yes | `(-inf, +inf)` | Model input |
| `es_open_vs_prior_va` | `(today's open - prior session's POC) / prior value-area width` — where the session started relative to yesterday's value, width-normalised rather than `R`-normalised. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_prior_hi` | `(price - prior session's high) / R` — the prior **range boundary**, a different state from leaving VAH behind. | Yes | `(-inf, +inf)` | Model input |
| `es_vs_prior_lo` | `(price - prior session's low) / R`. | Yes | `(-inf, +inf)` | Model input |
| `es_gap_r` | `(today's open - prior session's close) / R`. NaN if no prior close is available. | Yes | `(-inf, +inf)` | Model input |
| `es_n_bounds` | Count of multi-session KDE-clustered structural bounds in scope (`src/features/cash_session_profile.levels_for_price`, up to `lookback_sessions` closed sessions, extended if price sits outside the whole clustered range) — the same historic volume-profile structure Stage 3's `tgt_*`/`n_bounds`/`inside_bound` target against, now read at the Stage 1 decision bar too. NaN if no `level_hist` was supplied. | No | `[0, +inf)` integer | Model input |
| `es_inside_bound` | `1` when price sits inside one of those bounds right now, else `0`. | No | `{0, 1}` or NaN | Model input |
| `es_inside_bound_strength` | Strength (session-agreement-weighted mass) of the containing bound; `0` when not inside one. | No | `[0, +inf)` | Model input |
| `es_inside_bound_tests` | Cumulative count of entries into the containing bound from outside, so far this session (`bound_interaction_matrices`, causal — reflects only prints at or before the decision instant). `0` when not inside a bound. | No | `[0, +inf)` integer | Model input |
| `es_inside_bound_breaks` | Cumulative count of traverses clean through the containing bound's far side. | No | `[0, +inf)` integer | Model input |
| `es_inside_bound_rejects` | Cumulative count of entries that reversed back out the side they came from — the event a trader reads as the level holding. | No | `[0, +inf)` integer | Model input |
| `es_inside_bound_hold_ratio` | `rejects / max(tests, 1)` for the containing bound — same formula as `cash_session_profile.bound_candidates`'s `hold_ratio`; a zone broken more often than it has held is weaker overhead. | No | `[0, 1]` | Model input |
| `es_channel_width_r` | Gap between the nearest bound above price and the nearest bound below, divided by `R`. NaN unless both flank price. | No | `(0, +inf)` or NaN | Model input |
| `es_vs_bound_above_r` | `(price - nearest bound-above's near edge) / R`; zero or negative by construction, same sign convention as `es_vs_session_high`. NaN if no bound sits above (including when even `max_lookback_sessions` cannot find one). | Yes | `(-inf, 0]` or NaN | Model input |
| `es_vs_bound_below_r` | `(price - nearest bound-below's near edge) / R`; zero or positive by construction, same sign convention as `es_vs_session_low`. | Yes | `[0, +inf)` or NaN | Model input |
| `es_pos_in_channel` | Position between the flanking bounds, `(price - below.hi) / channel_width - 0.5` — `0` at the channel midpoint, mirrored for shorts like `es_profile_skew`. NaN unless both bounds flank price. | Yes | `[-0.5, 0.5]` or NaN | Model input |
| `es_ret_5m` | `(price - price 5 minutes ago) / R`; clamps to the session start if fewer than 5 minutes have elapsed. | Yes | `(-inf, +inf)` | Model input |
| `es_ret_15m` | Same construction over 15 minutes. | Yes | `(-inf, +inf)` | Model input |
| `es_ret_60m` | Same construction over 60 minutes. | Yes | `(-inf, +inf)` | Model input |
| `es_atr_r` | Mean high-low range of the latest 5 **completed** 1-minute candles (e.g. `databento.tbbo_1min`), divided by `R`; held constant from the instant a candle actually completes until the next one does, so a still-forming candle is never read. The 5 candles may sit before 08:30 (contiguous Globex minutes, [ADR-137](../decisions/ADR-137-continuous-lookback-adjacency.md)). NaN only when fewer than 5 completed candles exist in that history or no 1-minute source is supplied. | No | `(-inf, +inf)`, typically `≥ 0` | Model input |
| `es_realized_vol` | Rolling 5-sample standard deviation of 1-minute log returns from the same completed-candle series, scaled by `√5`; same causal hold and NaN conditions as `es_atr_r`. | No | `[0, +inf)` | Model input |
| `es_cvd_session` | Session cumulative volume delta at the decision bar (`merge_asof` backward onto the native 15s hive), scaled by the session's peak absolute CVD. | Yes | Approximately `[-1, 1]` | **Ablation-only drop** |
| `es_cvd_slope_5m` | Change in `es_cvd_session`'s underlying CVD over the trailing 5 minutes, same scale. | Yes | Approximately `[-2, 2]` | **Ablation-only drop** |
| `es_cvd_slope_15m` | Same over the trailing 15 minutes. | Yes | Approximately `[-2, 2]` | **Ablation-only drop** |
| `es_delta_1m` | Rolling 60-second sum of signed aggressor volume from `of_delta`, same CVD scale. | Yes | Approximately `[-2, 2]` | **Ablation-only drop** |
| `es_delta_5m` | Same rolling sum over 300 seconds. | Yes | Approximately `[-2, 2]` | **Ablation-only drop** |
| `es_ofi_cum` | Cumulative order-flow imbalance (`of_ofi_cum`) at the decision bar, scaled by its own session peak absolute value. | Yes | Approximately `[-1, 1]` | **Ablation-only drop** |
| `es_cvd_price_divergence` | `(CVD change - price change/R)` over the trailing 900 seconds — positive means aggression outpaced the resulting move (or ran opposite it). | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `es_min_since_open` | Minutes since 08:30 CT. | No | `[0, 375]` inside the entry window | Model input |
| `es_min_to_close` | Minutes until 15:00 CT. | No | `[15, 390]` inside the entry window | Model input |

`es_atr_r` and `es_realized_vol` are not in `_order_flow`'s output and are **not** dropped by `--no-order-flow` even though they use price history only — the ablation flag matches on column name substring (`cvd`/`ofi`/`delta`) and prefix (`fs_`), not on data lineage.

### Flow and Trend State: `fs_` (39 columns)

From `src/features/flow_state.py`, computed on the 1-second cash-session path with TBBO `delta`/`ofi` where available. Every measure is aggression **relative to the price response it produced** (Patzelt & Bouchaud's arc-shaped OFI-to-return relation, ADR-122-era finding retained here) rather than raw flow magnitude. `fs_clv` and the `fs_ad_line`/`fs_ad_slope_*`/`fs_ad_price_divergence_*` columns (Chaikin CLV/A-D line) were removed: CLV needs a bar's full high/low range, which the 1s bar grid cannot give causally without a bar-width redesign, so they were dropped rather than patched.

| Feature | Meaning and value determination | Directional? | Value range | Impact |
|---|---|---|---|---|
| `fs_delta_1m` | Rolling 60-second sum of signed bar-level `delta` (aggressor buy volume minus sell volume). | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_delta_ratio_1m` | `fs_delta_1m` divided by traded volume over the same 60 seconds. | Yes | Approximately `[-1, 1]` | **Ablation-only drop** |
| `fs_ofi_1m` | Rolling 60-second sum of bar-level `ofi` (falls back to `delta` when BBO-derived OFI is unavailable). | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_delta_per_body_1m` | `\|fs_delta_1m\|` divided by the absolute 60-second candle body (floored at 0.25 pts) — signed flow per point of price movement produced. | No | `[0, +inf)` | **Ablation-only drop** |
| `fs_delta_5m` | Same as `fs_delta_1m` over 300 seconds. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_delta_ratio_5m` | Same as `fs_delta_ratio_1m` over 300 seconds. | Yes | Approximately `[-1, 1]` | **Ablation-only drop** |
| `fs_ofi_5m` | Same as `fs_ofi_1m` over 300 seconds. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_delta_per_body_5m` | Same as `fs_delta_per_body_1m` over 300 seconds. | No | `[0, +inf)` | **Ablation-only drop** |
| `fs_absorption_1m` | Residual `β·OFI - Δprice` over 60 seconds, `β` a rolling Kyle's-lambda estimate (Cont-Kukanov-Stoikov) fit on strictly prior bars, divided by `R`. Positive means the move was **smaller** than the flow implied — flow was absorbed. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_impact_efficiency_1m` | `Δprice / fs_ofi_1m` (unnormalised) over 60 seconds — the arc-shaped relation directly, separating a move flow paid for from one it did not. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_absorption_5m` | Same as `fs_absorption_1m` over 300 seconds. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_impact_efficiency_5m` | Same as `fs_impact_efficiency_1m` over 300 seconds. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_vwap_dist_r` | `(price - cumulative session VWAP) / R`. Both terms are built from individual prints when the raw tape is supplied: `price` is the last print at or before the decision instant and VWAP is the true `Σ(price×size)/Σ(size)` over prints up to that same instant, not a `(H+L+C)/3` bar proxy. Falls back to the bar's own close/OHLC proxy, not causally clean, when the tape is unavailable. | Yes | `(-inf, +inf)` | Model input |
| `fs_vwap_velocity_r` | Slope of cumulative VWAP over the trailing 60 seconds (`vwap.diff(60) / R`). | Yes | `(-inf, +inf)` | Model input |
| `fs_vwap_accel_r` | 60-second slope of `fs_vwap_velocity_r` itself. | Yes | `(-inf, +inf)` | Model input |
| `fs_vwap_velocity_5m_r` | VWAP slope over the trailing 300 seconds. | Yes | `(-inf, +inf)` | Model input |
| `fs_vwap_velocity_15m_r` | VWAP slope over the trailing 900 seconds. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema6_dist_r` | `(price - 6-bar 1-minute-close EMA) / R`, `price` the same tape-causal current price as `fs_vwap_dist_r`; the EMA itself was already causal (built from minute-resampled, right-labelled closes). The span includes contiguous 1-minute closes before 08:30 when they are loaded ([ADR-137](../decisions/ADR-137-continuous-lookback-adjacency.md)); EMA20 / EMA60 use the same prefix. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema6_slope_r` | 60-second slope of the 6-bar EMA, divided by `R`. | Yes | `(-inf, +inf)` | Model input |
| `fs_vwap_ema_spread_r` | `(EMA6 - VWAP) / R` — legacy name, kept for the original short EMA. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema20_slope_r` | 60-second slope of the 20-bar 1-minute-close EMA, divided by `R`. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema60_slope_r` | 60-second slope of the 60-bar 1-minute-close EMA, divided by `R`. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema20_vwap_diff_r` | `(EMA20 - VWAP) / R`. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema60_vwap_diff_r` | `(EMA60 - VWAP) / R`. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema6_slope_vwap_diff_r` | `fs_ema6_slope_r - fs_vwap_velocity_r` — is EMA6 moving faster or slower than VWAP over the same 60-second window. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema20_slope_vwap_diff_r` | Same for EMA20. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema60_slope_vwap_diff_r` | Same for EMA60. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema6_slope_vwap_diff_accel_r` | 60-second slope of `fs_ema6_slope_vwap_diff_r` — whether the EMA6-vs-VWAP momentum gap is widening or closing. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema20_slope_vwap_diff_accel_r` | Same for EMA20. | Yes | `(-inf, +inf)` | Model input |
| `fs_ema60_slope_vwap_diff_accel_r` | Same for EMA60. | Yes | `(-inf, +inf)` | Model input |
| `fs_accept_continue` | Causal 60-second rolling sum of trend-side large prints (`size ≥ large_order_size`) landing in the candle body, each contributing `sign(candle) × (size / large_order_size) × (body_range / full_range)` (ADR-131). | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_absorb_continue` | Same rolling sum for counter-side large prints in the candle's bottom half (up candle) / top half (down candle), split at the body midpoint. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_absorb_reversal` | Same rolling sum for trend-side large prints in the far wick, scaled by `wick_range / body_range` (capped at `max_reversal_scale = 10.0`). | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_reject_reversal` | Same rolling sum for counter-side large prints in the far wick, same wick/body scale and cap. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_flow_score` | Sum of the four large-print categories above, rolled over the trailing 60 seconds — one composite net-signed flow-quality reading. | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_flow_score_5bar` | Same composite, rolled over the trailing 300 seconds (5× `fs_flow_score`'s window). | Yes | `(-inf, +inf)` | **Ablation-only drop** |
| `fs_large_share_1m` | Sum of qualifying large-print contract sizes over 60 seconds, divided by traded volume over the same window. | No | `[0, +inf)`, typically `[0, 1]` | **Ablation-only drop** |
| `fs_large_share_5m` | Same over 300 seconds. | No | `[0, +inf)`, typically `[0, 1]` | **Ablation-only drop** |
| `fs_last_reversal_level` | Raw execution price of the most recent `absorb_reversal`/`reject_reversal` print, forward-filled for 300 seconds then reset to NaN. **Not mirrored for shorts** (absolute price, deliberately absent from `DIRECTIONAL_SUFFIXES`) — see *Open Concerns* below. | No | Raw instrument price or NaN | **Ablation-only drop**; see note |

### Structural Transitions: `st_` (26 columns)

From `src/features/structural_transitions.py` ([ADR-134](../decisions/ADR-134-cascade-ssot-structural-transitions-policy.md)), measured against the **single immediately-preceding closed cash session's** `lo/val/poc/vah/hi` — not the multi-session KDE-clustered bounds Stage 3's target geometry uses. A 2-point touch tolerance and causal 1-second path state apply throughout; `-1` generally means the event has not occurred (age columns) or has no valid reading. Stage 3's `CONTEXT` uses 19 of these 26; Stage 1 uses all 26.

| Feature | Meaning and value determination | Directional? | Value range | Impact |
|---|---|---|---|---|
| `st_zone` | Ordinal location relative to the prior session's profile: outside low/high, below/inside/above value, or at POC. Symmetric, side-mirrored codes. | Yes | Discrete `[-3, 3]` | Model input |
| `st_zone_dwell_s` | Wall-clock seconds continuously in the current `st_zone`. | No | `[0, +inf)` | Model input |
| `st_dist_prior_hi_r` | `(price - prior session high) / R`. | Yes | `(-inf, +inf)` | Model input |
| `st_dist_prior_lo_r` | `(price - prior session low) / R`. | Yes | `(-inf, +inf)` | Model input |
| `st_va_entry_age_s` | Seconds since price most recently entered the prior value area from outside. | No | `[-1, +inf)` | Model input |
| `st_va_entry_depth_r` | Signed distance travelled into the prior value area since that entry, divided by `R`. | Yes | `(-inf, +inf)`, NaN before entry | Model input |
| `st_va_time_inside_s` | Cumulative wall-clock seconds spent inside the prior value area so far today. | No | `[0, +inf)` | Model input |
| `st_break_age_s` | Seconds since price crossed through a prior value-area edge (VAH up or VAL down). | No | `[-1, +inf)` | Model input |
| `st_break_level` | Direction of the active break: `+1` up through VAH, `-1` down through VAL, `0` with no active break. | Yes | `{-1, 0, 1}` | Model input |
| `st_break_extension_r` | Current signed distance beyond the broken edge, divided by `R`. | Yes | `(-inf, +inf)` | Model input |
| `st_break_max_extension_r` | Maximum extension reached since the break, divided by `R`. | No | `[0, +inf)` | Model input |
| `st_break_pullback_r` | `st_break_max_extension_r` minus the current extension — retracement from the break's extreme. | No | `[0, +inf)` | Model input |
| `st_reclaim_flag` | `1` once price has crossed back through the broken edge after the break. | No | `{0, 1}` | Model input |
| `st_reclaim_age_s` | Seconds since that reclaim. | No | `[-1, +inf)` | Model input |
| `st_reject_flag` | `1` if the break was reclaimed within `reject_window_s` (300s) of occurring — a break undone quickly is a different event from one undone hours later. | No | `{0, 1}` | Model input |
| `st_bounce_confirm` | `1` once price has retraced to the broken edge (within `bounce_tol_pts`) and then extended further in the break direction — a geometric retrace-and-continue. | No | `{0, 1}` | Model input |
| `st_bounce_accept` | Same bounce state, additionally requiring the signed accept/reject taxonomy (`fs_accept_continue`/`fs_reject_reversal`) to confirm flow in the break direction at that moment; NaN if no taxonomy was supplied. | No | `{0, 1}` or NaN | Model input |
| `st_vwap_cross_age_s` | Seconds since the most recent causal session-VWAP cross. | No | `[-1, +inf)` | Model input |
| `st_vwap_cross_side` | Side of price relative to VWAP at the most recent cross (`+1`/`-1`). | Yes | `{-1, 0, 1}` | Model input |
| `st_vwap_cross_strength` | `fs`-scale VWAP velocity (`/R`) at the moment of that cross. | Yes | `(-inf, +inf)`, NaN before any cross | Model input |
| `st_vwap_cross_count` | Number of VWAP crosses so far today. | No | `[0, +inf)` integer | Model input |
| `st_vwap_reclaim_flag` | `1` if the two most recent VWAP crosses occurred within `vwap_reclaim_window` (300s) of each other — a failed cross. | No | `{0, 1}` | Model input |
| `st_level_touch_age_s` | Seconds since price was last within `touch_tol_pts` (2.0) of any of the prior session's five levels. | No | `[-1, +inf)` | Model input |
| `st_level_touch_kind` | Code for the most recently touched level: `lo=-3, val=-2, poc=0, vah=2, hi=3`, mirrored for side. | Yes | Discrete `[-3, 3]` | Model input |
| `st_level_touch_accept` | Signed `fs_accept_continue` reading sampled at the moment of that touch. | Yes | Approximately `[-1, 1]` or NaN | Model input |
| `st_level_touch_reject` | Signed `fs_reject_reversal` reading sampled at the moment of that touch. | Yes | Approximately `[-1, 1]` or NaN | Model input |

All 26 `st_` columns come back all-NaN for a session with no prior closed session in `session_levels.parquet`, or with a non-finite `r_value` — never a synthetic estimate.

## Label and Excluded Columns

`y_move` is computed once per `(decision timestamp, side)` by `src/training/move_label.py` scanning the print tape forward from the decision to `min(decision + horizon_seconds, flatten_time)`:

- `1` if the favourable-excursion threshold (`move_r · R`, currently `0.25 R` via `cascade_policy.MIN_TGT_R`) is reached **and** the adverse excursion measured only up to that point stays below `adverse_r · R` (`0.125 R` via `cascade_policy.STOP_RATIO × MIN_TGT_R`, [ADR-134](../decisions/ADR-134-cascade-ssot-structural-transitions-policy.md));
- `0` otherwise, including a move that clears the favourable threshold only after an earlier disqualifying adverse excursion.

The following are stored in the parquet but excluded from Stage 1 features because they are outcomes, identifiers, or absolute prices:

| Column | Why excluded | Later use |
|---|---|---|
| `y_move` | Training label, not an input. | XGBoost target |
| `reached` | Favourable threshold cleared, ignoring adverse excursion — kept separate from `y_move` because scoring a stop rule against `y_move` is circular (that label already caps adverse excursion). | Stop-rule probing (`probe_stop_rule.py`) |
| `t_move` | Seconds to the favourable threshold, `-1` if never reached. Future event time. | Uniqueness weighting (`label_uniqueness`), replay |
| `mfe_pts`, `mae_pts` | Maximum favourable/adverse excursion in points — future path outcomes. | Diagnostics, `_gate_stats` reporting |
| `price` | Absolute decision-bar price. | Blotter, replay, downstream Stage 3 entry price |
| `ts`, `session`, `side` | Row identity and directional contract. | Fold splits (purged by session/month), replay, side display |
| `r_value` | Risk scale used to normalise every other column; excluded from the feature list itself by name, separately from `LEAK`. | Geometry normalisation |
| `weight` | Uniqueness weight added at evaluation time, not present in the built parquet. | Sample weighting, effective sample size (`n_eff`) |

## Open Concerns

- **`fs_last_reversal_level` is a raw absolute price and is not curated out of Stage 1's feature list.** Stage 3's `CONTEXT` tuple deliberately excludes it (`docs/reference/STAGE3_FEATURES.md` marks it "carried metadata"), and the codebase elsewhere treats absolute price columns as regime memorisation to be removed (`tgt_level`/`stp_level` removal "improved every metric" on the superseded first-passage track). Stage 1's uncurated `sorted(c for c in df.columns if c not in LEAK ...)` selection does **not** exclude it, so it currently rides into the Stage 1 classifier as a live input. Unmeasured whether this matters in practice; flagged here because Stage 1 has no equivalent of Stage 3's explicit allowlist to catch it.
- **The `--no-order-flow` ablation is name-based, not lineage-based.** It drops every `fs_` column and any column containing `cvd`, `ofi`, or `delta`, which correctly removes all TBBO-derived reads, but also happens to leave `es_atr_r`/`es_realized_vol` (price-only) in either configuration — coincidentally correct today, not guaranteed if a future column is named without one of those substrings.
- **Feature count has grown across three rebuilds.** This document reflects the code as written under [ADR-131](../decisions/ADR-131-large-print-taxonomy-redesign.md)/[ADR-133](../decisions/ADR-133-ema-vwap-trend-agreement-features.md)/[ADR-134](../decisions/ADR-134-cascade-ssot-structural-transitions-policy.md), plus the tape-causality fixes and clustered-bound columns below. The last **measured** Stage 1 result (AUC 0.6627, ROADMAP 2026-09 entry) predates all of the above and used a smaller `es_`/`fs_` set with no `st_` columns at all.
- **Tape-causality fixes (`move_matrix_v3`).** `tape_to_path` floor-labels 1-second bars by their start, so a bar covering `[T, T+1)` is not actually complete at decision instant `T` — a bar's own close/high/low can carry a print up to ~1s in the future. `fs_clv`/`fs_ad_line`/`fs_ad_slope_*`/`fs_ad_price_divergence_*` were removed outright (CLV needs the bar's full range, not fixable by swapping one field). `fs_vwap_dist_r`, `fs_ema6_dist_r`, every `px`-based `es_` column (`es_vs_dev_*`, `es_vs_session_*`, `es_vs_prior_*`, `es_ret_*`, `es_gap_r`, `es_open_vs_prior_va`, `es_in_value_area`), and every `st_` column now read the raw print tape directly for "current price" rather than the bar frame. `es_atr_r`/`es_realized_vol` were separately rebuilt off completed 1-minute candles. `classify_large_prints()` (feeding `fs_accept_continue`/`fs_absorb_continue`/`fs_absorb_reversal`/`fs_reject_reversal` and their downstream `fs_flow_score*`/`fs_large_share_*`/`fs_last_reversal_level`) had a separate, non-causality bug fixed alongside this: it matched each print to the *next* bar over rather than the bar it actually printed in, because its own bar-matching logic assumed end-labelled bars against the actual start-labelled ones — so the OHLC used for body/wick classification was the wrong candle's. `fs_delta_*`/`fs_ofi_*`/`fs_delta_ratio_*`/`fs_delta_per_body_*`/`fs_absorption_*`/`fs_impact_efficiency_*` were then rebuilt off individual prints (causal running-total window-diffs) instead of the floor-labelled bar grid's `delta`/`ofi` columns, closing the same exposure diluted across their 60/300s windows.
- **Multi-session clustered bounds (`move_matrix_v4`).** `es_n_bounds`/`es_inside_bound*`/`es_channel_width_r`/`es_vs_bound_*`/`es_pos_in_channel` read the same KDE-clustered, multi-session `Bound` structure Stage 3's target geometry uses (`src/features/cash_session_profile.py`), via `src/training/entry_state.py`'s `_bound_state()`. Before this, Stage 1 only ever saw the single **immediately-preceding** session's five levels (`es_vs_prior_*`); it had no visibility into levels multiple sessions agreed on, nor any read of whether price has already tested/broken/rejected a level today. `build_move_matrix.py` threads `history_before(levels, session, lcfg.max_lookback_sessions)` (the same session-profile SSOT Stage 3's `build_target_matrix.py` reads) into `build_entry_state()` per session. **Performance note**: a persistently trending price can drift far enough that even `levels_for_price`'s own history-extension cannot bracket it; naively re-triggering a rebuild on every following row would each time repeat the `O(tape_length × bound_count)` interaction-matrix pass, which measured as a 300x+ slowdown (a single session went from ~1s to over 40 minutes before the fix). `_bound_state()` pins the escaped side's span to `±inf` once the lookback has maxed out and price is still beyond it, since no amount of further escape can change a result already built from every available session. **Measured**: Stage 1 weighted AUC 0.5257 (`move_v3`) → 0.5266 (`move_v4`); `es_pos_in_channel` reached #15 in the SHAP top-20 (mean\|SHAP\| 0.0076), the other new columns did not crack the top 20. Cascaded to Stage 3 (`target_matrix_v4`/`target_scores_v4.parquet`, same `CONTEXT` set — the new `es_` columns were not added to Stage 3's curated context): weighted AUC 0.8068 (`v3`) → 0.7989 (`v4`) on a near-identical sample (n_eff 1,559 → 1,578), attributable to the shift in which decisions Stage 1's gate selects rather than any change to Stage 3's own features.

## End-to-End Influence

```text
causal es_/fs_/st_ state (full column set, no curation)
                    ↓
             Stage 1 XGBoost
                    ↓
         p_move = P(y_move = 1)
                    ↓
     Stage 1 gate (τ quantile on p_move)
                    ↓
  gated decisions feed Stage 3's build_target_matrix.py
```

Stage 1's features influence only `p_move`. They do not alter the label, the side, or any candidate-target geometry — those are Stage 3's concern (`STAGE3_FEATURES.md`). A subset of Stage 1's own `es_`/`fs_`/`st_` values, measured at the same decision bar, is copied onto every Stage 3 row as `CONTEXT` so Stage 3 can see market state without seeing Stage 1's prediction (`p_move` itself is excluded from `CONTEXT` for exactly that reason).
