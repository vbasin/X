# Stage 3 Cascade Feature Reference

> **Scope**: the 53 numeric inputs used by `scripts/features/eval_target_model.py` for the curated Stage 3 target selector.
>
> **Current variant**: `target_scores_context.parquet`, built from `target_matrix_context/`.
>
> **Row grain**: one row per decision timestamp, side, and candidate target bound.
>
> **Causality**: market-context values are measured at or before the decision bar. Target geometry describes the candidate being scored. Future path outcomes are kept out of the model inputs.

## Impact Scope

| Scope | Meaning |
|---|---|
| **Model input** | Passed to the Stage 3 XGBoost classifier as a numeric feature. The model uses it to estimate `p_reach`. |
| **Label** | Used to construct `y_reach`; it is not a model input. |
| **Carried metadata** | Stored in the target/score parquet for later gates, analysis, or replay, but excluded from Stage 3 training. |
| **Runtime policy** | Used after scoring to select candidates, gate trades, or manage the book. |

Stage 3 trains `XGBClassifier` on `y_reach` only. `y_reach` is the dependent variable; all rows below are independent inputs unless marked otherwise. Sample weights are uniqueness weights derived from `t_reach` and the label horizon.

## Context Features

These 42 columns are copied from the Stage 1 move matrix onto each candidate target row. They expose the market state without exposing Stage 1's prediction.

### Auction State: `es_`

| Feature | Meaning and value determination | Directional? | Value range | Impact |
|---|---|---|---|---|
| `es_ret_5m` | Price return over the prior 5 minutes, normalized by the current `R` value and mirrored into the trade direction. | Yes | `(-inf, +inf)` | Model input |
| `es_ret_15m` | Price return over the prior 15 minutes, normalized by `R` and direction-mirrored. | Yes | `(-inf, +inf)` | Model input |
| `es_cvd_slope_5m` | Change in session cumulative volume delta over the prior 5 minutes, normalized by the CVD scale and mirrored for side. Positive means aggression supports the trade direction. | Yes | Approximately `[-2, 2]` after scale normalization | Model input |
| `es_delta_1m` | Rolling 60-second sum of signed aggressor volume (buy minus sell), normalized by the session's CVD scale and mirrored for side. | Yes | Approximately `[-2, 2]` after scale normalization | Model input |
| `es_delta_5m` | Same rolling signed aggressor volume sum over the trailing 300 seconds, normalized by the session's CVD scale and mirrored for side. | Yes | Approximately `[-2, 2]` after scale normalization | Model input |

`es_` values are causal entry-state measurements. Missing order-flow history remains missing rather than being replaced with a price-only aggression proxy.

### Flow and Trend State: `fs_`

| Feature | Meaning and value determination | Directional? | Value range | Used in Stage 3 model? | Impact |
|---|---|---|---|---|---|
| `fs_vwap_dist_r` | - `(current close - cumulative session VWAP) / R`.<br>- Positive means price is above VWAP in the absolute frame. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_vwap_velocity_r` | - Slope of the cumulative session VWAP itself over the trailing 60-second window (`vwap.diff(60) / R`), not the slope of `fs_vwap_dist_r`.<br>- Positive means VWAP is rising. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema6_slope_r` | - Slope of the six-bar 1-minute close EMA over the configured fast window, divided by `R`. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_vwap_velocity_5m_r` | - Slope of the cumulative session VWAP over the trailing 5 minutes (`vwap.diff(300) / R`). | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_vwap_velocity_15m_r` | - Slope of the cumulative session VWAP over the trailing 15 minutes (`vwap.diff(900) / R`). | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema20_slope_r` | - Slope of the 20-bar 1-minute close EMA over the trailing 60 seconds, divided by `R`. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema60_slope_r` | - Slope of the 60-bar 1-minute close EMA over the trailing 60 seconds, divided by `R`. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_vwap_ema_spread_r` | - `(EMA6 - VWAP) / R` — value difference between the 6-bar EMA and the cumulative session VWAP. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema20_vwap_diff_r` | - `(EMA20 - VWAP) / R`, the same value difference for the 20-bar EMA. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema60_vwap_diff_r` | - `(EMA60 - VWAP) / R`, the same value difference for the 60-bar EMA. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema6_slope_vwap_diff_r` | - `fs_ema6_slope_r - fs_vwap_velocity_r` — how much faster or slower the 6-bar EMA is moving than VWAP over the same 60-second window. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema20_slope_vwap_diff_r` | - `fs_ema20_slope_r - fs_vwap_velocity_r`, the same slope gap for the 20-bar EMA. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema60_slope_vwap_diff_r` | - `fs_ema60_slope_r - fs_vwap_velocity_r`, the same slope gap for the 60-bar EMA. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema6_slope_vwap_diff_accel_r` | - 60-second slope of `fs_ema6_slope_vwap_diff_r` itself — whether the EMA6-vs-VWAP momentum gap is widening or closing. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema20_slope_vwap_diff_accel_r` | - 60-second slope of `fs_ema20_slope_vwap_diff_r`. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_ema60_slope_vwap_diff_accel_r` | - 60-second slope of `fs_ema60_slope_vwap_diff_r`. | Yes | `(-inf, +inf)` | Yes | Model input |
| `fs_accept_continue` (1) | - Qualifying print: TBBO `size >= 20` contracts by default.<br>- Up candle: buy print in the body; the buyer lifted the offer/paid the ask.<br>- Down candle: sell print in the body; the seller hit the bid.<br>- Contribution per print = `sign(candle) * (size / large_order_size) * (body_range / full_range)`; a continuation read on a nearly-all-body candle counts more than one on a mostly-wick candle.<br>- Summed over qualifying prints, rolled over the trailing 60 seconds. | Yes | `(-inf, +inf)` | **No** | Carried metadata (diagnostic; folded into `fs_flow_score`) |
| `fs_absorb_continue` (2) | - Qualifying print: TBBO `size >= 20` contracts by default.<br>- Up candle: sell print in the **bottom half** of the candle (candle low through the body midpoint — this spans the entire lower wick plus the lower half of the body).<br>- Down candle: buy print in the top half (body midpoint through the candle high).<br>- Contribution per print = `sign(candle) * (size / large_order_size) * (body_range / full_range)`.<br>- Summed over qualifying prints, rolled over the trailing 60 seconds. | Yes | `(-inf, +inf)` | **No** | Carried metadata (diagnostic; folded into `fs_flow_score`) |
| `fs_absorb_reversal` (3) | - Qualifying print: TBBO `size >= 20` contracts by default.<br>- Up candle: buy print in the upper wick (above the candle body).<br>- Down candle: sell print in the lower wick (below the candle body).<br>- Contribution per print = `-sign(candle) * (size / large_order_size) * min(wick_range / body_range, max_reversal_scale)`; the ratio is capped (default 10.0) so a near-zero body cannot blow the scale up unbounded.<br>- Summed over qualifying prints, rolled over the trailing 60 seconds. | Yes | `(-inf, +inf)` | **No** | Carried metadata (diagnostic; folded into `fs_flow_score`) |
| `fs_reject_reversal` (4) | - Qualifying print: TBBO `size >= 20` contracts by default.<br>- Up candle: sell print in the upper wick.<br>- Down candle: buy print in the lower wick.<br>- Contribution per print = `-sign(candle) * (size / large_order_size) * min(wick_range / body_range, max_reversal_scale)`, same cap as `fs_absorb_reversal`.<br>- Summed over qualifying prints, rolled over the trailing 60 seconds. | Yes | `(-inf, +inf)` | **No** | Carried metadata (diagnostic; folded into `fs_flow_score`) |
| `fs_flow_score` (5) | - Sum of features 1-4 (`fs_accept_continue + fs_absorb_continue + fs_absorb_reversal + fs_reject_reversal`) over the trailing 60 seconds — one composite net-signed flow-quality reading per bar. | Yes | `(-inf, +inf)` | **Yes** | Model input |
| `fs_flow_score_5bar` (6) | - The same composite signal (features 1-4 combined) rolled over the trailing 300 seconds — 5 x the `fs_flow_score` window, i.e. the aggregate of the last 5 one-minute `fs_flow_score` reads. | Yes | `(-inf, +inf)` | **Yes** | Model input |
| `fs_last_reversal_level` (7) | - Raw execution price of the most recent print classified as `fs_absorb_reversal` or `fs_reject_reversal`.<br>- Forward-filled while within the trailing 300 seconds (5 bars); reset to NaN once older than that. | No — this is an absolute price, never sign-mirrored for shorts | Raw instrument price or NaN | **No** | Carried metadata (reserved for future entry/stop level logic) |

**Naming**: numbers in parentheses correspond to the discrete feature list used when this taxonomy was specified; only features 5 and 6 are exposed to the Stage 3 classifier today. Features 1-4 remain queryable on the same parquet for diagnostics, and feature 7 is a level carried for future use, not yet consumed by any gate or model.


### Structural Transitions: `st_`

The structural module uses the **prior-session profile** — the single previous closed cash session's `lo`, `val`, `poc`, `vah`, `hi` — never the multi-session clustered bounds described in [Target and Level Geometry](#target-and-level-geometry). A 2-point touch tolerance and causal 1-second path state apply throughout. Signed values are mirrored for the trade side. A value of `-1` generally means the event has not occurred or has no valid age.

| Feature | Meaning and value determination | Directional? | Value range | Impact |
|---|---|---|---|---|
| `st_zone` | Ordinal location relative to the **prior-session profile**: outside low/high, below/inside/above value, or around POC. Codes are symmetric and side-mirrored. | Yes | Discrete `[-3, 3]` codes | Model input |
| `st_zone_dwell_s` | Wall-clock seconds spent continuously in the current structural zone. | No | `[0, +inf)` seconds | Model input |
| `st_break_age_s` | Seconds since price crossed through a **prior-session** value-area edge (`VAH` upward or `VAL` downward). | No | `[-1, +inf)`; `-1` means no active break | Model input |
| `st_break_level` | Direction of the active **prior-session** value-area break: positive upward, negative downward, zero with no active break. | Yes | Discrete `{-1, 0, 1}` | Model input |
| `st_break_extension_r` | Current signed distance beyond the broken **prior-session** value edge, divided by `R`. | Yes | `(-inf, +inf)` | Model input |
| `st_break_pullback_r` | Maximum extension since the break minus current extension, divided by `R`; measures retracement from the extreme. | No | `[0, +inf)` | Model input |
| `st_reclaim_flag` | Whether price has crossed back through the broken edge after the break. | No | `{0, 1}` | Model input |
| `st_reject_flag` | Whether that break was undone within the 300-second rejection window. | No | `{0, 1}` | Model input |
| `st_bounce_confirm` | Geometric break-retrace-continuation state: price returned to the broken edge within tolerance and then extended farther in the break direction. | No | `{0, 1}` | Model input |
| `st_bounce_accept` | Same bounce state, additionally confirmed by signed `fs_accept_continue` flow in the break direction. NaN when flow confirmation is unavailable. | No | `{0, 1}` or NaN | Model input |
| `st_va_entry_age_s` | Seconds since price most recently entered the **prior-session** value area from outside. | No | `[-1, +inf)`; `-1` means no entry | Model input |
| `st_va_entry_depth_r` | Signed distance traveled into the **prior-session** value area since that entry, divided by `R`. | Yes | `(-inf, +inf)`; NaN before entry | Model input |
| `st_vwap_cross_age_s` | Seconds since the most recent causal cross of session VWAP. | No | `[-1, +inf)`; `-1` means no cross | Model input |
| `st_vwap_cross_strength` | Signed VWAP-relative velocity at the most recent cross, divided by `R`. | Yes | `(-inf, +inf)`; NaN before cross | Model input |
| `st_vwap_reclaim_flag` | Whether two VWAP crosses occurred within the 300-second reclaim window. | No | `{0, 1}` | Model input |
| `st_level_touch_kind` | Code for the most recently touched **prior-session profile** level: `lo=-3`, `val=-2`, `poc=0`, `vah=2`, `hi=3`, mirrored for side. | Yes | Discrete `[-3, 3]` codes | Model input |
| `st_level_touch_accept` | Signed acceptance flow recorded at the most recent **prior-session-level** touch. | Yes | Approximately `[-1, 1]` or NaN | Model input |
| `st_dist_prior_hi_r` | Current price minus the **prior-session** high, divided by `R`. | Yes | `(-inf, +inf)` | Model input |
| `st_dist_prior_lo_r` | Current price minus the **prior-session** low, divided by `R`. | Yes | `(-inf, +inf)` | Model input |

These features are the direct path/state additions intended to distinguish entering value, crossing through value, breaking out, reclaiming a broken edge, rejecting it, and bouncing into continuation. They can affect `p_reach`, but they do not change candidate generation or `y_reach`.

## Target and Level Geometry

These 11 columns describe the candidate **clustered structural bound** — built from up to `lookback_sessions` (default 20, extendable to `max_lookback_sessions` = 120) historical closed sessions' `lo/val/poc/vah/hi`, KDE-clustered and buffered in `cash_session_profile.py` — and its relationship to the decision/fill price. This is a materially different level source from the single **prior-session profile** the `st_` features above use. They are deliberately retained because Stage 3 must distinguish a near target from a distant target. Raw absolute target points are excluded; risk-normalized geometry is retained.

| Feature | Meaning and value determination | Directional? | Value range | Impact |
|---|---|---|---|---|
| `inside_bound` | `1` when the entry price is inside a **clustered, multi-session structural bound** (not the prior-session profile above), otherwise `0`. | No | `{0, 1}` | Model input |
| `n_bounds` | Number of available **clustered structural bounds** in the current multi-session level history/lookback. | No | `[0, +inf)` integer | Model input |
| `stp_dist_r` | Candidate-specific stop distance in `R`: `min(0.5 * tgt_dist_r, 0.25)`. | No | `[0, 0.25]` | Model input; also defines label barrier |
| `tgt_dist_r` | Absolute distance from entry to the candidate target edge, divided by `R`. | No | `[0, +inf)` | Model input |
| `tgt_frac_poc` | Fraction of the target bound's contributing levels that were POC levels. | No | `[0, 1]` | Model input |
| `tgt_frac_value_edges` | Fraction of contributing target levels that were VAH or VAL. | No | `[0, 1]` | Model input |
| `tgt_n_levels` | Number of historical session levels contributing to the target bound. | No | `[1, +inf)` integer | Model input |
| `tgt_newest_age` | Age, in sessions, of the newest contributing level in the target bound. | No | `[0, 119]` sessions under the 120-session cap | Model input |
| `tgt_rank` | Candidate order in the trade direction: nearest eligible bound is `0`, then `1`, through the maximum target count. | No | `{0, 1, 2, 3, 4}` with five-target default | Model input |
| `tgt_strength` | Recency-decayed, kind-weighted KDE mass assigned to the target bound. VA edges weigh more than POC, which weighs more than extremes. | No | `[0, +inf)` | Model input |
| `tgt_width_r` | Target bound width divided by `R`. Bound edges come from observed levels plus the configured point buffer. | No | `[0, +inf)` | Model input |

The target geometry comes from the cash-session profile and KDE bound builder. It is not future information: the bound history is restricted to closed prior sessions and the candidate is known at the decision bar.

## Label and Excluded Columns

`y_reach` is computed separately for every `(timestamp, side, target)` row by scanning the TBBO print tape from the entry/fill through the 1,800-second horizon:

- `1` if the target is touched before its candidate-specific stop;
- `0` if the stop is touched first or neither barrier is touched before the horizon.

The following are stored in the parquet but excluded from Stage 3 features because they are outcomes, identifiers, absolute prices, or the other cascade stage's prediction:

| Column/group | Why excluded | Later use |
|---|---|---|
| `y_reach` | Training label, not an input. | XGBoost target |
| `t_reach`, `t_stop` | Future event times. | Exit timing and book occupancy |
| `t_fill`, `decision_price` | Resting-entry outcome or pre-fill price. | Fill timing/replay; `t_fill` is absent from immediate-entry semantics |
| `price`, `stop_px`, `tgt_level` | Absolute entry/barrier prices and replay fields. | Blotter and P&L |
| `ts`, `session`, `side` | Row identity and directional contract. | Fold splits, replay, side display |
| `r_value`, `weight` | Risk scale and derived uniqueness weight. | Geometry, training weighting, effective sample size |
| `p_move` | Stage 1 prediction. | Runtime Stage 1 gate and joint frontier; intentionally not a Stage 3 input |

## End-to-End Influence

```text
causal es_/fs_/st_ state + target geometry
                    ↓
             Stage 3 XGBoost
                    ↓
        p_reach = P(y_reach = 1)
                    ↓
  p_reach gate + candidate selection + position book
                    ↓
       WebUI blotter / replay outcome
```

The features influence the model only through `p_reach`. They do not directly alter the label, side, rank, target set, or stop policy. The current replay selector chooses the surviving candidate with the highest `p_reach` per timestamp; `p_move` is applied as a separate Stage 1 gate before that selection.

