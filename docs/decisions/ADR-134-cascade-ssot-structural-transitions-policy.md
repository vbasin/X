# ADR-134: Session-profile SSOT, structural-transition features, and cross-stage policy reconciliation

## Status
Accepted — 2026-09-17. Extends the active `mnq-auction-cascade` track
([ADR-127](ADR-127-tbbo-tape-and-cascade-architecture.md),
[ADR-130](ADR-130-target-adaptive-stops.md)). Amended same-day by
[ADR-131](ADR-131-large-print-taxonomy-redesign.md) and
[ADR-133](ADR-133-ema-vwap-trend-agreement-features.md), both of which
build on the `CONTEXT` shape this ADR introduces. **Code-only**:
`data/sources/labels/move_matrix*` and `target_matrix*` still reflect the
prior builder and do not yet carry `st_` columns or a resting-entry Stage
3. Rebuild is deliberately deferred until ADR-131/133's changes land in
the same pass, per the user's explicit instruction.

## Date
2026-09-17

## Context

### 1. Every parallel shard re-derived the same session history

`build_move_matrix.py` and the earlier `probe_stop_rule.py` each rebuilt
every prior session's volume/delta profile inside their own session loop
and carried it forward in a `deque`, so a worker handed sessions 40-60
still had to replay sessions 1-39 first just to get a valid 20-session
lookback. That work is identical across every shard and every builder
that needs it — it belongs in one place, computed once.

### 2. A signed distance cannot express a path

The `es_`/`fs_` feature families answer "how far" and "how fast," but
nothing answered "what happened on the way here." `es_vs_prior_vah`
reads the same whether price is breaking down through the value-area
high or rallying back into it from above — a signed distance is
symmetric under the exact ambiguity a trader resolves by remembering the
sequence of events. Distinguishing entering value from outside, breaking
out, retracing to the broken edge and continuing, failing back through
it, and crossing VWAP with force requires state that persists across
bars: age since an event, and which of several mutually exclusive things
happened first.

### 3. Stage 1 and Stage 3 disagreed about what counted as the same trade

Before this change, three thresholds were unrelated literals:
`MoveConfig.move_r = 0.25`, `MoveConfig.adverse_r = 0.10` (Stage 1's move
label), and `min_tgt_r = 0.25` / `stop_ratio = 0.5` (Stage 3's target
policy, [ADR-130](ADR-130-target-adaptive-stops.md)). Stage 1 qualified a
move as clean if it stayed within 0.10 R adverse excursion; Stage 3, for
that same 0.25 R target, would have stopped it out at 0.125 R
(`0.5 × 0.25`). A move Stage 1 called clean could already have been
stopped by the stop Stage 3 actually trades with, and a move Stage 1
rejected as too adverse might never have touched Stage 3's wider
effective stop. The two stages were scoring different trades while being
composed as if they were the same one.

### 4. Stage 3 always assumed a fill at the decision close

Every Stage 3 row entered at `price` — the tape price at the decision
timestamp — with no wait and no possibility of not filling. The
superseded `mnq-auction-firstpass` track already solved this problem
differently: [ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md)
rests an order at the far edge of the containing bound and emits no row
at all if it never fills, because a trade that did not happen is not a
flat one. That geometry argument applies identically to Stage 3's
clustered bounds, but nothing had ported it over.

## Decision

### `src/features/session_profile.py` — persisted per-session profile SSOT

1. Two artifacts, built once by `scripts/features/build_session_profiles.py`
   (parallel by date range, `--workers`):
   - `session_levels.parquet` — one row per session: `lo/val/poc/vah/hi`,
     open, close, volume, delta, `n_prints`, `va_width_pts`, `shape`
     (D/P/b/double via `profile_shape()`), `r_value`. A few hundred rows,
     so a 20-session lookback is one read instead of 20 tape rebuilds.
   - `profile_<session>.parquet` — the full volume/delta-at-price
     histogram at the native 0.25-point NQ tick (`tick_histogram()`).
2. Levels come from `src.features.tape.tape_levels`, so they carry
   ADR-129 tick parity with the WebUI — not the bar-smeared histogram in
   `src.features.prior_volume_profile`, which is the approximation
   ADR-127 retired for this substrate.
3. Read-side helpers give every consumer the same primitives:
   `load_levels()`, `history_before(levels, session, n)` (newest-first,
   shaped for `cash_session_profile.build_bounds`'s recency decay),
   `prior_row()` (single immediately-preceding session), `r_before()`
   (median `r_value` over `n` prior sessions, NaN until the window is
   full — the same ATR-like current-conditions reading ADR-126
   established, now shared rather than reimplemented per builder).
4. `write_levels()` merges by session key (newest build wins), so a
   partial or incremental run never duplicates a session row.

### `src/features/structural_transitions.py` — `st_` path-state features (19 columns)

1. Computed against the **single immediately-preceding closed cash
   session's** five levels (`SessionLevels`), deliberately not the
   multi-session KDE-clustered bounds Stage 3's target geometry uses —
   different question (what happened relative to yesterday vs. which
   level is a viable target).
2. Three families, all causal on the 1-second path from
   `tape_to_path`:
   - **Zone and dwell** — `st_zone` (ordinal, symmetric about POC,
     side-mirrored), `st_zone_dwell_s`.
   - **Break state machine** — a break is a cross through VAH or VAL in
     either direction. Tracks `st_break_age_s`, signed `st_break_level`,
     `st_break_extension_r`, `st_break_max_extension_r`,
     `st_break_pullback_r`, `st_reclaim_flag`/`_age_s`, `st_reject_flag`
     (undone within `reject_window_s = 300`), and a bounce pair —
     `st_bounce_confirm` (retrace to the broken edge within
     `bounce_tol_pts` then extend further) and `st_bounce_accept` (same,
     confirmed by signed acceptance flow in the break direction; NaN
     when no flow taxonomy was supplied).
   - **VWAP cross** and **level touch** — age/side/strength of the most
     recent causal VWAP cross, a reclaim flag for two crosses inside
     `vwap_reclaim_window`, and the age/kind/accept/reject reading at the
     most recently touched level (`touch_tol_pts = 2.0`).
3. `accept`/`reject` parameters take the **signed** `fs_accept_continue`
   / `fs_reject_reversal` series (ADR-131 naming) on the same index, used
   only to tag *which* structural touches were flow-confirmed — this
   module does not compute flow itself.
4. `tests/test_structural_transitions.py` hand-constructs price paths for
   the break/failed-break/retrace-and-continue distinction specifically,
   since all three look identical to a signed distance and only differ in
   what price did after crossing an edge.

### `src/training/cascade_policy.py` — single source of truth for cross-stage geometry

1. Stage 3's target policy becomes the input, not a parallel literal:

   ```
   move_r    = min_tgt_r                 the smallest target worth taking
   adverse_r = stop_ratio * min_tgt_r    the stop that target would carry
   horizon   = shared with Stage 3
   ```

2. `MoveConfig.move_r`/`adverse_r`/`horizon_seconds`/`decision_grid_seconds`
   in `src/training/move_label.py` now default from `cascade_policy`
   instead of restating the numbers: `move_r = 0.25` (unchanged value,
   now `MIN_TGT_R`), `adverse_r = 0.125` (**changed** from 0.10 — derived
   as `STOP_RATIO * MIN_TGT_R`, not chosen independently).
3. `src/backtest/cascade_backtest.py` reads `policy.MIN_TGT_R` for its
   runtime gate default instead of restating `0.25`.
4. `describe()` renders one line (`move_r=... adverse_r=... min_tgt_r=...
   stop_ratio=... max_stop_r=... horizon=...s grid=...s`) so any rebuild
   log records the exact cross-stage geometry it ran under.

### Stage 3 resting entry (`scripts/features/build_target_matrix.py`)

1. New opt-in `--resting-entry` flag (default off — the fixed-fill
   behaviour ADR-127/130 measured against is unchanged unless passed)
   with `--max-fill-wait` (default 900s, matching ADR-126's
   `max_fill_wait_seconds`).
2. When enabled: the order rests at the far edge of the containing bound
   (`entry_levels()`, same primitive ADR-126 introduced), and a decision
   whose order does not fill within the wait window emits **no row** —
   population changes, not just values, matching ADR-126's reasoning
   that an unfilled order is not a flat trade.
3. All downstream geometry — `tgt_dist_r`, `stp_dist_r`, `t_reach`,
   `t_stop` — is restated against the fill price, not the decision
   price. `t_fill` and `decision_price` are added to the row and to the
   Stage 3 leak set (never model inputs).
4. `src/backtest/cascade_backtest.py`'s `_hold_seconds()` gains an
   optional `t_fill` term so the position book blocks for
   `t_fill + t_reach`/`t_stop`, not just the post-fill hold — capital
   commits at the decision bar, exactly as ADR-126 established for the
   other track.
5. `tests/test_target_matrix_entry.py` pins the two failure modes most
   likely to regress silently: an order that never fills must drop its
   row entirely, and geometry must be measured from the fill, not the
   decision price.

### `CONTEXT` in `build_target_matrix.py`

`CONTEXT_PREFIXES` becomes `("es_", "fs_", "st_")`. Before ADR-131/133,
`CONTEXT` was 28 columns: 3 base `fs_` + 3 `es_` + 3 raw large-print `fs_`
states + 19 `st_` (this ADR). ADR-131 nets that to 27 (raw states →
2 composites); ADR-133 takes it to 42. The 19 `st_` columns introduced
here are present at every point in that chain and are documented in full
in [docs/reference/STAGE3_FEATURES.md](../reference/STAGE3_FEATURES.md).

## Consequences

**Not yet measured against outcomes.** Like ADR-131/133, this changes
construction only. No `move_matrix`/`target_matrix` has been rebuilt with
`st_` columns, `cascade_policy`-derived thresholds, or resting entry; no
model has trained against any of it. The `adverse_r` change from 0.10 to
0.125 alone changes Stage 1's move label population and must be
re-evaluated, not assumed neutral.

**Files:**
- `src/features/session_profile.py`, `scripts/features/build_session_profiles.py` — new.
- `src/features/structural_transitions.py`, `tests/test_structural_transitions.py` — new.
- `src/training/cascade_policy.py` — new.
- `scripts/features/build_move_matrix.py` — session-profile SSOT read path
  replaces per-shard warmup; `build_structural_transitions()` wired in and
  merged onto the move matrix; `ST_DIRECTIONAL` added to the short-side
  mirror.
- `scripts/features/build_target_matrix.py` — `cascade_policy` import;
  `CONTEXT` gains the 19 `st_` columns; `--resting-entry`/`--max-fill-wait`
  CLI, `t_fill`/`decision_price` columns, population-changing fill logic.
- `src/training/move_label.py` — `MoveConfig` defaults sourced from
  `cascade_policy` (`adverse_r` value changes 0.10 → 0.125).
- `src/backtest/cascade_backtest.py` — `min_tgt_r` default from
  `cascade_policy`; `_hold_seconds()` accepts an optional fill offset.
- `tests/test_target_matrix_entry.py` — new.
- `docs/reference/STAGE3_FEATURES.md` — new; documents the full current
  `es_`/`fs_`/`st_` CONTEXT plus target geometry and excluded columns.

**Open concerns / deferred work:**
- Rebuild `move_matrix` and `target_matrix` (context + context_rest
  variants) with `st_` columns, the corrected `adverse_r`, and (as a
  separate sweep) `--resting-entry`, then retrain and re-evaluate Stage 1
  and Stage 3 before any promotion decision — deferred until ADR-131 and
  ADR-133's changes are combined into the same rebuild, per the user's
  explicit instruction.
- Whether `adverse_r = 0.125` changes Stage 1's measured AUC/precision
  frontier is unmeasured — this is a label-population change, not purely
  additive like the feature changes in ADR-131/133.
- Whether resting entry helps or hurts Stage 3 (it removes unfillable
  candidates but adds wait-time risk and shrinks the row population) is
  unmeasured; the flag defaults off specifically so the fixed-fill
  baseline stays reproducible until that comparison is run.
- `st_bounce_accept`/`st_level_touch_accept`'s value depends on which
  flow taxonomy is passed as `accept`/`reject` — currently
  `fs_accept_continue`/`fs_reject_reversal` (ADR-131 names). If the
  taxonomy changes again, the call site in `build_move_matrix.py` must
  change with it.

## Related

- [ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md) — resting
  entry and `max_fill_wait_seconds`, originated on the superseded
  first-passage track, ported here to Stage 3.
- [ADR-127](ADR-127-tbbo-tape-and-cascade-architecture.md) — cascade
  architecture; `MoveConfig.adverse_r` origin.
- [ADR-129](ADR-129-tick-level-session-profile-parity.md) — tick parity
  that `session_profile.py`'s levels inherit via `tape_levels`.
- [ADR-130](ADR-130-target-adaptive-stops.md) — `min_tgt_r`/`stop_ratio`,
  now the source of truth `cascade_policy.py` derives Stage 1's
  thresholds from.
- [ADR-131](ADR-131-large-print-taxonomy-redesign.md),
  [ADR-133](ADR-133-ema-vwap-trend-agreement-features.md) — same-day
  `CONTEXT` amendments layered on top of the 19 `st_` columns this ADR
  introduces; all three share one deferred rebuild.
