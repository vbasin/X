# ADR-130: Target-adaptive stops and Stage 2 fold-in

## Status
Accepted — 2026-09-15. Supersedes the fixed 0.45 R stop from
[ADR-127](ADR-127-tbbo-tape-and-cascade-architecture.md), Stage 2. The
cascade retains Stage 1 (move detector) and Stage 3 (target selector); the
former Stage 2 is no longer a separately-computed protective stop but a
formula applied per-target inside Stage 3.

## Date
2026-09-15

## Context

### 1. The fixed stop can be structurally larger than the target

ADR-127 chose a fixed 0.45 R stop after showing that structural stop rules
add ~1 pp of survival over a distance-matched random control. That result
justified the choice against structural stops, but not against
target-conditional stops. A nearest-bound target at 0.14 R paired with a
0.45 R stop is a **0.31 : 1** payoff — the trade cannot break even at
~50 % hit rate, and the historical run at `(τ_move ≥ 0.50, τ_reach ≥ 0.75)`
already sits at 67.9 %.

The published book (`+0.111 R / trade`, `t = 4.67`) held up statistically
because the model preferentially picks probable-and-tight targets, but the
expectancy is being suppressed by structurally unfavourable R:R on those
same tight targets.

### 2. The stop should scale with the target — but be capped

The user's proposal is `stop_dist_r = min(stop_ratio × tgt_dist_r, max_stop_r)`.
Two things fall out:

- **A 2:1 R:R floor.** With `stop_ratio = 0.5`, every trade whose target
  lies below the cap has a stop half the target's size — so R:R is 2:1
  regardless of how close the target is. A 0.25 R target risks 0.125 R;
  a 0.5 R target risks 0.25 R.
- **A hard risk cap.** With `max_stop_r = 0.25` R, any target beyond
  0.5 R saturates the stop. A 7 R target still only risks 0.25 R, which
  is the 28:1 R:R that the old fixed-stop label file could not represent.

Because the stop depends on which target row is being tested, the
per-decision-bar abstraction of ADR-127's Stage 2 no longer fits the row
grain. Stage 3 always was per-target; Stage 2 becomes a formula folded
into that same row.

### 3. Small-target trades are usually not worth taking at all

Even with an improved R:R, targets below ~0.25 R are dominated by spread
and slippage at MNQ tick sizes. A distinct `min_tgt_r` gate lets the
operating point be swept without re-training — the model still trains on
all candidates, and the WebUI (or any replay tool) filters at decision
time.

## Decision

1. **Stop is target-adaptive per row.** For every Stage 3 label row:
   `stop_dist_r = min(stop_ratio · tgt_dist_r, max_stop_r)` with defaults
   `stop_ratio = 0.5`, `max_stop_r = 0.25` R. Signed stop level:
   `stop_px = price − side · stop_dist_r · r_prior`.
2. **Stage 2 is folded into the Stage 3 build.** `scripts/features/build_target_matrix.py`
   no longer calls `stop_candidates` / `resting_stop`. The `StopConfig`
   dataclass and `src/training/stop_rule.py` stay in the codebase for the
   ADR-127 stop-rule probe (`scripts/features/probe_stop_rule.py`), but
   the cascade no longer imports them.
3. **`min_tgt_r` is a runtime gate.** Added as a first-class field in the
   WebUI request (`BacktestRun.min_tgt_r`) and forwarded through
   `datasources.run_backtest_quicklook` to `run_cascade_inmemory`. Applied
   in `_apply_gates` when `> 0`. Default is `0.25` R in the frontend.
4. **Both OOF variants are published.** The fixed-stop OOF stays at
   `data/sources/labels/target_scores.parquet` (unchanged). The
   target-adaptive OOF is written to
   `data/sources/labels/target_scores_adaptive.parquet` from a fresh label
   build in `data/sources/labels/target_matrix_adaptive/`. The WebUI
   discovers both via `_cascade_oof_paths()` and labels them explicitly so
   they can be A/B'd from the model dropdown.
5. **Exit reasons stay honest** (from ADR-127 follow-up on `t_stop`): each
   row still carries `t_reach` and `t_stop`; the WebUI adapter emits
   `target`, `stop`, or `timeout` and uses those seconds for both the trade
   ledger and the position-book busy interval.

## Consequences

**Measured on the fresh adaptive OOF (2025-09 → 2026-06, 214 sessions,
559,688 rows, same 11 geometry features, 3 folds):**

| metric | ADR-127 fixed 0.45 R | ADR-130 adaptive |
|---|---|---|
| weighted AUC | 0.857 | **0.819** |
| reach rate (all rows) | 0.234 | 0.171 |
| median target (R) | 2.70 | 2.59 |
| best-cell expectancy (R/trade) | +0.111 | +0.099 |
| median target / stop @ book | 13.50 / 16.09 pts | **11.75 / 5.25 pts** |
| **realised R:R at the book** | **0.84 : 1** | **2.00 : 1** |
| frictionless net (pts) | +14,080 (t 13.72) | +10,855 (t 13.24) |
| 1-tick slip + $1 RT (pts) | +10,127 (t 9.87) | +6,054 (t 7.38) |
| 2-tick slip + $1 RT (pts) | +8,150 (t 7.94) | +3,653 (t 4.45) |

The adaptive stop trades **less alpha per trade** (0.099 R vs 0.111 R) and
lower AUC (tighter stops fire more often, so reaches are rarer and the
label is harder), but **doubles realised R:R** and remains statistically
significant (t = 13.16 at the book). The cost sensitivity is the honest
picture: the smaller per-trade edge is more sensitive to friction, which
is exactly the trade-off the cap makes explicit.

**Verified on 2026-06-01, `p_reach ≥ 0.75, p_move ≥ 0.5, min_tgt_r = 0.25,
1 slot`**:

- 62 accepted trades (vs 97 under fixed 0.45 R at the same score gates)
- 25 targets, 37 stops, 0 timeouts
- net +97.44 pts, avg hold 6.1 bars (30 s), first target at 13:30:50 UTC
- the 13:32:00 UTC short still stops out honestly at 13:32:02 with a
  −4.12 pt loss (stop is now `0.5 × 0.168 R = 0.084 R ≈ 2.5 pts`, not
  the old 13.4 pts)

**Files:**
- `scripts/features/build_target_matrix.py` — adaptive stop formula, new
  `--stop-ratio` / `--max-stop-r` CLI, `StopConfig` import removed.
- `scripts/features/eval_target_model.py` — unchanged (features are the
  same, `stp_dist_r` is now a per-row column that varies with `tgt_dist_r`).
- `src/backtest/cascade_backtest.py` — `min_tgt_r` runtime gate,
  target-adaptive note in `weights_note`.
- `src/webui/app.py`, `src/webui/datasources.py`, `src/webui/static/app.js`,
  `src/webui/static/index.html` — new numeric input, request field,
  discovery of `target_scores*.parquet`, asset-revision bump.
- `data/sources/labels/target_matrix_adaptive/` (214 files),
  `data/sources/labels/target_scores_adaptive.parquet` (fresh).
- `data/sources/labels/target_matrix/` is left in a mixed state (~58
  sessions from the interrupted adaptive rebuild, ~156 sessions from the
  ADR-127 backfill). Nothing critical reads it — `target_scores.parquet`
  was written from a fully-consistent fixed-stop matrix before the
  interrupted rebuild. If the fixed-stop matrix is ever needed, it can be
  regenerated from tape via a temporary restore of the pre-ADR-130
  `build_target_matrix.py`.

**Open concerns:**
- **AUC drop is expected but should not be assumed harmless.** The stop
  distribution changed, and the label distribution changed with it. Any
  claim built on ADR-127's AUC does not carry forward.
- **Cost sensitivity is stricter.** At 2 ticks/side + $1 RT the t-stat
  drops to 4.45 (from 7.94 under fixed). Live promotion has to include a
  realistic slippage model.
- **The `stop_ratio` and `max_stop_r` are two more knobs.** Both are
  measured, not searched, but they add to the DSR/PBO burden already
  flagged in [ADR-122](ADR-122-structural-context-and-closed-research-loop.md).

## Alternatives considered

- **Keep fixed 0.45 R, apply `min_tgt_r` runtime gate only.** Cheapest
  change (no rebuild) but leaves the structural R:R inversion in place.
  The cost sensitivity above shows the fixed-stop path only survives at
  0.84 : 1 R:R because the model finds high-hit-rate close targets — that
  is not a robust edge.
- **Structural stop from the recent auction.** Measured in ADR-127; ~1 pp
  incremental over a distance-matched control, no evidence structure
  beats distance.
- **Search over `(stop_ratio, max_stop_r)` on the label file itself.** The
  ADR-127 stop-rule probe showed searching over the stop is how the
  incumbent overfit; two measured constants at a defensible ratio (2:1 R:R
  cap 0.25 R) is chosen deliberately over a grid.

## Related

- [ADR-127](ADR-127-tbbo-tape-and-cascade-architecture.md) — cascade
  architecture (Stage 2 fixed 0.45 R stop, now superseded).
- [ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md) — level
  geometry (unchanged, still Stage 3's target set).
- [ADR-122](ADR-122-structural-context-and-closed-research-loop.md) —
  research-loop discipline (DSR/PBO budget grows with the two new knobs).
