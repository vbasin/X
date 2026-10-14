# ADR-126: KDE level grouping, R decoupling and resting-order entry for `mnq-auction-firstpass`

## Status
Accepted — 2026-09-11. Amends the Layer 0/1/2 construction defined in
[ADR-123](ADR-123-first-passage-candidate-exit-surface.md). All first-passage
labels and every result derived from them **before 2026-09-11** are superseded:
the label rows themselves changed, so metrics are not comparable across the
boundary.

## Date
2026-09-11

## Context

### 1. Single-linkage clustering produced zones wider than a trading session

[ADR-123](ADR-123-first-passage-candidate-exit-surface.md) established
volatility-scaled clustering (`cluster_r = 0.50`) after a fixed 5-point merge
left 20 sessions of level history as 78 bounds with the next target 0.25 R away.
That fixed the *starvation* symptom but not the mechanism.

`build_bounds` merges on a max-gap threshold, which is **single linkage** and
therefore transitive: A joins B and B joins C regardless of the distance from A
to C. With a 20–30 session lookback the mean spacing between the 5 levels per
session falls well below `cluster_pts`, and long chains form.

Audited on 2026-06-25 (R = 74.75, `cluster_pts` = 37.375):

| | bounds | median width | max width |
|---|---|---|---|
| single linkage, uncapped | 21 | 51.2 pts | **435.4 pts (5.83 R)** |

The 435-point bound held 45 member levels spanning 2026-05-08 → 2026-06-24. Its
largest internal gap was 32.00 pts against a 37.375-pt threshold — one member
short of breaking the chain. A zone 5.8× the median session range is not a level;
it is a record of everywhere price has been, and its `strength` and `n_levels`
descriptors are correspondingly meaningless. Cluster identity was also fragile:
removing any single intermediate level would have split the zone in two.

Shrinking `cluster_r` does not fix this. It relocates where chains happen to
break, trading long chains for a mix of long chains and singletons.

### 2. Entry filled at the decision close, inside the magnet

`label_session` set `entry_price = close[entry_i]`. A decision bar is usually
*inside* a bound, and inside a zone the auction is rotating rather than
travelling — the level is a magnet. Filling at market in the middle of a rotation
is the worst available price and assumes the trade is worth taking wherever price
happens to be when the model fires.

This also made the label answer a question nobody trades: "if I buy right here,
does the target come before the stop", rather than "if price leaves this level in
my direction, does the target come before the stop".

### 3. R was doing three unrelated jobs

[ADR-123](ADR-123-first-passage-candidate-exit-surface.md) scaled the merge
distance and the buffer by R (`cluster_r`, `buffer_r`). That conflates two
questions that are not the same question:

- **Is this the same level?** A property of price and tick size. Two prints 3
  points apart on NQ are 3 points apart whether R is 30 or 150.
- **Is this level far enough away to be worth aiming at?** A property of
  volatility, and genuinely R's job.

Tying zone geometry to R also made the whole bound set rebuild whenever R moved,
and coupled level identity to the price regime: the same structure resolved
differently at NQ 12,000 and NQ 60,000.

### 4. Three lookbacks, only one of them deliberate

`LevelConfig.lookback_sessions` defaulted to 20 as intended, but the
`build_first_passage.py` CLI defaulted to **30** and overrode it, and R came from
`PathLabelConfig.vol_lookback_sessions`, which defaults to **5**. The level
lookback and the R lookback are unrelated concepts — how much market memory to
read, versus how recent an ATR-like yardstick should be — and were being set by
one number in some places and silently by another elsewhere.

## Decision

### Bounds are KDE peaks, with edges taken from the grouped levels (Variant B)

Grouping is a two-step job:

1. **The density decides grouping.** Each level is deposited on a `grid_step`
   (0.25 pt = 1 MNQ tick) grid as a point mass of
   `LEVEL_WEIGHTS[kind] * 0.5 ** (age / recency_halflife)`, and the sum is
   convolved with a Gaussian of width `sigma_pts`. Every local maximum is a bound
   centre.
2. **Each level joins its nearest peak by price**, and the bound spans
   `min(member prices) .. max(member prices)`, widened by `buffer_pts` and
   clipped at the midpoint of each gap so bounds stay disjoint.

Variant B was chosen over taking the peak's FWHM as the interval (Variant A)
because it preserves the invariant that **bound edges are real observed levels**.
An edge at the half-maximum of a smoothed density is an artefact of the
bandwidth; an edge at a prior session's VAH is a price the market actually
turned at, and the whole point of the level set is that a trader could audit it.
`strength`, `n_levels` and the kind breakdown remain plain sums over the assigned
members, unchanged in meaning from gap merging.

Chaining is eliminated because the grouping decision is now **local**: a level
joins the peak it is nearest to. Two distant clusters cannot fuse through a run
of intermediates, because the intermediates belong to their own peaks.

### `sigma_pts` is the single resolution knob, and it is in points

Nothing in `LevelConfig` is a function of R any more; `scaled()` is gone. Every
length is in index points, so the same code runs unchanged at NQ 12,000 and NQ
60,000 and at R = 30 or R = 150. R now reaches only `bound_candidates` and
`bound_features`, which decide how far away a level is worth aiming at.

`sigma_pts` sets what a bound *means*. Measured on 2026-06-25 (100 levels from 20
sessions, R = 74.75):

| σ (pts) | bounds | singletons | ≥3 members | max width | max/R | price inside a zone |
|---|---|---|---|---|---|---|
| 1.0 | 90 | 80 | 0 | 4.0 | 0.05 R | no |
| **5.0** | **62** | **35** | **9** | **22.5** | **0.30 R** | sometimes |
| 8.0 | 46 | 17 | 14 | 47.0 | 0.63 R | yes |
| 12.0 | 36 | 12 | 17 | 86.8 | 1.16 R | no |

At σ = 1.0 a bound is one level, multi-session agreement never registers, and the
resting-entry rule below degenerates to its fallback because price is essentially
never within ±1 pt of a prior level. By σ = 12.0 zones are wide enough to
reintroduce the width problem that killed gap merging. **σ = 5.0** is the default:
enough smoothing for genuine agreement to form a zone, narrow enough that no
bound approaches a session's range. It is the first parameter in the sweep.

Rebuild cost is 2.3 ms for 20 sessions of history, so resolution is no longer
traded against build time.

### Entry is a resting order at the far edge of the containing bound

New `entry_levels(bounds, price, r_value, cfg)` returns the `(long, short)`
prices at which the order rests. Inside a bound that is the **far edge** in the
direction of the trade — the first price at which the level has demonstrably
failed to hold. Between bounds there is no magnet to clear, so the order rests
`entry_offset_r = 0.05` away, enough to require the move to continue rather than
to reverse into the fill.

`label_session` takes `entry_levels_by_entry` and:

- finds the fill offset with `first_fill` (high for a resting buy, low for a
  resting sell), bounded by `max_fill_wait_seconds = 900`;
- **emits no row when the order does not fill** — an unfilled order is a trade
  that never happened, not a flat one, and scoring it as a zero would dilute the
  conditional expectation being fitted;
- runs the first-passage race from the **fill** bar at the **fill** price;
- re-splits candidates into targets and stops against the fill price and restates
  `tgt_dist_*` / `stp_dist_*` against it, so the geometry the model sees is the
  geometry the trade actually had;
- drops candidates within `min_dist_r = 0.10` of the fill, which the offset
  between decision price and fill price can otherwise create.

Features remain measured at the **decision** bar. The decision is taken there and
nothing looks ahead; only the fill moves. `t_fill`, `fill_ts` and
`decision_price` are outcome columns and are in the leak set of both
`eval_first_passage_model.py` and `explain_trade.py`.

The 1-position book now blocks for `t_fill + t_event`: capital is committed at
the decision, not at the fill.

### Separate candidate budgets for structural levels and the ladder

The shared per-side cap ranked structural levels above the ladder
(`is_structural * 1000 + strength`). Once KDE grouping made bounds plentiful,
that silently evicted the entire ladder on rich sides and kept an arbitrary
tie-broken remnant on others. Structural and ladder candidates now get
`max_per_side` each, because they do different jobs: the bounds supply levels the
auction agrees on, the ladder keeps stop geometry free, and the `is_structural`
flag is the contrast that lets the model discover which matters
([ADR-123](ADR-123-first-passage-candidate-exit-surface.md) required that flag to
be learnable, not constant).

### The two lookbacks are separated, and CLI defaults derive from the model

`lookback_sessions` (how much market memory builds the levels) and the R lookback
(how recent the volatility yardstick is) are now distinct settings:

| | owner | default |
|---|---|---|
| level lookback | `LevelConfig.lookback_sessions` | **20** |
| R lookback | `PathLabelConfig.vol_lookback_sessions` | **5** |

`build_first_passage.py` gains `--r-lookback-sessions`, and **both CLI flags take
their defaults from the model classes** (`default=LevelConfig.lookback_sessions`,
`default=PathLabelConfig.vol_lookback_sessions`) rather than repeating a literal.
A CLI default that silently disagreed with the model default is exactly how the
30-session level lookback went unnoticed. `--sigma-pts` follows the same rule.

R stays at 5 sessions: it is an ATR-like measure of *current* conditions, and a
20-session median is neither current nor cheap — the 20-session variant was
visibly slower and lagged regime changes by a month of sessions.

Warm-up remains the sum of the two (`20 + 5 = 25` sessions ahead of the requested
window) rather than the max. Both histories fill in the same forward pass so the
max would suffice to prime them, but the extra sessions give `levels_for_price`
depth to extend into when price leaves the clustered range, which is exactly the
breakout case the extension exists for. A 2025-05-01 → 2026-06-30 build therefore
reads 307 session files to label 282 sessions.

## Consequences

- **All first-passage labels are invalidated and rebuilt.** Entry prices, fill
  times, target/stop distances and the row population all changed. The τ frontier
  and book figures in ADR-123 and in the pre-2026-09-11 ROADMAP table describe a
  different label set.
- Bound geometry is now independent of R and of the price level, so level
  structure is comparable across eras and instruments. `*_r` candidate columns
  still move with R, because distance-to-target legitimately should.
- Zone descriptors become informative. The 435-pt zone that produced
  `strength = 9.96` over 45 members is gone; the widest bound on 2026-06-25 is
  22.5 pts (0.30 R).
- `sigma_pts` is a live modelling choice, not an implementation detail: it decides
  whether a bound is one level or a multi-session zone, and it is the first
  parameter in the sweep.
- Fill is no longer guaranteed, so trade count per session falls and is now a
  function of `max_fill_wait_seconds`. That parameter joins `sigma_pts`,
  `buffer_pts`, `entry_offset_pts`, `lookback_sessions` and `large_order_size` in
  the sweep that still owes a Deflated Sharpe / PBO accounting
  ([ADR-122](ADR-122-structural-context-and-closed-research-loop.md)).

## Alternatives rejected

- **Shrink `cluster_r` alone** — relocates chain breaks without bounding width;
  produces long chains and singletons side by side.
- **Complete-linkage merging** (join only if the whole cluster's diameter stays
  within `cluster_pts`) — bounds width correctly, but folds the width limit into
  the merge threshold so resolution and tradeability cannot be tuned separately,
  and leaves grouping tied to R.
- **A hard width cap on gap-merged chains** (`max_width_r`, recursive split at
  the largest internal gap) — implemented first and it did bound the width, but
  it treats the symptom: grouping stayed transitive and R-dependent, and cluster
  identity stayed fragile to a single intermediate level. Superseded by KDE
  grouping before any labels built with it were used.
- **Variant A: KDE peaks with FWHM edges** — the interval comes from the
  half-maximum of the smoothed density. Cleaner in principle, but the edges are
  bandwidth artefacts rather than prices the market turned at, which breaks the
  auditability the level set exists to provide.
- **Pullback limit entry** (rest *behind* price for a better fill) — improves
  reward-to-risk but is the opposite statement: it bets the level holds, while
  the model's thesis is that price is leaving it.
- **Anchoring the volatility ladder on the fill price** — would keep the ladder a
  clean R grid, but the two sides fill at different prices while sharing one
  candidate set, so no single anchor is correct. Restating distances after the
  fill and applying `min_dist_r` achieves the same end without per-side candidate
  generation.
- **Scoring unfilled orders as flat** — treats "no trade" as an outcome of the
  strategy rather than an absence of one, and biases the conditional expectation
  toward zero in exactly the regimes where price never left the level.
