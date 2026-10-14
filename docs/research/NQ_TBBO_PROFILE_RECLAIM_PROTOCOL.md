# NQ TBBO confluent-profile reclaim — protocol

Status: experimental. Recorded before running this policy's outcome tests on
2025–2026 tape data. The objective is a one-contract NQ policy with at most two
non-overlapping attempts per cash session, a **planned** maximum loss of 20
points per trade, and a $800 closed-loss re-entry boundary. A stop is a market
order, so a fast gap can exceed the planned risk; a realised loss worse than
$500, or observed daily liquidation drawdown worse than $1,000, rejects the
policy rather than being hidden by its stop setting.

## Hypothesis

A price excursion through a multi-session, confluent volume-profile zone can be
responsive rather than initiative activity. The specific event is an attempted
auction beyond the upper (lower) edge of a pre-open bound, strong same-direction
aggressive TBBO volume including a large print, and a completed one-minute
reclaim back inside the bound. If the opposite profile node provides at least a
2R runway, price should rotate to it often enough to compensate for executable
bid/ask costs. This is a falsifiable profile-location plus flow-failure claim,
not a generic VWAP, CVD, or opening-range signal.

The levels are exact tick-volume profiles of prior *closed* cash sessions
(08:30–15:00 Chicago), clustered by the existing KDE bound mechanism. Current
session state uses only prints and BBO snapshots at or before each completed
one-minute bar's right-labelled timestamp. A 20+ contract print is a feature,
not proof of a fillable order-book queue. Crossed/invalid BBO snapshots are
never used for a fill or depth feature.

## Policy family — discovery only

For each eligible session, calculate bounds from up to the preceding 20 eligible
cash sessions; the quarterly expiry week is excluded beforehand because tape
contract-roll parity has not been established. A candidate upper failure must:

1. approach from below, trade at least 1 or 2 points above a strong bound's
   upper edge in the completed three-minute window, and close back below that
   edge;
2. show outward signed volume and an outward 20- or 50-lot print in that window;
3. reclaim at least two points from the window high and close on the reverting
   side of the current causal session VWAP; and
4. have a distinct lower strong bound at least two planned risks away.

Lower failures are the exact mirror. The signal becomes known only at the
one-minute close. The market entry is the first valid BBO at least 250 ms later
and no more than five seconds later. The stop is 20 points from that executable
entry; the target is the pre-open opposing bound. Target/stop triggers queue a
market exit for 250 ms and fill at the next valid executable BBO. At 15:00 CT an
open trade is liquidated by the same rule. There is no same-bar fill, level
constructed from today, profit-target fitting, overlap, position scaling or
unbounded martingale. A session has at most two attempts and no new attempt once
closed daily P&L is at or below -$800.

The fixed candidate grid has eight rows: bound membership at least 2 or 3; sweep
buffer 1 or 2 points; minimum large-print size 20 or 50. All candidates use exact levels,
the 20-point planned stop, two-risk target runway, 250ms latency, and the same
daily guard. All eight rows are retained whether profitable or not. This grid is a
research choice, not a hidden search for the result.

## Chronology and gates

Only NQ print tapes from 2025-05-01 onward are eligible. Because they are
unadjusted lead out-rights, a bound uses only prior sessions carrying the same
algorithmic CME lead contract; it is not carried across a quarterly roll. This
creates a deliberate 20-session post-roll warmup exclusion. May 2025–March 2026
is discovery. Candidate selection uses its chronological monthly folds only;
April–June 2026 is frozen historical validation. These dates were visible to
unrelated experiments, so they are chronological OOF for this policy rather than
a universally virgin sample. July–September 2026 remains a one-time reserve and
is not read unless the prior gates pass. Existing exposure from unrelated
experiments cannot be undone and is disclosed in reports.

Acceptance requires every one of: the base 0.25-point-per-side adverse-fill
assumption and 1-point stress both positive in validation; positive validation mean daily net
after costs with a positive 95% five-session block-bootstrap lower bound; mean
net at least $1,000 per eligible validation day; maximum realised trade loss no
worse than $500; observed quote-mark daily drawdown no worse than $1,000; and
no unresolved required executions. The same mechanical risk and return checks
are reported for compatibility. A failure rejects the target claim and locks the
reserve; it does not authorize threshold changes based on validation.

## Evidence and limits

The run persists raw-file hashes, profile/bound provenance, all candidate
ledgers, frozen configuration, daily marks, block-bootstrap interval, exact
quote fill audit and a no-future prefix test. TBBO stores a print-time BBO
snapshot, not queue position, market depth beyond the top level, hidden
liquidity, or broker fills. A profitable backtest therefore cannot establish a
live edge or guarantee a stop's dollar loss. The primary mechanism motivation is
[Cont, Kukanov & Stoikov (2014)](https://arxiv.org/abs/1011.6402) on
depth-normalized order-flow impact and the CME-futures evidence in
[Ma et al. (2025)](https://doi.org/10.1111/ajae.12479); neither is treated as
evidence that this exact NQ rule works.
