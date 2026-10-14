# ADR-149 — Reconciling the MBO resting book against exchange snapshot groups

**Status:** Accepted — implemented 2026-09-30
**Supersedes the core state-transition assumption of:** [ADR-147](ADR-147-mbo-incremental-resting-book.md)

## Context

ADR-147 built `databento.mbo_resting_orders` by replaying `A`/`M`/`F`/`C` events and asserted that
the remaining MBO actions — Trade prints, Clear, None — "don't change book state". That assumption
is false for `R` (Clear), and the consequence was visible in the Live Chart: an order-map overlay
spanning 2026-09-25 → 09-29 drew a 99-lot offer placed Friday 09-25 14:40 CT as still resting on
Monday 09-28.

### What the data actually says

Investigated against NQZ6, 2026-09-20 → 09-29.

**Orders do legitimately persist across both daily and weekend closes.** GTC/GTD orders keep the
*same* `order_id` and are re-listed at every boundary — 1,030 of the 2,993 orders resting at
09-24 19:00 CT were still there on Sunday 09-27. Persistence alone is not an error.

**But CME purges Day orders with no Cancel message.** Measured at two boundaries:

| Boundary | Orders that left | Had an explicit `C`/`F` | Silently purged |
|---|---|---|---|
| Daily, 09-23 19:00 → 09-24 19:00 CT | 2,142 | 2,142 | **0** |
| Weekend, 09-24 19:00 → 09-27 19:00 CT | 1,963 | 1,837 | **126** |

Daily closes are clean. The weekend is where orders die silently, and the *only* signal is the
order's absence from the next snapshot group.

**`R` exists in the feed in two distinct flavours:**

| Flavour | Flags | When | Meaning |
|---|---|---|---|
| Real exchange clear | 128 (`F_LAST`), real sequence | Sundays only — 09-20 07:54, 09-27 09:32 CT | CME wipes the book; a full re-listing follows ~11 min later |
| Databento synthetic | 40 (`F_SNAPSHOT`+`F_BAD_TS_RECV`), sequence 0 | 19:00:00 CT daily (= 00:00 UTC) | Head marker of a UTC-midnight snapshot group at the same `ts_recv` |

**Snapshot groups are authoritative and safely delimited.** A group is every row sharing one
`ts_recv` with `F_SNAPSHOT` set. Verified: a snapshot `ts_recv` carries *no other rows* (0
non-snapshot rows at each of four boundaries checked), so a group is contiguous in
`(ts_recv, sequence)` order, and because 00:00 UTC is itself a 30-min grid boundary a group never
straddles two checkpoint ticks.

### Measured damage to v1

Diffed v1's 09-28 19:00 CT checkpoint against the exchange's own snapshot for that instant:

| Exchange truth | v1 book | Phantoms v1 showed | Real orders v1 missed |
|---|---|---|---|
| 2,951 | 3,758 | **808 (27% of the displayed book)** | **1** |

v1 added correctly and never subtracted. Sweeping every boundary with
`scripts/ops/verify_resting_book.py` localises it exactly:

| Boundary (CT) | Exchange truth | v1 book | Phantoms |
|---|---|---|---|
| 09-20 19:00 → 09-24 19:00 (5 weekday boundaries) | 3,153 … 2,993 | matches | **0** |
| 09-27 19:00 (first after the weekend) | 3,132 | 3,940 | **808** |
| 09-28 19:00 | 2,951 | 3,758 | **808** |

**v1 was clean on weekdays and broke entirely at the weekend** — all 808 phantoms appear in one
step. Note 808 ≫ the 126 from the table above, and the two are consistent: the 126 counts only
orders that had survived long enough to appear in the 09-24 19:00 snapshot, whereas the 808 also
includes every Day order *created during Friday's session* and killed at Friday's close. The
99-lot in the original report — placed Friday 09-25 14:40 CT — is in the second group, which is
why it never appeared in any snapshot at all: its entire life was one `Add`.

Separately, 1,051 of 5,043 orders in v1's 09-28 12:00 checkpoint carried a `t_start` of exactly
19:00:00 CT, because snapshot re-listings were applied as fresh Adds and reset the origin — so
genuinely long-resting orders displayed a truncated age while phantoms displayed an unbounded one.

## Decision

1. **A snapshot group replaces the book.** On encountering one, the book becomes exactly the
   group's contents. Orders absent from it were purged by the exchange and are dropped — the only
   mechanism that can remove them. This also heals pull-window gaps for free, since every boundary
   re-establishes ground truth.
2. **Survivors keep their `t_start`.** A re-listing takes the snapshot's price/size but never
   re-dates an order we were already tracking.
3. **`t_start_exact`** marks an origin that is only a lower bound — an order first seen already
   resting, or one that turned up in a snapshot at a price we had not recorded (a missed Modify).
4. **Real `R` clears the book**, stashing pre-clear state so a re-listing later in the same batch
   restores true ages for survivors. The synthetic `R` needs no separate handling; its group does
   the work.
5. **Events must be fed in strict chronological order** (`ts_recv`, `sequence`). The previous
   per-`order_id` ordering was harmless for independent per-order arithmetic but is actively wrong
   for whole-book operations.
6. **A new table, `mbo_resting_orders_v2`.** A corrected checkpoint has different rows for the same
   `(snapshot_ts, order_id)` key, so v1 could not be fixed by appending, and rewriting it in place
   would break this pipeline's append-only rule. v1 is frozen intact as a rollback path.

### A bug the sample replay caught, worth recording

The first implementation collapsed the book to the all-closed sentinel at every weekly clear. The
exchange `Clear` marks every open order closed; the re-listing 11 minutes later restores the
survivors, and unchanged ones are restored as the *same* object instance, so they never appear in
`apply_events`' `updated` set. The caller's "carry forward what is neither updated nor closed"
arithmetic then dropped them all. Replaying real 09-27 data exposed it immediately: the 15:00 UTC
checkpoint held one `order_id = -1` sentinel row instead of the re-listing's 1,466 orders.

Fix: `closed -= book.keys()` before returning — anything still resting is by definition not closed.
Regression test:
`test_clear_then_relisting_carries_unchanged_survivors_not_just_changed_ones`.

This is the reason a scoped sample run against a scratch table exists as a first-class option
(`--symbol`/`--start`/`--end`/`--table`): the unit tests all passed while this bug was live, because
it lived in the *interaction* between `apply_events`' return contract and the caller's set
arithmetic, not in either half alone.

## Is there a unique key for an MBO record? (investigated 2026-09-30)

Asked because if some field combination uniquely identified a record, the append path could
deduplicate instead of relying on window arithmetic. Databento's own shipped type stubs
(`databento_dbn/_lib.pyi`) define `order_id` as "The order ID assigned at the venue" and `sequence`
as "The message sequence number assigned at the venue" — neither is documented as unique.

Measured on NQZ6, one RTH hour (4,198,004 rows):

| Candidate key | Distinct | Verdict |
|---|---|---|
| `order_id` | 1,695,957 | not unique — one order has many events |
| `(ts_recv, sequence)` | 3,790,108 | not unique; 15.5% of rows share one, biggest group **69 rows** |
| `(ts_event, sequence)` | 4,014,420 | not unique |
| `(ts_recv, sequence, order_id)` | 4,169,820 | not unique |
| `(ts_event, instrument_id, sequence, order_id, action)` | 4,192,401 | not unique |
| **every field in the record** | 4,197,904 | **not unique — 100 collisions** |

**No unique key exists, not even the complete record.** Every full-row duplicate inspected was an
`F` (Fill) on the same order, same price, same 1-lot size, same packet — i.e. one resting order
taking several separate 1-lot fills inside a single matching-engine packet. Those are *genuinely
distinct events*: deduplicating them would silently destroy fill quantity and corrupt the book's
size arithmetic. So content-based dedup is not merely impossible here, it would be wrong.

**The append path is nonetheless sound**, for a reason worth writing down: `sequence` is strictly
packet-scoped — 0 of 1,966,978 sequences spanned more than one `ts_recv`. Since every row sharing a
`(ts_recv, sequence)` therefore shares an identical `ts_recv`, and the pull window is a half-open
interval on `ts_recv`, a tie group can never be split across two pulls. The
`(ts_recv, sequence) > (checkpoint)` filter excludes all rows of the checkpoint's own tie group, and
all of them were necessarily ingested by the previous window. No row is lost or doubled.

**`order_id` is effectively unique per order**, which is what matters for book state: of 10,996,693
order_ids with a genuine Add on 2026-09-28, exactly **one** was reused (`6882662848119` — cancelled
04:47:52 CT, reissued 04:59:28 CT; 2 Adds and 2 Closes, strictly sequential, never concurrent), and
NQU6/NQZ6 share **zero** order_ids. So `mbo_resting_orders_v2`'s `(snapshot_ts, order_id)` primary
key is valid. One consequence to know: `build_segments` stops at an order's first `C`, so the second
lifecycle of a reused id is invisible in the overlay. At 1-in-11-million this is documented, not
fixed — the `break` is what keeps the first lifecycle correct.

### The remaining ~1-order residual: iceberg replenishment

After the rebuild, v2 matches the exchange exactly at most boundaries but is short by exactly 1
order at a few (3 of 15 for NQU6). Traced to native iceberg/reserve orders. Example
`6881584016268` (NQU6, ask 29300, 2026-08-30):

```
18:58:08  A  size 1      <- displays 1 lot
18:58:15  F  size 1      <- fully fills the displayed quantity -> we close it
18:58:15  M  size 1      <- venue replenishes the display from hidden reserve
18:58:22  F  size 9      <- a fill NINE times the displayed size: hidden reserve
18:58:22  M  size 1      <- replenished again
```

The original `F` handling closed the order when remaining hit 0, so the replenishing `M` landed in
the "Modify for an order we never saw an Add for" no-op branch and the order stayed out of the book
until the next snapshot re-added it (≤1–3 orders of ~3,000, ~0.03%).

**Fixed 2026-09-30: a fill never closes an order — only `C` does.** Four independent reasons this is
the right fix rather than the "treat a Modify for an unknown order as a re-add" repair first
considered:

1. It matches CME semantics: a fill of the *displayed* quantity does not end an iceberg, the reserve
   is still resting.
2. Closing on a fill was **redundant** — all **592,977** fully-filled orders on 2026-09-28 (NQZ6)
   received an explicit `C`, zero exceptions.
3. It removes a real inconsistency: `build_segments` already ends a segment only on `C`, and this
   module's docstring already promises identical semantics.
4. Snapshot reconciliation is the safety net — a `C` lost to a gap now costs at most one checkpoint,
   which is precisely why this is safe *now* and would not have been before this ADR.

It prevents the state loss rather than repairing it, so the true `t_start` survives instead of being
resurrected with an unknown origin. Size clamps at 0 rather than going negative. Verified on the case
above: rebuilt into a scratch table, order `6881584016268` is present at the 08-30 19:00 CT boundary
with its replenished 1-lot display and `t_start = 08-30 18:58:08.06071` still marked exact, and the
boundary matches the exchange exactly (3,120 = 3,120) where v2 had been short by one.

### Reload chains are not icebergs — and icebergs are currently invisible

Worth recording because the two are easy to conflate. A **reload chain** (`find_chains`) is N
*distinct* order_ids successively replacing each other at one price; it requires
`next_id != order_id`, so a native iceberg — which is **one** order_id whose display the venue
replenishes — can never match one. Same economic intent (work size while showing little), opposite
mechanics: cross-order versus intra-order.

The consequence is a blind spot. Measured on NQZ6, 2026-09-28, using the unambiguous test (a single
fill larger than the whole displayed size, impossible without hidden reserve):

| Displayed size | Icebergs | Contracts absorbed | Biggest single fill |
|---|---|---|---|
| < 20 — **invisible to the order map** | **872** | **7,311** | 79 lots |
| ≥ 20 — visible | **0** | 0 | — |

Every iceberg that day was invisible, and structurally so: the overlay's `big` CTE filters on
*displayed* Add size, and displaying small is the entire point of an iceberg. On the looser test
(cumulative fills > display) it is 1,461 orders absorbing 12,285 contracts against 435 for the
visible ones — a ~28x blind spot.

Implemented: icebergs are detected as a sibling of reload chains — similar presentation, entirely
separate detection — and **selected by absorbed size, not displayed size**, so `min_size` stops
hiding them. The signal is higher-confidence than a chain: "filled more than was showing" is proof,
whereas a chain needs `dt_s`, size-ratio and `n_links` thresholds tuned.

#### Attribution must be per *visit to a price level*, not per order

The first two attempts were wrong, both caught by a user spotting a self-contradicting tooltip —
"filled 0 of 1" beside "absorbed 10" — on an order at a price the market never reached.

1. **Per order.** `absorbed` was summed per `order_id` and stamped on every segment. An order that
   re-prices produces one segment per level (one walked **94** levels), so the whole absorbed total
   was claimed at each of them. 1,326 of 3,550 marks asserted reserve where nothing had filled.
2. **Per (order, price).** Better, but an order that leaves a level and returns has several
   segments at that same price, and each inherited the level's combined total. 4 marks survived.
3. **Per segment (correct).** `build_segments` now tracks `filled` and `disp` (high-water displayed
   size) *per segment*, resetting both on a price-changing Modify. The badge is simply
   `filled > disp`, so `absorbed` **is** that segment's fill count and cannot contradict it.
   `ICEBERG_SQL` is demoted to selecting which order_ids to fetch — its totals no longer reach the
   display.

| | marks drawn | absorbed | marks with 0 fills there | absorbed ≠ filled |
|---|---|---|---|---|
| per order | 3,550 | 31,293 | 1,326 | 1,326 |
| per (order, price) | 360 | 15,955 | 4 | 38 |
| **per segment** | **334** | **14,851** | **0** | **0** |

The same per-segment accounting fixes a pre-existing bug of the same shape: `end="filled"` used the
order's cumulative fill count, so a moved order was drawn as filled at whatever level it happened to
die at. Filled marks in one 2026-09-21..23 window fell from **2,978 to 1,063** — 1,915 of them had
been painted at prices that never traded.

The lesson worth keeping: an order-level fact must never be rendered on a per-level mark. Every
attribute the chart draws at a price has to be computed at that price.

#### Two independent proofs of reserve, neither subsuming the other

The first implementation used only a *consequence* test — "took more than it ever showed here". A
*mechanism* test was then proposed: the fill-then-replenish cycle. Both are valid, and measuring
them against each other (NQZ6, 2026-09-22) showed they disagree far more than expected:

| | orders |
|---|---|
| both signals | 1,536 |
| **refresh only** (consequence test misses) | **1,072** |
| overfill only (mechanism test misses) | 575 |

A 50-lot display filled 10 and restored to 50 must have held at least 60 — reserve proven — but
`absorbed > shown` never fires because 10 < 50. Conversely a single fill larger than the display
proves reserve with no refresh cycle at all. The badge therefore fires on **either**.

Two traps in the mechanism test worth recording, because the naive form is badly wrong:

* **`F` followed by `M` is also an ordinary partial fill** — the `M` reports the reduced remainder.
  Matching on adjacency alone flagged 8,105 orders in one day, most of them perfectly normal. The
  discriminator is *direction*: an iceberg's Modify puts the size back **up**, a partial fill's takes
  it **down** (`int(r.size) > cur`).
* **Requiring repetition costs coverage for no gain.** One refresh is already proof; demanding two
  would have cut 2,608 orders to 702.

Supporting evidence that these really are venue replenishments rather than an algo re-sending:
**99.4%** of refills (8,054 of 8,105) arrive within 1 ms of their fill, i.e. engine-atomic.

Honest limit: MBO cannot distinguish a *native* CME iceberg from a trader's algo topping up at the
same level within a millisecond. Both are replenished hidden liquidity at a price — which is what
matters for reading the tape — so "iceberg" here names the behaviour, not a certified order type.

Final measured effect on one 2026-09-21..23 window: 706 marks over 675 orders, 27,690 contracts
absorbed, **0** self-contradicting marks, against 3,550 marks and 1,326 contradictions in the first
implementation.

### What should change as a result

Nothing in the raw table: no unique constraint, no dedup step. But the *checkpoint* table's
inability to hold a corrected version of a checkpoint — the thing that forced a new table here — is
fixable and worth doing before the next correction: add a revision discriminator to the key
(`PRIMARY KEY (snapshot_ts, order_id, book_rev)`), with readers taking the highest `book_rev` for a
`snapshot_ts`. Corrected checkpoints could then be *appended* alongside the originals, keeping the
append-only rule intact and making a future fix a reprocess rather than a table migration.

## Consequences

- Phantom orders stop accumulating and existing ones are gone from the rebuilt history.
- Resting durations are correct for GTC orders and honestly bounded for snapshot-discovered ones.
- The chart's order map no longer draws dead orders across weekends.
- Iceberg orders stay in the order-map payload and can be hidden on the chart without another query.
  The Order map parameter card has an **Icebergs** checkbox (default on, `lc_mbo_icebergs`). Off
  drops every reserve segment from the drawing, hover targets, and Fit all orders; the stats line
  then reads “N icebergs hidden”. On restores the same payload.
- One cheap, exact invariant now exists for monitoring: at any snapshot boundary the book must
  equal the snapshot. Phantoms > 0 is a hard failure signal.
- v1 remains on disk until the v2 history has been in use long enough to retire it; that deletion
  needs explicit approval.

## The chart's "R" marker

Implemented alongside the book fix. `CLEARS_SQL` and `SNAP_BOUNDARIES_SQL` feed `build_segments`,
which closes a still-open segment with `end="cleared"` / `outcome="CLEARED"` at whichever comes
first: a real exchange `Clear`, or a snapshot boundary the order was not re-listed at. Rendered as a
cyan **R** glyph at the segment end — deliberately not the cancel cross, since the trader did not do
it — with a legend entry, a status-line count, and a tooltip reading "removed by the exchange at a
session close — not cancelled by the trader".

Verified live against `/api/chart/mbo-orders` for 2026-09-25 19:00Z → 09-27 16:00Z: 30 orders now
marked `CLEARED` at the 09-27 09:32:28 CT exchange clear (the far-from-market GTC book: 250 lots @
27200, 150 @ 27500, 100 @ 32000, …), each of which previously drew as resting to the edge of the
window.

## Not done here

- **A cleared order is not re-drawn after its post-clear re-listing.** `build_segments` terminates
  the segment at the clear and ignores the snapshot `A` that re-adds the order ~11 minutes later, so
  in a window spanning the weekly clear those orders simply stop at the R rather than resuming. A
  window starting after the re-listing is unaffected (it seeds from the checkpoint, which is
  correct). Fixing it means detecting the clear during iteration rather than after it, so a fresh
  segment can start — deferred as a display refinement, not a data problem.
- `t_start_exact` is carried through the read path but the chart does not yet render "≥".
- The 2026-09-29 02:19 CT v1 book discontinuity (4,179 → 654 orders across a 79-minute checkpoint
  gap) was never root-caused — journal retention no longer covers it. v2's snapshot reconciliation
  bounds the blast radius of any such gap to the next boundary, but the handoff path still wants a
  guard that refuses to start a fresh book when the prior checkpoint predates the window.
