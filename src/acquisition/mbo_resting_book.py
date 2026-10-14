"""Pure logic for maintaining "what's currently resting" order-book state from a batch of new
MBO events. DB-free core (mirrors src/webui/mbo_orders.py's split of pure logic from SQL), used
by scripts/ops/mbo_incremental_pull.py to keep databento.mbo_resting_orders_v2 current.
See ADR-147 and ADR-149.

Event semantics:

``A``/``M``/``F``/``C``
    Incremental per-order changes, interpreted exactly as src/webui/mbo_orders.py::build_segments
    does (same "a price-changing Modify starts a fresh segment" rule for ``t_start``).

``R`` -- Clear book. Two distinct flavours, told apart by the ``F_SNAPSHOT`` flag:
    * **Real exchange clear** (no ``F_SNAPSHOT``, real sequence number). CME wipes the book.
      Observed once a week at Sunday pre-open (e.g. 2026-09-27 09:32:28 CT), ~11 minutes ahead of
      the full re-listing that repopulates it.
    * **Databento synthetic marker** (``F_SNAPSHOT`` set, ``sequence`` 0). Sits at the head of a
      UTC-midnight snapshot group at the same ``ts_recv``. The group itself carries all the
      information, so this marker needs no separate handling.

**Snapshot group** -- every row sharing one ``ts_recv`` with ``F_SNAPSHOT`` set. This is
    authoritative: the exchange's own statement of the *complete* resting book at that instant.
    Verified against NQZ6 (2026-09-20..09-28): a snapshot ``ts_recv`` carries no other rows, so a
    group is always contiguous in ``(ts_recv, sequence)`` order and never straddles a 30-min tick.

Why the snapshot group matters: **CME purges Day orders at the weekend close and sends no Cancel
message.** The only signal is the order's *absence* from the next snapshot group. ADR-147's
original assumption -- that A/M/F/C were the whole story and Clear/None/Trade "don't change book
state" -- therefore leaked every such order forever: 126 silently purged across the 2026-09-24 ->
09-27 weekend alone, accumulating to 808 phantom orders (27% of the displayed book) by 09-28,
measured against the exchange's own 09-28 19:00 CT snapshot. Reconciling against each snapshot
group is the only mechanism that can remove them, and it heals pull-window gaps for free.

Ordering contract: ``events`` must be in **global chronological** order (``ts_recv``, then
``sequence``) -- not grouped per order_id. ``R`` and snapshot groups are whole-book operations, so
their position relative to every other order's events is what makes them correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

F_SNAPSHOT = 32          # Databento flag bit: row belongs to a book snapshot re-listing


@dataclass(slots=True)
class RestingOrder:
    symbol: str
    side: str
    price: float
    size: int
    t_start: pd.Timestamp
    updated_at: pd.Timestamp
    # False when t_start is only a *lower bound*: the order was first seen already resting in a
    # snapshot group (so it was really placed at some unknown earlier time), or it turned up in a
    # snapshot at a price we had not recorded, meaning we missed the Modify that re-queued it.
    # Lets the chart say "resting >= Xh" instead of asserting an origin it cannot know.
    t_start_exact: bool = True


def apply_events(
    current: dict[int, RestingOrder], events: pd.DataFrame
) -> tuple[dict[int, RestingOrder], set[int]]:
    """Apply a batch of new MBO events to `current` resting state. Returns
    ``(updated_or_new, closed_order_ids)`` -- `current` itself is not mutated, so a caller can
    diff before writing.

    `events` columns: order_id, ts_recv, action, side, price, size, symbol, and (optionally)
    flags. Without a `flags` column every row is treated as an ordinary incremental event, which
    keeps older callers and unit tests working; snapshot reconciliation simply cannot engage.

    `closed` is guaranteed to cover every order_id that was in `current` and is no longer
    resting, however it went away -- Cancel, full Fill, exchange Clear, or omission from a
    snapshot group -- so the caller's "carry forward what is neither updated nor closed" set
    arithmetic yields exactly the new book.
    """
    book: dict[int, RestingOrder] = dict(current)
    closed: set[int] = set()
    # What was resting immediately before a real exchange Clear. Consulted by a snapshot group
    # later in this same batch so survivors of the weekly wipe keep their true t_start instead of
    # being re-dated to the re-listing. Empty unless a Clear has just been seen.
    pre_clear: dict[int, RestingOrder] = {}

    for is_snapshot, chunk in _chunks(events):
        if is_snapshot:
            _reconcile_snapshot(book, chunk, pre_clear, closed)
            pre_clear = {}
            continue
        for r in chunk.itertuples(index=False):
            if r.action == "R":
                pre_clear = dict(book)
                closed.update(book)
                book.clear()
            else:
                _apply_one(book, r, closed)

    closed.update(current.keys() - book.keys())
    # Anything still resting is by definition not closed. This matters for the weekly sequence:
    # the exchange Clear marks every open order closed, and the re-listing 11 minutes later then
    # restores the survivors -- unchanged ones as the *same* instance, so they never show up in
    # `updated`. Without this line the caller's "carry forward what is neither updated nor closed"
    # arithmetic drops them, and the whole book collapses to the all-closed sentinel. Caught by
    # replaying real 2026-09-27 data: the 15:00 UTC checkpoint came out empty instead of holding
    # the re-listing's 1,466 orders.
    closed -= book.keys()
    updated = {oid: o for oid, o in book.items() if current.get(oid) != o}
    return updated, closed


def _chunks(events: pd.DataFrame):
    """Split chronological `events` into runs of ordinary rows and whole snapshot groups,
    yielding ``(is_snapshot, sub_frame)``. A snapshot group is every row sharing one ``ts_recv``
    with ``F_SNAPSHOT`` set; ordinary rows run together regardless of timestamp."""
    if events.empty:
        return
    if "flags" not in events.columns:
        yield False, events
        return
    snap = (events["flags"].fillna(0).astype("int64").to_numpy() & F_SNAPSHOT) != 0
    if not snap.any():
        yield False, events
        return
    ts = events["ts_recv"].to_numpy()
    starts = np.zeros(len(events), dtype=bool)
    starts[0] = True
    # A new chunk begins wherever the snapshot mask flips, and additionally between two adjacent
    # snapshot groups that happen to abut (different ts_recv, both flagged).
    starts[1:] = (snap[1:] != snap[:-1]) | (snap[1:] & (ts[1:] != ts[:-1]))
    bounds = np.flatnonzero(starts)
    for i, lo in enumerate(bounds):
        hi = bounds[i + 1] if i + 1 < len(bounds) else len(events)
        yield bool(snap[lo]), events.iloc[lo:hi]


def _reconcile_snapshot(book: dict[int, RestingOrder], group: pd.DataFrame,
                        pre_clear: dict[int, RestingOrder], closed: set[int]) -> None:
    """Replace `book` with the snapshot group's contents -- the exchange's complete statement of
    what is resting at that instant.

    Orders in `book` that the group does not list were purged by the exchange with no Cancel ever
    sent (the weekend-close case); dropping them here is the only way they can ever leave the
    book. Survivors keep their existing `t_start`, so a nightly re-listing no longer truncates a
    long-resting order's age -- the bug that had re-dated 1,051 of 5,043 orders to exactly
    19:00:00 CT in the 2026-09-28 12:00 checkpoint."""
    fresh: dict[int, RestingOrder] = {}
    for r in group.itertuples(index=False):
        if r.action != "A":
            continue                      # the group's own leading synthetic Clear marker
        oid = int(r.order_id)
        price, size = float(r.price), int(r.size)
        prior = book.get(oid) or pre_clear.get(oid)
        if prior is None:
            # First time we have ever seen it: it was already resting, so its true origin is
            # some unknown earlier moment and this timestamp is only an upper bound on age.
            fresh[oid] = RestingOrder(symbol=r.symbol, side=r.side, price=price, size=size,
                                      t_start=r.ts_recv, updated_at=r.ts_recv,
                                      t_start_exact=False)
        elif prior.price == price and prior.size == size:
            fresh[oid] = prior            # unchanged -- keep the instance, no row churn
        else:
            # A size-only difference is an ordinary fill/resize and leaves the queue origin
            # alone. A price difference means we missed the Modify that re-queued it, so we keep
            # the best t_start we have but stop claiming it is exact.
            fresh[oid] = RestingOrder(
                symbol=prior.symbol, side=prior.side, price=price, size=size,
                t_start=prior.t_start, updated_at=r.ts_recv,
                t_start_exact=prior.t_start_exact and prior.price == price)
    closed.update(book.keys() - fresh.keys())
    book.clear()
    book.update(fresh)


def _apply_one(book: dict[int, RestingOrder], r, closed: set[int]) -> None:
    """One ordinary A/M/F/C event, applied in place."""
    oid = int(r.order_id)
    existing = book.get(oid)

    if r.action == "A":
        book[oid] = RestingOrder(symbol=r.symbol, side=r.side, price=float(r.price),
                                 size=int(r.size), t_start=r.ts_recv, updated_at=r.ts_recv)
        closed.discard(oid)
    elif existing is None:
        return  # M/F/C for an order we never saw an Add for -- placed before our own bootstrap
                # or lost to a data gap. Nothing to update; the next snapshot group settles it.
    elif r.action == "M":
        price_changed = float(r.price) != existing.price
        book[oid] = RestingOrder(
            symbol=existing.symbol, side=existing.side, price=float(r.price), size=int(r.size),
            t_start=r.ts_recv if price_changed else existing.t_start, updated_at=r.ts_recv,
            t_start_exact=True if price_changed else existing.t_start_exact)
    elif r.action == "F":
        # A fill never closes an order -- only `C` does. For a native iceberg the fill consumes the
        # *displayed* quantity and the venue immediately replenishes it from hidden reserve with an
        # `M`; closing here dropped the order, so that `M` then landed in the unknown-order no-op
        # above and the order vanished from the book until the next snapshot re-added it.
        # Measured on NQZ6, 2026-09-28: all 592,977 fully-filled orders receive an explicit `C`, so
        # closing on a fill was pure redundancy -- while 872 orders took a single fill LARGER than
        # their entire displayed size, which is only possible with hidden reserve. This also makes
        # the module match build_segments, which already ends a segment only on `C`. Safety net if a
        # `C` is ever lost to a gap: the next snapshot group drops the order (ADR-149).
        #
        # Only a fill at the order's OWN resting price shrinks it. One order can be aggressor and
        # resting order in the same life: a marketable limit sweeps the book (taking liquidity at
        # other prices, which also produces `F` rows carrying those prices) and rests the
        # remainder. Decrementing on those would shrink a resting order for quantity it bought
        # somewhere else entirely -- see 6882409490573, which swept 17 lots up to 30561.50 before
        # resting there.
        if float(r.price) != existing.price:
            return
        book[oid] = RestingOrder(symbol=existing.symbol, side=existing.side,
                                 price=existing.price,
                                 size=max(existing.size - int(r.size), 0),
                                 t_start=existing.t_start, updated_at=r.ts_recv,
                                 t_start_exact=existing.t_start_exact)
    elif r.action == "C":
        book.pop(oid, None)
        closed.add(oid)
