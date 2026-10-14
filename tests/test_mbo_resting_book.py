"""Tests for src/acquisition/mbo_resting_book.py -- synthetic events only, no database."""

from __future__ import annotations

import pandas as pd
import pytest

from src.acquisition.mbo_resting_book import F_SNAPSHOT, RestingOrder, apply_events

pytestmark = pytest.mark.acquisition

T0 = pd.Timestamp("2026-09-24 13:30:00", tz="UTC")


def _ts(sec: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(seconds=sec)


def _ev(rows: list[tuple]) -> pd.DataFrame:
    """rows: (sec, order_id, action, side, size, price, symbol[, flags])"""
    return pd.DataFrame(
        [(r[1], _ts(r[0]), r[2], r[3], r[5], r[4], r[6], r[7] if len(r) > 7 else 0)
         for r in rows],
        columns=["order_id", "ts_recv", "action", "side", "price", "size", "symbol", "flags"])


def _snap(sec: float, rows: list[tuple]) -> list[tuple]:
    """A snapshot group: every row shares one ts_recv and carries F_SNAPSHOT.
    rows: (order_id, side, size, price)"""
    return [(sec, r[0], "A", r[1], r[2], r[3], "NQZ6", F_SNAPSHOT) for r in rows]


def test_add_creates_a_new_resting_order():
    upd, closed = apply_events({}, _ev([(0, 1, "A", "B", 30, 100.0, "NQZ6")]))
    assert not closed
    assert upd[1] == RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))


def test_cancel_closes_an_existing_order():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(5, 1, "C", "B", 30, 100.0, "NQZ6")]))
    assert closed == {1} and 1 not in upd


def test_partial_fill_reduces_size_without_closing():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(5, 1, "F", "B", 10, 100.0, "NQZ6")]))
    assert not closed
    assert upd[1].size == 20 and upd[1].t_start == _ts(0)   # origin unchanged by a fill


def test_a_fill_that_empties_the_display_does_not_close_the_order():
    """Only `C` closes. A fill to zero is how a native iceberg's displayed quantity is consumed
    just before the venue replenishes it -- and empirically every genuinely-finished order does get
    an explicit `C`, so closing here was redundant as well as wrong."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(5, 1, "F", "B", 30, 100.0, "NQZ6")]))
    assert not closed
    assert upd[1].size == 0 and upd[1].t_start == _ts(0)


def test_fill_then_cancel_closes_the_order():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(5, 1, "F", "B", 30, 100.0, "NQZ6"),
                                         (5, 1, "C", "B", 30, 100.0, "NQZ6")]))
    assert closed == {1} and 1 not in upd


def test_iceberg_replenishment_keeps_the_order_alive_with_its_true_origin():
    """The real pattern from NQU6 order 6881584016268: display 1 lot, filled, replenished by `M`,
    then hit for 9 lots -- a fill nine times the displayed size, i.e. hidden reserve. The order must
    survive all of it and keep its original t_start."""
    cur = {1: RestingOrder("NQZ6", "A", 29300.0, 1, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([
        (7, 1, "F", "A", 1, 29300.0, "NQZ6"),      # displayed quantity consumed
        (7, 1, "M", "A", 1, 29300.0, "NQZ6"),      # venue replenishes the display
        (14, 1, "F", "A", 9, 29300.0, "NQZ6"),     # filled 9x the display: reserve
        (14, 1, "M", "A", 1, 29300.0, "NQZ6"),
    ]))
    assert not closed, "an iceberg must not be dropped by its own fills"
    assert upd[1].size == 1 and upd[1].price == 29300.0
    assert upd[1].t_start == _ts(0) and upd[1].t_start_exact is True


def test_oversized_fill_clamps_size_at_zero_rather_than_going_negative():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 5, _ts(0), _ts(0))}
    upd, _ = apply_events(cur, _ev([(5, 1, "F", "B", 40, 100.0, "NQZ6")]))
    assert upd[1].size == 0


def test_size_only_modify_keeps_original_t_start():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(5, 1, "M", "B", 20, 100.0, "NQZ6")]))
    assert not closed
    assert upd[1].size == 20 and upd[1].price == 100.0 and upd[1].t_start == _ts(0)


def test_price_changing_modify_resets_t_start():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(5, 1, "M", "B", 30, 101.0, "NQZ6")]))
    assert not closed
    assert upd[1].price == 101.0 and upd[1].t_start == _ts(5)   # new price = new segment origin


def test_mfc_for_an_unknown_order_is_a_harmless_no_op():
    """Placed before our bootstrap / a gap -- must not crash or fabricate a phantom order."""
    upd, closed = apply_events({}, _ev([(0, 9, "M", "B", 10, 100.0, "NQZ6"),
                                        (5, 9, "C", "B", 10, 100.0, "NQZ6")]))
    assert not upd and not closed


def test_add_then_cancel_within_one_batch_nets_to_closed_not_pending():
    """An order that both opens and fully closes inside the same 30-min tick must not appear in
    the update set at all (it was never truly resting as of the tick's end)."""
    upd, closed = apply_events({}, _ev([(0, 1, "A", "B", 30, 100.0, "NQZ6"),
                                        (5, 1, "C", "B", 30, 100.0, "NQZ6")]))
    assert closed == {1} and 1 not in upd


def test_events_for_multiple_orders_are_independent():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([
        (5, 2, "A", "A", 40, 105.0, "NQZ6"),
        (6, 1, "M", "B", 25, 100.0, "NQZ6"),
    ]))
    assert closed == set()
    assert upd[1].size == 25 and upd[2].size == 40 and upd[2].side == "A"


# --- snapshot reconciliation (ADR-148) -------------------------------------------------------

def test_order_absent_from_a_snapshot_group_is_dropped():
    """THE weekend-purge bug: CME kills a Day order with no Cancel, so the order's absence from
    the next snapshot group is the only evidence it is gone."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0)),      # survives
           2: RestingOrder("NQZ6", "A", 200.0, 99, _ts(0), _ts(0))}      # purged, never cancelled
    upd, closed = apply_events(cur, _ev(_snap(60, [(1, "B", 30, 100.0)])))
    assert 2 in closed and 2 not in upd
    assert set(upd) | (set(cur) - closed) == {1}


def test_snapshot_relisting_preserves_true_t_start():
    """The t_start-truncation bug: a nightly re-listing must not re-date a long-resting order."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev(_snap(60, [(1, "B", 30, 100.0)])))
    assert not closed
    carried = upd.get(1, cur[1])
    assert carried.t_start == _ts(0) and carried.t_start_exact is True


def test_order_first_seen_in_a_snapshot_is_marked_inexact():
    upd, closed = apply_events({}, _ev(_snap(60, [(7, "A", 50, 250.0)])))
    assert upd[7].t_start == _ts(60) and upd[7].t_start_exact is False


def test_snapshot_size_change_keeps_origin_but_price_change_marks_inexact():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0)),
           2: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, _ = apply_events(cur, _ev(_snap(60, [(1, "B", 20, 100.0),      # size only
                                              (2, "B", 30, 101.0)])))   # price moved unseen
    assert upd[1].size == 20 and upd[1].t_start == _ts(0) and upd[1].t_start_exact is True
    assert upd[2].price == 101.0 and upd[2].t_start == _ts(0) and upd[2].t_start_exact is False


def test_events_after_a_snapshot_group_apply_on_top_of_it():
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev(
        _snap(60, [(1, "B", 30, 100.0), (2, "A", 40, 105.0)])
        + [(70, 2, "C", "A", 40, 105.0, "NQZ6"),
           (75, 3, "A", "B", 25, 99.0, "NQZ6")]))
    assert 2 in closed and 2 not in upd
    assert upd[3].size == 25 and upd[3].t_start_exact is True


def test_real_exchange_clear_empties_the_book():
    """Weekly Sunday wipe with no snapshot re-listing in the same batch: honour the Clear."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0)),
           2: RestingOrder("NQZ6", "A", 200.0, 99, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([(60, 0, "R", "N", 0, 0.0, "NQZ6")]))
    assert closed == {1, 2} and not upd


def test_clear_then_relisting_in_one_batch_restores_true_t_start():
    """The real weekly sequence: Clear at 09:32 CT, full re-listing ~11 min later, both inside one
    30-min tick. Survivors must keep their true age, not be re-dated to the re-listing."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0)),
           2: RestingOrder("NQZ6", "A", 200.0, 99, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev(
        [(60, 0, "R", "N", 0, 0.0, "NQZ6")] + _snap(120, [(1, "B", 30, 100.0)])))
    assert 2 in closed                                   # not re-listed => genuinely purged
    carried = upd.get(1, cur[1])
    assert carried.t_start == _ts(0) and carried.t_start_exact is True


def test_clear_then_relisting_carries_unchanged_survivors_not_just_changed_ones():
    """Regression, found by replaying real 2026-09-27 data: the Clear marks every order closed and
    the re-listing restores survivors unchanged, so they never enter `updated`. `closed` must not
    still claim them, or the caller's carry-forward arithmetic drops the entire book and writes the
    all-closed sentinel instead of the re-listing's contents."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0)),
           2: RestingOrder("NQZ6", "A", 200.0, 99, _ts(0), _ts(0)),
           3: RestingOrder("NQZ6", "B", 90.0, 40, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev(
        [(60, 0, "R", "N", 0, 0.0, "NQZ6")]
        + _snap(120, [(1, "B", 30, 100.0),     # survives, completely unchanged
                      (2, "A", 99, 200.0)])))  # survives, completely unchanged
    assert closed == {3}, "only the un-relisted order is closed"
    assert 1 not in closed and 2 not in closed
    # What the caller will actually persist must be the full re-listing, not just `updated`.
    new_state = {o: v for o, v in cur.items() if o not in upd and o not in closed} | upd
    assert set(new_state) == {1, 2}
    assert new_state[1].t_start == _ts(0) and new_state[2].t_start == _ts(0)


def test_synthetic_clear_marker_inside_a_snapshot_group_is_not_a_wipe():
    """The daily 19:00 CT group leads with an R that carries F_SNAPSHOT at sequence 0. The group
    is authoritative; that marker must not blank the book on its own."""
    cur = {1: RestingOrder("NQZ6", "B", 100.0, 30, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev(
        [(60, 0, "R", "N", 0, 0.0, "NQZ6", F_SNAPSHOT)] + _snap(60, [(1, "B", 30, 100.0)])))
    assert not closed
    assert (upd.get(1) or cur[1]).t_start == _ts(0)


def test_frame_without_a_flags_column_still_works():
    """Back-compat: callers predating flag plumbing get pure incremental behaviour."""
    ev = _ev([(0, 1, "A", "B", 30, 100.0, "NQZ6")]).drop(columns=["flags"])
    upd, closed = apply_events({}, ev)
    assert upd[1].size == 30 and not closed


def test_a_fill_at_another_price_does_not_shrink_the_resting_order():
    """A marketable limit can sweep (F rows at the prices it took) and then rest the remainder.
    Those aggressive fills must not decrement the resting order -- real case 6882409490573."""
    cur = {1: RestingOrder("NQZ6", "B", 30561.50, 40, _ts(0), _ts(0))}
    upd, closed = apply_events(cur, _ev([
        (5, 1, "F", "B", 8, 30570.00, "NQZ6"),   # bought aggressively 8.5 pts higher
    ]))
    assert not closed and not upd, "resting size at 30561.50 is untouched"
    assert cur[1].size == 40


def test_a_fill_at_the_orders_own_price_still_shrinks_it():
    cur = {1: RestingOrder("NQZ6", "B", 30561.50, 40, _ts(0), _ts(0))}
    upd, _ = apply_events(cur, _ev([(5, 1, "F", "B", 8, 30561.50, "NQZ6")]))
    assert upd[1].size == 32
