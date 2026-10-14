"""Tests for the Globex session-date rule and the five-chain root set.

The session date is what every subscription decision keys on, so these tests
pin the boundary behaviour precisely: it advances at the **16:00 CT close**,
not at Chicago midnight, because a CME equity-index session runs 17:00 CT to
16:00 CT the next day and is named for the day it ends.

The rule that matters in practice: at Monday 17:00 CT the chain expiring
*Tuesday* is 0DTE and the chain expiring *Wednesday* is 1DTE.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.acquisition.options_universe import (
    globex_session_date,
    live_option_session_date,
    session_option_targets,
    target_roots,
)

pytestmark = pytest.mark.acquisition

CT = ZoneInfo("America/Chicago")


def _ct(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=CT)


# ── the boundary ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("moment", "expected", "why"),
    [
        (_ct(2026, 9, 27, 18, 0), date(2026, 9, 28), "Sunday open settles Monday"),
        (_ct(2026, 9, 28, 9, 0), date(2026, 9, 28), "Monday RTH settles Monday"),
        (_ct(2026, 9, 28, 15, 59), date(2026, 9, 28), "one minute before the close"),
        (_ct(2026, 9, 28, 16, 0), date(2026, 9, 28), "closing snapshot stays with the ending session"),
        (_ct(2026, 9, 28, 16, 4), date(2026, 9, 28), "one minute before the roll"),
        (_ct(2026, 9, 28, 16, 5), date(2026, 9, 29), "the roll point"),
        (_ct(2026, 9, 28, 16, 30), date(2026, 9, 29), "inside the halt"),
        (_ct(2026, 9, 28, 18, 0), date(2026, 9, 29), "Monday evening settles Tuesday"),
        (_ct(2026, 10, 2, 15, 0), date(2026, 10, 2), "Friday RTH settles Friday"),
        (_ct(2026, 10, 3, 10, 0), date(2026, 10, 5), "Saturday → next trading day"),
    ],
)
def test_session_date_advances_at_the_close(
    moment: datetime, expected: date, why: str
) -> None:
    assert globex_session_date(moment) == expected, why


def test_the_whole_roll_sits_inside_the_dead_window() -> None:
    """16:01–16:44 CT writes no quotes, so the boundary costs no data.

    Measured on bbo_1s: the 16:00 CT minute carries one closing quote per
    symbol and 16:01 through 16:44 carries nothing at all, every session.
    """
    assert globex_session_date(_ct(2026, 9, 28, 15, 59)) == date(2026, 9, 28)
    # The roll point and the far end of the dead window both give the new
    # session, so the boundary cannot land where quotes are being written.
    assert globex_session_date(_ct(2026, 9, 28, 16, 5)) == date(2026, 9, 29)
    assert globex_session_date(_ct(2026, 9, 28, 16, 44)) == date(2026, 9, 29)
    # 16:00 is the closing snapshot: ~1,300 rows, one per instrument. It must
    # file under the session that is ending, not the one about to open.
    assert globex_session_date(_ct(2026, 9, 28, 16, 0)) == date(2026, 9, 28)


def test_non_trading_days_are_skipped_from_either_side_of_the_close() -> None:
    """Thanksgiving week, which steps over more than one day.

    ``src/common/cme_holidays.py`` excludes early-close sessions as well as full
    closures, on the grounds that truncated volume distorts features. So
    2026-11-26 (Thanksgiving, closed) *and* 2026-11-27 (open, early close) are
    both non-trading here, and the session after Wednesday's close is Monday.

    That calendar is tuned for feature building and is being reused for
    acquisition, where the trade-off is the opposite way round — see
    ``test_early_close_sessions_are_not_captured`` below.
    """
    assert globex_session_date(_ct(2026, 11, 25, 18, 0)) == date(2026, 11, 30)
    assert globex_session_date(_ct(2026, 11, 26, 10, 0)) == date(2026, 11, 30)
    assert globex_session_date(_ct(2026, 11, 27, 10, 0)) == date(2026, 11, 30)


def test_early_close_sessions_are_not_captured() -> None:
    """Documents an inherited gap rather than asserting desired behaviour.

    On an early-close session the resolver names the *next* full session, so the
    five chains subscribed are the wrong ones for the session actually trading:
    on 2026-11-27 it resolves Monday 11-30's chains while Friday's shortened
    session trades. Acquisition would rather capture a short session than skip
    it; excluding it loses roughly eight sessions a year.

    Change this test when the acquisition calendar is split from the
    feature-building one; it is here so that change is deliberate.
    """
    early_close = _ct(2026, 11, 27, 10, 0)
    assert globex_session_date(early_close) != date(2026, 11, 27)
    assert globex_session_date(early_close) == date(2026, 11, 30)


def test_utc_input_is_converted_before_the_comparison() -> None:
    """The caller passes UTC; 21:05 UTC is 16:05 CDT and must roll."""
    # 21:05 UTC is 16:05 CDT, the roll point.
    assert globex_session_date(
        datetime(2026, 9, 28, 21, 4, tzinfo=timezone.utc)
    ) == date(2026, 9, 28)
    assert globex_session_date(
        datetime(2026, 9, 28, 21, 5, tzinfo=timezone.utc)
    ) == date(2026, 9, 29)


def test_live_option_session_date_delegates() -> None:
    """One implementation of the calendar, so the two cannot drift apart."""
    for moment in (
        _ct(2026, 9, 28, 9, 0),
        _ct(2026, 9, 28, 18, 0),
        _ct(2026, 9, 27, 18, 0),
    ):
        assert live_option_session_date(moment) == globex_session_date(moment)


# ── what the roll means for the chains ───────────────────────────


def test_monday_evening_puts_tuesday_on_0dte_and_wednesday_on_1dte() -> None:
    """The operator-stated contract, asserted on expiry dates not root names."""
    session = globex_session_date(_ct(2026, 9, 28, 17, 0))
    assert session == date(2026, 9, 29)

    by_cat = {t.category: t for t in session_option_targets(session)}
    assert by_cat["0dte"].expiry == date(2026, 9, 29), "0DTE expires Tuesday"
    assert by_cat["1dte"].expiry == date(2026, 9, 30), "1DTE expires Wednesday"


def test_pre_roll_monday_afternoon_still_names_monday() -> None:
    """Before the close the open session is Monday's, so 0DTE expires Monday."""
    session = globex_session_date(_ct(2026, 9, 28, 14, 0))
    by_cat = {t.category: t for t in session_option_targets(session)}
    assert by_cat["0dte"].expiry == date(2026, 9, 28)
    assert by_cat["1dte"].expiry == date(2026, 9, 29)


# ── target_roots ─────────────────────────────────────────────────


def test_target_roots_is_the_five_categories_deduplicated() -> None:
    roots = target_roots(date(2026, 9, 28))
    assert roots == {"Q4AU6", "Q5BU6", "QNEU6", "QNEV6", "NQZ6"}


def test_target_roots_collapses_when_two_categories_share_a_chain() -> None:
    """On 2026-09-29 the next trading day *is* the month end.

    ``1dte`` and ``qne1`` both resolve to ``QNEU6``, so the session subscribes
    four distinct chains. The count is 3–5, never a fixed five.
    """
    session = date(2026, 9, 29)
    by_cat = {t.category: t.root for t in session_option_targets(session)}
    assert by_cat["1dte"] == by_cat["qne1"] == "QNEU6"
    assert len(target_roots(session)) == 4


def test_target_roots_rejects_everything_outside_the_five_chains() -> None:
    """The membership test the subscription gate relies on."""
    roots = target_roots(date(2026, 9, 28))
    for outside in (
        "QN2U6",      # a weekly that is not one of the five
        "NQU6",       # the superseded quarterly
        "QNEX6",      # a later QNE month
        "UD:2V:",     # combo prefix
        "NQZ6 C25000",  # a full strike symbol, not a root
    ):
        assert outside not in roots


def test_target_roots_is_hashable_and_immutable() -> None:
    """Held across a long-lived session, so it must not be mutable by callers."""
    roots = target_roots(date(2026, 9, 28))
    assert isinstance(roots, frozenset)
    with pytest.raises(AttributeError):
        roots.add("NEW")  # type: ignore[attr-defined]
