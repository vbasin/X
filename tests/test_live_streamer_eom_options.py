"""End-of-month (QNE) option symbols must parse in the live Greeks path.

``_register_resolved_weekly_from_bbo`` returns early when a symbol does not parse,
so an unparsed root never reaches ``_opt_meta`` and never enters ``_0dte_ids`` —
the live Greeks and IV-surface writes go silent for that whole session. Before QNE
was added this happened on every month-end expiry: on 2026-09-30 the entire 0DTE
chain was ``QNEU6``, raw BBO flowed normally, and
``databento.nq_option_surface_1s`` received nothing.
"""

from __future__ import annotations

import calendar
from datetime import date

import pytest

from src.acquisition.live_streamer import (
    _last_business_day_of_month,
    _parse_resolved_weekly_option_symbol,
)

pytestmark = pytest.mark.acquisition


def test_eom_chain_parses_and_expires_on_the_last_business_day():
    got = _parse_resolved_weekly_option_symbol("QNEU6 C24000", reference=date(2026, 9, 29))
    assert got is not None, "QNE root must parse"
    assert got["expiry"] == date(2026, 9, 30)
    assert got["instrument_class"] == "C"
    assert got["strike_price"] == 24000.0


def test_eom_settles_into_the_first_quarterly_at_or_after_expiry():
    """2026-09-30 is past the September quarterly, so it settles into NQZ6."""
    got = _parse_resolved_weekly_option_symbol("QNEU6 P30000", reference=date(2026, 9, 29))
    assert got["underlying"] == "NQZ6"


@pytest.mark.parametrize("symbol,expected", [
    ("Q1AU6 C24000", date(2026, 9, 7)),    # 1st Monday
    ("QN1U6 P30000", date(2026, 9, 4)),    # 1st Friday
])
def test_existing_weekly_roots_are_unchanged(symbol, expected):
    assert _parse_resolved_weekly_option_symbol(symbol, reference=date(2026, 9, 29))["expiry"] == expected


@pytest.mark.parametrize("year,month,expected", [
    (2026, 9, date(2026, 9, 30)),    # Wednesday
    (2026, 5, date(2026, 5, 29)),    # 31st is a Sunday -> Friday 29th
    (2026, 1, date(2026, 1, 30)),    # 31st is a Saturday -> Friday 30th
    (2026, 2, date(2026, 2, 27)),    # short month
])
def test_last_business_day_skips_weekends(year, month, expected):
    got = _last_business_day_of_month(year, month)
    assert got == expected
    assert got.weekday() < calendar.SATURDAY


def test_unknown_root_still_rejected():
    assert _parse_resolved_weekly_option_symbol("ZZZU6 C24000", reference=date(2026, 9, 29)) is None
