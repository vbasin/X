"""CME weekend halt and session-boundary helpers."""
from datetime import datetime, timezone

from src.common.market_session import (
    cme_next_close,
    cme_next_open,
    cme_weekend_closed,
    seconds_until_cme_close,
    seconds_until_cme_open,
)


def test_cme_weekend_closed_friday_4pm_ct() -> None:
    # 2026-08-21 is Friday. CDT = UTC-5.
    still_open = datetime(2026, 8, 21, 20, 59, tzinfo=timezone.utc)  # 15:59 CT
    just_closed = datetime(2026, 8, 21, 21, 0, tzinfo=timezone.utc)  # 16:00 CT
    assert cme_weekend_closed(still_open) is False
    assert cme_weekend_closed(just_closed) is True


def test_cme_weekend_closed_saturday() -> None:
    sat = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
    assert cme_weekend_closed(sat) is True
    assert seconds_until_cme_open(sat) > 0
    assert seconds_until_cme_close(sat) == 0.0


def test_cme_weekend_open_sunday_5pm_ct() -> None:
    before = datetime(2026, 8, 23, 21, 59, tzinfo=timezone.utc)  # 16:59 CT
    open_at = datetime(2026, 8, 23, 22, 0, tzinfo=timezone.utc)  # 17:00 CT
    assert cme_weekend_closed(before) is True
    assert cme_weekend_closed(open_at) is False
    assert seconds_until_cme_open(open_at) == 0.0


def test_cme_next_open_from_saturday_is_this_sunday() -> None:
    sat = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
    nxt = cme_next_open(sat)
    assert nxt.tzinfo is not None
    assert (nxt.year, nxt.month, nxt.day) == (2026, 8, 23)
    assert nxt.hour == 17


def test_cme_next_close_from_monday_is_this_friday() -> None:
    mon = datetime(2026, 8, 24, 14, 0, tzinfo=timezone.utc)
    nxt = cme_next_close(mon)
    assert (nxt.year, nxt.month, nxt.day) == (2026, 8, 28)
    assert nxt.hour == 16
