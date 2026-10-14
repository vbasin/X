"""NQ Globex market-session boundary utilities.

Session definition
------------------
NQ Globex session D spans **4:00 PM CT on D-1** through **3:00 PM CT on D**.
A tick at or after 3:00 PM CT belongs to session D+1 (maintenance window
and next-day overnight).  This matches the CME-published settlement
reference date (ts_ref) for NQ statistics records.

Distinct from the ML *trading session* (8:00–11:00 AM CT) used for
feature matrix time-filtering.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

CHICAGO = ZoneInfo("America/Chicago")

# CME equity-index weekend halt: Friday 4:00 PM CT → Sunday 5:00 PM CT.
_CME_FRI_CLOSE_MIN = 16 * 60
_CME_SUN_OPEN_MIN = 17 * 60


def market_session_date(ts_utc: pd.Timestamp) -> date:
    """Map a UTC timestamp to its NQ Globex session date.

    Session D: 4:00 PM CT D-1 → 3:00 PM CT D.
    Any tick at/after 3:00 PM CT belongs to session D+1.
    """
    ts_ct = ts_utc.tz_convert(CHICAGO)
    d = ts_ct.date()
    if ts_ct.hour >= 15:
        d = (ts_ct + timedelta(days=1)).date()
    return d


def _as_chicago(now: datetime | None) -> datetime:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(CHICAGO)


def cme_weekend_closed(now: datetime | None = None) -> bool:
    """True during the CME weekend halt (Fri 4pm CT – Sun 5pm CT).

    Weekday maintenance (4–5pm CT Mon–Thu) is *not* the weekend halt: the
    live streamer stays connected through that hour. Saturday 00:00 CT is
    inside this window, which is when ``live_option_session_date`` jumps
    Friday → Monday — the streamer must not open a new archive then.
    """
    ct = _as_chicago(now)
    minutes = ct.hour * 60 + ct.minute
    weekday = ct.weekday()  # Mon=0 … Sun=6
    if weekday == 4 and minutes >= _CME_FRI_CLOSE_MIN:
        return True
    if weekday == 5:
        return True
    if weekday == 6 and minutes < _CME_SUN_OPEN_MIN:
        return True
    return False


def cme_next_open(now: datetime | None = None) -> datetime:
    """Next Sunday 5:00 PM America/Chicago at or after ``now``."""
    ct = _as_chicago(now)
    days_ahead = (6 - ct.weekday()) % 7  # 0 on Sunday
    candidate_date = ct.date() + timedelta(days=days_ahead)
    hour, minute = divmod(_CME_SUN_OPEN_MIN, 60)
    candidate = datetime(
        candidate_date.year, candidate_date.month, candidate_date.day,
        hour, minute, tzinfo=CHICAGO,
    )
    if ct >= candidate:
        candidate_date = candidate_date + timedelta(days=7)
        candidate = datetime(
            candidate_date.year, candidate_date.month, candidate_date.day,
            hour, minute, tzinfo=CHICAGO,
        )
    return candidate


def cme_next_close(now: datetime | None = None) -> datetime:
    """Next Friday 4:00 PM America/Chicago at or after ``now``."""
    ct = _as_chicago(now)
    days_ahead = (4 - ct.weekday()) % 7  # 0 on Friday
    candidate_date = ct.date() + timedelta(days=days_ahead)
    hour, minute = divmod(_CME_FRI_CLOSE_MIN, 60)
    candidate = datetime(
        candidate_date.year, candidate_date.month, candidate_date.day,
        hour, minute, tzinfo=CHICAGO,
    )
    if ct >= candidate:
        candidate_date = candidate_date + timedelta(days=7)
        candidate = datetime(
            candidate_date.year, candidate_date.month, candidate_date.day,
            hour, minute, tzinfo=CHICAGO,
        )
    return candidate


def seconds_until_cme_open(now: datetime | None = None) -> float:
    """0 if CME is in session; otherwise seconds until Sunday 5pm CT."""
    if not cme_weekend_closed(now):
        return 0.0
    nxt = cme_next_open(now)
    current = _as_chicago(now)
    return max(0.0, (nxt - current).total_seconds())


def seconds_until_cme_close(now: datetime | None = None) -> float:
    """Seconds until Friday 4pm CT. 0 if already in the weekend halt."""
    if cme_weekend_closed(now):
        return 0.0
    nxt = cme_next_close(now)
    current = _as_chicago(now)
    return max(0.0, (nxt - current).total_seconds())


def rth_open_utc(d: date) -> pd.Timestamp:
    """Return 8:30 AM Chicago time for *d* as a UTC-aware Timestamp.

    Handles CDT/CST automatically:
      * CDT (Mar–Nov): 8:30 AM CDT = 13:30 UTC
      * CST (Nov–Mar): 8:30 AM CST = 14:30 UTC
    """
    dt_ct = datetime(d.year, d.month, d.day, 8, 30, tzinfo=CHICAGO)
    return pd.Timestamp(dt_ct).tz_convert("UTC")


def rth_close_utc(d: date) -> pd.Timestamp:
    """Return 3:00 PM Chicago time for *d* as a UTC-aware Timestamp.

    * CDT: 3:00 PM CDT = 20:00 UTC
    * CST: 3:00 PM CST = 21:00 UTC
    """
    dt_ct = datetime(d.year, d.month, d.day, 15, 0, tzinfo=CHICAGO)
    return pd.Timestamp(dt_ct).tz_convert("UTC")
