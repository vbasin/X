"""CME quarterly futures contract calendar (NQ, ES, and similar roots).

Computes the active lead-month contract for a given date using the CME
roll convention: *the Monday prior to the third Friday of the expiration
month*. All roll dates are derived algorithmically from `calendar.monthcalendar`
— no hardcoded roll table is used.

Quarterly contract months (CME equity index futures):
    Mar (H), Jun (M), Sep (U), Dec (Z)

Symbol convention (Databento / CME GLBX.MDP3): single-letter month + single-
digit year, e.g. ``"NQZ5"`` for Dec 2025, ``"ESH6"`` for Mar 2026.

The lead contract for any given `d` is the earliest quarterly contract whose
roll date is **strictly after** `d`. On the roll date itself, the lead contract
has already flipped to the next quarter.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache

# Quarterly expiration months and their CME month codes
QUARTERLY_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
MONTH_CODE: dict[int, str] = {3: "H", 6: "M", 9: "U", 12: "Z"}


def _third_friday(year: int, month: int) -> date:
    """Return the date of the third Friday of (year, month)."""
    weeks = calendar.monthcalendar(year, month)
    fridays = [w[calendar.FRIDAY] for w in weeks if w[calendar.FRIDAY] != 0]
    return date(year, month, fridays[2])


@lru_cache(maxsize=256)
def roll_date(year: int, quarter_month: int) -> date:
    """Return the CME roll date for a given quarterly contract.

    Roll date = Monday prior to the third Friday of the expiration month
    (third Friday minus 4 days).

    Parameters
    ----------
    year : int
        Full year of the contract expiration (e.g. 2025).
    quarter_month : int
        One of {3, 6, 9, 12}.
    """
    if quarter_month not in QUARTERLY_MONTHS:
        raise ValueError(f"quarter_month must be in {QUARTERLY_MONTHS}, got {quarter_month}")
    return _third_friday(year, quarter_month) - timedelta(days=4)


def _symbol(root: str, year: int, quarter_month: int) -> str:
    """Format a CME futures symbol, e.g. 'NQ' + 2025 + 12 → 'NQZ5'."""
    return f"{root}{MONTH_CODE[quarter_month]}{year % 10}"


def active_contract(root: str, d: date) -> str:
    """Return the active lead-month contract symbol for `root` on date `d`.

    The active contract is the earliest quarterly contract whose roll date
    is strictly after `d`. On the roll date itself, the lead contract has
    already flipped to the next quarterly month.

    Examples
    --------
    >>> active_contract("NQ", date(2025, 12, 12))
    'NQZ5'
    >>> active_contract("NQ", date(2025, 12, 15))   # roll day
    'NQH6'
    >>> active_contract("ES", date(2026, 3, 16))    # Mar 2026 roll day
    'ESM6'
    """
    # Walk candidate quarters starting at the quarter containing `d`,
    # advancing forward until we find one whose roll date is > d.
    year = d.year
    # Start at the first quarter of the current year and scan forward.
    for _ in range(6):  # safety cap; always terminates within 2 years
        for qm in QUARTERLY_MONTHS:
            if roll_date(year, qm) > d:
                return _symbol(root, year, qm)
        year += 1
    raise RuntimeError(f"Could not resolve active contract for {root} on {d}")


def active_contracts_in_range(
    root: str, start: date, end: date
) -> list[tuple[date, str]]:
    """Return `[(date, active_symbol), ...]` for every calendar day in [start, end].

    Includes both endpoints. Weekends are included; callers should filter
    to trading days as needed.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    out: list[tuple[date, str]] = []
    d = start
    while d <= end:
        out.append((d, active_contract(root, d)))
        d += timedelta(days=1)
    return out


def is_roll_date(d: date) -> bool:
    """True if `d` is a CME quarterly roll date for equity index futures."""
    # Check if d matches any quarterly roll date in its own year.
    for qm in QUARTERLY_MONTHS:
        if roll_date(d.year, qm) == d:
            return True
    return False


def days_since_last_roll(d: date, max_lookback: int = 120) -> int:
    """Days since the most recent CME roll on or before `d`."""
    # Scan backward from `d` up to max_lookback days for a roll date.
    for i in range(max_lookback + 1):
        candidate = d - timedelta(days=i)
        if is_roll_date(candidate):
            return i
    raise RuntimeError(f"No roll date found within {max_lookback} days before {d}")


def roll_dates_in_range(start: date, end: date) -> list[date]:
    """Return all CME quarterly roll dates in [start, end] (inclusive)."""
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    rolls: list[date] = []
    # Scan quarters spanning the range, plus one year of padding on either side.
    for year in range(start.year, end.year + 2):
        for qm in QUARTERLY_MONTHS:
            rd = roll_date(year, qm)
            if start <= rd <= end:
                rolls.append(rd)
    rolls.sort()
    return rolls
