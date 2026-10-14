"""CME holiday calendar — dates to exclude from feature building and training.

Includes both full-close and early-close/modified-hours sessions, as both
produce abnormal volume/microstructure that contaminates ML features.

Coverage: 2025–2026.  Extend _HOLIDAYS for future years as needed.
"""

from __future__ import annotations

from datetime import date

# All dates where CME is fully closed OR running early/modified hours.
# Both categories are excluded: early-close sessions have truncated volume,
# abnormal bid-ask spreads, and missing post-close auction data.
_HOLIDAYS: frozenset[date] = frozenset({
    # ── 2025 ─────────────────────────────────────────────────────────────
    date(2025,  1,  1),   # New Year's Day (closed)
    date(2025,  1, 20),   # MLK Day (early close / modified)
    date(2025,  2, 17),   # Presidents' Day (early close / modified)
    date(2025,  4, 18),   # Good Friday (closed)
    date(2025,  5, 26),   # Memorial Day (early close / modified)
    date(2025,  6, 19),   # Juneteenth (early close / modified)
    date(2025,  7,  4),   # Independence Day (closed)
    date(2025,  9,  1),   # Labor Day (early close / modified)
    date(2025, 11, 11),   # Veterans Day (modified processing)
    date(2025, 11, 27),   # Thanksgiving (closed)
    date(2025, 11, 28),   # Black Friday (early close)
    date(2025, 12, 24),   # Christmas Eve (early close)
    date(2025, 12, 25),   # Christmas (closed)

    # ── 2026 ─────────────────────────────────────────────────────────────
    date(2026,  1,  1),   # New Year's Day (closed)
    date(2026,  1, 19),   # MLK Day (early close / modified)
    date(2026,  2, 16),   # Presidents' Day (early close / modified)
    date(2026,  4,  2),   # Day before Good Friday (early close)
    date(2026,  4,  3),   # Good Friday (closed)
    date(2026,  5, 22),   # Friday before Memorial Day (early close)
    date(2026,  5, 25),   # Memorial Day (closed)
    date(2026,  6, 19),   # Juneteenth (closed)
    date(2026,  7,  3),   # Independence Day observed (NQ early close 12:00 CT)
    date(2026,  9,  7),   # Labor Day (closed)
    date(2026, 11, 26),   # Thanksgiving (closed)
    date(2026, 11, 27),   # Black Friday (early close)
    date(2026, 12, 24),   # Christmas Eve (early close)
    date(2026, 12, 25),   # Christmas (closed)
})


def holiday_iso_dates() -> list[str]:
    """Sorted ``YYYY-MM-DD`` strings for the CME holiday set (UI / chart axes)."""
    return sorted(d.isoformat() for d in _HOLIDAYS)


def is_cme_holiday(d: date) -> bool:
    """True if ``d`` is a CME full-close or early/modified-hours date.

    Weekends are not holidays; use :func:`is_trading_day` to exclude both.
    """
    return d in _HOLIDAYS


def is_trading_day(d: date) -> bool:
    """Return True if ``d`` is a regular CME trading day.

    Excludes weekends and all CME full-close / early-close holiday dates.
    """
    return d.weekday() < 5 and d not in _HOLIDAYS


def trading_days_in_range(start: date, end: date) -> list[date]:
    """Return all trading days in [start, end) in ascending order."""
    from datetime import timedelta
    days = []
    current = start
    while current < end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days
