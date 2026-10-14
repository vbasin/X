"""Economic calendar features: FOMC, CPI, NFP day flags.

Gap #17: Binary event-day flags + minutes-to-announcement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Known high-impact economic events for 2025-2026.
# These are FOMC announcement dates, CPI release dates, and NFP dates.
# Update or extend as needed.
_FOMC_DATES = {
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
}

# CPI is typically released on the 2nd or 3rd Tuesday/Wednesday of the month
_CPI_DATES = {
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-15", "2026-10-13", "2026-11-10", "2026-12-09",
}

# NFP is typically the first Friday of the month
_NFP_DATES = {
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
}

# All events have standard announcement times in CT
_EVENT_TIMES = {
    "fomc": (13, 0),   # 1:00 PM CT (2:00 PM ET)
    "cpi":  (7, 30),   # 7:30 AM CT (8:30 AM ET)
    "nfp":  (7, 30),   # 7:30 AM CT
}


def compute(
    index: pd.DatetimeIndex,
    timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """Compute economic calendar features.

    Parameters
    ----------
    index : DatetimeIndex at the feature timestep resolution.
    timezone : Timezone for computing minutes-to-event.
    """
    ts = index.tz_convert(timezone) if index.tz is not None else index.tz_localize("UTC").tz_convert(timezone)
    features = pd.DataFrame(index=index)

    dates_str = pd.Series(ts.date).astype(str).values

    features["ec_is_fomc_day"] = np.isin(dates_str, list(_FOMC_DATES)).astype(np.float32)
    features["ec_is_cpi_day"] = np.isin(dates_str, list(_CPI_DATES)).astype(np.float32)
    features["ec_is_nfp_day"] = np.isin(dates_str, list(_NFP_DATES)).astype(np.float32)

    # Minutes to nearest event today (0 if no event today)
    bar_minutes = ts.hour * 60 + ts.minute
    minutes_to_event = pd.Series(0.0, index=index, dtype=np.float32)

    for event_type, event_dates, (ev_h, ev_m) in [
        ("fomc", _FOMC_DATES, _EVENT_TIMES["fomc"]),
        ("cpi", _CPI_DATES, _EVENT_TIMES["cpi"]),
        ("nfp", _NFP_DATES, _EVENT_TIMES["nfp"]),
    ]:
        event_minute = ev_h * 60 + ev_m
        is_event_day = np.isin(dates_str, list(event_dates))
        remaining = event_minute - bar_minutes
        # Only show positive (before event); 0 after event has passed
        remaining = np.where(remaining > 0, remaining, 0)
        minutes_to_event = np.where(
            is_event_day & (remaining > minutes_to_event),
            remaining,
            minutes_to_event,
        )

    features["ec_minutes_to_event"] = minutes_to_event.astype(np.float32) if isinstance(minutes_to_event, np.ndarray) else minutes_to_event

    return features
