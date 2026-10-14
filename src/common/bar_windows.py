"""Central bar-window conversion helper.

All feature modules should use the functions in this module to convert
time-based window specifications (in minutes or seconds) to bar counts
for the current timestep.  This ensures that when the timestep changes
(e.g. 10s → 5s) every rolling window automatically adjusts.

Usage
-----
    from src.common.bar_windows import bars, bars_m

    # 5 minutes → bar count for current timestep
    bars_m(5)   # → 60 at 5s, 30 at 10s

    # 30 seconds → bar count
    bars(30)    # → 6 at 5s, 3 at 10s
"""

from __future__ import annotations

import os

_TIMESTEP_SECONDS: int | None = None


def _get_timestep_seconds() -> int:
    """Resolve the current timestep in seconds (cached after first call)."""
    global _TIMESTEP_SECONDS
    if _TIMESTEP_SECONDS is not None:
        return _TIMESTEP_SECONDS

    raw = os.environ.get("VTECH_FEATURES_TIMESTEP", "5s")
    raw = raw.strip().lower()
    if raw.endswith("s"):
        _TIMESTEP_SECONDS = int(raw[:-1])
    elif raw.endswith("min"):
        _TIMESTEP_SECONDS = int(raw[:-3]) * 60
    elif raw.endswith("m"):
        _TIMESTEP_SECONDS = int(raw[:-1]) * 60
    else:
        _TIMESTEP_SECONDS = int(raw)
    return _TIMESTEP_SECONDS


def reset() -> None:
    """Reset the cached timestep (for testing)."""
    global _TIMESTEP_SECONDS
    _TIMESTEP_SECONDS = None


def bars(seconds: int) -> int:
    """Convert *seconds* to the number of bars at the current timestep.

    Returns at least 1.
    """
    return max(1, seconds // _get_timestep_seconds())


def bars_m(minutes: float) -> int:
    """Convert *minutes* to the number of bars at the current timestep.

    Returns at least 1.
    """
    return max(1, int(minutes * 60) // _get_timestep_seconds())


def get_timestep() -> str:
    """Return the timestep string (e.g. '5s') suitable for pd.resample."""
    secs = _get_timestep_seconds()
    return f"{secs}s"
