"""Governor — pre-cycle AND-gate (W5).

Evaluated before any cycle starts. Every condition is AND-gated: a single failure returns a
logged no-go and the cycle becomes a no-op — the GPU is never touched. Holds a single global
advisory lock so overlapping timer fires cannot run two cycles at once.

Reuses :func:`common.db.get_pool` and the ``research_director.gpu_budget_ledger`` /
``research_journal_events`` tables. Caller contract: call :meth:`may_run`; if it returns
``ok=True`` the lock is held — call :meth:`release_lock` in a ``finally``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from common.db import get_pool

log = logging.getLogger(__name__)

# Stable 32-bit key for pg_advisory_lock (ascii 'RD' << 16 | 1).
_ADVISORY_KEY = 0x52440001
_SCHEMA = "research_director"


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str = ""


class Governor:
    def __init__(
        self,
        *,
        pool: Any = None,
        min_gap_s: int = 1800,                 # >= 30 min between cycles
        act_cap_24h_s: int = 4 * 3600,         # <= 4 GPU-hours per rolling 24 h
        usd_cap_day: float = 20.0,
        usd_cap_week: float = 80.0,
        suppression_windows: list[tuple[int, int, str]] | None = None,
    ) -> None:
        self._pool = pool or get_pool()
        self._min_gap_s = min_gap_s
        self._act_cap_24h_s = act_cap_24h_s
        self._usd_cap_day = usd_cap_day
        self._usd_cap_week = usd_cap_week
        # (utc_hour, utc_minute, label) ± 15 min; defaults: oci-sync 02:00, pg-backup 03:00.
        self._suppression = suppression_windows or [(2, 0, "oci-sync"), (3, 0, "pg-backup")]
        self._locked = False

    # ── public gate ────────────────────────────────────────────────────────────
    def may_run(self, *, now: Any = None) -> GateResult:
        if not self._try_lock():
            return GateResult(False, "another cycle holds the advisory lock")
        checks: list[Callable[[], GateResult]] = [
            lambda: self._check_suppression(now),
            self._check_min_gap,
            self._check_gpu_activity,
            self._check_spend,
            self._check_health,
        ]
        for check in checks:
            result = check()
            if not result.ok:
                self.release_lock()
                return result
        return GateResult(True)

    # ── advisory lock ───────────────────────────────────────────────────────────
    def _try_lock(self) -> bool:
        with self._pool.connection() as conn:
            got = conn.execute("SELECT pg_try_advisory_lock(%s)", (_ADVISORY_KEY,)).fetchone()[0]
        self._locked = bool(got)
        return self._locked

    def release_lock(self) -> None:
        if not self._locked:
            return
        with self._pool.connection() as conn:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_ADVISORY_KEY,))
        self._locked = False

    # ── individual gates ─────────────────────────────────────────────────────────
    def _check_suppression(self, now: Any = None) -> GateResult:
        from datetime import datetime, timezone
        now = now or datetime.now(timezone.utc)
        for hour, minute, label in self._suppression:
            if now.hour == hour and abs(now.minute - minute) <= 15:
                return GateResult(False, f"suppression window: {label}")
        return GateResult(True)

    def _check_min_gap(self) -> GateResult:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"""SELECT EXTRACT(EPOCH FROM now() - max(ts))
                    FROM {_SCHEMA}.research_journal_events
                    WHERE actor='director' AND reason='created'"""
            ).fetchone()
        gap = row[0]
        if gap is not None and gap < self._min_gap_s:
            return GateResult(False, f"last cycle {gap:.0f}s ago < min gap {self._min_gap_s}s")
        return GateResult(True)

    def _check_gpu_activity(self) -> GateResult:
        with self._pool.connection() as conn:
            secs = conn.execute(
                f"""SELECT COALESCE(SUM(seconds_active), 0) FROM {_SCHEMA}.gpu_budget_ledger
                    WHERE category='gpu' AND ts > now() - interval '24 hours'"""
            ).fetchone()[0]
        if secs >= self._act_cap_24h_s:
            return GateResult(False, f"gpu active {secs:.0f}s in 24h >= cap {self._act_cap_24h_s}s")
        return GateResult(True)

    def _check_spend(self) -> GateResult:
        with self._pool.connection() as conn:
            day = conn.execute(
                f"SELECT COALESCE(SUM(usd_debited),0) FROM {_SCHEMA}.gpu_budget_ledger "
                f"WHERE ts > now() - interval '1 day'"
            ).fetchone()[0]
            week = conn.execute(
                f"SELECT COALESCE(SUM(usd_debited),0) FROM {_SCHEMA}.gpu_budget_ledger "
                f"WHERE ts > now() - interval '7 days'"
            ).fetchone()[0]
        if day >= self._usd_cap_day:
            return GateResult(False, f"daily spend ${day:.2f} >= cap ${self._usd_cap_day:.2f}")
        if week >= self._usd_cap_week:
            return GateResult(False, f"weekly spend ${week:.2f} >= cap ${self._usd_cap_week:.2f}")
        return GateResult(True)

    def _check_health(self) -> GateResult:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
            return GateResult(True)
        except Exception as exc:  # noqa: BLE001 - fail-closed on any DB error
            return GateResult(False, f"db unhealthy: {exc}")
