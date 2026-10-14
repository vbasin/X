"""Sentinel — observability + tiered alerting (component #19, W10; CPU-side).

Runs health checks (DB, MLflow, research-loop activity) and emits tiered alerts:
  - **P0** (page):   DB unreachable; daily $ budget breach.
  - **P1** (alert):  repeated run failures; MLflow tracking server down.
  - **P2** (digest): opportunity drought (no new experiments); stale pending promotions.

The GPU-orphan **dead-man's-switch is W9-coupled and intentionally not wired here** — no GPU
is contacted. ``gpu_orphan_check()`` is a logged advisory until the GPU plane lands (W9).

The classification logic (:func:`classify`) is pure and unit-testable; ``Sentinel.check``
gathers inputs from the journal + ledger and logs each alert at its tier's level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from common.db import get_pool

log = logging.getLogger(__name__)

P0 = "P0"
P1 = "P1"
P2 = "P2"
_SCHEMA = "research_director"
_TIER_LEVEL = {P0: logging.CRITICAL, P1: logging.WARNING, P2: logging.INFO}


@dataclass(frozen=True)
class Alert:
    tier: str
    code: str
    message: str


@dataclass
class HealthInputs:
    db_ok: bool
    mlflow_ok: bool
    fail_count_24h: int
    last_experiment_age_h: float | None        # None => no experiments ever
    stale_promotions: int
    spend_day: float
    usd_cap_day: float


def classify(
    h: HealthInputs,
    *,
    fail_streak_threshold: int = 3,
    drought_age_h: float = 48.0,
) -> list[Alert]:
    """Pure mapping from health inputs to tiered alerts (no I/O)."""
    alerts: list[Alert] = []
    if not h.db_ok:
        alerts.append(Alert(P0, "DB_DOWN", "research_director DB unreachable"))
    if h.usd_cap_day > 0 and h.spend_day >= h.usd_cap_day:
        alerts.append(Alert(P0, "BUDGET_BREACH",
                            f"daily spend ${h.spend_day:.2f} >= cap ${h.usd_cap_day:.2f}"))
    if not h.mlflow_ok:
        alerts.append(Alert(P1, "MLFLOW_DOWN", "MLflow tracking server not responding"))
    if h.fail_count_24h >= fail_streak_threshold:
        alerts.append(Alert(P1, "REPEATED_FAILURES", f"{h.fail_count_24h} FAILED_RUN in 24h"))
    if h.last_experiment_age_h is None or h.last_experiment_age_h > drought_age_h:
        age = "never" if h.last_experiment_age_h is None else f"{h.last_experiment_age_h:.0f}h ago"
        alerts.append(Alert(P2, "OPPORTUNITY_DROUGHT", f"last experiment {age}"))
    if h.stale_promotions > 0:
        alerts.append(Alert(P2, "PENDING_PROMOTIONS",
                            f"{h.stale_promotions} model(s) await operator ack"))
    return alerts


class Sentinel:
    def __init__(
        self,
        *,
        pool=None,
        mlflow_uri: str | None = None,
        usd_cap_day: float = 20.0,
        fail_streak_threshold: int = 3,
        drought_age_h: float = 48.0,
        stale_promotion_age_h: float = 72.0,
    ) -> None:
        self._pool = pool or get_pool()
        self._mlflow_uri = mlflow_uri
        self._usd_cap_day = usd_cap_day
        self._fail_streak = fail_streak_threshold
        self._drought_age_h = drought_age_h
        self._stale_h = stale_promotion_age_h

    # ── public ───────────────────────────────────────────────────────────────
    def gather(self) -> HealthInputs:
        db_ok = self._db_ok()
        return HealthInputs(
            db_ok=db_ok,
            mlflow_ok=self._mlflow_ok(),
            fail_count_24h=self._fail_count_24h() if db_ok else 0,
            last_experiment_age_h=self._last_experiment_age_h() if db_ok else None,
            stale_promotions=self._stale_promotions() if db_ok else 0,
            spend_day=self._spend_day() if db_ok else 0.0,
            usd_cap_day=self._usd_cap_day,
        )

    def check(self) -> list[Alert]:
        alerts = classify(
            self.gather(),
            fail_streak_threshold=self._fail_streak,
            drought_age_h=self._drought_age_h,
        )
        for a in alerts:
            log.log(_TIER_LEVEL.get(a.tier, logging.INFO), "[%s] %s: %s", a.tier, a.code, a.message)
        return alerts

    def gpu_orphan_check(self) -> list[Alert]:
        """Deferred to W9 (GPU plane). Intentionally does NOT contact the GPU instance."""
        log.debug("Sentinel.gpu_orphan_check: deferred to W9 — no GPU contacted")
        return []

    # ── queries (constants in SQL; only data values are parameterized) ─────────
    def _db_ok(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - fail-closed health probe
            return False

    def _fail_count_24h(self) -> int:
        with self._pool.connection() as conn:
            return int(conn.execute(
                f"SELECT count(*) FROM {_SCHEMA}.research_journal "
                f"WHERE state='FAILED_RUN' AND updated_at > now() - interval '1 day'"
            ).fetchone()[0])

    def _last_experiment_age_h(self) -> float | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT EXTRACT(EPOCH FROM now() - max(created_at)) / 3600 "
                f"FROM {_SCHEMA}.research_journal"
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def _stale_promotions(self) -> int:
        with self._pool.connection() as conn:
            return int(conn.execute(
                f"SELECT count(*) FROM {_SCHEMA}.research_journal "
                f"WHERE state='STAGED_PENDING_PROMOTION' "
                f"AND updated_at < now() - make_interval(hours => %s)",
                (int(self._stale_h),),
            ).fetchone()[0])

    def _spend_day(self) -> float:
        with self._pool.connection() as conn:
            return float(conn.execute(
                f"SELECT COALESCE(SUM(usd_debited), 0) FROM {_SCHEMA}.gpu_budget_ledger "
                f"WHERE ts > now() - interval '1 day'"
            ).fetchone()[0])

    def _mlflow_ok(self) -> bool:
        if not self._mlflow_uri:
            return True                        # not configured → don't alert
        try:
            import urllib.request  # noqa: PLC0415
            with urllib.request.urlopen(self._mlflow_uri, timeout=3) as resp:  # noqa: S310
                return 200 <= resp.status < 500
        except Exception:  # noqa: BLE001 - any failure means not reachable
            return False
