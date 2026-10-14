"""Single-writer journal — the research system of record (component C14, W5).

Wraps the ``research_director.research_journal`` + ``research_journal_events`` tables. The
Director is the only caller that mutates control-plane state; the Conductor owns the
execution-phase transitions on its own runs (same tables, consistent pattern). Every state
change appends an immutable event row, so the lifecycle is fully auditable.

States follow RESEARCH_DIRECTOR.md §9. Reuses :func:`common.db.get_pool`; JSONB columns are
written with ``psycopg.types.json.Jsonb`` (psycopg3 does not auto-adapt ``dict``).
"""

from __future__ import annotations

import logging
from typing import Any

from psycopg.types.json import Jsonb

from common.db import get_pool

log = logging.getLogger(__name__)

# ── Lifecycle states ─────────────────────────────────────────────────────────
PROPOSED = "PROPOSED"
PLANNED = "PLANNED"
SPEC_VALIDATED = "SPEC_VALIDATED"
QUEUED = "QUEUED"
RUNNING = "RUNNING"
EVALUATING = "EVALUATING"
STAGED_PENDING_PROMOTION = "STAGED_PENDING_PROMOTION"
PROMOTED = "PROMOTED"
SUCCEEDED_CLOSED = "SUCCEEDED_CLOSED"
FAILED_RUN = "FAILED_RUN"
DISPROVEN_CLOSED = "DISPROVEN_CLOSED"
NEEDS_REFINEMENT = "NEEDS_REFINEMENT"
INCONCLUSIVE = "INCONCLUSIVE"
NO_OP = "NO_OP"
EXTERNAL_DATA_REQUIRED = "EXTERNAL_DATA_REQUIRED"

# ── Model-improvement loop states (RD rebuild) ───────────────────────────────
# The RD now emits a ModelImprovementPlan and auto-validates its top pick synchronously, so the
# lifecycle is: PROPOSED → PLAN_SYNTHESIZED → one terminal verdict. (The old config_diff execution
# FSM — QUEUED/RUNNING/EVALUATING/promotion — is retired; those constants remain only for any
# residual readers.)
PLAN_SYNTHESIZED = "PLAN_SYNTHESIZED"
VALIDATED_IMPROVED = "VALIDATED_IMPROVED"
VALIDATED_NO_IMPROVEMENT = "VALIDATED_NO_IMPROVEMENT"
HANDOFF_REQUIRED = "HANDOFF_REQUIRED"

# Ideas that can be (re)selected for work.
OPEN_STATES = (PROPOSED, NEEDS_REFINEMENT)
# Execution states that a crash can leave stranded (recovered by reconcile()).
EXECUTION_STATES = (QUEUED, RUNNING, EVALUATING)

# Agent verdict → terminal journal state.
_VERDICT_STATE = {
    "SUCCEEDED": SUCCEEDED_CLOSED,
    "DISPROVEN": DISPROVEN_CLOSED,
    "NEEDS_REFINEMENT": NEEDS_REFINEMENT,
    "INCONCLUSIVE": INCONCLUSIVE,
}

_SCHEMA = "research_director"


class Journal:
    """CRUD + state-machine interface to the experiment journal."""

    def __init__(self, pool: Any = None) -> None:
        self._pool = pool or get_pool()

    # ── writes ────────────────────────────────────────────────────────────────
    def create_experiment(
        self,
        *,
        program_id: str,
        title: str,
        scope: str | None,
        spec_type: str | None,
        hypothesis: str | None,
        config_diff: dict[str, Any] | None,
        metric: str | None,
        delta_min: float | None,
        folds: int | None,
        seed: int | None,
        est_epochs: int | None,
        est_cost_usd: float | None,
        priority: float | None,
        spec_hash: str | None,
        created_by: str = "director",
        data_requirement: dict[str, Any] | None = None,
        state: str = PROPOSED,
    ) -> int:
        """Insert one experiment row + its creation event; returns the experiment_id."""
        with self._pool.connection() as conn:
            row = conn.execute(
                f"""
                INSERT INTO {_SCHEMA}.research_journal
                    (program_id, state, scope, spec_type, title, hypothesis, config_diff,
                     metric, delta_min, folds, seed, est_epochs, est_cost_usd, priority,
                     spec_hash, data_requirement, created_by)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING experiment_id
                """,
                (
                    program_id, state, scope, spec_type, title, hypothesis,
                    Jsonb(config_diff) if config_diff is not None else None,
                    metric, delta_min, folds, seed, est_epochs, est_cost_usd, priority,
                    spec_hash, Jsonb(data_requirement) if data_requirement else None,
                    created_by,
                ),
            ).fetchone()
            eid = int(row[0])
            conn.execute(
                f"""
                INSERT INTO {_SCHEMA}.research_journal_events
                    (experiment_id, from_state, to_state, actor, reason)
                VALUES (%s, NULL, %s, %s, %s)
                """,
                (eid, state, created_by, "created"),
            )
        return eid

    def transition(
        self,
        experiment_id: int,
        to_state: str,
        actor: str,
        *,
        reason: str | None = None,
        run_id: int | None = None,
    ) -> None:
        """Atomic state update + immutable event row."""
        with self._pool.connection() as conn:
            cur = conn.execute(
                f"SELECT state FROM {_SCHEMA}.research_journal WHERE experiment_id=%s",
                (experiment_id,),
            ).fetchone()
            from_state = cur[0] if cur else None
            conn.execute(
                f"UPDATE {_SCHEMA}.research_journal SET state=%s, updated_at=now() WHERE experiment_id=%s",
                (to_state, experiment_id),
            )
            conn.execute(
                f"""
                INSERT INTO {_SCHEMA}.research_journal_events
                    (experiment_id, run_id, from_state, to_state, actor, reason)
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (experiment_id, run_id, from_state, to_state, actor, reason),
            )

    def set_conclusion(
        self,
        experiment_id: int,
        *,
        verdict: str,
        closure_reason: str | None = None,
        tweaks: list[str] | None = None,
        actor: str = "agent",
    ) -> str:
        """Write the agent's post-analysis verdict and move to the matching terminal state."""
        to_state = _VERDICT_STATE.get(verdict, INCONCLUSIVE)
        with self._pool.connection() as conn:
            conn.execute(
                f"""UPDATE {_SCHEMA}.research_journal
                    SET closure_reason=%s, tweaks=%s, updated_at=now()
                    WHERE experiment_id=%s""",
                (closure_reason, Jsonb(tweaks) if tweaks else None, experiment_id),
            )
        self.transition(experiment_id, to_state, actor, reason=closure_reason)
        return to_state

    def record_external_data(
        self,
        experiment_id: int,
        data_requirement: dict[str, Any] | None,
        actor: str = "director",
    ) -> None:
        """Store the journaled data requirement and set EXTERNAL_DATA_REQUIRED."""
        with self._pool.connection() as conn:
            conn.execute(
                f"""UPDATE {_SCHEMA}.research_journal
                    SET data_requirement=%s, updated_at=now() WHERE experiment_id=%s""",
                (Jsonb(data_requirement) if data_requirement else None, experiment_id),
            )
        self.transition(experiment_id, EXTERNAL_DATA_REQUIRED, actor, reason="external data needed")

    # ── reads ─────────────────────────────────────────────────────────────────
    _GET_COLS = (
        "experiment_id", "program_id", "state", "scope", "spec_type", "title", "hypothesis",
        "config_diff", "metric", "delta_min", "folds", "seed", "est_epochs", "est_cost_usd",
        "priority", "spec_hash", "data_requirement", "closure_reason", "tweaks",
    )

    def get(self, experiment_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {', '.join(self._GET_COLS)} FROM {_SCHEMA}.research_journal "
                f"WHERE experiment_id=%s",
                (experiment_id,),
            ).fetchone()
        return dict(zip(self._GET_COLS, row)) if row else None

    def select_open(self, program_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Open ideas (PROPOSED / NEEDS_REFINEMENT) ranked by priority, oldest-first tiebreak."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT experiment_id, state, scope, title, priority
                FROM {_SCHEMA}.research_journal
                WHERE program_id=%s AND state = ANY(%s)
                ORDER BY priority DESC NULLS LAST, updated_at ASC
                LIMIT %s
                """,
                (program_id, list(OPEN_STATES), limit),
            ).fetchall()
        cols = ["experiment_id", "state", "scope", "title", "priority"]
        return [dict(zip(cols, r)) for r in rows]

    def recent_signatures(self, program_id: str, *, limit: int = 50) -> list[str]:
        """spec_hash list for the Policy Engine's near-duplicate dedup gate."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT spec_hash FROM {_SCHEMA}.research_journal
                WHERE program_id=%s AND spec_hash IS NOT NULL
                ORDER BY updated_at DESC LIMIT %s
                """,
                (program_id, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def open_count(self, program_id: str) -> int:
        """Count open ideas — drives backlog-replenish (discovery mode below a floor)."""
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT count(*) FROM {_SCHEMA}.research_journal "
                f"WHERE program_id=%s AND state = ANY(%s)",
                (program_id, list(OPEN_STATES)),
            ).fetchone()
        return int(row[0])

    def stalled(self, program_id: str) -> list[dict[str, Any]]:
        """Experiments stranded in an execution state (recovered by reconcile())."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT experiment_id, state, updated_at
                FROM {_SCHEMA}.research_journal
                WHERE program_id=%s AND state = ANY(%s)
                ORDER BY updated_at ASC
                """,
                (program_id, list(EXECUTION_STATES)),
            ).fetchall()
        cols = ["experiment_id", "state", "updated_at"]
        return [dict(zip(cols, r)) for r in rows]
