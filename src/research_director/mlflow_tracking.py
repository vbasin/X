"""MLflow run-nesting + cost/verdict logging for the research loop (W7, RESEARCH_DIRECTOR §7).

The lineage spine: each research **iteration** is a parent run; each agent's **experiment** is
a nested run; cost tally + the deterministic verdict + journal artifacts are logged. Mirrors
the graceful-degradation pattern in ``trainer.py`` — if MLflow is unavailable or disabled,
every method is a safe no-op, so the loop runs without a tracking server (and the unit tests
exercise the no-op path with no network).

Full MLflow 3 Tracing (OpenTelemetry) of the agent tool loop is a follow-up; this module
provides the run-nesting + metrics/artifacts lineage that ADR-012 requires.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(
        self,
        *,
        enabled: bool = True,
        tracking_uri: str | None = None,
        experiment_name: str = "nq-50pt-intraday",
    ) -> None:
        self._mlflow: Any = None
        if not enabled:
            return
        try:
            import mlflow  # noqa: PLC0415
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001 - tracking must never break the loop
            log.warning("MLflowTracker disabled (%s) — running without MLflow", exc)
            self._mlflow = None

    @property
    def enabled(self) -> bool:
        return self._mlflow is not None

    @contextmanager
    def iteration(self, program_id: str, cycle: int) -> Iterator[Any]:
        """Parent run for one research cycle."""
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=f"cycle-{cycle}") as run:
            self._safe(lambda: self._mlflow.set_tags(
                {"program_id": program_id, "cycle": cycle, "kind": "iteration"}))
            yield run

    @contextmanager
    def agent_run(self, agent: str, experiment_id: int) -> Iterator[Any]:
        """Nested run for one agent's experiment under the current iteration."""
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=f"{agent}-exp{experiment_id}", nested=True) as run:
            self._safe(lambda: self._mlflow.set_tags(
                {"agent": agent, "experiment_id": experiment_id, "kind": "agent"}))
            yield run

    def log_cost(self, tally: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key, value in tally.items():
            if isinstance(value, (int, float)):
                self._safe(lambda k=key, v=value: self._mlflow.log_metric(k, float(v)))

    def log_verdict(self, *, gate: str, verdict: str, primary_delta: float | None = None) -> None:
        if not self.enabled:
            return
        self._safe(lambda: self._mlflow.set_tags({"gate": gate, "verdict": verdict}))
        if primary_delta is not None:
            self._safe(lambda: self._mlflow.log_metric("primary_delta", float(primary_delta)))

    def log_text(self, name: str, text: str) -> None:
        if not self.enabled:
            return
        self._safe(lambda: self._mlflow.log_text(text, name))

    @staticmethod
    def _safe(fn: Any) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - logging must never break the loop
            log.debug("MLflowTracker log failed: %s", exc)
