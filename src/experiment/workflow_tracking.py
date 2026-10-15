"""Required MLflow evidence for the bounded end-to-end workflow.

Unlike the legacy best-effort loggers, a tracking failure fails this workflow.
Every operation uses an explicitly configured client; the fluent MLflow active
run and global tracking/registry URIs are never changed. Registry writes are
restricted to the ``workflow-test-`` namespace and never set a deployment alias.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path, PurePosixPath
import time
from typing import Any, Iterator, Mapping, Sequence


def _value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _artifact_path(path: str | None) -> str | None:
    if path is None:
        return None
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts or "\\" in path:
        raise ValueError("artifact path must be a nonempty relative POSIX path")
    return str(parsed)


class WorkflowTracker:
    """One explicit tracking backend and experiment, without global state.

    ``artifact_location`` sets the root only when creating a new experiment.
    Supply a temporary file URI with SQLite in tests. Existing server experiments
    retain their configured artifact root; a conflicting requested root fails.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> None:
        if not tracking_uri or not tracking_uri.strip():
            raise ValueError("tracking_uri is required")
        if not experiment_name or not experiment_name.strip():
            raise ValueError("experiment_name is required")
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient

        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            try:
                experiment_id = self.client.create_experiment(
                    experiment_name, artifact_location=artifact_location,
                    tags={"vtech.workflow": "transformer_smoke"},
                )
                experiment = self.client.get_experiment(experiment_id)
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None or experiment.lifecycle_stage != "active":
            raise RuntimeError("MLflow experiment is unavailable or deleted")
        if artifact_location is not None and (
            experiment.artifact_location.rstrip("/") != artifact_location.rstrip("/")
        ):
            raise ValueError("existing experiment has a different artifact location")
        self.experiment_id = experiment.experiment_id
        self.experiment_name = experiment_name

    @contextmanager
    def run(
        self,
        name: str,
        *,
        tags: Mapping[str, Any] | None = None,
        parent_run_id: str | None = None,
    ) -> Iterator[TrackedRun]:
        """Create a run, verify its evidence, and persist FINISHED or FAILED."""
        if not name:
            raise ValueError("run name is required")
        run_tags = {str(k): _value(v) for k, v in (tags or {}).items()}
        if "mlflow.parentRunId" in run_tags:
            raise ValueError("use parent_run_id for parent lineage")
        run_tags["vtech.workflow"] = "transformer_smoke"
        if parent_run_id is not None:
            parent = self.client.get_run(parent_run_id)
            if parent.info.experiment_id != self.experiment_id:
                raise ValueError("parent must belong to this workflow experiment")
            if parent.info.status != "RUNNING":
                raise ValueError("parent run must be RUNNING")
            run_tags["mlflow.parentRunId"] = parent_run_id
        created = self.client.create_run(self.experiment_id, run_name=name, tags=run_tags)
        tracked = TrackedRun(self, created.info.run_id)
        tracked._tags.update({**run_tags, "mlflow.runName": name})
        try:
            yield tracked
            tracked.verify()
            self.client.set_terminated(tracked.run_id, status="FINISHED")
            tracked.verify(status="FINISHED")
        except BaseException as exc:
            # Preserve the actual training/tracking failure if the backend also
            # fails while recording terminal state. Never turn it into success.
            try:
                self.client.set_tag(tracked.run_id, "vtech.failure_type", type(exc).__name__)
                self.client.set_terminated(tracked.run_id, status="FAILED")
            except Exception as terminal_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"MLflow failure status could not be persisted: {terminal_error}")
            raise
        finally:
            tracked._closed = True


class TrackedRun:
    """A run handle with synchronous writes and read-back verification."""

    def __init__(self, tracker: WorkflowTracker, run_id: str) -> None:
        self.tracker = tracker
        self.run_id = run_id
        self._params: dict[str, str] = {}
        self._metrics: dict[str, float] = {}
        self._metric_history: dict[tuple[str, int], float] = {}
        self._metric_steps: dict[str, int] = {}
        self._tags: dict[str, str] = {}
        self._artifacts: dict[str, int] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("run context is already closed")

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log complete design settings; values are canonical JSON or strings."""
        self._ensure_open()
        from mlflow.entities import Param

        values = {str(k): _value(v) for k, v in params.items()}
        self.tracker.client.log_batch(
            self.run_id, params=[Param(k, v) for k, v in values.items()], synchronous=True,
        )
        self._params.update(values)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int = 0) -> None:
        """Log finite metrics at an epoch step, preserving their full history."""
        self._ensure_open()
        from mlflow.entities import Metric

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("metric step must be a nonnegative integer")
        values = {str(k): float(v) for k, v in metrics.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("workflow metrics must be finite")
        timestamp = int(time.time() * 1000)
        self.tracker.client.log_batch(
            self.run_id,
            metrics=[Metric(k, v, timestamp, step) for k, v in values.items()],
            synchronous=True,
        )
        for key, value in values.items():
            if step >= self._metric_steps.get(key, -1):
                self._metrics[key] = value
                self._metric_steps[key] = step
            self._metric_history[key, step] = value

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        """Log provenance and measured workflow status."""
        self._ensure_open()
        from mlflow.entities import RunTag

        if "mlflow.parentRunId" in tags:
            raise ValueError("parent lineage is immutable in this workflow")
        values = {str(k): _value(v) for k, v in tags.items()}
        self.tracker.client.log_batch(
            self.run_id, tags=[RunTag(k, v) for k, v in values.items()], synchronous=True,
        )
        self._tags.update(values)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Upload one existing file and remember its expected stored size."""
        self._ensure_open()
        source = Path(path)
        destination = _artifact_path(artifact_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.tracker.client.log_artifact(self.run_id, str(source), destination)
        relative = str(PurePosixPath(destination or "") / source.name)
        self._artifacts[relative] = source.stat().st_size

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Upload a nonempty artifact directory, retaining every file expectation."""
        self._ensure_open()
        source = Path(path)
        destination = _artifact_path(artifact_path)
        if not source.is_dir():
            raise NotADirectoryError(source)
        files = [p for p in source.rglob("*") if p.is_file()]
        if not files:
            raise ValueError("artifact directory is empty")
        self.tracker.client.log_artifacts(self.run_id, str(source), destination)
        for file in files:
            relative = str(PurePosixPath(destination or "") / file.relative_to(source).as_posix())
            self._artifacts[relative] = file.stat().st_size

    def _listed_artifacts(self) -> dict[str, int | None]:
        result: dict[str, int | None] = {}
        pending: list[str | None] = [None]
        while pending:
            for item in self.tracker.client.list_artifacts(self.run_id, pending.pop()):
                if item.is_dir:
                    pending.append(item.path)
                else:
                    result[item.path] = item.file_size
        return result

    def verify(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        artifacts: Sequence[str] = (),
        status: str | None = None,
    ) -> dict[str, Any]:
        """Read the backend and require all recorded evidence plus expectations.

        Artifacts are checked using the repository's path and byte-size metadata;
        model load/prediction parity separately verifies checkpoint usability.
        """
        stored = self.tracker.client.get_run(self.run_id)
        errors: list[str] = []
        expected_params = {**self._params, **{str(k): _value(v) for k, v in (params or {}).items()}}
        for key, value in expected_params.items():
            if stored.data.params.get(key) != value:
                errors.append(f"parameter {key}")
        for key, value in {**self._metrics, **(metrics or {})}.items():
            actual = stored.data.metrics.get(key)
            if actual is None or not math.isclose(actual, float(value), rel_tol=1e-9, abs_tol=1e-12):
                errors.append(f"metric {key}")
        for key, value in self._tags.items():
            if stored.data.tags.get(key) != value:
                errors.append(f"tag {key}")
        for key in {key for key, _ in self._metric_history}:
            history = self.tracker.client.get_metric_history(self.run_id, key)
            for (metric_key, step), expected in self._metric_history.items():
                if metric_key == key and not any(
                    item.step == step and math.isclose(item.value, expected, rel_tol=1e-9, abs_tol=1e-12)
                    for item in history
                ):
                    errors.append(f"metric history {key} step={step}")
        listed = self._listed_artifacts()
        for path in set(self._artifacts).union(artifacts):
            if path not in listed:
                errors.append(f"artifact {path}")
            elif path in self._artifacts and listed[path] != self._artifacts[path]:
                errors.append(f"artifact size {path}")
        if status is not None and stored.info.status != status:
            errors.append(f"status {stored.info.status}, expected {status}")
        if errors:
            raise RuntimeError("MLflow read-back verification failed: " + "; ".join(errors))
        return {
            "run_id": self.run_id, "experiment_id": stored.info.experiment_id,
            "status": stored.info.status, "params": dict(stored.data.params),
            "metrics": dict(stored.data.metrics), "artifacts": sorted(listed),
            "verified": True,
        }

    def register_model(
        self,
        name: str,
        artifact_path: str,
        *,
        tags: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register an experimental test artifact; never assign a serving alias.

        This is a registry transport check. Loading the source bundle and checking
        its inference contract remain the runner's responsibility.
        """
        self._ensure_open()
        from mlflow.exceptions import MlflowException

        if not name.startswith("workflow-test-") or len(name) <= len("workflow-test-"):
            raise ValueError("registry smoke names must start with workflow-test-")
        path = _artifact_path(artifact_path)
        assert path is not None
        listed = self._listed_artifacts()
        if path not in listed and not any(item.startswith(path + "/") for item in listed):
            raise ValueError("registry source must be an existing artifact in this run")
        client = self.tracker.client
        registry_tags = {str(k): _value(v) for k, v in (tags or {}).items()}
        registry_tags.update({"vtech.workflow": "transformer_smoke", "lifecycle": "EXPERIMENTAL"})
        try:
            existing = client.get_registered_model(name)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            existing = client.create_registered_model(name, tags=registry_tags)
        if existing.tags.get("vtech.workflow") != "transformer_smoke":
            raise ValueError("refusing to modify a registered model outside the workflow")
        if existing.aliases:
            raise ValueError("workflow test registry must not carry deployment aliases")
        # Resolve through this client's run to avoid ambient fluent tracking URI
        # during runs:/ resolution inside older MLflow registry implementations.
        source = client.get_run(self.run_id).info.artifact_uri.rstrip("/") + "/" + path
        version = client.create_model_version(
            name, source, run_id=self.run_id, tags=registry_tags, await_creation_for=30,
        )
        recorded = client.get_model_version(name, version.version)
        if recorded.run_id != self.run_id or recorded.status != "READY":
            raise RuntimeError("MLflow model registry read-back verification failed")
        if recorded.tags.get("lifecycle") != "EXPERIMENTAL":
            raise RuntimeError("MLflow registry lost the experimental lifecycle tag")
        return {
            "name": name, "version": str(recorded.version), "source": recorded.source,
            "run_id": self.run_id, "status": recorded.status, "lifecycle": "EXPERIMENTAL",
        }
