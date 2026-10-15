"""Real, isolated SQLite MLflow contract tests; no market DB or live service."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.workflow_tracking import WorkflowTracker


pytestmark = pytest.mark.ml_workflow


@pytest.fixture
def tracker(tmp_path: Path) -> WorkflowTracker:
    return WorkflowTracker(
        f"sqlite:///{tmp_path / 'tracking.sqlite'}", "transformer-smoke",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )


def test_nested_runs_persist_design_epoch_history_and_artifacts(
    tracker: WorkflowTracker, tmp_path: Path,
) -> None:
    import mlflow

    tracking_uri_before = mlflow.get_tracking_uri()
    registry_uri_before = mlflow.get_registry_uri()
    active_before = mlflow.active_run()
    artifact = tmp_path / "design.json"
    artifact.write_text('{"d_model": 8}')
    with tracker.run("workflow", tags={"objective_id": "transformer-workflow-test"}) as parent:
        with tracker.run("tiny-transformer", parent_run_id=parent.run_id) as child:
            child.log_params({"d_model": 8, "layers": [8, 16], "seed": 42})
            child.log_metrics({"loss": 0.8}, step=0)
            child.log_metrics({"loss": 0.3}, step=1)
            child.set_tags({"lifecycle": "EXPERIMENTAL"})
            child.log_artifact(artifact, "design")
            evidence = child.verify(params={"d_model": 8}, metrics={"loss": 0.3})
            assert evidence["artifacts"] == ["design/design.json"]
        parent.log_params({"selected_run_id": child.run_id})
        parent.log_metrics({"designs_completed": 1})

    assert child.verify(status="FINISHED")["verified"]
    assert parent.verify(status="FINISHED")["verified"]
    stored = tracker.client.get_run(child.run_id)
    assert stored.data.tags["mlflow.parentRunId"] == parent.run_id
    assert stored.data.params["layers"] == "[8,16]"
    assert [(item.step, item.value) for item in tracker.client.get_metric_history(child.run_id, "loss")] == [(0, 0.8), (1, 0.3)]
    assert mlflow.get_tracking_uri() == tracking_uri_before
    assert mlflow.get_registry_uri() == registry_uri_before
    assert mlflow.active_run() is active_before


def test_training_failure_marks_both_child_and_parent_failed(tracker: WorkflowTracker) -> None:
    with pytest.raises(RuntimeError, match="training exploded"):
        with tracker.run("parent") as parent:
            with tracker.run("child", parent_run_id=parent.run_id) as child:
                raise RuntimeError("training exploded")
    assert tracker.client.get_run(parent.run_id).info.status == "FAILED"
    assert tracker.client.get_run(child.run_id).info.status == "FAILED"
    assert tracker.client.get_run(child.run_id).data.tags["vtech.failure_type"] == "RuntimeError"


def test_backend_logging_failure_is_not_silenced(
    tracker: WorkflowTracker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise OSError("backend unavailable")

    monkeypatch.setattr(tracker.client, "log_batch", fail)
    with pytest.raises(OSError, match="backend unavailable"):
        with tracker.run("failed-log") as run:
            run.log_params({"d_model": 8})
    assert tracker.client.get_run(run.run_id).info.status == "FAILED"


def test_readback_failure_prevents_success(tracker: WorkflowTracker) -> None:
    with pytest.raises(RuntimeError, match="artifact missing.json"):
        with tracker.run("missing-evidence") as run:
            run.log_params({"d_model": 8})
            run.verify(artifacts=["missing.json"])
    assert tracker.client.get_run(run.run_id).info.status == "FAILED"


def test_conflicting_design_parameter_fails_run(tracker: WorkflowTracker) -> None:
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException):
        with tracker.run("immutable-design") as run:
            run.log_params({"d_model": 8})
            run.log_params({"d_model": 16})
    assert tracker.client.get_run(run.run_id).info.status == "FAILED"


def test_registry_roundtrip_preserves_experimental_source_and_has_no_alias(
    tracker: WorkflowTracker, tmp_path: Path,
) -> None:
    bundle = tmp_path / "serving"
    bundle.mkdir()
    (bundle / "model.keras").write_bytes(b"registry transport fixture, not a fitted model")
    (bundle / "feature_schema.json").write_text('{"feature_cols":["signal"]}')
    with tracker.run("registry-check") as run:
        run.log_artifacts(bundle, "serving")
        recorded = run.register_model("workflow-test-transformer", "serving")
    model = tracker.client.get_registered_model(recorded["name"])
    version = tracker.client.get_model_version(recorded["name"], recorded["version"])
    assert model.aliases == {}
    assert version.run_id == run.run_id
    assert version.tags["lifecycle"] == "EXPERIMENTAL"
    assert version.current_stage == "None"
    assert version.source.endswith(f"{run.run_id}/artifacts/serving")
    downloaded = tracker.client.download_artifacts(run.run_id, "serving/model.keras", str(tmp_path))
    assert Path(downloaded).read_bytes() == (bundle / "model.keras").read_bytes()


def test_registry_refuses_production_name_and_unknown_source(tracker: WorkflowTracker) -> None:
    with tracker.run("registry-guards") as run:
        with pytest.raises(ValueError, match="workflow-test-"):
            run.register_model("nq-50pt-econ", "serving")
        with pytest.raises(ValueError, match="existing artifact"):
            run.register_model("workflow-test-empty", "serving")


def test_invalid_metrics_paths_and_closed_run_are_rejected(
    tracker: WorkflowTracker, tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.json"
    artifact.write_text("{}")
    with tracker.run("input-guards") as run:
        with pytest.raises(ValueError, match="finite"):
            run.log_metrics({"loss": float("nan")})
        with pytest.raises(ValueError, match="step"):
            run.log_metrics({"loss": 0.1}, step=-1)
        with pytest.raises(ValueError, match="relative"):
            run.log_artifact(artifact, "../escape")
    with pytest.raises(RuntimeError, match="closed"):
        run.log_params({"after_finish": True})


def test_experiment_artifact_root_conflict_is_not_silently_accepted(
    tracker: WorkflowTracker, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="different artifact"):
        WorkflowTracker(
            tracker.tracking_uri, tracker.experiment_name,
            artifact_location=(tmp_path / "different-artifacts").as_uri(),
        )


def test_parent_requires_matching_experiment_and_running_state(
    tracker: WorkflowTracker, tmp_path: Path,
) -> None:
    other = WorkflowTracker(tracker.tracking_uri, "other", (tmp_path / "other-artifacts").as_uri())
    with tracker.run("parent") as parent:
        with pytest.raises(ValueError, match="this workflow experiment"):
            with other.run("child", parent_run_id=parent.run_id):
                pass
    with pytest.raises(ValueError, match="RUNNING"):
        with tracker.run("late-child", parent_run_id=parent.run_id):
            pass
