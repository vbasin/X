"""Fast orchestration safety contracts; no TensorFlow fit or live services."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from src.experiment import transformer_workflow as workflow

pytestmark = pytest.mark.ml_workflow


def test_config_round_trip_is_fully_resolved_and_unknown_fields_fail(tmp_path):
    path = tmp_path / "config.json"
    original = workflow.WorkflowConfig()
    path.write_text(json.dumps(asdict(original)))
    assert workflow.WorkflowConfig.read(path) == original
    path.write_text(json.dumps({**asdict(original), "unused_training_option": True}))
    with pytest.raises(TypeError, match="unused_training_option"):
        workflow.WorkflowConfig.read(path)


@pytest.mark.parametrize("overrides", [
    {"seed": True},
    {"objective_id": "nq-50pt-econ"},
    {"seed": -1},
    {"n_rows": 999},
    {"n_rows": 10001},
    {"horizon": 0},
    {"seq_len": 33},
    {"d_model": 15, "n_heads": 2},
    {"n_heads": 0},
    {"layers": (1,)},
    {"layers": (1, 1)},
    {"layers": (1, 4)},
    {"layers": (True, 2)},
    {"epochs": 31},
    {"batch_size": 0},
    {"learning_rate": float("nan")},
    {"dropout": float("inf")},
    {"confidence_threshold": 1.1},
    {"min_accuracy": -0.1},
    {"max_runtime_seconds": 0},
    {"max_runtime_seconds": 1741},
    {"max_runtime_seconds": 1800},
])
def test_config_rejects_invalid_shapes_hyperparameters_and_budgets(overrides):
    with pytest.raises(ValueError):
        workflow.WorkflowConfig(**overrides)


def test_record_is_immutable_and_preserves_existing_evidence(tmp_path):
    record = tmp_path / "nested" / "record.json"
    workflow.write_record(record, {"status": "frozen", "selected_model": "a"})
    original = record.read_bytes()
    with pytest.raises(FileExistsError):
        workflow.write_record(record, {"status": "retuned", "selected_model": "b"})
    assert record.read_bytes() == original


def test_invalid_record_never_creates_a_partial_immutable_artifact(tmp_path):
    record = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        workflow.write_record(record, {"metric": float("nan")})
    assert not record.exists()


def test_acceptance_requires_accuracy_floor_and_beating_train_prior():
    baseline = {"accuracy": 0.35, "log_loss": 1.1, "macro_f1": 0.2}
    workflow.acceptance_gate({"accuracy": 0.75, "log_loss": 0.5, "macro_f1": 0.7}, baseline, 0.5)
    for rejected in (
        {"accuracy": 0.49, "log_loss": 0.5, "macro_f1": 0.4},
        {"accuracy": 0.75, "log_loss": 1.1, "macro_f1": 0.7},
        {"accuracy": 0.75, "log_loss": 1.2, "macro_f1": 0.7},
        {"accuracy": 0.75, "log_loss": 0.5, "macro_f1": float("nan")},
    ):
        with pytest.raises(ValueError):
            workflow.acceptance_gate(rejected, baseline, 0.5)


def test_acceptance_rejects_nonfinite_baseline_instead_of_silently_passing():
    baseline = {"accuracy": 0.35, "log_loss": float("nan"), "macro_f1": 0.2}
    with pytest.raises(ValueError):
        workflow.acceptance_gate({"accuracy": 0.75, "log_loss": 0.5, "macro_f1": 0.7}, baseline, 0.5)


def test_supervisor_timeout_kills_actual_child_and_retains_failure_evidence(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []
    cleanup_calls = []

    def harmless_worker(args, **kwargs):
        # Exercise actual process creation, wait, process-group signaling and
        # reap behavior. The child only sleeps; no workflow/model is imported.
        child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(workflow.subprocess, "Popen", harmless_worker)
    monkeypatch.setattr(workflow, "_terminate_tracking", cleanup_calls.append)
    started = time.monotonic()
    try:
        assert workflow.supervise(tmp_path, timeout_seconds=1) == 124
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    assert time.monotonic() - started < 8
    assert len(children) == 1
    assert children[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(children[0].pid, 0)
    evidence = json.loads((tmp_path / "supervisor_failure.json").read_text())
    assert evidence["status"] == "TIMED_OUT"
    assert evidence["budget_seconds"] == 1
    assert evidence["elapsed_seconds"] >= 1
    assert (tmp_path / "worker.log").is_file()
    assert not (tmp_path / "summary.json").exists()
    assert cleanup_calls == [tmp_path]


def test_worker_exception_leaves_structured_failure(tmp_path, monkeypatch):
    def fail_execution(directory: Path):
        raise RuntimeError("intentional isolated test failure")

    monkeypatch.setattr(workflow, "execute", fail_execution)
    with pytest.raises(RuntimeError, match="intentional isolated"):
        workflow.main(["--execute", str(tmp_path)])
    evidence = json.loads((tmp_path / "failure.json").read_text())
    assert evidence == {"type": "RuntimeError", "error": "intentional isolated test failure"}


def test_serving_compatibility_fails_before_run_allocation(tmp_path, monkeypatch):
    def incompatible():
        raise RuntimeError("incompatible serving libraries")

    monkeypatch.setattr(workflow, "check_serving_environment", incompatible)
    root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="incompatible serving"):
        workflow.main(["--data-root", str(root)])
    assert not root.exists()


def test_crashed_worker_records_failure_and_closes_tracking(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    cleanup_calls = []

    def crashed_worker(args, **kwargs):
        return real_popen([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)

    monkeypatch.setattr(workflow.subprocess, "Popen", crashed_worker)
    monkeypatch.setattr(workflow, "_terminate_tracking", cleanup_calls.append)
    assert workflow.supervise(tmp_path, timeout_seconds=5) == 7
    evidence = json.loads((tmp_path / "supervisor_failure.json").read_text())
    assert evidence["status"] == "FAILED"
    assert evidence["worker_exit_code"] == 7
    assert cleanup_calls == [tmp_path]


def test_promotion_cannot_bypass_objective_contract_with_smoke_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**asdict(workflow.WorkflowConfig()), "objective_id": "synthetic-regime-service"}))
    with pytest.raises(SystemExit) as result:
        workflow.main(["--config", str(config), "--data-root", str(tmp_path / "artifacts")])
    assert result.value.code == 2
    assert not (tmp_path / "artifacts").exists()


def test_objective_settings_cannot_be_overridden(tmp_path):
    with pytest.raises(SystemExit) as result:
        workflow.main(["--objective", "anything.json", "--config", "override.json"])
    assert result.value.code == 2


def test_existing_promotion_is_preserved_before_allocating_new_run(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "check_serving_environment", lambda: None)
    pointer = tmp_path / "models/model_registry/synthetic-regime-service/LATEST.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"previous":"preserve"}')
    with pytest.raises(FileExistsError, match="already has a promoted model"):
        workflow.main(["--objective", str(workflow.REPO / "config/workflows/synthetic_regime_objective.json"),
                       "--data-root", str(tmp_path)])
    assert pointer.read_text() == '{"previous":"preserve"}'
    assert not (tmp_path / "research/runs").exists()
