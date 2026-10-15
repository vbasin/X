"""Real SQLite registry tests for the synthetic objective promotion boundary.

Model computation and HTTP are exercised by the full workflow. These tests use
small artifact fixtures and replace only that expensive inference acceptance
step; prediction gates, MLflow lifecycle/aliases and canonical pointers are real.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiment.workflow_gates import evaluate_gate
from src.experiment.workflow_tracking import WorkflowTracker
from src.serving import workflow_promotion as module
from src.serving.workflow_model import write_bundle_integrity

pytestmark = pytest.mark.ml_workflow


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    directory = tmp_path / "research" / "runs" / module.OBJECTIVE_ID / "test-run"
    directory.mkdir(parents=True)
    policy = {
        "min_accuracy": 0.8, "min_macro_f1": 0.8, "min_class_recall": 0.75,
        "min_rows": 60, "min_class_rows": 10, "min_prior_log_loss_improvement": 0.2,
        "persistence_accuracy_margin": 0.02, "max_log_loss": 0.5, "max_runtime_seconds": 900,
    }
    config = {"seed": 123, "max_runtime_seconds": 900}
    objective = {"objective_id": module.OBJECTIVE_ID, "synthetic_only": True,
                 "gates": policy, "model_config": config}
    _json(directory / "objective.json", objective)
    _json(directory / "manifest.json", {"objective_id": module.OBJECTIVE_ID})
    (directory / "synthetic.parquet").write_bytes(b"source transport fixture; full inference test revalidates the real dataset")
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.sqlite'}"
    _json(directory / "request.json", {"tracking_uri": tracking_uri, "objective": objective,
        "objective_sha256": module._sha(directory / "objective.json"), "config": config})
    labels = np.tile(np.arange(3), 20)
    probabilities = np.full((len(labels), 3), 0.05)
    probabilities[np.arange(len(labels)), labels] = 0.9
    prior = np.full_like(probabilities, 1 / 3)
    persistence = probabilities.copy()
    for partition in ("validation", "holdout"):
        np.savez(directory / f"{partition}_evidence.npz", labels=labels, probabilities=probabilities,
                 prior_probability=prior, persistence_probability=persistence)
        _json(directory / f"{partition}_gate.json", evaluate_gate(labels, probabilities, prior, persistence, policy))
    _json(directory / "negative_controls.json", {
        "prior_only": evaluate_gate(labels, prior, prior, persistence, policy),
        "shuffled_predictions": evaluate_gate(labels, np.random.default_rng(config["seed"] + 1).permutation(probabilities),
                                               prior, persistence, policy),
    })
    bundle = directory / "pass_1" / "model_abc"
    bundle.mkdir(parents=True)
    (bundle / "model.keras").write_bytes(b"small registry transport fixture")
    _json(bundle / "feature_schema.json", {"feature_cols": ["signal"]})
    _json(bundle / "finalize_manifest.json", {"candidate_id": "abc", "objective_id": module.OBJECTIVE_ID,
        "family": "transformer", "lifecycle": "EXPERIMENTAL", "synthetic_only": True})
    write_bundle_integrity(bundle)
    deployment = directory / "deployment"
    deployment.mkdir()
    (deployment / "MLmodel").write_text("transport fixture; no inference deserialization in this test")
    http = {key: True for key in ("verified", "probability_parity", "direction_parity", "timestamp_parity", "local_only", "server_stopped")}
    http["invalid_feature_status"] = 400
    _json(directory / "deployment_verification.json", {"artifact_download_reload": True, "probability_parity": True, "http": http})
    _json(directory / "selection.json", {"selected": "abc", "holdout_consulted": False})
    _json(directory / "holdout_consultation.json", {"selected": "abc", "consultations": 1})
    for filename in ("promotion_input.json", "promotion_expected.json"):
        _json(directory / filename, {"columns": [], "index": [], "data": []})
    tracker = WorkflowTracker(tracking_uri, "promotion-test", (tmp_path / "artifacts").as_uri())
    with tracker.run("workflow") as parent:
        with tracker.run("selected", parent_run_id=parent.run_id) as child:
            child.log_params({"depth": 1})
        parent.log_artifacts(deployment, "deployment")
        registered = parent.register_model("workflow-test-promotion", "deployment")
    _json(directory / "registry.json", registered)
    _json(directory / "summary.json", {"selected_model_id": "abc", "mlflow_parent_run_id": parent.run_id,
        "designs": [{"model_id": "abc", "directory": str(bundle), "mlflow_run_id": child.run_id}]})
    _json(directory / "completion.json", {"status": "PASSED", "tracking_status": "FINISHED",
        "mlflow_parent_run_id": parent.run_id, "elapsed_seconds": 10.0})
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_verify_prediction_evidence", lambda *args: None)
    monkeypatch.setattr(module, "_verify_promoted_inference", lambda **kwargs: {
        "alias_uri": f"models:/{registered['name']}@{module.ALIAS}", "alias_version": registered["version"],
        "alias_reload_verified": True, "canonical_bundle_verified": True, "probability_parity": True, "http": http,
    })
    return directory, tracker, registered, bundle, child.run_id


def test_real_registry_promotion_commits_alias_and_canonical_pointer(completed_run) -> None:
    from src.serving.promote import resolve_latest_bundle

    directory, tracker, registered, bundle, _ = completed_run
    original = module._sha(bundle / "finalize_manifest.json")
    receipt = module.promote_test_objective(directory)
    alias = tracker.client.get_model_version_by_alias(registered["name"], module.ALIAS)
    assert str(alias.version) == registered["version"]
    assert alias.tags["lifecycle"] == "PRODUCTION"
    assert receipt["lifecycle"] == "PRODUCTION"
    assert receipt["training_origin_lifecycle"] == "EXPERIMENTAL"
    assert receipt["trading_activation"] is False
    assert [transition["to"] for transition in receipt["transitions"]] == ["CANDIDATE", "PRODUCTION"]
    assert resolve_latest_bundle(module.OBJECTIVE_ID) == Path(receipt["path"])
    assert Path(receipt["mlflow_model_path"], "MLmodel").is_file()
    assert module._sha(bundle / "finalize_manifest.json") == original
    assert json.loads((directory / "promotion.json").read_text()) == receipt


@pytest.mark.parametrize("problem", ["gate", "predictions", "control", "objective", "bundle", "holdout", "http", "child", "budget"])
def test_invalid_evidence_never_creates_serving_alias_or_pointer(completed_run, problem: str) -> None:
    from src.serving.promote import latest_pointer_path

    directory, tracker, registered, bundle, child_id = completed_run
    if problem == "gate":
        gate = json.loads((directory / "holdout_gate.json").read_text())
        gate["metrics"]["accuracy"] = 0.99
        _json(directory / "holdout_gate.json", gate)
    elif problem == "predictions":
        path = directory / "holdout_evidence.npz"
        with np.load(path) as evidence:
            values = {key: evidence[key].copy() for key in evidence.files}
        values["probabilities"][:] = 1 / 3
        np.savez(path, **values)
    elif problem == "control":
        _json(directory / "negative_controls.json", {})
    elif problem == "objective":
        objective = json.loads((directory / "objective.json").read_text())
        objective["gates"]["min_accuracy"] = 0.5
        _json(directory / "objective.json", objective)
    elif problem == "bundle":
        (bundle / "model.keras").write_bytes(b"corrupted")
    elif problem == "holdout":
        _json(directory / "holdout_consultation.json", {"selected": "abc", "consultations": 2})
    elif problem == "http":
        data = json.loads((directory / "deployment_verification.json").read_text())
        data["http"]["invalid_feature_status"] = 200
        _json(directory / "deployment_verification.json", data)
    elif problem == "child":
        tracker.client.set_terminated(child_id, "FAILED")
    elif problem == "budget":
        data = json.loads((directory / "completion.json").read_text())
        data["elapsed_seconds"] = 901
        _json(directory / "completion.json", data)
    with pytest.raises((ValueError, RuntimeError)):
        module.promote_test_objective(directory)
    assert tracker.client.get_registered_model(registered["name"]).aliases == {}
    assert not latest_pointer_path(module.OBJECTIVE_ID).exists()


def test_failed_alias_inference_rolls_back_new_alias_and_retains_failure(completed_run, monkeypatch) -> None:
    from src.serving.promote import latest_pointer_path

    directory, tracker, registered, _, _ = completed_run
    def fail(**kwargs):
        raise RuntimeError("HTTP prediction mismatch")
    monkeypatch.setattr(module, "_verify_promoted_inference", fail)
    with pytest.raises(RuntimeError, match="HTTP prediction mismatch"):
        module.promote_test_objective(directory)
    assert tracker.client.get_registered_model(registered["name"]).aliases == {}
    assert tracker.client.get_model_version(registered["name"], registered["version"]).tags["lifecycle"] == "CANDIDATE"
    latest = latest_pointer_path(module.OBJECTIVE_ID)
    assert not latest.exists()
    failure = json.loads((latest.parent / directory.name / "PROMOTION_FAILURE.json").read_text())
    assert failure["canonical_pointer_published"] is False


def test_existing_pointer_is_preserved(completed_run) -> None:
    from src.serving.promote import latest_pointer_path

    directory, tracker, registered, _, _ = completed_run
    latest = latest_pointer_path(module.OBJECTIVE_ID)
    latest.parent.mkdir(parents=True)
    latest.write_text('{"preserve":"existing model"}')
    with pytest.raises(FileExistsError, match="already has LATEST"):
        module.promote_test_objective(directory)
    assert latest.read_text() == '{"preserve":"existing model"}'
    assert tracker.client.get_registered_model(registered["name"]).aliases == {}


def test_existing_alias_is_preserved(completed_run) -> None:
    directory, tracker, registered, _, _ = completed_run
    tracker.client.set_registered_model_alias(registered["name"], "existing", registered["version"])
    with pytest.raises(ValueError, match="existing deployment alias"):
        module.promote_test_objective(directory)
    aliases = tracker.client.get_registered_model(registered["name"]).aliases
    assert {name: str(version) for name, version in aliases.items()} == {"existing": registered["version"]}


def test_concurrent_winner_preserved_and_losing_alias_removed(completed_run, monkeypatch) -> None:
    from src.serving.promote import latest_pointer_path

    directory, tracker, registered, _, _ = completed_run
    latest = latest_pointer_path(module.OBJECTIVE_ID)
    accepted = module._verify_promoted_inference
    winner = {"candidate_id": "another-run", "lifecycle": "PRODUCTION"}

    def another_run_publishes(**kwargs):
        # Simulate a different run committing between our preflight and commit.
        module._write_once(latest, winner)
        return accepted(**kwargs)

    monkeypatch.setattr(module, "_verify_promoted_inference", another_run_publishes)
    with pytest.raises(FileExistsError):
        module.promote_test_objective(directory)
    assert json.loads(latest.read_text()) == winner
    assert tracker.client.get_registered_model(registered["name"]).aliases == {}
    assert tracker.client.get_model_version(registered["name"], registered["version"]).tags["lifecycle"] == "DEPRECATED"
    failure = json.loads((latest.parent / directory.name / "PROMOTION_FAILURE.json").read_text())
    assert failure["canonical_pointer_published"] is False
    assert failure["canonical_pointer_exists"] is True


def test_real_mlflow_alias_download_metadata_preserves_payload_hash(completed_run, tmp_path, monkeypatch) -> None:
    import mlflow.artifacts

    directory, tracker, registered, _, _ = completed_run
    # MLflow 3.10's models:/ repository also resolves an ambient client, even
    # when download_artifacts receives explicit URIs. Match the runner's pin.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracker.tracking_uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracker.tracking_uri)
    tracker.client.set_registered_model_alias(registered["name"], module.ALIAS, registered["version"])
    destination = tmp_path / "downloaded-alias"
    destination.mkdir()
    downloaded = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{registered['name']}@{module.ALIAS}", dst_path=str(destination),
        tracking_uri=tracker.tracking_uri, registry_uri=tracker.tracking_uri,
    ))
    assert (downloaded / "registered_model_meta").is_file()
    original_hash = module._package_sha(directory / "deployment")
    assert module._package_sha(downloaded) != original_hash
    assert module._package_sha(downloaded, registered_model=(registered["name"], registered["version"])) == original_hash


@pytest.mark.parametrize("metadata", [
    None,
    {"model_name": "wrong-model", "model_version": 1},
    {"model_name": "expected-model", "model_version": 2},
    {"model_name": "expected-model", "model_version": True},
    {"model_name": "expected-model", "model_version": 1, "unverified": "extra"},
    {"model_name": "expected-model"},
])
def test_registry_transport_metadata_must_match_exact_resolved_identity(tmp_path, metadata) -> None:
    (tmp_path / "MLmodel").write_text("model payload")
    if metadata is not None:
        _json(tmp_path / "registered_model_meta", metadata)
    with pytest.raises(ValueError, match="registry metadata|registry alias"):
        module._package_sha(tmp_path, registered_model=("expected-model", "1"))


def test_valid_transport_metadata_does_not_hide_other_payload_changes(tmp_path) -> None:
    (tmp_path / "MLmodel").write_text("model payload")
    original = module._package_sha(tmp_path)
    _json(tmp_path / "registered_model_meta", {"model_name": "expected-model", "model_version": 1})
    identity = ("expected-model", "1")
    assert module._package_sha(tmp_path, registered_model=identity) == original
    (tmp_path / "unexpected-file").write_text("must remain part of payload comparison")
    assert module._package_sha(tmp_path, registered_model=identity) != original
    (tmp_path / "MLmodel").write_text("changed model payload")
    assert module._package_sha(tmp_path, registered_model=identity) != original
