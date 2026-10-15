"""Evidence-gated promotion for the isolated synthetic regime service objective.

The scientific bundle remains immutable. Current lifecycle belongs to the
promotion receipt and MLflow version, while the bundle's original training
manifest preserves its historical EXPERIMENTAL status. No trading objective,
paper service, or live configuration is changed by this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

from src.serving.lifecycle import LifecycleStatus, is_valid_transition

OBJECTIVE_ID = "synthetic-regime-service"
ALIAS = "test-serving"
_EVIDENCE_FILES = (
    "objective.json", "request.json", "selection.json", "validation_gate.json",
    "holdout_gate.json", "holdout_consultation.json", "deployment_verification.json",
    "registry.json", "completion.json", "summary.json", "validation_evidence.npz",
    "holdout_evidence.npz", "promotion_input.json", "promotion_expected.json",
    "negative_controls.json",
    "manifest.json", "synthetic.parquet",
)


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"promotion evidence must be a regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"promotion evidence must be an object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_sha(directory: Path, *, registered_model: tuple[str, str] | None = None) -> str:
    """Hash model payload, authenticating MLflow's optional download metadata.

    MLflow's models:/ transport adds a top-level registered_model_meta file.
    Only that exact file can be excluded, and only when its complete contents
    match the alias version already resolved through the registry client.
    """
    metadata_name = "registered_model_meta"
    if registered_model is not None:
        import yaml

        metadata_path = directory / metadata_name
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError("MLflow alias download requires regular registry metadata")
        metadata = yaml.safe_load(metadata_path.read_text())
        name, version = registered_model
        if (not isinstance(metadata, dict) or set(metadata) != {"model_name", "model_version"}
                or metadata["model_name"] != name or type(metadata["model_version"]) not in (int, str)
                or str(metadata["model_version"]) != str(version)):
            raise ValueError("MLflow download metadata differs from the resolved registry alias")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("deployment packages must not contain symbolic links")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(directory).as_posix()
            if registered_model is not None and relative == metadata_name:
                continue
            files[relative] = _sha(path)
    if "MLmodel" not in files:
        raise ValueError("deployment package requires MLmodel")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    """Publish complete JSON atomically, without replacing an existing path.

    The fsynced preparation file is retained as audit evidence. A hard link
    provides create-only atomic publication on the same filesystem.
    """
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = path.parent / f".{path.name}.{uuid.uuid4().hex}.prepared"
    with prepared.open("x") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(prepared, path)


def _require_passed_gates(directory: Path, objective: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from src.experiment.workflow_gates import evaluate_gate

    reports = {}
    expected_keys = {"labels", "probabilities", "prior_probability", "persistence_probability"}
    for partition in ("validation", "holdout"):
        source = directory / f"{partition}_evidence.npz"
        if source.is_symlink() or not source.is_file():
            raise ValueError("promotion predictions must be regular NPZ evidence")
        with np.load(source, allow_pickle=False) as evidence:
            if set(evidence.files) != expected_keys:
                raise ValueError(f"{partition} prediction evidence has unexpected arrays")
            report = evaluate_gate(
                evidence["labels"], evidence["probabilities"], evidence["prior_probability"],
                evidence["persistence_probability"], objective["gates"],
            )
        if report.get("passed") is not True:
            raise ValueError(f"promotion refused: {partition} evaluation gates failed")
        if report != _read(directory / f"{partition}_gate.json"):
            raise ValueError(f"promotion refused: {partition} recorded gate differs from recomputation")
        reports[partition] = report
    with np.load(directory / "holdout_evidence.npz", allow_pickle=False) as evidence:
        controls = {
            "prior_only": evaluate_gate(evidence["labels"], evidence["prior_probability"],
                evidence["prior_probability"], evidence["persistence_probability"], objective["gates"]),
            "shuffled_predictions": evaluate_gate(evidence["labels"],
                np.random.default_rng(objective["model_config"]["seed"] + 1).permutation(evidence["probabilities"]),
                evidence["prior_probability"], evidence["persistence_probability"], objective["gates"]),
        }
    if any(report["passed"] for report in controls.values()) or controls != _read(directory / "negative_controls.json"):
        raise ValueError("promotion requires independently rejected negative controls")
    return reports


def _validate_http(record: dict[str, Any]) -> None:
    required = ("verified", "probability_parity", "direction_parity", "timestamp_parity",
                "local_only", "server_stopped")
    if any(record.get(key) is not True for key in required):
        raise ValueError("promotion requires complete HTTP acceptance evidence")
    if record.get("invalid_feature_status") not in (400, 422):
        raise ValueError("promotion requires HTTP rejection of malformed features")


def _verify_prediction_evidence(directory: Path, bundle: Path, objective: dict[str, Any]) -> None:
    """Recompute predictions and both training-only baselines from source data."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from src.experiment.workflow_data import FEATURE_COLUMNS, chronological_partitions, make_synthetic_dataset
    from src.experiment.workflow_gates import fit_persistence_baseline, persistence_probabilities
    from src.serving.workflow_model import WorkflowPythonModel

    source = directory / "synthetic.parquet"
    source_sha = _sha(source)
    manifest = _read(directory / "manifest.json")
    datasets = manifest.get("datasets", [])
    if (manifest.get("objective_id") != OBJECTIVE_ID or len(datasets) != 2
            or any(item.get("content_sha256") != source_sha for item in datasets)):
        raise ValueError("promotion source dataset differs from canonical provenance")
    data = pd.read_parquet(source)
    config = objective["model_config"]
    regenerated = make_synthetic_dataset(n_rows=config["n_rows"], seed=config["seed"], horizon=config["horizon"])
    pd.testing.assert_frame_equal(data, regenerated, check_freq=False)
    parts = chronological_partitions(data, sequence_length=config["seq_len"])
    prior = np.bincount(parts["train"].label.to_numpy(), minlength=3).astype(float) + 1.0
    prior /= prior.sum()
    persistence = fit_persistence_baseline(parts["train"])
    model = WorkflowPythonModel()
    model.load_context(SimpleNamespace(artifacts={"bundle": str(bundle)}))
    if model.seq_len != config["seq_len"] or model.feature_cols != list(FEATURE_COLUMNS):
        raise ValueError("promotion model differs from the frozen synthetic feature/sequence contract")
    if _sha(bundle / "validation_evidence.npz") != _sha(directory / "validation_evidence.npz"):
        raise ValueError("selected validation predictions differ from the integrity-checked model evidence")
    for partition in ("validation", "holdout"):
        frame = parts[partition]
        evaluated = frame.iloc[config["seq_len"] - 1:]
        actual = model.predict(None, frame[list(FEATURE_COLUMNS)].reset_index()).iloc[config["seq_len"] - 1:]
        with np.load(directory / f"{partition}_evidence.npz", allow_pickle=False) as evidence:
            np.testing.assert_array_equal(evidence["labels"], evaluated.label.to_numpy())
            np.testing.assert_allclose(evidence["probabilities"], actual[["p_down", "p_flat", "p_up"]], atol=1e-6, rtol=0)
            np.testing.assert_allclose(evidence["prior_probability"], np.tile(prior, (len(evaluated), 1)), atol=1e-12, rtol=0)
            np.testing.assert_allclose(evidence["persistence_probability"], persistence_probabilities(evaluated, persistence), atol=1e-12, rtol=0)


def _verify_promoted_inference(
    *, client: Any, tracking_uri: str, model_name: str, version: str,
    bundle: Path, package_sha: str, directory: Path, evidence_directory: Path,
) -> dict[str, Any]:
    """Resolve the real alias, reload its artifact, and exercise HTTP inference."""
    import mlflow.artifacts
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from src.serving.workflow_http import verify_http_deployment
    from src.serving.workflow_model import WorkflowPythonModel, verify_bundle_integrity

    resolved = client.get_model_version_by_alias(model_name, ALIAS)
    if str(resolved.version) != version:
        raise RuntimeError("MLflow serving alias resolved to a different model version")
    uri = f"models:/{model_name}@{ALIAS}"
    download_root = evidence_directory / "alias_download"
    download_root.mkdir(exist_ok=False)
    artifact = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=uri, dst_path=str(download_root),
        tracking_uri=tracking_uri, registry_uri=tracking_uri,
    ))
    if _package_sha(artifact, registered_model=(model_name, version)) != package_sha:
        raise RuntimeError("MLflow alias package differs from the canonical deployment package")
    # MLflow copies the native bundle under artifacts/<original-directory-name>.
    native_bundles = [path.parent for path in artifact.rglob("bundle_integrity.json")]
    if len(native_bundles) != 1:
        raise RuntimeError("alias artifact must contain exactly one integrity-checked bundle")
    if verify_bundle_integrity(native_bundles[0]) != verify_bundle_integrity(bundle):
        raise RuntimeError("MLflow alias artifact differs from the canonical registry bundle")
    inputs = pd.read_json(io.StringIO((directory / "promotion_input.json").read_text()), orient="split")
    expected = pd.read_json(io.StringIO((directory / "promotion_expected.json").read_text()), orient="split")
    alias_model = mlflow.pyfunc.load_model(str(artifact))
    actual = alias_model.predict(inputs)
    canonical_pyfunc = mlflow.pyfunc.load_model(str(bundle.parent / "deployment"))
    canonical = canonical_pyfunc.predict(inputs)
    local_model = WorkflowPythonModel()
    local_model.load_context(SimpleNamespace(artifacts={"bundle": str(bundle)}))
    local = local_model.predict(None, inputs)
    probability_columns = ["p_down", "p_flat", "p_up", "confidence"]
    for result in (actual, local, canonical):
        np.testing.assert_allclose(result[probability_columns], expected[probability_columns], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(result.direction.to_numpy(), expected.direction.to_numpy())
        np.testing.assert_array_equal(pd.to_datetime(result.ts, utc=True).array.asi8,
                                      pd.to_datetime(expected.ts, utc=True).array.asi8)
    http = verify_http_deployment(artifact, inputs, actual, evidence_directory)
    _validate_http(http)
    return {"alias_uri": uri, "alias_version": version, "alias_reload_verified": True,
            "canonical_bundle_verified": True, "probability_parity": True, "http": http}


def promote_test_objective(run_directory: Path) -> dict[str, Any]:
    """Promote a completed test run only after independently checking its evidence.

    The namespace is fixed and LATEST/aliases are create-only: this cannot
    replace an existing production pin. Failed attempts retain their evidence;
    only an alias created by this invocation is removed on verification failure.
    """
    from mlflow.tracking import MlflowClient
    from src.common.data_paths import model_registry_dir
    from src.serving.promote import latest_pointer_path, registry_objective_dir, resolve_latest_bundle
    from src.serving.workflow_model import verify_bundle_integrity

    started = time.monotonic()
    directory = Path(run_directory).resolve()
    hashes = {}
    for name in _EVIDENCE_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe promotion evidence: {name}")
        hashes[name] = _sha(path)
    objective = _read(directory / "objective.json")
    request = _read(directory / "request.json")
    if objective.get("objective_id") != OBJECTIVE_ID:
        raise ValueError("promotion is restricted to synthetic-regime-service")
    if request.get("objective") != objective:
        raise ValueError("promotion objective differs from its pretraining request snapshot")
    if request.get("objective_sha256") != hashes["objective.json"]:
        raise ValueError("promotion objective checksum differs from its pretraining snapshot")
    if objective.get("synthetic_only") is not True or objective.get("model_config") != request.get("config"):
        raise ValueError("promotion requires the frozen synthetic model recipe")
    gates = _require_passed_gates(directory, objective)
    selection = _read(directory / "selection.json")
    summary = _read(directory / "summary.json")
    consultation = _read(directory / "holdout_consultation.json")
    selected = selection.get("selected")
    if (not selected or selection.get("holdout_consulted") is not False
            or consultation.get("selected") != selected or consultation.get("consultations") != 1
            or summary.get("selected_model_id") != selected):
        raise ValueError("promotion requires frozen selection and exactly one final holdout consultation")
    completion = _read(directory / "completion.json")
    parent_id = summary.get("mlflow_parent_run_id")
    if (completion.get("status") != "PASSED" or completion.get("tracking_status") != "FINISHED"
            or completion.get("mlflow_parent_run_id") != parent_id):
        raise ValueError("promotion requires a completed, FINISHED MLflow workflow")
    elapsed_before = float(completion["elapsed_seconds"])
    budget = int(objective["gates"]["max_runtime_seconds"])
    if not 0 <= elapsed_before < budget:
        raise ValueError("promotion runtime budget has already expired")
    deployment = _read(directory / "deployment_verification.json")
    if deployment.get("artifact_download_reload") is not True or deployment.get("probability_parity") is not True:
        raise ValueError("promotion requires verified deployment artifact reload and parity")
    _validate_http(deployment.get("http", {}))
    designs = summary.get("designs", [])
    matches = [item for item in designs if item.get("model_id") == selected]
    if len(matches) != 1:
        raise ValueError("promotion requires exactly one selected model design")
    design = matches[0]
    bundle = Path(design["directory"])
    if bundle.is_symlink() or not bundle.resolve().is_relative_to(directory):
        raise ValueError("selected bundle must belong to the completed run")
    bundle = bundle.resolve()
    bundle_hashes = verify_bundle_integrity(bundle)
    manifest = _read(bundle / "finalize_manifest.json")
    if (manifest.get("candidate_id") != selected or manifest.get("objective_id") != OBJECTIVE_ID
            or manifest.get("synthetic_only") is not True):
        raise ValueError("selected serving bundle does not match the synthetic objective and model")
    _verify_prediction_evidence(directory, bundle, objective)
    registered = _read(directory / "registry.json")
    model_name = registered.get("name", "")
    version = str(registered.get("version", ""))
    if not model_name.startswith("workflow-test-") or registered.get("run_id") != parent_id:
        raise ValueError("promotion requires this run's isolated workflow model registration")
    tracking_uri = request["tracking_uri"]
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    parent = client.get_run(parent_id)
    child = client.get_run(design["mlflow_run_id"])
    if (parent.info.status != "FINISHED" or child.info.status != "FINISHED"
            or child.data.tags.get("mlflow.parentRunId") != parent_id):
        raise ValueError("promotion requires FINISHED parent and selected child MLflow runs")
    model = client.get_registered_model(model_name)
    recorded = client.get_model_version(model_name, version)
    if model.tags.get("vtech.workflow") != "transformer_smoke" or model.aliases:
        raise ValueError("promotion refuses a foreign model or an existing deployment alias")
    expected_source = parent.info.artifact_uri.rstrip("/") + "/deployment"
    if (recorded.run_id != parent_id or recorded.status != "READY" or recorded.source != expected_source
            or registered.get("source") != recorded.source):
        raise ValueError("registered version does not match the completed deployment artifact")
    lifecycle = LifecycleStatus(recorded.tags.get("lifecycle"))
    if lifecycle not in (LifecycleStatus.EXPERIMENTAL, LifecycleStatus.CANDIDATE):
        raise ValueError("only an experimental or candidate workflow version can be promoted")
    root = model_registry_dir()
    objective_directory = registry_objective_dir(OBJECTIVE_ID, registry_root=root)
    latest_path = latest_pointer_path(OBJECTIVE_ID, registry_root=root)
    if latest_path.exists() or latest_path.is_symlink():
        raise FileExistsError("synthetic objective already has LATEST; use an isolated data root for another full test")
    destination = objective_directory / directory.name
    destination.mkdir(parents=True, exist_ok=False)
    copied_bundle = destination / "bundle"
    shutil.copytree(bundle, copied_bundle)
    if verify_bundle_integrity(copied_bundle) != bundle_hashes:
        raise RuntimeError("registry bundle copy differs from accepted artifact")
    copied_pyfunc = destination / "deployment"
    package_sha = _package_sha(directory / "deployment")
    shutil.copytree(directory / "deployment", copied_pyfunc)
    if _package_sha(copied_pyfunc) != package_sha:
        raise RuntimeError("registry pyfunc copy differs from accepted deployment artifact")
    evidence_directory = destination / "post_promotion"
    evidence_directory.mkdir()
    for name in _EVIDENCE_FILES:
        shutil.copyfile(directory / name, destination / name)
    transitions = []
    alias_created = False
    production_tag_set = False
    pointer_published = False
    try:
        if lifecycle == LifecycleStatus.EXPERIMENTAL:
            assert is_valid_transition(lifecycle, LifecycleStatus.CANDIDATE)
            client.set_model_version_tag(model_name, version, "lifecycle", LifecycleStatus.CANDIDATE.value)
            transitions.append({"from": lifecycle.value, "to": LifecycleStatus.CANDIDATE.value,
                                "reason": "frozen validation evaluation gates passed"})
            lifecycle = LifecycleStatus.CANDIDATE
        assert is_valid_transition(lifecycle, LifecycleStatus.PRODUCTION)
        # Keep the lifecycle CANDIDATE while testing the new serving alias.
        client.set_registered_model_alias(model_name, ALIAS, version)
        alias_created = True
        serving = _verify_promoted_inference(
            client=client, tracking_uri=tracking_uri, model_name=model_name, version=version,
            bundle=copied_bundle, package_sha=package_sha, directory=directory, evidence_directory=evidence_directory,
        )
        if any(_sha(directory / name) != digest for name, digest in hashes.items()):
            raise RuntimeError("promotion evidence changed during verification")
        if verify_bundle_integrity(bundle) != bundle_hashes or verify_bundle_integrity(copied_bundle) != bundle_hashes:
            raise RuntimeError("promotion model changed during verification")
        if _package_sha(copied_pyfunc) != package_sha:
            raise RuntimeError("promotion deployment package changed during verification")
        elapsed_total = elapsed_before + time.monotonic() - started
        if elapsed_total > budget:
            raise TimeoutError("promotion exceeded the frozen total workflow runtime budget")
        client.set_model_version_tag(model_name, version, "lifecycle", LifecycleStatus.PRODUCTION.value)
        production_tag_set = True
        client.set_model_version_tag(model_name, version, "objective_id", OBJECTIVE_ID)
        client.set_model_version_tag(model_name, version, "synthetic_only", "true")
        client.set_model_version_tag(model_name, version, "promotion_evidence_sha256", hashes["objective.json"])
        final_version = client.get_model_version_by_alias(model_name, ALIAS)
        if str(final_version.version) != version or final_version.tags.get("lifecycle") != LifecycleStatus.PRODUCTION.value:
            raise RuntimeError("promoted MLflow alias/lifecycle read-back failed")
        transitions.append({"from": lifecycle.value, "to": LifecycleStatus.PRODUCTION.value,
                            "reason": "frozen holdout, artifact integrity and alias HTTP acceptance passed"})
        receipt = {
            "objective_id": OBJECTIVE_ID, "version": directory.name, "candidate_id": selected,
            "family": "transformer", "path": str(copied_bundle), "run_directory": str(directory),
            "mlflow_model_path": str(copied_pyfunc),
            "lifecycle": LifecycleStatus.PRODUCTION.value, "synthetic_only": True,
            "training_origin_lifecycle": manifest.get("lifecycle"),
            "trading_activation": False, "promotion_scope": "synthetic regime inference service only",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "promoted_by": "src.serving.workflow_promotion", "tracking_uri": tracking_uri,
            "mlflow_parent_run_id": parent_id, "registry": {"name": model_name, "version": version, "alias": ALIAS},
            "transitions": transitions, "evidence_sha256": hashes, "bundle_sha256": bundle_hashes,
            "pyfunc_package_sha256": package_sha,
            "gates": gates, "serving": serving,
            "total_elapsed_seconds": elapsed_total, "runtime_budget_seconds": budget,
        }
        _write_once(destination / "PROMOTION.json", receipt)
        _write_once(directory / "promotion.json", receipt)
        # Final commit: canonical consumers have no pointer before all checks pass.
        _write_once(latest_path, receipt)
        pointer_published = True
        if resolve_latest_bundle(OBJECTIVE_ID, registry_root=root) != copied_bundle.resolve():
            raise RuntimeError("canonical registry LATEST did not resolve the promoted bundle")
        return receipt
    except BaseException as exc:
        # Only remove this invocation's new alias, never somebody else's version.
        if alias_created and not pointer_published:
            try:
                current = client.get_model_version_by_alias(model_name, ALIAS)
                if str(current.version) == version:
                    client.delete_registered_model_alias(model_name, ALIAS)
                rollback_status = LifecycleStatus.DEPRECATED if production_tag_set else LifecycleStatus.CANDIDATE
                client.set_model_version_tag(model_name, version, "lifecycle", rollback_status.value)
            except Exception as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"new alias cleanup failed: {cleanup_error}")
        failure = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                   "canonical_pointer_published": pointer_published,
                   "canonical_pointer_exists": latest_path.exists(), "evidence_sha256": hashes}
        _write_once(destination / "PROMOTION_FAILURE.json", failure)
        raise
