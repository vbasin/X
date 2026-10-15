"""Verified, portable recorded evidence for the Transformer showcase.

Viewing imports neither TensorFlow nor the operational database/WebUI runtime.
MLflow is imported lazily only when preparing an isolated local tracking store.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a portable manifest path, rejecting escapes and symbolic links."""
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts or "\\" in relative
            or path.as_posix() != relative):
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes example: {relative!r}")
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Symbolic links are not supported: {relative!r}")
        current = current.parent
    return target


def example_root(path: Path) -> Path:
    """Resolve either a fixture directory or a prepared output's relative pointer."""
    root = Path(path).resolve()
    if (root / "showcase.json").is_file():
        return _safe_path(root, _read(root / "showcase.json")["example_root"])
    return root


def verify_example(path: Path) -> dict[str, Any]:
    """Check every allowlisted file before reading it; return the manifest."""
    root = example_root(path)
    if (root / "showcase_manifest.json").is_symlink():
        raise ValueError("Symbolic manifest is not supported")
    manifest = _read(root / "showcase_manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("synthetic_only") is not True:
        raise ValueError("Unsupported or non-synthetic showcase manifest")
    seen = set()
    for entry in manifest["artifacts"]:
        relative = entry["path"]
        if relative in seen:
            raise ValueError(f"Duplicate artifact: {relative}")
        seen.add(relative)
        target = _safe_path(root, relative)
        if not target.is_file():
            raise ValueError(f"Missing artifact: {relative}")
        if target.stat().st_size != entry["size"]:
            raise ValueError(f"Artifact size mismatch: {relative}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    all_paths = list(root.rglob("*"))
    if any(file.is_symlink() for file in all_paths):
        raise ValueError("Symbolic links are not supported anywhere in a fixture")
    actual = {file.relative_to(root).as_posix() for file in all_paths
              if file.is_file() and file != root / "showcase_manifest.json"}
    if actual != seen:
        raise ValueError("Fixture contains files outside its verified allowlist")
    required = {manifest["dataset"], "mlflow_export.json", "predictions/holdout.parquet"}
    required.update(f"evidence/{name}.json" for name in (
        "summary", "promotion_completion", "holdout_gate", "validation_gate", "splits",
        "negative_controls", "comparison", "selection", "objective"))
    for model in manifest["models"]:
        _safe_path(root, model["path"])
        required.update(f"{model['path']}/{name}" for name in (
            "model.json", "model.keras", "scaler.pkl", "validation_gate.json", "calibration.json",
            "oof_predictions.parquet", "trade_ledger.parquet", "economic_metrics.json",
            "cost_sensitivity.json"))
    if not required.issubset(seen):
        raise ValueError(f"Required artifacts are not verified: {sorted(required - seen)}")
    if hashlib.sha256(_safe_path(root, manifest["dataset"]).read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset provenance mismatch")
    return manifest


def _records(frame: Any) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def load_example(path: Path) -> dict[str, Any]:
    """Return presentation data from verified original/derived files, without MLflow."""
    import pandas as pd

    manifest = verify_example(path)
    root = example_root(path)
    frame = pd.read_parquet(root / manifest["dataset"]).reset_index()
    if len(frame) != manifest["n_rows"] or "ts" not in frame.columns:
        raise ValueError("Dataset row count or timestamp contract differs from manifest")
    def evidence(name: str) -> Any:
        return _read(root / "evidence" / f"{name}.json")
    source_summary = evidence("summary")
    promotion = evidence("promotion_completion")
    holdout_gate = evidence("holdout_gate")
    model_records = []
    validation: dict[str, Any] = {}
    backtest: dict[str, Any] = {}
    for model in manifest["models"]:
        mid = model["design_id"]
        directory = _safe_path(root, model["path"])
        specification = _read(directory / "model.json")
        gate = _read(directory / "validation_gate.json")
        selected = mid == manifest["selected_model_id"]
        model_records.append({"design_id": mid, "depth": model["depth"], "selected": selected,
                              "parameters": specification["hyperparameters"],
                              "validation": gate["metrics"], "validation_gate": gate,
                              "holdout": holdout_gate["metrics"] if selected else None,
                              "calibration": _read(directory / "calibration.json"),
                              "source_mlflow_run_id": model["source_mlflow_run_id"]})
        validation[mid] = _records(pd.read_parquet(directory / "oof_predictions.parquet"))
        trades = pd.read_parquet(directory / "trade_ledger.parquet")
        trades["cumulative_net_dollars"] = trades["net_dollars"].cumsum()
        backtest[mid] = {"trades": _records(trades), "population": "validation",
                         "economics": _read(directory / "economic_metrics.json"),
                         "cost_sensitivity": _read(directory / "cost_sensitivity.json"),
                         "scope": "synthetic integration replay; not a financial promotion gate"}
    summary = {"name": manifest["name"], "source_run_id": manifest["source_run_id"],
               "objective_id": manifest["objective_id"], "n_rows": len(frame),
               "feature_count": len(manifest["features"]), "synthetic_only": True,
               "evidence_kind": "recorded_example", "selected_model_id": manifest["selected_model_id"],
               "status": promotion["status"], "lifecycle": promotion["lifecycle"],
               "promotion_scope": promotion["scope"], "elapsed_seconds": promotion["elapsed_seconds"],
               "dataset_sha256": manifest["dataset_sha256"], "seed": manifest["seed"],
               "source_mlflow_parent_run_id": source_summary["mlflow_parent_run_id"],
               "holdout_rows": holdout_gate["n_rows"], "holdout_metrics": holdout_gate["metrics"],
               "validation_rows": evidence("validation_gate")["n_rows"],
               "start": frame["ts"].iloc[0].isoformat(), "end": frame["ts"].iloc[-1].isoformat(),
               "promotion_label": "Original run promoted; imported recorded example",
               "trading_activation": False}
    payload = {"schema_version": 1, "summary": summary,
               "data": {"columns": list(frame.columns), "features": manifest["features"],
                        "outcomes": manifest["outcomes"], "rows": _records(frame),
                        "splits": evidence("splits"), "label_names": {"0": "down", "1": "flat", "2": "up"}},
               "models": model_records,
               "predictions": {"validation": validation,
                               "holdout": _records(pd.read_parquet(root / "predictions/holdout.parquet"))},
               "backtest": backtest,
               "evaluation": {"gates": {"validation": evidence("validation_gate"), "holdout": holdout_gate},
                              "baselines": holdout_gate["baselines"],
                              "negative_controls": evidence("negative_controls"), "promotion": promotion,
                              "comparison": evidence("comparison"), "selection": evidence("selection"),
                              "objective": evidence("objective")},
               "artifacts": manifest["artifacts"], "provenance": manifest["provenance_note"]}
    if (Path(path) / "showcase.json").is_file():
        state = _read(Path(path) / "showcase.json")
        payload["mlflow"] = state["mlflow"]
    return payload


@contextmanager
def _data_root(root: Path) -> Iterator[None]:
    """Use the canonical path resolver during one synchronous preparation call."""
    previous = os.environ.get("VTECH_DATA_DIR")
    os.environ["VTECH_DATA_DIR"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VTECH_DATA_DIR", None)
        else:
            os.environ["VTECH_DATA_DIR"] = previous


def _import_mlflow(fixture: Path, output: Path) -> dict[str, Any]:
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    from src.common.data_paths import layout

    manifest = verify_example(fixture)
    exported = _read(fixture / "mlflow_export.json")
    generation = uuid.uuid4().hex[:12]
    with _data_root(output):
        tracking_directory = layout("mlflow_artifacts") / generation
    tracking_directory.mkdir(parents=True)
    database = tracking_directory / "tracking.sqlite"
    artifact_root = tracking_directory / "artifacts"
    client = MlflowClient(tracking_uri="sqlite:///" + str(database))
    experiment = client.create_experiment("Robin recorded Transformer example", artifact_location=artifact_root.as_uri())
    mapping: dict[str, str] = {}
    imported_at = datetime.now(timezone.utc).isoformat()
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    for run in exported["runs"]:
        source_id = run["source_run_id"]
        source_parent = run["tags"].get("mlflow.parentRunId")
        tags = {key: str(value) for key, value in run["tags"].items()
                if not key.startswith("mlflow.") and key not in {"lifecycle", "workflow.promotion_status"}}
        tags.update({"evidence_kind": "recorded_example", "source_run_id": source_id,
                     "source_start_time": str(run["source_start_time"]),
                     "source_end_time": str(run["source_end_time"]),
                     "source_status": run["source_status"], "imported_at": imported_at,
                     "source_model_version": str(exported["registry"]["version"]),
                     "source_dataset_sha256": manifest["dataset_sha256"],
                     "promotion_status": "recorded_original_promotion_not_reexecuted",
                     "mlflow.runName": "Recorded: " + run["tags"].get("mlflow.runName", source_id)})
        if source_parent:
            if source_parent not in mapping:
                raise ValueError("MLflow export must list parent before its children")
            tags["mlflow.parentRunId"] = mapping[source_parent]
            tags["source_parent_run_id"] = source_parent
        created = client.create_run(experiment, tags=tags, start_time=now)
        rid = created.info.run_id
        mapping[source_id] = rid
        metrics = [Metric(key, point["value"], point["timestamp"], point["step"])
                   for key, points in run["metrics"].items() for point in points]
        client.log_batch(rid, metrics=metrics, params=[Param(key, str(value)) for key, value in run["params"].items()])
        # Only the curated fixture is imported; source store and private code never are.
        model_id = run["tags"].get("model_id")
        if model_id:
            client.log_artifacts(rid, str(_safe_path(fixture, f"models/{model_id}")), artifact_path="design")
        else:
            client.log_artifacts(rid, str(fixture / "evidence"), artifact_path="evidence")
            client.log_artifacts(rid, str(fixture / "predictions"), artifact_path="predictions")
            for name in ("showcase_manifest.json", "mlflow_export.json", "synthetic.parquet"):
                client.log_artifact(rid, str(fixture / name), artifact_path="recorded_example")
        client.set_terminated(rid, status="FINISHED")
    return {"tracking_db": database.relative_to(output).as_posix(),
            "artifact_root": artifact_root.relative_to(output).as_posix(),
            "experiment_id": str(experiment), "run_id_map": mapping,
            "parent_run_id": mapping[exported["parent_run_id"]],
            "source_parent_run_id": exported["parent_run_id"],
            "source_registry": exported["registry"], "imported_at": imported_at,
            "evidence_kind": "recorded_example", "registered_locally": False}


def prepare_example(example_dir: Path, output: Path) -> dict[str, Any]:
    """Copy verified evidence to a new isolated canonical run root and import MLflow."""
    from src.common.data_paths import layout
    from src.common.run_layout import run_dir

    source = example_root(example_dir)
    manifest = verify_example(source)
    output = Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the immutable source fixture")
    if output.exists():
        raise FileExistsError(f"Preparation requires a new output directory: {output}")
    output.mkdir(parents=True)
    with _data_root(output):
        run = run_dir(manifest["objective_id"], manifest["source_run_id"] + "-recorded")
        registry = layout("model_registry") / manifest["objective_id"] / "recorded-example"
    run.mkdir(parents=True)
    for entry in manifest["artifacts"]:
        target = _safe_path(run, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_safe_path(source, entry["path"]), target)
    shutil.copyfile(source / "showcase_manifest.json", run / "showcase_manifest.json")
    verify_example(run)
    registry.mkdir(parents=True)
    _write(registry / "recorded_promotion.json", {
        "evidence_kind": "recorded_example", "source_run_id": manifest["source_run_id"],
        "source_promotion": _read(run / "evidence/promotion_completion.json"),
        "local_promotion_executed": False, "trading_activation": False})
    state = {"schema_version": 1, "example_root": run.relative_to(output).as_posix(),
             "evidence_kind": "recorded_example", "source_run_id": manifest["source_run_id"],
             "mlflow": _import_mlflow(run, output), "prepared_location": str(output)}
    _write(output / "showcase.json", state)
    return state


def tracking_uri(output: Path) -> str:
    """Resolve the local SQLite URI from a prepared output, independent of cwd."""
    output = Path(output).resolve()
    state = _read(output / "showcase.json")
    return "sqlite:///" + str(_safe_path(output, state["mlflow"]["tracking_db"]))


def relocate_prepared(output: Path) -> dict[str, Any]:
    """After relocation, reimport via MLflow API into a fresh portable generation.

    Old imported IDs/stores remain preserved. No direct backend SQL rewriting is
    needed, and the new mapping continues to carry original training IDs.
    """
    output = Path(output).resolve()
    state = _read(output / "showcase.json")
    verify_example(output)
    if state["prepared_location"] != str(output):
        _write(output / f"showcase.previous-{uuid.uuid4().hex[:12]}.json", state)
        state["mlflow"] = _import_mlflow(example_root(output), output)
        state["prepared_location"] = str(output)
        _write(output / "showcase.json", state)
    return state
