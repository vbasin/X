"""Integrity-checked MLflow adapter for the bounded Transformer workflow.

The model artifact owns its feature schema, calibration and decision threshold.
Inference delegates to the existing serving implementation after validation.
This module does not create runs or change MLflow's global tracking settings.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.serving.bundle import load_serving_bundle
from src.serving.predict import direction_from_proba, predict_canonical

_INTEGRITY = "bundle_integrity.json"
_OUTPUT_COLUMNS = ["ts", "direction", "confidence", "p_down", "p_flat", "p_up"]
_PROBABILITY_COLUMNS = ["p_down", "p_flat", "p_up"]


def _bundle_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"serving bundle directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"serving bundle must not contain symlinks: {path}")
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative != _INTEGRITY:
            files[relative] = path
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle_integrity(directory: str | Path) -> dict[str, str]:
    """Write immutable per-file hashes after all bundle artifacts are complete."""
    root = Path(directory)
    hashes = {name: _sha256(path) for name, path in _bundle_files(root).items()}
    if not hashes:
        raise ValueError("cannot fingerprint an empty serving bundle")
    target = root / _INTEGRITY
    if target.exists():
        if json.loads(target.read_text()) != hashes:
            raise ValueError("serving bundle already has a different integrity record")
    else:
        with target.open("x") as stream:
            json.dump(hashes, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return hashes


def verify_bundle_integrity(directory: str | Path) -> dict[str, str]:
    """Reject missing, modified or extra artifacts before model deserialization."""
    root = Path(directory)
    files = _bundle_files(root)
    integrity = root / _INTEGRITY
    if not integrity.is_file():
        raise ValueError("serving bundle requires bundle_integrity.json")
    hashes = json.loads(integrity.read_text())
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("serving bundle integrity record must be a nonempty filename/hash mapping")
    for name, digest in hashes.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or str(relative) != name
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"invalid serving bundle integrity entry: {name}")
    if set(hashes) != set(files):
        raise ValueError("serving bundle integrity file set differs from recorded artifacts")
    for name, digest in hashes.items():
        if _sha256(files[name]) != digest:
            raise ValueError(f"serving bundle checksum mismatch: {name}")
    return hashes


class WorkflowPythonModel(mlflow.pyfunc.PythonModel):
    """MLflow input contract around existing sequence-serving and calibration."""

    def load_context(self, context: Any) -> None:
        directory = Path(context.artifacts["bundle"])
        verify_bundle_integrity(directory)
        calibration = json.loads((directory / "calibration.json").read_text())
        temperature = float(calibration["temperature"])
        threshold = float(calibration["confidence_threshold"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("calibration temperature must be finite and positive")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("calibration confidence_threshold must be finite and in [0, 1]")
        schema = json.loads((directory / "feature_schema.json").read_text())
        columns = list(schema.get("feature_cols") or schema.get("columns") or [])
        if (not columns or len(columns) != len(set(columns)) or "ts" in columns
                or any(not isinstance(column, str) for column in columns)):
            raise ValueError("serving bundle requires unique named feature columns")
        manifest = json.loads((directory / "finalize_manifest.json").read_text())
        sequence_length = manifest.get("seq_len") or (manifest.get("spec_params") or {}).get("seq_len")
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("serving bundle requires a positive integer seq_len")
        if str(manifest.get("family")) != "transformer":
            raise ValueError("workflow serving requires a transformer bundle")
        # The artifact passed by MLflow lives outside the platform's research
        # roots. Its complete integrity record above is checked before loading.
        self.bundle = load_serving_bundle(directory, validate=False)
        self.feature_cols = columns
        self.seq_len = sequence_length
        self.temperature = temperature
        self.confidence_threshold = threshold

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return one prediction per ordered input row, preserving warmup rows."""
        if params:
            raise ValueError("workflow model parameters are fixed by its artifact")
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("workflow input must be a pandas DataFrame")
        expected_columns = ["ts", *self.feature_cols]
        if list(model_input.columns) != expected_columns:
            raise ValueError(f"workflow input columns must be exactly {expected_columns} in this order")
        if len(model_input) < self.seq_len:
            raise ValueError(f"workflow input requires at least {self.seq_len} rows")
        timestamps = pd.to_datetime(model_input["ts"], errors="raise")
        if (not isinstance(timestamps.dtype, pd.DatetimeTZDtype)
                or str(timestamps.dt.tz) != "UTC"):
            raise ValueError("workflow input timestamps must be timezone-aware UTC")
        if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("workflow input timestamps must be unique, increasing and non-null")
        try:
            values = model_input[self.feature_cols].to_numpy(dtype="float64")
        except (ValueError, TypeError) as exc:
            raise ValueError("workflow input features must be finite numeric values") from exc
        if not np.isfinite(values).all():
            raise ValueError("workflow input features must be finite numeric values")
        features = pd.DataFrame(values, columns=self.feature_cols, index=pd.DatetimeIndex(timestamps))
        output = predict_canonical(features, self.bundle)
        if len(output) != len(model_input):
            raise ValueError("serving predictions are not aligned to input rows")
        output = output.loc[:, _OUTPUT_COLUMNS].copy()
        if not pd.DatetimeIndex(pd.to_datetime(output["ts"], utc=True)).equals(features.index):
            raise ValueError("serving prediction timestamps are not aligned to input timestamps")
        from src.experiment.workflow_data import apply_temperature

        start = self.seq_len - 1
        calibrated = apply_temperature(
            output.iloc[start:][_PROBABILITY_COLUMNS].to_numpy(dtype="float64"),
            self.temperature,
        )
        direction, confidence = direction_from_proba(
            calibrated[:, 0], calibrated[:, 1], calibrated[:, 2], self.confidence_threshold,
        )
        for i, column in enumerate(_PROBABILITY_COLUMNS):
            output.iloc[start:, output.columns.get_loc(column)] = calibrated[:, i]
        output.iloc[start:, output.columns.get_loc("direction")] = direction
        output.iloc[start:, output.columns.get_loc("confidence")] = confidence
        return output


def tensorflow_distribution() -> str:
    """Record the installed TensorFlow wheel, including the smaller CPU profile."""
    for package in ("tensorflow-cpu", "tensorflow"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        return package
    raise importlib.metadata.PackageNotFoundError("tensorflow-cpu or tensorflow")


def save_workflow_pyfunc(
    bundle_dir: str | Path, output_dir: str | Path, input_example: pd.DataFrame,
) -> Path:
    """Export a portable pyfunc with signature, example, code and pinned runtime."""
    from mlflow.models import infer_signature

    root = Path(bundle_dir).resolve()
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"MLflow model export path already exists: {output}")
    model = WorkflowPythonModel()
    model.load_context(SimpleNamespace(artifacts={"bundle": str(root)}))
    # MLflow's datetime signature coerces to timezone-naive numpy datetimes.
    # An ISO UTC string retains the offset through both pyfunc and HTTP JSON.
    example = input_example.copy()
    model.predict(None, example)  # Validate the caller's original schema first.
    example["ts"] = pd.to_datetime(example["ts"]).map(lambda value: value.isoformat())
    prediction = model.predict(None, example)
    packages = ["mlflow", tensorflow_distribution(), "keras", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "PyYAML",
                "fastapi", "starlette", "uvicorn"]
    requirements = [f"{package}=={importlib.metadata.version(package)}" for package in packages]
    mlflow.pyfunc.save_model(
        path=str(output),
        python_model=WorkflowPythonModel(),
        artifacts={"bundle": str(root)},
        code_paths=[str(Path(__file__).resolve().parents[1])],
        input_example=example,
        signature=infer_signature(example, prediction),
        pip_requirements=requirements,
    )
    return output
