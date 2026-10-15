"""No-network checks for workflow serving contracts and artifact integrity."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.experiment.workflow_data import apply_temperature
from src.serving import workflow_model as module

pytestmark = pytest.mark.ml_workflow


def _bundle(tmp_path, *, calibration: dict[str, Any] | None = None):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "model.keras").write_bytes(b"test-artifact")
    (root / "feature_schema.json").write_text(json.dumps({"feature_cols": ["x"]}))
    (root / "finalize_manifest.json").write_text(json.dumps({
        "family": "transformer", "seq_len": 3,
    }))
    (root / "calibration.json").write_text(json.dumps(calibration or {
        "temperature": 2.0, "confidence_threshold": 0.5,
    }))
    module.write_bundle_integrity(root)
    return root


def _loaded(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    monkeypatch.setattr(module, "load_serving_bundle", lambda *args, **kwargs: {
        "feature_cols": ["x"], "seq_len": 3, "kind": "sequence", "family": "transformer",
    })

    def predict(frame, bundle):
        n = len(frame)
        result = pd.DataFrame({
            "ts": frame.index, "direction": "up", "confidence": 0.6,
            "p_down": 0.15, "p_flat": 0.25, "p_up": 0.6,
        })
        result.loc[:1, ["direction", "confidence", "p_down", "p_flat", "p_up"]] = [
            "flat", 0.0, 0.0, 1.0, 0.0,
        ]
        assert len(result) == n
        return result

    monkeypatch.setattr(module, "predict_canonical", predict)
    model = module.WorkflowPythonModel()
    model.load_context(SimpleNamespace(artifacts={"bundle": str(root)}))
    frame = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"), "x": 1.0})
    return model, frame, root


def test_calibration_preserves_warmup_and_timestamps(tmp_path, monkeypatch) -> None:
    model, frame, _ = _loaded(tmp_path, monkeypatch)
    result = model.predict(None, frame)
    np.testing.assert_array_equal(result.loc[:1, "confidence"], 0.0)
    np.testing.assert_array_equal(result.loc[:1, "p_flat"], 1.0)
    expected = apply_temperature(np.array([[0.15, 0.25, 0.6]]), 2.0)[0]
    np.testing.assert_allclose(result.loc[2:, ["p_down", "p_flat", "p_up"]], np.tile(expected, (3, 1)))
    assert result["direction"].tolist() == ["flat"] * len(frame)
    assert pd.DatetimeIndex(result["ts"]).equals(pd.DatetimeIndex(frame["ts"]))
    iso = frame.copy()
    iso["ts"] = iso["ts"].map(lambda value: value.isoformat())
    pd.testing.assert_frame_equal(model.predict(None, iso), result)


@pytest.mark.parametrize("problem", [
    "missing", "extra", "order", "duplicate_column", "duplicate_timestamp", "unordered",
    "naive", "non_utc", "short", "nan", "infinite", "nonnumeric", "nat",
])
def test_invalid_input_is_rejected_before_inference(tmp_path, monkeypatch, problem) -> None:
    model, frame, _ = _loaded(tmp_path, monkeypatch)
    if problem == "missing":
        frame = frame.drop(columns="x")
    elif problem == "extra":
        frame["other"] = 1.0
    elif problem == "order":
        frame = frame[["x", "ts"]]
    elif problem == "duplicate_column":
        frame = pd.concat([frame, frame[["x"]]], axis=1)
    elif problem == "duplicate_timestamp":
        frame.loc[1, "ts"] = frame.loc[0, "ts"]
    elif problem == "unordered":
        frame = frame.iloc[::-1]
    elif problem == "naive":
        frame["ts"] = frame["ts"].dt.tz_localize(None)
    elif problem == "non_utc":
        frame["ts"] = frame["ts"].dt.tz_convert("America/Chicago")
    elif problem == "short":
        frame = frame.iloc[:2]
    elif problem == "nan":
        frame.loc[1, "x"] = np.nan
    elif problem == "infinite":
        frame.loc[1, "x"] = np.inf
    elif problem == "nonnumeric":
        frame["x"] = "invalid"
    elif problem == "nat":
        frame.loc[1, "ts"] = pd.NaT
    monkeypatch.setattr(module, "predict_canonical", lambda *_: pytest.fail("inference reached"))
    with pytest.raises(ValueError):
        model.predict(None, frame)


def test_corruption_is_detected_before_deserialization(tmp_path, monkeypatch) -> None:
    root = _bundle(tmp_path)
    (root / "model.keras").write_bytes(b"damaged-artifact")
    monkeypatch.setattr(module, "load_serving_bundle", lambda *_args, **_kwargs: pytest.fail("loaded corrupt data"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.WorkflowPythonModel().load_context(SimpleNamespace(artifacts={"bundle": str(root)}))
    with pytest.raises(ValueError, match="different integrity"):
        module.write_bundle_integrity(root)


def test_unrecorded_file_and_symlink_are_rejected(tmp_path) -> None:
    root = _bundle(tmp_path)
    (root / "unexpected.pkl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="file set"):
        module.verify_bundle_integrity(root)
    (root / "redirect").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        module.verify_bundle_integrity(root)


@pytest.mark.parametrize("calibration", [
    {"temperature": 0, "confidence_threshold": 0.5},
    {"temperature": float("nan"), "confidence_threshold": 0.5},
    {"temperature": 1, "confidence_threshold": 1.1},
])
def test_invalid_calibration_is_rejected(tmp_path, calibration) -> None:
    root = _bundle(tmp_path, calibration=calibration)
    with pytest.raises(ValueError, match="calibration"):
        module.WorkflowPythonModel().load_context(SimpleNamespace(artifacts={"bundle": str(root)}))


def test_export_pins_runtime_and_uses_utc_string_signature(tmp_path, monkeypatch) -> None:
    _, frame, root = _loaded(tmp_path, monkeypatch)
    recorded = {}
    monkeypatch.setattr(module.mlflow.pyfunc, "save_model", lambda **kwargs: recorded.update(kwargs))
    destination = tmp_path / "mlflow_model"
    assert module.save_workflow_pyfunc(root, destination, frame) == destination
    assert recorded["artifacts"] == {"bundle": str(root)}
    assert recorded["signature"].inputs.inputs[0].type.name == "string"
    assert recorded["input_example"]["ts"].str.endswith("+00:00").all()
    assert any(value.startswith(("tensorflow==", "tensorflow-cpu==")) for value in recorded["pip_requirements"])
    assert any(value.startswith("mlflow==") for value in recorded["pip_requirements"])
    assert recorded["code_paths"][0].endswith("/src")


@pytest.mark.parametrize("installed, expected", [
    ({"tensorflow-cpu"}, "tensorflow-cpu"),
    ({"tensorflow"}, "tensorflow"),
])
def test_runtime_pins_the_installed_tensorflow_distribution(monkeypatch, installed, expected) -> None:
    def version(package):
        if package not in installed:
            raise module.importlib.metadata.PackageNotFoundError(package)
        return "2.21.0"

    monkeypatch.setattr(module.importlib.metadata, "version", version)
    assert module.tensorflow_distribution() == expected
