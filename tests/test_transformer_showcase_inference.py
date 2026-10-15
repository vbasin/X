"""Optional CPU acceptance: public weights reproduce the retained holdout.

This reads only the reviewed synthetic example, and writes to pytest's sandbox.
It is skipped in the lightweight viewer environment without TensorFlow.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.ml_workflow


def test_public_package_reproduces_original_holdout(tmp_path, monkeypatch) -> None:
    tf = pytest.importorskip("tensorflow")
    import mlflow.pyfunc

    from src.common.vm_config import get_vm_config
    from src.experiment.workflow_data import FEATURE_COLUMNS, chronological_partitions
    from src.serving.workflow_model import save_workflow_pyfunc, write_bundle_integrity

    workers = get_vm_config().workers_for(mem_per_worker_gb=2.0)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(workers)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, workers // 2))
    except RuntimeError:
        pass  # Another test may already have initialized TensorFlow.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.sqlite'}")
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    fixture = Path(__file__).resolve().parents[1] / "examples/transformer-regime"
    native = fixture / "models/d7facd489218"
    expected_weight_sha = "daed32f3d9c0c76390067245ce55e88b0bf44f786a02a03cb1d15b17f03983e3"
    assert hashlib.sha256((native / "model.keras").read_bytes()).hexdigest() == expected_weight_sha
    bundle = tmp_path / "reviewed_native"
    shutil.copytree(native, bundle)
    write_bundle_integrity(bundle)

    data = pd.read_parquet(fixture / "synthetic.parquet")
    holdout = chronological_partitions(data, sequence_length=12)["holdout"]
    inputs = holdout[list(FEATURE_COLUMNS)].reset_index()
    inputs["ts"] = inputs["ts"].astype(str)
    package = save_workflow_pyfunc(bundle, tmp_path / "public_package", inputs.iloc[:20])
    restored = mlflow.pyfunc.load_model(str(package))
    actual = restored.predict(inputs).iloc[11:]
    with np.load(fixture / "evidence/holdout_evidence.npz") as recorded:
        expected = recorded["probabilities"]
        np.testing.assert_array_equal(holdout.iloc[11:].label, recorded["labels"])
    assert len(actual) == len(expected) == 289
    predicted = actual[["p_down", "p_flat", "p_up"]].to_numpy()
    np.testing.assert_allclose(predicted, expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(
        pd.to_datetime(actual.ts, utc=True).array.asi8,
        holdout.iloc[11:].index.asi8,
    )
    packaged_weight = package / "artifacts/reviewed_native/model.keras"
    assert hashlib.sha256(packaged_weight.read_bytes()).hexdigest() == expected_weight_sha
    receipt = {
        "weight_sha256": expected_weight_sha,
        "rows": len(actual),
        "max_abs_probability_error": float(np.max(np.abs(predicted - expected))),
        "tolerance": 1e-6,
        "package_source": "reviewed X source tree",
        "original_operator_code_copied": False,
    }
    (tmp_path / "inference_parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
