"""Export only reviewed synthetic Transformer files, never deployment code or stores.

Run from the checkout with ``python -m scripts.showcase.export_transformer``.
The destination must not exist. Original evidence is never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _projection(value: Any, removed: list[str], prefix: str = "") -> Any:
    """Remove deployment coordinates; keep metrics, policy, IDs and source hashes."""
    excluded = {"directory", "run_directory", "tracking_uri", "mlflow_model_path",
                "path", "endpoint", "log_path", "server_pid"}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            location = f"{prefix}.{key}".lstrip(".")
            if key in excluded:
                removed.append(location)
            else:
                result[key] = _projection(item, removed, location)
        return result
    if isinstance(value, list):
        return [_projection(item, removed, f"{prefix}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, str) and (value.startswith(("/opt/", "file:/", "sqlite:/"))):
        removed.append(prefix)
        return "[deployment location omitted; see source hash]"
    return value


def export_example(source_run: Path, destination: Path) -> dict[str, Any]:
    """Create a new portable fixture from a completed synthetic workflow run."""
    from mlflow.tracking import MlflowClient

    source = source_run.resolve()
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Export destination must be outside the immutable source run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export requires a new destination: {destination}")
    objective = _json(source / "objective.json")
    summary = _json(source / "summary.json")
    completion = _json(source / "promotion_completion.json")
    if not objective.get("synthetic_only") or completion.get("status") != "PASSED":
        raise ValueError("Only a completed synthetic promotion can be exported")
    expected = completion["promotion"]["evidence_sha256"]["synthetic.parquet"]
    if _hash(source / "synthetic.parquet") != expected:
        raise ValueError("Dataset differs from original promotion evidence")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    def copy(relative: str, target: str | None = None) -> None:
        src, dst = source / relative, destination / (target or relative)
        winner_prefix = f"pass_1/model_{summary['selected_model_id']}/"
        if relative.startswith(winner_prefix):
            original_hashes = completion["promotion"]["bundle_sha256"]
            expected_hash = original_hashes.get(relative.removeprefix(winner_prefix))
            if expected_hash is not None and _hash(src) != expected_hash:
                raise ValueError(f"Model artifact differs from original promotion: {relative}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        artifacts.append({"path": dst.relative_to(destination).as_posix(), "size": dst.stat().st_size,
                          "sha256": _hash(dst), "source_path": relative,
                          "source_sha256": _hash(src), "kind": "original"})

    def project(relative: str, target: str) -> None:
        removed: list[str] = []
        value = _projection(_json(source / relative), removed)
        dst = destination / target
        _write(dst, value)
        artifacts.append({"path": target, "size": dst.stat().st_size, "sha256": _hash(dst),
                          "source_path": relative, "source_sha256": _hash(source / relative),
                          "kind": "projection", "omitted_fields": removed})

    copy("synthetic.parquet")
    evidence = ("objective", "summary", "comparison", "selection", "splits", "holdout",
                "validation_gate", "holdout_gate", "negative_controls", "persistence_baseline",
                "promotion_completion", "deployment_verification", "holdout_consultation")
    for name in evidence:
        project(f"{name}.json", f"evidence/{name}.json")
    copy("holdout_evidence.npz", "evidence/holdout_evidence.npz")
    copy("validation_evidence.npz", "evidence/validation_evidence.npz")
    models = []
    native = ("model.keras", "scaler.pkl", "model.json", "feature_schema.json", "calibration.json",
              "finalize_manifest.json", "metrics.json", "baseline.json", "fold_metrics.json",
              "selected_population.json", "economic_metrics.json", "cost_sensitivity.json",
              "validation_gate.json", "validation_evidence.npz", "oof_predictions.parquet",
              "trade_ledger.parquet")
    for design in summary["designs"]:
        mid = design["model_id"]
        if not mid.isalnum():
            raise ValueError("Unsafe model ID")
        for name in native:
            copy(f"pass_1/model_{mid}/{name}", f"models/{mid}/{name}")
        project(f"pass_1/model_{mid}/manifest.json", f"models/{mid}/manifest.json")
        models.append({"design_id": mid, "depth": design["depth"],
                       "source_mlflow_run_id": design["mlflow_run_id"], "path": f"models/{mid}"})
    frame = pd.read_parquet(source / "synthetic.parquet")
    splits = _json(source / "splits.json")
    holdout = frame.loc[splits["holdout"]["start"]:splits["holdout"]["end"]]
    warmup = int(objective["model_config"]["seq_len"]) - 1
    holdout = holdout.iloc[warmup:]
    with np.load(source / "holdout_evidence.npz", allow_pickle=False) as values:
        if not np.array_equal(holdout["label"].to_numpy(), values["labels"]):
            raise ValueError("Holdout timestamp reconstruction does not match original labels")
        probabilities = values["probabilities"]
        labels = np.array(["down", "flat", "up"])
        view = pd.DataFrame({"ts": holdout.index, "direction": labels[probabilities.argmax(axis=1)],
                             "direction_true": labels[values["labels"].astype(int)],
                             "confidence": probabilities.max(axis=1),
                             "p_down": probabilities[:, 0], "p_flat": probabilities[:, 1],
                             "p_up": probabilities[:, 2]})
    derived = destination / "predictions/holdout.parquet"
    derived.parent.mkdir(parents=True)
    view.to_parquet(derived, index=False)
    artifacts.append({"path": "predictions/holdout.parquet", "size": derived.stat().st_size,
                      "sha256": _hash(derived), "kind": "derived",
                      "source_sha256": _hash(source / "holdout_evidence.npz"),
                      "source_path": "holdout_evidence.npz",
                      "derivation": f"Original probabilities joined to original holdout partition after {warmup}-row sequence warmup; labels asserted equal.",
                      "additional_sources": {"synthetic.parquet": expected,
                                             "splits.json": _hash(source / "splits.json")}})
    # API export deliberately excludes absolute storage locations and full code bundles.
    client = MlflowClient(tracking_uri="sqlite:///" + str(source / "tracking.sqlite"))
    parent_id = summary["mlflow_parent_run_id"]
    runs = []
    for rid in [parent_id] + [design["mlflow_run_id"] for design in summary["designs"]]:
        run = client.get_run(rid)
        histories = {key: [{"value": m.value, "timestamp": m.timestamp, "step": m.step}
                           for m in client.get_metric_history(rid, key)] for key in run.data.metrics}
        removed = []
        tags = _projection(run.data.tags, removed)
        runs.append({"source_run_id": rid, "source_start_time": run.info.start_time,
                     "source_end_time": run.info.end_time, "source_status": run.info.status,
                     "params": _projection(run.data.params, removed), "tags": tags,
                     "metrics": histories, "omitted_fields": removed})
    mlflow_export = {"schema_version": 1, "parent_run_id": parent_id,
                     "export_method": "MlflowClient get_run/get_metric_history",
                     "source_store_sha256": _hash(source / "tracking.sqlite"), "runs": runs,
                     "registry": completion["promotion"]["registry"]}
    _write(destination / "mlflow_export.json", mlflow_export)
    artifacts.append({"path": "mlflow_export.json", "size": (destination / "mlflow_export.json").stat().st_size,
                      "sha256": _hash(destination / "mlflow_export.json"), "kind": "api_export",
                      "source_path": "tracking.sqlite", "source_sha256": _hash(source / "tracking.sqlite")})
    manifest = {"schema_version": 1, "name": "Robin — synthetic Transformer workflow",
                "example_id": "transformer-regime", "objective_id": objective["objective_id"],
                "source_run_id": source.name, "dataset": "synthetic.parquet",
                "dataset_sha256": expected, "n_rows": len(frame), "synthetic_only": True,
                "selected_model_id": summary["selected_model_id"], "models": models,
                "evidence_kind": "recorded_example", "seed": objective["model_config"]["seed"],
                "generator": "src.experiment.workflow_data.make_synthetic_dataset",
                "features": [col for col in frame.columns if col.startswith("wf_")],
                "outcomes": ["label", "future_price", "future_end"],
                "original_code_bundle_included": False,
                "original_deployment_package_included": False,
                "provenance_note": "Source run is immutable. Original files retain exact bytes; projections record omitted deployment coordinates and source hashes. This fixture records prior gate execution, not a new training or promotion.",
                "artifacts": sorted(artifacts, key=lambda a: a["path"])}
    _write(destination / "showcase_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_example(args.run_dir, args.output)
    print(json.dumps({"output": str(args.output), "files": len(manifest["artifacts"]),
                      "dataset_sha256": manifest["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
