"""Bounded, tracked Transformer integration workflow on synthetic data.

Run ``python -m src.experiment.transformer_workflow --help``. This composes
existing model, run-layout, serving and economics implementations. Its isolated
default deployment is EXPERIMENTAL. A frozen synthetic-service objective can
pass independent gates and promote within its own test-serving namespace.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OBJECTIVE = "transformer-workflow-smoke"


@dataclass(frozen=True)
class WorkflowConfig:
    """Fully resolved synthetic workflow settings; unknown fields are refused."""

    objective_id: str = OBJECTIVE
    seed: int = 42
    n_rows: int = 3000
    horizon: int = 4
    seq_len: int = 12
    d_model: int = 16
    n_heads: int = 2
    layers: tuple[int, ...] = (1, 2)
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.003
    dropout: float = 0.1
    confidence_threshold: float = 0.5
    min_accuracy: float = 0.5
    max_runtime_seconds: int = 900

    def __post_init__(self) -> None:
        import math
        if self.objective_id not in (OBJECTIVE, "synthetic-regime-service"):
            raise ValueError("unsupported workflow objective_id")
        bounds = {"seed": (0, 2**31 - 1), "n_rows": (1000, 10000),
                  "horizon": (1, 16), "seq_len": (2, 32), "d_model": (8, 64),
                  "n_heads": (1, 8), "epochs": (1, 30), "batch_size": (8, 256),
                  "max_runtime_seconds": (1, 1740)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if len(self.layers) != 2 or len(set(self.layers)) != 2 or any(
            type(n) is not int or not 1 <= n <= 3 for n in self.layers
        ):
            raise ValueError("layers must name two distinct depths in [1, 3]")
        for name, lo, hi in (("learning_rate", 0.00001, 0.1), ("dropout", 0, 0.8),
                             ("confidence_threshold", 0, 1), ("min_accuracy", 0, 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be finite in [{lo}, {hi}]")

    @classmethod
    def read(cls, path: Path) -> WorkflowConfig:
        values = json.loads(path.read_text())
        if "layers" in values:
            values["layers"] = tuple(values["layers"])
        return cls(**values)


def write_record(path: Path, value: Any) -> None:
    """Create a durable record once, refusing to replace existing evidence."""
    serialized = json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(serialized)


def event(directory: Path, stage: str, **details: Any) -> None:
    """Append stage telemetry; completed artifacts themselves are immutable."""
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": stage, **details}
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, default=str, allow_nan=False) + "\n")
    filled = min(20, int(details.get("progress", 0)))
    print(f"{'█' * filled}{'░' * (20 - filled)} {stage}", flush=True)


def acceptance_gate(metrics: dict[str, float], baseline: dict[str, float], minimum: float) -> None:
    """Require actual learning on an untouched population, not just execution."""
    import math
    if not all(math.isfinite(float(v)) for v in [*metrics.values(), *baseline.values()]):
        raise ValueError("nonfinite predictive metrics")
    if metrics["accuracy"] < minimum or metrics["log_loss"] >= baseline["log_loss"]:
        raise ValueError(f"learning gate failed: metrics={metrics}, prior_baseline={baseline}")


def check_serving_environment() -> None:
    """Reject the measured MLflow/Starlette API mismatch before training."""
    import inspect
    from fastapi import FastAPI
    from mlflow.pyfunc import scoring_server
    if not hasattr(FastAPI, "route") and "@app.route(" in inspect.getsource(scoring_server.init):
        raise RuntimeError(
            "Installed MLflow scoring requires FastAPI.route, removed by Starlette 1.x. "
            "Install the tested environment with python -m pip install "
            "-r requirements-transformer.lock, then rerun this command."
        )


def _environment(threads: int) -> dict[str, Any]:
    from src.common.vm_config import get_vm_config
    from src.serving.workflow_model import tensorflow_distribution
    packages = (tensorflow_distribution(), "keras", "mlflow", "numpy", "pandas", "scikit-learn", "pyarrow", "fastapi", "starlette", "uvicorn")
    return {"python": sys.version, "platform": sys.platform,
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "vm": asdict(get_vm_config()), "tensorflow_threads": threads,
            "device": "CPU", "paid_resources_created": False}


def execute(directory: Path) -> dict[str, Any]:
    """Execute the worker; the CLI supervisor supplies the hard wall-clock cap."""
    import joblib
    import numpy as np
    import pandas as pd
    from src.common.code_bundle import build_bundle
    from src.common.code_identity import git_identity
    from src.common.data_paths import logical_path
    from src.common.run_layout import model_dir
    from src.common.vm_config import get_vm_config
    from src.experiment.comparison import compare_runs, compare_results
    from src.experiment.manifest import CodeIdentity, DatasetRef, RunManifest, ValidationDesign
    from src.experiment.model_identity import allocate_model_id, write_model_json
    from src.experiment.workflow_data import (
        FEATURE_COLUMNS, make_synthetic_dataset, chronological_partitions,
        classification_metrics, fit_temperature, apply_temperature, evaluate_synthetic_economics, calibration_report,
    )
    from src.experiment.workflow_tracking import WorkflowTracker
    from src.models.base import ModelSpec
    from src.experiment.schemas import RunMetrics, FoldMetrics
    from src.evaluation.economic import TradeResult
    from src.models.transformer_plugin import TransformerPlugin
    from src.serving.bundle import load_serving_bundle
    from src.serving.predict import predict_canonical

    started = time.monotonic()
    request = json.loads((directory / "request.json").read_text())
    # MLflow pyfunc internals also resolve a tracking client. Pin the dedicated
    # worker's environment, including its serving child, to the explicit store.
    os.environ["MLFLOW_TRACKING_URI"] = request["tracking_uri"]
    os.environ["MLFLOW_REGISTRY_URI"] = request["tracking_uri"]
    os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
    os.environ["DO_NOT_TRACK"] = "true"
    cfg = WorkflowConfig(**{**request["config"], "layers": tuple(request["config"]["layers"])})
    objective = request.get("objective")
    if cfg.objective_id != OBJECTIVE:
        from src.experiment.workflow_gates import load_objective
        frozen = load_objective(directory / "objective.json")
        if frozen != objective or frozen["model_config"] != json.loads(json.dumps(asdict(cfg))):
            raise ValueError("frozen objective/configuration differs from pretraining request")
        if hashlib.sha256((directory / "objective.json").read_bytes()).hexdigest() != request["objective_sha256"]:
            raise ValueError("frozen objective checksum mismatch")
    workers = get_vm_config().workers_for(mem_per_worker_gb=2.0)
    # One candidate at a time; each dense operation can use the detected cores.
    os.environ["OMP_NUM_THREADS"] = str(workers)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(workers)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, workers // 2))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(workers)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, workers // 2))
    from src.common.repro import set_global_seed
    from src.serving.workflow_model import save_workflow_pyfunc, write_bundle_integrity

    write_record(directory / "environment.json", _environment(workers))
    digest, bundle = build_bundle(REPO)
    (directory / "code.tzst").write_bytes(bundle)
    identity = git_identity()
    if identity is None:
        raise RuntimeError("git identity unavailable: cannot verify this workflow")
    code = CodeIdentity(identity[0], identity[1], bundle_sha256=digest)
    event(directory, "prepare causal synthetic data", progress=2)
    data = make_synthetic_dataset(n_rows=cfg.n_rows, seed=cfg.seed, horizon=cfg.horizon)
    source = directory / "synthetic.parquet"
    data.to_parquet(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    parts = chronological_partitions(data, sequence_length=cfg.seq_len)
    split_record = {name: {"start": str(frame.index.min()), "end": str(frame.index.max()),
                           "max_label_end": str(frame.future_end.max()), "rows": len(frame)}
                    for name, frame in parts.items()}
    write_record(directory / "splits.json", split_record)
    datasets = tuple(DatasetRef(f"synthetic-sequence-{kind}", semantic_type=kind,
                               content_sha256=source_sha, n_rows=len(data), location=logical_path(source))
                     for kind in ("features", "outcomes"))
    validation = ValidationDesign(split="chronological_five_way", n_folds=1,
                                  purge_bars=cfg.horizon, embargo_bars=0,
                                  weighting="uniform", fold_windows=(split_record,))
    features = list(FEATURE_COLUMNS)
    priors = np.bincount(parts["train"].label.to_numpy(), minlength=3).astype(float) + 1.0
    priors /= priors.sum()
    tracker = WorkflowTracker(request["tracking_uri"], "vtech-transformer-workflow",
                              artifact_location=request.get("artifact_location"))
    persistence = None
    if objective:
        from src.experiment.workflow_gates import fit_persistence_baseline, persistence_probabilities, evaluate_gate
        persistence = fit_persistence_baseline(parts["train"])
        write_record(directory / "persistence_baseline.json", {"fit_partition": "train", "probabilities": persistence.tolist()})
    manifests = []
    designs = []
    with tracker.run(directory.name, tags={"workflow": cfg.objective_id, "synthetic_only": "true"}) as parent:
        write_record(directory / "tracking.json", {"tracking_uri": request["tracking_uri"], "parent_run_id": parent.run_id})
        parent.log_params({**asdict(cfg), "data_sha256": source_sha, "code_digest": digest,
                           "cpu_threads": workers, "selection_metric": "validation.log_loss"})
        for index, depth in enumerate(cfg.layers):
            tf.keras.backend.clear_session()
            set_global_seed(cfg.seed)
            params = {"seq_len": cfg.seq_len, "d_model": cfg.d_model, "n_heads": cfg.n_heads,
                      "n_layers": depth, "dropout": cfg.dropout, "epochs": cfg.epochs,
                      "batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate,
                      "class_balance": None, "head": "multiclass", "n_features": len(features)}
            mid = allocate_model_id(cfg.objective_id, directory.name, 1, "transformer", params,
                                    feature_spec=features, target="label")
            output = model_dir(cfg.objective_id, directory.name, 1, mid)
            output.mkdir(parents=True, exist_ok=False)
            with tracker.run(f"transformer-depth-{depth}", parent_run_id=parent.run_id,
                             tags={"model_id": mid, "lifecycle": "EXPERIMENTAL"}) as tracked:
                tracked.log_params(params)
                tracked.log_params({"seed": cfg.seed, "dataset_sha256": source_sha})

                class EpochTracking(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                        tracked.log_metrics({f"train.{k}": float(v) for k, v in (logs or {}).items()}, step=epoch)
                        event(directory, f"design {index + 1}/2 epoch {epoch + 1}/{cfg.epochs}", progress=4 + index * 4)

                plugin = TransformerPlugin()
                model = plugin.build(ModelSpec("transformer", params), {})
                fold = {"X_train": parts["train"][features].to_numpy(), "y_train": parts["train"].label.to_numpy(),
                        "X_val": parts["earlystop"][features].to_numpy(), "y_val": parts["earlystop"].label.to_numpy()}
                fitted = plugin.fit(model, [fold], objective={}, callbacks=[EpochTracking()])
                fitted.artifact.save(output / "model.keras")
                joblib.dump(fitted.meta["scaler"], output / "scaler.pkl")
                write_record(output / "feature_schema.json", {"feature_cols": features})
                write_record(output / "finalize_manifest.json", {"family": "transformer", "kind": "sequence",
                             "seq_len": cfg.seq_len, "spec_params": params, "objective_id": cfg.objective_id,
                             "candidate_id": mid, "lifecycle": "EXPERIMENTAL", "synthetic_only": True})
                serving = load_serving_bundle(output)
                cal_frame = parts["calibration"]
                cal_prob = predict_canonical(cal_frame[features], serving).iloc[cfg.seq_len - 1:][["p_down", "p_flat", "p_up"]].to_numpy()
                temperature = fit_temperature(cal_prob, cal_frame.label.iloc[cfg.seq_len - 1:].to_numpy())
                write_record(output / "calibration.json", calibration_report(
                    cal_prob, cal_frame.label.iloc[cfg.seq_len - 1:].to_numpy(), temperature, cfg.confidence_threshold))
                raw = predict_canonical(parts["validation"][features], serving).iloc[cfg.seq_len - 1:].reset_index(drop=True)
                prob_cols = ["p_down", "p_flat", "p_up"]
                # Verify the original in-memory weights against the production bundle loader.
                before = predict_canonical(parts["validation"][features], {
                    **serving, "booster": fitted.artifact, "scaler": fitted.meta["scaler"],
                }).iloc[cfg.seq_len - 1:][prob_cols].to_numpy()
                parity = float(np.max(np.abs(before - raw[prob_cols].to_numpy())))
                if parity > 1e-6:
                    raise RuntimeError(f"save/reload prediction mismatch: {parity}")
                prob = apply_temperature(raw[prob_cols].to_numpy(), temperature)
                evaluated = parts["validation"].iloc[cfg.seq_len - 1:]
                metrics = classification_metrics(evaluated.label.to_numpy(), prob)
                baseline = classification_metrics(evaluated.label.to_numpy(), np.tile(priors, (len(evaluated), 1)))
                gate = None
                if objective:
                    baseline_prob = np.tile(priors, (len(evaluated), 1))
                    persistence_prob = persistence_probabilities(evaluated, persistence)
                    gate = evaluate_gate(evaluated.label.to_numpy(), prob, baseline_prob, persistence_prob, objective["gates"])
                    write_record(output / "validation_gate.json", gate)
                    np.savez(output / "validation_evidence.npz", labels=evaluated.label.to_numpy(), probabilities=prob,
                             prior_probability=baseline_prob, persistence_probability=persistence_prob)
                economics = evaluate_synthetic_economics(evaluated, prob, confidence_threshold=cfg.confidence_threshold)
                from src.serving.predict import direction_from_proba
                direction, confidence = direction_from_proba(prob[:, 0], prob[:, 1], prob[:, 2], cfg.confidence_threshold)
                oof = pd.DataFrame({"ts": evaluated.index, "direction": direction, "confidence": confidence,
                                    "direction_true": np.array(["down", "flat", "up"])[evaluated.label.to_numpy()],
                                    "p_down": prob[:, 0], "p_flat": prob[:, 1], "p_up": prob[:, 2]})
                oof.to_parquet(output / "oof_predictions.parquet", index=False)
                write_record(output / "metrics.json", asdict(RunMetrics(len(evaluated), metrics)))
                write_record(output / "fold_metrics.json", {"folds": [asdict(FoldMetrics(
                    fold_id="validation", train_start=str(parts["train"].index.min()), train_end=str(parts["train"].index.max()),
                    test_start=str(evaluated.index.min()), test_end=str(evaluated.index.max()),
                    n_train=len(parts["train"]) - cfg.seq_len + 1, n_test=len(evaluated), metrics=metrics))]})
                write_record(output / "selected_population.json", {"n_raw": len(oof),
                             "n_selected": int((oof.direction != "flat").sum()), "selection_rule": "fixed confidence threshold",
                             "population_id": hashlib.sha256(evaluated.index.asi8.tobytes()).hexdigest()})
                pd.DataFrame(economics["ledger"], columns=[field.name for field in fields(TradeResult)]).to_parquet(
                    output / "trade_ledger.parquet", index=False)
                write_record(output / "economic_metrics.json", economics["summary"])
                write_record(output / "cost_sensitivity.json", economics["sensitivity"])
                write_record(output / "baseline.json", baseline)
                write_model_json(output, model_id=mid, model_type="transformer", hyperparameters=params,
                                 feature_columns=features, target="label", oof_columns=list(oof.columns),
                                 inputs={"join_keys": ["ts"], "population_cache": logical_path(source)})
                artifacts = {name: [filename] for name, filename in {
                    "metrics": "metrics.json", "fold_metrics": "fold_metrics.json", "model": "model.json",
                    "calibration": "calibration.json", "oof_predictions": "oof_predictions.parquet",
                    "selected_population": "selected_population.json", "trade_ledger": "trade_ledger.parquet",
                    "economic_metrics": "economic_metrics.json", "cost_sensitivity": "cost_sensitivity.json"}.items()}
                manifest = RunManifest(run_id=f"{directory.name}-{mid}", objective_id=cfg.objective_id,
                    hypothesis="Compare Transformer encoder depth with identical data, split and training settings.",
                    responsibility_versions={"model_strategy": "src.models.transformer_plugin", "validation": "workflow_data.chronological_partitions",
                                             "economic_evaluation": "src.evaluation.economic"}, code=code, datasets=datasets,
                    config_snapshot=params, config_snapshot_resolved=True, validation=validation, seeds={"all": cfg.seed},
                    execution_assumptions=economics["assumptions"],
                    decision_policy={"method": "confidence", "threshold": cfg.confidence_threshold},
                    calibration={"method": "temperature", "fit_partition": "calibration"}, artifacts=artifacts,
                    extra={"mlflow_run_id": tracked.run_id, "synthetic_only": True, "temperature": temperature})
                manifest.write(output)
                if manifest.missing_evidence():
                    raise RuntimeError(f"incomplete provenance: {manifest.missing_evidence()}")
                tracked.log_metrics({f"validation.{key}": value for key, value in metrics.items()})
                tracked.log_metrics({"reload_max_abs_error": parity, "temperature": temperature})
                tracked.log_artifacts(str(output), artifact_path="design")
                evidence = tracked.verify(params={"n_layers": depth}, metrics={f"validation.{key}": value for key, value in metrics.items()},
                                          artifacts=("design/model.keras", "design/manifest.json"))
                write_record(output / "tracking_verification.json", evidence)
                tracked.log_artifact(str(output / "tracking_verification.json"), artifact_path="design")
                manifests.append(manifest)
                designs.append({"model_id": mid, "depth": depth, "directory": str(output), "metrics": metrics,
                                "baseline": baseline, "temperature": temperature, "mlflow_run_id": tracked.run_id,
                                "reload_max_abs_error": parity, "validation_gate": gate})
                del fitted, model, serving

        event(directory, "compare designs and freeze selection", progress=13)
        setup = compare_runs(*manifests)
        results = compare_results(Path(designs[0]["directory"]), Path(designs[1]["directory"]), *manifests)
        write_record(directory / "comparison.json", {"setup": asdict(setup), "results": asdict(results), "designs": designs})
        if not setup.controlled_comparison or results.missing_expected or results.invalid_evidence:
            raise RuntimeError("comparison evidence is incomplete or uncontrolled; inspect comparison.json")
        eligible = [d for d in designs if not objective or d["validation_gate"]["passed"]]
        if not eligible:
            raise ValueError("no design passed the preregistered validation gates")
        winner = min(eligible, key=lambda item: item["metrics"]["log_loss"])
        if objective:
            import shutil
            write_record(directory / "validation_gate.json", winner["validation_gate"])
            shutil.copyfile(Path(winner["directory"]) / "validation_evidence.npz", directory / "validation_evidence.npz")
        acceptance_gate(winner["metrics"], winner["baseline"], cfg.min_accuracy)
        write_record(directory / "selection.json", {"selected": winner["model_id"], "criterion": "minimum validation log_loss among eligible designs",
                     "holdout_consulted": False, "designs": designs})
        # Exclusive marker precedes the first use of holdout outcomes. A run cannot resume/re-tune it.
        write_record(directory / "holdout_consultation.json", {"selected": winner["model_id"], "consultations": 1})
        event(directory, "evaluate frozen model on final holdout", progress=15)
        selected_path = Path(winner["directory"])
        serving = load_serving_bundle(selected_path)
        holdout = parts["holdout"]
        hold_raw = predict_canonical(holdout[features], serving).iloc[cfg.seq_len - 1:]
        hold_prob = apply_temperature(hold_raw[["p_down", "p_flat", "p_up"]].to_numpy(), winner["temperature"])
        hold_eval = holdout.iloc[cfg.seq_len - 1:]
        hold_metrics = classification_metrics(hold_eval.label.to_numpy(), hold_prob)
        hold_baseline = classification_metrics(hold_eval.label.to_numpy(), np.tile(priors, (len(hold_eval), 1)))
        write_record(directory / "holdout.json", {"metrics": hold_metrics, "baseline": hold_baseline, "synthetic_only": True})
        if objective:
            hold_prior = np.tile(priors, (len(hold_eval), 1))
            hold_persistence = persistence_probabilities(hold_eval, persistence)
            hold_gate = evaluate_gate(hold_eval.label.to_numpy(), hold_prob, hold_prior, hold_persistence, objective["gates"])
            write_record(directory / "holdout_gate.json", hold_gate)
            np.savez(directory / "holdout_evidence.npz", labels=hold_eval.label.to_numpy(), probabilities=hold_prob,
                     prior_probability=hold_prior, persistence_probability=hold_persistence)
            # Controls use the exact same gate; their rejection is mandatory.
            controls = {
                "prior_only": evaluate_gate(hold_eval.label.to_numpy(), hold_prior, hold_prior, hold_persistence, objective["gates"]),
                "shuffled_predictions": evaluate_gate(hold_eval.label.to_numpy(),
                    np.random.default_rng(cfg.seed + 1).permutation(hold_prob), hold_prior, hold_persistence, objective["gates"]),
            }
            write_record(directory / "negative_controls.json", controls)
            if not hold_gate["passed"] or any(report["passed"] for report in controls.values()):
                raise ValueError("frozen holdout gates or negative controls failed; no promotion")
        acceptance_gate(hold_metrics, hold_baseline, cfg.min_accuracy)
        parent.log_metrics({f"holdout.{key}": value for key, value in hold_metrics.items()})

        event(directory, "export and register isolated inference package", progress=17)
        write_bundle_integrity(selected_path)
        example = parts["validation"][features].iloc[:cfg.seq_len + 8].reset_index()
        example["ts"] = example["ts"].astype(str)
        package = save_workflow_pyfunc(selected_path, directory / "deployment", example)
        import mlflow.pyfunc
        deployed = mlflow.pyfunc.load_model(str(package))
        expected = deployed.predict(example)
        if objective:
            write_record(directory / "promotion_input.json", json.loads(example.to_json(orient="split", date_format="iso")))
            write_record(directory / "promotion_expected.json", json.loads(expected.to_json(orient="split", date_format="iso")))
        parent.log_artifacts(str(package), artifact_path="deployment")
        registered = parent.register_model(name=f"workflow-test-transformer-{directory.name}", artifact_path="deployment")
        write_record(directory / "registry.json", registered)
        (directory / "downloaded").mkdir()
        downloaded = tracker.client.download_artifacts(parent.run_id, "deployment", str(directory / "downloaded"))
        restored = mlflow.pyfunc.load_model(downloaded)
        actual = restored.predict(example)
        np.testing.assert_allclose(expected[["p_down", "p_flat", "p_up"]], actual[["p_down", "p_flat", "p_up"]], atol=1e-6, rtol=0)
        comparison = apply_temperature(predict_canonical(parts["validation"][features].iloc[:cfg.seq_len + 8], serving)
                                       .iloc[cfg.seq_len - 1:][["p_down", "p_flat", "p_up"]].to_numpy(), winner["temperature"])
        np.testing.assert_allclose(actual.iloc[cfg.seq_len - 1:][["p_down", "p_flat", "p_up"]], comparison, atol=1e-6, rtol=0)
        from src.serving.workflow_http import verify_http_deployment
        http = verify_http_deployment(Path(downloaded), example, actual, directory)
        write_record(directory / "deployment_verification.json", {"artifact_download_reload": True, "probability_parity": True,
                     "lifecycle": "EXPERIMENTAL", "registry": registered, "input_rows": len(example), "http": http})
        elapsed = time.monotonic() - started
        summary = {"status": "VALIDATED", "elapsed_seconds": elapsed, "selected_model_id": winner["model_id"],
                   "holdout_metrics": hold_metrics, "holdout_baseline": hold_baseline, "designs": designs,
                   "controlled_comparison": True, "deployment_verified": True, "synthetic_only": True,
                   "mlflow_parent_run_id": parent.run_id, "tracking_uri": request["tracking_uri"], "run_directory": str(directory)}
        write_record(directory / "summary.json", summary)
        # Manifest for the orchestration run; model design manifests remain independently comparable.
        root_manifest = RunManifest(run_id=directory.name, objective_id=cfg.objective_id,
            hypothesis="The complete Transformer workflow learns, tracks, compares and serves reproducibly within its time budget.",
            responsibility_versions={"orchestration": "src.experiment.transformer_workflow"}, code=code, datasets=datasets,
            config_snapshot=asdict(cfg), config_snapshot_resolved=True, validation=validation, seeds={"all": cfg.seed},
            execution_assumptions=manifests[0].execution_assumptions, decision_policy=manifests[0].decision_policy,
            calibration=manifests[0].calibration,
            artifacts={key: [str(Path(d["directory"]).relative_to(directory) / paths[0]) for d in designs]
                       for key, paths in manifests[0].artifacts.items()},
            passes=({"pass": 1, "role": "design_comparison", "models": [{"model_id": d["model_id"]} for d in designs]},),
            extra={"synthetic_only": True, "mlflow_run_id": parent.run_id})
        root_manifest.write(directory)
        for name in ("manifest.json", "summary.json", "comparison.json", "selection.json", "holdout.json", "splits.json", "environment.json", "code.tzst", "deployment_verification.json"):
            parent.log_artifact(str(directory / name), artifact_path="workflow")
        if objective:
            for name in ("request.json", "objective.json", "validation_gate.json", "holdout_gate.json",
                         "validation_evidence.npz", "holdout_evidence.npz", "negative_controls.json",
                         "holdout_consultation.json", "persistence_baseline.json"):
                parent.log_artifact(str(directory / name), artifact_path="workflow")
        parent.verify(metrics={f"holdout.{key}": value for key, value in hold_metrics.items()},
                      artifacts=("workflow/summary.json", "deployment/MLmodel"))
    event(directory, "training and deployment checks completed", progress=18 if objective else 20,
          elapsed_seconds=time.monotonic() - started)
    write_record(directory / "completion.json", {"status": "PASSED", "elapsed_seconds": time.monotonic() - started,
                 "mlflow_parent_run_id": parent.run_id, "tracking_status": "FINISHED",
                 "scope": "training_and_deployment_only" if objective else "smoke_workflow",
                 "promotion_required": bool(objective)})
    if objective:
        event(directory, "promote gated synthetic service and verify promoted inference", progress=19)
        from src.serving.workflow_promotion import promote_test_objective
        receipt = promote_test_objective(directory)
        tracker.client.log_artifact(parent.run_id, str(directory / "promotion.json"), artifact_path="workflow")
        write_record(directory / "promotion_completion.json", {"status": "PASSED", "lifecycle": "PRODUCTION",
                     "scope": "synthetic_service_only", "elapsed_seconds": time.monotonic() - started,
                     "promotion": receipt})
        tracker.client.log_artifact(parent.run_id, str(directory / "promotion_completion.json"), artifact_path="workflow")
        tracker.client.set_tag(parent.run_id, "workflow.promotion_status", "PASSED")
        event(directory, "promotion completed", progress=20, elapsed_seconds=time.monotonic() - started)
    return summary


def supervise(directory: Path, *, timeout_seconds: int) -> int:
    """Kill this worker's process group at the deadline, retaining failure evidence."""
    started = time.monotonic()
    with (directory / "worker.log").open("x") as log:
        worker = subprocess.Popen([sys.executable, "-m", "src.experiment.transformer_workflow", "--execute", str(directory)],
                                  cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(worker.args, timeout_seconds)
                try:
                    status = worker.wait(timeout=min(15, remaining))
                    break
                except subprocess.TimeoutExpired:
                    stages = (directory / "events.jsonl")
                    stage = json.loads(stages.read_text().splitlines()[-1])["stage"] if stages.exists() else "initializing"
                    print(f"██████████░░░░░░░░░░ {stage} ({time.monotonic() - started:.0f}s)", flush=True)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            status = 124 if isinstance(exc, subprocess.TimeoutExpired) else 130
            write_record(directory / "supervisor_failure.json", {"status": "TIMED_OUT" if status == 124 else "CANCELLED",
                         "elapsed_seconds": time.monotonic() - started, "budget_seconds": timeout_seconds})
            _terminate_tracking(directory)
    if status != 0:
        if not (directory / "failure.json").exists() and not (directory / "supervisor_failure.json").exists():
            write_record(directory / "supervisor_failure.json", {"status": "FAILED", "worker_exit_code": status,
                         "elapsed_seconds": time.monotonic() - started})
            _terminate_tracking(directory)
        print((directory / "worker.log").read_text()[-7000:], file=sys.stderr)
    else:
        print(f"✅ Done in {time.monotonic() - started:.1f}s — {directory}", flush=True)
    return status


def _terminate_tracking(directory: Path) -> None:
    """Best effort after a killed process; failure is recorded, never suppressed."""
    path = directory / "tracking.json"
    if not path.exists():
        return
    try:
        from mlflow.tracking import MlflowClient
        record = json.loads(path.read_text())
        client = MlflowClient(tracking_uri=record["tracking_uri"])
        parent = client.get_run(record["parent_run_id"])
        runs = client.search_runs([parent.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'")
        for run in [*runs, parent]:
            if run.info.status == "RUNNING":
                client.set_terminated(run.info.run_id, "KILLED")
    except Exception as exc:
        write_record(directory / "tracking_cleanup_failure.json", {"error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--objective", type=Path, help="Frozen synthetic-service objective, including model settings and promotion gates.")
    parser.add_argument("--data-root", type=Path, help="Override VTECH_DATA_DIR for an isolated workflow test.")
    parser.add_argument("--tracking-uri", help="Explicit MLflow endpoint; default: isolated SQLite in this run.")
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute:
        try:
            execute(args.execute)
        except BaseException as exc:
            write_record(args.execute / "failure.json", {"type": type(exc).__name__, "error": str(exc)})
            raise
        return 0
    objective = None
    if args.objective:
        if args.config:
            parser.error("--objective includes frozen model settings; --config cannot override them")
        from src.experiment.workflow_gates import load_objective
        objective = load_objective(args.objective)
        cfg = WorkflowConfig(**{**objective["model_config"], "layers": tuple(objective["model_config"]["layers"])})
    else:
        cfg = WorkflowConfig.read(args.config or REPO / "config/workflows/transformer_smoke.json")
        if cfg.objective_id != OBJECTIVE:
            parser.error("promotion requires --objective with a frozen evaluation contract")
    check_serving_environment()
    if args.data_root:
        os.environ["VTECH_DATA_DIR"] = str(args.data_root.resolve())
    if objective:
        from src.serving.promote import latest_pointer_path
        if latest_pointer_path(cfg.objective_id).exists():
            raise FileExistsError("This test objective already has a promoted model. Use a new --data-root for a fresh test; existing promotion is preserved.")
    # No implicit live MLFLOW_TRACKING_URI: integration tests default to their own DB.
    from src.common.run_layout import new_run_dir
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = new_run_dir(cfg.objective_id, rid)
    from shutil import disk_usage
    existing = directory
    while not existing.exists():
        existing = existing.parent
    if disk_usage(existing).free < 256 * 1024 * 1024:
        raise OSError(f"Insufficient artifact space at {existing}; need 256 MiB. Use --data-root on a volume with space.")
    directory.mkdir(parents=True, exist_ok=False)
    tracking = args.tracking_uri or f"sqlite:///{directory / 'tracking.sqlite'}"
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    if objective:
        write_record(directory / "objective.json", objective)
    write_record(directory / "request.json", {"config": asdict(cfg), "tracking_uri": tracking,
                 "artifact_location": None if args.tracking_uri else (directory / "mlflow_artifacts").as_uri(),
                 "objective": objective,
                 "objective_sha256": hashlib.sha256((directory / "objective.json").read_bytes()).hexdigest() if objective else None})
    print(f"Run: {directory}\nBudget: {cfg.max_runtime_seconds}s\nMLflow: {tracking}", flush=True)
    return supervise(directory, timeout_seconds=cfg.max_runtime_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
