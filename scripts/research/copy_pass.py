#!/usr/bin/env python3
"""Copy one existing population and its scores file into a run.

The population is copied once, to ``pass_<n>/training_matrix.parquet``.
The OOF is projected to row keys, fold, outcomes, predictions and
train-derived thresholds; every column the population already carries as a
feature is dropped from it.
Feature columns stay in the population and are named on ``model.json``.
Run metadata is ``manifest.json``. ``run.json``, ``contract.json`` and
``folds.json`` are not written.

``model_id`` is a unique id, not the model type. Omit ``--model-id`` to
allocate one from the model type, the hyperparameters below, the copied
matrix columns, and ``--target``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.economic import assumptions_record  # noqa: E402
from src.experiment.artifacts import compact_oof_columns  # noqa: E402
from src.experiment.model_identity import (  # noqa: E402
    allocate_model_id,
    validate_model_id,
    write_model_json,
)
from src.common.run_layout import (  # noqa: E402
    compact_parquets,
    merge_run_json,
    model_dir,
    new_run_ts,
    previous_model,
    oof_predictions_path,
    population_cache_path,
)
from src.training.cascade_run_spec import CASCADE_PASS, xgb_hyperparameters  # noqa: E402
from src.objectives.loader import load_objective  # noqa: E402


def _matrix_files(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    files = sorted(p for p in src.glob("*.parquet") if p.is_file())
    if not files:
        raise SystemExit(f"no parquet under {src}")
    return files


def _compact_columns(score_columns: list[str], population_columns: list[str]) -> list[str]:
    """The OOF columns to keep: row keys, fold, outcomes, predictions, thresholds.

    A historical scores file carries the feature columns it was scored on.
    Those live in the population, so the OOF drops them --
    ``compact_oof_columns`` keeps a column that is also a key, an outcome or a
    prediction even when the population carries it too.
    """
    return compact_oof_columns(list(score_columns), list(population_columns))


def _write_compact_oof(
    scores_path: Path, dest: Path, columns: list[str],
) -> tuple[int, list[str]]:
    """Write the OOF with feature columns projected out.

    Only ``columns`` is read, so a wide historical scores file never has to be
    materialized in full.
    """
    scores = pd.read_parquet(scores_path, columns=columns)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    scores.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(dest)
    return len(scores), list(scores.columns)


def _label_build_id(matrix: Path) -> str | None:
    """``build_id`` from a genuine ``_build.json`` beside the label dataset.

    Written by ``label_store.write_build_sidecar`` only when real code/config/
    upstream evidence exists, so its presence means the realization identity is
    trustworthy. Absent for every historical ``*_v4`` directory -- in which case
    the ref stays contract-only, which is weaker but true.
    """
    import json

    from src.training.label_store import BUILD_NAME

    sidecar = (matrix if matrix.is_dir() else matrix.parent) / BUILD_NAME
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text()).get("build_id") or None
    except (OSError, ValueError):
        return None


def _label_contract_id(matrix: Path) -> str | None:
    """`contract_id` from the `_contract.json` beside the label dataset.

    A pass that cannot name its outcome contract has no dataset identity at
    all, so the manifest would be INCOMPLETE on `datasets` for a reason that
    is purely a missing read.
    """
    import json

    from src.training.label_store import CONTRACT_NAME

    sidecar = (matrix if matrix.is_dir() else matrix.parent) / CONTRACT_NAME
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text()).get("contract_id") or None


def _folds_from_scores(scores_path: Path) -> dict:
    df = pd.read_parquet(scores_path, columns=["session", "fold"])
    df["session"] = pd.to_datetime(df["session"], utc=True)
    folds = []
    for fold, g in df.groupby("fold"):
        folds.append({
            "fold": int(fold),
            "test_start": g["session"].min().date().isoformat(),
            "test_end": g["session"].max().date().isoformat(),
            "n_rows": int(len(g)),
        })
    folds.sort(key=lambda r: r["fold"])
    last = df[df["fold"] == df["fold"].max()]
    hold = sorted({ts.date().isoformat() for ts in last["session"]})
    digest = hashlib.sha256("\n".join(hold).encode()).hexdigest()
    return {
        "split": "expanding_month_blocks",
        "hold_out_dates": hold,
        "hold_out_hash": digest,
        "hold_out_hash_of": "hold_out_session_dates",
        "folds": folds,
    }


def copy_pass(args: argparse.Namespace) -> str:
    run_ts = args.run_ts or new_run_ts()
    hyper = xgb_hyperparameters(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, seed=args.seed)
    matrix_files = _matrix_files(Path(args.matrix))
    if args.model_id:
        model_id = validate_model_id(args.model_id, args.model_type)
        feature_spec = None
    else:
        columns = list(pq.read_schema(matrix_files[0]).names)
        feature_spec = {"columns": columns}
        model_id = validate_model_id(
            allocate_model_id(
                args.objective, run_ts, args.pass_n, args.model_type,
                hyper, feature_spec=feature_spec, target=args.target),
            args.model_type)
    matrix_dest = population_cache_path(args.objective, run_ts, args.pass_n)
    print(f"population cache: {len(matrix_files)} files -> {matrix_dest}", flush=True)
    if matrix_dest.exists() and not args.force:
        print("population cache already present", flush=True)
    else:
        n = compact_parquets(matrix_files, matrix_dest)
        print(f"population rows {n:,}", flush=True)

    population_columns = list(pq.read_schema(matrix_files[0]).names)
    score_columns = list(pq.read_schema(args.scores).names)
    compact = _compact_columns(score_columns, population_columns)
    #: This model's features are the population columns the compact OOF drops.
    feature_columns = [c for c in population_columns if c not in set(compact)]
    dropped = len(score_columns) - len(compact)
    oof = oof_predictions_path(args.objective, run_ts, args.pass_n, model_id)
    if oof.exists() and not args.force:
        print(f"oof already present: {oof}", flush=True)
        score_columns = list(pq.read_schema(oof).names)
    else:
        rows, score_columns = _write_compact_oof(Path(args.scores), oof, compact)
        print(f"oof rows {rows:,}, {len(compact)} columns "
              f"({dropped} feature columns dropped) -> {oof}", flush=True)

    trained_on_paths = args.trained_on or [str(args.matrix)]
    label_contract_id = args.label_contract_id or _label_contract_id(Path(args.matrix))
    label_build_id = _label_build_id(Path(args.matrix))
    if not label_contract_id:
        print(f"WARNING: no {'_contract.json'} beside {args.matrix}; this pass will "
              "have no outcome identity. Pass --label-contract-id to supply it.",
              flush=True)
    rel_cache = f"pass_{args.pass_n}/training_matrix.parquet"
    rel_oof = f"pass_{args.pass_n}/model_{model_id}/oof_predictions.parquet"
    inputs = {
        "features": [{
            "semantic_type": "features",
            "location": rel_cache,
            "columns": feature_columns,
            "availability": "timestamp",
            "time_column": "ts",
        }],
        "join_keys": ["ts", "side"],
        "filters": [],
        "population_cache": rel_cache,
        "target": args.target,
    }
    if label_contract_id:
        inputs["labels"] = {
            "semantic_type": "outcomes",
            "logical_id": Path(args.matrix).name,
            "contract_id": label_contract_id,
            "location": str(args.matrix),
        }
        if label_build_id:
            inputs["labels"]["build_id"] = label_build_id
    #: A downstream pass names the upstream model whose predictions it gates on.
    #: Recorded as its own `predictions` input so the manifest carries a
    #: DatasetRef(semantic_type="predictions") rather than hiding the lineage.
    upstream_refs: list[dict] = [
        {"kind": "features", "location": p} for p in trained_on_paths
    ]
    if label_contract_id:
        labels_ref: dict = {
            "kind": "labels",
            "label_contract_id": label_contract_id,
            "logical_id": Path(args.matrix).name,
            "path": str(args.matrix),
        }
        if label_build_id:
            # Strongest truthful identity available: a genuine build sidecar.
            labels_ref["build_id"] = label_build_id
        upstream_refs.append(labels_ref)
    if args.pass_n > 1:
        prev_n, prev_mid = previous_model(args.objective, run_ts, args.pass_n)
        prev_rel = f"pass_{prev_n}/model_{prev_mid}/oof_predictions.parquet"
        prediction_input = {
            "semantic_type": "predictions",
            "logical_id": f"pass_{prev_n}_oof",
            "location": prev_rel,
            "model_id": prev_mid,
            "pass": prev_n,
            "availability": "timestamp",
            "time_column": "ts",
            "build_id": prev_mid,
        }
        inputs["predictions"] = [prediction_input]
        upstream_refs.append({
            "kind": "oof_core",
            "pass": prev_n,
            "model_id": prev_mid,
            "logical_id": f"pass_{prev_n}_oof",
            "build_id": prev_mid,
            "path": prev_rel,
            "columns": "oof_core_columns",
        })
    write_model_json(
        model_dir(args.objective, run_ts, args.pass_n, model_id),
        model_id=model_id, model_type=args.model_type, hyperparameters=hyper,
        oof_columns=score_columns, feature_spec=feature_spec,
        feature_columns=feature_columns, inputs=inputs, target=args.target,
        label_contract_id=label_contract_id,
    )
    described = CASCADE_PASS.get(args.pass_n, {})
    validation = None
    snapshot = None
    objective_doc = dict(load_objective(args.objective))
    if args.pass_n == 1:
        info = _folds_from_scores(Path(args.scores))
        validation = {
            "split": info["split"],
            "n_folds": len(info["folds"]) or 1,
            "hold_out_dates": info["hold_out_dates"],
            "fold_windows": info["folds"],
        }
        snapshot = dict(objective_doc)
        names = pq.read_schema(args.scores).names
        snapshot["tau_thresholds"] = sorted(
            float(c.removeprefix("tau_")) for c in names if c.startswith("tau_"))
        snapshot["hold_out_hash"] = info["hold_out_hash"]
        snapshot["hold_out_hash_of"] = info["hold_out_hash_of"]
    merge_run_json(args.objective, run_ts, {
        "pass": args.pass_n,
        "role": described.get("role", f"pass_{args.pass_n}"),
        "purpose": described.get("purpose", ""),
        "population_cache": rel_cache,
        "label_contract_id": label_contract_id,
        "inputs": inputs,
        "upstream": upstream_refs,
        "models": [{
            "model_id": model_id,
            "model_type": args.model_type,
            "model_json": f"pass_{args.pass_n}/model_{model_id}/model.json",
            "oof_predictions": rel_oof,
        }],
    }, config_snapshot=snapshot, config_snapshot_resolved=snapshot is not None,
       validation=validation,
       # Genuinely applied: --seed is the seed this fit used.
       seeds={"model": int(args.seed)},
       execution_assumptions=assumptions_record(objective_doc),
       extra=None if label_build_id else {"evidence": {"datasets": (
           f"{Path(args.matrix).name} is contract-identified only: it has no "
           "_build.json, so no genuine realization identity exists for it. Not "
           "reconstructed from its part inventory (retired, tenth pass)."
       )}})
    print(f"run_ts {run_ts}", flush=True)
    return run_ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--run-ts", default=None)
    ap.add_argument("--pass-n", type=int, required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model-id", default=None,
                    help="unique model id. Omit to allocate one from the model type, "
                         "hyperparameters, copied matrix columns, and --target. "
                         "Not the model type.")
    ap.add_argument("--model-type", default="xgboost")
    ap.add_argument("--target", default=None,
                    help="label column this model fits, when a pass can hold more "
                         "than one. Recorded in the model id.")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trained-on", action="append", default=None)
    ap.add_argument("--label-contract-id", default=None,
                    help="outcome contract_id for this pass. Read from the "
                         "_contract.json beside --matrix when omitted.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    copy_pass(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
