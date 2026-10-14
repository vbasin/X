"""Offline synthetic showcase of the platform's research contracts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.common.causal import as_of_index, assert_known_at_or_before
from src.common.code_identity import git_identity
from src.evaluation.economic import (
    EconomicAssumptions, Leg, LegRole, OrderType, Quote, Trade,
    write_research_economics,
)
from src.experiment.manifest import CodeIdentity, DatasetRef, RunManifest, ValidationDesign
from src.experiment.schemas import FoldMetrics, RunMetrics, validate_oof_columns
from src.models.base import ModelSpec, validate_canonical
from src.models.tabular_plugins import LogisticRegressionPlugin
from src.validation.walk_forward import apply_schedule_to_arrays, build_expanding_schedule

FEATURES = ["demo_return", "demo_momentum", "demo_volatility", "demo_context"]
DIRECTIONS = np.array(["down", "flat", "up"])


def synthetic_frame(seed: int = 42, *, days: int = 8) -> pd.DataFrame:
    """Generate decision-time state and one-minute forward outcomes; no market data."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    ts = pd.to_datetime(np.concatenate([
        pd.date_range(day + timedelta(hours=14), periods=91, freq="min").asi8
        for day in pd.date_range("2026-01-05", periods=days, tz="UTC")
    ]), utc=True)
    moves = rng.normal(0, 0.8, len(ts))
    price = pd.Series(20000 + np.cumsum(moves), index=ts)
    returns = price.diff().fillna(0)
    context = returns.rolling(5, min_periods=1).mean().iloc[::5]
    pos = as_of_index(context.index.asi8, ts.asi8)
    assert_known_at_or_before(context.index.asi8[pos], ts.asi8)
    frame = pd.DataFrame({
        "price": price,
        "demo_return": returns,
        "demo_momentum": returns.rolling(5, min_periods=1).mean(),
        "demo_volatility": returns.rolling(5, min_periods=2).std().fillna(0),
        "demo_context": context.to_numpy()[pos],
        "context_available_ts": context.index[pos],
        "exit_price": price.shift(-1),
        "exit_ts": pd.Series(ts, index=ts).shift(-1),
    }, index=ts)
    frame.index.name = "ts"
    # Exclude overnight outcomes and the unfinished final outcome.
    frame = frame.loc[(frame["exit_ts"] - frame.index) == np.timedelta64(60, "s")].copy()
    forward = frame["exit_price"] - frame["price"]
    frame["label"] = np.where(forward > 0.25, 2, np.where(forward < -0.25, 0, 1))
    return frame


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_report(path: Path, metrics: dict, economics: dict, fold_metrics: list[dict]) -> None:
    rows = "".join(
        f"<tr><td>{r['fold_id']}</td><td>{r['n_train']}</td><td>{r['n_test']}</td>"
        f"<td>{r['metrics']['accuracy']:.3f}</td></tr>" for r in fold_metrics
    )
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>X | Synthetic research showcase</title><style>'
        'body{font:17px system-ui;background:#111827;color:#e5e7eb;max-width:960px;'
        'margin:3rem auto;padding:1rem}h1,h2{color:#67e8f9}section{background:#1f2937;'
        'padding:1.5rem;margin:1rem 0;border-radius:12px}td,th{text-align:left;'
        'padding:.6rem 1.3rem}a{color:#67e8f9}strong{color:#fcd34d}</style>'
        '<h1>X: ML platform showcase</h1>'
        '<p><strong>SYNTHETIC DATA ONLY. Not evidence of trading performance.</strong></p>'
        '<section><h2>Research pipeline</h2><p>Synthetic quotes &rarr; causal features '
        '&rarr; purged calendar folds &rarr; CPU model &rarr; OOF predictions '
        '&rarr; cost-explicit ledger &rarr; experiment manifest.</p></section>'
        f'<section><h2>Out-of-fold evaluation</h2><p>{metrics["n_samples"]} decisions; '
        f'accuracy {metrics["metrics"]["accuracy"]:.3f}; '
        f'always-flat baseline {metrics["metrics"]["always_flat_accuracy"]:.3f}.</p>'
        '<table><tr><th>Fold</th><th>Train</th><th>Test</th><th>Accuracy</th></tr>'
        f'{rows}</table></section>'
        f'<section><h2>Illustrative execution economics</h2><p>{economics["n_trades"]} '
        f'completed one-minute trades; net ${economics["net_dollars"]:.2f}; '
        f'commission ${economics["total_commission"]:.2f}.</p>'
        '<p>Executable synthetic bid/ask, one commission per round trip and explicit '
        'adverse slippage. No significance, promotion, or live-serving claim.</p></section>'
        '<section><h2>Artifacts</h2><p><a href="manifest.json">Manifest</a> &middot; '
        '<a href="metrics.json">Metrics</a> &middot; '
        '<a href="fold_metrics.json">Fold metrics</a> &middot; '
        '<a href="economic_metrics.json">Economics</a></p>'
        '<p>This offline report is not the operational FastAPI console. The console '
        'source is included; its database/cloud integrations require separate setup.</p>'
        '</section></html>'
    )


def run_demo(output: Path, *, seed: int = 42) -> Path:
    """Write a new run directory; refuse to overwrite any existing output."""
    start = time.monotonic()
    print("[████░░░░░░░░░░░░░░░░] Generating synthetic state", flush=True)
    frame = synthetic_frame(seed)
    schedule = build_expanding_schedule(
        start="2026-01-05", stop="2026-01-12",
        train_n=3, train_unit="days", test_n=1, test_unit="days",
    )
    folds = apply_schedule_to_arrays(
        frame[FEATURES].to_numpy(), frame["label"].to_numpy(), frame.index.to_numpy(),
        schedule, FEATURES, tz_name="UTC", purge_seconds=60,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    oof = []
    fold_metrics = []
    params = {"C": 1.0, "max_iter": 500, "class_weight": "balanced", "random_state": seed}
    for fold in folds:
        plugin = LogisticRegressionPlugin()
        estimator = plugin.build(ModelSpec(family=plugin.name, params=params), {})
        # Use the plugin's estimator with a fold-local pipeline. The full training
        # orchestrator writes operational job state and is not needed offline.
        model = make_pipeline(RobustScaler(), estimator)
        model.fit(fold["X_train"], fold["y_train"])
        probabilities = model.predict_proba(fold["X_oof"])
        labels = model.classes_[probabilities.argmax(axis=1)].astype(int)
        predictions = pd.DataFrame({
            "ts": pd.to_datetime(fold["ts_oof"], utc=True),
            "direction": DIRECTIONS[labels],
            "confidence": probabilities.max(axis=1),
            "direction_true": DIRECTIONS[fold["y_oof"]],
            "fold": fold["fold"],
        })
        validate_canonical(predictions, plugin_name="synthetic_demo")
        validate_oof_columns(predictions.columns)
        oof.append(predictions)
        fold_metrics.append(asdict(FoldMetrics(
            fold_id=str(fold["fold"]),
            train_start=pd.Timestamp(fold["ts_train"][0]).isoformat(),
            train_end=pd.Timestamp(fold["ts_train"][-1]).isoformat(),
            test_start=pd.Timestamp(fold["ts_oof"][0]).isoformat(),
            test_end=pd.Timestamp(fold["ts_oof"][-1]).isoformat(),
            n_train=len(fold["y_train"]), n_test=len(labels),
            metrics={"accuracy": float(accuracy_score(fold["y_oof"], labels))},
        )))
    predictions = pd.concat(oof, ignore_index=True)
    metrics = asdict(RunMetrics(n_samples=len(predictions), metrics={
        "accuracy": float((predictions.direction == predictions.direction_true).mean()),
        "always_flat_accuracy": float((predictions.direction_true == "flat").mean()),
    }, notes="Synthetic demonstration; no real-data generalization claim."))
    print("[████████████░░░░░░░░] Pricing illustrative trades", flush=True)
    selected = predictions.loc[predictions["direction"] != "flat"]
    trades = []
    for row in selected.itertuples(index=False):
        state = frame.loc[row.ts]
        trades.append(Trade(
            trade_id=f"synthetic-{len(trades)}",
            direction=1 if row.direction == "up" else -1,
            entry=Leg(OrderType.MARKET, quote=Quote(state.price - 0.125, state.price + 0.125),
                      ts=row.ts.isoformat()),
            exit=Leg(OrderType.MARKET, role=LegRole.EXIT_TIME,
                     quote=Quote(state.exit_price - 0.125, state.exit_price + 0.125),
                     ts=state.exit_ts.isoformat()),
            sequence=float(len(trades)),
        ))
    assumptions = EconomicAssumptions(
        contract_multiplier=20, commission_per_trade=3.36, slippage_points_per_side=0.05,
        economic_contract_unit="synthetic_NQ_equivalent", underlying="SYNTHETIC",
        fill_model="synthetic one-minute executable bid/ask; no latency simulation",
    )
    economics = write_research_economics(output, trades=trades, assumptions=assumptions)
    input_bytes = frame.reset_index().to_csv(index=False).encode()
    (output / "synthetic_input.csv").write_bytes(input_bytes)
    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "fold_metrics.json", fold_metrics)
    _write_json(output / "model.json", {"family": "logistic_regression", "params": params,
                "features": FEATURES, "preprocessing": "RobustScaler fitted per fold",
                "checkpoint": None, "note": "No serving model is exported."})
    identity = git_identity()
    manifest = RunManifest(
        run_id=f"synthetic-demo/{uuid.uuid4().hex}", objective_id="synthetic-demo",
        hypothesis="Exercise platform contracts without market data or external services.",
        responsibility_versions={
            "features": "src.demo.synthetic_frame", "validation": "expanding_calendar",
            "model": "LogisticRegressionPlugin", "economics": "src.evaluation.economic",
        },
        code=CodeIdentity(*identity) if identity is not None else None,
        datasets=(DatasetRef(logical_id="synthetic-input", semantic_type="raw_source",
                  content_sha256=hashlib.sha256(input_bytes).hexdigest(),
                  location="synthetic_input.csv", n_rows=len(frame)),),
        config_snapshot={"seed": seed, "params": params, "features": FEATURES,
                         "schedule": schedule, "purge_seconds": 60},
        config_snapshot_resolved=True,
        validation=ValidationDesign(split="expanding_calendar", n_folds=len(folds),
                                    purge_bars=1, embargo_bars=0),
        seeds={"numpy_generator": seed, "estimator": seed},
        execution_assumptions=asdict(assumptions),
        decision_policy={"rule": "argmax; flat abstains; one-minute holding period"},
        calibration={"method": "none", "note": "Raw scores, not calibrated probabilities."},
        artifacts={"oof_predictions": ["oof_predictions.parquet"],
                   "metrics": ["metrics.json"], "fold_metrics": ["fold_metrics.json"],
                   "trade_ledger": ["trade_ledger.parquet"],
                   "economic_metrics": ["economic_metrics.json"],
                   "model": ["model.json"], "report": ["report.html"]},
        extra={"synthetic_only": True, "promotion_eligible": False},
    )
    manifest.write(output)
    _write_report(output / "report.html", metrics, economics, fold_metrics)
    print(f"[████████████████████] Done in {time.monotonic() - start:.1f}s: {output / 'report.html'}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for synthetic artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run_demo(args.output, seed=args.seed)
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"Demo refused: {error}\n")


if __name__ == "__main__":
    main()
