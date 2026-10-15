"""Small causal fixtures and evaluation for the Transformer workflow check.

This deliberately learnable synthetic market validates plumbing, not trading
edge. No production dataset, database, tracking store or path is opened here.
The only explicit I/O is loading the canonical execution-cost profile.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.causal import as_of_index, assert_known_at_or_before
from src.evaluation.economic import (
    Leg,
    LegRole,
    OrderType,
    Quote,
    Trade,
    TradeLedger,
    evaluate_cost_sensitivity,
    load_assumptions,
)

DATASET_VERSION = "causal-synthetic-regimes-v1"
FEATURE_COLUMNS = (
    "wf_return_1",
    "wf_momentum_4",
    "wf_momentum_12",
    "wf_volatility_12",
    "wf_context",
)
PARTITION_NAMES = ("train", "earlystop", "calibration", "validation", "holdout")


def make_synthetic_dataset(
    n_rows: int = 3000, seed: int = 42, horizon: int = 4
) -> pd.DataFrame:
    """Generate causal features and forward outcomes on a UTC minute grid.

    An observable, noisy regime sensor is published every four minutes.
    Regimes persist for 32 minutes; the next price increment depends on the
    *current* regime. Separate random streams make prefixes identical when
    ``n_rows`` changes. Labels are down=0, flat=1, up=2 from the signed price
    change over ``horizon`` minutes (deadband 0.15 point per horizon minute).
    ``price``, ``future_price`` and ``future_end`` are evaluation metadata,
    never model inputs. All model inputs have the ``wf_`` prefix.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, (int, np.integer)) or n_rows < 1:
        raise ValueError("n_rows must be a positive integer")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    total = n_rows + horizon
    timestamps = pd.date_range("2024-01-02T14:30:00Z", periods=total, freq="min", name="ts")
    regime = ((np.arange(total) // 32 + seed) % 3 - 1).astype(float)
    price_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    sensor_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    increments = 0.55 * regime[:-1] + price_rng.normal(0.0, 0.16, total - 1)
    prices = np.r_[20000.0, 20000.0 + np.cumsum(increments)]
    price = pd.Series(prices[:n_rows], index=timestamps[:n_rows])
    returns = price.diff().fillna(0.0)

    publication_positions = np.arange(0, total, 4)
    publication_ts = timestamps[publication_positions]
    sensor = regime[publication_positions] + sensor_rng.normal(0.0, 0.12, len(publication_positions))
    known = as_of_index(publication_ts.asi8, price.index.asi8)
    if np.any(known < 0):
        raise ValueError("synthetic sensor has no causal observation")
    assert_known_at_or_before(publication_ts.asi8[known], price.index.asi8)

    future_prices = prices[horizon : horizon + n_rows]
    delta = future_prices - price.to_numpy()
    deadband = 0.15 * horizon
    frame = pd.DataFrame(
        {
            "wf_return_1": returns,
            "wf_momentum_4": returns.rolling(4, min_periods=1).mean(),
            "wf_momentum_12": returns.rolling(12, min_periods=1).mean(),
            "wf_volatility_12": returns.rolling(12, min_periods=1).std(ddof=0),
            "wf_context": sensor[known],
            "label": np.where(delta < -deadband, 0, np.where(delta > deadband, 2, 1)),
            "price": price,
            "future_price": future_prices,
            "future_end": timestamps[horizon : horizon + n_rows],
        },
        index=price.index,
    )
    frame.attrs.update(dataset_version=DATASET_VERSION, seed=int(seed), horizon=int(horizon))
    return frame


def chronological_partitions(
    frame: pd.DataFrame,
    sequence_length: int = 16,
    *,
    fractions: tuple[float, ...] = (0.55, 0.10, 0.10, 0.15, 0.10),
) -> dict[str, pd.DataFrame]:
    """Split, then purge labels reaching a later partition's first timestamp.

    Model sequence construction must run separately on each returned frame.
    Thus its initial ``sequence_length - 1`` rows are context only, with no
    feature or label window crossing train/earlystop/calibration/validation/
    holdout boundaries. Scaling is deliberately left to the model plugin,
    which must fit it on ``train`` only.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("frame requires a timezone-aware DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("frame timestamps must be strictly increasing")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, (int, np.integer)) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if len(fractions) != len(PARTITION_NAMES) or not all(np.isfinite(f) and f > 0 for f in fractions):
        raise ValueError("five positive finite partition fractions are required")
    if not np.isclose(sum(fractions), 1.0, atol=1e-12, rtol=0):
        raise ValueError("partition fractions must sum to one")
    if "future_end" not in frame:
        raise ValueError("frame requires future_end for label purging")
    future_end = pd.DatetimeIndex(frame["future_end"])
    if future_end.tz is None or future_end.hasnans or np.any(future_end.asi8 <= frame.index.asi8):
        raise ValueError("every future_end must be a finite timestamp after its decision")
    boundaries = np.r_[0, (np.cumsum(fractions)[:-1] * len(frame)).astype(int), len(frame)]
    parts: dict[str, pd.DataFrame] = {}
    for number, name in enumerate(PARTITION_NAMES):
        left, right = int(boundaries[number]), int(boundaries[number + 1])
        segment = frame.iloc[left:right].copy()
        if right < len(frame):
            segment = segment.loc[segment["future_end"] < frame.index[right]].copy()
        if len(segment) < sequence_length + 2:
            raise ValueError(f"partition {name} has too few rows after purging for sequence_length={sequence_length}")
        segment.attrs.update(partition=name, sequence_length=int(sequence_length))
        parts[name] = segment
    return parts


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return a float64 three-class probability matrix or fail loudly."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("probabilities must have nonempty shape (n, 3)")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("probability rows must sum to one")
    return values / values.sum(axis=1, keepdims=True)


def _validated_labels(labels: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows or not np.isin(values, (0, 1, 2)).all():
        raise ValueError("labels must be a matching one-dimensional array of 0, 1, 2")
    return values.astype(np.int64)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply one calibration temperature identically in evaluation and serving."""
    values = validate_probabilities(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit only the supplied calibration rows; never reads other partitions.

    A bounded scalar temperature minimizes calibration log loss. Temperature
    one is retained if the numerical optimizer cannot improve the baseline.
    Supplying calibration rows (never holdout rows) is the runner's contract.
    """
    from scipy.optimize import minimize_scalar

    values = validate_probabilities(probabilities)
    targets = _validated_labels(labels, len(values))

    def loss(log_temperature: float) -> float:
        calibrated = apply_temperature(values, float(np.exp(log_temperature)))
        return float(-np.log(np.clip(calibrated[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())

    result = minimize_scalar(loss, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-5})
    return float(np.exp(result.x)) if result.success and result.fun < loss(0.0) else 1.0


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Score a fixed three-class contract, including missing-class partitions."""
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    values = validate_probabilities(probabilities)
    targets = _validated_labels(labels, len(values))
    predicted = values.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "log_loss": float(log_loss(targets, values, labels=[0, 1, 2])),
        "macro_f1": float(f1_score(targets, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
    }


def calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Describe calibration rows using the canonical reliability contract.

    The score is calibrated top-label confidence and the binary outcome is
    whether that predicted class was correct. Ten fixed, equal-width bins
    retain counts; empty bins report zero rates and count zero. The Brier
    score uses this same binary correctness event, not multiclass Brier.
    Persisted temperature and policy threshold are also the serving inputs.
    """
    from src.experiment.schemas import CalibrationResult

    values = apply_temperature(probabilities, temperature)
    targets = _validated_labels(labels, len(values))
    if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    confidence = values.max(axis=1)
    correct = (values.argmax(axis=1) == targets).astype(float)
    bins = np.minimum((confidence * 10).astype(int), 9)
    counts = np.bincount(bins, minlength=10)
    denominator = np.maximum(counts, 1)
    predicted = np.bincount(bins, weights=confidence, minlength=10) / denominator
    observed = np.bincount(bins, weights=correct, minlength=10) / denominator
    result = CalibrationResult(
        method="temperature",
        n_samples=len(values),
        bucket_edges=tuple(float(x) for x in np.linspace(0.0, 1.0, 11)),
        predicted_rate=tuple(float(x) for x in predicted),
        observed_rate=tuple(float(x) for x in observed),
        brier_score=float(np.mean((confidence - correct) ** 2)),
    )
    return {
        **asdict(result),
        "bucket_counts": counts.tolist(),
        "score_semantics": "top_label_confidence_vs_correctness",
        "temperature": float(temperature),
        "confidence_threshold": float(confidence_threshold),
    }


def _ledger_summary(ledger: TradeLedger) -> dict[str, Any]:
    return {
        name: None if isinstance(value, float) and not np.isfinite(value) else value
        for name, value in asdict(ledger).items()
        if name != "trades"
    }


def evaluate_synthetic_economics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    cost_config: str | Path = "config/execution_costs.yaml",
    confidence_threshold: float = 0.5,
    spread_points: float = 0.25,
) -> dict[str, Any]:
    """Evaluate non-overlapping synthetic positions using canonical economics.

    The existing serving decision policy selects directions using the frozen
    confidence threshold, including its tie behavior. Positions enter at the
    executable side of a synthetic bid/ask quote and exit at the fixed horizon.
    Flat decisions abstain. The ledger prices MNQ-equivalent units
    using the repository fee profile, retaining synthetic provenance. The
    sensitivity scenarios double commission or add 0.25 point adverse
    slippage per side. This is an integration check, not market evidence.
    """
    from src.serving.predict import direction_from_proba

    values = validate_probabilities(probabilities)
    if len(frame) != len(values):
        raise ValueError("evaluation frame and probabilities must have matching rows")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("evaluation timestamps must be timezone-aware and strictly increasing")
    if not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    if not np.isfinite(spread_points) or spread_points < 0:
        raise ValueError("spread_points must be finite and nonnegative")
    required = {"price", "future_price", "future_end"}
    if not required.issubset(frame.columns):
        raise ValueError(f"evaluation frame requires {sorted(required)}")
    if not np.isfinite(frame[["price", "future_price"]].to_numpy(dtype=float)).all():
        raise ValueError("evaluation prices must be finite")
    future_end = pd.DatetimeIndex(frame["future_end"])
    if future_end.tz is None or future_end.hasnans or np.any(future_end.asi8 <= frame.index.asi8):
        raise ValueError("evaluation future_end must follow every decision")

    assumptions = load_assumptions(
        cost_config,
        "MNQ_equivalent",
        label="base",
        fill_model=f"synthetic fixed-horizon executable quotes; spread={spread_points} points",
    )
    assumptions = replace(assumptions, underlying="SYNTHETIC_NQ_SCALE")
    trades: list[Trade] = []
    next_entry: pd.Timestamp | None = None
    predicted, _ = direction_from_proba(values[:, 0], values[:, 1], values[:, 2], confidence_threshold)
    prices = frame["price"].to_numpy(dtype=float)
    future_prices = frame["future_price"].to_numpy(dtype=float)
    half_spread = spread_points / 2.0
    for row in range(len(frame)):
        timestamp = frame.index[row]
        if predicted[row] == "flat":
            continue
        if next_entry is not None and timestamp < next_entry:
            continue
        exit_time = future_end[row]
        trades.append(
            Trade(
                trade_id=f"synthetic-{len(trades):06d}",
                direction=-1 if predicted[row] == "down" else 1,
                entry=Leg(
                    OrderType.MARKET,
                    LegRole.ENTRY,
                    quote=Quote(prices[row] - half_spread, prices[row] + half_spread),
                    ts=timestamp.isoformat(),
                    fill_source="synthetic_quote",
                ),
                exit=Leg(
                    OrderType.MARKET,
                    LegRole.EXIT_TIME,
                    quote=Quote(future_prices[row] - half_spread, future_prices[row] + half_spread),
                    ts=exit_time.isoformat(),
                    fill_source="synthetic_quote",
                ),
                sequence=float(row),
            )
        )
        next_entry = exit_time
    scenarios = (
        assumptions,
        replace(assumptions, commission_per_trade=assumptions.commission_per_trade * 2, label="double_commission"),
        replace(assumptions, slippage_points_per_side=assumptions.slippage_points_per_side + 0.25, label="adverse_slippage"),
    )
    ledgers = evaluate_cost_sensitivity(trades, scenarios)
    baseline = ledgers["base"]
    return {
        "synthetic_only": True,
        "summary": _ledger_summary(baseline),
        "ledger": [asdict(trade) for trade in baseline.trades],
        "assumptions": asdict(assumptions),
        "sensitivity": {name: _ledger_summary(ledger) for name, ledger in ledgers.items()},
    }
