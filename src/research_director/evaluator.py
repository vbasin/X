"""Evaluator / Adjudicator — the deterministic promotion gate (component, #13, W6).

Pure metric logic — **no GPU, no network, no DB**. Given a run's metrics and the baseline's
metrics it returns a verdict; the Director maps that verdict onto journal states. The
LLM only ever writes advisory commentary (W7) — it never sets the gate.

Metric contract (what the GPU runner must emit per run, W9):
  {
    "<primary metric name>": float,        # e.g. val_precision_at_tau on the test set
    "test_folds":   [float, ...],          # per-fold primary metric (walk-forward)
    "guardrails":   {"bd_precision": float, "calibration_error": float, ...},
    "n_signals":    {"BU": int, "BD": int},
    "test_window_hash": "sha256...",        # identical-split check vs baseline
  }
The baseline metrics use the same shape (read from training_result.json).

PASS requires ALL of (baseline-relative path):
  1. primary delta >= delta_min on the test set,
  2. guardrail non-regression (within GUARDRAIL_EPSILON, higher-is-better),
  3. multi-fold consistency: improvement on >= ceil(2/3) of folds,
  4. bootstrap 95% CI on the aggregate fold delta excludes 0.
Cold-start (no baseline): absolute floor — primary > prior_mult x class_prior AND
min(n_signals) >= min_signals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Gate outcomes
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

# Agent-facing verdicts (consumed by Journal.set_conclusion)
SUCCEEDED = "SUCCEEDED"
DISPROVEN = "DISPROVEN"
NEEDS_REFINEMENT = "NEEDS_REFINEMENT"


@dataclass
class EvalInput:
    metric: str
    delta_min: float
    run_primary: float | None
    baseline_primary: float | None              # None => cold start
    run_folds: list[float] = field(default_factory=list)
    baseline_folds: list[float] = field(default_factory=list)
    guardrails_run: dict[str, float] = field(default_factory=dict)
    guardrails_baseline: dict[str, float] = field(default_factory=dict)
    n_signals: dict[str, int] = field(default_factory=dict)
    class_prior: float = 0.09
    run_test_hash: str | None = None
    baseline_test_hash: str | None = None


@dataclass
class EvalVerdict:
    gate: str                        # PASS | FAIL | INCONCLUSIVE
    verdict: str                     # SUCCEEDED | DISPROVEN | NEEDS_REFINEMENT | INCONCLUSIVE
    reason: str
    primary_delta: float | None = None
    fold_pass: int = 0
    n_folds: int = 0


class Evaluator:
    def __init__(
        self,
        *,
        guardrail_epsilon: float = 0.01,
        fold_consistency_min_frac: float = 2 / 3,
        bootstrap_iters: int = 2000,
        bootstrap_seed: int = 42,
        cold_start_prior_mult: float = 2.0,
        cold_start_min_signals: int = 50,
    ) -> None:
        self._eps = guardrail_epsilon
        self._fold_frac = fold_consistency_min_frac
        self._boot_iters = bootstrap_iters
        self._boot_seed = bootstrap_seed
        self._prior_mult = cold_start_prior_mult
        self._min_signals = cold_start_min_signals

    # ── adapter from training_result-shaped dicts ──────────────────────────────
    def build_input(
        self,
        run: dict[str, Any],
        baseline: dict[str, Any] | None,
        *,
        metric: str,
        delta_min: float,
        class_prior: float = 0.09,
    ) -> EvalInput:
        def primary(d: dict[str, Any] | None) -> float | None:
            if not d:
                return None
            return d.get(metric, d.get("primary"))

        b = baseline or None
        return EvalInput(
            metric=metric,
            delta_min=delta_min,
            run_primary=primary(run),
            baseline_primary=primary(b),
            run_folds=list(run.get("test_folds", []) or []),
            baseline_folds=list((b or {}).get("test_folds", []) or []),
            guardrails_run=dict(run.get("guardrails", {}) or {}),
            guardrails_baseline=dict((b or {}).get("guardrails", {}) or {}),
            n_signals=dict(run.get("n_signals", {}) or {}),
            class_prior=class_prior,
            run_test_hash=run.get("test_window_hash"),
            baseline_test_hash=(b or {}).get("test_window_hash"),
        )

    # ── the gate ────────────────────────────────────────────────────────────────
    def evaluate(self, inp: EvalInput) -> EvalVerdict:
        # Cold start: no baseline to compare against → absolute floor.
        if inp.baseline_primary is None:
            return self._cold_start(inp)

        # Apples-to-apples: refuse to score a different test window.
        if inp.run_test_hash and inp.baseline_test_hash and inp.run_test_hash != inp.baseline_test_hash:
            return EvalVerdict(INCONCLUSIVE, "INCONCLUSIVE", "test-window hash mismatch vs baseline")
        if inp.run_primary is None:
            return EvalVerdict(INCONCLUSIVE, "INCONCLUSIVE", "missing run primary metric")

        delta = inp.run_primary - inp.baseline_primary
        fold_deltas = [r - b for r, b in zip(inp.run_folds, inp.baseline_folds)]
        n_folds = len(fold_deltas)
        fold_pass = sum(1 for d in fold_deltas if d >= inp.delta_min)
        fold_consistent = n_folds > 0 and fold_pass >= math.ceil(self._fold_frac * n_folds)
        guardrail_ok = all(
            inp.guardrails_run.get(k, 0.0) >= v - self._eps
            for k, v in inp.guardrails_baseline.items()
        )
        ci_low = self._bootstrap_ci_low(fold_deltas) if n_folds >= 2 else None
        ci_ok = ci_low is not None and ci_low > 0.0

        if delta >= inp.delta_min and fold_consistent and guardrail_ok and ci_ok:
            return EvalVerdict(
                PASS, SUCCEEDED,
                f"delta={delta:+.4f}>=δ, folds {fold_pass}/{n_folds}, guardrails ok, CI_low={ci_low:.4f}>0",
                primary_delta=delta, fold_pass=fold_pass, n_folds=n_folds,
            )

        # Failure classification: disproven (clearly wrong) vs needs-refinement (promising).
        wrong_sign_folds = sum(1 for d in fold_deltas if d < 0)
        disproven = (
            delta < 0
            and n_folds > 0
            and wrong_sign_folds >= math.ceil(self._fold_frac * n_folds)
        )
        if disproven:
            return EvalVerdict(
                FAIL, DISPROVEN,
                f"delta={delta:+.4f}<0 on {wrong_sign_folds}/{n_folds} folds — hypothesis disproven",
                primary_delta=delta, fold_pass=fold_pass, n_folds=n_folds,
            )
        return EvalVerdict(
            FAIL, NEEDS_REFINEMENT,
            f"delta={delta:+.4f}, folds {fold_pass}/{n_folds}, guardrails={guardrail_ok}, "
            f"CI_low={ci_low if ci_low is None else round(ci_low, 4)} — promising, needs tweaks",
            primary_delta=delta, fold_pass=fold_pass, n_folds=n_folds,
        )

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _cold_start(self, inp: EvalInput) -> EvalVerdict:
        if inp.run_primary is None or not inp.n_signals:
            return EvalVerdict(INCONCLUSIVE, "INCONCLUSIVE", "cold-start: missing run metrics")
        floor = self._prior_mult * inp.class_prior
        min_sig = min(inp.n_signals.values())
        if inp.run_primary > floor and min_sig >= self._min_signals:
            return EvalVerdict(
                PASS, SUCCEEDED,
                f"cold-start: primary={inp.run_primary:.4f}>{floor:.4f} and min_signals={min_sig}>={self._min_signals}",
                primary_delta=None,
            )
        return EvalVerdict(
            FAIL, NEEDS_REFINEMENT,
            f"cold-start floor not met: primary={inp.run_primary:.4f} vs {floor:.4f}, min_signals={min_sig}",
            primary_delta=None,
        )

    def _bootstrap_ci_low(self, fold_deltas: list[float], *, alpha: float = 0.05) -> float:
        """Lower bound of the (1-alpha) bootstrap CI on the mean fold delta (deterministic)."""
        rng = np.random.default_rng(self._boot_seed)
        arr = np.asarray(fold_deltas, dtype=float)
        means = arr[rng.integers(0, arr.size, size=(self._boot_iters, arr.size))].mean(axis=1)
        return float(np.percentile(means, 100 * (alpha / 2)))
