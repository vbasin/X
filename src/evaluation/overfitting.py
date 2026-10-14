"""Selection-bias correction for strategy evaluation (ADR-122, Phase B).

Implements the Bailey / Borwein / López de Prado / Zhu family of corrections:

* :func:`probabilistic_sharpe_ratio` — PSR, the probability that an observed
  Sharpe exceeds a benchmark, corrected for skew and kurtosis
* :func:`expected_max_sharpe` — the Sharpe you expect from the best of ``N``
  independent trials when every trial has **zero** true skill
* :func:`deflated_sharpe_ratio` — DSR, PSR benchmarked against that expected
  maximum, so it discounts explicitly for the number of configurations tried
* :func:`probability_of_backtest_overfitting` — PBO via Combinatorially
  Symmetric Cross-Validation (CSCV): the probability that the in-sample-best
  configuration ranks below the out-of-sample median

Why this module exists: purging and embargo (already implemented in
:mod:`src.validation.walk_forward`) control leakage *within* a single
evaluation. They do nothing about selection bias *across* evaluations. With
enough trials a strategy whose true Sharpe is zero will show an arbitrarily
high backtest Sharpe purely by selection, and an automated search loop scales
that number of trials by orders of magnitude.

Reference: ``research_cases/papers/ssrn2326253_bailey2015_pbo_cscv.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329

# Reject a candidate whose selection procedure lands below the OOS median more
# often than not — the procedure is then worse than a coin flip.
PBO_REJECT_ABOVE = 0.5
# Standard DSR confidence: the observed Sharpe must beat the expected maximum
# under the null with at least this probability.
DSR_ACCEPT_BELOW = 0.95


def _moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    """Sharpe (per-observation), skew, kurtosis, n — for finite values only."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan"), float("nan"), float("nan"), n
    sr = float(r.mean()) / sd
    z = (r - r.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    return sr, skew, kurt, n


def probabilistic_sharpe_ratio(
    returns: np.ndarray, *, benchmark_sr: float = 0.0,
) -> float:
    """P(true Sharpe > ``benchmark_sr``), corrected for skew and kurtosis.

    ``benchmark_sr`` and the observed Sharpe are both **per-observation** — pass
    returns at the frequency the benchmark is expressed in.
    """
    sr, skew, kurt, n = _moments(returns)
    if not math.isfinite(sr) or n < 3:
        return float("nan")
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return float("nan")
    stat = (sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm.cdf(stat))


def expected_max_sharpe(*, n_trials: int, trial_sr_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` zero-skill trials.

    This is the benchmark the Deflated Sharpe Ratio tests against. It rises with
    the number of configurations tried, which is precisely the penalty an
    automated search loop must pay.
    """
    n = int(n_trials)
    v = float(trial_sr_variance)
    if n < 2 or not math.isfinite(v) or v <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / n)
    b = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(v) * ((1.0 - g) * a + g * b))


def deflated_sharpe_ratio(
    returns: np.ndarray, *, n_trials: int, trial_sr_variance: float,
) -> float:
    """PSR benchmarked against :func:`expected_max_sharpe`.

    ``trial_sr_variance`` is the variance of the Sharpe ratios observed across
    the trial family — read it from the trial ledger, never guess it.
    """
    sr_star = expected_max_sharpe(
        n_trials=n_trials, trial_sr_variance=trial_sr_variance
    )
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star)


@dataclass(frozen=True)
class PBOResult:
    """Outcome of a CSCV run."""

    pbo: float
    n_splits: int
    logits: np.ndarray

    @property
    def overfit(self) -> bool:
        return bool(math.isfinite(self.pbo) and self.pbo > PBO_REJECT_ABOVE)


def probability_of_backtest_overfitting(
    matrix: np.ndarray, *, n_splits: int = 16,
) -> PBOResult:
    """PBO via Combinatorially Symmetric Cross-Validation.

    ``matrix`` is ``(T observations, N configurations)`` of per-period returns
    for every configuration in the family that was searched — not just the
    winner. Passing only the winner makes the statistic meaningless.

    The rows are partitioned into ``n_splits`` contiguous blocks; every balanced
    combination of half the blocks forms an in-sample set with its complement as
    out-of-sample. For each split the in-sample-best configuration is located
    and its **relative rank out of sample** recorded. PBO is the fraction of
    splits where that rank falls at or below the median.
    """
    m = np.asarray(matrix, dtype="float64")
    if m.ndim != 2 or m.shape[1] < 2:
        return PBOResult(float("nan"), 0, np.array([]))
    s = int(n_splits)
    if s % 2:
        s -= 1
    t, n_cfg = m.shape
    if s < 2 or t < s * 2:
        return PBOResult(float("nan"), 0, np.array([]))

    blocks = np.array_split(np.arange(t), s)
    logits: list[float] = []
    for combo in combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(s) if i not in combo])
        is_perf = _sharpe_columns(m[is_rows])
        oos_perf = _sharpe_columns(m[oos_rows])
        if not np.isfinite(is_perf).any() or not np.isfinite(oos_perf).any():
            continue
        best = int(np.nanargmax(is_perf))
        finite = np.isfinite(oos_perf)
        if not finite[best] or finite.sum() < 2:
            continue
        # Relative rank of the IS winner within the OOS results.
        rank = float((oos_perf[finite] <= oos_perf[best]).sum()) / float(finite.sum() + 1)
        rank = min(max(rank, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        return PBOResult(float("nan"), 0, np.array([]))
    arr = np.asarray(logits, dtype="float64")
    return PBOResult(float((arr <= 0).mean()), len(arr), arr)


def _sharpe_columns(block: np.ndarray) -> np.ndarray:
    """Per-observation Sharpe for each column of ``block``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(block, axis=0)
        sd = np.nanstd(block, axis=0, ddof=1)
        out = np.where(sd > 0, mu / sd, np.nan)
    return out


def promotion_gate(
    returns: np.ndarray,
    *,
    n_trials: int,
    trial_sr_variance: float,
    family_matrix: np.ndarray | None = None,
    n_splits: int = 16,
) -> dict[str, float | bool | None]:
    """Combined DSR + PBO verdict for a promotion candidate.

    ``passed`` requires DSR ≥ 0.95 **and**, when a family matrix is supplied,
    PBO ≤ 0.5. A missing family matrix leaves ``pbo`` as ``None`` and the
    verdict rests on DSR alone — record that in the promotion note, because the
    selection procedure itself then goes unmeasured.
    """
    dsr = deflated_sharpe_ratio(
        returns, n_trials=n_trials, trial_sr_variance=trial_sr_variance
    )
    psr0 = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    pbo: float | None = None
    if family_matrix is not None:
        pbo = probability_of_backtest_overfitting(
            family_matrix, n_splits=n_splits
        ).pbo

    ok_dsr = bool(math.isfinite(dsr) and dsr >= DSR_ACCEPT_BELOW)
    ok_pbo = True if pbo is None else bool(
        math.isfinite(pbo) and pbo <= PBO_REJECT_ABOVE
    )
    return {
        "psr": psr0,
        "dsr": dsr,
        "pbo": pbo,
        "n_trials": int(n_trials),
        "passed": ok_dsr and ok_pbo,
    }
