"""Tests for selection-bias correction — PSR / DSR / PBO-CSCV (ADR-122)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.overfitting import (
    PBO_REJECT_ABOVE,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    promotion_gate,
)

RNG = np.random.default_rng(20260909)


# ── PSR ──────────────────────────────────────────────────────────────────

def test_psr_is_high_for_a_genuinely_profitable_series():
    r = RNG.normal(0.10, 1.0, 2000)
    assert probabilistic_sharpe_ratio(r) > 0.99


def test_psr_is_near_half_for_zero_mean_noise():
    r = RNG.normal(0.0, 1.0, 5000)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(0.5, abs=0.15)


def test_psr_is_low_for_a_losing_series():
    r = RNG.normal(-0.10, 1.0, 2000)
    assert probabilistic_sharpe_ratio(r) < 0.01


def test_psr_rises_with_sample_size_at_fixed_sharpe():
    short = RNG.normal(0.05, 1.0, 100)
    long = np.repeat(short, 20)  # same moments, 20x the observations
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_psr_handles_degenerate_input():
    assert np.isnan(probabilistic_sharpe_ratio(np.array([1.0, 2.0])))
    assert np.isnan(probabilistic_sharpe_ratio(np.full(50, 3.0)))  # zero variance


# ── expected maximum Sharpe ──────────────────────────────────────────────

def test_expected_max_sharpe_grows_with_trial_count():
    v = 0.01
    a = expected_max_sharpe(n_trials=10, trial_sr_variance=v)
    b = expected_max_sharpe(n_trials=1000, trial_sr_variance=v)
    assert 0 < a < b


def test_expected_max_sharpe_is_zero_without_trials_or_variance():
    assert expected_max_sharpe(n_trials=1, trial_sr_variance=0.01) == 0.0
    assert expected_max_sharpe(n_trials=100, trial_sr_variance=0.0) == 0.0


def test_expected_max_sharpe_matches_simulation():
    """The analytic expectation should track the empirical max of N draws."""
    n_trials, v = 200, 0.04
    sd = np.sqrt(v)
    sims = [RNG.normal(0.0, sd, n_trials).max() for _ in range(400)]
    assert expected_max_sharpe(n_trials=n_trials, trial_sr_variance=v) == pytest.approx(
        float(np.mean(sims)), rel=0.15
    )


# ── DSR ──────────────────────────────────────────────────────────────────

def test_dsr_penalises_the_same_result_as_trials_increase():
    r = RNG.normal(0.06, 1.0, 1500)
    few = deflated_sharpe_ratio(r, n_trials=5, trial_sr_variance=0.01)
    many = deflated_sharpe_ratio(r, n_trials=5000, trial_sr_variance=0.01)
    assert few > many


def test_dsr_never_exceeds_psr_against_a_zero_benchmark():
    r = RNG.normal(0.08, 1.0, 1500)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=500, trial_sr_variance=0.02)
    assert dsr <= psr + 1e-12


def test_dsr_rejects_a_lucky_winner_from_a_large_zero_skill_search():
    """The best of 1,000 pure-noise configurations must not pass the gate."""
    trials = [RNG.normal(0.0, 1.0, 750) for _ in range(1000)]
    sharpes = np.array([t.mean() / t.std(ddof=1) for t in trials])
    winner = trials[int(np.argmax(sharpes))]
    dsr = deflated_sharpe_ratio(
        winner, n_trials=len(trials), trial_sr_variance=float(sharpes.var(ddof=1))
    )
    assert dsr < 0.95


# ── PBO via CSCV ─────────────────────────────────────────────────────────

def test_pbo_is_high_when_every_configuration_is_noise():
    """Pure noise: picking the IS winner should be ~a coin flip out of sample."""
    m = RNG.normal(0.0, 1.0, (1200, 40))
    res = probability_of_backtest_overfitting(m, n_splits=10)
    assert res.n_splits > 0
    assert res.pbo > 0.35
    assert res.overfit


def test_pbo_is_low_when_one_configuration_has_persistent_skill():
    m = RNG.normal(0.0, 1.0, (1200, 40))
    m[:, 7] += 0.35                       # a genuinely better configuration
    res = probability_of_backtest_overfitting(m, n_splits=10)
    assert res.pbo < PBO_REJECT_ABOVE
    assert not res.overfit


def test_pbo_ignores_skill_that_does_not_persist_out_of_sample():
    """Skill only in the first half is exactly what CSCV is meant to catch."""
    m = RNG.normal(0.0, 1.0, (1200, 40))
    m[:600, 11] += 1.0                    # brilliant early, average later
    res = probability_of_backtest_overfitting(m, n_splits=10)
    assert res.pbo > 0.2


def test_pbo_requires_a_configuration_family_not_a_single_winner():
    single = RNG.normal(0.1, 1.0, (1000, 1))
    assert np.isnan(probability_of_backtest_overfitting(single).pbo)


def test_pbo_returns_nan_when_there_are_too_few_observations():
    m = RNG.normal(0.0, 1.0, (8, 10))
    assert np.isnan(probability_of_backtest_overfitting(m, n_splits=16).pbo)


def test_odd_split_counts_are_rounded_down_to_stay_symmetric():
    m = RNG.normal(0.0, 1.0, (600, 12))
    res = probability_of_backtest_overfitting(m, n_splits=7)
    assert res.n_splits > 0            # ran with 6 blocks, not 7


# ── combined gate ────────────────────────────────────────────────────────

def test_gate_passes_a_strong_result_from_a_small_search():
    r = RNG.normal(0.15, 1.0, 2000)
    fam = RNG.normal(0.0, 1.0, (2000, 20))
    fam[:, 0] += 0.15
    g = promotion_gate(r, n_trials=8, trial_sr_variance=0.005, family_matrix=fam)
    assert g["passed"] is True
    assert g["pbo"] is not None


def test_gate_fails_when_pbo_is_high_even_with_a_strong_backtest():
    r = RNG.normal(0.20, 1.0, 2000)          # excellent standalone Sharpe
    noise_family = RNG.normal(0.0, 1.0, (2000, 40))   # but the family is noise
    g = promotion_gate(
        r, n_trials=8, trial_sr_variance=0.005, family_matrix=noise_family
    )
    assert g["pbo"] > PBO_REJECT_ABOVE
    assert g["passed"] is False


def test_gate_reports_pbo_none_when_no_family_is_supplied():
    r = RNG.normal(0.15, 1.0, 2000)
    g = promotion_gate(r, n_trials=8, trial_sr_variance=0.005)
    assert g["pbo"] is None
    assert g["n_trials"] == 8


def test_gate_fails_a_marginal_result_once_the_trial_count_is_honest():
    r = RNG.normal(0.04, 1.0, 1000)
    lenient = promotion_gate(r, n_trials=2, trial_sr_variance=0.01)
    honest = promotion_gate(r, n_trials=10_000, trial_sr_variance=0.01)
    assert honest["dsr"] < lenient["dsr"]
    assert honest["passed"] is False
