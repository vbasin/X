"""IV solver contract: NaN for anything that did not actually solve.

Guards the fix for the leak where ``_implied_vol_vec`` returned its 0.3 seed (or a
half-converged Newton iterate) as though it were an implied vol. On 0DTE days where
ADR-151's one-hour DTE error collapsed vega this reached 24% of rows in
``databento.nq_option_surface_1s``, all of them the literal value 0.3000.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from src.features.options_greeks import _implied_vol_vec_S
from src.features.options_surface import _implied_vol_vec

R = 0.05
SEED = 0.3  # the solver's initial guess — must never be returned as a solution


def _bs(S, K, T, sigma, is_call, r=R):
    """Black-Scholes price on a future, vectorised, for building round-trip inputs."""
    sq = np.sqrt(T)
    sign = np.where(is_call, 1.0, -1.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    return sign * (S * ndtr(sign * d1) - K * np.exp(-r * T) * ndtr(sign * d2))


def _sample(n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    S = 20_000.0
    K = np.round(rng.uniform(18_000, 22_000, n) / 25) * 25
    T = rng.uniform(1 / (365.25 * 24 * 60), 0.25, n)   # 1 minute .. 3 months
    sigma = rng.uniform(0.05, 1.5, n)
    is_call = rng.random(n) < 0.5
    return S, K, T, sigma, is_call, _bs(S, K, T, sigma, is_call)


def test_round_trip_recovers_sigma_within_sigma_tol():
    S, K, T, sigma, is_call, mid = _sample()
    iv = _implied_vol_vec(mid, S, K, T, R, is_call)
    ok = np.isfinite(iv)
    assert ok.mean() > 0.95, f"only {ok.mean():.1%} of well-posed quotes inverted"
    # sigma_tol is the solver's advertised identifiability bound, not a fudge factor.
    assert np.abs(iv[ok] - sigma[ok]).max() <= 1e-3


def test_never_returns_the_seed_as_a_solution():
    S, K, T, sigma, is_call, mid = _sample()
    iv = _implied_vol_vec(mid, S, K, T, R, is_call)
    # A true sigma of exactly 0.3 is not in the sample, so any 0.3 is the seed leaking.
    assert not np.any(iv[np.isfinite(iv)] == SEED)


def test_degenerate_quotes_are_nan_not_the_seed():
    """Zero/negative/absurd mids and zero tenor carry no vol information."""
    n = 6
    mid = np.array([0.0, -1.0, 1e-12, 1e12, 5.0, 5.0])
    K = np.full(n, 20_000.0)
    T = np.array([0.01, 0.01, 0.01, 0.01, 0.0, -1.0])
    iv = _implied_vol_vec(mid, 20_000.0, K, T, R, np.full(n, True))
    assert np.all(np.isnan(iv))


def test_deep_itm_quote_is_rejected_not_guessed():
    """A deep ITM call is nearly vol-insensitive: almost any sigma reprices it.

    The residual check alone passes such a row, so the solver must reject it on
    identifiability (vega) instead of reporting whatever sigma it happened to hold.
    """
    K = np.array([5_000.0])           # S = 20000 -> ~15000 points in the money
    T = np.array([1.0 / (365.25 * 24)])   # one hour
    mid = _bs(20_000.0, K, T, np.array([0.3]), np.array([True]))
    iv = _implied_vol_vec(mid, 20_000.0, K, T, R, np.array([True]))
    assert np.isnan(iv[0])


def test_scalar_and_per_row_underlying_agree():
    """``_implied_vol_vec_S`` is now an alias — the two paths cannot drift."""
    S, K, T, sigma, is_call, mid = _sample(n=5_000)
    a = _implied_vol_vec(mid, S, K, T, R, is_call)
    b = _implied_vol_vec_S(mid, np.full(K.size, S), K, T, R, is_call)
    assert np.allclose(a, b, equal_nan=True)


def test_sigma_tol_is_a_first_order_error_budget():
    """Tightening ``sigma_tol`` must buy accuracy and cost coverage, monotonically.

    It is a first-order budget (error ~ tol/vega from a local linearisation), so it
    binds tightly when small and degrades when loosened — assert that shape rather
    than treating it as a hard bound.
    """
    S, K, T, sigma, is_call, mid = _sample(n=20_000)
    errs, counts = [], []
    for st in (1e-3, 1e-2, 1e-1):
        iv = _implied_vol_vec(mid, S, K, T, R, is_call, sigma_tol=st)
        ok = np.isfinite(iv)
        errs.append(np.abs(iv[ok] - sigma[ok]).max())
        counts.append(int(ok.sum()))
    assert errs[0] <= 1e-3                      # holds at the default
    assert errs[0] < errs[1] < errs[2]          # looser budget -> more error
    assert counts[0] < counts[1] < counts[2]    # ... and more rows admitted
