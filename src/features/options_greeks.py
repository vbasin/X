"""Full per-option Greeks (delta, gamma, vega, theta, IV) + a delta-bucket surface selector.

Durable, reusable options analytics over ``databento.bbo_1s`` + ``databento.definitions``,
extending the machinery already in :mod:`src.features.options_surface` (vectorised IV solve,
BS delta) and :mod:`src.features.dealer_gex` (BS gamma) to the full Greek set — for now and for
future research.

Two layers:
  * :func:`option_greeks` — vectorised Greeks for an arbitrary set of live option quotes at one
    (or per-row) underlying price (Black-Scholes on the future; r flat). Returns
    iv/delta/gamma/vega/theta + mid + spread per option.
  * :func:`delta_bucket_row` — the LSTM feature selector: from a one-second Greeks frame, pick the
    call AND put on the ÷25 strike grid whose |delta| is closest to each target {0.10,0.25,0.40,
    ATM≈0.50}, and emit their Greeks + mid + spread as a single fixed-width row (+ the NQ price).
    Columns are keyed by delta bucket, not strike, so the representation is stable as spot moves.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

# Reuse the vetted vectorised IV solver + BS delta from the surface builder.
from src.features.options_surface import _bs_delta_vec, _implied_vol_vec, _norm_pdf

_TARGET_DELTAS = (0.10, 0.25, 0.40, 0.50)   # 0.50 == ATM
_STRIKE_GRID = 25                            # liquid NQ option strike interval

# Compact surface schema constants (8 legs × 7 fields + nq_price).
# All-lowercase leg tags so Postgres unquoted identifiers match Python keys
# (cATM would fold to catm).
SURFACE_LEGS = ("c10", "p10", "c25", "p25", "c40", "p40", "catm", "patm")
SURFACE_FIELDS = ("iv", "delta", "gamma", "vega", "theta", "mid", "spread")
SURFACE_FEATURE_COLS = [f"{leg}_{fld}" for leg in SURFACE_LEGS for fld in SURFACE_FIELDS] + ["nq_price"]
SURFACE_META_COLS = ("ts", "expiry", "underlying_sym", "n_legs_filled")


def _d1(S, K, T, r, sigma):
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _implied_vol_vec_S(
    mid: np.ndarray, S: np.ndarray, K: np.ndarray, T: np.ndarray,
    r: float, is_call: np.ndarray,
    tol: float = 1e-5, max_iter: int = 30,
) -> np.ndarray:
    """Newton IV solver with per-row underlying ``S`` (broadcast-safe).

    Thin alias for :func:`src.features.options_surface._implied_vol_vec`, which
    now handles scalar and per-row ``S`` alike. It used to be a near-verbatim
    copy, which meant the convergence bug had to be fixed twice and the two
    could silently diverge; there is one solver now.
    """
    return _implied_vol_vec(mid, S, K, T, r, is_call, tol=tol, max_iter=max_iter)


def option_greeks(
    strike: np.ndarray, mid: np.ndarray, bid: np.ndarray, ask: np.ndarray,
    is_call: np.ndarray, dte_years: np.ndarray,
    underlying: float | np.ndarray, r: float = 0.05,
) -> dict[str, np.ndarray]:
    """Vectorised BS Greeks for a set of option quotes.

    ``underlying`` may be a scalar (one snapshot) or a per-row array (batch of
    seconds). Returns arrays: iv, delta, gamma, vega (per 1 vol pt), theta
    (per calendar day), mid, spread. NaN IV rows propagate NaN Greeks.
    """
    K = strike.astype(np.float64)
    T = np.maximum(dte_years.astype(np.float64), 1e-9)
    mid64 = mid.astype(np.float64)
    S_in = np.asarray(underlying, dtype=np.float64)

    if S_in.ndim == 0 or (S_in.size == 1 and np.ndim(underlying) == 0):
        S = float(np.asarray(S_in).reshape(-1)[0])
        iv = _implied_vol_vec(mid64, S, K, T, r, is_call)
        delta = _bs_delta_vec(S, K, T, r, iv, is_call)
        d1 = _d1(S, K, T, r, iv)
        pdf = _norm_pdf(d1)
        sqrtT = np.sqrt(T)
        gamma = pdf / (S * np.maximum(iv, 1e-9) * sqrtT)
        vega = (S * pdf * sqrtT) / 100.0
        d2 = d1 - iv * sqrtT
        disc = np.exp(-r * T)
        term1 = -(S * pdf * iv) / (2.0 * sqrtT)
        theta = np.where(
            is_call,
            term1 - r * K * disc * ndtr(d2),
            term1 + r * K * disc * ndtr(-d2),
        ) / 365.25
    else:
        S = np.broadcast_to(S_in.astype(np.float64), K.shape).copy()
        iv = _implied_vol_vec_S(mid64, S, K, T, r, is_call)
        # delta / d1 with per-row S
        sqrtT = np.sqrt(T)
        sigma_safe = np.maximum(iv, 1e-9)
        d1 = (np.log(np.where((S > 0) & (K > 0), S / K, 1.0))
              + (r + 0.5 * sigma_safe ** 2) * T) / (sigma_safe * sqrtT)
        delta = np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)
        pdf = _norm_pdf(d1)
        gamma = pdf / (S * sigma_safe * sqrtT)
        vega = (S * pdf * sqrtT) / 100.0
        d2 = d1 - iv * sqrtT
        disc = np.exp(-r * T)
        term1 = -(S * pdf * iv) / (2.0 * sqrtT)
        theta = np.where(
            is_call,
            term1 - r * K * disc * ndtr(d2),
            term1 + r * K * disc * ndtr(-d2),
        ) / 365.25
        # Invalidate where IV failed
        bad = ~np.isfinite(iv)
        for arr in (delta, gamma, vega, theta):
            arr[bad] = np.nan

    return {
        "iv": iv, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
        "mid": mid64, "spread": (ask - bid).astype(np.float64),
    }


def delta_bucket_row(g, nq_price: float, *, targets=_TARGET_DELTAS, strike_grid=_STRIKE_GRID):
    """Select the call/put nearest each target delta on the ÷grid strike lattice → one LSTM row.

    ``g`` is a DataFrame for ONE timestamp (single front expiry) with columns
    strike_price, is_call, iv, delta, gamma, vega, theta, mid, spread. Returns a flat dict:
    ``{c10_iv, c10_delta, ..., pATM_spread, nq_price}`` — fixed width, keyed by bucket not
    strike. Missing legs (no strike near a target) yield NaNs so the width stays constant.
    """
    fields = SURFACE_FIELDS
    row: dict[str, float] = {"nq_price": float(nq_price)}
    grid = g[(g["strike_price"].astype("int64") % strike_grid == 0) & g["iv"].notna()]
    calls = grid[grid["is_call"]]
    puts = grid[~grid["is_call"]]
    for tgt in targets:
        tag = "atm" if tgt >= 0.5 else f"{int(tgt * 100):02d}"
        for side, pool, sign in (("c", calls, +1.0), ("p", puts, -1.0)):
            key = f"{side}{tag}"
            if len(pool):
                pick = pool.iloc[(pool["delta"] - sign * tgt).abs().argmin()]
                for f in fields:
                    row[f"{key}_{f}"] = float(pick[f])
            else:
                for f in fields:
                    row[f"{key}_{f}"] = float("nan")
    return row


def n_legs_filled(row: dict) -> int:
    """Count how many of the 8 surface legs have a finite IV."""
    return sum(1 for leg in SURFACE_LEGS if np.isfinite(row.get(f"{leg}_iv", float("nan"))))
