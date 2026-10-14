# ADR-003: Vectorised IV/GEX Solver

## Status
Accepted

## Date
2026-04-18 (Phase 15 — v3 first training cycle)

---

## Context

The options IV surface module (`iv_*`, 13 features) and dealer GEX module (`gx_*`, 4 features) require computing Black-Scholes implied volatility and gamma for all NQ options on each trading day. A typical day has ~400 options in the quarterly universe (196 NQZ5 instruments on 2025-10-15) and will have significantly more once 0DTE weekly definitions are ingested.

The original Python implementation used a row-by-row loop with `scipy.optimize.brentq` for each (strike, expiry, mid_price) tuple. On a day with 400 options and one IV computation per 30-second bar, this required ~200 Brent iterations × 400 options × 360 time steps = ~29M function evaluations per day. At ~0.05ms per call, runtime was ~50 minutes/day — longer than the entire rest of the feature engine combined.

---

## Options Considered

| Option | Estimated runtime | Pros | Cons |
|--------|-----------------|------|------|
| Row-by-row `scipy.brentq` | ~50 min/day | Simple | Completely unusable at 5s resolution |
| Vectorised Newton-Raphson (chosen) | **<1 min/day** | Batch numpy, natural GPU extension | Requires implementing N-R manually; needs convergence tuning |
| Per-day cache (compute once, reuse) | ~50 min for first run, 0ms after | No code change | Cache invalidates on any price change; stale during live streaming |
| Numba JIT | ~5 min/day | Avoids reimplementation | Requires numba dependency, JIT warmup, harder to maintain |
| `py_vollib` / `mibian` | ~30 min/day | Library handles math | Still row-by-row internally; doesn't vectorise over option space |

---

## Decision

Implement a **vectorised Newton-Raphson** IV solver in `options_surface.py` operating on full numpy arrays, combined with a moneyness pre-filter and a 30-second compute cadence.

Key design choices:
1. **Batch N-R**: Compute IV for all options simultaneously via numpy broadcasting. 30 iterations, tolerance=1e-5. Converges in <10 iterations for normally-priced options.
2. **Moneyness prefilter**: Exclude options with `|log(K/S)| > 0.15` (~80% of the chain per snapshot). Deep OTM options have near-zero IV sensitivity and are rarely ATM — including them wastes compute.
3. **30-second cadence**: IV computation runs every 30 bars (150 seconds), then results are forward-filled to the 5s feature grid. The Black-Scholes surface does not change materially in <30 seconds.
4. **Expiry window**: Load options with `expiration ≤ d + 400 days` to include all 4 quarterly expirations.

Same vectorised Black-Scholes gamma approach applies to `dealer_gex.py`.

---

## Consequences

- **+** Runtime: ~50 min → <1 min/day (50× speedup). Total feature engine runtime: ~5m13s/day.
- **+** Same vectorised pattern extends cleanly to 0DTE weekly options (thousands of additional instruments per expiry day)
- **+** N-R in numpy is straightforward to test and audit
- **−** Forward-fill introduces up to 30s of staleness in IV/GEX features. Acceptable given that the prediction horizon is 5–30 minutes.
- **−** Deep ITM options are excluded by moneyness filter — not a concern for normal trading conditions
- **Constraint**: The 30s cadence and 0.15 moneyness cutoff are hardcoded constants in `options_surface.py` and `dealer_gex.py`. Change these only with performance impact analysis.

Implementation: `src/features/options_surface.py` `_implied_vol_vec()`, `src/features/dealer_gex.py`
