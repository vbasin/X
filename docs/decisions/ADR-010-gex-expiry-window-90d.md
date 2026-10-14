# ADR-010: GEX Expiry Selection — 90-Day Window Replaces Nearest-Monthly

**Date:** 2026-05-05  
**Status:** Accepted  
**Phase:** 17H  
**Supersedes:** none  
**Cross-references:** [FEATURE_CATALOG.md](../reference/FEATURE_CATALOG.md#gx_-dealer-gex), [dealer_gex.py](../../src/features/dealer_gex.py), [patch_gex_columns.py](../../scripts/patch_gex_columns.py)

---

## Context

`dealer_gex.py` estimates net dealer gamma exposure (GEX) from NQ options.
The expiry selection logic used a two-bucket approach:

```python
is_0dte = dte_f < 1 / 365.25           # same-day options
is_monthly_candidate = dte_f >= 20 / 365.25  # any expiry >= 20 DTE
nearest_monthly_exp = monthly_exps.min()     # pick the nearest one
sel_mask = is_0dte | is_monthly
```

**Bug:** `nearest_monthly_exp = min(expiries >= 20 DTE)` selects the *nearest*
qualifying expiry — in October 2025 this resolves to Oct 27–28 weekly chains
(24–25 DTE), **not** the Dec 19 quarterly (77 DTE), despite the quarterly
being far more liquid.

BBO density for a 30-second window at session open on 2025-10-03:

| Chain | DTE | Instruments | BBO rows / 30s |
|---|---|---|---|
| 0DTE (Oct 3) | 0 | 341C / 342P | 5,456 / 5,276 |
| Oct 27 weekly *(old "monthly")* | 24 | 5C / 1P | 80 / 6 |
| Oct 28 weekly *(old "monthly")* | 25 | 12C / 7P | 327 / 180 |
| **Dec 19 quarterly *(excluded)*** | **77** | **42C / 77P** | **10,964 / 14,808** |

The Dec 19 quarterly has **~100× more rows** than the selected weeklies.
It represents genuine institutional dealer hedging (large fund rolls,
covered-call overwriting, portfolio protection) while the thin weeklies
carry negligible open interest.

Gamma magnitude comparison at NQ ≈ 19,800, σ = 0.30:
- 0DTE ATM gamma ≈ 0.00296 per point² (highest — drives intraday pinning)
- Dec 19 ATM gamma ≈ 0.000138 per point² (~21× lower per unit OI)

Even at 21× lower gamma-per-contract, the quarterly's 100× higher OI means
it contributes ~5× more to total GEX than the thin weeklies. The old code
therefore systematically *underweighted* the dominant structural hedging chain.

## Decision

Replace the two-bucket expiry selection with a single **0 < DTE ≤ 90 days**
window:

```python
max_dte = 90.0 / 365.25
sel_mask = (dte_f > 0) & (dte_f <= max_dte)
```

### Why 90 days?

- **0DTE options**: dte_f ≈ 0.0001–0.003 y → inside window ✓
- **Dec 19 quarterly from Oct 3**: 77 DTE → inside window ✓
- **Mar 20, 2026 from Dec 22–31**: 79–88 DTE → inside window ✓ (covers roll-off)
- **Mar 20, 2026 from Oct 3**: 168 DTE → **outside** window ✓ (far-dated, negligible ATM gamma)
- Thin weeklies (10–27 DTE): inside window, but negligible OI → low gamma weight, does not distort total

The 90-day ceiling is chosen to include the front quarterly for the entire
Oct–Dec 2025 dataset while excluding longer-dated contracts whose gamma
contribution at current strikes is negligible.

### What changes

| Before | After |
|---|---|
| 0DTE + single nearest weekly ≥ 20 DTE | 0DTE + all expiries ≤ 90 DTE |
| Dec 19 quarterly **excluded** | Dec 19 quarterly **included** |
| Oct 27–28 weeklies (dominant by selection, negligible by OI) | All weeklies (natural OI weighting) |
| Front-quarterly GEX signal absent | Front-quarterly GEX signal captured |

### What does NOT change

- Moneyness prefilter `|log(K/S)| < 0.15` — unchanged
- OI proxy (bid_sz + ask_sz when open_interest unavailable) — unchanged
- Constant IV proxy `iv_const = 0.30` — unchanged (see note below)
- Contract multiplier `$20/point` — unchanged
- 30s internal cadence → ffill to 5s — unchanged
- Column names and count: still 4 columns (`gx_dealer_gex`, `gx_gex_per_strike_max`, `gx_gex_flip_dist`, `gx_gex_chg_24`)

### Note on constant IV

Using `iv_const = 0.30` (flat volatility) is a known approximation. Using
actual per-strike IV from `options_surface.py` (now fixed in Phase 17H)
would improve gamma accuracy, especially for skewed IV surfaces. However,
this would require joining bbo_1s + IV surface per 30s bar, significantly
increasing memory and compute. Constant IV is acceptable for the current
feature set. A future ADR may revisit this.

## Consequences

- **Parquet patch required**: All 62 existing parquets need gx_ columns
  recomputed via `scripts/patch_gex_columns.py`. Same OOM routing as iv_
  patch (Nov 21, Dec 15–18 serial; others parallel with 4 workers).
- **GEX values will change**: Both the level (`gx_dealer_gex`) and the flip
  distance (`gx_gex_flip_dist`) will shift because the quarterly options have
  different strike concentrations from the thin weeklies. Expected direction:
  gx_dealer_gex magnitude increases (more OI included); flip levels move to
  reflect quarterly strike clustering (typically at round-number intervals
  like 19000, 19500, 20000 for Dec contracts).
- **All checkpoints invalidated**: Any model trained on gx_ features before
  this patch should be retrained. This coincides with the iv_ and _close
  patches already planned under Phase 17H.
- **Run order**: patch_iv_columns.py → patch_gex_columns.py → patch_close_5s.py → retrain

## Alternatives Considered

**0DTE + front quarterly only (OI-argmax)**: Select the expiry with the most
instruments in the ≥ 30 DTE range. Rejected because it requires an extra
groupby per 30s bar, is more complex, and the 90-day window achieves the same
result naturally (the quarterly dominates by OI weight).

**0DTE only**: Much simpler but loses the structural dealer book that drives
multi-day support/resistance levels. Rejected.

**All expiries (no upper bound)**: Theoretically correct but includes back
quarterly contracts (168+ DTE) with gamma so small that they contribute noise
not signal. The 90-day ceiling provides a principled cutoff.
