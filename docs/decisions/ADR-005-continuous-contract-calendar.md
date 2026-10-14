# ADR-005: CME Calendar-Based Contract Selection

## Status
Accepted

## Date
2025-12-15 (Phase 15B — introduced for roll handling)

---

## Context

NQ futures are quarterly contracts (Mar/Jun/Sep/Dec expiry). The feature engine and training pipeline need to know which contract is "active" (front-month, most liquid) on any given trading day. This matters for:

1. **Feature computation**: `tbbo` and `bbo_1s` queries must filter to the correct NQ contract. Mixing NQZ5 and NQH6 prices in the same feature window produces artificial discontinuities.
2. **Back-adjusted `ht_*` features**: The higher-timeframe module needs a continuous price series across roll boundaries. Without back-adjustment, the Dec 15, 2025 roll from NQZ5 to NQH6 introduced a ~50-point artificial step in `ht_daily_trend_*` features.
3. **Parquet metadata**: Each file records `_meta_nq_contract` for audit purposes.

**Original approach (v4)**: The engine queried `tbbo` without contract filtering — it mixed NQZ5 and NQH6 prices in the 35-day lookback at the roll date. This caused `ht_daily_trend_10` and other features to see a spurious 50-point trend reversal at Dec 15.

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Hardcoded roll date table | Simple | Must be manually updated every quarter; error-prone |
| **Algorithmic CME calendar** (chosen) | Self-updating; correct by construction | Requires implementing the CME roll rule |
| Query DB for highest-volume contract | Self-calibrating | Requires additional DB query; can fail near roll when both contracts trade actively |
| Always use the first contract in definitions | Works if sorted correctly | Fragile to definition record ordering |

---

## Decision

Implement an **algorithmic CME calendar** in `src/common/contract_calendar.py`.

CME roll rule: **Monday prior to the third Friday of the expiry month**. This is the date when volume migrates from the expiring front month to the next contract. Implemented as:

```python
def roll_date(expiry_month_year) -> date:
    third_friday = nth_weekday_of_month(expiry_month_year, weekday=FRIDAY, n=3)
    return third_friday - timedelta(days=4)  # preceding Monday

def active_contract(symbol: str, d: date) -> str:
    # Returns e.g. "NQZ5", "NQH6" based on roll_date comparison
```

Back-adjustment for `ht_*` features:
```python
def back_adjusted_ohlc(symbol: str, start: date, end: date) -> pd.DataFrame:
    # 1. Identify contract segments in [start, end]
    # 2. At each roll boundary: delta = new_close(D-1) - old_close(D-1)
    # 3. Apply additive delta to all prior segments (Panama method)
```

Roll date reference (algorithmically computed — no table needed):
- NQZ5: active until Dec 15, 2025
- NQH6: active Dec 15, 2025 → Mar 16, 2026
- NQM6: active Mar 16, 2026 → Jun 15, 2026

---

## Consequences

- **+** Zero maintenance — correct for any future roll date
- **+** `ht_*` features no longer show artificial steps at roll boundaries
- **+** `_meta_nq_contract` in Parquet is accurate for every file
- **+** The `cme_holidays.py` module (25-date frozenset) complements this for trading-day logic
- **−** The CME roll rule ("Monday before third Friday") is correct for NQ/ES quarterly contracts but may not apply to all futures. Do not reuse this code for other asset classes without verifying their roll convention.
- **Constraint**: The 4-day offset (Monday = Thursday expiry minus 4 = the Monday prior) is hardcoded. This matches historical CME practice but should be verified if CME ever changes its roll convention.

Implementation: `src/common/contract_calendar.py` `roll_date()`, `active_contract()`, `active_contracts_in_range()`, `back_adjusted_ohlc_range()`
