# ADR-083: Native 15s RTH CVD, cumulative OFI, 5-bar slopes, 5-bar ATR

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

The native 15s hive (ADR-075) stored only 1-bar tape statistics. Dual-binary
bases on `nq-50pt-econ-lg-st` need session-level signed flow and short-horizon
slope/volatility of those series. Consecutive 15s `of_ofi` values are already
additive tick-sums (ADR-068); the hive did not persist signed volume
(`of_delta`), a cash-session cumulative, or a 5-bar relative slope.

## Decision

1. **`of_delta`** — 1-bar signed print volume, same identity as 5s
   `order_flow.of_delta`: `Σ size × (+1 if side=B, −1 if side=A, 0 else)`.
   Defined on every Globex 15s bar that has prints.

2. **RTH window** (America/Chicago, CST/CDT — not fixed UTC−6):
   - First included closed bar: `bar_end_ts = 08:30:00` CT
     (`[08:29:45, 08:30:00)` — the last fully closed 15s bar as cash open
     prints).
   - Last included closed bar: `bar_end_ts = 15:00:00` CT
     (`[14:59:45, 15:00:00)`).
   - **`of_cvd`** = running sum of `of_delta` on that window.
   - **`of_ofi_cum`** = running sum of `of_ofi` on that window.
   - Both **reset each Globex session**. Outside the window they are **NaN**
     (not a frozen last value, not 0).

3. **5-bar relative slope** on parquet rows (5 printed bars, grouped by
   Globex session so a weekend gap does not leak):

   `(v[t] − v[t−5]) / abs(v[t−5])`

   Zero or missing denominator → NaN.
   - `of_ofi_cum_slope_5` applies this to `of_ofi_cum` (NaN outside RTH).
   - `cs_return_slope_5` applies this to 1-bar `cs_return` on all Globex
     bars of the session (not RTH-clipped).

4. **`cs_atr_5`** — SMA of the last 5 15s true ranges
   `max(H−L, |H−prev_C|, |L−prev_C|)`. First bar of a session uses H−L
   only. **Not** Wilder RMA and **not** 5s-catalog `cs_atr_5m`. High/low
   come from the 15s OHLC (or recovered from `cs_close`/`cs_range`/`cs_clv`
   on a hive patch).

5. **Builder / patch.** `compute_bar_features` emits the 1-bar columns plus
   the ADR-083 set. Existing hive:
   `python scripts/features/patch_15s_rth_flow.py` (TBBO fill for
   `of_delta` if missing, then attach + rewrite month parts). Full rebuild
   remains `build_native_tf_15s.py`. GPU `timeframe=15s/` mirror must be
   re-synced after the hive rewrite (ADR-078).

## Consequences

- Catalog v1.1 17 → **v1.2 23**. `of_` 1 → 5; `cs_` 4 → 6.
- Amends ADR-075 “1-bar only” for these five extra columns; 1-bar tape
  identities are unchanged.
- `native_15s` view picks the new `of_` / `cs_` names up by prefix.
- Live pin unchanged.

## Related

- [ADR-068](ADR-068-tbbo-event-ofi.md) — tick OFI formula (additive per bar)
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s 1-bar hive
- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) — 10m Wilder ATR
  target geometry (distinct from `cs_atr_5`)
- [FEATURE_CATALOG_15s.md](../reference/FEATURE_CATALOG_15s.md)
- `src/features/bar_features.py` (`attach_rth_session_features`)
- `scripts/features/patch_15s_rth_flow.py`
