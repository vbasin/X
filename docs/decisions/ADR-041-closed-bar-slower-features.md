# ADR-041: Closed-bar visibility for slower-than-5s features

## Status
Accepted — implemented 2026-08-21.

## Date
2026-08-21

## Context

A Custom Test on `cs_` prefixes showed `cs_5m_clv` / `cs_10m_clv` (rewritten
with `label="right"`) near-zero SHAP while `cs_15m_clv` dominated. Registry
champion `xgb_tbm_topgroups_20260715T234308Z` uses the same 15m/30m columns.

Investigation of `features_*.parquet` vs 1-min candles found three leaks of
the same class: a slower bucket’s **final** OHLC was visible from the **start**
of that bucket (then ffilled).

1. **15m/30m CS** — original `resample(tf)` used pandas default `label="left"`.
   Code now uses `label="right"`; most historical files were never rewritten.
2. **Opening range** — `time_context` aggregated the full 5/15/30-minute OR
   once per day and broadcast it to every bar, including 08:00 CT premarket.
3. **1-min family** — `tbbo_1min` `time_bucket` is left (`T` = `[T, T+1m)`).
   Engine ffill from `T` let 5s bars inside the minute see the completed
   minute (up to 55s). `iv_vix_close` already delayed +1 minute; `cs_clv` /
   `so_*` / `ca_*` / `ht_ytd_*` did not.

Wavelets use `log_prices[i-window:i]` (exclude current 1-min bar) and are
**not** in this leak. 5s tick features and `iv_*`/`gx_*` 30s `(t-30s, t]`
windows are also out of scope.

A full parquet rebuild would rewrite ~200 unchanged columns (options/GEX/OOM).
A blind 12-bar shift would mis-lag `wv_*` and already-right `cs_5m_clv`.

## Decision

1. **Closed-bar rule.** A bucket’s OHLC is visible only at/after that bucket’s
   close, then stale-forward. Helper: `src/features/closed_bars.py`.
2. **15m/30m CS** stay `label="right"` + ffill. Patch rewrites stored columns.
3. **Opening range** is a **running** max/min of **closed** 1-min candles in
   `[RTH open, min(t, open+dur))`. NaN before 08:30 CT. Freeze after the
   window. Not “wait until 08:35 to show anything.”
4. **1-min CS / stochastic / cross-asset / YTD** shift the left-labeled 1-min
   index +1 minute (same as `iv_vix_close`) then ffill onto 5s. Do not change
   `tbbo_1min` DDL.
5. **Live** drops the incomplete last 1-min bucket from TBBO-synthesized
   candles so Live matches batch.
6. **Retroactive** `--modules causal` in `patch_parquets.py` (1-min NQ/ES
   only). CPU GPU-dispatch **pushes** parquet to object storage before
   enqueue; GPU pull uses size **and** LastModified.

Existing registry weights trained on leaked 15m/1-min CS are **not** refit
in this change. Causal replica measured in
[ADR-046](ADR-046-causal-tbm-topgroups-proof.md): same prefixes on patched
parquet fail the economics gate (3-class NE 6.85 / HR 0.247 vs leaky 26.9 /
0.605). Per-candle volume footprint (`vf_`) uses the same closed-bar rule —
[ADR-056](ADR-056-volume-footprint-wick-body.md) (column set; module from
[ADR-054](ADR-054-volume-candle-footprint.md)).

## Consequences

- Premarket `tc_or*_high/low` become NaN until the first closed OR minute
  (08:31 CT for OR5).
- `cs_clv` at `T+25s` equals the **previous** 1-min CLV, not the forming
  minute.
- Fold cache rebuilds (parquet size/mtime change). GPU jobs see patched
  files only after push+pull.

## Supersedes

None. Complements [ADR-001](ADR-001-5s-timestep.md) (5s closed bars live)
and the v5.4 `label="right"` 5m/10m CLV patch.
