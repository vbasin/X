# ADR-091: Native 1h / 1d HTF-context feature parquet

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs 1h and 1d as **context**, not as
predictive bases. Hive dirs exist (ADR-070) empty. 15s / 1m / 5m files
are on disk. The remaining parquet gap was blocking walk-forward OOF
meta. Tick-level TBBO rebuild (the 5m path) is unnecessary here:
`databento.tbbo_1h` is ~18k rows and `tbbo_1d` ~1k rows.

Requested product: higher-timeframe **trend / regime**, **volatility
regime**, and **prior retracement / support / resistance**, light enough
to finish in one pass so the stack can be tested.

## Decision

1. **Cagg source, not ticks.** `src/features/native_tf_htf.py` reads
   `databento.tbbo_1h` / `tbbo_1d`, keeps lead-month NQ (max-volume
   fallback on a missing lead), and writes monthly hive parts. No
   volume-profile / footprint / tick VWAP.

2. **Closed-bar timestamp.** `bar_end_ts` matches 15s/1m/5m: 1h
   left-closed `[T, T+1h)` labeled `T+1h`. 1d Globex
   `[T-1 17:00, T 17:00)` CT is labeled at the **next 17:00 CT** (DST
   23/24/25 UTC hours), not UTC+24h.

3. **Same 45 columns on both stores** (`FEATURE_HTF_COLS`):

   | Prefix | Contents |
   |--------|----------|
   | `cs_` | OHLCV + range + CLV + Wilder ATR(14) |
   | `tr_` | EMA8/21/50 dist, EMA stack, slope 10/20 / ATR, gap |
   | `ps_` | Pine SAR trend / next-dist / flip |
   | `zz_` | EXT ZigZagHighLow factory (signal, state, swing H/L dist) |
   | `rv_` | RV 5/10/20, vol-of-vol 10, RV ratio, ATR(5)/ATR(20) |
   | `sr_` | Donchian 20/50, prior-period H/L/C, floor pivots, Fib 38.2/50/61.8 of current ZZ swing |

4. **Gap / pivots differ by TF.** 1h `tr_gap` is close vs Globex session
   open; 1h pivots and `sr_prior_*` use the **prior Globex session**.
   1d `tr_gap` is open vs prior close; 1d `sr_prior_*` is the prior
   session; 1d floor pivots use a **rolling 5-session** H/L + last close.

5. **Windows are native bars.** 20 hours ≈ one Globex session; 20 days ≈
   one month. Do not keep 5s wall-clock windows.

6. **Monthly parts** `features_{tf}_partN.parquet` under
   `timeframe={1h|1d}/symbol=NQ/year=/month=`. Amends ADR-070 for 1h/1d.

7. **Catalogs.** [FEATURE_CATALOG_1h.md](../reference/FEATURE_CATALOG_1h.md),
   [FEATURE_CATALOG_1d.md](../reference/FEATURE_CATALOG_1d.md). Custom Test
   stores `1h` / `1d`, views `native_1h` / `native_1d`.

Rejected: rebuilding 1h/1d from TBBO ticks; copying the 5m VP/VF/VWAP
set; vendor calendar-day 1d (ADR-067 already chose Globex session).

## Consequences

- Automated search `tf_oof_meta` is still fail-closed; OOF-only composer:
  [ADR-092](ADR-092-tf-oof-meta-composer.md). Custom Test may pick store
  `1h` or `1d` with meta off; the YAML bases stay 15s/1m/5m.
- GPU prefixes `timeframe=1h/` and `timeframe=1d/` are already in
  `oci_sync.conf`. Building the hive is a **CPU** cagg job.
- Live pin unchanged.
- **Amended [ADR-095](ADR-095-regime-context-fire-aux.md):** the 45-col
  hive is unchanged. `nq-50pt-econ-lg-st` consumers (bases + meta) asof
  a 13-col `regime_v1` pack (vol / trend / prior-session location), not
  raw OHLC / Fib / pivots. The same pack is joined onto 15s/1m/5m
  frames so predictive models see HTF, not only the meta.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — consumer objective
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — cagg substrate
- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys
- [ADR-090](ADR-090-native-tf-5m-features.md) — 5m sibling (tick path)
- Builder: `scripts/features/build_native_tf_htf.py`
