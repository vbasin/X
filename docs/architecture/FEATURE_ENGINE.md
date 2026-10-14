# Feature Engine

> **Present**: v5.15 5s candle OHLCV on `cs_` (`cs_open`…`cs_return`, ADR-069); `cs_close` = `_close`. Canonical 5s cache is monthly parts under `timeframe=5s/` (`bar_end_ts`, ≤100 MB/part, ADR-079 / ADR-081). `build_date` / `patch_date` / `extend_date` and live_runtime write those parts. Native 15s files (ADR-075 / ADR-083): 14 1-bar tape + 5 RTH/5-bar flow (CVD, cum OFI, 5-bar slopes, 5-bar ATR) + 4 head-private target-geometry columns (ADR-082), Globex hours, `bar_end_ts`, monthly `features_15s_partN.parquet`. Fold cache for `nq-50pt-econ-lg-st` clips **08:30–11:00 CT** and, on the **5m** store only, attaches 11 `svp_*` columns from Globex 17:00 CT session POC / 70% VA + break/reclaim ([ADR-099](../decisions/ADR-099-lg-st-morning-onset-session-vp.md)). Native 1m files (ADR-085): 71 cols `so_`/`cs_`/`ps_`/`vw_`/`rv_`/`vp_`/`ms_` (no options IV), Globex hours, `features_1m_partN.parquet`. Native 5m files (ADR-090): 53 cols `cs_`/`vp_`/`vw_`/`ps_`/`zz_`/`vf_`/`rv_` (VWAP ±2σ bands; Wilder ATR(5) on the 5m bar; EXT ZigZag), Globex hours, `features_5m_partN.parquet`. Native 1h / 1d HTF-context files (ADR-091): 45 cols `cs_`/`tr_`/`ps_`/`zz_`/`rv_`/`sr_` from `tbbo_1h` / Globex `tbbo_1d`; `nq-50pt-econ-lg-st` consumes a 13-col `regime_v1` pack on 15s/1m/5m bases and meta ([ADR-095](../decisions/ADR-095-regime-context-fire-aux.md)). GPU jobs mirror `timeframe={tf}/` (ADR-078 / ADR-081).  
> **Supersedes**: v5.14 tick-level OFI (`of_ofi`, ADR-068).  
> **Next**: overlay `vf_` / `CandleFootprint` on Live Chart. Serving asof for `tf_oof_meta` (ADR-093 joint search exists; live pin unchanged). Live Chart CVD origin + Volume Profile display overlays are ADR-097 / ADR-108 (VP SSOT; 110 / 113 / 115 folded); display ADR-109; large prints ADR-111; stacked panes ADR-112; replay ADR-114; multi-EMA ADR-132; Level Blend VP ADR-135 (not hive columns).  
> **Source of truth for**: engine design, module orchestration, session configuration, known limitations  
> **Current state**: v5.15 — 25 modules, 256 features (`cs_` 33 including 5s OHLCV; `of_` 17; `mbp_` 4; `ps_` 4; `vf_` 20; `tl_` 16; `so_` 6), ~0.5–1.5% typical NaN (mbp all-NaN outside the L2 window); research full-day ~minutes; live extend appends new 5s rows only

---

## 1. Engine Overview

`src/features/engine.py` — coordinates all feature modules and merges each session day into `timeframe=5s/…/features_5s_partN.parquet` (`persist_session_day`, ADR-081). Live `extend_date` appends new 5s rows into the same month parts.

```python
# Public API
engine.build_date(target_date)     # full day replace (research / --rebuild)
engine.patch_date(target_date)     # default CLI/nightly: fill missing file/rows/NaNs
engine.extend_date(target_date)    # live: append only newly closed 5s bars
```

### Nightly patch (`patch_date`) — 2026-08-15

`vtech-feature-builder` (`python -m src.features.engine --date=today`) **patches** the day file:

1. **Missing file** → full `build_date` for the widest window (engine hours ∪ live objective ∪ existing span).
2. **Missing 5s rows** on that grid → compute the hole (suffix-only uses a lookback window) and merge.
3. **NaN cells** → recompute and `combine_first` so only NaNs are filled.
4. **Finite existing values are kept** (live 08:30–14:45 is not replaced by an 08:00–11:00 research rebuild).

Force a replace with `--rebuild`.

### Full day (`build_date`)

1. Resolve primary NQ and ES contract (`contract_calendar.active_contract()`)
2. Load raw data from TimescaleDB (loaders — see §3). `_query` streams a server-side cursor and compacts each batch; it does not `fetchall` a full options-BBO day into Python tuples.
3. Build 5s timestamp grid over training hours
4. Call each module → concatenate columns
5. Reindex `ONE_MIN_MODULES` outputs to 5s grid with ffill (v5 fix — was 83% NaN before)
6. Apply `VTECH_BBO_PRIMARY_ONLY` filter for OOM dates when set
7. Write `features_YYYY-MM-DD.parquet` + manifest sidecar (atomic replace)

### Always-on live runtime (`live_runtime`) — 2026-08-14

Service: `vtech-live-features` → `python -m src.features.live_runtime`

1. Long-lived process keeps warm TBBO / futures BBO / options BBO / candle buffers.
2. On each closed 5s bar: **delta-load** new ticks only, trim lookback, recompute **full** feature schema on warm frames, **append** new rows only.
3. Scores via `build_snapshot` and writes heartbeat `ops/webui/live_features_heartbeat.json`.
4. WebUI reads the hive and does not rebuild while the heartbeat is fresh (`is_runtime_fresh()`).
5. **cs_*/wv_***: 1m modules are ffilled onto the 5s grid through `data_end`; live runtime builds 1m OHLCV from warm TBBO so the trailing edge is not all-NaN.

Fallback: if the runtime is down during the session, WebUI calls `extend_date`. That is a windowed recompute when the hive day already has rows. A full `build_date` runs only when that day is empty, or when `force_rebuild` is set. Loading the whole options-BBO day into the web UI process is what OOM-killed the API mid chart request (2026-09-30).

### Live incremental (`extend_date`) — 2026-08-14

Used by the WebUI Live worker while the objective session is open and the live-features heartbeat is stale:

1. If the day parquet is missing/empty → one full `build_date` (bootstrap). Any lag, including multi-hour, is a windowed recompute from `last_existing − 2h`. Prior rows are not rebuilt.
2. Else if already at the last **closed** 5s bar → no-op.
3. Else load a **lookback window** of raw ticks (~2h, primary-filtered options BBO), recompute the **full feature module set** (~189 cols — same as `build_date`, including GEX/order-flow/etc.), **keep prior parquet rows unchanged**, and **append** only bars with `ts > last_existing` and `ts <= last_closed`.
4. Atomic write of the merged day file.

Prior rows are never recomputed on the live path. New rows get the **full feature population**, not the 43-column serving subset (that subset is only what the live model *scores*). Disk still rewrites the day file for Parquet practicality; the expensive part removed is re-deriving historical rows.

---

## 1.0 5s hive monthly parts (ADR-079)

Canonical 5s research/training cache:

`sources/features/timeframe=5s/symbol=NQ/year=/month=/features_5s_partN.parquet`

Datetime column `bar_end_ts` (UTC, first column). Duplicate timestamps keep
the latest-mtime source row. Builder (does not delete day files):

`python scripts/features/migrate_parquet_5s_hive.py`

## 1.1 Native 15s 1-bar hive (ADR-075)

Separate from the 5s day-file engine. `src/features/bar_features.py` computes
14 1-bar columns at `timestep="15s"` from lead-month TBBO over the full Globex
session, plus ADR-083 RTH CVD / cum OFI (08:30–15:00 CT) and 5-bar
`of_ofi_cum` / `cs_return` slopes and 5-bar ATR, plus four causal
target-geometry columns (`lg_target_v_atr` /
`lg_target_v_close` / `st_target_v_atr` / `st_target_v_close`: ±50 vs 10m
ATR and last print, ADR-082). `scripts/features/build_native_tf_15s.py` writes
them on a full-concat attach; existing hives:
`python scripts/features/patch_15s_rth_flow.py` (flow) and
`python scripts/features/patch_15s_target_geometry.py` (geometry).
`load_feature_frame("15s")` attaches geometry if missing, and RTH flow when
`of_delta` is already on the frame. Dual-binary LSTM concatenates
`lg_*` only onto the long trunk and `st_*` only onto the short trunk.

`sources/features/timeframe=15s/symbol=NQ/year=/month=/features_15s_partN.parquet`

with datetime column `bar_end_ts`. Catalog:
[FEATURE_CATALOG_15s.md](../reference/FEATURE_CATALOG_15s.md).
Custom Test (ADR-076) can train a single-base model on this store: pick
**15s** in the feature bank, **dual_binary** head, meta off. Fold cache
joins `parquet_labels_15s` on `ts` = `bar_end_ts`. GPU jobs push/pull the
hive root under `timeframe=15s/` plus that labels prefix ([ADR-078](../decisions/ADR-078-gpu-native-tf-hive-mirror.md)).
Sequence families write named `p_long` / `p_short` heads; diagnostics
are per-head, not 3-class softmax ([ADR-080](../decisions/ADR-080-dual-binary-sequence-heads.md)).

## 1.2 Native 1m hive (ADR-085)

Separate from the 5s day-file engine and from native 15s 1-bar tape.
`src/features/native_tf_1m.py` emits one closed Globex minute
(`bar_end_ts`) with the entire 5s-catalog groups `so_` `cs_` `ps_`
`vw_` `rv_` `vp_` `ms_` (71 columns; options `iv_*` not loaded).
`cs_open` / `cs_close` are the 1-minute candle. Builder:
`python scripts/features/build_native_tf_1m.py`.

`sources/features/timeframe=1m/symbol=NQ/year=/month=/features_1m_partN.parquet`

Catalog: [FEATURE_CATALOG_1m.md](../reference/FEATURE_CATALOG_1m.md).
Custom Test store **1m** / view **native_1m**.

## 1.3 Native 5m hive (ADR-090)

Separate from the 5s day-file engine and from native 15s / 1m.
`src/features/native_tf_5m.py` emits one closed Globex 5-minute bar
(`bar_end_ts`) with `cs_` (OHLCV + Wilder ATR(5) on this bar), `vp_`,
`vw_` (session VWAP + distance to ±2σ bands), `ps_` (SAR on 5m),
`zz_` (EXT ZigZagHighLow factory knobs), `vf_` (5m wick/body + candle
POC), `rv_` (53 columns). `cs_open` / `cs_close` are the 5-minute
candle. Builder: `python scripts/features/build_native_tf_5m.py`.

`sources/features/timeframe=5m/symbol=NQ/year=/month=/features_5m_partN.parquet`

Catalog: [FEATURE_CATALOG_5m.md](../reference/FEATURE_CATALOG_5m.md).
Custom Test store **5m** / view **native_5m**. Building this hive is a
CPU TBBO job; GPU only pulls `timeframe=5m/` when a job uses store `5m`.

## 1.4 Native 1h / 1d HTF context (ADR-091)

Hive is the 45-col HTF catalog. `nq-50pt-econ-lg-st` consumers asof a
slim `regime_v1` pack onto **15s/1m/5m bases and the meta** (ADR-095),
not the full 45 columns and not meta-only.
`src/features/native_tf_htf.py` reads `databento.tbbo_1h` / `tbbo_1d`
(lead-month NQ, no tick scan) and emits 45 columns: `cs_` (OHLCV +
ATR(14)), `tr_` (EMA stack / slope / gap), `ps_` (SAR), `zz_` (EXT
ZigZag), `rv_` (vol regime), `sr_` (Donchian, prior-period H/L/C, floor
pivots, Fib of last ZZ swing). 1h `tr_gap` is vs Globex session open;
1d `tr_gap` is overnight. 1h pivots use the prior Globex session; 1d
pivots use a rolling 5-session week. `bar_end_ts`: 1h = left hour + 1h;
1d = next Chicago 17:00 (DST-aware). Builder:
`python scripts/features/build_native_tf_htf.py`.

`sources/features/timeframe={1h|1d}/symbol=NQ/year=/month=/features_{tf}_partN.parquet`

Catalogs: [FEATURE_CATALOG_1h.md](../reference/FEATURE_CATALOG_1h.md),
[FEATURE_CATALOG_1d.md](../reference/FEATURE_CATALOG_1d.md).
Custom Test stores **1h** / **1d**, views **native_1h** / **native_1d**.

---

## 2. Session Configuration

| Parameter | Value | Config Key |
|-----------|-------|-----------|
| Timestep | **5 seconds** | `features.timestep` |
| Session start | **08:00 CST** (14:00 UTC) | `features.trading_hours_start` |
| Session end | **11:00 CST** (17:00 UTC) | `features.trading_hours_end` |
| **Market open** | **08:30 CST** (NYSE open) | `VTECH_MARKET_OPEN` env var |
| Bars per session | **1,080** | (3 hr × 720 bars/hr at 5s) |
| Sequence length | **128 bars** (10.7 min) | `features.sequence_length` |

> Overnight and premarket bars are excluded. All features are computed only over the defined 3-hour session. See [ADR-002](../decisions/ADR-002-08-11-session-filter.md). `trading_hours_start` (08:00) is the **data-window boundary** and is distinct from `market_open` (08:30, the CME RTH / NYSE open) — the latter is used as the zero-reference for all time-context features.

---

## 3. Data Loaders (8 total)

All loaders use `ts_recv` timestamp range for TimescaleDB chunk pruning:
```sql
WHERE ts_recv >= '2025-10-15 00:00:00+00'
  AND ts_recv <  '2025-10-16 00:00:00+00'
```

| Loader | Table | Filters | Used by |
|--------|-------|---------|---------|
| `_load_tbbo` | `tbbo` | NQ primary contract | of_, tl_, ss_, lt_, ta_, ms_, rv_, sd_, vp_, vf_, vw_, ca_, cs_15s/30s, cs_5s OHLCV, ps_lh/flip |
| `_load_mbp10_depth_bars` | `mbp_10` | NQ primary; `time_bucket` + `AVG` (no raw pull) | mbp_ |
| `_load_bbo_1s` | `bbo_1s` | Futures (instrument_class='F') + options (expiry ≤ d+400d) | bp_, iv_, gx_ |
| `_load_definitions` | `definitions` | NQ.OPT all C/P including 0DTE weekly (Q1A–Q5D, QN1–QN4) | iv_, gx_ |
| `_load_statistics` | `statistics` | NQ.FUT, 7-day lookback | dc_ |
| `_load_1min_bars` | `tbbo_1min` (cagg) | NQ primary, 35-day lookback | wv_, cs_, ht_, so_, atr, ps_ |
| `_load_daily_continuous_ohlc` | `tbbo_1min` (cagg) | NQ all segments, 35-day lookback, back-adjusted | ht_ |
| `_load_es_tbbo` | `tbbo` | ES primary contract | ca_ |

### BBO-1s OOM Filter (`VTECH_BBO_PRIMARY_ONLY=1`)

For 5 OOM-killed dates (Nov 21, Dec 15-18, 2025), `_load_bbo_1s` applies a stricter filter:
```sql
WHERE instrument_class = 'F'
   OR (expiration BETWEEN d AND d + INTERVAL '400 days'
       AND symbol >= 'NQ' AND symbol < 'NR')
```
Reduces row count from ~37M to ~30M on peak days. Set this env var before rebuilding those dates.

### Back-Adjusted Daily OHLC (v5 — `ht_` features)

`_load_daily_continuous_ohlc`:
1. `active_contracts_in_range("NQ", start, d)` identifies contract segments in the 35-day window
2. At each roll boundary: `delta = new_close(D−1) − old_close(D−1)`, applied additively to all OHLC in older segments
3. `ht_gap_pct` uses raw primary-contract close (not back-adjusted)
4. All other `ht_` features use the stitched series

Pre-v5: mixed NQZ5/NQH6 prices at Dec 15 roll caused ±50 pt artificial step in `ht_` features.

---

## 4. `bar_windows.py` — Central Window Helper

**File**: `src/common/bar_windows.py` | **Added**: Phase 12A (v3 architecture overhaul)

All 22 feature modules use this helper to convert time-based window specifications to bar counts. When `feature_timestep` changes, every rolling window automatically adjusts without touching module code.

```python
from src.common.bar_windows import bars, bars_m, get_timestep

bars(30)       # 30 seconds → 6 bars  @5s  (or 3 bars @10s)
bars(60)       # 60 seconds → 12 bars @5s
bars_m(5)      # 5 minutes  → 60 bars @5s  (or 30 bars @10s)
bars_m(10)     # 10 minutes → 120 bars @5s
bars_m(30)     # 30 minutes → 360 bars @5s
get_timestep() # returns "5s" (reads from config.yaml features.timestep)
```

`trade_location.py` follows this contract ([ADR-037](../decisions/ADR-037-size-weighted-tl-zscore-diff.md)):
size-weighted `tl_trade_location` and `tl_aggressive_pct` on the 5s grid, then
equal-weight MAs at 30s / 5m / 30m only (no 1m / 60m). Pair stats are 30s vs 5m
and 5m vs 30m: `ma_short − ma_long` and `(ma_short − ma_long) / std_long`, for
both location and aggression (16 columns).

### Why this was critical: the forward_windows bug

In v2 (`feature_timestep=10s`), `forward_windows: [5, 10, 15, 30]` were interpreted as bar counts, not minutes:
- 5 bars × 10s = **50 seconds** actual (intended: 5 minutes)
- 30 bars × 10s = **5 minutes** actual (intended: 30 minutes)

The model was predicting 50-point NQ moves within 50 seconds — catching flash crashes, not momentum moves. `bar_windows.py` was introduced in v3 so that `forward_windows: [60, 120, 180, 360]` are explicitly in bars, and all module windows are expressed in seconds/minutes and converted via `bars()`/`bars_m()`. This makes the intent of every window unambiguous regardless of timestep.

---

## 5. Module Orchestration

### 5.1 Execution Order

```python
# engine.py — simplified module call order
bp_features  = book_pressure.compute(bbo_fut_1s, ts_grid)
of_features  = order_flow.compute(tbbo, ts_grid)
tl_features  = trade_location.compute(tbbo, ts_grid)
ss_features  = same_side.compute(tbbo, ts_grid)
lt_features  = large_trade_cvd.compute(tbbo, ts_grid)
ta_features  = trade_arrival.compute(tbbo, ts_grid)
ms_features  = microstructure.compute(tbbo, ts_grid)
rv_features  = realized_vol.compute(tbbo, ts_grid)
sd_features  = sub_bar_dynamics.compute(bbo_fut_1s, tbbo, ts_grid)
vp_features  = volume_profile.compute(tbbo, ts_grid)
vf_features  = volume_footprint.compute(tbbo, ts_grid)  # 1m/5m wick/body, then ffill
iv_features  = options_surface.compute(bbo_opt_1s, defs, ts_grid)   # 30s cadence
gx_features  = dealer_gex.compute(bbo_opt_1s, defs, ts_grid)        # 30s cadence
dc_features  = daily_context.compute(stats, ts_grid)
wv_features  = wavelets.compute(bars_1min, ts_grid)     # ONE_MIN_MODULE
cs_features  = candle_structure.compute(bars_1min, ts_grid)  # ONE_MIN_MODULE
cs_multitf   = candle_structure.compute_multitimeframe(bars_1min)  # 5/10/15/30m
cs_submin    = candle_structure.compute_subminute_clv(tbbo, ts_grid)  # 15s/30s
cs_5s_ohlc   = candle_structure.compute_5s_ohlc(tbbo, grid, close=_close)  # after _close
vw_features  = vwap.compute(tbbo, ts_grid)
ca_features  = cross_asset.compute(es_tbbo, tbbo, ts_grid)
tc_features  = time_context.compute(ts_grid)
ec_features  = economic_calendar.compute(ts_grid)
hp_features  = hawkes_clustering.compute(tbbo, ts_grid)
ht_features  = higher_timeframe.compute(daily_ohlc, ts_grid)  # ONE_MIN_MODULE
ht_ytd       = higher_timeframe.compute_ytd(prior_hl, bars_1min, price, ts_grid)
atr_features = atr.compute(bars_1min, ts_grid)                # 5m/10m Wilder, stale-forward
so_features  = stochastic.compute(bars_1min)                  # ONE_MIN_MODULE
ps_features  = psar.compute(bars_1min, ts_grid, tbbo=tbbo)    # 1m trail + 5s pierce/flip
mbp_features = mbp_depth.compute(mbp_5s_avgs, ts_grid)       # SQL AVG of MBP-10 L0–L9
vp_yday      = volume_profile.compute_prior_session(prev_tbbo, price, ts_grid)
vix_feat     = vix.compute_vix(tt_candles_vix, ts_grid)       # public.tt_candles
```

### 5.2 ONE_MIN_MODULES ffill (v5 — Phase 17 fix)

Modules that operate on 1-minute candle data (`wv_`, `cs_`, `ht_`) produce output at 1-minute frequency. Closed 1-min / 5-min volume footprints (`vf_`) are right-labeled at candle close, then included in `_SLOW_FEATURE_PREFIXES` so the concat step stale-forwards them onto the 5-second grid. Parabolic SAR (`ps_`) self-aligns: 1m trend / next-dist are ffilled inside the module; `ps_lh_dist_sar` / `ps_flip` are 5s-native and must **not** be concat-ffilled. The eight ADR-069 5s OHLCV columns (`cs_open`…`cs_return`) are also 5s-native and are excluded from that ffill (empty bars are volume 0, not the previous bar's volume). The engine reindexes 1-min modules to the 5-second grid using `ffill` at the `pd.concat` step:

```python
ONE_MIN_MODULES = {"wv_", "cs_", "ht_"}
df = pd.concat([...all module outputs...], axis=1)
for prefix in ONE_MIN_MODULES:
    cols = [c for c in df.columns if c.startswith(prefix)]
    df[cols] = df[cols].ffill()
```

Pre-fix: ffill was applied per-module before concat but the 5s grid was created only at concat time → ~83% NaN on `wv_` and `cs_` columns. Fixed in v5.

### 5.3 IV/GEX 30-Second Cadence

`options_surface.py` and `dealer_gex.py` compute at 30-second intervals (not 5s) due to the cost of the IV solver and GEX computation. Results are forward-filled to the 5-second output grid.

IV solver specifics:
- Newton-Raphson, 30 iterations, tolerance=1e-5
- Moneyness pre-filter: `|log(K/S)| < 0.15` — removes ~80% deep OTM per snapshot
- Throughput: ~1,000 options in 2.2 ms → <1 min/day (was 50 min/day before vectorization)
- Expiry window: `d + timedelta(days=400)` — loads all 4 quarterly expirations

### 5.4 skip_existing Flag

```python
engine.build_date(date, skip_existing=True)  # skips if parquet already exists
engine.build_date(date, skip_existing=False) # force rebuild
```

The batch rebuild script (`scripts/build_cache_per_day.sh`) always uses `skip_existing=True` for safe resume.

---

## 6. Phase 17 Feature Fixes Summary

All fixes are applied in the current codebase. This table is for historical context:

| Feature | File | Root Cause | Fix |
|---------|------|-----------|-----|
| `tc_or{5,15,30}` (100% NaN) | `time_context.py` | Dictionary `.map()` keyed by date but index was datetime | Fixed to date-keyed dict |
| `cs_*` / `wv_*` (83% NaN) | `engine.py` | `ONE_MIN_MODULES` ffill happened before 5s grid was created | Moved ffill to post-concat |
| `gx_*` (100% NaN) | `dealer_gex.py` | tz-aware comparison `snap_f["expiration"].values ==` stripped tz → all-False | Changed to `.eq()` |
| `tl_aggressive_pct` (constant 1.0) | `trade_location.py` | `price >= ask_px` matched all trades due to tz/type mismatch | Direct sweep: `price > ask_px` (strict) |
| `tl_effective_spread` (removed) | `trade_location.py` | NQ runs at a near-constant 1-tick (0.25 pt) bid-ask spread during RTH; feature value ≈0.25 on >99% of bars with near-zero variance — redundant with `bp_spread_*` and `tl_aggressive_pct` | Feature dropped; `tl_` count 3 → 2, total features 184 → 183 |
| `dc_oi_chg` (constant 0.0) | `daily_context.py` | Diff computed on already-reindexed 5s series → zero diff at every tick | Diff at daily granularity, then ffill |
| `wv_approx_slope` (near-zero noise) | `wavelets.py` | Only fitted polyfit on last 2 coefficients | Changed to `np.polyfit` across all coefficients |
| `hp_cluster_ratio` (NaN on quiet days) | `hawkes_clustering.py` | 0/0 when no large moves → NaN | Added `fillna(0.0)` |
| `iv_0dte_atm` / `iv_term_slope` | `options_surface.py` | Missing 0DTE definitions + wrong DTE threshold | Fixed Phase 17D: true 0DTE filter (DTE < 2/365.25) + 0DTE vs quarterly term slope |
| `tc_minutes_into_session`, `tc_block_open_5m/15m`, `tc_or{5,15,30}_*`, `tc_session_progress` (wrong reference) | `time_context.py` | All open-anchored features used `trading_start` (08:00, the data-window start / pre-market) as the market-open reference instead of the actual CME RTH open | Added `market_open="08:30"` parameter; `tc_minutes_into_session` now −30→0 during pre-market; session blocks/OR anchor moved to 08:30; `tc_session_progress` denominator 180 min → 150 min |
| `vw_vwap_*` (wrong session reset) | `vwap.py` | Session VWAP accumulated from UTC midnight (≈7 PM CDT) via `+8h` shift trick; ad-hoc and DST-wrong for 3 PM CT close | Replaced with `market_session_date()` groupby — resets at 4 PM CT open / 3 PM CT close, DST-aware |
| `lt_cvd` (wrong session reset) | `large_trade_cvd.py` | Absolute CVD level accumulated from UTC midnight; absolute level was exported as a model feature, making it session-inconsistent | Replaced `.cumsum()` with `.groupby("session")["lt_delta"].cumsum()` using `market_session_date()`; same applied to internal `total_cvd` |

---

## 7. Feature Count by Module

| Module prefix | Count | One-min? | Options data? |
|--------------|-------|---------|--------------|
| `bp_` | 24 | — | — |
| `of_` | 17 | — | — |
| `tl_` | 16 | — | — |
| `ss_` | 3 | — | — |
| `lt_` | 4 | — | — |
| `ta_` | 3 | — | — |
| `ms_` | 7 | — | — |
| `rv_` | 6 | — | — |
| `sd_` | 8 | — | — |
| `vp_` | 5 | — | — |
| `iv_` | 13 | — | ✅ |
| `gx_` | 4 | — | ✅ |
| `dc_` | 4 | — | — |
| `wv_` | 14 | ✅ | — |
| `cs_` | 33 | ✅ 1m/slower + 5s OHLCV native | — |
| `vw_` | 5 | — | — |
| `ca_` | 12 | — | — |
| `tc_` | 22 | — | — |
| `ec_` | 4 | — | — |
| `hp_` | 3 | — | — |
| `ht_` | 5 | ✅ | — |
| `so_` | 6 | ✅ | — |
| `vf_` | 20 | ✅ 1m/5m closed | — |
| `ps_` | 4 | ✅ 1m trail + 5s pierce | — |
| `mbp_` | 4 | — | — (L2 MBP-10) |
| **TOTAL** | **256** | | |

---

## 8. Known Limitations

| Limitation | Impact | Planned Fix |
|-----------|--------|------------|
| `gx_dealer_gex` excludes weekly options | Understates GEX on expiry days | Phase 17C/D |
| `ca_*` stale after 2026-04-02 | Potentially stale cross-asset signals | Phase 17 Live — restart live streamer |
| 5 OOM dates: Nov 21, Dec 15-18, 2025 | Missing from current cache | Use `VTECH_BBO_PRIMARY_ONLY=1` to rebuild |
| ht_ features require cagg refresh before build | Stale if cagg not refreshed on build day | Engine calls `REFRESH MATERIALIZED VIEW` if `len(df)<10` |
| `ps_*` live 2h lookback re-inits SAR | Live 1m trail / `ps_lh_dist_sar` can differ from full-day parquet until a 1m reversal in-window | Pass 1-min from session start, or persist SAR state |
| `mbp_*` only on the 1-month MBP-10 window | All-NaN on older 5s parquet dates until more L2 is ingested | Patch `--modules mbp` on dates that have `databento.mbp_10` |

---

## Related Docs

- [reference/FEATURE_CATALOG.md](../reference/FEATURE_CATALOG.md) — full feature descriptions and sources
- [ADR-069](../decisions/ADR-069-cs-5s-ohlc.md) — 5s candle OHLCV on `cs_` (`cs_close` = `_close`)
- [ADR-068](../decisions/ADR-068-tbbo-event-ofi.md) — tick-level OFI on the 5s parquet (`of_ofi`)
- [ADR-057](../decisions/ADR-057-mbp10-5s-mean-depth-imbalance.md) — 5s mean MBP-10 L0–L9 depth imbalance (`mbp_`)
- [ADR-056](../decisions/ADR-056-volume-footprint-wick-body.md) — closed 1m/5m wick/body volume, delta, 1m POC EMA
- [ADR-054](../decisions/ADR-054-volume-candle-footprint.md) — prior HVN/gap `vf_` encoding (superseded)
- [ADR-055](../decisions/ADR-055-parabolic-sar-closed-bars.md) — closed-bar Parabolic SAR (`ps_`)
- [ADR-001](../decisions/ADR-001-5s-timestep.md) — 5s timestep rationale
- [ADR-003](../decisions/ADR-003-vectorised-iv-solver.md) — IV solver design
- [operations/CACHE_REBUILD.md](../operations/CACHE_REBUILD.md) — how to rebuild
- [operations/TROUBLESHOOTING.md](../operations/TROUBLESHOOTING.md) — NaN diagnostics
