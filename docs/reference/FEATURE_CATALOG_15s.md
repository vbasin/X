# Feature Catalog — Native 15s (1-bar)

> **Current version**: v1.2 (2026-08-28)
> **Total**: 23 features across 7 modules
> **Bar width**: 15 seconds (one output row = one bar)
> **Typical NaN rate**: ~0% on 1-bar tape (`cs_clv` NaN on a flat bar; `tl_*` NaN when the spread is 0). RTH cumulatives are NaN outside 08:30–15:00 CT; 5-bar slope/ATR NaN until 5 prior session bars (and when the slope denominator is 0)
> **Coverage**: every Globex interval that has lead-month NQ TBBO — not the 5s 08:00–11:00 CT clip
> **Store**: `$VTECH_DATA_DIR/sources/features/timeframe=15s/symbol=NQ/year=/month=/features_15s_partN.parquet`
> **ADR**: [ADR-075](../decisions/ADR-075-native-tf-15s-bar-features.md), [ADR-083](../decisions/ADR-083-15s-rth-cvd-ofi-slope-atr.md)

This catalog is the native-15s product. The 5s research matrix remains
[FEATURE_CATALOG.md](FEATURE_CATALOG.md) (256 columns, 08:00–11:00 CT). Names
are reused (`of_ofi`, `cs_close`, …) but the **bar is 15s**, so values are not
interchangeable with the 5s file.

Fourteen features are a statistic of **one bar** (ADR-075 plus `of_delta`).
Five columns (ADR-083) are RTH-window cumulatives or 5-bar transforms:
`of_cvd`, `of_ofi_cum`, `of_ofi_cum_slope_5`, `cs_return_slope_5`, `cs_atr_5`.
Four target-geometry columns (`lg_*` / `st_*`, ADR-082) are causal 10-minute
ATR / last-print scale for the ±50 point TBM target and are **not** 1-bar
tape. They are concatenated only onto the matching dual-binary head.

### Non-Feature Parquet Columns

| Column | Description | Source |
|--------|-------------|--------|
| `bar_end_ts` | Closed-bar UTC timestamp: left-closed bucket `[T, T+15s)` is labeled `T+15s` | `databento.tbbo.ts_event` |
| `_meta_nq_contract` | Lead-month NQ symbol used for that Globex session (e.g. `NQU6`). Excluded from training (`_` prefix). | CME roll calendar |

---

## Module Index

| # | Prefix | Name | Count | Source | NaN Rate |
|---|--------|------|-------|--------|----------|
| 1 | `of_` | [Order Flow](#of) | 5 | TBBO | ~0% 1-bar; CVD/cum OFI NaN outside 08:30–15:00 CT |
| 2 | `cs_` | [Candle Structure](#cs) | 6 | TBBO | ~0% (`cs_clv` NaN when high = low; slope/ATR warmup 5 bars) |
| 3 | `tl_` | [Trade Location](#tl) | 2 | TBBO | ~0% (NaN when spread is 0) |
| 4 | `ss_` | [Same-Side Runs](#ss) | 2 | TBBO | ~0% |
| 5 | `sd_` | [Sub-Bar Dynamics](#sd) | 4 | TBBO | ~0% |
| 6 | `lg_` | [Long target geometry](#lg) | 2 | 15s CS + 10m ATR | warmup on first ~14 closed 10m bars |
| 7 | `st_` | [Short target geometry](#st) | 2 | 15s CS + 10m ATR | warmup on first ~14 closed 10m bars |

**Module**: `src/features/bar_features.py` (`compute_bar_features(tbbo, timestep="15s")`).
Builder: `python scripts/features/build_native_tf_15s.py`.

---

<a id="of"></a>
## 1. Order Flow — `of_` (5 features)

**Source**: `databento.tbbo` (lead-month NQ)  
**Module**: `src/features/bar_features.py` (same tick formulas as `src/features/order_flow.py`)  
**ADR**: [ADR-068](../decisions/ADR-068-tbbo-event-ofi.md) OFI; [ADR-075](../decisions/ADR-075-native-tf-15s-bar-features.md) 1-bar store; [ADR-083](../decisions/ADR-083-15s-rth-cvd-ofi-slope-atr.md) RTH CVD / cum OFI / 5-bar slope

Quote-event imbalance (Cont 2014) and signed trade volume summed over the
15s bar. CVD and cumulative OFI reset each Globex session and only accumulate
on closed bars with `bar_end_ts` in **08:30:00–15:00:00 America/Chicago**
(the last closed 15s bar as cash open prints through the last closed bar at
15:00). Outside that window they are NaN. The 5s parquet also stores
`of_ofi_zscore_60`; that rolling z-score is **not** on the native-15s file.

| Feature | Description |
|---------|-------------|
| `of_ofi` | Tick-level order-flow imbalance summed over the 15s bar (Cont 2014). First print of the Globex session has no previous quote and is dropped. |
| `of_delta` | Signed volume this 15s bar: `Σ size × (+1 if side=B, −1 if side=A, 0 otherwise)` |
| `of_cvd` | Session cumulative `of_delta` on the 08:30–15:00 CT window. NaN outside. Resets each Globex session |
| `of_ofi_cum` | Session cumulative `of_ofi` on the same 08:30–15:00 CT window. NaN outside. Resets each Globex session |
| `of_ofi_cum_slope_5` | 5-bar relative slope of `of_ofi_cum`: `(v[t] − v[t−5]) / abs(v[t−5])`. NaN when `v[t−5]` is 0/NaN or fewer than 5 prior session bars |

**Glossary**

| Term | Definition |
|------|-----------|
| **OFI (order-flow imbalance)** | Per TBBO quote event: `I(bid≥prev_bid)·bid_sz − I(bid≤prev_bid)·prev_bid_sz − I(ask≤prev_ask)·ask_sz + I(ask≥prev_ask)·prev_ask_sz`. Same-price size changes count. Bar value is the sum of events in `[T, T+15s)`. Previous quote may be the last print of the prior bar. Consecutive 15s `of_ofi` values are additive (tick-sum). |
| **of delta** | Signed print volume this bar (contracts). Distinct from OFI (quote events). Additive across bars. |
| **CVD** | Running sum of `of_delta` from the 08:30:00 CT closed bar through 15:00:00 CT. Not the 5s `of_cvd_chg_*` window diffs. |
| **cum OFI** | Same window and reset as CVD, summing `of_ofi` instead of `of_delta`. |
| **5-bar relative slope** | `(latest − 5 bars back) / abs(5 bars back)` on the parquet row grid (5 printed bars, not necessarily 75s if a slot had no prints). Sign follows the change; magnitude is relative to the absolute level 5 bars ago. |
| **TBBO** | Top-of-book trade with BBO at the print (Databento). One row per CME fill. |
| **1-bar** | The feature uses only prints inside this 15s interval (plus the previous quote for OFI). |

---

<a id="cs"></a>
## 2. Candle Structure — `cs_` (6 features)

**Source**: `databento.tbbo` prints in the 15s bar  
**Module**: `src/features/bar_features.py`  
**ADR**: OHLC identities match [ADR-069](../decisions/ADR-069-cs-5s-ohlc.md) (`cs_return` = `(close−open)/open`); computed on the **15s** bar, not 5s and not 1-minute `cs_clv` from the 5s catalog. 5-bar slope / ATR: [ADR-083](../decisions/ADR-083-15s-rth-cvd-ofi-slope-atr.md).

| Feature | Description |
|---------|-------------|
| `cs_return` | `(close − open) / open` of this 15s bar |
| `cs_return_slope_5` | 5-bar relative slope of `cs_return`: `(v[t] − v[t−5]) / abs(v[t−5])`. Globex session grouped (not RTH-clipped). NaN when `v[t−5]` is 0/NaN |
| `cs_clv` | Close location value of this 15s bar: `(2·close − high − low) / (high − low)`. NaN when high = low |
| `cs_range` | `high − low` of this 15s bar (points) |
| `cs_close` | Last print in the 15s bar |
| `cs_atr_5` | SMA of the last 5 15s true ranges (`max(H−L, |H−prev_C|, |L−prev_C|)`). First bar of a Globex session uses H−L only. **Not** Wilder and **not** 5-minute `cs_atr_5m` |

**Glossary**

| Term | Definition |
|------|-----------|
| **open / high / low / close** | First, max, min, last trade price in the left-closed 15s bucket. |
| **cs return** | Bar body as a fraction of open. Distinct from a close-to-close return. |
| **CLV (close location value)** | Where the close sits in the bar's range: +1 = close at the high, −1 = close at the low, 0 = midpoint. |
| **cs range** | High minus low in index points. |
| **cs close** | Last print. On 5s this is also `_close`; native 15s has no `_close` column. |
| **cs atr 5** | Average true range over 5 native 15s bars (75s when every slot has a print). Distinct from 5s-catalog `cs_atr_5m` (Wilder ATR of 5-minute candles). |

---

<a id="tl"></a>
## 3. Trade Location — `tl_` (2 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/bar_features.py` (same identities as `src/features/trade_location.py` bar values)  
**ADR**: [ADR-037](../decisions/ADR-037-size-weighted-tl-zscore-diff.md) size-weighted bar; 15s file does **not** include 30s/5m/30m MAs.

| Feature | Description |
|---------|-------------|
| `tl_trade_location` | Size-weighted `(trade_price − mid) / half_spread` over the 15s bar: `sum(size × loc) / sum(size)` |
| `tl_aggressive_pct` | Size-weighted sweep share over the 15s bar: `sum(size × sweep) / sum(size)`, sweep iff `price > ask` or `price < bid` |

**Glossary**

| Term | Definition |
|------|-----------|
| **tl trade location** | Normalized position of each print in the spread: 0 = mid, +1 = at ask, −1 = at bid. Bar value is size-weighted. |
| **half spread** | `(ask − bid) / 2`. Location is NaN when the spread is 0. |
| **tl aggressive pct** | Fraction of volume that printed strictly through the BBO (true sweep). |

---

<a id="ss"></a>
## 4. Same-Side Runs — `ss_` (2 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/bar_features.py`  
These columns do **not** exist on the 5s parquet. 5s `ss_buy_run` / `ss_sell_run` are last-run-at-close and may span prior bars.

| Feature | Description |
|---------|-------------|
| `ss_buy_run_max` | Longest consecutive buy-initiated (`side='B'`) streak **inside this 15s bar**. Runs reset at the bar boundary |
| `ss_sell_run_max` | Longest consecutive sell-initiated (`side='A'`) streak **inside this 15s bar**. Runs reset at the bar boundary |

**Glossary**

| Term | Definition |
|------|-----------|
| **same-side run** | Consecutive prints on the same aggressor side with no opposite print in between. |
| **ss buy run max / ss sell run max** | Maximum run length observed in this bar only. A streak that started in the previous bar does not carry over. |
| **buy-initiated / sell-initiated** | Databento `side`: `B` lifted the offer, `A` hit the bid. |

---

<a id="sd"></a>
## 5. Sub-Bar Dynamics — `sd_` (4 features)

**Source**: `databento.tbbo` (first/last quote sizes and print high/low/count inside the bar)  
**Module**: `src/features/bar_features.py`  
On 5s, `sd_bid_sz_delta` / `sd_ask_sz_delta` come from BBO-1s snapshots. Native 15s uses TBBO first/last so the same definition works overnight (no BBO-1s dependency).

| Feature | Description |
|---------|-------------|
| `sd_tick_count` | Number of TBBO prints in the 15s bar |
| `sd_bid_sz_delta` | Last bid size minus first bid size in the bar |
| `sd_ask_sz_delta` | Last ask size minus first ask size in the bar |
| `sd_price_impact_per_tick` | Bar range / tick count |

**Glossary**

| Term | Definition |
|------|-----------|
| **sd tick count** | CME fills in the bar (TBBO rows). Quiet overnight bars can be 1; RTH bars are typically tens to hundreds. |
| **sd bid sz delta / sd ask sz delta** | Change in displayed top-of-book size from the first print of the bar to the last. Sudden drops = quotes pulled. |
| **sd price impact per tick** | `(high − low) / tick_count` — points moved per print inside the bar. |

---

<a id="lg"></a>
## 6. Long target geometry — `lg_` (2 features)

**Source**: native 15s `cs_close` / `cs_range` / `cs_clv` → causal Wilder ATR(14) of 10-minute OHLC  
**Module**: `src/features/target_geometry.py`  
**ADR**: [ADR-082](../decisions/ADR-082-aux-labels-and-head-target-geometry.md)

Last-bar context for the **long** dual-binary head only. The shared LSTM encoder does not see these columns. `move_threshold` is 50 (nq-50pt).

| Feature | Description |
|---------|-------------|
| `lg_target_v_atr` | `50 / ATR_10m`. How many 10-minute ATRs the long profit target is |
| `lg_target_v_close` | `50 / cs_close`. Target as a fraction of last print |

---

<a id="st"></a>
## 7. Short target geometry — `st_` (2 features)

**Source**: same ATR and close as `lg_`  
**Module**: `src/features/target_geometry.py`  
**ADR**: [ADR-082](../decisions/ADR-082-aux-labels-and-head-target-geometry.md)

Last-bar context for the **short** dual-binary head only. Signed opposite of the long pair.

| Feature | Description |
|---------|-------------|
| `st_target_v_atr` | `-50 / ATR_10m` |
| `st_target_v_close` | `-50 / cs_close` |

**Glossary**

| Term | Definition |
|------|-----------|
| **10m ATR** | Wilder ATR(14) of 10-minute OHLC reconstructed from 15s high/low (`cs_close`, `cs_range`, `cs_clv`). Last *closed* 10m bar, ffilled. Same identity as 5s `cs_atr_10m` but from the 15s candle, not the 1-minute resample. |
| **head-private** | Concatenated onto that side's trunk after the shared encoder. Opposite-side columns are dropped for that head (trees: column mask; LSTM: separate `long_ctx` / `short_ctx` inputs). |

---

## File layout

Hive partitions (ADR-070) with monthly size-split parts (ADR-075):

```
timeframe=15s/symbol=NQ/year=YYYY/month=MM/features_15s_partN.parquet
```

`year` / `month` are UTC of `bar_end_ts`. A part file grows until ~100 MB, then
`partN+1` starts. Current density is well under 100 MB per month, so most
months are a single `features_15s_part1.parquet`.

Rebuild (full TBBO 1-bar + RTH flow + target geometry):

```bash
cd /opt/vtech && source venv/bin/activate && set -a && source .env && set +a
python scripts/features/build_native_tf_15s.py
```

Patch existing hive (fills `of_delta` from TBBO if missing, then CVD / cum OFI / slopes / ATR):

```bash
python scripts/features/patch_15s_rth_flow.py
```
