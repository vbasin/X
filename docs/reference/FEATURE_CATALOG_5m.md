# Feature Catalog — Native 5m

> **Current version**: v1.1 (2026-08-29)
> **Total**: 64 features across 8 modules (53 hive + 11 fold-cache `svp_`)
> **Bar width**: 5 minutes (one output row = one closed Globex 5-minute bar)
> **Coverage**: every Globex interval that has lead-month NQ TBBO — not the 5s 08:00–11:00 CT clip
> **Store**: `$VTECH_DATA_DIR/sources/features/timeframe=5m/symbol=NQ/year=/month=/features_5m_partN.parquet`
> **ADR**: [ADR-090](../decisions/ADR-090-native-tf-5m-features.md)

This catalog is the native-5m product. Names that also appear on the 5s
[FEATURE_CATALOG.md](FEATURE_CATALOG.md) or [FEATURE_CATALOG_1m.md](FEATURE_CATALOG_1m.md)
are the **5-minute** candle / window, not a ffill of a faster grid.
Rolling suffixes are **native 5m bars** (`rv_realized_vol_5` = five closed
5-minute log returns = 25 minutes of wall-clock when bars are contiguous).

### Non-Feature Parquet Columns

| Column | Description | Source |
|--------|-------------|--------|
| `bar_end_ts` | Closed-bar UTC timestamp: left-closed `[T, T+5m)` is labeled `T+5m` | `databento.tbbo.ts_event` |
| `_meta_nq_contract` | Lead-month NQ symbol for that Globex session. Excluded from training (`_` prefix). | CME roll calendar |

---

## Module Index

| # | Prefix | Name | Count | Source | NaN Rate |
|---|--------|------|-------|--------|----------|
| 1 | `cs_` | [Candle + ATR](#cs) | 7 | 5-min OHLC from TBBO | ATR until 5 closed bars |
| 2 | `vp_` | [Volume Profile](#vp) | 10 | TBBO (30m window + prior session) | yday NaN on first session |
| 3 | `vw_` | [VWAP](#vw) | 7 | TBBO; session reset; ±2σ bands | z-score / slope warmup |
| 4 | `ps_` | [Parabolic SAR](#ps) | 4 | Closed 5-min OHLC | until 2nd closed bar / ATR(8) |
| 5 | `zz_` | [ZigZag High/Low](#zz) | 8 | Closed 5-min OHLC (EXT ZigZag factory) | undefined until first trend |
| 6 | `vf_` | [Volume Footprint](#vf) | 11 | TBBO prints in the closed 5-min candle | poc_ema_5 after 5 closed bars |
| 7 | `rv_` | [Realized Volatility](#rv) | 6 | 5-min close | 5/10/30 bar warmup |
| 8 | `svp_` | [Session volume profile](#svp) | 11 | fold-cache 5m OHLC+volume, Globex 17:00 CT | not on hive parquet |

**Module**: `src/features/native_tf_5m.py`.
Builder: `python scripts/features/build_native_tf_5m.py`.

---

<a id="cs"></a>
## 1. Candle + ATR — `cs_` (7 features)

**Source**: 5-min NQ OHLC from TBBO  
**Module**: `src/features/native_tf_5m.py`, Wilder ATR via `src/features/psar.py` (`_rma`)

`cs_open`…`cs_close` are the **5-minute** candle. `cs_atr_5` is Wilder ATR
with period **5 on this 5-minute bar** (not the 15s hive's 5-bar 15s ATR, and
not the 5s catalog's `cs_atr_5m` which is ATR(14) of 5-minute bars held on 5s).

| Feature | Description |
|---------|-------------|
| `cs_open` | First TBBO print in the 5-minute bar |
| `cs_high` | Max TBBO print in the 5-minute bar |
| `cs_low` | Min TBBO print in the 5-minute bar |
| `cs_close` | Last TBBO print in the 5-minute bar |
| `cs_volume` | Sum of TBBO `size` in the 5-minute bar |
| `cs_range` | `cs_high − cs_low` |
| `cs_atr_5` | Wilder ATR(5) on closed 5-minute OHLC (SMA seed, then RMA) |

---

<a id="vp"></a>
## 2. Volume Profile — `vp_` (10 features)

**Source**: TBBO. 30-minute trailing tick window; yesterday = prior CME session.  
**Module**: `src/features/native_tf_5m.py` / `src/features/volume_profile.py`

Same ten columns as the 5s / 1m catalogs. POC / VA are rebuilt from ticks
in the 30 minutes ending at this 5-minute close.

| Feature | Description |
|---------|-------------|
| `vp_poc_dist` | `(price − 30m POC) / price` |
| `vp_va_high_dist` | Distance to 30m value-area high |
| `vp_va_low_dist` | Distance to 30m value-area low |
| `vp_in_value_area` | 1 if price inside the 70% value area |
| `vp_poc_slope` | 30m POC change over two 5-minute bars |
| `vp_yday_poc_dist` | `(price − prior-session POC) / price` |
| `vp_yday_va_high_dist` | Distance to yesterday's VA high |
| `vp_yday_va_low_dist` | Distance to yesterday's VA low |
| `vp_yday_in_value_area` | 1 if today's price is inside yesterday's 70% VA |
| `vp_yday_poc_slope` | Yesterday's first-30m vs last-30m POC migration |

---

<a id="vw"></a>
## 3. VWAP — `vw_` (7 features)

**Source**: TBBO. Session reset via `market_session_date()` (3:00/4:00 PM CT).

Session VWAP from prints. Z-score and slope windows are **5m bars**
(5 / 10 / 2 / 10). Two extra columns are distance from close to the
session VWAP Bollinger bands at **± 2 volume-weighted standard deviations
of price around VWAP** (`σ = sqrt(E[price²] − VWAP²)` accumulated in the
session).

| Feature | Description |
|---------|-------------|
| `vw_vwap_dist` | `(close − VWAP) / VWAP` at the 5-minute close |
| `vw_vwap_zscore_5` | Z-score of VWAP distance over 5 bars (25 min) |
| `vw_vwap_zscore_10` | Z-score of VWAP distance over 10 bars (50 min) |
| `vw_vwap_slope_2` | d(VWAP)/VWAP over 2 bars (10 min) |
| `vw_vwap_slope_10` | d(VWAP)/VWAP over 10 bars (50 min) |
| `vw_bb_upper_dist` | `(close − (VWAP + 2σ)) / VWAP` — 0 on the upper band, positive above it |
| `vw_bb_lower_dist` | `(close − (VWAP − 2σ)) / VWAP` — 0 on the lower band, negative below it |

---

<a id="ps"></a>
## 4. Parabolic SAR — `ps_` (4 features)

**Source**: Closed 5-min OHLC  
**Module**: `src/features/psar.py` (`sar_on_closed`)  
**ADR**: [ADR-055](../decisions/ADR-055-parabolic-sar-closed-bars.md)

Same Pine SAR engine as the 1m / 5s `ps_*` columns, run on **this 5-minute
bar** (`start=0.02`, `increment=0.02`, `maximum=0.2`, internal ATR(8)).

| Feature | Description |
|---------|-------------|
| `ps_5m_next_dist_atr` | `(5m close − nextBarSAR) / ATR(8)` at this 5-minute close |
| `ps_5m_trend` | `+1` uptrend / `−1` downtrend at this 5-minute close |
| `ps_lh_dist_sar` | This bar's low−SAR (up) or SAR−high (down) vs the in-effect trail from the prior close. Negative = wick pierced |
| `ps_flip` | `1` when `ps_lh_dist_sar < 0` this bar |

---

<a id="zz"></a>
## 5. ZigZag High/Low — `zz_` (8 features)

**Source**: Closed 5-min OHLC  
**Module**: `src/external_models/zigzag.py` (`zzhl_on_closed`)  
**ADR**: [ADR-063](../decisions/ADR-063-zigzag-high-low.md)

Causal Thinkorswim ZigZagHighLow **state** (not the repainting plot line).
Knobs are the EXT factory card (`config/external_models/zigzag.yaml`):
`percentage_reversal=2.5`, `atr_length=5`, `atr_reversal=1.0`,
`absolute_reversal=0`, `tick_reversal=0`.

| Feature | Description |
|---------|-------------|
| `zz_signal` | `+1` uptrend / `−1` downtrend / `0` init or undefined |
| `zz_state` | TOS state: 0 init, 1 undefined, 2 uptrend, 3 downtrend |
| `zz_swing_high_dist` | `(close − swing high) / close` |
| `zz_swing_low_dist` | `(close − swing low) / close` |
| `zz_reversal_pts` | Points needed to flip (percent of extreme + ATR term + absolute floor) |
| `zz_reversal_frac` | Retracement toward the reverse trigger / `zz_reversal_pts` (1.0 = at trigger) |
| `zz_new_max` | `1` if this bar made a new swing high |
| `zz_new_min` | `1` if this bar made a new swing low |

---

<a id="vf"></a>
## 6. Volume Footprint — `vf_` (11 features)

**Source**: TBBO prints inside the closed 5-minute candle  
**Module**: `src/features/volume_footprint.py`  
**ADR**: [ADR-056](../decisions/ADR-056-volume-footprint-wick-body.md)

Same wick / body / lower split as the 5s catalog's `vf_5m_*`, plus candle
POC location and a 5-bar EMA of that location (the 5s hive stores POC only
on 1-minute candles; the native 5m store keeps it on this bar).

| Feature | Description |
|---------|-------------|
| `vf_5m_upper_vol_frac` | Fraction of 5-min volume in the upper wick |
| `vf_5m_body_vol_frac` | Fraction of 5-min volume in the body (open and close inclusive) |
| `vf_5m_lower_vol_frac` | Fraction of 5-min volume in the lower wick |
| `vf_5m_upper_delta` | 5-min upper-wick ask volume − bid volume |
| `vf_5m_body_delta` | 5-min body ask volume − bid volume |
| `vf_5m_lower_delta` | 5-min lower-wick ask volume − bid volume |
| `vf_5m_upper_delta_scaled` | `upper_vol_frac × (upper_delta / upper_wick_vol)` |
| `vf_5m_body_delta_scaled` | Scaled body delta on the 5-min candle |
| `vf_5m_lower_delta_scaled` | Scaled lower-wick delta on the 5-min candle |
| `vf_5m_poc_loc` | 5-min POC position in the candle range (0 = low, 1 = high) |
| `vf_5m_poc_ema_5` | 5-bar EMA (`span=5`) of closed 5-min `vf_5m_poc_loc` |

---

<a id="rv"></a>
## 7. Realized Volatility — `rv_` (6 features)

**Source**: 5-min log returns of `cs_close`. Windows are native 5m bars.

| Feature | Description |
|---------|-------------|
| `rv_realized_vol_5` | Rolling std of 5-min log returns over 5 bars (25 min) |
| `rv_realized_vol_10` | Rolling std over 10 bars (50 min) |
| `rv_realized_vol_30` | Rolling std over 30 bars (150 min) |
| `rv_vol_of_vol_10` | Std of `rv_realized_vol_5` over 10 bars |
| `rv_vol_of_vol_30` | Std of `rv_realized_vol_5` over 30 bars |
| `rv_vol_ratio` | `rv_realized_vol_5 / rv_realized_vol_30` |

---

<a id="svp"></a>
## 8. Session volume profile — `svp_` (11 features, fold-cache)

**Source**: running Globex-session VBP from **17:00 CT prior day**; 5m
`cs_volume` + typical price.  
**Module**: `src/features/session_vp.py`  
**ADR**: [ADR-099](../decisions/ADR-099-lg-st-morning-onset-session-vp.md)

Attached only onto the **5m** fold-cache frame (not 15s/1m, not hive
parquet). Distinct from 30-minute rolling `vp_*` on this store. 70% VA is
contiguous from POC (Live Chart SVP identity). Break/reclaim flags are
session-ever.

| Feature | Description |
|---------|-------------|
| `svp_poc_dist` | `(close − session POC) / close` |
| `svp_va_high_dist` | `(close − VAH) / close` |
| `svp_va_low_dist` | `(close − VAL) / close` |
| `svp_in_value_area` | 1 if close inside 70% VA |
| `svp_above_poc` | 1 if close > POC |
| `svp_broke_vah` | 1 if close is currently above VAH |
| `svp_broke_val` | 1 if close is currently below VAL |
| `svp_reclaim_vah` | 1 if this Globex session had been above VAH and close is back inside VA |
| `svp_reclaim_val` | 1 if this session had been below VAL and close is back inside VA |
| `svp_poc_cross_up` | 1 on the bar close crosses from ≤POC to >POC |
| `svp_poc_cross_dn` | 1 on the bar close crosses from >POC to <POC |

---

## Rebuild

```
python scripts/features/build_native_tf_5m.py
python scripts/features/build_native_tf_5m.py --start 2025-04-07 --end 2025-04-30
```
