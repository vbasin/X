# Feature Catalog — Native 1m

> **Current version**: v1.1 (2026-08-28)
> **Total**: 71 features across 7 modules
> **Bar width**: 1 minute (one output row = one closed Globex minute)
> **Coverage**: every Globex interval that has lead-month NQ TBBO — not the 5s 08:00–11:00 CT clip
> **Store**: `$VTECH_DATA_DIR/sources/features/timeframe=1m/symbol=NQ/year=/month=/features_1m_partN.parquet`
> **ADR**: [ADR-085](../decisions/ADR-085-native-tf-1m-features.md)

This catalog is the native-1m product. Names are reused from the 5s
[FEATURE_CATALOG.md](FEATURE_CATALOG.md) but the **bar is 1 minute**.
`cs_open` / `cs_close` are the 1-minute candle. Rolling suffixes that the
5s catalog documents as minutes (`rv_realized_vol_60` = 5 min) are
wall-clock windows on this grid. `so_*` / 1m `cs_*` shape / `ps_1m_*` are
the same 1-minute modules the 5s parquet ffill-holds.

Options `iv_*` is **not** on this hive. Selected IV columns can be patched
on later.

### Non-Feature Parquet Columns

| Column | Description | Source |
|--------|-------------|--------|
| `bar_end_ts` | Closed-bar UTC timestamp: left-closed `[T, T+1m)` is labeled `T+1m` | `databento.tbbo.ts_event` |
| `_meta_nq_contract` | Lead-month NQ symbol for that Globex session. Excluded from training (`_` prefix). | CME roll calendar |

---

## Module Index

| # | Prefix | Name | Count | Source | NaN Rate |
|---|--------|------|-------|--------|----------|
| 1 | `so_` | [Stochastic](#so) | 6 | 1-min OHLC | warmup K=60 |
| 2 | `cs_` | [Candle Structure](#cs) | 33 | 1-min OHLC + TBBO 15s/30s CLV | MTF until first 5m/10m/15m/30m close |
| 3 | `ps_` | [Parabolic SAR](#ps) | 4 | Closed 1-min OHLC | until 2nd closed bar / ATR(8) |
| 4 | `vw_` | [VWAP](#vw) | 5 | TBBO | Globex session reset |
| 5 | `rv_` | [Realized Volatility](#rv) | 6 | 1-min close | 5/10/30 min warmup |
| 6 | `vp_` | [Volume Profile](#vp) | 10 | TBBO (30m window + prior session) | yday NaN on first session |
| 7 | `ms_` | [Microstructure](#ms) | 7 | TBBO | VPIN bucket warmup |

**Module**: `src/features/native_tf_1m.py`.
Builder: `python scripts/features/build_native_tf_1m.py`.

---

<a id="so"></a>
## 1. Stochastic — `so_` (6 features)

**Source**: 1-min NQ OHLC from TBBO  
**Module**: `src/features/stochastic.py`  
**ADR**: [ADR-038](../decisions/ADR-038-so-d-only-full-60.md), [ADR-039](../decisions/ADR-039-so-sync-exp-envelope.md)

Same four ThinkorSwim %D lines and two sync scores as the 5s catalog, stored once per closed minute.

| Feature | Description |
|---------|-------------|
| `so_fast_d_9_3` | Fast %D: kPeriod=9, Wilder dPeriod=3 |
| `so_fast_d_14_3` | Fast %D: kPeriod=14, Wilder dPeriod=3 |
| `so_fast_d_40_4` | Fast %D: kPeriod=40, Wilder dPeriod=4 |
| `so_full_d_60_10` | Full %D: kPeriod=60, slowing=1, EMA dPeriod=10 |
| `so_sync_score_all` | Sync of all four %Ds, ∈ [−1, +1] |
| `so_sync_score_9_60` | Same model on Fast(9,3) + Full(60,10) only, ∈ [−1, +1] |

---

<a id="cs"></a>
## 2. Candle Structure — `cs_` (33 features)

**Source**: 1-min OHLC from TBBO; 15s/30s CLV from prints; 5m/10m/15m/30m resample `label="right"`  
**Module**: `src/features/candle_structure.py`, `src/features/atr.py`

`cs_open`…`cs_return` are the **1-minute** candle (not 5s). Shape / CLV-MA / range-MA use 1-minute bars (same as the 5s parquet's 1m CS block).

| Feature | Description |
|---------|-------------|
| `cs_open` | First TBBO print in the 1-minute bar |
| `cs_high` | Max TBBO print in the 1-minute bar |
| `cs_low` | Min TBBO print in the 1-minute bar |
| `cs_close` | Last TBBO print in the 1-minute bar |
| `cs_volume` | Sum of TBBO `size` in the 1-minute bar |
| `cs_range` | `cs_high − cs_low` |
| `cs_direction` | `sign(cs_close − cs_open)` |
| `cs_return` | `(cs_close − cs_open) / cs_open` |
| `cs_body_range_ratio` | `abs(open − close) / (high − low)` |
| `cs_body_direction` | +1 if close > open, −1 otherwise |
| `cs_clv` | 1-min Close Location Value ∈ [−1, +1] |
| `cs_clv_ma_4` | Mean of 1-min CLV over 4 bars |
| `cs_clv_ma_6` | Mean of 1-min CLV over 6 bars |
| `cs_clv_ma_10` | Mean of 1-min CLV over 10 bars |
| `cs_clv_ma_20` | Mean of 1-min CLV over 20 bars |
| `cs_upper_shadow` | Upper wick / range |
| `cs_lower_shadow` | Lower wick / range |
| `cs_range_ma_6` | Rolling 1-min range mean, 6 bars |
| `cs_range_ma_12` | Rolling 1-min range mean, 12 bars |
| `cs_range_ma_60` | Rolling 1-min range mean, 60 bars |
| `cs_range_ma_120` | Rolling 1-min range mean, 120 bars |
| `cs_vol_regime` | Current 1-min range / `cs_range_ma_12` |
| `cs_range_chg` | Bar-to-bar 1-min range ratio − 1 |
| `cs_15s_clv` | Last completed 15s CLV as of this minute close |
| `cs_30s_clv` | Last completed 30s CLV as of this minute close |
| `cs_5m_clv` | Last completed 5-minute CLV (held) |
| `cs_10m_clv` | Last completed 10-minute CLV (held) |
| `cs_15m_body_range` | Last completed 15-minute body/range (held) |
| `cs_15m_clv` | Last completed 15-minute CLV (held) |
| `cs_30m_body_range` | Last completed 30-minute body/range (held) |
| `cs_30m_clv` | Last completed 30-minute CLV (held) |
| `cs_atr_5m` | Wilder ATR(14) on 5-minute OHLC, held until next 5m close |
| `cs_atr_10m` | Wilder ATR(14) on 10-minute OHLC, held until next 10m close |

---

<a id="ps"></a>
## 3. Parabolic SAR — `ps_` (4 features)

**Source**: Closed 1-min OHLC  
**Module**: `src/features/psar.py`  
**ADR**: [ADR-055](../decisions/ADR-055-parabolic-sar-closed-bars.md)

| Feature | Description |
|---------|-------------|
| `ps_1m_next_dist_atr` | `(1m close − nextBarSAR) / ATR(8)` at this minute close |
| `ps_1m_trend` | `+1` uptrend / `−1` downtrend at this minute close |
| `ps_lh_dist_sar` | This minute's low−SAR (up) or SAR−high (down) vs the in-effect trail from the prior close. Negative = wick pierced |
| `ps_flip` | `1` when `ps_lh_dist_sar < 0` this minute |

---

<a id="vw"></a>
## 4. VWAP — `vw_` (5 features)

**Source**: TBBO. Session reset via `market_session_date()` (3:00/4:00 PM CT).

| Feature | Description |
|---------|-------------|
| `vw_vwap_dist` | `(close − VWAP) / VWAP` at the minute close |
| `vw_vwap_zscore_60` | Z-score of VWAP distance over 5 minutes |
| `vw_vwap_zscore_120` | Z-score of VWAP distance over 10 minutes |
| `vw_vwap_slope_24` | d(VWAP)/VWAP over 2 minutes |
| `vw_vwap_slope_120` | d(VWAP)/VWAP over 10 minutes |

---

<a id="rv"></a>
## 5. Realized Volatility — `rv_` (6 features)

**Source**: 1-min log returns of `cs_close`. Suffixes keep 5s catalog names; windows are wall-clock.

| Feature | Description |
|---------|-------------|
| `rv_realized_vol_60` | Rolling std of 1-min log returns over 5 minutes |
| `rv_realized_vol_120` | Rolling std over 10 minutes |
| `rv_realized_vol_360` | Rolling std over 30 minutes |
| `rv_vol_of_vol_120` | Std of `rv_realized_vol_60` over 10 minutes |
| `rv_vol_of_vol_360` | Std of `rv_realized_vol_60` over 30 minutes |
| `rv_vol_ratio` | `rv_realized_vol_60 / rv_realized_vol_360` |

---

<a id="vp"></a>
## 6. Volume Profile — `vp_` (10 features)

**Source**: TBBO. 30-minute trailing tick window; yesterday = prior CME session.

| Feature | Description |
|---------|-------------|
| `vp_poc_dist` | `(price − 30m POC) / price` |
| `vp_va_high_dist` | Distance to 30m value-area high |
| `vp_va_low_dist` | Distance to 30m value-area low |
| `vp_in_value_area` | 1 if price inside the 70% value area |
| `vp_poc_slope` | 30m POC change over 2 minutes |
| `vp_yday_poc_dist` | `(price − prior-session POC) / price` |
| `vp_yday_va_high_dist` | Distance to yesterday's VA high |
| `vp_yday_va_low_dist` | Distance to yesterday's VA low |
| `vp_yday_in_value_area` | 1 if today's price is inside yesterday's 70% VA |
| `vp_yday_poc_slope` | Yesterday's first-30m vs last-30m POC migration |

---

<a id="ms"></a>
## 7. Microstructure — `ms_` (7 features)

**Source**: TBBO. VPIN is volume-bucketed (timestep-independent). Kyle / Amihud / z-scores use 5-min and 10-min wall-clock windows.

| Feature | Description |
|---------|-------------|
| `ms_vpin` | Volume-Synchronized Probability of Informed Trading |
| `ms_kyle_lambda_60` | Price impact per unit signed volume over 5 minutes |
| `ms_vpin_zscore_60` | VPIN z-score over 5 minutes |
| `ms_vpin_zscore_120` | VPIN z-score over 10 minutes |
| `ms_amihud_60` | Amihud `|return| / dollar volume` over 5 minutes |
| `ms_amihud_120` | Amihud over 10 minutes |
| `ms_amihud_zscore_120` | Amihud z-score over 10 minutes |

---

## Rebuild

```
python scripts/features/build_native_tf_1m.py
python scripts/features/build_native_tf_1m.py --start 2025-04-07 --end 2025-04-30
```
