# Feature Catalog — Native 1h (HTF context)

> **Current version**: v1.0 (2026-08-28)
> **Total**: 45 features across 6 modules
> **Bar width**: 1 clock hour (one output row = one closed Globex hour)
> **Coverage**: lead-month NQ hours that have `databento.tbbo_1h` — not the 5s 08:00–11:00 CT clip
> **Store**: `$VTECH_DATA_DIR/sources/features/timeframe=1h/symbol=NQ/year=/month=/features_1h_partN.parquet`
> **ADR**: [ADR-091](../decisions/ADR-091-native-tf-1h-1d-htf-context.md)
> **Role**: 45-col hive (ADR-091). `nq-50pt-econ-lg-st` asof-joins a 7-col `regime_v1` subset onto 15s/1m/5m bases **and** the meta ([ADR-095](../decisions/ADR-095-regime-context-fire-aux.md)); not itself a predictive base

This catalog is the native-1h product. Rolling suffixes are **native 1h bars**
(`rv_realized_vol_5` = five closed hourly log returns). Empty hours are
omitted. Source is the 1h cagg, not TBBO ticks.

### Non-Feature Parquet Columns

| Column | Description | Source |
|--------|-------------|--------|
| `bar_end_ts` | Closed-bar UTC timestamp: left-closed `[T, T+1h)` is labeled `T+1h` | `databento.tbbo_1h.bucket` |
| `_meta_nq_contract` | Lead-month NQ symbol for that hour. Excluded from training (`_` prefix). | CME roll calendar |

---

## Module Index

| # | Prefix | Name | Count | Source | NaN Rate |
|---|--------|------|-------|--------|----------|
| 1 | `cs_` | [Candle + ATR](#cs) | 8 | 1h OHLC from `tbbo_1h` | ATR until 14 closed hours |
| 2 | `tr_` | [Trend regime](#tr) | 7 | Closed 1h close / session open | EMA 8/21/50 warmup |
| 3 | `ps_` | [Parabolic SAR](#ps) | 3 | Closed 1h OHLC | until 2nd closed hour / ATR(8) |
| 4 | `zz_` | [ZigZag High/Low](#zz) | 4 | Closed 1h OHLC (EXT ZigZag factory) | undefined until first trend |
| 5 | `rv_` | [Vol regime](#rv) | 6 | 1h close / ATR(5)/ATR(20) | 5/10/20 bar warmup |
| 6 | `sr_` | [Support / resistance](#sr) | 17 | Donchian, prior Globex session, floor pivots, ZZ Fib | Donchian / prior-session warmup |

**Module**: `src/features/native_tf_htf.py`.
Builder: `python scripts/features/build_native_tf_htf.py --tf 1h`.

---

<a id="cs"></a>
## 1. Candle + ATR — `cs_` (8 features)

**Source**: 1-hour NQ OHLC from `databento.tbbo_1h` (lead-month)  
**Module**: `src/features/native_tf_htf.py`, Wilder ATR via `src/features/psar.py` (`_rma`)

`cs_open`…`cs_close` are the **hourly** candle. `cs_atr_14` is Wilder ATR(14)
on this 1-hour bar (not the 5m hive's `cs_atr_5`).

| Feature | Description |
|---------|-------------|
| `cs_open` | First print in the hour |
| `cs_high` | Max print in the hour |
| `cs_low` | Min print in the hour |
| `cs_close` | Last print in the hour |
| `cs_volume` | Sum of TBBO `size` in the hour |
| `cs_range` | `cs_high − cs_low` |
| `cs_atr_14` | Wilder ATR(14) on closed hourly OHLC |
| `cs_clv` | Close location value `(2·close − high − low) / (high − low)` |

---

<a id="tr"></a>
## 2. Trend regime — `tr_` (7 features)

**Source**: closed hourly close; `tr_gap` vs Globex session open  
**Module**: `src/features/native_tf_htf.py`

EMA spans and slope windows are **hour counts**. `tr_ema_stack` is +1 if
EMA8 > EMA21 > EMA50, −1 if the inverse, else 0. Slope is OLS on the last
N closes, divided by ATR(14).

| Feature | Description |
|---------|-------------|
| `tr_ema8_dist` | `(close − EMA8) / close` |
| `tr_ema21_dist` | `(close − EMA21) / close` |
| `tr_ema50_dist` | `(close − EMA50) / close` |
| `tr_ema_stack` | +1 bull stack, −1 bear stack, 0 mixed |
| `tr_slope_10` | 10-hour linreg slope / ATR(14) |
| `tr_slope_20` | 20-hour linreg slope / ATR(14) |
| `tr_gap` | `(close − Globex session open) / session open` |

---

<a id="ps"></a>
## 3. Parabolic SAR — `ps_` (3 features)

**Source**: closed hourly OHLC  
**Module**: `src/features/psar.py` (`sar_on_closed`)

| Feature | Description |
|---------|-------------|
| `ps_trend` | +1 uptrend / −1 downtrend |
| `ps_next_dist_atr` | `(close − nextBarSAR) / ATR(8)` |
| `ps_flip` | 1 on the hour the trail is pierced |

---

<a id="zz"></a>
## 4. ZigZag High/Low — `zz_` (4 features)

**Source**: closed hourly OHLC (EXT Thinkorswim ZigZagHighLow factory knobs)  
**Module**: `src/external_models/zigzag.py`

| Feature | Description |
|---------|-------------|
| `zz_signal` | +1 uptrend / −1 downtrend / 0 undefined |
| `zz_state` | TOS state enum (init / undefined / up / down) |
| `zz_swing_high_dist` | `(close − current swing high) / close` |
| `zz_swing_low_dist` | `(close − current swing low) / close` |

---

<a id="rv"></a>
## 5. Vol regime — `rv_` (6 features)

**Source**: hourly log returns; ATR(5)/ATR(20) on this bar  
**Module**: `src/features/native_tf_htf.py`

| Feature | Description |
|---------|-------------|
| `rv_realized_vol_5` | Std of last 5 hourly log returns |
| `rv_realized_vol_10` | Std of last 10 hourly log returns |
| `rv_realized_vol_20` | Std of last 20 hourly log returns |
| `rv_vol_of_vol_10` | Std of `rv_realized_vol_5` over 10 hours |
| `rv_vol_ratio` | `rv_5 / rv_20` |
| `rv_atr_ratio` | ATR(5) / ATR(20) on hourly OHLC |

---

<a id="sr"></a>
## 6. Support / resistance — `sr_` (17 features)

**Source**: Donchian of native hours; prior **Globex session** H/L/C for
distances and classic floor pivots; Fib 38.2 / 50 / 61.8 of the current ZZ
swing (low + f·range).  
**Module**: `src/features/native_tf_htf.py`

Distances are `(close − level) / close`. Positive = close above the level.

| Feature | Description |
|---------|-------------|
| `sr_donch_high_20_dist` | Distance to 20-hour rolling high |
| `sr_donch_low_20_dist` | Distance to 20-hour rolling low |
| `sr_donch_high_50_dist` | Distance to 50-hour rolling high |
| `sr_donch_low_50_dist` | Distance to 50-hour rolling low |
| `sr_pos_in_range_20` | `(close − 20h low) / (20h high − 20h low)` |
| `sr_pos_in_range_50` | Position in 50-hour range |
| `sr_prior_high_dist` | Distance to prior Globex session high |
| `sr_prior_low_dist` | Distance to prior Globex session low |
| `sr_prior_close_dist` | Distance to prior Globex session close |
| `sr_pivot_dist` | Distance to classic PP from prior session H/L/C |
| `sr_r1_dist` | Distance to R1 (`2·PP − prior low`) |
| `sr_s1_dist` | Distance to S1 (`2·PP − prior high`) |
| `sr_r2_dist` | Distance to R2 (`PP + prior range`) |
| `sr_s2_dist` | Distance to S2 (`PP − prior range`) |
| `sr_fib_382_dist` | Distance to 38.2% of current ZZ swing |
| `sr_fib_500_dist` | Distance to 50% of current ZZ swing |
| `sr_fib_618_dist` | Distance to 61.8% of current ZZ swing |

---

## Rebuild

```bash
python scripts/features/build_native_tf_htf.py --tf 1h
python scripts/features/build_native_tf_htf.py --tf 1h --start 2025-04-07 --end 2026-08-28
```
