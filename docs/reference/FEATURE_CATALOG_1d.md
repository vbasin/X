# Feature Catalog — Native 1d (HTF context)

> **Current version**: v1.0 (2026-08-28)
> **Total**: 45 features across 6 modules
> **Bar width**: one Globex session `[T-1 17:00, T 17:00)` America/Chicago
> **Coverage**: lead-month NQ sessions that have `databento.tbbo_1d` — not the 5s 08:00–11:00 CT clip
> **Store**: `$VTECH_DATA_DIR/sources/features/timeframe=1d/symbol=NQ/year=/month=/features_1d_partN.parquet`
> **ADR**: [ADR-091](../decisions/ADR-091-native-tf-1h-1d-htf-context.md)
> **Role**: 45-col hive (ADR-091). `nq-50pt-econ-lg-st` asof-joins a 6-col `regime_v1` subset onto 15s/1m/5m bases **and** the meta ([ADR-095](../decisions/ADR-095-regime-context-fire-aux.md)); not itself a predictive base

This catalog is the native-1d product. Rolling suffixes are **native session
bars** (`rv_realized_vol_5` = five closed Globex sessions). Source is the
Globex-session 1d cagg, not TBBO ticks. `bar_end_ts` is the **session close**
(next 17:00 CT), which is 23/24/25 UTC hours around DST — not UTC+24h.

### Non-Feature Parquet Columns

| Column | Description | Source |
|--------|-------------|--------|
| `bar_end_ts` | Closed-bar UTC timestamp: session open 17:00 CT labeled at the next 17:00 CT | `databento.tbbo_1d.bucket` |
| `_meta_nq_contract` | Lead-month NQ symbol for that session. Excluded from training (`_` prefix). | CME roll calendar |

---

## Module Index

| # | Prefix | Name | Count | Source | NaN Rate |
|---|--------|------|-------|--------|----------|
| 1 | `cs_` | [Candle + ATR](#cs) | 8 | 1d OHLC from `tbbo_1d` | ATR until 14 sessions |
| 2 | `tr_` | [Trend regime](#tr) | 7 | Closed session close / overnight gap | EMA 8/21/50 warmup |
| 3 | `ps_` | [Parabolic SAR](#ps) | 3 | Closed session OHLC | until 2nd session / ATR(8) |
| 4 | `zz_` | [ZigZag High/Low](#zz) | 4 | Closed session OHLC (EXT ZigZag factory) | undefined until first trend |
| 5 | `rv_` | [Vol regime](#rv) | 6 | Session close / ATR(5)/ATR(20) | 5/10/20 bar warmup |
| 6 | `sr_` | [Support / resistance](#sr) | 17 | Donchian, prior session, weekly floor pivots, ZZ Fib | Donchian / 5-session warmup |

**Module**: `src/features/native_tf_htf.py`.
Builder: `python scripts/features/build_native_tf_htf.py --tf 1d`.

---

<a id="cs"></a>
## 1. Candle + ATR — `cs_` (8 features)

**Source**: Globex-session NQ OHLC from `databento.tbbo_1d` (lead-month)  
**Module**: `src/features/native_tf_htf.py`, Wilder ATR via `src/features/psar.py` (`_rma`)

`cs_open`…`cs_close` are the **session** candle. `cs_atr_14` is Wilder ATR(14)
on this daily bar.

| Feature | Description |
|---------|-------------|
| `cs_open` | First print of the Globex session |
| `cs_high` | Session high |
| `cs_low` | Session low |
| `cs_close` | Last print of the Globex session |
| `cs_volume` | Sum of TBBO `size` in the session |
| `cs_range` | `cs_high − cs_low` |
| `cs_atr_14` | Wilder ATR(14) on closed session OHLC |
| `cs_clv` | Close location value `(2·close − high − low) / (high − low)` |

---

<a id="tr"></a>
## 2. Trend regime — `tr_` (7 features)

**Source**: closed session close; `tr_gap` is overnight (open vs prior close)  
**Module**: `src/features/native_tf_htf.py`

EMA spans and slope windows are **session counts** (8/21/50 trading days).
`tr_ema_stack` is +1 if EMA8 > EMA21 > EMA50, −1 if the inverse, else 0.
Slope is OLS on the last N closes, divided by ATR(14).

| Feature | Description |
|---------|-------------|
| `tr_ema8_dist` | `(close − EMA8) / close` |
| `tr_ema21_dist` | `(close − EMA21) / close` |
| `tr_ema50_dist` | `(close − EMA50) / close` |
| `tr_ema_stack` | +1 bull stack, −1 bear stack, 0 mixed |
| `tr_slope_10` | 10-session linreg slope / ATR(14) |
| `tr_slope_20` | 20-session linreg slope / ATR(14) |
| `tr_gap` | `(open − prior close) / prior close` |

---

<a id="ps"></a>
## 3. Parabolic SAR — `ps_` (3 features)

**Source**: closed session OHLC  
**Module**: `src/features/psar.py` (`sar_on_closed`)

| Feature | Description |
|---------|-------------|
| `ps_trend` | +1 uptrend / −1 downtrend |
| `ps_next_dist_atr` | `(close − nextBarSAR) / ATR(8)` |
| `ps_flip` | 1 on the session the trail is pierced |

---

<a id="zz"></a>
## 4. ZigZag High/Low — `zz_` (4 features)

**Source**: closed session OHLC (EXT Thinkorswim ZigZagHighLow factory knobs)  
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

**Source**: session log returns; ATR(5)/ATR(20) on this bar  
**Module**: `src/features/native_tf_htf.py`

| Feature | Description |
|---------|-------------|
| `rv_realized_vol_5` | Std of last 5 session log returns |
| `rv_realized_vol_10` | Std of last 10 session log returns |
| `rv_realized_vol_20` | Std of last 20 session log returns |
| `rv_vol_of_vol_10` | Std of `rv_realized_vol_5` over 10 sessions |
| `rv_vol_ratio` | `rv_5 / rv_20` |
| `rv_atr_ratio` | ATR(5) / ATR(20) on session OHLC |

---

<a id="sr"></a>
## 6. Support / resistance — `sr_` (17 features)

**Source**: Donchian of native sessions; prior session H/L/C for distances;
classic floor pivots from the prior **5 sessions** (rolling week: max high,
min low, last close); Fib 38.2 / 50 / 61.8 of the current ZZ swing
(low + f·range).  
**Module**: `src/features/native_tf_htf.py`

Distances are `(close − level) / close`. Positive = close above the level.

| Feature | Description |
|---------|-------------|
| `sr_donch_high_20_dist` | Distance to 20-session rolling high |
| `sr_donch_low_20_dist` | Distance to 20-session rolling low |
| `sr_donch_high_50_dist` | Distance to 50-session rolling high |
| `sr_donch_low_50_dist` | Distance to 50-session rolling low |
| `sr_pos_in_range_20` | `(close − 20d low) / (20d high − 20d low)` |
| `sr_pos_in_range_50` | Position in 50-session range |
| `sr_prior_high_dist` | Distance to prior session high |
| `sr_prior_low_dist` | Distance to prior session low |
| `sr_prior_close_dist` | Distance to prior session close |
| `sr_pivot_dist` | Distance to classic PP from prior 5-session H/L/C |
| `sr_r1_dist` | Distance to R1 (`2·PP − week low`) |
| `sr_s1_dist` | Distance to S1 (`2·PP − week high`) |
| `sr_r2_dist` | Distance to R2 (`PP + week range`) |
| `sr_s2_dist` | Distance to S2 (`PP − week range`) |
| `sr_fib_382_dist` | Distance to 38.2% of current ZZ swing |
| `sr_fib_500_dist` | Distance to 50% of current ZZ swing |
| `sr_fib_618_dist` | Distance to 61.8% of current ZZ swing |

---

## Rebuild

```bash
python scripts/features/build_native_tf_htf.py --tf 1d
python scripts/features/build_native_tf_htf.py --tf 1d --start 2025-04-07 --end 2026-08-28
```
