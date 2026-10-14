# Feature Catalog

> **Current version**: v5.15 (2026-08-27)
> **Total**: 256 features across 25 modules
> Native 15s 1-bar catalog (separate Globex product): [FEATURE_CATALOG_15s.md](FEATURE_CATALOG_15s.md)
> Native 1m catalog (so_/cs_/ps_/vw_/rv_/vp_/ms_; no options IV): [FEATURE_CATALOG_1m.md](FEATURE_CATALOG_1m.md)
> Native 5m catalog (cs_/vp_/vw_/ps_/zz_/vf_/rv_; VWAP ±2σ; ATR(5) on the 5m bar): [FEATURE_CATALOG_5m.md](FEATURE_CATALOG_5m.md)
> Native 1h HTF-context catalog (cs_/tr_/ps_/zz_/rv_/sr_): [FEATURE_CATALOG_1h.md](FEATURE_CATALOG_1h.md)
> Native 1d HTF-context catalog (cs_/tr_/ps_/zz_/rv_/sr_, Globex session): [FEATURE_CATALOG_1d.md](FEATURE_CATALOG_1d.md)
> **Typical NaN rate**: ~0.5–1.5% (`iv_1dte_atm` and `iv_term_slope` NaN until 1DTE BBO ingested)
> **Excluded from training**: `_meta_nq_contract`, `_meta_es_contract` (Parquet metadata columns), `_close` (label-generation price — see below)

### Non-Feature Parquet Columns

| Column | Description | Source |
|--------|-------------|--------|
| `_close` | Last NQ trade price per 5s bar (`tbbo.price` resampled to 5s, forward-filled). Label generation / backtest passthrough (not a model input). The same series is exposed as the feature `cs_close` (ADR-069). **Phase 17H** (ADR-009): true 5s last, not 1-min close ffill. | `databento.tbbo` |
| `_meta_nq_contract` | Lead-month NQ futures symbol active on this date (e.g. `NQZ5`) | CME roll calendar |
| `_meta_es_contract` | Lead-month ES futures symbol active on this date (e.g. `ESZ5`) | CME roll calendar |

---

## Module Index

| # | Prefix | Name | Count | Source | NaN Rate |
|---|--------|------|-------|--------|----------|
| 1 | `bp_` | [Book Pressure](#bp) | 24 | BBO-1s (futures) | ~0% |
| 2 | `of_` | [Order Flow](#of) | 17 | TBBO | ~0% (`of_ofi_zscore_60` 5-min warmup) |
| 3 | `tl_` | [Trade Location](#tl) | 16 | TBBO | ~0% (pair z-score warmup 5m/30m) |
| 4 | `ss_` | [Same-Side Runs](#ss) | 3 | TBBO | ~0% |
| 5 | `lt_` | [Large-Trade CVD](#lt) | 4 | TBBO (≥10 contracts) | ~0% |
| 6 | `ta_` | [Trade Arrival](#ta) | 3 | TBBO | ~0% |
| 7 | `ms_` | [Microstructure](#ms) | 7 | TBBO | ~0% |
| 8 | `rv_` | [Realized Volatility](#rv) | 6 | Price (from TBBO) | ~0% |
| 9 | `sd_` | [Sub-Bar Dynamics](#sd) | 8 | BBO-1s + TBBO | ~0% |
| 10 | `vp_` | [Volume Profile](#vp) | 10 | TBBO (today + prior session) | ~0% |
| 11 | `iv_` | [Options IV Surface](#iv) | 16 | BBO-1s (options) + Definitions + `tt_candles` VIX ticks | ~0.5% (`iv_rank_365` needs a 365d `iv_atm_mean` history; `iv_vix_close` from `public.tt_candles`) |
| 12 | `gx_` | [Dealer GEX](#gx) | 4 | BBO-1s (options) + Definitions | ~0.03% (after Phase 17H fix; values corrected for quarterly expiry inclusion) |
| 13 | `dc_` | [Daily Context](#dc) | 4 | Statistics | ~0% |
| 14 | `wv_` | [Wavelets](#wv) | 14 | 1-min candles (tbbo_1min cagg) | ~0% |
| 15 | `cs_` | [Candle Structure](#cs) | 33 | 5s TBBO OHLCV + 1-min candles + 15s/30s CLV + 5m/10m ATR | ~0% (5s OHLC NaN only before the first print) |
| 16 | `vw_` | [VWAP](#vw) | 5 | TBBO | ~0% |
| 17 | `ca_` | [Cross-Asset](#ca) | 12 | ES TBBO | ~0% |
| 18 | `tc_` | [Time Context](#tc) | 22 | Timestamp | ~0% |
| 19 | `ec_` | [Economic Calendar](#ec) | 4 | Static calendar | ~0% |
| 20 | `hp_` | [Hawkes Clustering](#hp) | 3 | Price (from TBBO) | ~0% |
| 21 | `ht_` | [Higher Timeframe](#ht) | 7 | Back-adjusted daily OHLC + YTD H/L | ~0% |
| 22 | `so_` | [Stochastic](#so) | 6 | 1-min OHLC | ~0% (warmup for K=60) |
| 23 | `vf_` | [Volume Candle Footprint](#vf) | 20 | TBBO prints in closed 1-min / 5-min candles | ~0% after first close of each TF; `poc_ema_5` after 5 closed 1m bars |
| 24 | `ps_` | [Parabolic SAR](#ps) | 4 | 1-min OHLC (closed) + 5s TBBO high/low | 1m cols NaN until 2nd closed bar; `next_dist_atr` until ATR(8) |
| 25 | `mbp_` | [MBP-10 depth imbalance](#mbp) | 4 | `databento.mbp_10` (NQU6 L2) | ~0% on dates with MBP-10; all-NaN outside the 1-month table |

---

<a id="bp"></a>
## 1. Book Pressure — `bp_` (24 features)

**Source**: `databento.bbo_1s` (NQ futures `instrument_class='F'`)  
**Module**: `src/features/book_pressure.py`

Measures the state of the limit order book at the NQ futures top-of-book — spread width, bid/ask size imbalance, total displayed depth, and how quickly these metrics are changing. Large one-sided imbalances and sudden depth withdrawals are among the earliest detectable precursors to a directional NQ move.

| Feature | Description |
|---------|-------------|
| `bp_spread_mean` | Mean bid-ask spread within the 5s bar |
| `bp_spread_max` | Maximum spread within the 5s bar |
| `bp_spread_zscore_12` | Spread z-score relative to trailing 12-bar (1 min) average |
| `bp_spread_zscore_60` | Spread z-score relative to trailing 60-bar (5 min) average |
| `bp_spread_zscore_120` | Spread z-score relative to trailing 120-bar (10 min) average |
| `bp_spread_vol_12` | Rolling std of spread over 12 bars (1 min) |
| `bp_spread_vol_60` | Rolling std of spread over 60 bars (5 min) |
| `bp_imbalance` | Bid/ask size imbalance at bar close: `(bid_sz − ask_sz) / (bid_sz + ask_sz)` |
| `bp_imbalance_ma12` | Smoothed imbalance (EWM α at 12 bars) |
| `bp_imbalance_velocity_12` | First difference of imbalance over 12-bar window |
| `bp_imbalance_velocity_60` | First difference of imbalance over 60-bar window |
| `bp_depth_total` | Total displayed depth: `bid_sz + ask_sz` |
| `bp_depth_min` | Minimum depth within the bar |
| `bp_depth_chg_12` | Depth change ratio over 12 bars |
| `bp_depth_chg_60` | Depth change ratio over 60 bars |
| `bp_depth_zscore_12` | Depth z-score (12 bars) |
| `bp_depth_zscore_60` | Depth z-score (60 bars) |
| `bp_depth_velocity_12` | d(depth)/dt over 12 bars |
| `bp_bid_ask_ratio` | Persistent directional pressure: EWM of imbalance sign |
| `bp_fleeting_bid_cv_12` | Coefficient of variation of bid size over 12 bars (liquidity withdrawal signal) |
| `bp_fleeting_bid_cv_60` | Coefficient of variation of bid size over 60 bars |
| `bp_fleeting_ask_cv_12` | CV of ask size (12 bars) |
| `bp_fleeting_ask_cv_60` | CV of ask size (60 bars) |
| `bp_fleeting_score` | Composite liquidity withdrawal score: `(bid_cv + ask_cv) / 2` |

**Glossary**

| Term | Definition |
|------|-----------|
| **bp spread** | The gap between the best bid and best offer prices. Widens during uncertainty or low liquidity. |
| **bp imbalance** | Normalized difference between bid size and ask size at top of book: `(bid_sz − ask_sz) / (bid_sz + ask_sz)`. Positive = more buying pressure queued. |
| **bp imbalance velocity** | Rate of change of the imbalance — how quickly the directional bias in the book is shifting. |
| **bp depth** | Total displayed liquidity at the top of book (`bid_sz + ask_sz`). Drops sharply before large directional moves as market makers pull quotes. |
| **bp depth velocity** | Rate of change of total depth — rapid drops signal order pulling before a move. |
| **bp bid_ask_ratio** | EWM of the sign of imbalance — captures sustained directional pressure over recent bars. |
| **bp fleeting** | Rapidly appearing and disappearing quotes. High CV of bid/ask size means displayed liquidity is being pulled and replaced quickly — a precursor to a sharp move. |
| **bp fleeting score** | Composite average of bid and ask CV — overall liquidity withdrawal intensity. |
| **BBO-1s** | Best Bid and Offer snapshot at 1-second granularity from the Databento feed. |
| **5s bar** | The base time resolution — a 5-second aggregation window. |
| **EWM** | Exponentially Weighted Moving Average: a rolling average where recent observations receive higher weight, controlled by span (number of bars). |
| **CV (Coefficient of Variation)** | Standard deviation divided by the mean — measures relative variability independent of scale. |
| **Z-score** | Number of standard deviations from the rolling mean — normalizes signals for comparison across different market regimes. |

---

<a id="of"></a>
## 2. Order Flow — `of_` (17 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/order_flow.py`  
**ADR**: [ADR-068](../decisions/ADR-068-tbbo-event-ofi.md) (`of_ofi` / `of_ofi_zscore_60`)

Measures the net directional aggression of market participants — who is lifting offers vs. hitting bids. The CVD running total and its short-term changes are among the most predictive signals for identifying momentum or exhaustion in NQ intraday moves. `of_ofi` is quote-event imbalance (Cont 2014), not trade-side volume; it is distinct from `of_delta`.

| Feature | Description |
|---------|-------------|
| `of_delta` | Signed volume per bar: `Σ size × (+1 if side=B, −1 if side=A)` |
| `of_delta_pct` | Delta as % of total bar volume |
| `of_buy_ratio` | Buy-initiated trade fraction |
| `of_trade_count` | Number of trades within the 5s bar |
| `of_cvd_chg_6` | Cumulative Volume Delta change over 6 bars (30 sec) |
| `of_cvd_chg_12` | CVD change over 12 bars (1 min) |
| `of_cvd_chg_24` | CVD change over 24 bars (2 min) |
| `of_cvd_chg_60` | CVD change over 60 bars (5 min) |
| `of_delta_zscore_6` | Z-scored delta over 6 bars |
| `of_delta_zscore_12` | Z-scored delta over 12 bars |
| `of_delta_zscore_24` | Z-scored delta over 24 bars |
| `of_delta_zscore_60` | Z-scored delta over 60 bars |
| `of_volume_ma_60` | 5-minute volume moving average |
| `of_volume_ratio` | Current bar volume / 5-min volume MA |
| `of_cvd_price_divergence` | Sign divergence between CVD direction and price direction |
| `of_ofi` | Tick-level order-flow imbalance summed over the 5s bar (Cont 2014; see glossary) |
| `of_ofi_zscore_60` | `of_ofi` z-score vs rolling 5-minute mean/std (`bars_m(5)` = 60 bars @ 5s). NaN until the window fills (and when std is 0) |

**Glossary**

| Term | Definition |
|------|-----------|
| **of delta** | Signed volume per bar — buys minus sells, in contracts. Positive = net buying aggression. |
| **of delta pct** | Delta as a fraction of total bar volume — normalizes for varying activity levels across sessions. |
| **of buy ratio** | Fraction of volume that was buy-initiated (aggressor lifted the offer). |
| **CVD (Cumulative Volume Delta)** | Running sum of signed volume over the session. Rising CVD = sustained buy aggression; falling = sustained sell aggression. |
| **cvd_chg** | Change in CVD over a lookback window — captures momentum in directional flow without the absolute level. |
| **of delta zscore** | Bar delta normalized by rolling mean and std — identifies when current flow is unusually one-sided relative to recent history. |
| **of volume_ma_60** | 5-minute volume moving average — baseline for gauging whether current bar volume is elevated. |
| **of volume_ratio** | Current bar volume relative to the 5-minute baseline — >1 means above-average participation. |
| **cvd price divergence** | When CVD is moving in the opposite direction of price — a warning that the price move is unsupported by underlying flow. |
| **OFI (order-flow imbalance)** | Per TBBO quote event: `I(bid≥prev_bid)·bid_sz − I(bid≤prev_bid)·prev_bid_sz − I(ask≤prev_ask)·ask_sz + I(ask≥prev_ask)·prev_ask_sz`. Same-price size changes count. First tick of the series has no previous quote and is dropped. Bar value is the sum of events in `[T, T+5s)`. Source columns: `bid_px`, `ask_px`, `bid_sz`, `ask_sz`. |
| **of ofi zscore 60** | `of_ofi` minus its 60-bar mean, divided by 60-bar std. Warmup matches `of_delta_zscore_60`. |
| **TBBO** | Top-of-Book Best-Offer feed — raw trade-by-trade data with bid/ask context from Databento. |
| **aggressor / initiated** | The party that placed a market order to immediately execute against a resting limit order; determines the buy/sell side classification of a trade. |

---

<a id="tl"></a>
## 3. Trade Location — `tl_` (16 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/trade_location.py`  
**ADR**: [ADR-037](../decisions/ADR-037-size-weighted-tl-zscore-diff.md)

Measures where within the spread individual trades are being executed. Aggressive traders cross the spread and lift the offer or hit the bid; passive trades fill near the mid. High aggression signals conviction and often precedes continuation in NQ moves.

Per family (location / aggression): four levels (5s bar + 30s / 5m / 30m MAs) and two pair stats (30s vs 5m, 5m vs 30m) each as short−long and z-score. Bar values are size-weighted; MAs are equal-weight rolls of those bars (`bar_windows`). 1m and 60m MAs are not computed.

| Feature | Description |
|---------|-------------|
| `tl_trade_location` | Size-weighted `(trade_price − mid) / half_spread` per 5s bar: `sum(size × loc) / sum(size)` |
| `tl_loc_ma_30s` | Equal-weight rolling mean of `tl_trade_location` over the last 6 bars (30s @ 5s) |
| `tl_loc_ma_5m` | Equal-weight rolling mean of `tl_trade_location` over the last 60 bars (5m @ 5s) |
| `tl_loc_ma_30m` | Equal-weight rolling mean of `tl_trade_location` over the last 360 bars (30m @ 5s) |
| `tl_loc_diff_30s_5m` | Pair short−long: `tl_loc_ma_30s − tl_loc_ma_5m` |
| `tl_loc_diff_5m_30m` | `tl_loc_ma_5m − tl_loc_ma_30m` |
| `tl_loc_zscore_30s_5m` | `(ma_30s − ma_5m) / std_5m` of the location series |
| `tl_loc_zscore_5m_30m` | `(ma_5m − ma_30m) / std_30m` of the location series |
| `tl_aggressive_pct` | Size-weighted sweep share per 5s bar: `sum(size × sweep) / sum(size)`, sweep iff `price > ask` or `price < bid` |
| `tl_aggr_30s` | Equal-weight rolling mean of `tl_aggressive_pct` over the last 6 bars |
| `tl_aggr_5m` | Equal-weight rolling mean of `tl_aggressive_pct` over the last 60 bars |
| `tl_aggr_30m` | Equal-weight rolling mean of `tl_aggressive_pct` over the last 360 bars |
| `tl_aggr_diff_30s_5m` | Pair short−long: `tl_aggr_30s − tl_aggr_5m` |
| `tl_aggr_diff_5m_30m` | `tl_aggr_5m − tl_aggr_30m` |
| `tl_aggr_zscore_30s_5m` | `(ma_30s − ma_5m) / std_5m` of the aggression series |
| `tl_aggr_zscore_5m_30m` | `(ma_5m − ma_30m) / std_30m` of the aggression series |

**Glossary**

| Term | Definition |
|------|-----------|
| **tl trade location** | Normalized position of each trade within the spread: 0 = at mid, +1 = at/above ask, −1 = at/below bid. Bar feature `tl_trade_location` is size-weighted over 5s; `tl_loc_ma_*` rolls that series; pair `diff` / `zscore` compare 30s vs 5m and 5m vs 30m. |
| **half spread** | Half of the bid-ask spread — the distance from mid to either the best bid or best ask. Used to center the trade location calculation. |
| **tl aggressive pct** | Size-weighted share of volume that fully crossed the BBO — a direct sweep signal indicating urgency. |
| **BBO sweep** | A trade that executes outside (or through) the current best bid/offer, indicating a market participant was willing to move the price to get filled immediately. |

---

<a id="ss"></a>
## 4. Same-Side Runs — `ss_` (3 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/same_side.py`

Counts consecutive trades on the same side (all buys or all sells). Long same-side runs indicate persistent directional aggression from a single actor or coordinated flow, which is a reliable predictor of short-term NQ continuation.

| Feature | Description |
|---------|-------------|
| `ss_buy_run` | Current consecutive buy-initiated trade run length |
| `ss_sell_run` | Current consecutive sell-initiated trade run length |
| `ss_max_run_24` | Maximum same-side run in trailing 24 bars (2 min) |

**Glossary**

| Term | Definition |
|------|-----------|
| **same-side run** | A sequence of consecutive trades all initiating on the same side (all buys or all sells) with no opposing trade interrupting the streak. |
| **ss buy run / ss sell run** | Length of the current uninterrupted streak of buy- or sell-initiated trades. Longer streaks indicate sustained directional aggression. |
| **ss max run** | The longest same-side run observed in the trailing 24-bar (2-minute) window — captures recent peaks of directional commitment. |
| **buy-initiated / sell-initiated** | A trade where the aggressor was a buyer (lifted the ask) or a seller (hit the bid), respectively. Determined from the TBBO side field. |

---

<a id="lt"></a>
## 5. Large-Trade CVD — `lt_` (4 features)

**Source**: `databento.tbbo` (filtered to trades ≥ 10 contracts)  
**Module**: `src/features/large_trade_cvd.py`

Tracks CVD computed only from trades ≥ 10 contracts — isolating "informed" or institutional flow from noise. When large-trade CVD diverges from overall CVD, it suggests smart money is positioned against the crowd, a key setup signal for NQ.

| Feature | Description |
|---------|-------------|
| `lt_cvd` | Cumulative large-trade CVD since CME session open (4 PM CT); resets per session via `market_session_date()` |
| `lt_cvd_chg_24` | Large-trade CVD change over 24 bars (2 min) |
| `lt_cvd_chg_120` | Large-trade CVD change over 120 bars (10 min) |
| `lt_cvd_divergence` | Binary: `sign(large_trade_delta) ≠ sign(total_delta)` — smart money vs crowd divergence |

**Glossary**

| Term | Definition |
|------|-----------|
| **lt cvd** | Cumulative Volume Delta for large trades only — the running net of big-lot buy vs. sell volume, accumulated from the CME Globex session open (4:00 PM CT). Resets at each session open via `market_session_date()` (DST-aware). See CVD in Order Flow glossary. |
| **informed flow proxy** | The assumption that large trades (≥10 contracts) are more likely placed by institutional or informed participants rather than retail noise traders. |
| **lt cvd divergence** | Binary flag set when the sign of large-trade delta differs from the sign of total delta in the same bar — indicates that institutional-sized orders are pushing against the crowd direction. |
| **smart money vs crowd** | A divergence pattern where institutional-sized orders favor one direction while the overall tape favors the other — often a reversal or acceleration signal depending on context. |
| **10-contract threshold** | The minimum trade size to qualify as a "large trade." NQ's multiplier is $20/point, making a 10-contract trade meaningfully large in dollar terms. |

---

<a id="ta"></a>
## 6. Trade Arrival — `ta_` (3 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/trade_arrival.py`

Measures the rate and acceleration of trade execution events. A sudden surge in trade arrivals indicates urgency in the market — panic buying/selling or absorption activity — which often precedes or accompanies large NQ moves.

| Feature | Description |
|---------|-------------|
| `ta_arrival_rate` | Trades per second (EWM smoothed) |
| `ta_arrival_accel` | First difference of arrival rate |
| `ta_arrival_zscore_60` | Z-score of arrival rate relative to 60-bar (5 min) window |

**Glossary**

| Term | Definition |
|------|-----------|
| **trade arrival** | The event of a new trade being executed; measured by count per unit time (trades per second). |
| **ta arrival rate** | Trades per second, smoothed with EWM to reduce bar-to-bar noise. Spikes indicate bursts of market activity. |
| **ta arrival accel** | First difference of arrival rate — positive = accelerating trade flow, negative = decelerating. Captures the onset of urgency before it's fully reflected in price. |
| **ta arrival zscore** | How many standard deviations above or below the 5-minute average the current arrival rate is — normalizes for time-of-day effects (e.g., open vs. midday). |
| **absorption** | When large incoming orders are met by resting liquidity that holds price steady despite high trade arrival rates — signals a strong hidden buyer or seller. |

---

<a id="ms"></a>
## 7. Microstructure — `ms_` (7 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/microstructure.py`

Advanced market microstructure metrics that quantify adverse selection risk and market impact. High VPIN or Kyle lambda signals that informed traders are active, which is associated with larger and more sustained directional NQ moves.

| Feature | Description |
|---------|-------------|
| `ms_vpin` | Volume-Synchronized Probability of Informed Trading — identifies adverse selection risk |
| `ms_kyle_lambda_60` | Price impact per unit signed volume (5-min Kyle lambda) |
| `ms_vpin_zscore_60` | VPIN z-score over 60-bar (5 min) window |
| `ms_vpin_zscore_120` | VPIN z-score over 120-bar (10 min) window |
| `ms_amihud_60` | Amihud illiquidity ratio over 60 bars: `|return| / volume` |
| `ms_amihud_120` | Amihud illiquidity ratio over 120 bars |
| `ms_amihud_zscore_120` | Amihud z-score (120 bars) |

**Glossary**

| Term | Definition |
|------|-----------|
| **VPIN (Volume-Synchronized Probability of Informed Trading)** | Measures how imbalanced recent volume is between buys and sells within fixed-volume buckets. High VPIN = high probability of informed order flow; historically precedes volatility. |
| **adverse selection** | The risk that the counterparty in a trade has superior information, causing the market maker to systematically lose on their position. |
| **Kyle lambda** | Price impact coefficient: how much price moves per unit of signed volume. Higher = less liquid or more information-sensitive market. Computed via rolling regression of returns on signed volume. |
| **ms amihud** | Amihud illiquidity ratio: `|return| / volume` per bar. High values mean the market is thin and price moves more per contract traded — vulnerable to amplified moves. |
| **ms amihud zscore** | Amihud ratio normalized by its rolling distribution — identifies when illiquidity is anomalously elevated relative to recent norms. |
| **ms vpin zscore** | VPIN normalized relative to its rolling distribution — flags when informed activity is unusually elevated. |
| **volume bucket** | A fixed-quantity unit of volume (e.g., 50 contracts) used to synchronize time in the VPIN computation, making it robust to varying trade rates. |

---

<a id="rv"></a>
## 8. Realized Volatility — `rv_` (6 features)

**Source**: Price series from `databento.tbbo`  
**Module**: `src/features/realized_vol.py`

Measures how much NQ has actually moved over recent lookback windows, derived from log returns. The model uses volatility levels and regime transitions (compression vs. expansion) to calibrate expected move magnitudes and avoid triggering on noise during quiet periods.

| Feature | Description |
|---------|-------------|
| `rv_realized_vol_60` | Rolling std of log returns over 60 bars (5 min) — annualized |
| `rv_realized_vol_120` | Realized vol over 120 bars (10 min) |
| `rv_realized_vol_360` | Realized vol over 360 bars (30 min) |
| `rv_vol_of_vol_120` | Std of `rv_realized_vol_60` over 120-bar window (vol-of-vol) |
| `rv_vol_of_vol_360` | Std of `rv_realized_vol_60` over 360-bar window |
| `rv_vol_ratio` | `rv_realized_vol_60 / rv_realized_vol_360` — compression/expansion signal |

**Glossary**

| Term | Definition |
|------|-----------|
| **realized vol** | The rolling standard deviation of log returns over a window, annualized — actual observed price variability as opposed to options-implied (forward-looking) volatility. |
| **log return** | `ln(price_t / price_{t−1})` — the continuously compounded return for a single bar. Preferred over simple returns for statistical properties. |
| **annualized** | Scaled from the 5-second bar frequency to a yearly equivalent (`× sqrt(252 × 6.5 × 3600 / 5)`) for comparability across assets and timeframes. |
| **vol of vol** | The volatility of realized volatility itself — measures how erratically the volatility regime is fluctuating. High vol-of-vol precedes regime transitions. |
| **rv vol ratio** | Short-term realized vol (5 min) divided by long-term (30 min). >1 = expanding volatility; <1 = compression. A key regime signal. |
| **vol compression** | A period where short-term realized vol is below the long-term baseline — often precedes a sharp breakout as energy builds. |
| **vol expansion** | Short-term vol rising above the long-term baseline — confirms a move is underway with increasing energy. |

---

<a id="sd"></a>
## 9. Sub-Bar Dynamics — `sd_` (8 features)

**Source**: `databento.bbo_1s` (futures) + `databento.tbbo`  
**Module**: `src/features/sub_bar_dynamics.py`

Resolves what happened *within* each 5-second bar using two sub-second sources: `tbbo` (one row per trade execution) for count, volume, and price-range features; `bbo_1s` (up to 5 snapshots per 5s bar) for quote-side features. Intra-bar volume distribution, spread changes, and mid-price micro-volatility reveal whether a move is orderly or impulsive — helping the model distinguish real conviction from noise at the finest available resolution.

| Feature | Description |
|---------|-------------|
| `sd_tick_count` | Number of individual trade executions (TBBO rows) within the 5s bar |
| `sd_price_impact_per_tick` | Bar price range / tick count — directional urgency |
| `sd_volume_front_load` | Volume in first half of bar / total bar volume |
| `sd_bid_sz_delta` | Change in bid size within bar |
| `sd_ask_sz_delta` | Change in ask size within bar |
| `sd_spread_range` | `spread_max − spread_min` within bar |
| `sd_mid_micro_vol` | Std of mid-price within the 5s bar |
| `sd_imbalance_trajectory` | Ending imbalance − starting imbalance within bar |

**Glossary**

| Term | Definition |
|------|-----------|
| **sd tick count** | Number of individual trade executions (TBBO rows) within the 5s bar. Source is `databento.tbbo`, one row per CME fill, so values of 50–200+ are normal for active NQ sessions. Cannot be confused with BBO-1s snapshots (which are capped at 5 per bar). |
| **sd price impact per tick** | Bar price range divided by tick count — measures price movement per sub-observation; high values signal directional urgency at the intra-bar level. |
| **sd volume front load** | Volume in the first half of the bar divided by total bar volume. >0.5 means a burst of activity at bar open (e.g., a market order hitting the tape). |
| **sd bid sz delta / sd ask sz delta** | How much the displayed best bid/ask size changed from start to end of the bar — sudden drops indicate quotes are being pulled. |
| **sd spread range** | `spread_max − spread_min` within the bar — how much the spread widened and contracted intra-bar; reflects intra-bar liquidity stress. |
| **sd mid micro vol** | Standard deviation of the mid-price across 1-second snapshots within the 5s bar — measures intra-bar choppiness vs. directional clarity. |
| **sd imbalance trajectory** | Change in bid/ask imbalance from the start to the end of the bar — whether book pressure increased or flipped direction during the bar. |
| **1-second BBO snapshot** | A single top-of-book record sampled at 1s resolution, providing up to 5 sub-samples per 5s bar. |

---

<a id="vp"></a>
## 10. Volume Profile — `vp_` (10 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/volume_profile.py`

Maps where volume has been traded at each price level throughout the session. The Point of Control and Value Area act as structural reference levels — NQ tends to mean-revert toward them in ranging conditions and accelerate away from them during breakouts.

| Feature | Description |
|---------|-------------|
| `vp_poc_dist` | Distance of current price from session Point of Control (max volume price level) |
| `vp_va_high_dist` | Distance from Value Area high (70% volume zone upper bound) |
| `vp_va_low_dist` | Distance from Value Area low |
| `vp_in_value_area` | Binary: 1 if current price is within the 70% value area |
| `vp_poc_slope` | Session POC migration direction (smoothed) |
| `vp_yday_poc_dist` | `(price − yesterday_POC) / price` — prior session's volume profile vs today |
| `vp_yday_va_high_dist` | Distance to yesterday's value-area high |
| `vp_yday_va_low_dist` | Distance to yesterday's value-area low |
| `vp_yday_in_value_area` | 1 if today's price is inside yesterday's 70% value area |
| `vp_yday_poc_slope` | Yesterday's session POC migration (first 30m vs last 30m), constant today |

**Glossary**

| Term | Definition |
|------|-----------|
| **POC (Point of Control)** | The price level where the most volume traded during the session. Acts as a gravitational center; price often revisits it in ranging markets. |
| **vp poc dist** | How far current price is from the POC, normalized by price — large values mean price has moved away from the equilibrium zone. |
| **Value Area** | The range of price levels containing 70% of session volume. Based on the Market Profile concept — the market spends most of its time here in balanced conditions. |
| **vp va high dist / vp va low dist** | Distance from the upper or lower edge of the Value Area — proximity to these edges signals potential support/resistance or breakout levels. |
| **vp in value area** | Binary flag — 1 when price is within the 70% volume zone (balanced, mean-reverting regime), 0 when outside (trending/imbalanced). |
| **vp poc slope** | Direction and rate of POC migration during the session — a drifting POC signals a trending, one-sided market as the dominant price level shifts with the move. |

---

<a id="iv"></a>
## 11. Options IV Surface — `iv_` (16 features)

**Source**: `databento.bbo_1s` (options) + `databento.definitions`  
**Module**: `src/features/options_surface.py`  
**Cadence**: 30s IV computation → forward-filled to 5s grid  
**Engine**: Vectorised Newton-Raphson (30 iter, tol=1e-5), moneyness prefilter `|log(K/S)| < 0.15`, expiry window `d + 400d`

Derives the implied volatility surface from live NQ option quotes. IV reflects what options market makers are pricing in for future volatility. ATM IV level, skew, term structure slope, and velocity provide direct signals about institutional hedging demand and expected move magnitude — critical inputs for identifying when the options market is pricing in a large move.

**v4 re-engineering (2026-05-06):** `iv_atm_mean` and all derived surface features now use only the next two NQE end-of-month chains (`QNE`-prefix, e.g. `QNEV5`, `QNEX5`) interpolated to 30-day constant maturity using VIX-style variance weighting. This produces a cleaner, more stable IV signal by eliminating noise from short-dated weeklies and far-dated quarterly options. `iv_term_slope` is redefined as 30d CM-IV minus `iv_1dte_atm`. New feature `iv_1dte_atm` added.

| Feature | Description | Status |
|---------|-------------|--------|
| `iv_atm_mean` | 30-day constant-maturity ATM IV — interpolated from the two nearest NQE end-of-month chains (QNE prefix) using VIX-style variance weighting | ✅ v4 |
| `iv_skew` | OTM put IV − OTM call IV, linearly interpolated to 30d CM from NQE chains | ✅ v4 |
| `iv_butterfly` | Wings − ATM IV, interpolated to 30d CM from NQE chains | ✅ v4 |
| `iv_0dte_atm` | ATM IV for same-day expiry options (expiration date = session date UTC); NaN when no 0DTE options in BBO | ✅ unchanged |
| `iv_1dte_atm` | ATM IV for options expiring on the next CME trading day (1DTE); NaN until 1DTE BBO data is ingested via `scripts/ingest_nq_opt.py` | ✅ v4 new |
| `iv_25d_risk_reversal` | 25-delta call IV − 25-delta put IV, interpolated to 30d CM from NQE chains | ✅ v4 |
| `iv_25d_butterfly` | 25-delta wings − ATM IV, interpolated to 30d CM from NQE chains | ✅ v4 |
| `iv_term_slope` | 30d CM-IV (`iv_atm_mean`) − `iv_1dte_atm`; positive = term structure in contango (1DTE IV < 30d IV). NaN until 1DTE data ingested. | ✅ v4 redefined |
| `iv_atm_chg_6` | IV velocity over 6 bars (30 sec) — based on 30d CM `iv_atm_mean` | ✅ |
| `iv_atm_chg_12` | IV velocity over 12 bars (1 min) | ✅ |
| `iv_atm_chg_24` | IV velocity over 24 bars (2 min) | ✅ |
| `iv_atm_zscore_6` | IV z-score (6 bars) — based on 30d CM `iv_atm_mean` | ✅ |
| `iv_atm_zscore_12` | IV z-score (12 bars) | ✅ |
| `iv_atm_zscore_24` | IV z-score (24 bars) | ✅ |
| `iv_vix_close` | VIX spot from `public.tt_candles` (`VIX{=m}`); 1-min close visible after that minute ends, then ffilled on 5s. Restored v5.5. | ✅ v5.5 |
| `iv_rank_365` | `100 × (iv_atm_mean − 365d min) / (365d max − 365d min)`. Daily history is prior sessions; today's running IV is included. | ✅ v5.5 |

> **NQE end-of-month chain identification**: `symbol LIKE 'QNE%'` in `databento.definitions`. One chain per calendar month, expiring on the last business day (e.g. QNEV5=Oct 31 2025, QNEX5=Nov 28, QNEZ5=Dec 31). The two nearest future-expiry chains are used. If only one chain is available (e.g. early in a month when chain 1 has already expired), interpolation is skipped and the single chain's IV is used directly.

> **1DTE data dependency**: `iv_1dte_atm` and `iv_term_slope` are NaN until 1DTE BBO data is ingested. Run `scripts/download_1dte_options.py` + `scripts/ingest_nq_opt.py --data-dir data/raw/backfill/nq_1dte_options` to populate. Definitions for 1DTE instruments are already in the table (same weekly strike symbols as 0DTE, identified by `expiration.date == next_trading_day(session_date)`).

> **v3→v4 behavioural changes**: (1) `iv_atm_mean` no longer uses all expirations — now strictly 30d CM from NQE chains. (2) `iv_term_slope` was `quarterly_atm − 0dte_atm`; now `30d_CM − 1dte_atm`. (3) `iv_skew`, `iv_butterfly`, `iv_25d_*` now NQE-chain-only (eliminates noise from thin weeklies). Existing parquets with these columns use v3 values and must be rebuilt via `scripts/patch_iv_columns.py` to get v4 values.

**Glossary**

| Term | Definition |
|------|-----------|
| **IV (Implied Volatility)** | The market's forward-looking estimate of volatility, backed out from option prices using a pricing model. Higher IV = market expects bigger future moves. |
| **ATM (At-The-Money)** | Options whose strike price is near the current underlying price. ATM IV is the most liquid and widely-referenced volatility level. |
| **NQE / QNE chains** | NQ end-of-month options (CME symbol prefix `QNE`), expiring on the last business day of each calendar month. Provide a cleaner, more liquid monthly term structure than mixing weeklies and quarterlies. |
| **30-day constant maturity (CM-IV)** | IV interpolated to a fixed 30-day horizon regardless of the session date. Constructed from two NQE chains (T1 < 30 and T2 > 30 typically) using VIX-style variance weighting: σ_30² = w1·σ1² + w2·σ2². |
| **VIX-style variance weighting** | Weights based on proportion of time intervals: w1=(T2−30)/(T2−T1), w2=(30−T1)/(T2−T1). Interpolates total variance (not IV directly) for maturity consistency. |
| **iv atm mean** | 30-day CM ATM IV — the market's best estimate of 30-day forward NQ volatility at each bar. More stable than raw ATM IV because the 30d target smooths out time-decay effects. |
| **iv skew** | OTM put IV minus OTM call IV at 30d CM. Positive skew means the market pays more for downside protection, reflecting fear of a sharp drop. |
| **iv butterfly** | Wings IV minus ATM IV at 30d CM — measures the premium for far-OTM options in both directions. High butterfly = market pricing fat tails (large moves). |
| **0DTE (Zero Days to Expiration)** | Options expiring on the current trading day. Their IV is highly sensitive to same-session event risk and can move explosively. |
| **iv 0dte atm** | ATM IV of 0DTE options — an intraday fear gauge specific to same-day moves. Unchanged from v3. |
| **1DTE (One Day to Expiration)** | Options expiring on the next CME trading day. Captures near-term event risk with better liquidity than 0DTE at session open. |
| **iv 1dte atm** | ATM IV of 1DTE options. New in v4. Bridge between the 0DTE spot-event fear gauge and the 30d CM structural level. |
| **iv term slope** | 30d CM-IV minus 1DTE ATM IV. Positive = term structure in contango (1DTE calm, longer-term concern); negative = inversion (immediate stress elevated vs. 30d expectation). v4 redefinition: was quarterly−0DTE in v3. |
| **25-delta** | Options with approximately 25% delta — roughly 1 standard deviation OTM. Standard reference point for skew and risk reversal calculations. |
| **iv 25d risk reversal** | 25-delta call IV minus 25-delta put IV at 30d CM. Negative = market fears downside more; positive = upside premium elevated. |
| **iv 25d butterfly** | 25-delta wings minus ATM IV at 30d CM — the kurtosis premium; how much extra the market charges for OTM options relative to ATM. |
| **Newton-Raphson** | Iterative numerical root-finding method used to solve for the IV that makes the Black-Scholes formula match an observed option price. |
| **iv atm chg** | Short-term velocity of 30d CM ATM IV — how quickly IV is rising or falling; rising IV precedes large moves. |
| **iv atm zscore** | 30d CM ATM IV normalized by its recent distribution — identifies when IV is unusually high or low relative to recent history. |
| **moneyness** | `log(K/S)` — how far an option's strike (K) is from the current spot price (S). Used to classify options as ATM, OTM, or ITM. |

---

<a id="gx"></a>
## 12. Dealer GEX — `gx_` (4 features)

**Source**: `databento.bbo_1s` (options) + `databento.definitions`  
**Module**: `src/features/dealer_gex.py`  
**Cadence**: 30s → ffill to 5s  
**Expiry selection**: All expirations where `0 < DTE ≤ 90 days` — captures 0DTE options (highest gamma per unit OI), the front quarterly contract (Dec/Mar/Jun/Sep; dominant institutional OI, ~42-77 instruments and ~14K BBO rows/30s on Oct 3), and near-term weeklies (negligible OI contribution). The old "nearest_monthly = min(expiries ≥ 20 DTE)" logic has been removed — it incorrectly selected thin weekly chains (5–12 instruments, ~300 rows/30s) while excluding the quarterly. See ADR-010.  
**IV proxy**: Constant σ = 0.30 (flat vol); actual per-strike IV from `options_surface` not used (memory cost, future work).

Estimates the aggregate gamma exposure of market makers who have sold options. Dealers hedge their gamma by trading the underlying — when net GEX is large and positive, they buy dips and sell rallies (stabilizing NQ). When GEX flips negative, dealers amplify moves instead of dampening them, creating explosive runs.

| Feature | Description |
|---------|-------------|
| `gx_dealer_gex` | Net dealer gamma exposure (sum of Black-Scholes gamma × OI × spot² × multiplier across all strikes) |
| `gx_gex_per_strike_max` | Maximum GEX at any single strike — identifies gamma pin risk |
| `gx_gex_flip_dist` | Distance from current price to the GEX sign-flip (zero-gamma) level |
| `gx_gex_chg_24` | Rate of change in `gx_dealer_gex` over 24 bars (2 min) |

**Glossary**

| Term | Definition |
|------|-----------|
| **GEX (Gamma Exposure)** | Aggregate dealer gamma exposure in dollar terms: `Σ (BS_gamma × OI × spot² × multiplier)`. Positive = dealers are net long gamma (stabilizing); negative = net short gamma (amplifying). |
| **gamma** | Second derivative of option price with respect to the underlying price — measures how much delta changes per point move. Highest near ATM and near expiry. |
| **gx dealer gex** | Net GEX across all strikes — the sign determines whether dealer hedging will dampen or accelerate NQ moves. |
| **gx gex per strike max** | The largest GEX concentration at a single strike — identifies the "gamma pin" level where dealer hedging is most intense. |
| **gamma pin** | A price level where large gamma concentrations cause dealers to continuously buy/sell to stay delta-neutral, pinning price near that strike as expiry approaches. |
| **gx gex flip dist** | Distance from current price to the zero-GEX level. When price crosses this boundary, dealer hedging dynamics flip from dampening to amplifying moves. |
| **gx gex chg** | Rate of change in total dealer GEX — rising GEX means dealers are accumulating gamma (more stabilizing); falling means they are reducing or being assigned. |
| **OI (Open Interest)** | Total number of outstanding option contracts at a given strike/expiry — used to weight each option's contribution to GEX. |
| **delta-neutral hedging** | A dealer practice of offsetting option delta by trading the underlying so that their net position has zero directional exposure, regardless of small price moves. |
| **BS gamma** | Gamma calculated from the Black-Scholes formula given current spot, strike, time to expiry, and implied volatility. |

---

<a id="dc"></a>
## 13. Daily Context — `dc_` (4 features)

**Source**: `databento.statistics` (7-day lookback)  
**Module**: `src/features/daily_context.py`

Anchors intraday price action relative to key daily reference levels — prior settlement, session high/low, and open interest changes. These structural levels define where the market is "comfortable" vs. overextended, providing context for whether intraday moves are likely to extend or reverse.

| Feature | Description |
|---------|-------------|
| `dc_prev_settle_dist` | `(current_price − prior_settlement) / prior_settlement` |
| `dc_daily_high_dist` | Distance from session high: `(session_high − price) / price` |
| `dc_daily_low_dist` | Distance from session low: `(price − session_low) / price` |
| `dc_oi_chg` | Open interest change (7-day diff at daily granularity, then forward-filled). Fixed v5: was constant 0 before. |

**Glossary**

| Term | Definition |
|------|-----------|
| **dc prev settle dist** | Distance of current price from the prior day's official settlement price — measures how far the market has moved from the overnight anchor level. |
| **prior settlement** | The official closing/settlement price of the NQ futures contract on the previous trading day, set by the exchange. |
| **dc daily high dist / dc daily low dist** | How far current price is from the session's high or low — proximity to extremes signals potential exhaustion or breakout zones. |
| **session high/low** | The highest/lowest traded price since the session open (08:00 CST). |
| **dc oi chg** | Change in open interest over 7 days. Rising OI confirms new money entering positions; falling OI signals position unwinding or expiration. |
| **open interest (OI)** | Total number of outstanding, unsettled futures contracts. A proxy for market participation and conviction in the current trend direction. |

---

<a id="wv"></a>
## 14. Wavelets — `wv_` (14 features)

**Source**: `databento.tbbo_1min` continuous aggregate (1-min candles)  
**Module**: `src/features/wavelets.py`  
**Cadence**: 1-minute → forward-filled to 5s grid

Applies a Discrete Wavelet Transform (DWT) to the 1-minute price series to decompose price action into independent frequency bands. This lets the model separately perceive micro-noise (1-min scale) vs. trend (32-min scale), and detect when energy is concentrating in a specific frequency — a precursor to breakouts.

| Feature | Description |
|---------|-------------|
| `wv_detail_1` | Directional DWT detail coefficient, scale 1 (1 min — highest frequency) |
| `wv_detail_2` | Detail coefficient, scale 2 (2 min) |
| `wv_detail_4` | Detail coefficient, scale 4 (4 min) |
| `wv_detail_8` | Detail coefficient, scale 8 (8 min) |
| `wv_detail_16` | Detail coefficient, scale 16 (16 min) |
| `wv_detail_32` | Detail coefficient, scale 32 (32 min — lowest frequency) |
| `wv_energy_1` | Squared amplitude at scale 1 (energy) |
| `wv_energy_2` | Energy at scale 2 |
| `wv_energy_4` | Energy at scale 4 |
| `wv_energy_8` | Energy at scale 8 |
| `wv_energy_16` | Energy at scale 16 |
| `wv_energy_32` | Energy at scale 32 |
| `wv_approx_slope` | Smoothest trend direction: `np.polyfit` slope across all approximation coefficients. Fixed v5: was near-zero noise before. |
| `wv_fine_coarse_ratio` | `wv_energy_1 / wv_energy_32` — high-frequency energy relative to trend energy |

**Glossary**

| Term | Definition |
|------|-----------|
| **DWT (Discrete Wavelet Transform)** | A signal processing technique that decomposes a time series into components at different frequency scales simultaneously — analogous to a frequency spectrum but for non-stationary signals. |
| **wv detail coefficient** | The directional (signed) component at a given scale from the DWT — captures movement attributed to that specific frequency. Positive = upward pressure at that scale; negative = downward. |
| **wv energy** | Squared DWT detail coefficient at a given scale — a non-negative measure of how much activity exists at that frequency. |
| **scale** | The time horizon of a wavelet component. Scale 1 = 1-min fluctuations (noise); scale 32 = 32-min fluctuations (dominant trend). |
| **wv approx slope** | Linear trend fit across the low-frequency approximation coefficients — the smoothed dominant direction of price over the full window. |
| **approximation coefficients** | The low-frequency remainder after all detail components are extracted — represents the underlying trend direction. |
| **wv fine coarse ratio** | Energy at scale 1 divided by energy at scale 32. >1 means short-term noise dominates; <1 means a coarser trend is the primary driver of price movement. |

---

<a id="cs"></a>
## 15. Candle Structure — `cs_` (33 features)

**Source**: `databento.tbbo` (5s OHLCV) + `databento.tbbo_1min` (1-min structure) + TBBO prints for 15s/30s CLV  
**Module**: `src/features/candle_structure.py`  
**Cadence**: 5s OHLCV is native on the parquet grid (left-labeled `[T, T+5s)`); 1-minute OHLC visible at bar **close** (`T+1m`, ADR-041) then ffill to 5s; 5m/10m/15m/30m CLV `label="right"`; 15s/30s CLV on the 5s grid  
**v5.4 (2026-08-15)**: `cs_5m_clv`, `cs_10m_clv`, `cs_range_ma_6`, `cs_15s_clv`, `cs_30s_clv`  
**ADR-041 (2026-08-21)**: 15m/30m rewritten to `label="right"`; 1-min `cs_clv` / body / shadows / range MAs delayed +1 minute (same as `iv_vix_close`)  
**ADR-069 (2026-08-27)**: 5s `cs_open` / `cs_high` / `cs_low` / `cs_close` / `cs_volume` / `cs_range` / `cs_direction` / `cs_return`. `cs_close` is `_close` (ADR-009) exposed as a feature.

Extracts the 5s candle plus shape/momentum from 1-minute OHLC. Candle body size, shadow ratios, and the Close Location Value capture whether buyers or sellers dominated each bar and whether that pressure is building or fading. 15s/30s CLV uses the same formula on OHLC synthesized from TBBO prints (`label="right"`, no look-ahead). Multi-minute CLV (5/10/15/30) is the last **completed** bucket, held until the next close.

| Feature | Description |
|---------|-------------|
| `cs_open` | First TBBO print in the 5s bar. Empty bar (no prints): last close |
| `cs_high` | Max TBBO print in the 5s bar (includes `cs_close`) |
| `cs_low` | Min TBBO print in the 5s bar (includes `cs_close`) |
| `cs_close` | Last TBBO print in the 5s bar, forward-filled — same series as `_close` |
| `cs_volume` | Sum of TBBO `size` in the 5s bar; `0` when there are no prints |
| `cs_range` | `cs_high − cs_low` |
| `cs_direction` | `sign(cs_close − cs_open)`: `+1` up, `−1` down, `0` flat (empty bars are 0) |
| `cs_return` | `(cs_close − cs_open) / cs_open`. Empty bars are 0 |
| `cs_body_range_ratio` | `abs(open − close) / (high − low)` — body as fraction of total range |
| `cs_body_direction` | +1 if close > open, −1 otherwise |
| `cs_clv` | Close Location Value: `(close − low − (high − close)) / (high − low)` ∈ [−1, +1] |
| `cs_clv_ma_4` | EWM of CLV over 4 bars (4 min) |
| `cs_clv_ma_6` | EWM of CLV over 6 bars |
| `cs_clv_ma_10` | EWM of CLV over 10 bars |
| `cs_clv_ma_20` | EWM of CLV over 20 bars (20 min) |
| `cs_upper_shadow` | Upper shadow / total range: `(high − max(open, close)) / (high − low)` |
| `cs_lower_shadow` | Lower shadow / total range: `(min(open, close) − low) / (high − low)` |
| `cs_range_ma_6` | Rolling mean of 1-min range over 6 bars (6 min) |
| `cs_range_ma_12` | Rolling mean of bar range over 12 bars (12 min) |
| `cs_range_ma_60` | Rolling mean of bar range over 60 bars |
| `cs_range_ma_120` | Rolling mean of bar range over 120 bars |
| `cs_vol_regime` | `(high − low) / cs_range_ma_12` — current range relative to short-term average |
| `cs_range_chg` | Bar-to-bar range change ratio |
| `cs_15s_clv` | 15-second CLV from TBBO print OHLC, ffilled onto the 5s grid |
| `cs_30s_clv` | 30-second CLV from TBBO print OHLC, ffilled onto the 5s grid |
| `cs_5m_clv` | 5-minute aggregated CLV from 1-min candles (`label="right"`) |
| `cs_10m_clv` | 10-minute aggregated CLV from 1-min candles |
| `cs_15m_body_range` | 15-minute aggregated body range ratio (`label="right"`, no look-ahead) |
| `cs_15m_clv` | 15-minute aggregated CLV (`label="right"`) |
| `cs_30m_body_range` | 30-minute aggregated body range ratio (`label="right"`) |
| `cs_30m_clv` | 30-minute aggregated CLV (`label="right"`) |
| `cs_atr_5m` | Wilder ATR(14) on 5-minute OHLC; updated at each 5m close, held 60×5s bars |
| `cs_atr_10m` | Wilder ATR(14) on 10-minute OHLC; updated at each 10m close, held 120×5s bars |

**Glossary**

| Term | Definition |
|------|-----------|
| **cs 5s OHLCV** | Native 5s candle from TBBO prints. `cs_close` = `_close` (labels still use `_close`). Empty bars are a flat candle at the last close with volume 0. Not stale-forwarded with 1-min `cs_*`. |
| **cs range** | 5s high − low. Zero on empty bars. Distinct from `cs_range_ma_*` (1-min range averages). |
| **cs direction** | 5s close vs open. Distinct from 1-min `cs_body_direction`. |
| **cs return** | `(close − open) / open` on the 5s bar. |
| **cs body range ratio** | The candle body as a fraction of total high-low range. ≈1.0 means buyers or sellers dominated the entire bar with no retracement; ≈0 means a doji (indecision). |
| **cs body direction** | Whether the candle closed up (+1) or down (−1) — the bar's directional outcome. |
| **CLV (Close Location Value)** | `(close − low − (high − close)) / (high − low)` ∈ [−1, +1]. +1 = closed at the high (full buyer control); −1 = closed at the low (full seller control). |
| **cs clv ma** | EWM of CLV over a window — smoothed directional pressure indicator. Sustained high values signal persistent buying pressure. |
| **cs upper shadow** | The wick above the body (rejected upside) as a fraction of bar range — large upper shadow signals selling pressure capping rallies. |
| **cs lower shadow** | The wick below the body (rejected downside) as a fraction of bar range — large lower shadow signals buying support. |
| **shadow / wick** | The portion of a candlestick that extends beyond the body — represents price levels visited but not sustained into the close. |
| **cs range ma** | Rolling average of bar range (high − low) over a window — establishes the "normal" bar size for volatility context. |
| **cs vol regime** | Current bar range divided by its short-term average — >1 means this bar is unusually large (volatility expanding). |
| **cs range chg** | Bar-to-bar ratio of range — captures whether successive bars are getting larger or smaller. |
| **15m / 30m aggregated** | Features computed by grouping consecutive 1-min candles into 15- or 30-minute windows for a higher-timeframe perspective. |
| **5m / 10m CLV** | Same aggregation as 15m/30m, CLV only. Restored in v5.4 after SHAP showed intermediate CLV horizons carry side. |
| **15s / 30s CLV** | CLV on OHLC from TBBO prints in a 15s or 30s bucket. Visible only after the bucket closes. |

---

<a id="vw"></a>
## 16. VWAP — `vw_` (5 features)

**Source**: `databento.tbbo`  
**Module**: `src/features/vwap.py`

The Volume-Weighted Average Price is the institutional execution benchmark for the session. Large players measure execution quality against VWAP; when price deviates significantly, mean-reversion pressure builds. The slope of VWAP signals the trend's momentum with volume confirmation.

| Feature | Description |
|---------|-------------|
| `vw_vwap_dist` | `(price − VWAP) / VWAP` — session VWAP distance |
| `vw_vwap_zscore_60` | Z-score of VWAP distance over 60 bars (5 min) |
| `vw_vwap_zscore_120` | Z-score of VWAP distance over 120 bars (10 min) |
| `vw_vwap_slope_24` | d(VWAP)/dt over 24 bars (2 min) |
| `vw_vwap_slope_120` | d(VWAP)/dt over 120 bars (10 min) |

**Glossary**

| Term | Definition |
|------|-----------|
| **VWAP (Volume-Weighted Average Price)** | Session average price weighted by volume at each bar: `Σ(price × volume) / Σ(volume)`. Resets at the CME Globex session open (4:00 PM CT / 3:00 PM CT close) using `market_session_date()` (DST-aware). Institutional benchmark for execution quality. Note: the first ~3 hours of each session (16:00–19:00 CT) fall in the prior UTC day's data load and are absent; accumulation begins from the load boundary (~19:00 CT). |
| **vw vwap dist** | How far current price is from VWAP, normalized by VWAP. Positive = above (buyers in control); negative = below. Large deviations create mean-reversion pressure. |
| **vw vwap zscore** | VWAP distance normalized by its recent standard deviation — identifies when the deviation is unusually extreme relative to recent intraday patterns. |
| **vw vwap slope** | Rate of change of VWAP itself. Rising VWAP slope means volume-weighted price is drifting upward, confirming buy-side dominance. |
| **institutional benchmark** | Large buy-side firms often target VWAP for execution, creating systematic buying below VWAP and selling above it, which drives mean-reversion dynamics. |

---

<a id="ca"></a>
## 17. Cross-Asset — `ca_` (12 features)

**Source**: ES futures from `databento.tbbo`  
**Module**: `src/features/cross_asset.py`  
**Note**: ES data last updated 2026-04-02. Features stale until live streamer restarted.

Captures the relationship between NQ (Nasdaq) and ES (S&P 500) futures. NQ typically leads or lags ES, and breakdowns in their correlation often signal sector-specific news or rotation. Beta and relative strength quantify whether NQ is outperforming or underperforming its macro anchor — a key context for the model's directional bias.

| Feature | Description |
|---------|-------------|
| `ca_corr_60` | NQ/ES return correlation over 60 bars (5 min) |
| `ca_corr_120` | Correlation over 120 bars (10 min) |
| `ca_corr_360` | Correlation over 360 bars (30 min) |
| `ca_ratio` | NQ price / ES price |
| `ca_ratio_zscore_120` | Z-score of `ca_ratio` over 120 bars |
| `ca_es_ret_lag1` | ES return lagged 1 bar (5 sec) |
| `ca_es_ret_lag2` | ES return lagged 2 bars (10 sec) |
| `ca_ret_divergence` | NQ return − ES return in bar |
| `ca_beta_120` | Rolling NQ/ES beta (10 min) |
| `ca_beta_360` | Rolling NQ/ES beta (30 min) |
| `ca_rel_strength_120` | NQ cumulative return − ES cumulative return (10 min) |
| `ca_es_mom_60` | ES 5-minute momentum |

**Glossary**

| Term | Definition |
|------|-----------|
| **ca corr** | Rolling Pearson correlation between NQ and ES bar returns. Near 1.0 = moving in lockstep (macro-driven); <0.8 = divergence (sector-specific or lead/lag signal). |
| **ES (E-Mini S&P 500)** | The S&P 500 futures contract — NQ's primary macro peer and the broadest US equity benchmark. NQ is a tech-heavy subset of ES. |
| **ca ratio** | NQ price divided by ES price — the tech-to-broad-market relative pricing ratio. Rising = tech outperforming. |
| **ca ratio zscore** | NQ/ES ratio normalized by its rolling distribution — identifies when the relative valuation is at an extreme. |
| **ca es ret lag1/lag2** | ES return from 1 or 2 bars ago — captures NQ's tendency to follow ES with a short delay, common when macro news hits. |
| **ca ret divergence** | NQ bar return minus ES bar return. Persistently positive = NQ leading/outperforming; negative = NQ lagging/underperforming. |
| **beta** | Rolling regression coefficient of NQ returns on ES returns. >1 = NQ amplifies ES moves (typical); <1 = NQ is less reactive than usual. |
| **ca rel strength** | Cumulative NQ return minus cumulative ES return over a window — sustained positive values signal tech sector leadership. |
| **ca es mom** | ES 5-minute momentum — directional signal for the broad market tape, independent of NQ-specific dynamics. |

---

<a id="tc"></a>
## 18. Time Context — `tc_` (24 features)

**Source**: Timestamp only  
**Module**: `src/features/time_context.py`

Encodes time-of-day and day-of-week patterns as model features. NQ has strong intraday seasonality — open, mid-session, and afternoon hours differ significantly in liquidity and directional character. Opening range breakouts (5/30 min) are a well-established setup framework. Hourly session blocks allow the model to learn period-specific behaviour across the full Globex day.

> **Note**: blocks from `11am_60m` onward are always 0 at the current `trading_end: "11:00"` config but are included for forward compatibility when the session window is extended.

| Feature | Description |
|---------|-------------|
| `tc_minutes_into_session` | Minutes elapsed since market open (08:30 CST). Pre-market bars (08:00–08:30) have negative values (−30→0) |
| `tc_dow` | Day of week (0=Mon, 4=Fri) |
| `tc_block_premkt_30m` | Binary: 1 during the 30-minute pre-market window (08:00–08:30 CST) |
| `tc_block_open_5m` | Binary: 1 during first 5 minutes after market open (08:30–08:35 CST) |
| `tc_block_open_30m` | Binary: 1 during the first 30 minutes after market open (08:30–09:00 CST) |
| `tc_block_9am_60m` | Binary: 1 during 09:00–10:00 CST |
| `tc_block_10am_60m` | Binary: 1 during 10:00–11:00 CST |
| `tc_block_11am_60m` | Binary: 1 during 11:00–12:00 CST |
| `tc_block_12pm_60m` | Binary: 1 during 12:00–13:00 CST |
| `tc_block_1pm_60m` | Binary: 1 during 13:00–14:00 CST |
| `tc_block_2pm_60m` | Binary: 1 during 14:00–15:00 CST |
| `tc_or5_high` | Running OR high from **closed** 1-min candles in 08:30–08:35 CST; NaN before RTH; frozen after 08:35 (ADR-041) |
| `tc_or5_low` | Running OR low — same closed-bar rule as `tc_or5_high` |
| `tc_or15_high` | Running OR high, closed 1-min candles, 08:30–08:45 CST (ADR-041) |
| `tc_or15_low` | Running OR low — same rule as `tc_or15_high` |
| `tc_or30_high` | Running OR high, closed 1-min candles, 08:30–09:00 CST (ADR-041) |
| `tc_or30_low` | Running OR low — same rule as `tc_or30_high` |
| `tc_or5_breakout_up` | Binary: price > `tc_or5_high` |
| `tc_or5_breakout_down` | Binary: price < `tc_or5_low` |
| `tc_or15_breakout_up` | Binary: price > `tc_or15_high` |
| `tc_or15_breakout_down` | Binary |
| `tc_or30_breakout_up` | Binary: price > `tc_or30_high` |
| `tc_or30_breakout_down` | Binary |
| `tc_session_progress` | Fraction of RTH session elapsed: 0 at market open (08:30), 1 at close (11:00). Pre-market bars clipped to 0. Denominator = 150 min |

> `tc_or{5,15,30}` is a running range of **closed** 1-min candles (ADR-041). Premarket is NaN; the value freezes after the window. Breakouts compare `_close` to that running high/low.

**Glossary**

| Term | Definition |
|------|-----------|
| **tc minutes into session** | Minutes elapsed since the 08:30 CST market open (NYSE open / CME RTH). Zero at 08:30, −30 at 08:00 (pre-market), +150 at 11:00 close. The 08:00 session-filter start is the data-window boundary, not the market open reference. |
| **tc dow** | Day of week as an integer (0=Monday … 4=Friday) — NQ exhibits different volatility and drift characteristics by day. |
| **tc block** | Binary flag indicating a specific session phase. `open_5m`/`open_15m` fire at the actual market open (08:30 CST), NOT the 08:00 data-window start which is pre-market. |
| **OR (Opening Range)** | The high and low established during the first N minutes after the actual market open (08:30 CST). A classic setup framework: the OR defines early support/resistance levels. Was previously (incorrectly) anchored to 08:00 (pre-market). |
| **tc or breakout up/down** | Binary flag triggered when price closes above the OR high (breakout up) or below the OR low (breakout down). A common momentum entry signal in day trading. |
| **tc session progress** | Fraction of the 150-minute RTH session elapsed (0 = 08:30 market open, 1 = 11:00 close). Pre-market bars (08:00–08:30) are clamped to 0. Previously (incorrectly) used a 180-minute denominator anchored to 08:00. |

---

<a id="ec"></a>
## 19. Economic Calendar — `ec_` (4 features)

**Source**: Static calendar (hardcoded 2025–2026 FOMC/CPI/NFP dates)  
**Module**: `src/features/economic_calendar.py`

Flags scheduled macro events that reliably spike NQ volatility. FOMC and CPI dates are the highest-impact events for equities; knowing when they occur allows the model to adjust its behavior and avoid false signals from the pre-announcement compression regime.

| Feature | Description |
|---------|-------------|
| `ec_is_fomc_day` | Binary: 1 on FOMC announcement days |
| `ec_is_cpi_day` | Binary: 1 on CPI release days |
| `ec_is_nfp_day` | Binary: 1 on NFP (non-farm payrolls) days |
| `ec_minutes_to_event` | Minutes until next scheduled macro event (negative after event) |

**Glossary**

| Term | Definition |
|------|-----------|
| **FOMC (Federal Open Market Committee)** | The Fed committee that sets US interest rates. FOMC announcement days cause sharp, often unpredictable NQ moves as rate expectations shift. |
| **CPI (Consumer Price Index)** | A measure of consumer inflation. CPI releases directly impact rate expectations and NQ valuations — surprise readings cause large gaps or reversals. |
| **NFP (Non-Farm Payrolls)** | Monthly US employment report released the first Friday of each month. Moves equity index futures significantly on release. |
| **ec is fomc/cpi/nfp day** | Binary flag — 1 on the day of the respective macro release. Allows the model to treat these days as a separate regime. |
| **ec minutes to event** | Minutes until the next scheduled macro event. Negative after the event fires. Captures the pre-event compression (low activity) and post-event expansion (high volatility) regime. |
| **macro event regime** | A behavioral shift in NQ: participants reduce activity before a scheduled release (compression), then react sharply after (expansion). The model must distinguish these from normal setups. |

---

<a id="hp"></a>
## 20. Hawkes Clustering — `hp_` (3 features)

**Source**: Price series from `databento.tbbo`  
**Module**: `src/features/hawkes_clustering.py`

Models the self-exciting nature of large price moves — big moves beget more big moves. A Hawkes process captures the clustering of extreme events in time. When the cluster ratio is elevated, the market is in a "hot" regime where the probability of another large NQ move is above baseline.

| Feature | Description |
|---------|-------------|
| `hp_large_move_ewm_60` | EWM count of large price moves with 5-minute halflife |
| `hp_large_move_ewm_240` | EWM count with 20-minute halflife |
| `hp_cluster_ratio` | Recent / background intensity — clustering signal. `fillna(0.0)` when no large moves. Fixed v5: was NaN on quiet days before. |

**Glossary**

| Term | Definition |
|------|-----------|
| **Hawkes process** | A stochastic point process where each event (large move) temporarily increases the probability of future events — models the self-exciting, clustering nature of volatility. |
| **self-exciting** | Each large move makes subsequent large moves more likely for a period, creating bursts of activity rather than uniformly distributed events. |
| **hp large move ewm** | Exponentially-weighted count of recent large price moves — more weight on recent events, decaying exponentially over time via the halflife parameter. |
| **halflife** | In EWM decay, the time after which a past event's contribution to the weighted sum is halved. 5-minute halflife = short memory; 20-minute halflife = medium memory. |
| **hp cluster ratio** | Recent event intensity divided by background (baseline) intensity — >1 means large moves are clustering above the normal rate, signaling a "hot" regime. |
| **large move threshold** | A price change that exceeds a statistical threshold to qualify as a "large move" event counted in the Hawkes model. |
| **background intensity** | The baseline rate of large moves expected in a calm market, against which the recent rate is compared. |

---

<a id="ht"></a>
## 21. Higher Timeframe — `ht_` (7 features)

**Source**: Back-adjusted daily OHLC from `tbbo_1min` cagg + `contract_calendar`  
**Module**: `src/features/higher_timeframe.py`  
**Cadence**: Daily → forward-filled to 5s grid

Provides daily-scale context for intraday signals. NQ intraday momentum is more reliable when it aligns with the multi-day trend. Features include the slope and range position of the trend over 10 and 20 days, overnight gap size, and whether short-term volatility is expanding or contracting relative to its longer-term average.

| Feature | Description |
|---------|-------------|
| `ht_daily_trend_10` | 10-day linear regression slope / ATR (back-adjusted NQ price) |
| `ht_daily_trend_20` | 20-day linear regression slope / ATR |
| `ht_daily_pos_in_range_20` | Position within 20-day high/low range: `(price − low_20d) / (high_20d − low_20d)` ∈ [0, 1] |
| `ht_gap_pct` | `(today_open − yesterday_close) / yesterday_close` — overnight gap (raw primary contract, not back-adjusted) |
| `ht_daily_atr_ratio` | 5-day ATR / 20-day ATR — volatility regime |
| `ht_ytd_high_dist` | `(price − YTD high) / price`. YTD high = max of completed session highs this calendar year + today's running 1m high |
| `ht_ytd_low_dist` | `(price − YTD low) / price`. Same construction on session lows |

> Back-adjustment algorithm (v5): at each contract roll boundary, an additive delta `new_close(D−1) − old_close(D−1)` is applied to all OHLC in older contract segments. Eliminates the artificial ±50 pt step that was present in v4 at the Dec 15, 2025 roll.

**Glossary**

| Term | Definition |
|------|-----------|
| **back-adjusted price** | A continuous price series constructed by applying additive offsets at contract roll boundaries, eliminating artificial price gaps so trend calculations are meaningful across roll dates. |
| **contract roll** | When the front-month futures contract expires and traders shift to the next month — creates a price discontinuity that back-adjustment removes. |
| **ht daily trend** | Linear regression slope of back-adjusted daily closes over 10 or 20 days, divided by ATR for normalization. Positive = multi-day uptrend; negative = downtrend. |
| **ATR (Average True Range)** | A volatility measure accounting for overnight gaps: `mean(max(high − low, |high − prev_close|, |low − prev_close|))` over N days. |
| **ht daily pos in range** | Position of current price within the 20-day high/low range: 0 = at 20-day low, 1 = at 20-day high. Measures where in the recent range the market is trading. |
| **ht gap pct** | Overnight gap size as a fraction of prior close — large gaps signal strong overnight news or sentiment shift that may continue or fill intraday. |
| **ht daily atr ratio** | 5-day ATR divided by 20-day ATR. >1 = recent days are more volatile than the longer baseline (expanding regime); <1 = compression. |
| **linear regression slope** | The gradient of the best-fit line across N daily price observations — positive = uptrend, negative = downtrend, normalized by ATR for comparability. |
| **forward-fill** | Carrying the last computed daily value forward to every 5-second bar within the day, since daily features do not change intraday. |

---

<a id="so"></a>
## 22. Stochastic — `so_` (6 features)

**Source**: 1-min NQ OHLC (`tbbo_1min`)  
**Module**: `src/features/stochastic.py`  
**Cadence**: 1-minute → ffilled to 5s  
**ADR**: [ADR-038](../decisions/ADR-038-so-d-only-full-60.md), [ADR-039](../decisions/ADR-039-so-sync-exp-envelope.md)

Four ThinkorSwim stochastic **%D** lines on 1-minute NQ OHLC, plus two signed sync scores. Only %D is stored; %K is computed internally because %D is a moving average of that %K. Names: `so_{fast|full}_{d|k}_{kPeriod}_{dPeriod}` (all four studies emit the `d` line).

ThinkorSwim:

- **StochasticFast**: FastK = raw Lane %K; FastD = MA(FastK, dPeriod). Here MA = Wilder (`α = 1 / dPeriod`).
- **StochasticFull**: FullK = MA(raw %K, slowingPeriod); FullD = MA(FullK, dPeriod). Here slowing = 1 (FullK = FastK) and MA = exponential (`α = 2 / (n + 1)`), dPeriod = 10.

### Sync score calculation

Each member %D casts a **bull** vote and a **bear** vote in [0, 1] every 1-minute bar:

1. **Soft (pre-cross).** Bull = how far %D is into 0–25 times how fast it is rising (capped at +3 points/min). Bear = how far %D is into 75–100 times how fast it is falling. A slow oscillator can vote **before** it tags 20 or 80.
2. **Armed envelope (post-cross).** On a **20-up** or **80-down**, that oscillator arms. While %D is between the print and midpoint **50**, its vote is an affine exponential of the fraction of that 30-point trip (\(\lambda = 1\)): **1 at 20/80**, **0 at 50**. Recross of 20/80 against the signal, or %D strictly through 50, clears the arm. Two consecutive contrary \(\Delta d\) bars zero the envelope that minute (1-bar pause).
3. Per oscillator, vote = max(soft, envelope).
4. \(\text{bull}_m\) / \(\text{bear}_m\) = mean of the N members. Agreement = (count of members with vote > 0.25 on the dominant side) / N.
5. \(\texttt{so\_sync\_score\_*} = \mathrm{clip}((\text{bull}_m - \text{bear}_m)\,(0.5 + 0.5 \times \text{agreement}), -1, 1)\).

`so_sync_score_all` uses all four %Ds (N = 4). `so_sync_score_9_60` is the same model on Fast(9,3) + Full(60,10) only (N = 2).

| Score | Meaning |
|-------|---------|
| **+1** | Strong **buy**: members at a fresh 20-up (or still in the 20→50 envelope / oversold-and-rising), in agreement |
| **positive** | Partial 20-up / oversold-rising setup; not unanimous |
| **~0** | Mid-range, mixed 20-up vs 80-down, or no setup |
| **negative** | Partial 80-down / overbought-falling setup |
| **−1** | Strong **sell**: members at a fresh 80-down (or still in the 80→50 envelope / overbought-and-falling), in agreement |

| Feature | Description |
|---------|-------------|
| `so_fast_d_9_3` | Fast %D: kPeriod=9, Wilder dPeriod=3 |
| `so_fast_d_14_3` | Fast %D: kPeriod=14, Wilder dPeriod=3 |
| `so_fast_d_40_4` | Fast %D: kPeriod=40, Wilder dPeriod=4 |
| `so_full_d_60_10` | Full %D: kPeriod=60, slowing=1, EMA dPeriod=10 |
| `so_sync_score_all` | Sync of all four %Ds, ∈ [−1, +1] |
| `so_sync_score_9_60` | Same model on Fast(9,3) + Full(60,10) only, ∈ [−1, +1] |

**Glossary**

| Term | Definition |
|------|-----------|
| **stochastic oscillator** | Momentum study that places the close inside the recent high–low range on a 0–100 scale. High = close near the window high (overbought); low = close near the window low (oversold). |
| **%K (Lane / FastK)** | Raw stochastic: `100 × (Close − LowestLow(n)) / (HighestHigh(n) − LowestLow(n))`. Computed internally; not stored. |
| **%D (FastD / FullD)** | Smoothed %K — the stored `so_*_d_*` columns. Fast uses Wilder MA of FastK; Full uses EMA of FullK. |
| **kPeriod** | Lookback bars for HighestHigh / LowestLow (the `9`, `14`, `40`, `60` in the column name). |
| **dPeriod** | Smoothing length applied to %K to produce %D (the last number in the column name: 3, 3, 4, 10). |
| **slowing period** | TOS Full only: extra MA of raw %K before %D. Here slowing = 1, so FullK = FastK. |
| **Wilder's MA** | EMA with `α = 1 / length` (slower than a standard EMA of the same length). Used for Fast %D. |
| **standard EMA** | EMA with `α = 2 / (length + 1)`. Used for Full %K / Full %D. |
| **overbought / oversold** | TOS levels 80 and 20. %D above 80 = overbought (80-down is the sell setup); %D below 20 = oversold (20-up is the buy setup). |
| **20-up / 80-down** | %D crossing up through 20 (buy family) or down through 80 (sell family). Arms the post-cross envelope. |
| **armed envelope** | After a 20-up or 80-down, the oscillator keeps a decaying vote while %D travels from the print toward midpoint 50. 1 at the print, 0 at 50. |
| **soft pre-cross** | Vote from location × direction *before* a 20/80 print: oversold (0–25) and rising, or overbought (75–100) and falling. |
| **midpoint 50** | Neutral stochastic level. Envelope is 0 here; going strictly through 50 ends that episode. |
| **so_sync_score_all** | Signed [−1, +1] agreement of all four %Ds on the 20-up (buy) vs 80-down (sell) family. |
| **so_sync_score_9_60** | Same score using only the fastest Fast(9,3) and slowest Full(60,10) — ignores 14 and 40. |

---

<a id="vf"></a>
## 23. Volume Candle Footprint — `vf_` (20 features)

**Source**: `databento.tbbo` prints (`price`, `size`, `side`) inside each closed candle  
**Module**: `src/features/volume_footprint.py`  
**Cadence**: 1-min and 5-min **closed** buckets (`label="right"`) → ffilled to 5s  
**ADR**: [ADR-056](../decisions/ADR-056-volume-footprint-wick-body.md) (supersedes ADR-054 HVN/gap columns)

Splits each completed 1-minute or 5-minute NQ candle into **upper wick / body /
lower wick** and reports volume share, ask−bid delta, and scaled delta in each
region. Body includes the open and close ticks. 1-minute bars also store POC
location in the candle range and a 5-bar EMA of that location. Distinct from
session `vp_*` and from `cs_*` (OHLC shape). `viz_payload(tbbo, timeframe)`
returns the tick ladder plus wick/body bands for overlay.

Ask volume = buy-aggressor size (`side=B`, traded at the ask). Bid volume =
sell-aggressor size (`side=A`, traded at the bid). `delta = ask − bid`.
Scaled delta = `(region vol / candle vol) × (region delta / region vol)`; 0
when the region is empty.

| Feature | Description |
|---------|-------------|
| `vf_1m_upper_vol_frac` | Fraction of 1-min volume traded in the upper wick |
| `vf_1m_body_vol_frac` | Fraction of 1-min volume in the body (open and close inclusive) |
| `vf_1m_lower_vol_frac` | Fraction of 1-min volume in the lower wick |
| `vf_1m_upper_delta` | 1-min upper-wick ask volume − bid volume |
| `vf_1m_body_delta` | 1-min body ask volume − bid volume |
| `vf_1m_lower_delta` | 1-min lower-wick ask volume − bid volume |
| `vf_1m_upper_delta_scaled` | `upper_vol_frac × (upper_delta / upper_wick_vol)` |
| `vf_1m_body_delta_scaled` | `body_vol_frac × (body_delta / body_vol)` |
| `vf_1m_lower_delta_scaled` | `lower_vol_frac × (lower_delta / lower_wick_vol)` |
| `vf_1m_poc_loc` | 1-min POC position in the candle range (0 = low, 1 = high) |
| `vf_1m_poc_ema_5` | 5-bar EMA (`span=5`) of closed 1-min `vf_1m_poc_loc` |
| `vf_5m_upper_vol_frac` | Fraction of 5-min volume in the upper wick |
| `vf_5m_body_vol_frac` | Fraction of 5-min volume in the body (open and close inclusive) |
| `vf_5m_lower_vol_frac` | Fraction of 5-min volume in the lower wick |
| `vf_5m_upper_delta` | 5-min upper-wick ask volume − bid volume |
| `vf_5m_body_delta` | 5-min body ask volume − bid volume |
| `vf_5m_lower_delta` | 5-min lower-wick ask volume − bid volume |
| `vf_5m_upper_delta_scaled` | `upper_vol_frac × (upper_delta / upper_wick_vol)` on the 5-min candle |
| `vf_5m_body_delta_scaled` | Scaled body delta on the 5-min candle |
| `vf_5m_lower_delta_scaled` | Scaled lower-wick delta on the 5-min candle |

**Glossary**

| Term | Definition |
|------|-----------|
| **volume footprint** | Per-candle volume-at-price ladder plus wick/body volume split. |
| **body** | Ticks from `min(open, close)` through `max(open, close)`, including both. |
| **upper / lower wick** | Ticks strictly above / below the body, out to the candle high / low. |
| **ask volume** | Size of buy-initiated prints (aggressor lifted the ask). |
| **bid volume** | Size of sell-initiated prints (aggressor hit the bid). |
| **delta** | Ask volume − bid volume in that region. |
| **scaled delta** | Region volume share times region delta as a fraction of region volume; equals `region_delta / candle_volume` when the region traded. |
| **POC (of the candle)** | Tick inside that candle with the most volume. Not the session POC (`vp_poc_dist`). |
| **poc_loc** | `(POC − low) / (high − low)` ∈ [0, 1]; 0.5 when high = low. |
| **poc_ema_5** | Exponential moving average of the last five **closed** 1-min `poc_loc` values (`span=5`). |
| **closed candle** | Visible only at/after the interval end (1-min at `T+1m`, 5-min at `T+5m`), then held. |
| **viz payload** | `viz_payload(tbbo, timeframe)` — ladder + wick/body bands for drawing 1-min or 5-min footprints. |

---

<a id="ps"></a>
## 24. Parabolic SAR — `ps_` (4 features)

**Source**: 1-min NQ OHLC (`tbbo_1min`) + TBBO prints for 5s high/low  
**Module**: `src/features/psar.py`  
**Cadence**: 1-min trail state at candle close (held 1m); 5s pierce vs that trail  
**ADR**: [ADR-055](../decisions/ADR-055-parabolic-sar-closed-bars.md)

Wilder Parabolic SAR on **closed 1-minute** candles (Pine `start=0.02`,
`increment=0.02`, `maximum=0.2`). SAR, nextBarSAR, AF, EP, 1m distance, and
1m flip are computed internally to produce the four stored columns. They are
not parquet features.

| Feature | Description |
|---------|-------------|
| `ps_1m_next_dist_atr` | `(1m close − nextBarSAR) / ATR(8)` at 1m close; held for the next minute |
| `ps_1m_trend` | `+1` uptrend / `−1` downtrend at that 1m close; held 1 minute |
| `ps_lh_dist_sar` | Uptrend: `5s_low − SAR`. Downtrend: `SAR − 5s_high`. SAR = in-effect `nextBarSAR` from the last closed 1m. Negative = this 5s candle pierced the trail (wick counts) |
| `ps_flip` | `1` on a 5s bar with `ps_lh_dist_sar < 0` and the next 11 bars (60s) |

**Glossary**

| Term | Definition |
|------|-----------|
| **Parabolic SAR** | Wilder trailing stop: SAR steps toward the extreme point (EP) by an acceleration factor (AF) that increases on new extremes, capped at 0.20. |
| **nextBarSAR** | `SAR + AF × (EP − SAR)` after the 1m close — the in-effect trail for the **next** minute (and every 5s bar inside it). |
| **EP (extreme point)** | Best price in the current 1m SAR trend: running high in an uptrend, running low in a downtrend. Reset on a 1m reversal. |
| **AF (acceleration factor)** | Starts at 0.02; adds 0.02 on a new 1m extreme; cap 0.20; resets to 0.02 on a 1m reversal. |
| **1m reversal (internal)** | 1m low strictly below in-effect SAR (uptrend) or 1m high strictly above it (downtrend). Updates nextBarSAR / trend at 1m close; not stored. |
| **ps_lh_dist_sar pierce** | 5s high/low vs the **current** in-effect trail. Does not wait for 1m close. `ps_1m_trend` does not flip intra-minute. |
| **closed 1m** | Left-labeled `[T, T+1m)` is visible at `T+1m`, then held on the 5s grid. |

---

<a id="mbp"></a>
## 25. MBP-10 depth imbalance — `mbp_` (4 features)

**Source**: `databento.mbp_10` (NQ lead, currently NQU6)  
**Module**: `src/features/mbp_depth.py`  
**ADR**: [ADR-057](../decisions/ADR-057-mbp10-5s-mean-depth-imbalance.md)

Per book-changing MBP-10 event, `bid_count` / `ask_count` are the sum of the
ten resting sizes (NULL level → 0). The 5s parquet stores the **mean** of those
event values over `[T, T+5s)`. The denominator is the number of events used in
that average (`imbalance_log` drops events with either side 0).

| Feature | Description |
|---------|-------------|
| `mbp_bid_count` | Mean of `sum(bid_sz_00..09)` over MBP-10 events in the 5s bar |
| `mbp_ask_count` | Mean of `sum(ask_sz_00..09)` over those events |
| `mbp_imbalance_diff` | Mean of `(bid_count − ask_count)` per event |
| `mbp_imbalance_log` | Mean of `ln(bid_count) − ln(ask_count)` over events with both sums > 0 |

Empty 5s bars (no MBP updates) are NaN. Not ffilled. The 5s parquet for
2026-07-21 → 2026-08-20 was built while `databento.mbp_10` held that month;
the hypertable itself was emptied 2026-09-30 (DDL kept). Rebuilding `mbp_*`
needs the L2 rows again.

---

## Removed Features (not in v5)

| Module | Features | Reason for removal |
|--------|----------|-------------------|
| `equity_context.py` | 23 (NVDA, TSLA, XLK, SMH) | Daily data forward-filled to 5s — constant within day, no intraday signal |
| `macro_sentiment.py` | 10 (VIXY) | Same daily-resolution issue |
| `of_delta_sum_{w}` | 4 | Redundant with `of_delta_zscore` |
| `cs_clv_bullish_count_{w}` | 4 | Redundant with `cs_clv_ma` |
| `cs_5m_{body_range, clv}` | 2 | 128-bar sequence already covers >5 min |
| `vw_{above,below}_band_{1,2}` | 4 | Binary thresholds of a continuous z-score |
| `tc_block_{overnight, premarket}` | 2 | Always 0 with 08:00–11:00 session filter |
