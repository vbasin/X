# ADR-123: First-passage candidate-exit surface replaces fixed +50/−6 for `mnq-auction-firstpass`

## Status
Accepted — 2026-09-09. Layers 0–2 implemented and validated; Layers 3–5 not built.
Live pin unchanged. Nothing promoted.

Amended by [ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md) — 2026-09-11.
The Layer 0 bound construction is now width-capped and the Layer 2 entry is a
resting order rather than a fill at the decision close. **Every label row and
every measured result below predates that change and is not comparable to it.**

## Date
2026-09-09

## Context

The `nq-50pt-econ-lg-st` objective asks a model to predict whether a **fixed**
+50 / −6 point event occurs in the next 10 minutes. An audit of that objective
found three defects that no amount of hyperparameter search can remove.

### 1. Effective sample size is ~15–19× smaller than what is scored

A 10-minute label on a 15s decision grid means 40 consecutive rows describe the
same price move. Measured on the active label contract `b7befbc356d43c8a`
(08:30–11:00 CT, 290 sessions):

| | rows scored | independent episodes | inflation |
|---|---|---|---|
| `y_long` | 13,949 | **931** | 15.0× |
| `y_short` | 16,044 | **835** | 19.2× |

Diagnostics report `n_rows: 174,290` and compute pooled ROC AUC as if those were
independent draws. Correcting the count widens the May–Jun 2026 holdout interval
on the 15s LSTM from `[0.617, 0.635]` to **`[0.566, 0.686]`** (z 26.6 → 4.1).
The skill is real but small; every candidate ranking made at ΔAUC ≈ 0.02 was
made inside the noise band.

A second consequence: at the episode level the class balance is ~21% long /
19% short / 59% neither. The ~8% event rate that `class_balance=sqrt` vs
`balanced` was tuned against ([ADR-044](ADR-044-lstm-class-balance-sqrt.md),
and the "compressed vs over-hot, no in-between" dead end recorded in
[ROADMAP](../roadmap/ROADMAP.md)) is an artifact of oversampling, not a property
of the market.

### 2. The labeller is structurally optimistic, and ADR-106 widened it

`_classify_triple_barrier` in [`src/training/labels.py`](../../src/training/labels.py)
tests the **target on the intrabar extreme** and the **stop on the 5s close**:

```python
(first_up_target, high_diff >= big_threshold),    # target: HIGH
(first_dn_target, low_diff  <= -big_threshold),   # target: LOW
(first_up_stop,   close_diff >= mae_tolerance),   # stop: CLOSE
(first_dn_stop,   close_diff <= -mae_tolerance),  # stop: CLOSE
```

with `up_wins = first_up_target <= first_dn_stop` when
`target_observation != "candle_close"` — ties resolve to the target. A bar that
traded 9 points against the entry and closed +1 is a clean winner.
[ADR-052](ADR-052-path-realised-account-simulator.md) already measured the cost
of this on the causal tree book: HR 0.328 → **0.190**, NE 11.34 → **3.97**.
[ADR-106](ADR-106-configurable-tbm-target-observation.md) then selected
`target_observation: tbbo`, which widens the asymmetry rather than closing it.

At 50/−6 the break-even hit rate is 10.71%. Observed OOF hit rates are 14–17%.
Discounted by the ADR-052 factor, the edge is at or below break-even.

### 3. A fixed point geometry is not a stationary event

50 points is a large move in a 40-point-range session and a small one in a
120-point session, so the model must learn regime and signal simultaneously from
~1,766 episodes. [ADR-053](ADR-053-vol-normalized-path-labels-and-encoder-ablation.md)
identified this and built vol-normalised path labels; they were never adopted by
the live-track objectives.

### What the operator actually wants

Entry, target and stop chosen per trade from prior sessions' volume/delta
structure and statistical probability, maximising profit against drawdown —
not a prediction that one hard-coded event occurs.

### What the literature says

Consistent with [ADR-122](ADR-122-structural-context-and-closed-research-loop.md)'s
survey, and adding four sources that bear directly on the label scheme:

- **[TLOB](https://arxiv.org/abs/2502.15757)** (Berti & Kasneci, 2025) — a plain
  MLP surpasses SoTA LOB models; the paper's own contribution includes a new
  labelling method that removes horizon bias, and it reports that redefining
  trends to include spread costs collapses performance. Labels dominate
  architecture.
- **[LOBCAST](https://arxiv.org/abs/2308.01915)** (Prata et al., 2023) — all 15
  SoTA deep models "exhibit a significant performance drop when exposed to new
  data." The 0.696 → 0.626 OOF-to-holdout fall is the expected behaviour, not a
  local bug.
- **[Deep LOB Forecasting](https://arxiv.org/abs/2403.09267)** (Briola et al.,
  2024) — "high forecasting power does not necessarily correspond to actionable
  trading signals; traditional ML metrics fail to adequately assess the quality
  of forecasts."
- **[Better Inputs Matter More Than Stacking Another Hidden Layer](https://arxiv.org/abs/2506.05764)**
  (Wang, 2025) — XGBoost and logistic regression match or beat DeepLOB once
  preprocessing is right.

## Decision

Do not learn the decision. **Learn the conditional law of the path, then solve
the decision.** For an entry at *t*, side *d*, target distance *p*, stop
distance *s*, max horizon *H*:

$$\mathbb{E}[\text{PnL}] = p\,P(T_p < T_s \le H) - s\,P(T_s < T_p \le H) + \mathbb{E}[R_H\mid\text{neither}]\,P(\text{neither})$$

Every term is a first-passage probability. Estimating that law conditional on
features prices **any** exit geometry at inference time, so target selection
moves out of training and into an optimiser.

### Layer 0 — candidate generation ([`src/training/auction_levels.py`](../../src/training/auction_levels.py))

Volume-at-price and delta-at-price per session from the 1s path; each bar spreads
its volume across the bins its range covers (difference array + `cumsum`, exact,
O(n)). Recency-weighted composite over the prior **30** sessions. Candidates are
local extrema (HVN/LVN) of that composite **unioned with today's developing
profile**, capped at **10 per side**.

Nodes are searched per profile and then unioned, never on the summed profile:
the 30-session composite carries ~30× the volume of a developing session, so a
summed search makes the candidate set go blind exactly when price trends out of
the historical range. This was observed in practice — 0 of 375 decision bars had
candidates on 2026-05-06 before the fix.

### Layer 1 — continuous descriptors, no level enum

A candidate is a **price described by properties**, never a categorical type.
There is deliberately no `level_type` column: `dist_r`, `vol_at_level`,
`delta_proxy_at_level`, `node_strength`, `n_sessions_confluence_{5,10,30}`,
`path_volume_between`, `room_beyond`, `position_vs_value`. A model trained on
these learns the physics of a node and generalises to levels nobody enumerated;
a model trained on `level_type == "VAH"` memorises a name.

Descriptors are expressed in the **trade frame** — signed relative to trade
direction — so a long and a short are the same problem rotated, and one model
serves both sides at full sample size.

**Delta requires TBBO — measured, not assumed.** `parquet_path1s` carries OHLCV
only. A CLV-based aggressor proxy was validated against true `of_delta` on 60
overlap sessions ([`scripts/features/validate_delta_proxy.py`](../../scripts/features/validate_delta_proxy.py)):

| test | result |
|---|---|
| per-bar (15s) | pearson r **0.52**, sign agreement **65%** |
| session-cumulative CVD | median r 0.50, but **p10 = −0.45** (anti-correlated on some sessions) |
| **delta-at-price profile** (the actual use case) | **r ≈ 0.00**, spearman 0.01 |

At 1s the CLV signal is dominated by bid-ask bounce, and aggregating by *price*
rather than by time does not cancel it. The proxy is therefore rejected.
:func:`session_profile` takes an explicit `delta` array and
:func:`allocate_tf_delta` spreads true 15s `of_delta` onto 1s rows in proportion
to the volume each contributed — sign from TBBO, only intra-bar price
attribution approximated. Reconstruction is exact (max error 0.0 over 1,979 of
1,980 bars on 2026-05-11). Sessions without hive coverage get a **zero** delta
channel, never an estimated one.

**Consequence for history:** delta-derived features are gated to TBBO
(2025-04→). Deep 2018→2026 history can still supply price and volume structure,
but not aggression.

### Layer 2 — first-passage labels ([`src/training/first_passage.py`](../../src/training/first_passage.py))

One row per (decision bar, side, target, stop), resolved on the 1s high/low path
with `tie_policy="stop_first"` and a **session-close time barrier** at 15:00 CT.
Censoring at the flatten is a first-class outcome, not a defect — the book
flattens, so "neither barrier touched" is a real exit at the closing price and
the competing-risks head in [`src/models/heads.py`](../../src/models/heads.py)
consumes it directly.

`path_labels._first_touch` is **not** reused: it walks the path once per
geometry at O(horizon × entries), which for a 10×10 grid over a 6.5-hour session
means walking ~23,400 seconds a hundred times per entry. Instead the path is
walked **once per entry** to build cumulative extrema, which are monotone, so
every candidate's first-touch time is a `searchsorted` — O(T + K log T). Verified
identical to brute force over 300 randomised paths
([`tests/test_first_passage.py`](../../tests/test_first_passage.py)).

### Objective parameters

| | value |
|---|---|
| instrument | **MNQ** ($2/index point). NQ prices unchanged — only the multiplier differs |
| session | 08:30–15:00 CT, entries to 14:45, flatten 15:00 |
| decision grid | 1 minute |
| time barrier | session close |
| concurrency | 1 position, long XOR short |
| account / risk | $5,000 / 1–2% per trade, fixed fraction |
| KPI | net profit via `portfolio_sim.py`, subject to `n_trades >= 150` and an MDD cap |

MNQ replaces NQ because $1,000 risk per trade on a $5,000 account is 20% per
trade — five losses to ruin, ~99.99% probability of ruin over 150 trades at the
observed 14–22% hit rates — and because a $20/point contract admits exactly one
position size on that account, making opportunity-scaled sizing inexpressible.

## Measured baseline (282 sessions, 2025-05-01 → 2026-06-30, 18.3M rows)

**Zero unconditional edge, by construction.** Diagonal cells price to
`mean_pnl = 0.0000` exactly and off-diagonal cells mirror to machine precision
(+0.1113 / −0.1113, +2.6853 / −2.6853). Because both sides are labelled on the
same candidate set, any positive expectancy a model reports must come from
conditioning — the geometry itself is fair. This is the property the +50/−6
labeller lacked.

| metric | value |
|---|---|
| outcome mix | target 44.6% / stop 44.6% / censored at 15:00 **10.9%** |
| median holding time | **13.0 min** (2.0 min at the tightest target band, 30.1 min at the widest) |
| independent slots/session | **~28.8** → **8,135** over 282 sessions |

That is **4.6× the independent sample count of the +50/−6 objective on the same
calendar**, before any extension to deep history.

**Descriptor separation is weak univariately.** Ranked by mean `pnl_r` spread
between top and bottom quintile: `vol_at_level` 0.057, `node_strength` 0.045,
`position_vs_value` 0.023, `dist_r` 0.014, delta 0.007,
`path_volume_between` 0.005. All Spearman |ρ| < 0.09. Two caveats: this baseline
ran with the **rejected CLV delta proxy** (since replaced by true TBBO delta),
and the entry-state feature group is not built yet. On the evidence of Kyungsub
Lee ([arXiv 2409.14157](https://arxiv.org/abs/2409.14157) — "relying solely on
the price process, directional prediction performance is not substantial; volume
imbalance improves it") the entry-state group is where the signal is expected.

**Half the stop distribution is untradeable at this account size.** At 1.5% risk
($75) on MNQ: an 11.8-point stop sizes to 3.19 contracts and a 33.5-point stop to
1.12, but 59.8 and 96.8-point stops size to 0.63 and 0.39. Stops beyond
**~45 points (`dist_r` ≈ 1.0)** cannot be taken on $5,000. Layer 4 must reject
candidates that cannot be sized to ≥1 contract at max risk.

## First conditional result (walk-forward, purged by one session)

`scripts/features/build_training_matrix.py` joins entry state onto the label
rows (bar stride 15, 1.23M rows); `scripts/features/eval_first_passage_model.py`
fits an XGBoost regression on `pnl_r` per fold, takes the Layer-3 argmax over
candidates per bar, and runs the Layer-4 one-position book. Tradability filter
(stop ≤ 50 pts at 2% of $5,000 on MNQ) keeps 55.7% of rows.

**Conditioning ranks.** On identical bars: random candidate **−0.76 pts/trade**,
model argmax **+2.38** (t = 3.45, n = 4,771).

**The threshold frontier is monotone**, which is the substantive evidence — a
lucky cell does not produce a monotone curve:

| τ (train quantile) | n | trades/session | mean pts | hit rate | t |
|---|---|---|---|---|---|
| 0.50 | 4,589 | 23.1 | 2.82 | 0.367 | 3.98 |
| 0.75 | 3,073 | 15.4 | 3.18 | 0.364 | 3.44 |
| 0.90 | 965 | 4.8 | 8.64 | 0.400 | 4.51 |
| **0.95** | **405** | **2.0** | **16.30** | **0.467** | **5.17** |
| 0.99 | 36 | 0.2 | 37.68 | 0.667 | 3.30 |

At τ = 0.95 all five folds are positive (+17.98 / +17.84 / +14.62 / +4.37 /
+29.49 pts), four with t > 2. Book on one MNQ contract, no costs: **$13,199 net,
$32.59/trade, MDD −$1,308**, 246 long / 159 short.

**Two features had to be removed as regime crutches.** `tgt_level` / `stp_level`
are absolute prices; a tree splitting on 29,426 memorises the price regime and
cannot generalise to an unseen range. Removing them *improved* every metric,
which is the expected sign. `r_value` is causal but session-constant; dropping it
changed little, so it was not acting as a period identifier.

**The edge is entirely order flow.** Ablating the TBBO-derived columns
(`cvd`/`ofi`/`delta`) inverts the result to **negative at every threshold**
(−1.78 to −4.14 pts, t down to −3.17). Structure alone — volume-profile levels,
value area, prior session, returns, volatility, clock — is worse than random.
Top importances are `es_ofi_cum`, `es_cvd_session`, `es_session_range_r`.

Two consequences follow:

1. **Deep 2018→2024 history is not a path to this edge.** The usable dataset is
   the TBBO window, 282 sessions. The sample-count problem is not solved by more
   calendar; it must be solved by information per episode.
2. **The Layer 0 candidate machinery is not yet earning its place.** No `tgt_*`
   or `stp_*` descriptor appears in the top 15. The model is learning *when* to
   trade from auction context, not *which level* to target. That is a real
   negative result for the level-selection thesis as currently featurised.

### Caveats — this is a hypothesis, not a validated edge

- **τ was chosen by inspecting the OOF frontier.** The threshold *value* comes
  from train quantiles, but the *choice of quantile* did not. τ = 0.95 is
  pre-registered for the next run; if it does not hold there, it was selection.
- Three feature configurations were evaluated and the best reported. No Deflated
  Sharpe or PBO has been computed, and ADR-122 notes no cross-run trial ledger
  exists to compute one from.
- MDD −$1,308 is **26% of a $5,000 account** — above a sane operating limit even
  though far below the 50% liquidation level.
- No commission or slippage is modelled.
- 282 sessions, 5 folds, and folds 3–4 carry n = 58 and n = 41.

## Consequences

- The +50/−6 artifacts do not transfer: base OOF AUCs, `labels.fire: 0.40`
  ([ADR-089](ADR-089-dual-binary-argmax-fire.md)), the `lg_*`/`st_*` target
  geometry ([ADR-082](ADR-082-aux-labels-and-head-target-geometry.md)), the aux
  heads ([ADR-084](ADR-084-objective-opt-in-aux-heads.md),
  [ADR-086](ADR-086-side-specific-aux-targets.md),
  [ADR-088](ADR-088-aux-side-rank-decision.md)) and the 15s label contracts are
  all keyed to ±50 / 10m. They are superseded for this objective, not deleted;
  `nq-50pt-econ-lg-st` is unaffected.
- The three-base `tf_oof_meta` stack ([ADR-092](ADR-092-tf-oof-meta-composer.md),
  [ADR-093](ADR-093-tf-oof-meta-joint-search.md)) is not carried forward. The
  bases are aggregations of one tape and their OOF vectors are near-collinear —
  dropping the LSTM moved meta AUC 0.712 → 0.707. One decision grid, multi-scale
  as columns.
- `portfolio_sim.py` becomes the **scoring function**, not a post-hoc report.
  Ranking on `label_implied_pnl` is adverse selection, not merely optimism: the
  gap is largest on high-RV bars, which is exactly what the proxy rewards
  (ADR-050 found high RV → more 50/6 ups). ADR-047's τ=0.65 was chosen on the
  proxy and τ=0.55 dominated it path-realised (+$1,230 vs −$110 median month).
  AUC, precision and F1 demote to diagnostics.
- Sampling must weight by concurrency uniqueness. The session-close time barrier
  *increases* overlap relative to a fixed 10-minute horizon; realized holding
  time, not max horizon, is now the variable that sets effective sample size.
- Variable position sizing needs **calibrated** probabilities, which
  `class_balance` actively destroys. Deferred: fixed fractional risk needs only
  ranking. Isotonic/Platt plus Brier and reliability curves become gates when
  opportunity-scaled sizing is switched on.
- Layer 0 shares substrate with ADR-122 §2 (`src/features/prior_volume_profile.py`,
  prefix `pv_`). ADR-122 builds prior-session VP as model **features**; this ADR
  uses the same structure to generate candidate **exits**. They should share one
  profile implementation when `pv_` lands.

## Alternatives rejected

- **Direct regression head predicting the stop.** There is no label for "the
  correct stop" — obtaining one requires evaluating the outcome under every
  candidate geometry, which *is* this label set. A direct head would first
  compute the surface and then discard all but one number from it.
- **RL over (entry, target, stop).** Directly optimises the objective, but at
  ~8k independent episodes the sample efficiency is hopeless.
- **Keeping the fixed geometry and tuning harder.** Eight one-change Custom Test
  cycles (2026-08-28) produced no convergence, which is the expected result when
  candidate separation is smaller than the confidence interval.

## Not built

Layer 1 **entry-state** features (auction context at the decision bar: developing
profile shape, CVD/OFI, trend, time of day), Layer 3 optimiser, Layer 4 policy
and sizing, Layer 5 evaluation wiring, and the objective YAML.
