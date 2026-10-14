# ADR-095: Slim HTF regime pack, fire 0.40, aux on trees, pinned base HPs

## Status
Accepted — implemented 2026-08-28. Amended by [ADR-104](ADR-104-lstm-session-pmax-and-ui-tau.md) (15s HTF is last-bar context, not the LSTM tape). Live pin unchanged.

## Date
2026-08-28

## Context

Joint LSTM+XGB+meta `20260828T215046Z` (ADR-094) was the first sa.md
stack on disk. Economics were bad: LSTM fired 70% of 15s bars at
τ=0.1 (HR 8.6%, NE −1.20). Breakeven on 50/6 is HR 10.7%. Meta did
not beat the 1m ranking Custom Test.

Four independent defects, measured on that OOF:

1. **Fire τ=0.1 is not an event cut.** Independent BCE + `sqrt`
   class weights put mean p around 0.21. Argmax and `max(p)≥0.1` is
   almost always a trade. The same LSTM at τ=0.40 is n=57k HR 0.123
   NE +0.86; holdout meta at 0.40 is n=5k HR 0.169 NE **+3.45**.
   τ=0.50 holdout collapsed (10 trades) because last-block p_final
   rarely reaches 0.5. ADR-089's 0.1 was chosen so a 0.47 LSTM long
   could fire; it also lets 0.11 vs 0.10 fire.

2. **1h/1d oversaturated, and only the meta saw them.** Composer
   dumped 17 cols × 2 TFs (raw ATR points, EMA distances, SAR, Fib,
   pivots) onto a depth-3 XGB. Short-head gain was 44% HTF. Those
   values are piecewise-constant for 240 15s bars, so the meta
   memorizes a day's tag instead of normalizing the 50-pt target.
   Predictive bases (LSTM tape, 1m/5m trees) never saw vol regime,
   so +50 was the same label in an 80-pt ATR hour and a 200-pt hour.

3. **Joint search ignored tuned HPs.** `iter_tf_oof_bases` copied
   catalog defaults: LSTM `seq_len=128` `epochs=15`; XGB
   `n_estimators=150` `max_depth=4` `lr=0.05`. The 2026-08-28 keepers
   were LSTM 60/10 and 1m XGB 250/12/0.02.

4. **Aux did not teach 50-before-6.** LSTM Huber heads included
   `Time_to_TGT_*` (mean 588s — almost always the 600s censor) and
   1m/5m forward returns. Trees ignored `y_aux` entirely. MAE
   regression does not put a cliff at 6 points.

Rejected: another tree-HPO cycle on the same 15s/1m/5m catalogs;
raising τ only in KPI while leaving OOF `direction` at 0.1; dumping
all 45 HTF columns into the LSTM.

## Decision

1. **`labels.fire: 0.40`** on `nq-50pt-econ-lg-st`. Dual-binary book
   is still argmax and `max(p) ≥ fire`. Library default
   `DUAL_BINARY_FIRE = 0.1` is unchanged for other objectives.
   `dual_binary_fire(objective)` is the SSOT (`labels.fire`, else
   `kpi.tau`, else 0.1). Plugins, composer, and `fire_tau` for
   dual-binary read it. Per-head `confusion_at_fire` stays 0.5.
   Amends [ADR-089](ADR-089-dual-binary-argmax-fire.md) for this
   objective only.

2. **Slim HTF pack `regime_v1`** in `src/features/htf_context.py`.
   Hive files stay 45 cols (ADR-091). Consumers asof-join:

   | TF | Columns |
   |----|---------|
   | 1h | `tr_ema_stack`, `tr_slope_10`, `tr_gap`, `zz_signal`, `rv_atr_ratio`, `rv_vol_ratio`, `sr_pos_in_range_20` |
   | 1d | `tr_ema_stack`, `tr_gap`, `zz_signal`, `rv_atr_ratio`, `sr_prior_high_dist`, `sr_prior_low_dist` |

   No raw OHLC, Fib, floor pivots, SAR distances, or duplicate EMA
   distances. Prefixes stay `h1_` / `d1_`. Causal `merge_asof`
   backward, same tolerances as ADR-092.

3. **The same pack is joined onto 15s / 1m / 5m feature frames** in
   the fold cache when `topology.context` lists 1h/1d. Shared LSTM
   encoder and both tree heads see vol/trend/location as
   conditioning, not as a second-stage afterthought. Cache key
   includes `htf=regime_v1`. 1h/1d stores themselves are not
   predictive bases.

4. **Topology `params` merge over catalog defaults.** YAML pins
   LSTM `seq_len=60` `epochs=10` (15 minutes of 15s tape, matching
   the 10m horizon plus lookback) and the Custom Test keeper trees
   (1m 250/12/0.02, 5m 250/8/0.05).

5. **Aux is the 50/6 teacher set, not a kitchen sink.**
   `labels.aux` is MFE long/short, MAE_b4 long/short, and
   `forward_return_10m`. Dropped censored `Time_to_TGT_*` (600s on
   no-hit) and 1m/5m returns. **Amended [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md):**
   `Time_to_TGT_*` is back; timeout is **NaN**, Huber trains hits only.
   LSTM still Huber-predicts that subset, plus `side_rank`, plus
   new binary `long_stop_first` / `short_stop_first`
   (`MAE_b4 ≥ 6`, BCE weight 0.25) so the 6-pt cliff is a competing
   risk, not a z-scored regression. Dual-binary trees multiply
   class-balance weights by a near-miss factor
   `1 + clip(MFE_side/50)` on negatives. Aux is still not KPI or
   serving.

## Consequences

- Live pin unchanged (`nq-50pt-econ` is 3-class).
- Existing OOF `20260828T215046Z` was written at τ=0.1 with catalog
  HPs and fat HTF — do not promote; re-run joint search.
- GPU 15s jobs must hydrate `timeframe=1h/` and `timeframe=1d/`
  (already in `oci_sync.conf`) or the LSTM trains without HTF while
  local trees have it.
- Custom Test with `topology.mode: none` does not inject HTF
  (no `context` on the overlay). Joint search does.

## Related

- [ADR-089](ADR-089-dual-binary-argmax-fire.md) — library default 0.1
- [ADR-091](ADR-091-native-tf-1h-1d-htf-context.md) — 45-col hive
- [ADR-092](ADR-092-tf-oof-meta-composer.md) — meta still XGB on OOF
- [ADR-084](ADR-084-objective-opt-in-aux-heads.md) — YAML aux opt-in
- [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md) — Time_to_TGT NaN on no-hit
- [ADR-094](ADR-094-15s-lstm-temporal-base.md) — 15s is LSTM
- Objective: `config/objectives/nq-50pt-econ-lg-st.yaml`
