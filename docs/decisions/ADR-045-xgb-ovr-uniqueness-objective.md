# ADR-045: XGB one-vs-rest + train uniqueness objective (`nq-50pt-xgb-unique`)

## Status
Accepted — implemented 2026-08-21.
Does **not** replace `nq-50pt-econ` (still the live pin). Complementary
research objective: trees, not 3-class sequence DL.
**Feature view superseded in part by
[ADR-046](ADR-046-causal-tbm-topgroups-proof.md)** (`tbm_topgroups` replaces
`momentum_edge` on this objective after the first unique wave missed CS).

## Date
2026-08-21

---

## Context

TBM +50/−6 / 10m on 5s L1 bars is a rare-event, overlapping-label, asymmetric
payoff problem (~87% flat). 3-class LSTM/TCN softmax learned **activity**
(AUC ~0.70) and not **side**. Registry champion is already XGB
(`xgb_tbm_topgroups`, NE 26.9 after 1 pt @ τ=0.8).

Needed a declarative objective that encodes the policy: one-vs-rest trees,
train uniqueness, calendar walk-forward, 1 pt cost, meta on proposed trades,
L1-only data (no paid Databento / no MBP-10).

## Decision

1. **New objective** `config/objectives/nq-50pt-xgb-unique.yaml`. Same TBM
   50/6/10m and session as econ. `cost_per_trade: 1.0`. Eligible family
   originally **`xgboost_ovr` only** with `momentum_edge`.
   [ADR-046](ADR-046-causal-tbm-topgroups-proof.md) added 3-class `xgboost`
   and switched the view to `tbm_topgroups` (`cs_/rv_/vp_/iv_`).
   `primary_meta` logistic on
   `regime_core` with 120-bar embargo. Calendar WF 2025-05-01→2026-06-30,
   2-month train/test (6 folds), 120-bar train purge at OOF boundary.
2. **`eval.train_uniqueness_bars: 24`** (2 minutes). Train keeps ≤1 bar per
   24×5s; **OOF is every bar**. Full 10-minute uniqueness (~40 samples/day)
   is too thin for XGB; 2 minutes cuts overlap 12× and keeps thousands of
   rows. Schema field on `eval`.
3. **Catalog `xgboost_ovr`**: `plugin_family: xgboost`, `head: ovr`. Two
   binary XGBs (up vs rest, down vs rest) expose 3-class `predict_proba`
   (`OvrDirectional`, joblib `model.pkl`). `dir_threshold` default 0.55 so
   a side can fire without beating a 3-way flat mass.
4. Scoring τ **0.65** (requirement `precision_at_tau`) — R≈8.3 does not need
   0.8 to print NE≥20; 0.8 stays a target diagnostic @ 0.9.

Rejected: making 3-class LSTM the primary; uniqueness=120 (too few rows);
changing `config.yaml` `objective_id` (live pin stays econ until promote).

## Consequences

- Search: `python -m src.search.controller --objective nq-50pt-xgb-unique --no-ensemble`
- GPU reconstructs `family=xgboost` + `spec_params.head=ovr`.
- Serving loads `model.pkl` with `predict_proba` → 3 columns (existing
  canonical path).
- OOF-every-bar scoring is **not** a live book — see
  [ADR-047](ADR-047-inference-uniqueness-open-policy.md) (10m cooldown).
