# ADR-103: tf_oof_meta last-fold serving asof for Backtest

**Date:** 2026-08-29  
**Status:** Accepted — implemented 2026-08-29. Live pin unchanged.

Amends [ADR-092](ADR-092-tf-oof-meta-composer.md) clause 7 (serving asof
was out of scope). Related: [ADR-100](ADR-100-backtest-native-tf-bundle-store.md)
(single-store last-fold hive), [ADR-102](ADR-102-gpu-htf-context-hydrate.md)
(OOF-only lab listing until this pack).

## Context

Joint search `20260829T182514Z` wrote
`models/tf_oof_meta__lstm_15s_xgb_1m_xgb_5m/` as `meta_estimator.pkl` +
`compose.json` (`serving: false`) and no `feature_schema.json`.
`is_serving_bundle` was false, so WebUI Backtest would not load the
stack. ADR-033 `output_meta` scores every base on the **same** 5s
matrix. This stack cannot: 15s LSTM, 1m XGB, and 5m XGB each have
their own hive, plus 1h/1d HTF on the meta (and on the trees).

The meta is two XGBs (`DualBinaryPair`), not Keras.

## Decision

1. **Lab serving layout** (same contract as `meta_stack`):
   `feature_schema.json` + `meta.pkl` + `meta.json` + copied
   `bases/<id>/` last-fold artifacts. `inference_policy.json` carries
   `labels.fire` (0.40) and 08:30–11:00 CT so the Backtest τ slider
   fills. `compose.json` `serving: true`.
2. **Load** (`src/serving/bundle.py`): `meta.json` topology
   `tf_oof_meta` is not an ADR-033 approve/reject gate.
3. **Predict** (`predict_canonical`): score the 15s frame with the
   LSTM; `load_feature_frame` 1m/5m for the same window (HTF / `svp_*`
   attached as trained); causal asof (`align_native_oof`); meta
   `predict_pair`; fire = argmax and `max(p) ≥ labels.fire`.
4. **Decision grid** remains 15s. `infer_bundle_feature_store` for
   `family=tf_oof_meta` is 15s.
5. **Pack** existing `20260829T182514Z` in place; joint-search
   `persist_bundle` writes this layout next time. Not promoted; Live
   dropdown stays staging/registry.

Rejected: Keras meta (it is not a net). Rejected: OOF-parquet replay
as the only Backtest path for this folder — that is still available
if the operator picks `oof.parquet`; selecting the model folder runs
last-fold serving asof, same as `lstm_15s` / `xgb_1m` bundles.

## Consequences

- `#/backtest` lists
  `…/20260829T182514Z/models/tf_oof_meta__lstm_15s_xgb_1m_xgb_5m`.
- Scoring a long window runs the GPU LSTM last-fold graph on CPU
  (slow). Narrow the date range.
- LSTM last-fold is still 23 hive cols (no HTF) until a GPU job with
  ADR-102 hydrate + ADR-101 Globex tape is retrained. The pack does
  not retrain.
- Live pin remains `nq-50pt-econ`.
