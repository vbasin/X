# ADR-066: Native-timeframe dual-head meta (`nq-50pt-econ-lg-st`)

## Status
Accepted — objective declared 2026-08-26. OOF composer
[ADR-092](ADR-092-tf-oof-meta-composer.md). Joint search
[ADR-093](ADR-093-tf-oof-meta-joint-search.md) trains YAML bases per
store then meta. 15s base is LSTM ([ADR-094](ADR-094-15s-lstm-temporal-base.md)).
Live pin unchanged.

## Date
2026-08-26

## Context

`nq-50pt-econ` is a 5s 3-class primary + trade/no-trade gate. The next
economics objective needs independent long and short probabilities for the
same TBM +50/−6 / 10m event, from native 15s / 1m / 5m features, with 1h/1d
as context only. Precision is not a gate.

This is not ADR-048 (P(side | directional)). Dual-binary is P(long event)
and P(short event) on every decision timestamp.

## Decision

1. New objective `config/objectives/nq-50pt-econ-lg-st.yaml`. Same TBM 50/6
   session/costs/KPI economics as `nq-50pt-econ`. No `precision_at_tau` in
   requirement or target. `n_signals` floor kept.
2. `labels.encoding: dual_binary`. Topology `tf_oof_meta`: 15s
   **Attention-LSTM** (temporal encoder on the tape,
   [ADR-094](ADR-094-15s-lstm-temporal-base.md)), 1m and 5m `xgboost_ovr`;
   context 1h, 1d; meta XGBoost on walk-forward OOF emits `P_final_long` /
   `P_final_short`.
3. `search.input: native_tf`. Per-TF parquets are on disk. Automated
   search still fail-closes (`assert_tf_oof_meta_ready`). OOF-only
   compose is `python -m src.search.tf_oof_meta` (ADR-092).
4. Walk-forward: 3 expanding-calendar folds, 4-month train / 4-month test.
   Meta `n_folds: 3`.
5. Dual-binary labels are the 5s TBM series on the 15s grid. Both columns
   are written at every timestamp: BIG_UP → `y_long=1, y_short=0`; BIG_DOWN
   → `y_long=0, y_short=1`; else both 0. Shared target for 15s/1m/5m bases.

## Consequences

- Live pin stays `nq-50pt-econ`.
- Native 15s 1-bar store is on disk ([ADR-075](ADR-075-native-tf-15s-bar-features.md)).
  Native 1m store is on disk ([ADR-085](ADR-085-native-tf-1m-features.md)).
  Native 5m store is on disk ([ADR-090](ADR-090-native-tf-5m-features.md)).
  Native 1h / 1d HTF-context stores are on disk ([ADR-091](ADR-091-native-tf-1h-1d-htf-context.md)).
  Hive dirs: [ADR-070](ADR-070-native-tf-hive-layout.md). Automated
  search still fail-closed; OOF-only composer: [ADR-092](ADR-092-tf-oof-meta-composer.md).
- Candle substrate: `databento.tbbo_15s` / `tbbo_5m` / `tbbo_1h` / `tbbo_1d` (ADR-067).

## Related

- [ADR-027](ADR-027-economics-meta-sparse-objective.md)
- [ADR-033](ADR-033-output-meta-n-bases.md)
- [ADR-045](ADR-045-xgb-ovr-uniqueness-objective.md)
- [ADR-048](ADR-048-binary-side-head-rejected.md)
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — NQ 15s / 5m / 1h / Globex 1d caggs
- [ADR-070](ADR-070-native-tf-hive-layout.md) — native TF hive dirs
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s 1-bar parquet
- [ADR-085](ADR-085-native-tf-1m-features.md) — 1m so_/cs_/ps_/vw_/rv_/vp_/ms_ (no options IV)
- [ADR-090](ADR-090-native-tf-5m-features.md) — 5m cs_/vp_/vw_/ps_/zz_/vf_/rv_ (VWAP ±2σ)
- [ADR-091](ADR-091-native-tf-1h-1d-htf-context.md) — 1h/1d HTF context cs_/tr_/ps_/zz_/rv_/sr_
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — Custom Test 15s single-base, dual-binary head
- Objective: `config/objectives/nq-50pt-econ-lg-st.yaml`
- [ADR-092](ADR-092-tf-oof-meta-composer.md) — OOF-only dual-head composer
