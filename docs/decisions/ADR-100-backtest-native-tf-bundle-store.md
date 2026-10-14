# ADR-100: Backtest native-TF last-fold bundles on their own hive

## Status
Accepted — implemented 2026-08-29.

## Date
2026-08-29

## Context

`nq-50pt-econ-lg-st` trains **1m** and **5m** XGB on native-TF hives
(`so_` / `vp_` / `svp_*` / `h1_*` / `d1_*`). The 15s LSTM is the decision
grid. WebUI Backtest of a last-fold **bundle** (not the OOF parquet) loaded
features via `default_parquet_dir(objective)` → **15s hive** because
`search.input=native_tf` and `decision_timestep=15s`.

On 2026-02-13 that is 24 15s columns vs 84 on `xgb_1m` (4 present, 80
missing). `predict_canonical` reindexes missing cols to NaN; DualBinaryPair
then emits `p_max ≈ 0.07`. Dual-binary `direction` is already `flat` before
the UI confidence gate. Unchecking τ admits directional rows only — there
are none. Same for `xgb_5m` (77-col schema including fold-cache `svp_*`).

This is not “the 1m model has no 2026 skill.” OOF on its own 1m tape has
112 fires @ 0.40 (106 in 2026). Backtest was scoring the wrong tape.
Serving asof of live 15s/1m/5m bases is still not wired ([ADR-093](ADR-093-tf-oof-meta-joint-search.md)).

## Decision

1. Infer the bundle store from `manifest.native_tf` / `candidate_id`
   (`xgb_1m` → `1m`) / schema prefixes (`so_` → 1m, `svp_` → 5m).
2. Load that hive (`load_feature_frame`), then attach `h1_*`/`d1_*` and
   `svp_*` when the schema includes them (same as fold cache).
3. Dual-binary predict uses objective `labels.fire` (0.40 on lg-st), not
   the library 0.1 default and not bundle `dir_thr`. Unchecking τ only
   skips `conf ≥ τ`; it does not turn every bar into a trade.
4. Live snapshot stays on 5s until serving asof exists.
5. **Historical Backtest of a Custom Test model folder** replays
   `models/<id>/oof.parquet` on that store’s hive (walk-forward, as-of).
   Last-fold weights are not scored on train dates. OCO/TBBO still uses
   the bundle (needs serving predict).

## Consequences

- Backtest of Custom Test `xgb_1m` / `xgb_5m` uses 1m / 5m closes for
  path PnL, not 15s `_close`.
- Picking the **model folder** is now OOF replay (as-of), same as
  `oof_xgb_1m.parquet`. 2026H1 1m: 106 OOF fires align, 80 fills
  (26 skipped while already in a trade).
- Unchecking confidence still does nothing if you override `parquet_dir`
  to the 15s hive.
- Live still cannot score 1m/5m on the 5s snapshot.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md), [ADR-076](ADR-076-custom-test-store-head-labels.md)
- [ADR-089](ADR-089-dual-binary-argmax-fire.md), [ADR-095](ADR-095-regime-context-fire-aux.md)
- `src/backtest/weights_score.py` (`infer_bundle_feature_store`, `load_bundle_feature_frame`)
