# ADR-092: OOF-only native-TF dual-head meta composer

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.  
Serving asof (clause 7) superseded by [ADR-103](ADR-103-tf-oof-meta-serving-asof.md).

## Date
2026-08-28

## Context

[ADR-066](ADR-066-native-tf-dual-head-meta.md) declared `tf_oof_meta` on
`nq-50pt-econ-lg-st`: 15s / 1m / 5m bases emit independent `P_long` /
`P_short` for the same TBM +50/−6 / 10m event; a meta XGBoost on
walk-forward OOF plus 1h / 1d context emits `P_final_long` /
`P_final_short`. Search stayed fail-closed until the composer existed.

Native stores are on disk (ADR-075 / 085 / 090 / 091). Custom Test trained
those bases **one store per run** — the fold cache is still a single
parquet tree. A controller wave cannot jointly fit LSTM-15s + XGB-1m +
XGB-5m. The 2026-08-28 base-tuning wave stopped at eight hypothesis
cycles; the next step was meta on existing OOF, not another base retrain.

This is not [ADR-033](ADR-033-output-meta-n-bases.md) (binary
approve/reject on a resolved side) and not `meta_stack` (signed-evidence
LR for 3-class). Dual-binary stacking keeps two heads.

## Decision

1. **Second-stage composer** `src/search/tf_oof_meta.py`. It does not
   train bases. Operator CLI:

   ```bash
   python -m src.search.tf_oof_meta \
     --objective nq-50pt-econ-lg-st \
     --base lstm_15s=20260828T175240Z \
     --base xgb_1m=20260828T200724Z \
     --base xgb_5m=20260828T192637Z
   ```

2. **Decision grid = 15s.** 1m / 5m / 1h / 1d join with causal
   `merge_asof(direction="backward")` and native-bar tolerance (1m = 60s,
   5m = 5 min, 1h = 65 min, 1d = 26 h). A row is kept only when every
   **predictive** base has a last-closed bar inside tolerance. Missing
   1h/1d context fills 0.

3. **Meta learner** = two independent XGBs (`DualBinaryPair`) for
   `p_final_long` / `p_final_short`. Default `class_balance: sqrt`,
   `n_estimators=80`, `max_depth=3`, `learning_rate=0.08`. Fire rule is
   still ADR-089 (argmax and `max(p) ≥ 0.1`).

4. **Features.** Per-base `p_long`, `p_short`, `spread` (`p_long − p_short`),
   `p_max`; Chicago `tod_min` / `session_min` / `dow`; slim 1h/1d
   `regime_v1` pack ([ADR-095](ADR-095-regime-context-fire-aux.md):
   trend stack / slope / gap, ZigZag, ATR/vol ratio, 1h range position,
   1d prior high/low). No raw OHLC, Fib, or floor pivots.

5. **Walk-forward OOF.** `n_folds: 3`, `embargo_bars: 40` (10 minutes on
   15s). Block 0 is warmup (no meta fires). Block *k* is scored by a
   model fit on blocks `0..k-1` with the embargo dropped from the train
   tail. Calendar edges come from the 15s base’s `walk_forward.yaml` OOF
   starts when composing from Custom Test. The last block is reported
   separately as the untouched test. Labels are the 15s dual-binary
   `direction_true`.

6. **Automated search stays fail-closed.** Joint multi-store training is
   not in the fold cache. Custom Test with `mode: none` is unchanged
   (ADR-076). Collided Custom Test dirs with two model folders (e.g.
   `20260828T200039Z`) are refused unless `candidate=` is set.

7. **Report-only artifacts** under
   `research/custom_test/<obj>/<run_ts>/models/tf_oof_meta/` (`oof.parquet`,
   `diagnostics.json`, `meta_estimator.pkl`, `compose.json`). **Not** a
   serving bundle: no `feature_schema.json`, so Backtest/Live will not
   list it. Serving asof of live 15s/1m/5m bases is out of scope.
   **Superseded:** [ADR-103](ADR-103-tf-oof-meta-serving-asof.md) packs
   last-fold bases + `meta.pkl` so Backtest can score the stack. Live
   pin unchanged.

8. **First production compose** uses ranking 1m `20260828T200724Z`,
   economics 5m `20260828T192637Z`, 15s LSTM `20260828T175240Z`. Report
   bundle `20260828T210653Z` (`sqrt`, all three p). Live pin stays
   `nq-50pt-econ`. Optional CLI: `--class-balance ""`, `--skip-features PREFIX`.

## Consequences

- Meta can improve ranking / calibration on the 15s grid without
  retraining LSTM on GPU.
- A later better base OOF is a re-compose, not a new ADR.
- Joint `tf_oof_meta` search (one job, three stores) still needs a
  multi-store fold builder.

## Related

- [ADR-033](ADR-033-output-meta-n-bases.md) — N-base output-meta gate
- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — objective + fail-closed declaration
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — Custom Test one store per run
- [ADR-089](ADR-089-dual-binary-argmax-fire.md) — fire τ=0.1
- [ADR-091](ADR-091-native-tf-1h-1d-htf-context.md) — 1h/1d context files
- Composer: `src/search/tf_oof_meta.py`
- Objective: `config/objectives/nq-50pt-econ-lg-st.yaml`
