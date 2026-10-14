# ADR-072: Opt-in 3×3 expected-cost train loss

## Status
Accepted — implemented 2026-08-27.

## Date
2026-08-27

## Context

Walk-forward 3-class heads (LSTM, TCN, XGBoost, …) minimize **cross-entropy**
on `{down, flat, up}`. `class_balance` only reweights by the **true** class, so
a miss (predict flat when the bar was up) and a wrong-side fire get the same
weight. YAML `kpi.requirement` / `net_expectancy` are **post-fit** scores, not
the train loss.

`nq-50pt-tbm-opt` needs a train-time pay table: reward a correct up/down much
more than a correct flat, charge a **medium** miss, and charge the **same large**
penalty for any incorrect up or down (wrong side or fire on a true flat).

That cannot be a YAML toggle on CE. It is a new loss, gated so other objectives
stay on CE.

## Decision

1. **`custom_optimization: true`** on the Trade Objective opts in. The loader
   requires a full **`cost_matrix`** (3×3, labels `down` / `flat` / `up`, rows =
   true class, cols = predicted class). Omitted or `false` → existing CE path.
2. **Train minimizes expected cost** `E[C] = C[true] · p` over the softmax.
   Lower is better. Correct-class cells can be 0; they are not a bonus above
   zero — they are cheaper than the alternatives.
3. **Wired families** (3-class softmax / `multi:softprob` only):
   Attention-LSTM, TCN, CNN, Transformer, XGBoost, LightGBM.
4. **Not wired**: sklearn RF / logistic / SVM (no custom 3×3 obj);
   `xgboost_ovr` and binary meta gates (not 3-class). Those log and keep their
   default loss. `nq-50pt-tbm-opt` lists only wired families as eligible.
5. **`class_balance` sample weights are skipped** while the matrix is active
   (the table already encodes class importance). Early-stop remains `val_loss`
   (now expected cost). Serving is still argmax of `{direction, confidence}`.
6. **Champion gate / Custom Test report path unchanged** (ADR-071). This is
   only the inner train objective.

Default matrix on `nq-50pt-tbm-opt` (exp-like scale vs TBM 50/6):

| true \ pred | down | flat | up |
|---|---|---|---|
| down | 0 | 8 (miss) | 50 |
| flat | 50 | 3 | 50 |
| up | 50 | 8 (miss) | 0 |

## Consequences

- Changing the table is a YAML edit; changing *who* honors it is a code edit.
- XGBoost/LightGBM use a positive Hessian floor of the expected-cost gradient
  (the exact Hessian can change sign).
- Live pin stays `nq-50pt-econ`. This objective is research-only until a run
  is scored.

## Related

- [ADR-011](ADR-011-focal-loss-directional-f1.md) — legacy trainer focal +
  `val_directional_f1` (not the search/Custom Test plugin path)
- [ADR-044](ADR-044-lstm-class-balance-sqrt.md) — CE sample weights (skipped
  when this loss is on)
- [ADR-014](ADR-014-requirement-target-kpi.md) / [ADR-027](ADR-027-economics-meta-sparse-objective.md) —
  post-fit KPI, not train loss
- `src/training/cost_matrix.py`, `src/models/keras_layers.py` (`ExpectedCostLoss`)
