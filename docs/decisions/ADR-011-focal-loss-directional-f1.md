# ADR-011 — Focal Loss + Directional F1 EarlyStopping Monitor

**Date**: 2026-05-19  
**Status**: Accepted  
**Supersedes**: class-weight-only approach from [ADR-008](ADR-008-3class-softmax-class-weight.md) (augments, does not replace)

---

## Context

v010 3-fold walk-forward training failed its targets (BU P@τ=0.80 > 45%, BD F1 > 0.05).  
Post-mortem on fold results:

| Fold | Epochs run / best | BD F1 | BU F1 | BU P@0.8 |
|------|-------------------|-------|-------|----------|
| 1 | 7 / ep2 | 0.023 | 0.032 | n/a (0 signals) |
| 2 | 10 / ep5 | 0.045 | 0.050 | 78.6% (n=14) |
| 3 | 9 / ep4 | 0.090 | 0.077 | 4.9% (n=370) |

Two root causes were identified:

### Root cause 1 — val_loss monitor peaked at epoch 2

The training set is ~92% NO_SIGNAL. Cross-entropy loss is dominated by this majority class: once the model learns to predict NO_SIGNAL confidently (p_NS ≈ 0.97), val_loss reaches its floor and `EarlyStopping(patience=5)` fires within 2–3 epochs. The directional classes (BIG_DOWN, BIG_UP) have not converged at this point and never get a chance to. The best checkpoint is restored at epoch 2 — the worst possible point for directional predictions.

### Root cause 2 — class_weight=15 insufficient

With `class_weight={0:15, 1:1, 2:15}` the effective gradient ratio is BIG_DOWN:NO_SIGNAL:BIG_UP = 15:1:15. But the raw class frequency ratio is roughly 1:12:1 (directional bars ≈ 8% of training set). After weighting, the effective ratio becomes 15:12:15 — only a 1.25× directional advantage. The model still takes the easy path of over-predicting NO_SIGNAL.

---

## Decision

### 1. Replace `sparse_categorical_crossentropy` with `focal_loss(γ=2.0)`

Focal loss (Lin et al., 2017, RetinaNet) modulates cross-entropy by `(1 − p_t)^γ`:

$$\mathcal{L}_{focal} = -\sum_i (1 - p_{t,i})^\gamma \log(p_{t,i})$$

At γ=2.0, a NO_SIGNAL bar predicted with p_NS=0.97 contributes only `(1−0.97)² = 0.09%` of the normal CE loss. A misclassified BIG_UP bar with p_BU=0.05 contributes `(1−0.05)² = 90%` of normal CE. The loss landscape becomes dominated by hard examples — exactly the rare directional bars that matter.

Implementation: `focal_loss(gamma)` in `src/models/attention_lstm.py` uses `tf.gather_nd` to extract true-class probabilities from logits for sparse integer labels, applying the modulation factor before computing log loss.

### 2. Replace `val_loss` monitor with `val_directional_f1`

`val_directional_f1 = (F1_BIG_DOWN + F1_BIG_UP) / 2`

This metric directly measures the quantity we care about. EarlyStopping now fires when directional F1 stops improving — not when NO_SIGNAL loss stops improving. The best checkpoint saved by `ModelCheckpoint` is also selected on this metric.

Implemented via `_DirectionalF1Callback(tf.keras.callbacks.Callback)` in `trainer.py`, placed first in the callbacks list so the metric is in `logs` when `EarlyStopping.on_epoch_end` runs.

### 3. Raise patience and class weight

| Parameter | v010 | v011 | Rationale |
|-----------|------|------|-----------|
| `class_weight_directional` | 15 | **40** | Focal loss already suppresses NS; combined with CW=40 the gradient budget is overwhelmingly directional |
| `early_stop_patience` | 5 | **12** | val_directional_f1 is noisier than val_loss; needs room to recover from LR-reduction plateaus |
| `lr_reduce_patience` | 3 | **5** | Matching rhythm: LR halved at ep5 if no improvement, stop at ep12 if still no improvement |
| `epochs` | 30 | **50** | Safety ceiling — EarlyStopping expected to fire ~ep28; 50 is unreachable in practice |

### 4. Add `TrainingConfig.intraop_threads` for parallel fold execution

`intraop_threads: int = 0` (default: TF uses all available cores). When set to N > 0, `trainer.main()` calls `tf.config.threading.set_intra_op_parallelism_threads(N)` before any TF operations. This allows 3 folds to run in parallel (3 × 2 cores = 6 vCPUs) without core contention. The value lives in `config.yaml` — no external env vars needed.

---

## Consequences

**Positive**
- EarlyStopping now tracks the signal we care about — model cannot "escape" by collapsing to NO_SIGNAL
- Focal loss removes the need to fight class imbalance purely through class_weight; the two mechanisms compound
- Thread budget is self-documenting in `config.yaml` and applied automatically by the trainer
- Fold 2's v010 result (BU P@0.8 = 78.6%) confirms the features contain real directional signal; v011 should unlock it across all folds

**Negative / risks**
- Focal loss with γ=2.0 is an untested hyperparameter for this dataset; γ may need tuning if directional F1 does not improve
- `val_directional_f1` is computed via a full forward pass on the validation set each epoch (+3% overhead per epoch); acceptable given the ~9 min/epoch wall time
- With `class_weight=40` and focal loss combined, the model may over-predict directional bars (false positive rate increase); precision@τ sweep will detect this

**Rollback**
Set `use_focal_loss: false` and `class_weight_directional: 15` in `config.yaml`; revert `trainer.py` callbacks to `monitor="val_loss"`. No schema or data changes.
