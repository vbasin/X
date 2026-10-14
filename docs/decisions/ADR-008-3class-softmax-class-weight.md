# ADR-008: 3-Class Softmax with Inverse-Frequency Class Weights

## Status
Accepted

## Date
2026-05-05 (Phase 17G)

## Supersedes
[ADR-007](ADR-007-detection-2-mae-filter.md) — 2-class detection-2 softmax with `sample_weight=0` masking. The 2-class architecture is replaced by a 3-class softmax where NO_SIGNAL is a full class receiving gradient.

---

## Context

`model_v008_detection` (ADR-007's design) produced `test_f1_macro=0.504` on the signal-only test set but exhibited severe overfit: val_loss climbed monotonically from epoch 1, and the model fired BIG_DOWN or BIG_UP on every bar at near-50/50 split — operationally a coin flip.

Root cause analysis identified two structural problems with the 2-class softmax + `sample_weight=0` design:

1. **Forced binary partition.** `P_down + P_up = 1` by construction. On every bar — including the 88.4% NO_SIGNAL bars — the model must assign probability mass to one of two directional classes. There is no output slot for "no signal"; uncertainty cannot be expressed as a low-confidence abstention. The model is structurally incapable of saying "I don't know."

2. **Zero-weight masking does not suppress NO_SIGNAL bias.** `sample_weight=0` prevents NO_SIGNAL bars from contributing gradient, but those bars are still forward-passed and the predictions are still made. The model never learns that NO_SIGNAL inputs should produce a distinctive output pattern, because NO_SIGNAL produces no training signal at all. The 88.4% of training time spent on zero-weight bars is wasted, and the 11.6% of directional bars must divide all gradient between only BIG_DOWN and BIG_UP — with no neutral anchor to prevent the model from collapsing into alternating between the two under noisy gradient.

Industry literature (DeepLOB, Deep Momentum Networks, TCN/TFT for limit-order books) consistently uses a 3-class label scheme `{down, stationary, up}` because the stationary class provides the neutral anchor: the model can express uncertainty by routing probability into the stationary slot. Removing the stationary class forces every bar to be labelled and creates the binary-partition problem above.

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Status quo (2-class + `sample_weight=0`) | No code change | Forced binary partition; no abstention slot; coin-flip at inference; 88% of training wasted |
| Focal loss (γ=2 + α class-weighting) | Principled hard-example mining; down-weights easy majority | Requires custom Keras loss; more hyperparameters; harder to validate |
| Asymmetric cost matrix | Explicitly penalises false directional signals more than misses | Costly to reverse if calibration is wrong; deferred |
| **3-class softmax + inverse-frequency class weights** (chosen) | Restores neutral class; NO_SIGNAL gets full gradient; standard DeepLOB design; `class_weight` is native Keras | Reverts 2-class architectural decision; requires test-set composition change |

---

## Decision

Adopt a **3-class softmax over `{BIG_DOWN=0, NO_SIGNAL=1, BIG_UP=2}`** — the raw label encoding already emitted by `generate_labels` — with **inverse-frequency class weights** to compensate for the 88/12 class imbalance:

```python
class_weight = {0: 15.0, 1: 1.0, 2: 15.0}
```

The multiplier `15.0` is the approximate inverse frequency of the directional classes at the Oct–Dec 2025 signal rate of ~11.6% (1/0.058 ≈ 17.2; rounded to 15.0 for robustness). Configured as `model.class_weight_directional: 15.0` in `config.yaml`.

**Training-time semantics:**
- Raw labels `{0, 1, 2}` are fed to the model directly — no remapping.
- `model.fit(X, y, class_weight={0:15, 1:1, 2:15}, validation_data=(X_val, y_val))` — no `sample_weight` argument.
- All bars contribute gradient; BIG_DOWN and BIG_UP bars receive 15× more weight than NO_SIGNAL bars.
- Validation loss is computed over all bars (unweighted by default in Keras when `class_weight` is used).

**Test-time semantics:**
- Test set includes all bars `{BIG_DOWN, NO_SIGNAL, BIG_UP}`. `test_accuracy` and `test_f1_macro` are full-population scores.
- Signal counts (`n_signal_test`, `n_signal_train`) are logged separately for monitoring.
- A Precision@τ sweep is computed for BIG_DOWN and BIG_UP across τ ∈ [0.50, 0.95] (step 0.05). This is the operationally meaningful metric: "at confidence τ, what fraction of directional calls are correct?"

**⚠️ Metric incomparability.** Test-set composition changed (signal-only → all-bars). `test_f1_macro` from v009 onward is not directly comparable to v008's signal-only `test_f1_macro=0.504`. Use Precision@τ and per-class f1 scores for cross-version comparison.

**Inference-time semantics (planned, Phase 18):**
- The 3-class softmax produces `[P_down, P_neutral, P_up]` for every bar.
- A directional signal is emitted when `P_class >= τ` for class `∈ {BIG_DOWN, BIG_UP}`.
- Because τ ≥ 0.5 and P_down + P_neutral + P_up = 1, BIG_DOWN and BIG_UP signals are mutually exclusive by construction — no post-processing needed.
- `P_neutral` high → no signal; model is genuinely expressing abstention.

---

## Consequences

- **+** Model can express uncertainty via `P(NO_SIGNAL)` — eliminates forced binary partition.
- **+** All bars provide training signal (gradient through NO_SIGNAL at weight 1.0); 88% of training time is no longer wasted.
- **+** Precision@τ sweep provides actionable threshold selection for Phase 18 inference gate, replacing the ad-hoc `max(P_down, P_up) >= τ` 2-class formula.
- **+** Architecture matches DeepLOB/Deep Momentum Networks standard 3-class design.
- **−** Reverts ADR-007's 2-class decision. The CONSOL escape-hatch concern from ADR-007 is partially mitigated by using non-zero class weight for NO_SIGNAL (the model must *earn* NO_SIGNAL predictions, not just leak probability there for free).
- **−** Metrics not comparable across v008 / v009 boundary. Use Precision@τ for cross-version comparison.
- **−** If `class_weight_directional` is too high, the model may over-predict directional classes (abstention collapse in the opposite direction). If too low, it may collapse to all-NO_SIGNAL. Monitor confusion matrix row for NO_SIGNAL after v009 training; if `P(NO_SIGNAL) ≈ 1.0` on virtually all test bars, increase multiplier from 15 to 20.
- **Constraint**: removing `_remap_to_directional` and `_signal_mask` from `trainer.py` means any rollback to 2-class requires restoring those functions. The functions are documented in ADR-007.

Config keys: `model.num_classes: 3`, `model.class_weight_directional: 15.0`  
Implementation: `src/training/trainer.py` (removed `_remap_to_directional`, `_signal_mask`; added `class_weight` dict), `src/training/evaluate.py` (`CLASS_NAMES = ["BIG_DOWN", "NO_SIGNAL", "BIG_UP"]`; added `_precision_at_tau_sweep`), `src/training/config.py` (`ModelConfig.num_classes=3`, `ModelConfig.class_weight_directional=15.0`), `src/models/attention_lstm.py` (`num_classes` default 3)  
See also: [ML_MODEL.md](../architecture/ML_MODEL.md), [ADR-007](ADR-007-detection-2-mae-filter.md) (superseded)
