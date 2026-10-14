# ADR-007: Detection-2 Architecture with MAE-Filtered Labels

## Status
Superseded by [ADR-008](ADR-008-3class-softmax-class-weight.md) (2026-05-05) — 2-class design reverted to 3-class softmax with inverse-frequency class weights after v008 coin-flip diagnosis.

## Date
2026-04-30 (Phase 17F)

## Supersedes
[ADR-004](ADR-004-adaptive-labels.md) — adaptive ATR labelling within the 3-class softmax framework. The 3-class framework itself is replaced by detection-2; ADR-004's adaptive-threshold mechanism is no longer used.

---

## Context

Phase 17E delivered `model_v007_0dte` with `test_f1_macro=0.321` on a 3-class softmax over {BIG_DOWN=0, CONSOLIDATION=1, BIG_UP=2}, trained with inverse-frequency class weights. Closer inspection of the per-class scores revealed that the macro-f1 number was almost entirely driven by the CONSOLIDATION class (f1_CONSOL=0.820) on a test set that was 92 % CONSOL by composition. The directional class scores were f1_BIG_DOWN<0.05 and f1_BIG_UP=0.144 — i.e. the model did not actually predict direction on bars where direction matters.

Two structural problems with the 3-class + class-weight design surfaced:

1. **CONSOL escape-hatch.** With `class_weight={0:w0, 1:0, 2:w2}` (CONSOL zeroed-out for the gradient), the softmax is still a 3-vector. The model can output `[0.1, 0.8, 0.1]` and pay zero gradient cost because CONSOL is excluded from the loss. The optimisation surface rewards routing uncertainty through the CONSOL slot rather than expressing it as a low-confidence directional probability. The accuracy metric still sees these CONSOL outputs as "correct" on the 92 % of test bars that are genuinely CONSOL, producing a misleadingly high `test_accuracy`.

2. **Spike-and-retrace mislabelling.** The peak-excursion label `up_move >= threshold` marks a bar as BIG_UP whenever the price *touches* threshold within the forward window — even if it then fully retraces. A trader entering at the labelled bar and exiting on a fixed stop would have been stopped out. The model is therefore being trained to predict "moves that touch the target" rather than "moves that a trader can capture." The two are not the same and the gap costs precision at inference time.

ADR-004's adaptive ATR threshold mitigated class imbalance (raising the directional rate from ~2 % to ~45 % of bars in the v006 era) but did nothing about either problem above. The CONSOL slot was still in the softmax, and the labels still had no MAE constraint.

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Status quo (3-class + adaptive labels + class weights) | No code change | CONSOL escape-hatch persists; metrics inflated by majority class; spike-and-retrace still mislabelled |
| 3-class with non-zero CONSOL weight | Forces gradient through all three classes | Reintroduces severe imbalance; model collapses back to majority CONSOL prediction |
| Regression on signed forward return | Continuous target; no class boundary issues | Loss landscape rewards predicting the conditional mean (~0); needs custom loss to extract directional signal; harder to gate at inference |
| Two binary heads (P_up, P_down independent) | Symmetric; can both be high in volatile regimes | Two losses to balance; output not a probability simplex; gating logic unclear |
| **Detection-2 (2-class softmax) + MAE-filtered labels** (chosen) | Removes CONSOL slot from softmax; MAE constraint matches trader-capturable moves; uncertainty expressed via `max(P_up, P_down) < τ` rather than a third class | Requires sample_weight masking at training time; test set must drop NO_SIGNAL bars to avoid metric inflation; new `mae_tolerance` hyperparameter |

---

## Decision

Adopt a **2-class softmax over {BIG_DOWN=0, BIG_UP=1}** with NO_SIGNAL bars masked at training time via `sample_weight=0`, and use a **Maximum Adverse Excursion (MAE) filter** for label generation:

```python
up_move   = max(price[t+1 .. t+60]) - price[t]
down_move = price[t] - min(price[t+1 .. t+60])

if up_move   >= big_move_threshold AND down_move < mae_tolerance: → BIG_UP   (raw 2)
if down_move >= big_move_threshold AND up_move   < mae_tolerance: → BIG_DOWN (raw 0)
else                                                              : → NO_SIGNAL (raw 1, masked)
```

Defaults: `big_move_threshold=50` NQ points, `mae_tolerance=5` NQ points.

**Training-time semantics:**
- Labeller emits `{0=BIG_DOWN, 1=NO_SIGNAL, 2=BIG_UP}`. The trainer remaps `{0,1,2}→{0,0,1}` for the model and builds `sample_weight = (y_raw != 1).astype(float32)`. NO_SIGNAL bars contribute zero gradient.
- `model.fit(X, y, sample_weight=sw, validation_data=(X_val, y_val, sw_val))` — no `class_weight` argument.
- The validation loss is also computed on the masked weights, so early-stopping tracks the directional loss only.

**Test-time semantics:**
- NO_SIGNAL rows are dropped entirely from the test set before metric computation. `test_accuracy` and `test_f1_macro` are therefore signal-universe scores only — they answer the question "given that the bar is BIG_UP or BIG_DOWN, does the model predict it correctly?"

**Inference-time semantics (planned, Phase 18):**
- The 2-class softmax produces `[P_down, P_up]` for every bar. A directional signal is emitted only when `max(P_down, P_up) >= τ` (target τ ≈ 0.6). Bars below the threshold are operationally NO_SIGNAL — equivalent to the masked NO_SIGNAL training label, but derived from model confidence rather than a third output class.

The MAE filter formalises the "clean move" intuition: a labelled BIG_UP must reach +50 pts in the next 5 minutes *and* never have moved more than 5 pts against the entry direction along the way. Spike-and-retrace patterns that briefly touch +60 pts and fade back to flat are correctly classified as NO_SIGNAL because their `down_move` exceeds tolerance. The forward-window-spreading nature of peak-excursion labelling is preserved (a clean buildup that culminates in a +50pt move 60 bars later still labels the buildup bars as BIG_UP), so masking does not strip out the buildup context.

---

## Consequences

- **+** No CONSOL escape-hatch. Uncertainty must be expressed as a low directional probability gap; the model cannot route it through a third class to avoid loss.
- **+** Metrics report only what matters operationally. `model_v008_detection`: signal-universe `test_f1_macro=0.504` (BIG_DOWN f1=0.560, BIG_UP f1=0.448) vs v007's directional f1 < 0.10 on the same data window.
- **+** MAE-filtered labels match trader-capturable moves. Precision at any inference threshold τ now reflects "moves the trader actually catches" rather than "moves that happened to print in the window."
- **+** Inference gating becomes natural. Phase 18's confidence gate `max(P_up, P_down) >= τ` operates directly on the softmax output without a CONSOL post-processing step.
- **−** Two new hyperparameters interact (`big_move_threshold`, `mae_tolerance`). Both are now in the HPO sweep (`scripts/hpo_search.py`).
- **−** Training set shrinks materially. With defaults, the directional rate on Oct–Dec 2025 is ~11.6 % (10,836 of 93,632 sequences). A loose MAE (`mae_tolerance=10`) admits more buildup-and-rise patterns; tightening (`mae_tolerance=3`) approaches strict trend-only labelling. This trades sample size against label noise.
- **−** Test set is also smaller (signal-only, 1,308 sequences for v008). Metric variance is higher than on a full 10k-row test set; walk-forward validation across multiple folds is needed to confirm single-split numbers.
- **−** Reported accuracy is no longer comparable across the v007/v008 boundary. v007's `test_acc=0.86` and v008's `test_acc=0.51` describe different test populations. Use only signal-universe `f1_macro` for cross-version comparisons.
- **Constraint**: any change to `big_move_threshold`, `mae_tolerance`, or `forward_windows_bars` invalidates the cached labels — labels are regenerated at training time from the Parquet `_close` series, so the change is automatic on the next training run, but every previously trained checkpoint becomes incomparable.
- **Constraint**: the MAE filter as currently specified is asymmetric in the *favourable* direction. A +60 pt spike that fully retraces to entry still satisfies `down_move = entry − min(future) ≥ 0` and labels BIG_UP because the trade was profitable at peak; it only fails the filter if the price first dips below entry by `mae_tolerance`. To reject post-target retraces, replace `min(future)` with a bidirectional cumulative travel filter using `cummax`/`cummin` — deferred to a future ADR if exit-strategy tuning warrants it.

Config keys: `labels.threshold_points: 50`, `labels.mae_tolerance: 5`, `model.num_classes: 2`  
Implementation: `src/training/labels.py` (`_classify_detection_2`, `generate_labels`), `src/training/trainer.py` (`_remap_to_directional`, `_signal_mask`), `src/training/evaluate.py` (`CLASS_NAMES = ["BIG_DOWN", "BIG_UP"]`)  
See also: [ML_MODEL.md](../architecture/ML_MODEL.md), [PHASE_17_RESULTS.md §17F](../changelog/phases/PHASE_17_RESULTS.md), [ADR-004](ADR-004-adaptive-labels.md) (superseded label-design history)
