# ADR-043: cuDNN LSTM + single-base train cap on GPU

## Status
Accepted — implemented 2026-08-21.
Amends [ADR-019](ADR-019-user-override-manual-candidate-spec.md): Custom Test
`max_rows` is opt-in (default **off** = full train). Automated search still
uses catalog `search_max_rows` (100k for LSTM/TCN). Tree / CPU manual specs
stay full-matrix by default.
Follow-on: [ADR-044](ADR-044-lstm-class-balance-sqrt.md) (`class_balance=sqrt`
on the LSTM plugin + Custom Test params).

## Date
2026-08-21

---

## Context

Custom Test `20260821T074303Z-custom_test-5afe7c21` (`attention_lstm` on
`nq-50pt-econ`, 3 expanding folds, ~1.28M last-fold train bars) sat on the A10
for 68 minutes at ~70% SM / 6.4 GB VRAM and was still in fold 0/1. Host CPU was
~2% of 30 vCPUs. The job had not failed.

Four independent costs stacked:

1. **`LSTM(..., unroll=True)`** blocked NVIDIA's cuDNN LSTM kernel. Seq_len 128
   became ~128 sequential GEMMs per layer per step (~14 ms/step at batch 32).
2. **Batch 32**, 15 epochs, no early stopping → ~1.2M train steps.
3. **Manual-mode `max_rows` omitted = full matrix** (ADR-019). Catalog
   `search_max_rows=100_000` never applied. `_run_one` also nulled any cap when
   topology was `meta_stack` / `output_meta`, even for a **single** base.
4. **Same Keras weights + no RobustScaler across folds** (TCN had already
   learned this poisons later OOF). Random 100k row picks then `make_windows`
   would also treat gapped bars as consecutive — invalid for LSTM.

Projected wall at the measured 70 steps/s: **~5 hours**. TCN on a similar
3-fold window was 25 minutes.

## Decision

1. **`unroll=False`** in `build_attention_lstm` so TF uses cuDNN LSTM
   (`recurrent_dropout` stays 0; input dropout is fine).
2. **Catalog `batch_size=128`**, Keras `EarlyStopping(patience=10)` like TCN,
   Sequence-fed `fit` unchanged (A10 `_EagerConst` safety).
3. **Per-fold `RobustScaler` + rebuild** in `AttentionLSTMPlugin` (same
   contract as TCN). Last-fold scaler is stored on `FittedModel.meta`.
4. **Keep `search_max_rows` when topology has ≤1 base.** Multi-base
   output_meta / meta_stack still drop the cap so bases share train rows.
5. **Sequence subsample is last-N contiguous train bars**, not a random
   gapped draw. OOF is never subsampled.
6. **Custom Test compile** does **not** auto-apply catalog `search_max_rows`.
   Default is full train (`max_rows` omitted / off). The params panel exposes
   an on/off **max_rows** switch; on uses last-N train bars (suggested 100k).
   Automated catalog search still caps GPU sequence families at
   `search_max_rows`.

Expected wall for this Custom Test after the change: **~8–20 min** (100k
last-N train × cuDNN × batch 128) vs ~5 h. Full-matrix LSTM (explicit
`max_rows: null`) should land in the TCN ballpark (~30–50 min).

## Consequences

- A Custom Test LSTM board no longer silently trains 1.2M unrolled windows.
- Full-fidelity LSTM is still available via `max_rows: null` on the slot.
- Automated multi-base waves (xgb+tcn) still train uncapped so OOF joins
  stay aligned on train if anything still used train rows.

## Rejected

- Leaving `unroll=True` (XLA-friendly but 10× slower on seq 128).
- Random 100k draws for sequence families (windows would not be 5s-consecutive).
- Changing ADR-019's full-matrix default for trees / explicit manual specs.
