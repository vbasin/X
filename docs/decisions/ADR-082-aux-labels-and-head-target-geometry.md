# ADR-082: Dual-binary aux path labels + per-head target geometry

## Status
Accepted — implemented 2026-08-28. **Amended by [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md):**
`Time_to_TGT_*` is NaN when the 10m window never hits ±50 (was horizon 600s).
**Amended by [ADR-084](ADR-084-objective-opt-in-aux-heads.md):**
aux heads are YAML opt-in (`labels.aux`), not auto-on for every dual-binary
LSTM that sees the parquet columns. **Amended 2026-08-28 (serving pack):**
Live/Backtest `predict_canonical` splits `lg_*`/`st_*` into last-bar
`long_ctx`/`short_ctx` for multi-input Keras graphs (was a single
`(N, T, F)` window — `Layer "attention_lstm" expects 3 input(s)`).
**Amended by [ADR-086](ADR-086-side-specific-aux-targets.md):** named
`long_forward_return_*` / `short_forward_return_*` no longer share the
same y (`short` trains on `−z(return)`); shared `MAE_before_target_or_MFE`
Keras heads dropped. Live pin unchanged.

## Date
2026-08-28

## Context

15s dual-binary LSTM (`nq-50pt-econ-lg-st`) has two jobs on the same 10-minute
TBM event: whether a qualifying move is pending, and which side. Shared-encoder
`p_long` / `p_short` (ADR-080) collapsed to nearly identical probabilities
because the 13 1-bar columns and a single embedding do not give either head a
side-specific scale or a dense path-quality target.

The 50-point target is a different fraction of realised range on a quiet vs
an FOMC day, and of index level over the contract. Feeding unsigned reductions
(OFI, CLV, size deltas) through a shared LSTM cannot express “+50 is 0.7 ATR
here” vs “−50 is 0.7 ATR here” as two different questions.

## Decision

1. **Auxiliary labels** on 5s and 15s label parquets (computed on the **5s**
   close path, then subsampled to 15s — same path as TBM). Columns:

   | Column | Unit | Meaning |
   |--------|------|---------|
   | `MFE_long_10m` / `MFE_short_10m` | points | Peak favorable close excursion in the 10-minute horizon |
   | `MAE_b4_TGTorMFE_long_10m` / `_short_10m` | points | Peak adverse close excursion until the earlier of that side's first target touch and that side's MFE bar |
   | `forward_return_1m` / `_5m` / `_10m` | points | Close-to-close delta |
   | `MAE_before_target_or_MFE` | points | MAE of the side with the larger MFE |
   | `Time_to_TGT_long` / `_short` | seconds | First ±threshold touch; **NaN if never touched** (not the window cap — [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md)) |

   Primary `y_long` / `y_short` are unchanged. Aux is **not** the serving
   contract.

2. **Multi-task LSTM (dual-binary).** Each directional trunk predicts its
   MFE / MAE-before / time-to-target **and** the shared path labels
   (`forward_return_*`, unsigned MAE) under unique Keras names
   (`long_forward_return_1m` vs `short_forward_return_1m`; **same y in
   this ADR**, superseded by [ADR-086](ADR-086-side-specific-aux-targets.md)
   which trains short on `−z(return)` and drops the shared MAE heads).
   Loss is Huber on **per-fold z-scored** aux, `loss_weight` 0.15
   (directional) / 0.05 (forward-return copies). Primary heads stay
   independent BCE at weight 1.0.
   Canonical OOF / serving still read only `p_long` / `p_short`.
   **Amended [ADR-088](ADR-088-aux-side-rank-decision.md):** `aux=True`
   also trains `side_rank = p_long / (p_long + p_short)` on which side
   had the larger 10m MFE, so the aux path regularizes the **decision
   order** without changing fire/serve.

3. **Target-geometry features** on the 15s hive (catalog v1.1):

   | Feature | Formula |
   |---------|---------|
   | `lg_target_v_atr` | `50 / ATR_10m` |
   | `lg_target_v_close` | `50 / cs_close` |
   | `st_target_v_atr` | `-50 / ATR_10m` |
   | `st_target_v_close` | `-50 / cs_close` |

   ATR is causal Wilder ATR(14) of 10-minute OHLC reconstructed from 15s
   `cs_close` / `cs_range` / `cs_clv` (last closed 10m bar, ffilled). Not a
   1-bar tape statistic.

4. **Head-private context.** `lg_*` are concatenated only onto the long
   trunk (`long_ctx` input); `st_*` only onto the short trunk. The shared
   LSTM / TCN / CNN / Transformer encoder never sees them. Tree dual-binary
   drops the opposite pair per estimator. This is required: `st_*` is exactly
   `-lg_*`, so a shared encoder would see a sign-flip duplicate.

Rejected: putting all four columns in the sequence tensor; predicting aux
from a third shared head only (operator asked each directional head to
predict the path labels); rebuilding 15s from TBBO (geometry attaches from
existing `cs_*`).

## Consequences

- Live pin unchanged. 3-class path unchanged.
- Label rebuild: `python scripts/features/build_dual_binary_15s.py --force`
  (and 5s `--force` if those files should carry aux). Fold-cache key for
  `encoding=dual_binary` includes `aux=path_v1|tg=lgst_v1`.
- 15s hive: `python scripts/features/patch_15s_target_geometry.py` persists
  the four columns. `load_feature_frame("15s")` also attaches them if missing
  so a GPU code bundle works before the hive is rewritten.
- GPU 15s mirror (ADR-078) should be re-pushed after the hive patch so the
  A10 has the four columns on disk (load-time attach still covers a stale
  mirror if `cs_*` are present).
- Saved `model.keras` from ADR-080 two-output graphs remain loadable;
  new dual-binary LSTM graphs with aux/context are multi-input / multi-output.
  Serving inspects Keras input names (`input` / `long_ctx` / `short_ctx`) and
  packs via `window_pack_for_model` (`src.models.dual_binary`); 3-class and
  no-ctx dual-binary graphs still get one `(N, T, F)` tensor.

## Related

- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s 1-bar store
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — dual-binary head slot
- [ADR-080](ADR-080-dual-binary-sequence-heads.md) — named `p_long` / `p_short`
- [ADR-086](ADR-086-side-specific-aux-targets.md) — side-signed forward-return aux; drop shared MAE heads
- Catalog: [FEATURE_CATALOG_15s.md](../reference/FEATURE_CATALOG_15s.md)
