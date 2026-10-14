# ADR-098: Time_to_TGT is NaN when the 10m window does not hit ±50

## Status
Accepted — implemented 2026-08-29. Live pin unchanged.  
2026-08-29: rebuilt **all** 15s label days (eval `date_range` had left
April 2025 / post-Jun 2026 files at timeout=600). Load path sanitizes
stale 600s. Fold-cache key `|ttt=nan_timeout`.

## Date
2026-08-29

## Context

[ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) stored
`Time_to_TGT_long` / `Time_to_TGT_short` as seconds to first ±`move_threshold`
close, **or the horizon (600s) if never touched**. [ADR-095](ADR-095-regime-context-fire-aux.md)
dropped those columns from `nq-50pt-econ-lg-st` `labels.aux` because the
censored series was ~constant (mean ~588s; ~88% of 15s rows were 600).

The 600 is the **time barrier**, not a target. On actual +50/−50 hits the
mean is ~280s (median ~260s). Huber on the mixed series trained “almost
always 600,” not speed.

Keras aux already maps NaN → dummy 0 with **sample weight 0**
(`pack_aux_outputs` / `aux_sample_weight`), so a timeout-NaN series trains
only on hits. Primary `p_long` / `p_short` BCE is unchanged.

## Decision

1. **Writer:** `compute_aux_labels` writes **NaN** when that side never
   prints ±threshold inside the horizon. Hits stay `(bar_index + 1) × src_step_s`.
2. **`nq-50pt-econ-lg-st` `labels.aux`** includes `Time_to_TGT_long` and
   `Time_to_TGT_short` again. Z-score is `nanmean`/`nanstd` on hit times.
3. Rebuild `parquet_labels_15s` (`build_dual_binary_15s.py --force`).
   Fold-cache key includes the aux list, so the next LSTM run rebuilds
   `y_aux`. Live pin unchanged. Not KPI / not serving.

Rejected: keep 600 and drop the head (ADR-095); censor-aware survival loss
(no library path).

## Consequences

- Existing label parquets with 600 are stale until rebuild.
- GPU `parquet_labels_15s` sidecar must be re-pushed before the next
  A10 LSTM job (ADR-078).
- Amends ADR-082 (column meaning) and ADR-095 / ADR-084 (lg-st aux list).

## Related

- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md)
- [ADR-084](ADR-084-objective-opt-in-aux-heads.md)
- [ADR-095](ADR-095-regime-context-fire-aux.md)
- `src/training/aux_labels.py`
