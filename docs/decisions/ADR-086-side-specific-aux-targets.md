# ADR-086: Side-specific dual-binary aux targets

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

ADR-082 attached seven Huber heads to each dual-binary trunk. Three of
those are genuinely side-specific (`MFE_*`, `MAE_b4_TGTorMFE_*`,
`Time_to_TGT_*`). The other four used unique Keras names
(`long_forward_return_1m` vs `short_forward_return_1m`,
`long_MAE_before_target_or_MFE` vs `short_MAE_before_target_or_MFE`) but
the **same parquet y**: signed `forward_return_*` and unsigned
`MAE_before_target_or_MFE`.

Custom Test `nq-50pt-econ-lg-st/20260828T101050Z` showed the two trunks
collapsing: `corr(p_long, p_short) = 0.93`, predicted `MFE_long` vs
`MFE_short` **+0.73** (true labels **−0.21**), predicted
`long_forward_return_10m` vs `short_forward_return_10m` **+0.95**. The
`long_` / `short_` suffixes on those four copies did not keep each head
on its direction; they trained both Dense(1) layers on one signed path.

MFE labels themselves already mean opposite excursions (peak upside vs
peak downside, both ≥ 0). Independent Huber on a **shared** LSTM
embedding can still fit both as “how large is the next 10-minute range”
without learning which side wins. Identical-y forward-return copies
reward that collapse. This ADR does not claim MFE predictions will
anti-correlate after the map change; it removes the copies that forced
the trunks to reconstruct the same number.

Rejected: rebuilding parquet columns; clipping `forward_return` at 0
(that discards the signed drift the long head should own).

## Decision

1. **`LONG_AUX_MAP` / `SHORT_AUX_MAP`** are `(keras_name, parquet_col,
   sign)` triples. `pack_aux_outputs` multiplies the per-fold z-score by
   *sign*.

2. **Forward return is opposite across trunks.**
   `long_forward_return_*` trains on `+z(forward_return_*)`.
   `short_forward_return_*` trains on `−z(forward_return_*)`.
   A +20 pt rally is a positive long aux target and a negative short
   aux target. Parquet column unchanged.

3. **Drop the shared MAE heads.** `MAE_before_target_or_MFE` stays on
   the label parquet (schema enum kept). Each trunk already has
   `MAE_b4_TGTorMFE_{long|short}_10m`. `nq-50pt-econ-lg-st` `labels.aux`
   no longer lists the shared column (fold-cache key changes).

4. Per-side MFE / MAE-before / time-to-target maps are unchanged
   (still `sign=+1` onto that side’s parquet column). Serving still
   reads only `p_long` / `p_short`. No label-parquet rebuild.

## Consequences

- Live pin unchanged. 3-class path unchanged. Saved
  `20260828T101050Z` graph still loads (extra aux outputs ignored at
  serve). Do not reuse that run as a post-fix comparison — re-run
  Custom Test so the LSTM trains with the new maps.
- Huber graph: 12 aux outputs (was 16).
- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) “same y”
  sentence for the named forward-return copies is superseded.

## Related

- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) — aux columns
  + original shared copies
- [ADR-084](ADR-084-objective-opt-in-aux-heads.md) — YAML `labels.aux`
- [ADR-080](ADR-080-dual-binary-sequence-heads.md) — named `p_long` /
  `p_short`
- [ADR-088](ADR-088-aux-side-rank-decision.md) — `side_rank` on
  `p_long / (p_long + p_short)` from MFE winner
