# ADR-049: ±10 TBM side model transferred onto +50/−6 events (rejected)

## Status
Accepted as a **negative result** — implemented and measured 2026-08-21.
Does **not** promote. Live pin remains `nq-50pt-econ`.

## Date
2026-08-21

## Context

ADR-048’s binary head on **+50/−6** directional labels had OOF AUC 0.49.
The remaining hypothesis (operator): (1) unused parquet columns may carry
side; (2) train a **separate** side model on an easier label
(`move=10`, `mae=10`); (3) join it to the +50/−6 **event/timer** with a
gate (ADR-047 `p_up≥0.65`, open, 10-minute cooldown).

That stack is the right next experiment: denser, nearly balanced ±10
first-touch labels, full 216-col matrix, walk-forward, economic scoring
still TBM 50/6 cost 1.

## Decision

1. Univariate P(TBM-50/6 up | directional, open) on **every** cache
   column (`univariate_side_auc`).
2. Per-day TBM 10/10 labels from parquet `_close` (no overnight
   concatenation). `alt_tbm_y` in `src/search/side_head.py`.
3. Walk-forward binary XGB on 10/10 directional rows, **all** features,
   10-minute train uniqueness. Combine with the existing 50/6 timer.
   Margin chosen on train using **50/6** PnL, not 10/10 accuracy.

No paid Databento. No `config.yaml` pin change.

## Results (`20260821T120555Z`)

**10/10 labels:** 92.2% directional (675k down / 670k up / 114k flat).
Almost every 10-minute L1 path hits ±10. The “easy side” task is which
10-point barrier prints first — still close to a coin flip.

| Quantity | Value |
|----------|-------|
| 10/10 in-domain OOF AUC (open, dir) | **0.511** (folds 0.49–0.53) |
| Transfer AUC onto 50/6 dir | **0.532** (folds 0.50–0.57) |
| Long-only 50/6 timer | n=57 HR 0.333 NE **11.67** L/S **57/0** |
| Timer + 10/10 side | n=48 HR 0.333 NE **11.67** L/S **47/1** |
| 10/10 side alone, open 10m cd | n=2241 HR 0.119 NE −0.33 L/S 975/1266 side-acc 0.55 |

The join does **not** produce a two-sided 50/6 book.

**Univariate 50/6 (open, dir), all 216 cols:** nothing above 0.553. The
leaders are **activity** (RV, range, spread vol, Kyle λ), not location
or order-flow. `so_*`, `cs_*` CLV, `vw_vwap_dist`, `of_*`, `ca_*` are
inside |AUC−0.5|<0.04 (not in the top 40). Transforms of the same L1
tape (extra CLV windows, signed returns) would duplicate columns that
already lost.

## Consequences

- The two-model stack (easy side + hard event + meta/gate) is implemented
  and falsified on current parquet. It is not a serving path.
- Causal live policy remains ADR-047 long-only open timer.
- A true side channel would need data not in the files (e.g. MBP-10) or
  a different live stop/horizon — not another rearrange of L1.

## Supersedes

None. Closes the “unused columns / ±10 side / join to 50/6 events”
request. Complements [ADR-048](ADR-048-binary-side-head-rejected.md).
