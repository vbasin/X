# ADR-023: Triple-Barrier Label Scheme as a Selectable Objective Parameter

## Status
Accepted — extends [ADR-007](ADR-007-detection-2-mae-filter.md) (Detection-2 MAE filter) and [ADR-018](ADR-018-objective-doc-single-source-labels.md) (objective doc owns label params). Detection-2 remains the default scheme; nothing is superseded.

## Date
2026-07-11 (TBM three-model test)

---

## Context

The `nq-50pt` objective labels a bar directional with the **Detection-2** rule
([ADR-007](ADR-007-detection-2-mae-filter.md)): within the forward window,
`peak_up ≥ move_threshold AND peak_down < mae_tolerance` (symmetric for shorts).
The two excursions are compared **independently over the whole window** — the rule is
**order-agnostic**. Two trades a stop-based trader experiences very differently get the
same label:

1. **Target-then-retrace** — price rallies +50 (trade closed at target), then crashes −100
   inside the same window. Detection-2 rejects it (`down_move ≥ mae`), yet the trader won.
2. **Stop-then-target** — price dips −(mae) first (trader stopped out), then rallies +50.
   Detection-2 with a small `mae_tolerance` correctly rejects this, but the symmetric
   inverse of case 1 (crash-then-rally) is also mislabeled NO_SIGNAL for shorts.

A new objective (`nq-50pt-tbm`) wants the **Triple Barrier Model** (López de Prado):
profit barrier at ±`move_threshold`, stop barrier at `mae_tolerance`, time barrier at
`horizon.max_trade_horizon` — the label is decided by **which barrier the path touches
first**. The label semantics are the *ground-truth definition of a qualifying move*, which
per [ADR-018](ADR-018-objective-doc-single-source-labels.md) belongs in the objective doc.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Fork the labeller per objective (separate module) | No shared-code risk | Duplicates window logic; fold cache/builders need per-objective branches anyway |
| Hardcode TBM as the new default | Simplest wiring | Breaks nq-50pt comparability; violates append-only objective semantics |
| **`labels.scheme` enum in the objective schema + `LabelConfig.scheme` dispatch** (chosen) | One label entry point (`generate_labels`); objective doc stays single source; default unchanged | New enum must thread through fold cache keys |

## Decision

1. **`labels.scheme: detection_2 | triple_barrier`** added to
   `config/objectives/objective.schema.json` (optional; default `detection_2`).
   `nq-50pt-tbm` declares `scheme: triple_barrier` with `mae_tolerance: 6` (a 5-pt adverse
   excursion does not breach the 6-pt stop) — everything else identical to `nq-50pt`.
2. **`generate_labels` dispatches on `LabelConfig.scheme`** (`src/training/labels.py`).
   TBM implementation `generate_labels_tbm` / `_classify_triple_barrier`: vectorised
   first-touch indices (O(n·window); per-offset numpy update), strict `<` ordering
   (`first_touch(target) < first_touch(stop)`), stop touch at `diff ≤ −mae` — matching
   Detection-2's strict `< mae` survival semantics. Directional classes are mutually
   exclusive by path continuity. TBM admits a **superset** of Detection-2's directional
   bars at equal params (first-touch is strictly more permissive than whole-window MAE).
3. **Fold builders are scheme- and timestep-aware.** `src/search/fold_cache.py` probes the
   bar interval from the parquet index (`probe_timestep_seconds`; 5s feature parquets and
   1s raw parquets both work — the objective's `10m` horizon becomes 120 or 600 bars
   respectively) and passes `scheme` into `LabelConfig`. The cache key appends
   `|scheme=…`/`|step=…` **only for non-default values**, so existing detection-2 cache
   entries remain valid. `controller._build_folds` was refactored to consume the shared
   fold cache (one label path; removes the duplicated inline load).

## Consequences

- **+** One dispatch point: trainer, search controller, fold cache, GPU `job_exec` all
  inherit the scheme from the objective doc with zero per-consumer branches.
- **+** `nq-50pt` behavior is bit-identical (default scheme, unchanged cache keys; 40
  existing search/manual/GPU tests + 14 new label tests pass).
- **+** First-touch labels are the canonical companion to meta-labeling
  (`src/training/meta_labels.py`, prepared inert per `config/meta_ab.yaml`).
- **−** Label rate rises (~17.6% → ~21.5% directional on a sample day at 50/6 vs 50/5):
  TBM-trained checkpoints/scores are **not comparable** to Detection-2 runs — compare only
  within one `objective_id`.
- **−** Both schemes still label across session boundaries on the concatenated close
  series (forward window of the last ~10 min of a day sees the next session's prices) —
  pre-existing Detection-2 caveat, now shared and documented; per-session masking is a
  future labeller change if warranted.
- **Constraint**: truncated tail windows: Detection-2's rolling-NaN quirk maps missing
  future data to NO_SIGNAL; TBM keeps a genuine early first-touch label if one occurred
  within the available bars (honest label; can only miss, never fabricate).

Config keys: `labels.scheme` (objective doc), `LabelConfig.scheme`
Implementation: `src/training/labels.py` (`generate_labels`, `generate_labels_tbm`,
`_classify_triple_barrier`), `src/training/config.py` (`LabelConfig`,
`_labels_from_objective`), `src/search/fold_cache.py` (`probe_timestep_seconds`,
`_label_params`, `_cache_key`), `src/search/controller.py` (`_build_folds`)
Tests: `tests/test_tbm_labels.py`, `tests/test_meta_labels.py`
See also: [ML_MODEL.md](../architecture/ML_MODEL.md), [ADR-007](ADR-007-detection-2-mae-filter.md), [ADR-014](ADR-014-requirement-target-kpi.md)
