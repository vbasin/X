# ADR-018: Trade Objective Doc is the Single Source of Truth for Objective-Level Params

## Status
Accepted — refines [ADR-006](ADR-006-config-yaml-single-source.md) (config.yaml single source) and builds on [ADR-014](ADR-014-requirement-target-kpi.md) / the Phase-23 objective doc.

## Date
2026-07-03 (Phase 23)

---

## Context

Phase 23 introduced the versioned trade objective doc `config/objectives/<id>.yaml`
(schema-validated, loaded via `src/objectives/loader.load_objective`) as the declarative
entry point for a program: it defines the labels, session, horizon, and KPIs. The
model-selection **search** already read label params from it
(`src/search/fold_cache._label_params`).

However, the legacy `program objectives` / `success thresholds` / `labels` blocks from
[ADR-006](ADR-006-config-yaml-single-source.md) still lived in `config.yaml`, and the
**trainer** (`src/training/config.py` `MLConfig` → `LabelConfig`) read its label params
from that `labels` block. This created **two sources of truth** for the same values and
they had drifted:

- `config.yaml` `labels.mae_tolerance` = **10**
- `config/objectives/nq-50pt.yaml` `labels.mae_tolerance` = **5**

Because a `BIG_UP`/`BIG_DOWN` bar requires `adverse_move < mae_tolerance`, the trainer was
building a **looser positive class** (≤10 pt adverse) than the objective/search target
(≤5 pt). A champion selected by the search was then fully trained/evaluated against a
different label definition — metrics were not comparable across stages. Every other
authoritative source (objective doc, `LabelConfig` default, `SYSTEM.md`,
[ADR-007](ADR-007-detection-2-mae-filter.md)) specified **5**; `config.yaml`'s `10` was a
drift bug.

---

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Keep both, just fix the value (10→5) | Smallest change | Two sources remain — drift will recur |
| Delete `labels` from config.yaml, rely on `LabelConfig` code defaults | Slim config | Label params become implicit; editing the objective doc would not affect training |
| **Objective doc is the sole source; `config.yaml` references it via `objective_id`** (chosen) | One source of truth; trainer + search read the same values; no drift possible | Trainer gains a dependency on the objectives loader |
| Rewire trainer to take `--objective` CLI arg only | No config reference | Breaks existing callers (`hpo_search`, `permutation_importance`, ablation) that call `MLConfig.from_env()` with no args |

---

## Decision

The trade objective doc `config/objectives/<objective_id>.yaml` is the **single source of
truth** for all objective-level parameters: `labels` (`move_threshold` / `mae_tolerance`),
`session`, `horizon`, and `kpi` (requirement / target thresholds). `config.yaml` **must not
duplicate** any of those values — it only names which objective is active:

1. `config.yaml` gains a top-level `objective_id: "nq-50pt"` **reference** (a pointer, not a
   value) and its `labels` block and duplicated session hours (`dataset parameters.trading_start/end/timezone`) are removed.
2. `MLConfig.from_env()` loads label params from the objective via new helper
   `_labels_from_objective(objective_id)` (maps objective `move_threshold` →
   `LabelConfig.big_move_threshold`, `mae_tolerance` → `mae_tolerance`). It falls back to
   `LabelConfig` dataclass defaults with a warning if the objective is missing/invalid, so
   no caller hard-fails.
3. `forward_windows` remains a training-only knob on `LabelConfig` (not objective-defining).

This refines [ADR-006](ADR-006-config-yaml-single-source.md): `config.yaml` stays the single
source for non-objective runtime knobs (feature toggles, model hyperparameters, training,
mlflow, snapshot, logging), but **objective-level params are delegated to the objective doc**.

---

## Consequences

- **+** No possible drift between the search and the trainer — both read the same file.
- **+** The trainer's positive class now matches the objective (`mae_tolerance` corrected 10→5).
- **+** Adding a new program = add `config/objectives/<id>.yaml` + point `config.yaml`'s `objective_id` at it.
- **−** Models trained before this fix used `mae_tolerance=10`; they must be re-trained to be comparable against search-selected champions.
- **−** The trainer now depends on `src/objectives/loader` (a low-level shared module — acceptable).

### Known follow-ups (not addressed here)
- **Session hours**: the feature engine reads `trading_start` from the `VTECH_TRADING_HOURS_START` env var / default, not the objective. True single-source for the session window still requires wiring the feature engine to the objective's `session` block.
- **Horizon mismatch**: the trainer's primary label uses `forward_windows[0]` = 60 bars (5 min), while the objective `horizon.max_trade_horizon` = 10 min (120 bars) — which the search uses. Aligning the trainer's primary window to the objective horizon is a separate label-semantics decision.
- **`dataset parameters` section**: the config loaders read `data:` / `features.*`, not `dataset parameters:`, so that block is largely inert — a pre-existing config-loading bug independent of this ADR.
