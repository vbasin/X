# CONFIG_AUDIT.md — Configuration Consumption Audit

**Status: CANONICAL, CURRENT.** Field-by-field audit of `config.yaml` and `config/objectives/nq-50pt-econ.yaml`, derived from reading the actual consuming code (not from what the comments claim). Read this before trusting any config field's stated value as the field's *effective* runtime behavior — several are confirmed dead or contradicted.

## 1. `config.yaml` — field-by-field

| Field | Declared value | Actually consumed by | Effective behavior | Status |
|---|---|---|---|---|
| `objective_id` | `"nq-50pt-econ"` | `src/objectives/loader.py`, `src/webui/live.py:122-127` | Resolves live-serving model + Research Director's objective doc | **LIVE, authoritative for Track B** |
| `dataset parameters.feature_timestep` | `5s` | `src/search/fold_cache.py:80-94` **empirically verifies** the parquet's actual timestep rather than trusting this blindly | Correct when parquet matches; falls back to this value only if the probe fails | **LIVE but defensively double-checked, not blindly trusted** |
| `dataset parameters.scaler_type` | `robust` | `src/training/trainer.py`, `src/models/attention_lstm_plugin.py`, `src/models/svm_plugin.py` (NOT `xgb_trainer.py`, which explicitly omits scaling) | Applied only to logistic_regression/svm/NN families | **LIVE, partially applicable** |
| `dataset parameters.bbo_primary_only` | `true` | Track B feature engine | Restricts to lead contract | **LIVE** |
| `model hyperparameters.*` (`sequence_length`, `num_classes`, `class_weight_directional: 4.0`, `lstm_units: [128]`, `attention_heads: 0`, `dense_units`, `dense_dropout`) | various | `src/training/trainer.py` only | **`src/search/custom_test.py` (the trainer that actually runs Track B's real recent candidates) never imports `trainer.py` — it dispatches through `src/models/catalog.py` instead.** | **DEAD for the live trainer.** Confirmed contradiction: `catalog.py`'s `attention_lstm` entry defaults `attention_heads: 4`, actually used unmodified in real runs — the opposite of what this file declares. |
| `training.*` (`batch_size`, `epochs`, `learning_rate`, `use_focal_loss: false`, `use_balanced_sampling: true`, etc.) | various | `src/training/trainer.py` only | Same as above | **DEAD for the live trainer.** Actual mechanism: `class_balance: "sqrt"` set per-model in `src/models/catalog.py`, consumed by `src/models/tabular_plugins.py:_make_sample_weight()` — an entirely different weighting scheme than `class_weight_directional: 4.0`. |
| `mlflow.tracking_uri` / `artifact_root` | — | MLflow client init | **LIVE** — `vtech-mlflow.service` confirmed active | **LIVE** |

**Action implied, not taken this pass** (would be a config-file edit, out of scope for structural-only restructuring): the entire `model hyperparameters`/`training` block should either be deleted (if `src/training/trainer.py` itself is confirmed dead — not verified this pass) or explicitly re-scoped/labeled as "legacy trainer only, not consumed by Custom Test Lab" so a fresh agent doesn't assume it governs Track B's real behavior.

## 2. `config/objectives/nq-50pt-econ.yaml` — field-by-field

| Field | Declared value | Actually consumed by | Effective behavior | Status |
|---|---|---|---|---|
| `labels.scheme` | `triple_barrier` | `src/training/labels.py::_classify_triple_barrier` | Path-dependent target-before-stop, genuinely (not naive OR-window) | **LIVE** |
| `labels.move_threshold` | `50` (points) | same | Fixed-point barrier width, not volatility-adaptive | **LIVE.** Audit hypothesis (Medium confidence): a fixed-point barrier applied uniformly across NQ's regime-varying realized vol is a plausible contributor to the near-total majority-class collapse observed in real Aug 2026 runs. Not proven; flagged as the top experiment candidate in the audit (`E3`), not yet run. |
| `labels.mae_tolerance` | `6` (points) | same | ~8.3:1 target:stop ratio in raw points | **LIVE** |
| `labels.max_trade_horizon` | `"10m"` | `src/search/fold_cache.py::_label_params()` → 600s → 120 bars @5s | **LIVE, correctly resolved** (matches the horizon the user described) |
| `session.*` (08:02–14:45 CT train, 08:30–14:45 CT trade) | — | Track B feature/label pipeline | **LIVE** | |
| `costs.cost_per_trade` | `0` | `src/backtest/engine.py` | Zero commission in Tier B backtest | **LIVE but likely unintended** — ADR-027's own design text (`docs/decisions/ADR-027-economics-meta-sparse-objective.md`) states `cost_per_trade: 1.0` for this objective; the live YAML disagrees with its own founding ADR. The promoted model's registry entry (`data/models/model_registry/nq-50pt-econ/LATEST.json`) was itself scored at `cost_per_trade: 1.0` at promotion time (2026-08-05) — later leaderboard entries (through 2026-08-22) used `cost_per_trade: 0.0`, meaning **later research candidates were scored under looser cost assumptions than the currently-promoted model**, making them not directly comparable to it. |
| `costs.slippage_model` | `fixed_1pt` | **`src/backtest/engine.py` does not read this key at all** — it only reads a literal `slippage_points` key, which is never set anywhere for this objective | **CONFIRMED DEAD / SILENT BUG.** The translation from `slippage_model: fixed_1pt` → 1.0 points exists only in `src/paper/policy.py:83-86` (paper trading), not in the Tier B research backtest. Every Tier-B backtest number produced for `nq-50pt-econ` (including the one behind the promoted model) ran at **effectively zero slippage** despite the objective declaring otherwise. |
| `kpi.primary` | `net_expectancy` | promotion gate (ADR-071) | Ranks candidates by this | **LIVE** |
| `kpi.champion_gate` | `true` | `src/serving/promote.py` | Enables the requirement-floor gate | **LIVE** |
| `kpi.requirement` (`net_expectancy>=20, n_signals>=30, hit_rate>=0.55, max_drawdown>=-400`) | — | promotion gate | Hard floor before any promotion | **LIVE** — but note: given the `cost_per_trade`/`slippage_model` findings above, a candidate can clear this floor under weaker cost assumptions than the currently-promoted model was held to. |
| `search.default_feature_view` (`momentum_edge` / `sequence_baseline`) | — | `src/models/feature_views.py` | Resolves to a ~60-80 dim subset of the full catalog, not the full ~256-column feature set | **LIVE** — full matrix is a last-resort fallback only (ADR-027 §4 resolution order), not the default. |
| `eval.walk_forward_folds` | `3` | `src/search/controller.py:2038` (`int(eval_cfg.get("walk_forward_folds", 3))`), also `:475`; resulting folds written by `controller.py:504` → `folds.json` | Resolves to 3 expanding walk-forward folds, **honored correctly** | **LIVE AND CORRECT.** See the retraction note below. |
| `eval.hold_out` | `"e68510982d9f"` (reused from `nq-50pt-tbm`'s pin, per the file's own comment) | not re-traced this pass | A locked-holdout reference **is** declared — this nuances (does not contradict) the original audit's "no locked holdout" finding, which was about `hpo_model_search.py`'s trial-scoring behavior, not about whether a holdout identifier exists. Whether this declared holdout is actually excluded from HPO trial scoring is **`DEFERRED — REQUIRES DATA-SIDE VALIDATION`** (would need to trace `src/search/custom_test.py`/`hpo_model_search.py` against this specific hash, out of scope for this pass). |

### `eval.walk_forward_folds` — corrected twice; this is the verified statement

**The repository has two walk-forward engines with incompatible configuration contracts.** Which one runs determines whether `eval.walk_forward_folds` means anything at all.

| Engine | `split` value | Implementation | Does it read `eval.walk_forward_folds`? |
|---|---|---|---|
| Equal-count expanding | `expanding_equal_count` | `src/search/controller.py:2038` (`int(eval_cfg.get("walk_forward_folds", 3))`) | **Yes — it is an INPUT.** Declared value honored. |
| Calendar expanding (ADR-042) | `expanding_calendar` | `src/validation/walk_forward.py::schedule_from_eval` | **No.** It reads `eval.walk_forward.{train,test,val}` and `eval.date_range`. The fold count is a **derived OUTPUT** (`"walk_forward_folds": len(schedule)`). `src/search/walk_forward_calendar.py` does not exist. |

Verified by enumerating all **54** `folds.json` files while those runs lived under `data/research/custom_test/` (that directory is gone; the fitted runs are under `research/runs/`): `n_folds` is 3 in 38 runs, 6 in 12, 4 in 1, and **1 in 3 runs** (`nq-50pt-econ/2026082{2T050238Z,2T043233Z,2T034344Z}`). Every 3-fold run is `expanding_equal_count`; every other count is `expanding_calendar`. The equal-count fallback is still `src/search/controller.py::_build_folds`. Track C evaluators use `src/validation/folds_from_objective`, not a private `_folds()`.

**The material finding:** the three `n_folds=1` runs used `expanding_calendar`, whose schedule collapsed to a single window. A single window is **one train/test split — no walk-forward validation at all** — produced silently, under an objective doc that declares `walk_forward_folds: 3`. The declared value was not violated so much as *not consulted*, because a different engine was selected.

**Correction history (recorded because both errors are instructive).** The second pass claimed `n_folds=1` in *every* run — false (3 of 54), asserted from an unrepresentative sample without enumerating the population. The third pass then over-corrected to "never 1" — also false, from a `sort | uniq -c` whose output was misread. Only the full enumeration above is reliable. A summary of the files is not the files; and a partial count is not a population.

**Correction note (2026-09-22, second restructuring pass)**: an earlier version of this document's `KNOWN_OBJECTIVE_KEYS` allowlist (used by the validator above) was built from secondhand summary notes and covered only 6 of this file's 16 real top-level keys, producing false-positive "unknown key" warnings when first run against the real file. Fixed by reading `config/objectives/nq-50pt-econ.yaml` directly; the corrected allowlist and this section's `eval.walk_forward_folds`/`eval.hold_out` rows are the direct result. Left in as a documented example of exactly the failure mode this whole document exists to prevent — a summary of a file is not the file.

## 3. Track C objective document

Present: the market-entry cascade objective is `config/objectives/nq-Rscaled-mkt.yaml`. `mnq-auction-firstpass.yaml` is the superseded first-passage objective. The numeric defaults still also live in `src/training/cascade_policy.py` and the matrix builders. The YAML is the objective identity those runs record; it is not a second label engine.

Historical note: before those YAML files existed, the same parameters (`move_r=0.25`, `adverse_r=0.125`, `horizon=1800s`, `stop_ratio=0.5`, `max_stop_r=0.25`, `lookback_sessions=20`) were only hardcoded constants and CLI defaults.

## 4. Cross-cutting findings

1. **Config/runtime divergence is not hypothetical — it happened at least twice, confirmed**: (a) `attention_heads` (Track B), (b) `slippage_model` (Track B, both tracks structurally). This is the concrete justification for `docs/ARCHITECTURE.md`'s "unknown/dead config should fail loudly" principle — no validator currently enforces this anywhere in the repo.
2. **`nq-50pt-econ-lg-st` differs from `nq-50pt-econ` in more ways than its name suggests**: dual long/short heads (`encoding: dual_binary`), native 15s/1m/5m/1h/1d feature stack (not the 5s `momentum_edge` view), and `target_observation: tbbo` (true intrabar high/low) instead of `candle_close`. A fresh agent should not assume config for one transfers to the other.
3. `src/objectives/loader.py`'s own docstring claims config.yaml is "objective-agnostic" (auto-discovery), but `config.yaml:11` explicitly pins `objective_id` — stale comment, functionally harmless (explicit pin still wins) but a small instance of the same doc/code drift pattern.

## 5. What this document does not do

It documents **current runtime truth** — what `config.yaml`/`config/objectives/*.yaml` actually control today, including the dead/contradictory fields above. It does not itself change any config file's values or structure.

## 6. Current vs. Target configuration architecture

**Do not conflate these two things** — they answer different questions and this document intentionally keeps them separate:

- **CURRENT (this document, sections 1-4)**: what `config.yaml` + `config/objectives/*.yaml` actually do at runtime today, verified against consuming code, including known dead fields and doc/code contradictions. This is authoritative for "what will happen if I run the system right now."
- **TARGET (this section)**: the composable, per-responsibility config architecture `docs/ARCHITECTURE.md` and `docs/RESEARCH_WORKFLOW.md` describe as the eventual destination — where each of the 12 canonical responsibilities' configuration is validated and resolved independently (e.g. Outcome Builder config validated on its own, Economic Evaluator's `EconomicAssumptions` validated on its own — see `src/evaluation/economic.py`'s `__post_init__`, which already fails loudly on invalid values), then composed into one `RunManifest.config_snapshot` (`src/experiment/manifest.py`) per experiment rather than living in one monolithic objective YAML. This is **not implemented as a replacement config system** — `config.yaml` and the objective documents remain authoritative. There is no `src/canonical/` package. Shared experiment code is `src/experiment/`.

What **is** implemented toward the target, as of this pass: `src/experiment/config_validation.py` provides fail-loud validation of the *current* `config.yaml`/objective-doc structure (unknown top-level keys, known-dead fields, missing required fields, the specific `slippage_model`/`slippage_points` mismatch) — this is a diagnostic tool that reads current config and reports against the findings in sections 1-4 above, not a new config format. Running it against the real files in this repository is recorded in `docs/MIGRATION_MANIFEST.md`'s validation log.

Do not build a second, parallel config-loading path for any existing script based on this section — that would be exactly the kind of divergence (declared vs. effective behavior) this document exists to catch, not repeat.

## 2026-09-24 — independent NQ experiment and causal correction

`config/objectives/nq-selective-intraday.yaml` identifies the new research
objective. The preregistered runner's resolved `protocol.json`/manifest records
all actually used candidate and policy settings. The NQ commission is loaded
from `config/execution_costs.yaml`; base slippage is explicitly overridden to
0.25 point per side with 250ms quote-event latency. It does not inherit the
legacy objective's dead slippage field or claim live connectivity.

Earlier claims that every part of `entry_state.py` is causal are incorrect:
`_order_flow` normalizes CVD/delta/OFI by full-request maxima. Prefix-invariance
fails. Existing affected cached features are excluded from the new experiment.
See [measured research evidence](research/NQ_SELECTIVE_RESULTS.md).
