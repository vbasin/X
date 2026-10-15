# RESEARCH_WORKFLOW.md — Canonical Research Lifecycle

**Status: CANONICAL, CURRENT.** Defines the one intended end-to-end workflow. Read `docs/ARCHITECTURE.md` first for the responsibility definitions this workflow sequences.

## 1. The canonical lifecycle

```
HYPOTHESIS
  → EXPERIMENT (data contract + feature spec + outcome spec + model strategy + validation plan, all versioned — see EXPERIMENTS.md)
  → TRAIN (Model Strategy, per Validator's folds)
  → VALIDATE (Validator: fold metrics, purge/embargo respected, n_eff reported)
  → CALIBRATE (Calibrator)
  → APPLY DECISION POLICY (thresholding/gating/fill-mode, population persisted)
  → ECONOMIC EVALUATION (Economic Evaluator, explicit cost assumptions persisted — never implicit zero-cost)
  → COMPARE against baseline/parent run (see EXPERIMENTS.md §Comparison)
  → ACCEPT or REJECT the hypothesis
  → (only once, at the very end, never during iteration) LOCKED HOLDOUT evaluation
  → PAPER (Serving, paper mode)
  → PROMOTION (Model Lifecycle: EXPERIMENTAL → CANDIDATE → PRODUCTION)
  → LIVE (Serving, live mode — not currently active for any track, see project_manifest.yaml)
```

**Today's actual entry points, mapped onto this workflow** (neither track currently implements the full chain faithfully — gaps are the point of this document, not hidden):

| Workflow stage | Track B entry point | Track C entry point |
|---|---|---|
| Feature build | `scripts/features/build_native_tf_*.py`, `src/features/engine.py` | `scripts/features/build_move_matrix.py` |
| Outcome build | `scripts/features/build_labels_5s.py` | `scripts/features/build_move_matrix.py` (labels inline), `scripts/features/build_target_matrix.py` |
| Train + Validate | `src/search/custom_test.py` → `src/models/catalog.py` (walk-forward infra exists, real runs use `n_folds=1` — see `docs/CONFIG_AUDIT.md`) | `scripts/features/eval_move_model.py`, `scripts/features/eval_target_model.py` |
| Calibrate | **none** | **none** |
| Decision Policy | `tau` confidence gate + champion-gate requirement floors (`config/objectives/*.yaml`) | `p_move>=tau_q` gate + target-adaptive stop (`scripts/features/build_target_matrix.py`) |
| Economic Evaluation | `src/evaluation/_kpi.py` (Tier A) + `src/backtest/engine.py` (Tier B, buggy) | `src/backtest/cascade_backtest.py` (zero-cost) + console block in `eval_target_model.py` |
| Compare | **none formal** — manual log/leaderboard reading | **none formal** — manual log reading (this session's own working pattern) |
| Locked holdout | **none** — HPO reuses the same OOF split across trials | **none** — same fixed walk-forward folds re-consulted every iteration |
| Paper | `src/paper/runner.py` (`vtech-paper.service`, inactive since 2026-08-05) | **none** |
| Promotion | `src/serving/promote.py`, ADR-071 champion gate | **none** |
| Live | not active | not active |

## 2. Hard rules for any new experiment (canonical, applies regardless of track)

1. **No experiment's scientific identity is a version-suffixed filename.** Use the `RunManifest` contract (`src/experiment/manifest.py`, see `docs/EXPERIMENTS.md`) even while the rest of the chain above is still track-specific.
2. **Economic evaluation is mandatory and cost-explicit before any claim of "improvement."** Use `src/evaluation/economic.py`'s `EconomicAssumptions` — zero-cost must be an explicit, visible choice, never a silent default from a missing config key (see `docs/CONFIG_AUDIT.md` for exactly how that happened in Track B).
3. **A locked holdout is consulted at most once per hypothesis**, only after the hypothesis is otherwise accepted on non-locked validation folds. Repeated peeking at the same fixed OOS window to decide "did this change work" (confirmed practice on both tracks, including this session's own Track C iterations) is a known, unmitigated multiple-comparisons exposure — do not add to it.
4. **Any result that would influence a research decision must be persisted as a durable artifact**, not left as console output. See `docs/EXPERIMENTS.md` for the artifact contract.
5. **A cascade/gate/meta-layer is an optional Model Strategy + Decision Policy composition, never a required starting point.** Don't reach for one because Track C has one; justify it against a simpler joint-model baseline first.
6. **Don't reach for a sequence model (LSTM/TCN) by default.** Current evidence (Track B's real Aug 2026 runs) shows XGBoost meaningfully outperforming attention-LSTM/TCN on this data; a sequence model needs to win a controlled ablation before further investment, not be assumed superior because the data is sequential.

## 3. Comparing two runs

See `docs/EXPERIMENTS.md` §Comparison for the full contract. In short: never conclude "feature X helped" from two runs that differ in more than one dimension (e.g. Track C's own v3→v4 comparison this session changed both the feature set *and*, via the resulting model, the Decision Policy's gated population — the two effects were never isolated). The comparison tooling's job is to make that kind of confound visible automatically, not just to diff metrics.

## 4. Canonical command surface

`python -m src.experiment.cli <subcommand>` — the one obvious command surface for what's actually implemented so far: `status` (prints `project_manifest.yaml`), `validate-config` (fail-loud config check, see `docs/CONFIG_AUDIT.md` §6), `run-inspect`/`run-compare` (read/diff `RunManifest`s), `verify` (RUNS the fast DB-free suite and propagates pytest's exit code — it is named `verify`, not `test`, precisely because a verification command that only prints a suggestion cannot fail). It does **not** wrap or reimplement any Track B/C training/evaluation command — `status`'s own output points at this document's §1 table for those instead of pretending a canonical train/eval command exists yet.

## 5. What this document does not yet do

This document describes the target workflow. As of 2026-09-22, only the Experiment and Economic Evaluator responsibilities have real canonical (`src/`) implementations, and neither is wired into Track B's or Track C's actual scripts yet — that wiring is explicitly out of scope for the structural-only restructuring pass that created this document (see `docs/decisions/ADR-136-canonical-research-architecture.md` §Non-goals). Track B and Track C's existing entry points (table above) remain how you actually run something today.

## Independent NQ retry entry points (2026-09-24)

ADR-139 adds an experimental policy-research path. Preparation aggregates existing
source 1-second bars; two confirmed truncated discovery sessions were recovered
from the vendor only after exactly-zero cost checks, into a separate artifact.
`run_nq_cadence_area.py` selects on 2011–2020, writes a frozen policy before
2021–2024 validation, and retains every candidate. The wider parameter grid was
an explicitly logged adaptation to discovery results; no validation outcomes
were consulted to choose that extension. Published research knowledge still
limits claims that historical validation is globally untouched.

`eval_nq_cadence_quotes.py --phase compatibility` evaluates the frozen policy on
already-exposed April 2025–June 2026. `--phase holdout` refuses to read the final
reserve until historical and compatibility gates pass, then writes a consultation
marker before any reserved price read. Reserve results cannot trigger retuning.
This is rolling past-state evaluation, not shuffled OOF or future-fitted scaling.
Independent quote marking, real-data prefix checks, fixed long/cash benchmarks,
and the PDF/CSV exporter complete the evidence path. See
[measured status](research/NQ_RETRY_RESULTS.md). No automatic deployment follows.


## Public synthetic Transformer application — 2026-10-15

The independent `synthetic-regime-service` objective applies the research
lifecycle to a small classification service. `src.experiment.transformer_workflow`
performs purged training, early stopping, separate calibration, comparison,
validation-before-holdout gates, negative controls, verified HTTP inference and
scoped model promotion. Its diagnostic synthetic trading ledger is not the
promotion objective. It does not waive economic/paper gates for trading tracks.
The recorded result and an optional fresh run are accessible through the
[file-backed showcase walkthrough](examples/TRANSFORMER_WORKFLOW.md), with
MLflow parent/child lineage and explicit original-versus-imported provenance.
