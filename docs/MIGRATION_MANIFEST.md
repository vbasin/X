# MIGRATION_MANIFEST.md — Canonical Architecture Restructuring

**Status: append-only record.** Sections 1–12 are the 2026-09-22 Tranche 1/2 record and are preserved verbatim, including statements that were later found wrong. Where one of those statements was corrected inside this file, it carries an inline SUPERSEDED note. Later dated sections record what changed after that. They are history, not a second copy of the current tree.

**Present:** `src/canonical/` does not exist. Shared experiment code is `src/experiment/`. Causality is `src/common/causal.py`. Economics is `src/evaluation/economic.py`, wired into `portfolio_sim` and Track C's persisted book. Validation is `src/validation/`; Track B's equal-count fallback remains in `src/search/controller.py::_build_folds`. New runs write `research/runs/`. `research/custom_test/` is gone. `research/search/` is a compatibility root. HPO (`scripts/search/hpo_model_search.py`) writes under `research/runs/` and records the folds it scored. The last section of this file is the repair note for that state.

## 1. Files created

### Tranche 1
`CLAUDE.md`, `project_manifest.yaml`, `docs/ARCHITECTURE.md`, `docs/RESEARCH_WORKFLOW.md`, `docs/EXPERIMENTS.md`, `docs/DATA_CONTRACTS.md`, `docs/CONFIG_AUDIT.md`, `docs/MIGRATION_MANIFEST.md`, `docs/decisions/ADR-136-canonical-research-architecture.md`, `docs/archive/AGENTS.pre-canonical-2026-09-22.md`, `src/__init__.py` + 12 responsibility subpackages (`data_contracts/{__init__,causal}.py`, `features/__init__.py`, `outcomes/__init__.py`, `models/__init__.py`, `validation/__init__.py`, `calibration/__init__.py`, `decision_policy/__init__.py`, `evaluation/{__init__,economic}.py`, `experiment/{__init__,manifest}.py`, `lifecycle/{__init__,status}.py`, `serving/__init__.py`, `orchestration/__init__.py`), `tests/canonical/{__init__,test_causal,test_economic_evaluation,test_experiment_manifest,test_lifecycle}.py`.

### Tranche 2
| New path | Purpose | Status |
|---|---|---|
| `docs/STORAGE_CONTRACT.md` | Canonical storage/artifact layout — **extends**, does not duplicate, `docs/architecture/DATA_LAYOUT.md` | CANONICAL |
| `docs/ENVIRONMENT.md` | Reproducible setup, logging expectations, fast verification loop | CANONICAL |
| `docs/SCRIPT_INVENTORY.md` | Script-by-script/directory-level classification | CANONICAL |
| `docs/decisions/ADR_INDEX.md` | Topic/status index over all 137 ADRs (bucketed, not individually re-read) | CANONICAL |
| `src/experiment/config_validation.py` | Fail-loud validator for the CURRENT config.yaml/objective-doc structure | CANONICAL, real, tested, integration-checked against real files |
| `src/experiment/comparison.py` | `compare_runs`/`ComparisonReport` — implements `docs/EXPERIMENTS.md` §3's multi-change warning | CANONICAL, real, tested |
| `src/experiment/schemas.py` | `FoldMetrics`/`CalibrationResult`/`SelectedPopulation` result schemas | CANONICAL, real, tested |
| `src/experiment/cli.py` | `python -m src.canonical.cli {status,validate-config,run-inspect,run-compare,verify}` | CANONICAL, real, tested |
| `tests/canonical/test_config_validation.py`, `test_comparison.py`, `test_schemas.py`, `test_cli.py` | Tests for the above | CANONICAL |

## 2. Files modified

**Pre-existing repository files** (predate this restructuring task entirely — edits here fall under the copy-first/preserve-original discipline):

| Path | Pass | Change | Original preserved where |
|---|---|---|---|
| `AGENTS.md` | Tranche 1 | Rewritten: kept all operational content, added orientation content | `docs/archive/AGENTS.pre-canonical-2026-09-22.md` (verbatim snapshot before any edit) — **this is the sole recorded exception to copy-first**, formalized in `AGENTS.md` §4's own text as of Tranche 2 |
| `AGENTS.md` | Tranche 2 | Hardened: trimmed volatile Track A/B/C status/numbers out of §2 (now a pointer to `project_manifest.yaml` instead of restating dated figures); restructured §5 to separate the permanent "DB access is normal for data/feature work" rule from the task-scoped "structural-only work must not touch DB" rule (previously conflated, risked being misread as a blanket future ban); added the explicit copy-first-exception record for its own Tranche-1 rewrite | same snapshot as above remains the rollback reference |
| `CLAUDE.md` | Tranche 2 | Trimmed a bullet that restated (rather than deferred to) `AGENTS.md` §4/§5's content | N/A — this file was itself created in Tranche 1 of this same task, not pre-existing; no separate snapshot needed, git history covers it |
| `docs/README.md` | Tranche 2 | **Additive only** — inserted one pointer paragraph above the existing "Document Map" table naming the new canonical docs as the required entry point; zero existing content removed or reworded | N/A — pure insertion, not a rewrite; does not need the copy-first-exception framing that `AGENTS.md`'s substantial rewrite needed |
| `docs/roadmap/ROADMAP.md` | Tranche 2 | **Additive only** — inserted one "ARCHIVED (ADR-136)" banner paragraph immediately before the legacy Phase 20 (Track A) section; zero existing content removed | N/A — pure insertion |

**Files created by this restructuring task itself, then further edited within the same task** (not pre-existing repo content — normal iterative authoring, no copy-first exception needed, listed for completeness):

`project_manifest.yaml`, `docs/ARCHITECTURE.md`, `docs/CONFIG_AUDIT.md`, `docs/EXPERIMENTS.md`, `docs/RESEARCH_WORKFLOW.md`, `src/experiment/__init__.py`, `src/evaluation/__init__.py` — all edited in Tranche 2 to fix a bug (`ZERO_COST` re-export), correct the doc-classification overreach (§3 below), add the new modules' exports, add cross-references to the new Tranche-2 docs, and record the `eval.walk_forward_folds` finding discovered by Tranche 2's own fresh-agent test.

No other existing file was modified. No file was moved, renamed, or deleted, in either tranche.

## 3. Corrections made during Tranche 2 (hardening findings against Tranche 1's own output)

These are documented explicitly because the user's Tranche-2 instructions specifically asked for tranche-1 to be audited against the original spec, not just extended:

1. **`AGENTS.md` §2 contained volatile, datestamped Track A/B/C performance numbers** (specific AUC/hit-rate/expectancy figures, "inactive since 2026-08-05" style dates) that would go stale in a file meant to be durable. **Fixed**: trimmed to a pointer at `project_manifest.yaml`, which is explicitly designed to hold exactly this kind of current-status data.
2. **`AGENTS.md` §5's DB restriction read as a permanent, unconditional ban** on `psql`/DB access, contradicting §7's own SQL/testing standards (which assume DB access is normal). **Fixed**: restructured into two explicitly separated cases — normal engineering work (DB access is fine) vs. structural-only tasks (DB access is prohibited) — with a rule for which case applies.
3. **`docs/ARCHITECTURE.md` §4 originally classified all of `docs/architecture/*.md` as "historical, superseded"**. This was wrong — those 9 files carry recent "Present" dates (2026-08-05 through 2026-09-03) and are actively maintained per the repo's own ADR-028 convention. **Fixed**: replaced with a per-file table; the correct characterization is "current, authoritative, but with a Track C blind spot" (most were written without reference to the cascade), not "historical."
4. **`docs/EXPERIMENTS.md` invented a new top-level `artifacts/runs/<run_id>/` storage root**, when `docs/architecture/DATA_LAYOUT.md` §4 already plans an `experiments/` grouping for exactly this purpose. **Fixed**: corrected to `$VTECH_DATA_DIR/experiments/runs/<run_id>/`, documented in both `docs/EXPERIMENTS.md` and the new `docs/STORAGE_CONTRACT.md`. — **SUPERSEDED by §15.2 (third pass): this "fix" was itself wrong.** `experiments/` was never adopted; `DATA_LAYOUT.md` §4 is titled "Target shape (recommended)". The adopted root is `research/custom_test/<objective_id>/<run_id>/`.
5. **`docs/CONFIG_AUDIT.md`'s validator (`KNOWN_OBJECTIVE_KEYS`) was built from secondhand summary notes, not the real file**, covering only 6 of `config/objectives/nq-50pt-econ.yaml`'s 16 actual top-level keys — running the validator against the real file (part of this tranche's own fresh-agent test, §6 below) produced 10 false-positive "unknown key" warnings. **Fixed**: read the real file directly, corrected the allowlist, and — as a direct product of doing this properly — discovered a genuinely new finding (`eval.walk_forward_folds` declares 3 but real runs show `n_folds=1`, the same class of bug as the already-confirmed `slippage_model`/`attention_heads` findings). Recorded in `docs/CONFIG_AUDIT.md` and `project_manifest.yaml`. — **SUPERSEDED TWICE by the third pass.** (a) The `n_folds=1` finding was false and was then over-corrected; only the 54-file enumeration in §15.1 is reliable. (b) The hand-written allowlist should never have existed: `config/objectives/objective.schema.json` is the authority and already sets `additionalProperties: false`. The allowlist has been removed (§15.3b).
6. **`src/evaluation/__init__.py` didn't re-export `ZERO_COST`**, breaking the documented import path. Caught by Tranche 1's own import smoke-test, fixed same-session.
7. Test-bug (not a code bug): a parametrized causal-index test had a wrong expected value (`9` instead of `99` for a 100-element array). Caught immediately by the test suite itself; fixed the test, not the code under test.

Items 1-5 are exactly the kind of thing the Tranche-2 task asked to be hardened against ("resolve contradictions," "classify docs individually," "current vs target," "avoid arbitrary path design") — recorded here rather than silently fixed so the correction itself is auditable.

## 4. Validation performed

- `pytest tests/canonical/` — Tranche 2 recorded **76 passed**; that figure was never re-verified and an adversarial review flagged it as untenable against a static count of the test files. **Current verified figure: 114 passed, 0 failed, 0.89s** (third pass, `pytest tests/canonical/` executed). Zero database dependency (verified: no import in `src/*` or `tests/canonical/*` touches `psycopg`/`SQLAlchemy`/any DB client).
- **Real-fixture integration check** (the "old vs new, compare against existing fixtures" requirement, applied to the one component judged safe to integrate this pass — see §5): `src/experiment/config_validation.py` run against the actual `config.yaml` and `config/objectives/nq-50pt-econ.yaml` in this repository (read-only, no mutation). Result: correctly reproduces the manually-audited findings (`model hyperparameters`/`training` sections dead; `slippage_model`/`slippage_points` bug) with zero false positives after the correction in §3 item 5, plus surfaced one genuinely new finding (`eval.walk_forward_folds`). This is real parity validation against production data, not a synthetic-only test.
- `python -m src.canonical.cli status` and `validate-config` — both run successfully against the real repository as part of the Fresh-Agent Test (§6).

## 5. Priority integration checklist (what was/wasn't wired into an existing executable path, and why)

Per the task's explicit instruction: integrate only where safe parity can be demonstrated without touching model science; otherwise leave unwired and document exactly why.

| Priority | Candidate | Attempted? | Result |
|---|---|---|---|
| 1 | Config validation (`config_validation.py`) wired against real `config.yaml`/objective docs | **YES — integrated as a read-only diagnostic** | Real-fixture parity demonstrated (§4). This is pure inspection (loads YAML, reports findings) — it changes nothing about how `config.yaml` is actually consumed by any existing script, so there is no "old vs new executable behavior" to risk; the only risk was the validator itself being wrong, which the real-fixture run against real files was specifically designed to catch (and did — see §3 item 5). |
| 2 | Run manifest/artifact creation wired into an existing script (e.g. `build_move_matrix.py`, `eval_target_model.py`) | **NO — deferred** | Would require modifying an executing Track C script to write a manifest, which touches its execution path; more importantly, demonstrating "old vs new parity" for something that doesn't exist yet in the old system (no run currently has any manifest to diff against) isn't meaningful — there's no old behavior to preserve parity with, only a new behavior to add. Adding it without running the (expensive, multi-minute) actual build to prove it doesn't break anything would violate "don't launch large training/build jobs merely to validate architecture cleanup." **Precise reason for deferral**: cannot demonstrate parity for a responsibility that has no prior implementation to compare against, and proving non-interference would require re-running an expensive existing job. |
| 3 | Economic Evaluator (`economic.py`) wired to replace/front `src/backtest/cascade_backtest.py` or `src/backtest/engine.py` | **NO — deferred** | Would require either (a) modifying `cascade_backtest.py`'s/`engine.py`'s actual PnL computation (direct model-science-adjacent executable-behavior change, explicitly out of bounds without extensive validation), or (b) reconstructing the exact per-trade population those scripts operate on from a persisted parquet artifact to run `economic.py` against it as a read-only parity check. (b) was considered and rejected this pass: correctly reconstructing that population (the "book"-reweighted, argmax-per-bar-selected trade set behind `eval_target_model.py`'s console cost-sensitivity numbers) requires precisely replicating non-trivial selection logic from `build_target_matrix.py`/`eval_target_model.py`; doing that carelessly risks producing a **false parity claim** (matching numbers by coincidence, not by having actually replicated the logic), which would be worse than not attempting it. **Precise reason for deferral**: the risk of an incorrect-but-plausible-looking parity claim outweighs the value of attempting it without deeper, out-of-scope tracing work. |
| 4 | Experiment comparison (`comparison.py`) wired against two real historical runs | **NO — deferred** | Direct consequence of priority 2 being deferred: there are no two real `RunManifest`s to compare yet (nothing has ever written one). Comparison logic itself is tested against synthetic manifests (`tests/canonical/test_comparison.py`); wiring it against real data is blocked on priority 2. |

**Net effect**: one real, safe, demonstrated integration (config validation) that already produced a genuine new finding; three explicitly deferred with precise, non-generic reasons rather than attempted half-way.

## 6. Fresh-Agent Test (executed 2026-09-22, this pass)

Per the task's explicit requirement, this was actually run, not narrated. A fresh agent must be able to determine, from the repository alone:

| # | Requirement | How verified | Result |
|---|---|---|---|
| 1 | Project objective and conceptual architecture | Read `AGENTS.md` (98 lines) → `docs/ARCHITECTURE.md` | Both readable in under a minute; architecture's 12 responsibilities and their current implementations are explicit |
| 2 | Active/experimental/production/legacy implementations | `python -m src.canonical.cli status` (executed, see output below) | Prints all responsibility statuses + all 3 tracks' status from `project_manifest.yaml` in one command, no DB |
| 3 | Current vs. target config truth | `docs/CONFIG_AUDIT.md` §6 | Explicit, separated |
| 4 | Canonical commands | `python -m src.canonical.cli --help`-equivalent (subcommand list) | `status`, `validate-config`, `run-inspect`, `run-compare`, `verify` (the Tranche-2 record said `test`; renamed in the third pass because it now actually runs pytest and propagates its exit code) |
| 5 | Canonical storage/path conventions | `docs/STORAGE_CONTRACT.md` | Extends `DATA_LAYOUT.md`; both current |
| 6 | Where features/outcomes/models/predictions/reports/trades/results live | `docs/STORAGE_CONTRACT.md` §1-2, `docs/architecture/DATA_LAYOUT.md` | Both real paths (Track B/C current state) and target paths (canonical) given |
| 7 | How a run is reproduced and compared | `docs/EXPERIMENTS.md` §3-4 | Contract defined; honestly marked not-yet-wired where true |
| 8 | What is canonical vs. compatibility/legacy | `project_manifest.yaml`'s `responsibilities[].status` + `tracks[].status` | Explicit per-item, machine-readable |
| 9 | What tests/invariants apply | `docs/ENVIRONMENT.md` §7, `AGENTS.md` §7 | `pytest tests/canonical/ -q`, causality invariant named |
| 10 | What files to read first / historical areas to avoid | `AGENTS.md` §1, `docs/decisions/ADR_INDEX.md` | Explicit; ADR index prevents "read all 137" |
| 11 | How a materially different future predictive topology plugs in without another restructure | `docs/ARCHITECTURE.md` §3 ("What is NOT canonical") | Explicit: no responsibility assumes a cascade/gate/meta-layer/specific model count |

**Actual command output, captured live against this repository** (not fabricated):

```
$ python -m src.canonical.cli status
canonical docs: /opt/vtech/project_manifest.yaml

responsibilities:
  data_contracts       PARTIAL_IMPLEMENTATION
  features             INTERFACE_ONLY
  outcomes             INTERFACE_ONLY
  model_strategy       INTERFACE_ONLY
  validation           INTERFACE_ONLY
  calibration          NOT_IMPLEMENTED
  decision_policy      INTERFACE_ONLY
  economic_evaluation  PARTIAL_IMPLEMENTATION
  experiment           PARTIAL_IMPLEMENTATION
  model_lifecycle      INTERFACE_ONLY
  serving              INTERFACE_ONLY
  orchestration        INTERFACE_ONLY

tracks:
  A: legacy detection models -- ARCHIVED
  B: nq-50pt-econ (objectives framework / Research Director / Custom Test Lab) -- PRODUCTION_WIRED_NON_FUNCTIONAL
  C: MNQ Auction Cascade -- ACTIVE_RESEARCH_NOT_PRODUCTION_WIRED

$ python -m src.canonical.cli validate-config --path config.yaml --objective config/objectives/nq-50pt-econ.yaml
Config validation: config.yaml
  WARNING: section 'model hyperparameters' is present but confirmed DEAD ...
  WARNING: section 'training' is present but confirmed DEAD ...

Config validation: config/objectives/nq-50pt-econ.yaml
  ERROR: costs.slippage_model is set but costs.slippage_points is not ...
  WARNING: eval.walk_forward_folds declares 3, but the audit found real ... n_folds=1 ...
exit code: 1
```

**Verdict**: the Fresh-Agent Test passes for everything implemented so far. It does **not** yet pass for "train/evaluate a model canonically" — no such command exists, and `status`'s own output honestly says so rather than pretending otherwise. That gap is real, tracked as open technical debt (§9), not hidden.

## 7. Requirements audit (against the original master specification)

| Requirement | Status | Canonical file/path | Remaining gap |
|---|---|---|---|
| Map current reality (Phase 1) | COMPLETE | `docs/ARCHITECTURE.md`, `project_manifest.yaml` | None material |
| Derive clean-sheet conceptual architecture, not standardize legacy concepts (Phase 2) | COMPLETE | `docs/ARCHITECTURE.md` §0-3 | None |
| Legacy → canonical concept map | COMPLETE | `docs/ARCHITECTURE.md` (embedded in §1 per-responsibility), original chat report | Could be extracted into its own standalone table file; judged not worth the duplication risk |
| Target repository tree / OLD→NEW path map | COMPLETE | This file §1-2, `docs/STORAGE_CONTRACT.md` | None |
| Canonical workflow | COMPLETE | `docs/RESEARCH_WORKFLOW.md` | Not wired to an actual command chain yet (by design — see command surface row below) |
| Configuration architecture: current source of truth + validation | COMPLETE (current); PARTIAL (target) | `docs/CONFIG_AUDIT.md`, `src/experiment/config_validation.py` | Target composable-per-responsibility config is designed, not implemented as a replacement system (correctly deferred — would be model-science-adjacent) |
| Experiment specification | COMPLETE | `docs/EXPERIMENTS.md`, `src/experiment/manifest.py` | Not wired to any script (§5 priority 2) |
| Artifact contract | COMPLETE (contract); NOT WIRED (usage) | `docs/EXPERIMENTS.md` §2, `src/experiment/schemas.py` | Same as above |
| Comparison architecture | COMPLETE (contract + tool); NOT WIRED (real data) | `docs/EXPERIMENTS.md` §3, `src/experiment/comparison.py` | Blocked on Experiment being wired (§5 priority 4) |
| Canonical command interface | PARTIAL | `src/experiment/cli.py` | Only `status`/`validate-config`/`run-inspect`/`run-compare`/`verify` exist; no train/evaluate command — correctly documents legacy entry points instead of faking one |
| Result schemas | COMPLETE | `src/experiment/schemas.py`, `src/evaluation/economic.py` | Not applied to any real persisted artifact yet |
| Script hygiene | PARTIAL | `docs/SCRIPT_INVENTORY.md` | ~40 of ~130 scripts individually verified; rest classified by directory-level pattern, honestly flagged as such |
| Documentation hierarchy (README pointer, ADR classification, doc-currency correction) | COMPLETE | `docs/README.md` (pointer added), `docs/decisions/ADR_INDEX.md`, `docs/ARCHITECTURE.md` §4 (corrected per-file classification) | ADR index is bucket-level, not per-ADR (137 files — explicitly out of proportional scope, stated as such) |
| Universal agent-instruction architecture (`AGENTS.md`/`CLAUDE.md` split, no duplication) | COMPLETE | `AGENTS.md`, `CLAUDE.md` | None material |
| Nested `AGENTS.md` / `.claude/rules/` | COMPLETE (decision: none created) | This file, `AGENTS.md` §1 | Correctly not created — no subtree yet has genuinely local rules; revisit once a `src/*` subpackage gets a real (not interface-only) implementation with local invariants |
| Agent-optimized machine-readable manifest | COMPLETE | `project_manifest.yaml` | Requires manual upkeep discipline — noted in the file's own `supersedes_note` |
| ADR discipline | PARTIAL | `docs/decisions/ADR_INDEX.md`, `docs/decisions/ADR-136` | Bucket-level only, by explicit design choice (proportionality) — not a full per-ADR audit |
| Interfaces/contracts for durable responsibilities | COMPLETE (4 of 12 real, 8 interface-only by design) | `src/*` | Correctly not over-scaffolded further this pass, per explicit instruction |
| Causality as repository-wide invariant + tests | COMPLETE | `docs/DATA_CONTRACTS.md` §3, `src/common/causal.py`, `tests/canonical/test_causal.py` | Extracted/tested as a primitive; not yet required-by-construction in Track B/C's own code (would be a Track B/C code change, out of scope) |
| Test architecture (risk-prioritized, fast, DB-free) | COMPLETE | `tests/canonical/` (76 tests, ~0.26s, zero DB) | Does not add new tests to Track B/C's own existing suites — correctly out of scope (those already exist, e.g. `tests/test_entry_state.py`) |
| Data/code/artifact separation | COMPLETE (documented); N/A (no physical change) | `docs/STORAGE_CONTRACT.md` | No physical migration performed, by explicit instruction |
| Environment reproducibility | COMPLETE | `docs/ENVIRONMENT.md` | None material |
| Logging standardization | COMPLETE (principle stated); PARTIAL (enforcement) | `docs/ENVIRONMENT.md` §6 | Not mechanically enforced (e.g. no lint rule against bare `print()` for results) — judged disproportionate for this pass |
| Behavior-preservation validation for any integration | COMPLETE (for the one integration attempted) | This file §4-5 | N/A |
| Migration manifest + user cleanup plan | COMPLETE | This file | None |
| Fresh-Agent Test | COMPLETE, executed and persisted | This file §6 | None |
| Database/data safety boundary maintained | COMPLETE | This file §10 | None |

## 8. Follow-up NOT done (explicitly out of scope, listed so it isn't silently forgotten)

1. Full per-ADR status classification across all 137 ADRs (bucket-level only — `docs/decisions/ADR_INDEX.md` §4 states this explicitly).
2. The four deferred integrations in §5 (run manifest, economic evaluator, comparison-against-real-data — all blocked on either an expensive rebuild or a risk of false-parity claims).
3. `config.yaml`'s confirmed-dead `model hyperparameters`/`training` block was documented and is now mechanically flagged by the validator, but not edited or removed — a config-file change, left to the user.
4. No lint/CI enforcement was added for any of the new invariants (unknown-config-field detection, causal-primitive usage, result-persistence-over-console-output) — all are currently opt-in (a script/agent has to choose to run the validator/use the primitive), not mechanically forced. Flagged as a reasonable next step, not done here (would touch CI config, judged out of a documentation/architecture-only pass's proportional scope).
5. `docs/architecture/RESEARCH_DIRECTOR.md`'s own note about deleted Phase-19/20 modules, and other individual docs' internal staleness (e.g. `DATABASE_SCHEMA.md`'s row-count snapshot) were flagged but not corrected in place — out of scope (would be editing pre-existing docs beyond the minimal pointer/banner insertions this task's safety rules support).

## 9. Remaining technical debt (carried forward, for the user's prioritization)

- Nothing in this repository has been shown to have real trading edge net of realistic execution costs (this is a modeling/economics finding, not a structural one — restated here only because it's the thing all this structure now makes easy to check).
- The gate-induced population-shift confound in Track C (`docs/ARCHITECTURE.md` §1.7) is structurally documented but not tooled.
- The `nq-50pt-econ` walk-forward-folds discrepancy (§3 item 5, new this pass) is not root-caused.
- No calibration exists anywhere (`src/calibration` remains `NOT_IMPLEMENTED`).

## 10. Database / data / destructive-operation safety confirmation

Confirmed for both tranches of this restructuring: **zero** `psql` invocations, **zero** `psycopg`/SQLAlchemy/any DB client imports or connections, **zero** `mv`/`rm`/`git mv`/`git rm`/`git clean`/destructive `git reset` operations, **zero** files under `data/` created, moved, or modified (all `data/` access was read-only `find`/`ls`/`Read` for documentation purposes), **zero** large training/build jobs launched. The one *rewritten* pre-existing file (`AGENTS.md`) had its original preserved first.

> **CORRECTION (third pass).** The sentence that stood here — "`git status --short | grep -v '^??'` shows `AGENTS.md` as the only modified pre-existing tracked file from this task" — was **false, and contradicted this document's own §2 table**, which already listed `docs/README.md` and `docs/roadmap/ROADMAP.md` as Tranche-2 edits. It also failed to account for three pre-existing deleted junk filenames (`e()`, `et`, `"t output\" 2>&1"`) visible in `git status` at session start. The accurate, per-file attribution split is in **§16**. The substantive safety claims above (no DB, no data mutation, no destructive operations) remain accurate; the Git *evidence* sentence did not, and an unverified self-report inside a safety confirmation is exactly the kind of claim that should not have been written.

## 11. User-controlled cleanup plan

### SAFE TO REMOVE AFTER YOUR VALIDATION
Nothing. Both tranches created no superseded old paths.

### KEEP TEMPORARILY
- `docs/archive/AGENTS.pre-canonical-2026-09-22.md` — rollback reference for the one in-place rewrite; safe to keep indefinitely regardless.

### ARCHIVE RATHER THAN DELETE
- Not applicable yet — no existing implementation has been superseded by a `src/*` equivalent.

### DO NOT REMOVE
- Everything under `src/`, `scripts/`, `docs/architecture/`, `docs/decisions/` (all 137 ADRs), `data/` — all still authoritative/operationally required, none superseded by either tranche of this restructuring.

## 12. Next 5 highest-value follow-on tasks (ranked, not by ease)

1. **Wire `RunManifest` into one real Track C script** (e.g. `build_move_matrix.py`) as a genuine, validated integration — unblocks priorities 2-4 in §5 and is the single highest-leverage next step toward this restructuring actually changing daily workflow rather than just describing it.
2. ~~**Root-cause the `eval.walk_forward_folds=3` vs. observed `n_folds=1` discrepancy**~~ — **RESOLVED in the third pass** (§15.1): it is a two-engine config-contract incompatibility, not a discrepancy in one engine. The captured validator output above (§6) predates that correction and is retained verbatim as the historical record.
3. **Reconstruct one real trade population from a persisted Track C artifact and validate `economic.py` against `eval_target_model.py`'s console cost-sensitivity numbers** — the deferred-for-risk item in §5 priority 3, worth doing properly (not rushed) as its own focused task.
4. **Full per-ADR status pass** (§8 item 1) — now that the bucket-level index exists, a systematic pass (even automated-assisted) could convert "Accepted-until-shown-otherwise" into real per-file status headers.
5. **Add mechanical enforcement** (§8 item 4) — a pre-commit or CI check that runs `validate-config` and fails on ERROR, at minimum, turning the "fail loud" principle from opt-in into actually-enforced.

---

# THIRD PASS (2026-09-22) — canonical-contract + full functional-coverage pass

Scope: audit the code the canonical architecture was meant to describe, *before*
wiring anything into Track B/C. No integration was performed. No file under
`data/` was created, moved, copied, rebuilt or deleted. No database was accessed.
No model science was changed.

## 13. Files created (third pass)

| Path | Purpose |
|---|---|
| `docs/FUNCTIONAL_COVERAGE.md` | The bottom-up functional + persistence audit, reverse-coverage validation (§13), and both coverage summaries (§14, §15) |

## 14. Files modified (third pass)

| Path | Change |
|---|---|
| `docs/ARCHITECTURE.md` | Replaced the flat 12-responsibility model with the **two-plane model**; added §1.0 (12 responsibilities the audit found unowned); renumbered §2–§5 |
| `docs/STORAGE_CONTRACT.md` | Rewritten around the **full persistence map** (§2, 35 artifact classes, 9 columns each); added the path-status vocabulary and §0 designating `data_paths.py` canonical |
| `docs/EXPERIMENTS.md` | Rewritten §1–§3 against the hardened `manifest.py`/`comparison.py`; added §2.1 **required-vs-conditional artifact contract** |
| `docs/FUNCTIONAL_COVERAGE.md` | §5.2/§5.3 corrected (see §15 below); §2 U6 economic count corrected; §13–§15 appended |
| `docs/CONFIG_AUDIT.md` | Added the verified two-engine `walk_forward_folds` statement plus the full correction history of both earlier errors |
| `project_manifest.yaml` | `architecture_model: two_plane` + `planes:`; 12 new responsibility entries (24 total); stale `n_folds=1` and "4 divergent implementations" notes corrected; `open_critical_findings` expanded to 9 |
| `AGENTS.md` | Trimmed the dated ADR-136 in-place-rewrite exception to a one-line pointer here; **restored 5 durable rules** dropped in the Tranche-1 rewrite, found by diffing `docs/archive/AGENTS.pre-canonical-2026-09-22.md` (TREE_INVENTORY trigger row, ADR-028 `Present/Supersedes/Next` header note, progress-bar convention, ~7 GB/worker memory budget, `data/raw/` removal rule) |
| `src/experiment/manifest.py` | Rewritten: `SCHEMA_VERSION`, `DatasetRef`, `CodeIdentity` (non-defaulted `dirty`), `ValidationDesign`, `SCIENTIFIC_FIELDS`, `scientific_fingerprint()`, immutability enforcement in `write()`, version guard in `read()` |
| `src/experiment/comparison.py` | Rewritten: `COMPARED_DIMENSIONS`, `EVIDENCE_REQUIRED`, recursive leaf diffing, `controlled_comparison`/`uncontrolled_dimensions`/`missing_evidence` |
| `src/experiment/config_validation.py` | Removed the false `walk_forward_folds` warning; added the two-engine contradiction rule; **removed the hand-written `KNOWN_OBJECTIVE_KEYS` allowlist** in favour of reading `config/objectives/objective.schema.json` |
| `src/experiment/schemas.py` | Added `RunMetrics`, `OOF_PREDICTION_COLUMNS`, `validate_oof_columns()`; documented which contract artifacts deliberately have no schema |
| `src/evaluation/economic.py` | `ZERO_COST` → `zero_cost(contract_multiplier, *, fill_model)`; `t_stat` → `t_stat_iid`; `is_significant()` **removed**, replaced by an explicit deferral note |
| `src/experiment/cli.py` | `test` → `verify` (actually runs pytest and propagates its exit code); added `resolve_run()` (by run_id or path) and `_run_roots()` |
| `tests/canonical/{test_experiment_manifest,test_comparison,test_cli,test_config_validation,test_economic_evaluation,test_schemas}.py` | Rewritten/extended — **105 tests passing** |

## 15. Corrections made during the third pass

Recorded rather than silently fixed, so the corrections are themselves auditable.

1. **`eval.walk_forward_folds` — wrong twice.** First claim: "real runs use `n_folds=1` in every case" (false). Second claim, an over-correction: "`n_folds` is never 1" (also false, from a misread `sort | uniq -c`). Only mechanical enumeration of all **54** `folds.json` is reliable: 3×`n_folds=1`, 38×3, 1×4, 12×6. The verified statement is a **two-engine** one — see `docs/CONFIG_AUDIT.md`. A subagent independently repeated the *first* error by citing `project_manifest.yaml` back at me; an agent's agreement is not evidence.
2. **Arbitrary path invention, twice.** Pass 2 invented `artifacts/runs/`; its own "correction" replaced it with `experiments/runs/`, equally unadopted (`DATA_LAYOUT.md` §4 is titled "Target shape (recommended)"). Corrected to the **already-adopted** `research/custom_test/<objective_id>/<run_id>/` via `data_paths.custom_test_dir()`.
3. **The canonical layer duplicated existing superior logic, twice.** (a) Dataset identity — `src/training/label_store.py` already implements content-addressed identity **[TERMINOLOGY CORRECTED, ninth pass: it is contract-addressed — `label_contract_id()` hashes the semantic contract, never the parquet bytes; see §29]**; `DatasetRef.contract_id` now reuses it instead of a free-form `data_reference`. (b) Objective structural validation — `config/objectives/objective.schema.json` already sets `additionalProperties: false` and is enforced by `src/objectives/loader.py`; the canonical validator's hand-written allowlist duplicated it **and was wrong** (omitted `custom_optimization` and `cost_matrix`, producing false warnings on valid objectives).
4. **Durable rules were lost in the Tranche-1 `AGENTS.md` rewrite.** Found by diffing against the preserved pre-canonical snapshot; 5 restored. This is the concrete cost of that copy-first exception, and the reason the snapshot was required.
5. **A test failure was fixed in the fixture, not the module.** `test_run_compare_reports_controlled_comparison` failed because the fixture omitted validation/seeds/execution_assumptions — `comparison.py` correctly reported missing evidence. The module was behaving as designed; the fixture was the bug.
6. **Overstated Git-safety claim corrected** — see §16.
7. **Statistical overreach removed from `economic.py`.** `is_significant()` implied inferential validity for overlapping, autocorrelated trades. Removed; `t_stat` renamed `t_stat_iid` to make the assumption visible in the name. Inference is **DEFERRED** until a dependence-aware estimator exists — the evaluator is descriptive.

## 16. Git attribution — restructuring vs unrelated pre-existing changes

The earlier claim that "only `AGENTS.md` was modified among pre-existing tracked files" was **wrong**. Accurate split of `git status`:

| Path | Attribution |
|---|---|
| `AGENTS.md` | **This restructuring** (Tranche 1 rewrite, Tranche 2 hardening, third-pass trim + 5 restored rules) |
| `docs/README.md`, `docs/roadmap/ROADMAP.md` | **This restructuring** (additive pointer/banner insertions, Tranche 2) |
| `src/backtest/cascade_backtest.py` | **NOT this restructuring** — Track C modeling work (ADR-126 `t_fill` hold accounting) that predates and is independent of it |
| `D e()`, `D et`, `D "t output\" 2>&1"` | **NOT this restructuring** — pre-existing junk filenames already showing as deleted in the working tree at session start. No agent in this session deleted anything |
| all `??` untracked paths | **This restructuring** (new canonical docs, `src/`, `tests/canonical/`, `project_manifest.yaml`, `CLAUDE.md`) |

## 17. Third-pass requirement status

| AREA | STATUS | EVIDENCE | REMAINING GAP |
|---|---|---|---|
| 1. Bottom-up functional audit | **COMPLETE** | `docs/FUNCTIONAL_COVERAGE.md` §1–§12; 20 packages / ~84,400 lines, all mapped (§13.1) | `scripts/` mapped at inventory level (`docs/SCRIPT_INVENTORY.md`), not per-function |
| 2. Path / persistence / intermediate-data audit | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §2 — 35 artifact classes, producer→resolver→path→consumer→mutability | 8 classes still bypass the resolver; 2 outputs not persisted at all |
| 3. Identity derived from existing behavior | **COMPLETE** | `FUNCTIONAL_COVERAGE.md` §5, §11 (7 existing mechanisms); `src/objectives/loader.py` + `objective.schema.json` is the mechanism `research_objective_id` rests on | — |
| 4. Run-artifact contract | **COMPLETE** | `docs/EXPERIMENTS.md` §2.1 — 22 artifacts, required/conditional, each with its existing implementation | Nothing writes the 7 `PROPOSED-THIS-PASS` artifacts yet (by design — that is integration) |
| 5. RunManifest hardening | **COMPLETE** | `manifest.py`; `tests/canonical/test_experiment_manifest.py` (immutability, fingerprint sensitivity, schema-version guard) | `extra` is still unconstrained — acceptable only because the 16 `SCIENTIFIC_FIELDS` are fingerprinted and `extra` is not |
| 6. Comparison semantics | **COMPLETE** | `comparison.py`; 12 adversarial tests | Population overlap needs `selected_population.json` from both runs — not computable from manifests alone |
| 7. Economic evaluator review | **COMPLETE (descriptive); inference DEFERRED** | `FUNCTIONAL_COVERAGE.md` §10; `economic.py` corrections (§15.7) | Dependence-aware inference (`n_eff`) deliberately not implemented. **Deliberately NOT wired into Track B/C** |
| 8. CLI / config / result contracts | **COMPLETE** | `verify` runs pytest and propagates its exit code; `resolve_run()` resolves by run_id; validator now reads the real JSON Schema; `schemas.py` covers every contract artifact lacking a home | Config validation remains **PARTIAL** for nested/type semantics — and cannot see divergence between the ≥6 config-loading paths |
| 9. `AGENTS.md` streamlining | **COMPLETE** | Dated ADR-136 rationale moved here; 5 durable rules restored from the pre-canonical snapshot | — |
| 10. Documentation/status reconciliation | **COMPLETE** | Stale `n_folds=1` and "4 divergent implementations" claims removed from `project_manifest.yaml`; `FUNCTIONAL_COVERAGE.md` §5.2/§5.3 corrected; Git attribution split above | — |
| 11. Reverse-coverage validation | **COMPLETE** | `FUNCTIONAL_COVERAGE.md` §13.1–§13.4, §14, §15; `pytest tests/canonical/` = **114 passed**. Independent fresh-context adversarial review executed; outcomes and the 7 defects it found (2 of them live code defects in modules whose tests passed) are in `FUNCTIONAL_COVERAGE.md` §16 | `extra` on `RunManifest` remains unconstrained by convention rather than by code |
| DB-side validation | **DEFERRED — REQUIRES DATA-SIDE VALIDATION** | `psql` prohibited for this task | Any claim about hypertable contents is unverified |

## 18. First integration target — RECOMMENDED, then DONE (fourth pass)

**`src/search/controller.py:504` — write a `manifest.json` alongside the `run.json` it already writes.**

Why this one, and only this one:

- It is **purely additive**. Nothing reads `manifest.json` yet, so no existing consumer can break.
- The directory, the resolver (`custom_test_dir()`) and the write point already exist.
- Every required field is already available at that call site or one hop away: `objective_id`, the resolved spec, `folds.json`, `label_contract.json`, and `code_bundle`'s sha256.
- It immediately closes two of the nine open critical findings' preconditions: runs would record a seed and a git SHA, and a leaderboard row would gain a traceable `run_id`.
- It changes **no model science** — no label math, no features, no thresholds, no training objective.

Explicitly *not* recommended first: wiring `economic.py` (would change reported PnL numbers), enforcing `lifecycle/status.py` (touches promotion), or adding a `sources/labels/` `_LAYOUT` key (touches 14+ hand-built call sites).


---

# FOURTH PASS (2026-09-22) — contract enforcement + first integration

## 19. Changes

| Area | Change |
|---|---|
| `RunManifest` | Added `REQUIRED_FOR_CANONICAL_COMPLETE`, `missing_evidence()`, `evidence_level` (`CANONICAL_COMPLETE`/`INCOMPLETE`), both persisted and **recomputed on read**. `hypothesis` demoted from construction gate to evidence. Identity classified: `hypothesis`/`experiment_family`/`baseline_run_id` are comparison context and left out of the fingerprint (previously they forked identity on a prose edit); `parent_run_id` stays in as real lineage. `SCIENTIFIC_FIELDS` is now 12 fields |
| `DatasetRef` | Requires `contract_id` **or** `content_sha256`. A location alone is refused |
| `comparison.py` | Added `compare_results()` / `ResultComparison` / `RESULT_ARTIFACTS`. Setup and result comparison are reported separately; artifacts are expected only under the manifest condition implying the step ran |
| `cli.py` | `run-compare` prints evidence level, then `--- SETUP ---` and `--- RESULTS ---`, and states that result differences under an uncontrolled setup are observations, not findings |
| `economic.py` | Documented that **there is no separate spread term** — `slippage_points_per_side` subsumes crossing the spread; `portfolio_sim.py`'s three-way tick model is a *fill* model, not a cost-accounting model. Added `EconomicAssumptions.instrument` (justified by the confirmed 1.0/2.0/20.0 multiplier spread) and `Trade.entry_ts`/`exit_ts` (unused in every calculation; recorded because overlapping holding periods are exactly why `t_stat_iid` is biased, so persisting them is what makes the DEFERRED inference resolvable). **Deliberately not added**: `quantity` (`portfolio_sim.py:675` records that slippage does not scale linearly at size) and entry/exit prices (a second source of truth for `gross_points`, with no consumer). Still NOT wired into Track B/C — **SUPERSEDED by §21**: the spread/slippage and quantity/price decisions in this row were reversed by explicit user direction |
| Docs | `AGENTS.md` two-plane + `docs/architecture/*` reclassification; `ARCHITECTURE.md` two-engine/54-run and ≥10-implementations corrections; `EXPERIMENTS.md` §1/§3 aligned to the code; `STORAGE_CONTRACT.md` + `FUNCTIONAL_COVERAGE.md` cascade-backtest/cost-sensitivity destinations reconciled with the run-artifact contract; this file's "two passes" header corrected |

## 20. First integration (COMPLETE)

`src/search/controller.py` writes `manifest.json` alongside the `run.json`/`folds.json`/`search_result.json` it already writes, via the new Track B adapter `src/search/run_manifest.py`.

Properties, each covered by a test in `tests/canonical/test_search_run_manifest.py`:

* **Additive.** A test asserts `run.json`, `folds.json` and `search_result.json` are byte-identical after the manifest is written. No existing consumer reads `manifest.json`.
* **Best-effort.** `write_manifest()` never raises; failures are logged, matching the results-index and outcome-episode writes beside it.
* **Never fabricated.** Populated from values genuinely on the path: `objective_id`, the resolved objective dict (`config_snapshot`), `costs:` (`execution_assumptions`), `fold_split`/`fold_meta` (`validation`), `label_contract.json` (`datasets` — reusing `label_store.py`'s contract addressing [corrected from "content addressing", ninth pass, §29], key names verified against a real file on disk), and git HEAD **plus a real dirty-tree check** (`controller._git_sha` has none). **Ninth-pass correction**: this same `datasets` population had been surfacing `label_contract.json`'s `contract_sha256` as `DatasetRef.content_sha256` — a hash of contract metadata mislabeled as a content hash. "Never fabricated" was true for every OTHER field on this path; it was not true for that one. Fixed, §29.
* **Honestly incomplete.** Manifests from this path report `evidence_level: INCOMPLETE` with `missing_evidence = (hypothesis, datasets, seeds)` — or without `datasets` when a label contract exists. No hypothesis or seed is recorded anywhere on the search path, so neither is invented.
* **Immutability preserved.** A rerun rewrites idempotently; the same `run_id` with a changed resolved config is refused and the original is left intact.

**Blockers recorded rather than worked around** (scope deliberately not expanded):

1. **No seed.** `set_global_seed()` has zero callers in `src/search/`. Every manifest this path writes will be INCOMPLETE until the search path sets and records a seed.
2. **No hypothesis.** There is no field anywhere in the objective or spec for it. Supplying one requires a user-facing input, not a code change.
3. **`datasets` empty when `search.input != native_tf`.** Only that path writes `label_contract.json`. Other inputs have no contract mechanism to reuse, and a location-only `DatasetRef` is refused by design.


---

# FIFTH PASS (2026-09-22) — execution contract, provenance and evidence hardening

## 21. Changes

| Area | Change |
|---|---|
| Economic execution contract | `economic.py` rewritten around **execution prices**. `OrderType` MARKET/LIMIT/STOP, `Quote`, `Leg`, `execution_price()`. MARKET fills at executable bid/ask, never midpoint (long entry ask, short entry bid, long exit bid, short exit ask); slippage is applied adversely *after* that. STOP price is a trigger; once triggered it follows MARKET semantics at the quote at trigger time. LIMIT fills at its limit, improved only when upstream fill logic explicitly reports it (`reported_fill_price`), a fill through the limit is rejected, and **no slippage**. STOP_LIMIT not modelled — nothing requires it |
| Spread vs slippage | **Reverses §19.** The previous pass documented slippage as subsuming the spread. That double-counts the spread once real quotes are used. Now: bid/ask pricing *is* the spread cost; `slippage_points_per_side` is only further adverse movement, persisted, currently 0; latency not estimated |
| Quantity / prices | **Reverses §19** (by explicit user direction). `Trade.quantity` persisted; entry/exit prices are now the *source* of gross PnL rather than a duplicate of it. `gross_dollars = signed price change × multiplier × quantity`; `total_commission = commission_per_trade × quantity`; `net = gross − commission` |
| Commission | One combined all-in figure per completed trade (`commission_per_trade`), not split by fee category. `config/execution_costs.yaml` (new) + `load_assumptions()`: NQ $3.36; MNQ-equivalent $1.03 as an **independent** assumption; only the MNQ-equivalent *multiplier* is derived (0.1 × NQ = 2.0). `underlying` (market data, NQ) is separate from `instrument` (what is sized) — "MNQ-equivalent" implies neither MNQ data nor an MNQ purchase |
| Partial fills | Not supported and not invented: a `Trade` is a completed round trip, so commission is charged exactly once per trade |
| Code provenance | `src/search/run_manifest.py` no longer falls back to `CodeIdentity(git_sha="unknown", dirty=True)`. `RunManifest.code` may be `None` → missing evidence → INCOMPLETE. `code` added to `comparison.EVIDENCE_REQUIRED` so two runs both lacking it cannot read as agreeing |
| Invalid evidence | `compare_results()` reports an expected artifact that exists but cannot be parsed as `invalid_evidence` instead of silently skipping it; `ResultComparison.clean` is False whenever any evidence is invalid; CLI prints it |
| Docs | `AGENTS.md` two-plane wording; `ARCHITECTURE.md` §1.8 (≥10 implementations + execution contract) and §1.9 (RunManifest integrated into Track B search) |

Still DEFERRED: dependence-aware inference; wiring the evaluator into Track B/C.


---

# SIXTH PASS (2026-09-22) — objective contract, run/input/output standardisation

## 22. Changes

| Area | Change |
|---|---|
| Terminology | `research_objective_id` → the repository's existing `objective_id` in `RunManifest`, fingerprint, comparison, Track B adapter, docs, tests. Manifest `SCHEMA_VERSION` 1 → 2; v1 files are read and normalised, never rewritten or deleted; both keys with different values → refused |
| Objective contract | `objective.schema.json` annotated with `x-objective-role` (identity / default / meta / mixed, sub-roles inside session, labels, kpi) — the existing schema, not a second structure. Edited surgically (no reformat); all 9 existing objectives still validate. `src/experiment/objective.py` reads the tags (`objective_identity`, `objective_identity_sha256`) |
| Schema generalisation (for Track C) | `labels.scheme += first_passage_r`; `labels.target_observation += print_tape`; `labels.stop_rule` as an alternative to `mae_tolerance`; `labels.unit_definition`; `session.flatten`, `session.decision_grid_seconds`; top-level `directions`; `costs.execution_cost_profile`; `kpi.required` → `[primary]` |
| Track C objective | New `config/objectives/nq-r-first-passage-30m.yaml` (later renamed to `mnq-auction-firstpass.yaml`), every identity value copied from `cascade_policy.py` / `move_label.py` (test-asserted). Checked first that the existing `nq-atr-auction.yaml` is NOT Track C (10 min ATR first-touch vs 30 min R-multiple). **Not wired** |
| Run identity | `run_id = <objective_id>/<run_ts>` (logical); `run_ts` = directory leaf only; no `run_key` |
| Input contract | `DatasetRef` gained `semantic_type`, `store_id`, `resolution`, `identity_strength`; refuses absolute locations. `data_paths` gained `_LAYOUT["labels"]`, `labels_dir()`, `logical_path()`; `label_store.label_set_dir` routed through it (path-identical, tested). Adapter records relative location, `outcomes`, resolution |
| Output contract | `src/experiment/artifacts.py` — one registry of canonical classes, filenames, minimum content, manifest-derived applicability, and classification of existing files. `RunManifest.artifacts` became `class → files`. Economic artifacts keyed on a declared `economic_evaluation` responsibility, not on recorded cost assumptions |
| Economic naming | `EconomicAssumptions.instrument` → `economic_contract_unit`; `underlying` stays separate |
| Invalid evidence | `compare_results` validates every existing artifact on each side (parse + minimum content) **before** the presence comparison; one-sided corrupt files and schema-invalid files are `invalid_evidence` |

**Remaining gaps recorded, not worked around:** 13+ hand-built `sources/labels` sites and the `data_paths.py:192` fallback; whether Track C's `session_r_from_tape` is causal at decision time is UNVERIFIED; Track C manifest integration not started.

---

# SEVENTH PASS (2026-09-22) — Track C truth correction, identity classification hardening, evidence-completeness enforcement

Corrections to the sixth pass's own output, found by re-checking it against source rather than against the sixth pass's own claims. No DB, no data rebuild, no model-science change, no Track C manifest integration.

## 23. Changes

| Area | Change |
|---|---|
| **Track C's R definition — was wrong** | The sixth pass described `unit: R` as `session_r_from_tape()` on the CURRENT session. Actual runtime path: `build_move_matrix.py`/`build_target_matrix.py` call `r_before(levels, session, r_lookback=5)` (`session_profile.py:219-228`), the median of the 5 most recently **completed prior** sessions' own `r_value` (each itself a `session_r_from_tape` over that one session). The current session contributes nothing. `labels.unit_definition` is now a structured, schema-validated object (`source: prior_session_median`, `lookback_sessions: 5`, `excludes_current_session: true`, `per_session_metric: median_print_block_range`, `per_session_block_seconds: 300`) rather than free text, and a test asserts every field against the cited source lines |
| **Track C's economic sizing default — was wrong** | Set to `NQ`; corrected to `MNQ_equivalent` (the cascade's actual current default, ADR-127). The underlying market/data stays `NQ` — sizing and underlying are independent claims, enforced by keeping `execution_cost_profile` (sizing) separate from `traded products[].symbol` (underlying) |
| **`cascade_backtest.py` false 4× claim** | Corrected to 10× ($20/pt NQ ÷ $2/pt MNQ-equivalent, per `config/execution_costs.yaml`) |
| **Objective identity over-scoped, non-recursive** | `decision_grid_seconds` (sampling cadence) and `kpi.primary` (a ranking metric) reclassified `identity` → `default`. New optional top-level `economic_intent` (enum, currently one value: `positive_selective_net_expectancy_after_costs`) carries the durable goal `kpi.primary` no longer can. `traded products[].note` tagged `meta`. `objective.py` rewritten to resolve roles **recursively**: an `identity` container's untagged children inherit identity, but an explicit `meta`/`default` tag on a nested field (e.g. `note`) now removes it even though its container is `identity` — previously a whole `identity`-tagged array/object was included wholesale |
| **`config_snapshot` could read as complete when it wasn't** | New `RunManifest.config_snapshot_resolved: bool` (default `False`, excluded from the fingerprint). `missing_evidence()` now requires both a non-empty `config_snapshot` **and** this explicit claim. The Track B adapter always leaves it `False` — a Custom Test run sweeps multiple candidates with their own resolved hyperparameters, which no single run-level mapping can truthfully represent — while still recording the post-override objective plus any `spec.manual.yaml`/`topology.yaml`/`walk_forward.yaml` overlay already written to the run dir |
| **`store_id` alone could satisfy dataset version evidence** | `missing_evidence()`'s `datasets` check now also fails if any `DatasetRef.identity_strength == "store"` — a mutable store name identifies WHICH dataset, not WHICH version. `store_id` remains fully valid to construct and record |
| **`manifest.artifacts` could be polluted with compatibility-status keys** | `RunManifest.__post_init__` now refuses any `artifacts` key that is not a real canonical class from `artifacts.BY_CLASS`. The v1→v2 legacy-artifacts migration on `read()` was updated to match (re-classifies each legacy path via `classify_existing()`, drops what realises no canonical class, instead of bucketing everything under an invented `"unclassified"` key) |
| **Label-path standardisation, continued** | `data_root() / "sources" / "labels" / ...` replaced with `labels_dir()` in the 4 Track C entry points most relevant to the next integration: `scripts/features/build_move_matrix.py`, `build_target_matrix.py`, `eval_move_model.py`, `eval_target_model.py` (all path-identical, verified). Remaining hand-built sites (`src/webui/datasources.py`, `src/gpu_runner/job_exec.py`) left untouched — not directly relevant to the next integration |
| **`parquet_labels_15s` "disagreement" — was a false finding** | `data_paths.py`'s `parquet_labels_15s_dir()` uses `_prefer(layout(...), <legacy path>)` — the resolver's own standard canonical/legacy pattern, used identically in 23 other accessors in the same file. Reclassified `COMPAT`, not a bug. `docs/FUNCTIONAL_COVERAGE.md` §3.2 and `docs/STORAGE_CONTRACT.md` corrected |
| **`economic_contract_unit`** | Already renamed from `instrument` in the fifth pass; unaffected this pass (task item 7's economic-naming ask was already done — verified, not re-done) |
| **Stale doc claims from the sixth pass** | `docs/EXPERIMENTS.md`: "not yet wired" header corrected (Track B is wired); `schema_version: 1` → `2`; "Track C has no objective document" removed; `datasets`/`config_snapshot` field rows updated for the new completeness rules; `EVIDENCE_REQUIRED` corrected from "7 dimensions, code refused outright" to the actual 8 (code is evidence-required, not refused, since the "unknown" placeholder was removed in the fifth pass); run-root code blocks corrected from `<run_id>/` to `<run_ts>/` as the directory leaf. `docs/STORAGE_CONTRACT.md` §3's code block corrected the same way; its run-manifest row no longer says "nothing yet" / `NOT YET WIRED` for Track B. `src/experiment/manifest.py`'s module docstring no longer says the module is unwired |

## 24. Corrections made during the seventh pass

Documented per this session's own convention (see §15, §3): a defect found in this session's own prior output, not silently fixed.

1. Every "corrected" claim above was FOUND, not asserted — each traces to a specific source citation (`session_profile.py:219-228`, `cascade_backtest.py`'s point_value/instrument docstring, `config/execution_costs.yaml`'s multipliers) read directly, not inferred from the sixth pass's own prose.
2. The `test_manifest_artifact_keys_are_always_canonical_classes` fix required simultaneously updating the v1-legacy-artifacts read path (§23), which the initial `__post_init__` validation alone would have broken (a legacy manifest with `artifacts: ["run.json"]` would have failed to read at all, since the old migration invented an `"unclassified"` key). Caught by `pytest`, not by inspection.
3. Several existing tests encoded the pre-correction semantics as assertions (`test_track_c_underlying_is_nq_and_mnq_equivalent_is_sizing_only` expected `NQ`; `test_manifest_is_honestly_incomplete_and_names_the_gaps` expected a 3-item missing-evidence set; `test_bookkeeping_change_does_not_break_immutability` used the pre-fourth-pass tuple shape for `artifacts`). Per this session's instruction, these were updated to the corrected semantics rather than preserved for test-count continuity.

## 25. Seventh-pass requirement status

| AREA | STATUS | EVIDENCE |
|---|---|---|
| Track C R definition corrected | **COMPLETE** | `nq-r-first-passage-30m.yaml` (now `mnq-auction-firstpass.yaml`) `labels.unit_definition`; `test_track_c_identity_matches_cascade_policy_source_of_truth` (extended) |
| Track C economic sizing corrected | **COMPLETE** | YAML `costs.execution_cost_profile: MNQ_equivalent`; `cascade_backtest.py` docstring; test |
| Objective identity vs defaults corrected | **COMPLETE** | `objective.schema.json` role edits; recursive `objective.py`; 4 new tests (decision_grid/kpi.primary/note/target changes) |
| `config_snapshot` truthfulness | **COMPLETE** | `config_snapshot_resolved` field + `missing_evidence()` gating + adapter always `False`; 4 new tests |
| Dataset version evidence (`store_id`) | **COMPLETE** | `missing_evidence()` `datasets` check; 2 new tests |
| `manifest.artifacts` canonical-only | **COMPLETE** | `__post_init__` validation + legacy-read fix; 3 new tests |
| Label-path standardisation (directly relevant sites) | **COMPLETE** | 4 Track C scripts routed through `labels_dir()`, path-identical (verified); remaining sites recorded, not touched |
| `parquet_labels_15s` classified | **COMPLETE** | Reclassified `COMPAT`, matching resolver's own established pattern; not a code change |
| Stale docs reconciled | **COMPLETE** | `docs/EXPERIMENTS.md`, `docs/STORAGE_CONTRACT.md`, `docs/FUNCTIONAL_COVERAGE.md`, `manifest.py` module docstring — 10 distinct corrections, listed in §23 |

**Remaining gaps, unchanged from the sixth pass:** `src/webui/datasources.py` and `src/gpu_runner/job_exec.py` still hand-build `sources/labels` paths (not directly relevant to the next integration); whether Track C's underlying tape read is causal at decision time is UNVERIFIED; Track C manifest integration not started; no seed anywhere on the search path; the economic evaluator remains unwired.

---

# EIGHTH PASS (2026-09-22) — repository-wide `/opt/vtech/data` storage and artifact architecture

Independent, read-only re-derivation of the canonical storage architecture from the actual
live filesystem (111G, verified via bounded `find`/`du`/`ls`/schema sampling across seven
parallel inventory passes, one per major subtree) and actual producer/consumer code — not
from the existing docs, which were treated as hypotheses and cross-checked. No DB queried or
mutated; no data moved, rebuilt, copied, renamed or deleted; no training/backtests run.

## 26. Changes

| Area | Change |
|---|---|
| New evidence file | `docs/STORAGE_INVENTORY.md` — the full verified filesystem snapshot, producer→artifact→consumer tables, and current→canonical migration matrix (too large to inline in `STORAGE_CONTRACT.md` without burying its rules) |
| `docs/STORAGE_CONTRACT.md` extended | New §7 terminology, §8 canonical target tree, §9 filename rules, §10 format rules, §11 partitioning rules, §12 lifecycle/mutability classification, §13 resolver contract + this pass's changes, §14 risk register (pointer), §15 clean-sheet self-review, §16 unresolved decisions / prioritized sequence. Header corrected: `docs/architecture/DATA_LAYOUT.md` §2 ("Present inventory," dated 2026-09-03) and §4 ("Target shape") are now stale/superseded — confirmed by fresh measurement, not merely asserted |
| Top-level directory names affirmed canonical | `sources/`, `research/`, `models/`, `evaluation/`, `ops/`, `legacy/` — **not renamed**. `DATA_LAYOUT.md` §4's `market/`/`experiments/`/`_legacy/` proposal was never adopted (the reorg that happened used different, adequate names); extending `docs/FUNCTIONAL_COVERAGE.md` §5.2's existing "TARGET-UNADOPTED" verdict for `experiments/` specifically to the whole of §4 |
| Resolver gaps closed (additive, path-identical, all 5 verified with `pytest tests/canonical/ tests/test_tape_cascade.py` — 226 passed) | `_LAYOUT["live"]` + `live_dir()` added, closing H13 (`webui/live.py`, `webui/regmodel.py`, `execution/config.py` each independently hardcoded `data_root()/"live"` before); `_LAYOUT["session_profile"]` added, closing H14 (`session_profile.py::session_profile_dir()`'s default branch, feeding the cascade's structural SSOT); `scripts/search/regime_analysis.py`'s `regimes/` output, previously a hardcoded absolute path ignoring `VTECH_DATA_DIR` entirely, now `research_director_dir()/"regimes"`, closing H15 |
| Terminology fix | `src/experiment/artifacts.py`'s `"resolved_config"` artifact class renamed `"config_snapshot"` — the class name now matches the `RunManifest` field it realises; whether the snapshot is genuinely resolved stays the separate, explicit `config_snapshot_resolved` claim, never implied by the class name (task item 9). `docs/EXPERIMENTS.md` §2.1 row updated to match |
| Terminology fix (Track C, re-review) | `src/backtest/cascade_backtest.py::run_cascade_inmemory`'s `instrument: str = "MNQ"` parameter and its two output-payload `"instrument"` keys renamed to `economic_contract_unit: str = "MNQ_equivalent"` — the exact anti-pattern `EconomicAssumptions.economic_contract_unit`'s own docstring already warns against, present in a module the fifth pass's rename did not reach because the canonical Economic Evaluator is deliberately not wired into the cascade. Verified as the label's only use (assembled into an output dict, never used in any calculation) and its only caller (`src/webui/datasources.py:1199`, no explicit `instrument=` passed) before renaming |
| New findings, documented not fixed (each requires a behavior change, a database write, or touches more independent call sites than "small, unambiguous" allows) | `research/search/_fold_cache/` grew 14G (2026-08) → 39G (2026-09-22) with no eviction, ~89% of `research/`, nested inside a "never-delete" audit-trail root (STOR-1); `move_matrix`/`target_matrix` consumer scripts (`eval_move_model.py`, `build_target_matrix.py`, `eval_target_model.py`) default one generation behind their producer's current output — a live, present-tense correctness risk, not historical debt (STOR-2, also recorded as a new `open_critical_findings` entry in `project_manifest.yaml`); MLflow's `mlflow_artifacts` split-brain between `ops/` and the pre-reorg top-level directory, caused by MLflow pinning `artifact_location` per-experiment at creation time (STOR-3); Track C's outcome datasets still have no version identity beyond a directory-name suffix, applied inconsistently across `move_matrix` (source-edit-bumped) and `target_matrix` (two naming semantics mixed in one namespace) (STOR-4); systemic resolver bypass across 9+ `sources/raw/backfill` producer scripts (STOR-5); `parquet_options_surface` (2.5G, largest `sources/features/` subdirectory) has no confirmed in-repo reader (STOR-6); model/leaderboard lineage breaks at the same point on both the leaderboard side (no FK to a serving-bundle version) and the search-run side (`finalize_manifest.json` points at a `search_run_dir` with no `manifest.json`) (STOR-7); Track B's `RunManifest` integration exists only in the uncommitted working tree — zero `manifest.json` files found anywhere on disk despite `controller.py` calling the writer (STOR-8); orphaned loose top-level files (STOR-9); `ops/tmp` (zero call sites, files 4+ months old, no cleanup) and `legacy/checkpoints` (dead old-architecture model bundles alongside actively-written ingest-job JSON) each mix live and dead content under one directory name (STOR-10) |
| `project_manifest.yaml` | `storage_inventory: docs/STORAGE_INVENTORY.md` added to `canonical_docs`; `storage_sync` responsibility note extended with this pass's summary; new `open_critical_findings` entry `track-c-consumer-scripts-default-to-stale-dataset-version` (HIGH) |
| `docs/FUNCTIONAL_COVERAGE.md` | §3.2 addendum (move_matrix/target_matrix drift re-confirmed, `1s/` dead, `firstpass` family fully superseded per the project's own roadmap); §3.4 gained H13 (fixed)/H14/H15 rows + a STOR-5 pointer; §3.5 gained the `_fold_cache` growth finding; §15 gained an eighth-pass addendum item |

## 27. Corrections made during the eighth pass

1. The `evaluation/<objective>/` directories physically holding `latest_leaders.json`/`leaderboard_v*.jsonl` alongside `contract_v*.json` — duplicating what `leaders_dir()`'s own docstring says lives exclusively under `models/leaders/<objective>/` — was found during the §15 self-review pass (asking "are immutable scientific outputs clearly separated from mutable operational state?"), not during the initial inventory. Folded into `docs/STORAGE_INVENTORY.md` §2.5 and §3's migration matrix as `NEEDS_DECISION` in the same pass, per the task's own instruction to revise before finalizing rather than deferring a self-review finding to a future pass.
2. The initial plan to relocate `_fold_cache` under a dedicated `_LAYOUT["fold_cache"]` key was reconsidered and dropped: `fold_cache.py::_CACHE_ROOT = search_dir() / "_fold_cache"` already inherits `search_dir()`'s resolver backing and honours `VTECH_DATA_DIR`; adding a redundant top-level key for a private implementation detail of one module would have violated this pass's own "avoid unnecessary nesting" principle (`STORAGE_CONTRACT.md` §8). Recorded as a deliberate non-change, not an oversight.
3. `scripts/search/regime_analysis.py:39` also hardcodes `Path("/opt/vtech")/"data/sources/features/parquet_5s"` as a fold-cache input store — a second hardcoded path in the same file. Left unfixed: `parquet_5s` is confirmed absent from disk (§2.2 of `STORAGE_INVENTORY.md`), so choosing a replacement store is a model-input decision, not a pure path-resolver substitution; only the file's `regimes/` *output* path (a pure resolver substitution) was fixed.

## 28. Eighth-pass requirement status

| AREA | STATUS | EVIDENCE |
|---|---|---|
| Real filesystem inventory (read-only) | **COMPLETE** | `docs/STORAGE_INVENTORY.md` §1; seven parallel bounded inventory passes, 2026-09-22 |
| Producer→artifact→consumer map | **COMPLETE** | `docs/STORAGE_INVENTORY.md` §2, extending `docs/STORAGE_CONTRACT.md` §2 (not duplicated) |
| Canonical terminology | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §7 |
| Canonical target tree | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §8 — top-level roots affirmed unchanged, subordinate patterns specified |
| Filename / format / partitioning rules | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §9–§11 |
| Identity/version rules | **COMPLETE** (extends existing §2.1a/§4/§5, no new mechanism invented) | `docs/STORAGE_CONTRACT.md` §5, §12 |
| Resolver classification + small additive fixes | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §13; 5 code changes, 226 tests pass |
| Current → canonical migration matrix | **COMPLETE** | `docs/STORAGE_INVENTORY.md` §3, representative Track B + Track C rows |
| Storage/data integrity risk list | **COMPLETE** | `docs/STORAGE_INVENTORY.md` §4, STOR-1…STOR-10 |
| Clean-sheet self-review | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §15; one real finding (evaluation/ leaderboard duplication) surfaced and folded in during the review itself |
| Docs reconciled | **COMPLETE** | `docs/STORAGE_CONTRACT.md`, `docs/STORAGE_INVENTORY.md` (new), `docs/FUNCTIONAL_COVERAGE.md`, `docs/EXPERIMENTS.md`, `project_manifest.yaml`, this file |

**Remaining gaps, explicitly deferred (each is a behavior change, a database action, or exceeds "small/unambiguous" scope — not oversights):** Track C outcome-dataset content-addressing (STOR-4, requires rebuilding/hashing ~2G of existing data); `_fold_cache` LRU eviction (STOR-1, a runtime code change, already designed and deferred elsewhere); the MLflow artifact split-brain (STOR-3, requires a database admin action); `legacy/checkpoints`'s live/dead content split (STOR-10, touches 6+ active ingestion scripts); `sources/raw/backfill`'s systemic resolver bypass (STOR-5, 9+ independent scripts). Track C manifest integration was not started, per this pass's own instruction.

---

# NINTH PASS (2026-09-22) — targeted correction: contract identity vs. content identity

A targeted correction pass on the eighth pass's own output, found by auditing the actual
`label_store.py`/`run_manifest.py` implementation against what the eighth pass's docs claimed,
not by repeating the filesystem inventory. No DB, no data rebuild, no large-file hashing, no
Track C manifest integration.

## 29. Changes

| Area | Change |
|---|---|
| **Track B was mislabeled content-addressed — it is contract-addressed** | `label_contract_id()` hashes `normalize_label_contract(contract)`, the normalized SEMANTIC contract (label rule, horizon, stop/target semantics, resolution) — never the label parquet's bytes. `write_label_parts(..., force=True)` can replace every part beneath an unchanged `contract_id` directory, so the mechanism is idempotent by default, **not content-immutable**. Every "content-addressed"/"immutable" claim in `docs/STORAGE_CONTRACT.md`, `docs/STORAGE_INVENTORY.md`, `docs/FUNCTIONAL_COVERAGE.md`, `docs/EXPERIMENTS.md`, `src/search/run_manifest.py` corrected to "contract-addressed"/"idempotent by default, not content-immutable" |
| **`DatasetRef.content_sha256` was fabricated from contract metadata — fixed** | `src/search/run_manifest.py::_datasets_from_run_dir()` surfaced `label_contract.json`'s `contract_sha256` (a hash of the `_contract.json` metadata sidecar, `write_label_contract_snapshot()`) as `DatasetRef.content_sha256` (which claims to hash dataset CONTENT). The two fields answer different questions and the metadata hash cannot detect a `force=True` rebuild (it stays identical, since `_contract.json`'s bytes are a pure function of the contract + a preserved `created_at_utc`). Fixed: `content_sha256` is now always `None` on this path — no genuine content digest is computed anywhere in this codebase today, and none is fabricated to fill the gap |
| **Three distinct version concepts formalised** | `DatasetRef` gained a fourth field, `build_id: str \| None`, sitting between `contract_id` (semantic) and `content_sha256` (content) in `identity_strength`'s ranking (`content > build > contract > store`). Populated for Track B from a deterministic, order-independent fingerprint of the already-recorded part inventory (`{name, bytes}` per part) — genuinely available evidence, no new file hashing. `__post_init__`'s scientific-identity requirement extended to accept `build_id`. Backward compatible: `DatasetRef(**d)` on an old-shape dict missing the `build_id` key defaults it to `None`, no migration code needed (verified by test) |
| **Track C's proposed target design corrected** | The eighth pass's migration matrix said Track C's target identity should be "content hash of (source rev + build params)" — not a content hash by any definition. Corrected to a `contract_id` (Track C's own semantic contract, fields TBD) + `build_id` (realization fingerprint) + an upstream `DatasetRef` reference from target→move (reusing the existing `DatasetRef`/`RunManifest.datasets` mechanism for cross-dataset lineage rather than inventing a second one) |
| **One canonical outcome-dataset architecture for Track B and Track C** | New `docs/STORAGE_CONTRACT.md` §17: `sources/labels/<logical_dataset_id>/<contract_id>/` — not a new pattern, the *generalization* of Track B's existing `<resolution>/<contract_id>/` layout to use `move`/`target` in the same slot `5s`/`15s` already occupies. No new `outcomes/` wrapper directory (would carry no semantic/partition information `<logical_dataset_id>` doesn't already carry). Existing physical paths for both tracks preserved as `KEEP_COMPATIBILITY`; no files moved |
| **`live/` vs. `ops/` contradiction resolved** | `ops/live/` declared the canonical target (same lifecycle class as `ops/execution/`, `ops/paper/`; the codebase's own `ops/execution/bridge/` already proves external-facing file bridges belong under `ops/`, undermining the only plausible exception argument). Top-level `live/` stays the current physical location — no migration performed. `live_dir()` now returns `ops/live/` if it exists, else `layout("live")`; **not** implemented via the shared `_prefer()` helper, whose fallback-to-`candidates[0]` rule would have silently defaulted any fresh/empty `VTECH_DATA_DIR` to `ops/live` — caught by a dedicated test before it shipped (§30 below). A fourth independent hardcoded `data_root()/"live"` construction, missed by the eighth pass's H13 sweep, found in `scripts/ops/tail_live_signals.py` and fixed the same way |
| **Feature-hive lifecycle reclassified** | `sources/features/timeframe=*/` (and the rest of `sources/features/`) was called a "reproducible cache" with "no backup requirement." Corrected: `src/features/engine.py`'s own module docstring states it loads from TimescaleDB (confirmed via its `psycopg` usage), not from `sources/raw/` files directly, and `sources/raw/` retention is independently confirmed non-uniform (STOR-5) — so the hive's true reproducibility is conditional on TimescaleDB state this pass did not and could not verify (no DB queries permitted). Reclassified "derived feature store, conditionally reproducible," distinct from `_fold_cache` (confirmed genuinely pure, file-only inputs) |
| **Canonical filename/format/content contract completed** | New `docs/STORAGE_CONTRACT.md` §18: one table covering raw/source files through live/operational event streams, reusing `artifacts.py`/`schemas.py`/`label_store.py` for column-level detail rather than reproducing it |
| **Migration matrix completed** | `docs/STORAGE_INVENTORY.md` §3's "representative rows only" disclaimer retracted; expanded from ~20 rows to a per-area breakdown (§3.1–§3.7) covering every class identified in §1–§2, grouping truly homogeneous paths rather than repeating rows |
| **`_LAYOUT["session_profile"]`, `research_director/regimes` terminology** | Unaffected by this pass (already fixed eighth pass); re-verified consistent with the corrected `identity_strength` ranking wherever both were mentioned together |

## 30. Corrections made during the ninth pass

Self-referential correction log, per this session's own established convention.

1. **The `live_dir()` fix itself shipped a bug on the first attempt.** `_prefer(root/"ops"/"live", layout("live"))` looked correct by analogy to `execution_dir()`'s existing `_prefer()` usage, but `_prefer()`'s documented fallback ("return the first candidate when none exist") meant any fresh/empty `VTECH_DATA_DIR` — including every test using `monkeypatch.setenv("VTECH_DATA_DIR", tmp_path)` — would silently resolve to `ops/live` instead of `live/`, a real behavior change contradicting this pass's own "path-identical, no migration performed" claim. Caught by `tests/test_live_dir_resolver.py::test_live_dir_resolves_to_top_level_when_ops_live_absent`, written specifically to exercise the empty-environment case, before the docs were finalized. Fixed with direct `exists()`-then-fallback logic instead of `_prefer()`.
2. **The `_datasets_from_run_dir()` guard condition needed re-deriving, not just a field swap.** The original `if not (contract_id or sha): return ()` became `if not (contract_id or build_id): return ()` — a mechanical rename would have been wrong, since `build_id` is derived (from `parts`), not read directly like `sha` was; the derivation (`_build_id_from_parts`) had to be written and order-normalized (sorted by name) before the guard could reference it meaningfully.
3. **Considered and rejected**: adding a `_LAYOUT["fold_cache"]` key while touching `data_paths.py` again in this pass. Re-confirmed the eighth pass's own reasoning still holds (`fold_cache.py`'s `search_dir()/"_fold_cache"` already inherits resolver backing) — not re-litigated, just re-checked for consistency with the new `live_dir()` precedent before concluding no change was needed.

## 31. Ninth-pass requirement status

| AREA | STATUS | EVIDENCE |
|---|---|---|
| Contract vs. content terminology corrected | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §5a; `docs/STORAGE_INVENTORY.md`, `docs/FUNCTIONAL_COVERAGE.md`, `docs/EXPERIMENTS.md` all swept |
| False `DatasetRef.content_sha256` provenance removed | **COMPLETE** | `src/search/run_manifest.py::_datasets_from_run_dir()`; `test_contract_metadata_hash_is_never_exposed_as_content_sha256` |
| Three version concepts formalised | **COMPLETE** | `DatasetRef.build_id` (new field); `docs/STORAGE_CONTRACT.md` §5a; `test_dataset_ref_identity_strength_order` |
| Track C target design corrected | **COMPLETE** (mechanism only — no rebuild performed) | `docs/STORAGE_CONTRACT.md` §17.3 |
| One canonical outcome-dataset architecture | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §17; `docs/STORAGE_INVENTORY.md` §3.3; `test_canonical_outcome_path_pattern_is_topology_neutral` |
| `live/` vs. `ops/` contradiction resolved | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §8a; `live_dir()`; `tests/test_live_dir_resolver.py`; `test_live_path_status_is_internally_consistent_in_docs` |
| Feature-hive lifecycle corrected | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §12; `docs/STORAGE_INVENTORY.md` §4 STOR-12 |
| Filename/format/content contract completed | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §18 |
| Migration matrix completed | **COMPLETE** | `docs/STORAGE_INVENTORY.md` §3.1–§3.7 |
| Large migrations deferred | **COMPLETE (by design)** | No rebuild, no hashing, no `_fold_cache` change, no MLflow DB action, no physical `live/`/legacy relocation performed |
| Tests | **COMPLETE** | 10 new tests across `tests/canonical/test_objective_contract.py`, `tests/canonical/test_search_run_manifest.py`, `tests/test_live_dir_resolver.py` (new file); 249 passed, 0 failed, `tests/canonical/` + affected fast tests only |

**Remaining gaps, unchanged or newly deferred:** Track C's semantic contract fields (which `build_move_matrix.py`/`build_target_matrix.py` parameters are semantic vs. incidental) are not yet defined — a definitional exercise for the pass that executes §17, not this one; all STOR-1…STOR-10 gaps from the eighth pass remain open exactly as recorded there; `ops/job_state/` still lacks a `_LAYOUT` key (noted, not added, to keep this pass's code changes to the set already verified against tests).

---

# TENTH PASS (2026-09-22) — Storage Implementation Tranche 1: final canonical outcome-dataset syntax

A targeted correction-and-implementation pass on the ninth pass's own design. Found (not
repeated from a fresh filesystem inventory) by re-auditing `run_manifest.py`'s actual
`build_id` derivation and the ninth pass's `<logical_dataset_id>` pattern against this
pass's own stricter test: is a directory level carrying a real, single semantic dimension,
or two conflated ones? No DB, no training, no backtests, no Track C data rebuild or
multi-GB hashing, no physical `live/` relocation.

## 32. Changes

| Area | Change |
|---|---|
| **`build_id` mechanism retired and replaced** | The ninth pass's `_build_id_from_parts()` (a fingerprint of output filenames + byte sizes) is deleted — it was never genuine build/provenance evidence, only a directory-listing fingerprint. `src/search/run_manifest.py::_datasets_from_run_dir()` now always records `build_id=None` (no genuine code/config/upstream evidence is available on that adapter's path). A NEW, genuine mechanism, `src/training/label_store.py::write_build_sidecar()`, derives `build_id` from real evidence: `src.common.code_identity.git_identity()` (git SHA + dirty flag, extracted from `run_manifest.py::_git_identity()` so both share one mechanism), the caller's build/population configuration, and upstream `DatasetRef`-shaped lineage entries |
| **Contract-only dataset evidence downgraded to INCOMPLETE** | `RunManifest.missing_evidence()`'s `datasets` check now flags `identity_strength in ("store", "contract")`, not just `"store"` — a `contract_id`-only `DatasetRef` states what a dataset MEANS, not WHICH realization was used (the same `contract_id` can silently name different bytes after a `force=True` rebuild), so it is real but insufficient evidence for `CANONICAL_COMPLETE`. Only `build`/`content` now satisfy it |
| **`RunManifest.schema_version` bumped 2 → 3** | `DatasetRef.build_id`'s MEANING changed (not just its presence): a v2 `build_id` was part-inventory-derived and (together with `contract_id`) counted as sufficient dataset evidence; a v3 `build_id` is genuine-evidence-only and `contract_id` alone no longer suffices. v1 and v2 files remain readable (a v2 file's `build_id` value, however it was derived, is read back as-is); new writes are always v3; a `schema_version` newer than the running code is still rejected |
| **One outcome-dataset directory syntax, two dimensions, finalized** | `docs/STORAGE_CONTRACT.md` §17.1 replaced: `sources/labels/<outcome_family>/<resolution>/<contract_id>/`, splitting the ninth pass's single overloaded `<logical_dataset_id>` slot into `<outcome_family>` (WHAT KIND of outcome — `triple_barrier`, `directional_first_passage`, `candidate_level_reachability`; never `move`/`target`/a stage number/a model family/a version) and `<resolution>` (decision-time cadence — `5s`, `15s`; a genuine partitioning dimension, distinct from outcome semantics). No `outcomes/` wrapper directory. Existing Track B/C physical paths unchanged, preserved as `KEEP_COMPATIBILITY` |
| **Canonical filenames finalized** | `_contract.json` (unchanged, reused) + `_build.json` (new sidecar, `label_store.BUILD_NAME`, written only when genuine build evidence exists) + `part{NNN}.parquet` (zero-padded size-split parts, `label_store.outcome_part_name()`, generalizing and fixing the un-padded `labels_15s_part10.parquet` sort-order defect in the compatibility naming) or `part_<session-date>.parquet` (session-keyed, for Track C's per-session producers). No date/session in filenames for size-split parts (contract-hash partitioning per §11); session date only in the session-keyed variant, which is itself a legitimate physical grain, not a violation of that rule |
| **`_contract.json` gained an explicit `schema_version` field** | `label_store.CONTRACT_SCHEMA_VERSION = 1`, added to the sidecar payload AFTER hashing (like `created_at_utc`) and popped by the new `normalize_contract()` before hashing — so it never enters `contract_id` and never changes any EXISTING Track B contract directory's hash. Closes the "no explicit version field" gap the ninth pass's §18 table flagged and left open |
| **Track-neutral hashing added without touching Track B's existing mechanism** | New `label_store.normalize_contract()`/`outcome_contract_id()`/`outcome_set_dir()`/`write_outcome_contract()` — same sha256-of-normalized-dict algorithm `label_contract_id()` uses, but without Track B's TBM-specific `target_observation` default (which would otherwise silently pollute a Track C contract and mismatch the directory it's written under). `label_contract_id()`/`label_set_dir()`/`write_contract()` are UNCHANGED; a regression test confirms Track B's hash output is byte-identical to before this pass |
| **Track C's semantic contract fields classified** | New `docs/STORAGE_CONTRACT.md` §19: every `build_move_matrix.py`/`build_target_matrix.py` CLI field classified as semantic (→ `contract_id`, via `label_store.move_outcome_contract()`/`target_outcome_contract()`), build/population configuration (→ `_build.json.population_config`), or execution-only. `--scores` (Track C's move-model predictions) classified as neither a builder parameter nor an outcome — an UPSTREAM INPUT, recorded via the new `DatasetRef.semantic_type = "predictions"` |
| **`predictions` added as a distinct `DatasetRef` semantic type** | `src/experiment/manifest.py::DATASET_SEMANTIC_TYPES` gained `"predictions"` — a model's OOF/scored output is a claim about the future, not ground truth, and must never be recorded as `"outcomes"` or folded into another dataset's `build_id` |
| **`--canonical-write` support added to both Track C builders (not executed)** | `build_move_matrix.py`/`build_target_matrix.py` gained an opt-in `--canonical-write` flag: when set, computes the canonical contract + directory via the new helpers, writes `_contract.json`/`_build.json`, and switches per-session part filenames to the canonical convention. Default (unset) behavior is completely unchanged — the legacy `--out`/`move_matrix_v4`/`target_matrix` paths remain the default, no rebuild performed |
| **Stale Track C consumer defaults eliminated** | `eval_move_model.py --dir` and `eval_target_model.py --dir` are now `required=True` (previously silently defaulted to `move_matrix_v3`/unsuffixed `target_matrix`, STOR-2). `build_target_matrix.py --move-dir` now fails clearly at startup when required (`--context != none`) and absent, instead of silently defaulting to `move_matrix_v3`. No consumer picks a version-suffixed historical dataset implicitly anymore |
| **`live_dir()`'s ninth-pass auto-cutover reverted** | `src/common/data_paths.py::live_dir()` now always returns `layout("live")` — it no longer checks for or prefers `ops/live/` under any condition. `ops/live/` remains the documented future canonical target; switching requires an explicit future code change, not a filesystem probe. `docs/STORAGE_CONTRACT.md` §8a and `tests/test_live_dir_resolver.py` updated to match; no files moved |
| **New `docs/STORAGE_CONTRACT.md` §20** | One authoritative compact table (8 semantic classes: raw source, native features, session/tape features, outcome datasets, predictions, research runs, model bundles, operational/live) making the exact canonical write path, filenames and identity mechanism obvious without reading Track B/C code |
| **New `src/common/code_identity.py`** | `git_identity() -> (sha, dirty) | None`, extracted from `run_manifest.py::_git_identity()` so Track C's `write_build_sidecar()` shares one code-identity mechanism rather than a second copy of the same subprocess calls |
| **`docs/STORAGE_INVENTORY.md`, `docs/EXPERIMENTS.md`** | §3.3's Track C rows updated to the finalized target paths (`directional_first_passage`/`candidate_level_reachability`, not `move`/`target`); STOR-2 marked FIXED; STOR-4/STOR-11 annotated `[CORRECTED, tenth pass]`/`[UPDATED, tenth pass]` without rewriting their historical text; `EXPERIMENTS.md` §1's `datasets` row updated for `schema_version: 3`, the corrected `build_id` semantics, and `predictions` |
| **`project_manifest.yaml`** | `storage_sync` responsibility note extended with a tenth-pass summary; `open_critical_findings/track-c-consumer-scripts-default-to-stale-dataset-version` marked resolved (the stale-default HALF of it is fixed; the underlying STOR-4 version-identity mechanism is designed and implemented but not executed, tracked separately) |

## 33. Corrections made during the tenth pass

1. **`write_contract()` could not be reused as-is for Track C.** The first design draft planned to call the existing `write_contract()` for Track C's canonical writer too. Tracing it revealed `write_contract()` always calls `normalize_label_contract()` internally (Track B's TBM-specific normalization, including the `target_observation` default) — reusing it for a Track C move/target contract would have silently injected an irrelevant default into the stored JSON AND, worse, made the stored `contract_id` field disagree with `outcome_contract_id()`'s own hash of the same dict (the value `outcome_set_dir()` uses to build the directory name) — a self-inconsistent sidecar. Fixed by adding a separate, genuinely track-neutral `write_outcome_contract()` using `normalize_contract()`/`outcome_contract_id()` throughout, rather than parameterizing `write_contract()` after the fact.
2. **`normalize_contract()` needed to pop `schema_version` too, not just `contract_id`/`created_at_utc`.** Adding an explicit `schema_version` field to the contract sidecar (closing a documented gap) was initially drafted as just adding it to the write payload. Tracing `contract_matches()`'s read-modify-compare path showed that without also popping `schema_version` in the shared normalization function, a contract read back FROM DISK (now carrying `schema_version`) would hash differently from a freshly-constructed contract dict (without it) — spuriously breaking `contract_matches()` for every already-written Track B contract the moment this pass shipped. Fixed by popping it in `normalize_contract()` alongside the two existing bookkeeping keys, mirroring how `created_at_utc` is already handled.
3. **The canonical part-filename convention had to accommodate two genuinely different physical grains, not one.** The first draft assumed a single `part{NNN}.parquet` convention (Track B's existing size-split pattern, generalized). Tracing Track C's builders showed they write ONE FILE PER SESSION inside a bounded-memory streaming loop (not buffered size-split parts), and changing that write pattern was judged too risky to make part of a not-executed pass (buffering years of rows before a size-split write departs from the documented ~340MB/shard memory budget). `outcome_part_name()` was widened to accept either an integer (zero-padded size-split part) or a `date` (session-keyed part `part_<date>.parquet`) — both match one `part*.parquet` glob, neither carries track vocabulary, and the session-keyed variant lets Track C's canonical-write path use the finalized convention without a structural rewrite of its streaming loop.

## 34. Tenth-pass requirement status

| AREA | STATUS | EVIDENCE |
|---|---|---|
| `build_id`/`content_sha256` fabrication mechanism retired | **COMPLETE** | `src/search/run_manifest.py` (`_build_id_from_parts` deleted); `test_part_filename_and_size_inventory_is_not_build_identity` |
| Contract-only dataset evidence downgraded | **COMPLETE** | `src/experiment/manifest.py::missing_evidence()`; `test_contract_only_dataset_evidence_is_incomplete_but_build_or_content_satisfies_it` |
| `RunManifest.schema_version` bumped to 3, v1/v2 readable | **COMPLETE** | `manifest.py::SCHEMA_VERSION = 3`; `test_v2_manifest_is_readable_and_new_writes_are_v3`, `test_future_schema_version_is_still_rejected` |
| One finalized outcome-dataset directory syntax | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §17.1; `src/common/data_paths.py::outcome_dataset_dir()`; `src/training/label_store.py::outcome_set_dir()`; `tests/canonical/test_outcome_dataset_paths.py` |
| Finalized canonical filenames | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §17.1, §18; `label_store.py::outcome_part_name()`/`BUILD_NAME`/`CONTRACT_NAME` |
| Track C semantic contract fields classified | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §19; `label_store.py::move_outcome_contract()`/`target_outcome_contract()` |
| Predictions represented separately from outcomes | **COMPLETE** | `manifest.py::DATASET_SEMANTIC_TYPES`; `build_target_matrix.py --canonical-write`'s upstream lineage; `test_predictions_is_a_distinct_semantic_type_from_outcomes` |
| Canonical-write support added to Track C builders | **COMPLETE (not executed)** | `build_move_matrix.py`/`build_target_matrix.py --canonical-write` |
| Stale Track C consumer defaults eliminated | **COMPLETE** | `eval_move_model.py`/`eval_target_model.py --dir required=True`; `build_target_matrix.py --move-dir` explicit failure |
| `live_dir()` stabilized (no auto-cutover) | **COMPLETE** | `src/common/data_paths.py::live_dir()`; `tests/test_live_dir_resolver.py` |
| Documentation contradictions corrected | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §5a, §8a, §17–§20; `docs/STORAGE_INVENTORY.md` STOR-2/4/11 annotations; `docs/EXPERIMENTS.md` §1 |
| Final authoritative canonical-syntax table | **COMPLETE** | `docs/STORAGE_CONTRACT.md` §20 |
| Tests | **COMPLETE** | 272 passed, 0 failed (`tests/canonical/`, `tests/test_tape_cascade.py`, `tests/test_smoke_execution.py`, `tests/test_execution.py`, `tests/test_live_dir_resolver.py`, `tests/test_track_c_no_stale_dataset_defaults.py`); no DB, no training, no backtests, no data rebuild |

**Remaining items, explicitly deferred (not oversights):** Track C's existing ~2G of `move_matrix`/`target_matrix` data is not rebuilt or backfilled into the canonical layout — `--canonical-write` exists but has not been run; the producers' own version-suffixed OUTPUT defaults (`move_matrix_v4`, unsuffixed `target_matrix`) are unchanged (that is STOR-4's producer-side self-versioning, distinct from the consumer-side stale-default risk this pass fixed); all STOR-1/3/5/6/7/8/9/10 gaps from the eighth pass remain open exactly as recorded; `ops/job_state/` still lacks a `_LAYOUT` key.

---

# 2026-09-23 — first-passage objective document corrected

`config/objectives/mnq-auction-firstpass.yaml` had been given the cascade's identity (the sixth/seventh-pass rows above). That was wrong. The file now states the ADR-123 objective as amended by ADR-126: 60-second decisions, resting entry, session-close barrier, 1s high/low, candidate grid, `pnl_r` regression. The cascade identity remains `config/objectives/nq-Rscaled-mkt.yaml`, and `test_track_c_identity_matches_cascade_policy_source_of_truth` reads that file. No `sources/labels/firstpass*` bytes were copied.

---

# 2026-09-23 — objective runs use manifest.json as the only run-level file

`RunManifest` schema is 4. `passes` absorbs what `run.json` held, `config_snapshot` absorbs `contract.json`, and `validation` absorbs the fold design in `folds.json`. New objective-run writers call `merge_pass` and do not create those three files. Readers still accept them. `model.json` is the per-model record (type, hyperparameters, target, feature columns, join spec). Label contracts are not updated with a list of consuming runs. `training_matrix.parquet` is an optional pass-level population cache. OOF files omit feature columns; `src/experiment/analysis.py` joins them back for inspection and SHAP. The tenth-pass rows above that name schema v3 and `test_v2_manifest_is_readable_and_new_writes_are_v3` are the record of that pass; the test is now `test_v2_manifest_is_readable_and_new_writes_are_current`.

---

# 2026-09-23 — feature joins are as-of

`src/experiment/analysis.py` no longer equality-merges feature, context, or upstream-prediction files. It calls `as_of_index` and refuses a value whose availability is after the decision `ts`. Feature and context sources declare `availability` (`timestamp`, `bar_close`, `session_close`). Labels may still use the forward path; `AGENTS.md` §7's causality sentence was corrected in place to say that, and `docs/DATA_CONTRACTS.md` §3 is the contract. No run parquet was rewritten.

---

# 2026-09-23 — Track B custom_test and evaluation mapped, not copied

`docs/new_file_map.md` now names every `data/research/custom_test/` entry and the `evaluation/nq-50pt*`, `portfolio_sim/adr047_stop_first`, `move_analysis`, and `context_analysis` files, with producer, consumer, and a class (canonical, optional diagnostic, reproducible/derived, legacy compatibility, safe future deletion candidate). No dataset in those trees was copied or deleted. `evaluation/firstpass_trade_example.png` and `evaluation/newmodel_trade_example.png` moved to `data/adhoc/trade_examples/` with the same filenames. The runtime artifact list in `src/experiment/artifacts.py` was not changed; the file map records the extra weight files (`model.keras`, `model.pkl`, `meta.pkl`, `meta_estimator.pkl`, `scaler.pkl`) a Track B migration has to keep.

---

# 2026-09-23 — Optuna HPO shares one objective run

An HPO study is `research/runs/<objective_id>/<hpo_run_ts>/` with `hpo.json` and `trials_summary.parquet` next to `manifest.json`, and one `pass_<n>/model_<model_id>/` per trial (`trial.json`, `model.json`, compact OOF, `metrics.json`). `validation.split` is `last_range` with `n_folds` 1. `allocate_model_id` is the trial id. `src/experiment/artifacts.py` names `hpo.json` and `trials_summary.parquet` and applies them when `config_snapshot.study.kind` is `hpo`. No study was written and `research/search/` was not moved.

---

# 2026-09-23 — canonical run layout: audit remediation, steps 1–8

A review of the canonical run layout found the design sound and the one migrated run on disk non-conforming: it carried the legacy `run.json`/`contract.json`/`folds.json` trio with no `manifest.json`, its population cache sat at the legacy per-model path, its `model.json` omitted `target`/`feature_columns`/`inputs`, and its `oof_predictions.parquet` still carried feature columns. Eight steps, in dependency order. No label dataset was deleted; the non-conforming run directory was left in place.

| Step | Change |
|---|---|
| 1. Failing tests | `tests/test_track_c_no_stale_dataset_defaults.py` asserted argparse-level `required=True` on `--dir`, which the evaluators had correctly relaxed to "`--dir`, or `--objective` and `--run-ts`". Test updated to the real contract; the no-silent-stale-default guarantee is unchanged |
| 2. Track B classification | `classify_existing()` returned `unclassified` for **every** model weight file, including `model.ubj`. Added `MODEL_ARTIFACT_FILENAMES` (one `model_artifact` class over `.ubj`/`.keras`/`.pkl`/`meta.pkl`/`meta_estimator.pkl`), added the serving-bundle and report files to `COMPATIBILITY_ARTIFACTS`, and made patterns match a bundle path's final component so `models/<candidate>/model.keras` classifies like its migrated copy |
| 3. Path resolver scope | `run_layout.py` had grown model-type and stage vocabulary. Model identity and `model.json` content moved to `src/experiment/model_identity.py`; `CASCADE_PASS`, `xgb_hyperparameters`, `move_feature_spec`, `target_feature_spec` moved to `src/training/cascade_run_spec.py`. Seven importers repointed. `write_model_json` no longer creates an empty `report/` as a side effect (§21's own rule) |
| 4. Doc reconciliation | §20's authoritative table gained a **price tape** row (`tape_<YYYY-MM-DD>.parquet`; `partNNN` is a deferred option, not the name) and a real **predictions** row (`pass_<n>/model_<id>/oof_predictions.parquet`). Added the one-time statement that a label's home is `sources/labels/<outcome_family>/<resolution>/<contract_id>/` and `training_matrix.parquet` is a cache of a join. `new_file_map.md`: label→matrix rows retargeted to the pass-level cache, the legacy trio rows reclassified as written-in-error, the tape row retargeted, `_fold_cache` changed from `drop` to `keep` (it is bound at import by `fold_cache.py` and constructed by three more modules) |
| 5. Re-emitted run | `copy_pass.py` now projects the OOF through `compact_oof_columns` instead of copying the scores file, derives `feature_columns` as the population columns the OOF drops, reads `_contract.json` beside the population to record the outcome `contract_id`, and records the upstream OOF as its own `predictions` input carrying the producing `model_id` as `build_id`. Fixed `merge_pass`, which froze `datasets` after the first pass and so silently discarded pass 2's dataset identity; `RunManifest.write` now accepts an additive dataset change (`_only_adds_datasets`) while still refusing a changed or dropped ref. Re-emitted `nq-Rscaled-mkt/20260923T132829Z`: `manifest.json` only, pass-level caches, OOF 125→18 columns (362M→37M) and 77→27 columns (17.6M→5.3M), `model.json` with `target`/`feature_columns`/`inputs`, three dataset refs. `build_oof_analysis.py` rebuilds the full 125-column table from the compact OOF plus the cache, as-of enforced |
| 6. Economic boundary | `prepare_path`/`session_grid`/`parse_hhmm` extracted verbatim to `src/features/session_path.py` — path primitives, not economics, previously imported as **private** names from `portfolio_sim` by six modules. `EconomicAssumptions` gained optional per-leg slippage so the simulator's finer assumption is expressible in the canonical layer; `SimConfig.as_assumptions()`/`from_assumptions()` are the whole bridge and round-trip losslessly. The sim's defaults are **not** silently replaced by `config/execution_costs.yaml` ($5.00 round trip vs the NQ profile's $3.36 would restate every historical KPI); adopting the profile takes an explicit call. A test pins that both layers net the same dollars for one trade |
| 7. One run reader | `read_run_record()` extended to read a Track B `search`/`custom_test` run (`run.json` + `search_result.json`), so one function covers all three run shapes. The candidate list is surfaced as `candidates` and deliberately not promoted into `passes`. WebUI verified: `list_search_runs()` still returns 185 runs and `search_run_detail` still resolves. No directory moved |
| 8. Hold-out pin | `EvalContract.from_manifest(run_dir)` reads the pin from a run's manifest; `load_latest(objective_id, run_dir=...)` returns the manifest pin when the two agree and raises `ContractHashMismatch` when they **disagree**, rather than resolving by precedence. Callers passing no `run_dir` are unchanged, so the four pinned `nq-50pt*` objectives resolve exactly as before |

Deliberately not done: no Track B `custom_test` bundle was copied (62 runs); `research/search/` was not moved; `_fold_cache` eviction (STOR-1) is untouched; the non-conforming `nq-Rscaled-mkt/20260923T005724Z` directory was left on disk. `seeds` and `execution_assumptions` are still absent from a `copy_pass`-written manifest, so a re-emitted run is `INCOMPLETE` on those two plus `datasets` (the v4 label directories have no `_build.json`, so their refs are contract-only — truthful, not a bug).

---

# 2026-09-23 — Track B joins the canonical architecture (steps 1–5)

Track C already ran through the canonical layout. This pass put Track B on the same architecture, closed the manifest's evidence gaps where the evidence genuinely exists, and finished the economic consolidation. Copy-first throughout: nothing under `research/custom_test/` was moved, renamed or deleted, and no legacy compatibility artifact was retired.

| Step | Change |
|---|---|
| 1. Representative Track B run migrated | `nq-50pt-econ-lg-st/20260829T182514Z` → `research/runs/nq-50pt-econ-lg-st/20260829T182514Z/`. Chosen because it exercises everything at once: 3 base models on 3 native timeframes (one Keras, two pickled XGBoost), a `tf_oof_meta` pass 2, per-model scalers and serving files, a multi-class SHAP report, a run-level label-contract snapshot. Historical `candidate_id` is the model id. SHAP routed to its real owner (`xgb_5m`, per `shap_meta.json`), not the meta. The meta's 28 fitted columns split into 12 base-prediction columns (3 `predictions` refs naming their upstream `model_id`), 7 `h1_*` + 6 `d1_*` `context` columns (`bar_close`, `1h`/`1d`) and 3 calendar columns (`timestamp`) — `validate_model_inputs` clean. Track B writes no training matrix, so the canonical run correctly has no population cache |
| 2. Future Track B runs write it directly | Core extracted to `src/search/canonical_run.py::emit_canonical_run`; `scripts/research/migrate_custom_test.py` is now a thin CLI over it, and `src/search/controller.py` calls it when a search run finishes (best-effort, additive, skipped on `dry_run`). Weight/bundle files are **hard-linked** on the live path — verified same inode — so the canonical tree is complete at no extra storage. Post-hoc copying is no longer the mechanism for anything but pre-existing runs |
| 3. Evidence gaps closed, never fabricated | `merge_pass`/`merge_run_json` gained `seeds`, `execution_assumptions`, `code` and `extra`, each filling an empty field only. New `economic.assumptions_record()` is the single builder for a run's cost record: resolves `costs.execution_cost_profile` through `config/execution_costs.yaml`, refuses to read an unparsed `slippage_model` as 0.0, and records `declared_not_applied` when a run applied a different cost than its objective declared. Track C run `20260923T143018Z`: `seeds={"model":42}`, costs fully resolved from the declared `MNQ_equivalent` profile — only `datasets` still missing (the `*_v4` label dirs have no `_build.json`; `copy_pass` now reads one when present). Track B run: `execution_assumptions` closed and it surfaced the ADR-136 dead-config finding (`slippage_model: fixed_1pt` declared, none applied); `seeds`/`config_snapshot`/`datasets` left absent with the reason recorded in `extra.evidence`. **Fixed a real provenance error**: `merge_pass` stamped every manifest with the CURRENT commit, so a migrated historical run claimed code it never saw — `emit_canonical_run` now reads the run's own `finalize_manifest.json` `git_sha` (`4aee5bc…`), rejects the literal `"HEAD"` one bundle recorded, and refuses a single run-level identity when the fits disagree. `read_run_record` also now folds a Track B run's `spec.manual.yaml`/`topology.yaml` overlays into `config_snapshot` |
| 4. Economic consolidation finished | `portfolio_sim.trade_ledger()` prices the simulator's book through `economic.compute_trade_ledger`, handing fills over as `LIMIT` legs because their prices already include entry/stop/target slippage and `execution_price` adds none to a LIMIT leg — no double count. `portfolio_sim.cost_sensitivity()` delegates to `economic.evaluate_cost_sensitivity`. Historical behaviour preserved: the blotter's `net_usd` and the KPI block are unchanged, and `test_canonical_ledger_agrees_with_the_simulator_blotter` is the explicit comparison — equal per trade and in aggregate under non-zero commission and 3 contracts — that licenses treating the canonical layer as the authority |
| 5. No legacy artifact retired | New `tests/canonical/test_legacy_retirement_gate.py` pins both halves per file for `run.json`, `search_result.json`, `folds.json`, `feature_schema.json`, `finalize_manifest.json`, `compose.json`: a real consumer still reads it (so do not delete), and the canonical replacement exists (so migration is possible). When the last reader moves, that row fails — and the failure is the signal to retire the file and the row in one change |

Also closed two registry gaps the new cross-track test found: `classify_existing()` did not recognise bare `diagnostics.json` or `migration_provenance.json`.

New tests: `tests/canonical/test_both_tracks_one_architecture.py` (parameterised over every emitted run — one reader, one manifest, usable join spec, compact OOF matching `model.json`, downstream passes naming their upstream `model_id` with build identity, no unclassified file, every reported gap explained), `tests/canonical/test_legacy_retirement_gate.py`, and four economic-boundary tests in `tests/test_portfolio_sim.py`.

Deliberately not done: the other 61 `custom_test` runs were not migrated (the mechanism is proven and scripted; migrating them is a bulk operation); `research/search/` was not inventoried or moved; no legacy file deleted; `_fold_cache` untouched. One superseded intermediate Track C run created earlier in the same session was removed in favour of `20260923T143018Z`, which has the complete evidence record.

---

# 2026-09-23 — aggressive simplification: src/canonical retired, economic contract corrected

Four changes, all consolidation rather than new layers.

## 1. `src/canonical/` no longer exists

Eight of its twelve packages contained **zero** functions or classes — only docstrings restating `docs/ARCHITECTURE.md` §1.1–§1.12. A package that only documents is a document, so they were deleted and the definitions live in the doc alone. The four with real implementations moved to the obvious existing home:

| From | To | Why |
|---|---|---|
| `src/canonical/experiment/*` (manifest, artifacts, model_identity, analysis, comparison, config_validation, objective, schemas, cli) | `src/experiment/` | genuinely shared run infrastructure; `src/experiment/` is the same thing without the parallel tree |
| `src/canonical/evaluation/economic.py` | `src/evaluation/economic.py` | joins `_kpi.py`, `contract.py`, `leaderboard.py`; now exported from `src.evaluation` |
| `src/canonical/data_contracts/causal.py` | `src/common/causal.py` | a pure as-of primitive with no feature-store knowledge |
| `src/canonical/lifecycle/status.py` | `src/serving/lifecycle.py` | beside `promote.py`, the lifecycle implementation it governs |
| `src/canonical/{features,outcomes,models,validation,calibration,decision_policy,serving,orchestration}/` | **removed** | zero executable content |

CLI is now `python -m src.experiment.cli`; the DB-free suite is `tests/architecture/` (renamed from `tests/canonical/`, which was a phantom name once the source tree went). `project_manifest.yaml` lost `canonical_package_root`, and `canonical_module:` became `implementation_module:` with the eight documentation-only entries dropped.

Caught during the move: `objective.py` and `config_validation.py` computed the repo root as `parents[3]`, correct under the old nesting and one level too far under the new one, which silently pointed the objective schema at `/opt/config/`. Both now use `parents[2]`.

## 2. Economic contract: order intent is independent of resolved fill

The previous adapter rewrote resolved MARKET/STOP fills as `LIMIT` legs to stop slippage being applied twice. That preserved PnL but discarded order semantics and made slippage sensitivity silently ineffective. Replaced:

- `Leg.resolved_fill_price` (renamed from the LIMIT-only `reported_fill_price`) is valid on **every** order type and must name its `fill_source`; when present it is used verbatim and no slippage is added. A resolved STOP stays a STOP; a market entry stays MARKET.
- New `LegRole` (`ENTRY`/`EXIT_TARGET`/`EXIT_STOP`/`EXIT_TIME`) selects per-leg slippage, so `entry_slippage_points`/`stop_slippage_points`/`target_slippage_points` genuinely participate in `compute_trade_ledger` — previously they were recorded and only `slippage_points_per_side` was used. A role with no explicit value falls back to the generic number, so the single-number model is unchanged.
- `TradeResult` records each leg's order type, role, fill source and the slippage actually applied (0.0 for a resolved fill).
- **Two sensitivities separated**: `evaluate_cost_sensitivity` (fixed fills; commission/multiplier only) now *refuses* a scenario that varies slippage when every fill is resolved, because it could not take effect and would return an identical ledger under a misleading label. `evaluate_execution_sensitivity(scenarios, resimulate=...)` is the re-derivation, and `portfolio_sim.execution_sensitivity()` supplies it by re-simulating the path.
- `portfolio_sim._legs_for()` maps exit reason to real intent: `stop` → STOP with its trigger, `target` → LIMIT when no target slippage is assumed and MARKET when the fill model fills *through* the level (a limit order cannot), everything else → MARKET with `EXIT_TIME`.
- Databento TBBO seam: a higher-fidelity source returns legs with `quote` and/or `resolved_fill_price` and `fill_source=FILL_SOURCE_TBBO`, keeping its order type and role. Nothing else in either module changes.

Historical PnL unchanged — `test_canonical_ledger_agrees_with_the_simulator_blotter` still holds per trade and in aggregate.

## 3. Track B historical migration

All **42** fitted `research/custom_test/` runs migrated to `research/runs/<objective_id>/<run_ts>/` — 56 models, **0** missing weights, **0** missing OOF, **0** missing metrics, 8 with a report set. Two lineage gaps found and fixed while migrating:

- a `primary_meta` candidate records its bases in `meta.json`, not `compose.json`; both are now read.
- a single-base meta names prediction-derived columns without a `<base>__` prefix (`primary_confidence`, `is_up`); with exactly one base the attribution is unambiguous, so they are recorded as that base's prediction columns.

Link policy corrected per instruction: hard-link only large immutable fitted artifacts (weights, `.pkl`, the SHAP `.npz`); small JSON/config/provenance files are copied, since a hard link on a small JSON saves nothing and makes an edit to one path silently change the other.

**Deleted** (0.376 GB), each verified to have no reader: 139 `folds/fold_NN.parquet` (reproducible; `fold_NN.json` kept, it is the per-fold metrics source), 6 nested `models/*/bases/` weight duplicates (`predict.py` reads compose's `bases` *list*, not this directory), 35 `progress*.json` on finished runs.

**Not deleted, with reasons** — see the deliverable summary. In particular the run-level `oof_<candidate>.parquet` files were checked as byte-identical to their per-model twin but are still read by `src/webui/datasources.py:541` and `src/backtest/runner.py:270`.

## 4. New Track B runs stop reconstructing evidence

`build_manifest` gained `seeds` and `datasets` parameters, and `controller.py` now passes what the runtime genuinely holds: `seeds={"fold_subsample": 42}` (applied on every search path by `_subsample_folds`) and `runtime_datasets(objective)`, which derives the label `DatasetRef` from the in-memory objective instead of re-reading the `label_contract.json` the run just wrote. `execution_assumptions` is now built by the one canonical builder (`economic.assumptions_record`) rather than copied raw from the objective's `costs` block, so a declared cost profile is resolved and an unparsed `slippage_model` is never read as zero. Both new parameters fall back to the on-disk reconstruction, so a migrated or third-party run still produces a manifest.

Verification: **573 passed, 34 skipped** on the architecture + simulator + execution suites; full suite **36 failures, identical to the pre-existing baseline — zero regressions**.

---

# 2026-09-23 — narrow repair of the interrupted refactor

No layout change. Three bugs in the half-converted code, and the docs that still described the previous HPO note.

- `eval_target_model.py` was still unpacking fold rows as month lists. It now uses `_fold_masks`, and both Track C evaluators write `validation.split` from the schedule rather than a hardcoded `expanding_month_blocks`.
- The persisted Track C ledger used `decision_price` → `price`. It now uses the fill (`price`) as entry and `tgt_level` / `stop_px` as exit, keeping the full-stop charge on a row that did not reach.
- `runner._resolve_oof_source` raised `NameError` on `search_root` when resolution failed. The error names `run_roots()`, and a canonical `pass_<n>/model_<id>/oof_predictions.parquet` resolves.
- `hpo_model_search.py` scored whatever `_build_folds` built, then wrote `last_range` / `n_folds=1` and `len(y_test or [])`. Provenance now records those folds. `y_oof` is counted without a boolean test of the array. The Optuna objective is unchanged. The earlier note in this file that the writer was not landed described the layout pass; the writer is this script, and it does not emit a per-trial OOF or weight file.
- `research/runs/` is the primary new-run root. `data/research/custom_test/` is gone. `research/search/` remains a compatibility root. `src/canonical/` stays removed.

## 2026-09-24 — additive independent NQ research (ADR-138)

Added `src/features/nq_selective.py`, `src/backtest/nq_selective.py`,
`scripts/research/train_nq_selective.py`, `scripts/research/audit_nq_selective.py`,
`tests/architecture/test_nq_selective_{features,execution,policy}.py`,
`config/objectives/nq-selective-intraday.yaml`, `docs/research/NQ_SELECTIVE_PROTOCOL.md`,
`docs/research/NQ_SELECTIVE_RESULTS.md`, and ADR-138. Artifacts use canonical
`data/research/runs/nq-selective-intraday/`; no parallel path resolver.

Existing architecture, contracts, config audit, feature reference, tree inventory,
roadmap, changelog, ADR index and project manifest receive additive notices/records
only. No pre-existing text, source, matrix, or artifact is deleted/replaced; no
preserve-then-rewrite exception is used. Existing dirty user changes remain intact.
The experiment is rejected and its reserved period unconsulted. 74 relevant tests
pass; objective schema and CANONICAL_COMPLETE manifest validate. A second run
reproduces the first completed ledger exactly after reporting/cost-provenance fixes.

## 2026-09-24 — TBBO profile-reclaim experiment (ADR-140)

Added `src/features/nq_profile_reclaim.py`,
`src/backtest/nq_profile_reclaim_quotes.py`,
`scripts/research/run_nq_profile_reclaim.py`,
`tests/architecture/test_nq_profile_reclaim.py`, the protocol/results records,
and ADR-140. This is a standalone experimental policy: it does not replace or
wire into Track B/C, serving, paper trading, or the raw tape store.

Before additive status/index/changelog notices, prior copies were retained in
`docs/archive/pre_profile_reclaim_20260924/`. No source or data file was moved,
deleted, or overwritten. The immutable run is
`data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/`; it is rejected,
and its July–September reserve was not opened.


## 2026-10-06 — normal option subscription and vendor definition acquisition

The user requested implementation of the agreed normal lifecycle, including
vendor-only definition acquisition and mapping preservation during rotation.
For this authorized in-place exception, all existing files are preserved before
edits under `archive/code_snapshots/20261006_normal_subscription/`, retaining
their relative paths and the pre-existing uncommitted work. This covers
`src/acquisition/{live_streamer,rotating_stream}.py`, `Makefile`,
`tests/test_{live_options_subscription,definition_gap_detection,rotating_dbn_stream}.py`,
`docs/ARCHITECTURE.md`, `docs/architecture/{DATA_PIPELINE,DATABASE_SCHEMA,TREE_INVENTORY}.md`,
`docs/roadmap/ROADMAP.md`, `docs/changelog/CHANGELOG.md`,
`docs/decisions/ADR_INDEX.md`, and this manifest. No old path is moved or deleted.
The historical parser remains available; active ingestion stops reconstructing
definition rows. New files: `src/acquisition/definition_acquisition.py`,
`tests/test_definition_acquisition.py`, ADR-156 and lifecycle operations guidance.
Validation and deployment evidence are recorded in the operations guidance.


Final validation: 178 acquisition tests, scope consistency, compilation and
scoped whitespace checks pass. Deployed PID 3255074 at 16:27:26 UTC; all 2,814
subscribed symbols have usable vendor definitions, zero pending/absent rows.
2,229 active legacy synthetic rows upgraded and 85 missing rows added through
exact-zero-cost acquisition. Full evidence and remaining roll observation:
[normal lifecycle operations](operations/NORMAL_OPTION_DEFINITION_LIFECYCLE.md).

## 2026-10-06 — bounded raw gap recovery (ADR-157)

The operator authorized the recovery plan's next steps. New recovery modules extend the mature acquisition/ingestion responsibilities; existing loaders remain in place. `scripts/data/recover_databento_gaps.py` executes the reviewed request manifest; `scripts/data/rebuild_gap_outputs.py` creates isolated downstream candidates and preserves files before publishing accepted products. No existing source/raw file or DB target row is deleted.

Before documentation and Makefile edits, each touched original was copied to `archive/code_snapshots/20261006_gap_recovery/` with its relative path retained. This is the preserve-then-edit exception for this authorized implementation. Before a downstream file replacement, the original is separately copied under the recovery run's `preserved/` tree and recorded in its publication manifest.

A follow-up preserves the initial recovery acquisition module and its test under `archive/code_snapshots/20261006_gap_recovery_timeout/` before raising timeseries read inactivity from 20 to 60 seconds after measured vendor preparation timeouts. Metadata calls retain a 20-second timeout.

Before the advisory-lock wait correction, module/test copies were preserved under `archive/code_snapshots/20261006_gap_recovery_advisory_wait/`. Before vectorizing timestamp CSV formatting, the module was preserved under `archive/code_snapshots/20261006_gap_recovery_csv/`. SQL TEMP integration tests verify unchanged microsecond precision, nulls, multiplicity and retry behavior. The advisory queue uses nonblocking probes with a 600-second deadline; the 2-second relation-lock timeout remains in force.

Before gating the unsafe broad feature rebuild, the recovery runner was preserved under `archive/code_snapshots/20261006_gap_recovery_feature_gate/`. The blocked branch leaves all existing feature files untouched. The first option publication attempt was stopped during its read-only snapshot phase before permanent writes; its artifacts remain under `data/tmp/gap_outputs_20261006_options_1758/`. The bounded retry uses the new `data/tmp/gap_outputs_20261006_options_publication_retry/` path, keeps original candidate paths, and records preserved affected DB rows, SQL plans and hourly commit checkpoints.

## 2026-10-14 — isolated public export for X

The original operational checkout is preserved at `/opt/vtech`; public changes
exist only in the separately cloned destination on `publication/platform-showcase`.
This is copy-first, not an in-place rewrite of the operating platform.

Public-only changes: synthetic demo and its explicit-path test gate, packaging
namespace/optional dependencies, portable README/setup, notebook import and
example IDs, third-party notices, disclosure cleanup, and publication scope.
Excluded archives, market data, credentials, and local state were never copied.
The source working tree includes uncommitted work, whose current bytes were
intentionally included when allowlisted. No private source history was copied.
The destination's existing `Opportunity Analysis` is preserved unchanged.
No files are moved/deleted, no DB is accessed, no data is rebuilt, and no push
is authorized by this preparation.


## 2026-10-15 — working Transformer showcase for X

The user authorized the working showcase and README. Implementation is isolated
in the X checkout on `showcase/transformer-robin`, based on `4e498b14`; the
operational `/opt/vtech` checkout and original accepted run remain untouched.
Before edits, existing files were copied with relative paths to
`archive/code_snapshots/20261015_transformer_showcase/` (ignored, local retention).
This preserve-then-edit exception covers README, pyproject, Makefile, CI,
conftest/scope consistency, transformer_plugin, architecture/publication/layout/
roadmap/changelog docs and this manifest. No existing path was removed or moved.
New file-backed showcase modules reuse canonical run paths and present an
allowlisted exact synthetic fixture; the original operational deployment bundle
is excluded. Copied workflow modules/configs retain the frozen objective and
promotion gate. Public projections record original evidence hashes rather than
silently rewriting promotion receipts. See `docs/examples/TRANSFORMER_WORKFLOW.md`.

The same preserved-original exception also covers `.gitignore`,
`project_manifest.yaml` and `docs/RESEARCH_WORKFLOW.md`. Root snapshots remain
local and ignored; the only public binary additions are the allowlisted
synthetic fixture and the actual console screenshot under `docs/assets/`.


## 2026-10-15 — preserve both README entry paths

At the user's request, README now begins with an explicit choice between the
existing platform/model-stack guide and the added Robin Transformer showcase.
Original offline-demo, source-map and optional-integration sections retain their
published bodies and order; the complete Robin instructions follow them. This
is a documentation-only refinement with no runtime changes. Before edits,
README, this manifest and CHANGELOG were preserved under the ignored local
`archive/code_snapshots/20261015_readme_entry_points/` directory. No path removed
or remote push performed.
