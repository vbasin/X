# ADR-122: Structural market context (prior-session VP + CVD) and the closed research loop

## Status
Proposed — 2026-09-09. No code written under this ADR yet.

## Date
2026-09-09

## Context

The operator asked how the relationship between price, **prior sessions' volume
profiles**, and **cumulative volume delta** could be implemented alongside the
existing rule-based `vwap_cross` engine, and then used to drive an automated
learning system capable of discovering new relationships on its own.

We inventoried the repository and independently surveyed the current literature
before answering. Both are summarised below; the papers are filed under
[research_cases/papers](../../research_cases/papers) as entries 10–13 of
[INDEX.md](../../research_cases/INDEX.md).

### Finding 1 — the feature the operator is asking about does not exist

`src/features/volume_profile.py` produces five `vp_` features
(`vp_poc_dist`, `vp_va_high_dist`, `vp_va_low_dist`, `vp_in_value_area`,
`vp_poc_slope`), and all five are computed on a **rolling window inside the
current session**. There are no prior-session levels anywhere in the feature
catalog: no prior POC, VAH, VAL, prior high/low/close, no naked/untested POCs,
no value-area overlap, no open-type classification.

The logic already exists — but in the wrong plane. `src/external_models/vwap_cross.py`
computes prior-session VP levels for its `vp_level` profit-target mode. It has
never been promoted to `src/features/`, so no model has ever seen it.

### Finding 2 — CVD is present as magnitude, absent as structure

`src/features/order_flow.py` (15 `of_`), `large_trade_cvd.py` (4 `lt_`) and
`trade_location.py` (2 `tl_`) give session-reset CVD, its differences at
6/12/24/60 bars, z-scores, and a **binary** divergence flag. What is missing is
CVD *relative to price structure*: delta accumulated at a level (absorption),
divergence magnitude rather than sign, cross-session normalisation, and delta
measured from the last VWAP cross. Those are the quantities the operator's chart
is actually being read for.

### Finding 3 — the research loop is open, not closed

`src/research_director/` implements assess → research → plan → reflect, with a
tool registry, pgvector memory, and cost governance. Accepted specs are written
to `data/ops/experiment_queue/`. **Nothing drains that queue.** There is no
executor, no SOTA library, no de-duplication, no scheduler, and no feedback path
from experimental result back into hypothesis generation. The systemd timer is
disabled. In its present form the system cannot learn, only propose.

### Finding 4 — the evaluation substrate is half-built

Purging and embargo **do exist** and are on by default in the production search
path, contrary to an earlier draft of this ADR:
`src/search/walk_forward_calendar.py` defines `DEFAULT_PURGE_SECONDS = 600` —
exactly the TBM horizon — drops the last purge window from every train fold, and
`resolve_purge_rows()` emits an explicit warning when a `purge_bars` row count
implies a duration shorter than the horizon. Embargo defaults are set in
`src/search/custom_test.py` (`embargo_bars: 120`) and applied by
`output_meta.py`, `meta_stack.py` and `tf_oof_meta.py`.
`src/training/walk_forward.py` splits on calendar-month boundaries where the
overnight gap already exceeds the label horizon.

What is genuinely absent is everything that corrects for **selection bias across
runs**:

- No durable, cross-run trial ledger. Optuna reports a per-study `n_trials`, but
  nothing accumulates `N` across searches, so a Deflated Sharpe Ratio cannot be
  computed.
- No PBO/CSCV estimate is produced for any candidate.
- Holdout periods are readable by the search process, so repeated evaluation
  silently converts a holdout into a training set.
- The label scheme's MAE stop-first heuristic is already flagged in
  [ROADMAP](../roadmap/ROADMAP.md) as teaching the model that intrabar stop-outs
  are wins.

### What the literature says

**[RD-Agent(Q)](../../research_cases/papers/2505.15155_li2025_rdagent_quant.md)**
(Li et al., NeurIPS 2025) is the closest published analogue to what
`research_director` is attempting. Five units — Specification, Synthesis,
Implementation (Co-STEER), Validation, Analysis — in a closed loop. Results on
CSI 300: ~2× annualised return versus classical factor libraries using >70%
fewer factors, for **under $10 of LLM spend per 12-hour run**. Three ablation
results transfer directly:

1. **The factor branch drives signal quality; the model branch stabilises risk.**
   Factor-only ablation reached IC 0.0497 / ARR 0.1184; model-only reached
   IC 0.0469 / ARR 0.1009 but the best MDD (−0.0694).
2. **A contextual Thompson-sampling bandit beats both random scheduling and
   letting the LLM choose** — on every metric, and it completed 44 loops versus
   33 in the same wall clock.
3. **De-duplicate by IC.** Reject any new factor whose max IC against the
   existing SOTA library is ≥ 0.99.

Their leakage control is architectural: the LLM sees only schema, never raw data
and never split boundaries.

**[AlphaEvolve](../../research_cases/papers/2506.13131_novikov2025_alphaevolve.md)**
(Novikov et al., DeepMind 2025) needs exactly two inputs — an initial algorithm
in code and a programmatic evaluator — and then mutates the code itself.
`simulate_vwap_cross()` plus `/api/backtest/run` already satisfy both. This is
the only mechanism on the table that can reach **structural** changes to the
entry rule; every tuning session to date has been a parameter sweep, and a
parameter sweep cannot invent "replace the strict cross-bar test with a
prior-day VAL rejection".

**[Agentic Quantitative Trading survey](../../research_cases/papers/2608.31041_hua2026_agentic_quant_survey.md)**
(Hua et al., Aug 2026) supplies the constraint: *"strong model or forecasting
capability does not reliably translate into trading performance under live
market conditions."* This is diagnostic for vtech — OOF AUC has climbed to
0.696/0.641 on the 15s LSTM while the live pin has not moved since ADR-104.

**[PBO/CSCV + Deflated Sharpe](../../research_cases/papers/ssrn2326253_bailey2015_pbo_cscv.md)**
(Bailey, Borwein, López de Prado, Zhu, *J. Computational Finance* 2015) supplies
the statistics. Selection bias scales with the number of configurations tried;
automated search scales that number by orders of magnitude.

Volume profile as an ML feature is, notably, **not** an established academic
literature. It is practitioner territory. That means plausible edge with zero
free external validation — every claim must be established on our own data.

## Decision

### 1. The LLM is a hypothesis generator and code writer, not a market learner

We reject the framing of "training an LLM to understand these relationships."
Knowledge accumulates in a **structured artifact store** — a SOTA feature
library plus a falsification journal — arbitrated by a deterministic evaluator.
The LLM proposes; the evaluator decides.

### 2. New feature module `src/features/prior_volume_profile.py`, prefix `pv_`

Promoted from the prior-session VP logic already present in
`src/external_models/vwap_cross.py`. Conforms to the standard feature-module
contract (`ts` index, prefixed `float32` columns). All distances normalised by
daily ATR so features are scale-free across the 2011→2026 price range.

| Group | Columns |
|---|---|
| Signed distances | `pv_dist_poc`, `pv_dist_vah`, `pv_dist_val`, `pv_dist_phigh`, `pv_dist_plow`, `pv_dist_pclose` — each $(P-L)/\text{ATR}_\text{daily}$ |
| Positional state | `pv_zone` (ordinal: below VAL / lower VA / at POC / upper VA / above VAH) |
| Interaction history | `pv_bars_since_poc_touch`, `pv_poc_test_count_today`, `pv_val_rejection`, `pv_vah_rejection`, `pv_va_acceptance` |
| Multi-day stack | `pv_naked_poc_above_dist`, `pv_naked_poc_below_dist`, `pv_naked_poc_above_age`, `pv_naked_poc_below_age` |
| Open type | `pv_open_vs_prior_va` (above VAH / inside / below VAL), `pv_gap_atr`, `pv_va_overlap_pct` |
| Profile shape | `pv_shape` (D / P / b / double-distribution from histogram skew+kurtosis), `pv_va_width_atr` |

`pv_va_overlap_pct` together with `pv_open_vs_prior_va` encodes balance versus
imbalance — the regime variable that determines whether a VWAP-cross signal
should be treated as mean-reverting or trend-following.

### 3. Extend `of_` with CVD-relative-to-structure features

| Column | Definition |
|---|---|
| `of_cvd_at_level` | CVD accumulated while price was within *k* ticks of a prior-day level — absorption |
| `of_cvd_divergence_slope` | Regression slope of CVD against price over N bars; continuous and signed (replaces the binary `of_cvd_price_divergence` as the primary signal) |
| `of_cvd_session_pctile` | Current CVD versus the distribution of same-time-of-day CVD over the prior 20 sessions |
| `of_cvd_vwap_reversion` | CVD change since the last VWAP cross, signed by cross direction |
| `lt_cvd_at_level` | Large-trade (≥10 lot) version of `of_cvd_at_level` |

`of_cvd_vwap_reversion` and `lt_cvd_at_level` feed the `vwap_cross` engine
directly: they distinguish a cross confirmed by aggressive flow from a cross
into absorption.

### 4. Meta-labeling over raw direction

The `vwap_cross` engine produces the **side**. A GBDT trained on the
`pv_`/`of_` context vector produces $P(\text{this signal is profitable})$.

This is chosen deliberately over further direction-prediction work. The short
AUC ceiling of 0.66–0.72 is the expected symptom of asking a model to produce
direction from scratch, and the Aug-2026 survey predicts that improving it will
not move live performance. Meta-labeling trains on the rule engine's **actual
path-realised PnL**, which sidesteps the triple-barrier label-geometry question
entirely. `topology: primary_meta` is already wired in the objective YAML
([ADR-027](ADR-027-economics-meta-sparse-objective.md)); only the structural
context is missing.

### 5. Regime conditioning learns thresholds, not models

Sessions clustered on (prior VA width / ATR, open type, overnight range, gap)
into ~5 regimes. Per-regime **thresholds** for the `vwap_cross` parameters that
have been hand-tuned to date (chop caps, stop offset, profit target,
`ema_cross_length`). Per-regime *models* are rejected: insufficient data.

### 6. Close the research loop — executor, SOTA library, bandit

| Unit | Action |
|---|---|
| Specification | Prompt contract restricted to schema only. The LLM must never receive raw market data or split boundaries |
| Synthesis | SOTA feature library + hypothesis-embedding clustering for explore/exploit balance |
| Implementation | Code-gen agent writing to the `src/features/` module contract, with bounded retry and a growing (task, code, feedback) knowledge base |
| Validation | **Executor daemon** draining `data/ops/experiment_queue/` through the existing Custom Test walk-forward; IC-dedup at 0.99; leakage health check |
| Analysis | Contextual Thompson-sampling bandit over arms {new feature, feature refinement, model/hparam, rule-engine parameter, rule-engine structure}; SOTA promotion; falsification journal |

This supersedes nothing in [ADR-025](ADR-025-scoping-v2-agentic-research-loop.md) or
[ADR-030](ADR-030-shared-agentic-substrate.md) — it completes them. The Phase 20
plan in [PHASE-20-RESEARCH-DIRECTOR-REBUILD.md](../roadmap/PHASE-20-RESEARCH-DIRECTOR-REBUILD.md)
already anticipates W5/W6; this ADR adds the bandit scheduler, the SOTA library,
and IC-dedup, which that plan does not contain.

### 7. Overfitting control is a hard prerequisite, not a later refinement

Required before the executor, the RD timer, or program evolution is enabled.
Purging and embargo already exist (Finding 4) and are **not** listed here.

1. **Global trial ledger** — every backtest and training run records
   `(config_hash, sharpe, n_obs, timestamp)` durably, across runs. $N$ is read,
   never estimated.
2. **Deflated Sharpe Ratio** gate on every promotion candidate.
3. **PBO via CSCV** ($S=16$) on the candidate's OOS return series across its
   config family. **Reject at PBO > 0.5** regardless of backtest Sharpe.
4. **Frozen holdout enforced at the data-access layer** — 2026-07-01 onward
   unreadable by the executor process, not merely by convention. Evaluated
   manually, at most monthly.
5. **Net expectancy after costs and slippage** as the sole promotion metric.
   Never AUC, never IC.

### 8. Structural evolution of the rule engine, gated on §7

The trigger block of `simulate_vwap_cross()` is wrapped in
`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`; exit and accounting scaffolding
stays frozen. Multi-objective fitness on (net expectancy, PF, −MDD, n_trades
floor) via the existing backtest endpoint, with an island-model population.

## Sequencing

| Phase | Contents | Gate to next |
|---|---|---|
| **A — Foundation** | Honest labels; `pv_` module; CVD-vs-structure features; meta-label `vwap_cross` trades | Does structural context lift net expectancy on real trades? |
| **B — Rigor** | Trial ledger; DSR; PBO/CSCV; data-layer frozen holdout | All gates computable and enforced |
| **C — Closed loop** | Executor daemon; SOTA library + IC-dedup; bandit scheduler; falsification journal; enable RD timer | Loop completes cycles without human intervention |
| **D — Evolution** | AlphaEvolve-style mutation of the `vwap_cross` trigger block | — |

Phase A contains the cheap falsification test and must run first: if prior-session
VP and structural CVD do not lift a meta-label model on the rule engine's real
trades, no autonomous loop will conjure alpha from them. RD-Agent(Q)'s component
ablation independently supports front-loading the feature branch.

Phases C and D are explicitly **blocked** on Phase B. The rate at which they
generate candidates is the rate at which they generate false positives absent
those gates.

## Consequences

**Positive**

- The operator's chart semantics become model-visible for the first time.
- Meta-labeling attacks a materially easier learning problem than the direction
  task that has stalled at 0.66 short AUC.
- The research loop becomes capable of learning rather than only proposing.
- Program evolution can reach rule structures unreachable by parameter search.

**Negative / risks**

- `pv_` and the new `of_` columns require a feature-store rebuild across all
  affected timeframes.
- DSR and PBO will reject candidates that current review would accept. This is
  the intended effect and will feel like regression.
- Volume profile has no supporting academic literature; Phase A may falsify the
  entire premise. That outcome is a successful result of the phase, not a failure
  of it.

**Neutral**

- LLM cost is not a constraint (RD-Agent(Q): <$10 per 12-hour run). The binding
  constraint is backtest compute.

## Alternatives considered

- **Fine-tune an LLM on market data.** Rejected. LLMs are poor at learning
  numeric relationships from time series, and the survey shows agentic systems
  already over-concentrate here.
- **Keep improving direction prediction.** Rejected. Two years of AUC gains have
  not moved the live pin; the survey explains why this is expected.
- **Enable the RD timer now and add statistics later.** Rejected. This
  industrialises selection bias — see §7 and the PBO note.
- **Add prior-session VP directly to `vwap_cross` as more parameters.** Rejected.
  It would remain invisible to every model and add to a parameter surface that is
  already being hand-tuned past the point of statistical defensibility.

## References

- [research_cases/papers/2505.15155_li2025_rdagent_quant.md](../../research_cases/papers/2505.15155_li2025_rdagent_quant.md)
- [research_cases/papers/2506.13131_novikov2025_alphaevolve.md](../../research_cases/papers/2506.13131_novikov2025_alphaevolve.md)
- [research_cases/papers/2608.31041_hua2026_agentic_quant_survey.md](../../research_cases/papers/2608.31041_hua2026_agentic_quant_survey.md)
- [research_cases/papers/ssrn2326253_bailey2015_pbo_cscv.md](../../research_cases/papers/ssrn2326253_bailey2015_pbo_cscv.md)
- [ADR-025](ADR-025-scoping-v2-agentic-research-loop.md) — dual-perspective agentic research loop
- [ADR-027](ADR-027-economics-meta-sparse-objective.md) — economics ranking + primary/meta topology
- [ADR-030](ADR-030-shared-agentic-substrate.md) — shared agentic substrate
- [ADR-118](ADR-118-orb-engine-and-nq-capital-adequacy.md) — prior-day VP levels in the ORB engine
- [FEATURE_ENGINE.md](../architecture/FEATURE_ENGINE.md) — feature-module contract
