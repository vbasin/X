# ADR-020: User-Defined Rule/Cadence Strategy Backtest as a Third Experiment Entry Path

## Status
Accepted. **Extended by [ADR-026](ADR-026-pnl-economics-and-predictions-backtest.md)** — a
fourth entry path (model-prediction replay) reuses this engine, and label-implied PnL
economics become a first-class part of every run.

Adds a third experiment entry path alongside the automated model-selection search
([Phase 23 W3](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md)) and the user-override manual
model spec ([ADR-019](ADR-019-user-override-manual-candidate-spec.md)). Upholds the
"objective never encodes a model choice" invariant of
[ADR-018](ADR-018-objective-doc-single-source-labels.md): a strategy is scored on the same
frozen evaluation contract as any model candidate.

## Date
2026-07-04 (Phase 23)

---

## Context

Phase 23 originally offered exactly one way to start a trade-objective experiment: the
automated model-selection search. ADR-019 added a second (user pins the ML model(s) +
features). A large class of trading ideas, however, is not a *model* at all but an explicit
**rule**: "when RSI breaks a threshold" / "when price is stretched below VWAP and order flow
turns positive" / "buy the first trading day of every month and hold to the next". These have
no training step; they are deterministic functions of feature values and their interactions,
plus a calendar cadence. The system had no way to express, execute, or *score* such a
strategy, and therefore no way to compare a hand-specified strategy against the ML champions
on the same yardstick.

Three requirements shaped the design:

1. **User interface** — the user types the strategy out declaratively (entry/exit conditions
   on feature values & interactions) or as a standard buying/settling cadence
   (daily/weekly/monthly), and selects the trade date range.
2. **A real backtester** — a bar-by-bar execution engine over the 5s feature parquets with
   trade economics (PnL, win rate, expectancy, profit factor, drawdown, MAE/MFE).
3. **Comparability** — the strategy must land on the *same* frozen evaluation contract
   (canonical `{ts, direction, confidence}` frame → objective KPI) so its signal quality is
   directly comparable with model candidates, and it must feed the same outcome-analysis loop
   ([ADR-021](ADR-021-outcome-episode-pattern-loop.md)).

---

## Options Considered

| Option | Pros | Cons |
|---|---|---|
| **A. Free-form Python `eval` of a user expression** | Maximum flexibility | Arbitrary code execution (OWASP A03 injection); un-auditable; unsafe to persist/replay |
| **B. Structured rule DSL (chosen)** | Safe (data, not code); schema-validated; vectorised; replayable; feature-name-checked | Bounded expressiveness (comparisons + boolean nesting + cross ops) |
| **C. Reuse the labels engine as the "strategy"** | No new code | Labels are the *target*, not a strategy; cannot express entries/exits, cadence, or costs |

---

## Decision

Add `src/backtest/` with a **structured, code-free rule DSL** (Option B):

- **Spec** (`config/strategy_spec.schema.json`, loader `src/backtest/spec.py`): a
  `strategy_id`, an optional `objective_id` (defaults source + cross-check), and **exactly one
  of** `entry` (per-side rule blocks) or `schedule` (calendar cadence). Plus optional
  `filters` (global gate), `exit` (target/stop/time/rule/eod), `backtest` (date range,
  holdout split, cooldown, point value, per-day cap) and `costs` (commission + slippage).
- **Rule DSL** (`src/backtest/rules.py`): a rule block is `{all: [...]}` (AND) or `{any: [...]}`
  (OR) over *conditions* — `{feature, op, value|feature_rhs}` — with ops
  `gt/lt/gte/lte/eq/ne/between/abs_gt/abs_lt/cross_above/cross_below`. Everything is
  vectorised pandas boolean algebra. **No `eval`/`exec`; the spec is data.** Cross ops respect
  session boundaries (no phantom crosses at the day open).
- **Engine** (`src/backtest/engine.py`): a sequential trade state machine with a vectorised
  per-trade exit scan. Documented conservatisms — fills at **bar close** ± slippage, exits on
  subsequent bar closes (no intrabar), **stop wins** on a same-bar stop/target tie, one
  position at a time, optional cooldown + per-day cap. Calendar mode supports
  `eod`/`hold_bars`/`hold_days`/`next_entry` settle rules with holiday roll-forward.
- **Runner** (`src/backtest/runner.py`, CLI `python -m src.backtest.runner --spec …`): loads
  features, runs the engine, and emits a trade ledger + trade-economics KPIs **and** a
  full-window canonical frame scored on the objective's `canonical_trade_kpi` /
  `compute_diagnostics` — the exact contract used for model candidates — so a strategy earns a
  leaderboard row and an outcome report just like a model.

### The comparability rule (non-negotiable)
A strategy's signal quality is scored by the **objective's KPI on Detection-2 labels via the
canonical prediction frame** (confidence ≡ 1.0; a rule either fires or deliberately does not).
This makes precision@τ / recall / missed-signal metrics real and directly comparable with the
model families. Trade economics (from the ledger) are reported *alongside* — never *instead of*
— the contract score.

---

## Consequences

**Positive**
- A third, first-class experiment entry path with zero unsafe code paths.
- Strategies and models share one leaderboard, one contract, one outcome-analysis loop.
- Deterministic + replayable: a strategy spec + a data version fully reproduce a backtest.

**Negative / limits**
- The DSL is intentionally bounded (comparisons + boolean nesting + crosses); genuinely novel
  math needs a `NEW_COMPONENT` feature or a model, not a rule.
- 5s-bar-close execution understates intrabar stop/target dynamics — accepted as conservative.

**Neutral**
- `jsonschema` (already a dependency) validates the spec; no new runtime dependencies.

---

## Related
- [ADR-018](ADR-018-objective-doc-single-source-labels.md) — objective owns labels/KPI, never a model/strategy
- [ADR-019](ADR-019-user-override-manual-candidate-spec.md) — the second entry path (manual model spec)
- [ADR-021](ADR-021-outcome-episode-pattern-loop.md) — the outcome-analysis loop this feeds
- [PHASE-23-MODEL-SELECTION-SEARCH.md](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md) — W-strategy workstream
