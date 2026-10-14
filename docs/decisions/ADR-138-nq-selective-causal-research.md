# ADR-138 — Independent selective NQ expected-return experiment

Date: 2026-09-24
Status: Accepted engineering substrate; trading hypothesis REJECTED

## Context

The user requested an independently designed, tested NQ intraday strategy with
large consistent net profits and limited loss/drawdown. Existing cascade topology
and metrics were not accepted as a benchmark. Raw outright print tapes offered
303 regular development sessions, April 2025–June 2026, plus a reserved July–
September period. A fresh audit also demonstrated a full-session maximum
normalization leak in `src/training/entry_state.py::_order_flow`.

## Decision

Build an isolated, causal experiment using new `nq_` features, with no change to
existing models, feature matrices or live routing. The feature and outcome
implementations are `src/features/nq_selective.py` and
`src/backtest/nq_selective.py`. The runner is
`scripts/research/train_nq_selective.py`; the objective is
`config/objectives/nq-selective-intraday.yaml`.

Predict both directional net risk-unit returns with shallow boosted trees or
regularized linear regression; select long, short or cash. Compare price-only
and flow-augmented state. Use two bounded intraday outcome templates (15m/1.5R
and 30m/2R, 1R stop), closed-bar trailing volatility, executable quotes and
latency. Exit barriers trigger market orders, including targets; no optimistic
limit fill or stop-level clamp. This is new outcome/model-output semantics,
not a replacement for existing ADR-127 labels.

Use nested chronological training/selection/OOF with a full session purge at
both boundaries; freeze fitted weights and policy for each OOF window. Reserve
July–September from this experiment and consult it only after development
acceptance. All search trials and economic results are retained. Raw data byte
hashes and source snapshots identify the exact dirty-tree implementation.
Reuse canonical paths, causal joins, economic accounting and RunManifest.

## Outcome and consequences

The predeclared development test rejected the strategy: 182 trades, net
-$19,631.52, PF 0.727, closed-trade maximum drawdown $21,528.08, 105 losses.
No positive active outer fold; the third fold chose cash from its inner window.
Stress replay also lost money. The final reserve was not consulted. The simple
opening-breakout/long-only benchmarks were marginally positive but had about
$20,000 drawdown and do not establish the requested edge.

No model was promoted or deployed. This preserves falsified research and a
reusable causal harness; it does not claim a profitable or globally optimal
strategy. Historical simulation remains unable to prove future profits.

The legacy leakage is explicitly documented but not silently repaired in place:
a repair changes feature semantics and requires a separate causal rebuild and
revalidation of affected cached matrices. New features exclude every legacy
es_cvd/es_delta/es_ofi input. See `docs/research/NQ_SELECTIVE_PROTOCOL.md` and
`docs/research/NQ_SELECTIVE_RESULTS.md` for protocol, evidence and commands.
