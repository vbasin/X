# ADR-026: Trade-PnL Economics — Label-Implied Expectancy on Every Run + Model-Prediction Backtest Mode

## Status
Accepted — adds trade **PnL/economics** as a two-tier capability on top of the existing
evaluation + backtest stack. Extends the rule/cadence backtester of
[ADR-020](ADR-020-rule-strategy-backtest.md) with a fourth experiment entry path
(model-prediction replay) and makes the label-implied trade economics a first-class,
always-computed part of the diagnostics report. Uses the asymmetric barrier payoff defined
by the Triple Barrier objective ([ADR-023](ADR-023-triple-barrier-label-scheme.md)) and
upholds the "objective owns the economics, never a model choice" invariant of
[ADR-018](ADR-018-objective-doc-single-source-labels.md). Artifacts flow through the unified
run layout of [ADR-024](ADR-024-unified-experiment-artifact-flow.md).

## Date
2026-07-15 (Phase 23)

---

## Context

Model runs reported classification quality (precision@τ, F1, confusion, `trades_at_gate`)
but **no money**. The objective's economics are asymmetric and already fully specified:
a qualifying trade wins `labels.move_threshold` (+50 pt profit barrier) and a losing trade
is capped at `labels.mae_tolerance` (the −5/−6 pt stop). Two distinct questions were
unanswered:

1. *"Is this model economically worth trading, and at what confidence?"* — a cheap scalar
   that belongs next to precision on **every** run and leaderboard row.
2. *"If I only trade at τ=0.9, one position at a time, over these dates — what does the
   account curve actually look like?"* — a real, path-dependent backtest.

A latent trap made this urgent: `src/evaluation/_kpi.py` already contained a PnL block, but
it was (a) **disabled** — gated behind `costs.cost_per_trade`, which both objectives leave
unset — and (b) **economically wrong** — it modelled a *symmetric* ±`move_threshold` payoff,
charging a losing trade the full −50 instead of the −stop. Enabling it as-written would have
roughly halved reported expectancy (≈+16 vs ≈+31 pt/trade on `xgb_tbm_full` at τ=0.8).

---

## Options Considered

| Option | Pros | Cons |
|---|---|---|
| **A. Bake path-realised PnL into training/model outputs** | One number per model | Freezes τ / position rules / costs / date window at train time; re-simulates ~1M bars per candidate; couples the model (predicts) to pricing (a separate concern) |
| **B. Label-implied expectancy only** (confusion-matrix proxy) | Free on every run; no price path | Charges every "flat/timeout" miss the full stop; blind to gap/slippage-through-stop; not a real account curve |
| **C. Path-realised backtest only** | True economics | Too heavy to attach to every run; nothing cheap to sort a leaderboard by |
| **D. Two tiers (chosen): B on every run + C on demand** | Cheap screen everywhere; truth when asked; each reuses existing machinery | Two PnL numbers that can differ — must be documented so they are not confused |

---

## Decision

**Tier A — label-implied expectancy, always on** (`src/evaluation/_kpi.py`):

- `payoff_params(objective)` → `(win=move_threshold, loss=mae_tolerance, cost=cost_per_trade|0)`.
- `label_implied_pnl(dir_signals, …)` scores the **asymmetric** first-touch payoff: a correct
  directional call `+win − cost`, a wrong call `−loss − cost`. Computed unconditionally
  (cost defaults to 0), so every scored candidate carries expectancy.
- `compute_diagnostics` now emits `pnl_at_gate` and a `pnl_at_tau_sweep` (the
  expectancy-vs-confidence curve — which τ maximises PnL), read-only and regenerable from the
  persisted OOF. Surfaces automatically on the leaderboard / webui `net_expectancy` sort key.
- **Explicitly conservative**: a "flat" (time-barrier / no-touch) outcome is charged the full
  stop, though its realised excursion lies in `(−loss, +win)`. The proxy therefore *understates*
  expectancy on the dominant miss bucket; the real ledger usually reads higher.

**Tier B — model-prediction backtest** (fourth entry path; `src/backtest/`):

- A new `predictions` spec mode (`strategy_spec.schema.json`, `spec.py`): `{source, tau, sides}`
  where `source` is a model's `oof_*.parquet` (path / run dir / run id). Mutually exclusive with
  `entry`/`schedule`.
- `engine.py` gains a `predictions` entry branch — gate `confidence ≥ τ`, trade the predicted
  side — that **reuses the entire existing execution model**: one position at a time, cooldown,
  per-day cap, stop/target/time/eod exits, ledger (PnL/MAE/MFE), economics KPIs, and IS/OOS
  split. "Max one position held" needs no new knob — the engine already enforces it.
- `runner.py` resolves the OOF source, joins `_close` from the feature parquets, defaults
  exit target/stop/max_holding to the objective barriers, and scores the realised trades on the
  same frozen contract as any candidate. CLI: `--predictions <src> --tau 0.9 [--start --end]`.
- **Concurrency knob**: `backtest.max_concurrent_positions` (default 1 = the classic
  one-at-a-time book) lets up to N positions be open at once — a candidate is skipped only when
  all slots are full at its bar; each position's exit is scanned independently. `cooldown=0`
  never blocks a free slot.
- **Web console tab** (`src/webui`, ADR-024's console): a `#/backtest` view — dropdown for
  **staged/registry serving bundles** (same pool as Live), confidence gate τ, open positions,
  and date range — scores historical feature parquets via **`src.serving`** (shared load +
  `predict_canonical` with Live; no separate OOF UI path) then runs an **in-memory** path
  backtest via `GET /api/backtest/models` + `POST /api/backtest/run`. Renders stat tiles
  (net / gross profit / gross loss / win-rate / profit-factor / max-drawdown / Sharpe /
  exposure), equity curve, confusion matrix, and trade blotter. CLI OOF replay
  (`--predictions`) remains for research archaeology; the console always uses current staged
  weights. Cap: 400 sessions.

**Drawdown definition (both paths).** Max drawdown is peak-to-trough of the cumulative-PnL
curve with the **high-water mark seeded at 0** (pre-trade equity): a *leading* losing trade
counts as drawdown — any dip below the running peak, including the starting 0, contributes.
Unified in `_kpi.max_drawdown` and `engine._compute_kpis` (the prior unseeded `cummax` hid a
leading loss as 0). Verified on real data: a −12.75 pt first trade reports maxDD −12.75.

---

## Consequences

- Every model run now answers "worth trading, and at what τ" for free; the leaderboard is
  sortable by expectancy.
- A trained model can be replayed as a real strategy (`--predictions … --tau 0.9`) with a
  genuine ledger + account curve, comparable to rule strategies and ML candidates alike.
- **Two PnL numbers coexist by design.** `pnl_at_gate` (proxy) and `trade_kpis` (realised) can
  diverge — e.g. a nominal −6 stop that gaps to −12.75 on the next 5s close. This is a feature:
  the proxy is a cheap conservative screen; the backtest is the truth. Consumers must label
  which one they show.
- `_objective_labels` now honours `labels.scheme`, so TBM objectives are scored against
  triple-barrier (not Detection-2) labels — fixes a latent mismatch in the rule-strategy path.

## Cross-links
- Extends [ADR-020](ADR-020-rule-strategy-backtest.md) (backtest engine + entry paths).
- Payoff economics from [ADR-023](ADR-023-triple-barrier-label-scheme.md).
- Artifact layout per [ADR-024](ADR-024-unified-experiment-artifact-flow.md); episodes per
  [ADR-021](ADR-021-outcome-episode-pattern-loop.md).
