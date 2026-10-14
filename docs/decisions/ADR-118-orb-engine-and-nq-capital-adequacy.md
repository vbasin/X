# ADR-118: Opening Range Breakout engine and the /NQ capital-adequacy finding

## Status
Accepted — implemented 2026-09-08.

## Date
2026-09-08

## Context

The operator asked for an ORB model for /NQ using the opening range, prior-day
and current-day volume profile levels, and Large Prints, sized for a specific
account: **$5,000 starting balance, up to 3 /NQ contracts, $1,000 margin per
contract, $20 per point, no commissions, and a hard rule that the balance never
falls below $4,000** (a $1,000 cumulative drawdown barrier). The goal stated was
to maximise profit while minimising drawdown.

### What the literature says

Two Zarattini papers dominate the public ORB evidence base:

- **Zarattini, Barbon & Aziz (2024)**, US equities, 7,000+ names, 2016–2023.
  A plain 5-minute ORB returned 3.2 % annualised (Sharpe 0.48) — effectively
  nothing. Restricting to the 20 stocks per day with the highest *opening
  relative volume* took it to 41.6 % annualised (Sharpe 2.81). Their explicit
  conclusion: **the stock selection, not the chart pattern, carried the edge.**
- **Zarattini & Aziz (2025)**, QQQ/TQQQ. Reports 24 % win rate, +0.13R average,
  22–28 % maximum drawdown, carried by rare ~10R winners.

Neither paper tests NQ futures, and neither models a hard drawdown barrier.

### Two findings from our own data that drove the design

We measured 367 sessions of lead-month /NQ 1-minute bars (2025-04 → 2026-09):

| Measure | Value |
|---|---|
| 15-minute opening range, median | **110 points** ($2,200 per /NQ contract) |
| Daily ATR, median | 380 points |
| 1-minute bar range, median | **12.5 points** |
| 1-minute bar range, 90th pct | 29.8 points |

**Finding 1 — the academic payoff profile is incompatible with a $1,000 floor.**
A 24 %-win-rate process routinely strings 8–12 losses together. At any risk
level that makes the strategy worth trading, that sequence ends the account
before the tail winner arrives. Expectancy and survival are different
objectives, and here survival binds.

**Finding 2 — /NQ is too large for this account, and the arithmetic is not
close.** A structurally-valid ORB stop (opposite side of the range, or even the
range midpoint) is 55–110 points, i.e. **$1,100–$2,200 on a single contract** —
one stop exceeds the entire $1,000 loss allowance. Sizing down does not rescue
it: the drawdown budget only affords a ~10-point stop, but the *median
one-minute bar* is 12.5 points, so a 10-point stop sits **inside the noise** and
is hit at random. Our first backtest with a 10-point cap produced exactly that:
20 % win rate, profit factor 0.23.

## Options considered

| Option | Verdict |
|--------|---------|
| A. Replicate the academic ORB on /NQ at the stated size | Rejected — one structural stop exceeds the entire drawdown allowance |
| B. Shrink stops until they fit the $1,000 budget | Rejected — a ~10 pt stop is below the median 1-minute bar range; measured PF 0.23 |
| C. Rule DSL in the Model Design tool | Rejected — position sizing, scale-outs, a daily loss limit and an equity floor are account-level state the chart-model interpreter does not carry |
| D. Dedicated engine owning its own account-aware simulator (chosen) | Expresses the risk governor directly, registers as a normal external model, runs unchanged on the Backtest page |

## Decision

1. **New engine** `src/external_models/orb.py` + card
   `config/external_models/orb.yaml`, id `orb`, engine `orb`,
   `standalone: true`. It does **not** feed the shared book
   ([ADR-062](ADR-062-ext-model-split-and-stack.md)) and is not stackable,
   because sizing and the equity floor are state `simulate_book` does not model.
   `runner.run_backtest` short-circuits to `simulate_orb` for a single `orb` leg.

2. **Three setups**, all chosen so the stop sits close to the entry:
   - `breakout_retest` — continuation entered on the pullback to the broken
     extreme, stop just beyond it.
   - `failed_break` — price exceeds the extreme by ≥ `min_poke_atr` × ATR then
     closes back inside; fade the trapped side.
   - `fade` — inside-range rotation on balanced sessions. This is what keeps the
     model productive on sessions that never trend.

   Entries require a rejection close (`rejection_clv`), an opening-range quality
   band (`min_or_atr` … `max_or_atr` against daily ATR), and abnormal opening
   participation (`min_rvol`).

3. **Risk is contract-sized, never percentage-sized.** `_size()` derives
   contracts from the stop distance so each trade risks ≈ `risk_usd_per_trade`,
   clamped by `max_contracts`, by `margin_per_contract`, and by
   `headroom_frac` × remaining distance to the floor. **A trade whose stop
   cannot be sized to even one contract is skipped, not downsized.** This is the
   single most important rule in the engine: it is what keeps the loss
   distribution inside the barrier.

4. **Exits**: partial at `t1_r` (`scale_out_pct`), optional breakeven on the
   runner, optional trail, `t2_r` for the remainder, session flatten. This
   deliberately **inverts** the academic payoff — trading tail upside for a
   higher hit rate — because a drawdown barrier, not volatility, is the binding
   constraint.

5. **Governor**: `daily_loss_limit`, `max_trades_per_day`, and a permanent halt
   if equity touches `initial_balance − max_drawdown`.

6. **`point_value` is a parameter** (20 = /NQ, 2 = /MNQ) so the
   capital-adequacy question can be answered inside the model rather than in a
   spreadsheet.

## Measured results

Lead-month /NQ, 2025-04-07 → 2026-09-05 (~350 sessions), factory defaults,
$5,000 start, $1,000 floor, no commissions:

| | /NQ ($20/pt) | /MNQ ($2/pt) |
|---|---|---|
| Trades taken | 17 | 43 |
| Signals skipped — unsizable | **446** | 0 |
| Win rate | 64.7 % | 60.5 % |
| Profit factor | 2.08 | — |
| Final balance | $6,055 (+21.1 %) | $6,207 (+24.1 %) |
| Max drawdown | $130 | $138 |
| Floor breached | No | No |
| Average size | 1.00 | 2.07 |

The floor holds in both cases. But the /NQ column only holds because the
governor **refused 446 of 463 signals** it could not size — the account is
riding a 4 % participation rate.

### Honest statistical position

The configuration above was selected from a parameter sweep, and it is **not
statistically significant**:

```
breakout_retest, min_rvol ≥ 1.3
n = 43   mean R = +0.209   sd = 0.989
t-stat = 1.39                        (needs |t| > 2.0)
bootstrap 95 % CI for mean R = [-0.070, +0.488]   → contains zero
P(mean R > 0) = 91.9 %
first half +0.273R · second half +0.143R          (sign-stable)
```

The relative-volume sweep was also **non-monotonic** (rvol ≥ 0.9 positive,
≥ 1.1 negative, ≥ 1.3 positive), which is the signature of noise rather than a
gradient. Both sample halves are positive and the direction agrees with the
published US-equity result, so this is a *plausible* effect — but on 43 trades
it is a hypothesis, not a validated edge, and it is recorded here as such.

## Consequences

- **/NQ at $20/point is the wrong instrument for a $5,000 / $1,000-drawdown
  account.** This is arithmetic, not opinion: one structurally-valid stop
  exceeds the whole allowance. /MNQ at $2/point makes the same strategy
  tradable at ~2 contracts with the same drawdown outcome and 2.5× the
  participation. Switching is a one-parameter change (`point_value`).
- The engine is intentionally conservative: on /NQ it will sit out most
  sessions. That is the governor behaving correctly, not a bug.
- The reported +21 % rests on 17 trades. It must not be treated as a validated
  edge. Before any capital is risked this needs walk-forward validation on a
  longer history, plus slippage sensitivity — the current default assumes
  0.25 points of entry slippage and zero commission, per the brief.
- Large Prints were **not** wired into the engine. Tape data is only available
  from 2025-04-06 and adding a second unvalidated filter to a 43-trade sample
  would compound the overfitting risk. The hook is left for a later revision
  once the base signal is validated.

## Related

- [ADR-062](ADR-062-ext-model-split-and-stack.md) — external-model plane and the
  shared book this engine deliberately bypasses
- [ADR-117](ADR-117-live-chart-model-design.md) — the chart-scoped rule DSL,
  rejected here for want of account-level state
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — volume-profile
  levels
- Zarattini, Barbon & Aziz (2024), *A Profitable Day Trading Strategy for the
  U.S. Equity Market*, SFI Research Paper 24-98
- Zarattini & Aziz (2025), *Can Day Trading Really Be Profitable?*
- Code: `src/external_models/orb.py`, `config/external_models/orb.yaml`
- Tests: `tests/test_orb.py`
- UI: `#/external` → Opening Range Breakout; Backtest tab → EXT Model → `orb`
