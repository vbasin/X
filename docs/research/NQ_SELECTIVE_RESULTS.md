# Independent NQ model research — 2026-09-24

**The requested profitable, low-drawdown model was not established. Nothing was
promoted or deployed.** Six preregistered candidate configurations were evaluated
through nested past-only selection and three chronological outer folds. A simple
cash policy outperformed the selected learned strategy.

| Development OOF, Jan–Jun 2026 | One NQ contract |
|---|---:|
| Trades / observed regular sessions | 182 / 121 |
| Net P&L after execution costs | -$19,631.52 |
| Profit factor | 0.727 |
| Win rate | 42.31% |
| Closed-trade maximum drawdown | $21,528.08 |
| Independently audited observed-mark drawdown | $21,769.72 |
| Losing trades / total losses | 105 / $71,917.80 |
| Worst trade / worst day | -$1,218.36 / -$2,211.72 |
| Longest losing streak | 7 |
| 95% block-bootstrap daily-mean interval | [-$324.87, -$12.23] |

The model used opening-range/VWAP geometry, trailing volatility, trend/return
features, signed volume and quote imbalance. XGBoost and ridge predicted long
and short expected net risk-unit returns; past-only inner windows chose the
model, holding template and minimum forecast. One position maximum, four entries
per day maximum, and a two-loss/2R realized daily stop controlled exposure.
The May–June fold selected cash; it is correctly included as zero-P&L sessions.

NQ economics use $20/point and a 0.25-point tick, consistent with
[CME contract specifications](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html).
Base replay pays recorded spread, 0.25 point slippage per side, 250ms latency,
and the repository's $3.36 round-trip commission assumption. Slippage stresses
0.5/1 point lost $20,343.16/$22,788.16; 1000ms latency lost $18,298.16.
These are modeled fills, not live fills; actual fees, impact and latency remain
unverified. No account size or percentage-return assumption was imposed.

The opening-breakout baseline earned $4,393.68 but drew down $19,791.92; long-only
earned $6,635.08 but drew down $20,836.44. Neither demonstrates the requested
large, consistent, low-loss edge. All four baseline daily-mean confidence intervals include zero; the intervals are in the independent audit.
The July 1–September 22 reserve remains unread by this experiment because the
predeclared development gate failed. Previous repository exposure to those dates
cannot be undone; this was never represented as a globally virgin sample.

## Physical evidence

- [Printable PDF report](../../data/research/runs/nq-selective-intraday/20260924T033352Z/pass_1/model_selective_ev/report/NQ_Research_Report.pdf)

- [Detailed measured report](../../data/research/runs/nq-selective-intraday/20260924T033352Z/pass_1/model_selective_ev/report/summary.md)
- [Independent raw-tape audit](../../data/research/runs/nq-selective-intraday/20260924T033352Z/pass_1/model_selective_ev/report/audit/audit.md)
- [Equity/drawdown chart](../../data/research/runs/nq-selective-intraday/20260924T033352Z/pass_1/model_selective_ev/report/equity.png)
- [Trade ledger](../../data/research/runs/nq-selective-intraday/20260924T033352Z/pass_1/model_selective_ev/trade_ledger.parquet)
- [Canonical manifest](../../data/research/runs/nq-selective-intraday/20260924T033352Z/manifest.json)
- [Protocol](NQ_SELECTIVE_PROTOCOL.md)

The run also retains each fold's fitted model, exact feature list, compact OOF
predictions, every inner selection trial, baseline ledgers, stress ledgers,
source snapshots, source hashes and raw input SHA256 hashes. `RunManifest`
validates as CANONICAL_COMPLETE; this is provenance completeness, not evidence
of trading success. First completed run `20260924T033118Z` is retained; final
run only corrects reporting/cost provenance and verifies result reproduction.
An earlier empty-holiday-frame build failed before any model fit and is retained.

Reproduce from `/opt/vtech`:

```bash
venv/bin/python -m scripts.research.train_nq_selective
venv/bin/python -m pytest tests/architecture/test_nq_selective_features.py tests/architecture/test_nq_selective_execution.py tests/architecture/test_nq_selective_policy.py tests/architecture/test_economic_evaluation.py tests/architecture/test_causal.py -q
```

74 relevant tests passed. Feature tests include prefix invariance, adversarial
future mutations, closed-bar boundaries, duplicate quote ordering and staleness.
Execution/policy tests cover adverse gaps, latency, censoring, side selection,
position overlap, daily stops and zero-trade days. Four real sessions additionally
passed 14 prefix comparisons. This is evidence against the tested leakage modes,
not an assertion that every possible market-data issue has been eliminated.

## Critical legacy finding

`src/training/entry_state.py::_order_flow` normalizes CVD/delta by a maximum
computed across all requested decision rows; OFI uses the same pattern. If those
rows span the session, earlier features depend on later values. CVD `[10,20,100]`
produces earlier normalized values `[0.1,0.2]`, while its two-row prefix produces
`[0.5,1.0]`. New features bypass this module. Existing affected matrices/models
require causal rebuild and revalidation before their reported OOF results can
be treated as clean evidence.

The holiday exclusion uses a pre-known calendar, including July 3, 2025. CME's
[2025 Independence Day notice](https://www.cmegroup.com/tools-information/holiday-calendar/files/2025/us-independence-day-settlement-times-2025.pdf)
confirms modified equity settlement that day; the experiment does not infer
eligibility from later tape coverage.
