# NQ retry: independent mechanism hypotheses

Date: 2026-09-24. Status: proposed hypotheses, not validated edges. This document was
written without inspecting local return outcomes or the locked July–September 2026
holdout. Freeze the implemented specification before evaluating it; deviations
must be recorded as additional research trials.

## Primary evidence and the important interval distinction

Gao, Han, Li and Zhou (2018) study SPY in 1993–2013. Their early signal is
**previous cash close to 10:00**, including overnight movement; the target is
**15:30–16:00**. The additional predictor is **15:00–15:30**. This is not evidence
that a 09:30–10:00 breakout must continue immediately after 10:00. They find
stronger relationships under high volatility/volume and macro announcements.
Those conditioning variables must be available before entry, rather than measured
over the completed day. [Published paper](https://www.sciencedirect.com/science/article/pii/S0304405X18301351).

Baltussen, Da, Lammers and Martens (2021) include NQ among futures studied through
May 2020. Their principal signal is **previous cash close to current 15:30**,
predicting **15:30–16:00**. It generally beats the early signal, including when
their directions differ. Proposed mechanism: short-gamma hedging and leveraged
product rebalancing concentrate near the close. They explicitly warn that costs
can consume profits; their positive net ES example assumes a one-tick cost.
Their regressions use at least 500 observations before OOS forecasting.
These results motivate replication, not a claim about present NQ profitability.
[Author-hosted paper, equations 1–7 and section 3.5](https://academicweb.nd.edu/~zda/intramom.pdf).

## Three falsifiable hypotheses

**H1 — Closing continuation (highest evidence priority).** At 15:30 New York,
trade in the sign of `P(15:30)/P(previous 16:00)-1`, exiting at 16:00. First run
this one-trade-per-day rule without fitted thresholds. Compare against always-long,
always-short, the early signal, and cash-open-to-15:30 signal. A single selective
variant requires the early and full signals to agree. Null: net daily mean is
nonpositive, or conditional direction adds nothing over always-long. Do not infer
dealer gamma sign from option open interest alone; observable options positioning
would be a separate hypothesis requiring timestamp and position-sign evidence.

**Source constraint communicated before testing:** the proposed long sample uses
unadjusted `NQ.c.0` without contract identifiers. Its prior-close return cannot be
certified across rolls. Exclude that predictor unless continuity is independently
established. Cash-open-to-15:30 New York is an intraday-only proxy and an explicit
departure from the published main signal. If source timestamps are Chicago time,
the corresponding decision/exit clocks are 14:30/15:00; verify the timezone rather
than inferring it from the file's visible hour range. A volume/volatility filter,
if added, must compare observed activity only through decision time against prior
sessions at that same clock; it is an additional variant, not the exact paper rule.

**H2 — Failed opening auction excursion (original, weaker evidence).** Freeze
the 09:30–09:45 high/low. Between 09:45 and 10:30, find the first completed
one-minute bar closing beyond a boundary. If one of the next five completed bars
closes back inside the range, enter opposite that excursion after that bar closes.
Exit after 30 minutes, with stop beyond the observed excursion extreme plus one
tick. One attempt per session. Mechanism conjecture: temporary opening imbalance
exhausts; trapped breakout positions unwind. Null: failure predicts no positive
net opposite-direction return. Report gross directional return as well as stopped
trades so exit geometry cannot masquerade as predictive information. No cited
paper here establishes this exact NQ reversal rule.

**H3 — Accepted opening excursion (original comparator).** Use H2's frozen range
and first breakout, but enter in its direction only after two consecutive closed
one-minute bars remain beyond that boundary. Exit after 30 minutes; stop at the
opposite opening-range boundary. One attempt per session. Mechanism conjecture:
opening information is incorporated gradually and sustained acceptance distinguishes
it from H2's failed excursion. Null: acceptance does not improve net continuation
over an unconditional time-matched directional baseline. Holmberg, Lönnbark and
Lundström's ORB study concerns crude oil, with unstable subperiod results; it does
not validate this NQ adaptation. [Published study](https://www.sciencedirect.com/science/article/pii/S1544612312000438).

## Causal implementation and evaluation requirements

- Use New York local cash-session times, including DST. Skip shortened sessions
  under this fixed-clock protocol. For H1, compare the same contract across both
  cash closes or explicitly skip roll discontinuities; missing prior-close data
  must not silently become a current-open predictor.
- A bar is available at its close. Retrieve context through the repository's
  `as_of_index`; fill on the first executable quote after decision latency.
  Use bid/ask sides, commission and extra adverse slippage; stop prices are triggers.
  Exits at 16:00 mean executable futures quotes, not an assumed cash closing print.
- First establish whether simple gross return predictability exists. Fit a small
  regularized return model only if it adds value in nested chronological tests.
  All scaling, volatility cutoffs, risk limits, feature choices and model selection
  use earlier completed sessions. Purge labels crossing training boundaries.
- Reused development outcomes are development evidence even with chronological
  fits. Reserve the untouched holdout for one frozen final policy, and never tune
  on it. Log every variant and rejected candidate, not just the selected winner.
- Report trade ledger, daily marked-to-market equity, net mean, drawdown, losing
  trades, worst day, subperiods, block-bootstrap uncertainty, cost stress and the
  always-long baseline. One contract fixes exposure for comparison; adding leverage
  cannot establish an edge. Small samples cannot substantiate consistent profit.
