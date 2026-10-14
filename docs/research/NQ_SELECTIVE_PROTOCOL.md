# NQ selective intraday experiment — preregistered 2026-09-24

Status: EXPERIMENTAL. This protocol is written before inspecting strategy results.

Hypothesis: opening-auction geometry, trailing volatility, VWAP displacement and
recent signed flow contain conditional information about continuation versus
reversal sufficient to pay executable spread, commission and adverse slippage.
A small expected-return model may select long, short or cash. No cascade is assumed.

Data: existing outright NQ print tapes, April 2025 onward, resolved with the
existing tape/path APIs. Rebuild features from raw observations. No existing
feature matrix, upstream predictions or full-session normalization is consumed.
One-minute bars become available at their close; decision grid is five minutes,
08:45–14:25 America/Chicago. Risk unit is twice the trailing 20-minute mean
one-minute true range, rounded up to a tick, minimum 8 points; abstain above
60 points, spread above 1 point, or quote age above 2 seconds. All eligibility
uses information available at decision time.

Two preregistered trade templates: 15-minute horizon/1.5R target and 30-minute
horizon/2R target; both have 1R stops. Market entry after 250ms, at the next print
quote within five seconds; exits trigger on liquidation-side quotes and execute
after 250ms. Actual adverse gaps are retained. Targets also use market execution.
Base slippage is 0.25 point each side beyond bid/ask, commission $3.36 round trip
from the existing NQ cost profile, one NQ contract ($20/point). Stress reruns use
0.5 and 1 point per side and 1000ms latency. Missing paths are censored and
reported, never silently turned into winners or excluded using future knowledge.

Candidate family: XGBoost depth 2, 250 trees, learning rate .03, min child weight
100, L2 20, subsample .8, column sample .8, fixed seed 20260924; full causal
features versus price/session features alone. Ridge regression (alpha 100,
train-only standardization) on price/session features is a simpler comparator.
Each predicts long and short net return in risk units. No early stopping on OOF.
Two horizons × three models = six candidates. Policy threshold grid .05/.15/.25R
is chosen only on an inner chronological window. Compare with cash and fixed
trend/reversal/opening-range baselines; no benchmark is retrospectively promoted.

Every policy holds at most one position, permits at most four entries per day,
and stops new entries after two losing trades or realized loss of 2R. No position
sizing optimization, averaging down or overnight exposure. Account capital is
unspecified, so dollars/one contract are reported without a percentage-return claim.

Outer validation windows: Jan–Feb, Mar–Apr, May–Jun 2026. For each, train only
before the preceding two calendar months, choose candidate/threshold in those
two months, and freeze both model weights and policy for the outer window. Purge
one entire observed session between training/selection and selection/test; assert
every training label endpoint precedes selection, every selection endpoint precedes
test. Inner ranking: daily mean minus half its standard error, minimum 30 trades,
profit factor >=1.10; otherwise cash. This is a small, recorded search, not a
claim to global optimality. OOF predictions never train a model on their own future.

Reserved final period: 2026-07-01–2026-09-22. Do not load prices/outcomes from this
period until development passes the gate below and the final policy is frozen.
This is held out from THIS experiment; prior human/repository exposure cannot be
undone and is disclosed. No changes are allowed after its one evaluation.

Development acceptance: >=100 completed OOF trades, profit factor >=1.15,
positive net PnL in at least two of three outer folds, positive lower endpoint
of a 95% five-session moving-block bootstrap mean-PnL interval (5000 replicates),
positive net under both slippage stresses and latency stress, and no unresolved
selected positions or material data-quality failures. Final evaluation must meet
the same economic/uncertainty criteria (minimum 30 trades; fold criterion does not
apply). Failure is a rejected research hypothesis, not permission to tune holdout.

Persist: protocol/config and code hashes, input byte hashes, fold boundaries and
selection trials, trained estimators, compact OOF predictions, selected trade
ledger, daily equity (including zero-trade observed sessions), drawdown, loss
count/amount, monthly and fold results, block-bootstrap intervals, stress reports,
feature importance, human-readable report and standalone equity chart.

Known limitation: print-associated BBO is not a complete quote/order-book replay;
latency and queue/impact estimates remain assumptions. Historical simulation
cannot prove future profitability. Existing es_cvd/es_delta/es_ofi normalizers
were found to depend on full-session maxima; these inputs are excluded entirely.

Implementation clarification before results (2026-09-24): exclude pre-known
holiday/modified sessions with `src.common.cme_holidays.is_trading_day`, plus
scheduled 2025-07-03 early close. This calendar-based exclusion does not inspect
future tape coverage. The first build stopped on an empty holiday feature frame
before any model fit or result; that empty-schema defect was corrected.
