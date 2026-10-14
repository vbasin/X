# NQ cadence-confirmed noise breakout: research protocol

Recorded 2026-09-24 after discovery-only diagnosis of the minute-ratcheted
variant. Its 2021–2024 results have not been consulted. This is a separate
hypothesis, not a relabeling of a failed experiment. Status: EXPERIMENTAL.

## Mechanism and correction

The prior implementation's tight continuous exit generated excessive turnover.
The authors' [primary implementation](https://concretumgroup.com/backtesting-7-years-of-free-data-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/)
checks exposure on half-hour intervals. We test a futures adaptation with
cadence-confirmed exits. This is not an exact replication: cash-open-centered
bands avoid unknown overnight contract-roll adjustments; one NQ, capped entries,
loss stop, and optional emergency stop replace the paper's allocation scheme.

## Frozen discovery family

Same causal prior-14-session, same-clock mean absolute move bands, cash calendar,
expiry-week exclusion, raw data, cost assumptions, chronological partitions, and
selection rule as `NQ_NOISE_AREA_PROTOCOL.md`. Features never use current-session
future values. No cross-session labels. Training data are 2011–2020 only.

24 policies: band multiplier 0.75 / 1 / 1.25; cadence 30 / 60 minutes;
both directions / long only; fixed emergency stop enabled / disabled.
VWAP condition is always enabled. At a cadence close, an existing long exits if
close is no longer above BOTH current upper band and VWAP (short symmetric).
There is no monotonic soft-stop ratchet and no intrabar soft stop. Exit executes
at the next available bar open, with adverse friction. No simultaneous reversal.
The optional continuously active emergency stop is the OPPOSITE band at entry
decision, fixed for the trade and tick-rounded adversely. Gaps execute at the
worse available open. Maximum three entries, stop entering after two net losses,
entry decisions from 09:00 through 14:00 CT, flatten at 15:00 CT.

Discovery runs all policies at 1 and 2 points round-trip price friction plus
$3.36 commission. Select the largest worst-half daily Sharpe among policies with
positive net in both five-year halves at 2-point friction, >=6 positive years,
PF>=1.10, >=300 trades, and no unresolved outcomes in either replay. If none
qualifies, a single exploratory selection is reported but cannot unlock reserve.

The selected policy is written and hashed before validation, then replayed once
on 2021–2024 at 1/2/4-point costs. Validation must have PF>=1.10, positive net in
both two-year halves, positive lower 95% five-session block-bootstrap bound for
mean daily dollars, positive 2-point net, and no unresolved outcomes.

Already-exposed April 2025–June 2026 is execution compatibility, not independent
OOF. The frozen policy is replayed at actual recorded bid/ask, 250ms latency,
0.25/0.5/1 point adverse slippage per fill, and a separate 1s latency stress.
Only fully passed historical and quote gates unlock July–September 2026 reserve.
No changes may follow reserve consultation. No claim of globally optimal policy,
future profitability, actual executable liquidity, or live-trading validation.

Missing data do not justify ex-post removal of a trading day. Unresolved trades
remain explicit and block acceptance; any recovery must use independently
available raw records, not assumed prices or outcome-conditioned exclusions.

## Fixed comparison, recorded before validation results

Report cash (zero P&L) and one-contract long exposure from 09:00 to 15:00 CT
each eligible session, with the same price friction and fees. These are
comparisons, never candidate inputs or reasons to retune the frozen strategy.
For the baseline, missing required entry/exit observations remain unresolved.
Report both raw net and drawdown: positive long-only results alone do not prove
incremental timing skill relative to NQ's directional drift.


## Discovery extension before any validation consultation

The first 24-policy grid produced positive base-cost economics but no policy
passed the 2-point early-half cost requirement. Its best worst-half Sharpe was
at the largest band multiplier, 1.25. Therefore extend the DISCOVERY grid with
1.5, 2, 2.5, and 3; all other parameters and every acceptance gate remain fixed.
This is adaptive training research, not a preregistered 56-policy initial claim.
All 56 policy trials are retained. Wider boundaries test the specific mechanism
that low-amplitude excursions do not pay round-trip costs. The preliminary run
`20260924T041357Z` is retained; although its runner computed the first selected
policy's historical validation, those outcomes were NOT consulted before this
extension. Future model selection cannot use them.

Independently, two genuinely truncated 2020 sessions were confirmed in both
local files and source DB. Vendor cost estimates for their missing 1-second
OHLC intervals were exactly $0.00. Any recovered source data are preserved
separately, checked against overlapping original records, and incorporated in a
new preparation artifact; the original dataset and runs remain intact.
